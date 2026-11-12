#!/bin/bash
set -e

# Tune fused MoE Triton kernel configs for the current GPU.
# SGLang logs "Using default MoE kernel config" when no GPU-specific configs exist.
# This script runs the upstream tuning benchmark (~1280 configs per dtype/shape),
# persists the optimal JSON to a shared hostPath, and rsyncs to spark2.
#
# Env vars (all from Job spec):
#   SGLANG_MODEL        — HF model ID (e.g. Qwen/Qwen3-235B-A22B-Instruct-2507-AWQ)
#   TP                  — tensor parallelism (e.g. 2)
#   EP                  — expert parallelism (e.g. 2)
#   HF_TOKEN            — HuggingFace token
#   MOE_CONFIG_DIR      — container path for config output (e.g. /root/.cache/huggingface/moe_configs)
#   RSYNC_TARGETS       — comma-separated IPs of worker nodes for rsync
#   HF_CACHE_HOST_PATH  — host path for hf-cache (e.g. /var/lib/hf-cache)
#   FORCE_RETUNE        — set "true" to overwrite existing configs
#   SGLANG_TUNE_BRANCH  — GitHub branch for tuning scripts (default: main; a -smNNN image suffix is stripped)
#   TUNE_DTYPE          — tuner --dtype (default: auto)
#   TUNE_BLOCK_SHAPE    — override the derived block_shape, e.g. "128,128" or "none"
#   TUNE_BATCH_SIZES    — space/comma-separated subset of batch sizes (default: upstream list)
#   TUNE_RESUME         — "true" (default) resumes from the per-batch-size partial file

echo "=== SGLang MoE Triton Kernel Tuning ==="
echo "Model: ${SGLANG_MODEL}"
echo "TP: ${TP}, EP: ${EP}"

# 0. Patch CUTLASS BlockScaledMmaOp to support SM121 for FP4 (NVIDIA/cutlass#2800).
for mma_py in \
  /usr/local/lib/python3.12/dist-packages/nvidia_cutlass_dsl/python_packages/cutlass/cute/nvgpu/tcgen05/mma.py \
  /usr/local/lib/python3.12/dist-packages/flashinfer/data/cutlass/python/CuTeDSL/cutlass/cute/nvgpu/tcgen05/mma.py; do
  if [ -f "$mma_py" ] && grep -q 'admissible_archs = \[' "$mma_py" 2>/dev/null; then
    if ! grep -q 'sm_121a' "$mma_py" 2>/dev/null; then
      sed -i 's/Arch\.sm_100a,/Arch.sm_100a, Arch.sm_120a, Arch.sm_121a,/' "$mma_py"
      echo "Patched mma.py: added sm_120a + sm_121a to BlockScaledMmaOp.admissible_archs"
    fi
  fi
done

# NOTE: nvfp4_blockwise_moe.cuh SM121 tile fix not possible via runtime patch.
# Requires SM121-specific sgl-kernel build. Use flashinfer_cutlass MoE for NVFP4.

# 1. Install tools
apt-get update -qq && apt-get install -y -qq rsync openssh-client curl >/dev/null 2>&1

# 2. Detect Triton version slug (e.g. "3.1.0" → "3_1_0")
triton_version=$(python3 -c "import triton; print(triton.__version__)")
triton_slug=$(echo "$triton_version" | tr '.' '_')
echo "Triton version: ${triton_version} (slug: ${triton_slug})"

# 3. Detect GPU device name (e.g. "NVIDIA GB10" → "NVIDIA_GB10")
gpu_name=$(python3 -c "import torch; print(torch.cuda.get_device_name(0))")
gpu_slug=$(echo "$gpu_name" | tr ' ' '_')
echo "GPU: ${gpu_name} (slug: ${gpu_slug})"

# 4. Build output path
config_dir="${MOE_CONFIG_DIR}/configs/triton_${triton_slug}"
mkdir -p "$config_dir"
echo "Config output dir: ${config_dir}"

rsync_configs() {
  [ -n "$RSYNC_TARGETS" ] || return 0
  local ssh_opts="-o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null"
  local target
  IFS=',' read -ra targets <<< "$RSYNC_TARGETS"
  for target in "${targets[@]}"; do
    echo "Rsyncing configs to ${target} ..."
    rsync -ah -e "ssh $ssh_opts" "${MOE_CONFIG_DIR}/" "root@${target}:${HF_CACHE_HOST_PATH}/moe_configs/"
    echo "Rsync to ${target} complete."
  done
}

# 5. Download tuning scripts from SGLang GitHub
branch="${SGLANG_TUNE_BRANCH:-main}"
branch="${branch%-sm[0-9]*}"
base_url="https://raw.githubusercontent.com/sgl-project/sglang/${branch}/benchmark/kernels/fused_moe_triton"
workdir=$(mktemp -d)
echo "Downloading tuning scripts from branch '${branch}' ..."
curl -fsSL "${base_url}/tuning_fused_moe_triton.py" -o "${workdir}/tuning_fused_moe_triton.py"
curl -fsSL "${base_url}/common_utils.py" -o "${workdir}/common_utils.py"
# Create __init__.py so common_utils can be imported
touch "${workdir}/__init__.py"

# REMOVED 2026-07-28: "Patch 1" (fixed upstream by PR #20232, see FIXED_SGLANG_MOE_TUNE_UPSTREAM_BUG.md).
# Patches 2-3 and the progress/resume patches moved from inline heredocs to tune_moe_patch.py.
python3 /scripts/tune_moe_patch.py "$workdir"

echo "Scripts downloaded to ${workdir}"

# 6. Run the tuning benchmark: exit 3 = exact target JSON exists; finished batch sizes go to MOE_TUNE_PARTIAL
tune_dtype="${TUNE_DTYPE:-auto}"
partial_dir="${MOE_CONFIG_DIR}/tune_partial"
partial_id=$(echo "${SGLANG_MODEL}_tp${TP}_ep${EP}_${tune_dtype}_bs${TUNE_BLOCK_SHAPE:-auto}_${gpu_slug}_triton_${triton_slug}" | tr '/,' '_x')
export MOE_TUNE_PARTIAL="${partial_dir}/${partial_id}.jsonl"
mkdir -p "$partial_dir"
if [ "${TUNE_RESUME:-true}" != "true" ] && [ -f "$MOE_TUNE_PARTIAL" ]; then
  mv "$MOE_TUNE_PARTIAL" "${MOE_TUNE_PARTIAL%.jsonl}.discarded-$(date +%Y%m%d%H%M%S).jsonl"
fi
[ -f "$MOE_TUNE_PARTIAL" ] && echo "Resume file: ${MOE_TUNE_PARTIAL} ($(wc -l < "$MOE_TUNE_PARTIAL") records)"
echo "Partial results: ${MOE_TUNE_PARTIAL} (snapshot: ${MOE_TUNE_PARTIAL%.jsonl}.json)"

batch_args=()
if [ -n "${TUNE_BATCH_SIZES:-}" ]; then
  read -ra _bs <<< "${TUNE_BATCH_SIZES//,/ }"
  batch_args=(--batch-sizes "${_bs[@]}")
fi

export MOE_TUNE_EXISTING_DIR="$config_dir"
echo "Starting MoE kernel tuning (dtype=${tune_dtype}, block_shape=${TUNE_BLOCK_SHAPE:-derived}) ..."
cd "$workdir"
python3 tuning_fused_moe_triton.py \
  --model "$SGLANG_MODEL" \
  --tp-size "$TP" \
  --ep-size "$EP" \
  --dtype "$tune_dtype" \
  "${batch_args[@]}" \
  --tune || tune_rc=$?
if [ "${tune_rc:-0}" -eq 3 ]; then
  echo "Set FORCE_RETUNE=true to re-run. Skipping."
  rsync_configs
  exit 0
elif [ "${tune_rc:-0}" -ne 0 ]; then
  echo "ERROR: tuning failed (exit ${tune_rc})"
  exit "$tune_rc"
fi
echo "Tuning complete."

# 7. Move generated JSON configs to persistent config dir
# The tuning script writes JSON files to the current directory
move_ts=$(date +%Y%m%d%H%M%S)
for f in *.json; do
  [ -f "$f" ] || continue
  if [ -f "${config_dir}/${f}" ]; then
    backup="${config_dir}/${f}.bak-${move_ts}"
    cp -p "${config_dir}/${f}" "$backup"
    echo "Backed up existing ${f} → ${backup}"
    if [ -n "${TUNE_BATCH_SIZES:-}" ]; then
      printf '!%.0s' {1..100}; echo
      echo "!!! ${config_dir}/${f} now holds ONLY batch sizes [${TUNE_BATCH_SIZES}] (subset run)."
      echo "!!! The previous full file is ${backup}. To merge (new values win):"
      echo "!!!   python3 -c 'import json,sys; o,n=sys.argv[1:]; d=json.load(open(o)); d.update(json.load(open(n))); json.dump(dict(sorted(d.items(), key=lambda kv: int(kv[0]))), open(n,\"w\"), indent=4)' '${backup}' '${config_dir}/${f}'"
      printf '!%.0s' {1..100}; echo
    fi
  fi
  echo "Moving ${f} → ${config_dir}/"
  mv "$f" "${config_dir}/"
done

done_ts=$(date +%Y%m%d%H%M%S)
[ -f "$MOE_TUNE_PARTIAL" ] && mv "$MOE_TUNE_PARTIAL" "${MOE_TUNE_PARTIAL%.jsonl}.done-${done_ts}.jsonl"
[ -f "${MOE_TUNE_PARTIAL%.jsonl}.json" ] && mv "${MOE_TUNE_PARTIAL%.jsonl}.json" "${MOE_TUNE_PARTIAL%.jsonl}.done-${done_ts}.json"

echo "Configs in ${config_dir}:"
ls -la "${config_dir}/"

# 8. Rsync config dir to all worker nodes via SSH
rsync_configs

echo "=== MoE Triton Kernel Tuning Job Done ==="
