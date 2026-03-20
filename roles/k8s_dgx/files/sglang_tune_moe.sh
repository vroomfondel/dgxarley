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
#   RSYNC_TARGET        — IP of spark2 for rsync
#   HF_CACHE_HOST_PATH  — host path for hf-cache (e.g. /var/lib/hf-cache)
#   FORCE_RETUNE        — set "true" to overwrite existing configs
#   SGLANG_TUNE_BRANCH  — GitHub branch for tuning scripts (default: main)

echo "=== SGLang MoE Triton Kernel Tuning ==="
echo "Model: ${SGLANG_MODEL}"
echo "TP: ${TP}, EP: ${EP}"

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

# 5. Idempotency check — skip if matching JSON exists for this GPU
existing=$(find "$config_dir" -name "*${gpu_slug}*.json" 2>/dev/null | head -1)
if [ -n "$existing" ] && [ "$FORCE_RETUNE" != "true" ]; then
  echo "Tuned config already exists: ${existing}"
  echo "Set FORCE_RETUNE=true to re-run. Skipping."
  # Still rsync existing configs
  if [ -n "$RSYNC_TARGET" ]; then
    echo "Rsyncing existing configs to ${RSYNC_TARGET} ..."
    ssh_opts="-o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null"
    dst="root@${RSYNC_TARGET}:${HF_CACHE_HOST_PATH}/moe_configs/"
    rsync -ah -e "ssh $ssh_opts" "${MOE_CONFIG_DIR}/" "$dst"
    echo "Rsync complete."
  fi
  exit 0
fi

# 6. Download tuning scripts from SGLang GitHub
branch="${SGLANG_TUNE_BRANCH:-main}"
base_url="https://raw.githubusercontent.com/sgl-project/sglang/${branch}/benchmark/kernels/fused_moe_triton"
workdir=$(mktemp -d)
echo "Downloading tuning scripts from branch '${branch}' ..."
curl -fsSL "${base_url}/tuning_fused_moe_triton.py" -o "${workdir}/tuning_fused_moe_triton.py"
curl -fsSL "${base_url}/common_utils.py" -o "${workdir}/common_utils.py"
# Create __init__.py so common_utils can be imported
touch "${workdir}/__init__.py"

# Patch: upstream get_model_config() calls config.get_text_config() before
# accessing config.architectures, but the text_config sub-object doesn't carry
# the architectures field (it's only on the top-level multimodal config).
# This breaks Qwen3.5 MoE (Qwen3_5MoeForConditionalGeneration).
# Fix: save architectures before unwrapping text_config.
python3 -c "
import pathlib, re
p = pathlib.Path('${workdir}/common_utils.py')
src = p.read_text()
old = '''    if hasattr(config, \"text_config\"):
        config = config.get_text_config()'''
new = '''    if hasattr(config, \"text_config\"):
        _architectures = config.architectures
        config = config.get_text_config()
        if config.architectures is None:
            config.architectures = _architectures'''
if old in src:
    p.write_text(src.replace(old, new, 1))
    print('Patched common_utils.py: preserve architectures across text_config unwrap')
else:
    print('WARNING: patch target not found in common_utils.py — upstream may have fixed this')
"

# Patch: replace Ray tqdm progress bar with print-based progress.
# Ray's tqdm_ray renders in the Ray dashboard, not in kubectl logs — so the
# tuning loop appears silent. Replace with periodic prints to stdout.
python3 -c "
import pathlib
p = pathlib.Path('${workdir}/tuning_fused_moe_triton.py')
src = p.read_text()
old = '        for config in tqdm(search_space):'
new = '''        _total = len(search_space)
        for _cfg_idx, config in enumerate(search_space, 1):
            if _cfg_idx == 1 or _cfg_idx % 50 == 0 or _cfg_idx == _total:
                _pct = _cfg_idx / _total * 100
                _best_str = f'best={best_time:.1f}us' if best_time < float('inf') else 'searching...'
                print(f'  [{_cfg_idx}/{_total}] ({_pct:.0f}%) {_best_str}', flush=True)'''
if old in src:
    p.write_text(src.replace(old, new, 1))
    print('Patched tuning_fused_moe_triton.py: replaced Ray tqdm with print progress')
else:
    print('WARNING: tqdm patch target not found — upstream may have changed the loop')
"

echo "Scripts downloaded to ${workdir}"

# 7. Run the tuning benchmark
echo "Starting MoE kernel tuning (this may take 30-90 minutes) ..."
cd "$workdir"
python3 tuning_fused_moe_triton.py \
  --model "$SGLANG_MODEL" \
  --tp-size "$TP" \
  --ep-size "$EP" \
  --dtype fp8_w8a8 \
  --tune
echo "Tuning complete."

# 8. Move generated JSON configs to persistent config dir
# The tuning script writes JSON files to the current directory
for f in *.json; do
  [ -f "$f" ] || continue
  echo "Moving ${f} → ${config_dir}/"
  mv "$f" "${config_dir}/"
done

echo "Configs in ${config_dir}:"
ls -la "${config_dir}/"

# 9. Rsync config dir to spark2 via SSH
if [ -n "$RSYNC_TARGET" ]; then
  echo "Rsyncing configs to ${RSYNC_TARGET} ..."
  ssh_opts="-o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null"
  dst="root@${RSYNC_TARGET}:${HF_CACHE_HOST_PATH}/moe_configs/"
  rsync -ah -e "ssh $ssh_opts" "${MOE_CONFIG_DIR}/" "$dst"
  echo "Rsync to ${RSYNC_TARGET} complete."
fi

echo "=== MoE Triton Kernel Tuning Job Done ==="
