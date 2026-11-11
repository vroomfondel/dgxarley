#!/usr/bin/env bash
#
# run_standalone.sh — run the dgxarley SGLang image on a single DGX Spark with plain podman
#
# Fetches roles/k8s_dgx/files/sglang_patches from GitHub into a local cache, mounts
# it at /patches, downloads MODEL into HF_CACHE unless complete, runs the patches in
# filename order (as sglang_launch.sh does) and then starts sglang.launch_server in
# the foreground (Ctrl-C stops and removes the container). Needs only podman, curl
# and tar on the host.
#
# Presets (TP=1, one preset_<name> function per model, PRESET=list shows them):
#   qwen38_nvidia    nvidia/Qwen3.8-Flash-Next-NVFP4, modelopt_mixed, EAGLE MTP (default)
#   qwen38_radixark  RadixArk/Qwen3.8-Flash-Next-NVFP4, modelopt_fp4, no MTP
#   nemotron35_lightning  nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4, DSPARK draft
#   qwen38_27b       vroomfondel/Qwen3.8-27B-NVFP4-ModelOpt, DSPARK draft (SPEC=eagle: MTP head)
#   nemotron3_super  nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4, EAGLE MTPv2 draft (SPEC=builtin: MTP head)
#   custom           any MODEL, args only from MODEL_ARGS / the command line
# Both Qwen3.8 checkpoints exceed the unified pool, so their presets offload the 51 GB
# PLE table to a file-backed mmap (PLE_DIR, must be local NVMe). Not cluster-validated.
#
# Usage:
#   scripts/run_standalone.sh [extra sglang args]
#
#   scripts/run_standalone.sh
#   PRESET=qwen38_radixark scripts/run_standalone.sh --context-length 65536
#   SPEC=0 MEM_FRACTION=0.75 scripts/run_standalone.sh
#   PRESET=custom MODEL=Qwen/Qwen3-8B scripts/run_standalone.sh --reasoning-parser qwen3
#   PRESET=list scripts/run_standalone.sh
#
# Environment overrides:
#   PRESET        qwen38_nvidia | qwen38_radixark | nemotron35_lightning | qwen38_27b | nemotron3_super | custom | list
#   MODEL         from the preset (required for custom)
#   IMAGE         from the preset (xomoxcc/dgx-spark-sglang:0.5.23-sm121)
#   MODEL_ARGS    sglang args replacing the preset args entirely
#   SPEC          1 (qwen38_nvidia: 0 = no MTP; nemotron35_lightning: 0 = no draft, eagle = built-in MTP head)
#   GAMMA         7 (nemotron35_lightning, qwen38_27b: DSPARK block size; num_draft_tokens = GAMMA + 1)
#   MTP_STEPS     3 (qwen38_nvidia, qwen38_27b SPEC=eagle; nemotron3_super: 4; num_draft_tokens = MTP_STEPS + 1)
#   ADAPTIVE      0 (qwen38_nvidia: 1 = --speculative-adaptive, tiers 1/8 -> [1,2,3], 32 -> [1,2], start 2)
#   DRAFT_MODEL   RadixArk/Qwen3.8-27B-DSpark (qwen38_27b: DSPARK draft repo or container-local dir)
#   DRAFT_MODELS  from the preset (comma-separated repos pre-downloaded next to MODEL)
#   MEM_FRACTION  0.88 (qwen38 presets; nemotron3_super: 0.90)
#   MAX_RUNNING   4 (qwen38 presets: max running requests = decode cuda-graph max bs; qwen38_27b: 32, decode cuda-graph max bs only)
#   MAMBA_CACHE   128 (nemotron3_super: --max-mamba-cache-size, 5 slots per request with MTP 4/1/5; qwen38_27b: 160)
#   MAMBA_SSM_DTYPE bfloat16 (nemotron3_super: float32 doubles the cache memory, pair it with MAMBA_CACHE=64; qwen38_27b: bfloat16; qwen38_nvidia/radixark: float32)
#   PATCHES_DIR   local sglang_patches dir (skips the GitHub fetch, e.g. an unpushed working tree)
#   REF           main (branch, tag or commit; resolved to a SHA, cached as patches-<sha>,
#                 offline fallback: last SHA seen for REF, else the newest cache)
#   REPO          vroomfondel/dgxarley
#   CACHE_DIR     ${XDG_CACHE_HOME:-$HOME/.cache}/dgxarley-standalone
#   HF_CACHE      $HOME/.cache/huggingface
#   PLE_DIR       /var/tmp/sglang-ple-offload (presets with P_PLE=1)
#   PLE_KEEP      0 (1 = keep the previous PLE table file instead of deleting it before boot)
#   NAME          sglang-standalone
#   PORT          30000
#   HF_TOKEN      passed through when set
#   SKIP_DOWNLOAD 1 = no pre-download, SGLang resolves MODEL itself (online)

set -euo pipefail

# Each preset_<name> sets the defaults for one model: P_MODEL, P_IMAGE, P_ARGS
# (sglang args), P_ENV (extra container env), P_PLE (1 = mount PLE_DIR at /ple) and
# P_DRAFTS (extra repos to pre-download, e.g. an external speculative draft).

qwen38_common() {
    P_IMAGE="xomoxcc/dgx-spark-sglang:0.5.23-sm121"
    P_ENV=(
        SGLANG_ENABLE_JIT_DEEPGEMM=false
        SGLANG_QWEN4_PLE_FILE_PREFETCH=1
        SGLANG_QWEN4_PLE_REUSE=1  # not read by 0.5.21, which reuses the /ple file by name+size
        SGLANG_MAMBA_SSM_DTYPE="${MAMBA_SSM_DTYPE:-float32}"  # non-float32 breaks spec decode (unverified)
    )
    P_PLE=1
    P_DRAFTS=()
    P_ARGS=(
        --trust-remote-code
        --kv-cache-dtype auto
        --attention-backend flashinfer
        --moe-runner-backend flashinfer_cutlass
        --mem-fraction-static "${MEM_FRACTION:-0.87}"
        --context-length 262144
        --page-size 64
        --chunked-prefill-size 4096
        --max-running-requests "${MAX_RUNNING:-4}"
        --cuda-graph-max-bs-decode "${MAX_RUNNING:-4}"
        --mamba-radix-cache-strategy extra_buffer_lazy
        --reasoning-parser qwen3
        --tool-call-parser qwen3_coder
        --ple-offload-embedding
        --ple-offload-backend file
        --ple-offload-dir /ple
    )
}

# roles/k8s_dgx/model_profiles/nvidia-qwen3.8-flash-next-nvfp4.yml; SPEC=0 disables MTP
preset_qwen38_nvidia() {
    qwen38_common
    P_MODEL="nvidia/Qwen3.8-Flash-Next-NVFP4"
    P_ARGS+=(
        --quantization modelopt_mixed
        --fp4-gemm-backend flashinfer_cutlass
        --mm-attention-backend triton_attn
    )
    if [[ "${SPEC:-1}" == "1" ]]; then
        # num_draft_tokens > 4 raises in QwenSparseAttnBackend (indexer_compress_ratio)
        # adaptive starts at 2: the start step must be in every tier's candidate_steps
        local steps="${MTP_STEPS:-$([[ "${ADAPTIVE:-0}" == "1" ]] && echo 2 || echo 3)}"
        P_ARGS+=(
            --speculative-algo EAGLE
            --speculative-num-steps "${steps}"
            --speculative-eagle-topk 1
            --speculative-num-draft-tokens "$((steps + 1))"
            --speculative-moe-runner-backend triton
        )
        if [[ "${ADAPTIVE:-0}" == "1" ]]; then
            # batch sizes route to the closest tier key; steps capped at 3 by indexer_compress_ratio
            P_ADAPTIVE_JSON='{"1": {"candidate_steps": [1, 2, 3]}, "8": {"candidate_steps": [1, 2, 3]}, "32": {"candidate_steps": [1, 2]}}'
            P_ARGS+=(--speculative-adaptive --speculative-adaptive-config /launch/adaptive_config.json)
        fi
    fi
}

# roles/k8s_dgx/model_profiles/radixark-qwen3.8-flash-next-nvfp4.yml
preset_qwen38_radixark() {
    qwen38_common
    P_MODEL="RadixArk/Qwen3.8-Flash-Next-NVFP4"
    P_ARGS+=(--quantization modelopt_fp4)
}

# roles/k8s_dgx/model_profiles/nvidia-nvidia-nemotron-3.5-lightning-30b-a3b-nvfp4.yml; SPEC=0 disables, SPEC=eagle uses the MTP head
preset_nemotron35_lightning() {
    P_IMAGE="xomoxcc/dgx-spark-sglang:0.5.23-sm121"
    P_MODEL="nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4"
    P_ENV=(SGLANG_ENABLE_JIT_DEEPGEMM=false)
    P_PLE=0
    P_DRAFTS=()
    P_ARGS=(
        --kv-cache-dtype fp8_e4m3
        --mem-fraction-static "${MEM_FRACTION:-0.78}"
        --context-length 262144
        --cuda-graph-max-bs-decode "${MAX_RUNNING:-4}"
        --mamba-ssm-dtype float16
        --mamba-radix-cache-strategy extra_buffer
        --reasoning-parser nemotron_3
        --tool-call-parser qwen3_coder
    )
    if [[ "${SPEC:-1}" == "1" ]]; then
        local gamma="${GAMMA:-7}"
        P_DRAFTS=("nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4-DSpark")
        # attention-sink drafts raise unless the draft backend is trtllm_mha; num_draft_tokens must be block_size + 1
        P_ARGS+=(
            --speculative-algorithm DSPARK
            --speculative-draft-model-path "${P_DRAFTS[0]}"
            --speculative-num-steps 1
            --speculative-eagle-topk 1
            --speculative-dspark-block-size "${gamma}"
            --speculative-num-draft-tokens "$((gamma + 1))"
            --speculative-draft-attention-backend trtllm_mha
        )
    elif [[ "${SPEC}" == "eagle" ]]; then
        P_ARGS+=(
            --speculative-algorithm EAGLE
            --speculative-draft-model-path "${P_MODEL}"
            --speculative-num-steps 5
            --speculative-eagle-topk 1
            --speculative-num-draft-tokens 6
        )
    fi
}

# roles/k8s_dgx/model_profiles/vroomfondel-qwen3.8-27b-nvfp4-modelopt.yml at TP=1; SPEC=0 disables, SPEC=eagle uses the MTP head
preset_qwen38_27b() {
    P_IMAGE="xomoxcc/dgx-spark-sglang:0.5.23-sm121"
    P_MODEL="vroomfondel/Qwen3.8-27B-NVFP4-ModelOpt"
    P_ENV=(
        SGLANG_ENABLE_JIT_DEEPGEMM=false
        SGLANG_RAGGED_VERIFY_MODE=static
        PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
    )
    P_PLE=0
    P_DRAFTS=()
    P_ARGS=(
        --trust-remote-code
        --quantization modelopt_fp4
        --kv-cache-dtype fp8_e4m3
        --attention-backend flashinfer
        --fp4-gemm-backend flashinfer_cutlass
        --cuda-graph-backend-prefill disabled
        --mem-fraction-static "${MEM_FRACTION:-0.80}"
        --context-length 262144
        --max-running-requests 48
        --cuda-graph-max-bs-decode "${MAX_RUNNING:-32}"
        # DSPARK gamma 7 needs 5 cache slots per request
        --max-mamba-cache-size "${MAMBA_CACHE:-160}"
        --mamba-ssm-dtype "${MAMBA_SSM_DTYPE:-bfloat16}"
        --mamba-radix-cache-strategy extra_buffer
        --reasoning-parser qwen3
        --tool-call-parser qwen3_coder
    )
    if [[ "${SPEC:-1}" == "1" ]]; then
        local gamma="${GAMMA:-7}"
        P_DRAFTS=("${DRAFT_MODEL:-RadixArk/Qwen3.8-27B-DSpark}")
        P_ARGS+=(
            --speculative-algorithm DSPARK
            --speculative-draft-model-path "${P_DRAFTS[0]}"
            --speculative-draft-model-quantization unquant
            --speculative-draft-attention-backend flashinfer
            --speculative-num-steps 1
            --speculative-eagle-topk 1
            --speculative-dspark-block-size "${gamma}"
            --speculative-num-draft-tokens "$((gamma + 1))"
        )
    elif [[ "${SPEC}" == "eagle" ]]; then
        local steps="${MTP_STEPS:-3}"
        P_ARGS+=(
            --speculative-algorithm EAGLE
            --speculative-draft-model-path "${P_MODEL}"
            --speculative-num-steps "${steps}"
            --speculative-eagle-topk 1
            --speculative-num-draft-tokens "$((steps + 1))"
        )
    fi
}

# roles/k8s_dgx/model_profiles/nvidia-nvidia-nemotron-3-super-120b-a12b-nvfp4.yml at TP=1; SPEC=0 disables, SPEC=builtin uses the in-checkpoint MTP head
preset_nemotron3_super() {
    P_IMAGE="xomoxcc/dgx-spark-sglang:0.5.23-sm121"
    P_MODEL="nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4"
    P_ENV=(SGLANG_ENABLE_JIT_DEEPGEMM=false)
    P_PLE=0
    P_DRAFTS=()
    P_ARGS=(
        --trust-remote-code
        --quantization modelopt_fp4
        --kv-cache-dtype fp8_e4m3
        --attention-backend flashinfer
        --moe-runner-backend flashinfer_cutlass
        --fp4-gemm-backend flashinfer_cutlass
        --cuda-graph-backend-prefill disabled
        --mem-fraction-static "${MEM_FRACTION:-0.90}"
        --context-length 262144
        --max-running-requests "${MAX_RUNNING:-32}"
        --cuda-graph-max-bs-decode "${MAX_RUNNING:-32}"
        # the auto-sized mamba cache caps max_running_requests at 4 with MTP (5 slots per request)
        --max-mamba-cache-size "${MAMBA_CACHE:-128}"
        --mamba-ssm-dtype "${MAMBA_SSM_DTYPE:-bfloat16}"
        --reasoning-parser nemotron_3
        --tool-call-parser qwen3_coder
    )
    if [[ "${SPEC:-1}" == "1" || "${SPEC}" == "builtin" ]]; then
        local steps="${MTP_STEPS:-4}"
        local draft="${P_MODEL}"
        if [[ "${SPEC:-1}" == "1" ]]; then
            P_DRAFTS=("nvidia/Nemotron-3-Super-120B-A12B-BF16-MTPv2")
            draft="${P_DRAFTS[0]}"
        fi
        P_ARGS+=(
            --speculative-algorithm EAGLE
            --speculative-draft-model-path "${draft}"
            --speculative-draft-model-quantization unquant
            --speculative-num-steps "${steps}"
            --speculative-eagle-topk 1
            --speculative-num-draft-tokens "$((steps + 1))"
        )
    fi
}

# Any model: MODEL is required, sglang args come from MODEL_ARGS and the command line.
preset_custom() {
    P_IMAGE="xomoxcc/dgx-spark-sglang:0.5.23-sm121"
    P_MODEL=""
    P_ARGS=()
    P_ENV=()
    P_PLE=0
    P_DRAFTS=()
}

list_presets() {
    declare -F | awk '$3 ~ /^preset_/ {sub(/^preset_/, "", $3); print "  " $3}'
}

PRESET="${PRESET:-qwen38_nvidia}"
if [[ "${PRESET}" == "list" ]]; then
    list_presets
    exit 0
fi
declare -F "preset_${PRESET}" >/dev/null || {
    echo "ERROR: unknown PRESET '${PRESET}', available:" >&2
    list_presets >&2
    exit 2
}
"preset_${PRESET}"

IMAGE="${IMAGE:-${P_IMAGE}}"
MODEL="${MODEL:-${P_MODEL}}"
DRAFT_MODELS="${DRAFT_MODELS-$(IFS=,; echo "${P_DRAFTS[*]}")}"
[[ -n "${MODEL}" ]] || { echo "ERROR: PRESET=${PRESET} needs MODEL" >&2; exit 2; }
REF="${REF:-main}"
REPO="${REPO:-vroomfondel/dgxarley}"
CACHE_DIR="${CACHE_DIR:-${XDG_CACHE_HOME:-$HOME/.cache}/dgxarley-standalone}"
HF_CACHE="${HF_CACHE:-$HOME/.cache/huggingface}"
PLE_DIR="${PLE_DIR:-/var/tmp/sglang-ple-offload}"
NAME="${NAME:-sglang-standalone}"
PORT="${PORT:-30000}"
TP=1

PATCH_PATH="roles/k8s_dgx/files/sglang_patches"

[[ "${1:-}" == "--" ]] && shift
EXTRA_ARGS=("$@")

if podman container exists "${NAME}"; then
    echo "ERROR: container '${NAME}' exists; remove it with 'podman rm -f ${NAME}'" >&2
    exit 2
fi

mkdir -p "${CACHE_DIR}"
if [[ -n "${PATCHES_DIR:-}" ]]; then
    patch_dir="$(cd "${PATCHES_DIR}" && pwd)"
    [[ -n "$(compgen -G "${patch_dir}/p[0-9][0-9]_*.py")" ]] \
        || { echo "ERROR: no p[0-9][0-9]_*.py patches in PATCHES_DIR=${PATCHES_DIR}" >&2; exit 2; }
    sha="local"
    echo "=== local patches (PATCHES_DIR): ${patch_dir}"
else
    ref_file="${CACHE_DIR}/ref-${REF//\//_}"

    sha="$(curl -fsSL --max-time 15 -H "Accept: application/vnd.github.sha" \
        "https://api.github.com/repos/${REPO}/commits/${REF}" 2>/dev/null || true)"
    if [[ "${sha}" =~ ^[0-9a-f]{40}$ ]]; then
        echo "=== ${REPO}@${REF} -> ${sha}"
        echo "${sha}" > "${ref_file}"
    else
        sha=""
        if [[ -s "${ref_file}" && -d "${CACHE_DIR}/patches-$(< "${ref_file}")" ]]; then
            sha="$(< "${ref_file}")"
        else
            newest="$(ls -1td "${CACHE_DIR}"/patches-* 2>/dev/null | head -n1 || true)"
            [[ -n "${newest}" ]] && sha="${newest##*/patches-}"
        fi
        [[ -n "${sha}" ]] || { echo "ERROR: cannot resolve ${REPO}@${REF} and no cached patches in ${CACHE_DIR}" >&2; exit 2; }
        echo "WARNING: cannot resolve ${REPO}@${REF}, falling back to cached patches-${sha}" >&2
    fi

    patch_dir="${CACHE_DIR}/patches-${sha}"
    if [[ -d "${patch_dir}" ]]; then
        echo "    cached: ${patch_dir}"
    else
        tmp_dir="$(mktemp -d "${CACHE_DIR}/.tmp.XXXXXX")"
        trap 'rm -rf "${tmp_dir}"' EXIT
        echo "=== fetching ${PATCH_PATH} @ ${sha}"
        curl -fsSL "https://codeload.github.com/${REPO}/tar.gz/${sha}" \
            | tar -xzf - -C "${tmp_dir}" --strip-components=1 --wildcards "*/${PATCH_PATH}/*.py"
        [[ -n "$(compgen -G "${tmp_dir}/${PATCH_PATH}/p[0-9][0-9]_*.py")" ]] \
            || { echo "ERROR: no patches found under ${PATCH_PATH} at ${sha}" >&2; exit 2; }
        mv "${tmp_dir}/${PATCH_PATH}" "${patch_dir}"
    fi
fi
touch "${patch_dir}"
echo "    $(ls "${patch_dir}"/p[0-9][0-9]_*.py | wc -l) patches -> ${patch_dir}"

mkdir -p "${HF_CACHE}"

launch_dir="${CACHE_DIR}/launch-${NAME}"
mkdir -p "${launch_dir}"
[[ -n "${P_ADAPTIVE_JSON:-}" ]] && printf '%s\n' "${P_ADAPTIVE_JSON}" > "${launch_dir}/adaptive_config.json"

cat > "${launch_dir}/download.py" <<'EOF'
import os, subprocess, sys
models = [os.environ["SGLANG_MODEL"]]
models += [m for m in os.environ.get("SGLANG_DRAFT_MODELS", "").split(",") if m]
code = (
    "import sys; from huggingface_hub import snapshot_download; "
    "print(snapshot_download(repo_id=sys.argv[1], cache_dir='/root/.cache/huggingface/hub', "
    "local_files_only=sys.argv[2] == '1'))"
)
# no-xet retry: hf_xet "Unable to parse string as hex hash value" (FIXED_UPSTREAM_HF_XET_BUG.md)
attempts = [("xet", {}, "0"), ("no-xet", {"HF_HUB_DISABLE_XET": "1"}, "0"), ("local cache", {}, "1")]
for model in models:
    if os.path.isdir(model):
        continue
    for label, extra_env, offline in attempts:
        r = subprocess.run([sys.executable, "-c", code, model, offline],
                           env={**os.environ, **extra_env}, stdout=subprocess.PIPE, text=True)
        if r.returncode == 0:
            print(f"=== model ready ({label}): {r.stdout.strip().splitlines()[-1]}", flush=True)
            break
        print(f"WARNING: {model} download via {label} failed", file=sys.stderr, flush=True)
    else:
        sys.exit(f"ERROR: {model} is neither downloadable nor complete in the local cache")
EOF

cat > "${launch_dir}/launch.sh" <<'EOF'
#!/usr/bin/env bash
set -e
if [ "${SKIP_DOWNLOAD}" != "1" ]; then
    python3 /launch/download.py
    export HF_HUB_OFFLINE=1
fi
for p in /patches/p[0-9][0-9]_*.py; do python3 "$p"; done
exec python3 -m sglang.launch_server "$@"
EOF

if [[ -n "${MODEL_ARGS:-}" ]]; then
    read -r -a model_args <<< "${MODEL_ARGS}"
else
    model_args=("${P_ARGS[@]}")
fi

env_args=(
    -e "SGLANG_MODEL=${MODEL}"
    -e "SGLANG_DRAFT_MODELS=${DRAFT_MODELS}"
    -e "TP=${TP}"
    -e HF_HOME=/root/.cache/huggingface
    -e "SKIP_DOWNLOAD=${SKIP_DOWNLOAD:-0}"
)
for e in "${P_ENV[@]}"; do env_args+=(-e "${e}"); done
[[ -n "${HF_TOKEN:-}" ]] && env_args+=(-e "HF_TOKEN=${HF_TOKEN}")

mount_args=(
    -v "${HF_CACHE}:/root/.cache/huggingface"
    -v "${patch_dir}:/patches:ro"
    -v "${launch_dir}:/launch:ro"
)
if [[ "${P_PLE}" == "1" ]]; then
    mkdir -p "${PLE_DIR}"
    # SGLang rewrites the whole table each boot: ~17 MB/s into a populated file, GB/s into a fresh sparse one
    if [[ "${PLE_KEEP:-0}" != "1" ]]; then
        rm -f "${PLE_DIR}"/ple_table_*.bin
    fi
    mount_args+=(-v "${PLE_DIR}:/ple")
fi

tty_args=(-i)
[[ -t 0 && -t 1 ]] && tty_args+=(-t)

echo "=== starting ${NAME}: preset ${PRESET}, ${IMAGE} / ${MODEL} on :${PORT}"
podman run --rm "${tty_args[@]}" --name "${NAME}" \
    --label "dgxarley.patches.repo=${REPO}" \
    --label "dgxarley.patches.ref=${REF}" \
    --label "dgxarley.patches.sha=${sha}" \
    --label "dgxarley.preset=${PRESET}" \
    --device nvidia.com/gpu=all \
    --ipc=host --network=host \
    --ulimit memlock=-1 --ulimit stack=67108864 \
    "${mount_args[@]}" \
    "${env_args[@]}" \
    --entrypoint /usr/bin/tini \
    "${IMAGE}" -- bash /launch/launch.sh \
    --model-path "${MODEL}" \
    --tp "${TP}" \
    --host 0.0.0.0 --port "${PORT}" \
    "${model_args[@]}" \
    "${EXTRA_ARGS[@]}"
