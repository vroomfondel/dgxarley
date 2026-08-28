#!/bin/bash
set -e

# Runtime deps (tini/ping/ip/ethtool/net-tools/curl — used for ARP priming + the
# QSFP peer-wait) are NOT in the stock upstream sglang image and get apt-installed
# at boot. Our custom sm121 image bakes them in, so skip the whole apt bootstrap
# when either signal says the deps are already there:
#   1. DGXARLEY_RUNTIME_LIBS_BAKED=1 — explicit marker set in our image Dockerfile.
#   2. Every needed binary is already on PATH (works even without the marker).
# This keeps the stock image working (installs below) while our image starts
# instantly and never touches the mirror.
_skip_apt=true
if [ "${DGXARLEY_RUNTIME_LIBS_BAKED:-0}" = "1" ]; then
  echo "[launch] DGXARLEY_RUNTIME_LIBS_BAKED=1 — runtime deps baked into image, skipping apt bootstrap"
else
  for _b in tini ip ping ethtool netstat curl; do
    if ! command -v "$_b" >/dev/null 2>&1; then _skip_apt=false; break; fi
  done
  [ "$_skip_apt" = true ] && echo "[launch] runtime deps already on PATH — skipping apt bootstrap"
fi

# apt bootstrap for the stock image only. Canonical's ports.ubuntu.com is US-hosted
# behind a ~40-60% packet-loss Zayo transit path from this site (measured 2026-07-15:
# 60% ICMP loss from the workstation, path exits via Zayo → Boston/US; 1.1.1.1 is 0%
# loss), so a single apt attempt fails ~80% of the time and, under `set -e`,
# crashloops the pod with exit 100. Three mitigations: (1) force IPv4 (cluster is
# IPv4-only; belt to the CoreDNS no-AAAA change); (2) repoint apt at the well-peered
# German FAU Erlangen ports mirror (0% loss, ~29ms, full noble suites verified);
# (3) retry update+install in a bounded loop. Non-fatal on total failure.
if [ "$_skip_apt" != true ]; then
  echo 'Acquire::ForceIPv4 "true";' > /etc/apt/apt.conf.d/99force-ipv4
  sed -i 's|ports.ubuntu.com/ubuntu-ports|ftp.fau.de/ubuntu-ports|g' \
    /etc/apt/sources.list /etc/apt/sources.list.d/*.sources /etc/apt/sources.list.d/*.list 2>/dev/null || true

  apt_ok=false
  for i in $(seq 1 15); do
    apt-get update -qq 2>/dev/null || true
    if apt-get install -y -qq tini iproute2 iputils-ping net-tools curl ethtool >/dev/null 2>&1; then
      apt_ok=true
      break
    fi
    echo "[launch] apt install attempt ${i}/15 failed (lossy mirror), retrying in 5s..."
    sleep 5
  done
  [ "$apt_ok" = true ] || echo "[launch] WARNING: apt install failed after 15 attempts; continuing (ping-dependent steps may fail)"
fi

# accelerate: required by SGLang's ModelOptModelLoader
# (srt/model_loader/loader.py → _load_modelopt_base_model). Triggered by:
#   - GLM-5-NVFP4 (modelopt base model load path)
#   - EAGLE3/speculative decoding with a modelopt-quantized target model
#     (e.g. nvidia/Qwen3-235B-A22B-NVFP4 + lmsys EAGLE3 draft) — the draft
#     worker loads the target's embeddings via _load_modelopt_base_model
#     and hits ImportError if accelerate is missing.
# Upstream scitrera/dgx-spark-sglang image does NOT ship accelerate.
if [[ "$SGLANG_MODEL" == *"GLM-5"* ]] || [ "$SGLANG_SPECULATIVE_ENABLED" = "true" ]; then
  python3 -c "import accelerate" 2>/dev/null || pip install accelerate
fi

# GLM-5 specific: transformers upgrade + mem_get_info patch.
# Only needed for glm_moe_dsa models — skip for MiniMax, Qwen, etc.
if [[ "$SGLANG_MODEL" == *"GLM-5"* ]]; then
  echo "GLM-5 model detected — applying GLM-5 specific patches..."

  # transformers ≥5.3.0: required for glm_moe_dsa model type.
  # Must also pull huggingface_hub >=1.3.0 (transformers 5.3.0 dependency).
  python3 -c "from transformers.models.auto.configuration_auto import CONFIG_MAPPING; assert 'glm_moe_dsa' in CONFIG_MAPPING" 2>/dev/null \
    || pip install transformers==5.3.0 huggingface_hub==1.3.0

else
  echo "SKIPPING GLM-5 specific patches..."
fi

# ============================================================================
# Hunyuan (Hy3 / HYV3) special-token suffix backport — SGLang PR #29920
# ----------------------------------------------------------------------------
# This image predates PR #29920 (merged 2026-07-04, "resolve special-token
# suffix at runtime"). Its hunyuan tool-call AND reasoning detectors HARDCODE
# the bare structural tokens (<think>, <tool_calls>, <tool_call>, <tool_sep>,
# <arg_key>, <arg_value>). The shipping Hy3 checkpoints (vroomfondel/Hy3-NVFP4-
# W4A4, tencent/Hy3, ...) append a shared suffix to EVERY such token — the
# chat template's HYTK, mirrored by tokenizer_config.json "token_suffix", e.g.
# ":opensource" so the model emits <think:opensource>, <tool_calls:opensource>,
# ... (verified: these are single special tokens; the BARE forms are not tokens
# at all). With the bare-token detectors, the reasoning split never fires and NO
# tool calls are parsed → breaks honcho/hindsight function-calling.
#
# Faithful backport of PR #29920's resolve_hunyuan_tokens() into the two detector
# files. Upstream threads the tokenizer through ~8 caller files to feed the vocab
# to resolve_hunyuan_tokens(); this image threads no tokenizer, so we instead feed
# the resolved suffix via SGLANG_HUNYUAN_TOKEN_SUFFIX (read below from the model's
# tokenizer_config.json "token_suffix"). Empty suffix (preview checkpoints, or a
# non-Hy3 model) → bare tokens = unchanged upstream-preview behavior.
# RE-SYNC: when bumping to an image that already contains PR #29920, DELETE this
# block — the "resolve_hunyuan_tokens" grep guard already makes it a no-op then.
if [[ "$SGLANG_MODEL" == *"Hy3"* || "$SGLANG_MODEL" == *"Hunyuan"* \
      || "$SGLANG_TOOL_CALL_PARSER" == "hunyuan" || "$SGLANG_REASONING_PARSER" == "hunyuan" ]]; then
  export SGLANG_HUNYUAN_TOKEN_SUFFIX="$(python3 - <<'HY_SUFFIX_EOF'
import json, os
model = os.environ.get("SGLANG_MODEL", "")
suffix = ""
path = None
try:
    if os.path.isdir(model):
        cand = os.path.join(model, "tokenizer_config.json")
        path = cand if os.path.exists(cand) else None
    if path is None:
        try:
            from transformers.utils import cached_file
            path = cached_file(model, "tokenizer_config.json", local_files_only=True)
        except Exception:
            from huggingface_hub import hf_hub_download
            path = hf_hub_download(model, "tokenizer_config.json", local_files_only=True)
    if path:
        with open(path) as fh:
            suffix = json.load(fh).get("token_suffix", "") or ""
except Exception:
    suffix = ""
print(suffix, end="")
HY_SUFFIX_EOF
)"
  echo "Hunyuan token-suffix backport: SGLANG_HUNYUAN_TOKEN_SUFFIX='${SGLANG_HUNYUAN_TOKEN_SUFFIX}'"

fi



















# Prime ARP table on the QSFP link before NCCL tries to connect.
# Without this, the first TCP SYNs get dropped until ARP resolves,
# causing ~230s delay in "Init torch distributed".
# Full-mesh: every node pings ALL other nodes so NCCL's ring/tree
# topology can communicate immediately over any path.
IFS=',' read -ra peers <<< "$QSFP_PEER_IPS"
pids=()
for peer in "${peers[@]}"; do
  (
    echo "Waiting for QSFP peer ${peer} ..."
    while true; do
      ping -c3 -W1 "$peer" 2>&1 | sed "s/^/[${peer}] /"
      [[ ${PIPESTATUS[0]} -eq 0 ]] && break
      sleep 1
    done
    echo "QSFP peer ${peer} reachable."
  ) &
  pids+=($!)
done
for pid in "${pids[@]}"; do wait "$pid"; done
echo "All ${#peers[@]} QSFP peers reachable."


# NOTE: nvfp4_blockwise_moe.cuh SM121 tile fix was attempted but cannot be solved
# via runtime patching. The CUTLASS FP4 grouped GEMM kernel requires SM121-specific
# sgl-kernel build with proper TMA tile shapes — not achievable by editing the .cuh
# source at startup. See CUTLASS_NVFP4_SM121_PRD.md for full analysis.
# For NVFP4 models on SM121: use flashinfer_cutlass MoE runner (avoids cutlass_moe_fp4).





# Version gate: warn if the container image changed — patches below may need review.
# Dev builds report __version__=0.0.0 (no setuptools-scm), so we check the image
# tag (injected as SGLANG_IMAGE env var by Ansible) instead of the Python version.
# The grep guards still prevent patching if the target code has changed.
# Current production line is the 0.5.11/0.5.12/0.5.13 -sm121 family (the old
# -sm121-dev1 dev tags were retired 2026-06-19). The grep guards on each patch
# still no-op safely if a specific target's source has moved.
SGLANG_EXPECTED_IMAGE_PATTERN="xomoxcc/dgx-spark-sglang:.*-sm121"
if [ -n "$SGLANG_IMAGE" ] && ! echo "$SGLANG_IMAGE" | grep -qE "^${SGLANG_EXPECTED_IMAGE_PATTERN}$"; then
  echo "WARNING: SGLang image does not match expected pattern ${SGLANG_EXPECTED_IMAGE_PATTERN} (got ${SGLANG_IMAGE})."
  echo "         Monkey-patches may no longer apply or may need updating."
fi

# When load_format is sharded_state, wait for the shard marker then use sharded path
model_path="$SGLANG_MODEL"
if [ "$SGLANG_LOAD_FORMAT" = "sharded_state" ]; then
  model_slug=$(echo "$SGLANG_MODEL" | sed 's|/|--|g')
  shard_suffix="sglang-TP${TP}"
  if [ -n "$EP" ] && [ "$EP" != "1" ]; then
    shard_suffix="${shard_suffix}-EP${EP}"
  fi
  if [ -n "$SGLANG_QUANTIZATION" ]; then
    shard_suffix="${shard_suffix}-${SGLANG_QUANTIZATION}"
  fi
  if [ -n "$SGLANG_MOE_RUNNER_BACKEND" ]; then
    shard_suffix="${shard_suffix}-${SGLANG_MOE_RUNNER_BACKEND}"
  fi
  sharded_path="/root/.cache/huggingface/sharded/${model_slug}-${shard_suffix}"
  marker="${sharded_path}/model.safetensors.index.json"
  echo "Waiting for sharded checkpoint at ${marker} ..."
  while [ ! -f "$marker" ]; do
    echo "  $(date '+%H:%M:%S') shard not ready yet, waiting 30s ..."
    sleep 30
  done
  model_path="$sharded_path"
  echo "Using pre-sharded model at ${model_path}"
fi



# [moved 2026-07-16] ALL runtime source-patches that used to live here as inline
# `python3 - <<'PATCH_*_EOF'` heredocs are now one file per patch under
# roles/k8s_dgx/files/sglang_patches/ (ConfigMap-mounted at $SGLANG_PATCH_DIR,
# executed by the patch runner below). Nothing was dropped: each patch kept its
# full comment context in its module docstring, and the bash `if` gates became
# each patch's own when= gate. What stays here is launcher logic only: the apt/pip
# bootstrap, the SGLANG_HUNYUAN_TOKEN_SUFFIX export (it exports an env var the
# patches read), the .pth installs, the flag build and the exec.
# See the runner comment below and sglang_launch_patch_refactor_plan.md.

# [removed 2026-07-12] quantization/utils.py dot-boundary is_layer_skipped patch (PR #23467)
# dropped: verified present in the 0.5.14-sm121 image (_module_path_match already defined in
# layers/quantization/utils.py) — the grep guard was already a no-op. Restore from git if an
# older base image is pinned again.




# [removed 2026-07-12] ModelOptModelLoader sharded_state patch dropped: verified the
# 0.5.14-sm121 image already routes LoadFormat.SHARDED_STATE to ShardedStateLoader at loader
# selection (get_model_loader), before ModelOptModelLoader.load_model — dead code. Restore from
# git if a pre-v0.5.14 base image is pinned again. Ref: SGLANG_TP_EP_MOE_UPSTREAM_BUG.md.



# [removed 2026-07-12] DSV4 indexer seq_lens 2-D patch dropped: verified fp8_paged_mqa_logits_torch
# in the 0.5.14-sm121 image already squeezes seq_lens to 1-D before the shape assert (upstream).
# Restore from git if a pre-v0.5.14 base image is pinned again. See UPSTREAM_DSV4_BUGS.md §6.


# ============================================================================
# Runtime source-patch runner
# ----------------------------------------------------------------------------
# Runs every patch in $SGLANG_PATCH_DIR (ConfigMap-mounted, one .py per patch)
# in filename order. Each patch gates itself on the SGLANG_* env vars and is
# idempotent, so the runner stays dumb: no registry, no conditionals here.
# See sglang_patches/_patchlib.py for the contract.
#
# Placed AFTER the inline patch section so patches can read env vars this script
# exports (SGLANG_HUNYUAN_TOKEN_SUFFIX), and BEFORE the .pth installs + flag
# build. Ordering between patches only matters when two touch the same file;
# those live in one file (see the NN_ prefix groups in the refactor plan).
#
# A failing patch must never crashloop the pod: patches themselves swallow
# anchor drift, and the `||` here catches an interpreter-level blowup (syntax
# error in a patch, missing _patchlib) so we degrade to unpatched SGLang the
# same way the inline heredocs did.
# SGLANG_PATCH_ONLY=1 stops the script right after this runner, before the .pth
# installs and the exec. It exists for the patch-refactor verification harness:
# apply the patches in a plain podman container (no GPU, no k3s), snapshot the
# resulting dist-packages tree and diff it against the pre-refactor script's
# tree. Byte-identical tree == the refactor is behaviour-preserving. Never set
# in the pod spec.
SGLANG_PATCH_DIR="${SGLANG_PATCH_DIR:-/patches}"
if [ -d "$SGLANG_PATCH_DIR" ]; then
  for _p in "$SGLANG_PATCH_DIR"/p[0-9][0-9]_*.py; do
    [ -e "$_p" ] || continue
    python3 "$_p" || echo "[launch] WARNING: patch $(basename "$_p") exited non-zero, continuing"
  done
else
  echo "[launch] WARNING: no patch dir at $SGLANG_PATCH_DIR — SGLang runs UNPATCHED"
fi

if [ "${SGLANG_PATCH_ONLY:-0}" = "1" ]; then
  echo "[launch] SGLANG_PATCH_ONLY=1 — patch phase complete, exiting before server start"
  exit 0
fi

# DeepSeek-V4-Flash FlashMLA sparse-decode hook activation (sm_121a / GB10).
# The image bakes deepseek_v4_kernel, but its sitecustomize.py is SHADOWED:
# Ubuntu ships /usr/lib/python3.12/sitecustomize.py (apport) earlier on sys.path,
# and Python imports only the FIRST sitecustomize it finds — so the hook never
# ran and V4-Flash died with "Unsupported architecture for sparse decode fwd".
# A .pth is immune: site.py runs EVERY import-line in EVERY .pth across ALL site
# dirs (no "first wins"), in main AND every spawned sglang worker.
#
# CRITICAL — install ONLY the flash_mla wrapper (_patch_flash_mla_pkg), NOT the
# kernel's patch_flash_mla()/install(). install() also runs
# _patch_sglang_indexer_fallbacks(), which imports sglang…nsa.tilelang_kernel →
# loads tilelang's libcudart_stub.so. At site-init that stub loads BEFORE
# flashinfer.comm, so flashinfer's find_loaded_library("libcudart") grabs the
# stub (no cudaDeviceReset) → hard AttributeError at import (NOT caught by
# sglang's `except ImportError`). sglang imports tilelang itself LATER (after
# flashinfer), so the indexer fallback is not ours to bootstrap. Guarded: a
# broken kernel just falls through to stock flash_mla. Idempotent (also written
# by dockerfile-dsv4-flashmla.patch on rebuilt images). See UPSTREAM_DSV4_BUGS.md §7.
DSV4_DP="/usr/local/lib/python3.12/dist-packages"
if [ -d "$DSV4_DP/deepseek_v4_kernel" ]; then
  cat > "$DSV4_DP/dsv4_autopatch.py" <<'DSV4_AUTOPATCH_EOF'
import os, sys
if os.environ.get("DSV4_KERNEL_DISABLE", "0") not in ("1", "true", "yes"):
    try:
        from deepseek_v4_kernel._patch import _patch_flash_mla_pkg
        _patch_flash_mla_pkg()
    except Exception as exc:
        print("[dsv4_autopatch] flash_mla patch skipped:", exc, file=sys.stderr)
DSV4_AUTOPATCH_EOF
  echo 'import dsv4_autopatch' > "$DSV4_DP/zz_dsv4_autopatch.pth"
  echo "Installed DSV4 FlashMLA autopatch (flash_mla wrapper only): $DSV4_DP/zz_dsv4_autopatch.pth"
else
  echo "DSV4 FlashMLA kernel not present in image, skipping autopatch"
fi

# DSV4 unified-memory load probe (diagnostic, gated by SGLANG_MEMPROBE=1).
# Copies /scripts/dsv4_memprobe.py into dist-packages and drops a .pth so every
# sglang worker arms it at startup (env DSV4_MEMPROBE=1, read by the module). It
# brackets ModelRunner.load_model / init_memory_pool / cuda-graph capture and the
# per-call cuda-alloc delta of Fp8(MoE|Linear)Method.process_weights_after_loading,
# plus a 0.2s ticker — to find which post-load action doubles GB10 unified memory.
# Output → stderr → Loki (grep "[memprobe"). Inert unless SGLANG_MEMPROBE=1.
if [ "${SGLANG_MEMPROBE:-0}" = "1" ] && [ -f /scripts/dsv4_memprobe.py ]; then
  cp /scripts/dsv4_memprobe.py "$DSV4_DP/dsv4_memprobe.py"
  echo 'import dsv4_memprobe' > "$DSV4_DP/zz_dsv4_memprobe.pth"
  export DSV4_MEMPROBE=1
  # memray for the HOST native+mmap allocation profiler (the probe uses it if
  # importable). Best-effort install; absence just disables host profiling.
  pip install -q memray >/dev/null 2>&1 && echo "memprobe: memray installed" || echo "memprobe: memray install failed (host profiling off)"
  echo "Installed DSV4 memprobe (DSV4_MEMPROBE=1): $DSV4_DP/zz_dsv4_memprobe.pth"
else
  rm -f "$DSV4_DP/zz_dsv4_memprobe.pth" "$DSV4_DP/dsv4_memprobe.py" 2>/dev/null || true
fi

# Manual pipeline-stage layer boundaries. SGLang reads SGLANG_PP_LAYER_PARTITION
# directly from the env (os.getenv in get_pp_indices), NOT as a CLI flag. We pass
# our own PP_LAYER_PARTITION and promote it ONLY when non-empty: an empty value
# would make SGLang parse int("") and crash. Empty = SGLang default even split.
if [ -n "${PP_LAYER_PARTITION:-}" ]; then
  export SGLANG_PP_LAYER_PARTITION="$PP_LAYER_PARTITION"
  echo "PP layer partition (manual): SGLANG_PP_LAYER_PARTITION=$SGLANG_PP_LAYER_PARTITION"
fi

args=(
  tini -s --
  python3 -m sglang.launch_server
  --model-path "$model_path"
  --context-length "$SGLANG_CONTEXT_LENGTH"
  --kv-cache-dtype "$SGLANG_KV_CACHE_DTYPE"
  --mem-fraction-static "$SGLANG_MEM_FRACTION"
  --tp-size "$TP"
  --pp-size "$PP"
  --nnodes "$NNODES"
  --node-rank "$NODE_RANK"
  --nccl-init-addr "${QSFP_IP_SPARK1}:${NCCL_PORT}"
  --port "$SGLANG_PORT"
)
# Model compute dtype (--dtype). Unset/"auto" → SGLang reads torch_dtype from the
# model config (correct for almost everything, incl. NVFP4 targets whose config
# already declares dtype bfloat16 — the FP4 weights are unaffected, --dtype only
# sets the compute/activation dtype). Set explicitly when a model needs it —
# notably EAGLE MTP with a Mistral-native draft: the draft params.json carries NO
# dtype field, so --dtype auto falls back to fp32→fp16 and collides with the bf16
# target's shared embed/head → "out_dtype must be Half or BFloat16" at draft graph
# capture. Forcing bfloat16 pins the draft to the target's dtype (SGLang cookbook
# Mistral-Medium-3.5 §3.3: "--dtype bfloat16 is required").
if [ -n "$SGLANG_MODEL_DTYPE" ] && [ "$SGLANG_MODEL_DTYPE" != "auto" ]; then
  args+=(--dtype "$SGLANG_MODEL_DTYPE")
fi
# Debug: per-layer forward-output tensor dump (localise the first inf/nan). OFF unless
# SGLANG_DEBUG_TENSOR_DUMP_OUTPUT_FOLDER is set (registers a forward hook on every layer
# → dumps outputs to the folder; SGLang model_runner.py register_forward_hook_for_model).
# Layers empty = all; else pass the id list through (space-separated for --debug-tensor-dump-layers).
if [ -n "${SGLANG_DEBUG_TENSOR_DUMP_OUTPUT_FOLDER:-}" ]; then
  args+=(--debug-tensor-dump-output-folder "$SGLANG_DEBUG_TENSOR_DUMP_OUTPUT_FOLDER")
  if [ -n "${SGLANG_DEBUG_TENSOR_DUMP_LAYERS:-}" ]; then
    args+=(--debug-tensor-dump-layers ${SGLANG_DEBUG_TENSOR_DUMP_LAYERS})
  fi
fi
# PP async micro-batching: overlap forward passes across pipeline stages.
if [ -n "$PP_ASYNC_BATCH_DEPTH" ] && [ "$PP_ASYNC_BATCH_DEPTH" != "0" ]; then
  args+=(--pp-async-batch-depth "$PP_ASYNC_BATCH_DEPTH")
fi
# Expert parallelism: partitions the TP group for MoE layers.
# EP=TP → MoE uses all-to-all, attention stays tensor-parallel.
if [ -n "$EP" ] && [ "$EP" != "1" ]; then
  args+=(--expert-parallel-size "$EP")
fi
if [ "$SGLANG_ENABLE_EPLB" = "true" ]; then
  args+=(--enable-eplb)
fi
if [ "$SGLANG_ENABLE_EXPERT_DISTRIBUTION_METRICS" = "true" ]; then
  args+=(--enable-expert-distribution-metrics)
fi
# Prometheus exporter: serves /metrics on the HTTP server port (only effective on
# the head; workers run no HTTP server). Gated per-instance via SGLANG_ENABLE_METRICS.
if [ "$SGLANG_ENABLE_METRICS" = "true" ]; then
  args+=(--enable-metrics)
fi
if [ -n "$SGLANG_HOST" ]; then
  args+=(--host "$SGLANG_HOST")
fi
if [ -n "$SGLANG_LOAD_FORMAT" ] && [ "$SGLANG_LOAD_FORMAT" != "auto" ]; then
  args+=(--load-format "$SGLANG_LOAD_FORMAT")
fi
# Diagnostic/tuning knob for the weight-load read-buffer concurrency. The default
# loader uses buffered_multi_thread_safetensors_weights_iterator with 8 workers
# (DEFAULT_NUM_THREADS) — up to 8 shards buffered at once on top of the resident
# weights. Pass e.g. {"enable_multithread_load": false} (no buffering pool) or
# {"num_threads": 1} to shrink the load-time source-buffer peak. NOTE: prefetch
# (weight_loader_prefetch_checkpoints) is OFF by default, so its num_threads is
# inert — THIS is the active buffering control.
if [ -n "$SGLANG_MODEL_LOADER_EXTRA_CONFIG" ]; then
  args+=(--model-loader-extra-config "$SGLANG_MODEL_LOADER_EXTRA_CONFIG")
fi
if [ -n "$SGLANG_QUANTIZATION" ]; then
  args+=(--quantization "$SGLANG_QUANTIZATION")
fi
if [ "$SGLANG_TRUST_REMOTE_CODE" = "true" ]; then
  args+=(--trust-remote-code)
fi
if [ -n "$SGLANG_JSON_MODEL_OVERRIDE_ARGS" ]; then
  args+=(--json-model-override-args "$SGLANG_JSON_MODEL_OVERRIDE_ARGS")
fi
if [ -n "$SGLANG_MOE_RUNNER_BACKEND" ]; then
  args+=(--moe-runner-backend "$SGLANG_MOE_RUNNER_BACKEND")
fi
if [ -n "$SGLANG_REASONING_PARSER" ]; then
  args+=(--reasoning-parser "$SGLANG_REASONING_PARSER")
fi
if [ -n "$SGLANG_TOOL_CALL_PARSER" ]; then
  args+=(--tool-call-parser "$SGLANG_TOOL_CALL_PARSER")
fi
# --chat-template: a built-in template NAME or a path to a .jinja file present in
# the container (from the profile's `chat_template` knob via SGLANG_CHAT_TEMPLATE).
# Empty -> no flag -> SGLang falls back to the model tokenizer's chat_template.
# Distinct from the --chat-template-kwargs handling further below.
if [ -n "$SGLANG_CHAT_TEMPLATE" ]; then
  args+=(--chat-template "$SGLANG_CHAT_TEMPLATE")
fi
if [ "$SGLANG_SPECULATIVE_ENABLED" = "true" ]; then
  args+=(--speculative-algo "$SGLANG_SPECULATIVE_ALGO")
  args+=(--speculative-num-steps "$SGLANG_SPECULATIVE_NUM_STEPS")
  args+=(--speculative-eagle-topk "$SGLANG_SPECULATIVE_EAGLE_TOPK")
  args+=(--speculative-num-draft-tokens "$SGLANG_SPECULATIVE_NUM_DRAFT_TOKENS")
  # External draft model (EAGLE/EAGLE3): use speculative_draft_model_path from profile.
  if [ -n "$SGLANG_SPECULATIVE_DRAFT_MODEL_PATH" ]; then
    args+=(--speculative-draft-model-path "$SGLANG_SPECULATIVE_DRAFT_MODEL_PATH")
  fi
  # Draft model quantization override. By default SGLang inherits the target's
  # quantization for the draft, which breaks when the target is modelopt-
  # quantized (NVFP4) but the draft ships as plain BF16 (typical for external
  # EAGLE3 drafts). Setting "unquant" forces the draft to load without
  # quantization, bypassing the modelopt loader and its Qwen3MoE state-dict
  # shape mismatch against the single-layer EAGLE3 checkpoint.
  if [ -n "$SGLANG_SPECULATIVE_DRAFT_MODEL_QUANTIZATION" ]; then
    args+=(--speculative-draft-model-quantization "$SGLANG_SPECULATIVE_DRAFT_MODEL_QUANTIZATION")
  fi
  # DSV4/SM121: the nextn draft MoE hardcodes an sm100 trtllm kernel that crashes
  # on GB10. Force marlin (SM80+) via this arg — requires the modelopt_quant
  # marlin-branch in sglang-dsv4-nvfp4-pr25820.patch (image rebuild).
  if [ -n "$SGLANG_SPECULATIVE_MOE_RUNNER_BACKEND" ]; then
    args+=(--speculative-moe-runner-backend "$SGLANG_SPECULATIVE_MOE_RUNNER_BACKEND")
  fi
  # DSPARK-only knobs (DeepSeek-V4-Flash-0731 and later). Both empty by default;
  # SGLang ignores neither silently nor gracefully if passed for another algo, so
  # they stay strictly opt-in via the profile.
  #  - block-size = the draft block gamma. Unset -> inferred from the target
  #    checkpoint's dspark_block_size. When set, the profile MUST also set
  #    speculative_num_draft_tokens to gamma+1 (asserted in sglang_instance.yml).
  #  - sps-table-path = pre-profiled cost table for the ragged-verify budget,
  #    built offline with sglang.benchmark.dspark_sps_profiler. Must be a path
  #    that exists INSIDE the container (nothing mounts it for you).
  if [ -n "$SGLANG_SPECULATIVE_DSPARK_BLOCK_SIZE" ]; then
    args+=(--speculative-dspark-block-size "$SGLANG_SPECULATIVE_DSPARK_BLOCK_SIZE")
  fi
  if [ -n "$SGLANG_SPECULATIVE_DSPARK_SPS_TABLE_PATH" ]; then
    if [ -f "$SGLANG_SPECULATIVE_DSPARK_SPS_TABLE_PATH" ]; then
      args+=(--speculative-dspark-sps-table-path "$SGLANG_SPECULATIVE_DSPARK_SPS_TABLE_PATH")
    else
      # Do NOT pass a missing path: SGLang would fail the launch over an
      # optional throughput knob. Warn and fall back to the flat default table.
      echo "[launch] WARNING: dspark sps table '$SGLANG_SPECULATIVE_DSPARK_SPS_TABLE_PATH' not found in the container -> ignoring (ragged-verify falls back to verify-all)"
    fi
  fi
  # WORKAROUND (SGLang 0.5.9): sharded_state + speculative decoding crash.
  # The draft model's ModelRunner inherits load_format=sharded_state from
  # server_args. ShardedStateLoader then fails with KeyError because the
  # per-rank shard files don't contain the draft/MTP model weight keys.
  # Fix: force auto load format for the draft model and point it to the
  # original HF model ID (resolved from HF cache) instead of the shard dir.
  # See SGLANG_SHARDED_SPECULATIVE_UPSTREAM_BUG.md for details.
  if [ "$SGLANG_LOAD_FORMAT" = "sharded_state" ]; then
    args+=(--speculative-draft-load-format auto)
    # Only override draft model path if not already set by profile
    if [ -z "$SGLANG_SPECULATIVE_DRAFT_MODEL_PATH" ]; then
      args+=(--speculative-draft-model-path "$SGLANG_MODEL")
    fi
  fi
  # Adaptive Spec V2 (SGLang ≥0.5.12, PR #23336). Dynamically retunes
  # num_steps / num_draft_tokens at runtime. Only meaningful with
  # EAGLE/EAGLE3 + speculative_eagle_topk=1 — SGLang silently disables
  # otherwise (adaptive_unsupported_reason() in
  # srt/speculative/adaptive_spec_params.py). NEXTN is NOT supported.
  if [ "$SGLANG_SPECULATIVE_ADAPTIVE" = "true" ]; then
    args+=(--speculative-adaptive)
    if [ -n "$SGLANG_SPECULATIVE_ADAPTIVE_CONFIG_JSON" ] \
        && [ "$SGLANG_SPECULATIVE_ADAPTIVE_CONFIG_JSON" != "{}" ] \
        && [ "$SGLANG_SPECULATIVE_ADAPTIVE_CONFIG_JSON" != "null" ]; then
      printf '%s' "$SGLANG_SPECULATIVE_ADAPTIVE_CONFIG_JSON" \
        > /tmp/speculative_adaptive_config.json
      args+=(--speculative-adaptive-config /tmp/speculative_adaptive_config.json)
    fi
  fi
  # GDN ReplaySSM ring spec-verify (SGLang >=0.5.16, PR #28695). Replaces the
  # per-draft SSM snapshot with a ring buffer; upstream reports 11.5 GB -> 1.8 GB
  # speculative scratch per GPU (6.4x) on Qwen3.5-35B-A3B at TP1, at accuracy and
  # throughput parity. Only valid for GDN/linear-attention models with a LINEAR
  # draft chain (speculative_eagle_topk in {None, 1}); SGLang rejects other
  # configurations. Default off upstream and here.
  #
  # ⚠ FLAG NAME IS VERSION-DEPENDENT, and unlike the --mamba-scheduler-strategy
  # case below the two spellings do NOT overlap in every pinned image (checked
  # against the tags, 2026-08-17):
  #   v0.5.15  : neither, the feature does not exist
  #   v0.5.16  : ONLY --enable-gdn-replayssm-spec (the field is literally named
  #              enable_gdn_replayssm_spec there, which is where our old knob
  #              name came from)
  #   v0.5.17  : renamed to --enable-linear-replayssm-spec; the old spelling
  #              survives as a DeprecatedStoreTrueAction alias that still sets
  #              the dest but prints "is deprecated and will be removed".
  # We emit the NEW name, which is correct for every image any profile currently
  # pins (no profile pins 0.5.16, and the knob is off by default everywhere).
  # If you ever pin a 0.5.16 image AND set enable_linear_replayssm_spec: true,
  # this line has to go back to --enable-gdn-replayssm-spec or SGLang aborts on
  # an unrecognized argument.
  # The env var was SGLANG_ENABLE_GDN_REPLAYSSM_SPEC before 2026-08-17; renamed
  # with the profile key and the role var to track the upstream rename.
  if [ "$SGLANG_ENABLE_LINEAR_REPLAYSSM_SPEC" = "true" ]; then
    args+=(--enable-linear-replayssm-spec)
  fi
fi
# --mamba-scheduler-strategy ist deprecated (Warnung seit mindestens 0.5.15,
# nachweisbar in Loki: {namespace="sglang"} |= "is deprecated and will be
# removed"). Beide Schreibweisen existieren in 0.5.15.post1 UND 0.5.16 (am
# 2026-07-28 in beiden Images gegen `--help` geprüft), der Wechsel ist für die
# gepinnten Images also gefahrlos.
if [ -n "$SGLANG_MAMBA_SCHEDULER_STRATEGY" ]; then
  args+=(--mamba-radix-cache-strategy "$SGLANG_MAMBA_SCHEDULER_STRATEGY")
fi
# Mamba state-cache pool sizing (hybrid SSM models). Empty = SGLang auto-fit.
# max_mamba_cache_size // mamba_ratio is the parallelism ceiling on hybrid models.
if [ -n "$SGLANG_MAMBA_FULL_MEMORY_RATIO" ]; then
  args+=(--mamba-full-memory-ratio "$SGLANG_MAMBA_FULL_MEMORY_RATIO")
fi
if [ -n "$SGLANG_MAX_MAMBA_CACHE_SIZE" ] && [ "$SGLANG_MAX_MAMBA_CACHE_SIZE" != "0" ]; then
  args+=(--max-mamba-cache-size "$SGLANG_MAX_MAMBA_CACHE_SIZE")
fi
if [ -n "$SGLANG_MAX_RUNNING_REQUESTS" ] && [ "$SGLANG_MAX_RUNNING_REQUESTS" != "0" ]; then
  args+=(--max-running-requests "$SGLANG_MAX_RUNNING_REQUESTS")
fi
# Absolute KV-cache pool cap (tokens). Unset/0 -> sized by mem_fraction_static.
# Used to pin a co-located instance's memory footprint deterministically.
if [ -n "$SGLANG_MAX_TOTAL_TOKENS" ] && [ "$SGLANG_MAX_TOTAL_TOKENS" != "0" ]; then
  args+=(--max-total-tokens "$SGLANG_MAX_TOTAL_TOKENS")
fi
if [ -n "$SGLANG_SCHEDULE_POLICY" ]; then
  args+=(--schedule-policy "$SGLANG_SCHEDULE_POLICY")
fi
if [ -n "$SGLANG_CHUNKED_PREFILL_SIZE" ] && [ "$SGLANG_CHUNKED_PREFILL_SIZE" != "0" ]; then
  args+=(--chunked-prefill-size "$SGLANG_CHUNKED_PREFILL_SIZE")
fi
if [ -n "$SGLANG_DIST_TIMEOUT" ]; then
  args+=(--dist-timeout "$SGLANG_DIST_TIMEOUT")
fi
if [ -n "$SGLANG_WATCHDOG_TIMEOUT" ]; then
  args+=(--watchdog-timeout "$SGLANG_WATCHDOG_TIMEOUT")
fi
if [ -n "$SGLANG_ATTENTION_BACKEND" ]; then
  args+=(--attention-backend "$SGLANG_ATTENTION_BACKEND")
fi
# Phasengetrennte Attention-Backends. Leer -> kein Flag -> beide Phasen nehmen
# --attention-backend oben. Gesetzt überschreiben sie es pro Phase.
#
# Anwendungsfall NVFP4-KV auf GB10 (kv_cache_dtype: nvfp4): prefill=flashinfer,
# decode=trtllm_mha. Einheitliches Backend scheitert dort dreifach — flashinfer an
# der KV4-MHA-Allowlist, trtllm_mha am SM100-Prefill-Gate, triton am fehlenden
# FP4-Typ in Triton. Volle Herleitung + Messwerte: nvfp4_kv.md.
#
# ⚠ Die beiden Flags sind KEINE Ergänzung zu --attention-backend, sondern haben
# Vorrang. Nur eines von beiden zu setzen ist zulässig und lässt die andere Phase
# auf --attention-backend; genau daraus entsteht aber die "prefill != decode"-
# Konstellation, die SGLangs strenge KV4-Allowlist umgeht — beabsichtigt, aber
# beim Setzen bewusst prüfen.
if [ -n "${SGLANG_PREFILL_ATTENTION_BACKEND:-}" ]; then
  args+=(--prefill-attention-backend "$SGLANG_PREFILL_ATTENTION_BACKEND")
fi
if [ -n "${SGLANG_DECODE_ATTENTION_BACKEND:-}" ]; then
  args+=(--decode-attention-backend "$SGLANG_DECODE_ATTENTION_BACKEND")
fi
# DSA paged-MQA-logits backend (DeepSeek Sparse Attention indexer decode kernel).
# Empty → no flag → SGLang default ("auto" → DeepGEMM). Set "torch" on GB10/SM121 to use
# the _sgl_ torch fallback (DeepGEMM asserts on sm_121). Other choices: deepgemm/cutedsl/aiter.
if [ -n "$SGLANG_DSA_PAGED_MQA_LOGITS_BACKEND" ]; then
  args+=(--dsa-paged-mqa-logits-backend "$SGLANG_DSA_PAGED_MQA_LOGITS_BACKEND")
fi
# DSA indexer top-k backend. Replaces the retired SGLANG_TOPK_TRANSFORM_512_TORCH
# env (deleted upstream in v0.5.18, PR #34926; on v0.5.17 the env is ORed with
# is_torch() at dsv4/indexer.py:808, so the arg covers that site there too).
# "torch" additionally requires unfused top-k (SGLANG_DSA_FUSE_TOPK=0), otherwise
# the fused dispatch raises "Unsupported ... for SGLANG_DSA_FUSE_TOPK". NOTE:
# broader than the old env, which flipped only the 512-transform site; this
# switches every indexer top-k path to torch. Empty → no flag → SGLang default
# ("sgl-kernel"). ${:-} guard: key may be absent from a not-yet-rerendered ConfigMap.
if [ -n "${SGLANG_DSA_TOPK_BACKEND:-}" ]; then
  args+=(--dsa-topk-backend "$SGLANG_DSA_TOPK_BACKEND")
  if [ "$SGLANG_DSA_TOPK_BACKEND" = "torch" ]; then
    export SGLANG_DSA_FUSE_TOPK=0
  fi
fi
# DSA decode attention backend (the MLA attention step over the indexer's top-k KV
# selection). Empty → no flag → SGLang default ("auto" → trtllm-gen FMHA, dead on
# SM121). Set "flashinfer_gather" on GB10/SM121 to use the _sgl_ gather+dense-fa2
# fallback patched above. Other choices: flashmla_sparse/flashmla_kv/flashmla_auto/
# fa3/tilelang/aiter/trtllm (all dead ends on SM121, see DSA_speedup.md).
if [ -n "$SGLANG_DSA_DECODE_BACKEND" ]; then
  args+=(--dsa-decode-backend "$SGLANG_DSA_DECODE_BACKEND")
fi
# DSA PREFILL attention backend (the MLA attention step for extend/prefill tokens).
# Empty → no flag → SGLang default ("auto" → trtllm-gen FMHA, dead on SM121, same
# crash class as the decode backend). Set "flashinfer_gather" on GB10/SM121 to reuse
# the SAME gather+dense-fa2 fallback as decode (patched above; no separate kernel).
# Other choices: flashmla_sparse/flashmla_kv/flashmla_auto/fa3/tilelang/aiter/trtllm
# (all dead ends on SM121, see DSA_speedup.md).
if [ -n "$SGLANG_DSA_PREFILL_BACKEND" ]; then
  args+=(--dsa-prefill-backend "$SGLANG_DSA_PREFILL_BACKEND")
fi
# Multimodal (vision/audio) attention backend. Empty → no flag → SGLang default.
# Choices (0.5.12): sdpa, fa3, fa4, triton_attn, ascend_attn, aiter_attn,
# flashinfer_cudnn. Only relevant for multimodal models (e.g. MiMoV2 omni).
if [ -n "$SGLANG_MM_ATTENTION_BACKEND" ]; then
  args+=(--mm-attention-backend "$SGLANG_MM_ATTENTION_BACKEND")
fi
# KV-cache page size (tokens per page). Empty/0 → no flag → SGLang default.
# Some attention backends / hybrid-SWA paths want a larger page (MiMoV2 card: 64).
if [ -n "$SGLANG_PAGE_SIZE" ] && [ "$SGLANG_PAGE_SIZE" != "0" ]; then
  args+=(--page-size "$SGLANG_PAGE_SIZE")
fi
# Hybrid-SWA KV budget: ratio of SWA-layer KV tokens to full-layer KV tokens
# (SGLang default 0.8). Empty → no flag → SGLang default. Only meaningful on
# hybrid sliding-window models (MiMoV2, Gemma, Llama4). For MiMoV2 + hierarchical
# cache SGLang internally resets this to 1.0.
if [ -n "$SGLANG_SWA_FULL_TOKENS_RATIO" ]; then
  args+=(--swa-full-tokens-ratio "$SGLANG_SWA_FULL_TOKENS_RATIO")
fi
# Diffusion-LLM (dLLM) decode path — when SGLANG_DLLM_ALGORITHM is set (e.g.
# "Gemma4Renoise" for DiffusionGemma) launch_server runs the block-diffusion
# scheduler instead of the autoregressive one. SGLang's _handle_dllm_inference
# auto-forces triton attention, eager mode (cuda graph disabled), and unchunked
# prefill for Gemma4Renoise, so the autoregressive cuda-graph / attention flags
# above are overridden internally. Empty for all autoregressive models → no flag
# is added, zero impact. Requires the 0.5.13-gemmadiffusion image (PR #28054
# baked); other images reject --dllm-algorithm for Gemma4.
if [ -n "$SGLANG_DLLM_ALGORITHM" ]; then
  args+=(--dllm-algorithm "$SGLANG_DLLM_ALGORITHM")
fi
# Server log level. Empty → no flag → SGLang's built-in default ('info'). Set
# SGLANG_LOG_LEVEL=debug (via sglang_log_level in defaults/main/sglang.yml or a model
# profile) to surface SGLang's logger.debug diagnostics — notably the Frozen-KV
# MTP draft-bind skip ("Draft model <class> does not implement ... skipping
# frozen-kv bind."), which names the class a Gemma-4 assistant draft loads as.
if [ -n "$SGLANG_LOG_LEVEL" ]; then
  args+=(--log-level "$SGLANG_LOG_LEVEL")
fi
if [ -n "$SGLANG_FP8_GEMM_RUNNER_BACKEND" ] && [ "$SGLANG_FP8_GEMM_RUNNER_BACKEND" != "auto" ]; then
  args+=(--fp8-gemm-backend "$SGLANG_FP8_GEMM_RUNNER_BACKEND")
fi
if [ -n "$SGLANG_FP4_GEMM_BACKEND" ] && [ "$SGLANG_FP4_GEMM_BACKEND" != "auto" ]; then
  args+=(--fp4-gemm-backend "$SGLANG_FP4_GEMM_BACKEND")
fi
# CUDA-Graph-Flags in der neuen Schreibweise (die alten sind deprecated, siehe
# den Loki-Nachweis beim mamba-Flag oben). Zwei davon sind KEINE reine
# Umbenennung, sondern werden auf die neuen per-Phase-Backends abgebildet:
#   --disable-cuda-graph           -> --cuda-graph-backend-decode disabled
#                                     + --cuda-graph-backend-prefill disabled
#   --disable-piecewise-cuda-graph -> --cuda-graph-backend-prefill disabled
#   --cuda-graph-max-bs            -> --cuda-graph-max-bs-decode
# _cg_prefill_disabled merkt sich, ob prefill schon abgeschaltet wurde, damit die
# beiden Zweige das Flag nicht doppelt setzen, wenn ein Profil beides angibt
# (disable_cuda_graph UND disable_piecewise_cuda_graph).
_cg_prefill_disabled=0

if [ "$SGLANG_DISABLE_CUDA_GRAPH" = "true" ] || [ "$SGLANG_CUDA_GRAPH_MAX_BS" = "0" ]; then
  args+=(--cuda-graph-backend-decode disabled --cuda-graph-backend-prefill disabled)
  _cg_prefill_disabled=1
elif [ -n "$SGLANG_CUDA_GRAPH_MAX_BS" ] && [ "$SGLANG_CUDA_GRAPH_MAX_BS" != "256" ]; then
  args+=(--cuda-graph-max-bs-decode "$SGLANG_CUDA_GRAPH_MAX_BS")
fi
if [ "$SGLANG_DISABLE_PIECEWISE_CUDA_GRAPH" = "true" ] && [ "$_cg_prefill_disabled" != "1" ]; then
  args+=(--cuda-graph-backend-prefill disabled)
fi
if [ "$SGLANG_WEIGHT_LOADER_DISABLE_MMAP" = "true" ]; then
  args+=(--weight-loader-disable-mmap)
fi
if [ "$SGLANG_WEIGHT_LOADER_DROP_CACHE_AFTER_LOAD" = "true" ]; then
  args+=(--weight-loader-drop-cache-after-load)
fi
if [ "$SGLANG_DISABLE_OVERLAP_SCHEDULE" = "true" ]; then
  args+=(--disable-overlap-schedule)
fi
if [ "$SGLANG_DISABLE_FLASHINFER_CUTLASS_MOE_FP4_ALLGATHER" = "true" ]; then
  args+=(--disable-flashinfer-cutlass-moe-fp4-allgather)
fi
if [ -n "$SGLANG_SERVED_MODEL_NAME" ]; then
  args+=(--served-model-name "$SGLANG_SERVED_MODEL_NAME")
fi
# Chat template kwargs (enable_thinking, etc.)
# These control Jinja2 chat template rendering — NOT sampling parameters.
# Sampling defaults (temperature, top_p, ...) are set via generation_config.json
# overlay above, because SGLang has no CLI flags for individual sampling params.
# NOTE: thinking_budget does NOT go here — it uses SGLang's custom logit processor
# system (--enable-custom-logit-processor), not the chat template.
if [ -n "$SGLANG_CHAT_TEMPLATE_KWARGS" ] && [ "$SGLANG_CHAT_TEMPLATE_KWARGS" != "{}" ]; then
  # CAPABILITY GUARD: some SGLang builds (e.g. the 0.5.12/0.5.14-sm121 images) do
  # NOT expose a server-side --chat-template-kwargs flag — passing it aborts the
  # server at argparse ("unrecognized arguments: --chat-template-kwargs") and
  # crash-loops the head. The flag's `dest` (chat_template_kwargs) is present in
  # server_args.py ONLY on builds that support it, so grep that file to decide.
  # When the flag is UNsupported we skip it and rely on the LiteLLM extra_body
  # default (_sglang_extra_body_defaults, same profile key) — which still applies
  # the reasoning default to all LiteLLM-routed traffic. Direct-to-SGLang requests
  # get no server-side default on such images (pass reasoning_effort per request).
  SGLANG_ARGS_FILE="/usr/local/lib/python3.12/dist-packages/sglang/srt/server_args.py"
  if grep -q "chat_template_kwargs" "$SGLANG_ARGS_FILE" 2>/dev/null; then
    args+=(--chat-template-kwargs "$SGLANG_CHAT_TEMPLATE_KWARGS")
  else
    echo "[launch] NOTE: this SGLang build has no --chat-template-kwargs flag; skipping the" \
         "server-side chat_template_kwargs default ($SGLANG_CHAT_TEMPLATE_KWARGS)." \
         "LiteLLM extra_body still carries it for proxied requests."
  fi
fi
# Enable custom logit processors (required for per-request thinking_budget via
# Qwen3ThinkingBudgetLogitProcessor). Safe to always enable — no-op if unused.
args+=(--enable-custom-logit-processor)

# Echo the exact launch command + relevant ENV to stdout so the head/worker
# pod logs (and Loki) capture them verbatim. printf '%q ' produces a shell-safe,
# copy-pasteable form (handles spaces, quotes, JSON args). Logged before exec
# so it appears even if the server crashes during startup.
#
# ENV filter: SGLANG_*, NCCL_*, FLASHINFER_*, TORCH*, CUDA_*, HF_*, plus a few
# named knobs that gate behavior at runtime (mamba/spec/JIT). Excludes generic
# vars (PATH, HOME, K8S_*, KUBERNETES_*, POD_*) to keep output focused.
printf '=== sglang launch ENV (filtered, secrets redacted) ===\n'
env | grep -E '^(SGLANG_|NCCL_|FLASHINFER_|TORCH(_|INDUCTOR_)|CUDA_|HF_|GLOO_|UCX_|RDMAV_|MASTER_|RANK=|WORLD_SIZE=|LOCAL_RANK=|NODE_RANK=|NNODES=|DIST_INIT_ADDR=|MAMBA_|SPEC_V2)' \
  | grep -vE '^(SGLANG_EXPECTED_IMAGE_PATTERN=|HF_HUB_OFFLINE_PATH=)' \
  | sed -E 's/^([A-Z_0-9]*(TOKEN|SECRET|KEY|PASSWORD|PASS|API|CREDENTIAL)[A-Z_0-9]*)=.*/\1=***REDACTED***/' \
  | LC_ALL=C sort
printf '=== end sglang launch ENV ===\n'

printf '=== sglang launch command (%d args) ===\n' "${#args[@]}"
printf '%q ' "${args[@]}"
printf '\n=== end sglang launch command ===\n'

exec "${args[@]}"
