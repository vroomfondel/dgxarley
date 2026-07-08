#!/bin/bash
set -e

# ---------------------------------------------------------------------------
# Minimal SGLang launcher for the EMBEDDING instance (always single-node, tp=1).
#
# Deliberately SEPARATE from the ~1800-line generation launcher
# (sglang_launch.sh): an embedding server needs none of that script's chat/
# generation machinery (reasoning/tool parsers, speculative decoding, MoE runner
# backends, NCCL/QSFP distributed rendezvous, custom logit processors). Selected
# per-instance by sglang_instance.yml when inst.is_embedding is true (mounted as
# /scripts/launch.sh in place of the big script).
#
# Reads the SAME sglang-<prefix>-config ConfigMap env vars as the big launcher,
# but only the subset an embedding server actually consumes: the model/serving
# basics PLUS the full resource envelope (max_total_tokens, max_running_requests,
# mem_fraction_static, context_length, chunked_prefill_size, kv_cache_dtype,
# page_size, schedule_policy, cuda-graph, dtype, log_level). Everything else in
# the ConfigMap is generation-only (speculative decoding, MoE runner backends,
# NCCL/RoCE, mamba, DSV4/DeepGEMM kernel toggles) and is INTENTIONALLY ignored —
# inert for a dense embedding model. Single-node means NO --nnodes/--node-rank/
# --nccl-init-addr (no distributed group).
#
# Model choice rationale (Qwen3-Embedding, decoder arch): serving bge-m3
# (XLM-RoBERTa) here would hit sglang#7590 on GB10/SM121 (position-tensor assert
# in roberta.py, crash on the 2nd request). Qwen3-Embedding is unaffected.
# ---------------------------------------------------------------------------

# tini for correct signal handling / zombie reaping (not in the sglang image).
apt-get update -qq && apt-get install -y -qq tini >/dev/null 2>&1

# HF id resolves against the mounted HF cache (HF_HUB_OFFLINE=1 from the ConfigMap;
# the model-download initContainer has already populated /root/.cache/huggingface).
model_path="$SGLANG_MODEL"

args=(
  tini -s --
  python3 -m sglang.launch_server
  --model-path "$model_path"
  --is-embedding
  --tp-size "${TP:-1}"
  --context-length "$SGLANG_CONTEXT_LENGTH"
  --mem-fraction-static "$SGLANG_MEM_FRACTION"
  --port "$SGLANG_PORT"
  # Radix cache (prefix KV reuse) is a decode-time optimization, useless for
  # one-shot embed passes. No env knob — always off for embedding.
  --disable-radix-cache
)

# --- Serving envelope, wired from the shared sglang-<prefix>-config ConfigMap --
# The embed server's MEMORY should be pinned by the ABSOLUTE --max-total-tokens
# (a deterministic KV-pool token cap), NOT by --mem-fraction-static: the latter
# is (weights + KV pool) / TOTAL GPU capacity, so on the shared/time-sliced GB10
# the derived pool size shifts with whatever a co-tenant already holds and with
# instance start order. SGLang allocates a KV pool even in --is-embedding mode
# (sgl-project/sglang#9181), so this cap is what actually bounds it. The profile
# therefore sets mem_fraction_static high (headroom only) and max_total_tokens as
# the real, start-order-independent cap. Each knob is guarded so an empty/sentinel
# value (defaults/main/sglang.yml) is a no-op — set -e safe via if/then, NOT `&&` chains
# (a failed `[ ]` test under set -e would kill the script).
if [ -n "${SGLANG_MAX_TOTAL_TOKENS:-}" ]; then
  args+=(--max-total-tokens "$SGLANG_MAX_TOTAL_TOKENS")
fi
if [ -n "${SGLANG_MAX_RUNNING_REQUESTS:-}" ]; then
  args+=(--max-running-requests "$SGLANG_MAX_RUNNING_REQUESTS")
fi
# -1 = single-chunk prefill (the embedding default; sidesteps chunked-prefill
# position handling). Tunable via the profile's chunked_prefill_size.
if [ -n "${SGLANG_CHUNKED_PREFILL_SIZE:-}" ]; then
  args+=(--chunked-prefill-size "$SGLANG_CHUNKED_PREFILL_SIZE")
fi
if [ -n "${SGLANG_KV_CACHE_DTYPE:-}" ]; then
  args+=(--kv-cache-dtype "$SGLANG_KV_CACHE_DTYPE")
fi
# Quantization escape hatch: compressed-tensors FP8 (chroma-core FP8-Dynamic) is
# AUTO-DETECTED from the checkpoint's quantization_config, so the profile leaves
# this empty. Set profile.quantization only if a load error shows auto-detect
# missed it; then it flows here.
if [ -n "${SGLANG_QUANTIZATION:-}" ]; then
  args+=(--quantization "$SGLANG_QUANTIZATION")
fi
if [ -n "${SGLANG_SCHEDULE_POLICY:-}" ]; then
  args+=(--schedule-policy "$SGLANG_SCHEDULE_POLICY")
fi
# page_size / cuda_graph_max_bs use 0 as the "no override" sentinel (defaults).
if [ -n "${SGLANG_PAGE_SIZE:-}" ] && [ "${SGLANG_PAGE_SIZE}" != "0" ]; then
  args+=(--page-size "$SGLANG_PAGE_SIZE")
fi
if [ "${SGLANG_DISABLE_CUDA_GRAPH:-false}" = "true" ]; then
  args+=(--disable-cuda-graph)
elif [ -n "${SGLANG_CUDA_GRAPH_MAX_BS:-}" ] && [ "${SGLANG_CUDA_GRAPH_MAX_BS}" != "0" ]; then
  args+=(--cuda-graph-max-bs "$SGLANG_CUDA_GRAPH_MAX_BS")
fi
# dtype: "auto" (the resolver default) == SGLang's own default → skip to avoid
# clutter; only pass when the profile forces a specific dtype.
if [ -n "${SGLANG_MODEL_DTYPE:-}" ] && [ "${SGLANG_MODEL_DTYPE}" != "auto" ]; then
  args+=(--dtype "$SGLANG_MODEL_DTYPE")
fi
if [ -n "${SGLANG_LOG_LEVEL:-}" ]; then
  args+=(--log-level "$SGLANG_LOG_LEVEL")
fi

# --host 127.0.0.1 (from the pod env): the HAProxy sidecar forwards
# 0.0.0.0:<inst.port> → 127.0.0.1:<inst.internal_port>, same EADDRINUSE fix as
# the generation head (SGLang's Scheduler binds <pod-ip>:port; uvicorn on
# 0.0.0.0 would collide).
if [ -n "${SGLANG_HOST:-}" ]; then
  args+=(--host "$SGLANG_HOST")
fi
# Encoder-friendly attention backend (profile default triton; SGLang's BGE note).
if [ -n "${SGLANG_ATTENTION_BACKEND:-}" ]; then
  args+=(--attention-backend "$SGLANG_ATTENTION_BACKEND")
fi
if [ "${SGLANG_TRUST_REMOTE_CODE:-false}" = "true" ]; then
  args+=(--trust-remote-code)
fi
# Stable served name (kept == the old ollama logical name so consumers that
# reference the model by name need no change on cutover).
if [ -n "${SGLANG_SERVED_MODEL_NAME:-}" ]; then
  args+=(--served-model-name "$SGLANG_SERVED_MODEL_NAME")
fi
# Prometheus exporter (/metrics on the HTTP port), gated per-instance.
if [ "${SGLANG_ENABLE_METRICS:-false}" = "true" ]; then
  args+=(--enable-metrics)
fi

# Echo the exact launch command to stdout (Loki-captured), same convention as
# the generation launcher.
printf '=== sglang EMBED launch command (%d args) ===\n' "${#args[@]}"
printf '%q ' "${args[@]}"
printf '\n=== end sglang EMBED launch command ===\n'

exec "${args[@]}"
