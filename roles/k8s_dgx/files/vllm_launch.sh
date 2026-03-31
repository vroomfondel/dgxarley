#!/bin/bash
set -e

# Install tools for ARP priming (not included in vllm image)
apt-get update -qq && apt-get install -y -qq tini iproute2 iputils-ping net-tools curl ethtool >/dev/null 2>&1

# Prime ARP table on the QSFP link before NCCL tries to connect.
# Without this, the first TCP SYNs get dropped until ARP resolves,
# causing ~230s delay in "Init torch distributed".
# Full-mesh: every node pings ALL other nodes so NCCL's ring/tree
# topology can communicate immediately over any path.
IFS=',' read -ra peers <<< "$QSFP_PEER_IPS"
for peer in "${peers[@]}"; do
  echo "Waiting for QSFP peer ${peer} ..."
  until ping -c10 -W1 "$peer" ; do
    sleep 1
  done
  echo "QSFP peer ${peer} reachable."
done

# When load_format is sharded_state, wait for the shard marker then use sharded path
model_path="$VLLM_MODEL"
if [ "$VLLM_LOAD_FORMAT" = "sharded_state" ]; then
  model_slug=$(echo "$VLLM_MODEL" | sed 's|/|--|g')
  shard_suffix="vllm-TP${TP}"
  if [ "$VLLM_ENABLE_EP" = "true" ]; then
    # vLLM EP size = TP × DP; with DP=1, EP = TP
    shard_suffix="${shard_suffix}-EP${TP}"
  fi
  if [ -n "$VLLM_QUANTIZATION" ]; then
    shard_suffix="${shard_suffix}-${VLLM_QUANTIZATION}"
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

# Patch moe_wna16 weight loader for EP-aware qzeros handling (vLLM 0.17.0 bug).
# Two bugs in the w13_qzeros/w2_qzeros branches:
#   1. Uses raw global expert_id (0-127) instead of local EP index (0-63)
#   2. Uses global tp_rank for TP-slice, but moe_tp_size=tp/ep — need tp_rank % moe_tp_size
# Safe no-op when ep_size=1 (identity mapping, tp_rank unchanged).
MOE_WNA16="/usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/quantization/moe_wna16.py"
if grep -q 'param\.data\[expert_id' "$MOE_WNA16" 2>/dev/null; then
  python3 << 'PATCH_QZEROS_EOF'
import sys
f = "/usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/quantization/moe_wna16.py"
with open(f) as fh:
    code = fh.read()
old_w13 = '''            if "w13_qzeros" in weight_name:
                tensor = loaded_weight.view(
                    layer.moe_tp_size, -1, loaded_weight.size(1)
                )[tp_rank]
                if shard_id == "w1":
                    param.data[expert_id, : shard_size // 2] = tensor
                else:
                    param.data[expert_id, shard_size // 2 :] = tensor'''
new_w13 = '''            if "w13_qzeros" in weight_name:
                _local_id = layer._map_global_expert_id_to_local_expert_id(expert_id)
                if _local_id == -1:
                    return
                _moe_tp_rank = tp_rank % layer.moe_tp_size
                tensor = loaded_weight.view(
                    layer.moe_tp_size, -1, loaded_weight.size(1)
                )[_moe_tp_rank]
                if shard_id == "w1":
                    param.data[_local_id, : shard_size // 2] = tensor
                else:
                    param.data[_local_id, shard_size // 2 :] = tensor'''
old_w2 = '''            elif "w2_qzeros" in weight_name:
                param.data[expert_id] = loaded_weight.view(
                    loaded_weight.size(0), layer.moe_tp_size, -1
                )[:, tp_rank]'''
new_w2 = '''            elif "w2_qzeros" in weight_name:
                _local_id = layer._map_global_expert_id_to_local_expert_id(expert_id)
                if _local_id == -1:
                    return
                _moe_tp_rank = tp_rank % layer.moe_tp_size
                param.data[_local_id] = loaded_weight.view(
                    loaded_weight.size(0), layer.moe_tp_size, -1
                )[:, _moe_tp_rank]'''
if old_w13 not in code:
    print("w13_qzeros: already patched or source changed, skipping")
    sys.exit(0)
if old_w2 not in code:
    print("w2_qzeros: already patched or source changed, skipping")
    sys.exit(0)
code = code.replace(old_w13, new_w13, 1)
code = code.replace(old_w2, new_w2, 1)
with open(f, 'w') as fh:
    fh.write(code)
print("Patched moe_wna16.py: EP-aware expert_id + tp_rank remapping for qzeros")
PATCH_QZEROS_EOF
fi

args=(
  tini -s --
  python3 -m vllm.entrypoints.openai.api_server
  --model "$model_path"
  --max-model-len "$VLLM_CONTEXT_LENGTH"
  --kv-cache-dtype "$VLLM_KV_CACHE_DTYPE"
  --gpu-memory-utilization "$VLLM_MEM_FRACTION"
  --tensor-parallel-size "$TP"
  --nnodes "$NNODES"
  --node-rank "$NODE_RANK"
  --master-addr "$QSFP_IP_SPARK1"
  --master-port "$NCCL_PORT"
  --distributed-executor-backend mp
  --port "$VLLM_PORT"
)

# Head node (rank 0): bind to all interfaces so the K8s Service can reach it.
# vLLM doesn't have the SGLang EADDRINUSE bug — no HAProxy sidecar needed.
# Worker node (rank 1+): use --headless (no API server) and bind to the QSFP pod IP.
if [ "$NODE_RANK" = "0" ]; then
  args+=(--host 0.0.0.0)
else
  args+=(--headless)
  args+=(--host "$VLLM_HOST")
fi

if [ -n "$VLLM_QUANTIZATION" ]; then
  args+=(--quantization "$VLLM_QUANTIZATION")
fi
if [ -n "$VLLM_LOAD_FORMAT" ] && [ "$VLLM_LOAD_FORMAT" != "auto" ]; then
  args+=(--load-format "$VLLM_LOAD_FORMAT")
fi
if [ -n "$VLLM_REASONING_PARSER" ]; then
  args+=(--reasoning-parser "$VLLM_REASONING_PARSER")
fi
if [ -n "$VLLM_TOOL_CALL_PARSER" ]; then
  args+=(--tool-call-parser "$VLLM_TOOL_CALL_PARSER")
  args+=(--enable-auto-tool-choice)
fi
if [ -n "$VLLM_SERVED_MODEL_NAME" ]; then
  args+=(--served-model-name "$VLLM_SERVED_MODEL_NAME")
fi
# Disable CUDA graph capture (useful for debugging or low-VRAM scenarios)
if [ "$VLLM_ENFORCE_EAGER" = "true" ]; then
  args+=(--enforce-eager)
fi
# Expert parallelism: partitions experts across the TP group for MoE layers.
if [ "$VLLM_ENABLE_EP" = "true" ]; then
  args+=(--enable-expert-parallel)
fi
if [ -n "$VLLM_MAX_NUM_SEQS" ] && [ "$VLLM_MAX_NUM_SEQS" != "0" ]; then
  args+=(--max-num-seqs "$VLLM_MAX_NUM_SEQS")
fi
if [ -n "$VLLM_DIST_TIMEOUT" ]; then
  args+=(--distributed-timeout-seconds "$VLLM_DIST_TIMEOUT")
fi
# Chat template kwargs (enable_thinking, etc.)
# Controls Jinja2 chat template rendering — NOT sampling parameters.
if [ -n "$VLLM_CHAT_TEMPLATE_KWARGS" ] && [ "$VLLM_CHAT_TEMPLATE_KWARGS" != "{}" ]; then
  args+=(--chat-template-kwargs "$VLLM_CHAT_TEMPLATE_KWARGS")
fi

exec "${args[@]}"
