# Pipeline Parallelism (PP) vs Tensor Parallelism (TP) — Weight Distribution

Comparison of per-node GPU memory usage across parallelism strategies on the 3-node DGX Spark cluster (128 GB VRAM per GPU).

## Current: PP=3, TP=1 — `nvidia/MiniMax-M2.5-NVFP4`

229B/8-active MoE, NVFP4 quantized, 62 layers.

| | Total | Per Node (3 nodes) |
|---|---|---|
| **Weights** | ~139 GB | ~46 GB |
| **Free for KV cache** | — | ~82 GB |

Each node receives ~21 of 62 layers as a contiguous pipeline stage. All 3 GPUs are utilized.

**Why PP instead of TP:** MiniMax-M2.5 has `num_key_value_heads=8`, which is not divisible by 3, making TP=3 impossible. TP=2 would leave the third node idle. PP=3 distributes layers evenly across all 3 nodes with lower per-GPU weight load.

**Trade-off:** Higher per-request latency (pipeline bubbles) but equivalent throughput with sufficient concurrency (micro-batching hides bubbles).

## Previous: PP=1, TP=2 — Qwen MoE models

With TP=2, every weight tensor is split across 2 GPUs. Only 2 of 3 nodes are used; the third is idle.

| Model | Quant | Total Weights | Per Node (2 nodes) | Free for KV cache |
|---|---|---|---|---|
| `Qwen3-235B-A22B-Instruct-2507-FP8` | FP8 | ~237 GB | ~118.5 GB | ~9.5 GB |
| `Qwen3.5-122B-A10B-FP8` | FP8 | ~127 GB | ~63.5 GB | ~64.5 GB |
| `QuantTrio/Qwen3-235B-A22B-*-AWQ` | AWQ 4-bit | ~124 GB | ~62 GB | ~66 GB |

## Summary

| | PP=3, TP=1 (MiniMax) | PP=1, TP=2 (Qwen) |
|---|---|---|
| **Nodes used** | 3 of 3 | 2 of 3 |
| **Weights per GPU** | ~46 GB | ~62-118 GB |
| **KV cache headroom** | ~82 GB | ~10-66 GB |
| **Latency** | Higher (pipeline bubbles) | Lower (no pipeline stalls) |
| **Throughput** | Same with enough concurrency | Same |
| **Constraint** | KV heads not divisible by nnodes | Needs even KV head count per TP degree |

## Expert Parallelism (EP) — Inter-Node Traffic Considerations

For MoE models (MiniMax, Qwen3-MoE, GLM-MoE, …) the choice of `ep_size` determines where expert computation lives — and crucially, **where the inter-node communication happens**. Counterintuitively, higher `ep` typically means *less* cross-node traffic, not more.

### `tp=4, ep=1` on 4 nodes

The TP group spans **all 4 nodes** (1 GPU per node in the TP group). Every layer — attention AND expert FFN — requires a cross-node all-reduce over RoCE for every token. The three other GPUs per node sit idle unless a separate DP replica is scheduled on them.

- **Dense layers:** cross-node all-reduce (RoCE) at every layer boundary.
- **Expert layers:** cross-node all-reduce (RoCE) — experts are TP-sharded across nodes.
- **Active GPUs:** 4 (1 per node).
- **Inter-node traffic:** every layer, every token.

### `tp=4, ep=4` on 4 nodes

TP now stays **intra-node** (4 GPUs per node connected via NVLink). EP distributes experts across the 4 nodes, so each node owns 1/4 of the expert pool. Only the MoE routing step needs to cross the network.

- **Dense layers:** intra-node TP via NVLink (~900 GB/s). No inter-node traffic.
- **Expert layers:** cross-node all-to-all (RoCE) for token→expert routing only.
- **Active GPUs:** 16 (4 per node).
- **Inter-node traffic:** only for expert dispatch, not for dense computation.

### Comparison

| | `tp=4, ep=1` | `tp=4, ep=4` |
|---|---|---|
| **TP group spans** | 4 nodes (cross-node) | 1 node (NVLink) |
| **Active GPUs** | 4 | 16 |
| **Dense layer comms** | Cross-node all-reduce per layer | Intra-node NVLink |
| **Expert layer comms** | Cross-node all-reduce per layer | Cross-node all-to-all (routing only) |
| **RoCE traffic** | every layer, every token | only MoE dispatch |
| **Expected throughput** | baseline | 2-4× higher |

**Key insight:** Under-estimating `ep` on a multi-node MoE deployment doesn't save communication — it *maximizes* it, because TP is forced across the slow inter-node link. Matching `ep_size` to `nnodes` keeps TP on NVLink and only pays the network cost for the (relatively sparse) expert dispatch step.

**When `ep=1` still makes sense:** if the entire model fits on a single node and you run pure data-parallel replicas (one full copy per node), each node serves requests independently with *zero* inter-node traffic. This only works when `weights_per_node ≤ GPU_VRAM × GPUs_per_node`, and trades off KV-cache headroom against replica count.
