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
