# SGLang Test Log — MiniMax M2.5 NVFP4, 3 Nodes, v0.5.10rc0

## Environment

| Component | Value |
|-----------|-------|
| GPU | NVIDIA GB10 (SM121/Blackwell), 128 GB per node |
| Driver | 580.142 |
| CUDA | 13.0 |
| Kernel | 6.17.0-1014-nvidia |
| OS | Ubuntu 24.04.4 LTS (aarch64) |
| K3s | v1.35.3+k3s1 |
| Nodes | spark1, spark2, spark3 (1 GPU each) |
| Image | `scitrera/dgx-spark-sglang:0.5.10rc0` |
| Model | `nvidia/MiniMax-M2.5-NVFP4` |

Previous test series with `0.5.9-dev2-acab24a7-t5`: see `TESTLOG_nv580.142_sglang-0.5.9-dev2_minimax-m2.5-nvfp4_3n.md`.

---

## Baseline: Winner config from 0.5.9-dev2 series (Test 13)

```
tp_size=1, pp_size=3, ep_size=1
moe_runner_backend=triton
attention_backend=flashinfer
fp4_gemm_backend=flashinfer_cutlass
disable_cuda_graph=false
disable_piecewise_cuda_graph=true
pp_async_batch_depth=0
cuda_graph_max_bs=8
disable_deep_gemm=true
quantization=modelopt_fp4
kv_cache_dtype=fp8_e4m3
mem_fraction_static=0.80
context_length=196608
max_running_requests=32
```

Throughput on 0.5.9-dev2: 16.1 tok/s (1∥), 31.5 tok/s (4∥ avg), 50.6 tok/s (4∥ peak).

---

## 2026-04-01: v0.5.10rc0 — initial test

### Test 1: NCCL init failures with RoCE

Initial attempts to use RoCE transport failed due to multiple NCCL config issues:

- `NCCL_NET=""` (empty string) ≠ unset → NCCL tries to match plugin named `""` → `ncclInvalidUsage` at PP CommSplit
- `NCCL_IB_DISABLE=""` residue from strategic merge (old ConfigMap key not removed)
- `NCCL_IB_HCA="roceenp1s0f0v0"` (wrong — included `en` prefix from netdev name, correct: `rocep1s0f0v0`)
- Missing `NCCL_IB_MERGE_NICS=0` → PF+VF merge heuristic breaks VF-only communication

All fixed by: removing stale keys (delete + recreate ConfigMap), correcting HCA name, adding MERGE_NICS=0.

### Test 2: RoCE transport, winner config

- **Config:** Same as baseline, plus `nccl_transport=roce` (NCCL_IB_HCA=rocep1s0f0v0, NCCL_IB_GID_INDEX=3, NCCL_IB_MERGE_NICS=0, no NCCL_NET, no NCCL_IB_DISABLE).
- **NCCL transport confirmed:** `NET/IBext_v11` using `rocep1s0f0v0:1/RoCE`, GPU Direct RDMA (DMABUF) enabled. NOT Socket.
- **NCCL version:** 2.29.2+cuda13.1 (different from 0.5.9-dev2 which had 2.29.3)
- **Result:** **STABLE** — server starts, CUDA graph capture succeeds, inference works.
- **Throughput:**

  | Metric | 1 request | 4 parallel |
  |--------|-----------|------------|
  | Successful / failed | 1 / 0 | 3 / 1 (REP stop) |
  | Aggregate throughput | 7.9 tok/s | 16.1 tok/s |
  | Avg per-request tok/s | 7.9 | 6.8 |
  | Peak concurrent tok/s | — | ~41 (server-side) |

- **Note:** 1 of 4 requests stopped early (repetition detection), so client-side aggregate is skewed. Server-side gen throughput peak ~41 tok/s during the 4∥ run.
- **vs. 0.5.9-dev2 Socket (Test 13):** server-side peak **-16%** (41 vs 49). Image regression, not transport-related.

### Test 3: Socket transport (for comparison)

- **Config:** Same as Test 2 but `nccl_transport=socket` (NCCL_NET=Socket, NCCL_IB_DISABLE=1).
- **Purpose:** Isolate whether the throughput regression is from RoCE or from the 0.5.10rc0 image itself.
- **Result:** *pending*

---

## Configuration Matrix

All tests use: `tp=1, pp=3, ep=1, quantization=modelopt_fp4, kv_cache_dtype=fp8_e4m3, mem_fraction_static=0.80, disable_deep_gemm=true, context_length=196608, max_running_requests=32, schedule_policy=lpm, watchdog_timeout=3600, dist_timeout=1800` unless noted.

| # | nccl_transport | moe_runner | attention | fp4_gemm | dis_cuda_graph | dis_piecewise | pp_async | cuda_graph_max_bs | Stability | 1∥ tok/s | 4∥ avg | 4∥ peak | 8∥ avg | 8∥ peak |
|---|----------------|------------|-----------|----------|----------------|---------------|----------|-------------------|-----------|---------|--------|---------|--------|---------|
| 1 | roce (broken) | triton | flashinfer | fi_cutlass | false | true | 0 | 8 | NCCL invalid usage | — | — | — | — | — |
| 2 | roce | triton | flashinfer | fi_cutlass | false | true | 0 | 8 | **STABLE** | 7.9 | 16.1 | ~41 (srv) | 33.8 | ~51 (srv) |
| 3 | socket | triton | flashinfer | fi_cutlass | false | true | 0 | 8 | *pending* | — | — | — | — | — |

### Column Legend

| Column | Description |
|--------|-------------|
| moe_runner | `moe_runner_backend` — MoE expert dispatch kernel (`fi_cutlass` = flashinfer_cutlass, `triton` = triton→cutlass_moe_fp4 fallback for NVFP4) |
| attention | `attention_backend` — attention kernel |
| fp4_gemm | `fp4_gemm_backend` — FP4 dense GEMM kernel |
| dis_cuda_graph | `disable_cuda_graph` — true = eager mode, false = capture CUDA graphs |
| dis_piecewise | `disable_piecewise_cuda_graph` — true = only fixed-BS graphs |
| pp_async | `pp_async_batch_depth` — async micro-batches in PP pipeline (0 = synchronous) |
| cuda_graph_max_bs | `cuda_graph_max_bs` — largest batch size to capture |
| 1∥ tok/s | Aggregate throughput with 1 sequential request |
| 4∥ avg | Aggregate throughput with 4 parallel requests (total output tokens / wall time) |
| 4∥ peak | Peak concurrent throughput at 4∥ = sum of per-request tok/s while all requests active |
| 8∥ avg | Aggregate throughput with 8 parallel requests |
| 8∥ peak | Peak concurrent throughput at 8∥ |
