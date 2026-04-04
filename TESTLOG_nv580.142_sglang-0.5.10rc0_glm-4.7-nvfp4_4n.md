# SGLang Test Log — GLM 4.7 NVFP4, 4 Nodes, v0.5.10rc0

## Environment

| Component | Value                                          |
|-----------|------------------------------------------------|
| GPU | NVIDIA GB10 (SM121/Blackwell), 128 GB per node |
| Driver | 580.142                                        |
| CUDA | 13.0                                           |
| Kernel | 6.17.0-1014-nvidia                             |
| OS | Ubuntu 24.04.4 LTS (aarch64)                   |
| K3s | v1.35.3+k3s1                                   |
| Nodes | spark1, spark2, spark3, spark4 (1 GPU each)      |
| Image | `scitrera/dgx-spark-sglang:0.5.10rc0`         |
| Model | `nvidia/GLM-4.7-NVFP4`                    |

---

## Configuration Matrix

All tests use: `tp=4, pp=1, ep=4, quantization=modelopt_fp4, kv_cache_dtype=fp8_e4m3, mem_fraction_static=0.60, disable_deep_gemm=true, context_length=196608, max_running_requests=32, schedule_policy=lpm, watchdog_timeout=3600, dist_timeout=1800` unless noted.

| # | nccl_transport | moe_runner | attention | fp4_gemm | dis_cuda_graph | dis_piecewise | pp_async | cuda_graph_max_bs | Stability | 1∥ tok/s | 4∥ tok/s | 8∥ tok/s |
|---|----------------|------------|-----------|----------|----------------|---------------|----------|-------------------|-----------|---------|---------|---------|
| 1 | socket | triton | flashinfer | fi_cutlass | false | true | 0 | 8 | **startup_crash** | — | — | — |
| 2 | socket | triton | flashinfer | fi_cutlass | true | true | 0 | — | **infer_error** | — | — | — |
| 3 | socket | triton | flashinfer | fi_cutlass | false | false | 0 | 8 | **startup_crash** | — | — | — |
| 4 | socket | triton | triton | fi_cutlass | false | true | 0 | 8 | **startup_crash** | — | — | — |
| 5 | socket | triton | triton | fi_cutlass | true | true | 0 | — | **bench_crash** | — | — | — |
| 6 | socket | triton | triton | fi_cutlass | false | false | 0 | 8 | **startup_crash** | — | — | — |
| 7 | socket | triton | flashinfer | fi_cudnn | false | true | 0 | 8 | **startup_crash** | — | — | — |
| 8 | socket | triton | flashinfer | fi_cudnn | true | true | 0 | — | **infer_error** | — | — | — |
| 9 | socket | triton | flashinfer | fi_cudnn | false | false | 0 | 8 | **startup_crash** | — | — | — |
| 10 | socket | triton | triton | fi_cudnn | false | true | 0 | 8 | **startup_crash** | — | — | — |
| 11 | socket | triton | triton | fi_cudnn | true | true | 0 | — | **infer_error** | — | — | — |
| 12 | socket | triton | triton | fi_cudnn | false | false | 0 | 8 | **startup_crash** | — | — | — |
| 13 | socket | fi_cutlass | flashinfer | fi_cutlass | false | true | 0 | 8 | **startup_crash** | — | — | — |
| 14 | socket | fi_cutlass | flashinfer | fi_cutlass | true | true | 0 | — | **infer_error** | — | — | — |
| 15 | socket | fi_cutlass | flashinfer | fi_cutlass | false | false | 0 | 8 | **startup_crash** | — | — | — |
| 16 | socket | fi_cutlass | triton | fi_cutlass | false | true | 0 | 8 | **startup_crash** | — | — | — |
| 17 | socket | fi_cutlass | triton | fi_cutlass | true | true | 0 | — | pending | — | — | — |
| 18 | socket | fi_cutlass | triton | fi_cutlass | false | false | 0 | 8 | pending | — | — | — |
| 19 | socket | fi_cutlass | flashinfer | fi_cudnn | false | true | 0 | 8 | pending | — | — | — |
| 20 | socket | fi_cutlass | flashinfer | fi_cudnn | true | true | 0 | — | pending | — | — | — |
| 21 | socket | fi_cutlass | flashinfer | fi_cudnn | false | false | 0 | 8 | pending | — | — | — |
| 22 | socket | fi_cutlass | triton | fi_cudnn | false | true | 0 | 8 | pending | — | — | — |
| 23 | socket | fi_cutlass | triton | fi_cudnn | true | true | 0 | — | pending | — | — | — |
| 24 | socket | fi_cutlass | triton | fi_cudnn | false | false | 0 | 8 | pending | — | — | — |
| 25 | socket | cutlass | flashinfer | fi_cutlass | false | true | 0 | 8 | pending | — | — | — |
| 26 | socket | cutlass | flashinfer | fi_cutlass | true | true | 0 | — | pending | — | — | — |
| 27 | socket | cutlass | flashinfer | fi_cutlass | false | false | 0 | 8 | pending | — | — | — |
| 28 | socket | cutlass | triton | fi_cutlass | false | true | 0 | 8 | pending | — | — | — |
| 29 | socket | cutlass | triton | fi_cutlass | true | true | 0 | — | pending | — | — | — |
| 30 | socket | cutlass | triton | fi_cutlass | false | false | 0 | 8 | pending | — | — | — |
| 31 | socket | cutlass | flashinfer | fi_cudnn | false | true | 0 | 8 | pending | — | — | — |
| 32 | socket | cutlass | flashinfer | fi_cudnn | true | true | 0 | — | pending | — | — | — |
| 33 | socket | cutlass | flashinfer | fi_cudnn | false | false | 0 | 8 | pending | — | — | — |
| 34 | socket | cutlass | triton | fi_cudnn | false | true | 0 | 8 | pending | — | — | — |
| 35 | socket | cutlass | triton | fi_cudnn | true | true | 0 | — | pending | — | — | — |
| 36 | socket | cutlass | triton | fi_cudnn | false | false | 0 | 8 | pending | — | — | — |

### Column Legend

| Column | Description |
|--------|-------------|
| nccl_transport | `sglang_nccl_transport` — NCCL inter-node transport (`socket` = TCP/IP, `roce` = RDMA/RoCE via IBext) |
| moe_runner | `moe_runner_backend` — MoE expert dispatch kernel (`fi_cutlass` = flashinfer_cutlass, `triton` = triton→cutlass_moe_fp4 fallback for NVFP4, `cutlass` = cutlass direct) |
| attention | `attention_backend` — attention kernel (`flashinfer` = FlashInfer, `triton` = Triton) |
| fp4_gemm | `fp4_gemm_backend` — FP4 dense GEMM kernel (`fi_cutlass` = flashinfer_cutlass, `fi_cudnn` = flashinfer_cudnn; valid choices: auto, flashinfer_cudnn, flashinfer_cutlass, flashinfer_trtllm) |
| dis_cuda_graph | `disable_cuda_graph` — true = eager mode, false = capture CUDA graphs |
| dis_piecewise | `disable_piecewise_cuda_graph` — true = only fixed-BS graphs, false = piecewise variable-length graphs |
| pp_async | `pp_async_batch_depth` — async micro-batches in PP pipeline (0 = synchronous). Irrelevant with PP=1 (TP-only), always 0 here. |
| cuda_graph_max_bs | `cuda_graph_max_bs` — largest batch size to capture (— = N/A when graphs disabled) |
| 1∥ tok/s | Throughput with 1 sequential request (= per-request tok/s) |
| 4∥ tok/s | Peak concurrent throughput at 4∥ (sum of per-request tok/s) |
| 8∥ tok/s | Peak concurrent throughput at 8∥ (sum of per-request tok/s) |

---

## Test Details

> **Note:** `mem_fraction_static` varies per run (effective profile values shown in details).

### #1 — triton moe / flashinfer attn / fi_cutlass fp4 / cuda_graph

- **Outcome:** startup_crash (multiple runs)
- **1st run:** mem_frac=0.70, 2026-04-04 09:44–09:50 UTC — worker-1 **OOMKilled**
- **2nd run:** mem_frac=0.60, 2026-04-04 10:02–10:09 UTC — head + all 3 workers restarted
- **Note:** Different failure mode than v0.5.9dev2 — no cutlass_moe_fp4 device-side assert. 1st run was OOM, 2nd run at lower mem_frac still crashed (likely CUDA graph capture failure)

### #2 — triton moe / flashinfer attn / fi_cutlass fp4 / no-cuda-graph

- **Outcome:** infer_error — server stable, all requests returned errors (0 tokens)
- **1st run:** mem_frac=0.70, 2026-04-04 09:50–09:58 UTC — bench_crash, smoke test (n=8) OOMKilled
- **2nd run:** mem_frac=0.60, 2026-04-04 ~12:16 CEST — server started, 0/1 + 0/4 + 0/8 successful

### #3 — triton moe / flashinfer attn / fi_cutlass fp4 / piecewise

- **Outcome:** startup_crash
- **Error:** Worker-1 restarted (total=1)
- **Time:** 2026-04-04 10:16–10:23 UTC

### #4 — triton moe / triton attn / fi_cutlass fp4 / cuda_graph

- **Outcome:** startup_crash
- **Error:** Head + all 3 workers restarted (total=1 each)
- **Time:** 2026-04-04 10:24–10:30 UTC

### #5 — triton moe / triton attn / fi_cutlass fp4 / no-cuda-graph

- **Outcome:** bench_crash — head + worker-2 crashed during benchmark
- **Time:** 2026-04-04 10:31 UTC
- n=1: 0/1 successful (1 error, 0 tokens)

### #6 — triton moe / triton attn / fi_cutlass fp4 / piecewise

- **Outcome:** startup_crash
- **Error:** Worker-2 + worker-3 restarted (total=1 each)
- **Time:** 2026-04-04 10:38–10:44 UTC

### #7 — triton moe / flashinfer attn / fi_cudnn fp4 / cuda_graph

- **Outcome:** startup_crash
- **Error:** Head + all 3 workers restarted (total=1 each)
- **Time:** 2026-04-04 10:45–10:52 UTC

### #8 — triton moe / flashinfer attn / fi_cudnn fp4 / no-cuda-graph

- **Outcome:** infer_error — server stable, all requests returned errors (0 tokens)
- **Time:** 2026-04-04 ~12:59 CEST
- n=1: 0/1, n=4: 0/4, n=8: 0/8 successful

### #9 — triton moe / flashinfer attn / fi_cudnn fp4 / piecewise

- **Outcome:** startup_crash
- **Error:** Head + all 3 workers restarted (total=1 each)
- **Time:** 2026-04-04 10:59–11:06 UTC

### #10 — triton moe / triton attn / fi_cudnn fp4 / cuda_graph

- **Outcome:** startup_crash
- **Error:** Head + all 3 workers restarted (total=1 each)
- **Time:** 2026-04-04 11:06–11:13 UTC

### #11 — triton moe / triton attn / fi_cudnn fp4 / no-cuda-graph

- **Outcome:** infer_error — server stable, all requests returned errors (0 tokens)
- **Time:** 2026-04-04 ~13:20 CEST
- n=1: 0/1, n=4: 0/4, n=8: 0/8 successful

### #12 — triton moe / triton attn / fi_cudnn fp4 / piecewise

- **Outcome:** startup_crash
- **Error:** Head + all 3 workers restarted (total=1 each)
- **Time:** 2026-04-04 11:20–11:28 UTC

### #13 — fi_cutlass moe / flashinfer attn / fi_cutlass fp4 / cuda_graph

- **Outcome:** startup_crash — **OOMKilled** (workers 1, 2, 3)
- **Time:** 2026-04-04 11:28–11:34 UTC

### #14 — fi_cutlass moe / flashinfer attn / fi_cutlass fp4 / no-cuda-graph

- **Outcome:** infer_error — server stable, all requests returned errors (0 tokens)
- **Time:** 2026-04-04 ~13:41 CEST
- n=1: 0/1, n=4: 0/4, n=8: 0/8 successful

### #15 — fi_cutlass moe / flashinfer attn / fi_cutlass fp4 / piecewise

- **Outcome:** startup_crash — **OOMKilled** (workers 1, 2, 3)
- **Time:** 2026-04-04 11:41–11:48 UTC

### #16 — fi_cutlass moe / triton attn / fi_cutlass fp4 / cuda_graph

- **Outcome:** startup_crash — **OOMKilled** (workers 1, 2)
- **Time:** 2026-04-04 11:48–11:54 UTC
