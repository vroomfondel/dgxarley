# SGLang Test Log — GLM 4.7 NVFP4, 4 Nodes, v0.5.9dev2

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
| Image | `scitrera/dgx-spark-sglang:0.5.9-dev2-acab24a7-t5`         |
| Model | `nvidia/GLM-4.7-NVFP4`                    |

Previous test series with 3 nodes: see `TESTLOG_nv580.142_sglang-0.5.10rc0_GLM-4.7-NVFP4_3n.md`.

---

## Configuration Matrix

All tests use: `tp=4, pp=1, ep=4, quantization=modelopt_fp4, kv_cache_dtype=fp8_e4m3, mem_fraction_static=0.60, disable_deep_gemm=true, context_length=196608, max_running_requests=32, schedule_policy=lpm, watchdog_timeout=3600, dist_timeout=1800` unless noted.

| # | nccl_transport | moe_runner | attention | fp4_gemm | dis_cuda_graph | dis_piecewise | pp_async | cuda_graph_max_bs | Stability | 1∥ tok/s | 4∥ tok/s | 8∥ tok/s |
|---|----------------|------------|-----------|----------|----------------|---------------|----------|-------------------|-----------|---------|---------|---------|
| 1 | socket | triton | flashinfer | fi_cutlass | false | true | 0 | 8 | **startup_crash** | — | — | — |
| 2 | socket | triton | flashinfer | fi_cutlass | true | true | 0 | — | **infer_error** | — | — | — |
| 3 | socket | triton | flashinfer | fi_cutlass | false | false | 0 | 8 | **deploy_failed** | — | — | — |
| 4 | socket | triton | triton | fi_cutlass | false | true | 0 | 8 | **startup_crash** | — | — | — |
| 5 | socket | triton | triton | fi_cutlass | true | true | 0 | — | **deploy_failed** | — | — | — |
| 6 | socket | triton | triton | fi_cutlass | false | false | 0 | 8 | pending | — | — | — |
| 7 | socket | triton | flashinfer | fi_cudnn | false | true | 0 | 8 | pending | — | — | — |
| 8 | socket | triton | flashinfer | fi_cudnn | true | true | 0 | — | pending | — | — | — |
| 9 | socket | triton | flashinfer | fi_cudnn | false | false | 0 | 8 | pending | — | — | — |
| 10 | socket | triton | triton | fi_cudnn | false | true | 0 | 8 | pending | — | — | — |
| 11 | socket | triton | triton | fi_cudnn | true | true | 0 | — | pending | — | — | — |
| 12 | socket | triton | triton | fi_cudnn | false | false | 0 | 8 | pending | — | — | — |
| 13 | socket | fi_cutlass | flashinfer | fi_cutlass | false | true | 0 | 8 | **startup_crash** | — | — | — |
| 14 | socket | fi_cutlass | flashinfer | fi_cutlass | true | true | 0 | — | **bench_crash** | — | — | — |
| 15 | socket | fi_cutlass | flashinfer | fi_cutlass | false | false | 0 | 8 | **startup_crash** | — | — | — |
| 16 | socket | fi_cutlass | triton | fi_cutlass | false | true | 0 | 8 | **startup_crash** | — | — | — |
| 17 | socket | fi_cutlass | triton | fi_cutlass | true | true | 0 | — | **infer_error** | — | — | — |
| 18 | socket | fi_cutlass | triton | fi_cutlass | false | false | 0 | 8 | **startup_crash** | — | — | — |
| 19 | socket | fi_cutlass | flashinfer | fi_cudnn | false | true | 0 | 8 | pending | — | — | — |
| 20 | socket | fi_cutlass | flashinfer | fi_cudnn | true | true | 0 | — | pending | — | — | — |
| 21 | socket | fi_cutlass | flashinfer | fi_cudnn | false | false | 0 | 8 | pending | — | — | — |
| 22 | socket | fi_cutlass | triton | fi_cudnn | false | true | 0 | 8 | pending | — | — | — |
| 23 | socket | fi_cutlass | triton | fi_cudnn | true | true | 0 | — | pending | — | — | — |
| 24 | socket | fi_cutlass | triton | fi_cudnn | false | false | 0 | 8 | pending | — | — | — |
| 25 | socket | cutlass | flashinfer | fi_cutlass | false | true | 0 | 8 | **startup_crash** | — | — | — |
| 26 | socket | cutlass | flashinfer | fi_cutlass | true | true | 0 | — | **infer_error** | — | — | — |
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

> **Note:** `mem_fraction_static` varies per test batch (effective profile values):
> - #1: `0.60` (1st run), `0.80` (2nd run), #2: `0.80`, #3: `0.70`, #4: `0.50`, #5: `0.70`, #13–18: `0.80`, #25–26: `0.80`
> All other parameters match the matrix header defaults unless noted.

### #1 — triton moe / flashinfer attn / fi_cutlass fp4 / cuda_graph

- **Outcome:** startup_crash (both runs)
- **1st run:** mem_frac=0.60, 2026-04-04 08:59–09:06 UTC — head + all 3 workers restarted
- **2nd run:** mem_frac=0.80, 2026-04-04 09:26–09:34 UTC — head restarted (cutlass_moe_fp4 device-side assert at nvfp4_blockwise_moe.cuh:78 during CUDA graph capture)

### #2 — triton moe / flashinfer attn / fi_cutlass fp4 / no-cuda-graph

- **Outcome:** infer_error — server started (stable), all requests returned errors
- **Time:** 2026-04-04 09:34–09:42 UTC
- **mem_fraction_static:** 0.80
- n=1: 0/1 successful (1 error)
- n=4: 0/4 successful (4 errors)
- n=8: 0/8 successful (8 errors)
- All requests: 0 output_tokens, null ttft, null tokens_per_sec

### #3 — triton moe / flashinfer attn / fi_cutlass fp4 / piecewise

- **Outcome:** deploy_failed (Ansible status: canceled)
- **Time:** 2026-04-04 09:42:03–09:42:44 UTC
- **mem_fraction_static:** 0.70

### #4 — triton moe / triton attn / fi_cutlass fp4 / cuda_graph

- **Outcome:** startup_crash
- **Error:** Head + all 3 workers restarted (total=1 each)
- **Time:** 2026-04-04 09:18–09:24 UTC
- **mem_fraction_static:** 0.50

### #5 — triton moe / triton attn / fi_cutlass fp4 / no-cuda-graph

- **Outcome:** deploy_failed (Ansible status: canceled)
- **Time:** 2026-04-04 09:42:44–09:42:49 UTC
- **mem_fraction_static:** 0.70

### #13 — fi_cutlass moe / flashinfer attn / fi_cutlass fp4 / cuda_graph

- **Outcome:** startup_crash
- **Error:** Pod sglang-worker-2 restarted (total=1)
- **Time:** 2026-04-04 08:02–08:08 UTC

### #14 — fi_cutlass moe / flashinfer attn / fi_cutlass fp4 / no-cuda-graph

- **Outcome:** bench_crash
- **Error:** Smoke test crash (n=8, max_tokens=128) — workers 1, 2, 3 all restarted
- **Time:** 2026-04-04 08:09–08:16 UTC

### #15 — fi_cutlass moe / flashinfer attn / fi_cutlass fp4 / piecewise

- **Outcome:** startup_crash
- **Error:** Head + all 3 workers restarted (total=1 each)
- **Time:** 2026-04-04 08:17–08:23 UTC

### #16 — fi_cutlass moe / triton attn / fi_cutlass fp4 / cuda_graph

- **Outcome:** startup_crash
- **Error:** Pod sglang-worker-2 restarted (total=1)
- **Time:** 2026-04-04 08:24–08:29 UTC

### #17 — fi_cutlass moe / triton attn / fi_cutlass fp4 / no-cuda-graph

- **Outcome:** infer_error — server started, all requests returned errors
- **Time:** 2026-04-04 ~10:36 CEST
- n=1: 0/1 successful (1 error)
- n=4: 0/4 successful (4 errors)
- n=8: 0/8 successful (8 errors)
- All requests: 0 output_tokens, null ttft, null tokens_per_sec

### #18 — fi_cutlass moe / triton attn / fi_cutlass fp4 / piecewise

- **Outcome:** startup_crash
- **Error:** Head + all 3 workers restarted (worker-3 restarted twice)
- **Time:** 2026-04-04 08:36–08:43 UTC
- **mem_fraction_static:** 0.80

### #25 — cutlass moe / flashinfer attn / fi_cutlass fp4 / cuda_graph

- **Outcome:** startup_crash
- **Error:** Head + all 3 workers restarted (total=1 each)
- **Time:** 2026-04-04 08:44–08:51 UTC
- **mem_fraction_static:** 0.80

### #26 — cutlass moe / flashinfer attn / fi_cutlass fp4 / no-cuda-graph

- **Outcome:** infer_error — server started (stable), all requests returned errors
- **Time:** 2026-04-04 10:58 CEST
- **mem_fraction_static:** 0.80
- n=1: 0/1 successful (1 error)
- n=4: 0/4 successful (4 errors)
- n=8: 0/8 successful (8 errors)
- All requests: 0 output_tokens, null ttft, null tokens_per_sec
