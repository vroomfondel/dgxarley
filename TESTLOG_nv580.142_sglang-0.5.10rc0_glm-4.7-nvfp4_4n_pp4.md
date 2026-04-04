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

All tests use: `tp=1, pp=4, ep=1, quantization=modelopt_fp4, kv_cache_dtype=fp8_e4m3, mem_fraction_static=0.60, disable_deep_gemm=true, context_length=196608, max_running_requests=32, schedule_policy=lpm, watchdog_timeout=3600, dist_timeout=1800` unless noted.

| # | nccl_transport | moe_runner | attention | fp4_gemm | dis_cuda_graph | dis_piecewise | pp_async | cuda_graph_max_bs | Stability | 1∥ tok/s | 4∥ tok/s | 8∥ tok/s |
|---|----------------|------------|-----------|----------|----------------|---------------|----------|-------------------|-----------|---------|---------|---------|
| 1 | socket | triton | flashinfer | fi_cutlass | false | true | 0 | 8 | **n=1 only** | 5.64 | — | — |
| 2 | socket | triton | flashinfer | fi_cutlass | true | true | 0 | — | **n=1 only** | 3.67 | — | — |
| 3 | socket | triton | flashinfer | fi_cutlass | false | false | 0 | 8 | **n=1 only** | 5.61 | — | — |
| 4 | socket | triton | triton | fi_cutlass | false | true | 0 | 8 | **startup_crash** | — | — | — |
| 5 | socket | triton | triton | fi_cutlass | true | true | 0 | — | **startup_crash** | — | — | — |
| 6 | socket | triton | triton | fi_cutlass | false | false | 0 | 8 | pending   | — | — | — |
| 7 | socket | triton | flashinfer | fi_cudnn | false | true | 0 | 8 | pending   | — | — | — |
| 8 | socket | triton | flashinfer | fi_cudnn | true | true | 0 | — | pending   | — | — | — |
| 9 | socket | triton | flashinfer | fi_cudnn | false | false | 0 | 8 | pending   | — | — | — |
| 10 | socket | triton | triton | fi_cudnn | false | true | 0 | 8 | pending   | — | — | — |
| 11 | socket | triton | triton | fi_cudnn | true | true | 0 | — | pending   | — | — | — |
| 12 | socket | triton | triton | fi_cudnn | false | false | 0 | 8 | pending   | — | — | — |
| 13 | socket | fi_cutlass | flashinfer | fi_cutlass | false | true | 0 | 8 | pending   | — | — | — |
| 14 | socket | fi_cutlass | flashinfer | fi_cutlass | true | true | 0 | — | pending   | — | — | — |
| 15 | socket | fi_cutlass | flashinfer | fi_cutlass | false | false | 0 | 8 | pending   | — | — | — |
| 16 | socket | fi_cutlass | triton | fi_cutlass | false | true | 0 | 8 | pending   | — | — | — |
| 17 | socket | fi_cutlass | triton | fi_cutlass | true | true | 0 | — | pending   | — | — | — |
| 18 | socket | fi_cutlass | triton | fi_cutlass | false | false | 0 | 8 | pending   | — | — | — |
| 19 | socket | fi_cutlass | flashinfer | fi_cudnn | false | true | 0 | 8 | pending   | — | — | — |
| 20 | socket | fi_cutlass | flashinfer | fi_cudnn | true | true | 0 | — | pending   | — | — | — |
| 21 | socket | fi_cutlass | flashinfer | fi_cudnn | false | false | 0 | 8 | pending   | — | — | — |
| 22 | socket | fi_cutlass | triton | fi_cudnn | false | true | 0 | 8 | pending   | — | — | — |
| 23 | socket | fi_cutlass | triton | fi_cudnn | true | true | 0 | — | pending   | — | — | — |
| 24 | socket | fi_cutlass | triton | fi_cudnn | false | false | 0 | 8 | pending   | — | — | — |
| 25 | socket | cutlass | flashinfer | fi_cutlass | false | true | 0 | 8 | pending   | — | — | — |
| 26 | socket | cutlass | flashinfer | fi_cutlass | true | true | 0 | — | pending   | — | — | — |
| 27 | socket | cutlass | flashinfer | fi_cutlass | false | false | 0 | 8 | pending   | — | — | — |
| 28 | socket | cutlass | triton | fi_cutlass | false | true | 0 | 8 | pending   | — | — | — |
| 29 | socket | cutlass | triton | fi_cutlass | true | true | 0 | — | pending   | — | — | — |
| 30 | socket | cutlass | triton | fi_cutlass | false | false | 0 | 8 | pending   | — | — | — |
| 31 | socket | cutlass | flashinfer | fi_cudnn | false | true | 0 | 8 | pending   | — | — | — |
| 32 | socket | cutlass | flashinfer | fi_cudnn | true | true | 0 | — | pending   | — | — | — |
| 33 | socket | cutlass | flashinfer | fi_cudnn | false | false | 0 | 8 | pending   | — | — | — |
| 34 | socket | cutlass | triton | fi_cudnn | false | true | 0 | 8 | pending   | — | — | — |
| 35 | socket | cutlass | triton | fi_cudnn | true | true | 0 | — | pending   | — | — | — |
| 36 | socket | cutlass | triton | fi_cudnn | false | false | 0 | 8 | pending   | — | — | — |

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

### #1 — triton moe / flashinfer attn / fi_cutlass fp4 / cuda_graph

- **Outcome:** n=1 success, n=4+ crash (FlashInfer `merge_state` invalid argument)
- **n=1:** 5.64 tok/s, 2891 tokens (1644 think + 1472 content), TTFT 1.4s, finish=stop
- **n=4:** 0/4 errors (FlashInfer cascade merge crash)
- **n=8:** 0/8 errors (same)

### #2 — triton moe / flashinfer attn / fi_cutlass fp4 / no-cuda-graph

- **Outcome:** n=1 success (slow), n=4+ crash
- **n=1:** 3.67 tok/s, 2990 tokens, **TTFT 250.6s** (extreme first-token latency — no cuda graph means no cached computation paths)
- **n=4:** 0/4 errors (0 tokens, instant)

### #3 — triton moe / flashinfer attn / fi_cutlass fp4 / piecewise

- **Outcome:** n=1 success, n=4+ crash
- **n=1:** 5.61 tok/s, 3916 tokens (1333 think + 2328 content), TTFT 1.3s, finish=stop
- **n=4:** 0/4 errors (0 tokens, instant — FlashInfer merge_state crash)

### #4 — triton moe / triton attn / fi_cutlass fp4 / cuda_graph

- **Outcome:** startup_crash
- **Error:** Workers 2, 3 restarted
- **Time:** 2026-04-04 17:30–17:35 UTC

### #5 — triton moe / triton attn / fi_cutlass fp4 / no-cuda-graph

- **Outcome:** startup_crash
- **Error:** Head + all 3 workers restarted
- **Time:** 2026-04-04 17:36–17:42 UTC
- **Note:** `triton` attention crashes on PP=4 even without cuda graphs — only `flashinfer` attention works for PP mode
