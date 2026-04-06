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

All tests use: `tp=1, pp=4, ep=1, quantization=modelopt_fp4, kv_cache_dtype=fp8_e4m3, mem_fraction_static=0.60, disable_deep_gemm=true, context_length=196608, max_running_requests=32, schedule_policy=lpm, watchdog_timeout=3600, dist_timeout=1800` unless noted.

| # | nccl_transport | moe_runner | attention | fp4_gemm | dis_cuda_graph | dis_piecewise | pp_async | cuda_graph_max_bs | Stability | 1∥ tok/s | 4∥ tok/s | 8∥ tok/s |
|---|----------------|------------|-----------|----------|----------------|---------------|----------|-------------------|-----------|---------|---------|---------|
| 1 | socket | triton | flashinfer | fi_cutlass | false | true | 0 | 8 | pending | — | — | — |
| 2 | socket | triton | flashinfer | fi_cutlass | true | true | 0 | — | pending | — | — | — |
| 3 | socket | triton | flashinfer | fi_cutlass | false | false | 0 | 8 | pending | — | — | — |
| 4 | socket | triton | triton | fi_cutlass | false | true | 0 | 8 | pending | — | — | — |
| 5 | socket | triton | triton | fi_cutlass | true | true | 0 | — | pending | — | — | — |
| 6 | socket | triton | triton | fi_cutlass | false | false | 0 | 8 | pending | — | — | — |
| 7 | socket | triton | flashinfer | fi_cudnn | false | true | 0 | 8 | pending | — | — | — |
| 8 | socket | triton | flashinfer | fi_cudnn | true | true | 0 | — | pending | — | — | — |
| 9 | socket | triton | flashinfer | fi_cudnn | false | false | 0 | 8 | pending | — | — | — |
| 10 | socket | triton | triton | fi_cudnn | false | true | 0 | 8 | pending | — | — | — |
| 11 | socket | triton | triton | fi_cudnn | true | true | 0 | — | pending | — | — | — |
| 12 | socket | triton | triton | fi_cudnn | false | false | 0 | 8 | pending | — | — | — |
| 13 | socket | fi_cutlass | flashinfer | fi_cutlass | false | true | 0 | 8 | pending | — | — | — |
| 14 | socket | fi_cutlass | flashinfer | fi_cutlass | true | true | 0 | — | pending | — | — | — |
| 15 | socket | fi_cutlass | flashinfer | fi_cutlass | false | false | 0 | 8 | pending | — | — | — |
| 16 | socket | fi_cutlass | triton | fi_cutlass | false | true | 0 | 8 | pending | — | — | — |
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

