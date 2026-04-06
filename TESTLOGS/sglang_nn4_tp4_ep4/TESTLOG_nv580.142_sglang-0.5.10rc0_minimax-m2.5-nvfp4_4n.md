# SGLang Test Log — MiniMax M2.5 NVFP4, 4 Nodes, v0.5.10rc0

## Environment

| Component | Value |
|-----------|-------|
| GPU | NVIDIA GB10 (SM121/Blackwell), 128 GB per node |
| Driver | 580.142 |
| CUDA | 13.0 |
| Kernel | 6.17.0-1014-nvidia |
| OS | Ubuntu 24.04.4 LTS (aarch64) |
| K3s | v1.35.3+k3s1 |
| Nodes | spark1, spark2, spark3, spark4 (1 GPU each) |
| Image | `scitrera/dgx-spark-sglang:0.5.10rc0` |
| Model | `nvidia/MiniMax-M2.5-NVFP4` |

Previous test series with 3 nodes: see `TESTLOG_nv580.142_sglang-0.5.10rc0_minimax-m2.5-nvfp4_3n.md`.

---

## Configuration Matrix

All tests use: `tp=4, pp=1, ep=4, quantization=modelopt_fp4, kv_cache_dtype=fp8_e4m3, mem_fraction_static=0.70, disable_deep_gemm=true, context_length=196608, max_running_requests=32, schedule_policy=lpm, watchdog_timeout=3600, dist_timeout=1800` unless noted.

| # | nccl_transport | moe_runner | attention | fp4_gemm | dis_cuda_graph | dis_piecewise | pp_async | cuda_graph_max_bs | Stability | 1∥ tok/s | 4∥ tok/s | 8∥ tok/s |
|---|----------------|------------|-----------|----------|----------------|---------------|----------|-------------------|-----------|---------|---------|---------|
| 1 | socket | triton | flashinfer | fi_cutlass | false | true | 0 | 16 | FAIL (startup crash) worker restart | — | — | — |
| 2 | socket | triton | flashinfer | fi_cutlass | true | true | 0 | — | FAIL (bench crash) all requests errored | — | — | — |
| 3 | socket | triton | flashinfer | fi_cutlass | false | false | 0 | 16 | FAIL (startup crash) CUDA graph: nvfp4_blockwise_moe.cu | — | — | — |
| 4 | socket | triton | triton | fi_cutlass | false | true | 0 | 16 | FAIL (startup crash) CUDA graph: nvfp4_blockwise_moe.cu | — | — | — |
| 5 | socket | triton | triton | fi_cutlass | true | true | 0 | — | FAIL (bench crash) worker restart during bench | — | — | — |
| 6 | socket | triton | triton | fi_cutlass | false | false | 0 | 16 | FAIL (startup crash) CUDA graph: nvfp4_blockwise_moe.cu | — | — | — |
| 7 | socket | triton | flashinfer | fi_cudnn | false | true | 0 | 16 | FAIL (startup crash) CUDA graph: nvfp4_blockwise_moe.cu | — | — | — |
| 8 | socket | triton | flashinfer | fi_cudnn | true | true | 0 | — | FAIL (bench crash) worker restart during bench | — | — | — |
| 9 | socket | triton | flashinfer | fi_cudnn | false | false | 0 | 16 | FAIL (startup crash) CUDA graph: nvfp4_blockwise_moe.cu | — | — | — |
| 10 | socket | triton | triton | fi_cudnn | false | true | 0 | 16 | FAIL (startup crash) CUDA graph: nvfp4_blockwise_moe.cu | — | — | — |
| 11 | socket | triton | triton | fi_cudnn | true | true | 0 | — | FAIL (bench crash) all requests errored | — | — | — |
| 12 | socket | triton | triton | fi_cudnn | false | false | 0 | 16 | FAIL (startup crash) CUDA graph: nvfp4_blockwise_moe.cu | — | — | — |
| 13 | socket | fi_cutlass | flashinfer | fi_cutlass | false | true | 0 | 16 | FAIL (startup crash) OOMKilled | — | — | — |
| 14 | socket | fi_cutlass | flashinfer | fi_cutlass | true | true | 0 | — | FAIL (bench crash) all requests errored | — | — | — |
| 15 | socket | fi_cutlass | flashinfer | fi_cutlass | false | false | 0 | 16 | FAIL (startup crash) OOMKilled | — | — | — |
| 16 | socket | fi_cutlass | triton | fi_cutlass | false | true | 0 | 16 | FAIL (startup crash) OOMKilled | — | — | — |
| 17 | socket | fi_cutlass | triton | fi_cutlass | true | true | 0 | — | FAIL (bench crash) OOMKilled during bench | — | — | — |
| 18 | socket | fi_cutlass | triton | fi_cutlass | false | false | 0 | 16 | FAIL (startup crash) OOMKilled | — | — | — |
| 19 | socket | fi_cutlass | flashinfer | fi_cudnn | false | true | 0 | 16 | FAIL (startup crash) OOMKilled | — | — | — |
| 20 | socket | fi_cutlass | flashinfer | fi_cudnn | true | true | 0 | — | FAIL (bench crash) all requests errored | — | — | — |
| 21 | socket | fi_cutlass | flashinfer | fi_cudnn | false | false | 0 | 16 | FAIL (startup crash) OOMKilled | — | — | — |
| 22 | socket | fi_cutlass | triton | fi_cudnn | false | true | 0 | 16 | FAIL (startup crash) worker restart | — | — | — |
| 23 | socket | fi_cutlass | triton | fi_cudnn | true | true | 0 | — | FAIL (bench crash) all requests errored | — | — | — |
| 24 | socket | fi_cutlass | triton | fi_cudnn | false | false | 0 | 16 | FAIL (startup crash) OOMKilled | — | — | — |
| 25 | socket | cutlass | flashinfer | fi_cutlass | false | true | 0 | 8 | FAIL (startup crash) worker restart | — | — | — |
| 26 | socket | cutlass | flashinfer | fi_cutlass | true | true | 0 | — | FAIL (bench crash) worker restart during bench | — | — | — |
| 27 | socket | cutlass | flashinfer | fi_cutlass | false | false | 0 | 8 | FAIL (startup crash) worker restart | — | — | — |
| 28 | socket | cutlass | triton | fi_cutlass | false | true | 0 | 8 | FAIL (startup crash) worker restart | — | — | — |
| 29 | socket | cutlass | triton | fi_cutlass | true | true | 0 | — | not run | — | — | — |
| 30 | socket | cutlass | triton | fi_cutlass | false | false | 0 | 16 | not run | — | — | — |
| 31 | socket | cutlass | flashinfer | fi_cudnn | false | true | 0 | 16 | not run | — | — | — |
| 32 | socket | cutlass | flashinfer | fi_cudnn | true | true | 0 | — | not run | — | — | — |
| 33 | socket | cutlass | flashinfer | fi_cudnn | false | false | 0 | 16 | not run | — | — | — |
| 34 | socket | cutlass | triton | fi_cudnn | false | true | 0 | 16 | not run | — | — | — |
| 35 | socket | cutlass | triton | fi_cudnn | true | true | 0 | — | not run | — | — | — |
| 36 | socket | cutlass | triton | fi_cudnn | false | false | 0 | 16 | not run | — | — | — |

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

## Failure Analysis

### triton MoE runner: CUDA graph capture crashes (tests 3, 4, 6, 7, 9, 10, 12)

All `triton` MoE configurations with CUDA graphs enabled (`disable_cuda_graph=false`) crash during startup with:

```
Capture cuda graph failed: Runtime check failed at sglang/jit_kernel/csrc/moe/nvfp4_blockwise_moe.cu
```

The triton MoE runner falls back to `cutlass_moe_fp4` for NVFP4, which fails CUDA graph capture on 4-node EP=4. Eager mode (tests 2, 5, 8, 11) starts up but crashes during bench (worker pod restarts).

### fi_cutlass MoE runner: OOMKilled (tests 13, 15, 16, 17, 18, 19, 21, 22, 24)

`flashinfer_cutlass` MoE runner causes OOMKilled on one or more worker nodes during startup or early inference. With `mem_fraction_static=0.70` and 4-node EP=4, the fi_cutlass MoE kernel allocates more memory than available. Tests 14, 17, 20, 23 (no-cuda-graph variants) reached the bench phase but all requests errored (bench_crash).

### cutlass MoE runner: worker restart (tests 25, 26, 27, 28)

The `cutlass` MoE runner (direct, not flashinfer_cutlass) causes worker pod restarts during startup (startup_crash) or bench (bench_crash). These tests used `cuda_graph_max_bs=8` (reduced from 16). Test 26 (no-cuda-graph) reached bench but crashed with worker restarts.

### Tests 29–36: not run

Matrix runner did not reach tests 29–36 (cutlass MoE + triton/flashinfer attention + fi_cudnn fp4_gemm). Session ended after test 28.

---

## NEXTN Speculative Decoding

Attempted: `speculative_algo=NEXTN, speculative_num_steps=3, speculative_num_draft_tokens=4, speculative_draft_model_path=""`

**Result:** Startup crash — `AttributeError: 'MiniMaxM2ForCausalLM' object has no attribute 'set_embed_and_head'`

The model implements `get_embed_and_head` but is missing `set_embed_and_head`, which `eagle_worker.py` calls to share the target model's embedding/lm_head weights with the draft model. Every other NEXTN-capable model (DeepSeek, GLM, Llama) has this method. A monkey-patch has been added to `sglang_launch.sh`.

Additionally, NEXTN causes a second full weight load (~36 GB) for the MTP heads. With `mem_fraction_static=0.60`, the KV cache pool consumes too much memory, leaving insufficient room for the second load → OOM. Fix: reduce `mem_fraction_static` (e.g., 0.40) to shrink the KV cache budget.
