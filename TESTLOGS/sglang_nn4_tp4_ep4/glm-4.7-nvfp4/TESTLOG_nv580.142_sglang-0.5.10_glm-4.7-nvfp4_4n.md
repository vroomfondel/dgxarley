# SGLang Test Log — GLM 4.7 NVFP4, 4 Nodes, v0.5.10

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
| Image | `scitrera/dgx-spark-sglang:0.5.10` |
| Model | `nvidia/GLM-4.7-NVFP4` |

Previous test series: v0.5.10rc0 (`TESTLOG_nv580.142_sglang-0.5.10rc0_glm-4.7-nvfp4_4n.md`).

---

## Configuration Matrix

All tests use: `tp=4, pp=1, ep=4, quantization=modelopt_fp4, kv_cache_dtype=fp8_e4m3, mem_fraction_static=0.60, disable_deep_gemm=true, context_length=196608, max_running_requests=32, schedule_policy=lpm, watchdog_timeout=3600, dist_timeout=1800` unless noted.

32 of 37 tests skipped — known failures from v0.5.10rc0 with root causes unchanged in v0.5.10:
- Tests 1–12 (triton MoE): `cutlass_moe_fp4` fallback crashes (CUDA graph: `nvfp4_blockwise_moe.cuh` device-side assert; eager: bench crash)
- Tests 13, 15, 16, 18, 19, 21, 22, 24 (fi_cutlass MoE + CUDA graphs): OOMKilled during graph capture
- Tests 25–36 (cutlass MoE): worker crashes on NVFP4+EP=4

| # | nccl_transport | moe_runner | attention | fp4_gemm | dis_cuda_graph | dis_piecewise | pp_async | cuda_graph_max_bs | Stability | 1∥ tok/s | 4∥ tok/s | 8∥ tok/s |
|---|----------------|------------|-----------|----------|----------------|---------------|----------|-------------------|-----------|---------|---------|---------|
| 1 | socket | triton | flashinfer | fi_cutlass | false | true | 0 | 8 | *skip* | — | — | — |
| 2 | socket | triton | flashinfer | fi_cutlass | true | true | 0 | — | *skip* | — | — | — |
| 3 | socket | triton | flashinfer | fi_cutlass | false | false | 0 | 8 | *skip* | — | — | — |
| 4 | socket | triton | triton | fi_cutlass | false | true | 0 | 8 | *skip* | — | — | — |
| 5 | socket | triton | triton | fi_cutlass | true | true | 0 | — | *skip* | — | — | — |
| 6 | socket | triton | triton | fi_cutlass | false | false | 0 | 8 | *skip* | — | — | — |
| 7 | socket | triton | flashinfer | fi_cudnn | false | true | 0 | 8 | *skip* | — | — | — |
| 8 | socket | triton | flashinfer | fi_cudnn | true | true | 0 | — | *skip* | — | — | — |
| 9 | socket | triton | flashinfer | fi_cudnn | false | false | 0 | 8 | *skip* | — | — | — |
| 10 | socket | triton | triton | fi_cudnn | false | true | 0 | 8 | *skip* | — | — | — |
| 11 | socket | triton | triton | fi_cudnn | true | true | 0 | — | *skip* | — | — | — |
| 12 | socket | triton | triton | fi_cudnn | false | false | 0 | 8 | *skip* | — | — | — |
| 13 | socket | fi_cutlass | flashinfer | fi_cutlass | false | true | 0 | 8 | *skip* | — | — | — |
| 14 | socket | fi_cutlass | flashinfer | fi_cutlass | true | true | 0 | — | **bench_crash** | — | — | — |
| 15 | socket | fi_cutlass | flashinfer | fi_cutlass | false | false | 0 | 8 | *skip* | — | — | — |
| 16 | socket | fi_cutlass | triton | fi_cutlass | false | true | 0 | 8 | *skip* | — | — | — |
| 17 | socket | fi_cutlass | triton | fi_cutlass | true | true | 0 | — | **bench_crash** | — | — | — |
| 18 | socket | fi_cutlass | triton | fi_cutlass | false | false | 0 | 8 | *skip* | — | — | — |
| 19 | socket | fi_cutlass | flashinfer | fi_cudnn | false | true | 0 | 8 | *skip* | — | — | — |
| 20 | socket | fi_cutlass | flashinfer | fi_cudnn | true | true | 0 | — | *pending* | — | — | — |
| 21 | socket | fi_cutlass | flashinfer | fi_cudnn | false | false | 0 | 8 | *skip* | — | — | — |
| 22 | socket | fi_cutlass | triton | fi_cudnn | false | true | 0 | 8 | *skip* | — | — | — |
| 23 | socket | fi_cutlass | triton | fi_cudnn | true | true | 0 | — | *pending* | — | — | — |
| 24 | socket | fi_cutlass | triton | fi_cudnn | false | false | 0 | 8 | *skip* | — | — | — |
| 25 | socket | cutlass | flashinfer | fi_cutlass | false | true | 0 | 8 | *skip* | — | — | — |
| 26 | socket | cutlass | flashinfer | fi_cutlass | true | true | 0 | — | *skip* | — | — | — |
| 27 | socket | cutlass | flashinfer | fi_cutlass | false | false | 0 | 8 | *skip* | — | — | — |
| 28 | socket | cutlass | triton | fi_cutlass | false | true | 0 | 8 | *skip* | — | — | — |
| 29 | socket | cutlass | triton | fi_cutlass | true | true | 0 | — | *skip* | — | — | — |
| 30 | socket | cutlass | triton | fi_cutlass | false | false | 0 | 8 | *skip* | — | — | — |
| 31 | socket | cutlass | flashinfer | fi_cudnn | false | true | 0 | 8 | *skip* | — | — | — |
| 32 | socket | cutlass | flashinfer | fi_cudnn | true | true | 0 | — | *skip* | — | — | — |
| 33 | socket | cutlass | flashinfer | fi_cudnn | false | false | 0 | 8 | *skip* | — | — | — |
| 34 | socket | cutlass | triton | fi_cudnn | false | true | 0 | 8 | *skip* | — | — | — |
| 35 | socket | cutlass | triton | fi_cudnn | true | true | 0 | — | *skip* | — | — | — |
| 36 | socket | cutlass | triton | fi_cudnn | false | false | 0 | 8 | *skip* | — | — | — |
| 37 | socket | fi_cutlass | triton | fi_cudnn | true | true | 0 | — | *pending* | — | — | — |

> **#37** = #23 winner config + MTP speculative decoding (NEXTN, 3 steps, 4 draft tokens)

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

### #14 — fi_cutlass MoE / flashinfer attn / fi_cutlass fp4 / no-cuda-graph

- **Outcome:** bench_crash — `cudaErrorIllegalInstruction` (Xid 13) in TMA descriptor initialization
- **Time:** 2026-04-08 09:49 CEST
- **n=1:** error after 442 tokens (10.53 tok/s, TTFT 1.21s) — generated tokens then TMA crash killed head
- **n=4:** 0/4 — instant errors (server dead after n=1 crash)
- **n=8:** 0/8 — instant errors

**Loki crash trace (07:49:26 UTC):**
```
HEAD:    Error: Failed to initialize the TMA descriptor 715 (6×)
HEAD:    CUDA error: an illegal instruction was encountered
HEAD:    → flashinfer/fused_moe/core.py:490 cutlass_fused_moe
HEAD:    Fatal Python error: Aborted (exit code -6)
WORKER1: NCCL error: remote process exited (SeqNum=39117, ALLREDUCE)
WORKER2: TCPStore: Failed to recv, got 0 bytes (head gone)
WORKER3: TCPStore: Broken pipe
HEAD:    Subprocess scheduler_0 crashed → SIGQUIT cleanup
```

**vs rc0:** On rc0, test 14 returned 0 tokens (infer_error). On v0.5.10, it generates 442 tokens before hitting the TMA illegal instruction. Something changed in the FlashInfer CUTLASS MoE codepath that makes it partially work — the TMA descriptor failure occurs mid-inference, not at startup.

### #17 — fi_cutlass MoE / triton attn / fi_cutlass fp4 / no-cuda-graph

- **Outcome:** bench_crash — `cudaErrorIllegalInstruction` (Xid 13) on Worker-1 (spark2)
- **Time:** 2026-04-08 09:56 CEST
- **n=1:** aborted after 84s (8.29 tok/s, TTFT 12.5s) — generated tokens then Xid 13 killed Worker-1
- **n=4/n=8:** not reached (Worker-1 restart detected by matrix runner)

**Loki crash trace (07:56:07 UTC):**
```
W1 (spark2): cudaErrorIllegalInstruction — NCCL watchdog terminated, Fatal Python error: Aborted
W2 (spark3): NCCL error: remote process exited (SeqNum=55587, ALLREDUCE)
W3 (spark4): NCCL dump signal from rank 2
HD (spark1): NCCL error: remote process exited
```

**vs rc0:** OOMKilled on rc0 (jit_max_jobs=16 exhausted memory). On v0.5.10 with jit_max_jobs=4, memory survives but the FlashInfer CUTLASS MoE kernel hits Xid 13 after ~700 tokens. Same root cause as #14 — `fi_cutlass` MoE + `fi_cutlass` fp4 is unstable on SM121.

### #20 — fi_cutlass MoE / flashinfer attn / fi_cudnn fp4 / no-cuda-graph

*pending*

### #23 — fi_cutlass MoE / triton attn / fi_cudnn fp4 / no-cuda-graph

*pending* — **rc0 WINNER** (8.06 / 21.94 / 30.01 tok/s). Expecting stable on v0.5.10.

### #37 — #23 winner + MTP speculative decoding (NEXTN)

*pending*
