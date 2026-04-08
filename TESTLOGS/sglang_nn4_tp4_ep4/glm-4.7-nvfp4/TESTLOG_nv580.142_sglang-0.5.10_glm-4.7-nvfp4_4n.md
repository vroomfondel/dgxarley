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
| 17 | socket | fi_cutlass | triton | fi_cutlass | true | true | 0 | — | **partial** | 8.4 | 20.8 | — |
| 18 | socket | fi_cutlass | triton | fi_cutlass | false | false | 0 | 8 | *skip* | — | — | — |
| 19 | socket | fi_cutlass | flashinfer | fi_cudnn | false | true | 0 | 8 | *skip* | — | — | — |
| 20 | socket | fi_cutlass | flashinfer | fi_cudnn | true | true | 0 | — | **infer_error** | — | — | — |
| 21 | socket | fi_cutlass | flashinfer | fi_cudnn | false | false | 0 | 8 | *skip* | — | — | — |
| 22 | socket | fi_cutlass | triton | fi_cudnn | false | true | 0 | 8 | *skip* | — | — | — |
| 23 | socket | fi_cutlass | triton | fi_cudnn | true | true | 0 | — | **infer_error** | — | — | — |
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
| 37 | socket | fi_cutlass | triton | fi_cudnn | true | true | 0 | — | **startup_crash** | — | — | — |

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

### First run (speculative_enabled=true by mistake)

Tests 14, 17, 20 ran with `speculative_enabled: true` inherited from the GLM-4.7 model profile — matrix patches didn't override it. Results were invalid:
- Test 14: TMA descriptor crash (Xid 13) after 442 tokens — possibly speculative codepath
- Test 17: Xid 13 on Worker-1 after ~700 tokens — possibly speculative codepath
- Test 20: `cuDNN is not available` crash — speculative/EAGLE codepath doesn't load cuDNN

All tests below are from the **re-run with `speculative_enabled: false`**.

---

### #14 — fi_cutlass MoE / flashinfer attn / fi_cutlass fp4 / no-cuda-graph

- **Outcome:** bench_crash — Worker-3 restart
- **Time:** 2026-04-08 10:17 CEST
- **n=1:** aborted (7.72 tok/s, TTFT 16.0s, 0 output tokens)
- **n=4/n=8:** not reached

**vs rc0:** infer_error (0 tokens). v0.5.10 gets further (starts generating) but still crashes. `fi_cutlass` fp4 + `flashinfer` attn is unstable on SM121.

### #17 — fi_cutlass MoE / triton attn / fi_cutlass fp4 / no-cuda-graph

- **Outcome:** **partial** — n=1 and n=4 stable, n=8 crashed. Best result in this matrix.
- **Time:** 2026-04-08 10:28 CEST
- **n=1:** **8.4 tok/s**, 3072 tokens, TTFT 6.4s, finish=length
- **n=4:** **4/4 success**, 5.2 tok/s per-req, **20.8 tok/s peak**, TTFT 2.4–3.0s
- **n=8:** 0/8 — all aborted (2.1–2.8 tok/s before abort)

**vs rc0:** OOMKilled on rc0. On v0.5.10, n=1 and n=4 fully work. n=8 overloads and crashes. `fi_cutlass` MoE + `triton` attn + `fi_cutlass` fp4 is the new best combo (replacing `fi_cudnn` fp4 which regressed).

### #20 — fi_cutlass MoE / flashinfer attn / fi_cudnn fp4 / no-cuda-graph

- **Outcome:** infer_error — server stable, all requests returned errors (0 tokens)
- **Time:** 2026-04-08 10:44 CEST
- **n=1:** 0/1 error, **n=4:** 0/4, **n=8:** 0/8

**vs rc0:** On rc0, n=1 worked (7.95 tok/s). **Regression in v0.5.10** — `fi_cudnn` fp4_gemm backend produces 0 tokens.

### #23 — fi_cutlass MoE / triton attn / fi_cudnn fp4 / no-cuda-graph (rc0 WINNER)

- **Outcome:** infer_error — server stable, all requests returned errors (0 tokens)
- **Time:** 2026-04-08 10:49 CEST
- **n=1:** 0/1 error, **n=4:** 0/4, **n=8:** 0/8

**vs rc0:** **MAJOR REGRESSION.** rc0 WINNER (8.06 / 21.94 / 30.01 tok/s) now produces 0 tokens. `flashinfer_cudnn` fp4_gemm is completely broken in v0.5.10 for GLM-4.7 on SM121. Both test 20 and 23 use `fi_cudnn` — confirms this is an fp4_gemm backend issue, not attention-related.

### #37 — #23 winner + MTP speculative decoding (NEXTN)

- **Outcome:** startup_crash — massive crash loop (52 restarts on all pods)
- **Time:** 2026-04-08 17:46 CEST
- **Note:** Even without the `fi_cudnn` regression, MTP speculative on this model is unstable.

---

## Summary — v0.5.10 vs v0.5.10rc0

| Change | Detail |
|--------|--------|
| **`fi_cudnn` fp4_gemm REGRESSION** | Tests 20, 23: 0 tokens on v0.5.10 (rc0: 8.06/30.01 tok/s). The `flashinfer_cudnn` FP4 GEMM backend is broken. |
| **`fi_cutlass` fp4_gemm IMPROVED** | Test 17: n=1+n=4 now work (rc0: OOMKilled). `flashinfer_cutlass` FP4 GEMM survives with `jit_max_jobs=4`. |
| **New best config** | Test 17: fi_cutlass MoE + triton attn + fi_cutlass fp4 + eager → 8.4 / 20.8 tok/s (n=1/n=4). n=8 unstable. |
| **Speculative (NEXTN)** | Test 37: startup crash loop. Not viable on this model/version. |
