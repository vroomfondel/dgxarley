# SGLang Test Log — GLM-4.7 FP8, 4 Nodes, v0.5.10rc0

## Environment

| Component | Value                                          |
|-----------|-------------------------------------------------|
| GPU | NVIDIA GB10 (SM121/Blackwell), 128 GB per node |
| Driver | 580.142                                        |
| CUDA | 13.0                                           |
| Kernel | 6.17.0-1014-nvidia                             |
| OS | Ubuntu 24.04.4 LTS (aarch64)                   |
| K3s | v1.35.3+k3s1                                   |
| Nodes | spark1, spark2, spark3, spark4 (1 GPU each)    |
| Image | `scitrera/dgx-spark-sglang:0.5.10rc0`          |
| Model | `zai-org/GLM-4.7-FP8`                          |

## Result: All configurations crash at startup

GLM-4.7-FP8 (160 experts, TP=4, EP=4) fails to start on v0.5.10rc0 across **all 27 tested configurations**. No configuration reached a healthy serving state.

**Failure patterns:**

- **triton MoE** (cases #1–12): ~10 min per attempt (weight loading completes, crash during CUDA graph capture or init). Worker-1 crashes first (single worker), or head + all workers crash together.
- **fi_cutlass MoE** (cases #13–24): ~1–2 min per attempt (crashes very early, before weight loading completes — likely FlashInfer MoE JIT compilation failure on SM121 with FP8).
- **cutlass MoE** (cases #25–27): ~1–2 min per attempt (same rapid crash pattern as fi_cutlass).

**Conclusion:** FP8 quantization on GLM-4.7 is not supported by SGLang v0.5.10rc0 on SM121/Blackwell (DGX Spark). The NVFP4 variant (`nvidia/GLM-4.7-NVFP4`) does work — see `TESTLOG_nv580.142_sglang-0.5.10rc0_glm-4.7-nvfp4_4n.md`.

---

## Configuration Matrix

All tests use: `tp=4, pp=1, ep=4, kv_cache_dtype=fp8_e4m3, mem_fraction_static=0.60, disable_deep_gemm=true, context_length=131072, max_running_requests=32, schedule_policy=lpm, watchdog_timeout=3600, dist_timeout=1800, jit_max_jobs=4` unless noted.

| # | nccl_transport | moe_runner | attention | fp8_gemm | dis_cuda_graph | dis_piecewise | pp_async | cuda_graph_max_bs | Stability | 1∥ tok/s | 4∥ tok/s | 8∥ tok/s |
|---|----------------|------------|-----------|----------|----------------|---------------|----------|-------------------|-----------|---------|---------|---------|
| 1 | socket | triton | flashinfer | cutlass | false | true | 0 | 8 | **startup_crash** | — | — | — |
| 2 | socket | triton | flashinfer | cutlass | true | true | 0 | — | **startup_crash** | — | — | — |
| 3 | socket | triton | flashinfer | cutlass | false | false | 0 | 8 | **startup_crash** | — | — | — |
| 4 | socket | triton | triton | cutlass | false | true | 0 | 8 | **startup_crash** | — | — | — |
| 5 | socket | triton | triton | cutlass | true | true | 0 | — | **startup_crash** | — | — | — |
| 6 | socket | triton | triton | cutlass | false | false | 0 | 8 | **startup_crash** | — | — | — |
| 7 | socket | triton | flashinfer | fi_cutlass | false | true | 0 | 8 | **startup_crash** | — | — | — |
| 8 | socket | triton | flashinfer | fi_cutlass | true | true | 0 | — | **startup_crash** | — | — | — |
| 9 | socket | triton | flashinfer | fi_cutlass | false | false | 0 | 8 | **startup_crash** | — | — | — |
| 10 | socket | triton | triton | fi_cutlass | false | true | 0 | 8 | **startup_crash** | — | — | — |
| 11 | socket | triton | triton | fi_cutlass | true | true | 0 | — | **startup_crash** | — | — | — |
| 12 | socket | triton | triton | fi_cutlass | false | false | 0 | 8 | **startup_crash** | — | — | — |
| 13 | socket | fi_cutlass | flashinfer | cutlass | false | true | 0 | 8 | **startup_crash** | — | — | — |
| 14 | socket | fi_cutlass | flashinfer | cutlass | true | true | 0 | — | **startup_crash** | — | — | — |
| 15 | socket | fi_cutlass | flashinfer | cutlass | false | false | 0 | 8 | **startup_crash** | — | — | — |
| 16 | socket | fi_cutlass | triton | cutlass | false | true | 0 | 8 | **startup_crash** | — | — | — |
| 17 | socket | fi_cutlass | triton | cutlass | true | true | 0 | — | **startup_crash** | — | — | — |
| 18 | socket | fi_cutlass | triton | cutlass | false | false | 0 | 8 | **startup_crash** | — | — | — |
| 19 | socket | fi_cutlass | flashinfer | fi_cutlass | false | true | 0 | 8 | **startup_crash** | — | — | — |
| 20 | socket | fi_cutlass | flashinfer | fi_cutlass | true | true | 0 | — | **startup_crash** | — | — | — |
| 21 | socket | fi_cutlass | flashinfer | fi_cutlass | false | false | 0 | 8 | **startup_crash** | — | — | — |
| 22 | socket | fi_cutlass | triton | fi_cutlass | false | true | 0 | 8 | **startup_crash** | — | — | — |
| 23 | socket | fi_cutlass | triton | fi_cutlass | true | true | 0 | — | **startup_crash** | — | — | — |
| 24 | socket | fi_cutlass | triton | fi_cutlass | false | false | 0 | 8 | **startup_crash** | — | — | — |
| 25 | socket | cutlass | flashinfer | cutlass | false | true | 0 | 8 | **startup_crash** | — | — | — |
| 26 | socket | cutlass | flashinfer | cutlass | true | true | 0 | — | **startup_crash** | — | — | — |
| 27 | socket | cutlass | flashinfer | cutlass | false | false | 0 | 8 | **startup_crash** | — | — | — |

### Column Legend

| Column | Description |
|--------|-------------|
| nccl_transport | `sglang_nccl_transport` — NCCL inter-node transport (`socket` = TCP/IP, `roce` = RDMA/RoCE via IBext) |
| moe_runner | `moe_runner_backend` — MoE expert dispatch kernel (`fi_cutlass` = flashinfer_cutlass, `triton` = triton, `cutlass` = cutlass direct) |
| attention | `attention_backend` — attention kernel (`flashinfer` = FlashInfer, `triton` = Triton) |
| fp8_gemm | `fp8_gemm_runner_backend` — FP8 dense GEMM kernel (`cutlass` = CUTLASS, `fi_cutlass` = flashinfer_cutlass) |
| dis_cuda_graph | `disable_cuda_graph` — true = eager mode, false = capture CUDA graphs |
| dis_piecewise | `disable_piecewise_cuda_graph` — true = only fixed-BS graphs, false = piecewise variable-length graphs |
| pp_async | `pp_async_batch_depth` — async micro-batches in PP pipeline (0 = synchronous). Irrelevant with PP=1 (TP-only), always 0 here. |
| cuda_graph_max_bs | `cuda_graph_max_bs` — largest batch size to capture (— = N/A when graphs disabled) |
| 1∥ tok/s | Throughput with 1 sequential request (= per-request tok/s) |
| 4∥ tok/s | Peak concurrent throughput at 4∥ (sum of per-request tok/s) |
| 8∥ tok/s | Peak concurrent throughput at 8∥ (sum of per-request tok/s) |

---

## Test Details

### #1 — triton moe / flashinfer attn / cutlass fp8 / cuda_graph

- **Outcome:** startup_crash
- **Error:** Worker-1 restarted (total=1)
- **Time:** 2026-04-05 08:21–08:31 UTC

### #2 — triton moe / flashinfer attn / cutlass fp8 / no-cuda-graph

- **Outcome:** startup_crash
- **Error:** Head + all 3 workers restarted (total=1 each)
- **Time:** 2026-04-05 08:31–08:41 UTC

### #3 — triton moe / flashinfer attn / cutlass fp8 / piecewise

- **Outcome:** startup_crash
- **Error:** Head + all 3 workers restarted (total=1 each)
- **Time:** 2026-04-05 08:42–08:51 UTC

### #4 — triton moe / triton attn / cutlass fp8 / cuda_graph

- **Outcome:** startup_crash
- **Error:** Head + all 3 workers restarted (total=1 each)
- **Time:** 2026-04-05 08:52–09:02 UTC

### #5 — triton moe / triton attn / cutlass fp8 / no-cuda-graph

- **Outcome:** startup_crash
- **Error:** Head + all 3 workers restarted (total=1 each)
- **Time:** 2026-04-05 09:02–09:12 UTC

### #6 — triton moe / triton attn / cutlass fp8 / piecewise

- **Outcome:** startup_crash
- **Error:** Head + all 3 workers restarted (total=1 each)
- **Time:** 2026-04-05 09:12–09:22 UTC

### #7 — triton moe / flashinfer attn / fi_cutlass fp8 / cuda_graph

- **Outcome:** startup_crash
- **Error:** Head + all 3 workers restarted (total=1 each)
- **Time:** 2026-04-05 09:23–09:33 UTC

### #8 — triton moe / flashinfer attn / fi_cutlass fp8 / no-cuda-graph

- **Outcome:** startup_crash
- **Error:** Head + all 3 workers restarted (total=1 each)
- **Time:** 2026-04-05 09:33–09:43 UTC

### #9 — triton moe / flashinfer attn / fi_cutlass fp8 / piecewise

- **Outcome:** startup_crash
- **Error:** Worker-2 restarted (total=1)
- **Time:** 2026-04-05 09:43–09:53 UTC

### #10 — triton moe / triton attn / fi_cutlass fp8 / cuda_graph

- **Outcome:** startup_crash
- **Error:** Head + all 3 workers restarted (total=1 each)
- **Time:** 2026-04-05 09:54–10:03 UTC

### #11 — triton moe / triton attn / fi_cutlass fp8 / no-cuda-graph

- **Outcome:** startup_crash
- **Error:** Head + all 3 workers restarted (total=1 each)
- **Time:** 2026-04-05 10:04–10:13 UTC

### #12 — triton moe / triton attn / fi_cutlass fp8 / piecewise

- **Outcome:** startup_crash
- **Error:** Head + all 3 workers restarted (total=1 each)
- **Time:** 2026-04-05 10:14–10:24 UTC

### #13–24 — fi_cutlass MoE (all configs)

- **All startup_crash.** Crashes within ~1–2 min (much faster than triton MoE), suggesting failure during FlashInfer MoE JIT initialization — before weight loading completes.
- **Time range:** 2026-04-05 10:24–10:47 UTC
- Every combination of flashinfer/triton attention × cutlass/fi_cutlass fp8_gemm × cuda_graph/no-cuda-graph/piecewise crashed.

### #25–27 — cutlass MoE (flashinfer attn / cutlass fp8)

- **All startup_crash.** Same ~1–2 min rapid crash pattern as fi_cutlass MoE. Test matrix truncated after 3 configs (all cutlass MoE combos already failing).
- **Time range:** 2026-04-05 10:48–10:53 UTC
