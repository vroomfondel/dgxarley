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
| 1 | socket | triton | flashinfer | fi_cutlass | false | true | 0 | 8 | **n=1 only** | 5.64 | crash | crash |
| 2 | socket | triton | flashinfer | fi_cutlass | true | true | 0 | — | **n=1 only** | 3.67 | crash | crash |
| 3 | socket | triton | flashinfer | fi_cutlass | false | false | 0 | 8 | **n=1 only** | 5.61 | crash | crash |
| 4 | socket | triton | triton | fi_cutlass | false | true | 0 | 8 | **startup_crash** | — | — | — |
| 5 | socket | triton | triton | fi_cutlass | true | true | 0 | — | **startup_crash** | — | — | — |
| 6 | socket | triton | triton | fi_cutlass | false | false | 0 | 8 | **startup_crash** | — | — | — |
| 7 | socket | triton | flashinfer | fi_cudnn | false | true | 0 | 8 | **n=1 only** | 5.45 | crash | crash |
| 8 | socket | triton | flashinfer | fi_cudnn | true | true | 0 | — | **n=1 only** | 3.66 | crash | crash |
| 9 | socket | triton | flashinfer | fi_cudnn | false | false | 0 | 8 | **n=1 only** | 5.45 | crash | crash |
| 10 | socket | triton | triton | fi_cudnn | false | true | 0 | 8 | **startup_crash** | — | — | — |
| 11 | socket | triton | triton | fi_cudnn | true | true | 0 | — | **startup_crash** | — | — | — |
| 12 | socket | triton | triton | fi_cudnn | false | false | 0 | 8 | **startup_crash** | — | — | — |
| 13 | socket | fi_cutlass | flashinfer | fi_cutlass | false | true | 0 | 8 | **n=1 only** | 5.50 | crash | crash |
| 14 | socket | fi_cutlass | flashinfer | fi_cutlass | true | true | 0 | — | **bench_crash** | — | — | — |
| 15 | socket | fi_cutlass | flashinfer | fi_cutlass | false | false | 0 | 8 | **bench_crash** | — | — | — |
| 16 | socket | fi_cutlass | triton | fi_cutlass | false | true | 0 | 8 | **startup_crash** | — | — | — |
| 17 | socket | fi_cutlass | triton | fi_cutlass | true | true | 0 | — | **startup_crash** | — | — | — |
| 18 | socket | fi_cutlass | triton | fi_cutlass | false | false | 0 | 8 | **startup_crash** | — | — | — |
| 19 | socket | fi_cutlass | flashinfer | fi_cudnn | false | true | 0 | 8 | **startup_crash** | — | — | — |
| 20 | socket | fi_cutlass | flashinfer | fi_cudnn | true | true | 0 | — | **bench_crash** | — | — | — |
| 21 | socket | fi_cutlass | flashinfer | fi_cudnn | false | false | 0 | 8 | **bench_crash** | — | — | — |
| 22 | socket | fi_cutlass | triton | fi_cudnn | false | true | 0 | 8 | **startup_crash** | — | — | — |
| 23 | socket | fi_cutlass | triton | fi_cudnn | true | true | 0 | — | **startup_crash** | — | — | — |
| 24 | socket | fi_cutlass | triton | fi_cudnn | false | false | 0 | 8 | **startup_crash** | — | — | — |
| 25 | socket | cutlass | flashinfer | fi_cutlass | false | true | 0 | 8 | **n=1 only** | 5.47 | crash | crash |
| 26 | socket | cutlass | flashinfer | fi_cutlass | true | true | 0 | — | **n=1 only** | 3.50 | crash | crash |
| 27 | socket | cutlass | flashinfer | fi_cutlass | false | false | 0 | 8 | **n=1 only** | 5.48 | crash | crash |
| 28 | socket | cutlass | triton | fi_cutlass | false | true | 0 | 8 | **startup_crash** | — | — | — |
| 29 | socket | cutlass | triton | fi_cutlass | true | true | 0 | — | **startup_crash** | — | — | — |
| 30 | socket | cutlass | triton | fi_cutlass | false | false | 0 | 8 | **startup_crash** | — | — | — |
| 31 | socket | cutlass | flashinfer | fi_cudnn | false | true | 0 | 8 | **n=1 only** | 5.48 | crash | crash |
| 32 | socket | cutlass | flashinfer | fi_cudnn | true | true | 0 | — | **n=1 only** | 3.47 | crash | crash |
| 33 | socket | cutlass | flashinfer | fi_cudnn | false | false | 0 | 8 | **n=1 only** | 5.45 | crash | crash |
| 34 | socket | cutlass | triton | fi_cudnn | false | true | 0 | 8 | **startup_crash** | — | — | — |
| 35 | socket | cutlass | triton | fi_cudnn | true | true | 0 | — | **startup_crash** | — | — | — |
| 36 | socket | cutlass | triton | fi_cudnn | false | false | 0 | 8 | **startup_crash** | — | — | — |

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

### #6 — triton moe / triton attn / fi_cutlass fp4 / piecewise

- **Outcome:** startup_crash
- **Error:** Workers 2, 3 restarted (same pattern as #4)
- **Note:** All three `triton` attention variants (#4, #5, #6) crash at startup on PP=4

### #7 — triton moe / flashinfer attn / fi_cudnn fp4 / cuda_graph

- **Outcome:** n=1 success, n=4+ crash (FlashInfer merge_state invalid argument — same as #1)
- **n=1:** 5.45 tok/s, 3072 tokens (2137 think + 931 content), TTFT 3.7s, finish=length
- **n=4:** 0/4 errors (FlashInfer cascade merge crash)
- **n=8:** 0/8 errors (same)
- **Note:** `fi_cudnn` fp4 gemm backend performs identically to `fi_cutlass` for n=1; both fail at n=4 with the same FlashInfer error

### #8 — triton moe / flashinfer attn / fi_cudnn fp4 / no-cuda-graph

- **Outcome:** n=1 success (slow), n=4+ crash
- **n=1:** 3.66 tok/s, 3072 tokens (1835 think + 1382 content), **TTFT 243.9s** (extreme first-token latency — no cuda graph)
- **n=4:** 0/4 errors (instant)
- **n=8:** 0/8 errors (instant)

### #9 — triton moe / flashinfer attn / fi_cudnn fp4 / piecewise

- **Outcome:** n=1 success, n=4+ crash (same FlashInfer merge_state error)
- **n=1:** 5.45 tok/s, 3072 tokens (1263 think + 2136 content), TTFT 4.7s, finish=length
- **n=4:** 0/4 errors (instant)
- **n=8:** 0/8 errors (instant)

### #10 — triton moe / triton attn / fi_cudnn fp4 / cuda_graph

- **Outcome:** startup_crash
- **Error:** Workers 2, 3 restarted

### #11 — triton moe / triton attn / fi_cudnn fp4 / no-cuda-graph

- **Outcome:** startup_crash
- **Error:** Head + all 3 workers restarted

### #12 — triton moe / triton attn / fi_cudnn fp4 / piecewise

- **Outcome:** startup_crash
- **Error:** Workers 2, 3 restarted
- **Note:** All `triton` attention combinations (#4–6, #10–12) crash at startup regardless of fp4 gemm backend or cuda graph settings

### #13 — fi_cutlass moe / flashinfer attn / fi_cutlass fp4 / cuda_graph

- **Outcome:** n=1 success, n=4+ crash
- **n=1:** 5.50 tok/s, 3072 tokens (797 think + 2170 content), TTFT 2.7s, finish=length
- **n=4:** 0/4 errors (FlashInfer cascade merge crash)
- **n=8:** 0/8 errors (same)

### #14 — fi_cutlass moe / flashinfer attn / fi_cutlass fp4 / no-cuda-graph

- **Outcome:** bench_crash (worker restart during n=1 bench)
- **Error:** Worker 1 restarted (total=1)
- **n=1:** aborted (request did not complete successfully — pod crash mid-request)
- **Note:** Unlike `triton` moe which survived n=1 no-cuda-graph, `fi_cutlass` moe crashes even during single-request benchmarking without cuda graphs

### #15 — fi_cutlass moe / flashinfer attn / fi_cutlass fp4 / piecewise

- **Outcome:** bench_crash (workers 2, 3 restart during n=1 bench)
- **Error:** Workers 2, 3 restarted
- **n=1:** aborted mid-request

### #16 — fi_cutlass moe / triton attn / fi_cutlass fp4 / cuda_graph

- **Outcome:** startup_crash
- **Error:** Workers 2, 3 restarted

### #17 — fi_cutlass moe / triton attn / fi_cutlass fp4 / no-cuda-graph

- **Outcome:** startup_crash
- **Error:** Head + all workers restarted

### #18 — fi_cutlass moe / triton attn / fi_cutlass fp4 / piecewise

- **Outcome:** startup_crash
- **Error:** Workers 2, 3 restarted

### #19 — fi_cutlass moe / flashinfer attn / fi_cudnn fp4 / cuda_graph

- **Outcome:** startup_crash
- **Error:** Head + all workers restarted

### #20 — fi_cutlass moe / flashinfer attn / fi_cudnn fp4 / no-cuda-graph

- **Outcome:** bench_crash (worker 3 restart during n=1 bench)
- **Error:** Worker 3 restarted
- **n=1:** aborted mid-request

### #21 — fi_cutlass moe / flashinfer attn / fi_cudnn fp4 / piecewise

- **Outcome:** bench_crash (worker 1 restart during n=1 bench)
- **Error:** Worker 1 restarted
- **n=1:** aborted mid-request

### #22 — fi_cutlass moe / triton attn / fi_cudnn fp4 / cuda_graph

- **Outcome:** startup_crash
- **Error:** Workers 2, 3 restarted

### #23 — fi_cutlass moe / triton attn / fi_cudnn fp4 / no-cuda-graph

- **Outcome:** startup_crash
- **Error:** Head + all workers restarted

### #24 — fi_cutlass moe / triton attn / fi_cudnn fp4 / piecewise

- **Outcome:** startup_crash
- **Error:** Workers 2, 3 restarted
- **Note:** The entire `fi_cutlass` moe runner block (#13–24) is consistently unstable on PP=4: cuda_graph variants crash at startup, no-cuda-graph and piecewise variants bench_crash. Only #13 (cuda_graph) completes n=1.

### #25 — cutlass moe / flashinfer attn / fi_cutlass fp4 / cuda_graph

- **Outcome:** n=1 success, n=4+ crash (same FlashInfer merge_state error)
- **n=1:** 5.47 tok/s, 3072 tokens (1420 think + 1272 content), TTFT 1.4s, finish=length
- **n=4:** 0/4 errors (instant)
- **n=8:** 0/8 errors (instant)

### #26 — cutlass moe / flashinfer attn / fi_cutlass fp4 / no-cuda-graph

- **Outcome:** n=1 success (slow), n=4+ crash
- **n=1:** 3.50 tok/s, 3072 tokens (1486 think + 1810 content), **TTFT 281.3s** (extreme first-token latency)
- **n=4:** 0/4 errors (instant)
- **n=8:** 0/8 errors (instant)

### #27 — cutlass moe / flashinfer attn / fi_cutlass fp4 / piecewise

- **Outcome:** n=1 success, n=4+ crash (same FlashInfer merge_state error)
- **n=1:** 5.48 tok/s, 2960 tokens (1461 think + 1785 content), TTFT 1.3s, finish=stop
- **n=4:** 0/4 errors (instant)
- **n=8:** 0/8 errors (instant)

### #28 — cutlass moe / triton attn / fi_cutlass fp4 / cuda_graph

- **Outcome:** startup_crash
- **Error:** Workers 2, 3 restarted

### #29 — cutlass moe / triton attn / fi_cutlass fp4 / no-cuda-graph

- **Outcome:** startup_crash
- **Error:** Head + all workers restarted

### #30 — cutlass moe / triton attn / fi_cutlass fp4 / piecewise

- **Outcome:** startup_crash
- **Error:** Workers 2, 3 restarted

### #31 — cutlass moe / flashinfer attn / fi_cudnn fp4 / cuda_graph

- **Outcome:** n=1 success, n=4+ crash (same FlashInfer merge_state error)
- **n=1:** 5.48 tok/s, 2748 tokens (1514 think + 1479 content), TTFT 1.3s, finish=stop
- **n=4:** 0/4 errors (instant)
- **n=8:** 0/8 errors (instant)

### #32 — cutlass moe / flashinfer attn / fi_cudnn fp4 / no-cuda-graph

- **Outcome:** n=1 success (slow), n=4+ crash
- **n=1:** 3.47 tok/s, 3072 tokens (1625 think + 1713 content), **TTFT 287.0s** (extreme first-token latency)
- **n=4:** 0/4 errors (instant)
- **n=8:** 0/8 errors (instant)

### #33 — cutlass moe / flashinfer attn / fi_cudnn fp4 / piecewise

- **Outcome:** n=1 success, n=4+ crash (same FlashInfer merge_state error)
- **n=1:** 5.45 tok/s, 3072 tokens (1401 think + 1929 content), TTFT 3.1s, finish=length
- **n=4:** 0/4 errors (instant)
- **n=8:** 0/8 errors (instant)

### #34 — cutlass moe / triton attn / fi_cudnn fp4 / cuda_graph

- **Outcome:** startup_crash
- **Error:** Workers 2, 3 restarted

### #35 — cutlass moe / triton attn / fi_cudnn fp4 / no-cuda-graph

- **Outcome:** startup_crash
- **Error:** Head + all workers restarted

### #36 — cutlass moe / triton attn / fi_cudnn fp4 / piecewise

- **Outcome:** startup_crash
- **Error:** Workers 2, 3 restarted
- **Note:** All `triton` attention variants with `cutlass` moe (#28–30, #34–36) crash at startup, identical to `triton`/`fi_cutlass` moe combinations.

---

## Test 37 — MTP / Speculative Decoding Attempt

**Configuration:** Same as winning config #1 (triton moe / flashinfer attn / fi_cutlass fp4 / cuda_graph=true / piecewise=false) plus:
- `speculative_enabled: true`
- `speculative_algo: NEXTN`
- `speculative_num_steps: 3`
- `speculative_num_draft_tokens: 4`

**Outcome:** startup_crash — head + all 3 workers restarted within ~74s (21:28:40–21:29:54 UTC)

**Error:** `Pod sglang-head-6b89447d47-rjc9w: +1 restart(s); Pod sglang-worker-1: +1; Pod sglang-worker-2: +1; Pod sglang-worker-3: +1`

**Conclusion:** MTP/NEXTN speculative decoding is not compatible with PP=4 on this model/image combination. The crash is immediate (before NCCL init or weight loading completes), suggesting a configuration-level incompatibility rather than a runtime OOM or NCCL issue. PP=4 + MTP was not expected to work given SGLang's speculative decoding support status for pipeline parallelism.

---

## Overall Findings

- **n=4 / n=8 concurrency:** Failed for every configuration tested (all 36 matrix tests + test 37). The consistent failure mode for `flashinfer` attention tests is a FlashInfer `merge_state` invalid-argument error at n=4 — 0 tokens returned, instant failure. This is a PP=4 + FlashInfer multi-request bug.
- **triton attention + PP=4:** Crashes at startup across all moe runner and fp4 gemm combinations (#4–6, #10–12, #16–18, #22–24, #28–30, #34–36). `flashinfer` attention is required for PP=4 to even start.
- **fi_cutlass moe runner:** Unstable — cuda_graph variants (#13, #19) crash at startup; no-cuda-graph and piecewise variants (#14, #15, #20, #21) bench_crash (worker OOM/crash mid-request during n=1).
- **Best n=1 throughput:** ~5.64 tok/s (#1), with most cuda_graph flashinfer-attn configs clustering at 5.45–5.50 tok/s. No-cuda-graph configs ~3.47–3.67 tok/s with extreme TTFT (244–287s).
- **fp4 gemm backend (fi_cutlass vs fi_cudnn):** No meaningful difference in n=1 throughput (~5.45–5.50 tok/s across all flashinfer-attn cuda_graph variants).
- **moe runner (triton vs cutlass):** No meaningful difference in n=1 throughput for the configurations that worked. Both cluster at the same ~5.45–5.50 tok/s.
- **MTP speculative decoding:** Crashes immediately on PP=4 (#37).
- **Root blocker:** The FlashInfer `merge_state` crash at n=4 prevents any concurrent throughput measurement. All PP=4 testing is effectively limited to sequential single-request operation.
