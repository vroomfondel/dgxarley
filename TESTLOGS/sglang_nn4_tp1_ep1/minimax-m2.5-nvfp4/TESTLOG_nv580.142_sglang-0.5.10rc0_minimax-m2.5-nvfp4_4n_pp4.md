# SGLang Test Log — MiniMax M2.5 NVFP4, 4 Nodes PP=4, v0.5.10rc0

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

Previous test series: 3-node PP=3 (`TESTLOG_nv580.142_sglang-0.5.10rc0_minimax-m2.5-nvfp4_3n.md`), 4-node TP=4 EP=4 (`TESTLOG_nv580.142_sglang-0.5.10rc0_minimax-m2.5-nvfp4_4n.md`).

---

## Configuration Matrix

All tests use: `tp=1, pp=4, ep=1, quantization=modelopt_fp4, kv_cache_dtype=fp8_e4m3, mem_fraction_static=0.80, disable_deep_gemm=true, context_length=196608, max_running_requests=32, schedule_policy=lpm, watchdog_timeout=3600, dist_timeout=1800` unless noted.

| # | nccl_transport | moe_runner | attention | fp4_gemm | dis_cuda_graph | dis_piecewise | pp_async | cuda_graph_max_bs | Stability | 1∥ tok/s | 4∥ tok/s | 8∥ tok/s |
|---|----------------|------------|-----------|----------|----------------|---------------|----------|-------------------|-----------|---------|---------|---------|
| 1 | socket | triton | flashinfer | fi_cutlass | false | true | 0 | 16 | OK | 16.2 | 39.3 | 61.2 |
| 2 | socket | triton | flashinfer | fi_cutlass | true | true | 0 | — | OK | 5.5 | 42.4 | 63.9 |
| 3 | socket | triton | flashinfer | fi_cutlass | false | false | 0 | 16 | OK | 14.9 | 42.5 | 56.8 |
| 4 | socket | triton | flashinfer | fi_cutlass | false | true | 2 | 16 | OK | 15.9 | 36.9 | 50.0 |
| 5 | socket | triton | triton | fi_cutlass | false | true | 0 | 16 | OK | 16.1 | 50.1 | 52.2 |
| 6 | socket | triton | triton | fi_cutlass | true | true | 0 | — | OK | 7.1 | 43.5 | 53.5 |
| 7 | socket | triton | triton | fi_cutlass | false | false | 0 | 16 | OK | 15.4 | 41.1 | 51.6 |
| 8 | socket | triton | triton | fi_cutlass | false | true | 2 | 16 | OK | 15.9 | 41.7 | 59.5 |
| 9 | socket | triton | flashinfer | fi_cudnn | false | true | 0 | 16 | OK | 16.0 | 38.1 | 48.9 |
| 10 | socket | triton | flashinfer | fi_cudnn | true | true | 0 | — | OK | 5.9 | 42.1 | 52.9 |
| 11 | socket | triton | flashinfer | fi_cudnn | false | false | 0 | 16 | OK | 16.0 | 39.6 | 49.0 |
| 12 | socket | triton | flashinfer | fi_cudnn | false | true | 2 | 16 | OK | 15.8 | 36.6 | 59.8 |
| 13 | socket | triton | triton | fi_cudnn | false | true | 0 | 16 | OK | 15.8 | 40.9 | 62.0 |
| 14 | socket | triton | triton | fi_cudnn | true | true | 0 | — | OK | 5.3 | 42.7 | 56.9 |
| 15 | socket | triton | triton | fi_cudnn | false | false | 0 | 16 | OK | 15.1 | 41.8 | 55.0 |
| 16 | socket | triton | triton | fi_cudnn | false | true | 2 | 16 | OK | 16.0 | 35.9 | 59.3 |
| 17 | socket | fi_cutlass | flashinfer | fi_cutlass | false | true | 0 | 16 | CRASH | — | — | — |
| 18 | socket | fi_cutlass | flashinfer | fi_cutlass | true | true | 0 | — | CRASH | — | — | — |
| 19 | socket | fi_cutlass | flashinfer | fi_cutlass | false | false | 0 | 16 | CRASH | — | — | — |
| 20 | socket | fi_cutlass | flashinfer | fi_cutlass | false | true | 2 | 16 | CRASH | — | — | — |
| 21 | socket | fi_cutlass | triton | fi_cutlass | false | true | 0 | 16 | CRASH | — | — | — |
| 22 | socket | fi_cutlass | triton | fi_cutlass | true | true | 0 | — | CRASH | 12.6 | — | — |
| 23 | socket | fi_cutlass | triton | fi_cutlass | false | false | 0 | 16 | CRASH | — | — | — |
| 24 | socket | fi_cutlass | triton | fi_cutlass | false | true | 2 | 16 | CRASH | — | — | — |
| 25 | socket | fi_cutlass | flashinfer | fi_cudnn | false | true | 0 | 16 | FAIL | 15.8 | — | — |
| 26 | socket | fi_cutlass | flashinfer | fi_cudnn | true | true | 0 | — | FAIL | — | — | — |
| 27 | socket | fi_cutlass | flashinfer | fi_cudnn | false | false | 0 | 16 | CRASH | — | — | — |
| 28 | socket | fi_cutlass | flashinfer | fi_cudnn | false | true | 2 | 16 | CRASH | 15.8 | — | — |
| 29 | socket | fi_cutlass | triton | fi_cudnn | false | true | 0 | 16 | CRASH | 15.2 | — | — |
| 30 | socket | fi_cutlass | triton | fi_cudnn | true | true | 0 | — | CRASH | — | — | — |
| 31 | socket | fi_cutlass | triton | fi_cudnn | false | false | 0 | 16 | CRASH | — | — | — |
| 32 | socket | fi_cutlass | triton | fi_cudnn | false | true | 2 | 16 | FAIL | — | — | — |

### Column Legend

| Column | Description |
|--------|-------------|
| nccl_transport | `sglang_nccl_transport` — NCCL inter-node transport (`socket` = TCP/IP, `roce` = RDMA/RoCE via IBext) |
| moe_runner | `moe_runner_backend` — MoE expert dispatch kernel (`fi_cutlass` = flashinfer_cutlass, `triton` = triton→cutlass_moe_fp4 fallback for NVFP4) |
| attention | `attention_backend` — attention kernel (`flashinfer` = FlashInfer, `triton` = Triton) |
| fp4_gemm | `fp4_gemm_backend` — FP4 dense GEMM kernel (`fi_cutlass` = flashinfer_cutlass, `fi_cudnn` = flashinfer_cudnn; valid choices: auto, flashinfer_cudnn, flashinfer_cutlass, flashinfer_trtllm) |
| dis_cuda_graph | `disable_cuda_graph` — true = eager mode, false = capture CUDA graphs |
| dis_piecewise | `disable_piecewise_cuda_graph` — true = only fixed-BS graphs, false = piecewise variable-length graphs |
| pp_async | `pp_async_batch_depth` — async micro-batches in PP pipeline (0 = synchronous) |
| cuda_graph_max_bs | `cuda_graph_max_bs` — largest batch size to capture (— = N/A when graphs disabled) |
| 1∥ tok/s | Throughput with 1 sequential request (= per-request tok/s) |
| 4∥ tok/s | Peak concurrent throughput at 4∥ (sum of per-request tok/s) |
| 8∥ tok/s | Peak concurrent throughput at 8∥ (sum of per-request tok/s) |

---

## RoCE Transport Tests

Manual tests with `nccl_transport=roce` (RDMA/RoCE via IBext over QSFP SR-IOV VFs). Same base config as socket matrix above.

| # | nccl_transport | moe_runner | attention | fp4_gemm | dis_cuda_graph | dis_piecewise | pp_async | cuda_graph_max_bs | Stability | 1∥ tok/s | 4∥ tok/s | 8∥ tok/s |
|---|----------------|------------|-----------|----------|----------------|---------------|----------|-------------------|-----------|---------|---------|---------|
| R1 | roce | triton | triton | fi_cutlass | false | true | 0 | 16 | OK | — | — | 50.8 |

### R1 Detail (= socket test 5 equivalent, with RoCE)

8∥ run: all 8 requests completed successfully.

| Request | tok/s | TTFT | Total | Output tok | Finish |
|---------|-------|------|-------|------------|--------|
| 1 | 6.1 | 0.91s | 268.4s | 1645 | stop |
| 2 | 6.7 | 0.91s | 307.4s | 2048 | length |
| 3 | 6.0 | 1.50s | 223.9s | 1348 | stop |
| 4 | 6.2 | 1.50s | 271.6s | 1674 | stop |
| 5 | 6.4 | 1.51s | 293.9s | 1877 | stop |
| 6 | 6.0 | 1.50s | 230.0s | 1386 | stop |
| 7 | 6.7 | 0.30s | 307.0s | 2048 | length |
| 8 | 6.7 | 0.91s | 307.4s | 2048 | length |

Peak 8∥: 50.8 tok/s. Avg per-request: 6.3 tok/s. Wall time: 307.5s.

---

## Notes

### fi_cutlass MoE is broken (tests 17–32)

All 16 `moe_runner_backend=flashinfer_cutlass` tests failed. Most crashed with worker/head restarts; the rest (25, 26, 32) ran without pod crashes but all requests failed or were aborted. Only tests 22, 25, 28, 29 completed a single n1 request before failing at higher concurrency.

| Test | Outcome | Detail |
|------|---------|--------|
| 17   | CRASH   | Worker-1 restart |
| 18   | CRASH   | Worker-1 + Worker-2 restart |
| 19   | CRASH   | Worker-1 restart |
| 20   | CRASH   | Worker-1 + Worker-2 + Worker-3 restart |
| 21   | CRASH   | Head + all 3 workers restart |
| 22   | CRASH   | n1 OK (12.6 tok/s), n4 all aborted, then 3 workers restart |
| 23   | CRASH   | Head + all 3 workers restart |
| 24   | CRASH   | Head + all 3 workers restart |
| 25   | FAIL    | n1 OK (15.8 tok/s), n4+n8 all requests failed (no pod crash) |
| 26   | FAIL    | All requests failed at every concurrency (no pod crash) |
| 27   | CRASH   | Worker-1 restart |
| 28   | CRASH   | n1 OK (15.8 tok/s), n4 all aborted, Worker-3 restart |
| 29   | CRASH   | n1 OK (15.2 tok/s), n4 all aborted, Worker-3 restart |
| 30   | CRASH   | Worker-2 + Worker-3 restart |
| 31   | CRASH   | Worker-2 + Worker-3 restart |
| 32   | FAIL    | All requests failed at every concurrency (no pod crash) |

The `triton` MoE runner (tests 1–16) is stable across all configurations. `flashinfer_cutlass` MoE is completely broken on this model/version combo — no concurrency level beyond n1 succeeded, and even n1 only worked in 4 of 16 cases.

### Eager mode (disable_cuda_graph=true) single-request penalty

Tests 2, 6, 10, 14 (eager mode) show ~5–7 tok/s at 1∥ vs ~15–16 tok/s with CUDA graphs. At 8∥ they recover to competitive throughput (53–64 tok/s), suggesting the overhead is amortized under concurrency.

### Best configurations (8∥ peak throughput)

| Rank | Test | Config summary | 8∥ tok/s |
|------|------|----------------|---------|
| 1 | 2 | triton moe, flashinfer attn, fi_cutlass fp4, eager | 63.9 |
| 2 | 13 | triton moe, triton attn, fi_cudnn fp4, cuda graph | 62.0 |
| 3 | 1 | triton moe, flashinfer attn, fi_cutlass fp4, cuda graph | 61.2 |
| 4 | 12 | triton moe, flashinfer attn, fi_cudnn fp4, pp-async-2 | 59.8 |
| 5 | 8 | triton moe, triton attn, fi_cutlass fp4, pp-async-2 | 59.5 |
