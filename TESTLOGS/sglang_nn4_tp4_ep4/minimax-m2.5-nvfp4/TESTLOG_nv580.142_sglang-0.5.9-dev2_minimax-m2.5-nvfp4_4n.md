# SGLang Test Log — MiniMax M2.5 NVFP4, 4 Nodes, v0.5.9dev2

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
| Model | `nvidia/MiniMax-M2.5-NVFP4`                    |

Previous test series with 3 nodes: see `TESTLOG_nv580.142_sglang-0.5.10rc0_minimax-m2.5-nvfp4_3n.md`.

---

## Configuration Matrix

All tests use: `tp=4, pp=1, ep=4, quantization=modelopt_fp4, kv_cache_dtype=fp8_e4m3, mem_fraction_static=0.60, disable_deep_gemm=true, context_length=196608, max_running_requests=32, schedule_policy=lpm, watchdog_timeout=3600, dist_timeout=1800` unless noted.

| # | nccl_transport | moe_runner | attention | fp4_gemm | dis_cuda_graph | dis_piecewise | pp_async | cuda_graph_max_bs | Stability | 1∥ tok/s | 4∥ tok/s | 8∥ tok/s |
|---|----------------|------------|-----------|----------|----------------|---------------|----------|-------------------|-----------|---------|---------|---------|
| 1 | socket | triton | flashinfer | fi_cutlass | false | true | 0 | 8 | startup crash (workers restarted) | — | — | — |
| 2 | socket | triton | flashinfer | fi_cutlass | true | true | 0 | — | EP inference bug → worker crash | — | — | — |
| 3 | socket | triton | flashinfer | fi_cutlass | false | false | 0 | 8 | startup crash (piecewise graph capture) | — | — | — |
| 4 | socket | triton | triton | fi_cutlass | false | true | 0 | 8 | startup crash (graph capture) | — | — | — |
| 5 | socket | triton | triton | fi_cutlass | true | true | 0 | — | EP inference bug → worker crash | — | — | — |
| 6 | socket | triton | triton | fi_cutlass | false | false | 0 | 8 | startup crash (piecewise graph capture) | — | — | — |
| 7 | socket | triton | flashinfer | fi_cudnn | false | true | 0 | 8 | startup crash (graph capture) | — | — | — |
| 8 | socket | triton | flashinfer | fi_cudnn | true | true | 0 | — | EP inference bug → worker crash | — | — | — |
| 9 | socket | triton | flashinfer | fi_cudnn | false | false | 0 | 8 | startup crash (piecewise graph capture) | — | — | — |
| 10 | socket | triton | triton | fi_cudnn | false | true | 0 | 8 | startup crash (graph capture) | — | — | — |
| 11 | socket | triton | triton | fi_cudnn | true | true | 0 | — | EP inference bug → worker crash | — | — | — |
| 12 | socket | triton | triton | fi_cudnn | false | false | 0 | 8 | startup crash (piecewise graph capture) | — | — | — |
| 13 | socket | fi_cutlass | flashinfer | fi_cutlass | false | true | 0 | 8 | **STABLE** | 15.3 | 46.3 | 65.9 |
| 14 | socket | fi_cutlass | flashinfer | fi_cutlass | true | true | 0 | — | **STABLE** (3/4 at 4∥) | 14.4 (TTFT 14.7s) | 35.0 | 65.0 |
| 15 | socket | fi_cutlass | flashinfer | fi_cutlass | false | false | 0 | 8 | **STABLE** | 14.9 | 45.7 | 65.4 |
| 16 | socket | fi_cutlass | triton | fi_cutlass | false | true | 0 | 8 | EP crash at concurrency | 15.5 | — | — |
| 17 | socket | fi_cutlass | triton | fi_cutlass | true | true | 0 | — | **STABLE** | 14.9 (TTFT 9.8s) | 48.2 | 70.7 |
| 18 | socket | fi_cutlass | triton | fi_cutlass | false | false | 0 | 8 | EP crash at concurrency (1/4 at 4∥) | 15.4 | 11.3 | — |
| 19 | socket | fi_cutlass | flashinfer | fi_cudnn | false | true | 0 | 8 | **STABLE** (5/8 at 8∥) | 15.6 | 48.2 | 37.5 |
| 20 | socket | fi_cutlass | flashinfer | fi_cudnn | true | true | 0 | — | EP crash at concurrency | 15.1 | — | — |
| 21 | socket | fi_cutlass | flashinfer | fi_cudnn | false | false | 0 | 8 | **STABLE** | 14.5 | 44.0 | 67.0 |
| 22 | socket | fi_cutlass | triton | fi_cudnn | false | true | 0 | 8 | **STABLE** | 15.7 | 47.9 | 65.3 |
| 23 | socket | fi_cutlass | triton | fi_cudnn | true | true | 0 | — | timeout (not ready after 900s) | — | — | — |
| 24 | socket | fi_cutlass | triton | fi_cudnn | false | false | 0 | 8 | EP crash at 8∥ (n1/n4 ok) | 15.3 | 47.0 | — |
| 25 | socket | cutlass | flashinfer | fi_cutlass | false | true | 0 | 8 | startup crash | — | — | — |
| 26 | socket | cutlass | flashinfer | fi_cutlass | true | true | 0 | — | EP inference bug → worker crash | — | — | — |
| 27 | socket | cutlass | flashinfer | fi_cutlass | false | false | 0 | 8 | startup crash | — | — | — |
| 28 | socket | cutlass | triton | fi_cutlass | false | true | 0 | 8 | startup crash | — | — | — |
| 29 | socket | cutlass | triton | fi_cutlass | true | true | 0 | — | EP inference bug (0/N reqs) | — | — | — |
| 30 | socket | cutlass | triton | fi_cutlass | false | false | 0 | 8 | startup crash | — | — | — |
| 31 | socket | cutlass | flashinfer | fi_cudnn | false | true | 0 | 8 | startup crash | — | — | — |
| 32 | socket | cutlass | flashinfer | fi_cudnn | true | true | 0 | — | EP inference bug → worker crash | — | — | — |
| 33 | socket | cutlass | flashinfer | fi_cudnn | false | false | 0 | 8 | startup crash | — | — | — |
| 34 | socket | cutlass | triton | fi_cudnn | false | true | 0 | 8 | startup crash | — | — | — |
| 35 | socket | cutlass | triton | fi_cudnn | true | true | 0 | — | EP inference bug → worker crash | — | — | — |
| 36 | socket | cutlass | triton | fi_cudnn | false | false | 0 | 8 | startup crash | — | — | — |

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

## NEXTN Speculative Decoding (Test 37)

| Setting | Value |
|---------|-------|
| Config | winner config (fi_cutlass MoE, flashinfer attn, fi_cutlass fp4, CUDA graphs, BS=8) |
| `speculative_algo` | `NEXTN` |
| `speculative_num_steps` | 3 |
| `speculative_num_draft_tokens` | 4 |
| `speculative_draft_model_path` | `""` (native MTP heads) |
| `mem_fraction_static` | `0.40` (reduced from 0.60 to fit second weight load) |
| Patches | `set_embed_and_head` monkey-patch in `sglang_launch.sh` |

**Result:** n=1 success, n=4 crash (worker restart)

- Started: 2026-04-06 07:33:54 UTC
- Error at n=4: `Pod sglang-worker-3: +1 restart(s) (total=1)`

### n=1 — SUCCESS

| Metric | Value |
|--------|-------|
| tok/s | **9.57** |
| TTFT | 1.17s |
| Output tokens | 3072 (hit max_tokens) |
| Think tokens | ~1487 |
| Content tokens | ~1806 |
| Total time | 321s |
| Finish reason | `length` |

**9.57 tok/s with NEXTN vs ~7.5 tok/s without** — ~28% throughput improvement from speculative decoding at n=1.

### n=4 — CRASH

All 4 requests aborted after ~50s. Worker-3 restarted. Requests had partial think tokens (~260-300 each) but 0 output_tokens. The crash is concurrency-related — single requests work fine.

### Conclusion

The `set_embed_and_head` monkey-patch and reduced `mem_fraction_static=0.40` fixed both startup blockers (AttributeError + OOM). NEXTN speculative decoding **works** with a significant speed-up (~10-12 tok/s, accept rate 77%, ~3 tokens/step), but crashes after ~667 generated tokens with `cudaErrorIllegalInstruction` in `eagle_worker.py:_draft_preprocess_decode` at `torch.sum(batch.seq_lens)`. This is a CUDA kernel bug in the EAGLE draft worker — likely an out-of-bounds access in the draft model's CUDA graph at longer sequences. The crash is not concurrency-related (happens at n=1 too, just takes ~667 tokens to trigger). At n=4 all requests abort because the crash kills the scheduler.
