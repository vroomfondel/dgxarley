# SGLang Test Log — Qwen3.5 397B-A17B NVFP4, 4 Nodes, v0.5.10rc0

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
| Model | `nvidia/Qwen3.5-397B-A17B-NVFP4` |

---

## Model Notes

- 397B total / 17B active MoE (512 experts, top-10, softmax routing), NVFP4 quantized (~234 GB).
- Hybrid attention: 15 full GQA layers + 45 linear attention layers (every 4th layer is full attention). 60 layers total.
- 1 shared expert + 512 routed experts per MoE layer. Multimodal (text+image+video).
- Has MTP head (1 layer) for speculative decoding (NEXTN).
- `num_attention_heads=32, num_key_value_heads=2` — TP=4 per model card, but KV heads=2 could limit to TP=2 for full-attention layers. Needs testing.
- NVFP4: only routed expert MoE FFN weights are FP4; attention, shared experts, vision encoder, lm_head, and MTP layer remain BF16.
- ~234 GB / 4 GPUs ≈ ~59 GB/GPU — should fit on 4× DGX Spark.

---

## Configuration Matrix

All tests use: `tp=4, pp=1, ep=4, quantization=modelopt_fp4, kv_cache_dtype=fp8_e4m3, mem_fraction_static=0.80, disable_deep_gemm=true, context_length=196608, max_running_requests=32, schedule_policy=lpm, watchdog_timeout=3600, dist_timeout=1800` unless noted.

| # | nccl_transport | moe_runner | attention | fp4_gemm | dis_cuda_graph | dis_piecewise | pp_async | cuda_graph_max_bs | Stability | 1∥ tok/s | 4∥ tok/s | 8∥ tok/s |
|---|----------------|------------|-----------|----------|----------------|---------------|----------|-------------------|-----------|---------|---------|---------|
| 1 | socket | triton | flashinfer | fi_cutlass | false | true | 0 | 8 | CRASH | — | — | — |
| 2 | socket | triton | flashinfer | fi_cutlass | true | true | 0 | — | CRASH† | — | — | — |
| 3 | socket | triton | flashinfer | fi_cutlass | false | false | 0 | 8 | CRASH | — | — | — |
| 4 | socket | triton | triton | fi_cutlass | false | true | 0 | 8 | CRASH | — | — | — |
| 5 | socket | triton | triton | fi_cutlass | true | true | 0 | — | error | — | — | — |
| 6 | socket | triton | triton | fi_cutlass | false | false | 0 | 8 | CRASH | — | — | — |
| 7 | socket | triton | flashinfer | fi_cudnn | false | true | 0 | 8 | CRASH | — | — | — |
| 8 | socket | triton | flashinfer | fi_cudnn | true | true | 0 | — | error | — | — | — |
| 9 | socket | triton | flashinfer | fi_cudnn | false | false | 0 | 8 | CRASH | — | — | — |
| 10 | socket | triton | triton | fi_cudnn | false | true | 0 | 8 | CRASH | — | — | — |
| 11 | socket | triton | triton | fi_cudnn | true | true | 0 | — | error | — | — | — |
| 12 | socket | triton | triton | fi_cudnn | false | false | 0 | 8 | CRASH | — | — | — |
| 13 | socket | fi_cutlass | flashinfer | fi_cutlass | false | true | 0 | 8 | CRASH† | 12.75 | 32.61† | — |
| 14 | socket | fi_cutlass | flashinfer | fi_cutlass | true | true | 0 | — | CRASH† | 9.64† | — | — |
| 15 | socket | fi_cutlass | flashinfer | fi_cutlass | false | false | 0 | 8 | CRASH† | 12.22† | — | — |
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
| fp4_gemm | `fp4_gemm_backend` — FP4 dense GEMM kernel (`fi_cutlass` = flashinfer_cutlass, `fi_cudnn` = flashinfer_cudnn) |
| dis_cuda_graph | `disable_cuda_graph` — true = eager mode, false = capture CUDA graphs |
| dis_piecewise | `disable_piecewise_cuda_graph` — true = only fixed-BS graphs, false = piecewise variable-length graphs |
| pp_async | `pp_async_batch_depth` — async micro-batches in PP pipeline (0 = synchronous). Irrelevant with PP=1 (TP-only), always 0 here. |
| cuda_graph_max_bs | `cuda_graph_max_bs` — largest batch size to capture (— = N/A when graphs disabled) |
| 1∥ tok/s | Throughput with 1 sequential request (= per-request tok/s) |
| 4∥ tok/s | Peak concurrent throughput at 4∥ (sum of per-request tok/s) |
| 8∥ tok/s | Peak concurrent throughput at 8∥ (sum of per-request tok/s) |

---

## Results (Tests 1–15)

**Legend:** `CRASH` = startup crash (pods restarted before healthy). `CRASH†` = bench crash (started OK, worker crashed during inference). `error` = server healthy but all inference requests return error. `†` on throughput = requests were aborted (partial streaming, worker crash mid-generation).

### triton MoE (tests 1–12): all failed

- **CUDA graphs enabled (tests 1, 3, 4, 6, 7, 9, 10, 12):** All startup_crash — all 4 pods restart. CUDA graph capture with `triton` MoE on Qwen3.5 (512 experts) is not viable.
- **CUDA graphs disabled (tests 2, 5, 8, 11):** Server starts and passes health checks, but inference fails:
  - Test 2: bench_crash — 2 workers restarted, n1 request returned error (no tokens).
  - Tests 5, 8, 11: Server stays up through all 3 bench rounds (n1/n4/n8), but every single request returns `status=error` with `tokens_per_sec=null`. The `triton` MoE runner cannot dispatch 512 experts correctly on this model.

### flashinfer_cutlass MoE (tests 13–15): partially working

- **Test 13** (CUDA graph on, no piecewise): **Best config.** n1 completed successfully: **12.75 tok/s**. n4 ran but all 4 requests aborted when worker-2 crashed — peak throughput before crash: **32.61 tok/s** (7.97 + 7.84 + 8.52 + 8.28). No n8 (test aborted after worker crash).
- **Test 14** (CUDA graph off): n1 aborted at **9.64 tok/s** (worker crash after ~465 think tokens). 25% slower than test 13 — confirms CUDA graphs help.
- **Test 15** (CUDA graph on + piecewise): n1 aborted at **12.22 tok/s** (worker crash after ~963 think tokens). Close to test 13, but piecewise mode didn't prevent the crash.

### Summary

- Only `flashinfer_cutlass` MoE works for Qwen3.5-397B. `triton` MoE is broken (512 experts).
- Best throughput: **12.75 tok/s** (n1) / **~32.61 tok/s** (n4, aborted) — test 13.
- All fi_cutlass configs crash during bench (worker OOM or NCCL timeout) — not yet stable enough for production.
- Tests 16–36 (remaining fi_cutlass/cutlass MoE + triton attn + fi_cudnn fp4 combos) pending.

---

## Known Risks / Open Questions

1. **TP=4 with num_kv_heads=2:** Model card says TP=4, but only 2 KV heads in full-attention layers. May crash at TP=4 if SGLang doesn't handle KV head replication. Fallback: TP=2 EP=2.
2. **model_type=qwen3_5_moe:** May require `transformers` upgrade in the container image (similar to GLM-5 requiring transformers ≥5.3.0).
3. **Hybrid attention (linear + GQA):** Linear attention layers may behave differently under different `attention_backend` settings. FlashInfer may not support linear attention — triton might be required.
4. **NVFP4 on GB10/SM121:** MiniMax-M2.5-NVFP4 confirmed working on GB10. Qwen3.5 NVFP4 is untested on this specific hardware — possible ARM64/SM121 kernel incompatibility (seen in Qwen3.5-35B).
5. **mem_fraction_static=0.80:** ~234 GB weights → ~59 GB/GPU. KV cache budget = `128 × 0.80 - 59 = ~43 GB` per GPU — generous. May need tuning if CUDA graph capture or activation memory is higher than expected.
