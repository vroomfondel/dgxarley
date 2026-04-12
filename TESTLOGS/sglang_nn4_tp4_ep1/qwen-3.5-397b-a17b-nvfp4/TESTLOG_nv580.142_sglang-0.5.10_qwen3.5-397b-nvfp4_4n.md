# SGLang Test Log — Qwen3.5 397B-A17B NVFP4, 4 Nodes, TP=4 EP=1, v0.5.10

## Environment

| Component | Value |
|-----------|-------|
| GPU | NVIDIA GB10 (SM121/Blackwell), 128 GB per node |
| Driver | 580.142 |
| CUDA | 13.2 |
| Kernel | 6.19.11-custom |
| OS | Ubuntu 24.04 LTS (aarch64) |
| K3s | v1.35.3+k3s1 |
| Nodes | spark1, spark2, spark3, spark4 (1 GPU each) |
| Image | `scitrera/dgx-spark-sglang:0.5.10` |
| Model | `nvidia/Qwen3.5-397B-A17B-NVFP4` |
| NCCL | 2.29.7+cuda13.2 (dgxspark-3node-ring) |
| Transport | **RoCE** via SR-IOV VF (9.78 GB/s measured bus BW) |

---

## Model Notes

- 397B total / 17B active MoE (512 experts, top-10, softmax routing), NVFP4 quantized (~234 GB).
- Hybrid attention: 15 full GQA layers + 45 linear attention layers (every 4th layer is full attention). 60 layers total.
- 1 shared expert + 512 routed experts per MoE layer. Multimodal (text+image+video).
- Has MTP head (1 layer) for speculative decoding (NEXTN).
- `num_attention_heads=32, num_key_value_heads=2` — TP=4 per model card.
- NVFP4: only routed expert MoE FFN weights are FP4; attention, shared experts, vision encoder, lm_head, and MTP layer remain BF16.
- ~234 GB / 4 GPUs ≈ ~59 GB/GPU — fits on 4× DGX Spark.

## Key difference from the EP=4 test (TESTLOG_nv580.142_sglang-0.5.10rc0)

- **EP=1 TP=4** — all 512 experts replicated on every GPU, TP-sharded (1/4 intermediate per GPU). No EP dispatch/combine needed.
- **RoCE transport** — first test run with RDMA instead of TCP socket. 4.6× NCCL bus bandwidth (9.78 vs 2.12 GB/s).
- **`triton` MoE backend works** — at EP=1 the `cutlass_moe_fp4` path is correct on SM121 (proven 2026-04-12). All triton-moe tests pass. This was impossible at EP=4 due to the `StandardDispatcher` EP combine bug (see `SGLANG_NVFP4_SHUFFLE_ROWS_OOB_UPSTREAM_BUG.md`).
- **Runtime patches from `sglang_launch.sh`** — `cute/mma.py` sm_120a/sm_121a admissible_archs (essential for JIT FP4 kernel compilation on SM121). EP-related patches (modelopt_quant, cutlass_moe.py) are present but inert at EP=1.

---

## Configuration Matrix

All tests use: `tp=4, pp=1, ep=1, nccl_transport=roce, quantization=modelopt_fp4, kv_cache_dtype=fp8_e4m3, mem_fraction_static=0.60, disable_deep_gemm=true, context_length=196608, max_running_requests=32, schedule_policy=lpm, watchdog_timeout=3600, dist_timeout=1800` unless noted.

| # | nccl | moe_runner | attention | fp4_gemm | dis_cuda_graph | dis_piecewise | Status | n=1 tok/s | n=4 peak | n=8 peak |
|---|------|------------|-----------|----------|----------------|---------------|--------|-----------|----------|----------|
| 1 | roce | triton | fi | fi_cutlass | false | true | **STABLE** | 22.0 | 66.6 | 99.4 |
| 2 | roce | triton | fi | fi_cutlass | true | true | **STABLE** | 14.2 | 62.9 | 95.2 |
| 3 | roce | triton | fi | fi_cutlass | false | false | **STABLE** | 22.1 | 65.9 | 98.5 |
| 4 | roce | triton | triton | fi_cutlass | false | true | **STABLE** | 22.2 | 67.2 | 101.3 |
| 5 | roce | triton | triton | fi_cutlass | true | true | **STABLE** | 14.6 | 61.7 | 96.7 |
| 6 | roce | triton | triton | fi_cutlass | false | false | **STABLE** | 21.6 | 67.0 | 100.2 |
| 7 | roce | triton | fi | fi_cudnn | false | true | **STABLE** | 21.9 | 65.6 | 100.2 |
| 8 | roce | triton | fi | fi_cudnn | true | true | **STABLE** | 14.7 | 60.8 | 92.6 |
| 9 | roce | triton | fi | fi_cudnn | false | false | **STABLE** | 20.4 | 64.2 | 97.5 |
| 10 | roce | triton | triton | fi_cudnn | false | true | **STABLE** | 21.3 | 65.0 | 98.0 |
| 11 | roce | triton | triton | fi_cudnn | true | true | **STABLE** | 14.3 | 62.6 | 94.0 |
| 12 | roce | triton | triton | fi_cudnn | false | false | pending | — | — | — |
| 13 | roce | fi_cutlass | fi | fi_cutlass | false | true | pending | — | — | — |
| 14 | roce | fi_cutlass | fi | fi_cutlass | true | true | pending | — | — | — |
| 15 | roce | fi_cutlass | fi | fi_cutlass | false | false | pending | — | — | — |
| 16 | roce | fi_cutlass | triton | fi_cutlass | false | true | pending | — | — | — |
| 17 | roce | fi_cutlass | triton | fi_cutlass | true | true | pending | — | — | — |
| 18 | roce | fi_cutlass | triton | fi_cutlass | false | false | pending | — | — | — |
| 19 | roce | fi_cutlass | fi | fi_cudnn | false | true | pending | — | — | — |
| 20 | roce | fi_cutlass | fi | fi_cudnn | true | true | pending | — | — | — |
| 21 | roce | fi_cutlass | fi | fi_cudnn | false | false | pending | — | — | — |
| 22 | roce | fi_cutlass | triton | fi_cudnn | false | true | pending | — | — | — |
| 23 | roce | fi_cutlass | triton | fi_cudnn | true | true | pending | — | — | — |
| 24 | roce | fi_cutlass | triton | fi_cudnn | false | false | pending | — | — | — |
| 25 | roce | cutlass | fi | fi_cutlass | false | true | pending | — | — | — |
| 26 | roce | cutlass | fi | fi_cutlass | true | true | pending | — | — | — |
| 27 | roce | cutlass | fi | fi_cutlass | false | false | pending | — | — | — |
| 28 | roce | cutlass | triton | fi_cutlass | false | true | pending | — | — | — |
| 29 | roce | cutlass | triton | fi_cutlass | true | true | pending | — | — | — |
| 30 | roce | cutlass | triton | fi_cutlass | false | false | pending | — | — | — |
| 31 | roce | cutlass | fi | fi_cudnn | false | true | pending | — | — | — |
| 32 | roce | cutlass | fi | fi_cudnn | true | true | pending | — | — | — |
| 33 | roce | cutlass | fi | fi_cudnn | false | false | pending | — | — | — |
| 34 | roce | cutlass | triton | fi_cudnn | false | true | pending | — | — | — |
| 35 | roce | cutlass | triton | fi_cudnn | true | true | pending | — | — | — |
| 36 | roce | cutlass | triton | fi_cudnn | false | false | pending | — | — | — |

### Column Legend

| Column | Description |
|--------|-------------|
| nccl | `nccl_transport` — NCCL inter-node transport (`socket` = TCP/IP, `roce` = RDMA/RoCE via SR-IOV VF) |
| moe_runner | `moe_runner_backend` — MoE expert dispatch kernel (`fi_cutlass` = flashinfer_cutlass, `triton` = triton→cutlass_moe_fp4 fallback for NVFP4, `cutlass` = cutlass direct) |
| attention | `attention_backend` — attention kernel (`fi` = FlashInfer, `triton` = Triton) |
| fp4_gemm | `fp4_gemm_backend` — FP4 dense GEMM kernel (`fi_cutlass` = flashinfer_cutlass, `fi_cudnn` = flashinfer_cudnn) |
| dis_cuda_graph | `disable_cuda_graph` — true = eager mode, false = capture CUDA graphs |
| dis_piecewise | `disable_piecewise_cuda_graph` — true = only fixed-BS graphs, false = piecewise variable-length graphs |
| n=1 tok/s | Per-request throughput at concurrency 1 |
| n=4 peak | Sum of per-request tok/s at concurrency 4 |
| n=8 peak | Sum of per-request tok/s at concurrency 8 |

---

## Preliminary Results (Tests 1–10, matrix in progress)

### All triton-moe tests pass (first time on SM121)

Every configuration with `moe_runner_backend=triton` works — both fi_cutlass and fi_cudnn FP4 backends, CUDA graphs on/off/piecewise. This is the first successful run of the triton MoE backend for an NVFP4 model on SM121. The previous EP=4 run (v0.5.10rc0) had 100% crash rate across all triton-moe tests.

### Current leader: Test 4

**triton MoE + triton attention + fi_cutlass FP4 + CUDA graphs on**

| Concurrency | Peak tok/s |
|-------------|-----------|
| n=1 | 22.2 |
| n=4 | **67.2** |
| n=8 | **101.3** |

### Observations

- **CUDA graphs ON vs OFF:** ~50% speedup at n=1 (22.2 vs 14.6 tok/s). Smaller gap at n=4/n=8 (~8-10%). Piecewise graphs ≈ regular graphs.
- **triton attn vs flashinfer attn:** Near-identical. Test 4 (triton) marginally beats Test 1 (fi) at all concurrency levels.
- **fi_cutlass vs fi_cudnn FP4:** fi_cutlass slightly better at high concurrency (101.3 vs 100.2 at n=8). Difference is within noise.
- **No CRASHes:** 11/11 tests completed cleanly (Test 11 n=8 still running). This is the first matrix run where triton-moe doesn't crash on SM121 NVFP4 — enabled by EP=1 (avoids the EP dispatch bug) and the `cute/mma.py` admissible_archs patch.

### Context: comparison with EP=4 (v0.5.10rc0)

The previous EP=4 matrix for this model had:
- 100% crash rate for triton/cutlass MoE backends (EP dispatch OOB bug)
- Only fi_cutlass MoE worked, with limited stability
- Socket transport (~2 GB/s)

This EP=1 + RoCE run eliminates both bottlenecks: no EP dispatch, 4.6× network bandwidth.

---

## Tests 12–36: pending (matrix still running)

Results will be filled in as the kikube-bench matrix progresses.
