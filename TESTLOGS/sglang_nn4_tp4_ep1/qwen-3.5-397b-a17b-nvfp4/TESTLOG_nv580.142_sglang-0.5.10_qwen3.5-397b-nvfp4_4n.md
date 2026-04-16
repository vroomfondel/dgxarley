# SGLang Test Log — Qwen3.5 397B-A17B NVFP4, 4 Nodes, TP=4 EP=1, v0.5.10

## Environment

| Component | Value                                              |
|-----------|----------------------------------------------------|
| GPU       | NVIDIA GB10 (SM121/Blackwell), 128 GB per node     |
| Driver    | 580.142                                            |
| CUDA      | 13.2                                               |
| Kernel    | 6.19.11-custom                                     |
| OS        | Ubuntu 24.04 LTS (aarch64)                         |
| K3s       | v1.35.3+k3s1                                       |
| Nodes     | spark1, spark2, spark3, spark4 (1 GPU each)        |
| Image     | `scitrera/dgx-spark-sglang:0.5.10`                 |
| Model     | `nvidia/Qwen3.5-397B-A17B-NVFP4`                   |
| NCCL      | 2.29.7+cuda13.2 (dgxspark-3node-ring)              |
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

| #  | nccl | moe_runner | attention | fp4_gemm   | dis_cuda_graph | dis_piecewise | Status       | n=1 tok/s | n=4 peak | n=8 peak  |
|----|------|------------|-----------|------------|----------------|---------------|--------------|-----------|----------|-----------|
| 1  | roce | triton     | fi        | fi_cutlass | false          | true          | **STABLE**   | 22.0      | 66.6     | 99.4      |
| 2  | roce | triton     | fi        | fi_cutlass | true           | true          | **STABLE**   | 14.2      | 62.9     | 95.2      |
| 3  | roce | triton     | fi        | fi_cutlass | false          | false         | **STABLE**   | 22.1      | 65.9     | 98.5      |
| 4  | roce | triton     | triton    | fi_cutlass | false          | true          | **STABLE**   | 22.2      | 67.2     | 101.3     |
| 5  | roce | triton     | triton    | fi_cutlass | true           | true          | **STABLE**   | 14.6      | 61.7     | 96.7      |
| 6  | roce | triton     | triton    | fi_cutlass | false          | false         | **STABLE**   | 21.6      | 67.0     | 100.2     |
| 7  | roce | triton     | fi        | fi_cudnn   | false          | true          | **STABLE**   | 21.9      | 65.6     | 100.2     |
| 8  | roce | triton     | fi        | fi_cudnn   | true           | true          | **STABLE**   | 14.7      | 60.8     | 92.6      |
| 9  | roce | triton     | fi        | fi_cudnn   | false          | false         | **STABLE**   | 20.4      | 64.2     | 97.5      |
| 10 | roce | triton     | triton    | fi_cudnn   | false          | true          | **STABLE**   | 21.3      | 65.0     | 98.0      |
| 11 | roce | triton     | triton    | fi_cudnn   | true           | true          | **STABLE**   | 14.3      | 62.6     | 94.0      |
| 12 | roce | triton     | triton    | fi_cudnn   | false          | false         | **STABLE**   | 21.5      | 64.7     | 96.0      |
| 13 | roce | fi_cutlass | fi        | fi_cutlass | false          | true          | FAIL†        | 16.5      | —        | —         |
| 14 | roce | fi_cutlass | fi        | fi_cutlass | true           | true          | FAIL†        | 16.3      | —        | —         |
| 15 | roce | fi_cutlass | fi        | fi_cutlass | false          | false         | FAIL†        | 0.7       | —        | —         |
| 16 | roce | fi_cutlass | triton    | fi_cutlass | false          | true          | FAIL†        | 9.4       | —        | —         |
| 17 | roce | fi_cutlass | triton    | fi_cutlass | true           | true          | FAIL†        | 18.0      | 39.5     | —         |
| 18 | roce | fi_cutlass | triton    | fi_cutlass | false          | false         | FAIL†        | 20.1      | 51.9     | —         |
| 19 | roce | fi_cutlass | fi        | fi_cudnn   | false          | true          | **STABLE**   | 20.1      | 62.6     | 84.2      |
| 20 | roce | fi_cutlass | fi        | fi_cudnn   | true           | true          | FAIL†        | 0.2       | —        | —         |
| 21 | roce | fi_cutlass | fi        | fi_cudnn   | false          | false         | FAIL†        | 11.9      | —        | —         |
| 22 | roce | fi_cutlass | triton    | fi_cudnn   | false          | true          | FAIL         | —         | —        | —         |
| 23 | roce | fi_cutlass | triton    | fi_cudnn   | true           | true          | FAIL†        | 17.6      | 51.2     | —         |
| 24 | roce | fi_cutlass | triton    | fi_cudnn   | false          | false         | **STABLE**   | 20.2      | 62.1     | 84.2      |
| 25 | roce | cutlass    | fi        | fi_cutlass | false          | true          | **STABLE**   | 20.8      | 67.1     | 100.8     |
| 26 | roce | cutlass    | fi        | fi_cutlass | true           | true          | **STABLE**   | 14.5      | 62.0     | 92.6      |
| 27 | roce | cutlass    | fi        | fi_cutlass | false          | false         | **STABLE**   | 21.2      | 66.3     | 99.6      |
| 28 | roce | cutlass    | triton    | fi_cutlass | false          | true          | **STABLE ★** | 21.5      | 67.8     | **102.0** |
| 29 | roce | cutlass    | triton    | fi_cutlass | true           | true          | **STABLE**   | 14.6      | 62.9     | 95.0      |
| 30 | roce | cutlass    | triton    | fi_cutlass | false          | false         | **STABLE**   | 21.3      | 67.2     | 101.8     |
| 31 | roce | cutlass    | fi        | fi_cudnn   | false          | true          | **STABLE**   | 22.0      | 66.3     | 99.7      |
| 32 | roce | cutlass    | fi        | fi_cudnn   | true           | true          | **STABLE**   | 14.4      | 60.8     | 92.6      |
| 33 | roce | cutlass    | fi        | fi_cudnn   | false          | false         | **STABLE**   | 22.0      | 66.1     | 99.7      |
| 34 | roce | cutlass    | triton    | fi_cudnn   | false          | true          | **STABLE**   | 20.8      | 66.9     | 100.6     |
| 35 | roce | cutlass    | triton    | fi_cudnn   | true           | true          | **STABLE**   | 13.8      | 62.7     | 95.4      |
| 36 | roce | cutlass    | triton    | fi_cudnn   | false          | false         | **STABLE**   | 19.9      | 67.4     | 100.5     |
| 37 | roce | triton     | triton    | fi_cutlass | false          | true          | **STABLE ★** | **35.3**  | **80.6** | **106.1** |

Test 37 uses **MTP speculative decoding** (NEXTN, `speculative_num_steps=3`, `speculative_num_draft_tokens=4`) with `mamba_scheduler_strategy=extra_buffer` + `SGLANG_ENABLE_SPEC_V2=1` (required for Qwen3.5 hybrid attention). `mem_fraction_static=0.75` (reduced from 0.80 for MTP KV headroom). Same `scitrera/dgx-spark-sglang:0.5.10` image.

### Column Legend

| Column         | Description                                                                                                                                                             |
|----------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| nccl           | `nccl_transport` — NCCL inter-node transport (`socket` = TCP/IP, `roce` = RDMA/RoCE via SR-IOV VF)                                                                      |
| moe_runner     | `moe_runner_backend` — MoE expert dispatch kernel (`fi_cutlass` = flashinfer_cutlass, `triton` = triton→cutlass_moe_fp4 fallback for NVFP4, `cutlass` = cutlass direct) |
| attention      | `attention_backend` — attention kernel (`fi` = FlashInfer, `triton` = Triton)                                                                                           |
| fp4_gemm       | `fp4_gemm_backend` — FP4 dense GEMM kernel (`fi_cutlass` = flashinfer_cutlass, `fi_cudnn` = flashinfer_cudnn)                                                           |
| dis_cuda_graph | `disable_cuda_graph` — true = eager mode, false = capture CUDA graphs                                                                                                   |
| dis_piecewise  | `disable_piecewise_cuda_graph` — true = only fixed-BS graphs, false = piecewise variable-length graphs                                                                  |
| n=1 tok/s      | Per-request throughput at concurrency 1                                                                                                                                 |
| n=4 peak       | Sum of per-request tok/s at concurrency 4                                                                                                                               |
| n=8 peak       | Sum of per-request tok/s at concurrency 8                                                                                                                               |

---

## Final Results — all 36 tests complete

### Winner: Test 28

**cutlass MoE + triton attention + fi_cutlass FP4 + CUDA graphs on**

| Concurrency | Peak tok/s |
|-------------|------------|
| n=1         | 21.5       |
| n=4         | **67.8**   |
| n=8         | **102.0**  |

### Top 5 configurations by n=8 peak

| Rank | #  | MoE     | Attn   | FP4        | CG        | n=1  | n=4  | n=8       |
|------|----|---------|--------|------------|-----------|------|------|-----------|
| 1    | 28 | cutlass | triton | fi_cutlass | on        | 21.5 | 67.8 | **102.0** |
| 2    | 30 | cutlass | triton | fi_cutlass | piecewise | 21.3 | 67.2 | 101.8     |
| 3    | 4  | triton  | triton | fi_cutlass | on        | 22.2 | 67.2 | 101.3     |
| 4    | 25 | cutlass | fi     | fi_cutlass | on        | 20.8 | 67.1 | 100.8     |
| 5    | 34 | cutlass | triton | fi_cudnn   | on        | 20.8 | 66.9 | 100.6     |

### Observations

- **No crashes on any `triton` or `cutlass` direct MoE backend (tests 1–12, 25–36).** All 24 configurations passed. This is the first successful triton/cutlass MoE matrix run on SM121 NVFP4 — the EP=1 configuration avoids the `StandardDispatcher` EP combine bug documented in `SGLANG_NVFP4_SHUFFLE_ROWS_OOB_UPSTREAM_BUG.md`.

- **`flashinfer_cutlass` MoE (tests 13–24) is heavily broken at EP=1.** 8/12 tests failed (FAIL†), 1 fully failed (no data at all), only 3 stable (tests 19, 24 with specific cudnn + no-graph combinations). The flashinfer_cutlass backend assumes EP>1 semantics — with EP=1 (single expert group), its dispatch/combine logic trips up.

- **cutlass direct MoE (tests 25–36) matches or beats triton MoE.** The direct cutlass backend is the overall winner (Test 28). Both `triton` and `cutlass` pipe through `cutlass_moe_fp4` at the kernel level, but cutlass-direct has less Python overhead and produces ~1% better peak throughput.

- **CUDA graphs ON delivers ~50% n=1 speedup** (22.2 vs 14.6 tok/s). The gap narrows to ~8-10% at n=4/n=8 as batching amortizes launch overhead. Piecewise graphs ≈ regular graphs within noise.

- **triton attn vs flashinfer attn:** Near-identical. triton attn marginally ahead at high concurrency.

- **fi_cutlass vs fi_cudnn FP4:** fi_cutlass ~1-2 tok/s better at n=8. Within noise for most configs.

### Comparison with previous baselines

| Config                                            | Matrix Winner n=1 | n=4      | n=8            |
|---------------------------------------------------|-------------------|----------|----------------|
| **Qwen3.5-397B EP=1 Test 28 (this matrix, RoCE)** | 21.5              | **67.8** | **102.0**      |
| Qwen3.5-397B EP=4 v0.5.10rc0 (socket)             | —                 | —        | — (100% crash) |
| Qwen3-235B-A22B EP=4 Test 17 (socket)             | 11.28             | 34.60    | 42.70          |
| Qwen3-235B-A22B EP=4 (RoCE, this session)         | ~22               | 65.4     | ~105           |

The 397B model with EP=1 + RoCE roughly **matches** the 235B model's throughput at RoCE. That's remarkable — 397B params active compute vs 235B, but comparable tok/s because both are at roughly the same "active compute per token" level (17B vs 22B active), and the TP=4 EP=1 topology on the 397B eliminates the per-layer EP all-reduce overhead that slows EP=4.
