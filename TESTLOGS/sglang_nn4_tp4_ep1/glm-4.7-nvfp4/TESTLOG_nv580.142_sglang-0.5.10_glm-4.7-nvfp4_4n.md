# SGLang Test Log — GLM 4.7 NVFP4, 4 Nodes, TP=4 EP=1, v0.5.10

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
| Model | `nvidia/GLM-4.7-NVFP4` |
| NCCL | 2.29.7+cuda13.2 (dgxspark-3node-ring) |
| Transport | **RoCE** via SR-IOV VF (9.78 GB/s measured bus BW) |

Matrix file: `kikube/matrixtest_matrices/sglang_nn4_tp4_ep1/glm-4.7-nvfp4/nv580.142_sglang-0.5.10_glm-4.7-nvfp4_n4_ep1.yaml`

Previous test series: v0.5.10 EP=4 (`../../sglang_nn4_tp4_ep4/glm-4.7-nvfp4/TESTLOG_nv580.142_sglang-0.5.10_glm-4.7-nvfp4_4n.md`).

---

## Model Notes

- 358B total / ~58B active MoE (160 experts, top-8, sigmoid routing), NVFP4 quantized (~214 GB).
- GLM-4 MoE architecture: 92 layers (first 3 dense, rest MoE), standard GQA (num_kv_heads=8, 12:1 ratio).
- 1 shared expert + 160 routed experts per MoE layer.
- Has MTP head (1 layer) for speculative decoding (NEXTN).
- `num_attention_heads=96, num_key_value_heads=8` → TP=4 works (2 KV heads/GPU).
- NVFP4: only MoE FFN weights are FP4; attention projections, lm_head, and MTP layer remain BF16.
- ~214 GB / 4 GPUs ≈ ~54 GB/GPU — fits on 4× DGX Spark.

## Key difference from the EP=4 test (TESTLOG_nv580.142_sglang-0.5.10_glm-4.7-nvfp4_4n)

- **EP=1 TP=4** — all 160 experts replicated on every GPU, TP-sharded (1/4 intermediate per GPU). No EP dispatch/combine needed.
- **RoCE transport** — RDMA instead of TCP socket. 4.6× NCCL bus bandwidth (9.78 vs 2.12 GB/s).
- **`triton` and `cutlass` MoE expected to work** — at EP=1 the `cutlass_moe_fp4` path avoids the `StandardDispatcher` EP combine bug (proven at EP=1 for Qwen3.5-397B, 2026-04-12). The shared-memory / EP assertion crashes that plague GLM-4.7 at EP=4 should be sidestepped.
- **Runtime patches from `sglang_launch.sh`** — `cute/mma.py` sm_120a/sm_121a admissible_archs (essential for JIT FP4 kernel compilation on SM121). EP-related patches (modelopt_quant, cutlass_moe.py) are present but inert at EP=1.

---

## Configuration Matrix

All tests use: `tp=4, pp=1, ep=1, nccl_transport=roce, quantization=modelopt_fp4, kv_cache_dtype=fp8_e4m3, mem_fraction_static=0.60, disable_deep_gemm=true, context_length=202752, max_running_requests=32, schedule_policy=lpm, watchdog_timeout=3600, dist_timeout=1800, reasoning_parser=glm45, tool_call_parser=glm47` unless noted.

| # | nccl | moe_runner | attention | fp4_gemm | dis_cuda_graph | dis_piecewise | Status | n=1 tok/s | n=4 peak | n=8 peak |
|---|------|------------|-----------|----------|----------------|---------------|--------|-----------|----------|----------|
| 1 | roce | triton | fi | fi_cutlass | false | true | **STABLE** | 14.58 | 40.41 | 59.83 |
| 2 | roce | triton | fi | fi_cutlass | true | true | **STABLE** | 10.82 | 37.88 | 59.18 |
| 3 | roce | triton | fi | fi_cutlass | false | false | **startup_crash** | — | — | — |
| 4 | roce | triton | triton | fi_cutlass | false | true | **STABLE** | 14.28 | 40.11 | 59.71 |
| 5 | roce | triton | triton | fi_cutlass | true | true | **STABLE** | 10.75 | 39.66 | 59.80 |
| 6 | roce | triton | triton | fi_cutlass | false | false | **startup_crash** | — | — | — |
| 7 | roce | triton | fi | fi_cudnn | false | true | **startup_crash** | — | — | — |
| 8 | roce | triton | fi | fi_cudnn | true | true | **bench_crash** | — | — | — |
| 9 | roce | triton | fi | fi_cudnn | false | false | **startup_crash** | — | — | — |
| 10 | roce | triton | triton | fi_cudnn | false | true | **startup_crash** | — | — | — |
| 11 | roce | triton | triton | fi_cudnn | true | true | *running* | — | — | — |
| 12 | roce | triton | triton | fi_cudnn | false | false | *pending* | — | — | — |
| 13 | roce | fi_cutlass | fi | fi_cutlass | false | true | *pending* | — | — | — |
| 14 | roce | fi_cutlass | fi | fi_cutlass | true | true | *pending* | — | — | — |
| 15 | roce | fi_cutlass | fi | fi_cutlass | false | false | *pending* | — | — | — |
| 16 | roce | fi_cutlass | triton | fi_cutlass | false | true | *pending* | — | — | — |
| 17 | roce | fi_cutlass | triton | fi_cutlass | true | true | *pending* | — | — | — |
| 18 | roce | fi_cutlass | triton | fi_cutlass | false | false | *pending* | — | — | — |
| 19 | roce | fi_cutlass | fi | fi_cudnn | false | true | *pending* | — | — | — |
| 20 | roce | fi_cutlass | fi | fi_cudnn | true | true | *pending* | — | — | — |
| 21 | roce | fi_cutlass | fi | fi_cudnn | false | false | *pending* | — | — | — |
| 22 | roce | fi_cutlass | triton | fi_cudnn | false | true | *pending* | — | — | — |
| 23 | roce | fi_cutlass | triton | fi_cudnn | true | true | *pending* | — | — | — |
| 24 | roce | fi_cutlass | triton | fi_cudnn | false | false | *pending* | — | — | — |
| 25 | roce | cutlass | fi | fi_cutlass | false | true | *pending* | — | — | — |
| 26 | roce | cutlass | fi | fi_cutlass | true | true | *pending* | — | — | — |
| 27 | roce | cutlass | fi | fi_cutlass | false | false | *pending* | — | — | — |
| 28 | roce | cutlass | triton | fi_cutlass | false | true | *pending* | — | — | — |
| 29 | roce | cutlass | triton | fi_cutlass | true | true | *pending* | — | — | — |
| 30 | roce | cutlass | triton | fi_cutlass | false | false | *pending* | — | — | — |
| 31 | roce | cutlass | fi | fi_cudnn | false | true | *pending* | — | — | — |
| 32 | roce | cutlass | fi | fi_cudnn | true | true | *pending* | — | — | — |
| 33 | roce | cutlass | fi | fi_cudnn | false | false | *pending* | — | — | — |
| 34 | roce | cutlass | triton | fi_cudnn | false | true | *pending* | — | — | — |
| 35 | roce | cutlass | triton | fi_cudnn | true | true | *pending* | — | — | — |
| 36 | roce | cutlass | triton | fi_cudnn | false | false | *pending* | — | — | — |
| 37 | roce | fi_cutlass | triton | fi_cudnn | true | true | *pending (MTP, NEXTN k=3/4)* | — | — | — |

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

## Results

_Run in progress — 10/37 complete, Test 11 running as of 2026-04-14._

### Test 1 — triton MoE + flashinfer attn + fi_cutlass FP4, CUDA graphs on

- **STABLE** — all three concurrencies passed (0 failed requests).
- Peak tok/s: **14.58 / 40.41 / 59.83** (n=1/n=4/n=8, sum of per-request tok/s).
- Per-request at n=8: 8× ~7.48 tok/s (very even distribution).
- TTFT: 0.63s (n=1), 1.28s (n=4 p50), 1.38s (n=8 p50).
- First successful `triton` MoE run on GLM-4.7-NVFP4 at EP=1 — confirms the EP=1 topology avoids the `cutlass_moe_fp4` shared-memory / EP-assertion crashes that blocked all triton/cutlass MoE configs at EP=4.

### Test 2 — triton MoE + flashinfer attn + fi_cutlass FP4, eager (no CUDA graphs)

- **STABLE** — all three concurrencies passed.
- Peak tok/s: **10.82 / 37.88 / 59.18** (n=1/n=4/n=8).
- Eager costs ~26% at n=1 vs Test 1 (CG on), ~6% at n=4, and is within noise at n=8 — batching amortizes CUDA graph launch overhead.
- Noteworthy: unlike Qwen3.5-397B at EP=1, eager does **not** trigger `!`-token collapse here — the `cutlass_moe_fp4` eager-mode bug documented in CLAUDE.md applies specifically to the Qwen3.5-397B path. GLM-4.7 with `triton` MoE at EP=1 is stable in both eager and CG modes.

### Test 3 — triton MoE + flashinfer attn + fi_cutlass FP4, piecewise CUDA graphs

- **startup_crash** — worker-1 restarted once during startup, bench never began.
- Root cause pending log inspection.

### Test 4 — triton MoE + triton attn + fi_cutlass FP4, CUDA graphs on

- **STABLE** — 14.28 / 40.11 / 59.71 (n=1/n=4/n=8).
- Within noise of Test 1 (fi attn) — on GLM-4.7 the attention backend choice is throughput-neutral.

### Test 5 — triton MoE + triton attn + fi_cutlass FP4, eager

- **STABLE** — 10.75 / 39.66 / 59.80.
- Matches Test 2 (fi attn + eager) within noise. Second confirmation that eager is stable with `triton` MoE at EP=1 on GLM-4.7.

### Test 6 — triton MoE + triton attn + fi_cutlass FP4, piecewise CUDA graphs

- **startup_crash** — same symptom as Test 3 (piecewise variant).
- Pattern: both `piecewise` configs (Tests 3 and 6) crash at startup regardless of attention backend — suggests the `disable_piecewise_cuda_graph: false` path itself is broken for `triton` MoE + `fi_cutlass` FP4 on GLM-4.7 at EP=1.

### Test 7 — triton MoE + flashinfer attn + fi_cudnn FP4, CUDA graphs on

- **startup_crash** — first fi_cudnn config, crashed before bench.

### Test 8 — triton MoE + flashinfer attn + fi_cudnn FP4, eager

- **bench_crash** — server started, but all 13 benchmark requests (n=1 + n=4 + n=8) returned errors. 0 successful across all concurrencies.
- Together with Test 7, this points to a broader **`fp4_gemm_backend=flashinfer_cudnn` regression on GLM-4.7 at EP=1** — tracks the rc0 finding that fi_cudnn regressed in v0.5.10 (0 tokens), documented in the EP=4 testlog.

### Tests 9–10 — triton MoE + fi_cudnn FP4 (remaining variants)

- **Test 9** (`fi` attn, piecewise): **startup_crash**.
- **Test 10** (`triton` attn, CG on): **startup_crash**.
- Combined with Tests 7–8: **all 4 `triton` MoE + `fi_cudnn` FP4 configs tested so far fail on GLM-4.7 at EP=1** — 3× startup_crash, 1× bench_crash (0 successful requests). `fi_cudnn` FP4 is unusable here.

