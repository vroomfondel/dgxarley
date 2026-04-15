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
| 7 | roce | triton | fi | fi_cudnn | false | true | **STABLE** ‡ | 14.33 | 39.60 | 59.56 |
| 8 | roce | triton | fi | fi_cudnn | true | true | **STABLE** ‡ | 10.40 | 38.86 | 58.63 |
| 9 | roce | triton | fi | fi_cudnn | false | false | **startup_crash** | — | — | — |
| 10 | roce | triton | triton | fi_cudnn | false | true | **STABLE** ‡ | 14.32 | 40.09 | 59.91 |
| 11 | roce | triton | triton | fi_cudnn | true | true | **STABLE** ‡ | 10.89 | 39.79 | 60.46 |
| 12 | roce | triton | triton | fi_cudnn | false | false | **startup_crash** | — | — | — |
| 13 | roce | fi_cutlass | fi | fi_cutlass | false | true | **worker_restart** | — | — | — |
| 14 | roce | fi_cutlass | fi | fi_cutlass | true | true | **worker_restart** | — | — | — |
| 15 | roce | fi_cutlass | fi | fi_cutlass | false | false | **startup_crash** | — | — | — |
| 16 | roce | fi_cutlass | triton | fi_cutlass | false | true | **worker_restart** | — | — | — |
| 17 | roce | fi_cutlass | triton | fi_cutlass | true | true | **bench_crash** | — | — | — |
| 18 | roce | fi_cutlass | triton | fi_cutlass | false | false | **startup_crash** | — | — | — |
| 19 | roce | fi_cutlass | fi | fi_cudnn | false | true | **startup_crash** | — | — | — |
| 20 | roce | fi_cutlass | fi | fi_cudnn | true | true | **bench_crash** | — | — | — |
| 21 | roce | fi_cutlass | fi | fi_cudnn | false | false | **startup_crash** | — | — | — |
| 22 | roce | fi_cutlass | triton | fi_cudnn | false | true | **startup_crash** | — | — | — |
| 23 | roce | fi_cutlass | triton | fi_cudnn | true | true | **bench_crash** | — | — | — |
| 24 | roce | fi_cutlass | triton | fi_cudnn | false | false | **startup_crash** | — | — | — |
| 25 | roce | cutlass | fi | fi_cutlass | false | true | **STABLE** | 14.20 | 39.56 | 59.18 |
| 26 | roce | cutlass | fi | fi_cutlass | true | true | **STABLE** | 10.88 | 39.32 | 59.07 |
| 27 | roce | cutlass | fi | fi_cutlass | false | false | **startup_crash** | — | — | — |
| 28 | roce | cutlass | triton | fi_cutlass | false | true | **STABLE** | 14.13 | 39.97 | 59.93 |
| 29 | roce | cutlass | triton | fi_cutlass | true | true | **STABLE** | 10.71 | 40.37 | 58.80 |
| 30 | roce | cutlass | triton | fi_cutlass | false | false | **startup_crash** | — | — | — |
| 31 | roce | cutlass | fi | fi_cudnn | false | true | **STABLE ★** ‡ | **14.51** | **40.60** | 60.03 |
| 32 | roce | cutlass | fi | fi_cudnn | true | true | **STABLE** ‡ | 10.43 | 39.01 | 58.54 |
| 33 | roce | cutlass | fi | fi_cudnn | false | false | **startup_crash** | — | — | — |
| 34 | roce | cutlass | triton | fi_cudnn | false | true | **STABLE** ‡ | 14.33 | 39.72 | 59.70 |
| 35 | roce | cutlass | triton | fi_cudnn | true | true | **STABLE ★** ‡ | 10.96 | 39.79 | **60.59** |
| 36 | roce | cutlass | triton | fi_cudnn | false | false | *pending* | — | — | — |
| 37 | roce | triton | triton | fi_cudnn | true | true | *pending (MTP, NEXTN k=3/4, needs cuDNN image)* | — | — | — |
| 38 | roce | cutlass | triton | fi_cudnn | true | true | *pending (MTP, NEXTN k=3/4, needs cuDNN image)* | — | — | — |

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
| ★ | Overall winner for that concurrency column |
| ‡ | Re-run with `xomoxcc/dgx-spark-sglang:0.5.10-cudnn` image (cuDNN wheels installed). Tests 7, 8, 10, 11, 31, 32, 34, 35 were originally startup/bench_crash on the upstream scitrera image due to a missing `nvidia-cudnn-cu12` dependency inside flashinfer; with the cuDNN wheels present these configs are fully stable — see "Overall conclusion" below for details. |

---

## Results

_Main matrix complete (36/37). Tests 7, 8, 10, 11, 31, 32, 34, 35 re-run on 2026-04-14 with `xomoxcc/dgx-spark-sglang:0.5.10-cudnn` image after identifying a missing cuDNN dependency as the root cause of the fi_cudnn failures. Test 37 (MTP) still pending on an edited YAML variant._

### Overall conclusion (36/37)

With 36 of 37 tests now accounted for — including the 8 fi_cudnn re-runs that unblocked a previously-unknown code path — the GLM-4.7-NVFP4 EP=1 matrix on v0.5.10 is fully characterized.

**Headline numbers — all 15 STABLE configs cluster in the same ±2% performance band:**

| Concurrency | Peak tok/s | Winning config |
|-------------|-----------:|----------------|
| n=1, CG on  | **14.51** | Test 31 (cutlass/fi/fi_cudnn, CG on) |
| n=1, eager  | **10.96** | Test 35 (cutlass/triton/fi_cudnn, eager) |
| n=4         | **40.60** | Test 31 (cutlass/fi/fi_cudnn, CG on) |
| n=8         | **60.59** | Test 35 (cutlass/triton/fi_cudnn, eager) |

All 15 STABLE configurations (Tests 1, 2, 4, 5, 7, 8, 10, 11, 25, 26, 28, 29, 31, 32, 34, 35) land within ±2% of each other at every concurrency level. **The MoE runner (`triton` vs `cutlass`-direct), the attention backend (`fi` vs `triton`), the FP4 GEMM backend (`fi_cutlass` vs `fi_cudnn`), and eager vs CG on all collapse to the same throughput band at n=4 and n=8.** Only at n=1 does CG-on vs eager separate (~25-30% gap, launch overhead dominates single-request latency), and that gap closes completely under batching. Interesting: the best n=8 configs are both `eager` (Tests 11 and 35 at ~60.5), marginally beating their CG-on twins — eager has slightly better n=8 tail behavior here.

**Two hard rules for GLM-4.7-NVFP4 at EP=1 on v0.5.10 / SM121:**

1. **`disable_piecewise_cuda_graph=true` is mandatory.** Every `piecewise=false` variant (Tests 3, 6, 15, 18, 21, 24, 27, 30, 33, 36) crashes at startup regardless of MoE runner / attention backend / FP4 GEMM backend — 0/10 STABLE. The piecewise graph capture path has a hard failure on GLM-4.7 at EP=1; likely the same graph-capture regression that's been seen across the SM121 matrix for other models when `disable_piecewise_cuda_graph: false` is forced.

2. **`moe_runner_backend=flashinfer_cutlass` is actively broken — the OPPOSITE of the EP=4 finding.** 0/12 STABLE, a mix of worker-restart mid-bench, bench_crash (0/13 requests), and startup_crash. At EP=4 `fi_cutlass` MoE was the only working runner on GLM-4.7; at EP=1 it's the only broken one. Hypothesis: `fi_cutlass` MoE's dispatch/combine path assumes EP>1 semantics (separate expert groups per rank, non-trivial all-to-all); at EP=1 the expert group is degenerate and the dispatch math trips up. This mirrors the Qwen3.5-397B EP=1 finding where `fi_cutlass` MoE similarly failed, and matches the general pattern that EP-aware kernels are fragile at EP=1. Neither the cuDNN fix nor any setting in this matrix changes this.

**`flashinfer_cudnn` FP4 GEMM — corrected finding:**

The original run had 0/10 STABLE on `fp4_gemm_backend=flashinfer_cudnn`, initially interpreted as a v0.5.10 regression. Root cause was actually a **missing Python dependency** — the upstream `scitrera/dgx-spark-sglang:0.5.10` image ships flashinfer without the `nvidia-cudnn-cu12` / `nvidia-cudnn-frontend` wheels, so flashinfer's runtime cuDNN availability check raises:

```
flashinfer/gemm/gemm_base.py:1666 _check_cudnn_availability
RuntimeError: cuDNN is not available. Please install cuDNN to use FP8 GEMM functions.
```

CG-on variants died during startup (warmup forward triggers the check); eager variants survived startup and crashed at first benchmark request with all-N-failed. With the cuDNN wheels added (`scripts/build_cudnn_image.sh` produces `xomoxcc/dgx-spark-sglang:0.5.10-cudnn`), **all 8 re-run configs are now STABLE** (Tests 7, 8, 10, 11, 31, 32, 34, 35 — the non-piecewise non-fi_cutlass-MoE subset).

Performance-wise, `fi_cudnn` is **within noise of `fi_cutlass`** (max delta ±1.8 tok/s, ≤3%). No clear winner in either direction; pick either based on other constraints. The rc0 → release "regression" documented in the EP=4 testlog (same 0-token symptom) is almost certainly the same missing-dep issue, not an upstream code regression — worth re-verifying with the cuDNN image.

**Equivalence of `triton` and `cutlass`-direct MoE** — both pipe through `cutlass_moe_fp4` at the kernel level. The direct `cutlass` path saves ~1% of Python dispatch overhead; within noise on GLM-4.7 at EP=1. Either is a valid choice for the production profile.

**Equivalence of `fi` and `triton` attention** — both within noise (±1% at every concurrency). At EP=4 `triton` attn was required for fi_cutlass MoE stability, but at EP=1 either works.

**Eager vs CUDA graphs:**
- n=1: CG on is ~30% faster (~14.3 vs ~10.7 tok/s) — launch overhead dominates single-request latency.
- n=4: CG on ≈ eager (both ~40 tok/s) — within noise.
- n=8: **eager marginally ahead** (60.5 vs 59.8) — the best n=8 configs are both eager (Tests 11, 35). Batching fully amortizes launch overhead and eager's lower capture cost pays off slightly.
- **Conclusion:** CG on is the right default for interactive / low-concurrency workloads because of the n=1 latency gap. For pure n≥4 throughput workloads, eager is at least as fast and avoids the ~45s graph-capture startup cost.

**Throughput gap vs Qwen3.5-397B EP=1** — GLM-4.7 EP=1 peaks at ~60.6 tok/s (n=8) vs Qwen3.5-397B EP=1 at ~102 tok/s (n=8). GLM-4.7 has ~58B active params vs Qwen3.5-397B's 17B — a 3.4× active-compute ratio explains most of the throughput gap (60/102 ≈ 0.59× ≈ implied 1.7× active-compute efficiency for GLM-4.7, reasonable given its dense + MoE hybrid). GLM-4.7 is doing ~3.4× more work per token but only running ~1.7× slower — TP=4 and RoCE are being used efficiently.

**Comparison with GLM-4.7 EP=4 baseline** (from `sglang_nn4_tp4_ep4/glm-4.7-nvfp4/TESTLOG_nv580.142_sglang-0.5.10_glm-4.7-nvfp4_4n.md`):

| Config | n=1 | n=4 | n=8 |
|--------|----:|----:|----:|
| GLM-4.7 EP=4 v0.5.10 Test 17 (fi_cutlass MoE eager, socket) | 8.4 | 20.8 | unstable |
| **GLM-4.7 EP=1 Test 31 (cutlass MoE + fi attn + fi_cudnn, CG on, RoCE)** | **14.51** | **40.60** | 60.03 |
| **GLM-4.7 EP=1 Test 35 (cutlass MoE + triton attn + fi_cudnn, eager, RoCE)** | 10.96 | 39.79 | **60.59** |
| GLM-4.7 EP=4 rc0 Test 23 (fi_cutlass + fi_cudnn, socket) | 8.06 | 21.94 | 30.01 |

**EP=1 + RoCE delivers ~2× the throughput of EP=4 + socket on GLM-4.7** (40.6 vs 20.8 at n=4, 60.6 vs 30.0 at n=8). This combines three wins: (a) socket → RoCE (~2× from NCCL bus BW, 9.78 vs 2.12 GB/s); (b) avoiding the `fi_cutlass` MoE stability issues at EP=4 by running at EP=1, which enables `triton`/`cutlass`-direct MoE; (c) installing the missing cuDNN dependency, which unblocks the `fi_cudnn` FP4 GEMM path and gives us a second-source FP4 backend for resilience.

**Production profile recommendation for GLM-4.7-NVFP4 on 4×GB10 SM121:**

```yaml
ep_size: 1
tp_size: 4
nccl_transport: roce
moe_runner_backend: cutlass       # or triton — within noise
attention_backend: fi             # or triton — within noise
fp4_gemm_backend: flashinfer_cutlass  # fi_cudnn works too with cudnn image
disable_piecewise_cuda_graph: true
disable_cuda_graph: false         # CG on for good n=1 latency; eager for max n=8
image: xomoxcc/dgx-spark-sglang:0.5.10-cudnn  # if fi_cudnn is ever selected
```

The cuDNN image is backwards-compatible (fi_cutlass still works), so there's no downside to using it as the default base going forward.

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

### Tests 7–12 — triton MoE + fi_cudnn FP4 (all 6 variants, after cuDNN image re-run)

| # | Attn | CG | Pcw | First run | After cuDNN re-run |
|---|------|-----|-----|-----------|--------------------|
| 7  | fi     | on    | off | startup_crash | **STABLE** 14.33 / 39.60 / 59.56 |
| 8  | fi     | eager | off | bench_crash | **STABLE** 10.40 / 38.86 / 58.63 |
| 9  | fi     | on    | on  | startup_crash | (not re-run — piecewise still broken) |
| 10 | triton | on    | off | startup_crash | **STABLE** 14.32 / 40.09 / 59.91 |
| 11 | triton | eager | off | bench_crash | **STABLE** 10.89 / 39.79 **60.46** |
| 12 | triton | on    | on  | startup_crash | (not re-run — piecewise still broken) |

Root cause of the first-run failures was the missing `nvidia-cudnn-cu12` wheel in the upstream image — flashinfer's `_check_cudnn_availability` raised `RuntimeError: cuDNN is not available` at first FP4 GEMM call. CG-on variants hit the check during warmup (→ startup_crash); eager variants reached the benchmark stage and failed every request (→ bench_crash with 0/N successful). With the cuDNN wheels added to the image, all 4 non-piecewise configs are stable and produce numbers within ±1.5 tok/s of their `fi_cutlass` FP4 twins.

### Triton-MoE block summary (Tests 1–12, after re-run)

- **STABLE: 8/12** — Tests 1, 2, 4, 5 (fi_cutlass FP4) + Tests 7, 8, 10, 11 (fi_cudnn FP4 after cuDNN fix). All non-piecewise `triton` MoE configs are stable.
- **startup_crash: 4/12** — all `piecewise=false` configs (3, 6, 9, 12). The piecewise crash is independent of FP4 backend.
- **Actionable config settings:** `disable_piecewise_cuda_graph=true` is non-negotiable for `triton` MoE on GLM-4.7 at EP=1. `fp4_gemm_backend=flashinfer_cutlass` or `flashinfer_cudnn` both work (with cuDNN image), performance identical within noise.

### fi_cutlass-MoE block summary (Tests 13–24)

**0/12 STABLE** — `flashinfer_cutlass` MoE is completely broken on GLM-4.7 at EP=1.

| # | Attn | FP4 | CG | Pcw | Outcome |
|---|------|-----|-----|-----|---------|
| 13 | fi | fi_cutlass | on | off | worker_restart (pod crash mid-bench) |
| 14 | fi | fi_cutlass | eager | off | worker_restart |
| 15 | fi | fi_cutlass | on | on | startup_crash |
| 16 | triton | fi_cutlass | on | off | worker_restart |
| 17 | triton | fi_cutlass | eager | off | bench_crash (0/13 requests) |
| 18 | triton | fi_cutlass | on | on | startup_crash |
| 19 | fi | fi_cudnn | on | off | startup_crash |
| 20 | fi | fi_cudnn | eager | off | bench_crash |
| 21 | fi | fi_cudnn | on | on | startup_crash |
| 22 | triton | fi_cudnn | on | off | startup_crash |
| 23 | triton | fi_cudnn | eager | off | bench_crash |
| 24 | triton | fi_cudnn | on | on | startup_crash |

This is the **reverse** of the GLM-4.7 EP=4 behavior: at EP=4, `fi_cutlass` MoE was the **only** working MoE runner. At EP=1 it crashes in every variant — 3× mid-bench worker restart with CG on + fi_cutlass FP4, plus the usual piecewise + fi_cudnn failures. The likely root cause is that `fi_cutlass` MoE's dispatch/combine assumes EP>1 semantics (per the Qwen3.5-397B EP=1 analysis on the same code) — with EP=1 (single expert group), its combine logic is trivially trivial and trips up. Needs log-level confirmation but the symptom is identical across 3 CG-on cases: inference runs briefly, then worker pod restarts mid-bench.

### cutlass-direct MoE block (Tests 25–36, after cuDNN re-run)

| # | Attn | FP4 | CG | Pcw | First run | After cuDNN re-run |
|---|------|-----|-----|-----|-----------|--------------------|
| 25 | fi     | fi_cutlass | on    | off | STABLE 14.20 / 39.56 / 59.18 | — |
| 26 | fi     | fi_cutlass | eager | off | STABLE 10.88 / 39.32 / 59.07 | — |
| 27 | fi     | fi_cutlass | on    | on  | startup_crash | — |
| 28 | triton | fi_cutlass | on    | off | STABLE 14.13 / 39.97 / 59.93 | — |
| 29 | triton | fi_cutlass | eager | off | STABLE 10.71 / 40.37 / 58.80 | — |
| 30 | triton | fi_cutlass | on    | on  | startup_crash | — |
| 31 | fi     | fi_cudnn   | on    | off | startup_crash | **STABLE ★** **14.51** / **40.60** / 60.03 |
| 32 | fi     | fi_cudnn   | eager | off | startup_crash | **STABLE** 10.43 / 39.01 / 58.54 |
| 33 | fi     | fi_cudnn   | on    | on  | startup_crash | (not re-run — piecewise still broken) |
| 34 | triton | fi_cudnn   | on    | off | startup_crash | **STABLE** 14.33 / 39.72 / 59.70 |
| 35 | triton | fi_cudnn   | eager | off | bench_crash | **STABLE ★** 10.96 / 39.79 / **60.59** |
| 36 | triton | fi_cudnn   | on    | on  | startup_crash | (not re-run — piecewise still broken) |

**8/12 STABLE** (Tests 25, 26, 28, 29 from the first run; Tests 31, 32, 34, 35 after the cuDNN re-run). All 4 remaining failures are `piecewise=true` configs — the same pattern as in the triton MoE block, independent of the FP4 backend.

**New overall winners** come from the cuDNN re-run:
- **Test 31** — cutlass MoE + fi attn + fi_cudnn FP4 + CG on: **best n=1 (14.51)** and **best n=4 (40.60)**.
- **Test 35** — cutlass MoE + triton attn + fi_cudnn FP4 + **eager**: **best n=8 (60.59)**.

The n=8 winner being an eager config is the most interesting finding of the re-run: batching fully amortizes CUDA graph launch overhead, and eager's lower capture cost pays off slightly. Test 11 (triton MoE + triton attn + fi_cudnn + eager) is a close second at 60.46, confirming the pattern across both MoE runners.

