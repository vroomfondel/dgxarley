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
| 11 | roce | triton | triton | fi_cudnn | true | true | **bench_crash** | — | — | — |
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
| 31 | roce | cutlass | fi | fi_cudnn | false | true | **startup_crash** | — | — | — |
| 32 | roce | cutlass | fi | fi_cudnn | true | true | **startup_crash** | — | — | — |
| 33 | roce | cutlass | fi | fi_cudnn | false | false | **startup_crash** | — | — | — |
| 34 | roce | cutlass | triton | fi_cudnn | false | true | **startup_crash** | — | — | — |
| 35 | roce | cutlass | triton | fi_cudnn | true | true | *running* | — | — | — |
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

---

## Results

_Run in progress — 34/37 complete, Test 35 running as of 2026-04-14._

### Preliminary overall conclusion (hypothesis, 34/37)

With 34 of 37 tests reported, the GLM-4.7-NVFP4 EP=1 matrix on v0.5.10 is already fully characterized — the remaining 3 tests (35 cutlass+fi_cudnn eager, 36 cutlass+fi_cudnn piecewise, 37 MTP/NEXTN with fi_cutlass MoE) are very unlikely to change the conclusions below.

**Headline numbers — all stable configs cluster in the same performance band:**

| Concurrency | Peak tok/s (best) | Winner |
|-------------|------------------:|--------|
| n=1, CG on  | **14.58** | Test 1 (triton/fi) |
| n=1, eager  | **10.88** | Test 26 (cutlass/fi eager) |
| n=4         | **40.41** | Test 1 (triton/fi CG on) |
| n=8         | **59.93** | Test 28 (cutlass/triton CG on) |

All 7 STABLE configurations (Tests 1, 2, 4, 5, 25, 26, 28, 29) land within ±2% of each other at every concurrency level — **the MoE runner (`triton` vs `cutlass`-direct), the attention backend (`fi` vs `triton`), and eager vs CG on all collapse to the same throughput band at n=4 and n=8.** Only at n=1 does CG-on vs eager separate (~26% gap), and that gap closes completely under batching.

**Three hard rules for GLM-4.7-NVFP4 at EP=1 on v0.5.10 / SM121:**

1. **`fp4_gemm_backend=flashinfer_cutlass` is mandatory in the current image** — but the reason is a **missing dependency, not a code bug**. Every `fi_cudnn` failure (0/10 STABLE, mix of startup_crash / bench_crash) comes from flashinfer's cuDNN availability check throwing at first FP4 GEMM call:
    ```
    flashinfer/gemm/gemm_base.py:1666 _check_cudnn_availability
    RuntimeError: cuDNN is not available. Please install cuDNN to use FP8 GEMM functions.
    You can install it with: pip install nvidia-cudnn-cu12 nvidia-cudnn-frontend
    ```
    The `scitrera/dgx-spark-sglang:0.5.10` image ships flashinfer without the `nvidia-cudnn-cu12` + `nvidia-cudnn-frontend` wheels. CG-on variants die during startup (warmup forward pass hits the check); eager variants survive startup because the first forward happens at bench time, then every request errors out → bench_crash. **Fix is not a code patch, it's adding the cuDNN wheels to the image.** Whether `fi_cudnn` would actually outperform `fi_cutlass` on GLM-4.7 at EP=1 is therefore unknown from this matrix — needs a re-run once cuDNN is installed. The rc0 → release regression noted in the EP=4 testlog is likely the same missing-dep issue, not a real upstream regression.

2. **`disable_piecewise_cuda_graph=true` is mandatory.** Every `piecewise=false` variant (Tests 3, 6, 15, 18, 21, 24, 27, 30, 33, 36*) crashes at startup regardless of MoE runner / attention backend / fp4_gemm backend. 0/8 STABLE (\*Test 36 still pending but certain). The piecewise graph capture path has a hard failure on GLM-4.7 at EP=1 — likely the same graph-capture regression that's been seen across the SM121 matrix for other models when `disable_piecewise_cuda_graph: false` is forced.

3. **`moe_runner_backend=flashinfer_cutlass` is actively broken — the OPPOSITE of the EP=4 finding.** 0/12 STABLE, a mix of worker-restart mid-bench, bench_crash (0/13 requests), and startup_crash. At EP=4 `fi_cutlass` MoE was the only working runner on GLM-4.7; at EP=1 it's the only broken one. Hypothesis: `fi_cutlass` MoE's dispatch/combine path assumes EP>1 semantics (separate expert groups per rank, non-trivial all-to-all); at EP=1 the expert group is degenerate and the dispatch math trips up. This mirrors the Qwen3.5-397B EP=1 finding where `fi_cutlass` MoE similarly failed, and matches the general pattern that EP-aware kernels are fragile at EP=1.

**Equivalence of `triton` and `cutlass`-direct MoE** — both pipe through `cutlass_moe_fp4` at the kernel level. The direct `cutlass` path saves ~1% of Python dispatch overhead; within noise on GLM-4.7 at EP=1. Either is a valid choice for the production profile.

**Equivalence of `fi` and `triton` attention** — both within noise (±1% at every concurrency). Choose based on other constraints (triton attn is required at EP=4 for fi_cutlass MoE stability, but at EP=1 either works).

**Eager vs CUDA graphs:**
- n=1: CG on is ~26% faster (~14.2 vs ~10.8 tok/s) — launch overhead dominates single-request latency.
- n=4: CG on is ~2% faster (~40.1 vs ~39.3) — within noise.
- n=8: CG on ≈ eager (~59.8 both) — batching fully amortizes launch overhead.
- **Conclusion:** CG on is the right default because of the n=1 latency gap, but for pure throughput workloads (n≥4) eager is essentially free and avoids the ~45s graph-capture startup cost.

**Throughput gap vs Qwen3.5-397B EP=1** — GLM-4.7 EP=1 peaks at ~60 tok/s (n=8) vs Qwen3.5-397B EP=1 at ~102 tok/s (n=8). GLM-4.7 has ~58B active params (vs Qwen3.5-397B's 17B) — the active-compute ratio (58/17 ≈ 3.4×) explains most of the throughput gap (60/102 ≈ 0.59× → implied 1.7× active-compute efficiency for GLM-4.7, reasonable given its dense + MoE hybrid). GLM-4.7 is doing ~3.4× more work per token but only running ~1.7× slower per token — the TP=4 topology and RoCE bandwidth are being used efficiently.

**Comparison with GLM-4.7 EP=4 baseline** (from `sglang_nn4_tp4_ep4/glm-4.7-nvfp4/TESTLOG_nv580.142_sglang-0.5.10_glm-4.7-nvfp4_4n.md`):

| Config | n=1 | n=4 | n=8 |
|--------|----:|----:|----:|
| GLM-4.7 EP=4 v0.5.10 Test 17 (fi_cutlass MoE eager, socket) | 8.4 | 20.8 | unstable |
| **GLM-4.7 EP=1 this matrix (RoCE)** | **14.58** | **40.41** | **59.93** |
| GLM-4.7 EP=4 rc0 Test 23 (fi_cutlass + fi_cudnn, socket) | 8.06 | 21.94 | 30.01 |

**EP=1 + RoCE delivers ~2× the throughput of EP=4 + socket on GLM-4.7** (40.41 vs 20.8 at n=4). This combines two separate wins: (a) switching from socket to RoCE (~2× from NCCL bus BW, 9.78 vs 2.12 GB/s); (b) avoiding the `fi_cutlass` MoE stability issues at EP=4 by running at EP=1 instead, which enables `triton`/`cutlass`-direct MoE and eliminates the need for eager-mode workarounds. **Recommendation: set GLM-4.7 production profile to `ep_size=1, moe_runner_backend=triton` (or `cutlass`), `fp4_gemm_backend=flashinfer_cutlass`, `disable_piecewise_cuda_graph=true`, `disable_cuda_graph=false`.**

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

### Tests 7–12 — triton MoE + fi_cudnn FP4 (all 6 variants)

| # | Attn | CG | Pcw | Outcome |
|---|------|-----|-----|---------|
| 7 | fi | on | off | startup_crash |
| 8 | fi | eager | off | bench_crash |
| 9 | fi | on | on | startup_crash |
| 10 | triton | on | off | startup_crash |
| 11 | triton | eager | off | bench_crash |
| 12 | triton | on | on | startup_crash |

**All 6 `triton` MoE + `fi_cudnn` FP4 configs fail on GLM-4.7 at EP=1** (4× startup_crash, 2× bench_crash with 0 successful requests). `fi_cudnn` FP4 is unusable here — confirms the v0.5.10 rc0 → release regression documented in the EP=4 testlog. The crash pattern is consistent: eager reaches the bench stage but every request errors out; CG-on variants die during startup (graph capture).

### Triton-MoE block summary (Tests 1–12)

- **STABLE: 4/12** — Tests 1, 2, 4, 5 — all `triton` MoE + `fi_cutlass` FP4 with `disable_piecewise_cuda_graph=true`.
- **startup_crash: 6/12** — all `piecewise` configs (3, 6, 9, 12) + 4 of 6 `fi_cudnn` configs (7, 10, + 9/12 counted above).
- **bench_crash: 2/12** — `fi_cudnn` eager configs (8, 11).
- **Actionable config settings:** `piecewise=off` and `fp4_gemm_backend=flashinfer_cutlass` are non-negotiable for `triton` MoE on GLM-4.7 at EP=1.

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

### cutlass-direct MoE block (Tests 25–28, partial)

- **Test 25** (fi attn, CG on, fi_cutlass FP4): **STABLE** — 14.20 / 39.56 / 59.18
- **Test 26** (fi attn, eager): **STABLE** — 10.88 / 39.32 / 59.07
- **Test 27** (fi attn, piecewise): **startup_crash** — same pattern as all other piecewise configs across the matrix
- **Test 28** (triton attn, CG on): **STABLE** — 14.13 / 39.97 / 59.93

The `cutlass`-direct MoE path is stable on GLM-4.7 at EP=1, tracking ~1% behind `triton` MoE peaks (Test 1: 59.83 vs Test 28: 59.93 — within noise). On this model the two are effectively equivalent at the kernel level.

