# SGLang Test Log — Qwen3.6 35B-A3B-NVFP4 (MoE), 4 Nodes, TP=4 EP=1, v0.5.12 (first NVFP4 contact)

## Environment

| Component | Value                                                                       |
|-----------|-----------------------------------------------------------------------------|
| GPU       | NVIDIA GB10 (SM121/Blackwell), 128 GB per node                              |
| Driver    | 580.159                                                                     |
| CUDA      | 13.2 host / 13.0 image (PR #21498)                                          |
| Kernel    | 6.17.0-1018-nvidia                                                          |
| OS        | Ubuntu 24.04 LTS (aarch64)                                                  |
| K3s       | v1.35.3+k3s1                                                                |
| Nodes     | spark1, spark2, spark3, spark4 (1 GPU each)                                 |
| Image     | `scitrera/dgx-spark-sglang:0.5.12`                                          |
| Model     | `RedHatAI/Qwen3.6-35B-A3B-NVFP4`                                            |
| NCCL      | 2.29.7+cuda13.2 (dgxspark-3node-ring)                                       |
| Transport | **RoCE** via SR-IOV VF                                                      |
| AllReduce | Legacy (both `SGLANG_USE_JIT_ALL_REDUCE=0` + `SGLANG_OPT_..._V2=0`)         |

Matrix file: `kikube/matrixtest_matrices/sglang_nn4_tp4_ep1/qwen-3.6-35b-a3b-nvfp4/nv580.159_sglang-0.5.12_qwen-3.6-35b-a3b-nvfp4_n4_ep1.yaml`

**First NVFP4 contact for this model** — no prior baseline. Direct A/B references go to the FP8 sibling:
- `TESTLOGS/sglang_nn4_tp4_ep1/qwen-3.6-35b-a3b-fp8/TESTLOG_nv580.159_sglang-0.5.12_qwen-3.6-35b-a3b-fp8_4n.md` (same driver/image, FP8 path).

NVFP4 candidate selection: `RedHatAI/Qwen3.6-35B-A3B-NVFP4` chosen over `unsloth/...`, `mmangkad/...`, `sakamakismile/...` — 2.24M HF downloads as of 2026-05-22, orders of magnitude the most-pulled NVFP4 variant for this model, highest likelihood of community-vetted packaging.

Toolchain delta vs FP8 sibling: identical (same image). NVFP4-relevant 0.5.12 changes:
- FlashInfer 0.6.8.post1 → 0.6.11.post1 (PRs #24452, #25129, #25310, #25335)
- sgl-kernel 0.4.2 → 0.4.2.post2 (#24457, #25326)
- PR #23590 (Cute-DSL FP4 GEMM reland) + PR #23745 (Cute-DSL NVFP4 quant kernels) — relevant for Block D (fi_cutedsl MoE).
- `SGLANG_OPT_FP8_WO_A_GEMM` default-on (#25181) — irrelevant on this NVFP4 model (FP4 path).
- Spec V2 fixes (#25204 frozen-KV, #24635 stuck-MTP DSA, #24662 breakable CG bs>1) — Block E (MTP cases).
- JIT Custom All-Reduce default-on (#24363) — explicitly disabled via `sglang_jit_allreduce=false` + dual env-var injection (`SGLANG_USE_JIT_ALL_REDUCE=0` + `SGLANG_OPT_USE_CUSTOM_ALL_REDUCE_V2=0`) to keep the collective path comparable across runs.

See `SGLANG_v0.5.12_VERSION_CHANGES.md` for the full release delta.

---

## Model Notes

- 35B total / 3B active **MoE** (Gated DeltaNet hybrid). **NVFP4** quantization (~10–12 GB on-device weights, vs ~35 GB for the FP8 variant).
- Architecture: 10 × (3 × (Gated DeltaNet → MoE) + 1 × (Gated Attention → MoE)) = 40 layers.
  - Gated DeltaNet: 32 V-heads, 16 QK-heads, head_dim=128.
  - Gated Attention: 16 Q-heads, 2 KV-heads, head_dim=256, RoPE dim=64.
  - 256 routed experts (top-8) + 1 shared = 9 active per token, expert intermediate=512.
- Native context 262 144 (extensible to ~1 010 000 via YaRN).
- HF arch class: `Qwen3_5MoeForConditionalGeneration` (inherits `Qwen3VLForConditionalGeneration`).
- VL-capable (vision encoder), we run text-only — no special flags.
- **NVFP4-specific**: weights are ~3× smaller than FP8 → expect significantly more headroom in the KV cache for long-context, and FP4 tensor-core MMA paths become available (`cutlass`, `flashinfer_cutlass`, `flashinfer_cutedsl` MoE runners that all crash on the FP8 variant).

## Why this matrix exists

- FP8 baseline is the throughput champion of the cluster: 0.5.12 winners ~402–406 tok/s n=8 (no-MTP) / 426.76 peak (MTP s=2). The question is whether NVFP4 — with FP4 tensor cores on GB10/SM121 plus the ~3× smaller memory footprint — can match or beat that, and what it does for KV-cache headroom.
- Unlike FP8 on this model, NVFP4 unlocks `cutlass` and `flashinfer_cutlass` MoE runners (FP8 path 6/6 crashes there with `'Fp8MoEMethod' object has no attribute 'runner'`) — fresh axis to sweep.
- First-contact validation, so the matrix is wider (42 cases vs 24 on FP8) — 4×{moe_runner} × 2×{attention} × 2×{fp4_gemm_backend} × 3×{CG variant} with `cutedsl × cuDNN-FP4` cut as a too-experimental cross-product, plus 6 MTP cases (2 anchors + 4 num_steps sweep).

## What changes vs the FP8 sibling matrix

1. **NVFP4 MoE runners now in play.** FP8 was stuck on `triton` (only one that worked); NVFP4 sweeps `triton`, `flashinfer_cutlass`, `cutlass`, `flashinfer_cutedsl` — three more MoE backends to characterize, the most important being `flashinfer_cutlass` (matching the SM121-default for most other NVFP4 models per CLAUDE.md).
2. **`fp4_gemm_backend` axis added.** New axis (`flashinfer_cutlass` vs `flashinfer_cudnn`) — independent of `moe_runner_backend`. Per CLAUDE.md: `flashinfer_cutlass` works almost everywhere; cuDNN-FP4 is the more experimental of the two.
3. **No FP8 → NVFP4 quality migration baseline.** TTR_min / coherence floors are unknown on NVFP4. Per-case output-quality check is mandatory.
4. **`cutlass_moe_fp4` known-bad on SM121 for most NVFP4 models** (CLAUDE.md note: cutlass-direct + eager produces `!`-token collapse; cutlass-direct only works for `nvidia/Qwen3.5-397B-A17B-NVFP4`). On Qwen3.6-35B-NVFP4 the SM121 default is `flashinfer_cutlass` — Block C (`cutlass` direct, Tests 25–36) will likely produce either crashes or word-salad in eager rows (26, 29, 32, 35). Validate before relying on any cutlass-direct number.
5. **`flashinfer_cutedsl` MoE expected to work on NVFP4.** On FP8 it pre-checks fail-fast with `Invalid quantization 'None'. FlashInfer CuteDSL MOE currently supports only: 'modelopt_fp4'.` — the path is NVFP4-designed. Block D (Tests 37–42) is its first real test on our cluster.
6. **MTP sweep mirrors FP8 0.5.12 winners.** Anchors (43–44) reuse FP8 Tests 13–14 shapes; num_steps sweep (45–48) reuses FP8 winner shape (Test 03 / Tests 21–24) for direct A/B. If a different no-MTP shape wins on NVFP4, the sweep should be re-run on the actual winner.

## Configuration Matrix

All tests use: `tp=4, pp=1, ep=1, nccl_transport=roce, kv_cache_dtype=fp8_e4m3, mem_fraction_static=0.50, disable_deep_gemm=true, context_length=262144, num_experts=256, enable_eplb=false` unless noted. NVFP4 → FP4 GEMM backend sweep instead of an FP8 one. `flashinfer_cutedsl × flashinfer_cudnn` cross-product cut (too-experimental).

### Block A — triton MoE (FP8-validated reliable baseline) — Tests 01–12

| #  | moe_runner | attention | fp4_gemm     | dis_cuda_graph | dis_piecewise | Status     | n=1 tok/s | n=4 peak | n=8 peak   |
|----|------------|-----------|--------------|----------------|---------------|------------|-----------|----------|------------|
| 01 | triton     | fi        | fi_cutlass   | false          | true          | ok         | 71.89     | 245.49   | 407.25     |
| 02 | triton     | fi        | fi_cutlass   | true           | true          | ok         | 15.07     | 84.94    | 169.44     |
| 03 | triton     | fi        | fi_cutlass   | false          | false         | ok⁺        | 75.43     | 244.98   | 353.06     |
| 04 | triton     | triton    | fi_cutlass   | false          | true          | ok         | 61.13     | 245.13   | 404.19     |
| 05 | triton     | triton    | fi_cutlass   | true           | true          | **in-prog**| —         | —        | —          |
| 06 | triton     | triton    | fi_cutlass   | false          | false         | TBD    | —         | —        | —        |
| 07 | triton     | fi        | fi_cudnn     | false          | true          | TBD    | —         | —        | —        |
| 08 | triton     | fi        | fi_cudnn     | true           | true          | TBD    | —         | —        | —        |
| 09 | triton     | fi        | fi_cudnn     | false          | false         | TBD    | —         | —        | —        |
| 10 | triton     | triton    | fi_cudnn     | false          | true          | TBD    | —         | —        | —        |
| 11 | triton     | triton    | fi_cudnn     | true           | true          | TBD    | —         | —        | —        |
| 12 | triton     | triton    | fi_cudnn     | false          | false         | TBD    | —         | —        | —        |

### Block B — flashinfer_cutlass MoE (crashed on FP8) — Tests 13–24

| #  | moe_runner | attention | fp4_gemm     | dis_cuda_graph | dis_piecewise | Status | n=1 tok/s | n=4 peak | n=8 peak |
|----|------------|-----------|--------------|----------------|---------------|--------|-----------|----------|----------|
| 13 | fi_cutlass | fi        | fi_cutlass   | false          | true          | TBD    | —         | —        | —        |
| 14 | fi_cutlass | fi        | fi_cutlass   | true           | true          | TBD    | —         | —        | —        |
| 15 | fi_cutlass | fi        | fi_cutlass   | false          | false         | TBD    | —         | —        | —        |
| 16 | fi_cutlass | triton    | fi_cutlass   | false          | true          | TBD    | —         | —        | —        |
| 17 | fi_cutlass | triton    | fi_cutlass   | true           | true          | TBD    | —         | —        | —        |
| 18 | fi_cutlass | triton    | fi_cutlass   | false          | false         | TBD    | —         | —        | —        |
| 19 | fi_cutlass | fi        | fi_cudnn     | false          | true          | TBD    | —         | —        | —        |
| 20 | fi_cutlass | fi        | fi_cudnn     | true           | true          | TBD    | —         | —        | —        |
| 21 | fi_cutlass | fi        | fi_cudnn     | false          | false         | TBD    | —         | —        | —        |
| 22 | fi_cutlass | triton    | fi_cudnn     | false          | true          | TBD    | —         | —        | —        |
| 23 | fi_cutlass | triton    | fi_cudnn     | true           | true          | TBD    | —         | —        | —        |
| 24 | fi_cutlass | triton    | fi_cudnn     | false          | false         | TBD    | —         | —        | —        |

### Block C — cutlass MoE (direct, was N/A on FP8) — Tests 25–36

| #  | moe_runner | attention | fp4_gemm     | dis_cuda_graph | dis_piecewise | Status | n=1 tok/s | n=4 peak | n=8 peak |
|----|------------|-----------|--------------|----------------|---------------|--------|-----------|----------|----------|
| 25 | cutlass    | fi        | fi_cutlass   | false          | true          | TBD    | —         | —        | —        |
| 26 | cutlass    | fi        | fi_cutlass   | true           | true          | TBD    | —         | —        | —        |
| 27 | cutlass    | fi        | fi_cutlass   | false          | false         | TBD    | —         | —        | —        |
| 28 | cutlass    | triton    | fi_cutlass   | false          | true          | TBD    | —         | —        | —        |
| 29 | cutlass    | triton    | fi_cutlass   | true           | true          | TBD    | —         | —        | —        |
| 30 | cutlass    | triton    | fi_cutlass   | false          | false         | TBD    | —         | —        | —        |
| 31 | cutlass    | fi        | fi_cudnn     | false          | true          | TBD    | —         | —        | —        |
| 32 | cutlass    | fi        | fi_cudnn     | true           | true          | TBD    | —         | —        | —        |
| 33 | cutlass    | fi        | fi_cudnn     | false          | false         | TBD    | —         | —        | —        |
| 34 | cutlass    | triton    | fi_cudnn     | false          | true          | TBD    | —         | —        | —        |
| 35 | cutlass    | triton    | fi_cudnn     | true           | true          | TBD    | —         | —        | —        |
| 36 | cutlass    | triton    | fi_cudnn     | false          | false         | TBD    | —         | —        | —        |

### Block D — flashinfer_cutedsl MoE (NVFP4-only design, was crash B on FP8) — Tests 37–42

| #  | moe_runner | attention | fp4_gemm     | dis_cuda_graph | dis_piecewise | Status | n=1 tok/s | n=4 peak | n=8 peak |
|----|------------|-----------|--------------|----------------|---------------|--------|-----------|----------|----------|
| 37 | fi_cutedsl | fi        | fi_cutlass   | false          | true          | TBD    | —         | —        | —        |
| 38 | fi_cutedsl | fi        | fi_cutlass   | true           | true          | TBD    | —         | —        | —        |
| 39 | fi_cutedsl | fi        | fi_cutlass   | false          | false         | TBD    | —         | —        | —        |
| 40 | fi_cutedsl | triton    | fi_cutlass   | false          | true          | TBD    | —         | —        | —        |
| 41 | fi_cutedsl | triton    | fi_cutlass   | true           | true          | TBD    | —         | —        | —        |
| 42 | fi_cutedsl | triton    | fi_cutlass   | false          | false         | TBD    | —         | —        | —        |

### Block E — MTP (NEXTN) anchors + winner-shape num_steps sweep — Tests 43–48

| #  | moe_runner | attention | fp4_gemm     | dis_cuda_graph | dis_piecewise | spec      | Status | n=1 tok/s | n=4 peak | n=8 peak |
|----|------------|-----------|--------------|----------------|---------------|-----------|--------|-----------|----------|----------|
| 43 | triton     | triton    | fi_cutlass   | false          | false         | NEXTN s=3 | TBD    | —         | —        | —        |
| 44 | triton     | fi        | fi_cutlass   | false          | false         | NEXTN s=3 | TBD    | —         | —        | —        |
| 45 | triton     | fi        | fi_cutlass   | false          | false         | NEXTN s=2 | TBD    | —         | —        | —        |
| 46 | triton     | fi        | fi_cutlass   | false          | false         | NEXTN s=3 | TBD    | —         | —        | —        |
| 47 | triton     | fi        | fi_cutlass   | false          | false         | NEXTN s=4 | TBD    | —         | —        | —        |
| 48 | triton     | fi        | fi_cutlass   | false          | false         | NEXTN s=5 | TBD    | —         | —        | —        |

### Column Legend

| Column         | Description                                                                                                                    |
|----------------|--------------------------------------------------------------------------------------------------------------------------------|
| moe_runner     | `moe_runner_backend` — `triton`, `flashinfer_cutlass` (`fi_cutlass`), `cutlass` (direct), `flashinfer_cutedsl` (`fi_cutedsl`)  |
| attention      | `attention_backend` — `fi` = FlashInfer, `triton` = Triton                                                                     |
| fp4_gemm       | `fp4_gemm_backend` — `fi_cutlass` = `flashinfer_cutlass`, `fi_cudnn` = `flashinfer_cudnn`                                      |
| dis_cuda_graph | `disable_cuda_graph` — true = eager, false = capture CUDA graphs                                                               |
| dis_piecewise  | `disable_piecewise_cuda_graph` — true = fixed-BS graphs only, false = piecewise variable-length graphs                         |
| spec           | speculative decoding — `NEXTN s=N` = MTP with `speculative_num_steps=N`, `eagle_topk=1`, `num_draft_tokens=N+1`                |

---

## Pre-run hypotheses

1. **Block A (triton MoE, Tests 01–12)**: most likely to all pass. The `fi_cutlass` FP4 GEMM (Tests 01–06) should be a stable baseline. The `fi_cudnn` FP4 GEMM (Tests 07–12) is the experimental axis — risk of CG-capture crash or output regression. Expect n=8 peak in the 350–450 tok/s ballpark — could go either way vs FP8 (FP4 tensor-core MMA is faster, but the FP4 dequant path has overhead).
2. **Block B (fi_cutlass MoE, Tests 13–24)**: should work on NVFP4 (the path that crashes on FP8 with `Fp8MoEMethod.runner` is now a real FP4 path). This is the CLAUDE.md SM121-default for most NVFP4 models — strong candidate for the overall winner. EP=1 sidesteps the `cutlass_moe_fp4` EP-assertion bug, and there's no MoE-in-eager case here that would trigger the `!`-token collapse outside of Block C.
3. **Block C (cutlass MoE direct, Tests 25–36)**: known-bad on SM121 for most NVFP4 models — `cutlass_moe_fp4` shared-mem and EP-assert crashes. EP=1 keeps the EP-assert bug latent, but eager rows (26, 29, 32, 35) are high-risk for `!`-collapse per CLAUDE.md. Even if they don't crash, output-quality check is the gate.
4. **Block D (fi_cutedsl MoE, Tests 37–42)**: first real test on our cluster. PR #23590 (Cute-DSL FP4 GEMM reland) + PR #23745 (Cute-DSL NVFP4 quant kernels) make this potentially viable. If it works, the eager case (38, 41) is worth a hard look — cutedsl is a re-implementation in DSL form, eager-mode bugs would be a new class of failure.
5. **Block E (MTP, Tests 43–48)**: anchors (43–44) tie back to FP8 0.5.12 Tests 13–14 (which landed in the 396–402 tok/s n=8 range). The num_steps sweep mirrors FP8 Tests 21–24 — on FP8 0.5.12 the sweet spot was s=2 (peak 426.76). NVFP4 mamba-state path is the same kernel as FP8 — expect similar shape, but absolute numbers depend entirely on how NVFP4 vs FP8 compute throughput nets out.
6. **Output-quality**: hybrid-mamba word-salad regression (0.5.11 FP8 issue) was fixed via the `0c2bdd4` profile patch (`is_layer_skipped` substring + `sampling_overrides={}`). The NVFP4 profile should inherit the same fix — verify by pattern-grep + TTR + tail-eyeball per case. NVFP4 weights can themselves degrade quality slightly vs FP8; baseline floor unknown.

---

## Results

**Matrix run in progress (started 2026-05-22 ~17:22 UTC+2).** 48 cases planned. 4 cases complete (Tests 01–04), Test 05 in flight. The remaining 43 cases still `TBD`.

### Completed cases so far

| #  | Config                                              | n=1 tok/s | n=4 peak | n=8 peak | n=8 avg/req | n=8 ok | Finish reasons      | n=8 TTR_min | Output      |
|----|-----------------------------------------------------|----------:|---------:|---------:|------------:|--------|---------------------|------------:|-------------|
| 01 | triton-moe + fi-attn + fi_cutlass-fp4, CG on        |     71.89 |   245.49 |   407.25 |       50.91 | 8/8    | length×8            |       0.691 | clean ✓     |
| 02 | …+ **cuda_graph off**                               |     15.07 |    84.94 |   169.44 |       21.18 | 8/8    | length×8            |       0.643 | clean ✓     |
| 03 | …+ **piecewise**                                    |     75.43 |   244.98 |   353.06 |       50.44 | 7/8    | length×6, stop×1    |       0.609 | **flag** ⚠  |
| 04 | triton-moe + **triton-attn** + fi_cutlass-fp4, CG on|     61.13 |   245.13 |   404.19 |       50.52 | 8/8    | length×8            |       0.642 | clean ✓     |

### Preliminary observations

1. **Test 01 n=8 peak 407.25 vs FP8-0.5.12 Test 01 peak 406.44**: NVFP4 essentially ties FP8 with the triton MoE runner — **no speedup despite ~3× smaller weights**. Consistent with the FP8 model already being compute-bound on the dense triton MoE path on GB10; the FP4 tensor-core MMA is not exercised when the MoE runner is `triton` (only the `fp4_gemm_backend` for the GEMM, not the MoE kernel). Real NVFP4 upside should land in Blocks B (`fi_cutlass`) and C (`cutlass`).
2. **Test 02 eager mode regression: 169.44 peak vs FP8-0.5.12 Test 02 198.09 → −14.5 %**. NVFP4 eager is slower than FP8 eager — the FP4 dequant overhead on the per-step Python path costs more than the tensor-core saves. Eager-mode is rarely the production choice on this model anyway, but worth noting as a baseline data point.
3. **Test 03 piecewise CG: throughput NOT regressed — 1/8 repetition-abort is a counting artifact.** Raw peak 353.06 vs Test 01's 407.25 looks like a −13 % regression, but the 7 successful requests each landed at 50.43–50.45 tok/s (tight cluster, same as Test 01's 50.91). Had `req 4` completed normally, sum would be ~8 × 50.43 = **~403** tok/s — essentially tying Test 01 (407.25). The piecewise-CG path runs the same speed as full-CG on NVFP4. What's real is the **1/8 repetition-abort on req 4** (Monty-Hall prompt): bench framework killed the request at TTFT with `status: repetition`. Visible head/tail of the response is clean (a legitimate enumeration table that may have fooled the naïve repetition detector), but the truncated middle 2.2 kchar is opaque. Plus **`req 2` finish_reason = stop** (the other Block A no-MTP cases finish length×8). Open question: real output-quality regression on the piecewise path, or bench-detector false-positive on a degenerate-looking table? Watch Tests 06/09/12 (other piecewise variants) — if any of them replicate the repetition flag, it's a real piecewise-CG NVFP4 quality issue.
4. **Output quality otherwise clean.** Tests 01/02 TTR_min ≥ 0.64, no word-salad pattern hits in the visible response text. The `0c2bdd4` profile fix (`is_layer_skipped` substring + `sampling_overrides={}`) is applied; nothing in the visible output suggests it's failing on NVFP4.
5. **n=8 throughput-vs-FP8 comparison will be a per-block discussion**, not a single number. Block A here suggests "matches FP8 in the best case (Test 01) and regresses on the variants" — Blocks B/C are where the FP4 tensor-core paths actually kick in.

(Will continue to fill the matrix as cases complete. Re-run via `kikube-bench matrix matrixtest_matrices/sglang_nn4_tp4_ep1/qwen-3.6-35b-a3b-nvfp4/nv580.159_sglang-0.5.12_qwen-3.6-35b-a3b-nvfp4_n4_ep1.yaml`.)

---

## Action items after the matrix run

- [ ] Fill the four block tables with actual results
- [ ] Verify output quality on every `ok` case (pattern-grep + token-distribution + tail-eyeball) — first NVFP4 contact, no prior quality floor
- [ ] Identify the no-MTP winner shape; if it diverges from the FP8 winner (triton-moe + fi-attn + piecewise CG), re-anchor the MTP sweep
- [ ] Compute Δ vs FP8 sibling on the corresponding 0.5.12 winners (Test 01 / 03 / 14 / 21 of FP8 ↔ closest NVFP4 shape)
- [ ] If `fi_cutlass` MoE (Block B) wins: candidate to flip the production profile (current FP8 profile uses triton MoE — see `model_profiles/Qwen--Qwen3.6-35B-A3B-FP8.yml`; an NVFP4 profile file would need to be added)
- [ ] If Block C `cutlass`-direct crashes or produces `!`-collapse: confirm CLAUDE.md note ("cutlass-direct only works for nvidia/Qwen3.5-397B-A17B-NVFP4") and document for this model explicitly
- [ ] If Block D `fi_cutedsl` works: this is the first datapoint on our cluster — log absolute n=8 peak and a TTR check for any DSL-vs-cutlass numerical divergence
