# SGLang Test Log — Qwen3.6 35B-A3B-FP8 (MoE), 4 Nodes, TP=4 EP=1, v0.5.12

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
| Model     | `Qwen/Qwen3.6-35B-A3B-FP8`                                                  |
| NCCL      | 2.29.7+cuda13.2 (dgxspark-3node-ring)                                       |
| Transport | **RoCE** via SR-IOV VF                                                      |
| AllReduce | Legacy (both `SGLANG_USE_JIT_ALL_REDUCE=0` + `SGLANG_OPT_..._V2=0`)         |

Matrix file: `kikube/matrixtest_matrices/sglang_nn4_tp4_ep1/qwen-3.6-35b-a3b-fp8/nv580.159_sglang-0.5.12_qwen-3.6-35b-a3b-fp8_n4_ep1.yaml`

Previous testlog: `TESTLOG_nv580.142_sglang-0.5.11_qwen-3.6-35b-a3b-fp8_4n.md` (driver 580.142, image 0.5.11). The 0.5.12 run bumps **driver 580.142 → 580.159** AND **image 0.5.11 → 0.5.12** simultaneously — Δ is not purely image-attributable. There is no cross-image 580.142-vs-580.159 baseline; if findings are ambiguous, a 0.5.11 re-run on 580.159 is needed to disambiguate.

Toolchain delta vs `_sglang-0.5.11_*` testlog:
- FlashInfer 0.6.8.post1 → 0.6.11.post1 (PRs #24452, #25129, #25310, #25335)
- sgl-kernel 0.4.2 → 0.4.2.post2 (#24457, #25326)
- DeepGEMM split out into its own `sgl-deep-gemm` wheel (#24268, #24348, #24385) — we run `disable_deep_gemm=true` anyway, so indirectly relevant.
- DeepEP swapped from `fzyzcjy` fork to `deepseek-ai/DeepEP@hybrid-ep` (#25113)
- `SGLANG_OPT_FP8_WO_A_GEMM` now default-on (#25181) — was opt-in on 0.5.11. **Touches every FP8 GEMM path in this matrix**; part of the 0.5.11 → 0.5.12 delta is attributable to this single switch.
- Fused SiLU+clamp+FP8 quant kernel (#24897) — FP8 MoE path.
- Spec V2: breakable CUDA-Graph for `bs > 1` (#24662) → MTP cases (13/14, 21–24) under n=4/n=8 concurrency.
- Spec V2: stuck-MTP on DSA-models fix (#24635), frozen-KV `bonus_tokens=None` crash fix (#25204) — both relevant for the hybrid-mamba MTP path.
- JIT Custom All-Reduce default-on (#24363) — **explicitly disabled on our side via `sglang_jit_allreduce=false` plus dual env-var injection (`SGLANG_USE_JIT_ALL_REDUCE=0` + `SGLANG_OPT_USE_CUSTOM_ALL_REDUCE_V2=0`)** so the collective path stays comparable across 0.5.11 ↔ 0.5.12. See TODO_0.5.12.md item 1.

See `SGLANG_v0.5.12_VERSION_CHANGES.md` for the full release delta.

---

## Model Notes

- 35B total / 3B active **MoE** (Gated DeltaNet hybrid). Fine-grained FP8 (block 128).
- Architecture: 10 × (3 × (Gated DeltaNet → MoE) + 1 × (Gated Attention → MoE)) = 40 layers.
  - Gated DeltaNet: 32 V-heads, 16 QK-heads, head_dim=128.
  - Gated Attention: 16 Q-heads, 2 KV-heads, head_dim=256, RoPE dim=64.
  - 256 routed experts (top-8) + 1 shared = 9 active per token, expert intermediate=512.
- Native context 262 144 (extensible to ~1 010 000 via YaRN).
- HF arch class: `Qwen3_5MoeForConditionalGeneration` (inherits `Qwen3VLForConditionalGeneration`).
- VL-capable (vision encoder), we run text-only — no special flags.

## What changes vs the 0.5.11 sweep

1. **`SGLANG_OPT_FP8_WO_A_GEMM` is default-on** (#25181). On 0.5.11 this was opt-in (default `0`); on 0.5.12 it flips to `1`. We leave the 0.5.12 default in place → all FP8 paths in the matrix run with the Weight-Only A-GEMM optimization path. If a regression surfaces, a sub-run with `SGLANG_OPT_FP8_WO_A_GEMM=0` override is the next step.
2. **JIT Custom All-Reduce is default-on in 0.5.12** (#24363) — but we explicitly set `0` via Ansible default (`sglang_jit_allreduce: false`), with both env-var names injected into head + worker. This keeps the collective path apples-to-apples with the 0.5.11 baseline.
3. **`flashinfer_cutedsl` MoE (Tests 15–20)** — on 0.5.11 this was an explicit FP4-only pre-check crash (`server_args.py:2975 _handle_moe_kernel_config`). PR #23590 (Cute-DSL FP4 GEMM reland) and PR #23745 (Cute-DSL NVFP4 quant kernels) were merged — the pre-check logic is likely unchanged, but re-validate. If FP8 still hits fail-fast crash B → expected.
4. **`fi_cutlass` MoE (Tests 7–12)** — previously 6/6 crash A (`'Fp8MoEMethod' object has no attribute 'runner'`). FlashInfer bump 0.6.8.post1 → 0.6.11.post1 + sgl-kernel bump 0.4.2 → 0.4.2.post2: re-check whether upstream patched the dispatcher gap. Bug tracked in `SGLANG_FP8_MOEMETHOD_FLASHINFER_CUTLASS_UPSTREAM_BUG.md`.
5. **MTP / Spec V2** (Tests 13–14, 21–24). On 0.5.11 MTP on hybrid-mamba was slower than no-MTP across the board (Test 03 winner = 402.62 tok/s @ n=8, MTP-best = 389.92 @ n=8). 0.5.12 brings two MTP fixes (#25204 frozen-KV bonus-tokens, #24635 stuck-MTP DSA) + breakable CG bs>1 (#24662). Hypothesis: the MTP regression on hybrid-mamba is not addressed (none of the three PRs target the mamba-state-update path directly), but the CG bs>1 fix might improve n=4/n=8 stability.
6. **Word-salad regression** on hybrid-mamba from the 0.5.11 sweep (see appendix of the 0.5.11 testlog): gone after `0c2bdd4` (`is_layer_skipped` substring fix + `sampling_overrides={}`). This matrix inherits the fixed profile; if the bug resurfaces on 0.5.12 despite the profile fix → per-case output-quality check is mandatory (pattern-grep + token-distribution + tail-eyeball, see `feedback_output_quality_evidence` memory).

## Configuration Matrix

All tests use: `tp=4, pp=1, ep=1, nccl_transport=roce, kv_cache_dtype=fp8_e4m3, mem_fraction_static=0.50, disable_deep_gemm=true, fp8_gemm_runner_backend=cutlass, context_length=262144, num_experts=256, enable_eplb=false` unless noted. FP8 → no FP4 sweep. `cutlass` MoE skipped (FP4-only).

| #  | moe_runner | attention | dis_cuda_graph | dis_piecewise | spec      | Status      | n=1 tok/s | n=4 peak | n=8 peak |
|----|------------|-----------|----------------|---------------|-----------|-------------|-----------|----------|----------|
| 1  | triton     | fi        | false          | true          | —         | ok          | 64.15     | 265.36   | 406.44   |
| 2  | triton     | fi        | true           | true          | —         | ok          | 21.45     | 105.65   | 198.09   |
| 3  | triton     | fi        | false          | false         | —         | pending     | —         | —        | —        |
| 4  | triton     | triton    | false          | true          | —         | pending     | —         | —        | —        |
| 5  | triton     | triton    | true           | true          | —         | pending     | —         | —        | —        |
| 6  | triton     | triton    | false          | false         | —         | pending     | —         | —        | —        |
| 7  | fi_cutlass | fi        | false          | true          | —         | pending     | —         | —        | —        |
| 8  | fi_cutlass | fi        | true           | true          | —         | pending     | —         | —        | —        |
| 9  | fi_cutlass | fi        | false          | false         | —         | pending     | —         | —        | —        |
| 10 | fi_cutlass | triton    | false          | true          | —         | pending     | —         | —        | —        |
| 11 | fi_cutlass | triton    | true           | true          | —         | pending     | —         | —        | —        |
| 12 | fi_cutlass | triton    | false          | false         | —         | pending     | —         | —        | —        |
| 13 | triton     | triton    | false          | false         | NEXTN s=3 | pending     | —         | —        | —        |
| 14 | triton     | fi        | false          | false         | NEXTN s=3 | pending     | —         | —        | —        |
| 15 | fi_cutedsl | fi        | false          | true          | —         | pending     | —         | —        | —        |
| 16 | fi_cutedsl | fi        | true           | true          | —         | pending     | —         | —        | —        |
| 17 | fi_cutedsl | fi        | false          | false         | —         | pending     | —         | —        | —        |
| 18 | fi_cutedsl | triton    | false          | true          | —         | pending     | —         | —        | —        |
| 19 | fi_cutedsl | triton    | true           | true          | —         | pending     | —         | —        | —        |
| 20 | fi_cutedsl | triton    | false          | false         | —         | pending     | —         | —        | —        |
| 21 | triton     | fi        | false          | false         | NEXTN s=2 | pending     | —         | —        | —        |
| 22 | triton     | fi        | false          | false         | NEXTN s=3 | pending     | —         | —        | —        |
| 23 | triton     | fi        | false          | false         | NEXTN s=4 | pending     | —         | —        | —        |
| 24 | triton     | fi        | false          | false         | NEXTN s=5 | pending     | —         | —        | —        |

### Column Legend

| Column         | Description                                                                                                                    |
|----------------|--------------------------------------------------------------------------------------------------------------------------------|
| moe_runner     | `moe_runner_backend` — `triton`, `flashinfer_cutlass` (`fi_cutlass`), `flashinfer_cutedsl` (`fi_cutedsl`, PR #21339, FP4-only) |
| attention      | `attention_backend` — `fi` = FlashInfer, `triton` = Triton                                                                     |
| dis_cuda_graph | `disable_cuda_graph` — true = eager, false = capture CUDA graphs                                                               |
| dis_piecewise  | `disable_piecewise_cuda_graph` — true = fixed-BS graphs only, false = piecewise variable-length graphs                         |
| spec           | speculative decoding — `NEXTN s=N` = MTP with `speculative_num_steps=N`, `eagle_topk=1`, `num_draft_tokens=N+1`                 |

---

## Results

**Matrix run in progress (started 2026-05-21 17:59 UTC+2).** 2/24 cases completed.

Output-quality check per `ok` case (see `feedback_output_quality_evidence` memory):
1. pattern-grep for word-salad triggers (`retire retire`, `masterpiece masterpiece`, `STOP THIS LOOPING`, runs of `(\w+){3,}` repetition). `Self-Correction` removed from the trigger list after a false positive on a regular Qwen3.6-thinking reasoning marker (`*Self-Correction/Verification during output gen prep:*`).
2. token-distribution check (Type-Token-Ratio, reported as `ttr_min` across requests).
3. tail-eyeball of the last ~200 tokens per sample.

### Completed cases

| #  | Config                                                      |   n=1 | n=4 agg | n=4 per-req |  n=8 agg | n=8 per-req | Failures | Finish reasons         | n=8 TTR min | Output quality |
|----|-------------------------------------------------------------|------:|--------:|------------:|---------:|------------:|----------|------------------------|------------:|----------------|
| 01 | triton-moe + fi-attn, cuda_graph on, piecewise off          | 64.15 |  265.17 |       66.34 |   406.29 |       50.81 | 0/13     | length×13              |       0.650 | coherent ✓     |
| 02 | triton-moe + fi-attn, **cuda_graph off**, piecewise off     | 21.45 |  104.87 |       26.41 |   198.05 |       24.76 | 0/13     | length×12, stop×1      |       0.592 | coherent ✓     |

### Δ vs 0.5.11 baseline (driver 580.142)

| #  | 0.5.11 (n=1 / n=4 / n=8) | 0.5.12 (n=1 / n=4 / n=8)    | Δ n=1        | Δ n=4        | Δ n=8        |
|----|--------------------------|-----------------------------|-------------:|-------------:|-------------:|
| 01 | 76.77 / 254.78 / 396.26  | **64.15 / 265.36 / 406.44** | **−16.4 %**  | **+4.2 %**   | **+2.6 %**   |
| 02 | 22.64 / 107.12 / 209.91  | **21.45 / 105.65 / 198.09** | **−5.3 %**   | **−1.4 %**   | **−5.6 %**   |

### Findings so far (preliminary, 2/24)

1. **Test 01 n=8 +2.6 %** — within the predicted "+2…5 %" range from `SGLANG_OPT_FP8_WO_A_GEMM` default-on (#25181) + fused SiLU+clamp+FP8 quant (#24897). No AllReduce-default-flip leakage suspected (our `sglang_jit_allreduce=false` override is doing its job).
2. **Test 01 n=1 −16.4 %, TTFT 11.38 s vs 6.81 s on 0.5.11** — almost 2× higher TTFT. Strong indicator of **cold-cache effect** at matrix start: FlashInfer JIT autotune cache + Triton kernel cache empty on the very first request after pod spinup. To verify, compare with later same-shape cases (Tests 03, 04, 06) once they land — if their n=1 is also depressed, this is a real 0.5.12 regression; if they recover to ~76 tok/s, the n=1 hit is cache-warmup.
3. **Test 02 (eager) n=8 −5.6 %** — eager-mode regression. With CUDA graphs disabled, the new default `SGLANG_OPT_FP8_WO_A_GEMM` path does NOT compensate for whatever Python-side overhead grew between 0.5.11 and 0.5.12. n=4 is within noise (−1.4 %); n=8 is mildly worse. Eager mode is a diagnostic config, not production — acceptable.
4. **Output quality clean on both cases.** TTR_min ≥ 0.59 across all 26 sampled requests, no word-salad triggers. The Qwen3.6-thinking "Self-Correction/Verification" marker in coherent reasoning traces is **not** the bug from the 0.5.11 word-salad regression — that was synonym-walk loops + `retire retire retire` collapse, which is absent here.
5. **`mamba_usage: 0.02`** consistently in head decode logs — no hybrid-mamba KV-pool pressure at this context length and n=8.

---

## Baseline comparison (0.5.11, driver 580.142)

Winners from `TESTLOG_nv580.142_sglang-0.5.11_qwen-3.6-35b-a3b-fp8_4n.md`, for direct Δ calculation once 0.5.12 results land:

| #     | Config                                                     |   n=1 |    n=4 |        n=8 |
|-------|------------------------------------------------------------|------:|-------:|-----------:|
| 01    | triton-moe + fi-attn, cuda_graph on, piecewise off         | 76.77 | 254.78 |     396.26 |
| 02    | triton-moe + fi-attn, **cuda_graph off**, piecewise off    | 22.64 | 107.12 |     209.91 |
| 03    | triton-moe + fi-attn, cuda_graph on, **piecewise on**      | 71.14 | 261.70 | **402.62** |
| 04    | triton-moe + **triton-attn**, cuda_graph on, piecewise off | 77.34 | 254.90 |     400.56 |
| 06    | triton-moe + triton-attn, cuda_graph on, **piecewise on**  | 62.60 | 257.93 |     400.61 |
| 13    | triton-moe + triton-attn, piecewise on, **+MTP** (s=3)     | 84.09 | 250.25 |     373.76 |
| 14    | triton-moe + fi-attn, piecewise on, **+MTP** (s=3)         | 93.47 | 261.66 |     379.34 |
| 21    | winner shape + MTP s=2                                     | 79.49 | 261.69 |     389.92 |
| 23    | winner shape + MTP s=4                                     | 80.57 | 263.44 |     364.62 |
| 24    | winner shape + MTP s=5                                     | 57.67 | 221.55 |     339.21 |
| 07-12 | fi_cutlass × {fi, triton} × {CG on/off/piecewise}          |     — |      — |          — |  (6/6 **crash A**: `Fp8MoEMethod` has no `runner`)
| 15-20 | fi_cutedsl × {fi, triton} × {CG on/off/piecewise}          |     — |      — |          — |  (6/6 **crash B**: FP4-only pre-check)

**Expected delta hypotheses for 0.5.12** (pre-run):

1. **Tests 01–06 (no-MTP triton)**: slight speedup likely from `SGLANG_OPT_FP8_WO_A_GEMM` default-on (#25181) + fused SiLU+clamp+FP8 quant (#24897). Ballpark guess: +2…5 % at n=4/n=8 — if substantially more, the AllReduce default flip is the suspect (our `sglang_jit_allreduce=false` override should neutralise it, but worth re-checking the injected env vars on a running pod).
2. **Tests 07–12 (fi_cutlass MoE)**: likely still crash A. If now ok → upstream fix landed (possible, but not visible in the changelog).
3. **Tests 15–20 (fi_cutedsl MoE)**: likely still crash B. If now ok on FP8 → pre-check was loosened, but the path is NVFP4-designed — output-quality check would be critical.
4. **Tests 13–14, 21–24 (MTP)**: Spec V2 polish (#23456, #25204, #24635) + breakable CG bs>1 (#24662) might recover some of the n=4/n=8 MTP regression from 0.5.11 (−9 % at n=8 vs 0.5.10). Sweet spot probably still `s=2..3` for n=1, no-MTP still winner for n=8.
5. **Output quality**: word-salad should not resurface (the profile fix `0c2bdd4` is orthogonal to image version). Still — mandatory pattern check per case.

---

## Action items after the matrix run

- [ ] Fill the table with actual results
- [ ] Verify output quality on every `ok` case (pattern-grep + token-distribution + tail-eyeball)
- [ ] Compute Δ vs 0.5.11 (driver 580.142) — careful: mixed driver + image Δ; if findings are unambiguously pro-0.5.12, consider a 0.5.11 re-run on 580.159 to disambiguate
- [ ] Update the production recommendation in `model_profiles/Qwen--Qwen3.6-35B-A3B-FP8.yml` if warranted
- [ ] If fi_cutlass works now: update `SGLANG_FP8_MOEMETHOD_FLASHINFER_CUTLASS_UPSTREAM_BUG.md` status
- [ ] Sub-run with `sglang_jit_allreduce=true` (winner shape only) to quantify the V2 speedup in isolation
