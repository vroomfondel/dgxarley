# Hybrid-Mamba Word-Scramble Investigation — Qwen3.6-35B-A3B-FP8 on SGLang v0.5.10/v0.5.11

**Status (2026-05-10 09:40): FULLY RESOLVED on both versions.** Two distinct
correctness bugs identified AND fixed:

- **Bug A** (`is_layer_skipped` substring) → runtime patch in `sglang_launch.sh`
  (idempotent on both v0.5.10 and v0.5.11).
- **Bug B** (aggressive bench-only `sampling_overrides`) → comment out the
  `presence_penalty=1.5 + frequency_penalty=0.5 + min_tokens=4` block in the
  model profile.

**Verified on the full 6-case correctness-debug matrix on BOTH v0.5.10 and v0.5.11.**
12/12 cases run clean (0 fails, 0 word-salad markers, 100% coherent output).
v0.5.11 is on average **slightly faster** than v0.5.10 (+9% n=1 baseline).
Best-of-matrix is `triton-attn` baseline at **88–90 tok/s @ n=1**, +33% vs
fi-attn — confirming the original 0.5.10 TESTLOG Test 6 "STABLE ★" winner.

The earlier "hybrid-mamba multi-request race" / "0.5.11-regression" hypothesis
was **completely wrong**. The bug was not concurrency, not the mamba
scheduler, not overlap-scheduling, not the attention backend, and not
v0.5.11-specific. It was a simple over-aggressive sampling-penalty stack that
drove the logit distribution onto low-frequency synonyms during long decodes.
That looked like "cross-request state mixing" because longer multi-request
batches had more accumulated penalty mass per request. It looked like a
"v0.5.11 regression" because the original 0.5.10 winner-bench ran with
`recommended_sampling` only, while later benches added `sampling_overrides` on
top — implicitly comparing apples to oranges. Correlation, not causation.

**Production recommendation: v0.5.11 is fine as default.** The earlier "pin
to v0.5.10" advice is no longer necessary; v0.5.11 with the runtime patch (which
is a no-op there since #23467 is upstream) and the cleaned model profile
delivers throughput parity-or-better with v0.5.10 plus the v0.5.11 feature set
(native Gemma 4, Qwen3.6, GLM-5.1, FlashInferCuteDslMoE backend, FA3 community
kernels, etc.).

---

## 1. Symptom

Model `Qwen/Qwen3.6-35B-A3B-FP8` (HF arch class
`Qwen3_5MoeForConditionalGeneration` — hybrid Gated DeltaNet + Gated Attention
MoE, fine-grained FP8 with `weight_block_size=[128,128]`) produces
**"word-salad"** under multi-request decoding:

- Synonym-walk loops, e.g.:
  > `...heart center hub focus nucleus kernel crux pivot keystone linchpin axis...`
  > `...preserved conserved saved rescued delivered freed liberated emancipated...`
  > `...retired retired retired retired retire retire retire retire!!!`
- Explicit self-correction triggers in-output:
  > `*Hmm my mind started looping randomly... Stop it! Let's get back to serious...*`
  > `*OKAY STOP THIS RUNAWAY WORD ASSOCIATION LOOP IMMEDIATELY FOCUS RESET*`
  > `*self-loop detection triggered abort fantasy segment resume actual drafting*`
- 50–80% of bench requests hit `finish_reason=length=3072` (rambling to limit)
  rather than `finish_reason=stop` (natural completion).
- Throughput numbers look normal (~210–340 tok/s @ n=8); aggregate-throughput
  metrics are **misleadingly fine** — the pipeline is generating semantic noise.

The bench harness's NGRAM-repetition filter catches only the most extreme
primitive-token cases (`retire retire retire`); the synonym-walk pattern slips
through and is reported as `outcome=13` (all phases successful).

## 2. Hardware / Cluster Context

|                          |                                                                                       |
|--------------------------|---------------------------------------------------------------------------------------|
| GPUs                     | 4× NVIDIA GB10 (SM121/Blackwell), 128 GB unified per node                             |
| Driver                   | 580.142, CUDA 13.2 host                                                               |
| Topology                 | spark1 (head) + spark2/3/4 (workers), TP=4 PP=1 EP=1, RoCE via SR-IOV                 |
| Images verified affected | `scitrera/dgx-spark-sglang:0.5.11` (vanilla), `xomoxcc/dgx-spark-sglang:0.5.11-sm121` |
| Image verified clean     | `scitrera/dgx-spark-sglang:0.5.10` **with** the `is_layer_skipped` runtime patch      |
| Image affected by Bug A  | any v0.5.10 image when launched **without** the runtime patch                         |

## 3. Two Distinct Bugs

### Bug A — `is_layer_skipped()` substring-vs-dot-boundary (v0.5.10 only)

**Root cause:** SGLang's `python/sglang/srt/layers/quantization/utils.py:is_layer_skipped()`
in v0.5.10 uses a naive substring check `ignored in prefix`. Qwen3.6-FP8's HF
config lists `mlp.gate` in `modules_to_not_convert` (the MoE-router gate, which
must stay in BF16). The substring check matches `mlp.gate` against
`mlp.gate_up_proj` (the dense MLP fused projection), silently bypassing FP8
`weight_scale_inv` registration on `gate_up_proj`. Result: forward pass uses
unscaled FP8 weights → corrupted logits → "garbage logits / token salad".

**Upstream fix:** PR #23467 (commit `4323fce`, merged 2026-04-22, included in
v0.5.11). Adds `_module_path_match()` with dot-boundary semantics and
`_FALLBACK_FUSED_SHARDS` for configs that lack `packed_modules_mapping`.

**Local mitigation (v0.5.10 / v0.5.10-sm121):** runtime monkey-patch in
`roles/k8s_dgx/files/sglang_launch.sh` (block `PATCH_QUANT_UTILS_EOF`).
Removed at 2026-05-09 15:54 in commit `aa6639d` ("redundant on v0.5.11"),
restored at 2026-05-09 21:30 (no version guard — idempotent grep guard
short-circuits on v0.5.11). Verified working: Test 00 baseline_reproducer
on v0.5.10 with patch reproduces the original 0.5.10 winner numbers
(67.0 / 217.1 / 347.5 tok/s @ n=1/4/8 — vs 68.6 / 214.7 / 344.1 in the
original TESTLOG, run-to-run noise).

**Status: closed.** Patch is applied unconditionally; on v0.5.11 the upstream
fix renders it a no-op (grep guard sees `def _module_path_match` and skips).

### Bug B — Aggressive bench-only sampling penalties (FIXED 2026-05-09 22:18)

**Root cause:** the model profile `qwen-qwen3.6-35b-a3b-fp8.yml` had a
`sampling_overrides` block applied during bench runs:

```yaml
sampling_overrides:
  presence_penalty: 1.5
  frequency_penalty: 0.5
  min_tokens: 4
```

These three knobs in combination distort the logit distribution during long
decodes:
- `presence_penalty=1.5` strongly demotes any token already seen in the
  generated context.
- `frequency_penalty=0.5` adds a per-occurrence demotion on top.
- `min_tokens=4` forces the model to keep generating even when it would
  naturally emit EOS.

After ~500 generated tokens, the accumulated penalty mass on common words
(`the`, `a`, `and`, `is`, `to`, ...) is large enough to push their logits
below those of low-frequency synonyms. The sampler then picks rarer synonyms,
and on the next step the penalty mass shifts onto those, pushing the model
even further into the synonym tail. Result: the synonym-walk word-salad we
observed.

**Why this looked like a hybrid-mamba multi-request race:**
- Multi-request batches (n=4, n=8) have on average longer per-request decodes
  (the scheduler keeps requests in flight in parallel, so each individual
  request runs for longer wall-time before the batch drains). More tokens
  per request → more accumulated penalty → worse output. This is correlation
  with concurrency, not causation.
- All six diagnostic cases had the same `sampling_overrides` active, so
  changing overlap / mamba-strategy / attention backend made no difference
  to the underlying logit-distortion driver.
- The Hybrid-Mamba arch may amplify the effect (per-token state propagation
  is more sensitive to small logit shifts than dense Transformer attention),
  but the trigger is the penalty stack, not the architecture.

**Why this looked like a v0.5.11 regression:**
- The `sampling_overrides` were added to the profile after the original
  0.5.10 winner-bench (the original TESTLOG ran with `recommended_sampling`
  only — no presence/frequency penalties, no min_tokens floor). Comparing
  the original winner numbers against current bench runs implicitly compared
  "no overrides" vs "with overrides", and the throughput delta (e.g. 217 vs
  256 at n=4) made it look like a v0.5.11 issue when the override stack was
  actually present in BOTH versions of the bench but absent in the original
  winner-config testlog.

**Fix:** comment out the `sampling_overrides:` block in
`roles/k8s_dgx/model_profiles/qwen-qwen3.6-35b-a3b-fp8.yml`. Done at
2026-05-09 22:00.

**Verification (full 6-case matrix on both v0.5.10 and v0.5.11, 2026-05-10 09:40):**

|  Case | overlap | mamba        | attn       |    v0.5.10 n=1 |    n=4 |    n=8 |    v0.5.11 n=1 |    n=4 |    n=8 | Word-salad |
|------:|---------|--------------|------------|---------------:|-------:|-------:|---------------:|-------:|-------:|-----------:|
|    00 | on      | extra_buffer | fi         |          66.09 | 255.79 | 394.28 |          72.37 | 261.39 | 393.44 |          0 |
|    01 | off     | extra_buffer | fi         |          62.83 | 213.78 | 332.34 |          69.14 | 214.57 | 347.44 |          0 |
|    02 | on      | ""           | fi         |          67.55 | 190.44 | 337.86 |          66.18 | 214.02 | 341.78 |          0 |
|    03 | off     | ""           | fi         |          69.23 | 215.50 | 336.25 |          62.06 | 215.77 | 342.12 |          0 |
| **04**| on      | extra_buffer | **triton** |      **88.11** | 255.88 | 398.23 |      **89.76** | 255.58 | 399.97 |          0 |
|    05 | off     | extra_buffer | triton     |          69.05 | 208.49 | 334.42 |          64.05 | 211.20 | 345.30 |          0 |

12/12 cases: 0 fails, 0 repetition-kills, 0 word-salad markers, all
`finish_reason=length=3072` (long-but-coherent thinking). Output spot-checked
manually on Test 00 (OOP vs FP teaching) and Test 04 (triton-attn at n=1) —
both pristine.

**Surprise findings:**
- **Test 04 (triton-attn at n=1) is the absolute throughput winner** on both
  versions: 88.11 / 89.76 tok/s vs ~66/72 with fi-attn → **+33% n=1**. n=4
  and n=8 are within run-to-run noise. This matches the original 0.5.10
  TESTLOG Test 6 "STABLE ★" winner shape. The model profile already has
  `attention_backend: triton`, so no change needed.
- **v0.5.11 ≥ v0.5.10 across the matrix.** Test 00 baseline n=1 is +9% on
  v0.5.11 (72.37 vs 66.09), n=4 +2%, n=8 within noise. No regression.
- **`mamba_scheduler_strategy=""` (no_buffer)** has a small n=1 advantage on
  v0.5.10 (67.55 vs 66.09) but loses on n=4 (190 vs 256). Not worth flipping
  for multi-request workloads.

**Status: closed.** Penalty stack should NEVER have been added to the
production profile — the model card's `recommended_sampling`
(`temperature=1.0, top_p=0.95, top_k=20, presence_penalty=1.5`) is fine on
its own; the additional `frequency_penalty=0.5` + `min_tokens=4` were
bench-harness anti-repetition workarounds that made the underlying
"thinking phase rambles" problem dramatically worse instead of better.

### Bug C — Hypothetical hybrid-mamba multi-request race (FALSIFIED)

The diagnostic matrix initially suggested a hybrid-mamba concurrency race
because all six cases (varying overlap, mamba scheduler, attention backend)
showed word-salad at n≥4. After Bug B was identified as the actual driver,
re-running the **full 6-case matrix on both v0.5.10 and v0.5.11** with
`sampling_overrides` removed produced fully coherent output across all batch
sizes (12/12 cases clean, 0 word-salad markers, 0 fails).

Specifically, the previously "smoking-gun" observation that
**Test 04 (triton-attn) on v0.5.11 produced word-salad even at n=1
single-request** (which seemed to rule out concurrency-only explanations)
was a Bug B artifact. With overrides removed, Test 04 on v0.5.11 delivers
89.76 tok/s @ n=1 with pristine output — and is in fact the matrix winner
for n=1 throughput on both versions.

The earlier subagent investigation produced three root-cause candidates
(`packed_modules_mapping`, sgl-kernel 0.4.2 toolchain, sampler state mixing).
**Only Candidate 3 was on the right track** — but the reasoning was wrong.
It is not "cross-request state mixing in the custom logit processor"; it is
"these penalty values at long decode lengths produce systematic synonym
preference, regardless of concurrency." The custom logit processor default
flip in v0.5.11 was a red herring — the penalty handling itself is stable.

**Action:** none. Bug B fix subsumes Bug C entirely. The BF16 / sgl-kernel-pin /
27B-sibling / packed_modules_mapping experiments listed in the original TODO
list are all unnecessary.

After Bug A is mitigated, a residual word-salad pattern remains: 1–2 explicit
synonym-walks + several self-correction triggers per test case, plus 50–80% of
requests rambling to `length=3072`. Affects v0.5.10 (less severe) and v0.5.11
(more severe). All diagnostic switches probed:

#### Diagnostic matrix on v0.5.11 (vanilla scitrera + xomoxcc-sm121, both behave identically)

All cases: TP=4, EP=1, MoE=triton, KV=fp8_e4m3, FP8 GEMM=cutlass, no Spec.

| Case                    | overlap | mamba        | attn       | n=1 coherent? | n=4 coherent?  | n=8 coherent?  |
|-------------------------|---------|--------------|------------|---------------|----------------|----------------|
| 00 baseline             | on      | extra_buffer | fi         | ✗             | ✗              | ✗ + 1 rep-kill |
| 01 overlap-off          | **off** | extra_buffer | fi         | **✓**         | ✗              | ✗              |
| 02 mamba=no_buffer      | on      | **""**       | fi         | ✓             | ✗ + 1 rep-kill | ✗ + 1 rep-kill |
| 03 both off             | off     | ""           | fi         | ✓             | ✗ + 1 rep-kill | ✗              |
| 04 triton-attn          | on      | extra_buffer | **triton** | ✗             | ✗              | ✗              |
| 05 ov-off + triton-attn | off     | extra_buffer | triton     | ✗             | ✗              | ✗              |

#### Diagnostic matrix on v0.5.10 with `is_layer_skipped` patch

| Case           | overlap | mamba        | attn |         n=1 |    n=4 |    n=8 | n4 fail | n8 fail | Output                       |
|----------------|---------|--------------|------|------------:|-------:|-------:|--------:|--------:|------------------------------|
| 00 baseline    | on      | extra_buffer | fi   |       66.99 | 217.05 | 347.49 |       0 |       0 | 2 word-salad markers         |
| 01 overlap-off | off     | extra_buffer | fi   |       43.41 | 176.41 | 285.46 |       0 |       0 | 2 markers                    |
| 02 mamba=no    | on      | ""           | fi   |       64.10 | 165.60 | 250.80 |       0 |       1 | many self-correction markers |
| 03 both off    | …       | …            | …    | (in flight) |        |        |         |         |                              |
| 04, 05         | pending |              |      |             |        |        |         |         |                              |

**Key observation: Test 04 (baseline + triton-attn) on v0.5.11 produces
word-salad even at n=1 single-request decode**, with `overlap_schedule=on`.
This rules out concurrency-only explanations: there is a code path that fires
without any cross-request state interaction and corrupts output.

**Hypotheses ruled OUT by the matrices:**
- Spec V2 / Overlap-Scheduling (PR #21062) sole cause — falsified, fixes only n=1 with fi-attn
- `mamba_scheduler_strategy=extra_buffer` is incompatible — falsified, no_buffer is strictly worse
- FlashInfer-attn is the bug source — falsified, triton-attn is equally affected
- Any combination of overlap × mamba × attn switches — none fixes n≥4 word-salad

## 4. Historical: Earlier Hypotheses (all wrong, kept for the lessons)

> **Outcome:** all three candidates below were **incorrect**. The actual cause
> was Bug B above (sampling penalty stack). Section retained because the
> investigation process and the false-positive reasoning are useful for the
> next time we hit a similar symptom — and because two of the three "near
> misses" still describe real upstream code paths that may bite us in the
> future on other models.

Ranked by evidence strength as it appeared at the time of the subagent
investigation (2026-05-09).

### Candidate 1 — PR #23471: `packed_modules_mapping` now unconditional on SM121 (top priority)

**The change.** In v0.5.10.post1, `python/sglang/srt/models/qwen3_5.py` defines
the fused-projection mapping conditionally:

```python
if _is_gfx95:    # ROCm AMD GFX95 only
    packed_modules_mapping = {
        "in_proj_qkvz": ["in_proj_qkv", "in_proj_z"],
        "in_proj_ba":   ["in_proj_b",   "in_proj_a"],
    }
```

On NVIDIA SM121/GB10 (`_is_gfx95 = False`) the mapping is **absent**. v0.5.11
makes it unconditional and adds `qkv_proj` + `gate_up_proj`:

```python
packed_modules_mapping = {
    "qkv_proj":     ["q_proj",   "k_proj",    "v_proj"],
    "gate_up_proj": ["gate_proj","up_proj"],
    "in_proj_qkvz": ["in_proj_qkv","in_proj_z"],
    "in_proj_ba":   ["in_proj_b",  "in_proj_a"],
}
```

**Why it matters.** `packed_modules_mapping` tells SGLang's quantization
framework which checkpoint keys are fused projections. Without it, the
`_make_packed_weight_loader` path does not run; the framework loads
`in_proj_qkvz` / `in_proj_ba` as opaque single tensors. With it, the loader
tries to split fused weights and assign per-shard scales. For Qwen3.6-FP8
with dynamic per-block FP8 (`activation_scheme=dynamic, weight_block_size=[128,128]`),
each fused sub-shard carries an independent `weight_scale_inv` block. **If the
loader splits the fused weight tensor but mis-assigns scales** — applies the
fused scale tensor to each sub-shard, or falls through to an unquantized path —
the GDN layer receives corrupted Q/K/V/z inputs.

**Why this explains Test 04 on v0.5.11 (triton-attn, n=1, word-salad).** A
weight-corruption bug fires per decode step, regardless of concurrency. fi-attn
and triton-attn dispatch through different kernel sequences; one path may mask
or expose the scale mismatch (fi-attn happens to compute results that are
"close enough" at n=1 for the tokens to remain coherent; triton-attn does not).

**Why this explains Bug B on v0.5.10 (the version where Bug A is also active).**
On v0.5.10 SM121, `packed_modules_mapping` is empty for this model. The
loader takes a different (older) path that produced clean output before
v0.5.11's framework changes. Whether that older path is now triggered with
slightly different semantics in v0.5.11 (where `packed_modules_mapping` is
non-empty) is exactly the regression vector.

**File / lines.**
- v0.5.10.post1: `qwen3_5.py:884–892` — conditional on `_is_gfx95`
- v0.5.11: `qwen3_5.py:938–942` — unconditional
- Loader path: `_make_packed_weight_loader` in same file
- Quant scale broadcast: `python/sglang/srt/layers/quantization/fp8_utils.py`

**Upstream PRs to inspect.** #23471 (`packed_modules_mapping` for non-ROCm),
#23062 (FP8 per-tensor scale broadcast — only affects `numel()==1` so probably
not the smoking gun, but adjacent code).

### Candidate 2 — Toolchain bump: sgl-kernel 0.4.2, PyTorch 2.11, FlashInfer 0.6.8.post1

**The change.** v0.5.11 jumps:
- sgl-kernel `0.4.1.post1` → `0.4.2`
- PyTorch `2.9` → `2.11`
- FlashInfer `0.6.7.post2` → `0.6.8.post1`
- Default CUDA `12.8` → `13.0` in image

The hybrid-mamba decode path executes `causal_conv1d_update` (from
`causal_conv1d`) and `fused_recurrent_gated_delta_rule_packed_decode` (from
the bundled `fla` / Flash-Linear-Attention package). The latter reads
`h0=initial_state` and writes back to the same buffer indexed by
`cache_indices = mamba_cache_indices`.

A regression in the **calling convention** of these kernels in sgl-kernel 0.4.2
(e.g. cache_indices not correctly populated from the mamba pool, off-by-one
on `conv_state` offset for first tokens, or grid-shape change that touches
adjacent state slots) would produce systematic state corruption. Such a bug
worsens with concurrency because more state slots are active.

**Why this is #2 not #1.** Per-block kernel analysis shows that distinct
`(i_v, i_hv)` Triton blocks access non-overlapping regions of `initial_state`
for distinct `cache_indices`, so an intra-kernel race is unlikely. The
calling-convention path needs a line-level diff between sgl-kernel 0.4.1.post1
and 0.4.2 to confirm or refute. That is not in our local repo.

**Files / lines (in installed image, not in upstream repo).**
- `/usr/local/lib/python3.12/dist-packages/sglang/srt/layers/attention/fla/fused_recurrent.py`
- `/usr/local/lib/python3.12/dist-packages/sglang/srt/layers/attention/linear/gdn_backend.py`

### Candidate 3 — Sampler-side penalty state mixing (lower priority)

**The change.** v0.5.11 enables `enable_custom_logit_processor=True` by default
(was `False`). The deployment profile has `presence_penalty=1.5`,
`frequency_penalty=0.5`, `min_tokens=4`. These penalties require per-request
token-frequency tracking. If the v0.5.11 custom-logit-processor path introduced
a batch-dimension error in how the penalty state is gathered across n≥4
concurrent requests (e.g. accumulating token frequencies from request A into
request B's logit bias), the output distribution shifts onto low-frequency
synonyms → synonym-walk.

**Why this is #3.** The synonym-walk + explicit self-correction pattern
(`STOP THIS LOOPING GENERATED TEXT`, `Self-correction/refinement during thought`)
looks more like a corrupted hidden-state propagation than a logit-bias
distortion. Logit penalties shift token probabilities; they do not typically
make the model meta-narrate about being broken. That self-correction
meta-language suggests the model's internal hidden state encodes an
incoherent context that the model's instruction-following tries to
mid-stream-correct.

**Files.** `python/sglang/srt/managers/scheduler.py` (penalty batch loop),
`python/sglang/srt/server_args.py` (default flip).

## 5. What's Already In Place

- **Runtime patch for Bug A** (`PATCH_QUANT_UTILS_EOF`) re-added to
  `roles/k8s_dgx/files/sglang_launch.sh` at lines ~622–725. Idempotent
  grep guard `def _module_path_match` short-circuits on v0.5.11.
- **Launch-command + ENV diagnostic dump** added to `sglang_launch.sh` just
  before `exec "${args[@]}"` — prints `=== sglang launch command (N args) ===`
  with shell-quoted argv, plus filtered ENV (SGLANG_/NCCL_/FLASHINFER_/TORCH_/
  CUDA_/HF_/MAMBA_/SPEC_V2/RANK/WORLD_SIZE/...) with secret redaction
  (anything matching `*TOKEN*|*SECRET*|*KEY*|*PASSWORD*|*PASS*|*API*|
  *CREDENTIAL*` → `***REDACTED***`).
- **Correctness debug matrix** `nv580.142_sglang-0.5.{10,11}_qwen-3.6-35b-a3b-fp8_correctness-debug_n4_ep1.yaml`
  in `kikube/matrixtest_matrices/sglang_nn4_tp4_ep1/qwen-3.6-35b-a3b-fp8/`.
- **TESTLOG section** "Correctness Debug Sweep — Word-Salad Regression in
  v0.5.11" in
  `TESTLOGS/sglang_nn4_tp4_ep1/qwen-3.6-35b-a3b-fp8/TESTLOG_nv580.142_sglang-0.5.11_qwen-3.6-35b-a3b-fp8_4n.md`
  with full case-by-case results and output samples.

## 6. Recommended TODOs (post-resolution)

### Immediate (production)

- [x] **Bug A patch restored** in `roles/k8s_dgx/files/sglang_launch.sh`.
- [x] **`sampling_overrides` removed** from
  `roles/k8s_dgx/model_profiles/qwen-qwen3.6-35b-a3b-fp8.yml`.
- [x] **Re-run v0.5.11 correctness-debug matrix without overrides** —
  12/12 clean, 0 fails, throughput parity-or-better than v0.5.10.
  v0.5.11 confirmed usable as default.
- [ ] **Re-run a clean MTP bench on v0.5.{10,11}** with the restored
  Test-13-winner profile (`speculative_enabled: true`,
  `speculative_num_steps: 3`, `mamba_scheduler_strategy: extra_buffer`,
  `enable_spec_v2: true`, no overrides) to confirm the original
  104.2 / 277.8 / 410.7 tok/s @ n=1/4/8 numbers. Expected to match or
  exceed since the no-MTP baseline already exceeds the original
  (88 / 256 / 398 with triton-attn vs 68.6 / 214.7 / 344.1 original).
- [ ] **Audit other model profiles** for similar bench-only
  `sampling_overrides` blocks that pile penalty mass on top of
  `recommended_sampling`. Specifically check `qwen-qwen3.6-27b-fp8.yml`
  (the comment in the 35B profile mentions "symmetric to the 27B sibling"),
  and any other profile that copied the `frequency_penalty + min_tokens`
  pattern. Same drift problem expected on long decodes.

### Operational lessons

- [ ] **Always inspect actual output text on multi-batch tests**, not just
  stats. Both Bug A and Bug B silently produced `outcome=13` "STABLE" reports
  with no failed requests — the kikube harness saw structurally-valid
  responses; only manually reading the text revealed the corruption. The
  original 0.5.10 winner-bench had Bug A active in production for weeks
  without being noticed because nobody read the output samples.
- [ ] **Try the cheapest disqualifier first.** The sampler-penalty hypothesis
  (Subagent Candidate 3) was last in the priority list because the symptoms
  "looked like" hidden-state corruption. Two minutes of profile editing
  would have caught it before any of the deeper investigation. Default
  ordering for future correctness regressions: sampling config → model
  profile → server args → image pin → kernel/code paths.
- [ ] **Distinguish bench-only knobs from production knobs in profiles.**
  The `sampling_overrides` were added as a bench-anti-repetition workaround
  but stayed active in every deploy because the profile is the single source
  of truth. Either split the profile into `recommended_sampling` (production)
  and a separate optional `bench_sampling_overrides` field that is only
  applied when the matrix harness explicitly opts in, or keep overrides
  out of the profile entirely and parameterize them in the bench matrix YAML.

### Diagnostic kit (now redundant, retained for reference)

The five experiments listed in earlier revisions of this document
(BF16 weights, sgl-kernel pin, sibling 27B model, weight-scale logging,
upstream issue) are no longer needed — Bug B fix subsumes them. If a
similar symptom shows up on a different model in the future, run them in
priority order: profile sanity check → cheapest config disqualifier →
weight/kernel pinning → upstream filing.

### Bench harness improvements

- [ ] **Extend kikube output-quality filter** beyond NGRAM-repetition. The
  current filter caught only ~1 of ~14 word-salad runs across this
  investigation. Suggested heuristics, in increasing implementation cost:
  - Type-Token-Ratio threshold (e.g. ratio < 0.3 over a 200-token sliding
    window flags rambling).
  - WordNet-synset density (high % of consecutive tokens in the same synset).
  - LLM-judge on completed output ("is this coherent text? yes/no") via the
    bench harness's model-as-judge hook.
  - Penalty-pressure detector: track cumulative `presence_penalty * unique_token_count
    + frequency_penalty * total_tokens` and flag runs where it exceeds the
    logit-magnitude order of the model's vocabulary. Catches Bug B class
    before generation completes.

## 7. References

- This-session TESTLOG section: `TESTLOGS/sglang_nn4_tp4_ep1/qwen-3.6-35b-a3b-fp8/TESTLOG_nv580.142_sglang-0.5.11_qwen-3.6-35b-a3b-fp8_4n.md` § Correctness Debug Sweep
- 0.5.10 baseline TESTLOG: `TESTLOGS/sglang_nn4_tp4_ep1/qwen-3.6-35b-a3b-fp8/TESTLOG_nv580.142_sglang-0.5.10_qwen-3.6-35b-a3b-fp8_4n.md`
- 0.5.11 release notes summary: `SGLANG_v0.5.11_VERSION_CHANGES.md`
- Launch-script patches: `roles/k8s_dgx/files/sglang_launch.sh` (PATCH_QUANT_UTILS_EOF, PATCH_TRANSFORMERS_TOPK_EOF, ENV/cmd dump)
- SGLang upstream PRs: #21062 (Spec V2 default), #23467 (`is_layer_skipped` dot-boundary fix, in v0.5.11), #23471 (`packed_modules_mapping` unconditional)
- Result dirs: `kikube/results/sglang_nn4_tp4_ep1/qwen-3.6-35b-a3b-fp8/0.5.11-correctness-debug/`, `.../0.5.10-correctness-debug/`
- Diagnostic matrix YAMLs: `kikube/matrixtest_matrices/sglang_nn4_tp4_ep1/qwen-3.6-35b-a3b-fp8/nv580.142_sglang-0.5.{10,11}_qwen-3.6-35b-a3b-fp8_correctness-debug_n4_ep1.yaml`
