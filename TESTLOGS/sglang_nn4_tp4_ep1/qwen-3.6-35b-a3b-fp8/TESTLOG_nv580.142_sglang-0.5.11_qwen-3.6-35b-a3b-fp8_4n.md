# SGLang Test Log — Qwen3.6 35B-A3B-FP8 (MoE), 4 Nodes, TP=4 EP=1, v0.5.11

## Environment

| Component | Value                                              |
|-----------|----------------------------------------------------|
| GPU       | NVIDIA GB10 (SM121/Blackwell), 128 GB per node     |
| Driver    | 580.142                                            |
| CUDA      | 13.2 host / 13.0 image (PR #21498)                 |
| Kernel    | 6.19.13-custom                                     |
| OS        | Ubuntu 24.04 LTS (aarch64)                         |
| K3s       | v1.35.3+k3s1                                       |
| Nodes     | spark1, spark2, spark3, spark4 (1 GPU each)        |
| Image     | `scitrera/dgx-spark-sglang:0.5.11`                 |
| Model     | `Qwen/Qwen3.6-35B-A3B-FP8`                         |
| NCCL      | 2.29.7+cuda13.2 (dgxspark-3node-ring)              |
| Transport | **RoCE** via SR-IOV VF                             |

Matrix file: `kikube/matrixtest_matrices/sglang_nn4_tp4_ep1/qwen-3.6-35b-a3b-fp8/nv580.142_sglang-0.5.11_qwen-3.6-35b-a3b-fp8_n4_ep1.yaml`

Toolchain delta vs `_sglang-0.5.10_*` testlog: PyTorch 2.9 → 2.11, CUDA 13 default,
sgl-kernel 0.4.1.post1 → 0.4.2, FlashInfer 0.6.7.post2 → 0.6.8.post1. Spec V2 with
Overlap-Scheduling is now baseline (PR #21062). New `flashinfer_cutedsl` MoE
backend (PR #21339) added in Tests 15–20. See `SGLANG_v0.5.11_VERSION_CHANGES.md`.

---

## Model Notes

- 35B total / 3B active **MoE** (Gated DeltaNet hybrid). Fine-grained FP8 (block 128).
- Architecture: 10 × (3 × (Gated DeltaNet → MoE) + 1 × (Gated Attention → MoE)) = 40 layers.
  - Gated DeltaNet: 32 V-heads, 16 QK-heads, head_dim=128.
  - Gated Attention: 16 Q-heads, 2 KV-heads, head_dim=256, RoPE dim=64.
  - 256 routed experts (top-8) + 1 shared = 9 active per token, expert intermediate=512.
- Native context 262 144 (extensible to ~1 010 000 via YaRN).
- HF arch class: `Qwen3_5MoeForConditionalGeneration` (inherits `Qwen3VLForConditionalGeneration`).
- VL-fähig (Vision-Encoder), wir fahren rein Text — keine speziellen Flags.

## What changes vs the 0.5.10 sweep

1. **`flashinfer_cutedsl` MoE runner is new** (Tests 15–20). On 0.5.10 only
   `triton`, `cutlass`, `flashinfer_cutlass` existed; `cutlass_moe_fp4` is
   FP4-only and `flashinfer_cutlass` was 6/6 startup_crash on FP8 due to
   `Fp8MoEMethod.runner` missing. Open question: does the new cutedsl backend
   work on FP8 weights? If yes, it's a fourth option besides `triton`.
2. **Spec V2 + Overlap-Scheduling is default** (PR #21062). MTP cases (13–14)
   should benefit from the lower per-step CPU overhead. The
   `mamba_scheduler_strategy=extra_buffer` + `enable_spec_v2=true` knobs are
   still required for hybrid-mamba radix-cache compat.
3. **FlashInfer 0.6.8.post1 + sgl-kernel 0.4.2** under the hood. Tests 7–12
   (fi_cutlass MoE) are re-runnable to see if the FP8 incompatibility
   (`Fp8MoEMethod has no attribute 'runner'`) was fixed by the lib bumps.

## Configuration Matrix

All tests use: `tp=4, pp=1, ep=1, nccl_transport=roce, kv_cache_dtype=fp8_e4m3, mem_fraction_static=0.50, disable_deep_gemm=true, fp8_gemm_runner_backend=cutlass, context_length=262144, num_experts=256, enable_eplb=false` unless noted. FP8 → no FP4 sweep. `cutlass` MoE skipped (FP4-only).

| #  | moe_runner       | attention | dis_cuda_graph | dis_piecewise | spec  | Status | n=1 tok/s | n=4 peak | n=8 peak |
|----|------------------|-----------|----------------|---------------|-------|--------|-----------|----------|----------|
| 1  | triton           | fi        | false          | true          | —     | ok     | 76.77     | 254.78   | 396.26   |
| 2  | triton           | fi        | true           | true          | —     | ok     | 22.64     | 107.12   | 209.91   |
| 3  | triton           | fi        | false          | false         | —     | tbd    | —         | —        | —        |
| 4  | triton           | triton    | false          | true          | —     | tbd    | —         | —        | —        |
| 5  | triton           | triton    | true           | true          | —     | tbd    | —         | —        | —        |
| 6  | triton           | triton    | false          | false         | —     | tbd    | —         | —        | —        |
| 7  | fi_cutlass       | fi        | false          | true          | —     | tbd    | —         | —        | —        |
| 8  | fi_cutlass       | fi        | true           | true          | —     | tbd    | —         | —        | —        |
| 9  | fi_cutlass       | fi        | false          | false         | —     | tbd    | —         | —        | —        |
| 10 | fi_cutlass       | triton    | false          | true          | —     | tbd    | —         | —        | —        |
| 11 | fi_cutlass       | triton    | true           | true          | —     | tbd    | —         | —        | —        |
| 12 | fi_cutlass       | triton    | false          | false         | —     | tbd    | —         | —        | —        |
| 13 | triton           | triton    | false          | false         | NEXTN | tbd    | —         | —        | —        |
| 14 | triton           | fi        | false          | false         | NEXTN | tbd    | —         | —        | —        |
| 15 | fi_cutedsl       | fi        | false          | true          | —     | tbd    | —         | —        | —        |
| 16 | fi_cutedsl       | fi        | true           | true          | —     | tbd    | —         | —        | —        |
| 17 | fi_cutedsl       | fi        | false          | false         | —     | tbd    | —         | —        | —        |
| 18 | fi_cutedsl       | triton    | false          | true          | —     | tbd    | —         | —        | —        |
| 19 | fi_cutedsl       | triton    | true           | true          | —     | tbd    | —         | —        | —        |
| 20 | fi_cutedsl       | triton    | false          | false         | —     | tbd    | —         | —        | —        |

### Column Legend

| Column         | Description |
|----------------|-------------|
| moe_runner     | `moe_runner_backend` — `triton`, `flashinfer_cutlass` (`fi_cutlass`), or **new** `flashinfer_cutedsl` (`fi_cutedsl`, PR #21339) |
| attention      | `attention_backend` — `fi` = FlashInfer, `triton` = Triton |
| dis_cuda_graph | `disable_cuda_graph` — true = eager, false = capture CUDA graphs |
| dis_piecewise  | `disable_piecewise_cuda_graph` — true = only fixed-BS graphs, false = piecewise variable-length graphs |
| spec           | speculative decoding (`NEXTN` = MTP, num_steps=3, eagle_topk=1, num_draft_tokens=4 + extra_buffer + spec_v2) |

---

## Results

**Run in progress (started 2026-05-10 ~09:55 UTC, driver on elite800)** — Cases 01 and 02 complete, Cases 03–20 pending.

Result dir: `kikube/matrixtest/2026-05-10/results/sglang_nn4_tp4_ep1/qwen-3.6-35b-a3b-fp8/0.5.11/`.

Image: `scitrera/dgx-spark-sglang:0.5.11` (vanilla upstream, **not** the `xomoxcc/...sm121` build that produced the word-salad reproducer in the correctness debug sweep below).

Re-run unblocked by the two correctness fixes from commit `0c2bdd4` ("Fixed two correctness bugs in Qwen3.6-35B-A3B-FP8 on SGLang v0.5.10/v0.5.11"):
- **Bug A:** `is_layer_skipped` substring match in the FP8 quantizer
- **Bug B:** aggressive `sampling_overrides` in the model profile (now `{}` as in the matrix above)

### Completed cases

| #  | Config                                | n=1   | n=4 agg | n=4 per-req | n=8 agg | n=8 per-req | Failures | Finish reasons   | Output quality |
|----|---------------------------------------|------:|--------:|------------:|--------:|------------:|----------|------------------|----------------|
| 01 | triton-moe + fi-attn (cuda_graph on)  | 76.77 |  254.78 |       63.74 |  396.26 |       49.56 | 0/13     | length×13        | coherent ✓     |
| 02 | + `disable_cuda_graph=true`           | 22.64 |  107.12 |       26.79 |  209.91 |       26.42 | 0/13     | length×12, stop×1| coherent ✓     |

- TTFT n=1: 6.81 s (case 01) → 11.55 s (case 02). Eager mode (no CUDA graphs) costs ~3.4× per-request throughput at n=1, ~2.4× at n=4, ~1.9× at n=8.
- Sample outputs across both cases follow the same `Here's a thinking process: 1. Analyze user input ...` reasoning pattern with no synonym-walk loops, self-correction triggers, or `retire retire retire` repetition. **The 0.5.11 word-salad regression documented in the correctness debug sweep below is no longer reproducible** with the post-`0c2bdd4` profile.

### Comparison to 0.5.10 baseline

Reference winners from `TESTLOG_nv580.142_sglang-0.5.10_qwen-3.6-35b-a3b-fp8_4n.md`:

| Config | n=1 | n=4 | n=8 |
|--------|----:|----:|----:|
| Test 6 (triton MoE + triton attn + piecewise on, no MTP) | 69.0 | 212.0 | 345.8 |
| Test 13 (triton MoE + triton attn + piecewise on + MTP) — winner | **104.2** | **277.8** | **410.7** |

After the 0.5.11 run, populate the table above and write a short delta vs 0.5.10
section here (toolchain bump impact + cutedsl viability + MTP under default
Spec V2 + Overlap-Scheduling).

---

## Correctness Debug Sweep — Word-Salad Regression in v0.5.11 (HISTORICAL)

**Status: complete as of 2026-05-09 19:35 (all 6 cases done, regression confirmed but root cause not isolated).**
**Resolved 2026-05-10 by commit `0c2bdd4` — see "Completed cases" above; the word-salad reproducer no longer triggers on `scitrera/dgx-spark-sglang:0.5.11` once Bug A (`is_layer_skipped` substring) and Bug B (aggressive `sampling_overrides`) are fixed.**

Matrix: `kikube/matrixtest_matrices/sglang_nn4_tp4_ep1/qwen-3.6-35b-a3b-fp8/nv580.142_sglang-0.5.11_qwen-3.6-35b-a3b-fp8_correctness-debug_n4_ep1.yaml`

Result dir: `kikube/results/sglang_nn4_tp4_ep1/qwen-3.6-35b-a3b-fp8/0.5.11-correctness-debug/`

Image: `xomoxcc/dgx-spark-sglang:0.5.11-sm121` (sm121 patches, **without** gemma4 source patches).

### Background

While running the main 0.5.11 matrix (Tests 01–20, see `MATRIX_SUMMARY` in `0.5.11/`),
Test 01 (triton MoE + fi-attn, the prior 0.5.10 stable shape) reported severely
degraded aggregate throughput at n=4/n=8 with one `status: repetition` failure at n=8.
Inspection of the actual generated text revealed the model producing **word-salad**:
synonym-walk loops with explicit self-correction triggers, e.g.

> ...customers drive business outcomes revenue impact measurable meaningful
> contributions valued appreciated respected admired legendary status achieved
> through dedication excellence pursuit mastery relentless improvement never settle
> mediocre strive superior innovate disrupt transform industries shape future
> generations legacy endure inspire others follow lead pave way extraordinary
> achievements monumental feats history remember celebrated honored immortalized
> timeless classic masterpiece **masterpiece masterpiece masterpiece...Wait stop
> rambling. Generate structured response. Focus.**

This is **not** classic n-gram repetition (the kikube NGRAM-filter only catches
the most extreme primitive repetition cases, e.g. `retire retire retire`). Most
runs ramble all the way to the 3072-token hard limit (`finish_reason=length`),
which the matrix-test scoring counts as a *successful* run. Output quality must
be verified manually.

The bug reproduces both on `scitrera/dgx-spark-sglang:0.5.11` (vanilla) and on
`xomoxcc/dgx-spark-sglang:0.5.11-sm121` (our build). It does **not** occur on
v0.5.10 with the same configuration (compare TESTLOG `_sglang-0.5.10_*` Test 1).

`speculative_algorithm=None` in the server args of all reproducer cases — i.e.
**this is not the MTP / Spec V2 path**.

### Hypotheses (from the matrix preamble)

1. **Spec V2 Overlap-Scheduling default (PR #21062)** triggers a KV-cache update
   race even with `speculative_algorithm=None` (the overlap pipelining still runs).
2. **`mamba_scheduler_strategy=extra_buffer`** under hybrid-mamba is incompatible
   with FlashInfer 0.6.8.post1 / sgl-kernel 0.4.2.
3. Some deeper hybrid-mamba + FP8-logits issue, independent of (1) and (2).

### Configuration Matrix (6 cases)

All cases: `tp=4 ep=1 nccl=roce moe_runner=triton kv_cache_dtype=fp8_e4m3 disable_cuda_graph=false disable_piecewise_cuda_graph=true cuda_graph_max_bs=8 speculative_enabled=false`.

| #  | overlap | mamba_strategy     | attention  | Notes                                     |
|----|---------|--------------------|------------|-------------------------------------------|
| 00 | on      | extra_buffer       | fi         | 1:1 0.5.10 winner config — must reproduce |
| 01 | **off** | extra_buffer       | fi         | Hypothesis 1                              |
| 02 | on      | **""** (no_buffer) | fi         | Hypothesis 2                              |
| 03 | **off** | **""** (no_buffer) | fi         | Hypothesis 1+2 combined                   |
| 04 | on      | extra_buffer       | **triton** | Rule out FlashInfer-attn                  |
| 05 | **off** | extra_buffer       | **triton** | overlap-off + triton-attn                 |

### Results

| #  | overlap | mamba        | attn       | n=1 throughput |    n=1 finish |    n=4 agg | n=4 fails | n=8 agg | n=8 fails | Output quality                       |
|----|---------|--------------|------------|---------------:|--------------:|-----------:|----------:|--------:|----------:|--------------------------------------|
| 00 | on      | extra_buffer | fi         |          82.29 |   length=3072 |     177.99 |       0/4 |  284.20 |   **1/8** | **Word-salad** all phases            |
| 01 | **off** | extra_buffer | fi         |          67.70 | **stop=1179** |     145.38 |       0/4 |  284.90 |       0/8 | **n=1 coherent**, n=4/n=8 word-salad |
| 02 | on      | **""**       | fi         |          63.20 |   length=3072 |      93.21 |   **1/4** |  247.41 |   **1/8** | n=1 coherent, n=4/n=8 word-salad     |
| 03 | **off** | **""**       | fi         |          45.94 | **stop=1034** |     159.03 |   **1/4** |  306.66 |       0/8 | n=1 coherent, n=4/n=8 word-salad     |
| 04 | on      | extra_buffer | **triton** |          59.33 |   length=3072 | **210.70** |       0/4 |  303.46 |       0/8 | Word-salad all phases                |
| 05 | **off** | extra_buffer | **triton** |          65.07 |   length=3072 |     164.23 |       0/4 |  239.29 |       0/8 | Word-salad all phases                |

### Output samples

**Test 00 n=1 (baseline reproducer, word-salad):**

> ...completed finalized concluded terminated ended ceased stopped halted
> interrupted broken disrupted scrambled scrambled fragmented shredded torn ripped
> slashed cut chopped hacked dissected autopsied ... retired retired retired
> retired retired retire retire retire retire!!! *(Self-correction/refocus needed
> immediately!! Stop rambling mental loop start producing structured coherent
> response now!!!)*

**Test 01 n=1 (overlap-off, COHERENT):**

> 1.  **Analyze User Input:**
>    - **Role:** Science communicator with 10 years experience writing for
>      popular science magazines
>    - **Audience:** Curious 12-year-old
>    - *Quantum Entanglement:* Two particles share a single quantum state.
>      Measuring one instantly determines the state of the other, regardless of
>      distance...

**Test 01 n=4 (overlap-off, word-salad regressed):**

> ...lost missing found retrieved recovered rescued saved delivered liberated
> forever nonstop continuously perpetually incessantly unremittingly...

**Test 01 n=8 (overlap-off, word-salad worse):**

> ...awareness presence being existence life reality truth wisdom knowledge
> understanding insight intuition perception sensation feeling emotion thought
> idea concept notion theory hypothesis conjecture speculation supposition
> assumption premise axiom theorem lemma corollary proposition statement assertion
> claim argument proof demonstration evidence...

**Test 02 n=4 (mamba=no_buffer, model rebels against itself):**

> ...**[OUTPUT GENERATION PHASE ACTIVATED]** *(self-correction during ... focus
> back task at hand produce precise accurate professional response as ... OKAY
> STOP THIS RUNAWAY WORD ASSOCIATION LOOP IMMEDIATELY FOCUS RESET ...

**Test 03 n=8 (both switches off, word-salad still present):**

> ...break recess interval gap pause stop halt cease quit resign retire withdraw...
> *(Self-Correction During Drafting Phase)* That list went completely... thematic
> coherence! Let us regain focus diligently restoring precision...

> ...knowledge expanded awareness heightened perception sharpened focus clarified...

**Test 04 n=4 (baseline + triton-attn — falsifies FlashInfer-attn hypothesis):**

> ...heart center hub focus nucleus kernel crux pivot keystone linchpin axis...
> ...preserved conserved saved rescued delivered freed liberated emancipated...
> *Hmm my mind started looping randomly... Stop it! Let's get back to serious...*
> ...malicious evil wicked bad bad bad bad... STOP THIS LOOPING GENERATED TEXT
> ...self-loop detection triggered abort fantasy segment resume actual drafting...

**Test 05 n=4 (overlap-off + triton-attn):**

> ...lation_lines_phase_locked_loop_clock_generator_frequency_synthesizer_vco_var
> *(Self-Correction/Refinement during thought process)*
> ...also deriving its negation leads external observer aware correctness σ
> ...restoring wholeness completing circles closing loops breaking cycles...

### Interim conclusions

1. **The bug is a concurrency race**, not a single-flag misconfig. Both Tests 01
   and 02 fix `n=1` (single-request decode) but leave `n=4` and `n=8` broken.
   The race surfaces under multi-request decoding regardless of which one of the
   two switches we flip.
2. **Overlap-Scheduling default in 0.5.11 makes the race show up even at `n=1`**,
   because overlap pipelining gives the scheduler internal multi-batch state.
   Disabling it cleans up `n=1` but does not eliminate the underlying race.
3. **`mamba_scheduler_strategy=""` (no_buffer) is strictly worse** — it costs
   ~47% throughput at n=4 and produces a `repetition` kill at both n=4 and n=8,
   while not fixing the word-salad. Hypothesis 2 falsified.
4. **Test 03 (both switches off) does NOT fix it either** — n=1 coherent,
   n=4 with 1 repetition kill, n=8 6×length=3072 with explicit synonym-walk
   visible in the output. So neither overlap-scheduling nor mamba-strategy
   alone or in combination is the culprit at multi-request concurrency.
5. **Test 04 (baseline + triton-attn) confirms word-salad on triton-attn too.**
   Hypothesis 3 falsified — the bug is **not** FlashInfer-attn-specific. Test 04
   even has the highest n=4 throughput of the matrix (210.70, ~vs 0.5.10's 212.0)
   while still producing rambling output, so the regression is purely a
   correctness issue, not a performance one.
6. **Test 05 (overlap-off + triton-attn) likewise word-salad** at all phases.
   Even the n=1 single-request decode produced length=3072 rambling — the
   "overlap-off fixes n=1" effect from Test 01 does NOT carry over when the
   attention backend changes.
7. **Final verdict:** the regression is in the multi-request KV-cache or
   mamba-state update path on the hybrid-mamba `Qwen3_5MoeForConditionalGeneration`
   architecture, between v0.5.10.post1 and v0.5.11. **None of the three
   diagnostic switches (overlap-schedule, mamba-strategy, attention-backend)
   alone or in combination eliminates the bug at multi-request concurrency.**
   The true culprit is somewhere deeper in the 0.5.11 stack — candidates:
   - The `Qwen3_5MoeForConditionalGeneration` model code itself (small model
     touches in the 588-PR window between v0.5.10.post1 and v0.5.11)
   - The Mamba SSM kernel (mamba_backend=triton, possibly affected by sgl-kernel
     0.4.2 or torch 2.11)
   - FP8 logits post-processing or sampler-side state
   - DeepGemm JIT (we have `disable_deep_gemm=true`, so this is *probably* not it)

### Action items

- **File upstream issue** with reproducer: model + prompt + server-args + concrete
  word-salad output sample. Reference PRs that touched
  `Qwen3_5MoeForConditionalGeneration` between v0.5.10.post1 and v0.5.11 (e.g.
  #19767 MTP/radix-cache compat) and the hybrid-mamba scheduler/Mamba-cache code.
- **Pin Qwen3.6-35B-A3B-FP8 to scitrera/dgx-spark-sglang:0.5.10** in production
  via per-profile `sglang_image` override until upstream fix lands. Other models
  (Gemma-4, GLM, Qwen3.5-397B-NVFP4) can move to 0.5.11 individually.
- **Do not run the main 20-case 0.5.11 matrix to completion** for this model —
  every aggregate-throughput number is meaningless if 50–80% of the output
  tokens are rambling. The matrix is fine to run on **other models** (different
  arch families) where word-salad has not been observed.
- **Bisect candidates** for upstream issue (priority order):
  1. Hybrid-mamba scheduler / Mamba-state-cache code under multi-request
     concurrency. The race surfaces at n=4 even with all known mitigations.
  2. Sampler-side state. With `enable_custom_logit_processor=true` (default on
     0.5.11) and `min_tokens=4 + frequency_penalty=0.5 + presence_penalty=1.5`
     in the profile, a per-batch state-mixing bug in penalty application could
     produce exactly this synonym-walk pattern.
  3. sgl-kernel 0.4.2 mamba kernels.
- **Extend kikube NGRAM-filter** to catch synonym-walk patterns. Current filter
  caught only 4 of ~30 word-salad rambles across the 6 cases. Suggestions:
  Type-Token-Ratio threshold, WordNet-synset density, or LLM-judge on output.
- **Verify Qwen3.6-27B-FP8** (sibling hybrid-mamba arch) for the same bug —
  if also affected, this is an arch-family regression, not a model-specific one.
- **Verify Gemma-4 / GLM / non-mamba models** are unaffected on 0.5.11 — those
  matrices can proceed normally.

