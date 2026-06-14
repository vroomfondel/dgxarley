# FlashInfer Upstream Bug: head_dim=512 not supported (Gemma-4 global attention)

## Status 2026-06-14

PR #3576 remains **OPEN and unmerged**. SGLang v0.5.13 (released 2026-06-13) ships flashinfer 0.6.12 (PR #26854) but does NOT contain the head_dim=512 fix — PR #3576 is not in v0.6.12 or v0.6.13rc1. `attention_backend: triton` workaround remains required on all Gemma-4 profiles. `scitrera/dgx-spark-sglang:0.5.13` not yet on DockerHub; cluster default stays on 0.5.12 images regardless.

## Status (re-verified 2026-06-11 — Fix-PR #3576 opened by maintainer)

**PR #2959 is necessary but NOT sufficient. The `head_dim=512` dispatch gap
persists in FlashInfer 0.6.11 / 0.6.11.post1 / 0.6.11.post2 / 0.6.11.post3
for a specific MMA-tile configuration that Gemma-4's global-attention
layers actually use on SM121.** FlashInfer shipped three post-releases
since the original discovery — v0.6.11.post1 (2026-05-13),
v0.6.11.post2 (2026-05-14), and v0.6.11.post3 (2026-05-15) — none
contains a fix for the missing dispatch tuple; their release notes
mention no `head_dim`, `prefill.cuh`, or Gemma-4 changes. As of
2026-05-31, **v0.6.12 stable shipped 2026-05-29** and still does **not**
contain the **two** missing `head_dim=512` dispatch instantiations — decode
`(NUM_MMA_Q=1, NUM_MMA_KV=1, NUM_WARPS_Q=1, NUM_WARPS_KV=4)` and prefill/extend
`(NUM_MMA_Q=1, NUM_MMA_KV=2, NUM_WARPS_Q=4, NUM_WARPS_KV=1)` (no `head_dim` /
`prefill.cuh` entry in the v0.6.12 changelog; see "What PR #2959 fixed" below
for both tuples and the CG-on-vs-eager reason only one shows per run).
**v0.6.13rc1 (2026-06-10) likewise contains NO head_dim=512 fix.**
Note also that SGLang 0.5.12 /
0.5.12.post1 still pin flashinfer at **0.6.11.post1**, so even moving to
the current default image `xomoxcc/dgx-spark-sglang:0.5.12.post1-sm121`
does not bring v0.6.12 — and would not fix this even if it did. Upstream
issue [#3297](https://github.com/flashinfer-ai/flashinfer/issues/3297)
had zero comments as of 2026-05-31; as of 2026-06-11 it has **4 comments**:
third-party repro (2026-06-03), our own root-cause analysis (2026-06-04),
maintainer @bkryu confirmation (2026-06-09) noting that PR #2959 covers only
trtllm kernels (not an option for SM120/121), and maintainer @bkryu
announcing PR #3576 (2026-06-11T05:37Z). Fix-PR #3576 is **open and unreviewed**
— not in v0.6.12 nor v0.6.13rc1; see "Upstream PRs" table below.
**The workaround (`attention_backend=triton`) remains required until #3576
merges, ships in a flashinfer release, and our image's flashinfer pin is
bumped to that release.** The bug was
prematurely marked "fixed" in the 2026-05-10 status; a fresh
`gemma-4-26b-a4b-it` BF16 matrix sweep on 2026-05-11 (image
`xomoxcc/dgx-spark-sglang:0.5.11-sm121`, flashinfer 0.6.11) shows that
`attention_backend: flashinfer` still crashes deterministically on every
Gemma-4 forward — exact same `Invalid configuration` error message, just at
a different `prefill.cuh` line (2978 instead of 2615).

Concrete results from the 2026-05-11 sweep (18 cases,
`results/sglang_nn4_tp4_ep1/gemma-4-26b-a4b-it/0.5.11/`):

| Backend combo | Result |
|---|---:|
| `triton`-MoE + `flashinfer`-attn (Tests 01–03, all 3 CG variants) | 3× `startup_crash` |
| `triton`-MoE + `triton`-attn (Tests 04–06) | 3× outcome=13 ✓ |
| `fi_cutlass`-MoE + `flashinfer`-attn (Tests 07–09) | 2× `startup_crash`, 1× `bench_crash` |
| `fi_cutlass`-MoE + `triton`-attn (Tests 10–12) | 3× outcome=13 ✓ |
| `fi_cutedsl`-MoE + `flashinfer`-attn (Tests 13–15) | 3× `startup_crash` |
| `fi_cutedsl`-MoE + `triton`-attn (Tests 16–18) | 3× `startup_crash` |

**Summary**: every `attention_backend: flashinfer` case fails. Every
`attention_backend: triton` case (with `triton`-MoE or `fi_cutlass`-MoE)
succeeds. The `flashinfer_cutedsl` MoE backend is independently broken on
Gemma-4 — separate failure mode, not covered by this doc.

The Triton-attention path remains the only working option for Gemma-4
under SGLang 0.5.11 on SM121 — same workaround as before, but now confirmed
to be **still required** on the current image generation, not just
"intentional pending A/B benchmark".

**Image situation today:**
- `xomoxcc/dgx-spark-sglang:0.5.11-sm121` and
  `xomoxcc/dgx-spark-sglang:0.5.11-gemma4-sm121` — both pin
  **`FLASHINFER_VERSION=0.6.11`** in their recipes. Despite the recipe
  comment that this would unblock fi-attn for Gemma-4 global layers,
  it does NOT — see the new dispatch-gap detail below.
- All four Gemma-4 model profiles
  (`google-gemma-4-26b-a4b-it.yml`, `google-gemma-4-31b-it.yml`,
  `bg-digitalservices-gemma-4-26b-a4b-it-nvfp4.yml`,
  `nvidia-gemma-4-31b-it-nvfp4.yml`) keep `attention_backend: triton` —
  this is the right setting, do not change.
- Stock SGLang **v0.5.11** (`scitrera/dgx-spark-sglang:0.5.11`) bumps
  flashinfer to 0.6.8.post1 — also still affected (one version below the
  0.6.11 we tested, but even 0.6.11 doesn't fix the dispatch gap).
- Legacy dev1 recipes (`sglang-sm121-dev1.recipe`,
  `sglang-gemma4-sm121-dev1.recipe`) still pin 0.6.8.post1 — still affected,
  kept for rollback only.

## What PR #2959 actually fixed vs. what's still missing

PR #2959 added `head_dim=512` support to a subset of the FlashInfer
`BatchPrefillWithPagedKVCacheDispatched` template parameter space — enough
to make `head_dim=512` syntactically valid in the dispatch table. But the
template is also parameterized by `NUM_MMA_Q × NUM_MMA_KV × NUM_WARPS_Q ×
NUM_WARPS_KV`, and **not all combinations of those parameters were
instantiated for `head_dim=512`**.

**TWO distinct tuples are missing**, not one. Both Gemma-4 models (26B-A4B
MoE *and* 31B dense) hit exactly these two at `head_dim=512`
(`NUM_MMA_D_QK = NUM_MMA_D_VO = 32`), both rejected at `prefill.cuh:2978`.
Re-verified 2026-06-04 from the raw head-pod logs of the 2026-05-11/15
0.5.11 sweep (`kikube/results/.../gemma-4-26b-a4b-it/0.5.11/` and
`.../gemma-4-31b-it/`):

| Path | Missing tuple | Fires when |
|---|---|---|
| **`forward_decode`** | `NUM_MMA_Q=1 NUM_MMA_KV=1 NUM_WARPS_Q=1 NUM_WARPS_KV=4` | CUDA-graph **capture** (CG-on) → `startup_crash` |
| **`forward_extend`** (prefill) | `NUM_MMA_Q=1 NUM_MMA_KV=2 NUM_WARPS_Q=4 NUM_WARPS_KV=1` | **first forward** in eager / `--disable-cuda-graph` → `bench_crash` |

```
NUM_MMA_D_QK=32     (= head_dim / 16 = 512 / 16)   ← head_dim=512 on BOTH QK and VO
NUM_MMA_D_VO=32     (= 512 / 16)
NUM_MMA_Q=1         ← ALWAYS 1, in both tuples and across every crash (incl. dense 31B
                      with long prompts). Q is split across NUM_WARPS_Q, not MMA-Q.
```

So it is **not** purely the decode tile — the prefill/extend path does **not**
resolve either. You only ever observe one tuple per run: with CUDA graphs on,
SGLang captures the *decode* graph at startup, so the decode tuple aborts
capture before any request runs; in eager mode there's no capture, so the
very first forward is a *prefill* and the extend tuple aborts before decode is
ever reached. Neither run survives to touch the other path. The distinguishing
axes between the two are `NUM_MMA_KV` (1 vs 2) and the warp split
(`WARPS_Q × WARPS_KV` = 1×4 vs 4×1) — **not** `NUM_MMA_Q`.

The dispatch macro in `prefill.cuh:2978` enumerates the compiled template
instantiations and rejects these tuples as "Invalid configuration", asking
the user to file an upstream issue. The error is deterministic, fires on
the first global-attention layer, and is identical across all four TP ranks.

Note that this is a **separate** dispatch line from the original symptom in
the 2026-04 era of this doc (`prefill.cuh:2615`) — that one was the truly
unsupported `head_dim=512` itself. PR #2959 fixed the 2615 line; the 2978
line is a sister gap in the same dispatch macro. (Older flashinfer 0.6.8.post1
runs print the same two tuples at line 2615; #2959 only moved the macro to
2978 and left both tuples uninstantiated.)

## Affected models

Any model with `head_dim > 256` using `attention_backend=flashinfer`:

- `google/gemma-4-26B-A4B-it` (MoE, `global_head_dim=512` on full-attention layers)
- `google/gemma-4-31B-it` (dense, same `global_head_dim=512`)
- `bg-digitalservices/Gemma-4-26B-A4B-it-NVFP4` (NVFP4 MoE, same architecture)
- `nvidia/Gemma-4-31B-IT-NVFP4` (NVFP4 dense, same architecture)

Gemma-4 uses hybrid attention: sliding-window layers have `head_dim=256` (works
fine), but every 6th layer is a full-attention layer with `global_head_dim=512`
(crashes). The crash triggers on the first global-attention layer during forward.

Other models with standard `head_dim` (≤256) are unaffected.

## Symptom

Crash at the first global-attention layer, either during CUDA-graph capture
(CG-on → decode tuple) or on the first forward in eager mode (→ extend/prefill
tuple). Same dispatch macro, two slightly different line numbers depending on
FlashInfer version.

**Decode tuple** (CG-on, capture aborts):
```
RuntimeError: Error in function 'BatchPrefillWithPagedKVCacheDispatched'
  at flashinfer/data/include/flashinfer/attention/prefill.cuh:2978:
  FlashInfer Internal Error: Invalid configuration :
    NUM_MMA_Q=1 NUM_MMA_D_QK=32 NUM_MMA_D_VO=32 NUM_MMA_KV=1
    NUM_WARPS_Q=1 NUM_WARPS_KV=4
  please create an issue and report the issue to the developers.
```

**Extend/prefill tuple** (eager / `--disable-cuda-graph`, first forward aborts):
```
RuntimeError: Error in function 'BatchPrefillWithPagedKVCacheDispatched'
  at flashinfer/data/include/flashinfer/attention/prefill.cuh:2978:
  FlashInfer Internal Error: Invalid configuration :
    NUM_MMA_Q=1 NUM_MMA_D_QK=32 NUM_MMA_D_VO=32 NUM_MMA_KV=2
    NUM_WARPS_Q=4 NUM_WARPS_KV=1
  please create an issue and report the issue to the developers.
```

(`prefill.cuh:2615` in flashinfer ≤ 0.6.9; `prefill.cuh:2978` in 0.6.11 after
PR #2959 expanded the dispatch table — the dispatch logic was shifted lower
in the file, both tuples remained unsupported.)

Decode stack trace (CG-on, captured 2026-05-11 from a flashinfer-0.6.11 build):
```
gemma4_causal.py:529  self.self_attn(...)
  → gemma4_causal.py:367  self.attn(...)
    → radix_attention.py:138  forward_batch.attn_backend.forward(...)
      → base_attn_backend.py:95  self.forward_decode(...)
        → flashinfer_backend.py:920  decode_wrapper.forward(...)
          → flashinfer/decode.py:1509  self._cached_module.paged_run(...)
            → flashinfer/prefill.py:707  paged_run_func(...)
              → flashinfer/attention/prefill.cuh:2978  DISPATCH FAILS
```

Extend stack trace (eager, Test 02 / Test 08): the crash is in `forward_extend`
(`model_runner.py:3063` → `flashinfer_backend.py:813 forward_extend`), the
prefill counterpart of the decode trace above — same `prefill.cuh:2978`.

Wrapped in:
```
Exception: Capture cuda graph failed: ... Invalid configuration ...
  (Possible solutions hint: lower --mem-fraction-static / --cuda-graph-max-bs,
   disable --enable-torch-compile, or use --disable-cuda-graph)
```

The hint is misleading — `disable_cuda_graph: true` does NOT fix this; it just
trades the decode tuple for the extend tuple. Test 02
(`02_triton-moe_fi-attn_no-cuda-graph`) and Test 08 crash on `forward_extend`
with the `NUM_MMA_KV=2 NUM_WARPS_Q=4 NUM_WARPS_KV=1` tuple, i.e. the prefill
path is independently broken, not just CUDA-graph capture.

The `NUM_MMA_D_QK=NUM_MMA_D_VO=32` comes from `head_dim/16 = 512/16 = 32` (on
both QK and VO → genuine `head_dim=512`). Both dispatch gaps are compile-time
checks, not runtime kernel failures — deterministic, affect all TP ranks
simultaneously, fire before any benchmark workload runs.

## Root cause (verified from source 2026-06-04)

**This is NOT a missing dispatch-table entry — it is a register-budget
rejection that head_dim=512 categorically cannot pass in the FA2/FA3 prefill
kernel.** Earlier revisions of this doc called it a "missing instantiation PR
#2959 forgot to add"; reading the shipped `prefill.cuh` (flashinfer 0.6.11, in
`xomoxcc/dgx-spark-sglang:0.5.11-sm121`) disproves that.

### Why we always see exactly two tuples, both with `NUM_MMA_Q=1`

In `BatchPrefillWithPagedKVCacheDispatched`, `NUM_MMA_Q`, `NUM_WARPS_Q` and
`NUM_WARPS_KV` are **pure functions of `CTA_TILE_Q`** (`prefill.cuh:2923–2925`
calling `get_num_mma_q`/`get_num_warps_q`/`get_num_warps_kv`, lines 72–90).
`CTA_TILE_Q` is dispatched at runtime to one of `{16, 64, 128}`
(`DISPATCH_CTA_TILE_Q`, `utils.cuh:113`):

| `CTA_TILE_Q` | `NUM_MMA_Q` (>64→2) | `NUM_WARPS_Q` (>16→4) | `NUM_WARPS_KV` (=4/wq) |
|---|---|---|---|
| 16  | 1 | 1 | 4 |
| 64  | 1 | 4 | 1 |
| 128 | 2 | 4 | 1 |

Our two crash tuples map **exactly**:
- **decode** (query-len 1 → `CTA_TILE_Q=16`): `NUM_MMA_Q=1 NUM_WARPS_Q=1 NUM_WARPS_KV=4`
- **prefill/extend** (`CTA_TILE_Q=64`): `NUM_MMA_Q=1 NUM_WARPS_Q=4 NUM_WARPS_KV=1`

`NUM_MMA_KV` is then derived at runtime from the smem/register budget
(`DISPATCH_NUM_MMA_KV(min(max_num_mma_kv_smem, max_num_mma_kv_reg), …)`,
line 2964) → 1 for decode, 2 for prefill in our logs. The prefill tuple is
**not** a "multi-MMA-Q" shape — its 64 query rows come from `NUM_WARPS_Q=4`
(4×16), with `NUM_MMA_Q=1`. `NUM_MMA_Q` only reaches 2 at `CTA_TILE_Q=128`,
which is never selected here: the head_dim=512 Q-smem tile alone
(`128 × 512 × 2 B = 128 KB`) won't fit. That's why `NUM_MMA_Q=1` in 100% of
crashes (228/228, both models incl. the dense 31B with long prompts).

### What actually rejects the config: `KTraits::IsInvalid()`

The error at `prefill.cuh:2978` fires because `KTraits::IsInvalid()`
(lines 158–165) returns `true`. The decisive clause (line 162):

```cpp
NUM_MMA_Q * (8 * NUM_MMA_D_VO + 2 * sizeof(DTypeQKAccum) * NUM_MMA_KV) >= 256
```

For head_dim=512, `NUM_MMA_D_VO = 512/16 = 32`, so `8 * NUM_MMA_D_VO = 256`
**hits the threshold on its own** — before any `NUM_MMA_Q` / `NUM_MMA_KV`
contribution. Evaluated for our tuples (BF16 → `DTypeQKAccum=float`, 4 B):
- decode  `1·(8·32 + 2·4·1) = 264 ≥ 256` → invalid
- prefill `1·(8·32 + 2·4·2) = 272 ≥ 256` → invalid

This is a **register-budget guard**: a head_dim=512 output tile needs ~256
accumulator registers/thread (8 regs × 32 VO-MMA tiles), at/over the
~255-reg/thread hardware limit. So **every** head_dim=512 configuration is
rejected by this kernel — any `NUM_MMA_Q≥1`, any `NUM_MMA_KV`, BF16 or
fp16-accum alike (the `8·32` term dominates regardless). The FA2/FA3 prefill
kernel categorically tops out near head_dim=256, exactly as PR #2959's own
description says:

> "FlashInfer FA2/FA3 kernels don't support head_dim > 256"

### Consequence for the fix

"Adding the missing tuple to the dispatch table" (the natural first instinct)
**will not work** for `BatchPrefillWithPagedKVCacheDispatched` — there is no
table to extend; `IsInvalid()` rejects head_dim=512 on register grounds.
PR #2959 added head_dim=512 to the **trtllm** kernels (per its title:
*"Add head_dim=512 support for trtllm attention kernels"*), a different code
path. The real bug is that SGLang's flashinfer decode/prefill wrapper on SM121
routes Gemma-4's head_dim=512 attention into this FA2 path
(`decode.py:1509 paged_run → prefill.py:707 → prefill.cuh`) instead of the
trtllm kernel. The fix lives at that dispatch decision, not in the FA2 tuple
table.

> **Confidence:** the `IsInvalid()` arithmetic and the `CTA_TILE_Q` mapping
> above are verified from the shipped 0.6.11 source. The "route to trtllm
> instead" conclusion is a strong inference from PR #2959's title — not yet
> traced end-to-end through the FlashInfer/SGLang wrapper dispatch logic.

## Workaround

**`attention_backend=triton`** — SGLang's Triton attention backend handles
`head_dim=512` correctly. PR #22079 in sgl-project/sglang added SM120/121-specific
block sizes for Triton attention that prevent the PTX register exhaustion that
originally affected Gemma-4 on GB200/sm100a. On our SM121/GB10 cluster, Triton
attention with `head_dim=512` works for both CG-on and eager modes.

This is the recommended workaround for all Gemma-4 model profiles until
FlashInfer gains `head_dim=512` support.

## Upstream PRs

| Repo                     | PR                                                               | Title                                                                                                                                                       | Status                                                                                                                                                                                       |
|--------------------------|------------------------------------------------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| flashinfer-ai/flashinfer | [#2959](https://github.com/flashinfer-ai/flashinfer/pull/2959)   | [Fmha] Add head_dim=512 support for trtllm attention kernels                                                                                                | **merged 2026-04-22** (in v0.6.10rc1 / v0.6.10 / v0.6.10.post1 / v0.6.11), but **incomplete** — covers neither the decode tuple `(NUM_MMA_KV=1, NUM_WARPS_Q=1, NUM_WARPS_KV=4)` nor the prefill/extend tuple `(NUM_MMA_KV=2, NUM_WARPS_Q=4, NUM_WARPS_KV=1)` (both `NUM_MMA_Q=1, head_dim=512`) |
| sgl-project/sglang       | [#22079](https://github.com/sgl-project/sglang/pull/22079)       | [nvidia] Gemma4 nvfp4 fix                                                                                                                                   | **merged** (2026-04-10)                                                                                                                                                                      |
| flashinfer-ai/flashinfer | [#3297](https://github.com/flashinfer-ai/flashinfer/issues/3297) | [Bug] head_dim=512 dispatch gap on SM121 (Gemma-4 global attention) — NUM_MMA_Q=1 NUM_MMA_KV=1 NUM_WARPS_Q=1 NUM_WARPS_KV=4 not instantiated after PR #2959 | **OPEN** (filed 2026-05-12; 4 comments as of 2026-06-11: third-party repro 2026-06-03, our root-cause analysis 2026-06-04, maintainer @bkryu confirmation 2026-06-09 noting PR #2959 covers only trtllm kernels, @bkryu announcing PR #3576 2026-06-11T05:37Z; not addressed in v0.6.12 or v0.6.13rc1) |
| flashinfer-ai/flashinfer | [#3576](https://github.com/flashinfer-ai/flashinfer/pull/3576)   | feat(attention): head_dim=512 support for attention prefill & decode for Gemma 4 on SM120/121 | **OPEN** (filed 2026-06-11T05:32Z by maintainer @bkryu; internal CI kicked off 05:45 UTC; no reviews yet; NOT in v0.6.12 or v0.6.13rc1). Scope: adds head_dim=512 to the `backend='fa2'` path for `BatchDecodeWithPagedKVCacheWrapper`, `BatchPrefillWithPagedKVCacheWrapper`, `BatchPrefillWithRaggedKVCacheWrapper` — prefill + decode; Q=bfloat16, KV=bfloat16 or fp8_e4m3; explicitly targets SM12x (RTX PRO 6000, DGX Spark / GB10). Benchmarks on DGX Spark/GB10 (SM121): prefill hd512 34.6 TFLOP/s @ s_qo=8192; decode KV-bandwidth-bound; author notes prefill likely has more headroom. Files: `include/flashinfer/attention/prefill.cuh`, `persistent.cuh`, `utils.cuh`, jinja kernel instantiations, `flashinfer/aot.py`, `test_fp8_prefill.py`, `test_batch_decode_kernels.py`. **Workaround remains required until this PR merges, ships in a release, and our image's flashinfer pin is bumped.** |

PR #22079 in SGLang fixed the **Triton attention** side of the `head_dim=512`
problem (PTX register exhaustion on SM100a/GB200). The companion FlashInfer
attention fix (PR #2959) merged on 2026-04-22 and is in v0.6.10rc1+ (stable
release v0.6.10 was tagged 2026-05-04) — but, as the 2026-05-11 sweep proves,
**does not cover the two parameter tuples Gemma-4 actually hits on SM121**
(one on the decode path, one on prefill/extend — see "What PR #2959 fixed").

Our **`xomoxcc/dgx-spark-sglang:0.5.11-(gemma4-)sm121`** image (current
production for Gemma-4 profiles) pins **flashinfer 0.6.11**, which contains
PR #2959 but does not contain the still-missing instantiations. Therefore on
the currently deployed Gemma-4 image:
- `attention_backend=triton` works (PR #22079 fix active) — this is what
  the profiles use today, and it remains the only working option.
- `attention_backend=flashinfer` is **still broken** for Gemma-4 on SM121
  pending a follow-up FlashInfer release with the missing template
  instantiations.

**Upstream-Issue gefiled 2026-05-12**: [flashinfer-ai/flashinfer#3297](https://github.com/flashinfer-ai/flashinfer/issues/3297)
mit Repro-Konfig (`Gemma-4 26B-A4B-it`, TP=4, SM121, flashinfer 0.6.11,
`attention_backend=flashinfer`, `cuda_graph_max_bs=8`, BF16), vollem Stack-Trace
ab `prefill.cuh:2978`, den zwei fehlenden Dispatch-Tuples (decode + prefill),
betroffenen Modellen und Cross-Links auf verwandte Issues/PRs (#2959, #3016,
#2555, #3170, vllm#40677, Dao-AILab/flash-attention#2427).

**Issue-Verlauf (verifiziert 2026-06-11):** 4 Kommentare insgesamt — Third-Party-Repro
2026-06-03; unsere Root-Cause-Analyse 2026-06-04; Maintainer @bkryu
2026-06-09 bestätigt den Bug und weist darauf hin, dass PR #2959 nur
trtllm-Kernels abdeckt (kein Pfad für SM120/121); @bkryu kündigt 2026-06-11T05:37Z
Fix-PR [#3576](https://github.com/flashinfer-ai/flashinfer/pull/3576) an
("feat(attention): head_dim=512 support for attention prefill & decode for
Gemma 4 on SM120/121"). PR #3576 ist noch nicht in einem Release enthalten
(weder v0.6.12 noch v0.6.13rc1).

## Relationship to other bugs

- **Independent of** the Gemma-4 v0.5.10 Transformers fallback bugs
  (`SGLANG_GEMMA4_UPSTREAM_BUG.md`) — those are about missing native model
  implementation. This bug exists even WITH the native implementation.
- **Independent of** the FlashInfer FP4 dynamo tracing bug
  (`FLASHINFER_CUDA_VERSION_SUBPROCESS_UPSTREAM_BUG.md`) — different FlashInfer
  component (attention vs quantization).
- **Related to** SGLang issue #22277 (Gemma4 E4B fp8 KV cache crash with
  `num_kv_shared_layers > 0`) — same model family, different failure mode.

## Files

- `roles/k8s_dgx/model_profiles/google-gemma-4-26b-a4b-it.yml` — must use
  `attention_backend: triton` (workaround, still required on 0.5.11+0.6.11).
- `roles/k8s_dgx/model_profiles/google-gemma-4-31b-it.yml` — same.
- NVFP4 profiles are blocked by other bugs before reaching this one, but
  would also need `attention_backend: triton` once unblocked.
- `matrixtest_matrices/sglang_nn4_tp4_ep1/gemma-4-26b-a4b-it/...0.5.11_..._n4_ep1.yaml` —
  the 2026-05-11 sweep that re-discovered this dispatch gap. All 9
  `attention_backend: flashinfer` test cases (Tests 01–03, 07–09, 13–15)
  crashed with the `Invalid configuration` error documented above. Reference
  data for the upstream issue.
- `results/sglang_nn4_tp4_ep1/gemma-4-26b-a4b-it/0.5.11/nv580.142_sglang-0.5.11_gemma-4-26b-a4b-it_4n_1pp_4tp_ep1_01_triton-moe_fi-attn/...head_*.log` —
  full head-pod log with the traceback and `prefill.cuh:2978` reference.
