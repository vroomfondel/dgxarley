# FlashInfer Upstream Bug: head_dim=512 not supported (Gemma-4 global attention)

## Status (re-verified 2026-05-12 — bug RE-OPENED on a different code path)

**PR #2959 is necessary but NOT sufficient. The `head_dim=512` dispatch gap
persists in FlashInfer 0.6.11 for a specific MMA-tile configuration that
Gemma-4's global-attention layers actually use on SM121.** The bug was
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
instantiated for `head_dim=512`**. The combination Gemma-4's global-attention
layers trigger at decode time on SM121 is:

```
NUM_MMA_D_QK=32     (= head_dim / 16 = 512 / 16)
NUM_MMA_D_VO=32
NUM_MMA_Q=1         (single-query decode batch)
NUM_MMA_KV=1        (single KV tile per step)
NUM_WARPS_Q=1
NUM_WARPS_KV=4      (4 warps over KV dimension)
```

The dispatch macro in `prefill.cuh:2978` enumerates the compiled template
instantiations and rejects this tuple as "Invalid configuration", asking
the user to file an upstream issue. The error is deterministic, fires on
the first global-attention layer of every forward, and is identical across
all four TP ranks.

Note that this is a **separate** dispatch line from the original symptom in
the 2026-04 era of this doc (`prefill.cuh:2615`) — that one was the truly
unsupported `head_dim=512` itself. PR #2959 fixed the 2615 line; the 2978
line is a sister gap in the same dispatch macro for the small-batch decode
parameter combination.

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

Crash during CUDA graph capture (or first forward in eager mode) at the first
global-attention layer's decode path. Two slightly different line numbers in
the same dispatch macro, depending on FlashInfer version:

```
RuntimeError: Error in function 'BatchPrefillWithPagedKVCacheDispatched'
  at flashinfer/data/include/flashinfer/attention/prefill.cuh:2978:
  FlashInfer Internal Error: Invalid configuration :
    NUM_MMA_Q=1 NUM_MMA_D_QK=32 NUM_MMA_D_VO=32 NUM_MMA_KV=1
    NUM_WARPS_Q=1 NUM_WARPS_KV=4
  please create an issue and report the issue to the developers.
```

(`prefill.cuh:2615` in flashinfer ≤ 0.6.9; `prefill.cuh:2978` in 0.6.11 after
PR #2959 expanded the dispatch table — the dispatch logic was shifted lower
in the file, the unsupported tuple remained unsupported.)

Stack trace (captured 2026-05-11 from a flashinfer-0.6.11 build):
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

Wrapped in:
```
Exception: Capture cuda graph failed: ... Invalid configuration ...
  (Possible solutions hint: lower --mem-fraction-static / --cuda-graph-max-bs,
   disable --enable-torch-compile, or use --disable-cuda-graph)
```

The hint is misleading — `disable_cuda_graph: true` does NOT fix this. Test 02
(`02_triton-moe_fi-attn_no-cuda-graph`) crashed identically. The dispatch gap
is hit during normal prefill setup too, not just CUDA-graph capture.

The `NUM_MMA_D_QK=32` comes from `head_dim/16 = 512/16 = 32`. The dispatch
gap for the `(NUM_MMA_Q=1, NUM_MMA_KV=1, NUM_WARPS_Q=1, NUM_WARPS_KV=4)` tuple
in combination with `head_dim=512` is a compile-time dispatch check, not a
runtime kernel failure — deterministic, affects all TP ranks simultaneously,
fires before any benchmark workload runs.

## Root cause

FlashInfer's FA2/FA3 attention kernel templates are compiled with fixed dispatch
tables that cover `head_dim` values up to 256 (which covers most models: 64,
80, 96, 128, 256). The dispatch macro in `prefill.cuh` enumerates valid
`(NUM_MMA_D_QK, NUM_MMA_D_VO, NUM_MMA_Q, NUM_MMA_KV, NUM_WARPS_Q, NUM_WARPS_KV)`
tuples and rejects any that don't match.

PR #2959 added `head_dim=512` instantiations for *some* of those tuples — enough
for the existing FlashInfer test cases — but did NOT cover the small-batch
single-warp-Q decode tuple that Gemma-4 actually hits on SM121:

```
NUM_MMA_D_QK = 32   ← head_dim 512 / 16   (added by PR #2959)
NUM_MMA_D_VO = 32                          (added by PR #2959)
NUM_MMA_Q    = 1    ← small query batch
NUM_MMA_KV   = 1    ← single KV tile per step
NUM_WARPS_Q  = 1    ← single Q warp
NUM_WARPS_KV = 4    ← 4 KV warps
```

This tuple is what `BatchPrefillWithPagedKVCacheDispatched` produces for the
decode path of Gemma-4's global-attention layers when running TP=4 on
4× DGX Spark. Other models with `head_dim=512` might land on a different
parameter tuple that *was* covered by PR #2959 — but this one isn't.

This is documented in FlashInfer PR #2959's description:
> "FlashInfer FA2/FA3 kernels don't support head_dim > 256"

— but the description should have said "*not all combinations of head_dim=512*"
rather than implying full head_dim=512 coverage.

## Workaround

**`attention_backend=triton`** — SGLang's Triton attention backend handles
`head_dim=512` correctly. PR #22079 in sgl-project/sglang added SM120/121-specific
block sizes for Triton attention that prevent the PTX register exhaustion that
originally affected Gemma-4 on GB200/sm100a. On our SM121/GB10 cluster, Triton
attention with `head_dim=512` works for both CG-on and eager modes.

This is the recommended workaround for all Gemma-4 model profiles until
FlashInfer gains `head_dim=512` support.

## Upstream PRs

| Repo | PR | Title | Status |
|------|-----|-------|--------|
| flashinfer-ai/flashinfer | [#2959](https://github.com/flashinfer-ai/flashinfer/pull/2959) | [Fmha] Add head_dim=512 support for trtllm attention kernels | **merged 2026-04-22** (in v0.6.10rc1 / v0.6.10 / v0.6.10.post1 / v0.6.11), but **incomplete** — does not cover the `(NUM_MMA_Q=1, NUM_MMA_KV=1, NUM_WARPS_Q=1, NUM_WARPS_KV=4)` decode tuple |
| sgl-project/sglang | [#22079](https://github.com/sgl-project/sglang/pull/22079) | [nvidia] Gemma4 nvfp4 fix | **merged** (2026-04-10) |
| flashinfer-ai/flashinfer | [#3297](https://github.com/flashinfer-ai/flashinfer/issues/3297) | [Bug] head_dim=512 dispatch gap on SM121 (Gemma-4 global attention) — NUM_MMA_Q=1 NUM_MMA_KV=1 NUM_WARPS_Q=1 NUM_WARPS_KV=4 not instantiated after PR #2959 | **OPEN** (filed 2026-05-12 with full trace + repro + env table) |

PR #22079 in SGLang fixed the **Triton attention** side of the `head_dim=512`
problem (PTX register exhaustion on SM100a/GB200). The companion FlashInfer
attention fix (PR #2959) merged on 2026-04-22 and is in v0.6.10rc1+ (stable
release v0.6.10 was tagged 2026-05-04) — but, as the 2026-05-11 sweep proves,
**does not cover the parameter tuple Gemma-4 actually hits on SM121**.

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
ab `prefill.cuh:2978`, fehlendem Dispatch-Tuple, betroffenen Modellen und
Cross-Links auf verwandte Issues/PRs (#2959, #3016, #2555, #3170, vllm#40677,
Dao-AILab/flash-attention#2427).

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
