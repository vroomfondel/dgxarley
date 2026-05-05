# FlashInfer Upstream Bug: head_dim=512 not supported (Gemma-4 global attention)

## Status (re-verified 2026-05-05)

**Upstream fix merged and shipped in stable, image rebuild pending.** FlashInfer
0.6.7.post3's FA2/FA3 attention kernels do not support `head_dim > 256`.
Upstream fix
[flashinfer PR #2959](https://github.com/flashinfer-ai/flashinfer/pull/2959)
**merged 2026-04-22**, shipped first in **v0.6.10rc1** (2026-04-30) and then
in the **v0.6.10 stable release** (2026-05-04). Not runtime-patchable — the
kernel dispatch table is compiled into the FlashInfer binary, so the cluster
only benefits once the SGLang image is rebuilt against flashinfer ≥ 0.6.10.
Current production image still pins 0.6.7.post3, so the
`attention_backend=triton` workaround remains in effect for now.

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
global-attention layer's decode path:

```
RuntimeError: Error in function 'BatchPrefillWithPagedKVCacheDispatched'
  at flashinfer/data/include/flashinfer/attention/prefill.cuh:2615:
  FlashInfer Internal Error: Invalid configuration :
    NUM_MMA_Q=1 NUM_MMA_D_QK=32 NUM_MMA_D_VO=32 NUM_MMA_KV=1
    NUM_WARPS_Q=1 NUM_WARPS_KV=4
  please create an issue and report the issue to the developers.
```

Stack trace:
```
gemma4_causal.py:367  self.attn(...)
  → radix_attention.py:127  forward_batch.attn_backend.forward(...)
    → flashinfer_backend.py:912  decode_wrapper.forward(...)
      → flashinfer/decode.py:1444  self._cached_module.paged_run(...)
        → flashinfer/prefill.py:717  paged_run_func(...)
          → flashinfer/attention/prefill.cuh:2615  DISPATCH FAILS
```

The `NUM_MMA_D_QK=32` comes from `head_dim/16 = 512/16 = 32`. FlashInfer's
dispatch macro does not have a compiled kernel for this MMA tile configuration.
The error is a compile-time dispatch check, not a runtime kernel failure — it's
deterministic and affects all TP ranks simultaneously.

## Root cause

FlashInfer's FA2/FA3 attention kernel templates are compiled with fixed dispatch
tables that cover `head_dim` values up to 256 (which covers most models: 64,
80, 96, 128, 256). The dispatch macro in `prefill.cuh` enumerates valid
`(NUM_MMA_D_QK, NUM_MMA_D_VO)` tuples and rejects any that don't match.

`head_dim=512` requires `NUM_MMA_D_QK=32` which is not in the table → "Invalid
configuration" → hard crash.

This is documented in FlashInfer PR #2959's description:
> "FlashInfer FA2/FA3 kernels don't support head_dim > 256"

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
| flashinfer-ai/flashinfer | [#2959](https://github.com/flashinfer-ai/flashinfer/pull/2959) | [Fmha] Add head_dim=512 support for trtllm attention kernels | **merged** (2026-04-22, in v0.6.10rc1 / v0.6.10 stable 2026-05-04) |
| sgl-project/sglang | [#22079](https://github.com/sgl-project/sglang/pull/22079) | [nvidia] Gemma4 nvfp4 fix | **merged** (2026-04-10) |

PR #22079 in SGLang fixed the **Triton attention** side of the `head_dim=512`
problem (PTX register exhaustion on SM100a/GB200). The companion FlashInfer
attention fix (PR #2959) merged on 2026-04-22 and is in v0.6.10rc1+ (stable
release v0.6.10 was tagged 2026-05-04).

Our `main-gemma4-sm121` image includes PR #22079 (pinned to its merge commit)
but still uses FlashInfer 0.6.7.post3, which **predates** PR #2959. Therefore,
on the currently deployed image:
- `attention_backend=triton` works (PR #22079 fix active)
- `attention_backend=flashinfer` crashes (PR #2959 not yet present in the
  pinned flashinfer wheel)

A rebuild against flashinfer ≥ 0.6.10 should make `attention_backend=flashinfer`
viable for Gemma-4 — needs verification on SM121.

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
  `attention_backend: triton` (workaround).
- `roles/k8s_dgx/model_profiles/google-gemma-4-31b-it.yml` — same.
- NVFP4 profiles are blocked by other bugs before reaching this one, but
  would also need `attention_backend: triton` once unblocked.
