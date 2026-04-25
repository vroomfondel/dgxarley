# SGLang Upstream Bug: Gemma-4 NVFP4 blocked on SM121

## Status (as of 2026-04-25)

- **BF16 variants — WORKING** on our `main-gemma4-sm121` image (SGLang main
  built post-PR-#21952). Both dense (`google/gemma-4-31B-it`) and MoE
  (`google/gemma-4-26B-A4B-it`) deploy and serve, with the MoE producing
  **180.5 tok/s @ n=8** — the fastest model on the cluster. Required:
  `attention_backend=triton` (FlashInfer crashes on `global_head_dim=512`,
  see `FLASHINFER_HEAD_DIM_512_UPSTREAM_BUG.md`).

- **NVFP4 variants — STILL BLOCKED.** Both dense (`nvidia/Gemma-4-31B-IT-NVFP4`)
  and MoE (`bg-digitalservices/Gemma-4-26B-A4B-it-NVFP4`) require four sm120/121-
  specific upstream PRs that have all been **OPEN since 2026-04-16 with no
  movement** (last verified 2026-04-25 via `gh pr view`). Until these merge,
  NVFP4 Gemma-4 cannot run on our SM121 hardware.

The original v0.5.10 blockers (Transformers fallback, dual head_dim, top_k_experts
naming) are no longer relevant for our deployment because we build the image
from SGLang main, not from the v0.5.10 release. The remaining issues are
NVFP4-MoE-on-SM121-specific.

## Affected models

| Model | Type | Quantization | Current status (main-gemma4-sm121 image) |
|-------|------|-------------|----------------------------|
| `google/gemma-4-26B-A4B-it` | MoE (128 experts, 26B/3.8B active) | BF16 | **STABLE ★** — 39.8 / 114.6 / **180.5** tok/s (n=1/4/8) |
| `google/gemma-4-31B-it` | Dense (30.7B) | BF16 | **STABLE ★** — 10.6 / 36.8 / **70.6** tok/s (n=1/4/8) |
| `bg-digitalservices/Gemma-4-26B-A4B-it-NVFP4` | MoE (128 experts, 26B/3.8B active) | NVFP4 | **blocked** — `modelopt_quant.py` crash, needs PRs #22929 + #22928 + #22927 |
| `nvidia/Gemma-4-31B-IT-NVFP4` | Dense (30.7B) | NVFP4 | **untested** — expected to need PR #22928 + #22927 (dense path skips per-expert loading, but the FP4-on-SM121 NaN/scale issues still apply) |

All use `Gemma4ForConditionalGeneration` as architecture. The BF16 variants
get a native SGLang model class via PR #21952 (merged 2026-04-07). The NVFP4
variants additionally hit per-expert weight loading + cutlass FP4 kernel
issues that are SM120/121-specific.

Throughput numbers are from `TESTLOGS/sglang_nn4_tp4_ep1/gemma-4-*/` (run
2026-04-16/17, RoCE/SR-IOV, `attention_backend=triton`, `kv_cache_dtype=fp8_e4m3`,
piecewise CUDA graphs enabled). The BF16 MoE figure is the highest measured
throughput on this cluster across all tested models.

## Root cause

Gemma-4 has several architectural features that the Transformers fallback
backend in v0.5.10 does not support:

1. **Dual head dimensions** — sliding-window layers use `head_dim=256`, global
   attention layers use `global_head_dim=512`. The fallback backend creates
   RMSNorm weights uniformly with one dimension, causing shape mismatches when
   the model alternates between layer types.

2. **MoE config naming** — Gemma-4 uses `top_k_experts` instead of the standard
   `num_experts_per_tok` / `top_k` that the fallback's `_getattr_first` lookup
   expects.

3. **NVFP4 per-expert weight format** — NVFP4 checkpoints store MoE expert
   weights in unfused per-expert format, which the fallback's weight mapper
   doesn't support.

4. **GEGLU activation** — Gemma-4 MoE uses GEGLU (`gelu_tanh`), but
   `cutlass_moe_fp4()` hardcodes `silu_and_mul()`.

5. **FP4 block scale NaN** — SM120/121 specific: uint8=127 in E4M3 block scales
   triggers NaN in the CUTLASS FP4 group GEMM kernel.

Issue (1) affects **all** Gemma-4 variants (BF16 and NVFP4, dense and MoE).
Issues (2–5) affect NVFP4 variants specifically.

## Failure details

### BF16 MoE (`google/gemma-4-26B-A4B-it`) — confirmed

Crash during warmup forward at the first global-attention layer's `v_norm`:

```
gemma4/modeling_gemma4.py:1220  value_states = self.v_norm(value_states)
  → layernorm.py:207  rmsnorm(x, self.weight.data, self.variance_epsilon)
  → flashinfer/norm/rmsnorm.py:1310  kernel(...)
ValueError: Mismatched mW.shape[0] on argument #1 when calling:
  `__call__(mX: Tensor([n0, 256], bfloat16), mW: Tensor([256], bfloat16),
            mY: Tensor([n0, 256], bfloat16), M: int32, eps: float32)`,
  expected to be 256
```

The `v_norm` layer has weight `[256]` (sliding-window `head_dim`), but on a
global-attention layer the value states have dimension 512 (`global_head_dim`).
The Transformers fallback creates all attention norms with the same dimension,
not distinguishing between sliding and global layers. The native implementation
(PR #21952) has separate norm configs per layer type.

### NVFP4 MoE (`bg-digitalservices/Gemma-4-26B-A4B-it-NVFP4`) — confirmed

Three sequential failures, each uncovered after patching the previous:

1. **`Cannot determine top_k from config`** — `_getattr_first` lookup tuple
   in `transformers.py:1197` doesn't include `top_k_experts`.
   - Runtime-patched in `sglang_launch.sh` (`PATCH_TRANSFORMERS_TOPK_EOF`).
   - First patch revision had a syntax bug: inline marker comment broke the
     closing paren of `_getattr_first(...)` → `'(' was never closed` on line
     1197. Fixed by placing marker on a separate line above.

2. **`No module or parameter named 'model.language_model.layers.0.moe'`** —
   NVFP4 checkpoints store MoE expert weights in unfused per-expert format.
   The Transformers backend's weight mapper only knows the fused format.
   - **Not runtime-patchable.**
   - **Upstream fix: PR #22929** (open, 2026-04-16).

3. **(Latent) GEGLU activation mismatch** — `cutlass_moe_fp4()` hardcodes
   `silu_and_mul()`. Gemma-4 MoE uses GEGLU → garbage output even if weights
   loaded.
   - **Upstream fix: PR #22928** (open, 2026-04-16).

### Dense variants (BF16 + NVFP4)

Not separately tested. `_is_moe_model()` returns `False` → dispatches to
`TransformersMultiModalForCausalLM` (no MoEMixin) → avoids issues 1–3 above
but still hits the dual head_dim RMSNorm crash (issue 1 in the root cause list),
which is shared across all variants.

## Upstream PRs

Last `gh pr view` check: 2026-04-25.

| PR | Title | Status | Merged | Relevance |
|----|-------|--------|--------|-----------|
| [#21952](https://github.com/sgl-project/sglang/pull/21952) | [New Model] Gemma 4 | **merged** | 2026-04-07 | Native `gemma4_causal.py`, `gemma4_mm.py`, `gemma4_vision.py`, `gemma4_audio.py`. Foundation for all Gemma-4 support. Fixes the dual head_dim issue. **In our `main-gemma4-sm121` image — BF16 variants run thanks to this.** |
| [#22079](https://github.com/sgl-project/sglang/pull/22079) | [nvidia] Gemma4 nvfp4 fix | **merged** | 2026-04-10 | Triton attention PTX register exhaustion fix for NVFP4 on GB200/sm100a. fp8 kv cache dtype autodetection. In our image. |
| [#22929](https://github.com/sgl-project/sglang/pull/22929) | Add NVFP4 per-expert weight loading for Gemma 4 MoE | **open** | — | Per-expert → fused weight mapping for NVFP4 MoE checkpoints. **No movement since 2026-04-16.** |
| [#22928](https://github.com/sgl-project/sglang/pull/22928) | fix(sm120): MoE GEGLU activation + FP4 block scale NaN clamp | **open** | — | GEGLU activation for `cutlass_moe_fp4()` + E4M3 NaN clamp. SM120/121 critical. **No movement since 2026-04-16.** |
| [#22927](https://github.com/sgl-project/sglang/pull/22927) | fix(sm120): NVFP4 NaN from E4M3 scale overflow + 3D tensor shape crashes | **open** | — | Sister PR to #22928, also SM120/121-specific. Affects NVFP4 dense + MoE both. **No movement since 2026-04-16.** |
| [#22615](https://github.com/sgl-project/sglang/pull/22615) | Fix fp8 KV cache crash with KV-shared layers in triton backend | **open** | — | fp8 kv cache + `num_kv_shared_layers > 0` (Gemma-4 has KV-shared layers). Open since 2026-04-12. |
| [#22408](https://github.com/sgl-project/sglang/pull/22408) | [CI] Adding Gemma 4 to Nightly CI | **merged** | 2026-04-17 | Adds Gemma-4 to nightly accuracy tests. Increases pressure on the open NVFP4 PRs to land cleanly but doesn't itself fix anything for us. |
| [#23575](https://github.com/sgl-project/sglang/pull/23575) | [AMD] fused qk gemma norm kernels | **merged** | 2026-04-25 | AMD-specific perf optimization, no impact on our NVIDIA SM121 deployment. |

## What's needed to run Gemma-4 on our cluster

### BF16 variants (google/gemma-4-*) — DONE

Minimum was PR #21952 (native Gemma-4). Already merged into main and baked
into our `xomoxcc/dgx-spark-sglang:main-gemma4-sm121` image. Both
`google/gemma-4-31B-it` (dense) and `google/gemma-4-26B-A4B-it` (MoE) deploy
and serve. The model profiles in `roles/k8s_dgx/model_profiles/` are pinned
to the working configuration:

- `attention_backend: triton` (mandatory — FlashInfer crashes on `head_dim=512`)
- `kv_cache_dtype: fp8_e4m3`
- `mem_fraction_static: 0.85`
- `disable_piecewise_cuda_graph: false` (BF16 is unaffected by the fp4-quantize
  dynamo bug; piecewise gives ~6.5% over fixed-BS graphs at n=8)

To activate: set `sglang_active_model` in your inventory and run
`ansible-playbook k8s_dgx.yml --tags sglang -e sglang_enabled=true`.

### NVFP4 variants (nvidia/*, bg-digitalservices/*) — STILL BLOCKED

All of the following must be present:

1. PR #21952 — native Gemma-4 model implementation (merged ✓)
2. PR #22079 — NVFP4 quantization + fp8 kv cache fixes (merged ✓)
3. PR #22929 — per-expert NVFP4 weight loading for MoE (**open**)
4. PR #22928 — GEGLU activation + FP4 block scale NaN clamp, SM120/121 (**open**)
5. PR #22927 — NVFP4 NaN from E4M3 scale overflow + 3D tensor shape, SM120/121 (**open**)
6. PR #22615 — fp8 kv cache with KV-shared layers (**open**, may or may not apply)

The four open PRs (#22929, #22928, #22927, #22615) have all been sitting since
2026-04-12/16 with no review activity through 2026-04-25. Until they merge
upstream, we have three options:

- **Wait** — most upstream-maintenance-friendly. No work for us until merge.
- **Vendor the open PRs as our own patches** in `scripts/build_sm121_image.sh`
  (similar to the existing Gemma-4 patches). Risk: PRs are still under review
  and may change — we'd have to re-rebase if upstream tweaks them. Pre-condition:
  PR #22929 and #22928 were developed on RTX 5090 (SM120); they need
  validation on SM121/GB10, which we'd be the first to do.
- **Comment on the PRs with our SM121 test data**, push for review and merge.
  Cheapest, most likely to actually unblock things if the maintainers are
  waiting on SM121 confirmation.

## Our runtime patches (v0.5.10)

The `top_k_experts` patch in `sglang_launch.sh` (`PATCH_TRANSFORMERS_TOPK_EOF`)
remains useful — it fixes the `_getattr_first` lookup for any future model that
uses `top_k_experts` instead of `num_experts_per_tok`. However, it's insufficient
to make any Gemma-4 variant work on v0.5.10 because the dual head_dim, weight
loading, and activation function issues are not patchable at runtime.

## Relationship to other bugs

- **Independent of** the FlashInfer FP4 dynamo tracing bug
  (`FLASHINFER_CUDA_VERSION_SUBPROCESS_UPSTREAM_BUG.md`) — that affects
  piecewise CUDA graphs on all NVFP4 models, not Gemma-4 specifically.
- **Independent of** the SM121 JIT arch mismatch (`kvcache.cuh:196` illegal
  instruction) — that's in sglang's own jit_kernel, not the model loader.
- **Related to** issue #22277 (Gemma4 E4B fp8 KV cache crash) — same model
  family, overlapping root cause (KV-shared layers + fp8).

## Files

- `roles/k8s_dgx/files/sglang_launch.sh` — `top_k_experts` runtime patch
  (left in place; harmless on the main-gemma4-sm121 image, useful as a
  safety net for any other model that ever uses `top_k_experts`).
- `roles/k8s_dgx/model_profiles/google-gemma-4-26b-a4b-it.yml` — BF16 MoE
  profile, **production-ready** with the main-gemma4-sm121 image.
- `roles/k8s_dgx/model_profiles/google-gemma-4-31b-it.yml` — BF16 dense
  profile, **production-ready** with the main-gemma4-sm121 image.
- `roles/k8s_dgx/model_profiles/bg-digitalservices-gemma-4-26b-a4b-it-nvfp4.yml`
  — NVFP4 MoE profile, blocked on PRs #22929 + #22928 + #22927.
- `roles/k8s_dgx/model_profiles/nvidia-gemma-4-31b-it-nvfp4.yml` — NVFP4 dense
  profile, blocked on PRs #22928 + #22927 (per-expert loading not needed
  for dense, but FP4-on-SM121 NaN/scale issues still apply).
- `scripts/build_sm121_image.sh` — applies our locally vendored Gemma-4
  patches (`sglang-gemma4-nvfp4-expert-loading.patch`,
  `sglang-gemma4-geglu-nan-clamp.patch`, `dockerfile-gemma4-nvfp4.patch`).
  These are placeholders / early attempts at the upstream fixes — useful
  as a starting point if we decide to vendor the open PRs.
- `TESTLOGS/sglang_nn4_tp4_ep1/gemma-4-26b-a4b-it/TESTLOG_*.md` — full BF16
  MoE config sweep, 2026-04-16. Test 6 = winner (180.5 tok/s @ n=8).
- `TESTLOGS/sglang_nn4_tp4_ep1/gemma-4-31b-it/TESTLOG_*.md` — full BF16
  dense config sweep, 2026-04-17. Test 6 = winner (70.6 tok/s @ n=8).
- `TESTLOGS/sglang_nn4_tp4_ep1/gemma-4-26b-a4b-it-nvfp4/TESTLOG_*.md` —
  36-config NVFP4 MoE sweep, 2026-04-16. All blocked at `modelopt_quant.py`.
