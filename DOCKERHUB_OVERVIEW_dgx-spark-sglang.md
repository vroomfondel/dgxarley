<!-- short: SGLang for NVIDIA DGX Spark / GB10 (SM121): CUTLASS NVFP4 + DeepSeek-V4 NVFP4 MoE, arm64 -->

# dgx-spark-sglang

Custom [SGLang](https://github.com/sgl-project/sglang) container images for the
**NVIDIA DGX Spark / ASUS Ascent GX10 (GB10, SM121, arm64)**.

Upstream SGLang / sgl-kernel binaries do not target `sm_121` and silently fall
back to JIT or to kernels that crash on the GB10's 101 KB shared-memory budget
(notably the `cutlass_moe_fp4` NVFP4 MoE path — device-side assert at
`nvfp4_blockwise_moe.cuh:78`). These images carry a stack of patches against
`sgl-kernel` that make the NVFP4 MoE runner fit SM121 and prune Hopper-only
kernels (FA3, sm90 targets, FlashMLA) that never run on GB10. For the
DeepSeek-V4-Flash path they also **carry the unmerged DeepSeek-V4 NVFP4 MoE
support** (upstream [PR #25820](https://github.com/sgl-project/sglang/pull/25820),
rebased onto v0.5.13) so the `nvidia/DeepSeek-V4-Flash-NVFP4` checkpoint can be
served on SM121 at all (see below) — `sm_121` is a compute-capability tier the
DeepGEMM / FlashMLA / cutlass kernel ecosystem is only now starting to ship
kernels for.

- **Source**: [github.com/vroomfondel/dgxarley](https://github.com/vroomfondel/dgxarley)
- **Hardware target**: NVIDIA GB10 / SM121 (DGX Spark, ASUS Ascent GX10) — arm64 only
- **License**: Apache-2.0 (same as SGLang)

## What's inside

- **SGLang** built from upstream tags (currently `v0.5.14`)
- **sgl-kernel** with SM121 patches: CUTLASS NVFP4 blockwise MoE
  (`StageCount<1>` + `KernelPtrArrayTmaWarpSpecialized`), arch-prune to
  `sm_121` only, FA3 / sm90 / FlashMLA stripped (the bundled FlashMLA is
  Hopper-only)
- **DeepSeek-V4-Flash NVFP4 MoE (sm_121, experimental)** — the `0.5.13-sm121`
  tag carries upstream [PR #25820](https://github.com/sgl-project/sglang/pull/25820)
  ("DeepSeek-V4 NVFP4 MoE", unmerged at build time) rebased onto v0.5.13, so the
  `nvidia/DeepSeek-V4-Flash-NVFP4` checkpoint can be served on GB10. PR #25820
  is upstream-validated only on B200 (SM100) and default-routes to
  `flashinfer_trtllm_routed`, which is not runnable on SM121
  ([#26324](https://github.com/sgl-project/sglang/issues/26324)) — the model
  profile pins `flashinfer_cutlass` explicitly. **First-contact / unvalidated.**
  V4's sparse-decode path itself no longer needs a vendored kernel as of
  v0.5.13: upstream [PR #24692](https://github.com/sgl-project/sglang/pull/24692)
  ships a native SM120/121 Triton path (`major==12`, covers GB10), so the
  sm_121a-retargeted [`0xSero/deepseek-v4-flash-sm120`](https://github.com/0xSero/deepseek-v4-flash-sm120)
  kernel bake that earlier tags (`0.5.12.post1-sm121` and below) used is
  **dropped** here. Full wall-by-wall breakdown — DeepGEMM `hc_prenorm` +
  `paged_mqa_logits` torch fallbacks, `wo_a` fp8→bf16
  (`SGLANG_OPT_FP8_WO_A_GEMM=0`), `mem_fraction_static`, node swap for the load
  peak — in
  [`UPSTREAM_DSV4_BUGS.md`](https://github.com/vroomfondel/dgxarley/blob/main/UPSTREAM_DSV4_BUGS.md).
- **flashinfer bumped to `0.6.13rc2`** (over the v0.5.13 upstream pin of
  `0.6.12`). `0.6.13rc2` (tagged 2026-06-17) lands flashinfer
  [PR #3576](https://github.com/flashinfer-ai/flashinfer/pull/3576)
  (`head_dim=512` dispatch for SM120/121) plus NVFP4 quant-kernel
  improvements. **Caveat for Gemma-4:** SGLang's own attention-backend
  allowlist hard-rejects `flashinfer` for the Gemma-4 architecture (only
  `trtllm_mha | triton | intel_xpu` are accepted), so PR #3576 turns out to
  be moot for Gemma *attention* — the Gemma-4 profiles still set
  `attention_backend=triton` (no longer a flashinfer-version limitation but a
  SGLang allowlist constraint). The 0.6.13rc2 bump still pays off on the
  NVFP4 MoE quant path. Roll back to the upstream pin with
  `FLASHINFER_VERSION=0.6.12`.
- **transformers pinned to `5.8.1`** (exactly SGLang v0.5.13's pyproject
  pin) — required for the Gemma-4 `*-assistant` drafter checkpoints used by
  NEXTN/MTP speculative decoding (`google/gemma-4-{26B-A4B,31B}-it-assistant`).
  Earlier transformers releases don't know the drafter's config subclass
  and the SGLang head exits with `Unrecognized configuration class` during
  drafter weight-loading. **Exception:** the `0.5.14-gemmadiffusion-sm121`
  image pins `5.11.0` instead — `diffusion_gemma` is an unregistered
  `model_type` before then (AutoConfig `KeyError`), and 5.11.0 is the
  version DiffusionGemma's upstream PR #28054 pins.
- **Gemma-4 MTP (Frozen-KV) speculative-decoding patch** — the
  `0.5.11-gemma4-sm121` tag carries a cherry-pick of upstream
  [PR #24436](https://github.com/sgl-project/sglang/pull/24436)
  ("Gemma 4 — Adding MTP support", merged 2026-05-07, after the v0.5.11
  release tag). **Native in v0.5.12+** — the `0.5.12*` tags ship it from
  upstream and the cherry-pick is no longer applied. Adds the dedicated `Gemma4AssistantForCausalLM` model and a
  new `FROZEN_KV_MTP` speculative algorithm (recurrent hidden-state draft
  loop with frozen target KV cache). At runtime SGLang auto-promotes
  `--speculative-algorithm NEXTN → FROZEN_KV_MTP` once the drafter is
  detected. Without this patch the stock NEXTN/EAGLE worker crashes with
  `ValueError: No module or parameter named 'model.language_model' in
  TransformersMultiModalForCausalLM` during drafter weight-load.
  Verified working on the 4-node DGX Spark cluster — see the 31B-it
  TESTLOG, [Test 07 (`num_steps=2`, `num_draft_tokens=3`)](https://github.com/vroomfondel/dgxarley/blob/main/TESTLOGS/sglang_nn4_tp4_ep1/gemma-4-31b-it/TESTLOG_nv580.142_sglang-0.5.11_gemma-4-31b-it_4n.md#mtp-sweep-tests-711--partial-15-cases-done):
  **+98 % at n=1** (10.49 → 20.83 tok/s), **+76 % at n=4** (44.06 → 77.67
  tok/s), drafter acceptance rate median ~0.68, 5/5 requests stopped on
  natural EOS. The 26B-A4B MoE sibling's MTP sweep is still in progress
  ([TESTLOG](https://github.com/vroomfondel/dgxarley/blob/main/TESTLOGS/sglang_nn4_tp4_ep1/gemma-4-26b-a4b-it/TESTLOG_nv580.142_sglang-0.5.11_gemma-4-26b-a4b-it_4n.md)).
- **NemotronH MTP + radix cache (experimental)** — the
  `0.5.13-dev-nemotronh-mtp-sm121` tag carries upstream
  [PR #27998](https://github.com/sgl-project/sglang/pull/27998) (unmerged),
  which enables native MTP speculative decoding for
  `nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4` **without**
  `--disable-radix-cache`. It differs from the production `0.5.13-sm121` tag
  by exactly this one Python-only source patch so the MTP behaviour can be
  A/B'd against production. **First-contact / unvalidated** — validate
  `accept_len > 1` and no NaN logits (an NVFP4-MTP risk,
  [#27828](https://github.com/sgl-project/sglang/issues/27828)) on first boot.
- **DiffusionGemma dLLM (experimental, main-ahead)** — the
  `0.5.14-gemmadiffusion-sm121` tag is **the unified Gemma-4 image**: one
  build that serves all five Gemma-4 profiles — BF16 + FROZEN_KV_MTP,
  NVFP4, and the diffusion-LLM `nvidia/diffusiongemma-26B-A4B-it-NVFP4`. It
  is pinned forward to SGLang `main @ 3a1417a` (post-v0.5.13, 2026-06-12)
  rather than the v0.5.13 tag because (1)
  [PR #28081](https://github.com/sgl-project/sglang/pull/28081) fixes a
  short-lived broken `FrozenKVMTPCudaGraphRunner` that crashes Gemma-4 BF16
  MTP at boot on the v0.5.13 tag, and (2) the DiffusionGemma bake
  ([PR #28054](https://github.com/sgl-project/sglang/pull/28054), unmerged)
  applies far more cleanly main-vs-main. Carries the Gemma-4 GeGLU/FP4
  NaN-clamp source patch ([PR #22928](https://github.com/sgl-project/sglang/pull/22928))
  and uses the `-mainahead` sgl-kernel patch variants (one day of main drift
  shifted the mscclpp link lines). **First-contact / main-ahead, not a
  tagged release.**
- Built on a CUDA 13.2 + PyTorch 2.12 + NCCL 2.30.7 base for the GB10 codegen
  path (CUDA 13.1 / PyTorch 2.10 fallback is ~45 % slower end-to-end). **Known
  issue:** the NCCL 2.30.x NVLS path has a regression that can silently hang
  high-expert-count MoE weight loads (≥256 experts) on GB10/RoCE
  ([NVIDIA/nccl#2167](https://github.com/NVIDIA/nccl/issues/2167)) — set
  `NCCL_NVLS_ENABLE=0` when running these (free on non-NVLink hardware; the
  Ansible role does this for you)

## Tags

| Tag                                 | Notes                                                                       |
|-------------------------------------|------------------------------------------------------------------------------|
| `0.5.14-sm121`                      | SGLang v0.5.14 + SM121 patches; native NVFP4-MoE dispatch (PR #25820), native MTP for Nemotron-3 Super 120B, ModelOptMixedPrecisionConfig / W4A16_NVFP4 support for Qwen3.6; flashinfer 0.6.13 — **(current)** |
| `0.5.13-sm121`                      | SGLang v0.5.13 + SM121 patches + DeepSeek-V4 NVFP4 MoE (PR #25820); native SM120/121 FlashMLA (PR #24692), no vendored kernel; flashinfer 0.6.13rc2 |
| `0.5.14-gemmadiffusion-sm121`       | **Unified Gemma-4 image** — main-ahead (`3a1417a`, post-v0.5.13) serving all five Gemma-4 profiles incl. DiffusionGemma dLLM (PR #28054) + FROZEN_KV_MTP fix (PR #28081). First-contact |
| `0.5.13-gemmadiffusion-sm121`       | Gemma-4 diffusion build on v0.5.13 base (2026-06-19); rollback / A/B against `0.5.14-gemmadiffusion-sm121` |
| `0.5.13-dev-nemotronh-mtp-sm121`    | v0.5.13 + SM121 patches + NemotronH MTP/radix-cache (unmerged PR #27998); A/B against `0.5.13-sm121`. Experimental |
| `0.5.13-gemma4-sm121`               | v0.5.13 + SM121 patches + Gemma-4 NVFP4 source patch (PR #22928); for NVFP4 Gemma-4 on flashinfer 0.6.13rc2 (DSV4 deliberately omitted — mutually exclusive) |
| `0.5.12.post1-sm121`                | SGLang v0.5.12.post1 + SM121 patches + vendored sm_121 DeepSeek-V4-Flash FlashMLA kernel (previous, kept for rollback / A/B) |
| `0.5.12-sm121`                      | SGLang v0.5.12 + SM121 patches                                              |
| `0.5.12-gemma4-sm121`               | v0.5.12 + Gemma-4 NVFP4 source patches                                       |
| `0.5.11-sm121`                      | SGLang v0.5.11 + SM121 patches (kept for rollback / A/B)                     |
| `0.5.11-gemma4-sm121`               | v0.5.11 + unmerged Gemma-4 NVFP4 source patches (PRs #22929, #22928) + MTP cherry-pick |
| `0.5.10-20260429-sm121-dev1`        | Legacy v0.5.10 line, kept for rollback / A/B                                 |
| `0.5.10-20260429-gemma4-sm121-dev1` | Legacy v0.5.10 + Gemma-4 patches                                             |

All tags are **`linux/arm64` only** — these images are useless on x86_64 and on
non-GB10 NVIDIA hardware (the kernels are arch-pruned to `sm_121`).

## Build & deploy context

Everything that produces these images — Dockerfiles, sgl-kernel patches, recipe
files, the cross-arch podman build driver, and the Ansible roles that deploy
SGLang on a 4-node DGX Spark K3s cluster — lives in
[github.com/vroomfondel/dgxarley](https://github.com/vroomfondel/dgxarley).

Relevant entry points:

- [`scripts/build_sm121_image.sh`](https://github.com/vroomfondel/dgxarley/blob/main/scripts/build_sm121_image.sh)
  — remote-podman build driver (x86 control host → arm64 build runner)
- [`scripts/patches/sglang-0.5.13-sm121.recipe`](https://github.com/vroomfondel/dgxarley/blob/main/scripts/patches/sglang-0.5.13-sm121.recipe)
  — recipe pinned by the build (SGLang + flashinfer + transformers + DSV4 NVFP4 pins)
- [`scripts/patches/sgl-kernel-sm121.patch`](https://github.com/vroomfondel/dgxarley/blob/main/scripts/patches/sgl-kernel-sm121.patch)
  — the core CUTLASS NVFP4 SM121 fix
- [`scripts/patches/sglang-dsv4-nvfp4-pr25820.patch`](https://github.com/vroomfondel/dgxarley/blob/main/scripts/patches/sglang-dsv4-nvfp4-pr25820.patch)
  — DeepSeek-V4 NVFP4 MoE support (upstream PR #25820, rebased onto v0.5.13)
- `CUTLASS_NVFP4_SM121_PRD.md` — NVFP4 root cause + fix rationale (in repo)
- `UPSTREAM_DSV4_BUGS.md` — DeepSeek-V4-Flash sm_121 boot chain, wall by wall (in repo)
- [`roles/k8s_dgx/`](https://github.com/vroomfondel/dgxarley/tree/main/roles/k8s_dgx)
  — Ansible role that deploys SGLang head + workers (Multus + RoCE-over-SR-IOV
  NCCL, HAProxy sidecar for the head's EADDRINUSE workaround, model profiles)

## Status / support

These images are built and exercised on a private 4-node DGX Spark cluster
(`spark1`–`spark4`). They are published in case someone else has the same
hardware and runs into the same SM121 crashes — there is no commercial support
and tags may be retagged or removed without notice. Open an issue on the
GitHub repo if something is broken.
