<!-- short: SGLang for NVIDIA DGX Spark / GB10 (SM121) — CUTLASS NVFP4 + DeepSeek-V4 FlashMLA patches, arm64, built from source. -->

# dgx-spark-sglang

Custom [SGLang](https://github.com/sgl-project/sglang) container images for the
**NVIDIA DGX Spark / ASUS Ascent GX10 (GB10, SM121, arm64)**.

Upstream SGLang / sgl-kernel binaries do not target `sm_121` and silently fall
back to JIT or to kernels that crash on the GB10's 101 KB shared-memory budget
(notably the `cutlass_moe_fp4` NVFP4 MoE path — device-side assert at
`nvfp4_blockwise_moe.cuh:78`). These images carry a stack of patches against
`sgl-kernel` that make the NVFP4 MoE runner fit SM121 and prune Hopper-only
kernels (FA3, sm90 targets, FlashMLA) that never run on GB10 — and, for the
DeepSeek-V4-Flash path, **vendor an sm_121 FlashMLA sparse-decode kernel back
in** (see below), since `sm_121` is a compute-capability tier the DeepGEMM /
FlashMLA / cutlass kernel ecosystem does not yet ship kernels for.

- **Source**: [github.com/vroomfondel/dgxarley](https://github.com/vroomfondel/dgxarley)
- **Hardware target**: NVIDIA GB10 / SM121 (DGX Spark, ASUS Ascent GX10) — arm64 only
- **License**: Apache-2.0 (same as SGLang)

## What's inside

- **SGLang** built from upstream tags (currently `v0.5.12.post1`)
- **sgl-kernel** with SM121 patches: CUTLASS NVFP4 blockwise MoE
  (`StageCount<1>` + `KernelPtrArrayTmaWarpSpecialized`), arch-prune to
  `sm_121` only, FA3 / sm90 / FlashMLA stripped (the bundled FlashMLA is
  Hopper-only; a standalone sm_121 build is vendored separately for V4 — below)
- **DeepSeek-V4-Flash sparse-decode kernel (sm_121, experimental)** — V4's
  attention backend hard-`import`s `flash_mla` with no fallback, and upstream
  FlashMLA ships no sm_120/sm_121 sparse-decode kernel, so
  `sgl-project/DeepSeek-V4-Flash-FP8` otherwise dies at the first forward
  (`ModuleNotFoundError: flash_mla` / `Unsupported architecture for sparse
  decode fwd`). These images install stock
  [`deepseek-ai/FlashMLA`](https://github.com/deepseek-ai/FlashMLA) (interface +
  host-side `get_mla_metadata`) plus an sm_121a-retargeted build of
  [`0xSero/deepseek-v4-flash-sm120`](https://github.com/0xSero/deepseek-v4-flash-sm120)'s
  sparse-decode CUDA extension, monkey-patched in at interpreter start so the
  sparse-FP8-decode path lands in the sm_121 kernel (inert for non-V4 models,
  which fall through to stock FlashMLA). **First-contact / unvalidated.** Full
  wall-by-wall breakdown — DeepGEMM `hc_prenorm` + `paged_mqa_logits` TileLang/
  torch fallbacks, `wo_a` fp8→bf16 (`SGLANG_OPT_FP8_WO_A_GEMM=0`),
  `mem_fraction_static`, node swap for the load peak — in
  [`UPSTREAM_DSV4_BUGS.md`](https://github.com/vroomfondel/dgxarley/blob/main/UPSTREAM_DSV4_BUGS.md).
- **flashinfer** bumped to a version with the `head_dim=512` fix
  (unblocks Gemma-4 global attention)
- **transformers pinned to `5.8.0`** (released 2026-05-05) — required for
  the Gemma-4 `*-assistant` drafter checkpoints used by NEXTN/MTP
  speculative decoding (`google/gemma-4-{26B-A4B,31B}-it-assistant`).
  Earlier transformers releases don't know the drafter's config subclass
  and the SGLang head exits with `Unrecognized configuration class` during
  drafter weight-loading.
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
- Built on a CUDA 13.2 + PyTorch 2.12 base for the GB10 codegen path
  (CUDA 13.1 / PyTorch 2.10 fallback is ~45 % slower end-to-end)

## Tags

| Tag                                 | Notes                                                                       |
|-------------------------------------|------------------------------------------------------------------------------|
| `0.5.12.post1-sm121`                | SGLang v0.5.12.post1 + SM121 patches + DeepSeek-V4-Flash FlashMLA kernel (current default) |
| `0.5.11-sm121`                      | SGLang v0.5.11 + SM121 patches (previous default, kept for rollback / A/B)   |
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
- [`scripts/patches/sglang-0.5.12-sm121.recipe`](https://github.com/vroomfondel/dgxarley/blob/main/scripts/patches/sglang-0.5.12-sm121.recipe)
  — recipe pinned by the build (SGLang + flashinfer + FlashMLA + V4 kernel pins)
- [`scripts/patches/sgl-kernel-sm121.patch`](https://github.com/vroomfondel/dgxarley/blob/main/scripts/patches/sgl-kernel-sm121.patch)
  — the core CUTLASS NVFP4 SM121 fix
- [`scripts/patches/dockerfile-dsv4-flashmla.patch`](https://github.com/vroomfondel/dgxarley/blob/main/scripts/patches/dockerfile-dsv4-flashmla.patch)
  — builds the vendored sm_121 FlashMLA sparse-decode kernel for DeepSeek-V4-Flash
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
