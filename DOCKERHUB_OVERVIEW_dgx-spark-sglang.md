<!-- short: SGLang for NVIDIA DGX Spark / GB10 (SM121) — CUTLASS NVFP4 patches, arm64, built from source. -->

# dgx-spark-sglang

Custom [SGLang](https://github.com/sgl-project/sglang) container images for the
**NVIDIA DGX Spark / ASUS Ascent GX10 (GB10, SM121, arm64)**.

Upstream SGLang / sgl-kernel binaries do not target `sm_121` and silently fall
back to JIT or to kernels that crash on the GB10's 101 KB shared-memory budget
(notably the `cutlass_moe_fp4` NVFP4 MoE path — device-side assert at
`nvfp4_blockwise_moe.cuh:78`). These images carry a small stack of patches
against `sgl-kernel` that make the NVFP4 MoE runner fit SM121 and prune
Hopper-only kernels (FA3, sm90 targets, FlashMLA) that never run on GB10.

- **Source**: [github.com/vroomfondel/dgxarley](https://github.com/vroomfondel/dgxarley)
- **Hardware target**: NVIDIA GB10 / SM121 (DGX Spark, ASUS Ascent GX10) — arm64 only
- **License**: Apache-2.0 (same as SGLang)

## What's inside

- **SGLang** built from upstream tags (currently `v0.5.11`)
- **sgl-kernel** with SM121 patches: CUTLASS NVFP4 blockwise MoE
  (`StageCount<1>` + `KernelPtrArrayTmaWarpSpecialized`), arch-prune to
  `sm_121` only, FA3 / sm90 / FlashMLA stripped
- **flashinfer** bumped to a version with the `head_dim=512` fix
  (unblocks Gemma-4 global attention)
- Built on a CUDA 13.2 + PyTorch 2.11 base for the GB10 codegen path
  (CUDA 13.1 / PyTorch 2.10 fallback is ~45 % slower end-to-end)

## Tags

| Tag                                 | Notes                                                        |
|-------------------------------------|--------------------------------------------------------------|
| `0.5.11-sm121`                      | SGLang v0.5.11 + SM121 patches (current default)             |
| `0.5.11-gemma4-sm121`               | + unmerged Gemma-4 NVFP4 source patches (PRs #22929, #22928) |
| `0.5.10-20260429-sm121-dev1`        | Legacy v0.5.10 line, kept for rollback / A/B                 |
| `0.5.10-20260429-gemma4-sm121-dev1` | Legacy v0.5.10 + Gemma-4 patches                             |

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
- [`scripts/patches/sglang-0.5.11-sm121.recipe`](https://github.com/vroomfondel/dgxarley/blob/main/scripts/patches/sglang-0.5.11-sm121.recipe)
  — recipe pinned by the build
- [`scripts/patches/sgl-kernel-sm121.patch`](https://github.com/vroomfondel/dgxarley/blob/main/scripts/patches/sgl-kernel-sm121.patch)
  — the core CUTLASS NVFP4 SM121 fix
- `CUTLASS_NVFP4_SM121_PRD.md` — root cause + fix rationale (in repo)
- [`roles/k8s_dgx/`](https://github.com/vroomfondel/dgxarley/tree/main/roles/k8s_dgx)
  — Ansible role that deploys SGLang head + workers (Multus + RoCE-over-SR-IOV
  NCCL, HAProxy sidecar for the head's EADDRINUSE workaround, model profiles)

## Status / support

These images are built and exercised on a private 4-node DGX Spark cluster
(`spark1`–`spark4`). They are published in case someone else has the same
hardware and runs into the same SM121 crashes — there is no commercial support
and tags may be retagged or removed without notice. Open an issue on the
GitHub repo if something is broken.
