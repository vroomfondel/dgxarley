<!-- short: PyTorch 2.12 + CUDA 13.2.1 + NCCL 2.30 base image for DGX Spark / GB10 (SM121), arm64, from source. -->

# dgx-spark-pytorch-dev

Custom **PyTorch + CUDA + NCCL base image** for the
**NVIDIA DGX Spark / ASUS Ascent GX10 (GB10, SM121, arm64)**.

This is the base layer used by
[`xomoxcc/dgx-spark-sglang`](https://hub.docker.com/r/xomoxcc/dgx-spark-sglang).
It exists because the upstream
[`scitrera/dgx-spark-pytorch-dev:2.10.0-v2-cu131`](https://hub.docker.com/r/scitrera/dgx-spark-pytorch-dev)
fallback (PyTorch 2.10 / CUDA 13.1) is **~45 % slower end-to-end on GB10**
than a 2.12 / CUDA 13.2.1 build, due to nvcc codegen + cuBLAS/cuDNN
regressions across that toolchain delta.

- **Source / build script**: [github.com/vroomfondel/dgxarley](https://github.com/vroomfondel/dgxarley)
  (see [`scripts/build_pytorch_base_image.sh`](https://github.com/vroomfondel/dgxarley/blob/main/scripts/build_pytorch_base_image.sh) and
  [`scripts/patches/pytorch-2.12.0-dev-v1.recipe`](https://github.com/vroomfondel/dgxarley/blob/main/scripts/patches/pytorch-2.12.0-dev-v1.recipe))
- **Hardware target**: NVIDIA GB10 / SM121 (DGX Spark, ASUS Ascent GX10) — arm64 only
- **License**: same upstream licenses as PyTorch / NCCL / CUDA components

## What's inside

- **PyTorch 2.12.0** — built from source for `sm_120` + `sm_121`. The 2.12 bump
  brings cuBLAS Blackwell 32-MiB workspaces ([PyTorch PR #175344](https://github.com/pytorch/pytorch/pull/175344)),
  a direct GB10 win
- **torchvision 0.27.0** — lockstep with torch 2.12.0 (PyPI strict-requires it)
- **torchaudio 2.11.0** — not bumped to 2.12 (pytorch/audio hadn't tagged 2.12
  at build time); the ABI lag is harmless for SGLang text-only inference, where
  the audio module is never imported
- **NCCL 2.30.7** — built from upstream
  [`NVIDIA/nccl`](https://github.com/NVIDIA/nccl) at `v2.30.7-1`. (Earlier builds
  pinned the `zyang-dev/nccl` `dgxspark-3node-ring` fork at `2.29.7-1`; reviewing
  that fork's diff showed its sole patch is a **default-off** subnet-aware-routing
  feature — a verified no-op on our single-`/24` switched QSFP fabric — so we
  track upstream now and lose nothing.) **Note:** the NCCL 2.30.x NVLS path has a
  regression that can hang high-expert-count MoE weight loads on GB10/RoCE
  ([NVIDIA/nccl#2167](https://github.com/NVIDIA/nccl/issues/2167)); set
  `NCCL_NVLS_ENABLE=0` at runtime (free on these non-NVLink systems)
- **CUDA 13.2.1** runtime + headers
- Built via the upstream `scitrera/cuda-containers` `pytorch_builder` stage
  (`Dockerfile.base`) with our recipe pinned in the build script

## Tags

| Tag               | Notes                                                                            |
|-------------------|----------------------------------------------------------------------------------|
| `2.12.0-v1-cu132` | PyTorch 2.12.0 + CUDA 13.2.1 + torchvision 0.27.0 + NCCL 2.30.7, arm64 (current) |
| `2.11.0-v1-cu132` | PyTorch 2.11.0 + CUDA 13.2.0 + NCCL 2.30.4, arm64 (previous, rollback)           |

`linux/arm64` only — there is no x86_64 variant and the kernels are not useful
on non-GB10 hardware.

## Why a separate image

`dgx-spark-sglang` builds sgl-kernel from source and that compile picks up
PyTorch + CUDA headers from `BASE_IMAGE`. Pinning the base image avoids:

- re-compiling PyTorch on every sglang rebuild (a 2-3 h step on GB10)
- accidental ABI drift between sgl-kernel and the runtime PyTorch
- the upstream 2.10 / cu131 fallback's codegen regression

If you don't need to rebuild sgl-kernel yourself, you probably want the
ready-to-run [`xomoxcc/dgx-spark-sglang`](https://hub.docker.com/r/xomoxcc/dgx-spark-sglang)
images directly.

## Status / support

Built and exercised on a private 4-node DGX Spark cluster. Published in case
someone else has the same hardware and wants to skip the 3-5 h cold rebuild.
No commercial support; tags may be retagged or removed without notice. Open
an issue on the GitHub repo if something is broken.
