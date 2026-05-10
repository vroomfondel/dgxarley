<!-- short: PyTorch 2.11 + CUDA 13.2 + NCCL 2.29 base image for DGX Spark / GB10 (SM121), arm64, from source. -->

# dgx-spark-pytorch-dev

Custom **PyTorch + CUDA + NCCL base image** for the
**NVIDIA DGX Spark / ASUS Ascent GX10 (GB10, SM121, arm64)**.

This is the base layer used by
[`xomoxcc/dgx-spark-sglang`](https://hub.docker.com/r/xomoxcc/dgx-spark-sglang).
It exists because the upstream
[`scitrera/dgx-spark-pytorch-dev:2.10.0-v2-cu131`](https://hub.docker.com/r/scitrera/dgx-spark-pytorch-dev)
fallback (PyTorch 2.10 / CUDA 13.1) is **~45 % slower end-to-end on GB10**
than a 2.11 / CUDA 13.2 build, due to nvcc codegen + cuBLAS/cuDNN
regressions across that toolchain delta.

- **Source / build script**: [github.com/vroomfondel/dgxarley](https://github.com/vroomfondel/dgxarley)
  (see [`scripts/build_pytorch_base_image.sh`](https://github.com/vroomfondel/dgxarley/blob/main/scripts/build_pytorch_base_image.sh) and
  [`scripts/patches/pytorch-2.11.0-dev-v1.recipe`](https://github.com/vroomfondel/dgxarley/blob/main/scripts/patches/pytorch-2.11.0-dev-v1.recipe))
- **Hardware target**: NVIDIA GB10 / SM121 (DGX Spark, ASUS Ascent GX10) — arm64 only
- **License**: same upstream licenses as PyTorch / NCCL / CUDA components

## What's inside

- **PyTorch 2.11.0** — built from source for `sm_121`
- **torchvision 0.26**, **torchaudio 2.11** — built from source against the same PyTorch
- **NCCL 2.29.7** — built from the
  [`zyang-dev/nccl`](https://github.com/zyang-dev/nccl) `dgxspark-3node-ring`
  fork (resolved to `2.29.7-1`), required for the multi-host ring topology used
  by the cluster
- **CUDA 13.2** runtime + headers
- The full upstream `scitrera/cuda-containers` `pytorch_builder` stage with the
  recipe ref pinned in our build script — same recipe scitrera committed on
  2026-04-08 but did not publish to Docker Hub themselves

## Tags

| Tag | Notes |
|---|---|
| `2.11.0-v1-cu132` | PyTorch 2.11.0 + CUDA 13.2 + NCCL 2.29.7, arm64 |

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
