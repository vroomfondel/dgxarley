<!-- short: PyTorch 2.14 + CUDA 13.2.1 + NCCL 2.30 base image for DGX Spark / GB10 (SM121), arm64, from source. -->

# dgx-spark-pytorch-dev

Custom **PyTorch + CUDA + NCCL base image** for the
**NVIDIA DGX Spark / ASUS Ascent GX10 (GB10, SM121, arm64)**.

This is the base layer used by
[`xomoxcc/dgx-spark-sglang`](https://hub.docker.com/r/xomoxcc/dgx-spark-sglang).
It exists because the upstream
[`scitrera/dgx-spark-pytorch-dev:2.10.0-v2-cu131`](https://hub.docker.com/r/scitrera/dgx-spark-pytorch-dev)
fallback (PyTorch 2.10 / CUDA 13.1) is **~45 % slower end-to-end on GB10**
than a 2.12+ / CUDA 13.2.1 build, due to nvcc codegen + cuBLAS/cuDNN
regressions across that toolchain delta.

- **Source / build script**: [github.com/vroomfondel/dgxarley](https://github.com/vroomfondel/dgxarley)
  (see [`scripts/build_pytorch_base_image.sh`](https://github.com/vroomfondel/dgxarley/blob/main/scripts/build_pytorch_base_image.sh) and
  [`scripts/patches/pytorch-2.14.1-dev-v2.recipe`](https://github.com/vroomfondel/dgxarley/blob/main/scripts/patches/pytorch-2.14.1-dev-v2.recipe))
- **Hardware target**: NVIDIA GB10 / SM121 (DGX Spark, ASUS Ascent GX10) — arm64 only
- **License**: same upstream licenses as PyTorch / NCCL / CUDA components

## What's inside

- **PyTorch 2.14.1**: built from source for `sm_120` + `sm_121`. This is the
  version SGLang v0.5.23 pins upstream (`2.13.0` for v0.5.18 through v0.5.21).
  The lineage also carries the 2.12 bump's cuBLAS Blackwell 32-MiB workspaces
  ([PyTorch PR #175344](https://github.com/pytorch/pytorch/pull/175344)), a
  direct GB10 win. torch 2.14 builds via scikit-build-core and rejects
  `setup.py bdist_wheel` ([pytorch#152276](https://github.com/pytorch/pytorch/issues/152276)),
  so the wheel is built with `python -m build --wheel --no-isolation` (Dockerfile
  patch `dockerfile-base-torch-pep517-build.patch`)
- **torchvision 0.29.1**: lockstep with torch 2.14.1
- **torchaudio 2.11.0** — not bumped (pytorch/audio has tagged nothing past
  2.11.0, so it skipped the whole 2.12 cycle too); the ABI lag is harmless for
  SGLang text-only inference, where the audio module is never imported
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
- **nccl-tests** pinned to `v2.21.1` and built with `-DOMPI_SKIP_MPICXX` (its
  device_api/gin tests otherwise fail to link against the OpenMPI C++ bindings;
  patch `dockerfile-base-nccl-tests-skip-mpicxx.patch`)
- **Slim variant (`2.14.1-v2-cu132`)**: stage `pytorch_slim` is the
  `pytorch_builder` minus the torch/vision/audio source + build tree (`/build`,
  ~8.7 GB), the CUDA `*_static.a` files except `libcudart_static` /
  `libcudadevrt` / `libculibos`, cuda `compat` / `compat_orin` (the forward-compat
  libcuda 595 is not in the ld cache, the host driver is used), nsight-compute,
  compute-sanitizer, cuda-gdb and `/usr/share/{doc,man,info}`. nvcc, the CUDA
  headers and shared libs, gcc/g++, ninja and cmake stay for JIT. It is squashed
  once into **one layer of 6.3 GB** (vs 23.4 GB for `v1`), keeps all ENV of
  `pytorch_builder` and uses `WORKDIR /data`
- Built via the upstream `scitrera/cuda-containers` `pytorch_builder` stage
  (`Dockerfile.base`) with our recipe pinned in the build script (new recipe
  knobs in `build_pytorch_base_image.sh`: `NCCL_TESTS_REF`, `DOCKERFILE_PATCHES`,
  `SQUASH_ALL`)

## Tags

| Tag               | Notes                                                                                                                                                                                           |
|-------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `2.14.1-v2-cu132` | PyTorch 2.14.1 + CUDA 13.2.1 + torchvision 0.29.1 + torchaudio 2.11.0 + NCCL 2.30.7, arm64, **slim** (single 6.3 GB layer vs 23.4 GB for v1; base of `dgx-spark-sglang:0.5.23-sm121`) (current) |
| `2.14.1-v1-cu132` | PyTorch 2.14.1 + CUDA 13.2.1 + torchvision 0.29.1 + torchaudio 2.11.0 + NCCL 2.30.7, arm64, full `pytorch_builder` incl. `/build` (23.4 GB)                                                     |
| `2.13.0-v1-cu132` | PyTorch 2.13.0 + CUDA 13.2.1 + torchvision 0.28.0 + NCCL 2.30.7, arm64 (previous)                                                                                                               |
| `2.12.0-v1-cu132` | PyTorch 2.12.0 + CUDA 13.2.1 + torchvision 0.27.0 + NCCL 2.30.7, arm64 (older)                                                                                                                  |
| `2.11.0-v1-cu132` | PyTorch 2.11.0 + CUDA 13.2.0 + NCCL 2.30.4, arm64 (older, rollback)                                                                                                                             |

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
