# syntax=docker/dockerfile:1.7
#
# xomoxcc/comfyui:sm121 — ComfyUI pre-baked for DGX Spark (GB10 / SM_121 / ARM64).
#
# Default BASE is scitrera's published pytorch-dev image (ARM64, CUDA, torch
# with Blackwell support). Override with --build-arg BASE=<image> to swap in
# the xomoxcc 2.11/cu132 base for the ~45% perf uplift documented in
# reference_sm121_build_base_regression (if available on the build host).
#
# What this image contains
#   - Frozen ComfyUI checkout at /opt/comfyui (pinned via COMFYUI_REF)
#   - ComfyUI's own pip requirements + our extras (see requirements-extra.txt)
#   - xformers built from source for SM_121
#   - SageAttention v2 built from source for SM_121
#
# What this image deliberately does NOT contain
#   - Model weights (multi-GB, churn fast, belong on hostPath / PVC / S3)
#   - ComfyUI-Manager or any custom nodes (belong under /workspace/custom_nodes
#     so they persist across image upgrades without rebuilds)

# Default base is the NGC PyTorch image (verified-correct SDPA on sm121).
# scripts/build_comfyui_image.sh overrides this via --build-arg BASE=...; the
# default here only matters for direct `podman build` calls without our
# wrapper. 26.03-py3 is the latest available NGC tag (torch 2.11 / cu13.2).
# Avoid scitrera/dgx-spark-pytorch-dev:2.10.0-v2-cu131 and the rebuilt
# xomoxcc/dgx-spark-pytorch-dev:2.11.0-v1-cu132 — both reproduce the
# silent-corruption SDPA bug documented in UPSTREAM_PYTORCH_SDPA_SM121.md.
ARG BASE=nvcr.io/nvidia/pytorch:26.03-py3
FROM ${BASE}

# ---- System deps -------------------------------------------------
# tini: minimal init/PID-1. Reaps zombies and forwards SIGTERM/SIGINT to the
# child process group. Python ignores SIGTERM by default when running as PID 1
# (no default handler installed there), so without tini the pod hangs for the
# full terminationGracePeriodSeconds on rollout/delete and gets SIGKILLed.
# Invoked from the Deployment via `command: ["/usr/bin/tini", "-g", "--", ...]`.
RUN apt-get update && apt-get install -y --no-install-recommends \
        tini \
        git git-lfs ffmpeg libgl1 libglib2.0-0 \
        build-essential ninja-build cmake pkg-config \
        ca-certificates curl && \
    rm -rf /var/lib/apt/lists/*

# ---- Env --------------------------------------------------------
# TORCH_CUDA_ARCH_LIST=12.1 keeps the build narrow (SM_121 only) so CUTLASS
# template expansion stays inside the GB10 memory budget. Add 9.0a;12.0 if
# the resulting image must also run on Hopper or early Blackwell silicon.
#
# MAX_JOBS=8 is the empirically safe ceiling on GB10 (128 GB unified);
# 16 OOM-kills CUTLASS template instantiations (see feedback_build_jobs_gb10).
# Declared as ARG so build_comfyui_image.sh can override via --build-arg
# (e.g. BUILD_COMFYUI_BUILD_JOBS=4 for memory-pressure scenarios).
ARG MAX_JOBS=8
ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    TORCH_CUDA_ARCH_LIST="12.1" \
    CMAKE_CUDA_ARCHITECTURES=121 \
    MAX_JOBS=${MAX_JOBS} \
    NVCC_THREADS=2 \
    HF_HOME=/workspace/.cache/huggingface \
    COMFYUI_PATH=/opt/comfyui

# ---- ComfyUI checkout -------------------------------------------
ARG COMFYUI_REF=master
RUN git clone https://github.com/comfyanonymous/ComfyUI.git ${COMFYUI_PATH} && \
    cd ${COMFYUI_PATH} && git checkout ${COMFYUI_REF} && \
    git rev-parse HEAD > ${COMFYUI_PATH}/.commit

# ---- Python deps (ComfyUI + extras) -----------------------------
# --upgrade-strategy only-if-needed prevents torch/torchvision/triton from
# the base image being replaced. The base ships Blackwell-capable wheels;
# replacing them silently regresses perf.
#
# We additionally filter `torch`, `torchaudio`, and `torchvision` (without
# the `torch_*` separators — `torchsde`/`torchao`/`torch_tensorrt` etc. are
# unrelated and stay) out of the requirements before pip touches them. Why:
# NGC's PyTorch wheels carry version strings like `2.11.0a0+nv26.3.46836102`
# which pip classifies as PRE-RELEASE. A bare requirement like `torchaudio`
# (no version specifier) is treated by pip as "any stable release", and a
# pre-release does not count as satisfying it — so pip silently *replaces*
# NGC's torchaudio with stock PyPI's `torchaudio==2.11.0`. The stock wheel's
# libtorchaudio.so was built against stock libtorch and uses different
# dtype-symbol mangling than NGC's, leading to a runtime
#   undefined symbol: torch_dtype_float4_e2m1fn_x2
# crash on `import torchaudio`. ComfyUI imports torchaudio unconditionally
# (audio-VAE module), so the whole pod fails to start. Same risk applies in
# principle to torch and torchvision, but pip's resolver tends to leave
# those alone because they're heavy dependencies of many other packages.
# The strict cure is to strip all three from the requirements file — NGC's
# already-installed wheels stay untouched.
RUN grep -vE '^(torch|torchaudio|torchvision)\b' ${COMFYUI_PATH}/requirements.txt \
        > /tmp/comfyui-requirements-filtered.txt && \
    pip install --upgrade-strategy only-if-needed \
        -r /tmp/comfyui-requirements-filtered.txt

COPY requirements-extra.txt /tmp/requirements-extra.txt
RUN grep -vE '^(torch|torchaudio|torchvision)\b' /tmp/requirements-extra.txt \
        > /tmp/requirements-extra-filtered.txt && \
    pip install --upgrade-strategy only-if-needed \
        -r /tmp/requirements-extra-filtered.txt

# ---- Triton ------------------------------------------------------
# SageAttention has a hard import-time dependency on triton. xformers also
# uses triton for several backend paths and prints
#   "A matching Triton is not available, some optimizations will not be enabled"
# without it. The scitrera 2.10.0-v2-cu131 base does NOT ship triton on
# aarch64, so we install it explicitly here, BEFORE the kernel builds.
#
# Strategy: read torch's own Requires-Dist metadata and install the exact
# triton package + version that torch declares. That way we don't pick a
# random PyPI version that mismatches torch's expected ABI. Falls back to
# PyPI 'triton' only when torch declares no triton dep at all (rare, only
# happens on bases stripped of the dep metadata).
#
# Pitfall encountered before: the package `pytorch-triton` on PyPI is a
# defensive squat stub that hard-crashes on import with
#   RuntimeError("Should never be installed")
# The real pytorch-triton wheels live on download.pytorch.org/whl/. By
# reading torch's metadata we automatically get the right name (whether
# that's `triton`, `pytorch-triton`, or a versioned variant), pinned to
# torch's ABI.
ARG INSTALL_TRITON=1
RUN --mount=type=cache,target=/root/.cache/pip \
    if [ "${INSTALL_TRITON}" = "1" ]; then \
        TRITON_REQ=$(python3 -c "import importlib.metadata as m; reqs = m.requires('torch') or []; tr = [r for r in reqs if 'triton' in r.split(';')[0].lower() and 'extra ==' not in r]; print('\n'.join(tr))") && \
        if [ -n "${TRITON_REQ}" ]; then \
            echo "torch declares triton requirement: ${TRITON_REQ}" && \
            echo "${TRITON_REQ}" | pip install --upgrade-strategy only-if-needed -r /dev/stdin ; \
        else \
            echo "torch declares no triton dep — falling back to PyPI 'triton'" && \
            pip install --upgrade-strategy only-if-needed triton ; \
        fi && \
        python3 -c "import triton; print('triton', triton.__version__)" ; \
    fi

# ---- torchaudio (from source against NGC torch) ------------------
# NGC PyTorch 26.03 (and 25.12 — both checked empirically) does NOT ship
# torchaudio in its aarch64 wheel set. ComfyUI imports torchaudio
# unconditionally via `comfy/ldm/lightricks/vae/audio_vae.py` (re-exported
# through `comfy/sd.py:15`), so the pod fails to start with
# `ModuleNotFoundError: No module named 'torchaudio'` without it.
#
# Stock-PyPI `torchaudio==2.11.0` does NOT solve this — it has an ABI
# mismatch against NGC's torch (different dtype-symbol mangling) and
# crashes at import-time with
#   undefined symbol: torch_dtype_float4_e2m1fn_x2
# from inside `libtorchaudio.abi3.so`. The fix is to build torchaudio
# from source against the in-place NGC torch, same pattern as the
# xformers / SageAttention builds below.
#
# Build-flag minimisation: ComfyUI only uses `torchaudio.load` and basic
# transforms like `torchaudio.transforms.MelScale`. The sox / ffmpeg /
# RNN-T / CTC-decoder backends aren't exercised, so we skip them — saves
# ~5 min of build time and trims image size.
#
# TORCHAUDIO_REF=v2.11.0 matches NGC's torch major.minor (2.11.0a0+nv26.x).
# If a future base bump regresses against this tag, fall back to master
# or to a minor-bumped tag and update TORCHAUDIO_REF here.
ARG BUILD_TORCHAUDIO=1
ARG TORCHAUDIO_REF=v2.11.0
RUN --mount=type=cache,target=/root/.cache/pip \
    if [ "${BUILD_TORCHAUDIO}" = "1" ]; then \
        git clone --depth 1 --branch "${TORCHAUDIO_REF}" --recurse-submodules \
            https://github.com/pytorch/audio.git /tmp/torchaudio && \
        cd /tmp/torchaudio && \
        USE_CUDA=1 \
        BUILD_SOX=0 \
        BUILD_RNNT=0 \
        BUILD_CTC_DECODER=0 \
        USE_FFMPEG=0 \
        TORCH_CUDA_ARCH_LIST="12.1" \
        pip install --no-build-isolation -v . && \
        cd / && \
        rm -rf /tmp/torchaudio && \
        python3 -c "import torchaudio; print('torchaudio', torchaudio.__version__, '— builds and loads OK')" ; \
    fi

# ---- Acceleration kernels (from source, SM_121) -----------------
# Both are optional; ComfyUI falls back to torch SDPA without them. With
# SageAttention especially, SDXL/FLUX throughput improves substantially.
ARG BUILD_XFORMERS=1
ARG BUILD_SAGE_ATTN=1

# xformers: pinned to a release tag (v0.0.32). setuptools-scm may still
# synthesize a 0.0.33+<sha>.dXXXXXX version on top of the pinned tag —
# that's expected and harmless.
#
# Why we clone + patch instead of `pip install git+...`:
#
# v0.0.32 has TWO structural incompatibilities with sm121 (Blackwell GB10)
# that cannot be fixed via build flags alone:
#
#   1) cutlass.FwOp / cutlass.BwOp route to PyTorch's compiled
#      `aten::_efficient_attention_forward`. PyTorch's kernel dispatcher
#      hard-rejects sm121 with
#        FATAL: kernel '..._sm80' is for sm80-sm100, but was built for sm121
#      We can't patch PyTorch's compiled binary, so we patch xformers'
#      cutlass.py to mark itself as "not supported" on devices with
#      compute capability >= 12.0. The xformers dispatcher then skips
#      cutlass and falls through to triton_splitKF (which JITs at runtime
#      and is sm121-clean).  See patches/xformers-disable-cutlass-on-sm121.patch
#
#   2) flash3 (FA3 / Hopper) is hardcoded to sm90+ SASS and has no sm121
#      binary; runtime call → "no kernel image is available". We disable
#      FA3 at BUILD time via XFORMERS_DISABLE_FLASH_ATTN=1 (setup.py:280
#      reads this and skips emitting FA3 nvcc archs entirely → FA3 is not
#      compiled into the wheel). The runtime patch in flash3.py is a
#      belt-and-braces guard so the dispatcher also hides FA3 even if a
#      pre-compiled wheel slipped through.
#      See patches/xformers-fa3-runtime-belt-and-braces.patch
#
# Arch list (8.0 + 12.1): keeps sm80-family CUTLASS kernel compiles in
# range (the dispatcher's family-tagging conventions expect sm80 binaries
# for `_sm80` kernels) plus blackwell-native sm121 for everything else.
# We deliberately omit 9.0/sm90: no Hopper hw on this cluster, sm90 just
# bloats the image and roughly doubles NVCC memory pressure during build
# (CUTLASS template instantiation; OOM-killed empirically at MAX_JOBS=8
# with 3 archs).
#
# Verify check explanation: do NOT do `from xformers import _C` —
# xformers' `_C.so` is a torch-ops library (loaded via
# `torch.ops.load_library`), not a regular CPython extension, so it has
# no `PyInit__C` symbol and direct Python import always raises
#   ImportError: dynamic module does not define module export function (PyInit__C)
# even on a perfectly healthy install. The meaningful test is whether
# the public API (ops.memory_efficient_attention) is exposed after
# import — that proves torch.ops.load_library succeeded and the kernels
# are bound.
ARG XFORMERS_REF=v0.0.32
COPY patches/xformers-disable-cutlass-on-sm121.patch /tmp/patches/xformers-disable-cutlass-on-sm121.patch
COPY patches/xformers-fa3-runtime-belt-and-braces.patch /tmp/patches/xformers-fa3-runtime-belt-and-braces.patch
RUN --mount=type=cache,target=/root/.cache/pip \
    if [ "${BUILD_XFORMERS}" = "1" ]; then \
        git clone --depth 1 --branch "${XFORMERS_REF}" --recurse-submodules \
            https://github.com/facebookresearch/xformers.git /tmp/xformers && \
        cd /tmp/xformers && \
        echo "Applying sm121 patches:" && \
        git apply --verbose /tmp/patches/xformers-disable-cutlass-on-sm121.patch && \
        git apply --verbose /tmp/patches/xformers-fa3-runtime-belt-and-braces.patch && \
        XFORMERS_DISABLE_FLASH_ATTN=1 \
        TORCH_CUDA_ARCH_LIST="8.0;12.1" \
        CMAKE_CUDA_ARCHITECTURES="80;121" \
        pip install --no-build-isolation -v . && \
        cd / && \
        rm -rf /tmp/xformers && \
        python3 -c "import xformers, xformers.ops as xo; \
                    assert hasattr(xo, 'memory_efficient_attention'), \
                        'xformers build is broken: ops.memory_efficient_attention missing'; \
                    from xformers.ops.fmha import cutlass, flash3; \
                    assert cutlass.FwOp.CUDA_MAXIMUM_COMPUTE_CAPABILITY == (12, 0), \
                        'cutlass sm121-disable patch did not apply'; \
                    assert flash3._C_flashattention3 is None, \
                        'FA3 was not disabled (XFORMERS_DISABLE_FLASH_ATTN guard failed)'; \
                    print('xformers', xformers.__version__, '— mea ok, cutlass<12.0 only, FA3 disabled')" ; \
    fi

# Same arch-list override rationale as for xformers above: SageAttention
# names its CUDA-side .so files per arch family (`_qattn_sm80.so`,
# `_qattn_sm89.so`, `_qattn_sm90.so`, ...). With the image-wide
# TORCH_CUDA_ARCH_LIST="12.1" each of those would be (mis-)compiled for
# sm121 only and the runtime would reject them on the family check just
# like xformers does. Build for the family-native archs so the right
# binary exists for each `_qattn_smXX` module. SageAttention currently
# ships sm80/sm89/sm90 variants — we build the sm80 + sm89 families plus
# sm121 native, deliberately skipping sm90: no Hopper hw on this cluster,
# and the extra arch doubles NVCC memory pressure for no runtime benefit.
RUN --mount=type=cache,target=/root/.cache/pip \
    if [ "${BUILD_SAGE_ATTN}" = "1" ]; then \
        git clone https://github.com/thu-ml/SageAttention.git /tmp/sage && \
        cd /tmp/sage && \
        TORCH_CUDA_ARCH_LIST="8.0;8.9;12.1" \
        CMAKE_CUDA_ARCHITECTURES="80;89;121" \
        pip install --no-build-isolation -v . && \
        cd / && \
        rm -rf /tmp/sage && \
        python3 -c "import sageattention; print('sageattention ok')" ; \
    fi

# ---- Custom nodes (opt-in bake-in) -------------------------------
# Runtime-Install ist der Default-Weg: Custom-Nodes landen per
# `git clone` unter /workspace/custom_nodes/ (hostPath), damit
# Updates keinen Image-Rebuild erzwingen. Beispiel:
#
#   ssh root@spark4 '
#     cd /var/lib/k8s-data/comfyui/custom_nodes &&
#     git clone https://github.com/Acly/comfyui-inpaint-nodes.git
#   '
#   kubectl -n comfyui rollout restart deploy/comfyui
#
# Falls stattdessen hart ins Image gebacken werden soll (reproduzierbar,
# aber Update = Rebuild), folgenden Block entkommentieren:
#
# ARG INPAINT_NODES_REF=main
# RUN git clone https://github.com/Acly/comfyui-inpaint-nodes.git \
#         ${COMFYUI_PATH}/custom_nodes/comfyui-inpaint-nodes && \
#     cd ${COMFYUI_PATH}/custom_nodes/comfyui-inpaint-nodes && \
#     git checkout ${INPAINT_NODES_REF}

# ---- Entrypoint -------------------------------------------------
COPY entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh

# ---- Build stamp ------------------------------------------------
# BUILDTIME wird vom build_comfyui_image.sh als --build-arg auf
# UTC-ISO-8601 gesetzt (z. B. "2026-04-24T18:42:01Z"). Als ENV im
# Image verfügbar, damit der Entrypoint-Log + Image-Label-Inspektion
# sofort zeigen, welcher Build läuft — praktisch gegen das Phänomen
# "Pod läuft auf alter Image-Layer trotz neuem Tag".
ARG BUILDTIME=unknown
ENV BUILDTIME=${BUILDTIME}
LABEL org.opencontainers.image.created="${BUILDTIME}"

WORKDIR /workspace
EXPOSE 8188
ENTRYPOINT ["/usr/bin/tini", "-g", "--", "/usr/local/bin/entrypoint.sh"]
