# ============================================================================
# dgx-spark-quant-sm121.Dockerfile
#
# NVIDIA ModelOpt PTQ toolchain layered on top of the sm121 SGLang serving image,
# so a small model can be QUANTIZED (NVFP4 / modelopt_fp4) directly ON a DGX Spark
# (GB10, sm121, arm64) instead of on a rented Hopper box -- and then smoke-served
# from the same image on the same node (no scp of the checkpoint).
#
# Base = the full serving image ON PURPOSE (so quantize + smoke_sglang_spark.sh
# share one image). Quant itself needs only torch + CUDA + modelopt, NOT sglang,
# so if the dependency layering below ever conflicts, the clean fallback is to
# rebase onto xomoxcc/dgx-spark-pytorch-dev:2.12.0-v1-cu132 (same torch/cuda,
# no sglang) -- see NOTE at the pip step.
#
# CONSUMED BY (from the kikube quantizer dir, copied onto the Spark):
#   test_quant_dryrun.sh   configs/qwen3-30b-a3b.yaml   (Phase 0 gate)
#   quantize_modelopt_nvfp4.sh configs/qwen3-30b-a3b.yaml (Phase 1 real run)
#   smoke_sglang_spark.sh  configs/qwen3-30b-a3b.yaml   (already ran on this base)
# The scripts already run $PY hf_ptq.py directly and are env-overridable; on this
# image point VENV=/usr (system dist-packages, no venv) or just let them create a
# throwaway venv -- but the toolchain is baked in here so no runtime pip is needed.
#
# BUILD: arm64-only, natively on a Spark (no QEMU), same remote-podman flow as
# build_sm121_image.sh. From the x86 control host with a registered podman
# connection to the arm64 build host (e.g. spark4):
#   podman --connection spark4 build \
#     --build-arg BASE_IMAGE=xomoxcc/dgx-spark-sglang:0.5.15-sm121 \
#     -f scripts/patches/dgx-spark-quant-sm121.Dockerfile \
#     -t xomoxcc/dgx-spark-quant:0.5.15-sm121 .
#   podman image scp spark4::xomoxcc/dgx-spark-quant:0.5.15-sm121
#   podman push xomoxcc/dgx-spark-quant:0.5.15-sm121
# (Nothing here is built/pushed automatically -- this is a reviewable draft.)
# ============================================================================

ARG BASE_IMAGE=xomoxcc/dgx-spark-sglang:0.5.15-sm121
FROM ${BASE_IMAGE}

# WHAT THE BASE ALREADY SHIPS (verified on 0.5.14-sm121): nvidia-modelopt 0.45.0,
# torch 2.12.0/cu132, transformers 5.8.1, datasets 5.0.0, huggingface-hub 1.23.0 +
# the `hf` CLI, ninja, typer. So this image only has to add the two pieces the quant
# scripts need that are genuinely MISSING: `accelerate` (device_map) and
# `hf_transfer` (fast download). Everything else is reused as-is.
#
# WHY NOT reinstall the rest: the base has an internally-inconsistent pin
# (datasets 5.0.0 requires fsspec<=2026.4.0 but the image ships fsspec 2026.6.0). It
# works at runtime, but asking pip to (re)install `datasets` forces a fresh resolve
# that trips over it (ResolutionImpossible). So we install ONLY the missing leaves.

# --- 1. freeze the serving stack so the accelerate/hf_transfer install can only ADD,
#        never MOVE, anything already in the image (torch/flashinfer/sgl-kernel/etc.)
RUN python3 -m pip freeze --all > /tmp/serving-constraints.txt \
 && echo ">>> froze $(wc -l < /tmp/serving-constraints.txt) serving packages as constraints"

# --- 2. add ONLY the genuinely-missing leaves, pinned by the freeze ----------------
# accelerate has no fsspec/datasets dep, so it resolves cleanly against the freeze.
# py-spy is a self-contained Rust binary (no python deps) for sampling-profiler
# tracing of a running serving process, so it also resolves cleanly.
# flash-linear-attention (top-level module `fla`) is added for the NON-SGLANG
# encoding/quant work done directly in this image: SGLang bundles its OWN copy under
# sglang.srt.layers.attention.fla, which a plain `import fla` outside the serving
# path CANNOT reach — so any encoding script that uses fla here needs the real PyPI
# package. It is pure Python + Triton (no CUDA compile); under the freeze constraint
# it can only ADD leaves (einops, ...), never MOVE torch/triton/transformers/
# sgl-kernel. If a future base makes it ResolutionImpossible against the freeze,
# install it separately with `--no-deps` and add its pure-python leaves explicitly.
# NOTE (fallback): should a future base bump make even this conflict, rebase FROM
# xomoxcc/dgx-spark-pytorch-dev:2.12.0-v1-cu132 (no sglang stack to constrain) and
# run smoke serving from the serving image instead.
RUN python3 -m pip install --no-cache-dir -c /tmp/serving-constraints.txt \
      accelerate hf_transfer py-spy flash-linear-attention

# --- 3. assert the base modelopt is new enough for the (later) mixed-precision recipe
# path (>=0.45 = the release NVIDIA built Qwen3.5-397B-NVFP4-V2 with; first with the
# --recipe MIXED_PRECISION system). The pilot's uniform nvfp4 needs less, but catch a
# base downgrade at build, not mid-run.
RUN python3 -c "import modelopt,sys; from packaging.version import Version; v=Version(modelopt.__version__); print('modelopt',v); sys.exit(0 if v>=Version('0.45') else 1)"

# --- 4. drop deepspeed if present (import aborts without CUDA_HOME; irrelevant to PTQ)
RUN python3 -m pip uninstall -y deepspeed 2>/dev/null || true

# --- 5. build-time smoke: the whole quant stack must import together ----------------
# NOTE: single-line `python3 -c` on purpose -- the imagebuilder on the arm64 build
# host (podman 4.9.3) does NOT support `RUN <<'HEREDOC'` and parses its body lines as
# Dockerfile instructions. Keep any in-Dockerfile python as one-liners.
RUN python3 -c "import torch, transformers, datasets, accelerate, modelopt, fla; import modelopt.torch.quantization; print('>>> OK: torch', torch.__version__, '(cuda', torch.version.cuda, ') transformers', transformers.__version__, 'datasets', datasets.__version__, 'accelerate', accelerate.__version__, 'modelopt', modelopt.__version__, 'fla', getattr(fla, '__version__', '?'))"

# The scripts expect HF_XET_HIGH_PERFORMANCE for fast downloads; harmless if unused.
ENV HF_XET_HIGH_PERFORMANCE=1
