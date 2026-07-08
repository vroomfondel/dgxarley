# ComfyUI on DGX Spark (GB10, SM121, ARM64) — Image Build Guide

This guide describes, step by step, how to build a working container image
that runs ComfyUI **GPU-accelerated** on a DGX Spark (NVIDIA GB10 Grace‑Blackwell,
compute capability **SM_121**, `aarch64/ARM64`) — including FP8/FP4 checkpoints
such as `flux1-schnell-fp8.safetensors`.

> **Source-of-truth note.** The canonical production build is in this
> repo at [`scripts/comfyui/Dockerfile`](scripts/comfyui/Dockerfile)
> driven by [`scripts/build_comfyui_image.sh`](scripts/build_comfyui_image.sh).
> The snippets below are illustrative explainers — when in doubt, the
> repo files are authoritative. The guide is kept aligned with the
> repo state but reflects design decisions you'll want to understand
> before editing the Dockerfile yourself.

---

## 0. Prerequisites

- Build host on `aarch64` (one of the Sparks, e.g. `spark4`) — ComfyUI
  wheels must be built natively on ARM64; QEMU cross-builds are 10–20×
  slower and often break on CUTLASS templates.
- Docker or Podman with `nvidia-container-toolkit`, default runtime `nvidia`.
- At least **80 GB free disk** for build cache + final image.
- NVIDIA driver ≥ **580.x** on the host (Blackwell/SM_121 support).
- Registry account (Docker Hub / GHCR), example here: `xomoxcc/comfyui`.

Quick check on the build host:

```bash
uname -m                               # -> aarch64
nvidia-smi --query-gpu=compute_cap,driver_version --format=csv
# expected: 12.1, 580.x or higher
```

---

## 1. Choosing a base image

For SM_121 we need **CUDA ≥ 13.0** and **PyTorch with Blackwell support**.

**Current default (verified 2026-04-26): `nvcr.io/nvidia/pytorch:26.03-py3`**
(torch 2.11.0a0+nv26.03 / cu13.2). NGC's PyTorch wheels build sm_121-correct
SDPA `EFFICIENT_ATTENTION` kernels (verified empirically — see
[`UPSTREAM_PYTORCH_SDPA_SM121.md`](UPSTREAM_PYTORCH_SDPA_SM121.md)).

| Base | torch / cu | sm121 SDPA | Use |
|---|---|---|---|
| `nvcr.io/nvidia/pytorch:26.03-py3` | 2.11.0a0+nv / cu13.2 | ✓ correct | **default** |
| `nvcr.io/nvidia/pytorch:26.02-py3` | 2.11.0a0+nv / cu13.1 | ✓ correct | alternative |
| `nvcr.io/nvidia/pytorch:25.12-py3` | 2.10.0a0+nv / cu13.1 | ✓ correct | last 2.10 NGC |
| `scitrera/dgx-spark-pytorch-dev:2.10.0-v2-cu131` | 2.10.0 / cu13.1 | **✗ broken** | DO NOT USE |
| `xomoxcc/dgx-spark-pytorch-dev:2.11.0-v1-cu132` | 2.11.0 / cu13.2 | **✗ broken** | DO NOT USE |

> **Avoid scitrera-pipeline images on sm121.** Both
> `scitrera/dgx-spark-pytorch-dev:2.10.0-v2-cu131` and our own rebuild
> at 2.11/cu132 (built from scitrera's recipe) ship a torch wheel
> whose SDPA EFFICIENT_ATTENTION backend silently returns numerically
> corrupt output on sm121 — no NaN, no exception, just garbage
> embeddings. Discovered via ComfyUI text-to-image workflows that
> rendered prompt-unrelated images. Root cause traced to
> `NVCC_GENCODE=-gencode=arch=compute_121,code=sm_121` in scitrera's
> `Dockerfile.base` (no family fallback, family-range mismatch in the
> CUTLASS dispatcher). Full forensics in
> [`UPSTREAM_PYTORCH_SDPA_SM121.md`](UPSTREAM_PYTORCH_SDPA_SM121.md);
> end-to-end discovery story in
> [`COMFYUI_PROMPT_FAIL.md`](COMFYUI_PROMPT_FAIL.md).

**Trade-off with NGC bases:** ~13 GB image (vs ~6–8 GB for scitrera).
Acceptable cost for guaranteed correctness; see Section 11 for leaner
options if image size is a hard constraint.

> **2026-06-11 note — CUTLASS 4.5.0 and future NGC bases:** CUTLASS 4.5.0 (2026-05-13) added working SM120/SM121 block-scaled MMA ("Block Scaled MMA for SM120 now works on Spark"). NGC PyTorch monthlies post-26.03 that pick up CUTLASS 4.5.x are therefore worth testing as base images once available. The statement that NGC remains the only verified-correct PyTorch base for sm121 is still accurate as of 2026-06-11 — scitrera recipes still carry the broken sm_121-only `NVCC_GENCODE` flag at tag v0.5.12.post1. CUTLASS **4.5.2** (2026-06-16) and **4.4.3** (2026-06-18) have since been released; 4.5.2 is now the latest 4.5.x patch.
>
> **2026-06-12 note — NGC table currency:** the table above lists `26.03-py3` as the default (verified 2026-04-26). As of 2026-06-12 this is likely 2–3 NGC monthly releases behind (26.04, 26.05, and possibly 26.06 may be out — NGC release cadence is not verifiable via GitHub API). Re-verify against the current NGC monthly before starting a new image build; post-26.03 monthlies that pick up CUTLASS 4.5.x (per the 2026-06-11 note above) are worth preferring.

---

## 2. Project layout

```
comfyui-sm121/
├── Dockerfile
├── requirements-extra.txt
├── entrypoint.sh
└── patches/
    └── (optional: sm121 patches for xformers/flash-attn)
```

---

## 3. `requirements-extra.txt`

Packages that are **not** in the base image but that ComfyUI really wants.
Install everything with `--upgrade-strategy only-if-needed` so the
`torch`/`torchvision` version preinstalled in the base is **not**
overwritten (otherwise Blackwell support is lost).

```txt
# Core
comfyui-frontend-package
comfyui-workflow-templates
comfyui-embedded-docs

# Samplers / schedulers
einops
torchsde
kornia>=0.7.1
spandrel
soundfile
av>=14.2.0
pydantic~=2.0
pydantic-settings~=2.0
alembic
SQLAlchemy

# Utility
huggingface_hub[cli]
transformers>=4.37.2
tokenizers>=0.13.3
sentencepiece
safetensors>=0.4.2
aiohttp>=3.11.8
yarl>=1.18.0
psutil
tqdm
Pillow
scipy
numpy>=1.25.0
```

> ComfyUI pins its own `requirements.txt` directly in the repo. We do **not**
> copy that file statically into the image; instead we pull it from the
> upstream checkout at build time so we don't chase a moving target.

---

## 4. `Dockerfile`

```dockerfile
# syntax=docker/dockerfile:1.7
ARG BASE=nvcr.io/nvidia/pytorch:26.03-py3
FROM ${BASE}

# ---- System deps -------------------------------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
        tini \
        git git-lfs ffmpeg libgl1 libglib2.0-0 \
        build-essential ninja-build cmake pkg-config \
        ca-certificates curl && \
    rm -rf /var/lib/apt/lists/*

# ---- Python env --------------------------------------------------
ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    TORCH_CUDA_ARCH_LIST="12.1"              \
    CMAKE_CUDA_ARCHITECTURES=121             \
    MAX_JOBS=8                               \
    NVCC_THREADS=2                           \
    HF_HOME=/workspace/.cache/huggingface    \
    COMFYUI_PATH=/opt/comfyui

# TORCH_CUDA_ARCH_LIST=12.1 compiles SM_121 kernels for the kernels we
# build ourselves (xformers, SageAttention, torchaudio). The base
# image's torch is consumed as-is — we DO NOT rebuild it.
# MAX_JOBS=8 is the empirically safe ceiling on GB10 (16 OOM-kills CUTLASS).

# ---- Clone ComfyUI (pinned commit for reproducibility) -----------
ARG COMFYUI_REF=master
RUN git clone https://github.com/comfyanonymous/ComfyUI.git ${COMFYUI_PATH} && \
    cd ${COMFYUI_PATH} && git checkout ${COMFYUI_REF} && \
    git rev-parse HEAD > ${COMFYUI_PATH}/.commit

# ---- ComfyUI's own requirements (WITHOUT overwriting torch) ------
# CRITICAL: filter torch / torchaudio / torchvision out of the requirements
# file before pip touches it. ComfyUI's requirements.txt lists them
# unpinned; pip then sees NGC's torch wheels (e.g. `2.11.0a0+nv26.03`)
# as pre-release and silently REPLACES them with stock PyPI wheels —
# which have ABI-incompatible mangling against NGC libtorch and crash at
# import time with `undefined symbol: torch_dtype_float4_e2m1fn_x2`.
# The `\b` word-boundary leaves siblings like torchsde / torchao alone.
RUN grep -vE '^(torch|torchaudio|torchvision)\b' ${COMFYUI_PATH}/requirements.txt \
        > /tmp/comfyui-requirements-filtered.txt && \
    pip install --upgrade-strategy only-if-needed \
        -r /tmp/comfyui-requirements-filtered.txt

# ---- Our extra packages -----------------------------------------
COPY requirements-extra.txt /tmp/requirements-extra.txt
RUN grep -vE '^(torch|torchaudio|torchvision)\b' /tmp/requirements-extra.txt \
        > /tmp/requirements-extra-filtered.txt && \
    pip install --upgrade-strategy only-if-needed \
        -r /tmp/requirements-extra-filtered.txt

# ---- torchaudio (from source against NGC torch) ------------------
# NGC PyTorch 25.12 / 26.02 / 26.03 do NOT ship torchaudio in their
# aarch64 wheel set. ComfyUI imports torchaudio unconditionally
# (audio_vae.py via comfy/sd.py:15), so the pod fails to start with
# ModuleNotFoundError without it. Stock-PyPI torchaudio has ABI mismatch
# (see comment above). Build from source against the in-place NGC torch.
ARG TORCHAUDIO_REF=v2.11.0
RUN --mount=type=cache,target=/root/.cache/pip \
    git clone --depth 1 --branch "${TORCHAUDIO_REF}" --recurse-submodules \
        https://github.com/pytorch/audio.git /tmp/torchaudio && \
    cd /tmp/torchaudio && \
    USE_CUDA=1 BUILD_SOX=0 BUILD_RNNT=0 BUILD_CTC_DECODER=0 USE_FFMPEG=0 \
    TORCH_CUDA_ARCH_LIST="12.1" \
    pip install --no-build-isolation -v . && \
    cd / && rm -rf /tmp/torchaudio && \
    python3 -c "import torchaudio; print('torchaudio', torchaudio.__version__)"

# ---- Acceleration kernels (from source, SM_121) -----------------
# Both optional — ComfyUI runs on torch SDPA without them. With them
# enabled, significantly more it/s on SDXL/FLUX.

# (a) xformers v0.0.32 + sm121 patches (cutlass disable, FA3 disable).
#     The patches are needed because xformers' v0.0.32 dispatcher would
#     route to PyTorch's compiled CUTLASS-FMHA dispatcher and crash on
#     sm121. See COMFYUI_SM121_PATCHES.md for the patch content and
#     scripts/comfyui/patches/ for the actual diffs.
#
#     NOTE (2026-06-12): this snippet pins v0.0.32, but per
#     COMFYUI_SM121_PATCHES.md the patches were validated against v0.0.35
#     (patch-compatible across v0.0.32–v0.0.35). Consider bumping
#     XFORMERS_REF to v0.0.35 (latest release as of 2026-06-12); see
#     COMFYUI_SM121_PATCHES.md Maintenance section before bumping past
#     v0.0.35 (upstream main restructured fmha into the mslk package).
RUN --mount=type=cache,target=/root/.cache/pip \
    git clone --depth 1 --branch v0.0.32 --recurse-submodules \
        https://github.com/facebookresearch/xformers.git /tmp/xformers && \
    cd /tmp/xformers && \
    git apply /tmp/patches/xformers-disable-cutlass-on-sm121.patch && \
    git apply /tmp/patches/xformers-fa3-runtime-belt-and-braces.patch && \
    XFORMERS_DISABLE_FLASH_ATTN=1 \
    TORCH_CUDA_ARCH_LIST="8.0;12.1" \
    pip install --no-build-isolation -v . && \
    cd / && rm -rf /tmp/xformers

# (b) Sage-Attention v2 — faster than xformers on Blackwell for most
#     ComfyUI workloads.
RUN --mount=type=cache,target=/root/.cache/pip \
    git clone https://github.com/thu-ml/SageAttention.git /tmp/sage && \
    cd /tmp/sage && \
    TORCH_CUDA_ARCH_LIST="8.0;8.9;12.1" \
    pip install --no-build-isolation -v . && \
    rm -rf /tmp/sage

# ---- Entrypoint --------------------------------------------------
COPY entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh

WORKDIR /workspace
EXPOSE 8188
ENTRYPOINT ["/usr/bin/tini", "-g", "--", "/usr/local/bin/entrypoint.sh"]
```

**About `TORCH_CUDA_ARCH_LIST` and the family vs specific gencode:**
The image-wide `TORCH_CUDA_ARCH_LIST="12.1"` only governs kernels we
build ourselves. The base image's torch wheel is consumed as-is.
For our own kernels we use `8.0;12.1` (xformers) or `8.0;8.9;12.1`
(SageAttention) — the family ranges matter because CUTLASS dispatcher
metadata is family-tagged. **Never use `12.1a` (architecture-specific)
without a family fallback** — see UPSTREAM_PYTORCH_SDPA_SM121.md for
why scitrera's identical mistake produces silently corrupt SDPA on sm121.

---

## 5. `entrypoint.sh`

The image is immutable; dynamic data (models, outputs, custom nodes) lives
in the mounted `/workspace`. The entrypoint prepares that on first start
and then launches ComfyUI from the **image path** (`/opt/comfyui`), but
pulls models from `/workspace/models`.

```bash
#!/usr/bin/env bash
set -euo pipefail

DATA=/workspace
mkdir -p "$DATA"/{models/checkpoints,models/vae,models/clip,models/loras,output,temp,custom_nodes,user}

# Redirect ComfyUI's models folders to /workspace/models
if [ ! -f /opt/comfyui/extra_model_paths.yaml ]; then
cat > /opt/comfyui/extra_model_paths.yaml <<EOF
comfyui:
    base_path: /workspace
    checkpoints: models/checkpoints
    vae: models/vae
    clip: models/clip
    loras: models/loras
    custom_nodes: custom_nodes
EOF
fi

# Optional: pull FLUX.1-schnell on first start (ungated, Apache-2.0)
FLUX="$DATA/models/checkpoints/flux1-schnell-fp8.safetensors"
if [ ! -f "$FLUX" ] && [ "${DOWNLOAD_FLUX:-1}" = "1" ]; then
    huggingface-cli download Comfy-Org/flux1-schnell \
        flux1-schnell-fp8.safetensors \
        --local-dir "$DATA/models/checkpoints"
fi

cd /opt/comfyui
exec python main.py \
    --listen 0.0.0.0 \
    --port "${COMFYUI_PORT:-8188}" \
    --output-directory "$DATA/output" \
    --temp-directory "$DATA/temp" \
    --user-directory "$DATA/user" \
    "${COMFYUI_EXTRA_ARGS:-}"
```

With `--use-sage-attention` or `--use-pytorch-cross-attention` you can tell
ComfyUI explicitly which attention backend to use. Via env var:
`COMFYUI_EXTRA_ARGS="--use-sage-attention --fp8_e4m3fn-text-enc"`.

---

## 6. Building the image

**Always build on a Spark, never on `k3smaster` (x86_64)** — otherwise
kernels get compiled for the wrong architecture or emulated through QEMU.

In this repo the build is wrapped:

```bash
# From the control host (x86 is fine — the wrapper drives a remote
# podman socket on spark4 over SSH):
bash scripts/build_comfyui_image.sh                 # default: NGC 26.03 base
bash scripts/build_comfyui_image.sh --base nvcr.io/nvidia/pytorch:26.02-py3
bash scripts/build_comfyui_image.sh --no-torchaudio # skip torchaudio source build
bash scripts/build_comfyui_image.sh --no-push       # don't push to Docker Hub
```

The wrapper handles: registering the podman SSH connection, cloning
ComfyUI into the build context, applying our xformers patches,
streaming the result back to the control host, optional Docker Hub
push, and parallel k3s containerd distribution to all sparks via a
throwaway local registry. See the script header for full options.

The initial build takes ~50–80 min (xformers + SageAttention + torchaudio
are CUDA compiles). With `--mount=type=cache` and an unchanged Dockerfile,
later builds finish in <15 min.

On OOM kills during kernel builds: try `MAX_JOBS=4` instead of `8`
(see `feedback_build_jobs_gb10` — `16` empirically kills CUTLASS, `8` is
the safe ceiling, but under memory pressure go lower still).

---

## 7. Local smoke test

**Before** pushing to the registry, test on the build host:

```bash
docker run --rm -it --gpus all \
    -p 8188:8188 \
    -v /var/lib/k8s-data/comfyui:/workspace \
    -e DOWNLOAD_FLUX=0 \
    xomoxcc/comfyui:sm121 \
    bash -c "python -c 'import torch; print(torch.cuda.is_available(), torch.cuda.get_device_capability())'"
# expected: True (12, 1)
```

Smoke test with the real UI:

```bash
docker run --rm -it --gpus all -p 8188:8188 \
    -v /var/lib/k8s-data/comfyui:/workspace \
    xomoxcc/comfyui:sm121
# -> browser: http://<spark4-ip>:8188
```

In the UI:
1. *Load Default* workflow → simple SD1.5 test prompt (FLUX needs the
   previously pulled checkpoint plus a text encoder).
2. `Queue Prompt` → the pod log should **not** show the line
   `no kernel image is available for execution on the device`.
3. `nvidia-smi dmon -s u` on the host during generation → GPU util > 0.

---

## 8. Pushing to the registry

```bash
docker login docker.io -u xomoxcc
docker push xomoxcc/comfyui:sm121
docker push xomoxcc/comfyui:sm121-$(date +%Y%m%d)
```

---

## 9. Switching the Ansible role over

Change `roles/k8s_dgx/defaults/main/`:

```yaml
comfyui_image: "xomoxcc/comfyui:sm121"
```

The launch script in `roles/k8s_dgx/tasks/comfyui.yml` can be slimmed
down drastically: the entire pip install block and the ComfyUI git clone
go away, because both are already baked into the image. Effectively only
the model download is left — and even that is already handled by the new
`entrypoint.sh`. Alternative: remove the ConfigMap launch script
entirely and drop the Deployment's `command` line (the image's ENTRYPOINT
is enough).

Rollout:

```bash
ansible-playbook k8s_dgx.yml --tags comfyui -e comfyui_enabled=true
```

> **Never deploy without explicit approval** — this guide describes the
> procedure; do not run the command above until you have reviewed the
> changes.

---

## 10. Troubleshooting

| Symptom | Cause / fix |
|---|---|
| `RuntimeError: CUDA error: no kernel image is available` | Arch list missing 12.1 → rebuild image with `TORCH_CUDA_ARCH_LIST="9.0a;12.0;12.1"` |
| Build kills itself during SageAttention / torchaudio compile | RAM exhausted → `MAX_JOBS=4` or `2`, keep the host otherwise idle |
| ComfyUI starts, but generation is **slower** than the NGC image | Base-image skew — check torch/cu version in the new image (`python -c "import torch; print(torch.__version__, torch.version.cuda)"`) and switch to a base with torch ≥ 2.11/cu13.2 if needed |
| **Image renders but prompts are ignored / wrong content** (red apple → loft scene, "blue cube" → tiki statue, Flux → handwriting cards) | scitrera-pipeline base image — torch wheel has broken SDPA EFFICIENT_ATTENTION on sm121. **Switch to NGC base.** See [`COMFYUI_PROMPT_FAIL.md`](COMFYUI_PROMPT_FAIL.md) and [`UPSTREAM_PYTORCH_SDPA_SM121.md`](UPSTREAM_PYTORCH_SDPA_SM121.md). |
| `OSError: undefined symbol: torch_dtype_float4_e2m1fn_x2` from torchaudio at pod start | Stock-PyPI torchaudio over NGC torch (ABI mismatch). Filter `torchaudio` out of pip requirements before install (Section 4 above) and build it from source against in-place NGC torch. |
| `ModuleNotFoundError: No module named 'torchaudio'` at pod start | torchaudio source build skipped (`--no-torchaudio`) but ComfyUI imports it unconditionally via `comfy/sd.py`. Re-enable the torchaudio build step. |
| `!` tokens / NaN / black images on FLUX-fp8 | fp8 attention without kernel support → set `--use-sage-attention` or pull the fp16 variant instead |
| Pod stays `ContainerCreating` after an image push | `imagePullPolicy: IfNotPresent` + new tag → recreate the pod, or switch to `Always` + a versioned tag |
| `nvidia-smi` inside the pod shows no GPU | Time-slicing share gone? → `kubectl describe pod` → check the `nvidia.com/gpu` request. The cluster has 4 replicas/GPU via time-slicing (SGLang + ComfyUI share that) |

---

## 11. Variants / outlook

- **Leaner base:** once a sm121-correct PyTorch wheel ships outside NGC,
  a `nvidia/cuda:13.2.0-cudnn-devel-ubuntu24.04` + manual-torch base
  could drop the image from ~13 GB to ~6 GB. Until then NGC remains the
  only verified-correct option for sm121 SDPA — see
  [`UPSTREAM_PYTORCH_SDPA_SM121.md`](UPSTREAM_PYTORCH_SDPA_SM121.md) for
  the cross-validation matrix.
- **Models baked into the image instead of a hostPath:** not recommended —
  FLUX alone is 17 GB, and swapping models forces image rebuilds.
- **Preinstall ComfyUI-Manager:** clone it as a custom node into
  `/opt/comfyui/custom_nodes/ComfyUI-Manager` in the Dockerfile; it
  installs its own deps on first UI start.
- **Multi-GPU / pipeline parallelism:** ComfyUI is single-GPU. For batch
  generation, schedule multiple ComfyUI pods on different Sparks (one
  time-slice share each), don't try to spread a single instance across
  multiple GPUs.
