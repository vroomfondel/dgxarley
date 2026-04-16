#!/usr/bin/env bash
#
# build_sm121_image.sh — Build xomoxcc/dgx-spark-sglang:0.5.10-sm121.
#
# Produces a custom SGLang image where cutlass_moe_fp4 (the triton/cutlass
# MoE runner codepath for NVFP4) is patched to fit SM121's 101 KB shared
# memory budget. The upstream scitrera/dgx-spark-sglang:0.5.10 image crashes
# any NVFP4 MoE config that uses moe_runner_backend=triton/cutlass with a
# device-side assert at nvfp4_blockwise_moe.cuh:78. This image fixes that
# by patching run_fp4_blockwise_scaled_group_mm_sm120() to use StageCount<1>
# + KernelPtrArrayTmaWarpSpecialized (non-pingpong).
#
# Full root cause + fix rationale: CUTLASS_NVFP4_SM121_PRD.md
# Reference implementation: TensorRT-LLM PR #12141
# CUTLASS root cause: NVIDIA/cutlass#3144
#
# Workflow (all steps run on the x86 control host)
# -------------------------------------------------
# 1. Preflight: verify patch files + podman + git + patch are available.
# 2. Ensure a registered podman connection to the arm64 build host (spark4)
#    that uses a dedicated unencrypted SSH key (Podman's Go SSH client cannot
#    use ssh-agent or encrypted keys). Create it on demand if missing.
# 3. Clone or update scitrera/cuda-containers locally on x86. Switch to a
#    local 'sm121' branch, hard-reset to origin/main (idempotent), drop in
#    the sgl-kernel patch + Dockerfile patch + recipe file.
# 4. Invoke `podman --connection <name> build` — the build context is
#    streamed from x86 to spark4 over the podman socket, the actual build
#    runs natively on arm64 (no QEMU), and the resulting image is stored
#    in spark4's local podman image store. The x86 host never writes
#    credentials to spark4.
# 5. `podman image scp` to pull the built image from spark4 back to x86.
# 6. `podman push` from x86 using the x86 host's pre-existing registry
#    credentials. spark4 never has Docker Hub credentials.
#
# Prerequisites on the x86 control host
# --------------------------------------
# - podman (`apt install podman`)
# - An unencrypted SSH key for podman: generate with
#     ssh-keygen -t ed25519 -f ~/.ssh/id_podman -N ""
#     ssh-copy-id -i ~/.ssh/id_podman root@spark4
#   The key MUST be unencrypted — podman's Go SSH client does not support
#   ssh-agent or encrypted keys. Override via BUILD_SM121_SSH_IDENTITY.
# - `podman login docker.io -u xomoxcc` already done on the x86 host.
# - ~10 GB free disk for the image after scp.
# - git, patch (cuda-containers clone + patch apply happens on x86).
#
# Prerequisites on spark4 (the build host)
# ----------------------------------------
# - podman (`apt install podman`)
# - podman.socket enabled as root:
#     systemctl enable --now podman.socket
#   This exposes /run/podman/podman.sock which the x86 client connects to.
# - ~50 GB free disk (sgl-kernel layers + final image in local image store).
# - NO credentials, NO clone, NO patches, NO local scripts. spark4 is a
#   dumb remote build runner.
#

set -euo pipefail

# ============================================================================
# Configuration
# ============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PATCHES_DIR="${SCRIPT_DIR}/patches"

CUDA_CONTAINERS_REPO="https://github.com/scitrera/cuda-containers.git"
# The clone lives on the x86 control host. Overridable via env.
CUDA_CONTAINERS_DIR="${BUILD_SM121_CC_DIR:-${HOME}/pythondev_workspace/cuda-containers}"

# Local-only scratch branch in the cuda-containers clone; never exists
# upstream and is never pushed. Created from origin/main on first run,
# hard-reset to origin/main on every subsequent run (see prepare_cuda_containers)
# so that re-applying the patch stack is idempotent and drift in the working
# tree from previous runs is discarded.
BRANCH_NAME="sm121"
# RECIPE_NAME="sglang-0.5.10-sm121"
RECIPE_NAME="sglang-main-gemma4"
# IMAGE_TAG="xomoxcc/dgx-spark-sglang:0.5.10-sm121"
IMAGE_TAG="xomoxcc/dgx-spark-sglang:main-gemma4-sm121"

# Remote build host (spark4, arm64). Uses a registered podman connection
# with a dedicated unencrypted SSH key. The connection name is derived from
# this value by stripping the user@ prefix so that `podman system connection
# list` shows a clean "spark4" entry.
#
# Both can also be set by flags (--remote-host / --podman-connection); the
# derivation of the connection name from the host happens after argparse
# so that a late --remote-host override propagates.
REMOTE_HOST="${BUILD_SM121_REMOTE_HOST:-root@spark4.local}"
PODMAN_CONNECTION="${BUILD_SM121_PODMAN_CONNECTION:-}"
PODMAN_SSH_IDENTITY="${BUILD_SM121_SSH_IDENTITY:-${HOME}/.ssh/id_podman}"

# Build-time parallelism. scitrera's Dockerfile.sglang-nightly defaults to
# ARG BUILD_JOBS=2 which uses only 2 of the DGX Spark GB10's 20 ARM cores
# (10%). MAX_JOBS is set from this ARG and propagates to sgl-kernel,
# flashinfer, and any Python extension build that honors it.
#
# GB10 topology: 10 Cortex-X925 + 10 Cortex-A725, 128 GB unified memory
# (of which ~113 GiB is typically free on an otherwise-idle spark host).
#
# EMPIRICAL RESULT (2026-04-10): BUILD_JOBS=16 → OOM-killed during the
# CUTLASS template instantiation phase of sgl-kernel. CUTLASS TU peaks are
# heavier than the "~5 GB per TU" rule of thumb suggests — closer to
# 7-10 GB when multiple heavy TUs hit template expansion simultaneously.
# 16 × ~7 GB = ~112 GB → no margin for buffers/caches → OOM.
#
# 8 is the verified safe default on this hardware. If you want to push:
#   - 10 should still fit (10 × 7 = 70 GB worst case, ~40 GB headroom)
#   - 12 is the theoretical ceiling (12 × 7 = 84 GB, ~30 GB headroom)
#   - 16+ has been confirmed to OOM-kill, don't try again without first
#     reducing per-TU memory pressure (e.g. SGL_KERNEL_COMPILE_THREADS=1
#     in the CMake args — which the Dockerfile already sets for nvcc).
BUILD_JOBS="${BUILD_SM121_BUILD_JOBS:-8}"

PUSH_IMAGE=1

# When set to 1 (via --no-local-copy), skip the ~15 min save|load transfer
# of the built image from the remote build host back to this control host.
# Implies PUSH_IMAGE=0 because `run_push` reads from the local podman store
# which would be empty. Intended use: immediately follow the build with a
# `distrsm121image.sh --source <host>` call that distributes from the
# build host's podman store directly via a throwaway registry:2.
NO_LOCAL_COPY=0

# Base image selection. The recipe ships with a default BASE_IMAGE
# (currently our custom xomoxcc 2.11/cu132 build); --base lets you swap
# it at build time without editing the recipe. Supported aliases:
#
#   xomoxcc   xomoxcc/dgx-spark-pytorch-dev:2.11.0-v1-cu132
#             Our locally-built 2.11/cu132 base (scripts/build_pytorch_base_image.sh).
#             Only present on spark4's podman store — never published.
#             This is the recipe default and what you want for performance.
#
#   scitrera  scitrera/dgx-spark-pytorch-dev:2.10.0-v2-cu131
#             scitrera's published upstream base. Pulled from Docker Hub.
#             Produces a working build but ~45% slower end-to-end due to
#             the torch 2.10/cu131 vs 2.11/cu132 codegen regression (see
#             reference_sm121_build_base_regression memory). Use only for
#             fallback / A/B comparison, not production.
#
# Any other --base VALUE is passed through verbatim as the BASE_IMAGE.
# BUILD_SM121_BASE_IMAGE env var overrides --base for scripting.
BASE_XOMOXCC_IMAGE="xomoxcc/dgx-spark-pytorch-dev:2.11.0-v1-cu132"
BASE_SCITRERA_IMAGE="scitrera/dgx-spark-pytorch-dev:2.10.0-v2-cu131"
BASE_IMAGE_ALIAS=""
BASE_IMAGE_OVERRIDE="${BUILD_SM121_BASE_IMAGE:-}"
EFFECTIVE_BASE_IMAGE=""
BASE_IMAGE_SOURCE=""

# sgl-kernel source patch toggles. These map to build-args consumed by the
# patched Dockerfile (APPLY_SGL_KERNEL_ARCH_PRUNE / APPLY_SGL_KERNEL_DISABLE_FA3
# / APPLY_SGL_KERNEL_SKIP_SM90_TARGET) and are applied conditionally inside
# the builder stage BEFORE the sgl-kernel wheel build. The patch FILES
# themselves are always copied into the build context by apply_patches() —
# only the in-container `patch` invocation is gated, which keeps the build
# deterministic regardless of toggle state.
#
# All three default to ON. Disable switches:
#   --no-arch-prune         Keep the upstream arch list. Needed only if
#                           you plan to deploy the wheel on non-GB10
#                           Blackwell / Hopper hardware.
#   --keep-fa3              Keep FlashAttention-3 (Hopper-only) in the
#                           build. Default skips it since is_fa3_supported()
#                           returns False on GB10 anyway and __init__.py
#                           never imports flash_attn unconditionally. See
#                           patch header for the evidence chain.
#   --keep-sm90-target      Keep the common_ops_sm90_build target. Default
#                           skips it since load_utils._load_architecture_
#                           specific_ops() on GB10 always loads from
#                           sgl_kernel/sm100/, never sm90/.
APPLY_ARCH_PRUNE=1
APPLY_DISABLE_FA3=1
APPLY_SKIP_SM90_TARGET=1
APPLY_SKIP_FLASHMLA=1

# sm121-debug: opt-in diagnostic that adds a cudaStreamSynchronize +
# cudaGetLastError check immediately after gemm_op.run() in
# run_fp4_blockwise_scaled_group_mm_sm120(). Gated at runtime by the
# SGL_SM121_DEBUG_CUTLASS env var so it can be toggled without a rebuild,
# but the patch itself needs to be compiled into the image first (via
# --sm121-debug below). Default OFF.
APPLY_SM121_DEBUG=0

# ============================================================================
# Helpers
# ============================================================================

log()  { printf '\n\033[1;34m=== %s ===\033[0m\n' "$*"; }
warn() { printf '\033[1;33mWARN: %s\033[0m\n' "$*" >&2; }
die()  { printf '\033[1;31mERROR: %s\033[0m\n' "$*" >&2; exit 1; }

usage() {
    cat <<EOF
Usage: $(basename "$0") [--base xomoxcc|scitrera|<image>]
                        [--remote-host user@host] [--podman-connection NAME]
                        [--no-arch-prune] [--keep-fa3] [--keep-sm90-target]
                        [--keep-flashmla] [--sm121-debug]
                        [--no-local-copy] [--no-push] [--help]

Builds ${IMAGE_TAG} on the remote build host via podman socket, copies
the result back to this host (unless --no-local-copy), and pushes it
from here (unless --no-push or --no-local-copy).

Options:
  --base VALUE Select the PyTorch dev base image this build sits on:
                 xomoxcc   ${BASE_XOMOXCC_IMAGE}
                           (recipe default; locally-built 2.11/cu132; fast path)
                 scitrera  ${BASE_SCITRERA_IMAGE}
                           (upstream published; 2.10/cu131; ~45% slower)
                 <image>   arbitrary image reference, passed verbatim.
               Omitted → recipe default is used.
  --remote-host user@host
               Remote arm64 build host reachable via SSH + podman socket.
               Default: ${REMOTE_HOST}
  --podman-connection NAME
               Registered podman connection name to use (or create). If
               omitted, derived from --remote-host (strip user@ and domain).
  --no-arch-prune
               Skip sgl-kernel-arch-prune.patch. Default is to apply it.
               Disable only if you need a build that runs on non-GB10
               Blackwell / Hopper hardware.
  --keep-fa3   Skip sgl-kernel-disable-fa3.patch. Default IS to apply
               (i.e. FA3 is disabled by default since it is Hopper-only
               and cannot run on GB10). Use this to keep FA3 symbols in
               the build for testing on Hopper GPUs elsewhere.
  --keep-sm90-target
               Skip sgl-kernel-skip-sm90-target.patch. Default IS to apply
               (i.e. common_ops_sm90_build target is dead-coded since
               GB10 always loads sgl_kernel/sm100/common_ops.*). Use this
               if you plan to run the wheel on an actual Hopper GPU.
  --keep-flashmla
               Skip sgl-kernel-skip-flashmla.patch. Default IS to apply
               (i.e. the entire flashmla_ops target is dead-coded — its
               ~25 TUs target sm_90a / sm_100a and none run on GB10).
               Use this if you plan to run the wheel on Hopper/B200
               silicon for MLA-based inference (DeepSeek-V3, Kimi K2,
               etc.) — those models need the sm90 FlashMLA kernels.
  --sm121-debug
               Apply sgl-kernel-sm121-debug.patch on top of the primary
               sm121 patch. Default is NOT to apply. When applied, the
               JIT-compiled sm120 GEMM path gets a post-launch
               cudaStreamSynchronize + cudaGetLastError diagnostic block,
               gated at runtime by the SGL_SM121_DEBUG_CUTLASS env var.
               Set that env var on the sglang pod to turn the diagnostic
               on at runtime without any additional rebuild.
  --no-local-copy
               Skip the 'podman save | podman load' transfer of the built
               image from the remote build host back to this control host.
               Use when you will distribute the image directly via
               scripts/distrsm121image.sh, which runs a temporary registry
               on the build host and lets all K3s nodes pull from there —
               the local copy would be a pure ~15-minute time sink.
               Implies --no-push (you cannot push without a local copy;
               the build host has no Docker Hub credentials by design).
  --no-push    Skip 'podman push' after build + scp.
  --help       Show this help.

Environment overrides:
  BUILD_SM121_REMOTE_HOST        user@host for spark4 SSH.
                                 Default: ${REMOTE_HOST}
  BUILD_SM121_PODMAN_CONNECTION  Registered podman connection name.
                                 Default: derived from --remote-host
                                 (strip user@ and domain suffix).
  BUILD_SM121_SSH_IDENTITY       Unencrypted SSH private key for podman.
                                 Default: ${PODMAN_SSH_IDENTITY}
  BUILD_SM121_CC_DIR             Local cuda-containers clone path (on x86).
                                 Default: ${CUDA_CONTAINERS_DIR}
  BUILD_SM121_BUILD_JOBS         Parallel compile jobs on the build host
                                 (--build-arg BUILD_JOBS, sets MAX_JOBS env).
                                 Upstream Dockerfile default is 2 — useless
                                 on GB10's 20-core CPU. Push higher if more
                                 memory headroom is available.
                                 Default: ${BUILD_JOBS}
  BUILD_SM121_BASE_IMAGE         Direct BASE_IMAGE override. Wins over --base
                                 and over the recipe default. Use for
                                 scripting when --base aliases are too coarse.

The entire script runs on the x86 control host. spark4 is used purely as
a remote podman build runner — it holds no credentials, no clone, and no
state between runs (except the local podman image store and layer cache,
which persist and accelerate rebuilds).
EOF
}

# ============================================================================
# Argument parsing
# ============================================================================

while [[ $# -gt 0 ]]; do
    case "$1" in
        --no-push) PUSH_IMAGE=0; shift ;;
        --no-local-copy) NO_LOCAL_COPY=1; PUSH_IMAGE=0; shift ;;
        --base)
            shift
            [[ $# -gt 0 ]] || die "--base requires an argument (xomoxcc|scitrera|<image>)"
            BASE_IMAGE_ALIAS="$1"
            shift
            ;;
        --base=*)
            BASE_IMAGE_ALIAS="${1#--base=}"
            shift
            ;;
        --remote-host)
            shift
            [[ $# -gt 0 ]] || die "--remote-host requires an argument (user@host)"
            REMOTE_HOST="$1"
            shift
            ;;
        --remote-host=*)
            REMOTE_HOST="${1#--remote-host=}"
            shift
            ;;
        --podman-connection)
            shift
            [[ $# -gt 0 ]] || die "--podman-connection requires an argument"
            PODMAN_CONNECTION="$1"
            shift
            ;;
        --podman-connection=*)
            PODMAN_CONNECTION="${1#--podman-connection=}"
            shift
            ;;
        --no-arch-prune)
            APPLY_ARCH_PRUNE=0
            shift
            ;;
        --keep-fa3)
            APPLY_DISABLE_FA3=0
            shift
            ;;
        --keep-sm90-target)
            APPLY_SKIP_SM90_TARGET=0
            shift
            ;;
        --keep-flashmla)
            APPLY_SKIP_FLASHMLA=0
            shift
            ;;
        --sm121-debug)
            APPLY_SM121_DEBUG=1
            shift
            ;;
        --help|-h) usage; exit 0 ;;
        *)         die "Unknown argument: $1 (use --help)" ;;
    esac
done

# Derive the podman connection name from REMOTE_HOST if the user didn't
# set it explicitly. "root@spark4.local" -> "spark4".
if [[ -z "${PODMAN_CONNECTION}" ]]; then
    PODMAN_CONNECTION="${REMOTE_HOST##*@}"
    PODMAN_CONNECTION="${PODMAN_CONNECTION%%.*}"
fi

# ============================================================================
# Preflight
# ============================================================================

preflight() {
    log "Preflight"

    local missing=0
    for f in sgl-kernel-sm121.patch sgl-kernel-sm121-debug.patch \
             sgl-kernel-arch-prune.patch sgl-kernel-disable-fa3.patch \
             sgl-kernel-skip-sm90-target.patch sgl-kernel-skip-flashmla.patch \
             dockerfile-sm121.patch build-image-sh-podman.patch \
             "${RECIPE_NAME}.recipe"; do
        if [[ ! -f "${PATCHES_DIR}/${f}" ]]; then
            warn "Missing patch file: ${PATCHES_DIR}/${f}"
            missing=1
        fi
    done
    (( missing == 0 )) || die "Required patch files missing (see warnings above)"

    for tool in git patch podman; do
        command -v "${tool}" >/dev/null || die "Required tool not found: ${tool}"
    done

    if [[ ! -f "${PODMAN_SSH_IDENTITY}" ]]; then
        cat >&2 <<EOF

ERROR: SSH identity '${PODMAN_SSH_IDENTITY}' not found.

Podman's Go SSH client cannot use the ssh-agent or encrypted keys, so a
dedicated unencrypted key is required. Create it with:

  ssh-keygen -t ed25519 -f ${PODMAN_SSH_IDENTITY} -N ""
  ssh-copy-id -i ${PODMAN_SSH_IDENTITY} ${REMOTE_HOST}

Then re-run this script.
EOF
        exit 1
    fi

    echo "Patch files present, tools available, SSH identity found"
}

# ============================================================================
# Podman connection to the remote build host
# ============================================================================

ensure_podman_connection() {
    log "Ensuring podman connection '${PODMAN_CONNECTION}' → ${REMOTE_HOST}"

    if podman system connection list --format '{{.Name}}' | grep -qxF "${PODMAN_CONNECTION}"; then
        echo "Connection '${PODMAN_CONNECTION}' already registered"
    else
        echo "Registering new podman connection..."

        # Resolve the remote podman socket path. For root on spark4 this is
        # /run/podman/podman.sock; for rootless it would be
        # /run/user/<uid>/podman/podman.sock. We always SSH as root to the
        # cluster (per project convention), so root-socket is the default.
        local remote_uid
        remote_uid="$(ssh -i "${PODMAN_SSH_IDENTITY}" -o BatchMode=yes -o ConnectTimeout=5 \
            "${REMOTE_HOST}" id -u 2>/dev/null)" \
            || die "SSH to ${REMOTE_HOST} failed — verify the key is authorized"

        local sock_path
        if [[ "${remote_uid}" == "0" ]]; then
            sock_path="/run/podman/podman.sock"
        else
            sock_path="/run/user/${remote_uid}/podman/podman.sock"
        fi

        podman system connection add "${PODMAN_CONNECTION}" \
            "ssh://${REMOTE_HOST}${sock_path}" \
            --identity "${PODMAN_SSH_IDENTITY}" \
            || die "Failed to register podman connection '${PODMAN_CONNECTION}'"
    fi

    echo "Validating connection..."
    if ! podman --connection "${PODMAN_CONNECTION}" info >/dev/null 2>&1; then
        cat >&2 <<EOF

ERROR: Podman connection '${PODMAN_CONNECTION}' is not responding.

Check on ${REMOTE_HOST}:
  systemctl status podman.socket
  systemctl enable --now podman.socket

And that the socket path matches what this script registered:
  podman system connection list
EOF
        exit 1
    fi

    local remote_arch
    remote_arch="$(podman --connection "${PODMAN_CONNECTION}" info --format '{{.Host.Arch}}')"
    if [[ "${remote_arch}" != "arm64" && "${remote_arch}" != "aarch64" ]]; then
        die "Remote host is ${remote_arch}, expected arm64/aarch64 (scitrera base image is arm64)"
    fi
    echo "Remote podman is reachable (arch=${remote_arch})"
}

# ============================================================================
# Verify the pytorch dev base image is present in the remote podman store
# ============================================================================
#
# Our sglang-0.5.10-sm121.recipe references
#   BASE_IMAGE=xomoxcc/dgx-spark-pytorch-dev:2.11.0-v1-cu132
# which is NOT on Docker Hub — it's built locally via
# scripts/build_pytorch_base_image.sh and kept in spark4's podman store.
# If the sglang build runs before the base image exists, podman will try
# to pull from docker.io and fail with a 404 after a long retry cycle.
# Fail fast here instead with a clear diagnostic pointing at the fix.

resolve_base_image() {
    # Resolves the effective BASE_IMAGE. Order of precedence:
    #   1. BUILD_SM121_BASE_IMAGE env var (highest — scripting override)
    #   2. --base <value> CLI flag (xomoxcc/scitrera alias or verbatim image)
    #   3. Recipe default (lowest)
    # Fills the globals EFFECTIVE_BASE_IMAGE and BASE_IMAGE_SOURCE.
    if [[ -n "${EFFECTIVE_BASE_IMAGE}" ]]; then
        return 0   # already resolved
    fi

    if [[ -n "${BASE_IMAGE_OVERRIDE}" ]]; then
        EFFECTIVE_BASE_IMAGE="${BASE_IMAGE_OVERRIDE}"
        BASE_IMAGE_SOURCE="BUILD_SM121_BASE_IMAGE env"
        return 0
    fi

    case "${BASE_IMAGE_ALIAS}" in
        xomoxcc)
            EFFECTIVE_BASE_IMAGE="${BASE_XOMOXCC_IMAGE}"
            BASE_IMAGE_SOURCE="--base xomoxcc"
            return 0
            ;;
        scitrera)
            EFFECTIVE_BASE_IMAGE="${BASE_SCITRERA_IMAGE}"
            BASE_IMAGE_SOURCE="--base scitrera"
            return 0
            ;;
        "")
            ;;   # fall through to recipe default
        *)
            EFFECTIVE_BASE_IMAGE="${BASE_IMAGE_ALIAS}"
            BASE_IMAGE_SOURCE="--base (verbatim)"
            return 0
            ;;
    esac

    # Fall-through: no override, use recipe default.
    local recipe_file="${PATCHES_DIR}/${RECIPE_NAME}.recipe"
    if [[ -f "${recipe_file}" ]]; then
        EFFECTIVE_BASE_IMAGE="$(grep -E '^BASE_IMAGE=' "${recipe_file}" | head -1 | cut -d= -f2-)"
        BASE_IMAGE_SOURCE="recipe default"
    fi
}

ensure_base_image_present() {
    resolve_base_image
    local base_image="${EFFECTIVE_BASE_IMAGE}"
    [[ -n "${base_image}" ]] || return 0   # no base image declared, nothing to check

    log "Verifying base image '${base_image}' is present on '${PODMAN_CONNECTION}' (from ${BASE_IMAGE_SOURCE})"

    # Check both short-name and docker.io/ FQN forms. build_pytorch_base_image.sh
    # tags with both, but a hand-built image or earlier script version might
    # only have one. Either is acceptable for the downstream Dockerfile's
    # FROM resolution.
    if podman --connection "${PODMAN_CONNECTION}" image exists "docker.io/${base_image}" 2>/dev/null; then
        echo "Base image found on ${PODMAN_CONNECTION} as docker.io/${base_image}"
        return 0
    fi
    if podman --connection "${PODMAN_CONNECTION}" image exists "${base_image}" 2>/dev/null; then
        warn "Base image exists as short name only ('${base_image}'), not as docker.io/${base_image}."
        warn "The Dockerfile FROM step may normalize to docker.io/... and fail to find it."
        warn "Recommend retagging: podman --connection ${PODMAN_CONNECTION} tag ${base_image} docker.io/${base_image}"
        return 0
    fi

    # Not found locally — try pulling from Docker Hub before giving up.
    echo "Base image not found locally on ${PODMAN_CONNECTION}. Attempting pull from Docker Hub..."
    if podman --connection "${PODMAN_CONNECTION}" pull "docker.io/${base_image}" 2>&1; then
        echo "Base image pulled successfully from Docker Hub."
        return 0
    fi

    # Pull failed. Give a specific diagnostic for xomoxcc images (which may
    # not be published), generic hint for everything else.
    case "${base_image}" in
        xomoxcc/dgx-spark-pytorch-dev:*)
            cat >&2 <<EOF

ERROR: Base image '${base_image}' is not present on ${PODMAN_CONNECTION}
and could not be pulled from Docker Hub.

If the image has not been pushed to Docker Hub, build it first:
  bash ${SCRIPT_DIR}/build_pytorch_base_image.sh

That build takes approximately 3-5 hours (cold) or 30-60 min (with warm
ccache from a prior run). It produces the CUDA 13.2 + PyTorch 2.11 base
that this sglang build depends on.
EOF
            exit 1
            ;;
        *)
            die "Base image '${base_image}' not found locally and pull from Docker Hub failed."
            ;;
    esac
}

# ============================================================================
# cuda-containers clone + branch management (runs on x86)
# ============================================================================

prepare_cuda_containers() {
    log "Preparing cuda-containers at ${CUDA_CONTAINERS_DIR}"

    if [[ ! -d "${CUDA_CONTAINERS_DIR}/.git" ]]; then
        echo "Cloning scitrera/cuda-containers..."
        mkdir -p "$(dirname "${CUDA_CONTAINERS_DIR}")"
        git clone "${CUDA_CONTAINERS_REPO}" "${CUDA_CONTAINERS_DIR}"
    fi

    cd "${CUDA_CONTAINERS_DIR}"
    echo "Fetching upstream..."
    git fetch origin

    # Create or reset local sm121 branch from origin/main so re-runs are idempotent.
    if git show-ref --verify --quiet "refs/heads/${BRANCH_NAME}"; then
        git checkout "${BRANCH_NAME}"
        git reset --hard origin/main
        git clean -fd
    else
        git checkout -b "${BRANCH_NAME}" origin/main
    fi

    echo "Branch: $(git rev-parse --abbrev-ref HEAD)"
    echo "HEAD:   $(git rev-parse --short HEAD) ($(git log -1 --format=%s))"
}

# ============================================================================
# Apply patches into the cuda-containers clone (runs on x86)
# ============================================================================

apply_patches() {
    log "Applying SM121 patches"

    cd "${CUDA_CONTAINERS_DIR}"

    # 1. Drop all sgl-kernel source patches into the build context.
    # The Dockerfile COPY steps read from container-build/patches/ and the
    # in-container `patch` invocations are conditionally gated by the
    # APPLY_SGL_KERNEL_* build-args — we always copy the files so the
    # build context is deterministic regardless of toggle state.
    mkdir -p container-build/patches
    for p in sgl-kernel-sm121.patch sgl-kernel-sm121-debug.patch \
             sgl-kernel-arch-prune.patch sgl-kernel-disable-fa3.patch \
             sgl-kernel-skip-sm90-target.patch sgl-kernel-skip-flashmla.patch \
             sglang-gemma4-nvfp4-expert-loading.patch \
             sglang-gemma4-geglu-nan-clamp.patch; do
        if [[ -f "${PATCHES_DIR}/${p}" ]]; then
            install -m 0644 "${PATCHES_DIR}/${p}" "container-build/patches/${p}"
            echo "Installed container-build/patches/${p}"
        fi
    done

    # 2. Patch the Dockerfile (adds `patch` apt-get dependency + COPY/RUN step).
    #    Dry-run first so any upstream drift fails early with a clear diagnostic
    #    rather than half-applying.
    echo "Applying dockerfile-sm121.patch..."
    patch --dry-run -p1 < "${PATCHES_DIR}/dockerfile-sm121.patch" \
        || die "Dockerfile patch dry-run failed — upstream Dockerfile drifted; regenerate dockerfile-sm121.patch"
    patch -p1 < "${PATCHES_DIR}/dockerfile-sm121.patch"
    grep -q 'patches/sgl-kernel-sm121.patch' container-build/Dockerfile.sglang-nightly \
        || die "Dockerfile patch verification failed"
    echo "Dockerfile patched"

    # 2b. Gemma-4 NVFP4 Dockerfile patch (optional — only if the patch file exists).
    #     Adds COPY + apply steps for PR #22929 (per-expert weight loading) and
    #     PR #22928 (GEGLU activation + NaN clamp). Stacks on top of the sm121 patch.
    if [[ -f "${PATCHES_DIR}/dockerfile-gemma4-nvfp4.patch" ]]; then
        echo "Applying dockerfile-gemma4-nvfp4.patch..."
        patch --dry-run -p1 < "${PATCHES_DIR}/dockerfile-gemma4-nvfp4.patch" \
            || die "Gemma4 Dockerfile patch dry-run failed — regenerate dockerfile-gemma4-nvfp4.patch"
        patch -p1 < "${PATCHES_DIR}/dockerfile-gemma4-nvfp4.patch"
        grep -q 'sglang-gemma4-nvfp4-expert-loading.patch' container-build/Dockerfile.sglang-nightly \
            || die "Gemma4 Dockerfile patch verification failed"
        echo "Gemma4 Dockerfile patched"
    fi

    # 3. Drop in the recipe file. run_build() parses it inline and calls
    #    `podman build` directly, bypassing container-build/build-image.sh
    #    (which uses `docker buildx build` — podman has no buildx subcommand).
    install -m 0644 "${PATCHES_DIR}/${RECIPE_NAME}.recipe" \
        "container-recipes/${RECIPE_NAME}.recipe"
    echo "Installed container-recipes/${RECIPE_NAME}.recipe"

    # 4. Defense-in-depth: also patch build-image.sh to replace
    #    `docker buildx build` with `docker build`. Our run_build() bypasses
    #    build-image.sh entirely, so this is not strictly required — but the
    #    patched clone is then also usable for any other recipe via podman,
    #    and re-runs of this script hard-reset the branch so there's no
    #    accumulation cost.
    echo "Applying build-image-sh-podman.patch..."
    patch --dry-run -p1 < "${PATCHES_DIR}/build-image-sh-podman.patch" \
        || die "build-image.sh patch dry-run failed — upstream build-image.sh drifted; regenerate build-image-sh-podman.patch"
    patch -p1 < "${PATCHES_DIR}/build-image-sh-podman.patch"
    grep -q '^BUILD_CMD=(docker build)$' container-build/build-image.sh \
        || die "build-image.sh patch verification failed"
    echo "build-image.sh patched"
}

# ============================================================================
# Build via remote podman socket
# ============================================================================

run_build() {
    log "Running podman build on '${PODMAN_CONNECTION}' (60–90 minutes expected)"
    cd "${CUDA_CONTAINERS_DIR}"

    # Source the recipe in a subshell so we can read the build-time values
    # without polluting our own environment, then pass them as --build-arg.
    # The script's own IMAGE_TAG is verified against the recipe for consistency.
    local recipe_file="container-recipes/${RECIPE_NAME}.recipe"
    [[ -f "${recipe_file}" ]] || die "Recipe not found: ${recipe_file}"

    local R_DOCKERFILE R_TARGET R_BASE_IMAGE R_FLASHINFER_VERSION
    local R_TRANSFORMERS_VERSION R_SGLANG_VERSION R_SGLANG_REF R_IMAGE_TAG
    # shellcheck disable=SC1090
    source <(
        set -a
        # shellcheck disable=SC1090
        source "${recipe_file}"
        echo "R_DOCKERFILE='${DOCKERFILE}'"
        echo "R_TARGET='${TARGET}'"
        echo "R_BASE_IMAGE='${BASE_IMAGE}'"
        echo "R_FLASHINFER_VERSION='${FLASHINFER_VERSION}'"
        echo "R_TRANSFORMERS_VERSION='${TRANSFORMERS_VERSION}'"
        echo "R_SGLANG_VERSION='${SGLANG_VERSION}'"
        echo "R_SGLANG_REF='${SGLANG_REF}'"
        echo "R_IMAGE_TAG='${IMAGE_TAG}'"
    )

    if [[ "${R_IMAGE_TAG}" != "${IMAGE_TAG}" ]]; then
        die "Recipe IMAGE_TAG (${R_IMAGE_TAG}) does not match script IMAGE_TAG (${IMAGE_TAG})"
    fi

    # Resolve the effective BASE_IMAGE once (may already be populated from
    # ensure_base_image_present). If nothing overrode the recipe, fall back
    # to the recipe's BASE_IMAGE value so the --build-arg is always set.
    resolve_base_image
    local effective_base_image="${EFFECTIVE_BASE_IMAGE:-${R_BASE_IMAGE}}"
    local effective_base_source="${BASE_IMAGE_SOURCE:-recipe default}"

    echo "Recipe values:"
    echo "  DOCKERFILE           = ${R_DOCKERFILE}"
    echo "  TARGET               = ${R_TARGET}"
    echo "  BASE_IMAGE (recipe)  = ${R_BASE_IMAGE}"
    echo "  BASE_IMAGE (in use)  = ${effective_base_image}  [${effective_base_source}]"
    echo "  FLASHINFER_VERSION   = ${R_FLASHINFER_VERSION}"
    echo "  TRANSFORMERS_VERSION = ${R_TRANSFORMERS_VERSION}"
    echo "  SGLANG_VERSION       = ${R_SGLANG_VERSION}"
    echo "  SGLANG_REF           = ${R_SGLANG_REF}"
    echo "  IMAGE_TAG            = ${IMAGE_TAG}"
    echo "  BUILD_JOBS           = ${BUILD_JOBS} (overrides Dockerfile ARG default of 2)"
    echo "  sgl-kernel patches:"
    echo "    sm121 JIT kernel   = ALWAYS (late stage, cheap to re-apply)"
    echo "    sm121-debug        = $([ ${APPLY_SM121_DEBUG} -eq 1 ] && echo APPLY || echo skip)  (--sm121-debug opts in; runtime-gated by SGL_SM121_DEBUG_CUTLASS env)"
    echo "    arch-prune         = $([ ${APPLY_ARCH_PRUNE} -eq 1 ] && echo APPLY || echo skip)  (--no-arch-prune opts out)"
    echo "    disable-fa3        = $([ ${APPLY_DISABLE_FA3} -eq 1 ] && echo APPLY || echo skip)  (--keep-fa3 opts out)"
    echo "    skip-sm90-target   = $([ ${APPLY_SKIP_SM90_TARGET} -eq 1 ] && echo APPLY || echo skip)  (--keep-sm90-target opts out)"
    echo "    skip-flashmla      = $([ ${APPLY_SKIP_FLASHMLA} -eq 1 ] && echo APPLY || echo skip)  (--keep-flashmla opts out)"

    # The build context is container-build/ (contains Dockerfile + patches/
    # subdir). Podman streams it to the remote build host over the socket;
    # the build runs natively on arm64 and the result lands in the remote
    # host's local image store.
    podman --connection "${PODMAN_CONNECTION}" build \
        -f "container-build/${R_DOCKERFILE}" \
        --target "${R_TARGET}" \
        --build-arg "BASE_IMAGE=${effective_base_image}" \
        --build-arg "FLASHINFER_VERSION=${R_FLASHINFER_VERSION}" \
        --build-arg "TRANSFORMERS_VERSION=${R_TRANSFORMERS_VERSION}" \
        --build-arg "SGLANG_VERSION=${R_SGLANG_VERSION}" \
        --build-arg "SGLANG_REF=${R_SGLANG_REF}" \
        --build-arg "BUILD_JOBS=${BUILD_JOBS}" \
        --build-arg "APPLY_SGL_KERNEL_ARCH_PRUNE=${APPLY_ARCH_PRUNE}" \
        --build-arg "APPLY_SGL_KERNEL_DISABLE_FA3=${APPLY_DISABLE_FA3}" \
        --build-arg "APPLY_SGL_KERNEL_SKIP_SM90_TARGET=${APPLY_SKIP_SM90_TARGET}" \
        --build-arg "APPLY_SGL_KERNEL_SKIP_FLASHMLA=${APPLY_SKIP_FLASHMLA}" \
        --build-arg "APPLY_SGL_KERNEL_SM121_DEBUG=${APPLY_SM121_DEBUG}" \
        -t "${IMAGE_TAG}" \
        -t "docker.io/${IMAGE_TAG}" \
        container-build/

    if ! podman --connection "${PODMAN_CONNECTION}" image exists "docker.io/${IMAGE_TAG}"; then
        die "Build finished but docker.io/${IMAGE_TAG} not present in remote image store — check podman build output above"
    fi
    echo "Remote build complete: ${IMAGE_TAG} (also tagged as docker.io/${IMAGE_TAG})"
}

# ============================================================================
# Transfer built image from remote to local
# ============================================================================

transfer_image_from_remote() {
    log "Copying docker.io/${IMAGE_TAG} from ${PODMAN_CONNECTION} to local image store"

    # If an older local copy exists under any of the possible tags, remove
    # it first so the save→load pipeline doesn't silently keep stale layers
    # around. `localhost/` is included because `podman load` of a short-name
    # RepoTag normalizes the reference to `localhost/...` — that tag would
    # otherwise linger across runs and keep old image layers dangling.
    podman image rm "${IMAGE_TAG}" 2>/dev/null || true
    podman image rm "docker.io/${IMAGE_TAG}" 2>/dev/null || true
    podman image rm "localhost/${IMAGE_TAG}" 2>/dev/null || true

    # Stream save → load so pv can show progress. Use the docker.io/ FQN
    # so the loaded image lands under the same name the downstream tools
    # (containerd, ansible, k3s) expect.
    #
    # Image size comes from `podman image inspect --format '{{.Size}}'` and
    # is passed to pv as -s so the progress bar can compute percent + ETA.
    # The raw byte total is echoed first — pv's -ptebar format does not
    # print the absolute target itself, only the running count and percent,
    # so without this line the user has no way to tell up-front how big
    # the transfer will be.
    local size size_human
    size=$(podman --connection "${PODMAN_CONNECTION}" image inspect \
            --format '{{.Size}}' "docker.io/${IMAGE_TAG}" 2>/dev/null || echo "")
    if [[ -n "${size}" ]] && command -v numfmt >/dev/null 2>&1; then
        size_human=$(numfmt --to=iec --suffix=B "${size}")
        echo "Transfer target: ${size_human} (${size} bytes)"
    elif [[ -n "${size}" ]]; then
        echo "Transfer target: ${size} bytes"
    else
        warn "Could not determine image size on ${PODMAN_CONNECTION}; pv will run without ETA"
    fi

    if command -v pv >/dev/null 2>&1; then
        local pv_args=(-ptebar)
        [[ -n "${size}" ]] && pv_args+=(-s "${size}")
        set -o pipefail
        podman --connection "${PODMAN_CONNECTION}" image save "docker.io/${IMAGE_TAG}" \
            | pv "${pv_args[@]}" \
            | podman image load \
            || die "streamed image transfer failed"
    else
        set -o pipefail
        podman --connection "${PODMAN_CONNECTION}" image save "docker.io/${IMAGE_TAG}" \
            | podman image load \
            || die "streamed image transfer failed"
    fi

    # `podman image save docker.io/${IMAGE_TAG} | podman image load` does NOT
    # preserve the `docker.io/` prefix on the receiving side: podman's loader
    # strips the registry component and re-applies the short name, which then
    # gets normalized to `localhost/${IMAGE_TAG}`. Retag unconditionally so
    # downstream `podman push docker.io/${IMAGE_TAG}` finds the image. `podman
    # tag` atomically moves the target tag, so this is safe even if the tag
    # already points elsewhere from a prior run.
    if podman image exists "localhost/${IMAGE_TAG}"; then
        podman tag "localhost/${IMAGE_TAG}" "docker.io/${IMAGE_TAG}"
    elif podman image exists "${IMAGE_TAG}"; then
        podman tag "${IMAGE_TAG}" "docker.io/${IMAGE_TAG}"
    fi

    podman image inspect "docker.io/${IMAGE_TAG}" >/dev/null \
        || die "Image not present locally after transfer — check podman output"
    echo "Image transferred: docker.io/${IMAGE_TAG}"
}

# ============================================================================
# Push (from x86, using x86's credentials)
# ============================================================================

run_push() {
    if (( PUSH_IMAGE == 0 )); then
        log "Skipping push (--no-push)"
        return
    fi

    log "Pushing docker.io/${IMAGE_TAG} to Docker Hub from x86"

    # Podman looks for auth in $REGISTRY_AUTH_FILE or $XDG_RUNTIME_DIR/containers/auth.json
    # (rootless) or /run/containers/<uid>/auth.json. Do a soft check against
    # the most common location; if missing, let `podman push` produce the
    # definitive error.
    local auth_file="${REGISTRY_AUTH_FILE:-${XDG_RUNTIME_DIR:-/run}/containers/auth.json}"
    if [[ ! -f "${auth_file}" ]]; then
        # Legacy fallback: docker config, which podman also reads.
        auth_file="${HOME}/.docker/config.json"
    fi
    if [[ ! -f "${auth_file}" ]]; then
        warn "No registry auth file found (checked \$REGISTRY_AUTH_FILE and ~/.docker/config.json)"
        echo "Run 'podman login docker.io -u xomoxcc' on this host, then re-run."
        die "Registry authentication missing"
    fi

    podman push "docker.io/${IMAGE_TAG}"
    echo "Image pushed: docker.io/${IMAGE_TAG}"
}

# ============================================================================
# Next steps
# ============================================================================

print_next_steps() {
    if (( NO_LOCAL_COPY == 1 )); then
        cat <<EOF

$(log "Remote-only build complete")

Image: ${IMAGE_TAG}
Location: docker.io/${IMAGE_TAG} in ${PODMAN_CONNECTION}'s podman store only
          (NOT on this control host, NOT pushed to Docker Hub)

Next step — distribute to all 4 K3s nodes via the throwaway registry on
the build host (uses QSFP 200 GbE for the heavy transfers, source-fast-
path avoids the registry roundtrip for the build host itself):

  ./scripts/distrsm121image.sh --source ${REMOTE_HOST#*@} \\
      --registry-host <QSFP-IP of ${REMOTE_HOST#*@}>

Then redeploy:
  ansible-playbook k8s_dgx.yml --tags sglang

EOF
        return
    fi

    cat <<EOF

$(log "Build + push complete")

Image: ${IMAGE_TAG}

Next steps (on the x86 control host in ~/pythondev_workspace/dgxarley):

1. Bump sglang_image in roles/k8s_dgx/defaults/main.yml:
     sglang_image: "${IMAGE_TAG}"

2. Bump SGLANG_EXPECTED_IMAGE in roles/k8s_dgx/files/sglang_launch.sh:
     SGLANG_EXPECTED_IMAGE="${IMAGE_TAG}"

3. Pick a smoke-test profile override for Qwen3-235B-A22B-NVFP4
   (e.g. temporarily edit the entry in defaults/main.yml):
     moe_runner_backend: "triton"               # previously broken, now the target
     fp4_gemm_backend:   "flashinfer_cutlass"
     attention_backend:  "triton"
     disable_cuda_graph: true
     disable_piecewise_cuda_graph: true

4. Deploy:
     ansible-playbook k8s_dgx.yml --tags sglang

5. Watch logs:
     kubectl --context=ht@dgxarley -n sglang logs -f <head-pod>
   Look for:
     - NO 'nvfp4_blockwise_moe.cuh:78: CUDA error: device-side assert'
     - Normal model-load progression

6. Smoke-test inference:
     sglang-test --n 1 --model nvidia/Qwen3-235B-A22B-NVFP4

7. If n=1 works, reactivate Qwen3-235B matrix tests 1–6 and 25–30
   (previously all startup_crash / infer_error) and record in a new
   TESTLOG_nv580.142_sglang-0.5.10-sm121_*.md file.

If the triton MoE path does not beat the current flashinfer_cutlass baseline
(Qwen3-235B test 17: 42.70 tok/s @ n=8), roll back sglang_image to
scitrera/dgx-spark-sglang:0.5.10 — the custom image is then proof-of-concept
only and the upstream flashinfer_cutlass path remains the production default.

EOF
}

# ============================================================================
# Main
# ============================================================================

main() {
    preflight
    ensure_podman_connection
    ensure_base_image_present
    prepare_cuda_containers
    apply_patches
    run_build
    if (( NO_LOCAL_COPY == 0 )); then
        transfer_image_from_remote
        run_push
    else
        log "Skipping local copy + push (--no-local-copy) — image stays on ${PODMAN_CONNECTION}"
    fi
    print_next_steps
}

main "$@"
