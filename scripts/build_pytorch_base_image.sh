#!/usr/bin/env bash
#
# build_pytorch_base_image.sh — Build xomoxcc/dgx-spark-pytorch-dev:2.11.0-v1-cu132.
#
# Produces the PyTorch 2.11.0 + CUDA 13.2 + NCCL 2.29.7 base image that our
# custom sgl-kernel sm121 sglang image builds on top of. This replaces the
# scitrera/dgx-spark-pytorch-dev:2.10.0-v2-cu131 fallback we've been using
# (see reference_sm121_build_base_regression memory for context on why the
# fallback was forced and what ~45% perf regression it cost us).
#
# scitrera committed a recipe for this in their repo on 2026-04-08 but has
# NOT published the resulting image to Docker Hub (their main-dir recipes
# are built manually by the maintainer, no CI). The recipe also references
# an experimental NCCL fork that may be why the manual build is pending.
# We build it ourselves from the same recipe with a resolved NCCL version
# and our own image tag.
#
# Workflow (runs on the x86 control host, same pattern as build_sm121_image.sh)
# ------------------------------------------------------------------------------
# 1. Preflight: verify the active pytorch-*-dev-v1.recipe is present, podman is
#    installed locally, and the dedicated SSH identity for podman is usable.
# 2. Ensure a registered podman connection to the arm64 build host (default
#    spark4 — override via --remote-host / BUILD_PYTORCH_REMOTE_HOST).
#    Reuses the same connection name as build_sm121_image.sh.
# 3. Clone or update scitrera/cuda-containers locally on x86 (shared clone
#    with build_sm121_image.sh). Switch to a local 'sm121' branch, hard-
#    reset to origin/main (idempotent), drop our custom recipe file in.
# 4. Invoke `podman --connection <name> build` — the build context is
#    streamed from x86 to the build host over the podman socket. The
#    pytorch_builder target stage compiles NCCL + PyTorch + torchvision +
#    torchaudio all from source on arm64 (versions per the active recipe).
# 5. Result is stored in the build host's local podman image store as
#    ${IMAGE_TAG}. By default it is ALSO scp'd back to the x86 control host
#    and pushed to Docker Hub (same flow as build_sm121_image.sh) — pass
#    --no-push to keep it local-only so the subsequent sgl-kernel sm121
#    build (build_sm121_image.sh) finds it by name on that same host without
#    a pull attempt.
# 6. build_sm121_image.sh's sgl-kernel recipe should reference
#    BASE_IMAGE=${IMAGE_TAG} to consume this image.
#
# Build time expectations
# -----------------------
# Cold build on a GB10 Spark node: approximately 3-5 hours. Breakdown:
#   - NCCL from source:                    ~5-10 min
#   - NCCL tests:                          ~5 min
#   - PyTorch 2.11 from source:            ~120-180 min  (the bulk)
#   - torchvision 0.26 from source:        ~15-25 min
#   - torchaudio 2.11 from source:         ~15-25 min
# With warm ccache from a previous run: expect 30-60 min for rebuilds of
# the same commits (most TUs hit cache).
#
# Memory footprint
# ----------------
# PyTorch from source is heavy. Each nvcc/cc1plus TU can peak at 8-12 GiB
# RSS during template expansion. BUILD_JOBS=8 with 8 parallel heavy TUs
# could touch ~80-100 GiB peak. The 121 GiB GB10 tolerates this but leaves
# little headroom. If OOM-kill happens, drop BUILD_JOBS to 6 or 4 via the
# env override.
#
# Prerequisites on the x86 control host
# --------------------------------------
# - podman installed (`apt install podman`)
# - Unencrypted SSH key at ~/.ssh/id_podman (same as build_sm121_image.sh;
#   override via BUILD_PYTORCH_SSH_IDENTITY). Podman's Go SSH client does
#   not support ssh-agent or encrypted keys.
# - git, rsync (for local cuda-containers clone management)
#
# Prerequisites on the remote build host (default spark4; --remote-host /
# BUILD_PYTORCH_REMOTE_HOST selects a different one)
# ------------------------------------------------------------------------
# - podman installed, podman.socket enabled as root
# - ~200 GB free disk (PyTorch from-source is bulky; intermediate layers
#   before the final squashed image easily hit 100+ GB)
# - No other heavy workload running (OOM risk under concurrent load)
#

set -euo pipefail

# ============================================================================
# Configuration
# ============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PATCHES_DIR="${SCRIPT_DIR}/patches"

CUDA_CONTAINERS_REPO="https://github.com/scitrera/cuda-containers.git"
# Shared clone with build_sm121_image.sh. Uses the same sm121 branch so
# both scripts can operate on the same working tree without conflict.
CUDA_CONTAINERS_DIR="${BUILD_PYTORCH_CC_DIR:-${HOME}/pythondev_workspace/cuda-containers}"

# Same scratch branch name as build_sm121_image.sh (intentionally shared).
BRANCH_NAME="sm121"
# Last verified vs scitrera/cuda-containers main on 2026-04-26: this is still
# the most recent dev recipe upstream (no 2.12 / cu13.3 / newer; only an
# `experimental/pytorch-2.11.0-runtime.recipe` exists alongside, which is the
# runtime-only variant of the same source line). When updating, re-run:
#   git -C ~/pythondev_workspace/cuda-containers fetch origin
#   ls ~/pythondev_workspace/cuda-containers/container-recipes/*pytorch*
#RECIPE_NAME="pytorch-2.11.0-dev-v1"
#IMAGE_TAG="xomoxcc/dgx-spark-pytorch-dev:2.11.0-v1-cu132"
RECIPE_NAME="pytorch-2.12.0-dev-v1"
IMAGE_TAG="xomoxcc/dgx-spark-pytorch-dev:2.12.0-v1-cu132"

# Remote build host. Defaults match build_sm121_image.sh so the same
# registered podman connection can be reused.
REMOTE_HOST="${BUILD_PYTORCH_REMOTE_HOST:-root@spark4.local}"
PODMAN_CONNECTION="${BUILD_PYTORCH_PODMAN_CONNECTION:-${REMOTE_HOST##*@}}"
# Shorten a DNS name to its first label (spark4.local -> spark4), but keep an
# IPv4 address whole ("192.168.0.5" must NOT collapse to "192").
if [[ ! "${PODMAN_CONNECTION}" =~ ^[0-9]+(\.[0-9]+){3}$ ]]; then
    PODMAN_CONNECTION="${PODMAN_CONNECTION%%.*}"
fi
PODMAN_SSH_IDENTITY="${BUILD_PYTORCH_SSH_IDENTITY:-${HOME}/.ssh/id_podman}"

# Build-time parallelism. Same considerations as build_sm121_image.sh but
# PyTorch from source is more memory-hungry than sgl-kernel. Start at 8
# and drop if OOM; 6 is a safer conservative choice if you know other
# workloads are running on the build host.
BUILD_JOBS="${BUILD_PYTORCH_BUILD_JOBS:-8}"

# Docker Hub push. ON by default to match build_sm121_image.sh's behavior —
# pass --no-push to keep the image local-only on the remote build host. Push
# uses the x86 host's pre-configured registry credentials after scp'ing the
# image back from the remote build host.
PUSH_IMAGE=1

# ============================================================================
# Helpers
# ============================================================================

log()  { printf '\n\033[1;34m=== %s ===\033[0m\n' "$*"; }
warn() { printf '\033[1;33mWARN: %s\033[0m\n' "$*" >&2; }
die()  { printf '\033[1;31mERROR: %s\033[0m\n' "$*" >&2; exit 1; }

usage() {
    cat <<EOF
Usage: $(basename "$0") [--no-push] [--remote-host user@host] [--help]

Builds ${IMAGE_TAG} on the remote build host (${REMOTE_HOST}) via remote podman socket.

This is a one-time-ish build that produces the base image consumed by
build_sm121_image.sh. By default the result is scp'd back to the x86
control host and pushed to Docker Hub (same flow as build_sm121_image.sh).
Expected duration: 3-5 hours cold build.

Options:
  --no-push              Skip the scp+push steps and keep the image only in
                         the remote build host's local podman store. Useful
                         for iteration on the recipe without consuming Docker
                         Hub bandwidth. Default: push.
  --remote-host HOST     user@host for the arm64 build host, overriding both
                         the default and BUILD_PYTORCH_REMOTE_HOST. Also
                         re-derives the podman connection name from HOST
                         unless BUILD_PYTORCH_PODMAN_CONNECTION is set.
                         Default: ${REMOTE_HOST}
  --help                 Show this help.

Environment overrides:
  BUILD_PYTORCH_REMOTE_HOST        user@host for the remote build host's SSH.
                                   Overridden by --remote-host if both are given.
                                   Default: ${REMOTE_HOST}
  BUILD_PYTORCH_PODMAN_CONNECTION  Registered podman connection name. Pins
                                   the connection regardless of --remote-host.
                                   Default: derived from REMOTE_HOST (${PODMAN_CONNECTION})
  BUILD_PYTORCH_SSH_IDENTITY       Unencrypted SSH private key for podman.
                                   Default: ${PODMAN_SSH_IDENTITY}
  BUILD_PYTORCH_CC_DIR             Local cuda-containers clone path (on x86).
                                   Default: ${CUDA_CONTAINERS_DIR}
  BUILD_PYTORCH_BUILD_JOBS         Parallel compile jobs. PyTorch is memory-
                                   hungry — reduce to 6 or 4 if OOM-killed.
                                   Default: ${BUILD_JOBS}

After a successful build, run build_sm121_image.sh which will consume this
image as BASE_IMAGE for the sgl-kernel sm121 layer.
EOF
}

# ============================================================================
# Argument parsing
# ============================================================================

while [[ $# -gt 0 ]]; do
    case "$1" in
        --no-push) PUSH_IMAGE=0; shift ;;
        --remote-host)
            [[ $# -ge 2 ]] || die "--remote-host requires a value (e.g. --remote-host root@spark4.local)"
            REMOTE_HOST="$2"
            shift 2
            ;;
        --help|-h) usage; exit 0 ;;
        *)         die "Unknown argument: $1 (use --help)" ;;
    esac
done

# Re-derive the podman connection name from a --remote-host override, unless
# the user pinned it explicitly via BUILD_PYTORCH_PODMAN_CONNECTION (that
# env var must win regardless of CLI-vs-default REMOTE_HOST).
if [[ -z "${BUILD_PYTORCH_PODMAN_CONNECTION:-}" ]]; then
    PODMAN_CONNECTION="${REMOTE_HOST##*@}"
    # IPv4 stays whole; a DNS name -> first label (see note at the initial assignment).
    if [[ ! "${PODMAN_CONNECTION}" =~ ^[0-9]+(\.[0-9]+){3}$ ]]; then
        PODMAN_CONNECTION="${PODMAN_CONNECTION%%.*}"
    fi
fi

# ============================================================================
# Preflight
# ============================================================================

preflight() {
    log "Preflight"

    if [[ ! -f "${PATCHES_DIR}/${RECIPE_NAME}.recipe" ]]; then
        die "Recipe not found: ${PATCHES_DIR}/${RECIPE_NAME}.recipe"
    fi

    for tool in git podman; do
        command -v "${tool}" >/dev/null || die "Required tool not found: ${tool}"
    done

    if [[ ! -f "${PODMAN_SSH_IDENTITY}" ]]; then
        cat >&2 <<EOF

ERROR: SSH identity '${PODMAN_SSH_IDENTITY}' not found.

Create it with:
  ssh-keygen -t ed25519 -f ${PODMAN_SSH_IDENTITY} -N ""
  ssh-copy-id -i ${PODMAN_SSH_IDENTITY} ${REMOTE_HOST}
EOF
        exit 1
    fi

    echo "Recipe present, tools available, SSH identity found"
}

# ============================================================================
# Podman connection to the remote build host
# (same logic as build_sm121_image.sh — reuses the same connection if already
# registered)
# ============================================================================

ensure_podman_connection() {
    log "Ensuring podman connection '${PODMAN_CONNECTION}' → ${REMOTE_HOST}"

    if podman system connection list --format '{{.Name}}' | grep -qxF "${PODMAN_CONNECTION}"; then
        echo "Connection '${PODMAN_CONNECTION}' already registered"
    else
        echo "Registering new podman connection..."
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
        die "Podman connection '${PODMAN_CONNECTION}' is not responding. On ${REMOTE_HOST} check: systemctl status podman.socket"
    fi

    local remote_arch
    remote_arch="$(podman --connection "${PODMAN_CONNECTION}" info --format '{{.Host.Arch}}')"
    if [[ "${remote_arch}" != "arm64" && "${remote_arch}" != "aarch64" ]]; then
        die "Remote host is ${remote_arch}, expected arm64/aarch64"
    fi
    echo "Remote podman is reachable (arch=${remote_arch})"
}

# ============================================================================
# cuda-containers clone + branch management
# (same logic as build_sm121_image.sh)
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
# Install our custom recipe into the cuda-containers clone
# ============================================================================

install_recipe() {
    log "Installing ${RECIPE_NAME}.recipe"

    cd "${CUDA_CONTAINERS_DIR}"

    # Drop our customized recipe into the clone's container-recipes/ directory.
    # This overrides scitrera's upstream version of the same filename (which
    # has the unresolved NCCL "2.29.XX" placeholder and the scitrera/ image tag).
    install -m 0644 "${PATCHES_DIR}/${RECIPE_NAME}.recipe" \
        "container-recipes/${RECIPE_NAME}.recipe"
    echo "Installed container-recipes/${RECIPE_NAME}.recipe"

    # Verify the recipe has our expected IMAGE_TAG (sanity check — catches
    # the case where someone edits the recipe without updating the script).
    grep -q "^IMAGE_TAG=${IMAGE_TAG}$" "container-recipes/${RECIPE_NAME}.recipe" \
        || die "Recipe IMAGE_TAG doesn't match expected ${IMAGE_TAG}"
}

# ============================================================================
# Build via remote podman socket
# ============================================================================

run_build() {
    log "Running podman build for pytorch base on '${PODMAN_CONNECTION}' (3-5 hours expected)"
    cd "${CUDA_CONTAINERS_DIR}"

    local recipe_file="container-recipes/${RECIPE_NAME}.recipe"
    [[ -f "${recipe_file}" ]] || die "Recipe not found: ${recipe_file}"

    local R_DOCKERFILE R_TARGET R_CUDA_VERSION R_CUDA_SHORT
    local R_NCCL_VERSION R_NCCL_REF R_NCCL_REPO
    local R_BUILD_JOBS R_TORCH_CUDA_ARCH_LIST R_NVCC_GENCODE
    local R_TORCH_VERSION R_TORCH_AUDIO_VERSION R_TORCH_VISION_VERSION
    local R_IMAGE_TAG
    # shellcheck disable=SC1090
    source <(
        set -a
        # shellcheck disable=SC1090
        source "${recipe_file}"
        echo "R_DOCKERFILE='${DOCKERFILE}'"
        echo "R_TARGET='${TARGET}'"
        echo "R_CUDA_VERSION='${CUDA_VERSION}'"
        echo "R_CUDA_SHORT='${CUDA_SHORT}'"
        echo "R_NCCL_VERSION='${NCCL_VERSION}'"
        echo "R_NCCL_REF='${NCCL_REF}'"
        echo "R_NCCL_REPO='${NCCL_REPO}'"
        echo "R_BUILD_JOBS='${BUILD_JOBS}'"
        echo "R_TORCH_CUDA_ARCH_LIST='${TORCH_CUDA_ARCH_LIST}'"
        echo "R_NVCC_GENCODE='${NVCC_GENCODE}'"
        echo "R_TORCH_VERSION='${TORCH_VERSION}'"
        echo "R_TORCH_AUDIO_VERSION='${TORCH_AUDIO_VERSION}'"
        echo "R_TORCH_VISION_VERSION='${TORCH_VISION_VERSION}'"
        echo "R_IMAGE_TAG='${IMAGE_TAG}'"
    )

    if [[ "${R_IMAGE_TAG}" != "${IMAGE_TAG}" ]]; then
        die "Recipe IMAGE_TAG (${R_IMAGE_TAG}) does not match script IMAGE_TAG (${IMAGE_TAG})"
    fi

    # Allow env override of BUILD_JOBS even if the recipe says something
    # different. The env var is our on-the-spot memory-pressure control.
    local effective_jobs="${BUILD_JOBS}"

    echo "Recipe + runtime values:"
    echo "  DOCKERFILE           = ${R_DOCKERFILE}"
    echo "  TARGET               = ${R_TARGET}"
    echo "  CUDA_VERSION         = ${R_CUDA_VERSION}"
    echo "  NCCL_VERSION         = ${R_NCCL_VERSION}"
    echo "  NCCL_REF             = ${R_NCCL_REF}"
    echo "  NCCL_REPO            = ${R_NCCL_REPO}"
    echo "  TORCH_VERSION        = ${R_TORCH_VERSION}"
    echo "  TORCH_VISION_VERSION = ${R_TORCH_VISION_VERSION}"
    echo "  TORCH_AUDIO_VERSION  = ${R_TORCH_AUDIO_VERSION}"
    echo "  TORCH_CUDA_ARCH_LIST = ${R_TORCH_CUDA_ARCH_LIST}"
    echo "  NVCC_GENCODE         = ${R_NVCC_GENCODE}"
    echo "  BUILD_JOBS (effective)= ${effective_jobs}  (overrides recipe if env set)"
    echo "  IMAGE_TAG            = ${IMAGE_TAG}"
    echo "  IMAGE_TAG (FQN)      = docker.io/${IMAGE_TAG}"

    # Tag with BOTH the short name and the docker.io/ fully-qualified name.
    # Why both: podman stores images built with short-name `-t` arguments
    # under a `localhost/` prefix by default — which means `FROM xomoxcc/...`
    # in a downstream Dockerfile would not find it (buildah normalizes the
    # short name to `docker.io/...` during FROM resolution). Explicitly
    # tagging with `docker.io/<name>` stores the image under that FQN so
    # both the short-name lookup and the docker.io-resolution work. This
    # was the source of a real problem in an earlier sgl-kernel build where
    # the image ended up as `localhost/xomoxcc/dgx-spark-sglang:0.5.10-sm121`
    # and needed a manual `podman tag` before `podman save` could be used.
    podman --connection "${PODMAN_CONNECTION}" build \
        -f "container-build/${R_DOCKERFILE}" \
        --target "${R_TARGET}" \
        --build-arg "CUDA_VERSION=${R_CUDA_VERSION}" \
        --build-arg "BUILD_JOBS=${effective_jobs}" \
        --build-arg "NCCL_VERSION=${R_NCCL_VERSION}" \
        --build-arg "NCCL_REF=${R_NCCL_REF}" \
        --build-arg "NCCL_REPO=${R_NCCL_REPO}" \
        --build-arg "TORCH_CUDA_ARCH_LIST=${R_TORCH_CUDA_ARCH_LIST}" \
        --build-arg "NVCC_GENCODE=${R_NVCC_GENCODE}" \
        --build-arg "TORCH_VERSION=${R_TORCH_VERSION}" \
        --build-arg "TORCH_AUDIO_VERSION=${R_TORCH_AUDIO_VERSION}" \
        --build-arg "TORCH_VISION_VERSION=${R_TORCH_VISION_VERSION}" \
        -t "${IMAGE_TAG}" \
        -t "docker.io/${IMAGE_TAG}" \
        container-build/

    # Verify the image exists under both names we tagged.
    if ! podman --connection "${PODMAN_CONNECTION}" image exists "docker.io/${IMAGE_TAG}"; then
        die "Build finished but docker.io/${IMAGE_TAG} not present in remote image store"
    fi
    echo "Remote build complete: ${IMAGE_TAG} (also tagged as docker.io/${IMAGE_TAG})"
}

# ============================================================================
# Transfer image from remote to local (only when pushing)
# ============================================================================

transfer_image_from_remote() {
    if (( PUSH_IMAGE == 0 )); then
        log "Skipping image scp (--no-push)"
        return
    fi

    log "Copying docker.io/${IMAGE_TAG} from ${PODMAN_CONNECTION} to local image store"

    # Remove any older local copy under either tag so the scp doesn't silently
    # keep stale layers around.
    podman image rm "${IMAGE_TAG}" 2>/dev/null || true
    podman image rm "docker.io/${IMAGE_TAG}" 2>/dev/null || true

    # Stream save → load so pv can show progress. `podman image scp` would
    # do the same internally but hides throughput/ETA.
    local size
    size=$(podman --connection "${PODMAN_CONNECTION}" image inspect \
            --format '{{.Size}}' "docker.io/${IMAGE_TAG}" 2>/dev/null || echo "")

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

    podman image inspect "docker.io/${IMAGE_TAG}" >/dev/null \
        || die "Image not present locally after transfer"
    echo "Image transferred to local store: docker.io/${IMAGE_TAG}"
}

# ============================================================================
# Push to Docker Hub (only when --push is set, runs on x86)
# ============================================================================

run_push() {
    if (( PUSH_IMAGE == 0 )); then
        log "Skipping push (--no-push)"
        return
    fi

    log "Pushing docker.io/${IMAGE_TAG} to Docker Hub from x86"

    local auth_file="${REGISTRY_AUTH_FILE:-${XDG_RUNTIME_DIR:-/run}/containers/auth.json}"
    if [[ ! -f "${auth_file}" ]]; then
        auth_file="${HOME}/.docker/config.json"
    fi
    if [[ ! -f "${auth_file}" ]]; then
        warn "No registry auth file found (checked \$REGISTRY_AUTH_FILE and ~/.docker/config.json)"
        echo "Run 'podman login docker.io -u xomoxcc' on this host, then re-run with --push."
        die "Registry authentication missing"
    fi

    podman push "docker.io/${IMAGE_TAG}"
    echo "Image pushed: docker.io/${IMAGE_TAG}"
}

# ============================================================================
# Next steps
# ============================================================================

print_next_steps() {
    cat <<EOF

$(log "pytorch base build complete")

Image: ${IMAGE_TAG}
Location: remote podman store on ${PODMAN_CONNECTION} (NOT pushed to Docker Hub).

Verify on the build host:
  podman --connection ${PODMAN_CONNECTION} image inspect ${IMAGE_TAG} \\
      --format '{{.Config.Env}} {{.Created}}'

Next steps:

1. Verify that scripts/patches/sglang-{,gemma4-}sm121-dev1.recipe reference
   this image as BASE_IMAGE:
     BASE_IMAGE=${IMAGE_TAG}

2. Run the sgl-kernel sm121 build on top of this base:
     bash scripts/build_sm121_image.sh --no-push

3. Distribute to all 4 sparks via the pipe pattern:
     podman save | ssh target "ctr -n k8s.io image import -"

4. Redeploy SGLang on K3s:
     ansible-playbook k8s_dgx.yml --tags sglang

The resulting sglang image should run at ~baseline perf instead of the
~45% regression documented in reference_sm121_build_base_regression.

EOF
}

# ============================================================================
# Main
# ============================================================================

main() {
    preflight
    ensure_podman_connection
    prepare_cuda_containers
    install_recipe
    run_build
    transfer_image_from_remote
    run_push
    print_next_steps
}

main "$@"
