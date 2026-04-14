#!/usr/bin/env bash
#
# build_cudnn_image.sh — Build xomoxcc/dgx-spark-sglang:0.5.10-cudnn.
#
# Stacks nvidia-cudnn-cu12 + nvidia-cudnn-frontend Python wheels on top
# of scitrera/dgx-spark-sglang:0.5.10 (or any other compatible base image
# — override via BUILD_CUDNN_BASE_IMAGE) so that SGLang's fi_cudnn FP4
# GEMM backend (fp4_gemm_backend=flashinfer_cudnn) becomes usable.
#
# Base image choice: we deliberately build on the upstream scitrera image,
# not on xomoxcc/dgx-spark-sglang:0.5.10-sm121. The sm121 CUTLASS MoE
# patch is irrelevant for the current matrix workloads (the GLM-4.7-NVFP4
# EP=1 sweep proved triton + cutlass-direct MoE are stable on SM121
# without the patch, because EP=1 avoids the shared-memory / EP-assert
# crash), and the sm121 image carries a ~45% perf regression vs upstream
# (see reference_sm121_build_base_regression memory: forced fallback to
# torch 2.10/cu13.1 instead of upstream's torch 2.11/cu13.2). Building on
# scitrera keeps the upstream perf profile and only adds the cuDNN layer.
#
# Why this exists
# ---------------
# flashinfer 0.4.x ships a runtime cuDNN availability check in the
# _is_problem_size_supported wrapper for the cudnn FP4 GEMM requirement.
# If libcudnn isn't loadable, the check raises:
#   flashinfer/gemm/gemm_base.py:_check_cudnn_availability
#   RuntimeError: cuDNN is not available. Please install cuDNN to use
#   FP8 GEMM functions. You can install it with:
#     pip install nvidia-cudnn-cu12 nvidia-cudnn-frontend
# scitrera/dgx-spark-sglang:0.5.10 ships flashinfer without these wheels,
# so every matrix row that selects fp4_gemm_backend=flashinfer_cudnn
# crashes (CG-on → startup_crash during warmup forward; eager →
# bench_crash at first request). See the GLM-4.7-NVFP4 EP=1 testlog for
# the full picture.
#
# This Dockerfile is intentionally tiny — one pip install layer plus a
# post-install smoke test that invokes the same _check_cudnn_availability
# function so the build fails fast if the wheels don't actually make
# flashinfer happy.
#
# Workflow
# --------
# 1. Preflight: verify Dockerfile + podman + SSH identity.
# 2. Ensure a registered podman connection to the arm64 build host
#    (spark4). Reuses the same connection name as build_sm121_image.sh.
# 3. podman build via the remote socket — the Dockerfile is just
#    FROM + RUN pip install, so the build context is essentially empty
#    and the build time is dominated by the pip download itself.
# 4. Optionally transfer the image back to x86 and push to Docker Hub.
#
# Build time: ~5-10 min cold (pip downloads ~700 MB of cuDNN wheels).
# Re-runs reuse the cached pip layer.
#
# Prerequisites: same as build_sm121_image.sh — unencrypted SSH key at
# ~/.ssh/id_podman, podman on both ends, podman.socket enabled on
# spark4 as root. See that script's header for the full setup walkthrough.
#

set -euo pipefail

# ============================================================================
# Configuration
# ============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PATCHES_DIR="${SCRIPT_DIR}/patches"

DOCKERFILE="${PATCHES_DIR}/sglang-0.5.10-cudnn.Dockerfile"

BASE_IMAGE="${BUILD_CUDNN_BASE_IMAGE:-scitrera/dgx-spark-sglang:0.5.10}"
IMAGE_TAG="${BUILD_CUDNN_IMAGE_TAG:-xomoxcc/dgx-spark-sglang:0.5.10-cudnn}"

# Remote build host + podman connection. Defaults match
# build_sm121_image.sh / build_pytorch_base_image.sh so the same registered
# podman connection can be reused. Both can also be set by flags
# (--remote-host / --podman-connection); PODMAN_CONNECTION is derived in
# the flag handler (not here) so a late --remote-host override propagates.
REMOTE_HOST="${BUILD_CUDNN_REMOTE_HOST:-root@spark4.local}"
PODMAN_CONNECTION="${BUILD_CUDNN_PODMAN_CONNECTION:-}"
PODMAN_SSH_IDENTITY="${BUILD_CUDNN_SSH_IDENTITY:-${HOME}/.ssh/id_podman}"

# Docker Hub push. ON by default to match build_sm121_image.sh's behavior —
# pass --no-push to keep a local copy on x86 without pushing, or
# --no-local-copy to also skip the scp back so the image stays only on
# the remote build host. Push uses the x86 host's pre-configured registry
# credentials after streaming the image back from the remote build host.
PUSH_IMAGE=1
NO_LOCAL_COPY=0

# Temporary build-context directory. Declared at script scope so the EXIT
# trap can clean it up regardless of which function raised an error. We do
# NOT use `trap ... RETURN` inside run_build — RETURN is global, so it
# would re-fire for every subsequent function return and, under `set -u`,
# crash trying to expand an unset ctx_dir after run_build has already
# cleaned it up.
BUILD_CTX_DIR=""

# ============================================================================
# Helpers
# ============================================================================

log()  { printf '\n\033[1;34m=== %s ===\033[0m\n' "$*"; }
warn() { printf '\033[1;33mWARN: %s\033[0m\n' "$*" >&2; }
die()  { printf '\033[1;31mERROR: %s\033[0m\n' "$*" >&2; exit 1; }

usage() {
    cat <<EOF
Usage: $(basename "$0") [--remote-host user@host] [--podman-connection NAME]
                        [--no-local-copy] [--no-push] [--help]

Builds ${IMAGE_TAG} on the remote build host via the podman socket.
Adds nvidia-cudnn-cu12 + nvidia-cudnn-frontend wheels on top of
${BASE_IMAGE}. Expected duration: ~5-10 min cold.

Options:
  --remote-host user@host
               SSH target for the remote build host. Must be the same user
               whose podman.socket is running and whose registered podman
               connection you want to use. Default: ${REMOTE_HOST}
  --podman-connection NAME
               Name of the registered podman connection to use. When
               omitted, derived from --remote-host (strip user@ and domain).
  --no-local-copy
               Skip the image scp back to this host AFTER the remote build
               finishes. The image stays only in the remote build host's
               podman store. Use this when a subsequent distribute script
               (e.g. distrcudnnimage.sh) will pull the image directly from
               the build host via a throwaway registry, so the slow LAN
               transfer back to x86 is unnecessary.
               Implies --no-push.
  --no-push    Skip 'podman push' to Docker Hub after the local copy.
               A local copy still happens (unless --no-local-copy is also
               given). Useful when you want the image in your x86 podman
               store for inspection but don't want to publish it yet.
  --help       Show this help.

Environment overrides (lower precedence than flags):
  BUILD_CUDNN_BASE_IMAGE         FROM image for the Dockerfile.
                                 Default: ${BASE_IMAGE}
  BUILD_CUDNN_IMAGE_TAG          Output tag for the built image.
                                 Default: ${IMAGE_TAG}
  BUILD_CUDNN_REMOTE_HOST        user@host for remote build SSH.
                                 Default: ${REMOTE_HOST}
  BUILD_CUDNN_PODMAN_CONNECTION  Registered podman connection name.
                                 Default: derived from REMOTE_HOST
  BUILD_CUDNN_SSH_IDENTITY       Unencrypted SSH private key for podman.
                                 Default: ${PODMAN_SSH_IDENTITY}

Typical iteration flow (build remote, distribute via registry, no x86 copy):
  ./scripts/build_cudnn_image.sh --remote-host root@spark4.local --no-local-copy
  ./scripts/distrcudnnimage.sh   --source spark4.local --registry-host 10.10.10.4
EOF
}

# ============================================================================
# Argument parsing
# ============================================================================

while [[ $# -gt 0 ]]; do
    case "$1" in
        --no-push)
            PUSH_IMAGE=0
            shift
            ;;
        --no-local-copy)
            # Skip the save|load transfer back to x86 (and therefore also
            # skip the push — you can't push what you don't have locally).
            NO_LOCAL_COPY=1
            PUSH_IMAGE=0
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
        --help|-h)
            usage
            exit 0
            ;;
        *)
            die "Unknown argument: $1 (use --help)"
            ;;
    esac
done

# Derive PODMAN_CONNECTION from REMOTE_HOST if still unset. Done AFTER flag
# parsing so that a late --remote-host override propagates to the derived
# connection name. --podman-connection always wins.
if [[ -z "${PODMAN_CONNECTION}" ]]; then
    PODMAN_CONNECTION="${REMOTE_HOST##*@}"
    PODMAN_CONNECTION="${PODMAN_CONNECTION%%.*}"
fi

# ============================================================================
# Preflight
# ============================================================================

preflight() {
    log "Preflight"

    [[ -f "${DOCKERFILE}" ]] || die "Dockerfile not found: ${DOCKERFILE}"

    command -v podman >/dev/null || die "Required tool not found: podman"

    if [[ ! -f "${PODMAN_SSH_IDENTITY}" ]]; then
        cat >&2 <<EOF

ERROR: SSH identity '${PODMAN_SSH_IDENTITY}' not found.

Create it with:
  ssh-keygen -t ed25519 -f ${PODMAN_SSH_IDENTITY} -N ""
  ssh-copy-id -i ${PODMAN_SSH_IDENTITY} ${REMOTE_HOST}
EOF
        exit 1
    fi

    echo "Dockerfile present, tools available, SSH identity found"
    echo "BASE_IMAGE = ${BASE_IMAGE}"
    echo "IMAGE_TAG  = ${IMAGE_TAG}"
}

# ============================================================================
# Podman connection to the remote build host
# (same logic as build_sm121_image.sh / build_pytorch_base_image.sh)
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
# Verify the base image is available on the remote host
# ============================================================================

ensure_base_image() {
    log "Checking for base image ${BASE_IMAGE} on ${PODMAN_CONNECTION}"

    # Try a few name variants since podman may store images under short or
    # FQN forms depending on how they were pulled/built.
    for candidate in "${BASE_IMAGE}" "docker.io/${BASE_IMAGE}" "localhost/${BASE_IMAGE}"; do
        if podman --connection "${PODMAN_CONNECTION}" image exists "${candidate}" 2>/dev/null; then
            echo "Found base image: ${candidate}"
            return 0
        fi
    done

    echo "Base image not present locally on ${PODMAN_CONNECTION}; attempting to pull..."
    if podman --connection "${PODMAN_CONNECTION}" pull "${BASE_IMAGE}" 2>/dev/null; then
        echo "Pulled ${BASE_IMAGE}"
        return 0
    fi

    die "Base image ${BASE_IMAGE} not found on ${PODMAN_CONNECTION} and could not be pulled. If this is a local-only image (e.g. the sm121 image built with --no-push), run build_sm121_image.sh first or pass BUILD_CUDNN_BASE_IMAGE=scitrera/dgx-spark-sglang:0.5.10 to build on top of the upstream image instead."
}

# ============================================================================
# Build via remote podman socket
# ============================================================================

cleanup_build_ctx() {
    if [[ -n "${BUILD_CTX_DIR}" && -d "${BUILD_CTX_DIR}" ]]; then
        rm -rf "${BUILD_CTX_DIR}"
    fi
}
trap cleanup_build_ctx EXIT

run_build() {
    log "Building ${IMAGE_TAG} on '${PODMAN_CONNECTION}' (~5-10 min cold)"

    # The build context is just the Dockerfile — copy it into a temp dir so
    # we don't accidentally upload the whole scripts/ tree as context. The
    # EXIT trap above cleans it up regardless of which function errors later.
    BUILD_CTX_DIR="$(mktemp -d)"
    cp "${DOCKERFILE}" "${BUILD_CTX_DIR}/Dockerfile"

    # Tag with BOTH the short name and the docker.io/ fully-qualified name.
    # Reason: same as in build_pytorch_base_image.sh — podman stores images
    # built with short `-t` arguments under `localhost/` by default, which
    # breaks downstream `FROM` resolution that normalizes to `docker.io/`.
    podman --connection "${PODMAN_CONNECTION}" build \
        -f "${BUILD_CTX_DIR}/Dockerfile" \
        --build-arg "BASE_IMAGE=${BASE_IMAGE}" \
        -t "${IMAGE_TAG}" \
        -t "docker.io/${IMAGE_TAG}" \
        "${BUILD_CTX_DIR}"

    if ! podman --connection "${PODMAN_CONNECTION}" image exists "docker.io/${IMAGE_TAG}"; then
        die "Build finished but docker.io/${IMAGE_TAG} not present in remote image store"
    fi
    echo "Remote build complete: ${IMAGE_TAG} (also tagged as docker.io/${IMAGE_TAG})"
}

# ============================================================================
# Transfer image from remote to local (only when pushing)
# ============================================================================

transfer_image_from_remote() {
    if (( NO_LOCAL_COPY == 1 )); then
        log "Skipping image scp (--no-local-copy) — image stays on ${PODMAN_CONNECTION}"
        return
    fi

    log "Copying docker.io/${IMAGE_TAG} from ${PODMAN_CONNECTION} to local image store"

    # Remove any older local copy under either tag so the scp doesn't silently
    # keep stale layers around.
    podman image rm "${IMAGE_TAG}" 2>/dev/null || true
    podman image rm "docker.io/${IMAGE_TAG}" 2>/dev/null || true

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
# Push to Docker Hub (only when pushing, runs on x86)
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

$(log "cuDNN image build complete")

Image: ${IMAGE_TAG}
Base:  ${BASE_IMAGE}

Verify on the build host:
  podman --connection ${PODMAN_CONNECTION} image inspect ${IMAGE_TAG} \\
      --format '{{.Created}}  size={{.Size}}'

Quick fi_cudnn sanity check (run inside the image on any spark):
  podman run --rm ${IMAGE_TAG} python3 -c \\
      "from flashinfer.gemm.gemm_base import _check_cudnn_availability; \\
       _check_cudnn_availability(); print('cuDNN OK')"

Next steps:

1. Distribute to all 4 sparks via a throwaway registry on the build host:
     bash scripts/distrcudnnimage.sh --source ${PODMAN_CONNECTION}.local \\
                                     --registry-host 10.10.10.4

2. Point the matrix test runs at the new image by editing
   matrixtest_matrices/sglang_nn4_tp4_ep1/glm-4.7-nvfp4/*.yaml:
     image: "${IMAGE_TAG}"

3. Re-run only the fi_cudnn rows where cuDNN was the sole blocker —
   skip piecewise (always crashes) and skip fi_cutlass MoE rows (broken
   at EP=1 independently). For GLM-4.7 EP=1 that's 8 tests:
     tests 7, 8, 10, 11   (triton MoE + fi_cudnn non-piecewise)
     tests 31, 32, 34, 35 (cutlass MoE + fi_cudnn non-piecewise)
   Plus the new MTP variants 37 (triton MoE) and 38 (cutlass MoE) if
   you want to measure speculative decoding on the cuDNN path.

EOF
}

# ============================================================================
# Main
# ============================================================================

main() {
    preflight
    ensure_podman_connection
    ensure_base_image
    run_build
    transfer_image_from_remote
    run_push
    print_next_steps
}

main "$@"
