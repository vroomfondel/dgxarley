#!/usr/bin/env bash
#
# build_dgx_spark_quant_image.sh — Build xomoxcc/dgx-spark-quant:*-sm121.
#
# Thin sibling of build_sm121_image.sh: same remote-podman-on-a-Spark flow, but
# for the NVFP4/ModelOpt quantization toolchain image
# (scripts/patches/dgx-spark-quant-sm121.Dockerfile) instead of the SGLang
# serving image. That Dockerfile is a SMALL layer on top of the serving image
# (FROM xomoxcc/dgx-spark-sglang:<tag> + `pip install accelerate hf_transfer` +
# import asserts — NO patch stack, NO cuda-containers clone, NO recipe), so this
# script is correspondingly small.
#
# WHY arm64-only-on-a-Spark (no QEMU, no buildx multiarch): the base is a full
# CUDA/sm121 image and the layers do real torch/modelopt imports at build time.
# Emulated arm64 would be unusably slow, so — exactly like build_sm121_image.sh —
# the build runs NATIVELY on an arm64 Spark via a registered podman connection,
# then the result is `podman image scp`'d back here and pushed from here.
#
# Workflow (all steps run on the x86 control host)
# -------------------------------------------------
# 1. Ensure a registered podman connection to the arm64 build host (default
#    spark4) using a dedicated UNENCRYPTED SSH key (podman's Go SSH client
#    cannot use ssh-agent or encrypted keys). Created on demand if missing.
# 2. Verify the BASE_IMAGE (the serving image) is present in the remote podman
#    store, pulling it from Docker Hub if needed — fail fast with a clear hint
#    otherwise (a missing base makes the FROM step 404 after a long retry).
# 3. `podman --connection <name> build` — context is streamed x86 → Spark over
#    the podman socket, the build runs natively on arm64, the image lands in the
#    Spark's local podman store. (Context is an EMPTY temp dir: the Dockerfile
#    COPYs nothing, so there is nothing to stream — keeps it instant.)
# 4. `podman image scp <name>::<image>` pulls the built image Spark → x86.
# 5. `podman push` from x86 using this host's pre-existing registry creds
#    (the Spark never holds Docker Hub credentials).
#
# Subcommands / usage
# -------------------
#   build_dgx_spark_quant_image.sh            # build (remote) → scp local → push
#   build_dgx_spark_quant_image.sh --no-push  # build → scp local, no push
#   build_dgx_spark_quant_image.sh check      # validate remote podman connection only
#   build_dgx_spark_quant_image.sh pull       # pull the pushed image from the registry to local
#   build_dgx_spark_quant_image.sh --help
#
# Prerequisites (identical to build_sm121_image.sh)
# -------------------------------------------------
# x86 control host: podman; an unencrypted SSH key for podman
#   (ssh-keygen -t ed25519 -f ~/.ssh/id_podman -N "" ; ssh-copy-id -i ~/.ssh/id_podman root@spark4);
#   `podman login docker.io -u xomoxcc` already done.
# Spark build host: podman + `systemctl enable --now podman.socket` (root socket).
#

set -euo pipefail

# ============================================================================
# Configuration
# ============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PATCHES_DIR="${SCRIPT_DIR}/patches"
DOCKERFILE="${PATCHES_DIR}/dgx-spark-quant-sm121.Dockerfile"

# Tag suffix, shared by the produced image and the serving base it layers on.
# Bump this in lockstep with the serving image you are quantizing against.
TAG="${BUILD_QUANT_TAG:-0.5.14-sm121}"

# Produced image. Overridable wholesale via BUILD_QUANT_IMAGE.
IMAGE="${BUILD_QUANT_IMAGE:-xomoxcc/dgx-spark-quant:${TAG}}"

# The serving image this layers on (Dockerfile ARG BASE_IMAGE). Defaults to the
# same tag; override with --base or BUILD_QUANT_BASE_IMAGE (e.g. to rebase onto
# xomoxcc/dgx-spark-pytorch-dev:2.12.0-v1-cu132 per the Dockerfile's NOTE).
BASE_IMAGE="${BUILD_QUANT_BASE_IMAGE:-xomoxcc/dgx-spark-sglang:${TAG}}"

# Remote arm64 build host + its registered podman connection name + SSH key.
# Same defaults/derivation as build_sm121_image.sh.
REMOTE_HOST="${BUILD_QUANT_REMOTE_HOST:-root@spark4.local}"
PODMAN_CONNECTION="${BUILD_QUANT_PODMAN_CONNECTION:-}"
PODMAN_SSH_IDENTITY="${BUILD_QUANT_SSH_IDENTITY:-${HOME}/.ssh/id_podman}"

PUSH_IMAGE=1
NO_LOCAL_COPY=0
SUBCOMMAND="build"

# ============================================================================
# Helpers (style matched to build_sm121_image.sh)
# ============================================================================

log()  { printf '\n\033[1;34m=== %s ===\033[0m\n' "$*"; }
warn() { printf '\033[1;33mWARN: %s\033[0m\n' "$*" >&2; }
die()  { printf '\033[1;31mERROR: %s\033[0m\n' "$*" >&2; exit 1; }

usage() {
    cat <<EOF
Usage: $(basename "$0") [SUBCOMMAND] [options]

Builds ${IMAGE} on the remote arm64 build host via the podman socket, copies
the result back here (unless --no-local-copy), and pushes it from here
(unless --no-push or --no-local-copy).

Subcommands:
  build   (default)  Ensure connection → remote build → scp local → push.
  check              Validate the remote podman connection and exit.
  pull               Pull ${IMAGE} from the registry to the local podman store.

Options:
  --tag TAG          Tag suffix for BOTH the produced image and its serving base
                     (image → xomoxcc/dgx-spark-quant:TAG, base →
                     xomoxcc/dgx-spark-sglang:TAG). Current: ${TAG}
  --base IMAGE       Serving/base image to layer on (Dockerfile ARG BASE_IMAGE),
                     verbatim. Current: ${BASE_IMAGE}
  --image IMAGE      Produced image reference, verbatim. Current: ${IMAGE}
  --remote-host user@host
                     Remote arm64 build host (SSH + podman socket).
                     Current: ${REMOTE_HOST}
  --podman-connection NAME
                     Registered podman connection to use/create. If omitted,
                     derived from --remote-host (strip user@ and domain).
  --no-local-copy    Skip the scp of the built image back here. Implies --no-push
                     (you cannot push what was never copied local).
  --no-push          Skip 'podman push' after build + scp.
  --help             Show this help.

Environment overrides:
  BUILD_QUANT_TAG                Tag suffix.               Default: ${TAG}
  BUILD_QUANT_IMAGE             Produced image (verbatim). Default: derived from TAG
  BUILD_QUANT_BASE_IMAGE        Serving/base image.        Default: derived from TAG
  BUILD_QUANT_REMOTE_HOST       user@host for the Spark.   Default: ${REMOTE_HOST}
  BUILD_QUANT_PODMAN_CONNECTION Podman connection name.    Default: derived from host
  BUILD_QUANT_SSH_IDENTITY      Unencrypted SSH key.       Default: ${PODMAN_SSH_IDENTITY}
EOF
}

# ============================================================================
# Argument parsing
# ============================================================================

while [[ $# -gt 0 ]]; do
    case "$1" in
        build|check|pull) SUBCOMMAND="$1"; shift ;;
        --no-push) PUSH_IMAGE=0; shift ;;
        --no-local-copy) NO_LOCAL_COPY=1; PUSH_IMAGE=0; shift ;;
        --tag) shift; [[ $# -gt 0 ]] || die "--tag requires an argument"; TAG="$1"; shift
               # Re-derive image/base from the new tag unless they were overridden by env.
               [[ -n "${BUILD_QUANT_IMAGE:-}" ]] || IMAGE="xomoxcc/dgx-spark-quant:${TAG}"
               [[ -n "${BUILD_QUANT_BASE_IMAGE:-}" ]] || BASE_IMAGE="xomoxcc/dgx-spark-sglang:${TAG}" ;;
        --tag=*) TAG="${1#--tag=}"; shift
               [[ -n "${BUILD_QUANT_IMAGE:-}" ]] || IMAGE="xomoxcc/dgx-spark-quant:${TAG}"
               [[ -n "${BUILD_QUANT_BASE_IMAGE:-}" ]] || BASE_IMAGE="xomoxcc/dgx-spark-sglang:${TAG}" ;;
        --base) shift; [[ $# -gt 0 ]] || die "--base requires an argument"; BASE_IMAGE="$1"; shift ;;
        --base=*) BASE_IMAGE="${1#--base=}"; shift ;;
        --image) shift; [[ $# -gt 0 ]] || die "--image requires an argument"; IMAGE="$1"; shift ;;
        --image=*) IMAGE="${1#--image=}"; shift ;;
        --remote-host) shift; [[ $# -gt 0 ]] || die "--remote-host requires an argument"; REMOTE_HOST="$1"; shift ;;
        --remote-host=*) REMOTE_HOST="${1#--remote-host=}"; shift ;;
        --podman-connection) shift; [[ $# -gt 0 ]] || die "--podman-connection requires an argument"; PODMAN_CONNECTION="$1"; shift ;;
        --podman-connection=*) PODMAN_CONNECTION="${1#--podman-connection=}"; shift ;;
        --help|-h) usage; exit 0 ;;
        *) die "Unknown argument: $1 (use --help)" ;;
    esac
done

# Derive the podman connection name from REMOTE_HOST if not set explicitly.
# "root@spark4.local" -> "spark4"; an IPv4 literal is kept whole.
if [[ -z "${PODMAN_CONNECTION}" ]]; then
    PODMAN_CONNECTION="${REMOTE_HOST##*@}"
    if [[ ! "${PODMAN_CONNECTION}" =~ ^[0-9]+(\.[0-9]+){3}$ ]]; then
        PODMAN_CONNECTION="${PODMAN_CONNECTION%%.*}"
    fi
fi

# ============================================================================
# Podman connection to the remote build host (lifted from build_sm121_image.sh)
# ============================================================================

ensure_podman_connection() {
    log "Ensuring podman connection '${PODMAN_CONNECTION}' → ${REMOTE_HOST}"

    command -v podman >/dev/null || die "podman not found on this host"

    if [[ ! -f "${PODMAN_SSH_IDENTITY}" ]]; then
        cat >&2 <<EOF

ERROR: SSH identity '${PODMAN_SSH_IDENTITY}' not found.

Podman's Go SSH client cannot use ssh-agent or encrypted keys, so a dedicated
unencrypted key is required. Create it with:

  ssh-keygen -t ed25519 -f ${PODMAN_SSH_IDENTITY} -N ""
  ssh-copy-id -i ${PODMAN_SSH_IDENTITY} ${REMOTE_HOST}
EOF
        exit 1
    fi

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
        cat >&2 <<EOF

ERROR: Podman connection '${PODMAN_CONNECTION}' is not responding.

On ${REMOTE_HOST}:  systemctl enable --now podman.socket
Then check:         podman system connection list
EOF
        exit 1
    fi

    local remote_arch
    remote_arch="$(podman --connection "${PODMAN_CONNECTION}" info --format '{{.Host.Arch}}')"
    if [[ "${remote_arch}" != "arm64" && "${remote_arch}" != "aarch64" ]]; then
        die "Remote host is ${remote_arch}, expected arm64/aarch64 (sm121 base is arm64)"
    fi
    echo "Remote podman is reachable (arch=${remote_arch})"
}

# ============================================================================
# Verify the serving BASE_IMAGE is present in the remote podman store
# ============================================================================

ensure_base_image_present() {
    log "Verifying base image '${BASE_IMAGE}' is present on '${PODMAN_CONNECTION}'"

    if podman --connection "${PODMAN_CONNECTION}" image exists "docker.io/${BASE_IMAGE}" 2>/dev/null \
       || podman --connection "${PODMAN_CONNECTION}" image exists "${BASE_IMAGE}" 2>/dev/null; then
        echo "Base image found on ${PODMAN_CONNECTION}"
        return 0
    fi

    echo "Base image not found locally on ${PODMAN_CONNECTION}. Attempting pull from Docker Hub..."
    if podman --connection "${PODMAN_CONNECTION}" pull "docker.io/${BASE_IMAGE}" 2>&1; then
        echo "Base image pulled successfully."
        return 0
    fi

    die "Base image '${BASE_IMAGE}' not present on ${PODMAN_CONNECTION} and pull from Docker Hub failed. Build/push the serving image first (scripts/build_sm121_image.sh)."
}

# ============================================================================
# Build / copy / push / pull
# ============================================================================

run_build() {
    [[ -f "${DOCKERFILE}" ]] || die "Dockerfile not found: ${DOCKERFILE}"

    ensure_podman_connection
    ensure_base_image_present

    # Empty build context: the Dockerfile COPYs nothing, so streaming the repo
    # would be pure waste. -f points at the (out-of-context) Dockerfile by path.
    local ctx
    ctx="$(mktemp -d)"
    trap 'rm -rf "${ctx}"' RETURN

    log "Building ${IMAGE} on '${PODMAN_CONNECTION}' (BASE_IMAGE=${BASE_IMAGE})"
    podman --connection "${PODMAN_CONNECTION}" build \
        --build-arg "BASE_IMAGE=${BASE_IMAGE}" \
        -f "${DOCKERFILE}" \
        -t "${IMAGE}" \
        "${ctx}" \
        || die "Remote build failed"
    echo "Built ${IMAGE} on ${PODMAN_CONNECTION}"

    if (( NO_LOCAL_COPY == 1 )); then
        warn "--no-local-copy: image stays only on ${PODMAN_CONNECTION} (not copied here, not pushed)"
        return 0
    fi

    log "Copying ${IMAGE} from ${PODMAN_CONNECTION} → this host (podman image scp)"
    podman image scp "${PODMAN_CONNECTION}::${IMAGE}" || die "podman image scp failed"
    echo "Image present in local podman store: ${IMAGE}"

    if (( PUSH_IMAGE == 1 )); then
        run_push
    else
        echo "Skipping push (--no-push). To push later: $(basename "$0") --image ${IMAGE} && podman push ${IMAGE}"
    fi
}

run_push() {
    log "Pushing ${IMAGE} (using this host's registry credentials)"
    podman image exists "${IMAGE}" || die "Image '${IMAGE}' not in the local podman store — build/scp it first"
    podman push "${IMAGE}" || die "podman push failed (is 'podman login docker.io' done on this host?)"
    echo "Pushed ${IMAGE}"
}

run_pull() {
    command -v podman >/dev/null || die "podman not found on this host"
    log "Pulling ${IMAGE} from the registry → local podman store"
    podman pull "${IMAGE}" || die "podman pull failed"
    echo "Pulled ${IMAGE}"
}

# ============================================================================
# Main
# ============================================================================

case "${SUBCOMMAND}" in
    build) run_build ;;
    check) ensure_podman_connection; log "Remote connection OK" ;;
    pull)  run_pull ;;
    *)     die "Unknown subcommand: ${SUBCOMMAND}" ;;
esac
