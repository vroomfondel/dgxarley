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
# 2. Ensure a registered podman connection to the arm64 build host (spark1)
#    that uses a dedicated unencrypted SSH key (Podman's Go SSH client cannot
#    use ssh-agent or encrypted keys). Create it on demand if missing.
# 3. Clone or update scitrera/cuda-containers locally on x86. Switch to a
#    local 'sm121' branch, hard-reset to origin/main (idempotent), drop in
#    the sgl-kernel patch + Dockerfile patch + recipe file.
# 4. Invoke `podman --connection <name> build` — the build context is
#    streamed from x86 to spark1 over the podman socket, the actual build
#    runs natively on arm64 (no QEMU), and the resulting image is stored
#    in spark1's local podman image store. The x86 host never writes
#    credentials to spark1.
# 5. `podman image scp` to pull the built image from spark1 back to x86.
# 6. `podman push` from x86 using the x86 host's pre-existing registry
#    credentials. spark1 never has Docker Hub credentials.
#
# Prerequisites on the x86 control host
# --------------------------------------
# - podman (`apt install podman`)
# - An unencrypted SSH key for podman: generate with
#     ssh-keygen -t ed25519 -f ~/.ssh/id_podman -N ""
#     ssh-copy-id -i ~/.ssh/id_podman root@spark1
#   The key MUST be unencrypted — podman's Go SSH client does not support
#   ssh-agent or encrypted keys. Override via BUILD_SM121_SSH_IDENTITY.
# - `podman login docker.io -u xomoxcc` already done on the x86 host.
# - ~10 GB free disk for the image after scp.
# - git, patch (cuda-containers clone + patch apply happens on x86).
#
# Prerequisites on spark1 (the build host)
# ----------------------------------------
# - podman (`apt install podman`)
# - podman.socket enabled as root:
#     systemctl enable --now podman.socket
#   This exposes /run/podman/podman.sock which the x86 client connects to.
# - ~50 GB free disk (sgl-kernel layers + final image in local image store).
# - NO credentials, NO clone, NO patches, NO local scripts. spark1 is a
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
RECIPE_NAME="sglang-0.5.10-sm121"
IMAGE_TAG="xomoxcc/dgx-spark-sglang:0.5.10-sm121"

# Remote build host (spark4, arm64). Uses a registered podman connection
# with a dedicated unencrypted SSH key. The connection name is derived from
# this value by stripping the user@ prefix so that `podman system connection
# list` shows a clean "spark1" entry.
REMOTE_HOST="${BUILD_SM121_REMOTE_HOST:-root@spark4.local}"
PODMAN_CONNECTION="${BUILD_SM121_PODMAN_CONNECTION:-${REMOTE_HOST##*@}}"
PODMAN_CONNECTION="${PODMAN_CONNECTION%%.*}"   # "spark1.local" -> "spark1"
PODMAN_SSH_IDENTITY="${BUILD_SM121_SSH_IDENTITY:-${HOME}/.ssh/id_podman}"

# Build-time parallelism. scitrera's Dockerfile.sglang-nightly defaults to
# ARG BUILD_JOBS=2 which uses only 2 of the DGX Spark GB10's 20 ARM cores
# (10%). MAX_JOBS is set from this ARG and propagates to sgl-kernel,
# flashinfer, and any Python extension build that honors it.
#
# GB10 topology: 10 Cortex-X925 + 10 Cortex-A725, 128 GB unified memory.
# CUTLASS template compiles can peak at ~5 GB per TU, so 16 parallel jobs
# leave ~33 GB headroom over the 80 GB worst-case footprint — comfortable.
# Push to 20 only if you've verified no other workload is running on the
# build host (else OOM-kill risk on the heavy CUTLASS translation units).
BUILD_JOBS="${BUILD_SM121_BUILD_JOBS:-16}"

PUSH_IMAGE=1

# ============================================================================
# Helpers
# ============================================================================

log()  { printf '\n\033[1;34m=== %s ===\033[0m\n' "$*"; }
warn() { printf '\033[1;33mWARN: %s\033[0m\n' "$*" >&2; }
die()  { printf '\033[1;31mERROR: %s\033[0m\n' "$*" >&2; exit 1; }

usage() {
    cat <<EOF
Usage: $(basename "$0") [--no-push] [--help]

Builds ${IMAGE_TAG} on spark1 via remote podman socket, copies the result
back to this host, and pushes it from here.

Options:
  --no-push    Skip 'podman push' after build + scp.
  --help       Show this help.

Environment overrides:
  BUILD_SM121_REMOTE_HOST        user@host for spark1 SSH.
                                 Default: ${REMOTE_HOST}
  BUILD_SM121_PODMAN_CONNECTION  Registered podman connection name.
                                 Default: derived from REMOTE_HOST (${PODMAN_CONNECTION})
  BUILD_SM121_SSH_IDENTITY       Unencrypted SSH private key for podman.
                                 Default: ${PODMAN_SSH_IDENTITY}
  BUILD_SM121_CC_DIR             Local cuda-containers clone path (on x86).
                                 Default: ${CUDA_CONTAINERS_DIR}

The entire script runs on the x86 control host. spark1 is used purely as
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
        --help|-h) usage; exit 0 ;;
        *)         die "Unknown argument: $1 (use --help)" ;;
    esac
done

# ============================================================================
# Preflight
# ============================================================================

preflight() {
    log "Preflight"

    local missing=0
    for f in sgl-kernel-sm121.patch dockerfile-sm121.patch \
             build-image-sh-podman.patch "${RECIPE_NAME}.recipe"; do
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

        # Resolve the remote podman socket path. For root on spark1 this is
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

    # 1. Drop sgl-kernel source patch into the build context (Dockerfile COPY reads from here).
    mkdir -p container-build/patches
    install -m 0644 "${PATCHES_DIR}/sgl-kernel-sm121.patch" \
        container-build/patches/sgl-kernel-sm121.patch
    echo "Installed container-build/patches/sgl-kernel-sm121.patch"

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

    echo "Recipe values:"
    echo "  DOCKERFILE           = ${R_DOCKERFILE}"
    echo "  TARGET               = ${R_TARGET}"
    echo "  BASE_IMAGE           = ${R_BASE_IMAGE}"
    echo "  FLASHINFER_VERSION   = ${R_FLASHINFER_VERSION}"
    echo "  TRANSFORMERS_VERSION = ${R_TRANSFORMERS_VERSION}"
    echo "  SGLANG_VERSION       = ${R_SGLANG_VERSION}"
    echo "  SGLANG_REF           = ${R_SGLANG_REF}"
    echo "  IMAGE_TAG            = ${IMAGE_TAG}"

    # The build context is container-build/ (contains Dockerfile + patches/
    # subdir). Podman streams it to spark1 over the socket; the build runs
    # natively on arm64 and the result lands in spark1's local image store.
    podman --connection "${PODMAN_CONNECTION}" build \
        -f "container-build/${R_DOCKERFILE}" \
        --target "${R_TARGET}" \
        --build-arg "BASE_IMAGE=${R_BASE_IMAGE}" \
        --build-arg "FLASHINFER_VERSION=${R_FLASHINFER_VERSION}" \
        --build-arg "TRANSFORMERS_VERSION=${R_TRANSFORMERS_VERSION}" \
        --build-arg "SGLANG_VERSION=${R_SGLANG_VERSION}" \
        --build-arg "SGLANG_REF=${R_SGLANG_REF}" \
        -t "${IMAGE_TAG}" \
        container-build/

    if ! podman --connection "${PODMAN_CONNECTION}" image inspect "${IMAGE_TAG}" >/dev/null 2>&1; then
        die "Build finished but ${IMAGE_TAG} not present in remote image store — check podman build output above"
    fi
    echo "Remote build complete: ${IMAGE_TAG}"
}

# ============================================================================
# Transfer built image from remote to local
# ============================================================================

transfer_image_from_remote() {
    log "Copying ${IMAGE_TAG} from ${PODMAN_CONNECTION} to local image store"

    # If an older local copy exists, remove it first so scp doesn't silently
    # keep stale layers around.
    podman image rm "${IMAGE_TAG}" 2>/dev/null || true

    podman image scp "${PODMAN_CONNECTION}::${IMAGE_TAG}" \
        || die "podman image scp failed"

    podman image inspect "${IMAGE_TAG}" >/dev/null \
        || die "Image not present locally after scp — check podman output"
    echo "Image transferred: ${IMAGE_TAG}"
}

# ============================================================================
# Push (from x86, using x86's credentials)
# ============================================================================

run_push() {
    if (( PUSH_IMAGE == 0 )); then
        log "Skipping push (--no-push)"
        return
    fi

    log "Pushing ${IMAGE_TAG} to Docker Hub from x86"

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

    podman push "${IMAGE_TAG}"
    echo "Image pushed: ${IMAGE_TAG}"
}

# ============================================================================
# Next steps
# ============================================================================

print_next_steps() {
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
    prepare_cuda_containers
    apply_patches
    run_build
    transfer_image_from_remote
    run_push
    print_next_steps
}

main "$@"
