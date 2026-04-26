#!/usr/bin/env bash
#
# build_comfyui_image.sh — Build xomoxcc/comfyui:sm121.
#
# Builds a ComfyUI image pre-baked for DGX Spark (GB10 / SM_121 / ARM64) on
# top of scitrera/dgx-spark-pytorch-dev. The image ships a frozen ComfyUI
# checkout + all pip deps + xformers + SageAttention v2 compiled for SM_121
# — no git clone / pip install at container start.
#
# Structure mirrors build_sm121_image.sh:
#   - entire script runs on the x86 control host
#   - spark4 (arm64) is a dumb remote podman build runner
#   - after build, a throwaway registry:2 is started on the build host
#     (bound to its QSFP IP). All downstream transfers go through it:
#       * k3s containerd pull on all 4 sparks in parallel (unless
#         --no-k3s-distribute) — QSFP at 200 GbE, loopback on the build
#         host itself to avoid a NIC bounce. Gives consistent sha256 on
#         every node.
#       * podman pull on this control host via the build host's LAN IP
#         (unless --no-local-copy) — hyperion is not on the QSFP mesh.
#   - push to Docker Hub from x86 using the x86 host's credentials
#     (unless --no-push or --no-local-copy)
#   - registry container is torn down via EXIT trap (also on failure)
#
# Build context lives in scripts/comfyui/:
#   - Dockerfile
#   - entrypoint.sh
#   - requirements-extra.txt
#
# Prerequisites on the x86 control host
# --------------------------------------
# - podman (`apt install podman`)
# - Unencrypted SSH key for podman (podman's Go SSH client does not use
#   ssh-agent or encrypted keys):
#     ssh-keygen -t ed25519 -f ~/.ssh/id_podman -N ""
#     ssh-copy-id -i ~/.ssh/id_podman root@spark4
#   Override path via BUILD_COMFYUI_SSH_IDENTITY.
# - `podman login docker.io -u xomoxcc` already done on this host.
# - ~25 GB free disk for the image after scp.
#
# Prerequisites on spark4 (the build host)
# ----------------------------------------
# - podman (`apt install podman`)
# - podman.socket enabled as root: `systemctl enable --now podman.socket`
# - ~60 GB free disk (base + intermediate layers + final image).
# - NO credentials, NO local scripts, NO state between runs (except the
#   local podman image store + layer cache, which accelerate rebuilds).
#

set -euo pipefail

# ============================================================================
# Configuration
# ============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONTEXT_DIR="${SCRIPT_DIR}/comfyui"

IMAGE_TAG="${BUILD_COMFYUI_IMAGE_TAG:-xomoxcc/comfyui:sm121}"
IMAGE_TAG_DATED="${IMAGE_TAG}-$(date +%Y%m%d)"

# Upstream ComfyUI ref to bake into the image. `master` = latest green;
# pin a specific commit SHA here for reproducible builds.
COMFYUI_REF="${BUILD_COMFYUI_REF:-master}"

# Base image aliases. --base <value> or BUILD_COMFYUI_BASE_IMAGE env var
# override these.
#
# The 'nvidia' base (NGC PyTorch) is the default since 2026-04-26: both
# scitrera-pipeline images on sm121 ship a torch wheel whose SDPA
# EFFICIENT_ATTENTION backend silently returns numerically corrupt output
# on Blackwell GB10 (12-27× off from a CPU reference, no NaN/Inf — see
# UPSTREAM_PYTORCH_SDPA_SM121.md for the reproducer and cross-validation
# matrix). The defect is byte-identical between scitrera-2.10/cu131 and
# the xomoxcc-rebuild-of-scitrera-2.11/cu132, so it lives in the
# scitrera/cuda-containers build pipeline rather than in any specific
# torch/CUDA version. NGC's own builds are correct on the same hardware.
#
# The 'scitrera' and 'xomoxcc' aliases are kept for diagnostic use (e.g.
# re-running the SDPA reproducer to confirm the bug is still present)
# and as fallbacks if NGC's image goes away. They do NOT produce a
# correct ComfyUI image on sm121 unless paired with the §4c
# sitecustomize SDPA-MATH workaround in
# roles/k8s_dgx/templates/comfyui_launch.sh.j2.

# NGC PyTorch tag mapping verified 2026-04-26 (probed each tag for
# torch.__version__ + torch.version.cuda; ran the SDPA reproducer on
# sm121 against a CPU reference where applicable):
#
#   25.10-py3 → torch 2.9.0a0+nv25.10  / cu13.0   (too old, untested SDPA)
#   25.12-py3 → torch 2.10.0a0+nv25.12 / cu13.1   ← current default
#                                                   (last 2.10/cu131 tag,
#                                                    SDPA verified correct)
#   26.01-py3 → torch 2.10.0a0+nv26.01 / cu13.1   (still 2.10, fresher build)
#   26.02-py3 → torch 2.11.0a0+nv26.02 / cu13.1   (first 2.11 tag, SDPA verified correct)
#   26.03-py3 → torch 2.11.0a0+nv26.03 / cu13.2   (latest release as of 2026-04-26,
#                                                   SDPA verified correct,
#                                                   newest CUDA major available)
#   26.04-py3 → not yet released
#
# Default is 26.03 (latest torch 2.11 + cu13.2, SDPA verified correct).
# Uncomment one of the alternatives below — or pass --base <image-tag> at
# runtime — for a deliberate downgrade (e.g. to stay on torch 2.10 for
# xformers v0.0.32 source-pin compatibility, until that pin is bumped).
# BASE_NVIDIA_IMAGE="nvcr.io/nvidia/pytorch:25.12-py3"  # torch 2.10 / cu13.1, last 2.10/cu131 tag
# BASE_NVIDIA_IMAGE="nvcr.io/nvidia/pytorch:26.01-py3"  # torch 2.10 / cu13.1, fresher 2.10 build
# BASE_NVIDIA_IMAGE="nvcr.io/nvidia/pytorch:26.02-py3"  # torch 2.11 / cu13.1, first 2.11 release
BASE_NVIDIA_IMAGE="nvcr.io/nvidia/pytorch:26.03-py3"  # torch 2.11 / cu13.2, latest available
BASE_SCITRERA_IMAGE="scitrera/dgx-spark-pytorch-dev:2.10.0-v2-cu131"
BASE_XOMOXCC_IMAGE="xomoxcc/dgx-spark-pytorch-dev:2.11.0-v1-cu132"

BASE_IMAGE_ALIAS=""
BASE_IMAGE_OVERRIDE="${BUILD_COMFYUI_BASE_IMAGE:-}"
EFFECTIVE_BASE_IMAGE=""
BASE_IMAGE_SOURCE=""

# Remote build host + podman connection. Same pattern as build_sm121_image.sh.
REMOTE_HOST="${BUILD_COMFYUI_REMOTE_HOST:-root@spark4.local}"
PODMAN_CONNECTION="${BUILD_COMFYUI_PODMAN_CONNECTION:-}"
PODMAN_SSH_IDENTITY="${BUILD_COMFYUI_SSH_IDENTITY:-${HOME}/.ssh/id_podman}"

# Parallel compile jobs on the build host. GB10 safe ceiling is 8 (16 OOM-
# kills CUTLASS template expansion — see feedback_build_jobs_gb10). Honored
# via the MAX_JOBS env baked into the Dockerfile plus --build-arg overrides.
BUILD_JOBS="${BUILD_COMFYUI_BUILD_JOBS:-4}"

# Optional kernel builds. Default ON. xformers+sage roughly double the
# end-to-end build time; torchaudio adds ~5-10 min. Turn any off for quick
# iterations on the ComfyUI layer itself.
BUILD_XFORMERS=1
BUILD_SAGE_ATTN=1
# torchaudio is required for ComfyUI to import (audio_vae chain via comfy/sd.py).
# NGC PyTorch base images dropped torchaudio in their aarch64 wheels, so we
# build it from source against the NGC torch — same pattern as xformers/sage.
# Disable only if you're using a base image that already ships a matching
# torchaudio (e.g. a future NGC build that re-adds it, or a non-NGC base).
BUILD_TORCHAUDIO=1

# Triton install. SageAttention hard-depends on triton, xformers uses it
# for several backend paths. Default ON — disable only if you're sure the
# base image already provides triton (most aarch64 bases don't).
INSTALL_TRITON=1

PUSH_IMAGE=1
NO_LOCAL_COPY=0

# Registry-based distribution configuration. The throwaway registry runs on
# the build host (REMOTE_HOST), bound to REGISTRY_HOST:REGISTRY_PORT. QSFP
# IP is the default so all 4 sparks pull at 200 GbE; control host pulls
# over LAN (hyperion isn't on the QSFP mesh).
REGISTRY_HOST="${BUILD_COMFYUI_REGISTRY_HOST:-10.10.10.4}"
REGISTRY_PORT="${BUILD_COMFYUI_REGISTRY_PORT:-5000}"
REGISTRY_CONTAINER="tmp-distr-registry-comfyui"
REGISTRY_STARTED=0

# k3s containerd distribution — three modes:
#   "self"  (default) pull only into the build host's own k3s containerd
#           (via loopback). Needed so pods scheduled on this spark can
#           actually use the image; podman and containerd have separate
#           stores.
#   "all"   also pull into the other sparks' k3s containerd (parallel,
#           over QSFP). Enabled via --k3s-distribute.
#   "none"  skip k3s containerd entirely. Enabled via --no-k3s-distribute.
ALL_SPARK_TARGETS=(spark1.local spark2.local spark3.local spark4.local)
K3S_DISTRIBUTION_MODE="self"

SSH_OPTS=(-o BatchMode=yes -o StrictHostKeyChecking=accept-new)

# ============================================================================
# Helpers
# ============================================================================

log()  { printf '\n\033[1;34m=== %s ===\033[0m\n' "$*"; }
warn() { printf '\033[1;33mWARN: %s\033[0m\n' "$*" >&2; }
die()  { printf '\033[1;31mERROR: %s\033[0m\n' "$*" >&2; exit 1; }

usage() {
    cat <<EOF
Usage: $(basename "$0") [--base nvidia|scitrera|xomoxcc|<image>]
                        [--comfyui-ref REF]
                        [--remote-host user@host] [--podman-connection NAME]
                        [--no-xformers] [--no-sage-attn] [--no-torchaudio] [--no-triton]
                        [--no-local-copy] [--no-push]
                        [--k3s-distribute | --no-k3s-distribute]
                        [--registry-host IP] [--registry-port PORT]
                        [--help]

Builds ${IMAGE_TAG} on the remote build host via podman socket, copies
the result back to this host (unless --no-local-copy), and pushes it
from here (unless --no-push or --no-local-copy).

Options:
  --base VALUE  PyTorch dev base image this build sits on:
                  nvidia    ${BASE_NVIDIA_IMAGE}        (default; NGC PyTorch,
                                                         verified-correct SDPA on sm121)
                  scitrera  ${BASE_SCITRERA_IMAGE}  (BROKEN SDPA on sm121 —
                                                     diagnostic use only, see
                                                     UPSTREAM_PYTORCH_SDPA_SM121.md)
                  xomoxcc   ${BASE_XOMOXCC_IMAGE}   (custom 2.11/cu132, also
                                                     BROKEN — same scitrera
                                                     pipeline; diagnostic use only)
                  <image>   arbitrary image reference, passed verbatim.
  --comfyui-ref REF
                Git ref (branch, tag, commit SHA) of comfyanonymous/ComfyUI
                to freeze into the image. Default: ${COMFYUI_REF}
  --remote-host user@host
                Remote arm64 build host reachable via SSH + podman socket.
                Default: ${REMOTE_HOST}
  --podman-connection NAME
                Registered podman connection name (or created on demand). If
                omitted, derived from --remote-host (strip user@ and domain).
  --no-xformers   Skip compiling xformers from source (saves ~15 min).
  --no-sage-attn  Skip compiling SageAttention v2 from source (saves ~10 min).
  --no-torchaudio Skip compiling torchaudio from source. Only safe if the
                  base image already ships an ABI-compatible torchaudio —
                  NGC's aarch64 wheels do NOT, so leaving this enabled is
                  required for the default --base nvidia path.
  --no-triton     Skip 'pip install triton' (default: install). Only use if
                  the base image already ships triton — without it,
                  SageAttention will fail to import at runtime and xformers
                  loses its triton kernel paths.
  --no-local-copy Skip pulling the built image back to this host (via the
                  temporary registry on the build host). Implies --no-push
                  (push reads from the local podman store).
  --no-push       Skip 'podman push' to Docker Hub after build + local pull.
  --k3s-distribute
                  Also pull the image into the k3s containerd k8s.io
                  namespace on all 4 sparks in parallel (via the temporary
                  registry on the build host, over QSFP). Default: off —
                  by default only the build host's own k3s containerd is
                  populated (via loopback), since pods scheduled on this
                  spark need the image in containerd, not podman.
  --no-k3s-distribute
                  Skip ALL k3s containerd population, including the build
                  host itself. Image stays in the build host's podman
                  store (and, unless --no-local-copy, on this control
                  host + Docker Hub). Useful for pure Docker-Hub-only
                  builds.
  --registry-host IP
                  IP the temporary registry binds to on the build host. Must
                  be reachable by every target over the fast network. On this
                  cluster that's the QSFP mesh (10.10.10.0/24). Default:
                  ${REGISTRY_HOST}. Override when --remote-host is not spark4.
  --registry-port PORT
                  Port for the temporary registry. Default: ${REGISTRY_PORT}.
  --help          Show this help.

Environment overrides:
  BUILD_COMFYUI_IMAGE_TAG           Default: ${IMAGE_TAG}
  BUILD_COMFYUI_REF                 Default: ${COMFYUI_REF}
  BUILD_COMFYUI_BASE_IMAGE          Direct BASE override, wins over --base.
  BUILD_COMFYUI_REMOTE_HOST         Default: ${REMOTE_HOST}
  BUILD_COMFYUI_PODMAN_CONNECTION   Derived from --remote-host if unset.
  BUILD_COMFYUI_SSH_IDENTITY        Default: ${PODMAN_SSH_IDENTITY}
  BUILD_COMFYUI_BUILD_JOBS          Default: ${BUILD_JOBS}
  BUILD_COMFYUI_REGISTRY_HOST       Default: ${REGISTRY_HOST}
  BUILD_COMFYUI_REGISTRY_PORT       Default: ${REGISTRY_PORT}
EOF
}

# ============================================================================
# Argument parsing
# ============================================================================

while [[ $# -gt 0 ]]; do
    case "$1" in
        --base)
            shift
            [[ $# -gt 0 ]] || die "--base requires an argument (xomoxcc|scitrera|<image>)"
            BASE_IMAGE_ALIAS="$1"; shift ;;
        --base=*)
            BASE_IMAGE_ALIAS="${1#--base=}"; shift ;;
        --comfyui-ref)
            shift
            [[ $# -gt 0 ]] || die "--comfyui-ref requires a git ref"
            COMFYUI_REF="$1"; shift ;;
        --comfyui-ref=*)
            COMFYUI_REF="${1#--comfyui-ref=}"; shift ;;
        --remote-host)
            shift
            [[ $# -gt 0 ]] || die "--remote-host requires an argument (user@host)"
            REMOTE_HOST="$1"; shift ;;
        --remote-host=*)
            REMOTE_HOST="${1#--remote-host=}"; shift ;;
        --podman-connection)
            shift
            [[ $# -gt 0 ]] || die "--podman-connection requires an argument"
            PODMAN_CONNECTION="$1"; shift ;;
        --podman-connection=*)
            PODMAN_CONNECTION="${1#--podman-connection=}"; shift ;;
        --no-xformers)   BUILD_XFORMERS=0; shift ;;
        --no-sage-attn)  BUILD_SAGE_ATTN=0; shift ;;
        --no-torchaudio) BUILD_TORCHAUDIO=0; shift ;;
        --no-triton)     INSTALL_TRITON=0; shift ;;
        --no-local-copy) NO_LOCAL_COPY=1; PUSH_IMAGE=0; shift ;;
        --no-push)       PUSH_IMAGE=0; shift ;;
        --k3s-distribute)    K3S_DISTRIBUTION_MODE="all"; shift ;;
        --no-k3s-distribute) K3S_DISTRIBUTION_MODE="none"; shift ;;
        --registry-host)
            shift
            [[ $# -gt 0 ]] || die "--registry-host requires an argument"
            REGISTRY_HOST="$1"; shift ;;
        --registry-host=*)
            REGISTRY_HOST="${1#--registry-host=}"; shift ;;
        --registry-port)
            shift
            [[ $# -gt 0 ]] || die "--registry-port requires an argument"
            REGISTRY_PORT="$1"; shift ;;
        --registry-port=*)
            REGISTRY_PORT="${1#--registry-port=}"; shift ;;
        --help|-h)       usage; exit 0 ;;
        *)               die "Unknown argument: $1 (use --help)" ;;
    esac
done

if [[ -z "${PODMAN_CONNECTION}" ]]; then
    PODMAN_CONNECTION="${REMOTE_HOST##*@}"
    PODMAN_CONNECTION="${PODMAN_CONNECTION%%.*}"
fi

# Bare host portion of REMOTE_HOST, used for direct ssh and for the
# control-host's pull-back address (hyperion reaches the build host via
# LAN DNS, not via the QSFP mesh).
SSH_HOST="${REMOTE_HOST##*@}"

# ============================================================================
# Base image resolution
# ============================================================================

resolve_base_image() {
    if [[ -n "${EFFECTIVE_BASE_IMAGE}" ]]; then
        return 0
    fi
    if [[ -n "${BASE_IMAGE_OVERRIDE}" ]]; then
        EFFECTIVE_BASE_IMAGE="${BASE_IMAGE_OVERRIDE}"
        BASE_IMAGE_SOURCE="BUILD_COMFYUI_BASE_IMAGE env"
        return 0
    fi
    case "${BASE_IMAGE_ALIAS}" in
        nvidia)   EFFECTIVE_BASE_IMAGE="${BASE_NVIDIA_IMAGE}";   BASE_IMAGE_SOURCE="--base nvidia" ;;
        xomoxcc)  EFFECTIVE_BASE_IMAGE="${BASE_XOMOXCC_IMAGE}";  BASE_IMAGE_SOURCE="--base xomoxcc" ;;
        scitrera) EFFECTIVE_BASE_IMAGE="${BASE_SCITRERA_IMAGE}"; BASE_IMAGE_SOURCE="--base scitrera" ;;
        "")       EFFECTIVE_BASE_IMAGE="${BASE_NVIDIA_IMAGE}";   BASE_IMAGE_SOURCE="default (nvidia)" ;;
        *)        EFFECTIVE_BASE_IMAGE="${BASE_IMAGE_ALIAS}";    BASE_IMAGE_SOURCE="--base (verbatim)" ;;
    esac
}

# ============================================================================
# Preflight
# ============================================================================

preflight() {
    log "Preflight"

    for f in Dockerfile entrypoint.sh requirements-extra.txt; do
        [[ -f "${CONTEXT_DIR}/${f}" ]] || die "Missing build context file: ${CONTEXT_DIR}/${f}"
    done

    for tool in git podman; do
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

    # Registry orchestration uses direct ssh (agent-based, NOT the podman
    # unencrypted key), so verify reachability ahead of time. Needed when
    # anything downstream of the build pulls from the registry — "self"
    # and "all" both do; only "none" + --no-local-copy skip it entirely.
    if [[ "${K3S_DISTRIBUTION_MODE}" != "none" ]] || (( NO_LOCAL_COPY == 0 )); then
        if ! ssh "${SSH_OPTS[@]}" "${REMOTE_HOST}" true 2>/dev/null; then
            die "Direct ssh to ${REMOTE_HOST} failed. Needed for registry orchestration. Ensure ssh-agent has the right key or ~/.ssh/config resolves ${SSH_HOST}."
        fi
        if [[ "${K3S_DISTRIBUTION_MODE}" == "all" ]]; then
            local t
            for t in "${ALL_SPARK_TARGETS[@]}"; do
                if ! ssh "${SSH_OPTS[@]}" "root@${t}" true 2>/dev/null; then
                    die "Direct ssh root@${t} failed. Needed for --k3s-distribute. Drop the flag or switch to --no-k3s-distribute."
                fi
            done
        fi
    fi

    echo "Build context present, tools available, SSH identity found"
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
        die "Remote host is ${remote_arch}, expected arm64/aarch64 (scitrera base is arm64-only)"
    fi
    echo "Remote podman is reachable (arch=${remote_arch})"
}

# ============================================================================
# Verify base image is available on the build host
# ============================================================================

ensure_base_image_present() {
    resolve_base_image
    local base_image="${EFFECTIVE_BASE_IMAGE}"
    log "Verifying base image '${base_image}' on '${PODMAN_CONNECTION}' (from ${BASE_IMAGE_SOURCE})"

    # Resolve to a fully-qualified name. Images already prefixed with a
    # registry hostname (first slash-segment contains a '.' or ':' — e.g.
    # `nvcr.io/...` or `localhost:5000/...`) are used verbatim. Docker Hub
    # short names like `xomoxcc/foo:bar` get an explicit `docker.io/` prefix.
    # Without this guard, an NGC image like `nvcr.io/nvidia/pytorch:26.03-py3`
    # would be pulled as `docker.io/nvcr.io/nvidia/pytorch:26.03-py3` which
    # Docker Hub rejects with "requested access to the resource is denied".
    local first_segment="${base_image%%/*}"
    local fqn_base
    if [[ "${first_segment}" == *.* || "${first_segment}" == *:* ]]; then
        fqn_base="${base_image}"
    else
        fqn_base="docker.io/${base_image}"
    fi

    if podman --connection "${PODMAN_CONNECTION}" image exists "${fqn_base}" 2>/dev/null; then
        echo "Base image found as ${fqn_base}"
        return 0
    fi
    if podman --connection "${PODMAN_CONNECTION}" image exists "${base_image}" 2>/dev/null; then
        echo "Base image found as ${base_image}"
        return 0
    fi

    echo "Base image not found locally — pulling ${fqn_base}..."
    if podman --connection "${PODMAN_CONNECTION}" pull "${fqn_base}"; then
        echo "Base image pulled"
        return 0
    fi

    case "${base_image}" in
        xomoxcc/dgx-spark-pytorch-dev:*)
            die "xomoxcc base image '${base_image}' is not on ${PODMAN_CONNECTION} and not pullable — build it first via scripts/build_pytorch_base_image.sh, or switch to --base scitrera."
            ;;
        *)
            die "Base image '${base_image}' not found locally and pull from '${fqn_base}' failed."
            ;;
    esac
}

# ============================================================================
# Build via remote podman socket
# ============================================================================

run_build() {
    log "Running podman build on '${PODMAN_CONNECTION}' (45–75 minutes expected)"

    resolve_base_image

    # UTC-ISO-8601 timestamp, gestempelt zu Build-Start. Läuft als
    # --build-arg BUILDTIME in den Dockerfile-ARG (Default "unknown"),
    # landet als ENV BUILDTIME + OCI-Label im Image und wird vom
    # Entrypoint in den Container-Log geschrieben.
    local buildtime
    buildtime="$(date -u +%Y-%m-%dT%H:%M:%SZ)"

    echo "Build parameters:"
    echo "  IMAGE_TAG          = ${IMAGE_TAG}"
    echo "  IMAGE_TAG (dated)  = ${IMAGE_TAG_DATED}"
    echo "  BASE_IMAGE         = ${EFFECTIVE_BASE_IMAGE}  [${BASE_IMAGE_SOURCE}]"
    echo "  COMFYUI_REF        = ${COMFYUI_REF}"
    echo "  BUILD_JOBS         = ${BUILD_JOBS}"
    echo "  BUILD_XFORMERS     = ${BUILD_XFORMERS}"
    echo "  BUILD_SAGE_ATTN    = ${BUILD_SAGE_ATTN}"
    echo "  BUILD_TORCHAUDIO   = ${BUILD_TORCHAUDIO}"
    echo "  INSTALL_TRITON     = ${INSTALL_TRITON}"
    echo "  BUILDTIME          = ${buildtime}"

    # The build context is scripts/comfyui/. Podman streams it to the remote
    # host over the socket; the build runs natively on arm64 (no QEMU) and
    # the result lands in the remote host's local image store.
    podman --connection "${PODMAN_CONNECTION}" build \
        -f "${CONTEXT_DIR}/Dockerfile" \
        --build-arg "BASE=${EFFECTIVE_BASE_IMAGE}" \
        --build-arg "COMFYUI_REF=${COMFYUI_REF}" \
        --build-arg "BUILD_XFORMERS=${BUILD_XFORMERS}" \
        --build-arg "BUILD_SAGE_ATTN=${BUILD_SAGE_ATTN}" \
        --build-arg "BUILD_TORCHAUDIO=${BUILD_TORCHAUDIO}" \
        --build-arg "INSTALL_TRITON=${INSTALL_TRITON}" \
        --build-arg "MAX_JOBS=${BUILD_JOBS}" \
        --build-arg "BUILDTIME=${buildtime}" \
        -t "${IMAGE_TAG}" \
        -t "docker.io/${IMAGE_TAG}" \
        -t "docker.io/${IMAGE_TAG_DATED}" \
        "${CONTEXT_DIR}"

    if ! podman --connection "${PODMAN_CONNECTION}" image exists "docker.io/${IMAGE_TAG}"; then
        die "Build finished but docker.io/${IMAGE_TAG} is not in the remote image store — check podman output above"
    fi
    echo "Remote build complete: ${IMAGE_TAG} (also tagged as docker.io/${IMAGE_TAG} and ${IMAGE_TAG_DATED})"
}

# ============================================================================
# Registry-based distribution (build-host podman store → k3s containerd +
# optionally → control-host podman store)
# ============================================================================

# Throwaway registry:2 runs on the build host, host-networked so it's
# reachable via both the QSFP IP (REGISTRY_HOST, for the 4 sparks) and
# the build host's LAN IP (SSH_HOST, for this control host). Plain HTTP,
# LAN-only, short-lived — torn down by the EXIT trap. Replaces the old
# serial `podman save | podman load` pipe: pulls run in parallel, layer
# transfers are concurrent, and the sha256 digest is identical on every
# target (containerd pulls are the digest-preserving code path).

ssh_build_host() {
    ssh "${SSH_OPTS[@]}" "${REMOTE_HOST}" "$@"
}

cleanup_temp_registry() {
    if (( REGISTRY_STARTED == 1 )); then
        echo "=== cleanup: stopping temporary registry on ${REMOTE_HOST} ==="
        ssh_build_host "podman rm -f ${REGISTRY_CONTAINER} >/dev/null 2>&1 || true" || true
        REGISTRY_STARTED=0
    fi
}
trap cleanup_temp_registry EXIT

start_temp_registry() {
    log "Starting temporary registry:2 on ${REMOTE_HOST} (binds 0.0.0.0:${REGISTRY_PORT}; targets pull via ${REGISTRY_HOST})"

    ssh_build_host "podman rm -f ${REGISTRY_CONTAINER} >/dev/null 2>&1 || true; \
                    podman run -d --name ${REGISTRY_CONTAINER} --network host \
                        -e REGISTRY_HTTP_ADDR=0.0.0.0:${REGISTRY_PORT} \
                        docker.io/library/registry:2 >/dev/null" \
        || die "Failed to start temporary registry on ${REMOTE_HOST}"
    REGISTRY_STARTED=1

    local i
    for i in {1..15}; do
        if ssh_build_host "curl -sf http://${REGISTRY_HOST}:${REGISTRY_PORT}/v2/ >/dev/null"; then
            echo "registry ready on ${REGISTRY_HOST}:${REGISTRY_PORT}"
            return 0
        fi
        sleep 1
    done
    die "Temporary registry did not become ready in time"
}

push_to_temp_registry() {
    log "Pushing docker.io/${IMAGE_TAG} from ${REMOTE_HOST} podman store into temp registry"
    # Podman pushes layers in parallel. Transfer stays host-local on the
    # build host (loopback-over-host-net), so no NIC round trip.
    ssh_build_host "podman push --tls-verify=false \
        'docker.io/${IMAGE_TAG}' \
        '${REGISTRY_HOST}:${REGISTRY_PORT}/${IMAGE_TAG}'" \
        || die "Failed to push ${IMAGE_TAG} to temporary registry"
}

resolve_k3s_targets() {
    # Echoes the target hostnames one per line based on K3S_DISTRIBUTION_MODE.
    # "self" → just the build host; "all" → every spark; "none" → nothing.
    case "${K3S_DISTRIBUTION_MODE}" in
        self) echo "${SSH_HOST}" ;;
        all)  printf '%s\n' "${ALL_SPARK_TARGETS[@]}" ;;
        none) ;;
        *)    die "Unknown K3S_DISTRIBUTION_MODE: ${K3S_DISTRIBUTION_MODE}" ;;
    esac
}

distribute_to_k3s_targets() {
    local targets=()
    mapfile -t targets < <(resolve_k3s_targets)
    if (( ${#targets[@]} == 0 )); then
        return 0
    fi

    log "Pulling ${IMAGE_TAG} from temp registry into k3s containerd on: ${targets[*]}"

    local build_host_short="${SSH_HOST%%.*}"
    local pids=() host short src_ref
    for host in "${targets[@]}"; do
        short="${host%%.*}"
        # Build host pulls via loopback — avoids a NIC bounce back into
        # its own QSFP interface and the associated TCP overhead.
        if [[ "${short}" == "${build_host_short}" ]]; then
            src_ref="127.0.0.1:${REGISTRY_PORT}/${IMAGE_TAG}"
        else
            src_ref="${REGISTRY_HOST}:${REGISTRY_PORT}/${IMAGE_TAG}"
        fi
        (
            ssh "${SSH_OPTS[@]}" "root@${host}" \
                "set -e; \
                 k3s ctr -n k8s.io image pull --plain-http '${src_ref}'; \
                 k3s ctr -n k8s.io image rm 'docker.io/${IMAGE_TAG}' >/dev/null 2>&1 || true; \
                 k3s ctr -n k8s.io image tag '${src_ref}' 'docker.io/${IMAGE_TAG}'; \
                 k3s ctr -n k8s.io image rm '${src_ref}' >/dev/null" \
                2>&1 | sed "s/^/[${short}] /"
        ) &
        pids+=($!)
    done

    local fail=0 pid
    for pid in "${pids[@]}"; do
        wait "${pid}" || fail=1
    done
    if (( fail != 0 )); then
        die "k3s distribution failed on one or more targets (see [hostname] prefixed output above)"
    fi
    echo "k3s distribution complete on ${#targets[@]} target(s)"
}

pull_via_registry_to_local() {
    # Control host is not on the QSFP mesh, so pull via the build host's
    # LAN name (SSH_HOST). Temp reference is retagged to docker.io/...
    # so the subsequent `podman push` to Docker Hub finds the image under
    # its canonical name.
    log "Pulling ${IMAGE_TAG} into local podman store via ${SSH_HOST}:${REGISTRY_PORT}"

    local local_ref="${SSH_HOST}:${REGISTRY_PORT}/${IMAGE_TAG}"

    podman image rm "docker.io/${IMAGE_TAG}" >/dev/null 2>&1 || true
    podman image rm "${IMAGE_TAG}" >/dev/null 2>&1 || true
    podman image rm "localhost/${IMAGE_TAG}" >/dev/null 2>&1 || true

    podman pull --tls-verify=false "${local_ref}" \
        || die "Failed to pull ${local_ref} into local podman store"

    podman tag "${local_ref}" "docker.io/${IMAGE_TAG}"
    podman rmi "${local_ref}" >/dev/null

    podman image inspect "docker.io/${IMAGE_TAG}" >/dev/null \
        || die "Image not present locally after registry pull"
    echo "Image pulled locally: docker.io/${IMAGE_TAG}"
}

# ============================================================================
# Push
# ============================================================================

run_push() {
    if (( PUSH_IMAGE == 0 )); then
        log "Skipping push (--no-push)"
        return
    fi

    log "Pushing docker.io/${IMAGE_TAG} to Docker Hub"

    local auth_file="${REGISTRY_AUTH_FILE:-${XDG_RUNTIME_DIR:-/run}/containers/auth.json}"
    [[ -f "${auth_file}" ]] || auth_file="${HOME}/.docker/config.json"
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
    local distr_line
    case "${K3S_DISTRIBUTION_MODE}" in
        self) distr_line="k3s containerd (k8s.io namespace): ${SSH_HOST} only (default)" ;;
        all)  distr_line="k3s containerd (k8s.io namespace): ${ALL_SPARK_TARGETS[*]}" ;;
        none) distr_line="k3s containerd: SKIPPED (--no-k3s-distribute)" ;;
    esac

    if (( NO_LOCAL_COPY == 1 )); then
        cat <<EOF

$(log "Remote-only build complete")

Image: ${IMAGE_TAG}
Location: docker.io/${IMAGE_TAG} in ${PODMAN_CONNECTION}'s podman store
          ${distr_line}
          (NOT on this control host, NOT pushed to Docker Hub)
EOF
        return
    fi

    cat <<EOF

$(log "Build + distribute + push complete")

Image: ${IMAGE_TAG}
       ${distr_line}

Next steps:

1. Bump comfyui_image in roles/k8s_dgx/defaults/main.yml:
     comfyui_image: "${IMAGE_TAG}"

2. Simplify roles/k8s_dgx/tasks/comfyui.yml: the git-clone + pip-install
   block in the launch ConfigMap is now obsolete (both are baked into
   the image). The container's ENTRYPOINT starts ComfyUI directly; you
   can drop the \`command\` override + launch-script ConfigMap entirely.

3. Deploy (only after explicit user approval):
     ansible-playbook k8s_dgx.yml --tags comfyui -e comfyui_enabled=true

4. Verify:
     kubectl --context=ht@dgxarley -n comfyui logs -f deploy/comfyui
   Expected in the log:
     [entrypoint] torch: 2.10.x cuda 13.x cap (12, 1)
EOF
}

# ============================================================================
# Main
# ============================================================================

main() {
    preflight
    ensure_podman_connection
    ensure_base_image_present
    run_build

    # Registry is only needed if somebody downstream is going to pull
    # from it: k3s distribution ("self" or "all") OR a local-host pull.
    # --no-k3s-distribute + --no-local-copy → skip entirely and leave the
    # image sitting in the build host's podman store.
    if [[ "${K3S_DISTRIBUTION_MODE}" != "none" ]] || (( NO_LOCAL_COPY == 0 )); then
        start_temp_registry
        push_to_temp_registry
        if [[ "${K3S_DISTRIBUTION_MODE}" != "none" ]]; then
            distribute_to_k3s_targets
        fi
        if (( NO_LOCAL_COPY == 0 )); then
            pull_via_registry_to_local
            run_push
        fi
        cleanup_temp_registry
    else
        log "Skipping distribution + local copy + push — image stays in ${PODMAN_CONNECTION} podman store only"
    fi

    print_next_steps
}

main "$@"
