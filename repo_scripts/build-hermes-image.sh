#!/bin/bash
set -euo pipefail

# Build & push xomoxcc/hermes-agent-tools:<tag> (scripts/hermes/Dockerfile), multiarch.
#
# build-container-multiarch.sh (kikube) re-pointed at the hermes tools image: only the configuration block differs,
# everything from BUILD_SCRIPT_VERSION downward is verbatim.
#
# Usage:
#   repo_scripts/build-hermes-image.sh            # build + push (amd64 + arm64)
#   repo_scripts/build-hermes-image.sh login      # docker/registry login only
#   repo_scripts/build-hermes-image.sh check      # validate remote arm64 connection
#   repo_scripts/build-hermes-image.sh onlylocal  # native-arch local build, no push
#
# Tag = hermes.image_tag from roles/k8s_infra/defaults/main/hermes.yml (override: HERMES_TAG=v0.x.y), used as both the
# nousresearch/hermes-agent base tag and the derived tag. arm64 via REMOTE_ARM64_CONNECTION from include.local.sh, else
# the podman VM.

#=============================================================================
# CONFIGURATION
#=============================================================================
readonly SCRIPT_DIR="$(dirname "$(realpath "$0")")"
readonly INCLUDE_SH="include.sh"
readonly PODMAN_VM_INIT_DISK_SIZE=100
HERMES_TAG="${HERMES_TAG:-$(sed -n 's/^  image_tag: "\([^"]*\)".*/\1/p' "${SCRIPT_DIR}/../roles/k8s_infra/defaults/main/hermes.yml" | head -1)}"
[[ -n "${HERMES_TAG}" ]] || { echo "ERROR: cannot read hermes.image_tag, set HERMES_TAG" >&2; exit 1; }
readonly HERMES_TAG
readonly DOCKER_IMAGE="${HERMES_IMAGE_REPO:-docker.io/xomoxcc/hermes-agent-tools}:${HERMES_TAG}"
readonly DOCKER_IMAGE_LATEST="${DOCKER_IMAGE%:*}:latest"

readonly PLATFORMS=("linux/amd64" "linux/arm64")
readonly DOCKERFILE=../scripts/hermes/Dockerfile
readonly DOCKER_BUILD_CONTEXT=$(dirname "$(realpath --relative-to="${SCRIPT_DIR}" "${SCRIPT_DIR}/${DOCKERFILE}")")
REMOTE_ARM64_CONNECTION="${REMOTE_ARM64_CONNECTION:-}"  # e.g. "user@rock5b"; made readonly after sourcing include.sh
# Podman's Go SSH client cannot use the SSH agent or encrypted keys.
# Use a dedicated unencrypted key (generate with: ssh-keygen -t ed25519 -f ~/.ssh/id_podman -N "")
REMOTE_ARM64_SSH_IDENTITY="${REMOTE_ARM64_SSH_IDENTITY:-}"
readonly BUILDER_NAME=mbuilder
readonly ENABLE_PARALLEL_BUILDS=1
readonly BUILDTIME="$(date +'%Y-%m-%d %H:%M:%S %Z')"

readonly BUILD_BASE_ARGS=(
  "-f" "${DOCKERFILE}"
  "--build-arg" "buildtime=${BUILDTIME}"
  "--build-arg" "BASE_TAG=${HERMES_TAG}"
  )

# podman machine ssh df -h
# podman --connection podman-machine-default system prune -a
# podman --connection podman-machine-default system df
# podman machine inspect podman-machine-default | grep -i disk


readonly BUILD_SCRIPT_VERSION="2026-06-12_11:11:11"

# Runtime state
PODMAN_VM_STARTED=0
DOCKER_IS_PODMAN=0
DOCKER_TOKEN_LOGIN=0
TOKEN_LOGIN_CONFIG_DIR=""

#=============================================================================
# HELPER FUNCTIONS
#=============================================================================
die() {
  echo "ERROR: $*" >&2
  exit 1
}

log() {
  echo "==> $*"
}

format_duration() {
  local seconds="$1"
  local mins=$(( seconds / 60 ))
  local secs=$(( seconds % 60 ))
  if (( mins > 0 )); then
    printf '%dm %ds' "${mins}" "${secs}"
  else
    printf '%ds' "${secs}"
  fi
}

is_podman() {
  # Note: Don't use grep -q with pipefail, causes SIGPIPE (exit 141)
  docker --version 2>&1 | grep -i podman >/dev/null
}

#=============================================================================
# SETUP FUNCTIONS
#=============================================================================
setup_environment() {
  cd "${SCRIPT_DIR}" || die "Could not change to script directory"

  if [ -e "${INCLUDE_SH}" ] ; then
    source "${INCLUDE_SH}"
  fi
  # include.sh / include.local.sh own the cred logic: after sourcing,
  # DOCKER_TOKEN / DOCKER_TOKENUSER hold whatever those files decided (they may
  # read ENV, hardcode, override — their choice). We just act on the result.

  # Docker config precedence:
  #   1. repo-local  docker-config/   (pre-baked auth) -> use as-is, no login
  #   2.            ../docker-config/  (pre-baked auth) -> use as-is, no login
  #   3. DOCKER_TOKEN + DOCKER_TOKENUSER set AND no local docker-config -> log in
  #      to a throwaway config dir (so a stale ~/.docker can't silently
  #      authenticate the push as the wrong user). Cleaned up on exit.
  #   4. ~/.docker fallback (existing behavior)
  DOCKER_TOKEN_LOGIN=0
  local local_cfg
  if local_cfg="$(realpath -e docker-config 2>/dev/null)" \
     || local_cfg="$(realpath -e ../docker-config 2>/dev/null)"; then
    DOCKER_CONFIG="${local_cfg}"
  elif [[ -n "${DOCKER_TOKEN:-}" && -n "${DOCKER_TOKENUSER:-}" ]]; then
    DOCKER_CONFIG="$(mktemp -d)"
    TOKEN_LOGIN_CONFIG_DIR="${DOCKER_CONFIG}"
    DOCKER_TOKEN_LOGIN=1
  else
    DOCKER_CONFIG="${HOME}/.docker"
  fi

  export DOCKER_CONFIG
  export REGISTRY_AUTH_FILE="${DOCKER_CONFIG}/config.json"

  # Lock REMOTE_ARM64_CONNECTION after include.sh (and include.local.sh) had a chance to set it
  readonly REMOTE_ARM64_CONNECTION

  if is_podman; then
    DOCKER_IS_PODMAN=1
  fi
}

ensure_docker_login() {
  if (( DOCKER_TOKEN_LOGIN == 1 )); then
    # Creds set (by include.*/ENV) and no local docker-config: log in explicitly
    # so that token (not a leftover ~/.docker) authenticates the push.
    log "Logging in to Docker registry as '${DOCKER_TOKENUSER}' (token login, isolated config)..."
    echo "${DOCKER_TOKEN}" | docker login --username "${DOCKER_TOKENUSER}" --password-stdin \
      || die "Docker login failed"
  elif [[ ! -e "${REGISTRY_AUTH_FILE}" ]]; then
    if [[ -n "${DOCKER_TOKEN:-}" && -n "${DOCKER_TOKENUSER:-}" ]]; then
      log "No docker config at ${REGISTRY_AUTH_FILE} — logging in as '${DOCKER_TOKENUSER}'..."
      echo "${DOCKER_TOKEN}" | docker login --username "${DOCKER_TOKENUSER}" --password-stdin \
        || die "Docker login failed"
    else
      die "No docker config at ${REGISTRY_AUTH_FILE} and no DOCKER_TOKEN/DOCKER_TOKENUSER set to log in with"
    fi
  else
    log "Using existing docker config: ${REGISTRY_AUTH_FILE}"
  fi
}

setup_docker_buildx() {
  if ! docker buildx inspect "${BUILDER_NAME}" --bootstrap >/dev/null 2>&1; then
    log "Setting up Docker buildx builder..."
    docker run --privileged --rm tonistiigi/binfmt --install all
    docker buildx create --name "${BUILDER_NAME}"
    docker buildx use "${BUILDER_NAME}"
  fi
}

#=============================================================================
# PODMAN VM FUNCTIONS
#=============================================================================
ensure_podman_vm_running() {
  if ! podman machine list | grep -q "podman-machine-default"; then
    log "Initializing podman machine..."
    podman machine init --disk-size ${PODMAN_VM_INIT_DISK_SIZE}
  fi

  if ! podman machine list --format "{{.Running}}" | grep -q "true"; then
    log "Starting podman machine..."
    podman machine start || die "Could not start podman machine (VM)"
    PODMAN_VM_STARTED=1
  fi
}

stop_podman_vm_if_started() {
  if (( PODMAN_VM_STARTED == 1 )); then
    log "Stopping podman machine..."
    podman machine stop
  fi
}

platform_needs_vm() {
  local platform="$1"
  ! podman run --rm --platform "${platform}" alpine uname -m >/dev/null 2>&1
}

ensure_remote_arm64_connection() {
  local user_host="${REMOTE_ARM64_CONNECTION}"
  # Podman's Go SSH client cannot use the SSH agent or encrypted keys.
  # Use a dedicated unencrypted key (generate with: ssh-keygen -t ed25519 -f ~/.ssh/id_podman -N "")
  local ssh_identity="${REMOTE_ARM64_SSH_IDENTITY}"

  if [[ ! -f "${ssh_identity}" ]]; then
    die "SSH identity '${ssh_identity}' not found. Create it with: ssh-keygen -t ed25519 -f ${ssh_identity} -N \"\" && ssh-copy-id -i ${ssh_identity} ${user_host}"
  fi

  # Check if connection already registered
  if podman system connection list --format "{{.Name}}" | grep -qxF "${user_host}"; then
    log "Podman connection '${user_host}' already registered"
  else
    log "Registering podman connection '${user_host}'..."
    local remote_uid
    remote_uid="$(ssh "${user_host}" id -u)" \
      || die "SSH to '${user_host}' failed — check SSH key auth"
    local sock_path
    if [[ "${remote_uid}" == "0" ]]; then
      sock_path="/run/podman/podman.sock"
    else
      sock_path="/run/user/${remote_uid}/podman/podman.sock"
    fi

    podman system connection add "${user_host}" "ssh://${user_host}${sock_path}" \
      --identity "${ssh_identity}" \
      || die "Failed to register podman connection '${user_host}'"
  fi

  # Validate connection is viable
  log "Validating remote connection '${user_host}'..."
  podman --connection "${user_host}" info >/dev/null 2>&1 \
    || die "Podman connection '${user_host}' is not responding — ensure podman.socket is enabled on the remote host"
}

copy_image_from_remote() {
  local connection="$1"
  local image="$2"
  log "Copying image from ${connection} to host: ${image}"

  # Transfer via `podman image save | podman image load` instead of
  # `podman image scp`. scp shells out to the podman binary over SSH on the
  # remote and is brittle (needs a matching remote podman in $PATH, no
  # progress, silent failures); a streamed save->load only needs the podman
  # socket connection we already validated.
  #
  # Drop any stale local copy first so the pipeline doesn't keep dangling
  # layers around. `podman load` normalizes a short-name RepoTag to
  # localhost/..., so clean all three possible forms.
  podman image rm "${image}" 2>/dev/null || true
  podman image rm "docker.io/${image}" 2>/dev/null || true
  podman image rm "localhost/${image}" 2>/dev/null || true

  # Image size (from the remote store) feeds pv's -s so it can show percent + ETA.
  local size
  size=$(podman --connection "${connection}" image inspect \
          --format '{{.Size}}' "${image}" 2>/dev/null || echo "")
  if [[ -n "${size}" ]] && command -v numfmt >/dev/null 2>&1; then
    echo "Transfer target: $(numfmt --to=iec --suffix=B "${size}") (${size} bytes)"
  elif [[ -n "${size}" ]]; then
    echo "Transfer target: ${size} bytes"
  fi

  if command -v pv >/dev/null 2>&1; then
    local -a pv_args=(-ptebar)
    [[ -n "${size}" ]] && pv_args+=(-s "${size}")
    set -o pipefail
    podman --connection "${connection}" image save --format oci-archive "${image}" \
      | pv "${pv_args[@]}" \
      | podman image load \
      || die "streamed image transfer of ${image} failed"
  else
    set -o pipefail
    podman --connection "${connection}" image save --format oci-archive "${image}" \
      | podman image load \
      || die "streamed image transfer of ${image} failed"
  fi

  # `podman load` strips the registry component and re-normalizes a short name
  # to localhost/...; retag so the reference the manifest step uses resolves.
  if podman image exists "localhost/${image}"; then
    podman tag "localhost/${image}" "${image}"
  fi

  podman image inspect "${image}" >/dev/null \
    || die "Image ${image} not present locally after transfer"
}

#=============================================================================
# BUILD FUNCTIONS
#=============================================================================
build_with_docker() {
  log "Building with Docker buildx (multi-arch)..."

  setup_docker_buildx

  printf -v platforms_csv '%s,' "${PLATFORMS[@]}"
  platforms_csv="${platforms_csv%,}"

  local -a build_args=("${BUILD_BASE_ARGS[@]}")
  if [[ -n "${DOCKER_IMAGE_LATEST}" && "${DOCKER_IMAGE}" != *:latest ]]; then
    build_args+=("-t" "${DOCKER_IMAGE_LATEST}")
  fi

  build_args+=("-t" "${DOCKER_IMAGE}")

  docker buildx build \
    "${build_args[@]}" \
    --platform "${platforms_csv}" \
    --push \
    "${DOCKER_BUILD_CONTEXT}"
}

build_with_podman() {
  log "Building with Podman manifest workflow..."

  # Remove existing manifest/image if it exists
  echo podman manifest rm "${DOCKER_IMAGE}"
  podman manifest rm "${DOCKER_IMAGE}" 2>/dev/null || true
  echo podman image rm "${DOCKER_IMAGE}"
  podman image rm "${DOCKER_IMAGE}" 2>/dev/null || true

  if [[ -n "${DOCKER_IMAGE_LATEST}" && "${DOCKER_IMAGE}" != *:latest ]]; then
    echo podman manifest rm "${DOCKER_IMAGE_LATEST}"
    podman manifest rm "${DOCKER_IMAGE_LATEST}" 2>/dev/null || true
    echo podman image rm "${DOCKER_IMAGE_LATEST}"
    podman image rm "${DOCKER_IMAGE_LATEST}" 2>/dev/null || true
  fi

  # Track platform-specific data
  local -a platform_tags=()
  local -A platform_connect_args=()
  local -a build_pids=()
  local -A build_durations=()
  local builds_start=$SECONDS
  local timing_dir
  timing_dir="$(mktemp -d)"

  # Ensure remote arm64 connection is set up and viable
  if [[ -n "${REMOTE_ARM64_CONNECTION}" ]]; then
    ensure_remote_arm64_connection
  fi

  # Add latest tag if not already latest
  local -a build_args=("${BUILD_BASE_ARGS[@]}")

  # Build for each platform (in parallel)
  for platform in "${PLATFORMS[@]}"; do
    local arch="${platform#*/}"
    local platform_tag="${DOCKER_IMAGE}-${arch}"
    local connect_arg=""

    # Check if platform needs remote builder or VM
    if [[ "${arch}" == "arm64" && -n "${REMOTE_ARM64_CONNECTION}" ]]; then
      log "Platform ${platform} delegated to remote: ${REMOTE_ARM64_CONNECTION}"
      connect_arg="--connection ${REMOTE_ARM64_CONNECTION}"
    elif platform_needs_vm "${platform}"; then
      log "Platform ${platform} needs VM for emulation"
      ensure_podman_vm_running
      connect_arg="--connection podman-machine-default"
    fi

    platform_tags+=("${platform_tag}")
    platform_connect_args["${platform}"]="${connect_arg}"

    local label="[${arch}]"
    if [[ -n "${connect_arg}" ]]; then
      label="[${arch}/remote]"
    fi

    log "Building for ${platform} -> ${platform_tag}..."
    log "ENABLE_PARALLEL_BUILDS: ${ENABLE_PARALLEL_BUILDS}"
    # shellcheck disable=SC2086
    if (( ${ENABLE_PARALLEL_BUILDS:-0} == 1 )) ; then
      log "PARALLEL_BUILDS ENABLED"

      echo "(podman ${connect_arg} build \"${build_args[@]}\" --platform \"${platform}\" -t \"${platform_tag}\" .) &"
      (
        set -o pipefail
        build_start=$SECONDS
        podman ${connect_arg} build \
          "${build_args[@]}" \
          --platform "${platform}" \
          -t "${platform_tag}" \
          "${DOCKER_BUILD_CONTEXT}" 2>&1 | sed "s|^|${label} |" || exit 1
        echo $(( SECONDS - build_start )) > "${timing_dir}/${arch}"
      ) &
      build_pids+=($!)
    else
      log "PARALLEL_BUILDS DISABLED"

      local build_start=$SECONDS
      echo podman ${connect_arg} build "${build_args[@]}" --platform "${platform}" -t "${platform_tag}" "${DOCKER_BUILD_CONTEXT}"
      podman ${connect_arg} build "${build_args[@]}" --platform "${platform}" -t "${platform_tag}" "${DOCKER_BUILD_CONTEXT}" | sed "s|^|${label} |" || exit 1
      build_durations["${platform}"]=$(( SECONDS - build_start ))
      log "Build ${platform} finished in $(format_duration ${build_durations["${platform}"]})"
    fi
  done


  # Wait for all builds to complete
  if (( ${#build_pids[@]} > 0 )); then
    log "Waiting for ${#build_pids[@]} parallel builds..."
    local failed=0
    for pid in "${build_pids[@]}"; do
      if ! wait "$pid"; then
        log "Build failed (PID $pid)"
        failed=1
      fi
    done
    (( failed == 1 )) && die "One or more builds failed"
    log "All parallel builds finished in $(format_duration $(( SECONDS - builds_start )))"
    # Collect per-platform durations from subshells
    for f in "${timing_dir}"/*; do
      [[ -f "${f}" ]] || continue
      local plat_arch
      plat_arch="$(basename "${f}")"
      build_durations["linux/${plat_arch}"]="$(< "${f}")"
    done
  fi
  rm -rf "${timing_dir}"

  # Copy images built on remote to host
  local copy_start=$SECONDS
  local copy_happened=0
  for platform in "${!platform_connect_args[@]}"; do
    if [[ -n "${platform_connect_args[$platform]}" ]]; then
      local arch="${platform#*/}"
      local platform_tag="${DOCKER_IMAGE}-${arch}"
      local conn="${platform_connect_args[$platform]}"
      # Extract connection name from "--connection <name>"
      conn="${conn##*--connection }"
      copy_image_from_remote "${conn}" "${platform_tag}"
      copy_happened=1
    fi
  done
  local copy_duration=$(( SECONDS - copy_start ))

  # Create manifest from all platform images.
  # No --replace: this podman's `manifest create` doesn't support it. Instead
  # remove any leftover list right before creating. A list tagged under BOTH
  # names must be removed in ONE `manifest rm` call (single-name rm fails on a
  # multi-tagged list); fall back to a forced image rm per name.
  local -a stale_names=("${DOCKER_IMAGE}")
  if [[ -n "${DOCKER_IMAGE_LATEST}" && "${DOCKER_IMAGE}" != *:latest ]]; then
    stale_names+=("${DOCKER_IMAGE_LATEST}")
  fi
  podman manifest rm "${stale_names[@]}" 2>/dev/null \
    || podman image rm -f "${stale_names[@]}" 2>/dev/null || true
  log "Creating manifest: ${DOCKER_IMAGE}"
  echo podman manifest create "${DOCKER_IMAGE}" "${platform_tags[@]}"
  podman manifest create "${DOCKER_IMAGE}" "${platform_tags[@]}"

  if [[ -n "${DOCKER_IMAGE_LATEST}" && "${DOCKER_IMAGE}" != *:latest ]]; then
    log "Tagging as latest: ${DOCKER_IMAGE_LATEST}"
    podman tag "${DOCKER_IMAGE}" "${DOCKER_IMAGE_LATEST}"
  fi

  # Push
  local push_start=$SECONDS
  # log "To push, run:"
  # echo "  podman manifest push ${DOCKER_IMAGE} docker://${DOCKER_IMAGE}"
  echo podman manifest push "${DOCKER_IMAGE}" "docker://${DOCKER_IMAGE}"
  podman manifest push "${DOCKER_IMAGE}" "docker://${DOCKER_IMAGE}"

  if [[ -n "${DOCKER_IMAGE_LATEST}" && "${DOCKER_IMAGE}" != *:latest ]]; then
    log "Pushing as latest: ${DOCKER_IMAGE_LATEST}"
    podman manifest push "${DOCKER_IMAGE}" "docker://${DOCKER_IMAGE_LATEST}"
  fi
  local push_duration=$(( SECONDS - push_start ))

  # Timing summary
  log "--- Timing summary ---"
  for platform in "${!build_durations[@]}"; do
    log "  Build ${platform}: $(format_duration ${build_durations["${platform}"]})"
  done
  if (( copy_happened )); then
    log "  Copy images: $(format_duration ${copy_duration})"
  fi
  log "  Push: $(format_duration ${push_duration})"
  log "  Total: $(format_duration $(( SECONDS - builds_start )))"
}

build_local_only() {
  log "Building local image only..."

  export BUILDKIT_PROGRESS=plain

  # Detect native platform
  local native_arch
  native_arch="$(uname -m)"
  case "${native_arch}" in
    x86_64)  native_arch="amd64" ;;
    aarch64) native_arch="arm64" ;;
  esac
  local native_platform="linux/${native_arch}"

  local -a build_args=("${BUILD_BASE_ARGS[@]}")
  build_args+=("--platform" "${native_platform}")
  if [[ -n "${DOCKER_IMAGE_LATEST}" && "${DOCKER_IMAGE}" != *:latest ]]; then
    build_args+=("-t" "${DOCKER_IMAGE_LATEST}")
  fi

  build_args+=("-t" "${DOCKER_IMAGE}")

  echo docker build \
    "${build_args[@]}" \
    "${DOCKER_BUILD_CONTEXT}"

  docker build \
    "${build_args[@]}" \
    "${DOCKER_BUILD_CONTEXT}"
}

#=============================================================================
# MAIN
#=============================================================================
main() {
  local main_start=$SECONDS
  setup_environment

  case "${1:-}" in
    login)
      ensure_docker_login
      ;;
    check)
      if [[ -z "${REMOTE_ARM64_CONNECTION}" ]]; then
        die "REMOTE_ARM64_CONNECTION is not set"
      fi
      ensure_remote_arm64_connection
      log "Remote arm64 connection OK"
      exit 0
      ;;
    onlylocal)
      ensure_docker_login
      build_local_only
      ;;
    *)
      ensure_docker_login
      if (( DOCKER_IS_PODMAN == 1 )); then
        build_with_podman
      else
        build_with_docker
      fi
      ;;
  esac

  log "Total time: $(format_duration $(( SECONDS - main_start )))"
}

# Run main and ensure cleanup
cleanup() {
  stop_podman_vm_if_started || true
  # Remove the throwaway docker config dir created for a token login (secrets).
  if [[ -n "${TOKEN_LOGIN_CONFIG_DIR:-}" && -d "${TOKEN_LOGIN_CONFIG_DIR}" ]]; then
    rm -rf "${TOKEN_LOGIN_CONFIG_DIR}"
  fi
}
trap cleanup EXIT
main "$@"
