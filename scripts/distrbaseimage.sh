#!/usr/bin/env bash
#
# distrbaseimage.sh — Quickly distribute a container image from one DGX Spark
# to the other sparks via a throwaway podman-hosted registry:2 on the source
# host, using the QSFP network (10.10.10.0/24, 200 GbE) for the heavy lifting.
#
# Flow:
#   1. Start temporary registry:2 on ${SOURCE} (binds 0.0.0.0:${REGISTRY_PORT})
#   2. podman push ${IMAGE} from source's local podman store into the registry
#   3. On each other spark in parallel, k3s ctr pulls from the registry into
#      the k8s.io containerd namespace, retags to the original name, and
#      drops the ephemeral registry-ref tag
#   4. Tear down the registry on source
#
# Prerequisites
# -------------
# - The ${SOURCE} host already has ${IMAGE} in its local podman store.
# - Root SSH from control host to all sparks works.
# - All targets can reach ${REGISTRY_HOST}:${REGISTRY_PORT} over QSFP.
#

set -euo pipefail

IMAGE="${IMAGE:-scitrera/dgx-spark-sglang:0.5.12}"

# Source spark (management address) holding the image in its podman store.
SOURCE="spark4.local"

# QSFP address the registry listens on and targets pull from. Must match
# the --source host's QSFP IP.
REGISTRY_HOST="10.10.10.4"
REGISTRY_PORT="5000"

USE_TMUX=1

usage() {
    cat <<EOF
Usage: $(basename "$0") [--image REF] [--source HOST] [--registry-host IP]
                        [--no-tmux] [--help]

Distribute a container image from one DGX Spark to the other sparks via a
throwaway registry:2 on the source host, over QSFP.

Options:
  --image REF            Image reference (must already exist in the source
                         host's podman store under exactly this name).
                         Default: ${IMAGE}
  --source HOST          Management address of the spark that holds the
                         image in its podman store. SSH is used to run
                         podman commands on this host.
                         Default: ${SOURCE}
  --registry-host IP     QSFP address the targets pull from. Typically the
                         --source host's QSFP IP.
                         Default: ${REGISTRY_HOST}
  --no-tmux              Disable the tmux pane view for the parallel pulls
                         and use flat merged output. Auto-enabled when tmux
                         is not installed or the script runs without a TTY.
  --help                 Show this help.

Examples:
  # Default — spark4 to the other 3 sparks
  $(basename "$0")

  # Image lives on spark3 instead
  $(basename "$0") --source spark3.local --registry-host 10.10.10.3

  # Distribute a different image
  $(basename "$0") --image scitrera/dgx-spark-vllm:0.17.0-t5
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --image)
            shift
            [[ $# -gt 0 ]] || { echo "ERROR: --image requires an argument" >&2; exit 1; }
            IMAGE="$1"
            shift
            ;;
        --image=*)
            IMAGE="${1#--image=}"
            shift
            ;;
        --source)
            shift
            [[ $# -gt 0 ]] || { echo "ERROR: --source requires an argument" >&2; exit 1; }
            SOURCE="$1"
            shift
            ;;
        --source=*)
            SOURCE="${1#--source=}"
            shift
            ;;
        --registry-host)
            shift
            [[ $# -gt 0 ]] || { echo "ERROR: --registry-host requires an argument" >&2; exit 1; }
            REGISTRY_HOST="$1"
            shift
            ;;
        --registry-host=*)
            REGISTRY_HOST="${1#--registry-host=}"
            shift
            ;;
        --no-tmux)
            USE_TMUX=0
            shift
            ;;
        --help|-h)
            usage
            exit 0
            ;;
        *)
            echo "ERROR: Unknown argument: $1 (use --help)" >&2
            exit 1
            ;;
    esac
done

REGISTRY_IMAGE="${REGISTRY_HOST}:${REGISTRY_PORT}/${IMAGE}"
REGISTRY_CONTAINER="tmp-distr-registry"

# Target list = all sparks except the source (source already has the image).
ALL_SPARKS=(spark1.local spark2.local spark3.local spark4.local)
TARGETS=()
for host in "${ALL_SPARKS[@]}"; do
    [[ "${host}" == "${SOURCE}" ]] && continue
    TARGETS+=("${host}")
done
if (( ${#TARGETS[@]} == 0 )); then
    echo "ERROR: source ${SOURCE} is not in the known spark list — nothing to do" >&2
    exit 1
fi

SSH_OPTS=(-o BatchMode=yes -o StrictHostKeyChecking=accept-new)

cleanup() {
    echo "=== cleanup: stopping temporary registry on ${SOURCE} ==="
    ssh "${SSH_OPTS[@]}" "root@${SOURCE}" \
        "podman rm -f ${REGISTRY_CONTAINER} >/dev/null 2>&1 || true" || true
}
trap cleanup EXIT

# 1. Throwaway registry:2 on the source. --network host avoids slirp/netns
# overhead for the large blob transfers and binds on 0.0.0.0 so the QSFP
# IP is reachable. Short-lived + LAN-only + plain HTTP — no security issue.
echo "=== starting temporary registry on ${SOURCE} (0.0.0.0:${REGISTRY_PORT}, clients pull via ${REGISTRY_HOST}) ==="
ssh "${SSH_OPTS[@]}" "root@${SOURCE}" \
    "podman rm -f ${REGISTRY_CONTAINER} >/dev/null 2>&1 || true; \
     podman run -d --name ${REGISTRY_CONTAINER} --network host \
        -e REGISTRY_HTTP_ADDR=0.0.0.0:${REGISTRY_PORT} \
        docker.io/library/registry:2 >/dev/null"

for i in {1..10}; do
    if ssh "${SSH_OPTS[@]}" "root@${SOURCE}" \
        "curl -sf http://${REGISTRY_HOST}:${REGISTRY_PORT}/v2/ >/dev/null"; then
        echo "registry ready"
        break
    fi
    if [[ ${i} -eq 10 ]]; then
        echo "registry did not become ready in time" >&2
        exit 1
    fi
    sleep 1
done

# 2. Push from source's podman store into the local registry (loopback).
echo "=== podman push ${IMAGE} -> ${REGISTRY_IMAGE} ==="
ssh "${SSH_OPTS[@]}" "root@${SOURCE}" \
    "podman push --tls-verify=false '${IMAGE}' '${REGISTRY_IMAGE}'"

# 3. Parallel pull on all targets. Layer fetch is concurrent per-node AND
# concurrent across nodes; sha256 + snapshotter unpack are CPU-bound and
# scale that way. After pull: retag to the original name and drop the
# ephemeral registry-ref tag so only ${IMAGE} remains.

parallel_pull_flat() {
    echo "=== parallel pull on targets (flat mode) ==="
    local pids=() host short
    for host in "${TARGETS[@]}"; do
        short="${host%%.*}"
        (
            ssh "${SSH_OPTS[@]}" "root@${host}" \
                "set -e; \
                 k3s ctr -n k8s.io image pull --plain-http '${REGISTRY_IMAGE}'; \
                 k3s ctr -n k8s.io image rm '${IMAGE}' >/dev/null 2>&1 || true; \
                 k3s ctr -n k8s.io image tag '${REGISTRY_IMAGE}' '${IMAGE}'; \
                 k3s ctr -n k8s.io image rm '${REGISTRY_IMAGE}' >/dev/null" \
                2>&1 | sed "s/^/[${short}] /"
        ) &
        pids+=($!)
    done
    local fail=0 pid
    for pid in "${pids[@]}"; do
        wait "${pid}" || fail=1
    done
    return ${fail}
}

parallel_pull_tmux() {
    local work_dir session host short first_short i
    work_dir=$(mktemp -d -t "distrbase-pull.XXXXXX")
    session="distrbase-pull-$$"

    # Per-target driver scripts: avoid nested-quoting hell when shoving
    # commands into tmux "..." arguments. Each script ssh-es to its target,
    # runs the registry pull, writes its exit code to work_dir/${short}.rc,
    # and prints a final banner so each pane shows success/failure at a glance.
    for host in "${TARGETS[@]}"; do
        short="${host%%.*}"
        cat > "${work_dir}/${short}.sh" <<EOF
#!/usr/bin/env bash
set -o pipefail
echo "=== ${short}: registry pull from ${REGISTRY_HOST}:${REGISTRY_PORT} ==="
ssh -o BatchMode=yes -o StrictHostKeyChecking=accept-new "root@${host}" \\
    "set -e; \\
     k3s ctr -n k8s.io image pull --plain-http '${REGISTRY_IMAGE}'; \\
     k3s ctr -n k8s.io image rm '${IMAGE}' >/dev/null 2>&1 || true; \\
     k3s ctr -n k8s.io image tag '${REGISTRY_IMAGE}' '${IMAGE}'; \\
     k3s ctr -n k8s.io image rm '${REGISTRY_IMAGE}' >/dev/null"
rc=\$?
echo "\${rc}" > "${work_dir}/${short}.rc"
echo
if [[ \${rc} -eq 0 ]]; then
    echo "================================================================"
    echo "  ${short}: OK  —  detach with Ctrl+B then d once all done"
    echo "================================================================"
else
    echo "################################################################"
    echo "  ${short}: FAILED (rc=\${rc})"
    echo "################################################################"
fi
EOF
        chmod +x "${work_dir}/${short}.sh"
    done

    # Tiled layout: detached session with first target, then split for the
    # rest, retile after each. remain-on-exit keeps panes visible after their
    # command finishes so the final banner stays on screen.
    tmux kill-session -t "${session}" 2>/dev/null || true
    first_short="${TARGETS[0]%%.*}"
    tmux new-session -d -s "${session}" -x 220 -y 50 \
        "bash ${work_dir}/${first_short}.sh"
    tmux set-option -t "${session}" remain-on-exit on
    tmux set-option -t "${session}" -g pane-border-status top 2>/dev/null || true
    tmux select-pane -t "${session}.0" -T "${first_short}"

    for ((i=1; i<${#TARGETS[@]}; i++)); do
        short="${TARGETS[$i]%%.*}"
        tmux split-window -t "${session}" "bash ${work_dir}/${short}.sh"
        tmux select-layout -t "${session}" tiled >/dev/null
        tmux select-pane -t "${session}.${i}" -T "${short}"
    done
    tmux select-layout -t "${session}" tiled >/dev/null

    # Background watcher: once all .rc files exist, auto-detach after a
    # short grace period so the user gets a beat to read the final banners.
    (
        while true; do
            local ready=0 w_host w_short
            for w_host in "${TARGETS[@]}"; do
                w_short="${w_host%%.*}"
                [[ -f "${work_dir}/${w_short}.rc" ]] && ready=$((ready + 1))
            done
            if (( ready == ${#TARGETS[@]} )); then
                sleep 5
                tmux detach-client -s "${session}" 2>/dev/null || true
                exit 0
            fi
            sleep 1
        done
    ) &
    local watcher_pid=$!

    echo "=== attaching to tmux session '${session}' ==="
    echo "=== Auto-detach 5 s after all panes finish (or Ctrl+B d to leave early) ==="
    sleep 1
    tmux attach-session -t "${session}"
    kill "${watcher_pid}" 2>/dev/null || true
    wait "${watcher_pid}" 2>/dev/null || true
    tmux kill-session -t "${session}" 2>/dev/null || true

    local fail=0 rc
    for host in "${TARGETS[@]}"; do
        short="${host%%.*}"
        rc=$(cat "${work_dir}/${short}.rc" 2>/dev/null || echo "missing")
        if [[ "${rc}" == "0" ]]; then
            echo "[${short}] ok"
        else
            echo "[${short}] FAILED (rc=${rc})"
            fail=1
        fi
    done
    rm -rf "${work_dir}"
    return ${fail}
}

# Dispatcher: prefer tmux, fall back to flat if tmux is missing or there
# is no TTY. --no-tmux forces flat regardless.
if (( USE_TMUX == 1 )) && command -v tmux >/dev/null 2>&1 && [[ -t 0 ]] && [[ -t 1 ]]; then
    parallel_pull_tmux
    fail=$?
else
    if (( USE_TMUX == 1 )); then
        if ! command -v tmux >/dev/null 2>&1; then
            echo "=== tmux not installed, falling back to flat mode ===" >&2
        elif ! [[ -t 0 && -t 1 ]]; then
            echo "=== no TTY, falling back to flat mode ===" >&2
        fi
    fi
    parallel_pull_flat
    fail=$?
fi

if [[ ${fail} -ne 0 ]]; then
    echo "=== one or more targets failed ==="
    exit 1
fi

echo "=== distribution complete ==="
