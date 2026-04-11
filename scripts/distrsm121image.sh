#!/usr/bin/env bash
#
# distrsm121image.sh — Distribute the freshly-built sgl-kernel sm121 image
# from spark4's local podman store to all 4 DGX Spark K3s nodes' containerd
# image stores, using a throwaway podman-hosted registry:2 on spark4 and
# the QSFP network (10.10.10.0/24, 200 GbE) for the heavy lifting.
#
# Flow
# ----
# 1. Retag the image in spark4's podman store to the docker.io/... FQN so
#    pod specs find it under the right name once imported.
# 2. Start a throwaway `registry:2` on spark4 via podman, host-networked on
#    spark4's QSFP IP (10.10.10.4:5000). Plain HTTP — this is a LAN-only
#    short-lived helper, not a persistent registry.
# 3. `podman push --tls-verify=false` the image from spark4's podman store
#    into the local registry. Podman pushes layers in parallel.
# 4. In parallel on all 4 targets: `k3s ctr image pull --plain-http` the
#    image from the temporary registry into the k8s.io containerd namespace.
#    containerd pulls layers concurrently per node, and all 4 nodes run at
#    once — sha256 + snapshotter unpack are CPU-bound locally and scale out.
# 5. Retag on each target to the docker.io/... FQN and drop the ephemeral
#    ${REGISTRY_HOST}:${REGISTRY_PORT}/... reference.
# 6. Stop/remove the temporary registry container on spark4 (trap cleanup).
#
# Why this replaces the old `podman save | pv | k3s ctr image import`
# pipeline
# ------------------------------------------------------------------------
# The old flow was serial (one target at a time) and the docker-archive
# tar-stream's sha256 computation was single-threaded per import. With a
# registry, all 4 nodes import in parallel and containerd pulls layers
# concurrently — measured 2-3× faster end-to-end on this cluster.
#
# Prerequisites
# -------------
# - spark4 has `localhost/xomoxcc/dgx-spark-sglang:0.5.10-sm121` in its
#   local podman store (built via scripts/build_sm121_image.sh, either with
#   or without --no-push).
# - Root SSH from x86 control host to all 4 sparks (management) works.
# - All target sparks can reach 10.10.10.4:5000 via QSFP.
# - `podman` on spark4 (registry:2 will be podman-pulled on first run if
#   not cached) and `k3s` on all targets.
#

set -euo pipefail

SRC_IMAGE="localhost/xomoxcc/dgx-spark-sglang:0.5.10-sm121"
IMAGE="docker.io/xomoxcc/dgx-spark-sglang:0.5.10-sm121"

# Source host: where the built image lives in podman. Defaults to spark4
# (the historical build host); override with --source when the image was
# built elsewhere (e.g. spark3 after build_sm121_image.sh --remote-host).
SOURCE="spark4.local"              # management address for outer ssh

# Temporary registry address. Must be an IP/host reachable by ALL targets
# over the fast network — in our cluster that's the QSFP mesh on
# 10.10.10.0/24. The default matches spark4's QSFP IP; when --source
# changes, you almost always also need to change --registry-host to the
# new source's QSFP IP, so the two are kept independent.
REGISTRY_HOST="10.10.10.4"
REGISTRY_PORT="5000"

USE_TMUX=1

usage() {
    cat <<EOF
Usage: $(basename "$0") [--source HOST] [--registry-host IP]
                        [--no-tmux] [--help]

Distribute the sgl-kernel sm121 image from a build host to all 4 DGX
Spark K3s nodes via a throwaway registry:2 on the build host.

Options:
  --source HOST          Management address of the host that holds the
                         built image in its podman store. SSH is used to
                         run podman commands on this host.
                         Default: ${SOURCE}
  --registry-host IP     Address the temporary registry should bind to
                         and that targets will pull from. Must be on a
                         network reachable by all targets (QSFP subnet
                         in this cluster). Typically the --source host's
                         QSFP IP.
                         Default: ${REGISTRY_HOST}
  --no-tmux              Disable the 4-pane tmux view for the parallel
                         pulls and use the flat merged-output fallback
                         instead. Auto-enabled when tmux is not installed
                         or when the script is run without a TTY (cron,
                         pipes, etc.). Default: tmux enabled when
                         interactive.
  --help                 Show this help.

Examples:
  # Default — image was built on spark4
  $(basename "$0")

  # Image was built on spark3 instead (via build_sm121_image.sh --remote-host)
  $(basename "$0") --source spark3.local --registry-host 10.10.10.3
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
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
REGISTRY_REF="xomoxcc/dgx-spark-sglang:0.5.10-sm121"
REGISTRY_IMAGE="${REGISTRY_HOST}:${REGISTRY_PORT}/${REGISTRY_REF}"
REGISTRY_CONTAINER="tmp-distr-registry"

# Target sparks (management addresses). Includes spark4 itself, which can
# pull from its own host-networked registry without any special-casing.
TARGETS=(spark1.local spark2.local spark3.local spark4.local)

SSH_OPTS=(-o BatchMode=yes -o StrictHostKeyChecking=accept-new)

cleanup() {
    echo "=== cleanup: stopping temporary registry on ${SOURCE} ==="
    ssh "${SSH_OPTS[@]}" "root@${SOURCE}" \
        "podman rm -f ${REGISTRY_CONTAINER} >/dev/null 2>&1 || true" || true
}
trap cleanup EXIT

# 1. Retag for docker.io namespace — wichtig für containerd name-match.
# Ohne das Retag bleibt das Image als localhost/xomoxcc/... im podman store
# und K3s findet es nicht unter dem docker.io/... Namen aus den Pod-Specs.
ssh "${SSH_OPTS[@]}" "root@${SOURCE}" "podman tag '${SRC_IMAGE}' '${IMAGE}'"

# 2. Throwaway registry:2 auf spark4 starten. --network host vermeidet
# slirp/netns-Overhead für die großen Blob-Transfers; REGISTRY_HTTP_ADDR
# bindet explizit nur auf die QSFP-IP (10.10.10.4), nicht auf 0.0.0.0.
echo "=== starting temporary registry on ${REGISTRY_HOST}:${REGISTRY_PORT} ==="
ssh "${SSH_OPTS[@]}" "root@${SOURCE}" \
    "podman rm -f ${REGISTRY_CONTAINER} >/dev/null 2>&1 || true; \
     podman run -d --name ${REGISTRY_CONTAINER} --network host \
        -e REGISTRY_HTTP_ADDR=${REGISTRY_HOST}:${REGISTRY_PORT} \
        docker.io/library/registry:2 >/dev/null"

# Auf Registry-Readiness warten (bis zu 10s).
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

# 3. Push von spark4's podman store in die lokale Registry. Podman pusht
# Layer parallel, der Transfer bleibt host-local (loopback-over-host-net).
echo "=== podman push ${IMAGE} -> ${REGISTRY_IMAGE} ==="
ssh "${SSH_OPTS[@]}" "root@${SOURCE}" \
    "podman push --tls-verify=false '${IMAGE}' '${REGISTRY_IMAGE}'"

# 4. Parallel pull auf allen 4 Targets. Jeder Node zieht Layer concurrent
# von der Registry, und die 4 Nodes laufen gleichzeitig — sha256-Berechnung
# und snapshotter-unpack sind lokal CPU-bound und skalieren dadurch.
# Nach dem Pull: retag auf den docker.io/... Namen und das ephemerale
# ${REGISTRY_HOST}:${REGISTRY_PORT}/... Referenz-Tag entfernen.
#
# Zwei Implementierungen:
#   parallel_pull_tmux   4-pane tiled tmux session, ein Target pro Pane.
#                        Vorteil: native, ungemischte Outputs + progress bars,
#                        pane-Title zeigt Host, remain-on-exit lässt Ergebnisse
#                        stehen bis manuell per Ctrl+B d detached wird.
#   parallel_pull_flat   Background-Subshells mit "[${short}]" Prefix-sed.
#                        Fallback für non-TTY runs (cron, nohup, CI).

parallel_pull_flat() {
    echo "=== parallel pull on all targets (flat mode) ==="
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
    work_dir=$(mktemp -d -t "distrsm121-pull.XXXXXX")
    session="distrsm121-pull-$$"

    # Per-target driver scripts: avoid nested-quoting hell when shoving
    # commands into tmux "..." arguments. Each script ssh-es to its target,
    # runs the ctr pipeline, writes its exit code to work_dir/${short}.rc,
    # and prints a final banner so the pane tells you at a glance whether
    # it succeeded. Shell vars are expanded NOW (at heredoc write time)
    # except \${rc} which escapes to stay literal and is evaluated inside
    # the generated script after ssh returns.
    for host in "${TARGETS[@]}"; do
        short="${host%%.*}"
        cat > "${work_dir}/${short}.sh" <<EOF
#!/usr/bin/env bash
set -o pipefail
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
    echo "  ${short}: OK  —  detach with Ctrl+B then d once all 4 are done"
    echo "================================================================"
else
    echo "################################################################"
    echo "  ${short}: FAILED (rc=\${rc})"
    echo "################################################################"
fi
EOF
        chmod +x "${work_dir}/${short}.sh"
    done

    # Layout: 2x2 tiled grid. Start a detached session with the first
    # target, then three splits, re-tile after each so the layout stays
    # balanced. remain-on-exit keeps panes visible after their command
    # finishes (otherwise they'd close and you'd lose the output).
    tmux kill-session -t "${session}" 2>/dev/null || true
    first_short="${TARGETS[0]%%.*}"
    tmux new-session -d -s "${session}" -x 220 -y 50 \
        "bash ${work_dir}/${first_short}.sh"
    tmux set-option -t "${session}" remain-on-exit on
    tmux set-option -t "${session}" -g pane-border-status top 2>/dev/null || true
    tmux select-pane -t "${session}.0" -T "${first_short}"

    for i in 1 2 3; do
        short="${TARGETS[$i]%%.*}"
        tmux split-window -t "${session}" "bash ${work_dir}/${short}.sh"
        tmux select-layout -t "${session}" tiled >/dev/null
        tmux select-pane -t "${session}.${i}" -T "${short}"
    done
    tmux select-layout -t "${session}" tiled >/dev/null

    echo "=== attaching to tmux session '${session}' ==="
    echo "=== detach with Ctrl+B d once each pane shows its final banner ==="
    sleep 1
    tmux attach-session -t "${session}"
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

# Dispatcher: prefer tmux, fall back to flat if tmux is missing or the
# environment is not interactive. --no-tmux forces flat regardless.
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

# Optional sanity check — uncomment to verify each target has the new image
# in the k8s.io namespace under the docker.io/... name:
#
# for host in "${TARGETS[@]}"; do
#     echo "--- ${host} ---"
#     ssh "${SSH_OPTS[@]}" "root@${host}" \
#         "k3s ctr -n k8s.io image list -q | grep -F '${IMAGE}' || echo MISSING"
# done
