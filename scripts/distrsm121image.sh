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

# Source host: where the built image lives in podman.
SOURCE="spark4.local"              # management address for outer ssh

# Temporary registry on spark4, reachable from all targets via QSFP.
REGISTRY_HOST="10.10.10.4"
REGISTRY_PORT="5000"
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
echo "=== parallel pull on all targets ==="
pids=()
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

fail=0
for pid in "${pids[@]}"; do
    if ! wait "${pid}"; then
        fail=1
    fi
done

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
