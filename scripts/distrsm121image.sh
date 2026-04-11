#!/usr/bin/env bash
set -euo pipefail

SRC_IMAGE="localhost/xomoxcc/dgx-spark-sglang:0.5.10-sm121"
IMAGE="docker.io/xomoxcc/dgx-spark-sglang:0.5.10-sm121"
SOURCE="spark4.local"
TARGETS=(spark1.local spark2.local spark3.local spark4.local)
INNERSOURCE="10.10.10.4"  # spark4 via qsfp
SSH_OPTS=()

# Auf spark4 — retag für docker.io namespace (wichtig für containerd name-match)
ssh "${SSH_OPTS[@]}" "root@${SOURCE}" "podman tag ${SRC_IMAGE} ${IMAGE}"

# Image von spark4 via Pipe an alle 4 Nodes streamen (inkl. spark4 selbst).
# image rm vorher ist nicht nötig: 'k3s ctr image import' ersetzt das Tag atomar.
for host in "${TARGETS[@]}"; do
  echo "=== streaming to ${host} ==="
  if [[ "${host}" == "${SOURCE}" ]]; then
    # Lokal auf der Quelle: alles in einer SSH-Session
    ssh "${SSH_OPTS[@]}" "root@${host}" \
      "podman save --format docker-archive ${IMAGE} | k3s ctr -n k8s.io image import -"
  else
    ssh "${SSH_OPTS[@]}" "root@${INNERSOURCE}" "podman save --format docker-archive ${IMAGE}" \
      | ssh "${SSH_OPTS[@]}" "root@${host}" "k3s ctr -n k8s.io image import -"
  fi
done
