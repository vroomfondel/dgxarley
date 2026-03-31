#!/bin/sh
# Poll-loop waiting for worker pods to become Ready.
# Uses active polling instead of `kubectl wait` to avoid the watch-based
# race condition where the condition is already met before the watch starts.
#
# Environment variables:
#   WAIT_NAMESPACE  — Kubernetes namespace (required)
#   WAIT_LABEL      — pod label selector (required)
#   WAIT_COUNT      — number of worker pods to wait for (default: 1)
#   WAIT_TIMEOUT    — timeout in seconds (default: 600)
#   WAIT_INTERVAL   — poll interval in seconds (default: 5)

NS="${WAIT_NAMESPACE:?WAIT_NAMESPACE is required}"
LABEL="${WAIT_LABEL:?WAIT_LABEL is required}"
WAIT_COUNT="${WAIT_COUNT:-1}"
TIMEOUT="${WAIT_TIMEOUT:-600}"
INTERVAL="${WAIT_INTERVAL:-5}"

elapsed=0
echo "Waiting for ${WAIT_COUNT} pod(s) (label: ${LABEL}) in namespace ${NS} ..."

while [ $elapsed -lt $TIMEOUT ]; do
  # --field-selector filters out stale Completed/Failed pods from previous Jobs
  ready_count=$(kubectl get pod -n "$NS" -l "$LABEL" --field-selector=status.phase=Running \
    -o jsonpath='{range .items[*]}{.status.conditions[?(@.type=="Ready")].status}{"\n"}{end}' \
    2>/dev/null | grep -c "True" || echo 0)
  echo "$(date '+%H:%M:%S') workers ready: ${ready_count}/${WAIT_COUNT} (${elapsed}s/${TIMEOUT}s)"
  if [ "$ready_count" -ge "$WAIT_COUNT" ]; then
    echo "All ${WAIT_COUNT} worker pod(s) are Ready."
    exit 0
  fi
  sleep $INTERVAL
  elapsed=$((elapsed + INTERVAL))
done

echo "ERROR: Timed out after ${TIMEOUT}s waiting for ${WAIT_COUNT} worker(s). Only ${ready_count:-0} Ready."
exit 1
