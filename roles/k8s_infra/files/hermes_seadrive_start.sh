#!/bin/bash
# Shared seadrive FUSE sidecar entrypoint for BOTH the hermes-agent and
# hermes-webui pods. Each pod runs its own independent seadrive instance against
# the same Seafile `hermes-workspace` library; the Seafile server is the source
# of truth, with a few-seconds sync lag between the pods.
#
# Flow:
#   1. Ensure /etc/fuse.conf has user_allow_other so containers running as a
#      different UID can read /workspace contents.
#   2. Launch seadrive in the background, mounting all libs at /seadrive —
#      `hermes-workspace` appears at /seadrive/${SEAFILE_LIBRARY_NAME}.
#   3. Wait for that subdir to materialise (FUSE async cache; ~3-10s).
#   4. bind-mount it onto /workspace (the shared emptyDir mount point); via
#      Bidirectional propagation this becomes visible to the other containers
#      (webui + hermes-email, or the agent) as their /workspace contents.
#   5. wait on the seadrive process; preStop / signal trap unmounts cleanly.
#
# Env (from the hermes-seafile-<user> Secret + the sidecar's env):
#   SEAFILE_SERVER / SEAFILE_USER / SEAFILE_TOKEN — account credentials.
#   WANTED_UID / WANTED_GID — FUSE mount ownership.
#   SEAFILE_LIBRARY_NAME — the library to bind onto /workspace.
set -euo pipefail
mkdir -p /etc /var/cache/seadrive /seadrive
grep -q '^user_allow_other' /etc/fuse.conf 2>/dev/null \
  || echo 'user_allow_other' >> /etc/fuse.conf

cat > /tmp/seadrive.conf <<EOF
[account]
server = ${SEAFILE_SERVER}
username = ${SEAFILE_USER}
token = ${SEAFILE_TOKEN}
is_pro = false

[general]
client_name = hermes-${HOSTNAME}
EOF
chmod 0600 /tmp/seadrive.conf

# Background seadrive so we can run the bind-mount step in parallel.
seadrive \
  -c /tmp/seadrive.conf \
  -d /var/cache/seadrive \
  -l /var/log/seadrive.log \
  -o allow_other,uid=${WANTED_UID},gid=${WANTED_GID},umask=0077 \
  /seadrive &
SEADRIVE_PID=$!

# Wait for the library to appear (FUSE caches lazily; ~3-10s).
for i in $(seq 1 60); do
    if [ -d "/seadrive/${SEAFILE_LIBRARY_NAME}" ]; then
        break
    fi
    sleep 1
done
if [ ! -d "/seadrive/${SEAFILE_LIBRARY_NAME}" ]; then
    echo "seadrive: library '${SEAFILE_LIBRARY_NAME}' did not appear within 60s — check Seafile credentials/server reachability" >&2
    cat /var/log/seadrive.log >&2 || true
    kill ${SEADRIVE_PID} 2>/dev/null || true
    exit 1
fi

# Propagate the lib content onto the shared emptyDir mount.
mount --bind "/seadrive/${SEAFILE_LIBRARY_NAME}" /workspace
echo "seadrive: ${SEAFILE_LIBRARY_NAME} mounted at /workspace (pid=${SEADRIVE_PID})"

# Forward signals to seadrive so K8s graceful shutdown works.
trap 'umount /workspace 2>/dev/null || true; fusermount -u /seadrive 2>/dev/null || true; kill -TERM ${SEADRIVE_PID} 2>/dev/null || true' TERM INT
wait ${SEADRIVE_PID}
