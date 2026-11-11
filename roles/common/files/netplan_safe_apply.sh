#!/bin/bash
# netplan-safe-apply BACKUP_TAR TIMEOUT CHECK_GW [OVS_BRIDGE...]
# Applies the current /etc/netplan, then waits TIMEOUT seconds for both a passing self-check (ping CHECK_GW) and
# the confirm file written by the controller; otherwise restores BACKUP_TAR and re-applies the old config.
set -u
BACKUP="$1"
TIMEOUT="$2"
CHECK_GW="$3"
shift 3
STATE_DIR=/run/netplan-safe-apply
mkdir -p "${STATE_DIR}"
rm -f "${STATE_DIR}/confirm"

state() {
    echo "$1" > "${STATE_DIR}/state"
    logger -t netplan-safe-apply "state=$1 ${2:-}"
}

links() {
    ip -o link show | awk -F': ' '{print $2}' | cut -d@ -f1 | sort
}

rollback() {
    state rolling-back "$1"
    comm -13 "${STATE_DIR}/links.before" <(links) | while read -r l; do ip link delete "${l}" 2>/dev/null; done
    find /etc/netplan -maxdepth 1 -name '*.yaml' -delete
    rm -rf /etc/systemd/network/*.d /etc/networkd-dispatcher/configured.d/*
    tar -xpf "${BACKUP}" -C /
    systemctl daemon-reload
    timeout 600 netplan apply || { sleep 5; timeout 600 netplan apply; } || logger -t netplan-safe-apply "rollback apply failed twice"
    state rolled-back "$1"
    exit 1
}

state applying
links > "${STATE_DIR}/links.before"
for br in "$@"; do
    ovs-vsctl --timeout=10 --if-exists del-br "${br}" || rollback "del-br ${br} failed"
done
timeout 600 netplan apply || rollback "netplan apply failed"

state awaiting-confirm
deadline=$(( $(date +%s) + TIMEOUT ))
gw_ok=0
while [ "$(date +%s)" -lt "${deadline}" ]; do
    if [ "${gw_ok}" -eq 0 ] && ping -c1 -W1 "${CHECK_GW}" >/dev/null 2>&1; then
        gw_ok=1
        logger -t netplan-safe-apply "gateway ${CHECK_GW} reachable"
    fi
    if [ "${gw_ok}" -eq 1 ] && [ -f "${STATE_DIR}/confirm" ]; then
        state committed
        exit 0
    fi
    sleep 1
done
rollback "no confirmation within ${TIMEOUT}s (gateway_ok=${gw_ok})"
