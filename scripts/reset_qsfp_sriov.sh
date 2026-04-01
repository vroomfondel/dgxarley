#!/bin/bash
# reset_qsfp_sriov.sh — Emergency recovery for SR-IOV VFs stuck in orphaned CNI namespaces.
#
# When: After force-deleting SGLang pods (or any pod using host-device CNI with SR-IOV VFs),
#       the VFs can get stuck in orphaned CNI network namespaces, preventing new pods from
#       starting ("failed to find host device: Link not found").
#
# What it does:
#   1. Deletes all orphaned CNI network namespaces (releases trapped VF interfaces)
#   2. Resets SR-IOV: sets numvfs=0, then re-creates VFs
#   3. Verifies all VFs are back on the host
#
# Usage: Run as root on each DGX Spark node.
#   ssh root@spark1 /usr/local/bin/reset_qsfp_sriov.sh
#   ssh root@spark2 /usr/local/bin/reset_qsfp_sriov.sh
#   ssh root@spark3 /usr/local/bin/reset_qsfp_sriov.sh
#
# Safe to run at any time — idempotent. But will disrupt any running pods that use VFs.

set -euo pipefail

# --- Configuration (matches dgx_prepare role defaults) ---
PF_PCI="${QSFP_SRIOV_PF_PCI:-0000:01:00.0}"
NUMVFS="${QSFP_SRIOV_NUMVFS:-8}"
SRIOV_PATH="/sys/bus/pci/devices/${PF_PCI}/sriov_numvfs"

# Derive PF interface name from PCI address
PF_NAME=$(ls "/sys/bus/pci/devices/${PF_PCI}/net/" 2>/dev/null | head -1)
if [ -z "$PF_NAME" ]; then
    echo "ERROR: No network interface found for PCI device ${PF_PCI}"
    exit 1
fi
VF_PREFIX="${PF_NAME%np0}"  # e.g. enp1s0f0np0 -> enp1s0f0

echo "=== QSFP SR-IOV Reset ==="
echo "PF: ${PF_NAME} (PCI: ${PF_PCI})"
echo "Target VFs: ${NUMVFS}"
echo ""

# --- Step 1: Clean orphaned CNI namespaces ---
echo "--- Step 1: Cleaning orphaned CNI network namespaces ---"
orphan_count=0
for ns in $(ip netns list 2>/dev/null | awk '{print $1}' | grep '^cni-'); do
    ip netns delete "$ns" 2>/dev/null && echo "  Deleted: $ns" && ((orphan_count++)) || true
done
if [ "$orphan_count" -eq 0 ]; then
    echo "  No orphaned CNI namespaces found."
else
    echo "  Cleaned ${orphan_count} orphaned namespace(s)."
fi
echo ""

# --- Step 2: Reset SR-IOV VFs ---
echo "--- Step 2: Resetting SR-IOV VFs ---"
current=$(cat "$SRIOV_PATH" 2>/dev/null || echo 0)
echo "  Current numvfs: ${current}"

if [ "$current" -ne 0 ]; then
    echo "  Disabling VFs (numvfs=0)..."
    echo 0 > "$SRIOV_PATH"
    sleep 1
fi

echo "  Creating ${NUMVFS} VFs..."
echo "$NUMVFS" > "$SRIOV_PATH"
sleep 1

# Bring VFs up with correct MTU (kernel defaults to DOWN + MTU 1500)
MTU="${QSFP_VF_MTU:-9000}"
echo "  Configuring VFs: MTU=${MTU}, state=UP..."
for i in $(seq 0 $((NUMVFS - 1))); do
    vf_name="${VF_PREFIX}v${i}"
    ip link set "$vf_name" mtu "$MTU" up 2>/dev/null || true
done
echo ""

# --- Step 3: Verify ---
echo "--- Step 3: Verifying VFs ---"
found=0
missing=()
for i in $(seq 0 $((NUMVFS - 1))); do
    vf_name="${VF_PREFIX}v${i}"
    if ip link show "$vf_name" &>/dev/null; then
        ((found++))
    else
        missing+=("$vf_name")
    fi
done

if [ "$found" -eq "$NUMVFS" ]; then
    echo "  OK: All ${NUMVFS} VFs present (${VF_PREFIX}v0..v$((NUMVFS-1)))"
else
    echo "  WARNING: Only ${found}/${NUMVFS} VFs found."
    echo "  Missing: ${missing[*]}"
    exit 1
fi

echo ""
echo "=== Done. Pods using SR-IOV VFs can now be (re)started. ==="
