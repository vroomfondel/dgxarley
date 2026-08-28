#!/usr/bin/env bash
# Downloads the upstream Multus thick DaemonSet manifest and applies 6 K3s-specific
# patches. Stores the SHA256 of the unpatched upstream in the output header so that
# subsequent runs can detect whether the upstream actually changed.
#
# The manifest is pulled from a RELEASE TAG, not master. It used to track
# master, which meant the image tag stayed at the rolling `snapshot-thick` and
# every pod restart could pull a different master build of the CNI daemon while
# the 5 K3s patches below were maintained against a moving target.
#
# Usage:  bash roles/k8s_dgx/files/update_multus_manifest.sh
#         MULTUS_REF=v4.4.0 bash roles/k8s_dgx/files/update_multus_manifest.sh
#
# Newest release: gh api repos/k8snetworkplumbingwg/multus-cni/releases/latest --jq '.tag_name'
set -euo pipefail

MULTUS_REF="${MULTUS_REF:-v4.3.0}"
UPSTREAM_URL="https://raw.githubusercontent.com/k8snetworkplumbingwg/multus-cni/${MULTUS_REF}/deployments/multus-daemonset-thick.yml"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
DEST="${SCRIPT_DIR}/multus-daemonset-thick.yml"
TMP="$(mktemp)"
trap 'rm -f "$TMP"' EXIT

echo "Downloading upstream Multus thick DaemonSet..."
curl -fsSL "$UPSTREAM_URL" -o "$TMP"

# --- SHA256 of unpatched upstream ---
UPSTREAM_HASH=$(sha256sum "$TMP" | awk '{print $1}')
echo "Upstream SHA256: ${UPSTREAM_HASH}"

if [[ -f "$DEST" ]]; then
    STORED_HASH=$(grep '^# Upstream SHA256:' "$DEST" | awk '{print $4}' || true)
    if [[ "$STORED_HASH" == "$UPSTREAM_HASH" ]]; then
        echo "Already up to date (upstream unchanged)."
        exit 0
    fi
    echo "Upstream changed (stored: ${STORED_HASH:-none}). Re-patching..."
else
    echo "No existing file found. Creating fresh patched manifest..."
fi

# ============================================================================
# Patch 1: cni volume hostPath
#   /etc/cni/net.d → /var/lib/rancher/k3s/agent/etc/cni/net.d
#   K3s stores Flannel CNI config there, not in the standard path.
# ============================================================================
sed -i 's|path: /etc/cni/net.d|path: /var/lib/rancher/k3s/agent/etc/cni/net.d|' "$TMP"
grep -q '/var/lib/rancher/k3s/agent/etc/cni/net.d' "$TMP" \
    || { echo "PATCH 1 FAILED: cni hostPath pattern not found" >&2; exit 1; }
echo "  Patch 1 applied: cni volume hostPath"

# ============================================================================
# Patch 2: cnibin paths — widen from /opt/cni/bin to K3s data root
#   Why the parent (/var/lib/rancher/k3s/data) instead of .../data/cni:
#   K3s bind-mounts <hash>/bin → .../data/cni. Inside, CNI plugins are absolute
#   symlinks pointing into .../data/<hash>/bin/cni. Mounting only .../data/cni
#   leaves the symlink targets dangling. The parent mount resolves them.
# ============================================================================

# 2a: volume hostPath
sed -i 's|path: /opt/cni/bin|path: /var/lib/rancher/k3s/data|' "$TMP"
grep -q 'path: /var/lib/rancher/k3s/data' "$TMP" \
    || { echo "PATCH 2a FAILED: cnibin volume hostPath" >&2; exit 1; }
echo "  Patch 2a applied: cnibin volume hostPath"

# 2b: main container mountPath
sed -i 's|mountPath: /opt/cni/bin|mountPath: /var/lib/rancher/k3s/data|' "$TMP"
grep -q 'mountPath: /var/lib/rancher/k3s/data' "$TMP" \
    || { echo "PATCH 2b FAILED: main container mountPath" >&2; exit 1; }
echo "  Patch 2b applied: main container mountPath"

# 2c: initContainer mountPath
sed -i 's|mountPath: /host/opt/cni/bin|mountPath: /host/var/lib/rancher/k3s/data|' "$TMP"
grep -q 'mountPath: /host/var/lib/rancher/k3s/data' "$TMP" \
    || { echo "PATCH 2c FAILED: initContainer mountPath" >&2; exit 1; }
echo "  Patch 2c applied: initContainer mountPath"

# 2d: initContainer install target (-d flag)
#     After 2c, only the quoted -d argument still references /host/opt/cni/bin.
#     Target is .../data/cni (not .../data) because that's where the plugin binary
#     must land — it's the K3s bind-mount destination for bundled CNI plugins.
sed -i 's|"/host/opt/cni/bin"|"/host/var/lib/rancher/k3s/data/cni"|' "$TMP"
grep -q '/host/var/lib/rancher/k3s/data/cni' "$TMP" \
    || { echo "PATCH 2d FAILED: initContainer -d target" >&2; exit 1; }
echo "  Patch 2d applied: initContainer install target"

# ============================================================================
# Patch 3: binDir in daemon-config.json
#   Without this, Multus defaults to /opt/cni/bin when delegating to flannel/
#   host-device/static. Also written into the auto-generated 00-multus.conf.
#   Uses Python for robust JSON-aware insertion (tolerates key reordering).
# ============================================================================
python3 - "$TMP" <<'PYEOF'
import sys, re

f = sys.argv[1]
content = open(f).read()

if '"binDir"' in content:
    print("  Patch 3: binDir already present in upstream, skipping insertion")
    sys.exit(0)

patched = re.sub(
    r'("cniVersion":\s*"[^"]+",)',
    r'\1\n        "binDir": "/var/lib/rancher/k3s/data/cni",',
    content,
    count=1,
)
if '"binDir"' not in patched:
    print("PATCH 3 FAILED: could not insert binDir after cniVersion", file=sys.stderr)
    sys.exit(1)

open(f, "w").write(patched)
PYEOF
grep -q '"binDir"' "$TMP" \
    || { echo "PATCH 3 FAILED: binDir not in result" >&2; exit 1; }
echo "  Patch 3 applied: binDir in daemon-config.json"

# ============================================================================
# Patch 4: initContainer mountPropagation
#   Bidirectional → HostToContainer.
#   CRITICAL: Bidirectional causes bind mount stacking at .../data/cni on every
#   DaemonSet restart, burying K3s's original bind mount and breaking all CNI
#   plugin resolution cluster-wide. HostToContainer is sufficient — the
#   initContainer only copies files, it never creates mounts.
# ============================================================================
sed -i 's|mountPropagation: Bidirectional|mountPropagation: HostToContainer|' "$TMP"
if grep -q 'Bidirectional' "$TMP"; then
    echo "PATCH 4 FAILED: Bidirectional still present in file" >&2
    exit 1
fi
echo "  Patch 4 applied: mountPropagation HostToContainer"

# ============================================================================
# Patch 5: kube-multus memory request/limit 50Mi → 100Mi / 512Mi
#   Upstream sets request == limit == 50Mi. That holds in steady state but not
#   on a cold node boot: multus-daemon forks one delegate CNI process (flannel,
#   bridge, host-device) per pending CNI ADD, and ~20 pods coming up at once
#   blow the cgroup limit. The daemon is OOMKilled (exit 137), every ADD fails,
#   the pods stay in Unknown, multus goes CrashLoopBackOff and the same herd
#   hits it again on the next start — a self-sustaining deadlock that does not
#   resolve on its own. Observed 2026-07-28 on elite800 after a reboot.
#   Request stays low (scheduling), only the burst headroom is widened.
# ============================================================================
python3 - "$TMP" <<'PYEOF'
import re
import sys

f = sys.argv[1]
content = open(f).read()

# Both 50Mi occurrences belong to the kube-multus container (the initContainer
# has requests-only with 15Mi), so a positional replace is unambiguous:
# first = requests, second = limits.
patched, n = re.subn(r'memory: "50Mi"', 'memory: "100Mi"', content, count=1)
if n != 1:
    print("PATCH 5 FAILED: requests memory 50Mi not found", file=sys.stderr)
    sys.exit(1)
patched, n = re.subn(r'memory: "50Mi"', 'memory: "512Mi"', patched, count=1)
if n != 1:
    print("PATCH 5 FAILED: limits memory 50Mi not found", file=sys.stderr)
    sys.exit(1)

open(f, "w").write(patched)
PYEOF
grep -q 'memory: "512Mi"' "$TMP" \
    || { echo "PATCH 5 FAILED: 512Mi limit not in result" >&2; exit 1; }
echo "  Patch 5 applied: kube-multus memory headroom"

# ============================================================================
# Patch 6: pin the container image tag
#   Upstream hardcodes the rolling `snapshot-thick` tag (a build off master) in
#   EVERY manifest, release tags included — checking out v4.3.0 does not pin the
#   image. Without this patch any pod restart can pull a different master build.
# ============================================================================
sed -i "s|multus-cni:snapshot-thick|multus-cni:${MULTUS_REF}-thick|g" "$TMP"
if grep -q 'multus-cni:snapshot-thick' "$TMP"; then
    echo "PATCH 6 FAILED: snapshot-thick still present" >&2
    exit 1
fi
grep -q "multus-cni:${MULTUS_REF}-thick" "$TMP" \
    || { echo "PATCH 6 FAILED: pinned tag not in result" >&2; exit 1; }
echo "  Patch 6 applied: image tag pinned to ${MULTUS_REF}-thick"

# ============================================================================
# Prepend header with upstream SHA256 and patch documentation
# ============================================================================
cat > "$DEST" <<EOF
# Downloaded from:
#   ${UPSTREAM_URL}
# Upstream release: ${MULTUS_REF}
# Upstream SHA256: ${UPSTREAM_HASH}
# Patched by: roles/k8s_dgx/files/update_multus_manifest.sh
#
# K3s fixes applied:
#
# 1. cni volume hostPath: /etc/cni/net.d → /var/lib/rancher/k3s/agent/etc/cni/net.d
#    (K3s stores Flannel CNI config there).
#
# 2. cnibin volume hostPath: /opt/cni/bin → /var/lib/rancher/k3s/data
#    (the K3s data root, NOT /var/lib/rancher/k3s/data/cni — see note below).
#    Main container mountPath and initContainer mountPath also widened to match.
#    The initContainer install target (-d flag) stays at /host/.../data/cni because
#    that's where K3s bind-mounts the bundled CNI plugins from <hash>/bin.
#
#    Why /var/lib/rancher/k3s/data (parent) instead of /var/lib/rancher/k3s/data/cni:
#    K3s bind-mounts <hash>/bin → /var/lib/rancher/k3s/data/cni. Inside that directory,
#    CNI plugins are absolute symlinks (e.g. flannel → /var/lib/rancher/k3s/data/<hash>/bin/cni).
#    If only /var/lib/rancher/k3s/data/cni is mounted into the container, the symlink
#    targets (/var/lib/rancher/k3s/data/<hash>/bin/cni) are outside the mount and dangling.
#    Mounting the parent ensures symlink targets resolve inside the Multus daemon container.
#
# 3. binDir in daemon-config.json: set to /var/lib/rancher/k3s/data/cni
#    (without it, Multus defaults to /opt/cni/bin when delegating to flannel).
#    This value is also written into the auto-generated 00-multus.conf CNI config.
#
# 4. initContainer mountPropagation: Bidirectional → HostToContainer.
#    CRITICAL: Bidirectional causes bind mount stacking at /var/lib/rancher/k3s/data/cni
#    on each DaemonSet restart, burying K3s's original bind mount (from <hash>/bin) and
#    breaking all CNI plugin resolution cluster-wide. HostToContainer is sufficient —
#    the initContainer only writes files, it doesn't create mounts.
#
# 5. kube-multus memory: request 50Mi → 100Mi, limit 50Mi → 512Mi.
#    Upstream's request == limit == 50Mi survives steady state but not a cold node
#    boot: multus-daemon forks one delegate CNI process per pending ADD, and ~20
#    pods starting at once exceed the cgroup limit. The daemon is OOMKilled (137),
#    all ADDs fail, pods stay Unknown, multus CrashLoopBackOffs and meets the same
#    herd again — a deadlock that does not clear by itself (seen 2026-07-28 on
#    elite800 after a reboot). CPU stays at 100m.
#
# 6. image tag: snapshot-thick → ${MULTUS_REF}-thick.
#    Upstream hardcodes the rolling snapshot-thick tag in every manifest, release
#    tags included, so the release checkout alone does not pin the image. Without
#    this any pod restart could pull a different master build of the daemon.
#
# Extra CNI plugins (host-device, static) not bundled with K3s are installed by the
# dgx_prepare role (tag: cni) — see roles/dgx_prepare/tasks/cni_plugins.yml.
#
EOF
cat "$TMP" >> "$DEST"

echo ""
echo "Done. Patched manifest written to: $(basename "$DEST")"
echo "Review: git diff roles/k8s_dgx/files/multus-daemonset-thick.yml"
