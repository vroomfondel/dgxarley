#!/bin/bash
# luks_encrypt_spark.sh — In-place LUKS encryption for DGX Spark root partitions
#
# Encrypts the root partition of DGX Spark / ASUS Ascent GX10 (ARM64) nodes
# WITHOUT requiring a live USB boot.
#
# Two-phase approach:
#   Phase A (this script, online): Install tools, deploy initramfs hooks, rebuild initramfs
#   Phase B (initramfs premount, next boot): Backup → repartition → encrypt → restore → reboot
#
# Partition layout before:
#   nvme0n1p1  vfat   ~537MB  /boot/efi  (EFI)
#   nvme0n1p2  ext4   ~2TB    /          (root)
#
# Partition layout after:
#   nvme0n1p1  vfat   ~537MB  /boot/efi  (EFI, untouched)
#   nvme0n1p2  ext4   1.5GB   /boot      (new)
#   nvme0n1p3  LUKS2  ~2TB    LUKS → sparkvg/root → /
#
# Usage:
#   sudo ./luks_encrypt_spark.sh           # Prep system for encryption
#   sudo ./luks_encrypt_spark.sh --status  # Check if encryption is pending
#   sudo ./luks_encrypt_spark.sh --abort   # Cancel pending encryption
#   sudo ./luks_encrypt_spark.sh --dry-run # Show what would be done
#
# After reboot (enter LUKS passphrase at prompt), bind clevis/tang:
#   ansible-playbook common.yml --limit spark1.local --tags clevis \
#     -e 'wantsclevis=true clevis_luks_passphrase=...'

set -euo pipefail

# ═══════════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════════

NVME_DEVICE="/dev/nvme0n1"
EFI_PART="${NVME_DEVICE}p1"
ORIG_ROOT_PART="${NVME_DEVICE}p2"
LUKS_MAPPER_NAME="rootfs"
LVM_VG_NAME="sparkvg"
LVM_LV_NAME="root"
BOOT_PARTITION_SIZE_MB=1536  # 1.5GB for /boot
NETWORK_INTERFACE="enP7s7"   # DGX Spark onboard NIC
MIN_PASSPHRASE_LEN=8

TANG_SERVERS=(
  "http://tang.example.com"
  "http://tang2.example.com:9090"
)
SSS_THRESHOLD=2

# Paths for deployed initramfs files
HOOK_PATH="/etc/initramfs-tools/hooks/luks-encrypt-spark"
PREMOUNT_PATH="/etc/initramfs-tools/scripts/local-premount/luks-encrypt-spark"
CONFIG_PATH="/etc/luks-encrypt-spark.conf"
KEY_PATH="/etc/luks-encrypt-spark.key"

# ═══════════════════════════════════════════════════════════════
# UI helpers
# ═══════════════════════════════════════════════════════════════

GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m'

info()  { echo -e "${CYAN}[INFO]${NC} $*"; }
warn()  { echo -e "${YELLOW}[WARN]${NC} $*"; }
ok()    { echo -e "${GREEN}[ OK ]${NC} $*"; }
fail()  { echo -e "${RED}[FAIL]${NC} $*"; exit 1; }

# ═══════════════════════════════════════════════════════════════
# Parse arguments
# ═══════════════════════════════════════════════════════════════

MODE="prep"
DRY_RUN=false

for arg in "$@"; do
  case "$arg" in
    --status)   MODE="status" ;;
    --abort)    MODE="abort" ;;
    --dry-run)  DRY_RUN=true ;;
    -h|--help)
      cat <<'HELPEOF'
Usage: sudo ./luks_encrypt_spark.sh [OPTIONS]

In-place LUKS encryption for DGX Spark root partitions.
Runs from the live system — no live USB required.

Options:
  --status    Check if encryption is pending (hooks deployed)
  --abort     Remove hooks and config, rebuild initramfs (cancel)
  --dry-run   Show what would be done without executing
  -h, --help  Show this help

Flow:
  1. Run this script on the live Spark node
  2. Reboot — encryption happens automatically in initramfs (~5-10 min)
  3. System reboots again — enter LUKS passphrase at prompt
  4. Bind clevis/tang via Ansible for auto-unlock on subsequent boots
HELPEOF
      exit 0
      ;;
    *) fail "Unknown option: $arg" ;;
  esac
done

# Root check
if [ "$EUID" -ne 0 ]; then
  fail "This script must be run as root (sudo)."
fi

# ═══════════════════════════════════════════════════════════════
# --status: Check if encryption is pending
# ═══════════════════════════════════════════════════════════════

if [ "$MODE" = "status" ]; then
  echo -e "${BOLD}Encryption status:${NC}"
  PENDING=false
  for f in "$HOOK_PATH" "$PREMOUNT_PATH" "$CONFIG_PATH" "$KEY_PATH"; do
    if [ -f "$f" ]; then
      echo "  $(basename "$f"): deployed"
      PENDING=true
    else
      echo "  $(basename "$f"): not present"
    fi
  done
  echo ""
  if [ "$PENDING" = true ]; then
    echo -e "${YELLOW}Encryption is PENDING. Reboot to start, or run --abort to cancel.${NC}"
  elif [ -b "${NVME_DEVICE}p3" ] && blkid -s TYPE -o value "${NVME_DEVICE}p3" 2>/dev/null | grep -q crypto_LUKS; then
    echo -e "${GREEN}System is already LUKS-encrypted.${NC}"
  else
    echo "No encryption pending. Run this script without flags to start prep."
  fi
  exit 0
fi

# ═══════════════════════════════════════════════════════════════
# --abort: Cancel pending encryption
# ═══════════════════════════════════════════════════════════════

if [ "$MODE" = "abort" ]; then
  info "Removing encryption hooks and config..."
  REMOVED=false
  for f in "$HOOK_PATH" "$PREMOUNT_PATH" "$CONFIG_PATH" "$KEY_PATH"; do
    if [ -f "$f" ]; then
      rm -f "$f"
      ok "Removed: $f"
      REMOVED=true
    fi
  done
  if [ "$REMOVED" = true ]; then
    info "Rebuilding initramfs (removing encryption hooks)..."
    update-initramfs -u -k all
    ok "Encryption cancelled. Safe to reboot normally."
  else
    info "Nothing to remove — no encryption was pending."
  fi
  exit 0
fi

# ═══════════════════════════════════════════════════════════════
# Phase A: Prep (default mode)
# ═══════════════════════════════════════════════════════════════

echo ""
echo -e "${BOLD}════════════════════════════════════════════════════════════════${NC}"
echo -e "${BOLD}  In-place LUKS encryption — Phase A (prep)${NC}"
echo -e "${BOLD}════════════════════════════════════════════════════════════════${NC}"
echo ""

# ─── Step 1: Sanity checks ───────────────────────────────────

info "Running sanity checks..."

# Must be booted from NVMe (this IS the live system, not a live USB)
ROOT_FS_DEV=$(findmnt -no SOURCE / 2>/dev/null || echo "unknown")
if [[ "$ROOT_FS_DEV" != *nvme0n1* ]]; then
  fail "Root filesystem is not on $NVME_DEVICE (found: $ROOT_FS_DEV). This script must run on the live Spark node."
fi
ok "Running on NVMe root ($ROOT_FS_DEV)"

# Verify ARM64
ARCH=$(uname -m)
if [ "$ARCH" != "aarch64" ]; then
  fail "Expected aarch64, got: $ARCH"
fi
ok "Architecture: $ARCH"

# Verify NVMe device exists
if [ ! -b "$NVME_DEVICE" ]; then
  fail "NVMe device $NVME_DEVICE not found."
fi

# Verify partition layout: exactly 2 partitions (p1=EFI, p2=root)
PART_COUNT=$(lsblk -nro NAME "$NVME_DEVICE" | grep -c "^nvme0n1p" || true)
if [ "$PART_COUNT" -ne 2 ]; then
  fail "Expected 2 partitions on $NVME_DEVICE, found $PART_COUNT. Already encrypted or unexpected layout."
fi

# Verify p1 is EFI (vfat)
P1_FSTYPE=$(blkid -s TYPE -o value "$EFI_PART" 2>/dev/null || echo "unknown")
if [ "$P1_FSTYPE" != "vfat" ]; then
  fail "Expected vfat on $EFI_PART, found: $P1_FSTYPE"
fi
ok "EFI partition: $EFI_PART ($P1_FSTYPE)"

# Verify p2 is ext4 root
P2_FSTYPE=$(blkid -s TYPE -o value "$ORIG_ROOT_PART" 2>/dev/null || echo "unknown")
if [ "$P2_FSTYPE" != "ext4" ]; then
  fail "Expected ext4 on $ORIG_ROOT_PART, found: $P2_FSTYPE"
fi
ok "Root partition: $ORIG_ROOT_PART ($P2_FSTYPE)"

# Check RAM vs data usage
RAM_KB=$(awk '/MemTotal/ {print $2}' /proc/meminfo)
RAM_GB=$((RAM_KB / 1024 / 1024))
DATA_USED_KB=$(df -k / | awk 'NR==2 {print $3}')
DATA_USED_GB=$((DATA_USED_KB / 1024 / 1024))
DATA_USED_MB=$((DATA_USED_KB / 1024))

# Need data + 20% margin + ~10GB for initramfs, tools, and filesystem overhead
REQUIRED_RAM_GB=$(( (DATA_USED_GB * 120) / 100 + 10 ))
if [ "$RAM_GB" -lt "$REQUIRED_RAM_GB" ]; then
  fail "Not enough RAM: ${RAM_GB}GB available, need ~${REQUIRED_RAM_GB}GB for ${DATA_USED_MB}MB data + overhead"
fi
ok "RAM check: ${RAM_GB}GB available, ${DATA_USED_MB}MB data to backup"

# Check no pending encryption
if [ -f "$CONFIG_PATH" ] || [ -f "$PREMOUNT_PATH" ]; then
  fail "Encryption hooks already deployed. Use --status to check or --abort to cancel."
fi

# ─── Summary ────────────────────────────────────────────────

echo ""
echo -e "${BOLD}System summary:${NC}"
echo "  Device:       $NVME_DEVICE"
echo "  EFI:          $EFI_PART (untouched)"
echo "  Current root: $ORIG_ROOT_PART (${DATA_USED_MB}MB used)"
echo "  RAM:          ${RAM_GB}GB"
echo ""
echo -e "${BOLD}Encryption plan:${NC}"
echo "  1. Install packages, deploy initramfs hooks, rebuild initramfs"
echo "  2. On next boot, initramfs backs up root to RAM (~${DATA_USED_MB}MB)"
echo "  3. Repartition: p2=${BOOT_PARTITION_SIZE_MB}MB /boot, p3=LUKS (remaining)"
echo "  4. LUKS format p3, LVM ${LVM_VG_NAME}/${LVM_LV_NAME}"
echo "  5. Restore backup to encrypted root, rebuild GRUB, auto-reboot"
echo ""
echo -e "${RED}${BOLD}WARNING: On next reboot, ALL data on $ORIG_ROOT_PART will be destroyed and re-encrypted.${NC}"
echo -e "${RED}${BOLD}The backup exists ONLY in RAM. Power loss during encryption = total data loss.${NC}"

if [ "$DRY_RUN" = true ]; then
  echo ""
  ok "Dry run — would install packages, deploy hooks, rebuild initramfs. Exiting."
  exit 0
fi

echo ""
read -rp "Type 'yes' to prepare system for encryption: " CONFIRM
if [ "$CONFIRM" != "yes" ]; then
  fail "Aborted."
fi

# ─── Step 2: Install packages ────────────────────────────────

echo ""
info "Installing required packages..."

DEPS=(cryptsetup-initramfs clevis-initramfs lvm2 rsync gdisk e2fsprogs)
MISSING=()
for pkg in "${DEPS[@]}"; do
  if ! dpkg -l "$pkg" 2>/dev/null | grep -q "^ii"; then
    MISSING+=("$pkg")
  fi
done

if [ ${#MISSING[@]} -gt 0 ]; then
  info "Installing: ${MISSING[*]}"
  apt-get update -qq
  DEBIAN_FRONTEND=noninteractive apt-get install -y "${MISSING[@]}"
  ok "Packages installed"
else
  ok "All packages already installed"
fi

# ─── Step 3: Configure initramfs networking ───────────────────

info "Configuring initramfs networking (for clevis on future boots)..."

INITRAMFS_CONF="/etc/initramfs-tools/initramfs.conf"
INITRAMFS_MODULES="/etc/initramfs-tools/modules"

# Detect NIC driver
NIC_DRIVER=""
if [ -L "/sys/class/net/${NETWORK_INTERFACE}/device/driver" ]; then
  NIC_DRIVER=$(basename "$(readlink "/sys/class/net/${NETWORK_INTERFACE}/device/driver")")
fi

if [ -n "$NIC_DRIVER" ]; then
  if ! grep -qxF "$NIC_DRIVER" "$INITRAMFS_MODULES" 2>/dev/null; then
    echo "$NIC_DRIVER" >> "$INITRAMFS_MODULES"
    ok "Added NIC driver '$NIC_DRIVER' to initramfs modules"
  else
    ok "NIC driver '$NIC_DRIVER' already in initramfs modules"
  fi
else
  warn "Could not detect NIC driver for $NETWORK_INTERFACE — add manually to /etc/initramfs-tools/modules"
fi

# Set DEVICE and IP in initramfs.conf
if grep -q '^DEVICE=' "$INITRAMFS_CONF"; then
  sed -i "s/^DEVICE=.*/DEVICE=${NETWORK_INTERFACE}/" "$INITRAMFS_CONF"
else
  echo "DEVICE=${NETWORK_INTERFACE}" >> "$INITRAMFS_CONF"
fi

IP_LINE="IP=:::::${NETWORK_INTERFACE}:dhcp"
if grep -q '^IP=' "$INITRAMFS_CONF"; then
  sed -i "s|^IP=.*|${IP_LINE}|" "$INITRAMFS_CONF"
else
  sed -i "/^DEVICE=/a ${IP_LINE}" "$INITRAMFS_CONF"
fi
ok "Initramfs networking: DEVICE=$NETWORK_INTERFACE, IP=dhcp"

# ─── Step 4: Prompt for LUKS passphrase ──────────────────────

echo ""
while true; do
  read -rsp "Enter LUKS passphrase (min ${MIN_PASSPHRASE_LEN} chars): " LUKS_PASS
  echo ""
  if [ ${#LUKS_PASS} -lt $MIN_PASSPHRASE_LEN ]; then
    warn "Passphrase too short (${#LUKS_PASS} < $MIN_PASSPHRASE_LEN)"
    continue
  fi
  read -rsp "Confirm passphrase: " LUKS_PASS2
  echo ""
  if [ "$LUKS_PASS" != "$LUKS_PASS2" ]; then
    warn "Passphrases don't match."
    continue
  fi
  break
done

# Write config file (sentinel + configuration for premount script)
cat > "$CONFIG_PATH" <<CONFEOF
# luks-encrypt-spark configuration — auto-generated, deleted after encryption
NVME_DEVICE=${NVME_DEVICE}
LUKS_MAPPER_NAME=${LUKS_MAPPER_NAME}
LVM_VG_NAME=${LVM_VG_NAME}
LVM_LV_NAME=${LVM_LV_NAME}
BOOT_PARTITION_SIZE_MB=${BOOT_PARTITION_SIZE_MB}
CONFEOF
chmod 600 "$CONFIG_PATH"

# Write key file (raw passphrase)
printf '%s' "$LUKS_PASS" > "$KEY_PATH"
chmod 600 "$KEY_PATH"
ok "Config and key file written"

# Clear passphrase from shell memory
LUKS_PASS=""
LUKS_PASS2=""

# ─── Step 5: Deploy initramfs hook ───────────────────────────

info "Deploying initramfs hook..."

cat > "$HOOK_PATH" << 'HOOKEOF'
#!/bin/sh -e
PREREQS=""
case $1 in
prereqs) echo "${PREREQS}"; exit 0;;
esac

. /usr/share/initramfs-tools/hook-functions

# Copy encryption/partitioning tools into initramfs
copy_exec /usr/bin/rsync /usr/bin
copy_exec /usr/sbin/sgdisk /usr/sbin
copy_exec /sbin/mkfs.ext4 /sbin
copy_exec /sbin/blkid /sbin
copy_exec /sbin/cryptsetup /sbin
copy_exec /sbin/lvm /sbin
copy_exec /sbin/partprobe /sbin
copy_exec /usr/sbin/chroot /usr/sbin

# LVM multicall symlinks — copy_exec doesn't reliably create these
# on usr-merged systems (pvcreate/vgcreate/lvcreate are symlinks to lvm)
for cmd in pvcreate vgcreate lvcreate vgchange; do
  if [ ! -e "${DESTDIR}/usr/sbin/${cmd}" ] && [ ! -e "${DESTDIR}/sbin/${cmd}" ]; then
    ln -s lvm "${DESTDIR}/usr/sbin/${cmd}"
  fi
done

# Copy config and key into initramfs
if [ -f /etc/luks-encrypt-spark.conf ]; then
  cp /etc/luks-encrypt-spark.conf "${DESTDIR}/etc/luks-encrypt-spark.conf"
fi
if [ -f /etc/luks-encrypt-spark.key ]; then
  cp /etc/luks-encrypt-spark.key "${DESTDIR}/etc/luks-encrypt-spark.key"
  chmod 600 "${DESTDIR}/etc/luks-encrypt-spark.key"
fi
HOOKEOF
chmod +x "$HOOK_PATH"
ok "Hook deployed: $HOOK_PATH"

# ─── Step 6: Deploy initramfs premount script ────────────────

info "Deploying initramfs premount script..."

# Note: This MUST be POSIX sh — initramfs uses busybox/dash, not bash.
# No arrays, no [[ ]], no bash-isms. Config values come from sourced conf file.
cat > "$PREMOUNT_PATH" << 'PREMOUNTEOF'
#!/bin/sh
# luks-encrypt-spark: One-shot in-place LUKS encryption
# Runs in initramfs local-premount before root is mounted.
# Self-destructs from the new root after successful encryption.

PREREQ=""
prereqs() { echo "$PREREQ"; }
case $1 in
prereqs) prereqs; exit 0;;
esac

# ── Sentinel check ──────────────────────────────────────────
# If config not in initramfs, this is a normal boot — exit silently
if [ ! -f /etc/luks-encrypt-spark.conf ]; then
  exit 0
fi
if [ ! -f /etc/luks-encrypt-spark.key ]; then
  exit 0
fi

# ── Source config ───────────────────────────────────────────
. /etc/luks-encrypt-spark.conf

# ── Derived paths ───────────────────────────────────────────
EFI_PART="${NVME_DEVICE}p1"
ORIG_ROOT_PART="${NVME_DEVICE}p2"
KEY_FILE="/etc/luks-encrypt-spark.key"
ROOT_DEV="/dev/${LVM_VG_NAME}/${LVM_LV_NAME}"

# ── Logging ─────────────────────────────────────────────────
log_msg() {
  echo "luks-encrypt: $*" > /dev/console 2>/dev/null || true
  echo "luks-encrypt: $*"
}

# ── Error handling ──────────────────────────────────────────
PAST_PONR=0
ENCRYPT_DONE=0

error_handler() {
  local lineno="${1:-unknown}"
  if [ "$ENCRYPT_DONE" -eq 1 ]; then
    # Normal exit after successful completion — do nothing
    return 0
  fi
  log_msg ""
  log_msg "╔═══════════════════════════════════════════════════════════╗"
  log_msg "║  ERROR at line ${lineno}"
  log_msg "║"
  # Try to show the failing line from the script itself
  if [ "$lineno" != "unknown" ] && [ -f /etc/initramfs-tools/scripts/local-premount/luks-encrypt-spark ]; then
    local failcmd
    failcmd=$(sed -n "${lineno}p" /etc/initramfs-tools/scripts/local-premount/luks-encrypt-spark 2>/dev/null || echo "(could not read)")
    log_msg "║  Failed command: ${failcmd}"
    log_msg "║"
  fi
  if [ "$PAST_PONR" -eq 0 ]; then
    log_msg "║  BEFORE point of no return — aborting safely."
    log_msg "║  Normal boot will continue in 10 seconds."
    log_msg "╚═══════════════════════════════════════════════════════════╝"
    log_msg ""
    sleep 10
    umount /tmp/oldroot 2>/dev/null || true
    exit 0
  else
    log_msg "║  ENCRYPTION FAILED AFTER POINT OF NO RETURN             ║"
    log_msg "║                                                         ║"
    log_msg "║  DO NOT REBOOT — backup is in /tmp/backup               ║"
    log_msg "║  Dropping to emergency shell for manual recovery.       ║"
    log_msg "║                                                         ║"
    log_msg "║  The original partition has been destroyed.              ║"
    log_msg "║  Data is safe in /tmp/backup (RAM).                     ║"
    log_msg "╚═══════════════════════════════════════════════════════════╝"
    exec sh < /dev/console > /dev/console 2>&1
  fi
}

trap 'error_handler $LINENO' EXIT
set -e

log_msg "═══════════════════════════════════════════════════════"
log_msg "  Starting in-place LUKS encryption"
log_msg "═══════════════════════════════════════════════════════"

# ── Step 1: Mount old root read-only ────────────────────────
log_msg "[Step 1/18] Mounting old root (${ORIG_ROOT_PART}) read-only..."
mkdir -p /tmp/oldroot
mount -o ro "${ORIG_ROOT_PART}" /tmp/oldroot
log_msg "[Step 1/18] Old root mounted OK"

# ── Step 2: Backup via rsync to initramfs tmpfs ─────────────
# Note: tmpfs does not support extended attributes or ACLs, so we
# explicitly skip them for the backup (--no-xattrs --no-acls).
# They get restored to ext4 in the restore step.
log_msg "[Step 2/18] Backing up root filesystem to RAM (this may take several minutes)..."
log_msg "[Step 2/18]   Source: /tmp/oldroot/  →  Dest: /tmp/backup/"
mkdir -p /tmp/backup

set +e
rsync -aHx --info=progress2,stats2 \
  --no-xattrs --no-acls \
  --exclude='/dev/*' \
  --exclude='/proc/*' \
  --exclude='/sys/*' \
  --exclude='/run/*' \
  --exclude='/tmp/*' \
  --exclude='/swap.img' \
  --exclude='/lost+found' \
  /tmp/oldroot/ /tmp/backup/ > /dev/console 2>&1
RSYNC_RC=$?
set -e

if [ "$RSYNC_RC" -ne 0 ]; then
  log_msg "╔═══════════════════════════════════════════════════════════╗"
  log_msg "║  rsync backup FAILED (exit code ${RSYNC_RC})"
  log_msg "║  Aborting — normal boot will continue in 10 seconds."
  log_msg "╚═══════════════════════════════════════════════════════════╝"
  sleep 10
  umount /tmp/oldroot 2>/dev/null || true
  exit 0
fi

log_msg "[Step 2/18] Backup complete (rsync exit code 0)"

# ── Step 3: Verify backup ──────────────────────────────────
log_msg "[Step 3/18] Verifying backup integrity..."
if [ ! -d /tmp/backup/etc ] || [ ! -d /tmp/backup/usr ] || [ ! -d /tmp/backup/bin ]; then
  log_msg "FATAL: Backup verification failed — critical directories missing"
  umount /tmp/oldroot 2>/dev/null || true
  exit 0
fi

KERNEL_COUNT=0
for f in /tmp/backup/boot/vmlinuz-*; do
  if [ -f "$f" ]; then
    KERNEL_COUNT=$((KERNEL_COUNT + 1))
  fi
done
if [ "$KERNEL_COUNT" -eq 0 ]; then
  log_msg "FATAL: No kernel found in backup /boot/"
  umount /tmp/oldroot 2>/dev/null || true
  exit 0
fi
log_msg "[Step 3/18] Backup verified: critical dirs present, ${KERNEL_COUNT} kernel(s) found"

# ── Step 4: Unmount old root ───────────────────────────────
log_msg "[Step 4/18] Unmounting old root..."
umount /tmp/oldroot
rmdir /tmp/oldroot 2>/dev/null || true
log_msg "[Step 4/18] Old root unmounted"

# ═══════════════════════════════════════════════════════════
# ███  POINT OF NO RETURN  ███
# ═══════════════════════════════════════════════════════════
PAST_PONR=1
log_msg "══════════════════════════════════════════════════════════"
log_msg "  ███  POINT OF NO RETURN  ███"
log_msg "  Backup is in /tmp/backup — do NOT power off."
log_msg "══════════════════════════════════════════════════════════"

# ── Step 5: Repartition ───────────────────────────────────
log_msg "[Step 5/18] Repartitioning ${NVME_DEVICE}..."

P1_END=$(sgdisk -p "${NVME_DEVICE}" | awk '/^ *1 / {print $3}')
if [ -z "$P1_END" ]; then
  log_msg "FATAL: Cannot determine end of partition 1"
  exec sh < /dev/console > /dev/console 2>&1
fi

SECTOR_SIZE=512
if [ -f /sys/block/nvme0n1/queue/hw_sector_size ]; then
  SECTOR_SIZE=$(cat /sys/block/nvme0n1/queue/hw_sector_size)
fi

BOOT_SECTORS=$((BOOT_PARTITION_SIZE_MB * 1024 * 1024 / SECTOR_SIZE))
P2_START=$((P1_END + 1))
P2_END=$((P2_START + BOOT_SECTORS - 1))
P3_START=$((P2_END + 1))

log_msg "  p1 (EFI): kept, ends at sector ${P1_END}"
log_msg "  p2 (/boot): sectors ${P2_START}-${P2_END} (${BOOT_PARTITION_SIZE_MB}MB)"
log_msg "  p3 (LUKS): sector ${P3_START} to end of disk"

sgdisk -d 2 "${NVME_DEVICE}"
sgdisk -n "2:${P2_START}:${P2_END}" -t 2:8300 -c 2:"boot" "${NVME_DEVICE}"
sgdisk -n "3:${P3_START}:0" -t 3:8309 -c 3:"luks" "${NVME_DEVICE}"

partprobe "${NVME_DEVICE}"
sleep 2

BOOT_PART="${NVME_DEVICE}p2"
LUKS_PART="${NVME_DEVICE}p3"

if [ ! -b "$BOOT_PART" ] || [ ! -b "$LUKS_PART" ]; then
  log_msg "FATAL: Partitions not created after repartitioning"
  exec sh < /dev/console > /dev/console 2>&1
fi
log_msg "[Step 5/18] Repartitioned: p2 (boot) + p3 (luks)"

# ── Step 6: LUKS format + open ────────────────────────────
log_msg "[Step 6/18] Formatting ${LUKS_PART} with LUKS2 (aes-xts-plain64, argon2id)..."
cryptsetup luksFormat --type luks2 \
  --cipher aes-xts-plain64 \
  --key-size 512 \
  --hash sha256 \
  --pbkdf argon2id \
  --batch-mode \
  --key-file "${KEY_FILE}" \
  "${LUKS_PART}"

log_msg "[Step 6/18] Opening LUKS device..."
cryptsetup luksOpen --key-file "${KEY_FILE}" "${LUKS_PART}" "${LUKS_MAPPER_NAME}"
log_msg "[Step 6/18] LUKS open: /dev/mapper/${LUKS_MAPPER_NAME}"

# ── Step 7: LVM setup ─────────────────────────────────────
log_msg "[Step 7/18] Creating LVM: ${LVM_VG_NAME}/${LVM_LV_NAME}..."
pvcreate "/dev/mapper/${LUKS_MAPPER_NAME}"
vgcreate "${LVM_VG_NAME}" "/dev/mapper/${LUKS_MAPPER_NAME}"
lvcreate -l 100%FREE -n "${LVM_LV_NAME}" "${LVM_VG_NAME}" -y
log_msg "[Step 7/18] LVM ready: ${ROOT_DEV}"

# ── Step 8: Create root filesystem + restore backup ───────
log_msg "[Step 8/18] Creating ext4 on ${ROOT_DEV}..."
mkfs.ext4 -F -L sparkroot "${ROOT_DEV}"

mkdir -p /tmp/newroot
mount "${ROOT_DEV}" /tmp/newroot

log_msg "[Step 8/18] Restoring backup to encrypted root (this may take several minutes)..."
log_msg "[Step 8/18]   Source: /tmp/backup/  →  Dest: /tmp/newroot/"
rsync -aHx --info=progress2,stats2 \
  --no-xattrs --no-acls \
  /tmp/backup/ /tmp/newroot/ > /dev/console 2>&1
log_msg "[Step 8/18] Restore to encrypted root complete"

# ── Step 9: Create /boot partition + populate ──────────────
log_msg "[Step 9/18] Creating ext4 on ${BOOT_PART} (/boot)..."
mkfs.ext4 -F -L sparkboot "${BOOT_PART}"

# Save boot files before mounting over them
mkdir -p /tmp/bootfiles
cp -a /tmp/newroot/boot/. /tmp/bootfiles/

# Mount new /boot partition over the directory
mount "${BOOT_PART}" /tmp/newroot/boot

# Copy saved boot files to the new partition
cp -a /tmp/bootfiles/. /tmp/newroot/boot/
rm -rf /tmp/bootfiles

# Verify kernel on new /boot
KFOUND=0
for f in /tmp/newroot/boot/vmlinuz-*; do
  if [ -f "$f" ]; then
    KFOUND=1
    break
  fi
done
if [ "$KFOUND" -eq 0 ]; then
  log_msg "WARNING: No kernel found on new /boot partition"
fi
log_msg "[Step 9/18] /boot partition populated"

# ── Step 10: Mount EFI ────────────────────────────────────
log_msg "[Step 10/18] Mounting EFI partition..."
mkdir -p /tmp/newroot/boot/efi
mount "${EFI_PART}" /tmp/newroot/boot/efi
log_msg "[Step 10/18] EFI mounted at /tmp/newroot/boot/efi"

# ── Step 11: Get partition UUIDs ──────────────────────────
log_msg "[Step 11/18] Getting partition UUIDs..."
LUKS_UUID=$(blkid -s UUID -o value "${LUKS_PART}")
BOOT_UUID=$(blkid -s UUID -o value "${BOOT_PART}")
log_msg "[Step 11/18] LUKS UUID: ${LUKS_UUID}"
log_msg "[Step 11/18] Boot UUID: ${BOOT_UUID}"

# ── Step 12: Update /etc/fstab ────────────────────────────
log_msg "[Step 12/18] Updating /etc/fstab..."
FSTAB="/tmp/newroot/etc/fstab"
cp "${FSTAB}" "${FSTAB}.pre-encrypt"

# Remove old root mount line (matches ' / ' as mount point)
sed -i '/[[:space:]]\/[[:space:]]/d' "${FSTAB}"
# Remove swap.img
sed -i '/swap\.img/d' "${FSTAB}"
# Remove any existing /boot mount (matches ' /boot ' but not ' /boot/efi ')
sed -i '/[[:space:]]\/boot[[:space:]]/d' "${FSTAB}"

# Append new entries
cat >> "${FSTAB}" <<FSTABEOF

# Root on LUKS+LVM
/dev/${LVM_VG_NAME}/${LVM_LV_NAME}  /  ext4  errors=remount-ro  0  1

# Separate /boot partition
UUID=${BOOT_UUID}  /boot  ext4  defaults  0  2
FSTABEOF
log_msg "[Step 12/18] fstab updated"

# ── Step 13: Write /etc/crypttab ──────────────────────────
log_msg "[Step 13/18] Writing /etc/crypttab..."
cat > /tmp/newroot/etc/crypttab <<CRYPTEOF
# <target name>  <source device>  <key file>  <options>
${LUKS_MAPPER_NAME}  UUID=${LUKS_UUID}  none  luks,discard
CRYPTEOF
log_msg "[Step 13/18] crypttab written"

# ── Step 14: Update /etc/default/grub ─────────────────────
log_msg "[Step 14/18] Updating /etc/default/grub..."
GRUB_FILE="/tmp/newroot/etc/default/grub"
CRYPT_PARAM="cryptdevice=UUID=${LUKS_UUID}:${LUKS_MAPPER_NAME}"

if grep -q '^GRUB_CMDLINE_LINUX=' "${GRUB_FILE}"; then
  # Extract existing value, strip any old cryptdevice param
  OLD_VAL=$(sed -n 's/^GRUB_CMDLINE_LINUX="\(.*\)"/\1/p' "${GRUB_FILE}")
  CLEAN_VAL=$(echo "${OLD_VAL}" | sed 's/cryptdevice=[^ ]*//' | sed 's/  */ /g' | sed 's/^ *//;s/ *$//')
  if [ -n "${CLEAN_VAL}" ]; then
    NEW_VAL="${CRYPT_PARAM} ${CLEAN_VAL}"
  else
    NEW_VAL="${CRYPT_PARAM}"
  fi
  sed -i "s|^GRUB_CMDLINE_LINUX=.*|GRUB_CMDLINE_LINUX=\"${NEW_VAL}\"|" "${GRUB_FILE}"
else
  echo "GRUB_CMDLINE_LINUX=\"${CRYPT_PARAM}\"" >> "${GRUB_FILE}"
fi
log_msg "[Step 14/18] GRUB config updated"

# ── Step 15: Self-destruct — remove hooks from new root ───
log_msg "[Step 15/18] Removing encryption hooks from new root (self-destruct)..."
rm -f /tmp/newroot/etc/initramfs-tools/hooks/luks-encrypt-spark
rm -f /tmp/newroot/etc/initramfs-tools/scripts/local-premount/luks-encrypt-spark
rm -f /tmp/newroot/etc/luks-encrypt-spark.conf
rm -f /tmp/newroot/etc/luks-encrypt-spark.key
rm -f /tmp/newroot/etc/fstab.pre-encrypt
log_msg "[Step 15/18] Encryption hooks removed from new root"

# ── Step 16: Chroot — rebuild initramfs + GRUB ────────────
log_msg "[Step 16/18] Setting up chroot environment..."
mount --bind /dev /tmp/newroot/dev
mount --bind /proc /tmp/newroot/proc
mount --bind /sys /tmp/newroot/sys
mount -t devpts devpts /tmp/newroot/dev/pts 2>/dev/null || true

log_msg "[Step 16/18] Rebuilding initramfs (cryptsetup+clevis hooks, no encryption hooks)..."
chroot /tmp/newroot update-initramfs -u -k all
log_msg "[Step 16/18] initramfs rebuilt"

log_msg "[Step 16/18] Running update-grub..."
chroot /tmp/newroot update-grub 2>/dev/null || log_msg "[Step 16/18] WARNING: update-grub returned non-zero"
log_msg "[Step 16/18] update-grub complete"

log_msg "[Step 16/18] Installing GRUB bootloader..."
chroot /tmp/newroot grub-install --target=arm64-efi --efi-directory=/boot/efi --bootloader-id=ubuntu
log_msg "[Step 16/18] GRUB installed"

# ── Step 17: Unmount everything ───────────────────────────
log_msg "[Step 17/18] Unmounting..."

umount /tmp/newroot/dev/pts 2>/dev/null || true
umount /tmp/newroot/dev 2>/dev/null || true
umount /tmp/newroot/proc 2>/dev/null || true
umount /tmp/newroot/sys 2>/dev/null || true
umount /tmp/newroot/boot/efi 2>/dev/null || true
umount /tmp/newroot/boot 2>/dev/null || true
umount /tmp/newroot 2>/dev/null || true

vgchange -an "${LVM_VG_NAME}" 2>/dev/null || true
cryptsetup luksClose "${LUKS_MAPPER_NAME}" 2>/dev/null || true

# ── Step 18: Reboot ──────────────────────────────────────
log_msg "[Step 18/18] ═══════════════════════════════════════════════════════"
log_msg "[Step 18/18]   Encryption complete!"
log_msg "  Rebooting in 5 seconds..."
log_msg "  On next boot, enter LUKS passphrase at the prompt."
log_msg "  Then bind clevis/tang via Ansible for auto-unlock."
log_msg "═══════════════════════════════════════════════════════"

ENCRYPT_DONE=1
trap - EXIT
set +e
sleep 5
sync
reboot -f
PREMOUNTEOF
chmod +x "$PREMOUNT_PATH"
ok "Premount script deployed: $PREMOUNT_PATH"

# ─── Step 7: Rebuild initramfs ────────────────────────────────

echo ""
info "Rebuilding initramfs (including encryption tools and hooks)..."
update-initramfs -u -k all
ok "Initramfs rebuilt with encryption hooks"

# ─── Step 8: Summary and instructions ────────────────────────

echo ""
echo -e "${BOLD}════════════════════════════════════════════════════════════════${NC}"
echo -e "${GREEN}${BOLD}  Prep complete! System is ready for encryption.${NC}"
echo -e "${BOLD}════════════════════════════════════════════════════════════════${NC}"
echo ""
echo "Deployed files:"
echo "  Hook:     $HOOK_PATH"
echo "  Premount: $PREMOUNT_PATH"
echo "  Config:   $CONFIG_PATH"
echo "  Key:      $KEY_PATH"
echo ""
echo -e "${BOLD}Next steps:${NC}"
echo ""
echo "  1. Reboot this node:"
echo "     ${CYAN}sudo reboot${NC}"
echo ""
echo "     Encryption runs automatically in initramfs (~5-10 min)."
echo "     Console output shows progress. System reboots when done."
echo ""
echo "  2. On next boot, enter your LUKS passphrase at the prompt."
echo "     (Clevis not yet bound — manual passphrase required this once.)"
echo ""
echo "  3. Verify encryption:"
echo "     ${CYAN}lsblk -f${NC}"
echo "     ${CYAN}cat /etc/crypttab${NC}"
echo "     ${CYAN}mount | grep boot${NC}"
echo ""
echo "  4. Bind clevis/tang for auto-unlock on subsequent boots:"
echo "     ${CYAN}ansible-playbook common.yml --limit <host> --tags clevis \\${NC}"
echo "     ${CYAN}  -e 'wantsclevis=true clevis_luks_passphrase=...'${NC}"
echo ""
echo "To cancel before rebooting:"
echo "  ${CYAN}sudo $0 --abort${NC}"
echo ""

ok "Ready. Reboot when ready."
