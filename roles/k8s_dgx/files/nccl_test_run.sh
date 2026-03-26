#!/bin/bash
set -e

PEER_IP="${PEER_IP:?PEER_IP must be set}"
MASTER_ADDR="${MASTER_ADDR:?MASTER_ADDR must be set}"
MASTER_PORT="${MASTER_PORT:-48000}"
RANK="${RANK:?RANK must be set}"
WORLD_SIZE="${WORLD_SIZE:-2}"

echo "=== NCCL torch.distributed benchmark rank ${RANK} ==="
echo "MASTER_ADDR=${MASTER_ADDR}:${MASTER_PORT}"
echo "WORLD_SIZE=${WORLD_SIZE} RANK=${RANK}"
echo "NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME}"
echo ""

echo "Interfaces:"
python3 -c "
import fcntl, socket, struct, os
for iface in sorted(os.listdir('/sys/class/net/')):
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        ip = socket.inet_ntoa(fcntl.ioctl(s.fileno(), 0x8915, struct.pack('256s', iface.encode()))[20:24])
        with open(f'/sys/class/net/{iface}/mtu') as f:
            mtu = f.read().strip()
        with open(f'/sys/class/net/{iface}/address') as f:
            mac = f.read().strip()
        print(f'  {iface}: {ip}  mtu {mtu}  mac {mac}')
    except: pass
"
echo ""

echo "Warming ARP cache to ${PEER_IP}..."
for i in $(seq 1 30); do
    python3 -c "
import socket
try:
    socket.create_connection(('${PEER_IP}', 1), timeout=1)
except:
    pass
" 2>/dev/null
    sleep 0.5
    # check if ARP resolved
    python3 -c "
import subprocess, sys
out = subprocess.run(['cat', '/proc/net/arp'], capture_output=True, text=True).stdout
if '${PEER_IP}' in out and '00:00:00:00:00:00' not in [l.split()[3] for l in out.splitlines()[1:] if '${PEER_IP}' in l]:
    sys.exit(0)
sys.exit(1)
" 2>/dev/null && echo "ARP resolved after ${i} attempts" && break
done

echo "ARP table:"
cat /proc/net/arp 2>/dev/null || true
echo ""

echo "=== Starting benchmark ==="
exec python3 /scripts/nccl_bench.py
