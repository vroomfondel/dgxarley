# NCCL Transport Test Plan — RoCE vs Socket vs Host-Network

## Goal

Determine why RoCE is not delivering RDMA throughput (2.2 GB/s observed vs
20+ GB/s expected on 200 GbE ConnectX-7), and whether it can be fixed.

## Results log

### Baseline (2026-04-12 13:50)

| Transport | NCCL_IB_HCA | Peak bus BW | Actual NCCL transport | Notes |
|-----------|-------------|-------------|----------------------|-------|
| socket (VF7) | — | 2.12 GB/s | NET/Socket | NCCL_NET=Socket, NCCL_IB_DISABLE=1 |
| "roce" (VF7) | rocep1s0f0v7 | 2.20 GB/s | NET/Socket (fallback) | IB init failed, see below |

Both are ~2.1 GB/s → NCCL is falling back to TCP socket in the RoCE test.

### Test 1.1 → 2.2: RoCE debugging sequence (2026-04-12 13:30–15:00)

| Test | Config | NCCL transport | Peak BW | Finding |
|------|--------|---------------|---------|---------|
| 1.1 | VF7, non-privileged | Socket (fallback) | 2.20 GB/s | `NET/IB: No device found` — `/dev/infiniband/` not visible in container |
| 2.2a | PF hostNetwork, non-privileged | Socket (fallback) | 2.19 GB/s | Same `No device found` — hostNetwork alone doesn't expose IB char devices |
| 2.2b | PF hostNetwork, **privileged** | **IB/RoCE** | crash | `ibv_modify_qp failed: Network unreachable` — VF GID table has no IPv4 entries |
| fix | Added IPv4 addresses to VF netplan | — | — | RoCE GID table populated: `::ffff:10.10.100.x` at GID index 3 |
| 2.2c | PF hostNetwork, privileged, GIDs fixed | **IB/RoCE** | crash | QP connect uses VF sub-devices whose GIDs were still link-local |
| 1.1b | VF7, **privileged**, GIDs fixed | **IB/RoCE** | **9.78 GB/s** | All channels `via NET/IB/0` — first successful RoCE run |

### Final result (2026-04-12 15:15)

| Transport | Peak bus BW | Latency @4KB | Speedup vs socket |
|-----------|-------------|--------------|-------------------|
| socket (VF7) | 2.12 GB/s | 209 µs | 1× |
| **RoCE (VF7)** | **9.78 GB/s** | **40 µs** | **4.6×** |
| Theorie 200 GbE | ~25 GB/s | — | — |

9.78 GB/s = ~39% of 200 GbE line rate. The gap to theoretical is likely
from missing PFC/DCBX lossless Ethernet config, no GPU Direct RDMA (GDR),
and default NCCL tuning parameters.

### SGLang inference benchmark: RoCE vs Socket (2026-04-12 ~16:30)

Model: nvidia/Qwen3-235B-A22B-NVFP4, EP=4, TP=4, flashinfer_cutlass,
scitrera/dgx-spark-sglang:0.5.10, n=4 concurrent, max_tokens=2048.

| Metric | Test 17 Winner (Socket) | RoCE (new) | Speedup |
|--------|------------------------|------------|---------|
| Peak (sum tok/s) | 34.6 | **65.2** | **1.88×** |
| Aggregate tok/s | 31.0 | **65.4** | **2.1×** |
| Per-request tok/s | 8.65 | **16.3** | **1.88×** |
| Avg TTFT | 2.14s | **0.92s** | **2.3×** |
| Wall time (n=4) | 345s | **125s** | **2.8×** |

Nearly 2× throughput and 2.3× lower TTFT from two changes:
1. NCCL transport Socket → RoCE (9.78 GB/s vs 2.12 GB/s bus bandwidth)
2. Removed `CUDA_LAUNCH_BLOCKING="0"` from ConfigMap (was still serializing
   kernel launches despite the "0" value — CUDA checks env var presence,
   not value)

The Test 17 winner (34.6 tok/s) was the previous all-time best for this
model on this cluster. The new RoCE config replaces it as the production
baseline.

### Speculative decoding with RoCE (2026-04-12 ~15:25)

Model: nvidia/Qwen3-235B-A22B-NVFP4 + nvidia/Qwen3-235B-A22B-Eagle3
(EAGLE3), n=8 concurrent, RoCE transport.

| Config | n=8 gen throughput | accept rate |
|--------|-------------------|-------------|
| Normal decode (RoCE) | **~105 tok/s** | — |
| Speculative EAGLE3 (RoCE) | ~75-80 tok/s | **9%** |

Speculative is **~25% slower** than normal decode. Root cause: the EAGLE3
draft model's accept rate is only 9% — 91% of speculative tokens are
rejected. Each draft-verify cycle requires the same NCCL all-reduce
roundtrip as a normal decode step, but produces almost no accepted tokens.
The overhead of the wasted draft steps outweighs the throughput gain from
the few accepted ones.

This was also the case with socket transport (Test 37/38 in the kikube
matrix: 31.4 tok/s speculative vs 42.70 tok/s normal at n=8). RoCE makes
both modes faster but doesn't change the speculative-vs-normal ratio —
the bottleneck is the low accept rate, not network latency.

Conclusion: speculative decoding is not viable for this model on this
cluster. Normal decode with RoCE at n=8 (105 tok/s) is the fastest
configuration.

### Root causes identified

Two independent prerequisites for NCCL RoCE over SR-IOV VFs in K8s:

1. **VF IPv4 addresses in netplan** — without an IPv4 on the VF interface,
   the RoCE GID table only contains link-local `fe80::` entries. NCCL's IB
   plugin needs an IPv4-mapped GID (`::ffff:x.x.x.x`) at the configured
   GID index for RoCEv2 QP establishment. Fixed by adding `addresses:` to
   the VF block in `roles/dgx_prepare/templates/etc_netplan_10-qsfp.yaml.j2`.
   The VF IP matches the Multus IPAM-assigned pod IP (same IP on host and
   in pod).

2. **`privileged: true` on the pod** — `host-device` CNI moves the network
   interface into the pod namespace, but `/dev/infiniband/uverbs*` and
   `/sys/class/infiniband/rocep1s0f0vN/` remain on the host. Without
   `privileged: true` (which exposes all host `/dev/` and `/sys/` to the
   container), NCCL's IB plugin cannot open the RDMA verbs devices and
   falls back to TCP socket. `hostNetwork: true` alone is NOT sufficient.

### Test 1.1: RoCE VF7 with NCCL_DEBUG=INFO (2026-04-12 14:15)

**Result: RoCE FAILED — NCCL falls back to Socket.**

Key NCCL init log lines from rank0:
```
NCCL INFO NCCL_IB_HCA set to rocep1s0f0v7
NCCL INFO NET/IB : No device found.
NCCL INFO NET/IB : Using [RO]; OOB net1:10.10.100.241<0>
NCCL INFO Failed to initialize NET plugin IB
NCCL INFO NET/Socket : Using [0]net1:10.10.100.241<0>
NCCL INFO Initialized NET plugin Socket
```

All 8 channels: `via NET/Socket/0`.

**Root cause:** The `host-device` CNI moves the network interface
(`enp1s0f0v7` → `net1`) into the pod namespace, but the corresponding
IB/RoCE device (`/sys/class/infiniband/rocep1s0f0v7`) stays on the host.
NCCL's IB plugin looks for the device in `/sys/class/infiniband/` inside
the container and finds nothing → "No device found" → falls back to Socket.

**Implication:** RoCE over SR-IOV VFs in non-privileged containers requires
either `privileged: true` with IB device mounts, or the `k8s-rdma-shared-
dev-plugin` that exposes IB devices as K8s allocatable resources. The
current `host-device` CNI alone is insufficient for RDMA.

## Test matrix

All tests use `NCCL_DEBUG=INFO` (now baked into the pod template) so we can
see what transport NCCL actually negotiates.

### Phase 1: Diagnose RoCE on SR-IOV VFs (current setup)

**Test 1.1: RoCE VF7 with NCCL_DEBUG=INFO** (can run now)
```bash
ansible-playbook k8s_dgx.yml --tags nccl_test -e nccl_test_transport=roce
```
Goal: Read NCCL init logs to see WHY RoCE fails. Look for:
- `NCCL INFO NET/IB : Using [0]rocep1s0f0v7:1/RoCE` → RoCE is used
- `NCCL INFO NET/Socket : Using [0]net1:...` → fell back to socket
- Any `NCCL WARN` about IB/RoCE init failure

**Test 1.2: RoCE VF7 with different GID_INDEX values**
GID_INDEX=3 is for RoCEv2 over IPv4. But VF interfaces might need a
different GID. Try GID_INDEX=0,1,2,5:
```bash
ansible-playbook k8s_dgx.yml --tags nccl_test -e nccl_test_transport=roce -e nccl_test_gid_index=0
```
(Requires adding `nccl_test_gid_index` override to the roce transport config.)

### Phase 2: Test without SR-IOV (PF direct)

SR-IOV adds a layer of complexity. Testing on the bare PF eliminates VF-level
issues (eSwitch forwarding, VF RoCE capability, GID table per-VF).

**Prerequisite:** No pods running that use host-device on the QSFP PF.
Since nothing is running now, this is clear.

**Test 2.1: Host-network with QSFP PF (socket transport)**
```bash
ansible-playbook k8s_dgx.yml --tags nccl_test -e nccl_test_transport=host
```
This uses hostNetwork:true + NCCL_SOCKET_IFNAME=enP2p1s0f0np0. TCP socket
over the PF directly, no Multus, no VF. Establishes the PF socket baseline.

**Test 2.2: Host-network with QSFP PF (RoCE transport)**
Requires a new transport variant `host_roce` that uses:
- `hostNetwork: true`
- `NCCL_IB_HCA=mlx5_0` (or whatever the PF's RoCE device is)
- `NCCL_SOCKET_IFNAME=enP2p1s0f0np0` (for bootstrap)
- `NCCL_IB_GID_INDEX=3` (or auto)
- NO `NCCL_NET=Socket`, NO `NCCL_IB_DISABLE=1`

This tests RoCE on the bare PF with zero SR-IOV/Multus/CNI abstraction.
If this gives 20+ GB/s → SR-IOV VFs are the bottleneck.
If this also gives ~2 GB/s → RoCE is fundamentally broken on these NICs.

To find the PF RoCE device name:
```bash
ssh root@spark1.local 'ls /sys/class/infiniband/ && ibstat'
```

**Test 2.3: Host-network with iperf3 (raw TCP baseline)**
Not NCCL — just raw TCP throughput over the PF to establish the link-level
maximum:
```bash
# spark1:
iperf3 -s -B 10.10.10.1
# spark2:
iperf3 -c 10.10.10.1 -t 10 -P 4
```
Expected: 20-24 Gbit/s (~2.5-3 GB/s) for TCP, limited by kernel stack.
This is the ceiling for socket transport; RoCE should exceed it via RDMA bypass.

### Phase 3: RoCE prerequisites check

If Phase 2 RoCE also fails, check the hardware/driver prerequisites:

**Test 3.1: RoCE device existence**
```bash
for h in spark1 spark2 spark3 spark4; do
  echo "=== $h ==="
  ssh root@$h.local 'ls /sys/class/infiniband/ 2>/dev/null || echo "(no IB devices)"'
  ssh root@$h.local 'cat /sys/class/infiniband/*/ports/1/link_layer 2>/dev/null || echo "(no link_layer)"'
  ssh root@$h.local 'cat /sys/class/infiniband/*/ports/1/gids/3 2>/dev/null || echo "(no GID 3)"'
done
```

**Test 3.2: DCBX / PFC configuration**
RoCE requires lossless Ethernet (PFC enabled). Check:
```bash
ssh root@spark1.local 'mlnx_qos -i enp1s0f0np0 2>/dev/null || echo "mlnx_qos not available"'
ssh root@spark1.local 'ethtool --show-pause enp1s0f0np0'
```

**Test 3.3: RoCE mode**
ConnectX-7 supports RoCEv1 and RoCEv2. NCCL needs RoCEv2:
```bash
ssh root@spark1.local 'cma_roce_mode -d mlx5_0 -p 1 2>/dev/null || echo "cma_roce_mode not available"'
```

## Execution order

1. **Test 1.1** — RoCE VF7 with debug (immediate, just re-run with NCCL_DEBUG)
2. Read the NCCL init logs → understand failure mode
3. **Test 2.1** — Host-network socket baseline (PF direct)
4. **Test 3.1** — RoCE device check on hosts (SSH one-liners)
5. **Test 2.2** — Host-network RoCE on PF (if 3.1 shows IB devices exist)
6. **Test 2.3** — iperf3 TCP baseline (optional, for reference)
7. **Test 1.2** — GID_INDEX sweep (if 2.2 works but 1.1 doesn't → VF-specific)

## Implementation needed

- `NCCL_DEBUG=INFO` → done (in pod template)
- `host_roce` transport variant → needs adding to `nccl_test_run_one.yml`
  (hostNetwork:true + RoCE env vars, PF IPs as master_addr)
- `nccl_test_gid_index` override → minor template change
- SSH-based checks (Phase 3) → can run ad-hoc, no Ansible task needed

## Expected outcomes

| If... | Then... |
|-------|---------|
| Test 1.1 shows "NET/Socket" fallback | NCCL can't init RoCE on VF7 |
| Test 2.2 gives 20+ GB/s on PF | RoCE works on PF but not VFs → SR-IOV RoCE config issue |
| Test 2.2 also gives ~2 GB/s | RoCE not functional on these NICs → check Phase 3 prerequisites |
| Test 3.1 shows no IB devices | RDMA/IB kernel modules not loaded or NIC firmware issue |
| Test 3.2 shows no PFC | Lossless Ethernet not configured → RoCE will have drops under load |
