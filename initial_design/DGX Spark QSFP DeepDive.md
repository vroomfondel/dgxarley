# Why Does the DGX Spark Have Two QSFP Ports – and What Happens When You Use Both?

> A deep dive into the unusual PCIe topology of the NVIDIA GB10 ConnectX-7 and its practical implications for stacking, NCCL, and network planning.
>
> **Related:** For the general setup and configuration of the DGX Spark, see [[DGX Spark/DGX Spark Setup|DGX Spark Setup]].

## TL;DR

The two QSFP ports on the DGX Spark share **the same two PCIe Gen5 x4 links** to the GB10 SoC. The total bandwidth is ~200 Gbit/s – regardless of whether one or two cables are used. Using both ports simultaneously for different connections is possible but can degrade NCCL performance and requires careful interface management.

---

## 1. The Expectation: Two Ports = 400 Gbit/s?

On the back of the DGX Spark there are two QSFP sockets, connected via an NVIDIA ConnectX-7 NIC. The obvious assumption: each port delivers 200 Gbit/s, totaling 400 Gbit/s. Several sources have documented this expectation:

- [StorageReview](https://www.storagereview.com/review/nvidia-dgx-spark-review-the-ai-appliance-bringing-datacenter-capabilities-to-desktops) noted that at first glance one might expect 400G connectivity, but the PCIe limitation prevents this.
- In the [NVIDIA Developer Forum](https://forums.developer.nvidia.com/t/confusion-surrounding-the-qsfp-ports-and-bandwidth/356092), users explicitly asked whether two cables could deliver 400 Gbit/s.
- [Jeff Geerling](https://www.jeffgeerling.com/blog/2025/dells-version-dgx-spark-fixes-pain-points/) pointed out that the maximum real-world bandwidth is approximately 206 Gbit/s, regardless of configuration.

## 2. The Reality: Two PCIe Gen5 x4 Links, One ConnectX-7

[ServeTheHome](https://www.servethehome.com/the-nvidia-gb10-connectx-7-200gbe-networking-is-really-different/) uncovered the internal topology in detail. The result is surprising:

### 2.1 Architecture Overview

```
┌─────────────┐
│   GB10 SoC  │
│  (Grace     │
│  Blackwell) │
└──┬──────┬───┘
   │      │
   │ PCIe Gen5 x4   PCIe Gen5 x4
   │ (~100 Gbit/s)  (~100 Gbit/s)
   │      │
┌──▼──────▼───┐
│  ConnectX-7  │
│  (Multi-Host │
│   Mode)      │
│              │
│  ┌────┐ ┌────┐
│  │MAC0│ │MAC1│  ← Port 0 (QSFP left)
│  │100G│ │100G│
│  └────┘ └────┘
│  ┌────┐ ┌────┐
│  │MAC2│ │MAC3│  ← Port 1 (QSFP right)
│  │100G│ │100G│
│  └────┘ └────┘
└──────────────┘
```

The key point: The GB10 SoC can provide **at most PCIe Gen5 x4** per device. To reach the ConnectX-7's 200 Gbit/s, two separate x4 links are aggregated using the ConnectX-7's multi-host mode. Each physical QSFP port has two 100G MACs, each mapped to one of the two PCIe x4 links.

**Source:** NVIDIA employee Raphael Amorim confirmed this directly in the [ServeTheHome comments](https://www.servethehome.com/the-nvidia-gb10-connectx-7-200gbe-networking-is-really-different/#comment-767610):

> *"This is the expected behaviour due to a limitation in the GB10 chip. The SoC can't provide more than x4-wide PCIe per device, so, in order to achieve the 200gbps speed, we had to use the Cx7's multi-host mode, aggregating 2 separate x4-wide PCIe links."*

### 2.2 Four Interfaces in the Operating System

The OS shows **four** network interfaces, not two:

| Interface | PCIe Address | Physical Port | PCIe Link |
|---|---|---|---|
| `enp1s0f0np0` | `0000:01:00.0` | Port 0 (left) | Link A |
| `enp1s0f1np1` | `0000:01:00.1` | Port 1 (right) | Link A |
| `enP2p1s0f0np0` | `0002:01:00.0` | Port 0 (left) | Link B |
| `enP2p1s0f1np1` | `0002:01:00.1` | Port 1 (right) | Link B |

Each physical port is thus represented by **two logical interfaces** running over different PCIe links. The NVIDIA documentation recommends using only the `enp1s0f*` interfaces and ignoring the `enP2p1s0f*` variants, as they refer to the same physical port.

**Source:** [NVIDIA Connect Two Sparks Playbook](https://build.nvidia.com/spark/connect-two-sparks/stacked-sparks), [NVIDIA Developer Forum – ConnectX-7 NIC in DGX Spark](https://forums.developer.nvidia.com/t/connectx-7-nic-in-dgx-spark/350417)

### 2.3 Port Enumeration According to NVIDIA

An NVIDIA engineer explained the internal mapping in the [Developer Forum](https://forums.developer.nvidia.com/t/connectx-7-nic-in-dgx-spark/350417?page=4):

- Port 0, first half
- Port 1, first half
- *(Wi-Fi sits in between)*
- Port 0, second half
- Port 1, second half

Aggregation must occur between the two halves of the **same** port (e.g., `enp1s0f0np0` + `enP2p1s0f0np0` for Port 0).

## 3. Bandwidth Reality

### 3.1 One Cable Is Enough for Full 200 Gbit/s

The [official NVIDIA documentation](https://build.nvidia.com/spark/connect-two-sparks/stacked-sparks) states clearly:

> *"Full bandwidth can be achieved with just one QSFP cable."*

[LMSYS](https://lmsys.org/blog/2025-10-13-nvidia-dgx-spark/) also describes the two ports with an **aggregated** bandwidth of 200 Gbit/s, not 200 Gbit/s per port.

### 3.2 Measured Values

| Method | Result | Source |
|---|---|---|
| iperf3, 1 port, 1 stream | ~96 Gbit/s | [ServeTheHome](https://www.servethehome.com/the-nvidia-gb10-connectx-7-200gbe-networking-is-really-different/) |
| iperf3, 2 ports, 2× streams | ~96 Gbit/s (no gain) | [ServeTheHome](https://www.servethehome.com/the-nvidia-gb10-connectx-7-200gbe-networking-is-really-different/) |
| iperf3, 2 ports, 60–64 streams, jumbo | 160–198 Gbit/s | [ServeTheHome](https://www.servethehome.com/the-nvidia-gb10-connectx-7-200gbe-networking-is-really-different/) |
| RoCE / RDMA (perftest) | 185–190 Gbit/s | [ServeTheHome](https://www.servethehome.com/the-nvidia-gb10-connectx-7-200gbe-networking-is-really-different/) |
| Bonding (XOR/RR) | ≤ 112 Gbit/s | [NVIDIA Forum](https://forums.developer.nvidia.com/t/connectx-7-nic-in-dgx-spark/350417?page=4) |
| Two cables vs. one cable | No speed advantage | [NVIDIA Forum](https://forums.developer.nvidia.com/t/connectx-7-nic-in-dgx-spark/350417?page=4) |

### 3.3 Why TCP/IP Is Limited to ~100 Gbit/s

The reason: TCP/IP applications like iperf3 can typically only use a single logical interface. Each logical interface is attached to one PCIe Gen5 x4 link, which maxes out at ~100 Gbit/s. Bonding the interfaces should theoretically help, but in practice it only achieved ~112 Gbit/s and broke RDMA functionality.

**Source:** Forum user `eugr` in the [NVIDIA Developer Forum](https://forums.developer.nvidia.com/t/connectx-7-nic-in-dgx-spark/350417?page=4)

## 4. NCCL and the Two-Port Question

### 4.1 NCCL Aggregates Transparently – Over One Port

NCCL (NVIDIA Collective Communications Library) is topology-aware and can address both logical interfaces of a physical port simultaneously. This allows NCCL to achieve the full ~200 Gbit/s over **a single cable**.

As a forum user summarized: NCCL can work across both logical interfaces and thus extract 200G from a single port. This does not apply to regular TCP/IP traffic.

**Source:** [NVIDIA Developer Forum](https://forums.developer.nvidia.com/t/connectx-7-nic-in-dgx-spark/350417?page=4)

### 4.2 What Happens When Both Ports Are Used in Parallel?

This is where things get problematic. Since both physical ports **share the same two PCIe x4 links**, using both ports simultaneously for different connections creates resource contention:

1. **PCIe bandwidth is shared:** Each of the two PCIe x4 links must now serve traffic for both physical ports. NCCL loses the ability to exclusively use both x4 links for the Spark-to-Spark connection.

2. **NCCL performance can drop:** If other traffic is running on the second port simultaneously (e.g., NFS, data downloads, management), it competes with NCCL traffic on the PCIe links. The transparent aggregation to 200 Gbit/s is then no longer guaranteed.

3. **Interface mapping becomes complex:** With five GB10 systems, each having four logical interfaces, the mapping became extremely error-prone according to ServeTheHome. Incorrect interface mapping regularly caused two data streams to run over the same PCIe x4 link, dropping bandwidth to ~92–95 Gbit/s.

An NVIDIA engineer confirmed in the forum: two cables provide no speed advantage. Aggregation should occur between the two halves of the same port.

**Source:** NVIDIA engineer `isdias` in the [Developer Forum](https://forums.developer.nvidia.com/t/connectx-7-nic-in-dgx-spark/350417?page=4)

### 4.3 ServeTheHome's Recommendation

ServeTheHome summarizes the situation pragmatically: if you use both ports, think of them as **2× 100G links**, not 2× 200G. When possible, they recommend using a single 200G link over one port.

**Source:** [ServeTheHome](https://www.servethehome.com/the-nvidia-gb10-connectx-7-200gbe-networking-is-really-different/)

## 5. When Is the Second Port Still Useful?

Despite the limitations, there are legitimate use cases for the second port:

| Scenario | Assessment |
|---|---|
| **Redundancy:** Failover in case of cable/port failure | ✅ Useful |
| **Switch connectivity:** Port 1 → partner Spark, Port 2 → switch for multi-node clusters (>2 nodes) | ⚠️ Possible, but PCIe bandwidth is shared |
| **Double bandwidth to partner Spark** | ❌ No benefit |
| **Separation of management and data traffic** | ⚠️ Better solved via the dedicated 10 GbE RJ45 port |

For management traffic, SSH access, and general network traffic, the dedicated **10 GbE RJ45 port** is available, which does not affect QSFP bandwidth. NVIDIA's own stacking documentation explicitly uses this port (or Wi-Fi) for out-of-band management.

## 6. Why Two Ports at All?

The most likely explanation is a combination of several factors:

1. **ConnectX-7 design:** The ConnectX-7 is designed as a dual-port NIC. NVIDIA uses the chip in its standard configuration, which saves development and manufacturing costs.

2. **GB10 SoC limitation:** The SoC can only provide PCIe Gen5 x4 links per device. To achieve 200 Gbit/s, the ConnectX-7's multi-host mode with two x4 links was the most pragmatic approach. The resulting two ports are more of a byproduct than a feature.

3. **Flexibility for multi-node scenarios:** Those who want to connect more than two Sparks via a switch can use the second port for this – with the awareness that PCIe bandwidth is shared.

ServeTheHome comments that the design points more to SoC limitations than to deliberate architectural decisions.

## 7. Summary

```
                    Total PCIe Bandwidth to SoC
                    ┌───────────────────────────┐
                    │  2× PCIe Gen5 x4 ≈ 200G   │
                    └─────────┬─────────────────┘
                              │
                    ┌─────────▼─────────────────┐
                    │      ConnectX-7 NIC        │
                    │                             │
                    │   Port 0          Port 1    │
                    │   (QSFP left)   (QSFP right)│
                    │   2× 100G MAC   2× 100G MAC │
                    └───┬─────────────────┬───────┘
                        │                 │
                   Stacking cable    Optional:
                   to 2nd Spark      Switch / Redundancy
```

- **200 Gbit/s is the maximum** – across the entire NIC, not per port.
- **One cable is sufficient** for full NCCL bandwidth when stacking.
- **Two cables provide no throughput gain**, as the PCIe connection is the bottleneck.
- **Parallel use of both ports is possible**, but shares PCIe bandwidth and can degrade NCCL performance.
- **Use the 10 GbE RJ45 port for management traffic**, not the second QSFP.

---

## Sources

1. ServeTheHome: [The NVIDIA GB10 ConnectX-7 200GbE Networking is Really Different](https://www.servethehome.com/the-nvidia-gb10-connectx-7-200gbe-networking-is-really-different/) (December 2025)
2. NVIDIA Developer Forum: [ConnectX-7 NIC in DGX Spark](https://forums.developer.nvidia.com/t/connectx-7-nic-in-dgx-spark/350417) (November 2025)
3. NVIDIA: [Connect Two Sparks Playbook](https://build.nvidia.com/spark/connect-two-sparks/stacked-sparks)
4. NVIDIA: [DGX Spark User Guide – Spark Stacking](https://docs.nvidia.com/dgx/dgx-spark/spark-clustering.html)
5. LMSYS: [NVIDIA DGX Spark In-Depth Review](https://lmsys.org/blog/2025-10-13-nvidia-dgx-spark/) (October 2025)
6. Jeff Geerling: [Dell's version of the DGX Spark fixes pain points](https://www.jeffgeerling.com/blog/2025/dells-version-dgx-spark-fixes-pain-points/) (December 2025)
7. StorageReview: [NVIDIA DGX Spark Review](https://www.storagereview.com/review/nvidia-dgx-spark-review-the-ai-appliance-bringing-datacenter-capabilities-to-desktops) (November 2025)
8. NVIDIA Developer Forum: [Confusion surrounding the QSFP ports and bandwidth](https://forums.developer.nvidia.com/t/confusion-surrounding-the-qsfp-ports-and-bandwidth/356092) (December 2025)

---

*Created: March 2026. All information is based on publicly available sources and may change with future firmware or driver updates.*
