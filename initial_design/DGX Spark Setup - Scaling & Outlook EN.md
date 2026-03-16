# DGX Spark — Scaling, Model Alternatives & Outlook

_Extracted from [[DGX Spark/DGX Spark Setup]] — contains future planning, model alternatives, switch cost calculations, and references._

---

## Open Items & Future Plans

### Short-Term Relevant

- **SGLang MoE Kernel Tuning**: If you encounter OutOfResources errors with MoE models (GB10 has 101KB Shared Memory instead of the 147KB expected by SGLang), there are ready-made kernel configs on GitHub (BTankut/dgx-spark-sglang-moe-configs). The tuning takes ~9 hours but only needs to be done once.

- **EAGLE Speculative Decoding**: Once EAGLE3 draft models for Qwen3-235B become available, decode speed can increase by another 1.5–2×. SGLang supports this natively. In the implementation, speculative decoding is prepared as an Ansible variable (`sglang_speculative_enabled: false`, `NEXTN` algorithm, 3 steps, topk=1, 4 draft tokens) — activation by changing a flag.

- **Qwen3-Coder-Next FP8**: Runs on a single Spark at ~43 t/s — excellent as a fast secondary model for coding tasks.


### Medium-Term

- **NVFP4 Kernels for SM121**: NVIDIA is working on this. When available, FP4 models could run significantly faster — Qwen3-235B in NVFP4 would potentially fit on a single Spark.

- **vLLM Stabilization**: eugr is working on a nightly build system with automated tests. Once that's in place, vLLM could become an option again — currently not recommended.

- **llama.cpp NCCL Support**: Actively discussed in the community. If this lands, you'd get the stability of llama.cpp with the bandwidth of NCCL — that would be a game changer.

### Outlook: Multiple AI Pods per DGX Spark Node

The current setup uses **one AI pod per node** (SGLang Head on Spark 1, Worker on Spark 2). If in the future multiple smaller models should run in parallel on a single Spark, two bottlenecks need to be addressed:

**Bottleneck 1 — GPU (limiting factor):**

The GB10 has exactly **one** Blackwell GPU with 128 GB Unified Memory. GPU sharing options:

| Method | GB10 Support? | Memory Isolation? | Assessment |
|---|---|---|---|
| **MIG** (Multi-Instance GPU) | ❌ Not supported | ✅ Yes | MIG is only for A100/H100/B200 — GB10 has Unified Memory |
| **Time-Slicing** | ✅ Works | ❌ No | Pods share full GPU memory; if Pod A occupies 120 GB → Pod B gets OOM |
| **MPS** (Multi-Process Service) | ✅ Works | ❌ No | Concurrent kernels, same memory limitation |

> [!success] Implemented
> GPU Time-Slicing is active in the production K3s cluster with `nvidia_gpu_timeslice_replicas: 4`. This allows Ollama (bge-m3 Embedding) and SGLang to run in parallel as separate GPU consumers on the same node.

Time-Slicing makes the most sense for **smaller co-processes** (such as embedding models) alongside the main LLM. With Qwen3-235B-FP8 (~120 GB), there's little room for a second large model — but lightweight services like Ollama (bge-m3, ~1.2 GB) work without issues.

Configuration via the NVIDIA Device Plugin (no GPU Operator required):
```yaml
# ConfigMap for nvidia-device-plugin (production configuration)
sharing:
  timeSlicing:
    renameByDefault: false
    resources:
      - name: nvidia.com/gpu
        replicas: 4    # 4 pods can share the GPU (implemented)
```

**Bottleneck 2 — Network:**

The current `host-device` CNI **moves** the physical interface into the pod namespace — only **one** pod can use it. Alternatives for multi-pod:

| Method | RDMA-capable? | Effort | Note |
|---|---|---|---|
| **`hostNetwork: true`** | ✅ Yes | Low | All pods see all host interfaces, but port conflicts possible |
| **`macvlan`** + RDMA Shared Device Plugin | ✅ Partially | Medium | `macvlan` for IP, shared RDMA device as K8s resource |
| **SR-IOV** | ✅ Yes (per VF) | High | Each pod gets its own Virtual Function with its own RDMA device |

**SR-IOV on GB10 — open risk**: The ConnectX-7 in the GB10 runs in Multi-Host mode (2× PCIe Gen5 x4). Whether SR-IOV works in this configuration is **neither documented by NVIDIA nor tested by the community**. VFs would need to be distributed across two PCIe root complexes, which is not foreseen in classic SR-IOV. Additionally, GPUDirect RDMA is not supported on the Spark anyway — the main advantage of per-VF GPUDirect therefore does not apply.

With SR-IOV, the **NVIDIA Network Operator** would actually be useful — it automatically manages VF creation, the SR-IOV Device Plugin, and Multus NADs per VF.

**Recommendation for multi-pod**: `hostNetwork: true` + GPU Time-Slicing is the most pragmatic path — no SR-IOV, no Network Operator, minimal configuration effort. Only becomes relevant when significantly smaller models should run in parallel.


### Long-Term

- **NVIDIA Dynamo + SGLang**: Disaggregated Serving (separate Prefill and Decode phases on different nodes) could bring a major performance leap to multi-Spark setups.

- **Next model generation**: GLM-4.7, Qwen3.5, and upcoming MoE models are increasingly being optimized for Spark. The trend clearly points towards: larger models, better quantization, better kernel support.

> [!success] Current Status (March 2026)
> The hardware is absolutely solid. The complete software stack is deployed in production as a K3s cluster and automated via Ansible (`dgxarley` repo). SGLang 0.5.9 with NCCL multi-node, HAProxy sidecar pattern, model profile system, cert-manager/TLS, Prometheus/Grafana/Loki monitoring, PostgreSQL/pgvector, and DNS registration are deployed and stable.

### Model Alternative: GLM-5 (744B MoE, 40B active)

GLM-5 (Zhipu AI / Z.ai, released February 2026) is significantly larger than Qwen3-235B with 744B parameters and 40B active parameters per token (256 experts, 8 active). It uses DeepSeek Sparse Attention (DSA) and supports 204K token context. In benchmarks, GLM-5 leads especially in coding and agentic tasks.

**Memory requirements vs. DGX Spark configurations:**

| Quantization | Weight Size | Sparks needed | KV-Cache Headroom | Engine |
|---|---|---|---|---|
| FP8 ([zai-org/GLM-5-FP8](https://huggingface.co/zai-org/GLM-5-FP8)) | ~800 GB | 7 ❌ | ~96 GB | SGLang (TP=7) |
| Q8_0 (GGUF) | ~801 GB | 7 ❌ | ~95 GB | llama.cpp |
| Q5_K_M (GGUF) | ~535 GB | 5 | ~105 GB | llama.cpp |
| **UD-Q4_K_XL** ([unsloth/GLM-5-GGUF](https://huggingface.co/unsloth/GLM-5-GGUF)) | **~431 GB** | **4** ✅ | **~81 GB** | llama.cpp (RPC) |
| Q4_K_M (GGUF) | ~456 GB | 4 | ~56 GB | llama.cpp |
| NVFP4 (Blackwell-native) | ~400 GB (estimated) | 4 | ~112 GB | SGLang — **does not exist yet** |
| UD-Q2_K_XL (GGUF) | ~281 GB | 3 | ~103 GB | llama.cpp — quality questionable |

**FP8 is not realistic on DGX Spark** — 7 Sparks + enterprise switch would be required. FP8 is designed for 8× H200/B200 datacenter setups.

**4-bit (UD-Q4_K_XL) on 4× Spark is the realistic path:**

Concrete setup: 4× ASUS Ascent GX10, MikroTik CRS812 Switch (see scaling below), 4× QSFP56 DAC. Inference via llama.cpp RPC (not SGLang/NCCL, since GGUF):

```bash
# On Spark 2–4 (RPC workers):
llama-server --rpc-host 0.0.0.0 --rpc-port 50052

# On Spark 1 (Head):
llama-server \
  --model GLM-5-UD-Q4_K_XL.gguf \
  --rpc spark2:50052,spark3:50052,spark4:50052 \
  --n-gpu-layers 999 \
  --ctx-size 32768 \
  --port 8000
```

> [!warning] Trade-off: GLM-5@Q4 vs. Qwen3-235B@FP8
>
> | | **Qwen3-235B-FP8** (current) | **GLM-5-Q4_K_XL** |
> |---|---|---|
> | **Sparks** | 2 | 4 |
> | **Hardware cost** | ~6,000 € + DAC | ~12,000 € + Switch + DACs |
> | **Decode speed** | ~25 t/s (SGLang+NCCL/RDMA) | ~10–15 t/s (llama.cpp RPC/TCP) |
> | **Quantization quality** | FP8 ≈ 99% of BF16 | Q4 ≈ 92–95% of BF16 |
> | **Active parameters** | 22B (MoE) | 40B (MoE) |
> | **Max. context** | ~65K | ~32K (with 81 GB KV-Cache headroom) |
> | **Network transport** | NCCL/RDMA (~200 Gbps) | llama.cpp RPC/TCP (~100 Gbps) |
>
> GLM-5 has more active parameters (40B vs. 22B), but 4-bit quantization eats into part of this advantage. Whether GLM-5@Q4 is actually better than Qwen3-235B@FP8 depends heavily on the use case — for coding, GLM-5 leads; for general reasoning it's close.
>
> **Recommendation**: Stay with Qwen3-235B-FP8 on 2 Sparks. Wait for **GLM-5-NVFP4** — when that appears and works with SGLang, the upgrade path to 4 Sparks is clear (natively Blackwell-accelerated, SGLang+NCCL instead of llama.cpp RPC, ~20% faster than AWQ). GLM-4.7-NVFP4 [already exists](https://forums.developer.nvidia.com/t/running-glm-4-7-fp8-355b-moe-on-4x-dgx-spark-with-sglang-eagle-speculative-decoding/359256), GLM-5-NVFP4 will follow.

---

## Scaling to 4× DGX Spark: Switch & Costs

### Why a Switch is Required

For 2 Sparks, a direct DAC cable (point-to-point) is sufficient. For 3+ Sparks, a switch is mandatory — daisy-chaining is not possible. Each Spark has one ConnectX-7 port (200GbE, QSFP56/QSFP112-compatible).

### Switch Recommendation: MikroTik CRS812 DDQ

| Property | Value |
|---|---|
| **Model** | CRS812-8DS-2DQ-2DDQ-RM |
| **Ports** | 8× 50G SFP56 + **2× 200G QSFP56** + **2× 400G QSFP56-DD** |
| **Switching capacity** | 1.6 Tbps unidirectional |
| **Form factor** | 1U Rackmount, Dual-PSU (redundant) |
| **CPU** | Quad-Core ARM 2 GHz, 4 GB RAM |
| **RouterOS** | v7, Layer 3, hardware offloading |

**Community-confirmed**: A user in the NVIDIA Developer Forum is successfully running an 8-node Spark cluster with 2× CRS812 DDQ. Comment: *"It pretty much just works."*

### Cabling for 4× Spark

```
MikroTik CRS812 DDQ
├── QSFP56 Port #1 (200G) ──DAC──→ Spark 1
├── QSFP56 Port #2 (200G) ──DAC──→ Spark 2
├── QSFP-DD Port #1 (400G) ──Breakout──→ Spark 3 + Spark 4
└── QSFP-DD Port #2 (400G) ──(free for expansion to 6 Sparks)
```

- 2 Sparks on the native **200G QSFP56 ports** (one 200G DAC cable each)
- 2 Sparks via a **400G→2×200G breakout cable** on the QSFP-DD port
- The second QSFP-DD port remains free → **scalable up to 6 Sparks** without a second switch

**Important configuration:**
- Enable Jumbo Frames (MTU 9000+)
- Disable auto-negotiation on the QSFP-DD ports
- Manually split QSFP-DD port to 2×200G

### Prices in Germany (as of February 2026)

#### Switch

| Vendor | Price (incl. VAT) | Availability | Source |
|---|---|---|---|
| **idealo.de** (cheapest) | **from 1,047 €** | Available | [idealo.de](https://www.idealo.de/preisvergleich/OffersOfProduct/209057897_-crs812-ddq-mikrotik.html) |
| **Geizhals** | from 1,050 € | Available | [geizhals.de](https://geizhals.de/mikrotik-cloud-router-switch-crs812-ddq-rackmount-50g-managed-switch-crs812-8ds-2dq-2ddq-rm-a3636266.html) |
| **NAS Store** | 1,099 € | Ships in 2–4 business days | nasstore.de |
| **Galaxus.de** | 1,208 € | Available | [galaxus.de](https://www.galaxus.de/de/s1/product/mikrotik-crs812-ddq-14-ports-netzwerk-switch-62437969) |

#### Cables

| Cable | Units | Price per unit | Vendor |
|---|---|---|---|
| 200G QSFP56 DAC, 0.5m (Mellanox-compatible) | 2 | ~79 € | [FS.com DE](https://www.fs.com/de/products/115634.html) |
| 400G QSFP-DD → 2×200G QSFP56 Breakout DAC, 1.5m | 1 | ~157 € | [FS.com DE](https://www.fs.com/de/products/182806.html) |

> **Note on cables**: The FS.com cables are certified as Mellanox/NVIDIA-compatible and are explicitly marketed for DGX Spark Dual-System Interconnect. Alternatively, BlueOptics offers 200G DAC for ~123 €/unit.

#### Total cost estimate (network expansion only)

| Item | Cost |
|---|---|
| MikroTik CRS812 DDQ | ~1,050 € |
| 2× 200G QSFP56 DAC (0.5m) | ~158 € |
| 1× 400G→2×200G Breakout DAC (1.5m) | ~157 € |
| **Network total** | **~1,365 €** |

#### Total cost estimate (incl. 2 additional Sparks)

| Item | Cost |
|---|---|
| 2× DGX Spark (~3,450 € each) | ~6,900 € |
| Network (Switch + cables) | ~1,365 € |
| **Total** | **~8,265 €** |

### What does 4× Spark bring?

| Property | 2× Spark (current) | 4× Spark |
|---|---|---|
| **Unified Memory** | 256 GB | **512 GB** |
| **FP8 model + KV budget** | 237 GB model + ~14 GB KV | 237 GB model + **~270 GB KV** |
| **Context length (FP8 KV)** | ~140K | **Full 262K** (and more headroom) |
| **Decode speed (MoE)** | ~25 t/s | ~30–50 t/s (better parallelization) |
| **Prefill throughput** | ~23K t/s | **Significantly higher** (linear scaling) |
| **Alternative models** | Qwen3-235B max. | GLM-4.7-FP8 (355B MoE) possible |
| **SGLang launch command** | `--tp 2 --nnodes 2` | `--tp 4 --nnodes 4` |

**Most important advantage**: With 4× Spark and FP8 weights you would have enough memory for the **full 262K context window** — without compromises on model quality. Alternatively, you could run even larger models like GLM-4.7-FP8 (355B).

### Enterprise Alternative: Mellanox SN3700

If you want to scale beyond 6 Sparks or require enterprise support:

| Property | MikroTik CRS812 DDQ | Mellanox SN3700 (refurbished) |
|---|---|---|
| **Price** | ~1,050 € | ~14,000 € (Reef Telecom) |
| **Max. 200G ports** | 4 (2 native + 2 via breakout) | **32 native** |
| **Max. Sparks (1 switch)** | 6 | 32 |
| **RoCE/RDMA** | ✅ Basic | ✅ Natively optimized, GPUDirect |
| **Latency** | Standard | 425ns port-to-port |
| **Support** | Community | Enterprise (10-year warranty) |

For a home/office setup with ≤6 Sparks, the CRS812 DDQ is the clear choice. You only need the SN3700 for serious datacenter-scale deployments.

---

## Sources & References

- **SGLang Community Image**: `scitrera/dgx-spark-sglang:0.5.9-t5` (DockerHub) — 0.5.8-t5 still available for models without GDN-Attention

- **SGLang MoE Kernel Configs for GB10**: github.com/BTankut/dgx-spark-sglang-moe-configs

- **vLLM Community Docker (reference)**: github.com/eugr/spark-vllm-docker

- **vLLM Pre-built Images**: github.com/mark-ramsey-ri/vllm-dgx-spark

- **NVIDIA DGX Spark Playbooks**: build.nvidia.com/spark

- **NVIDIA SGLang Playbook**: build.nvidia.com/spark/sglang

- **Open WebUI SearXNG Docs**: docs.openwebui.com/features/web-search/searxng

- **Open WebUI Pipelines**: github.com/open-webui/pipelines

- **llama.cpp Spark Performance**: github.com/ggml-org/llama.cpp/discussions/16578

- **MikroTik CRS812 DDQ (product page)**: mikrotik.com/product/crs812_ddq

- **MikroTik CRS812 DDQ Review (ServeTheHome)**: servethehome.com — CRS812-8DS-2DQ-2DDQ-RM Review

- **CRS812 DDQ price comparison (idealo.de)**: idealo.de/preisvergleich/OffersOfProduct/209057897

- **200G QSFP56 DAC for DGX Spark (FS.com DE)**: fs.com/de/products/115634.html

- **400G→2×200G Breakout DAC (FS.com DE)**: fs.com/de/products/182806.html

- **NVIDIA Forum: Multi-Spark switch recommendations**: forums.developer.nvidia.com/t/connecting-multiple-dgx-spark-units-ethernet-switch-recommendations/345839

- **NVIDIA Forum: 6× Spark setup (community experiences)**: forums.developer.nvidia.com/t/6x-spark-setup/354399

- **NVIDIA DGX Spark Stacking Guide**: docs.nvidia.com/dgx/dgx-spark/spark-clustering.html

- **K3s Documentation**: [k3s.io/docs](https://docs.k3s.io/)

- **K3s ARM64 Support**: [docs.k3s.io/advanced#running-on-arm64](https://docs.k3s.io/advanced#running-on-arm64)

- **NVIDIA K8s Device Plugin**: [github.com/NVIDIA/k8s-device-plugin](https://github.com/NVIDIA/k8s-device-plugin)

- **Multus CNI** (dual-NIC for pods): [github.com/k8snetworking/multus-cni](https://github.com/k8snetworking/multus-cni)

- **Multus `host-device` plugin** (RDMA-capable interface passthrough): [containernetworking.github.io/plugins/main/host-device](https://www.cni.dev/plugins/current/main/host-device/)

- **ASUS Ascent GX10 Review** (OEM variant of DGX Spark): [servethehome.com](https://www.servethehome.com/asus-ascent-gx10-review-a-new-nvidia-gb10-solution/)

- **GB10 ConnectX-7 200GbE Networking — Dual-PCIe-x4 Architecture** (ServeTheHome deep dive): [servethehome.com](https://www.servethehome.com/the-nvidia-gb10-connectx-7-200gbe-networking-is-really-different/)

- **NVIDIA Forum: ConnectX-7 NIC in DGX Spark** (interface names, multi-host mode, NCCL aggregation): [forums.developer.nvidia.com](https://forums.developer.nvidia.com/t/connectx-7-nic-in-dgx-spark/350417)

- **NVIDIA Forum: ConnectX-7 bonding between Sparks** (balance-rr, MTU 9000, NCCL results): [forums.developer.nvidia.com](https://forums.developer.nvidia.com/t/bonding-spark-connectx-7-ports-between-two-sparks-with-jumbo-frames-works-fine/349872)

- **NVIDIA FAQ: GPUDirect RDMA not supported on DGX Spark**: [nvidia.custhelp.com](https://nvidia.custhelp.com/app/answers/detail/a_id/5780/~/is-gpudirect-rdma-supported-on-dgx-spark)

- **NVIDIA Forum: NCCL/RoCEv2 on DGX Spark** (transport detection, GPU Direct Disabled warnings): [forums.developer.nvidia.com](https://forums.developer.nvidia.com/t/nccl-rocev2-issues-with-duplicates-gpu-direct-rdma-and-fusion/351460)

- **NVIDIA Forum: nvidia-peermem on DGX Spark** (fails to load, architecture-specific reason): [forums.developer.nvidia.com](https://forums.developer.nvidia.com/t/gpu-direct-rdma-not-working-on-dgx-spark-systems-nvidia-peermem-module-fails-to-load/349837)

- **NVIDIA Forum: ASUS GX10 vs DGX Spark** (identical hardware, storage difference): [forums.developer.nvidia.com](https://forums.developer.nvidia.com/t/should-i-buy-asus-gx10-instead-nvidia-dgx-spark/347717)

- **OpenVINO Model Server (OVMS)**: [github.com/openvinotoolkit/model_server](https://github.com/openvinotoolkit/model_server)

- **OVMS Embeddings Demo** (incl. model export script): [github.com/openvinotoolkit/model_server/tree/main/demos/embeddings](https://github.com/openvinotoolkit/model_server/tree/main/demos/embeddings)

- **Optimum Intel (OpenVINO Export)**: [huggingface.co/docs/optimum/intel/openvino/export](https://huggingface.co/docs/optimum/main/en/intel/openvino/export)

- **TEI GitHub**: [github.com/huggingface/text-embeddings-inference](https://github.com/huggingface/text-embeddings-inference)

- **Podman CDI (Container Device Interface) Docs**: [docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/cdi-support.html](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/cdi-support.html)

- **podman-compose**: [github.com/containers/podman-compose](https://github.com/containers/podman-compose)

- **GLM-5-FP8** (744B MoE, official): [huggingface.co/zai-org/GLM-5-FP8](https://huggingface.co/zai-org/GLM-5-FP8)

- **GLM-5-GGUF** (Unsloth quantizations incl. Q4_K_XL): [huggingface.co/unsloth/GLM-5-GGUF](https://huggingface.co/unsloth/GLM-5-GGUF)

- **Running GLM-5 locally** (Unsloth guide, VRAM requirements): [unsloth.ai/docs/models/glm-5](https://unsloth.ai/docs/models/glm-5)

- **GLM-5 Specs & GPU VRAM Requirements**: [apxml.com/models/glm-5](https://apxml.com/models/glm-5)

- **GLM-4.7-FP8 on 4× DGX Spark** (SGLang + EAGLE, MoE kernel tuning): [forums.developer.nvidia.com](https://forums.developer.nvidia.com/t/running-glm-4-7-fp8-355b-moe-on-4x-dgx-spark-with-sglang-eagle-speculative-decoding/359256)

- **NVFP4 on DGX Spark** (20% faster than AWQ, Blackwell-native): [forums.developer.nvidia.com](https://forums.developer.nvidia.com/t/we-unlocked-nvfp4-on-the-dgx-spark-20-faster-than-awq/361163)

- **GPU Time-Slicing in Kubernetes** (NVIDIA GPU Operator Docs): [docs.nvidia.com](https://docs.nvidia.com/datacenter/cloud-native/gpu-operator/latest/gpu-sharing.html)
