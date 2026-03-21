# DGX Spark — Skalierung, Modell-Alternativen & Ausblick

_Ausgelagert aus [DGX Spark Setup](DGX%20Spark%20Setup.md) — enthält Zukunftsplanung, Modell-Alternativen, Switch-Kalkulation und Quellenverzeichnis._

---

## Offene Punkte & Zukunft

### Kurzfristig relevant

- **SGLang MoE-Kernel-Tuning**: Falls du OutOfResources-Fehler bei MoE-Modellen bekommst (GB10 hat 101KB Shared Memory statt der von SGLang erwarteten 147KB), gibt es fertige Kernel-Configs auf GitHub (BTankut/dgx-spark-sglang-moe-configs). Das Tuning dauert ~9 Stunden, muss aber nur einmal gemacht werden.

- **EAGLE Speculative Decoding**: Sobald EAGLE3-Draft-Modelle für Qwen3-235B verfügbar sind, kann die Decode-Speed nochmal 1,5-2× steigen. SGLang unterstützt das nativ. In der Implementierung ist Speculative Decoding als Ansible-Variable vorbereitet (`sglang_speculative_enabled: false`, `NEXTN`-Algorithmus, 3 Steps, topk=1, 4 Draft-Tokens) — Aktivierung durch Ändern eines Flags.

- **Qwen3-Coder-Next FP8**: Läuft auf einem einzelnen Spark mit ~43 t/s — hervorragend als schnelles Zweitmodell für Code-Aufgaben.


### Mittelfristig

- **NVFP4-Kernels für SM121**: NVIDIA arbeitet daran. Wenn verfügbar, könnten FP4-Modelle deutlich schneller laufen — Qwen3-235B in NVFP4 würde potenziell auf einem einzelnen Spark passen.

- **vLLM Stabilisierung**: eugr arbeitet an einem Nightly-Build-System mit automatisierten Tests. Wenn das steht, könnte vLLM wieder eine Option werden — aktuell nicht empfohlen.

- **llama.cpp NCCL-Support**: In der Community aktiv diskutiert. Wenn das kommt, hätte man die Stabilität von llama.cpp mit der Bandbreite von NCCL — wäre ein Gamechanger.

### Ausblick: Mehrere AI-Pods pro DGX Spark Node

Das aktuelle Setup nutzt **einen AI-Pod pro Node** (SGLang Head auf Spark 1, Worker auf Spark 2). Falls zukünftig mehrere kleinere Modelle parallel auf einem Spark laufen sollen, sind zwei Engpässe zu lösen:

**Engpass 1 — GPU (limitierender Faktor):**

Der GB10 hat genau **eine** Blackwell-GPU mit 128 GB Unified Memory. GPU-Sharing-Optionen:

| Methode | GB10-Support? | Memory-Isolation? | Bewertung |
|---|---|---|---|
| **MIG** (Multi-Instance GPU) | ❌ Nicht supported | ✅ Ja | MIG ist nur für A100/H100/B200 — GB10 hat Unified Memory |
| **Time-Slicing** | ✅ Funktioniert | ❌ Nein | Pods teilen vollen GPU-Speicher; wenn Pod A 120 GB belegt → Pod B bekommt OOM |
| **MPS** (Multi-Process Service) | ✅ Funktioniert | ❌ Nein | Concurrent Kernels, gleiche Memory-Einschränkung |

> [!success] Implementiert
> GPU Time-Slicing ist im produktiven K3s-Cluster aktiv mit `nvidia_gpu_timeslice_replicas: 4`. Das erlaubt Ollama (bge-m3 Embedding) und SGLang parallel als separate GPU-Consumer auf demselben Node.

Time-Slicing macht vor allem Sinn für **kleinere Co-Prozesse** (wie Embedding-Modelle) neben dem Haupt-LLM. Bei Qwen3-235B-FP8 (~120 GB) bleibt wenig Spielraum für ein zweites großes Modell — aber leichtgewichtige Services wie Ollama (bge-m3, ~1,2 GB) funktionieren problemlos.

Konfiguration über den NVIDIA Device Plugin (kein GPU Operator nötig):
```yaml
# ConfigMap für nvidia-device-plugin (produktive Konfiguration)
sharing:
  timeSlicing:
    renameByDefault: false
    resources:
      - name: nvidia.com/gpu
        replicas: 4    # 4 Pods können die GPU teilen (implementiert)
```

**Engpass 2 — Netzwerk:**

Das aktuelle `host-device`-CNI **verschiebt** das physische Interface in den Pod-Namespace — nur **ein** Pod kann es nutzen. Alternativen für Multi-Pod:

| Methode | RDMA-fähig? | Aufwand | Bemerkung |
|---|---|---|---|
| **`hostNetwork: true`** | ✅ Ja | Niedrig | Alle Pods sehen alle Host-Interfaces, aber Port-Konflikte möglich |
| **`macvlan`** + RDMA Shared Device Plugin | ✅ Teilweise | Mittel | `macvlan` für IP, shared RDMA-Device als K8s-Ressource |
| **SR-IOV** | ✅ Ja (pro VF) | Hoch | Jeder Pod bekommt eigene Virtual Function mit eigenem RDMA-Device |

**SR-IOV auf GB10 — offenes Risiko**: Die ConnectX-7 im GB10 läuft im Multi-Host-Modus (2× PCIe Gen5 x4). Ob SR-IOV in dieser Konfiguration funktioniert, ist **weder von NVIDIA dokumentiert noch von der Community getestet**. VFs müssten über zwei PCIe-Root-Complexes verteilt werden, was bei klassischem SR-IOV nicht vorgesehen ist. Außerdem ist GPUDirect RDMA auf dem Spark ohnehin nicht supported — der Hauptvorteil von per-VF-GPUDirect entfällt damit.

Bei SR-IOV wäre der **NVIDIA Network Operator** tatsächlich sinnvoll — er managed VF-Erstellung, SR-IOV Device Plugin und Multus NADs pro VF automatisch.

**Empfehlung für Multi-Pod**: `hostNetwork: true` + GPU Time-Slicing ist der pragmatischste Weg — kein SR-IOV, kein Network Operator, minimaler Config-Aufwand. Wird erst relevant, wenn deutlich kleinere Modelle parallel laufen sollen.


### Langfristig

- **NVIDIA Dynamo + SGLang**: Disaggregated Serving (separate Prefill- und Decode-Phasen auf verschiedenen Nodes) könnte auf Multi-Spark-Setups einen großen Performance-Sprung bringen.

- **Nächste Modellgeneration**: GLM-4.7, Qwen3.5, und kommende MoE-Modelle werden zunehmend für Spark optimiert. Der Trend geht klar in Richtung: größere Modelle, bessere Quantisierung, bessere Kernel-Unterstützung.

> [!success] Aktueller Stand (März 2026)
> Die Hardware ist absolut solide. Der komplette Software-Stack ist produktiv als K3s-Cluster implementiert und via Ansible automatisiert (`dgxarley`-Repo). SGLang 0.5.9 mit NCCL-Multi-Node, HAProxy-Sidecar-Pattern, Modell-Profil-System, cert-manager/TLS, Prometheus/Grafana/Loki-Monitoring, PostgreSQL/pgvector und DNS-Registrierung sind deployed und stabil.

### Modell-Alternative: GLM-5 (744B MoE, 40B aktiv)

GLM-5 (Zhipu AI / Z.ai, Release Februar 2026) ist mit 744B Parametern und 40B aktiven Parametern pro Token (256 Experten, 8 aktiv) deutlich größer als Qwen3-235B. Es nutzt DeepSeek Sparse Attention (DSA) und unterstützt 204K-Token-Kontext. In Benchmarks liegt GLM-5 vor allem bei Coding und Agentic-Tasks vorn.

**Speicherbedarf vs. DGX Spark Konfigurationen:**

| Quantisierung | Gewicht-Größe | Sparks nötig | KV-Cache Headroom | Engine |
|---|---|---|---|---|
| FP8 ([zai-org/GLM-5-FP8](https://huggingface.co/zai-org/GLM-5-FP8)) | ~800 GB | 8 | ~224 GB | SGLang (TP=8) — braucht 2× CRS812 oder SN3700 |
| **AWQ** ([QuantTrio/GLM-5-AWQ](https://huggingface.co/QuantTrio/GLM-5-AWQ)) | **~392 GB** | **4–5** ✅ | **~120–248 GB** | **SGLang (TP=4/5, NCCL)** |
| Q8_0 (GGUF) | ~801 GB | 7 ❌ | ~95 GB | llama.cpp |
| Q5_K_M (GGUF) | ~535 GB | 5 | ~105 GB | llama.cpp |
| **UD-Q4_K_XL** ([unsloth/GLM-5-GGUF](https://huggingface.co/unsloth/GLM-5-GGUF)) | **~431 GB** | **4** ✅ | **~81 GB** | llama.cpp (RPC) |
| Q4_K_M (GGUF) | ~456 GB | 4 | ~56 GB | llama.cpp |
| NVFP4 (Blackwell-nativ) | ~400 GB (geschätzt) | 4 | ~112 GB | SGLang — **existiert noch nicht** |
| UD-Q2_K_XL (GGUF) | ~281 GB | 3 | ~103 GB | llama.cpp — Qualität fragwürdig |

**FP8 mit 8× Spark ist technisch machbar** — SGLang unterstützt Multi-Node nativ via NCCL, und die 200 Gbit/s Interconnect reichen für die All-to-All-Kommunikation der 40B aktiven MoE-Parameter. Allerdings: 8 Sparks brauchen 2× MikroTik CRS812 oder einen Enterprise-Switch (Mellanox SN3700). Das HF-empfohlene TP=8-Setup ist 1:1 abbildbar.

**AWQ auf 4–5× Spark ist der Sweet Spot für SGLang:**

Die AWQ-Variante ([QuantTrio/GLM-5-AWQ](https://huggingface.co/QuantTrio/GLM-5-AWQ)) mit ~392 GB Gewichten läuft nativ in SGLang über NCCL — kein llama.cpp RPC nötig. Bei 5 Sparks (640 GB) bleiben ~248 GB für KV-Cache, genug für lange Kontexte. Der MoE-Vorteil (nur 40B aktive Parameter pro Token) hält den Cross-Node-Traffic über 200 Gbit/s handhabbar. Expert Parallelism (`--enable-expert-parallel`) verteilt die Experten optimal auf die Nodes. Qualitätsverlust AWQ vs. FP8 ist bei 744B-Modellen typischerweise marginal.

**4-bit GGUF (UD-Q4_K_XL) auf 4× Spark als llama.cpp-Alternative:**

Konkretes Setup: 4× ASUS Ascent GX10, MikroTik CRS812 Switch (siehe Skalierung unten), 4× QSFP56 DAC. Inference über llama.cpp RPC (nicht SGLang/NCCL, da GGUF):

```bash
# Auf Spark 2–4 (RPC-Worker):
llama-server --rpc-host 0.0.0.0 --rpc-port 50052

# Auf Spark 1 (Head):
llama-server \
  --model GLM-5-UD-Q4_K_XL.gguf \
  --rpc spark2:50052,spark3:50052,spark4:50052 \
  --n-gpu-layers 999 \
  --ctx-size 32768 \
  --port 8000
```

> [!warning] Abwägung: GLM-5 vs. Qwen3-235B@FP8
>
> | | **Qwen3-235B-FP8** (aktuell) | **GLM-5-AWQ** | **GLM-5-Q4_K_XL** |
> |---|---|---|---|
> | **Sparks** | 2 | 4–5 | 4 |
> | **Decode-Speed** | ~25 t/s (SGLang+NCCL/RDMA) | ~15–20 t/s (SGLang+NCCL/RDMA) | ~10–15 t/s (llama.cpp RPC/TCP) |
> | **Quantisierungs-Qualität** | FP8 ≈ 99% von BF16 | AWQ ≈ 95–97% von BF16 | Q4 ≈ 92–95% von BF16 |
> | **Aktive Parameter** | 22B (MoE) | 40B (MoE) | 40B (MoE) |
> | **Max. Kontext** | ~65K | ~64K (4 Sparks) / ~128K+ (5 Sparks) | ~32K (bei 81 GB KV-Cache Headroom) |
> | **Engine** | SGLang (NCCL) | SGLang (NCCL) | llama.cpp (RPC/TCP) |
>
> Die AWQ-Variante ist der pragmatischste GLM-5-Upgrade-Pfad: SGLang-nativ (NCCL statt TCP), deutlich bessere Quantisierungs-Qualität als Q4 GGUF, und mit 5 Sparks genug KV-Cache für lange Kontexte. GLM-5 hat mehr aktive Parameter (40B vs. 22B) — bei Coding und Agentic-Tasks liegt es vorn, bei General Reasoning ist es eng.
>
> **Empfehlung**: Bei 2 Sparks bei Qwen3-235B-FP8 bleiben. Für den GLM-5-Upgrade-Pfad auf 4–5 Sparks gibt es jetzt zwei konkrete Optionen: **GLM-5-AWQ** (sofort verfügbar, SGLang-nativ) oder auf **GLM-5-NVFP4** warten (Blackwell-nativ, ~20% schneller als AWQ). GLM-4.7-NVFP4 [existiert bereits](https://forums.developer.nvidia.com/t/running-glm-4-7-fp8-355b-moe-on-4x-dgx-spark-with-sglang-eagle-speculative-decoding/359256), GLM-5-NVFP4 wird folgen.

---

## Skalierung auf 4× DGX Spark: Switch & Kosten

### Warum ein Switch nötig ist

Für 2 Sparks reicht ein direktes DAC-Kabel (Point-to-Point). Für 3+ Sparks ist ein Switch zwingend erforderlich — Daisy-Chaining ist nicht möglich. Jeder Spark hat einen ConnectX-7 Port (200GbE, QSFP56/QSFP112-kompatibel).

### Switch-Empfehlung: MikroTik CRS812 DDQ

|Eigenschaft|Wert|
|---|---|
|**Modell**|CRS812-8DS-2DQ-2DDQ-RM|
|**Ports**|8× 50G SFP56 + **2× 200G QSFP56** + **2× 400G QSFP56-DD**|
|**Switching-Kapazität**|1,6 Tbps unidirektional|
|**Formfaktor**|1U Rackmount, Dual-PSU (redundant)|
|**CPU**|Quad-Core ARM 2 GHz, 4 GB RAM|
|**RouterOS**|v7, Layer 3, Hardware-Offloading|

**Community-bestätigt**: Ein Nutzer im NVIDIA Developer Forum betreibt ein 8-Node-Spark-Cluster mit 2× CRS812 DDQ erfolgreich. Kommentar: *"It pretty much just works."*

### Verkabelung für 4× Spark

```
MikroTik CRS812 DDQ
├── QSFP56 Port #1 (200G) ──DAC──→ Spark 1
├── QSFP56 Port #2 (200G) ──DAC──→ Spark 2
├── QSFP-DD Port #1 (400G) ──Breakout──→ Spark 3 + Spark 4
└── QSFP-DD Port #2 (400G) ──(frei für Erweiterung auf 6 Sparks)
```

- 2 Sparks an die nativen **200G QSFP56-Ports** (je ein 200G DAC-Kabel)
- 2 Sparks über ein **400G→2×200G Breakout-Kabel** am QSFP-DD-Port
- Der zweite QSFP-DD-Port bleibt frei → **skalierbar auf bis zu 6 Sparks** ohne zweiten Switch

**Wichtige Konfiguration:**
- Jumbo Frames aktivieren (MTU 9000+)
- Auto-Negotiation auf den QSFP-DD-Ports deaktivieren
- QSFP-DD-Port manuell auf 2×200G splitten

### Was bringt 4× Spark?

|Eigenschaft|2× Spark (aktuell)|4× Spark|
|---|---|---|
|**Unified Memory**|256 GB|**512 GB**|
|**FP8-Modell + KV-Budget**|237 GB Modell + ~14 GB KV|237 GB Modell + **~270 GB KV**|
|**Kontextlänge (FP8-KV)**|~140K|**Volle 262K** (und mehr Headroom)|
|**Decode-Speed (MoE)**|~25 t/s|~30–50 t/s (bessere Parallelisierung)|
|**Prefill-Throughput**|~23K t/s|**Deutlich höher** (linear skalierend)|
|**Alternative Modelle**|Qwen3-235B max.|GLM-4.7-FP8 (355B MoE), GLM-5-AWQ (744B MoE) möglich|
|**SGLang-Startbefehl**|`--tp 2 --nnodes 2`|`--tp 4 --nnodes 4`|

**Wichtigster Vorteil**: Mit 4× Spark und FP8-Weights hättest du genug Speicher für das **volle 262K-Kontextfenster** — ohne Kompromisse bei der Modellqualität. Alternativ könntest du noch größere Modelle wie GLM-4.7-FP8 (355B) fahren.

### Enterprise-Alternative: Mellanox SN3700

Falls du über 6 Sparks hinaus skalieren willst oder Enterprise-Support benötigst:

|Eigenschaft|MikroTik CRS812 DDQ|Mellanox SN3700 (refurbished)|
|---|---|---|
|**Max. 200G-Ports**|4 (2 nativ + 2 via Breakout)|**32 nativ**|
|**Max. Sparks (1 Switch)**|6|32|
|**RoCE/RDMA**|✅ Grundlegend|✅ Nativ optimiert, GPUDirect|
|**Latenz**|Standard|425ns Port-to-Port|
|**Support**|Community|Enterprise (10 Jahre Garantie)|

Für ein Heim-/Büro-Setup mit ≤6 Sparks ist der CRS812 DDQ die klare Wahl. Den SN3700 brauchst du nur bei ernsthafter Datacenter-Skalierung.

---

## Quellen & Referenzen

- **SGLang Community Image**: `scitrera/dgx-spark-sglang:0.5.9-t5` (DockerHub) — 0.5.8-t5 weiterhin verfügbar für Modelle ohne GDN-Attention

- **SGLang MoE Kernel Configs für GB10**: github.com/BTankut/dgx-spark-sglang-moe-configs

- **vLLM Community Docker (Referenz)**: github.com/eugr/spark-vllm-docker

- **vLLM Pre-built Images**: github.com/mark-ramsey-ri/vllm-dgx-spark

- **NVIDIA DGX Spark Playbooks**: build.nvidia.com/spark

- **NVIDIA SGLang Playbook**: build.nvidia.com/spark/sglang

- **Open WebUI SearXNG Doku**: docs.openwebui.com/features/web-search/searxng

- **Open WebUI Pipelines**: github.com/open-webui/pipelines

- **llama.cpp Spark Performance**: github.com/ggml-org/llama.cpp/discussions/16578

- **MikroTik CRS812 DDQ (Produktseite)**: mikrotik.com/product/crs812_ddq

- **MikroTik CRS812 DDQ Review (ServeTheHome)**: servethehome.com — CRS812-8DS-2DQ-2DDQ-RM Review

- **NVIDIA Forum: Multi-Spark Switch-Empfehlungen**: forums.developer.nvidia.com/t/connecting-multiple-dgx-spark-units-ethernet-switch-recommendations/345839

- **NVIDIA Forum: 6× Spark Setup (Community-Erfahrungen)**: forums.developer.nvidia.com/t/6x-spark-setup/354399

- **NVIDIA DGX Spark Stacking Guide**: docs.nvidia.com/dgx/dgx-spark/spark-clustering.html

- **K3s Dokumentation**: [k3s.io/docs](https://docs.k3s.io/)

- **K3s ARM64 Support**: [docs.k3s.io/advanced#running-on-arm64](https://docs.k3s.io/advanced#running-on-arm64)

- **NVIDIA K8s Device Plugin**: [github.com/NVIDIA/k8s-device-plugin](https://github.com/NVIDIA/k8s-device-plugin)

- **Multus CNI** (Dual-NIC für Pods): [github.com/k8snetworking/multus-cni](https://github.com/k8snetworking/multus-cni)

- **Multus `host-device` Plugin** (RDMA-fähiges Interface-Passthrough): [containernetworking.github.io/plugins/main/host-device](https://www.cni.dev/plugins/current/main/host-device/)

- **ASUS Ascent GX10 Review** (OEM-Variante der DGX Spark): [servethehome.com](https://www.servethehome.com/asus-ascent-gx10-review-a-new-nvidia-gb10-solution/)

- **GB10 ConnectX-7 200GbE Networking — Dual-PCIe-x4-Architektur** (ServeTheHome Deep-Dive): [servethehome.com](https://www.servethehome.com/the-nvidia-gb10-connectx-7-200gbe-networking-is-really-different/)

- **NVIDIA Forum: ConnectX-7 NIC in DGX Spark** (Interface-Namen, Multi-Host-Modus, NCCL-Aggregation): [forums.developer.nvidia.com](https://forums.developer.nvidia.com/t/connectx-7-nic-in-dgx-spark/350417)

- **NVIDIA Forum: ConnectX-7 Bonding zwischen Sparks** (balance-rr, MTU 9000, NCCL-Ergebnisse): [forums.developer.nvidia.com](https://forums.developer.nvidia.com/t/bonding-spark-connectx-7-ports-between-two-sparks-with-jumbo-frames-works-fine/349872)

- **NVIDIA FAQ: GPUDirect RDMA auf DGX Spark nicht supported**: [nvidia.custhelp.com](https://nvidia.custhelp.com/app/answers/detail/a_id/5780/~/is-gpudirect-rdma-supported-on-dgx-spark)

- **NVIDIA Forum: NCCL/RoCEv2 auf DGX Spark** (Transport-Erkennung, GPU Direct Disabled-Warnings): [forums.developer.nvidia.com](https://forums.developer.nvidia.com/t/nccl-rocev2-issues-with-duplicates-gpu-direct-rdma-and-fusion/351460)

- **NVIDIA Forum: nvidia-peermem auf DGX Spark** (lädt nicht, architekturbedingt): [forums.developer.nvidia.com](https://forums.developer.nvidia.com/t/gpu-direct-rdma-not-working-on-dgx-spark-systems-nvidia-peermem-module-fails-to-load/349837)

- **NVIDIA Forum: ASUS GX10 vs DGX Spark** (identische Hardware, Storage-Unterschied): [forums.developer.nvidia.com](https://forums.developer.nvidia.com/t/should-i-buy-asus-gx10-instead-nvidia-dgx-spark/347717)

- **OpenVINO Model Server (OVMS)**: [github.com/openvinotoolkit/model_server](https://github.com/openvinotoolkit/model_server)

- **OVMS Embeddings Demo** (inkl. Modell-Export-Script): [github.com/openvinotoolkit/model_server/tree/main/demos/embeddings](https://github.com/openvinotoolkit/model_server/tree/main/demos/embeddings)

- **Optimum Intel (OpenVINO Export)**: [huggingface.co/docs/optimum/intel/openvino/export](https://huggingface.co/docs/optimum/main/en/intel/openvino/export)

- **TEI GitHub**: [github.com/huggingface/text-embeddings-inference](https://github.com/huggingface/text-embeddings-inference)

- **Podman CDI (Container Device Interface) Docs**: [docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/cdi-support.html](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/cdi-support.html)

- **podman-compose**: [github.com/containers/podman-compose](https://github.com/containers/podman-compose)

- **GLM-5-FP8** (744B MoE, offiziell): [huggingface.co/zai-org/GLM-5-FP8](https://huggingface.co/zai-org/GLM-5-FP8)

- **GLM-5-AWQ** (744B MoE, 4-bit AWQ): [huggingface.co/QuantTrio/GLM-5-AWQ](https://huggingface.co/QuantTrio/GLM-5-AWQ)

- **GLM-5-GGUF** (Unsloth-Quantisierungen inkl. Q4_K_XL): [huggingface.co/unsloth/GLM-5-GGUF](https://huggingface.co/unsloth/GLM-5-GGUF)

- **GLM-5 Lokal ausführen** (Unsloth Guide, VRAM-Anforderungen): [unsloth.ai/docs/models/glm-5](https://unsloth.ai/docs/models/glm-5)

- **GLM-5 Specs & GPU VRAM Requirements**: [apxml.com/models/glm-5](https://apxml.com/models/glm-5)

- **GLM-4.7-FP8 auf 4× DGX Spark** (SGLang + EAGLE, MoE-Kernel-Tuning): [forums.developer.nvidia.com](https://forums.developer.nvidia.com/t/running-glm-4-7-fp8-355b-moe-on-4x-dgx-spark-with-sglang-eagle-speculative-decoding/359256)

- **NVFP4 auf DGX Spark** (20% schneller als AWQ, Blackwell-nativ): [forums.developer.nvidia.com](https://forums.developer.nvidia.com/t/we-unlocked-nvfp4-on-the-dgx-spark-20-faster-than-awq/361163)

- **GPU Time-Slicing in Kubernetes** (NVIDIA GPU Operator Docs): [docs.nvidia.com](https://docs.nvidia.com/datacenter/cloud-native/gpu-operator/latest/gpu-sharing.html)
