# SGLang v0.5.11 — Relevante Änderungen seit v0.5.10

Quelle: [Release Notes v0.5.11](https://github.com/sgl-project/sglang/releases/tag/v0.5.11) · Veröffentlicht 2026-05-05 · Diff: [`v0.5.10.post1...v0.5.11`](https://github.com/sgl-project/sglang/compare/v0.5.10.post1...v0.5.11)

Dieses Dokument hebt die Änderungen hervor, die für unseren DGX-Spark-Cluster (5 Nodes, GB10 / SM121, NVFP4-MoE, Multi-Node über RoCE/QSFP) relevant sind. Reine ROCm/Ascend/CPU/Diffusion-Themen werden nur kurz erwähnt.

---

## 1. Toolchain-Sprung — CUDA 13 + Torch 2.11

- **Default CUDA: 13.0** quer durch SGLang, sgl-kernel und alle Docker-Images (PRs #21247, #24162, #24183, #23593, #23119; Tracking-Issue #21498).
- **PyTorch: 2.9 → 2.11** (PR #21247).
- **sgl-kernel: 0.4.1.post1 → 0.4.2** (PRs #23720, #23733, #24170).
- **FlashInfer: 0.6.7.post2 → 0.6.8.post1** (PR #23281).

**Relevanz für uns:**
- Unser custom SM121-Image basiert aktuell auf CUDA 13.1 + Torch 2.10 (siehe Memory `reference_sm121_build_base_regression.md`). Mit 0.5.11 wäre **upstream selber auf CUDA 13.0 + Torch 2.11** — der Performance-Regression-Workaround (~45% slowdown durch Base-Image-Fallback) sollte sich damit potenziell lösen lassen, sobald wir auf einem 0.5.11-basierten Upstream-Image rebuilden.
- Achtung: Vor einem Bump unbedingt die GB10/SM121 NVFP4-Pfadtests (Qwen3-235B, GLM-4.7, GLM-5, Nemotron-3) gegenfahren — der `cutlass_moe_fp4` ist auf SM121 historisch instabil (siehe CLAUDE.md, NVFP4-Abschnitt).

## 2. Speculative Decoding — V2 + DFLASH

- **Spec V2 ist jetzt Default**, inkl. Overlap-Scheduling, das CPU-Overhead versteckt (PR #21062). Beschleunigt EAGLE / MTP / DFLASH-Pfade spürbar pro Step.
- **DFLASH** — neuer Spec-Decode-Kernel der Kernel-Community: initiale Unterstützung (#22077), zusätzliche Modell-Backends (#22358), AMD ROCm (#22342), Doku (#23553).
- Penalty-Support für Spec V2 (#22049), adaptive `speculative_num_steps` für EAGLE topk=1 (#21599), Piecewise CUDA-Graph + Spec gemeinsam erlaubt (#22128), Eagle3/DFLASH Aux-Hidden-State Capture in CUDA-Graph-Init gefixt (#22836).

**Relevanz:** Für die MTP-Konfigurationen unserer Modellprofile (GLM-4.7 MTP, GLM-5 MTP) sollte sich Spec V2 transparent auswirken — ggf. über `--speculative-algorithm`/`--speculative-num-steps` neu vermessen.

## 3. PD Disaggregation — Decode-Side Radix-Cache

- **Decode-Side Prefix Caching unter PD-Disaggregation** (PR #19746) — schließt eine echte Lücke: bislang ging der Radix-Hit auf der Decode-Seite verloren, sobald Prefill/Decode getrennt waren.
- Mooncake Incremental Transfer (#24257), `PrefillDelayer` in disagg-prefill (#23588).
- NIXL: heterogenes TP-KV-Transfer für Non-MLA (Qwen3.5 Step 1/2: #22145, Mamba-Slice Step 2/2: #22240).
- Fixes: IntraNode NVLink, MTP-Layer KV-Transfer, Disagg-Prefill DP-Rank-Resolution (#23252, #23539, #22901, #22990).

**Relevanz:** Wir fahren aktuell PD nicht disaggregiert (Head + 3 Worker, alle TP-tight gekoppelt). Falls wir später für Long-Context-Workloads PD-Disagg ausprobieren, ist das jetzt erstmals attraktiv.

## 4. Performance-Kernels

- **FA3-Kernels von der Kernel-Community** (PR #20796) — drop-in, neben FA4. Gibt uns eine zusätzliche Hochperformance-Option.
- Precompute FA3 `scheduler_metadata` — eliminiert Per-Layer-Prepare-Kosten (#21104).
- Attention-DtoD-Copy eliminiert (FA bekommt pre-allocated Output, #21985).
- KV-Cache im FA-Backend für Embedding-Mode geskippt (#21971).
- O(1) `RadixKey`-View für EAGLE-Bigram (#23106).
- PCG Inductor-Path für FP8-Modelle (#23227).
- Combo-Kernels für horizontale Fusion (#21977).
- Gemma-4 VLM mit PCG + fused RMSNorm + Residual-Add + Scalar (#24048).
- `gemma_weight` precomputed (#22673).
- torch.compile-Fusion für Top-K-Postprocessing wiederhergestellt (#21771).
- NSA-Indexer: weniger Kernels/Copies (#22232).

## 5. Context Parallel & Parallelism

- **All-Reduce + RMSNorm Fusion unter CP** (PR #21249) — End-to-End-Speedup.
- **`moe_dp_size = 1` mit beliebigem `attention_cp_size`** (PR #22003) — MoE- und Attention-Parallelism können jetzt unabhängig getuned werden.
- All-Reduce Fusion für DSA-Modelle (#22390).
- `reduce_scatterv` ersetzt All-Reduce + dp_scatter bei DP-Attention (#22642).
- Step3.5: All-Reduce in MoE-Layern optimiert (#22773).

**Relevanz:** Auf 4×Spark mit TP=4 / EP=4 ist CP für uns (noch) nicht aktiviert; die Fusion könnte aber für Long-Context-Profile (256k+) interessant werden.

## 6. MoE — FlashInfer CuteDSL Backend

- Neue **`FlashInferCuteDslMoE`-Layer** für den Standard-FP4-MoE-Pfad (PR #21339) — zusätzliche fused-MoE-Option neben `cutlass`, `triton`, `flashinfer_cutlass`.

**Relevanz:** Genau hier liegt unser Schmerzpunkt. Aktuell:
- Default für die meisten NVFP4-MoEs auf SM121: `moe_runner_backend: flashinfer_cutlass`.
- Ausnahme `nvidia/Qwen3.5-397B-A17B-NVFP4`: muss `cutlass` direkt fahren (fi_cutlass crasht in 12/12 Matrix-Reihen).
- Ausnahme `Minimax-M2.5 @ PP=4`: `triton`.

Mit `FlashInferCuteDslMoE` haben wir potenziell einen **vierten Backend-Kandidaten** für die Crash-Reihen. Das gehört in einen Test-Run (Qwen3.5-397B + GLM-4.7 + Nemotron-3) sobald wir auf 0.5.11 sind. → Action-Item.

## 7. Neue Modelle (Day-0)

Für unseren Cluster direkt einsetzbar:

| Modell                   | PRs                                    | Cookbook                                                                                            |
|--------------------------|----------------------------------------|-----------------------------------------------------------------------------------------------------|
| **Gemma 4**              | #21952, #22079, #24048, #22842         | [docs.sglang.io/cookbook/.../Gemma4](https://docs.sglang.io/cookbook/autoregressive/Google/Gemma4)  |
| **GLM-5.1**              | #22543, #23037                         | [cookbook.sglang.io/.../GLM-5.1](https://cookbook.sglang.io/autoregressive/GLM/GLM-5.1)             |
| **Qwen3.6**              | #23486                                 | [docs.sglang.io/.../Qwen3.6](https://docs.sglang.io/cookbook/autoregressive/Qwen/Qwen3.6)           |
| **MiMo-V2.5 / V2.5-Pro** | #23808, #23811, #23851, #23945, #24118 | [.../MiMo-V2.5](https://docs.sglang.io/cookbook/autoregressive/Xiaomi/MiMo-V2.5)                    |
| **Ling-2.6-Flash**       | #23947                                 | [.../Ling-2.6](https://docs.sglang.io/cookbook/autoregressive/InclusionAI/Ling-2.6)                 |
| **Mistral Medium 3.5**   | —                                      | [.../Mistral-Medium-3.5](https://docs.sglang.io/cookbook/autoregressive/Mistral/Mistral-Medium-3.5) |
| **Kimi-K2.6**            | #23394, #23408                         | [.../Kimi-K2.6](https://docs.sglang.io/cookbook/autoregressive/Moonshotai/Kimi-K2.6)                |
| Hunyuan v3 (preview)     | #23533                                 | [.../Hunyuan3-Preview](https://docs.sglang.io/cookbook/autoregressive/Tencent/Hunyuan3-Preview)     |

**Action-Item:** Für Gemma 4, GLM-5.1, Qwen3.6 jeweils ein Modellprofil unter `roles/k8s_dgx/model_profiles/` anlegen, sobald Image-Bump erfolgt ist. Wir laufen aktuell auf einem `gemma4`-Patch-Image (`0.5.10-20260429-gemma4-sm121-dev1`); 0.5.11 hat Gemma 4 nativ — der Patch wird redundant.

Nicht direkt relevant für uns: FLUX-Diffusion-Modelle, LTX-2.3, Voxtral, Parakeet, Moss-VL.

## 8. LoRA — auch für DeepSeek-V3 / Kimi-K2

- DeepSeek-V3 MLA LoRA (#22323), Kimi K2 LoRA (#22381). Damit ist Adapter-FT auf MLA-MoEs möglich.
- LoRADrainer gegen hohe P99 TTFT (#17913).
- Decoupled LoRA-MoE-Backend mit Marlin (#21858), Virtual Experts (#22122, #24007).
- Dual-MoE-CUDA-Graph-Capture für LoRA/NoLoRA-Batches (#22809).

**Relevanz:** Aktuell nutzen wir kein LoRA, daher nur als Option im Hinterkopf.

## 9. Observability

- Pending-Token-Count im Prefill-Log und in `get_load` (#22480).
- OpenTelemetry-Tracing für Speculative Decoding (#19545), Pipeline Parallelism (#23169), DiffGenerator (#21254).
- **Prometheus-Metrics-Endpoint im gRPC-Mode** (#20801).
- HTTP-Sidecar-Endpoints + FlushCache-gRPC im gRPC-Mode (#22500).
- **Raw-KV-Cache-Pool-Token-Counts als Prometheus-Gauges** (#22726).

**Relevanz:** Für unsere `promstack`-Integration zwei interessante Metriken — KV-Pool-Auslastung als Gauge und Pending-Token-Count. Lohnt sich, im Grafana-Dashboard zu ergänzen.

## 10. Quantisierung

- **NVFP4 KV-Cache** — Quant-Strategy-Abstraktion + Kernel (PR #21954).
- DeepSeek-R1-0528 w4a8 + DeepEP Low-Latency FP8 Dispatch (#22316).
- MXFP8 sm100 Path-Cleanup (#21881).
- GLM-5/5.1 MXFP4 Checkpoint-Inference-Compatibility-Fix (#22543).
- MXFP4 Dense auf AMD CDNA2/CDNA3 — eingeführt (#19143), in #23031 wieder zurückgerollt, Follow-Up offen.

**Relevanz:** **NVFP4 KV-Cache ist für uns spannend** — bislang fahren wir KV im Default-Datentyp (FP8 oder bfloat16, je nach Modell). Eine NVFP4-KV-Variante würde den KV-Cache-Footprint nochmal halbieren (bei 235B-MoE @ 256k Kontext signifikant). Bedarf Validierung gegen unsere bestehenden Profile.

## 11. Sicherheit

- **CVE-2026-5760** behoben (PR #23660).
- Trivy-CVEs + Cubin-Download-403 im Docker-Image gefixt (#22322).

**Relevanz:** Hard requirement, sobald wir auf 0.5.11 gehen — vorher CVE-Details ansehen, ob v0.5.10.post1 expose-relevant ist (Cluster ist intern, aber via Hermes/OpenWebUI über Traefik exponiert).

## 12. AMD / NPU / CPU — kurz

Nur informativ, betrifft uns nicht direkt:
- AMD: MiniMax-M2.5-Optimierungen (#23611, #23620), Aiter v0.1.12.post1 (#22264), DFLASH on ROCm (#22342).
- NPU/Ascend: Qwen3-MoE-CP, GLM-4.5V/GLM-4.7-Flash NPU, MTP für Qwen3.5, GGUF-Quant für NPU.
- CPU: GPTQ/AWQ-4-bit auf CPU (#22685), `gemma4_rmsnorm_cpu` (#22842), Qwen3.5-CPU-Optimierung.

---

## TL;DR — Was bedeutet das für uns konkret?

1. **Image-Bump auf 0.5.11 lohnt sich** — primär wegen CUDA 13.0 + Torch 2.11 (potentiell der Fix für unsere ~45% sm121-Build-Regression) und nativer Gemma 4 / GLM-5.1 / Qwen3.6 Unterstützung.
2. **Spec V2 ist neuer Default** — unsere MTP-Profile (GLM-4.7, GLM-5) müssen wir nach dem Bump erneut benchmarken.
3. **`FlashInferCuteDslMoE` testen** für die Problemmodelle Qwen3.5-397B-NVFP4 und Minimax-M2.5 PP=4 — vierte MoE-Backend-Option.
4. **NVFP4 KV-Cache evaluieren** — könnte Kontext-Footprint halbieren.
5. **Prometheus-Metriken erweitern** — `kv_cache_pool_*` Gauges + Pending-Token-Count ins Grafana-Dashboard.
6. **Gemma-4-Patch-Image abkündigen** — `0.5.10-20260429-gemma4-sm121-dev1` wird durch ein 0.5.11-Vanilla-Image abgelöst.
7. **Sicherheit:** CVE-2026-5760 vor Bump prüfen.

---

## Offene Fragen

- Funktioniert unser sgl-kernel-SM121-Patch noch gegen sgl-kernel 0.4.2 / Torch 2.11 / CUDA 13.0?
- Bleibt der HAProxy-Sidecar-Workaround für `EADDRINUSE` weiter nötig (PR #20468 wurde in 0.5.10 reverted und blieb in 0.5.10rc0/0.5.10 buggy — siehe CLAUDE.md)? Release-Notes nennen keinen direkten Fix dafür → vermutlich: ja, weiter nötig. Vor Rollout verifizieren.
- Eager-Mode auf `cutlass_moe_fp4` in 0.5.11 immer noch broken (`!`-Token-Collapse)? Keine Hinweise im Changelog, dass adressiert.
