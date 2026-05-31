# SGLang v0.5.12 — Relevante Änderungen seit v0.5.11

Quelle: [Release Notes v0.5.12](https://github.com/sgl-project/sglang/releases/tag/v0.5.12) · Veröffentlicht 2026-05-16 · Diff: [`v0.5.11...v0.5.12`](https://github.com/sgl-project/sglang/compare/v0.5.11...v0.5.12)

Dieses Dokument hebt die Änderungen hervor, die für unseren DGX-Spark-Cluster (5 Nodes, GB10 / SM121, NVFP4-MoE, Multi-Node über RoCE/QSFP) relevant sind. Reine ROCm/Ascend/CPU/Diffusion-Themen werden nur kurz erwähnt.

Das `scitrera/dgx-spark-sglang:0.5.12`-Image ist bereits in `roles/k8s_dgx/defaults/main.yml` (Zeile 28) als `default_sglang_image` gesetzt — aber **noch nicht durch Matrix-Tests gegen unsere Profile validiert**.

---

## 1. Toolchain — wenig Bewegung, aber DeepEP-Migration auf CUDA 13

- **FlashInfer: 0.6.8.post1 → 0.6.11.post1** (mit zwischenzeitlichem Revert: #24452, #25129, #25310, #25335).
- **sgl-kernel: 0.4.2 → 0.4.2.post2** (#24457, #25326), inkl. SM90 flashmla compile-fix (#24130).
- **DeepEP-Quelle umgestellt** — von der Community-Fork `fzyzcjy/DeepEP` auf `deepseek-ai/DeepEP@hybrid-ep`, damit DeepEP unter dem **CUDA-13-Default** sauber baut und läuft (#25113). Begleitend ein Triton-Kernel-Fix für gpt-oss.
- **Torch 2.11 Docker-Prep + Dependency-Cleanup** weiter konsolidiert (#23593, schon aus 0.5.11-Linie).
- **`sgl-deep-gemm`** als eigene Wheel + Release-Workflow ausgegliedert; `DeepGEMM` aus `sgl-kernel` deprecated (#24268, #24348, #24385).
- **Unified Docker-Tag** — `lmsysorg/sglang:v0.5.12` ist jetzt **ein einziges Image für alle Nvidia-GPUs** (Blackwell/Hopper/Ampere), statt SM-spezifischer Builds.

**Relevanz für uns:**
- Der CUDA-13-Sprung war 0.5.11 — 0.5.12 ist `.post`-Politur. Die `~45 %-sm121-Build-Regression` (siehe Memory `reference_sm121_build_base_regression.md`) wird primär in `roles/k8s_dgx/files/Dockerfile.sglang.sm121` durch Base-Image- und sgl-kernel-Versionen entschieden, nicht durch dieses Release.
- **`sgl-deep-gemm`-Wheel** könnte unseren Custom-Build vereinfachen: weniger sgl-kernel-Patch-Surface, weil DeepGEMM in ein separates Wheel wandert. Vor Rebuild prüfen, ob unser sgl-kernel-SM121-Patch (in `Dockerfile.sglang.sm121`) noch gegen `sgl-kernel 0.4.2.post2` cleanly anwendbar ist.
- Das Unified-Tag ist nett, ändert für uns aber nichts — wir bauen ohnehin SM121-spezifisch.

## 2. DeepSeek V4 — Day-0 Support

Großes Release-Highlight, aber für unseren Cluster nur Hintergrund:

- Full inference path (TP/EP/CP/DP-Attention, B300/B200/H200/H100/GB200/GB300/MI35X), PD-Disagg, HiSparse-KV-Offload, Reasoning- und Tool-Call-Parser, DeepGemm- und FlashMLA-Kernels inkl. **MegaMoE**.
- Post-Day-0: HiCache unter UnifiedTree (#24691), W4A4 MegaMoE (#25052), Marlin/FlashInfer W4A8 MoE auf Hopper (#24816, #24986), TP16 auf H100/H20 (#24949), Fused SiLU+clamp+FP8 Quant (#24897), MHC+DeepGemm-Pipeline-Fusion (#24775), Multi-Detokenizer (#24944), PP+PD für DSv4 (#24700).

**Relevanz:** Das **volle** DeepSeek-V4 ist ein 671B-Klasse-Modell — auf 4×GB10 mit insg. 4×128 GiB Unified Memory passt das selbst in W4A4-Quant grenzwertig (~340 GiB Weights + KV) und setzen wir nicht ein. Die kollateralen Kernel-Verbesserungen (MHC-Pipeline, Fused SiLU+Clamp+FP8) fließen ohnehin in alle MoE-Modelle ein.

> **Update 2026-05-31 — DeepSeek-V4-*Flash* ist eine separate, kleine Variante und jetzt unser Default-Versuch.** `sgl-project/DeepSeek-V4-Flash-FP8` (256 routed Experts / 6 aktiv, hidden 4096, 43 Layer, block-wise FP8 `ue8m0`) passt auf 4×GB10 und ist als `sglang_model` in `roles/k8s_dgx/defaults/main.yml` gesetzt (Image: `0.5.12.post1-sm121`). ⚠️ **UNTESTED / first-contact** — Profil-Kommentar und Boot/Coherence noch nicht validiert. Zwei vendored Workarounds nötig: (1) `kv_lora_rank: null`-Patch in `sglang_launch.sh` (V4-Flash nutzt q-LoRA + o-LoRA + GQA statt MLA-KV-Compression; transformers-5.x `_DeepseekV4ConfigAlias` lehnt `None` sonst per Strict-Dataclass ab), (2) FP8-Checkpoint statt NVFP4, weil RedHatAIs compressed-tensors-NVFP4-Repackage am `wqkv_a`-Matcher-Gap scheitert. Details: Modellprofil `sgl-project-deepseek-v4-flash-fp8.yml` + `SGLANG_v0.5.12.post1_VERSION_CHANGES.md` (dessen DSv4-Cherry-Picks dadurch relevant werden).

## 3. Speculative Decoding V2 — Reifegrad steigt, Gemma 4 MTP nativ

- **Adaptive Spec V2 (2/N)** (#23336) — passt `speculative_num_steps` zur Laufzeit an.
- **EAGLE-3 SWA-Support** (#24664), neuere EAGLE-3-Drafter (#24663).
- **Kimi K2.5 EAGLE-3 MLA** Spec-Decoding (#24826).
- **Gemma 3/4 + EAGLE-3** Support (#23976).
- **Gemma 4 MTP** — eigener MTP-Head für Gemma 4 (#24436, #24433). Cookbook-Recipe noch ausstehend.
- Custom Speculative-Algorithm Registry (#23991).
- Overlap stale-state Fix (#23456); `trtllm` decode kernel für draft-extend (#24566).
- Fix Kimi K2.5 MLA EAGLE + DP Attention (#25033); `ngram` off-by-1 in `num_accepted_drafts_per_req_cpu` (#24965); MTP-Crash bei `bonus_tokens=None` (#25204); stuck-MTP auf DSA-Modellen (#24635).
- Diverse Naming-Refactors über `EagleDraftExtendInput` (#24859, #24094, #25014, #25038, …).

**Relevanz:**
- Für unsere **GLM-4.7 / GLM-5 MTP-Profile** sollten Spec-V2-Reife + Overlap-Fix transparent wirken. Nach Image-Bump unbedingt erneut benchen — V2 ist seit 0.5.11 Default, kleine Bugs verschwinden mit 0.5.12.
- **Gemma 4 MTP** ist neu und passt zu unserem `gemma4`-Profil. Lohnt einen Versuch: MTP-Head-Acceptance bei Gemma 4 messen.

## 4. Performance — die Highlights für unsere Pfade

- **TMA bulk-store für `set_mla_kv_buffer`** — bis zu **12× über Baseline** (#25311). Hopper/Blackwell — auf GB10 (SM121, ähnlich Blackwell-Profil) potenziell relevant für MLA-Modelle (DSv3.2, Kimi K2.5, falls wir die je fahren).
- **PDL für DSv3.2 / GLM-5 Kernels** (#23965) und `torch.mm` für DSv3.2 Indexer-GEMM (#23856) — trimmt Low-Latency-Overhead auf FP4-Pfaden.
- **Cute-DSL FP4 Dense GEMM Reland** (#23590) und Cute-DSL NVFP4 Quantization Kernels (#23745).
- **`SGLANG_OPT_FP8_WO_A_GEMM` standardmäßig an** (#25181) — Weight-Only FP8 A-GEMM-Optimierung.
- **JIT Custom All-Reduce default** (#24363, #24742) und Env-Rename: **`SGLANG_USE_JIT_ALL_REDUCE` → `SGLANG_OPT_USE_CUSTOM_ALL_REDUCE_V2`** (#24297). ⚠️ **Breaking** falls wir die alte Variable irgendwo setzen.
- **DeepseekV2MoE**: Shared-Experts werden zurückgehalten, wenn der Routed-Kernel non-mutating ist (#25279). Spart Redundanz.
- **Gemma 4 MoE**: fused Q/K/V RMSNorm + per-Expert FP8 ckpt-loader (#24696).
- **Gemma 4 VLM**: PCG + fused RMSNorm + Residual (#24048, schon aus 0.5.11).
- Breakable CUDA-Graph für `bs > 1` (#24662).
- FA3: skip `scheduler_metadata` precompute unter DP-Attention (#24632).
- Eliminate Logits-H2D-Blocking-Copy (#24627); Hidden-States-D2H-Copy nur wenn nötig (#25155).
- Kimi tokenizer TTFT-Optimierung (#25265).
- `--prefill-only-disable-kv-cache` für Pure-Prefill-Worker (#23675).
- `aten::rms_norm` / `aten::mm.dtype` Registrierung im Batch-Invariant-Mode (#24459).

**Relevanz:**
- **Env-Rename** in unsere Launch-Skripts checken: `grep -r SGLANG_USE_JIT_ALL_REDUCE roles/`. Falls gesetzt → umbenennen oder entfernen (Default ist jetzt sowieso JIT).
- `SGLANG_OPT_FP8_WO_A_GEMM`-Default sollte FP8-Pfade (Qwen3.6-35B-A3B-FP8 als unser Hermes-LiteLLM-Default) marginal beschleunigen.
- Gemma-4-MoE-Fused-Kernels sind direkt relevant für ein `gemma-4`-MoE-Profil, falls wir das anlegen.

## 5. HiCache + UnifiedRadixTree

- **HiCache-Framework für UnifiedRadixTree** (#23316) inkl. SWA-Support (#23391).
- **HiCache für DeepSeek V4 unter UnifiedTree** (#24691).
- **SSD-Offload via Mooncake-Store** (#24277).
- **HiSparse FP8 KV-Cache** via `flashmla_kv`-Backend (#23013).
- UnifiedRadixCache Device-Match-Semantik mit HiCache angeglichen (#25277).
- Fixes: Partial-Match auf evicted+backuped Nodes (#24943), Tombstone-Lock-Replay (#24972), `_cascade_evict` Leaf-Determination (#25068), SWA-Chunk-Req Deferred (#24318), SWA-Component Host-Hit (#25085).
- Storage-Prefetch Default-Timeout (#23309); Mamba/SWA Radix-Cache KV-Events (#23678, #24718).

**Relevanz:** HiCache ist auf SSD-Offload + KV-Cache-Tiering ausgelegt — primär für Long-Context-Workloads. Auf unserem Cluster könnten wir das für Hermes-Sessions mit langer Skill-History theoretisch evaluieren, aber:
- Wir haben *keinen* dedizierten KV-Cache-SSD-Tier definiert.
- Unsere Single-Tenant-Workloads passen aktuell in den GPU-KV-Pool.

→ Future Work, kein sofortiger Handlungsbedarf.

## 6. PD Disaggregation

Viel Bewegung, aber wir fahren PD weiterhin nicht disaggregiert:

- NIXL: Staging-Buffer für heterogene-TP-KV-Transfer (#22536), async transfer (#23967), XPU-Pointer-Overflow + mismatched-P/D-TP-Fixes (#24188, #24648).
- Mooncake: Incremental Transfer + SSD-Offload (#24257, #24277).
- `PrefillDelayer` mit NCCL-Allgather für DP-Sync (#24768).
- Priority-Scheduling-Fix im PD-Mode (#25062).
- Multi-Node Prefill Bootstrap-Port Broadcast (#24378); Retry-with-Backoff für Bootstrap-Registration (#25125).
- SWA Memory-Prealloc für Disagg-Decode (#24857).
- IntraNode NVLink Docs (#23329).

**Relevanz:** Wenn wir je auf PD-Disagg gehen (z. B. dedizierter Prefill-Worker auf spark1, Decode auf spark2-4), ist 0.5.12 das erste Release mit gehärtetem Multi-Node-Bootstrap. Aktuell aber nicht geplant.

## 7. Neue Modelle (Day-0)

Mit Cookbook-Recipe:

| Modell                  | PRs                                            | Cookbook                                                                                                              |
|-------------------------|------------------------------------------------|-----------------------------------------------------------------------------------------------------------------------|
| **DeepSeek V4**         | #23882                                         | [DeepSeek-V4](https://docs.sglang.io/cookbook/autoregressive/DeepSeek/DeepSeek-V4) · [LMSYS-Blog](https://www.lmsys.org/blog/2026-04-25-deepseek-v4/) |
| **Intern-S2-Preview**   | #24875, #25115, #25134                         | [Intern-S2-Preview](https://docs.sglang.io/cookbook/autoregressive/InternLM/Intern-S2-Preview)                        |
| **MiniCPM-V 4.6**       | #24855, #24876, #24991, #24998                 | [MiniCPM-V-4.6](https://docs.sglang.io/cookbook/autoregressive/OpenBMB/MiniCPM-V-4_6)                                 |
| **Laguna-XS.2** (Poolside) | #24204, #24730                              | [Laguna-XS.2](https://docs.sglang.io/cookbook/autoregressive/Poolside/Laguna-XS.2)                                    |
| **Ring-2.6-1T**         | #25360, #25370                                 | [Ring-2.6-1T](https://docs.sglang.io/cookbook/autoregressive/InclusionAI/Ring-2.6-1T)                                 |
| **Gemma 4 MTP**         | #24436, #24433                                 | (Recipe ausstehend)                                                                                                   |

Ohne Cookbook-Recipe (oder nicht relevant): Trinity-mini (Ascend), HunyuanVideo / Qwen-Image ModelOpt FP8 (Diffusion).

**Relevanz für unseren Cluster:**
- **Gemma 4 MTP** ist die einzige direkt relevante Neuerung — bestehender `gemma4`-Profil-Patch könnte um MTP erweitert werden (sobald die Recipe da ist).
- **Ring-2.6-1T** (1-Trillion-Param Reasoning) und das **volle DeepSeek V4** (671B-Klasse) passen kapazitiv nicht auf 4×GB10. Die kleine **DeepSeek-V4-*Flash*-FP8**-Variante hingegen schon — sie ist seit 2026-05-31 unser Default-Versuch (siehe §2-Update).
- **MiniCPM-V 4.6** / **Intern-S2-Preview** sind VLM/Embedding-Kandidaten — vermutlich Vision/Multimodal, nicht unser aktueller Use-Case.
- **Laguna-XS.2** (Poolside) — Coding-Modell, eventuell interessant für Hermes-Tooling.

## 8. Quantisierung & Kernels

- **NVFP4 hot-reload-safe Weight-Loading** — Alias-when-same-shape (#25190); frees unused source-scales nach Weight-Processing (#25107).
- **Cute-DSL NVFP4 Quantization Kernels** (#23745).
- **Cute-DSL FP4 Dense GEMM Reland** (#23590).
- DSv4: W4A4 MegaMoE (#25052); W4(MXFP4)A16 auf Hopper (#24986); FlashInfer SM90 Cutlass MXFP4 MoE-Backend (#24816).
- KV-Compression V2 + Fused SiLU+Clamp+FP8 Quant aus DSv4-Dev-Branch portiert (#24890, #24897).
- BF16 EP-MoE für DeepGEMM (#17392).
- DeepGEMM aus `sgl-kernel` deprecated → eigenes `sgl-deep-gemm` Wheel (#24268, #24348, #24385).
- TRT-LLM A2A Dispatch: NaN-Sanitization in Padding-Slots (#24850).
- TRT-LLM BF16 MoE für MTP (#24260).
- MegaMoE von DeepEP-Backend entkoppelt — **anschließend reverted** (#24884, #25317).

**Relevanz für SM121 / NVFP4-MoE:**
- Das `cutlass_moe_fp4`-Problem (siehe CLAUDE.md, NVFP4-Abschnitt + `SGLANG_TP_EP_MOE_UPSTREAM_BUG.md`) wird im Changelog **nicht direkt adressiert**. Keine PR-Hinweise auf SM121-spezifische Crashes/Shared-Memory-Workarounds.
- Die **NVFP4-Hot-Reload-Weight-Loading-Verbesserung** (#25190, #25107) reduziert RAM-Druck beim Modell-Reload — könnte unsere Multi-Modell-Switching-Tests entspannen.
- **Cute-DSL NVFP4 Quantization Kernels + FP4 Dense GEMM Reland** sind potenzielle Performance-Tickets — vor allem die Reland (#23590) deutet auf eine Stabilisierung des Pfads hin, der in 0.5.11 wegen Bugs raus war. Ob `FlashInferCuteDslMoE` damit auf SM121 funktioniert, ist **offen** → Matrix-Test wert.
- **`sgl-deep-gemm`-Wheel-Split** ist eine Build-System-Vereinfachung für unser Custom-Image.

## 9. Frontend / API

- **`/v1/tokenize` Chat-Completion-Style Support** (#23981).
- **Multi-Detokenizer-Support** (#24944) — relevant für Multi-Tenant.
- **Strukturelle Tags für strict Tool-Calling & Reasoning** über mehrere Modelle (#21722).
- **Auto-Detect Reasoning / Tool-Call-Parser** aus Chat-Template (#23952).
- **Two-Phase Reasoning Grammar** + `--enable-strict-thinking` (#23953).
- **OpenAI `reasoning.enabled` → `thinking` + `enable_thinking`** Mapping (#23951).
- Kimi-K2.5: bare-numeric Tool-Call-IDs (#23950).
- Azure Blob Storage Connector (`az://`, `*.blob.core.windows.net`) (#23995).
- Adaptive Queue-based Prefill-Delayer-Trigger (#23189).
- **Reject `repetition_penalty=0`** in `SamplingParams.verify()` (#24874). ⚠️ Falls Hermes/OpenWebUI das je sendet — Error statt Silent-Fail.
- `--random-input-len` für `send_one.py` (#24464).
- Neue Env-Vars:
  - **`SGLANG_MAX_KV_CHUNK_CAPACITY`** (#25120).
  - **`SGLANG_RADIX_FORCE_MISS`** (#24726, #24950) — Radix-Cache-Bypass zum Debuggen.
  - **`SGLANG_TRACE_LEVEL`** für Startup-Trace-Level (#24716).

**Relevanz:**
- **Auto-Detect Reasoning-Parser + `--enable-strict-thinking`** ist für Hermes/OpenWebUI mit reasoning-fähigen Modellen (Qwen3.6 Thinking-Mode, GLM-5) potenziell interessant — derzeit konfigurieren wir Reasoning-Parser per Modellprofil (`reasoning_parser`), Auto-Detect könnte die Profile vereinfachen.
- **Multi-Detokenizer** — Hermes ist Multi-Tenant, aber wir fahren *einen* SGLang-Server pro Modell, nicht Multi-Tenant innerhalb eines Servers → kein direkter Mehrwert.

## 10. Observability

- **`sglang:get_loads_duration_seconds`** Prometheus-Metric (#25163).
- **Per-Iteration Forward-Pass-Metrics via ZMQ-PUB** (#22789) — Low-Overhead-Telemetrie pro Iteration.
- **`fwd_occupancy` Metric** in `SchedulerStats` + Prometheus-Collector (#24458).
- **SWA / Mamba Cache-Metrics** (#24396).
- Mamba/SWA Radix-Cache KV-Events (#23678, #24718).
- PD KV-Transfer-Metrics-Fix (#24416).
- Decode-Side Bootstrap/Alloc-Metrics + Non-Int Token-ID Filter (#24684).

**Relevanz:** Für unsere `promstack`-Integration (Grafana-Dashboard SGLang) sind `fwd_occupancy` und `get_loads_duration_seconds` zwei direkte Add-ins. **Action-Item:** Im SGLang-Grafana-Dashboard ergänzen, sobald Image-Bump validiert ist.

## 11. LoRA

Erneut Bewegung, aber wir nutzen kein LoRA — daher nur kurz:

- MLA-Attention LoRA (`q_b_proj` / `kv_b_proj`) (#25001).
- CSGMV-Backend mit Virtual Experts für MoE-LoRA (#24007).
- MoE-LoRA: CPU-GPU-Sync-Barrieren entfernt (#24246, #24262).
- LoRADrainer für hohes P99 TTFT (#17913).
- Deterministic `lora_id` für Multi-Node `--lora-paths` (#24555).

## 12. Sicherheit

**Keine Security-PRs in diesem Release-Window** (Release-Notes-O-Ton). Im Gegensatz zu 0.5.11 (CVE-2026-5760) hier keine speziellen CVE-Fixes — Standard-Dependency-Hygiene über FlashInfer/sgl-kernel-Bumps.

## 13. ROCm / NPU / CPU / MLX — kurz

Nur informativ:

- **AMD/ROCm**: DSv4 Flash / Pro Nightlies auf MI35x ROCm 7.2 (#24203, #24825, #25039); NSA-Indexer-Fallbacks (#24125, #23562, #25205); FP8 Blockwise Quant Combine für MoRI EP (#24879); aiter `fused_qk_rmsnorm` API-Shim (#24799).
- **NPU/Ascend**: `zbal`-Support (#24575); Trinity-mini (#18172); Shared-Expert Dual-Stream (#23827); MLA KV-Transfer in PP (#23893); GLM-5 DeepEP-Docs (#23708).
- **Apple Silicon / MLX**: On-the-fly `--quantization mlx_q4` / `mlx_q8` (#24907); Auto-Detect MLX-Format (#25191); Metal-Kernel-Support in `sgl-kernel` (#23449). → **Interessant für lokale Dev-Boxen**, nicht für den Cluster.
- **CPU/Intel**: w8a8 int8 ARM-CPU (#16045); Phase-1A ARM64 CI-Bootstrap (#22123); XPU PP auf Intel (#23472).
- **MUSA**: FlashInfer-Sampling-Backend (#24978); optimierte Kernels für Piecewise-CUDA-Graph (#23633).

## 14. Diffusion (sglang-diffusion) — nicht relevant für uns

Wir fahren keine Bild-/Video-Diffusion auf dem Cluster. Wer trotzdem reinschauen will: CFG-Parallelism für LTX-2 (#23736), Dynamic Batching (#18764), Performance-Mode Server-Args (#24491), Channels-Last 3D VAE Convs Default (#23200, #24315).

---

## TL;DR — Was bedeutet das für uns konkret?

1. **`scitrera/dgx-spark-sglang:0.5.12` ist bereits Default in `defaults/main.yml`** — aber **noch nicht** durch Matrix-Tests gegen unsere NVFP4-MoE-Profile (Qwen3-235B, GLM-4.7, GLM-5, Nemotron-3, Qwen3.5-397B) validiert. **Vor produktivem Rollout testen**, vor allem den `cutlass_moe_fp4`-Pfad auf SM121.
2. **Env-Var-Rename**: `SGLANG_USE_JIT_ALL_REDUCE` → `SGLANG_OPT_USE_CUSTOM_ALL_REDUCE_V2`. `grep -r SGLANG_USE_JIT roles/` durchführen.
3. **`SGLANG_OPT_FP8_WO_A_GEMM` jetzt Default-on** — Qwen3.6-35B-A3B-FP8 (Hermes-LiteLLM-Default) könnte minimal schneller werden.
4. **Cute-DSL FP4 GEMM Reland (#23590) + NVFP4 Cute-DSL Quant Kernels (#23745)** — `FlashInferCuteDslMoE` (aus 0.5.11) auf SM121 erneut testen. Crash-Modelle Qwen3.5-397B-NVFP4 und Minimax-M2.5 PP=4 sind Kandidaten.
5. **Gemma 4 MTP** — `gemma4`-Profil könnte um MTP-Head erweitert werden (Recipe noch ausstehend).
6. **Spec-V2-Polish** (Adaptive Spec-V2, Overlap-Stale-State-Fix, K2.5 MLA EAGLE Fix) — GLM-4.7/GLM-5 MTP-Profile nach Bump neu benchmarken.
7. **Prometheus**: `fwd_occupancy` + `get_loads_duration_seconds` ins SGLang-Grafana-Dashboard ergänzen.
8. **DeepEP** auf `deepseek-ai/DeepEP@hybrid-ep` umgestellt — relevant nur falls wir das Dockerfile selbst patchen; bei `scitrera`-Base ist das transparent.
9. **`sgl-deep-gemm`-Wheel-Split** — Custom-SM121-Image rebuilden und prüfen, ob `sgl-kernel-SM121-Patch` weiter cleanly anwendbar ist.
10. **Keine Security-PRs** in diesem Window — Standard-Patch-Hygiene reicht.

---

## Offene Fragen / Risiken

- **`cutlass_moe_fp4`-Crash auf SM121** wird im Changelog nicht adressiert. Vermutlich weiter mit den modellspezifischen Workarounds in den Profilen (Triton für Minimax-M2.5 PP=4; direktes `cutlass` für Qwen3.5-397B-NVFP4; `flashinfer_cutlass` als Default für die übrigen NVFP4-MoEs).
- **HAProxy-Sidecar `EADDRINUSE`-Workaround** weiter nötig? Release-Notes erwähnen keinen Scheduler-Bind-Fix. Annahme: ja, weiter nötig. Vor Rollout `kubectl exec ... -- ss -tlnp` gegen Head-Pod prüfen.
- **Eager-Mode auf `cutlass_moe_fp4`** (broken: `!`-Token-Collapse) — kein expliziter Fix-Hinweis im Changelog.
- **Drafter-Image-Kompatibilität** prüfen: Falls wir Gemma 4 MTP einsetzen wollen, muss der Drafter-Pfad mit unserem `model-download`-InitContainer (Auto-Anhang `speculative_draft_model_path`, siehe Memory `reference_drafter_autopreload.md`) funktionieren.
- **`SGLANG_OPT_FP8_WO_A_GEMM`-Default-on**: bricht das auf SM121 etwas? Bei FP8-Modellen (Qwen3.6 FP8) gegenmessen, ggf. via `env`-Override im Modellprofil deaktivieren.
