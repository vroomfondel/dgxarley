# REAP Requant: selektives Attention-NVFP4 (W4A16) für glm-5.2-reap-504B-v2 — Plan

Plan, noch nicht begonnen. Geschrieben 2026-07-16. Companion:
`../kikube/quantizer/configs/glm-5.2-reap-504b-v2-attn-selective_nvfp4.yaml`
(der Quant-Config-Entwurf mit demselben Ziel; DIESES Dokument ist der operative
Projektplan mit den Go/No-Go-Gates), `dsalogitrework.md` (p35 LIVE RESULT = die
Profiling-Motivation), `DSA_speedup.md`, Memory
`reference_glm52_dsa_indexer_deepgemm_sm121`.

## 0. Kontext für eine frische Session (alles Nötige, um HIER zu starten)

**Was läuft:** `0xSero/glm-5.2-reap-504B-v2` (GlmMoeDsaForCausalLM = MLA + DSA,
NVFP4/modelopt, 168 REAP-Experten, 78 Layer + 1 MTP) auf dem dgxarley-Cluster:
4× DGX Spark GB10 (**SM121**, consumer Blackwell, 128 GB unified, ARM64), TP4,
Image `xomoxcc/dgx-spark-sglang:0.5.15-sm121`. Profil:
`roles/k8s_dgx/model_profiles/0xsero-glm-5.2-reap-504b-v2.yml` — dort steht der
PATCH-ACTIVATION CONTRACT (attention_backend dsa + dsa_*_backend trtllm +
dsa_paged_mqa_logits_backend torch + dsa_indexer_triton). Deploy:
`ansible-playbook k8s_dgx.yml --tags sglang` (NIE ohne explizite User-Freigabe;
`kubectl --context=ht@dgxarley`, lokal). Endpoint:
`https://sglang.dgx.elasticc.io/v1`.

**Patch-Stack (Vorarbeit, ESSENTIELL):** SGLang wird zur Laufzeit über
Source-Patches in `roles/k8s_dgx/files/sglang_patches/p<NN>_*.py` gepatcht
(ConfigMap → `/patches`, Runner in `sglang_launch.sh`, Regeln in `_patchlib.py`
+ `sglang_hy3_dsa_handoff.md` — **den Handoff zuerst lesen**). Für DIESES
Projekt relevant: `p21_mixed_nvfp4_dispatch.py` +
`p22_modelopt_mixed_nvfp4_variant.py` (die partielle
MIXED_PRECISION/W4A16-Unterstützung aus der Qwen3.6-Arbeit, Memory
`reference_qwen36_nvfp4_modelopt_mixed`; Upstream-Referenzen sglang PR #27906 +
#28099). p30/p35 (Indexer) und p34 (native Sparse-Attention) sind der Grund,
warum die Attention-GEMVs jetzt der Boden sind.

**GPU-Testumgebung ohne Cluster (Standard-Methode aller Vorarbeiten):** spark5
(`ssh root@spark5.local`, NICHT im k3s) hat vollen GB10-Zugriff in podman:
`podman run --rm --device nvidia.com/gpu=all -v /root/patchtest:/patchtest
--entrypoint bash xomoxcc/dgx-spark-sglang:0.5.15-sm121 -c "python3 ..."`.
Unter `/root/patchtest/` liegen die Harness-Scripts der Vorarbeiten als
Vorlagen (`validate_p34.sh`, `sm120_sparse_mla_test.py`, `sm120_perf.py`,
`triton_indexer.py`, `bench_indexer.py`, `run_idem.sh`). Patch-Kette vor einem
Test anwenden: Repo-`sglang_patches/` nach spark5 kopieren, im Container
`for p in p[0-9][0-9]_*.py; do python3 $p; done`.

**Baselines (2026-07-16, live gemessen):** Decode single-stream **8.4 tok/s**
(cuda-graph, DSA nativ; vor MTP). GSM8K-Referenz: 2-shot, n=20, concurrency 8,
temp 0, max_tokens 768 → **85% (p34) / 90% (p35)**, 0 Fehler. Harness:
einfacher OpenAI-Client-Fewshot-Loop (`gsm8k_dsa.py`, env-gesteuert:
N/NUM_SHOTS/CONCURRENCY/MAX_TOKENS/LABEL/BASE_URL, schreibt
`gsm8k_<LABEL>_summary.json`); lag zuletzt im Session-tmp — bei Bedarf in 10 min
neu schreibbar (datasets: `openai/gsm8k`, Antwort-Extraktion per Regex auf
"answer is X").

**Wie die Profiling-Zahlen entstanden (reproduzierbar):** SGLang
`POST /start_profile {"output_dir":"/tmp/sglprof","num_steps":24,
"activities":["CPU","GPU"]}` gegen den Head (Port-Forward auf svc/sglang:8000),
Last erzeugen, Trace `*-TP-0.trace.json.gz` im Pod parsen (Kernel-Events nach
Name summieren). GB10-Eigenheit: `nvidia-smi` util taugt nicht (Memory
`reference_gb10_util_and_stuck_rank`).

**Checkpoint-Fundorte:** Cluster-HF-Cache auf JuiceFS (`/mnt/jfs`,
USB-HDD-Backend — Cold-Loads dauern Stunden, Memory
`reference_juicefs_backend_usb_hdd`); Recovery-Kit lokal:
`~/hf_downloads/GLM-5.2-504B-REAP-recovery-kit/` (RECONSTRUCT.md = Herkunft
aller 0xSero-5.2-Artefakte). Format-Ground-Truth für die modelopt-W4A16-Packung:
`nvidia/Qwen3.6-35B-A3B-NVFP4` (echter MIXED_PRECISION-Export, davon die
Tensor-/Scale-Konventionen ablesen).

**Hausregeln, die hier greifen:** kein Deploy/Pod-Delete ohne Freigabe; kein
HF-/externer Push ohne Freigabe; kein GPU-Debug-Pod auf spark1-4 während SGLang
serviert (Time-Slicing → NCCL-Timeout); Test-Pods `tail -f /dev/null`, nie
Label `app=sglang`; forward-fix, nie Image-Rollback.

## 1. Motivation (profiliert, nicht geschätzt)

Live-Profiling des Decode-Steps (2026-07-16, head, 22 Steps, 88% GPU-busy,
~131 ms Kernel-Zeit/Token) nach p34 (native Sparse-Attention) + p35
(Triton-Indexer):

| ms/Token | Anteil | Was |
|---|---|---|
| ~59 | ~45% | cuBLAS bf16 GEMV (166 µs × ~3.25/Layer) + dsv3_fused_a_gemm + lm_head |
| ~12 | ~10% | kleine bf16 wmma-GEMMs (kv_b / Indexer-Projektionen) |
| ~27 | ~21% | NVFP4-MoE (grouped cutlass) |
| ~10 | ~8%  | NCCL AllReduce (TP4) |
| ~3  | ~2.5% | Sparse-Attention (p34) |

Die **unquantisierten bf16-MLA-Projektionen sind der Boden** (~71 ms/Token,
reine Gewichts-Bandbreite bei bs=1). Die Byte-Brocken pro Layer: `o_proj`
(16384×6144 ≈ 100M Params) und `q_b_proj` (2048×16384 ≈ 34M); `q_a` (12.6M),
`kv_a` (3.5M), `kv_b` (~15M) sind klein UND quantisierungsempfindlich
(Low-Rank-Kompressionen der MLA).

**Ziel:** `o_proj` + `q_b_proj` (≈80% der Attention-Bytes) auf NVFP4
**weight-only (W4A16)** → erwartetes Decode ~78 ms/Token ≈ **~12-13 tok/s statt
8.4 (~1.5×)**, multiplikativ zu MTP.

## 2. Warum W4A16 statt W4A4, und warum direkt am publizierten v2

- **W4A16 = data-free.** Nur Gewichts-Rundung (fp4 gepackt + Block-Scales),
  KEINE Kalibrierung (die bräuchte nur die Aktivierungs-Quantisierung von
  W4A4). Der Decode-Gewinn kommt zu 100% aus der Gewichts-Bandbreite;
  Aktivierungen bf16 zu lassen vermeidet zusätzlich Qualitätsrisiko und lohnt
  bei bs=1-GEMV auch performanceseitig nicht (Quant-Overhead pro Step).
- **Die BF16-Attention im publizierten `0xSero/glm-5.2-reap-504B-v2` IST die
  KD-Recovery.** Recovery-Kit-Befund (2026-07-16, lokal unter
  `~/hf_downloads/GLM-5.2-504B-REAP-recovery-kit/`): v2 = `GLM-5.2-504B-Nvidia`
  (NVFP4-Base) + Router-KD-v2-Gates + logit-KD-LoRA (r16/α32), und der LoRA
  targetet exakt `q_a_proj, q_b_proj, kv_a_proj_with_mqa, kv_b_proj, o_proj`
  — gemerged in die BF16-Attention-Gewichte. Jeder Rebuild von einer Base
  würde den LoRA verlieren (Re-Apply-Scripts liegen im Kit, unnötige Arbeit).
  Eine öffentliche BF16-REAP-Base existiert Stand 2026-07-16 NICHT mehr
  (`...BF16-REAP-keep168-unified` = 401 privat/weg; `GLM-5.2-504B` ist heute
  selbst NVFP4, 309 GB).
- **Kein Kapazitätsproblem:** Shard-Rewrite streamt Tensor für Tensor, nie das
  ganze Modell im Speicher. Läuft auf spark5 oder der Workstation.

## 3. Zielformat

modelopt-MIXED_PRECISION-Konvention (Präzedenzfall `nvidia/Qwen3.6-35B-A3B-NVFP4`
= W4A16_NVFP4; Memory `reference_qwen36_nvfp4_modelopt_mixed`):

- Pro quantisierter Matrix: `weight` (fp4, 2 Werte/Byte, uint8-Container),
  `weight_scale` (fp8-Blockscales, Blockgröße 16), `weight_scale_2`
  (fp32 global). KEIN `input_scale` (W4A16).
- `config.json:quantization_config.ignore`: die per-Layer-Globs
  `model.layers.N.self_attn*` ersetzen durch die explizite Liste OHNE
  `o_proj`/`q_b_proj` (d.h. weiter drin: `q_a_proj`, `q_a_layernorm`,
  `kv_a_proj_with_mqa`, `kv_a_layernorm`, `kv_b_proj`, `*indexer*`).
  Layer 0-2 (dense MLP), `*shared_experts*`, `lm_head`, Layer 78 (MTP)
  UNVERÄNDERT ignoriert. `hf_quant_config.json` (falls vorhanden) analog.
- `model.safetensors.index.json` neu generieren (Tensor-Namen + Größen ändern
  sich in den betroffenen Shards).

## 4. Phasen mit Go/No-Go-Gates

### Phase 0 — W4A16-GEMV-Messung auf spark5 (DAS Entscheidungs-Gate)

Podman + GPU auf spark5 (Methode wie p34/p35-Validierung, kein Cluster):

1. Identifizieren, welchen Kernel SGLang für modelopt-W4A16-Linear auf SM121
   dispatcht (der Qwen3.6-Pfad; unsere p21/p22-Patches sind die Anlaufstelle,
   Upstream-Referenzen PR #27906 + #28099).
2. Microbench bei bs=1 (GEMV) mit den echten Per-Rank-Shapes (TP4):
   `o_proj` [6144 × 4096], `q_b` [4096 × 2048] (Eingangs-/Ausgangsdim je nach
   Shard-Achse), jeweils W4A16-NVFP4 vs. bf16-cuBLAS-Baseline (die gemessenen
   166 µs-Klasse-Calls).
3. **GO-Kriterium: ≥2.5× vs. bf16-GEMV** (theoretisch 4×; unter 2.5× frisst
   Dequant-Overhead den Gewinn und das Projekt lohnt nicht → stattdessen
   0xSero nach einem selektiven Export fragen oder es lassen).
4. Nebenbefund dokumentieren: Verhalten bei bs=8/32 (Concurrency-Regime).

### Phase 0 — ERGEBNIS 2026-07-16: GO (GB10-gemessen)

Synthetischer GEMV-Bench auf spark5 (bf16-cuBLAS vs. der servierte NVFP4-Pfad:
flashinfer.fp4_quantize Weight offline + Aktivierung dynamisch pro Call +
cutlass_scaled_fp4_mm; Timing INKL. Aktivierungs-Quant). Per-Rank-TP4-Shapes:

| Shape | bs=1 | bs=8 | bs=32 |
|---|---|---|---|
| o_proj (out 6144, in 4096) | **5.58x** | 3.15x | 3.86x |
| q_b (out 4096, in 2048) | **2.89x** | 0.91x | 1.22x |

**GO** (beide bs=1 > 2.5x). Zwei Einordnungen: (1) der Gewinn ist
SINGLE-STREAM (bs=1 GEMV, memory-bound = unser Decode-Boden); q_b faellt ab
bs=8 unter 1x (compute-bound + Quant-Overhead), o_proj bleibt auch bei Batch
schneller. (2) mean_rel_err ~0.13 ist auf SYNTHETISCHEN Zufallsgewichten mit
naivem Global-Scale, KEIN Qualitaetsbeweis -- die Genauigkeit klaert erst das
GSM8K-Gate (Phase 3) auf echten, modelopt-kalibrierten Gewichten. Bench-Skript:
`$CLAUDE_JOB_DIR/tmp/gemv_phase0.py` (auf spark5 unter /root/quantwork/).

NB: der servierte Pfad ist W4A4 (Aktivierung dynamisch fp4, keine Kalibrierung
noetig), nicht W4A16 -- bei bs=1 ist die Aktivierungs-Quant ein [1,in]-Tensor,
vernachlaessigbar, und das oben ist bereits inklusive. Das vereinfacht Phase 1:
Standard-modelopt-NVFP4 auf o_proj/q_b (dynamic input), kein Sonder-W4A16-Preset.

### Phase 1 — Shard-Rewrite-Script

`../kikube/quantizer/` (dort lebt das Quant-Tooling): Script
`surgical_attn_nvfp4.py`:

- Input: lokaler Snapshot von `0xSero/glm-5.2-reap-504B-v2` (liegt im
  Cluster-HF-Cache; für den Rewrite auf lokale Platte kopieren).
- Streaming über die Shards; für `model.layers.{3..77}.self_attn.{o_proj,q_b_proj}.weight`:
  NVFP4-Blockquant (16er-Blöcke, amax-basiert, exakt die modelopt-Packung),
  Ersetzen des bf16-Tensors durch die drei Quant-Tensoren. Alle anderen
  Tensoren byte-identisch durchreichen.
- config/ignore/index-Updates (§3), plus README-Stub mit Herkunft.
- **Verifikation im Script:** pro Matrix Dequant-Roundtrip gegen das
  bf16-Original (max relativer Fehler loggen); Stichproben-Vergleich der
  unangetassten Tensoren (Hash).
- Output-Name-Vorschlag: `glm-5.2-reap-504B-v2-attnq` (lokal; **kein
  HF-Push ohne explizite Freigabe** — Hausregel).
- Größenerwartung: ~294 GB → ~275 GB (o_proj+q_b: ~21 GB bf16 → ~5.3 GB).

### Phase 2 — Load-/Dispatch-Test auf spark5 (ohne Cluster)

Das volle Modell passt nicht auf einen Spark → gezielter Loader-Test:

- Mini-Harness im Container: `ModelOptFp4Config` mit der neuen ignore-Liste
  parsen, EINEN Decoder-Layer (z.B. Layer 10) instanziieren und dessen
  Gewichte aus den neuen Shards laden → Param-Klassen prüfen (o_proj/q_b als
  W4A16-Quant-Params, q_a/kv_a/kv_b als bf16), Forward auf Zufallsdaten
  gegen den bf16-Referenz-Layer (Toleranz: fp4-Rundungsniveau).
- Falls der Mixed-Dispatch klemmt: p21/p22 erweitern (bekannte Baustelle,
  Qwen3.6-Präzedenz; ggf. Upstream-PR #27906 vollständig cherry-picken).

### Phase 3 — Cluster-Deploy + Validierung (braucht Freigabe)

- Modell auf den Cluster bringen (JuiceFS-HDD-Cold-Load beachten: ~275 GB ≈
  Stunden; Preload einplanen wie beim Original).
- Neues Model-Profil (Clone von `0xsero-glm-5.2-reap-504b-v2.yml`, eigener
  Key, gleiche DSA/MTP-Konfiguration).
- Messlatte, in dieser Reihenfolge:
  1. Boot + Smoke (Kohärenz),
  2. **Decode-Durchsatz vs. 8.4 tok/s** (bzw. vs. der dann aktuellen
     MTP-Baseline — Erwartung ~1.5× auf den Nicht-MTP-Anteil),
  3. **GSM8K 2-shot n=20 conc 8 vs. 85-90%-Baseline** (Qualitäts-Gate;
     bei Regression >5pp → Abbruch, Revert aufs Original),
  4. Loop-/Attractor-Rate-Stichprobe (REPORT.md-Guardrail: der pruned
     Checkpoint hat 7.2% Loop-Rate; `recommended_sampling` wurde gegen
     dense Attention getunt und braucht ggf. Re-Tuning).
- Rollback ist trivial: Profil zurück aufs Original-Repo (bleibt im Cache).

## 5. Risiken / offene Punkte

| Risiko | Einschätzung |
|---|---|
| W4A16-GEMV-Kernel auf SM121 langsam/fehlend | DAS Gate (Phase 0); dann lohnt nichts weiter |
| Mixed-Dispatch-Lücken in SGLang | bekanntes Terrain (Qwen3.6, p21/p22); Aufwand begrenzt |
| Qualität: Quant-Rundung auf KD-LoRA-tragenden Gewichten | begrenzt auf o_proj/q_b; GSM8K + Loop-Gate in Phase 3 |
| modelopt-Packungs-Details (Scale-Layout, Transponierung) | in Phase 1 gegen einen echten modelopt-W4A16-Export (Qwen3.6) verifizieren, nicht raten |
| dsv3_fused_a_gemm (15 ms/Token) bleibt bf16 | bewusst: das ist der q_a/kv_a-Pfad (dense by design); Rest-Floor einkalkuliert |
| Hosting/Verteilung des neuen Checkpoints | lokal/Cluster only; HF-Push nur nach Freigabe |

## 6. Einordnung / Reihenfolge

Erst **MTP live bewerten** (läuft), dann Phase 0. MTP und dieses Projekt
multiplizieren sich (~1.8× × ~1.5× ≈ 2.5-3× single-stream Ziel). Phase 0 ist
ein halber spark5-Tag und entscheidet alles Weitere; Phasen 1-2 sind je
überschaubar; Phase 3 ist ein normaler Deploy-Zyklus.
