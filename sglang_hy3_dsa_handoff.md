# Handoff: Hy3 + DSA-Backend nach dem Patch-Refactor (Stand 2026-07-16)

Kontext-Briefing für einen frischen Assistenten, der am **Hy3**- oder **DSA-Backend**-Thema
weiterarbeitet. Branch: `refactor-sglang-patches` (Commits `6b77d64`, `a760767`, `12bb560`).
Es geht hier um `dgxarley`: 5-Node-K3s-Cluster, 4× DGX Spark (GB10, **SM121**, ARM64), SGLang
verteilt via TP4. Image: `xomoxcc/dgx-spark-sglang:0.5.15-sm121`.

## 1. Das Wichtigste zuerst: die Patches sind umgezogen

Bis heute lagen ~36 Runtime-Source-Patches gegen `dist-packages/sglang/...` als inline
`python3 - <<'PATCH_*_EOF'`-Heredocs in `roles/k8s_dgx/files/sglang_launch.sh` (3899 Zeilen).
**Das ist vorbei.** `launch.sh` hat jetzt **670 Zeilen und keinen einzigen Source-Patch mehr**.

```
roles/k8s_dgx/files/sglang_patches/
  _patchlib.py                 # gemeinsame Helfer (LIES DAS ZUERST)
  p13_cuda_mem_fallback.py     # ... 37 Patches, je eine Datei
  p30_dsa_torch_backend.py     # (36 aus dem Refactor + p34, neu 2026-07-16)
  ...
```

* Ausgeliefert als ConfigMap `<prefix>-patch-scripts`, gemountet auf **`/patches`** (198 KB von
  1 MiB Limit). Definiert in `roles/k8s_dgx/tasks/sglang_instance.yml`.
* `launch.sh` iteriert am Ende der Patch-Phase über `$SGLANG_PATCH_DIR/p[0-9][0-9]_*.py`
  **in Dateinamen-Reihenfolge** und ruft jede mit `python3` auf. Der Runner ist dumm: keine
  Registry, keine Bedingungen.
* In `launch.sh` blieb nur Launcher-Logik: apt/pip-Bootstrap, der
  `SGLANG_HUNYUAN_TOKEN_SUFFIX`-Export, die zwei `.pth`-Installationen, Flag-Bau, `exec`.

**Konsequenz für dich:** wenn du am DSA- oder Hy3-Verhalten schraubst, editierst du **nicht mehr
`launch.sh`**, sondern die betreffende `pNN_*.py`. Der alte Stand steht in `git show ae178d5:...`.

### Regeln, die beim Ändern eines Patches gelten (teuer gelernt)

1. **Gates leben im Patch**, nicht im Bash. `when=gate_model("Hy3", "Hunyuan") and
   gate_env("SGLANG_SPECULATIVE_ENABLED", "true")`. Kein `if` mehr drumrum.
2. **Die Already-applied-Probe wird VOR dem Anker geprüft.** `new` enthält meist `old` als Präfix.
3. **`replace()` vs `replace_all()`**: `replace()` ersetzt nur das ERSTE Vorkommen. Wenn das
   Original `s.replace(old, new)` ohne `count` machte (oder `sed`, das pro *Zeile* ersetzt), musst
   du `replace_all()` nehmen. In Phase 2 kostete das zwei halb gepatchte Dateien, bei denen das Log
   trotzdem "Patched" meldete. **Immer die Trefferzahl im echten Image zählen.**
4. **Nie raisen.** Anker-Drift ist eine Warnung, kein Crash: `launch.sh` läuft unter `set -e`, eine
   Exception würde den Pod crashloopen. `_patchlib` fängt das ab.
5. **Alles-oder-nichts pro Datei.** Edits werden gepuffert und einmal am Ende geschrieben.
6. **Dateiname muss gültiger Modulname sein** (`p30_...`, nicht `30_...`): mypy läuft strict über
   das Repo. `p23b_...` matcht den Runner-Glob NICHT (nach zwei Ziffern muss `_` folgen).
7. **Die Checksum-Annotation** `checksum/launch-script` enthält `_sglang_patches_blob`. Fass das
   nicht an: ohne sie rollt eine reine Patch-Änderung die Pods nicht, und Patches laufen nur beim
   Container-Start → die Änderung wäre stillschweigend wirkungslos.

## 2. Die DSA-Kette (das Herzstück, 4 Dateien, Reihenfolge ist kritisch)

Alle **ungegatet** (`when=True`) — sie sind inert, weil die gepatchten Dispatch-Zweige nur bei
`attention_backend=dsa` erreicht werden.

| Datei | Ziel(e) | Was |
|---|---|---|
| `p30_dsa_torch_backend.py` (567 Z.) | `dsa/paged_mqa_logits_backend.py`, `server_args.py`, **legt `dsa/torch_paged_mqa_logits.py` neu an**, `dsa_backend.py`, `dsa/dsa_indexer.py` | torch-Fallback für den Indexer (DeepGEMM `get_paged_mqa_logits` asserted auf SM121 hart) |
| `p31_dsa_flashinfer_gather.py` (300 Z.) | `server_args.py` + `dsa_backend.py` | `dsa_decode_backend=flashinfer_gather`: top-2048 KV gathern + dense fa2 drüber (FALLBACK, s.u.) |
| `p32_dsa_flashinfer_gather_prefill.py` (116 Z.) | `dsa_backend.py` | dieselbe Impl für prefill/extend (**design-kaputt**, nur Historie) |
| `p33_dsa_fig_graph_split.py` (329 Z.) | `dsa_backend.py` | cuda-graph plan/run-split für p31 (`plan()` ist nicht graph-recordbar) |
| `p34_dsa_trtllm_sparse_sm120.py` | `model_runner_kv_cache_mixin.py` + `dsa_backend.py` + `forward_mla.py` | **der AKTIVE Pfad**: routet `dsa_*_backend=trtllm` auf flashinfers NATIVE SM120/121-Sparse-MLA (Decode UND Prefill; `backend="auto"` + 656-packed-Pool + `kv_scale_format="arbitrary_fp32"` + Rope bleibt upstream/bf16-Query) |
| `p35_dsa_indexer_triton_logits.py` | **legt `dsa/triton_paged_mqa_logits.py` neu an** + editiert p30s `torch_paged_mqa_logits.py` | Triton-fused Indexer-Logits (61×; Early-Exit auf echter Seq unter statischem Graph-Grid). Kill-Switch: `SGLANG_DSA_INDEXER_TRITON=0` |

**Sechs Patches, Reihenfolge p30 → p31 → p32 → p33 → p34 → p35.** Die `pNN`-Nummern
kodieren das. Nicht umbenennen. p34s Edits liegen in Regionen, die p30-p33 nicht
anfassen; **p35 editiert die von p30 ERZEUGTE Datei** (muss also nach p30 laufen).

### Die Falle, in die ich getappt bin (und du auch wirst)

`p33` **schreibt genau den Text um, den `p31` injiziert.** Deshalb ist `p31`s injizierter Text
**keine haltbare Already-applied-Probe**: nach `p33` findet `p31` beim nächsten Lauf seine eigene
Probe nicht mehr, der Anker passt aber noch → **erneutes Anwenden → zerstörte Datei**. Und der
Runner läuft bei **jedem Pod-Restart** erneut.

Deshalb hat `p31.apply_b()` einen **Gruppen-Guard vorab**:

```python
if MARKER_B_INIT in p.code:
    return                     # alle drei B-Edits gemeinsam schon drin
```

Genau so machte es das Original. Wenn du an `p31`/`p33` etwas änderst: **Idempotenz testen**
(Runner zweimal, siehe §5), der Tree-Diff sieht diese Klasse Fehler prinzipiell nicht, weil er nur
einen Lauf vergleicht.

Marker der Kette (stabil, nicht ändern):
`_sgl_dsa_flashinfer_gather_choice_`, `_sgl_dsa_flashinfer_gather_init_`,
`_sgl_dsa_flashinfer_gather_prefill_`, `_sgl_dsa_fig_graph_split_`,
`_sgl_dsa_trtllm_sparse_sm120_`, `fp8_paged_mqa_logits_torch_dsa`.

### DSA-Sachstand (das inhaltliche Problem, nicht das Refactor-Problem)

Aktives Profil: `roles/k8s_dgx/model_profiles/0xsero-glm-5.2-reap-504b-v2.yml`
(`sglang_model: 0xSero/glm-5.2-reap-504B-v2`), `attention_backend: dsa`,
`dsa_paged_mqa_logits_backend: torch`, `dsa_decode_backend: trtllm`,
`dsa_prefill_backend: trtllm` (seit p34; vorher flashinfer_gather/flashinfer_gather).

* **Decode UND Prefill: gelöst via p34, LIVE-BEWIESEN 2026-07-16** (Boot sauber,
  Decode 8.4 tok/s cuda-graph, Prefill 873 tok/s input auf der gather-Killer-Shape,
  GSM8K 2-shot n=20 conc 8 = 85%, 0 Fehler, 0 Restarts). Achtung Chronologie: der
  ERSTE p34-Deploy crashte am Graph-Capture ("expects BF16 query, got float8_e4m3fn"),
  Fix = p34 Edit 3 (Rope bleibt auf SM12x upstream, kein fp8-Query-Quantize).
  flashinfer 0.6.14 im Image shippt native SM120/121-Sparse-MLA-Kernel (GLM_NSA-Typ,
  Decode ≤64 Tokens warp-spec, darüber Prefill-Orchestrator, vorgebaut). Der alte
  "trtllm-Wall" war NUR sglangs hartkodiertes `backend="trtllm-gen"` in
  `_forward_trtllm`. spark5-GPU-Verifikation gegen torch-Referenz: Decode bs4
  0.072 ms, Prefill 2400 Tokens 14.4 ms/Layer, cuda-graph captured direkt.
* **Historie (Chronologie in `dsalogitrework.md` PART 2-4):** Gather-Decode war live
  bewiesen (8.4 tok/s, `ae178d5`), der Gather-PREFILL war ein Designfehler
  (~4.7 MB/Query-Token → GSM8K conc-8 killte worker-2/-3; bs=1-Smoke bestand,
  darum überlebte es bis live). p31-p33 bleiben als Decode-Fallback, p32 gilt
  weiter als unsicher.
* Der torch-Indexer (p30, seit p35 Triton-beschleunigt) bleibt zwingend.
* **MTP/NEXTN: LIVE-VALIDIERT 2026-07-16** (`speculative_enabled: true`, NEXTN,
  3 Steps / 4 Draft-Tokens): accept len ~2.1, Decode 11.7-12.4 tok/s (+45%),
  GSM8K 85% bei halber Wall-Time (179.5 s), 0 Fehler/Restarts. Kette: p42
  (NVFP4-NextN-Load) + p34 (Verify durch den nativen Sparse-Kernel) + p30
  Phase 2 (Indexer next_n>=2, reines Pass-Through - Upstream expandiert die
  page_table selbst; der erste Fix-Versuch mit eigenem repeat_interleave
  DOPPELT-expandierte und crashte den Warmup). `dsalogitrework.md` MTP LIVE RESULT.
* Volltext: `dsalogitrework.md` (PART 4 zuerst), `dsa_cuda_graph_plan.md` (§8),
  `DSA_speedup.md` (FINAL-Box oben). Das Profil trägt den aktualisierten
  "PATCH-ACTIVATION CONTRACT" ab Zeile ~73 — lies den, bevor du Keys änderst.

## 3. Das Hy3-Set

| Datei | Ziel | Gate | Status |
|---|---|---|---|
| `p64_hunyuan_shared_experts.py` | `models/hunyuan_v3.py` | Hy3/Hunyuan **oder** `TOOL_CALL_PARSER==hunyuan` **oder** `REASONING_PARSER==hunyuan` | **LEBT**, patcht heute |
| `p62_hunyuan_tool_parser.py` | `function_call/hunyuan_detector.py` | s.o. | **TOTER NO-OP** auf diesem Image |
| `p63_hunyuan_reasoning_parser.py` | `parser/reasoning_parser.py` | s.o. | **TOTER NO-OP** auf diesem Image |
| `p40_hy3_nextn_bf16.py` | `models/hunyuan_v3_nextn.py` | Hy3/Hunyuan **und** `SPECULATIVE_ENABLED==true` | feuert nur mit MTP |
| `p41_hy3_nextn_finalnorm.py` | `models/hunyuan_v3_nextn.py` | s.o. | s.o., läuft NACH p40 (gleiche Datei!) |
| `p42_dsnextn_mixed_mtp.py` | `models/deepseek_nextn.py` | `SPECULATIVE_ENABLED==true` | betrifft GLM/DeepSeek-MTP, nicht Hy3 |

**`p62`/`p63` sind auf `0.5.15-sm121` dauerhaft wirkungslos**: das Image enthält PR #29920 bereits
(`resolve_hunyuan_tokens` ist nativ drin), der Guard greift, sie tun nichts. Ihr eigener
RE-SYNC-Hinweis sagt für genau diesen Fall "DELETE this block". Sie stehen noch da, die Entscheidung
liegt beim Owner. **Ihre Konversion ist folglich NICHT verifiziert** (beide Seiten tun nichts, der
Tree-Diff kann nichts beweisen). Wenn du sie brauchst, prüf sie von Hand.

`p64` ist der inhaltlich wichtige: HYV3-Checkpoints nennen den Shared Expert
`mlp.shared_experts.*`, SGLangs Modul heißt `shared_mlp`. Ohne Remap werden die (echten,
FP4-quantisierten) Gewichte still verworfen → `shared_mlp` bleibt zero-init → `down_proj`
FP4-quantisiert eine Null-Eingabe → **NaN ab Layer 1**. Hintergrund: `QUANT_HY3_GOTCHAS.md`.

**Hy3-Profile heute:** `vroomfondel-hy3-nvfp4-w4a4.yml` und `kodelow-hy3-nvfp4-w4a16.yml`, beide
`attention_backend: triton`, **`speculative_enabled: false`** → `p40`/`p41` feuern in Produktion
aktuell **nicht**. Der W4A4-NaN-Komplex ist laut Memory noch offen und NICHT als "hybrid bestätigt"
zu behandeln.

## 4. Berührte Dateien insgesamt (Referenzmengen der Harness)

Die Patch-Phase fasst **nicht nur `sglang/`** an, sondern auch `flashinfer` (3 Dateien: `jit/cpp_ext.py`,
`quantization/fp4_quantization.py`, die gebündelte CuTeDSL-`mma.py`) und `transformers` (1:
`models/deepseek_v3/configuration_deepseek_v3.py`). Wer nur `sglang/` snapshotet, übersieht ein
Viertel der Wirkung.

| Profil | Env | berührte Dateien (Refactor-Snapshot) |
|---|---|---|
| `neutral` | `SGLANG_MODEL=neutral/none` | 26 (alle ungegateten, inkl. **DSA**) |
| `hy3` | Hy3 + `SPECULATIVE_ENABLED=true` + hunyuan-Parser | 29 |
| `glm5` | GLM-5-Modellname | 27 |
| `spec` | `SPECULATIVE_ENABLED=true` | 27 |

Snapshot vom Refactor-Abschluss (36 Patches). Seit `p34` (ungegatet) kommt in JEDEM
Profil `model_runner_kv_cache_mixin.py` dazu (+1; `dsa_backend.py` war schon drin).

## 5. Verifikation: die Harness (spark5, podman, KEINE GPU, KEIN k3s nötig)

Läuft auf `root@spark5.local` (nicht im Cluster, deshalb ohne k3s-Layer). Das Image ist
unveränderlich, jede Phase läuft in einem **frischen** Container → kein Restore nötig. Die berührten
Dateien werden über ein md5-Manifest über **alle** `.py` in dist-packages ermittelt.

```bash
# /root/patchtest/ enthält: old_launch.sh, new_launch.sh, patches/, run_phase.sh,
#                           compare.sh, run_idem.sh, run_idem_old.sh
cd /root/patchtest
for PR in neutral hy3 glm5 spec; do for P in old new; do
  podman run --rm -v /root/patchtest:/patchtest --entrypoint bash \
    xomoxcc/dgx-spark-sglang:0.5.15-sm121 /patchtest/run_phase.sh $P $PR
done; done
bash compare.sh neutral hy3 glm5 spec       # -> "ALL PROFILES IDENTICAL"

# Idempotenz (Runner ZWEIMAL = jeder Pod-Restart). Fängt Fehler, die der Tree-Diff NICHT sieht:
podman run --rm -v /root/patchtest:/patchtest --entrypoint bash \
  xomoxcc/dgx-spark-sglang:0.5.15-sm121 /patchtest/run_idem.sh hy3
# run_idem_old.sh macht dasselbe mit dem PRE-REFACTOR-Script -> unterscheidet
# "meine Regression" von "Altlast". Nutz das, bevor du einen Bug dir selbst zuschreibst.
```

`old_launch.sh` = `git show ae178d5:roles/k8s_dgx/files/sglang_launch.sh | sed -n '1,3513p'`
(Schnitt vor der `.pth`-Sektion). `new_launch.sh` läuft mit `SGLANG_PATCH_ONLY=1`, was nach der
Patch-Phase aussteigt statt den Server zu starten. Beide Schnittpunkte sind identisch.

Ergebnis heute: alle 4 Profile `TREE-DIFF: IDENTICAL`, 0 Drift beidseitig, Idempotenz sauber.

**Seit p34 gilt:** der old-vs-new-Tree-Diff (`run_phase` + `compare.sh`) beweist NUR die
Refactor-Äquivalenz der 36 migrierten Patches und ist für NEUE Patches (p34+) per
Definition rot — ein neuer Patch IST eine gewollte Divergenz vom Alt-Stand. Für neue
Patches gilt stattdessen: frischer Container, kompletter Runner-Lauf (0 ANCHOR-DRIFT),
`py_compile` + Import der Ziele, Runner ZWEIMAL (Idempotenz), plus gezielte
Unit-/GPU-Tests (p34: `/root/patchtest/validate_p34.sh`, `sm120_sparse_mla_test.py`,
`sm120_perf.py`; podman auf spark5 hat via `--device nvidia.com/gpu=all` vollen
GB10-Zugriff — GPU-Verifikation braucht KEINEN Cluster).

## 6. Was NICHT verifiziert ist (nicht als grün behandeln)

* **Kein Deploy.** Nichts davon ist am Cluster gelaufen. Die ConfigMap-Auslieferung (Keys, Mount,
  Checksum-Rollout) ist nur render-getestet. Ein echter Head-Rollout steht aus.
* **`p62`/`p63`** — s. §3, tote No-Ops, Konversion unbeweisbar.
* ~~p34 end-to-end im Cluster~~ — **erledigt 2026-07-16**: Boot + Graph-Capture +
  Smoke + GSM8K conc 8 (85%, 0 Fehler, 0 Restarts) liefen live durch. Der alte
  Gather-PREFILL (p32) bleibt design-kaputt; er ist nur nicht mehr der aktive Pfad.
* **Decode-Perf-Boden: KORRIGIERT durch Live-Profiling 2026-07-16.** p35 (Triton-
  Indexer-Logits, bit-exakt) ist deployed und nachweislich aktiv (Logits-Kernel aus
  dem Decode-Profil verschwunden, Capture-Speicher halbiert), Decode blieb aber bei
  ~8.4 tok/s: die "Indexer ist der Boden"-These stimmte am Live-Shape NICHT
  (context_length=16384, nicht die 131k-Breite der Bench-Annahme). Profilierter
  echter Boden pro Token: ~59 ms **unquantisierte bf16-MLA-Projektions-GEMVs**
  (Checkpoint-inhärent), ~27 ms NVFP4-MoE, ~10 ms NCCL, ~3 ms Sparse-Attention.
  Hebel für kurze Kontexte: **MTP** (durch p34 freigeschaltet) + Batching, NICHT
  weitere DSA-Kernel-Arbeit. p35 zahlt bei langen Kontexten (Logits skaliert mit
  Table-Breite). Kill-Switch `dsa_indexer_triton: false` im Profil (verdrahtet).
  Details: `dsalogitrework.md` "p35 LIVE RESULT".
* Der Tree-Diff beweist **Verhaltensgleichheit zum Vorzustand**, nicht Korrektheit. Wenn ein Patch
  vorher falsch war, ist er es nachher identisch falsch.

## 7. Wichtige Repo-Regeln (aus CLAUDE.md, gelten weiter)

* `kubectl --context=ht@dgxarley ...` **lokal**, nicht per SSH auf den Master.
* **Nie deployen/Pods löschen ohne ausdrückliche Freigabe.** Nie das Image zurückrollen
  (forward-fix). Nie den CPU-Shard-Loader anfassen.
* GPU-Time-Slicing ist aktiv (4 Replicas) — keine Warnungen über GPU-Contention.
* Für Debug/Inspektion einen eigenen Debug-Pod (`tail -f /dev/null`), **niemals** mit Label
  `app=sglang`, und **keinen GPU-Debug-Pod** auf einem Spark, während SGLang serviert
  (Time-Slicing → NCCL-Timeout → TP-Gruppe kaputt).
* Für historische Pod-Logs Loki (`loki.loki.svc:3100`), nicht `kubectl --previous`.

## 8. Doc-Pointer

`dsalogitrework.md` (DSA-Logit-Umbau, PART 3 = Prefill-Analyse) · `dsa_cuda_graph_plan.md` ·
`DSA_speedup.md` · `QUANT_HY3_GOTCHAS.md` (Hy3-NaN-Story) · `TURBOQUANT.md` (NVFP4-Kernel-Matrix
SM121) · `sglang_launch_patch_refactor_plan.md` (dieser Refactor, alle Phasen + Lehren) ·
`UPSTREAM_*.md` / `SGLANG_*_UPSTREAM_BUG.md` (je ein Bug).
