# Plan: SGLang-Runtime-Patches aus `sglang_launch.sh` in `files/sglang_patches/` auslagern

Status: Vorschlag, nichts implementiert. Kein Deploy ohne Freigabe.

## Ausgangslage

`roles/k8s_dgx/files/sglang_launch.sh` ist auf 3476 Zeilen / 193 KB gewachsen. Der Löwenanteil sind
Runtime-Patches gegen `/usr/local/lib/python3.12/dist-packages/sglang/...`:

* ca. 25 Python-Heredocs (`python3 - <<'PATCH_*_EOF'`), z. B. `PATCH_HUNYUAN_SHARED_EOF`,
  `PATCH_MLLAMA4_LOADER_EOF`, `PATCH_DSA_TORCH_*` (5 Stück), `PATCH_DSA_FLASHINFER_GATHER_EOF`
  (270 Zeilen allein), `PATCH_MIXED_NVFP4_*`, `PATCH_QWEN35_*`, `PATCH_VLM_IGNORE_EOF`,
  `PATCH_NEMOTRONH_OMNI_WRAPPER_EOF`, `PATCH_FI_*`.
* ca. 8 sed-/grep-basierte Blöcke (`WEIGHT_UTILS`, `LOADER`, `MOE_WNA16`, `MODELOPT_QUANT`,
  `CUTLASS_MOE`, `MINIMAX_M2`, `DEEPSEEK_V3_CFG`, `FST_F`, `HF_UTILS`, `TOKPY`).
* echter Launcher-Anteil (apt-Bootstrap, `.pth`-Installation, Flag-Zusammenbau, `exec`):
  geschätzt unter 400 Zeilen.

Probleme daraus:

1. Jede Patch-Änderung ist ein Diff mitten in einer 3.5k-Zeilen-Datei, Review ist schwer.
2. Der Heredoc-Inhalt ist für Editor/Tooling kein Python: keine Syntaxprüfung, kein black, kein mypy.
3. Gate-Logik (Bash-`if` außen) und Patch-Logik (Python innen) sind getrennt, das Gate ist beim Lesen
   des Patches oft 50 Zeilen weiter oben.
4. Boilerplate wird copy-pasted: Datei lesen, Marker-Guard, `replace(..., 1)`, `ANCHOR-DRIFT`-print.
   Genau dort saß der Idempotenz-Bug vom 2026-07-16 (`old_buffered` ist Präfix von `new_buffered`).
5. Ein Patch, der nur ein Modell betrifft, rollt trotzdem jeden Pod neu (ein Checksum über die
   ganze Datei).

## Zielbild

```
roles/k8s_dgx/files/sglang_patches/
  _patchlib.py                        # gemeinsame Helfer, kein Patch
  10_weight_utils_tqdm_logger.py
  10_loader_shard_progress.py
  20_modelopt_mixed_nvfp4_dispatch.py
  20_modelopt_mixed_nvfp4_variant.py
  20_linear_nvfp4_scale.py
  20_vlm_should_ignore_layer.py
  20_moe_wna16_qzeros_ep.py
  30_dsa_torch_backend.py
  30_dsa_flashinfer_gather.py
  40_hy3_nextn_bf16.py
  40_ds_nextn_mixed_mtp.py
  50_hunyuan_token_suffix.py
  50_mllama4_loader.py
  ...
```

`sglang_launch.sh` schrumpft auf Bootstrap + Patch-Runner + Flag-Bau, realistisch 400 bis 500 Zeilen.

### Ein Patch = eine Datei = ein Python-Modul, self-gating

Jeder Patch entscheidet **selbst** anhand von `os.environ`, ob er zutrifft, statt von einem
Bash-`if` umschlossen zu werden. Das hält Gate und Patch beieinander und macht den Runner dumm.
Der Preis (ein `python3`-Start pro Patch, ca. 0,2 s, also ~5 s gesamt) ist gegen die 7 bis 8 Minuten
Head-Startup irrelevant.

```python
"""[dgxarley] hunyuan_v3.py: remap .shared_experts. -> .shared_mlp. in load_weights.

Grund: HYV3-Checkpoints benennen den Shared Expert `mlp.shared_experts.*`, SGLangs Modul
heißt `shared_mlp`; ohne Remap werden die (echten, FP4) Gewichte still verworfen -> NaN.
Upstream: noch nicht eingereicht.
Re-Sync: bei Image-Bump prüfen, ob load_weights den Remap schon hat (Guard no-opt dann).
"""
from _patchlib import Patch, gate_model

patch = Patch(
    name="hunyuan-shared-experts",
    target="sglang/srt/models/hunyuan_v3.py",
    when=gate_model("Hy3", "Hunyuan"),
)

@patch.run
def apply(p):
    p.insert_after(
        anchor="        for name, loaded_weight in weights:\n",
        text='            name = name.replace(".shared_experts.", ".shared_mlp.")\n',
        marker='replace(".shared_experts.", ".shared_mlp.")',
    )
```

`_patchlib.py` liefert genau das, was heute pro Block dupliziert wird:

* `Patch(name, target, when=...)`: löst `target` gegen `dist-packages` auf, meldet
  `ANCHOR-DRIFT: <name>: target file missing` statt zu crashen, überspringt bei `when=False`
  mit einer Zeile Log.
* `p.replace(old, new, marker=...)` / `p.insert_after(anchor, text, marker=...)`:
  Marker-Guard **zuerst** (der 2026-07-16-Bug ist damit strukturell ausgeschlossen), exakt eine
  Ersetzung, einheitliches `Patched <file>: <name>` bzw.
  `ANCHOR-DRIFT: <file>: <name> (SGLang version drift; re-check anchor)`.
* `p.write_new_file(relpath, content)` für die Fälle wie `PATCH_DSA_TORCH_NEWFILE_EOF`.
* Gate-Helfer: `gate_model(*substrings)`, `gate_env("SGLANG_SPECULATIVE_ENABLED", "true")`,
  `gate_always()`.
* Rückgabe-Konvention: Exit 0 immer (auch bei Drift), damit `set -e` im Launcher nicht den Pod
  killt. Genau das heutige Verhalten, aber an einer Stelle statt 30-mal.

### Was NICHT in `sglang_patches/` gehört

* apt-Bootstrap, `pip install accelerate`, Transformers-Upgrade.
* `.pth`-Installation (`zz_dsv4_autopatch.pth`, `zz_dsv4_memprobe.pth`), das sind Deployments,
  keine Source-Patches.
* Image-Pattern-Check, Flag-Zusammenbau, `exec`.
* `SGLANG_HUNYUAN_TOKEN_SUFFIX`-Ermittlung: liest `tokenizer_config.json` und **exportiert eine
  Env-Var** für den Serverprozess, ist also Launcher-Logik. Bleibt in der `.sh`, die beiden
  Detector-Patches wandern ins Patch-Verzeichnis und lesen die Var.

### Auslieferung

Neue ConfigMap `{{ inst.prefix }}-patch-scripts`, gemountet auf `/patches` (eigener Top-Level-Pfad,
kein Nested-Mount unter dem bestehenden `/scripts`). `sglang_launch.sh` bekommt:

```bash
SGLANG_PATCH_DIR="${SGLANG_PATCH_DIR:-/patches}"
if [ -d "$SGLANG_PATCH_DIR" ]; then
  export PYTHONPATH="$SGLANG_PATCH_DIR:${PYTHONPATH:-}"
  for _p in "$SGLANG_PATCH_DIR"/[0-9][0-9]_*.py; do
    [ -e "$_p" ] || continue
    python3 "$_p" || echo "[launch] WARNING: patch $(basename "$_p") exited non-zero, continuing"
  done
fi
```

Sortierung über das `NN_`-Präfix, also deterministisch und ohne Registry-Datei. Präfixgruppen:
`10` Loader/Progress, `20` Quant, `30` Attention/DSA, `40` Spekulativ/MTP, `50` Modelle/Parser,
`60` Flashinfer/Env.

Ansible-Seite in `roles/k8s_dgx/tasks/sglang_instance.yml`:

```yaml
- name: Create SGLang patch-scripts ConfigMap ({{ inst.prefix }})
  kubernetes.core.k8s:
    definition:
      apiVersion: v1
      kind: ConfigMap
      metadata:
        name: "{{ inst.prefix }}-patch-scripts"
        namespace: "{{ sglang_namespace }}"
      data: >-
        {{ dict(_sglang_patch_files | map('basename') | zip(
                _sglang_patch_files | map('_file_content'))) }}
  vars:
    _sglang_patch_files: "{{ query('fileglob', role_path ~ '/files/sglang_patches/*.py') | sort }}"
```

`lookup('file')` in einem `map()` geht nicht direkt, praktikabel ist stattdessen eine
`ansible.builtin.set_fact`-Schleife über `query('fileglob', ...)` mit
`combine({ item | basename: lookup('file', item) })`. Basenames sind gültige ConfigMap-Keys
(`[-._a-zA-Z0-9]`), Unterverzeichnisse gibt es bewusst nicht.

**Größenbudget:** ConfigMaps sind hart auf 1 MiB begrenzt. Heute liegen 193 KB (`launch.sh`)
plus `dsv4_memprobe.py` in einer ConfigMap. Nach dem Split etwa 30 KB `launch.sh` plus ca. 160 KB
verteilt auf die Patch-ConfigMap. Reserve bleibt reichlich, aber die Aufteilung auf zwei ConfigMaps
verdoppelt den Puffer, statt ihn zu verbrauchen.

### Der Checksum-Fallstrick (wichtig)

Heute:

```yaml
checksum/launch-script: "{{ (lookup('file', .../sglang_launch.sh) ~ lookup('file', .../dsv4_memprobe.py)) | hash('sha256') }}"
```

Zieht man die Patches raus, ohne die Annotation zu erweitern, rollt eine reine Patch-Änderung die
Pods **nicht** mehr neu, die ConfigMap-Änderung propagiert zwar in den Mount, aber der Patch läuft
nur beim Container-Start. Ergebnis wäre ein stiller Nicht-Effekt, der aussieht wie ein wirkungsloser
Patch. Die Annotation muss also (an beiden Stellen, Zeile 406 und 717) um den Verzeichnis-Hash
erweitert werden:

```yaml
checksum/launch-script: "{{ (lookup('file', .../sglang_launch.sh)
                            ~ lookup('file', .../dsv4_memprobe.py)
                            ~ _sglang_patches_blob) | hash('sha256') }}"
```

mit `_sglang_patches_blob` = Konkatenation der sortierten Patch-Dateien (ein `set_fact` vor den
Deployment-Tasks, einmal berechnet, von Head und Worker geteilt).

Merke außerdem: `lookup('file')` strippt das schließende `\n`, ein Vergleich ConfigMap-Inhalt gegen
Quelldatei mismatcht deshalb immer (siehe `reference_ansible_file_lookup_trailing_newline`). Für den
Hash ist das egal, solange beide Seiten denselben Weg gehen.

## Migration in Phasen (jede Phase ist einzeln deploybar und rückrollbar)

**Phase 0, Gerüst, kein Verhaltens-Delta.**
`_patchlib.py` + Runner-Loop + ConfigMap + Mount + Checksum-Erweiterung. Noch kein Patch verschoben.
Verifikation: Head bootet, Log identisch bis auf die neue Runner-Zeile.

**Phase 1, ein Pilot-Patch.**
`20_moe_wna16_qzeros_ep.py` (klein, unkonditioniert, gut getestet) raus aus der `.sh`.
Verifikation siehe unten (Tree-Diff).

**Phase 2, die unkonditionierten Patches.**
mllama4 (2), weight_utils/loader-Progress, linear NVFP4-Scale, VLM-ignore, Nemotron-Wrapper,
Transformers-topk, FP8-out-dtype, Flashinfer (2), modelopt/cutlass/minimax/deepseek-cfg.

**Phase 3, die gegateten Patches.**
Hunyuan-Detektoren (2) + shared-experts, HY3-NEXTN (2), DS-NEXTN-mixed-MTP, DSA-torch (5),
DSA-flashinfer-gather. Hier wandert das Bash-`if` in das `when=` des Patches, das ist der einzige
Schritt mit echtem Logik-Umzug, also der riskanteste. Die 5 `PATCH_DSA_TORCH_*` teilen ein Gate und
gehören zusammen in **eine** Datei (`30_dsa_torch_backend.py`), sonst laufen sie auseinander.

**Phase 4, Aufräumen.**
sed-Blöcke, die noch übrig sind, nach Python konvertieren (sie sind ohnehin Anchor-Replacements),
`.sh` durchlesen, Reste an Kommentar-Kontext zu den Patch-Docstrings verschieben, CLAUDE.md-Abschnitt
"SGLang ConfigMap scripts" um das Patch-Verzeichnis ergänzen.

## Verifikation: Tree-Diff statt Hoffnung

Der Refactor ist genau dann korrekt, wenn der **gepatchte dist-packages-Baum identisch** ist. Das ist
direkt messbar, ohne SGLang überhaupt zu starten:

1. Debug-Pod auf einem Spark (`tail -f /dev/null`, kein `app=sglang`-Label, siehe
   `feedback_debug_pod_no_sglang_label`), mit demselben Image und denselben `SGLANG_*`-Env-Vars wie
   der Head.
2. `cp -a /usr/local/lib/python3.12/dist-packages/sglang /tmp/base`
3. Alte `launch.sh` bis vor den `exec` laufen lassen (`SGLANG_PATCH_ONLY=1`-Guard einbauen, oder
   schlicht die Patch-Sektion per `sed -n` extrahieren), Baum nach `/tmp/old` sichern, aus
   `/tmp/base` zurückrollen.
4. Neue `launch.sh` + Runner, Baum nach `/tmp/new`.
5. `diff -r /tmp/old /tmp/new` muss leer sein.

Pro Phase einmal, mit den Env-Kombinationen der real genutzten Profile (mindestens: GLM-5.2-DSA,
Hy3-NVFP4-W4A4, ein mixed-NVFP4-Modell, ein Nicht-Gate-Modell). Das deckt die Gates ab, die der
Tree-Diff sonst nicht anfasst.

Zusätzlich zwei billige Dauerchecks:

* **Idempotenz-Test:** Runner zweimal laufen lassen, der zweite Lauf darf keine Datei mehr ändern
  (`diff -r` gegen den Zwischenstand leer) und muss für jeden Patch "already applied" loggen. Genau
  der Bug-Typ vom 2026-07-16, jetzt automatisch geprüft.
* **Lint:** die Patch-Dateien liegen als echtes Python im Repo, also greifen `make lint` (black,
  line-length 120) und `make tcheck` künftig darauf. `_patchlib.py` bekommt Typannotationen,
  kein `from __future__ import annotations` (Python 3.14+).

## Was der Split nicht löst

* Die Patches bleiben anchor-basiert und driften bei Image-Bumps weiterhin. Der Split macht die
  Drift nur sichtbarer (ein Patch = ein Dateiname im `ANCHOR-DRIFT`-Log statt einer Zeilennummer).
* Der Kommentar-Kontext (das eigentliche Wissen: warum, welcher Upstream-PR, wann löschbar) ist
  wertvoll und darf beim Umzug **nicht** verloren gehen, er wandert 1:1 in den Modul-Docstring.
  Kein Patch ohne Docstring mit Grund + Upstream-Status + Re-Sync-Hinweis.
* Startzeit ändert sich praktisch nicht (~5 s Interpreter-Starts gegen 7 bis 8 min Head-Boot).

## Offene Entscheidungen

1. `/patches` als eigener Mount (Vorschlag) oder zusätzliche Keys in der bestehenden
   `-launch-script`-ConfigMap mit `items[].path: patches/x.py`. Ersteres ist sauberer getrennt,
   letzteres spart einen Volume-Eintrag.
2. Ob `sglang_embed_launch.sh` denselben Runner bekommt (aktuell patcht es nichts, könnte aber von
   `20_*` profitieren) oder bewusst patchfrei bleibt.
3. Ob Patches pro Profil selektierbar werden sollen (`sglang_patches_disabled: [...]` als
   Profil-Knopf, der einzelne Dateien aus der ConfigMap auslässt). Nützlich zum Bisecten bei
   Image-Bumps, aber ein neuer Konfig-Knopf. Vorschlag: erst nach Phase 4, wenn überhaupt.
