# DeepSeek-V4-Flash `kv_lora_rank: null` config-parse crash (transformers 5.x + SGLang 0.5.12 / 0.5.13.x)

> **Status 2026-05-31:** Workaround aktiv als Monkey-Patch in
> `roles/k8s_dgx/files/sglang_launch.sh` (Block `PATCH_DSV4_KVLORA_EOF`,
> Commits `8b5a268` + `1b36164`). Klärt nur den Config-Parse-Blocker — Flash
> kann downstream noch auf weitere Upstream-Issues treffen (sgl-project/sglang
> #25165 / #23743). Begleitend zu Modellprofil
> `roles/k8s_dgx/model_profiles/sgl-project-deepseek-v4-flash-fp8.yml`.
>
> **Re-verifiziert 2026-06-22:** Workaround weiterhin nötig.
> `kv_lora_rank: int = 512` in transformers' `configuration_deepseek_v3.py`
> (`main`) unverändert; SGLang v0.5.13 / v0.5.13.post1 (letztes Release,
> 2026-06-15) typisieren das Feld nicht als Optional und ersetzen
> `_DeepseekV4ConfigAlias` nicht. Issues #25165 / #23743 weiterhin offen.
> Querverweis: `UPSTREAM_DSV4_BUGS.md` §1 / Wall 1 bestätigt den Patch „in
> v0.5.13 weiterhin nötig". (Titel nennt SGLang 0.5.12 — Patch gilt unverändert
> auch für v0.5.13.x.)
>
> **Re-verified 2026-06-24:** Issues #25165 / #23743 still OPEN;
> `kv_lora_rank: int = 512` unchanged in transformers main; no fix in v0.5.13.
> v0.5.13.post1 is a bare git tag (no GitHub Release, no Docker image) — not a
> delivery vehicle. Monkey-patch still required.
>
> **Re-verified 2026-06-29:** v0.5.14 released 2026-06-26 (now the latest
> release). Release notes contain no `kv_lora_rank` / DSV4-Flash config-parse
> fix; issues #25165 / #23743 still OPEN; `kv_lora_rank: int = 512` unchanged
> in transformers main. Monkey-patch still required.


## Summary

`sgl-project/DeepSeek-V4-Flash-FP8` crasht beim Startup im Config-Parse mit:

```
StrictDataclassFieldValidationError: Field 'kv_lora_rank' expected int,
got NoneType (value: None)
```

Grund: Die V4-**Flash**-`config.json` hat `kv_lora_rank: null` (Flash nutzt
q-LoRA + o-LoRA + GQA, **keine** MLA-KV-Compression), aber das von SGLang für
`model_type="deepseek_v4"` instanziierte Config-Objekt erbt einen strict
typisierten `int`-Field aus transformers.


## Root Cause

SGLang instanziiert für `model_type="deepseek_v4"` die Klasse
`_DeepseekV4ConfigAlias` (`sglang/srt/utils/hf_transformers/common.py`), die von
transformers' `DeepseekV3Config` **erbt**. Der strenge Dataclass-Field

```python
# transformers/models/deepseek_v3/configuration_deepseek_v3.py
kv_lora_rank: int = 512
```

wird unter transformers 5.x von `huggingface_hub`'s `@strict`-Validator geprüft.
`None` schlägt durch, weil der Field als reiner `int` deklariert ist.

Wichtig: Der Field lebt in **transformers**, NICHT in SGLangs eigenem
`configs/deepseek_v4.py` (diese Datei wird für `model_type="deepseek_v4"` nie
benutzt). Ein Patch an SGLang-Sources würde also ins Leere greifen.

DeepSeek-V3 / V3.2 / Kimi-K2 (die anderen Modelle, die diese Config teilen)
liefern immer einen `int`, daher fiel das bisher nie auf.


## Workaround

Vor dem Import die Annotation auf `int | None` verbreitern — dann baut `@strict`
einen Union-Validator, der `None` akzeptiert:

```python
old = "    kv_lora_rank: int = 512"
new = "    kv_lora_rank: int | None = 512"
```

Sichere Verbreiterung: Die anderen Modelle übergeben weiter einen `int` und sind
unbetroffen; nur V4-Flashs `null` passiert jetzt. **`None` wird bewusst NICHT zu
einem `int` gecoerct** — ein `int` würde das Modeling auf den MLA-KV-LoRA-Pfad
zwingen, den die Flash-Weights nicht haben. `.pyc` ist timestamp-invalidiert,
der Edit greift also beim Reimport.

Pfad (im Container): `/usr/local/lib/python3.12/dist-packages/transformers/models/deepseek_v3/configuration_deepseek_v3.py`.
Der Patch ist idempotent (Marker-Grep `kv_lora_rank: int = 512`) und no-op, wenn
bereits angewandt oder der Marker fehlt.


## Re-Sync-Regel

Beim Bumpen von transformers oder des SGLang-Image:

1. Prüfen, ob der Marker `    kv_lora_rank: int = 512` in
   `configuration_deepseek_v3.py` noch exakt so existiert (Pfad/Python-Version
   können sich ändern — aktuell `python3.12`).
2. Prüfen, ob Upstream den Field selbst auf `int | None` gezogen hat (dann
   Patch entfernen) — Tracking via sgl-project/sglang #25165 / #23743 und der
   transformers `deepseek_v3`-Config.
3. Prüfen, ob SGLang `_DeepseekV4ConfigAlias` durch eine eigene, korrekt
   typisierte V4-Config ersetzt hat (dann greift der transformers-Patch ggf.
   nicht mehr und muss verlagert werden).


## Verwandte Themen

- NVFP4 ist für V4-Flash auf SGLang aktuell ein Dead End: RedHatAIs
  compressed-tensors-Repackage zielt auf `fused_wqa_wkv`, SGLangs V4-Loader
  erwartet `wqkv_a` → `find_matched_target()` ValueError. Open: #23724. Deshalb
  fahren wir den FP8-Checkpoint. Details im Modellprofil-Header.
- Begleitdokumente: `SGLANG_v0.5.12_VERSION_CHANGES.md` §2,
  `SGLANG_v0.5.12.post1_VERSION_CHANGES.md` §1, `TODO_0.5.12.md` §0.
