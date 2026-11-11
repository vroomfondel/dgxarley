# DeepSeek-V4-Flash `kv_lora_rank: null` config-parse crash (transformers 5.x + SGLang 0.5.12 / 0.5.13.x / 0.5.14)

> **Status 2026-05-31:** Workaround aktiv als Monkey-Patch in
> `roles/k8s_dgx/files/sglang_launch.sh` (Block `PATCH_DSV4_KVLORA_EOF`,
> Commits `8b5a268` + `1b36164`). Klärt nur den Config-Parse-Blocker — Flash
> kann downstream noch auf weitere Upstream-Issues treffen (sgl-project/sglang
> #25165 / #23743). Begleitend zu Modellprofil
> `roles/k8s_dgx/model_profiles/nvidia-deepseek-v4-flash-nvfp4.yml` and
> `roles/k8s_dgx/model_profiles/redhatai-deepseek-v4-flash-nvfp4-fp8.yml`.
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
>
> **Re-verified 2026-07-23:** v0.5.15 (released 2026-07-10) and v0.5.15.post1
> (released 2026-07-14, now the latest release) checked — release notes
> contain no `kv_lora_rank` / DSV4-Flash config-parse fix; `transformers` main
> still declares `kv_lora_rank: int = 512` in `configuration_deepseek_v3.py`.
> Issues #25165 / #23743 are now **CLOSED (auto stale-bot 2026-07-13 /
> 2026-07-14, NOT resolved** — both closed by the `github-actions` inactivity
> bot with no fix landed, not by upstream action). Monkey-patch still
> required. **Location update:** the workaround was relocated on 2026-07-16
> (commit `a760767`, SGLang patches Phase 2) from `sglang_launch.sh`
> (`PATCH_DSV4_KVLORA_EOF`) to its own gated file
> `roles/k8s_dgx/files/sglang_patches/p56_deepseek_v3_kv_lora.py`.
>
> **Re-verified 2026-07-28:** checked against v0.5.16 (released 2026-07-25,
> now the latest release). Release notes contain no `kv_lora_rank` /
> DSV4-Flash config-parse fix. `transformers` main still declares
> `kv_lora_rank: int = 512` in `configuration_deepseek_v3.py`, unchanged.
> Monkey-patch still required, no location change.
>
> **Re-verified 2026-08-09:** checked against **v0.5.17** (released
> 2026-08-08, now the latest release; 582 PRs since v0.5.16). Release notes
> contain no `kv_lora_rank` / DSV4-Flash config-parse fix. `transformers`
> main still declares `kv_lora_rank: int = 512`
> (`configuration_deepseek_v3.py:93`), unchanged. Monkey-patch still
> required, no location change.
>
> **Re-verified 2026-08-15:** no new SGLang release, v0.5.17 remains latest.
> `transformers` shipped **v5.15.0** (2026-08-10), but `kv_lora_rank: int = 512`
> is still present unchanged at `configuration_deepseek_v3.py:93` on both
> `main` and the v5.15.0 tag. Monkey-patch still required.
>
> **Re-verified 2026-08-21:** transformers shipped a patch release v5.15.1
> (2026-08-19); `kv_lora_rank: int = 512` is still present unchanged at
> `configuration_deepseek_v3.py:93` on both `main` and the v5.15.1 tag
> (release notes: DFlash/MTP candidate-generator + image-processing fixes
> only, nothing DeepSeek-V4-related). SGLang still at v0.5.17 (no new
> release). Monkey-patch still required.
>
> **Re-verified 2026-08-28:** SGLang **v0.5.18** (released 2026-08-22, now
> the latest release) checked; release notes and diff contain no
> `kv_lora_rank` / DeepSeek-V4-Flash config-parse fix. `transformers` shipped
> two more patch releases since the 2026-08-21 check, **v5.16.0** and
> **v5.16.1** (both 2026-08-26; v5.16.1 adds GLM-5.3-Flash support plus two
> small fixes) — `kv_lora_rank: int = 512` is still present unchanged at
> `configuration_deepseek_v3.py:93` on both `main` and the v5.16.1 tag;
> neither release's changelog mentions DeepSeek-V3/V4 config or
> `sub_configs`/`PretrainedConfig` auto-init changes (only unrelated
> DeepSeekV2 test-OOM and vocab-size fixes). Monkey-patch still required.
>
> **Re-verified 2026-09-02:** no new transformers release since v5.16.1
> (2026-08-26); no new SGLang release since v0.5.18 (2026-08-22). Direct
> source check confirms `kv_lora_rank: int = 512` is still present, unchanged,
> at `configuration_deepseek_v3.py:93` on both the v5.16.1 tag and
> `transformers` `main`. Cross-reference: `UPSTREAM_DSV4_BUGS.md` §1/Wall 1
> re-confirmed the same for this cycle. Monkey-patch still required, no
> location change.
>
> **Re-verified 2026-09-10:** transformers shipped v5.17.0 (2026-09-09, now
> the latest release; headline is the new HYV4 model, an MLA+DSA architecture
> unrelated to DeepSeek-V3/V4's own config class). Direct source check
> confirms `kv_lora_rank: int = 512` is still present, unchanged, at
> `configuration_deepseek_v3.py:93` on both the v5.17.0 tag and
> `transformers` `main`. Release notes contain no DeepSeek-V3/V4 config or
> `sub_configs`/`PretrainedConfig` changes. No new SGLang release since
> v0.5.18 relevant to this bug (v0.5.19, released 2026-09-05, does not touch
> `kv_lora_rank`/DeepSeek-V4-Flash config parsing either; cross-reference:
> `UPSTREAM_DSV4_BUGS.md` §1/Wall 1 re-confirmed the same for this cycle).
> Monkey-patch still required, no location change.
>
> **Re-verified 2026-09-16:** no new transformers release since v5.17.0
> (2026-09-09); no new SGLang release since v0.5.19 (2026-09-05). Direct
> source check confirms `kv_lora_rank: int = 512` is still present, unchanged,
> at `configuration_deepseek_v3.py:93` on both the v5.17.0 tag and
> `transformers` `main`. Cross-reference: `UPSTREAM_DSV4_BUGS.md` §1/Wall 1
> re-confirmed the same for this cycle; local patch
> `roles/k8s_dgx/files/sglang_patches/p56_deepseek_v3_kv_lora.py` re-checked
> against the 2026-09-11 sglang-0.5.19 patch re-anchor (commit `958c98d`) and
> is unaffected (not touched by that commit, not listed in
> `RETIRED_PATCHES.md`). Monkey-patch still required, no location change.
>
> **Re-verified 2026-09-22:** SGLang **v0.5.20** released 2026-09-18 (now
> latest release, 713 PRs since v0.5.19; `gh release list` re-checked). No
> new transformers release since v5.17.0 (2026-09-09). Direct source check
> confirms `kv_lora_rank: int = 512` is still present, unchanged, at
> `configuration_deepseek_v3.py:93` on both the v5.17.0 tag and
> `transformers` `main`. SGLang's config-alias mechanism for
> `model_type="deepseek_v4"` (`_DeepseekV4ConfigAlias(_HFDeepseekV3Config)`,
> registered in `sglang/srt/utils/hf_transformers/common.py:189-193`) is
> unchanged at the v0.5.20 tag, still a subclass of transformers'
> `DeepseekV3Config`, still does not override or retype `kv_lora_rank`
> itself. Cross-reference: `UPSTREAM_DSV4_BUGS.md`'s 2026-09-22 entry covers
> v0.5.20's DeepSeek-V4 changes in depth (mostly DeepGEMM/SM120 enablement,
> explicitly NOT covering SM121/GB10 as shipped, see PR #39482,
> unreleased), none of it touches config parsing. Local patch
> `roles/k8s_dgx/files/sglang_patches/p56_deepseek_v3_kv_lora.py` anchor
> (`    kv_lora_rank: int = 512`) still matches verbatim. Monkey-patch still
> required, no location change.
>
> Also noted, informational only: `deepseek-ai/DeepSeek-V4.1-Flash` (HF,
> createdAt 2026-09-10, `model_type: deepseek_v41`) is now a real release,
> transformers has no `deepseek_v41` model directory yet (checked `main`),
> so this bug's scope (`deepseek_v3`/`deepseek_v4` config alias) does not
> currently extend to it either way. Re-audit if/when V4.1 gets transformers
> and SGLang support.
>
> **Re-verified 2026-09-25:** no new transformers release since v5.17.0
> (2026-09-09); no new SGLang release since v0.5.20 (2026-09-18, `gh release
> list` re-checked). Direct source check confirms `kv_lora_rank: int = 512`
> is still present, unchanged, at `configuration_deepseek_v3.py:93` on both
> the v5.17.0 tag and `transformers` `main`. Deploy note: the cluster's
> SGLang image was bumped to `xomoxcc/dgx-spark-sglang:0.5.20-sm121`
> (commit `3214ed2`, 2026-09-22; recipe
> `scripts/patches/sglang-0.5.20-sm121.recipe`), local patch
> `roles/k8s_dgx/files/sglang_patches/p56_deepseek_v3_kv_lora.py`'s anchor
> (`    kv_lora_rank: int = 512`) still matches verbatim and is untouched by
> the 0.5.20 patch set (absent from `RETIRED_PATCHES.md`), confirming the
> 09-16 prediction that p56 "now runs on the 0.5.20 image". Cross-reference:
> `UPSTREAM_DSV4_BUGS.md`'s 2026-09-25 entry covers v0.5.20/unreleased
> upstream DeepSeek-V4 movement in depth; none of it touches `kv_lora_rank`
> or the `_DeepseekV4ConfigAlias` mechanism. Monkey-patch still required, no
> location change.
>
> **Re-verified 2026-09-28:** no new transformers release since v5.17.0
> (2026-09-09); no new SGLang release since v0.5.20 (2026-09-18, `gh release
> list` re-checked). Direct source check confirms `kv_lora_rank: int = 512`
> is still present, unchanged, at `configuration_deepseek_v3.py:93` on both
> the v5.17.0 tag and `transformers` `main`. `_DeepseekV4ConfigAlias`
> (`sglang/srt/utils/hf_transformers/common.py`) unchanged on current
> upstream/main, no commits to that file since 09-25. Cross-reference:
> `UPSTREAM_DSV4_BUGS.md`'s 2026-09-28 entry found no movement affecting
> config parsing either. Local patch
> `roles/k8s_dgx/files/sglang_patches/p56_deepseek_v3_kv_lora.py`'s anchor
> unchanged, not touched by any commit since 09-25. Monkey-patch still
> required, no location change.
>
> **Re-verified 2026-10-03:** transformers shipped v5.18.0 (2026-09-30), SGLang shipped v0.5.21 (2026-10-02, tag commit e00930c5489053f26d86b179cee0d087f846acbb). Direct source check confirms `kv_lora_rank: int = 512` is still present, unchanged, at `configuration_deepseek_v3.py:93` on the v5.18.0 tag, so DeepSeek-V4-Flash's `kv_lora_rank: null` still fails strict validation upstream. SGLang v0.5.21 still pins `transformers==5.12.1` (same as v0.5.20); upstream main bumped to 5.17.0 via #39012 (547286bb5a, 2026-09-30), not in v0.5.21. `_DeepseekV4ConfigAlias(_HFDeepseekV3Config)` in `sglang/srt/utils/hf_transformers/common.py` is still present at v0.5.21 with no `kv_lora_rank` handling (the file only grew DeepSeek-V4.1 config class registration, +30 lines vs v0.5.20). Local patch `roles/k8s_dgx/files/sglang_patches/p56_deepseek_v3_kv_lora.py` anchor (`    kv_lora_rank: int = 512`) unchanged and not touched locally since 09-28. Monkey-patch still required, no location change. Cross-reference: `UPSTREAM_DSV4_BUGS.md` 2026-10-03 entry (v0.5.21 contains #39482/#38792; unrelated to config parsing).
>
> **Re-verified 2026-10-11:** transformers shipped v5.19.0 (2026-10-06, now latest; v5.18.0 was 2026-09-30). Direct source check confirms `kv_lora_rank: int = 512` is still present, unchanged, at `configuration_deepseek_v3.py:93` on the v5.19.0 tag, so DeepSeek-V4-Flash's `kv_lora_rank: null` still fails strict validation upstream. No new SGLang release since v0.5.21 (2026-10-02); `_DeepseekV4ConfigAlias(_HFDeepseekV3Config)` in `sglang/srt/utils/hf_transformers/common.py` at v0.5.21 still has no `kv_lora_rank` handling. Local patch `roles/k8s_dgx/files/sglang_patches/p56_deepseek_v3_kv_lora.py` anchor (`    kv_lora_rank: int = 512`) is unchanged and untouched locally (the cluster now runs the 0.5.21-sm121 image). Monkey-patch still required, no location change.

> **Re-verified 2026-11-10:** transformers shipped v5.20.2 (2026-10-19), v5.21 (2026-10-26), v5.22.0 (2026-10-29) and v5.22.2 (2026-11-06, now latest). Direct source check confirms `kv_lora_rank: int = 512` is still present, unchanged, at `configuration_deepseek_v3.py:93` on the v5.22.2 tag and on main; the only commits to that file since 10-12 are #48920 and #49160 (2026-10-05/06, MoE router/ep_plans, unrelated). transformers also ships its own `models/deepseek_v4` (`DeepseekV4Config`, no `kv_lora_rank` match; present at v5.19.0 and v5.22.2), but SGLang does not use it: `_CONFIG_REGISTRY["deepseek_v4"] = _DeepseekV4ConfigAlias(_HFDeepseekV3Config)` is unchanged with no `kv_lora_rank` handling at SGLang v0.5.22 (2026-10-19), v0.5.23 (2026-11-02) and main (170804b40a, 2026-11-10), all pinning `transformers==5.19.0`. On main the file moved to `python/sglang/srt/hf_transformers/common.py` (#48758, 93cfea6b12, 2026-11-05); tags still use `srt/utils/hf_transformers/common.py`. Local patch `roles/k8s_dgx/files/sglang_patches/p56_deepseek_v3_kv_lora.py` anchor (`    kv_lora_rank: int = 512`) is unchanged and untouched locally. Monkey-patch still required, no location change inside transformers; the SGLang-side alias path changes with the next tag after v0.5.23.


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
