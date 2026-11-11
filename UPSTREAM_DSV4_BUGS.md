# SGLang Upstream Bugs / Gaps: DeepSeek-V4-Flash on SM121 (DGX Spark)

## Status (verified 2026-05-31, upstream re-checked 2026-06-08, re-verified 2026-06-11, re-verifiziert 2026-06-14, 2026-06-22, 2026-06-29, 2026-07-23, 2026-07-28)

> **Update 2026-07-28 (drei Issue-States korrigiert, v0.5.16 released 2026-07-25).**
> Re-Check der 2026-07-23-Rechercheergebnisse deckte drei veraltete Issue-States
> auf:
> - **#23724** (compressed-tensors W4A16 für V4, §2) ist **CLOSED 2026-07-17**,
>   aufgelöst durch den gemergten PR **#25763** ("Support DeepSeek-V4 Wint4Abf16
>   and Win4Afp8", merged 2026-07-16T21:24Z, enthalten in v0.5.16). Wichtig,
>   damit der Unterschied nicht verlorengeht: #25763 liefert einen **nativen**
>   SGLang-Quant-Pfad (eigenes Checkpoint-Format und eigene Gewichtsnamen,
>   getestet gegen ein internes `.../DeepSeek-V4-Flash-FP8-W4ABF16-GS128`),
>   **keinen Fix des `wqkv_a`-vs-`fused_wqa_wkv`-Namensmatchers** aus §2. Ob
>   RedHatAI-, kylesayrs- oder canada-quant-NVFP4-Checkpoints damit laden, ist
>   NICHT verifiziert, der Matcher-Bug selbst ist unangetastet.
> - **#24111** ("About pre-converted FP8 checkpoints") ist **CLOSED 2026-07-11**
>   per Stale-Bot ("automatically closed due to inactivity"), NICHT gelöst
>   (gleiches Muster wie #25165 / #23743).
> - **#25526** (DSv4 Flash + HiCache breakable piecewise CUDA graph) ist
>   **CLOSED 2026-07-17**, ebenfalls per Stale-Bot, NICHT gelöst.
>
> **v0.5.16 released 2026-07-25** (aktuell neuestes Release). Im Abschnitt
> "DeepSeek V4" der Release Notes für uns relevant: **#30272** "Implement SM120
> DeepSeek V4 flashinfer_mxfp4 moe runner backend + TP2", explizit SM120 und
> damit **prüfenswert für GB10/SM121** (siehe §3, Notiz unten), **#30140** (non-
> paged indexer default für große Prefill-Chunks), **#30012** (BF16 statt FP32
> für die Indexer-Score-Berechnung), **#30645** (Fix Top-k-v2 invalide Indizes
> unter Tie-Overflow, IMA in FA3 sparse decode), **#31705** (Fix idle-rank
> dummy-extend sparse-prefill Crash unter DP breakable CUDA graph), **#32246**
> (Fix nvfp4 online scale mit pcg), **#32288** (Fix stale flashinfer-MLA-
> Fallback vergiftet spec-verify capture, trtllm_mla + tc_piecewise). Neues
> Known Issue [Korrektur 2026-08-07: **#31125 ist der gemergte Stopgap-PR**
> (2026-07-14), kein offenes Issue — es gibt KEIN offenes Tracking-Issue für
> das Problem selbst, siehe Referenztabelle]: Temp-0-Nondeterminismus unter
> DP-Attention mit
> breakable prefill CUDA graph auf der DSV4-Flash-**FP4**-Recipe (unser
> Checkpoint ist FP8, trifft uns also nur bei einem Wechsel), upstream nur per
> deaktiviertem Determinismus-Test entschärft, kein echter Fix. **Unvalidiert
> auf unserem Cluster, kein v0.5.16-Image gebaut, keine der obigen Walls
> praktisch getestet.**

> **Update 2026-07-23 — v0.5.15 + v0.5.15.post1 released; #25165/#23743 auto-closed (stale, unresolved).**
> SGLang **v0.5.15** (released 2026-07-10) and **v0.5.15.post1** (released
> 2026-07-14, now the latest release) shipped since the 2026-06-29 check.
> Both contain DeepSeek-V4/DSA work — `[DeepSeek V4] Enable FlashMLA sparse
> prefill by default` (#29775), `Add an opt-in non-paged indexer for
> long-context prefill` (#29619), and DSA top-k/page-table fusion (#26788,
> #30274, #27705) — plausibly relevant to **Wall 5** (`paged_mqa_logits` torch
> fallback) and **Wall 7** (`topk_transform_512` TVM crash), matching the
> 2026-06-19 "noch nicht geprüft" note below. **Unvalidated on our cluster —
> no v0.5.15-based image has been built/tested yet; Walls 5/7 workarounds
> stay in place until verified.** Issues **#25165** and **#23743** (listed
> below as "open") are now **CLOSED (auto stale-bot 2026-07-13 / 2026-07-14,
> NOT resolved** — closed by the `github-actions` inactivity bot, no fix
> landed). **#26324** (`flashinfer_trtllm` MoE runner asserts on DSV4-Flash
> NVFP4/B200) remains **open, unchanged** since 2026-06-15. The local
> kv_lora_rank workaround was relocated 2026-07-16 (commit `a760767`) from
> `sglang_launch.sh` to `roles/k8s_dgx/files/sglang_patches/p56_deepseek_v3_kv_lora.py`
> (see `SGLANG_DEEPSEEK_V4_FLASH_KV_LORA_RANK_UPSTREAM_BUG.md`).

> **Update 2026-06-29 — PR #25820 ist im offiziellen v0.5.14-Release enthalten (2026-06-26).**
> SGLang **v0.5.14** wurde am **2026-06-26T22:57Z** veröffentlicht und enthält
> [PR #25820](https://github.com/sgl-project/sglang/pull/25820) explizit in
> den Release Notes (Abschnitt „NVFP4 MoE for DeepSeek-V4"). Das Build-Recipe-Gate
> **`APPLY_DSV4_NVFP4_PR25820=1` ist damit obsolet** — bei einem Wechsel auf ein
> v0.5.14+-basiertes Image kann der Gate-Flag und der zugehörige Source-Patch
> entfernt werden (kein Eingriff in Build-Recipes/Dockerfiles nötig, bis das
> Cluster-Image auf v0.5.14 aktualisiert wird). Der `flashinfer_cutlass`-Pin im
> Modellprofil `nvidia-deepseek-v4-flash-nvfp4.yml` **bleibt für SM121 die
> richtige Wahl** — v0.5.14 default-routet weiterhin auf `flashinfer_trtllm_routed`
> (B200/SM100-only, vgl. #26324 noch offen). Ebenfalls in v0.5.14 enthalten:
> **PR #27986** „[dsv4] Prewarm MHC prenorm kernel at startup" (beseitigt den
> ~2-min-First-Request-JIT-Stall auf v0.5.14+, vgl. §6 Post-load warmup).

> **Update 2026-06-22 — PR #25820 ist GEMERGT (main).**
> [PR #25820](https://github.com/sgl-project/sglang/pull/25820) „[NVIDIA]
> Support NVFP4 MoE for DeepSeek-V4" wurde am **2026-06-22T02:35Z in den
> sglang-`main` gemergt** (vorher DIRTY / CHANGES_REQUESTED, vgl.
> 2026-06-16-Block). **Im offiziellen v0.5.14-Release seit 2026-06-26 enthalten**
> (zuvor: v0.5.13.post1, PyPI 2026-06-15, war vor dem Merge getaggt). Der PR
> routet weiterhin default auf `flashinfer_trtllm_routed` (B200/SM100-only,
> vgl. #26324; im PR-Body nur auf B200 getestet, kein SM120/121-Test) → der
> `flashinfer_cutlass`-Pin im Modellprofil `nvidia-deepseek-v4-flash-nvfp4.yml`
> bleibt für SM121 die richtige Wahl. #26324 (flashinfer_trtllm-Assert) war
> beim Merge noch offen und ist weiterhin offen.

> **2026-06-08 — DSV4 ist nicht mehr der aktive Default.** `defaults/main.yml`
> hat auf `sglang_model: RedHatAI/Qwen3.6-35B-A3B-NVFP4` auf Image
> `scitrera/dgx-spark-sglang:0.5.12` zurückgestellt; das DSV4-Flash-FP8-Modell
> und das `xomoxcc/…0.5.12.post1-sm121`-Image sind jetzt **auskommentiert**.
> Dieses Dokument gilt, wenn DSV4-Flash (wieder) aktiviert wird. Upstream-Deltas
> seit 2026-05-31: **#26209 (FP4 Indexer for V4) ist MERGED** (2026-06-02, in
> `main`, noch nicht released — nach v0.5.12.post1); **#19589 ist CLOSED**;
> **DeepGEMM #317 ist CLOSED** (Maintainer abgelehnt — kein SM120-Hardware —
> und hat auf Community-PR #318 verwiesen, der noch offen/ungemergt ist). Der
> NVFP4-MoE-PR #25820 ist weiterhin offen, ein vollständiger NVFP4-V4-Pfad auf
> SGLang ist also in Arbeit, aber nicht verfügbar.
>
> **2026-06-11 — SGLang-Tag v0.5.13 geschnitten** (2026-06-11T08:09:52Z);
> **offizielles GitHub-Release am 2026-06-13 veröffentlicht**. `sglang 0.5.13`
> wurde am 2026-06-11T10:16Z auf PyPI veröffentlicht. Enthält **PR
> #24692** (SM120-Support für DeepSeek-V4-Inference, merged 2026-06-01) sowie
> **#26209** (FP4 Indexer). Cluster-Images laufen weiterhin auf
> v0.5.12.post1-basierten Builds. PR #25820 (NVFP4 MoE) meldet laut letztem
> Kommentar vom 2026-06-11: „Flash NVFP4 is working now" (GSM8K 96.21% auf
> B200) — PR noch offen/ungemergt, kein SM120/121-Test erwähnt. Details zu den
> Wall-Auswirkungen von #24692 in §8.
>
> **Update 2026-06-12 — PR #25820 Status.** PR weiterhin offen; `/tag-and-rerun-ci`
> wurde 2026-06-12T07:18Z gepostet, `mergeable_state: blocked` (CI ausstehend),
> 13 Reviewer angefordert. Timeline: 2026-06-10 User-Report „Flash NVFP4 not
> working" (cuda_graph_runner exception) → 2026-06-10T17:28 trevor-m identifiziert
> veralteten quant_config im HF-Repo → 2026-06-11T11:50 „Flash NVFP4 is working
> now" nach HF-seitigem quant-config-Fix (GSM8K 96.21% auf B200; kein
> SM120/121-Test).
>
> **Re-verifiziert 2026-06-14 — v0.5.13 ist offizielles GitHub-Release seit
> 2026-06-13.** Alle früheren Formulierungen „noch kein GitHub-Release-Page" /
> „tag-only" / „Stand 2026-06-12" zu v0.5.13 sind damit veraltet.
> - PR #24692 (SM120-FlashMLA + FP8-Indexer) und PR #26209 (FP4-Indexer) sind
>   im offiziellen v0.5.13-Release enthalten.
> - PR #25820 (NVFP4 MoE für DSV4) ist weiterhin OFFEN und NICHT in v0.5.13
>   gemergt (updatedAt 2026-06-13, letzter Kommentar „update cookbook"). Die
>   lokale Build-Recipe `APPLY_DSV4_NVFP4_PR25820=1` bleibt nötig.
> - Wall 0 (FlashMLA sparse-decode) und Wall 6 (seq_lens assert) sind nativ in
>   v0.5.13 gelöst (via #24692 + `flash_mla_sparse_fwd` / `fp8_paged_mqa_logits_torch_sm120`) —
>   als redundant bestätigt (vgl. §8).
> - Wall 1 (kv_lora_rank=512 Patch) per Source-Inspektion in v0.5.13 weiterhin
>   nötig; `kv_lora_rank: int = 512` in `configuration_deepseek_v3.py`
>   unverändert. Walls 2/4/5/7 unverändert.
> - `scitrera/dgx-spark-sglang:0.5.13` noch nicht auf DockerHub; Cluster-Default
>   bleibt 0.5.12.
>
> **Update 2026-06-16 — PR #25820 vorangekommen (weiterhin offen/ungemergt).**
> Status hat sich seit dem 2026-06-12-Snapshot („`mergeable_state: blocked`,
> CI ausstehend, letzter Kommentar ‚update cookbook'") deutlich verändert:
> `mergeStateStatus: DIRTY` (Merge-Konflikt, NICHT mehr CI-pending),
> `reviewDecision: CHANGES_REQUESTED`. Neue Kommentare 2026-06-15: User SkqLiao
> bestätigt **sowohl DSV4-Flash-NVFP4 als auch DSV4-Pro-NVFP4 funktionierend**
> mit `flashinfer_trtllm_routed` (auf PR-Commit `29ac249be`); trevor-m bestätigt
> und verweist auf Begleit-PR #25702 für Performance. **Weiterhin OFFEN,
> ungemergt, kein SM120/121-Test im Thread** — der `flashinfer_cutlass`-Pin im
> Modellprofil bleibt für SM121 die richtige Wahl (PR default-routet auf
> `flashinfer_trtllm_routed`, B200/SM100-only, vgl. #26324). Lokales Recipe
> `APPLY_DSV4_NVFP4_PR25820=1` bleibt nötig.
>
> **Update 2026-06-12 — `nvidia/DeepSeek-V4-Flash-NVFP4` existiert.** Das HF-Repo
> `nvidia/DeepSeek-V4-Flash-NVFP4` (createdAt 2026-05-18T00:02Z) existiert —
> die Aussage in §5 „kein nvidia/DeepSeek-V4-Flash-NVFP4" war falsch. Siehe §5
> für die Korrektur und die Implikation für die Kapazitätsabschätzung.
>
> **Update 2026-06-12 (später) — lokaler Build-Pfad angelegt.**
> `scripts/patches/sglang-0.5.13-sm121.recipe` baut SGLang v0.5.13 (enthält
> #26209 + #24692) plus PR #25820 als Source-Patch
> (`sglang-dsv4-nvfp4-pr25820.patch`, rebased auf v0.5.13 vom PR-Head
> `f9c7cd475b`, Recipe-Gate `APPLY_DSV4_NVFP4_PR25820=1`) → Image-Tag
> `xomoxcc/dgx-spark-sglang:0.5.13-sm121`. Damit entfällt das Warten auf den
> #25820-Merge für einen Testboot. ACHTUNG: #25820 default-routet auf
> `flashinfer_trtllm_routed` (B200/SM100-only, vgl. #26324) — das Modellprofil
> `nvidia-deepseek-v4-flash-nvfp4.yml` pinnt deshalb `flashinfer_cutlass`.
> 0xSero-FlashMLA-Kernel + seq_lens-Squeeze sind in diesem Recipe entfernt
> (§8, Walls 0+6 nativ via #24692); Walls 1/2/4/5/7-Workarounds bleiben aktiv.

Summary of everything that blocks or constrains serving **DeepSeek-V4-Flash** on
our 4×GB10 / SM121 cluster under SGLang. Context (as written 2026-05-31): the
then-active model (selector today: `group_vars/all/main/sglang.yml`) was
**`sgl-project/DeepSeek-V4-Flash-FP8`** on image
**`xomoxcc/dgx-spark-sglang:0.5.12.post1-sm121`** (SGLang v0.5.12.post1) — see
the 2026-06-08 note above; this is no longer the default.
`DeepseekV4ForCausalLM` is mainline since PR #23882 (shipped in v0.5.12), but
Flash serving is still being stabilized upstream.

| #   | Issue                                                                                            | State                                                                                                                                                                                                                   | Impact for us                                                                                                                                                          |
|-----|--------------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| 1   | `kv_lora_rank=None` strict-dataclass crash at config parse                                       | **Worked around** (launch patch)                                                                                                                                                                                        | Blocks ANY V4-Flash checkpoint until patched                                                                                                                           |
| 2   | compressed-tensors `wqkv_a` vs `fused_wqa_wkv` target mismatch                                   | **Tracking issue #23724 closed 2026-07-17** (via merged PR #25763, a native Wint4A16/Wint4AFP8 quant path, NOT a fix of the `wqkv_a`/`fused_wqa_wkv` matcher, see 2026-07-28 update above)                              | Makes RedHatAI / kylesayrs / canada-quant NVFP4 unloadable                                                                                                             |
| 3   | NVFP4 MoE / FP4 indexer for V4 not implemented                                                   | **Partial** — FP4 indexer #26209 merged 2026-06-02 (main, in v0.5.13-Tag); NVFP4 MoE #25820 **merged 2026-06-22 (main, im offiziellen v0.5.14-Release seit 2026-06-26; default `flashinfer_trtllm_routed`, B200-only)** | NVFP4-Pfad in v0.5.14 verfügbar; `flashinfer_cutlass`-Pin für SM121 weiterhin nötig (vgl. #26324); `APPLY_DSV4_NVFP4_PR25820=1` bei Wechsel auf v0.5.14+-Image obsolet |
| 4   | NVFP4 runner instability on V4-Flash/Pro                                                         | **Open / partly closed** (#26324, #25704)                                                                                                                                                                               | Even where NVFP4 loads, output is NaN/garbage except EAGLE                                                                                                             |
| 5   | `nvidia/DeepSeek-V4-Pro-NVFP4` does not fit; `nvidia/DeepSeek-V4-Flash-NVFP4` needs verification | N/A (capacity)                                                                                                                                                                                                          | Pro: 913 GB vs 512 GB — does not fit. Flash NVFP4 (~162B params) may fit TP=4 — unverified, see §5                                                                     |

**Net conclusion (Stand 2026-05-31, teilweise überholt):**
`sgl-project/DeepSeek-V4-Flash-FP8` (block-wise FP8) is the **only viable V4
path on SGLang** for this cluster today. NVFP4 of V4-Flash remained a dead end
on two independent axes — incomplete runtime support (#25820 offen) **and**
fehlender SGLang-nativer Checkpoint. ~~no loadable checkpoint~~ — **Update
2026-06-12:** `nvidia/DeepSeek-V4-Flash-NVFP4` existiert seit 2026-05-18 (§5).
Die Achse „kein Checkpoint" entfällt damit sobald #25820 gemergt ist; die Achse
„unvollständige Runtime" bleibt bis dahin bestehen. NVFP4 via vLLM (RedHatAI
checkpoint, vLLM-targeted) ist weiterhin eine Alternative.

---

## 1. `kv_lora_rank=None` — strict-dataclass crash at config parse (WORKED AROUND)

**Symptom** — head/worker exit immediately at startup:
```
huggingface_hub.errors.StrictDataclassFieldValidationError: Validation error for
field 'kv_lora_rank':
    TypeError: Field 'kv_lora_rank' expected int, got NoneType (value: None)
```

**Root cause:**
- DeepSeek-V4-**Flash** `config.json` has `kv_lora_rank: null` — legitimate: Flash
  uses q-LoRA + o-LoRA + GQA (`q_lora_rank=1024`, `o_lora_rank=1024`,
  `num_key_value_heads=1`), **no MLA KV compression**, so `kv_lora_rank` is null.
- SGLang parses `model_type=deepseek_v4` via
  `_DeepseekV4ConfigAlias(DeepseekV3Config)` in
  `sglang/srt/utils/hf_transformers/common.py` (V4 "reuses the V3 config schema").
  The field `kv_lora_rank: int = 512` is declared on transformers'
  `DeepseekV3Config` (`transformers/models/deepseek_v3/configuration_deepseek_v3.py:81`).
- Under **transformers 5.x**, `PreTrainedConfig` is a `huggingface_hub` `@strict`
  dataclass, so the validator built from that annotation rejects `None`. Under
  transformers <5 (non-strict) the null was silently tolerated — which is why
  external guidance says "pin transformers==4.57.1". **We can't**: our image pins
  transformers 5.8.0 for the Gemma-4 assistant drafter.

**Not the file you'd expect:** SGLang's own `sglang/srt/configs/deepseek_v4.py`
(`DeepSeekV4Config`) is a *runtime/modeling* class used by `models/deepseek_v4.py`
and the DSA indexer/compressor — it is **NOT** the class that parses `config.json`
for `model_type=deepseek_v4`. Editing it has no effect on this crash.

**Workaround (ours):** a launch-time source patch — since 2026-07-16 (commit
`a760767`) its own gated file `roles/k8s_dgx/files/sglang_patches/p56_deepseek_v3_kv_lora.py`
(previously inline in `sglang_launch.sh`, block `PATCH_DSV4_KVLORA_EOF`) —
widens the annotation to `int | None` **before** `python3 -m sglang.launch_server`
(so `@strict` rebuilds a Union validator that accepts None; pyc is
timestamp-invalidated so the edit takes):
```
transformers/models/deepseek_v3/configuration_deepseek_v3.py:
    kv_lora_rank: int = 512   →   kv_lora_rank: int | None = 512
```
Safe widening — DeepSeek-V3 / V3.2 / Kimi-K2 (which share this config) always
supply an int; only V4-Flash's null now additionally passes. Verified in a
throwaway pod: `field.type` becomes `int | None` and `cls(kv_lora_rank=None)`
constructs cleanly. Log line on success:
`Patched DeepseekV3Config: kv_lora_rank now int|None (...)`.

**Upstream status:** not fixed — `kv_lora_rank: int = 512` is still present on
`main` and in v0.5.12.post1. No targeted upstream issue filed; related to the
general "V4 reuses V3 schema" design and the open Flash-serving tracker
(#25165 / #23743). Remove the launch patch once upstream types it Optional or
registers transformers' native `DeepseekV4Config` (which has no `kv_lora_rank`).

---

## 2. compressed-tensors module-naming mismatch — `wqkv_a` vs `fused_wqa_wkv` (OPEN)

**Symptom** — after the config parse succeeds, model build fails:
```
ValueError: Unable to find matching target for model.layers.0.self_attn.wqkv_a
            in the compressed-tensors config.
```
(`compressed_tensors/utils.py:find_matched_target` → no scheme, empty `ignore`.)

**Root cause:** third-party NVFP4 repackages (RedHatAI, kylesayrs, canada-quant)
are **compressed-tensors** format whose attention target regex is
`re:.*attn.*(fused_wqa_wkv|wq_b|wo_a|wo_b)$`. SGLang's V4 implementation names the
fused proj **`wqkv_a`** (`models/deepseek_v4.py:265`). 3 of 4 names match
(`wq_b`, `wo_a`, `wo_b`); only `wqkv_a` ≠ `fused_wqa_wkv` → no match → hard error.
SGLang's V4 loader expects deepseek-ai/sgl checkpoint naming
(`remap_weight_name_to_dpsk_hf_format`); the third-party repos follow
vLLM/transformers naming instead.

**Affected checkpoints (all hit this identically):**
- `RedHatAI/DeepSeek-V4-Flash-NVFP4-FP8` — target `fused_wqa_wkv`
- `kylesayrs/DeepSeek-V4-Flash-NVFP4-FP8-BLOCK` — target `fused_wqa_wkv`
- `canada-quant/DeepSeek-V4-Flash-NVFP4-FP8-MTP` — target `fused_wqa_wkv` (+ `compressor.fused_wkv_wgate`)
- None ship `hf_quant_config.json` → **none are ModelOpt format** (which would
  bypass the compressed-tensors matcher entirely).

**Upstream status:** the tracking issue
[#23724](https://github.com/sgl-project/sglang/issues/23724) is **closed since
2026-07-17**, resolved by merged PR #25763 ("Support DeepSeek-V4 Wint4Abf16 and
Win4Afp8", in v0.5.16). That PR ships a **native SGLang** Wint4A16/Wint4AFP8
quant path with its own checkpoint naming, it does NOT touch the
compressed-tensors matcher or the `wqkv_a`/`fused_wqa_wkv` mismatch itself. The
RedHatAI/kylesayrs/canada-quant checkpoints named in this section are still
unverified against current main and the mismatch described above is expected to
still reproduce. Not launch-patchable in a robust way (matcher + weight-name
remap + an NVFP4 scheme that doesn't exist yet — see §3). **Fix = use a
SGLang-native checkpoint**, not a patch (update 2026-07-28: #25763 is now a
concrete example of such a native path, though not NVFP4).

---

## 3. NVFP4 MoE / FP4 indexer for DeepSeek-V4 — not implemented (OPEN)

Even with a correctly-named NVFP4 checkpoint, SGLang's V4 path lacks NVFP4
support:
- `models/deepseek_v4.py` has exactly one quant branch: `w4afp8` (line ~1401).
  No NVFP4 / ModelOpt path for the attention projections.
- [#25820](https://github.com/sgl-project/sglang/issues/25820) **[NVIDIA] Support
  NVFP4 MoE for DeepSeek-V4** — **merged 2026-06-22 (main); im offiziellen
  v0.5.14-Release seit 2026-06-26; default `flashinfer_trtllm_routed`, B200-only.**
- [#26209](https://github.com/sgl-project/sglang/issues/26209) **Add FP4 Indexer
  for DeepSeek V4** — **merged 2026-06-02** (in `main`, merged_at
  2026-06-02T07:14:39Z; enthalten im offiziellen v0.5.13-Release seit 2026-06-13).
  Der DSA/Index-Attention-Pfad ist damit upstream verfügbar.
- Roadmap: [#23602](https://github.com/sgl-project/sglang/issues/23602) DeepSeek V4
  Roadmap.
- **Update 2026-07-28:** [PR #30272](https://github.com/sgl-project/sglang/pull/30272)
  "Implement SM120 DeepSeek V4 flashinfer_mxfp4 moe runner backend + TP2" ist im
  v0.5.16-Release (2026-07-25) enthalten und explizit SM120-adressiert (der
  Cluster-`major==12`-Check deckt SM121/GB10 sonst mit ab, siehe §6/§7/§8),
  **prüfenswert** als potenzieller Ersatz oder Ergänzung für den
  `flashinfer_cutlass`-Pin auf SM121, aber MXFP4 ist eine andere Quant-Familie
  als NVFP4 und die Kompatibilität zum `nvidia/DeepSeek-V4-Flash-NVFP4`-Checkpoint
  (§5) ist unverifiziert. Noch nicht getestet auf diesem Cluster.

---

## 4. NVFP4 runner instability on V4 (OPEN / partly closed)

Where NVFP4 V4 does run (e.g. Pro on B200), it is buggy:
- [#26324](https://github.com/sgl-project/sglang/issues/26324) — `flashinfer_trtllm`
  MoE runner **asserts on DeepSeek-V4-Flash NVFP4 on B200** (and corrupts
  MiniMax-M2.7-NVFP4 output). **Closed 2026-08-15 (stale-bot, unresolved)** —
  no fix commits landed, last substantive human comment still 2026-06-15
  (nvpohanh). Same pattern as #25165/#23743/#25526/#26647/#24111: the
  tracking issue is gone, the underlying assert is not fixed.
- [#25704](https://github.com/sgl-project/sglang/issues/25704) (closed) — DeepSeek-V4
  **Pro** NVFP4 on B200: non-speculative decode produces **NaN (TP=8) or garbage
  tokens** (DP+DeepEP); **only EAGLE works**.

On **SM121** specifically, our own `cutlass_moe_fp4` crashes apply on top (see
`CLAUDE.md` NVFP4 section, `TURBOQUANT.md`, `SGLANG_TP_EP_MOE_UPSTREAM_BUG.md`).

---

## 5. `nvidia/DeepSeek-V4-Pro-NVFP4` does not fit — `nvidia/DeepSeek-V4-Flash-NVFP4` needs verification (capacity, not a bug)

> **Update 2026-06-12:** Die ursprüngliche Aussage „there is no
> `nvidia/DeepSeek-V4-Flash-NVFP4`" und „the only NVIDIA ModelOpt-NVFP4 of V4
> is the Pro variant" waren **falsch**. Das HF-Repo
> `nvidia/DeepSeek-V4-Flash-NVFP4` existiert seit 2026-05-18T00:02Z (zuletzt
> geändert 2026-06-10T19:33Z, im Zuge des PR #25820 quant-config-Fix-Zyklus),
> 46 Safetensors-Shards (vs. 64 für Pro). `hf_quant_config.json` bestätigt:
> producer `modelopt`, version `dsv4-nvfp4-experts`, quant_algo
> `MIXED_PRECISION`, NVFP4 auf `layers.*.ffn.experts`, group_size 16 — gleiches
> Format wie das Pro-Checkpoint. Voraussetzung für die Nutzung: PR #25820 muss
> gemergt sein (Stand 2026-06-14: offen, ungemergt, vgl. Status-Block oben).
> Kapazitäts-Implikation: Die §5-Abschätzung „does not fit" galt für Pro (913
> GB); Flash NVFP4 (~162 B params, NVFP4 experts, ~46 Shards) ist deutlich
> kleiner und könnte bei TP=4 in 4×128 GB passen — **muss noch verifiziert
> werden**, war bisher keine betrachtete Option.

**`nvidia/DeepSeek-V4-Pro-NVFP4`** — Pro-Variante (nicht lauffähig auf diesem Cluster):
- **913 GB** of weights on disk (64 safetensors), **~910 B params**, 61 layers,
  hidden 7168, 384 routed experts.
- ModelOpt mixed precision: NVFP4 on **experts only** (`hf_quant_config.json`,
  producer `dsv4-nvfp4-experts`, `awq_block_size 16`); attention/dense/shared/
  router/embeddings stay FP8/BF16 → net ~1 byte/param.

Cluster has **4×128 GB = 512 GB** total. At TP/EP=4 that is **~228 GB/GPU**
needed vs **128 GB** available — ~1.8× over capacity, before KV cache /
activations / CUDA context. Even hypothetical pure NVFP4 (~455 GB) leaves no
runtime headroom. **Pro does not fit.**

**`nvidia/DeepSeek-V4-Flash-NVFP4`** — Flash-Variante (~162 B params, 46
safetensors-Shards): Kapazitäts-Abschätzung steht aus. Bei TP=4 und ~1
byte/param sind ~162 GB Gewichte zu erwarten — nominal in 4×128 GB passend,
aber KV-Cache / Aktivierungen / CUDA-Kontext müssen gegengerechnet werden.
Erst sinnvoll zu testen, sobald PR #25820 gemergt ist.

---

## 6. SM121 (GB10) boot chain for `DeepSeek-V4-Flash-FP8` — the walls, in order

Bringing the block-FP8 checkpoint up on 4×GB10 / SM121 is a sequence of
first-contact failures. Each is fixed by a profile flag in
`sgl-project-deepseek-v4-flash-fp8.yml` (plumbed into BOTH serving and the
`sglang_shard` Job — the runtime-kernel and weight-dtype choices must match
between shard-save and serve). The walls span config-parse → weight load →
warmup → and (verified 2026-05-31) the **first real forward request**: model
load + `Uvicorn running` is NOT the finish line — walls 6–8 only fire when the
first prompt is actually decoded. The order they surface:

| #   | Stage                  | Symptom                                                                                                                                   | Fix                                                      |
|-----|------------------------|-------------------------------------------------------------------------------------------------------------------------------------------|----------------------------------------------------------|
| 1   | config parse           | `StrictDataclassFieldValidationError: kv_lora_rank ... NoneType`                                                                          | §1 launch patch (int → int\|None)                        |
| 2   | weight load            | `ValueError: Downcasting not allowed: target fp8, loaded bf16` (on `wo_a`)                                                                | `SGLANG_OPT_FP8_WO_A_GEMM=0`                             |
| 3   | mem pool               | `RuntimeError: Not enough memory. Please try to increase --mem-fraction-static`                                                           | raise `mem_fraction_static` to **0.90**                  |
| 4   | kernel warmup          | `tf32_hc_prenorm_gemm → InternalError ... hyperconnection.hpp:56 Unsupported architecture`                                                | `SGLANG_OPT_DEEPGEMM_HC_PRENORM=0`                       |
| 5   | decode (pre-empted)    | `paged_mqa_logits` DeepGEMM kernel, same SM121 gap                                                                                        | `SGLANG_FP8_PAGED_MQA_LOGITS_TORCH=1`                    |
| —   | weight load (host RAM) | kernel **SIGKILL** (no CUDA trace) at end of load — MoE expert fusion in `process_weights_after_loading` doubles unified-memory footprint | node **swap** (see below)                                |
| 6   | first forward          | `AssertionError: assert seq_lens.shape == (batch_size,)` in `dsv4/indexer.py:fp8_paged_mqa_logits_torch`                                  | launch source-patch (squeeze trailing singleton)         |
| 7   | first forward (decode) | `topk_v2.cuh:472 CUDA error: invalid argument` — TVM jit `topk_transform_512_v2`                                                          | `SGLANG_TOPK_TRANSFORM_512_TORCH=1`                      |
| 0   | first forward          | `RuntimeError: Unsupported architecture for sparse decode fwd` (FlashMLA)                                                                 | baked FlashMLA sm_121a kernel + `.pth` hook — see **§7** |

**(2) `wo_a` fp8→bf16 — `SGLANG_OPT_FP8_WO_A_GEMM` (#25181, default-on since 0.5.12).**
The opt keeps the o-LoRA `wo_a` projection fp8 and runs an fp8 GEMM. V4-Flash-FP8
ships `wo_a` as **bf16**, so with the opt on the param is created fp8 and
`copy_with_check` (parameter.py) refuses the bf16→fp8 downcast. `=0` dequants
`wo_a` to bf16 (`deepseek_v4.py:_dequant_fp8_wo_a`), matching the checkpoint.
Same error class as [#19589](https://github.com/sgl-project/sglang/issues/19589)
(Qwen3.5-FP8). Cost is trivial — `wo_a` is low-rank (o_lora_rank=1024): ~1.44 B
params total, **+1.44 GB bf16-vs-fp8 (~0.36 GB/rank at TP=4)**.

**(3) `mem_fraction_static`.** Not a bug — a sizing constraint. The flag governs
the post-weight-load KV reserve, NOT a vLLM-style fraction-of-total: KV pool =
`post_load_free − pre_load_free × (1 − mem_fraction_static)`. Weights are ~73.5 GB
≈ 0.61 of the 121 GiB node, so the old `0.50` left a *negative* KV pool → the
"Not enough memory" error. **Higher = more KV.** 0.90 → ~28 GB KV, safe because
the profile disables CUDA-graph capture (the held-back reserve only covers
activations; MQA `num_key_value_heads=1` → tiny per-token KV).

**(4)+(5) DeepGEMM has no sm_121 kernels for two V4 families**
([DeepGEMM #317](https://github.com/deepseek-ai/DeepGEMM/issues/317),
[SGLang #23657](https://github.com/sgl-project/sglang/issues/23657)):
`tf32_hc_prenorm_gemm` (Manifold-Constrained Hyper-Connections / MHC) and
`paged_mqa_logits` (DSA Lightning-Indexer) both assert *Unsupported architecture*
on SM120/121. SGLang has fallbacks gated by env, OFF by the wrong default for us:
- `SGLANG_OPT_DEEPGEMM_HC_PRENORM=0` → MHC prenorm uses the **TileLang** path
  (`mhc.py:mhc_pre_gemm_sqrsum_tilelang`). Hits at `kernel_warmup`, then every
  forward. **Independent of `disable_deep_gemm`** (that's `SGLANG_ENABLE_JIT_DEEPGEMM`,
  the JIT-MoE switch — it does NOT cover these two kernels).
- `SGLANG_FP8_PAGED_MQA_LOGITS_TORCH=1` → indexer uses the **torch** fallback.
  Surfaces at decode, not load — pre-emptively flipped.
Both fallbacks are slower than the kernels but correct; perf-tune later (cluster
is CPU-TCP-bound, so the delta may not matter). **Both are runtime-kernel choices
→ required at inference, not just sharding.**

**Host-RAM OOM during load (the swap story).** On GB10 unified memory (CPU+GPU
share ~121 GiB), `process_weights_after_loading` builds the fused MoE expert
tensors while the per-expert sources are still resident → a transient ~2× spike
that SIGKILLs at the end of load (host kernel OOM, no CUDA trace — proving it's
host-pageable, not device-pinned). Fixed with a disk-backed node swapfile
(KEP-2400). Per-pod control via QoS under kubelet `swapBehavior=LimitedSwap`:
the `sglang_shard` Job is **Burstable** (memory request) → gets swap → absorbs
the spike; the serving Deployment is **BestEffort** (GPU-limit only) → `memory.swap.max=0`
→ never swaps. This is *why* sharded_state matters operationally: the serve load
skips the fusion, so it fits in RAM without swap (and serving must not swap —
paging weights/activations would wreck latency). Swap device: `dgx_prepare`
tag `swap`; kubelet policy: `k3sserver` gate `k3s_node_swap_enabled`.

**(6) C4-indexer torch fallback asserts 1-D `seq_lens` (forward-time).** Enabling
`SGLANG_FP8_PAGED_MQA_LOGITS_TORCH=1` (wall 5) routes the indexer to
`dsv4/indexer.py:fp8_paged_mqa_logits_torch`, but its caller `forward_c4_indexer`
unconditionally unsqueezes `c4_seq_lens` to 2-D `(batch, 1)` for the deep_gemm /
tilelang kernels — and the torch fallback `assert seq_lens.shape == (batch_size,)`.
→ `AssertionError` on the **first multi-token forward** (any prompt, not just
EAGLE; speculative is off here). sglang's own gap. **Note:** the vendored
0xSero `_patch_sglang_indexer_fallbacks()` was meant to fix exactly this but
targets the OLD module paths (`nsa.tilelang_kernel`, `compressed.indexer`) — on
v0.5.12.post1 the indexer moved to `attention/dsv4/indexer.py`, so it patches
nothing. Fix = a launch-time source patch that squeezes a trailing singleton
before the assert (`sglang_launch.sh`, marker `assert seq_lens.shape ==
(batch_size,)`; no-op when already 1-D).

**(7) C4-indexer top-k transform TVM kernel dies on sm_121 (decode).** The next
wall, immediately after (6): `forward_c4_indexer` picks `topk_transform_512_v2`
(a TVM jit kernel, `sglang/jit_kernel/...`) when `SGLANG_OPT_USE_TOPK_V2`
(default-on) and no capture buffer. On GB10 it aborts with
`topk_v2.cuh:472 CUDA error: invalid argument` at decode. sglang ships a pure-
torch fallback `topk_transform_512_pytorch_vectorized`, selected by
`SGLANG_TOPK_TRANSFORM_512_TORCH=1` — and the torch branch is checked **before**
the v2 branch, so the flag alone wins (no need to also clear `OPT_USE_TOPK_V2`).
Profile flag `topk_transform_512_torch: true` → env, plumbed serving-side
(decode-time kernel; irrelevant to the shard Job). Same fallback family as (5).

**Post-load warmup latency.** With (4)/(5)/(7) on the torch/TileLang fallbacks,
the **first** request after startup takes ~2 min (one-time JIT/compile of the
TileLang MHC-prenorm path et al. on first real forward) — the request can outlast
a 120 s client timeout while the server still completes it (200 OK). Warm
requests are normal latency. Do not mistake the first-request stall for a hang.
**Update (v0.5.14+):** [PR #27986](https://github.com/sgl-project/sglang/pull/27986)
„[dsv4] Prewarm MHC prenorm kernel at startup" ist im v0.5.14-Release (2026-06-26)
enthalten und löst diesen First-Request-JIT-Stall nativ — der Kernel wird beim
Serverstart vorcompiliert statt beim ersten Forward. Bei Verwendung eines
v0.5.14+-Images entfällt das ~2-min-Warm-up.

**On `load_format`.** Verified 2026-05-31: `load_format: auto` (in-process MoE
fusion, no pre-sharding) loaded cleanly on the BestEffort serving Deployment
**without** triggering the host-RAM SIGKILL of the swap story above — the ~2×
fusion spike fit under 121 GiB for this checkpoint at TP=4. `sharded_state` (skip
fusion at serve) remains the swap-free design, but is not strictly required here.

---

## 7. FlashMLA sparse-decode kernel for sm_121a + the `.pth` hook (wall #0)

V4's attention backend (`deepseek_v4_backend.py`) hard-imports `flash_mla` and
calls `flash_mla.flash_mla_with_kvcache(...)` with **no fallback**, and upstream
FlashMLA ships no sm_120/sm_121 sparse-decode kernel → the first forward dies
`RuntimeError: Unsupported architecture for sparse decode fwd`. We bake a
sm_121a kernel into the image (vendored from `0xSero/deepseek-v4-flash-sm120`,
retargeted `sm_120a → sm_121a`) + stock `flash_mla` for the interface, and
monkey-patch `flash_mla_with_kvcache` at interpreter startup. Three sub-walls,
each a non-obvious trap (all hit during bringup 2026-05-31):

**(a) The hook never ran — Ubuntu shadows our `sitecustomize.py`.** The kernel
ships `sitecustomize_hook.py`; dropping it at
`/usr/local/lib/python3.12/dist-packages/sitecustomize.py` looks right but Python
imports only the **first** `sitecustomize` on `sys.path`, and Ubuntu's
`/usr/lib/python3.12/sitecustomize.py` (apport) sits earlier → ours never loaded,
patch never applied, stock FlashMLA crashed. Fix = a **`.pth`** file instead:
`site.py` executes the `import` line in **every** `.pth` across all site dirs (no
"first wins"), in main **and** every spawned sglang worker. Drop
`zz_dsv4_autopatch.pth` → `import dsv4_autopatch`.

**(b) Don't call `install()` — it loads tilelang's libcudart stub too early.**
The kernel's `patch_flash_mla()` (= `install()`) also runs
`_patch_sglang_indexer_fallbacks()`, which imports `sglang…nsa.tilelang_kernel`
→ loads `tilelang/lib/libcudart_stub.so`. If that stub loads **before**
`flashinfer.comm`, flashinfer's `find_loaded_library("libcudart")` grabs the stub
(missing `cudaDeviceReset`) → `AttributeError` at import — a HARD crash, NOT caught
by sglang's `except ImportError` (unlike the benign `cuda.tile` ModuleNotFoundError
from flashinfer 0.6.12, which sglang tolerates → allreduce-unavailable fallback).
Fix = the autopatch calls **only** `_patch_flash_mla_pkg()` (installs the wrapper,
no tilelang). sglang imports tilelang itself LATER, after flashinfer, in its
natural (proven-safe) order; bootstrapping it early is not our job. (And
`_patch_sglang_indexer_fallbacks()` is a no-op here anyway — see §6 wall 6.)

**(c) The wrapper is gated, inert for non-V4.** `_make_wrapper` only takes over
when `indices is not None` AND device-cap major == 12 (GB10 reports `(12, 1)`)
AND `is_fp8_kvcache` AND `q.element_size() == 2`; otherwise it defers to stock
flash_mla. So the `.pth` is safe to ship image-wide.

Artifacts: `scripts/patches/dockerfile-dsv4-flashmla.patch` (bakes kernel +
`.pth` at build), `sglang_launch.sh` (writes `dsv4_autopatch.py` + `.pth` at
runtime — no rebuild needed), recipe knobs `FLASH_MLA_*` / `DSV4_KERNEL_*` in
`scripts/patches/sglang-0.5.12-sm121.recipe`.

---

## What we actually run: `sgl-project/DeepSeek-V4-Flash-FP8`

SGLang's own pre-converted checkpoint ([#24111](https://github.com/sgl-project/sglang/issues/24111)):
- Plain **block-wise FP8** — `quant_method=fp8`, `weight_block_size=[128,128]`,
  `scale_fmt=ue8m0`. **Not compressed-tensors**, so §2 does not apply; quant is
  auto-detected (no explicit flag), same as `zai-org/GLM-4.7-FP8`.
- Profile: `roles/k8s_dgx/model_profiles/sgl-project-deepseek-v4-flash-fp8.yml`.
  SM121 FP8 kernel choices mirror GLM-4.7-FP8: `moe_runner_backend=triton`,
  `fp8_gemm_runner_backend=cutlass`, `disable_deep_gemm=true`,
  `attention_backend=flashinfer`.
- Recommended sampling (model card / `generation_config.json`): **temperature 1.0,
  top_p 1.0** (`do_sample=true`, no top_k).
- MTP: per the SGLang DeepSeek-V4 cookbook, V4-Flash MTP runs under the **EAGLE**
  framework and requires env `SGLANG_ENABLE_SPEC_V2=1`. Presets — balanced
  (steps=1, draft=2), low-latency (steps=3, draft=4), eagle-topk=1;
  max-throughput disables MTP. Left disabled for first-contact.
- **Still requires the §1 launch patch** — same architecture, same
  `kv_lora_rank: null`.

Why the image is `0.5.12.post1` (not `.12`): the .post1 cherry-picks touch DSv4
FP8 paths directly — **#25733/#26063** (single-token-decode garbled text via
`deep_gemm` ue8m0 scale-packing — our checkpoint is exactly ue8m0 block FP8) and
**#25646/#26072** (HiSparse GSM8K accuracy 0.825→0.960). See
`SGLANG_v0.5.12.post1_VERSION_CHANGES.md`.

**Status — end-to-end verified 2026-05-31.** With walls 1–7 + the §7 FlashMLA
hook in place, the 4×GB10 deployment loads, serves, and **decodes coherent
output** (correct factual answer, `200 OK`, clean token distribution) — confirmed
by a real `/v1/completions` request at TP=4. The earlier "expect issues past
model load" warning was correct: walls 6, 7 and §7 all surfaced only on the first
forward, past `Uvicorn running`. Subsequent first-contact risk is lower but not
zero — Flash serving is still stabilizing upstream
([#25165](https://github.com/sgl-project/sglang/issues/25165) main "broke" with
V4-Flash, [#23743](https://github.com/sgl-project/sglang/issues/23743) GB200
serving tracker, [#25526](https://github.com/sgl-project/sglang/issues/25526)
HiCache piecewise CUDA graph, [#26647](https://github.com/sgl-project/sglang/issues/26647)
Mooncake HiCache hybrid cache) — features we don't enable (HiCache, MTP/EAGLE,
PD-disagg) may surface new walls if turned on. **Update 2026-07-28:** #25165,
#23743 and #25526 are all now closed by the stale-bot (unresolved, no fix
landed, see the 2026-07-28 status block); only #26647 is still genuinely open.
Stale-bot closure here does not mean the underlying gap is fixed.
**Update 2026-08-03:** #26647 has now gone the same way — auto-closed by the
stale-bot 2026-07-29 (one day after the previous check), unresolved, no fix
landed. No issue in this Flash-stabilization cluster remains genuinely open.
**Update 2026-08-07:** correction to the reference table — **#31125 is a
merged PR (2026-07-14), not an open Known Issue**: it is itself the stopgap
(disables the flaky DSV4-Flash FP4 BCG determinism test; the nondeterminism
comes from #30898's idle-rank dummy extend). No separate open tracking issue
for the underlying nondeterminism exists. Also noted, informational only: new
issue [#33636](https://github.com/sgl-project/sglang/issues/33636) "[NVIDIA]
DeepSeek V4 Perf Tracking" (opened 2026-08-05) cross-references our PR
#31480 but explicitly scopes OUT "SM120/SM121, ROCm, NPU, XPU, CPU" — no
impact on this cluster; and #32750 (PP + PD + DSpark hidden-state shape
mismatch, referenced from the #23602 roadmap thread) is unrelated to
anything tracked here. All other tracked refs re-verified unchanged
(`kv_lora_rank: int = 512` still on transformers main; #26324 still open,
idle since 2026-06-15; SGLang still at v0.5.16).
**Update 2026-08-09:** SGLang **v0.5.17 released 2026-08-08** (582 PRs since
v0.5.16) — release notes checked for every DSV4/DSA item (#31987 BCG on
dsa/deepep, #31888 Q8KV8 sparse prefill, #29016 SM90 MegaMoE DSV4 FP8,
#27657 DSV4 CP decode, #30240 DSV4 RunAI-streamer fix, #31017 top-k epsilon,
#31901 HiSparse PD transfers, #32490/#31346 dsa backend-selection/fp8 fixes):
**none touches `kv_lora_rank`, the `wqkv_a`/`fused_wqa_wkv` matcher (§2), or
SM121 NVFP4 runner routing** — all tracked walls unchanged in v0.5.17.
Source-re-confirmed: `wqkv_a` naming still present in
`sglang/srt/models/deepseek_v4.py` on `main`; `kv_lora_rank: int = 512`
still at `configuration_deepseek_v3.py:93` on transformers `main`. #33636
got one new comment (2026-08-08, a user offering 8×B200 compute for a
benchmark harness — informational, does not change its explicit SM120/SM121
scope-out). #32750 and #26324: no new activity.
**Update 2026-08-15:** no new SGLang release. **#26324 closed by the
stale-bot 2026-08-15, unresolved** (no fix commits, see the updated
reference table and §4). #33636 gained 5 comments between 2026-08-09 and
2026-08-12, all B200/MegaMoE/CP topics (W4A8 vs W4A4 validation, PR #34277,
PRs #31687/#32042/#33672 listed, shared-KV CP #32059 validation, plus a
maintainer note about low-quality LLM-generated posts) — none touches
SM120/SM121 scope, still informational only. #32750 idle since 2026-08-06.
All other tracked issues/PRs unchanged: every previously-closed stale-bot
closure re-confirmed, and merged PRs #25763/#25820/#26209/#30272/#31125
unchanged.
**Update 2026-08-21 (SGLang still v0.5.17, no new release; transformers
v5.15.1 released 2026-08-19).** `transformers` v5.15.1 (patch release,
2026-08-19) checked: `kv_lora_rank: int = 512` still present unchanged at
`configuration_deepseek_v3.py:93` on both the v5.15.1 tag and `main`;
release notes list only DFlash/MTP candidate-generator and image-processing
fixes, nothing DeepSeek-V4-related. No new SGLang release since v0.5.17
(2026-08-08).
**New, load-bearing:** [PR #34926](https://github.com/sgl-project/sglang/pull/34926)
"Clean deprecated DeepSeek V4 Environs" merged 2026-08-17 to `main` (not in
v0.5.17, not yet released) removes the `SGLANG_TOPK_TRANSFORM_512_TORCH` env
var entirely from `environ.py` and its two call sites in
`layers/attention/dsv4/indexer.py`, the flag our Wall 7 workaround
(`topk_transform_512_torch` profile key) sets. The torch fallback itself is
not gone, just re-routed: both call sites now gate on
`self.dsa_topk_backend.is_torch()`, the pre-existing CLI/server arg
`--dsa-topk-backend` (`server_args.py`, choices `sgl-kernel` / `torch` /
`flashinfer`, default `sgl-kernel`; documented since PR #31124, merged
2026-07-14). Action needed at the next image bump past this commit (or the
next SGLang release containing it): setting `SGLANG_TOPK_TRANSFORM_512_TORCH=1`
will silently do nothing (code path deleted), Wall 7 must switch to plumbing
`--dsa-topk-backend=torch` as a server arg instead of an env var. Not urgent
today: v0.5.17 predates this commit and still honors the env var; `wqkv_a`
naming (§2) is unchanged on `main`. The same PR also removed several
unrelated deprecated env vars we don't use (`SGLANG_OPT_USE_DEEPGEMM_MEGA_MOE`,
`SGLANG_OPT_FIX_MEGA_MOE_MEMORY`, `SGLANG_OPT_USE_JIT_KERNEL_FUSED_TOPK`,
`SGLANG_PREP_IN_CUDA_GRAPH`, `SGLANG_OPT_USE_FUSED_STORE_CACHE`, others);
Walls 2/4/5 env vars (`SGLANG_OPT_FP8_WO_A_GEMM`, `SGLANG_OPT_DEEPGEMM_HC_PRENORM`,
`SGLANG_FP8_PAGED_MQA_LOGITS_TORCH`) are all still present, unchanged.
#26324 (closed 2026-08-15, stale-bot) got one new human comment 2026-08-21
(user `stewtong`, replying to nvpohanh/trevor-m); issue was NOT reopened,
still closed. #33636 gained 4 comments 2026-08-12 through 2026-08-18 (PR
list additions #31687/#32042/#33672; a "Shared-KV CP validated" comment
from `stewtong` publicly flagged by maintainer `b8zhong` as likely
low-quality/LLM-generated; a 2026-08-18 note from nvpohanh on host-mem-cache
reuse and prefill chunk batching), all still B200/GB300/MegaMoE topics, no
SM120/SM121 mention, still informational only. #32750 idle since 2026-08-06.
#23602 roadmap idle since 2026-08-13. No other tracked ref changed.
**Update 2026-08-28 (SGLang v0.5.18 released 2026-08-22, contains PR #34926,
Wall 7 migration now acute; new finding on Walls 2/4/5 auto-routing since
v0.5.13).** SGLang **v0.5.18** released 2026-08-22 (tag cut
2026-08-20T14:29:24-07:00, 710 PRs since v0.5.17) is now the latest release.
Confirmed via `git merge-base --is-ancestor` on the local sglang clone that
commit `bc312d18` (PR #34926, "Clean deprecated DeepSeek V4 Environs",
merged 2026-08-17) is an ancestor of the v0.5.18 tag, and direct grep of
`environ.py` at that tag confirms `SGLANG_TOPK_TRANSFORM_512_TORCH` is gone.
**Wall 7's env-var migration is therefore acute now, not forward-looking**:
at the next image bump past v0.5.18, the profile flag must switch from
setting `SGLANG_TOPK_TRANSFORM_512_TORCH=1` to passing
`--dsa-topk-backend=torch` as a server arg (`layers/attention/dsv4/indexer.py`
now gates on `self.dsa_topk_backend.is_torch()`). Not yet urgent for the
currently-deployed image (`xomoxcc/dgx-spark-sglang:0.5.17-sm121`, which
predates this commit and still honors the env var). Walls 2/4/5 env vars
(`SGLANG_OPT_FP8_WO_A_GEMM`, `SGLANG_OPT_DEEPGEMM_HC_PRENORM`,
`SGLANG_FP8_PAGED_MQA_LOGITS_TORCH`) are confirmed present, unchanged, in
`environ.py` at the v0.5.18 tag, surviving both #34926 and PR #35060 ("Clean
up environ.py: remove dead env vars, unify deprecation handling", merged
2026-08-17, also confirmed an ancestor of v0.5.18).

**Correction to §8 (2026-06-14 conclusion on Walls 2/4/5 auto-routing).**
Direct source inspection of `server_args.py::_handle_model_specific_adjustments()`
(called unconditionally during server-arg postprocessing) shows that for
`model_arch in ["DeepseekV4ForCausalLM"]` AND `is_sm120_supported()`, SGLang
unconditionally sets `envs.SGLANG_OPT_FP8_WO_A_GEMM.set(False)` (Wall 2),
`envs.SGLANG_OPT_DEEPGEMM_HC_PRENORM.set(False)` (Wall 4), and
`envs.SGLANG_FP8_PAGED_MQA_LOGITS_TORCH.set(True)` (Wall 5).
`is_sm120_supported()` (`utils/common.py`) checks
`device_capability_majors=[12]` only, a major-only check that also matches
SM121/GB10. Git blame shows this block has been present, byte-identical
(aside from an unrelated `SGLANG_OPT_FUSE_MHC_POST_PRE` guard added
2026-08-14), since **PR #24692 itself** (commit `524ba10e`, merged
2026-06-01, in v0.5.13 and every release since, including the
currently-deployed `0.5.17-sm121` image). This means §8's 2026-06-14
statements "kein Auto-Routing für SM121 in #24692" (Wall 4) and
"Env-Var-Routing unverändert" (Wall 5) were incomplete: the auto-routing
exists and covers SM121 via the major-only check; §8's table text is left as
originally written above, this note is the correction. Practical effect: our
manual profile flags (`opt_fp8_wo_a_gemm: false`, `opt_deepgemm_hc_prenorm:
false`, `fp8_paged_mqa_logits_torch: true`) are very likely redundant since
v0.5.13 (upstream forces the identical values), not harmful since the values
match. Recommend verifying on a future test boot whether removing the manual
flags changes anything (expect no change); until verified, keep them as
belt-and-suspenders.

**New, hardware-specific: PR #34019 ("[SM12x] Default the fused MHC
post+pre path on", merged 2026-08-14) plus PR #35214 ("[DSV4] Turn on mhc
post pre fusion by default", merged 2026-08-18), both in v0.5.18.** These
default `SGLANG_OPT_FUSE_MHC_POST_PRE=True` on SM120/SM121 (guarded by
`is_set()`, so an explicit override still wins), replacing a previously
unfused fp32 SIMT cuBLAS fallback (`cutlass_80_simt_sgemm`, no tensor cores)
for the `hc_pre` GEMM with a fused TileLang kernel. PR #34019's own
benchmark, run on **2x DGX Spark (GB10/sm_121, TP=2)** with
`deepseek-ai/DeepSeek-V4-Flash-0731` plus DSPARK, measured about -7.3ms/step
from this PR alone (9.54ms to 0.86ms on the fp32 SIMT path), and combined
with an unrelated FP8 wo_a change (#34018) an overall decode step drop from
70.31ms to 63.19ms (-10.1%), with GSM8K accuracy unaffected (0.970 to 0.980
at n=200, within noise). Not bit-identical at temperature 0 (split-K
reduction changes accumulation order); set `SGLANG_OPT_FUSE_MHC_POST_PRE=0`
for bit-exact reproduction against the old default. Directly relevant to
this cluster once we move to a v0.5.18+ image; not yet tested here.

**Issue tracking, re-verified 2026-08-28:** #26324 (closed 2026-08-15,
stale-bot) had no new activity since the 2026-08-21 comment, still closed
and unresolved. #33636 gained 3 more comments 2026-08-18 through 2026-08-25
(nvpohanh: DSV4 GB300-disagg agentic-workload optimization items, plus a
list of PRs #35944/#35947/#33672), still exclusively B200/GB300 topics, no
SM120/SM121 mention, still informational only. #32750 idle since 2026-08-06.
#23602 roadmap idle since 2026-08-13. New model in v0.5.18's cookbook,
DeepSeek-V4-Pro-0813 (#34809), a Pro-variant refresh, not re-evaluated for
capacity here.

**Follow-up same day (2026-08-28, approved): Wall 7 plumbing migrated in-repo;
the replacement is semantically BROADER than the old env.** Repo changes
(uncommitted): the pod env `SGLANG_TOPK_TRANSFORM_512_TORCH` was removed from
`roles/k8s_dgx/tasks/sglang_instance.yml` (breadcrumb comment left); the same
profile key `topk_transform_512_torch` now renders `SGLANG_DSA_TOPK_BACKEND=torch`
into the shared instance ConfigMap, and `sglang_launch.sh` translates that into
`--dsa-topk-backend torch` plus `export SGLANG_DSA_FUSE_TOPK=0`. Source-verified
semantics on both tags: on v0.5.17 the old env and `dsa_topk_backend.is_torch()`
are ORed at the SAME dispatch site (`dsv4/indexer.py:808-809`), so the arg covers
the old behavior on the current image; on v0.5.18 the env is gone and
`is_torch()` (`indexer.py:811`) is the only path. Two caveats discovered during
verification: (1) the `torch` choice of `--dsa-topk-backend` requires unfused
top-k, `SGLANG_DSA_FUSE_TOPK=0` (default is 1; the fused dispatch ends in
`raise RuntimeError("Unsupported ... for SGLANG_DSA_FUSE_TOPK")` for torch,
`dsa/dsa_topk_backend.py:91/:199`), which launch.sh now sets automatically;
(2) unlike the old env, which surgically flipped only the 512-transform site
while the rest of the indexer stayed on `sgl-kernel` fused kernels, the arg
switches the WHOLE indexer top-k backend to torch (slower). v0.5.18+ offers no
narrow equivalent of the old env anymore. Operationally dormant either way: all
three DSV4 profiles set `topk_transform_512_torch: false` (JIT `topk_v1.cuh`
validated bit-identical on GB10 2026-06-20), so nothing is emitted today; the
knob is only a re-armable escape hatch, now with a perf cost attached if ever
re-enabled.

**Update 2026-09-02 (weekly audit, no new SGLang release).** SGLang remains at
v0.5.18 (2026-08-22, 710 PRs since v0.5.17), no newer tag or release exists
(`gh release list` / `gh api tags` both re-checked). Source re-confirmed at the
v0.5.18 tag: `wqkv_a` naming unchanged in `srt/models/deepseek_v4.py` (§2),
`SGLANG_TOPK_TRANSFORM_512_TORCH` confirmed absent from `main`'s `environ.py`
(only `SGLANG_DSA_FUSE_TOPK` / the `--dsa-topk-backend` machinery remain),
consistent with the 2026-08-28 Wall 7 migration note. Issue tracking: #26324
(closed 2026-08-15, stale-bot) had no new activity since the 2026-08-21
comment, still closed. #33636 gained no new comments after the 2026-08-25 one
already recorded above (one `subscribed` timeline event 2026-08-27, no
substance), still exclusively B200/GB300 topics, no SM120/SM121 mention,
still informational only. #32750 idle since 2026-08-06. #23602 roadmap idle
since 2026-08-13. #25165/#23743/#25526/#26647/#24111/#23724 all re-confirmed
still closed, no reopen activity on any. New, informational only: [PR
#35505](https://github.com/sgl-project/sglang/pull/35505) "[Deepseek-V4]
Enable shared-experts fusion on the flashinfer_mxfp4 (trtllm-gen) MoE path"
merged 2026-08-25 to `main` (not yet in any release), a follow-on to #30272
(§3, the SM120 flashinfer_mxfp4 backend already flagged as worth checking for
SM121) that fuses the shared expert into the MXFP4 MoE kernel. Not relevant
to us today (checkpoint is FP8, not MXFP4/NVFP4-MoE) and not yet released,
noted for when #30272's MXFP4 path is eventually evaluated. No wall status
changes, no FIXED_ rename warranted this cycle.

**Update 2026-09-04 (audit).** No new SGLang release; v0.5.18 (2026-08-22)
remains latest (`gh release list` re-checked). A `release/v0.5.19` branch
exists on the repo (`gh api repos/sgl-project/sglang/branches/release/v0.5.19`,
head `0bcd8223`), with a small number of cherry-picks already landing on it
(AMD/NPU DSV4 fixes, e.g. #37862/#37857), but it carries no tag and no GitHub
Release yet, so it is not a delivery vehicle. Issue tracking unchanged since
2026-09-02: #26324 still closed, no activity since the 2026-08-21 comment.
#33636 still open, no new comments since 2026-08-26 (`updatedAt` re-checked),
still exclusively B200/GB300 topics, no SM120/SM121 mention. #32750 idle
since 2026-08-06. #23602 idle since 2026-08-13.

New and directly relevant this cycle, all merged to `main` after v0.5.18 and
therefore UNRELEASED (nothing in v0.5.18 or in our deployed
`xomoxcc/dgx-spark-sglang:0.5.17-sm121` image):

- [PR #29927](https://github.com/sgl-project/sglang/pull/29927) "[SM120]
  DeepSeek-V4: DeepGEMM paged-MQA indexer + FP4 MoE + page-split", merged
  2026-09-02. Diff source-verified (`git show 19c30dff56`) to touch Walls 3,
  4 and 5 directly:
  - Wall 5 (`paged_mqa_logits` DeepGEMM SM121 gap): routes the sparse-MLA
    indexer to DeepGEMM's paged-MQA-logits kernel instead of the torch
    fallback our `fp8_paged_mqa_logits_torch: true` profile flag forces
    today. Opt-in is `SGLANG_FP8_PAGED_MQA_LOGITS_TORCH=0
    SGLANG_OPT_USE_TILELANG_INDEXER=0`.
  - Wall 4 (DeepGEMM MHC-prenorm gate): `deep_gemm_wrapper/configurer.py`'s
    exact-match `if sm_version == 120: return False` (the gate this doc has
    tracked since PR #24692, §8) is replaced by a probe of DeepGEMM's
    `m_grouped_fp8_fp4_gemm_nt_contiguous` entry point instead of a hard
    disable. `models/deepseek_v4.py` additionally now gates the
    `tf32_hc_prenorm_gemm` dispatch on `x.shape[0] >= 1024`
    (`_HC_PRENORM_DEEPGEMM_MIN_TOKENS`, prefill-sized batches only);
    `SGLANG_OPT_DEEPGEMM_HC_PRENORM` itself is unchanged and still required.
    The probed entry point ships in DeepGEMM#324 ("sm120 support"), merged
    upstream 2026-06-24, so any reasonably current vendored DeepGEMM build
    should already have it, but this is not independently confirmed for our
    image.
  - Wall 3 (NVFP4 MoE / `flashinfer_cutlass` pin): the same configurer.py
    probe plus a new `moe_runner/deep_gemm_sm120.py` "unblocks the DeepGEMM
    FP4 MoE backend" per the PR body.
  - Source-verified: every new SM120-only code path in this PR gates on
    `is_sm120_supported()` (`major==12` device-capability check,
    `srt/utils/common.py:290-294`) or `get_platform().is_sm120`
    (`runtime_context.py:1782`, confirmed to alias the same function), the
    same major-only SM120/SM121 convention this doc already relied on for
    PR #24692 (§8). SM121/GB10 applicability is therefore INFERRED from
    that convention, NOT independently tested on our hardware, and not yet
    available in any released image.
  - Split-out dependencies, both already merged and included in the same
    unreleased window: [PR #35116](https://github.com/sgl-project/sglang/pull/35116)
    "allocate the page-split buffer outside inference mode" (correctness
    fix, merged 2026-08-25), and [PR #35118](https://github.com/sgl-project/sglang/pull/35118)
    "hc-prenorm: fuse the combine step into a Triton kernel" (merged
    2026-09-01, upstream-labeled "arch-independent"; the PR body measures it
    at roughly 3% of the combined TPOT gain).
- Wall 7-adjacent (topk_transform_512 / SM121 TVM path), both merged
  2026-09-01, both unreleased: [PR #33237](https://github.com/sgl-project/sglang/pull/33237)
  adds a `flashinfer` choice to `--dsa-topk-backend` (fused top-k via
  FlashInfer, alongside the existing native/torch choices this doc already
  tracks for Wall 7). [PR #36831](https://github.com/sgl-project/sglang/pull/36831)
  is a pure rename ("drop the redundant 512 from the top-k transform
  entry-point names"; diff touches kernel/bench/test files only, confirmed
  NOT to touch `environ.py`, `SGLANG_DSA_TOPK_BACKEND` or
  `SGLANG_DSA_FUSE_TOPK`), no functional effect on this doc's tracking.

No wall status change and no FIXED_ rename warranted this cycle since none
of the above is released yet. This is nonetheless the first upstream
movement since the 2026-08-28 Wall 7 migration that touches Walls 3 through
5 directly rather than being B200/GB300-only or informational-adjacent;
worth a dedicated re-check against the actual merged diffs once a v0.5.19
(or later) tag exists, before the next `sglang-0.5.1x-sm121.recipe` bump.

**Update 2026-09-10 (audit).** SGLang v0.5.19 released 2026-09-05 (tag
confirmed via `gh release list`), now the latest release, 786 PRs since
v0.5.18. Checked whether it closes out the 2026-09-04 finding (#29927 for
Walls 3/4/5): it does not. Ancestry check (`gh api compare`) shows the
v0.5.19 tag is diverged from, not descended from, the #29927 merge commit
(`19c30dff56`); the tag is bit-identical to the `release/v0.5.19` branch head
(`0bcd8223`, unchanged since 09-04), which was cut from `main` before
#29927/#35118 merged and only carries its own separate AMD/NPU cherry-picks
since. Confirmed directly on the tag's file tree: `moe_runner/deep_gemm_sm120.py`
(new file from #29927) is absent, and `deep_gemm_wrapper/configurer.py`
still has the old exact-match `if sm_version == 120: return False` gate, not
#29927's probe. Official release notes list #35116 and #33237 as included
but do not mention #29927, #35118, or #36831. So of last cycle's four
tracked PRs: #35116 (page-split buffer correctness fix) and #33237
(flashinfer topk-backend choice, Wall 7-adjacent) ARE in v0.5.19; #29927
(Walls 3, 4, 5) and #35118 (hc-prenorm Triton fusion) are still unreleased;
#36831 remains a no-op rename either way. No wall status change, no FIXED_
rename this cycle either.

Issue tracking: #26324, #33636, #32750, #23602 all unchanged since 09-04
(no new comments, `updated_at` unchanged on all four).

New commits to `main` since 09-04 touching `models/deepseek_v4.py`,
`layers/attention/dsv4/`, or `deep_gemm_wrapper/`: all AMD/HIP/NPU-specific
(unified-KV pool/SWA ring accounting, oproj fp8 fusion, CP v1 deprecation)
or explicitly SM100/103 (PR #30805, "Integrate TRT-LLM DSv4 Attention for
SM100/103", merged 2026-09-10). Diff-checked the two non-AMD-tagged commits
for `is_sm120`/`sm121` references: none found. Nothing new relevant to our
SM121 walls.

Incidental, not a tracked wall: v0.5.19 release notes (PR #35919) say the
default MoE runner backend for FP4 DeepSeek-V4 checkpoints on SM90/SM100/SM120
flips to `flashinfer_mxfp4` when `--moe-runner-backend` is left at `auto`;
an explicit backend setting is preserved per the same note. Our model
profiles always pin explicitly, so no effect today, worth remembering only
if a future profile drops the explicit pin.

Next check: once #29927/#35118 land in an actual tag (watch for v0.5.20 or
a `release/v0.5.19` point release), re-verify Walls 3, 4, 5 against the
released `deep_gemm_wrapper/configurer.py` and `moe_runner/deep_gemm_sm120.py`
directly, same method as this cycle.

**Update 2026-09-16 (audit).** No new SGLang release; v0.5.19 (2026-09-05)
remains latest, no `v0.5.19.postN` and no `v0.5.20` (`gh release list` and
`gh api tags` re-checked). Re-verified the 09-10 finding directly: `gh api
compare/v0.5.19...19c30dff56` (the #29927 merge commit) still reports
`"status":"diverged"` (ahead_by 84, behind_by 14), so v0.5.19 still does not
contain #29927 (Walls 3, 4, 5), unchanged. Issue tracking: #26324, #32750 and
#23602 unchanged since 09-10. #33636's `updated_at` moved to 2026-09-14, but
the comments API shows no new comment text since the 08-26 comment already
on record; still open, still exclusively B200/GB300, no SM120/121 mention,
still informational only.

New, unreleased, directly relevant, both merged after the 09-10 cutoff:

- [PR #36655](https://github.com/sgl-project/sglang/pull/36655) "[SM120] Use
  exact query-head widths for DeepSeek-V4 sparse MLA decode", merged
  2026-09-10T22:40Z. Touches `flash_mla_sm120.py` and `models/deepseek_v4.py`.
  Source-verified (PR body) as the decode-side companion to #29927's
  native-width prefill path: removes the 16-to-64-head padding for TP4
  decode on SM120 via `get_platform().is_sm120`, the same major==12
  convention this doc already relies on for SM121 applicability (§8).
  Layered directly on top of #29927 (still unreleased, per above), therefore
  also unreleased. Does not reopen Wall 0 (native/redundant since v0.5.13,
  §8), but should be re-checked alongside #29927/#35118 once a release lands.
- [PR #33672](https://github.com/sgl-project/sglang/pull/33672) "[DSV4]
  Support raw-index output in TopK v2", merged 2026-09-11. Touches
  `layers/attention/dsv4/indexer.py` and the `topk_v2` kernel, same file
  family as Wall 7. Source-verified as a buffer-ordering correctness fix
  gated on `--enable-return-indexer-topk` (sparse-prefill raw-index output),
  not an SM120/121 auto-routing change. Does not affect our
  `--dsa-topk-backend torch` / `SGLANG_DSA_FUSE_TOPK=0` Wall 7 plumbing.
  Informational only.

Everything else new on `models/deepseek_v4.py`, `layers/attention/dsv4/` and
`deep_gemm_wrapper/` since 09-10 is AMD/gfx950-specific (#37413) or part of a
"DSV4.1" refactor/naming wave (#38954, #38947, plus further title hits like
#39674/#39648/#39646/#39370), none SM120/SM121-tagged, none touching
`deep_gemm_wrapper/`. Noted only as a heads-up: if DeepSeek-V4.1 becomes a
real HF release, this doc's walls (framed around V4/V4-Flash) need a fresh
audit, not an assumed carry-over.

Local-patch cross-check (commit `958c98d`, the 0.5.19 patch re-anchor):
neither `p11_weight_utils_dsv4_fastsafetensors.py` nor
`p56_deepseek_v3_kv_lora.py` were touched, and neither appears in
`RETIRED_PATCHES.md`. `p30_dsa_torch_backend.py`'s changes are a pure API
re-anchor (arch predicate moved to `get_platform().is_sm100`, the
`DSATopKBackend` constructor call replaced by `.resolve(model_runner)` per
upstream #36313, already in v0.5.19), functionally unchanged; the Wall 7
mechanism stands as documented.

No wall status change, no FIXED_ rename this cycle. Next check: same as
09-10 (watch for #29927/#35118 landing in a tag), now also watching #36655
land alongside them.

**Update 2026-09-22 (audit): SGLang v0.5.20 released 2026-09-18;
#29927/#35118/#36655/#33672 confirmed contained; GB10/SM121 explicitly
excluded from the new DeepGEMM enablement until unreleased PR #39482.**

SGLang **v0.5.20** released 2026-09-18T22:41Z (`gh release list` / `gh api
releases/tags/v0.5.20`), now the latest release, 713 PRs since v0.5.19.
Release-notes headline "DeepSeek-V4 on RTX PRO 6000": "On SM120 the
sparse-MLA indexer now runs on DeepGEMM's paged-MQA kernel and the DeepGEMM
FP4 MoE backend is enabled, replacing the torch fallback that was the only
working path" (#29927).

Containment checked via `gh api compare/v0.5.20...<merge-sha>` (all four
report `"status":"behind"`, i.e. contained):
- #29927 (merge `19c30dff56`) - behind_by 643, contained.
- #35118 (merge `b68702be99`) - behind_by 726, contained.
- #36655 (merge `d076eec427`) - behind_by 226, contained.
- #33672 (merge `335f6aab27`) - behind_by 185, contained.

File-tree-verified on the v0.5.20 tag (git objects,
`/home/thiess/pythondev_workspace/sglang`, `upstream` remote fetched):
`python/sglang/srt/layers/moe/moe_runner/deep_gemm_sm120.py` exists (new
file from #29927). `deep_gemm_wrapper/configurer.py`'s
`_compute_enable_deep_gemm()` no longer hard-disables on SM120, the old
`if sm_version == 120: return False` (v0.5.19) is now a probe of
`deep_gemm.m_grouped_fp8_fp4_gemm_nt_contiguous` (v0.5.20), matching the
doc's 2026-09-04 note.

**Correction to the 2026-09-04 block (this specific gate does NOT follow
the major==12 convention as claimed, GB10/SM121 was excluded from it as
shipped in v0.5.20).** The 09-04 entry states every new SM120-only code
path in #29927 "gates on `is_sm120_supported()` (`major==12`
device-capability check)... the same major-only SM120/SM121 convention this
doc already relied on... SM121/GB10 applicability is therefore INFERRED
from that convention." That is true for `moe_runner/deep_gemm_sm120.py` and
`attention/dsv4/metadata.py` (both literally
`_IS_SM120 = is_sm120_supported()`, confirmed unchanged at the v0.5.20
tag), but it is NOT true for the specific `deep_gemm_wrapper/configurer.py`
gate quoted in the same paragraph: that gate checks the raw
`sm_version == 120` (from `get_device_sm() = major*10+minor`, i.e. exactly
120, RTX PRO 6000 class), not `is_sm120_supported()`'s major-only `[12]`
set. For GB10 (`get_device_sm() == 121`), this exact-match check is False
and the probe branch is skipped entirely, same as it always was
pre-#29927 (v0.5.19's blanket `return False` was also exact-match-120, so
it likewise never fired for SM121; `ENABLE_JIT_DEEPGEMM` on GB10 has always
been controlled purely by `SGLANG_ENABLE_JIT_DEEPGEMM`, unaffected by this
line, in both v0.5.19 and v0.5.20). Practical effect: the v0.5.20
release-notes claim "DeepSeek-V4 on RTX PRO 6000 ... SM120 ... enabled" is
accurate to its own scope (RTX PRO 6000 = exact SM120) but does **not**
extend to GB10/SM121 as shipped, confirmed directly by a
same-day-after-release fix, see next paragraph.

**New, unreleased, directly closes the gap above:**
[PR #39482](https://github.com/sgl-project/sglang/pull/39482) "[Bugfix]
Include SM121 in DeepGEMM packed-scale selection", merged
2026-09-18T13:57Z+08:00 (i.e., the same UTC day as the v0.5.20 release cut,
but `git log v0.5.20..upstream/main` confirms it is NOT an ancestor of the
v0.5.20 tag, unreleased). Diff (`git show 3d1b9e7549a3`) touches exactly
the two lines flagged above in `deep_gemm_wrapper/configurer.py`:
`if sm_version == 120` -> `if sm_version in (120, 121)`, and
`get_device_sm() == 120` -> `get_device_sm() in (120, 121)` (the
`DEEPGEMM_SCALE_UE8M0` computation). New comment: "SM120/SM121 support
(including GB10) landed in DeepGEMM#324". This is upstream's own explicit
acknowledgment that v0.5.20 shipped #29927 without actually covering GB10
for this gate, and is the PR to watch for the actual Wall 3/4 enablement on
our hardware. Not yet in any tagged release.

**Wall-by-wall reading for a hypothetical 0.5.20-sm121 image (unreleased
#39482 not included):**
- **Wall 3** (NVFP4 MoE / `flashinfer_cutlass` pin): `moe_runner/deep_gemm_sm120.py`
  gates on `is_sm120_supported()` (major==12, includes SM121), so the new
  DeepGEMM FP4 MoE runner module IS importable/selectable on GB10 in
  v0.5.20, but whether it is actually reachable at runtime for our
  NVFP4-MoE checkpoints depends on MoE-runner-backend selection logic not
  traced this cycle; unrelated to our currently-deployed FP8 checkpoint
  (`sgl-project/DeepSeek-V4-Flash-FP8`) either way, so no action needed on
  the deployed model.
- **Wall 4** (`SGLANG_OPT_DEEPGEMM_HC_PRENORM`): the env-flag auto-default
  for SM12x is unchanged (`arg_groups/model_hook.py::handle_model_specific_adjustments`,
  formerly `server_args.py`, still sets `False` for `get_platform().is_sm120`
  == `is_sm120_supported()`, major==12, includes SM121), but as of
  v0.5.20 this is now guarded by
  `if not envs.SGLANG_OPT_DEEPGEMM_HC_PRENORM.is_set(): ...set(False)`, no
  longer unconditional (v0.5.19 set it unconditionally, no `is_set()`
  guard; the guard was added sometime between the two tags, exact commit
  not pinpointed this cycle). Our profile flag `opt_deepgemm_hc_prenorm:
  false` still matches the default, so behavior is unchanged either way.
  Separately, the actual `tf32_hc_prenorm_gemm` kernel dispatch
  (`models/deepseek_v4.py:2273`) is gated purely on
  `envs.SGLANG_OPT_DEEPGEMM_HC_PRENORM.get() and x.shape[0] >= 1024`,
  calling `deep_gemm.tf32_hc_prenorm_gemm` directly with no additional
  `ENABLE_JIT_DEEPGEMM`/probe check at that call site, so even once
  #39482 lands, flipping this flag to `true` on GB10 depends on the
  DeepGEMM package itself actually shipping a working SM121
  `tf32_hc_prenorm_gemm` kernel, not verified this cycle.
- **Wall 5** (`SGLANG_FP8_PAGED_MQA_LOGITS_TORCH`): same `is_set()`-guarded
  auto-default (`True` for SM12x major==12, unchanged), our profile flag
  matches. #29927's DeepGEMM paged-MQA-logits path for the indexer is the
  upstream replacement route (`SGLANG_FP8_PAGED_MQA_LOGITS_TORCH=0
  SGLANG_OPT_USE_TILELANG_INDEXER=0` per the 2026-09-04 note), untested on
  GB10, and per the Wall-3/4 finding above, likely also blocked by the same
  `sm_version==120` exact-match gate family until #39482 lands (not
  independently re-checked for this specific path this cycle; worth
  re-verifying once #39482 is released).
- **Wall 2** (`SGLANG_OPT_FP8_WO_A_GEMM`) and the newer
  **`SGLANG_OPT_USE_TOPK_V2`** flag: still set **unconditionally** (no
  `is_set()` guard) to `False` for SM12x major==12 in
  `handle_model_specific_adjustments`, any manual profile override of
  these two specific flags would currently be silently overwritten by
  SGLang itself; not an issue for us today since our profile values already
  match (`opt_fp8_wo_a_gemm: false`), just worth knowing if a future recipe
  ever wants to force it back on.
- **Wall 7** (`--dsa-topk-backend` / `SGLANG_DSA_FUSE_TOPK`): unchanged in
  v0.5.20 vs the 09-16 finding; `SGLANG_TOPK_TRANSFORM_512_TORCH` remains
  absent (removed by #34926, already known); the arg + `SGLANG_DSA_FUSE_TOPK`
  plumbing our launch.sh already migrated to is present unchanged.

**Recommendation for the pending 0.5.20-sm121 build:** the existing manual
profile flags (`opt_fp8_wo_a_gemm: false`, `opt_deepgemm_hc_prenorm:
false`, `fp8_paged_mqa_logits_torch: true`, `mem_fraction_static: "0.90"`)
remain correct/required for a v0.5.20 image and should NOT be removed or
changed for this bump, nothing in #29927/#35118/#36655/#33672 actually
flips a working code path on for GB10 yet, since the specific enablement
gate (`deep_gemm_wrapper/configurer.py`) still excludes SM121 as shipped.
No wall is newly resolved by v0.5.20 on this cluster.

**Issue tracking, re-verified 2026-09-22.** #26324: still closed (stale-bot,
unresolved), no activity since 2026-08-21. #33636: `updated_at` moved to
2026-09-14 but comments API shows no new comment text since 2026-08-25 (8
total, unchanged), still open, still exclusively B200/GB300/MegaMoE topics,
no SM120/SM121 mention. #32750: idle since 2026-08-06. #23602: idle since
2026-08-13.

**DSV4.1 wave, now a real HF release, still no SGLang release support.**
`deepseek-ai/DeepSeek-V4.1-Flash` (HF, createdAt 2026-09-10,
`model_type: deepseek_v41`, pipeline_tag `image-text-to-text`, i.e. this
Flash variant is now multimodal, unlike V4-Flash) exists, and
`nvidia/DeepSeek-V4.1-Flash-NVFP4` (HF, createdAt 2026-09-16) exists too.
SGLang's own cookbook page (`docs/cookbook/autoregressive/DeepSeek/DeepSeek-V4_1.mdx`,
present at the v0.5.20 tag) states explicitly: "DeepSeek-V4.1 Flash support
has **not shipped in an SGLang release yet**", pointing instead at a
preview Docker image `lmsysorg/sglang:dev-dsv41` (and `dev-dsv41-mi35x` for
ROCm), hardware cells offered are GB300 / H200 / B200 / B300 / MI350X
only, **no SM120/SM121/GB10/consumer-Blackwell cell**. No
`deepseek_v41`/`DeepseekV41` code exists anywhere in the v0.5.20 tag
outside docs (`git grep -il` across the whole tree). `transformers`
(checked at `main` and v5.17.0) also has no `deepseek_v41` model directory
yet (`deepseek_v2`/`v3`/`v32`/`v4` only). Confirms the 2026-09-16 heads-up:
this is real now, needs its own audit once SGLang ships actual support, not
assumed to carry over V4/V4-Flash's walls. `git log v0.5.20..upstream/main`
for `models/deepseek_v4.py` also shows active DSV4.1 model/runtime
integration landing post-release (`a6cf05817f` "dsv4.1: remaining model and
runtime integration" #38798, `35b7589e1a` "dsv4.1: candidate indexer
library" #39671, `7fac84b639` "[DSV4.1] Reduce mHC, metadata and
small-batch router overhead" #39704, `d1acbe0746` "[DSV4.1] Big fused wo_a
quant" #39957), all unreleased, not source-inspected this cycle beyond the
commit titles.

No wall status change, no FIXED_ rename this cycle. Next check: watch for
#39482 landing in a tag (the actual SM121 unblock for Walls 3/4/5's
DeepGEMM path), and for SGLang shipping real DeepSeek-V4.1 support in a
release.

**Update 2026-09-25 (audit): no new SGLang release; deployed image bumped to
0.5.20-sm121; new local patch p67 (unvalidated, gated off);
`_sm120_deep_gemm_apis_available()` refactor identified as PR #38792
(unreleased).**

SGLang **v0.5.20** (released 2026-09-18) remains the latest release
(`gh release list` / `gh api tags` re-checked, no `v0.5.20.postN`, no
`v0.5.21`). [PR #39482](https://github.com/sgl-project/sglang/pull/39482)
(the SM121 DeepGEMM packed-scale fix flagged 09-22) is confirmed still NOT
an ancestor of the v0.5.20 tag (`git merge-base --is-ancestor 3d1b9e7549a3
v0.5.20` -> not ancestor), unreleased, unchanged from last cycle.

**Local deploy, not an upstream change:** commit `3214ed2`
(2026-09-22T17:40:11+02:00, "Update default SGlang image version") bumped
`default_sglang_image` in `roles/k8s_dgx/defaults/main/sglang.yml` from
`xomoxcc/dgx-spark-sglang:0.5.19-sm121` to
`xomoxcc/dgx-spark-sglang:0.5.20-sm121`, recipe
`scripts/patches/sglang-0.5.20-sm121.recipe`. Per the 09-22 entry, v0.5.20
contains #29927/#35118/#36655/#33672 but excludes GB10/SM121 from the new
DeepGEMM enablement (`deep_gemm_wrapper/configurer.py`'s gate stays
exact-match `sm_version == 120`) until unreleased #39482 lands; model
profile flags are unchanged for this bump, no wall newly resolved, matching
the 09-22 recommendation.

**New local patch, not an upstream change:** `p67_deepgemm_sm121_gate.py`
(`roles/k8s_dgx/files/sglang_patches/`, commits `e69a490` + `882f914`,
2026-09-22) backports #39482 onto the v0.5.20-shaped `configurer.py`
(`sm_version == 120` -> `sm_version in (120, 121)` for both the capability
probe gate and the `DEEPGEMM_SCALE_UE8M0` selection), gated OFF by default
via `SGLANG_OPT_DEEPGEMM_SM121_PACKED_SCALE=1`. This is the local lever for
Walls 3/4/5 on GB10 until #39482 ships upstream; still unvalidated whether
the DeepGEMM package in our image actually ships working SM121 kernels from
DeepGEMM#324. No model profile enables it yet.

**`_sm120_deep_gemm_apis_available()` refactor identified:**
[PR #38792](https://github.com/sgl-project/sglang/pull/38792) "[PP][DeepSeek
V4] Overlap communication and optimize SM120 prefill", merged
2026-09-19T04:18:04Z. The 09-22 entry (via p67's docstring) noted this
refactor existed on `upstream/main` but did not name the PR.
Source-verified (`git show 5e9342d16f`): the diff replaces the single-API
probe (`from deep_gemm import m_grouped_fp8_fp4_gemm_nt_contiguous`, i.e.
#39482's own shape) with a new `_sm120_deep_gemm_apis_available()` helper
probing three DeepGEMM entry points (`fp8_einsum`,
`m_grouped_fp8_fp4_gemm_nt_contiguous`, `transform_sf_into_required_layout`)
so #38792 is layered on top of #39482 on `main`, both unreleased
(`git merge-base --is-ancestor 5e9342d16f v0.5.20` -> not ancestor; merge
timestamp is also after the v0.5.20 tag cut of 2026-09-18T22:41:33Z). #38792
additionally touches `deepseek_v4_backend.py`, `deepseek_v4.py`,
`environ.py`, PP scheduler/mem-pool files for SM120 prefill/communication
overlap, a PP feature, not one of our tracked walls, not evaluated further
this cycle (cluster does not use PP for this model).

**Other new commits since 09-22 on `models/deepseek_v4.py` /
`layers/attention/dsv4/` / `deep_gemm_wrapper/`:**
[PR #40672](https://github.com/sgl-project/sglang/pull/40672) "[Fix] Decide
the MoE padded-row bound from the layer scatter mode" (merged 2026-09-23,
`deepseek_v4.py` only) is a DP/TP-MoE-gather padding correctness fix, no
SM120/SM121 reference, not relevant.
[PR #40357](https://github.com/sgl-project/sglang/pull/40357) "[MM] Keep
scheduler padding in packed token arrays" (merged 2026-09-22, 2-line touch
to `deepseek_v4.py`) is multimodal padding, not relevant.
[PR #40637](https://github.com/sgl-project/sglang/pull/40637) "[Fix] Handle
chunked paged MQA metadata in DSV4.1 eager forwards" and
[PR #40352](https://github.com/sgl-project/sglang/pull/40352) "[DSv4.1]
Score prefill consumer index layers on candidate blocks with DeepGEMM"
(both merged 2026-09-22) are DeepSeek-**V4.1**-only (new
`candidate_indexer*`/`dense_prefill_indexer.py` files), outside this doc's
V4/V4-Flash scope. None of the four change wall status.

**Issue tracking, re-verified 2026-09-25.** #26324: still closed (stale-bot),
no activity since 2026-08-21, unchanged. #33636: `updated_at` unchanged at
2026-09-14, comments still 8, no new text since 2026-08-25, still open,
still B200/GB300/MegaMoE-only, no SM120/SM121 mention. #32750: idle since
2026-08-06, unchanged. #23602: idle since 2026-08-13, unchanged.

**DSV4.1, further unreleased integration progress, still no SGLang release
support.** Unlike 09-22 (when no `deepseek_v41` code existed anywhere in the
v0.5.20 tag outside docs), `upstream/main` now carries real DSV4.1 model
code: `python/sglang/srt/configs/deepseek_v41.py`,
`python/sglang/srt/models/deepseek_v41_vit.py`,
`python/sglang/srt/multimodal/deepseek_v41_image_processing.py` and
`.../processors/deepseek_v41.py` (confirmed absent at the v0.5.20 tag via
`git ls-tree`). The cookbook page
(`docs/cookbook/autoregressive/DeepSeek/DeepSeek-V4_1.mdx`) is unchanged in
substance: still states DeepSeek-V4.1 Flash "has not shipped in an SGLang
release yet", still points at the preview `lmsysorg/sglang:dev-dsv41` image,
still lists only GB300/H200/B200/B300/MI350X cells, no SM120/SM121/GB10
cell. `transformers` (checked `main`) still has no `deepseek_v41` model
directory. No change to this doc's scope or conclusion; noted for the next
cycle since the code footprint is growing.

No wall status change, no FIXED_ rename this cycle. Next check: same as
09-22 (watch for #39482 landing in a tag), now also watching whether
#38792's PP-overlap work or the DSV4.1 code footprint reaches a tagged
release.

**Update 2026-09-28 (audit): no movement. SGLang still v0.5.20; #39482/#38792
still unreleased; new commits since 09-25 are DSV4.1-only, none touch
tracked walls.**

SGLang **v0.5.20** (released 2026-09-18) remains the latest release (`gh
release list` re-checked, no `v0.5.20.postN`, no `v0.5.21`). PR #39482 and PR
#38792 (both flagged 09-22/09-25) confirmed still NOT ancestors of the
v0.5.20 tag (`git merge-base --is-ancestor`), still unreleased. Current
upstream/main `deep_gemm_wrapper/configurer.py` re-verified unchanged from
the 09-25 description: gate reads `if sm_version in (120, 121) and not
_sm120_deep_gemm_apis_available(): ...`.

Zero commits since 09-25 touch `deep_gemm_wrapper/`,
`moe_runner/deep_gemm_sm120.py`, `dsv4/candidate_indexer_deep_gemm.py`, or
`dsv4/metadata.py` (the Wall 3/4/5 lever files). Three new commits touch
adjacent files, all DSV4.1-only kernel-reorg work: #41243 (2026-09-26,
kernel-organization refactor, incidental `deepseek_v4.py` file moves),
#41125 (2026-09-26, "[DSv4.1] Move the low-ratio index top-k"), #41291
(2026-09-25, "[DSv4.1] Move the ratio-1/2 index top-k ops", removes a
DSV4.1-specific `deep_gemm_fp4_paged_mqa_logits()` helper from
`dsv4/indexer.py`, unrelated to our Wall 7 `--dsa-topk-backend`/
`SGLANG_DSA_FUSE_TOPK` plumbing). None change wall status. `wqkv_a` naming
(section 2) unchanged on current main.

Issue tracking, re-verified 2026-09-28: #26324 still closed, no activity
since 2026-08-21. #33636 still open, no new comments since 2026-08-25 (8
total), updated_at unchanged at 2026-09-14. #32750 idle since 2026-08-06.
#23602 idle since 2026-08-13.

DSV4.1: cookbook page unchanged in substance, still "not shipped in an
SGLang release yet", no SM120/SM121/GB10 cell. Code footprint keeps growing
(31 files now grep-match deepseek_v41/DeepseekV41 vs 4 named files at
09-25), still informational only.

Local: no changes since 09-25 to p56, p67, RETIRED_PATCHES.md, or
default_sglang_image (still xomoxcc/dgx-spark-sglang:0.5.20-sm121).

No wall status change, no FIXED_ rename this cycle. Next check: same as
09-25 (watch for #39482/#38792 landing in a tag, and DSV4.1 reaching a
tagged release).

**Update 2026-10-03 (audit): SGLang v0.5.21 released 2026-10-02 (tag commit e00930c5489053f26d86b179cee0d087f846acbb); it contains PR #39482 and PR #38792, so the exact-match `sm_version == 120` DeepGEMM gate is fixed in a release.**

`gh release list` shows v0.5.21 (published 2026-10-02T01:09:04Z) as Latest. `git merge-base --is-ancestor` confirms 3d1b9e7549a3 (#39482, commit date 2026-09-17) and 5e9342d16f (#38792, 2026-09-19) are both ancestors of v0.5.21. `deep_gemm_wrapper/configurer.py` at v0.5.21 reads `if sm_version in (120, 121) and not _sm120_deep_gemm_apis_available(): return False` (helper probes `fp8_einsum`, `m_grouped_fp8_fp4_gemm_nt_contiguous`, `transform_sf_into_required_layout`) and `DEEPGEMM_SCALE_UE8M0 = ENABLE_JIT_DEEPGEMM and (get_platform().is_sm100 or get_device_sm() in (120, 121))`. The SM121 exclusion is gone natively; whether DeepGEMM actually enables on GB10 still depends on the image's DeepGEMM build exposing the three probed APIs (not yet verified on hardware).

Local patch p67 (`SGLANG_OPT_DEEPGEMM_SM121_PACKED_SCALE`): replayed both anchors against v0.5.21 `configurer.py`. OLD_PROBE_GATE (`if sm_version == 120:` with the DeepGEMM#324 comment) no longer exists and OLD_SCALE_UE8M0 (`get_device_sm() == 120`) no longer exists either (already `in (120, 121)`), so p67 would report ANCHOR-DRIFT, write nothing and never raise. It is redundant on v0.5.21; it remains relevant only for the currently deployed 0.5.20-sm121 image (default_sglang_image unchanged, no local changes to p56, p67, RETIRED_PATCHES.md since 09-25).

Commits since 09-28 touching `deep_gemm_wrapper/` or `moe_runner/deep_gemm_sm120.py`: none. `deepseek_v4.py`/dsv4 changes on upstream/main (HEAD 7be5e3473cdbb2d2ffab252eed4b6b4bcfabdf0d, 2026-10-03) are DSV4.1/AMD/refactor only (#41251 Hopper paths and Blackwell prefill selection, #42128 SWA page size, #41163 c4 indexer pool, #39313 MegaMoE shared-expert fusion, #41308 gfx950); none change SM120/SM121 wall status beyond the gate above.

DSV4.1: now present in a tagged release (v0.5.21 notes list DeepSeek-V4.1 Flash, #40352 and the dsv4.1 kernel/model PRs; cookbook page DeepSeek-V4_1 listed). No SM120/SM121/GB10 support for V4.1 confirmed (cookbook cell not verified).

Issue tracking, 2026-10-03: #26324 closed (2026-08-21), #33636 open (updated 2026-09-14), #32750 open idle since 2026-08-06, #23602 open idle since 2026-08-13; #39482 and #38792 closed. Correction to the 09-28 entry: #39482's commit date is 2026-09-17 (GitHub close 2026-09-18).

Next check: a 0.5.21-sm121 image and recipe do not exist yet; the image bump and the p67 retirement are pending user decisions. Once an image exists, verify DeepGEMM enables on GB10 without p67. No FIXED_ rename yet (needs hardware confirmation).

**Update 2026-10-11 (audit): no upstream release since v0.5.21; the 0.5.21-sm121 image is now deployed, p67 is absorbed (no-op), sgl-deep-gemm 0.2.0 ships SM120 kernels; V4.1 still has no GB10 support.**

SGLang v0.5.21 (2026-10-02) is still the latest release (`gh release list`, no new tag); upstream/main is at ad8b1e1f85 (785 commits past v0.5.21). `deep_gemm_wrapper/configurer.py` on main is unchanged vs v0.5.21 (`sm_version in (120, 121)` and `get_device_sm() in (120, 121)`), `moe_runner/deep_gemm_sm120.py` has no commits since v0.5.21, and the V4.1 commits on main (#43228, #42808, #41603, #42273, #41659) are SM100/SM90/AMD paths with no SM12x content. The DeepSeek-V4.1 cookbook page lists only GB300, H200, B200, B300 and MI355X (preview image `lmsysorg/sglang:dev-dsv41`) and contains no SM120/SM121/GB10 mention, so V4.1 on GB10 remains unsupported upstream. Issues: #26324 closed, #33636 open (updated 2026-10-04), #23602 open (idle since 2026-08-13), #32750 now closed (2026-10-06).

Local: the cluster now runs `xomoxcc/dgx-spark-sglang:0.5.21-sm121` (commit 5a42af7, recipe `scripts/patches/sglang-0.5.21-sm121.recipe`). This supersedes the 10-03 statements that no 0.5.21 image exists and that p67 matters for 0.5.20-sm121. p67 now carries an "ABSORBED in v0.5.21" header and self-gates off via `not target_contains(TARGET, "_sm120_deep_gemm_apis_available")`, so it is a no-op on the current image (still active only for older pinned instances with the env gate on; it is not listed in RETIRED_PATCHES.md). `sgl-deep-gemm==0.2.0` in the 0.5.21 image ships the SM120 kernel family (paged/ragged fp8 and fp4 MQA logits, tf32 hc_prenorm, fp8/fp4 and bf16 GEMM) with `arch_major == 12` dispatch, and the July `Unsupported architecture` hard-block is gone (kernel-level verification on 0.5.21-sm121, 2026-10-08; on GLM-5.3-Flash the DeepGEMM paged and ragged MQA logits match a float32 reference at ~1e-7 relative error). This has not been exercised on the DSV4 path, since both DSV4 profiles keep `disable_deep_gemm: true` and `fp8_paged_mqa_logits_torch: true`, so the walls in section 6 are unchanged and no FIXED_ rename is warranted. Related DSA patches (GlmMoeDsa, not DSV4 walls): p68 (TileLang DSA v1 tile 32/1/128 for GB10's 101376 B shared-memory limit) and p69 (IndexerKPool paged logits via the torch/triton fallback). On main but not in v0.5.21: #32779 (CUDA fused Triton sparse-MLA prefill for DSA on SM120/SM90, merged 2026-10-06) and #42957 (TileLang FP8 KV on CUDA, 2026-10-11).

Next check: decide on retiring p67 into RETIRED_PATCHES.md and on a DeepGEMM-enabled DSV4 trial on the 0.5.21-sm121 image; watch for the next SGLang tag after v0.5.21.

Later 2026-10-11: p67 retired (file deleted, env/default wiring removed, entry in `roles/k8s_dgx/files/sglang_patches/RETIRED_PATCHES.md`). The DeepGEMM-enabled DSV4 trial is still open.

**Update 2026-10-12 (audit): no new SGLang tag; main now pins flashinfer 0.7.1; no DSV4/SM12x movement.**

SGLang v0.5.21 (2026-10-02) is still the latest release; upstream/main is at f006fb49ec (851 commits past v0.5.21, 66 since the 10-11 check). `python/pyproject.toml` on main now pins `flashinfer_python[cu13]==0.7.1` (#43494, merged 2026-10-12), `sgl-deep-gemm==0.2.2` and `transformers==5.19.0`; the v0.5.21 tag still pins 0.6.18 / 0.2.0 / 5.12.1. Zero commits since 10-11 touch `deep_gemm_wrapper/`, `moe_runner/deep_gemm_sm120.py` or `attention/dsv4/`; the DSV4 commits on main (#43228, #42808, #43077, #40323) are DSV4.1 MegaMoE/PD/AMD work with no SM12x content, and #43688 (transformers 5.18/5.19 adaptation) does not touch the V4 config path. Issues: #33636 open (updated 2026-10-04, no new comments since 2026-08-25), #23602 open (idle since 2026-08-13), #32750, #26324, #39482 and #38792 closed. #32779 and #42957 are still on main only, not in a tag.

Local: p67 is retired (d2ae7b2, RETIRED_PATCHES.md), superseding the 10-11 remark that it is not listed there. No wall status change, no FIXED_ rename. Next check: next SGLang tag after v0.5.21 (it will carry the flashinfer 0.7.1 pin).

**Update 2026-11-10 (audit): SGLang v0.5.22 (2026-10-19) and v0.5.23 (2026-11-02) released; no SM12x wall change in either tag; main carries SM120/SM121 DeepGEMM indexer fixes for DSV4.1.**

`gh release list` shows v0.5.23 (2026-11-02T22:14Z, tag commit 297e54fd10) as Latest, after v0.5.22 (2026-10-19, baa9befa38). Pins from `python/pyproject.toml`: v0.5.22 flashinfer 0.7.1 / sgl-deep-gemm 0.2.2 / transformers 5.19.0; v0.5.23 flashinfer 0.7.2 / sgl-deep-gemm 0.2.3.post2 / transformers 5.19.0; main (170804b40a, 2026-11-10) flashinfer 0.7.3 / sgl-deep-gemm 0.2.7 / transformers 5.19.0. `deep_gemm_wrapper/configurer.py` is unchanged at both tags vs v0.5.21 (`sm_version in (120, 121)`), and the release notes list DSV4.1 for H200, MI35x and GB300 only, with no GB10/SM121 content. The 10-12 remark that the next tag carries the flashinfer 0.7.1 pin is now fulfilled by v0.5.22. #32779 (e446698cdd) is contained in v0.5.22 and v0.5.23 (verified via merge-base).

On main only, not in v0.5.23: #48399 (27bc1a0615, 2026-11-06) restricted the DSV4.1 DeepGEMM indexer to strict SM100 and caused an 8x prefill regression on GB10 (#49173, reported on 4x DGX Spark TP4 DSV4.1-Flash, closed 2026-11-09); #49182 (5e6995dd09, 2026-11-08) restores SM120/SM121 for the dense indexer via `is_sm100_or_newer()` (the sparse candidate-table path stays SM100-only because DeepGEMM's paged sparse kernels assert arch == 10); #47658 (4fc912142f, 2026-11-09) honours `SGLANG_OPT_DEEPGEMM_HC_PRENORM` on SM120 for the V4.1 mhc projection; #48424 (1d062d2476) moves to the sgl-deep-gemm 0.2.6 MQA-logits API with an arch-12 dispatch. Net: the regression was introduced and fixed on main between the tags, so v0.5.22/v0.5.23 are unaffected; the 0.5.21-sm121 image is unaffected.

Issues: #23602 closed 2026-10-13 (stale bot; the Upstream references row still says open), #33636 open (updated 2026-10-04), #32750 closed 2026-10-06. No new SM121/GB10 issue or PR updated since 10-12. No wall status change, no FIXED_ rename. Next check: whether #49182 and #48424 land in v0.5.24, and a DeepGEMM-enabled DSV4 trial on a 0.5.22+ image (both profiles still run `disable_deep_gemm: true`).

---

## Upstream references

| Ref           | Title                                                                                                 | State                                                                                                                                                                                                                                                                     |
|---------------|-------------------------------------------------------------------------------------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| PR #23882     | DeepSeek-V4 day-0 support (`DeepseekV4ForCausalLM`)                                                   | merged (v0.5.12)                                                                                                                                                                                                                                                          |
| PR #24692     | feat: SM120 (Blackwell Desktop) support for DeepSeek-V4 inference                                     | **merged 2026-06-01; im offiziellen v0.5.13-GitHub-Release seit 2026-06-13**                                                                                                                                                                                              |
| #23602        | DeepSeek V4 Roadmap                                                                                   | open                                                                                                                                                                                                                                                                      |
| #23724        | Support DeepSeek-V4 Compressed-tensor W4A16                                                           | **closed 2026-07-17**, resolved by merged PR #25763, a native Wint4A16/Wint4AFP8 path, NOT a fix of the `wqkv_a`/`fused_wqa_wkv` matcher in §2 (see 2026-07-28 update)                                                                                                    |
| PR #25763     | [Feature] Support DeepSeek-V4 Wint4Abf16 and Win4Afp8                                                 | **merged 2026-07-16T21:24Z; in v0.5.16 (2026-07-25)**; closes #23724 tracking issue, not the §2 matcher bug                                                                                                                                                               |
| #25820        | [NVIDIA] Support NVFP4 MoE for DeepSeek-V4                                                            | **merged 2026-06-22 (main); im offiziellen v0.5.14-Release seit 2026-06-26; default `flashinfer_trtllm_routed`, B200-only, kein SM120/121-Test; `APPLY_DSV4_NVFP4_PR25820=1` bei v0.5.14+-Image obsolet**                                                                 |
| #26209        | Add FP4 Indexer for DeepSeek V4                                                                       | **merged 2026-06-02; im offiziellen v0.5.13-GitHub-Release seit 2026-06-13**                                                                                                                                                                                              |
| PR #30272     | Implement SM120 DeepSeek V4 flashinfer_mxfp4 moe runner backend + TP2                                 | **merged, in v0.5.16 (2026-07-25)**; SM120-specific, prüfenswert für SM121/GB10, see §3, not yet tested here                                                                                                                                                              |
| #26324        | flashinfer_trtllm MoE runner asserts on DeepSeek-V4-Flash NVFP4 (B200)                                | **closed 2026-08-15 (stale-bot, unresolved)**, no fix landed (last substantive comment 2026-06-15, nvpohanh)                                                                                                                                                              |
| #25704        | V4-Pro NVFP4 B200: NaN/garbage except EAGLE                                                           | closed                                                                                                                                                                                                                                                                    |
| #25165        | main branch broke with deepseek v4 flash deployment                                                   | **closed (stale, unresolved)** — auto-closed by stale-bot 2026-07-13, no fix landed                                                                                                                                                                                       |
| #23743        | [Tracking] DeepSeek V4 Flash GB200 serving fixes                                                      | **closed (stale, unresolved)** — auto-closed by stale-bot 2026-07-14, no fix landed                                                                                                                                                                                       |
| #25526        | DSv4 Flash + HiCache breakable piecewise CUDA graph                                                   | **closed 2026-07-17 (stale-bot, unresolved)**, no fix landed                                                                                                                                                                                                              |
| #26647        | Mooncake HiCache fails with DeepSeek-V4-Flash hybrid cache                                            | **closed 2026-07-29 (stale-bot, unresolved)**, no fix landed                                                                                                                                                                                                              |
| #24111        | About pre-converted FP8 checkpoints (sgl-project/DeepSeek-V4-Flash-FP8)                               | **closed 2026-07-11 (stale-bot, unresolved)**, no fix landed                                                                                                                                                                                                              |
| PR #31125     | Disable flaky DSV4-Flash FP4 BCG determinism test (nondeterminism from #30898 idle-rank dummy extend) | **merged 2026-07-14** — this PR IS the stopgap (disables the test); the underlying temp-0 nondeterminism under DP attention + breakable prefill CUDA graph has NO open tracking issue and is not fixed (corrected 2026-08-07: was mislabeled here as an open Known Issue) |
| DeepGEMM #317 | DeepSeek-V4 on SM120: `tf32_hc_prenorm_gemm` + `paged_mqa_logits` kernels missing                     | **closed 2026-04-30 (declined, no SM120 HW; community PR #318 open)**                                                                                                                                                                                                     |
| #23657        | DSv4 compressed attention: no SM120 fallback for Lightning Indexer                                    | **closed 2026-06-16**                                                                                                                                                                                                                                                     |
| #25181        | `SGLANG_OPT_FP8_WO_A_GEMM` default-on                                                                 | merged (v0.5.12)                                                                                                                                                                                                                                                          |
| #19589        | Qwen3.5 FP8 "Downcasting not allowed" (same error class as §6/2)                                      | **closed 2026-05-02**                                                                                                                                                                                                                                                     |

## 8. PR #24692 — SM120-Support für DeepSeek-V4-Inference (update 2026-06-11, re-verifiziert 2026-06-14)

**PR:** [#24692](https://github.com/sgl-project/sglang/pull/24692) „feat: SM120
(Blackwell Desktop) support for DeepSeek-V4 inference" — merged 2026-06-01,
enthalten im offiziellen v0.5.13-GitHub-Release (2026-06-13). Cluster läuft noch auf
v0.5.12.post1-basierten Images; dieser Abschnitt dokumentiert, was sich mit
einem v0.5.13-Image ändert.

**Scope des PR:**
- `flash_mla_sm120.py` — nativer SM120-FlashMLA-Triton-Kernel, aktiviert via
  `_is_sm120=True` in `deepseek_v4_backend.py` (Check: `major==12`, gilt also
  auch für SM121/GB10).
- `fp8_paged_mqa_logits_torch_sm120` in `indexer.py` — SM120-Pfad für den
  FP8-Paged-MQA-Indexer; handhabt den `seq_lens`-Squeeze intern.
- MXFP4 MoE Triton-Fallback für SM120 (neu im V4-Modellpfad).

**Auswirkung auf unsere Walls (Stand v0.5.12.post1 → v0.5.13):**

| Wall | Titel                                       | Status auf v0.5.12.post1                        | Status ab v0.5.13                                                                                                                                                                       |
|------|---------------------------------------------|-------------------------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| 0    | FlashMLA sparse-decode (SM121)              | 0xSero-vendored Kernel + `.pth`-Hook nötig (§7) | **Redundant** — nativ via `deepseek_v4_backend.py` (`_is_sm120=True`); Hook kann entfernt werden                                                                                        |
| 1    | `kv_lora_rank=None` strict-dataclass crash  | Launch-Patch nötig (`int → int\|None`)          | **Weiterhin nötig** — `kv_lora_rank: int = 512` in `configuration_deepseek_v3.py` unverändert; `_DeepseekV4ConfigAlias` in v0.5.13 überschreibt das Feld nicht                          |
| 2    | `wqkv_a` vs `fused_wqa_wkv` target mismatch | Nicht patchbar (§2)                             | **Unverändert** — kein Fix in #24692                                                                                                                                                    |
| 3    | NVFP4 MoE / FP4-Indexer                     | FP4-Indexer merged (#26209), MoE-Pfad offen     | **Gelöst in v0.5.14** — #26209 im v0.5.13-Release (2026-06-13); NVFP4 MoE (#25820) im v0.5.14-Release (2026-06-26); `flashinfer_cutlass`-Pin für SM121 weiterhin nötig                  |
| 4    | DeepGEMM MHC-prenorm (SM121)                | `SGLANG_OPT_DEEPGEMM_HC_PRENORM=0` nötig        | **Weiterhin nötig** — `configurer.py` gated DeepGEMM bei exakt `sm_version==120`, nicht 121; `mhc.py` prüft nur `SGLANG_OPT_DEEPGEMM_HC_PRENORM`; kein Auto-Routing für SM121 in #24692 |
| 5    | `paged_mqa_logits` DeepGEMM SM121-Lücke     | `SGLANG_FP8_PAGED_MQA_LOGITS_TORCH=1` nötig     | **Weiterhin nötig** (Env-Var-Routing unverändert)                                                                                                                                       |
| 6    | `seq_lens.shape`-Assert im torch-Fallback   | Launch-Source-Patch nötig                       | **Redundant** — `fp8_paged_mqa_logits_torch_sm120` in v0.5.13 handhabt Squeeze intern; Patch kann entfernt werden                                                                       |
| 7    | TVM `topk_transform_512_v2` auf SM121       | `SGLANG_TOPK_TRANSFORM_512_TORCH=1` nötig       | **Weiterhin nötig** — `SGLANG_TOPK_TRANSFORM_512_TORCH` hat kein SM120/121-Auto-Routing in v0.5.13, Default weiterhin False                                                             |

**Zusammenfassung für den Image-Wechsel auf v0.5.13:**
- Walls 0 und 6 werden durch #24692 nativ gelöst → die entsprechenden
  Launch-Patches und der `.pth`-Hook können beim Wechsel auf v0.5.13 entfernt
  werden.
- Walls 1, 2, 4, 5, 7 bleiben unverändert und erfordern weiterhin die
  bestehenden Workarounds.
- Der 0xSero-vendored FlashMLA-Kernel im Image (`scripts/patches/`) wird mit
  v0.5.13 redundant; der Build-Patch kann entfernt werden.

> **Update 2026-06-19 — Issue #23657 geschlossen; Wall 5 zu untersuchen.**
> SGLang-Issue #23657 („DSv4 compressed attention: no SM120 fallback for
> Lightning Indexer") wurde am **2026-06-16 geschlossen**. Da Wall 5
> (`SGLANG_FP8_PAGED_MQA_LOGITS_TORCH=1`) denselben Lightning-Indexer-Pfad
> betrifft (kein SM120-Fallback für `paged_mqa_logits`), ist es möglich, dass
> der in #23657 enthaltene Fix Wall 5 in v0.5.13 oder einem späteren Build
> nativ löst — `SGLANG_FP8_PAGED_MQA_LOGITS_TORCH=1` wäre dann nicht mehr
> nötig. **Noch nicht geprüft.** Beim nächsten Test auf v0.5.13 sollte
> Wall 5 explizit verifiziert werden: erst mit `SGLANG_FP8_PAGED_MQA_LOGITS_TORCH=1`
> (aktuelle Absicherung), dann testweise ohne, um festzustellen, ob der
> DeepGEMM-Pfad auf SM121 nun sauber läuft. Kein Änderungsbedarf am
> bestehenden Workaround, bis dies bestätigt ist.

---

## Local artifacts

- Launch patch: `roles/k8s_dgx/files/sglang_patches/p56_deepseek_v3_kv_lora.py` (DeepseekV3Config kv_lora_rank block; relocated 2026-07-16 from `sglang_launch.sh`, commit `a760767`)
- Model profile: `roles/k8s_dgx/model_profiles/sgl-project-deepseek-v4-flash-fp8.yml`
  — SM121 flags (§6): `opt_fp8_wo_a_gemm: false`, `opt_deepgemm_hc_prenorm: false`,
  `fp8_paged_mqa_logits_torch: true`, `mem_fraction_static: "0.90"`
- Env plumbing: `roles/k8s_dgx/tasks/sglang.yml` (serving ConfigMap) +
  `roles/k8s_dgx/tasks/sglang_shard.yml` (shard Job) — both read the same profile flags
- Node swap: device `roles/dgx_prepare/tasks/swap.yml` (tag `swap`, gate `k3s_node_swap_enabled`,
  size `dgx_swap_size`); kubelet policy `roles/k3sserver/templates/etc_rancher_k3s_kubelet-config.yaml.j2`
  (`failSwapOn: false`, `swapBehavior: LimitedSwap`)
- Active model: `group_vars/all/main/sglang.yml` (`sglang_model`)
- Image: `xomoxcc/dgx-spark-sglang:0.5.21-sm121`, recipe `scripts/patches/sglang-0.5.21-sm121.recipe` (updated 2026-10-11, commit `5a42af7`; was stale at 0.5.20)
- Release notes: see the 2026-06-29 block at the top of this file; v0.5.16 (2026-07-25) release notes summarized in the 2026-07-28 block
