# SGLang Upstream Bug: Gemma-4 NVFP4 blocked on SM121

## Status (re-verified 2026-06-11)

- **BF16 variants — WORKING** on our **`xomoxcc/dgx-spark-sglang:0.5.14-gemmadiffusion-sm121`**
  image (SGLang **v0.5.14** + SM121 sgl-kernel patches + flashinfer 0.6.13 +
  the two locally vendored Gemma-4 NVFP4 source patches). Both dense
  (`google/gemma-4-31B-it`) and MoE (`google/gemma-4-26B-A4B-it`) deploy and
  serve, with the MoE producing **180.5 tok/s @ n=8** — the fastest model on
  the cluster. Required: `attention_backend=triton` (the FlashInfer
  `global_head_dim=512` blocker was technically resolved by flashinfer 0.6.11,
  see `FLASHINFER_HEAD_DIM_512_UPSTREAM_BUG.md`; however, SGLang's
  `_handle_model_specific_adjustments` **hard-rejects `attention_backend=flashinfer`**
  for `Gemma4ForConditionalGeneration` with `AssertionError: "Gemma4 only
  supports trtllm_mha, triton, or intel_xpu attention backend"`, discovered
  2026-06-21 — making `triton` **permanently mandatory**, not a temporary bench
  choice. The triton-backend numbers above are what we bench against and remain
  the correct production config).
  Note: **SGLang v0.5.11** (released 2026-05-05) merged Gemma-4 native model
  support (PR #21952 plus follow-ups #22079, #24048, #22842 per the v0.5.11
  release notes), so the BF16 path is now in stable releases. Our current
  `0.5.14-gemmadiffusion-sm121` recipe is built on top of v0.5.14 — the older
  `0.5.11-gemma4-sm121`, dev1 / `main-gemma4-sm121` images are no longer needed
  for BF16 variants and are kept for rollback only.

- **NVFP4 variants — PARTIALLY UNBLOCKED UPSTREAM, LOCAL VALIDATION PENDING.**
  Update 2026-06-11: Two significant upstream merges change the picture:
  - **PR #25054** ("Support Gemma4 MoE NVFP4") merged to main **2026-05-21**.
    Fixes NVFP4 per-expert weight loading (`FusedMoE.make_expert_params_mapping`
    replaces the regex-based per-expert code in `gemma4_causal.py` /
    `gemma4_mm.py`) and the GEGLU activation issue in `modelopt_quant.py` — the
    weight-loading and GEGLU concerns tracked by PRs #22929/#22928 are partially
    superseded by this. **Caveat: benchmarked only on B200, NOT on SM121/GB10.**
  - **PR #26791** ("Fix Gemma4 NVFP4 MoE default attention backend") merged
    **2026-06-09**, cherry-picked to `release/v0.5.13`. Fixes a `trtllm_mha`
    default that caused MMLU 0.037 on Gemma4-26B-NVFP4 (completely broken output
    quality).
  - Both PRs are included in **v0.5.13** (tag cut 2026-06-11; GitHub Release
    published 2026-06-13). Neither is in v0.5.12.post1 (released 2026-05-26).
  - **What remains unmerged:** the SM121-specific E4M3 block-scale NaN clamp
    (#22928 part 2, #22927). These PRs remain open and stale since 2026-04-16
    (now 8 weeks), though their weight-loading/GEGLU portions are partially
    superseded by #25054. The NaN-clamp/SM121-specific portions are still
    unmerged.
  - **Bottom line:** NVFP4 Gemma4 MoE on SM121 moves from "blocked" to
    **"needs local validation once a release image carries v0.5.13"**. The
    E4M3 NaN-clamp question (#22928/#22927) remains the open risk for SM121.

  *(Superseded framing, kept for history: previously stated "STILL BLOCKED" as of
  2026-05-31. Three SM120/121 PRs #22929/#22928/#22927 were stale 45+ days, and
  #22615 was back to REVIEW_REQUIRED after a 2026-05-23 push invalidated the
  kpham-sgl approval. PRs #25054 and #26791 merged upstream since then
  partially address #22929/#22928's concerns — see above.)*

  SGLang **v0.5.12.post1** (released 2026-05-26) ships without any of the
  SM120/121-specific Gemma-4 NVFP4 fixes. **v0.5.13** (tag cut 2026-06-11;
  full GitHub Release published 2026-06-13) contains PRs #25054 and #26791.
  PR **#22615** (fp8 kv cache with KV-shared layers) remains open,
  REVIEW_REQUIRED, last updated 2026-05-23 — no change needed there.

  **Re-verified 2026-06-14:** v0.5.13 GitHub Release (published 2026-06-13)
  confirmed to contain PRs #25054 and #26791; PRs #22929/#22928/#22927/#22615
  remain **open** with no SM121 NaN-clamp fix in the release. `scitrera/dgx-spark-sglang:0.5.13`
  does not yet exist on DockerHub (latest scitrera remains `0.5.12`), so SM121
  validation with the upstream release image remains **pending**.

  **Re-verifiziert 2026-06-19:** PRs #22929/#22928/#22927 (SM121 NaN-clamp) und
  #22615 (fp8 KV cache + KV-shared layers) weiterhin **offen und unverändert**.
  Keine dieser Fixes ist in v0.5.13 (letztes Release 2026-06-13). SM121-Validierung
  von Gemma-4 NVFP4 bleibt ausstehend. Stand unverändert zu 2026-06-14.

  **Re-verifiziert 2026-06-22:** PRs #22929/#22928/#22927 (SM121 NaN-clamp)
  weiterhin **offen und unverändert**; keiner in v0.5.13 (weiterhin letztes
  Release, 2026-06-13 — kein v0.5.14). **Neu bei PR #22615 (fp8 KV cache +
  KV-shared layers): erste Aktivität seit 2026-05-23** — am **2026-06-20**
  untersucht der Autor einen CI-Fehler und Maintainer `ianliuy` wurde zur
  Review getaggt. Noch **nicht gemerged** (REVIEW_REQUIRED), aber nicht mehr
  vollständig stillgelegt. SM121-Validierung von Gemma-4 NVFP4 bleibt ausstehend.

  **Re-verified 2026-06-24:** PRs #22929/#22928/#22927 (SM121 NaN-clamp) still
  OPEN and stale since 2026-04-16; PR #22615 (fp8 KV cache + KV-shared layers)
  still OPEN (active, updated 2026-06-20). No change to status above. v0.5.13
  remains the latest release (2026-06-13). Note: v0.5.13.post1 (2026-06-15) is
  a bare git tag — no GitHub Release page, no scitrera Docker image — and can
  be dismissed as a delivery vehicle for these fixes.

  **Re-verified 2026-06-29:** v0.5.14 released 2026-06-26 (now the latest
  release). PRs #22929/#22928/#22927 (SM121 NaN-clamp) still OPEN and stale
  since 2026-04-16; PR #22615 (fp8 KV cache + KV-shared layers) still OPEN,
  REVIEW_REQUIRED. None of these fixes are in v0.5.14. SM121 validation of
  Gemma-4 NVFP4 remains pending.

  **Re-verified 2026-06-30:** v0.5.14 (released 2026-06-26) is the latest
  release; no change to open PRs — #22929/#22928/#22927 (SM121 NaN-clamp) still
  OPEN and stale since 2026-04-16; #22615 (fp8 KV cache + KV-shared layers)
  still OPEN, REVIEW_REQUIRED. BF16 image bumped to
  `xomoxcc/dgx-spark-sglang:0.5.14-gemmadiffusion-sm121` on 2026-06-29
  (flashinfer 0.6.13). SGLang's `_handle_model_specific_adjustments` allowlist
  hard-rejects `attention_backend=flashinfer` for `Gemma4ForConditionalGeneration`
  (discovered 2026-06-21) — `triton` is permanently mandatory for BF16 variants.
  SM121 validation of Gemma-4 NVFP4 remains pending.

The original v0.5.10 blockers (Transformers fallback, dual head_dim, top_k_experts
naming) are no longer relevant for our deployment because we build the image
from SGLang main, not from the v0.5.10 release — and they are also fixed in
v0.5.11 as noted above. The remaining issues are NVFP4-MoE-on-SM121-specific.

**Re-verified 2026-07-23:** SGLang **v0.5.15** (released 2026-07-10) and
**v0.5.15.post1** (2026-07-14) checked — PRs #22929/#22928/#22927 (SM121
NaN-clamp) still **OPEN**, unchanged since 2026-04-16 (now 14+ weeks); #22615
(fp8 KV cache + KV-shared layers) still **OPEN**, `REVIEW_REQUIRED`, no
activity since 2026-06-20. Neither release contains any of these fixes.

**Outstanding revert — now DONE (2026-07-23):** the two BF16 profiles
(`google-gemma-4-26b-a4b-it.yml`, `google-gemma-4-31b-it.yml`) are confirmed
back on `attention_backend: "triton"` with a breadcrumb comment explaining the
reverted 2026-06-24 flashinfer flip. Note: commit `f77a355` (2026-06-29) had
only updated this doc's text — the profile YAMLs themselves were not touched
until today. Both files are now consistent with the "triton permanently
mandatory" framing.

**New independent NVFP4 unblock path (flashinfer-side, unvalidated):**
flashinfer PR [#3744](https://github.com/flashinfer-ai/flashinfer/pull/3744)
("feat(moe): add gelu_tanh and swiglu_oai activations to b12x NVFP4 MoE for
SM12x", merged 2026-07-01, ships in **flashinfer v0.6.15**, released
2026-07-17) adds gated-GEGLU NVFP4 MoE activation support specifically for
SM12x (DGX Spark/GB10), addressing flashinfer issue #3683, plus transparent
zero-padding for non-128-aligned `intermediate_size` — the PR body names
"Gemma-4's 704" explicitly. This targets the same GEGLU-activation gap this
doc attributes to stalled SGLang PR #22928, via a different (flashinfer
kernel-level) vehicle that does not depend on #22928/#22929/#22927 merging.
**Unvalidated on our cluster:** our current Gemma image
(`xomoxcc/dgx-spark-sglang:0.5.14-gemmadiffusion-sm121`) pins flashinfer
**0.6.13** (`scripts/patches/sglang-0.5.14-gemma4-diffusion-sm121.recipe:60`),
not 0.6.15, and it's unconfirmed whether SGLang's MoE runner actually routes
Gemma-4 NVFP4 through the flashinfer path that gained this activation. See
`FLASHINFER_HEAD_DIM_512_UPSTREAM_BUG.md` (Status 2026-07-23) for the
flashinfer-side detail.

**Re-verified 2026-07-28:** SGLang **v0.5.16** (released 2026-07-25) is now
the latest release, and it contains none of the four tracked PRs. PRs
#22929/#22928/#22927 (SM121 NaN-clamp) and #22615 (fp8 KV cache) all remain
**open**, unchanged since the dates already logged above. Three of the four,
#22929, #22928 and #22927, now show `mergeable_state: dirty` on GitHub (a
rebase is needed before merge), which was not yet the case at the last
check. Source-verified on the v0.5.16 tag: `modelopt_quant.py`'s
`_SUPPORTED_ACT_STRS` is still `("silu", "relu2", "gelu")` (no `gelu_tanh`),
and the `assert not layer.moe_runner_config.is_gated` "intermediate size
required padding" guard is unchanged, so the GEGLU/gated-MoE blocker this
doc attributes to #22928 is still live upstream, byte-identical to v0.5.15.

New structural fact from v0.5.16: `--fp4-gemm-backend cutlass` is removed
(PRs #31109, #30448), so upstream NVFP4 GEMM now goes through FlashInfer
only. Separately, SGLang's own bump to flashinfer 0.6.15 was landed and then
reverted this cycle (PR #31502, reverted by #31625, performance regression),
so stock v0.5.16 stays pinned to flashinfer 0.6.14. The flashinfer-side
GEGLU unblock path (PR #3744, shipped in flashinfer 0.6.15) is therefore
still reachable only via our own custom pin, same as noted above, not via a
stock SGLang dependency bump.

**Re-verified 2026-08-09:** SGLang **v0.5.17** (released 2026-08-08) is now
the latest release; it contains none of the four tracked PRs. #22929/#22928
unchanged (idle since 2026-04-16), #22927 still `CONFLICTING`/dirty (as
already flagged 07-28), #22615 unchanged (idle since 2026-06-20).
Source-verified on the v0.5.17 tag: the Gemma4 attention-backend allowlist
in `server_args.py:5449-5465` is unchanged (`trtllm_mha`, `triton`,
`ascend`, `intel_xpu` — `triton` still mandatory), and
`_handle_model_specific_adjustments` still exists under that name (line
5039). v0.5.17's PR #25545 ("trtllm_mha for Gemma 4 **MTP draft** attention
backend", merged 2026-08-01) touches only the Frozen-KV-MTP draft backend
selection, not the main-model allowlist or the NaN-clamp/GEGLU items
tracked here — not applicable (we don't run Gemma4 with speculative
decoding). Conclusions unchanged.

**Re-verified 2026-08-15:** SGLang still **v0.5.17** (released 2026-08-08),
still the latest release; no new Gemma4-touching PR since 08-09. All four
tracked PRs unchanged: #22929 OPEN, idle since 2026-04-16; #22928 OPEN, idle
since 2026-04-16; #22927 OPEN, idle since 2026-04-16; #22615 OPEN,
REVIEW_REQUIRED, idle since 2026-06-20. The Gemma4 attention-backend
allowlist is byte-identical on the v0.5.17 tag (this doc's cited lines still
exact there) but has drifted on `main` HEAD (`0c072235`, 2026-08-15T09:20:30Z)
to lines 5601/5606 (allowlist tuple/assert) and 5163
(`_handle_model_specific_adjustments`), matching the line-number drift
documented today in `FLASHINFER_HEAD_DIM_512_UPSTREAM_BUG.md`. No change to
this doc's bottom line: `attention_backend: triton` remains permanently
mandatory for all four Gemma-4 profiles.

**Cross-reference (clearly scoped, does not change the bottom line):**
flashinfer PR [#3684](https://github.com/flashinfer-ai/flashinfer/pull/3684)
("asymmetric VO-split NVFP4 paged prefill for Gemma-4 on SM120/121") merged
to flashinfer `main` 2026-08-13T01:14:01Z as commit `8f9ad2000d`. This is a
**different subsystem** than the MoE-GEMM blockers tracked in this doc:
#22929/#22928/#22927 are about `cutlass_moe_fp4` per-expert weight loading,
GEGLU activation, and E4M3 block-scale NaN; #22615 is about the fp8
KV-shared-layer crash. PR #3684's split-KV NaN fix is for **NVFP4 KV cache**
attention, not MoE GEMM scales, do not conflate the two. It is
author-validated via vLLM only (RTX 5090/RTX PRO 6000 SM120, GB10/DGX Spark
SM121), has no SGLang-side adoption PR, and is not in any flashinfer stable
release (merged after v0.6.17). No change to this doc's bottom line: NVFP4
Gemma4 on SM121 remains blocked on #22928/#22927 for the reasons already
tracked above, independent of #3684.

**Re-verified 2026-08-21:** SGLang v0.5.17 remains the latest release, no new
release. Major tracking change: all four PRs this doc tracks were closed
unmerged between 2026-08-18 and 2026-08-19 (`mergedAt: null` for all):

- **#22929** closed 2026-08-19T00:22:24Z by `hnyls2002`, citing supersession
  by "merged PR #22079" (note: this doc's own prior tracking already
  correctly attributes the weight-loading fix to #25054, merged 2026-05-21,
  not #22079 which only added the triton PTX / kv-cache-dtype fix; the
  maintainer's PR citation looks off, but the underlying claim, that
  weight-loading is already fixed, was already tracked here as "partially
  superseded by #25054". No practical change to this doc's conclusion.)
- **#22928** closed 2026-08-18T22:03:57Z by `hnyls2002`: "Main has shifted
  SM120 MoE away from the in-kernel GEGLU/FP4-clamp approach ... cutlass_moe.py
  no longer exposes the call sites this PR patched."
- **#22927** closed 2026-08-19T01:40:06Z by `hnyls2002`: "SM120/NVFP4 support
  has been routed through a different mechanism on main - `is_sm120_supported()`
  in server_args.py picks a Marlin/CUTLASS backend that avoids the E4M3-scale
  NaN kernel entirely, rather than clamping inside the CUTLASS path."
- **#22615** closed 2026-08-19T23:09:59Z by `kpham-sgl`, no closing comment
  given.

Source-verified on upstream `main` (commit `dad6fd0f04`, 2026-08-21T18:12+08)
that the #22928/#22927 closing rationale does NOT hold for our SM120/121
(GB10/DGX Spark) configuration:

- `modelopt_quant.py:266`, `_SUPPORTED_ACT_STRS = ("silu", "relu2", "gelu")`,
  unchanged, still no `gelu_tanh`/GEGLU.
- `modelopt_quant.py:2392-2393`,
  `assert not layer.moe_runner_config.is_gated, "The intermediate size
  required padding, ..."`, unchanged text, in the CUTLASS weight-processing
  branch (the `else` of the flashinfer_trtllm check).
- The only code path that would route a gated NVFP4 MoE around that assert
  onto `flashinfer_trtllm` is `_gemma4_overrides`
  (`arg_groups/overrides.py:710-729`), and its auto-override at line 723 is
  gated `if is_sm100_supported() and server_args.moe_runner_backend ==
  "auto"`. `is_sm100_supported` (`utils/common.py`,
  `device_capability_majors=[10]`) checks datacenter Blackwell (B200) only
  and explicitly excludes SM120/121 (`is_sm120_supported` is the separate
  `majors=[12]` check). Our Gemma-4 NVFP4 profiles leave `moe_runner_backend`
  at auto (unset), so on SM121 this override never fires,
  `enable_flashinfer_trtllm_moe` stays False, and the 26B-A4B MoE's 704
  (non-128-aligned) intermediate_size still lands in the CUTLASS branch and
  still trips the `is_gated` assert. The Marlin fallback the maintainer may
  also have meant (`modelopt_quant.py` ~line 2485,
  `(8, 0) <= capability < (10, 0)`) is SM80-SM90 only, also excluding
  SM120/121. So the GEGLU/gated-MoE blocker (#22928) and the "different
  mechanism" claimed for #22927 are both still reproducible on our hardware
  as of today; only the tracking issue is gone.
- For #22615 (fp8 KV cache + KV-shared layers): `triton_backend.py:1240`,
  `if k is None and v is None:` still does `k = k_buffer[cache_loc]` with no
  dequant to `q.dtype`, the exact bug the PR's `k = k.to(q.dtype)` fix
  addressed. The underlying bug report, Issue #22277, was itself already
  auto-closed by the stale bot on 2026-07-06 (confirmed then and now), so
  both the bug report and its fix PR are now closed with zero merged code;
  the crash remains reproducible on main.

Net effect: NVFP4 Gemma-4 on SM121 remains exactly as blocked as before, but
upstream now has no open issue or PR to point at for this cluster's specific
SM120/121 gap; a fresh, hardware-specific issue may be worth filing (with the
source citations above), since a naive read of the closed PRs' rationale
would incorrectly suggest the blockers are gone.

**Correction to the flashinfer #3684 cross-reference above:** SGLang-side
adoption PRs DO exist. PR #29304 ("[Feature] NVFP4 KV cache: SM120 + SM121,
Gemma-4 VO-split, FP4 prefix-cache correctness (builds on #21954)") and PR
#29305 ("DiffusionGemma: retire Triton onto FA2 NVFP4 KV cache, stacked on
#29304 + #28054") were both filed 2026-06-25, predating flashinfer PR
#3684's 2026-08-13 merge; #29304's body says the flashinfer kernel work "is
up as flashinfer-ai/flashinfer#3684 ... this PR is the SGLang orchestration
that drives it," and a 2026-06-28 comment on #29305 says it is "parked as
draft, blocked on #29304 which is itself blocked on
flashinfer-ai/flashinfer#3684 + review." That flashinfer dependency landed
2026-08-13, but neither SGLang PR has been updated since (#29304 last
updated 2026-06-29, #29305 last updated 2026-06-28); both remain OPEN,
`mergeStateStatus` DIRTY, `reviewDecision` REVIEW_REQUIRED, and no code from
either has landed on main (`grep` for `SGLANG_FLASHINFER_VOSPLIT` and
`vo_split` across `python/sglang/srt/` returns nothing). This is still a
different subsystem (NVFP4 KV cache attention) from the MoE-GEMM blockers
(#22928/#22927/#22929) tracked in this doc, so the bottom line is unchanged:
NVFP4 Gemma4 MoE on SM121 remains blocked, `attention_backend: triton`
remains mandatory. flashinfer's latest stable release also remains v0.6.17
(no v0.6.18 stable yet, only nightly builds and rc tags up to v0.6.18rc7);
commit `8f9ad2000d` (#3684) is not in any tagged flashinfer release yet.

**Re-verified 2026-08-28:** SGLang **v0.5.18** released 2026-08-22, now the
latest release; contains no fix for the tracked SM120/121 blockers (only
unrelated CPU/Xeon Gemma4 support, PR #22498). A directory refactor moved
`server_args/arg_groups/overrides.py` to `arg_groups/overrides.py` and
collapsed `server_args/server_args.py` to a single `server_args.py`, causing
line drift but no logic change: `_SUPPORTED_ACT_STRS` unchanged
(`modelopt_quant.py:266`, no `gelu_tanh`); the `is_gated` padding assert
unchanged in substance (`modelopt_quant.py:2668`, was 2392-2393);
`_gemma4_overrides` (`arg_groups/overrides.py:1241`, was `overrides.py:710-729`)
still gates its MoE-runner override on `is_sm100_supported()` only, still
excluding SM120/121, so the CUTLASS `is_gated` assert remains reachable on
our hardware exactly as before; the Gemma4 attention-backend allowlist
(`server_args.py:6210-6223`, was 5601/5606) gained `intel_amx` (CPU-only,
irrelevant), `triton` still accepted and mandatory; the fp8 KV
triton-backend dequant gap (`#22615` target) is unchanged at
`triton_backend.py:1328` (was 1240). **flashinfer published v0.6.18rc10 today
(2026-08-28T06:15:01Z), flagged by GitHub as the current non-prerelease
"latest" release** (despite the rc-named tag), it now includes commit
`8f9ad2000d` (PR #3684, the SM120/121 VO-split NVFP4 KV-cache fix), which the
2026-08-21 entry noted was "not in any tagged flashinfer release yet."
SGLang v0.5.18's own bundled flashinfer only moved to v0.6.17 (cut
2026-08-11, before #3684 merged 2026-08-13), so stock SGLang still does not
carry it. The SGLang-side adoption PRs #29304/#29305 remain unchanged: both
OPEN, `mergeStateStatus: DIRTY`, `reviewDecision: REVIEW_REQUIRED`, no update
since 2026-06-29/2026-06-28, no `vo_split`/`SGLANG_FLASHINFER_VOSPLIT` code
on `main`. This is still the KV-cache-attention subsystem, not the MoE-GEMM
blocker tracked here, so no change to the bottom line. #22929/#22928/#22927/
#22615 remain CLOSED unmerged, unchanged since 2026-08-18/19, no reopen.
**Additional corroborating cross-reference found, not previously cited in
this doc:** Issue #30887 ("ModelOpt NVFP4 gated MoE fails to load with TP
when intermediate padding is required"), filed 2026-07-11 by a third party,
still OPEN (last activity 2026-07-17, so not new since 08-21 but newly
surfaced here), reproduces the exact `is_gated` assert on
`nvidia/Gemma-4-26B-A4B-NVFP4` at TP=2, with a comment reporting the same
assert on 8x B300 (SM100) for a different model, confirming the
padding-for-gated-activations gap is a general TP-sharding limitation, not
SM120/121-specific, and giving this cluster's blocker an existing upstream
thread (silent since 07-17, no maintainer response) rather than none. Net
effect: NVFP4 Gemma-4 on SM121 remains exactly as blocked as documented;
`attention_backend: triton` remains mandatory.

**Re-verified 2026-09-02:** No new SGLang release since v0.5.18 (2026-08-22,
still latest). flashinfer published a true stable **v0.6.18** release on
2026-08-29T02:01:33Z (tag `v0.6.18`, no rc suffix, GitHub-flagged "Latest"),
superseding the rc10 tag noted in the previous entry. Verified via `gh api
compare/v0.6.18...8f9ad2000d`: commit `8f9ad2000d` (PR #3684) is an ancestor
of the v0.6.18 tag, and the release notes explicitly credit "Gemma 4 gains
asymmetric VO-split NVFP4 paged prefill on SM120/121" with a direct link to
PR #3684. New development, not previously tracked: SGLang `main` bumped its
own bundled flashinfer pin from 0.6.17 to **0.6.18** via PR #36954 ("[Deps]
Bump FlashInfer to 0.6.18"), merged 2026-08-31, so `python/pyproject.toml`
on `main` now reads `flashinfer_python[cu13]==0.6.18`. This is still
unreleased, v0.5.18 (2026-08-22) predates the bump and remains the latest
tagged/released SGLang, so no shipped SGLang build carries flashinfer 0.6.18
yet. Source-verified on `main` (HEAD `f5866545`, 2026-09-02T13:39:54Z) that
this changes nothing for the MoE-GEMM blocker tracked here:
`modelopt_quant.py`'s `_SUPPORTED_ACT_STRS` is still `("silu", "relu2",
"gelu")` (no `gelu_tanh`, now line 295, was 266), and the `is_gated` padding
assert is unchanged in substance (now line 2751, was 2668). The repo
underwent another structural refactor since 08-28: `_gemma4_overrides`
moved from `arg_groups/overrides.py` to a new per-model file
`arg_groups/model_overrides/gemma4.py`, and its SM100-only gate was renamed
from `is_sm100_supported()` to `get_platform().is_sm100`, semantically
identical, still excluding SM120/121, so the CUTLASS `is_gated` assert
remains reachable on our hardware exactly as before. The Gemma4
attention-backend allowlist assert similarly moved from `server_args.py`
into `arg_groups/model_hook.py` (line 502), text unchanged: `trtllm_mha`,
`triton`, `ascend`, `intel_xpu`, `intel_amx`, triton still accepted and
mandatory. All four tracked PRs (#22929/#22928/#22927/#22615) remain CLOSED
unmerged with no reopen and no new comments since their 2026-08-18/19
closures. PRs #29304/#29305 remain OPEN, `mergeable_state: dirty`
(REST-verified), no update since 2026-06-29/2026-06-28. Issue #30887
remains OPEN, no new comment since 2026-07-17. No change to the bottom
line: NVFP4 Gemma-4 MoE on SM121 remains blocked, `attention_backend:
triton` remains mandatory for all four Gemma-4 profiles.

**Re-verified 2026-09-10:** SGLang **v0.5.19** released 2026-09-05 (tag
`v0.5.19`, commit `0bcd822377`, 2026-09-03T14:22:46-07:00), now the latest
release. Its own release notes confirm "FlashInfer moves to 0.6.18 (#36954)".
The bundled-flashinfer bump the 2026-09-02 entry found merged to `main` but
unreleased is now shipped in a stock SGLang release for the first time,
meaning stock SGLang now also carries flashinfer PR #3684 (SM120/121
VO-split NVFP4 KV-cache prefill) without a custom pin. This does not change
the MoE-GEMM blocker tracked here: source-verified on the v0.5.19 tag,
`modelopt_quant.py`'s `_SUPPORTED_ACT_STRS` is still `("silu", "relu2",
"gelu")` (line 281, no `gelu_tanh`), the `is_gated` padding assert is
unchanged in substance (line 2737), `_gemma4_overrides`
(`arg_groups/model_overrides/gemma4.py`) is byte-identical to the
2026-09-02 check and still gates its MoE-runner override on
`get_platform().is_sm100` only, excluding SM120/121, and the
attention-backend allowlist in `arg_groups/model_hook.py` is unchanged
(`trtllm_mha`, `triton`, `ascend`, `intel_xpu`, `intel_amx`), `triton` still
mandatory. Also confirmed byte-identical for all of the above on current
`main` (HEAD `908226fea2df`, 2026-09-10T19:22:24+08:00) versus the v0.5.19
tag. PRs #29304/#29305 (the SGLang-side orchestration for the now-released
flashinfer #3684) remain OPEN, `mergeStateStatus` DIRTY/UNKNOWN, no update
since 2026-06-29/2026-06-28, so the newly-released flashinfer dependency
still cannot be exercised through a stock SGLang code path; this is still
the KV-cache-attention subsystem, not the MoE-GEMM blocker tracked in this
doc. All four tracked PRs (#22929/#22928/#22927/#22615) remain CLOSED
unmerged, unchanged since 2026-08-18/19. Issue #30887 remains OPEN, no new
comment since 2026-07-17. No change to the bottom line: NVFP4 Gemma-4 MoE on
SM121 remains blocked, `attention_backend: triton` remains mandatory for all
four Gemma-4 profiles.

**Re-verified 2026-09-16:** No new SGLang release since v0.5.19 (2026-09-05,
still latest). All four tracked PRs (#22929/#22928/#22927/#22615) remain
CLOSED unmerged, unchanged since 2026-08-18/19. PRs #29304/#29305 remain
OPEN, draft, mergeStateStatus UNKNOWN, no update since 2026-06-28/29. New
development: issue #30887 (the ModelOpt gated-MoE intermediate-padding
assert, the same `is_gated` guard this doc's root cause item 4 tracks) was
CLOSED 2026-09-16T00:27:56Z by the `github-actions` idle-issue bot
("automatically closed due to inactivity"), labeled `inactive`. This is a
bot closure, not a fix: no new technical comment since the 2026-07-17
`kenzhangwangshu` report (a second affected model,
`stepfun-ai/Step-3.7-Flash-NVFP4`, hitting the same `is_gated` padding
assert), and `_SUPPORTED_ACT_STRS`/the padding assert in `modelopt_quant.py`
are unchanged in substance. Checked the one upstream commit that touched
`modelopt_quant.py` since the last cycle (`cebca698`, 2026-09-13, PR #39126,
"[Qwen3.8] Enable NVIDIA NVFP4 on DGX Spark ..."): its diff only adds
`ModelOptMixedPrecisionConfig` FP8_BLOCK_SCALES plumbing and VL-model
prefix-candidate handling, and does not touch `_SUPPORTED_ACT_STRS` or the
`is_gated` assert, so the Gemma4 MoE-GEMM blocker is unaffected. Also
checked `arg_groups/model_hook.py`'s latest commit (`2cbfaef`, 2026-09-16,
PR #38453): AMD/ROCm-only fp8 `wo_a` GEMM probe, does not touch the Gemma4
attention-backend allowlist. Local note: commit `780a65a` (this repo,
2026-09-16) is a comment-only cleanup of the three live Gemma-4 profiles'
image-rationale text (already correctly describing v0.5.19, `#30448`'s
removal of `cutlass_moe_fp4`, and the still-mandatory `triton` attention
backend); it reflects information already tracked in this doc and
introduces no new upstream fact. No change to the bottom line: NVFP4
Gemma-4 MoE on SM121 remains blocked, `attention_backend: triton` remains
mandatory for all four Gemma-4 profiles.

**Re-verified 2026-09-22:** SGLang **v0.5.20** released 2026-09-18 (tag commit
`94602c9c2b`, 2026-09-18T22:41:33Z), now the latest release. Release notes
(713 PRs) contain no Gemma4 NVFP4/SM120/121 MoE-GEMM fix; the only
Gemma4-touching items are AMD-specific (`aiter` attention backend for
Gemma-4, PR #38758, +17% throughput at concurrency 128 vs triton) and an
XPU RMSNorm test addition, neither relevant to SM121 NVFP4.
Source-verified on the v0.5.20 tag: `modelopt_quant.py`'s
`_SUPPORTED_ACT_STRS` is still `("silu", "relu2", "gelu")` (line 303, no
`gelu_tanh`), and the `is_gated` padding assert is unchanged in substance
(now line 2819, was 2751). `arg_groups/model_overrides/gemma4.py`'s
`_gemma4_overrides` still gates its MoE-runner override on
`get_platform().is_sm100` only, excluding SM120/121.
`arg_groups/model_hook.py`'s Gemma4 attention-backend allowlist gained
`aiter` this release (new vs v0.5.19's 5-entry tuple, confirmed by diff;
AMD-only, matches PR #38758); `triton` remains accepted and, for SM121,
still the only viable choice, since `flashinfer` is not in the allowlist
and the MoE-runner override never fires on our hardware. The CUTLASS
`is_gated` assert remains reachable exactly as before.

Tracking change: PRs **#29304** and **#29305** (the SGLang-side
NVFP4-KV-cache/VO-split orchestration PRs tracked here since 2026-08-21 as
the flashinfer #3684 adoption vehicle) were both **CLOSED unmerged
2026-09-18T01:22** by the `github-actions` idle-draft bot ("still a draft
and has not been updated in 80/81 days"), same closure mechanism that
already closed #22929/#22928/#22927/#22615 (2026-08-18/19). No reopen.
This is still the KV-cache-attention subsystem, not the MoE-GEMM blocker
tracked in this doc's root cause, so it does not change the bottom line,
but every upstream tracking vehicle this doc has ever cited for Gemma4
NVFP4 on SM121 is now closed unmerged. All four original tracked PRs
(#22929/#22928/#22927/#22615) remain CLOSED unmerged, unchanged since
2026-08-18/19. Issue #30887 remains CLOSED (bot-closed 2026-09-16), no
reopen. No change to the bottom line: NVFP4 Gemma-4 MoE on SM121 remains
blocked, `attention_backend: triton` remains mandatory for all four
Gemma-4 profiles. Local note: repo commits `63e72da`/`f3f29e1` add SGLang
v0.5.20 + SM121 build patches and update the build script accordingly (not
yet deployed, live image remains `xomoxcc/dgx-spark-sglang:0.5.19-sm121`);
none of the SM121 NVFP4 blockers tracked here are affected by that bump.

**Re-verified 2026-09-25:** No new SGLang release since v0.5.20 (2026-09-18,
still latest as of today). No commits since the 2026-09-22 check to
`arg_groups/model_overrides/gemma4.py` or `arg_groups/model_hook.py`.
`layers/quantization/modelopt_quant.py` on main HEAD gained two commits
since then (`5bb24e399`, 2026-09-24, ModelOpt MIXED_PRECISION block-FP8
dispatch/group_size refactor; `77983865d`, 2026-09-24, flashinfer_cutlass
swiglu_limit clamp plumbing), read both diffs directly: neither touches
`_SUPPORTED_ACT_STRS` (still `("silu", "relu2", "gelu")`, confirmed at line
303 on HEAD) or the `is_gated` padding assert, so the Gemma4 MoE-GEMM
blocker is unaffected. Local note: commit `3214ed2` (this repo,
2026-09-22) bumped `default_sglang_image` to
`xomoxcc/dgx-spark-sglang:0.5.20-sm121` (previous `0.5.19-sm121` kept as a
commented rollback line), so the cluster now runs 0.5.20. All four Gemma-4
profiles (`google-gemma-4-26b-a4b-it.yml`, `google-gemma-4-31b-it.yml`,
`nvidia-gemma-4-26b-a4b-nvfp4.yml`, `nvidia-gemma-4-31b-it-nvfp4.yml`)
still carry `attention_backend: "triton"`, confirmed unchanged and still
mandatory on the now-deployed 0.5.20 image. No change to the bottom line:
NVFP4 Gemma-4 MoE on SM121 remains blocked.

**Re-verified 2026-09-28:** No new SGLang release since v0.5.20 (2026-09-18,
still latest). No commits since the 2026-09-25 check to
`arg_groups/model_overrides/gemma4.py`, `arg_groups/model_hook.py`, or
`modelopt_quant.py` on upstream `main` (HEAD `df378bca42`, 2026-09-28). All
four originally tracked PRs (#22929/#22928/#22927/#22615), PRs #29304/#29305,
and Issue #30887 remain CLOSED unmerged with zero activity since their
previously logged closures, none reopened. No new issue or PR found for
`gemma4 nvfp4` or the `is_gated` padding assert opened after 2026-09-16.
Local note: `default_sglang_image` remains
`xomoxcc/dgx-spark-sglang:0.5.20-sm121`; all four Gemma-4 profiles confirmed
still on `attention_backend: "triton"`. No change to the bottom line: NVFP4
Gemma-4 MoE on SM121 remains blocked.

**Re-verified 2026-10-03:** SGLang **v0.5.21** was published 2026-10-02 (tag commit `e00930c548`, 2026-09-29), now the latest release, superseding v0.5.20. Checked at the v0.5.21 tag and on upstream `main` (HEAD `7be5e3473c`, 2026-10-03): `arg_groups/model_overrides/gemma4.py` and `arg_groups/model_hook.py` have no commits since v0.5.20. `modelopt_quant.py` took seven commits between 2026-09-19 and 2026-10-01 (including `5423a4d885`, `5bb24e399d`, `77983865d8`, `9b59fc5db5`, `56fee88e23`, `42875bcd2a`, `9cc7da2ab0`), none addressing the Gemma-4 NVFP4 SM121 NaN-clamp or the `is_gated` padding assert. `gemma4_causal.py` changed only by refactors (PP-group accessor, stage-boundary decoders in `71b04e02cf` and `7be5e3473c`, 2026-10-03) plus `40048f6e51` (DiffusionGemma). PRs #22929/#22928/#22927/#22615, #29304/#29305 and Issue #30887 remain CLOSED unmerged, none reopened. No new issue or PR found for `gemma4 nvfp4` or `gemma-4 sm121` after 2026-09-28. Local note: `default_sglang_image` remains `xomoxcc/dgx-spark-sglang:0.5.20-sm121`; all four Gemma-4 profiles remain on `attention_backend: "triton"`. No change to the bottom line: NVFP4 Gemma-4 MoE on SM121 remains blocked.

**Re-verified 2026-10-11:** No new SGLang release since v0.5.21 (2026-10-02, `e00930c548`); upstream/main HEAD `ad8b1e1f85` (2026-10-10). `arg_groups/model_overrides/gemma4.py` has no commits since 2026-10-03. `arg_groups/model_hook.py` gained `37678784c6` (PR #42019, 2026-10-10, "Enable Gemma4 FA4 on SM90 with head-dim 512 support"), which adds `fa4` to the Gemma4 accepted attention backends (now `trtllm_mha, triton, fa4, ascend, intel_xpu, intel_amx, aiter`); it targets SM90, `flashinfer` is still rejected, so the `attention_backend: triton` pin on SM121 is unchanged. `modelopt_quant.py` took `8ad24fb560` (#42927, cached exclusion-pattern compilation) and `d20bdc1443` (#39199, per-token NVFP4 MoE for ReLU2 activation); neither addresses the Gemma-4 NVFP4 SM121 NaN-clamp or the `is_gated` padding assert. `gemma4_causal.py` changed only by stage/layout refactors (`1f299e9f2f`, `7760d5264a`, `234ccba7f6`). PRs #22929/#22928/#22927/#22615, #29304/#29305 and Issue #30887 remain CLOSED unmerged, none reopened. Adjacent, not blocking: #28018 (Gemma-4-31B QAT W4A16 `gptq_marlin_repack` on SM121, CLOSED 2026-10-06) concerns a different quant path. No new issue or PR found for `gemma4 nvfp4` or `gemma-4 sm121` addressing this bug. Correction to the 10-03 local note: the deployed image is `xomoxcc/dgx-spark-sglang:0.5.21-sm121` since repo commit `5a42af7` (2026-10-03), not 0.5.20; all four Gemma-4 profiles remain on `attention_backend: "triton"` and are unchanged. `p70_modelopt_exclude_fused_layer_scope.py` (GLM MTP exclude scope) is unrelated to this bug. Bottom line unchanged: NVFP4 Gemma-4 MoE on SM121 remains blocked.

**Re-verified 2026-11-10:** SGLang v0.5.22 (2026-10-13) and v0.5.23 (2026-10-28) are new; both contain `37678784c6` (PR #42019, Gemma4 FA4), so fa4 is now in a tagged release (allowlist at v0.5.23: trtllm_mha, triton, fa4, ascend, intel_xpu, intel_amx, aiter). On `main` (HEAD `170804b40a`, 2026-11-10) the allowlist moved to `arg_groups/model_overrides/gemma4.py` and gained `fa3` via `62384de1ae` (#44399, 2026-11-06, per-layer backend: head_dim 256 layers on FA3, 512 and image-mask layers on FA4); `flashinfer` is still rejected and the SM121 default stays `triton`, so the `attention_backend: triton` pin is unchanged. Other `main` Gemma4 commits (`4b2137e5dc` #46663 prefill CUDA graphs, `1f7eef8e39` #41293, `a20aba1937` #48563) do not touch the SM121 NVFP4 NaN-clamp or the `is_gated` padding assert. PRs #22929/#22928/#22927/#22615, #29304/#29305 and Issue #30887 remain CLOSED unmerged. New open, not yet evaluated against our NVFP4 path: #48913 (Gemma 4 KV-shared layers read FP8 KV with scale 1.0) and #49491/#48919 (ModelOpt NVFP4 fused-linear global scale taken as max over parts, #49491 names Gemma 4). #26751 remains OPEN.

## Affected models

| Model                                         | Type                               | Quantization | Current status (`0.5.11-gemma4-sm121` image)                                                                                                                                                                                     |
|-----------------------------------------------|------------------------------------|--------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `google/gemma-4-26B-A4B-it`                   | MoE (128 experts, 26B/3.8B active) | BF16         | **STABLE ★** — 39.8 / 114.6 / **180.5** tok/s (n=1/4/8)                                                                                                                                                                          |
| `google/gemma-4-31B-it`                       | Dense (30.7B)                      | BF16         | **STABLE ★** — 10.6 / 36.8 / **70.6** tok/s (n=1/4/8)                                                                                                                                                                            |
| `bg-digitalservices/Gemma-4-26B-A4B-it-NVFP4` | MoE (128 experts, 26B/3.8B active) | NVFP4        | **needs local validation** — weight loading + GEGLU partially fixed by PR #25054 (merged 2026-05-21); attention backend fixed by PR #26791 (merged 2026-06-09); both in v0.5.13. SM121 NaN-clamp (#22928/#22927) still unmerged. |
| `nvidia/Gemma-4-31B-IT-NVFP4`                 | Dense (30.7B)                      | NVFP4        | **untested** — dense path skips per-expert loading; SM121 NaN/scale issues (#22928/#22927) still apply and remain unmerged.                                                                                                      |

All use `Gemma4ForConditionalGeneration` as architecture. The BF16 variants
get a native SGLang model class via PR #21952 (merged 2026-04-07). The NVFP4
variants additionally hit per-expert weight loading + cutlass FP4 kernel
issues that are SM120/121-specific.

Throughput numbers are from `TESTLOGS/sglang_nn4_tp4_ep1/gemma-4-*/` (run
2026-04-16/17, RoCE/SR-IOV, `attention_backend=triton`, `kv_cache_dtype=fp8_e4m3`,
piecewise CUDA graphs enabled). The BF16 MoE figure is the highest measured
throughput on this cluster across all tested models.

## Root cause

Gemma-4 has several architectural features that the Transformers fallback
backend in v0.5.10 does not support:

1. **Dual head dimensions** — sliding-window layers use `head_dim=256`, global
   attention layers use `global_head_dim=512`. The fallback backend creates
   RMSNorm weights uniformly with one dimension, causing shape mismatches when
   the model alternates between layer types.

2. **MoE config naming** — Gemma-4 uses `top_k_experts` instead of the standard
   `num_experts_per_tok` / `top_k` that the fallback's `_getattr_first` lookup
   expects.

3. **NVFP4 per-expert weight format** — NVFP4 checkpoints store MoE expert
   weights in unfused per-expert format, which the fallback's weight mapper
   doesn't support.

4. **GEGLU activation** — Gemma-4 MoE uses GEGLU (`gelu_tanh`), but
   `cutlass_moe_fp4()` hardcodes `silu_and_mul()`.

5. **FP4 block scale NaN** — SM120/121 specific: uint8=127 in E4M3 block scales
   triggers NaN in the CUTLASS FP4 group GEMM kernel.

Issue (1) affects **all** Gemma-4 variants (BF16 and NVFP4, dense and MoE).
Issues (2–5) affect NVFP4 variants specifically.

## Failure details

### BF16 MoE (`google/gemma-4-26B-A4B-it`) — confirmed

Crash during warmup forward at the first global-attention layer's `v_norm`:

```
gemma4/modeling_gemma4.py:1220  value_states = self.v_norm(value_states)
  → layernorm.py:207  rmsnorm(x, self.weight.data, self.variance_epsilon)
  → flashinfer/norm/rmsnorm.py:1310  kernel(...)
ValueError: Mismatched mW.shape[0] on argument #1 when calling:
  `__call__(mX: Tensor([n0, 256], bfloat16), mW: Tensor([256], bfloat16),
            mY: Tensor([n0, 256], bfloat16), M: int32, eps: float32)`,
  expected to be 256
```

The `v_norm` layer has weight `[256]` (sliding-window `head_dim`), but on a
global-attention layer the value states have dimension 512 (`global_head_dim`).
The Transformers fallback creates all attention norms with the same dimension,
not distinguishing between sliding and global layers. The native implementation
(PR #21952) has separate norm configs per layer type.

### NVFP4 MoE (`bg-digitalservices/Gemma-4-26B-A4B-it-NVFP4`) — confirmed

Three sequential failures, each uncovered after patching the previous:

1. **`Cannot determine top_k from config`** — `_getattr_first` lookup tuple
   in `transformers.py:1197` doesn't include `top_k_experts`.
   - Runtime-patched in `sglang_launch.sh` (`PATCH_TRANSFORMERS_TOPK_EOF`).
   - First patch revision had a syntax bug: inline marker comment broke the
     closing paren of `_getattr_first(...)` → `'(' was never closed` on line
     1197. Fixed by placing marker on a separate line above.

2. **`No module or parameter named 'model.language_model.layers.0.moe'`** —
   NVFP4 checkpoints store MoE expert weights in unfused per-expert format.
   The Transformers backend's weight mapper only knows the fused format.
   - **Not runtime-patchable.**
   - **Upstream fix: PR #22929** (open, 2026-04-16).

3. **(Latent) GEGLU activation mismatch** — `cutlass_moe_fp4()` hardcodes
   `silu_and_mul()`. Gemma-4 MoE uses GEGLU → garbage output even if weights
   loaded.
   - **Upstream fix: PR #22928** (open, 2026-04-16).

### Dense variants (BF16 + NVFP4)

Not separately tested. `_is_moe_model()` returns `False` → dispatches to
`TransformersMultiModalForCausalLM` (no MoEMixin) → avoids issues 1–3 above
but still hits the dual head_dim RMSNorm crash (issue 1 in the root cause list),
which is shared across all variants.

## Upstream PRs

Last `gh pr view` check: 2026-06-29. PRs #22929/#22928/#22927 still open and stale since 2026-04-16 (now 11+ weeks); #22615 still open, REVIEW_REQUIRED (last updated 2026-06-20). Their weight-loading and GEGLU concerns are partially superseded by #25054 (merged 2026-05-21); the SM121-specific NaN-clamp portions remain unmerged and are absent from both the v0.5.13 GitHub Release (published 2026-06-13) and v0.5.14 (released 2026-06-26, now the latest). The flashinfer-side blocker ([flashinfer #2959](https://github.com/flashinfer-ai/flashinfer/pull/2959)) shipped in flashinfer v0.6.10–v0.6.11 and is no longer a gating dependency.

| PR                                                         | Title                                                                    | Status     | Merged     | Relevance                                                                                                                                                                                                                                                                                                                                                                                                                                                       |
|------------------------------------------------------------|--------------------------------------------------------------------------|------------|------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| [#21952](https://github.com/sgl-project/sglang/pull/21952) | [New Model] Gemma 4                                                      | **merged** | 2026-04-07 | Native `gemma4_causal.py`, `gemma4_mm.py`, `gemma4_vision.py`, `gemma4_audio.py`. Foundation for all Gemma-4 support. Fixes the dual head_dim issue. **In our `main-gemma4-sm121` image — BF16 variants run thanks to this.**                                                                                                                                                                                                                                   |
| [#22079](https://github.com/sgl-project/sglang/pull/22079) | [nvidia] Gemma4 nvfp4 fix                                                | **merged** | 2026-04-10 | Triton attention PTX register exhaustion fix for NVFP4 on GB200/sm100a. fp8 kv cache dtype autodetection. In our image.                                                                                                                                                                                                                                                                                                                                         |
| [#22929](https://github.com/sgl-project/sglang/pull/22929) | Add NVFP4 per-expert weight loading for Gemma 4 MoE                      | **open**   | —          | Per-expert → fused weight mapping for NVFP4 MoE checkpoints. **No movement since 2026-04-16 (8 weeks).** Weight-loading concern partially superseded by #25054 (merged 2026-05-21) which uses `FusedMoE.make_expert_params_mapping`; SM121-specific NaN-clamp portion remains unmerged.                                                                                                                                                                         |
| [#22928](https://github.com/sgl-project/sglang/pull/22928) | fix(sm120): MoE GEGLU activation + FP4 block scale NaN clamp             | **open**   | —          | GEGLU activation for `cutlass_moe_fp4()` + E4M3 NaN clamp. SM120/121 critical. **No movement since 2026-04-16 (8 weeks).** GEGLU concern partially superseded by #25054 (merged 2026-05-21); SM121 NaN-clamp portion still unmerged.                                                                                                                                                                                                                            |
| [#22927](https://github.com/sgl-project/sglang/pull/22927) | fix(sm120): NVFP4 NaN from E4M3 scale overflow + 3D tensor shape crashes | **open**   | —          | Sister PR to #22928, also SM120/121-specific. Affects NVFP4 dense + MoE both. **No movement since 2026-04-16 (8 weeks).** SM121 NaN/scale-overflow portion still unmerged.                                                                                                                                                                                                                                                                                      |
| [#22615](https://github.com/sgl-project/sglang/pull/22615) | Fix fp8 KV cache crash with KV-shared layers in triton backend           | **open**   | —          | fp8 kv cache + `num_kv_shared_layers > 0` (Gemma-4 has KV-shared layers). Open since 2026-04-12. Approved by `kpham-sgl` 2026-04-22, rebased onto main 2026-04-30. **2026-05-23 push invalidated the approval** — `reviewDecision` is now `REVIEW_REQUIRED`. Re-verified 2026-06-11: still open, REVIEW_REQUIRED. **2026-06-20: first activity since 2026-05-23 — author investigating a CI failure, maintainer `ianliuy` tagged to review; still not merged.** |
| [#25054](https://github.com/sgl-project/sglang/pull/25054) | Support Gemma4 MoE NVFP4                                                 | **merged** | 2026-05-21 | Fixes NVFP4 per-expert weight loading (`FusedMoE.make_expert_params_mapping` replaces regex-based per-expert code in `gemma4_causal.py`/`gemma4_mm.py`) and the GEGLU activation issue in `modelopt_quant.py`. Partially supersedes #22929/#22928 weight-loading/GEGLU concerns. **Benchmarked only on B200, NOT on SM121/GB10.** In v0.5.13 (GitHub Release 2026-06-13).                                                                                       |
| [#26791](https://github.com/sgl-project/sglang/pull/26791) | Fix Gemma4 NVFP4 MoE default attention backend                           | **merged** | 2026-06-09 | Fixes `trtllm_mha` default causing MMLU 0.037 on Gemma4-26B-NVFP4 (completely broken output quality). Cherry-picked to `release/v0.5.13`. In v0.5.13 (GitHub Release 2026-06-13).                                                                                                                                                                                                                                                                               |
| [#22408](https://github.com/sgl-project/sglang/pull/22408) | [CI] Adding Gemma 4 to Nightly CI                                        | **merged** | 2026-04-17 | Adds Gemma-4 to nightly accuracy tests. Increases pressure on the open NVFP4 PRs to land cleanly but doesn't itself fix anything for us.                                                                                                                                                                                                                                                                                                                        |
| [#23575](https://github.com/sgl-project/sglang/pull/23575) | [AMD] fused qk gemma norm kernels                                        | **merged** | 2026-04-25 | AMD-specific perf optimization, no impact on our NVIDIA SM121 deployment.                                                                                                                                                                                                                                                                                                                                                                                       |

## What's needed to run Gemma-4 on our cluster

### BF16 variants (google/gemma-4-*) — DONE

Minimum was PR #21952 (native Gemma-4). Now in stable SGLang **v0.5.14**, baked
into our `xomoxcc/dgx-spark-sglang:0.5.14-gemmadiffusion-sm121` image (which is what
the BF16 profiles point at). Both `google/gemma-4-31B-it` (dense) and
`google/gemma-4-26B-A4B-it` (MoE) deploy and serve. The model profiles in
`roles/k8s_dgx/model_profiles/` are pinned to the working configuration:

- `attention_backend: triton` (mandatory — SGLang's `_handle_model_specific_adjustments` allowlist hard-rejects `attention_backend=flashinfer` for `Gemma4ForConditionalGeneration` (2026-06-21); see allowlist note in the Status block above and `FLASHINFER_HEAD_DIM_512_UPSTREAM_BUG.md`)
- `kv_cache_dtype: fp8_e4m3`
- `mem_fraction_static: 0.85`
- `disable_piecewise_cuda_graph: false` (BF16 is unaffected by the fp4-quantize
  dynamo bug; piecewise gives ~6.5% over fixed-BS graphs at n=8)

> **Revert DONE (2026-07-23):** The live BF16 profiles
> (`google-gemma-4-26b-a4b-it.yml`, `google-gemma-4-31b-it.yml`) had briefly
> carried `attention_backend: flashinfer` (set 2026-06-24 as a test) — this
> was **INCOMPATIBLE** with SGLang's allowlist (hard-rejects `flashinfer` for
> `Gemma4ForConditionalGeneration` with an `AssertionError`) and would have
> crashed on deploy. Note: commit `f77a355` (2026-06-29) only updated this
> doc's text; the profile YAMLs themselves were reverted to
> `attention_backend: "triton"` (with a breadcrumb comment on the reverted
> flip) on **2026-07-23**. Both BF16 profiles are confirmed back on `triton`
> as of today — safe to deploy.

To activate: set `sglang_active_model` in your inventory and run
`ansible-playbook k8s_dgx.yml --tags sglang -e sglang_enabled=true`.

### NVFP4 variants (nvidia/*, bg-digitalservices/*) — NEEDS LOCAL VALIDATION (v0.5.13)

Update 2026-06-11: The "STILL BLOCKED" framing is superseded. PRs #25054
(merged 2026-05-21) and #26791 (merged 2026-06-09) are now in v0.5.13
(GitHub Release published 2026-06-13). The remaining prerequisite list:

1. PR #21952 — native Gemma-4 model implementation (merged ✓)
2. PR #22079 — NVFP4 quantization + fp8 kv cache fixes (merged ✓)
3. PR #25054 — NVFP4 per-expert weight loading + GEGLU fix (merged ✓, v0.5.13; **B200 only, SM121 unvalidated**)
4. PR #26791 — correct attention backend default for NVFP4 MoE (merged ✓, v0.5.13)
5. PR #22928 — SM121-specific E4M3 NaN clamp (**open**, stale since 2026-04-16 — **remaining open risk**)
6. PR #22927 — SM121-specific NaN from E4M3 scale overflow + 3D shape (**open**, stale since 2026-04-16 — **remaining open risk**)
7. PR #22615 — fp8 kv cache with KV-shared layers (**open**, REVIEW_REQUIRED — may or may not apply)

Items 1–4 are now present in v0.5.13. Items 5–6 are the critical remaining
unknowns for SM121/GB10: PR #25054 was benchmarked on B200 only, and the
E4M3 block-scale NaN-clamp issue it doesn't cover may still crash or corrupt
output on SM121. Item 7 is a possible but unconfirmed risk.

*(Superseded: previously "All of the following must be present" listed #22929/#22928/#22927
as open blockers through 2026-05-31. #25054 partially addresses #22929/#22928
weight-loading/GEGLU; #22927/#22928 NaN-clamp portions still open.)*

Until they all merge upstream, we have three options:

- **Wait** — most upstream-maintenance-friendly. No work for us until merge.
- **Vendor the open PRs as our own patches** in `scripts/build_sm121_image.sh`
  (similar to the existing Gemma-4 patches). Risk: PRs are still under review
  and may change — we'd have to re-rebase if upstream tweaks them. Pre-condition:
  PR #22929 and #22928 were developed on RTX 5090 (SM120); they need
  validation on SM121/GB10, which we'd be the first to do.
- **Comment on the PRs with our SM121 test data**, push for review and merge.
  Cheapest, most likely to actually unblock things if the maintainers are
  waiting on SM121 confirmation.

## Our runtime patches (v0.5.10)

The `top_k_experts` patch in `sglang_launch.sh` (`PATCH_TRANSFORMERS_TOPK_EOF`)
remains useful — it fixes the `_getattr_first` lookup for any future model that
uses `top_k_experts` instead of `num_experts_per_tok`. However, it's insufficient
to make any Gemma-4 variant work on v0.5.10 because the dual head_dim, weight
loading, and activation function issues are not patchable at runtime.

## Relationship to other bugs

- **Independent of** the FlashInfer FP4 dynamo tracing bug
  (`FLASHINFER_CUDA_VERSION_SUBPROCESS_UPSTREAM_BUG.md`) — that affects
  piecewise CUDA graphs on all NVFP4 models, not Gemma-4 specifically.
- **Independent of** the SM121 JIT arch mismatch (`kvcache.cuh:196` illegal
  instruction) — that's in sglang's own jit_kernel, not the model loader.
- **Related to** issue #22277 (Gemma4 E4B fp8 KV cache crash) — same model
  family, overlapping root cause (KV-shared layers + fp8).

## Files

- `roles/k8s_dgx/files/sglang_launch.sh` — `top_k_experts` runtime patch
  (left in place; harmless on the main-gemma4-sm121 image, useful as a
  safety net for any other model that ever uses `top_k_experts`).
- `roles/k8s_dgx/model_profiles/google-gemma-4-26b-a4b-it.yml` — BF16 MoE
  profile, **production-ready** with the main-gemma4-sm121 image.
- `roles/k8s_dgx/model_profiles/google-gemma-4-31b-it.yml` — BF16 dense
  profile, **production-ready** with the main-gemma4-sm121 image.
- `roles/k8s_dgx/model_profiles/bg-digitalservices-gemma-4-26b-a4b-it-nvfp4.yml`
  — NVFP4 MoE profile, blocked on PRs #22929 + #22928 + #22927.
- `roles/k8s_dgx/model_profiles/nvidia-gemma-4-31b-it-nvfp4.yml` — NVFP4 dense
  profile, blocked on PRs #22928 + #22927 (per-expert loading not needed
  for dense, but FP4-on-SM121 NaN/scale issues still apply).
- `scripts/build_sm121_image.sh` — applies our locally vendored Gemma-4
  patches (`sglang-gemma4-nvfp4-expert-loading.patch`,
  `sglang-gemma4-geglu-nan-clamp.patch`, `dockerfile-gemma4-nvfp4.patch`).
  These are placeholders / early attempts at the upstream fixes — useful
  as a starting point if we decide to vendor the open PRs.
- `TESTLOGS/sglang_nn4_tp4_ep1/gemma-4-26b-a4b-it/TESTLOG_*.md` — full BF16
  MoE config sweep, 2026-04-16. Test 6 = winner (180.5 tok/s @ n=8).
- `TESTLOGS/sglang_nn4_tp4_ep1/gemma-4-31b-it/TESTLOG_*.md` — full BF16
  dense config sweep, 2026-04-17. Test 6 = winner (70.6 tok/s @ n=8).
- `TESTLOGS/sglang_nn4_tp4_ep1/gemma-4-26b-a4b-it-nvfp4/TESTLOG_*.md` —
  36-config NVFP4 MoE sweep, 2026-04-16. All blocked at `modelopt_quant.py`.
