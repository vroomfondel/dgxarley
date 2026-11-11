# Upstream Bug: SGLang can't serve modelopt-NVFP4 Llama-4 without local patches

## Status

**Open upstream, 5 local patches in place** (added 2026-07-13). A modelopt-NVFP4
Llama-4 checkpoint (`Llama4ForConditionalGeneration`, e.g.
`nvidia/Llama-4-Scout-17B-16E-Instruct-NVFP4`) does NOT load/serve on stock SGLang
0.5.15 (bug also on `main`). Five source patches are applied at container start by
`roles/k8s_dgx/files/sglang_launch.sh`, in two adjacent ungated blocks
(`PATCH_MLLAMA4_LOADER_EOF` = 3 load-blockers, `PATCH_MLLAMA4_KVSCALE_EOF` = 2
KV-scale quality). All are upstream-PR-worthy; drop them on an image that ships the
fixes (the per-anchor grep guards self-noop then).

**Ungated on purpose:** `models/llama4.py` and `models/mllama4.py` are imported
ONLY for the Llama4 arch, so patching them is inert for every other model; ungated
also covers Llama-4 checkpoints served by a local path or a non-matching repo name.
Idempotency + version-safety come from the per-anchor guards (an `ANCHOR NOT FOUND`
print on an image drift, not a silent unpatched serve).

**Profile side (NOT patches, config in the model profile):** serving Llama-4 also
needs `attention_backend: triton` (flashinfer is allowlist-rejected for Llama4, same
as Gemma-4) and `moe_runner_backend: triton` (flashinfer_cutlass asserts on
`apply_router_weight_on_input=True`, Llama-4's top-1 input-routing). See
`roles/k8s_dgx/model_profiles/nvidia-llama-4-scout-17b-16e-instruct-nvfp4.yml`.

## Affected

|        |                                                                             |
|--------|-----------------------------------------------------------------------------|
| SGLang | 0.5.15 (verified) and `main` (per patch author)                             |
| Image  | `xomoxcc/dgx-spark-sglang:0.5.15-sm121` (validated), also 0.5.14-sm121 gaps |
| Files  | `sglang/srt/models/mllama4.py`, `sglang/srt/models/llama4.py`               |
| Class  | `Llama4ForConditionalGeneration` (multimodal: text MoE + vision)            |

## The patches

### Block 1: loader (3 fixes, `mllama4.py`) - WITHOUT THESE THE MODEL DOES NOT LOAD

1. **`_handle_expert_scale_params`: fused 3D NVFP4 block-scale.** NVFP4 ships a
   per-expert 3D block-scale `[num_experts, in_blocks, out]` under one (name-less)
   key. The FP8-era code broadcasts a single 2D scale into every expert slot, which
   forces the whole 3D tensor into each 2D slot: `expand([16,512,5120],[5120,512])`
   crash. Fix: `if loaded_weight.dim() == 3` -> slice per expert + `.T`; 0-dim
   scalars (weight_scale_2 / input_scale) and shared 2D scales keep the broadcast.

2. **`permute_qk_weight_for_rotary`: real last-dim, not unpacked hidden.** The
   permute view used `config.hidden_size` (unpacked 5120), but the packed NVFP4 q/k
   weight last-dim is `hidden/2` (2560): `shape [8,64,2,5120] invalid for input of
   size 2621440`. Fix: `attn_out = w.shape[-1]`.

3. **`permute_qk_weight_for_rotary`: permute the weight_scale too.** Only `.weight`
   was row-permuted for rotary, not the per-output-row `weight_scale`, causing a
   scale<->weight desync and wrong q/k dequant. Fix: extend both the k and q
   branches from `modules[-1] == "weight"` to
   `modules[-1] in ("weight", "weight_scale")`.

### Block 2: KV-scale (2 fixes, QUALITY only - serves without them, at scale 1.0)

The modelopt-NVFP4 checkpoint bakes FP8 KV scales (`...k_proj.k_scale` /
`...v_proj.v_scale`). Without these two SGLang never loads them and falls back to
scale 1.0 ("less accurate results"). Both are needed (A alone is a no-op). Verified:
0 not-loaded warnings, and the triton attn backend uses the loaded scale
(`cache_k.div_(k_scale)`).

- **A) `llama4.py`: `RadixAttention` built without `quant_config`.** `Llama4Attention`
  created `self.attn = RadixAttention(...)` without `quant_config` (unlike
  `llama.py` / `qwen2.py`), so `create_weights` never ran and `k_scale`/`v_scale`
  stayed plain `None` attrs, never in `named_parameters()`. Fix: pass
  `quant_config=quant_config` (it is already in scope, used for q/k/v/o).

- **B) `mllama4.py`: `_handle_scale_remapping` never copied.** The function returned
  a bool but never copied the remapped scale into the param (the regular `llama.py`
  loader does; mllama4's reimplementation dropped the copy step). Fix: thread
  `loaded_weight` into the signature + call `self._handle_default_weight(...)` on the
  remapped name (prereq `_handle_default_weight` exists in mllama4.py). Two edits:
  signature/body + the call site.

## Not included (optional)

None outstanding. (An earlier "4th patch" idea, a KV-scale name-remap, turned out to
be exactly Block 2 above.) A custom pythonic-tool chat template is a possible future
add: this image ships NO `tool_chat_template_llama4_pythonic.jinja`, but a built-in
template NAME `llama-4` is registered (`sglang/lang/chat_template.py`) and is
selectable via the profile's `chat_template` knob (wired to `--chat-template`
2026-07-13). The Scout profile leaves `chat_template: ""` (use the tokenizer's own
template) until tool calls are shown to need the built-in.

## Verification (2026-07-13, image 0.5.15-sm121)

- Anchor dry-run: all 7 anchors (4 loader + 3 KV) match verbatim.
- End-to-end apply in a throwaway container: all 7 report `APPLIED`; both
  `llama4.py` and `mllama4.py` `ast.parse` cleanly after patching; second pass is
  idempotent (`already applied`).
- Patch author confirmed: model loads and produces coherent text; KV block gives 0
  not-loaded warnings.
- `bash -n sglang_launch.sh`: clean.
- NOT yet done: a live cluster serve at `tp_size=4` (throughput unmeasured); this
  was a functional TP=1 bring-up.

## Re-sync on image bump

When bumping the SGLang image, the `ANCHOR NOT FOUND (SGLang version drift?)` prints
in the pod log flag that an anchor moved. Re-fetch the two files, re-derive the
anchors, and update the two heredoc blocks in `sglang_launch.sh`. Verify with the
`podman run ... full_verify.py` pattern (all 7 `APPLIED` + `AST_OK`) before serving.

## Upstream tracking

- The 3 loader fixes + 2 KV-scale fixes are PR-worthy against `sgl-project/sglang`.
  **TODO:** file (needs sign-off before posting outward).
- Related: TensorRT-LLM PR #3492 (added `apply_router_weight_on_input` for Llama4
  FusedMoE) is the analogous upstream fix that motivates the profile's triton
  moe_runner requirement. SGLang issue #7994 ("Support NVFP4 masked layout MoE").

## Changelog

- **2026-07-13** - First bring-up. 3 loader patches + 2 KV-scale patches added to
  `sglang_launch.sh` (ungated); `--chat-template` wired from the profile; Scout
  profile set to triton/triton backends. Load + coherent text confirmed at TP=1;
  4-node throughput still unmeasured.
- **2026-07-28** - Re-verified against SGLang **v0.5.16** (released
  2026-07-25, now the latest release). Source-diffed `mllama4.py` and
  `llama4.py` between v0.5.15 and v0.5.16: all five patched anchors
  (`_handle_expert_scale_params`, both `permute_qk_weight_for_rotary` fixes,
  the `RadixAttention` missing `quant_config`, `_handle_scale_remapping`) are
  byte-identical across the two tags. The only diffs in either file are
  unrelated renames from an upstream `RuntimeContext` refactor
  (`get_global_server_args` to `get_server_args` in `mllama4.py`; a
  `use_reduce_scatter` parameter replaced by `get_forward().scoped(...)` in
  `llama4.py`), neither touching the patched code. No upstream issue or PR
  matching these five fixes found since 2026-07-13. All five patches remain
  required, unchanged.
- **2026-08-15** - Re-verified against SGLang **v0.5.17** (tag `b6a09f38f`,
  2026-08-08, no new release since the 2026-07-28 check). `llama4.py` is
  byte-identical to v0.5.16. `mllama4.py` has exactly one diff vs v0.5.16, an
  unrelated `RuntimeContext` rename (`get_server_args()` to `get_mm()`),
  touching none of the five patched anchors. All five anchors individually
  re-confirmed present and unfixed on v0.5.17: `_handle_expert_scale_params`
  (`mllama4.py:859`, still no `loaded_weight.dim()==3` branch, still
  broadcasts a single scale per expert loop); `permute_qk_weight_for_rotary`
  (`mllama4.py:618`, still `attn_out = self.language_model.config.hidden_size`
  and still `modules[-1] == "weight"` only, no `weight_scale` branch, for both
  k and q); the `RadixAttention` construction in `Llama4Attention`
  (`llama4.py:304`, still without `quant_config=quant_config` unlike the
  sibling q/k/v/o layers); `_handle_scale_remapping` (`mllama4.py:736`, still
  returns a bare bool without copying `loaded_weight`). A search for
  `mllama4` NVFP4, `permute_qk_weight_for_rotary`, `_handle_expert_scale_params`,
  and "Llama4 NVFP4 permute" turned up zero results, no upstream tracking has
  appeared. The doc's "TODO: file upstream (needs sign-off)" item remains
  outstanding.
- **2026-08-21** - SGLang v0.5.17 remains the latest release. `mllama4.py`/
  `llama4.py` are byte-identical between v0.5.17 and upstream `main` (commit
  `dad6fd0f04`); all five patched anchors unchanged at the same line numbers
  logged on 2026-08-15. New upstream tracking found: **PR #35032**
  ("[Fix] Llama4 NVFP4 checkpoint loading: fused expert scales and packed
  q/k permute", author `kihan-lee`, filed 2026-08-16) independently fixes
  patches 1 to 3 of our 5 (the fused 3D expert block-scale crash in
  `_handle_expert_scale_params` and both halves of the
  `permute_qk_weight_for_rotary` fix: real last-dim plus the missing
  `weight_scale` permute for q/k), all in `mllama4.py` only. It does NOT
  cover patches 4 to 5 (Block 2, KV-scale quality: the `RadixAttention`
  `quant_config` fix and the `_handle_scale_remapping` copy fix) — those
  remain untracked upstream. Author validated on a DGX Spark (GB10, SM121)
  at tp=1: main dies at 0/14 shards without the fix, GSM8K goes from
  "does not load" to 0.925 with it, and omitting only the weight_scale
  permute drops accuracy to 0.485, corroborating this doc's own finding that
  the scale desync silently corrupts output rather than crashing. PR #35032
  also references issue #34192 (open, filed 2026-08-09, "Gemma4/Llama4
  apply_router_weight_on_input is not supported for Flashinfer"), which
  independently confirms the "Profile side" `moe_runner_backend: triton`
  requirement noted at the top of this doc, and cites #9526 and #12649 as
  earlier closed-unmerged attempts at the same fix (confirmed both CLOSED,
  not merged). PR #35032 status as of 2026-08-21: OPEN, `mergeStateStatus`
  BLOCKED (two AMD-only CI jobs failing, unrelated to our arch),
  `reviewDecision` REVIEW_REQUIRED, no review comments yet. Not merged, so
  our 5 local runtime patches in `sglang_launch.sh` remain required and
  unaffected.
- **2026-08-28** - SGLang v0.5.18 released 2026-08-22, now the latest release.
  `mllama4.py`/`llama4.py` are byte-identical across v0.5.17, v0.5.18, and
  upstream `main` (commit `d56706459`, 2026-08-28); all five patched anchors
  unchanged at the same lines already logged (`mllama4.py:859/618/736`,
  `llama4.py:304`). PR #35032: still OPEN, no commits, reviews, or comments
  since 2026-08-16 (`updatedAt` unchanged); `mergeable: MERGEABLE` but
  `mergeStateStatus: BLOCKED` on failing non-CUDA CI (AMD ROCm-extra, ROCm
  7.2, MLX, MUSA finish jobs), still unrelated to our GB10/SM121 CUDA target.
  Not merged. Issue #34192 unchanged, still OPEN with 0 comments since
  2026-08-09. No new SM12x-related Llama4-NVFP4 issue or PR found. All five
  local patches in `sglang_launch.sh` remain required.
- **2026-09-02** - SGLang v0.5.18 remains the latest release, no new release
  since 2026-08-22. `mllama4.py`/`llama4.py` byte-identical to the 2026-08-28
  check on `main` (HEAD `f5866545186f27a8a01c1a083e8c32809b5506f3`); all five
  patched anchors unchanged at the same lines (`mllama4.py:618/736/859`,
  `llama4.py:304`, `RadixAttention` construction still without
  `quant_config=quant_config`). PR #35032 (loader fixes 1-3) remains OPEN,
  unchanged, no activity since 2026-08-16, still `mergeStateStatus: BLOCKED`
  on unrelated non-CUDA CI. **New, relevant to the "Profile side" workaround
  note:** PR [#35504](https://github.com/sgl-project/sglang/pull/35504)
  ("fix(moe): support Llama4 NVFP4 router input weights on SM120", author
  `janbernloehr`, created 2026-08-19, first review activity 2026-08-31/09-01
  so new since the 2026-08-28 check) fixes issue #34192 by adding top-1
  router-weight pre-scaling to the FlashInfer CUTLASS MoE path and removing
  the ModelOpt NVFP4 / FlashInfer CUTLASS assertions that reject
  `apply_router_weight_on_input=True` for top-1 routing (Llama4's case,
  top-k>1 and pre-quantized-dispatch configs stay explicitly rejected). If
  merged and released, this would obsolete this doc's `moe_runner_backend:
  triton` profile requirement for Llama-4-Scout NVFP4. Status as of today:
  OPEN, `reviewDecision: REVIEW_REQUIRED`, 2 reviews (`nvpohanh`,
  `janbernloehr`, both 2026-08-31), not merged. Issue #34192 itself
  unchanged, still OPEN, 0 comments. Does not affect the 5 local
  `sglang_launch.sh` runtime patches (those are load-path fixes, unrelated
  to the MoE-runner assertion); the profile's `moe_runner_backend: triton`
  pin stays correct until #35504 merges and ships in a release, re-check
  next cycle.
- **2026-09-04** - SGLang v0.5.18 remains the latest release, no new release
  since 2026-08-22. PR #35504 gained fresh activity since the 09-02 check:
  reviewer `nvpohanh` posted `/tag-and-rerun-ci` on 2026-09-01, then today
  asked the author directly, "@janbernloehr could you fix the conflicts?".
  Source-confirmed via the REST API: `mergeable: false`,
  `mergeable_state: dirty`, a real merge conflict against `main`, not just
  the two AMD-only CI jobs noted on 2026-08-21/08-28. The author pushed a
  style fixup (`4bee98856`, 08-31), a test entry point (`a08f2df3a`, 09-01),
  and a merge-from-main commit (`bb02d04a4`, 09-02) on top of the original
  `47b6b56e9`, but the conflict remains unresolved as of today. CI on the
  current head is now broadly red, not just the AMD/NPU jobs previously
  logged: base-b-test-1-gpu-large/small, base-b-test-2-gpu-large,
  base-c-test-perf (NPU), build-test (xeon-spr), and the aggregate
  finish/pr-test-finish/pr-gate checks are all failing, consistent with an
  unresolved conflict blocking a clean build. `reviewDecision` still
  REVIEW_REQUIRED, same 2 reviews from 2026-08-31, no new formal review,
  state OPEN, not merged. PR #35032 (loader fixes 1-3) unchanged, no
  activity since 2026-08-16, still BLOCKED on unrelated non-CUDA CI. Issue
  #34192 unchanged, still OPEN, 0 comments since 2026-08-09. Net effect:
  PR #35504 is further from merging than it was two days ago (a real
  conflict, not just CI noise), despite the continued reviewer attention.
  Does not affect the 5 local `sglang_launch.sh` runtime patches; the
  profile's `moe_runner_backend: triton` pin stays correct, re-check next
  cycle.
- **2026-09-10** - SGLang v0.5.19 released 2026-09-05, now the latest
  release. `mllama4.py`/`llama4.py` are byte-identical between v0.5.18 and
  v0.5.19 (confirmed via `git diff v0.5.18 v0.5.19 --`, zero output for
  both files) and unchanged on upstream `main` (HEAD
  `908226fea2df861769e2720161a75649ae4c6f92`, 2026-09-10); all five patched
  anchors remain at the same lines already logged (`mllama4.py:618/736/859`,
  `llama4.py:304`). PR #35032 (loader fixes 1-3) unchanged, no activity
  since 2026-08-16, `mergeable: true` but `mergeable_state: blocked` on
  unrelated non-CUDA CI. **PR #35504 moved significantly since the 09-04
  entry:** the merge conflict reviewer nvpohanh flagged on 09-04 is now
  resolved, the author merged `main` into the branch three more times
  (09-07, 09-09, 09-10) and REST now reports `mergeable: true`; however
  `mergeable_state` is now `blocked` for a different reason, CI is broadly
  red across `base-b-test-1-gpu-large/small`, `base-b-test-2-gpu-large`,
  `base-c-test-perf` (NPU), `pr-gate`, and the aggregate `finish` checks all
  failing on the current head. Reviewer `b8zhong` posted an `APPROVED`
  review on 2026-09-04, but `reviewDecision` is still `REVIEW_REQUIRED` (12
  reviewers requested from the CODEOWNERS list, `ch-wan`, `BBuf`,
  `Edwardf0t1`, `FlamingoPg`, `AniZpZ`, `HaiShaw`, `OrangeRedeng`,
  `Alisehen`, `mmangkad`, `merrymercy`, `Ying1123`, `Fridge003`, `ispobock`,
  none of whom have approved). nvpohanh re-ran CI three times (09-07,
  09-08, 09-09) without success. Net: closer to mergeable (conflict gone)
  but still blocked by broadly failing CI and an outstanding formal review
  requirement, not merged. Issue #34192 unchanged, still OPEN, 0 comments
  since 2026-08-09. Does not affect the 5 local `sglang_launch.sh` runtime
  patches; the profile's `moe_runner_backend: triton` pin stays correct,
  re-check next cycle.
- **2026-09-16** - SGLang v0.5.19 remains the latest release, no new release
  since 2026-09-05. `mllama4.py`/`llama4.py` unchanged (no new release to
  re-diff; last confirmed byte-identical through v0.5.19 in the 2026-09-10
  entry). PR #35032 (loader fixes 1-3) unchanged, no activity since
  2026-08-16, still blocked on unrelated non-CUDA CI. Issue #34192
  unchanged, still OPEN, 0 comments since 2026-08-09. **PR #35504 gained
  real forward movement since the 2026-09-10 entry:** reviewer `nvpohanh`,
  who has been driving CI reruns throughout, posted a formal **APPROVED**
  review on 2026-09-14 (a second approval, alongside `b8zhong`'s from
  2026-09-04), and the author merged `main` into the branch three more
  times on 2026-09-15 (on top of the three already logged for 09-07/09/10).
  REST confirms `mergeable: true` (conflict-free, unchanged from 09-10) and
  `mergeable_state: blocked`; CI on the current head is still broadly red
  across the same job set logged on 09-10 (base-b-test-1-gpu-large/small,
  base-b-test-2-gpu-large, pr-gate, and the aggregate finish/pr-test-finish
  checks), despite `nvpohanh` re-running failed CI six more times through
  today (09-07, 09-08, 09-09 x2, 09-14, and again 09-16 06:29 UTC this
  morning). `reviewDecision` remains `REVIEW_REQUIRED`: the same 13
  CODEOWNERS are still requested (`ch-wan`, `BBuf`, `Edwardf0t1`,
  `FlamingoPg`, `AniZpZ`, `HaiShaw`, `OrangeRedeng`, `Alisehen`, `mmangkad`,
  `merrymercy`, `Ying1123`, `Fridge003`, `ispobock`) and none have approved;
  `nvpohanh` is not among them, so the new approval does not move
  `reviewDecision`. Net: two approvals now (one from a CODEOWNERS-adjacent
  reviewer, one not), still conflict-free, but still blocked purely on CI
  and the outstanding CODEOWNERS review, not merged. Does not affect the 5
  local `sglang_launch.sh` runtime patches (`p52`/`p53` plus the others
  target `mllama4.py`/`llama4.py`, confirmed byte-identical through
  v0.5.19, so none needed re-anchoring in the 2026-09-11 patch pass, commit
  958c98d, consistent with their absence from that commit's changed-file
  list); the profile's `moe_runner_backend: triton` pin stays correct,
  re-check next cycle. Local note: cluster image bumped from
  `xomoxcc/dgx-spark-sglang:0.5.17-sm121` to
  `xomoxcc/dgx-spark-sglang:0.5.19-sm121` (repo commits f36d51b/958c98d/
  5c07a34, 2026-09-11..15); no effect on this doc's conclusions.
- **2026-09-22** - SGLang v0.5.20 released 2026-09-18 (tag commit
  `94602c9c2b7cbdb8efd5c52802dac6a1c180089e`), now the latest release,
  superseding v0.5.19. `mllama4.py`/`llama4.py` diff v0.5.19 to v0.5.20 is
  purely cosmetic (a formatter pass removing 8 blank lines that used to
  follow `class X(nn.Module):` declarations, part of the same wider
  black/ruff bump visible across other files this release, e.g.
  `moe_wna16.py`/`fp8.py`); all five patched anchors are unchanged in
  substance, only line-shifted: `_handle_expert_scale_params`
  (`mllama4.py:861`, still no `loaded_weight.dim()==3` branch, still the
  flat `param.data[expert_id] = loaded_weight` per-expert-loop assignment),
  `permute_qk_weight_for_rotary` (`mllama4.py:620`, still `attn_out =
  self.language_model.config.hidden_size` and still `modules[-1] ==
  "weight"` only, no `weight_scale` branch, for both k and q), the
  `RadixAttention` construction in `Llama4Attention.__init__`
  (`llama4.py:302`, still without `quant_config=quant_config`),
  `_handle_scale_remapping` (`mllama4.py:730`, still `return remapped_name
  != name` from `maybe_remap_kv_scale_name(...)`, no copy of
  `loaded_weight`). All five local patches (`p52_mllama4_loader.py`,
  `p53_mllama4_kvscale.py`) remain required, unchanged.

  **PR #35504 MERGED today, 2026-09-22T03:18:11Z** (merge commit
  `56fee88e236b7666b87adeceecfb14230a482f61`, base `main`), after two
  approvals (`b8zhong` on 2026-09-04, `nvpohanh` on 2026-09-14) finally
  cleared the outstanding CODEOWNERS review and CI that had blocked it
  since the 09-02 entry. The merge landed about 4.5 hours after the
  v0.5.20 tag was cut (2026-09-18T08:02 UTC, published 2026-09-18T22:41
  UTC), so **the fix is not in v0.5.20**, confirmed independently by the
  profile's `moe_runner_backend: triton` requirement staying necessary on
  that tag (the touched files, `modelopt_quant.py` and FlashInfer CUTLASS
  dispatch code, are outside the diff checked above). This merge also
  auto-closed issue #34192 ("Gemma4/Llama4 apply_router_weight_on_input is
  not supported for Flashinfer") via the PR's `Fixes #34192` keyword,
  closed at `2026-09-22T03:18:13Z`. Once an image ships a commit at or
  after `56fee88`, the profile's `moe_runner_backend: triton` pin for
  `nvidia-llama-4-scout-17b-16e-instruct-nvfp4.yml` can be re-evaluated
  against `flashinfer_cutlass`; re-check next cycle whether a v0.5.21+
  release (or a `main`-tracking build) picks it up. This does not touch
  the 5 local `sglang_launch.sh`/`p52`/`p53` load-path patches, which are
  a different file entirely from PR #35504's target.

  PR #35032 (loader fixes 1-3, `_handle_expert_scale_params` plus both
  `permute_qk_weight_for_rotary` halves) remains OPEN, unmerged, no
  activity since 2026-08-16, still `mergeable: true` / `mergeable_state:
  blocked` on the same unrelated non-CUDA CI jobs already logged. Local
  note: repo commits 63e72da/f3f29e1 (2026-09-22) add SGLang v0.5.20 SM121
  build patches and a recipe; the image is not yet built or deployed,
  cluster stays on `xomoxcc/dgx-spark-sglang:0.5.19-sm121`.
- **2026-09-25** - No new SGLang release since v0.5.20 (2026-09-18, still
  latest per `gh release list`/`gh api tags`, no v0.5.21 or v0.5.20.postN
  tagged). PR #35504's merge commit `56fee88e236b7666b87adeceecfb14230a482f61`
  therefore remains unreleased; independently re-confirmed via `gh api
  repos/sgl-project/sglang/compare/v0.5.20...56fee88e236b7666b87adeceecfb14230a482f61`
  which returns `status: diverged, ahead_by: 377, behind_by: 5`, i.e. the
  fix sits 377 commits ahead of the v0.5.20 tag on `main` and is not
  contained in it. PR #35032 (loader fixes 1-3) unchanged, still open, no
  activity since 2026-08-16. Issue #34192 stays closed via PR #35504's
  `Fixes` keyword (2026-09-22), unchanged. Local note: repo commit `3214ed2`
  (dated 2026-09-22 per `git log`) flipped `default_sglang_image` to
  `xomoxcc/dgx-spark-sglang:0.5.20-sm121`; the live cluster is now confirmed
  running that image on all sglang pods (head plus 3 workers, checked today
  via `kubectl`). This doc's 2026-09-22 entry already source-verified
  `mllama4.py`/`llama4.py` directly against the v0.5.20 tag (cosmetic
  formatter diff only, all five patched anchors unchanged in substance), so
  nothing new to re-verify now that the image is actually deployed instead
  of merely built. The 5 local `sglang_launch.sh`/`p52`/`p53` load-path
  patches and the profile's `moe_runner_backend: triton` pin stay correct.
  Re-check next cycle whether a v0.5.21+ release (or a `main`-tracking
  build) picks up `56fee88`.
- **2026-09-28** - No new SGLang release since v0.5.20 (2026-09-18, still
  latest). `git tag --contains
  56fee88e236b7666b87adeceecfb14230a482f61` returns empty on a freshly
  fetched `upstream/main`: PR #35504's fix is still not in any tag.
  `mllama4.py` has zero touching commits since 2026-09-25; `llama4.py` had
  two refactor commits (`81f27fb3a7`, `35b369ebba`, both
  `Llama4DecoderLayer` communicator/layer_facts plumbing) that do not touch
  `Llama4Attention`'s `RadixAttention(...)` construction. Confirmed on
  `main` HEAD (`df378bca42`, 2026-09-28T16:59:59+08:00): all five patched
  anchors are unchanged in substance, only line-shifted from the unrelated
  llama4.py commits (`_handle_expert_scale_params` at `mllama4.py:854`,
  `permute_qk_weight_for_rotary` at `mllama4.py:613`, still `attn_out =
  self.language_model.config.hidden_size` and `modules[-1] == "weight"`
  only for both q and k, `_handle_scale_remapping` at `mllama4.py:731`,
  still no `loaded_weight` copy; `RadixAttention(...)` in
  `Llama4Attention.__init__` at `llama4.py:296`, still without
  `quant_config=quant_config`). PR #35032 (loader fixes 1-3) unchanged,
  still OPEN, no activity since 2026-08-16. Issue #34192 stays closed via
  PR #35504's merge (2026-09-22), unchanged. The 5 local
  `sglang_launch.sh`/`p52`/`p53` patches and the profile's
  `moe_runner_backend: triton` pin stay correct. Re-check next cycle
  whether a v0.5.21+ release (or a `main`-tracking build) picks up
  `56fee88`.
- **2026-10-03** - SGLang v0.5.21 was released 2026-10-02 (v0.5.20
  previous, tag commit `e00930c548`). `git tag --contains
  56fee88e236b7666b87adeceecfb14230a482f61` now returns `v0.5.21`: PR
  #35504's fix (flashinfer_cutlass `apply_router_weight_on_input` support via
  `_prescale_router_weight_on_input()` in `moe_runner/flashinfer_cutlass.py`,
  plus removal of the two `assert not apply_router_weight_on_input` guards
  there and the matching one in `modelopt_quant.py`) first ships in this tag.
  Diff of `56fee88` checked: it touches only `flashinfer_cutlass.py` and
  `modelopt_quant.py`, never `llama4.py`/`mllama4.py`, so it does not fix any
  of the five patched anchors. Anchor check at the v0.5.21 tag AND on `main`
  HEAD (`7be5e3473c`, 2026-10-03): all five unchanged in substance:
  `_handle_expert_scale_params` (`mllama4.py:854`) still has the fused-3D
  scale gap, `permute_qk_weight_for_rotary` (`mllama4.py:613`) still
  `attn_out = self.language_model.config.hidden_size` and
  `modules[-1] == "weight"` only for both q and k, `_handle_scale_remapping`
  (`mllama4.py:731`) still no `loaded_weight` copy, and `RadixAttention(...)`
  in `Llama4Attention.__init__` (`llama4.py:302` at v0.5.21, `:299` on main)
  still without `quant_config=quant_config`. Ten refactor commits touched
  `llama4.py` since 09-28 (none changes the `RadixAttention(...)`
  construction or adds `quant_config`); `mllama4.py` has none. So the 5
  local `sglang_launch.sh`/`p52`/`p53` patches stay required on v0.5.21 and
  on main. The only item `56fee88` resolves is the profile-side
  `moe_runner_backend: triton` pin (flashinfer_cutlass no longer asserts on
  `apply_router_weight_on_input=True` from v0.5.21); it is unvalidated on
  this cluster (SM121; the PR title targets SM120), and whether to revisit
  the pin is a pending user decision. PR #35032 (loader fixes 1-3)
  unchanged, still OPEN, no activity since 2026-08-16. Issue #34192 stays
  closed via PR #35504's merge (2026-09-22).
- **2026-10-11** - No new SGLang release since v0.5.21 (2026-10-02, tag commit `e00930c548`);
  `56fee88` remains in v0.5.21 and no later tag exists. Anchor check on `main`
  HEAD (`ad8b1e1f85`, 2026-10-10) plus an offline replay of the real patch
  modules against the v0.5.21 tag, `7be5e3473c` and `main`:
  `p52_mllama4_loader` (all four edits) and `p53_mllama4_kvscale` (both parts)
  apply cleanly on all three. `mllama4.py` has no commits since 2026-10-03
  (`permute_qk_weight_for_rotary` at `:621` still `attn_out =
  self.language_model.config.hidden_size` with `modules[-1] == "weight"` only
  for q and k, `_handle_scale_remapping` at `:731` still without
  `loaded_weight`, `_handle_expert_scale_params` fused-3D gap unchanged).
  `Llama4Attention.__init__` `RadixAttention(...)` (`llama4.py:296`) still has
  no `quant_config`; the only `llama4.py` commits since 2026-10-03 are
  `4c5130ac5a` (#42426, parallel-group selection) and `1f299e9f2f` (#42481,
  append_stages), neither touching it. The local patches stay required on
  v0.5.21 and `main`. Local note: the profile's `moe_runner_backend: triton`
  pin is unchanged (repo commit `ce515ec`, 2026-10-10, only added LiteLLM cost
  fields); revisiting the pin for the `56fee88` flashinfer_cutlass path
  remains an open, unvalidated user decision on SM121. PR #35032 (loader fixes
  1-3) still OPEN, no activity since 2026-08-16; issue #34192 stays closed via
  PR #35504.
- **2026-11-10** - SGLang v0.5.22 (tag commit `baa9befa38`, Release 2026-10-19) and v0.5.23 (tag commit `297e54fd10`, Release 2026-11-02, latest) are out; `56fee88` is an ancestor of both (and of v0.5.21). Offline replay of the real patch modules `p52_mllama4_loader`, `p53_mllama4_kvscale` and `p70_modelopt_exclude_fused_layer_scope` against v0.5.21, v0.5.22, v0.5.23 and `main` HEAD (`170804b40a`, 2026-11-10): all apply cleanly and are idempotent. On `main`, `permute_qk_weight_for_rotary` is at `mllama4.py:603` (`attn_out = self.language_model.config.hidden_size` at `:611`), `_handle_scale_remapping` at `:721`, and `Llama4Attention.__init__` `RadixAttention(...)` (`llama4.py:294`) still has no `quant_config`; the commits touching `mllama4.py`/`llama4.py` since 2026-10-12 (`2a8e9db3b0`, `e01dfd3cbc`, `efdcb420c7`, `751fa99d7e`, `4f8f06faf3`, `a076243cea`, `6b13469f91`) are LoRA, multimodal-padding, import and refactor changes that leave the patched logic as is. The local patches stay required. PR #35032 (loader fixes 1-3) still OPEN, no activity since 2026-08-16; issue #34192 stays closed via PR #35504 (merged 2026-09-22). The profile's `moe_runner_backend: triton` pin is unchanged.
