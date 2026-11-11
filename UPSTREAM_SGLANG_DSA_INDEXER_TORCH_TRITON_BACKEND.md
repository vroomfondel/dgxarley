# UPSTREAM PR: `dsa_paged_mqa_logits_backend=torch` fallback (+ Triton fast path) for the DSA indexer

Status: **FILED 2026-07-16 als sgl-project/sglang PR #31480** (https://github.com/sgl-project/sglang/pull/31480). Local implementation =
`roles/k8s_dgx/files/sglang_patches/p30_dsa_torch_backend.py` (torch backend,
incl. next_n>=2) + `p35_dsa_indexer_triton_logits.py` (Triton fast path),
live-proven on dgxarley (DGX Spark GB10, SM121). After a merge lands in a
release we build on: DELETE p30 + p35.

Upstream state (checked 2026-07-16): `DSAPagedMQALogitsBackend` on `main` has
only DEEPGEMM / CUTEDSL / AITER; no arch-independent fallback exists. Companion
draft: UPSTREAM_SGLANG_DSA_SM12X_NATIVE_SPARSE_MLA.md (the attention side).

Re-checked 2026-07-23: PR #31480 still **OPEN**, `reviewDecision:
REVIEW_REQUIRED`, `mergeStateStatus: UNKNOWN` — no review activity since
filing. `DSAPagedMQALogitsBackend` on `main` is unchanged (still only
DEEPGEMM / CUTEDSL / AITER). SGLang released v0.5.15 (2026-07-10) and
v0.5.15.post1 (2026-07-14) since the PR was filed; both contain DSA
top-k/indexer-fusion work (#26788, #30274, #27705) but no arch-independent
`torch` backend.

Re-checked 2026-07-28: PR #31480 still **OPEN**, no new commits or comments
since 2026-07-16 (`updatedAt` unchanged). `mergeStateStatus` now reads
`BLOCKED` via the API instead of `UNKNOWN`, that is GitHub's lazy merge-state
computation catching up, not a real event. `DSAPagedMQALogitsBackend` on
`main` fetched again and confirmed unchanged (still only DEEPGEMM / CUTEDSL /
AITER, no `torch` value). SGLang released v0.5.16 on 2026-07-25 since the last
check, its "DeepSeek V4" section ships DSA-adjacent work (#30514 Q8KV8 FP8
Sparse MLA Prefill integration, #30140 non-paged indexer default, #30012 BF16
instead of FP32 for indexer score, #30645 top-k v2 fix) but none of it adds an
arch-independent `torch` paged-MQA-logits backend. No change to this doc's
conclusions.

Re-checked 2026-08-03 (morning): PR #31480 still **OPEN**, no new commits or
comments since 2026-07-16 (`updatedAt` unchanged). `mergeStateStatus` had
progressed `BLOCKED` → `DIRTY` with `mergeable: CONFLICTING` — i.e. the PR
needed a rebase against current main, the same state its companion PR #31481
has been in since 2026-07-28. `DSAPagedMQALogitsBackend` on `main` fetched
again and confirmed unchanged (still only DEEPGEMM / CUTEDSL / AITER, no
`torch` value). SGLang still at v0.5.16, no new release. Content/design
conclusions unchanged.

Re-checked 2026-08-03 (afternoon, post-rebase): the rebase flagged above is
DONE for #31480 — branch `dsa-indexer-torch-triton-backend` rebased onto
current main and force-pushed ~10:35 UTC (same single commit content, new
head `5480d1d28d`, `committedDate` 2026-08-03). Live state now
`mergeable: MERGEABLE`, `mergeStateStatus: BLOCKED` (i.e. only awaiting
review, no conflict), `reviewDecision: REVIEW_REQUIRED`, no new comments.
Companion PR #31481 was NOT rebased and remains `DIRTY`/`CONFLICTING` — the
two PRs have diverged in merge-readiness; #31481 still needs its own rebase.

Re-checked 2026-08-07: the afternoon entry above was itself overtaken the
same day — #31480 got a SECOND force-push (head now `685784f5ae`,
`committedDate` 2026-08-03T12:40Z, still a single commit), and #31481 WAS
rebased after all (2026-08-03 ~11:15 UTC, see its own doc), so the two PRs
are back in sync. At ~13:03 UTC the author commented on #31480: "Rebased
onto current main; 17/17 kernel tests pass on GB10 (SM121). Could a
maintainer add the `run-ci` label and take a look? Companion PR: #31481."
As of 2026-08-07: **no maintainer response, no `run-ci` label** (`labels`
still empty), no reviews, `mergeable: MERGEABLE`, `mergeStateStatus:
BLOCKED`, `reviewDecision: REVIEW_REQUIRED`, no new pushes. The
`pr-gate`/`call-gate` checks show FAILURE with all substantive jobs SKIPPED —
that is the label-gate mechanism, not a real test failure.
`DSAPagedMQALogitsBackend` on `main` re-fetched: still only
DEEPGEMM / CUTEDSL / AITER, no `torch` value. SGLang still at v0.5.16.
(External-interest footnote: third-party fork `AMD-AGI/Infera` PR #79,
2026-08-03, cross-references this PR.)

Re-checked 2026-08-09: no change — still **no maintainer response, no
`run-ci` label**, no reviews, no new pushes on #31480 (REST API:
`mergeable: true`, `mergeable_state: "blocked"`, `updated_at` still
2026-08-03T13:03:37Z; GraphQL transiently reports `UNKNOWN`/`UNKNOWN`
merge state — the known lazy-cache artifact after `main` moved, here the
**v0.5.17 release merge wave, 582 PRs, released 2026-08-08** — not a real
event). `DSAPagedMQALogitsBackend` on `main` re-fetched: still only
DEEPGEMM / CUTEDSL / AITER, no `torch` value; no merged PR since 08-07
touches `paged_mqa_logits_backend.py` or `dsa_indexer.py` dispatch.
v0.5.17 contains no fix for this gap.

Re-checked 2026-08-15: PR #31480 still **OPEN**, no new commits or comments
since 2026-08-03, no `run-ci` label, `reviewDecision: REVIEW_REQUIRED`. REST
API now reports `mergeable: false` / `mergeable_state: "dirty"` (was
`"blocked"` on 2026-08-09): main has drifted since the 08-03 rebase, so
#31480 needs another rebase (NOT done yet, pending approval). Companion
#31481 remains `mergeable: true` / `"blocked"`. `DSAPagedMQALogitsBackend`
on `main` unchanged (still only DEEPGEMM / CUTEDSL / AITER). Merged
DSA-touching PRs since 08-08 (#32755, #33006, #34443, #34167) add no
arch-independent backend. SGLang still at v0.5.17, no new release.

New finding to flag (needs live verification, do NOT treat as resolved):
this doc frames DeepGEMM's `auto` default as asserting "Unsupported
architecture" on SM120/121 because deepseek-ai/DeepGEMM declined SM12x
support (PR #318, still an unmerged PoC). But sglang actually vendors
`sgl-deep-gemm` (fork `sgl-project/DeepGEMM`, tracking an NVIDIA-maintained
`nv_dev` branch), which merged first-class SM120/SM121 support via DeepGEMM
PR #324 on 2026-06-24, including dedicated `sm120_fp8_paged_mqa_logits.cuh`
/ `sm120_fp4_paged_mqa_logits.cuh` kernels. Those SM120 paged-MQA-logits
kernel files are present in the sgl-deep-gemm wheel from version 0.1.5
(2026-07-24) onward, i.e. already in `sgl-deep-gemm==0.1.5.post1`, which
v0.5.17's `pyproject.toml` pins (sglang `main` now pins `0.1.5.post3`,
released 2026-08-15T01:11Z). UNVERIFIED whether sglang's JIT/dispatch layer
actually selects these SM120 templates at runtime, or whether another gate
still forces the assert; needs a live GB10 test (pending approval, do NOT
claim it works). Note that p30/p35 may still be worth keeping for the
Triton fast-path perf win even if deepgemm now boots.

Follow-up same day (2026-08-15, approved): #31480 rebased onto current main
`0c072235f` (2026-08-15). Old head `685784f5ae` -> new head `68cf312934`,
pushed `--force-with-lease` to the vroomfondel fork. Single conflict in
`python/sglang/srt/environ.py`: upstream's config-bag reorganization moved
the DSA env-var section to a new "DSA backend (GLM 5 and DeepSeek V3.2)"
location and inserted a new "DeepGEMM Mega MoE" section at the old spot; our
`SGLANG_DSA_INDEXER_TRITON` field was re-inserted at the end of the relocated
section. All other 7 touched files auto-merged clean; range-diff shows only
that relocation plus an upstream base-class rename in `dsa_indexer.py` diff
context (`Indexer(MultiPlatformOp)` -> `Indexer(DSANPUIndexerMixin,
BaseFusedOp)`); own diffstat unchanged (8 files, +1022/-12). REST now reports
`mergeable: true` / `"blocked"` (checks/review gating, not a conflict).

Second follow-up same day (2026-08-15, approved): the GB10 stock-image test
on spark5 (see UPSTREAM_SGLANG_DSA_SM12X_NATIVE_SPARSE_MLA.md for the full
setup) produced two data points for THIS doc:
- **p30 is NOT redundant on stock v0.5.17:** the stock CLI accepts only
  `--dsa-paged-mqa-logits-backend auto|deepgemm|cutedsl|aiter`; our
  profile's `torch` value (the choice p30 adds) fails argument parsing on
  the unpatched image. The launch had to fall back to `auto` to boot.
- With `auto`, the shrunk-config dummy-weight run (TP1, SM121, one full
  DSA indexer layer exercised through a 4k-token prefill + decode) was
  crash-free: no deepgemm "Unsupported architecture" assert appeared. The
  resolved indexer backend is not logged at INFO level, so WHICH backend
  `auto` picked on SM121 remains unconfirmed (the sgl-deep-gemm 0.1.5
  SM120 kernels flagged above are a plausible but unverified explanation).
  A DEBUG-level or source-instrumented run would pin it down; p30/p35 stay
  in place regardless (p35 also carries the Triton fast-path perf win).

Re-checked 2026-08-21: PR #31480 unchanged since 08-15, no new commits,
comments or reviews (REST API: `mergeable: true`, `mergeable_state:
"blocked"`, `updated_at` still 2026-08-15T10:12:46Z, head still
`68cf312934`). No `run-ci` label, `reviewDecision: REVIEW_REQUIRED` still.
`DSAPagedMQALogitsBackend`
(`python/sglang/srt/layers/attention/dsa/paged_mqa_logits_backend.py`) on
`upstream/main` has zero commits since the 08-15 base `0c072235f`; still
only DEEPGEMM / CUTEDSL / AITER, `resolve()` still maps auto/deepgemm to
DEEPGEMM unconditionally on non-ROCm, no `torch` value. SGLang still at
v0.5.17 (2026-08-08), no new release. `sgl-deep-gemm` pin in
`python/pyproject.toml` on `upstream/main` unchanged at `0.1.5.post3`;
PyPI's latest published version is also still `0.1.5.post3` (no `post4`
yet).

New context on the sgl-deep-gemm SM120 paged-MQA-logits kernel flagged
08-15: two open PRs against `deepseek-ai/DeepGEMM`'s `nv_dev` branch (the
branch `sgl-project/DeepGEMM` tracks), neither merged, neither shipped in
any sgl-deep-gemm release: #379 "SM120: fix FP8 MQA logits swizzle mode
for head_dim < 128" (`kSwizzleMode` hardcoded to 128 misreads SMEM for
head_dim 32/64; the author states the paged kernel's head_dim is currently
always 128, so explicitly no behavior change there, meaning this bug does
not affect our DSA cache layout, which hardcodes head_dim=128) and #406
"Add SM120 heuristic calibration harness" (a calibration/sweep tool, no
default constant changes). Neither is relevant to whether a torch/triton
fallback is still needed.

Only one merged sglang commit touches the indexer path in this window:
f7cb328eb7 "[AMD][GLM5] Skip DSA decode indexer when kv_len <= index_topk"
(merged 2026-08-17), but its new decode-skip branch is gated `if not
_is_hip: return False`, i.e. no behavior change on CUDA/SM121, and it does
not add or touch a paged-mqa-logits backend choice. No commit since
`0c072235f` touches `paged_mqa_logits_backend.py` itself. No new sglang
issue or PR found searching "dsa_paged_mqa_logits_backend torch" or
"flashinfer_sparse_mla" proposing an arch-independent torch indexer
backend.

Conclusion unchanged: p30/p35 remain necessary on stock v0.5.17/current
main; no upstream fix has landed or is imminent.

Re-checked 2026-08-28: PR #31480 head unchanged (`68cf312934`), `updated_at`
unchanged (2026-08-15T10:12:46Z), no new comments (still 3: the bot quota
warning, the companion link, the 08-03 rebase note) or reviews, no `run-ci`
label, `reviewDecision: REVIEW_REQUIRED`. Mergeable state flipped again: REST
API now reports `mergeable: false` / `mergeable_state: "dirty"` (was `true` /
`"blocked"` on 08-21), i.e. main has drifted since the 08-15 rebase and
#31480 needs another rebase (not done, pending approval, same pattern as the
08-15-morning entry). `DSAPagedMQALogitsBackend`
(`python/sglang/srt/layers/attention/dsa/paged_mqa_logits_backend.py`) on
`upstream/main` re-fetched via `git show`: zero commits touch this file
since baseline `dad6fd0f04`; still only DEEPGEMM/CUTEDSL/AITER, `resolve()`
still maps auto/deepgemm to DEEPGEMM unconditionally on non-ROCm, no `torch`
value. `server_args.py`'s `dsa_paged_mqa_logits_backend` CLI choices
confirmed still exactly `["auto", "deepgemm", "cutedsl", "aiter"]` on
upstream/main, so stock still rejects `torch` (p30 not redundant, unchanged
conclusion).

SGLang released v0.5.18 (2026-08-22, 710 PRs). Confirmed it contains #34926
("Clean deprecated DeepSeek V4 Environs", commit `bc312d185d`, removes
`SGLANG_TOPK_TRANSFORM_512_TORCH`) via `git log v0.5.17..v0.5.18
--grep=34926`. Release changelog DSA entries reviewed: nothing new beyond
already-tracked #33006/#34167; no arch-independent paged-MQA-logits backend
added.

`sgl-deep-gemm` pin in `python/pyproject.toml` on `upstream/main` unchanged
at `0.1.5.post3`; PyPI latest also still `0.1.5.post3` (full version list
checked, no `post4`). Both `deepseek-ai/DeepGEMM` PRs remain unmerged: #379
(`OPEN`, `updated_at` 2026-08-13) and #406 (`OPEN`, `updated_at`
2026-08-22, still just the calibration harness).

Six commits touched `python/sglang/srt/layers/attention/dsa/` since
baseline, none touching `paged_mqa_logits_backend.py` and none adding an
arch-independent backend: `7fd5454335` ([DSA] Route ragged prefill top-k to
v2 kernel, #35175, merged 08-21), `af39ad9349` (dots.note.omni model
support, unrelated), `3c9febc68b` ([Spec][DSA] Add
`--speculative-dsa-topk-backend`, #36313, merged 08-25, a separate,
pre-existing `DSATopKBackend` enum that already had a `torch` value before
this baseline, not the paged-MQA-logits backend this doc tracks), and three
mechanical "config:" ServerArgs-declaration-refactor commits (`8005df61d3`,
`ca1d7ed8e6`, `fd40a331bf`). No new sglang issue or PR found searching
"dsa_paged_mqa_logits_backend torch".

New (immature) activity to watch, not conclusion-changing: PR #36507
"GLM-5.3-Flash support" (opened 2026-08-26, updated 2026-08-28, no `run-ci`
label, unreviewed) adds new `dsa_indexer_kpool.py` / `kpool_fp8_index.py` /
`kpool_plan.py` infrastructure but does not touch `paged_mqa_logits_backend.py`
or `flash_mla_sm120.py`.

Conclusion unchanged: p30/p35 remain necessary on stock v0.5.18/current main.

Follow-up same day (2026-08-28, approved): #31480 rebased onto current main
`d56706459c` (2026-08-28, this same rebase's upstream/main HEAD). Old head
`68cf312934` -> new head `6b2e62e259`, pushed `--force-with-lease` to the
vroomfondel fork. Two conflicting files, both from the ongoing declarative
ServerArgs refactor (the "config bags" work, `A[..., Arg(...), NS(...)]`
field annotations replacing the old argparse style, and per-arg `choices`
now written inline instead of via a shared module-level constant list):
- `python/sglang/srt/layers/attention/dsa_backend.py`: our added
  `self.paged_mqa_logits_backend = DSAPagedMQALogitsBackend.resolve(...)`
  line collided with upstream moving the neighboring `dsa_prefill_impl` /
  `dsa_decode_impl` / `dsa_topk_backend` reads from `model_runner.server_args.*`
  to `get_exec().kernel.*`. Resolved by keeping upstream's new accessor style
  for those three lines and rewriting our own line to match the same
  pattern (`get_exec().kernel.dsa_paged_mqa_logits_backend` instead of
  `model_runner.server_args.dsa_paged_mqa_logits_backend`), matching the
  pre-existing analogous resolve call already present in
  `dsa/dsa_indexer.py` (which merged clean, since we never touched that
  line).
- `python/sglang/srt/server_args.py`: our `DSA_PAGED_MQA_LOGITS_BACKEND_CHOICES`
  module constant (with `torch` added) collided with upstream deleting the
  whole block of similar constants (`LORA_BACKEND_CHOICES`, `DSA_CHOICES`,
  `DSA_TOPK_BACKEND_CHOICES`, `MAMBA_BACKEND_CHOICES`, etc.) in favor of
  inline `choices=[...]` lists at each field. Resolved by dropping our
  constant and instead adding `"torch"` directly to the inline
  `choices=["auto", "deepgemm", "cutedsl", "aiter"]` list on the
  `dsa_paged_mqa_logits_backend` field, keeping our extended help text
  describing the `torch` option.
All other 6 touched files (`environ.py`, `dsa/dsa_indexer.py`,
`dsa/paged_mqa_logits_backend.py`, `dsa/torch_paged_mqa_logits.py`,
`dsa/triton_paged_mqa_logits.py`, `test/registered/kernels/test_dsa_paged_mqa_logits.py`)
auto-merged clean, no manual changes needed. Verified: no leftover conflict
markers repo-wide, `ast.parse` clean on all 8 touched files,
`git range-diff 0c072235f..68cf312934 upstream/main..HEAD` shows only the
two resolutions above as content drift (rest is pure context shift), own
diffstat unchanged at 8 files, +1022/-12 (identical to the 08-15 baseline).
REST/GraphQL now report `mergeable: MERGEABLE` / `mergeStateStatus: BLOCKED`
(checks/review gating only, no conflict), `headRefOid` confirmed
`6b2e62e259398b8e34b8eac5d92f6d6a7c9448ff` live on the PR.

> Re-checked 2026-09-02: PR #31480 unchanged since 08-28 (head still
> `6b2e62e259398b8e34b8eac5d92f6d6a7c9448ff`, `updated_at` 2026-08-28T10:48:29Z
> reflects only the rebase-triggered CI run completing, not a new comment or
> push), still 3 comments, 0 reviews, no `run-ci` label, `reviewDecision:
> REVIEW_REQUIRED`. REST API: `mergeable: true`, `mergeable_state: "blocked"`
> (stable since the 08-28 rebase, no new drift). `call-gate`/`pr-gate` still
> shows FAILURE with substantive jobs SKIPPED (label-gate mechanism,
> unchanged).

> `DSAPagedMQALogitsBackend` (`paged_mqa_logits_backend.py`) on
> `upstream/main`: exactly one commit touched the file since the 08-28 baseline
> (`d56706459c`): `7e751153eb` ("[Config] Round 5.1: the published-side readers
> ask the bags, and a platform fact gets one address", #37086), a pure API
> rename (`is_sm100_supported()` -> `get_platform().is_sm100`), no behavior
> change. Still only DEEPGEMM/CUTEDSL/AITER, no `torch` value.
> `server_args.py`'s `dsa_paged_mqa_logits_backend` choices re-verified still
> exactly `["auto", "deepgemm", "cutedsl", "aiter"]` (p30 not redundant,
> unchanged).

> Six commits touched `python/sglang/srt/layers/attention/dsa/` since baseline,
> none adding an arch-independent paged-MQA-logits backend: the same
> `7e751153eb` refactor above, `c66a285c94` ("[Kernel] GLM 5.3 Flash related
> kernels, ported from #36507", merged), `9978aaec8b` ("[ROCm][Bugfix] Cap the
> DSA MQA-logits budget at AITER's buffer_store limit", ROCm-only),
> `b6c06e1efb` ("[DSA] Drop the redundant 512 from the top-k transform
> entry-point names", cosmetic rename), `24c9251ac5` ("[AMD][Spec][PD] Enable
> the PD DSA fused-TopK seed remap on ROCm", ROCm-only), `ec962fd12d` (fall
> back to the process-group broadcast for DSA topk when PyNCCL is absent,
> unrelated fallback path).

> SGLang still at v0.5.18 (2026-08-22), no new release. PR #36507
> "GLM-5.3-Flash support" is still open, unreviewed, no `run-ci` label, but is
> landing piecewise (see `c66a285c94` above); its diff touches
> `_forward_trtllm`/`dsa_backend.py` plumbing (page-table padding, DCP LSE) but
> not `paged_mqa_logits_backend.py`, and adds a new model family
> `Glm5NextForConditionalGeneration` to the DeepSeek-family allowlist without
> altering the indexer-backend dispatch. No new sglang issue or PR found
> searching "dsa_paged_mqa_logits_backend torch".

> Re-checked 2026-09-04: PR #31480 unchanged since 08-28 (head still
> `6b2e62e259398b8e34b8eac5d92f6d6a7c9448ff`, `updated_at`
> 2026-08-28T10:48:29Z), still 3 comments, 0 reviews, no `run-ci` label.
> REST/GraphQL: `mergeable: MERGEABLE` / `mergeStateStatus: BLOCKED` (a first
> read returned transient `UNKNOWN`/`UNKNOWN`, GitHub's async
> merge-computation cache; two immediate repeat reads confirmed the stable
> value, not a real state change).

> Two new commits touched `python/sglang/srt/layers/attention/dsa/` since the
> 09-02 check: `ff1285cc28` ("[CP V1 Deprecation 2/5] Make strategy prefill
> CP canonical", #36223, merged 2026-09-03) rewrites
> `dsa/utils.py::is_dsa_enable_prefill_cp` to key off `is_hip()/is_npu()`
> instead of `SGLANG_ENABLE_CP_V2`, and `28262c20df` (CI: replace
> black-jupyter with ruff-format, #37210) is tooling-only. Neither touches
> `paged_mqa_logits_backend.py` (zero new commits there since the 08-28
> baseline `d56706459c`, still only the `7e751153eb` rename) or adds an
> arch-independent backend value. p30 not redundant, unchanged.

> PR #36507 "GLM-5.3-Flash support" is still open (`mergedAt: null`), gained
> 4 more reviews (kpham-sgl x3, JustinTong0323 x2, all COMMENTED, none
> APPROVED) and several substantive bring-up-bug comments through 09-03 (a
> kpool top-k radix-clip defect report from bold84, a qkvbfg-fusion gate fix
> from linkedlist771, a NEXTN embedding-OOB fix from BBuf, an
> EAGLE+DP+chunked-prefill fix from rohash123, filed as separate PRs
> #37540/#37744/#37728 against this branch). Diffed `dsa_backend.py`
> directly: the hunk touching `_forward_trtllm` only adds a
> `sparse_mla_top_k_lens` param and page-table-padding plumbing; the
> `backend="trtllm-gen"` line and `is_glm_sm12_fp8` selection are untouched
> context, confirming the 09-02 finding still holds. SGLang still v0.5.18, no
> new release.

> Conclusion unchanged: p30/p35 remain necessary on stock v0.5.18/current main;
> no upstream fix has landed or is imminent.

> Re-checked 2026-09-10: PR #31480 unchanged in content since 08-28 (head
> still `6b2e62e259398b8e34b8eac5d92f6d6a7c9448ff`, `updated_at`
> 2026-08-28T10:48:29Z, still 3 comments, 0 reviews, no `run-ci` label). But
> `mergeable`/`mergeStateStatus` flipped from `MERGEABLE`/`BLOCKED` to
> `CONFLICTING`/`DIRTY`, confirmed stable across two repeat `gh` reads 3s
> apart (not the transient GraphQL `UNKNOWN` seen on 09-04). Real merge
> conflict against current main, first observed this cycle.

> Root cause: `server_args.py` (one of p30's two touched files) was
> rewritten wholesale by the "[Config] Round 5/6" refactor series (multiple
> commits since the 08-28 baseline, most recently `f1a512c51c` "msgspec.
> Struct for the config tier" and `53dc77ff4e` "One writer for the
> declaration stash", both merged this week) into a declarative
> `A[...] / Arg(...) / NS(...)` field-descriptor system with
> `resolvable=True` namespaces, replacing the plain-argparse style our
> patch's diff targets. Structural rewrite, not a content collision.

> `paged_mqa_logits_backend.py` itself saw zero new commits since the
> 08-28 baseline (still only `7e751153eb`, the pure rename). Current
> `DSA_PAGED_MQA_LOGITS_BACKEND_CHOICES` is still exactly `["auto",
> "deepgemm", "cutedsl", "aiter"]`, no `torch` value added. p30 not
> redundant, design conclusion unchanged. Only the rebase mechanics got
> harder, the server_args.py hunk needs re-porting onto the new Arg/NS
> declaration style, not just textual conflict resolution.

> PR #36507 "GLM-5.3-Flash support" merged 2026-09-06 (`97c6978369`).
> Confirmed via direct diff it does not touch `paged_mqa_logits_backend.py`
> or the indexer-backend dispatch (it rewrites `dsa_backend.py` instead,
> relevant to the companion doc, not this one).

> SGLang released v0.5.19 on 2026-09-05 (a `release/v0.5.19` cherry-pick
> branch, not a main snapshot, confirmed by its `[Cherry-pick to
> release/v0.5.19]`-tagged commits). Confirmed it does NOT contain #29927
> (SM120 DeepGEMM paged-MQA indexer, merged 2026-09-02, apparently too late
> for the branch cut), a cross-reference note for the companion DSV4 doc,
> not this doc's scope. Contains #26928 (SM120 GLM-5.1 baseline, already
> known, unrelated to p30).

> Rebased 2026-09-10 (approved): #31480 rebased onto current upstream/main
> `12771786f2` (2026-09-10). Old head
> `6b2e62e259398b8e34b8eac5d92f6d6a7c9448ff` to new head
> `8d3944adb22eba57e258c7129f54333c7896fcf8`, pushed `--force-with-lease`
> to the vroomfondel fork. Old base was `d56706459c` (2026-08-28), three
> files conflicted.

> Correction to the root cause noted in the entry above: the
> `A[...] / Arg(...) / NS(...)` declarative style itself was NOT new, it
> already existed in server_args.py at the 08-28 baseline (confirmed by
> direct inspection of the baseline file, not the working tree, per this
> doc's own sourcing rule). The actual structural change is `ed82def55f`
> "[Config] Round 6.2: the field declarations move to their namespaces, and
> the record is assembled from them" (#38047, merged after 08-28):
> `dsa_paged_mqa_logits_backend` and its sibling kernel fields moved OUT of
> server_args.py entirely into `python/sglang/srt/arg_groups/fields/exec_.py`
> (class `ExecKernel(msgspec.Struct)`), each field's `NS("exec.kernel")`
> third tuple element dropped since the namespace is now implicit in the
> file. This produced the server_args.py conflict: our small choices-list
> edit sat inside a region upstream deleted wholesale. Resolved by taking
> upstream's server_args.py as-is (no DSA kernel fields remain there) and
> re-porting the choices edit onto the new location in exec_.py, same
> two-element `A[str, Arg(help=..., choices=[...])] = "auto"` shape, just
> adding `"torch"` to the inline list and to the help string.

> `dsa_indexer.py` had a small, unrelated conflict: upstream independently
> changed `seqlens_32_2d = seqlens_32.unsqueeze(-1)` to
> `.contiguous().view(-1, 1)` on the same line our `is_torch()` guard
> touches; kept both (upstream's tensor-op change plus our guard).
> `paged_mqa_logits_backend.py` auto-merged with zero conflict (it already
> reads `get_platform().is_sm100`, from the `7e751153eb` rename noted
> earlier, not the old `is_sm100_supported()`, so nothing needed adapting).
> `dsa_backend.py`'s only conflict was import ordering against the new
> `dsa_backend_kpool.py`/`kpool_plan.py` imports from #36507; resolved by
> alphabetical placement.

> Verification before push: no conflict markers repo-wide (`<<<<<<<`/
> `>>>>>>>` at line start, both scoped and repo-wide passes), `ast.parse`
> clean on all 9 touched/new files, `git range-diff
> d56706459c..6b2e62e259 upstream/main..HEAD` showed only the exec_.py
> relocation and the dsa_indexer.py context shift as content drift (rest
> pure context), a repo-wide scan for module-scope `X = get_platform().Y`
> assignments in the touched files came back empty, own diffstat unchanged
> at 8 files, +1022/-12 (identical to the 08-28 baseline shape). Pushed to
> `git@github.com:vroomfondel/sglang.git` (remote verified before
> pushing). Confirmed live: `gh pr view 31480` reports `headRefOid:
> 8d3944adb22eba57e258c7129f54333c7896fcf8`, `mergeable: MERGEABLE` /
> `mergeStateStatus: BLOCKED` (checks/review gating only, no conflict).
> Content/design conclusions unchanged, this was a submission-mechanics
> rebase only.

> Re-checked 2026-09-16: PR #31480 unchanged in content since the 09-10 rebase
> (head still `8d3944adb22eba57e258c7129f54333c7896fcf8`, `updatedAt`
> 2026-09-10T13:10:42Z, still 3 comments, 0 reviews, no `run-ci` label,
> `reviewDecision: REVIEW_REQUIRED`). A first `gh` read returned transient
> `mergeable: UNKNOWN` / `mergeStateStatus: UNKNOWN`; a repeat read 3s later
> confirmed the stable `MERGEABLE` / `BLOCKED` (the known lazy-cache artifact
> after `upstream/main` moved, not a real state change, same pattern as
> 09-04).

> `DSAPagedMQALogitsBackend` (`paged_mqa_logits_backend.py`) on `upstream/main`
> (checked against `upstream/main` tip `2cbfaefbf9`, 2026-09-16, fetched
> fresh): zero commits since the 09-10 baseline `12771786f2`. `resolve()`
> still gates CuteDSL on `get_platform().is_sm100`, still only DEEPGEMM/
> CUTEDSL/AITER, no `torch` value. The field itself lives in
> `sglang/srt/arg_groups/fields/exec_.py` (confirmed, not a module-level
> constant in `server_args.py`), inline `choices=["auto", "deepgemm",
> "cutedsl", "aiter"]`, still no `torch`. `dsa_indexer.py` reads it via
> `get_exec().kernel.dsa_paged_mqa_logits_backend`. `dsa_backend.py`'s topk-
> backend construction is `DSATopKBackend.resolve(model_runner)` (line 358).
> p30 not redundant, design conclusion unchanged.

> **Local patch cross-check (commit 958c98d, 2026-09-11, deployed as
> `xomoxcc/dgx-spark-sglang:0.5.19-sm121`):** verified p30's re-anchored
> `replace_any` targets directly against the live upstream state fetched this
> cycle, all three accurate: (1) `resolve()`'s SM100 gate spelled
> `get_platform().is_sm100` on `>= v0.5.19`, matches; (2) the
> `DSA_PAGED_MQA_LOGITS_BACKEND_CHOICES` module constant does not exist
> standalone for this field on `upstream/main`, the choices are inline in
> `arg_groups/fields/exec_.py`'s `Arg(choices=[...])`, matches; (3)
> `DSATopKBackend(server_args.dsa_topk_backend)` became
> `DSATopKBackend.resolve(model_runner)` (#36313), matches. p30's
> `replace_any` list correctly covers both the pre-0.5.19 module-constant
> spelling and the >=0.5.19 inline-in-exec_.py spelling.

> `dsa_backend.py` on `upstream/main`: one commit since baseline, `a66451c058`
> ("[GLM-5.3 Flash] Restore and enable KPool metadata fusion", #38845, merged
> 2026-09-12), does not touch the `trtllm-gen`/paged-MQA-logits dispatch this
> doc tracks (0 hits for either term in its diff). No new sglang issue or PR
> found searching "dsa_paged_mqa_logits_backend torch".

> SGLang still at v0.5.19 (2026-09-05), no new release.

> Conclusion unchanged: p30/p35 remain necessary on stock v0.5.19/current main;
> no upstream fix has landed or is imminent. `RETIRED_PATCHES.md` re-checked:
> no p30/p35 entry exists, correctly reflecting still-active status.

> Re-checked 2026-09-22: PR #31480 unchanged in content since the 09-10 rebase
> (head still `8d3944adb22eba57e258c7129f54333c7896fcf8`, `updatedAt`
> 2026-09-10T13:10:42Z, still 3 comments, 0 reviews, no `run-ci` label,
> `reviewDecision: REVIEW_REQUIRED`). `mergeable`/`mergeStateStatus` stable at
> `MERGEABLE`/`BLOCKED` on two reads 3s apart (same as 09-16, no state
> change).

> SGLang released v0.5.20 on 2026-09-18 (commit `94602c9c2b`), superseding
> v0.5.19 (2026-09-05). `DSAPagedMQALogitsBackend`
> (`layers/attention/dsa/paged_mqa_logits_backend.py`) source-verified
> directly against the `v0.5.20` tag: enum still only DEEPGEMM/CUTEDSL/AITER,
> `resolve()` still gates CUTEDSL on `get_platform().is_sm100`, no `torch`
> value. The field's `choices=["auto", "deepgemm", "cutedsl", "aiter"]` in
> `arg_groups/fields/exec_.py` unchanged. `dsa_backend.py`'s topk-backend
> construction `DSATopKBackend.resolve(model_runner)` confirmed at the same
> line 358 as the 09-16 check. p30/p35 remain necessary, not redundant.

> v0.5.20 release notes confirmed to contain #36507 (GLM-5.3-Flash, listed as
> a new model), #29927 (SM120 DeepGEMM paged-MQA indexer) and #36655 (SM120
> exact query-head widths), all three verified via
> `compare/v0.5.20...<merge-sha>` returning `status: behind` for each merge
> commit. Both #29927 and #36655 are explicitly scoped to DeepSeek-V4 in the
> release notes headline ("DeepSeek-V4 on RTX PRO 6000... sparse-MLA indexer
> now runs on DeepGEMM's paged-MQA kernel, replacing the torch fallback"), the
> DSV4 path (PR #24692) already has its own arch fallback, this doc's generic
> `dsa/dsa_indexer.py` path (GLM-5.2/5.3, DeepSeek-V3.2) is untouched. Same
> "DSV4-only, not redundancy-relevant" pattern flagged repeatedly since 08-21.

> **Local patch drift found:** unlike prior cycles, p30 needs real
> re-anchoring for v0.5.20 - `git status` on dgxarley shows
> `roles/k8s_dgx/files/sglang_patches/p30_dsa_torch_backend.py` with an
> UNCOMMITTED working-tree edit (not yet part of a dgxarley commit) that adds
> three v0.5.20-specific `alt_targets`/`replace_any` variants. Source-checked
> all three claims directly against the `v0.5.20` tag, all accurate: (1) the
> server-args patch target moves from `server_args.py` to
> `arg_groups/fields/exec_.py`, consistent with real merged PR #38375
> ("Retire get_global_server_args...", in the v0.5.19..v0.5.20 range); (2) the
> import-anchor cluster in `dsa_backend.py` now reads a `DSAIndexerMetadata`
> import immediately followed by a `DSAMetadataManagementMixin` import
> (verified at lines 60-62 on the tag, exact match to the new variant added);
> (3) the dispatch branch's kernel call is wrapped in a local
> `_chunked_fp8_paged_mqa_logits` helper in `dsa/dsa_indexer.py` instead of
> calling `deep_gemm.fp8_paged_mqa_logits` directly (verified at lines
> 864-896, 952, 966 on the tag). This is report-only verification of an
> already-drafted local patch, not something this audit changed.

> Conclusion unchanged: p30/p35 remain necessary on v0.5.20; no upstream fix
> has landed or is imminent. Local re-anchoring work for the pending
> 0.5.20-sm121 build is already in progress (uncommitted) and verifies
> correct against the tag.

> Re-checked 2026-09-25: PR #31480 unchanged in content since the 09-10 rebase
> (head still `8d3944adb22eba57e258c7129f54333c7896fcf8`, `updatedAt`
> 2026-09-10T13:10:42Z, still 3 comments, 0 reviews, no `run-ci` label,
> `reviewDecision: REVIEW_REQUIRED`). `mergeable`/`mergeStateStatus` stable at
> `MERGEABLE`/`BLOCKED` on two `gh` reads 3s apart, no transient `UNKNOWN` this
> cycle.

> SGLang still at v0.5.20 (2026-09-18, `gh release list` confirms it as
> latest, no new tag). `DSAPagedMQALogitsBackend` (`paged_mqa_logits_backend.py`)
> on `upstream/main` (fetched fresh): zero commits touch the file since the
> v0.5.20 tag; enum still only DEEPGEMM/CUTEDSL/AITER, `resolve()` unchanged.
> `exec_.py`'s `dsa_paged_mqa_logits_backend` field `choices` still exactly
> `["auto", "deepgemm", "cutedsl", "aiter"]`, no `torch` value. `dsa_indexer.py`
> had three commits since v0.5.20 (`0dc8b29e15` ROCm AITER fused FP8 indexer
> writer, `7b67a96640` DSV4 chunk the indexer MQA logits by query rows,
> `afe71f4b9e` process-group-through-runtime-context refactor), none adding an
> arch-independent backend or touching the paged-MQA-logits dispatch this doc
> tracks (first two are ROCm/DSV4-scoped, third is a mechanical accessor
> rename). p30 not redundant, design conclusion unchanged.

> **Deploy confirmed:** dgxarley commit `3214ed2` (2026-09-22) bumped the
> deployed image from `xomoxcc/dgx-spark-sglang:0.5.19-sm121` to
> `0.5.20-sm121`. Live head pod (`sglang-head-68c7bbf746-kfwlt`, created
> 2026-09-23T07:15:21Z, namespace `sglang`) confirmed running
> `xomoxcc/dgx-spark-sglang:0.5.20-sm121` (`kubectl get pod -o jsonpath`).
> `kubectl logs -c sglang` confirms p30/p35's re-anchored 0.5.20 targets
> (flagged uncommitted on 09-22) applied cleanly with no `ANCHOR-DRIFT`
> warning: "Patched paged_mqa_logits_backend.py: DSA torch-backend enum +
> resolve('torch')", "Patched exec_.py: DSA torch-backend CLI choice",
> "Patched dsa_backend.py: DSA torch-backend wiring", "Patched dsa_indexer.py:
> DSA torch-backend dispatch", "Patched torch_paged_mqa_logits.py: torch
> indexer fallback: Triton fast-path dispatch (env SGLANG_DSA_INDEXER_TRITON)".
> The 09-22 re-anchoring work is therefore verified correct in production, not
> just against the tag.

> Conclusion unchanged: p30/p35 remain necessary on v0.5.20/current main; no
> upstream fix has landed or is imminent.

> Re-checked 2026-09-28: PR #31480 unchanged in content since the 09-10 rebase
> (head still `8d3944adb22eba57e258c7129f54333c7896fcf8`, `updatedAt`
> 2026-09-10T13:10:42Z, still 3 comments, 0 reviews, no `run-ci` label,
> `reviewDecision: REVIEW_REQUIRED`). `mergeable`/`mergeStateStatus` stable at
> `MERGEABLE`/`BLOCKED` on two `gh` reads 3s apart, no drift since 09-25. Fork
> remote branch head (`git ls-remote origin`) matches the PR head exactly.

> SGLang still at v0.5.20 (2026-09-18, latest per `gh release list`, no new
> tag). `paged_mqa_logits_backend.py` on `upstream/main` (fetched fresh, tip
> `df378bca42`): zero commits since v0.5.20; still only DEEPGEMM/CUTEDSL/
> AITER, no `torch` value. `exec_.py`'s `dsa_paged_mqa_logits_backend` choices
> still exactly `["auto", "deepgemm", "cutedsl", "aiter"]`. `dsa_backend.py`:
> zero commits. `dsa_indexer.py` had one cosmetic import-reorg commit
> (`38ec649048`, #41243, merged 2026-09-26), zero dispatch-logic change. Two
> other DSA commits since 09-25 (`4bb645b8e2` #41311, `63d320c723` #40854)
> touch only `kpool_prefill_cuda_graph.py` / `dsa_indexer_kpool.py`
> (GLM-5.3-Flash KPool path), not the generic paged-MQA-logits dispatch this
> doc tracks. p30 not redundant, design conclusion unchanged.

> Dry-run rebase check (`git merge-tree --write-tree upstream/main
> dsa-indexer-torch-triton-backend`, no push, no worktree touched): clean
> merge tree, no conflicts. No rebase needed right now, the PR already
> reports `MERGEABLE`.

> Conclusion unchanged: p30/p35 remain necessary on v0.5.20/current main; no
> upstream fix has landed or is imminent. No factual errors found in the
> 09-25 entry.

> Re-checked 2026-10-03: PR #31480 unchanged in content since the 09-10 rebase
> (head still `8d3944adb22eba57e258c7129f54333c7896fcf8`, `updatedAt`
> 2026-09-10T13:10:42Z, no new comments or reviews, `reviewDecision:
> REVIEW_REQUIRED`). `mergeable`/`mergeStateStatus` stable at
> `MERGEABLE`/`BLOCKED` on two `gh` reads 5s apart. Fork remote branch head
> (`git ls-remote origin`) matches the PR head exactly.

> SGLang v0.5.21 was tagged 2026-10-02 (now latest; v0.5.20 was 2026-09-18).
> `upstream/main` (fetched fresh, tip `7be5e3473c`): `paged_mqa_logits_backend.py`
> has no commits since 09-28; `exec_.py`'s `dsa_paged_mqa_logits_backend`
> choices are still `["auto", "deepgemm", "cutedsl", "aiter"]`, no `torch`.
> Since 09-28 `dsa_backend.py` took `42f2e4fcb7` (#41987, k-pool indexer CPU
> seq_lens sync removal, 2 lines) and `3ed6367d3f` (#41337, FlashMLA KV format
> naming, +3 lines), plus `e9b0d0c0a3` (#40911, AMD CUDA gating) and
> `7889fbab9a` (#32196, EAGLE DP metadata). None changes the generic
> paged-MQA-logits dispatch this doc tracks. The remaining DSA/indexer
> commits are AMD gfx950 (#38583, #41488), NPU DCP (#37787) and QSA/Qwen
> (#39721, #39893), out of scope. p30 not redundant, design conclusion
> unchanged.

> Dry-run rebase check (`git merge-tree --write-tree upstream/main
> dsa-indexer-torch-triton-backend`, no push, no worktree touched): clean
> merge tree, no conflicts despite the `dsa_backend.py` edits above. No
> rebase needed right now, the PR already reports `MERGEABLE`.

> Conclusion unchanged: p30/p35 remain necessary on current main; no upstream
> fix has landed or is imminent. No factual errors found in the 09-28 entry.

> Re-checked 2026-10-11: PR #31480 unchanged (head `8d3944adb22eba57e258c7129f54333c7896fcf8`,
> `updatedAt` 2026-09-10T13:10:42Z, no new comments or reviews, `REVIEW_REQUIRED`,
> `MERGEABLE`/`BLOCKED` on two reads 5s apart; fork branch head matches). No new SGLang tag
> (v0.5.21 still latest); `upstream/main` tip `ad8b1e1f85` pins `sgl-deep-gemm==0.2.2`.
> `paged_mqa_logits_backend.py` has no commits since 2026-08-30; `exec_.py` choices still
> `["auto", "deepgemm", "cutedsl", "aiter"]`; `dsa_indexer_kpool.py` upstream still calls
> `deep_gemm.fp8_paged_mqa_logits` unconditionally. New k-pool/DSA commits (#42178, #42671,
> #42750, #42285, #42793, #43100, #42957) do not touch the generic dispatch. Dry-run
> `git merge-tree --write-tree upstream/main dsa-indexer-torch-triton-backend`: clean, no
> rebase needed.

> Local correction to the "DeepGEMM asserts on SM120/121" premise: image 0.5.21-sm121 ships
> sgl-deep-gemm 0.2.0 whose paged and ragged FP8 MQA logits kernels run on SM121 (kernel-level
> GPU verification on spark5 2026-10-08, ~1e-7 relative error, end-to-end unvalidated; see
> p69 header and `roles/k8s_dgx/defaults/main/sglang.yml`). So on 0.5.21+ the `torch`/Triton
> backend is no longer a crash fix but an opt-in fallback and a decode-speed option; the
> PR body's Problem section below describes pre-0.2.x DeepGEMM and would need the same
> qualifier before any further push. p30/p35 stay for older images and as the A/B lever. New
> local patch p69 routes `IndexerKPool._get_topk_paged` (GLM-5.3-Flash, `index_kpool=4`) to
> the same torch/Triton path; this call site is outside #31480's scope (the PR only wires
> `dsa_indexer.py`), so the `torch` value silently stays on DeepGEMM for k-pool models
> without p69. Candidate follow-up if the PR is pursued.

> Re-checked 2026-10-12: PR #31480 unchanged (head `8d3944adb22eba57e258c7129f54333c7896fcf8`,
> `updatedAt` 2026-09-10T13:10:42Z, no new comments or reviews, `REVIEW_REQUIRED`,
> `MERGEABLE`/`BLOCKED`; fork branch head matches). `git merge-tree --write-tree upstream/main`
> is clean against `f006fb49ec`. No new SGLang tag (v0.5.21 still latest). Since 2026-10-11
> upstream/main only moved flashinfer to 0.7.1 (#43494); `sgl-deep-gemm` stays 0.2.2 and no
> commit touched `dsa/paged_mqa_logits_backend.py`, `dsa_indexer.py` or `dsa_indexer_kpool.py`.
> Correction to the 2026-10-11 kpool note: `IndexerKPool` has resolved
> `DSAPagedMQALogitsBackend` since #39341 (2026-09-22, gfx950), but `_get_topk_paged` only
> branches aiter, then tilelang, else `deep_gemm.fp8_paged_mqa_logits`; there is still no
> torch/Triton or cutedsl route, so the p69 deletion condition is not met.
> Local: p67 was retired on 2026-10-11 (d2ae7b2, now listed in RETIRED_PATCHES.md); DSA_speedup.md and deepgemm_fix.md carry 2026-10-11 update banners stating the DeepGEMM SM121 block is gone; p30, p35, p68, p69 unchanged.

> Re-checked 2026-11-10: PR #31480 unchanged (head `8d3944adb22eba57e258c7129f54333c7896fcf8`,
> `updatedAt` 2026-09-10T13:10:42Z, no reviews, last comment 2026-08-03, `REVIEW_REQUIRED`; fork
> branch head matches). `git merge-tree --write-tree upstream/main` now CONFLICTS in
> `dsa_backend.py` (three hunks: cuda-graph schedule-metadata refresh and both
> `paged_mqa_schedule_metadata` build sites) against `170804b40a`. Cause: #32947 (merged
> 2026-10-13 as `b3991b53da`, in v0.5.22 and v0.5.23) routes the dsa indexer's DeepGEMM entry
> points through `resolve_paged_mqa_logits_metadata_fn` / `resolve_fp8_paged_mqa_logits_fn` /
> `resolve_num_sms` in `dsa/utils.py`; with `SGLANG_FP8_PAGED_MQA_LOGITS_TORCH=1` they use the
> dsv4 torch fallback (no `next_n > 1`, so no MTP, and no Triton fast path). That is an upstream
> env-flag escape hatch for the same problem, but not the `torch` backend value this PR adds;
> `exec_.py` choices are still `["auto", "deepgemm", "cutedsl", "aiter"]`. A rebase onto the
> resolvers is needed before any further push (not done). New tags: v0.5.22 (2026-10-13) pins
> flashinfer 0.7.1 / sgl-deep-gemm 0.2.2, v0.5.23 (2026-10-28) pins 0.7.2 / 0.2.3.post2; main
> pins 0.7.3 / 0.2.7. `IndexerKPool._get_topk_paged` still branches only aiter, tilelang, else
> `deep_gemm.fp8_paged_mqa_logits`, with no torch/triton/cutedsl route and no resolver use, so
> the p69 deletion condition is not met. Open #49098 (FlashInfer fused SM120 indexer top-k) is
> a different approach to the SM12x indexer. Local image 0.5.21-sm121 is unchanged.

> [DSA] Add an arch-independent `torch` paged-MQA-logits backend with a fused
> Triton fast path (unblocks DSA models on SM120/SM121)

## PR body draft (English)

### Problem

The DSA indexer's paged-MQA-logits kernel has no backend that runs on consumer
Blackwell (or any arch outside the current three):

- DEEPGEMM (the `auto` default): compiled C++ `Unsupported architecture` assert
  on SM120/121; DeepGEMM upstream declined SM12x support (PR #318).
- CUTEDSL: gated `is_sm100_supported()`, and structurally SM100-only —
  `_setup_mma` uses `tcgen05.MmaF8F6F4Op`, a datacenter-Blackwell ISA op.
- AITER: ROCm.

So every DSA model (DeepSeek-V3.2 family, GLM `GlmMoeDsaForCausalLM`) crashes
on its first decode step on SM120/121. The dsv4 path already solved this for
DeepSeek-V4 (`fp8_paged_mqa_logits_torch_sm120`, PR #24692), but the generic
`dsa/dsa_indexer.py` path has no equivalent.

### Changes

1. **New backend value `torch`** in `DSAPagedMQALogitsBackend` + server args
   (`--dsa-paged-mqa-logits-backend torch`). Opt-in only, NOT selected by
   `auto` — archs where DeepGEMM/CuteDSL work are unaffected.
2. **New module `dsa/torch_paged_mqa_logits.py`**: a vectorized, cuda-graph-safe
   (no `.item()`, no data-dependent control flow) pure-torch port of the dsv4
   fallback for the generic DSA path. Discards the DeepGEMM schedule metadata
   (`_ = deep_gemm_metadata`) — the torch path does no SM-tiled scheduling, so
   the two eager `get_paged_mqa_logits_metadata` call sites in
   `dsa_backend.py::init_forward_metadata` (and the graph-replay refresh
   helper) are skipped for this backend.
3. **Dispatch in `dsa_indexer.py::_get_topk_paged`**: pure pass-through for ALL
   modes — decode, target_verify, draft_extend(_v2). No reshaping is needed:
   q/weights are already sliced per token (`[:q_offset]`), verify seqlens come
   from `get_seqlens_expanded()` (per-token), and `init_forward_metadata`
   already `repeat_interleave`s the page table to per-token rows for every
   multi-token mode. (We learned this live: an extra repeat in the dispatch
   DOUBLE-expands and trips the kernel's shape assert at the MTP warmup.)
4. **Triton fast path `dsa/triton_paged_mqa_logits.py`** (env-gated,
   `SGLANG_DSA_INDEXER_TRITON`, default on; falls back to torch when triton is
   unavailable or num_heads < 16): one program per (token, 64-KV-page), STATIC
   launch grid over the full page-table width (cuda-graph-safe) with per-block
   EARLY EXIT on the true `seq_lens[b]` read at replay time; fused fp8 load +
   inline-scale dequant + q·k dot + relu + weighted head-sum, no fp32 HBM
   intermediates. Motivation: under cuda-graph the page-table width is a
   CAPTURE constant, and the pure-torch kernel pays the full width every step
   — 1.476 ms/layer at width 131072 regardless of the true seq len (~116 ms of
   a ~119 ms token on a 78-layer model). The Triton kernel is bit-exact vs the
   torch path and 61x at that shape (0.024 ms).

### Evidence (DGX Spark GB10, SM121)

- torch path: numerically verified (dominant-KV-slot unit tests, masking,
  no NaN/all-zero) and live-proven across decode, prefill, target-verify and
  draft-extend (MTP/NEXTN: accept ~2.1, GSM8K 85% @ conc 8, 0 errors).
- Triton path: bit-exact vs torch (identical -inf masks; max|diff| 0.0 direct,
  3.6e-7 through the module) across bs 1/4/32, seq 300/2048/131072, width
  131072; graph capture+replay correct INCLUDING a seq-len change in the
  static buffer between capture and replay; folded verify batches (B=4,
  next_n=4, mixed ctx 64..1900) bit-exact vs a per-token-loop reference.

### Notes for reviewers

- The kernel hardcodes the DSA cache layout (head_dim=128, block 64,
  64*128 fp8 + 64 fp32 scales per block) and asserts it.
- `clean_logits` handling matches the existing backends (cleaning happens in
  topk_transform).
- The Triton kernel needs `num_heads >= 16` (tl.dot minimum); GLM=32,
  DSv3.2=64 — the guard falls back to torch below that.
- Perf floor context for SM121 reviewers: with this backend + the sparse-MLA
  routing PR, a 504B GLM-5.2 REAP serves at 8.4 tok/s single-stream (decode is
  then bounded by unquantized bf16 attention projections, not by DSA code).

## Local mapping

| local                             | upstream file                                                                   |
|-----------------------------------|---------------------------------------------------------------------------------|
| p30 enum/server_args edits        | `dsa/paged_mqa_logits_backend.py`, `server_args.py`                             |
| p30 new file                      | `sglang/srt/layers/attention/dsa/torch_paged_mqa_logits.py`                     |
| p30 dsa_backend/dsa_indexer edits | `dsa_backend.py`, `dsa/dsa_indexer.py`                                          |
| p35 new file + dispatch           | `sglang/srt/layers/attention/dsa/triton_paged_mqa_logits.py` + the torch module |

Full local chronology: `dsalogitrework.md` (PART 1 = port plan, PHASE 2, p35
LIVE RESULT, MTP LIVE RESULT).

## Submission checklist (recherchiert 2026-07-16)

**Unit-Tests: JA, zwingend und mit direktem Präzedenzfall.** Die Contribution-
Guide (docs_new/docs/developer_guide/contribution_guide.mdx) verlangt Tests für
jedes Feature, und PR #24692 (der dsv4-Fallback, den p30 portiert) shippte
`test/registered/kernels/test_sm120_paged_mqa_logits.py` (314 Z.) — läuft auf
JEDER CUDA-GPU ("no SM120 hardware required"), registriert via
`register_cuda_ci(est_time=20, stage="base-b", runner_config="1-gpu-small")`.

**Unser Test** (neu, `test/registered/kernels/test_dsa_paged_mqa_logits.py`,
modelliert auf dem Präzedenzfall; Layout-Konstanten identisch — 64er-Pages,
head_dim 128, 8448 B/Page):
1. torch-Fn vs. loopy Referenz (Numerik + -inf-Masken),
2. Triton vs. torch **bit-exakt** (inkl. identischer -inf-Masken),
3. beide KV-dtype-Views (uint8 / float8_e4m3fn) — der historische
   Garbled-Output-Bug aus dem Präzedenztest,
4. variable per-Batch seq_lens + Masking-Semantik,
5. **per-Token-Shapes (verify/next_n>=2)**: expanded seqlens + per-Token-
   page_table-Zeilen (die Doppel-Expansions-Lektion als Regressionstest),
6. cuda-graph capture + replay, inkl. seq-len-Änderung im statischen Buffer
   zwischen Capture und Replay,
7. num_heads<16-Fallback-Guard (tl.dot-Minimum).
Alles CI-lauffähig auf H100-Runnern (torch + Triton sind arch-generisch);
Entwicklung/Vorvalidierung auf spark5 möglich (podman-Methode).

**Mechanik:**
- Fork + Branch; die p30/p35-Anker in ECHTE Diffs gegen main portieren.
  ACHTUNG: main hat die Kernel-Verzeichnisse restrukturiert
  (`python/sglang/kernels/ops/attention/...` statt `srt/layers/attention/...`
  für Teile) — Zielpfade gegen main verifizieren, nicht gegen v0.5.15.
- `pre-commit run --all-files` (Pflicht laut Guide; ggf. zweimal laufen lassen).
- CI-Registrierung des Tests via `register_cuda_ci` (Stage/Runner wie Präzedenz).
- PR-Body aus diesem Dokument; Companion-PR verlinken.
- Offen zu prüfen beim Einreichen: DCO/Sign-off-Pflicht (in der Guide nicht
  gesehen, beim ersten Push gegen die PR-Checks verifizieren).

## Review-Lektionen aus dem Präzedenz-PR #24692

Siehe den ausführlichen Abschnitt im Companion-Dokument
(UPSTREAM_SGLANG_DSA_SM12X_NATIVE_SPARSE_MLA.md): #24692 brauchte 24 Tage,
27 Commits, 64 Review-Kommentare. Für DIESEN PR besonders relevant: kanonische
Arch-Utils statt roher Checks, `-1`-Sentinel explizit testen UND kommentieren,
kein stilles Durchfallen für nicht unterstützte Pfade (next_n-Randfälle,
CP-Zweige), Assert-Messages beschreibend halten, Scope eng (nur das
Indexer-Backend, kein Beifang).


## STATUS 2026-07-16: Branch gebaut + GPU-validiert, bereit zum Einreichen

> [Historischer Schnappschuss von VOR dem Einreichen — noch am selben Tag als
> PR #31480 gepusht/gefiled (siehe Status oben); "NICHT gepusht" unten gilt
> nicht mehr. Seit 2026-08-03 zudem auf aktuellen main rebased (`5480d1d28d`).]

Branch `dsa-indexer-torch-triton-backend` im Fork-Worktree
`../sglang-wt-indexer` (von upstream/main 1f34911de7), EIN Commit
`d0b442f14f`, pre-commit clean, NICHT gepusht. Test
`test/registered/kernels/test_dsa_paged_mqa_logits.py`: **17/17 auf GB10**
(inkl. Verify-per-Token-Shapes, Doppel-Expansions-Regressionstest,
Graph-Capture/Replay). Env-Gate sauber in upstreams environ.py-Registry.
Bemerkenswerte main-Drift-Funde im Port-Report: dsa_indexer resolved das
Backend inzwischen selbst; ForwardMode hat kein non-v2 draft_extend mehr
(toter upstream-Zweig).