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

| local | upstream file |
|---|---|
| p30 enum/server_args edits | `dsa/paged_mqa_logits_backend.py`, `server_args.py` |
| p30 new file | `sglang/srt/layers/attention/dsa/torch_paged_mqa_logits.py` |
| p30 dsa_backend/dsa_indexer edits | `dsa_backend.py`, `dsa/dsa_indexer.py` |
| p35 new file + dispatch | `sglang/srt/layers/attention/dsa/triton_paged_mqa_logits.py` + the torch module |

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