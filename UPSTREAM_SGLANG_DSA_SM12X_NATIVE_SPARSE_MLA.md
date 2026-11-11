# UPSTREAM PR: DSA sparse attention on SM120/SM121 via flashinfer's native sparse-MLA backend

Status: **FILED 2026-07-16 als sgl-project/sglang PR #31481** (https://github.com/sgl-project/sglang/pull/31481). Local implementation =
`roles/k8s_dgx/files/sglang_patches/p34_dsa_trtllm_sparse_sm120.py`, live-proven
on the dgxarley cluster (4× DGX Spark GB10, SM121, TP4,
`0xSero/glm-5.2-reap-504B-v2` = GlmMoeDsaForCausalLM). After a merge lands in a
release we build on: DELETE p34 (its docstring carries the same re-sync rule).

Upstream state (checked 2026-07-16): `sglang/srt/layers/attention/dsa_backend.py`
on `main` still hardcodes `backend="trtllm-gen"` in `_forward_trtllm`; no cc-12x
branch exists anywhere in the DSA path.

Re-checked 2026-07-23: PR #31481 still **OPEN**, `reviewDecision:
REVIEW_REQUIRED`, `mergeStateStatus: DIRTY` — no human review yet (only a
2026-07-16 gemini-code-assist bot comment reporting its own quota limit).
`dsa_backend.py` on `main` is unchanged (still hardcodes `backend="trtllm-gen"`).
SGLang released v0.5.15 (2026-07-10) and v0.5.15.post1 (2026-07-14) since the
PR was filed; neither touches this code path.

Re-checked 2026-07-28: PR #31481 still **OPEN**, no new commits or comments
since 2026-07-16 (`updatedAt` unchanged). GitHub now reports `mergeable:
CONFLICTING` alongside `mergeStateStatus: DIRTY`, meaning the branch needs a
**rebase against current main** before it can be submitted, main has moved on
since the `1f34911de7` base. `dsa_backend.py::_forward_trtllm` fetched again
from `main` and confirmed unchanged (still hardcodes `backend="trtllm-gen"`,
line ~2967). `calculate_mla_kv_cache_dim` (now in
`mem_cache/kv_cache_configurator.py`, matching this doc's own port note) still
gates the plain-layout early return purely on `dsa_prefill_backend` /
`dsa_decode_backend == "trtllm"`, no SM12x branch. SGLang v0.5.16 (released
2026-07-25) reviewed, its DeepSeek-V4/DSA/GLM-5.2 changelog entries do not
touch either file's SM12x gating logic. No change to this doc's conclusions,
rebase is a submission-mechanics item, not a content one.

Re-checked 2026-08-07: the rebase flagged above is DONE — #31481 was rebased
onto current main and force-pushed 2026-08-03 ~11:15 UTC (single commit, new
head `0864022c3d`). Author comment at ~13:03 UTC: "Rebased onto current main
and re-validated: 11/11 CI-safe unit tests, 5/5 SM121 kernel tests on GB10
(kwarg semantics checked against the pinned flashinfer 0.6.15.post1 sources,
live-tested on 0.6.16). Could a maintainer add the `run-ci` label so the
suites actually run? Review welcome. Companion PR: #31480." (Companion
#31480 was rebased the same day, head `685784f5ae` — both PRs back in sync.)
As of 2026-08-07: **no maintainer response, no `run-ci` label** (only the
pre-existing `deepseek` label, on the PR since 2026-07-16), no reviews;
`mergeable: MERGEABLE`, `mergeStateStatus: BLOCKED`, `reviewDecision:
REVIEW_REQUIRED`, no new pushes. `dsa_backend.py::_forward_trtllm` re-fetched
from `main`: still hardcodes `backend="trtllm-gen"` (line ~3264, drifted from
~2967 by unrelated merges). flashinfer `mla/_sparse_mla_sm120.py` unchanged
(still `@supported_compute_capability([120, 121])`). SGLang still at v0.5.16,
flashinfer stable at v0.6.16.post2 (2026-08-06, tvm-ffi ABI hotfix only).

Re-checked 2026-08-09: no change — still **no maintainer response, no
`run-ci` label** (only the pre-existing `deepseek` label), no reviews, no
new pushes on #31481 (REST API: `mergeable: true`, `mergeable_state:
"blocked"`, `updated_at` still 2026-08-03T13:03:37Z; GraphQL transiently
reports `UNKNOWN` merge state — lazy-cache artifact after the **v0.5.17
release merge wave**, SGLang v0.5.17 released 2026-08-08, not a real
event). `dsa_backend.py::_forward_trtllm` on `main` still hardcodes
`backend="trtllm-gen"` (now line ~3268, continued drift from unrelated
merges); flashinfer `mla/_sparse_mla_sm120.py` unchanged
(`@supported_compute_capability([120, 121])` at all 4 sites). flashinfer
stable is now **v0.6.16.post3** (2026-08-08; single change = SM90 CUTLASS
MoE backend revert #4412, unrelated to sparse MLA). Neither v0.5.17 nor
post3 changes any conclusion here.

Re-checked 2026-08-15: PR #31481 still **OPEN**, no new commits or comments
since 2026-08-03, `deepseek` label only, `reviewDecision: REVIEW_REQUIRED`,
`mergeable: true` / `"blocked"` (no rebase needed). `dsa_backend.py::
_forward_trtllm` on `main` still hardcodes `backend="trtllm-gen"` at line
~3268 (no drift since 08-09). `flashinfer/mla/_sparse_mla_sm120.py` DID
change: flashinfer PR #4380 (consolidate DSV4 sparse MLA top-k 192/256
support) merged 2026-08-08T11:33Z (landed in v0.6.16.post4 / v0.6.17), but
the diff is scoped entirely to `_DECODE_DSV4_DISPATCH` plus a new
`ValueError` for unsupported decode shapes; it does NOT touch
`_DECODE_DSV3_2_DISPATCH` (the GLM_NSA path this PR needs), the four
`@supported_compute_capability([120, 121])` decorators, or model_type
resolution. So correct the standing claim from "file unchanged" to
"unchanged in the parts relevant to GLM/DSv3.2". flashinfer stable is now
v0.6.17 (2026-08-11); release notes have nothing touching DSv3.2/GLM_NSA
sparse-MLA dispatch.

**MAJOR finding** (flag prominently, do NOT assert as fact until tested):
upstream already solved this problem via a different code path, and it is
in the v0.5.17 release. SGLang PR #26928 ("SM120 (Blackwell Desktop)
support for GLM-5.1 inference", merged 2026-07-28T21:52:34Z) adds an
opt-in DSA backend value `flashinfer_sparse_mla` (`dsa_backend.py`,
`_forward_flashinfer_sparse_mla`) calling
`flashinfer.mla.trtllm_batch_decode_with_kv_cache_mla` with
`backend="sparse"`, `kv_scale_format="arbitrary_fp32"`, functionally the
same flashinfer API and kwargs our PR #31481 proposes patching into
`_forward_trtllm`, wired as a separate named backend instead of changing
the default. Restricted to `GlmMoeDsaForCausalLM` + SM12x + FP8 E4M3 KV +
page size 64, exactly our deployed model class
(`0xSero/glm-5.2-reap-504B-v2`). Auto-selection wiring exists on `main` AND
on the v0.5.17 tag (`sglang/srt/arg_groups/overrides.py::
_dsa_split_backend_resolution`): for model_arch `GlmMoeDsaForCausalLM` +
compute major 12 + `kv_cache_dtype fp8_e4m3` (non-ROCm), unless the user
explicitly overrides `--dsa-prefill-backend`/`--dsa-decode-backend`, sglang
automatically sets both to `flashinfer_sparse_mla`, bypassing the
hardcoded `trtllm-gen` crash path by default for our exact configuration.
Follow-up PR #33075 (merged 2026-08-11) fixed a HiSparse-validator
conflict blocking this backend when `--enable-hisparse` is set, with live
end-to-end validation on 2x8 RTX PRO 6000 (SM120) running GLM-5.2-NVFP4
PD-disaggregated ("Set DSA backends for GLM FP8 KV Cache on SM120/SM121:
prefill=flashinfer_sparse_mla, decode=flashinfer_sparse_mla").

Honest accounting: every re-check from 07-28 through 08-09 missed this
because they only re-verified that the trtllm-named backend
(`_forward_trtllm`, hardcoded `backend="trtllm-gen"`) was unchanged, which
was true but irrelevant since GLM-on-SM12x traffic is auto-routed around
that function entirely as of 07-28.

Implication (flag, do NOT assert as fact): a stock v0.5.17 image may
already auto-select a working SM120/121 sparse-MLA path for our model
without our p34 patch. A GB10 live test of unpatched v0.5.17 against the
deployed model is required before concluding p34 is redundant (test
pending explicit approval). PR #31481 may then be revised, closed in favor
of #26928/#33075, or kept for coverage of the plain trtllm backend name
for non-GLM DSA models / DeepSeek-V3.2 family, which the
`is_glm_sm12_fp8` arm does not cover.

Follow-up same day (2026-08-15, approved): **GB10 test executed on spark5**
(podman, stock image `xomoxcc/dgx-spark-sglang:0.5.17-sm121` with NO
`/patches` mount, verified patch-free via p34 marker grep; shrunk
`0xSero/glm-5.2-reap-504B-v2` config `num_hidden_layers` 78 -> 7 keeping
dense layers 0-2 + a complete full/shared indexer group at layers 3-6;
`--load-format dummy`, TP1, `--kv-cache-dtype fp8_e4m3`, profile flags
mirrored EXCEPT no `--dsa-prefill-backend`/`--dsa-decode-backend` and no
speculative args). Results:
- Auto-selection fired verbatim: "Set DSA backends for GLM FP8 KV Cache on
  SM120/SM121: prefill=flashinfer_sparse_mla, decode=flashinfer_sparse_mla"
  and `server_args` confirmed both fields. It is not just a default: the
  validator `_validate_flashinfer_sparse_mla_backend`
  (`sglang/kernels/ops/attention/flash_mla_sm120.py`) FORBIDS any other
  backend for GlmMoeDsa + SM12x + fp8 KV, so the selection is exclusive.
- Healthy in ~2m20s (weight load 2.97s, MoE autotune ~1m37s first-run,
  decode CUDA-graph capture 3.05s, no crash, matching p34's
  "captures directly under cuda-graph" claim).
- 4 requests all HTTP 200: 3 short (max_new_tokens 32, finish_reason
  length, garbage text as expected with dummy weights) + 1 long prefill
  (prompt_tokens 4001, prefill 516 tok/s, decode ~5 tok/s). Zero
  traceback/NaN/crash matches in the full log.
- Verdict: stock v0.5.17 auto-selects AND successfully runs
  flashinfer_sparse_mla on SM121 for our model class, **supporting p34
  redundancy** for the DSA prefill/decode backend. Caveats: TP1, dummy
  weights, 7-layer shrink (code-path validation, not quality); the real
  confirmation is a TP4 cluster run with real weights (approval-gated,
  not done). p34 retirement decision therefore still pending.
- Side finding for the indexer doc: stock v0.5.17's
  `--dsa-paged-mqa-logits-backend` accepts only `auto|deepgemm|cutedsl|
  aiter`; the profile's `torch` value (added by p30) does not parse on
  stock, so p30 is NOT redundant. With `auto` the run was crash-free
  (no deepgemm "Unsupported architecture" assert), but the resolved
  indexer backend is not logged at INFO level (inconclusive which one ran).
  Test artifacts kept at spark5:/root/gb10-test-p34/ (configs + logs only).

Re-checked 2026-08-21: PR #31481 unchanged since 08-03, no new commits,
comments or reviews (REST API: `mergeable: true`, `mergeable_state:
"blocked"`, `updated_at` still 2026-08-03T13:03:37Z, head still
`0864022c3d`). Only the pre-existing `deepseek` label, `reviewDecision:
REVIEW_REQUIRED` still. `dsa_backend.py::_forward_trtllm` on
`upstream/main` still hardcodes `backend="trtllm-gen"` at line ~3268,
unchanged position and content since 08-15 (zero commits touch that
function). SGLang still at v0.5.17, no new release. flashinfer stable
still v0.6.17 (2026-08-11); only nightly builds released since
(`v0.6.18-2026081x` series), no new stable tag.

New finding (does not change the redundancy question, but is new upstream
activity in a file this doc tracks): sglang PR #32779 "[SM120&90] Add CUDA
fused Triton sparse-MLA prefill backend for DSA" (opened 2026-07-29 by
yunyang1999, not previously flagged in this doc, still OPEN) adds an
opt-in `--dsa-prefill-backend triton_sparse_mla` and widens
`_validate_flashinfer_sparse_mla_backend` in
`python/sglang/kernels/ops/attention/flash_mla_sm120.py`, the exact
validator our 08-15 GB10 test exercised, from `selected -
{"flashinfer_sparse_mla"}` to `selected - {"flashinfer_sparse_mla",
"triton_sparse_mla"}`. Per the PR's own code comment, `flashinfer_sparse_mla`
stays the auto-selected default; `triton_sparse_mla` is only a validated
alternative a user may select explicitly, so this does not change our
finding that `flashinfer_sparse_mla` is auto-selected by default for
GlmMoeDsa + SM12x + fp8 KV. As of today the PR is `mergeable: false`,
`mergeable_state: "dirty"` (needs rebase), labels
performance/run-ci/jit-kernel/GLM, review discussion active (maintainer
nvpohanh asked 2026-08-12 whether the dispatch could be simplified, author
yunyang1999 responded 2026-08-17 that a refactor folding
`triton_sparse_mla` into the existing flashmla_sparse family is in
progress). Not merged, not close to merge; flagged as upstream activity to
watch, not a redundancy-relevant change.

Also checked and ruled irrelevant: sglang PR #33022 "[ROCm] Use the AITER
sparse-MLA kernel for DSA prefill and decode" (open, ROCm-specific, does
not touch the SM12x/flashinfer_sparse_mla path) and flashinfer PR #4551
"fix(sm120): explain sparse-MLA decode dispatch misses; add config query
API" (open, not merged, a follow-up to #4380 that only improves error
diagnostics for the DSV4 decode-dispatch path `_DECODE_DSV4_DISPATCH`,
explicitly does not touch `_DECODE_DSV3_2_DISPATCH`, i.e. the GLM_NSA path
this PR needs).

`_dsa_split_backend_resolution` in `sglang/srt/arg_groups/overrides.py`
re-verified unchanged (function body identical); the only overrides.py
edits in this window, from the environ.py cleanup PRs #35060 and #34926,
touch unrelated cutlass-MoE and megamoe env hooks, not the DSA
auto-selection logic.

p34 retirement decision remains pending: no TP4/real-weight confirmation
run has been done since 08-15 (approval-gated, not requested this cycle).
No upstream change alters the standing 08-15 verdict (stock v0.5.17
auto-selects and runs flashinfer_sparse_mla; p34 redundancy supported at
TP1/dummy-weight level only).

Re-checked 2026-08-28: PR #31481 head unchanged (`0864022c3d`), `updated_at`
unchanged (2026-08-03T13:03:37Z), no new comments (still 2) or reviews,
only the `deepseek` label, `reviewDecision: REVIEW_REQUIRED`. REST now
reports `mergeable: true` / `mergeable_state: "unstable"` (was `"blocked"`
on 08-21), a status-check-noise flip, not a content change (no `run-ci`
label, so checks stay skipped/pending). `dsa_backend.py::_forward_trtllm`
on `upstream/main` re-fetched: still hardcodes `backend="trtllm-gen"`, now
line 3269 (drifted 1 line from 3268, unrelated commits); `_forward_
flashinfer_sparse_mla` still present separately at line 2766.
`overrides.py::_dsa_split_backend_resolution` diffed directly between the
08-21-era commit (`af39ad9349`) and current `upstream/main`: zero lines of
the function changed (confirmed via `git diff` on `overrides.py` showing 0
hits on `_dsa_split_backend_resolution`/`is_glm_sm12_fp8`/
`flashinfer_sparse_mla`), despite 18 other commits touching the file in
this window (ongoing ServerArgs config-declaration refactor plus new model
support, e.g. MiniCPM-SALA, Ling-3.0-flash, DeepEPv2). The `is_glm_sm12_fp8`
auto-selection (GlmMoeDsaForCausalLM + SM12x + fp8_e4m3 KV, non-ROCm) still
unconditionally sets both prefill/decode to `flashinfer_sparse_mla` unless
the user overrides. `flash_mla_sm120.py::_validate_flashinfer_sparse_mla_
backend` still present (now line 572). One functional (non-validator)
change landed in that file: PR #35116 "allocate the page-split buffer
outside inference mode" (merged 2026-08-25) fixes a CUDA-graph-capture
mutation bug in `_split_kv_pages_to_64`; does not touch the validator or
auto-selection dispatch.

PR #32779 has real forward progress since 08-21. A second commit,
`2a625a4396` "Shrink the DSA prefill dispatch surface for
triton_sparse_mla", was pushed to the branch 2026-08-16T15:49:51Z,
delivering the refactor the author promised in the 08-17 comment:
`triton_sparse_mla` folds into the existing `flashmla_sparse` family
instead of a sibling branch, and `--dsa-triton-dense-prefix` was dropped.
Confirmed via the PR's current diff on `flash_mla_sm120.py`:
`_validate_flashinfer_sparse_mla_backend`'s is_glm_sm12_fp8 arm is now
`selected - {"flashinfer_sparse_mla", "triton_sparse_mla"}` (was
`{"flashinfer_sparse_mla"}`), with an added comment stating
`flashinfer_sparse_mla` stays the auto-selected default and
`triton_sparse_mla` is only a validated opt-in alternative, same
substantive change already recorded 08-21, now confirmed present on the
pushed commit too. REST now reports `mergeable: true` / `mergeable_state:
"unstable"` (was `false`/`"dirty"` on 08-21), 9 files changed, +1365/-8.
Labels unchanged (performance, run-ci, jit-kernel, GLM). New maintainer
activity, most recent first: nvpohanh `/rerun-failed-ci`
(2026-08-27T15:13:28Z, most recent event); nvpohanh "@yunyang1999 could you
fix the conflicts?" (2026-08-27T12:29:08Z and again 2026-08-24T03:50:41Z);
nvpohanh asked b8zhong to check yunyang1999's response (2026-08-24T03:50:27Z).
These "fix the conflicts" requests directly conflict with the API's
`mergeable: true` reading as of this check, flagging as an open discrepancy
(stale cache vs. a real, more-recent conflict) rather than asserting
either. Not merged, `reviewDecision: REVIEW_REQUIRED`, one empty-body
`COMMENTED` review from b8zhong (2026-08-12) is the only review on record.

flashinfer stable still v0.6.17 (2026-08-11); zero commits to
`flashinfer/mla/_sparse_mla_sm120.py` since then (checked via the commits
API). A new release channel tag `v0.6.18rc10` was published today
(2026-08-28T06:15:01Z, flagged non-prerelease in metadata) but its
changelog is SM107 (Rubin) fixes only, unrelated to SM120/121
GLM_NSA/DSv3.2 dispatch. flashinfer PR #4551 still open (`updated_at`
2026-08-25), not merged, still scoped to DSV4 decode-dispatch diagnostics
only, still explicitly not touching the DSv3.2/GLM_NSA path this PR needs,
prior "ruled irrelevant" conclusion unchanged.

SGLang v0.5.18 (2026-08-22, 710 PRs) reviewed for DSA/GLM-5.2 entries:
#33436/#33945 (FA4 GLM4.7-flash), #33474 (DeepGEMM layout selection),
#30531/#33857 (indexer/DSV4 logits skip optimizations), #33793 (GLM-5.2 PP
MoE weight restriction), all either unrelated model work or DSV4-specific;
none touch the GLM_NSA/DSv3.2 SM12x auto-selection or the `_forward_trtllm`
hardcode.

p34 retirement decision remains pending: no TP4/real-weight confirmation
run since 08-15 (approval-gated, not requested this cycle). New unmerged
draft-scale PR #36507 "GLM-5.3-Flash support" (opened 08-26, updated 08-28)
touches `dsa_backend.py`/`overrides.py`/`environ.py` among ~50+ files but
not `flash_mla_sm120.py` or the sparse-MLA dispatch; flagged as activity to
watch only.

> Re-checked 2026-09-02: PR #31481 head unchanged (`0864022c3d`), `updated_at`
> unchanged (2026-08-03T13:03:37Z), still 2 comments, 0 reviews, only the
> `deepseek` label, `reviewDecision: REVIEW_REQUIRED`. Mergeable state flipped
> again: REST API now reports `mergeable: false` / `mergeable_state: "dirty"`
> (was `true`/`"unstable"` on 08-28, confirmed on 3 repeated reads), i.e. main
> has drifted since the 08-03 rebase and #31481 needs another rebase (not done,
> pending approval, same recurring pattern as #31480's 08-15 and 08-28
> entries). Likely driver: `overrides.py` (which
> `_dsa_split_backend_resolution` lives in) took 6 more commits since the 08-28
> baseline (`d56706459c`), mostly the ongoing "config bags"/platform-facts
> refactor plus unrelated model work (qwen 3.8 rebase,
> speculative-draft-capacity decoupling).

> `dsa_backend.py::_forward_trtllm` on `upstream/main` still hardcodes
> `backend="trtllm-gen"` at line 3269 (unchanged position/content; the only
> commit touching the file since baseline is the same `7e751153eb`
> platform-rename refactor noted in the companion doc).
> `overrides.py::_dsa_split_backend_resolution` / `is_glm_sm12_fp8` diffed
> directly (function body) between baseline and current `upstream/main`:
> substantively byte-identical, only `is_npu()`/`is_xpu()`/`is_hip()` calls
> renamed to `get_platform().is_npu`/`.is_xpu`/`.is_hip` properties (same
> refactor). The GlmMoeDsaForCausalLM+SM12x+fp8_e4m3-KV auto-selection to
> `flashinfer_sparse_mla` is unchanged. `flash_mla_sm120.py`: zero commits
> since baseline.

> PR #32779 has real forward progress: author pushed a "Merge branch 'main'
> into dsa-triton-sparse-mla-prefill" today (2026-09-02, new head `56bfaf9527`,
> up from `2a625a4396`), resolving the conflicts nvpohanh had asked about on
> 08-24/08-27; `mergeable` flipped from `false`/`"dirty"` (08-28) to
> `true`/`"blocked"` today. nvpohanh triggered `/rerun-failed-ci` today at
> 11:55 UTC. Still 9 files changed (unchanged scope). Re-diffed the validator:
> `_validate_flashinfer_sparse_mla_backend`'s `is_glm_sm12_fp8` arm is still
> `selected - {"flashinfer_sparse_mla", "triton_sparse_mla"}` with
> `flashinfer_sparse_mla` explicitly staying the auto-selected default (comment
> unchanged), so this remains not redundancy-relevant, only upstream activity
> to watch. Not merged, `reviewDecision: REVIEW_REQUIRED`, still only the one
> empty-body `COMMENTED` review from b8zhong (2026-08-12), labels unchanged
> (performance/run-ci/jit-kernel/GLM).

> flashinfer bumped its stable release to v0.6.18 (2026-08-29, was v0.6.17
> 08-11). Compared v0.6.17..v0.6.18 via the GitHub compare API: the only files
> touching `sparse_mla_sm120` are a DSv4-only benchmark script (+39/-1,
> unrelated to GLM_NSA/DSv3.2) and three `.cu` kernel files reported with 0
> additions/0 deletions (no functional diff surfaced by the compare API);
> `flashinfer/mla/_sparse_mla_sm120.py` (the Python dispatcher) does not appear
> in the file list at all. Ruled not relevant to the redundancy question, same
> pattern as prior releases.

> SGLang still at v0.5.18 (2026-08-22), no new release. PR #36507
> "GLM-5.3-Flash support" gained several new labels (including
> `release-highlight`) and is landing piecewise (a component already merged as
> `c66a285c94`, see companion doc); its diff touches `_forward_trtllm` (new
> page-table padding / DCP LSE plumbing) but the `backend="trtllm-gen"`
> hardcode itself and the `is_glm_sm12_fp8` selection are untouched; it adds a
> new model family `Glm5NextForConditionalGeneration` to the DeepSeek-family
> allowlist, unrelated to our tracked `GlmMoeDsaForCausalLM` class. Flagged as
> activity to watch only.

> p34 retirement decision remains pending: no TP4/real-weight confirmation run
> since 08-15 (approval-gated, not requested this cycle). No upstream change
> alters the standing 08-15 verdict.

> Follow-up same day (2026-09-03, approved): #31481 rebased onto current main
> `a6001478f4` (2026-09-03). Old head `0864022c3d` -> new head `924c838c1d`,
> pushed `--force-with-lease` to the vroomfondel fork. Old base was
> `d48ab2d386` (2026-08-03); one file conflicted, `dsa_backend.py`, the same
> platform-facts refactor pattern #31480 hit on 08-28: our added
> `is_sm100_supported, is_sm120_supported` import collided with upstream
> deleting both helpers from `sglang.srt.utils` entirely (folded into
> `get_platform().is_sm100`/`.is_sm120`). Resolved by dropping the import and
> reading `get_platform().is_sm120` live at the point of use instead of caching
> it at module import time, per an upstream CI test that AST-scans all of
> `srt/` and fails on exactly that pattern
> (`test_platform_address_not_frozen.py`, "a module-level `_is_sm120 =
> get_platform().is_sm120` defeats [override_platform] completely").

> The other two files in the diff (`kv_cache_configurator.py`,
> `forward_mla.py`) auto-merged without a textual conflict, but both still
> referenced the now-deleted `is_sm120_supported` (module-level
> `_is_sm120`/`_IS_SM120` constants) since upstream's deletion did not touch
> the same lines. Git's clean auto-merge does not mean working code: both were
> caught by grepping the full diff for the dead names post-rebase and fixed the
> same way (live `get_platform().is_sm120`, no module-level caching). A
> repo-wide AST scan for `X = get_platform().Y` at module scope (the exact
> check the upstream CI test runs) came back empty on the rebased branch.

> Three of the four new test files in this PR's own diff also referenced the
> removed `_IS_SM120`/`is_sm120_supported` (as a `mock.patch.object` target or
> a direct import) and one additionally still passed a `server_args=` kwarg to
> `calculate_mla_kv_cache_dim`, which no longer accepts one (it now reads
> `get_exec().kernel.dsa_prefill_backend`/`.dsa_decode_backend` from the
> published config bag instead). Fixed `test_dsa_kv_cache_dim_sm12x.py` to
> publish a dummy ServerArgs via `get_context().override_server_args(...)` (the
> same sanctioned test hook `test_pool_configurator.py` uses) and to scope
> SM120 via `override_platform(is_sm120=...)`; fixed
> `test_dsa_backend_trtllm_sm12x_kwargs.py` and
> `test_deepseek_mla_fuse_rope_trtllm_sm12x.py` by mocking `get_platform`
> itself (same style already used there for `get_exec`/`get_attn_backend`). One
> test constant, `flashinfer_gather`, was never a real `dsa_prefill_backend`
> choice; swapped for `flashmla_sparse`.

> Verification: no conflict markers repo-wide on the 7 changed files,
> `ast.parse` clean on all 7, `git range-diff d48ab2d386..0864022c3d
> upstream/main..HEAD` shows a single commit pair with the expected content
> drift only (rest is pure context shift), diffstat vs the new merge-base
> stayed essentially stable at 7 files (+862/-7, was +830/-7 at the old base;
> the +32 delta is the dropped dead imports/constants plus the added
> get_platform()-based lines, explanatory comments, and the rewritten mem_cache
> test). `black`/`mypy` could not be run in this environment (no local
> sglang/torch install); line lengths were checked by hand against the repo's
> 88-char default and kept under it.

> Pushed to `git@github.com:vroomfondel/sglang.git` (verified push remote
> before pushing), confirmed live: `gh pr view 31481` now reports `headRefOid:
> 924c838c1d56b4a6b36726d609cdbc65130e1de5`, `mergeable: MERGEABLE` /
> `mergeStateStatus: BLOCKED` (REST: `mergeable: true` / `mergeable_state:
> "blocked"`), i.e. back to checks/review gating only, no conflict, matching
> #31480's post-rebase state pattern. Content/design conclusions of this doc
> are unchanged; this was a submission-mechanics rebase only.

> Re-checked 2026-09-04: PR #31481 unchanged since the 09-03 rebase (head
> still `924c838c1d56b4a6b36726d609cdbc65130e1de5`, `updated_at`
> 2026-09-03T09:40:10Z), still 2 comments, 0 reviews, only the `deepseek`
> label, `mergeable: MERGEABLE` / `mergeStateStatus: BLOCKED` unchanged.
> `dsa_backend.py`, `overrides.py` and `flash_mla_sm120.py` on
> `upstream/main`: zero commits touched any of the three since the 09-03
> rebase base (`a6001478f4`) through current tip `67248e04b4`.
> `_forward_trtllm`'s `backend="trtllm-gen"` hardcode and the
> `is_glm_sm12_fp8` auto-selection remain unchanged.

> PR #32779 got a routine "Merge branch 'main' into
> dsa-triton-sparse-mla-prefill" sync commit today (2026-09-04T07:17:26Z, new
> head `74ad6930ef`, up from `56bfaf9527` on 09-02), the fourth such sync
> merge since 08-31, no authored change. Diff scope unchanged (still the same
> 9 files, +1365/-8 net vs main); re-diffed
> `_validate_flashinfer_sparse_mla_backend`'s `is_glm_sm12_fp8` arm directly:
> still `selected - {"flashinfer_sparse_mla", "triton_sparse_mla"}` with
> `flashinfer_sparse_mla` explicitly staying the auto-selected default.
> `mergeable: MERGEABLE` / `mergeStateStatus: BLOCKED` unchanged, still only
> the one COMMENTED review from b8zhong (2026-08-12), labels unchanged
> (performance/run-ci/jit-kernel/GLM). Not merged, not redundancy-relevant,
> same as prior cycles.

> flashinfer still v0.6.18 (2026-08-29), SGLang still v0.5.18 (2026-08-22),
> no new releases. PR #36507 gained more labels and heavy piecewise-landing
> activity (see companion doc for detail) but confirmed still not touching
> the SM12x dispatch this doc tracks.

> p34 retirement decision remains pending: no TP4/real-weight confirmation
> run since 08-15 (approval-gated, not requested this cycle). No upstream
> change alters the standing 08-15 verdict.

> Re-checked 2026-09-10: PR #31481 unchanged in content since the 09-03
> rebase (head still `924c838c1d56b4a6b36726d609cdbc65130e1de5`,
> `updated_at` 2026-09-03T09:40:10Z, still 2 comments, 0 reviews, only the
> `deepseek` label). But `mergeable`/`mergeStateStatus` flipped from
> `MERGEABLE`/`BLOCKED` to `CONFLICTING`/`DIRTY`, confirmed stable across
> two repeat `gh` reads. Real merge conflict against current main, first
> observed this cycle, same pattern as the companion indexer PR.

> Root cause is heavy drift on the exact files p34 touches. PR #36507
> "GLM-5.3-Flash support" merged 2026-09-06 (`97c6978369`, was still open
> and unreviewed on 09-04) and rewrites large parts of `dsa_backend.py`
> (new KPool imports, new `DSAMetadata` fields, new
> `dsa_backend_kpool.py`/`kpool_plan.py` modules). Diffed directly: the
> `_forward_trtllm` hunk it touches only adds a `sparse_mla_top_k_lens`
> param and page-table-padding plumbing, already noted on 09-02/09-04. The
> `backend="trtllm-gen"` line p34 patches (now around line 3570 on main)
> remains untouched context. Also landed since 09-03: the CP V1
> Deprecation series continued (`ed183d45ac` "Canonicalize prefill CP API
> names", part 4/5, merged after 09-04) touching `dsa_backend.py`'s CP
> branches, and a unified-cache reorg (`abed680320`, `0645398a32`) touching
> the `kv_cache_configurator.py` area p34's change 1 lives in.

> `calculate_mla_kv_cache_dim` (now in `kv_cache_configurator.py`) still
> dispatches purely on `get_exec().kernel.dsa_prefill_backend` /
> `dsa_decode_backend == "trtllm"` (the "plain iff a backend is named
> trtllm" coupling this doc's change 1 exists to fix), no SM12x
> special-casing added upstream. `overrides.py`'s `is_glm_sm12_fp8` block
> (sets `flashinfer_sparse_mla` for GLM plus SM120/121 plus fp8 KV) is
> unchanged despite several other commits touching that file. p34's design
> conclusion is unchanged, still necessary. Only the rebase got harder, same
> structural-drift pattern as p30, driven by the same broader refactor wave,
> not a content collision here.

> PR #32779 (Triton sparse MLA prefill): head advanced from `74ad6930ef`
> (09-04) to `083eca458f` (latest, several more "fix conflicts" / "fix
> lint" pings from nvpohanh through 09-09), still just the one COMMENTED
> review from b8zhong (08-12), zero approvals, `run-ci` / `performance` /
> `jit-kernel` / `GLM` labels unchanged. Its own `mergeable` flipped to
> `CONFLICTING` and `mergeStateStatus` to `DIRTY` too (author still
> wrestling with conflicts as of 09-10). Not merged. Re-diffed
> `_validate_flashinfer_sparse_mla_backend` on the new head directly: the
> `is_glm_sm12_fp8` arm is still `{"flashinfer_sparse_mla",
> "triton_sparse_mla"}`, with `flashinfer_sparse_mla` staying the
> auto-selected default. Not redundancy-relevant.

> SGLang released v0.5.19 on 2026-09-05 (release/v0.5.19 cherry-pick
> branch). Contains #26928 (SM120 GLM-5.1 baseline, already known). Does
> NOT contain #29927 (SM120 DeepGEMM paged-MQA indexer, merged 2026-09-02)
> or #36507 (merged 2026-09-06), both too late for the tag. Cross-reference
> for the companion DSV4 doc.

> p34 retirement decision remains pending: no TP4/real-weight confirmation
> run since 08-15 (approval-gated, not requested this cycle). No upstream
> change alters the standing 08-15 verdict.

> Rebased 2026-09-10 (approved): #31481 rebased onto current upstream/main
> `12771786f2` (2026-09-10). Old head
> `924c838c1d56b4a6b36726d609cdbc65130e1de5` to new head
> `e5c0b4c7f0efee056af3e5794f5a0386f78e7fc7`, pushed `--force-with-lease`
> to the vroomfondel fork. Old base was `a6001478f4` (2026-09-03), two
> files conflicted.

> `dsa_backend.py` had two hunks in `_forward_trtllm`. First: upstream
> (from the GLM-5.3-Flash landing, independent of our patch) changed
> `merge_query = q_rope is not None` to `q_rope is not None and
> self.qk_rope_head_dim > 0` (a BF16 no-RoPE carve-out) on the same line
> our `and not _sparse_sm120` addition touches; kept both. Second: the same
> landing added a new `sparse_mla_top_k_lens` kwarg to the decode call
> (`None` unless `qk_rope_head_dim == 0`, orthogonal to backend selection);
> merged it in unconditionally alongside our existing `_sparse_sm120`-gated
> kwargs (`backend`, `kv_scale_format`,
> `skip_softmax_threshold_scale_factor`, `multi_ctas_kv_counter_buffer`),
> none of which needed changes themselves. Confirmed post-merge that the
> `is_glm_sm12_fp8` auto-selection in `overrides.py` and the plain
> `backend="trtllm-gen"` hardcode this doc's problem statement describes
> are both still intact on the merged file.

> `kv_cache_configurator.py`: upstream independently added disaggregation-
> mode-aware dispatch to `calculate_mla_kv_cache_dim` (checks
> `get_disagg().disaggregation_mode` and reads only the local role's
> backend in "decode"/"prefill" mode, falling back to the old
> either-backend check otherwise), landing after the 09-03 baseline.
> Resolved by keeping upstream's disaggregation-aware `uses_trtllm_kv_
> layout` computation as the outer condition and nesting p34's SM12x
> carve-out (`if not get_platform().is_sm120: return kv_cache_dim`) inside
> it, in the spirit of the new structure rather than reverting it.
> `forward_mla.py` auto-merged with no conflict.

> Verification before push: no conflict markers repo-wide, `ast.parse`
> clean on all 7 touched/new files, a repo-wide scan for module-scope
> `X = get_platform().Y` assignments in the touched files came back empty,
> own diffstat vs the new merge-base essentially stable at 7 files
> (+864/-7, was +862/-7 at the old base, the +2 delta is the merged
> `sparse_mla_top_k_lens` passthrough comment and the nested-condition
> reshape). `black`/`mypy` could not be run in this environment (no local
> sglang/torch install, same limitation as the 09-03 rebase); line lengths
> around the resolved hunks were checked by hand against the repo's
> 88-char default and stayed under it (the few over-88 lines nearby are
> pre-existing upstream comments, untouched by this patch). Pushed to
> `git@github.com:vroomfondel/sglang.git` (remote verified before
> pushing). Confirmed live: `gh pr view 31481` reports `headRefOid:
> e5c0b4c7f0efee056af3e5794f5a0386f78e7fc7`, `mergeable: MERGEABLE` /
> `mergeStateStatus: BLOCKED` (checks/review gating only, no conflict).
> Content/design conclusions unchanged, this was a submission-mechanics
> rebase only.

> Re-checked 2026-09-16: PR #31481 unchanged in content since the 09-10 rebase
> (head still `e5c0b4c7f0efee056af3e5794f5a0386f78e7fc7`, `updatedAt`
> 2026-09-10T13:13:45Z, still 2 comments, 0 reviews, only the `deepseek` label,
> `reviewDecision: REVIEW_REQUIRED`). First `gh` read returned transient
> `mergeable: UNKNOWN` / `mergeStateStatus: UNKNOWN`; repeat read 3s later
> confirmed stable `MERGEABLE` / `BLOCKED`, same lazy-cache artifact as the
> companion PR, not a real change.

> `dsa_backend.py::_forward_trtllm` on `upstream/main` (checked against
> `upstream/main` tip `2cbfaefbf9`, 2026-09-16, fetched fresh) still hardcodes
> `backend="trtllm-gen"`, now at line 3455 (drift from unrelated commits, same
> pattern every cycle). `overrides.py::is_glm_sm12_fp8` block still sets both
> prefill/decode to `flashinfer_sparse_mla` for `GlmMoeDsaForCausalLM` + SM12x
> + fp8_e4m3 KV, function body unchanged. Five commits touched `overrides.py`
> since the 09-10 baseline: unified hybrid-SWA memory budget (#36729,
> 09-15), PD disaggregation for unified pool shapes (#37506), a prefill-
> buffer-ceiling provider hook (#39182, 09-12), DCP comm-backend default
> (#39165, 09-11), NemotronH_Omni_Reasoning_V3 model support (#35599, 09-10),
> a dead-code cleanup (#39295, 09-16); none touch
> `is_glm_sm12_fp8`/`flashinfer_sparse_mla` (checked via diff grep, 0 hits
> each). `calculate_mla_kv_cache_dim` (`mem_cache/kv_cache_configurator.py`)
> still dispatches purely on `get_exec().kernel.dsa_prefill_backend`/
> `dsa_decode_backend == "trtllm"` (lines 2588-2604), no SM12x special-casing
> added; four commits touched the file (unified hybrid-SWA memory, PD
> disagg, two Qwen3.8-Next cleanups), none touch this function.

> One new SM120-labeled commit needs a call-out: `d076eec427` ("[SM120] Use
> exact query-head widths for DeepSeek-V4 sparse MLA decode", #36655, merged
> 2026-09-10) touches `flash_mla_sm120.py`, adding
> `_flashinfer_dsv4_decode_capabilities()` and
> `flashinfer_dsv4_decode_supports_num_heads()`. Despite the "SM120" tag in
> the title, both new functions are explicitly DSV4-scoped by name and do not
> touch `_validate_flashinfer_sparse_mla_backend` or any GLM_NSA/DSv3.2 logic
> (direct diff: only `flash_mla_sm120.py`, +30/-1, validator untouched). Same
> "DSV4-only, not relevant to our GLM_NSA path" pattern as flashinfer PR
> #4551 flagged 08-21, not redundancy-relevant, cross-reference note for the
> companion DSV4 doc only.

> `forward_mla.py`: one commit, `833bce9df5` ("[AMD][DCP 1/N] add dcp support
> for aiter backend", #34432, 09-11), touches only `is_dcp_mla_decode_phase`,
> not `_fuse_rope_for_trtllm_mla`. Not relevant (ROCm-scoped, as expected).

> **Local patch cross-check (commit 958c98d, 2026-09-11, deployed as
> `xomoxcc/dgx-spark-sglang:0.5.19-sm121`):** p34's re-anchor claims
> `kv_cache_configurator.py` server-arg reads are now spelled
> `get_exec().kernel.<arg>` (citing #35907). Verified directly against
> `upstream/main`: accurate, lines 2588-2604 read exactly
> `get_exec().kernel.dsa_prefill_backend`/`.dsa_decode_backend`. p34's third
> `replace_any` spelling for the mixin patch targets this correctly.

> p34 retirement decision remains pending: no TP4/real-weight confirmation
> run since 08-15 (approval-gated, not requested this cycle). No upstream
> change alters the standing 08-15 verdict (stock auto-selects and runs
> `flashinfer_sparse_mla`, redundancy supported at TP1/dummy-weight level
> only).

> PR #32779 (Triton sparse MLA prefill) has real forward progress this cycle:
> resolved its 09-10 `CONFLICTING`/`DIRTY` state, head advanced from
> `083eca458f` (09-10) through three more commits (`fb5a78052a`, a lint fix
> `af62e4aedb`, `165d63974e`) to current head
> `2600a0a51be421d7db7fd8c3f8f78f108acedf18` (a "Merge branch 'main'" sync),
> now `mergeable: MERGEABLE` / `mergeStateStatus: BLOCKED`. More
> significantly: it got its first `APPROVED` review, from maintainer
> nvpohanh on 2026-09-14 (the pre-existing `COMMENTED` review from b8zhong,
> 08-12, is still the only other one). nvpohanh kept pinging
> `/rerun-failed-ci` through 2026-09-16 (today), suggesting CI is still not
> fully green despite the approval. Diffstat now +1366/-8, 9 files (was
> +1365/-8 on 09-10), `run-ci`/`performance`/`jit-kernel`/`GLM` labels
> unchanged. Re-diffed `_validate_flashinfer_sparse_mla_backend` directly on
> the current PR head (fetched `pull/32779/head`): the `is_glm_sm12_fp8` arm
> is still `selected - {"flashinfer_sparse_mla", "triton_sparse_mla"}`,
> comment unchanged stating `flashinfer_sparse_mla` stays the auto-selected
> default. Not merged, not redundancy-relevant to this doc's conclusion, but
> flag this PR as the one closest to landing of anything tracked across
> either doc this cycle, worth a closer look next audit if it merges or gets
> a second approval.

> SGLang still at v0.5.19 (2026-09-05), no new release. `RETIRED_PATCHES.md`
> re-checked: no p34 entry exists, correctly reflecting still-active status.

> Re-checked 2026-09-22: PR #31481 unchanged in content since the 09-10 rebase
> (head still `e5c0b4c7f0efee056af3e5794f5a0386f78e7fc7`, `updatedAt`
> 2026-09-10T13:13:45Z, still 2 comments, 0 reviews, only the `deepseek`
> label, `reviewDecision: REVIEW_REQUIRED`). Unlike the companion PR,
> `mergeable`/`mergeStateStatus` changed: two reads 3s apart both returned
> `CONFLICTING`/`DIRTY` (was `MERGEABLE`/`BLOCKED` on 09-16). Head SHA is
> unchanged (no new commits on the PR branch), so this is `upstream/main`
> moving past the PR's base since the 09-10 rebase, not new PR activity, the
> branch now needs a fresh rebase before it is mergeable again, same pattern
> as the 08-03/08-07 cycle.

> `dsa_backend.py::_forward_trtllm` source-verified directly against the
> `v0.5.20` tag (commit `94602c9c2b`, released 2026-09-18): still hardcodes
> `backend="trtllm-gen"`, at line 3455, the identical line cited on 09-16
> (checked against `upstream/main` tip at the time), no drift.
> `overrides.py::is_glm_sm12_fp8` (`arg_groups/overrides.py`) function body
> unchanged: still sets both prefill/decode to `flashinfer_sparse_mla` for
> `GlmMoeDsaForCausalLM` + SM12x + fp8_e4m3 KV. `calculate_mla_kv_cache_dim`
> (`mem_cache/kv_cache_configurator.py`) still gates the plain-layout early
> return purely on `get_exec().kernel.dsa_prefill_backend`/
> `dsa_decode_backend == "trtllm"` (now split by `disaggregation_mode`
> decode/prefill/else, functionally the same predicate), no SM12x
> special-casing added. p34's three `replace_any` anchors remain accurate,
> and the file is unmodified in the working tree (`git status` shows no
> changes to `p34_dsa_trtllm_sparse_sm120.py`, unlike the companion patch
> p30, see the indexer doc's 2026-09-22 entry).

> v0.5.20 confirmed to contain #36507 (GLM-5.3-Flash) and #29927/#36655 (both
> DeepSeek-V4-scoped per the release notes headline), same finding as the
> companion doc, not redundancy-relevant to this doc's GLM_NSA/SM12x path.

> **PR #32779 (Triton sparse MLA prefill) has substantial forward progress
> this cycle.** Head advanced from `2600a0a51be...` (09-16) to
> `6848fe723e7487e6c4b44ae7bb44b74463ac66aa`, several more commits since. The
> `run-ci` label is now present (was absent on 09-16), alongside
> `performance`/`jit-kernel`/`GLM`. `mergeable`/`mergeStateStatus` reads
> `MERGEABLE`/`BLOCKED`. Still only one `APPROVED` review (nvpohanh,
> 2026-09-14) and one `COMMENTED` (b8zhong, 08-12), no second approval yet.
> Most notable: nvpohanh commented "All NV pipelines have passed." today
> (2026-09-22T02:43:07Z), the first all-green CI signal recorded across this
> PR's tracked history (nvpohanh had been pinging `/rerun-failed-ci` roughly
> every few days since 07-30). Still `state: OPEN`, not merged. Re-diffed
> `_validate_flashinfer_sparse_mla_backend` on the current head (fetched
> `pull/32779/head`, now living at `kernels/ops/attention/flash_mla_sm120.py`
> after an upstream path move): the `is_glm_sm12_fp8` arm is unchanged, still
> permits `selected - {"flashinfer_sparse_mla", "triton_sparse_mla"}` with
> `flashinfer_sparse_mla` staying the documented auto-selected default, no
> change to this doc's redundancy conclusion if it merges. Flag for next
> audit: this is the closest any tracked PR has come to merging across either
> doc; check merge status and, if merged, whether it lands in the next
> release and whether it changes GlmMoeDsa+SM12x auto-selection.

> p34 retirement decision remains pending, unchanged from 08-15 (no new
> TP4/real-weight confirmation run requested this cycle).

> SGLang v0.5.20 released 2026-09-18 (up from v0.5.19, 2026-09-05).
> `RETIRED_PATCHES.md` (`roles/k8s_dgx/files/sglang_patches/RETIRED_PATCHES.md`)
> re-checked: no p34 entry exists, correctly reflecting still-active status.

> Rebased 2026-09-22 (the CONFLICTING/DIRTY state flagged above): #31481
> rebased onto current `upstream/main` `790551c382` (fetched fresh, old base
> `12771786f2`). Old head `e5c0b4c7f0efee056af3e5794f5a0386f78e7fc7` -> new
> head `7aa9049335de83ed1c2227c176041301a6309a29`, pushed to the vroomfondel
> fork with `--force-with-lease`. Only one real conflict, in
> `dsa_backend.py::_forward_trtllm`'s `multi_ctas_kv_counter_buffer` kwarg:
> upstream refactored the counter-buffer growth logic into a
> `self._multi_ctas_kv_counter_for(batch_size)` helper returning a local
> variable of the same name, replacing the old
> `self._multi_ctas_kv_counter_buffer` attribute this PR's `_sparse_sm120`
> gate had been built against. Resolved by keeping our gate and switching to
> the new local-variable spelling:
> `multi_ctas_kv_counter_buffer=(None if _sparse_sm120 else
> multi_ctas_kv_counter_buffer)`. `kv_cache_configurator.py` and
> `forward_mla.py` auto-merged clean, no manual changes needed there.
> Diffstat vs the new merge-base is unchanged from the old baseline: 7 files,
> +864/-7. All 7 changed files parse clean
> (`ast.parse`), `git range-diff` against the pre-rebase commit shows only
> the one expected hunk (the conflict resolution), everything else
> byte-identical. No module-scope `X = get_platform().Y` assignments in any
> touched file. `gh pr view 31481` confirms `MERGEABLE`/`BLOCKED` with the
> new head live (first read, no UNKNOWN state). Content/design conclusions
> unchanged, this was a submission-mechanics rebase only.

> Re-checked 2026-09-25: PR #31481 unchanged in content since the 09-22
> rebase (head still `7aa9049335de83ed1c2227c176041301a6309a29`, `updatedAt`
> 2026-09-22T09:03:28Z, still 2 comments, 0 reviews, only the `deepseek`
> label, `reviewDecision: REVIEW_REQUIRED`). `mergeable`/`mergeStateStatus`
> stable at `MERGEABLE`/`BLOCKED` on two `gh` reads 3s apart, i.e. the 09-22
> rebase held, no new drift.

> SGLang still at v0.5.20 (2026-09-18, `gh release list` confirms it as
> latest, no new tag). `dsa_backend.py::_forward_trtllm` on `upstream/main`
> (fetched fresh) still hardcodes `backend="trtllm-gen"`, now at line 3506
> (drifted from line 3455 on 09-22, unrelated commits, mostly the
> `multi_ctas_kv_counter_buffer` helper refactor that drove the 09-22 rebase
> conflict). `overrides.py::is_glm_sm12_fp8` unchanged: still sets both
> `dsa_prefill_backend`/`dsa_decode_backend` to `flashinfer_sparse_mla` for
> `GlmMoeDsaForCausalLM` + SM12x + `fp8_e4m3` KV. `flash_mla_sm120.py` (now
> `python/sglang/kernels/ops/attention/flash_mla_sm120.py`): zero commits
> since v0.5.20, validator untouched.

> PR #32779 did **not** merge this cycle: still `state: OPEN`,
> `mergedAt: null`, head unchanged since 09-22
> (`6848fe723e7487e6c4b44ae7bb44b74463ac66aa`), still one `APPROVED` review
> (nvpohanh, 09-14) and one `COMMENTED` (b8zhong, 08-12), no second approval,
> `run-ci`/`performance`/`jit-kernel`/`GLM` labels unchanged, no new comments
> since "All NV pipelines have passed." (09-22). Re-confirmed on its current
> head: `_validate_flashinfer_sparse_mla_backend`'s `is_glm_sm12_fp8` arm
> still only widens the allowed set to `{"flashinfer_sparse_mla",
> "triton_sparse_mla"}`, `flashinfer_sparse_mla` stays the documented
> auto-selected default. So even if it merges, it would not change the
> auto-selection this doc's p34-redundancy question depends on, a merge would
> only add an opt-in alternative backend.

> **Deploy confirmed:** dgxarley commit `3214ed2` (2026-09-22) bumped the
> deployed image from `xomoxcc/dgx-spark-sglang:0.5.19-sm121` to
> `0.5.20-sm121`. Live head pod (`sglang-head-68c7bbf746-kfwlt`, created
> 2026-09-23T07:15:21Z) confirmed running `0.5.20-sm121`. `kubectl logs -c
> sglang` confirms p34's three re-anchored targets applied cleanly with no
> `ANCHOR-DRIFT` warning: "Patched kv_cache_configurator.py: MLA KV dim: keep
> the 656-byte packed layout for trtllm on SM12x", "Patched dsa_backend.py:
> _forward_trtllm: backend=auto + GLM_NSA scale format on SM12x", "Patched
> forward_mla.py: _fuse_rope_for_trtllm_mla: no fused rope+fp8-q on SM12x
> (sparse wants bf16 q)". Unlike p30, p34 had no uncommitted working-tree
> edit flagged on 09-22 (already stable for 0.5.20), consistent with a clean
> apply here.

> p34 retirement decision remains pending, unchanged from 08-15 (no new
> TP4/real-weight confirmation run requested this cycle).

> Re-checked 2026-09-28: PR #31481 unchanged in content since the 09-22
> rebase (head still `7aa9049335de83ed1c2227c176041301a6309a29`, `updatedAt`
> 2026-09-22T09:03:28Z, still 2 comments, 0 reviews, only the `deepseek`
> label, `reviewDecision: REVIEW_REQUIRED`). `mergeable`/`mergeStateStatus`
> stable at `MERGEABLE`/`BLOCKED` on two `gh` reads 3s apart, i.e. the 09-22
> rebase still holds, no new drift. Fork remote branch head matches the PR
> head exactly.

> SGLang still at v0.5.20 (2026-09-18, latest, no new tag). `dsa_backend.py::
> _forward_trtllm` on `upstream/main` (fetched fresh, tip `df378bca42`): zero
> commits since the 09-22 rebase base; still hardcodes `backend="trtllm-gen"`
> at line 3506 (unchanged from the 09-25 check). `overrides.py::
> is_glm_sm12_fp8` block: zero commits touch the function (one unrelated
> ROCm mxfp8-runner-backend commit, `425a1f8f24`, touched the file
> elsewhere). `kv_cache_configurator.py`: one commit since 09-25
> (`50e50e13ab`, #39731, PD admission/load reporting) does not touch
> `calculate_mla_kv_cache_dim`. `forward_mla.py` and `flash_mla_sm120.py`:
> zero commits. p34's redundancy question and design conclusion unchanged.

> Dry-run rebase check (`git merge-tree --write-tree upstream/main
> dsa-sm12x-native-sparse-mla`, no push, no worktree touched): clean merge
> tree, no conflicts. No rebase needed right now, PR already `MERGEABLE`.

> PR #32779 (Triton sparse MLA prefill) did not merge this cycle: still
> `state: OPEN`, `mergedAt: null`, head unchanged since 09-22
> (`6848fe723e7487e6c4b44ae7bb44b74463ac66aa`), still one `APPROVED` review
> (nvpohanh, 09-14) and one `COMMENTED` (b8zhong, 08-12), no second
> approval, labels unchanged (performance/run-ci/jit-kernel/GLM), no new
> activity since "All NV pipelines have passed." (09-22). Broad grep of
> upstream commits since 09-25 for indexer/sparse_mla/sm12/GB10 found
> nothing else relevant (NPU top-k support and DSv4.1/AMD low-ratio-indexer
> work, both out of scope for the GLM_NSA/SM12x path this doc tracks).

> p34 retirement decision remains pending, unchanged from 08-15 (no new
> TP4/real-weight confirmation run requested this cycle). No factual errors
> found in the 09-25 entry.

> Re-checked 2026-10-03: PR #31481 unchanged in content since the 09-22
> rebase (head still `7aa9049335de83ed1c2227c176041301a6309a29`, `updatedAt`
> 2026-09-22T09:03:28Z, no new comments or reviews, `reviewDecision:
> REVIEW_REQUIRED`). `mergeable`/`mergeStateStatus` stable at
> `MERGEABLE`/`BLOCKED` on two `gh` reads 5s apart. Fork remote branch head
> matches the PR head exactly.

> SGLang v0.5.21 was tagged 2026-10-02 (now latest). `upstream/main` (fetched
> fresh, tip `7be5e3473c`): `flash_mla_sm120.py` zero commits;
> `forward_mla.py` only the module-rename refactor `de9f850b36` (#41555);
> `kv_cache_configurator.py` took `8055ccd2cd`, `ec0006673a`, `eb9c9ee99d` and
> `f0e4001930` (linear-attention accounting, NPU DCP, QSA indexer fp8 cache,
> SWA pool memory-saver), none checked against `calculate_mla_kv_cache_dim`
> beyond a clean merge-tree; `dsa_backend.py` took `42f2e4fcb7` (#41987),
> `3ed6367d3f` (#41337), `e9b0d0c0a3` (#40911) and `7889fbab9a` (#32196), none
> touching the `_forward_trtllm` backend selection. p34's redundancy question
> and design conclusion unchanged.

> Dry-run rebase check (`git merge-tree --write-tree upstream/main
> dsa-sm12x-native-sparse-mla`, no push, no worktree touched): clean merge
> tree, no conflicts. No rebase needed right now, PR already `MERGEABLE`.

> PR #32779 (Triton sparse MLA prefill) did not merge: still `OPEN`,
> `mergedAt: null`, `MERGEABLE`/`BLOCKED`, `REVIEW_REQUIRED`, still only
> nvpohanh's `APPROVED` (09-14), no second approval. Head moved from
> `6848fe723e` to `f7348866ac` (updatedAt 2026-10-02T16:36Z) after ch-wan's
> review of 09-29 (11 findings: unvalidated union tile shape, unbounded
> `_UNION_WS`, int32 overflow at large head counts, `index_kpool > 1` not
> rejected, per-layer host sync, stale PR description, and a request to launch
> end to end on SM120 or leave the `flash_mla_sm120.py` allowlist alone) and
> nvpohanh's 10-01 comment (union numbers measured on the old bytemap path).
> yunyang1999 answered both on 10-02: per-group dedup replaced the bytemap,
> validator added, docs and tests added, description rewritten, recommended
> setting `--dsa-triton-union 2`. CI was re-triggered (`/tag-and-rerun-ci`,
> `/rerun-failed-ci` x4) and the rollup shows 20 FAILURE / 117 SUCCESS / 48
> SKIPPED, so the 09-22 "All NV pipelines have passed" state no longer holds.
> The author also states mainline DSA already runs on SM120 via
> auto-selected `flashinfer_sparse_mla` (consistent with our p34 note); the
> PR's SM120 gain is about 1% per-token, 1.13-1.14x with union at 4096 tokens
> or more.

> p34 retirement decision remains pending, unchanged from 08-15 (no new
> TP4/real-weight confirmation run requested this cycle). Even if #32779
> merges it only adds an opt-in prefill backend and does not change the p34
> question. No factual errors found in the 09-28 entry other than that
> "green NV CI" is now stale.

> Re-checked 2026-10-11: PR #31481 unchanged (head `7aa9049335de83ed1c2227c176041301a6309a29`,
> `updatedAt` 2026-09-22T09:03:28Z, no new comments or reviews, `REVIEW_REQUIRED`,
> `MERGEABLE`/`BLOCKED` on two reads 5s apart; fork branch head matches). No new SGLang tag
> (v0.5.21 still latest); `upstream/main` tip `ad8b1e1f85`. Dry-run `git merge-tree
> --write-tree upstream/main dsa-sm12x-native-sparse-mla`: clean, no rebase needed.
> `dsa_backend.py` still has the hardcoded `backend="trtllm-gen"` this PR changes; upstream
> main already contains the `flashinfer_sparse_mla` dispatch.

> PR #32779 MERGED 2026-10-06 as `e446698cdd` (final head `3e35818947`): ch-wan approved
> 2026-10-04 after the 10-02 rework, nvpohanh queued the merge, kpham-sgl waved through the
> remaining failures as flaky on main (rollup 120 SUCCESS / 22 FAILURE, labels `run-ci`,
> `bypass-fail-fast`). It adds opt-in `--dsa-prefill-backend triton_sparse_mla` and
> `--dsa-triton-union {0,2,4}` (default 0) and widens the GLM SM12x fp8-KV validator in
> `flash_mla_sm120.py` to {`flashinfer_sparse_mla`, `triton_sparse_mla`}; flashinfer stays
> the default. Orthogonal to #31481 (clean merge-tree), does not change the p34 question.
> Not in a release tag yet. The 10-03 "did not merge / 20 FAILURE" paragraph is superseded.

> Local: image 0.5.21-sm121 is deployed and GLM-5.3-Flash now runs TileLang DSA prefill and
> decode via new runtime patch p68 (v1 kernel, `tail_dim == 0`, tile block_I=32 /
> num_stages=1 / threads=128 on SM12x; GPU-verified spark5 2026-10-08, rel err ~2.2e-3 at
> H=16 and H=32); v2 (`tail_dim > 0`) stays dead on GB10. That is a separate route from the
> flashinfer-native path in this PR. p34 retirement decision still pending.
> Re-checked 2026-10-12: PR #31481 unchanged (head `7aa9049335de83ed1c2227c176041301a6309a29`,
> `updatedAt` 2026-09-22T09:03:28Z, no new comments or reviews, `REVIEW_REQUIRED`,
> `MERGEABLE`/`BLOCKED`; fork branch head matches). `git merge-tree --write-tree upstream/main`
> is clean against `f006fb49ec`. No new SGLang tag (v0.5.21 still latest), so e446698cdd
> (#32779, merged 2026-10-06) and e289989933 (#41807) are still in no tag (both postdate the
> v0.5.21 tag). Upstream moved flashinfer to 0.7.1 (#43494, 2026-10-11) and left `sgl-deep-gemm`
> at 0.2.2; no commit touched `flash_mla_sm120.py` or the DSA backend validator. No new
> SM120/SM121/GB10 PR or issue concerns sparse-MLA routing (nearest: open #42012, GLM-5.3-Flash
> on SM120, fa4 crashes at CUDA-graph capture, triton is the only working backend). The local
> image xomoxcc/dgx-spark-sglang:0.5.21-sm121 pins flashinfer 0.6.18.post1 (recipe
> scripts/patches/sglang-0.5.21-sm121.recipe); the 0.7.1 bump matters only for the next image bump.

> Re-checked 2026-11-10: PR #31481 unchanged (head `7aa9049335de83ed1c2227c176041301a6309a29`,
> `updatedAt` 2026-09-22T09:03:28Z, no reviews, last comment 2026-08-03, `REVIEW_REQUIRED`; fork
> branch head matches). `git merge-tree --write-tree upstream/main` now CONFLICTS in
> `dsa_backend.py` (two hunks: a DCP / autotune-dummy-run block, and the refactor of the
> `trtllm_batch_decode_with_kv_cache_mla` call into `attention_kwargs = dict(kv_cache=kv, ...)`,
> which collides with the `kv.view(torch.uint8) if _sparse_sm120` change) against `170804b40a`;
> a rebase is needed before any further push (not done). Correction to 10-12: v0.5.22
> (2026-10-13) and v0.5.23 (2026-10-28) both contain e446698cdd (#32779) and e289989933
> (#41807); pins are flashinfer 0.7.1 / 0.7.2 and sgl-deep-gemm 0.2.2 / 0.2.3.post2 (main:
> 0.7.3 / 0.2.7, transformers 5.19.0 throughout). Local image 0.5.21-sm121 (flashinfer
> 0.6.18.post1, sgl-deep-gemm 0.2.0) is behind both tags. `flash_mla_sm120.py` and the
> `triton_sparse_mla` validator are unchanged; `dsa_backend.py` gained DCP, KV-shard, typed KV
> view and kimi-k3.1 trtllm changes (#43735, #44280, #46990, #44503, #48714). Open: #43105
> (SM120 GLM DSA NVFP4 KV with FlashInfer), #47237 (refuse trtllm at launch outside
> SM100/SM103/SM107), #38969 (per-call Triton fallback when FlashInfer lacks the prefill shape).
> #42012 still open (fix PR #42333 closed unmerged 2026-11-05, triton the only working backend).
> Local: p34 retirement decision still pending.


## Proposed PR title

> [DSA] Enable sparse MLA decode+prefill on SM120/SM121 (consumer Blackwell) via
> flashinfer's packed sparse backend

## PR body draft (English)

### Problem

`attention_backend=dsa` with `dsa_decode_backend/dsa_prefill_backend=trtllm`
crashes on SM120/SM121 (DGX Spark GB10, RTX PRO Blackwell) with
`TllmGenFmhaRunner ... Unsupported architecture`: `_forward_trtllm` hardcodes
`backend="trtllm-gen"`, which only exists for datacenter Blackwell (SM100/103).

flashinfer (>= 0.6.x) already ships a native SM120/121 sparse-MLA implementation
(`flashinfer/mla/_sparse_mla_sm120.py`, `@supported_compute_capability([120, 121])`,
GLM_NSA/DSv3.2 model types, warp-spec decode kernels for num_tokens<=64 plus a
prefill orchestrator above that), and its dispatcher
`trtllm_batch_decode_with_kv_cache_mla` routes `cc==12 && sparse_mla_top_k>0`
to it automatically — but only with `backend="auto"`. Three small changes make
the whole DSA path (decode AND prefill) work on consumer Blackwell.

### Changes

1. **`model_runner_kv_cache_mixin.py::calculate_mla_kv_cache_dim`** — do not
   early-return the plain `kv_lora_rank + qk_rope_head_dim` layout for
   trtllm-backends on SM12x. The SM120 sparse kernel consumes the 656-byte
   packed inline-scale layout (512 fp8 nope + 4×fp32 tile scales + 64 bf16
   rope) that `quantize_k_cache` already writes; `dsa_kv_cache_store_fp8` then
   derives True automatically. SM100 keeps the early return (plain layout for
   trtllm-gen).
2. **`dsa_backend.py::_forward_trtllm`** — gate on
   `device_sm_major == 12 and self.dsa_kv_cache_store_fp8`:
   - `backend="auto"` instead of `"trtllm-gen"` (flashinfer picks `"sparse"`),
   - pass the KV cache as a `uint8` view (the sm120 checker requires it),
   - `kv_scale_format="arbitrary_fp32"`: sglang's `quantize_k_cache` writes
     amax/448 **arbitrary** fp32 tile scales, not pow2/ue8m0 — flashinfer's
     GLM_NSA scale semantics; the default "auto" (pow2) would misread them,
   - `skip_softmax_threshold_scale_factor=None` (unsupported by the sparse
     backend),
   - skip the fused rope+fp8-query-quantize branch (`mla_quantize_and_rope_for_fp8`)
     — the sparse kernel requires a **BF16 query** and dequants KV itself via
     the inline scales (first live deploy died at decode graph capture with
     "SM120 sparse MLA v32/GLM expects BF16 query, got torch.float8_e4m3fn").
3. **`deepseek_common/attention_forward_methods/forward_mla.py::_fuse_rope_for_trtllm_mla`**
   — return False for the dsa branch on SM12x so rope stays in
   `forward_absorb_prepare` and the query reaches `_forward_trtllm` in bf16
   (consequence of 2; the second call site — the extra cos_sin_cache args —
   is gated by the same function and stays consistent).

All conditionals keep the upstream values on the non-SM12x side: SM100/SM103
behaviour is byte-identical.

### Evidence (DGX Spark GB10, SM121, flashinfer 0.6.14)

- Kernel-level (GPU, real `quantize_k_cache` pool, torch dequant+softmax
  reference): decode bs=4 topk=2048 max|diff| 0.008 @ 0.072 ms/call; prefill
  2400 extend tokens max|diff| 0.016 @ 14.4 ms/layer-call; seq_lens>topk
  clamps safely (sglang passes UNCLIPPED cache_seqlens on decode); -1 padding
  skipped natively; cuda-graph capture+replay works directly.
- Live (GlmMoeDsa 504B NVFP4, TP4): boot clean, decode graph capture 18 s,
  decode 8.4 tok/s cuda-graph, prefill 873 tok/s input on a 7-seq/2240-token
  batch, GSM8K 2-shot n=20 conc 8 = 85%, 0 errors/restarts. MTP/NEXTN verify
  runs through the same kernel (accept ~2.1, +45% decode).

### Notes for reviewers

- Requires flashinfer >= 0.6.x with `mla/_sparse_mla_sm120.py` (prebuilt in
  current wheels).
- The layout coupling in `calculate_mla_kv_cache_dim` ("plain iff a backend is
  named trtllm") is the reason change 1 is needed at all; a cleaner long-term
  fix might key the layout on the actually-selected kernel rather than the
  backend name.
- Hardware for validation: any SM120/121 device (DGX Spark, RTX PRO/50-series
  Blackwell). Datacenter-Blackwell CI is unaffected by construction.
- Related context: DeepGEMM upstream declined SM12x (PR #318), so consumer
  Blackwell needs the flashinfer route; the DSA indexer needs its own fallback
  (companion PR draft: UPSTREAM_SGLANG_DSA_INDEXER_TORCH_TRITON_BACKEND.md).

## Local mapping

| local                | upstream file                                                                |
|----------------------|------------------------------------------------------------------------------|
| p34 mixin edit       | `sglang/srt/model_executor/model_runner_kv_cache_mixin.py`                   |
| p34 edits 0-3        | `sglang/srt/layers/attention/dsa_backend.py` (`_forward_trtllm`)             |
| p34 forward_mla edit | `sglang/srt/models/deepseek_common/attention_forward_methods/forward_mla.py` |

Full local chronology: `dsalogitrework.md` (PART 4 + LIVE-DEPLOY RESULT),
`DSA_speedup.md`, `dsa_cuda_graph_plan.md` §8.

## Submission checklist (recherchiert 2026-07-16)

**Unit-Tests: JA (Guide-Pflicht), aber der Kernel selbst braucht SM12x-Hardware
— Split-Strategie:**
1. **CI-lauffähig ohne SM12x** (`test/registered/unit/` spiegelt den Source-
   Tree): (a) `calculate_mla_kv_cache_dim`-Unit-Test mit gemocktem
   `get_device_capability` (12 vs. 10) und Fake-Config → 656/576-Matrix
   (Testlogik existiert schon: unsere spark5-Validierung `validate_p34.sh`
   prüft exakt diese drei Fälle); (b) `_forward_trtllm`-Kwarg-Selektion:
   flashinfer-Call monkeypatchen, Fake-Backend mit `device_sm_major=12` +
   `dsa_kv_cache_store_fp8=True` → assert `backend="auto"`,
   `kv_scale_format="arbitrary_fp32"`, uint8-View, `skip_softmax=None`, und
   auf dem 10er-Pfad Byte-Gleichheit der Upstream-Kwargs.
2. **SM12x-gebundener Kernel-Test** (skip-gated wie die bestehenden
   sm120-Quant-Tests, z.B. `test/registered/quant/test_nvfp4_gemm_sm120.py`):
   der Numerik-Test aus unserer spark5-Validierung (echtes
   `quantize_k_cache`-Pool, torch-Referenz, decode/prefill/overshoot/graph) —
   läuft in CI nur, wenn ein SM120/121-Runner existiert, sonst skip.
3. Hardware-Evidenz im PR-Body (GB10-Zahlen aus diesem Dokument), da die
   Maintainer vermutlich kein SM121 in CI haben (gleiche Lage wie DeepGEMM
   PR #318).

**Mechanik:** wie beim Companion-PR (Fork/Branch, echte Diffs gegen main —
Pfad-Restrukturierung in main beachten, `pre-commit run --all-files`,
`register_cuda_ci` für die CI-fähigen Tests, DCO beim ersten Push prüfen).
flashinfer-Mindestversion (>= 0.6.x mit `_sparse_mla_sm120`) im PR nennen und
ggf. als Import-Guard kodieren.

## Review-Lektionen aus dem Präzedenz-PR #24692 (NICHT als Vorbild, als Warnung)

Der Präzedenz-PR war KEIN Selbstläufer: 24 Tage bis zum Merge, 27 Commits,
**64 Review-Kommentare** (gemini-Bot + mehrere Menschen). Die dort gerissenen
Punkte, präventiv auf UNS angewendet:

1. **Kanonische Arch-Utils statt roher Capability-Checks** ("There is a util
   for both is_cuda and is_sm120_supported. Do not export this either"): unser
   Port ersetzt `device_sm_major == 12` / `get_device_capability()[0] == 12`
   durch die existierenden sglang-Utils (`is_sm120_supported`-Familie) — die
   lokalen Patches durften das nicht (Anker-Minimalität), der PR muss es.
2. **`-1`-Sentinel-Behandlung wurde dort explizit angemahnt** (topk_ids-
   Padding): bei uns semantisch zentral (Kernel skippt -1) → eigener Testfall
   + Kommentar an der Stelle, nicht nur Verhalten.
3. **Kein stilles Durchfallen**: für auf SM12x nicht unterstützte Randpfade
   (z.B. DSA-CP-Zweige) explizites NotImplementedError statt Misrouting —
   beim Port die `dsa_use_prefill_cp`-Zweige daraufhin prüfen.
4. **Keine degradierten Assert-Messages, keine irreführenden Aliase,
   Backend-Selektion klar strukturieren** (drei der Bot-Findings dort).
5. **Scope klein halten**: #24692 bündelte MoE+MLA+Indexer+Docs in einem PR —
   ein Grund für die 64 Kommentare. Unsere Zwei-PR-Teilung (Attention-Routing
   vs. Indexer-Backend) ist die richtige Antwort darauf; im PR-Body auf den
   Companion verweisen, aber nicht mergen-lassen-abhängig machen.
6. **Erwartungsmanagement**: mehrwöchige Review mit Iterationen einplanen;
   gemini-Bot-Review kommt zuerst und ist gründlich.


## STATUS 2026-07-16: Branch gebaut + GPU-validiert, bereit zum Einreichen

Branch `dsa-sm12x-native-sparse-mla` im Fork-Worktree `../sglang-wt-sparsemla`
(von upstream/main 1f34911de7), EIN Commit `200a884d44`, pre-commit clean,
NICHT gepusht. Tests: 11/11 CI-safe Units (kv-dim-Matrix, kwarg-Selektion,
fuse-rope-Regression) + **5/5 SM121-Kernel-Test auf GB10** (nach zwei Fixes:
KV-View muss die echte 64er-Page-Geometrie haben, sonst sind die
Decode-Kernel nicht dispatchbar und der Orchestrator rejected num_tokens<=64;
num_kv als 64er-Vielfaches). Port-Highlights: calculate_mla_kv_cache_dim ist
in main nach mem_cache/kv_cache_configurator.py umgezogen; kanonische Util
is_sm120_supported() (major==12, deckt SM121 mit ab) statt roher Checks;
expliziter NotImplementedError für DSA-Prefill-CP auf dem Sparse-Pfad.