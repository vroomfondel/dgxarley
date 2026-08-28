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

| local | upstream file |
|---|---|
| p34 mixin edit | `sglang/srt/model_executor/model_runner_kv_cache_mixin.py` |
| p34 edits 0-3 | `sglang/srt/layers/attention/dsa_backend.py` (`_forward_trtllm`) |
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