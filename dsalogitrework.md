# DSA Paged-MQA-Logits Torch Fallback — Port Plan

Plan only, no implementation. Written 2026-07-16 against image
`xomoxcc/dgx-spark-sglang:0.5.15-sm121`. Companion to `DSA_speedup.md`
(background: why GB10/SM121 needs this) and memory
`reference_glm52_dsa_indexer_deepgemm_sm121`.

## 1. Problem, exact failure point

`0xSero/glm-5.2-reap-504B-v2` (`GlmMoeDsaForCausalLM`) uses DeepSeek Sparse Attention
via `sglang/srt/layers/attention/dsa/dsa_indexer.py::Indexer`. On decode, `Indexer`
routes through `_get_topk_paged()` (dsa_indexer.py:793), which needs a paged-MQA-logits
kernel to score the paged KV cache before top-k selection. The only backends SGLang
offers for that kernel are enumerated in
`sglang/srt/layers/attention/dsa/paged_mqa_logits_backend.py`:

```python
class DSAPagedMQALogitsBackend(Enum):
    DEEPGEMM = "deepgemm"
    CUTEDSL = "cutedsl"
    AITER = "aiter"
```

- `deepgemm` (the `auto` default on CUDA): calls `deep_gemm.get_paged_mqa_logits_metadata()`
  and `deep_gemm.fp8_paged_mqa_logits()`. On GB10/SM121, `get_paged_mqa_logits_metadata`
  throws a COMPILED C++ assert `Unsupported architecture`
  (`deepgemm/csrc/apis/attention.hpp:227`) — verified live in a GPU debug pod 2026-07-16.
- `cutedsl`: gated at `DSAPagedMQALogitsBackend.resolve()` behind `is_sm100_supported()`;
  raises `ValueError` at config-resolve time on GB10 (SM121 != SM100 in that check), before
  any kernel runs.
- `aiter`: requires ROCm, not applicable on GB10 (CUDA).

There is currently **no fourth option**. This plan adds one: `torch`, a pure-PyTorch
paged-MQA-logits implementation that runs on any CUDA arch (no DeepGEMM, no CuTeDSL),
trading kernel throughput for correctness + availability.

### The torch fallback is the ONLY viable path — both hardware kernel routes are dead ends

This is not a "try the cheap option first" ordering call — the other two routes were
checked (by parallel investigation, 2026-07-16) and are **definitively closed** on
GB10/SM121, not just currently-unconfigured:

- **DeepGEMM SM121 support was rejected upstream.** DeepGEMM's own maintainer declined
  the SM121 admission PR (upstream PR #318) citing lack of hardware/capacity to support
  it. This is not a local gate we can flip — there is no admitted SM121 arch in DeepGEMM's
  compiled kernel set to relax into, and no indication that will change on a timeline
  relevant to this work.
- **cutedsl fails on a real ISA boundary, not a soft version gate.** `CuteDSLPagedMQALogitsRunner`'s
  MMA setup (`_setup_mma`) uses `tcgen05.MmaF8F6F4Op`, a Blackwell **datacenter-only**
  (SM100/SM103) tensor-core instruction family that does not exist on consumer Blackwell
  (SM121). `cute.compile()` fails with an explicit architecture-mismatch error to that
  effect ("expects arch sm_100a/103a..., got sm_121a; on consumer Blackwell use
  warp-level cute.nvgpu.warp.MmaF16BF16Op"). This is unlike the cutlass.cute FP4-GEMM
  admission gap the image already patches (which was a missing arch string in an
  otherwise-portable kernel) — here the kernel genuinely does not run on SM121 hardware
  at the ISA level. Relaxing `is_sm100_supported()` in `resolve()` would not help; the
  kernel itself would still fail to compile/dispatch for this GPU. No image-level fix,
  no flag.

Both routes are closed for structural/upstream reasons independent of anything in this
repo's Dockerfile or patch set. **The torch fallback in this plan is not "the fast option
to try first" — it is the only remaining path to a working sparse indexer on this
hardware**, short of DeepGEMM eventually adding SM121 support upstream (Section 6 of
`DSA_speedup.md`, marked as the least controllable / longest-timeline option there too).

### Two call sites that must both be handled

`deep_gemm.get_paged_mqa_logits_metadata()` is called in **two** places, not one:

1. `sglang/srt/layers/attention/dsa_backend.py` (`DeepseekSparseAttnBackend.init_forward_metadata`,
   ~lines 953-991 and ~1297-1320, two near-duplicate code paths) — this precomputes
   `paged_mqa_schedule_metadata` **unconditionally on `is_cuda()`** for every decode /
   target-verify / draft-extend-v2 batch, independent of which `paged_mqa_logits_backend`
   is configured. This runs BEFORE the indexer is ever reached.
2. `sglang/srt/layers/attention/dsa/dsa_indexer.py::_get_topk_paged` (~line 883) — a
   fallback call, only reached `if schedule_metadata is None` (i.e. if step 1 didn't already
   produce it, which today it always does on CUDA).

**A torch fallback must skip DeepGEMM's metadata call at both sites**, not just avoid the
kernel call. Missing this is the single most likely way a first attempt half-works then
still crashes on the metadata precompute.

### Resolved: does the torch path even need the DeepGEMM schedule metadata? No.

This is the load-bearing fact that makes skipping both call sites safe rather than just
convenient. Verified directly in the reference implementation (`dsv4/indexer.py`, source
inspected in the debug pod): both torch fallback functions receive a `deep_gemm_metadata`
parameter and immediately discard it —

```python
def fp8_paged_mqa_logits_torch_sm120(..., deep_gemm_metadata: Any, ...) -> torch.Tensor:
    """CUDA-graph-compatible FP8 paged MQA logits for SM120 (vectorized, no .item())."""
    _ = deep_gemm_metadata
    ...
```

— identically in the non-sm120 sibling `fp8_paged_mqa_logits_torch`. The parameter is
never read anywhere in either function body. This is not an oversight; it reflects a real
structural difference: DeepGEMM's kernel is a **scheduled, SM-tiled** kernel launch, and
`get_paged_mqa_logits_metadata()` computes the per-SM work partition it needs to run
efficiently. The torch path does no such scheduling — it does one full vectorized gather
of all relevant KV pages via `page_table` indexing (`kvcache_flat[page_ids]`) followed by
a dense `torch.bmm`, i.e. it recomputes the same mathematical result a completely
different way that has no notion of an SM schedule at all. **The two DeepGEMM metadata
call sites can therefore simply pass `None` (or be skipped, leaving the field unset) when
`paged_mqa_logits_backend == TORCH`; the torch kernel function does not consume it in any
form.** This is the concrete, verified answer to whether the metadata call is a second,
independent blocker to work around, or a cheap no-op to skip: it is the latter — nothing
downstream of the torch path depends on `paged_mqa_schedule_metadata` ever being computed.

### Provenance of the reference implementation

`fp8_paged_mqa_logits_torch_sm120` was added upstream via SGLang PR #24692 (merged
2026-06-01), first shipped in v0.5.13, and is present unchanged in this cluster's
v0.5.15-sm121 image (confirmed: the function exists at the expected location with the
expected CUDA-graph-safe, no-`.item()` structure, matching the PR's stated design goal).
It is CUDA-graph-compatible by construction and was written specifically to be a
hardware-availability fallback, the same category of problem this plan addresses for DSA.
It is currently wired ONLY into `dsv4/indexer.py` (DeepSeek-V4's C4 indexer) — the goal of
this plan is to give `dsa_indexer.py` (deepseek_v2 / GlmMoeDsa's indexer) an equivalent.

### Where this blocks (both target and draft)

`_get_topk_paged` is invoked whenever `forward_mode.is_decode_or_idle() or
is_target_verify() or is_draft_extend_v2()` (dsa_indexer.py ~line 1897) — i.e. on EVERY
decode step, target or draft, single-token or speculative-verify batch. This is why the
earlier live test found `attention_backend="dsa"` crashes the TARGET on its very first
decode token, and why MTP/NEXTN (whose `is_nextn` indexer always computes topk_indices,
per `forward_mla.py`) cannot avoid it either. A working torch fallback unblocks both:
plain decode under `attention_backend="dsa"`, and the MTP draft/target-verify path.

## 2. Reference implementation (already in the image, for a different model)

DeepSeek-V4's indexer (`sglang/srt/layers/attention/dsv4/indexer.py`) already ships a
CUDA-graph-safe torch fallback, gated by env var `SGLANG_FP8_PAGED_MQA_LOGITS_TORCH` +
`is_sm120_supported()`:

```python
elif envs.SGLANG_FP8_PAGED_MQA_LOGITS_TORCH.get():
    if is_sm120_supported():
        fn = fp8_paged_mqa_logits_torch_sm120
    else:
        fn = fp8_paged_mqa_logits_torch
```

`fp8_paged_mqa_logits_torch_sm120(q_fp8, kvcache_fp8, weight, seq_lens, page_table,
deep_gemm_metadata, max_seq_len, clean_logits=True)` (dsv4/indexer.py:163-222):

- Inputs: `q_fp8` shape `(batch, 1, num_heads, head_dim=128)`, fp8; `kvcache_fp8` shape
  `(num_pages, block_size=64, 1, head_dim+4=132)` uint8 (128 fp8 bytes + 4 bytes storing an
  fp32 dequant scale per kv slot, reinterpreted via `.view(dtype=...)`); `weight` shape
  `(batch, num_heads)`; `seq_lens` shape `(batch,)`; `page_table` shape `(batch, max_pages)`;
  `deep_gemm_metadata` — **ignored** (`_ = deep_gemm_metadata`), which is exactly the hook we
  need: the torch path never touches DeepGEMM's schedule metadata.
- Computation: gather KV pages via `page_table` indexing, split each 132-byte row into 128
  fp8 bytes (view as `FP8_DTYPE`) + 4 bytes (view as `float32` scale), dequantize
  (`kv_value.to(float32)`), `torch.bmm(kv_value, q.transpose(1,2))`, ReLU, multiply by
  per-head `weight`, sum over heads, multiply by `kv_scale`, mask positions beyond
  `seq_lens` with `-inf`, output `(batch, max_seq_len)` float32 logits.
- No `.item()` / no data-dependent Python control flow → CUDA-graph-capture-safe (the
  `_sm120` variant specifically; the non-sm120 `fp8_paged_mqa_logits_torch` is the
  eager-mode-only sibling, `assert clean_logits == False` in both).

`deepgemm_paged_mqa_logits_split` (the DSA-side wrapper actually invoked for plain decode,
`sglang/jit_kernel/dsa/paged_mqa_logits.py:35-56`) calls its `fn` with:

```python
q_fp8 = q_fp8.unsqueeze(1)          # (num_tokens, n_heads, head_dim) -> (num_tokens, 1, n_heads, head_dim)
fn(q_fp8[:q_offset], kv_cache_fp8, weights[:q_offset], ctx_lens_2d, block_tables,
   schedule_metadata, max_seq_len, clean_logits=False)
```

This is a **near-exact signature match** to `fp8_paged_mqa_logits_torch_sm120` for the
plain-decode case (`num_tokens` plays the role of dsv4's `batch_size`). The KV cache layout
also matches: DSA's `kv_cache_fp8.view(shape[0], 64, 1, 132)` (dsa_indexer.py ~line 887) is
byte-for-byte the same `(pages, 64, 1, 132)` layout as dsv4's. This is why porting is
tractable rather than a rewrite.

## 3. Where DSA's shapes diverge from dsv4 (the actual porting work)

- **`q_fp8` is natively 3D in DSA** (`(num_tokens, n_heads, head_dim)`, asserted at
  dsa_indexer.py:839 `assert len(q_fp8.shape) == 3`), vs dsv4's natively-4D
  `(batch, 1, num_heads, head_dim)`. The DSA wrapper (`deepgemm_paged_mqa_logits_split`)
  already does the `unsqueeze(1)` to produce the 4D shape dsv4 expects — so for the
  **plain-decode path** (next_n == 1, i.e. `deepgemm_paged_mqa_logits_split`), NO reshape
  work is needed beyond what the existing wrapper already does; `fp8_paged_mqa_logits_torch_sm120`
  can be called as-is.
- **Target-verify / speculative multi-token (`deepgemm_paged_mqa_logits_native`,
  `next_n >= 2`)** reshapes to `(B, next_n, n_heads, head_dim)` — a genuinely 4D batch with
  `next_n > 1` in the second dim, which `fp8_paged_mqa_logits_torch_sm120` CANNOT handle as
  written (it hardcodes `assert q_fp8.shape == (batch_size, 1, num_heads, head_dim)`, i.e.
  dim 1 must be exactly 1). Using it for `next_n > 1` needs either (a) a loop over the
  `next_n` slices calling the sm120 fn per-slice (simple, correctness-first, no CUDA-graph
  concern if this path is only used outside piecewise capture — needs checking), or (b) a
  small generalization of the sm120 function to fold `next_n` into the batch dim before the
  `bmm` and reshape back after (`q.view(B*next_n, ...)`, everything else broadcasts the same
  way since `page_table`/`seq_lens` are already per-(batch,next_n) via `block_tables[::next_n]`
  repeat-pattern — needs verifying against `ctx_lens_2d`'s actual shape at that call site).
- **`ctx_lens_2d` vs 1D `seq_lens`**: `_get_topk_paged` passes `seqlens_32_2d` (shape
  `(B, 1)` normally, or `(B, next_n)` for the native/target-verify path). dsv4's fallback
  already handles the `(B, 1)` case (`if seq_lens.dim() > 1: seq_lens = seq_lens.squeeze(-1)`);
  the `(B, next_n)` case needs the same generalization as the q_fp8 reshape above.

**Recommended scope split:**
- **MVP (phase 1):** plain decode only (`forward_mode.is_decode_or_idle()`, next_n==1,
  i.e. `deepgemm_paged_mqa_logits_split`'s call shape). This alone unblocks: (a) the target
  model's ordinary token-by-token decode under `attention_backend="dsa"` (restores real
  sparse attention instead of the current silent dense-MLA fallback under `flashinfer`),
  and (b) each individual MTP/NEXTN draft step (the draft autoregressively decodes one token
  at a time internally, before the multi-token verify batch is assembled) — this is likely
  sufficient to unblock the `topk_indices` TypeError crash we hit at draft cuda-graph capture,
  since graph capture happens over the draft's own single-token decode shape.
- **Phase 2 (stretch):** `is_target_verify()` / `next_n >= 2` (the batched draft-token
  verification step on the target side). Needed for MTP to be fully correct+fast in the
  verify phase, not just bootable. Do only after phase 1 is live-validated; the shape
  generalization is small but must not be assumed correct without a numeric check (Section 5).

## 4. Concrete change list

1. **`sglang/srt/layers/attention/dsa/paged_mqa_logits_backend.py`**
   - Add `TORCH = "torch"` to `DSAPagedMQALogitsBackend`.
   - Add `is_torch(self) -> bool`.
   - In `resolve()`: accept `value == "torch"` → return `TORCH` (no arch gate needed — that's
     the whole point of this backend). Optionally, make `auto` prefer `TORCH` when
     `not is_sm100_supported()` and DeepGEMM's own arch probe would fail — but doing this
     silently changes the default for every DSA model incl. ones DeepGEMM DOES support
     (H100/B200); **do not** make `auto` imply `torch`, keep it opt-in only, to avoid
     silently regressing performance elsewhere. This mirrors the `_sgl_mixed_nvfp4_variant_`
     lesson: an automatic behavior change for models that already work correctly is worse
     than an explicit opt-in flag.

2. **`sglang/srt/server_args.py`**
   - Add `"torch"` to `DSA_PAGED_MQA_LOGITS_BACKEND_CHOICES` (line ~298).
   - Update the `--dsa-paged-mqa-logits-backend` help string to mention the new option
     (line ~1379).

3. **New function, `sglang/srt/layers/attention/dsa/torch_paged_mqa_logits.py`** (new file,
   keeps the DSA package self-contained rather than importing across from dsv4):
   - Phase 1: adapt `fp8_paged_mqa_logits_torch_sm120` verbatim (it already matches DSA's
     post-`unsqueeze(1)` call shape and KV layout) — copy, don't import cross-module (dsv4 and
     dsa are meant to be independent code paths; a shared import creates an unwanted coupling
     and risks a future dsv4-only refactor breaking dsa). Rename to
     `fp8_paged_mqa_logits_torch_dsa` for clarity.
   - Phase 2: extend with a `next_n` dim (see Section 3) once phase 1 is validated.

4. **`sglang/srt/layers/attention/dsa_backend.py`** (`init_forward_metadata`, both call
   sites ~969 and ~1307):
   - Gate the `deep_gemm.get_paged_mqa_logits_metadata(...)` call: skip it (leave
     `paged_mqa_schedule_metadata = None`) when
     `self.paged_mqa_logits_backend is DSAPagedMQALogitsBackend.TORCH` (this object/attribute
     needs to be accessible on the backend instance — check how `dsa_indexer.py`'s
     `self.paged_mqa_logits_backend` is populated at `Indexer.__init__` line 448-449 and mirror
     that in `dsa_backend.py` if it is not already the same instance/resolve call).

5. **`sglang/srt/layers/attention/dsa/dsa_indexer.py::_get_topk_paged`** (~lines 883-945):
   - Skip the fallback `deep_gemm.get_paged_mqa_logits_metadata(...)` call (~883) when
     `self.paged_mqa_logits_backend.is_torch()`.
   - Add a new branch parallel to the existing `is_aiter()` / `use_cute_dsl` / `use_dg_native`
     / else-split chain:
     ```python
     elif self.paged_mqa_logits_backend.is_torch():
         logits = fp8_paged_mqa_logits_torch_dsa(
             q_fp8.unsqueeze(1)[:q_offset],
             kv_cache_fp8,
             weights[:q_offset],
             seqlens_32_2d,
             block_tables,
             None,              # schedule_metadata unused by the torch path
             max_seq_len,
             clean_logits=False,
         )
     ```
     (exact arg order/slicing to be copied from `deepgemm_paged_mqa_logits_split`'s call,
     Section 2 — this pseudocode is illustrative, verify against the live function signature
     at implementation time since line numbers will have shifted since this plan was written.)
   - `use_cute_dsl` / `use_dg_native` selection logic upstream in the same function must also
     be short-circuited to `False` when `is_torch()`, so the torch branch is reached instead
     of falling into the cutedsl/dg_native paths for target-verify batches (phase 1 explicitly
     does not support those — must raise a clear `NotImplementedError` there, not silently
     mis-route, until phase 2 lands).

## 5. Correctness verification

**"It boots and doesn't crash" is explicitly NOT sufficient sign-off for this patch — it is
a necessary but nowhere near sufficient condition.** A community fork (kt-sglang) shipped
torch/tilelang indexer fallbacks for SM120 that run to completion without error but return
WRONG results (all-zero or NaN logits) — i.e. a plausible-looking, crash-free indexer that
silently produces garbage top-k selections. That failure mode is worse than a crash: a
crash is loud and gets fixed, a silently-wrong sparse indexer would quietly serve degraded
or nonsensical completions while everything LOOKS healthy (server Ready, no errors in the
log, requests return text). Any implementation of this plan must clear a real numeric bar
before being considered done, not just a boot check:

- **Numeric reference**: DeepGEMM's `fp8_paged_mqa_logits` cannot run on this hardware to
  compare against directly. Two options: (a) run the torch fallback against DeepGEMM on a
  borrowed SM100/H100 box if one becomes available (out of scope for this cluster), or
  (b) derive a second, independent reference by hand from the DSA indexer's DEFINITION of
  the scoring function (dot-product of dequantized fp8 K against fp8 Q, ReLU, weighted sum
  over heads, scaled — this is exactly what `fp8_paged_mqa_logits_torch_sm120` already
  implements, and the DSA/dsv4 KV cache layouts are confirmed identical in Section 2, so a
  faithful port should be correct by construction if the layout/shape assertions hold).
  Practical verification without SM100 hardware: unit-test the ported function against a
  small hand-built KV cache + known expected top-k indices (a handful of tokens, checkable
  by hand — e.g. construct KV pages where exactly one page has an obviously-dominant score
  by construction, assert the indexer picks it), not against DeepGEMM output. Also
  explicitly check for the kt-sglang failure signature: assert the output logits/top-k
  indices are NOT all-zero, NOT all-NaN, and NOT identical across distinct random KV
  content (a fallback that returns a constant regardless of input would still "not crash").
- **End-to-end**: boot GLM-5.2-REAP with `attention_backend="dsa"`,
  `dsa_paged_mqa_logits_backend="torch"`, MTP off first. Confirm: no DeepGEMM assert, no
  `topk_indices` TypeError, model boots to Ready, and — critically — run the existing GSM8K
  harness (`eval_gsm8k.py`, same 50-question / temp 0.6 setup already used for this model)
  and compare accuracy against the `flashinfer`-backend (dense MLA) baseline. A correct sparse
  indexer should score close to or better than dense MLA on GSM8K, not measurably worse; a
  large accuracy regression signals a bug in the torch port, not an acceptable trade-off.
  Given the kt-sglang precedent, a GSM8K score that looks merely "plausible" is not enough
  confirmation on its own — pair it with the unit-level checks above, since a broken indexer
  that always selects the same/no tokens could still pass some fraction of GSM8K by luck on
  short-context questions.
- **Then** re-enable MTP (`speculative_enabled: true`) and re-run the same GSM8K harness +
  check the draft acceptance rate in the head log, now that the draft's decode steps have a
  working indexer path.

## 6. Performance expectation

The torch fallback is not free — it replaces a fused custom kernel with `bmm` + elementwise
ops issued as separate torch/CUDA kernel launches. Expected to be markedly slower than a
native DeepGEMM/CuteDSL kernel would be (on hardware where those work), but the indexer's
paged-logits step is a comparatively small part of one decode step's total compute vs the
main MoE/MLA forward pass — the actual throughput impact needs live measurement (`gen
throughput` in the head log, per `DSA_speedup.md`'s established methodology), not assumed.
Two outcomes are both plausible and both useful to know: (a) real sparse attention working
via torch fallback, even if the indexer itself is slower than dense-MLA-on-fa2's baseline —
worth it if MTP's speculative speedup outweighs a slower indexer; or (b) net neutral/negative
vs. staying on dense MLA — in which case the value of this work is specifically enabling MTP
(which structurally requires the indexer to run at all), not raw decode speed.

## 7. Delivery mechanism

**Runtime patch in `sglang_launch.sh`**, following the established anchor + `ANCHOR-DRIFT:`
pattern used throughout that file (see the `_sgl_dsnextn_mixed_mtp_` patch as the closest
precedent — also a DSA/MTP-related source patch). Concretely: one Python heredoc block per
touched file (5 files: `paged_mqa_logits_backend.py`, `server_args.py`, the new
`torch_paged_mqa_logits.py` (written via `p.write_text()`, not a patch — it doesn't exist
upstream), `dsa_backend.py` (2 call sites), `dsa_indexer.py`), each with a marker string for
idempotency and an `ANCHOR-DRIFT:` fallback message if the anchor no longer matches on a
future image bump. Gate the whole block on the model being DSA-capable
(`GlmMoeDsaForCausalLM` / model name match), matching the existing pattern for other
model-specific patches, to keep it a no-op for non-DSA models.

**Why not an upstream PR first:** this is exploratory/local-only until phase 1 is
live-validated (Section 5) — a PR without a working correctness story is not useful upstream.
Once validated on this cluster, upstreaming becomes reasonable (SGLang already has the dsv4
precedent for exactly this kind of hardware-availability fallback, so a `dsa`-side equivalent
is a plausible, reviewable PR) — but that is a follow-up decision, not part of this plan.

## 8. Risks

- **Two-call-site gating (Section 1) is easy to half-do.** If only `dsa_indexer.py`'s
  fallback call is gated but `dsa_backend.py::init_forward_metadata`'s eager precompute is
  missed, the DeepGEMM assert still fires before the indexer is ever reached — this must be
  the first thing verified, not assumed from reading the code.
- **CUDA-graph capture compatibility.** GLM-5.2-REAP's MTP draft is captured into a CUDA
  graph (`eagle_draft_cuda_graph_runner.py`); the torch fallback must be graph-capture-safe
  (no `.item()`, no dynamic Python control flow depending on tensor values) — `sm120`'s
  vectorized design already satisfies this for dsv4, but must be re-verified after any
  DSA-specific shape changes (Section 3) don't reintroduce a `.item()` call.
  A capture failure here would look similar to the original `topk_indices` crash but with a
  different traceback — do not assume phase 1 is done just because it imports cleanly.
  reference: [[reference_glm52_dsa_indexer_deepgemm_sm121]].
- **Phase 2 shape generalization is unverified guesswork** (Section 3) until implemented and
  tested — do not ship it without the numeric verification in Section 5; scope phase 1 and 2
  as separate, separately-validated changes, not one patch.
- **`self.paged_mqa_logits_backend` instance sharing between `dsa_backend.py` and
  `dsa_indexer.py`** needs confirming — if they resolve independently from server args
  rather than sharing one object, both need the `TORCH` value wired consistently or the
  backend and indexer could disagree on which path to take.
- **Silent wrong-results, not crash, is the realistic failure mode — treat it as the
  default assumption, not an edge case.** The kt-sglang precedent (Section 5) is a fork
  that shipped essentially this same kind of fallback and got it subtly wrong (all-zero/NaN
  logits) without it being obvious from normal operation. Do not let "server booted, GSM8K
  didn't obviously collapse" stand in for real verification; budget time for the unit-level
  numeric checks in Section 5 as a mandatory step, not an optional nice-to-have, before
  calling any phase of this plan done.
- **Both hardware kernel routes being closed (see the new subsection in Section 1) means
  there is no fallback-to-a-fallback if the torch port has unfixable correctness or
  performance problems.** If the torch path turns out unusable, the only remaining options
  are staying on dense MLA (current state, known-correct, known-slow, no MTP) or waiting on
  upstream DeepGEMM SM121 support with no committed timeline. Scope expectations
  accordingly before investing significant implementation time.
