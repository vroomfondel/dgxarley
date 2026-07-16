# DSA Paged-MQA-Logits Torch Fallback — Port Plan

Plan only, no implementation. Written 2026-07-16 against image
`xomoxcc/dgx-spark-sglang:0.5.15-sm121`. Companion to `DSA_speedup.md`
(background: why GB10/SM121 needs this) and memory
`reference_glm52_dsa_indexer_deepgemm_sm121`.

> **READ PART 4 (end of file) FIRST for the current state**: flashinfer ships a
> NATIVE SM120/121 sparse-MLA (decode + prefill); `p34` wires it. The attention
> chronology in PARTs 2-3 (gather decode, prefill design error) is the trail
> that led there; the indexer part of THIS document (PART 1) remains current.

## STATUS 2026-07-16 (post-implementation): DONE but NOT sufficient on its own

Phase 1 of this plan was implemented (5 runtime patch blocks in `sglang_launch.sh`, opt-in
`dsa_paged_mqa_logits_backend=torch`), numerically verified, committed, and LIVE-DEPLOYED. It
WORKS: the boot got past the DeepGEMM indexer assert and past KV alloc (all 3 metadata call
sites gated). BUT a full DSA kernel-path survey then showed the indexer is only ONE of several
SM121 blocks: the MLA decode attention (`dsa_decode_backend=trtllm` -> trtllm-gen FMHA) also
hard-asserts on sm_121, and every other attention backend is dead too except `tilelang`, which
compiles on sm121 but OOMs GB10 shared memory. So porting THIS indexer fallback does not by
itself unblock DSA-sparse / MTP; the attention side needs its own kernel work (tilelang tile
re-tuning for GB10 smem) or upstream consumer-Blackwell support. Full survey + verdict:
`DSA_speedup.md` (KERNEL-PATH SURVEY 2026-07-16). Prod stays on `attention_backend="flashinfer"`
(dense MLA). This plan/port remains valid + committed for when the attention side is unblocked.

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

---

# PART 2 - The attention decode: gather + reuse dense fa2 (implementation plan)

The indexer torch fallback (Part 1) got the boot past the DeepGEMM assert but then crashed at
the MLA decode attention (`dsa_decode_backend=trtllm` -> trtllm-gen FMHA `Unsupported architecture`).
A full survey found EVERY dedicated DSA attention kernel dead on SM121 (trtllm-gen ISA, flashmla
not built, fa3 hard gate, tilelang smem/compile contradiction - proven). But a GPU-prototyped
alternative works: **gather the top-2048 selected KV and run flashinfer's DENSE MLA decode (fa2,
already working on SM121) over the gathered subset.** This is REUSE, not a new kernel.

## Why it works (GPU-prototyped 2026-07-16, PASS)

- The gather prep ALREADY exists: `dsa_backend.py::forward_decode` builds
  `page_table_1 = transform_index_page_table_decode(page_table, topk_indices, page_size=1)`
  (`dsa/transform_index.py`) -> `[bs, 2048]` int32 KV-slot indices, for every backend. In the
  real sparse regime (seq>2048) `index_score.topk(min(topk, end_pos))` returns exactly 2048 valid
  indices, no -1 padding -> static shape, CUDA-graph-friendly.
- `flashinfer_mla_backend.py::BatchMLAPagedAttentionWrapper.plan()` takes `page_size` as a param and
  holds arbitrary non-contiguous `kv_indices`/`kv_indptr` (gathers in-kernel). Feed
  `plan(page_size=1, kv_indices=page_table_1.flatten(), kv_len=2048)` + `.run(q_nope, q_pe, k_buffer)`.
  Auto-selects fa2 on SM121 - the same kernel that already serves the dense base model.
- Numerics: vs a pure-torch gather + Q.K^T softmax .V reference: max abs diff 0.00068 (bf16), no NaN,
  seed-varying. Perf: sparse-2048 0.26 ms vs dense-8192 0.80 ms (sub-linear -> real win at long ctx).

## THE blocker: fp8 KV

flashinfer's MLA decode rejects `kv_cache_dtype=fp8_e4m3` on EVERY backend
(`FP8 kv_data_type for MLA is only supported with the fa3 backend on SM90`; fa3 is SM121-dead).
Fix options, cheapest first:
1. **Dequant ONLY the gathered 2048 KV to bf16 per decode step** (recommended). Cheap because it is
   2048 entries x dims, not the whole 256k cache. The gather already materialises those 2048 slots;
   add a `.to(bf16)` (with the fp8 scale) on the gathered k_buffer before `.run()`.
2. bf16 KV cache for this path - doubles KV memory (13 GB -> 26 GB), tight at mem_fraction 0.9-0.99
   (would need a lower max_total_tokens).
3. Wait for flashinfer upstream fp8 MLA on consumer Blackwell.

## Implementation (Phase 1 = plain decode, next_n==1)

1. New `dsa_decode_backend` value `flashinfer_gather` (enum + choices + resolve, mirroring how we
   added `torch` to `dsa_paged_mqa_logits_backend`). Opt-in, no arch gate, not in `auto`.
2. `_forward_flashinfer_gather(...)` branch in `dsa_backend.py::forward_decode`: take the existing
   `page_table_1`, construct/plan a `BatchMLAPagedAttentionWrapper` with `page_size=1` +
   `kv_indices=page_table_1`, dequant the gathered 2048 KV to bf16 (option 1), run, return.
3. Deliver as runtime patch block(s) in `sglang_launch.sh` per the established anchor + ANCHOR-DRIFT
   pattern. Wire a profile knob (`dsa_decode_backend` + the render, mirroring
   `dsa_paged_mqa_logits_backend`). Do NOT change behavior on non-SM121 / other backends.

## Verification

- Static: bash -n, real-anchor apply against the live image, py_compile, idempotency.
- GPU numeric: the gathered-subset fa2 result vs a torch gather+softmax reference (max abs diff,
  no NaN/all-zero, seed-varying) - already prototyped, re-check inside the wired path.
- Live (deploy): base with `attention_backend=dsa` + `dsa_paged_mqa_logits_backend=torch` +
  `dsa_decode_backend=flashinfer_gather`: boots past both the indexer AND the attention, GSM8K
  correct vs the dense baseline, throughput.

## Open points

fp8 dequant correctness (scale handling), CUDA-graph compat of the wrapper plan-rebuild
(plan-per-batch vs graph replay), mixed batches where some requests have seq<2048 (-1 padding in
`page_table_1`), and `next_n>=2` (MTP multi-token draft verify - not covered by Phase 1, a follow-up).

### RESOLVED 2026-07-16 (live)

- **CUDA-graph compat: SOLVED.** The plan-rebuild is moved out of the captured region
  (per-bs wrapper, real `.plan()` once, monkeypatch to `fast_mla_decode_plan`, only
  `wrapper.run()` captured). Implemented as `PATCH_DSA_FIG_GRAPH_SPLIT`, validated
  bit-exact vs eager on synthetic tensors AND live: capture 100%, decode
  `cuda graph: True` @ **8.4 tok/s** (vs ~5 eager). Full record: `dsa_cuda_graph_plan.md`
  §6 STATUS / §7bis / §7ter.
- **fp8 dequant + the -1 padding: NOT the crash.** `_pad_topk_indices` pads with `-1`
  and `_get_fused_topk_page_table` passes topk_indices through unchanged, but
  `flat_kv_cache[-1]` is *legal* torch (negative indexing = last row), so -1 cannot
  cause an illegal access. The clamp/mask hypothesis was investigated and FALSIFIED.

# PART 3 - The PREFILL reuse was a DESIGN ERROR (2026-07-16, live crash)

**Do not reuse `_forward_flashinfer_gather` for prefill/extend.** The
`dsa_prefill_backend=flashinfer_gather` patch ("reuses the decode implementation
UNCHANGED") is memory-naive and was the direct cause of a live worker crash.

## The failure

GSM8K (2-shot, n=20, concurrency 8) killed worker-2 + worker-3 (head survived; the
dead TP ranks broke the group). Identical traceback:

```
dsa_backend.py:2089  forward_extend         <- PREFILL/extend
  -> _forward_flashinfer_gather
dsa_backend.py:2433  ckv = gathered[..., :v_head_dim].contiguous()
torch.AcceleratorError: CUDA error: an illegal memory access was encountered
```

(CUDA errors are async - the illegal access happens in the gather a line earlier, the
`.contiguous()` is only where it surfaces.)

## Root cause: O(num_tokens_q x topk) memory AND work

`_forward_flashinfer_gather` materialises `[num_tokens_q * topk, 576]` **plus an fp32
intermediate** (`gathered_fp8.to(torch.float32)`) = **4.7 MB per query token**
(2048 * 576 * 4 B).

| case | num_tokens_q | fp32 intermediate |
|---|---|---|
| decode (bs<=32) | <=32 | ~75 MB - fine, this is why decode works |
| smoke prefill (bs=1, #new-token 64) | 64 | ~0.3 GB - worked |
| GSM8K 2-shot x8 concurrent | ~2400 | **~11.3 GB** (avail was ~11 GB) -> crash |
| chunked_prefill_size cap | 8192 | **~38 GB** |

**Decode's `num_tokens_q` is the batch size (1 query token per request). Prefill's is
EVERY extend token in the batch.** That is the asymmetry the "reuse unchanged" claim
missed. It explains why a bs=1 smoke passed in both the eager and the cuda-graph
deploys while concurrent prefill died: the defect scales with prefill tokens.

It is also why **prefill measured ~1-5 tok/s**: gathering top-2048 KV *per query token*
is 614k gathered KV rows per layer for a 300-token prompt, x60+ layers.

## The right fix: DENSE prefill (and why it is a project on SM121)

For `seq <= 2048`, top-2048 selects the ENTIRE context -> **dense prefill is numerically
identical to sparse**, with one fused kernel instead of 614k gathers. It fixes the crash
AND the slowness. (`seq > 2048` = real sparse prefill, still a follow-up.) This is what
`dsa_cuda_graph_plan.md` §1 prescribed from the start; the implementation missed it.

But no ready dense path exists inside `DeepseekSparseAttnBackend` on SM121:

- `_forward_trtllm_extend` (dsa_backend.py:2669) uses
  `flashinfer.prefill.trtllm_ragged_attention_deepseek` for `device_sm_major >= 10`;
  GB10 is sm_major=12 -> `TllmGenFmhaRunner ... Unsupported architecture`. **Dead.**
- `_forward_standard_mha` requires real `k`/`v` + dense multi-head config; this model
  runs MLA-**absorb** (`forward_absorb_core`). Not usable.
- The only SM121-working dense MLA prefill is `BatchPrefillWithRaggedKVCacheWrapper`
  (`FlashInferMLAAttnBackend`, what the dense baseline uses). Porting it into the DSA
  backend = rebuilding the absorb / KV-layout plumbing. **Project, not a patch.**

**Chunking the gather is NOT a solution:** it bounds peak memory (crash goes away) but
not the work -> prefill stays ~1-5 tok/s (8-40 min of prefill for one 8-request batch),
so GSM8K stays impractical.

## Consequence

`dsa_prefill_backend=flashinfer_gather` should be considered **unsafe for concurrent /
long prefill** in its current form. Before building the dense-prefill project, measure
the dense baseline (`attention_backend: flashinfer`, one profile line, zero code) to get
the number DSA must justify itself against. See `dsa_cuda_graph_plan.md` §7ter(d).

# PART 4 - RESOLVED: flashinfer ships NATIVE SM120/121 sparse MLA; p34 wires it (2026-07-16)

**Supersedes PART 3's "dense prefill = project" verdict AND replaces PART 2's gather
decode as the primary path.** During the planned dense-prefill inspection, the image's
flashinfer 0.6.14 turned out to already contain `flashinfer/mla/_sparse_mla_sm120.py`:
a native sparse-MLA paged attention, `@supported_compute_capability([120, 121])`,
PREBUILT (no JIT wall), with an explicit **GLM_NSA model type** (d_qk=576,
arbitrary-fp32 inline scales), `(16, 2048)` in the decode dispatch set (= our TP4
heads + index_topk), native `-1`-padding skip, warp-spec decode kernels for
num_tokens<=64 and a **prefill orchestrator** above that.

## The whole "trtllm wall" was ONE hardcoded argument

`flashinfer.decode.trtllm_batch_decode_with_kv_cache_mla` (the function
`_forward_trtllm` ALREADY calls, with `sparse_mla_top_k=index_topk`) routes
`cc==12 && sparse_mla_top_k>0` to the native sparse backend - but only with
`backend="auto"`. SGLang hardcodes `backend="trtllm-gen"` -> the
`TllmGenFmhaRunner Unsupported architecture` assert that motivated the entire
gather workaround. Upstream main still hardcodes it (checked 2026-07-16) ->
upstream-PR candidate.

## What p34 does (`p34_dsa_trtllm_sparse_sm120.py`)

1. `model_runner_kv_cache_mixin.py::calculate_mla_kv_cache_dim`: skip the
   "trtllm -> plain 576 layout" early-return on SM12x, so the pool keeps the
   656-byte packed layout (512 fp8 + 4x fp32 tile scales + 128 B bf16 rope) the
   sm120 kernel consumes - the SAME layout `quantize_k_cache` already writes
   (live-tested by the gather deploys). `dsa_kv_cache_store_fp8` flips True
   automatically (derived from the override dim).
2. `dsa_backend.py::_forward_trtllm`: on `device_sm_major==12 &&
   dsa_kv_cache_store_fp8` pass `backend="auto"`, a uint8 view of the KV buffer,
   and `kv_scale_format="arbitrary_fp32"` - sglang's quantizer writes amax/448
   arbitrary fp32 scales (source-verified, NOT pow2/ue8m0), which is exactly
   flashinfer's GLM_NSA semantics; the default "auto" would misread them as
   DSv3.2 pow2. skip_softmax forced None (sparse backend raises on it).
   SM100/SM103 byte-identical via the else sides.

Profile: `dsa_decode_backend: trtllm` + `dsa_prefill_backend: trtllm` (was
flashinfer_gather/flashinfer_gather). `dsa_paged_mqa_logits_backend: torch` (p30)
stays - the indexer is untouched by all of this and remains the decode perf floor.
p31/p32/p33 remain as the documented gather fallback (decode-only; the gather
PREFILL stays design-broken per PART 3).

## GPU verification (spark5 podman, GB10, real quantize_k_cache, torch reference)

| case | numerics | time |
|---|---|---|
| decode bs=4 topk=2048 | max\|diff\| 0.008 | 0.072 ms/call (gather: 0.26 ms) |
| decode bs=32 | ok | 0.236 ms/call |
| decode seq_lens>topk (sglang's UNCLIPPED cache_seqlens) | PASS, kernel clamps | - |
| seq_lens=None / heavy -1 padding | PASS | - |
| prefill 2400 extend tokens (the GSM8K conc-8 killer) | max\|diff\| 0.016 | 14.4 ms/layer |
| prefill 8192 (chunked_prefill_size) | ok | 48.5 ms/layer |
| cuda-graph capture+replay | finite, works DIRECTLY | 0.239 ms/replay |

The seq_lens subtlety: the sparse entry interprets a `[num_tokens]`-shaped
seq_lens as ACTIVE top-k length per token (a `[batch]`-shaped one is ignored ->
all columns active, -1 skipped). Both of sglang's shapes (per-token
`dsa_cache_seqlens_int32` for extend - already topk-clipped - and per-request
`cache_seqlens_int32` for decode, unclipped) verified safe.

Prefill for the GSM8K batch: ~2400 tokens x 79 layers ~= 1.1 s attention total,
vs the gather impl's 11.3 GB OOM crash and 8-40 min projection. And cuda-graph
needs NO plan/run split on this path (the kernel is a plain custom op).

Validation: full patch chain applied in a pristine container (36 patched, 0
ANCHOR-DRIFT), py_compile + import OK, idempotency (second run changed nothing),
mixin unit test on GB10 (656 trtllm+fp8 / 656 gather-combo unchanged / 576 bf16).
mypy strict + black clean. NOT yet cluster-deployed.

## LIVE-DEPLOY RESULT 2026-07-16 (p34, two iterations)

**Deploy 1 (FAILED at decode graph capture, all 4 ranks):** `ValueError: SM120
sparse MLA v32/GLM expects BF16 query, got torch.float8_e4m3fn`. Root cause: for
dsa+trtllm+fp8, `_fuse_rope_for_trtllm_mla` (forward_mla.py) skips rope upstream
and `_forward_trtllm` fuse-ropes AND fp8-quantizes the query
(`mla_quantize_and_rope_for_fp8`, meant for trtllm-gen's fp8xfp8 BMM). The sparse
kernel dequants KV itself (inline scales) and requires a BF16 query. The spark5
kernel test had covered the flashinfer call, not sglang's query prep before it.
Fix = p34 edit 3 (`_fuse_rope_for_trtllm_mla` returns False on SM12x -> rope runs
normally upstream) + skipping the fp8-quantize branch for `_sparse_sm120`; q stays
bf16, k/k_rope reach the packed store in bf16 (its quantize_k_cache asserts that
anyway). Crash location itself was informative: patches, weight load, packed KV
alloc (656) and the routing INTO `_trtllm_batch_decode_sparse_mla_v32_sm120` were
all correct.

**Deploy 2 (SUCCESS, the current state):**

| milestone | result |
|---|---|
| boot | clean, 0 restarts, 0 ANCHOR-DRIFT, all 37 patches applied |
| weight load | 1056 s, avail 26.57 GB |
| KV alloc | 131072 tokens, 7.51 GB, fp8 packed (656 B) |
| decode graph capture (deploy-1 crash point) | 18.02 s, bs 1..32, avail 11.55 GB after |
| smoke | coherent ("Paris. The city is located on the banks of the Seine...") |
| decode | **8.4 tok/s, cuda graph: True** (= gather path; indexer-bound as predicted) |
| prefill, the gather-killer shape | **7 seqs / 2240 tokens @ 873 tok/s input** (gather: 1-5 tok/s + OOM) |
| GSM8K 2-shot n=20 conc 8 (the crash repro) | **17/20 = 85%, 0 errors, 0 restarts, 420.7 s total** |

## NEXT: the indexer is now the only decode lever (plan, user-approved 2026-07-16)

Decode is pinned at ~8.4 tok/s by the p30 torch indexer (unfused reference impl;
attention is now ~5.7 ms of a ~119 ms token). Agreed sequence:

1. **Option 1 - flashinfer-native indexer?** Inspect flashinfer 0.6.14 on spark5
   for a native SM120/121 fp8-paged-MQA-logits / DSA-indexer kernel: whoever built
   the GLM_NSA sparse-attention path plausibly ships the indexer too. Same method
   as the p34 find (module scan + GPU numerics test vs the torch fallback). If it
   exists: wire like p34 (small patch), done.
2. **Option 2 - Triton fusion (fallback):** fuse the torch chain (index-k gather +
   dequant + q.k logits + relu-weighted head sum) into one Triton kernel on the
   p30 dispatch point. Bounded, SM121-safe, replaces only the kernel inside the
   existing `dsa_paged_mqa_logits_backend=torch` plumbing.

Context for prioritisation: concurrency already amortises the indexer (aggregate
conc-8 throughput >> 8.4), and MTP (now reachable, verify runs through the same
sparse kernel) multiplies decode independently -- it may be worth more than
indexer work; decide after Option 1's answer is known.

## NEXT-PLAN RESULTS 2026-07-16 (same day): Option 1 NEGATIVE, Option 2 IMPLEMENTED (p35)

**Option 1 (flashinfer-native indexer): NEGATIVE, definitively.** flashinfer 0.6.14
in the image has no indexer/fp8-paged-MQA-logits kernel (module scan), and the
upstream flashinfer main tree (2294 files, scanned 2026-07-16) contains zero
mqa/indexer/lightning candidates. flashinfer covers the sparse ATTENTION only.
sgl_kernel ships the topk side (`fast_topk*`) and a DSV4 q-prep fusion
(`dsv4_fused_q_indexer_rope_hadamard_quant`), but not the logits scoring.

**The measurement that pinned the floor** (spark5 GB10, exact live shapes): the p30
torch logits kernel costs **1.476 ms/layer x 79 = 116.6 ms/token** at bs=1,
page-table width 131072 (the cuda-graph CAPTURE width), true seq 300 -- live decode
is ~119 ms/token (8.4 tok/s). Match to within ~2%. Root cause is structural:
`max_seq_len = block_tables.shape[1] * 64` (dsa_indexer.py:827) is the capture
constant, and torch gathers + fp32-dequants + bmms the FULL width every step;
torch cannot do data-dependent work under graph capture. Counterfactual: a tight
table costs 0.097 ms -> ~10x decode headroom. (Prefill is unaffected: its
max_seq_len is the true batch max, eager -- hence the healthy 873 tok/s.)

**Option 2 (Triton fusion): IMPLEMENTED as `p35_dsa_indexer_triton_logits.py`.**
One Triton program per (request, 64-token page): STATIC launch grid over the full
table width (graph-safe) with per-block EARLY EXIT on the true `seq_lens[b]` read
at replay time, fused fp8-load + inline-scale dequant + q.k dot + relu + weighted
head-sum, no fp32 HBM intermediates. Delivered as a new module
(`dsa/triton_paged_mqa_logits.py`) + a dispatch inserted into the p30 torch
fallback after its shape asserts; env kill-switch `SGLANG_DSA_INDEXER_TRITON=0`
reverts to pure torch. Activation contract unchanged
(`dsa_paged_mqa_logits_backend: torch`).

GPU-verified on spark5: **bit-exact** vs the torch reference (max|diff|=0.0 direct;
3.6e-7 through the patched module) incl. identical -inf masks; cuda-graph
capture+replay correct WITH a seq-len change in the static buffer between capture
and replay; **0.024 ms vs 1.476 ms at the live decode point (61x)** -> indexer cost
~1.9 ms/token instead of ~116.6. Harness: 0 drift, idempotent. Expected live
effect: decode moves from 8.4 tok/s to the next floor (attention ~5.7 ms + MoE
weight-bandwidth; plausibly ~25-40 tok/s single-stream). NOT yet deployed.

## p35 LIVE RESULT 2026-07-16 + the REAL decode floor (profiled; hypothesis CORRECTED)

**p35 deployed and working:** boot clean, decode graph capture 16.8 s with the
Triton kernel captured (capture memory HALVED: 0.71 vs 1.45 GB - no fp32
intermediates), smoke coherent, dispatch verified active in the live server.
The indexer logits kernel no longer appears in the top-16 CUDA kernels of a
live decode profile.

**BUT decode stayed at ~8.4 tok/s - the "indexer is the floor" claim was WRONG
at the live shapes, and the earlier 116.6-vs-119 ms match was coincidence.** The
bench assumed page-table width 131072 (the KV pool); live `context_length` is
16384 (profile cap) -> table width 16384, where the torch logits cost only a
small fraction. Lesson: the width assumption was never verified against
server_args.

**The REAL floor (torch profiler, live head, 22 decode steps, 88% GPU-busy,
~131 ms kernel time/token):**

| ms/token | share | what |
|---|---|---|
| ~59 | ~45% | cuBLAS **bf16 GEMV** (166 us x ~3.25/layer) + dsv3_fused_a_gemm + lm_head: the UNQUANTIZED bf16 MLA projections (modelopt kept attention dense) - pure weight-bandwidth at bs=1 |
| ~27 | ~21% | cutlass grouped GEMM: the NVFP4 MoE |
| ~12 | ~10% | small bf16 wmma GEMMs (kv_b / indexer projections) |
| ~10 | ~8%  | NCCL AllReduce (61 us x 2/layer, TP4) |
| ~3  | ~2.5%| sparse_mla_decode_dsv3_2 (the p34 attention - as predicted, tiny) |

**Correctness regression PASSED:** GSM8K 2-shot n=20 conc 8 with the Triton
kernel in the decode graph: **18/20 = 90%, 0 errors, 0 restarts, 342 s** (p34
baseline: 17/20, 421 s; the accuracy delta is one problem = noise at n=20).

**Consequences:** for SHORT contexts there is nothing left to win in the DSA
software stack; the levers are now (a) **MTP** (multiplies tokens per weight
read; unlocked by p34, verify runs through the same sparse kernel) and
(b) batching. The bf16 projections are checkpoint-inherent (a re-quantization
question, not a serving-code one). p35 remains valuable: the logits cost scales
linearly with table width, so at a 128k-context profile it WOULD be the wall
(1.476 ms/layer measured at width 131072); plus the halved capture memory.

## PHASE 2 IMPLEMENTED 2026-07-16 (MTP verify support in the torch/triton indexer)

The p30 dispatch's `next_n >= 2` NotImplementedError is replaced by the Section-3
"option b" folding, and it turned out even smaller than planned: at the call site,
`seqlens_32_2d` for target-verify/draft-extend-v2 comes from
`get_seqlens_expanded()` and is ALREADY per-token `[q_offset, 1]`, and q/weights are
already sliced per token (`[:q_offset]`). The ONLY per-request tensor is
`block_tables [B, W]` -> `repeat_interleave(next_n, dim=0)` maps row b to tokens
b*next_n..(b+1)*next_n-1. The torch module and the p35 Triton kernel need ZERO
changes (both are per-token by construction; the folding happens in the dispatch).
Graph-safe (static shapes).

Verified on spark5 (B=4, next_n=4, mixed ctx lens 64..1900, live 256-page width):
folded call vs a per-token-loop reference is **bit-exact on BOTH kernel paths**
(triton and pure torch); patch chain idempotent, 0 drift.

Profile: `speculative_enabled: true` (NEXTN, 3 steps, 4 draft tokens). All MTP
preconditions are now met: p42 (NVFP4 NextN weight load, live-validated), p34
(verify runs through the native sparse kernel via the decode impl), Phase 2
(indexer next_n>=2). Rationale: the profiled decode floor is bf16-projection
weight BANDWIDTH -> MTP multiplies tokens per weight read (GLM-4.7 reference:
+68% single-stream). NOT yet deployed.
