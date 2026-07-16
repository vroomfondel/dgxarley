# DSA `flashinfer_gather` CUDA-Graph Plan/Run Split — Implementation Plan

Plan only, no implementation. Written 2026-07-16 against image
`xomoxcc/dgx-spark-sglang:0.5.15-sm121`. Companion to `dsalogitrework.md`
(the gather+dense-fa2 attention decode fallback this document fixes the
CUDA-graph story for) and `DSA_speedup.md`.

## 1. The problem (live crash, 2026-07-16)

`_forward_flashinfer_gather` (`sglang_launch.sh` patch block
`PATCH_DSA_FLASHINFER_GATHER`, method added to `DeepseekSparseAttnBackend` in
`dsa_backend.py`) calls `wrapper.plan(...)` (flashinfer
`BatchMLAPagedAttentionWrapper.plan`) **inline**, every forward call. This
crashes during decode CUDA-graph capture:

```
decode_cuda_graph_runner.py:821 run_once
  -> _forward_flashinfer_gather (dsa_backend.py:2344)
  -> wrapper.plan(...)
  -> flashinfer/mla/_core.py:1648 plan
Scheduler hit an exception ...
Hint: 2. set --cuda-graph-max-bs-decode to a smaller value
      3. disable decode CUDA graph by --cuda-graph-backend-decode=disabled
```

`plan()` does host-side dynamic work (stream sync / allocation) that is not
CUDA-graph-recordable. The SM121 hardware walls for the DSA DECODE are cleared
(indexer via the torch fallback, no `trtllm-gen` FMHA assert on decode, the
576-byte KV dequant fix) — this is the last DECODE-side implementation gap, not
a hardware dead end.

### PREREQUISITE discovered 2026-07-16: the PREFILL hits the same wall

`disable_cuda_graph: true` clears the decode-capture crash but does NOT by itself
make the model serve: the first forward (warmup = a PREFILL) crashes with
`TllmGenFmhaRunner ... Unsupported architecture` (fmhaRunner.cuh:37), because
`dsa_prefill_backend=trtllm` uses the SAME trtllm-gen FMHA the decode used to. So
the PREFILL needs its own flashinfer-reuse fallback (`dsa_prefill_backend`, a new
value; reuse flashinfer's dense MLA prefill like the base model — dense for
seq<=2048 which covers the GSM8K/smoke prompts, sparse >2048 a follow-up), being
implemented separately. That prefill fix is a PREREQUISITE for BOTH: (a) the
eager functional test (disable_cuda_graph) to serve at all, AND (b) this decode
cuda-graph fix (§7 step 3 redeploys with cuda graph back on, which only works once
the model boots + serves = prefill fixed). NOTE: prefill CUDA graph is auto-disabled
(`cuda_graph_config prefill.backend='disabled'`), so THIS document stays DECODE-only;
the prefill fallback does not need a plan/run-split, just a working SM121 kernel path.

Current workaround while both fixes are pending: `disable_cuda_graph: true` +
(once landed) the prefill fallback -> proves functional correctness (GSM8K) at
eager (slow) speed, then this decode cuda-graph fix restores decode-graph perf.

## 2. Correction to the assumed hook interface

The task that spawned this plan assumed the classic (deprecated)
`init_forward_metadata_capture_cuda_graph` / `init_forward_metadata_replay_cuda_graph`
override pair. **This SGLang version has migrated away from those** — they
are removed from the ABC. The current contract
(`sglang/srt/layers/attention/base_attn_backend.py:18-99`, `AttentionBackend`
docstring, verbatim):

- `init_forward_metadata(fb)` — eager entry point; default wraps
  `_out_graph(fb)` + `_in_graph(fb)`.
- `init_forward_metadata_out_graph(fb, in_capture=False)` — per-iteration
  metadata prep, runs **outside** `with graph.capture():`. Called at capture
  time (`in_capture=True`, once per shape, before `run_once`/the graph
  region) **and** at every replay (`in_capture=False`, before
  `graph.replay()`). Host ops / dynamic-shape / `.item()` / stream-sync logic
  belongs here.
- `init_forward_metadata_in_graph(fb)` — graph-recordable, static-shape GPU
  ops only, runs inside `with graph.capture():` and auto-replays. Lint
  contract: **must not** call `.item()`/`.cpu()`/`.tolist()`/dynamic-shape
  `torch.empty()`.

`DeepseekSparseAttnBackend` (our target class) already implements both
`init_forward_metadata_out_graph` (`dsa_backend.py:697`, dispatches to
`self._apply_cuda_graph_metadata(...)`) and `init_cuda_graph_state`
(`dsa_backend.py:1085`) for its existing decode impls (trtllm, flashmla,
tilelang, aiter). We need to extend these, not invent a parallel mechanism.

## 3. How the native `FlashInferMLAAttnBackend` gets this right

Read in full from `flashinfer_mla_backend.py`. The mechanism, precisely:

**`forward_decode` (line 595) never calls `.plan()`.** It only calls
`decode_wrapper.run(q_nope, q_rope, k_buffer_nope, k_buffer_rope, out=o, ...)`
— `run()` is the only wrapper method invoked inside the graph region.

**`.plan()` is called exclusively from `FlashInferMLAIndicesUpdaterDecode
.call_begin_forward`** (line ~727), itself only reachable via
`indices_updater_decode.update(...)`, itself only called from
`init_forward_metadata_out_graph` / `_apply_cuda_graph_metadata` — i.e.
always outside the graph, both at capture-prep and at every replay-prep:

- **Capture** (`init_forward_metadata_out_graph(fb, in_capture=True)`,
  line 308): builds a fresh `BatchMLAPagedAttentionWrapper(use_cuda_graph=True,
  qo_indptr=self.cuda_graph_qo_indptr[...], kv_indptr=self.cuda_graph_kv_indptr[...],
  kv_indices=self.cuda_graph_kv_indices, kv_len_arr=self.cuda_graph_kv_lens[...])`
  — **static, pre-allocated buffers**, sized once in `init_cuda_graph_state`
  (line 426) for `max_bs`. Calls `indices_updater_decode.update(...,
  init_metadata_replay=False)` → the **real** `.plan()` runs once, which
  populates `wrapper._cached_module` (the resolved/dispatched kernel handle).
  Only **after** that call completes:
  `decode_wrapper.plan = partial(fast_mla_decode_plan, decode_wrapper)`
  — the wrapper's `.plan` attribute is monkey-patched to a fast variant for
  every subsequent call on this wrapper instance.
- **Replay** (`in_capture=False`): calls `_apply_cuda_graph_metadata(...)`
  (line 457), which recomputes `kv_len_arr_cpu` fresh from
  `seq_lens_cpu[:bs]` (host, but this call is *outside* the graph, so a host
  op here is fine), writes it and the cumsum-derived `kv_indptr_cpu` **in
  place** into the same buffers allocated in `init_cuda_graph_state`
  (`self.cuda_graph_kv_indptr_cpu[1:bs+1] = torch.cumsum(...)`), then calls
  `indices_updater_decode.update(..., decode_wrapper=self.decode_cuda_graph_metadata[bs],
  init_metadata_replay=True, **fast_decode_kwargs)` → this reaches
  `wrapper.plan(fast_decode_kwargs["qo_indptr_cpu"], fast_decode_kwargs["kv_indptr_cpu"],
  kv_indices, fast_decode_kwargs["kv_len_arr_cpu"], ...)` — but `wrapper.plan`
  is now `fast_mla_decode_plan`, a **module-level, backend-agnostic function**
  (`flashinfer_mla_backend.py:1085`) that skips the real `.plan()`'s stream
  sync and just calls `self._cached_module.plan(...)` directly with the
  already-resolved kernel handle. This runs on every decode step (every
  replay), still outside the graph, cheaply.

**Why `.plan()` must be re-invoked every step at all:** `kv_len_arr` (real
per-request context length) genuinely changes every decode step (context
grows by 1 token each time) even for the native dense backend, so the
schedule really is recomputed every replay — just cheaply, via the fast path.

**`init_cuda_graph_state`** (line 426) is where the static buffers live:
`cuda_graph_kv_indices` (`[max_bs * max_context_len]`), `cuda_graph_qo_indptr`
/ `cuda_graph_kv_indptr` (cloned from the base `q_indptr_decode`/`kv_indptr`),
`cuda_graph_kv_lens`, plus CPU mirrors (`cuda_graph_qo_indptr_cpu`,
`cuda_graph_kv_indptr_cpu`) bundled into `self.fast_decode_kwargs`.

`DSAMetadata.page_table_1` (the pre-indexer, per-token page table, **not**
the post-topk-selection gather indices) already follows exactly this
static-buffer + in-place-`.copy_()` pattern generically for every existing
DSA decode impl (`dsa_backend.py:_apply_cuda_graph_metadata`,
`_build_forward_metadata_cuda_graph`) — this machinery is inherited "for
free"; the topk-selection / indexer output feeding into our
`_forward_flashinfer_gather`'s `page_table_1` argument is the thing this
plan is scoped to (see Section 6 open risk: unverified whether the indexer
itself is graph-safe on this stack, since no DSA decode impl has ever
reached graph capture successfully on SM121 before this backend).

## 4. The concrete fix for `_forward_flashinfer_gather`

**Core change: move `wrapper.plan(...)` entirely out of
`_forward_flashinfer_gather`.** The method should only build `ckv`/`kpe`
(gather + dequant, as today) and call `wrapper.run(q_nope, q_rope, ckv, kpe,
...)`, reading a wrapper that was already `.plan()`-ed by the out-of-graph
hook for this batch size.

Concretely, in the `sglang_launch.sh` `PATCH_DSA_FLASHINFER_GATHER` block
(extend, keep the existing marker/idempotency pattern):

1. **State (extend `__init__`, B1):** replace the single
   `self._flashinfer_gather_wrapper = None` slot with a per-bs cache dict
   (`self._flashinfer_gather_wrappers: dict[int, BatchMLAPagedAttentionWrapper] = {}`),
   mirroring `decode_cuda_graph_metadata`. Also add static buffers sized for
   `max_bs * topk` in a new `init_cuda_graph_state` extension (new patch
   anchor into `DeepseekSparseAttnBackend.init_cuda_graph_state`,
   `dsa_backend.py:1085`): `self._fig_kv_indptr_cpu`, `self._fig_qo_indptr`
   (both are pure `arange`-derived from `bs`/`topk`, no dynamic content —
   see Section 5), and `self._fig_kv_len_arr_cpu` (a pinned/CPU buffer,
   `[max_bs]`, the one genuinely-dynamic quantity).

2. **New `init_forward_metadata_out_graph` branch:** `DeepseekSparseAttnBackend
   .init_forward_metadata_out_graph` (`dsa_backend.py:697`) already calls
   `self._apply_cuda_graph_metadata(...)` for every decode impl. Add a
   post-step there (or a sibling private method called right after) gated on
   `self.dsa_decode_impl == "flashinfer_gather"` and
   `forward_mode.is_decode_or_idle()`:
   - Compute `kv_len_arr_cpu = metadata.dsa_cache_seqlens_int32[:bs].clamp(max=topk).cpu()`
     (or, if a CPU mirror of `dsa_cache_seqlens_int32` already exists on
     `metadata` from the shared DSA cuda-graph metadata build, reuse it
     instead of a fresh `.cpu()` sync — check `DSAMetadata` fields before
     assuming a new sync is needed).
   - Write it into `self._fig_kv_len_arr_cpu[:bs]` in place.
   - On first use for this `bs` (capture, `bs not in self._flashinfer_gather_wrappers`):
     construct `BatchMLAPagedAttentionWrapper(self.workspace_buffer,
     use_cuda_graph=True, qo_indptr=self._fig_qo_indptr[:bs+1],
     kv_indptr=self._fig_kv_indptr_cpu-derived-GPU-indptr[:bs+1],
     kv_indices=<static index buffer, see Section 5>, kv_len_arr=self._fig_kv_len_arr_cpu[:bs])`,
     call the **real** `.plan(...)` once (via the still-unpatched method),
     then monkey-patch: `wrapper.plan = partial(fast_mla_decode_plan, wrapper)`
     (import `fast_mla_decode_plan` from
     `sglang.srt.layers.attention.flashinfer_mla_backend` — it is a plain
     module-level function, generic to any `BatchMLAPagedAttentionWrapper`,
     **not** MLA-backend-specific; reuse it rather than reimplementing).
     Store in `self._flashinfer_gather_wrappers[bs]`.
   - On every call (capture-prep **and** every replay-prep): call
     `wrapper.plan(qo_indptr_cpu, kv_indptr_cpu, kv_indices, kv_len_arr_cpu, ...)`
     — now the fast variant — with the freshly-updated `kv_len_arr_cpu`.
     This mirrors `_apply_cuda_graph_metadata`'s per-replay `wrapper.plan()`
     call for the native backend exactly.

3. **`_forward_flashinfer_gather` becomes graph-body-only:** drop the
   `wrapper.plan(...)` call entirely; look up
   `wrapper = self._flashinfer_gather_wrappers[bs]` (bs from
   `q_nope.shape[0]` or passed through `metadata`), keep the gather + dequant
   (unchanged, produces `ckv`/`kpe`), call `wrapper.run(q_nope, q_rope, ckv,
   kpe, return_lse=False)`. This is the only wrapper method now invoked
   inside `run_once`/the graph region.

4. **Eager (non-graph) path must keep working identically:** when
   `disable_cuda_graph: true` (today's functional-test config) or for shapes
   that never get captured, `init_forward_metadata` (the eager entry, not
   `_out_graph`) still needs a `.plan()` to have run before `forward_decode`.
   Cleanest: have the plan step live in a small shared helper called from
   both `init_forward_metadata_out_graph` (graph path) and a
   `dsa_decode_impl == "flashinfer_gather"` branch added to
   `DeepseekSparseAttnBackend.init_forward_metadata`'s eager body (the
   override at `dsa_backend.py:716`) — do **not** duplicate the plan logic;
   factor it into one private method (e.g. `self._plan_flashinfer_gather(bs,
   metadata, use_fast=in_capture_or_replay)`) called from both entry points.

## 5. `kv_indices`/`qo_indptr`/`kv_indptr` for the gathered buffer: mostly static, one dynamic piece

Unlike the native backend (whose `kv_indices` point into the persistent,
never-reallocated KV cache and are genuinely per-request/per-step dynamic),
our post-gather addressing is **almost entirely compile-time-static** given
a fixed captured batch size `bs` and fixed `topk` (2048, a config constant):

- `qo_indptr = arange(0, bs+1)` — static, depends only on `bs`.
- `kv_indptr = qo_indptr * topk` — static, depends only on `bs`/`topk`.
- `kv_indices = arange(0, bs*topk)` — static: post-gather, the dequantized
  buffer is already dense/sequential per request (see the existing
  `_forward_flashinfer_gather` docstring: "page_size=1 post-gather: the
  freshly gathered/dequantized buffer is already dense per request").

**The one genuinely dynamic quantity is `kv_len_arr`**, because a request's
*real* context length can be less than `topk` (early decode steps, or short
prompts) — the DSA indexer's `topk(min(topk, end_pos))` returns fewer than
2048 valid indices in that regime, and `metadata.dsa_cache_seqlens_int32`
(already clamped to `topk` in the existing eager code) legitimately grows by
1 each decode step until it saturates at `topk`, exactly analogous to the
native backend's real `kv_len_arr`. **This means the "plan once per shape,
reuse forever" simplification is NOT valid** — `wrapper.plan()` must be
re-invoked (via the fast path) on every replay with the current
`kv_len_arr_cpu`, matching Section 4's design, not a cheaper "static-plan"
shortcut. (An initial hypothesis before this line of investigation was that
`flashinfer_gather`'s indices might be fully static and could skip the
per-step `fast_mla_decode_plan` call entirely — this section is where that
hypothesis is falsified; do not re-attempt that shortcut without first
proving `kv_len_arr` is truly step-invariant for this workload, which it is
not once short sequences are considered.)

## 6. Open risks

> **STATUS 2026-07-16 (isolated GPU micro-test, PASSED):** the core plan/run-split
> mechanic (§4) was validated on synthetic tensors in a debug pod on spark1
> (image `xomoxcc/dgx-spark-sglang:0.5.15-sm121`, GB10/SM121) BEFORE touching the
> real patch, exactly as §7 step 2 prescribes. Script:
> `scratchpad/microtest.py`. It builds `BatchMLAPagedAttentionWrapper(
> use_cuda_graph=True, backend="fa2")` over our gather shapes (bs=4, topk=2048,
> num_heads=32, head_dim_ckv=512, head_dim_kpe=64, page_size=1, **causal=True**),
> calls the REAL `.plan()` once, monkey-patches `wrapper.plan =
> partial(fast_mla_decode_plan, wrapper)`, captures a graph whose body is
> `wrapper.run(...)` ONLY, and replays. Results:
> - REPLAY#1 (same kv_len): `max|graph-eager| = 0.000000`, no NaN. **Bit-exact.**
> - REPLAY#2 (short kv_len `[2048,128,512,1]`, the padded-tail / real-context<topk
>   case): `max|graph-eager| = 0.000000`, no NaN. **Bit-exact.**
>
> This RESOLVES two of the four "open risks" below:
> - **Risk #4 (`fast_mla_decode_plan` + `causal=True`): RESOLVED.** Source-confirmed
>   (`flashinfer_mla_backend.py:1085`): `fast_mla_decode_plan` threads `causal`
>   generically to `self._cached_module.plan(...)` (no hardcoded `False`, no
>   MLA-backend gate); its `kv_indices` param is accepted but unused (the indices
>   were baked into `_cached_module` by the one real `.plan()`). Micro-test ran the
>   whole split with `causal=True` bit-exact.
> - **Risk #2 (short-seq `kv_len_arr` correctness): substantially de-risked.** The
>   per-replay fast-plan with a fresh `kv_len_arr_cpu` (mixed lengths incl. len=1)
>   matched eager bit-exact. The graph mechanic handles per-request length variation
>   inside one captured `bs`. (What remains unproven is only the *upstream* wiring:
>   that `metadata.dsa_cache_seqlens_int32` feeds the indexer's own top-k the right
>   way for len<topk. That is a model-internal question, not a graph-mechanic one.)
>
> Also source-confirmed by the live-pod extraction: `metadata.dsa_cache_seqlens_int32`
> is ALREADY a static, in-place-`.copy_()`-updated buffer (refreshed every replay by
> the existing `_apply_cuda_graph_metadata` / `fused_dsa_decode_metadata` machinery),
> so the `kv_len_arr` source needs NO new `.cpu()` sync (matches §5). And
> `self.workspace_buffer` IS allocated on SM121 (device_sm_major=12 >= 10), so the
> wrapper can be built. `DeepseekSparseAttnBackend` has NO
> `init_forward_metadata_in_graph` (base no-op) and an INDEPENDENT 298-line
> `init_forward_metadata` eager override that does NOT delegate to `_out_graph` — so
> the plan step must be added to BOTH `init_forward_metadata_out_graph` (graph) and
> the eager path (prefill is eager: `cuda_graph prefill.backend='disabled'`).
>
> Context: this micro-test only became safe to run after the live eager
> deployment was scaled to 0 (the sparks are time-sliced; an earlier debug pod
> sharing spark2's GB10 with a live TP worker triggered an NCCL collective-timeout
> that restarted worker-1 and broke the TP group — never co-locate a GPU debug pod
> with live SGLang serving on these nodes).

REMAINING open risks (deploy-only):

- **Indexer graph-safety, unverified.** No DSA decode backend has ever
  reached CUDA-graph capture successfully on this SM121 stack before
  `flashinfer_gather` (trtllm/tilelang/flashmla all died on kernel asserts
  pre-capture). The generic `page_table_1`/`_apply_cuda_graph_metadata`
  machinery exists in the source and is presumably exercised on other
  hardware, but we have zero live confirmation it works end-to-end here.
  If the indexer's own forward (torch fallback,
  `dsa_paged_mqa_logits_backend=torch`) does anything graph-unsafe
  (`.item()`, dynamic shapes) that was never hit before because capture
  never got this far, it becomes a **new** blocker discovered only once this
  fix is deployed. Test the indexer under `disable_cuda_graph=false`
  specifically, isolated if possible.
- **`kv_len_arr` correctness for the padded-tail (short-sequence) case.**
  Flagged as an open point in `dsalogitrework.md` already — the numeric
  verification there only proved plumbing (no crash/NaN), not correctness,
  for `page_table_1 < topk`. This plan's `fast_mla_decode_plan` reuse
  inherits that same unproven area; verify explicitly once graph mode is
  back on.
- **Gather+dequant buffer identity across replay (lower-confidence risk).**
  The `ckv`/`kpe` gather (`flat_kv_cache[page_table_1.reshape(-1).long()]`
  or `dequantize_k_cache_paged(...)`) still runs *inside* `run_once`/the
  graph region every call, materializing a fresh output tensor each time via
  fancy indexing (not a static-buffer in-place write). PyTorch's CUDA-graph
  private memory pool generally handles repeated same-shape allocations
  inside a captured region correctly (same pool address reused across
  captures/replays as long as the allocation pattern is identical every
  call) — this is *not* the same class of problem as `.plan()`'s host sync,
  and is not expected to be the source of the original crash (the traceback
  pointed at `wrapper.plan`, not the gather). Flag as a secondary risk to
  watch during the Step 8 GPU micro-test, not a required structural change
  up front. If it does misbehave, the fix is `torch.index_select(...,
  out=<pre-allocated static buffer>)` instead of fancy indexing.
- **`fast_mla_decode_plan` compatibility with our config unverified.** It is
  generic (any `BatchMLAPagedAttentionWrapper`), but has only ever been
  exercised with the native backend's `page_size=1, causal=False` (see
  `call_begin_forward`'s real-`.plan()` call args) vs. our
  `page_size=1, causal=True` (per the current `_forward_flashinfer_gather`
  docstring/code, `wrapper.plan(..., 1, True, ...)`). Confirm
  `fast_mla_decode_plan`'s internal `_cached_module.plan(...)` call accepts
  and correctly threads a `causal=True` config before relying on it.
- **`next_n >= 2` (MTP multi-token draft verify) is out of scope**, as in
  `dsalogitrework.md` — this plan only covers plain decode
  (`forward_mode.is_decode_or_idle()`), matching the existing
  `_forward_flashinfer_gather` scope.
- **Mixed batches** (some requests real-context < topk, others saturated)
  inside one captured `bs` shape: `kv_len_arr_cpu` naturally handles
  per-request variation (it is a `[bs]` vector already), so this should fall
  out of the design for free — but has not been tested.

## 7bis. IMPLEMENTED 2026-07-16 (`PATCH_DSA_FIG_GRAPH_SPLIT`)

The plan is implemented in `roles/k8s_dgx/files/sglang_launch.sh` as a new heredoc
block `PATCH_DSA_FIG_GRAPH_SPLIT_EOF` (after the decode + prefill gather blocks),
7 anchored edits to `dsa_backend.py` (`# [patch] _sgl_dsa_fig_graph_split_`):
- **S1** `__init__`: `self._flashinfer_gather_wrapper` (eager, kept) + new
  `_flashinfer_gather_wrappers` (per-bs graph), `_fig_static`, `_fig_plan_params`.
- **S2** `_forward_flashinfer_gather` signature: `+ is_decode: bool = True`.
- **S3/S4** method body: gather+dequant unchanged; then split — `is_graph_decode`
  (= `is_decode and decode_cuda_graph_metadata.get(bs) is not None`) → run-only on
  the per-bs wrapper; else eager inline plan+run (original behavior, verbatim).
- **S5** two new methods: `_fig_build_graph_wrapper` (build + REAL plan once +
  monkeypatch `fast_mla_decode_plan`) and `_fig_replan_graph` (out-of-graph
  build-if-params-known / fast-replan fresh kv_len).
- **S6** `init_forward_metadata_out_graph`: after `_apply_cuda_graph_metadata`,
  call `_fig_replan_graph(bs)` for decode. **S7** prefill dispatch: `is_decode=False`.

Param sourcing (the §4 "layer not available in out_graph" problem): `sm_scale` +
`num_heads` are stashed from the layer on the FIRST eager `_forward` (the
memory-profile prefill, which runs before decode capture; MLA dims are model-wide
constant so a prefill-sourced stash is correct for decode). The wrapper is then
built by whichever runs first with params known: out_graph capture-prep, or the
uncaptured warmup `run_once`. `head_dim_ckv`/`head_dim_kpe` come from
`self.kv_lora_rank`/`self.qk_rope_head_dim` (on the backend at init).

**Validated before deploy (§7 steps 1-2):**
- `bash -n` on `sglang_launch.sh`: OK.
- Full patch chain (torch-newfile/backend/indexer + gather decode/prefill + this
  block) replayed against a GB10 debug pod's `dsa_backend.py`: all "Patched...",
  **no ANCHOR-DRIFT**; `py_compile` OK; module imports; new methods + `is_decode` +
  out_graph hook + `is_graph_decode` branch all present; `fast_mla_decode_plan`
  importable. (p1 torch-backend touches out_graph-adjacent sites but the S6 anchor
  still matched after it — no clobber.)
- Core mechanic bit-exact vs eager (§6 STATUS box: microtest.py).
- Indexer graph-safety static scan: `torch_paged_mqa_logits.py` documents itself
  capture-safe (no `.item()`); `dsa_indexer.py` already has an
  `is_current_stream_capturing()` guard around its `.item()`-heavy chunking path.
- `disable_cuda_graph: false` set in the profile.

PENDING: live deploy (§7 steps 3-5) — the only remaining unverified item is the
indexer under REAL capture (deploy-only). Revert lever: `disable_cuda_graph: true`.

## 7ter. LIVE DEPLOY RESULT 2026-07-16 — cuda-graph SUCCEEDED, prefill then failed

Chronological, so the two outcomes are not conflated: **the cuda-graph fix this
document specifies WORKED and is proven. A SEPARATE, pre-existing defect in the
prefill (not introduced by this fix) then crashed the run.**

### (a) The cuda-graph plan/run-split: PROVEN (§7 steps 3-4 PASS)

Profile: `disable_cuda_graph: false`, `max_total_tokens: 131072`, dsa +
torch-indexer + flashinfer_gather decode/prefill. Deploy 2026-07-16 ~11:43.

```
Load weight end. elapsed=1063.08 s, avail mem=26.47 GB, mem usage=78.99 GB
KV Cache is allocated. #tokens: 131072, KV size: 7.51 GB      (was 15.01 GB @256k)
Capture target decode CUDA graph begin. backend=full, bs=[1,2,4,8,12,16,24,32]
Capturing batches (bs=32..1): 0/8 -> 100%   avail_mem ~11.5-12.8 GB throughout
Capture target decode CUDA graph end. elapsed=21.39 s, mem usage=1.66 GB
The server is fired up and ready to roll!
```

- **Capture completed 100%, no crash.** No `wrapper.plan` inside capture, no
  `flashinfer_gather graph wrapper missing` assert (the build-ordering
  belt-and-suspenders held), and **the §6 "indexer graph-safety" open risk did NOT
  materialise** — the torch indexer captured fine (consistent with the static scan:
  `torch_paged_mqa_logits.py` is `.item()`-free and `dsa_indexer.py` already has an
  `is_current_stream_capturing()` guard around its `.item()`-heavy chunking path).
- Smoke (bs=1) coherent + correct ("Paris"), 120 tok in ~15 s wall.
- **Decode: `cuda graph: True`, gen throughput 8.19-8.41 tok/s vs ~5 tok/s eager
  (+65%)** (§7 step 5). The graph removed the launch/Python overhead; the remaining
  floor is the torch indexer's own (unfused) compute, not the graph.

**Conclusion: this document's fix is done and validated. The remaining DSA perf
ceiling is the indexer kernel (DSA_speedup.md), not cuda-graph.**

### (b) The prefill defect that ended the run (NOT a cuda-graph issue)

GSM8K (2-shot, n=20, concurrency 8) against the exposed endpoint killed
**worker-2 and worker-3** (head survived, 0 restarts; TP group broken by the dead
ranks). Identical traceback on both:

```
dsa_backend.py:2089  forward_extend          <- PREFILL/extend, not decode
  -> _forward_flashinfer_gather
dsa_backend.py:2433  ckv = gathered[..., :v_head_dim].contiguous()
torch.AcceleratorError: CUDA error: an illegal memory access was encountered
```

**Root cause: memory explosion in the prefill gather, NOT a bad index.**
Investigated and falsified in order:
- Padding-sentinel hypothesis REJECTED: `_pad_topk_indices` pads with **-1** and
  `_get_fused_topk_page_table` returns topk_indices unchanged, but
  `flat_kv_cache[-1]` is *legal* in torch (negative indexing = last row). -1 cannot
  produce an illegal access. A clamp would have been the wrong fix.
- REAL cause: `_forward_flashinfer_gather` materialises `[num_tokens_q * topk, 576]`
  **plus an fp32 intermediate** (`gathered_fp8.to(torch.float32)`) = **4.7 MB per
  query token** (2048*576*4). Decode has `num_tokens_q = bs <= 32` -> ~75 MB, fine.
  **Prefill has `num_tokens_q` = ALL extend tokens in the batch:**

  | case | num_tokens_q | fp32 intermediate |
  |---|---|---|
  | smoke (bs=1, #new-token 64) | 64 | ~0.3 GB — worked |
  | GSM8K 2-shot x8 concurrent | ~2400 | **~11.3 GB** — avail was ~11 GB -> boom |
  | chunked_prefill_size cap | 8192 | **~38 GB** |

  This is why the bs=1 smoke passed in BOTH the eager and the cuda-graph deploys and
  only concurrent/ragged prefill crashed: the defect scales with prefill tokens.

**The design error is mine and predates the cuda-graph fix:** the prefill patch
"reuse the decode impl unchanged" is memory-naive. Gathering top-2048 KV *per query
token* is O(num_tokens_q x topk) in both memory AND work — 614k gathered KV rows per
layer for a 300-token prompt, x60+ layers. That is also why prefill measured
**~1-5 tok/s**. §1 of THIS document already prescribed the right thing and the
implementation missed it: *"reuse flashinfer's DENSE MLA prefill like the base
model — dense for seq<=2048 ... sparse >2048 a follow-up"*.

### (c) Why the prefill fix is a project, not a patch (SM121 dead-end)

- **Chunking the gather** bounds peak memory (fixes the crash) but NOT the work:
  prefill stays ~1-5 tok/s, i.e. an 8-request batch needs 8-40 min of prefill.
  GSM8K remains impractical. Crash-fix only, not a solution.
- **Dense prefill is the correct fix** (for `seq <= 2048`, top-2048 IS the whole
  context -> dense is NUMERICALLY IDENTICAL, with one fused kernel instead of 614k
  gathers; it would fix the crash AND the slowness). But on SM121 there is no
  ready path inside `DeepseekSparseAttnBackend`:
  - `_forward_trtllm_extend` (dsa_backend.py:2669) takes
    `flashinfer.prefill.trtllm_ragged_attention_deepseek` for `device_sm_major >= 10`
    — GB10 is sm_major=12 -> the `TllmGenFmhaRunner ... Unsupported architecture`
    assert. **Dead.**
  - `_forward_standard_mha` needs real `k`/`v` + "dense multi-head config"; this
    model runs MLA-**absorb** (`forward_absorb_core` in the traceback). Not usable.
  - The only SM121-working dense MLA prefill is
    `BatchPrefillWithRaggedKVCacheWrapper` in `FlashInferMLAAttnBackend` (what the
    dense baseline uses). Porting it into the DSA backend means rebuilding the
    absorb / KV-layout plumbing = a real implementation project.

### (d) Status / decision taken

DSA on GB10 as of 2026-07-16:

| component | status |
|---|---|
| indexer (torch fallback) | works; is the decode perf floor |
| decode (flashinfer_gather + cuda-graph) | **works, proven, 8.4 tok/s** |
| prefill | **no viable path**: trtllm dead, gather memory-explosive + ~1-5 tok/s, chunking insufficient, dense = project |

**Decision (user, 2026-07-16): STOP here, commit + document.** Before investing in
the dense-prefill project, the honest next step is to measure the DENSE BASELINE
(`attention_backend: flashinfer`, one profile line, zero code) to get the tok/s
number DSA must justify itself against. If dense clearly beats 8.4 tok/s decode AND
has fast prefill, the DSA prefill rebuild is not justifiable on this hardware.

Nothing here is wasted: the patches are INERT unless their backend combo is selected
(see the profile's PATCH-ACTIVATION CONTRACT), the plan/run-split is proven and
reusable, and the indexer fallback stays available.

**Live state at stop:** sglang deployments scaled/left with the main instance
deployed (head 2/2 healthy, worker-2/-3 restarted once -> TP group needs a clean
restart before serving again). embed/vision were torn down by the user.

## 7. Verification / test plan

1. Static: `bash -n` on the extended `sglang_launch.sh`; the extracted
   heredoc applied against the live image (`Patched`/`already applied`
   idempotency, per the established pattern); `py_compile` on the mutated
   files.
2. **GPU micro-test before touching the real patch further:** in an isolated
   debug pod, construct a tiny synthetic decode CUDA graph (small `bs`,
   `topk`) that exercises exactly the new split — real `.plan()` once,
   monkey-patch to `fast_mla_decode_plan`, `graph.capture()` a body that only
   calls `wrapper.run(...)`, then `graph.replay()` after mutating
   `kv_len_arr_cpu`/re-running the fast plan with a *different* real KV
   content — and numerically compare against the eager (non-graph) path for
   the same inputs. This isolates the CUDA-graph mechanics from the rest of
   the 25-minute model boot and would have caught the original crash in
   seconds instead of a full deploy cycle.
3. Live deploy: flip `disable_cuda_graph: false` back on in the profile,
   redeploy, confirm capture completes without the `wrapper.plan` traceback,
   confirm `Capturing batches` reaches 100%, head goes Ready.
4. Correctness: re-run the GSM8K smoke/eval already used for the eager
   (`disable_cuda_graph: true`) functional test, compare graph-mode output
   token-for-token (or at least accuracy-for-accuracy) against the eager
   run's captured baseline — a silent graph-replay staleness bug (stale
   buffer contents from a previous batch size/shape) would not crash, it
   would just produce wrong tokens.
5. Throughput: compare `gen throughput` graph-on vs. graph-off (today's
   eager baseline) and vs. the dense `flashinfer` baseline, to quantify the
   actual win before investing further (e.g. the warp-MMA path from
   `DSA_speedup.md`).
