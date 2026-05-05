# SGLang Upstream Bug: `cutlass_moe_fp4` `a_map` uninitialized-memory OOB under EP

## Status (re-verified 2026-05-04)

**Partial progress, semantic fix invalidated.** 2026-04-11 session outcome
(no further investigation since; PR #20869 still stale at upstream — last
activity 2026-03-18, re-checked 2026-05-04). The cluster-level workaround
remains: NVFP4 MoE profiles default to `moe_runner_backend: flashinfer_cutlass`,
which avoids `cutlass_moe_fp4` entirely (see CLAUDE.md "NVFP4 MoE runner is
model-specific, not global"):

- **Crash is fixed**: the device-side assert at `nvfp4_blockwise_moe.cuh:78`
  no longer fires — `torch.empty → torch.zeros` on `a_map`/`c_map` in
  `cutlass_moe_fp4` eliminates the immediate OOB.
- **Output is garbage**: generated tokens are meaningless (repeated `!`
  characters, Chinese junk chars) even though the pipeline no longer
  crashes. The "non-local slots multiply by zero weights" hypothesis
  behind `torch.zeros` is wrong — the fake-gathered row-0 values DO
  contribute to the final output.
- **Our `sgl-kernel-sm121.patch` was unnecessary**: an A/B test with
  the vanilla upstream `scitrera/dgx-spark-sglang:0.5.10` image (no
  sm121 CUTLASS patches) produced the same behavior — no cuh:78
  crash, same garbage output. Auto-carveout on SM121 already picks
  a valid stage count; CUTLASS#3144 either doesn't apply to this
  kernel's tile shapes or was fixed at the CUTLASS level we're
  pulling. The sm121 kernel patches remain in `scripts/patches/`
  as documented defense-in-depth in case #3144 resurfaces for
  other models or tile shapes.

The `_shuffle_rows_torch` OOB itself is now fully explained and the
surface fix works, but the semantic path needs a second layer of fix
(topk_weights masking or c2/c_map scatter discipline) before the
triton/cutlass MoE backend under EP produces correct outputs. That
work is in progress — see "Open: semantic fix" below.

**Unreported as a root-cause analysis** upstream, but adjacent work exists:

- [PR #20869](https://github.com/sgl-project/sglang/pull/20869) ("fix(moe): support EP
  for modelopt FP4 MoE weight processing") — **open, unmerged, stale since 2026-03-18**
  (re-verified 2026-05-04, no review activity in 6+ weeks). Fixes the two earlier errors in the chain
  (shape mismatch on input-scales, `num_experts != num_local_experts` assertion)
  with Python-level changes to `modelopt_quant.py`, but **does not fix the
  `_shuffle_rows_torch` OOB described here**. The PR author instead sidesteps
  it by changing `server_args.py` to auto-route SM120 to the `flashinfer_cutlass`
  backend, which bypasses `cutlass_moe_fp4` entirely.
- [PR #21630](https://github.com/sgl-project/sglang/pull/21630) — narrower
  overlapping fix for the same input-scale slicing, still unmerged.
- [Issue #20011](https://github.com/sgl-project/sglang/issues/20011) — same
  class of bug on 8×B200 + Kimi-K2-Thinking-NVFP4, surfacing as an IMA via
  NCCL watchdog instead of the device-side assert.

Bug exists in SGLang v0.5.10 (and v0.5.10.post1 by inspection — same code path).

The final root cause (uninitialized `torch.empty` on `a_map`) was identified
during our sm121 CUTLASS SMEM debug session on 2026-04-11 after chasing it
through three wrong suspects — see "The ordeal" below.

Files:
- `sglang/jit_kernel/nvfp4.py`, function `scaled_fp4_experts_quant` (calls `_shuffle_rows_torch` at line ~300)
- `sglang/jit_kernel/nvfp4.py`, function `_shuffle_rows_torch` at line 257 (performs the OOB `index_select`)
- Called from `sglang/srt/layers/moe/cutlass_moe.py`, `cutlass_moe_fp4` line ~451
- Called from `sglang/srt/layers/quantization/modelopt_quant.py`, `ModelOptNvFp4FusedMoEMethod.apply` line ~2027

This is the actual root cause of the device-side assert previously observed at
`sglang/jit_kernel/csrc/moe/nvfp4_blockwise_moe.cuh:78` (documented in
`SGLANG_TP_EP_MOE_UPSTREAM_BUG.md`, section "Deeper Issue: cutlass_fp4_group_mm CUDA kernel assert with EP").
With `CUDA_LAUNCH_BLOCKING=1` the crash surfaces one kernel earlier, inside
`scaled_fp4_experts_quant`, before the FP4 group-GEMM is ever launched — so the
CUTLASS kernel itself is not necessarily broken, it was just the first synchronous
point after an out-of-bounds `index_select` on a preceding stream.

## Affected Configuration

- Quantization: `modelopt_fp4` (NVFP4-quantized MoE models)
- Expert Parallelism: `ep_size > 1`
- MoE runner backend: `triton` (which falls back to `cutlass_moe_fp4` for NVFP4) — also `cutlass` direct, same code path
- Tested with: `nvidia/Qwen3-235B-A22B-NVFP4` (128 experts), TP=4, EP=4, `scitrera/dgx-spark-sglang:0.5.10`, SM121 (GB10)

The `flashinfer_cutlass` MoE runner backend takes a different code path
(`flashinfer` fused MoE) and is **not affected** — it does not go through
`cutlass_moe_fp4` → `scaled_fp4_experts_quant`.

## The Bug

`cutlass_moe_fp4` quantizes the routed hidden states per expert by calling
`scaled_fp4_experts_quant`, which internally builds a `dst2src_map` (= `a_map`)
and shuffles the input rows to group tokens by destination expert:

```python
# sglang/jit_kernel/nvfp4.py, _shuffle_rows_torch (line 257)
output = input_tensor.index_select(0, dst2src_map.to(dtype=torch.int64))
```

The actual root cause is a **pure Python-level uninitialized-memory bug**:

```python
# sglang/srt/layers/moe/cutlass_moe.py, cutlass_moe_fp4 (lines 436-437)
a_map = torch.empty((topk_ids.numel()), dtype=torch.int32, device=device)
c_map = torch.empty((topk_ids.numel()), dtype=torch.int32, device=device)
prepare_moe_input(topk_ids, ..., a_map, c_map, params.num_experts, ...)
```

Under EP > 1, `StandardDispatcher` remaps `topk_ids` from global expert IDs
(0..127 for Qwen3-235B, 0..255 for MiniMax-M2.5) to local IDs [0..num_local_experts-1]
with **-1 sentinels** for experts that belong to other EP ranks.

`prepare_moe_input` dispatches a CUDA kernel `compute_arg_sorts` that iterates
`blockIdx.x` over `[0, num_experts)` and, for each block, scans the flat
`topk_ids` looking for entries equal to its own `expert_id`. When it finds
one, it atomically increments an offset counter and writes
`a_map[slot] = i / topk` (the source row index in the local token tensor `a`).

The critical observation: **there is no block for `expert_id = -1`**. Slots
in `a_map` whose corresponding `topk_ids[i]` is -1 are therefore **never
written**. They retain whatever `torch.empty` allocated — uninitialized GPU
memory, i.e. garbage.

Downstream, `_shuffle_rows_torch` does `input_tensor.index_select(0, a_map)`
on the **full** `a_map` (including the garbage slots), and the garbage values
trip torch's CUDA `vectorized_gather_kernel` bounds check:

```
/build/pytorch/aten/src/ATen/native/cuda/IndexKernelUtils.cu:16:
    vectorized_gather_kernel: block: [..], thread: [..]
    Assertion `ind >=0 && ind < ind_dim_size
        && "vectorized gather kernel index out of bounds"` failed.
```

(Thousands of such lines in the log — one per thread in the failing launches.)

## The ordeal (how we got here)

This bug cost an entire afternoon of debugging across three wrong suspects,
mainly because every layer of the failure chain lied about where the crash
was coming from. The full chain:

1. **First suspect: our sm121 CUTLASS SMEM-budget patch.** The initial symptom
   was a RuntimeError at `nvfp4_blockwise_moe.cuh:78` during the first forward
   pass of any NVFP4 MoE model. Our `sgl-kernel-sm121.patch` (StageCount<2> +
   Cooperative schedule) targets exactly that kernel, so it was the obvious
   suspect. We tried alternate tile shapes (<_128,_64,_128>, <_64,_128,_128>) —
   all of those failed CUTLASS template deduction at compile time
   ([`cute/atom/copy_traits_sm90_tma.hpp:744`](https://github.com/NVIDIA/cutlass/blob/main/include/cute/atom/copy_traits_sm90_tma.hpp),
   TMA SLayout static_assert). So we left the patch at its working config and
   looked elsewhere.

2. **Second suspect: upstream kernel before our CUTLASS GEMM.** We built
   `sgl-kernel-sm121-debug.patch` which inserts two diagnostic probes in
   `run_fp4_blockwise_scaled_group_mm_sm120()`:

   - An **entry probe** right after `const cudaStream_t stream = ...` and
     BEFORE `get_cached_workspace()`, calling `cudaStreamSynchronize` +
     `cudaGetLastError`. If the stream is already in error state at entry,
     some kernel launched BEFORE this function was called must have asserted.
   - A **post-launch probe** immediately after `gemm_op.run()` returns.

   Both probes are gated on `DGXARLEY_SM121_DEBUG` env var (runtime), and the
   whole patch is opt-in via `APPLY_SGL_KERNEL_SM121_DEBUG=1` build-arg. After
   one rebuild + redeploy we saw:

   ```
   [dgxarley sm121-debug] sm120 PRE-WORKSPACE stream error:
   sync=710 (device-side assert triggered) last=710 (device-side assert triggered)
   -- UPSTREAM kernel asserted, not our cutlass GEMM
   ```

   on both TP0/EP0 and TP2/EP2 — deterministic, all ranks, every forward pass.
   This definitively proved our CUTLASS GEMM is never even reached: some
   kernel launched BEFORE `run_fp4_blockwise_scaled_group_mm_sm120()` is the
   one asserting, and the `cuh:78` line is just the next `cudaMallocAsync`
   sync-point that happens to surface it.

3. **Third suspect: the real one — `_shuffle_rows_torch`'s `index_select`.**
   We added `CUDA_LAUNCH_BLOCKING=1` to the sglang pod env (via the
   ConfigMap in `roles/k8s_dgx/tasks/sglang.yml`), forcing every CUDA kernel
   launch to be synchronous so asserts surface at their actual launch site
   instead of at the next sync. That gave the real traceback:

   ```
   File ".../cutlass_moe.py", line 451, in cutlass_moe_fp4
       rep_a_fp4, rep_a_blockscale = scaled_fp4_experts_quant(...)
   File ".../jit_kernel/nvfp4.py", line 300, in scaled_fp4_experts_quant
       input_tensor = _shuffle_rows_torch(...)
   File ".../jit_kernel/nvfp4.py", line 257, in _shuffle_rows_torch
       output = input_tensor.index_select(0, dst2src_map.to(dtype=torch.int64))
   torch.AcceleratorError: CUDA error: device-side assert triggered
   ```

   Plus thousands of lines of:

   ```
   /build/pytorch/aten/src/ATen/native/cuda/IndexKernelUtils.cu:16:
   vectorized_gather_kernel: Assertion
   `ind >= 0 && ind < ind_dim_size` failed
   ```

   spread across many blocks and threads — **systemic**, not an edge case.

4. **Wrong root-cause hypothesis.** Initial theory: `params.num_experts`
   is passed as global (128) but the weights are local (32), so
   `prepare_moe_input` builds offsets over 128 buckets and something in
   the output references slots that don't exist in the local `a` tensor.
   This turned out to be half-right — the num_experts mismatch is a bug
   (fixed by PR #20869 hunks 1+2), but it is NOT the cause of the `a_map`
   OOB. The real cause is one layer deeper.

5. **Three parallel upstream-analysis agents** dissected the full code
   path in a single afternoon: (a) the native
   `sgl-kernel/csrc/moe/prepare_moe_input.cu` kernel source, (b) the
   Python call site in `cutlass_moe.py` + `modelopt_quant.py` including
   how `StandardDispatcher` remaps `topk_ids`, and (c) a GitHub search
   for existing upstream issues/PRs. Agent (a) revealed the
   `compute_arg_sorts` kernel has no `ep_rank` parameter and writes
   `a_map[slot] = i / topk` **only** where `topk_ids[i] == expert_id`.
   Agent (c) found PR #20869, whose description explicitly documents the
   same "topk_ids=-1 for non-local experts" failure mode as error #4 —
   and admits it is sidestepped by flashinfer_cutlass auto-routing
   rather than fixed in-place.

6. **Synthesis.** Combining all three: `a_map` is `torch.empty` (uninitialized),
   the kernel only writes slots for local experts, the -1 slots from
   the dispatcher remap leave matching positions in `a_map` as garbage,
   and `index_select` then trips on the garbage. Fix hypothesis:
   `torch.empty` → `torch.zeros`. Zero is always a valid row index;
   the fake-gathered rows for -1 slots should get grouped-gemm'd into
   garbage outputs which multiply by the (assumed) zeroed topk_weights
   for those slots and vanish in the final reduction.

7. **Crash gone, output garbage.** After deploying the `torch.zeros`
   monkey-patch via `sglang_launch.sh` the cuh:78 assert stopped
   firing, both debug probes went green (`sm120 pre-workspace clean`
   + `sm120 post-launch OK`), and sglang served its first
   `moe_runner_backend=triton` forward pass on SM121 with EP=4. But
   the generated tokens were meaningless — repeated `!` characters
   and Chinese filler like `豬`. The "non-local slots contribute zero"
   assumption was wrong: either the dispatcher does NOT zero
   topk_weights for -1 slots, or the fake output goes through the
   c2/c_map scatter and pollutes a real output row. The crash fix
   works at the surface level but is semantically incomplete.

8. **Vanilla upstream A/B test invalidates the CUTLASS patch.** To
   isolate whether our `sgl-kernel-sm121.patch` was ever necessary, we
   switched the pod image from our custom `xomoxcc/dgx-spark-sglang:
   0.5.10-sm121` to the upstream `scitrera/dgx-spark-sglang:0.5.10`.
   Same python monkey-patches (ConfigMap-mounted sglang_launch.sh),
   no sgl-kernel-level patches, no debug probes. Result: identical
   behavior — no cuh:78 crash, same garbage output. This definitively
   proves:

   - `StageCountAutoCarveout` on SM121 already picks a valid stage
     count for the Sm120 blockscaled CollectiveBuilder's default
     Pingpong schedule. The 99 KiB SMEM budget (per CUTLASS#3144) is
     respected automatically for the `<_128,_128,_128>` tile shape
     and `<1,1,1>` cluster shape in this specific kernel.
   - The earlier rejected "Cooperative alone with
     StageCountAutoCarveout" attempt (see patch header of
     `sgl-kernel-sm121.patch`, "Previous iterations that did NOT
     work") was actually fine at the CUTLASS level — we rejected
     it only because the cuh:78 crash persisted, not realizing the
     crash came from the downstream Python OOB that would have hit
     any stage-count configuration.
   - CUTLASS#3144 may still apply to other kernels, tile shapes, or
     build configurations — the SMEM budget fact is objective —
     but for THIS kernel it does not. The `sgl-kernel-sm121.patch`
     and the entire debug infrastructure around it stays in
     `scripts/patches/` as documented defense-in-depth, but is not
     required by the current deployment.

The net outcome after steps 1-8: we chased a phantom CUTLASS issue
for most of the session, and the actual bug was always purely in
the Python code ~4 layers up the call stack. Our surface fix
(`torch.zeros`) gets past the crash site but does not make the
output semantically correct — that requires a second layer of
investigation into how non-local slots propagate through the
grouped GEMM and scatter-back. The scitrera A/B test means we also
don't need a custom SGLang image for correctness; the build chain
in `scripts/` remains operational for future SM121 work but
the current production config can ship on the stock upstream
image.

## Why this was previously misattributed to `cutlass_fp4_group_mm`

Without `CUDA_LAUNCH_BLOCKING=1`, the async OOB `index_select` does not fault
synchronously. The CUDA context is poisoned, and the **next** kernel launch that
synchronizes — `cutlass_fp4_group_mm` inside `nvfp4_blockwise_moe.cuh` — is the
one that surfaces the error to the host, giving the misleading traceback:

```
File ".../sglang/jit_kernel/nvfp4.py", line 504, in _cutlass_fp4_group_mm_custom_op
    module.cutlass_fp4_group_mm(
RuntimeError: Runtime check failed at .../nvfp4_blockwise_moe.cuh:78:
    CUDA error: device-side assert triggered
```

The CUTLASS C++ kernel at `nvfp4_blockwise_moe.cuh:78` is a `TORCH_CHECK` that
happens to be the first synchronization point after the bad stream — it is
**not** the origin of the fault. Re-running with
`CUDA_LAUNCH_BLOCKING=1` (set in our Ansible playbook via commit `bdc069e`,
2026-04-11) pins the fault to the preceding `index_select` in
`_shuffle_rows_torch`.

## Traceback (with `CUDA_LAUNCH_BLOCKING=1`)

```
[2026-04-11 17:45:51 TP2 EP2] Scheduler hit an exception: Traceback (most recent call last):
  ...
  File ".../sglang/srt/layers/quantization/modelopt_quant.py", line 2027, in apply
    output = cutlass_moe_fp4(
  File ".../sglang/srt/layers/moe/cutlass_moe.py", line 451, in cutlass_moe_fp4
    rep_a_fp4, rep_a_blockscale = scaled_fp4_experts_quant(
  File ".../sglang/jit_kernel/nvfp4.py", line 300, in scaled_fp4_experts_quant
    input_tensor = _shuffle_rows_torch(
  File ".../sglang/jit_kernel/nvfp4.py", line 257, in _shuffle_rows_torch
    output = input_tensor.index_select(0, dst2src_map.to(dtype=torch.int64))
torch.AcceleratorError: CUDA error: device-side assert triggered
```

(Config: `nvidia/Qwen3-235B-A22B-NVFP4`, TP=4, EP=4, `moe_runner_backend=triton`
→ falls back to `cutlass_moe_fp4`, NCCL socket transport, 4 × GB10/SM121.)

## Relationship to the other two `modelopt_quant`/CUTLASS EP bugs

This bug is the **third** problem in the same EP/NVFP4 code path, downstream of
two Python-level bugs already documented in `SGLANG_TP_EP_MOE_UPSTREAM_BUG.md`:

1. `CutlassMoEParams(num_experts=layer.num_experts, ...)` — should be
   `num_local_experts`. Monkey-patched at container startup.
2. `ModelOptNvFp4FusedMoEMethod.process_weights_after_loading` else-branch
   doesn't EP-slice `w13_input_scale` / `w2_input_scale`. Monkey-patched at
   container startup. (See also upstream issue #21602 and PRs #20869 / #21630.)
3. **This bug.** After the two Python patches are applied, `scaled_fp4_experts_quant`
   still produces an OOB `dst2src_map` under EP because the expert-offset /
   row-mapping logic inside this helper was not updated for EP-local row
   tensors. Unlike (1) and (2), this one is in a JIT-kernel Python helper
   whose logic still assumes global-expert indexing.

## The Fix (surface level, incomplete)

**Status: this eliminates the crash but produces garbage output. Do not ship
as-is. See "Open: semantic fix" below.**



The root cause is uninitialized memory, so the fix is trivial once you know
where to look: replace `torch.empty` with `torch.zeros` for `a_map` and
`c_map` in `cutlass_moe_fp4` (lines 436-437 in v0.5.10).

```python
# sglang/srt/layers/moe/cutlass_moe.py, cutlass_moe_fp4 lines 436-437
# WAS:
a_map = torch.empty((topk_ids.numel()), dtype=torch.int32, device=device)
c_map = torch.empty((topk_ids.numel()), dtype=torch.int32, device=device)
# NOW:
a_map = torch.zeros((topk_ids.numel()), dtype=torch.int32, device=device)
c_map = torch.zeros((topk_ids.numel()), dtype=torch.int32, device=device)
```

Why this works:

- Zero is always a valid row index into `a` on every rank that receives
  tokens (i.e. `a.size(0) > 0`).
- `prepare_moe_input` still only writes the slots for local experts. The
  `-1` slots now read as 0 instead of garbage, so `a.index_select(0, a_map)`
  no longer OOBs.
- Non-local slots fake-gather row 0 of `a`, get grouped-gemm'd with fake
  FP4 quantization, and then combine with `topk_weights`. The dispatcher
  already zeros the topk_weights for -1 slots, so the fake outputs
  multiply by zero and vanish in the final reduction.
- Cost: a few CUDA multiply-by-zero cycles per non-local slot. No
  semantic change.
- No-op when `ep_size=1` (dispatcher doesn't write -1 sentinels in that
  case, so every slot is written by `compute_arg_sorts` regardless of
  `empty` vs `zeros`).

This is a strictly simpler and more direct fix than what upstream PR #20869
does — the PR touches `modelopt_quant.py` to pass `num_local_experts` and
slice input scales (both necessary, both already monkey-patched by us at
container start — see `roles/k8s_dgx/files/sglang_launch.sh`), but then
gives up on `cutlass_moe_fp4` and instead changes `server_args.py` to
auto-route SM120 to `flashinfer_cutlass`. The PR author acknowledges that
the `cutlass_moe_fp4` codepath remains broken; our one-line `torch.empty`
→ `torch.zeros` change is the first actual fix we are aware of.

## Our Workaround

Applied as a Python monkey-patch at container startup via
`roles/k8s_dgx/files/sglang_launch.sh` (same mechanism as the two earlier
`modelopt_quant.py` EP fixes already in that script):

```python
# Patch cutlass_moe.py: a_map/c_map zero-init for EP shuffle_rows OOB
old = '''    num_topk = topk_ids.shape[1]
    device = a.device
    a_map = torch.empty((topk_ids.numel()), dtype=torch.int32, device=device)
    c_map = torch.empty((topk_ids.numel()), dtype=torch.int32, device=device)'''
new = '''    num_topk = topk_ids.shape[1]
    device = a.device
    a_map = torch.zeros((topk_ids.numel()), dtype=torch.int32, device=device)
    c_map = torch.zeros((topk_ids.numel()), dtype=torch.int32, device=device)'''
```

The patch is discriminated from the unrelated FP8 `torch.empty` call at
`cutlass_moe.py:145-146` (inside `cutlass_fused_experts_fp8`, not affected
by this bug) by the surrounding `num_topk = topk_ids.shape[1]` context
line, which only exists in `cutlass_moe_fp4`.

With this patch applied, `moe_runner_backend: "triton"` and
`moe_runner_backend: "cutlass"` should both work under EP > 1 for
NVFP4 models. Until it is verified end-to-end on the GB10 cluster, the
previous workaround — `moe_runner_backend: "flashinfer_cutlass"` — remains
the safe choice and matches the 4-node Qwen3-235B NVFP4 test matrix
winner (`TESTLOGS/sglang_nn4_tp4_ep4/qwen-3-235b-a22b-nvfp4/`).

## Reproduction

1. Deploy an NVFP4 MoE model with EP > 1 on SGLang v0.5.10 (SM121/Blackwell):
   ```yaml
   moe_runner_backend: "triton"          # or "cutlass"
   fp4_gemm_backend: "flashinfer_cutlass"
   quantization: "modelopt_fp4"
   tp_size: 4
   ep_size: 4
   ```
2. Set `CUDA_LAUNCH_BLOCKING=1` in the worker environment to surface the real
   crash location (otherwise it appears at `nvfp4_blockwise_moe.cuh:78`).
3. Send any inference request. The scheduler crashes during the first
   `forward_extend` of the first layer that dispatches routed experts,
   with the traceback shown above.

Remove `CUDA_LAUNCH_BLOCKING=1` for normal operation once the torch.zeros
monkey-patch is in place and verified — it carries significant overhead from
serialized kernel launches.

## Verification (crash elimination only, NOT end-to-end correctness)

The first clean run of the triton MoE backend **without** the cuh:78 crash,
2026-04-11 18:24 UTC, after deploying the three sglang_launch.sh python
monkey-patches (modelopt_quant.py EP input-scale slicing + num_local_experts,
plus our new cutlass_moe.py a_map/c_map zero-init):

```
[2026-04-11 18:24:04] INFO:     10.68.2.1:0 - "GET /v1/models HTTP/1.1" 200 OK
[2026-04-11 18:24:19] INFO:     10.68.2.1:0 - "GET /v1/models HTTP/1.1" 200 OK
[dgxarley sm121-debug] sm120 pre-workspace clean (num_experts=32)
[dgxarley sm121-debug] sm120 post-launch OK (num_experts=32)
[dgxarley sm121-debug] sm120 pre-workspace clean (num_experts=32)
[dgxarley sm121-debug] sm120 post-launch OK (num_experts=32)
```

Three independent signals of success:

1. **`num_experts=32`** — not 128. Confirms the two existing
   `modelopt_quant.py` monkey-patches (CutlassMoEParams uses
   `num_local_experts`, input scales sliced per EP rank) are applied and
   effective. These came from PR #20869 hunks 1+2, already in
   `sglang_launch.sh` from a previous debug session.

2. **`sm120 pre-workspace clean`** — the entry-probe from
   `sgl-kernel-sm121-debug.patch` sees NO prior stream-error state when
   `run_fp4_blockwise_scaled_group_mm_sm120()` is called. Directly proves
   the `torch.empty` → `torch.zeros` fix in `cutlass_moe.py` worked:
   `_shuffle_rows_torch`'s `a.index_select(0, a_map)` no longer OOBs, so
   the stream reaches the CUTLASS GEMM entry in a clean state.

3. **`sm120 post-launch OK`** — the exit-probe after `gemm_op.run()`
   returns with `cudaSuccess`. First time our
   `sgl-kernel-sm121.patch` (StageCount<2> + `KernelPtrArrayTma
   WarpSpecializedCooperative`) has executed a real GEMM launch. The
   patch itself was designed in `CUTLASS_NVFP4_SM121_PRD.md` around
   the 99 KiB SM121 SMEM budget and the CUTLASS#3144 root-cause
   numbers, but until today could never be verified end-to-end
   because the upstream `shuffle_rows` OOB was crashing the pipeline
   two layers earlier.

Config that produced these lines:

- Model: `nvidia/Qwen3-235B-A22B-NVFP4` (128 routed experts, top-8)
- Topology: TP=4, EP=4 on 4 × DGX Spark GB10 (SM121), NCCL socket
- Image: `scitrera/dgx-spark-sglang:0.5.10-sm121` (our sm121 build with
  `sgl-kernel-sm121.patch` + `sgl-kernel-sm121-debug.patch` compiled in)
- Pod env: `DGXARLEY_SM121_DEBUG=1` (probe output), `CUDA_LAUNCH_BLOCKING=1`
  (kept on for the verification run so any regression would still surface
  at its actual launch site)
- Runtime monkey-patches from `sglang_launch.sh`: modelopt_quant.py
  EP-aware input-scale slicing, modelopt_quant.py CutlassMoEParams
  num_local_experts, cutlass_moe.py a_map/c_map zero-init
- Backend: `moe_runner_backend=triton` (the previously-broken path).
  The `flashinfer_cutlass` winner config was not needed for this run.

## Open: semantic fix (Variante B — in progress)

The `torch.zeros` patch gets past the crash but the output is garbage.
This means at least one of the assumptions behind the "fake-gather row 0
multiplies by zero weight and vanishes" story is wrong. Candidates to
investigate in order:

1. **`topk_weights` are NOT zeroed for -1 slots.** `StandardDispatcher`'s
   `local_expert_mapping` may only remap `topk_ids` (global → local or
   -1 sentinel) without touching the parallel `topk_weights` tensor.
   If that is the case, non-local slots carry their original (non-zero)
   softmax weights, and the fake row-0 gather contributes to the final
   reduction with real weight. Verification path: read
   `python/sglang/srt/layers/moe/token_dispatcher/standard.py` around
   the `local_expert_mapping` call and see if it also masks
   `topk_weights`.

2. **`c2` is `torch.empty`.** Independently of `a_map`, the grouped-GEMM
   OUTPUT tensor `c2` in `cutlass_moe_fp4` is also allocated with
   `torch.empty` (line ~471 in v0.5.10). The GEMM only writes the
   `expert_offsets[num_experts]`-sized active range; positions beyond
   that remain uninitialized. The downstream
   `apply_shuffle_mul_sum(c2, output, c_map, topk_weights)` then scatters
   based on `c_map`, and since our fix zero-initialized `c_map`, the
   non-written positions now all scatter into output row 0 — adding
   garbage `c2` values multiplied by `topk_weights[non_local]`.
   Candidate fix: also zero-init `c2` (and any other intermediate
   tensors that the grouped GEMM only partially writes — `intermediate`,
   `rep_a_fp4`, `rep_a_blockscale`).

3. **The scatter direction is wrong for our padded layout.**
   `apply_shuffle_mul_sum` may iterate `c2.size(0)` positions and
   unconditionally scatter, not knowing that only the
   `expert_offsets[-1]`-prefix is meaningful. Fix would be to pass an
   explicit `valid_length` parameter or truncate `c2` before the
   scatter.

4. **A completely separate EP bug.** Possible that
   `cutlass_moe_fp4`'s grouped GEMM has additional assumptions about
   `a` being fully-replicated-for-routing rather than locally-sharded,
   and our monkey-patches cover only two of several required changes.

Next debug step: add a small print-probe after `prepare_moe_input` to
inspect the actual `a_map`, `c_map`, `topk_weights` values on one rank
for one batch, then trace which positions get written by the GEMM and
which scatter path they take. If (2) turns out to be the issue, the
fix is another one-line monkey-patch in `sglang_launch.sh`. If (1),
the fix is a `topk_weights.masked_fill_(topk_ids == -1, 0)` injection
before the `cutlass_moe_fp4` call.

## Follow-ups (after the semantic fix lands)

- Remove `CUDA_LAUNCH_BLOCKING=1` from the sglang ConfigMap env vars
  once the semantic fix is validated — the serialized-kernel overhead
  was needed only to pin the assert site during debugging. (Already
  set back to `"0"` after the crash-fix verification.)
- Submit a PR to `sgl-project/sglang` with the combined fix
  (`torch.zeros` for `a_map`/`c_map`/`c2` + whatever masking is
  needed) on top of the (still unmerged) PR #20869 hunks. Rationale:
  keep `cutlass_moe_fp4` operational under EP instead of letting
  upstream permanently sidestep it by auto-routing to
  `flashinfer_cutlass`, which matters for NVFP4 MoE models where
  `triton`/`cutlass` may have a latency edge.
- Re-run the 4-node Qwen3-235B NVFP4 test matrix
  (`TESTLOGS/sglang_nn4_tp4_ep4/qwen-3-235b-a22b-nvfp4/`) once
  correct outputs are confirmed, to see whether the `triton` /
  `cutlass` backends outperform the `flashinfer_cutlass` winner
  (Test 17: 11.28 / 34.60 / 42.70 tok/s at n=1 / n=4 peak / n=8 peak).
- The `sgl-kernel-sm121.patch`, `sgl-kernel-sm121-debug.patch`, and
  all four `sgl-kernel-*.patch` build-time optimizations stay in
  `scripts/patches/` as retained infrastructure. Not required by the
  current deployment (vanilla upstream image handles SM121 correctly
  for this tile shape), but re-usable if CUTLASS#3144 resurfaces for
  other kernels or tile shapes, and the build+distribute chain in
  `scripts/` is generic infrastructure beyond this specific session.
