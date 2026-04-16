# SGLang/vLLM Upstream Bugs: MoE + Expert Parallelism (moe_wna16 + modelopt_quant)

## Status

**Open upstream (vLLM only)** as of 2026-04-16 (re-verified: no review movement on any open PR since last check). Bug exists in both SGLang and vLLM (code originated in vLLM PR #14447). Present in SGLang v0.5.10 (2026-04-06) and v0.5.10.post1 (2026-04-09, flashinfer bump only). No new release since v0.5.10.post1. EPLB Qwen3 fix (PR #21822) merged 2026-04-09 but not in any release yet (post1 was tagged ~4h before merge); all other referenced PRs re-verified 2026-04-16 — none merged, none reviewed (vllm#35598 last touched 2026-04-13 by author rebase, vllm#36026 stale since 2026-03-29, sglang#20869 stale since 2026-03-18, sglang#21630 stale since 2026-03-29, sglang#21612 stale since 2026-03-29, sglang#20963 last collaborator activity 2026-04-06).

- vLLM: [PR #35598](https://github.com/vllm-project/vllm/pull/35598) — open since 2026-02-28, not merged. Author rebased onto `main` on 2026-04-13 (commit `c56eae0e`, merge-from-main only, no code changes); prior rebase 2026-03-05. Still only the initial Gemini bot review from 2026-02-28 — no human reviewer has engaged
- vLLM: [PR #36026](https://github.com/vllm-project/vllm/pull/36026) — fix wrong num_experts in moe_wna16 kernel dispatch, open since 2026-03-29, author pinged for review 2026-03-29, still unreviewed
- SGLang: no upstream issue or PR filed

Files:
- SGLang: `sglang/srt/layers/quantization/moe_wna16.py`, lines 491–504 (v0.5.9)
- vLLM: `vllm/model_executor/layers/quantization/moe_wna16.py`, lines 492–505

## Affected Configuration

- Quantization: `moe_wna16` (AWQ/GPTQ 4-bit MoE models with `zero_point: true`)
- Expert Parallelism: `ep_size > 1`
- Tensor Parallelism: `tp_size > 1`
- Tested with: Qwen3-235B-A22B MoE (128 experts), TP=2, EP=2, AWQ 4-bit

Models with `zero_point: false` (symmetric quantization) are **not affected** — qzeros loading is skipped entirely via early return.

## The Bug

The `moe_wna16_weight_loader` closure in `MoeWNA16Method.get_weight_loader()` has three code paths:

1. `if "w13_qzeros" in weight_name` — custom inline logic for gate/up qzeros
2. `elif "w2_qzeros" in weight_name` — custom inline logic for down qzeros
3. `else` — delegates to `FusedMoE.weight_loader` (handles qweight, scales)

The `else` path correctly calls `layer._map_global_expert_id_to_local_expert_id()` for EP remapping and uses the MoE-local TP rank for slicing. The qzeros paths bypass this and have **two bugs**:

### Bug 1: Global expert_id instead of local EP index

```python
# BUGGY: expert_id is global (0-127), param.data has shape [64, ...] with EP=2
param.data[expert_id, : shard_size // 2] = tensor   # IndexError when expert_id >= 64
```

The weight loader is called for all 128 experts on every rank. `FusedMoE.weight_loader` maps global → local and skips non-local experts. The qzeros branches index directly with the global id.

### Bug 2: Global tp_rank instead of MoE-local tp_rank

```python
# BUGGY: tp_rank is global (0 or 1), but moe_tp_size = tp_size/ep_size = 1 with EP=2
tensor = loaded_weight.view(layer.moe_tp_size, -1, loaded_weight.size(1))[tp_rank]
# view creates dimension of size 1, tp_rank=1 → IndexError
```

With EP, MoE layers don't use TP splitting (`moe_tp_size = 1`). The correct index is `layer.moe_tp_rank` (or equivalently `tp_rank % layer.moe_tp_size`), which is 0 on both ranks.

## Root Cause

The qzeros branches were written as special cases that bypass the generic `FusedMoE.weight_loader`. They handle the tensor reshaping differently from qweight/scales (different view dimensions), which is why they can't simply delegate. But they failed to replicate the EP-aware expert remapping and MoE-local TP rank logic that `FusedMoE.weight_loader` provides.

## Fix

For each qzeros branch, add EP remapping and use MoE-local TP rank:

```python
if "w13_qzeros" in weight_name:
    _local_id = layer._map_global_expert_id_to_local_expert_id(expert_id)
    if _local_id == -1:
        return
    _moe_tp_rank = tp_rank % layer.moe_tp_size  # or: layer.moe_tp_rank
    tensor = loaded_weight.view(
        layer.moe_tp_size, -1, loaded_weight.size(1)
    )[_moe_tp_rank]
    if shard_id == "w1":
        param.data[_local_id, : shard_size // 2] = tensor
    else:
        param.data[_local_id, shard_size // 2 :] = tensor
elif "w2_qzeros" in weight_name:
    _local_id = layer._map_global_expert_id_to_local_expert_id(expert_id)
    if _local_id == -1:
        return
    _moe_tp_rank = tp_rank % layer.moe_tp_size
    param.data[_local_id] = loaded_weight.view(
        loaded_weight.size(0), layer.moe_tp_size, -1
    )[:, _moe_tp_rank]
```

The fix is a no-op when `ep_size=1`: `_local_id == expert_id` (identity mapping) and `_moe_tp_rank == tp_rank` (modulo has no effect).

## Our Workaround

We monkey-patch `moe_wna16.py` at container startup in `sglang_launch.sh` and `sglang_shard_launch.sh` (Python string-replace before SGLang starts). Same pattern as the existing ShardedStateLoader progress-logging patch.

## Caveat: Is moe_wna16 + EP even the right combination?

The reason `moe_wna16` was originally chosen is to avoid the **Marlin repack memory peak**.
Without `--quantization moe_wna16`, SGLang auto-detects `AWQMarlinConfig`, which repacks
AWQ weights into Marlin format at load time. During repack, old and new tensors coexist in
GPU memory — for Qwen3-235B-A22B AWQ with TP=2, this peaks at ~109 GB per GPU (vs. 128 GB
available on DGX Spark).

However, this calculation assumes **TP=2 without EP**. With EP=2, the memory situation changes
fundamentally:

- **TP=2 only**: each GPU holds all 128 experts (TP-split) → ~62 GB weights, ~109 GB repack peak
- **TP=2 + EP=2**: each GPU holds only 64 experts (full weight per expert) → ~31–35 GB weights, ~55–60 GB repack peak → **fits comfortably in 128 GB**

This means the original motivation for `moe_wna16` (avoiding repack OOM) **disappears with EP**.
The standard `AWQMarlinConfig` code path goes through `FusedMoE.weight_loader`, which is
fully EP-aware and well-tested. The `moe_wna16` code path, by contrast, has the qzeros bug
documented above — precisely because it bypasses `FusedMoE.weight_loader` for a niche case
that few people apparently test with EP.

**Practical takeaway**: When using EP, consider dropping `quantization: "moe_wna16"` and
letting auto-detection use `AWQMarlinConfig` instead. This avoids the bug entirely and uses
the mainstream, well-tested code path. The monkey-patch documented here remains valid for
anyone who does need `moe_wna16 + EP` (e.g., on GPUs with less memory headroom), but it may
be solving a problem that doesn't need to exist.

## Additional Bug: EPLB crashes with Qwen3MoE and Qwen3.5MoE

**Upstream status** as of 2026-04-06:
- Qwen3.5: fixed via [PR #19767](https://github.com/sgl-project/sglang/pull/19767) (merged 2026-03-09, included in v0.5.10)
- Qwen3: [PR #21461](https://github.com/sgl-project/sglang/pull/21461) — closed without merge 2026-03-30 (CI failure), superseded by #21822
- Qwen3: [PR #21822](https://github.com/sgl-project/sglang/pull/21822) — **merged 2026-04-09 at 07:13 UTC**. Addresses `AttributeError: 'LazyValue' object has no attribute 'keys'` in `eplb_manager.py` for Qwen3 MoE. (Duplicate [PR #21820](https://github.com/sgl-project/sglang/pull/21820) was closed same day in favour of #21822.) **Not in v0.5.10.post1**: that tag was cut on 2026-04-09 at 03:21 UTC — ~4h *before* #21822 was merged — so the EPLB fix misses post1 by a few hours and will only land in the next release after post1

When `--enable-eplb` is active with EP, the `EPLBManager` crashes after its first rebalance
interval (default: 1000 forward passes):

```
File ".../sglang/srt/eplb/eplb_manager.py", line 110, in _compute_update_layer_ids_chunks
    list(self._model_runner.model.routed_experts_weights_of_layer.keys())
AttributeError: 'Qwen3MoeForCausalLM' object has no attribute 'routed_experts_weights_of_layer'
```

The EPLB rebalancer needs models to expose a `routed_experts_weights_of_layer` property
(a dict mapping layer IDs to their expert weight tensors) so it can transfer weights between
GPUs. Neither `Qwen3MoeForCausalLM` nor `Qwen3_5MoeForConditionalGeneration` implements this —
likely only `DeepseekV3ForCausalLM` (or similar) was tested with EPLB.

**Impact**: The crash kills the scheduler, which triggers SIGQUIT → full restart of both nodes.
This happens reliably after ~1000 inference passes (~8 min wall time under moderate load).

**Confirmed failing on both architectures:**

| Model class | Image | Date | Pod |
|---|---|---|---|
| `Qwen3MoeForCausalLM` (Qwen3-235B-A22B) | 0.5.9-t5 | 2026-03-19 | sglang-head-855c5799c4 |
| `Qwen3_5MoeForConditionalGeneration` (Qwen3.5-122B-A10B-FP8) | 0.5.9-dev2-acab24a7-t5 | 2026-03-20 | sglang-head-5d7585955 |

The Qwen3.5 crash on dev2-acab24a7-t5 proves that [PR #19767](https://github.com/sgl-project/sglang/pull/19767)
("Fix qwen3.5 mtp eplb related issues", merged 2026-03-09) is either not included in
commit `acab24a7` (2026-03-11), or does not actually fix the `routed_experts_weights_of_layer`
attribute for `Qwen3_5MoeForConditionalGeneration` despite the claim. The exact same
`AttributeError` on the same code path (`eplb_manager.py:110`) occurs.

**Workaround**: Disable EPLB (`--enable-eplb` removed). EP=2 still works with static expert
assignment (experts 0–63 → GPU 0, experts 64–127 → GPU 1). The static assignment is suboptimal
if expert activation is highly skewed, but in practice Qwen3-235B shows ~0.82 balancedness
which is acceptable.

## Additional Bug: modelopt_quant.py NVFP4 input_scale not EP-aware

### Status

**Reported** as of 2026-03-28: [sgl-project/sglang#21602](https://github.com/sgl-project/sglang/issues/21602). Bug exists in SGLang `sglang/srt/layers/quantization/modelopt_quant.py`, class `ModelOptNvFp4FusedMoEMethod`.

Two competing fix PRs have been filed (neither merged as of 2026-04-09):
- [PR #20869](https://github.com/sgl-project/sglang/pull/20869) (2026-03-18) — broader fix: EP-slices input_scale, passes `num_local_experts` to `CutlassMoEParams`, extends SM120 support. No human review, stale since 2026-03-18. Likely to be superseded by the #20963 modelopt refactoring (see below)
- [PR #21630](https://github.com/sgl-project/sglang/pull/21630) (2026-03-29) — narrower fix: only the `else` branch (non-FlashInfer backends). Code updated 2026-03-29, no review yet

Maintainer feedback on [#21602](https://github.com/sgl-project/sglang/issues/21602) (2026-03-30): maintainer `wenscarl` confirmed the bug is real but noted that `w13_input_scale` shape is model-dependent — some models (e.g. DSR1 NVFP4) require the full `num_experts` dimension. The fix needs to be model-aware rather than a blanket slice to `num_local_experts`. The broader modelopt refactoring ([PR #20963](https://github.com/sgl-project/sglang/pull/20963)) is likely the vehicle for this fix

### Affected Configuration

- Quantization: `modelopt_fp4` (NVFP4-quantized MoE models)
- Expert Parallelism: `ep_size > 1`
- Backend: the `else` fallback branch in `process_weights_after_loading` (i.e., when neither `enable_flashinfer_cutlass_moe`, `enable_flashinfer_trtllm_moe`, nor `enable_flashinfer_cutedsl_moe` is active — this is the path hit by the **shard job** which doesn't configure a MoE runner backend)
- Tested with: nvidia/MiniMax-M2.5-NVFP4 (256 experts), TP=2, EP=2, shard job on 0.5.9-dev2-acab24a7-t5

The `flashinfer_cutlass` and `trtllm` branches are **not affected** (they reduce input_scale to a scalar via `.max()`). The `cutedsl` branch is **not affected** (it has a `_slice_scale()` helper that correctly slices to local experts).

### The Bug

In `process_weights_after_loading()`, the `else` branch computes:

```python
w13_input_scale = layer.w13_input_scale.max(dim=-1).values.to(torch.float32)  # shape: (num_experts,)
w2_input_scale = layer.w2_input_scale                                          # shape: (num_experts,)
```

These are then multiplied with EP-local weight scales:

```python
(w13_input_scale * w13_weight_scale_2).to(torch.float32)  # (256,) * (128,) → RuntimeError
(w2_input_scale * layer.w2_weight_scale_2).to(torch.float32)  # same
```

`w13_weight_scale_2` has shape `(num_local_experts,)` = 128 with EP=2, but `w13_input_scale` remains at `(num_experts,)` = 256.

### Root Cause

`w13_input_scale` and `w2_input_scale` are allocated as global tensors (flagged with `_sglang_require_global_experts = True`) because the weight loader fills them for all experts. The `cutedsl` branch correctly slices them to local experts via `_slice_scale()`, but this helper is defined inside the `elif` block and is not available to the `else` branch. The `else` branch was never tested with EP > 1.

### Crash Output

```
[2026-03-27 18:41:27 TP0 EP0] Scheduler hit an exception:
  File ".../modelopt_quant.py", line 1560, in process_weights_after_loading
    (w13_input_scale * w13_weight_scale_2).to(torch.float32),
RuntimeError: The size of tensor a (256) must match the size of tensor b (128) at non-singleton dimension 0
```

### Fix

Add EP-aware slicing in the `else` branch, same logic as `_slice_scale()`:

```python
        else:
            w13_input_scale = layer.w13_input_scale.max(dim=-1).values.to(torch.float32)
            w2_input_scale = layer.w2_input_scale
            # EP-aware slicing: no-op when ep_size=1
            if layer.moe_ep_size > 1:
                _ep_start = layer.moe_ep_rank * layer.num_local_experts
                _ep_end = _ep_start + layer.num_local_experts
                w13_input_scale = w13_input_scale[_ep_start:_ep_end]
                w2_input_scale = w2_input_scale[_ep_start:_ep_end]
```

### Our Workaround

Monkey-patched in `sglang_launch.sh` and `sglang_shard_launch.sh` (same string-replace pattern as the `moe_wna16` patch). The patch inserts the EP slicing block after the two assignments in the `else` branch.

## Additional Bug: ModelOptModelLoader doesn't support sharded_state load format

### Status

**Reported** as of 2026-03-28: [sgl-project/sglang#21603](https://github.com/sgl-project/sglang/issues/21603). Bug exists in SGLang `sglang/srt/model_loader/loader.py`, class `ModelOptModelLoader`.

- Fix PR: [#21612](https://github.com/sgl-project/sglang/pull/21612) — "fix: fix sharded state for ModelOptModelLoader", opened 2026-03-28 with unit tests. Delegates to `ShardedStateLoader` when `load_format=sharded_state` and model is already quantized. Stalled as of 2026-04-09 (only Gemini auto-review, no human review)

### Affected Configuration

- Quantization: `modelopt_fp4` (NVFP4-quantized models)
- Load format: `sharded_state` (pre-sharded checkpoints from `save_sharded_model`)
- Any model loaded through `ModelOptModelLoader` with `--load-format sharded_state`
- Tested with: nvidia/MiniMax-M2.5-NVFP4, pre-sharded TP=2 EP=2, on 0.5.9-dev2-acab24a7-t5

Non-NVFP4 models (FP8, AWQ, etc.) are **not affected** — they use `DefaultModelLoader` or `ShardedStateLoader` directly.

### The Bug

`ModelOptModelLoader` inherits from `DefaultModelLoader`. For pre-quantized models (the common case for NVFP4), `load_model()` calls `super().load_model()` → `DefaultModelLoader._prepare_weights()`. This method handles `LoadFormat.AUTO`, `SAFETENSORS`, `PT`, `MISTRAL`, `NPCACHE`, and `DUMMY` — but **not `SHARDED_STATE`**, raising:

```
ValueError: Unknown load_format: LoadFormat.SHARDED_STATE
```

Meanwhile, `ShardedStateLoader` (which handles `SHARDED_STATE`) is a separate class inheriting from `BaseModelLoader`, not `DefaultModelLoader`. The two loaders are not composed.

### Root Cause

`ModelOptModelLoader` inherits `DefaultModelLoader` whose `_prepare_weights()` doesn't handle `LoadFormat.SHARDED_STATE` → raises `ValueError: Unknown load_format`. `ShardedStateLoader` (which handles `SHARDED_STATE`) is a separate class inheriting `BaseModelLoader` — the two are not composed.

**Note**: After fixing the loader dispatch, we initially hit a secondary `IndexError` (tensor shape mismatch during shard loading). This was **not an upstream bug** but a config issue on our side: the shard job didn't pass `moe_runner_backend`, causing `process_weights_after_loading` to take a different branch (per-expert 1D scales) than serving (scalar scales via `flashinfer_cutlass`). Fixed by passing `SGLANG_MOE_RUNNER_BACKEND` to the shard job so both use the same code path.

### Fix

In `ModelOptModelLoader.load_model()`, check for `SHARDED_STATE` and delegate to `ShardedStateLoader`:

```python
if model_config._is_already_quantized():
    if self.load_config.load_format == LoadFormat.SHARDED_STATE:
        _sharded_loader = ShardedStateLoader(self.load_config)
        return _sharded_loader.load_model(
            model_config=model_config, device_config=device_config
        )
    return super().load_model(...)
```

### Our Workaround

- **Loader dispatch**: monkey-patched in `sglang_launch.sh` and `sglang_shard_launch.sh` (delegates to `ShardedStateLoader` when `load_format == SHARDED_STATE`)
- **Shard job config**: `SGLANG_MOE_RUNNER_BACKEND` env var added to shard job containers in `sglang_shard.yml`, passed through to `Engine()` in `sglang_save_sharded.py` — ensures consistent `process_weights_after_loading` branches between shard and serving

## Additional Bug: CutlassMoEParams uses global num_experts with EP

### Status

**Unreported** as of 2026-04-09. Bug exists in SGLang v0.5.10rc0, v0.5.10, and v0.5.10.post1
`sglang/srt/layers/quantization/modelopt_quant.py`, method
`_maybe_init_cutlass_moe_params()`, and
`sglang/srt/layers/moe/cutlass_moe.py`, function `cutlass_moe_fp4()`.

### Affected Configuration

- Quantization: `modelopt_fp4` (NVFP4-quantized MoE models)
- Expert Parallelism: `ep_size > 1`
- MoE runner backend: `triton` (which falls back to `cutlass_moe_fp4` for FP4)
- Tested with: nvidia/MiniMax-M2.5-NVFP4 (256 experts), TP=2, EP=2, v0.5.10rc0

The `flashinfer_cutlass` backend takes a different code path and is **not affected**
by this specific assertion — but has its own issues with CUDA graph capture JIT OOM.

### The Bug

`_maybe_init_cutlass_moe_params()` in `modelopt_quant.py` creates `CutlassMoEParams`
with `num_experts=layer.num_experts` (global, 256). With EP=2, each rank only holds
128 experts — the weight tensors `w1_fp4` and `w2_fp4` have shape `[128, ...]`.

In `cutlass_moe_fp4()`, the assertion on line 419 checks:

```python
e_w1, nx2_w1, half_k_w1 = w1_fp4.shape          # e_w1 = 128 (local)
assert e_w1 == e_w2 and e_w1 == params.num_experts  # params.num_experts = 256 (global)
# → AssertionError: ('Number of experts must match', ' between weights.')
```

Additionally, `prepare_moe_input()` on line 98 receives `params.num_experts` (256),
but `topk_ids` only reference local expert indices 0–127. All internal tensors
(strides, expert_offsets, problem_sizes) are sized for 256 experts instead of 128.

### Traceback

```
File ".../modelopt_quant.py", line 2010, in apply
    output = cutlass_moe_fp4(
File ".../cutlass_moe.py", line 419, in cutlass_moe_fp4
    assert e_w1 == e_w2 and e_w1 == params.num_experts, (
AssertionError: ('Number of experts must match', ' between weights.')
```

### Fix

Use `layer.num_local_experts` instead of `layer.num_experts` when creating
`CutlassMoEParams`. This is a no-op when `ep_size=1` (num_local_experts == num_experts).

```python
# In _maybe_init_cutlass_moe_params():
layer.cutlass_moe_params = CutlassMoEParams(
    CutlassMoEType.BlockscaledFP4,
    device,
    num_experts=layer.num_local_experts,  # was: layer.num_experts
    intermediate_size_per_partition=inter_size,
    hidden_size=hidden_size,
)
```

Note: PR #20869 (open, 2026-03-18) already includes this fix as part of a broader
EP-awareness patch for `modelopt_quant.py`, but has not been merged.

### Deeper Issue: cutlass_fp4_group_mm CUDA kernel assert with EP

> **Update 2026-04-11:** Re-running with `CUDA_LAUNCH_BLOCKING=1` (commit `bdc069e`)
> showed that `nvfp4_blockwise_moe.cuh:78` is **not** the origin of the fault — it is
> the first synchronizing kernel after an out-of-bounds `index_select` inside
> `scaled_fp4_experts_quant` → `_shuffle_rows_torch`. The real root cause and
> reproduction details are documented separately in
> [`SGLANG_NVFP4_SHUFFLE_ROWS_OOB_UPSTREAM_BUG.md`](SGLANG_NVFP4_SHUFFLE_ROWS_OOB_UPSTREAM_BUG.md).
> The section below is kept for historical context — the symptom is real, the
> attribution to the CUTLASS C++ kernel is not.

Even after fixing `CutlassMoEParams` to use `num_local_experts`, the underlying
CUTLASS FP4 MoE GEMM kernel (`nvfp4_blockwise_moe.cuh:78`) triggers a device-side
assert when called with EP-sliced expert tensors. This is a compiled C++/CUDA kernel
— not patchable from Python.

```
File ".../sglang/jit_kernel/nvfp4.py", line 504, in _cutlass_fp4_group_mm_custom_op
    module.cutlass_fp4_group_mm(
RuntimeError: Runtime check failed at .../sglang/jit_kernel/csrc/moe/nvfp4_blockwise_moe.cuh:78:
    CUDA error: device-side assert triggered
```

Tested: nvidia/MiniMax-M2.5-NVFP4 (256 experts), TP=2, EP=2, v0.5.10rc0,
`moe_runner_backend=triton` (which falls back to `cutlass_moe_fp4` for NVFP4).
The Python-level `CutlassMoEParams` patch (num_local_experts) resolves the
assertion in `cutlass_moe_fp4()`, but the CUTLASS kernel itself does not support
EP-partitioned expert tensors.

**Impact**: `moe_runner_backend: "triton"` is **unusable** with NVFP4 + EP > 1.
The only working MoE backend for NVFP4 + EP is `flashinfer_cutlass`, which uses
a different code path that does not go through `cutlass_moe_fp4`.

### Our Workaround

The `CutlassMoEParams` Python-level fix is monkey-patched in `sglang_launch.sh`
and `sglang_shard_launch.sh`: `sed` replaces `layer.num_experts` with
`layer.num_local_experts` in both the `CutlassMoEParams` constructor call and
the cache-invalidation check. The patch is guarded by a grep for the upstream
comment `# global num experts` — if upstream fixes this, the patch auto-skips.

However, the CUDA kernel-level issue cannot be patched. For NVFP4 + EP > 1, use
`moe_runner_backend: "flashinfer_cutlass"` (not `"triton"`).

## Related Upstream Issues & PRs

### Directly addressing our bugs
- vLLM [PR #35598](https://github.com/vllm-project/vllm/pull/35598) — fix moe_wna16 qzeros EP (open, author rebased onto main 2026-04-13, only bot review)
- SGLang [PR #21461](https://github.com/sgl-project/sglang/pull/21461) — fix EPLB Qwen3 missing `routed_experts_weights_of_layer` (closed without merge 2026-03-30, CI failure)
- SGLang [PR #19767](https://github.com/sgl-project/sglang/pull/19767) — fix EPLB Qwen3.5 (merged 2026-03-09)
- SGLang [#21602](https://github.com/sgl-project/sglang/issues/21602) — our report: NVFP4 input_scale not EP-aware
  - Fix PR: [#20869](https://github.com/sgl-project/sglang/pull/20869) — broader fix incl. CutlassMoEParams + SM120 (open, stale since 2026-03-18, no human review)
  - Fix PR: [#21630](https://github.com/sgl-project/sglang/pull/21630) — narrower fix, else-branch only (open, 2026-03-29)
- SGLang [#21603](https://github.com/sgl-project/sglang/issues/21603) — our report: ModelOptModelLoader doesn't support sharded_state
  - Fix PR: [#21612](https://github.com/sgl-project/sglang/pull/21612) — fix sharded state for ModelOptModelLoader (open, awaiting review)

### Related but not fixing our bugs
- vLLM #12647 — moe_wna16 AssertionError (KV cache conflict, unrelated)
- vLLM #22961 — TypeError in moe_wna16_weight_loader (return_success param, unrelated)
- vLLM PR #14447 — introduced moe_wna16 marlin kernel (origin of this code)
- vLLM [PR #36026](https://github.com/vllm-project/vllm/pull/36026) — fix wrong num_experts in moe_wna16 kernel dispatch (open, different sub-bug)
- SGLang PR #17137 — non-Marlin WNA16MoE port (does not fix EP bug)
- SGLang #14158 — update_weights_from_tensor for WNA16MoE (unrelated)
- SGLang [PR #13715](https://github.com/sgl-project/sglang/pull/13715) — fix EPLB + FP4 weight tensor filtering (merged, different issue)
- SGLang [PR #20963](https://github.com/sgl-project/sglang/pull/20963) — Nvidia modelopt refactoring (1/N). Under active review: reviewer `Edwardf0t1` asked for end-to-end verification 2026-03-31, author `wenscarl` responded 2026-04-01 and posted 3 further inline review responses 2026-04-06. Not stalled but awaiting approval. Migrates the NVFP4 code as-is — expected vehicle for EP-awareness fixes (#20869, #21630). Watch this PR for resolution of the NVFP4 input_scale and CutlassMoEParams bugs
- SGLang [PR #21822](https://github.com/sgl-project/sglang/pull/21822) — EPLB/Qwen3 fix. **Merged 2026-04-09**. Addresses `LazyValue.keys()` AttributeError. Will be in next release after v0.5.10.post1
