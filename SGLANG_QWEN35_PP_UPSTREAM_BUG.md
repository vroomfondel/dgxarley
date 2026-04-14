# SGLang Upstream Bug: Qwen3.5 Pipeline Parallelism (PP > 1) Weight Loading Crash

## Status

**Fixed upstream** as of 2026-04-06 (re-verified 2026-04-14: all three PRs still merged, our v0.5.10 image includes them). Three PRs fixed three cascading bugs:

| PR | What it fixed | Merged | In v0.5.10 |
|----|--------------|--------|-----------|
| [#19670](https://github.com/sgl-project/sglang/pull/19670) | Initial PP support for Qwen3.5 (broken: `make_layers()` missing `pp_rank`/`pp_size`) | 2026-03-07 | Yes |
| [#21070](https://github.com/sgl-project/sglang/pull/21070) | Fixed `make_layers()` + fused expert `params_dict` guard | 2026-03-21 | Yes |
| [#21448](https://github.com/sgl-project/sglang/pull/21448) | Fixed non-fused expert path + Mamba/SSM cache PP sharding + VLM `start_layer`/`end_layer` | 2026-03-30 | Yes |

Related issues: [#19500](https://github.com/sgl-project/sglang/issues/19500) (initial report), [#21184](https://github.com/sgl-project/sglang/issues/21184), [#21185](https://github.com/sgl-project/sglang/issues/21185). Superseded alternative: [PR #21217](https://github.com/sgl-project/sglang/pull/21217) (open, not needed).

**Our image `scitrera/dgx-spark-sglang:0.5.10` includes all three fixes.** PP support is available.

Files: `sglang/srt/models/qwen3_5.py`, `sglang/srt/models/qwen3_vl.py`, `sglang/srt/model_executor/model_runner_kv_cache_mixin.py`

## Affected Configuration

- Model: `nvidia/Qwen3.5-397B-A17B-NVFP4` (or any Qwen3.5 MoE variant)
- Pipeline Parallelism: `pp_size > 1`
- SGLang: v0.5.10rc0 or any version before 2026-03-30
- Architecture: hybrid GatedDeltaNet (linear attention) + GatedAttention (full attention), 60 layers, 512 experts

PP=1 (TP-only or TP+EP) is **not affected**.

## Observed Crash

Tested: `nvidia/Qwen3.5-397B-A17B-NVFP4`, PP=4, TP=1, EP=4, 4× DGX Spark (128 GB/GPU), `scitrera/dgx-spark-sglang:0.5.10rc0` (crash observed on this image; fixed in 0.5.10).

PP3 (pipeline stage 3, layers 45–59) attempts to load weights for layer 37 (which belongs to PP2, layers 30–44):

```
[2026-04-06 13:48:43 PP3] Parameter model.layers.37.input_layernorm.weight not found in params_dict
[2026-04-06 13:48:43 PP3] Scheduler hit an exception: Traceback (most recent call last):
  File ".../sglang/srt/managers/scheduler.py", line 3561, in run_scheduler_process
    ...
  File ".../sglang/srt/models/qwen3_5.py", line 1592, in load_weights
    param = params_dict[name_mapped]
            ~~~~~~~~~~~^^^^^^^^^^^^^
KeyError: 'model.layers.37.mlp.experts.w2_weight'
```

Expected PP=4 layer assignment (60 layers):
- PP0: layers 0–14
- PP1: layers 15–29
- PP2: layers 30–44 ← layer 37 belongs here
- PP3: layers 45–59 ← should NOT load layer 37

## The Three Bugs

### Bug 1: `make_layers()` without `pp_rank`/`pp_size` (PR #19670 → fixed by #21070)

The initial PP support (PR #19670) called `make_layers()` without pipeline arguments:

```python
# WRONG — all ranks instantiate all 60 layers as real objects
self.layers = make_layers(
    config.num_hidden_layers,
    get_layer,
    prefix=f"{prefix}.layers",
)
```

Without `pp_rank`/`pp_size`, every GPU built all 60 decoder layers instead of only its ~15. Non-local layers should become `PPMissingLayer` stubs. The effect: every GPU loaded the full model weights (~234 GB / 4 GPUs = should be ~59 GB, but was ~234 GB each → OOM on smaller GPUs, silently wrong on larger ones).

**Fix (PR #21070):** Pass `pp_rank` and `pp_size` to `make_layers()` so non-local layers become `PPMissingLayer` stubs.

### Bug 2: Fused expert weight loader — bare `params_dict[name]` (fixed by #21070)

Once Bug 1 was fixed, a new KeyError surfaced. The `load_fused_expert_weights()` inner function did:

```python
def load_fused_expert_weights(name, params_dict, loaded_weight, shard_id, num_experts):
    param = params_dict[name]  # KeyError — PPMissingLayer stubs have no weights in params_dict
```

Because layers outside the PP rank's range are now `PPMissingLayer` stubs, their expert parameters don't exist in `params_dict`. The layer_id range check at the top of the `load_weights` loop should skip these, but the fused expert weight name mapping (`name.replace(weight_name, param_name)`) produced names like `model.layers.37.mlp.experts.w2_weight` that passed the range check (which looked at the original tensor name, not the mapped name).

**Fix (PR #21070):** Added guard in `load_fused_expert_weights`:

```python
def load_fused_expert_weights(name, params_dict, loaded_weight, shard_id, num_experts):
    if name not in params_dict:  # GUARD ADDED
        return False
    param = params_dict[name]
```

### Bug 3: Non-fused expert path — same bare access (fixed by #21448)

PR #21070 fixed the fused path but the non-fused `else` branch was missed:

```python
else:
    if name_mapped.endswith(ignore_suffixes) and name_mapped not in params_dict:
        continue
    param = params_dict[name_mapped]  # KeyError if layer is on another rank
```

**Fix (PR #21448):** Added early layer_id skip at the top of all four `load_weights` methods:

```python
layer_id = get_layer_id(name)
if (
    layer_id is not None
    and hasattr(self, "start_layer")
    and (layer_id < self.start_layer or layer_id >= self.end_layer)
):
    continue
```

PR #21448 also fixed two additional issues:
- **Mamba/SSM cache not PP-sharded:** GatedDeltaNet layers maintain recurrent state (`HybridReqToTokenPool`, `HybridLinearKVPool`). Pre-fix, these caches allocated memory for all 60 layer IDs regardless of PP rank — wasting ~3/4 of cache memory.
- **VLM `start_layer`/`end_layer` missing:** `Qwen3VLForConditionalGeneration` (base class for `Qwen3_5MoeForConditionalGeneration`) didn't expose `start_layer`/`end_layer`, so the early-exit guard (`hasattr(self, "start_layer")`) silently bypassed the check.

## Hybrid Architecture and PP

Qwen3.5 uses a repeating pattern: 3× GatedDeltaNet (linear attention/SSM) + 1× GatedAttention (full GQA), 15 repetitions = 60 layers. PP considerations:

1. **Two-type layer dispatch:** `get_layer()` uses `config.layers_block_type[idx]` to pick between `Qwen3_5AttentionDecoderLayer` and `Qwen3_5LinearDecoderLayer`. `make_layers()` with `pp_rank`/`pp_size` handles this correctly — the block type lookup only fires for layers within the rank's range.

2. **SSM state cache sharding (Bug 3b):** The GatedDeltaNet layers have recurrent state that must be partitioned per PP rank. Fixed in PR #21448 by filtering `mamba2_cache_params.layers` to `[start_layer, end_layer)`.

3. **KV heads vs. PP:** The model has `num_key_value_heads=2` for GQA layers. With PP, each rank handles all KV heads for its layer subset — no KV head splitting across ranks. This is not a problem (unlike TP where KV heads must divide evenly).

## Our Situation

- **Current image:** `scitrera/dgx-spark-sglang:0.5.10` — all three PP fixes included. PP is available.
- **No monkey-patch was needed:** Updated to v0.5.10 final which includes all fixes.
- **Previous workaround (no longer needed):** TP=4 EP=4 PP=1 (no pipeline parallelism).

## Why PP=4 Was Attempted

With Qwen3.5-397B-A17B-NVFP4 (~234 GB), PP=4 would give each GPU ~59 GB weights (same as TP=4), but with different tradeoffs:
- PP=4: each GPU holds complete tensors for 15 layers → no all-reduce between GPUs during forward pass within a stage, only point-to-point between stages
- TP=4: each GPU holds 1/4 slices of all 60 layers → all-reduce after every attention/MoE layer

PP could potentially reduce inter-node communication overhead on the QSFP/Socket transport, at the cost of pipeline bubbles. Testing this requires v0.5.10 final.
