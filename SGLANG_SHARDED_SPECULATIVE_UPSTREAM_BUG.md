# SGLang Upstream Bug: sharded_state + Speculative Decoding (NEXTN/EAGLE)

## Status

**Unreported** as of 2026-04-29 (re-verified: GitHub search for "sharded_state speculative" / "speculative_draft_load_format sharded" in `sgl-project/sglang` returns no issues or PRs). Bug exists in SGLang v0.5.9, v0.5.10rc0, v0.5.10, and v0.5.10.post1 (latest release; no new SGLang release since 2026-04-09).

- File: `sglang/srt/managers/scheduler.py`, method `maybe_init_draft_worker()`
- Root cause in: `sglang/srt/managers/tp_worker.py`, method `_init_model_config()`

## Affected Configuration

- Load format: `sharded_state` (pre-sharded TP checkpoints)
- Speculative decoding: any algorithm (`NEXTN`, `EAGLE`, `EAGLE3`)
- Tested with: Qwen3.5-122B-A10B-FP8, TP=2, EP=2, NEXTN, SGLang 0.5.9

Models loaded with `--load-format auto` (default) are **not affected**.

## The Bug

When `load_format=sharded_state` is combined with speculative decoding, the draft
model's `ModelRunner` inherits the same `load_format` and `model_path` as the main
model. The `ShardedStateLoader` then attempts to load draft model weights from the
per-rank shard files, which only contain main model weight keys. Draft/MTP model
parameters that don't exist in the shard state dict cause a `KeyError` crash:

```
Scheduler hit an exception: Traceback (most recent call last):
  File ".../scheduler.py", line 3130, in run_scheduler_process
    scheduler = Scheduler(...)
  File ".../scheduler.py", line 368, in __init__
    self.init_model_worker()
  File ".../scheduler.py", line 565, in init_model_worker
    self.maybe_init_draft_worker()
  File ".../scheduler.py", line 561, in maybe_init_draft_worker
    self.draft_worker = DraftWorkerClass(**draft_worker_kwargs)
  File ".../speculative/eagle_worker.py", line 142, in __init__
    super().__init__(...)
  File ".../managers/tp_worker.py", line 247, in __init__
    self._init_model_runner()
  ...
  File ".../model_loader/loader.py", line 1426, in load_model
    param_data = state_dict[key].data
KeyError: 'model.layers.47.input_layernorm.weight'
```

The crash happens deterministically on every startup attempt. The main model loads
successfully (~5 min for 13 shards), CUDA graphs are captured, but then the Scheduler
subprocess dies when initializing the draft worker. Exit code 137 (SIGKILL from the
child's SIGQUIT propagation).

## Root Cause

The code path for draft model loading:

1. `Scheduler.maybe_init_draft_worker()` creates `draft_worker_kwargs` with
   `server_args=self.server_args` (same object as the main model's server_args).

2. If `speculative_draft_load_format` is set, `maybe_init_draft_worker` overrides
   `self.server_args.load_format` in-place. But if it's `None` (the default), the
   draft model inherits whatever `load_format` the main model uses — including
   `sharded_state`.

3. `TpModelWorker._init_model_config()` correctly selects
   `speculative_draft_model_path` for the draft model's `model_path`. But when
   `speculative_draft_model_path` is also `None` (default), `ServerArgs.__init__`
   sets it to `self.model_path` — i.e., the sharded directory.

4. The draft model's `ModelRunner` thus uses `ShardedStateLoader` pointing at the
   sharded directory. The shard files contain per-rank weight slices for the **main**
   model architecture. The **draft/MTP** model has a different architecture with
   different weight keys, causing `KeyError` on the first non-matching key.

The fundamental issue: `ServerArgs` defaults make the draft model silently inherit
both `load_format` and `model_path` from the main model, which is correct for
`auto` loading (the HF cache contains all weights) but broken for `sharded_state`
(the shard files only contain main model weights).

## Fix (upstream)

SGLang should either:

1. Default `speculative_draft_load_format` to `"auto"` when `load_format` is
   `"sharded_state"` (since pre-sharded checkpoints are never created with draft
   model awareness), or

2. Default `speculative_draft_model_path` to the original model ID/path (not the
   sharded directory) when `load_format` is `"sharded_state"`.

The infrastructure is already in place — `speculative_draft_load_format` and
`speculative_draft_model_path` exist and work correctly. The bug is purely in
the defaults not accounting for the `sharded_state` case.

## Our Workaround

No monkey-patch needed. SGLang already provides CLI flags for draft model overrides.
In `sglang_launch.sh`, when both speculative and sharded_state are active:

```bash
if [ "$SGLANG_SPECULATIVE_ENABLED" = "true" ]; then
  args+=(--speculative-algo "$SGLANG_SPECULATIVE_ALGO")
  # ... other speculative args ...

  # WORKAROUND: force auto load format for draft model when using sharded_state
  if [ "$SGLANG_LOAD_FORMAT" = "sharded_state" ]; then
    args+=(--speculative-draft-load-format auto)
    args+=(--speculative-draft-model-path "$SGLANG_MODEL")
  fi
fi
```

This forces the draft model to:
- Use `DefaultModelLoader` (`auto`) instead of `ShardedStateLoader`
- Load from the original HF model ID (resolved from HF cache) instead of the
  sharded directory

The main model continues to use `sharded_state` for fast loading. The draft model
loads from the HF cache, which adds a few seconds to startup but works correctly.

**Note:** `maybe_init_draft_worker` modifies `self.server_args.load_format` in-place
(not a copy) when `speculative_draft_load_format` is set. This mutates the shared
`server_args` object. It's harmless because the main model is already loaded at this
point, but it's a latent bug that could bite if SGLang ever re-reads `load_format`
after draft worker initialization.

## Impact

Without the workaround, speculative decoding (NEXTN/EAGLE) is completely unusable
with `--load-format sharded_state`. The pod enters a crash loop: each restart loads
the main model (~5 min), then immediately crashes on draft worker init. The worker
pod's NCCL connection breaks on each head restart, requiring a livenessProbe-driven
restart of the worker as well — doubling the blast radius.

## Related

- `SGLANG_TP_EP_MOE_UPSTREAM_BUG.md` — moe_wna16 qzeros + EP bug (same version)
- SGLang `ServerArgs.speculative_draft_load_format` — the escape hatch we use
- SGLang `TpModelWorker._init_model_config()` — correctly dispatches draft vs main
  model path, but relies on correct defaults upstream
