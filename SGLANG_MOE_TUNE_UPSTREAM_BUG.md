# SGLang Upstream Bug: MoE Triton Tuning breaks on multimodal MoE models (text_config unwrap)

## Status

**Fixed upstream** as of 2026-03-27. Re-verified 2026-04-14: no regressions reported; fix remains in current `main`.

- Fix PR: [#20232](https://github.com/sgl-project/sglang/pull/20232) — "[fix] qwen3.5 fuse_moe_triton_tune bug", merged 2026-03-27 (commit `e2b8463c`)
- File: `benchmark/kernels/fused_moe_triton/common_utils.py`, function `get_model_config()`
- Transformers version: 5.3.0 (unchanged)
- Included in: v0.5.10rc0 (tagged 2026-03-28, ~6h after merge), v0.5.10 stable (2026-04-06), and v0.5.10.post1 (2026-04-09). Not in v0.5.9 (2026-02-24).
- Our previous image (`0.5.9-dev2-acab24a7`, 2026-03-11) predated the fix. Current image (`0.5.10`) includes the fix — `sglang_tune_moe.sh` patch auto-skips (target string no longer found)

## Affected Configuration

- Model: any multimodal MoE model where `architectures` and `quantization_config` live only on the top-level config, not on the text sub-config
- Tested with: `Qwen/Qwen3.5-122B-A10B-FP8` (`Qwen3_5MoeForConditionalGeneration`)
- Tuning script: `benchmark/kernels/fused_moe_triton/tuning_fused_moe_triton.py`

Dense MoE models (Qwen3-235B, DeepSeek-V3, etc.) are **not affected** — they have no `text_config` nesting.

## Bug 1: `architectures` lost → crash

`get_model_config()` calls `config.get_text_config()` to unwrap the text sub-config for multimodal models **before** accessing `config.architectures[0]`. The text sub-config does **not** carry `architectures` — that field only exists on the top-level config.

```python
def get_model_config(model_name, tp_size, ep_size=1, ...):
    config = get_config(model_name, trust_remote_code=True)

    if hasattr(config, "text_config"):
        config = config.get_text_config()  # <-- architectures is lost here

    # ... quantization_config handling ...

    architecture = config.architectures[0]  # <-- TypeError: 'NoneType' is not subscriptable
```

For `Qwen/Qwen3.5-122B-A10B-FP8`:
- Top-level config: `architectures = ["Qwen3_5MoeForConditionalGeneration"]`
- `config.get_text_config()`: `architectures = None`

### Traceback

```
File "tuning_fused_moe_triton.py", line 515, in <module>
    main(args)
File "tuning_fused_moe_triton.py", line 370, in main
    model_config = get_model_config(...)
File "common_utils.py", line 65, in get_model_config
    architecture = config.architectures[0]
TypeError: 'NoneType' object is not subscriptable
```

## Bug 2: `quantization_config` lost → wrong config filename (no block_shape)

Same root cause. `quantization_config` (containing `weight_block_size: [128, 128]` for FP8 fine-grained models) also lives on the top-level config and is lost after `get_text_config()`. This causes `block_shape = None`, so the generated JSON filename omits `block_shape`:

```
Generated: E=128,N=1024,device_name=NVIDIA_GB10,dtype=fp8_w8a8.json
SGLang expects: E=128,N=1024,device_name=NVIDIA_GB10,dtype=fp8_w8a8,block_shape=[128, 128].json
```

SGLang logs: `Using default MoE kernel config. Performance might be sub-optimal!` — the tuned config exists but SGLang can't find it because the filename doesn't match.

## Additional: no `_down` MoE config support in main tuning script

SGLang also looks for a `_down.json` variant (down-projection MoE kernel), but the main tuning script (`tuning_fused_moe_triton.py`) does not support generating this. SGLang falls back to defaults for the down projection. Non-critical.

```
Using MoE kernel config with down_moe=False. Performance might be sub-optimal!
Config file not found at .../E=128,N=1024,device_name=NVIDIA_GB10,dtype=fp8_w8a8,block_shape=[128, 128]_down.json
```

A separate script `tuning_fused_moe_triton_sep.py` (introduced by PR #10567, 2025-10-28) **does** support
`_down.json` generation — it tunes gate/up and down-projection kernels separately via `save_configs_sep()`
with `down_moe=True`. Our `sglang_tune_moe.sh` currently uses the main script, not the `_sep` variant.

## Fix

Save both `architectures` and `quantization_config` before unwrapping `text_config`:

```python
if hasattr(config, "text_config"):
    _architectures = config.architectures
    _quant_config = getattr(config, "quantization_config", None)
    config = config.get_text_config()
    if config.architectures is None:
        config.architectures = _architectures
    if not hasattr(config, "quantization_config") and _quant_config is not None:
        config.quantization_config = _quant_config
```

## Our Workaround

`roles/k8s_dgx/files/sglang_tune_moe.sh` patches `common_utils.py` after downloading it from GitHub. The patch is idempotent — if upstream fixes the code and the patch target is no longer found, it prints a warning and continues.

Since the upstream fix (PR #20232) landed 2026-03-27, the patch target will no longer be found when
downloading from current `main`. The script handles this gracefully (warns and proceeds with the
unpatched upstream file, which now works correctly). The workaround can be removed on the next
image rebuild that includes the fix.
