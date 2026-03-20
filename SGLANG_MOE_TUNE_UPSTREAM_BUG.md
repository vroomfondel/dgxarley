# SGLang Upstream Bug: MoE Triton Tuning crashes on Qwen3.5 (multimodal config)

## Status

**Unreported** as of 2026-03-20. Bug exists on SGLang `main` branch.

- File: `benchmark/kernels/fused_moe_triton/common_utils.py`, function `get_model_config()`
- Transformers version: 5.3.0

## Affected Configuration

- Model: any multimodal MoE model where `architectures` lives only on the top-level config, not on the text sub-config
- Tested with: `Qwen/Qwen3.5-122B-A10B-FP8` (`Qwen3_5MoeForConditionalGeneration`)
- Tuning script: `benchmark/kernels/fused_moe_triton/tuning_fused_moe_triton.py`

Dense MoE models (Qwen3-235B, DeepSeek-V3, etc.) are **not affected** — they have no `text_config` nesting.

## The Bug

`get_model_config()` calls `config.get_text_config()` to unwrap the text sub-config for encoder-decoder / multimodal models **before** accessing `config.architectures[0]`. The problem is that `get_text_config()` returns the inner text config object, which does **not** carry the `architectures` field — that field only exists on the top-level multimodal config.

```python
def get_model_config(model_name, tp_size, ep_size=1, ...):
    config = get_config(model_name, trust_remote_code=True)

    # This correctly unwraps the text config for MoE fields (num_experts, etc.)
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

## Fix

Save `architectures` before unwrapping `text_config`:

```python
if hasattr(config, "text_config"):
    _architectures = config.architectures
    config = config.get_text_config()
    if config.architectures is None:
        config.architectures = _architectures
```

## Our Workaround

`roles/k8s_dgx/files/sglang_tune_moe.sh` patches `common_utils.py` after downloading it from GitHub. The patch is idempotent — if upstream fixes the code and the patch target is no longer found, it prints a warning and continues.
