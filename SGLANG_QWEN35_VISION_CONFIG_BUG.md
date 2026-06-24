# Qwen3.5 vision_config dict bug (transformers 5.x + SGLang ≤ 0.5.12)

> **Re-verified 2026-05-18:** v0.5.12 (released 2026-05-16) does **not** fix
> this — release notes contain no vision_config / Qwen3.5 config fix. Code at
> `python/sglang/srt/configs/qwen3_5.py` is structurally unchanged (same
> hand-written `__init__` that gets bypassed by transformers 5.x's
> auto-generated init for sub-configs). PR #22839 still open (last update
> 2026-04-15), no merge. Monkey-patch in `sglang_launch.sh` still required.
>
> **Re-verified 2026-06-11:** PR #22839 still open (last update 2026-04-15),
> PR #22618 still open (last update 2026-04-14). `qwen3_5.py` untouched on
> main and `release/v0.5.13` since 2026-04-16. **v0.5.13** (official GitHub
> Release published 2026-06-13) does **not** contain a fix.
> Monkey-patch in `sglang_launch.sh` still required.
>
> **Re-verified 2026-06-14:** v0.5.13 is an official GitHub Release since
> 2026-06-13. PR #22839 still OPEN (last update 2026-06-11); PR #22618 still
> OPEN. The v0.5.13 GitHub Release contains no Qwen3.5 vision_config fix —
> `qwen3_5.py` still lacks `from_dict`/`__post_init__` upstream. Monkey-patch
> in `sglang_launch.sh` still required.
>
> **Re-verified 2026-06-24:** PR #22839 still OPEN (no activity since
> 2026-06-11); PR #22618 still OPEN (no activity since 2026-04-14).
> `qwen3_5.py` on main unchanged — no `from_dict` / `__post_init__` added,
> auto-generated `__init__` bypass still present. No fix in v0.5.13.
> Monkey-patch in `sglang_launch.sh` still required.
> Note: v0.5.13.post1 (2026-06-15) is a bare git tag with no GitHub Release
> page and no scitrera Docker image — it is not a delivery vehicle.
>
> **Update 2026-06-12:** PR #22839 received a new third-party comment on
> 2026-06-11T20:28Z (user `SeedSource`) confirming the bug is still present on
> SGLang 0.5.13 and identifying a second broken sub-config in the same scope:
> `norm_topk_prob` accessed via `text_config`. Exact quote (truncated at 600
> chars per API): *"Confirming this is still live and now blocks two paths: on
> stock 0.5.12/0.5.13 + transformers 5.x, Qwen3.5-397B-A17B-NVFP4 cannot load
> at all — we hit the vision_config promotion failure this PR fixes, plus a
> second sub-config in the same scope (norm_topk_prob via text_config).
> Reproduced June 11 on 4x DGX Spark; dgxarley's published workaround is
> semantically correct but file-level patching is required since transformers
> 5.x binds __post_init__ at class creation."*
> The `norm_topk_prob via text_config` failure is an additional attribute
> access on the still-dict `text_config`; our monkey-patch converts all dict
> sub-configs generically and already covers this case. No further cluster
> action required.


## Summary

`Qwen3VLMoeVisionModel.__init__` crashes with `AttributeError: 'dict' object has no attribute 'hidden_size'` when loading Qwen3.5-397B-A17B-NVFP4 (and likely any Qwen3.5/Qwen3-VL model) on SGLang 0.5.10 with transformers 5.5.0.

## Root Cause

Transformers 5.x auto-generates `__init__` for `PretrainedConfig` subclasses that declare `sub_configs`. This auto-generated init uses dataclass-style `setattr(self, field.name, value)` to set attributes directly from kwargs, **bypassing** the manual dict-to-config conversion in the hand-written `__init__`:

```python
# Qwen3_5Config.__init__ (sglang/srt/configs/qwen3_5.py)
def __init__(self, vision_config=None, ...):
    if isinstance(vision_config, dict):
        self.vision_config = self.sub_configs["vision_config"](**vision_config)  # NEVER REACHED
```

The auto-generated init calls `__post_init__()` instead, but the conversion logic lives in `__init__`, not `__post_init__`. Result: `config.vision_config` remains a raw `dict`.

When `Qwen3VLMoeVisionModel` (qwen3_vl.py:307) accesses `vision_config.hidden_size`, it fails because `dict` has no attributes — only keys.

### Why only Qwen3_5MoeConfig?

`Qwen3_5Config` (the non-MoE parent) works because its `__init__` IS the hand-written one. But `Qwen3_5MoeConfig` inherits `Qwen3_5Config` and declares its own `sub_configs`:

```python
class Qwen3_5MoeConfig(Qwen3_5Config):
    model_type = "qwen3_5_moe"
    sub_configs = {
        "vision_config": Qwen3_5MoeVisionConfig,
        "text_config": Qwen3_5MoeTextConfig,
    }
```

Transformers 5.x sees `sub_configs` and generates a new `__init__` for `Qwen3_5MoeConfig`, shadowing the inherited one from `Qwen3_5Config`.

Verified: `Qwen3_5MoeConfig.__init__ is Qwen3_5Config.__init__` → `False`.

### Reproduction

```python
from sglang.srt.configs.qwen3_5 import Qwen3_5MoeConfig

config = Qwen3_5MoeConfig.from_pretrained(
    "nvidia/Qwen3.5-397B-A17B-NVFP4", trust_remote_code=True
)
print(type(config.vision_config))  # <class 'dict'> — should be VisionConfig
print(config.vision_config.hidden_size)  # AttributeError!
```

## Affected versions

- **SGLang:** 0.5.10 (and likely 0.5.10rc0 — any version with `Qwen3_5MoeConfig` in `_CONFIG_REGISTRY`)
- **Transformers:** 5.5.0 (any 5.x with auto-generated `__init__` for sub_configs)
- **Models:** `nvidia/Qwen3.5-397B-A17B-NVFP4`, likely all Qwen3.5-MoE variants

Not affected: pure text models (no `vision_config`), non-MoE Qwen3.5 (uses parent `__init__` directly).

## Symptoms

Two cascading crashes:

1. **`vision_config`**: `AttributeError: 'dict' object has no attribute 'hidden_size'` at `qwen3_vl.py:307`
2. **`text_config`**: `AttributeError: 'PreTrainedConfig' object has no attribute 'layers_block_type'` at `qwen3_5.py:919` — the language model constructor receives a generic `PreTrainedConfig` instead of `Qwen3_5MoeTextConfig`

Both are caused by the same root issue: dict sub-configs not converted to their proper config classes.

## Monkey-patch (sglang_launch.sh)

Injects `__post_init__` into `Qwen3_5Config` (inherited by `Qwen3_5MoeConfig`) that converts dict sub-configs to their proper config classes. The auto-generated `__init__` calls `__post_init__` after setting all fields, so this is the correct hook point.

```python
def __post_init__(self, **kwargs):
    for key, config_cls in self.sub_configs.items():
        val = getattr(self, key, None)
        if isinstance(val, dict):
            setattr(self, key, config_cls(**val))
    super().__post_init__(**kwargs)
```

Guard: `grep -q 'class Qwen3_5MoeConfig'` + no `__post_init__` already present.

## Proper fix (upstream)

Two options:

1. **SGLang fix:** Move the dict-to-config conversion into `__post_init__()` (which the auto-generated init DOES call) instead of `__init__`:
   ```python
   class Qwen3_5Config(PretrainedConfig):
       def __post_init__(self, **kwargs):
           if isinstance(self.vision_config, dict):
               self.vision_config = self.sub_configs["vision_config"](**self.vision_config)
           if isinstance(self.text_config, dict):
               self.text_config = self.sub_configs["text_config"](**self.text_config)
           super().__post_init__(**kwargs)
   ```

2. **Transformers fix:** The auto-generated `__init__` should respect `sub_configs` and auto-convert dict values to the declared config classes, same as what the hand-written init does.

## Related issues

### sgl-project/sglang#20973 — `linear_attn.in_proj_a.input_scale` not found

Reported for `AxionML/Qwen3.5-35B-A3B-NVFP4` on DGX Spark. Different root cause: that checkpoint appears to have quantized linear_attn layers, causing `input_scale` parameters that the model code doesn't expect.

**Does NOT affect `nvidia/Qwen3.5-397B-A17B-NVFP4`:** the official NVIDIA NVFP4 checkpoint correctly excludes linear_attn from quantization (all `linear_attn.*` in the `ignore` list). The checkpoint has 0 `input_scale` params for linear_attn layers — only MoE expert layers are quantized (92,160 `input_scale` params = 512 experts × 3 projections × 60 layers).

### sgl-project/sglang#22618 — Qwen3.5 `linear_attn` quantization guard for compressed-tensors NVFP4

PR opened 2026-04-12, status **OPEN** (re-verified 2026-06-11, last touched 2026-04-14 — no movement since). Fixes silent weight-dropping for `compressed-tensors` NVFP4 checkpoints of Qwen3.5 hybrid models (RedHatAI/Qwen3.5-35B-A3B-NVFP4, RedHatAI/Qwen3.5-122B-A10B-NVFP4). Related to #20973 (same `linear_attn.in_proj_a.input_scale` symptom, different checkpoint format). **Not the same bug as the vision_config dict issue documented here** — this is a quantization guard issue in `qwen3_5.py`, not a `sub_configs` / auto-generated `__init__` problem.

## Upstream references

- [PR #22839](https://github.com/sgl-project/sglang/pull/22839) — "fix(config): Add from_dict() for Qwen3VL config classes" (opened 2026-04-15 by `libermeng`, **OPEN** as of 2026-06-11 — no movement since 2026-04-15 author push, not in v0.5.11/v0.5.12/v0.5.13). Addresses the same root cause this doc describes ("Transformers 5.5.0+ natively supports Qwen3-VL, causing `AutoConfig.from_pretrained()` to skip sglang's config conversion logic. This leaves nested `vision_config` and `text_config` as dicts instead of config objects."). Different fix vehicle than ours: adds a `from_dict()` classmethod to `Qwen3VLConfig`/`Qwen3VLMoeConfig`/`Qwen3_5Config`/`Qwen3_5MoeConfig` and registers them in `_CONFIG_REGISTRY`, instead of hooking `__post_init__`. Also fixes `Qwen3_5MoeTextConfig` MoE attributes. Includes 9 unit tests. CI re-run requested 2026-04-15 — no human review since. **Not yet merged → our monkey-patch is still required.**
- Related: transformers 5.x `PretrainedConfig.__init_subclass__` auto-init behavior
- Related: sgl-project/sglang#20973 (different Qwen3.5 NVFP4 checkpoint, different bug)
