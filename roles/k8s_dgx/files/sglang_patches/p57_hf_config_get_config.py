"""[dgxarley] hf_transformers/config.py: convert dict sub_configs after loading (transformers 5.5.0 bug).

Patch SGLang get_config() to convert dict sub_configs after loading (transformers 5.5.0 bug).
Transformers 5.x auto-generates __init__ for PretrainedConfig subclasses with sub_configs,
bypassing dict→config conversion. from_pretrained() also bypasses __post_init__.
Both vision_config and text_config arrive as raw dicts → AttributeError on .hidden_size etc.
Fix: patch get_config() to convert dict sub-configs after loading for any config with sub_configs.

RE-ANCHORED 2026-07-16: sglang/srt/utils/hf_transformers_utils.py is now a bare
backward-compat shim ("all code has moved to sglang.srt.utils.hf_transformers") — it
has no get_config()/"return config" left, so the old anchor silently never matched on
this image. The real get_config() lives in sglang/srt/utils/hf_transformers/config.py.
The underlying transformers-5.x sub_configs bug is STILL UNSOLVED there: only the
Mistral parser path calls the sglang-native _ensure_sub_configs() helper (for
text_config/vision_config); the generic "hf" parser path (used by everything else,
incl. GLM/DeepSeek/NemotronH) does not call it anywhere — verified by grepping the
whole hf_transformers package. So this patch still applies, just at the new home.

RE-ANCHORED 2026-08-28 for v0.5.18: the gguf block just above the return grew
sidecar-config.json support, so its condition (`if is_gguf and not
gguf_has_sidecar_config:`) and its raise message (now three lines) changed.
Only the anchor moved; the injected code is identical, and both spellings are
kept (replace_any) for instances pinned to older images.
"""

from _patchlib import Patch

patch = Patch(
    name="get_config() sub_configs dict->config conversion", target="sglang/srt/utils/hf_transformers/config.py"
)

MARKER = "sub_configs dict fix"

# Find the final "return config" in get_config() and add sub_configs conversion
# before it. The gguf branch keeps getting refactored around that return
# (model_type/config.update -> _set_architectures helper; then the sidecar-config
# condition and the multi-line raise in v0.5.18), so the anchor is kept as one
# explicit variant per spelling. One ConfigMap feeds instances pinned to
# different images, so both must keep working.
INJECT = """
    # [patch] sub_configs dict fix — transformers 5.x from_pretrained() leaves sub-configs
    # as raw dicts instead of converting to their declared config classes.
    _sub_cfgs = getattr(config, "sub_configs", None)
    if _sub_cfgs:
        for _key, _cls in _sub_cfgs.items():
            _val = getattr(config, _key, None)
            if isinstance(_val, dict):
                try:
                    setattr(config, _key, _cls(**_val))
                except Exception:
                    pass  # non-critical: some sub-configs may not accept all dict keys

    # [patch] Qwen3.5 MoE: text_config lacks norm_topk_prob (Qwen2MoeSparseMoeBlock expects it).
    # Qwen3.5 uses softmax routing — renormalize=True is correct default.
    _tc = getattr(config, "text_config", None)
    if _tc is not None and not isinstance(_tc, dict) and not hasattr(_tc, "norm_topk_prob"):
        _tc.norm_topk_prob = True

    return config"""

# <= v0.5.17
OLD_GGUF = """    if is_gguf:
        if config.model_type not in MODEL_FOR_CAUSAL_LM_MAPPING_NAMES:
            raise RuntimeError(f"Can't get gguf config for {config.model_type}.")
        _set_architectures(config, MODEL_FOR_CAUSAL_LM_MAPPING_NAMES[config.model_type])

    return config"""

# >= v0.5.18: gguf sidecar config.json support (the branch is now conditional on
# gguf_has_sidecar_config and the raise message spans three lines).
OLD_GGUF_SIDECAR = """    if is_gguf and not gguf_has_sidecar_config:
        if config.model_type not in MODEL_FOR_CAUSAL_LM_MAPPING_NAMES:
            raise RuntimeError(
                f"Can't get gguf config for {config.model_type}. Place a "
                "config.json next to the .gguf file to load the config from "
                "there instead."
            )
        _set_architectures(config, MODEL_FOR_CAUSAL_LM_MAPPING_NAMES[config.model_type])

    return config"""


def _variant(old: str) -> tuple[str, str]:
    """Keep the gguf block verbatim, replace only its trailing `return config`."""
    assert old.endswith("\n    return config")
    return old, old[: -len("\n    return config")] + INJECT


@patch.run
def apply(p: Patch) -> None:
    p.replace_any(
        [_variant(OLD_GGUF), _variant(OLD_GGUF_SIDECAR)],
        marker=MARKER,
        what="get_config sub_configs dict fix",
    )
