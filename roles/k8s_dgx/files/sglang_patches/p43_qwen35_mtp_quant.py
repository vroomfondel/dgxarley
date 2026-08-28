"""[dgxarley] qwen3_5_mtp.py: keep quant_config for a QUANTIZED MTP/NEXTN draft head.

qwen3_5_mtp.py hardcodes "The MTP model is unquantized in the nvfp4 checkpoint"
and nulls quant_config the moment quant_config.get_name() is "modelopt_fp4" /
"modelopt_mixed":

    if quant_config and quant_config.get_name() in (
        "modelopt_fp4",
        "modelopt_mixed",
    ):
        quant_config = None

That assumption holds for every NVIDIA-published Qwen3.5/3.6 NVFP4 checkpoint,
which ships a BF16 draft head (mtp.* on the exclude list). But a checkpoint that
DELIBERATELY quantizes the MTP head (kikube's surgical MTP requant --
quantizer/requant_mtp_nvfp4.py, mirroring the main model's per-expert NVFP4 MoE +
FP8 KV) is exactly the case this shortcut excludes: with quant_config nulled the
draft head is built UNQUANTIZED, so its FusedMoE/linears allocate BF16 shapes and
the weight loader then narrows the UNPACKED intermediate dim (e.g. 512) over the
NVFP4-PACKED expert weight (256) ->
"RuntimeError: start (0) + length (512) exceeds dimension size (256)" in
fused_moe_triton/layer.py:_load_w2, mid-load of the draft worker. (SGLang itself
already sets speculative_draft_model_quantization='modelopt_fp4' in server_args,
so the null is internally inconsistent.)

Fix: only null quant_config when the MTP really is BF16. Probe the exclude list --
if a representative MTP expert is NOT excluded, the checkpoint's MTP is quantized,
so keep quant_config. Standard NVIDIA NVFP4 (BF16 MTP, blanket mtp* exclude ->
the probe IS excluded) keeps the original behaviour, so this is INERT for every
existing checkpoint. No model-name gate: qwen3_5_mtp.py is imported only for the
Qwen3.5 MTP arch. Deletable once SGLang upstream honours a quantized MTP for
modelopt_fp4 (track alongside sgl-project/sglang MTP-quant support).

RE-ANCHORED 2026-08-28 for v0.5.18: upstream moved the decision out of
Qwen3_5ForCausalLMMTP.__init__ into the module-level helper
_mtp_quant_config(), which RETURNS None instead of assigning (4-space body),
and narrowed the modelopt_fp4 arm to is_checkpoint_nvfp4_serialized. Same
intent, so the probe is unchanged and both spellings are kept (replace_any):
one ConfigMap serves instances pinned to pre- and post-0.5.18 images.
"""

from _patchlib import Patch

patch = Patch(
    name="keep quant_config for a quantized MTP draft head",
    target="sglang/srt/models/qwen3_5_mtp.py",
)

# The MTP quantization decision was extracted into the module-level helper
# _mtp_quant_config() in v0.5.18 (same intent, but it RETURNS None instead of
# assigning, sits at 4-space indent, and the modelopt_fp4 arm now additionally
# requires is_checkpoint_nvfp4_serialized). One ConfigMap feeds instances pinned
# to different images, so both spellings must keep working: the probe body is
# shared, only the statement that disables quantization differs.
_PROBE = """            # [patch dgxarley] keep quant_config when THIS checkpoint's MTP is
            # quantized (kikube surgical requant). Probe the exclude list: a
            # representative MTP expert NOT excluded => MTP is quantized => keep.
            _mtp_probe = "mtp.layers.0.mlp.experts.0.down_proj"
            _mtp_is_quantized = hasattr(quant_config, "is_layer_excluded") and not quant_config.is_layer_excluded(_mtp_probe)
            if not _mtp_is_quantized:
                {disable}"""

# <= v0.5.17: inline in Qwen3_5ForCausalLMMTP.__init__, 8-space body.
OLD_INLINE = """        if quant_config and quant_config.get_name() in (
            "modelopt_fp4",
            "modelopt_mixed",
        ):
            quant_config = None"""
NEW_INLINE = """        if quant_config and quant_config.get_name() in (
            "modelopt_fp4",
            "modelopt_mixed",
        ):
""" + _PROBE.format(disable="quant_config = None")

# >= v0.5.18: the free function _mtp_quant_config(), 4-space body, returns None.
_HELPER_HEAD = """    if quant_config and (
        quant_config.get_name() == "modelopt_mixed"
        or (
            quant_config.get_name() == "modelopt_fp4"
            and quant_config.is_checkpoint_nvfp4_serialized
        )
    ):
"""
OLD_HELPER = _HELPER_HEAD + """        return None"""
NEW_HELPER = _HELPER_HEAD + "\n".join(
    line[4:] if line.startswith("    ") else line for line in _PROBE.format(disable="return None").split("\n")
)


@patch.run
def apply(p: Patch) -> None:
    p.replace_any(
        [(OLD_INLINE, NEW_INLINE), (OLD_HELPER, NEW_HELPER)],
        marker="_mtp_is_quantized",
        what="mtp quant-keep",
    )
