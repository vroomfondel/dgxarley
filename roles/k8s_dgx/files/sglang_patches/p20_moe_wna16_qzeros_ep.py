"""[dgxarley] moe_wna16.py: EP-aware expert_id + tp_rank remapping for qzeros.

SGLang 0.5.9 bug, still present. Two bugs in the w13_qzeros/w2_qzeros branches
of the moe_wna16 weight loader:

  1. Uses the raw GLOBAL expert_id (0-127) to index a param sized for the LOCAL
     expert count (0-63 under EP2).
  2. Uses the global tp_rank for the TP slice, but moe_tp_size = tp/ep, so the
     rank must be tp_rank % moe_tp_size.

Safe no-op when ep_size=1: the global->local map is the identity there and
tp_rank % moe_tp_size == tp_rank.

Same bug exists in vLLM 0.17.0; the same monkey-patch is applied in
vllm_launch.sh. Background: SGLANG_TP_EP_MOE_UPSTREAM_BUG.md.

No model gate: moe_wna16.py is only imported for moe_wna16-quantized models, so
this is inert everywhere else, and applying it unconditionally also covers
checkpoints served from a local path whose repo name we cannot match on.

Re-sync: drop this file once the image ships the fix (the already-applied guards
make it a no-op then, so it is safe to keep across a bump).
"""

from _patchlib import Patch

patch = Patch(
    name="EP-aware expert_id + tp_rank remapping for qzeros", target="sglang/srt/layers/quantization/moe_wna16.py"
)

OLD_W13 = """            if "w13_qzeros" in weight_name:
                tensor = loaded_weight.view(
                    layer.moe_tp_size, -1, loaded_weight.size(1)
                )[tp_rank]
                if shard_id == "w1":
                    param.data[expert_id, : shard_size // 2] = tensor
                else:
                    param.data[expert_id, shard_size // 2 :] = tensor"""

NEW_W13 = """            if "w13_qzeros" in weight_name:
                _local_id = layer._map_global_expert_id_to_local_expert_id(expert_id)
                if _local_id == -1:
                    return
                _moe_tp_rank = tp_rank % layer.moe_tp_size
                tensor = loaded_weight.view(
                    layer.moe_tp_size, -1, loaded_weight.size(1)
                )[_moe_tp_rank]
                if shard_id == "w1":
                    param.data[_local_id, : shard_size // 2] = tensor
                else:
                    param.data[_local_id, shard_size // 2 :] = tensor"""

OLD_W2 = """            elif "w2_qzeros" in weight_name:
                param.data[expert_id] = loaded_weight.view(
                    loaded_weight.size(0), layer.moe_tp_size, -1
                )[:, tp_rank]"""

NEW_W2 = """            elif "w2_qzeros" in weight_name:
                _local_id = layer._map_global_expert_id_to_local_expert_id(expert_id)
                if _local_id == -1:
                    return
                _moe_tp_rank = tp_rank % layer.moe_tp_size
                param.data[_local_id] = loaded_weight.view(
                    loaded_weight.size(0), layer.moe_tp_size, -1
                )[:, _moe_tp_rank]"""


@patch.run
def apply(p: Patch) -> None:
    p.replace(OLD_W13, NEW_W13, what="w13_qzeros EP-aware remap")
    p.replace(OLD_W2, NEW_W2, what="w2_qzeros EP-aware remap")
