"""[dgxarley] Route IndexerKPool's paged MQA logits to the torch/triton fallback on SM121.

GLM-5.3-Flash (index_kpool=4) scores its pooled index keys in
dsa/dsa_indexer_kpool.py::IndexerKPool._get_topk_paged with
`deep_gemm.fp8_paged_mqa_logits`, a call site p30 does not touch (p30 only
wires the regular Indexer in dsa_indexer.py). With `dsa_paged_mqa_logits_backend:
torch` the KPool decode/verify path therefore silently stayed on DeepGEMM. On the
0.5.21-sm121 image DeepGEMM's paged logits + metadata DO run on SM121 (GPU-verified
2026-10-08, ~1e-7 rel err), so this patch is a consistency/speed measure (triton
early-exit on the true length), not a crash fix; `auto` keeps the DeepGEMM path.

Edits (v0.5.21 image, three anchors, inert unless the backend resolves to "torch"):
  1. dsa_indexer_kpool.py: `elif` arm in `_get_topk_paged` calling p30's
     `fp8_paged_mqa_logits_torch_dsa` (p35 turns that into the triton kernel).
     The index-K buffer is uint8 there; the arm passes a float8_e4m3fn view because
     the triton kernel converts loaded bytes numerically.
  2. dsa_indexer_kpool.py: `build_schedule_metadata` is False for the torch backend,
     so `_get_kpool_decode_metadata` never calls `deep_gemm.get_paged_mqa_logits_metadata`.
  3. dsa_backend_kpool.py: `_build_kpool_paged_mqa_schedule_metadata` returns False for
     the torch backend, which keeps kpool_plan's own schedule-metadata builders inert
     (every replay copy site guards on `is not None`).

The ragged prefill call (`_fp8_mqa_logits` -> `deep_gemm.fp8_mqa_logits`) is NOT patched:
GPU-verified on spark5 (2026-10-08, image 0.5.21-sm121) that it runs on SM121 and
matches a float32 reference.

The backend is compared via `.value == "torch"` so a missing p30 enum method cannot
break non-torch servers. Must run after p30 and p35 (filename order).

DELETE WHEN upstream routes IndexerKPool through the DSAPagedMQALogitsBackend dispatch.
"""

from _patchlib import Patch

K_TARGET = "sglang/srt/layers/attention/dsa/dsa_indexer_kpool.py"
BACKEND_TARGET = "sglang/srt/layers/attention/dsa/dsa_backend_kpool.py"

patch_indexer = Patch(name="IndexerKPool paged logits torch arm", target=K_TARGET)

OLD_SCHEDULE_FLAG = """                build_schedule_metadata=not (
                    use_aiter_paged_mqa or use_tilelang_paged_mqa
                ),"""
NEW_SCHEDULE_FLAG = """                build_schedule_metadata=not (
                    use_aiter_paged_mqa
                    or use_tilelang_paged_mqa
                    or self.paged_mqa_logits_backend.value == "torch"  # dgxarley p69
                ),"""

# >= v0.5.23: the argument fits on one line.
OLD_SCHEDULE_FLAG_V0523 = "            build_schedule_metadata=not (use_aiter_paged_mqa or use_tilelang_paged_mqa),"
NEW_SCHEDULE_FLAG_V0523 = """            build_schedule_metadata=not (
                use_aiter_paged_mqa
                or use_tilelang_paged_mqa
                or self.paged_mqa_logits_backend.value == "torch"  # dgxarley p69
            ),"""
SCHEDULE_FLAG_VARIANTS = [
    (OLD_SCHEDULE_FLAG, NEW_SCHEDULE_FLAG),
    (OLD_SCHEDULE_FLAG_V0523, NEW_SCHEDULE_FLAG_V0523),
]
MARKER_SCHEDULE_FLAG = 'self.paged_mqa_logits_backend.value == "torch"  # dgxarley p69'

OLD_ARM = """        else:
            logits = deep_gemm.fp8_paged_mqa_logits(
                q_fp8.unsqueeze(1),
                kv_cache_fp8,
                weights,
                pool_context_lens,"""
NEW_ARM = """        elif self.paged_mqa_logits_backend.value == "torch":  # dgxarley p69 torch arm
            from sglang.srt.layers.attention.dsa.torch_paged_mqa_logits import (
                fp8_paged_mqa_logits_torch_dsa,
            )

            logits = fp8_paged_mqa_logits_torch_dsa(
                q_fp8.unsqueeze(1),
                kv_cache_fp8.view(torch.float8_e4m3fn),
                weights,
                pool_context_lens[: q_fp8.shape[0]],
                pool_block_tables[: q_fp8.shape[0]],
                None,
                pool_max_seq_len,
                clean_logits=False,
            )
""" + OLD_ARM


@patch_indexer.run
def apply_indexer(p: Patch) -> None:
    p.replace_any(SCHEDULE_FLAG_VARIANTS, marker=MARKER_SCHEDULE_FLAG, what="build_schedule_metadata flag")
    p.replace(OLD_ARM, NEW_ARM, marker="# dgxarley p69 torch arm", what="paged logits torch arm")


patch_backend = Patch(name="KPool backend schedule-metadata flag", target=BACKEND_TARGET)

OLD_FLAG_FN = """    def _build_kpool_paged_mqa_schedule_metadata(self) -> bool:
        if self.device_sm_major == 9:"""
NEW_FLAG_FN = """    def _build_kpool_paged_mqa_schedule_metadata(self) -> bool:
        _backend = getattr(self, "paged_mqa_logits_backend", None)  # dgxarley p69
        if _backend is not None and _backend.value == "torch":
            return False
        if self.device_sm_major == 9:"""


@patch_backend.run
def apply_backend(p: Patch) -> None:
    p.replace(OLD_FLAG_FN, NEW_FLAG_FN, marker="# dgxarley p69", what="schedule-metadata flag")
