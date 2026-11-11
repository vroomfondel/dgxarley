"""[dgxarley] DSA paged-MQA-logits TORCH FALLBACK (GB10/SM121: DeepGEMM + CuteDSL both hard NO-GO).

GlmMoeDsa / DeepSeek-V3.2-family DSA models score paged KV via a paged-MQA-logits kernel
(sglang/srt/layers/attention/dsa/dsa_indexer.py::_get_topk_paged) before top-k selection.
On GB10/SM121, BOTH hardware kernel routes are dead ends, not just unconfigured:
  - DeepGEMM (the default): deep_gemm.get_paged_mqa_logits_metadata() throws a compiled
    C++ "Unsupported architecture" assert on SM121. Upstream declined SM120/SM121 support
    (DeepGEMM PR #318, maintainer cited lack of hardware/capacity) -- not a local gate.
  - cutedsl: gated behind is_sm100_supported()==False on GB10, and even bypassing that
    gate, the kernel's _setup_mma uses tcgen05.MmaF8F6F4Op, a datacenter-Blackwell-only
    (SM100/SM103) tensor-core instruction that does not exist on consumer Blackwell
    (SM121) -- a real ISA boundary, verified 2026-07-16 in a GPU debug pod.
This crashes the TARGET model's every decode step under attention_backend="dsa" (not just
the MTP/NEXTN draft, whose is_nextn indexer always computes topk_indices and needs this
kernel too). Full investigation + design: DSA_speedup.md, dsalogitrework.md (repo root).

Fix: port SGLang's own dsv4-side answer to this exact problem --
fp8_paged_mqa_logits_torch_sm120 (dsv4/indexer.py, upstream PR #24692, merged 2026-06-01,
in this image since v0.5.13) -- into the generic dsa/dsa_indexer.py path GlmMoeDsa uses,
which has no equivalent. The torch path discards the DeepGEMM schedule metadata
(`_ = deep_gemm_metadata`), so all 3 eager DeepGEMM-metadata call sites (2 in
dsa_backend.py::init_forward_metadata + the shared _refresh_paged_mqa_schedule_metadata
cuda-graph-replay helper, which alone covers 2 more replay call sites) can simply be
skipped, not just the logits kernel itself -- verified live (numeric unit tests: dominant-
KV-slot test matches a hand-derived reference exactly, masking correct, no NaN/all-zero/
constant-across-seeds -- the failure signature a community SM120 fallback fork silently
hit). New backend value "torch", OPT-IN ONLY via --dsa-paged-mqa-logits-backend torch (NOT
selected by "auto", so archs where DeepGEMM/CuteDSL already work are unaffected).

PHASE 1 ONLY (dsalogitrework.md Section 3): plain decode, next_n==1 -- unblocks ordinary
target-model decode AND each individual MTP/NEXTN draft step (the draft decodes one token
at a time). Phase 2 (target-verify / draft-extend-v2, next_n>=2) added 2026-07-16 for
MTP: seqlens_expanded is already per-token and q/weights are sliced per token, so the
dispatch only repeat_interleaves block_tables to per-token rows; the kernel itself is
shape-agnostic (batch dim = flattened tokens). Numerically verified against a
per-request-slice reference (see dsalogitrework.md Section 5 discipline).

5 files patched/written, unconditionally (no model-name gate): paged_mqa_logits_backend.py
and server_args.py only add an opt-in enum value / CLI choice (zero behavior change unless
explicitly selected); dsa_backend.py / dsa_indexer.py / the new torch_paged_mqa_logits.py
are DSA-specific source files exercised only by actual DSA models (mirrors the mllama4.py
precedent above: the files are inert for every other model by construction).

This module carries all 5 edits, one Patch object per target file, run in the same order
the original sglang_launch.sh heredocs ran in (PATCH_DSA_TORCH_ENUM -> _SERVERARGS ->
_NEWFILE -> _BACKEND -> _INDEXER): they are one feature (the torch paged-MQA-logits
fallback backend) and the later edits depend on the enum value / new module the earlier
ones add, so the order matters.

The new-file step (torch_paged_mqa_logits.py) is NOT expressed as a Patch: Patch.run()
skips when the target file does not exist, which is always true for a file this patch is
supposed to create. It is written directly with plain os.path + an exists()-and-marker
guard against DIST_PACKAGES, mirroring Patch.run's logging so the launch log reads the
same as the other 4 edits.


RE-ANCHORED 2026-09-11 for v0.5.19, three spots, all via replace_any so instances
pinned to older images keep working:
  * resolve(): the SM100 gate is now `get_platform().is_sm100` instead of
    `is_sm100_supported()` (v0.5.19 moved the arch predicates onto the new
    srt/runtime_context platform object).
  * server_args.py: the module-level DSA_PAGED_MQA_LOGITS_BACKEND_CHOICES
    constant is GONE, the list is inline in the Arg(choices=[...]) declaration.
  * dsa_backend 4b-init: DSATopKBackend(server_args.dsa_topk_backend) became
    DSATopKBackend.resolve(model_runner) (#36313 --speculative-dsa-topk-backend).
"""

import os

from _patchlib import DIST_PACKAGES, Patch, write_module

# ── 1) paged_mqa_logits_backend.py: DSAPagedMQALogitsBackend gets a TORCH member ──

patch_enum = Patch(
    name="DSA torch-backend enum + resolve('torch')",
    target="sglang/srt/layers/attention/dsa/paged_mqa_logits_backend.py",
)

OLD_ENUM = """class DSAPagedMQALogitsBackend(Enum):
    DEEPGEMM = "deepgemm"
    CUTEDSL = "cutedsl"
    AITER = "aiter"

    def is_deepgemm(self) -> bool:
        return self == DSAPagedMQALogitsBackend.DEEPGEMM

    def is_cutedsl(self) -> bool:
        return self == DSAPagedMQALogitsBackend.CUTEDSL

    def is_aiter(self) -> bool:
        return self == DSAPagedMQALogitsBackend.AITER"""
NEW_ENUM = """# [patch] _sgl_dsa_torch_fallback_enum_
class DSAPagedMQALogitsBackend(Enum):
    DEEPGEMM = "deepgemm"
    CUTEDSL = "cutedsl"
    AITER = "aiter"
    TORCH = "torch"  # pure-torch fallback for archs DeepGEMM/CuteDSL don't cover (e.g. SM121/GB10)

    def is_deepgemm(self) -> bool:
        return self == DSAPagedMQALogitsBackend.DEEPGEMM

    def is_cutedsl(self) -> bool:
        return self == DSAPagedMQALogitsBackend.CUTEDSL

    def is_aiter(self) -> bool:
        return self == DSAPagedMQALogitsBackend.AITER

    def is_torch(self) -> bool:
        return self == DSAPagedMQALogitsBackend.TORCH"""


def _resolve_variant(sm100_probe: str) -> tuple[str, str]:
    """(old, new) for one spelling of resolve()'s SM100 gate."""
    head = """        if value == "auto" or value == "deepgemm":
            return DSAPagedMQALogitsBackend.DEEPGEMM
        if value == "aiter":
            raise ValueError("dsa_paged_mqa_logits_backend='aiter' requires ROCm.")
        if value == "cutedsl":
            if not {probe}:
                raise ValueError(
                    "dsa_paged_mqa_logits_backend='cutedsl' requires SM100 (Blackwell)."
                )
            return DSAPagedMQALogitsBackend.CUTEDSL
""".format(probe=sm100_probe)
    tail_old = """        raise ValueError(f"Unknown dsa_paged_mqa_logits_backend: {value!r}")"""
    tail_new = """        if value == "torch":
            # No arch gate: that is the whole point of this backend. NOT selected by
            # "auto" (opt-in only) to avoid silently regressing perf on archs where
            # DeepGEMM/CuteDSL already work (see dsalogitrework.md Section 4.1).
            return DSAPagedMQALogitsBackend.TORCH
        raise ValueError(f"Unknown dsa_paged_mqa_logits_backend: {value!r}")"""
    return head + tail_old, head + tail_new


RESOLVE_VARIANTS = [
    _resolve_variant("is_sm100_supported()"),  # <= v0.5.18
    # >= v0.5.19: arch predicates moved onto the runtime_context platform object.
    _resolve_variant("get_platform().is_sm100"),
]
MARKER_RESOLVE = "return DSAPagedMQALogitsBackend.TORCH"


@patch_enum.run
def apply_enum(p: Patch) -> None:
    p.replace(OLD_ENUM, NEW_ENUM, what="DSAPagedMQALogitsBackend enum")
    p.replace_any(RESOLVE_VARIANTS, marker=MARKER_RESOLVE, what="resolve()")


# ── 2) server_args.py: add 'torch' to the CLI choice list + help string ──

patch_serverargs = Patch(
    name="DSA torch-backend CLI choice",
    # >= v0.5.20: #38375 moved the arg declarations out of server_args.py into
    # arg_groups/fields/<bag>.py. Tried FIRST because server_args.py still EXISTS
    # on that ref (a shim whose get_global_server_args raises), so an alt_targets
    # order that checks it first would pick the wrong file and report drift.
    target="sglang/srt/arg_groups/fields/exec_.py",
    alt_targets=("sglang/srt/server_args.py",),
)

MARKER_CHOICES = '"aiter", "torch"]'
OLD_CHOICES = """DSA_PAGED_MQA_LOGITS_BACKEND_CHOICES = ["auto", "deepgemm", "cutedsl", "aiter"]"""
NEW_CHOICES = """# [patch] _sgl_dsa_torch_fallback_choice_
DSA_PAGED_MQA_LOGITS_BACKEND_CHOICES = ["auto", "deepgemm", "cutedsl", "aiter", "torch"]"""
OLD_HELP = """            help="DSA indexer paged MQA logits kernel backend. Options: 'auto' (default; DeepGEMM on CUDA, aiter on ROCm), 'deepgemm', 'cutedsl' (CuTe DSL kernel, SM 100 (Blackwell) only; wins at low batch size and long context), 'aiter' (ROCm only).","""
NEW_HELP = """            help="DSA indexer paged MQA logits kernel backend. Options: 'auto' (default; DeepGEMM on CUDA, aiter on ROCm), 'deepgemm', 'cutedsl' (CuTe DSL kernel, SM 100 (Blackwell) only; wins at low batch size and long context), 'aiter' (ROCm only), 'torch' (pure-torch fallback, any CUDA arch, e.g. SM120/SM121 where neither DeepGEMM nor CuteDSL run; slower, opt-in only).","""


@patch_serverargs.run
def apply_serverargs(p: Patch) -> None:
    p.replace_any(
        [
            (OLD_CHOICES, NEW_CHOICES),  # <= v0.5.18: module-level constant
            # >= v0.5.19: the constant is gone, the list is inline in the Arg().
            (
                '            choices=["auto", "deepgemm", "cutedsl", "aiter"],',
                '            choices=["auto", "deepgemm", "cutedsl", "aiter", "torch"],',
            ),
        ],
        marker=MARKER_CHOICES,
        what="DSA_PAGED_MQA_LOGITS_BACKEND_CHOICES",
    )
    p.replace(OLD_HELP, NEW_HELP, what="dsa_paged_mqa_logits_backend help-string")


# ── 3) new file: dsa/torch_paged_mqa_logits.py (Phase 1 torch fallback kernel) ──
# Not a Patch (see docstring): written directly, guarded by the same already-applied
# marker style Patch.replace() uses.

_TORCH_KERNEL_TARGET = "sglang/srt/layers/attention/dsa/torch_paged_mqa_logits.py"
_TORCH_KERNEL_PATH = os.path.join(DIST_PACKAGES, _TORCH_KERNEL_TARGET)
# Bumped 2026-07-28 together with the fp8_kernel import fix below. The old marker
# ("fp8_paged_mqa_logits_torch_dsa") is present in BOTH the broken and the fixed
# source, so a container that already carries the broken file would have hit the
# already_written shortcut and kept it. Version the marker with the source.
_TORCH_KERNEL_MARKER = """fp8_paged_mqa_logits_torch_dsa  # dgxarley-v2"""

_TORCH_KERNEL_SOURCE = '''# SPDX-License-Identifier: Apache-2.0
"""
DSA paged-MQA-logits pure-torch fallback (Phase 1: plain decode, next_n == 1).

Ported from sglang.srt.layers.attention.dsv4.indexer.fp8_paged_mqa_logits_torch_sm120
(upstream SGLang PR #24692, merged 2026-06-01, first shipped v0.5.13) for the generic
DSA indexer path (sglang.srt.layers.attention.dsa.dsa_indexer, used by GlmMoeDsa /
DeepSeek-V3.2-family models), which has no equivalent fallback upstream. Copied rather
than imported cross-module: dsv4 and dsa are independent code paths in SGLang, and a
shared import would create an unwanted coupling.

Why this exists: on GB10/SM121 (consumer Blackwell), deep_gemm.get_paged_mqa_logits_metadata
/ fp8_paged_mqa_logits throw a compiled C++ "Unsupported architecture" assert (DeepGEMM
upstream declined SM120/SM121 support, PR #318), and the CuteDSL alternative fails on a
real ISA boundary (tcgen05 MMA is datacenter-Blackwell-only, SM100/SM103). See
DSA_speedup.md and dsalogitrework.md (repo root) for the full investigation.

Phase 1 scope: plain decode only (next_n == 1), matching the call shape of
sglang.jit_kernel.dsa.paged_mqa_logits.deepgemm_paged_mqa_logits_split. Phase 2
(target-verify / next_n >= 2): handled by the DISPATCH in dsa_indexer.py, which folds
the verify batch to per-token form (seqlens_expanded is per-token already; block_tables
repeat_interleaved) -- this kernel stays per-token and needs no next_n awareness.

CORRECTNESS WARNING (dsalogitrework.md Section 5): a community fork (kt-sglang) shipped
a similar-looking torch fallback for SM120 that ran without error but returned WRONG
results (all-zero/NaN logits). "Runs without crashing" is not sufficient verification of
this function — see the numeric checks required in dsalogitrework.md Section 5 before
trusting output from this path in production.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F

# dgxarley: version-tolerant import. RFC #29630 moved fp8_kernel out of
# srt/layers/quantization in SGLang v0.5.16; the pre-0.5.16 images we still pin
# only have the old path. Getting this wrong is SILENT: the ImportError is
# swallowed by SGLang's model registry ("Ignore import error when loading ...")
# and merely disables ~31 model classes (deepseek_v2/v4, glm4_moe, kimi_*, ...).
# marker: fp8_paged_mqa_logits_torch_dsa  # dgxarley-v2
try:  # SGLang >= 0.5.16
    from sglang.kernels.ops.quantization.fp8_kernel import is_fp8_fnuz
except ImportError:  # SGLang <= 0.5.15.post1
    from sglang.srt.layers.quantization.fp8_kernel import is_fp8_fnuz

FP8_DTYPE = torch.float8_e4m3fnuz if is_fp8_fnuz() else torch.float8_e4m3fn


def fp8_paged_mqa_logits_torch_dsa(
    q_fp8: torch.Tensor,
    kvcache_fp8: torch.Tensor,
    weight: torch.Tensor,
    seq_lens: torch.Tensor,
    page_table: torch.Tensor,
    deep_gemm_metadata: Any,
    max_seq_len: int,
    clean_logits: bool = True,
) -> torch.Tensor:
    """CUDA-graph-compatible FP8 paged MQA logits, pure torch (no DeepGEMM/CuteDSL).

    Verbatim port of dsv4.indexer.fp8_paged_mqa_logits_torch_sm120 (vectorized,
    no `.item()` / no data-dependent control flow -> CUDA-graph-capture-safe).
    `deep_gemm_metadata` is accepted for call-site signature compatibility but
    unused: this path does no SM-tiled scheduling (unlike DeepGEMM's kernel), so
    it has no notion of a schedule to consume. Callers may pass None.
    """
    _ = deep_gemm_metadata
    batch_size, _, num_heads, head_dim = q_fp8.shape
    block_size = kvcache_fp8.shape[1]
    device = q_fp8.device

    assert head_dim == 128, "Vectorized torch impl hardcodes DSA indexer head_dim=128"
    assert (
        block_size == 64
    ), "Vectorized torch impl hardcodes block_size=64 cache layout"
    assert q_fp8.shape == (batch_size, 1, num_heads, head_dim)
    assert kvcache_fp8.shape[1:] == (block_size, 1, head_dim + 4)
    assert weight.shape == (batch_size, num_heads)
    if seq_lens.dim() > 1:
        seq_lens = seq_lens.squeeze(-1)
    assert seq_lens.shape == (batch_size,)
    assert page_table.shape[0] == batch_size
    assert clean_logits == False

    max_pages = (max_seq_len + block_size - 1) // block_size
    max_padded_seq = max_pages * block_size

    kvcache_flat = kvcache_fp8.view(-1, block_size * (head_dim + 4))
    SCALE_OFFSET = block_size * head_dim

    page_ids = page_table[:, :max_pages]
    kvcache_gathered = kvcache_flat[page_ids]

    kv_value_raw = kvcache_gathered[..., :SCALE_OFFSET]
    kv_scale_raw = kvcache_gathered[..., SCALE_OFFSET:]

    kv_value = kv_value_raw.contiguous().view(dtype=FP8_DTYPE).to(torch.float32)
    kv_value = kv_value.view(batch_size, max_padded_seq, head_dim)

    kv_scale = kv_scale_raw.contiguous().view(dtype=torch.float32)
    kv_scale = kv_scale.view(batch_size, max_padded_seq)

    q = q_fp8[:, 0].to(torch.float32)

    score = torch.bmm(kv_value, q.transpose(1, 2))

    score = F.relu(score)
    score = score * weight.unsqueeze(1)
    score = score.sum(dim=2)

    score = score * kv_scale

    out_width = min(max_padded_seq, max_seq_len)
    logits = score.new_full((batch_size, max_seq_len), float("-inf"))
    logits[:, :out_width] = score[:, :out_width]

    positions = torch.arange(max_seq_len, device=device)
    invalid_mask = positions.unsqueeze(0) >= seq_lens.unsqueeze(1)
    logits.masked_fill_(invalid_mask, float("-inf"))

    return logits
'''


def _write_torch_kernel_module() -> None:
    """Create torch_paged_mqa_logits.py if missing or not yet carrying the marker.

    Mirrors the original heredoc's guard exactly: `f3.exists() and marker3 in
    f3.read_text()` -> skip; anything else (missing, or present without the marker)
    -> (re)write unconditionally.
    """
    already_written = False
    if os.path.isfile(_TORCH_KERNEL_PATH):
        with open(_TORCH_KERNEL_PATH) as fh:
            already_written = _TORCH_KERNEL_MARKER in fh.read()
    if already_written:
        print("dsa/torch_paged_mqa_logits.py: already written, skipping")
        return
    # write_module() imports the result before declaring victory. Plain open()/write()
    # let the 2026-07-28 fp8_kernel-import breakage through unnoticed; see its docstring.
    write_module(_TORCH_KERNEL_PATH, _TORCH_KERNEL_SOURCE, "fp8_paged_mqa_logits_torch_dsa (phase 1)")


_write_torch_kernel_module()


# ── 4) dsa_backend.py: resolve the backend + gate 2 metadata call sites + the ──
# ── shared cuda-graph-replay refresh helper (5 sub-edits, in source order) ──

patch_dsa_backend = Patch(
    name="DSA torch-backend wiring (dsa_backend.py)",
    target="sglang/srt/layers/attention/dsa_backend.py",
)

MARKER_BACKEND = "# [patch] _sgl_dsa_torch_fallback_backend_"

# The import block this hooks onto was respelled in v0.5.17: the metadata base
# class moved out of dsa_indexer.py into its own module
# (dsa_indexer_metadata.DSAIndexerMetadata, replacing dsa_indexer.
# BaseIndexerMetadata). Only the neighbouring line changed, the anchor is still
# the top of the same import cluster and the injected import is identical.
# paged_mqa_logits_backend.py itself did NOT move, so the import we add is
# valid on both layouts. Both spellings must keep working (one ConfigMap,
# instances on different pinned images), hence replace_any.
_BACKEND_IMPORT_TAIL = """from sglang.srt.layers.attention.dsa.dsa_topk_backend import (
    DSATopKBackend,
    TopkTransformMethod,
)"""
_BACKEND_IMPORT_ADD = """
from sglang.srt.layers.attention.dsa.paged_mqa_logits_backend import (
    DSAPagedMQALogitsBackend,
)"""


def _backend_import_variant(head: str) -> tuple[str, str]:
    old = head + "\n" + _BACKEND_IMPORT_TAIL
    new = MARKER_BACKEND + "\n" + old + _BACKEND_IMPORT_ADD
    return old, new


BACKEND_IMPORT_VARIANTS = [
    # <= v0.5.16
    _backend_import_variant("from sglang.srt.layers.attention.dsa.dsa_indexer import BaseIndexerMetadata"),
    # >= v0.5.17
    _backend_import_variant("from sglang.srt.layers.attention.dsa.dsa_indexer_metadata import DSAIndexerMetadata"),
    # >= v0.5.20: the kpool rework put dsa_metadata_manager between the two lines
    # of the old anchor. Same cluster, same injected import.
    _backend_import_variant(
        "from sglang.srt.layers.attention.dsa.dsa_indexer_metadata import DSAIndexerMetadata\n"
        "from sglang.srt.layers.attention.dsa.dsa_metadata_manager import (\n"
        "    DSAMetadataManagementMixin,\n"
        ")"
    ),
]


def _backend_init_variant(ctor: str, arg_read: str) -> tuple[str, str]:
    """(old, new) for one spelling of the DSATopKBackend construction.

    `arg_read` is how THIS ref spells the server-arg read. It must match what
    dsa_indexer.py's Indexer.__init__ does on the same ref, see the comment in
    the injected body: the two resolves have to agree, so reading from a
    different source than the indexer would defeat the point.
    """
    old = ctor
    new = ctor + """
        # Independent resolve mirroring dsa_indexer.py's Indexer.__init__ (both must agree
        # on the backend so the eager metadata precompute here and the indexer's dispatch
        # don't disagree about whether DeepGEMM is being used).
        self.paged_mqa_logits_backend = DSAPagedMQALogitsBackend.resolve(
            {arg_read}
        )""".format(arg_read=arg_read)
    return old, new


BACKEND_INIT_VARIANTS = [
    # <= v0.5.18
    _backend_init_variant(
        """        self.dsa_topk_backend: DSATopKBackend = DSATopKBackend(
            model_runner.server_args.dsa_topk_backend
        )""",
        "model_runner.server_args.dsa_paged_mqa_logits_backend",
    ),
    # >= v0.5.19: #36313 (--speculative-dsa-topk-backend) gave DSATopKBackend a
    # resolve() classmethod that takes the model_runner, and server-arg reads moved
    # to the resolved exec context - which is also what the Indexer now uses, and
    # get_exec is already imported in this file on that ref.
    _backend_init_variant(
        """        self.dsa_topk_backend: DSATopKBackend = DSATopKBackend.resolve(model_runner)""",
        "get_exec().kernel.dsa_paged_mqa_logits_backend",
    ),
]
MARKER_BACKEND_INIT = "self.paged_mqa_logits_backend = DSAPagedMQALogitsBackend.resolve("


OLD_BACKEND_SITE_A = """        paged_mqa_schedule_metadata = None
        paged_mqa_ctx_lens_2d = None
        if is_cuda() and (
            forward_batch.forward_mode.is_decode_or_idle()
            or forward_batch.forward_mode.is_target_verify()
            or forward_batch.forward_mode.is_draft_extend_v2()
        ):
            paged_mqa_ctx_lens_2d = self._build_paged_mqa_schedule_2d_ctx_lens(
                forward_batch.forward_mode,
                cache_seqlens_int32,
                seqlens_expanded,
                forward_batch.batch_size,
            )
            # NOTE: block_kv arg must be 64 here — DG computes SPLIT_KV =
            # block_kv * 4 and both DG's and the indexer's compute kernels
            # require SPLIT_KV = 256; this is independent of the cache page size.
            paged_mqa_schedule_metadata = deep_gemm.get_paged_mqa_logits_metadata(
                paged_mqa_ctx_lens_2d, 64, deep_gemm.get_num_sms()
            )"""
NEW_BACKEND_SITE_A = """        paged_mqa_schedule_metadata = None
        paged_mqa_ctx_lens_2d = None
        if is_cuda() and (
            forward_batch.forward_mode.is_decode_or_idle()
            or forward_batch.forward_mode.is_target_verify()
            or forward_batch.forward_mode.is_draft_extend_v2()
        ):
            paged_mqa_ctx_lens_2d = self._build_paged_mqa_schedule_2d_ctx_lens(
                forward_batch.forward_mode,
                cache_seqlens_int32,
                seqlens_expanded,
                forward_batch.batch_size,
            )
            # ctx_lens_2d is still needed unconditionally (consumed downstream as
            # seqlens_32_2d regardless of logits backend); only the DeepGEMM schedule
            # metadata call is skipped for the torch backend, which discards it anyway
            # (dsalogitrework.md Section 2: `_ = deep_gemm_metadata` in the torch fn).
            if not self.paged_mqa_logits_backend.is_torch():
                # NOTE: block_kv arg must be 64 here — DG computes SPLIT_KV =
                # block_kv * 4 and both DG's and the indexer's compute kernels
                # require SPLIT_KV = 256; this is independent of the cache page size.
                paged_mqa_schedule_metadata = deep_gemm.get_paged_mqa_logits_metadata(
                    paged_mqa_ctx_lens_2d, 64, deep_gemm.get_num_sms()
                )"""
OLD_BACKEND_SITE_B = """        paged_mqa_schedule_metadata = None
        paged_mqa_ctx_lens_2d = None
        if is_cuda() and (
            forward_mode.is_decode_or_idle()
            or forward_mode.is_target_verify()
            or forward_mode.is_draft_extend_v2()
        ):
            paged_mqa_ctx_lens_2d = self._build_paged_mqa_schedule_2d_ctx_lens(
                forward_mode, cache_seqlens_int32, seqlens_expanded, bs
            )
            paged_mqa_schedule_metadata = deep_gemm.get_paged_mqa_logits_metadata(
                paged_mqa_ctx_lens_2d, 64, deep_gemm.get_num_sms()
            )"""
NEW_BACKEND_SITE_B = """        paged_mqa_schedule_metadata = None
        paged_mqa_ctx_lens_2d = None
        if is_cuda() and (
            forward_mode.is_decode_or_idle()
            or forward_mode.is_target_verify()
            or forward_mode.is_draft_extend_v2()
        ):
            paged_mqa_ctx_lens_2d = self._build_paged_mqa_schedule_2d_ctx_lens(
                forward_mode, cache_seqlens_int32, seqlens_expanded, bs
            )
            if not self.paged_mqa_logits_backend.is_torch():
                paged_mqa_schedule_metadata = deep_gemm.get_paged_mqa_logits_metadata(
                    paged_mqa_ctx_lens_2d, 64, deep_gemm.get_num_sms()
                )"""
OLD_BACKEND_REFRESH = """    def _refresh_paged_mqa_schedule_metadata(
        self,
        metadata: DSAMetadata,
        seqlens_32_2d: torch.Tensor,
    ) -> None:
        new_schedule = deep_gemm.get_paged_mqa_logits_metadata(
            seqlens_32_2d, 64, deep_gemm.get_num_sms()
        )
        if metadata.paged_mqa_schedule_metadata is None:
            object.__setattr__(metadata, "paged_mqa_schedule_metadata", new_schedule)
        else:
            metadata.paged_mqa_schedule_metadata.copy_(new_schedule)"""
NEW_BACKEND_REFRESH = """    def _refresh_paged_mqa_schedule_metadata(
        self,
        metadata: DSAMetadata,
        seqlens_32_2d: torch.Tensor,
    ) -> None:
        # Torch backend: schedule metadata is unused (discarded by the torch fn) and
        # was never allocated (init_forward_metadata skips it too) -> nothing to refresh.
        # This single helper covers BOTH cuda-graph-replay refresh call sites, so gating
        # it here is sufficient without touching each call site separately.
        if self.paged_mqa_logits_backend.is_torch():
            return
        new_schedule = deep_gemm.get_paged_mqa_logits_metadata(
            seqlens_32_2d, 64, deep_gemm.get_num_sms()
        )
        if metadata.paged_mqa_schedule_metadata is None:
            object.__setattr__(metadata, "paged_mqa_schedule_metadata", new_schedule)
        else:
            metadata.paged_mqa_schedule_metadata.copy_(new_schedule)"""


def _respell_resolve(pair: tuple[str, str]) -> tuple[str, str]:
    """(old, new) in the >= v0.5.23 spelling: DeepGEMM calls went through resolve_*()."""
    return (
        pair[0]
        .replace("deep_gemm.get_paged_mqa_logits_metadata(", "resolve_paged_mqa_logits_metadata_fn()(")
        .replace("deep_gemm.get_num_sms()", "resolve_num_sms()"),
        pair[1]
        .replace("deep_gemm.get_paged_mqa_logits_metadata(", "resolve_paged_mqa_logits_metadata_fn()(")
        .replace("deep_gemm.get_num_sms()", "resolve_num_sms()"),
    )


SITE_A_VARIANTS = [(OLD_BACKEND_SITE_A, NEW_BACKEND_SITE_A), _respell_resolve((OLD_BACKEND_SITE_A, NEW_BACKEND_SITE_A))]
SITE_B_VARIANTS = [(OLD_BACKEND_SITE_B, NEW_BACKEND_SITE_B), _respell_resolve((OLD_BACKEND_SITE_B, NEW_BACKEND_SITE_B))]
REFRESH_VARIANTS = [
    (OLD_BACKEND_REFRESH, NEW_BACKEND_REFRESH),
    _respell_resolve((OLD_BACKEND_REFRESH, NEW_BACKEND_REFRESH)),
]
MARKER_SITE_A = "# ctx_lens_2d is still needed unconditionally"
MARKER_SITE_B = "forward_mode, cache_seqlens_int32, seqlens_expanded, bs\n            )\n            if not self.paged_mqa_logits_backend.is_torch():"
MARKER_REFRESH = "# Torch backend: schedule metadata is unused"


@patch_dsa_backend.run
def apply_dsa_backend(p: Patch) -> None:
    p.replace_any(BACKEND_IMPORT_VARIANTS, marker=MARKER_BACKEND, what="4a-import")
    p.replace_any(BACKEND_INIT_VARIANTS, marker=MARKER_BACKEND_INIT, what="4b-init")
    p.replace_any(SITE_A_VARIANTS, marker=MARKER_SITE_A, what="4c-site-A")
    p.replace_any(SITE_B_VARIANTS, marker=MARKER_SITE_B, what="4d-site-B")
    p.replace_any(REFRESH_VARIANTS, marker=MARKER_REFRESH, what="4e-refresh-helper")


# ── 5) dsa_indexer.py: dispatch to the torch fallback (4 sub-edits, in source order) ──

patch_dsa_indexer = Patch(
    name="DSA torch-backend dispatch (dsa_indexer.py)",
    target="sglang/srt/layers/attention/dsa/dsa_indexer.py",
)

OLD_INDEXER_IMPORT = """from sglang.srt.layers.attention.dsa.paged_mqa_logits_backend import (
    DSAPagedMQALogitsBackend,
)"""
NEW_INDEXER_IMPORT = """# [patch] _sgl_dsa_torch_fallback_dispatch_
from sglang.srt.layers.attention.dsa.paged_mqa_logits_backend import (
    DSAPagedMQALogitsBackend,
)
from sglang.srt.layers.attention.dsa.torch_paged_mqa_logits import (
    fp8_paged_mqa_logits_torch_dsa,
)"""
OLD_USE_DG_NATIVE = """        use_dg_native = (
            not use_cute_dsl
            and _is_cuda
            and forward_batch.forward_mode.is_target_verify()
            and next_n >= 2
            and ctx_2d is not None
            and ctx_2d.shape == (B, next_n)
        )"""
NEW_USE_DG_NATIVE = """        use_dg_native = (
            not use_cute_dsl
            and not self.paged_mqa_logits_backend.is_torch()
            and _is_cuda
            and forward_batch.forward_mode.is_target_verify()
            and next_n >= 2
            and ctx_2d is not None
            and ctx_2d.shape == (B, next_n)
        )"""
OLD_FALLBACK_METADATA = """        if _is_cuda:
            if schedule_metadata is None:
                schedule_metadata = deep_gemm.get_paged_mqa_logits_metadata(
                    seqlens_32_2d, blocksize, self.sm_count
                )"""
NEW_FALLBACK_METADATA = """        if _is_cuda and not self.paged_mqa_logits_backend.is_torch():
            if schedule_metadata is None:
                schedule_metadata = deep_gemm.get_paged_mqa_logits_metadata(
                    seqlens_32_2d, blocksize, self.sm_count
                )"""
OLD_DISPATCH_BRANCH = """        elif use_dg_native:
            logits = deepgemm_paged_mqa_logits_native(
                deep_gemm.fp8_paged_mqa_logits,
                q_fp8,
                kv_cache_fp8,
                weights,
                seqlens_32_2d,
                block_tables,
                schedule_metadata,
                max_seq_len,
                q_offset=q_offset,
                B=B,
                next_n=next_n,
            )
        else:
            logits = deepgemm_paged_mqa_logits_split(
                deep_gemm.fp8_paged_mqa_logits,
                q_fp8,
                kv_cache_fp8,
                weights,
                seqlens_32_2d,
                block_tables,
                schedule_metadata,
                max_seq_len,
                q_offset=q_offset,
            )"""
NEW_DISPATCH_BRANCH = """        elif use_dg_native:
            logits = deepgemm_paged_mqa_logits_native(
                deep_gemm.fp8_paged_mqa_logits,
                q_fp8,
                kv_cache_fp8,
                weights,
                seqlens_32_2d,
                block_tables,
                schedule_metadata,
                max_seq_len,
                q_offset=q_offset,
                B=B,
                next_n=next_n,
            )
        elif self.paged_mqa_logits_backend.is_torch():
            # Phase 1+2 (dsalogitrework.md Section 3): the torch/triton kernel is
            # per-TOKEN (batch dim = flattened q tokens), and EVERY input here
            # already is: q/weights are sliced per token ([:q_offset]),
            # seqlens_32_2d is get_seqlens_expanded() = per-token [q_offset, 1]
            # for verify/draft-extend, and block_tables (= metadata
            # real_page_table) is ALREADY repeat_interleaved to per-token rows by
            # init_forward_metadata for ALL multi-token modes (target_verify /
            # draft_extend_v2 / draft_extend -- dsa_backend.py builds
            # `page_table = torch.repeat_interleave(page_table, ...)` before
            # storing the metadata; source-verified 2026-07-16). So next_n>=2
            # needs NO reshaping at all -- an extra repeat here DOUBLE-expands
            # and trips the kernel's shape assert (live crash at the MTP warmup,
            # first deploy attempt).
            logits = fp8_paged_mqa_logits_torch_dsa(
                q_fp8.unsqueeze(1)[:q_offset],
                kv_cache_fp8,
                weights[:q_offset],
                seqlens_32_2d,
                block_tables,
                None,  # schedule_metadata unused by the torch path
                max_seq_len,
                clean_logits=False,
            )
        else:
            logits = deepgemm_paged_mqa_logits_split(
                deep_gemm.fp8_paged_mqa_logits,
                q_fp8,
                kv_cache_fp8,
                weights,
                seqlens_32_2d,
                block_tables,
                schedule_metadata,
                max_seq_len,
                q_offset=q_offset,
            )"""


# The two deepgemm arms are handed the raw kernel by name, and that name was
# respelled in v0.5.20: the kpool rework wraps it in a local chunking helper
# (_chunked_fp8_paged_mqa_logits). Nothing else about the branch changed, and the
# arm this patch INSERTS does not call it at all, so one blanket substitution
# covers both spellings. Both must keep working (one ConfigMap, instances on
# different pinned images), hence replace_any.
_DISPATCH_KERNEL_SPELLINGS = (
    "deep_gemm.fp8_paged_mqa_logits",  # <= v0.5.19
    "_chunked_fp8_paged_mqa_logits",  # >= v0.5.20
)
DISPATCH_BRANCH_VARIANTS = [
    (
        OLD_DISPATCH_BRANCH.replace("deep_gemm.fp8_paged_mqa_logits", _fn),
        NEW_DISPATCH_BRANCH.replace("deep_gemm.fp8_paged_mqa_logits", _fn),
    )
    for _fn in _DISPATCH_KERNEL_SPELLINGS
]
MARKER_DISPATCH = "logits = fp8_paged_mqa_logits_torch_dsa("


FALLBACK_METADATA_VARIANTS = [
    (OLD_FALLBACK_METADATA, NEW_FALLBACK_METADATA),
    _respell_resolve((OLD_FALLBACK_METADATA, NEW_FALLBACK_METADATA)),
]
MARKER_FALLBACK_METADATA = (
    "if _is_cuda and not self.paged_mqa_logits_backend.is_torch():\n            if schedule_metadata is None:"
)


@patch_dsa_indexer.run
def apply_dsa_indexer(p: Patch) -> None:
    p.replace(OLD_INDEXER_IMPORT, NEW_INDEXER_IMPORT, what="5a-import")
    p.replace(OLD_USE_DG_NATIVE, NEW_USE_DG_NATIVE, what="5b-use-dg-native")
    p.replace_any(FALLBACK_METADATA_VARIANTS, marker=MARKER_FALLBACK_METADATA, what="5c-fallback-metadata")
    p.replace_any(DISPATCH_BRANCH_VARIANTS, marker=MARKER_DISPATCH, what="5d-dispatch-branch")
