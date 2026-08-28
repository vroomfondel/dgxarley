"""[dgxarley] Qwen Sparse Attention (QSA) on GB10/SM121: give sparse DECODE a
kernel path that BOOTS, and keep it off the one that silently corrupts.

>>> READ THE EVIDENCE SECTION BEFORE TOUCHING EITHER EDIT. The obvious "fix"
>>> for the boot crash (widening the trtllm gate to SM12x) is measured to
>>> return exclamation marks instead of text at agent-scale context, without
>>> logging anything. We got this wrong once already, on 2026-08-27, and only
>>> the upstream thread caught it.

TWO SEPARATE DEFECTS, PULLING IN OPPOSITE DIRECTIONS

  (1) BOOT CRASH. QwenSparseAttnBackend cannot resolve a decode kernel on
      family-12 devices without the flashinfer trtllm path, so it falls through
      to the pip `flash_attn.cute.interface`, which fails to compile on GB10:

        cutlass MLIRError: Operation creation failed:
        error: unknown: expects `coord` and shape of view are weakly congruent,
        but got '!cute.layout<"(?,?):(?{i64 div=8},1)">', '!cute.coord<"(_,_,?)">'

      Reached from three call sites (decode CUDA-graph capture, the MTP draft
      via qwen4_exp_mtp, eager decode via hybrid_linear_attn_backend), so no
      launch flag avoids it, and `--attention-backend triton` does NOT help
      because QSA calls flash_attn directly. Root cause is upstream in the
      stock `flash_attn_4-4.0.0b19` wheel: it ships with the TMA-O varlen
      guard commented out (sgl-project/sglang#36716 bug 3, with wheel-vs-image
      md5 forensics). Upstream issues: #36558 (SM121, our hardware), #36531
      (SM120). Fix: #36556 hunk 2 = EDIT 2 below.

  (2) SILENT LONG-CONTEXT CORRUPTION IF trtllm IS ENABLED ON SM121. trtllm-gen
      attention kernels are SM100-only; on sm_121 the kernel RUNS and emits NaN
      logits (#36716). Measured on 2x DGX Spark GB10 TP=2 with real weights and
      exactly our checkpoint (RadixArk/Qwen3.8-Flash-Next-NVFP4), single-shot
      cold prefill, greedy, thinking off, verdict = a run of >=50 consecutive
      token-0 ("!") in the output:

        prompt tokens   trtllm forced ON      trtllm off + repaired fallback
             120k       1/4 corrupt           6/6 clean
             160k       1/4 corrupt           -
             190k       2/4 corrupt           6/6 clean
             210k       4/4 corrupt           6/6 clean
             240k       -                     6/6 clean (chunk 1024, mem 0.82)

      Three properties make this worse than a normal wrong-output bug, and are
      the reason this patch carries a DEFENSIVE edit rather than trusting the
      shipped gate: it is SILENT (well-formed response, finish_reason length,
      nothing in the log); it is STOCHASTIC (about half the time at 190k, so
      any n=1 probe reads as a pass); and a liveness canary cannot see it (three
      short greedy prompts passed 3/3 while the same server corrupted 100% of
      210k requests). The reporter also notes the engine does not recover:
      once it fires, later requests fail at any depth, /flush_cache does not
      clear it, only a restart does.
      SM120 (RTX Blackwell, cc 12.0) is the OPPOSITE case: there the trtllm
      path is REQUIRED for correctness (exact needle retrieval at 229.5K),
      while the varlen fallback is correct only within a single KV page. So
      this is genuinely per-SKU, not "family 12".

UPSTREAM STATE AS OF 2026-08-28 (branch qwen4-main-squashed @ 99c9362e, which
is what scripts/patches/sglang-qwen4exp-pr36497.patch pins):
  * Defect (2) is CORRECTLY handled: commit 99c9362e (#36806) narrowed the gate
    to `is_sm100_supported() or is_sm120()`, where `is_sm120()` is EXACTLY
    (12, 0), with an in-source comment saying widening it "silently corrupts
    long-context decode on SM121/GB10". Note the branch passed through the
    wrong state first: #36649 had set `is_sm100_supported() or is_sm121()`,
    i.e. exactly inverted, before the corruption evidence landed.
  * Defect (1) is NOT handled: the varlen fallback still imports the pip cute
    interface, so GB10 still crashes at backend init. #36556 is still open.

WHAT THIS PATCH DOES

  EDIT 1 (DEFENSIVE, no upstream equivalent): an unconditional SM121 veto at
    the top of `_resolve_trtllm_sparse_decode`, injected just before the
    flashinfer import rather than by rewriting the gate expression. Deliberate:
    the gate expression has been respelled three times in three days
    (`is_sm100_supported()` -> `... or is_sm121()` -> `... or is_sm120()`), and
    #36556 as currently written would respell it a fourth time to
    `is_sm120_supported()` (= major 12), which re-includes GB10 and re-opens
    defect (2). A veto placed AFTER whatever gate the image happens to ship is
    spelling-independent and survives all of that. It costs SM100/SM120 nothing
    (is_sm121() is False there) and it is idempotent.
    KEEP THIS EVEN IF #36556 MERGES. Delete it only when the shipped gate
    provably excludes cc (12, 1) AND the NaN-logit mechanism in #36716 is
    fixed.

  EDIT 2 (upstream #36556 hunk 2, byte-identical): route family-12 devices to
    SGLang's own FA4 dispatcher (sglang.kernels.ops.attention.flash_attention_v4)
    instead of the incompatible pip `flash_attn.cute.interface`. This is the
    half that makes the engine boot on GB10. Verified upstream on GB10 as a
    dummy-weight e2e A/B on the day-0 image: base = the MLIRError above,
    reproducible and deterministic; + this hunk = autotune completes, decode
    CUDA graphs capture, server serves, and test/registered/kernels/test_qsa.py
    goes 2F/53P -> 55P on the same box.
    Kept byte-identical on purpose: if #36556 merges, the probe below matches
    and this edit reports "already applied" instead of drifting.

  What EDIT 2 is NOT: validated for long-context OUTPUT on SM121. The clean
    120k-240k column in the table above was produced with a DIFFERENT varlen
    fallback (a Triton one from MiaAI-Lab/Qwen3.8-Flash-Next-Dual-DGX-Sparks),
    not with sglang's flash_attention_v4. So the evidence supports the
    DIRECTION of edit 2, not this exact kernel. FIRST REAL RUN MUST CHECK
    OUTPUT AT DEPTH, not just that the server answers: prompt at >=190k and
    grep the completion for runs of "!". A short smoke test is worthless here,
    see the liveness-canary note above.

GATE: target_contains on the resolver, not gate_model. The subject file only
exists in an image built with sglang-qwen4exp-pr36497.patch; `Patch` reports a
missing target as ANCHOR-DRIFT, which would make the drift report useless on
every other image in this cluster. A False gate logs one honest
"gate not matched" line instead.

RE-SYNC on image bump: re-read `_resolve_trtllm_sparse_decode` in the new image
FIRST and confirm cc (12, 1) is excluded, then re-check whether #36556 (or a
successor) landed hunk 2. Do not assume either.
"""

from _patchlib import Patch, target_contains

TARGET = "sglang/srt/layers/attention/qwen_sparse_attn_backend.py"

MARKER = "# [patch] _sgl_qsa_sm121_trtllm_veto"

patch_qsa = Patch(
    name="QSA decode: veto trtllm on SM121, repair the varlen fallback",
    target=TARGET,
    when=target_contains(TARGET, "def _resolve_trtllm_sparse_decode"),
)


@patch_qsa.run
def apply_qsa(p: Patch) -> None:
    # EDIT 1 -- the safety edit. Anchored on the flashinfer import block, which
    # has been stable across every spelling of the gate above it, so the veto
    # lands after whatever gate the image ships and cannot be bypassed by an
    # upstream respelling. See defect (2) in the docstring.
    p.replace(
        """    try:
        from flashinfer.decode import trtllm_batch_decode_with_kv_cache
    except ImportError:
        return None
    return trtllm_batch_decode_with_kv_cache
""",
        f"""    {MARKER}
    # trtllm-gen attention kernels are SM100-only. On sm_121 (GB10) the kernel
    # RUNS and emits NaN logits (upstream #36716), which surfaces as runs of
    # token id 0 ("!") past the sparse-selection boundary -- silently, and
    # stochastically, so no smoke test catches it. Measured 4/4 corrupt at 210k
    # prompt tokens on 2x Spark TP=2 with real weights. Veto placed AFTER the
    # shipped gate on purpose: that expression has been respelled three times
    # in three days and #36556 would respell it again to include major-12.
    # Costs SM100/SM120 nothing. See the patch docstring before removing.
    from sglang.srt.utils import is_sm121

    if is_sm121():
        return None
    try:
        from flashinfer.decode import trtllm_batch_decode_with_kv_cache
    except ImportError:
        return None
    return trtllm_batch_decode_with_kv_cache
""",
        marker=MARKER,
        what="SM121 veto on the trtllm sparse-decode path",
    )

    # EDIT 2 -- upstream #36556 hunk 2, byte-identical (hence the probe on the
    # injected import): without it QSA cannot resolve ANY decode kernel on GB10
    # and the pod dies at backend init in the FA4 CuTe epilogue.
    p.replace(
        '''    """The dense varlen kernel behind the packed sparse-decode fallback.

    Classic flash_attn (FA2, Ampere/Hopper) is preferred when installed;
    flash-attn-4's cute interface serves the same call shape on Blackwell.
    """
    try:
        from flash_attn import flash_attn_varlen_func
''',
        '''    """The dense varlen kernel behind the packed sparse-decode fallback.

    SM120 uses SGLang's architecture-owned FA4 dispatcher. Other platforms
    prefer classic flash_attn (FA2) before flash-attn-4's cute interface.
    """
    from sglang.srt.utils import is_sm120_supported

    if is_sm120_supported():
        from sglang.kernels.ops.attention.flash_attention_v4 import (
            flash_attn_varlen_func,
        )

        return flash_attn_varlen_func
    try:
        from flash_attn import flash_attn_varlen_func
''',
        marker="from sglang.kernels.ops.attention.flash_attention_v4 import (",
        what="varlen fallback -> SGLang FA4 dispatcher on family-12",
    )
