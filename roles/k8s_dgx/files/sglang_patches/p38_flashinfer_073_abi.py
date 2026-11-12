"""[dgxarley] SGLang v0.5.23 call sites vs. the flashinfer 0.7.3 ABI.

The 0.5.23-sm121 image ships flashinfer 0.7.3 while SGLang v0.5.23 pins 0.7.2.
Two call sites still use the 0.7.2 signatures:

  1. speculative/sampling.cuh: DeviceSamplingFromProb() gained an
     `int& last_valid_index` parameter. The JIT module
     sgl_kernel_jit_speculative_sampling_* (EAGLE verify, first non-greedy
     request) fails to compile, the scheduler gets SIGQUIT and the whole TP
     group restarts (exit 137) on every such request.
  2. flashinfer_backend.py fast_prefill_plan(): plan() gained a trailing
     `fixed_uniform_q_len` argument; the hand-built arg list is one short
     (TypeError on the first call, same failure class as p36).

Both hunks are upstream sgl-project/sglang dc4b17b108 (#48468, flashinfer 0.7.3
bump). Each is gated on the installed flashinfer carrying the new signature, so
the patch is inert on images with flashinfer <= 0.7.2.

Re-sync: drop this file once the image ships SGLang >= the release containing
dc4b17b108 (the already-applied guards make it a no-op until then).
"""

from _patchlib import Patch, target_contains

sampling = Patch(
    name="DeviceSamplingFromProb last_valid_index (flashinfer 0.7.3)",
    target="sglang/kernels/jit/include/sgl_kernel/speculative/sampling.cuh",
    when=target_contains("flashinfer/data/include/flashinfer/sampling.cuh", "int& last_valid_index,"),
)

DECL_OLD = """  DType aggregate_relu_q_minus_p(0);
  for (uint32_t i = 0; i < ceil_div(d, BLOCK_THREADS * VEC_SIZE); ++i) {"""
DECL_NEW = """  DType aggregate_relu_q_minus_p(0);
  int last_valid_index = -1;
  for (uint32_t i = 0; i < ceil_div(d, BLOCK_THREADS * VEC_SIZE); ++i) {"""

CALL_OLD = (
    """        i, d, [&](DType x) { return x > 0; }, u, relu_q_minus_p_vec, aggregate_relu_q_minus_p, &temp_storage);"""
)
CALL_NEW = """        i, d, [&](DType x) { return x > 0; }, u, relu_q_minus_p_vec, aggregate_relu_q_minus_p, last_valid_index,
        &temp_storage);"""


@sampling.run
def apply_sampling(p: Patch) -> None:
    p.replace(DECL_OLD, DECL_NEW, what="last_valid_index declaration")
    p.replace(CALL_OLD, CALL_NEW, what="DeviceSamplingFromProb call")


plan = Patch(
    name="fast_prefill_plan fixed_uniform_q_len (flashinfer 0.7.3)",
    target="sglang/srt/layers/attention/flashinfer_backend.py",
    when=target_contains("flashinfer/prefill.py", "args.append(False)  # fixed_uniform_q_len"),
)

PLAN_OLD = """        0,  # uniform_q_len
    ]
    self._plan_info = self._cached_module.plan(*args)"""
PLAN_NEW = """        0,  # uniform_q_len
        False,  # fixed_uniform_q_len
    ]
    self._plan_info = self._cached_module.plan(*args)"""


@plan.run
def apply_plan(p: Patch) -> None:
    p.replace(PLAN_OLD, PLAN_NEW, what="fast_prefill_plan tail arg")
