# SGLang Upstream Bug: `Fp8MoEMethod` raises `AttributeError` for `flashinfer_cutlass` MoE backend

## Status

**Open upstream — no fix in flight for vanilla `Fp8MoEMethod` as of 2026-06-11.**
Originally verified on SGLang `v0.5.11` ("The Tenacity Release", tagged 2026-05-05)
with the upstream image `scitrera/dgx-spark-sglang:0.5.11` (FlashInfer 0.6.10,
sgl-kernel 0.4.2). Reproduced on Qwen3.6-35B-A3B-FP8, 4×GB10 (SM12.0a), TP=4,
during the matrix run on 2026-05-10 (case `07_fi_cutlass-moe_fi-attn` in
`kikube/matrixtest/2026-05-10/results/sglang_nn4_tp4_ep1/qwen-3.6-35b-a3b-fp8/0.5.11/`).
Confirmed still present in **v0.5.12.post1** (the current default image
`xomoxcc/dgx-spark-sglang:0.5.12.post1-sm121`) by source inspection. Release
notes contain no `Fp8MoEMethod` + flashinfer_cutlass fix; PR #21872 still open
and unmerged (re-verified 2026-06-11, last update 2026-04-01); PR #22627 merged
2026-05-26 in v0.5.12.post1 but only patches `ModelOptFp8MoEMethod` — does NOT
fix the vanilla `Fp8MoEMethod` + `flashinfer_cutlass` bug described here. The
MoE-refactor series migrated related backends to `MoeRunner`: **PR #25525**
(merged 2026-05-17) migrated `flashinfer_cutedsl` + DeepEP to MoeRunner;
**PR #26489** ("[MoE Refactor] Migrate SM90 Cutlass W4A16 to MoeRunner", merged
2026-05-30) covers `flashinfer_mxfp4` (SM90 MXFP4). Both left `flashinfer_cutlass`
+ vanilla `Fp8MoEMethod` untouched — the `# TODO(cwan): refactor other backends`
comment in `fp8.py` is still there as of 0.5.12.post1. Issue #20719 remains open
(re-verified 2026-06-11).

The bug is plainly visible in the source — `Fp8MoEMethod.create_moe_runner`
ends with an explicit `# TODO(cwan): refactor other backends` for everything
that is not triton/aiter/deep_gemm/fi_trtllm.

**Update 2026-06-12:** A third party independently filed the issue and a
partial-fix PR on 2026-06-11:
[Issue #27951](https://github.com/sgl-project/sglang/issues/27951)
("`[Bug] --moe-runner-backend flashinfer_cutlass + FP8 weights crashes with
AttributeError: 'Fp8MoEMethod' object has no attribute 'runner'`", filed by
`gujialiang123`) and
[PR #27968](https://github.com/sgl-project/sglang/pull/27968)
("`fix(fp8/moe): raise clear error for unsupported MoE runner backend`", filed
by `Anai-Guo`, closes #27951). **PR #27968 does NOT implement
`flashinfer_cutlass` support for `Fp8MoEMethod`** — it raises a clean
`ValueError` for unsupported backends instead of the opaque `AttributeError`.
The SM12.0a hardware constraint (no SM90 CUTLASS FP8-block kernel) is
unchanged; the workaround (`moe_runner_backend: triton`) remains the correct
cluster configuration.

Adjacent open work:

- [PR #21872](https://github.com/sgl-project/sglang/pull/21872)
  ("Add FlashInfer CUTLASS fused MoE support for FP8 block-quantized models on SM90") —
  open since 2026-04-01, stagnant since 2026-04-01. Patches a **different
  class** (`CompressedTensorsW8A8Fp8MoEMethod` in
  `compressed_tensors/schemes/compressed_tensors_w8a8_fp8_moe.py`), not the
  vanilla `Fp8MoEMethod` we hit. Explicitly **SM90 only**.
- [PR #22627](https://github.com/sgl-project/sglang/pull/22627)
  ("Fix flashinfer_cutlass MoE crash when intermediate_size_per_partition is
  not 16-aligned") — **merged 2026-05-26 in v0.5.12.post1**. Patches
  `ModelOptFp8MoEMethod` in `modelopt_quant.py` and addresses a different
  symptom (alignment, not missing `runner`). Does **not** touch vanilla
  `Fp8MoEMethod`; the bug documented here is unaffected.
- [Issue #20719](https://github.com/sgl-project/sglang/issues/20719)
  ("CompressedTensorsW4A4Nvfp4MoE bypasses MoeRunner, hardcodes kernel
  dispatch in apply_weights") — open since 2026-03-16. Diagnostic write-up
  of the same anti-pattern (quant method bypassing `MoeRunner`); names
  `Fp8MoEMethod` as the *correct* reference path, not as bug. No fix.

None of these touch `quantization/fp8.py::Fp8MoEMethod`, which is the path
Qwen3.6-35B-A3B-FP8 (and every plain `--quantization fp8` MoE model) takes.

## Affected Configuration

- Quantization method: vanilla `Fp8MoEMethod`
  (`sglang/srt/layers/quantization/fp8.py`) — i.e. any FP8 MoE model whose HF
  config carries `quantization_config.quant_method == "fp8"` and is loaded
  through the standard `Fp8Config.get_quant_method` dispatch.
  Not affected: `CompressedTensorsW8A8Fp8MoEMethod` (PR #21872 territory),
  `ModelOptFp8MoEMethod` (PR #22627 territory).
- MoE runner backend: `--moe-runner-backend flashinfer_cutlass`
  (or the new `flashinfer_cutedsl` from PR #21339, same code-path gap).
- Tested models that hit this on our cluster:
  - `Qwen/Qwen3.6-35B-A3B-FP8` (256 experts, hybrid GDN, native 262K)
  - sibling FP8 MoE profiles in `roles/k8s_dgx/model_profiles/`:
    `qwen-qwen3-coder-30b-a3b-instruct-fp8`,
    `qwen-qwen3-next-80b-a3b-instruct-fp8`,
    `qwen-qwen3.5-35b-a3b-fp8`, `qwen-qwen3.5-122b-a10b-fp8`,
    `unsloth-qwen3-235b-a22b-instruct-2507-fp8`
    (all use vanilla `Fp8MoEMethod` and would crash identically).
- Hardware: confirmed on SM12.0a (NVIDIA GB10 / DGX Spark). Bug is
  hardware-independent at the Python level — would crash the same way on
  H100/H200/B200 if the user passed `--moe-runner-backend flashinfer_cutlass`
  with a vanilla FP8 MoE model. (See "Why this is also moot on SM120" for
  the deeper hardware constraint that makes any port to `Fp8MoEMethod` of
  questionable value on Blackwell-Edge.)

## The Bug

`Fp8MoEMethod.create_moe_runner`
(`sglang/srt/layers/quantization/fp8.py`, lines 1514–1545 on v0.5.11)
sets `self.runner` only for a closed allowlist of backends:

```python
def create_moe_runner(
    self, layer: torch.nn.Module, moe_runner_config: MoeRunnerConfig
):
    self.moe_runner_config = moe_runner_config
    moe_runner_backend = get_moe_runner_backend()

    if moe_runner_backend.is_auto():
        if self.is_deepgemm_moe_runner_backend_enabled():
            moe_runner_backend = MoeRunnerBackend.DEEP_GEMM
        elif (
            _is_hip
            and (_use_aiter or _use_hip_int4)
            and get_moe_a2a_backend().is_none()
        ):
            moe_runner_backend = MoeRunnerBackend.AITER
        else:
            moe_runner_backend = MoeRunnerBackend.TRITON

    if (
        moe_runner_backend.is_deep_gemm()
        or moe_runner_backend.is_triton()
        or moe_runner_backend.is_aiter()
        or moe_runner_backend.is_flashinfer_trtllm()
        or moe_runner_backend.is_flashinfer_trtllm_routed()
    ):
        self.runner = MoeRunner(moe_runner_backend, moe_runner_config)
    else:
        # TODO(cwan): refactor other backends
        pass
```

For `moe_runner_backend == flashinfer_cutlass` (and `flashinfer_cutedsl`)
the `else` branch is taken and `self.runner` is **never assigned**.

`Fp8MoEMethod.apply` (same file, lines 1566–1756) then unconditionally
dereferences `self.runner` from line 1605 onwards:

```python
and self.runner.runner_backend.is_aiter()      # 1605
...
return self.runner.run(dispatch_output, quant_info)  # 1612
...
if self.runner.runner_backend.is_deep_gemm():        # 1652
...
self.runner.runner_backend.is_flashinfer_trtllm()    # 1690
or self.runner.runner_backend.is_flashinfer_trtllm_routed()  # 1691
...
elif self.runner.runner_backend.is_triton():         # 1749
"Unsupported runner backend: %s" % self.runner.runner_backend  # 1753
return self.runner.run(dispatch_output, quant_info)  # 1756
```

The first dereference (line 1605, hit unconditionally on the standard
non-AITER path) raises `AttributeError`.

## Symptom

Server crashes deterministically during `Scheduler.__init__` →
`init_tp_model_worker` → `ModelRunner.__init__` →
`init_device_graphs` → CUDA-graph capture → first forward pass:

```
File "/usr/local/lib/python3.12/dist-packages/sglang/srt/managers/scheduler.py", line 432, in __init__
  self.init_model_worker()
...
File "/usr/local/lib/python3.12/dist-packages/sglang/srt/model_executor/cuda_graph_runner.py", line 884, in _capture_one_stream
  ) = self.capture_one_batch_size(bs, forward, stream_idx)
...
File "/usr/local/lib/python3.12/dist-packages/sglang/srt/models/qwen3_5.py", line 640, in forward
  hidden_states = self.mlp(...)
...
  return self.quant_method.apply(...)
File "/usr/local/lib/python3.12/dist-packages/sglang/srt/layers/quantization/fp8.py", line 1652, in apply
  if self.runner.runner_backend.is_deep_gemm():
AttributeError: 'Fp8MoEMethod' object has no attribute 'runner'
```

(Trace excerpt is from line 1652 because CUDA-graph capture took the
deep_gemm dispatch branch first; on `--disable-cuda-graph` it would
crash on line 1605 instead. Both lines fail for the same reason.)

The crash is reported by `kikube-bench matrix` as
`outcome=startup_crash` with the head pod restart-looping until the
matrix driver scales the deployment to 0 and moves on to the next case.

## Reproducer

Standard 4-node setup, vanilla upstream image, the FP8 model and the
flag in question — no other special config required:

```bash
docker run --rm --gpus all scitrera/dgx-spark-sglang:0.5.11 \
  python3 -m sglang.launch_server \
    --model-path Qwen/Qwen3.6-35B-A3B-FP8 \
    --tp-size 4 --pp-size 1 --nnodes 4 --node-rank 0 \
    --nccl-init-addr <head>:50000 \
    --kv-cache-dtype fp8_e4m3 \
    --mem-fraction-static 0.50 \
    --attention-backend flashinfer \
    --fp8-gemm-backend cutlass \
    --moe-runner-backend flashinfer_cutlass \
    --mamba-scheduler-strategy extra_buffer
```

`--moe-runner-backend flashinfer_cutlass` is the only line that matters —
remove it (or set it to `triton`) and the same model launches and serves
fine. This was reproduced on 0.5.10 and 0.5.10.post1 in earlier matrix
sweeps; verified again on 0.5.11 in the 2026-05-10 run.

## Why this is also moot on SM12.0a (GB10 / DGX Spark)

Even after a hypothetical port of PR #21872 from
`CompressedTensorsW8A8Fp8MoEMethod` to vanilla `Fp8MoEMethod`, the
underlying kernel call would be:

```python
flashinfer.fused_moe.cutlass_fused_moe(use_deepseek_fp8_block_scale=True, ...)
```

That code-path in FlashInfer 0.6.x is a **SM90 CUTLASS kernel**
(see PR #21872 description: "Requirements: SM90 GPU (H100/H800)").
The CUTLASS templates for FP8-block fused MoE have not been ported to
SM12.0a — there is no kernel for our hardware to dispatch to even if
the Python-level dispatch were fixed. So for the GB10 cluster the only
correct outcome of this bug is "remain on `triton`".

PR #21872's own benchmark numbers also argue against pushing this on
H100, in the CUDA-graph case that we (and most production setups) run:

| Mode          | Triton | FlashInfer CUTLASS | Δ        |
| ------------- | -----: | -----------------: | -------: |
| No CUDA graph | 152    | 163                | **+7.4 %** |
| CUDA graph    | 869    | 749                | **−13.8 %** |

i.e. fi_cutlass FP8 only beats triton in eager mode, where overall
throughput is so far below the CUDA-graph path that the relative gain
doesn't matter (compare our case-02 eager run at 22 tok/s vs case-01
CUDA-graph run at 76 tok/s on the same model).

## Workaround

Pin `moe_runner_backend: triton` in the per-model SGLang profile.
For our cluster this is already the default in
`roles/k8s_dgx/model_profiles/qwen-qwen3.6-35b-a3b-fp8.yml` and the
sibling FP8 MoE profiles. The matrix sweep deliberately probes
`flashinfer_cutlass` to (re-)confirm the upstream gap.

For users who want to test other FP8 MoE models on this code-path,
the safe fallback is:

```yaml
moe_runner_backend: triton          # vanilla Fp8MoEMethod path that works
# moe_runner_backend: flashinfer_cutlass   # AttributeError, see this doc
# moe_runner_backend: flashinfer_cutedsl   # AttributeError (same code-path gap)
```

## Action items

1. ~~**File upstream issue** in `sgl-project/sglang` with the stack trace,
   reproducer command, and a pointer to the
   `# TODO(cwan): refactor other backends` line in `fp8.py` and to
   PRs #21872 / #22627 / Issue #20719 as the existing partial work.
   Title proposal: "`Fp8MoEMethod.create_moe_runner` does not handle
   `flashinfer_cutlass` / `flashinfer_cutedsl` → `AttributeError:
   'Fp8MoEMethod' object has no attribute 'runner'` at first forward".~~
   **Moot as of 2026-06-12** — Issue #27951 and PR #27968 filed independently
   by third parties on 2026-06-11. See Status section update above.
2. **Keep cluster profile pinned to `triton`** for all vanilla-FP8
   MoE models. Document in `CLAUDE.md` once filed.
3. **Stop re-running the fi_cutlass cases (07–12) of the v0.5.11
   correctness/throughput matrix for FP8 MoE models** until either
   (a) upstream lands a port of #21872 to `Fp8MoEMethod` *and*
   (b) FlashInfer ships an SM12.0a CUTLASS FP8-block fused-MoE
   kernel. Both are independent prerequisites; neither is in flight.
4. **Re-evaluate `flashinfer_cutedsl` (cases 15–20)** independently —
   that backend has the same `create_moe_runner` gap on FP8, but the
   underlying kernel (PR #21339) targets a different SM range and may
   matter for SM12.0a once the dispatch is fixed.
