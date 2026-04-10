# PRD: CUTLASS NVFP4 SM121 Shared Memory Fix

## Problem

NVFP4 MoE on DGX Spark (GB10 / SM121) has historically shown two distinct crash modes on SGLang. As of 2026-04-10, Crash B appears resolved upstream, Crash A remains. NVFP4 MoE models affected: GLM-4.7-NVFP4, MiniMax-M2.5-NVFP4, Qwen3.5-397B-NVFP4, Qwen3-235B-NVFP4.

**Crash A — `triton` MoE runner (device-side assert):**
```
RuntimeError: Runtime check failed at nvfp4_blockwise_moe.cuh:78: CUDA error: device-side assert triggered
```
Occurs during both CUDA graph capture AND eager inference. Root cause: shared memory overflow in CUTLASS grouped GEMM kernel.

**Crash B — `flashinfer_cutlass` MoE runner (Xid 13 Illegal Instruction):** *(LIKELY FIXED in FlashInfer 0.6.7.post3, see update below)*
```
NVRM: Xid (PCI:000f:01:00): 13, Graphics SM Warp Exception: Illegal Instruction Parameter
ESR 0x1c81fb60:0x1174  (consistent across all crashes)
Fatal Python error: Aborted
  File "flashinfer/fused_moe/core.py", line 490 in cutlass_fused_moe
```
Occurred on v0.5.10 early builds. Original root cause suspected: FlashInfer CUTLASS MoE kernel regression between rc0 and 0.5.10. **Status update (2026-04-10):** The current `scitrera/dgx-spark-sglang:0.5.10` image ships FlashInfer `0.6.7.post3`, which includes two upstream fixes that address the SM121 Xid 13 symptom:
- [flashinfer#2798](https://github.com/flashinfer-ai/flashinfer/pull/2798) (merged 2026-03-19): CUTLASS 4.2.1 → 4.4.2 upgrade — fixes TMA descriptor bug on `tma_warp_specialized_generic_moe_gemm_kernelLauncher<Sm120, fp4>` (directly addresses [flashinfer#2776](https://github.com/flashinfer-ai/flashinfer/issues/2776) "NVFP4 MoE crash on GB10 during CUDA graph capture").
- [flashinfer#2913](https://github.com/flashinfer-ai/flashinfer/pull/2913) (merged 2026-04-01): GDC (Grid Dependency Control) flag fix for CUTLASS fused MoE kernel — fixes `cudaErrorIllegalInstruction` race on SM90/SM100/SM120/SM121 where host-side PDL activation raced against device-side GDC barrier no-ops.

**Verification:** Qwen3-235B-A22B-NVFP4 TP=4 EP=4 on v0.5.10 has 4 stable configurations with `flashinfer_cutlass` MoE + `flashinfer_cutlass` fp4_gemm (tests 13, 14, 16, 17 in `TESTLOGS/sglang_nn4_tp4_ep4/qwen-3-235b-a22b-nvfp4/TESTLOG_nv580.142_sglang-0.5.10_qwen3-235b-a22b-nvfp4_4n.md`), peaking at 42.70 tok/s @ n=8 (test 17). No Xid 13 observed. GLM-4.7 test 14 on v0.5.10 did show a single `bench_crash` during n=1 inference, but this is not the original systematic Xid 13 pattern — likely a separate model-specific issue.

**Combined impact:** `flashinfer_cutlass` MoE runner is now the **known-good** NVFP4 MoE path on v0.5.10 for SM121 (model-dependent). The `cutlass_moe_fp4` path (reached via `moe_runner_backend=triton` or `cutlass`) remains broken by Crash A and has no upstream fix.

## Root Cause Analysis

### Crash A: shared memory overflow in `nvfp4_blockwise_moe.cuh`

**File location (v0.5.10):** `python/sglang/jit_kernel/csrc/moe/nvfp4_blockwise_moe.cuh` inside the SGLang main package (moved from `sgl-kernel/csrc/moe/` via [#20012](https://github.com/sgl-project/sglang/pull/20012)). JIT-compiled via nvrtc at runtime.

The `run_fp4_blockwise_scaled_group_mm_sm120()` kernel path uses:

```cpp
using ArchTag = cutlass::arch::Sm120;
using ThreadBlockShape = Shape<_128, _128, _128>;
using StageCountType = cutlass::gemm::collective::StageCountAuto;
// In the CollectiveBuilder:
cutlass::gemm::collective::StageCountAutoCarveout<
    static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>,
cutlass::gemm::KernelPtrArrayTmaWarpSpecializedPingpong>::CollectiveOp;
```

`StageCountAutoCarveout` + `Pingpong` schedule double-buffers the tiles and requests exactly **102400 bytes (100 KiB)** of shared memory at kernel launch, **1 KiB over** the 101376-byte (99 KiB) device budget on any SM12x Blackwell. See [CUTLASS#3144](https://github.com/NVIDIA/cutlass/issues/3144) for the full root cause — including the verbatim failure (`102400 bytes required but device supports 101376`) and NVIDIA maintainer @depaulmillz's correction on which SM versions actually have 228 KiB.

| GPU | SM Version | Shared Memory per Block |
|-----|-----------|------------------------|
| B200/B100 (datacenter) | SM100 | **228 KiB** |
| RTX 5090, RTX PRO 6000 Blackwell (consumer/workstation) | SM120 | **99 KiB** |
| **DGX Spark GB10** | **SM121** | **99 KiB** |

**Key correction:** both SM120 and SM121 share the same 99 KiB budget — only SM100 (B200) has 228 KiB. The "228 KiB on Blackwell" figure that circulates in some documentation refers to SM100 specifically, not SM12x. This means the bug is **not SM121-specific**; it affects every SM120/SM121 device that tries to run the current SGLang NVFP4 MoE path. Upstream SGLang presumably tested on SM100 (B200) where the extra 129 KiB absorbs the 1 KiB overshoot; on any SM12x hardware the kernel launch triggers a device-side assert. All subsequent CUDA calls (including the `cudaMallocAsync` near line 78) return `cudaErrorAssert` (sticky error).

### Crash B: FlashInfer Xid 13 — likely resolved by FlashInfer 0.6.7.post3

**Initial hypothesis (2026-04-08):** FlashInfer regression between rc0 and 0.5.10.

Comparison of `scitrera/cuda-containers` recipes:

| Aspect | rc0 recipe | 0.5.10 recipe |
|--------|-----------|---------------|
| Base image | `nvcr.io/nvidia/pytorch:26.02-py3` | `scitrera/dgx-spark-pytorch-dev:2.11.0-v1-cu132` |
| FlashInfer version | unset (bundled with SGLang, likely 0.6.5/0.6.6) | `0.6.7.post3` (explicit pin) |
| sgl-kernel ref | v0.5.10rc0 | v0.5.10 |
| Transformers | unset | `5.5.0` |

**Revised (2026-04-10):** FlashInfer 0.6.7 (released 2026-03-25) introduced two SM121-targeted fixes that were NOT in 0.6.5/0.6.6 which rc0 likely bundled:
1. **CUTLASS 4.4.2 bump** ([flashinfer#2798](https://github.com/flashinfer-ai/flashinfer/pull/2798)) — fixes TMA descriptor OOB address generation mode that caused non-deterministic crashes in `tma_warp_specialized_generic_moe_gemm_kernelLauncher<Sm120, fp4>`. Directly addresses [flashinfer#2776](https://github.com/flashinfer-ai/flashinfer/issues/2776).
2. **GDC flag fix** ([flashinfer#2913](https://github.com/flashinfer-ai/flashinfer/pull/2913), in 0.6.7.post1+) — fixes `cudaErrorIllegalInstruction` on SM90/SM100/SM120/SM121 caused by host activating PDL while device GDC barriers were no-ops → race condition → Xid 13.

Both fixes are present in `0.6.7.post3`, the version shipped in `scitrera/dgx-spark-sglang:0.5.10`. Qwen3-235B-A22B-NVFP4 on v0.5.10 runs stably with `flashinfer_cutlass` MoE (4 of 36 configs STABLE, test 17: 42.70 tok/s @ n=8). The original "rc0 → 0.5.10 regression" theory was based on GLM-4.7 data only — Qwen3-235B never saw Xid 13 on v0.5.10.

**Conclusion:** Crash B is not a reproducible blocker on the current 0.5.10 image. The PRD's original "no NVFP4 MoE works on v0.5.10" is superseded by the Qwen3-235B test results. GLM-4.7 has a separate v0.5.10 regression (see Crash C below), not Crash B.

### Crash C: `flashinfer_cudnn` fp4_gemm fully broken in v0.5.10

**Symptom:** Any test with `fp4_gemm_backend=flashinfer_cudnn` fails on SM121 in v0.5.10:
- **CUDA graph variants:** 100% startup_crash during graph capture
- **Eager variants:** server stays alive but inference returns 0 tokens (infer_error)

**Coverage of the failure:**
| Model | Version | `fi_cudnn` tests | Outcome |
|-------|---------|------------------|---------|
| Qwen3-235B-A22B-NVFP4 | v0.5.10 | 18/18 | all startup_crash (tests 7–12, 19–24, 31–36) |
| GLM-4.7-NVFP4 | v0.5.10 | tests 20, 23 | infer_error (0 tokens) — rc0 WINNER (test 23: 8.06/21.94/30.01 tok/s) now produces 0 output |
| Qwen3.5-397B-A17B-NVFP4 | v0.5.10 | confirmed | same pattern |

**Regression vs rc0:** The GLM-4.7 rc0 winner (test 23: `fi_cutlass` MoE + `triton` attn + **`fi_cudnn` fp4** + eager → 8.06/21.94/30.01 tok/s) was the single known-stable n=8 config for GLM-4.7. On v0.5.10 this config produces 0 tokens. Both GLM-4.7 test 20 and test 23 use `fi_cudnn` fp4_gemm — confirming the regression is isolated to the `flashinfer_cudnn` FP4 GEMM backend, not attention or MoE runner.

**Root cause:** Not yet isolated. The FlashInfer `cudnn_fused_moe` / cuDNN FP4 GEMM path in 0.6.7.post3 differs from 0.6.5/0.6.6; possibly an interaction with CUDA 13 / cuDNN 9.x on SM121. Unlike Crash B, no upstream fix is identified yet.

**Workaround:** Use `fp4_gemm_backend=flashinfer_cutlass` (the Qwen3-235B winning path). Accept that GLM-4.7 loses its rc0 baseline and must either (a) run on the degraded v0.5.10 `fi_cutlass` fp4 path (test 17: partial stable to n=4, 8.4/20.8 tok/s) or (b) stay on v0.5.10rc0 until the cuDNN regression is fixed.

### Pattern 4: Piecewise CUDA graphs crash the `fi_cutlass` MoE family

Within the working `fi_cutlass` MoE + `fi_cutlass` fp4 family on SM121, the piecewise CUDA graph variants consistently crash at startup while fixed-BS and eager variants run:

| Model | Tests (piecewise ON) | Outcome |
|-------|---------------------|---------|
| Qwen3-235B-A22B-NVFP4 | 15, 18 | startup_crash |
| Qwen3-235B-A22B-NVFP4 | 13, 14, 16, 17 (non-piecewise) | STABLE |

**Implication:** Piecewise graph capture (`disable_piecewise_cuda_graph: false`) is incompatible with the SM121 FlashInfer CUTLASS MoE path. Must set `disable_piecewise_cuda_graph: true` (fixed-BS graphs) or `disable_cuda_graph: true` (eager).

### Previous investigations (ruled out)

1. **Python DSL `admissible_archs`** ([NVIDIA/cutlass#2800](https://github.com/NVIDIA/cutlass/issues/2800)): We patched `BlockScaledMmaOp.admissible_archs` to include `sm_120a` + `sm_121a`. **Ineffective** — the `.cuh` kernel is JIT-compiled via TVM/C++, not through the Python DSL.

2. **Runtime `.cuh` source patch**: We attempted to inject an SM121-specific kernel function with `Shape<_64, _128, _128>` tiles via python heredoc string replacement at startup. **Failed** — CUTLASS template validation rejects the smaller tile shape with the `KernelPtrArrayTmaWarpSpecializedPingpong` schedule:
   ```
   error: static assertion failed with "TMA requires CTA_Tile and SLayout top-level size equivalence."
   error: static assertion failed with "Shape Divisibility Condition"
   error: static assertion failed with "Could not find a common tile-gmem vectorization."
   ```
   The Pingpong schedule has strict TMA descriptor requirements that don't allow reducing M or N dimensions independently.

### What BTankut did differently (important correction)

The [BTankut/dgx-spark-sglang-moe-configs](https://github.com/BTankut/dgx-spark-sglang-moe-configs) repo targets **GLM-4.7-FP8**, not NVFP4. His base image `lmsysorg/sglang:spark` (SGLang v0.5.4.post2, FlashInfer 0.5.0) was inspected:

- The file is `nvfp4_blockwise_moe.cu` (compile-time C++), not `.cuh` (JIT-compiled via TVM in v0.5.10)
- **Identical SM120 function** to v0.5.10: `Shape<_128, _128, _128>`, `KernelPtrArrayTmaWarpSpecializedPingpong`, `StageCountAuto`
- **No SM121 code path at all** — SM121 falls into `TORCH_CHECK_NOT_IMPLEMENTED(false, "Unsupported SM version: " + std::to_string(sm_version))`
- Dispatch: `} else if (sm_version == 120) {` (strict `==`, not `>= 120`)

This means `lmsysorg/sglang:spark` would **crash on SM121** if you tried to run NVFP4 MoE on it. BTankut never ran NVFP4 MoE — his 20–27 tok/s result is for GLM-4.7-**FP8** with Triton MoE (tuned via the MoE kernel config JSONs for the 99 KiB shared memory budget; see [CUTLASS#3144](https://github.com/NVIDIA/cutlass/issues/3144)).

**The 356 TFLOPS NVFP4 result from the forum post** is a dense GEMM micro-benchmark, not a running MoE inference. It demonstrates that SM121 FP4 Tensor Cores work in principle, but does not show a working sgl-kernel + NVFP4 MoE pipeline.

**Conclusion: no known working NVFP4 MoE configuration exists on SM121 in any public sglang build.**

The v0.5.10 code change from `sm_version == 120` to `sm_version >= 120` was the SGLang team's attempt to enable SM121 — it "activates" the code path but fails at runtime because of the 1 KiB shared memory shortfall (102400 bytes requested vs 101376 available, per [CUTLASS#3144](https://github.com/NVIDIA/cutlass/issues/3144)). Copying the v0.5.4 `.cu` would regress us to `Unsupported SM version` crashes — worse than what we have now.

## Known-good baselines

**Primary (as of 2026-04-10): v0.5.10 with `flashinfer_cutlass` MoE + `flashinfer_cutlass` fp4_gemm, non-piecewise CUDA graphs or eager.**

- **Qwen3-235B-A22B-NVFP4** TP=4 EP=4 (test 17 winner): 11.28 / 34.60 / **42.70 tok/s** at n=1/n=4/n=8, fully stable. Test 16 (fixed-BS CUDA graphs): 12.54 / 30.40 / 41.36 tok/s, fully stable with lower n=1 TTFT.
- Image: `scitrera/dgx-spark-sglang:0.5.10` (FlashInfer 0.6.7.post3 ships both Crash B fixes).
- Requires `disable_piecewise_cuda_graph: true` (Pattern 4).

**Legacy baseline (retained for GLM-4.7 only): v0.5.10rc0** with `flashinfer_cutlass` MoE + `triton` attn + `flashinfer_cudnn` fp4 + eager.
- GLM-4.7-NVFP4 TP=4 EP=4: 8.06 / 21.94 / 30.01 tok/s (rc0 test #23).
- On v0.5.10 this config regresses to 0 tokens (see Crash C). GLM-4.7 has no fully-stable n=8 config on v0.5.10.

**Why did rc0 work for GLM-4.7?** The rc0 bundled FlashInfer (likely 0.6.5 or 0.6.6) predates both the CUTLASS 4.4.2 bump (#2798) and the GDC fix (#2913), but also predates the cuDNN FP4 GEMM regression (Crash C). rc0 was the last image where `flashinfer_cudnn` FP4 GEMM worked end-to-end on SM121 for GLM-4.7. On v0.5.10 that path broke for cuDNN, while `flashinfer_cutlass` fp4_gemm improved (Qwen3-235B now stable). Net: the model-specific winner shifted.

## Target: the `cutlass_moe_fp4` codepath

Crash B is (likely) gone and the `flashinfer_cutlass` MoE runner is production-viable for Qwen3-235B. The remaining unsolved problem is **Crash A**: the `cutlass_moe_fp4` codepath, reached via `moe_runner_backend=triton` (internal fallback to `cutlass_moe_fp4` for NVFP4 weights) and `moe_runner_backend=cutlass` (direct).

**Why bother fixing this path at all?**
1. Qwen3-235B v0.5.10 Tests 1–12 (triton MoE) and 25–36 (cutlass MoE) are all **startup_crash** or **infer_error** on the broken `nvfp4_blockwise_moe.cuh:78` path. That's 24 of 36 total configs permanently blocked.
2. The `triton` MoE runner path tends to produce more competitive n=1 throughput on SM120 when working (TRT-LLM's benchmark results). Whether that's true on SM121 is unknown because we've never seen it run.
3. Piecewise CUDA graphs (Pattern 4) may work on the `cutlass_moe_fp4` path (different kernel, different static_assert surface).
4. MoE Triton kernel tuning (`sglang_tune_moe.sh`) only benefits models that use the Triton MoE runner. Models using `flashinfer_cutlass` derive zero benefit from tuned `E=...,N=...,device_name=NVIDIA_GB10.json` configs — the tuning work is wasted unless the Triton path is reachable.

## Implementation Options

### Option 1: Pin FlashInfer to rc0 version — **OBSOLETE (2026-04-10)**

Original plan: pin FlashInfer back to the rc0-bundled version (likely 0.6.5/0.6.6) to recover the working `flashinfer_cutlass` MoE path.

**Why obsolete:** FlashInfer `0.6.7.post3` (shipped in the current `scitrera/dgx-spark-sglang:0.5.10` image) already contains the two fixes that were expected to resolve Crash B: `#2798` (CUTLASS 4.4.2 TMA) and `#2913` (GDC flag race). Downgrading to 0.6.5/0.6.6 would **lose** these fixes rather than gain them. Qwen3-235B Test 17 on v0.5.10 (fi_cutlass MoE + fi_cutlass fp4 + eager → 42.70 tok/s @ n=8, fully stable) empirically confirms the path works on the current version.

Option 1 is retained only as reference for the GLM-4.7 workaround ("stay on rc0" due to the independent Crash C `fi_cudnn` fp4 regression). It does not address Crash A.

### Option 2: Modify sgl-kernel source before build — **PRIMARY PATH**

Patch the NVFP4 blockwise MoE kernel source in the SGLang clone during the `scitrera/cuda-containers` build to add an SM121-compatible kernel path with reduced shared memory footprint.

**File location update (2026-04-10):** PR [#20012](https://github.com/sgl-project/sglang/pull/20012) ("Reland NVFP4 kernels to JIT", merged 2026-03-07) moved NVFP4 MoE kernels from `sgl-kernel/csrc/moe/nvfp4_blockwise_moe.cu` to a JIT-compiled template at **`python/sglang/jit_kernel/csrc/moe/nvfp4_blockwise_moe.cuh`** inside the main SGLang Python package. In v0.5.10, this is the authoritative source — the `.cu` file under `sgl-kernel/csrc/moe/` no longer exists. The kernel is JIT-compiled via nvrtc at runtime (see `python/sglang/jit_kernel/nvfp4.py`).

**Existence proof:** [TensorRT-LLM PR #12141](https://github.com/NVIDIA/TensorRT-LLM/pull/12141) (merged 2026-03-18) solved the identical SM121 shared memory overflow in TRT-LLM's `fp4_gemm_template.h` by making `CtaShape128x128x128B` the default tile and letting the autotuner profile all candidates. This proves the problem has a tractable solution within CUTLASS template constraints on SM121. TRT-LLM used a smaller tile (K=128 instead of K=256) + non-pingpong schedule. Our sgl-kernel equivalent is `StageCount<1>` + `KernelPtrArrayTmaWarpSpecialized` (non-pingpong) on the existing `Shape<_128, _128, _128>` tile.

**Recommendation: Option 2 is the primary path.** Option 1 is obsolete. The target is the `cutlass_moe_fp4` codepath (Crash A), which has no upstream fix and is the only remaining NVFP4 MoE limitation on SM121 for the current 0.5.10 image.

## Implementation (Option 2)

### Target file

**`python/sglang/jit_kernel/csrc/moe/nvfp4_blockwise_moe.cuh`** in the SGLang main-repo clone performed during the Docker build (NOT in sgl-kernel — that path was retired in PR #20012).

Function to patch: `run_fp4_blockwise_scaled_group_mm_sm120()`. This is the dispatch target for `sm_version >= 120` (including SM121). Key lines:
- `using StageCountType = cutlass::gemm::collective::StageCountAuto;` (followed by `StageCountAutoCarveout<sizeof(CollectiveEpilogue::SharedStorage)>` in the `CollectiveBuilder` template instantiation).
- `cutlass::gemm::KernelPtrArrayTmaWarpSpecializedPingpong` — the double-buffered pingpong schedule.

### Patch location in build pipeline

In `scitrera/cuda-containers/container-build/Dockerfile.sglang-nightly`, insert a `COPY` + `RUN patch` step **after** the SGLang main-repo clone and **before** any pip/uv install that processes the Python package:

```dockerfile
# Patch nvfp4_blockwise_moe.cuh for SM120/SM121 (99 KiB SMEM/block).
# The default SM120 kernel path uses Pingpong schedule + StageCountAutoCarveout,
# which requests 102400 bytes (100 KiB) — exactly 1 KiB over the 101376-byte
# device limit on *all* SM12x Blackwell (not just SM121). Only SM100 B200
# (228 KiB) has enough SMEM; upstream SGLang presumably tested there.
# Verbatim numbers + NVIDIA maintainer confirmation:
#   https://github.com/NVIDIA/cutlass/issues/3144
# Fix: StageCount<1> + non-pingpong schedule cuts mainloop SMEM well below
# the 99 KiB budget. Equivalent to TensorRT-LLM PR #12141's approach.
COPY patches/sgl-kernel-sm121.patch /tmp/sgl-kernel-sm121.patch
RUN set -e; cd /data/sglang && \
    patch --dry-run -p1 < /tmp/sgl-kernel-sm121.patch && \
    patch -p1 < /tmp/sgl-kernel-sm121.patch && \
    grep -q 'StageCount<1>' python/sglang/jit_kernel/csrc/moe/nvfp4_blockwise_moe.cuh && \
    echo 'SM121 patch applied successfully'
```

The `--dry-run` before the real apply gives an early, clear failure on any drift between the upstream file and the shipped patch. `grep -q` confirms the change landed.

### Patch content: primary approach + fallbacks

#### Approach 2a (PRIMARY): `StageCount<1>` + `KernelPtrArrayTmaWarpSpecialized`

Combines stage reduction and non-pingpong schedule in a single patch. This is the sgl-kernel equivalent of TRT-LLM PR #12141's approach (which uses `CtaShape128x128x128B` = smaller K-tile + single-stage in TRT-LLM's `fp4_gemm_template.h`).

```cpp
// Current (SM120, in run_fp4_blockwise_scaled_group_mm_sm120):
using StageCountType = cutlass::gemm::collective::StageCountAuto;
// ... in CollectiveMainloop builder:
cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(
    sizeof(typename CollectiveEpilogue::SharedStorage))>,
cutlass::gemm::KernelPtrArrayTmaWarpSpecializedPingpong>::CollectiveOp;

// SM121 patch:
using StageCountType = cutlass::gemm::collective::StageCount<1>;
// ... in CollectiveMainloop builder:
StageCountType,
cutlass::gemm::KernelPtrArrayTmaWarpSpecialized>::CollectiveOp;
```

- `StageCount<1>` — single-stage pipeline, no double buffering → removes the mainloop double-buffer.
- `KernelPtrArrayTmaWarpSpecialized` — non-pingpong grouped-GEMM schedule → removes the second warp-group SMEM duplication that Pingpong imposes.
- `Shape<_128, _128, _128>` — **unchanged**. TMA descriptor validation rejects M/N reduction but tolerates stage + schedule changes.

**Expected SMEM after fix:** well below the 99 KiB device budget. [CUTLASS#3144](https://github.com/NVIDIA/cutlass/issues/3144) documents the pre-fix request as 102400 bytes (100 KiB) and the device limit as 101376 bytes (99 KiB) — a 1 KiB overshoot. The StageCount<1> + non-pingpong combination cuts mainloop SMEM by far more than 1 KiB (each change alone would suffice in theory; combining them provides comfortable headroom). Exact post-fix SMEM is not documented in the issue and will only be measurable from a successful compile; the goal is simply "fits", not "maximally compact".

**Risk:** Lower throughput vs pingpong (no pipeline overlap on the mainloop). Any working kernel is better than 0 tokens.

#### Approach 2b (FALLBACK if 2a fails to compile): `StageCount<2>` + Pingpong

If CUTLASS rejects the combination of `StageCount<1>` with the non-pingpong schedule (some collective builders have schedule/stage coupling constraints), try halving stages while keeping pingpong:

```cpp
using StageCountType = cutlass::gemm::collective::StageCount<2>;
// Keep KernelPtrArrayTmaWarpSpecializedPingpong
```

Expected SMEM: very close to the 99 KiB limit. The pre-fix overshoot is only 1 KiB (per [CUTLASS#3144](https://github.com/NVIDIA/cutlass/issues/3144)), so halving stages alone might free just enough — but with Pingpong's dual warp-group duplication still in place, the margin is thin and this fallback may fail to compile with a "SMEM over budget" error.

#### Approach 2c (FALLBACK if 2a and 2b fail): Port TRT-LLM #12141 tile shape

Mirror TRT-LLM PR #12141 more literally: force `CtaShape128x128x128B` (in CUTLASS template language this maps to the K-dimension reduction within the collective builder). Requires deeper inspection of whether sgl-kernel's `Shape<_128, _128, _128>` tile shape (`M=128, N=128, K=128`) matches TRT-LLM's `128x128x128B` semantics exactly.

**Risk:** K=64 (if attempted) may violate CUTLASS FP4 granularity requirements (FP4 block-scaled uses group_size=16, K must be multiple of the scale block).

### Validation plan

Build sgl-kernel with the patch, then test each approach on a DGX Spark with GLM-4.7-NVFP4 (TP=4 EP=4, `triton` MoE runner, `disable_cuda_graph: true`):

1. Patch applies cleanly during Docker build
2. sgl-kernel wheel builds without CUTLASS static_assert errors
3. SGLang starts without crash
4. First inference request completes without device-side assert
5. Throughput at n=1, n=4, n=8 — compare to rc0 baseline (8.06/21.94/30.01)

If one approach fails to compile, try the next. Approach 2a (StageCount<1>) has the highest probability of success since it's the smallest change and doesn't touch TMA descriptor logic.

### Docker build integration

The patch file `sgl-kernel-sm121.patch` is stored in `scitrera/cuda-containers/container-build/patches/` and referenced from a new Dockerfile variant as shown above. The actual build flow is automated by `scripts/build_sm121_image.sh` in this repo, which handles:
- Local clone of `scitrera/cuda-containers` at `~/pythondev_workspace/cuda-containers`
- Dispatch to an arm64 build host (spark1) via SSH (since the base image is aarch64)
- Docker Hub push to `xomoxcc/dgx-spark-sglang:0.5.10-sm121`

See Phase 4 in `plans/squishy-napping-starlight.md` for the script structure. Manual invocation:

```bash
bash /home/thiess/pythondev_workspace/dgxarley/scripts/build_sm121_image.sh
```

Resulting image: `xomoxcc/dgx-spark-sglang:0.5.10-sm121`.

### Crash B fix — already shipped upstream

FlashInfer 0.6.7.post3 (in the current `scitrera/dgx-spark-sglang:0.5.10`) already contains [#2798](https://github.com/flashinfer-ai/flashinfer/pull/2798) (CUTLASS 4.4.2 TMA descriptor fix) and [#2913](https://github.com/flashinfer-ai/flashinfer/pull/2913) (GDC flag race fix). No downgrade or separate recipe change is needed for Crash B. See "Crash B" in Root Cause Analysis above.

## Validation — Success Criteria

1. **Build succeeds:** `docker build` completes without CUTLASS `static_assert` failures. `grep 'StageCount<1>'` on the patched `nvfp4_blockwise_moe.cuh` in the image confirms the diff landed.
2. **Crash A fixed:** Qwen3-235B-A22B-NVFP4 TP=4 EP=4 with `moe_runner_backend=triton` + `fp4_gemm_backend=flashinfer_cutlass` + eager comes up without `nvfp4_blockwise_moe.cuh:78` device-side assert (this is the reactivation of Qwen3-235B Test 2 which previously returned 0 tokens).
3. **Inference produces tokens:** Single-request test generates >0 output tokens.
4. **Throughput baseline:** `triton` MoE path matches or exceeds `flashinfer_cutlass` MoE baseline on Qwen3-235B (Test 17: 42.70 tok/s @ n=8). If it's significantly slower, the custom image is redundant — upstream `flashinfer_cutlass` is the better path.
5. **Matrix reactivation:** Qwen3-235B Tests 1–6 and 25–30 (previously all `startup_crash` or `infer_error`) should have at least some stable results with the patched kernel.

**Non-goals:** Piecewise CUDA graphs and `fi_cudnn` fp4_gemm are separate issues (Pattern 4, Crash C). Not in scope for this patch.

## Risks

1. **CUTLASS template constraints are strict:** `StageCount<1>` combined with `KernelPtrArrayTmaWarpSpecialized` may trip `static_assert` in the `CollectiveBuilder`. Fallback 2b (`StageCount<2>` + Pingpong) is the first alternative.
2. **Single-stage throughput is lower:** `StageCount<1>` eliminates mainloop pipeline overlap. Expect 20–40% slower per-kernel than upstream SM120. Still, `triton` MoE may beat `flashinfer_cutlass` MoE on n=1 latency because it avoids the FlashInfer dispatch overhead.
3. **JIT recompile per process:** The kernel is JIT-compiled via nvrtc at runtime (see `python/sglang/jit_kernel/nvfp4.py`). First-run startup is slower than an AOT-compiled kernel. The patched `.cuh` source is baked into the image, so the JIT picks it up automatically.
4. **Image maintenance overhead:** The patched image must be rebuilt for every SGLang upgrade. Patch anchors (`StageCountAuto`, `KernelPtrArrayTmaWarpSpecializedPingpong`) may drift. Patch uses `patch -p1` with hard fail on mismatch, so drift is detected immediately.
5. **Upstream fix may land first:** [sgl-project/sglang#11658](https://github.com/sgl-project/sglang/issues/11658) tracks SM121 support. [NVIDIA/cutlass#3144](https://github.com/NVIDIA/cutlass/issues/3144) is the CUTLASS-side root cause issue, open since 2026-04-02. If either resolves in the next 4 weeks, our custom build can be retired.
6. **Exit criterion — the custom image might turn out unnecessary:** If the patched `triton` MoE path does not exceed the current `flashinfer_cutlass` baseline (42.70 tok/s @ n=8 on Qwen3-235B), the patch is proof-of-concept only. The Qwen3-235B cluster can continue using upstream 0.5.10 with `flashinfer_cutlass` MoE in production.

## References

### Primary (our fix path)
- [NVIDIA/TensorRT-LLM#12141](https://github.com/NVIDIA/TensorRT-LLM/pull/12141) — merged 2026-03-18. **Our reference approach.** Makes `CtaShape128x128x128B` the default tile in `fp4_gemm_template.h` for SM120+ and lets the autotuner profile all candidates. Solves the identical SM121 shared memory overflow in TRT-LLM. Proves the problem has a tractable CUTLASS-level solution.
- [NVIDIA/cutlass#3144](https://github.com/NVIDIA/cutlass/issues/3144) — open since 2026-04-02. **Authoritative root cause + SMEM numbers for this PRD.** NVIDIA maintainer @depaulmillz correction: SM100 (B200) has 228 KiB SMEM/block, but SM120 (RTX 5090, RTX PRO 6000 Blackwell) AND SM121 (DGX Spark GB10) both only have **99 KiB**. Exact failure quoted in the issue: "102400 bytes required but device supports 101376" — the kernel requests 100 KiB and the device allows 99 KiB, exactly 1 KiB over. The bug therefore affects every SM12x Blackwell device, not just DGX Spark; upstream SGLang's NVFP4 MoE path works on B200 (SM100) only.
- [sgl-project/sglang#20012](https://github.com/sgl-project/sglang/pull/20012) — merged 2026-03-07. Reland of NVFP4 kernel JIT migration. Moved `nvfp4_blockwise_moe.cu` from `sgl-kernel/csrc/moe/` to `python/sglang/jit_kernel/csrc/moe/nvfp4_blockwise_moe.cuh`. This is why the v0.5.10 patch target path differs from the original PRD.

### Crash B related (already fixed upstream)
- [flashinfer-ai/flashinfer#2798](https://github.com/flashinfer-ai/flashinfer/pull/2798) — merged 2026-03-19. CUTLASS 4.2.1 → 4.4.2 upgrade. Fixes TMA descriptor OOB bug in `tma_warp_specialized_generic_moe_gemm_kernelLauncher<Sm120, fp4>` on DGX Spark SM121. Shipped in FlashInfer 0.6.7.
- [flashinfer-ai/flashinfer#2913](https://github.com/flashinfer-ai/flashinfer/pull/2913) — merged 2026-04-01. GDC flag fix for CUTLASS fused MoE kernel. Fixes `cudaErrorIllegalInstruction` (Xid 13) race on SM90/100/120/121 where host PDL activation raced against device GDC barrier no-ops. Shipped in FlashInfer 0.6.7.post1+.
- [flashinfer-ai/flashinfer#2776](https://github.com/flashinfer-ai/flashinfer/issues/2776) — the original "NVFP4 MoE models crash on GB10 (SM121) during CUDA graph capture" issue, resolved by #2798.

### Tracking / context
- [sgl-project/sglang#11658](https://github.com/sgl-project/sglang/issues/11658) — SM121 tracking issue (closed 2026-01-17 as "completed" for basic boot + CUDA Graph, but NVFP4 MoE kernel support remains unresolved per JCorners68's 2026-03-29 comment).
- [sgl-project/sglang#19637](https://github.com/sgl-project/sglang/issues/19637) — SM120 Performance Optimization Plan. Item "improve grouped GEMM (highest importance) on SM120" is still in progress.
- [sgl-project/sglang#21314](https://github.com/sgl-project/sglang/pull/21314) — merged 2026-04-01. Separated SM100 and SM120 dense NVFP4 GEMM into `nvfp4_scaled_mm_sm100.cuh` and `nvfp4_scaled_mm_sm120.cuh`. Dense-GEMM only, does NOT cover MoE grouped GEMM (Crash A remains).
- [NVIDIA/cutlass#2800](https://github.com/NVIDIA/cutlass/issues/2800) — Python DSL `admissible_archs` restriction. NOT the root cause — our patches in `sglang_launch.sh:146` and `sglang_tune_moe.sh:24-34` are harmless but ineffective for Crash A (the kernel path is nvrtc-compiled C++, not Python DSL).

### Not applicable
- [BTankut/dgx-spark-sglang-moe-configs](https://github.com/BTankut/dgx-spark-sglang-moe-configs) — GLM-4.7-**FP8** workaround for v0.5.4. No NVFP4 support. Last update February 2026, no SM121 NVFP4 fix.
- [Forum: SM121 CUTLASS optimization](https://forums.developer.nvidia.com/t/sm121-cutlass-kernel-optimization-results-nvfp4-356-tflops-moe-grouped-gemm-on-dgx-spark/359960) — 356 TFLOPS NVFP4 **dense GEMM** micro-benchmark only (not a working MoE inference pipeline).
- [Forum: Marlin Fix NVFP4 actually works on SM121 (2026-03-30)](https://forums.developer.nvidia.com/t/marlin-fix-nvfp4-actually-works-on-sm121-dgx-spark/365119) — vLLM-specific (Marlin dequantizes FP4 → BF16). Not a native FP4 MoE solution and not applicable to SGLang.
- [scitrera/cuda-containers](https://github.com/scitrera/cuda-containers) — base image build recipes. Used as build substrate for this PRD's custom image.
