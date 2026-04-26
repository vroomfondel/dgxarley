# scitrera PyTorch 2.10/cu13.1 sm121 Build: SDPA EFFICIENT_ATTENTION Silently Returns Wrong Output

**Likely target repository:** [scitrera/cuda-containers](https://github.com/scitrera/cuda-containers) (build-tooling for `scitrera/dgx-spark-pytorch-dev:2.10.0-v2-cu131`).
**Possible secondary report (still TBD):** upstream PyTorch — only after confirmation that bare upstream PyTorch 2.10 sources, built fresh against sm121, also reproduce the failure.
**Severity:** silent correctness — no NaN, no Inf, no exception, no warning. Output magnitude can be 12–27× off from a CPU reference.
**Status:** observed and fully isolated to scitrera's `dgx-spark-pytorch-dev:2.10.0-v2-cu131` wheel; locally worked around (see `COMFYUI_PROMPT_FAIL.md`); upstream issue not yet filed.

> **Important scoping update — added 2026-04-26.** This bug was
> originally framed as a generic PyTorch 2.10 / sm121 issue.
> Cross-validation against `nvcr.io/nvidia/pytorch:26.02-py3` (NVIDIA's
> NGC build of PyTorch 2.11.0a0+nv26.02 with their own CUDA 13.1 patch
> stack) on the **same** Blackwell GB10 hardware shows EFFICIENT
> producing correct output for every tested shape and dtype. The bug
> is therefore not in NVIDIA's PyTorch build for sm121, but is present
> in scitrera's `dgx-spark-pytorch-dev:2.10.0-v2-cu131` build.
> Whether the underlying defect is in the bare PyTorch 2.10 sources
> scitrera is building from, in scitrera's compile flags / cherry-picks,
> or in a CUDA-13.1-vs-13.x kernel-template difference, is not yet
> isolated. The remaining sections describe what we know in detail.

This document is a fileable bug-report draft. Everything below the
`---` line can be copy-pasted into a scitrera issue with minor edits;
filing it upstream against PyTorch requires the additional bare-source
test described in *Open questions before filing PyTorch upstream*.

---

## TL;DR

On NVIDIA GB10 (Blackwell, compute capability 12.1, the chip in DGX
Spark / ASUS Ascent GX10), the PyTorch 2.10 / CUDA 13.1 build shipped
in `scitrera/dgx-spark-pytorch-dev:2.10.0-v2-cu131` dispatches
`torch.nn.functional.scaled_dot_product_attention` to the
`EFFICIENT_ATTENTION` backend, which executes successfully but returns
output with **norm 1.5×–27× off** from a CPU `MATH` reference,
component-wise drift far larger than the tensor's standard deviation,
and **no NaN/Inf**. The MATH backend matches CPU bit-near. The
FLASH_ATTENTION backend, where dispatchable, also matches. Only
EFFICIENT is broken.

The same reproducer on `nvcr.io/nvidia/pytorch:26.02-py3` (PyTorch
2.11.0a0+nv26.02 with NVIDIA's CUDA 13.1 stack) on identical GB10
hardware produces correct output across all shapes — so the failure
is not intrinsic to "any sm121 PyTorch build", but specific to the
scitrera 2.10/cu13.1 wheel.

The failure mode is silent because every downstream check
(NaN guards, attention masks, softmax normalisation in the next layer)
sees finite numbers in plausible ranges. In practice this corrupts every
transformer text encoder running through SDPA on sm121 — we discovered
it via ComfyUI text-to-image conditioning that produced visually
plausible but text-unrelated outputs.

The kernel-table-probe loop in `aten::_efficient_attention_forward`
emits diagnostic lines of the form

```
FATAL: kernel `fmha_cutlassF_*_sm80` is for sm80-sm100, but was built for sm121
```

every time it walks past a non-matching CUTLASS family entry. The
"survivor" kernel that the dispatcher eventually settles on appears to
be the source of the wrong output.

## Environment

```
torch:     2.10.0   (built locally with TORCH_CUDA_ARCH_LIST=8.0;12.1)
cuda:      13.1
device:    NVIDIA GB10
cap:       (12, 1)   — sm121, Blackwell, ARM64 Grace+Blackwell SoC
host:      ASUS Ascent GX10 / DGX Spark, Ubuntu 24.04 ARM64
driver:    580.95.05
```

This is a custom PyTorch build because no official sm121 wheels exist
yet at the time of writing. The build flags follow the standard
`TORCH_CUDA_ARCH_LIST` convention; nothing in the diagnosis below is
specific to a build flag — the failure is in the kernel selection
logic, which is identical in upstream sources.

The same code path (`aten::_efficient_attention_forward`) is what gets
invoked by any future official Blackwell-consumer-GPU wheel built with
sm121 in the arch list, so this issue is expected to be reproducible
on any sm121 device with PyTorch 2.10.

## Reproducer

Pure PyTorch, no third-party deps. Compares MATH / EFFICIENT_ATTENTION
/ FLASH_ATTENTION against a CPU fp32 `MATH` reference for several
shapes and dtypes. Shapes chosen to mirror common transformer use
(CLIP-L: 77 tokens × 12 heads × 64 dim; SDXL self-attention bottleneck:
4096 tokens × 8 heads × 64 dim).

```python
"""sdpa_sm121_repro.py — minimal SDPA correctness check on sm121."""
import torch
from torch.nn.attention import sdpa_kernel, SDPBackend

torch.manual_seed(0)
print(f"torch={torch.__version__}  cuda={torch.version.cuda}")
print(f"device={torch.cuda.get_device_name(0)}  cap={torch.cuda.get_device_capability(0)}")
print()

shapes = [
    (1, 12,   77, 64, torch.float32),   # CLIP-L-ish, fp32
    (1, 12,   77, 64, torch.bfloat16),  # CLIP-L-ish, bf16 (default in transformers)
    (1, 12,   77, 64, torch.float16),   # CLIP-L-ish, fp16
    (1,  8, 4096, 64, torch.bfloat16),  # SDXL self-attention bottleneck
]

for B, H, N, D, dt in shapes:
    print(f"=== B={B} H={H} N={N} D={D} dtype={str(dt).split('.')[-1]} ===")
    q = torch.randn(B, H, N, D, device="cuda", dtype=dt)
    k = torch.randn(B, H, N, D, device="cuda", dtype=dt)
    v = torch.randn(B, H, N, D, device="cuda", dtype=dt)

    # CPU fp32 reference, computed via the unambiguous MATH path implicitly
    # (small enough that it runs in seconds even at 4096 tokens).
    ref = torch.nn.functional.scaled_dot_product_attention(
        q.cpu().float(), k.cpu().float(), v.cpu().float()
    )

    print(f"  reference (CPU fp32): norm={ref.norm():.3f}  std={ref.std():.4f}")
    print(f"  {'backend':<22} {'norm':>10} {'max|Δ|':>10} {'mean|Δ|':>10} {'has_nan':>8}")
    for backend in [SDPBackend.MATH,
                    SDPBackend.EFFICIENT_ATTENTION,
                    SDPBackend.FLASH_ATTENTION]:
        try:
            with sdpa_kernel([backend]):
                out = torch.nn.functional.scaled_dot_product_attention(q, k, v)
            of = out.cpu().float()
            d = (of - ref).abs()
            has_nan = bool(torch.isnan(out).any().item())
            print(f"  {backend.name:<22} {of.norm().item():>10.3f}"
                  f" {d.max().item():>10.4f} {d.mean().item():>10.4f}"
                  f" {str(has_nan):>8}")
        except Exception as e:
            print(f"  {backend.name:<22} ERROR: {type(e).__name__}: {str(e)[:80]}")
    print()
```

## Observed output (sm121, PyTorch 2.10)

```
torch=2.10.0  cuda=13.1
device=NVIDIA GB10  cap=(12, 1)

=== B=1 H=12 N=77 D=64 dtype=float32 ===
  reference (CPU fp32): norm=43.900  std=0.1805
  backend                      norm     max|Δ|    mean|Δ|  has_nan
  MATH                       43.900     0.0000     0.0000    False
  EFFICIENT_ATTENTION        85.923     1.8189     0.3161    False
  FLASH_ATTENTION    ERROR: RuntimeError: No available kernel. Aborting execution.

=== B=1 H=12 N=77 D=64 dtype=bfloat16 ===
  reference (CPU fp32): norm=43.584  std=0.1792
  backend                      norm     max|Δ|    mean|Δ|  has_nan
  MATH                       43.584     0.0033     0.0002    False
  EFFICIENT_ATTENTION         3.506     1.1487     0.1418    False
  FLASH_ATTENTION            43.584     0.0033     0.0003    False

=== B=1 H=12 N=77 D=64 dtype=float16 ===
  reference (CPU fp32): norm=43.676  std=0.1796
  backend                      norm     max|Δ|    mean|Δ|  has_nan
  MATH                       43.676     0.0005     0.0000    False
  EFFICIENT_ATTENTION       250.800     2.5419     0.7931    False
  FLASH_ATTENTION            43.676     0.0005     0.0000    False

=== B=1 H=8 N=4096 D=64 dtype=bfloat16 ===
  reference (CPU fp32): norm=37.586  std=0.0260
  backend                      norm     max|Δ|    mean|Δ|  has_nan
  MATH                       37.592     0.0005     0.0000    False
  EFFICIENT_ATTENTION      1023.564     4.8858     0.4092    False
  FLASH_ATTENTION            37.591     0.0005     0.0000    False
```

Magnitude of the breakage:

| Shape / dtype | Reference norm | EFFICIENT norm | Ratio |
|---|---|---|---|
| 77 × fp32 | 43.900 | 85.923 | **1.96×** |
| 77 × bf16 | 43.584 | 3.506 | **0.080×** (12× too small) |
| 77 × fp16 | 43.676 | 250.800 | **5.74×** |
| 4096 × bf16 | 37.586 | 1023.564 | **27.2×** |

`max|Δ|` of 1.1–4.9 against a `std` of 0.02–0.18 means component-wise
drift of 5×–250× the natural variation of the output — which is well
above any plausible "fp accuracy" threshold and indicates that the
kernel is computing something genuinely unrelated to the input.

`has_nan=False` everywhere is the worrying part: nothing flags the
result as malformed.

## Expected behaviour

All three backends, where dispatchable, should produce output within
the dtype-appropriate tolerance of the reference:

- MATH: bit-near at fp32, ≤1e-3 max\|Δ\| at bf16/fp16 — confirmed.
- FLASH_ATTENTION: ≤1e-3 max\|Δ\| at fp16/bf16, no fp32 dispatch by
  design — confirmed where dispatchable.
- EFFICIENT_ATTENTION: should match MATH within fp16/bf16 rounding —
  **fails** on sm121.

## Root cause hypothesis

The CUTLASS FMHA kernels bundled into PyTorch 2.10 carry family-range
metadata that hard-codes their applicable compute capabilities as
`sm80` through `sm100`. The dispatcher's probe loop in
`aten::_efficient_attention_forward` walks the kernel table, calls
`(kernel.arch in [family_min, family_max])` for each entry, and rejects
every CUTLASS entry on sm121 because `(12, 1) > (10, 0)`. This is what
emits the diagnostic line

```
FATAL: kernel `fmha_cutlassF_bf16_aligned_64x64_rf_sm80` is for sm80-sm100, but was built for sm121
```

(once per kernel entry, dozens per call). The kernel that ultimately
runs after the probe loop appears to fall outside this CUTLASS family
range and to produce output that is not the attention computation —
its shape is correct, its dtype is correct, its values are finite, but
its norm and component-wise distribution show no relation to a real
attention output.

The `xformers` project hit the symmetric problem and resolved it by
having `cutlass.FwOp` and `cutlass.BwOp` self-reject in
`not_supported_reasons` on `device_capability >= (12, 0)` — this routes
xformers users to `fa2F@2.5.7-pt` (PyTorch's own FA2 bundle) which
runs cleanly. The patch is in our local image build (see
`COMFYUI_SM121_PATCHES.md`), but it does not help the symptom reported
here, because PyTorch's own SDPA dispatcher does not consult xformers.

## Cross-validation matrix

Three runs of the reproducer above on the same Blackwell GB10 hardware
(spark4, kubelet-scheduled nvidia.com/gpu time-slice), against three
different PyTorch builds:

### Run A — our overlay image `xomoxcc/comfyui:sm121`

```
torch.__version__         = 2.10.0
torch.version.cuda        = 13.1
torch.version.git_version = 449b1768410104d3ed79d3bcfe4ba1d65c7f22c0
overlays present:           xformers (patched), sage-attention, ComfyUI custom nodes
```

EFFICIENT broken (numbers in the table below).

### Run B — bare scitrera base `scitrera/dgx-spark-pytorch-dev:2.10.0-v2-cu131`

```
torch.__version__         = 2.10.0
torch.version.cuda        = 13.1
torch.version.git_version = 449b1768410104d3ed79d3bcfe4ba1d65c7f22c0   (identical to Run A)
overlays:                   none — no xformers, no sage, no comfy
```

EFFICIENT broken with byte-identical numbers to Run A. This rules out
every overlay we apply on top of the base. The bug is in the scitrera
wheel itself.

### Run C — NVIDIA NGC `nvcr.io/nvidia/pytorch:26.02-py3`

```
torch.__version__         = 2.11.0a0+eb65b36914.nv26.02
torch.version.cuda        = 13.1
overlays:                   none (NGC PyTorch with NVIDIA's CUDA 13.1 stack)
```

EFFICIENT **correct** — within fp16/bf16 rounding tolerance of CPU
reference and indistinguishable from MATH/FLASH on the same hardware.

### Comparison table (EFFICIENT_ATTENTION norm only)

| Shape × dtype | CPU ref | overlay (A) | scitrera bare (B) | NGC (C) | A=B? | C correct? |
|---|---|---|---|---|---|---|
| 77 × fp32 | 43.900 | 85.923 | 85.923 | 43.900 | ✓ identical | ✓ |
| 77 × bf16 | 43.584 |  3.506 |  3.506 | 43.584 | ✓ identical | ✓ |
| 77 × fp16 | 43.676 | 250.800 | 250.800 | 43.676 | ✓ identical | ✓ |
| 4096 × bf16 | 37.586 | 1023.564 | 1023.564 | 37.592 | ✓ identical | ✓ |

The Run-A vs Run-B equality rules out every layer we maintain on top
of scitrera. The Run-B vs Run-C divergence (same hardware, different
PyTorch build, opposite outcome) is what scopes the defect to the
scitrera build.

## Open questions before filing PyTorch upstream

The cross-validation isolates the failure to the scitrera 2.10/cu13.1
wheel, but does **not** by itself prove the upstream PyTorch 2.10
source tree is fine. NGC ships PyTorch 2.11.0a0 — i.e. a newer source
revision than scitrera's 2.10 — plus NVIDIA's own patch stack on top.
Either of those deltas could carry the fix. Concretely, the
following would still be informative:

1. **Bare upstream PyTorch 2.10** built fresh against sm121 in a
   minimal CUDA 13.1 container. If the bug reproduces, the defect is
   in the upstream 2.10 source and should be filed at
   `pytorch/pytorch` (and is presumably already addressed in the
   2.10→2.11 timeline that NGC tracks).
2. **scitrera's newer build** `scitrera/dgx-spark-pytorch-dev:2.11.0-v1-cu132`
   (referenced in our `CUTLASS_NVFP4_SM121_PRD.md`). If the bug is
   gone there, scitrera has effectively fixed it on their own master
   line, and only `2.10.0-v2-cu131` users (i.e. us, until we rebuild
   against the newer base) are affected.
3. **NGC PyTorch source diff** between their 2.11.0a0+nv26.02 and the
   tagged upstream `v2.11.0a0`. If NVIDIA carries a sm121-specific
   CUTLASS or SDPA patch, that's the upstream-relevant fix to land.

Until at least one of these is done, this report is most safely
filed at scitrera, not at PyTorch.

## Suggested fix

Two non-exclusive options, in increasing order of effort:

1. **Self-reject EFFICIENT on sm121 until a working kernel exists.**
   In `aten/src/ATen/native/transformers/cuda/sdp_utils.cpp`, extend
   the check in `use_mem_efficient_attention(...)` (or equivalent) to
   return `false` for compute capability `>= (12, 0)`. The dispatcher
   then falls through to FLASH (where dtype-applicable) or MATH, both
   of which produce correct output on sm121. This mirrors the xformers
   approach.

2. **Extend the CUTLASS family range.** If the existing CUTLASS
   templates are ABI-compatible with sm121 (they appear to be —
   `TORCH_CUDA_ARCH_LIST=8.0;12.1` builds them without compile errors,
   and a viable kernel runs without crashing), update the family-range
   metadata so the probe loop in `aten::_efficient_attention_forward`
   accepts sm121 kernels rather than rejecting them. This is a longer
   investigation: the rejected kernels were not the source of the
   garbage; some *other* kernel that the dispatcher selected after
   rejecting them was. Identifying that survivor and validating its
   correctness is necessary before extending the range.

For users on sm121 today, option 1 is the safe ship.

## Workaround (no PyTorch changes)

Wrap any SDPA-driven forward with the MATH backend forced via the
public API:

```python
from torch.nn.attention import sdpa_kernel, SDPBackend

with sdpa_kernel([SDPBackend.MATH]):
    out = model(...)
```

For text encoders specifically (where MATH's O(n²) memory and compute
cost is irrelevant — sequences are short), this is essentially free.
For diffusion-style attention at 4096+ tokens, MATH is much slower; in
that case prefer FLASH where dtype allows (bf16/fp16), since FLASH
matches the reference on sm121 in our measurements.

We use this approach in
[`roles/k8s_dgx/templates/comfyui_launch.sh.j2`](roles/k8s_dgx/templates/comfyui_launch.sh.j2)
§4c to wrap ComfyUI's text-encoder forward, which yields end-to-end
correctness for SDXL and Flux text-to-image with no measurable
performance penalty.

## How we found this

1. ComfyUI workflows on sm121 produced visually plausible but
   text-unrelated outputs (a "red apple" prompt rendered an interior
   loft scene; "blue cube" rendered a tiki statue).
2. Sage attention ruled out: removing `--use-sage-attention` produced
   different garbage outputs, not correct ones.
3. Cross-model reproduction with Flux (T5-XXL encoder, completely
   different tokeniser and architecture from SDXL's CLIP-L+CLIP-G)
   showed identical-character failure, narrowing to a path common to
   both.
4. Direct CLIP-L embedding test in the pod via
   `transformers.CLIPTextModel`: GPU produced NaN in fp16/bf16, garbage
   in fp32; CPU was correct.
5. Per-backend isolation via `sdpa_kernel([...])` showed MATH and
   eager-attention bit-near to CPU; EFFICIENT off by orders of
   magnitude. FLASH not dispatchable from `transformers`' tensor-stride
   pattern but works on raw randn tensors as in the reproducer above.

The full forensic trail is in
[`COMFYUI_PROMPT_FAIL.md`](COMFYUI_PROMPT_FAIL.md).

## Related issues / PRs

Search terms for upstream triage:

- `sm121` / `compute capability 12.1` / `Blackwell GB10` / `Spark`
- `_efficient_attention_forward` family check / kernel range
- `FATAL: kernel ... is for sm80-sm100, but was built for sm121`
- `mem_efficient_attention sm121`

(none filed by us at the time of writing)

## Cross-references inside this repository

| Path | Role |
|---|---|
| `COMFYUI_PROMPT_FAIL.md` | End-to-end discovery, full diagnostic chain, fix verification |
| `COMFYUI_SM121_PATCHES.md` | Earlier xformers-side sm121 fixes (cutlass self-reject + FA3 disable) |
| `roles/k8s_dgx/templates/comfyui_launch.sh.j2` §4c | Production workaround: sitecustomize shim wrapping `SDClipModel.forward` with `sdpa_kernel([MATH])` |

## Once filed

Add the GitHub issue URL here and link from
`COMFYUI_PROMPT_FAIL.md` § Future Maintenance. When PyTorch closes the
issue (either by extending the family range or by self-rejecting
EFFICIENT on sm121), re-run the reproducer above; if EFFICIENT now
matches the reference, the §4c workaround in
`comfyui_launch.sh.j2` can be removed.
