# xformers SM_121 Patches for ComfyUI on DGX Spark

This document describes the two patches we apply to xformers v0.0.32 so
that the `xomoxcc/comfyui:sm121` image boots and runs cleanly on
NVIDIA GB10 / Blackwell GPUs (compute capability 12.1, aarch64).

The patches live under `scripts/comfyui/patches/` and are applied at
image-build time by `scripts/comfyui/Dockerfile`. They are needed
because v0.0.32 has two structural incompatibilities with sm121 that
cannot be fixed via build flags alone.

---

## TL;DR

| Issue | Where it lives | Fix |
|---|---|---|
| `cutlass.{Fw,Bw}Op` route to PyTorch's compiled `aten::_efficient_attention_forward`, which hard-rejects sm121 with `FATAL: kernel '..._sm80' is for sm80-sm100, but was built for sm121`. | PyTorch's compiled binary (we cannot patch it). | `xformers-disable-cutlass-on-sm121.patch`: mark cutlass.FwOp/BwOp as "not supported" on capability >= (12, 0). The xformers dispatcher then falls through to `fa2F@2.5.7-pt` (PyTorch's bundled Flash-Attention 2), which is sm121-clean. |
| Flash-Attention 3 has no sm121 SASS (FA3 is sm90-only); a default-dispatch call to `memory_efficient_attention` lands on FA3 first and crashes with `CUDA error: no kernel image is available`. | `xformers/ops/fmha/flash3.py` plus FA3 binaries in the wheel. | Build-time: `XFORMERS_DISABLE_FLASH_ATTN=1` env var (read by `setup.py:280`) — FA3 is not compiled into the wheel. Runtime belt-and-braces: `xformers-fa3-runtime-belt-and-braces.patch` makes `flash3.py` honour the same env var at import time, so even a stray FA3 wheel never registers with the dispatcher. |

After both patches, the dispatcher's effective FMHA priority list on
sm121 is `[fa3F (skipped, disabled), fa2F@2.5.7-pt, cutlassF-pt
(rejected via patch)]`. The first viable op the dispatcher actually
selects is `fa2F@2.5.7-pt` — PyTorch's natively-bundled Flash-Attention
2 (note the `-pt` suffix; this is *not* xformers' own FA2 submodule,
which `XFORMERS_DISABLE_FLASH_ATTN=1` skipped at build time). Verified
runtime behaviour on sm121: 0.40 ms steady-state for a 4096×8×64
bf16 attention, comparable to SageAttention v2 (0.33 ms) on the same
shape. For ComfyUI specifically we additionally launch with
`--use-sage-attention` so Sage replaces xformers entirely on the hot
path; the patches above are the safety net for any code that still
goes through xformers' default dispatch (custom nodes, VAE fallbacks,
upstream changes that bypass the global flag).

---

## Background: why this is necessary

### The cutlass FATAL message

On the first integration run after enabling xformers in the image,
ComfyUI's VAE encode crashed with:

```
AttributeError: module 'xformers.ops' has no attribute 'memory_efficient_attention'
```

We traced this to a partially-broken xformers wheel (xformers `_C.so`
present but unloadable, family-tagged kernels mis-built for sm121).
After fixing triton + arch-list + verify-step in the Dockerfile,
xformers built cleanly and `memory_efficient_attention` was exposed —
but the very first call from a Python smoke test hit:

```
FATAL: kernel `fmha_cutlassF_bf16_aligned_64x64_rf_sm80`
       is for sm80-sm100, but was built for sm121
FATAL: kernel `fmha_cutlassF_bf16_aligned_64x64_rf_sm80` ... (x30+)
```

Investigation: in xformers v0.0.32 the cutlass FMHA kernels are no
longer shipped in xformers' own source tree. `cutlass.FwOp` defaults to
PyTorch's built-in `aten::_efficient_attention_forward` whenever
`is_pt_cutlass_compatible()` returns True. The FATAL message is
emitted by PyTorch's compiled CUTLASS dispatcher, which has a
hard-coded family-range check:

- The kernel's name encodes the family (`_sm80` claims to cover sm80-sm100)
- The binary's actual gencode tag is sm121 (we built the wheel with `TORCH_CUDA_ARCH_LIST=8.0;12.1`)
- Dispatcher: `kernel_arch in [family_min, family_max]` -> sm121 fails the check on the sm80 family entry.

Patching PyTorch's compiled binary is out of scope. The cleanest
xformers-side fix is to prevent xformers from selecting cutlass.FwOp
on sm121 in the first place — the dispatcher already has a
`not_supported_reasons()` mechanism for exactly this purpose.

### The FA3 no-kernel-image crash

Once we tested xformers' default dispatch (no `op=` kwarg), we saw:

```
CUDA error (.../flash-attention/hopper/flash_fwd_launch_template.h:188):
  no kernel image is available for execution on the device
```

FA3 in xformers v0.0.32 is bundled from
`third_party/flash-attention/hopper/`, hard-targeted to sm90/sm90a.
There is no FA3 SASS for sm121, so any actual launch on Blackwell
fails. xformers' `fa3_available()` only checks `compute_capability >= (9, 0)` — sm121 (12, 1) passes that, so FA3 is offered to the dispatcher first.

The clean fix is to disable FA3 entirely on this image: we don't have
any Hopper hardware in the cluster, and even if FA3 existed for
Blackwell we'd want to keep the build small and predictable.

---

## Patch 1: `xformers-disable-cutlass-on-sm121.patch`

**File touched:** `xformers/ops/fmha/cutlass.py` (single file, 4 hunks).

**Effect:**
- Adds class-level constant `CUDA_MAXIMUM_COMPUTE_CAPABILITY = (12, 0)` to `FwOp`. `BwOp` inherits the same value via `BwOp.CUDA_MAXIMUM_COMPUTE_CAPABILITY = FwOp.CUDA_MAXIMUM_COMPUTE_CAPABILITY` so the threshold lives in one place.
- In both `FwOp.not_supported_reasons` and `BwOp.not_supported_reasons`, before the alignment checks, inserts a guard:

```python
if d.device.type == "cuda":
    device_capability = torch.cuda.get_device_capability(d.device)
    if device_capability >= cls.CUDA_MAXIMUM_COMPUTE_CAPABILITY:
        reasons.append(
            f"requires device with capability < {cls.CUDA_MAXIMUM_COMPUTE_CAPABILITY} "
            f"(got {device_capability}; CUTLASS FMHA kernels only cover sm60-sm100)"
        )
```

The check is `>= (12, 0)`, so it fires on sm121 (12, 1) and any future
Blackwell or post-Blackwell revision until upstream xformers ships a
sm121-aware kernel.

**Effect on dispatch:** Calls to
`xformers.ops.memory_efficient_attention(q, k, v)` go through
`_dispatch_fw_priority_list`, which calls `op.not_supported_reasons()`
on each candidate and skips ops whose reasons are non-empty. With this
patch, `cutlass.FwOp` always has at least one reason on sm121, so the
dispatcher walks past it to the next entry.

The runtime priority list on sm121 with both patches active is:

```
['fa3F@0.0.0', 'fa2F@2.5.7-pt', 'cutlassF-pt']
```

- `fa3F` is in the list (xformers always populates `ALL_FW_OPS` statically) but `flash3._C_flashattention3 is None` thanks to Patch 2, so it skips itself in `not_supported_reasons`.
- `fa2F@2.5.7-pt` is the first viable entry. The `-pt` suffix marks it as PyTorch's bundled Flash-Attention 2 implementation (loaded via `torch._C`), distinct from xformers' own FA2 submodule which `XFORMERS_DISABLE_FLASH_ATTN=1` excluded from the wheel. PyTorch's FA2 has sm121 SASS and runs cleanly.
- `cutlassF-pt` is at the end of the list and would also have been viable on older hardware, but on sm121 our patch makes it self-reject before the dispatcher tries it.

Measured throughput on sm121 (4096×8×64 bf16, default dispatch picking fa2F-pt): **0.40 ms steady-state** — within striking distance of SageAttention v2's 0.33 ms on the same shape. Both are well under what a typical ComfyUI workflow would care about per layer.

**Behavioural delta visible to user code:**
- Default dispatch silently picks Triton instead of CUTLASS. No error.
- An explicit `op=cutlass.FwOp` call now fails fast with `NotImplementedError(...requires device with capability < (12, 0)...)` instead of crashing inside the kernel. The error message names the constraint, so future readers can find this document.

**What this patch does *not* do:**
- It does not patch PyTorch. PyTorch's `_efficient_attention_forward` is still buggy on sm121; we just don't call into it via xformers.
- It does not extend Cutlass to actually run on sm121. We give up on cutlass on sm121 entirely. If/when upstream PyTorch+CUTLASS ship sm121 kernels, this patch should be reverted.

---

## Patch 2: `xformers-fa3-runtime-belt-and-braces.patch`

**File touched:** `xformers/ops/fmha/flash3.py` (single file, 1 hunk).

**Effect:** turns the top-level FA3 init block into an env-var-aware
chain:

```python
if os.environ.get("XFORMERS_DISABLE_FLASH_ATTN", "0") != "0":
    logger.info(
        "Flash-Attention 3 disabled at import time via XFORMERS_DISABLE_FLASH_ATTN env var"
    )
elif importlib.util.find_spec("...flash_attn_3._C", package=__package__):
    # original FA3 import + registration block, unchanged
    ...
```

`os` is already imported at the top of `flash3.py`, so no new import
is needed. With the env var set (or with FA3 simply not built), the
elif branch never runs, `_C_flashattention3` stays `None`,
`fa3_available()` returns False, and the dispatcher's priority-list
builder omits FA3 altogether.

**Why is this needed if we already disable FA3 at build time?**

The build-time disable (`XFORMERS_DISABLE_FLASH_ATTN=1` env var read by
`setup.py:280`) prevents FA3 from being compiled into the wheel.
That alone is sufficient under normal conditions. The runtime patch
exists as defence-in-depth for two scenarios:

1. A pre-built FA3 wheel slips in via the pip cache (`--mount=type=cache`)
   from a previous build that didn't have the env var set.
2. The image is later mutated post-build (e.g., a user installs an
   xformers wheel with FA3 baked in over the top).

In both cases, setting `XFORMERS_DISABLE_FLASH_ATTN=1` in the
container's runtime env is enough to keep FA3 out of the dispatcher.

---

## Build integration

Relevant block in `scripts/comfyui/Dockerfile`:

```dockerfile
ARG XFORMERS_REF=v0.0.32
COPY patches/xformers-disable-cutlass-on-sm121.patch /tmp/patches/...
COPY patches/xformers-fa3-runtime-belt-and-braces.patch /tmp/patches/...
RUN --mount=type=cache,target=/root/.cache/pip \
    if [ "${BUILD_XFORMERS}" = "1" ]; then \
        git clone --depth 1 --branch "${XFORMERS_REF}" --recurse-submodules \
            https://github.com/facebookresearch/xformers.git /tmp/xformers && \
        cd /tmp/xformers && \
        git apply --verbose /tmp/patches/xformers-disable-cutlass-on-sm121.patch && \
        git apply --verbose /tmp/patches/xformers-fa3-runtime-belt-and-braces.patch && \
        XFORMERS_DISABLE_FLASH_ATTN=1 \
        TORCH_CUDA_ARCH_LIST="8.0;12.1" \
        CMAKE_CUDA_ARCHITECTURES="80;121" \
        pip install --no-build-isolation -v . && \
        cd / && rm -rf /tmp/xformers && \
        python3 -c "..."     # see verify-step below
```

Notable points:

- We do `git clone` rather than `pip install git+...` because the
  `git+...` form has no convenient hook for `git apply` between clone
  and install.
- `--recurse-submodules` populates `third_party/cutlass` and
  `third_party/flash-attention`, both required by xformers' setup.py
  even when FA3 itself is disabled (cutlass templates are still pulled
  for the few non-FMHA cutlass uses in xformers).
- The arch list `8.0;12.1` keeps build memory in the GB10 budget. We
  deliberately omit 9.0/sm90: no Hopper hardware in this cluster, and
  the third arch roughly doubles NVCC peak memory pressure.
- `XFORMERS_DISABLE_FLASH_ATTN=1` is a per-command env prefix on the
  `pip install` line; it does not persist into subsequent layers.
  The runtime patch in `flash3.py` is what carries the disable forward
  to runtime.

### Verify step

After install, the Dockerfile runs:

```python
import xformers, xformers.ops as xo
assert hasattr(xo, 'memory_efficient_attention')
from xformers.ops.fmha import cutlass, flash3
assert cutlass.FwOp.CUDA_MAXIMUM_COMPUTE_CAPABILITY == (12, 0), \
    'cutlass sm121-disable patch did not apply'
assert flash3._C_flashattention3 is None, \
    'FA3 was not disabled (XFORMERS_DISABLE_FLASH_ATTN guard failed)'
print('xformers', xformers.__version__,
      '— mea ok, cutlass<12.0 only, FA3 disabled')
```

Each assertion catches one specific regression mode:

1. xformers builds but `_C` doesn't load (early-2026 builds had this from
   a partial build path; the assertion catches it before runtime).
2. Patch 1 didn't apply or was silently dropped — `CUDA_MAXIMUM_COMPUTE_CAPABILITY`
   wouldn't be set.
3. FA3 leaked through (build env wasn't set, or the wheel cache returned
   a pre-built FA3-enabled wheel) — `_C_flashattention3` would be a
   non-None torch.ops module.

Any failure here aborts the image build with a clear error rather than
producing a silently-broken image.

---

## ComfyUI integration

The patches make xformers safe to *import* on sm121, but ComfyUI
internally still calls `xformers.ops.memory_efficient_attention(...)`
in some VAE/attention paths (e.g.,
`comfy/ldm/modules/diffusionmodules/model.py:298`). With the patches,
those calls dispatch to Triton and work, but the launch script in
`roles/k8s_dgx/templates/comfyui_launch.sh.j2` additionally sets
`--use-sage-attention` so Sage replaces xformers wholesale on the hot
path:

```bash
exec "$PY" main.py \
  --listen 0.0.0.0 \
  --port {{ comfyui_port }} \
  --use-sage-attention \
  ...
```

Reasoning: SageAttention v2 was verified working on sm121 in the same
image build (smoke test: 0.33 ms/call steady-state for 4096-token
multi-head bf16 attention) and is the recommended Blackwell backend.
The xformers patches are the safety net for code paths that ignore
the global flag.

---

## Maintenance

### When to revisit these patches

- **Upstream xformers version bump.** The patches are pinned to
  v0.0.32 source layout. After bumping `XFORMERS_REF`, run the patches
  through `git apply --check` against the new tag and update the diff
  context if hunks have moved. Both files (`cutlass.py`, `flash3.py`)
  are stable areas of the codebase, so context drift should be small.
- **Upstream xformers ships sm121 support.** Watch
  `xformers/ops/fmha/cutlass.py` for any device-capability handling.
  When a release adds proper sm121 dispatch, drop Patch 1 entirely and
  let cutlass run natively. Patch 2 can stay as long as we don't have
  Hopper hardware.
- **PyTorch ships a sm121-aware `_efficient_attention_forward`.**
  Then Patch 1 becomes obsolete even without xformers changes. Test
  by removing Patch 1 and running the smoke test; if no FATAL appears,
  drop it.
- **FA3 ships a sm121 SASS.** Then drop both `XFORMERS_DISABLE_FLASH_ATTN=1`
  and Patch 2.

> **2026-06-11 — xformers releases v0.0.33–v0.0.35; patch status update (last checked 2026-06-11 against v0.0.35):**
>
> xformers v0.0.33 (2025-11), v0.0.34 (2026-01-23), and v0.0.35 (2026-02-20, titled "Rely on upstream FA3") have been released since the v0.0.32 pin.
>
> **Patch 1 — still required on v0.0.35.** Verified: `xformers/ops/fmha/cutlass.py` in v0.0.35 still has `CUDA_MAXIMUM_COMPUTE_CAPABILITY = (9, 0)` with no sm121 dispatch added in any of v0.0.33–v0.0.35. Patch 1 remains necessary.
>
> **Patch 2 — rationale changes on v0.0.35+.** v0.0.35 stopped bundling prebuilt Flash-Attention 3 and instead relies on PyTorch-index wheels ("Rely on upstream FA3"). This means the **build-time** `XFORMERS_DISABLE_FLASH_ATTN=1` disable may be unnecessary on v0.0.35+ (there is no bundled FA3 to suppress at build time). However, the **runtime belt-and-braces** patch in `flash3.py` may still be warranted as defence against a stray FA3 wheel from the pip cache or a post-build install. Verify against a v0.0.35 build before dropping either: confirm `flash3._C_flashattention3 is None` without the build-time env var, then decide. Do NOT drop Patch 2 until verified. `XFORMERS_REF` and both patches are **unchanged** — evaluate before bumping.

### How to verify a candidate xformers version

```bash
# Re-clone fresh
git clone --depth 1 --branch <NEW_TAG> https://github.com/facebookresearch/xformers.git /tmp/xfresh
cd /tmp/xfresh

# Apply both patches (dry-run; -p0/-p1 unchanged from current diff)
git apply --check /path/to/patches/xformers-disable-cutlass-on-sm121.patch
git apply --check /path/to/patches/xformers-fa3-runtime-belt-and-braces.patch
```

If either `--check` fails, regenerate the diff:

1. Apply manually (with `-3` for 3-way fallback if needed).
2. Re-export with `git diff -U5 > /path/to/patches/<patch-name>.patch`.
3. Re-run `git apply --check` on a clean tree to confirm.

### End-to-end validation

Verified via `dgxarley.integration.comfyui_integration_test` against the
deployed pod on 2026-04-25 (image BUILDTIME `2026-04-25T15:07:41Z`):

```
[PASS] health (0.08s)
[PASS] system_stats (0.02s) — devices=['cuda:0 NVIDIA GB10 : native']
[PASS] queue (0.02s)
[PASS] object_info_checkpoints (0.12s) — 3 checkpoints
[PASS] image_generation (24.87s)
[PASS] text2image_realvisxl (30.38s)
[PASS] text2image_flux_schnell (16.21s)
[PASS] text2image_flux_dev (54.65s)
8/8 passed in 126.4s
```

All three model families (SDXL via RealVisXL, FLUX schnell, FLUX dev)
complete their full pipeline (UNet + CLIP + VAE) and produce real
images — no crashes, no hangs.

> **CAVEAT — added 2026-04-26.** The integration test above only
> validates *workflow completion* (HTTP 200, PNG returned), **not
> semantic correctness of the output**. On the scitrera-pipeline
> PyTorch wheel for sm121, text-encoder forwards run through
> `aten::_efficient_attention_forward` and silently return numerically
> incorrect output (norm 12–27× off from the CPU reference, no NaN/Inf).
> The integration test does not compare prompt-to-image semantics, so
> it passed cleanly while the actual outputs were prompt-unrelated
> garbage. **Production resolution: switched the ComfyUI image base to
> `nvcr.io/nvidia/pytorch:26.03-py3`**, whose torch wheel has correct
> SDPA EFFICIENT_ATTENTION on sm121. The earlier
> [`comfyui_launch.sh.j2`](roles/k8s_dgx/templates/comfyui_launch.sh.j2)
> §4c sitecustomize shim (wraps `SDClipModel.forward` with
> `sdpa_kernel([SDPBackend.MATH])`) is now disabled by default
> (`SM121_SDPA_SHIM_ENABLED=0`); kept as a fallback toggle in case a
> future base-image regression brings the bug back. Full diagnostic
> trail in [`COMFYUI_PROMPT_FAIL.md`](COMFYUI_PROMPT_FAIL.md), upstream
> bug write-up in
> [`UPSTREAM_PYTORCH_SDPA_SM121.md`](UPSTREAM_PYTORCH_SDPA_SM121.md).
> The xformers patches in this document remain correct and necessary —
> they cover xformers' own dispatcher, a separate concern from the
> SDPA-EFFICIENT path that was broken on the scitrera build.

### `FATAL: kernel` log noise — filtered on stdout (and stderr)

During a real workflow, PyTorch's CUTLASS kernel selector emits log
lines like:

```
FATAL: kernel `fmha_cutlassF_f32_aligned_64x64_rf_sm80`
       is for sm80-sm100, but was built for sm121
FATAL: kernel `fmha_cutlassF_f32_aligned_64x64_rf_sm80` ... (~30 per VAE call)
```

This is **not a crash** — the dispatcher walks the kernel table and
prints `FATAL: ...` for every entry that does not match the current
device, then returns from the call. ComfyUI continues, the workflow
runs to completion, a PNG is returned.

> **CAVEAT — added 2026-04-26, updated for NGC base.** Earlier
> revisions of this section called the FATAL noise "cosmetic" and
> asserted that the surviving kernel "runs successfully — confirmed by
> the 8/8 integration test pass." That second half was wrong on the
> scitrera-pipeline base: the kernel surviving the probe loop returned
> numerically incorrect output (see
> [`UPSTREAM_PYTORCH_SDPA_SM121.md`](UPSTREAM_PYTORCH_SDPA_SM121.md) for
> the reproducer). After the switch to `nvcr.io/nvidia/pytorch:26.03-py3`
> the FATAL noise pattern is GONE on the production image — NGC's
> dispatcher uses a different family-range table that recognises sm121
> kernels. If you see `FATAL: kernel ... is for sm80-sm100` lines on
> any future build, that's a strong signal the base image regressed to
> the scitrera-style build and SDPA correctness needs to be re-verified
> (see the smoke test in `UPSTREAM_PYTORCH_SDPA_SM121.md`). We keep the
> stdout/stderr filter in `comfyui_launch.sh.j2` defensively because
> the volume on a regressed image would be intolerable — and because
> the filter is harmless on a healthy image (no FATAL lines means no
> work for grep).
> The bf16 smoke test in this document never triggers them because bf16
> takes a different selector path that exits early; only the f32
> probing path emits the spam, and ComfyUI's VAE attention runs in
> f32.

**Important — these lines go to stdout, not stderr.** The C-level
`printf(...)` in PyTorch's kernel registry writes to fd 1 by
definition. Verified empirically inside the running pod by forcing
the EFFICIENT_ATTENTION path on f32: 2048 lines on fd 1, 0 on fd 2.
An earlier version of the filter only redirected fd 2 and was a
no-op. We now redirect both fds in the launch script
(`roles/k8s_dgx/templates/comfyui_launch.sh.j2`):

```bash
# Drop "FATAL: kernel ..." lines on both fds before exec
exec  > >(grep --line-buffered -v '^FATAL: kernel ')
exec 2> >(grep --line-buffered -v '^FATAL: kernel ' >&2)
exec "$PY" main.py ...
```

The printf comes from C++ below `TORCH_CPP_LOG_LEVEL` and can't be
silenced via env var, so the shell-side filter is the simplest
working option.

Trade-offs of this filter:

- Each `grep` becomes a child process; when python exits, both get
  EOF and tear down. Adds <1 ms of latency per filtered line and a
  few KB of memory per fd.
- `--line-buffered` keeps multi-line Python tracebacks intact —
  they pass through unchanged because `^FATAL: kernel ` only matches
  the literal noise lines.
- Pattern is intentionally narrow: any FATAL not starting with
  `FATAL: kernel ` (e.g., a real torch.distributed FATAL) still
  reaches the log.
- A future PyTorch change in the FATAL string format — or a switch
  back to `fprintf(stderr, ...)` — would silently defeat the filter.
  Re-validate after torch upgrades by running the EFFICIENT_ATTENTION
  f32 reproducer and counting `FATAL` on each fd.

Alternatives we deliberately did *not* take:

- `TORCH_CPP_LOG_LEVEL=ERROR`: silences torch C++ INFO/WARN, but the
  FATAL printf is below the log framework, so it doesn't help here.
- Patching PyTorch's source to demote the printf: out of scope; would
  require a full torch source rebuild.
- LD_PRELOAD'ing a custom `write()` shim: more surgical but
  harder to debug than a one-line shell redirect.

### Smoke test in the running pod

```bash
POD=$(kubectl --context=ht@dgxarley -n comfyui get pod -l app=comfyui -o name | head -1)
kubectl --context=ht@dgxarley -n comfyui exec ${POD} -- python3 -c '
import torch
from xformers.ops.fmha import cutlass, flash3
import xformers.ops as xo

# Patch 1 applied?
print("cutlass.MAX_CC:", cutlass.FwOp.CUDA_MAXIMUM_COMPUTE_CAPABILITY)

# Patch 2 + build-time disable both effective?
print("flash3._C:", flash3._C_flashattention3)

# Default dispatch picks something that runs on sm121?
q = torch.randn(1, 16384, 1, 64, device="cuda", dtype=torch.bfloat16)
out = xo.memory_efficient_attention(q, q, q)
print("dispatch ok, out shape:", tuple(out.shape))
'
```

Expected output:

```
cutlass.MAX_CC: (12, 0)
flash3._C: None
dispatch ok, out shape: (1, 16384, 1, 64)
```

If `cutlass.MAX_CC` is missing (`AttributeError`), Patch 1 didn't
apply. If `flash3._C` is not None, FA3 leaked through. If the dispatch
crashes, the priority list collapsed to nothing — likely an upstream
change in `_dispatch_fw_priority_list` and a sign that the patches
need to be revisited against the new layout.

---

## Related files

| Path | Role |
|---|---|
| `scripts/comfyui/patches/xformers-disable-cutlass-on-sm121.patch` | Patch 1 |
| `scripts/comfyui/patches/xformers-fa3-runtime-belt-and-braces.patch` | Patch 2 |
| `scripts/comfyui/Dockerfile` | Builds xformers from source, applies patches, runs verify step |
| `scripts/build_comfyui_image.sh` | Wraps the Dockerfile build, registry distribution, optional Docker Hub push |
| `roles/k8s_dgx/templates/comfyui_launch.sh.j2` | Sets `--use-sage-attention` so ComfyUI bypasses xformers on the hot path |
| `COMFYUI_ARM64_SM121.md` | Higher-level guide to building the image for DGX Spark |
