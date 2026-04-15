# Flashinfer Upstream Bug: FP4 quantization lazy init breaks under torch.compile/dynamo

Two related failure modes in `flashinfer.quantization.fp4_quantization`'s lazy
initialization path, both hit when sglang's piecewise CUDA graph capture runs
`torch.compile` over a forward that reaches `fp4_quantize`:

1. **subprocess spawn** for `nvcc --version` from inside a dynamo trace.
2. **`pathlib.Path.exists()` → `os.stat`** from inside a dynamo trace
   (dynamo marks `posix.stat` as a skipped builtin).

Both come from the same underlying design issue: flashinfer's FP4 JIT build
chain is lazily triggered on the first `fp4_quantize()` call, and if that
first call is inside a traced forward, the build-time filesystem/subprocess
operations blow up dynamo.

## Status

**Patch 1 shipped and stable. Patch 2 is unresolved — see
"Update 2026-04-15 evening" below.** 2026-04-15 morning session outcome:

- **Issue 1 root cause**: `flashinfer.jit.cpp_ext.get_cuda_version()` calls
  `subprocess.check_output([nvcc, "--version"])` on its first invocation (it's
  `@functools.cache`-decorated, so only once per process). When that first
  invocation is reached from inside a `torch.compile` / dynamo trace context,
  dynamo tries to polyfill the `subprocess.Popen` call via
  `torch/_dynamo/polyfills/__init__.py:392 getattr_and_trace` — the polyfill
  cannot handle `Popen.__init__`'s internal fork/threading machinery, and the
  child process dies with sigquit. The sglang launcher observes the child
  failure and restarts the pod → `startup_crash`.
- **Issue 1 fix in `sglang_launch.sh`**: short-circuit `get_cuda_version()` to
  return `Version(torch.version.cuda)` directly (which is always populated on
  our CUDA-built PyTorch and matches what `nvcc --version` reports for the same
  install). The subprocess path remains as an untaken fallback for PyTorch
  builds without CUDA support. Idempotent, grep-guarded, and the marker
  `_fi_cuda_ver_subprocess_bypass_` prevents double application.
- **Issue 2 root cause** (uncovered after Issue 1 was fixed): with the
  subprocess call removed, the next call in the JIT chain is
  `JitSpec.build_and_load()` → `self.is_aot` → `self.aot_path.exists()` →
  `pathlib.Path.stat()` → `os.stat()` → `posix.stat`. Dynamo marks `posix.stat`
  as a skipped builtin and raises
  `torch._dynamo.exc.Unsupported: Attempted to call function marked as skipped`.
  The failure now gets caught by sglang's piecewise CUDA graph runner as
  `Piecewise CUDA Graph failed with error: ...`, which is less violent than
  the sigquit from Issue 1 but still aborts the test.
- **Issue 2 fix in `sglang_launch.sh`**: patch
  `flashinfer/quantization/fp4_quantization.py` at image startup to:
  1. Wrap `get_fp4_quantization_module` in `functools.cache` so the
     `build_and_load()` chain runs at most once per process.
  2. Append an import-time pre-warm that calls
     `get_fp4_quantization_module(f"{sm_major}{sm_minor}")` once, during
     normal Python import (always outside any dynamo trace). This populates
     the cache with the built module.
  When sglang later re-enters `fp4_quantize()` from inside a dynamo-traced
  forward, the wrapped function returns the cached module instantly, with no
  `build_and_load`, no `is_aot`, no stat, no subprocess. Idempotent via the
  marker `_fi_fp4_cache_and_prewarm_`.
- **Not reported upstream yet** — needs a minimal repro (`torch.compile` + any
  flashinfer FP4 quant call from inside the traced region is enough, for both
  issues). Adjacent issues exist but none match these exact failure modes.
  See "Upstream status".

Bug exists in flashinfer **0.6.7.post3** (the version shipped in
`scitrera/dgx-spark-sglang:0.5.10`) and is structurally present in all
flashinfer releases that have `get_cuda_version()` spawning `nvcc` at call time
— i.e. everything since the subprocess-based version lookup was introduced.
Verified absent in our patched `xomoxcc/dgx-spark-sglang:0.5.10-cudnn` image
after the runtime patch is applied.

## Summary

Flashinfer's FP4 quantization path JIT-compiles kernels on first use. The JIT
build calls `is_cuda_version_at_least("12.8")` → `get_cuda_version()` →
`subprocess.check_output([nvcc, "--version"])`. If the very first call happens
from a Python frame that dynamo is actively tracing (e.g., piecewise CUDA
graph capture on a forward pass that hits `fp4_quantize`), the subprocess spawn
crashes the child process instead of returning the version string.

The function is `@functools.cache`-decorated, so if it's called **once outside**
a dynamo trace before the traced path is hit, the cache is populated and the
subprocess spawn never happens again — and flashinfer works fine. That's why
sglang's non-piecewise configs are unaffected (the graph-capture path doesn't
go through dynamo the same way): the JIT build fires at a "safe" moment or
uses a different backend module that was already compiled.

Our monkey-patch removes the subprocess path entirely, so it doesn't matter
when or from where `get_cuda_version()` is called.

## Symptom (Issue 1: subprocess from dynamo trace)

Observed on GLM-4.7-NVFP4 at EP=1 on 4× DGX Spark (SM121/GB10) with
`fp4_gemm_backend=flashinfer_cudnn`, `disable_piecewise_cuda_graph=false`,
running on the `xomoxcc/dgx-spark-sglang:0.5.10-cudnn` image (which already
has the `nvidia-cudnn-cu12` wheels installed, so the old cuDNN missing-dep
is out of the way). Verbatim stack from the sglang head pod
(`sglang-head-6c984df886-zvxmb`):

```
File "/usr/local/lib/python3.12/dist-packages/sglang/srt/models/glm4_moe.py", line 174, in forward
    gate_up, _ = self.gate_up_proj(x)
File "/usr/local/lib/python3.12/dist-packages/sglang/srt/layers/linear.py", line 460, in forward
    output_parallel = self.quant_method.apply(self, input_, bias)
File "/usr/local/lib/python3.12/dist-packages/sglang/srt/layers/quantization/modelopt_quant.py", line 1482, in apply
    x_fp4, x_scale_interleaved = fp4_quantize(x, layer.input_scale_inv)
File "/usr/local/lib/python3.12/dist-packages/flashinfer/quantization/fp4_quantization.py", line 700, in fp4_quantize
    x_q, sf = get_fp4_quantization_module(f"{major}{minor}").fp4_quantize_sm100(
File "/usr/local/lib/python3.12/dist-packages/torch/_dynamo/polyfills/__init__.py", line 392, in getattr_and_trace
    return fn(*args[2:], **kwargs)
File "/usr/local/lib/python3.12/dist-packages/flashinfer/quantization/fp4_quantization.py", line 170, in get_fp4_quantization_module
    module = backend_modules[backend]().build_and_load()
File "/usr/local/lib/python3.12/dist-packages/flashinfer/quantization/fp4_quantization.py", line 107, in gen_fp4_quantization_sm120f_module
    return gen_fp4_quantization_module(sm120f_nvcc_flags, "120f")
File "/usr/local/lib/python3.12/dist-packages/flashinfer/quantization/fp4_quantization.py", line 131, in gen_fp4_quantization_module
    "-DENABLE_FP4" if is_cuda_version_at_least("12.8") else "",
File "/usr/local/lib/python3.12/dist-packages/flashinfer/jit/cpp_ext.py", line 91, in is_cuda_version_at_least
    return get_cuda_version() >= Version(version_str)
File "/usr/local/lib/python3.12/dist-packages/flashinfer/jit/cpp_ext.py", line 73, in get_cuda_version
    txt = subprocess.check_output([nvcc, "--version"], text=True)
File "/usr/lib/python3.12/subprocess.py", line 466, in check_output
    return run(*popenargs, stdout=PIPE, timeout=timeout, check=True,
File "/usr/lib/python3.12/subprocess.py", line 548, in run
    with Popen(*popenargs, **kwargs) as process:
File "/usr/lib/python3.12/subprocess.py", line 828, in __init__
    self._waitpid_lock = threading.Lock()

Set TORCHDYNAMO_VERBOSE=1 for the internal stack trace (please do this especially if you're reporting a bug to PyTorch). For even more developer context, set TORCH_LOGS="+dynamo"

[2026-04-15 07:45:17] Received sigquit from a child process. It usually means the child failed.
```

The two tells:
1. `File "torch/_dynamo/polyfills/__init__.py", line 392, in getattr_and_trace` in
   the middle of the flashinfer stack — dynamo is actively tracing when the
   subprocess call happens.
2. `Set TORCHDYNAMO_VERBOSE=1 ...` footer from torch — dynamo itself emitted
   this hint, so it was involved in the failure.

Final line is the sglang launcher observing the child process died.

## Symptom (Issue 2: posix.stat during is_aot check)

After the Issue 1 subprocess patch was applied, the same test configuration
re-ran and died with a different error during `piecewise_cuda_graph_runner.warmup_compile`.
Verbatim stack from `sglang-head-56cc54b554-l8vmd`:

```
Compiling num tokens (num_tokens=8192):   0%|          | 0/58 [00:00<?, ?it/s]/usr/local/lib/python3.12/dist-packages/torch/_dynamo/variables/functions.py:2082: UserWarning: Dynamo does not know how to trace the builtin `posix.stat.`
[2026-04-15 08:10:01 TP0] Piecewise CUDA Graph failed with error: Attempted to call function marked as skipped
  Explanation: Dynamo does not know how to trace the builtin `posix.stat.` This function is either a Python builtin (e.g. _warnings.warn) or a third-party C/C++ Python extension (perhaps created with pybind).
  ...
  Developer debug context: module: posix, qualname: stat, skip reason: <missing reason>
  For more details about this graph break, please visit: https://meta-pytorch.github.io/compile-graph-break-site/gb/gb0007.html

from user code:
   File "/usr/local/lib/python3.12/dist-packages/sglang/srt/models/glm4_moe.py", line 1069, in forward
     hidden_states, residual = layer(
   File "/usr/local/lib/python3.12/dist-packages/sglang/srt/models/glm4_moe.py", line 907, in forward
     hidden_states = self.mlp(
   File "/usr/local/lib/python3.12/dist-packages/sglang/srt/models/glm4_moe.py", line 174, in forward
     gate_up, _ = self.gate_up_proj(x)
   File "/usr/local/lib/python3.12/dist-packages/sglang/srt/layers/linear.py", line 460, in forward
     output_parallel = self.quant_method.apply(self, input_, bias)
   File "/usr/local/lib/python3.12/dist-packages/sglang/srt/layers/quantization/modelopt_quant.py", line 1482, in apply
     x_fp4, x_scale_interleaved = fp4_quantize(x, layer.input_scale_inv)
   File "/usr/local/lib/python3.12/dist-packages/flashinfer/quantization/fp4_quantization.py", line 700, in fp4_quantize
     x_q, sf = get_fp4_quantization_module(f"{major}{minor}").fp4_quantize_sm100(
   File "/usr/local/lib/python3.12/dist-packages/torch/_dynamo/polyfills/__init__.py", line 392, in getattr_and_trace
     return fn(*args[2:], **kwargs)
   File "/usr/local/lib/python3.12/dist-packages/flashinfer/quantization/fp4_quantization.py", line 170, in get_fp4_quantization_module
     module = backend_modules[backend]().build_and_load()
   File "/usr/local/lib/python3.12/dist-packages/flashinfer/jit/core.py", line 310, in build_and_load
     if self.is_aot:
   File "/usr/local/lib/python3.12/dist-packages/flashinfer/jit/core.py", line 261, in is_aot
     return self.aot_path.exists()
   File "/usr/lib/python3.12/pathlib.py", line 862, in exists
     self.stat(follow_symlinks=follow_symlinks)
   File "/usr/lib/python3.12/pathlib.py", line 842, in stat
     return os.stat(self, follow_symlinks=follow_symlinks)

torch._dynamo.exc.Unsupported: Attempted to call function marked as skipped
```

Same call site (`fp4_quantize` → `get_fp4_quantization_module` →
`build_and_load`), same traced context (`piecewise_cuda_graph_runner.warmup_compile`
→ sglang's `compile.py:192 trampoline` → `torch._dynamo.eval_frame`), but
this time dynamo chokes on `os.stat` (via `pathlib.Path.exists`) instead of
`subprocess.check_output`. Without the Issue 1 subprocess patch, execution
wouldn't even reach this code path — Issue 2 was latent underneath Issue 1.

This failure mode is caught more gracefully than Issue 1: sglang's piecewise
CUDA graph runner catches the `torch._dynamo.exc.Unsupported` exception and
prints the helpful `To work around this error, add --disable-piecewise-cuda-graph`
hint before the scheduler process exits. The outcome is still
`startup_crash` (the whole worker goes down), but no sigquit / pod restart
loop.

## Root cause

The un-patched upstream code in `flashinfer/jit/cpp_ext.py`:

```python
@functools.cache
def get_cuda_version() -> Version:
    # Try to query nvcc for CUDA version; if nvcc is unavailable, fall back to torch.version.cuda
    try:
        cuda_home = get_cuda_path()
        nvcc = os.path.join(cuda_home, "bin/nvcc")
        txt = subprocess.check_output([nvcc, "--version"], text=True)
        matches = re.findall(r"release (\d+\.\d+),", txt)
        if not matches:
            raise RuntimeError(
                f"Could not parse CUDA version from nvcc --version output: {txt}"
            )
        return Version(matches[0])
    except (RuntimeError, FileNotFoundError, subprocess.CalledProcessError) as e:
        # NOTE(Zihao): when nvcc is unavailable, fall back to torch.version.cuda
        if torch.version.cuda is None:
            raise RuntimeError(
                "nvcc not found and PyTorch is not built with CUDA support. "
                "Could not determine CUDA version."
            ) from e
        return Version(torch.version.cuda)


def is_cuda_version_at_least(version_str: str) -> bool:
    return get_cuda_version() >= Version(version_str)
```

The `torch.version.cuda` fallback is already present — it just only runs **on
exception**. On the happy path, `subprocess.check_output` is called, and this
is what explodes when reached from inside a dynamo trace context.

Dynamo's polyfills for unsupported builtins/library calls try to hoist the
side effect out of the traced region. For `subprocess.Popen`, the polyfill
can't model the fork/clone/pipe setup, so it raises a `Unsupported` exception
that bubbles up in a way that leaves a zombied child (hence `self._waitpid_lock
= threading.Lock()` as the last Python frame — dynamo aborted mid-`__init__`).
The sglang launcher picks up the child-gone signal and restarts the pod.

## Reproduction

Minimal repro should be:

```python
import torch
import flashinfer.quantization.fp4_quantization as fp4q

x = torch.randn(128, 4096, dtype=torch.bfloat16, device="cuda")
scale = torch.tensor(1.0, device="cuda")

@torch.compile(fullgraph=True, backend="inductor")
def f(x, scale):
    return fp4q.fp4_quantize(x, scale)

f(x, scale)  # first call in a fresh process, from inside a compiled region
```

The first call triggers `gen_fp4_quantization_sm120f_module.build_and_load()`,
which calls `is_cuda_version_at_least` while dynamo is tracing `f`, which
calls the subprocess, which dies.

Workaround for the repro (same as our fix): pre-warm the cache by calling
`fp4q.fp4_quantize(x, scale)` once **outside** the `@torch.compile` region, or
call `flashinfer.jit.cpp_ext.get_cuda_version()` at module-import time before
any compiled function runs.

## Our workarounds

`sglang_launch.sh` contains two startup-time patches, both applied at pod
startup before `python3 -m sglang.launch_server` is invoked. Together they
remove flashinfer's JIT build chain from any dynamo-traced code path.

### Patch 1 — `get_cuda_version` subprocess bypass

Rewrites the body of `flashinfer/jit/cpp_ext.py:get_cuda_version()` to return
`Version(torch.version.cuda)` directly, leaving the original subprocess path
as a never-taken fallback. Key properties:

- **Source file**: `/usr/local/lib/python3.12/dist-packages/flashinfer/jit/cpp_ext.py`
- **Marker**: `# [patch] _fi_cuda_ver_subprocess_bypass_` — idempotent check
  on re-runs.
- **Grep guard**: patch only applies if the exact pre-patch function signature
  (including the `# Try to query nvcc...` comment and the `try:` line) is
  present. If flashinfer upstream ever rewrites this function, the patch
  silently skips and prints a warning instead of corrupting the file.
- **Fallback path preserved**: the `try`/`except` block is still in the file
  below the short-circuit, so the patch survives a hypothetical flashinfer
  version bump that adds more validation — as long as the signature comment
  doesn't move, the new code just runs *before* the old code.

See the patch block in `sglang_launch.sh` (look for `PATCH_FI_CUDA_VER_EOF`).
Patched file contents, conceptually:

```python
@functools.cache
def get_cuda_version() -> Version:
    # [patch] _fi_cuda_ver_subprocess_bypass_
    # Short-circuit with torch.version.cuda to avoid spawning a `nvcc --version`
    # subprocess from inside a torch.compile/dynamo trace context.
    if torch.version.cuda is not None:
        return Version(torch.version.cuda)
    # Try to query nvcc for CUDA version; if nvcc is unavailable, fall back to torch.version.cuda
    try:
        ...  # original subprocess path, now unreachable on normal PyTorch builds
```

`is_cuda_version_at_least()` is unchanged — it still calls `get_cuda_version()`
by module-global name lookup, so the patched version takes effect automatically.

### Patch 2 — `get_fp4_quantization_module` cache + import-time pre-warm

Appends an idempotent block to the end of
`flashinfer/quantization/fp4_quantization.py` that:

1. Wraps `get_fp4_quantization_module` in `functools.cache`. The call site
   inside `fp4_quantize()` uses module-global name lookup, so the wrapped
   version takes effect automatically.
2. Runs `get_fp4_quantization_module(f"{sm_major}{sm_minor}")` once, at
   module import time, to force the `build_and_load()` chain to execute.
   Module imports are never inside a dynamo trace, so the full JIT build
   runs normally — compile-time nvcc invocation, `is_aot` stat check, disk
   I/O, everything.
3. Populates the `functools.cache` with the built backend module object as
   a side effect. From here on, every call to
   `get_fp4_quantization_module(sm)` in this process returns the cached
   object instantly, with no further filesystem or subprocess work.

Properties:

- **Source file**: `/usr/local/lib/python3.12/dist-packages/flashinfer/quantization/fp4_quantization.py`
- **Marker**: `# [patch] _fi_fp4_cache_and_prewarm_`
- **Grep guard**: checks for the marker (already-patched) AND for
  `def get_fp4_quantization_module(` (pattern still exists). Silently skips
  with a warning if either check fails.
- **Safe-fail pre-warm**: if CUDA is unavailable at import time, or the
  `get_fp4_quantization_module()` call raises, the patch logs a stderr
  warning and continues — the wrapping of the function still takes effect,
  so lazy init can still happen on first call (same as upstream behavior).

See the patch block in `sglang_launch.sh` (look for `PATCH_FI_FP4_PREWARM_EOF`).
Conceptually, the appended block looks like:

```python
import functools as _sglang_functools
if not hasattr(get_fp4_quantization_module, "__wrapped__"):
    get_fp4_quantization_module = _sglang_functools.cache(get_fp4_quantization_module)

def _sglang_prewarm_fp4_quantization_module():
    try:
        import torch as _t
        if not _t.cuda.is_available():
            return
        _p = _t.cuda.get_device_properties(0)
        get_fp4_quantization_module(f"{_p.major}{_p.minor}")
    except Exception as _e:
        import sys as _sys
        print(f"[fp4_quantization prewarm] skipped: {_e}", file=_sys.stderr)

_sglang_prewarm_fp4_quantization_module()
del _sglang_prewarm_fp4_quantization_module
```

### Why both patches are needed

Without Patch 1, Patch 2's pre-warm call (`get_fp4_quantization_module(sm)`)
fails at import time with the `subprocess.check_output` crash — the Python
module import can run subprocess fine, but flashinfer's build chain calls
`is_cuda_version_at_least` which goes through `get_cuda_version` →
`subprocess.check_output`. Wait, that's actually FINE during import because
we're not in a dynamo trace. But Patch 1 is still needed because sglang's
forward eventually re-calls `is_cuda_version_at_least` *again* via some
other path, and that re-call IS in a dynamo trace. So Patch 1 prevents any
call to `get_cuda_version` from ever running subprocess, and Patch 2 prevents
any call to `get_fp4_quantization_module` from ever doing disk I/O after the
first pre-warm. Together they fully decouple flashinfer's lazy init from
dynamo tracing.

## Upstream status

**No known open PR or issue for this specific failure mode.** Adjacent work:

- **flashinfer `get_cuda_version` history**: the subprocess-based implementation
  was introduced to support CUDA version gates for feature flags like
  `-DENABLE_FP4` (gated on CUDA ≥ 12.8). Previous versions used
  `torch.version.cuda` directly, which is what our patch reverts to. A clean
  upstream fix would either:
  1. Call `get_cuda_version()` at module import time to warm the cache
     unconditionally, or
  2. Change the function body to prefer `torch.version.cuda` when available
     and only fall back to `nvcc --version` when `torch.version.cuda is None`
     (i.e., reverse the try/except priority — exactly what our runtime patch does).

- **torch.compile + subprocess** is a general dynamo limitation, not a torch
  bug — dynamo explicitly documents that side-effecting calls like
  `subprocess.Popen` are "unsupported". Any library that calls subprocess
  lazily from its hot path will eventually trip this when used from a
  `@torch.compile`'d function.

**Report to file**: `flashinfer/issues` with the minimal repro above. Option
(2) is the preferred fix (one-line reorder, no behavior change on happy path,
fixes torch.compile compatibility).

## Relationship to other bugs

- **Orthogonal to** `SGLANG_NVFP4_SHUFFLE_ROWS_OOB_UPSTREAM_BUG.md` — that one
  is about `cutlass_moe_fp4` MoE dispatch under EP. This one is purely about
  flashinfer's CUDA-version detection in any FP4 quantize call.
- **Orthogonal to** the cuDNN missing-dep issue (`scripts/build_cudnn_image.sh`
  docstring) — that was a pip-package shipping problem. This one is a
  code/tracing interaction.
- **Explains the "piecewise crashes" rule on GLM-4.7 EP=1**: prior to this
  analysis, all 12 `disable_piecewise_cuda_graph=false` variants in the
  GLM-4.7 EP=1 matrix crashed at startup (see
  `TESTLOGS/sglang_nn4_tp4_ep1/glm-4.7-nvfp4/TESTLOG_nv580.142_sglang-0.5.10_glm-4.7-nvfp4_4n.md`).
  The original diagnosis was "piecewise path is broken". This bug document
  provides the actual mechanism: piecewise graph capture triggers torch.compile
  on the forward, which traces through `fp4_quantize`, which hits the
  subprocess spawn. With the patch applied, the piecewise configs **may** work
  — pending a matrix re-run.
- **Latent on non-piecewise configs**: technically the same failure path is
  *reachable* on non-piecewise + CG-on configs, but in practice `get_cuda_version`
  gets called during module loading (before graph capture) and the
  `@functools.cache` result is warm by the time graph capture starts. Only
  piecewise+fi_cudnn managed to reorder the calls so the first subprocess
  invocation landed inside a trace context.

## Test matrix impact (GLM-4.7-NVFP4 EP=1)

Before the patch, with `xomoxcc/dgx-spark-sglang:0.5.10-cudnn`:

| # | MoE | Attn | FP4 | Pcw | Outcome (pre-patch) |
|---|-----|------|-----|-----|---------------------|
| 9  | triton     | fi     | fi_cudnn | ✓ | startup_crash (this bug) |
| 12 | triton     | triton | fi_cudnn | ✓ | startup_crash (this bug) |
| 21 | fi_cutlass | fi     | fi_cudnn | ✓ | startup_crash (this bug + fi_cutlass MoE EP=1) |
| 24 | fi_cutlass | triton | fi_cudnn | ✓ | startup_crash (this bug + fi_cutlass MoE EP=1) |
| 33 | cutlass    | fi     | fi_cudnn | ✓ | startup_crash (this bug) |
| 36 | cutlass    | triton | fi_cudnn | ✓ | startup_crash (this bug) |

Post-patch expectation: tests 9, 12, 33, 36 should become STABLE (they only
had this bug blocking them); tests 21 and 24 should continue to crash for the
independent `fi_cutlass` MoE EP=1 dispatch bug documented in
`SGLANG_NVFP4_SHUFFLE_ROWS_OOB_UPSTREAM_BUG.md`. Needs a matrix re-run with
the patched `sglang_launch.sh` on all 4 sparks (re-deploy via
`ansible-playbook k8s_dgx.yml --tags sglang`) to confirm.

## Files

- `roles/k8s_dgx/files/sglang_launch.sh` — both runtime monkey-patches
  (`PATCH_FI_CUDA_VER_EOF` for Issue 1, `PATCH_FI_FP4_PREWARM_EOF` for
  Issue 2). Both live in the flashinfer-patch block around line ~175.
- `/usr/local/lib/python3.12/dist-packages/flashinfer/jit/cpp_ext.py` —
  Issue 1 patch target (`get_cuda_version` function body).
- `/usr/local/lib/python3.12/dist-packages/flashinfer/quantization/fp4_quantization.py`
  — Issue 2 patch target (append block at end-of-file). Also the call site
  where `fp4_quantize` / `get_fp4_quantization_module` are defined.
- `/usr/local/lib/python3.12/dist-packages/flashinfer/jit/core.py` — where
  `JitSpec.build_and_load` and `JitSpec.is_aot` live. Not directly patched,
  but their behavior is neutralized by Patch 2's cache (they never run again
  after the pre-warm).

---

## Update 2026-04-15 evening: Patch 2 was incomplete, deeper issues uncovered

The original Patch 2 ("`functools.cache` wrap + import-time prewarm") **does not
actually prevent** the dynamo tracing failure it was supposed to fix. It only
worked on paper. Re-running the GLM-4.7-NVFP4 EP=1 piecewise + `fi_cudnn` matrix
on `xomoxcc/dgx-spark-sglang:0.5.10-sm121` with Patch 2 applied reproduced the
`posix.stat` error **unchanged**. Debug-walking it layer by layer uncovered an
entire family of dynamo-incompatible behaviours inside the FP4 quantize
codepath, each one revealed only after the previous one was bypassed.

### Why the cache-wrap approach failed

Dynamo's `polyfills.getattr_and_trace` (triggered by the `.fp4_quantize_sm100`
attribute access on the *return value* of `get_fp4_quantization_module(...)`)
**inlines and re-traces the wrapped function body every time**, bypassing
`functools.cache` entirely. So the cache wrap was invisible to dynamo:

- `get_fp4_quantization_module` is already `@functools.cache`-decorated upstream.
  Wrapping it again changed nothing.
- The import-time prewarm populated the cache, but dynamo never consulted it
  during trace — it traced straight through `build_and_load` → `is_aot` →
  `Path.exists` → `os.stat` as if no cache existed.
- Verified empirically: the stat error landed at the exact same line,
  byte-for-byte identical to the pre-patch traceback.

### Failure chain (each layer unblocked, next layer revealed)

Four distinct unpatched failures sit stacked inside `fp4_quantize`, each
reachable only after fixing the prior one. Chronological order as discovered:

| # | Location | Dynamo verdict | Root cause |
|---|----------|----------------|------------|
| 2a | `flashinfer/jit/core.py:261 is_aot → pathlib.Path.exists → os.stat` | `Attempted to call function marked as skipped: posix.stat` (gb0007) | `posix.stat` is a C builtin, not traceable by dynamo |
| 2b | `flashinfer/quantization/fp4_quantization.py:222 fp4_quantize_sm100 → module.fp4_quantize(...)` | `Unsupported method call: Function.__call__` (gb0156) | The `module.fp4_quantize` returned by the JIT-built backend is a `torch.autograd.Function` subclass; dynamo cannot trace its `__call__` dispatch |
| 2c | `flashinfer/quantization/fp4_quantization.py:700 fp4_quantize` wrapped with `@torch.compiler.disable` | `Skip calling torch.compiler.disable()d function` (gb0098) | sglang's `PiecewiseCudaGraphRunner.warmup_compile` treats disabled calls as a hard error rather than a graph break |
| 2d | same, wrapped with `torch.compiler.allow_in_graph(fp4_quantize)` | `Dynamo failed to run FX node with fake tensors: call_function <function fp4_quantize>(FakeTensor(...), Parameter(FakeTensor(...)))` | `allow_in_graph` requires a meta/fake-kernel implementation so dynamo can propagate shapes/dtypes; `fp4_quantize` has none and cannot run on FakeTensors |

### Attempt log

All attempts to patch at the flashinfer-source level via `sglang_launch.sh`:

1. **`functools.cache` + import-time prewarm** — `_fi_fp4_cache_and_prewarm_`
   marker. No effect on dynamo re-tracing; identical `os.stat` failure.
2. **Pre-resolve the FP4 backend into a module-level constant `_SGLANG_FP4_MOD`**
   and rewrite the hot-path call to `_SGLANG_FP4_MOD.fp4_quantize_sm100(...)` —
   `_fi_fp4_prewarm_const_` marker. Eliminated the `get_fp4_quantization_module`
   lookup from the traced region, unblocked failure 2a, but hit failure 2b
   (`Function.__call__` inside `fp4_quantize_sm100`).
3. **`@torch.compiler.disable` decorator on `fp4_quantize`** —
   `_fi_fp4_compiler_disable_` marker. Sidestepped tracing into the body
   entirely, but sglang's piecewise compile path treats skipped calls as
   gb0098-unsupported (failure 2c).
4. **`torch.compiler.allow_in_graph(fp4_quantize)` at module-import time** —
   `_fi_fp4_allow_in_graph_` marker. Progressed furthest: dynamo accepts the
   opaque leaf node, gb0098 is gone, but fake-tensor propagation through the
   node fails with `RuntimeError when making fake tensor call` (failure 2d).

Currently on disk: revision 4 (`allow_in_graph`), which is itself broken.

### Verification via Loki

Each revision shows up cleanly in Loki logs and can be attributed to a
specific pod generation by the patch-log-line they emit at container startup
(`Patched flashinfer/quantization/fp4_quantization.py: ...`). Query example:

```logql
{namespace="sglang"} |~ "fp4_quantize|Piecewise CUDA Graph|posix.stat|Function\\.__call__|allow_in_graph"
```

Cross-referencing the error timestamps against the `Patched ...` startup line
gives an unambiguous "which revision produced which error" timeline per pod.

### Remaining options

1. **`torch.library.custom_op` + `register_fake`**: the canonical dynamo-safe
   path. Define a sglang-private op `sglang_patch::fp4_quantize` whose real
   kernel delegates to the original `fp4_quantize` and whose fake kernel
   computes output shapes manually:
   - `x_q` shape = `(M, K // 2)`, dtype = `torch.float4_e2m1fn_x2` (or `uint8`
     if the dtype isn't exposed in this torch build).
   - `sf` shape = `(round_up(M, 128), round_up(K // sf_vec_size, 4))`, dtype
     = `torch.uint8` (or `torch.float8_e4m3fn` depending on layout).
   - `sf_vec_size` and layout flags are hardcoded to sglang's default
     (`16`, `is_sf_swizzled_layout=True`) — sglang's `modelopt_quant.py:1482`
     only calls `fp4_quantize(x, layer.input_scale_inv)` positionally.
   - Rebind `flashinfer.quantization.fp4_quantization.fp4_quantize =
     sglang_patch_fp4_quantize` at module-import time.
   Pros: preserves piecewise CUDA graphs, robust if the shape formulas are
   right. Cons: fragile if flashinfer changes output layouts or dtypes; needs
   a test matrix re-run to prove correctness.

2. **`--disable-piecewise-cuda-graph` for NVFP4 profiles**: stop tripping the
   tracer by opting out of piecewise capture for models that hit this path.
   Pros: zero extra patch surface, sglang itself suggests this in the error
   message. Cons: gives up piecewise-compile wins (unquantified on this
   hardware for NVFP4). Non-piecewise `fi_cudnn` runs (tests 7 and 8 in the
   matrix) were green pre-bug, so this unblocks the whole 12-variant matrix
   immediately.

3. **Upstream fixes** in flashinfer: wrap `fp4_quantize` with `@torch.compile`'s
   `custom_op` machinery upstream, OR hoist `is_aot` / `build_and_load` out of
   the lazy path so a traced first-call doesn't trigger filesystem I/O. Either
   fix would solve this cleanly for everyone; neither is in any open PR as of
   2026-04-15.

### Decision pending

Given that non-piecewise `fi_cudnn` variants were already known to work and
that the custom-op shape formulas depend on flashinfer internals that are not
API-stable, the pragmatic unblock is **option 2** — make `disable_piecewise_cuda_graph=true`
the default for NVFP4 model profiles in `roles/k8s_dgx/model_profiles/*.yml`
and move on. Option 1 stays as a future task if piecewise numbers are ever
needed for these models specifically.

No changes made to model profiles yet — waiting for explicit go-ahead before
touching deployment configuration (per standing feedback).

### What to do with the current `sglang_launch.sh` patch

The `allow_in_graph` revision currently in `sglang_launch.sh` is strictly
worse than a no-op: it papers over failures 2a and 2b but introduces 2d. Once
option 2 is chosen, the whole `PATCH_FI_FP4_*` block can be deleted — with
piecewise off, dynamo never enters `fp4_quantize` in the first place, so none
of the tracing issues matter. Patch 1 (`get_cuda_version` subprocess bypass)
stays — it's independent and still needed.

### Files changed in this update cycle

- `roles/k8s_dgx/files/sglang_launch.sh` — `PATCH_FI_FP4_*` block iterated
  through four revisions (markers `_fi_fp4_cache_and_prewarm_` →
  `_fi_fp4_prewarm_const_` → `_fi_fp4_compiler_disable_` → `_fi_fp4_allow_in_graph_`).
  Current file content = revision 4. Each revision includes cleanup logic to
  strip earlier-revision append blocks and source edits, so repeated launches
  within the same container converge to a clean state.
- No commits made yet — current on-disk state is work-in-progress.
