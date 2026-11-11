"""[dgxarley] CUTLASS DSL mma.py: add sm_120a + sm_121a to BlockScaledMmaOp.admissible_archs.

Patch CUTLASS BlockScaledMmaOp to support SM121 (DGX Spark GB10) for FP4 operations.
Upstream CUTLASS restricts FP4 tensor ops to sm_100a only (issue NVIDIA/cutlass#2800).
SM121 has native FP4 Tensor Core support but is not in admissible_archs, so the
JIT-compiled nvfp4_blockwise_moe kernel falls back to an incompatible code path and
trips a device-side assert.
Fix: add sm_120a + sm_121a to admissible_archs in both CUTLASS DSL copies.
External validation: BTankut/dgx-spark-sglang-moe-configs achieved 356 TFLOPS NVFP4 on GB10.

Targets are NOT under sglang: CUTLASS DSL ships in two places and either may be the
one the JIT actually loads, so both get patched when present:
  1. nvidia_cutlass_dsl/...  (the standalone wheel)
  2. flashinfer/data/cutlass/...  (the copy bundled inside flashinfer)
On the 0.5.15-sm121 image only copy 2 exists.

Two deliberate deviations from the original bash, both documented rather than silent:

* A MISSING copy is not an error. The original only warned when NEITHER copy was
  found, and said nothing about a single missing one. A plain Patch() per path would
  print "target file missing" ANCHOR-DRIFT for the absent nvidia_cutlass_dsl copy on
  every boot, which is noise, not drift. So the paths are filtered by existence first
  and the "neither found" warning is kept verbatim.
* The original's outer guard grepped `admissible_archs = [` and then sed-replaced
  `Arch.sm_100a,`. If the former existed but the latter did not, sed silently changed
  nothing while still logging "Patched". Here the replaced text IS the anchor, so that
  case now reports ANCHOR-DRIFT instead of lying. Same tree, better diagnostic.

replace_all: the original used `sed -i 's/Arch\\.sm_100a,/.../'`, which substitutes the
first match on EVERY line, not just the first in the file. The pristine image has
exactly 1 occurrence, so this is equivalent today, but replace_all is what sed meant.

Re-sync: drop this file once CUTLASS admits sm_121a upstream (the sm_121a marker guard
makes it a no-op then).
"""

import os

from _patchlib import DIST_PACKAGES, Patch

MMA_PATHS = (
    "nvidia_cutlass_dsl/python_packages/cutlass/cute/nvgpu/tcgen05/mma.py",
    "flashinfer/data/cutlass/python/CuTeDSL/cutlass/cute/nvgpu/tcgen05/mma.py",
    # nvidia-cutlass-dsl >= 4.8: python sources moved to dsl_packages, `Arch` is spelled base_dsl.Arch.
    "nvidia_cutlass_dsl/dsl_packages/cutlass/cute/nvgpu/tcgen05/mma.py",
)

OLD_ARCHS = "Arch.sm_100a,"
NEW_ARCHS = "Arch.sm_100a, Arch.sm_120a, Arch.sm_121a,"
MARKER = "sm_121a"

OLD_ARCHS_V48 = "base_dsl.Arch.sm_100a,\n        base_dsl.Arch.sm_100f,"
NEW_ARCHS_V48 = "base_dsl.Arch.sm_100a,\n        base_dsl.Arch.sm_120a,\n        base_dsl.Arch.sm_121a,\n        base_dsl.Arch.sm_100f,"


def _patch_copy(relpath: str) -> None:
    patch = Patch(name="BlockScaledMmaOp admissible_archs += sm_120a/sm_121a", target=relpath)

    @patch.run
    def apply(p: Patch) -> None:
        if "base_dsl.Arch.sm_100a," in p.code:
            p.replace(OLD_ARCHS_V48, NEW_ARCHS_V48, marker=MARKER)
        else:
            p.replace_all(OLD_ARCHS, NEW_ARCHS, marker=MARKER)


_existing = [rel for rel in MMA_PATHS if os.path.isfile(os.path.join(DIST_PACKAGES, rel))]
for _rel in _existing:
    _patch_copy(_rel)

if not _existing:
    print(
        "ANCHOR-DRIFT: cutlass mma.py: neither CUTLASS DSL copy "
        "(nvidia_cutlass_dsl / flashinfer bundled) found (dep restructured/renamed?)"
    )
