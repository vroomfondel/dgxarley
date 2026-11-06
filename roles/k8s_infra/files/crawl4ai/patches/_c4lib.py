"""[dgxarley] Glue between the SGLang patch library and the Crawl4AI patches.

Reuses roles/k8s_dgx/files/sglang_patches/_patchlib.py unchanged (Patch, AnchorDrift, same contract: never raise, check
"already applied" before the anchor, all-or-nothing per file, idempotent). Its Patch resolves `target` with
os.path.join(DIST_PACKAGES, target), so an absolute path under CRAWL4AI_PATCH_ROOT works as is.

Added here: a unique-anchor edit (an anchor that matches twice is drift, not "first hit wins").
"""

import os

from _patchlib import AnchorDrift, Patch  # type: ignore[import-not-found,unused-ignore]

__all__ = ["AnchorDrift", "IMPORT_LINE", "Patch", "ROOT", "edit", "patch_for"]
ROOT = os.environ.get("CRAWL4AI_PATCH_ROOT", "/app")
IMPORT_LINE = "import crawl4ai_usermap as usermap\n"


def patch_for(name: str, filename: str) -> Patch:
    return Patch(name=name, target=os.path.join(ROOT, filename))


def edit(p: Patch, old: str, new: str, marker: str, what: str) -> None:
    if marker in p.code:
        return
    hits = p.code.count(old)
    if hits != 1:
        raise AnchorDrift(f"{what} anchor {'missing' if hits == 0 else f'ambiguous ({hits} hits)'}")
    p.replace(old, new, marker=marker, what=what)
