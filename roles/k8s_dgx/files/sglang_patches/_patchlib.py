"""[dgxarley] Shared helpers for the SGLang runtime source patches.

Every `p<NN>_*.py` next to this file is a standalone patch against the SGLang
install in the container's dist-packages. `sglang_launch.sh` runs them all, in
filename order, before starting the server. This module holds the boilerplate
that used to be copy-pasted into every `python3 - <<'PATCH_*_EOF'` heredoc:
target resolution, the already-applied guard, the anchor-drift reporting and
the write-back.

Contract every patch relies on:

* **Never raise, never exit non-zero.** A drifted anchor is a warning, not a
  crash: the launcher runs under `set -e` and an exception here would crashloop
  the pod. Patches degrade to "unpatched SGLang", which is the same behaviour
  the inline heredocs had.
* **Already-applied is checked FIRST**, before the anchor. `new` frequently
  contains `old` as a prefix (we mostly append to an anchor), so an
  `old in code` check would re-apply on a re-run. That exact bug bit the
  buffered-safetensors patch on 2026-07-16.
* **All-or-nothing per file.** Edits are buffered in memory and written once at
  the end; if any edit in the patch drifts, nothing is written. A file
  half-patched by a partially-drifted multi-edit patch is far worse to debug
  than an unpatched one.
* **Idempotent.** Running a patch twice must not change the file the second
  time. The runner is not transactional, and a pod restart re-runs everything.

Adding a patch: copy the shape of `p20_moe_wna16_qzeros_ep.py`. The module
docstring carries the knowledge (why it exists, upstream status, when it can be
deleted); do not add a patch without one.
"""

import os
from collections.abc import Callable

# The SGLang install inside the container image. Single source of truth: patches
# name their target relative to this, so an image that moves dist-packages needs
# one edit here rather than 30.
DIST_PACKAGES = "/usr/local/lib/python3.12/dist-packages"


class AnchorDrift(Exception):
    """An anchor no longer matches the shipped SGLang source.

    Raised by the edit helpers, caught by `Patch.run`, which turns it into an
    ANCHOR-DRIFT line and skips the write. Patches should not catch it.
    """


def gate_model(*needles: str) -> bool:
    """True when SGLANG_MODEL contains any of `needles` (the model-name gate)."""
    model = os.environ.get("SGLANG_MODEL", "")
    return any(needle in model for needle in needles)


def gate_env(name: str, value: str) -> bool:
    """True when env var `name` is exactly `value`."""
    return os.environ.get(name, "") == value


class Patch:
    """One patch against one SGLang source file.

    `target` is relative to DIST_PACKAGES. `when` is the gate: False means the
    patch does not apply to this model/config and is skipped with one log line
    (this replaces the bash `if` that used to wrap the heredoc, so gate and
    patch now live in the same file).
    """

    def __init__(self, name: str, target: str, when: bool = True) -> None:
        self.name = name
        self.target = target
        self.when = when
        self.path = os.path.join(DIST_PACKAGES, target)
        self.basename = os.path.basename(target)
        self._code = ""
        self._changed = False

    def replace(self, old: str, new: str, marker: str | None = None, what: str | None = None) -> None:
        """Replace the first occurrence of `old` with `new`.

        `marker` is the already-applied probe; it defaults to `new`, which is
        correct whenever `new` is unique to the patched state. Pass an explicit
        marker when `new` is not a reliable probe: when two patches inject the
        same string, or when the injected text is not unique in the file. Both
        cases have burned us, hence the parameter.
        """
        label = what or self.name
        probe = marker if marker is not None else new
        if probe in self._code:
            return
        if old not in self._code:
            raise AnchorDrift(f"{label} anchor missing")
        self._code = self._code.replace(old, new, 1)
        self._changed = True

    def insert_after(self, anchor: str, text: str, marker: str, what: str | None = None) -> None:
        """Insert `text` right after the first occurrence of `anchor`.

        `marker` is mandatory here: the injected text is appended to the anchor,
        so it is never a safe default probe on its own.
        """
        self.replace(anchor, anchor + text, marker=marker, what=what)

    def run(self, fn: Callable[["Patch"], None]) -> Callable[["Patch"], None]:
        """Decorator: run `fn` against this patch's file and write back.

        Used as `@patch.run` on the patch body, so the module reads
        declaratively top to bottom and the file is executed on import as a
        script. Returns `fn` unchanged so the decorated name stays callable
        (handy in tests).
        """
        if not self.when:
            print(f"[patch] {self.name}: gate not matched, skipping")
            return fn
        if not os.path.isfile(self.path):
            print(f"ANCHOR-DRIFT: {self.basename}: {self.name} target file missing (SGLang restructured/renamed?)")
            return fn
        with open(self.path) as fh:
            self._code = fh.read()
        try:
            fn(self)
        except AnchorDrift as exc:
            print(f"ANCHOR-DRIFT: {self.basename}: {exc} (SGLang version drift; re-check anchor)")
            return fn
        if not self._changed:
            print(f"[patch] {self.basename}: {self.name} already applied, skipping")
            return fn
        with open(self.path, "w") as fh:
            fh.write(self._code)
        print(f"Patched {self.basename}: {self.name}")
        return fn
