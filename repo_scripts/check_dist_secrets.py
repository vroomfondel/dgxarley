#!/usr/bin/env python3
"""Fail the build if a distribution artifact contains files that must never ship.

Background: this repo is an Ansible repo *and* a Python package, so most of the
working tree is deployment config rather than package source. A build backend
that ships "everything not gitignored" therefore packs far more than the package.

pyproject.toml pins an explicit allowlist for both build targets. This script is
the backstop that verifies the *built artifacts*, so a future edit to the build
config, or a switch of build backend, cannot silently widen what gets published.

Two independent checks per artifact:

  1. git-crypt check: no member may match a path pattern that .gitattributes
     marks with `filter=git-crypt`. The patterns are read from .gitattributes at
     runtime, so a newly protected path is covered without touching this script.
  2. allowlist check: every member must sit under a known-good top-level entry.
     This is the check that actually holds the line, since it also catches
     secret-bearing files that were never git-crypt protected in the first place.

Usage:
    check_dist_secrets.py [ARTIFACT ...]

With no arguments it checks every .tar.gz and .whl in ./dist. Exit code 0 means
the artifacts are clean, 1 means at least one offending member was found.
"""

import fnmatch
import re
import sys
import tarfile
import zipfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

# Top-level entries an artifact is allowed to contain. A name ending in "/" is a
# directory prefix, everything else must match a member path exactly.
SDIST_ALLOWED: tuple[str, ...] = (
    "dgxarley/",
    "tests/",
    ".gitignore",
    "LICENSE.md",
    "PKG-INFO",
    "pyproject.toml",
    "README_pypi.md",
)

WHEEL_ALLOWED: tuple[str, ...] = (
    "dgxarley/",
    "dgxarley-*.dist-info/",
)

# Stripped from sdist member paths before matching ("dgxarley-X.Y.Z/foo" -> "foo").
SDIST_PREFIX_RE = re.compile(r"^[^/]+-[0-9][^/]*/")


def gitcrypt_patterns() -> list[str]:
    """Return the path patterns .gitattributes hands to the git-crypt filter."""
    attributes = REPO_ROOT / ".gitattributes"
    if not attributes.is_file():
        return []

    patterns: list[str] = []
    for raw in attributes.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "filter=git-crypt" not in line:
            continue
        patterns.append(line.split()[0])
    return patterns


def pattern_to_regex(pattern: str) -> re.Pattern[str]:
    """Translate a gitattributes path pattern into an anchored regex.

    Handles the three wildcards that matter here: `**` spans directory
    separators, `*` and `?` do not. A pattern without a slash matches at any
    depth, which is the gitattributes rule that makes `*.local` cover
    `host_vars/x/y.local` as well.
    """
    if "/" not in pattern:
        pattern = "**/" + pattern

    out: list[str] = []
    i = 0
    while i < len(pattern):
        if pattern.startswith("**/", i):
            out.append("(?:.*/)?")
            i += 3
        elif pattern.startswith("**", i):
            out.append(".*")
            i += 2
        elif pattern[i] == "*":
            out.append("[^/]*")
            i += 1
        elif pattern[i] == "?":
            out.append("[^/]")
            i += 1
        else:
            out.append(re.escape(pattern[i]))
            i += 1
    return re.compile("^" + "".join(out) + "$")


def is_allowed(member: str, allowed: tuple[str, ...]) -> bool:
    for entry in allowed:
        if entry.endswith("/"):
            if fnmatch.fnmatch(member.split("/", 1)[0] + "/", entry):
                return True
        elif member == entry:
            return True
    return False


def artifact_members(path: Path) -> list[str]:
    if path.name.endswith(".whl"):
        with zipfile.ZipFile(path) as zf:
            return [n for n in zf.namelist() if not n.endswith("/")]

    with tarfile.open(path, "r:*") as tf:
        return [SDIST_PREFIX_RE.sub("", m.name) for m in tf.getmembers() if m.isfile()]


def check(path: Path, protected: list[re.Pattern[str]]) -> list[str]:
    """Return the offending members of one artifact (empty list means clean)."""
    allowed = WHEEL_ALLOWED if path.name.endswith(".whl") else SDIST_ALLOWED

    secrets: list[str] = []
    strays: list[str] = []
    for member in artifact_members(path):
        if any(rx.match(member) for rx in protected):
            secrets.append(f"{member}  [git-crypt protected]")
        elif not is_allowed(member, allowed):
            strays.append(f"{member}  [outside allowlist]")

    # git-crypt hits first: those are the ones that actually matter, and the
    # caller truncates the list. A stray README further down is noise.
    return secrets + strays


def main(argv: list[str]) -> int:
    if argv:
        artifacts = [Path(a) for a in argv]
    else:
        dist = REPO_ROOT / "dist"
        artifacts = sorted(dist.glob("*.tar.gz")) + sorted(dist.glob("*.whl"))

    if not artifacts:
        print("check_dist_secrets: no artifacts to check", file=sys.stderr)
        return 1

    protected = [pattern_to_regex(p) for p in gitcrypt_patterns()]
    failed = False

    for path in artifacts:
        if not path.is_file():
            print(f"check_dist_secrets: missing artifact {path}", file=sys.stderr)
            failed = True
            continue

        offenders = check(path, protected)
        if offenders:
            failed = True
            print(f"\nFAIL {path.name}: {len(offenders)} file(s) must not be published", file=sys.stderr)
            for member in offenders[:40]:
                print(f"    {member}", file=sys.stderr)
            if len(offenders) > 40:
                print(f"    ... and {len(offenders) - 40} more", file=sys.stderr)
        else:
            print(f"ok   {path.name}")

    if failed:
        print(
            "\nRefusing to publish. Fix [tool.hatch.build.targets.*] in pyproject.toml,\n"
            "rebuild, and re-run. Do NOT publish an artifact that fails this check.",
            file=sys.stderr,
        )
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
