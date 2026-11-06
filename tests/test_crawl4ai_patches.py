"""Tests for roles/k8s_infra/files/crawl4ai/patches/ (usermap hook patches) and crawl4ai_verify_hooks.py.

Tests marked needs_tree run the patch scripts against an unmodified checkout of unclecode/crawl4ai v0.9.5 given in
CRAWL4AI_SRC_TREE (a directory containing deploy/docker). The server is never started.
"""

import ast
import difflib
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
FILES = REPO / "roles" / "k8s_infra" / "files" / "crawl4ai"
PATCHES = FILES / "patches"
PATCHLIB = REPO / "roles" / "k8s_dgx" / "files" / "sglang_patches" / "_patchlib.py"
HOOKED = ["api.py", "auth_gate.py", "llm_broker.py", "mcp_bridge.py", "server.py"]

SRC_ENV = os.environ.get("CRAWL4AI_SRC_TREE")
needs_tree = pytest.mark.skipif(
    not SRC_ENV or not (Path(SRC_ENV) / "deploy" / "docker" / "api.py").is_file(),
    reason="CRAWL4AI_SRC_TREE (clean v0.9.5 checkout) not set",
)

EXPECTED_ADDED = {
    "import crawl4ai_usermap as usermap",
    "factory = usermap.bind(factory)",
    "usermap.precheck_llm(config, provider)",
    'cache_mode = usermap.cache_mode(CacheMode.ENABLED if cache == "1" else CacheMode.WRITE_ONLY, CacheMode.BYPASS)',
    'cache_mode = usermap.cache_mode(CacheMode.ENABLED if cache == "1" else CacheMode.WRITE_ONLY, '
    "CacheMode.BYPASS, filter_type == FilterType.LLM)",
    "await usermap.call_app(self.app, scope, receive, send)",
    "if (user := usermap.authenticate(token)) is not None:",
    "return user",
    '"api_token": usermap.llm_api_token(config, provider, get_llm_api_key(config, provider)),',
    'return usermap.mcp_auth_headers({"Authorization": f"Bearer {token}"})',
    "key_func=usermap.rate_limit_key(get_remote_address),",
    "",
}
EXPECTED_REMOVED = {
    'cache_mode = CacheMode.ENABLED if cache == "1" else CacheMode.WRITE_ONLY',
    "await self.app(scope, receive, send)",
    '"api_token": get_llm_api_key(config, provider),   # server credential',
    'return {"Authorization": f"Bearer {token}"}',
    "key_func=get_remote_address,",
}


def make_patch_dir(dst: Path) -> Path:
    dst.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(PATCHLIB, dst / "_patchlib.py")
    for p in PATCHES.glob("*.py"):
        shutil.copyfile(p, dst / p.name)
    shutil.copyfile(FILES / "crawl4ai_verify_hooks.py", dst / "crawl4ai_verify_hooks.py")
    shutil.copyfile(FILES / "crawl4ai_usermap.py", dst / "crawl4ai_usermap.py")
    return dst


def run_patches(patch_dir: Path, root: Path) -> str:
    out = ""
    for p in sorted(patch_dir.glob("c[0-9][0-9]_*.py")):
        res = subprocess.run(
            [sys.executable, str(p)],
            env={**os.environ, "CRAWL4AI_PATCH_ROOT": str(root)},
            capture_output=True,
            text=True,
            check=False,
        )
        assert res.returncode == 0, res.stderr
        out += res.stdout
    return out


def verify(patch_dir: Path, root: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(patch_dir / "crawl4ai_verify_hooks.py"), str(root)],
        capture_output=True,
        text=True,
        check=False,
    )


@pytest.fixture
def tree(tmp_path: Path) -> Path:
    assert SRC_ENV
    app = tmp_path / "app"
    shutil.copytree(Path(SRC_ENV) / "deploy" / "docker", app, ignore=shutil.ignore_patterns("__pycache__"))
    return app


@pytest.fixture
def patch_dir(tmp_path: Path) -> Path:
    return make_patch_dir(tmp_path / "patches")


def test_patch_scripts_parse_and_follow_naming() -> None:
    names = sorted(p.name for p in PATCHES.glob("c[0-9][0-9]_*.py"))
    expected = ["c10_auth_gate.py", "c20_llm_broker.py", "c30_api.py"]  # gitleaks:allow (file names)
    expected += ["c40_server.py", "c50_mcp_bridge.py"]
    assert names == expected
    for p in PATCHES.glob("*.py"):
        ast.parse(p.read_text())
    ast.parse((FILES / "crawl4ai_verify_hooks.py").read_text())


def test_verifier_reports_missing_hooks(tmp_path: Path, patch_dir: Path) -> None:
    for name in HOOKED:
        (tmp_path / name).write_text("x = 1\n")
    res = verify(patch_dir, tmp_path)
    assert res.returncode == 1
    assert res.stdout.count("HOOK-MISSING") >= 5


@needs_tree
def test_patches_apply_exactly_the_expected_lines(tree: Path, patch_dir: Path) -> None:
    assert SRC_ENV
    out = run_patches(patch_dir, tree)
    assert "ANCHOR-DRIFT" not in out and out.count("Patched ") == 5
    added: set[str] = set()
    removed: set[str] = set()
    for name in HOOKED:
        before = (Path(SRC_ENV) / "deploy" / "docker" / name).read_text().splitlines()
        after = (tree / name).read_text()
        ast.parse(after)
        for line in difflib.unified_diff(before, after.splitlines(), lineterm="", n=0):
            if line.startswith("+") and not line.startswith("+++"):
                added.add(line[1:].strip())
            elif line.startswith("-") and not line.startswith("---"):
                removed.add(line[1:].strip())
    assert added == EXPECTED_ADDED
    assert removed == EXPECTED_REMOVED
    shutil.copyfile(FILES / "crawl4ai_usermap.py", tree / "crawl4ai_usermap.py")
    assert verify(patch_dir, tree).returncode == 0


@needs_tree
def test_unhooked_files_are_byte_identical(tree: Path, patch_dir: Path) -> None:
    assert SRC_ENV
    run_patches(patch_dir, tree)
    src = Path(SRC_ENV) / "deploy" / "docker"
    for p in src.rglob("*"):
        if p.is_file() and p.name not in HOOKED and "__pycache__" not in p.parts:
            assert (tree / p.relative_to(src)).read_bytes() == p.read_bytes()


@needs_tree
def test_second_run_changes_nothing(tree: Path, patch_dir: Path) -> None:
    run_patches(patch_dir, tree)
    first = {n: (tree / n).read_bytes() for n in HOOKED}
    out = run_patches(patch_dir, tree)
    assert "ANCHOR-DRIFT" not in out and "already applied" in out and "Patched " not in out
    assert {n: (tree / n).read_bytes() for n in HOOKED} == first


@needs_tree
@pytest.mark.parametrize(
    "name,old",
    [
        ("auth_gate.py", "# 2) HS256 JWT minted by this server"),
        ("llm_broker.py", "get_llm_api_key(config, provider),   #"),
        ("api.py", "from work_queue import get_job_queue, QueueFull"),
        ("server.py", "key_func=get_remote_address"),
        ("mcp_bridge.py", '"Authorization": f"Bearer {token}"'),
    ],
)
def test_moved_anchor_is_drift_and_leaves_file_untouched(tree: Path, patch_dir: Path, name: str, old: str) -> None:
    path = tree / name
    text = path.read_text()
    assert old in text
    path.write_text(text.replace(old, old + "_moved", 1))
    before = path.read_bytes()
    out = run_patches(patch_dir, tree)
    assert "ANCHOR-DRIFT" in out
    assert path.read_bytes() == before


@needs_tree
def test_ambiguous_anchor_is_drift(tree: Path, patch_dir: Path) -> None:
    path = tree / "server.py"
    text = path.read_text()
    path.write_text(text + "\n" + "limiter2 = Limiter(\n    key_func=get_remote_address,\n)\n")
    before = path.read_bytes()
    out = run_patches(patch_dir, tree)
    assert "ANCHOR-DRIFT" in out and "ambiguous" in out
    assert path.read_bytes() == before
