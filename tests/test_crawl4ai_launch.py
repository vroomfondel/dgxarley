"""Tests for roles/k8s_infra/files/crawl4ai/crawl4ai_launch.sh with a stand-in app tree and stand-in server command.

Needs a clean v0.9.5 checkout in CRAWL4AI_SRC_TREE for the app tree; the server is never started. The user switch is
replaced by a stub through CRAWL4AI_DROP_CMD because the tests do not run as root.
"""

import os
import shutil
import stat
import subprocess
import sys
from pathlib import Path

import pytest

from .test_crawl4ai_patches import FILES, HOOKED, SRC_ENV, make_patch_dir, needs_tree

LAUNCH = FILES / "crawl4ai_launch.sh"


@pytest.fixture
def env_dir(tmp_path: Path) -> Path:
    assert SRC_ENV
    app = tmp_path / "app"
    shutil.copytree(Path(SRC_ENV) / "deploy" / "docker", app, ignore=shutil.ignore_patterns("__pycache__"))
    for p in app.rglob("*"):
        if p.is_file():
            p.chmod(0o444)
    app.chmod(0o555)
    make_patch_dir(tmp_path / "patches")
    (tmp_path / "start.sh").write_text('echo "started pwd=$PWD home=$HOME user=$USER" > "$MARK/started"\n')
    (tmp_path / "drop.sh").write_text('echo "dropped $*" > "$MARK/dropped"\nexec "$@"\n')
    (tmp_path / "mark").mkdir()
    return tmp_path


def launch(base: Path, usermap: str, extra: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
    env = {
        **os.environ,
        "CRAWL4AI_APP_DIR": str(base / "app"),
        "CRAWL4AI_PATCH_DIR": str(base / "patches"),
        "CRAWL4AI_PYTHON": sys.executable,
        "CRAWL4AI_START_CMD": f"bash {base}/start.sh",
        "CRAWL4AI_DROP_CMD": f"bash {base}/drop.sh",
        "CRAWL4AI_USERMAP": usermap,
        "MARK": str(base / "mark"),
        **(extra or {}),
    }
    return subprocess.run(["bash", str(LAUNCH)], env=env, capture_output=True, text=True, check=False, timeout=60)


def snapshot(app: Path) -> dict[str, bytes]:
    return {p.name: p.read_bytes() for p in app.glob("*.py")}


def writable_by_others(p: Path) -> bool:
    return bool(p.stat().st_mode & (stat.S_IWGRP | stat.S_IWOTH))


@needs_tree
@pytest.mark.parametrize("usermap", ["off", "on"])
def test_clean_tree_is_patched_then_server_starts_after_user_switch(env_dir: Path, usermap: str) -> None:
    app = env_dir / "app"
    res = launch(env_dir, usermap)
    assert res.returncode == 0, res.stderr
    assert "usermap hooks applied" in res.stderr
    assert "dropped env HOME=" in (env_dir / "mark" / "dropped").read_text()
    started = (env_dir / "mark" / "started").read_text()
    assert f"pwd={app}" in started and "user=appuser" in started and "home=/home/appuser" in started
    for name in [*HOOKED, "crawl4ai_usermap.py"]:
        mode = (app / name).stat().st_mode
        assert stat.S_IMODE(mode) == 0o444, name
    assert stat.S_IMODE(app.stat().st_mode) == 0o555
    assert "import crawl4ai_usermap as usermap" in (app / "server.py").read_text().splitlines()


@needs_tree
def test_second_launch_in_same_container_is_idempotent(env_dir: Path) -> None:
    app = env_dir / "app"
    assert launch(env_dir, "on").returncode == 0
    first = snapshot(app)
    res = launch(env_dir, "on")
    assert res.returncode == 0, res.stderr
    assert snapshot(app) == first


def drift(app: Path) -> None:
    path = app / "mcp_bridge.py"
    path.chmod(0o644)
    path.write_text(path.read_text().replace('return {"Authorization": f"Bearer {token}"}', "return {}"))
    path.chmod(0o444)


@needs_tree
def test_drift_with_usermap_off_starts_unpatched_and_touches_nothing(env_dir: Path) -> None:
    app = env_dir / "app"
    drift(app)
    before = snapshot(app)
    res = launch(env_dir, "off")
    assert res.returncode == 0, res.stderr
    assert "ANCHOR-DRIFT" in res.stderr and "starting UNPATCHED" in res.stderr
    assert snapshot(app) == before
    assert not (app / "crawl4ai_usermap.py").exists()
    assert (env_dir / "mark" / "started").is_file()


@needs_tree
def test_drift_with_usermap_on_refuses_to_start(env_dir: Path) -> None:
    app = env_dir / "app"
    drift(app)
    before = snapshot(app)
    res = launch(env_dir, "on")
    assert res.returncode == 1
    assert "ANCHOR-DRIFT" in res.stderr and "Not starting" in res.stderr
    assert snapshot(app) == before
    assert not (env_dir / "mark" / "started").exists()


REAL_FAIL = """import os
root = os.environ["CRAWL4AI_PATCH_ROOT"]
if root == os.environ["CRAWL4AI_APP_DIR"]:
    print("ANCHOR-DRIFT: simulated failure that only the real run sees")
"""


@needs_tree
@pytest.mark.parametrize("usermap,code", [("off", 0), ("on", 1)])
def test_failure_in_real_run_restores_originals(env_dir: Path, usermap: str, code: int) -> None:
    app = env_dir / "app"
    (env_dir / "patches" / "c99_realfail.py").write_text(REAL_FAIL)
    before = snapshot(app)
    res = launch(env_dir, usermap)
    assert res.returncode == code, res.stderr
    assert "restoring the original files" in res.stderr
    assert snapshot(app) == before
    assert not (app / "crawl4ai_usermap.py").exists()
    assert stat.S_IMODE(app.stat().st_mode) == 0o555
    assert not any(writable_by_others(p) for p in app.glob("*.py"))
    assert (env_dir / "mark" / "started").exists() == (code == 0)


@needs_tree
def test_unwritable_app_dir_as_non_owner_is_treated_as_failure(env_dir: Path) -> None:
    if os.getuid() == 0:
        pytest.skip("root can always write")
    res = launch(env_dir, "on", {"CRAWL4AI_APP_DIR": "/proc/self/cwd/../nonexistent"})
    assert res.returncode != 0
    assert not (env_dir / "mark" / "started").exists()


@needs_tree
def test_without_drop_command_and_non_root_no_switch_is_used(env_dir: Path) -> None:
    if os.getuid() == 0:
        pytest.skip("needs a non-root user")
    res = launch(env_dir, "off", {"CRAWL4AI_DROP_CMD": ""})
    assert res.returncode == 0, res.stderr
    assert not (env_dir / "mark" / "dropped").exists()
    assert (env_dir / "mark" / "started").is_file()


@needs_tree
def test_version_warning(env_dir: Path) -> None:
    res = launch(env_dir, "off")
    assert "hooks were last checked against 0.9.5" in res.stderr


def test_launch_script_never_execs_server_without_drop_step() -> None:
    text = LAUNCH.read_text()
    assert text.count("exec ") == 1
    assert 'exec "${DROP[@]}"' in text
    assert "refusing to start the server as root" in text
