"""Static checks for the Crawl4AI usermap wiring in roles/k8s_infra (defaults, dummies, tasks)."""

from pathlib import Path
from typing import Any

import yaml

REPO = Path(__file__).resolve().parent.parent
DEFAULTS = REPO / "roles" / "k8s_infra" / "defaults" / "main" / "crawl4ai.yml"
DUMMIES = REPO / "group_vars" / "all" / "main" / "crawl4ai.yml"
TASKS = REPO / "roles" / "k8s_infra" / "tasks" / "apps" / "crawl4ai.yml"


def load(path: Path) -> Any:
    return yaml.safe_load(path.read_text())


def test_usermap_is_off_by_default_everywhere() -> None:
    for path in (DEFAULTS, DUMMIES):
        data = load(path)
        assert data["crawl4ai_usermap_enabled"] is False
        assert data["crawl4ai_usermap_fallback_openai_key"] is False
        assert data["crawl4ai_users"] == []


def test_dummy_file_holds_no_real_users() -> None:
    text = DUMMIES.read_text()
    assert "sk-" not in text.replace("sk-changeme", "")
    assert all(
        line.lstrip().startswith("#")
        for line in text.splitlines()
        if line.strip().endswith('0"') or "      token:" in line
    )


def test_capability_set_is_minimal() -> None:
    text = TASKS.read_text()
    assert "'capabilities': {'drop': ['ALL'], 'add': ['SETUID', 'SETGID']}" in text
    assert "'DAC_OVERRIDE'" not in text and "'FOWNER'" not in text
    assert "'allowPrivilegeEscalation': false" in text
    assert "/patches/crawl4ai_launch.sh" in text


def test_usermap_additions_are_all_gated() -> None:
    text = TASKS.read_text()
    assert text.count("if (crawl4ai_usermap_enabled | bool)") >= 3
    base = text[text.index("_crawl4ai_container_base:") : text.index("_crawl4ai_volumes_base:")]
    assert "command" not in base and "securityContext" not in base
    assert "usermap" not in base


def test_users_secret_is_not_in_any_checksum() -> None:
    text = TASKS.read_text()
    checksum = text[text.index("_crawl4ai_pod_annotations: >-") :]
    checksum = checksum[: checksum.index("\n\n")]
    assert "_crawl4ai_users_yml" not in checksum
    assert "_crawl4ai_usermap_files" in checksum
