"""Tests for dgxarley.k3shelperstuff.pin_drift."""

import json
from pathlib import Path

import pytest
import requests
import yaml
from pydantic import ValidationError

from dgxarley.k3shelperstuff.pin_drift import (
    GITCRYPT_MAGIC,
    ClusterSnapshot,
    Finding,
    PinConfig,
    PinSpec,
    PinStatus,
    TagLookup,
    classify,
    evaluate,
    fetch_tags,
    find_config,
    image_repo,
    read_pin,
    same_version,
    version_key,
    with_running,
)


def _spec(**overrides: object) -> PinSpec:
    fields: dict[str, object] = {"name": "demo", "file": "pins.yml", "var": "demo_version", "github": "owner/repo"}
    fields.update(overrides)
    return PinSpec.model_validate(fields)


def _write(root: Path, text: str, name: str = "pins.yml") -> None:
    (root / name).write_text(text)


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("v1.2.3", (1, 2, 3)),
        ("v1.37.0+k3s1", (1, 37, 0, 1)),
        ("4.6.0-4.8.3-distroless", (4, 6, 0, 4, 8, 3)),
        ("13.2.0-ubuntu", (13, 2, 0)),
        ("latest", ()),
    ],
)
def test_version_key(text: str, expected: tuple[int, ...]) -> None:
    assert version_key(text) == expected


@pytest.mark.parametrize(
    ("pinned", "candidate", "expected"),
    [
        ((1, 2, 3), (1, 2, 3), PinStatus.CURRENT),
        ((1, 2, 3), (1, 2, 4), PinStatus.PATCH),
        ((1, 2, 3), (1, 3, 0), PinStatus.MINOR),
        ((1, 2, 3), (2, 0, 0), PinStatus.MAJOR),
        ((1, 2, 3), (1, 2, 2), PinStatus.CURRENT),
        ((5, 1), (5, 1, 3), PinStatus.CURRENT),
        ((5, 1), (5, 2, 0), PinStatus.MINOR),
        ((2,), (2, 5, 5), PinStatus.CURRENT),
        ((2,), (3, 0, 0), PinStatus.MAJOR),
        ((1, 37, 0, 1), (1, 37, 0, 2), PinStatus.PATCH),
        ((1, 2, 3), (1, 3), PinStatus.MINOR),
    ],
)
def test_classify(pinned: tuple[int, ...], candidate: tuple[int, ...], expected: PinStatus) -> None:
    assert classify(pinned, candidate) is expected


@pytest.mark.parametrize(
    "overrides",
    [
        {"image": "foo/bar"},
        {"var": None},
        {"forgejo": "codeberg.org/owner/repo"},
        {"dockerhub": "owner/image"},
        {"github": None},
        {"file": None},
        {"var": None, "pattern": "no-group"},
        {"var": None, "pattern": "(a)(b)"},
        {"var": None, "pattern": "(unbalanced"},
        {"tag_pattern": "(unbalanced"},
        {"unknown_key": 1},
        {"running_image": "foo/bar", "running_kubelet": True},
    ],
)
def test_spec_rejects_invalid(overrides: dict[str, object]) -> None:
    with pytest.raises(ValidationError):
        _spec(**overrides)


def test_config_rejects_duplicate_names() -> None:
    entry = {"name": "demo", "file": "pins.yml", "var": "demo_version", "github": "owner/repo"}
    with pytest.raises(ValidationError):
        PinConfig.model_validate({"pins": [entry, entry]})


def test_read_pin_var(tmp_path: Path) -> None:
    _write(tmp_path, '# demo_version: "0.0.1"\nother_demo_version: "9.9.9"\ndemo_version: "1.2.3"  # comment\n')
    assert read_pin(_spec(), tmp_path) == ("1.2.3", "")


def test_read_pin_image(tmp_path: Path) -> None:
    _write(tmp_path, "image: prom/prometheus:v3.14.0\nimage: other/prom/prometheus:v1.0.0\n")
    spec = _spec(var=None, image="prom/prometheus")
    assert read_pin(spec, tmp_path) == ("v3.14.0", "")


def test_read_pin_pattern_across_files(tmp_path: Path) -> None:
    _write(tmp_path, "image: ghcr.io/x/multus-cni:v4.3.0-thick\n", "manifest.yml")
    _write(tmp_path, 'MULTUS_REF="${MULTUS_REF:-v4.3.0}"\n', "update.sh")
    spec = _spec(file=None, files=["manifest.yml", "update.sh"], var=None, pattern=r"(?:multus-cni:|REF:-)(v[\d.]+)")
    assert read_pin(spec, tmp_path) == ("v4.3.0", "")


def test_read_pin_inconsistent(tmp_path: Path) -> None:
    _write(tmp_path, "image: foo/bar:1.0.0\nimage: foo/bar:1.0.1\n")
    pinned, problem = read_pin(_spec(var=None, image="foo/bar"), tmp_path)
    assert pinned is None
    assert "1.0.0, 1.0.1" in problem


def test_read_pin_problems(tmp_path: Path) -> None:
    assert read_pin(_spec(), tmp_path)[0] is None
    _write(tmp_path, "something_else: 1\n")
    assert read_pin(_spec(), tmp_path) == (None, "pins.yml: pin not found")
    (tmp_path / "pins.yml").write_bytes(GITCRYPT_MAGIC + b"\x00\x01\x02")
    assert read_pin(_spec(), tmp_path) == (None, "pins.yml: git-crypt locked")


def test_evaluate_update_and_filters(tmp_path: Path) -> None:
    _write(tmp_path, 'demo_version: "v2.9.0"\n')
    lookup = TagLookup(tags=("helm-chart-9.9.9", "v2.11.0", "v2.10.1", "v2.9.0", "v3.0.0-rc1"))
    finding = evaluate(_spec(), tmp_path, lookup)
    assert (finding.pinned, finding.latest, finding.status) == ("v2.9.0", "v2.11.0", PinStatus.MINOR)


def test_evaluate_major_names_same_major(tmp_path: Path) -> None:
    _write(tmp_path, 'demo_version: "15.0.7"\n')
    finding = evaluate(_spec(), tmp_path, TagLookup(tags=("v16.0.5", "v15.0.9", "v15.0.7")))
    assert finding.status is PinStatus.MAJOR
    assert finding.latest == "v16.0.5"
    assert finding.note == "v15.0.9 within the pinned major"


def test_evaluate_floating_and_ahead(tmp_path: Path) -> None:
    _write(tmp_path, 'demo_version: "5.1"\n')
    floating = evaluate(_spec(), tmp_path, TagLookup(tags=("v5.1.3", "v5.0.9")))
    assert (floating.status, floating.note) == (PinStatus.CURRENT, "floating tag")

    _write(tmp_path, 'demo_version: "v1.20.12"\n')
    ahead = evaluate(_spec(), tmp_path, TagLookup(tags=("v1.20.11",)))
    assert ahead.status is PinStatus.CURRENT
    assert "ahead" in ahead.note


def test_evaluate_unclear(tmp_path: Path) -> None:
    _write(tmp_path, 'demo_version: "v1.0.0"\n')
    assert evaluate(_spec(), tmp_path, TagLookup(error="HTTP 403")).status is PinStatus.UNCLEAR
    assert evaluate(_spec(), tmp_path, TagLookup(tags=("nightly",))).status is PinStatus.UNCLEAR
    _write(tmp_path, 'demo_version: "latest"\n')
    assert evaluate(_spec(), tmp_path, TagLookup(tags=("v1.0.0",))).status is PinStatus.UNCLEAR


def test_dockerhub_upstream() -> None:
    spec = _spec(github=None, dockerhub="owner/image")
    assert spec.upstream.source == "tags"
    assert spec.upstream.url == "https://hub.docker.com/v2/repositories/owner/image/tags"
    assert spec.upstream.display == "owner/image (dockerhub)"
    assert _spec(github=None, dockerhub="redis").upstream.url.endswith("/repositories/library/redis/tags")


def test_fetch_dockerhub_tags_follows_pages(monkeypatch: pytest.MonkeyPatch) -> None:
    pages = {
        "first": {"results": [{"name": "canary"}, {"name": "v1.0.0"}], "next": "second"},
        "second": {"results": [{"name": "v1.1.0"}], "next": None},
    }

    class _Response:
        def __init__(self, body: object) -> None:
            self.ok = True
            self.content = json.dumps(body).encode()

    def _get(url: str, **_: object) -> _Response:
        return _Response(pages["second" if url == "second" else "first"])

    monkeypatch.setattr(requests, "get", _get)
    lookup = fetch_tags(_spec(github=None, dockerhub="owner/image").upstream, None)
    assert lookup.tags == ("canary", "v1.0.0", "v1.1.0")
    assert lookup.error == ""


def test_find_config_walks_upwards(tmp_path: Path) -> None:
    nested = tmp_path / "a" / "b"
    nested.mkdir(parents=True)
    (tmp_path / "pin_drift.yml").write_text("pins: []\n")
    assert find_config(nested) == tmp_path / "pin_drift.yml"


def test_repo_config_is_valid() -> None:
    root = Path(__file__).resolve().parent.parent
    config = PinConfig.model_validate(yaml.safe_load((root / "pin_drift.yml").read_text()))
    assert config.pins
    for spec in config.pins:
        for relative in spec.paths:
            assert (root / relative).is_file(), f"{spec.name}: {relative} is missing"


@pytest.mark.parametrize(
    ("running", "pinned", "expected"),
    [
        ("0.40.0", "v0.40.0", True),
        ("v4.3.1-thick", "v4.3.1", True),
        ("v1.37.1+k3s1", "v1.37.1+k3s1", True),
        ("v4.3.0-thick", "v4.3.1", False),
        ("latest", "main", False),
    ],
)
def test_same_version(running: str, pinned: str, expected: bool) -> None:
    assert same_version(running, pinned) is expected


def test_image_repo_folds_docker_hub() -> None:
    assert image_repo("redis") == image_repo("docker.io/library/redis:8.10")
    assert image_repo("prom/prometheus:v3") == image_repo("index.docker.io/prom/prometheus")


def _cluster() -> ClusterSnapshot:
    images = {
        image_repo("nousresearch/hermes-agent"): frozenset({"v2026.9.24", "v2026.9.14"}),
        image_repo("prom/prometheus"): frozenset({"v3.15.0"}),
    }
    return ClusterSnapshot("test", images, frozenset({"v1.37.1+k3s1"}))


def test_cluster_running() -> None:
    cluster = _cluster()
    hermes = _spec(var=None, pattern="(x)", running_image="nousresearch/hermes-agent")
    assert cluster.running(hermes) == ("v2026.9.14", "v2026.9.24")
    assert cluster.running(_spec(var=None, image="docker.io/prom/prometheus")) == ("v3.15.0",)
    assert cluster.running(_spec(running_kubelet=True)) == ("v1.37.1+k3s1",)
    assert cluster.running(_spec()) == ()


def test_with_running_notes_rollout() -> None:
    cluster = _cluster()
    spec = _spec(var=None, pattern="(x)", running_image="nousresearch/hermes-agent")

    partial = with_running(Finding(spec, "v2026.9.24", "v2026.9.24", PinStatus.CURRENT), cluster)
    assert not partial.rolled_out
    assert partial.note == "partially rolled out"

    ahead = with_running(Finding(spec, "v2026.10.1", "v2026.10.1", PinStatus.CURRENT, "floating tag"), cluster)
    assert ahead.note == "floating tag; not rolled out"

    prom = _spec(var=None, image="prom/prometheus")
    done = with_running(Finding(prom, "v3.15.0", "v3.15.0", PinStatus.CURRENT), cluster)
    assert done.rolled_out
    assert done.note == ""

    absent = with_running(Finding(_spec(), "v1.0.0", "v1.0.0", PinStatus.CURRENT), cluster)
    assert (absent.running, absent.rolled_out) == ((), True)
