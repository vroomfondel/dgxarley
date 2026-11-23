#!/usr/bin/env python3
"""Find version pins in the repo that have fallen behind their upstream release.

Counterpart to :mod:`dgxarley.k3shelperstuff.keel_drift`: that one compares the
digest of a rolling tag, this one compares a fixed pin (``v1.2.3``) against the
release list of the project it comes from.

The pins are declared in ``pin_drift.yml``, searched upwards from the current
directory. Each entry names the files carrying the pin, how to extract it
(``var``, ``image`` or a raw ``pattern`` with one capture group) and where the
releases live (``github: owner/repo``, ``forgejo: host/owner/repo`` or
``dockerhub: owner/repo`` for the tags of an image). Nothing is ever written:
the script reads the repo and the release APIs only.

A pin shorter than the upstream version (``5.1`` against ``5.1.3``) is a
floating tag and counts as current until a newer ``5.2`` appears.

With a usable kubeconfig the table gains a ``running`` column: the tags of the
matching container images in the cluster's pods (``image`` pins match by
default, ``running_image`` overrides, ``running_kubelet`` reads the nodes'
kubelet version). A running version that differs from the pin is flagged as
not (or only partially) rolled out; it does not change the exit code.

GitHub allows 60 anonymous API requests per hour, so a token is picked up from
``GITHUB_TOKEN`` / ``GH_TOKEN`` or, failing that, from ``gh auth token``.

Installed as the ``pin-drift`` entry point (extra ``dgxarley[k3s]``), also
runnable as ``python -m dgxarley.k3shelperstuff.pin_drift``.

Examples:
    pin-drift                        # every declared pin
    pin-drift --updates-only         # hide the pins that are current
    pin-drift --only hermes-agent    # a single pin (repeatable)
    pin-drift --config ../pin_drift.yml
    pin-drift --context ht@dgxarley  # running versions from this context
    pin-drift --no-running           # repo against upstream only

Every option can also be given as a ``PIN_DRIFT_*`` environment variable (the
CLI option wins); ``--context`` additionally reads ``KUBE_CONTEXT``.

Exit codes:
    * ``0`` -- every pin is current (or unclear).
    * ``1`` -- at least one pin has an update available, so the tool doubles as
      a gate.
    * ``2`` -- no usable ``pin_drift.yml``, an unknown ``--only`` name or an
      unusable explicit ``--context``.

Attributes:
    CONFIG_NAME: File name of the pin declarations, searched upwards from the
        current directory.
    GITHUB_API_HOST: Host of the GitHub REST API.
    DOCKERHUB_API_HOST: Host of the Docker Hub REST API.
    DOCKERHUB_MAX_PAGES: Number of Docker Hub tag pages read per image.
    GITCRYPT_MAGIC: Leading bytes of a file that git-crypt has not decrypted.
    DEFAULT_TAG_PATTERN: Regular expression a release tag must match to count
        as a version (``1.2``, ``v1.2.3``, ...).
    REQUEST_TIMEOUT_SECONDS: Timeout of a single release or cluster API request.
    FINISHED_POD_PHASES: Pod phases whose images no longer count as running.
    MAX_WORKERS: Number of upstream projects queried concurrently.
    PAGE_SIZE: Per upstream kind, the query parameter name and the largest page
        size the API accepts.
    console: Rich console for the result table and the summary (stdout).
    err_console: Rich console for progress and error messages (stderr).
    CLI_HELP: Help text of the ``pin-drift`` command.
    app: Typer application exposing :func:`main`.
    UPDATE_STATUSES: Statuses that mean an update is available.
    STATUS_ORDER: Sort order of the findings, most urgent first.
    STATUS_STYLE: Rich style per status in the result table.
"""

import logging
import os
import re
import shutil
import subprocess
import sys
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, replace
from enum import StrEnum
from pathlib import Path
from typing import Final, Literal, Self, TypedDict

import requests
import typer
import urllib3
import yaml
from kubernetes import client, config
from kubernetes.client.exceptions import ApiException
from pydantic import BaseModel, ConfigDict, TypeAdapter, ValidationError, field_validator, model_validator
from rich.console import Console
from rich.table import Table

from dgxarley import configure_logging, glogger, print_banner
from dgxarley.k3shelperstuff.keel_drift import parse_image

configure_logging()
glogger.enable("dgxarley")

type VersionKey = tuple[int, ...]
type UpstreamKind = Literal["github", "forgejo", "dockerhub"]
type UpstreamSource = Literal["releases", "tags"]
type ImageRepo = tuple[str, str]

CONFIG_NAME: Final[str] = "pin_drift.yml"
GITHUB_API_HOST: Final[str] = "api.github.com"
DOCKERHUB_API_HOST: Final[str] = "hub.docker.com"
DOCKERHUB_MAX_PAGES: Final[int] = 10
GITCRYPT_MAGIC: Final[bytes] = b"\x00GITCRYPT"
DEFAULT_TAG_PATTERN: Final[str] = r"^v?\d+(?:\.\d+)+$"
REQUEST_TIMEOUT_SECONDS: Final[int] = 20
FINISHED_POD_PHASES: Final[frozenset[str]] = frozenset({"Succeeded", "Failed"})
MAX_WORKERS: Final[int] = 8
# GitHub and Docker Hub cap a page at 100 entries, Forgejo/Gitea at 50 by default.
PAGE_SIZE: Final[dict[UpstreamKind, tuple[str, int]]] = {
    "github": ("per_page", 100),
    "forgejo": ("limit", 50),
    "dockerhub": ("page_size", 100),
}

_WIDE: Final[int] = 200

console: Final[Console] = Console(width=None if sys.stdout.isatty() else _WIDE)
err_console: Final[Console] = Console(stderr=True, width=None if sys.stderr.isatty() else _WIDE)

CLI_HELP: Final[str] = """Check whether version pins in the repo lag behind their upstream release.

Reads the pins declared in pin_drift.yml from the repo files and compares each
against the release list of its upstream project. With a usable kubeconfig the
versions running in the cluster are shown alongside. Read-only.

The exit code is 1 as soon as at least one pin has an update available, so the
script works as a gate in a pipeline.
"""

app: Final[typer.Typer] = typer.Typer(add_completion=False)


class ReleaseEntry(TypedDict, total=False):
    """One entry of a release or tag listing, reduced to the fields used here.

    The same shape covers both endpoints of both forges: a ``releases`` entry
    carries ``tag_name`` (plus ``name`` as a free-form title), a ``tags`` entry
    carries the tag in ``name``. Every key is optional because each endpoint
    fills only its own subset.

    Attributes:
        tag_name: Git tag of a release.
        name: Release title (``releases``, may be null on GitHub) or the tag
            itself (``tags``).
        prerelease: Whether the release is marked as a pre-release.
        draft: Whether the release is an unpublished draft.
    """

    tag_name: str
    name: str | None
    prerelease: bool
    draft: bool


_RELEASE_LISTING: Final[TypeAdapter[list[ReleaseEntry]]] = TypeAdapter(list[ReleaseEntry])


class DockerHubTagPage(TypedDict, total=False):
    """One page of the Docker Hub tag listing, reduced to the fields used here.

    Attributes:
        results: The tags of this page, each carrying the tag in ``name``.
        next: URL of the next page, null on the last one.
    """

    results: list[ReleaseEntry]
    next: str | None


_DOCKERHUB_PAGE: Final[TypeAdapter[DockerHubTagPage]] = TypeAdapter(DockerHubTagPage)


class PinStatus(StrEnum):
    """Outcome of comparing a pin with the newest upstream version.

    Attributes:
        CURRENT: The pin is at (or ahead of) the newest upstream version.
        PATCH: A newer version differs from the pin beyond the minor component.
        MINOR: A newer version differs from the pin in the minor component.
        MAJOR: A newer version differs from the pin in the major component.
        UNCLEAR: The pin or the upstream list could not be evaluated.
    """

    CURRENT = "current"
    PATCH = "patch"
    MINOR = "minor"
    MAJOR = "MAJOR"
    UNCLEAR = "unclear"


UPDATE_STATUSES: Final[tuple[PinStatus, ...]] = (PinStatus.MAJOR, PinStatus.MINOR, PinStatus.PATCH)
STATUS_ORDER: Final[tuple[PinStatus, ...]] = (*UPDATE_STATUSES, PinStatus.UNCLEAR, PinStatus.CURRENT)
STATUS_STYLE: Final[dict[PinStatus, str]] = {
    PinStatus.CURRENT: "green",
    PinStatus.PATCH: "yellow",
    PinStatus.MINOR: "bold yellow",
    PinStatus.MAJOR: "bold red",
    PinStatus.UNCLEAR: "magenta",
}


@dataclass(frozen=True)
class Upstream:
    """An upstream project whose release list a pin is compared against.

    Hashable, so pins sharing an upstream share a single API request.

    Attributes:
        kind: Which forge or registry API to talk to.
        repo: ``owner/repo`` for GitHub and Docker Hub, ``host/owner/repo`` for Forgejo.
        source: Whether to read the ``releases`` or the ``tags`` endpoint; always
            ``tags`` for Docker Hub.
    """

    kind: UpstreamKind
    repo: str
    source: UpstreamSource

    @property
    def url(self) -> str:
        """API URL of the release or tag listing.

        Returns:
            The listing URL on ``api.github.com``, ``hub.docker.com`` or the Forgejo host.
        """
        if self.kind == "github":
            return f"https://{GITHUB_API_HOST}/repos/{self.repo}/{self.source}"
        if self.kind == "dockerhub":
            repo = self.repo if "/" in self.repo else f"library/{self.repo}"
            return f"https://{DOCKERHUB_API_HOST}/v2/repositories/{repo}/tags"
        host, _, path = self.repo.partition("/")
        return f"https://{host}/api/v1/repos/{path}/{self.source}"

    @property
    def display(self) -> str:
        """Human-readable name for the result table.

        Returns:
            The repo path, suffixed with ``(forgejo)`` or ``(dockerhub)`` unless on GitHub.
        """
        return self.repo if self.kind == "github" else f"{self.repo} ({self.kind})"


class PinSpec(BaseModel):
    """One pin as declared in ``pin_drift.yml``.

    Exactly one extraction method (``var``, ``image``, ``pattern``) and exactly
    one upstream (``github``, ``forgejo``, ``dockerhub``) must be set, plus at
    least one file.

    Attributes:
        name: Unique name of the pin, used by ``--only`` and in the table.
        file: A single file carrying the pin, relative to the config file.
        files: Further files carrying the same pin; all must agree.
        var: YAML key whose scalar value is the pin.
        image: Container image reference whose tag is the pin.
        pattern: Regular expression with exactly one capture group for the pin.
        github: GitHub ``owner/repo`` of the upstream.
        forgejo: Forgejo ``host/owner/repo`` of the upstream.
        dockerhub: Docker Hub ``owner/repo`` whose image tags are the upstream versions.
        source: Whether upstream versions come from releases or from tags;
            ignored for ``dockerhub``, which only has tags.
        tag_pattern: Regular expression an upstream tag must match to count.
        running_image: Image names whose tags in the cluster's pods are the
            running version; defaults to ``image``.
        running_kubelet: Take the running version from the nodes' kubelet.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str
    file: str | None = None
    files: tuple[str, ...] = ()
    var: str | None = None
    image: str | None = None
    pattern: str | None = None
    github: str | None = None
    forgejo: str | None = None
    dockerhub: str | None = None
    source: UpstreamSource = "releases"
    tag_pattern: str = DEFAULT_TAG_PATTERN
    running_image: tuple[str, ...] = ()
    running_kubelet: bool = False

    @field_validator("running_image", mode="before")
    @classmethod
    def _single_image(cls, value: object) -> object:
        """Accept a single image name in place of a list.

        Args:
            value: The configured value.

        Returns:
            ``value`` wrapped in a tuple if it is a string, else unchanged.
        """
        return (value,) if isinstance(value, str) else value

    @field_validator("pattern", "tag_pattern")
    @classmethod
    def _compiles(cls, value: str | None) -> str | None:
        """Reject a regular expression that does not compile.

        Args:
            value: The configured expression, or ``None`` if unset.

        Returns:
            The unchanged value.

        Raises:
            ValueError: If ``value`` is not a valid regular expression.
        """
        if value is not None:
            try:
                re.compile(value)
            except re.error as exc:
                raise ValueError(f"not a valid regular expression: {exc}") from exc
        return value

    @model_validator(mode="after")
    def _exactly_one_of_each(self) -> Self:
        """Enforce the either/or rules between the fields.

        Returns:
            The validated model.

        Raises:
            ValueError: If no file is given, if not exactly one extraction
                method or not exactly one upstream is set, if ``pattern``
                does not have exactly one capture group, or if both
                ``running_image`` and ``running_kubelet`` are set.
        """
        if not self.paths:
            raise ValueError("needs 'file' or 'files'")
        if sum(item is not None for item in (self.var, self.image, self.pattern)) != 1:
            raise ValueError("needs exactly one of 'var', 'image', 'pattern'")
        if sum(item is not None for item in (self.github, self.forgejo, self.dockerhub)) != 1:
            raise ValueError("needs exactly one of 'github', 'forgejo', 'dockerhub'")
        if self.pattern is not None and re.compile(self.pattern).groups != 1:
            raise ValueError("'pattern' needs exactly one capture group")
        if self.running_image and self.running_kubelet:
            raise ValueError("'running_image' and 'running_kubelet' exclude each other")
        return self

    @property
    def running_repos(self) -> frozenset[ImageRepo]:
        """Normalised image repositories that carry the running version.

        Returns:
            ``(registry, repository)`` pairs from ``running_image``, else from
            ``image``; empty for kubelet pins and pins without either.
        """
        if self.running_kubelet:
            return frozenset()
        names = self.running_image or ((self.image,) if self.image else ())
        return frozenset(image_repo(name) for name in names)

    @property
    def paths(self) -> tuple[str, ...]:
        """All files carrying the pin.

        Returns:
            ``file`` (if set) followed by ``files``.
        """
        return (*((self.file,) if self.file else ()), *self.files)

    @property
    def extractor(self) -> re.Pattern[str]:
        """Compiled expression that captures the pin from a file's text.

        Returns:
            A pattern with exactly one capture group, built from ``var``,
            ``image`` or ``pattern`` (whichever is set).
        """
        if self.var is not None:
            return re.compile(rf"^\s*{re.escape(self.var)}:\s*[\"']?([^\"'\s#]+)", re.MULTILINE)
        if self.image is not None:
            return re.compile(rf"(?<![\w./-]){re.escape(self.image)}:(\w[\w.+-]*)")
        return re.compile(self.pattern or "", re.MULTILINE)

    @property
    def upstream(self) -> Upstream:
        """The upstream project of this pin.

        Returns:
            An :class:`Upstream` for whichever of ``github`` / ``forgejo`` /
            ``dockerhub`` is set.
        """
        if self.github is not None:
            return Upstream("github", self.github, self.source)
        if self.dockerhub is not None:
            return Upstream("dockerhub", self.dockerhub, "tags")
        return Upstream("forgejo", self.forgejo or "", self.source)


class PinConfig(BaseModel):
    """Top-level structure of ``pin_drift.yml``.

    Attributes:
        pins: All declared pins, in file order.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    pins: tuple[PinSpec, ...]

    @model_validator(mode="after")
    def _unique_names(self) -> Self:
        """Reject two pins with the same name.

        Returns:
            The validated model.

        Raises:
            ValueError: If a pin name occurs more than once.
        """
        names = [spec.name for spec in self.pins]
        duplicates = sorted({name for name in names if names.count(name) > 1})
        if duplicates:
            raise ValueError(f"duplicate pin names: {', '.join(duplicates)}")
        return self


@dataclass(frozen=True)
class ClusterSnapshot:
    """Versions currently running in the cluster.

    Attributes:
        context: Description of the kubeconfig context the snapshot was taken from.
        images: Tags of every image repository in a live pod.
        kubelet: Kubelet versions of all nodes.
    """

    context: str
    images: dict[ImageRepo, frozenset[str]]
    kubelet: frozenset[str]

    def running(self, spec: PinSpec) -> tuple[str, ...]:
        """Running versions of one pin.

        Args:
            spec: The pin declaration.

        Returns:
            The distinct versions, oldest first; empty if nothing matches.
        """
        if spec.running_kubelet:
            found: set[str] = set(self.kubelet)
        else:
            found = {tag for repo in spec.running_repos for tag in self.images.get(repo, ())}
        return tuple(sorted(found, key=lambda tag: (version_key(tag), tag)))


@dataclass(frozen=True)
class TagLookup:
    """Result of querying one upstream: either its tags or an error.

    Attributes:
        tags: Published version tags (drafts and pre-releases excluded).
        error: Short reason why the lookup failed; empty on success.
    """

    tags: tuple[str, ...] = ()
    error: str = ""


@dataclass(frozen=True)
class Finding:
    """The evaluated state of one pin.

    Attributes:
        spec: The pin declaration.
        pinned: The version read from the repo, ``None`` if unreadable.
        latest: The newest matching upstream version, ``None`` if unknown.
        status: How far the pin lags behind ``latest``.
        note: Extra explanation for the table (problem, floating tag, ...).
        running: Versions running in the cluster, oldest first.
    """

    spec: PinSpec
    pinned: str | None
    latest: str | None
    status: PinStatus
    note: str = ""
    running: tuple[str, ...] = ()

    @property
    def rolled_out(self) -> bool:
        """Whether the cluster runs exactly the pinned version.

        Returns:
            ``True`` if nothing runs, the pin is unknown, or every running
            version equals the pin.
        """
        pinned = self.pinned
        return pinned is None or all(same_version(tag, pinned) for tag in self.running)


def image_repo(image: str) -> ImageRepo:
    """Normalise an image name to registry and repository, dropping tag and digest.

    Args:
        image: Reference such as ``redis``, ``prom/prometheus:v3`` or ``quay.io/ceph/ceph``.

    Returns:
        ``(registry, repository)`` with Docker Hub aliases folded together.
    """
    ref = parse_image(image)
    return ref.registry, ref.repository


def same_version(running: str, pinned: str) -> bool:
    """Compare a running tag with a pin.

    Args:
        running: A tag from the cluster such as ``v4.3.1-thick``.
        pinned: The pin such as ``v4.3.1``.

    Returns:
        Whether both are equal ignoring a leading ``v``, or carry the same
        non-empty version number (a flavour suffix such as ``-thick`` aside).

    Examples:
        >>> same_version("0.40.0", "v0.40.0")
        True
        >>> same_version("v4.3.1-thick", "v4.3.1")
        True
    """
    if running.removeprefix("v") == pinned.removeprefix("v"):
        return True
    key = version_key(pinned)
    return bool(key) and version_key(running) == key


def version_key(text: str) -> VersionKey:
    """Turn a version string into a comparable tuple of integers.

    Args:
        text: A tag or pin such as ``v1.2.3`` or ``13.2.0-ubuntu``.

    Returns:
        The numeric components in order, e.g. ``(1, 2, 3)``; empty if the text
        carries no version number.

    Examples:
        >>> version_key("v1.2.3")
        (1, 2, 3)
        >>> version_key("k3s-1.30")
        (1, 30)
    """
    # A number running into a letter belongs to a word ("k3s"), not to the version.
    return tuple(int(part) for part in re.findall(r"\d+(?![A-Za-z\d])", text))


def classify(pinned: VersionKey, candidate: VersionKey) -> PinStatus:
    """Classify how far ``candidate`` is ahead of ``pinned``.

    Args:
        pinned: Version key of the pin.
        candidate: Version key of an upstream release.

    Returns:
        :attr:`PinStatus.CURRENT` if the candidate is not newer at the pin's
        precision, otherwise the first differing component as MAJOR, MINOR or
        PATCH.

    Examples:
        >>> classify((5, 1), (5, 1, 3))
        <PinStatus.CURRENT: 'current'>
        >>> classify((1, 2, 3), (1, 3, 0))
        <PinStatus.MINOR: 'minor'>
    """
    # Truncating to the pin's precision makes a floating "5.1" cover every 5.1.x.
    head = candidate[: len(pinned)]
    if head <= pinned:
        return PinStatus.CURRENT
    index = next(i for i, (old, new) in enumerate(zip(pinned, head)) if old != new)
    return (PinStatus.MAJOR, PinStatus.MINOR)[index] if index < 2 else PinStatus.PATCH


def find_config(start: Path) -> Path | None:
    """Search ``start`` and its parents for the pin declarations.

    Args:
        start: Directory to start the search in.

    Returns:
        The first ``pin_drift.yml`` found, or ``None`` if there is none up to
        the filesystem root.
    """
    for directory in (start, *start.parents):
        candidate = directory / CONFIG_NAME
        if candidate.is_file():
            return candidate
    return None


def load_config(path: Path) -> PinConfig:
    """Read and validate the pin declarations.

    Args:
        path: Path of ``pin_drift.yml``.

    Returns:
        The validated configuration.

    Raises:
        typer.Exit: With code 2 if the file cannot be read, is not valid YAML
            or does not match the schema.
    """
    try:
        raw: object = yaml.safe_load(path.read_text())
        return PinConfig.model_validate(raw)
    except (OSError, yaml.YAMLError, ValidationError) as exc:
        err_console.print(f"[red]{path} is not usable:[/] {exc}")
        raise typer.Exit(code=2) from exc


def github_token() -> str | None:
    """Find a GitHub API token.

    Checks ``GITHUB_TOKEN`` and ``GH_TOKEN`` first, then asks the ``gh`` CLI.

    Returns:
        The token, or ``None`` to fall back to anonymous access.
    """
    for name in ("GITHUB_TOKEN", "GH_TOKEN"):
        if os.environ.get(name):
            return os.environ[name]
    gh = shutil.which("gh")
    if gh is None:
        return None
    try:
        result = subprocess.run([gh, "auth", "token"], capture_output=True, text=True, timeout=5, check=False)
    except OSError, subprocess.SubprocessError:
        return None
    return result.stdout.strip() or None


def fetch_tags(upstream: Upstream, token: str | None) -> TagLookup:
    """Fetch the first page of an upstream's releases or tags.

    Never raises on network or API problems; those end up in
    :attr:`TagLookup.error` so one broken upstream does not stop the run.

    Args:
        upstream: The project to query.
        token: GitHub token, sent only to GitHub; ``None`` for anonymous access.

    Returns:
        The published tags (drafts and pre-releases skipped), or an error.
    """
    if upstream.kind == "dockerhub":
        return fetch_dockerhub_tags(upstream)
    headers = {"Accept": "application/json"}
    if upstream.kind == "github" and token:
        headers["Authorization"] = f"Bearer {token}"
    size_key, size = PAGE_SIZE[upstream.kind]
    try:
        response = requests.get(
            upstream.url, headers=headers, params={size_key: str(size)}, timeout=REQUEST_TIMEOUT_SECONDS
        )
    except requests.exceptions.RequestException as exc:
        return TagLookup(error=f"unreachable: {type(exc).__name__}")
    if not response.ok:
        detail = response.text.strip()[:80].replace("\n", " ")
        return TagLookup(error=f"HTTP {response.status_code} {detail}")
    try:
        entries = _RELEASE_LISTING.validate_json(response.content)
    except ValidationError as exc:
        if any(error["type"] == "json_invalid" for error in exc.errors()):
            return TagLookup(error="no JSON in the response")
        return TagLookup(error="unexpected response shape")

    if upstream.source == "tags":
        return TagLookup(tags=tuple(name for entry in entries if (name := entry.get("name"))))
    return TagLookup(
        tags=tuple(
            tag
            for entry in entries
            if (tag := entry.get("tag_name")) and not entry.get("prerelease") and not entry.get("draft")
        )
    )


def fetch_dockerhub_tags(upstream: Upstream) -> TagLookup:
    """Fetch the tags of a Docker Hub image, following up to :data:`DOCKERHUB_MAX_PAGES` pages.

    Docker Hub lists the most recently pushed tags first, so a flood of rolling
    tags (``canary.<date>``) can push release tags onto later pages.

    Args:
        upstream: The Docker Hub image to query.

    Returns:
        All tag names read, or an error if the first page fails.
    """
    size_key, size = PAGE_SIZE["dockerhub"]
    url: str | None = upstream.url
    params: dict[str, str] | None = {size_key: str(size)}
    tags: list[str] = []
    for _ in range(DOCKERHUB_MAX_PAGES):
        if url is None:
            break
        try:
            response = requests.get(
                url, headers={"Accept": "application/json"}, params=params, timeout=REQUEST_TIMEOUT_SECONDS
            )
        except requests.exceptions.RequestException as exc:
            if tags:
                break
            return TagLookup(error=f"unreachable: {type(exc).__name__}")
        if not response.ok:
            if tags:
                break
            detail = response.text.strip()[:80].replace("\n", " ")
            return TagLookup(error=f"HTTP {response.status_code} {detail}")
        try:
            page = _DOCKERHUB_PAGE.validate_json(response.content)
        except ValidationError as exc:
            if tags:
                break
            if any(error["type"] == "json_invalid" for error in exc.errors()):
                return TagLookup(error="no JSON in the response")
            return TagLookup(error="unexpected response shape")
        tags.extend(name for entry in page.get("results", []) if (name := entry.get("name")))
        url = page.get("next")
        params = None
    return TagLookup(tags=tuple(tags))


def read_pin(spec: PinSpec, root: Path) -> tuple[str | None, str]:
    """Read the pinned version from all files of a pin.

    Args:
        spec: The pin declaration.
        root: Directory the pin's file paths are relative to.

    Returns:
        ``(version, "")`` if every file carries the same pin, otherwise
        ``(None, problem)`` describing the first unreadable, git-crypt-locked
        or pin-less file, or the conflicting values.
    """
    found: set[str] = set()
    for relative in spec.paths:
        try:
            data = (root / relative).read_bytes()
        except OSError as exc:
            return None, f"{relative}: {exc.strerror or type(exc).__name__}"
        if data.startswith(GITCRYPT_MAGIC):
            return None, f"{relative}: git-crypt locked"
        matches: list[str] = spec.extractor.findall(data.decode(errors="replace"))
        if not matches:
            return None, f"{relative}: pin not found"
        found.update(matches)
    if len(found) > 1:
        return None, f"inconsistent pins: {', '.join(sorted(found))}"
    return found.pop(), ""


def evaluate(spec: PinSpec, root: Path, lookup: TagLookup) -> Finding:
    """Compare one pin against its upstream's tags.

    Args:
        spec: The pin declaration.
        root: Directory the pin's file paths are relative to.
        lookup: The already fetched tags of the pin's upstream.

    Returns:
        The finding, with a note for a floating tag, a pin ahead of upstream
        or a newer release within the pinned major.
    """
    pinned, problem = read_pin(spec, root)
    if pinned is None:
        return Finding(spec, None, None, PinStatus.UNCLEAR, problem)
    pinned_key = version_key(pinned)
    if not pinned_key:
        return Finding(spec, pinned, None, PinStatus.UNCLEAR, "pin carries no version number")
    if lookup.error:
        return Finding(spec, pinned, None, PinStatus.UNCLEAR, f"upstream: {lookup.error}")

    matcher = re.compile(spec.tag_pattern)
    candidates: list[tuple[VersionKey, str]] = sorted(
        (version_key(tag), tag) for tag in lookup.tags if matcher.search(tag)
    )
    if not candidates:
        return Finding(spec, pinned, None, PinStatus.UNCLEAR, f"no upstream {spec.upstream.source} match tag_pattern")

    latest_key, latest = candidates[-1]
    status = classify(pinned_key, latest_key)
    notes: list[str] = []
    if status is PinStatus.CURRENT:
        if latest_key[: len(pinned_key)] < pinned_key:
            notes.append(f"pin is ahead of the newest upstream {spec.upstream.source}")
        elif len(pinned_key) < len(latest_key):
            notes.append("floating tag")
    elif status is PinStatus.MAJOR:
        same_major = [(key, tag) for key, tag in candidates if key[0] == pinned_key[0]]
        if same_major and classify(pinned_key, same_major[-1][0]) is not PinStatus.CURRENT:
            notes.append(f"{same_major[-1][1]} within the pinned major")
    return Finding(spec, pinned, latest, status, "; ".join(notes))


def load_cluster(context: str | None) -> ClusterSnapshot | None:
    """Read the image tags of all live pods and the nodes' kubelet versions.

    Args:
        context: kubeconfig context to use, ``None`` for the active one.

    Returns:
        The snapshot, or ``None`` if no cluster is reachable and no context was
        named explicitly.

    Raises:
        typer.Exit: With code 2 if an explicitly named context is unusable.
    """
    try:
        config.load_kube_config(context=context)  # type: ignore[no-untyped-call]
        if context is not None:
            where = f"context {context}"
        else:
            _, active = config.list_kube_config_contexts()  # type: ignore[no-untyped-call]
            where = f"current context {(active or {}).get('name', '?')}"
    except (config.ConfigException, OSError) as exc:
        if context is not None:
            err_console.print(f"[red]Context {context!r} not usable: {exc}[/]")
            raise typer.Exit(code=2) from exc
        try:
            config.load_incluster_config()  # type: ignore[no-untyped-call]
        except config.ConfigException:
            err_console.print("[yellow]No kubeconfig and no in-cluster context, skipping the running column.[/]")
            return None
        where = "in-cluster service account"

    core = client.CoreV1Api()
    try:
        pods = core.list_pod_for_all_namespaces(_request_timeout=REQUEST_TIMEOUT_SECONDS).items
        nodes = core.list_node(_request_timeout=REQUEST_TIMEOUT_SECONDS).items
    except (ApiException, urllib3.exceptions.HTTPError) as exc:
        if context is not None:
            err_console.print(f"[red]Cluster of context {context!r} not usable: {exc}[/]")
            raise typer.Exit(code=2) from exc
        err_console.print(f"[yellow]Cluster not reachable, skipping the running column: {exc}[/]")
        return None

    images: dict[ImageRepo, set[str]] = {}
    for pod in pods:
        if pod.status and pod.status.phase in FINISHED_POD_PHASES:
            continue
        if pod.spec is None:
            continue
        for container in (*(pod.spec.containers or ()), *(pod.spec.init_containers or ())):
            if container.image:
                ref = parse_image(container.image)
                images.setdefault((ref.registry, ref.repository), set()).add(ref.tag)
    kubelet = frozenset(
        node.status.node_info.kubelet_version for node in nodes if node.status and node.status.node_info
    )
    return ClusterSnapshot(where, {repo: frozenset(tags) for repo, tags in images.items()}, kubelet)


def with_running(finding: Finding, cluster: ClusterSnapshot) -> Finding:
    """Attach the running versions to a finding and note an incomplete rollout.

    Args:
        finding: Result of :func:`evaluate`.
        cluster: Snapshot from :func:`load_cluster`.

    Returns:
        A copy of ``finding`` with ``running`` set.
    """
    running = cluster.running(finding.spec)
    result = replace(finding, running=running)
    if result.rolled_out:
        return result
    pinned = finding.pinned or ""
    rollout = "partially rolled out" if any(same_version(tag, pinned) for tag in running) else "not rolled out"
    return replace(result, note="; ".join(filter(None, (finding.note, rollout))))


def analyse(
    specs: Sequence[PinSpec], root: Path, token: str | None, cluster: ClusterSnapshot | None = None
) -> list[Finding]:
    """Evaluate all pins, querying each distinct upstream once in parallel.

    Args:
        specs: The pins to check.
        root: Directory the pins' file paths are relative to.
        token: GitHub token, or ``None`` for anonymous access.
        cluster: Running versions to attach, or ``None`` to leave them out.

    Returns:
        One finding per pin, most urgent status first, then by name.
    """
    upstreams = sorted({spec.upstream for spec in specs}, key=lambda item: (item.kind, item.repo, item.source))
    with err_console.status(f"[cyan]asking {len(upstreams)} upstream projects...[/]"):
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
            lookups: dict[Upstream, TagLookup] = dict(
                zip(upstreams, pool.map(lambda item: fetch_tags(item, token), upstreams))
            )
    findings = [evaluate(spec, root, lookups[spec.upstream]) for spec in specs]
    if cluster is not None:
        findings = [with_running(finding, cluster) for finding in findings]
    findings.sort(key=lambda item: (STATUS_ORDER.index(item.status), item.spec.name))
    return findings


def running_cell(finding: Finding) -> str:
    """Format the running versions, highlighting those that differ from the pin.

    Args:
        finding: Result carrying the running versions.

    Returns:
        Rich markup for the ``running`` column.
    """
    if not finding.running:
        return "[dim]-[/]"
    pinned = finding.pinned or ""
    return ", ".join(f"[{'green' if same_version(tag, pinned) else 'yellow'}]{tag}[/]" for tag in finding.running)


def render(findings: Sequence[Finding], updates_only: bool, show_running: bool = False) -> None:
    """Print the findings as a table.

    Args:
        findings: The evaluated pins, already sorted.
        updates_only: Skip the pins whose status is current and that are
            rolled out.
        show_running: Add the ``running`` column.
    """
    title = "Pin drift: pinned version against upstream release"
    table = Table(title=f"{title} and cluster" if show_running else title)
    table.add_column("Pin")
    table.add_column("Upstream")
    table.add_column("pinned", justify="right")
    if show_running:
        table.add_column("running", justify="right")
    table.add_column("latest", justify="right")
    table.add_column("Status")
    table.add_column("Note", overflow="fold")

    for finding in findings:
        if updates_only and finding.status is PinStatus.CURRENT and finding.rolled_out:
            continue
        table.add_row(
            finding.spec.name,
            finding.spec.upstream.display,
            finding.pinned or "-",
            *((running_cell(finding),) if show_running else ()),
            finding.latest or "-",
            f"[{STATUS_STYLE[finding.status]}]{finding.status}[/]",
            finding.note,
        )

    console.print(table)


@app.command(help=CLI_HELP)
def main(
    config_path: Path | None = typer.Option(
        None,
        "--config",
        "-c",
        envvar="PIN_DRIFT_CONFIG",
        help=f"Pin declarations (default: {CONFIG_NAME}, searched upwards from the current directory).",
    ),
    only: list[str] | None = typer.Option(
        None, "--only", "-o", envvar="PIN_DRIFT_ONLY", help="Check only this pin (repeatable)."
    ),
    updates_only: bool = typer.Option(
        False, "--updates-only", envvar="PIN_DRIFT_UPDATES_ONLY", help="Hide the pins that are current and rolled out."
    ),
    context: str | None = typer.Option(
        None,
        "--context",
        envvar=["PIN_DRIFT_CONTEXT", "KUBE_CONTEXT"],
        help="kubeconfig context for the running column (default: the current context).",
    ),
    no_running: bool = typer.Option(
        False, "--no-running", envvar="PIN_DRIFT_NO_RUNNING", help="Do not ask the cluster for running versions."
    ),
    quiet: bool = typer.Option(
        False, "--quiet", "-q", envvar="PIN_DRIFT_QUIET", help="Suppress the table, print only the summary."
    ),
    verbose: bool = typer.Option(
        False, "--verbose", "-v", envvar="PIN_DRIFT_VERBOSE", help="DEBUG logging, third-party libraries included."
    ),
) -> None:
    """Check the declared pins and print the drift.

    Args:
        config_path: Pin declarations; ``None`` searches upwards from the
            current directory.
        only: Restrict the check to these pin names.
        updates_only: Hide the pins that are current and rolled out.
        context: kubeconfig context for the running versions.
        no_running: Skip the cluster entirely.
        quiet: Print only the summary line, not the table.
        verbose: Enable DEBUG logging.

    Raises:
        typer.Exit: Always. Code 0 if no pin has an update, 1 if at least one
            has, 2 if the configuration is missing, invalid, ``--only``
            names an unknown pin, or an explicit ``--context`` is unusable.
    """
    if verbose:
        logging.basicConfig(level=logging.DEBUG, force=True)
    print_banner(module=Path(__file__).stem)

    path = config_path or find_config(Path.cwd())
    if path is None:
        err_console.print(f"[red]No {CONFIG_NAME} found in {Path.cwd()} or any parent directory.[/]")
        raise typer.Exit(code=2)
    specs = list(load_config(path).pins)

    if only:
        unknown = sorted(set(only) - {spec.name for spec in specs})
        if unknown:
            err_console.print(f"[red]Unknown pin: {', '.join(unknown)}[/]")
            err_console.print(f"[yellow]Declared: {', '.join(sorted(spec.name for spec in specs))}[/]")
            raise typer.Exit(code=2)
        specs = [spec for spec in specs if spec.name in only]

    token = github_token()
    access = "authenticated" if token else "anonymous, 60 requests/h"
    err_console.print(f"[cyan]Checking {len(specs)} pins from {path} (GitHub API: {access}).[/]")

    cluster = None if no_running else load_cluster(context)
    if cluster is not None:
        err_console.print(f"[cyan]Running versions from {cluster.context}.[/]")

    findings = analyse(specs, path.resolve().parent, token, cluster)
    if not quiet:
        render(findings, updates_only, show_running=cluster is not None)

    counts: dict[PinStatus, int] = {status: sum(item.status is status for item in findings) for status in PinStatus}
    console.print(
        f"{len(findings)} pins checked, "
        f"[bold red]{counts[PinStatus.MAJOR]} major[/], "
        f"[bold yellow]{counts[PinStatus.MINOR]} minor[/], "
        f"[yellow]{counts[PinStatus.PATCH]} patch[/], "
        f"[magenta]{counts[PinStatus.UNCLEAR]} unclear[/]"
        + (f", [yellow]{sum(not item.rolled_out for item in findings)} not rolled out[/]" if cluster else "")
    )

    raise typer.Exit(code=1 if any(counts[status] for status in UPDATE_STATUSES) else 0)


if __name__ == "__main__":
    app()
