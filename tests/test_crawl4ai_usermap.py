"""Tests for roles/k8s_infra/files/crawl4ai/crawl4ai_usermap.py (stand-alone, no Crawl4AI needed)."""

import asyncio
import hashlib
import importlib.util
import logging
import os
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest

REPO = Path(__file__).resolve().parent.parent
DROPIN = REPO / "roles" / "k8s_infra" / "files" / "crawl4ai" / "crawl4ai_usermap.py"


def load_dropin() -> ModuleType:
    spec = importlib.util.spec_from_file_location("crawl4ai_usermap", DROPIN)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules["crawl4ai_usermap"] = mod
    spec.loader.exec_module(mod)
    return mod


m: Any = load_dropin()

ALICE_TOKEN = "alice-token-" + "a" * 32
BOB_TOKEN = "bob-token-" + "b" * 32
CAROL_TOKEN = "carol-token-" + "c" * 32
DAVE_TOKEN = "dave-token-" + "d" * 32
ENV_KEY = "sk-env-openai-invented"
DEFAULT_KEY = "sk-default-invented"


def sha(token: str) -> str:
    return hashlib.sha256(token.encode()).hexdigest()


def users_yaml(extra: str = "") -> str:
    return f"""version: 1
users:
  - name: alice
    token_sha256: "{sha(ALICE_TOKEN)}"
    litellm_key: "sk-alice-invented"
    models: ["hermes-default"]
  - name: bob
    token_sha256: "{sha(BOB_TOKEN)}"
    litellm_key_file: "KEYFILE"
  - name: carol
    token_sha256: "{sha(CAROL_TOKEN)}"
  - name: dave
    token_sha256: "{sha(DAVE_TOKEN)}"
    litellm_key: "sk-dave-invented"
    llm: false
{extra}"""


@pytest.fixture
def users_file(tmp_path: Path) -> Path:
    key = tmp_path / "bob.key"
    key.write_text("sk-bob-invented\n")
    f = tmp_path / "users.yml"
    f.write_text(users_yaml().replace("KEYFILE", str(key)))
    return f


def env_for(users_file: Path | None, enable: str = "on", fallback: str = "off") -> dict[str, str]:
    env = {"CRAWL4AI_USERMAP": enable, "CRAWL4AI_USERMAP_FALLBACK_OPENAI_KEY": fallback}
    if users_file is not None:
        env["CRAWL4AI_USERMAP_FILE"] = str(users_file)
    return env


@pytest.fixture
def activate(monkeypatch: pytest.MonkeyPatch, users_file: Path) -> Any:
    monkeypatch.setenv("OPENAI_API_KEY", ENV_KEY)

    def _activate(fallback: str = "off", enable: str = "on") -> None:
        for k, v in env_for(users_file, enable, fallback).items():
            monkeypatch.setenv(k, v)
        m.reset()

    yield _activate
    m.reset()


def as_identity(who: str | None) -> dict[str, str] | None:
    if who is None:
        return None
    if who in ("operator", "mcp-service", "jwt"):
        via = {"operator": "api_token", "mcp-service": ""}.get(who, "")
        ident = {"sub": who if who != "jwt" else "x@example.org", "scope": "admin" if who == "operator" else "data"}
        if via:
            ident["via"] = via
        return ident
    return {"sub": f"user:{who}", "user": who, "via": "user_token", "scope": "data"}


def run_as(who: str | None, fn: Any) -> Any:
    token = m._identity.set(as_identity(who))
    try:
        return fn()
    finally:
        m._identity.reset(token)


def status_of(exc: BaseException) -> int:
    return int(getattr(exc, "status_code", 403))


def test_inactive_is_identity(monkeypatch: pytest.MonkeyPatch) -> None:
    for k in ("CRAWL4AI_USERMAP", "CRAWL4AI_USERMAP_FILE", "CRAWL4AI_USERMAP_FALLBACK_OPENAI_KEY"):
        monkeypatch.delenv(k, raising=False)
    m.reset()
    assert m.authenticate(ALICE_TOKEN) is None
    assert m.llm_api_token({}, None, DEFAULT_KEY) == DEFAULT_KEY
    assert m.llm_api_token({}, None, None) is None
    assert m.cache_mode("enabled", "bypass") == "enabled"
    assert m.cache_mode("enabled", "bypass", True) == "enabled"
    default = lambda r: "ip"  # noqa: E731
    assert m.rate_limit_key(default) is default
    factory = lambda: asyncio.sleep(0)  # noqa: E731
    assert m.bind(factory) is factory
    hdr = {"Authorization": "Bearer x"}
    assert m.mcp_auth_headers(hdr) is hdr
    m.precheck_llm({}, None)
    m.reset()


def test_off_ignores_file_settings(monkeypatch: pytest.MonkeyPatch, users_file: Path) -> None:
    for k, v in env_for(users_file, enable="off", fallback="on").items():
        monkeypatch.setenv(k, v)
    m.reset()
    assert m.authenticate(ALICE_TOKEN) is None
    assert m.llm_api_token({}, None, DEFAULT_KEY) == DEFAULT_KEY
    m.reset()


def test_authenticate_hash_match(activate: Any) -> None:
    activate()
    got = m.authenticate(ALICE_TOKEN)
    assert got == {"sub": "user:alice", "user": "alice", "scope": "data", "via": "user_token"}
    assert m.authenticate(BOB_TOKEN)["user"] == "bob"
    assert m.authenticate("x" * 40) is None
    assert m.authenticate(sha(ALICE_TOKEN)) is None
    assert m.authenticate("") is None


def test_compare_digest_visits_all_entries(activate: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    activate()
    calls: list[int] = []
    real = m.hmac.compare_digest

    def spy(a: bytes, b: bytes) -> bool:
        calls.append(1)
        return bool(real(a, b))

    monkeypatch.setattr(m.hmac, "compare_digest", spy)
    assert m.authenticate(ALICE_TOKEN) is not None
    assert len(calls) == 4


@pytest.mark.parametrize(
    "mutate,message",
    [
        (lambda t: t.replace("version: 1", "version: 2"), "version"),
        (lambda t: t.replace(sha(ALICE_TOKEN), "abc"), "64 hex"),
        (lambda t: t.replace("name: bob", "name: alice"), "duplicate"),
        (lambda t: t.replace("name: dave", "name: 'bad name!'"), "invalid user name"),
        (lambda t: t.replace("llm: false", "llm: maybe"), "llm must be"),
        (lambda t: t + "    scope: root\n", "scope"),
        (lambda t: ":::not yaml", ""),
    ],
)
def test_parse_errors_stop_startup(
    activate: Any, users_file: Path, mutate: Any, message: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    users_file.write_text(mutate(users_file.read_text()))
    monkeypatch.setenv("CRAWL4AI_USERMAP", "on")
    monkeypatch.setenv("CRAWL4AI_USERMAP_FILE", str(users_file))
    m.reset()
    with pytest.raises(ValueError, match="unusable"):
        m.get_map()
    if message:
        with pytest.raises(ValueError, match=message):
            m.parse_registry(str(users_file))


def test_startup_requires_file_when_on(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("CRAWL4AI_USERMAP", "on")
    monkeypatch.delenv("CRAWL4AI_USERMAP_FILE", raising=False)
    m.reset()
    with pytest.raises(ValueError, match="requires CRAWL4AI_USERMAP_FILE"):
        m.get_map()
    monkeypatch.setenv("CRAWL4AI_USERMAP_FILE", str(tmp_path / "missing.yml"))
    m.reset()
    with pytest.raises(ValueError, match="unusable"):
        m.get_map()
    m.reset()


def test_invalid_switch_value(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CRAWL4AI_USERMAP", "perhaps")
    m.reset()
    with pytest.raises(ValueError, match="CRAWL4AI_USERMAP"):
        m.get_map()
    monkeypatch.setenv("CRAWL4AI_USERMAP", "off")
    monkeypatch.setenv("CRAWL4AI_USERMAP_FALLBACK_OPENAI_KEY", "2")
    m.reset()
    with pytest.raises(ValueError, match="FALLBACK"):
        m.get_map()
    m.reset()


def test_reload_picks_up_new_user_and_keeps_old_on_broken_file(
    activate: Any, users_file: Path, caplog: pytest.LogCaptureFixture
) -> None:
    activate()
    um = m.get_map()
    um.interval = 0
    erin = "erin-token-" + "e" * 32
    assert m.authenticate(erin) is None
    users_file.write_text(
        users_file.read_text()
        + f'  - name: erin\n    token_sha256: "{sha(erin)}"\n    litellm_key: "sk-erin-invented"\n'
    )
    os.utime(users_file, ns=(1, users_file.stat().st_mtime_ns + 10**9))
    assert m.authenticate(erin)["user"] == "erin"
    with caplog.at_level(logging.ERROR, logger="crawl4ai.usermap"):
        users_file.write_text("version: 1\nusers: [broken")
        os.utime(users_file, ns=(1, users_file.stat().st_mtime_ns + 2 * 10**9))
        assert m.authenticate(erin)["user"] == "erin"
        assert m.authenticate(ALICE_TOKEN)["user"] == "alice"
        assert m.authenticate(ALICE_TOKEN) is not None
    assert sum("reload failed" in r.message for r in caplog.records) == 1


def test_reload_on_key_file_change(activate: Any, tmp_path: Path) -> None:
    activate()
    um = m.get_map()
    um.interval = 0
    assert run_as("bob", lambda: m.llm_api_token({}, "openai/x", None)) == "sk-bob-invented"
    key = tmp_path / "bob.key"
    key.write_text("sk-bob-rotated-invented\n")
    os.utime(key, ns=(1, key.stat().st_mtime_ns + 10**9))
    assert run_as("bob", lambda: m.llm_api_token({}, "openai/x", None)) == "sk-bob-rotated-invented"


CASES = [
    # who, model, fallback off -> expected, fallback on -> expected
    ("alice", "openai/hermes-default", "sk-alice-invented", "sk-alice-invented"),
    ("bob", "openai/anything", "sk-bob-invented", "sk-bob-invented"),
    ("alice", "openai/other-model", 403, 403),
    ("dave", "openai/hermes-default", 403, 403),
    ("carol", "openai/hermes-default", 403, DEFAULT_KEY),
    ("operator", "openai/hermes-default", 403, DEFAULT_KEY),
    ("jwt", "openai/hermes-default", 403, DEFAULT_KEY),
    ("mcp-service", "openai/hermes-default", 403, DEFAULT_KEY),
    (None, "openai/hermes-default", 403, DEFAULT_KEY),
]


@pytest.mark.parametrize("fallback,col", [("off", 2), ("on", 3)])
@pytest.mark.parametrize("case", CASES)
def test_fallback_table(activate: Any, case: tuple[Any, ...], fallback: str, col: int) -> None:
    activate(fallback=fallback)
    who, model, *_ = case
    expected = case[col]
    cfg = {"llm": {"provider": "openai/hermes-default"}}

    def pre() -> None:
        m.precheck_llm(cfg, model)

    def tok() -> str | None:
        got: str | None = m.llm_api_token(cfg, model, DEFAULT_KEY)
        return got

    if expected == 403:
        with pytest.raises(Exception) as ei:
            run_as(who, pre)
        assert status_of(ei.value) == 403
        assert run_as(who, tok) not in (ENV_KEY, DEFAULT_KEY)
    else:
        run_as(who, pre)
        assert run_as(who, tok) == expected


def test_fallback_off_never_yields_env_key(activate: Any) -> None:
    activate(fallback="off")
    for who in (None, "operator", "jwt", "mcp-service", "carol", "dave"):
        got = run_as(who, lambda: m.llm_api_token({}, "openai/x", None))
        assert got == m.DENIED_KEY
        assert got != os.environ["OPENAI_API_KEY"]


def test_fallback_on_hands_back_upstream_default_unchanged(activate: Any) -> None:
    activate(fallback="on")
    assert run_as("operator", lambda: m.llm_api_token({}, "openai/x", None)) is None
    assert run_as("operator", lambda: m.llm_api_token({}, "openai/x", DEFAULT_KEY)) == DEFAULT_KEY


def test_removed_user_is_denied_even_with_fallback(activate: Any, users_file: Path) -> None:
    activate(fallback="on")
    um = m.get_map()
    um.interval = 0
    users_file.write_text(users_yaml().split("  - name: bob")[0])
    os.utime(users_file, ns=(1, users_file.stat().st_mtime_ns + 10**9))
    with pytest.raises(Exception) as ei:
        run_as("carol", lambda: m.precheck_llm({}, "openai/x"))
    assert status_of(ei.value) == 403


def test_audit_line_has_source_and_no_secrets(activate: Any, caplog: pytest.LogCaptureFixture) -> None:
    activate(fallback="on")
    with caplog.at_level(logging.INFO, logger="crawl4ai.usermap"):
        run_as("alice", lambda: m.precheck_llm({}, "openai/hermes-default"))
        run_as("operator", lambda: m.precheck_llm({}, "openai/hermes-default"))
        activate(fallback="off")
        with pytest.raises(Exception):
            run_as("carol", lambda: m.precheck_llm({}, "openai/hermes-default"))
    text = "\n".join(r.getMessage() for r in caplog.records)
    assert "source=user" in text and "source=default" in text and "source=denied" in text
    for secret in ("sk-", ALICE_TOKEN, sha(ALICE_TOKEN), ENV_KEY):
        assert secret not in text


def test_cache_mode_bypass_when_active(activate: Any) -> None:
    activate()
    assert m.cache_mode("enabled", "bypass") == "bypass"
    assert m.cache_mode("enabled", "bypass", False) == "enabled"


def test_rate_limit_key(activate: Any) -> None:
    activate()
    key = m.rate_limit_key(lambda r: "ip:1.2.3.4")
    user_req = SimpleNamespace(
        state=SimpleNamespace(principal={"sub": "user:alice", "user": "alice", "via": "user_token"})
    )
    op_req = SimpleNamespace(state=SimpleNamespace(principal={"sub": "operator", "via": "api_token"}))
    none_req = SimpleNamespace(state=SimpleNamespace())
    assert key(user_req) == "user:alice"
    assert key(op_req) == "ip:1.2.3.4"
    assert key(none_req) == "ip:1.2.3.4"


async def _scoped(scope: dict[str, Any], seen: dict[str, Any]) -> None:
    async def app(s: Any, r: Any, se: Any) -> None:
        seen["ident"] = m._identity.get()
        seen["cred"] = m._credential.get()
        seen["headers"] = m.mcp_auth_headers({"Authorization": "Bearer service"})

        async def job() -> str:
            return str(m._identity.get())

        seen["bound"] = m.bind(job)

    await m.call_app(app, scope, None, None)
    seen["after"] = (m._identity.get(), m._credential.get())


def test_call_app_sets_and_resets_context(activate: Any) -> None:
    activate()
    principal = m.authenticate(ALICE_TOKEN)
    scope = {
        "type": "http",
        "headers": [(b"authorization", f"Bearer {ALICE_TOKEN}".encode())],
        "state": {"principal": principal},
    }
    seen: dict[str, Any] = {}
    asyncio.run(_scoped(scope, seen))
    assert seen["ident"]["user"] == "alice"
    assert seen["cred"] == ALICE_TOKEN
    assert seen["headers"] == {"Authorization": f"Bearer {ALICE_TOKEN}"}
    assert seen["after"] == (None, None)
    assert asyncio.run(seen["bound"]()) == str(seen["ident"])


def test_call_app_operator_token_is_not_forwarded(activate: Any) -> None:
    activate()
    scope = {
        "type": "http",
        "headers": [(b"authorization", b"Bearer operator-secret")],
        "state": {"principal": {"sub": "operator", "scope": "admin", "via": "api_token"}},
    }
    seen: dict[str, Any] = {}
    asyncio.run(_scoped(scope, seen))
    assert seen["cred"] is None
    assert seen["headers"] == {"Authorization": "Bearer service"}


def test_call_app_websocket_query_token(activate: Any) -> None:
    activate()
    scope = {
        "type": "websocket",
        "headers": [],
        "query_string": f"token={BOB_TOKEN}".encode(),
        "state": {"principal": m.authenticate(BOB_TOKEN)},
    }
    seen: dict[str, Any] = {}
    asyncio.run(_scoped(scope, seen))
    assert seen["cred"] == BOB_TOKEN


def test_bound_job_keeps_identity_in_other_task(activate: Any) -> None:
    activate(fallback="off")

    async def main() -> tuple[str | None, str | None]:
        results: list[str | None] = []

        async def job() -> None:
            results.append(m.llm_api_token({}, "openai/hermes-default", None))

        async def request() -> Any:
            t = m._identity.set(as_identity("alice"))
            try:
                return m.bind(job)
            finally:
                m._identity.reset(t)

        bound = await request()
        await asyncio.create_task(bound())
        await asyncio.create_task(job())
        return results[0], results[1]

    first, second = asyncio.run(main())
    assert first == "sk-alice-invented"
    assert second == m.DENIED_KEY
