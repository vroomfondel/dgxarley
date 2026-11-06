"""
Per-user bearer tokens and per-user LLM keys for the Crawl4AI Docker server (drop-in module, not upstream).

Placed next to server.py by crawl4ai_launch.sh and called from the one-line hooks that patches/c*.py insert.
All logic lives here; no Crawl4AI import at module level.

Off unless CRAWL4AI_USERMAP=on: every entry point then returns its input unchanged and the server behaves exactly
like upstream. See README.md in this directory for the file format and the fallback rule.

Settings (environment only):
    CRAWL4AI_USERMAP                     on | off (default off)
    CRAWL4AI_USERMAP_FILE                path of the users YAML, required when on
    CRAWL4AI_USERMAP_FALLBACK_OPENAI_KEY on | off (default off), only effective when on
"""

import contextvars
import hashlib
import hmac
import logging
import os
import re
import threading
import time
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass
from typing import Any, TypeVar
from urllib.parse import parse_qs

import yaml

ENV_ENABLE = "CRAWL4AI_USERMAP"
ENV_FILE = "CRAWL4AI_USERMAP_FILE"
ENV_FALLBACK = "CRAWL4AI_USERMAP_FALLBACK_OPENAI_KEY"
RELOAD_SECONDS = 5.0
DENIED_KEY = "crawl4ai-usermap-denied"
VIA_USER = "user_token"

_NAME_RE = re.compile(r"^[A-Za-z0-9._@-]{1,64}$")
_HASH_RE = re.compile(r"^[0-9a-fA-F]{64}$")

log = logging.getLogger("crawl4ai.usermap")
# The server's root logger sits at WARNING, which would drop the audit line; found on the first rollout (2026-11-06).
log.setLevel(logging.INFO)

_identity: contextvars.ContextVar[dict[str, str] | None] = contextvars.ContextVar("c4_usermap_identity", default=None)
_credential: contextvars.ContextVar[str | None] = contextvars.ContextVar("c4_usermap_credential", default=None)

T = TypeVar("T")
Scope = Mapping[str, Any]
Receive = Callable[[], Awaitable[Any]]
Send = Callable[[Mapping[str, Any]], Awaitable[None]]
AsgiApp = Callable[[Any, Any, Any], Awaitable[None]]


@dataclass(frozen=True)
class User:
    name: str
    token_sha256: bytes
    llm: bool
    llm_key: str | None
    models: tuple[str, ...]
    scope: str


@dataclass(frozen=True)
class Registry:
    users: tuple[User, ...]
    sources: tuple[str, ...]


@dataclass(frozen=True)
class Decision:
    source: str
    key: str | None
    reason: str


def _read_secret_file(path: str) -> str | None:
    with open(path, encoding="utf-8") as fh:
        return fh.read().strip() or None


def parse_registry(path: str) -> Registry:
    with open(path, encoding="utf-8") as fh:
        doc = yaml.safe_load(fh)
    if not isinstance(doc, dict) or doc.get("version") != 1 or not isinstance(doc.get("users"), list):
        raise ValueError("users file: expected a mapping with 'version: 1' and a 'users' list")
    sources = [path]
    users: list[User] = []
    names: set[str] = set()
    digests: set[bytes] = set()
    for item in doc["users"]:
        if not isinstance(item, dict):
            raise ValueError("users file: every user must be a mapping")
        name = str(item.get("name", ""))
        if not _NAME_RE.match(name):
            raise ValueError(f"users file: invalid user name {name!r}")
        raw_hash = str(item.get("token_sha256", ""))
        if not _HASH_RE.match(raw_hash):
            raise ValueError(f"user {name}: token_sha256 must be 64 hex characters")
        digest = bytes.fromhex(raw_hash)
        if name in names or digest in digests:
            raise ValueError(f"user {name}: duplicate name or token")
        scope = str(item.get("scope", "data"))
        if scope not in ("data", "admin"):
            raise ValueError(f"user {name}: scope must be 'data' or 'admin'")
        llm = item.get("llm", True)
        if not isinstance(llm, bool):
            raise ValueError(f"user {name}: llm must be true or false")
        key: str | None = None
        if item.get("litellm_key_file"):
            key_path = str(item["litellm_key_file"])
            sources.append(key_path)
            key = _read_secret_file(key_path)
        elif item.get("litellm_key"):
            key = str(item["litellm_key"]).strip() or None
        models = item.get("models") or []
        if not isinstance(models, list) or not all(isinstance(m, str) for m in models):
            raise ValueError(f"user {name}: models must be a list of strings")
        users.append(User(name, digest, llm, key, tuple(models), scope))
        names.add(name)
        digests.add(digest)
    return Registry(tuple(users), tuple(sources))


def _signature(paths: tuple[str, ...]) -> tuple[tuple[int, int, int], ...]:
    sig = []
    for p in paths:
        st = os.stat(p)
        sig.append((st.st_mtime_ns, st.st_size, st.st_ino))
    return tuple(sig)


def parse_switch(name: str, env: Mapping[str, str]) -> bool:
    raw = (env.get(name) or "off").strip().lower()
    if raw in ("on", "true", "1", "yes"):
        return True
    if raw in ("off", "false", "0", "no", ""):
        return False
    raise ValueError(f"{name} must be 'on' or 'off', got {raw!r}")


def _provider_model(provider: str) -> str:
    return provider.split("/", 1)[-1]


def _forbidden(detail: str) -> Exception:
    try:
        from fastapi import HTTPException
    except ImportError:
        return PermissionError(detail)
    return HTTPException(status_code=403, detail=detail)


class UserMap:
    def __init__(self, env: Mapping[str, str]) -> None:
        self.enabled = parse_switch(ENV_ENABLE, env)
        self.fallback_openai = parse_switch(ENV_FALLBACK, env)
        self.path = (env.get(ENV_FILE) or "").strip() or None
        self.interval = RELOAD_SECONDS
        self._lock = threading.Lock()
        self._checked = time.monotonic()
        self._registry: Registry | None = None
        self._sig: tuple[tuple[int, int, int], ...] = ()
        if self.enabled:
            if not self.path:
                raise ValueError(f"{ENV_ENABLE}=on requires {ENV_FILE}")
            try:
                self._registry = parse_registry(self.path)
                self._sig = _signature(self._registry.sources)
            except Exception as exc:
                raise ValueError(f"{ENV_ENABLE}=on but {ENV_FILE}={self.path} is unusable: {exc}") from exc

    @property
    def configured(self) -> bool:
        return self.enabled

    def registry(self) -> Registry | None:
        if self._registry is None or self.path is None:
            return None
        now = time.monotonic()
        if now - self._checked >= self.interval:
            with self._lock:
                if now - self._checked >= self.interval:
                    self._checked = now
                    self._reload()
        return self._registry

    def _reload(self) -> None:
        assert self.path is not None and self._registry is not None
        try:
            sig = _signature(self._registry.sources)
            if sig == self._sig:
                return
            self._sig = sig
            self._registry = parse_registry(self.path)
            self._sig = _signature(self._registry.sources)
            log.info("users file reloaded: %d users", len(self._registry.users))
        except Exception as exc:
            log.error("users file reload failed, keeping the previous one: %s", type(exc).__name__)

    def authenticate(self, token: str) -> dict[str, str] | None:
        reg = self.registry()
        if reg is None:
            return None
        presented = hashlib.sha256(token.encode("utf-8")).digest()
        found: User | None = None
        for user in reg.users:
            if hmac.compare_digest(presented, user.token_sha256):
                found = user
        if found is None:
            return None
        return {"sub": f"user:{found.name}", "user": found.name, "scope": found.scope, "via": VIA_USER}

    def decide(self, config: Mapping[str, Any] | None, identity: Mapping[str, str] | None, provider: str) -> Decision:
        reg = self.registry()
        if reg is None:
            return Decision("upstream", None, "inactive")
        if identity and identity.get("via") == VIA_USER:
            user = next((u for u in reg.users if u.name == identity.get("user")), None)
            if user is None:
                return Decision("denied", None, "user_removed")
            if not user.llm:
                return Decision("denied", None, "llm_disabled")
            if user.llm_key:
                if user.models and _provider_model(provider) not in user.models:
                    return Decision("denied", None, "model_not_allowed")
                return Decision("user", user.llm_key, "user_key")
            reason = "user_without_key"
        else:
            reason = "no_mapping"
        if self.fallback_openai:
            return Decision("default", None, reason)
        return Decision("denied", None, reason)


_instance: UserMap | None = None
_instance_lock = threading.Lock()


def get_map() -> UserMap:
    global _instance
    if _instance is None:
        with _instance_lock:
            if _instance is None:
                _instance = UserMap(os.environ)
    return _instance


def reset() -> None:
    global _instance
    _instance = None


def _effective_provider(config: Mapping[str, Any] | None, provider: str | None) -> str:
    default = ((config or {}).get("llm") or {}).get("provider") or ""
    return provider or str(default)


def authenticate(token: str) -> dict[str, str] | None:
    return get_map().authenticate(token)


def _bearer(scope: Scope) -> str | None:
    for name, value in scope.get("headers", []):
        if name == b"authorization":
            raw = value.decode("latin-1")
            return raw[7:].strip() if raw[:7].lower() == "bearer " else None
    if scope.get("type") == "websocket":
        vals = parse_qs(scope.get("query_string", b"").decode("latin-1")).get("token")
        return vals[0] if vals else None
    return None


async def call_app(app: AsgiApp, scope: Any, receive: Any, send: Any) -> None:
    um = get_map()
    if not um.configured or scope.get("type") not in ("http", "websocket"):
        await app(scope, receive, send)
        return
    principal = (scope.get("state") or {}).get("principal")
    ident = None
    cred = None
    if isinstance(principal, dict):
        ident = {k: str(principal[k]) for k in ("sub", "user", "via", "scope") if k in principal}
        if principal.get("via") == VIA_USER:
            cred = _bearer(scope)
    t_ident = _identity.set(ident)
    t_cred = _credential.set(cred)
    try:
        await app(scope, receive, send)
    finally:
        _credential.reset(t_cred)
        _identity.reset(t_ident)


def bind(factory: Callable[[], Awaitable[T]]) -> Callable[[], Awaitable[T]]:
    if not get_map().configured:
        return factory
    ident = _identity.get()
    cred = _credential.get()

    async def bound() -> T:
        t_ident = _identity.set(ident)
        t_cred = _credential.set(cred)
        try:
            return await factory()
        finally:
            _credential.reset(t_cred)
            _identity.reset(t_ident)

    return bound


def precheck_llm(config: Mapping[str, Any] | None, provider: str | None) -> None:
    um = get_map()
    if not um.configured:
        return
    ident = _identity.get()
    prov = _effective_provider(config, provider)
    decision = um.decide(config, ident, prov)
    log.info(
        "llm key source=%s reason=%s user=%s via=%s provider=%s",
        decision.source,
        decision.reason,
        (ident or {}).get("sub", "-"),
        (ident or {}).get("via", "-"),
        prov,
    )
    if decision.source == "denied":
        raise _forbidden("LLM access is not available for this caller")


def llm_api_token(config: Mapping[str, Any] | None, provider: str | None, default_key: str | None) -> str | None:
    um = get_map()
    if not um.configured:
        return default_key
    decision = um.decide(config, _identity.get(), _effective_provider(config, provider))
    if decision.source == "user":
        return decision.key
    if decision.source == "default":
        return default_key
    return DENIED_KEY


def cache_mode(current: T, bypass: T, applies: bool = True) -> T:
    return bypass if applies and get_map().configured else current


def rate_limit_key(default: Callable[[Any], str]) -> Callable[[Any], str]:
    if not get_map().configured:
        return default

    def key(request: Any) -> str:
        principal = getattr(request.state, "principal", None)
        if isinstance(principal, dict) and principal.get("via") == VIA_USER:
            return f"user:{principal.get('user')}"
        return default(request)

    return key


def mcp_auth_headers(default: dict[str, str]) -> dict[str, str]:
    if not get_map().configured:
        return default
    ident = _identity.get()
    cred = _credential.get()
    if cred and ident and ident.get("via") == VIA_USER:
        return {"Authorization": f"Bearer {cred}"}
    return default
