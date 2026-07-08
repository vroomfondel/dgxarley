"""Make the gateway api_server's /health(/detailed) endpoints PUBLIC (no auth).

Silences the per-poll log line
    WARNING gateway.platforms.api_server: API server rejected invalid API key:
        ... path='/health/detailed'
AND lets the dashboard's "Gateway" badge actually read the richer /health/detailed
payload (PID, uptime, connected platforms) instead of only the thin /health body.

WHY THIS IS NEEDED (upstream regression, unfixed as of 2026-07-08):
  The dashboard's cross-container liveness probe
  (hermes_cli/web_server.py::_probe_gateway_health, driven by the DEPRECATED
  GATEWAY_HEALTH_URL env var) always hits /health/detailed FIRST with a plain
  urllib request carrying NO Authorization header, then falls back to /health.
  Upstream PR NousResearch/hermes-agent#56260 ("require auth for /health/detailed
  and fail closed on weak keys", merged 2026-07-01, shipped in v2026.7.1 and still
  true on main/v2026.7.7.2) added Bearer-auth enforcement to /health/detailed via
  APIServerAdapter._check_auth but NEVER updated the probe to send the key. Net
  effect: every poll logs a 401 WARNING and the badge only ever sees the /health
  fallback. There is NO upstream tracking issue for this specific mismatch (the
  auth PR and the probe live in different files and nobody reconciled them).

WHAT IT DOES:
  Wraps APIServerAdapter._check_auth so that requests to /health and
  /health/detailed are treated as public (return None = auth OK); EVERY OTHER
  route keeps full Bearer auth unchanged. The api_server binds 127.0.0.1 only and
  is same-pod with the dashboard, so exposing status in-pod is low-risk.

HOW IT LOADS:
  A .pth (zz_hermes_health_patch.pth) drops `import hermes_health_patch` into the
  venv site dir, so site.py runs it at interpreter start — immune to the
  sitecustomize "first wins" shadow (the image inherits Ubuntu's
  /usr/lib/python3.13/sitecustomize.py). The target module is not imported yet at
  site-init, so we patch it LAZILY via a one-shot meta_path hook (works in the
  gateway process and any subprocess it spawns). Idempotent + fail-safe: if the
  target is gone after an image bump it logs one line and changes nothing.

RE-SYNC on a hermes.image_tag bump: confirm gateway.platforms.api_server still
defines APIServerAdapter._check_auth(self, request) and that aiohttp still exposes
the path as request.path. If upstream finally fixes the probe (sends the key) or
makes /health/detailed public, DELETE this patch.
"""

import sys
from collections.abc import Sequence
from importlib.machinery import ModuleSpec, PathFinder
from types import ModuleType
from typing import Any

_TARGET = "gateway.platforms.api_server"
_PUBLIC_PATHS = ("/health", "/health/detailed")
# Marker attribute name — set/read via setattr/getattr so mypy strict doesn't
# see a method/attribute assignment on a statically-typed object.
_MARKER = "_hermes_health_public"


def _patch(module: ModuleType) -> None:
    """Wrap APIServerAdapter._check_auth to whitelist the health paths."""
    adapter = getattr(module, "APIServerAdapter", None)
    orig = getattr(adapter, "_check_auth", None)
    if adapter is None or orig is None:
        print(
            "[hermes_health_patch] APIServerAdapter._check_auth not found (image changed?) — leaving auth untouched",
            file=sys.stderr,
        )
        return
    if getattr(orig, _MARKER, False):
        return  # already wrapped (idempotent across re-imports / subprocesses)

    def _check_auth(self: Any, request: Any) -> Any:
        try:
            if getattr(request, "path", None) in _PUBLIC_PATHS:
                return None  # public: no Bearer required
        except Exception:
            pass
        return orig(self, request)

    setattr(_check_auth, _MARKER, True)
    adapter._check_auth = _check_auth
    print(
        "[hermes_health_patch] /health and /health/detailed are now public "
        "(no Bearer required) — see NousResearch/hermes-agent#56260",
        file=sys.stderr,
    )


# Fast path: module already loaded (e.g. a subprocess inheriting sys.modules).
if _TARGET in sys.modules:
    _patch(sys.modules[_TARGET])
else:
    import importlib.abc

    class _HealthPatchFinder(importlib.abc.MetaPathFinder):
        """One-shot finder: delegates to the normal loader, then patches."""

        def find_spec(
            self,
            fullname: str,
            path: Sequence[str] | None,
            target: ModuleType | None = None,
        ) -> ModuleSpec | None:
            if fullname != _TARGET:
                return None
            # Remove ourselves first so the delegated find + any re-entrancy
            # can't loop back through this finder.
            try:
                sys.meta_path.remove(self)
            except ValueError:
                pass
            spec = PathFinder.find_spec(fullname, path)
            if spec is None or spec.loader is None:
                return None
            _orig_exec = spec.loader.exec_module

            def exec_module(module: ModuleType, _orig_exec: Any = _orig_exec) -> None:
                _orig_exec(module)
                try:
                    _patch(module)
                except Exception as exc:  # never break the real import
                    print(
                        f"[hermes_health_patch] patch failed, auth left untouched: {exc}",
                        file=sys.stderr,
                    )

            # setattr (not `spec.loader.exec_module = ...`) so mypy strict doesn't
            # flag a method assignment on the typed Loader.
            setattr(spec.loader, "exec_module", exec_module)
            return spec

    sys.meta_path.insert(0, _HealthPatchFinder())
