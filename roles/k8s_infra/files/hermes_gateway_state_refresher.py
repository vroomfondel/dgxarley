#!/usr/bin/env python3
"""dgxarley sidecar — keep the WebUI's gateway-status badge green while the
hermes-email gateway is genuinely running.

Why this exists
---------------
The WebUI's ``api/agent_health.py`` reports the gateway as "running" only when
``{HERMES_HOME}/gateway_state.json`` has BOTH ``gateway_state == "running"`` AND
an ``updated_at`` ISO-8601 timestamp newer than ``GATEWAY_FRESHNESS_THRESHOLD_S``
(120 s). The agent (v2026.5.x..6.x) writes that file ONLY on lifecycle/platform
changes (``gateway/run.py`` -> ``_update_runtime_status``), so a healthy but idle
gateway goes stale within minutes and the UI falsely shows "Gateway not
configured".

Upstream's primary signal is a cross-process ``os.kill(pid, 0)`` on the gateway
PID, which needs the WebUI and the email sidecar to share a PID namespace. We
deliberately keep ``shareProcessNamespace: false`` because the email sidecar runs
the agent image's s6-overlay ``/init`` entrypoint, and ``s6-overlay-suexec``
refuses to start unless it is PID 1 — sharing the PID namespace makes the
pause-container PID 1 and crashloops the sidecar. So the kill-based liveness is
unavailable and the 120 s freshness fallback is all the WebUI has.

This refresher bumps ``updated_at`` periodically, but ONLY while the gateway is
actually alive — never falsifying liveness for a crashed (OOM/SIGKILL) gateway.

Liveness detection
------------------
We mirror ``gateway/status.py::is_gateway_runtime_lock_active``: the gateway holds
an ``flock(LOCK_EX)`` on ``{HERMES_HOME}/gateway.lock`` for its entire lifetime
(``acquire_gateway_runtime_lock``). A non-blocking ``flock(LOCK_EX | LOCK_NB)``
acquire by us therefore SUCCEEDS only if no process holds it — i.e. the gateway
is dead. Crucially the kernel releases an flock automatically when the holder
dies abruptly, so this is crash-safe in a way the PID file is not. ``/opt/data``
is local ext4 on k3smaster (the hermes pods are pinned there), so flock is fully
reliable here.

Sync note: keep the lock/state filenames below in sync with ``gateway/status.py``
(``_GATEWAY_LOCK_FILENAME`` / ``_RUNTIME_STATUS_FILE``) when bumping the image.
The momentary test-acquire matches upstream's own ``is_gateway_runtime_lock_active``
(and the WebUI calls it the same way), so the tiny race against the gateway's
one-time startup acquire is upstream-equivalent and accepted.
"""

import fcntl
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

HOME = Path(os.environ.get("HERMES_HOME", "/opt/data"))
LOCK_PATH = HOME / "gateway.lock"  # gateway/status.py::_GATEWAY_LOCK_FILENAME
STATE_PATH = HOME / "gateway_state.json"  # gateway/status.py::_RUNTIME_STATUS_FILE
INTERVAL_S = float(os.environ.get("REFRESH_INTERVAL_S", "30"))


def _gateway_alive() -> bool:
    """True iff some live process holds the gateway runtime lock.

    Mirrors gateway/status.py::is_gateway_runtime_lock_active — if we can take
    the lock, nobody holds it (gateway dead). Always release if we got it.
    """
    if not LOCK_PATH.exists():
        return False
    try:
        handle = open(LOCK_PATH, "a+", encoding="utf-8")
    except OSError:
        return False
    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        return True  # held by the live gateway
    else:
        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        return False
    finally:
        handle.close()


def _refresh_once() -> str:
    if not _gateway_alive():
        return "gateway not running (lock free) — skip"
    try:
        data = json.loads(STATE_PATH.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return "no/invalid gateway_state.json — skip"
    if not isinstance(data, dict):
        return "gateway_state.json is not a JSON object — skip"
    state = data.get("gateway_state")
    if state != "running":
        return f"gateway_state={state!r} — skip"
    data["updated_at"] = datetime.now(timezone.utc).isoformat()
    # Atomic replace in the same dir — mirrors utils.atomic_json_write.
    tmp_path = STATE_PATH.with_name(STATE_PATH.name + ".refresh.tmp")
    tmp_path.write_text(json.dumps(data), encoding="utf-8")
    os.replace(tmp_path, STATE_PATH)
    return "refreshed updated_at"


def main() -> None:
    print(f"[gw-refresh] start home={HOME} interval={INTERVAL_S}s", flush=True)
    while True:
        try:
            message = _refresh_once()
        except Exception as exc:  # best-effort cosmetic helper — never crash the sidecar
            message = f"error: {exc!r}"
        print(f"[gw-refresh] {message}", flush=True)
        time.sleep(INTERVAL_S)


if __name__ == "__main__":
    main()
