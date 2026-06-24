# Strategic merge of ops-managed keys into /opt/data/config.yaml +
# /opt/data/.env, run by the hermes-agent `merge-config` initContainer.
#
# We can NOT mount the seed files as subPath of a ConfigMap/Secret directly —
# that makes them bind-mounts, and Hermes' `atomic_yaml_write()` does rename(2)
# over the target which fails with EBUSY on bind-mounts (the original 2026 trace
# was triggered by a Settings -> Theme save).
#
# Strategy: every top-level key present in the seed (provider, default-model,
# base_url for config.yaml; OPENAI_API_KEY for .env) is enforced on every pod
# start. Keys the user added (theme, tool toggles, extra provider keys) are
# preserved. Lets ops update the cluster default and propagate to existing pods
# on next restart, while still letting users customise non-managed bits in the
# WebUI without losing their work.
#
# Runs under the agent image's venv (PyYAML available, guaranteed cached on the
# node). Target ownership comes from the MERGE_UID/MERGE_GID env vars so this
# file stays static/cluster-wide (no per-user Jinja2 templating).
#
# Shared by BOTH the hermes-agent and hermes-webui pods (the webui spawns its
# own `hermes` subprocesses reading the same config.yaml/.env). The webui pod
# additionally passes WEBUI_SETTINGS_B64 (base64 of the per-user webui_settings
# JSON) to also merge ops-managed WebUI preferences into
# /opt/data/.webui/settings.json; the agent pod never sets it and skips that.
import base64
import json
import os
from typing import Any

import yaml

UID, GID = int(os.environ["MERGE_UID"]), int(os.environ["MERGE_GID"])


def deep_override(user: dict[str, Any], ops: dict[str, Any]) -> None:
    for k, v in ops.items():
        if isinstance(v, dict) and isinstance(user.get(k), dict):
            deep_override(user[k], v)
        else:
            user[k] = v


# .env — KEY=VALUE merge (ops wins for managed keys, others kept)
def parse_env(p: str) -> dict[str, str]:
    out: dict[str, str] = {}
    if not os.path.exists(p):
        return out
    for line in open(p):
        s = line.strip()
        if not s or s.startswith("#") or "=" not in s:
            continue
        k, v = s.split("=", 1)
        out[k.strip()] = v
    return out


# config.yaml — strategic merge
ops = yaml.safe_load(open("/seed-config/config.yaml")) or {}
try:
    user = yaml.safe_load(open("/opt/data/config.yaml")) or {}
except FileNotFoundError:
    user = {}
deep_override(user, ops)
# Patch the LiteLLM virtual key from the (Secret-backed) .env into
# model.api_key so Hermes' custom-provider resolver picks it up as
# cfg_api_key (#1760). REQUIRED: Hermes host-gates OPENAI_API_KEY to
# openai.com/azure (GHSA / #28660), so the env fallback never
# authenticates against our litellm.* base_url. Kept out of the
# ConfigMap (secret-free); injected here at startup only.
ops_env = parse_env("/seed-secret/.env")
_openai_key = (ops_env.get("OPENAI_API_KEY") or "").strip()
if _openai_key:
    user.setdefault("model", {})["api_key"] = _openai_key
    # Same patch for the auxiliary vision model (image tasks) — ONLY when the
    # seed config actually declares auxiliary.vision (custom provider → our
    # litellm base_url). Hermes host-gates OPENAI_API_KEY to openai.com/azure,
    # so the aux vision client also needs cfg_api_key set explicitly. Guarded so
    # we never invent an auxiliary block when no vision model is configured.
    if isinstance(user.get("auxiliary"), dict) and isinstance(user["auxiliary"].get("vision"), dict):
        user["auxiliary"]["vision"]["api_key"] = _openai_key
with open("/opt/data/config.yaml", "w") as f:
    yaml.safe_dump(user, f, sort_keys=False)
os.chown("/opt/data/config.yaml", UID, GID)
os.chmod("/opt/data/config.yaml", 0o600)
user_env = parse_env("/opt/data/.env")
user_env.update(ops_env)
with open("/opt/data/.env", "w") as f:
    for k, v in user_env.items():
        f.write(f"{k}={v}\n")
os.chown("/opt/data/.env", UID, GID)
os.chmod("/opt/data/.env", 0o600)

# WebUI-only: merge STATE_DIR/settings.json with the per-user webui_settings
# (passed as base64 JSON via WEBUI_SETTINGS_B64 by the webui pod only). Mirrors
# the config.yaml/.env merge above — ops-managed keys WIN, all other keys the
# user has set via the Preferences panel (POST /api/settings) are preserved.
# The ops-relevant toggles live in api/config.py:_SETTINGS_DEFAULTS upstream
# (sidebar_density, show_cli_sessions, simplified_tool_calling, busy_input_mode,
# …); none has an env/config override. base64 round-trip because the rendered
# JSON contains lowercase true/false/null which are not valid Python literals.
_settings_b64 = os.environ.get("WEBUI_SETTINGS_B64", "").strip()
ops_settings = json.loads(base64.b64decode(_settings_b64)) if _settings_b64 else {}
if ops_settings:
    state_dir = "/opt/data/.webui"
    settings_path = os.path.join(state_dir, "settings.json")
    os.makedirs(state_dir, exist_ok=True)
    os.chown(state_dir, UID, GID)
    os.chmod(state_dir, 0o700)
    try:
        user_settings = json.load(open(settings_path))
        if not isinstance(user_settings, dict):
            user_settings = {}
    except (FileNotFoundError, json.JSONDecodeError):
        user_settings = {}
    user_settings.update(ops_settings)  # ops keys win
    with open(settings_path, "w") as f:
        json.dump(user_settings, f, indent=2, sort_keys=True)
    os.chown(settings_path, UID, GID)
    os.chmod(settings_path, 0o600)
