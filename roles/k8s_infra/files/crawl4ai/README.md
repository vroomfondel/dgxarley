# Crawl4AI per-user tokens and LLM keys

Drop-in for the self-hosted Crawl4AI Docker server (`unclecode/crawl4ai`, tested against source v0.9.5). A bearer token
identifies a user, and that user's LLM calls (`/md` with `f=llm`, `/llm/{url}`, `/llm/job`, MCP tool `md`) use a LiteLLM
virtual key that exists for Crawl4AI only. LiteLLM then attributes spend, budget and rate limits per user.

| File                             | Purpose                                                                                                 |
|----------------------------------|---------------------------------------------------------------------------------------------------------|
| `crawl4ai_usermap.py`            | All logic (drop-in). Placed at `/app/crawl4ai_usermap.py`; no Crawl4AI import at module level.          |
| `patches/c10_*.py` to `c50_*.py` | Anchor-based hook patches, one per hooked upstream file, SGLang style (`_patchlib.Patch`, idempotent).  |
| `patches/_c4lib.py`              | Glue: target root, unique-anchor edit. The patch library itself is SGLang's `_patchlib.py`, not copied. |
| `crawl4ai_verify_hooks.py`       | Post-patch check: every hooked file parses and carries all hook calls.                                  |
| `crawl4ai_launch.sh`             | Container command: patch `/app` in place as root, drop to `appuser`, `exec` the image's start.          |
| `users.example.yml`              | Mapping file with invented values.                                                                      |

Switch `CRAWL4AI_USERMAP` off (default) and every entry point returns its input unchanged: the server behaves like
upstream and all LLM calls use `OPENAI_API_KEY`.

## Settings (environment only)

| Variable                               | Values   | Default | Meaning                                                                                                    |
|----------------------------------------|----------|---------|------------------------------------------------------------------------------------------------------------|
| `CRAWL4AI_USERMAP`                     | on / off | off     | Recognise the user by token and use that user's LiteLLM key.                                               |
| `CRAWL4AI_USERMAP_FILE`                | path     | unset   | Users YAML. Required when `CRAWL4AI_USERMAP=on`.                                                           |
| `CRAWL4AI_USERMAP_FALLBACK_OPENAI_KEY` | on / off | off     | Only with `CRAWL4AI_USERMAP=on`: callers without a mapping use `OPENAI_API_KEY` (on) or are refused (off). |

Defaults: off/off keeps a stock deployment untouched (the module can ship in every image), and with the mapping on a
caller that has no own key is refused rather than silently billed to the shared key. `on`, `true`, `1`, `yes` count as
on; anything unrecognised stops the server.

`CRAWL4AI_USERMAP=on` with a missing, unreadable or invalid file (or without the path) does not fall back to off: the
server fails at import (traceback with the reason) and does not serve. A broken file on a later reload is logged and
the previous mapping stays active. The users file and the key files are re-read (checked every 5 s) when mtime, size or
inode changes, so Secret volumes without `subPath` update without a restart.

## Users file

`version: 1` and `users`:

| Key            | Meaning                                                                                          |
|----------------|--------------------------------------------------------------------------------------------------|
| `name`         | `[A-Za-z0-9._@-]{1,64}`, unique. The principal `sub` becomes `user:<name>`.                      |
| `token_sha256` | SHA-256 hex of the bearer token (tokens are never stored). Unique. Use `openssl rand -hex 32`.   |
| `litellm_key`  | LiteLLM virtual key of this user for Crawl4AI. Or `litellm_key_file` (path, whitespace trimmed). |
| `models`       | Optional allowlist of model names (the part after `provider/`). Empty: whatever the key allows.  |
| `llm`          | `false` blocks every LLM path for this user, also when the fallback is on. Optional.             |
| `scope`        | `data` (default) or `admin`.                                                                     |

Token comparison hashes the presented token and compares against every entry with `hmac.compare_digest` without early
exit.

## Fallback rule

"No mapping" means an authenticated caller for whom the file provides no LLM key. An unknown token is not part of
this: it is rejected with 401 by authentication.

| Caller                                        | Fallback off (default) | Fallback on      |
|-----------------------------------------------|------------------------|------------------|
| known user, own key, model allowed            | own key                | own key          |
| known user, own key, model not in `models`    | 403                    | 403              |
| known user with `llm: false`                  | 403                    | 403              |
| known user, entry without key                 | 403                    | `OPENAI_API_KEY` |
| static `CRAWL4AI_API_TOKEN` (operator)        | 403                    | `OPENAI_API_KEY` |
| JWT principal                                 | 403                    | `OPENAI_API_KEY` |
| `mcp-service` (MCP without a user token)      | 403                    | `OPENAI_API_KEY` |
| user removed from the file during the request | 403                    | 403              |
| unknown token                                 | 401                    | 401              |

With fallback off the key handed to the library is a fixed placeholder, so `OPENAI_API_KEY` is never used even when it
is set in the pod. Non-LLM filters (`f=fit`, `f=bm25`), `/html`, `/pdf` and so on are unaffected by all of this.
`llm: false` stays: it is one boolean and the only way to exclude a single user while the fallback is on.

Audit line (logger `crawl4ai.usermap`, INFO), one per LLM request, never containing a key:
`llm key source=user|default|denied reason=... user=user:alice via=user_token provider=openai/hermes-default`
(`default` means `OPENAI_API_KEY`).

## What else changes when active

- Page cache: LLM paths use `CacheMode.BYPASS` (the cache is keyed by URL only and shared).
- Rate limit: keyed per user instead of per remote address (default `1000/minute` bucket per user).
- Jobs: `/llm/job` binds the identity at enqueue and runs under it in the worker; nothing is written to Redis.
- MCP: the SSE/WebSocket connection's user token is forwarded on the loopback call, so tools run as that user.
  An operator token over MCP stays the anonymous `mcp-service` principal (no privilege change) and follows the fallback rule.

## Hooks

The patches import the module as `usermap` and insert these calls (no logic outside `crawl4ai_usermap.py`):

| File            | Hook                                                                                                                                                                                                            |
|-----------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `auth_gate.py`  | `usermap.authenticate(token)` after the static token; `usermap.call_app(...)` for the final app call                                                                                                            |
| `llm_broker.py` | `usermap.llm_api_token(config, provider, <default key>)` in `resolve_llm`                                                                                                                                       |
| `api.py`        | `usermap.precheck_llm(config, provider)` in `handle_llm_qa`, `handle_markdown_request` (LLM filter), `create_new_task`; `usermap.bind(factory)` in `_enqueue_job`; `usermap.cache_mode(...)` at two cache lines |
| `server.py`     | `key_func=usermap.rate_limit_key(get_remote_address)` (runs at import, so a broken users file stops startup)                                                                                                    |
| `mcp_bridge.py` | `usermap.mcp_auth_headers({...})` in `_service_auth_headers`                                                                                                                                                    |

## Start: `crawl4ai_launch.sh`

No init container and no copy of `/app`. The container starts as root, and the script

1. runs every `patches/c[0-9][0-9]_*.py` plus `crawl4ai_verify_hooks.py` against a scratch copy of the `*.py` files in
   `/app` (dry run: all anchors, nothing in `/app` written);
2. only if everything matched, makes the five files writable for their owner, copies the drop-in to
   `/app/crawl4ai_usermap.py`, runs the patches and the verifier on `/app` for real (a failure here restores the originals);
3. sets the patched files and the drop-in to `0444` (root-owned, readable for `appuser`, not writable) and restores
   the mode of `/app`;
4. `exec`s the image start command as `appuser`: `setpriv --reuid=appuser --regid=appuser --init-groups env HOME=/home/appuser USER=appuser LOGNAME=appuser bash entrypoint.sh`
   (`runuser -u appuser --` if `setpriv` is missing; if neither exists it refuses to start rather than run as root).

Every patch is idempotent (marker check before the anchor), so running the script twice in one container changes nothing.
A container restart starts from the image's filesystem again.

| Situation                                          | `CRAWL4AI_USERMAP=off`                      | `CRAWL4AI_USERMAP=on`           |
|----------------------------------------------------|---------------------------------------------|---------------------------------|
| all anchors match                                  | patched, server starts                      | patched, server starts          |
| an anchor drifted or a hook is missing (dry run)   | warning, `/app` untouched, starts unpatched | error, `/app` untouched, exit 1 |
| failure during the real write                      | originals restored, starts unpatched        | originals restored, exit 1      |
| installed Crawl4AI is not the version last checked | warning line only                           | warning line only               |
| root without `setpriv` and `runuser`               | exit 1                                      | exit 1                          |

Only whole sets are applied: all hooks or none. The version checked last is the constant `CHECKED_VERSION` (0.9.5) in the
script; the anchors are the real check.

Facts from the v0.9.5 image (Dockerfile, `deploy/docker/`):

- Base `python:3.12-slim-bookworm`. `groupadd -r appuser && useradd --no-log-init -r -g appuser appuser` (system user,
  primary group `appuser`, no supplementary groups, home `/home/appuser`, created and `chown`ed; Playwright browsers and
  `.cache` live there, hence `HOME` is set explicitly). `USER appuser`, `WORKDIR /app`, `CMD ["bash", "entrypoint.sh"]`.
- `entrypoint.sh` ends in `exec supervisord -c supervisord.conf`; supervisord runs redis and gunicorn with `user=appuser`.
  Started as `appuser`, supervisord and its children are `appuser` already; started as root it would stay root itself,
  which is why the script drops privileges first. Chromium is started by gunicorn's workers, so it inherits `appuser`.
- `/app` files are `root:root` with `chmod -R a-w`, flat `*.py` modules imported from the working directory
  (`server:app`); `PYTHONDONTWRITEBYTECODE=1`, so no stale `.pyc`; `deploy/` is not part of the pip package.
- `setpriv`, `runuser` and `su` come from `util-linux`, which is an essential package in Debian bookworm images. That is
  the basis for "nothing to install"; it is not verified on the built image (no Docker here).

## Wiring in the role

Implemented in `roles/k8s_infra/tasks/apps/crawl4ai.yml` (defaults in `defaults/main/crawl4ai.yml`, dummies in
`group_vars/all/main/crawl4ai.yml`, users file from `templates/crawl4ai/users.yml.j2`):

| Variable                               | Default         | Meaning                                                                                                                                           |
|----------------------------------------|-----------------|---------------------------------------------------------------------------------------------------------------------------------------------------|
| `crawl4ai_usermap_enabled`             | false           | Master switch. Needs `crawl4ai_llm_enabled`. False renders the plain Deployment, nothing else.                                                    |
| `crawl4ai_usermap_fallback_openai_key` | false           | `CRAWL4AI_USERMAP_FALLBACK_OPENAI_KEY`. Off also keeps `OPENAI_API_KEY` out of the pod.                                                           |
| `crawl4ai_usermap_mount_path`          | `/run/crawl4ai` | Directory of the Secret `crawl4ai-users` (key `users.yml`, mode 0444, no `subPath`).                                                              |
| `crawl4ai_users`                       | `[]`            | `name`, `token` (plain text in the vault, only its SHA-256 is rendered), `litellm_key_alias` or `litellm_key`, optional `models`, `llm`, `scope`. |

When on, the role adds: ConfigMap `crawl4ai-usermap` at `/patches` (launcher, drop-in, verifier, `_c4lib.py`, the five
`cNN_` scripts, and `_patchlib.py` read from `roles/k8s_dgx/files/sglang_patches/` without a copy), the Secret
`crawl4ai-users`, `command: ["bash", "/patches/crawl4ai_launch.sh"]`, the `securityContext` below, the env
`CRAWL4AI_USERMAP=on`, `CRAWL4AI_USERMAP_FILE`, `CRAWL4AI_USERMAP_FALLBACK_OPENAI_KEY`, and the pod annotation
`checksum/usermap-patch` (patch ConfigMap only). The users Secret is deliberately not in any checksum: user changes
are picked up by the reload without a restart. With the fallback off the role also deletes a stale `OPENAI_API_KEY`
from `crawl4ai-env` (a merge patch would keep it). When off, the role deletes `crawl4ai-usermap` and `crawl4ai-users`
if they exist.

LiteLLM keys: `litellm_key_alias` names an entry of `litellm_virtual_keys` (the existing declarative mechanism, tag
`litellm_keys`, `group_vars/all`, so visible to the `k8s_infra` play). Such an entry carries an optional `key:` (sk-...,
vault only); `roles/k8s_dgx/tasks/litellm_vkey.yml` then creates the LiteLLM key with exactly that value (otherwise
LiteLLM generates one that only lives in the Secret `litellm-vkey-<alias>`). Use a separate `key_alias` per user
(`<user>-crawl4ai`) with the user's usual `user_id` and a different key than the user's normal one. The value stays in
one place; the key is only applied on create (an existing key is never rotated). Run `litellm_keys` before the
Crawl4AI play.

Container `securityContext`: `runAsUser: 0`, `runAsNonRoot: false`, `allowPrivilegeEscalation: false`,
`capabilities: {drop: [ALL], add: [SETUID, SETGID]}`.

| Capability     | Needed | Why                                                                                                                                                     |
|----------------|--------|---------------------------------------------------------------------------------------------------------------------------------------------------------|
| `SETUID`       | yes    | `setpriv --reuid=appuser`.                                                                                                                              |
| `SETGID`       | yes    | `setpriv --regid=appuser --init-groups` (setgid and setgroups).                                                                                         |
| `DAC_OVERRIDE` | no     | The launcher is the owner of `/app` (root) and `chmod u+w`s the directory and the five files before writing, so the `0555`/`0444` modes never block it. |
| `FOWNER`       | no     | `chmod` on files owned by the caller needs no capability.                                                                                               |
| `CHOWN`        | no     | Nothing is chowned; new files are created by root and `chmod 444`.                                                                                      |
| `SETPCAP`      | no     | `setpriv` changes no capability sets; the kernel clears them on the switch to a non-zero uid.                                                           |

`allowPrivilegeEscalation: false` (no_new_privs) does not block the `setuid()` call, only gains through `execve`.
`readOnlyRootFilesystem` stays off. The role sets no `securityContext` at all when the usermap is off. This reasoning is
from reading the script and the kernel semantics; it has not been run in a pod.

Security effect: until the `exec`, a few seconds of shell and Python (patch scripts from the ConfigMap, the image's
`python3`) run as root inside the container with only `SETUID` and `SETGID`. Afterwards every process, including
supervisord, redis, gunicorn and Chromium, is `appuser` without capabilities, as before. `/app` stays root-owned and
not writable for `appuser`. An actor who can change the ConfigMap controls code that runs as root in that phase, so the
ConfigMap needs the same protection as the Deployment. Check any admission policy (Kyverno) against `runAsUser: 0`.

Why not an init container or a copy of `/app`: an init container needs a second volume and the same image twice; a copy
moves `/app`, and `supervisord.conf` hard-codes `directory=/app`.

## Reused SGLang patch library

`patches/_c4lib.py` imports `Patch` and `AnchorDrift` from `roles/k8s_dgx/files/sglang_patches/_patchlib.py` unchanged.
It fits because `Patch` resolves `os.path.join(DIST_PACKAGES, target)`, and an absolute `target` (here
`$CRAWL4AI_PATCH_ROOT/<file>`, default `/app`) wins in `os.path.join`. The contract is the same (never raise, marker check
before the anchor, per-file all-or-nothing, idempotent). What the library cannot do is signal drift by exit code (it
prints `ANCHOR-DRIFT` and exits 0), so the launcher greps for that line and additionally runs the verifier. `_c4lib.edit`
treats an anchor that matches more than once as drift. If `_patchlib.py` changes its path handling, the tests fail.

## What to do when an anchor no longer matches after an upstream update

The log shows `ANCHOR-DRIFT: <file>: <hook> anchor missing|ambiguous` (and `HOOK-MISSING: ...`). With the usermap off
the pod runs unpatched like upstream; with it on the pod stays down (Keel's new image will not start).

1. Pin the previous image tag in the Deployment to get back up, or set `CRAWL4AI_USERMAP=off` and accept no mapping.
2. Fetch the new tag: `git clone --depth 1 --branch vX.Y.Z https://github.com/unclecode/crawl4ai.git <dir>`.
3. Look at the named file in `deploy/docker/` and change the `old` text of that hook in `patches/cNN_*.py` (and the
   `marker`). Keep hooks one line plus the import.
4. `CRAWL4AI_SRC_TREE=<dir> pytest tests/test_crawl4ai_patches.py tests/test_crawl4ai_launch.py`; update the expected
   lines in the test if a hook changed shape, and `HOOKED`/`EXPECTED` in `crawl4ai_verify_hooks.py` if a file was added.
5. Set `CHECKED_VERSION` in `crawl4ai_launch.sh` to the new version and roll the ConfigMap.

## Not verified

The server and the container were never started; no Docker, no cluster. Verified: patches against the clean v0.9.5 tree
(exact added and removed lines, byte-identical other files, idempotence, drift and ambiguity), the launch script with
stand-ins for the server command and the user switch (both drift cases, restore after a failing real run, modes), the
drop-in unit tests, and the patched `llm_broker.py` / `auth_gate.py` behaviour was exercised earlier against stubs.
Open: the real `setpriv` switch and capability set under a pod `securityContext`, `setpriv`/`runuser` presence in the
image, that MCP tool handlers inherit the connection's context (otherwise MCP follows the `mcp-service` rule), and
that slowapi and Starlette see the principal the way the code assumes.

