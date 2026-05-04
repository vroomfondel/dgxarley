# NousResearch/hermes-agent Upstream Bug: Spurious npm install at every `dashboard --tui` launch breaks chat tab in non-root containers

## Status (as of 2026-05-03)

- **v2026.4.30 (v0.12.0) — BROKEN** in non-root container deployments.
  `hermes dashboard --tui` triggers an npm reinstall on every start; if
  `node_modules` is root-owned (as in the upstream image), the reinstall
  fails with `EACCES` and the TUI chat tab is permanently broken.

- **Partial upstream fix:** [PR #19520](https://github.com/NousResearch/hermes-agent/pull/19520)
  merged 2026-05-04 — fixes Trigger 1 only (peer-flag field-diff). No
  release tag as of 2026-05-03. Trigger 2 (workspace-link entries) is
  **unfixed** in `main`.

- **Workaround applied** in this repo via `copy-ui-tui` initContainer in
  `roles/k8s_dgx/templates/hermes/hermes_agent_deployment.yaml.j2`. See
  Workaround section below.

## Affected Versions

| Version | Status |
|---|---|
| v2026.4.30 (v0.12.0) | Confirmed broken |
| Earlier releases (npm 9 in toolchain) | Presumed affected |

npm 9 introduced a changed hidden-lock-file schema (`"peer"` key omission)
that activates Trigger 1. Trigger 2 (workspace-link entries) is a structural
issue independent of npm version.

## Symptoms

- Browser: chat tab shows orange `Chat unavailable: 1` or briefly
  `[session ended]`.
- Container stdout (default log level):
  ```
  Installing TUI dependencies…
  npm install failed.
  ```
- Container stdout with `loglevel=error`:
  ```
  npm ERR! code EACCES
  npm ERR! syscall rename
  npm ERR! path /opt/hermes/ui-tui/node_modules/@hermes/ink
  npm ERR! dest /opt/hermes/ui-tui/node_modules/@hermes/.ink-XXXX
  npm ERR! errno -13
  ```
- Reproducible on **every** container start, not just first-run.

## Root Cause

`hermes_cli.main._tui_need_npm_install()` compares the root lockfile
(`ui-tui/package-lock.json`) against npm's installed-state marker
(`ui-tui/node_modules/.package-lock.json`) and returns `True` on any
discrepancy. It has **two independent bugs** that each produce false
positives:

### Trigger 1: peer-flag field-diff (fixed in `main`, not yet released)

npm 9 omits `"peer": true` from the hidden lockfile for dev-dependencies
that are also declared as peers. The root `package-lock.json` retains
`"peer": true`; `node_modules/.package-lock.json` does not. The `comparable()`
helper in `_tui_need_npm_install()` sees a field difference → returns "different"
→ triggers reinstall on every start.

**Fix in [PR #19520](https://github.com/NousResearch/hermes-agent/pull/19520)**
(merged 2026-05-04): adds `"peer"` to `_NPM_LOCK_RUNTIME_KEYS` (alongside the
existing `"ideallyInert"` exclusion), so the field is ignored during comparison.

### Trigger 2: npm workspace-link entries not present in hidden lockfile (UNFIXED)

`ui-tui/` is an npm workspace with `packages/hermes-ink/` as a local
workspace package. The root `package-lock.json` lists workspace packages as
explicit entries (`packages/hermes-ink`, `packages/hermes-ink/node_modules/...`)
with `"link": true, "resolved": "packages/hermes-ink"`.

The hidden `node_modules/.package-lock.json` does **not** list these as
separate entries — only the symlink wrapper under
`node_modules/@hermes/ink` appears, with `"resolved": "file:packages/hermes-ink"`.

The "missing key" detection in `_tui_need_npm_install()` fires for the 9
workspace + transitive link entries:

```python
missing = [k for k in wanted if k not in installed
           and not (wanted[k].get("optional") or wanted[k].get("peer"))]
# → 9 entries in v2026.4.30: workspace links are neither optional nor peer
```

The current filter (`optional` or `peer`) does not cover `"link": true`
entries. PR #19520 does not address this — workspace-aware filtering is absent.

## Why Upstream Maintainers Don't Hit This

- **Local dev install**: running as the file owner → `npm install` has write
  access → the spurious refresh completes in ~1–2 s with no error.
- **`docker run` as root** (default): same — root owns everything in the image,
  no EACCES.
- The failure requires the combination of **non-root container UID** +
  **image-baked `node_modules` owned by root** — standard in hardened K8s
  deployments (e.g. `runAsUser: 1000`, no `supplementalGroups: 0`).

## Why PR #19520 Alone Does Not Fix Our Deployment

PR #19520 eliminates Trigger 1 (peer-flag diff). After release, the "different"
count drops by 1. However, empirically in v2026.4.30:

```
missing entries  : 9   (workspace-link entries — Trigger 2, unfixed)
peer-flag diffs  : 1   (Trigger 1, fixed by #19520)
```

With only Trigger 1 patched, `_tui_need_npm_install()` still returns `True` due
to the 9 missing workspace-link entries → npm reinstall → EACCES → broken chat tab.

## Diagnostic Snippet

Run inside the pod to reproduce the decision logic:

```python
# /opt/hermes/.venv/bin/python
import json
from pathlib import Path

root = Path('/opt/hermes/ui-tui')
wanted    = json.loads((root / 'package-lock.json').read_text()).get('packages') or {}
installed = json.loads((root / 'node_modules/.package-lock.json').read_text()).get('packages') or {}

missing = [k for k in wanted if k not in installed
           and not (wanted[k].get('optional') or wanted[k].get('peer'))]
print('missing:', len(missing))  # 9 in v2026.4.30

peer_diff = [k for k in wanted if k in installed
             and wanted[k].get('peer') != installed[k].get('peer')]
print('peer-flag diff:', len(peer_diff))  # 1 in v2026.4.30 (fixed by PR #19520)
```

## Workaround in This Repo

**File:** `roles/k8s_dgx/templates/hermes/hermes_agent_deployment.yaml.j2`

A `copy-ui-tui` initContainer copies `/opt/hermes/ui-tui/` from the
image layer into an `emptyDir` volume and `chown`s it to the container's
run UID. The main container mounts the `emptyDir` at
`/opt/hermes/ui-tui`, shadowing the image's root-owned directory.

When `_tui_need_npm_install()` returns `True`, npm now writes into the
user-owned `emptyDir` → no EACCES → spurious refresh completes silently
in ~2 s → TUI chat tab functions normally.

The `emptyDir` is ephemeral (cleared on pod restart), so the copy +
chown runs fresh on every pod start — acceptable given the directory is
small (~50 MB) and the node has fast local storage.

## Tracking

- Issue: [hermes-agent#18800](https://github.com/NousResearch/hermes-agent/issues/18800)
  — open, P2, labels `comp/tui`, `area/docker`
- Partial fix PR: [hermes-agent#19520](https://github.com/NousResearch/hermes-agent/pull/19520)
  — merged 2026-05-04, fixes Trigger 1 only; no release tag yet
- Our Trigger 2 report (workspace-link entries) posted 2026-05-04 as
  comment on #18800:
  [#issuecomment-4371280956](https://github.com/NousResearch/hermes-agent/issues/18800#issuecomment-4371280956).
  Includes the diagnostic snippet from this document and a one-line
  patch suggestion (add `wanted[k].get("link")` to the missing-check
  filter).
