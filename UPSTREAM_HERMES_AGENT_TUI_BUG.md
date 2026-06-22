# NousResearch/hermes-agent Upstream Bug: Spurious npm install at every `dashboard --tui` launch breaks chat tab in non-root containers

## Status (re-verified 2026-06-22)

> **Update 2026-06-22 — pinned image is now `v2026.6.19` (v0.17.0); the
> version references in the older blocks below (v2026.6.5 / v2026.5.16) are
> STALE.** `hermes.image_tag` in `roles/k8s_infra/defaults/main.yml` is
> **`v2026.6.19`** (verified in-repo), i.e. **v0.17.0**, currently the latest
> upstream release (published 2026-06-19). Consequences:
> - **`--tui` removal (PR #38591, merged 2026-06-04, shipped v0.16.0 /
>   v2026.6.5) is fully active** in the pinned image — the deployment template
>   must not emit `--tui` (it doesn't; `dashboard_tui` is a documented no-op).
> - **Entrypoint chown (PR #33045, v0.15.0 / v2026.5.28)** and the build-time
>   `npm_config_install_links=false` are both present in v2026.6.19, so the
>   `copy-ui-tui` initContainer-removal action items (§1–4) remain **actionable
>   but still pending** as of 2026-06-22 — not yet executed.
> - Trigger 2 (workspace-link entries) still unfixed in the reinstall logic
>   upstream; no follow-up issue filed (Action Item §5 still pending).

> **Update 2026-06-08 — the `--tui` flag itself was REMOVED upstream, and the
> repo is now pinned past the chown fix.** Two material changes since the
> 2026-05-31 check:
>
> 1. **`--tui` is gone.** [PR #38591](https://github.com/NousResearch/hermes-agent/pull/38591)
>    (merged 2026-06-04, shipped in **v0.16.0 / v2026.6.5**) removed the
>    `--tui` flag from `hermes dashboard`. Passing it now aborts with
>    `unrecognized arguments: --tui` → pod crash-loop. This repo's
>    `hermes.image_tag` is now **`v2026.6.5`**, so any user with
>    `dashboard_tui: true` in vault would crash-loop. **Fixed in
>    `hermes_agent_deployment.yaml.j2`**: the template no longer emits `--tui`
>    (the `dashboard_tui` gate is now a documented no-op until an upstream
>    replacement flag is identified). The original npm-reinstall bug this doc
>    is named for is moot on v2026.6.5+ because the flag that triggered it no
>    longer exists.
> 2. **`copy-ui-tui` initContainer is now removable.** Condition (a) (PR #33045,
>    entrypoint chown after UID remap) shipped in v0.15.0 and is present in our
>    now-pinned v2026.6.5; the stage2-hook chowns `.venv`, `ui-tui`, `gateway`,
>    `node_modules`. Trigger 2 stays structurally neutralised by
>    `npm_config_install_links=false`. Drop the initContainer + its volume/mount
>    and confirm chat still loads (see Action Items). Latest upstream release:
>    **v0.16.0 / v2026.6.5**.

> **Update 2026-05-31 — entrypoint chown fix has shipped (condition (a) met).**
> [PR #33045](https://github.com/NousResearch/hermes-agent/pull/33045)
> ("fix(docker): chown ui-tui and node_modules on UID remap so TUI esbuild
> works") **merged 2026-05-27, shipped in v0.15.0 / v2026.5.28**. It extends
> `docker/stage2-hook.sh` to `chown -R hermes:hermes $INSTALL_DIR/ui-tui
> $INSTALL_DIR/node_modules` **after** the `usermod` UID remap — exactly the
> "condition (a)" this doc was waiting for (entrypoint-level chown after
> remap). **Practical consequence:** the `copy-ui-tui` initContainer can be
> dropped **once `hermes.image_tag` is bumped to v2026.5.28+**. The repo is
> currently still pinned at `hermes.image_tag: "v2026.5.16"` (v0.14.0), which
> does NOT contain #33045, so the initContainer remains mandatory for now.
> Latest upstream release is **v2026.5.29.2 / v0.15.2** (2026-05-29).
> Trigger 2 (workspace-link entries) is structurally neutralised at build time
> by `npm_config_install_links=false` (present since v2026.5.16), so in
> practice the spurious-reinstall divergence no longer appears.

- **v2026.4.30 (v0.12.0) — BROKEN** in non-root container deployments.
  `hermes dashboard --tui` triggers an npm reinstall on every start; if
  `node_modules` is root-owned (as in the upstream image), the reinstall
  fails with `EACCES` and the TUI chat tab is permanently broken.

- **Partial upstream fix (logic):** [PR #19520](https://github.com/NousResearch/hermes-agent/pull/19520)
  merged 2026-05-04 — fixes Trigger 1 only (peer-flag field-diff).
  Trigger 2 (workspace-link entries) is **unfixed** in `main`, so the
  spurious reinstall still fires every start.

- **Partial upstream fix (image):** [PR #21267](https://github.com/NousResearch/hermes-agent/pull/21267)
  merged 2026-05-07 (issue [#18800](https://github.com/NousResearch/hermes-agent/issues/18800)
  closed via this PR; salvages [#19303](https://github.com/NousResearch/hermes-agent/pull/19303)).
  `chown`s `/opt/hermes/ui-tui` and `/opt/hermes/node_modules` to
  `hermes:hermes` (UID 10000) at image build time so the spurious
  reinstall can write through. **Helps Docker-compose users running as
  the image's baked-in `hermes` user (UID 10000)**; does **not** help
  our deployment (see next section).

- **First tagged release with both fixes: v2026.5.7 / v0.13.0** ("The Tenacity
  Release"), published 2026-05-07T16:23 UTC — about 3 h after PR #21267 merged
  (2026-05-07T13:17 UTC) and 3 days after PR #19520 (2026-05-04). Both fixes are
  in this tag. **Current latest release: v2026.5.29.2 / v0.15.2 (2026-05-29)** (re-checked 2026-05-31).
  Note: **PR #33045 (entrypoint chown after UID remap) merged 2026-05-27, in v0.15.0 / v2026.5.28** —
  see the Status update at the top; once we bump `hermes.image_tag` to v2026.5.28+ the initContainer
  is removable. Trigger 2 (workspace-link entries) is **still not fixed in the reinstall logic** in v0.15.x — only
  Trigger 1 is fixed by #19520; no follow-up issue specifically for Trigger 2 has been
  filed (two adjacent open issues #20739 and #25351 cover other TUI chat-tab bugs, not
  Trigger 2). **Our deployment still requires the workaround** because (a) the
  build-time `chown` from #21267 doesn't propagate through our `usermod`-based
  UID remap (see "Why PR #21267 Alone …" below), and (b) Trigger 2 (workspace-link
  entries) is still unfixed. Re-check on next image bump regardless.

- **Workaround applied** in this repo via `copy-ui-tui` initContainer in
  `roles/k8s_dgx/templates/hermes/hermes_agent_deployment.yaml.j2`. See
  Workaround section below. Workaround stays mandatory until **both**
  the logic fix (Trigger 2) ships **and** we drop the per-user UID
  remap, OR upstream extends the build-time `chown` to cover arbitrary
  runtime UIDs (e.g. world-writable `node_modules`, or entrypoint-time
  `chown -R` of `/opt/hermes/ui-tui`).

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
- **Docker-compose with the baked `hermes` user (UID 10000)**: PR
  #21267 shipped in v2026.5.7 (2026-05-07), so the build-time
  `chown hermes:hermes` now matches the runtime UID → `npm install` succeeds
  on that release and later.
- The failure still bites whenever the runtime UID does **not** equal
  numeric 10000 — typical in hardened K8s deployments and any setup
  that remaps the in-container user to a host-owned UID
  (e.g. `runAsUser: 1000`, NFS subPath ownership, our `usermod`-based
  entrypoint).

## Why PR #21267 Alone Does Not Fix Our Deployment

PR #21267 sets ownership of `/opt/hermes/ui-tui` and
`/opt/hermes/node_modules` to `hermes:hermes` at build time. The image's
entrypoint then `usermod`s the `hermes` user from UID 10000 to
`HERMES_UID/HERMES_GID` (per-user value from `hermes_users` vault). The
build-time `chown` is **resolved to numeric UID 10000** at layer
creation, so after `usermod` the files are still numerically owned by
10000 while our process runs as the per-user UID → EACCES persists.

Concretely: a deployment with `hermes_users[*].uid = 1001` keeps hitting
the same `npm ERR! code EACCES` on `rename(2)`, because the build-time
`chown` does not propagate through the entrypoint's UID remap.

The `copy-ui-tui` initContainer in this repo was therefore still
required after #21267 lands. Removing it would only be safe if upstream
either (a) extends the entrypoint to `chown -R "$HERMES_UID:$HERMES_GID"
/opt/hermes/ui-tui /opt/hermes/node_modules` after `usermod`, or (b)
makes those trees world-writable, or (c) ships the Trigger 2 logic fix
so `_tui_need_npm_install()` stops returning `True` spuriously and the
reinstall never fires.

> **Condition (a) is now satisfied (2026-05-31).** PR #33045 (merged
> 2026-05-27, in v0.15.0 / v2026.5.28) adds exactly `chown -R hermes:hermes
> $INSTALL_DIR/ui-tui $INSTALL_DIR/node_modules` to `docker/stage2-hook.sh`
> **after** the UID remap. So on v2026.5.28+ the initContainer is removable.
> We are still pinned at `hermes.image_tag: "v2026.5.16"` (v0.14.0), so it
> stays mandatory until that bump. The deployment comment "only chowns
> $HERMES_HOME, not /opt/hermes/" is correct for v0.14.0 but will be stale
> once we move to v0.15.0+.

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
  — **closed** 2026-05-07 as completed via PR #21267. Note: only the
  image-permission symptom is resolved; the underlying logic bug
  (Trigger 2) remains and the issue was closed without addressing it.
- Logic fix PR (Trigger 1 only): [hermes-agent#19520](https://github.com/NousResearch/hermes-agent/pull/19520)
  — merged 2026-05-04. Trigger 2 still unfixed in `main`.
- Image-permission fix PR: [hermes-agent#21267](https://github.com/NousResearch/hermes-agent/pull/21267)
  — merged 2026-05-07; salvages original PR
  [#19303](https://github.com/NousResearch/hermes-agent/pull/19303).
  Adds `chown -R hermes:hermes /opt/hermes/ui-tui /opt/hermes/node_modules`
  to the Dockerfile. Validated with `tests/tools/test_dockerfile_node_modules_perms.py`.
- First release containing both fixes: **v2026.5.7 / v0.13.0** (2026-05-07,
  "The Tenacity Release"). Current latest release: **v2026.6.5 / v0.16.0
  (2026-06-06)** — Trigger 2 still unfixed in the reinstall logic, but
  neutralised at build time by `npm_config_install_links=false` (unchanged).
  Re-checked 2026-06-11.
- **Entrypoint chown fix:** [hermes-agent#33045](https://github.com/NousResearch/hermes-agent/pull/33045)
  — merged 2026-05-27, shipped in **v0.15.0 / v2026.5.28**. Adds
  `chown -R hermes:hermes $INSTALL_DIR/ui-tui $INSTALL_DIR/node_modules` to
  `docker/stage2-hook.sh` **after** the UID remap → satisfies condition (a).
  `copy-ui-tui` initContainer becomes removable once `hermes.image_tag` ≥ v2026.5.28.
- Our Trigger 2 report (workspace-link entries) posted 2026-05-04 as
  comment on #18800:
  [#issuecomment-4371280956](https://github.com/NousResearch/hermes-agent/issues/18800#issuecomment-4371280956).
  Includes the diagnostic snippet from this document and a one-line
  patch suggestion (add `wanted[k].get("link")` to the missing-check
  filter). Since #18800 is now closed and no follow-up issue has been
  filed upstream, **this report still needs to be re-filed as a fresh
  issue** if we want upstream to act on Trigger 2. Re-checked 2026-06-11:
  no matching open issue found — filing is still pending.
- **PR #38591** (removes `--tui` flag from `hermes dashboard`) — merged
  2026-06-04, shipped in v0.16.0 / v2026.6.5. Re-verified 2026-06-11.
- **PRs #19520 / #21267** — merged (2026-05-04 / 2026-05-07). Re-verified 2026-06-11.
- **Issue #18800** — closed 2026-05-07 via PR #21267. Re-verified 2026-06-11.
- **PR #19303** — closed unmerged. Re-verified 2026-06-11.
- **Re-verified 2026-06-19:** hermes-agent **v2026.6.5** (2026-06-06) is still
  the latest release — no newer tag published. All tracked merges (#33045,
  #38591, #19520, #21267 merged; #18800 closed) still hold. The `copy-ui-tui`
  initContainer-removal action item (Action Items §1–4) remains pending —
  not yet executed.

## Action Items

> **2026-06-11 — NOW ACTIONABLE.** `hermes.image_tag` is pinned to
> **v2026.6.5**, which contains **both** PR #33045 (entrypoint chowns
> `$INSTALL_DIR/ui-tui` + `node_modules` after UID remap, merged 2026-05-27,
> shipped v0.15.0) **and** the build-time `npm_config_install_links=false`
> that neutralises Trigger 2. Items 1–4 below are therefore no longer
> hypothetical — they can be executed now.
>
> **Pending config changes (do NOT make without explicit approval):**
> - Test a pod with `copy-ui-tui` initContainer **disabled**; if EACCES is
>   gone, drop the initContainer + its volume + mount from
>   `roles/k8s_dgx/templates/hermes/hermes_agent_deployment.yaml.j2`.
> - The comment block in that template around lines ~172-181 (which states
>   the entrypoint only chowns `$HERMES_HOME`, not `/opt/hermes/`) is now
>   **outdated** for the pinned image (v2026.6.5 chowns `ui-tui` +
>   `node_modules` too). Update or remove that comment block when the
>   initContainer is dropped. Note: the template itself is NOT edited here
>   — this is a pending follow-up action.

1. Verify `_tui_need_npm_install()` still returns `True` in v2026.6.5 (it
   will, Trigger 2 is still unfixed in the reinstall logic — but
   `npm_config_install_links=false` neutralises it at build time).
2. Confirm `/opt/hermes/ui-tui` is owned by numeric UID 10000 inside
   v2026.6.5 (PR #21267 + PR #33045 effect).
3. Re-run a per-user pod with `hermes_users[*].uid != 10000` and the
   `copy-ui-tui` initContainer **disabled**; if no EACCES → initContainer
   is safe to drop. If EACCES still occurs → re-enable and investigate.
4. Once step 3 is green: drop the `copy-ui-tui` initContainer + its
   emptyDir volume + mount from `hermes_agent_deployment.yaml.j2`, and
   update the now-outdated entrypoint comment block in that template
   (lines ~172-181). Update this doc accordingly.
5. Open a fresh upstream issue referencing this document for Trigger 2
   (since #18800 is closed and no follow-up issue has been filed as of
   2026-06-11).
