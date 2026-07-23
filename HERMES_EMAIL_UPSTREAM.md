# Hermes email gateway — local patch and upstream PRs

Status as of 2026-06-30.

## Why this exists

The upstream `gateway/platforms/email.py` in `nousresearch/hermes-agent` has
three gaps that make the email adapter awkward to run as a long-lived agent
mailbox:

1. **No folder lifecycle.** Every processed mail stays in `INBOX` (just with
   the `\Seen` flag set, as a side effect of the `RFC822` FETCH). On a busy
   mailbox INBOX quickly becomes unreadable, and after a crash there is no
   way to tell which messages the agent was still working on.
2. **No Sent-folder copy.** Replies go out over SMTP and are never written
   back to the user's IMAP. Unless the SMTP provider auto-captures sent mail
   (Gmail does, most generic Dovecot/Postfix setups do not), the user has no
   audit trail of what the agent answered.
3. **Hard-coded "ignore existing INBOX" on startup.** Anything sitting in
   INBOX at startup is silently marked as seen and ignored forever. That is
   the right default for a long-running install, wrong for first-boot,
   intentional backlog ingestion, or restart after downtime.

The dgxarley cluster needs all three behaviours fixed, so a local patch was
written; three parallel PRs were also opened against upstream to push the
fixes back.

## Local patch (dgxarley)

A patched copy of the upstream file lives at:

- `roles/k8s_dgx/files/hermes_email_gateway_patched.py` — synced to upstream
  tag `v2026.6.19` (blob `d2f7e64a`, md5 `a3f7dc61f40388bf806481b189b48e00`,
  32908 bytes of `gateway/platforms/email.py`) with seven `[PATCH-N]` sections
  applied.

It is delivered to the running container without a fork+rebuild:

- `roles/k8s_dgx/tasks/hermes.yml` creates a cluster-wide ConfigMap
  `hermes-email-patch` in the `hermes` namespace, with the file content under
  key `email.py`.
- `roles/k8s_dgx/templates/hermes/hermes_webui_deployment.yaml.j2`
  subPath-mounts that key over `/opt/hermes/gateway/platforms/email.py` in
  the `hermes-email` sidecar (gated on `email.enabled`). The rest of
  `/opt/hermes` (venv, ui-tui, hooks, etc.) remains the image-baked tree.
- A `checksum/email-patch` annotation on the webui pod template
  auto-rolls when the patch file changes (subPath ConfigMap mounts do not
  surface updates without a pod restart).

### Environment variables added

All four ship per-user overrideable in `group_vars/all/main/hermes.yml` /
`group_vars/all/vault/hermes.yml` under the `email:` block; defaults are populated
by `roles/k8s_dgx/templates/hermes/hermes_env.j2`. Empty string opts out
of the respective stage.

| Variable                  | Default          | Effect                                                                                  |
|---------------------------|------------------|-----------------------------------------------------------------------------------------|
| `EMAIL_WORKING_FOLDER`    | `Hermes_Working` | INBOX → Working at fetch. `""` → skip the intermediate stage, INBOX → Done directly.    |
| `EMAIL_DONE_FOLDER`       | `Hermes_Done`    | Working → Done (or INBOX → Done) after `handle_message()` returns. `""` → no moves.     |
| `EMAIL_SENT_FOLDER`       | `Sent`           | IMAP APPEND target (with `\Seen`) after each successful SMTP send. `""` → SMTP-only.    |
| `EMAIL_PROCESS_EXISTING`  | `0`              | `1` → skip the upstream "mark all existing INBOX UIDs as seen on startup" pre-fill.     |

### Folder lifecycle in one picture

```
                       ┌───────────────────┐
                       │      INBOX        │
                       └─────────┬─────────┘
                                 │  UID MOVE (or COPY+EXPUNGE)
                                 ▼
                       ┌───────────────────┐
                       │  Hermes_Working   │   visible after a crash =
                       │                   │   "interrupted in mid-
                       └─────────┬─────────┘   processing"
                                 │
                                 │  handle_message() returns
                                 │  (try/finally — fires on success
                                 │   OR exception)
                                 ▼
                       ┌───────────────────┐
                       │   Hermes_Done     │   audit trail of every mail
                       │                   │   the agent has completed
                       └───────────────────┘
```

For outbound mail: after SMTP `send_message()` succeeds, the same MIME bytes
are mirrored via `IMAP APPEND` to `Sent` with the `\Seen` flag.

### Move semantics

The IMAP `_imap_move` helper prefers `UID MOVE` (RFC 6851). On servers
without MOVE it falls back to `UID COPY` + `UID STORE +FLAGS \Deleted` +
`UID EXPUNGE` (RFC 4315 UIDPLUS). If `UID EXPUNGE` is also unsupported the
helper falls back to a global `EXPUNGE`, which expunges every
`\Deleted`-flagged mail in the folder — documented limitation, only matters
on legacy servers (Dovecot, Gmail, mailcow, M365, Cyrus 2.5+ all support
both extensions).

## Upstream PRs

All three PRs were opened against `NousResearch/hermes-agent:main` from the
fork `vroomfondel/hermes-agent`. Each is scoped to one feature so reviewers
can merge them independently; if one stalls the others are not blocked.

| PR                                                                | Branch                          | Diff       | What it adds                                                                                                            |
|-------------------------------------------------------------------|---------------------------------|------------|-------------------------------------------------------------------------------------------------------------------------|
| [#28697](https://github.com/NousResearch/hermes-agent/pull/28697) | `feat/email-sent-folder`        | +148 / −0  | `EMAIL_SENT_FOLDER` env, `_append_to_sent()` helper, `IMAP APPEND` call site in all three `_send_email*` functions.     |
| [#28699](https://github.com/NousResearch/hermes-agent/pull/28699) | `feat/email-process-existing`   | +154 / −16 | `EMAIL_PROCESS_EXISTING` env; wraps the pre-fill loop in `connect()` in a conditional. Default keeps upstream behaviour. |
| [#28702](https://github.com/NousResearch/hermes-agent/pull/28702) | `feat/email-folder-lifecycle`   | +580 / −9  | `EMAIL_WORKING_FOLDER` + `EMAIL_DONE_FOLDER` envs, `_open_imap` / `_ensure_folder` / `_imap_move` / `_search_message_id` / `_finalize_message` helpers, `try/finally` around `handle_message()`. |

All three PRs pass the existing `tests/gateway/test_email.py` suite
unchanged and add new tests of their own (60 → 64 / 64 / 74 tests
respectively). Conventional Commits format, MIT license inherited (no
CLA required).

### Local worktrees (for follow-up review feedback)

The fork is cloned at `/home/thiess/hermes-fork`, with one `git worktree`
per PR:

- `/home/thiess/hermes-fork-sent`      → `feat/email-sent-folder`
- `/home/thiess/hermes-fork-existing`  → `feat/email-process-existing`
- `/home/thiess/hermes-fork-folders`   → `feat/email-folder-lifecycle`

All three worktrees share a single venv at `/home/thiess/hermes-fork/.venv`
(symlinked from each worktree). To run the test suite in any worktree:

```bash
cd /home/thiess/hermes-fork-<feature>
env -u VIRTUAL_ENV .venv/bin/python -m pytest tests/gateway/test_email.py -v
```

The `env -u VIRTUAL_ENV` prefix is required because the parent shell's
`VIRTUAL_ENV` (dgxarley's own venv) otherwise confuses `uv`.

## Re-sync procedure

### When upstream merges one of our PRs

1. Bump `hermes.image_tag` in `roles/k8s_infra/defaults/main/hermes.yml` to a
   release tag that contains the merge.
2. Remove the corresponding `[PATCH-N]` section(s) from
   `roles/k8s_dgx/files/hermes_email_gateway_patched.py`. The
   patch-header lists the seven sections by ID.
3. If all three PRs land, retire the local patch entirely: drop the
   `hermes-email-patch` ConfigMap task, drop the subPath mount, drop the
   `checksum/email-patch` annotation, drop `roles/k8s_dgx/files/hermes_email_gateway_patched.py`,
   trim the `email.*_folder` / `email.process_existing` fields back to
   plain env values in the secret.

### When `hermes.image_tag` is bumped before a PR merges

> **2026-05-31 check:** Latest upstream release is now **v2026.5.29.2 (v0.15.2)**.
> `gateway/platforms/email.py` blob SHA is still `0fffb82…` (the same commit as
> our pinned `v2026.5.16`) at v2026.5.29.2 — **identical, no re-sync required**.
> (Two `email.py`-adjacent commits landed in between — `fix(email): use real
> hermes version in IMAP ID command` and `…send IMAP ID extension to support
> 163/NetEase mailbox` — but they live outside `gateway/platforms/email.py`, so
> the blob SHA is unchanged and the patch stays clean.) All three PRs (#28697,
> #28699, #28702) are still open/unmerged (last activity 2026-05-19). The repo
> is still pinned at `hermes.image_tag: v2026.5.16` — if bumping to v2026.5.29.2,
> re-check the email.py SHA at that tag first.

> **2026-06-08 check:** Repo is now pinned at `hermes.image_tag: v2026.6.5`
> (**v0.16.0**, released 2026-06-06). `gateway/platforms/email.py` is still
> **byte-identical** across `v2026.5.16 .. v2026.6.5` and `main` (blob SHA
> `0fffb82d0b949820c380019de646a46a0a6de678`, md5 `318ae8f3e6d4b26718784e0c94bf8458`,
> 29097 bytes) — **no re-sync required**; the patch in
> `roles/k8s_dgx/files/hermes_email_gateway_patched.py` stays clean (its header
> already records this). All three PRs (#28697, #28699, #28702) remain open.

> **2026-06-11 check:** All three PRs (#28697, #28699, #28702) still open —
> keine Bewegung seit 2026-05-19. `gateway/platforms/email.py` blob SHA
> unverändert (`0fffb82d`) sowohl in v2026.6.5 als auch in `main` →
> **kein Re-Sync nötig**. v2026.6.5 ist weiterhin das aktuellste Release.

> **2026-06-12 check — ⚠ Upstream-Divergenz, Re-Sync beim nächsten Tag-Bump erforderlich:**
> Commit `f03f161b` landete am 2026-06-12T08:07:50Z auf `main`
> (`fix(gateway): classify email document attachments as DOCUMENT`) und
> ändert `gateway/platforms/email.py` → neuer Blob-SHA
> `4eb4972b24ec5b2e2a2b3e06624e456cf501badc` (29585 Bytes, +488 gegenüber `0fffb82d`).
> Die Änderung liegt im geerbten Upstream-Code innerhalb `_dispatch_message()`,
> **nicht** in einem unserer `[PATCH-N]`-Abschnitte:
> - Alt: `if att["type"] == "image": msg_type = MessageType.PHOTO`
> - Neu: Guard `and msg_type == MessageType.TEXT` auf den PHOTO-Zweig +
>   neuer `elif att["type"] == "document": msg_type = MessageType.DOCUMENT`
>   (DOCUMENT schlägt PHOTO bei gemischten Anhängen).
>
> **Gepinnte Version v2026.6.5 ist nicht betroffen** (Blob dort weiterhin
> `0fffb82d`) — kein sofortiger Handlungsbedarf. Beim nächsten Bump von
> `hermes.image_tag` auf einen Release, der `f03f161b` enthält, muss die
> neue DOCUMENT-Klassifizierung in
> `roles/k8s_dgx/files/hermes_email_gateway_patched.py` eingearbeitet werden
> (Schritt 2 der Re-Sync-Prozedur unten). Alle drei PRs (#28697, #28699,
> #28702) bleiben offen.

> **2026-06-14 check — ⚠ Weitere Upstream-Divergenz, Delta wächst:**
> Zwei neue Commits auf `main` am 2026-06-14, beide im `_send_email*`-Bereich
> (den auch unsere `[PATCH-N]`-Abschnitte berühren):
> - `fix(email): use SMTP_SSL for port 465 and fall back to IPv4 on timeout`
> - `fix(email): make IPv4 SMTP fallback use supported sockets`
>
> Neuer `main`-Blob-SHA: `7b247cdd`, **32736 Bytes** (+3151 gegenüber
> `4eb4972b`, +3639 gegenüber `0fffb82d`). **Gepinnte Version v2026.6.5
> ist weiterhin nicht betroffen** (Blob dort unverändert `0fffb82d`, 29097
> Bytes) — kein sofortiger Handlungsbedarf. Beim nächsten Bump auf einen
> Release, der diese Commits enthält, überschneiden sich die Upstream-Änderungen
> mit unseren `[PATCH-N]`-Abschnitten in `_send_email*` — Re-Sync wird
> aufwändiger als nach dem 2026-06-12-Delta. Alle drei PRs (#28697, #28699,
> #28702) bleiben offen. Kein neues Release-Tag (v2026.6.5 ist weiterhin
> das aktuellste).
>
> **2026-06-16 check — PRs unter aktiver Review (nicht mehr ruhend):** Alle drei
> PRs (#28697, #28699, #28702) sind weiterhin offen/ungemergt, aber **nicht
> dormant** — am 2026-06-15 gab es neue Review-Aktivität: Upstream-Reviewer
> (hermes-sweeper-Bot + Maintainer) haben Inline-Kommentare gesetzt, der
> PR-Autor hat am selben Tag geantwortet. Tenor des Feedbacks: die ENV-Vars
> (`EMAIL_SENT_FOLDER` etc.) sollen nach `config.yaml` unter
> `platforms.email.*` wandern (passt zur laufenden config.yaml-Migration).
> Die frühere Formulierung „keine Bewegung seit 2026-05-19" ist damit überholt.
> `main`-Blob-SHA unverändert `7b247cdd` (32736 Bytes); gepinnte v2026.6.5
> weiterhin nicht betroffen, kein neues Release-Tag.

> **Re-verified 2026-06-19:** hermes-agent v2026.6.5 still latest; PRs
> #28697 / #28699 / #28702 still open.

> **2026-06-21 — ✅ RE-SYNC DONE + tag bumped to v2026.6.19:** New release
> `v2026.6.19` (2026-06-19) carries the divergence flagged on 06-12/06-14.
> `gateway/platforms/email.py` is now blob `d2f7e64a` (md5
> `a3f7dc61f40388bf806481b189b48e00`, 32908 bytes; +3811 vs `0fffb82d`).
> `hermes.image_tag` bumped `v2026.6.5` → `v2026.6.19`, and the patch in
> `roles/k8s_dgx/files/hermes_email_gateway_patched.py` was re-synced against
> the new baseline. Folded-in upstream changes (all upstream-only, none
> collided with a `[PATCH-N]` section):
> - **SMTP port-aware connect + IPv4 fallback** — new module helpers
>   `_create_ipv4_connection` / `_IPv4SMTP` / `_IPv4SMTP_SSL` and a new
>   `EmailAdapter._connect_smtp()` (port 465 → implicit `SMTP_SSL`, else
>   `STARTTLS`; retries connection-level failures over IPv4 only). All four
>   SMTP call sites (`connect()` test + the three `_send_email*` senders) route
>   through it; our `[PATCH-7]` `_append_to_sent` calls sit *after* each SMTP
>   block and are unaffected. (Subsumes the 06-14 `7b247cdd` commits.)
> - **DOCUMENT attachment classification** in `_dispatch_message()`'s media
>   loop (`f03f161b`, 06-12 delta): image only wins while still `TEXT`; a
>   `document` attachment promotes to `MessageType.DOCUMENT`.
> - `send_image()` gained a `metadata` kwarg (base-class contract); new
>   `import socket`.
>
> Verification: all 107 upstream-added lines present (whitespace-insensitive),
> all removed direct-`smtplib.SMTP` call sites gone, `py_compile` + `black`
> (line-length 120) clean, all seven `[PATCH-N]` markers + `_append_to_sent` /
> `_finalize_message` / `_open_imap` / `process_existing` intact. subPath mount
> target `/opt/hermes/gateway/platforms/email.py` unchanged (v2026.6.19 is
> still pre-refactor). PRs #28697 / #28699 / #28702 rebased onto current `main`
> the same day (see below).
>
> **⚠ NEXT bump past v2026.6.19 — plugin refactor (commit `56001054`, NOT PR #41112):** `main`/`latest`
> after 2026-06-19 MOVE this file to `plugins/platforms/email/adapter.py` and
> replace the static `_PLATFORMS["email"]` dict with a `register_platform()`
> registry. The refactor landed as commit `56001054` (merged 2026-06-20, "refactor(gateway):
> migrate slack/dingtalk/whatsapp/matrix/feishu/telegram/wecom/email/sms adapters to bundled
> plugins") — the earlier reference to PR `#41112` was incorrect (that PR number does not
> exist on the upstream repo; `56001054` is the real merge commit). A future bump to a
> release containing `56001054` must (1) re-target the patch to
> `plugins/platforms/email/adapter.py` and (2) change the subPath `mountPath` in
> `hermes_webui_deployment.yaml.j2` from `/opt/hermes/gateway/platforms/email.py` to
> `/opt/hermes/plugins/platforms/email/adapter.py`.

> **2026-06-24 check — PINNED image unaffected; plugin refactor landed on `main`; PRs still open:**
> - **Latest release:** `v2026.6.19` (v0.17.0, 2026-06-19) — unchanged; no new tag published.
>   `gateway/platforms/email.py` at tag `v2026.6.19` is blob `d2f7e64a` (md5
>   `a3f7dc61f40388bf806481b189b48e00`, 32908 bytes), which matches our patch header exactly.
>   **Pinned deployment is unaffected.**
> - **Plugin refactor landed on `main`:** commit `56001054` (merged 2026-06-20, "refactor(gateway):
>   migrate slack/dingtalk/whatsapp/matrix/feishu/telegram/wecom/email/sms adapters to bundled
>   plugins") moved `gateway/platforms/email.py` out of the file tree entirely. The file now
>   lives at `plugins/platforms/email/adapter.py` (blob `3961d812`, ~1022 lines). The earlier
>   mention of PR `#41112` in this doc has been corrected in-place above — that PR number
>   returns 404 on the upstream repo; `56001054` is the real merge commit.
> - **Two further main-only commits** landed on `plugins/platforms/email/adapter.py` after the
>   refactor move: a host/config resolution fix (2026-06-20) and a blank-env OOM fix
>   (2026-06-21). Neither is in any released tag.
> - **Our patch features remain upstream-exclusive:** grep of
>   `plugins/platforms/email/adapter.py` (blob `3961d812`) for `_append_to_sent`,
>   `_finalize_message`, `_imap_move`, `_ensure_folder`, `working_folder`, `done_folder`,
>   `sent_folder`, `process_existing` = **zero hits**. All three features (folder lifecycle,
>   sent-folder APPEND, process-existing gate) are present only in our local patch.
> - **PRs #28697 / #28699 / #28702** remain open; `mergeable_state: blocked` (last updated
>   2026-06-22).
> - **Re-sync note:** when bumping `hermes.image_tag` to any release that includes commit
>   `56001054`, the patch target path changes from `gateway/platforms/email.py` to
>   `plugins/platforms/email/adapter.py` (see "NEXT bump" note in the 2026-06-21 block above).

> **2026-06-30 check — PRs updated 2026-06-29; v2026.6.19 still latest:**
> - **Latest release:** `v2026.6.19` (v0.17.0, 2026-06-19) — unchanged; no new tag published.
>   Pinned deployment remains unaffected.
> - **PRs #28697 / #28699 / #28702** all received a simultaneous update at ~11:53–11:54 UTC on
>   2026-06-29 (all three `updatedAt` timestamps coincide — consistent with a rebase / force-push
>   across the three branches). Plausibly the author responding to the earlier ENV→`config.yaml`
>   review feedback noted in the 2026-06-16 check. PRs remain open / unmerged.
> - `gateway/platforms/email.py` at tag `v2026.6.19` is unchanged (blob `d2f7e64a`); our patch
>   is still clean. No re-sync required.

> **2026-07-06 check — ⚠ New release ships the plugin refactor; re-sync target now confirmed:**
> - **Latest release:** `v2026.7.1` (v0.18.0, "The Judgment Release"), published 2026-07-01T20:08:06Z —
>   supersedes the 2026-06-30 check's "v2026.6.19 still latest" note.
> - **The plugin refactor anticipated in the 2026-06-21/06-24 "NEXT bump" warnings (commit
>   `56001054`) is now IN a real tag.** At ref `v2026.7.1`, `gateway/platforms/email.py` is
>   **gone (404)**; the file now lives at `plugins/platforms/email/adapter.py`, blob
>   `c9d1cb499fe6f31068119414540d2d1f61d1e095`, 49488 bytes.
> - **Our patch features remain upstream-exclusive:** grep of the new adapter.py for
>   `_append_to_sent`, `_finalize_message`, `_imap_move`, `_ensure_folder`, `working_folder`,
>   `done_folder`, `sent_folder`, `process_existing` = **zero hits**, same as the 2026-06-24 check
>   against the pre-release `main` blob.
> - **PRs #28697 / #28699 / #28702** all still open, `updatedAt` unchanged since 2026-06-29T11:53–54Z
>   — no new activity.
> - **This makes the "NEXT bump" re-sync warning above directly actionable:** when
>   `hermes.image_tag` is bumped to `v2026.7.1` (or later), the patch target moves from
>   `gateway/platforms/email.py` to `plugins/platforms/email/adapter.py`, and the subPath
>   `mountPath` in `hermes_webui_deployment.yaml.j2` must change to
>   `/opt/hermes/plugins/platforms/email/adapter.py`, exactly as described there.
> - **Pinned deployment (`hermes.image_tag: v2026.6.19`) is unaffected** — no action forced yet.

> **2026-07-23 check — ✅ RE-SYNC ALREADY DONE (visible in patch header, not yet logged here);
> new security fix folded in; tag bumped past v2026.7.1:**
> - `hermes.image_tag` is now pinned to **v2026.7.7.2** (two releases past v2026.7.1). The
>   header of `roles/k8s_infra/files/hermes_email_gateway_patched.py` shows the re-sync
>   anticipated by the 2026-07-06 entry has already been carried out: the adapter is synced
>   to upstream tag **v2026.7.7.2** (`plugins/platforms/email/adapter.py`, md5
>   `39ed5d135762806451a944a9b279b8ad`, 50848 bytes), superseding the v2026.7.1 baseline. The
>   subPath mount was re-targeted to `/opt/hermes/plugins/platforms/email/adapter.py` as
>   described in the "NEXT bump" note above — this log entry is catching the doc up to a
>   change already applied in-repo, not announcing a new one.
> - **New upstream security fix folded in during that re-sync:** **GHSA-rxqh-5572-8m77**
>   (sender-authentication hardening) — new module-level `_domain_of` / `_domains_aligned` /
>   `_verify_sender_authentication` helpers plus `_AUTH_METHOD_RE` / `_AUTH_PROP_RE` and an
>   `EmailAdapter._require_authenticated_sender` field (env `EMAIL_TRUST_FROM_HEADER` /
>   config). This advisory and its fix were not previously mentioned anywhere in this doc's
>   re-sync log.
> - **No further re-sync needed for the newer release:** `plugins/platforms/email/adapter.py`
>   is byte-identical between `v2026.7.7.2` and the newer `v2026.7.20` (2026-07-20) — same
>   blob SHA `572b5c11455d396e3d23d44b7bf724130ebce385`, 50848 bytes. The pinned patch stays
>   clean at the current tag. **`hermes.image_tag` was subsequently bumped to v2026.7.20 on
>   2026-07-23** (adapter re-fetched at that ref, verified byte-identical to the v2026.7.7.2
>   base); the patch-file header now records v2026.7.20 as the checked tag.
> - **PRs #28697 / #28699 / #28702** remain open/unmerged; all three show `updated_at:
>   2026-07-13` — later than the 2026-06-29 activity previously logged, but no merge/close.

1. Download the new upstream file:

   ```bash
   gh api repos/NousResearch/hermes-agent/contents/gateway/platforms/email.py?ref=<new-tag> \
     --jq '.content' | base64 -d > /tmp/email_new.py
   ```

2. Re-apply each surviving `[PATCH-N]` section by hand against the new
   baseline. The patch header at the top of the file lists which sections
   exist and which functions they touch.
3. Update the "synced to upstream tag …" line in the patch header to the
   new tag + commit SHA.
4. Run a sanity import:

   ```bash
   python3 -c "import ast; ast.parse(open('roles/k8s_dgx/files/hermes_email_gateway_patched.py').read()); print('OK')"
   ```

5. Roll the webui pod — `checksum/email-patch` annotation will trigger
   automatically once the ConfigMap is re-applied.

## Operational notes

- The patch is delivered as a subPath ConfigMap mount over a single file
  inside `/opt/hermes/`. The rest of the image (`/opt/hermes/.venv`,
  `/opt/hermes/ui-tui`, etc.) is untouched. No fork+rebuild needed.
- `imap.create()` is called every connect — idempotent on every server we
  have seen (returns `NO` for "already exists", swallowed).
- Sent-folder APPEND failures are warning-logged and swallowed. A failed
  APPEND must never propagate as a failed SMTP send.
- The Done-folder move runs in a `try/finally` around `handle_message()`,
  so a crash in agent processing still moves the mail out of Working — a
  mail stuck in `Hermes_Working` only happens if the gateway itself
  terminated before the `finally` executed.
- Per-user overrides live in `hermes_users[*].email.*` in
  `group_vars/all/vault/hermes.yml`. Setting any of `working_folder`,
  `done_folder`, or `sent_folder` to `""` opts out of the corresponding
  stage for that user without affecting cluster defaults.
