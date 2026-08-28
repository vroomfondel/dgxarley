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

- `roles/k8s_infra/files/hermes_email_gateway_patched.py` — synced to upstream
  tag `v2026.6.19` (blob `d2f7e64a`, md5 `a3f7dc61f40388bf806481b189b48e00`,
  32908 bytes of `gateway/platforms/email.py`) with seven `[PATCH-N]` sections
  applied. (Note, 2026-07-28: this path was corrected. Earlier text in this
  section named `roles/k8s_dgx/...`, but the file, its ConfigMap task and its
  mount all live under `roles/k8s_infra/`, see the paths below.)

It is delivered to the running container without a fork+rebuild:

- `roles/k8s_infra/tasks/apps/hermes.yml` creates a cluster-wide ConfigMap
  `hermes-email-patch` in the `hermes` namespace, with the file content under
  key `email.py`.
- `roles/k8s_infra/templates/hermes/hermes_webui_deployment.yaml.j2`
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
| [#28697](https://github.com/NousResearch/hermes-agent/pull/28697) | `feat/email-sent-folder`        | +371 / −1  | `platforms.email.sent_folder` config.yaml key (moved off the original `EMAIL_SENT_FOLDER` env in a 2026-06-15 rebase), a shared module-level `_imap_append_to_sent()` helper, `IMAP APPEND` call sites in all three `_send_email*` functions AND, since 2026-07-13, in the standalone SMTP path `_standalone_send` (used by `hermes send` and cron), added after a hermes-sweeper bot review flagged that path as missing coverage. |
| [#28699](https://github.com/NousResearch/hermes-agent/pull/28699) | `feat/email-process-existing`   | +146 / −10 | `platforms.email.process_existing` config.yaml key (moved off the original `EMAIL_PROCESS_EXISTING` env in a 2026-06-15 rebase); wraps the pre-fill loop in `connect()` in a conditional. Default keeps upstream behaviour. |
| [#28702](https://github.com/NousResearch/hermes-agent/pull/28702) | `feat/email-folder-lifecycle`   | +864 / −101 | `platforms.email.working_folder` + `platforms.email.done_folder` config.yaml keys (moved off the original envs in a 2026-06-15 rebase), `_open_imap` / `_ensure_folder` / `_imap_move` / `_search_message_id` / `_finalize_message` helpers, `try/finally` around `handle_message()`. Rebased again on 2026-06-21 across the plugin-migration conflict (commit `56001054`), moving the implementation from `gateway/platforms/email.py` to `plugins/platforms/email/adapter.py`. |

Note, 2026-07-28: the table above reflects the PRs' current state, not their
original submission. All three were substantially reshaped by review
feedback and rebases since first opened, see the diff sizes and the
per-PR notes above plus the re-verify log entries below. The original
diff sizes were `+148/-0`, `+154/-16` and `+580/-9`, and the original test
counts (`60 to 64/64/74`) are stale and no longer tracked here, since the
env-to-config.yaml rebases and, for #28702, the plugin-path move changed
the test layout each time. Author-reported counts from the review threads:
#28697 at 99 passing (2026-07-13), #28702 at 106/106 passing (2026-07-06,
up from 86 passing right after the 2026-06-21 plugin-migration rebase).
#28699's exact count was not restated in its review comments. Conventional
Commits format, MIT license inherited (no CLA required).

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
   `checksum/email-patch` annotation, drop `roles/k8s_infra/files/hermes_email_gateway_patched.py`,
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

> **2026-07-28 check, no new release, adapter.py unchanged, no merge, PR table above corrected
> to match current PR content:**
> - **Latest release:** still `v2026.7.20` (2026-07-20), no new tag published since the 2026-07-23
>   check.
> - **`plugins/platforms/email/adapter.py` unchanged** on both `main` and the pinned tag since
>   commit `88bd1c01` (2026-07-02), blob `572b5c11455d396e3d23d44b7bf724130ebce385`, 50848 bytes,
>   identical to what the 2026-07-23 entry recorded. The local patch stays byte-clean, no re-sync
>   is required.
> - **PRs #28697 / #28699 / #28702** all still `open`, `merged: false`, `updated_at` unchanged at
>   2026-07-13. No new comments or reviews since. The trigger "if a PR merges, the local patch
>   becomes redundant" has not fired.
> - **The "Upstream PRs" table above was rewritten** to reflect the PRs' current diffs
>   (`+371/-1`, `+146/-10`, `+864/-101`) and current mechanism (config.yaml keys under
>   `platforms.email.*`, not the original env vars), instead of the shape they had when first
>   opened. This is a documentation correction only. Nothing changed upstream between the
>   2026-07-23 and 2026-07-28 checks beyond ordinary review back and forth that had already
>   happened by 2026-07-13.
> - **Path correction:** the "Local patch" section and the re-sync steps in this doc previously
>   named `roles/k8s_dgx/files/hermes_email_gateway_patched.py`,
>   `roles/k8s_dgx/tasks/hermes.yml` and `roles/k8s_dgx/templates/hermes/hermes_webui_deployment.yaml.j2`.
>   Verified against the repo, all three live under `roles/k8s_infra/` instead
>   (`roles/k8s_infra/files/hermes_email_gateway_patched.py`,
>   `roles/k8s_infra/tasks/apps/hermes.yml`,
>   `roles/k8s_infra/templates/hermes/hermes_webui_deployment.yaml.j2`). Corrected in place above.

> **2026-08-03 check — new release v2026.7.30 (adapter unchanged there, no re-sync), but `main`
> has diverged again:**
> - **Latest release:** now **v2026.7.30** (v0.19.1, published 2026-07-30) — a rollup/patch
>   release with no curated changelog (~2,789 commits since v0.19.0; notes explicitly deferred
>   to v0.20.0). `plugins/platforms/email/adapter.py` at this tag is **byte-identical** to the
>   pinned v2026.7.20 baseline (blob `572b5c11…`) — the pinned deployment and local patch are
>   unaffected, no re-sync forced.
> - **`hermes.image_tag` bumped to `v2026.7.30` on 2026-08-03** (same day). The bump was
>   catch-up, not a rollout: keel had already pulled the tag on 2026-07-31 13:38 (the hermes
>   Deployments carry `keel.sh/policy: minor` + a 24h poll), so the pods had been running
>   v2026.7.30 for three days while `roles/k8s_infra/defaults/main/hermes.yml` still said
>   v2026.7.20 — a re-run of `--tags hermes` would have rolled the image BACK. Full patch-fit
>   check at the new tag came back clean (adapter byte-identical as above, plugin.yaml +
>   `__init__.py` identical, health-patch anchors intact, image/CLI/config contracts
>   unchanged), and both patches were verified live in the running 7.30 pod.
> - **`main` diverged 2026-08-02:** commit `ff89f1b862` ("fix(email): Slack-pattern helper for
>   unscoped default-profile adapter + scope ports/trust flag") modifies the email adapter —
>   blob now `f224202b…`, 52592 bytes (+1744 vs pinned) — not yet in any released tag. (An
>   earlier intermediate commit `f08f403157`, 2026-07-05, also touched the file without
>   shipping in a tag.) Same watch-pattern as the 2026-06-12/06-14 divergence warnings:
>   **re-sync check mandatory on the next `hermes.image_tag` bump.**
> - **PRs #28697 / #28699 / #28702** all still `open`, `merged: false`, `updated_at` unchanged
>   at 2026-07-13. No new activity.

> **2026-08-03 follow-up — the 2026-08-02 `main` divergence broke #28702's mergeability,
> fixed by a merge of current `main` (pushed):**
> - **Merge state after `ff89f1b862` / `f08f403157`:** #28702 is now `mergeable: CONFLICTING`,
>   `mergeStateStatus: DIRTY` (head `e84f4cc62e`, last rebase 2026-07-06 onto `3b5c64543`).
>   #28697 (`6bd60442`) and #28699 (`c05d0e06`) are still `MERGEABLE` / `BLOCKED` — git
>   auto-merges them, they only wait on required checks/approval. So the divergence costs
>   exactly one PR a rebase, not all three.
> - **The conflict is a single hunk** in `_dispatch_message`. Upstream's two commits move the
>   `EMAIL_*` reads to the profile-scoped `agent.secret_scope.get_secret` (via the new
>   `_get_esecret` / `_esecret_int` / `_esecret_bool` helpers, replacing `utils.env_int` /
>   `env_bool`), including the three allowlist drop-check reads (`EMAIL_ALLOWED_USERS`,
>   `EMAIL_ALLOW_ALL_USERS`). #28702 re-indents that exact block into its new `try/finally`,
>   so the two edits touch the same lines. Every other #28702 hunk auto-merges, and the PR
>   adds no env reads of its own (its four knobs are `config.extra` keys, not env).
> - **Resolution:** keep the `try`-scoped indentation from #28702, adopt main's `_get_secret`
>   reads verbatim. Merge commit `70de728cb` (parents `e84f4cc62e` + `7997c9ced8`) built in the
>   per-PR fork clone `/home/thiess/hermes-fork-folders` (branch `feat/email-folder-lifecycle`,
>   `origin` = `vroomfondel/hermes-agent`, `upstream` = `NousResearch/hermes-agent`; one clone
>   per PR: `-folders` = #28702, `-sent` = #28697, `-existing` = #28699). The local branch was
>   4 commits behind `origin` first (the 2026-07-06 merge was made in the GitHub UI) and was
>   fast-forwarded before merging. Post-merge diff vs `upstream/main` is unchanged at
>   `+864 / −101` and `tests/gateway/test_email.py` +
>   `test_email_robustness.py` + main's new `test_email_secret_scope.py` are **60/60 passing**.
> - **Pushed** to `vroomfondel/hermes-agent` `feat/email-folder-lifecycle`
>   (`e84f4cc62..70de728cb`) on 2026-08-03. GitHub re-evaluated #28702 to
>   `mergeable: MERGEABLE`, `mergeStateStatus: BLOCKED`, head `70de728cb0` — i.e. back in line
>   with #28697 / #28699, waiting only on required checks/approval. Still `open`, not merged.
> - **Local deployed patch is NOT affected.** `hermes.image_tag` is pinned to `v2026.7.30`
>   (bumped from `v2026.7.20` on 2026-08-03; both tags carry the same adapter blob
>   `572b5c11…`), which is still what `roles/k8s_infra/files/hermes_email_gateway_patched.py`
>   is synced against; the scoped-secret rewrite lives only on `main` and in no released tag.
>   When it does ship, the re-sync will have to fold the same `os.getenv` → `_get_esecret`
>   change into the `[PATCH-6]` `try/finally` region.

> **2026-08-07 check — the divergence HAS shipped: new release v2026.8.3 contains the
> scoped-secret adapter. Re-sync trigger is now armed on a real tag:**
> - **Latest release:** **v2026.8.3** (v0.20.0, "The Herald Release", published
>   2026-08-03T16:57:52Z — hours after the entries above were written). Its
>   `plugins/platforms/email/adapter.py` is blob `f224202b…`, 52592 bytes — **byte-identical
>   to `main`'s `ff89f1b862` divergence** flagged above. So the next `hermes.image_tag` bump
>   (v2026.7.30 → v2026.8.3 or later) fires the mandatory re-sync: fold the
>   `os.getenv` → `_get_esecret` scoped-secret change into the `[PATCH-6]` `try/finally`
>   region — the same resolution already built and validated for #28702's merge commit
>   `70de728cb` (see the 2026-08-03 follow-up above), so the conflict shape is known.
> - **Keel caveat (same trap as last time):** the hermes Deployments run `keel.sh/policy:
>   minor` with a 24h poll, and v0.19.1 → v0.20.0 is a minor bump — the running pods may
>   already have pulled v2026.8.3 while the pinned default still says v2026.7.30. Check the
>   live pod image before assuming "pinned deployment unaffected", and before any `--tags
>   hermes` re-run (which would roll BACK to the pinned tag).
> - **`main` has NOT diverged further:** `ff89f1b862` (2026-08-02) is still the newest commit
>   touching the adapter (re-checked 2026-08-05 and 2026-08-07).
> - **PRs #28697 / #28699 / #28702** all still `open`, `merged: false`, heads unchanged
>   (`6bd60442` / `c05d0e06` / `70de728cb`), all `MERGEABLE`/`BLOCKED`. No maintainer
>   activity since the 2026-08-03 push.

> **2026-08-09 check — synchronized PR rebase on 2026-08-07, and `main` diverged
> FURTHER with a data-loss-relevant charset fix (not in any tag yet):**
> - **All three PRs were rebased simultaneously 2026-08-07T17:05:53–17:06:03Z**
>   (same pattern as the 2026-06-29 event): new heads #28697 `6bd60442` →
>   `8e9790ce8f`, #28699 `c05d0e06` → `2ebfd651dd`, #28702 `70de728cb` →
>   `c1e9f056d9`. #28702 is now a single linear commit (author-dated
>   2026-05-19) — the merge-commit structure `70de728cb` recorded in the
>   2026-08-03 follow-up is gone, but the content is unchanged (diff still
>   +864/−101). All three still `open`, `mergeable: true` /
>   `mergeable_state: blocked`, no new review comments — a routine
>   rebase-onto-fresh-`main` (base `eaa53de4eb`, `main` HEAD at
>   2026-08-07T16:52:16Z), not maintainer feedback.
> - **New `main`-only adapter divergence:** commit `65f407184d`
>   (2026-08-08T18:55:14Z, "fix(email): never let unknown or malformed
>   charsets abort the IMAP fetch", fixes upstream #35901/#55381/#55383;
>   +49/−9 on `plugins/platforms/email/adapter.py`). Adds `_safe_decode()`
>   (charset alias table: `unknown-8bit`→utf-8, `gb2312/gbk`→gb18030,
>   `ks_c_5601-1987`→cp949, …, then utf-8, then latin-1 fallback) and
>   rewrites `_decode_header_value()` + `_extract_text_body()` to use it.
>   **This is a real data-loss bug that our pinned patched file has too:**
>   `hermes_email_gateway_patched.py` (lines ~489/501) uses
>   `.decode(charset, errors="replace")` — `errors="replace"` only guards
>   bad byte sequences, not an unknown/invalid codec *name*. A garbage
>   charset label (e.g. QQ Mail's RFC1428 `unknown-8bit`) raises
>   `LookupError`, aborting the whole fetch batch — and since UIDs are
>   marked seen before fetch, those messages are **permanently dropped**.
> - **Not in any release:** v2026.8.3 (still the latest tag, no new release)
>   still carries adapter blob `f224202b…`/52592 bytes; current `main` blob
>   is `8698dc93…`/54088 bytes (`main` HEAD `2446c8bb67`). The 2026-08-07
>   rebase base predates `65f407184d`, so none of the three PR branches
>   include the charset fix either. The next `hermes.image_tag` bump re-sync
>   therefore folds in BOTH the scoped-secret change (entry above) AND this
>   charset fix. Given the data-loss severity, pulling `_safe_decode()`
>   forward as an out-of-band edit to the local patch is worth considering
>   before any tag bump (config change — needs explicit approval, not done).
> - **Follow-up same day (2026-08-09, approved):** both points above are now
>   resolved. (a) The charset fix was forward-ported into
>   `hermes_email_gateway_patched.py` as section **`[PATCH-9]`**
>   (`[PATCH-8]` was already taken by the Sent-APPEND section) — remove that
>   section at the next tag re-sync once the baseline contains
>   `65f407184d`. (b) All three PRs were rebased again onto current `main`
>   (`2446c8bb67`, which includes `65f407184d`): new heads #28697
>   `0249e60835`, #28699 `8c517fcd6e`, #28702 `5e6f9d217e` — all clean
>   rebases, own diffs byte-identical (range-diff verified), pushed
>   `--force-with-lease` to the fork, all still `MERGEABLE`/`BLOCKED`.

> **2026-08-15 check — release v2026.8.13 shipped the charset fix; tag bump +
> re-sync + fresh PR rebases all executed same day (approved):**
> - Upstream released **v2026.8.13** (v0.20.1, 2026-08-13). Its `adapter.py`
>   (blob `317eae72`, 60496 bytes) is the FIRST tagged baseline containing
>   `65f407184d` (charset/`_safe_decode`), plus three further fixes over the
>   v2026.8.3 baseline: `a7f0abc845` (partial-batch dispatch, seen-after-fetch
>   UIDs, reconnect UID-baseline restore via `_seen_uids_snapshot`),
>   `9b8da52f41` (IMAP fetch failures surfaced through the fatal-error hook),
>   `91bc822330` (terminal connect-failure classification / retry escalation).
>   `main` additionally has `480342232a` (close leaked poller sockets,
>   2026-08-15), in no tag yet, NOT ported (we sync to tags).
> - **Re-sync done:** `hermes.image_tag` bumped `v2026.8.3` -> `v2026.8.13`;
>   `hermes_email_gateway_patched.py` rebuilt on the v2026.8.13 baseline.
>   **`[PATCH-9]` retired** (baseline now contains `65f407184d` natively,
>   present exactly once, no duplicate). [PATCH-2]/[PATCH-3]/[PATCH-6]/
>   [PATCH-7]/[PATCH-8] reapplied unchanged; **[PATCH-4] adapted** (our
>   `process_existing` gate now nests inside upstream's new "first connect /
>   no snapshot" branch, the `is_reconnect` snapshot-restore path stays pure
>   upstream); **[PATCH-5] adapted** (upstream extracted per-message parsing
>   into `_parse_fetched_message()`; the Working-folder MOVE now runs in the
>   caller after a non-None parse result). Verified: ast OK, diff vs baseline
>   contains only [PATCH-1..8], all four `config.extra` knobs still read.
>   Rollout happens at the next `--tags hermes` run (not executed here).
> - Side findings from the bump check: `GATEWAY_HEALTH_URL` still present and
>   still deprecated at v2026.8.13 (still the only cross-container health
>   mechanism); `HERMES_DASHBOARD_BASIC_AUTH_*` env vars unchanged, though
>   BasicAuthProvider moved into a bundled plugin
>   (`plugins/dashboard_auth/basic/`) with a new optional config.yaml
>   alternative (env still wins, deployment unaffected);
>   `hermes_health_patch.py` anchor unchanged.
> - **PR status:** GitHub had flipped #28699/#28702 to `CONFLICTING`/`DIRTY`
>   (heads unchanged; `main` moved under them, `a7f0abc845` colliding with our
>   `connect()`/dispatch hunks). All three PRs re-rebased onto `main`
>   `7a16840a` (2026-08-15): new heads #28697 `21715520f1` (clean,
>   byte-identical own diff), #28699 `f98737b7f8` (one conflict in
>   `connect()`, resolved by nesting `process_existing` under upstream's
>   first-connect branch, mirroring the [PATCH-4] re-sync shape), #28702
>   `3a916d7ff3` (two conflicts: folder-ensure moved inside upstream's
>   try/finally before the `is_reconnect` branch; Working-MOVE relocated
>   after `_parse_fetched_message()`, mirroring [PATCH-5]). All adapter tests
>   pass in each clone, range-diff reviewed, pushed `--force-with-lease` to
>   the fork, all three back to `MERGEABLE`/`BLOCKED`.

> **2026-08-17 check — release v2026.8.16 shipped the fd-leak fix; tag bump +
> re-sync executed same day:**
> - Upstream released **v2026.8.16** (v0.20.2, 2026-08-16). Exactly ONE commit
>   touched `plugins/platforms/email/adapter.py` since v2026.8.13:
>   `480342232a` ("fix(gateway): close leaked poller sockets in weixin/email
>   adapters", #79889) — the commit the previous entry flagged as "on `main`,
>   in no tag yet, NOT ported". New baseline blob `704524e4`, 62120 bytes.
>   `plugin.yaml` and `__init__.py` are byte-identical, so the ConfigMap
>   subPath mount target is unchanged.
> - **What the commit does:** adds a module-level `_close_imap(imap)` that
>   calls `logout()` and, on ANY exception, chases it with `shutdown()`.
>   `IMAP4.logout()` only guards `OSError`, but a broken connection makes
>   `_simple_command('LOGOUT')` raise `IMAP4.abort` (not an `OSError`), so
>   `logout()` propagates before its own `shutdown()` and the socket stays
>   open — one leaked fd per failed poll/connect until `[Errno 24] Too many
>   open files`.
> - **Re-sync done:** `hermes.image_tag` bumped `v2026.8.13` -> `v2026.8.16`;
>   `hermes_email_gateway_patched.py` rebuilt on the v2026.8.16 baseline. Both
>   upstream call sites land on our anchors, so this was a real reweave:
>   **[PATCH-4] re-indented** into upstream's new inner `try/finally` in
>   `connect()` (`imap = None` … `finally: if imap is not None:
>   _close_imap(imap)`), with our three `imap.logout()` calls removed
>   alongside upstream's and the `_seen_uids_snapshot` assignment left after
>   the inner block; **`_fetch_new_messages()`** gained upstream's
>   `imap: Optional[imaplib.IMAP4] = None` and its `finally` now calls
>   `_close_imap(imap)` (the [PATCH-3] `_open_imap()` routing and the
>   [PATCH-5] Working-MOVE are untouched by the diff).
>   [PATCH-1/2/6/7/8] reapplied unchanged at identical anchors. **dgxarley
>   extension:** our own two IMAP teardowns (`_imap_append_to_sent`,
>   `_finalize_message` — both [PATCH-3] code that does not exist upstream)
>   had the identical leaky `try: logout() except: pass` and were routed
>   through `_close_imap` too. Verified: `ast.parse()` OK, `black --check`
>   clean, full diff vs the v2026.8.16 baseline inspected hunk-by-hunk
>   (only [PATCH-1..8] + black reformatting + `[UPSTREAM]` comment markers,
>   nothing upstream dropped). Rollout happens at the next `--tags hermes`
>   run (not executed here).
> - Side findings from the bump check: `hermes_health_patch.py` **still
>   required and unchanged** — `APIServerAdapter._check_auth(self, request)`
>   still exists, `/health/detailed` still calls it while `/health` and the
>   `/v1/health` alias never do, and `_probe_gateway_health` still sends a
>   bare `urllib.request.Request(path, method="GET")` with no Authorization
>   header. `GATEWAY_HEALTH_URL` still present, still deprecated, still the
>   only cross-container mechanism. Dashboard CLI flags
>   (`--host`/`--insecure`/`--port`/`--no-open`),
>   `HERMES_DASHBOARD_BASIC_AUTH_*` reads, `gateway.lock`/
>   `gateway_state.json` all unchanged.
> - **PR status:** no work needed. `main` has contained `480342232a` since
>   2026-08-15, and all three PRs were already rebased past it on 2026-08-16
>   (heads #28697 `75775e5e91`, #28699 `16f1753624`, #28702 `aecb6942a5`);
>   all three are `MERGEABLE`/`BLOCKED` as of this check.

> **2026-08-21 check — no change, two new rollup releases confirmed clean, PR rebase cause clarified:**
> Latest releases **v2026.8.16.2** (2026-08-17) and **v2026.8.18** (2026-08-18) are both changelog-deferred rollups; `plugins/platforms/email/adapter.py` is byte-identical to our pinned `v2026.8.16` baseline (blob `704524e4`, 62120 bytes) across both tags and current `main`, so no re-sync is needed. `main`'s adapter path was last touched by `480342232a` (2026-08-15, already folded in). PRs #28697/#28699/#28702 remain `OPEN`/`MERGEABLE`/`BLOCKED`, heads unchanged since 2026-08-17 (`75775e5e91`/`16f1753624`/`aecb6942a5`), no activity since. Clarification: the 2026-08-16/17 rebase that produced those heads was in response to an automated `Enough1122` AI-review pass posted 2026-08-15T18:47 on all three PRs (author addressed the points in the same rebase), not solely the `main` divergence past `480342232a` as the prior entry implied. Conclusions unchanged.

> **2026-08-28 check — two new releases confirmed clean, main diverged again (unrelated to our patch anchors), PRs unchanged:**
> - **Latest releases:** v2026.8.19 (2026-08-21) and v2026.8.27 (2026-08-27), both since the 2026-08-21 check's v2026.8.18. `plugins/platforms/email/adapter.py` is byte-identical to the pinned v2026.8.16 baseline (blob 704524e4, 62120 bytes) across both new tags, so the pinned deployment and local patch remain unaffected, no re-sync forced.
> - **`main` diverged again, not yet in any tag:** current `main` HEAD adapter.py is blob 89ead8a8, 62238 bytes (+118 vs the pinned baseline). The added content is two lines at the very end of `connect()`'s success path, right before `return True`: a comment plus `self._wire_plugin_handlers(None)`, from commit 272f4e4a ("feat(plugins): generalize native platform handler registration to every gateway platform"). This sits after our reweaved [PATCH-4] try/finally block and does not collide with any [PATCH-N] anchor, diff verified line-by-line (2 lines added, nothing else). Not in v2026.8.19 or v2026.8.27 yet. Watch on the next tag bump.
> - **PRs #28697/#28699/#28702** remain OPEN/MERGEABLE/BLOCKED, heads unchanged since 2026-08-17 (75775e5e91/16f1753624/aecb6942a5), no new comments or reviews.

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
