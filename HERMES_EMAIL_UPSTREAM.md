# Hermes email gateway — local patch and upstream PRs

Status as of 2026-05-31.

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
  tag `v2026.5.16` (commit `0fffb82d0b949820c380019de646a46a0a6de678` of
  `gateway/platforms/email.py`) with seven `[PATCH-N]` sections applied.

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

All four ship per-user overrideable in `group_vars/all/main.yml` /
`group_vars/all/vault.yml` under the `email:` block; defaults are populated
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

1. Bump `hermes_image_tag` in `roles/k8s_dgx/defaults/main.yml` to a
   release tag that contains the merge.
2. Remove the corresponding `[PATCH-N]` section(s) from
   `roles/k8s_dgx/files/hermes_email_gateway_patched.py`. The
   patch-header lists the seven sections by ID.
3. If all three PRs land, retire the local patch entirely: drop the
   `hermes-email-patch` ConfigMap task, drop the subPath mount, drop the
   `checksum/email-patch` annotation, drop `roles/k8s_dgx/files/hermes_email_gateway_patched.py`,
   trim the `email.*_folder` / `email.process_existing` fields back to
   plain env values in the secret.

### When `hermes_image_tag` is bumped before a PR merges

> **2026-05-31 check:** Latest upstream release is now **v2026.5.29.2 (v0.15.2)**.
> `gateway/platforms/email.py` blob SHA is still `0fffb82…` (the same commit as
> our pinned `v2026.5.16`) at v2026.5.29.2 — **identical, no re-sync required**.
> (Two `email.py`-adjacent commits landed in between — `fix(email): use real
> hermes version in IMAP ID command` and `…send IMAP ID extension to support
> 163/NetEase mailbox` — but they live outside `gateway/platforms/email.py`, so
> the blob SHA is unchanged and the patch stays clean.) All three PRs (#28697,
> #28699, #28702) are still open/unmerged (last activity 2026-05-19). The repo
> is still pinned at `hermes_image_tag: v2026.5.16` — if bumping to v2026.5.29.2,
> re-check the email.py SHA at that tag first.

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
  `group_vars/all/vault.yml`. Setting any of `working_folder`,
  `done_folder`, or `sent_folder` to `""` opts out of the corresponding
  stage for that user without affecting cluster defaults.
