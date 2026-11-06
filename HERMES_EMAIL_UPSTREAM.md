# Hermes email gateway: local patch and upstream PRs

## Why this exists

The upstream email adapter (`plugins/platforms/email/adapter.py` in
`NousResearch/hermes-agent`) has gaps that make it awkward as a long-lived agent mailbox:

1. **No folder lifecycle.** Processed mail stays in `INBOX` (only `\Seen`). A busy INBOX
   becomes unreadable, and after a crash nothing shows which mail was in progress.
2. **No Sent-folder copy.** Replies go out over SMTP and are never written back to IMAP,
   so there is no audit trail unless the SMTP provider auto-captures sent mail.
3. **Existing INBOX is ignored on startup.** Everything already in INBOX is marked seen and
   dropped, which is wrong for first boot, backlog ingestion or restart after downtime.
4. **Replies do not quote the received mail.**

All four are fixed in a local patch; each fix is also submitted upstream as its own PR.

## Local patch

- File: `roles/k8s_infra/files/hermes_email_gateway_patched.py`, a full copy of upstream
  `adapter.py` with `[PATCH-N]` sections. Its header records the synced upstream tag + blob
  and the per-section account of the last re-sync.
- Delivery (no fork, no rebuild): `roles/k8s_infra/tasks/apps/hermes.yml` renders it into
  ConfigMap `hermes-email-patch` (key `adapter.py`), subPath-mounted over
  `/opt/hermes/plugins/platforms/email/adapter.py` in the `hermes-gateway` sidecar (dashboard
  pod, or the webui pod when the dashboard is off). The `checksum/email-patch` pod annotation
  rolls the pod when the file changes.
- Must stay in lockstep with `hermes.image_tag` in `roles/k8s_infra/defaults/main/hermes.yml`.

### Config keys

All are read from `config.yaml` `platforms.email.extra.*` (not env), rendered by
`roles/k8s_infra/templates/hermes/hermes_config.yaml.j2` from `hermes_users[*].email.*` in
`group_vars/all/vault/hermes.yml`. An empty string opts out of that stage for that user.

| Key                | dgxarley default                      | Effect                                                                                |
|--------------------|---------------------------------------|---------------------------------------------------------------------------------------|
| `working_folder`   | `Hermes_Working`                      | INBOX → Working at fetch. `""` → INBOX → Done directly.                               |
| `done_folder`      | `Hermes_Done`                         | → Done after `handle_message()`. `""` → no moves at all (INBOX + `\Seen`).            |
| `sent_folder`      | `Sent`                                | IMAP APPEND (`\Seen`) after each successful SMTP send. `""` → SMTP only.              |
| `process_existing` | `true`                                | Process the pre-existing UNSEEN backlog on first poll (upstream PR default: `false`). |
| `quote_original`   | `hermes.email.quote_original_default` | Quote the received mail below the first successful reply to it.                       |

### Folder lifecycle

```
INBOX ──UID MOVE──▶ Hermes_Working ──handle_message() returns (try/finally)──▶ Hermes_Done
                    (visible after a crash =
                     "interrupted mid-processing")
```

The Working MOVE only fires when both folders are set AND the mail has a Message-ID, so mail
is never stranded in Working. The `try/finally` covers every drop path (self, automated,
not allowlisted, unauthenticated From) as well as exceptions.

### Move semantics

`_imap_move` prefers `UID MOVE` (RFC 6851), falls back to `UID COPY` + `UID STORE +FLAGS
\Deleted` + `UID EXPUNGE` (UIDPLUS), and finally to a global `EXPUNGE`, which expunges every
`\Deleted` mail in the folder (only legacy servers; Dovecot, Gmail, mailcow, M365, Cyrus 2.5+
support both extensions).

### dgxarley-only sections

- `[PATCH-10]`: upstream defaults `imap_security` to `tls` on every port, which breaks our
  port-143 mailboxes. The patch restores the port-derived default (993 → tls, else starttls);
  an explicit setting still wins.

## Upstream PRs

Opened against `NousResearch/hermes-agent:main` from the fork `vroomfondel/hermes-agent`, one
feature per PR so they can merge independently. All still **open** (no merge yet).

| PR                                                                  | Branch                        | Adds                                                                                                                                          |
|---------------------------------------------------------------------|-------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------|
| [#28697](https://github.com/NousResearch/hermes-agent/pull/28697)   | `feat/email-sent-folder`      | `platforms.email.sent_folder`, `_imap_append_to_sent()`, APPEND in the adapter senders and `_standalone_send`                                 |
| [#28699](https://github.com/NousResearch/hermes-agent/pull/28699)   | `feat/email-process-existing` | `platforms.email.process_existing`, gating the startup pre-fill (default keeps upstream behaviour)                                            |
| [#28702](https://github.com/NousResearch/hermes-agent/pull/28702)   | `feat/email-folder-lifecycle` | `working_folder` + `done_folder`, `_ensure_folder` / `_imap_move` / `_search_message_id` / `_finalize_message`, `try/finally` around dispatch |
| [#113192](https://github.com/NousResearch/hermes-agent/pull/113192) | `feat/email-quote-original`   | `quote_original` (`[PATCH-11]`), incl. the `sender_granted` gate for pair/decline senders                                                     |

The PRs drift into `CONFLICTING`/`DIRTY` whenever upstream refactors the adapter; they are
then rebased onto `main` and pushed `--force-with-lease`.

### Fork clones

One clone per PR (not scratch worktrees): `~/hermes-fork-sent` (#28697),
`~/hermes-fork-existing` (#28699), `~/hermes-fork-folders` (#28702), `~/hermes-fork-quote`
(#113192); `~/hermes-fork` is the base. `origin` = fork, `upstream` = NousResearch. Run
tests with `env -u VIRTUAL_ENV` (dgxarley's own venv otherwise confuses `uv`):

```bash
cd ~/hermes-fork-<feature>
env -u VIRTUAL_ENV .venv/bin/python -m pytest tests/gateway/test_email.py -v
```

## Re-sync procedure

### On every `hermes.image_tag` bump

Check the live pod image first: keel (`policy: minor`, 24h poll) usually pulls a new tag
before the defaults move, and a `--tags hermes` run with the old pin rolls the pods back.

1. Fetch the new baseline and compare its blob with the one in the patch header:

   ```bash
   gh api "repos/NousResearch/hermes-agent/contents/plugins/platforms/email/adapter.py?ref=<tag>" \
     --jq '.content' | base64 -d > /tmp/adapter_new.py
   ```

   Also check `plugin.yaml` / `__init__.py` in the same directory (the mount target depends on
   the layout).
2. Byte-identical: only update the "synced to upstream tag" line in the patch header.
3. Changed: 3-way merge (`git merge-file`) of the black-formatted old baseline, our file and
   the black-formatted new baseline; re-apply conflicting `[PATCH-N]` sections by hand. Record
   what moved in the patch header.
4. Verify: `black --check`, `ast.parse`, pyflakes, diff vs the new baseline shows only
   `[PATCH-N]` sections. The patch must add no non-stdlib import beyond upstream's own.
5. Rollout happens with the next `--tags hermes` run (`checksum/email-patch` rolls the pod).

The same bump also re-checks the contracts listed above `hermes.image_tag` (the former
`hermes_health_patch.py` was retired 2026-10-11, see below).

### When an upstream PR merges

1. Bump `hermes.image_tag` to a release containing the merge.
2. Drop the corresponding `[PATCH-N]` section(s) and adapt the `config.yaml` keys if the
   merged shape differs.
3. Once every PR is merged and `[PATCH-10]` is no longer needed: drop the
   `hermes-email-patch` ConfigMap task, the subPath mount, the `checksum/email-patch`
   annotation and the patch file.

### Open watch item

- 2026-10-10: re-synced to v0.21.6 (adapter.py blob 5fb7141781, 60267 B). Upstream now
  authorizes BEFORE the RFC822 fetch (`81706c8b30`, bounded header preflight +
  `_mark_uid_consumed`), so a sender rejected there never reaches `_dispatch_message` and the
  [PATCH-6] finalize. [PATCH-5] gained an INBOX→Done MOVE inside `_mark_uid_consumed` to keep
  rejected mail out of INBOX. Sender authentication (`4f39bd4f35`) now fails CLOSED for every
  sender when `authserv_id` is unset and `require_authenticated_sender` is true (new env
  `EMAIL_AUTHSERV_ID`, warned once per account at connect); our default
  `require_authenticated_sender_default: false` is unaffected, a per-user `true` now also needs
  `authserv_id`. `c13ea774e6` (version_info) taken along. PRs #28697, #28699, #28702 and
  #113192 still open, no maintainer activity (`gh pr view` 2026-10-10).
- 2026-10-11: no change to the email baseline. Deployed tag v0.21.6 is a plain semver tag
  (annotated tag on `818c13be1d`, published 2026-10-08, no v2026.x date alias; upstream moved
  to a stable release pipeline, v0.21.7 only exists as rc tags so far). adapter.py at v0.21.6
  is blob 5fb7141781 (60267 B), identical to the blob named in the patch header. `c13ea774e6`,
  the six 2026-09-27 fix(email) commits, `9bcbe7b5df`, `33c7018054`, `81706c8b30` and
  `4f39bd4f35` are all ancestors of v0.21.6. `main` adapter.py is blob d983fd81c7 (60105 B),
  differing from the tag only by three ruff lint commits (`d76cce4f11`, `aff64c3eeb`,
  `2ff58849f4`, 2026-10-08/09). PRs #28697, #28699, #28702 and #113192 are open, rebased onto
  main `02c9603730` on 2026-10-10, now mergeable (state blocked, awaiting review); last
  maintainer comment remains teknium1 2026-06-15 on the first three, none on #113192. The
  dashboard container now also carries `API_SERVER_KEY` (365d026) so the upstream Bearer probe
  (#76051, `c061068b26`) can authenticate. Verified live 2026-10-11 on v0.21.6 (Bearer with that
  key: `/v1/models` and `/health/detailed` 200, 0 "rejected invalid API key"), after `hermes_env.j2`
  also started writing `API_SERVER_KEY` into `/opt/data/.env` (a stale per-user value there had
  overridden the container env). `hermes_health_patch.py` and the venv-probe pod that fed its
  mounts are retired.
- 2026-10-12: upstream released v0.21.7 (2026-10-12, 777 commits over v0.21.6). Its adapter.py is blob d983fd81c7 (identical to main), differing from the synced v0.21.6 base 5fb7141781 only by the three ruff lint commits (`d76cce4f11`, `aff64c3eeb`, `2ff58849f4`); no functional email change, so no re-sync is needed until `hermes.image_tag` moves (deployed and pinned tag is still v0.21.6). No upstream change to the dashboard gateway-health probe or `API_SERVER_KEY` handling since 10-11. PRs #28697, #28699, #28702 and #113192 are still open with no maintainer activity since 2026-10-10 (heads unchanged); upstream main has advanced to `7318e666c2` since the 10-10 rebase onto `02c9603730`, #28699 reports MERGEABLE, the other three mergeability UNKNOWN.
- 2026-10-12 (bump): `hermes.image_tag` → v0.21.7, patch re-synced to blob d983fd81c7 (60105 B). 3-way merge had two trivial conflicts (`_sender_accepted` docstring, `_send_email_with_attachments(final=)`); our [PATCH-N] sections took the same lint (builtin generics, unquoted annotations, `# noqa: BLE001` dropped), so the diff vs baseline is unchanged in shape. Upstream email tests on v0.21.7 with the patch: same 7 expected divergences as on v0.21.6 (process_existing default, [PATCH-10], `final=` kwarg), unpatched baseline 101/101. Image contracts unchanged (v0.21.6..v0.21.7 touches dashboard/gateway only with lint; `89963452ee` makes TERMINAL_LOCAL_PERSISTENT default true, matching prior real behaviour). PRs rebased onto upstream/main `f594424e58` without conflicts (backups `backup/pre-rebase-2026-10-12-<name>`), RUF100/UP006/UP037 fixups folded into the introducing commits. The blocking `scripts.code_health` ratchet (BLE001 ignores `# noqa`) was cleared: IMAP handlers narrowed to `(imaplib.IMAP4.error, OSError)`, the two best-effort boundaries (`_imap_append_to_sent`, `_finalize_message`) keep a catch-all that logs `exc_info=True`, `_ensure_folder` reads `imap.create(name)[0]` (MagicMock-backed upstream connect tests), the #28702 lifecycle tests moved to `tests/gateway/test_email_folders.py` (`test_email.py` is back to upstream byte-for-byte). Heads existing 68c3bd0f99 / sent f626f877dd / folders d3dfa7d948 / quote e85287cc83: ruff clean, code_health 0 blocking, email tests 108/117/131/137 green; pushed `--force-with-lease` 2026-10-12, all four MERGEABLE (BLOCKED = awaiting review). The local patch keeps its older, simpler [PATCH-3] excepts (not subject to code_health).
- 2026-10-12 (image pin, temporary): the git tag v0.21.7 was out before its image; `nousresearch/hermes-agent:v0.21.7` did not exist on Docker Hub yet (ImagePullBackOff). As a one-off exception `hermes.image_tag` pins `rc.15-v0.21.7` (git tag on the same commit `71d34bad10` as `v0.21.7`, so the patch baseline is identical). Switch back to `v0.21.7` once that image is published.
- 2026-10-14 (bump): `hermes.image_tag` → `v0.22.0` (git tag on `e06df9c2ea`, published 2026-10-14; Docker Hub `v0.22.0` = `latest` = `stable` = `rc.3-v0.22.0`), which also ends the `rc.15-v0.21.7` exception (a plain `v0.21.7` image never appeared). `plugins/platforms/email/` is unchanged v0.21.7..v0.22.0 (adapter.py still blob d983fd81c7, 60105 B), so only the header tag line moved. Upstream email tests on v0.22.0 with the patch give the same 6 expected divergences as on v0.21.7 under the same harness (unpatched baseline 71/71 on both tags). Image contracts unchanged: no commit in the range touches `GATEWAY_HEALTH_URL`, `HERMES_DASHBOARD_BASIC_AUTH_*`, `API_SERVER_KEY`, `HERMES_ORIG_CWD`, `HERMES_GATEWAY_LOCK_DIR`, `/health/detailed` or the dashboard flags; the `TERMINAL_CWD` hits are named-profile resume fixes (`21a9a1f0ec`, `b23600860f`) and tests; `docker/`/`Dockerfile` untouched; `web_server.py` only adds desktop-backend boot hooks; dependency changes are limited to dropping the `langfuse` extra and the uv `exclude-newer` block (ruamel.yaml still pinned). PRs #28697, #28699, #28702 and #113192 still open with heads unchanged since 10-12.
- 2026-10-23 (bump): `hermes.image_tag` → `v0.22.4` (published 2026-10-21; Docker Hub `v0.22.4` = `latest` = `stable`). Live deployments were still on `rc.15-v0.21.7` and scaled to 0, so v0.22.0 was never rolled out and keel had not drifted. adapter.py v0.22.0..v0.22.4: blob d983fd81c7 → 49abaf5f6a (60105 → 61613 B); the directory gained `gateway_env.py` (`__init__.py`/`plugin.yaml` unchanged; only adapter.py is subPath-mounted, so the relative `from . import gateway_env` resolves against the image). Upstream changes are confined to imports, `serial_receive_loop = True` on `EmailAdapter` and `register()` (send-target parser, `StandaloneSendPolicy`, dashboard metadata, `platform_hint`, `env_overrides`, and `allowed_users_env`/`allow_all_env` moved into `**gateway_env.AUTHZ`, which adds `allowlist_suppresses_pairing: True`). 3-way merge without conflicts; the patch delta (diff vs baseline) is byte-identical before and after (693 lines), and no non-stdlib import was added. `serial_receive_loop` (`de526521ab`) only lets `handle_message()` return once a busy-burst input is held; it already returned after spawning a background task, so Working→Done keeps its meaning ("handed to the gateway"). `allowlist_suppresses_pairing` sends fewer pair replies; the adapter-side `sender_granted` computation is unchanged, so the [PATCH-11] gate is unaffected. Upstream email tests on v0.22.4 (py3.14 venv): unpatched 71/71, patched the same 6 expected divergences as the old patch on v0.22.0 (identical test IDs). Image contracts: `GATEWAY_HEALTH_URL`, `HERMES_DASHBOARD_BASIC_AUTH_*`, `HERMES_ORIG_CWD`, `/health/detailed` and the dashboard flags have identical occurrences on both tags; the `API_SERVER_KEY`/`TERMINAL_CWD` deltas are a docstring and CLI/desktop refactors; the `HERMES_GATEWAY_LOCK_DIR` delta is Electron desktop code only. `Dockerfile` (`--cache` for the uv seed) and `docker/entrypoint-dispatch.sh` (`S6_LOGGING` only with a TTY; our pods set none) are harmless. PRs #28697, #28699, #28702 and #113192 still open (heads updated 2026-10-16, mergeability UNKNOWN).
- 2026-10-23 (PR rebase): all four PR branches rebased onto upstream/main `8edc5ea255` without conflicts (backups `backup/pre-rebase-2026-10-23-<name>`), pushed `--force-with-lease`. Heads sent 4872631987 / existing 1d9e4a889e / folders 539f7de229 / quote e064c73d5f: ruff 0.15.22 clean, `scripts/check --only health` 0 blocking, email tests 87/78/101/107 green. All four MERGEABLE (BLOCKED = awaiting review).
- 2026-11-06 (bump): `hermes.image_tag` → `v0.22.7` (published 2026-11-03; on Docker Hub). `plugins/platforms/email/` is unchanged v0.22.4..v0.22.7 (adapter.py still blob 49abaf5f6a, 61613 B), so only the header tag line moved. All base-class methods the patch calls still exist; `base.py` changes are Docker sandbox and media paths. Image contracts: `GATEWAY_HEALTH_URL`, `HERMES_ORIG_CWD`, `/health/detailed` and the `"notify"` marker have identical occurrence counts, api_server still routes `/health` + `/health/detailed`. `plugins/dashboard_auth/basic` now ignores a signing secret containing `$(`, a backtick or `openssl rand` (per-process key instead); our derived sha512 hex contains none, and decoding is unchanged. `Dockerfile`: Debian trixie base, `tmux`, extras `fal` + `documents`, webapp build stage; `stage2-hook.sh` skips the config migration only with `HERMES_CONFIG_BACKEND=remote`, which we do not set. Upstream email tests not re-run (baseline blob identical). Live state via context `ht@kikube-dgx`: all `hermes-agent-*` Deployments still on `rc.15-v0.21.7` and scaled to 0, no pods, so keel has not drifted. PRs #28697, #28699, #28702 and #113192 still open (mergeability UNKNOWN on 2026-11-04).
- 2026-11-06 (bump): `hermes.image_tag` → `v0.23.0` (published 2026-11-05; on Docker Hub for amd64 + arm64), superseding the v0.22.7 pin before it was rolled out. `plugins/platforms/email/` is unchanged v0.22.7..v0.23.0 (adapter.py still blob 49abaf5f6a, 61613 B), so only the header tag line moved. The base-class methods the patch calls still exist; `base.py` only adds `configured_bot_user` to `build_source`. The `"notify"` delta (31 → 36 outside tests) is the unrelated `wake` value in the new `tools/bot_mode_dm*.py`; the stamp in `gateway/platforms/base.py` is unchanged, so [PATCH-11] is unaffected. Image contracts: `GATEWAY_HEALTH_URL`, `HERMES_DASHBOARD_BASIC_AUTH_*`, `API_SERVER_KEY`, `HERMES_ORIG_CWD`, `TERMINAL_CWD`, `HERMES_GATEWAY_LOCK_DIR` and `/health/detailed` have identical occurrence counts. s6: per-profile gateway slots moved from `/run/service` to `/run/hermes-service` under the new `profile-gateways` service (`s6-svscan` as hermes); `015-supervise-perms` narrows to `root:hermes`, `02-reconcile-profiles` now hard-fails on its `chown`. Read against our pods (hermes + hermes-gateway start as root with `HERMES_UID` remap, no capability drop): no blocker. The dashboard container skips the reconcile, `gateway run` still redirects to slot `gateway-default` (one gateway; the adapter subPath mount applies in the same container), the slot run script is byte-identical and skips `s6-setuidgid` when non-root. Not verified by a boot: whether `profile-gateways` (depends on `base` only) starts after cont-init; a longrun should restart until the scandir exists. Release notes: the ten `browser_*` tools and agent-browser are gone (`browser_exec` only), user skills/prompts on `/opt/data` naming them must move; Word/PowerPoint/Excel skills merged into `microsoft-office`. Upstream email tests not re-run (baseline blob identical). Live state unchanged: `rc.15-v0.21.7`, scaled to 0, so this is the first v0.22+ rollout.


## Operational notes

- `imap.create()` runs on every connect; "already exists" returns `NO` and is swallowed.
- Sent-folder APPEND failures are logged as warnings and swallowed; they never turn into a
  failed SMTP send.
- A mail stuck in `Hermes_Working` means the gateway process died before the `finally` ran.
