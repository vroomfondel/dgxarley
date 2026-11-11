# LiteLLM Upstream Bug: `ollama/` Provider Forces TLS on Embedding Requests

## Summary

LiteLLM's `ollama/` provider ignores `ssl_verify: false` for embedding requests,
forcing TLS on plain HTTP connections. The `ollama_chat/` provider handles it correctly.

**Affected:** `ollama/` provider, embedding path only (`/api/embed`)
**Not affected:** `ollama_chat/` provider (chat path, `/api/chat`)
**Version:** observed on `ghcr.io/berriai/litellm:main-latest` (2026-03)

## Symptom

```
litellm.exceptions.APIConnectionError: OllamaException - Cannot connect to host
ollama-embed.ollama.svc.cluster.local:11434 ssl:<ssl.SSLContext object at 0x...>
[Connect call failed ('10.69.103.234', 11434)]
```

LiteLLM creates an `ssl.SSLContext` and attempts a TLS handshake against a plain HTTP
Ollama endpoint — even with `ssl_verify: false` in the model config.

## Root Cause

Two completely different HTTP client code paths:

### Chat path (`ollama_chat/`) — correct

1. `main.py` dispatches via `base_llm_http_handler.completion()`
2. `litellm_params` carries `ssl_verify=False` from the model config
3. `llm_http_handler.py` → `get_async_httpx_client(params={"ssl_verify": False})`
4. Cache key includes `ssl_verify_False` → separate no-SSL client is created and cached
5. Result: plain HTTP connection, works correctly

### Embedding path (`ollama/`) — broken

1. `main.py` calls `ollama.ollama_aembeddings()` directly (line ~5314)
2. **`ssl_verify` is never passed** to the embedding function
3. `handler.py` uses `litellm.module_level_aclient` — a **global singleton**
4. Singleton created by `_lazy_import_http_handlers()` with **no `ssl_verify` arg**
5. `HTTPHandler()` defaults → `get_ssl_configuration(None)` → full certifi SSL context
6. Cache key is just `"httpx_client"` — no `ssl_verify` variant exists
7. Result: TLS forced, connection to HTTP endpoint fails

### Code references (litellm source)

| File                                        | Line                         | Issue                                                           |
|---------------------------------------------|------------------------------|-----------------------------------------------------------------|
| `litellm/main.py`                           | ~5314                        | `ollama_aembeddings()` called without `ssl_verify`              |
| `litellm/llms/ollama/completion/handler.py` | embedding fn                 | Uses `litellm.module_level_aclient` singleton                   |
| `litellm/_lazy_imports.py`                  | `_lazy_import_http_handlers` | Creates `HTTPHandler(timeout=...)` without `ssl_verify`         |
| `litellm/llms/custom_httpx/http_handler.py` | `HTTPHandler.__init__`       | `ssl_verify=None` → `get_ssl_configuration(None)` → SSL enabled |

The source even contains: `[TODO]: migrate embeddings to a base handler as well.`
— that migration would fix this bug as a side effect.

## Workaround

HAProxy TLS sidecar in each Ollama pod:
- Self-signed cert generated at container start
- Listens on `ollama_tls_port` (11435) with TLS
- Proxies to `127.0.0.1:ollama_port` (11434) plain HTTP
- LiteLLM config points at `https://...:11435` with `ssl_verify: false`
  (cert verification disabled, but TLS handshake succeeds)

```yaml
# roles/k8s_dgx/defaults/main/litellm.yml
- model_name: "{{ ollama_embed_model }}"
  litellm_params:
    model: "ollama/{{ ollama_embed_model }}"
    api_base: "https://ollama-embed.ollama.svc.cluster.local:{{ ollama_tls_port }}"
    ssl_verify: false
```

See: `roles/k8s_dgx/tasks/ollama.yml` — ConfigMap `ollama-tls-haproxy-config`,
`tls-proxy` sidecar container in both `ollama` and `ollama-embed` Deployments.

## Upstream Status

**Still open** — re-verified 2026-07-06. No fix merged. LiteLLM has since
advanced to **v1.91.0** (stable, latest; published 2026-07-04 — contains no
ollama/ssl/embedding-related fix; intermediate releases v1.89.5 (07-02),
v1.90.1/v1.90.2/v1.90.3 (07-03) likewise touch none of it; v1.90.0 was
2026-06-27, v1.89.4 was 2026-06-25, v1.89.3 was 2026-06-20, v1.89.1 was
2026-06-16) and the bug is byte-identical: `ollama_aembeddings()` in
`litellm/llms/ollama/completion/handler.py` still calls
`litellm.module_level_aclient.post(...)` without `ssl_verify`, and the
`[TODO]: migrate embeddings to a base handler` comment is still at the top of
the file. (Previously tracked at v1.89.4, 2026-06-26.)
The ollama embedding path has still not been migrated to the base handler;
no PR addressing `ollama_aembeddings` + `ssl_verify` exists upstream as of
today, and the v1.91.0 release notes' embedding-related fixes (bedrock,
cohere, sagemaker, watsonx, caching) do not touch ollama or SSL.

- Related issue: [#6499](https://github.com/BerriAI/litellm/issues/6499) ("How to disable ssl verification for ollama?", closed). Maintainer acknowledged the embedding path was not migrated, but the issue was closed after only the chat path was fixed.
- The TODO comment `[TODO]: migrate embeddings to a base handler as well.` is still present at the top of `handler.py`.
- PR [#24704](https://github.com/BerriAI/litellm/pull/24704) — adjacent embedding bug (model name prefix stripping). **CLOSED unmerged 2026-07-05** (auto-closed after stale-bot inactivity, never merged) — the outcome this doc predicted after the 2026-06-27 stale flag. Never addressed SSL/TLS, so no loss to the ollama/ssl bug tracked here. **2026-07-06 note:** no upstream fix is in flight; the HAProxy TLS sidecar workaround below remains required.

**2026-07-23 re-verify — still broken at v1.93.0; more precise tracking issue found:**
LiteLLM has advanced from v1.91.0 to **v1.93.0** (stable, published 2026-07-19; intermediate
v1.90.6 / v1.91.4 / v1.92.1 also released 2026-07-19, plus in-progress v1.94.0-rc.x /
v1.95.0-dev.1 as of 2026-07-22). **The bug is confirmed still present at v1.93.0**:
`litellm/llms/ollama/completion/handler.py` still calls
`litellm.module_level_aclient.post(url=api_base, json=data)` with no `ssl_verify` argument
(same call site, ~line 82), and the `[TODO]: migrate embeddings to a base handler as well.`
comment is unchanged at the top of the file. A more precise upstream tracking issue has
surfaced: [#30778](https://github.com/BerriAI/litellm/issues/30778) ("`ssl_verify` not
propagated in `BaseLLMAIOHTTPHandler`"), filed 2026-06-18 by an unrelated third party, open
and un-triaged (0 comments). It documents the exact mechanism tracked in this doc — `ollama/`
embeddings route through `BaseLLMAIOHTTPHandler`, which never receives `ssl_verify` — plus a
second, related bug (the `AsyncHTTPHandler` retry path on `ConnectError`/`RemoteProtocolError`
also drops `ssl_verify`), and cross-references #6499, #17636, #9340, #26053, #21947. This
issue is now a better upstream reference than the closed #6499 for tracking a fix. Related
issue [#26053](https://github.com/BerriAI/litellm/issues/26053) (ssl_verify=false ignored for
streaming text completions) was closed on 2026-07-26 with `state_reason: not_planned`, auto-closed
by the stale-bot after no maintainer fix. Closed here does not mean fixed, the underlying bug
in that path is still unaddressed. No fix has landed for any of these paths — the HAProxy TLS
sidecar workaround remains required.

**2026-07-28 re-verify:** `litellm/llms/ollama/completion/handler.py` on `main` still calls
`litellm.module_level_aclient.post(...)` with no `ssl_verify` argument, the only commit touching
that file since 2026-06-27 is a pure ruff-format-width change, no functional edit. Issue #30778
remains open and uncommented since it was filed. The HAProxy TLS sidecar workaround remains
required.

**2026-08-07 re-verify — still broken at v1.95.0 (latest stable, 2026-08-03):** LiteLLM
shipped **v1.94.1** and **v1.95.0** (both stable, 2026-08-03; prereleases up to
v1.97.0-dev.1 as of 2026-08-05) — none of the release notes mention ollama, `ssl_verify`
or embedding fixes. The bug is byte-identical on current `main`: `handler.py` line 82
still calls `litellm.module_level_aclient.post(url=api_base, json=data)` with no
`ssl_verify`, `[TODO]` comment unchanged; the only commits touching the file (2026-08-01,
2026-08-04) are lint/ruff-enforcement passes with no functional change. Issue #30778
still open with 0 comments (untouched since 2026-06-18). **Two candidate fix PRs exist
but are stale and unmerged** (not previously listed here):
[#30810](https://github.com/BerriAI/litellm/pull/30810) ("fix(custom_httpx): propagate
ssl_verify through aiohttp handler + retry path (#30778)") and
[#30848](https://github.com/BerriAI/litellm/pull/30848) ("fix: propagate ssl_verify in
AsyncHTTPHandler retry paths and BaseLLMAIOHTTPHandler") — both open, both idle since
2026-06-20, no maintainer engagement. The HAProxy TLS sidecar workaround remains
required.

**2026-08-15 re-verify (still broken at v1.96.2, latest stable, 2026-08-11):**
LiteLLM also shipped **v1.95.1** and **v1.96.0** (both stable, 2026-08-11);
prereleases v1.97.0-rc.1 and v1.98.0-dev.2 exist but are not stable. The bug
is byte-identical on current `main`: `litellm/llms/ollama/completion/handler.py`
line 82 still calls `await litellm.module_level_aclient.post(url=api_base,
json=data)` with no `ssl_verify` argument, and the `[TODO]: migrate embeddings
to a base handler as well.` comment is unchanged at line 4. The only commits
touching the file remain the unrelated lint passes (2026-08-01 ruff autofix
b604e2b2, 2026-08-04 Final enforcement 2708620d). Issue #30778 remains open,
0 comments, idle since 2026-06-18. PRs #30810 and #30848 remain open, idle
since 2026-06-20. v1.96.0's only embedding-related release note (#35282,
provider stamping on embedding cache-hit spend logs) is unrelated, it touches
spend logging, not the HTTP client path. The HAProxy TLS sidecar workaround
remains required.

**2026-08-21 re-verify, still broken; new stable release v1.97.0 (2026-08-16) ships no fix:**
LiteLLM advanced from v1.96.2 to **v1.97.0** (stable, published
2026-08-16T02:51Z; prereleases v1.98.0-rc.1/dev.2 and v1.99.0-dev.1/dev.2
also exist but are not stable). Scanned the full v1.97.0 changelog (roughly
100 merged PRs), none touch ollama, `ssl_verify`, or the embedding HTTP
client path. The bug is byte-identical on current `main`:
`litellm/llms/ollama/completion/handler.py` line 4 still carries the
`[TODO]: migrate embeddings to a base handler as well.` comment, and line 82
still calls `await litellm.module_level_aclient.post(url=api_base,
json=data)` with no `ssl_verify` argument. No commits have touched the file
since 2026-08-04. Issue #30778 remains open, 0 comments, untouched since
filed 2026-06-18. PRs #30810 and #30848 remain open and unmerged, both idle
since 2026-06-20, now over two months with no maintainer engagement. The
HAProxy TLS sidecar workaround remains required.

**2026-08-28 re-verify, still broken; new stable release v1.98.0 (2026-08-23) ships no fix:**
LiteLLM advanced from v1.97.0 to **v1.98.0** (stable, published 2026-08-23T00:28:24Z;
prereleases v1.99.0-rc.1/dev.1/dev.2 and v1.100.0-dev.1/dev.2 also exist but are not
stable). A batch of backport point releases also landed the same window (v1.88.6 through
v1.96.2, all published 2026-08-11T22:0x, older-branch backports, not new content). Scanned
the v1.98.0 changelog: none of the entries touch ollama, `ssl_verify`,
`module_level_aclient`, `BaseLLMAIOHTTPHandler`, or the embedding HTTP client path. The bug
is byte-identical on current `main`: `litellm/llms/ollama/completion/handler.py` line 4
still carries the `[TODO]: migrate embeddings to a base handler as well.` comment, and line
82 still calls `await litellm.module_level_aclient.post(url=api_base, json=data)` with no
`ssl_verify` argument. Issue #30778 remains open, 0 comments, untouched since filed
2026-06-18. PRs #30810 and #30848 remain open and unmerged, both idle since 2026-06-20, now
over two months with no maintainer engagement. The HAProxy TLS sidecar workaround remains
required.

**2026-09-02 re-verify, still broken; new stable releases v1.99.0 (2026-09-01) and v1.99.1
(2026-09-02, Docker-only) ship no fix:**
LiteLLM advanced from v1.98.0 to **v1.99.0** (stable, published 2026-09-01T04:50:50Z, roughly
150 merged PRs) and then **v1.99.1** (Docker-image-only release, published 2026-09-02T06:17:51Z,
no PyPI package, carries a single OTel cache-token-count fix on top of v1.99.0). A Docker-only
maintenance release **v1.97.1** also landed the same day (2026-09-02, dependency and CVE
refresh on the 1.97 line, no functional change). Scanned the v1.99.0 and v1.99.1 changelogs:
none of the entries touch ollama, `ssl_verify`, `module_level_aclient`, or the embedding HTTP
client path (the only ssl-adjacent hit is an unrelated wolfi-base base-image digest bump for a
Docker CVE). The bug is byte-identical on current `main`: `litellm/llms/ollama/completion/handler.py`
still carries the `[TODO]: migrate embeddings to a base handler as well.` comment at line 4, and
still calls `await litellm.module_level_aclient.post(url=api_base, json=data)` with no
`ssl_verify` argument (now at line 98, shifted down from line 82 by an intervening typing and
basedpyright cleanup commit, 2026-08-21, with no functional change). Issue #30778 remains open,
0 comments, untouched since filed 2026-06-18. PRs #30810 and #30848 remain open and unmerged,
both idle since 2026-06-20, now over two months with no maintainer engagement. The HAProxy TLS
sidecar workaround remains required.

**Partial mitigation**: Setting `litellm.ssl_verify = False` **globally before the first embedding call** may work, because the singleton `HTTPHandler` picks up `litellm.ssl_verify` at creation time via `get_ssl_configuration()`. However, this is fragile — it depends on initialization order, and per-request `ssl_verify=false` (as used in our model config) is still silently dropped for the ollama embedding path.

**2026-09-10 re-verify, still broken; new stable release v1.100.0 (2026-09-06) ships no fix:**
LiteLLM advanced from v1.99.0/v1.99.1 to **v1.100.0** (stable, published 2026-09-06T05:33:17Z).
Prereleases v1.101.0-dev.1/dev.2, v1.101.0-rc.1/rc.2, and v1.102.0-dev.1 also exist but are not
stable (rc.2 published 2026-09-10, the same day as this check). The two Docker-only maintenance
releases already logged (v1.97.1, v1.99.1, both 2026-09-02) remain the only entries in that
window. Scanned the v1.100.0 changelog: no entry mentions ollama. The one ssl_verify-related fix
in the release, PR #38400 ("fix(aiohttp): honor global ssl_verify on the aiohttp_openai handler
path"), touches only `litellm/llms/custom_httpx/aiohttp_handler.py` for the unrelated
`aiohttp_openai/` model prefix's global `litellm_settings.ssl_verify`, a completely different code
path from the `ollama/` embedding bug tracked here; it does not touch
`litellm/llms/ollama/completion/handler.py` or `module_level_aclient`. The bug is byte-identical
on current `main`: `litellm/llms/ollama/completion/handler.py` line 4 still carries the
`[TODO]: migrate embeddings to a base handler as well.` comment, and line 98 still calls
`await litellm.module_level_aclient.post(url=api_base, json=data)` with no `ssl_verify` argument.
Issue #30778 remains open, 0 comments, untouched since filed 2026-06-18. PRs #30810 and #30848
remain open and unmerged, both idle since 2026-06-20, now nearly three months with no maintainer
engagement. The HAProxy TLS sidecar workaround remains required.

**2026-09-16 re-verify, still broken; new stable release v1.101.0 (2026-09-15) and Docker-only
point release v1.100.1 (2026-09-10) ship no fix:**
LiteLLM advanced from v1.100.0 to **v1.100.1** (Docker-image-only, published 2026-09-10T16:58:02Z,
three backport/revert PRs onto stable/1.100.x, none touching ollama/ssl_verify) and then
**v1.101.0** (stable, published 2026-09-15T03:41:59Z; prereleases v1.101.0-rc.2, v1.102.0-dev.1/
dev.2, v1.102.0-rc.1, and v1.103.0-dev.1 also exist but are not stable). Scanned the v1.101.0
changelog: the only ollama-adjacent entry is PR #39010 ("fix(ollama_chat): stamp finish_reason
tool_calls when tool calls streamed before the done chunk"), which touches chat-completion
streaming finish_reason handling, not the embedding path or ssl_verify. The bug is byte-identical
on current `main`: `litellm/llms/ollama/completion/handler.py` (blob
449952217b5e420c92ca20ff9cdff62aa95468e4, 4148 bytes) still carries the `[TODO]: migrate
embeddings to a base handler as well.` comment at line 4, and line 98 still calls `await
litellm.module_level_aclient.post(url=api_base, json=data)` with no `ssl_verify` argument. No
commit has touched the file since the 2026-08-21 basedpyright chore already logged. Issue #30778
remains open, 0 comments, untouched since filed 2026-06-18.

**Administrative event on PRs #30810/#30848, not maintainer engagement:** both PRs' base branch,
`litellm_internal_staging` (not `main`, these are staged against BerriAI's internal integration
branch), was deleted and recreated on 2026-09-13, which GitHub auto-closed both PRs at
04:25:10Z/04:25:18Z (`BaseRefDeletedEvent` + `ClosedEvent`) and they were then reopened at
06:00:59Z/06:01:43Z (`ReopenedEvent`). This is routine branch housekeeping on BerriAI's side, not
a review action: head commits are unchanged (247b21a5fa for #30810, e60e09c0d3 for #30848), no new
comments or reviews were posted, and both PRs' `mergeable_state` is now `dirty` (conflicts against
the recreated base) where it was previously unresolved/stale. Comment counts remain 4 and 2
respectively. Still no maintainer review or merge decision on either PR, now over three months
idle in substance. The HAProxy TLS sidecar workaround remains required.

**2026-09-22 re-verify, still broken; no new stable release:**
LiteLLM's latest stable release remains **v1.101.0** (2026-09-15). No newer stable shipped this
cycle (v1.102.0-rc.2, v1.103.0-dev.1/dev.2, and v1.103.0-rc.1 exist as prereleases only, latest
2026-09-20). The bug is byte-identical on current `main`: `litellm/llms/ollama/completion/handler.py`
(blob 449952217b5e420c92ca20ff9cdff62aa95468e4, 4148 bytes, unchanged since the 08-21 basedpyright
chore) still carries the `[TODO]: migrate embeddings to a base handler as well.` comment at line 4
and still calls `await litellm.module_level_aclient.post(url=api_base, json=data)` with no
`ssl_verify` argument at line 98. Issue #30778 remains open but received its first activity since
filing: a `github-actions[bot]` stale marker on 2026-09-17T00:02:18Z ("automatically marked as
stale, will be closed if no further activity occurs"). This is bot housekeeping, not maintainer
engagement, but it flags the issue is now at risk of auto-closure on the tracker's stale-bot
schedule if nobody comments. PRs #30810 and #30848 remain open, unmerged, and `dirty` (conflicts
against the recreated `litellm_internal_staging` base logged 2026-09-16); no new comments or
reviews, head commits unchanged (247b21a5fa / e60e09c0d3). The HAProxy TLS sidecar workaround
remains required.

**2026-09-25 re-verify, still broken; issue #30778 auto-closed by stale bot, two new stable
releases:**
LiteLLM advanced from v1.101.0 to **v1.102.0** and then **v1.102.1** (latest stable, published
2026-09-23T06:12:28Z), plus Docker-only backport releases v1.100.2/v1.100.3, v1.99.3/v1.99.4, and
v1.98.1 dated across 2026-09-23 to 09-25, and prerelease v1.104.0-dev.2. Scanned the v1.102.0
changelog: the only ollama-adjacent entry is PR #40311 ("move Anthropic, Vertex Anthropic, Ollama
and HF template fetches off the event loop"), which touches
`litellm/llms/ollama/completion/transformation.py` (confirmed via the PR's file list), not
`completion/handler.py` or `ssl_verify`. The v1.102.1 changelog contains only anthropic and
typesafe backport PRs, no ollama entries. v1.101.1 and v1.101.2 changelogs also have zero
ollama/ssl_verify hits. The bug is byte-identical on current `main`:
`litellm/llms/ollama/completion/handler.py` (blob 449952217b5e420c92ca20ff9cdff62aa95468e4, 4148
bytes, unchanged since the 08-21 basedpyright chore) still carries the `[TODO]: migrate embeddings
to a base handler as well.` comment at line 4 and still calls `await
litellm.module_level_aclient.post(url=api_base, json=data)` with no `ssl_verify` argument at
line 98.

**Issue #30778 auto-closed.** The 09-17 stale marker logged last cycle ran its course: GitHub
Actions closed the issue on 2026-09-24T00:03:04Z with no human ever commenting (timeline shows
only bot-authored labeled/commented/closed events, 1 total comment, the stale-bot notice itself).
The bug is still present on `main` as shown above, so this is a tracker-hygiene loss, not a
resolution, and the issue no longer represents upstream awareness of the bug; it should be
reopened or refiled if we want it tracked upstream.

PRs #30810 and #30848 remain open and `dirty` (unmergeable against `main`), head commits unchanged
(247b21a5fa / e60e09c0d3), no new comments; their `updatedAt` timestamps moved to 2026-09-23, which
is GitHub recomputing `mergeable_state` as `main` advances, not new PR activity. The HAProxy TLS
sidecar workaround remains required.

**2026-09-28 re-verify, still broken; new stable release v1.103.0 (2026-09-28) ships no fix:**
LiteLLM advanced from v1.102.1 to **v1.103.0** (stable, published 2026-09-28T05:43:48Z, the same
day as this check), plus Docker-only point releases v1.100.2/v1.100.3 (09-23/09-25), v1.99.3/
v1.99.4 (09-23/09-25), v1.98.1 (09-25), stable point releases v1.101.1/v1.101.2 (09-23/09-24), and
prereleases v1.103.0-rc.1, v1.104.0-dev.1/dev.2, v1.104.0-rc.1. Scanned the v1.103.0, v1.101.1 and
v1.101.2 changelogs: zero hits for ollama, ssl_verify, module_level_aclient or
BaseLLMAIOHTTPHandler. The bug is byte-identical on current `main`:
`litellm/llms/ollama/completion/handler.py` (blob 449952217b5e420c92ca20ff9cdff62aa95468e4, 4148
bytes, unchanged since the 08-21 basedpyright chore) still carries the `[TODO]: migrate embeddings
to a base handler as well.` comment at line 4 and still calls `await
litellm.module_level_aclient.post(url=api_base, json=data)` with no `ssl_verify` argument at
line 98. Issue #30778 remains closed (`state_reason: not_planned`, 1 comment, closed
2026-09-24T00:03:04Z), no new activity. PRs #30810 and #30848 remain open and unmerged, head
commits unchanged (247b21a5fa / e60e09c0d3), `updatedAt` unchanged at 2026-09-23T16:0x (still the
`mergeable_state` recompute from `main` advancing, not new PR activity). The HAProxy TLS sidecar
workaround remains required.

**2026-10-03 re-verify, still broken; no fix in v1.103.1/v1.103.2:** LiteLLM advanced from
v1.103.0 to **v1.103.2** (stable, published 2026-10-01T06:37Z; v1.103.1 on 09-30), plus point
releases v1.102.2 (09-30), v1.101.3 (09-30), v1.101.4 (10-01), v1.100.4 (09-30) and prereleases
v1.104.0-rc.1/rc.2 and v1.105.0-dev.1/dev.2 (to 10-02). Scanned the v1.103.1, v1.103.2, v1.101.3,
v1.101.4 and v1.104.0-rc.2 changelogs: zero hits for ollama, ssl_verify, module_level_aclient or
BaseLLMAIOHTTPHandler. `litellm/llms/ollama/completion/handler.py` on `main` is byte-identical
(blob 449952217b5e420c92ca20ff9cdff62aa95468e4, 4148 bytes): `[TODO]: migrate embeddings to a
base handler as well.` at line 4, `await litellm.module_level_aclient.post(url=api_base,
json=data)` without `ssl_verify` at line 98. Issue #30778 remains closed (`not_planned`, closed
2026-09-24T00:03:04Z, 1 comment). PRs #30810 and #30848 remain open and unmerged, head commits
unchanged (247b21a5fa / e60e09c0d3), `updatedAt` unchanged at 2026-09-23. The HAProxy TLS
sidecar workaround remains required.

**2026-10-11 re-verify, still broken; no fix in v1.103.3..v1.104.2:** LiteLLM advanced to
**v1.104.2** (stable, "Latest", published 2026-10-08T08:39Z), with v1.104.1 and v1.103.3/v1.103.4
(10-03/10-07), point releases v1.102.3/v1.102.4, v1.101.5/v1.101.6, v1.100.5 (10-07/10-08) and
prereleases v1.105.0-rc.2/rc.3 and v1.106.0-dev.1 to dev.3 (to 10-09). Scanned the v1.103.3,
v1.103.4, v1.104.1, v1.104.2, v1.102.4, v1.101.6, v1.105.0-rc.3 and v1.106.0-dev.3 changelogs:
zero hits for ollama, ssl_verify, module_level_aclient or BaseLLMAIOHTTPHandler.
`litellm/llms/ollama/completion/handler.py` on `main` has a new blob
(ad2f15cdd7c4d5c8602326cfdbce2e5809dfb6ab, 4154 bytes, previously
449952217b5e420c92ca20ff9cdff62aa95468e4) only because of the 2026-10-06 typing refactor
`d4a791d650` (#44798, `logging_obj: Any` -> `object`); the bug is unchanged: `[TODO]: migrate
embeddings to a base handler as well.` at line 4, `response: Final = await
litellm.module_level_aclient.post(url=api_base, json=data)` without `ssl_verify` at line 98.
Issue #30778 remains closed (`not_planned`, closed 2026-09-24T00:03:03Z, 1 comment). PRs #30810
and #30848 remain open and unmerged, head commits unchanged (247b21a5fa / e60e09c0d3),
`updatedAt` unchanged at 2026-09-23. The HAProxy TLS sidecar workaround remains required.

**2026-10-12 re-verify, still broken; no fix in v1.104.3/v1.105.0:** LiteLLM advanced to
**v1.105.0** (stable, published 2026-10-11T09:53Z) and **v1.104.3** (stable, 2026-10-11T10:37Z),
plus prerelease v1.106.0-rc.1 (2026-10-11). Scanned the v1.105.0, v1.104.3 and v1.106.0-rc.1
release notes: zero hits for ollama, ssl_verify, module_level_aclient or BaseLLMAIOHTTPHandler.
`litellm/llms/ollama/completion/handler.py` has blob 449952217b5e420c92ca20ff9cdff62aa95468e4
(4148 bytes) at v1.105.0, v1.104.3 and v1.104.2, so the 2026-10-06 typing refactor
(`d4a791d650`, blob ad2f15cdd7c4d5c8602326cfdbce2e5809dfb6ab, 4154 bytes) exists only on `main`,
not in any stable tag yet. On `main` (HEAD `4fd00513`, 2026-10-12) the bug is unchanged:
`[TODO]: migrate embeddings to a base handler as well.` at line 4, `response: Final = await
litellm.module_level_aclient.post(url=api_base, json=data)` without `ssl_verify` at line 98, and
no commit touched the file since 2026-10-07. Issue #30778 remains closed (`not_planned`, closed
2026-09-24T00:03:03Z, 1 comment). PRs #30810 and #30848 remain open and unmerged, head commits
unchanged (247b21a5fa / e60e09c0d3), `updatedAt` unchanged at 2026-09-23. The cluster runs
`ghcr.io/berriai/litellm:main-stable` (mutable tag; pod started 2026-10-11T16:19Z, image digest
sha256:60947db020b5). The HAProxy TLS sidecar workaround remains required.

**2026-11-10 re-verify, still broken; no fix in v1.105.x..v1.108.4, `main` now forwards credentials but not `ssl_verify`:**
LiteLLM stable advanced to **v1.108.4** (published 2026-11-09T23:00Z; v1.108.0 on 2026-11-04,
v1.108.1 to v1.108.3 on 11-05..11-07), with backport point releases up to v1.106.7, v1.105.7,
v1.104.8 and v1.103.10 (2026-10-27..11-04) and prereleases v1.109.0-rc.1 / v1.107.0-rc.3. Scanned
the release notes of every stable tag published since 2026-10-12: zero hits for
`module_level_aclient`, `BaseLLMAIOHTTPHandler` or an ollama `ssl_verify` change; the ollama entries
are chat-only (v1.106.0: #45053, #45079, #45177; v1.108.0: #39010) and #38400 touches only the
`aiohttp_openai/` handler. The handler differs between tags (v1.105.7 blob `449952217b5e` 4148 B,
v1.106.7 `ad2f15cdd7c4` 4154 B, v1.108.0 and v1.108.4 `816028499e81` 4249 B) but the bug is
unchanged in all of them: `[TODO]: migrate embeddings to a base handler as well.` at line 4, `await
litellm.module_level_aclient.post(url=_ollama_embed_url(api_base), json=data)` without `ssl_verify`
at line 102 in v1.108.x. On `main` the file is blob `e1fc9a8d262b` (5512 B, last commit `3f38730d2`,
2026-11-06, basedpyright typing cleanup). PR #48707 ("fix(ollama): forward embedding credentials,
headers and timeout", merged 2026-10-28, `3d7b94b34`) is the only functional change since the last
check: `main.py` now passes `api_key`, `headers` and `timeout` into `ollama_aembeddings`, but the
call still goes through the `litellm.module_level_aclient` singleton at line 125 with no
`ssl_verify`. #48707 is not yet in a stable tag. Issue #30778 remains closed (`not_planned`,
2026-09-24). PRs #30810 and #30848 remain open and unmerged, head commits unchanged (247b21a5fa /
e60e09c0d3), `updatedAt` 2026-10-13. The cluster pod `litellm-6d7686b9bc-2kp67`
(`ghcr.io/berriai/litellm:main-stable`, started 2026-11-04T14:44Z, digest sha256:c560d3171368) has
the same shape as v1.108.x: line 102 calls `module_level_aclient.post(...)` without `ssl_verify`.
The local TLS sidecar and the `https://...:11435` plus `ssl_verify: false` model entries were not
touched by the 2026-10-29/30 and 11-04/06 LiteLLM changes (virtual keys, extra hosts). The HAProxy
TLS sidecar workaround remains required.

## Upstream Fix

The embedding path in `litellm/llms/ollama/completion/handler.py` should stop using
`litellm.module_level_aclient` and instead use `get_async_httpx_client()` with the
per-call `ssl_verify` setting — the same pattern `llm_http_handler.py` uses for chat.
