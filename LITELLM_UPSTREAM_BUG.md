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

| File | Line | Issue |
|------|------|-------|
| `litellm/main.py` | ~5314 | `ollama_aembeddings()` called without `ssl_verify` |
| `litellm/llms/ollama/completion/handler.py` | embedding fn | Uses `litellm.module_level_aclient` singleton |
| `litellm/_lazy_imports.py` | `_lazy_import_http_handlers` | Creates `HTTPHandler(timeout=...)` without `ssl_verify` |
| `litellm/llms/custom_httpx/http_handler.py` | `HTTPHandler.__init__` | `ssl_verify=None` → `get_ssl_configuration(None)` → SSL enabled |

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
streaming text completions) remains open (last updated 2026-07-19). No fix has landed for any
of these paths — the HAProxy TLS sidecar workaround remains required.

**Partial mitigation**: Setting `litellm.ssl_verify = False` **globally before the first embedding call** may work, because the singleton `HTTPHandler` picks up `litellm.ssl_verify` at creation time via `get_ssl_configuration()`. However, this is fragile — it depends on initialization order, and per-request `ssl_verify=false` (as used in our model config) is still silently dropped for the ollama embedding path.

## Upstream Fix

The embedding path in `litellm/llms/ollama/completion/handler.py` should stop using
`litellm.module_level_aclient` and instead use `get_async_httpx_client()` with the
per-call `ssl_verify` setting — the same pattern `llm_http_handler.py` uses for chat.
