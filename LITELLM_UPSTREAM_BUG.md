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
# roles/k8s_dgx/defaults/main.yml
- model_name: "{{ ollama_embed_model }}"
  litellm_params:
    model: "ollama/{{ ollama_embed_model }}"
    api_base: "https://ollama-embed.ollama.svc.cluster.local:{{ ollama_tls_port }}"
    ssl_verify: false
```

See: `roles/k8s_dgx/tasks/ollama.yml` — ConfigMap `ollama-tls-haproxy-config`,
`tls-proxy` sidecar container in both `ollama` and `ollama-embed` Deployments.

## Upstream Status

**Still open** as of 2026-04-01. No fix merged.

- Related issue: [#6499](https://github.com/BerriAI/litellm/issues/6499) ("How to disable ssl verification for ollama?", closed). Maintainer acknowledged the embedding path was not migrated, but the issue was closed after only the chat path was fixed.
- The TODO comment `[TODO]: migrate embeddings to a base handler as well.` is still present at the top of `handler.py`.
- PR [#24704](https://github.com/BerriAI/litellm/pull/24704) (2026-03-28, open) fixes an adjacent embedding bug (model name prefix stripping) but does not address SSL/TLS.

**Partial mitigation**: Setting `litellm.ssl_verify = False` **globally before the first embedding call** may work, because the singleton `HTTPHandler` picks up `litellm.ssl_verify` at creation time via `get_ssl_configuration()`. However, this is fragile — it depends on initialization order, and per-request `ssl_verify=false` (as used in our model config) is still silently dropped for the ollama embedding path.

## Upstream Fix

The embedding path in `litellm/llms/ollama/completion/handler.py` should stop using
`litellm.module_level_aclient` and instead use `get_async_httpx_client()` with the
per-call `ssl_verify` setting — the same pattern `llm_http_handler.py` uses for chat.
