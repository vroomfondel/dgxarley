# SGLang Upstream Bug: AttributeError on abort in streaming /v1/chat/completions

## Status

**Fixed upstream** as of 2026-04-16 (PR [#22535](https://github.com/sgl-project/sglang/pull/22535),
commit [`f639425f`](https://github.com/sgl-project/sglang/commit/f639425ff06db7b5d379d749b6954eeb38d56972)).
Re-verified 2026-05-09 (and 2026-05-31: PR #22535 merged 2026-04-16, in v0.5.11/v0.5.12 — the current default image `0.5.12.post1-sm121` includes it; no in-repo workaround ever existed):

- **Still present** in legacy images `scitrera/dgx-spark-sglang:0.5.10` and
  `:0.5.10.post1` — both predate the fix commit.
- **Fixed** in **SGLang v0.5.11** (released 2026-05-05) and in our **dev1 image**
  `scitrera/dgx-spark-sglang:0.5.10-20260429-dev1` /
  `xomoxcc/dgx-spark-sglang:0.5.10-20260429-gemma4-sm121-dev1` (SGLang main commit
  `2bbd30a` from 2026-04-29) — both verified ancestors of the fix commit.

The fix aligns the streaming handler with the non-stream `_handle_abort_finish_reason`
path: only emit an error chunk when `status_code` is an actual `HTTPStatus` instance
(system errors like timeout/OOM/validation). Graceful aborts (`status_code=None`,
e.g. user-initiated `/abort_request`) now fall through to the normal chunk path with
`finish_reason="abort"`, same as `stop` or `length`. First observed in our cluster
2026-04-10 on `scitrera/dgx-spark-sglang:0.5.10` after calling `POST /abort_request`
with `{"abort_all": true}` while streaming chat completions were in-flight.

## Affected Configuration

- Endpoint: `/v1/chat/completions` with `stream: true`
- Trigger: scheduler-side abort of an in-flight streaming request (e.g. via
  `POST /abort_request` with `abort_all: true`, or any other path that sets
  `finish_reason.type == "abort"`)

Non-streaming `/v1/chat/completions` and `/v1/responses` are not affected by this
code path.

## The Bug

In `sglang/srt/entrypoints/openai/serving_chat.py` (lines 698-709 in 0.5.10), the
scheduler-abort branch crashes with:

```
File ".../sglang/srt/entrypoints/openai/serving_chat.py", line 706, in _generate_chat_stream
    code.name,
    ^^^^^^^^^
AttributeError: 'NoneType' object has no attribute 'name'
```

The offending block:

```python
if finish_reason_type == "abort":
    code = finish_reason.get(
        "status_code", HTTPStatus.INTERNAL_SERVER_ERROR
    )
    error = self.create_streaming_error_response(
        finish_reason.get("message", "Generation aborted."),
        code.name,
        code.value,
    )
```

`finish_reason` is a dict produced by the scheduler. When the abort path sets
`status_code` to `None` explicitly (rather than omitting the key), `dict.get(key, default)`
returns the stored `None` — the default is **only** used when the key is missing.
`code.name` then raises `AttributeError`.

Separately, even if `status_code` is present, the code assumes it is an `HTTPStatus`
enum instance. If the scheduler stores a plain `int` (e.g. `499`), `code.name`/`code.value`
would also fail — the code needs to coerce via `HTTPStatus(code)` first.

## Observed Trigger

Streaming chat completions were in-flight; a `POST /abort_request` with
`{"rid": "", "abort_all": true}` was issued. The streaming generator crashed inside
Starlette's `stream_response` → `_generate_chat_stream`, emitting a full traceback
into the pod log per aborted request. The 200 response to `/abort_request` succeeded;
the fallout is purely in the active streams being cleaned up.

Sample log excerpt (truncated):

```
File ".../starlette/responses.py", line 250, in stream_response
    async for chunk in self.body_iterator:
File ".../sglang/srt/entrypoints/openai/serving_chat.py", line 623, in prepend_first_chunk
    async for chunk in generator:
File ".../sglang/srt/entrypoints/openai/serving_chat.py", line 706, in _generate_chat_stream
    code.name,
AttributeError: 'NoneType' object has no attribute 'name'
[2026-04-10 15:04:40] INFO:     10.68.0.140:0 - "POST /abort_request HTTP/1.1" 200 OK
```

## Upstream Fix

PR [#22535](https://github.com/sgl-project/sglang/pull/22535) (`add check for none
status code in FinishAbort`), merged 2026-04-16 as commit
[`f639425f`](https://github.com/sgl-project/sglang/commit/f639425ff06db7b5d379d749b6954eeb38d56972).
The fix in `serving_chat.py` (and the parallel one in `serving_completions.py`):

```python
# After PR #22535 — at v0.5.11 line 809:
if finish_reason_type == "abort" and isinstance(
    finish_reason.get("status_code"), HTTPStatus
):
    code = finish_reason["status_code"]
    error = self.create_streaming_error_response(
        finish_reason.get("message", "Generation aborted."),
        code.name,
        code.value,
    )
    yield f"data: {error}\n\n"
    break
```

The `isinstance(..., HTTPStatus)` guard ensures `.name`/`.value` are only accessed
on a real enum. Graceful aborts (where `status_code` is `None`) take the normal
chunk path and emit `finish_reason="abort"` to the client.

## Our Workaround

None — and none needed once we deploy the dev1 image or v0.5.11. On legacy v0.5.10 /
v0.5.10.post1 the exception is raised inside the per-stream async generator, so it
only kills the individual stream being aborted — the server keeps running and
subsequent requests are unaffected. The log noise is cosmetic.

## Related

- `SGLANG_REASONING_TOKENS_UPSTREAM_BUG.md`
- `SGLANG_TP_EP_MOE_UPSTREAM_BUG.md`
- `SGLANG_SHARDED_SPECULATIVE_UPSTREAM_BUG.md`
