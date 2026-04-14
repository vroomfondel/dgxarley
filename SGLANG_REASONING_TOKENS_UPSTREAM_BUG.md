# SGLang Upstream Bug: reasoning_tokens always 0 in /v1/chat/completions usage

## Status

**Fixed upstream** as of 2026-04-05. Merged by maintainer `hnyls2002`. Re-verified 2026-04-14: fix still in place in current `main`, included in our v0.5.10 image.

- Issue: [#15660](https://github.com/sgl-project/sglang/issues/15660) (closed 2025-12-29, but bug persisted until fix)
- Fix PR: [#15562](https://github.com/sgl-project/sglang/pull/15562) — **merged 2026-04-04**. Re-tokenizes reasoning content at the API layer to compute `reasoning_tokens` and populates it into `UsageInfo`
- Closed as duplicate of #15562: [#17938](https://github.com/sgl-project/sglang/pull/17938), [#17764](https://github.com/sgl-project/sglang/pull/17764), [#17156](https://github.com/sgl-project/sglang/pull/17156) — all closed 2026-04-04
- Closed without merge: [#15875](https://github.com/sgl-project/sglang/pull/15875) — "fix(openai): include reasoning_tokens in streaming usage"
- Not in v0.5.10rc0 (2026-03-28). **Included in v0.5.10 stable** (released 2026-04-06) and v0.5.10.post1 (2026-04-09, flashinfer bump only)
- Our image: `0.5.10` — **bug is fixed** in this image

## Affected Configuration

- Endpoint: `/v1/chat/completions` (both streaming and non-streaming)
- Models with reasoning/thinking tokens (e.g., Qwen3, DeepSeek-R1)
- `completion_tokens_details.reasoning_tokens` is always `0` in usage response

The `/v1/responses` (Responses API) endpoint is **not affected** — it has a separate
code path that correctly computes reasoning_tokens.

## The Bug

The `UsageInfo` schema has `reasoning_tokens: Optional[int] = 0` (OpenAI-compatible),
but the field is never populated for `/v1/chat/completions`:

1. **`protocol.py`**: `UsageInfo` declares the field, defaulting to `0`
2. **`usage_processor.py`**: `calculate_streaming_usage()` and `calculate_response_usage()`
   never compute or pass `reasoning_tokens` — the field stays at the default
3. **`serving_chat.py`**: no reference to `reasoning_tokens` whatsoever

The reasoning tokens are correctly parsed and streamed as separate `reasoning_content`
in the response body, but the token counter in the `Req` object has no
`reasoning_tokens` accumulator. The usage accounting simply never counts them separately
from `completion_tokens`.

## Root Cause

The `Req` object tracks `completion_tokens` as a single counter. When reasoning tokens
are generated, they are included in the `completion_tokens` total but never split out
into a separate `reasoning_tokens` counter. The schema field exists for OpenAI API
compatibility, but the backend accounting was never wired up.

## Fix (upstream)

PR #15562 adds reasoning token counting in the output processor during extend and decode
stages, then surfaces it through `UsageInfo`. The PR has:

- 2 collaborator approvals (2026-02-04, 2026-03-04)
- Merge conflicts resolved by author (2026-03-25)
- CI GPU tests failing (infrastructure issue, not code)
- Blocked only on a maintainer merging it

## Our Workaround

None needed. The fix is merged upstream and included in our current image (`0.5.10`).

## Impact

Usage tracking and billing calculations that depend on `reasoning_tokens` being reported
separately (e.g., reasoning tokens are often priced differently) will undercount or
misattribute costs. The total `completion_tokens` is correct (reasoning tokens are
included), but the breakdown is missing.

## Related

- `SGLANG_TP_EP_MOE_UPSTREAM_BUG.md` — moe_wna16 qzeros + EP bug (same version)
- `SGLANG_SHARDED_SPECULATIVE_UPSTREAM_BUG.md` — sharded_state + speculative decoding (same version)
- SGLang PR [#18022](https://github.com/sgl-project/sglang/pull/18022) — merged test asserting `reasoning_tokens > 0` for tool calls (now passes with #15562 merged)
