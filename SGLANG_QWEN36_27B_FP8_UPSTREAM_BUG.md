# SGLang Upstream Bug: Qwen3.6-27B-FP8 token salad via FP8 scale bypass

## Status (re-verified 2026-05-09)

- **`Qwen/Qwen3.6-27B-FP8` — BROKEN** on `scitrera/dgx-spark-sglang:0.5.10`. Model
  loads and decode runs, but every request produces multilingual token salad with
  immediate NGRAM_FLOOD from the bench's RepetitionGuard. Root cause: substring
  collision in `is_layer_skipped()` silently bypasses FP8 dequantization scaling for
  the fused gate-up projection, yielding garbage logits.

- **Fix shipped upstream:** PR #23467 (commit `4323fce`) merged to main 2026-04-22.
  Verified ancestor of:
  - **SGLang v0.5.11** (released 2026-05-05) — fix is in this release.
  - Our **dev1 image** `scitrera/dgx-spark-sglang:0.5.10-20260429-dev1` and
    `xomoxcc/dgx-spark-sglang:0.5.10-20260429-gemma4-sm121-dev1`, both pinned to
    SGLang main commit `2bbd30a` (2026-04-29) — fix is already in the image, the
    runtime monkey-patch in `sglang_launch.sh` therefore becomes a no-op (idempotent
    sentinel: `def _module_path_match` already present).
  - **NOT in** v0.5.10, v0.5.10.post1, or v0.5.10rc0 — those still need the runtime
    patch.

- **Runtime patch in** `roles/k8s_dgx/files/sglang_launch.sh` is left in place as
  defense-in-depth: it auto-skips on dev1 / v0.5.11 images (sentinel match) and
  remains active on the older v0.5.10 / v0.5.10.post1 images.

## Affected models

| Model                        | Type                                                 | Quantization               | Status                                                                                                                                                                                                                          |
|------------------------------|------------------------------------------------------|----------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `Qwen/Qwen3.6-27B-FP8`       | Dense (27B, hybrid Gated-DeltaNet + Gated-Attention) | Fine-grained FP8 block-128 | **BROKEN** — token salad, NGRAM_FLOOD on every request                                                                                                                                                                          |
| `Qwen/Qwen3.6-35B-A3B-FP8`   | MoE                                                  | Fine-grained FP8 block-128 | **UNTESTED for this bug** — same Qwen3.6 FP8 quant config family, but `modules_to_not_convert` differs for MoE; the substring collision with `mlp.gate` may or may not reproduce depending on exact entries; needs verification |
| `Qwen/Qwen3.5-122B-A10B-FP8` | MoE                                                  | FP8                        | **CONFIRMED UNAFFECTED** — clean run on our cluster; ships a different `modules_to_not_convert` layout that does not collide with `mlp.gate_up_proj`                                                                            |

## Symptoms

### Weight loading warnings (server-side log, ~256 occurrences)

```
[TP0] Parameter model.layers.<N>.mlp.gate_gate_up_proj.weight_scale_inv not found in params_dict
[TP0] Parameter model.layers.<N>.mlp.gate_up_proj.weight_scale_inv not found in params_dict
```

The double `gate_gate_` prefix in the first warning is a secondary side effect (see
Root Cause below), not the primary failure. The second warning is the actual culprit:
`weight_scale_inv` is never registered in `params_dict`, so the FP8 dequant kernel
runs with the default identity scale (`1.0`), producing completely wrong logits.

### Decode output — token salad and repetition flood

Thinking stream excerpt from a typical request (English prompt, simple arithmetic):

```
...ValuePaireladoDEX, medium medium medium medium medium medium medium medium medium
Ро Pra isobaric Роль pract Pract...
```

Words like `ValuePaireladoDEX` (mangled concatenations), `medium` repeated nine times,
and Cyrillic characters (`Ро`, `Роль`) interspersed with Latin fragments are the
characteristic signature of the broken FP8 scale path. The output is entirely
unrelated to the input prompt.

### Bench-side RepetitionGuard diagnostic

```
NGRAM_FLOOD: ngram 'medium' repeated 9 times in 40-token window → ABORT
```

The repetition guard trips within the first 40 decode tokens on effectively every
request. No useful output is ever produced.

## Root cause

The bug is in `python/sglang/srt/layers/quantization/utils.py`, function
`is_layer_skipped()`.

### Substring vs dot-boundary matching

`is_layer_skipped()` decides whether a given layer prefix should be skipped
(left unquantized) by checking if any entry from the quant config's
`modules_to_not_convert` list is a substring of the layer prefix:

```python
# v0.5.10 — naive substring match
is_skipped = any(ignored in shard_prefix for ignored in ignored_layers)
```

Qwen3.6's FP8 quant config (`quantization_config.json`) includes an entry
`"mlp.gate"` in `modules_to_not_convert` (a MoE-template name for a standalone
gate linear). For `Qwen3.6-27B` — a dense model — there is no standalone gate
layer; instead there is `mlp.gate_up_proj`, the fused gate-up projection.

`"mlp.gate" in "model.layers.0.mlp.gate_up_proj"` evaluates to `True` because
`gate` is a plain substring of `gate_up_proj`. SGLang therefore concludes
`mlp.gate_up_proj` is an unquantized layer, skips registering `weight_scale_inv`
in `params_dict`, and the FP8 kernel silently uses scale=1.0. With fine-grained
block-128 FP8, a scale of 1.0 on weights that were quantized with per-block scales
in the range 1e-3..1e1 produces outputs that are 3–4 orders of magnitude off —
leading directly to garbage logits and token salad.

### Secondary warning: `gate_gate_up_proj`

The `qwen3_5.py:load_weights` function (shared between Qwen3.5 and Qwen3.6 dense)
performs an internal name fixup: it replaces `gate_proj` with `gate_up_proj` for
the fused weight. When `is_layer_skipped()` already mutated the prefix from
`gate_up_proj` to `gate_proj` via the fallback shard expansion, and then
`load_weights` applies the reverse substitution, the result is a doubly-applied
prefix: `gate_gate_up_proj`. This explains the `gate_gate_up_proj.weight_scale_inv`
warning in the logs — it is a name-mutation side effect, not an independent bug.

### Why Qwen3.5 is not affected

`Qwen/Qwen3.5-122B-A10B-FP8` ships `modules_to_not_convert` entries that are either
full qualified names or entries that do not substring-match any fused projection
in the dense MLP. The collision between `mlp.gate` and `mlp.gate_up_proj` is unique
to the Qwen3.6 FP8 config.

## Fix

### Upstream commit

- **Commit:** [`4323fce82a091fab154bf36baa5820659ec0fd16`](https://github.com/sgl-project/sglang/commit/4323fce82a091fab154bf36baa5820659ec0fd16)
- **PR:** [#23467](https://github.com/sgl-project/sglang/pull/23467)
- **Author:** Mick (`mickjagger19@icloud.com`)
- **Date:** 2026-04-22
- **File changed:** `python/sglang/srt/layers/quantization/utils.py` (+31 / -4)

The fix replaces all four bare `ignored in shard_prefix`/`ignored in prefix` substring
checks with a new dot-boundary matching function:

```python
def _module_path_match(ignored: str, prefix: str) -> bool:
    # Match on dotted module-path boundaries so that `mlp.gate` does NOT
    # match `mlp.gate_up_proj`. Needed for quant configs (e.g. Qwen3.6-FP8)
    # whose `modules_to_not_convert` lists MoE-template names like `mlp.gate`
    # that collide with fused dense MLP names by plain substring.
    if ignored == prefix:
        return True
    if prefix.startswith(ignored + "."):
        return True
    return ("." + ignored + ".") in ("." + prefix + ".")
```

This ensures `"mlp.gate"` only matches `"...mlp.gate"` (exact suffix) or
`"...mlp.gate.<something>"` (dot-qualified child), never `"...mlp.gate_up_proj"`.

The commit also introduces `_FALLBACK_FUSED_SHARDS` — a built-in fused→shard name
mapping used when the quant config doesn't supply `packed_modules_mapping` — so
`gate_up_proj` is correctly expanded to `[gate_proj, up_proj]` even without an
explicit mapping in the config. PEP 585 generics (`dict[str, list[str]]`) are used
in the inserted code so no additional `typing` imports are needed.

### Our deployment: runtime monkey-patch

We apply the fix as a runtime monkey-patch in
`roles/k8s_dgx/files/sglang_launch.sh` rather than rebuilding the image. The patch
follows the same Python heredoc `code.replace(old, new, 1)` idiom used for other
quant patches in that file. Three separate replace calls correspond to the three
diff hunks. Idempotency sentinel: `def _module_path_match` — the patch skips itself
if that function definition is already present in `utils.py`.

**Status:** No longer needed on v0.5.11 / dev1-based images (PR #23467 is in the
image itself). The patch's sentinel check (`def _module_path_match` already in
`utils.py`) makes it a no-op there. Still active on legacy v0.5.10 / v0.5.10.post1
images.

## References

- GitHub issue [#23687](https://github.com/sgl-project/sglang/issues/23687) — `Qwen3.6-27B-FP8` token salad report
- Upstream PR [#23467](https://github.com/sgl-project/sglang/pull/23467) — `fix(fp8): dot-boundary module path matching in is_layer_skipped`
- Upstream commit [`4323fce`](https://github.com/sgl-project/sglang/commit/4323fce82a091fab154bf36baa5820659ec0fd16)
- HuggingFace Discussion [Qwen/Qwen3.6-27B#5](https://huggingface.co/Qwen/Qwen3.6-27B/discussions/5) — Qwen maintainer `yjdong` documents the workaround
