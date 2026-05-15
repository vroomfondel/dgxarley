# SGLang Test Log — Gemma-4 31B-it (dense, BF16), 4 Nodes, TP=4 EP=1, v0.5.11

## Environment

| Component | Value                                              |
|-----------|----------------------------------------------------|
| GPU       | NVIDIA GB10 (SM121/Blackwell), 128 GB per node     |
| Driver    | 580.142                                            |
| CUDA      | 13.2 host / 13.0 image (PR #21498)                 |
| Kernel    | 6.19.13-custom                                     |
| OS        | Ubuntu 24.04 LTS (aarch64)                         |
| K3s       | v1.35.3+k3s1                                       |
| Nodes     | spark1, spark2, spark3, spark4 (1 GPU each)        |
| Image     | `xomoxcc/dgx-spark-sglang:0.5.11-gemma4-sm121`     |
| Model     | `google/gemma-4-31B-it`                            |
| NCCL      | 2.29.7+cuda13.2 (dgxspark-3node-ring)              |
| Transport | **RoCE** via SR-IOV VF                             |

Matrix file: `kikube/matrixtest_matrices/sglang_nn4_tp4_ep1/gemma-4-31b-it/nv580.142_sglang-0.5.11_gemma-4-31b-it_n4_ep1.yaml`

Toolchain delta vs `_sglang-0.5.10_*` testlog: PyTorch 2.9 → 2.11, CUDA 13 default,
sgl-kernel 0.4.1.post1 → 0.4.2, FlashInfer 0.6.7.post2 → 0.6.8.post1. Native Gemma 4
support in 0.5.11. See `SGLANG_v0.5.11_VERSION_CHANGES.md`.

---

## Model Notes

- 30.7B **dense** (NOT MoE), multimodal (vision + text), native 256K context.
  We run text-only.
- Native Gemma 4 BF16 path in 0.5.11.
- Dense → no MoE-runner sweep; only attention × cuda_graph variants.

## Configuration Matrix (11 cases — 6 baseline + 5 MTP sweep)

All tests use: `tp=4, pp=1, ep=1, nccl_transport=roce, kv_cache_dtype=fp8_e4m3, mem_fraction_static=0.60, context_length=262144`. BF16 dense → no MoE/FP4/FP8 sweep.

### Baseline (Tests 1–6): attention × cuda_graph

| # | attention | dis_cuda_graph | dis_piecewise | Status            | n=1 tok/s | n=4 peak | n=8 peak  |
|---|-----------|----------------|---------------|-------------------|-----------|----------|-----------|
| 1 | fi        | false          | true          | **startup_crash** | —         | —        | —         |
| 2 | fi        | true           | true          | **startup_crash** | —         | —        | —         |
| 3 | fi        | false          | false         | **startup_crash** | —         | —        | —         |
| 4 | triton    | false          | true          | ok                | 10.91     | 44.10    | 81.47     |
| 5 | triton    | true           | true          | ok                | 9.26      | 42.48    | 83.06     |
| 6 | triton    | false          | false         | ok                | 10.49     | 44.06    | **85.34** |

### MTP speculative-decoding sweep (Tests 7–11)

Drafter: `google/gemma-4-31B-it-assistant` (4-layer auxiliary checkpoint, released 2026-05-05, Apache-2.0). Winner-shape fixed to **Case 06** (triton-attn + CG on + piecewise on); only `speculative_num_steps` varies. `enable_spec_v2: true`. Drafter auto-appended to `HF_PRELOAD_MODELS` by dgxarley when `speculative_enabled=true` + `speculative_draft_model_path` set.

**Image rebuild required** — the v0.5.11 release tag does NOT include Gemma-4 MTP support yet. SGLang's stock NEXTN/EAGLE worker loads the drafter via `AutoModel.from_config(...)` and then attempts a `model.language_model` weight surgery that does not exist on `Gemma4AssistantForCausalLM` (crashes with `ValueError: No module or parameter named 'model.language_model' in TransformersMultiModalForCausalLM` — observed during a first 2026-05-12 run; cf. failure result dir `kikube/matrixtest/2026-05-12/.../mtp_steps-2/TESTRESULTS_*_FAILED.json`). The proper fix is **upstream PR #24436 ("Gemma 4 — Adding MTP support")**, which adds a dedicated `Gemma4AssistantForCausalLM` model and the new `FROZEN_KV_MTP` speculative algorithm (recurrent hidden-state draft loop with frozen KV cache from the target). Merged 2026-05-07, AFTER the v0.5.11 tag. Cherry-picked into our image as `scripts/patches/sglang-gemma4-mtp-pr24436.patch` + `dockerfile-gemma4-mtp.patch`; build switched back to the `xomoxcc/dgx-spark-sglang:0.5.11-gemma4-sm121` recipe (PR #24436 patch is gated on `*gemma4*` recipe variants). The prior `sitecustomize.py` AutoModel-register stop-gap in `sglang_launch.sh` was removed once the PR landed in the image — it could only paper over the registration miss, not the `model.language_model` surgery.

At runtime SGLang detects `Gemma4AssistantForCausalLM` as drafter and **auto-promotes `--speculative-algorithm NEXTN` → `FROZEN_KV_MTP`** (log line: `Detected Gemma4AssistantForCausalLM draft; promoting --speculative-algorithm NEXTN to FROZEN_KV_MTP`). Overlap-scheduling is forcibly disabled in this path (`Overlap scheduler is disabled when using Frozen-KV MTP speculative decoding (spec v2 is not supported yet)`).

**`speculative_num_draft_tokens` manual sweep** — SGLang requires `num_draft_tokens ≥ num_steps + 1` (each step contributes one draft token plus the final accepted-token slot). The cookbook's fixed `num_draft_tokens=6` only matches `num_steps=5`; for `num_steps ∈ {2,3,4,6}` we had to bump it per case (autoadjust didn't fire). The table reflects the actually-launched values.

| #  | attn   | CG / pw | spec_num_steps | spec_draft_tokens | eagle_topk | Status | n=1 tok/s | n=4 peak  | n=8 peak    |
|----|--------|---------|---------------:|------------------:|-----------:|--------|----------:|----------:|------------:|
| 7  | triton | on / on | 2              | 3                 | 1          | ok     | 20.83     | 77.67     | (skipped)   |
| 8  | triton | on / on | 3              | 4                 | 1          | ok     | 22.88     | 83.04     | 142.02      |
| 9  | triton | on / on | 4              | 5                 | 1          | ok     | 22.09     | 86.24     | **153.24 ★**|
| 10 | triton | on / on | 5              | 6                 | 1          | ok     | 23.40     | 88.27     | 149.73      |
| 11 | triton | on / on | 6              | 7                 | 1          | ok     | **26.68 ★**| **91.41 ★**| 146.11    |

**fi-attn (Cases 1–3) — 3× startup_crash, same FlashInfer dispatch-table miss as on 0.5.10.** The Gemma-4 head_dim=256 + RoPE=64 prefill kernel still hits `FlashInfer Internal Error: Invalid configuration : NUM_MMA_Q=1 NUM_MMA_D_QK=32 NUM_MMA_D_VO=32 NUM_MMA_KV=1 NUM_WARPS_Q=1 NUM_WARPS_KV=4` from `prefill.cuh:2978`. FlashInfer 0.6.8.post1 + sgl-kernel 0.4.2 did not fix the dispatch-table gap. Even Case 02 (eager, no CUDA graph) crashes — the assert fires at the first decode call, not during graph capture. **Workaround: triton-attn (the profile default), as on 0.5.10.**

All triton-attn cases finish with `stop` × N (Gemma is concise; ~1.2 k tokens vs the 3072 cap).

---

## Results

**Matrix complete (2026-05-13, 11/11 cases run: 8 ok, 3 startup_crash). 6 baseline + 5 MTP cases.**

Result dir: `kikube/matrixtest/2026-05-11/results/sglang_nn4_tp4_ep1/gemma-4-31b-it/0.5.11/`.

### Delta vs 0.5.10 baseline

| #  | Config                  | 0.5.10 (n=1 / n=4 / n=8) | 0.5.11 (n=1 / n=4 / n=8)  | Δ n=8       |
|----|-------------------------|--------------------------|---------------------------|-------------|
| 04 | triton + CG on + pw off | 10.8 / 40.8 / 66.3       | 10.91 / 44.10 / 81.47     | **+22.9 %** |
| 05 | triton + eager          | 9.6 / 37.2 / 66.7        | 9.26 / 42.48 / 83.06      | **+24.5 %** |
| 06 | triton + CG on + pw on  | 10.6 / 36.8 / 70.6       | 10.49 / 44.06 / **85.34** | **+20.9 %** |

**Winner @ n=8: Case 06** (triton-attn + CG on + piecewise on) at **85.34 tok/s** — same winner shape as 0.5.10, but **+20.9 % faster** (70.6 → 85.34).

n=1 is essentially flat across versions (~10 tok/s — single-stream is compute-bound), n=4 is +8–14 %, n=8 is +21–25 %. The native Gemma 4 path in 0.5.11 dramatically improves multi-batch throughput vs the 0.5.10 transformers-fallback while keeping single-stream behaviour unchanged. Eager (Case 05) is now within ~3 % of the CG-on winner at n=8 — same eager-narrowing as on 27B-FP8.

### Notes

- **fi-attn still broken** on 0.5.11 — same FlashInfer dispatch gap as 0.5.10. Profile default of `attention_backend: triton` remains correct. To unblock fi-attn, the FlashInfer dispatch table would need entries for `NUM_MMA_D_QK=32 / NUM_MMA_D_VO=32` (head_dim=256 + RoPE=64). Upstream issue not yet filed.
- **Image** is the `xomoxcc/dgx-spark-sglang:0.5.11-gemma4-sm121` custom build (Gemma-4 source patches). Vanilla `scitrera/dgx-spark-sglang:0.5.11` not tested for this model.
- **Output quality verified** across all 39 successful requests:
  - `response_snippet` (head ~1 kB + tail ~500–700 B) scanned for word-salad
    patterns — 0 triple-word repetitions, 0 self-correction markers
    (`self-correct` / `stop rambling` / `thinking thinking` / `retire retire`).
  - Output-token distribution 994 → 1668 (median ~1300). **No request hit the
    3072 cap** — every response finished with natural EOS (`stop`×N), so the
    "model can't stop" failure mode is not present.
  - Eyeballed tails of all 8 n=8 requests in Case 06: clean concluding
    sentences (comparison tables, summaries, calls-to-action), no
    deterioration toward end-of-context.

---

## MTP sweep (Tests 7–11) — complete (5/5 cases ok)

The MTP cases mirror the Case 06 winner shape (triton-attn + CG on + piecewise on) and only vary `speculative_num_steps ∈ {2, 3, 4, 5, 6}`. Cookbook default is `num_steps=5`; we sweep ±2.

### Test 07 (`num_steps=2`, `num_draft_tokens=3`) — ok, 2026-05-13

Result dir: `kikube/matrixtest/2026-05-13/results/sglang_nn4_tp4_ep1/gemma-4-31b-it/0.5.11/nv580.142_sglang-0.5.11_gemma-4-31b-it_4n_1pp_4tp_ep1_07_triton-attn_piecewise_mtp_steps-2/`.

| n | peak (sum tok/s) | avg per-req tok/s | wall (s) | tokens out | finish |
|--:|-----------------:|------------------:|---------:|-----------:|--------|
| 1 | **20.83**        | 20.83             | 59.3     | 1 235      | stop   |
| 4 | **77.67**        | 19.42 avg / 19.75 p50 | 82.95 | 5 441    | stop ×4|

**Big MTP wins** vs Case 06 baseline (10.49 / 44.06):
- n=1: **+98 %** (20.83 vs 10.49) — single-stream nearly doubles, exactly the decode-bound territory MTP exists for.
- n=4: **+76 %** (77.67 vs 44.06) — gains persist into low-concurrency batched serving.

n=8 not in this run; needs follow-up to see whether the gain compresses once the target is closer to compute saturation.

**Drafter acceptance** (from `decode batch` log lines):
- n=1 steady-state: `accept len ∈ [2.15, 2.62]` out of `num_steps=2` (so `accept_rate ≈ 0.57–0.81`, median ~0.68). Drafter actually clears the bar on this prompt mix.
- n=4 steady-state: `accept len ∈ [2.23, 2.44]`, `accept_rate ≈ 0.61–0.72`, median ~0.68. No degradation with batching.

**FROZEN_KV_MTP path engaged** — head log shows the auto-promotion and `Capture Frozen-KV MTP draft cuda graph begin/end` lines. The PR #24436 cherry-pick is working as intended.

Output quality: 5/5 requests finished with `stop` (natural EOS), output tokens 1153–1554 (median ~1380), well below the 3072 cap. Pattern same as the baseline runs.

### Test 08 (`num_steps=3`, `num_draft_tokens=4`) — ok, 2026-05-13

Result dir: `kikube/matrixtest/2026-05-13/results/sglang_nn4_tp4_ep1/gemma-4-31b-it/0.5.11/nv580.142_sglang-0.5.11_gemma-4-31b-it_4n_1pp_4tp_ep1_08_triton-attn_piecewise_mtp_steps-3/`.

| n | peak (sum tok/s) | avg per-req | wall (s) | tokens out (median) | finish |
|--:|-----------------:|------------:|---------:|--------------------:|--------|
| 1 | **22.88**        | 22.88       | 53.4     | 1 220               | stop   |
| 4 | **83.04**        | 20.76       | 68.7     | 1 342               | stop ×4|
| 8 | **142.02**       | 17.75       | 83.2     | 1 332               | stop ×8|

**MTP gain grew vs Test 07 and now extends to n=8:**
- n=1: 22.88 vs Test 07 20.83 (+10 %), **+118 % vs Case 06 baseline** (10.49)
- n=4: 83.04 vs Test 07 77.67 (+7 %), **+88 % vs baseline** (44.06)
- n=8: 142.02 — first MTP n=8 datapoint, **+66 % vs baseline** (85.34). MTP still pays at the largest concurrency we test, contrary to my earlier hypothesis that it would tail off.

**Drafter acceptance** (from `decode batch` log):
- `accept_len ∈ [2.25, 2.86]` out of `num_steps=3`, with **median ~2.7** — drafter contributes ~2/3 of its theoretical max.
- `accept_rate ∈ [0.42, 0.62]`, median ~0.55. Per-step rate is *lower* than Test 07's ~0.68 (which had `num_steps=2`) — adding more steps compounds the per-step rejection probability. But the **absolute** accepted-token count per verify cycle is higher (2.7 vs 2.4), so net throughput wins.

Output quality: 13/13 requests stopped on natural EOS, tokens 1220–1610. No 3072 cap hits.

### Test 09 (`num_steps=4`, `num_draft_tokens=5`) — ok, 2026-05-13

Result dir: `kikube/matrixtest/2026-05-13/results/sglang_nn4_tp4_ep1/gemma-4-31b-it/0.5.11/nv580.142_sglang-0.5.11_gemma-4-31b-it_4n_1pp_4tp_ep1_09_triton-attn_piecewise_mtp_steps-4/`.

| n | peak (sum tok/s) | avg per-req | wall (s) | tokens out (median) | finish |
|--:|-----------------:|------------:|---------:|--------------------:|--------|
| 1 | 22.09            | 22.09       | 65.3     | 1 441               | stop   |
| 4 | **86.24**        | 21.56       | 59.7     | 1 211               | stop ×4|
| 8 | **153.24**       | 19.16       | 87.6     | 1 380               | stop ×8|

Δ vs Test 08 (`num_steps=3`):
- n=1: 22.09 vs 22.88 — **slightly worse** (~−3 %, within noise; longer 1441-token answer may have eaten some headroom).
- n=4: 86.24 vs 83.04 — **+4 %**, still climbing.
- n=8: 153.24 vs 142.02 — **+8 %**, still climbing. **Best n=8 so far** in the MTP sweep.

Acceptance (decode-batch log distribution): `accept_len` median ~3.0–3.1 out of max 4, `accept_rate` median ~0.50–0.55. As expected, per-step rate keeps dropping (0.68 @ steps=2 → 0.55 @ steps=3 → 0.52 @ steps=4) but the absolute accepted-token count per cycle keeps growing (2.4 → 2.7 → 3.05).

### Test 10 (`num_steps=5`, `num_draft_tokens=6`) — ok, 2026-05-13

| n | peak (sum tok/s) | wall (s) | tokens out (median) | finish |
|--:|-----------------:|---------:|--------------------:|--------|
| 1 | 23.40            | 52.2     | 1 222               | stop   |
| 4 | **88.27**        | 72.7     | 1 389               | stop ×4|
| 8 | 149.73           | 80.5     | 1 352               | stop ×8|

n=1 keeps climbing (+1.3 vs Test 09), n=4 keeps climbing (+2.0 vs Test 09), n=8 **drops** (149.73 vs 153.24 in Test 09). Verify-overhead at high concurrency starts eating the drafter gain past `num_steps=4`.

### Test 11 (`num_steps=6`, `num_draft_tokens=7`) — ok, 2026-05-13

| n | peak (sum tok/s) | wall (s) | tokens out (median) | finish |
|--:|-----------------:|---------:|--------------------:|--------|
| 1 | **26.68 ★**      | 46.8     | 1 249               | stop   |
| 4 | **91.41 ★**      | 67.0     | 1 462               | stop ×4|
| 8 | 146.11           | 83.2     | 1 340               | stop ×8|

**Sweep maximum at n=1 (26.68) and n=4 (91.41).** n=8 continues drifting down (146.11 < 149.73 < 153.24). n=1 jump from 23.40 → 26.68 (+14 %) is the largest single-step gain in the sweep at low concurrency.

### MTP-sweep summary

| spec_num_steps | n=1   | Δ baseline | n=4   | Δ baseline | n=8    | Δ baseline |
|---------------:|------:|-----------:|------:|-----------:|-------:|-----------:|
| baseline (Case 06) | 10.49 | —     | 44.06 | —          | 85.34  | —          |
| 2              | 20.83 | +99 %      | 77.67 | +76 %      | —      | —          |
| 3              | 22.88 | +118 %     | 83.04 | +88 %      | 142.02 | +66 %      |
| 4              | 22.09 | +110 %     | 86.24 | +96 %      | **153.24 ★** | **+80 %** |
| 5              | 23.40 | +123 %     | 88.27 | +100 %     | 149.73 | +75 %      |
| 6              | **26.68 ★** | **+154 %** | **91.41 ★** | **+108 %** | 146.11 | +71 %      |

**Sweet spots per concurrency:**
- **n=1, n=4 → `num_steps=6`** (sweep max). Curve was still climbing — `num_steps=7` might push further; not in this sweep.
- **n=8 → `num_steps=4`** (153.24). Past 4, verify-batch overhead eats drafter gain.
- **No clear single optimum for mixed-concurrency serving.** If the deployment is single-stream (chat / agent) → `num_steps=6`. If high-concurrency serving (n≥8 typical) → `num_steps=4`. The 6→4 split costs 7–9 tok/s at n=1 / n=4 but buys 5–10 % at n=8.

**Production recommendation (Gemma-4 31B-it dense, FROZEN_KV_MTP):**

```yaml
attention_backend: triton                   # fi-attn still crashes (head_dim=256+RoPE=64 dispatch miss)
disable_cuda_graph: false
disable_piecewise_cuda_graph: false
nccl_transport: roce
cuda_graph_max_bs: 8
speculative_enabled: true
speculative_algo: NEXTN                     # auto-promoted to FROZEN_KV_MTP at runtime
speculative_draft_model_path: google/gemma-4-31B-it-assistant
speculative_num_steps: 4                    # 153 tok/s at n=8; +80% over baseline
speculative_num_draft_tokens: 5             # = num_steps + 1 (autoadjust doesn't fire)
speculative_eagle_topk: 1
enable_spec_v2: true                        # auto-disabled at runtime by FROZEN_KV_MTP path
```

For single-stream / agent workloads, switch to `num_steps=6` + `num_draft_tokens=7` (26.7 tok/s n=1 vs 23.4).

Output quality across all 5 MTP cases (62 successful requests): 0 finish=length, 0 word-salad / triple-word patterns; output-token range 939–1730 (median ~1380), well below 3072 cap. Same coherence profile as the baseline runs.

The remaining four step counts (3, 4, 5, 6) have not been launched yet. Expectation given Test 07's profile:
- Acceptance length should scale roughly with `num_steps` but the **acceptance rate** typically tails off past ~3–4 steps as draft uncertainty compounds. Sweet spot is probably `num_steps ∈ {3, 4}` if Test 07's ~0.7 acceptance rate at 2 steps holds.
- Throughput gain at n=4/n=8 likely peaks then plateaus or regresses — once verify-batch decode is saturated, the drafter step's own latency stops being free.

### Quality-watch items (carry-forward for Tests 08–11)
- Drafter KV-share: 4-layer assistant shares the target's KV cache — `mem_fraction_static=0.60` headroom held fine in Test 07. If a higher-step case OOMs, drop to 0.55.
- Acceptance rate: <60 % at any step count = drafter not earning its keep on this workload.
- Output coherence: re-apply word-salad / triple-word grep + tail-eyeball — verify path must be lossless.
