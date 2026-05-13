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

| #  | attn   | CG / pw | spec_num_steps | spec_draft_tokens | eagle_topk | Status      | n=1 tok/s | n=4 peak | n=8 peak |
|----|--------|---------|---------------:|------------------:|-----------:|-------------|----------:|---------:|---------:|
| 7  | triton | on / on | 2              | 3                 | 1          | ok          | **20.83** | **77.67**| —        |
| 8  | triton | on / on | 3              | 4                 | 1          | **pending** | —         | —        | —        |
| 9  | triton | on / on | 4              | 5                 | 1          | **pending** | —         | —        | —        |
| 10 | triton | on / on | 5              | 6                 | 1          | **pending** | —         | —        | —        |
| 11 | triton | on / on | 6              | 7                 | 1          | **pending** | —         | —        | —        |

**fi-attn (Cases 1–3) — 3× startup_crash, same FlashInfer dispatch-table miss as on 0.5.10.** The Gemma-4 head_dim=256 + RoPE=64 prefill kernel still hits `FlashInfer Internal Error: Invalid configuration : NUM_MMA_Q=1 NUM_MMA_D_QK=32 NUM_MMA_D_VO=32 NUM_MMA_KV=1 NUM_WARPS_Q=1 NUM_WARPS_KV=4` from `prefill.cuh:2978`. FlashInfer 0.6.8.post1 + sgl-kernel 0.4.2 did not fix the dispatch-table gap. Even Case 02 (eager, no CUDA graph) crashes — the assert fires at the first decode call, not during graph capture. **Workaround: triton-attn (the profile default), as on 0.5.10.**

All triton-attn cases finish with `stop` × N (Gemma is concise; ~1.2 k tokens vs the 3072 cap).

---

## Results

**Baseline matrix complete (2026-05-11, 6/6 cases run: 3 ok, 3 startup_crash). MTP sweep partial (2026-05-13, 1/5 cases ok; Tests 8–11 pending).**

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

## MTP sweep (Tests 7–11) — partial (1/5 cases done)

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

### Tests 08–11 — pending

The remaining four step counts (3, 4, 5, 6) have not been launched yet. Expectation given Test 07's profile:
- Acceptance length should scale roughly with `num_steps` but the **acceptance rate** typically tails off past ~3–4 steps as draft uncertainty compounds. Sweet spot is probably `num_steps ∈ {3, 4}` if Test 07's ~0.7 acceptance rate at 2 steps holds.
- Throughput gain at n=4/n=8 likely peaks then plateaus or regresses — once verify-batch decode is saturated, the drafter step's own latency stops being free.

### Quality-watch items (carry-forward for Tests 08–11)
- Drafter KV-share: 4-layer assistant shares the target's KV cache — `mem_fraction_static=0.60` headroom held fine in Test 07. If a higher-step case OOMs, drop to 0.55.
- Acceptance rate: <60 % at any step count = drafter not earning its keep on this workload.
- Output coherence: re-apply word-salad / triple-word grep + tail-eyeball — verify path must be lossless.
