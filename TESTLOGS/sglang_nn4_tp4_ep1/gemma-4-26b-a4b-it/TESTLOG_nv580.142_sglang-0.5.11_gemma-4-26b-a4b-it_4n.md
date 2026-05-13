# SGLang Test Log — Gemma-4 26B-A4B-it (MoE, BF16), 4 Nodes, TP=4 EP=1, v0.5.11

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
| Image     | `xomoxcc/dgx-spark-sglang:0.5.11-sm121`            |
| Model     | `google/gemma-4-26B-A4B-it`                        |
| NCCL      | 2.29.7+cuda13.2 (dgxspark-3node-ring)              |
| Transport | **RoCE** via SR-IOV VF                             |

Matrix file: `kikube/matrixtest_matrices/sglang_nn4_tp4_ep1/gemma-4-26b-a4b-it/nv580.142_sglang-0.5.11_gemma-4-26b-a4b-it_n4_ep1.yaml`

Toolchain delta vs `_sglang-0.5.10_*` testlog: PyTorch 2.9 → 2.11, CUDA 13 default,
sgl-kernel 0.4.1.post1 → 0.4.2, FlashInfer 0.6.7.post2 → 0.6.8.post1.
**Gemma 4 is now native in SGLang 0.5.11** (PR #21952 + follow-ups #22079, #24048,
#22842 — see [cookbook](https://docs.sglang.io/cookbook/autoregressive/Google/Gemma4)).
The `0.5.10-20260429-gemma4-sm121-dev1`-Patch image becomes redundant once we move
to a 0.5.11-based custom image. New `flashinfer_cutedsl` MoE backend (PR #21339)
added in Tests 13–18. See `SGLANG_v0.5.11_VERSION_CHANGES.md`.

---

## Model Notes

- 26B / 3.8B-active **MoE**, multimodal (vision + text), native 256K context.
  We run text-only.
- Native Gemma 4 path in 0.5.11 — the `transformers`-fallback monkey-patch from
  `sglang_launch.sh` is redundant for BF16 on this image.

## What changes vs the 0.5.10 sweep

1. **Native Gemma 4 path** (PR #21952 + follow-ups #22079, #24048, #22842) — replaces
   the prior `xomoxcc/dgx-spark-sglang:main-gemma4-sm121` custom-patched image, codepath
   is materially different.
2. **Image is now `xomoxcc/dgx-spark-sglang:0.5.11-sm121`** (no `-gemma4` suffix). BF16
   Gemma-4 needs no source patches under 0.5.11 — the two locally vendored Gemma-4
   patches in the `0.5.11-gemma4-sm121` recipe only matter for the NVFP4 path (PRs
   #22929 / #22928).
3. **fi-attn expected to WORK on 0.5.11.** Flashinfer 0.6.11 (≥ 0.6.10) ships PR #2959
   which adds the `head_dim=512` dispatch entry the Gemma-4 prefill kernel needed. On
   0.5.10 all six fi-attn cases (Tests 1–3, 7–9) crashed startup/bench. Re-running them
   on 0.5.11 is now a real perf comparison, not a regression check.
4. **`flashinfer_cutedsl` MoE runner** (PR #21339) — 4th MoE backend option, swept in
   Tests 13–18.
5. **PCG + fused RMSNorm + Residual-Add + Scalar** for Gemma-4 VLM (PR #24048).
6. **`gemma_weight` precomputed** to skip redundant per-forward add (PR #22673).
7. **Spec V2 + Overlap-Scheduling default** (PR #21062) — Gemma-4 is dense-attn VLM
   (not hybrid-mamba), so should be unaffected by the Word-Salad concurrency-race
   observed on Qwen3.6-35B-A3B-FP8. Verify output quality at n=4 / n=8 anyway.

## Configuration Matrix (23 cases — 18 baseline + 5 MTP sweep)

All tests use: `tp=4, pp=1, ep=1, nccl_transport=roce, kv_cache_dtype=fp8_e4m3, mem_fraction_static=0.85, context_length=262144`. BF16 weights → no FP4/FP8 GEMM sweep. `cutlass` MoE skipped (FP4-only).

### Baseline (Tests 1–18): moe_runner × attention × cuda_graph

| #  | moe_runner | attention | dis_cuda_graph | dis_piecewise | Status              | n=1 tok/s | n=4 peak | n=8 peak    |
|----|------------|-----------|----------------|---------------|---------------------|----------:|---------:|------------:|
| 1  | triton     | fi        | false          | true          | **crash A**         | —         | —        | —           |
| 2  | triton     | fi        | true           | true          | **crash A**         | —         | —        | —           |
| 3  | triton     | fi        | false          | false         | **crash A**         | —         | —        | —           |
| 4  | triton     | triton    | false          | true          | ok                  | 40.57     | 127.30   | 206.76      |
| 5  | triton     | triton    | true           | true          | ok                  | 23.42     | 122.16   | 203.17      |
| 6  | triton     | triton    | false          | false         | ok                  | 40.48     | 132.00   | 208.50      |
| 7  | fi_cutlass | fi        | false          | true          | **crash A**         | —         | —        | —           |
| 8  | fi_cutlass | fi        | true           | true          | **crash A (bench)** | —         | —        | —           |
| 9  | fi_cutlass | fi        | false          | false         | **crash A**         | —         | —        | —           |
| 10 | fi_cutlass | triton    | false          | true          | ok                  | 32.92     | 130.68   | 206.65      |
| 11 | fi_cutlass | triton    | true           | true          | ok                  | 21.36     | 121.36   | 206.35      |
| 12 | fi_cutlass | triton    | false          | false         | ok                  | 34.85     | 131.44   | **213.72 ★**|
| 13 | fi_cutedsl | fi        | false          | true          | **crash B**         | —         | —        | —           |
| 14 | fi_cutedsl | fi        | true           | true          | **crash B**         | —         | —        | —           |
| 15 | fi_cutedsl | fi        | false          | false         | **crash B**         | —         | —        | —           |
| 16 | fi_cutedsl | triton    | false          | true          | **crash B**         | —         | —        | —           |
| 17 | fi_cutedsl | triton    | true           | true          | **crash B**         | —         | —        | —           |
| 18 | fi_cutedsl | triton    | false          | false         | **crash B**         | —         | —        | —           |

**Crash A** (`attention_backend=flashinfer`, 6/6 cases): `RuntimeError: FlashInfer Internal Error: Invalid configuration : NUM_MMA_Q=1 NUM_MMA_D_QK=32 NUM_MMA_D_VO=32 NUM_MMA_KV=1 NUM_WARPS_Q=1 NUM_WARPS_KV=4` from `flashinfer/attention/prefill.cuh:2978`. **Same dispatch-table miss as on 0.5.10 — PR #2959 did NOT fix this.** The matrix preamble's hopeful note was wrong: PR #2959 fixes head_dim=512, but the crash signature shows `NUM_MMA_D_QK/VO=32` which corresponds to **head_dim=256 + RoPE_dim=64** (the Gemma-4 hybrid sliding/global attention config). That dispatch entry is still missing in flashinfer 0.6.11. Case 08 (eager) hits the same error at the first prefill call rather than at CUDA-graph capture, hence `bench_crash` instead of `startup_crash`, but the same root cause.

**Crash B** (`moe_runner_backend=flashinfer_cutedsl`, 6/6 cases): `AssertionError: Invalid quantization 'None'. FlashInfer CuteDSL MOE currently supports only: 'modelopt_fp4'.` Pre-check assertion at `server_args.py:2975`. Same fail-fast behaviour as on Qwen3.6-35B-A3B-FP8 — the new `fi_cutedsl` MoE backend (PR #21339) is FP4-only by design; BF16 weights are rejected before model load.

### MTP speculative-decoding sweep (Tests 19–23)

Drafter: `google/gemma-4-26B-A4B-it-assistant` (4-layer auxiliary checkpoint, released 2026-05-05, Apache-2.0). Winner-shape fixed to **Case 06** (triton-MoE + triton-attn + CG on + piecewise on — chosen as MTP base over Case 12's `fi_cutlass`-MoE despite the latter's 2.5 % edge, because cookbook-validated MTP path is `triton`-MoE). Only `speculative_num_steps` varies. `enable_spec_v2: true`.

CAVEAT (from matrix preamble): SGLang cookbook recommends `--tp 2` for the 26B-A4B + MTP combo on H200 (141 GB). We run 4-node TP=4 (Spark) — different topology, expected to fit (mem_fraction=0.85 leaves >2 GB/GPU headroom for the drafter shard). Reduce `mem_fraction_static` in a follow-up if OOMs.

**Image rebuild required** — v0.5.11 release does NOT include Gemma-4 MTP support; the stock NEXTN/EAGLE worker crashes with `ValueError: No module or parameter named 'model.language_model'` when loading `Gemma4AssistantForCausalLM` (observed on the 31B-it sibling on 2026-05-12). Cherry-picked **upstream PR #24436** ("Gemma 4 — Adding MTP support") into `xomoxcc/dgx-spark-sglang:0.5.11-gemma4-sm121` (`scripts/patches/sglang-gemma4-mtp-pr24436.patch` + `dockerfile-gemma4-mtp.patch`). PR adds `Gemma4AssistantForCausalLM` model + `FROZEN_KV_MTP` speculative algorithm (recurrent hidden-state draft loop, frozen target KV). Runtime auto-promotes `--speculative-algorithm NEXTN → FROZEN_KV_MTP` once the drafter is detected; overlap-scheduling is forcibly disabled on this path (spec v2 not yet supported).

**`speculative_num_draft_tokens` manual sweep** — SGLang requires `num_draft_tokens ≥ num_steps + 1`. The cookbook's fixed `6` only matches `num_steps=5`; for `num_steps ∈ {2,3,4,6}` we bumped per case (autoadjust didn't fire). Table reflects the actually-configured values.

| #  | moe    | attn   | CG / pw | spec_num_steps | spec_draft_tokens | eagle_topk | Status      | n=1 tok/s | n=4 peak | n=8 peak |
|----|--------|--------|---------|---------------:|------------------:|-----------:|-------------|----------:|---------:|---------:|
| 19 | triton | triton | on / on | 2              | 3                 | 1          | **pending** | —         | —        | —        |
| 20 | triton | triton | on / on | 3              | 4                 | 1          | **pending** | —         | —        | —        |
| 21 | triton | triton | on / on | 4              | 5                 | 1          | **pending** | —         | —        | —        |
| 22 | triton | triton | on / on | 5              | 6                 | 1          | **pending** | —         | —        | —        |
| 23 | triton | triton | on / on | 6              | 7                 | 1          | **pending** | —         | —        | —        |

---

### Column Legend

| Column         | Description                                                                                                                     |
|----------------|---------------------------------------------------------------------------------------------------------------------------------|
| moe_runner     | `moe_runner_backend` — `triton`, `flashinfer_cutlass` (`fi_cutlass`), or **new** `flashinfer_cutedsl` (`fi_cutedsl`, PR #21339) |
| attention      | `attention_backend` — `fi` = FlashInfer (now expected to work on Gemma-4 via flashinfer 0.6.11 + PR #2959), `triton` = Triton   |
| dis_cuda_graph | `disable_cuda_graph` — true = eager, false = capture CUDA graphs                                                                |
| dis_piecewise  | `disable_piecewise_cuda_graph` — true = only fixed-BS graphs, false = piecewise variable-length                                 |

---

## Results

**Baseline complete (started 2026-05-11 22:36 CEST, finished early morning 2026-05-12 — 18/18 baseline cases run: 6 ok, 12 crash). MTP sweep (Tests 19–23) pending.**

Result dir: `kikube/matrixtest/2026-05-11/results/sglang_nn4_tp4_ep1/gemma-4-26b-a4b-it/0.5.11/`.

### Δ vs 0.5.10 baseline

| #  | Config                                     | 0.5.10 (n=1 / n=4 / n=8) | 0.5.11 (n=1 / n=4 / n=8)    | Δ n=8       |
|----|--------------------------------------------|--------------------------|-----------------------------|-------------|
| 04 | triton MoE + triton attn + CG on + pw off  | 40.1 / 114.8 / 163.9     | 40.57 / 127.30 / 206.76     | **+26.1 %** |
| 05 | triton MoE + triton attn + **eager**       | 20.5 / 104.9 / 159.8     | 23.42 / 122.16 / 203.17     | **+27.1 %** |
| 06 | triton MoE + triton attn + CG on + **pw on** | 39.8 / 114.6 / 180.5   | 40.48 / 132.00 / 208.50     | **+15.5 %** |
| 10 | fi_cutlass MoE + triton attn + CG on + pw off | 39.7 / 109.4 / 174.1  | 32.92 / 130.68 / 206.65     | **+18.7 %** |
| 11 | fi_cutlass MoE + triton attn + **eager**   | 25.6 / 112.2 / 158.4     | 21.36 / 121.36 / 206.35     | **+30.3 %** |
| 12 | fi_cutlass MoE + triton attn + CG on + **pw on** | 40.2 / 110.8 / 172.4 | 34.85 / 131.44 / **213.72** | **+23.9 %** |

### Findings

1. **Winner @ n=8: Case 12** (fi_cutlass MoE + triton attn + CG on + piecewise on) at **213.72 tok/s** — **+23.9 % vs the 0.5.10 best** (Case 06 = 180.5 tok/s). The 0.5.11 winner shape is also different: 0.5.10 winner was triton-MoE/piecewise-on, but 0.5.11's fi_cutlass MoE now matches and slightly edges past triton-MoE thanks to the native Gemma-4 path.
2. **Native Gemma-4 path delivers a clean 15–30 % n=8 speedup** across all six working configs. n=4 also improves by +8–18 %; n=1 is mostly flat or slightly worse (single-stream is compute-bound, latency wins at the kernel level matter less per-request).
3. **triton-MoE vs fi_cutlass MoE: tied at n=8.** triton-MoE (Cases 4–6) lands at 203–209; fi_cutlass MoE (Cases 10–12) at 206–214. Within 3 % across the matched configs; fi_cutlass wins by 5 tok/s at the winner shape (piecewise on). Both MoE backends benefit equally from the native Gemma-4 path.
4. **Piecewise CG slightly better at n=8 on this model** (Case 06: 208.50 vs Case 04: 206.76; Case 12: 213.72 vs Case 10: 206.65). +1–3 % delta — within run-to-run noise on n=8 but consistent in direction.
5. **Eager penalty narrowed** vs 0.5.10: was ~10 % gap at n=8 on 0.5.10 (159.8 vs 180.5), now ~3 % (203.17 vs 208.50). Same pattern as 27B-FP8 and 31B-it under 0.5.11 (Spec V2 + Overlap defaults narrow the eager penalty across the board).
6. **fi-attn still broken** on Gemma-4 — same prefill.cuh dispatch table miss as on 0.5.10. PR #2959 in flashinfer 0.6.11 addresses head_dim=512, not Gemma-4's head_dim=256 + RoPE_dim=64 (which translates to NUM_MMA_D_QK/VO=32). To unblock fi-attn we'd need a separate flashinfer fix; not done as of 0.6.11. **Profile default `attention_backend: triton` remains correct.**
7. **fi_cutedsl MoE is FP4-only by design.** BF16 Gemma-4 will never work with this backend (same assertion as on the 35B-A3B-FP8 testlog). Not a regression — designed FP4-only.

### Output quality verified

- `response_snippet` (head ~1 kB + tail ~500–700 B) scanned across all 78 successful requests (6 cases × (n=1 + n=4 + n=8) = 6 × 13 = 78):
  - **0 triple-word repetitions.**
  - **0 self-correction / word-salad markers** (`self-correct` / `stop rambling` / `thinking thinking` / `retire retire` / `word salad`).
- **Output-token distribution: 910 → 1867** (median ~1400). No request hit the 3072 cap. All finish=stop (natural EOS), consistent with Gemma's concise style.
- **Tails eyeballed** for Case 12 n=8 (winner): all 8 requests end in clean concluding sentences (tables, summaries, calls-to-action — same shape as the 31B-it sibling).
- Gemma-4 dense-attn VLM is unaffected by the v0.5.11 hybrid-mamba word-salad concurrency-race — confirmed.

### Production recommendation

```yaml
moe_runner_backend: flashinfer_cutlass   # +5 tok/s vs triton @ winner shape n=8
attention_backend: triton                # fi-attn still crashes; PR #2959 ≠ this dispatch entry
disable_cuda_graph: false
disable_piecewise_cuda_graph: false      # piecewise on, +1–3 % at n=8
nccl_transport: roce
cuda_graph_max_bs: 8
```

Triton MoE (Cases 4–6) is essentially equivalent and is the current profile default — switching to `flashinfer_cutlass` would buy ~2.5 % n=8 with no quality difference. Optional.

### 0.5.10 baseline (reference)

From `TESTLOG_nv580.142_sglang-0.5.10_gemma-4-26b-a4b-it_4n.md`:

| #  | Config (moe_runner / attn / CG / piecewise) | n=1  | n=4   | n=8        | Note                  |
|----|---------------------------------------------|-----:|------:|-----------:|-----------------------|
| 04 | triton / triton / CG on / pw off            | 40.1 | 114.8 | 163.9      | stable                |
| 05 | triton / triton / **eager**                 | 20.5 | 104.9 | 159.8      | stable                |
| 06 | triton / triton / CG on / **pw on**         | 39.8 | 114.6 | **180.5 ★**| **0.5.10 winner**     |
| 10 | fi_cutlass / triton / CG on / pw off        | 39.7 | 109.4 | 174.1      | stable                |
| 11 | fi_cutlass / triton / **eager**             | 25.6 | 112.2 | 158.4      | stable                |
| 12 | fi_cutlass / triton / CG on / **pw on**     | 40.2 | 110.8 | 172.4      | stable                |
| 1–3, 7–9 | fi-attn (any MoE)                     | —    | —     | —          | startup/bench crash   |

---

## MTP sweep (Tests 19–23) — pending

The MTP cases mirror the Case 06 shape (triton-MoE + triton-attn + CG on + piecewise on) — the cookbook-validated path. Only `speculative_num_steps ∈ {2, 3, 4, 5, 6}` varies; cookbook default is `num_steps=5`.

**Expectation (revised after 31B-it Test 07 result, 2026-05-13).** The dense 31B-it sibling's Test 07 (`num_steps=2`) showed a *much larger* MTP win than initially predicted: n=1 +98 % (10.49 → 20.83), n=4 +76 % (44.06 → 77.67), accept_rate ~0.68 median. So MTP under FROZEN_KV_MTP is materially better than naked NEXTN/EAGLE would be on Gemma-4. The 26B-A4B is MoE (3.8 B active) and starts much higher at n=1 (**40.48 tok/s** baseline) — per-token compute is already low, so the absolute headroom for MTP is smaller. Revised hypothesis: solid n=1 win (target 55–65 tok/s if accept rate matches the 31B's 0.7), shrinking gain at n=4, breakeven-to-loss at n=8 where the target is already at 208 tok/s and verify-batch decode is saturated.

Quality-watch items once results land:
- **Drafter + MoE expert routing**: the auxiliary checkpoint is dense (4 layers) but the target is MoE — verify the assistant's outputs map cleanly to target tokens (no quality degradation from routing mismatch on rejected drafts).
- **mem_fraction=0.85 + drafter shard**: 26B target uses ~13 GB/GPU, drafter adds ~4 GB / TP=4 = ~1 GB/GPU; should fit with headroom. Watch for OOM in startup logs.
- **Token-acceptance rate**: small drafters on MoE targets historically accept poorly (~40 %) — if so, MTP is a loss here regardless of `num_steps`.
- **Output coherence**: re-apply baseline word-salad / triple-word grep + tail-eyeball.

