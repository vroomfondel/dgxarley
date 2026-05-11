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

## Configuration Matrix (6 cases)

All tests use: `tp=4, pp=1, ep=1, nccl_transport=roce, kv_cache_dtype=fp8_e4m3, mem_fraction_static=0.60, context_length=262144`. BF16 dense → no MoE/FP4/FP8 sweep.

| # | attention | dis_cuda_graph | dis_piecewise | Status            | n=1 tok/s | n=4 peak | n=8 peak  |
|---|-----------|----------------|---------------|-------------------|-----------|----------|-----------|
| 1 | fi        | false          | true          | **startup_crash** | —         | —        | —         |
| 2 | fi        | true           | true          | **startup_crash** | —         | —        | —         |
| 3 | fi        | false          | false         | **startup_crash** | —         | —        | —         |
| 4 | triton    | false          | true          | ok                | 10.91     | 44.10    | 81.47     |
| 5 | triton    | true           | true          | ok                | 9.26      | 42.48    | 83.06     |
| 6 | triton    | false          | false         | ok                | 10.49     | 44.06    | **85.34** |

**fi-attn (Cases 1–3) — 3× startup_crash, same FlashInfer dispatch-table miss as on 0.5.10.** The Gemma-4 head_dim=256 + RoPE=64 prefill kernel still hits `FlashInfer Internal Error: Invalid configuration : NUM_MMA_Q=1 NUM_MMA_D_QK=32 NUM_MMA_D_VO=32 NUM_MMA_KV=1 NUM_WARPS_Q=1 NUM_WARPS_KV=4` from `prefill.cuh:2978`. FlashInfer 0.6.8.post1 + sgl-kernel 0.4.2 did not fix the dispatch-table gap. Even Case 02 (eager, no CUDA graph) crashes — the assert fires at the first decode call, not during graph capture. **Workaround: triton-attn (the profile default), as on 0.5.10.**

All triton-attn cases finish with `stop` × N (Gemma is concise; ~1.2 k tokens vs the 3072 cap).

---

## Results

**Matrix complete (2026-05-11, 6/6 cases run: 3 ok, 3 startup_crash).**

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
