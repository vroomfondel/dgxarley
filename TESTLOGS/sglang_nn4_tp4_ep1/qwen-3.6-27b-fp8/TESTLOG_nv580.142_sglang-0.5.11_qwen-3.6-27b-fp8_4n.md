# SGLang Test Log — Qwen3.6 27B-FP8 (dense), 4 Nodes, TP=4 EP=1, v0.5.11

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
| Image     | `scitrera/dgx-spark-sglang:0.5.11`                 |
| Model     | `Qwen/Qwen3.6-27B-FP8`                             |
| NCCL      | 2.29.7+cuda13.2 (dgxspark-3node-ring)              |
| Transport | **RoCE** via SR-IOV VF                             |

Matrix file: `kikube/matrixtest_matrices/sglang_nn4_tp4_ep1/qwen-3.6-27b-fp8/nv580.142_sglang-0.5.11_qwen-3.6-27b-fp8_n4_ep1.yaml`

Toolchain delta vs `_sglang-0.5.10_*` testlog: PyTorch 2.9 → 2.11, CUDA 13 default,
sgl-kernel 0.4.1.post1 → 0.4.2, FlashInfer 0.6.7.post2 → 0.6.8.post1. Spec V2 with
Overlap-Scheduling is now baseline (PR #21062). See `SGLANG_v0.5.11_VERSION_CHANGES.md`.

---

## Model Notes

- 27B **dense** (NOT MoE), hybrid Gated DeltaNet + Gated Attention. Fine-grained FP8 (block 128).
- Architecture: 16 layers of (3× Gated DeltaNet → FFN) + (1× Gated Attention → FFN).
  - Gated DeltaNet: 48 linear-attn V-heads, 16 QK-heads, head_dim=128.
  - Gated Attention: 24 Q-heads, 4 KV-heads, head_dim=256, RoPE dim=64.
  - FFN intermediate: 17 408.
- Native context 262 144 (extensible to ~1 010 000 via YaRN).
- HF arch class: `Qwen3_5MoeForConditionalGeneration`-style hybrid.
- Same hybrid-mamba arch family as Qwen3.6-35B-A3B-FP8 — **inherits the same
  word-salad concurrency-race observed there in v0.5.11** (see
  `qwen-3.6-35b-a3b-fp8/TESTLOG_..._sglang-0.5.11_*` Correctness Debug Sweep).
  Verify output quality manually for n=4 and n=8.

## Configuration Matrix (18 cases)

All tests use: `tp=4, pp=1, ep=1, nccl_transport=roce, kv_cache_dtype=fp8_e4m3, mem_fraction_static=0.50, disable_deep_gemm=true, fp8_gemm_runner_backend=cutlass, context_length=262144`. Dense → no MoE-runner sweep. FP8 → no FP4 sweep. All speculative cases (7–18) use NEXTN with `mamba_scheduler_strategy=extra_buffer + enable_spec_v2=true`.

### Block A: backend baseline (no MTP, Tests 1–6)

| # | attention | dis_cuda_graph | dis_piecewise | Status | n=1 tok/s | n=4 peak | n=8 peak   |
|---|-----------|----------------|---------------|--------|-----------|----------|------------|
| 1 | fi        | false          | true          | ok     | 24.02     | 90.56    | 166.50     |
| 2 | fi        | true           | true          | ok     | 20.32     | 87.01    | 163.15     |
| 3 | fi        | false          | false         | ok     | 23.39     | 89.84    | **169.04** |
| 4 | triton    | false          | true          | ok§    | 23.61     | 91.08    | 135.29§    |
| 5 | triton    | true           | true          | ok     | 20.21     | 87.08    | 163.31     |
| 6 | triton    | false          | false         | ok     | 22.10     | 90.57    | 168.80     |

§ Case 04 n=8: 6/8 finish=length, **2/8 finish=stop** — two requests hit EOS early, leaving 8-stream peak at 135.29 tok/s. The other 7 cases at n=8 all came back 8/8 length. Not a system failure (no `failed_requests`, no repetition kill), but the early-stop behaviour drags peak. Per-request tps for the length-runs likely matches the 21 tok/s pack — re-running might give a clean 8/8 length result.

### Block B: MTP (NEXTN) baseline at num_steps=3, Tests 7–8

| # | attention | dis_piecewise | num_steps | drafts | topk | Status   | n=1 tok/s | n=4 peak | n=8 peak   |
|---|-----------|---------------|-----------|--------|------|----------|-----------|----------|------------|
| 7 | fi        | true          | 3         | 4      | 1    | ok       | 43.14     | 144.60   | **257.73** |
| 8 | triton    | true          | 3         | 4      | 1    | running† | 45.36     | 143.69   | tbd        |

† Case 08 n=1 finished with `stop` (model emitted EOS) — single request so the 45.36 tok/s reflects natural early termination, not a kill. n=4 was 4/4 length. n=8 still pending.

### Block C: winner-shape `speculative_num_steps` sweep (fi + CG on + piecewise off + MTP), Tests 9–12

| #  | num_steps | drafts | topk | Status | n=1 tok/s | n=4 peak | n=8 peak |
|----|-----------|--------|------|--------|-----------|----------|----------|
| 9  | 2         | 4      | 1    | tbd    | —         | —        | —        |
| 10 | 3         | 4      | 1    | tbd    | —         | —        | —        |
| 11 | 4         | 4      | 1    | tbd    | —         | —        | —        |
| 12 | 5         | 4      | 1    | tbd    | —         | —        | —        |

Test 10 is a re-run of Test 7 at the same num_steps=3 to validate stability of the
sweet-spot inside this block.

### Block D: piecewise CG **ON** + MTP, Tests 13–14

| #  | attention | dis_piecewise | num_steps | drafts | topk | Status | n=1 tok/s | n=4 peak | n=8 peak |
|----|-----------|---------------|-----------|--------|------|--------|-----------|----------|----------|
| 13 | fi        | false         | 3         | 4      | 1    | tbd    | —         | —        | —        |
| 14 | triton    | false         | 3         | 4      | 1    | tbd    | —         | —        | —        |

### Block E: winner-shape `speculative_num_draft_tokens` sweep, Tests 15–16

| #  | num_steps | drafts | topk | Status | n=1 tok/s | n=4 peak | n=8 peak |
|----|-----------|--------|------|--------|-----------|----------|----------|
| 15 | 3         | 6      | 1    | tbd    | —         | —        | —        |
| 16 | 3         | 8      | 1    | tbd    | —         | —        | —        |

### Block F: winner-shape `speculative_eagle_topk` sweep, Tests 17–18

| #  | num_steps | drafts | topk | Status | n=1 tok/s | n=4 peak | n=8 peak |
|----|-----------|--------|------|--------|-----------|----------|----------|
| 17 | 3         | 4      | 2    | tbd    | —         | —        | —        |
| 18 | 3         | 4      | 4    | tbd    | —         | —        | —        |

### Column Legend

| Column         | Description                                                                                     |
|----------------|-------------------------------------------------------------------------------------------------|
| attention      | `attention_backend` — `fi` = FlashInfer, `triton` = Triton                                      |
| dis_cuda_graph | `disable_cuda_graph` — true = eager, false = capture CUDA graphs                                |
| dis_piecewise  | `disable_piecewise_cuda_graph` — true = only fixed-BS graphs, false = piecewise variable-length |
| num_steps      | `speculative_num_steps` — NEXTN draft depth                                                     |
| drafts         | `speculative_num_draft_tokens` — verified per step                                              |
| topk           | `speculative_eagle_topk` — candidates per step (1 = pure NEXTN)                                 |

---

## Results

**Run in progress (started 2026-05-10 12:35:49 CEST on elite800).**

Result dir: `kikube/matrixtest/2026-05-10/results/sglang_nn4_tp4_ep1/qwen-3.6-27b-fp8/0.5.11/`.

### Progress log

| Time (CEST) | Case | n | tok/s peak | Notes                                                                                   |
|-------------|------|---|-----------:|-----------------------------------------------------------------------------------------|
| 12:39       | 01   | 1 |      24.02 | TTFT 0.39 s, length, output coherent (Sieve-of-Eratosthenes)                            |
| 12:42       | 01   | 4 |      90.56 | 4/4 length. Snippet `"Here's a thinking thinking sequence"` — minor stutter             |
| ~12:48      | 01   | 8 |     166.50 | 8/8 length. Stutter persists, bounded. Coherent through context-end                     |
| ~12:50      | 02   | 1 |      20.32 | TTFT **15.1 s** (eager), length, coherent                                               |
| ~12:52      | 02   | 4 |      87.01 | 3/4 length + 1/4 stop, coherent                                                         |
| ~12:54      | 02   | 8 |     163.15 | 8/8 length, coherent. Eager nearly closes the gap to CG-on                              |
| ~13:00      | 03   | 1 |      23.39 | TTFT 0.39 s, length, coherent                                                           |
| ~13:02      | 03   | 4 |      89.84 | 4/4 length, coherent                                                                    |
| ~13:08      | 03   | 8 | **169.04** | 8/8 length, coherent. **Block-A winner.** Piecewise-on edges piecewise-off by ~1.5 %    |
| ~13:10      | 04   | 1 |      23.61 | TTFT 3.28 s (triton-attn warm-up), length, coherent                                     |
| ~13:12      | 04   | 4 |      91.08 | 4/4 length, coherent — n=4 best of Block A                                              |
| ~13:18      | 04   | 8 |     135.29 | **6/8 length + 2/8 stop** — early-EOS drags peak. Throughput likely 165+ on a clean run |
| ~13:25      | 05   | 1 |      20.21 | TTFT **16.1 s** (triton + eager), length, coherent                                      |
| ~13:27      | 05   | 4 |      87.08 | 4/4 length, coherent                                                                    |
| ~13:30      | 05   | 8 |     163.31 | 8/8 length, coherent                                                                    |
| ~13:32      | 06   | 1 |      22.10 | TTFT 11.5 s (triton-attn first decode penalty), length, coherent                        |
| ~13:34      | 06   | 4 |      90.57 | 4/4 length, coherent                                                                    |
| ~13:35      | 06   | 8 |     168.80 | 8/8 length, includes the `"thinking thinking sequence"` stutter — bounded               |
| ~13:36      | 07   | 1 |      43.14 | **MTP win** TTFT 0.87 s, length, coherent                                               |
| ~13:37      | 07   | 4 |     144.60 | 4/4 length, coherent — MTP scales cleanly into batch                                    |
| ~13:38      | 07   | 8 | **257.73** | 8/8 length, coherent. **+52 % vs Case 03 non-MTP winner @ n=8**                         |
| ~13:40      | 08   | 1 |      45.36 | TTFT 0.79 s, **finish=stop** (natural EOS, single request), coherent                    |
| ~13:43      | 08   | 4 |     143.69 | TTFT 1.27 s, 4/4 length, coherent. Tracking Case 07 (144.60) within noise               |

### Block A (no-MTP) summary vs 0.5.10

| #  | Config                     | 0.5.10 (n=1 / n=4 / n=8) | 0.5.11 (n=1 / n=4 / n=8)   | Δ n=8       |
|----|----------------------------|--------------------------|----------------------------|-------------|
| 01 | fi + CG on + pw off        | 21.9 / 84.2 / 157.4      | 24.02 / 90.56 / 166.50     | +5.8 %      |
| 02 | fi + eager                 | 17.1 / 78.3 / 147.8      | 20.32 / 87.01 / 163.15     | +10.4 %     |
| 03 | fi + CG on + **pw on**     | 22.0 / 84.3 / 158.6      | 23.39 / 89.84 / **169.04** | +6.6 %      |
| 04 | triton + CG on + pw off    | 16.3 / 63.1 / 148.7      | 23.61 / 91.08 / 135.29§    | (early-EOS) |
| 05 | triton + eager             | 18.8 / 77.9 / 143.3      | 20.21 / 87.08 / 163.31     | +14.0 %     |
| 06 | triton + CG on + **pw on** | 21.6 / 84.0 / 157.9      | 22.10 / 90.57 / 168.80     | +6.9 %      |

**Findings so far:**
- **Block-A winner: Case 03** (fi + CG on + piecewise on) at **169.04 tok/s @ n=8**, edging Case 06 (168.80) and Case 01 (166.50). Same shape as the 35B-A3B-FP8 winner.
- **All non-trivial CG-on configs cluster within ~3 % at n=8** (163–169 tok/s), so attention backend and piecewise are practically interchangeable on this dense hybrid-mamba arch under 0.5.11.
- **Eager penalty shrunk** vs 0.5.10: was 6–9 % gap, now 2–4 %. Spec V2 + Overlap defaults likely the cause.
- **All Block-A cases are coherent.** Mild `"thinking thinking sequence"` stutter appears in Cases 01 and 06 at n=8 only — bounded, does not escalate to synonym-walk; matches the 35B testlog "minor stutter, no Word-Salad" pattern post-`0c2bdd4`.

### Block B (MTP) progress

| #  | Config                                    | 0.5.10 (n=1 / n=4 / n=8) | 0.5.11 (n=1 / n=4 / n=8)    | Δ n=8      |
|----|-------------------------------------------|--------------------------|-----------------------------|------------|
| 07 | fi + CG on + pw off + **MTP** num_steps=3 | 44.4 / 146.4 / 238.8     | 43.14 / 144.60 / **257.73** | **+7.9 %** |

**MTP n=8 throughput is +7.9 % vs 0.5.10 on this dense model** — opposite of the 35B-A3B-FP8 result where 0.5.11-MTP regressed −9 %. The Spec V2 / Overlap-Scheduling rework helps dense MTP but hurts hybrid-mamba MoE MTP. Model-architecture-dependent.

Cases 08–18 pending.

### Comparison to 0.5.10 baseline

Reference winners from `TESTLOG_nv580.142_sglang-0.5.10_qwen-3.6-27b-fp8_4n.md`:

- **Non-MTP winner:** Test 3 (fi + CG on + piecewise on) — 22.0 / 84.3 / 158.6 tok/s @ n=1/4/8.
- **MTP winner:** Test 8 (triton + CG on + piecewise off + MTP num_steps=3) — 36.6 / 152.6 / **239.4** tok/s.
  Test 7 (fi-attn variant) was the n=1 leader at 44.4 tok/s and tied at n=8 (238.8).
- **MTP gain over best non-MTP:** +102 % n=1, +74 % n=4, +52 % n=8.

For 0.5.11 the same Block A/B cases plus 4 sweep blocks (C–F) explore whether the
new defaults (Spec V2 + Overlap-Scheduling, sgl-kernel 0.4.2, FlashInfer 0.6.8.post1,
Eagle3/DFLASH CUDA-Graph-Init fix #22836) shift the optimum away from `num_steps=3 /
drafts=4 / topk=1 / piecewise=off`. Populate each block's table after the run, then
add a delta section here covering:

1. Block A vs 0.5.10 Tests 1–6 (toolchain delta only).
2. Block B vs 0.5.10 Tests 7–8 (Spec V2 default, MTP baseline).
3. Block C: best `num_steps` for this model (model card recommends 3 — does the sweep agree?).
4. Block D: does piecewise-on combine constructively with MTP, or does it cannibalise the speedup?
5. Block E/F: do larger draft pools or higher eagle_topk pay off, or do they cost more than they save?

Pay particular attention to **output quality at n=4 and n=8** — same hybrid-mamba
arch family as Qwen3.6-35B-A3B-FP8, which exhibited the word-salad concurrency-race
in v0.5.11 (see `qwen-3.6-35b-a3b-fp8/TESTLOG_..._sglang-0.5.11_*` Correctness Debug
Sweep). Verify token coherence per case before recording a `STABLE` status.

### DFLASH (intentionally not tested)

DFLASH symbols are present in the 0.5.11 image but require a separate draft-model
path (like EAGLE3); Qwen3.6 only ships built-in NEXTN/MTP heads. To bench DFLASH
on this model, a compatible draft would need to be sourced first.
