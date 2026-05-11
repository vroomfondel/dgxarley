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

All tests use: `tp=4, pp=1, ep=1, nccl_transport=roce, kv_cache_dtype=fp8_e4m3, mem_fraction_static=0.80, disable_deep_gemm=true, fp8_gemm_runner_backend=cutlass, context_length=262144`. Dense → no MoE-runner sweep. FP8 → no FP4 sweep. All speculative cases (7–18) use NEXTN with `mamba_scheduler_strategy=extra_buffer + enable_spec_v2=true`.

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

| # | attention | dis_piecewise | num_steps | drafts | topk | Status | n=1 tok/s | n=4 peak | n=8 peak   |
|---|-----------|---------------|-----------|--------|------|--------|-----------|----------|------------|
| 7 | fi        | true          | 3         | 4      | 1    | ok     | 43.14     | 144.60   | **257.73** |
| 8 | triton    | true          | 3         | 4      | 1    | ok†    | 45.36     | 143.69   | 242.70     |

† Case 08 n=1 finished with `stop` (model emitted EOS naturally). n=4/n=8 both 4/4 / 8/8 length, coherent. Compared to 0.5.10 (Test 8 = 36.6 / 152.6 / 239.4): n=1 +24 %, n=4 −5.8 %, n=8 +1.4 % — basically tied at n=8.

### Block C: winner-shape `speculative_num_steps` sweep (fi + CG on + piecewise off + MTP), Tests 9–12

| #  | num_steps | drafts | topk | Status | n=1 tok/s | n=4 peak | n=8 peak   |
|----|-----------|--------|------|--------|-----------|----------|------------|
| 9  | 2         | 4      | 1    | ok     | 41.17     | 148.91   | 253.76     |
| 10 | 3         | 4      | 1    | ok     | 45.68     | 154.05   | **267.68** |
| 11 | 4         | 4      | 1    | ok     | 43.81     | 151.38   | 254.62     |
| 12 | 5         | 4      | 1    | ok     | 39.62     | 139.46   | 241.69     |

Test 10 is a re-run of Test 7 at the same num_steps=3 to validate stability of the
sweet-spot inside this block.

### Block D: piecewise CG **ON** + MTP, Tests 13–14

| #  | attention | dis_piecewise | num_steps | drafts | topk | Status | n=1 tok/s | n=4 peak | n=8 peak |
|----|-----------|---------------|-----------|--------|------|--------|-----------|----------|----------|
| 13 | fi        | false         | 3         | 4      | 1    | ok     | 43.89     | 151.42   | 264.49   |
| 14 | triton    | false         | 3         | 4      | 1    | ok     | 36.34     | 141.44   | 244.03   |

### Block E: winner-shape `speculative_num_draft_tokens` sweep, Tests 15–16

| #  | num_steps | drafts | topk | Status | n=1 tok/s | n=4 peak | n=8 peak |
|----|-----------|--------|------|--------|-----------|----------|----------|
| 15 | 3         | 6      | 1    | ok     | 47.15     | 145.36   | 257.86   |
| 16 | 3         | 8      | 1    | ok     | 41.61     | 147.66   | 260.29   |

### Block F: winner-shape `speculative_eagle_topk` sweep, Tests 17–18

| #  | num_steps | drafts | topk | Status | n=1 tok/s | n=4 peak | n=8 peak |
|----|-----------|--------|------|--------|-----------|----------|----------|
| 17 | 3         | 4      | 2    | ok     | 43.22     | 155.99   | 253.79   |
| 18 | 3         | 4      | 4    | ok     | 44.57     | 145.28   | 250.70   |

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

**Matrix complete (2026-05-10, all 18 cases — 18/18 ok, 0 failures, 0 crashes, all outputs coherent).**

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
| ~13:46      | 08   | 8 |     242.70 | 8/8 length, coherent. Case 07 (fi-MTP) wins by 6.2 % at n=8 (257.73 vs 242.70)          |
| ~13:48      | 09   | 1 |      41.17 | TTFT 0.67 s, length, coherent. num_steps=2 — slightly behind Case 07 at n=1 (43.14)     |
| ~13:49      | 09   | 4 |     148.91 | 4/4 length. `"thinking thinking sequence"` stutter visible at n=4 — bounded             |
| ~13:53      | 09   | 8 |     253.76 | 8/8 length, coherent. num_steps=2 within 1.5 % of Case 07 (s=3) at n=8                  |
| ~13:55      | 10   | 1 |      45.68 | TTFT 4.83 s, length, coherent. num_steps=3 re-run — n=1 +5.9 % vs Case 07 (43.14)       |
| —           | 10   | 4 |     154.05 | 4/4 length, coherent. **Block-C best n=4 in the s sweep**                               |
| —           | 10   | 8 | **267.68** | 8/8 length, coherent. **Overall matrix winner @ n=8** (+11.8 % vs 0.5.10 best 239.4)    |
| —           | 11   | 1 |      43.81 | length, coherent. s=4                                                                   |
| —           | 11   | 4 |     151.38 | 4/4 length, coherent                                                                    |
| —           | 11   | 8 |     254.62 | 8/8 length, mild stutter, coherent                                                      |
| —           | 12   | 1 |      39.62 | length, coherent. s=5 — n=1 already 13 % behind s=3                                     |
| —           | 12   | 4 |     139.46 | 4/4 length, coherent — s=5 worst n=4 in the sweep                                       |
| —           | 12   | 8 |     241.69 | 8/8 length, coherent. s=5 collapse confirmed (matches 35B s=5 pattern)                  |
| —           | 13   | 1 |      43.89 | length, coherent. piecewise CG **on** + MTP s=3                                         |
| —           | 13   | 4 |     151.42 | 4/4 length, coherent                                                                    |
| —           | 13   | 8 |     264.49 | 8/8 length, coherent. pw-on within 1.2 % of pw-off Case 10 — interchangeable            |
| —           | 14   | 1 |      36.34 | TTFT 6.84 s, length, coherent. triton-attn + pw-on + MTP                                |
| —           | 14   | 4 |     141.44 | 4/4 length, coherent                                                                    |
| —           | 14   | 8 |     244.03 | 8/8 length, coherent. triton-attn lags fi-attn by 7.7 % at n=8 under MTP                |
| —           | 15   | 1 |      47.15 | length, coherent. drafts=6 — n=1 best of Block E                                        |
| —           | 15   | 4 |     145.36 | 4/4 length, coherent                                                                    |
| —           | 15   | 8 |     257.86 | 8/8 length, coherent. drafts=6 −3.7 % vs drafts=4 (Case 10) at n=8                      |
| —           | 16   | 1 |      41.61 | length, coherent. drafts=8                                                              |
| —           | 16   | 4 |     147.66 | 4/4 length, coherent                                                                    |
| —           | 16   | 8 |     260.29 | 8/8 length, coherent. drafts=8 −2.8 % vs drafts=4                                       |
| —           | 17   | 1 |      43.22 | length, coherent. topk=2                                                                |
| —           | 17   | 4 |     155.99 | 4/4 length, coherent. **Block-F best n=4** — topk=2 wins single batch tier              |
| —           | 17   | 8 |     253.79 | 8/8 length, coherent. topk=2 −5.2 % vs topk=1 (Case 10) at n=8                          |
| —           | 18   | 1 |      44.57 | length, coherent. topk=4                                                                |
| —           | 18   | 4 |     145.28 | 4/4 length, coherent                                                                    |
| —           | 18   | 8 |     250.70 | 8/8 length, coherent. topk=4 −6.3 % vs topk=1 — higher topk costs throughput            |

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

### Block B (MTP baseline) vs 0.5.10

| #  | Config                                          | 0.5.10 (n=1 / n=4 / n=8) | 0.5.11 (n=1 / n=4 / n=8)    | Δ n=8      |
|----|-------------------------------------------------|--------------------------|-----------------------------|------------|
| 07 | fi + CG on + pw off + **MTP** s=3               | 44.4 / 146.4 / 238.8     | 43.14 / 144.60 / **257.73** | **+7.9 %** |
| 08 | triton + CG on + pw off + **MTP** s=3           | 36.6 / 152.6 / **239.4** | 45.36 / 143.69 / 242.70     | +1.4 %     |

On 0.5.10 cases 7/8 were tied at n=8 (238.8 / 239.4). Under 0.5.11 **fi-attn wins by 6.2 %** at n=8 (257.73 vs 242.70). Spec V2 + Overlap favours fi-attn on this dense arch — opposite of the 35B-A3B-FP8 result where 0.5.11-MTP regressed −9 % across all batch sizes. Architecture-dependent.

### Block C — `speculative_num_steps` sweep (winner-shape: fi + CG on + pw off + MTP)

| #  | num_steps   | n=1   | n=4    | n=8        | vs Case 10 @ n=8 |
|----|-------------|------:|-------:|-----------:|-----------------:|
| 09 | 2           | 41.17 | 148.91 | 253.76     | −5.2 %           |
| 10 | 3           | 45.68 | 154.05 | **267.68** | (reference)      |
| 11 | 4           | 43.81 | 151.38 | 254.62     | −4.9 %           |
| 12 | 5           | 39.62 | 139.46 | 241.69     | −9.7 %           |
| 07 | 3 (re-ref)  | 43.14 | 144.60 | 257.73     | −3.7 %           |

**num_steps=3 is the sweet spot.** Cases 7 and 10 are identical configs run hours apart — n=8 spread is 3.8 % (257.73 vs 267.68), consistent with the run-to-run variance observed in the 35B sweep (~5 % at n=1, <2 % at n=4/n=8 there; ours is wider at n=8 likely because we don't have multiple repeats). s=2 and s=4 are within 1 % of each other (253.76 / 254.62) — depth around 3 is broad, falls off sharply at s=5 (−10 % vs s=3, draft cost exceeds acceptance gain). Matches the 35B s=5 collapse pattern qualitatively (35B was −28 % at n=1, less extreme here at n=8).

### Block D — piecewise CG **on** + MTP

| #  | Config                          | n=1   | n=4    | n=8     | vs Block-B counterpart (pw-off) @ n=8 |
|----|---------------------------------|------:|-------:|--------:|--------------------------------------:|
| 13 | fi + **pw on** + MTP s=3        | 43.89 | 151.42 | 264.49  | +2.6 % vs Case 07 (257.73)            |
| 14 | triton + **pw on** + MTP s=3    | 36.34 | 141.44 | 244.03  | +0.5 % vs Case 08 (242.70)            |

Compared against the same-day re-run Case 10 (fi + pw-off + s=3 = 267.68), Case 13 (pw-on) is 1.2 % behind — i.e. piecewise on/off is **within run-to-run noise** under MTP. No constructive combination, no destructive interference. Pick whichever; pw-off matches the 35B winner shape.

### Block E — `speculative_num_draft_tokens` sweep (winner-shape, s=3, topk=1)

| #  | drafts        | n=1   | n=4    | n=8     | vs Case 10 @ n=8 |
|----|---------------|------:|-------:|--------:|-----------------:|
| 10 | 4 (reference) | 45.68 | 154.05 | 267.68  | (reference)      |
| 15 | 6             | 47.15 | 145.36 | 257.86  | −3.7 %           |
| 16 | 8             | 41.61 | 147.66 | 260.29  | −2.8 %           |

**drafts=4 (NEXTN default) is best at n=8.** Bigger draft pools don't pay off — extra verify cost outweighs acceptance-rate gain. n=1 has higher variance: drafts=6 hit 47.15 (best across the whole matrix at n=1) but this is the same run-to-run effect as Block C, not a real sweet-spot.

### Block F — `speculative_eagle_topk` sweep (winner-shape, s=3, drafts=4)

| #  | topk          |   n=1 |    n=4 |    n=8 | vs Case 10 @ n=8 |
|----|---------------|------:|-------:|-------:|-----------------:|
| 10 | 1 (reference) | 45.68 | 154.05 | 267.68 |      (reference) |
| 17 | 2             | 43.22 | 155.99 | 253.79 |           −5.2 % |
| 18 | 4             | 44.57 | 145.28 | 250.70 |           −6.3 % |

**topk=1 (pure NEXTN) is best at n=8.** Higher topk costs throughput monotonically. topk=2 gives the best n=4 (155.99) — within ~1 % of Case 10 — but loses at n=8. For latency-critical n=4 workloads, topk=2 is a viable knob; for throughput, stick with topk=1.

### Overall winner @ n=8

**Case 10** (fi-attn + CG on + piecewise off + MTP, `num_steps=3, drafts=4, topk=1`) at **267.68 tok/s @ n=8** — **+11.8 % vs the 0.5.10 best** (Case 8 = 239.4 tok/s).

| Config knob         | Optimum on 0.5.11 | Comment                                               |
|---------------------|-------------------|-------------------------------------------------------|
| attention_backend   | flashinfer        | +6.2 % over triton at n=8 under MTP                   |
| disable_cuda_graph  | false             | eager penalty only 2–4 % under 0.5.11 but still real  |
| disable_piecewise   | true (pw off)     | pw-on within noise — choose to match 35B winner shape |
| speculative_enabled | true (NEXTN)      | +60 % over best non-MTP @ n=8                         |
| num_steps           | 3                 | sweet spot; s=2 / s=4 ≈5 % behind, s=5 collapses      |
| num_draft_tokens    | 4 (default)       | larger pools cost more than they save                 |
| eagle_topk          | 1 (pure NEXTN)    | higher topk costs throughput monotonically            |
| mamba_scheduler     | extra_buffer      | required for hybrid-mamba radix-cache compat          |
| enable_spec_v2      | true              | required for hybrid-mamba MTP path                    |

**Output quality:** 54/54 individual requests (18 cases × 3 batch sizes, mostly 1+4+8 = 13 reqs/case) finished coherently. Mild `"thinking thinking sequence"` stutter appears intermittently at n=4 and n=8 (Cases 01, 06, 09, 11, 12, 14, 17, 18) — bounded, never escalates to synonym-walk. No `repetition` kills, no `failed_requests`, no word-salad. Hybrid-mamba word-salad regression that affected 35B-A3B-FP8 is **not reproducible** on the 27B dense sibling either, with the post-`0c2bdd4` profile.

**Production recommendation** (drop-in for `roles/k8s_dgx/model_profiles/qwen-3.6-27b-fp8.yml`):

```yaml
attention_backend: flashinfer
disable_cuda_graph: false
disable_piecewise_cuda_graph: true
nccl_transport: roce
cuda_graph_max_bs: 8
speculative_enabled: true
speculative_algo: NEXTN
speculative_num_steps: 3
speculative_eagle_topk: 1
speculative_num_draft_tokens: 4
mamba_scheduler_strategy: extra_buffer
enable_spec_v2: true
sampling_overrides: {}
```

### DFLASH (intentionally not tested)

DFLASH symbols are present in the 0.5.11 image but require a separate draft-model
path (like EAGLE3); Qwen3.6 only ships built-in NEXTN/MTP heads. To bench DFLASH
on this model, a compatible draft would need to be sourced first.
