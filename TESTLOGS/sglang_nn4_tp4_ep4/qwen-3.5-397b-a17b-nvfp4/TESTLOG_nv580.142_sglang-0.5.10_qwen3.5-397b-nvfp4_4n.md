# SGLang Test Log — Qwen3.5 397B-A17B NVFP4, 4 Nodes, TP=4 EP=4, v0.5.10

## Environment

| Component | Value                                                                                                              |
|-----------|--------------------------------------------------------------------------------------------------------------------|
| GPU       | NVIDIA GB10 (SM121/Blackwell), 128 GB per node                                                                     |
| Driver    | 580.142                                                                                                            |
| CUDA      | 13.2                                                                                                               |
| Kernel    | 6.19.11-custom                                                                                                     |
| OS        | Ubuntu 24.04 LTS (aarch64)                                                                                         |
| K3s       | v1.35.3+k3s1                                                                                                       |
| Nodes     | spark1, spark2, spark3, spark4 (1 GPU each)                                                                        |
| Image     | `scitrera/dgx-spark-sglang:0.5.10`                                                                                 |
| Model     | `nvidia/Qwen3.5-397B-A17B-NVFP4`                                                                                   |
| NCCL      | 2.29.7+cuda13.2 (`dgxspark-3node-ring` build tag from scitrera image — functionally unrelated to our 4-node setup) |
| Transport | **RoCE** via SR-IOV VF (9.78 GB/s measured bus BW)                                                                 |

---

## Model Notes

- 397B total / 17B active MoE (512 experts, top-10, softmax routing), NVFP4 quantized (~234 GB).
- Hybrid attention: 15 full GQA layers + 45 linear attention layers (every 4th layer is full attention). 60 layers total.
- 1 shared expert + 512 routed experts per MoE layer. Multimodal (text+image+video).
- Has MTP head (1 layer) for speculative decoding (NEXTN).
- `num_attention_heads=32, num_key_value_heads=2` — TP=4 per model card.
- NVFP4: only routed expert MoE FFN weights are FP4; attention, shared experts, vision encoder, lm_head, and MTP layer remain BF16.
- ~234 GB / 4 GPUs ≈ ~59 GB/GPU — fits on 4× DGX Spark.

## Key difference from the EP=1 test

- **EP=4 TP=4** — 128 of 512 experts per GPU (sharded), full intermediate dimension (not TP-sharded within MoE). Better GEMM efficiency per expert vs EP=1, but requires per-layer EP all-reduce through the `StandardDispatcher` combine path.
- **RoCE transport** — same as EP=1 (9.78 GB/s NCCL bus bandwidth).
- **Known risks at EP=4 on NVFP4:**
  - `triton` and `cutlass` direct MoE backends go through `cutlass_moe_fp4` which has the `StandardDispatcher` EP combine bug (see `SGLANG_NVFP4_SHUFFLE_ROWS_OOB_UPSTREAM_BUG.md`). Our monkey-patches in `sglang_launch.sh` (`torch.zeros` for a_map/c_map, `topk_weights.masked_fill(topk_ids < 0, 0)`) eliminate the crash but produce garbage output (the `apply_shuffle_mul_sum` path is still broken).
  - `flashinfer_cutlass` MoE has its own EP all-to-all routing and bypasses the broken codepath. This is the only MoE backend that works correctly at EP>1 for NVFP4.
  - The previous v0.5.10rc0 EP=4 matrix for this model had 100% crash rate across all 36 tests — socket transport + no monkey-patches + older sglang release. This current v0.5.10 run is the first EP=4 attempt with all fixes in place.
- **Runtime patches from `sglang_launch.sh` active:** `cute/mma.py` sm_120a/sm_121a admissible_archs, modelopt_quant.py EP-aware input_scale slicing + num_local_experts, cutlass_moe.py a_map/c_map zero-init + topk_weights mask, moe_wna16 qzeros EP remapping.

---

## Configuration Matrix

All tests use: `tp=4, pp=1, ep=4, nccl_transport=roce, quantization=modelopt_fp4, kv_cache_dtype=fp8_e4m3, mem_fraction_static=0.70, disable_deep_gemm=true, context_length=196608, max_running_requests=32, schedule_policy=lpm, watchdog_timeout=3600, dist_timeout=1800` unless noted.

| #  | nccl | moe_runner | attention | fp4_gemm   | dis_cuda_graph | dis_piecewise | Status                                     | n=1 tok/s | n=4 peak | n=8 peak  |
|----|------|------------|-----------|------------|----------------|---------------|--------------------------------------------|-----------|----------|-----------|
| 1  | roce | triton     | fi        | fi_cutlass | false          | true          | **STABLE**                                 | 20.65     | 64.4     | 96.1      |
| 2  | roce | triton     | fi        | fi_cutlass | true           | true          | **FAIL†** (garbage @ n=8)                  | 13.16     | 75.4     | ~~156.4~~ |
| 3  | roce | triton     | fi        | fi_cutlass | false          | false         | **STABLE**                                 | 21.43     | 65.7     | 98.5      |
| 4  | roce | triton     | triton    | fi_cutlass | false          | true          | **STABLE**                                 | 19.37     | 61.2     | 93.2      |
| 5  | roce | triton     | triton    | fi_cutlass | true           | true          | **FAIL** (garbage all levels)              | ~~8.56~~  | ~~62.6~~ | ~~142.8~~ |
| 6  | roce | triton     | triton    | fi_cutlass | false          | false         | **STABLE**                                 | 19.61     | 60.7     | 91.0      |
| 7  | roce | triton     | fi        | fi_cudnn   | false          | true          | **STABLE**                                 | 19.49     | 62.4     | 94.1      |
| 8  | roce | triton     | fi        | fi_cudnn   | true           | true          | **FAIL** (garbage all levels)              | ~~13.42~~ | ~~68.1~~ | ~~144.2~~ |
| 9  | roce | triton     | fi        | fi_cudnn   | false          | false         | **STABLE**                                 | 20.10     | 61.0     | 93.7      |
| 10 | roce | triton     | triton    | fi_cudnn   | false          | true          | **STABLE**                                 | 19.37     | 61.8     | 94.0      |
| 11 | roce | triton     | triton    | fi_cudnn   | true           | true          | **FAIL** (garbage all levels)              | ~~8.54~~  | ~~62.0~~ | ~~143.1~~ |
| 12 | roce | triton     | triton    | fi_cudnn   | false          | false         | **STABLE**                                 | 19.84     | 60.8     | 93.9      |
| 13 | roce | fi_cutlass | fi        | fi_cutlass | false          | true          | **FAIL** (bench_crash @ n=8)               | 19.61     | —        | —         |
| 14 | roce | fi_cutlass | fi        | fi_cutlass | true           | true          | **FAIL** (bench_crash @ n=4)               | —         | —        | —         |
| 15 | roce | fi_cutlass | fi        | fi_cutlass | false          | false         | **FAIL** (bench_crash @ n=1)               | —         | —        | —         |
| 16 | roce | fi_cutlass | triton    | fi_cutlass | false          | true          | **FAIL** (bench_crash @ n=4)               | 19.32     | —        | —         |
| 17 | roce | fi_cutlass | triton    | fi_cutlass | true           | true          | **FAIL** (bench_crash @ n=1)               | —         | —        | —         |
| 18 | roce | fi_cutlass | triton    | fi_cutlass | false          | false         | **FAIL** (bench_crash @ n=4)               | 18.81     | —        | —         |
| 19 | roce | fi_cutlass | fi        | fi_cudnn   | false          | true          | **FAIL** (head crash @ n=4)                | 19.54     | —        | —         |
| 20 | roce | fi_cutlass | fi        | fi_cudnn   | true           | true          | **FAIL** (head crash @ n=1)                | —         | —        | —         |
| 21 | roce | fi_cutlass | fi        | fi_cudnn   | false          | false         | **FAIL** (bench_crash @ n=1)               | —         | —        | —         |
| 22 | roce | fi_cutlass | triton    | fi_cudnn   | false          | true          | **FAIL** (bench_crash @ n=4)               | 19.75     | —        | —         |
| 23 | roce | fi_cutlass | triton    | fi_cudnn   | true           | true          | **FAIL** (bench_crash @ n=1)               | —         | —        | —         |
| 24 | roce | fi_cutlass | triton    | fi_cudnn   | false          | false         | **FAIL** (bench_crash @ n=1)               | —         | —        | —         |
| 25 | roce | cutlass    | fi        | fi_cutlass | false          | true          | **STABLE**                                 | 20.26     | 64.5     | 93.4      |
| 26 | roce | cutlass    | fi        | fi_cutlass | true           | true          | **FAIL** (repetition @ n=4)                | 12.32     | ~~rep~~  | —         |
| 27 | roce | cutlass    | fi        | fi_cutlass | false          | false         | **STABLE**                                 | 20.12     | 61.9     | 93.6      |
| 28 | roce | cutlass    | triton    | fi_cutlass | false          | true          | **STABLE**                                 | 18.54     | 61.5     | 94.5      |
| 29 | roce | cutlass    | triton    | fi_cutlass | true           | true          | **FAIL** (eager: rep @ n=4, garbage @ n=8) | 8.18      | ~~14.5~~ | ~~144.6~~ |
| 30 | roce | cutlass    | triton    | fi_cutlass | false          | false         | **STABLE** (retest 2026-04-14)             | 19.86     | 61.1     | 93.1      |
| 31 | roce | cutlass    | fi        | fi_cudnn   | false          | true          | **STABLE**                                 | 19.35     | 61.7     | 93.8      |
| 32 | roce | cutlass    | fi        | fi_cudnn   | true           | true          | **FAIL** (eager: rep @ n=4, garbage @ n=8) | 13.04     | ~~—~~    | ~~147.0~~ |
| 33 | roce | cutlass    | fi        | fi_cudnn   | false          | false         | **STABLE**                                 | 19.52     | 61.7     | **95.2**  |
| 34 | roce | cutlass    | triton    | fi_cudnn   | false          | true          | **STABLE**                                 | 19.72     | 61.9     | 95.1      |
| 35 | roce | cutlass    | triton    | fi_cudnn   | true           | true          | **FAIL** (eager: rep @ n=4, garbage @ n=8) | 7.71      | ~~14.5~~ | ~~143.0~~ |
| 36 | roce | cutlass    | triton    | fi_cudnn   | false          | false         | **STABLE**                                 | 19.5      | 62.0     | 94.6      |

### Column Legend

| Column         | Description                                                                                                                                                             |
|----------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| nccl           | `nccl_transport` — NCCL inter-node transport (`socket` = TCP/IP, `roce` = RDMA/RoCE via SR-IOV VF)                                                                      |
| moe_runner     | `moe_runner_backend` — MoE expert dispatch kernel (`fi_cutlass` = flashinfer_cutlass, `triton` = triton→cutlass_moe_fp4 fallback for NVFP4, `cutlass` = cutlass direct) |
| attention      | `attention_backend` — attention kernel (`fi` = FlashInfer, `triton` = Triton)                                                                                           |
| fp4_gemm       | `fp4_gemm_backend` — FP4 dense GEMM kernel (`fi_cutlass` = flashinfer_cutlass, `fi_cudnn` = flashinfer_cudnn)                                                           |
| dis_cuda_graph | `disable_cuda_graph` — true = eager mode, false = capture CUDA graphs                                                                                                   |
| dis_piecewise  | `disable_piecewise_cuda_graph` — true = only fixed-BS graphs, false = piecewise variable-length graphs                                                                  |
| n=1 tok/s      | Per-request throughput at concurrency 1                                                                                                                                 |
| n=4 peak       | Sum of per-request tok/s at concurrency 4                                                                                                                               |
| n=8 peak       | Sum of per-request tok/s at concurrency 8                                                                                                                               |

---

## Results (matrix in progress)

### Legend

`STABLE` = all 3 concurrency levels (n=1, n=4, n=8) completed cleanly.
`FAIL` = matrix harness marked the test as failed (startup crash, watchdog timeout, or bench error).
`FAIL†` = partial results collected before failure (e.g. n=1 OK but n=4 or n=8 crashed).

### Expected patterns (based on EP=1 matrix + what we know)

- **Tests 1–12 (triton MoE):** expected to crash or produce garbage — `cutlass_moe_fp4` EP combine bug. Our monkey-patches eliminate the crash but the apply_shuffle_mul_sum path is still broken. If any stable row appears here, it is a surprise.
- **Tests 13–24 (flashinfer_cutlass MoE):** this is the expected winner region — fi_cutlass MoE has its own EP all-to-all routing. At EP=1 most of these FAILED, but EP=1 was outside the fi_cutlass "normal" operating range. EP=4 is where fi_cutlass MoE was designed to work.
- **Tests 25–36 (cutlass direct MoE):** same `cutlass_moe_fp4` codepath as triton MoE → same EP combine bug → expected to fail the same way.

### Comparison with EP=1 matrix winner

EP=1 Test 28 (cutlass direct MoE, triton attn, fi_cutlass fp4, CUDA graphs on):
- n=1: 21.5 tok/s
- n=4: 67.8 tok/s
- n=8: **102.0 tok/s**

Target for EP=4: match or exceed 102.0 tok/s at n=8. EP=4 has better GEMM efficiency per expert (full intermediate dimension per GPU) but adds per-layer EP all-reduce overhead. Net direction is hard to predict.

---

## Tests 1–36: matrix in progress

### Test 1 — `triton` MoE + `fi` attn + `fi_cutlass` fp4 (CUDA graphs on, piecewise off) — **STABLE** (surprise)

- n=1: 20.65 tok/s (ttft 2.80 s)
- n=4: 64.4 peak (16.1 per-request, ttft 0.85 s)
- n=8: 96.1 peak (12.02 per-request, ttft 1.15 s), 8/8 successful, 24,153 tokens in 255.78 s

Contrary to the expected "triton MoE crashes or garbage" prediction, this row is stable at EP=4. The `cutlass_moe_fp4` EP combine monkey-patches (`a_map/c_map` zero-init + `topk_weights` mask) are holding. Output quality not yet spot-checked — a passing bench only means no exceptions, not correct generations.

At n=8 this is ~6% below the EP=1 winner (102.0 tok/s).

### Test 2 — `triton` MoE + `fi` attn + `fi_cutlass` fp4, **CUDA graphs disabled** (eager, piecewise off) — **FAIL†** (garbage output at n=8)

- n=1: 13.16 tok/s (ttft 65.18 s) — **coherent output verified** (proper quantum-entanglement explanation)
- n=4: 75.4 peak (18.86 per-request, ttft 0.81 s) — **coherent output verified** (proper network-engineering content)
- n=8: ~~156.4~~ peak — **GARBAGE**: all 8 requests produced `Here!!!!!!!!!!!...` (the literal token `!` repeated for 3072 tokens). Bench harness reported 8/8 "successful" because every request hit `finish_reason=length` and max_tokens — no exception, no empty content.

The apparent "eager beats graphs by 63% at n=8" result is bogus: at n=8 the MoE dispatch degenerates into a single-token repetition loop, which runs ~3× faster than real decoding because no real work is done per step. The n=1 and n=4 runs were **real** (verified in pod stdout via `kubectl logs`), but something in the batching / EP-combine path at batch size 8 under eager mode breaks the output distribution and pins every logit onto `!`.

**Hypothesis (unverified):** this matches the `StandardDispatcher` / `apply_shuffle_mul_sum` EP combine bug from the header notes: the `a_map/c_map` zero-init + `topk_weights.masked_fill` patches suppress the crash but do not actually fix the combine math — at higher batch sizes (n≥8) the numerical corruption *may* become catastrophic and the model collapses onto a single token. At n≤4 the corruption *may* be small enough that the model still produces usable text (but quality may be subtly degraded — not yet measured). Needs confirmation: (1) verify failure reproduces at n=8, (2) check if disabling piecewise or re-enabling CUDA graphs changes the threshold, (3) compare logits/prob distributions at n=4 vs n=8 to see if the collapse is truly concurrency-triggered rather than some other stateful effect.

Detection note: the kikube-bench harness currently accepts any `finish_reason ∈ {stop, length}` as success. To flag this failure automatically we need an output-quality check — see the follow-up section below.

### Test 3 — `triton` MoE + `fi` attn + `fi_cutlass` fp4, piecewise CUDA graphs **on** (graphs on, piecewise on) — **STABLE**

- n=1: 21.43 tok/s (ttft 0.71 s) — **coherent output verified** (CTO encryption strategy brief, proper structure)
- n=4: 65.69 peak (16.42 per-request, ttft 0.82 s), wall 187.1 s — **coherent output verified** (TCP vs UDP explanation, proper markdown table, convoy analogy)
- n=8: 98.5 peak (12.31 per-request, ttft 1.14 s), 8/8 successful, wall 249.65 s — **coherent output verified** (real thinking content about bash disk-alert scripts, stateful alert flooding, coreutils trade-offs)

Test 3 tracks Test 1 very closely across all three concurrency levels (n=1 21.4 vs 20.7, n=4 65.7 vs 64.4, n=8 98.5 vs 96.1). Enabling piecewise CUDA graphs on top of regular graphs gives a ~2.4% bump at n=8 but otherwise changes nothing — the combined-graphs path is stable.

**This refines the hypothesis from Test 2:** the n=8 garbage collapse is **not** purely batch-size-driven. Both Test 1 (CUDA graphs on, piecewise off) and Test 3 (CUDA graphs on, piecewise on) handle n=8 cleanly. Only Test 2 (`disable_cuda_graph=true`, eager mode) corrupts at n=8. So the trigger is the interaction between **eager execution and batch size ≥ 8** on the `cutlass_moe_fp4` combine path — CUDA graph capture appears to serialize or pin the EP dispatch in a way that the monkey-patches can cope with, while eager mode re-dispatches per step and hits the unpatched numerical path at batch 8. Needs further verification before calling it confirmed.

### Test 4 — `triton` MoE + **`triton` attn** + `fi_cutlass` fp4 (CUDA graphs on, piecewise off) — **STABLE**

- n=1: 19.37 tok/s (ttft 1.64 s)
- n=4: 61.23 peak (15.31 per-request, ttft 0.81 s), think_tokens vary 1115–1981 across requests
- n=8: 93.20 peak (11.65 per-request, ttft 1.17 s), 8/8 successful, think_tokens vary 1087–1615 — **coherent output verified** (real architecture-review, arena-allocation, partition re-alerting content)

Switching attention backend from `flashinfer` (Test 1) to `triton` costs roughly 3% at each concurrency level (19.4 vs 20.7, 61.2 vs 64.4, 93.2 vs 96.1). Not a meaningful difference — triton attention is slightly slower but fully functional.

### Test 5 — `triton` MoE + `triton` attn + `fi_cutlass` fp4, **CUDA graphs disabled** (eager, piecewise off) — **FAIL** (garbage at every concurrency level)

- n=1: 8.56 tok/s (ttft 56.78 s), 996 output tokens, fr=stop — **DEGRADED**: thinking starts coherent (Monty Hall setup, Bayesian enumeration) then collapses mid-thinking into garbled LaTeX fragments (`*ft  and accred\`, `\ \text\ {1\ }`, `\1\*\10-12`) before emitting the end-of-thinking tag and stopping with only ~48 content tokens.
- n=4: ~~62.58~~ peak — **GARBAGE**: all 4 responses contain `!!!!!!!!!!!` runs visible in pod stdout.
- n=8: ~~142.80~~ peak — **GARBAGE**: all 8 requests produced identical stats (`tps=17.85, think_tokens_est=771`, `output_tokens=3072`, fr=length) — textbook batched-garbage signature. Pod stdout shows `Here's a thinking!!!!!!!!` for every request.

This is the second confirmed eager-mode failure (Test 2 was the first, same backend stack with `flashinfer` attention). Both eager-mode rows on the triton MoE path are corrupt. The Test 2 hypothesis now tightens: **eager mode (`disable_cuda_graph=true`) on the `cutlass_moe_fp4` combine path produces output corruption at all concurrency levels**, not just n=8 — Test 2 n=1/n=4 looked clean only because the prompts happened to be simple enough that the corruption didn't accumulate before a natural stop; Test 5 shows the same path can also break n=1 when the thinking phase runs long. The n=8 case is the worst because everything collapses onto a single token, giving the highest aggregate tps and fooling the bench harness into reporting a speed-up.

### Test 6 — `triton` MoE + `triton` attn + `fi_cutlass` fp4, piecewise CUDA graphs on (graphs on, piecewise on) — **STABLE**

- n=1: 19.61 tok/s (ttft 0.74 s), wall 156.7 s — **coherent output verified** (REST API design, nested vs flat URLs, architecture rationale)
- n=4: 60.66 peak (15.17 per-request, ttft 0.96 s), wall 201.2 s, think_tokens vary 1185–1533 — **output verified clean**
- n=8: 90.97 peak (11.37 per-request, ttft 1.15 s), 8/8 successful, think_tokens vary 1161–1708 — **coherent output verified** (Transformer/GPU utilization content, constructor code, no `!!!!` pattern anywhere in pod stdout)

Tracks Test 4 closely across all levels (19.6 vs 19.4, 60.7 vs 61.2, 91.0 vs 93.2). Piecewise graphs add no measurable benefit on the triton-attn path here. Worst n=8 peak of the triton-attn stable rows (vs 93.2 for Test 4), though the difference is within noise.

### Test 7 — `triton` MoE + `fi` attn + **`fi_cudnn` fp4** (CUDA graphs on, piecewise off) — **STABLE**

- n=1: 19.49 tok/s (ttft 0.74 s)
- n=4: 62.44 peak (15.61 per-request, ttft 0.78 s), think_tokens vary 1160–1300
- n=8: 94.12 peak (11.76 per-request, ttft 1.09 s), 8/8 successful, think_tokens vary 1134–1442 — **coherent output verified** (real TCP-vs-UDP senior-engineer content, no `!!!!` pattern)

Switching the fp4 GEMM backend from `flashinfer_cutlass` (Test 1) to `flashinfer_cudnn` costs ~2% at n=8 (94.1 vs 96.1). The cudnn path is slightly slower here but equally stable. Matches Test 4 closely (93.2 n=8) even though Test 4 varied the *attention* backend instead — both sub-backend swaps land in the same ~93-96 peak band on the stable triton-MoE path.

### Test 8 — `triton` MoE + `fi` attn + `fi_cudnn` fp4, **eager** — **FAIL** (garbage all levels)

- n=1: ~~13.42~~ (ttft 54.77 s) — 54s TTFT matches Test 2/5 eager-mode JIT penalty signature.
- n=4: ~~68.12~~ — all 4 requests have `output_tokens=3072`, all `finish_reason=length`, think_tokens cluster tightly (1419–1481).
- n=8: ~~144.24~~ — all 8 requests identical (`tt_est=768`, `ot=3072`, fr=length). **Garbage confirmed** via pod stdout grep (440 `!!!!!!` lines in Test 8 range).

Third confirmed eager-mode failure. The `fi_cudnn` fp4 GEMM swap doesn't rescue eager mode — same corruption pattern as Tests 2 and 5. Eager + triton MoE is broken regardless of attn/fp4-gemm sub-backend.

### Test 9 — `triton` MoE + `fi` attn + `fi_cudnn` fp4, piecewise CUDA graphs on — **STABLE**

- n=1: 20.10 tok/s (ttft 0.73 s)
- n=4: 61.00 peak (15.25 per-request, ttft 0.88 s), think_tokens vary 1078–1301
- n=8: 93.68 peak (11.71 per-request, ttft 1.24 s), 8/8 successful, think_tokens vary 1070–1632 — **clean output verified** (0 `!!!!` lines in pod stdout)

### Test 10 — `triton` MoE + **`triton` attn** + `fi_cudnn` fp4 (graphs on, piecewise off) — **STABLE**

- n=1: 19.37 tok/s (ttft 2.77 s)
- n=4: 61.80 peak (15.45 per-request, ttft 0.76 s), think_tokens vary 1086–1329
- n=8: 94.03 peak (11.76 per-request, ttft 1.16 s), 8/8 successful, think_tokens vary 1080–1430 — **clean output verified**

### Test 11 — `triton` MoE + `triton` attn + `fi_cudnn` fp4, **eager** — **FAIL** (garbage all levels)

- n=1: ~~8.54~~ (ttft 61.45 s, only 1046 ot, fr=stop) — same Test 5 n=1 degradation pattern (real thinking then LaTeX-ish collapse).
- n=4: ~~62.03~~ — one request has ot=1696 (short), another tt=2001 (unusual).
- n=8: ~~143.12~~ — all 8 requests identical (`tt_est=769`, `ot=3072`). **Garbage confirmed** via pod stdout grep (383 `!!!!!!` lines).

Fourth confirmed eager-mode failure. All four eager-mode rows (Tests 2, 5, 8, 11) on the triton MoE path have produced the same batched-garbage signature.

### Test 12 — `triton` MoE + `triton` attn + `fi_cudnn` fp4, piecewise CUDA graphs on — **STABLE**

- n=1: 19.84 tok/s (ttft 2.07 s)
- n=4: 60.83 peak (15.21 per-request, ttft 0.82 s), think_tokens vary 1103–1262
- n=8: 93.87 peak (11.73 per-request, ttft 1.14 s), 8/8 successful, think_tokens vary 1056–1693 — **clean output verified**

### Tests 13-15 — `flashinfer_cutlass` MoE — **bench_crash (`cudaErrorIllegalInstruction`)**

The fi_cutlass MoE region was supposed to be the "winner region" from the hypothesis in the header — it has its own EP all-to-all routing and bypasses the broken `cutlass_moe_fp4` combine path. In practice all three completed fi_cutlass MoE rows crashed a worker pod with the same root error: `torch.AcceleratorError: CUDA error: an illegal instruction was encountered` (`cudaErrorIllegalInstruction`).

**Root cause** (from per-test worker pod logs in the result directory):

```
[TPx EPx] Scheduler hit an exception: Traceback (most recent call last):
  File "scheduler.py", line 3499, in dispatch_event_loop
    scheduler.event_loop_normal()
  File "scheduler.py", line 1320, in event_loop_normal
    self.process_batch_result(batch, result)
  File "scheduler.py", line 2817, in process_batch_result
    self.process_batch_result_decode(batch, result)
  File "scheduler_output_processor_mixin.py", line 392, in process_batch_result_decode
    next_token_ids = next_token_ids.tolist()
torch.AcceleratorError: CUDA error: an illegal instruction was encountered
```

Test 14 caught the same fault inside the NCCL watchdog instead (`ProcessGroupNCCL::WorkNCCL::finishedGPUExecutionInternal` checking a CUDA event). Same underlying error.

This is async error propagation — the `.tolist()` D2H copy (or NCCL event query) is the first sync point that surfaces a fault from a previous CUDA stream. The actual offending kernel is somewhere in the fi_cutlass MoE forward path (running on the SM121 GB10), and it emits an instruction the device aborts on. Likely candidates: a sm_90a- or sm_100a-specific cubin in the flashinfer_cutlass library that gets dispatched on sm_121 because of a relaxed compute-cap check, or a JIT-PTX path that produces an instruction not present on Blackwell GB10.

**Crash timing pattern:**

- **Test 13** (graphs on, piecewise off): n=1 completed cleanly — 19.61 tok/s, real CAP-theorem output verified in head log. n=4 ran for ~2 minutes producing coherent content at ~62 tok/s sustained (`gen throughput (token/s): 60–71`), then worker-2 crashed mid-decode at 10:24:17. Bench harness drained 4 in-flight requests, marking them aborted with non-zero think-token estimates and zero output tokens.
- **Test 14** (eager): worker-1 crashed during n=4 with the same error, caught by the NCCL watchdog.
- **Test 15** (graphs+piecewise on): worker-2 crashed at the very first n=1 request — never produced a single completion.

**Important: this is a different failure than the eager-mode garbage in Tests 2/5/8/11.** Those produced corrupt-but-completing tokens; this is a hard kernel abort. The fi_cutlass MoE region is broken at the kernel level on SM121, not at the sampling level.

The crash is **not deterministic at startup** — Test 13 ran cleanly for ~2 minutes before failing — so it's not a "kernel cubin missing" issue. It's more likely a numerical edge case in the fi_cutlass MoE forward kernel that only triggers when expert-routing hits a specific token pattern, or a stale/misaligned tensor in the EP all-to-all path.

Recommended follow-up before re-running this region:
1. Try `disable_flashinfer_cutlass_moe_fp4_allgather=True` to take the fi_cutlass-specific allgather out of the loop.
2. Check whether `flashinfer_cutlass` ships sm_120a/sm_121a cubins for the MoE kernel, or if it's PTX-JIT'd at runtime (PTX path is the likely culprit on SM121).
3. Run with `CUDA_LAUNCH_BLOCKING=1` to get a precise kernel name in the stack trace instead of the async report.

### Tests 16-17 — `fi_cutlass` MoE + **`triton` attn** + `fi_cutlass` fp4 — **bench_crash** (same `illegal instruction`)

- **Test 16** (graphs on, piecewise off): n=1 completed cleanly at 19.32 tok/s with one full coherent response (1369 think_tokens, 3072 ot, status `done`). n=4 started, all 4 requests aborted at 0 ttft with very low think-token estimates (304–323) — the worker died very early, before the thinking phase even completed. `worker-1` log: `torch.AcceleratorError: CUDA error: an illegal instruction was encountered`.
- **Test 17** (eager): never completed n=1. Single request aborted with `tt_est=19, ot=0`. `worker-2` log shows the error coming up through Triton instead of CUDA runtime: `RuntimeError: Triton Error [CUDA]: an illegal instruction was encountered`. Same underlying fault, different first sync point — eager mode hits Triton kernel sync before the next D2H copy.

These confirm the fault is **not** sensitive to the attention backend (Tests 13/14/15 used `flashinfer` attn, Tests 16/17 use `triton` attn — same crash). Five consecutive fi_cutlass MoE rows have crashed (13/14/15/16/17), with the crash point ranging from "after 2 minutes of clean decode at n=4" to "before the first response token at n=1". The pattern is consistent with a stochastic numerical edge case in the fi_cutlass MoE forward kernel that always triggers eventually but at a rate that scales with how much compute happens — n=1 with simple prompts can survive longer than n=4 with thinking-heavy prompts.

### Test 18 — `fi_cutlass` MoE + `triton` attn + `fi_cutlass` fp4, piecewise on — **bench_crash @ n=4** (worker)

Matrix run was resumed and Test 18 ran in the new session (deploy 14:02:14, bench started 14:09:55).

- n=1: 18.81 peak, status `done`, 3021 ot, 1100 think_tokens, ttft 8.24 s — **coherent output verified** (real network engineer / QUIC content in pod stdout, 0 `!!!!!` lines)
- n=4: 4/4 `aborted`, ot=0, think_tokens 696–751 — crashed during the thinking phase before any content was emitted
- n=8: never ran
- `worker-2` died with the same `torch.AcceleratorError: CUDA error: an illegal instruction was encountered`

### Test 19 — `fi_cutlass` MoE + `fi` attn + **`fi_cudnn` fp4** (graphs on, piecewise off) — **FAIL** (head crash @ n=4 — new variant)

First fi_cutlass MoE row to swap the fp4 GEMM backend from `flashinfer_cutlass` to `flashinfer_cudnn`. Test was a deliberate probe to check whether the fp4 dense GEMM path was responsible for the SM121 fault.

- n=1: 19.54 peak, status `done`, 3072 ot, 1197 think_tokens, ttft 3.06 s — **coherent output verified** (Symmetric/Asymmetric crypto exec brief in pod stdout, 0 `!!!!!` lines)
- n=4: 4/4 status `error` with partial outputs (ot 665–759) — head pod crashed mid-batch
- n=8: 8/8 `error`, ot=0 — bench harness kept firing into a dead head
- **HEAD pod (TP0 EP0)** died with same stack as Tests 13–17:
  ```
  process_batch_result_decode → next_token_ids.tolist()
  torch.AcceleratorError: CUDA error: an illegal instruction was encountered
  ```
  This is the first fi_cutlass MoE row where the head pod takes the fault instead of a worker. The matrix harness recorded `outcome=1` (n=1 succeeded, then aborted on subsequent levels) instead of the usual `bench_crash`.

**Diagnostic conclusion from Test 19**: swapping the fp4 GEMM backend (`flashinfer_cutlass` → `flashinfer_cudnn`) does **not** fix the SM121 illegal-instruction fault. The bad kernel therefore sits inside the **fi_cutlass MoE forward path itself**, not in the fp4 dense GEMM kernels. This narrows the root cause significantly: it must be one of the routed-expert FFN kernels (gemm_swiglu, scale_and_combine, or the EP dispatch/combine ops) — but not the standalone fp4 dense GEMM and not (per our earlier diagnostic) the cutlass MoE allgather.

**Pattern across Tests 13–19** (6 fi_cutlass MoE rows): every single row produces a clean coherent n=1 response (~19 tok/s, real verified content), then the next concurrency level kills the rank that gets the unlucky token distribution. The fault is concentration-dependent — single-stream decode is fine, parallel decode at n≥4 is fatal.

### Tests 20-22 — `fi_cutlass` MoE + `fi_cudnn` fp4 (all 3 graph-mode permutations × 2 attention backends) — **FAIL**

- **Test 20** (fi attn, eager): **all 13 requests "failed"** — but the bench-pod streamed output captured in Loki shows the n=1 request was producing **real coherent content** for ~40 seconds (CAP-theorem brief: Brewer/Gilbert-Lynch derivation, PACELC explanation, ZooKeeper/etcd/Cassandra/DynamoDB/Spanner/CockroachDB examples, distributed-systems-architect persona — verified). The HEAD pod died mid-stream via the NCCL watchdog (`ProcessGroupNCCL.cpp:2119 ... CUDA error: an illegal instruction was encountered`), the client got `aiohttp.TransferEncodingError: 400, message='Not enough data to satisfy transfer length header.'`, and the harness logged `status=error`. Then n=4 and n=8 piled into the dead head, all also `error` with 0 ot. **No garbage** — it's a hard crash mid-coherent-stream, just the first fi_cutlass MoE row where even the brief n=1 window doesn't get to finish. Eager mode is more fragile than graph modes here, presumably because eager re-launches the buggy kernel on every step.
- **Test 21** (fi attn, piecewise on): n=1 aborted with `external abort (pod failure)` after 1071 think_tokens. Loki shows the streamed thinking was **coherent** (DNS resolution process for a frontend dev, "Staff Infrastructure Engineer" persona, structured walk through browser cache → stub resolver → recursive → root/TLD/auth, plus Anycast/DNSSEC/DoH/DoQ/CDN sections — verified). `worker-1` died with the usual `torch.AcceleratorError`.
- **Test 22** (triton attn, graphs on, piecewise off): n=1 cleanly delivered at **19.75 tok/s** with 1406 think_tokens and 3072 ot, status `done`. Loki content verification: real Transformer-architecture deep dive (deep-learning-researcher persona, RoPE / Flash Attention / MoE coverage, full structured plan + body — verified, 0 `!!!!!` lines). n=4 then aborted instantly: all 4 requests at only 257–289 think_tokens with 0 ot — barely past the prompt header before `worker-2` died with the same `torch.AcceleratorError`. n=8 never ran.

**No garbage in any of the streamed outputs** (verified via Loki bench-pod log retrieval). These confirm the same pattern as Tests 13–19: real coherent content streams cleanly until the offending kernel happens to fire on some rank, at which point that pod dies and the request gets cut. Eager mode + fi_cutlass MoE is the worst combination because the kernel fires earlier (within the first response), while graph modes typically allow at least one full n=1 response before dying.

### Test 23 — `fi_cutlass` MoE + `triton` attn + `fi_cudnn` fp4, **eager** — **FAIL** (bench_crash @ n=1)

- n=1: 6.46 peak, status `aborted`, 324 think_tokens, 0 ot. `worker-3` died with `[E ProcessGroupNCCL.cpp:2119] Process group watchdog thread terminated with exception: CUDA error: an illegal instruction was encountered`.
- Loki content verification: real coherent TCP-vs-UDP brief (Senior Network Engineer persona, HOL blocking, QUIC, "Big Four" structured outline) was streaming when the worker died. 0 `!!!!!!` lines. **No garbage.**

This is the second eager-mode fi_cutlass-MoE row (Test 20 was the first with `fi` attention) that fails before n=1 can finish — confirms again that eager mode kills the rank earlier than graph modes do, regardless of attention backend.

The fault remains uniformly present across **11 consecutive fi_cutlass MoE rows (Tests 13–23)** regardless of any sub-backend or graph mode permutation.

### Test 24 — `fi_cutlass` MoE + `triton` attn + `fi_cudnn` fp4, piecewise on — **FAIL** (bench_crash @ n=1)

Last fi_cutlass MoE row in the matrix. Same crash signature: n=1 streamed coherent content for 1224 think_tokens before `worker-1` died with `torch.AcceleratorError: CUDA error: an illegal instruction was encountered`. **12 of 12 fi_cutlass MoE rows crashed** — uniformly broken at EP=4 on SM121 across every backend/mode permutation in the matrix.

### Test 25 — `cutlass` direct MoE + `fi` attn + `fi_cutlass` fp4 (graphs on, piecewise off) — **STABLE** (major positive surprise)

Header expectation: cutlass-direct MoE goes through the same `cutlass_moe_fp4` codepath as triton MoE → expected to fail same way as fi_cutlass MoE region. **Wrong** — Test 25 is the first stable EP=4 row in the cutlass-MoE block:

- n=1: **20.26** tok/s, status `done`, 1368 think_tokens, 2816 ot, ttft 6.84 s
- n=4: **64.54** peak (16.13 per-request, ttft 0.85 s), 4/4 done, varied tt 1117–1295
- n=8: **93.39** peak (11.67 per-request, ttft 1.07 s), 8/8 done, varied tt 1033–1603 — **clean output verified** via Loki bench-pod log (diverse coherent topics: quantum physics for kids, GitOps, Anycast/BGP DDoS resilience, etc., 0 `!!!!!` lines over the entire n=8 window)

**Comparison to the EP=1 winner** (Test 28 from the prior `ep1` matrix: `cutlass`/`triton`/`fi_cutlass`/graphs on, 21.5 / 67.8 / 102.0 tok/s):

| Concurrency | EP=1 winner | EP=4 Test 25 | Delta |
|-------------|------------:|-------------:|------:|
| n=1         |        21.5 |        20.26 | -5.8% |
| n=4         |        67.8 |         64.5 | -4.9% |
| n=8         |       102.0 |         93.4 | -8.4% |

Functionally equivalent, slightly slower per token due to EP overhead. **First viable EP=4 path for the 397B in this matrix.** Note this is the cutlass-direct MoE backend going through `cutlass_moe_fp4` — *exactly* the codepath the header notes flagged as broken — but with CUDA graphs ON, the captured graph apparently freezes a working kernel variant. The buggy version of the kernel only fires in eager mode (see Test 26 below for the proof).

### Test 26 — `cutlass` direct MoE + `fi` attn + `fi_cutlass` fp4, **eager** — **FAIL** (`repetition` @ n=4) — *new failure mode caught by guard*

Test 26 is the first matrix row to surface a **garbage failure mode that previously slipped through as fake-success** in the older eager-mode runs (Tests 2/5/8/11). The bench harness's repetition guard caught it cleanly:

- n=1: 12.32 peak, status `done`, 1015 think_tokens, 2682 ot, ttft 70.79 s — single-stream eager mode survived (long ttft is the JIT warmup penalty typical of eager).
- n=4: all 4 requests started with **coherent thinking content** (real bash disk-monitor brief with `*Self-Correction during drafting*: Wait, if the disk is *so* full that we can't write the state file, the script might fail. We should write state to /var/run or /tmp...*`) — verified in Loki — then **collapsed mid-token into `Let's!!!!!!!!!!!!!!`**. The repetition guard fired with `trigger=SUFFIX_LOOP, repetitions=4` on a 30-character pattern. All 4 streams aborted with status `repetition`. Bench wrote `failed_requests=4`.
- n=8: never ran (test failed early).

This is **the same failure mode** as Tests 2/5/8/11 (triton MoE + eager) — the model collapses onto a single token after some real thinking — but caught automatically this time instead of slipping through as fake-success.

**Major win for the diagnostic infrastructure**: the bench harness now correctly classifies eager-mode garbage as a `repetition` failure with diagnostic payload (`trigger`, `repetitions`, `pattern_text`), instead of recording a bogus 142–156 tok/s "STABLE" winner. The hand-discovered garbage cases (Tests 2/5/8/11) would now be caught automatically.

#### **Finding from Test 25 + Test 26 paired comparison** *(the cutlass-direct MoE codepath story)*

Tests 25 and 26 are the same backend stack — `cutlass`-direct MoE / `fi` attn / `fi_cutlass` fp4 — differing only in CUDA-graph mode. Side-by-side, they lock in a clean mechanistic finding:

- **Test 25** (graphs ON): **STABLE** at 93.4 tok/s n=8 peak. Graph capture froze a working kernel variant — every replay fires the same good kernel.
- **Test 26** (eager): **FAIL** with `Let's!!!!!!!` loop caught by the repetition guard. Per-step re-dispatch lets the buggy `cutlass_moe_fp4` path fire and the model collapses onto a single token.

**Cutlass-direct MoE goes through the same buggy `cutlass_moe_fp4` combine path as triton MoE** — exactly as the header hypothesis predicted. The reason it doesn't crash in graph mode is that CUDA graph capture happens to lock in a kernel variant that doesn't trigger the SM121 corruption, and the captured DAG just keeps replaying that variant on every forward pass. As soon as the dispatch logic is allowed to pick its own path per step (eager mode), it hits the unpatched fault and the model collapses onto a single token (`!`).

This explains the entire eager-mode garbage cluster cleanly:
- Triton MoE in eager mode (Tests 2/5/8/11) → garbage
- Cutlass-direct MoE in eager mode (Test 26) → garbage
- Both go through `cutlass_moe_fp4`, both have the same `apply_shuffle_mul_sum` combine bug, both produce the same `!`-loop signature

And it explains why the existing monkey-patches in `sglang_launch.sh` (`a_map/c_map` zero-init + `topk_weights.masked_fill`) *almost* work but not completely: they prevent the crash, they let CUDA graph capture pick a working variant, but they don't fix the actual numerical math of the combine kernel — so eager mode still rolls the dice on every step and loses.

**Practical implication for the model profile**: eager mode is unusable for any backend that pipes through `cutlass_moe_fp4` (= every triton/cutlass MoE row). CUDA graphs ON is non-optional. The cutlass-direct + graphs ON config (Test 25) is the recommended EP=4 setting; we now have empirical evidence that it works.

### Tests 29, 32, 35 — `cutlass` direct MoE, **eager** (three more eager-mode garbage confirmations)

All three rows are the eager (`disable_cuda_graph=true`) variants of cutlass-direct MoE across the remaining sub-backend permutations (fi_cutlass fp4 × triton attn; fi_cudnn fp4 × fi attn; fi_cudnn fp4 × triton attn). **All three fail with the same signature** already established in Tests 2/5/8/11/26:

- **n=1**: degraded TTFT (57–68 s JIT warmup), low throughput (7.7–13.0 tok/s), short outputs (909–3072 tokens) — the model survives single-stream if the prompt is light enough.
- **n=4**: the repetition guard fires on 3 of 4 requests (`status=repetition`, `ot=0`) — same `Let's!!!!!...` / `Here's a thinking!!!` collapse as Test 26. The 4th request in each run completes at ~14.5 tok/s and is real text, but the test counts as failed.
- **n=8**: all 8 "complete" at bogus ~143–147 tok/s (18.08 / 18.37 / 17.88 tok/s per request × 8) with **identical stats** (`ttft≈1.14 s, tt_est=768, ot=3072, fr=length`) — textbook batched-garbage signature, same as Tests 2/5/8/11. The n=8 collapse still slips past the repetition guard as fake success.

| Test | attn   | fp4 GEMM   | n=1 tok/s        | n=4 outcome                 | n=8 garbage |
|------|--------|------------|------------------|-----------------------------|------------:|
| 29   | triton | fi_cutlass | 8.18 (57 s TTFT) | 3/4 repetition, 1/4 done    |       144.6 |
| 32   | fi     | fi_cudnn   | 13.04 (68 s TTFT)| 4/4 repetition              |       147.0 |
| 35   | triton | fi_cudnn   | 7.71 (64 s TTFT) | 3/4 repetition, 1/4 done    |       143.1 |

These confirm (again) that the eager-mode garbage is **completely insensitive to attention or fp4 GEMM sub-backend** — it is a pure `cutlass_moe_fp4` combine-path failure. Every eager-mode row in the entire matrix (Tests 2, 5, 8, 11, 26, 29, 32, 35 = 8 of 8) has now produced the same collapse. The repetition guard is doing its job at n=4 but still cannot detect the n=8 case where the collapse is uniform across the whole batch.

### Test 30 — `cutlass` direct MoE + `triton` attn + `fi_cutlass` fp4, piecewise on — **STABLE** (first run was spurious, verified via retest)

Config: cutlass-direct MoE / triton attn / fi_cutlass fp4 / `disable_cuda_graph=false`, `disable_piecewise_cuda_graph=false` — direct sibling of Test 28 (same stack, piecewise off — STABLE at 94.5) and Test 27 (same stack with fi attn — STABLE at 93.6).

**First run (2026-04-13 16:48)** failed in a way that didn't match any known signature:
- n=1 completed cleanly at 20.24 tok/s, 932 think_tokens, `fr=stop`.
- n=4: all 4 `status=error`, `ttft=None`, `ot=0`, total_time ≈ 6.04 s — immediate error, no tokens generated.
- n=8: all 8 `status=error`, same pattern, total_time ≈ 4.04 s.

Loki retrieval of the kikube bench pod later revealed the bench client was hitting `socket.gaierror: [Errno -3] Temporary failure in name resolution` against `sglang.dgx.elasticc.io` — a **transient cluster DNS hiccup** (CoreDNS or upstream), not a backend fault. The SGLang head and workers were healthy throughout; the bench simply couldn't reach them for the ~10 s window spanning the n=4 / n=8 start.

**Retest (2026-04-14 09:57)** with an explicit `--start-at 30 --end-at 30` single-case run on the same config delivered a full clean result:

- n=1: 19.86 tok/s, ttft 0.75 s, 1516 tt, 3072 ot, `done`.
- n=4: 61.11 peak (15.28 per-request, ttft 0.94 s), 4/4 done, tt vary 1050–1347.
- n=8: **93.12** peak (11.64 per-request, ttft 1.13 s), 8/8 done, tt vary 1157–1543 — **clean output verified** via Loki (Rust vs. alternative-language deep-dive with Drop-trait / tail-latency / Arc-vs-Rc content; 0 `!!!!!!` lines, 0 `REPETITION` events).

Test 30 is therefore **STABLE**, in-line with the other cutlass-direct graph-mode rows (25/27/28/31/33/34/36). The first-run failure was an infrastructure hiccup, not a backend bug. All **8 cutlass-direct graph-mode rows now verified stable**.

### Tests 31, 33, 34, 36 — `cutlass` direct MoE + `fi_cudnn` fp4 (four stable rows, new overall cutlass-direct winner)

The four remaining graph-mode rows on the cutlass-direct MoE path — all using `fi_cudnn` as the fp4 dense GEMM backend — are **uniformly stable** with coherent output at all three concurrency levels. Thinking-token counts vary per request (1074–1762 range), output lengths vary between `stop` and `length` finishes, and per-request throughput matches the other stable cutlass-direct rows — none of the uniform-garbage signatures from the eager-mode cluster.

| Test | attn   | graph mode                 | n=1 tok/s | n=4 peak | n=8 peak | Notes                        |
|------|--------|----------------------------|----------:|---------:|---------:|------------------------------|
| 31   | fi     | graphs on, piecewise off   |     19.35 |     61.7 |     93.8 | ttft 3.84 s at n=1           |
| 33   | fi     | graphs on + piecewise on   |     19.52 |     61.7 |     95.2 | best cutlass-direct row      |
| 34   | triton | graphs on, piecewise off   |     19.72 |     61.9 |     95.1 | within noise of Test 33      |
| 36   | triton | graphs on + piecewise on   |     19.50 |     62.0 |     94.6 | —                            |

**Test 33 is the new best cutlass-direct EP=4 row at 95.2 tok/s n=8 peak**, narrowly edging Test 34 (95.1), Test 28 (94.5), Test 36 (94.6), and Test 25 (93.4). Swapping the fp4 dense GEMM backend from `fi_cutlass` → `fi_cudnn` delivers a consistent ~0.7–1.5 tok/s improvement at n=8 on the cutlass-direct path (both attention backends, both graph modes), while leaving n=1 and n=4 essentially unchanged. This is a modest but real speedup attributable purely to the dense-FP4 GEMM kernel — orthogonal to the MoE expert path.

**The cutlass-direct region is now fully characterized**: 8 stable rows (25/27/28/31/33/34/36 in graphs-on mode, plus the piecewise/no-piecewise permutations) landing in a tight 93.4–95.2 tok/s n=8 band, with `fi_cudnn` fp4 GEMM the preferred backend. Still 3.3% below the triton-MoE winner (Test 3 at 98.5 tok/s), but this is the first cleanly verified cutlass-direct EP=4 configuration set on this model.

### Final matrix summary — 36 / 36 complete

| Category                            | Count | Stable | Failed                                   |
|-------------------------------------|------:|-------:|------------------------------------------|
| triton MoE (Tests 1–12)             |    12 |      8 | 4 (all eager garbage)                    |
| fi_cutlass MoE (Tests 13–24)        |    12 |      0 | 12 (all SM121 illegal instruction)       |
| cutlass direct MoE (Tests 25–36)    |    12 |      8 | 4 (all eager garbage)                    |
| Total                               |    36 |     17 | 19                                       |

**Backend ranking at n=8 peak tok/s** (stable rows only, top 8):

| Rank | Test | MoE     | attn   | fp4 GEMM   | Graph mode        | n=8 peak |
|-----:|-----:|---------|--------|------------|-------------------|---------:|
|    1 |    3 | triton  | fi     | fi_cutlass | graphs+piecewise  |     98.5 |
|    2 |    1 | triton  | fi     | fi_cutlass | graphs on         |     96.1 |
|    3 |   33 | cutlass | fi     | fi_cudnn   | graphs+piecewise  |     95.2 |
|    4 |   34 | cutlass | triton | fi_cudnn   | graphs on         |     95.1 |
|    5 |   36 | cutlass | triton | fi_cudnn   | graphs+piecewise  |     94.6 |
|    6 |   28 | cutlass | triton | fi_cutlass | graphs on         |     94.5 |
|    7 |    7 | triton  | fi     | fi_cudnn   | graphs on         |     94.1 |
|    8 |   10 | triton  | triton | fi_cudnn   | graphs on         |     94.0 |

**Confirmed patterns** (after the full 36-row run):

1. **Eager mode (`disable_cuda_graph=true`) is uniformly broken** on every MoE backend that touches the `cutlass_moe_fp4` combine path — 8 of 8 eager rows collapsed (4 triton-MoE + 1 cutlass-direct already documented, plus the 3 new ones here). CUDA graph capture freezes a working kernel variant; eager re-dispatches per step and hits the unpatched numerical bug.
2. **fi_cutlass MoE is uniformly broken at EP=4 on SM121** — 12 of 12 rows crashed with `cudaErrorIllegalInstruction` inside the fi_cutlass MoE forward kernel. Concentration-dependent: every row delivered a clean coherent n=1 response, then died on n≥4. Not fixable at the application layer (see follow-up diagnostic below).
3. **Triton MoE and cutlass-direct MoE are equivalently stable in graph mode.** Triton MoE peaks slightly higher (98.5 vs 95.2) but both are 3.3–8.4% below the EP=1 winner (102.0 tok/s). Cutlass-direct saves ~1% Python dispatch overhead, which does not translate into a net win here.
4. **Sub-backend choices move the number by ≤2%.** Attention (`fi` vs `triton`) and fp4 GEMM (`fi_cutlass` vs `fi_cudnn`) are near-neutral; `fi_cudnn` slightly prefers the cutlass-direct path, `fi_cutlass` slightly prefers triton MoE.
5. **Test 30 first-run anomaly was spurious** — the initial 2026-04-13 run failed with `socket.gaierror: Temporary failure in name resolution` (cluster DNS hiccup on the bench-pod side, not a backend fault). A single-case retest on 2026-04-14 (`--start-at 30 --end-at 30`) passed cleanly at 19.86 / 61.1 / 93.1 tok/s, matching the rest of the cutlass-direct graph-mode block. All **8 cutlass-direct graph-mode rows are now verified STABLE**.

**Recommended production config for this model (EP=4):** Test 3 (`triton` MoE / `fi` attn / `fi_cutlass` fp4 / graphs on + piecewise on) at **98.5 tok/s n=8 peak**, 3.4% below the EP=1 winner. Second-best alternative: Test 33 (`cutlass` direct MoE / `fi` attn / `fi_cudnn` fp4 / graphs on + piecewise on) at **95.2 tok/s n=8 peak**, if the cutlass-direct path is preferred for operational reasons.

*(Obsolete interim summary superseded by "Final matrix summary — 36 / 36 complete" above.)*

---

## Follow-up diagnostic — Test 13 config + `disable_flashinfer_cutlass_moe_fp4_allgather=true` + `CUDA_LAUNCH_BLOCKING=1`

Out-of-band single deploy (not part of the matrix run), aimed at answering whether the `--disable-flashinfer-cutlass-moe-fp4-allgather` switch fixes the SM121 `cudaErrorIllegalInstruction` crash in the fi_cutlass MoE region, and whether `CUDA_LAUNCH_BLOCKING=1` gives us a precise offending kernel name in the stack trace.

**Plumbing added to the repo for this run:**
- `roles/k8s_dgx/defaults/main.yml` — new var `sglang_disable_flashinfer_cutlass_moe_fp4_allgather` (default `false`)
- `roles/k8s_dgx/tasks/sglang.yml` — new env `SGLANG_DISABLE_FLASHINFER_CUTLASS_MOE_FP4_ALLGATHER` in the sglang ConfigMap
- `roles/k8s_dgx/files/sglang_launch.sh` — new args block appending `--disable-flashinfer-cutlass-moe-fp4-allgather` when the env is `true`
- `roles/k8s_dgx/model_profiles/nvidia-qwen3.5-397b-a17b-nvfp4.yml` — `disable_flashinfer_cutlass_moe_fp4_allgather: true`
- `tasks/sglang.yml` — `CUDA_LAUNCH_BLOCKING: "1"` re-added to the ConfigMap env block

**Run config** (identical to Test 13 except for the allgather disable + launch blocking):

| Setting                                          | Value                |
|--------------------------------------------------|----------------------|
| moe_runner_backend                               | flashinfer_cutlass   |
| attention_backend                                | flashinfer           |
| fp4_gemm_backend                                 | flashinfer_cutlass   |
| disable_cuda_graph                               | false                |
| disable_piecewise_cuda_graph                     | true                 |
| cuda_graph_max_bs                                | 8                    |
| ep_size                                          | 4                    |
| tp_size                                          | 4                    |
| **disable_flashinfer_cutlass_moe_fp4_allgather** | **true** (new)       |
| **CUDA_LAUNCH_BLOCKING**                         | **1** (env)          |

**Plumbing verified end-to-end**:
- ConfigMap `sglang-config` contains `CUDA_LAUNCH_BLOCKING: "1"` and `SGLANG_DISABLE_FLASHINFER_CUTLASS_MOE_FP4_ALLGATHER: "true"` (kubectl get cm).
- Head pod `server_args=ServerArgs(..., disable_flashinfer_cutlass_moe_fp4_allgather=True, ...)` confirmed in log — the flag reached SGLang's arg parser.

**Timeline** (from Loki retrieval, pods already cleaned up):

| Event                            | Time         | Δ from deploy   |
|----------------------------------|--------------|-----------------|
| Deploy (head + 3 workers)        | 11:20:34     | —               |
| Worker-2 weight load end         | 11:27:08     | +6:34 (383 s)   |
| Worker-2 NCCL init complete      | 11:27:59     | +7:25           |
| **Worker-2 scheduler exception** | **11:30:26** | **+9:52**       |

Worker-2 (TP2 EP2) ran cleanly through model load, NCCL rendezvous, CUDA graph capture, and into sustained decode for roughly 2–3 minutes before hitting the fault. Same crash-after-warmup latency pattern as Test 13.

**Stack trace** (new, more specific than Tests 13–17):

```
[2026-04-13 11:30:26 TP2 EP2] Scheduler hit an exception: Traceback (most recent call last):
  File "sglang/srt/managers/scheduler.py", line 1319, in event_loop_normal
    result = self.run_batch(batch)
  File "sglang/srt/managers/scheduler.py", line 2724, in run_batch
    batch_result = self.model_worker.forward_batch_generation(
  File "sglang/srt/managers/tp_worker.py", line 469, in forward_batch_generation
    out = self.model_runner.forward(
  File "sglang/srt/model_executor/model_runner.py", line 2739, in forward
    output = self._forward_raw(
  File "sglang/srt/model_executor/model_runner.py", line 2804, in _forward_raw
    ret = self.graph_runner.replay(
  File "sglang/srt/model_executor/cuda_graph_runner.py", line 1161, in replay
    self.graphs[graph_key].replay()
  File "torch/cuda/graphs.py", line 139, in replay
    super().replay()
torch.AcceleratorError: CUDA error: an illegal instruction was encountered
```

**Key observations:**

1. **The `disable_flashinfer_cutlass_moe_fp4_allgather=true` flag does NOT fix the crash.** Same `cudaErrorIllegalInstruction`, ~3 min into sustained decode, on worker-2. The fi_cutlass MoE region remains uniformly broken at EP=4 on SM121.

2. **But the flag DID move the crash-surface.** Tests 13–17 surfaced the fault at either `next_token_ids.tolist()` (D2H sync after decode) or the NCCL watchdog's CUDA-event query. This run surfaces it **directly at the CUDA graph replay boundary** (`graphs[graph_key].replay() → torch/cuda/graphs.py:139 super().replay()`). This is meaningful: the bad kernel is inside the captured forward graph, the allgather codepath that was taking the blame previously is actually clean, and whatever remains of the fi_cutlass MoE forward is where the real fault lives.

3. **`CUDA_LAUNCH_BLOCKING=1` did not help sharpen the stack.** CUDA graph replay is `cudaGraphLaunch`, which submits the entire captured DAG as a single unit to the device scheduler — launch-blocking serializes individual `cudaLaunchKernel` calls but has no effect on graph submission. The first sync-able failure point inside a captured graph is the replay return, which is exactly where we see it. To get a per-kernel kernel name, we would need to either disable CUDA graphs entirely (but we know eager mode on fi_cutlass MoE also crashes — Test 14) or rebuild the image with `TORCH_USE_CUDA_DSA=1` for device-side assertions (not feasible with the upstream `scitrera/dgx-spark-sglang:0.5.10` image).

4. **Head pod was unaffected during the sustained period** — only TP2 EP2 (worker-2) triggered. The fault is rank-local and non-deterministic: in Test 13 it was also worker-2 (different pod instance), in Test 14 it was worker-1, in Test 15 worker-2 again, in Test 16 worker-1, in Test 17 worker-2. No node-affinity or VF-pinning correlation; looks like a data-dependent numerical edge case that eventually fires on whichever rank happens to process the triggering token distribution first.

**Conclusion.** Two diagnostic switches (the allgather disable and launch blocking) have been exhausted without fixing or narrowing the fault any further. The fi_cutlass MoE forward kernel has an SM121-specific illegal instruction inside the captured CUDA graph, and the upstream `scitrera/dgx-spark-sglang:0.5.10` image does not give us a cleaner handle to isolate which kernel.

**Practical fallback paths:**

- **Recommended**: roll the profile back to Test 3 config (`triton` MoE / `fi` attn / `fi_cutlass` fp4 / graphs on + piecewise on / EP=4). Best stable EP=4 pathway in this matrix at **98.5 tok/s n=8 peak**, only 3.4% below the EP=1 winner and verified coherent in pod stdout.
- **Safest**: go back to EP=1 + `cutlass` direct MoE, the established pre-matrix winner at **102.0 tok/s n=8**. Costs nothing in throughput and avoids the whole EP-combine bug family.
- **Upstream path**: file a sglang / flashinfer issue against 0.5.10 with the stack trace, the `disable_flashinfer_cutlass_moe_fp4_allgather=true` non-fix, and the SM121 GB10 hardware context. The repro is deterministic (EP=4 + fi_cutlass MoE always crashes within ~10 min of sustained decode), which should make it actionable upstream.

---

### Test 25 — n=8 inter-node traffic

![Test 25 n=8 inter-node traffic](../../../media/Bildschirmfoto_2026-04-13_15-34-44.png)
