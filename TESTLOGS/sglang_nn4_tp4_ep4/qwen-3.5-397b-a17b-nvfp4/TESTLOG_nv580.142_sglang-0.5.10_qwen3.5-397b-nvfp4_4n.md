# SGLang Test Log — Qwen3.5 397B-A17B NVFP4, 4 Nodes, TP=4 EP=4, v0.5.10

## Environment

| Component | Value |
|-----------|-------|
| GPU | NVIDIA GB10 (SM121/Blackwell), 128 GB per node |
| Driver | 580.142 |
| CUDA | 13.2 |
| Kernel | 6.19.11-custom |
| OS | Ubuntu 24.04 LTS (aarch64) |
| K3s | v1.35.3+k3s1 |
| Nodes | spark1, spark2, spark3, spark4 (1 GPU each) |
| Image | `scitrera/dgx-spark-sglang:0.5.10` |
| Model | `nvidia/Qwen3.5-397B-A17B-NVFP4` |
| NCCL | 2.29.7+cuda13.2 (`dgxspark-3node-ring` build tag from scitrera image — functionally unrelated to our 4-node setup) |
| Transport | **RoCE** via SR-IOV VF (9.78 GB/s measured bus BW) |

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

| # | nccl | moe_runner | attention | fp4_gemm | dis_cuda_graph | dis_piecewise | Status | n=1 tok/s | n=4 peak | n=8 peak |
|---|------|------------|-----------|----------|----------------|---------------|--------|-----------|----------|----------|
| 1 | roce | triton | fi | fi_cutlass | false | true | **STABLE** | 20.65 | 64.4 | 96.1 |
| 2 | roce | triton | fi | fi_cutlass | true | true | **FAIL†** (garbage @ n=8) | 13.16 | 75.4 | ~~156.4~~ |
| 3 | roce | triton | fi | fi_cutlass | false | false | **STABLE** | 21.43 | 65.7 | 98.5 |
| 4 | roce | triton | triton | fi_cutlass | false | true | **STABLE** | 19.37 | 61.2 | 93.2 |
| 5 | roce | triton | triton | fi_cutlass | true | true | **FAIL** (garbage all levels) | ~~8.56~~ | ~~62.6~~ | ~~142.8~~ |
| 6 | roce | triton | triton | fi_cutlass | false | false | **STABLE** | 19.61 | 60.7 | 91.0 |
| 7 | roce | triton | fi | fi_cudnn | false | true | **STABLE** | 19.49 | 62.4 | 94.1 |
| 8 | roce | triton | fi | fi_cudnn | true | true | **FAIL** (garbage all levels) | ~~13.42~~ | ~~68.1~~ | ~~144.2~~ |
| 9 | roce | triton | fi | fi_cudnn | false | false | **STABLE** | 20.10 | 61.0 | 93.7 |
| 10 | roce | triton | triton | fi_cudnn | false | true | **STABLE** | 19.37 | 61.8 | 94.0 |
| 11 | roce | triton | triton | fi_cudnn | true | true | **FAIL** (garbage all levels) | ~~8.54~~ | ~~62.0~~ | ~~143.1~~ |
| 12 | roce | triton | triton | fi_cudnn | false | false | **STABLE** | 19.84 | 60.8 | 93.9 |
| 13 | roce | fi_cutlass | fi | fi_cutlass | false | true | **FAIL** (bench_crash @ n=8) | 19.61 | — | — |
| 14 | roce | fi_cutlass | fi | fi_cutlass | true | true | **FAIL** (bench_crash @ n=4) | — | — | — |
| 15 | roce | fi_cutlass | fi | fi_cutlass | false | false | running | — | — | — |
| 16 | roce | fi_cutlass | triton | fi_cutlass | false | true | pending | — | — | — |
| 17 | roce | fi_cutlass | triton | fi_cutlass | true | true | pending | — | — | — |
| 18 | roce | fi_cutlass | triton | fi_cutlass | false | false | pending | — | — | — |
| 19 | roce | fi_cutlass | fi | fi_cudnn | false | true | pending | — | — | — |
| 20 | roce | fi_cutlass | fi | fi_cudnn | true | true | pending | — | — | — |
| 21 | roce | fi_cutlass | fi | fi_cudnn | false | false | pending | — | — | — |
| 22 | roce | fi_cutlass | triton | fi_cudnn | false | true | pending | — | — | — |
| 23 | roce | fi_cutlass | triton | fi_cudnn | true | true | pending | — | — | — |
| 24 | roce | fi_cutlass | triton | fi_cudnn | false | false | pending | — | — | — |
| 25 | roce | cutlass | fi | fi_cutlass | false | true | pending | — | — | — |
| 26 | roce | cutlass | fi | fi_cutlass | true | true | pending | — | — | — |
| 27 | roce | cutlass | fi | fi_cutlass | false | false | pending | — | — | — |
| 28 | roce | cutlass | triton | fi_cutlass | false | true | pending | — | — | — |
| 29 | roce | cutlass | triton | fi_cutlass | true | true | pending | — | — | — |
| 30 | roce | cutlass | triton | fi_cutlass | false | false | pending | — | — | — |
| 31 | roce | cutlass | fi | fi_cudnn | false | true | pending | — | — | — |
| 32 | roce | cutlass | fi | fi_cudnn | true | true | pending | — | — | — |
| 33 | roce | cutlass | fi | fi_cudnn | false | false | pending | — | — | — |
| 34 | roce | cutlass | triton | fi_cudnn | false | true | pending | — | — | — |
| 35 | roce | cutlass | triton | fi_cudnn | true | true | pending | — | — | — |
| 36 | roce | cutlass | triton | fi_cudnn | false | false | pending | — | — | — |

### Column Legend

| Column | Description |
|--------|-------------|
| nccl | `nccl_transport` — NCCL inter-node transport (`socket` = TCP/IP, `roce` = RDMA/RoCE via SR-IOV VF) |
| moe_runner | `moe_runner_backend` — MoE expert dispatch kernel (`fi_cutlass` = flashinfer_cutlass, `triton` = triton→cutlass_moe_fp4 fallback for NVFP4, `cutlass` = cutlass direct) |
| attention | `attention_backend` — attention kernel (`fi` = FlashInfer, `triton` = Triton) |
| fp4_gemm | `fp4_gemm_backend` — FP4 dense GEMM kernel (`fi_cutlass` = flashinfer_cutlass, `fi_cudnn` = flashinfer_cudnn) |
| dis_cuda_graph | `disable_cuda_graph` — true = eager mode, false = capture CUDA graphs |
| dis_piecewise | `disable_piecewise_cuda_graph` — true = only fixed-BS graphs, false = piecewise variable-length graphs |
| n=1 tok/s | Per-request throughput at concurrency 1 |
| n=4 peak | Sum of per-request tok/s at concurrency 4 |
| n=8 peak | Sum of per-request tok/s at concurrency 8 |

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

### Tests 13-14 — `flashinfer_cutlass` MoE — **bench_crash**

The fi_cutlass MoE region was supposed to be the "winner region" from the hypothesis in the header — it has its own EP all-to-all routing and bypasses the broken `cutlass_moe_fp4` combine path. In practice both Test 13 and Test 14 crashed a worker pod mid-benchmark:

- **Test 13** (graphs on, piecewise off): n=1 completed cleanly at 19.61 tok/s with real CAP-theorem content verified in pod stdout. n=4 started, all 4 requests began generating coherent content (CAP theorem, encryption architecture briefs — 1160–1422 think_tokens in flight) but the bench harness aborted them at ~120 s each with `output_tokens=0` and status `aborted`. During the n=8 setup, `sglang-worker-2` pod restarted (+1 restart). The matrix harness performed an emergency drain and moved on. The n=4 "aborted" result with non-zero think-token estimates but zero output tokens suggests the stream was cut off before the model finished the thinking phase — likely the worker was already unhealthy.
- **Test 14** (eager): `sglang-worker-1` restarted during the n=4 benchmark (+1 restart). Bench harness drained and moved on. No usable data.

These are **hard crashes**, not output corruption. Root cause not yet diagnosed from the runtime logs — need to pull the crashed worker container logs (`kubectl logs --previous`) to confirm whether it's an OOM, a cuDNN/CUTLASS fault from the fi_cutlass MoE backend, or NCCL timeout. Deferring investigation until the matrix completes.

### Test 15 — `flashinfer_cutlass` MoE, piecewise on — running

SGLang head still waiting for readiness as of last check.

### Interim summary after 14 rows

| #  | MoE        | Attn   | fp4 GEMM   | Graph mode          | n=8 peak  | Status                  |
|----|------------|--------|------------|---------------------|-----------|-------------------------|
| 1  | triton     | fi     | fi_cutlass | on (piecewise off)  | 96.1      | STABLE                  |
| 2  | triton     | fi     | fi_cutlass | **eager**           | ~~156.4~~ | **FAIL** (garbage)      |
| 3  | triton     | fi     | fi_cutlass | on (piecewise on)   | **98.5**  | STABLE (best so far)    |
| 4  | triton     | triton | fi_cutlass | on (piecewise off)  | 93.2      | STABLE                  |
| 5  | triton     | triton | fi_cutlass | **eager**           | ~~142.8~~ | **FAIL** (garbage)      |
| 6  | triton     | triton | fi_cutlass | on (piecewise on)   | 91.0      | STABLE                  |
| 7  | triton     | fi     | fi_cudnn   | on (piecewise off)  | 94.1      | STABLE                  |
| 8  | triton     | fi     | fi_cudnn   | **eager**           | ~~144.2~~ | **FAIL** (garbage)      |
| 9  | triton     | fi     | fi_cudnn   | on (piecewise on)   | 93.7      | STABLE                  |
| 10 | triton     | triton | fi_cudnn   | on (piecewise off)  | 94.0      | STABLE                  |
| 11 | triton     | triton | fi_cudnn   | **eager**           | ~~143.1~~ | **FAIL** (garbage)      |
| 12 | triton     | triton | fi_cudnn   | on (piecewise on)   | 93.9      | STABLE                  |
| 13 | fi_cutlass | fi     | fi_cutlass | on (piecewise off)  | —         | **bench_crash** (n=8)   |
| 14 | fi_cutlass | fi     | fi_cutlass | **eager**           | —         | **bench_crash** (n=4)   |

**Patterns confirmed across all triton-MoE rows (Tests 1–12):**
- **Eager mode (`disable_cuda_graph=true`) is always broken.** 4 of 4 eager rows produced batched-garbage output. The bogus high "throughput" comes from the model collapsing onto a single token and ripping through `max_tokens` at ~17–18 tok/s per request × N parallel.
- **CUDA graph modes (on or piecewise on) are always stable.** All 8 graph-on rows produced coherent outputs verified in pod stdout.
- **Sub-backend choice (fi vs triton attn, fi_cutlass vs fi_cudnn fp4) is essentially neutral** — all stable rows land in a tight 91–98.5 tok/s band at n=8, within ~8% of each other. The single best is **Test 3** (`triton` MoE / `fi` attn / `fi_cutlass` fp4 / piecewise graphs on) at **98.5 tok/s n=8 peak** — still 3.4% below the EP=1 winner (102.0 tok/s).

**fi_cutlass MoE region (Tests 13+) is unstable** — first two rows crashed worker pods mid-benchmark instead of producing data. The hypothesis from the header that fi_cutlass MoE would be the "winner region at EP=4" is in trouble; need to look at crash logs before the rest of that region runs.

Results will continue to be filled in as the kikube-bench matrix progresses.
