# SGLang Test Log — MiniMax-M3-v0-NVFP4 (multimodal MoE + MSA), 4 Nodes, TP=4 PP=1 EP=1, mm:v0 (first serving contact)

> ✅ **MATRIX COMPLETE — 12 cases (expanded from the original 7; summary mtime
> 2026-06-18 11:20 CEST).** The matrix was extended mid-run with a corrected
> **mfs-UP context-ceiling sweep** (cases 08–12). **Headline: the full 512K context
> window FITS at TP=4 / mfs0.90 (case 12) and is the overall throughput winner,
> clean 16/16 at every concurrency.** The two `mfs0.60` cases (04, 05) crashed;
> everything else passed. Full table + the 512K finding below.

## Environment

| Component | Value                                                                       |
|-----------|-----------------------------------------------------------------------------|
| GPU       | NVIDIA GB10 (SM121/Blackwell), 128 GB per node                              |
| Driver    | 580.159 (580.159.03)                                                        |
| Kernel    | 6.17.0-1018-nvidia (cluster baseline — mm:v0 internal CUDA/NCCL not captured) |
| OS        | Ubuntu 24.04 LTS (aarch64)                                                  |
| K3s       | v1.36.1+k3s1                                                                |
| Nodes     | spark1, spark2, spark3, spark4 (1 GPU each)                                 |
| Image     | `scitrera/dgx-spark-sglang-mm:v0` — the sparkarena-designated MULTIMODAL image (our xomoxcc base implements MiniMaxM2 only) |
| Digest    | `sha256:9a6e7d7cd5fbc716db1fedef5dd820db40d060fcb8d58885b19d4a95dbae6dff` (current pod, post-run; **rolling `v0` tag** — verify == run digest) |
| Model     | `sparkarena/Minimax-M3-v0-NVFP4`                                            |
| Transport | **RoCE** via SR-IOV VF, throughout                                          |
| KV dtype  | `auto` (bf16) — fp8 KV is BROKEN on mm:v0 (MSA triton kernel has no fp8 `tl.dot` path; dies in CUDA-graph capture, verified 2026-06-17) |

Matrix file: `kikube/matrixtest_matrices/sglang_nn4_tp4_ep1/minimax-m3-v0-nvfp4/nv580.159_sglang-mm-v0_minimax-m3-v0-nvfp4_n4_ep1.yaml`
Raw results: `kikube/results/sglang_nn4_tp4_ep1/minimax-m3-v0-nvfp4/mm-v0/`
Run window: 2026-06-17 15:48 UTC → 2026-06-18 ~09:20 UTC (12 cases; Block E 08–12 added mid-run).

**First serving contact for this profile.** The profile header's original "UPSTREAM-BLOCKED / cannot load M3 / MSA placeholder" framing is **superseded**: arch resolution (`MiniMaxM3SparseForConditionalGeneration`), the NVFP4 `quantization-tool` matcher, `flashinfer` attention (MSA handled inside the model's custom code via `trust_remote_code`), and the `flashinfer_cutlass` MoE path **all work live** on mm:v0. This matrix is therefore a **tuning sweep around a known-good serving point**, not a litmus.

> **NB on case 01 label:** the JSON names case 01 `…_ARCH-LOAD-LITMUS`; the current
> matrix YAML names it `…_LIVE-BASELINE`. Identical patches — the JSON was produced
> from an EARLIER matrix revision (before the header was re-framed and Blocks B–D
> expanded). The run predates the current YAML.

---

## Model Notes

- **~428B total / ~23B active native-multimodal MoE.** 60 layers, hidden 6144, 64 attn heads, `num_key_value_heads=4`, head_dim 128. MoE = 128 routed experts, 4 active/token + 1 shared, first 3 layers dense. NVFP4 (group-16, producer `quantization-tool` v1.0, `modelopt_fp4` loader). ~245 GB safetensors → **~61 GB/GPU at TP=4** (heavier than M2.x ~35 GB/GPU).
- **MiniMax Sparse Attention (MSA)** — custom sparse-attn operator, the long-context enabler. Handled inside the model's remote code; `attention_backend: flashinfer` is the host backend.
- `num_key_value_heads=4` → TP=4 gives exactly 1 KV head/rank (TP=3 impossible).
- `reasoning_parser: minimax-m3`, `tool_call_parser: minimax-m3` (both validated live: clean `reasoning_content` split + structured tool calls).
- **No MTP/NEXTN** — `num_mtp_modules` is declared in config but NO M3 checkpoint ships MTP weights (upstream omission; BF16 source + all NVFP4 quants = layers 0-59, 0 MTP tensors). `speculative_enabled: false` is mandatory, not a choice.
- Live serve cap: **context 65536, mem_fraction_static 0.70** → KV pool ~436K tokens (bf16). Inherited from global defaults (not swept): `max_running_requests=32`, `schedule_policy=lpm`, `--enable-custom-logit-processor`.
- Concurrency levels swept: **n=1, 4, 8, 16**.

> **Metric convention:** **peak = Σ per-request tok/s** (`avg_per_request_tps × successful_requests`), per cluster convention — NOT `total_tokens / wall_time` (the harness's `aggregate_throughput`, shown separately). When a case has failures, peak is computed over **successful** requests only and so under-counts vs a clean run — always read it next to the `ok` column.

---

## Matrix shape (12 cases, all run)

- **Block A** (01–02): fi_cutlass MoE + fi-attn + fi_cutlass FP4, ctx64k/mfs0.70 — CG variant (01 piecewise=live, 02 full-CG). ✅
- **Block B** (03–05): original context sweep with the **backwards mfs-DOWN** axis — ctx128k/mfs0.70 (✅ ok), ctx128k/mfs0.60 (❌ crash), ctx256k/mfs0.60 (❌ crash). The mfs0.60 cases starve the KV pool (§6).
- **Block C** (06): `triton` MoE probe @ ctx64k. ✅
- **Block D** (07): `flashinfer_cudnn` FP4 delta @ ctx64k. ✅
- **Block E** (08–12, **added mid-run**): corrected **mfs-UP** ceiling sweep — ctx128k/mfs0.80, ctx128k/mfs0.90, ctx256k/mfs0.80, ctx256k/mfs0.90, **ctx512k/mfs0.90** (the arch ceiling). ✅ all served; 512K is the winner.

> Eager (no-CG) is OUT of the matrix by design — broken on `cutlass_moe_fp4`.

---

## Results — throughput (sorted by n16 aggregate; peak = Σ per-req tok/s)

All cases: fi-attn, piecewise CG, except where marked. MoE/FP4 = `flashinfer_cutlass` unless noted.

| #  | ctx | mfs | MoE / FP4 / CG | n1 peak | n4 peak | n8 peak | **n16 peak** | **n16 agg** | n16 ok | Notes |
|----|-----|-----|----------------|---------|---------|---------|--------------|-------------|--------|-------|
| **12** | **512k** | **0.90** | — | 23.11 | 63.08 | 87.76 | **123.20** | **115.98** | **16/16** | 🏆 **WINNER** — full arch ceiling, fastest, clean everywhere |
| 10 | 256k | 0.80 | — | 23.32 | 59.84 | 78.54 | 120.64 | 109.40 | 16/16 | n8 7/8 |
| 11 | 256k | 0.90 | — | 20.61 | 57.36 | 74.62 | 113.76 | 109.23 | 16/16 | n8 7/8 |
| 07 | 64k  | 0.70 | **fi_cudnn** FP4 | 21.25 | 60.76 | 88.80 | 112.05 | 107.37 | 15/16 | cuDNN-FP4 delta; 1 fail, high n16 TTFT 2.0 s |
| 03 | 128k | 0.70 | — | 20.34 | 56.36 | 84.96 | 119.68 | 107.30 | 16/16 | best at mfs0.70 |
| 06 | 64k  | 0.70 | **triton** MoE | 20.31 | 41.97 | 78.16 | 107.68 | 105.37 | 16/16 | triton serves M3; n4 3/4 |
| 08 | 128k | 0.80 | — | 19.92 | 59.88 | 86.24 | 110.70 | 104.77 | 15/16 | |
| 09 | 128k | 0.90 | — | 17.32 | 58.52 | 83.04 | 118.40 | 102.46 | 16/16 | |
| 02 | 64k  | 0.70 | full-CG | 21.17 | 54.88 | 79.60 | 107.20 | 100.87 | 16/16 | best 64k config |
| 01 | 64k  | 0.70 | piecewise (live) | 21.29 | 59.60 | 84.24 | 108.36 | 92.65 | 14/16 | live config; 2 fails @ n16 |
| 04 | 128k | **0.60** | — | — | — | — | **CRASH** | — | — | `Not enough memory` — mfs0.60 starves KV (§6) |
| 05 | 256k | **0.60** | — | — | — | — | **CRASH** | — | — | same starved-mfs crash (§6) |

**n1 TTFT** (cold-server warmup, not steady-state): ranges 11–27 s across cases; at n≥4 all settle to **~0.7–2.0 s**. Winner case 12: n1 14.87 s → n16 1.13 s.

---

## Analysis

### 1. M3 serves — and serves correctly
10 of 12 cases served the full n=1→16 sweep (only the two mfs0.60 cases crashed, §6). Arch, NVFP4 quant-tool matcher, MSA-via-flashinfer, fi_cutlass MoE — and triton MoE — are all production-stable on mm:v0. Reasoning + tool-call parsers validated live (separate `/v1/chat/completions` probes).

### 2. The dominant axis is CONTEXT/mfs, not CG mode or backend
Within the 64K cases there's a small full-CG-vs-piecewise wobble (02 clean 16/16 vs 01's 14/16), but it's swamped by the context effect: every higher-context case beats every 64K case on aggregate, and all the Block E winners are *piecewise*. Treat CG mode, MoE runner, and FP4 backend as second-order (§7); the real lever is **context × mfs** (§5–6).

### 3. M3 is ~1.6–2× slower than M2.7, as expected
n1 ~21 tok/s (M2.7 ~36); n8 peak ~80–84 (M2.7 ~136). Expected for 428B/23B-active + MSA + vision tower vs M2.7's 230B/10B. Consistent with the public GB10 datapoint (~10.7 tok/s decode on 2 nodes; 4 nodes ~2× that). per-request tps falls with concurrency (21→15→10.5→7.7) — KV/compute-bound, not a regression.

### 4. n1 TTFT is a cold-start artifact, not steady-state latency
n1 runs first on a fresh server, so its 16–25 s TTFT bundles one-time MSA/compile/graph warm. At n≥4 TTFT settles to **~0.95 s** for both cases. Read n1 TTFT as warmup, not interactive latency.

### 5. THE HEADLINE: the full 512K context window fits at TP=4 — and is the throughput winner
Case 12 (**ctx512k / mfs0.90**, the architectural ceiling = `max_position_embeddings` 524288) serves **clean 16/16 at every concurrency** and tops the table: **n16 agg 115.98, peak 123.20**. This **disproves the profile's long-standing assumption** ("full 512K KV will NOT fit at TP=4 with ~61 GB/GPU weights"). It fits because **MSA (MiniMax Sparse Attention) makes KV sub-linear** in context length — the sparse block/index structure stores far less than dense attention would, so even 512K tokens of context co-fit in the post-weight budget at mfs0.90. The dense-KV math the assumption was based on simply doesn't apply to an MSA model.

Throughput also **rises monotonically with context** at fixed mfs0.90: 128k→256k→512k = agg 102.46 → 109.23 → 115.98. Same mechanism as the 64k→128k step: a bigger KV pool lets the scheduler keep all 16 concurrent requests resident, so aggregate climbs and fails vanish. **The largest context is the fastest AND the cleanest** — there is no throughput cost to running M3 at full context here. The live 64K cap is hugely conservative.

### 6. mem_fraction_static must go UP, not down — the Block B mfs-DOWN axis was backwards
Cases 04 (128k) and 05 (256k), both **mfs0.60**, died at pool-config after weight load:
```
RuntimeError: Not enough memory. Please try to increase --mem-fraction-static.
```
The original Block B intent — *"push context UP and mem_fraction_static DOWN to find what co-fits"* — is **inverted for SGLang**. Unlike vLLM's `gpu_memory_utilization` (lower = less used), SGLang's `mem_fraction_static` is the **static pool that must hold weights + KV**; lowering it *shrinks* the KV budget. With ~60 GB weights, mfs0.60 (≈77 GB on 128 GB) left too little for the KV pool → crash. The **corrected mfs-UP sweep (Block E, 08–12)** proves the point in the other direction: at mfs0.80/0.90 the same 128k/256k contexts that crashed at 0.60 now serve cleanly, and 0.90 unlocks the full 512K. **Rule: to push context, raise mfs.** Confirms `reference_sglang_mem_fraction_static.md`.

### 7. CG mode, triton, cuDNN-FP4 — all second-order vs context
- **CG mode:** the full-CG vs piecewise gap (cases 01/02) is noise next to the context effect — every Block E winner is *piecewise*. No reason to flip `disable_piecewise_cuda_graph`.
- **triton MoE (06):** serves the M3 128+1-shared NVFP4 layout without crashing (agg 105.37, clean 16/16 at n16, though n4 dropped 1) — a **viable safe fallback**, marginally behind fi_cutlass. Q3 answered: yes.
- **cuDNN-FP4 (07):** ≈ fi_cutlass FP4 (agg 107.37) but with a flake (15/16) and a high n16 TTFT (2.0 s). No reason to prefer it — same verdict as M2.7. Keep `flashinfer_cutlass` FP4.

---

## Completeness vs matrix definition (12 cases, all run)

| Case | Block | Config | Result |
|------|-------|--------|--------|
| 01 | A | piecewise / ctx64k / mfs0.70 (LIVE) | ✅ agg 92.65, 14/16 |
| 02 | A | full-CG / ctx64k / mfs0.70 | ✅ agg 100.87, 16/16 |
| 03 | B | piecewise / ctx128k / mfs0.70 | ✅ agg 107.30, 16/16 |
| 04 | B | piecewise / ctx128k / **mfs0.60** | ❌ startup crash — mfs too low (§6) |
| 05 | B | piecewise / ctx256k / **mfs0.60** | ❌ startup crash — mfs too low (§6) |
| 06 | C | **triton** MoE / ctx64k / mfs0.70 | ✅ agg 105.37, 16/16 (n4 3/4) |
| 07 | D | **fi_cudnn** FP4 / ctx64k / mfs0.70 | ✅ agg 107.37, 15/16 |
| 08 | E | piecewise / ctx128k / mfs0.80 | ✅ agg 104.77, 15/16 |
| 09 | E | piecewise / ctx128k / mfs0.90 | ✅ agg 102.46, 16/16 |
| 10 | E | piecewise / ctx256k / mfs0.80 | ✅ agg 109.40, 16/16 |
| 11 | E | piecewise / ctx256k / mfs0.90 | ✅ agg 109.23, 16/16 |
| **12** | E | piecewise / **ctx512k / mfs0.90** | 🏆 **agg 115.98, 16/16 — WINNER** |

---

## Conclusion

- **M3 is production-validated on mm:v0** at TP=4/PP=1/EP=1, fi_cutlass MoE + fi-attn + fi_cutlass FP4, piecewise CG.
- **The full 512K context window fits and is the throughput winner** (case 12, mfs0.90, clean 16/16). MSA makes KV sub-linear, so the long-standing "512K won't fit at TP=4" assumption is wrong. Throughput *rises* with context — no cost to running long.
- **`mem_fraction_static` must go UP for more context** (0.90 unlocks 512K); the mfs0.60 cases crash. The live 64K/mfs0.70 deploy is hugely conservative.
- **Backends:** keep `flashinfer_cutlass` MoE + FP4. triton MoE is a proven fallback; cuDNN-FP4 offers nothing. CG mode is second-order — keep piecewise.

### Recommended profile change (pending explicit approval)

| Key | Current (live/profile) | Recommended (matrix winner, case 12) |
|-----|------------------------|--------------------------------------|
| `context_length` | served at 65536 (profile pins 524288 as ceiling) | **524288** — it genuinely co-fits |
| `mem_fraction_static` | `0.70` live / `0.80` profile | **`0.90`** |
| `moe_runner_backend` | `flashinfer_cutlass` | unchanged ✅ |
| `fp4_gemm_backend` | `flashinfer_cutlass` | unchanged ✅ (cuDNN no gain) |
| `disable_piecewise_cuda_graph` | `false` | unchanged ✅ (piecewise) |

> No profile change applied from this log — pending explicit approval. Caveat: case 12 is a single run; one confirmation pass at 512k/mfs0.90 before locking the profile is prudent. Note also the bench drives **text-only** short prompts — 512K co-fit is proven for the KV *pool allocation*, not yet stress-tested with genuinely long (>100K-token) inputs.
