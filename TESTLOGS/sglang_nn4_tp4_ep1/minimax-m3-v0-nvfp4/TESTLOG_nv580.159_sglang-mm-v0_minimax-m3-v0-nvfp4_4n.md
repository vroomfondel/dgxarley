# SGLang Test Log — MiniMax-M3-v0-NVFP4 (multimodal MoE + MSA), 4 Nodes, TP=4 PP=1 EP=1, mm:v0 (first serving contact)

> ⚠️ **MATRIX IN PROGRESS — 5 of 7 cases recorded (as of 2026-06-17 19:42 CEST).**
> Block A (01–02) + Block B 03 (ctx128k, **new winner**) in summary. **04 + 05 both
> startup-crashed** (mfs0.60 starves the KV pool at 128k *and* 256k — see §6;
> prediction confirmed). **Case 06 (triton MoE) is serving now** (head 2/2 — triton
> loads the M3 layout, no crash; benchmark in flight). cuDNN-FP4 probe (07) pending.
> See [§ Completeness](#completeness-vs-matrix-definition).

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
Run window: 2026-06-17 15:48–16:45 UTC.

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

## Matrix shape (7 cases defined; **5 recorded, matrix in progress**)

- **Block A** (01–02): `flashinfer_cutlass` MoE + fi-attn + fi_cutlass FP4, ctx64k/mfs0.70 — CG variant (01 piecewise = live config, 02 full-CG). ✅ **both ran**
- **Block B** (03–05): context-ceiling co-fit — ctx128k/mfs0.70, ctx128k/mfs0.60, ctx256k/mfs0.60. 🟡 **03 ran (new winner); 04 + 05 both crashed (mfs0.60 too low at 128k & 256k, §6)** (the matrix's stated main prize — but the mfs-DOWN axis is backwards, §6)
- **Block C** (06): `triton` MoE probe @ ctx64k. ⏳ **pending**
- **Block D** (07): `flashinfer_cudnn` FP4 delta @ ctx64k. ⏳ **pending**

> Eager (no-CG) is OUT of the matrix by design — broken on `cutlass_moe_fp4`.

---

## Results — throughput (peak Σ per-req tok/s)

| #  | MoE | Attn | FP4 | CG | ctx | mfs | n1 | n4 | n8 | **n16 peak** | n16 agg | n16 ok | Notes |
|----|-----|------|-----|----|-----|-----|----|----|----|--------------|---------|--------|-------|
| 03 | fi_cutlass | fi | fi_cutlass | piecewise | **128k** | 0.70 | 20.34 | 56.36 | 84.96 | **119.68** | **107.30** | **16/16** | **new overall winner** — more ctx, higher tput, clean |
| 02 | fi_cutlass | fi | fi_cutlass | **full-CG** | 64k | 0.70 | 21.17 | 54.88 | 79.60 | 107.20 | 100.87 | **16/16** | best 64k config, clean 16/16 |
| 01 | fi_cutlass | fi | fi_cutlass | **piecewise** (live) | 64k | 0.70 | 21.29 | 59.60 | 84.24 | 108.36 | 92.65 | **14/16** | live config; dropped 2/16 @ n16 |
| 04 | fi_cutlass | fi | fi_cutlass | piecewise | 128k | **0.60** | — | — | — | **CRASH** | — | — | `Not enough memory` — mfs0.60 starves the 128K KV pool (see §6) |

## Results — TTFT & single-stream (n=1 first, cold server)

| #  | CG | ctx | n1 tok/s | **n1 TTFT (s)** | n4 TTFT | n8 TTFT | n16 TTFT |
|----|----|-----|----------|-----------------|---------|---------|----------|
| 01 | piecewise | 64k  | 21.29 | **24.66** | 0.96 | 0.95 | 0.95 |
| 02 | full-CG   | 64k  | 21.17 | **16.92** | 1.10 | 0.94 | 0.94 |
| 03 | piecewise | 128k | 20.34 | **11.05** | 0.97 | 0.93 | 1.14 |

---

## Analysis

### 1. M3 serves — and serves correctly
Both cases ran the full n=1→16 sweep (`outcome 29` = 1+4+8+16 requests issued). Arch, NVFP4 quant-tool matcher, MSA-via-flashinfer, and fi_cutlass MoE are all production-stable on mm:v0. Reasoning + tool-call parsers validated live (separate `/v1/chat/completions` probes).

### 2. full-CG (02) edges piecewise (01) — OPPOSITE of the M2.7 finding
On M2.7, piecewise won and full-CG flaked 1/8. On M3 it inverts: **full-CG (02) ran clean 16/16 with the best aggregate (100.87)**, while piecewise (01, the live config) **dropped 2/16 at n16** (agg 92.65). On peak Σ they're a tie (108.36 vs 107.20) — but case 01's 108.36 is over only **14 surviving** requests, so normalized it's the weaker result. full-CG also halves the cold-start: n1 TTFT 16.9 s vs 24.7 s. **Tentative: full-CG is the better serving CG mode for M3.** Caveat: single n16 run each, modest gap — wants a repeat before locking in a profile change.

### 3. M3 is ~1.6–2× slower than M2.7, as expected
n1 ~21 tok/s (M2.7 ~36); n8 peak ~80–84 (M2.7 ~136). Expected for 428B/23B-active + MSA + vision tower vs M2.7's 230B/10B. Consistent with the public GB10 datapoint (~10.7 tok/s decode on 2 nodes; 4 nodes ~2× that). per-request tps falls with concurrency (21→15→10.5→7.7) — KV/compute-bound, not a regression.

### 4. n1 TTFT is a cold-start artifact, not steady-state latency
n1 runs first on a fresh server, so its 16–25 s TTFT bundles one-time MSA/compile/graph warm. At n≥4 TTFT settles to **~0.95 s** for both cases. Read n1 TTFT as warmup, not interactive latency.

### 5. Block B opens strong: 128K context is the NEW winner, and throughput went UP
Case 03 (ctx **128k**, piecewise, mfs0.70) is the new overall winner: **n16 peak 119.68 / agg 107.30, clean 16/16** — beating both 64K cases (02 agg 100.87, 01 agg 92.65). Counter-intuitively, **doubling the context window improved n16 throughput** rather than costing it. The reason is the failures, not the context: at 64K the KV pool is tight enough that piecewise (01) evicted/dropped 2/16 at n16; at 128K the larger `max_total_num_tokens` gives the scheduler enough KV headroom to keep all 16 in flight, so aggregate rises with no fails. So 128K co-fits comfortably at mfs0.70 — the ceiling is higher than 64K, and the live 64K cap is conservative. n1 cold-start TTFT was also the lowest of the three (11.05 s).

### 6. Case 04 crashed — and exposes a backwards premise in the Block B design
Case 04 (128k, **mfs0.60**, labelled "KV-HEADROOM") died at pool-config after weight load:
```
RuntimeError: Not enough memory. Please try to increase --mem-fraction-static.
```
The matrix's Block B intent — *"push context UP and mem_fraction_static DOWN to find what co-fits"* — is **inverted for SGLang**. Unlike vLLM's `gpu_memory_utilization` (lower = less used), SGLang's `mem_fraction_static` is the **static pool that must hold weights + KV**; lowering it *shrinks* the KV budget. With ~60 GB weights, mfs0.60 (≈77 GB static budget on 128 GB) left too little for a 128K KV pool → crash. Case 03 fit 128K **because** it used the *higher* 0.70. **Corrective:** to push context further you must **raise** mfs (toward 0.80–0.85), not lower it.

**Confirmed:** case 05 (256k @ **mfs0.60**) crashed identically (`startup_crash`) — bigger context, same starved fraction, even less viable. The real 256K test needs mfs ≥ 0.80. The Block B 04/05 axis as written cannot find the upper ceiling; it only re-proves the floor. **A corrected sweep (128k/256k at mfs 0.80→0.90) is the actual open item.**

---

## Completeness vs matrix definition

| Case | Block | Config | Status |
|------|-------|--------|--------|
| 01 | A | fi_cutlass / fi / fi_cutlass / **piecewise** / ctx64k / mfs0.70 (LIVE) | ✅ ran (as `ARCH-LOAD-LITMUS`) |
| 02 | A | fi_cutlass / fi / fi_cutlass / **full-CG** / ctx64k / mfs0.70 | ✅ ran |
| 03 | B | piecewise / **ctx128k** / mfs0.70 | ✅ ran (new winner) |
| 04 | B | piecewise / ctx128k / **mfs0.60** (KV-headroom) | ❌ **startup crash** — mfs too low (§6) |
| 05 | B | piecewise / **ctx256k** / mfs0.60 (ctx-push) | ❌ **startup crash** — same starved mfs (predicted, confirmed) |
| 06 | C | **triton** MoE / ctx64k / mfs0.70 (probe) | 🟡 serving now (head 2/2 — triton loads M3; bench in flight) |
| 07 | D | **fi_cudnn** FP4 / ctx64k / mfs0.70 (probe) | ⏳ pending |

**Matrix is running** (no manual restart needed); the harness appends each case to the summary as it lands. Re-check the summary mtime to pick up 04–07. To resume manually if interrupted:
`kikube-bench matrix matrixtest_matrices/sglang_nn4_tp4_ep1/minimax-m3-v0-nvfp4/nv580.159_sglang-mm-v0_minimax-m3-v0-nvfp4_n4_ep1.yaml --start-at 4`

---

## Conclusion (provisional — 3 of 7 cases, matrix running)

- **M3 is production-validated on mm:v0** at TP=4/PP=1/EP=1, fi_cutlass MoE + fi-attn + fi_cutlass FP4. The profile's serving point is real.
- **Context: 128K beats the live 64K** (case 03, new winner: n16 agg 107.30, clean 16/16). The live 64K cap is conservative — 128K co-fits at mfs0.70 and *raises* throughput by eliminating the KV-pressure fails the 64K piecewise run hit. Likely profile change once Block B finishes: **raise `context_length` toward 128K** (pending 04/05 confirming the ceiling and that 256K either holds or marks the OOM edge).
- **CG mode:** among the 64K cases full-CG (02) > piecewise (01) on stability; but the winner overall is a *piecewise* 128K run. CG mode and context interact — defer any `disable_piecewise_cuda_graph` change until the full Block B is in.
- **Triton fallback (06), cuDNN-FP4 (07) — still pending.**

> No profile change applied from this log — provisional, matrix still running (04 deploying). Revisit once 04–07 land; the headline (context ceiling) is resolving in 128K's favour so far.
