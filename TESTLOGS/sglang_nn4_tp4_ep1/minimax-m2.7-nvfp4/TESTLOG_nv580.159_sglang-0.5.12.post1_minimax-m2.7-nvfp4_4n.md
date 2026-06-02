# SGLang Test Log — MiniMax-M2.7-NVFP4 (MoE), 4 Nodes, TP=4 PP=1 EP=1, v0.5.12.post1 (first TP=4 contact)

## Environment

| Component | Value                                                                       |
|-----------|-----------------------------------------------------------------------------|
| GPU       | NVIDIA GB10 (SM121/Blackwell), 128 GB per node                              |
| Driver    | 580.159                                                                     |
| CUDA      | 13.2 (image base `cu132`)                                                   |
| Kernel    | 6.17.0-1018-nvidia                                                          |
| OS        | Ubuntu 24.04 LTS (aarch64)                                                  |
| K3s       | v1.36.1+k3s1                                                                |
| Nodes     | spark1, spark2, spark3, spark4 (1 GPU each)                                 |
| Image     | `xomoxcc/dgx-spark-sglang:0.5.12.post1-sm121`                              |
| Model     | `nvidia/MiniMax-M2.7-NVFP4`                                                 |
| NCCL      | 2.30.4+cuda13.2 (NVIDIA upstream — see note below)                          |
| Transport | **RoCE** via SR-IOV VF, throughout                                          |
| AllReduce | Legacy (`SGLANG_USE_JIT_ALL_REDUCE=0` + `SGLANG_OPT_USE_CUSTOM_ALL_REDUCE_V2=0`) |
| NVLS      | `enable_nccl_nvls=False` (SGLang default for this shape)                    |

Matrix file: `kikube/matrixtest_matrices/sglang_nn4_tp4_ep1/minimax-m2.7-nvfp4/nv580.159_sglang-0.5.12.post1_minimax-m2.7-nvfp4_n4_ep1.yaml`
Raw results: `kikube/matrixtest/2026-06-01/results/sglang_nn4_tp4_ep1/minimax-m2.7-nvfp4/0.5.12.post1/`

**First TP=4 contact for this profile.** The M2.7 profile was cloned from the M2.5 lineage, which ran **PP=4/TP=1** (8 KV heads not divisible by 3 → TP=3 impossible). 8 *is* divisible by 4, so TP=4/PP=1 splits cleanly to 2 KV heads/rank with **no pipeline bubbles**. Every inherited tuning value in the profile (triton MoE, eager, the "fi_cutlass MoE crashes" finding) was measured on PP=4 and does **not** transfer — this matrix re-establishes all of it.

**NCCL note:** the running image already carries the upstream `NVIDIA/nccl` **2.30.4** bump (banner `NCCL version 2.30.4+cuda13.2` on head + all 3 workers), replacing the former `zyang-dev` 3-node-ring fork (2.29.7). NVLS is off, so the 2.30.4 NVLS-MoE-hang regression ([NVIDIA/nccl#2167](https://github.com/NVIDIA/nccl/issues/2167)) is not a factor here.

---

## Model Notes

- **230B total / 10B active MoE.** 62 layers, hidden 3072, 256 routed experts / 8 active per token (sigmoid routing), `num_key_value_heads=8`. NVFP4 (`modelopt_fp4`, ~139 GB weights). Architecture identical to M2.5.
- TP=4/PP=1: each rank holds 2 KV heads + a 1/4 shard of every layer (~35 GB/GPU), leaving the bulk of the 128 GB unified memory for KV cache. Per-layer all-reduce traffic is RoCE-bound.
- `context_length: 196608` (config `max_position_embeddings`; the model card's "204800" is its benchmark ISL, not the arch window).
- `reasoning_parser: minimax`, `tool_call_parser: minimax-m2`, `kv_cache_dtype: fp8_e4m3`, `mem_fraction_static: 0.80` (0.40 for the MTP cases).
- Concurrency levels swept: **n=1, n=4, n=8**. `max_running_requests=32`, `schedule_policy=lpm`, `chunked_prefill_size=8192`.

## Matrix shape (14 cases)

- **Block A** (01–03): `flashinfer_cutlass` MoE + flashinfer attn + fi_cutlass FP4 — CG sweep (no-CG / piecewise / full-CG). Model-card-recommended MoE → winner-likely.
- **Block B** (04–06): `triton` MoE + flashinfer attn + fi_cutlass FP4 — CG sweep. PP=4 winner → proven-safe fallback.
- **PROBE** (07–08): triton-attention variants.
- **FP4-delta** (09–10): `flashinfer_cudnn` FP4 instead of fi_cutlass, on fi_cutlass-MoE and triton-MoE.
- **PROBE** (11–12): `flashinfer_trtllm` and `cutlass` (direct) MoE runners.
- **Block F** (13–14): NEXTN speculative decoding on the winner shape (balanced / low-latency sampling).

> **Metric convention:** **peak = Σ per-request tok/s** (`avg_per_request_tps × successful_requests`), per cluster convention — NOT `total_tokens / wall_time` (the harness's `aggregate_throughput`, shown separately). Aggregate under-counts by the prefill/queue overlap; peak reflects sustained decode.

---

## Results — throughput (peak Σ per-req tok/s)

| #  | MoE | Attn | FP4 | CG | n1 | n4 | **n8 peak** | n8 agg | n8 ok | Notes |
|----|-----|------|-----|----|----|----|-------------|--------|-------|-------|
| 01 | fi_cutlass | fi | fi_cutlass | no-CG  | 25.21 | 90.72 | **138.88** | 123.90 | 8/8 | harness winner (top n8 agg), but n1 TTFT 11.9 s ⚠️ |
| 07 | fi_cutlass | **triton** | fi_cutlass | piecewise | 35.01 | 95.84 | **138.48** | 120.67 | 8/8 | PROBE — triton-attn competitive |
| 02 | fi_cutlass | fi | fi_cutlass | **piecewise** | 35.87 | 92.28 | **136.00** | 121.57 | 8/8 | **best all-rounder** (see below) |
| 06 | triton | fi | fi_cutlass | full-CG | 33.72 | 90.88 | 135.68 | 113.54 | 8/8 | |
| 04 | triton | fi | fi_cutlass | no-CG  | 15.40 | 84.20 | 135.36 | 106.01 | 8/8 | n1 TTFT **67.4 s** ⚠️⚠️ |
| 05 | triton | fi | fi_cutlass | piecewise | 33.95 | 87.92 | 134.88 | 113.69 | 8/8 | |
| 08 | triton | **triton** | fi_cutlass | piecewise | 33.35 | 90.08 | 132.56 | 119.75 | 8/8 | PROBE |
| 10 | triton | fi | **fi_cudnn** | piecewise | 34.14 | 88.64 | 130.96 | 121.78 | 8/8 | cuDNN-FP4 delta |
| 03 | fi_cutlass | fi | fi_cutlass | full-CG | 36.33 | 95.76 | 125.09 | 107.23 | **7/8** | 1 fail @ n8 |
| 09 | fi_cutlass | fi | **fi_cudnn** | piecewise | 35.83 | 96.48 | 123.62 | 103.49 | **7/8** | cuDNN-FP4 delta, 1 fail @ n8 |
| 13 | fi_cutlass | fi | fi_cutlass | piecewise + **NEXTN** | 24.54 | 64.32 | 84.49 | 71.37 | 7/8 | spec balanced — net loss |
| 14 | fi_cutlass | fi | fi_cutlass | piecewise + **NEXTN** | 21.17 | 52.52 | 70.56 | 57.96 | 7/8 | spec low-latency — net loss |
| 11 | **flashinfer_trtllm** | fi | fi_cutlass | piecewise | — | — | **CRASH** | — | — | startup_crash (PROBE) |
| 12 | **cutlass** (direct) | fi | fi_cutlass | piecewise | — | — | **CRASH** | — | — | startup_crash (PROBE) |

## Results — single-stream latency (n=1)

| #  | shape | n1 tok/s | avg TTFT (s) | verdict |
|----|-------|----------|--------------|---------|
| 03 | fi_cutlass / full-CG       | 36.33 | 0.38 | best n1 tps + TTFT (but 1 fail @ n8) |
| 02 | fi_cutlass / piecewise     | 35.87 | 0.42 | top-tier, stable 8/8 everywhere |
| 09 | fi_cutlass / cuDNN / piecewise | 35.83 | 0.40 | (1 fail @ n8) |
| 07 | fi_cutlass / triton-attn / piecewise | 35.01 | 0.41 | |
| 06 | triton / full-CG           | 33.72 | 0.45 | |
| 05 | triton / piecewise         | 33.95 | 0.42 | |
| 01 | fi_cutlass / **no-CG**      | 25.21 | **11.94** | eager cold-start penalty ⚠️ |
| 04 | triton / **no-CG**          | 15.40 | **67.38** | eager + triton-MoE cold-start ⚠️⚠️ |

---

## Analysis

### 1. The harness "winner" (01) is a throughput-only winner — do not adopt it verbatim
Case **01** (`fi_cutlass-moe / fi-attn / fi_cutlass-fp4 / no-CG`) tops both n8 peak (138.88) and n8 aggregate (123.90), so the harness flagged it as winner. But it runs **eager** (`disable_cuda_graph + disable_piecewise`), which pays a brutal first-token penalty: **11.94 s TTFT at n=1** and only 25.21 tok/s single-stream. That is unusable for interactive serving. The eager penalty also drags case **04** (triton-MoE no-CG) to a **67.4 s** n=1 TTFT — the worst result in the matrix.

### 2. The top six are within ~3% — the differentiator is latency, and piecewise CUDA graph wins it
n8 peak spread across cases 01/07/02/06/04/05 is 138.88 → 134.88 (~3%, effectively tied). With throughput a wash, the tiebreak is single-stream latency, where the **piecewise / full-CG** variants beat the eager ones by 25–60×. **Case 02** (`fi_cutlass-moe / fi-attn / fi_cutlass-fp4 / piecewise`) is the best balance: n8 peak 136.00 (within 2% of max), n1 35.87 tok/s @ 0.42 s TTFT, and **clean 8/8 at every concurrency level**. Case 03 (full-CG) edges n1 slightly (36.33 @ 0.38 s) but dropped 1/8 at n8.

### 3. Block A's hypothesis confirmed: `flashinfer_cutlass` MoE is the winner on TP=4, NOT a crash
The inherited PP=4 finding ("fi_cutlass MoE crashes, Xid 13") was a **pipeline-stage interaction** and does not carry over. On TP=4/PP=1, `flashinfer_cutlass` is stable across all CG variants and tops the table — matching the NVIDIA model-card recommendation and the proven NVFP4 path on the cluster's other TP models (qwen3-235b, glm-4.7). Piecewise CUDA graphs (OOM'd on PP=4 — "58 chunks × 256 experts per stage") are **viable on PP=1** since there are no pipeline stages.

### 4. cuDNN-FP4 ≈ cutlass-FP4, marginally worse + one flake
Cases 09/10 (cuDNN FP4) land slightly below their cutlass-FP4 twins (02/05) and case 09 dropped 1/8 at n8. No reason to prefer cuDNN-FP4 here; `flashinfer_cutlass` FP4 stays the default.

### 5. triton attention is competitive (PROBE)
Case 07 (fi_cutlass-MoE + **triton** attention, piecewise) tied for #2 on peak (138.48) with good n1 latency (0.41 s). flashinfer attention stays the safe default, but triton attention is a legitimate fallback on this shape.

### 6. NEXTN speculative decoding is a NET LOSS on this model/shape
Both spec cases regressed hard vs the non-spec winner: case 13 (balanced) n8 peak 84.49, case 14 (low-latency) 70.56 — **~40–50% below** the 136–139 non-spec band, and *also* slower at n=1 (24.54 / 21.17 vs ~36 tok/s). Each also dropped 1/8 at n8. The MTP draft/verify overhead on M2.7 outweighs any acceptance gain at these batch sizes. **Keep `speculative_enabled: false`.**

### 7. Two MoE runners crash at startup (both expected PROBEs)
- **Case 11 — `flashinfer_trtllm` MoE:** `RuntimeError ... trtllm_batched_gemm_runner.cu:286: Error running GEMM!` during the FlashInfer AutoTuner warm-up. The failing kernel is `bmm_..._sm100f` — a Blackwell-**SM100** (datacenter) batched-GEMM kernel that has no SM121/GB10 variant. trtllm-MoE ships no sm_121 FP4 kernel.
- **Case 12 — `cutlass` (direct) MoE:** `NotImplementedError: Unsupported runner backend: MoeRunnerBackend.CUTLASS` — sglang 0.5.12.post1's `MoeRunner` registry simply does not implement the direct `cutlass` runner (only `flashinfer_cutlass`). Not a kernel issue; the backend isn't wired up.

Both crashed head + all 3 workers (1 restart each) and were correctly classified `startup_crash`. Neither is usable on this image; use `flashinfer_cutlass`.

---

## Conclusion & recommended profile change

**Winner (production shape):** `flashinfer_cutlass` MoE + `flashinfer` attention + `flashinfer_cutlass` FP4 + **piecewise CUDA graph** (matrix **case 02**) — the best throughput/latency/stability balance (n8 peak 136.0, n1 35.9 tok/s @ 0.42 s TTFT, 8/8 everywhere).

This means the current `nvidia-minimax-m2.7-nvfp4.yml` profile — which still carries the **PP=4-inherited** `moe_runner_backend: triton` + `disable_cuda_graph: true` + `disable_piecewise_cuda_graph: true` — should change to:

| Key | Current (PP=4 inherited) | Recommended (TP=4 matrix) |
|-----|--------------------------|---------------------------|
| `moe_runner_backend` | `triton` | `flashinfer_cutlass` |
| `disable_cuda_graph` | `true` (eager) | `false` |
| `disable_piecewise_cuda_graph` | `true` | `false` |
| `attention_backend` | `flashinfer` | `flashinfer` (unchanged) |
| `fp4_gemm_backend` | `flashinfer_cutlass` | `flashinfer_cutlass` (unchanged) |
| `speculative_enabled` | `false` | `false` (confirmed — spec is a net loss) |

The profile's own header already anticipated this ("switch this to flashinfer_cutlass if Block A wins") — Block A won.

> Not applied here — profile change pending explicit approval. The matrix only swept n≤8; the n8 fail in cases 03/09/13/14 (1/8 each) is worth a re-run before locking in full-CG vs piecewise, but piecewise (02) had zero fails across all 28 requests.
