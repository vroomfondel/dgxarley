# SGLang Test Log — Qwen3-235B-A22B NVFP4, 4 Nodes, v0.5.10

## Environment

| Component | Value |
|-----------|-------|
| GPU | NVIDIA GB10 (SM121/Blackwell), 128 GB per node |
| Driver | 580.142 |
| CUDA | 13.0 |
| Kernel | 6.17.0-1014-nvidia |
| OS | Ubuntu 24.04.4 LTS (aarch64) |
| K3s | v1.35.3+k3s1 |
| Nodes | spark1, spark2, spark3, spark4 (1 GPU each) |
| Image | `scitrera/dgx-spark-sglang:0.5.10` |
| Model | `nvidia/Qwen3-235B-A22B-NVFP4` |

First test series for this model on this setup — no prior version to compare against.

---

## Configuration Matrix

All tests use: `tp=4, pp=1, ep=4, quantization=modelopt_fp4, kv_cache_dtype=fp8_e4m3, mem_fraction_static=0.80, disable_deep_gemm=true, context_length=131072 (YaRN), max_running_requests=32, schedule_policy=lpm, watchdog_timeout=3600, dist_timeout=1800` unless noted.

**YaRN context extension enabled** via `--json-model-override-args '{"rope_scaling":{"rope_type":"yarn","factor":4.0,"original_max_position_embeddings":32768,"rope_theta":1000000.0}}'` + `SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1`. Without both, SGLang's `_derive_context_length` hard-rejects `context_length > 40960` (Qwen3 native) in v0.5.10+. `rope_theta` must be duplicated inside the `rope_scaling` dict — SGLang's `get_rope_config` (`hf_transformers_utils.py:162`) reads it from there, not from the config top level.

36 tests: systematic sweep of `moe_runner × attention × fp4_gemm × cuda_graph_variant`. pp_async always 0 (PP=1).

| # | nccl_transport | moe_runner | attention | fp4_gemm | dis_cuda_graph | dis_piecewise | pp_async | cuda_graph_max_bs | Stability | 1∥ tok/s | 4∥ tok/s | 8∥ tok/s |
|---|----------------|------------|-----------|----------|----------------|---------------|----------|-------------------|-----------|---------|---------|---------|
| 1 | socket | triton | flashinfer | fi_cutlass | false | true | 0 | 8 | **startup_crash** | — | — | — |
| 2 | socket | triton | flashinfer | fi_cutlass | true | true | 0 | — | **infer_error** | — | — | — |
| 3 | socket | triton | flashinfer | fi_cutlass | false | false | 0 | 8 | **startup_crash** | — | — | — |
| 4 | socket | triton | triton | fi_cutlass | false | true | 0 | 8 | **startup_crash** | — | — | — |
| 5 | socket | triton | triton | fi_cutlass | true | true | 0 | — | **infer_error** | — | — | — |
| 6 | socket | triton | triton | fi_cutlass | false | false | 0 | 8 | **startup_crash** | — | — | — |
| 7 | socket | triton | flashinfer | fi_cudnn | false | true | 0 | 8 | **startup_crash** | — | — | — |
| 8 | socket | triton | flashinfer | fi_cudnn | true | true | 0 | — | **startup_crash** | — | — | — |
| 9 | socket | triton | flashinfer | fi_cudnn | false | false | 0 | 8 | **startup_crash** | — | — | — |
| 10 | socket | triton | triton | fi_cudnn | false | true | 0 | 8 | **startup_crash** | — | — | — |
| 11 | socket | triton | triton | fi_cudnn | true | true | 0 | — | **startup_crash** | — | — | — |
| 12 | socket | triton | triton | fi_cudnn | false | false | 0 | 8 | **startup_crash** | — | — | — |
| 13 | socket | fi_cutlass | flashinfer | fi_cutlass | false | true | 0 | 8 | **STABLE** | 12.28 | 28.94 | 40.11 |
| 14 | socket | fi_cutlass | flashinfer | fi_cutlass | true | true | 0 | — | **STABLE** | 11.79 | 29.47 | 40.65 |
| 15 | socket | fi_cutlass | flashinfer | fi_cutlass | false | false | 0 | 8 | **startup_crash** | — | — | — |
| 16 | socket | fi_cutlass | triton | fi_cutlass | false | true | 0 | 8 | **STABLE** | **12.54** | 30.40 | 41.36 |
| 17 | socket | fi_cutlass | triton | fi_cutlass | true | true | 0 | — | **STABLE ★** | 11.28 | **34.60** | **42.70** |
| 18 | socket | fi_cutlass | triton | fi_cutlass | false | false | 0 | 8 | **startup_crash** | — | — | — |
| 19 | socket | fi_cutlass | flashinfer | fi_cudnn | false | true | 0 | 8 | **startup_crash** | — | — | — |
| 20 | socket | fi_cutlass | flashinfer | fi_cudnn | true | true | 0 | — | **startup_crash** | — | — | — |
| 21 | socket | fi_cutlass | flashinfer | fi_cudnn | false | false | 0 | 8 | **startup_crash** | — | — | — |
| 22 | socket | fi_cutlass | triton | fi_cudnn | false | true | 0 | 8 | **startup_crash** | — | — | — |
| 23 | socket | fi_cutlass | triton | fi_cudnn | true | true | 0 | — | **startup_crash** | — | — | — |
| 24 | socket | fi_cutlass | triton | fi_cudnn | false | false | 0 | 8 | **startup_crash** | — | — | — |
| 25 | socket | cutlass | flashinfer | fi_cutlass | false | true | 0 | 8 | **startup_crash** | — | — | — |
| 26 | socket | cutlass | flashinfer | fi_cutlass | true | true | 0 | — | **infer_error** | — | — | — |
| 27 | socket | cutlass | flashinfer | fi_cutlass | false | false | 0 | 8 | **startup_crash** | — | — | — |
| 28 | socket | cutlass | triton | fi_cutlass | false | true | 0 | 8 | **startup_crash** | — | — | — |
| 29 | socket | cutlass | triton | fi_cutlass | true | true | 0 | — | **infer_error** | — | — | — |
| 30 | socket | cutlass | triton | fi_cutlass | false | false | 0 | 8 | **startup_crash** | — | — | — |
| 31 | socket | cutlass | flashinfer | fi_cudnn | false | true | 0 | 8 | **startup_crash** | — | — | — |
| 32 | socket | cutlass | flashinfer | fi_cudnn | true | true | 0 | — | **startup_crash** | — | — | — |
| 33 | socket | cutlass | flashinfer | fi_cudnn | false | false | 0 | 8 | **startup_crash** | — | — | — |
| 34 | socket | cutlass | triton | fi_cudnn | false | true | 0 | 8 | **startup_crash** | — | — | — |
| 35 | socket | cutlass | triton | fi_cudnn | true | true | 0 | — | **startup_crash** | — | — | — |
| 36 | socket | cutlass | triton | fi_cudnn | false | false | 0 | 8 | **startup_crash** | — | — | — |

> ★ = Overall winner. **STABLE** = n=1, n=4, n=8 all fully pass. Throughput columns show 1∥ per-request tok/s and 4∥/8∥ peak concurrent throughput (sum of per-request tok/s).

### Column Legend

| Column | Description |
|--------|-------------|
| nccl_transport | `sglang_nccl_transport` — NCCL inter-node transport (`socket` = TCP/IP) |
| moe_runner | `moe_runner_backend` — MoE expert dispatch kernel (`fi_cutlass` = flashinfer_cutlass, `triton` = triton→cutlass_moe_fp4 fallback for NVFP4, `cutlass` = cutlass direct) |
| attention | `attention_backend` — attention kernel |
| fp4_gemm | `fp4_gemm_backend` — FP4 dense GEMM kernel (`fi_cutlass` = flashinfer_cutlass, `fi_cudnn` = flashinfer_cudnn) |
| dis_cuda_graph | `disable_cuda_graph` — true = eager, false = capture CUDA graphs |
| dis_piecewise | `disable_piecewise_cuda_graph` — true = only fixed-BS graphs, false = piecewise variable-length graphs |
| pp_async | `pp_async_batch_depth` — PP=1 here, always 0 |
| cuda_graph_max_bs | `cuda_graph_max_bs` — largest batch captured (— = N/A when graphs disabled) |
| 1∥ tok/s | 1 sequential request (= per-request tok/s) |
| 4∥ tok/s | Peak concurrent throughput at 4∥ (sum of per-request tok/s) |
| 8∥ tok/s | Peak concurrent throughput at 8∥ (sum of per-request tok/s) |

---

## Failure Patterns

### Pattern 1: `triton` MoE on NVFP4 (tests 1–6 with fi_cutlass fp4)

SGLang's `triton` MoE runner falls back internally to `cutlass_moe_fp4` for NVFP4 weights. This path is broken on SM121:
- **CUDA graph variants (1, 3, 4, 6):** startup_crash — `nvfp4_blockwise_moe.cuh:78` device-side assert during graph capture.
- **Eager variants (2, 5):** startup passes, but inference returns 0 tokens (infer_error). Server stays alive.

### Pattern 2: `fi_cudnn` fp4_gemm broken in v0.5.10 (tests 7–12, 19–24, 31–36 — ALL 18)

Every test using `fp4_gemm_backend=flashinfer_cudnn` crashes at startup, regardless of MoE runner, attention backend, or CUDA graph variant. **100% failure rate on this backend.** This matches the same v0.5.10 regression observed in GLM-4.7 NVFP4 and Qwen3.5-397B NVFP4 matrices — `flashinfer_cudnn` FP4 GEMM is completely non-functional on SM121 in v0.5.10.

### Pattern 3: `cutlass` MoE (direct) on NVFP4 (tests 25–30 with fi_cutlass fp4)

Same split as triton MoE: CUDA graph variants crash at startup, eager variants (26, 29) start but return 0 tokens. The `cutlass` direct MoE runner hits the same `nvfp4_blockwise_moe.cuh` path and is equally broken on SM121.

### Pattern 4: Piecewise CUDA graphs crash in the winning combo (tests 15, 18)

Within the working `fi_cutlass` MoE + `fi_cutlass` fp4 family, the piecewise CUDA graph variants (15, 18) crash at startup while the fixed-BS (13, 16) and eager (14, 17) variants all run. Piecewise graph capture is incompatible with this kernel family on SM121.

---

## Working Configurations (4 of 36)

All four winners share: `moe_runner_backend=flashinfer_cutlass`, `fp4_gemm_backend=flashinfer_cutlass`, non-piecewise CUDA graphs (either fixed-BS or disabled). Attention backend and CUDA-graph-vs-eager are orthogonal within this group.

| Test | attn | CUDA graphs | n=1 tok/s | n=1 TTFT | n=4 peak | n=4 TTFT | n=8 peak | n=8 TTFT |
|------|------|-------------|-----------|----------|----------|----------|----------|----------|
| 13 | flashinfer | fixed-BS | 12.28 | 5.60s | 28.94 | 1.96s | 40.11 | 2.08s |
| 14 | flashinfer | eager | 11.79 | 5.75s | 29.47 | 2.28s | 40.65 | 2.26s |
| 16 | triton | fixed-BS | **12.54** | **1.01s** | 30.40 | 2.46s | 41.36 | 1.86s |
| 17 | triton | eager | 11.28 | 10.97s | **34.60** | 2.14s | **42.70** | 2.39s |

**Test 17 ★ overall winner:** best peak at both n=4 (34.60) and n=8 (42.70). n=1 TTFT is high (~11s) but n=4/n=8 TTFT normalizes to ~2s.

**Test 16 best CUDA-graph config:** highest n=1 throughput (12.54 tok/s) and lowest TTFT (~1s). n=4/n=8 peaks slightly behind Test 17 but within ~3%.

n=1 per-request tok/s scales cleanly: 12 tok/s single → 5 tok/s each at n=8, with near-linear aggregate scaling 12 → 30 → 42 tok/s (3.4× at 8× concurrency).

---

## Summary — v0.5.10 Qwen3-235B-A22B NVFP4

| Finding | Detail |
|---------|--------|
| **Remarkably stable** | 4 fully stable configs (vs GLM-4.7's 0 on v0.5.10 TP=4 EP=4 and PP=4). Qwen3-235B is the best-behaved NVFP4 model tested so far in this setup. |
| **Only `fi_cutlass` MoE + `fi_cutlass` fp4 works** | All other MoE runners (triton, cutlass) crash on NVFP4 via the `nvfp4_blockwise_moe.cuh` path. |
| **`fi_cudnn` fp4 fully broken** | Same v0.5.10 regression as GLM-4.7, Qwen3.5-397B. 18/18 tests crash at startup. |
| **Piecewise CUDA graphs crash** | Must use fixed-BS (`disable_piecewise_cuda_graph: true`) or eager. |
| **Attention backend doesn't matter** | Both `flashinfer` and `triton` attention work with the winning MoE+FP4 combo. |
| **YaRN required for context > 40K** | Qwen3 ships `rope_scaling: null`; needs `json_model_override_args` + `SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1`. `rope_theta` must be included inside `rope_scaling` dict. |
| **No speculative decoding test** | Qwen3-235B has no MTP heads — NEXTN not applicable. |

### Recommended config

**Test 17 — fi_cutlass MoE + triton attn + fi_cutlass fp4 + eager (`disable_cuda_graph: true`, `disable_piecewise_cuda_graph: true`):**
- n=1: 11.28 tok/s (TTFT 11s — high but only first request)
- n=4: 34.60 tok/s peak (8.65 per-req)
- n=8: 42.70 tok/s peak (5.34 per-req)

Use Test 16 (same but with fixed-BS CUDA graphs) if low n=1 TTFT matters more than peak concurrent throughput — trades ~3% at n=4/n=8 for ~10× better n=1 TTFT.

### Profile key settings for `roles/k8s_dgx/defaults/main.yml`

```yaml
"nvidia/Qwen3-235B-A22B-NVFP4":
  tp_size: 4
  ep_size: 4
  context_length: 131072
  json_model_override_args: '{"rope_scaling":{"rope_type":"yarn","factor":4.0,"original_max_position_embeddings":32768,"rope_theta":1000000.0}}'
  quantization: "modelopt_fp4"
  kv_cache_dtype: "fp8_e4m3"
  mem_fraction_static: "0.80"
  moe_runner_backend: "flashinfer_cutlass"   # NOT triton — triton falls back to broken nvfp4_blockwise_moe
  attention_backend: "triton"                # or flashinfer — both work
  fp4_gemm_backend: "flashinfer_cutlass"     # NOT flashinfer_cudnn — broken in v0.5.10
  disable_cuda_graph: true                   # Test 17 winner; false+piecewise=true also works (Test 16)
  disable_piecewise_cuda_graph: true         # piecewise graphs crash
  disable_deep_gemm: true
  enable_eplb: false
  jit_max_jobs: 4
  num_experts: 128
```
