# SGLang Test Log — Qwen3.5 397B-A17B NVFP4, 4 Nodes, TP=4 EP=1, v0.5.12-cudnn

## Environment

| Component | Value                                              |
|-----------|----------------------------------------------------|
| GPU       | NVIDIA GB10 (SM121/Blackwell), 128 GB per node     |
| Driver    | 580.159                                            |
| Kernel    | 6.17.0-1021-nvidia                                 |
| OS        | Ubuntu 24.04.4 LTS (aarch64)                       |
| K3s       | v1.36.1+k3s1                                       |
| Nodes     | spark1, spark2, spark3, spark4 (1 GPU each)        |
| Image     | `xomoxcc/dgx-spark-sglang:0.5.12-cudnn`            |
| Model     | `nvidia/Qwen3.5-397B-A17B-NVFP4`                   |
| Transport | **RoCE** via SR-IOV VF                             |

> CUDA / NCCL / torch versions not captured (pods scaled to 0 after the run); see the image build (`scripts/patches/sglang-0.5.12-cudnn.Dockerfile`).
> Raw matrix: `kikube/results/sglang_nn4_tp4_ep1/qwen-3.5-397b-a17b-nvfp4/0.5.12-cudnn/MATRIX_SUMMARY_nv580.159_sglang-0.5.12-cudnn_qwen3.5-397b-nvfp4_4n_1pp_4tp_ep1.json` (run 2026-06-19).

---

## Model Notes

- 397B total / 17B active MoE (512 experts, top-10, softmax routing), NVFP4 quantized (~234 GB).
- Hybrid attention: 15 full GQA layers + 45 linear attention layers (every 4th layer is full attention). 60 layers total.
- 1 shared expert + 512 routed experts per MoE layer. Multimodal (text+image+video).
- Has MTP head (1 layer) for speculative decoding (NEXTN).
- `num_attention_heads=32, num_key_value_heads=2` — TP=4 per model card.
- NVFP4: only routed expert MoE FFN weights are FP4; attention, shared experts, vision encoder, lm_head, and MTP layer remain BF16.
- Parsers: `reasoning_parser=qwen3`, `tool_call_parser=qwen3_coder` (XML-style tool calls — see profile).

## Why this matrix (vs the 0.5.10 baseline `TESTLOG_nv580.142_sglang-0.5.10`)

- **New image `0.5.12-cudnn`, new driver 580.159, new kernel 6.17.0-1021-nvidia.** Forced by 0.5.12 dropping the standalone `cutlass` MoE runner (load_model raised `NotImplementedError: Unsupported runner backend: MoeRunnerBackend.CUTLASS`), which was the 0.5.10 winner (Test 38).
- **`flashinfer_cutlass` MoE is now FIXED at EP=1.** In 0.5.10 it was heavily broken at EP=1 (8/12 configs failed); here all 12 fi_cutlass-moe configs (13–24) are STABLE with 0 failed requests, and it is the **best** backend.
- **New probe dimension `flashinfer_trtllm` MoE** (25–27): startup-crashes on SM121 (pod restart loops) — not viable.
- **MTP sweep** over `speculative_num_steps`/`num_draft_tokens` (28–31) to re-tune for the new image.

---

## Configuration Matrix

All tests: `tp=4, pp=1, ep=1, nccl_transport=roce, quantization=modelopt_fp4, kv_cache_dtype=fp8_e4m3, mem_fraction_static=0.75, disable_deep_gemm=true, cuda_graph_max_bs=16, mamba_scheduler_strategy=extra_buffer, SGLANG_ENABLE_SPEC_V2=1` unless noted. n=X = aggregate (sum per-request) tok/s at concurrency X.

| #  | moe_runner | attn   | fp4_gemm   | cg  | mtp     | Status       | n=1      | n=4      | n=8       | n=16      |
|----|------------|--------|------------|-----|---------|--------------|----------|----------|-----------|-----------|
| 1  | triton     | fi     | fi_cutlass | on  | —       | STABLE       | 21.4     | 65.9     | 98.0      | 135.9     |
| 2  | triton     | fi     | fi_cutlass | off | —       | STABLE       | 13.6     | 61.8     | 94.5      | 136.5     |
| 3  | triton     | fi     | fi_cutlass | pw  | —       | STABLE       | 21.4     | 66.6     | 97.0      | 137.1     |
| 4  | triton     | triton | fi_cutlass | on  | —       | STABLE       | 20.8     | 65.7     | 97.5      | 137.4     |
| 5  | triton     | triton | fi_cutlass | off | —       | STABLE       | 14.2     | 62.6     | 96.4      | 137.4     |
| 6  | triton     | triton | fi_cutlass | pw  | —       | STABLE       | 21.0     | 63.1     | 97.9      | 137.1     |
| 7  | triton     | fi     | fi_cudnn   | on  | —       | STABLE       | 21.8     | 65.8     | 98.3      | 136.9     |
| 8  | triton     | fi     | fi_cudnn   | off | —       | STABLE       | 13.8     | 63.3     | 95.4      | 136.9     |
| 9  | triton     | fi     | fi_cudnn   | pw  | —       | STABLE       | 20.2     | 65.6     | 98.7      | 132.9     |
| 10 | triton     | triton | fi_cudnn   | on  | —       | STABLE       | 20.4     | 64.5     | 97.7      | 137.1     |
| 11 | triton     | triton | fi_cudnn   | off | —       | STABLE       | 13.8     | 63.0     | 96.3      | 137.4     |
| 12 | triton     | triton | fi_cudnn   | pw  | —       | STABLE       | 21.6     | 66.2     | 100.1     | 138.8     |
| 13 | fi_cutlass | fi     | fi_cutlass | on  | —       | STABLE       | 23.0     | 70.2     | 101.6     | 144.3     |
| 14 | fi_cutlass | fi     | fi_cutlass | off | —       | STABLE       | 20.3     | 66.4     | 101.5     | 144.4     |
| 15 | fi_cutlass | fi     | fi_cutlass | pw  | —       | STABLE       | 22.2     | 68.5     | 105.6     | 145.3     |
| 16 | fi_cutlass | triton | fi_cutlass | on  | —       | STABLE       | 22.0     | 69.5     | 105.0     | 146.3     |
| 17 | fi_cutlass | triton | fi_cutlass | off | —       | STABLE       | 20.0     | 66.9     | 101.9     | 146.1     |
| 18 | fi_cutlass | triton | fi_cutlass | pw  | —       | STABLE       | 23.1     | 68.0     | 104.2     | 147.7     |
| 19 | fi_cutlass | fi     | fi_cudnn   | on  | —       | STABLE       | 23.4     | 70.8     | 104.1     | 147.8     |
| 20 | fi_cutlass | fi     | fi_cudnn   | off | —       | STABLE       | 20.9     | 65.4     | 101.6     | 144.5     |
| 21 | fi_cutlass | fi     | fi_cudnn   | pw  | —       | STABLE       | 21.9     | 70.8     | 105.5     | 148.5     |
| 22 | fi_cutlass | triton | fi_cudnn   | on  | —       | STABLE       | 23.1     | 70.6     | 103.8     | 147.2     |
| 23 | fi_cutlass | triton | fi_cudnn   | off | —       | STABLE       | 20.9     | 67.7     | 101.0     | 146.0     |
| 24 | fi_cutlass | triton | fi_cudnn   | pw  | —       | STABLE       | 22.7     | 70.1     | **107.4** | 148.7     |
| 25 | fi_trtllm  | fi     | fi_cutlass | on  | —       | CRASH        | —        | —        | —         | —         |
| 26 | fi_trtllm  | fi     | fi_cutlass | pw  | —       | CRASH        | —        | —        | —         | —         |
| 27 | fi_trtllm  | triton | fi_cutlass | on  | —       | CRASH        | —        | —        | —         | —         |
| 28 | fi_cutlass | triton | fi_cutlass | on  | s1/d2   | STABLE       | 31.3     | 82.1     | 112.0     | 149.3     |
| 29 | fi_cutlass | triton | fi_cutlass | on  | s3/d4   | **STABLE ★** | **39.1** | **89.7** | **120.7** | **163.6** |
| 30 | fi_cutlass | triton | fi_cutlass | on  | s5/d5   | STABLE       | 36.6     | 81.8     | 112.2     | 155.3     |
| 31 | fi_cutlass | triton | fi_cutlass | on  | s5/d7   | STABLE       | 38.2     | 81.8     | 112.2     | 154.2     |
| 32 | triton     | triton | fi_cutlass | on  | s3/d4   | FAIL‡        | 36.7     | 87.2     | —         | —         |

- **cg**: `on` = CUDA graphs + fixed-BS; `off` = eager (`disable_cuda_graph=true`); `pw` = piecewise graphs (`disable_piecewise_cuda_graph=false`).
- **mtp**: MTP/NEXTN speculative `steps/draft` (e.g. `s3/d4` = `speculative_num_steps=3, speculative_num_draft_tokens=4`); `—` = MTP off.
- **‡ FAIL (32)**: triton-MoE + MTP — ran n=1/n=4 then 24 failed requests at n≥8 (unstable). fi_cutlass-MoE + MTP (29) is stable across all concurrency.

---

## Final Results

### Winner: Test 29 — fi_cutlass MoE + triton attn + fi_cutlass FP4 + CUDA graphs + MTP s3/d4

This is the config currently pinned in `roles/k8s_dgx/model_profiles/nvidia-qwen3.5-397b-a17b-nvfp4.yml`.

| Concurrency | Peak tok/s | avg TTFT |
|-------------|------------|----------|
| n=1         | **39.1**   | 0.82s    |
| n=4         | **89.7**   | 1.14s    |
| n=8         | **120.7**  | 1.33s    |
| n=16        | **163.6**  | 1.21s    |

### Top non-MTP configs by n=8 / n=16

| Rank | #  | MoE        | Attn   | FP4      | CG  | n=8       | n=16      |
|------|----|------------|--------|----------|-----|-----------|-----------|
| 1    | 24 | fi_cutlass | triton | fi_cudnn | pw  | **107.4** | 148.7     |
| 2    | 21 | fi_cutlass | fi     | fi_cudnn | pw  | 105.5     | **148.5** |
| 3    | 15 | fi_cutlass | fi     | fi_cutlass | pw | 105.6    | 145.3     |
| 4    | 18 | fi_cutlass | triton | fi_cutlass | pw | 104.2    | 147.7     |

### Observations

- **MTP is the single biggest lever.** Best non-MTP n=1 ≈ 23 tok/s; MTP s3/d4 (29) → 39.1 tok/s (**+70%**). MTP gain fades with concurrency (n=16: 163.6 vs ~148 non-MTP, +10%). MTP sweet spot is **steps=3/draft=4**; steps=1 underspeculates (28: 31.3), steps=5 overspeculates (30/31: 36–38).
- **`flashinfer_cutlass` MoE > `triton` MoE across the board.** fi_cutlass (13–24) beats triton (1–12) by ~5–8% at every concurrency (n=16 ≈ 145–149 vs 133–139). **This is the opposite of 0.5.10**, where fi_cutlass was broken at EP=1 and triton/cutlass-direct won — 0.5.12 fixed the fi_cutlass EP=1 dispatch.
- **`flashinfer_trtllm` MoE (25–27) startup-crashes** on SM121 (pod restart loops) — not usable.
- **CUDA graphs ON gives ~40–55% n=1 speedup** (e.g. 13: 23.0 on vs 14: 20.3 off; triton even larger: 21.4 vs 13.6). Gap closes by n=4. Piecewise ≈ fixed within noise (sometimes marginally ahead at n=8).
- **fp4_gemm cutlass vs cudnn:** a wash. cudnn edges slightly ahead at high concurrency (24 → n=8 107.4).
- **attn triton vs flashinfer:** near-identical.

### Comparison with the 0.5.10 baseline (Test 38, cutlass-direct MoE, image 0.5.10)

| Config                                              | n=1      | n=4      | n=8       | n=16      |
|-----------------------------------------------------|----------|----------|-----------|-----------|
| 0.5.10 Test 38 — cutlass-direct MoE + MTP s3/d4     | 40.0     | 84.3     | 110.9     | —         |
| **0.5.12-cudnn Test 29 — fi_cutlass MoE + MTP s3/d4** | **39.1** | **89.7** | **120.7** | **163.6** |
| Δ                                                   | −2%      | +6%      | +9%       | new       |

The forced switch off the (now-removed) cutlass-direct MoE costs nothing: 0.5.12-cudnn `flashinfer_cutlass` **matches** the old n=1 (within noise) and **beats** it at n≥4. The earlier ad-hoc n=1=31.7 regression was the plain (non-cudnn) 0.5.12 image with `cuda_graph_max_bs=8`; on 0.5.12-cudnn with `cuda_graph_max_bs=16` + MTP s3/d4 the gap is gone.

**Profile is the matrix winner.** All 17 winner-config params already matched the profile (verified by diff); the only change applied was pinning `sglang_image: "xomoxcc/dgx-spark-sglang:0.5.12-cudnn"` so the deploy uses the validated build instead of the default plain 0.5.12.
