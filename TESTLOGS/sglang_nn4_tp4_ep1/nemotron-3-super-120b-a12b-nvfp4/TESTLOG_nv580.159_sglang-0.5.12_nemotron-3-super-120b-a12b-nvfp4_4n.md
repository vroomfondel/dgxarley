# SGLang Test Log — NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4 (NemotronH hybrid), 4 Nodes, TP=4 EP=1, v0.5.12 (first contact)

## Environment

| Component | Value                                                                                   |
|-----------|-----------------------------------------------------------------------------------------|
| GPU       | NVIDIA GB10 (SM121/Blackwell), 128 GB per node                                          |
| Driver    | 580.159                                                                                 |
| CUDA      | 13.2 host                                                                               |
| Kernel    | 6.17.0-1018-nvidia                                                                      |
| OS        | Ubuntu 24.04 LTS (aarch64)                                                              |
| K3s       | v1.35.3+k3s1                                                                            |
| Nodes     | spark1 (head), spark2, spark3, spark4 (workers) — 1 GPU each                            |
| Image     | `scitrera/dgx-spark-sglang:0.5.12` — **UPSTREAM base**, NOT xomoxcc-sm121               |
| Model     | `nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4`                                        |
| Arch      | `NemotronHForCausalLM` (`model_type=nemotron_h`) — Mamba2 + MoE + attn hybrid           |
| Quant     | `modelopt_mixed` / `MIXED_PRECISION` (expert FFN FP4 g16, attn/latent/MTP/emb FP8/BF16) |
| NCCL      | 2.29.7+cuda13.2                                                                         |
| Transport | **RoCE** via SR-IOV VF                                                                  |

Matrix file: `kikube/matrixtest_matrices/sglang_nn4_tp4_ep1/nemotron-3-super-120b-a12b-nvfp4/nv580.159_sglang-0.5.12_nemotron-3-super-120b-a12b-nvfp4_n4_ep1.yaml`
Results: `kikube/matrixtest/2026-06-04/results/sglang_nn4_tp4_ep1/nemotron-3-super-120b-a12b-nvfp4/0.5.12/`

**First contact for this model on this cluster** — no prior baseline, no prior TESTLOG. The profile `roles/k8s_dgx/model_profiles/nvidia-nvidia-nemotron-3-super-120b-a12b-nvfp4.yml` carried first-contact defaults (explicitly marked UNVALIDATED); this matrix is their first validation.

### Image-choice note (the dominant pre-run risk, now resolved)

The matrix deliberately pinned the **upstream** `scitrera/dgx-spark-sglang:0.5.12` base rather than the active `xomoxcc:0.5.12.post1-sm121` override (decision 2026-06-03): for a brand-new arch the upstream build is the cleaner litmus, free of xomoxcc's patch-specific assumptions. The model card serves this model on a *dedicated* dev image (`lmsysorg/sglang:dev-cu13-nemotronh-nano-omni-reasoning-v3`), so it was genuinely open whether either of our images carried the `NemotronHForCausalLM` NVFP4 path + Mamba2 SM121 kernels at all.

**Litmus passed:** Case 01 loaded the arch and emitted coherent tokens. `scitrera:0.5.12` (upstream mainline) DOES carry the NemotronH NVFP4 path on SM121 — the "IMAGE CAVEAT" / per-profile `sglang_image:` override warning in the profile header can be **downgraded to resolved**. No dev-image and no per-profile override are needed for serving.

---

## Model Notes

- 120B total / 12B active **LatentMoE** hybrid. config.json: 88 layers (only ~8 
full-attention, 80 Mamba2), hidden 4096, 32 attn heads, `num_key_value_heads=2` (GQA), 512 routed + 1 shared experts, 
22 active/token,  
(tokens projected to a latent dim for expert routing), `ssm_state_size=128`, 
`mamba_num_heads=128`.
- **NoPE** — no positional embeddings; Mamba2 carries sequence order. There is no RoPE/YaRN to scale, so extending context past the config cap is literally raising the number + lifting `max_position_embeddings` via `json_model_override_args` (auto-sets `SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1`). Natively long-context-trained to 1M (LC-Phase CPT). RULER (base, 0-shot): 64K 92.3 / 128K 88.3 / 256K 84.6 / 512K 82.5 / 1M 71.0.
- Memory footprint (observed, TP=4): weights **23.31 GB/GPU** (`quant=modelopt_mixed`, MIXED_PRECISION), load ~162 s. At ctx 524288: KV cache **16.58M tokens, K 15.81 GB + V 15.81 GB**, `mem_fraction_static=0.80`, `avail_mem` ~23.6 GB after capture. Only 8/88 layers grow KV (the 80 Mamba2 layers keep constant recurrent state) → KV stays modest even at 1M.
- Reasoning ON by default in the chat template; parser `nemotron_3` (NOT vLLM/TRT-LLM's `super_v3`/`nano-v3`). Tool-call parser `qwen3_coder`.

## Why this matrix exists

First validation of the model on this cluster. Three questions:
1. **Does the arch even load on a mainline image?** (boot litmus — Case 01)
2. **How far does context scale, and at what throughput cost?** NoPE + Mamba-heavy hybrid → cheap KV. Sweep 262K → 512K → 1M (Block B).
3. **Which kernel knobs are actually viable?** CG variant (Block A), fp4_gemm backend (Block C), MoE runner (Block D).

MTP / speculative is **out of scope** (two open blockers): EAGLE MTP needs `--disable-radix-cache` which `sglang_launch.sh` doesn't yet expose, AND upstream #21138 makes NemotronH MTP reject ~all draft tokens (accept_rate ≈ 0.33 → no speedup). Profile keeps `speculative_enabled=false`.

## Closed axes (hard constraints, not swept)

- **attention = flashinfer ONLY.** triton attn is hard-asserted off on NemotronH (`apply_nemotron_h_defaults`: "does not support triton attention backend, as the first layer might not be an attention layer" — hybrid starts with Mamba, not attention). No triton-attn probe.
- **piecewise CUDA graph = off** in all serving cases (card sets `--disable-piecewise-cuda-graph`); Case 07 is a single PROBE to confirm it on this build.
- MoE FFN quant = `modelopt_fp4`; DeepGemm disabled (NVFP4 scale_fmt ≠ ue8m0).

---

## Configuration Matrix

All cases: `tp=4, pp=1, ep=1, nccl_transport=roce, attention_backend=flashinfer, kv_cache_dtype=fp8_e4m3, mem_fraction_static=0.80, disable_deep_gemm=true, max_running_requests=32, quantization=modelopt_fp4` unless noted.

CG variant encoding:
- **no-CG** : `disable_cuda_graph=true,  disable_piecewise_cuda_graph=true` (eager)
- **full-CG** : `disable_cuda_graph=false, disable_piecewise_cuda_graph=true` (profile default)
- **piecewise** : `disable_cuda_graph=false, disable_piecewise_cuda_graph=false` (PROBE only)

### Block A — CUDA-graph variant (fi_cutlass MoE, fi_cutlass-fp4, ctx262k) — Cases 01–02

| #  | moe_runner | fp4_gemm   | CG variant | ctx   | Status | n=1 tok/s | n=4 peak | n=8 peak | n=8 ok | Output |
|----|------------|------------|------------|-------|--------|----------:|---------:|---------:|--------|--------|
| 01 | fi_cutlass | fi_cutlass | no-CG (eager) | 262k | ok   | 22.10 | 80.97 | 125.57 | 8/8 | clean ✓ |
| 02 | fi_cutlass | fi_cutlass | full-CG    | 262k  | ok⁺  | 29.49 | 87.36 | 118.39 | 7/8 | **flag** ⚠ |

### Block B — context scaling (fi_cutlass MoE, fi_cutlass-fp4, full-CG) — Cases 02–04

| #  | ctx       | json_override                         | Status | n=1 tok/s | n=4 peak |   n=8 peak | n=8 ok | Output     |
|----|-----------|---------------------------------------|--------|----------:|---------:|-----------:|--------|------------|
| 02 | 262144    | `{}`                                  | ok⁺    |     29.49 |    87.36 |     118.39 | 7/8    | **flag** ⚠ |
| 03 | 524288 🏆 | `{"max_position_embeddings":524288}`  | ok     |     29.39 |    88.72 | **135.06** | 8/8    | clean ✓    |
| 04 | 1048576   | `{"max_position_embeddings":1048576}` | ok     |     29.51 |    87.95 |     134.60 | 8/8    | clean ✓    |

### Block C — fp4_gemm backend delta (fi_cutlass MoE, full-CG, ctx262k) — Case 05

| #  | fp4_gemm | Status      | Root cause                                                                                            |
|----|----------|-------------|-------------------------------------------------------------------------------------------------------|
| 05 | fi_cudnn | **crash S** | `RuntimeError: cuDNN is not available` during CG capture — `nvidia-cudnn-cu12` not in this base image |

### Block D — MoE runner PROBE (fi-attn, fi_cutlass-fp4, full-CG, ctx262k) — Case 06

| #  | moe_runner | Status      | Root cause                                                                                            |
|----|------------|-------------|-------------------------------------------------------------------------------------------------------|
| 06 | triton     | **crash S** | shape assert in `cutlass_moe_fp4` — triton runner is ignored on the NVFP4 modelopt path; see Findings |

### Block E — piecewise-CG PROBE (fi_cutlass MoE, fi_cutlass-fp4, ctx262k) — Case 07

| #  | CG variant | Status | n=1 tok/s | n=4 peak | n=8 peak | n=8 ok | Output     |
|----|------------|--------|----------:|---------:|---------:|--------|------------|
| 07 | piecewise  | ok⁺    |     29.42 |    87.01 |   116.08 | 7/8    | **flag** ⚠ |

### Column legend

| Column     | Description                                                                                                     |
|------------|-----------------------------------------------------------------------------------------------------------------|
| moe_runner | `moe_runner_backend` — `fi_cutlass` = `flashinfer_cutlass`, `triton`                                            |
| fp4_gemm   | `fp4_gemm_backend` — `fi_cutlass` = `flashinfer_cutlass`, `fi_cudnn` = `flashinfer_cudnn`                       |
| n=N peak   | **peak** throughput = Σ per-request tok/s over the *successful* requests (NOT aggregate total_tokens/wall_time) |
| crash S    | `startup_crash` — head/worker restart during CUDA-graph capture; never reaches inference                        |
| ok⁺ / flag | finished, but 1/8 requests aborted by the server-side **repetition** detector (out=0) — see Findings #6         |

---

## Detailed `ok` results (n=8)

| #  | Config                  |   n=1 | n=4 peak |   n=8 peak | n=8 avg/req | n=8 ok | Finish reasons   | n=8 TTR_min | avg_ttft | Output     |
|----|-------------------------|------:|---------:|-----------:|------------:|--------|------------------|------------:|---------:|------------|
| 01 | eager, ctx262k          | 22.10 |    80.97 |     125.57 |       15.70 | 8/8    | length×6, stop×2 |       0.617 |    1.568 | clean ✓    |
| 02 | full-CG, ctx262k        | 29.49 |    87.36 |     118.39 |       16.91 | 7/8    | length×5, stop×2 |       0.606 |    0.776 | **flag** ⚠ |
| 03 | full-CG, **ctx524k** 🏆 | 29.39 |    88.72 | **135.06** |       16.88 | 8/8    | length×6, stop×2 |       0.648 |    0.695 | clean ✓    |
| 04 | full-CG, ctx1M          | 29.51 |    87.95 |     134.60 |       16.82 | 8/8    | length×6, stop×2 |       0.647 |    0.696 | clean ✓    |
| 07 | piecewise, ctx262k      | 29.42 |    87.01 |     116.08 |       16.58 | 7/8    | length×5, stop×2 |       0.523 |    0.758 | **flag** ⚠ |

---

## Crash details

**Case 05 — `fp4_gemm_backend: flashinfer_cudnn` (startup_crash).** During CG capture:
```
flashinfer/gemm/gemm_base.py: _check_cudnn_availability →
  RuntimeError: cuDNN is not available. Please install cuDNN to use FP8 GEMM functions.
  pip install nvidia-cudnn-cu12 nvidia-cudnn-frontend
Exception: Capture cuda graph failed: cuDNN is not available.
```
Same signature as the Qwen3.6-35B-NVFP4 Block-A cuDNN cases: the upstream `scitrera:0.5.12` base does not ship the Python `nvidia-cudnn-cu12` wheel. fi_cudnn FP4 GEMM needs the cuDNN-layer image (`scripts/build_cudnn_image.sh` adds it on top), not this base. Independent of model — purely an image-packaging gap.

**Case 06 — `moe_runner_backend: triton` PROBE (startup_crash).** Despite `--moe-runner-backend triton`, the launch crashes inside the **cutlass** FP4 MoE path during CG capture:
```
fused_moe_triton/layer.py:1093 run_moe_core →
quantization/modelopt_quant.py:2156 apply → cutlass_moe_fp4(...)
moe/cutlass_moe.py:427:  nx2_w1 == params.intermediate_size_per_partition * 2
AssertionError: mismatch in expected `n`
```
The `triton` runner flag is effectively **ignored** for this NVFP4 model: `ModelOptFp4` always dispatches the MoE FFN through `cutlass_moe_fp4`, and the LatentMoE FFN shape (`moe_latent_size=1024`, expert-intermediate from the 512-expert layout) trips a hard shape assertion in the cutlass FP4 MoE kernel under the triton-runner config. Net: **triton MoE is not a usable runner on Nemotron-3-Super-NVFP4** — `flashinfer_cutlass` is the only viable MoE path.

---

## Findings

1. **Boot litmus passed on the upstream image.** `scitrera/dgx-spark-sglang:0.5.12` (mainline, no xomoxcc patches) carries the `NemotronHForCausalLM` NVFP4 path + Mamba2 kernels on SM121. The profile's "IMAGE CAVEAT" requiring a dedicated dev image or a per-profile `sglang_image:` override is **resolved** — neither is needed.

2. **`flashinfer_cutlass` is the only viable MoE runner; `flashinfer_cutlass` is the only viable fp4_gemm on this image.** 5/7 cases on the `fi_cutlass`-MoE + `fi_cutlass`-fp4 shape work; both deviations crash at startup (Case 05 fi_cudnn = image gap, Case 06 triton-MoE = kernel shape assert). The first-contact profile defaults were correct on both axes.

3. **Context scales to 1M at essentially zero throughput cost — the headline result.** Cases 02/03/04 (262K/512K/1M, full-CG) land at n=8 peak 118.39 / 135.06 / 134.60 and n=1 29.49 / 29.39 / 29.51. The 262K case (02) reads lower only because it lost a request to the repetition detector (7/8 → peak under-counts by ~1/8; normalized ≈ 135). NoPE + 80/88 Mamba layers mean the KV cache barely grows with context, so **512K and 1M are free relative to 262K**. RULER quality (82.5 @ 512K, 71.0 @ 1M) is the only reason to prefer a shorter cap, not performance.

4. **Winner = Case 03 (full-CG, ctx524k):** n=8 peak **135.06 tok/s**, 8/8 clean, TTR_min 0.648. This is exactly the active profile shape (`flashinfer_cutlass` MoE + fi-attn + fi_cutlass-fp4 + full-CG + `context_length=524288`) — **no profile change required**; first-contact defaults matched the eventual winner. Case 04 (1M) is a statistical tie (135.06 vs 134.60) and is the better pick if >512K context is ever needed.

5. **Eager (Case 01) is cheap on batch but expensive on single-stream.** n=8 eager 125.57 is only ~7% below full-CG 135 (8/8-normalized), but n=1 eager 22.10 vs full-CG 29.4 is a **−25%** single-stream hit, and eager avg_ttft 1.568 s vs ~0.70 s under CG. Full-CG wins overall; eager is only a boot-safety fallback.

6. **Repetition-detector aborts in 2 of 5 working cases (flag, not a hard failure).** Cases 02 and 07 each had exactly 1/8 n=8 request aborted with `status=repetition, output_tokens=0` (server-side repetition detector, not a crash). Cases 01/03/04 were clean 8/8. Visible-text spot-check of the winner (Case 03) shows diverse coherent content (REST API design, architecture trade-offs, network engineering, CAP/PACELC) — **no `!`-token collapse, no word-salad**. The repetition flag looks like an occasional single-request detector trip rather than a systemic quality regression, but it recurred on two separate configs — worth a re-run of Case 03 at higher concurrency to confirm it stays clean.

7. **piecewise CG (Case 07) gives nothing here.** n=8 peak 116.08 (7/8) vs full-CG 02's 118.39 (7/8) — within noise, and it also tripped the repetition detector. Confirms the card's `--disable-piecewise-cuda-graph` default: no benefit on this Mamba/attn hybrid. Keep piecewise off.

8. **No FP8 sibling to A/B against.** This model ships NVFP4-only on the cluster; numbers stand alone. For cross-model context, 135 tok/s n=8 on a 120B/12B-active hybrid at TP=4 is in line with the cluster's other large NVFP4 MoE models given the 12B active path.

---

## Production recommendation

Keep the active profile shape — it already matches the winner:

```yaml
# roles/k8s_dgx/model_profiles/nvidia-nvidia-nemotron-3-super-120b-a12b-nvfp4.yml
moe_runner_backend: "flashinfer_cutlass"   # ONLY viable runner (triton crashes — Finding #2/Case 06)
attention_backend: "flashinfer"            # triton hard-asserted off on NemotronH
fp4_gemm_backend: "flashinfer_cutlass"     # fi_cudnn needs the cudnn-layer image (Case 05)
disable_cuda_graph: false                  # full-CG; eager costs −25% n=1 (Finding #5)
disable_piecewise_cuda_graph: true         # piecewise gives nothing (Finding #7)
context_length: 524288                      # 512K free vs 262K; 1M also viable (Finding #3)
mem_fraction_static: "0.80"
```

Profile-header edits warranted after this run:
- Downgrade the `>>> IMAGE CAVEAT <<<` block to **RESOLVED** — boots on mainline `scitrera:0.5.12`, no override needed (Finding #1).
- Note that `moe_runner_backend` MUST stay `flashinfer_cutlass` — triton is not optional-alt, it crashes (Finding #2).

## Action items / follow-ups

- [ ] Re-run Case 03 (winner) once at higher concurrency to confirm the repetition-detector trip (Finding #6) is sporadic, not load-correlated.
- [ ] If fi_cudnn FP4 GEMM is ever wanted for this model, re-test Case 05 on the cuDNN-layer image (`scripts/build_cudnn_image.sh`) — but fi_cutlass already wins on the Qwen3.6-NVFP4 comparison, so low priority.
- [ ] Update the profile header per the two edits above.
- [ ] Revisit MTP once `--disable-radix-cache` is exposed in `sglang_launch.sh` AND upstream #21138 (NemotronH MTP accept-rate ≈ 0) closes.
