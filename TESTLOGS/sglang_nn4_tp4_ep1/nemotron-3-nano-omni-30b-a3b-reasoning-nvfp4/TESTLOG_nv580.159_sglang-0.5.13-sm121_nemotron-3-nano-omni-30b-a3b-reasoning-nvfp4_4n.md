# SGLang Test Log — Nemotron-3 Nano Omni 30B-A3B-Reasoning-NVFP4 (Omni MoE/Mamba hybrid), 4 Nodes, TP=4 EP=1, v0.5.13-sm121 (first contact)

## Environment

| Component | Value                                                                       |
|-----------|-----------------------------------------------------------------------------|
| GPU       | NVIDIA GB10 (SM121/Blackwell-Consumer), 128 GB unified per node             |
| Driver    | 580.159                                                                     |
| Kernel    | 6.17.0-1021-nvidia                                                          |
| OS        | Ubuntu 24.04.4 LTS (aarch64)                                                |
| K3s       | v1.36.1+k3s1                                                                |
| Nodes     | spark1 (head/rank0), spark2, spark3, spark4 (1 GB10 each)                   |
| Image     | `xomoxcc/dgx-spark-sglang:0.5.13-sm121` (PROFILE-PINNED)                    |
| Model     | `nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-NVFP4` (snapshot dc5f0b0…)   |
| Transport | **RoCE** via SR-IOV VF                                                      |
| Parallel  | tp=4, pp=1, ep=1                                                            |

Matrix file: `kikube/matrixtest_matrices/sglang_nn4_tp4_ep1/nemotron-3-nano-omni-30b-a3b-reasoning-nvfp4/nv580.159_sglang-0.5.13-sm121_nemotron-3-nano-omni-30b-a3b-reasoning-nvfp4_n4_ep1.yaml`
Profile: `roles/k8s_dgx/model_profiles/nvidia-nemotron-3-nano-omni-30b-a3b-reasoning-nvfp4.yml`

**First contact for this model** — no prior baseline. Architecture sibling for A/B reference is the validated **Super-120B** NemotronH:
- `TESTLOGS/sglang_nn4_tp4_ep4/nemotron-3-super-120b-a12b-nvfp4/TESTLOG_nv580.159_sglang-0.5.13-mtp_nemotron-3-super-120b-a12b-nvfp4_4n.md` (same hybrid family; Super HAS an MTP head, this one does NOT — see Model Notes).

Why the SM121 build: the stock `scitrera/dgx-spark-sglang` image device-asserts on the triton/cutlass NVFP4 MoE path on SM121 (see `CUTLASS_NVFP4_SM121_PRD.md`); the `xomoxcc/…:0.5.13-sm121` build carries both the SM121 NVFP4-MoE fix AND the Omni arch class. ⚠️ **Audio gap:** `librosa` is NOT in this image — the Parakeet audio path would fail at runtime; add it to the recipe before any audio test. This matrix is TEXT-ONLY.

---

## Model Notes

- OMNI-MODAL wrapper `NemotronH_Nano_Omni_Reasoning_V3` around a NemotronH text core (`NemotronHForCausalLM`, `model_type=nemotron_h`). **Mamba2 + MoE + attention HYBRID.**
- Text core: 52 layers, hidden 2688, 32 attn heads, num_kv_heads 2 (GQA), 128 routed + 1 shared experts, 6 active/token, expert_intermediate 1856, native `max_position_embeddings=262144`. NoPE (Mamba2 carries order).
- NVFP4 modelopt-MIXED (~21 GB weights): routed expert FFN FP4 (E2M1, per-block FP8 E4M3 scales, group_size 16); Mamba in/out_proj + shared experts + attn o_proj FP8; vision (C-RADIOv2-H) + audio (Parakeet) encoders stay BF16.
- Reasoning post-train (`<think>`), `enable_thinking` ON by default; toggle per-request via `extra_body={"chat_template_kwargs":{"enable_thinking":false}}`.
- **NO MTP / speculative decoding.** VERIFIED 2026-06-25 three ways: (1) the served `config.json` has no `num_nextn_predict_layers`/nextn/mtp/draft key anywhere (top-level or nested `llm_config`); (2) the Nano Omni paper (arXiv 2604.24954) never mentions MTP/speculative/draft; (3) MTP is a Nemotron-3 family technique but only the **Super** ships a usable head. No native draft, no external draft → `speculative_enabled=false` everywhere. (Generic web summaries claiming "native MTP" conflate the family/Super discussion — not true for Nano/Omni.)
- Hybrid-Mamba concurrency: `max_running_requests` is clamped by the Mamba state-cache pool (`MambaRadixCache`), NOT by KV/cuda_graph — same as the Super. Without MTP there's no extra_buffer doubling, so the ratio is smaller.

## Closed axes (NOT swept — hard constraints)

- **attention = flashinfer ONLY.** triton attn is HARD-ASSERTED off on NemotronH (`apply_nemotron_h_defaults`: first layer may be Mamba, not attention). Mamba2 SSM layers use their own kernels regardless.
- **moe_runner = flashinfer_cutlass ONLY.** triton is NOT an alternative on the modelopt NVFP4 path (falls through to `cutlass_moe_fp4`).
- **piecewise CUDA graph OFF everywhere** (`--disable-piecewise-cuda-graph`; the Mamba2/attn hybrid doesn't piecewise-capture cleanly; Super already showed this is the right default).
- quant = `modelopt_fp4`; DeepGemm disabled (NVFP4 scale_fmt != ue8m0); kv_cache_dtype = `fp8_e4m3`.
- tp_size=4, ep_size=1 (128 experts % 4 == 0 but EP=1 is the seed).
- context_length = 262144 NATIVE, NO override (no published RULER curve for the Omni text core).

## Dominant risk — Omni-wrapper MoE-defaults resolution (BOOT LITMUS)

The arch class loads, but `flashinfer_cutlass` MoE on this *wrapper* REQUIRES the `sglang_launch.sh` `_sgl_nemotronh_omni_wrapper_` patch (PR #25024). WITHOUT it the wrapper bypasses the NemotronH MoE-defaults hook → llm_config-nested MoE settings unresolved → backend falls to AUTO → the sm_100-only `cutlass_moe_fp4` path → trips the `nx2_w1` shape assert during the flashinfer NVFP4 autotune (even with `moe_runner_backend=flashinfer_cutlass` set). **Case 01 is the litmus**: if it dies at arch-registration, in a mamba kernel, or on the `nx2_w1`/`cutlass_moe_fp4` assert, ALL cases die identically.

---

## Configuration Matrix

All cases: `tp=4, pp=1, ep=1, nccl=roce, moe_runner=flashinfer_cutlass, attention=flashinfer, kv_cache_dtype=fp8_e4m3, disable_deep_gemm=true, disable_piecewise_cuda_graph=true, context_length=262144, num_experts=128, tool_call_parser=qwen3_coder, speculative=off`. Only the swept axes differ.

CG variant encoding:
- **no-CG**: `disable_cuda_graph=true` (eager, safest boot)
- **full-CG**: `disable_cuda_graph=false` (profile default; piecewise still off)

| #  | Block | swept axis            | reasoning   | mem_frac | fp4_gemm     | CG       | Status   | n=1 tok/s | n=4 peak | n=8 peak | Output |
|----|-------|-----------------------|-------------|----------|--------------|----------|----------|-----------|----------|----------|--------|
| 01 | A     | boot litmus (eager)   | nemotron_3  | 0.60     | fi_cutlass   | no-CG    | UNTESTED | —         | —        | —        | —      |
| 02 | A     | full-CG (profile dflt)| nemotron_3  | 0.60     | fi_cutlass   | full-CG  | UNTESTED | —         | —        | —        | —      |
| 03 | B     | reasoning_parser A/B  | deepseek-r1 | 0.60     | fi_cutlass   | full-CG  | UNTESTED | —         | —        | —        | —      |
| 04 | C     | mem headroom          | nemotron_3  | 0.75     | fi_cutlass   | full-CG  | UNTESTED | —         | —        | —        | —      |
| 05 | D     | fp4_gemm probe        | nemotron_3  | 0.60     | fi_cudnn     | full-CG  | UNTESTED | —         | —        | —        | —      |

### Column legend

| Column     | Description                                                                                 |
|------------|---------------------------------------------------------------------------------------------|
| reasoning  | `reasoning_parser` — `nemotron_3` (HF card / Super value, profile default) vs `deepseek-r1` (SGLang cookbook §4.8) |
| mem_frac   | `mem_fraction_static` — 0.60 (profile, conservative) vs 0.75 (card Spark guidance 0.70–0.80) |
| fp4_gemm   | `fp4_gemm_backend` — `fi_cutlass` = flashinfer_cutlass, `fi_cudnn` = flashinfer_cudnn        |
| CG         | `disable_cuda_graph` — no-CG = eager (true), full-CG = capture graphs (false)                |
| Output     | quality verdict — read the answer text in `kikube-bench-*.log`, confirm `<think>` is split out, pattern-grep + TTR + tail-eyeball |

---

## Pre-run hypotheses

1. **Case 01 (litmus, eager)** answers the only first-order question: does the Omni wrapper resolve its MoE defaults + emit coherent tokens on this image. If it boots, the rest carry information. ⚠️ Eager is broken on the *native* `cutlass_moe_fp4` path (CLAUDE.md), but here MoE is `flashinfer_cutlass` (FlashInfer/TRT-LLM autotune, `trtllm::fused_moe`) — likely survives eager; if it crashes in the MoE instead of the graph, that localizes it.
2. **Case 02 (full-CG)** is the profile default and the production-candidate shape. Risk is the CUDA-graph capture of the hybrid flashinfer attention (`hybrid_linear_attn_backend → flashinfer_backend.init_cuda_graph_state`) — an illegal-memory-access there was observed once on a manual boot (see Preliminary Observations) but cleared on redeploy, suggesting a transient GPU-state issue rather than a config defect.
3. **Case 03 (deepseek-r1 parser)** is a CORRECTNESS axis, not throughput — verify `<think>` is separated from content (no leaked tags) and compare against case 02's `nemotron_3`. The HF card uses `nemotron_3`; the SGLang cookbook §4.8 uses `deepseek-r1`. Pick whichever splits reasoning cleanly.
4. **Case 04 (mem 0.75)** — small model (~21 GB weights), the boot log already showed `available_gpu_mem=42.49 GB` and a huge KV pool at 0.60, so 0.75 should be safe and only widens the (Mamba-clamped) pool. Drop back if it OOMs.
5. **Case 05 (fi_cudnn FP4 GEMM)** is a kernel-delta PROBE vs case 02. ⚠️ The 0.5.13-sm121 build may NOT ship the cuDNN-FP4 wheels (those come from a cuDNN image layer) — case may fail to import. On the Qwen3.6-35B-NVFP4 sibling, `fi_cudnn` was either broken or ~10% slower than `fi_cutlass` — low expectation of a win here.

---

## Preliminary observations (manual boot — NOT a kikube-bench matrix run)

These are from running the model through the live `default` SGLang instance on 2026-06-25, BEFORE the matrix was driven. Recorded for context; they do not fill the matrix.

- **Profile-default shape (= case 02: full-CG, nemotron_3, mem 0.60, ctx 262144) BOOTS and SERVES.** Head `xomoxcc/dgx-spark-sglang:0.5.13-sm121` started 2026-06-25 15:25:37Z, **0 restarts**, head Ready 2/2 (the `/v1/models` readiness probe passes → it is serving). The Omni wrapper resolves its MoE defaults on this image (litmus concern did NOT materialize): NCCL init COMPLETE, weights loaded, FlashInfer autotune (`trtllm::fused_moe::gemm1/2`) completed, MoE backend = flashinfer_cutlass as set.
- Boot log facts: `Tree cache: MambaRadixCache hybrid_ssm=True`, `max_total_num_tokens=19556473`, `max_running_requests=32` (the Mamba-state-cache clamp — NOT KV/graph), `context_len=262144`, `available_gpu_mem=42.49 GB`, `Disable piecewise CUDA graph because --disable-piecewise-cuda-graph is set`.
- **Earlier transient:** a prior boot attempt crashed during CUDA-graph capture — `flashinfer_backend.py:693 init_cuda_graph_state: self.cuda_graph_kv_indices[i][0] = 0 → CUDA illegal memory access` → sigquit → head/worker restart cascade. A fresh redeploy (new head hash) cleared it with the SAME cuda-graph config, so it reads as a transient GPU/rank state, not a config defect (mem was not the cause: 44 GB free at capture). If it recurs, check per-node clocks/power FIRST before touching the profile.
- **Tokenizer warning (open):** transformers flags the NemotronH tokenizer with a Mistral-derived "incorrect regex pattern" and suggests `fix_mistral_regex=True`; tokenizer also stays `TokenizersBackend` after `--trust-remote-code` retries ("model-specific attributes may be missing"). No SGLang CLI passthrough for `fix_mistral_regex`. Impact on tokenization is UNMEASURED — encode-diff test pending before deciding whether to patch the cached tokenizer.

---

## Results

**UNTESTED — matrix not yet driven via kikube-bench.** Fill the table above + the per-case sections below once the run completes.

Run with:
```
kikube-bench matrix matrixtest_matrices/sglang_nn4_tp4_ep1/nemotron-3-nano-omni-30b-a3b-reasoning-nvfp4/nv580.159_sglang-0.5.13-sm121_nemotron-3-nano-omni-30b-a3b-reasoning-nvfp4_n4_ep1.yaml
```
(append `--dry-run` to preview, `--start-at N` to resume.)

### Crash legend (for when results land)

- **crash S** (`startup_crash`): head/worker pod restarts — never reaches inference. The kernel combo doesn't compile/load on SM121 for this model.
- **crash B** (`bench_crash`): pod starts, every benchmark request fails (0/n). Inference reachable, first forward pass errors.
- **timeout**: `SGLang not ready after 900s`.

---

## Action items

- [ ] Drive the matrix (5 cases) — run **case 01 (eager litmus) FIRST**; if it dies at arch-registration / mamba kernel / `nx2_w1`/`cutlass_moe_fp4` assert, STOP and confirm the `_sgl_nemotronh_omni_wrapper_` launch patch is in this image build.
- [ ] Verify output quality on every `ok` case (pattern-grep + token-distribution + tail-eyeball) — first contact, no prior quality floor.
- [ ] **Case 03 correctness:** confirm `<think>` splits cleanly under `nemotron_3` vs `deepseek-r1` — read the actual answer text in `matrixtest/<date>/kikube-bench-*.log`, not the TESTRESULTS JSON. Pick the parser that doesn't leak tags; update the profile if `deepseek-r1` wins.
- [ ] **Case 04:** if 0.75 holds, consider lifting the profile `mem_fraction_static` from 0.60.
- [ ] **Case 05:** if `fi_cudnn` fails to import, note that the 0.5.13-sm121 base lacks the cuDNN-FP4 layer (needs the cuDNN-rebuilt image); else log the Δ vs case 02.
- [ ] Record the Mamba-state-cache pool line + `max_running_requests` clamp; set `max_mamba_cache_size` explicitly if concurrency needs tuning.
- [ ] Resolve the tokenizer regex question: encode-diff `fix_mistral_regex=True`/`False` in a debug pod; patch the cached tokenizer only if token IDs actually differ.
- [ ] Once a clean boot + coherent-output case is confirmed, drop the profile's "UNVALIDATED / FIRST-CONTACT" header caveats for the validated axes.
