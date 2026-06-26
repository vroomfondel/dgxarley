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
| Parallel  | tp=4, pp=1, ep=1 (ep=4 probed in case 11)                                   |

Matrix file: `kikube/matrixtest_matrices/sglang_nn4_tp4_ep1/nemotron-3-nano-omni-30b-a3b-reasoning-nvfp4/nv580.159_sglang-0.5.13-sm121_nemotron-3-nano-omni-30b-a3b-reasoning-nvfp4_n4_ep1.yaml`
Profile: `roles/k8s_dgx/model_profiles/nvidia-nemotron-3-nano-omni-30b-a3b-reasoning-nvfp4.yml`

**First contact for this model** — no prior baseline. Architecture sibling for A/B reference is the validated **Super-120B** NemotronH:
- `TESTLOGS/sglang_nn4_tp4_ep4/nemotron-3-super-120b-a12b-nvfp4/TESTLOG_nv580.159_sglang-0.5.13-mtp_nemotron-3-super-120b-a12b-nvfp4_4n.md` (same hybrid family; Super HAS an MTP head + EP=4 winner, this one does NOT — see Model Notes).

Why the SM121 build: the stock `scitrera/dgx-spark-sglang` image device-asserts on the triton/cutlass NVFP4 MoE path on SM121 (see `CUTLASS_NVFP4_SM121_PRD.md`); the `xomoxcc/…:0.5.13-sm121` build carries both the SM121 NVFP4-MoE fix AND the Omni arch class. ⚠️ **Audio gap:** `librosa` is NOT in this image — the Parakeet audio path would fail at runtime; add it to the recipe before any audio test. This matrix is TEXT-ONLY.

---

## Model Notes

- OMNI-MODAL wrapper `NemotronH_Nano_Omni_Reasoning_V3` around a NemotronH text core (`NemotronHForCausalLM`, `model_type=nemotron_h`). **Mamba2 + MoE + attention HYBRID.**
- Text core: 52 layers, hidden 2688, 32 attn heads, num_kv_heads 2 (GQA), 128 routed + 1 shared experts, 6 active/token, expert_intermediate 1856, native `max_position_embeddings=262144`. NoPE (Mamba2 carries order → context extension is just a cap-lift, no rope_scaling).
- NVFP4 modelopt-MIXED (~21 GB weights): routed expert FFN FP4 (E2M1, per-block FP8 E4M3 scales, group_size 16); Mamba in/out_proj + shared experts + attn o_proj FP8; vision (C-RADIOv2-H) + audio (Parakeet) encoders stay BF16.
- Reasoning post-train (`<think>`), `enable_thinking` ON by default; toggle per-request via `extra_body={"chat_template_kwargs":{"enable_thinking":false}}`.
- **NO MTP / speculative decoding.** VERIFIED 2026-06-25 three ways: (1) the served `config.json` has no `num_nextn_predict_layers`/nextn/mtp/draft key anywhere (top-level or nested `llm_config`); (2) the Nano Omni paper (arXiv 2604.24954) never mentions MTP/speculative/draft; (3) MTP is a Nemotron-3 family technique but only the **Super** ships a usable head. No native draft, no external draft → `speculative_enabled=false` everywhere. (Generic web summaries claiming "native MTP" conflate the family/Super discussion — not true for Nano/Omni.)
- Hybrid-Mamba concurrency: `max_running_requests` is clamped by the Mamba state-cache pool (`MambaRadixCache`), NOT by KV/cuda_graph — same as the Super. Without MTP there's no extra_buffer doubling, so the ratio is smaller.

## Closed axes (NOT swept — hard constraints)

- **attention = flashinfer ONLY.** triton attn is HARD-ASSERTED off on NemotronH (`apply_nemotron_h_defaults`: first layer may be Mamba, not attention). Mamba2 SSM layers use their own kernels regardless. No triton-ATTN probe.
- **quant = `modelopt_fp4`**; DeepGemm disabled (NVFP4 scale_fmt != ue8m0).
- **tp_size = 4 fixed** — this is the nn4/TP=4 topology dir. The card's TP=1 single-Spark target (~21 GB fits one 128 GB node) is a DIFFERENT topology and belongs in a separate `sglang_nn4_tp1_ep1` / single-node matrix, not here.
- **speculative / MTP = OFF everywhere** (there is none — see Model Notes).

## Open axes (each case varies ONE axis off the Block-A full-CG baseline = case 02)

A CUDA graph · B reasoning_parser · C mem_fraction_static · D cuda_graph_max_bs · E kv_cache_dtype · F fp4_gemm · G context_length · H ep_size · I moe_runner_backend · J piecewise CUDA graph.

CG variant encoding:
- **no-CG**: `disable_cuda_graph=true` (eager, safest boot)
- **full-CG**: `disable_cuda_graph=false`, `disable_piecewise_cuda_graph=true` (profile default / baseline)
- **piecewise**: `disable_cuda_graph=false`, `disable_piecewise_cuda_graph=false` (PROBE only)

## Dominant risk — Omni-wrapper MoE-defaults resolution (BOOT LITMUS)

The arch class loads, but `flashinfer_cutlass` MoE on this *wrapper* REQUIRES the `sglang_launch.sh` `_sgl_nemotronh_omni_wrapper_` patch (PR #25024). WITHOUT it the wrapper bypasses the NemotronH MoE-defaults hook → llm_config-nested MoE settings unresolved → backend falls to AUTO → the sm_100-only `cutlass_moe_fp4` path → trips the `nx2_w1` shape assert during the flashinfer NVFP4 autotune (even with `moe_runner_backend=flashinfer_cutlass` set). **Case 01 is the litmus**: if it dies at arch-registration, in a mamba kernel, or on the `nx2_w1`/`cutlass_moe_fp4` assert, ALL cases die identically — stop, confirm the launch patch is in this image build, re-run.

---

## Configuration Matrix (13 cases, Blocks A–J)

**Baseline = case 02:** `moe_runner=flashinfer_cutlass, attention=flashinfer, fp4_gemm=flashinfer_cutlass, reasoning=nemotron_3, kv_cache_dtype=fp8_e4m3, mem_fraction_static=0.60, full-CG, cuda_graph_max_bs=32, context_length=262144, ep=1, tp=4`. Every other case = baseline with the **one bold Δ** shown.

| #  | Block | axis    | Δ vs case-02 baseline                | Status   | n=1 tok/s | n=4 peak | n=8 peak | Output |
|----|-------|---------|--------------------------------------|----------|-----------|----------|----------|--------|
| 01 | A     | CG      | **no-CG (eager)** — BOOT LITMUS      | **ok**     | 44.5      | 168.0    | 328.6    | clean ✓ |
| 02 | A     | CG      | — (baseline: full-CG)                | **ok 🏆**  | 90.1      | 268.1    | 437.9    | clean ✓ |
| 03 | B     | parser  | reasoning_parser **deepseek-r1**     | **ok**     | 87.8      | 264.2    | 432.8    | clean ✓ |
| 04 | C     | mem     | mem_fraction_static **0.75**         | **ok**     | 88.7      | 256.0    | 418.2    | clean ✓ |
| 05 | C     | mem     | mem_fraction_static **0.80**         | **ok**     | 90.1      | 263.2    | 428.2    | clean ✓ |
| 06 | D     | cgbs    | cuda_graph_max_bs **64**             | **ok**     | 85.3      | 267.1    | 435.0    | clean ✓ |
| 07 | D     | cgbs    | cuda_graph_max_bs **128**            | **ok**     | 89.8      | 265.0    | 432.6    | clean ✓ |
| 08 | E     | kv      | kv_cache_dtype **auto (bf16)**       | **ok**     | 90.5      | 263.4    | 424.5    | clean ✓ |
| 09 | F     | fp4_gemm| fp4_gemm **flashinfer_cudnn** PROBE  | **ok⚠**    | 89.3      | 267.1    | 438.6    | **flag** ⚠ |
| 10 | G     | context | context_length **524288** (2×) PROBE | **ok**     | 91.0      | 262.6    | 430.5    | clean ✓ |
| 11 | H     | ep      | ep_size **4** PROBE                  | **crash S**| —         | —        | —        | (capture)|
| 12 | I     | moe     | moe_runner **triton** PROBE          | **crash S**| —         | —        | —        | (assert) |
| 13 | J     | piecewise | **piecewise CG** PROBE             | ok (n/a)   | 91.2      | 268.8    | 436.6    | clean ✓§ |

### Column legend

| Column | Description |
|--------|-------------|
| axis   | which open axis this case varies off the case-02 baseline |
| Status | `UNTESTED` / `ok` / `crash S` (startup) / `crash B` (bench) / `timeout` |
| Output | quality verdict — read the answer text in `kikube-bench-*.log`, confirm `<think>` is split out, pattern-grep + TTR + tail-eyeball |

---

## Pre-run hypotheses (per block)

- **A — CG (01 eager LITMUS / 02 full-CG):** case 01 answers the only first-order question — does the Omni wrapper resolve its MoE defaults + emit coherent tokens. ⚠️ Eager is broken on the *native* `cutlass_moe_fp4` path (CLAUDE.md), but here MoE is `flashinfer_cutlass` (FlashInfer/TRT-LLM autotune, `trtllm::fused_moe`) — likely survives eager. Case 02 (full-CG) is the production-candidate; its risk is the hybrid flashinfer-attn graph capture (`hybrid_linear_attn_backend → flashinfer_backend.init_cuda_graph_state`) — an illegal-memory-access there was seen once on a manual boot but cleared on redeploy (Preliminary Observations).
- **B — reasoning_parser (03 deepseek-r1):** CORRECTNESS axis, not throughput. HF card uses `nemotron_3`; SGLang cookbook §4.8 uses `deepseek-r1`. Verify `<think>` is separated from content (no leaked tags); pick whichever splits cleanly. Judge from the answer text in the `kikube-bench-*.log`, NOT the TESTRESULTS JSON.
- **C — mem (04 / 05):** small model (~21 GB weights), manual boot already showed `available_gpu_mem=42.49 GB` and a huge KV pool at 0.60 → 0.75 and 0.80 should be safe and only widen the (Mamba-clamped) pool. Drop back if any OOMs.
- **D — cuda_graph_max_bs (06=64 / 07=128):** capture-memory headroom on a small model; larger bs can lift batched-decode throughput IF the hybrid Mamba/attn graph still captures cleanly at the larger batch. Keep 32 if a larger bs fails to capture or OOMs on graph memory.
- **E — kv (08 auto/bf16):** fp8 KV has been broken on some arches in this fleet — confirm `fp8_e4m3` holds on the Omni text core and measure the quality/throughput Δ vs bf16. bf16 KV roughly doubles per-token KV cost → smaller pool, but is the safe correctness reference.
- **F — fp4_gemm (09 fi_cudnn PROBE):** kernel delta vs case 02. ⚠️ the 0.5.13-sm121 base may NOT ship the cuDNN-FP4 wheels (cuDNN image layer) — may fail to import. On the Qwen3.6-35B-NVFP4 sibling `fi_cudnn` was broken pre-rebuild and ~10% slower than `fi_cutlass` after — low expectation of a win.
- **G — context (10 → 524288 PROBE):** NoPE → extension is a cap-lift only (`json_model_override_args` auto-sets `SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN`). Only answers "does the 2× extension boot + serve", NOT whether long-context quality holds (no published RULER curve). If it OOMs on the KV pool, drop mem_fraction_static or pin chunked_prefill_size.
- **H — ep (11 EP=4 PROBE):** 128 experts % 4 == 0 and the Super's EP=4 was its winner. Watch for the gated-padding / swizzle-pad asserts seen on other NVFP4 MoE models — does the 128-expert NVFP4 layout shard cleanly at EP=4 on this wrapper.
- **I — moe (12 triton PROBE):** on the modelopt NVFP4 path triton normally falls through to `cutlass_moe_fp4`. Single probe to confirm that behaviour on the Omni wrapper (likely no-op fallback or a crash) — keep `flashinfer_cutlass` regardless.
- **J — piecewise (13 PROBE):** profile/Super disable piecewise (the Mamba2/attn hybrid doesn't piecewise-capture cleanly). One probe to confirm that holds on this build (likely crashes / fails to capture). If it boots AND benches, piecewise could be promoted.

---

## Preliminary observations (manual boot — NOT a kikube-bench matrix run)

From running the model through the live `default` SGLang instance on 2026-06-25, BEFORE the matrix was driven. Recorded for context; they do NOT fill the matrix.

- **Profile-default shape (= case 02: full-CG, nemotron_3, mem 0.60, ctx 262144) BOOTS and SERVES.** Head `xomoxcc/dgx-spark-sglang:0.5.13-sm121` started 2026-06-25 15:25:37Z, **0 restarts**, head Ready 2/2 (the `/v1/models` readiness probe passes → it is serving). The Omni wrapper resolves its MoE defaults on this image (litmus concern did NOT materialize on the full-CG shape): NCCL init COMPLETE, weights loaded, FlashInfer autotune (`trtllm::fused_moe::gemm1/2`) completed, MoE backend = flashinfer_cutlass as set.
- Boot log facts: `Tree cache: MambaRadixCache hybrid_ssm=True`, `max_total_num_tokens=19556473`, `max_running_requests=32` (the Mamba-state-cache clamp — NOT KV/graph), `context_len=262144`, `available_gpu_mem=42.49 GB`, `Disable piecewise CUDA graph because --disable-piecewise-cuda-graph is set`.
- **Earlier transient:** a prior boot attempt crashed during CUDA-graph capture — `flashinfer_backend.py:693 init_cuda_graph_state: self.cuda_graph_kv_indices[i][0] = 0 → CUDA illegal memory access` → sigquit → head/worker restart cascade. A fresh redeploy (new head hash) cleared it with the SAME cuda-graph config, so it reads as a transient GPU/rank state, not a config defect (mem was not the cause: 44 GB free at capture). If it recurs, check per-node clocks/power FIRST before touching the profile.
- **Tokenizer warning (open):** transformers flags the NemotronH tokenizer with a Mistral-derived "incorrect regex pattern" and suggests `fix_mistral_regex=True`; tokenizer also stays `TokenizersBackend` after `--trust-remote-code` retries ("model-specific attributes may be missing"). No SGLang CLI passthrough for `fix_mistral_regex`. Impact on tokenization is UNMEASURED — encode-diff test pending before deciding whether to patch the cached tokenizer.

---

## Results

**COMPLETE** (2026-06-25 ~18:08 → ~19:1x). **13 / 13 cases attempted** — **11 ok** (1 with a quality flag), **2 startup_crash** (case 11 ep=4 inconclusive, case 12 triton-MoE confirmed-assert). Peak = sum of per-request tok/s. **WINNER: case 02 (full-CG baseline) — which is the existing profile default. No profile change needed.**

### All 13 cases

| #  | Block · Δ                  | n=1 peak | n=4 peak | n=8 peak | n=16 peak | ok      | quality |
|----|----------------------------|---------:|---------:|---------:|----------:|---------|---------|
| 01 | A · no-CG (eager) LITMUS   |    44.5  |  168.0   |  328.6   |   610.7   | 1/4/8/16 | clean ✓ |
| 02 | A · full-CG (baseline) 🏆  |  **90.1**| **268.1**| **437.9**| **660.8** | 1/4/8/16 | clean ✓ |
| 03 | B · deepseek-r1 parser     |    87.8  |  264.2   |  432.8   |   670.9   | 1/4/8/16 | clean ✓ |
| 04 | C · mem_fraction 0.75      |    88.7  |  256.0   |  418.2   |   644.6   | 1/4/8/16 | clean ✓ |
| 05 | C · mem_fraction 0.80      |    90.1  |  263.2   |  428.2   |   654.6   | 1/4/8/16 | clean ✓ |
| 06 | D · cuda_graph_max_bs 64   |    85.3  |  267.1   |  435.0   |   669.6   | 1/4/8/16 | clean ✓ |
| 07 | D · cuda_graph_max_bs 128  |    89.8  |  265.0   |  432.6   |   661.3   | 1/4/8/16 | clean ✓ |
| 08 | E · kv_cache bf16 (auto)   |    90.5  |  263.4   |  424.5   |   646.6   | 1/4/8/16 | clean ✓ |
| 09 | F · fp4_gemm fi_cudnn PROBE|    89.3  |  267.1   |  438.6   |  628.2†   | 1/4/8/**15** | flag ⚠ |
| 10 | G · context 524288 (2×)    |    91.0  |  262.6   |  430.5   |   665.4   | 1/4/8/16 | clean ✓‡ |
| 11 | H · ep_size 4 PROBE        |    —     |   —      |   —      |    —      | **crash S** | — |
| 12 | I · moe_runner triton PROBE|    —     |   —      |   —      |    —      | **crash S** | — |
| 13 | J · piecewise CG PROBE §   |    91.2  |  268.8   |  436.6   |   654.4   | 1/4/8/16 | clean ✓ |

`†` case-09 n16 drop is the 1 dropped (repetition-aborted) request, not a throughput regression. `‡` case-10 only proves the 2× extension boots+serves short prompts — long-context quality untested. `§` **case-13 did NOT actually exercise piecewise — and the plumbing is NOT the cause (verified from the case-13 head log).** The chain delivered `false` correctly: env `SGLANG_DISABLE_PIECEWISE_CUDA_GRAPH=false`, and the launch argv correctly OMITS `--disable-piecewise-cuda-graph`. Yet `server_args` still resolves `disable_piecewise_cuda_graph=True` — **SGLang itself forces it True for the NemotronH/Mamba hybrid arch** (the "…--disable-piecewise-cuda-graph is set" log line is misleading; the flag is not in the argv). So `disable_piecewise_cuda_graph: true` in the profile is effectively a no-op restating SGLang's own forced behavior; piecewise is simply unavailable for this model. The only lever would be `--enforce-piecewise-cuda-graph` (server_args `enforce_piecewise_cuda_graph`), which our launch doesn't wire and the hybrid path would likely reject anyway. Case 13 = effectively a re-run of case 02. **No plumbing fix needed.**

Findings so far:
1. **BOOT LITMUS PASSED.** Case 01 (eager) boots, serves, and emits coherent text — the Omni wrapper resolves its MoE defaults on this image (no `nx2_w1` / `cutlass_moe_fp4` assert, no mamba-kernel crash). The `_sgl_nemotronh_omni_wrapper_` launch patch is effective in `0.5.13-sm121`. All downstream cases are therefore meaningful.
2. **CUDA graphs are a large win** — full-CG (02) vs eager (01): n=1 **90.1 vs 44.5 (+102 %)**, n=8 **437.9 vs 328.6 (+33 %)**. (NOT the usual eager-MoE collapse — `flashinfer_cutlass` MoE graph-captures fine; the earlier manual-boot `init_cuda_graph_state` illegal-memory-access did NOT recur.) Case 02 is the current **winner** (matrix summary agrees).
3. **Output quality clean** on both: reasoning splits (`think_tokens_est` > 0 in 8/8), no `!`-token collapse, TTR_min 0.62 (02) / 0.65 (01) — well above the ~0.53 word-salad floor seen on the Qwen3.6-35B-NVFP4 sibling. Snippets are on-topic and diverse (DNS resolution, GC comparison, bash scripts, Gödel). ⚠️ Snippets begin with a "We need to answer as…" CoT-style preamble — likely the `<think>` segment leading the snippet; whether `nemotron_3` cleanly strips think from the *served* content (vs leaking) is exactly the **Block B (case 03)** correctness question — verify there from the `kikube-bench-*.log` answer text.
4. Throughput shape is concurrency-bound by the Mamba state-cache clamp (`max_running_requests=32`): per-request tok/s falls 90→67→55→41 as n goes 1→4→8→16 while peak still climbs — expected for a hybrid-Mamba MoE.
5. **Block B (03, deepseek-r1 parser): throughput-neutral vs nemotron_3** — n=8 432.8 vs case 02's 437.9 (within noise), as expected for a pure correctness axis. Quality clean (TTR 0.635, think-split 8/8). NOTE: the "We need to answer as…" CoT preamble in the snippets appears under BOTH parsers (02 nemotron_3 AND 03 deepseek-r1), so it's the model's reasoning style leading the snippet, not a parser artifact. The definitive "does `<think>` leak into *served content*" judgment still needs the `kikube-bench-*.log` answer text (the bench's `think_tokens_est` shows it IS being separated; both parsers behave equivalently here). No reason yet to switch the profile off `nemotron_3`.
6. **Block C complete (04 mem 0.75 / 05 mem 0.80): no throughput benefit — flat across 0.60/0.75/0.80.** n=8 = 437.9 / 418.2 / 428.2 (all within noise of the 0.60 baseline). Lifting mem_fraction widens the KV pool, but this model is Mamba-state-clamped (`max_running_requests=32`), not KV-bound, so a bigger KV pool buys nothing. All clean (TTR 0.652 / 0.657). **Keep the profile at mem_fraction_static 0.60** — no throughput reason to raise it (would only matter if a future workload needs the larger KV pool for long single requests).

7. **Block D complete (06 cgbs 64 / 07 cgbs 128): throughput-neutral — flat across 32/64/128.** n=8 = 437.9 / 435.0 / 432.6. Larger graph-capture batches capture cleanly (no OOM/capture failure on the hybrid Mamba/attn graph) but buy nothing at n≤16 — concurrency is Mamba-state-clamped (`max_running_requests=32`), so the decode batch never grows enough for a bigger captured bs to matter. Both clean (TTR 0.62). **Keep cuda_graph_max_bs 32.**

8. **Block E (08, kv_cache_dtype bf16/auto): works, fp8 marginally faster — keep fp8.** n=8 424.5 (bf16) vs 437.9 (fp8_e4m3, case 02) — fp8 KV is ~3 % faster AND uses half the KV memory, with no quality cost (bf16 clean TTR 0.641, fp8 clean too). **No fp8-KV breakage on the Omni text core** (the fleet-wide fp8-KV concern does NOT apply here). **Keep `kv_cache_dtype: fp8_e4m3`.**
9. **Block F (09, fp4_gemm flashinfer_cudnn PROBE): the build DOES carry cuDNN-FP4 — hypothesis wrong.** It did NOT fail to import (contra the pre-run note that `0.5.13-sm121` might lack the cuDNN-FP4 wheels — they're present). Throughput is **tied with fi_cutlass at n=8 (438.6 vs 437.9)**, NOT the ~10 % deficit seen on the Qwen3.6-35B-NVFP4 sibling. BUT a **quality wobble**: 1 of 16 requests at n=16 hit a `repetition` abort (the † on the n16 660.8→628.2 drop is the missing request, NOT a throughput regression — `feedback_throughput_failure_normalization`), and n=8 TTR_min 0.583 is the lowest of any case (fp8 cases sit 0.62–0.66). Reads as fi_cudnn FP4 being slightly numerically looser → occasional repetition degeneration under load. **Keep `fp4_gemm_backend: flashinfer_cutlass`** (the default) for reliability; fi_cudnn is viable but marginally less stable, no upside.

10. **Block G (10, context 524288 2× extension): boots + serves, throughput-neutral.** n=8 430.5 ≈ baseline; the NoPE cap-lift (`json_model_override_args={"max_position_embeddings":524288}`) works — KV pool fits (fp8 KV), no OOM. ‡ This ONLY proves the 2× extension boots and serves short prompts; long-context QUALITY is untested (no RULER curve). TTR_min 0.577 on the short bench prompts is the lowest of the clean cases but with no repetition abort — keep the profile at native 262144 until a real long-context quality number exists.
11. **Block H (11, ep_size=4 PROBE): startup_crash — CONFIRMED BROKEN (reproduced ≥2×, Loki-verified).** EP=4 dies DETERMINISTICALLY at CUDA-graph capture. The head log ends silently at `Capture cuda graph begin`, but Loki has the worker traceback: it's `cuda_graph_runner.py:543 → hybrid_linear_attn_backend.init_cuda_graph_state → flashinfer_backend.py:693 → self.cuda_graph_kv_indices[i][0] = 0 → torch.AcceleratorError: CUDA illegal memory access` on the EP-labeled rank. Reproduced on TWO independent ep>1 runs: a pre-matrix manual attempt (2026-06-25 **14:39–14:45Z**, ranks `EP2`/`EP3`, crashed repeatedly) AND matrix case 11 (**17:11Z**, `EP3`). **NOT an OOM** (35–44 GB free at capture) and **NOT an EP-specific kernel assert** — it's the FlashInfer attn graph-capture path. Crucially it's the **IDENTICAL stack** to the rare ep=1 "transient" first-boot flake (Preliminary Observations) — same `cuda_graph_kv_indices` init bug; ep>1 just makes it fire every time while ep=1 only flakes occasionally (and runs stably otherwise). EP=1 (the default) is the right call. WORKAROUND if EP is ever needed: ep>1 + `disable_cuda_graph` (eager) to skip `init_cuda_graph_state` — untested; and EP gives no expected throughput win on this small model, so not worth chasing.

12. **Block I (12, moe_runner triton PROBE): startup_crash — CONFIRMED closed-axis assumption.** Clear traceback (unlike case 11): `fused_moe_triton/layer.py forward → run_moe_core → cutlass_moe_fp4 (cutlass_moe.py:428) → AssertionError: mismatch in expected n`. So on the modelopt NVFP4 path, `moe_runner_backend=triton` falls through to `cutlass_moe_fp4`, which trips the `n`-dimension (nx2_w1) shape assert — exactly the failure the boot-litmus warned about. **triton MoE is NOT a viable alternative; keep `flashinfer_cutlass`.** NB this DIFFERENT, explicit assert (vs case 11's silent capture crash) reinforces that case 11's ep=4 crash is NOT the cutlass_moe_fp4 assert.
13. **Block J (13, piecewise PROBE): INVALID — piecewise never engaged, and the plumbing is NOT at fault (verified from the case-13 head log).** The case boots+benches `ok` (n=8 436.6, clean TTR 0.603) but ran identical to case 02. ROOT CAUSE (proven): env `SGLANG_DISABLE_PIECEWISE_CUDA_GRAPH=false` was delivered correctly and the launch argv correctly OMITS `--disable-piecewise-cuda-graph`, yet `server_args` resolves `disable_piecewise_cuda_graph=True` → **SGLang force-disables piecewise for the NemotronH/Mamba hybrid arch**, not a config/plumbing bug (the "…flag is set" log line is misleading — it's not in the argv). So piecewise is simply unavailable for this model; the profile's `disable_piecewise_cuda_graph: true` just restates SGLang's own behavior. Only `--enforce-piecewise-cuda-graph` could force it (unwired, and likely arch-rejected). **No plumbing fix; piecewise is moot here.**

---

## Conclusion

**Matrix complete: 11/13 ok, 2 startup_crash. The winner is case 02 — which is already the profile default, so the profile needs NO change.** The validated production shape is: `flashinfer_cutlass` MoE + `flashinfer` attn + `flashinfer_cutlass` FP4 GEMM + `nemotron_3` parser + `fp8_e4m3` KV + `full-CG` (piecewise off) + `cuda_graph_max_bs=32` + `mem_fraction_static=0.60` + `ep=1` + native `262144` context. Peak n=8 **437.9 tok/s**, n=16 660.8, all output clean.

Key takeaways:
- **CUDA graphs (full-CG) are the ONLY real throughput lever** — +33 % n=8 over eager. Every other A/B axis (reasoning_parser, mem_fraction 0.60/0.75/0.80, cuda_graph_max_bs 32/64/128, fp8-vs-bf16 KV) is throughput-neutral and quality-clean.
- **The Omni wrapper resolves its MoE defaults on `0.5.13-sm121`** (boot-litmus passed) — the `_sgl_nemotronh_omni_wrapper_` launch patch is effective; the earlier manual-boot capture crash did not recur in the matrix.
- **2× context (524288) boots + serves** (NoPE cap-lift works) — but long-context quality is untested (no RULER); stay native.
- **Confirmed-bad:** triton MoE (falls to cutlass_moe_fp4 → `n`-assert). **Skip:** fi_cudnn FP4 (works + tied throughput, but a repetition wobble + lowest TTR — no upside over fi_cutlass).
- **Confirmed-bad / follow-ups:** (a) **ep=4 is CONFIRMED BROKEN** — reproduced ≥2× (Loki: pre-matrix manual run 14:39–14:45Z with EP2/EP3 + matrix case 11 17:11Z), deterministic `illegal memory access` in `flashinfer_backend.init_cuda_graph_state` (`cuda_graph_kv_indices[i][0]=0`), NOT OOM. Same stack as the rare ep=1 first-boot flake — ep>1 makes it fire every time. Keep ep=1; if EP is ever needed, try ep>1 + eager. (b) **piecewise** is not a plumbing bug — SGLang force-disables it for this hybrid arch (verified: env=false + flag omitted, yet server_args=True); unavailable here, nothing to fix. (c) reasoning-parser think-leak: both parsers behave equivalently in the bench; the definitive `<think>`-vs-content split judgment still wants a look at a raw `kikube-bench-*.log` answer.

**Profile action:** none required — seed already matched the winner. Safe to drop the profile's "UNVALIDATED / FIRST-CONTACT" header caveats for the now-validated axes (kernels, CG, parser, mem, kv, context-boot), keeping the open notes only for ep=4, piecewise, audio (librosa), and long-context quality.

Run with:

Run with:
```
kikube-bench matrix matrixtest_matrices/sglang_nn4_tp4_ep1/nemotron-3-nano-omni-30b-a3b-reasoning-nvfp4/nv580.159_sglang-0.5.13-sm121_nemotron-3-nano-omni-30b-a3b-reasoning-nvfp4_n4_ep1.yaml
```
(append `--dry-run` to preview, `--start-at N` to resume; cases 04/05, 06/07, 08, 11 assume a clean boot from 01/02.)

### Crash legend (for when results land)

- **crash S** (`startup_crash`): head/worker pod restarts — never reaches inference. The kernel/axis combo doesn't compile/load on SM121 for this model.
- **crash B** (`bench_crash`): pod starts, every benchmark request fails (0/n). Inference reachable, first forward pass errors.
- **timeout**: `SGLang not ready after 900s`.

---

## Action items

- [ ] Drive the matrix (13 cases) — run **case 01 (eager litmus) FIRST**; if it dies at arch-registration / mamba kernel / `nx2_w1`/`cutlass_moe_fp4` assert, STOP and confirm the `_sgl_nemotronh_omni_wrapper_` launch patch is in this image build. Cases 03–13 carry information only after 01/02 boot clean.
- [ ] Verify output quality on every `ok` case (pattern-grep + token-distribution + tail-eyeball) — first contact, no prior quality floor.
- [ ] **B (03) correctness:** confirm `<think>` splits cleanly under `nemotron_3` vs `deepseek-r1` — read the actual answer text in `matrixtest/<date>/kikube-bench-*.log`, not the TESTRESULTS JSON. Update the profile if `deepseek-r1` wins.
- [ ] **C (04/05):** if 0.75/0.80 hold, consider lifting the profile `mem_fraction_static` from 0.60 to the best non-OOM value.
- [ ] **D (06/07):** if a larger `cuda_graph_max_bs` captures cleanly AND lifts n=8 peak, bump the profile; else keep 32.
- [ ] **E (08):** record the fp8_e4m3-vs-bf16 KV quality/throughput Δ; keep fp8 unless it regresses quality.
- [ ] **F (09):** if `fi_cudnn` fails to import, note the 0.5.13-sm121 base lacks the cuDNN-FP4 layer (needs the cuDNN-rebuilt image); else log the Δ vs case 02.
- [ ] **G (10):** if 524288 boots+serves, note it only proves boot, NOT long-context quality (no RULER curve) — keep native 262144 in the profile until a quality number exists.
- [ ] **H (11):** if EP=4 shards cleanly and helps, it's a candidate (mirrors the Super winner); watch for gated-padding/swizzle-pad asserts.
- [ ] **I (12) / J (13):** confirm the closed-axis assumptions (triton-MoE falls through / piecewise doesn't capture) hold on this build; document the failure signature.
- [ ] Record the Mamba-state-cache pool line + `max_running_requests` clamp; set `max_mamba_cache_size` explicitly if concurrency needs tuning.
- [ ] Resolve the tokenizer regex question: encode-diff `fix_mistral_regex=True`/`False` in a debug pod; patch the cached tokenizer only if token IDs actually differ.
- [ ] Once a clean boot + coherent-output winner is confirmed, drop the profile's "UNVALIDATED / FIRST-CONTACT" header caveats for the validated axes and flip the profile to the winning shape.
