# SGLang Test Log — NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4 (NemotronH hybrid), 4 Nodes, TP=4 EP=1, v0.5.13-dev MTP

> **STATUS: PENDING — pre-run skeleton.** Matrix authored + dry-run-validated 2026-06-16;
> not yet executed. Result cells are `TBD` and get filled in after the run. The 0.5.12
> first-contact TESTLOG (`TESTLOG_nv580.159_sglang-0.5.12_..._4n.md`) is the no-MTP
> reference; this run adds MTP on a dedicated image and re-validates the structural axes
> on that image.

## Environment

| Component | Value                                                                                   |
|-----------|-----------------------------------------------------------------------------------------|
| GPU       | NVIDIA GB10 (SM121/Blackwell), 128 GB per node                                          |
| Driver    | 580.159.03  *(verified on spark1–4, 2026-06-16 — matches the `nv580.159` filename label)* |
| CUDA      | 13.0.3 host toolkit (`/usr/local/cuda-13.0`); nvidia-smi reports max CUDA 13.0          |
| Kernel    | 6.17.0-1021-nvidia (aarch64)                                                            |
| OS        | Ubuntu 24.04.4 LTS (aarch64)                                                            |
| K3s       | v1.36.1+k3s1                                                                            |
| Nodes     | spark1 (head), spark2, spark3, spark4 (workers) — GB10, 1 GPU each; control-plane = elite800 (amd64, no GPU) |
| Image     | `xomoxcc/dgx-spark-sglang:0.5.13-dev-nemotronh-mtp-sm121` — **dedicated NemotronH-MTP** |
| Model     | `nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4`                                        |
| Arch      | `NemotronHForCausalLM` (`model_type=nemotron_h`) — Mamba2 + MoE + attn hybrid           |
| Quant     | `modelopt_mixed` / `MIXED_PRECISION` (expert FFN FP4 g16, attn/latent/MTP/emb FP8/BF16) |
| NCCL      | TBD — record from the boot log at run time (image-dependent; the image currently on-node carries libnccl 2.30.7) |
| Transport | **RoCE** via SR-IOV VF                                                                  |

Matrix file: `kikube/matrixtest_matrices/sglang_nn4_tp4_ep1/nemotron-3-super-120b-a12b-nvfp4/nv580.159_sglang-0.5.13-mtp_nemotron-3-super-120b-a12b-nvfp4_n4_ep1.yaml`
Results: `kikube/matrixtest/<run-date>/results/sglang_nn4_tp4_ep1/nemotron-3-super-120b-a12b-nvfp4/0.5.13-mtp/`
EP=4 sibling: `kikube/matrixtest_matrices/sglang_nn4_tp4_ep4/nemotron-3-super-120b-a12b-nvfp4/nv580.159_sglang-0.5.13-mtp_nemotron-3-super-120b-a12b-nvfp4_n4_ep4.yaml`

### Why a new image (the "extra image")

MTP on NemotronH-NVFP4 needs a build carrying the June-2026 MTP fixes; the upstream
`scitrera:0.5.12` base used for the first-contact run does NOT. This run pins the dedicated
`xomoxcc/dgx-spark-sglang:0.5.13-dev-nemotronh-mtp-sm121` (now also the profile's
`sglang_image:`). Relevant upstream (researched 2026-06-16):

- **#24955** "Support Nemotron DP attention and MTP" — MERGED 2026-06-12
- **#28102** "Fix DP attention + EP mode of Nemotron" — MERGED 2026-06-13
- **#27184** "fix Nemotron Super MTP deploy (spec-v2+B200)" — MERGED 2026-06-03
- **#27998** "[NemotronH] MTP with radix cache" — GB10-VALIDATED on THIS model (98% prefix
  reuse, MTP active, no regression); removes the `--disable-radix-cache` requirement and
  auto-selects mamba `extra_buffer` for spec+radix.

---

## Why this matrix exists — two questions, one run

The 0.5.12 first-contact run left MTP out of scope (two blockers) and ran on a different
image. This run answers:

1. **Q1 — Non-MTP image delta (Blocks A–E, `speculative_enabled=false`):** does
   0.5.13-dev-nemotronh-mtp run the Super better/differently than 0.5.12 **even without
   MTP**? Same 7 structural cases as the 0.5.12 matrix, re-run here so the image delta is
   isolated from the MTP delta. Without this, a faster MTP number could just be a faster image.
2. **Q2 — MTP payoff (Blocks F–G):** does EAGLE MTP finally pay off on NemotronH-NVFP4, and
   at which draft depth? **Denominator = the Block-A no-spec numbers ON THIS IMAGE**, not the
   0.5.12 135 tok/s (that is a cross-image cross-check only).

## Dominant risks / success criteria (fill the verdict after the run)

- **R1 — arch boot on the new dev image (Case 01).** Does `NemotronHForCausalLM` NVFP4 +
  Mamba2 SM121 still load + emit coherent tokens on 0.5.13-dev? If Case 01 dies at
  arch-registration or a mamba kernel, ALL cases die identically — stop. → **Verdict: TBD**
- **R2 — MTP NaN logits (#27828, DO-NOT-MERGE debug).** The NVFP4 MTP target-logits path
  chased a NaN on release/v0.5.13 — the riskiest path. Watch the first MTP case (08) boot log
  for NaN logits. → **Verdict: TBD**
- **R3 — accept rate (#21138 / #27998).** Old NemotronH MTP rejected ~all draft tokens
  (accept_len ≈ 0.33 → no speedup) because the MTP weight loader filtered `lm_head.weight` +
  `backbone.embeddings` out. **SUCCESS CRITERION for every MTP case: `accept_len > 1`**
  (target ~3.45 @ draft-len 7, TRT figure). `accept_len ≈ 1` = loader bug still bites, MTP
  pays nothing. → **Verdict: TBD**

## Closed axes (re-probed on the new image, not blindly trusted)

- **attention = flashinfer ONLY** — triton attn hard-asserted off on NemotronH. Not probed.
- **MoE runner = flashinfer_cutlass** — triton startup-crashed on 0.5.12 (Case 06 re-probes).
- **fp4_gemm = flashinfer_cutlass** — fi_cudnn FP4 wheel absent on 0.5.12 (Case 05 re-probes;
  dev image MAY now ship cuDNN-FP4).
- **piecewise CUDA graph = off** (card flag; Case 07 re-probes).
- **context** 262k/512k/1M all booted on 0.5.12 at ~equal throughput; 524288 serving default.
  `mem_fraction_static=0.80` comfortable.

---

## Configuration Matrix

All cases: `tp=4, pp=1, ep=1, nccl_transport=roce, attention_backend=flashinfer,
kv_cache_dtype=fp8_e4m3, mem_fraction_static=0.80, disable_deep_gemm=true,
quantization=modelopt_fp4` unless noted. MTP cases: `speculative_algo=EAGLE,
speculative_eagle_topk=1, speculative_draft_model_path=""` (built-in MTP layer).

CG variant encoding:
- **no-CG** : `disable_cuda_graph=true,  disable_piecewise_cuda_graph=true` (eager)
- **full-CG** : `disable_cuda_graph=false, disable_piecewise_cuda_graph=true` (serving)
- **piecewise** : `disable_cuda_graph=false, disable_piecewise_cuda_graph=false` (PROBE only)

### Block A — NO-SPEC boot litmus + CG (fi_cutlass MoE, fi_cutlass-fp4, ctx262k) — Cases 01–02

| #  | spec | CG variant     | ctx  | Status | n=1 tok/s | n=4 peak | n=8 peak | n=8 ok | Output |
|----|------|----------------|------|--------|----------:|---------:|---------:|--------|--------|
| 01 | off  | no-CG (eager)  | 262k | TBD    | TBD       | TBD      | TBD      | TBD    | TBD    |
| 02 | off  | full-CG        | 262k | TBD    | TBD       | TBD      | TBD      | TBD    | TBD    |

### Block B — NO-SPEC context scaling (fi_cutlass MoE, full-CG) — Cases 03–04

| #  | spec | ctx     | json_override                         | Status | n=1 tok/s | n=8 peak | n=8 ok | Output |
|----|------|---------|---------------------------------------|--------|----------:|---------:|--------|--------|
| 03 | off  | 524288  | `{"max_position_embeddings":524288}`  | TBD    | TBD       | TBD      | TBD    | TBD    |
| 04 | off  | 1048576 | `{"max_position_embeddings":1048576}` | TBD    | TBD       | TBD      | TBD    | TBD    |

### Block C — NO-SPEC fp4_gemm delta (fi_cutlass MoE, full-CG, ctx262k) — Case 05

| #  | fp4_gemm | Status | Note (does the dev image ship cuDNN-FP4 now?) |
|----|----------|--------|------------------------------------------------|
| 05 | fi_cudnn | TBD    | TBD — on 0.5.12 this crashed (`cuDNN not available`) |

### Block D — NO-SPEC MoE runner PROBE (full-CG, ctx262k) — Case 06

| #  | moe_runner | Status | Note (triton crashed on 0.5.12 — `cutlass_moe_fp4` shape assert) |
|----|------------|--------|------------------------------------------------------------------|
| 06 | triton     | TBD    | TBD                                                              |

### Block E — NO-SPEC piecewise-CG PROBE (fi_cutlass MoE, ctx262k) — Case 07

| #  | CG variant | Status | n=1 tok/s | n=8 peak | n=8 ok | Output |
|----|------------|--------|----------:|---------:|--------|--------|
| 07 | piecewise  | TBD    | TBD       | TBD      | TBD    | TBD    |

### Block F — MTP / EAGLE draft-depth sweep (fi_cutlass, full-CG, ctx262k) — Cases 08–11

> Speedup read against Block-A Case 02 (no-spec, same image). **accept_len > 1 required** (R3).

| #  | steps | draft | Status | accept_len | n=1 tok/s | n=1 vs 02 | n=8 peak | n=8 ok | NaN? (R2) | Output |
|----|------:|------:|--------|-----------:|----------:|----------:|---------:|--------|-----------|--------|
| 08 | 1     | 2     | TBD    | TBD        | TBD       | TBD       | TBD      | TBD    | TBD       | TBD    |
| 09 | 3     | 4     | TBD    | TBD        | TBD       | TBD       | TBD      | TBD    | TBD       | TBD    |
| 10 | 5     | 5     | TBD    | TBD        | TBD       | TBD       | TBD      | TBD    | TBD       | TBD    |
| 11 | 5     | 7     | TBD    | TBD        | TBD       | TBD       | TBD      | TBD    | TBD       | TBD    |

- **08** = MTP boot litmus (smallest draft) · **10** = current cookbook recipe (5/5) ·
  **11** = TRT SPEED-Bench accept-3.45 point (draft-len 7).

### Block G — MTP serving-context + robustness PROBES (cookbook 5/5) — Cases 12–14

| #  | variant                          | ctx    | Status | accept_len | n=1 tok/s | n=8 peak | n=8 ok | Output |
|----|----------------------------------|--------|--------|-----------:|----------:|---------:|--------|--------|
| 12 | serving memory                   | 524288 | TBD    | TBD        | TBD       | TBD      | TBD    | TBD    |
| 13 | `mamba_scheduler_strategy=extra_buffer` | 262144 | TBD | TBD   | TBD       | TBD      | TBD    | TBD    |
| 14 | `enable_spec_v2=true`            | 262144 | TBD    | TBD        | TBD       | TBD      | TBD    | TBD    |

- **12** = does MTP draft buffers + 512K KV co-fit (memory). **13** = manual `extra_buffer`
  (#27998 auto-selects it for spec+radix — verify it doesn't regress/reject). **14** =
  spec-v2 path (#27184) — NOT in the base recipe, PROBE only.

### Column legend

| Column     | Description                                                                                                     |
|------------|-----------------------------------------------------------------------------------------------------------------|
| spec       | `speculative_enabled` — off (no-spec baseline) / EAGLE MTP                                                       |
| accept_len | mean accepted draft tokens per step (from boot/decode log) — **> 1 = MTP paying off; ≈ 1 = loader bug (R3)**     |
| n=1 vs 02  | single-stream speedup vs the no-spec Case 02 on this image (the MTP denominator)                                 |
| n=N peak   | **peak** throughput = Σ per-request tok/s over the *successful* requests (NOT aggregate total_tokens/wall_time) |
| NaN? (R2)  | did the NVFP4 MTP target-logits path emit NaN at boot/decode? (`yes` = the #27828 gap)                          |

---

## Detailed results (n=8)

_TBD — fill after the run (mirror the 0.5.12 log's per-case n=8 table: finish reasons, TTR_min, avg_ttft, output spot-check)._

## Crash details

_TBD — record any startup_crash signatures (esp. Cases 05/06 re-probes, and any MTP NaN/assert)._

---

## Findings

_TBD after the run. Pre-registered questions to answer:_

1. **R1 — does the arch still boot on 0.5.13-dev-nemotronh-mtp?** (Case 01)
2. **Q1 — non-MTP image delta:** is the no-spec winner faster/slower than 0.5.12's 135 tok/s
   @ ctx524k? Did Cases 05 (fi_cudnn) / 06 (triton) change verdict on the new image?
3. **R3 — does MTP pay off?** accept_len across Cases 08–11; best draft depth; n=1 speedup vs
   Case 02. Did #27998's loader fix land in this image (accept_len > 1)?
4. **R2 — any NVFP4 MTP NaN logits?** (#27828)
5. **Memory:** does MTP + 512K KV co-fit at ctx524k (Case 12)?
6. **extra_buffer / spec_v2 probes** (Cases 13/14) — do they help, regress, or reject?
7. **Best overall serving shape** (no-spec winner vs best MTP config) and the resulting
   profile recommendation.

## Production recommendation

_TBD — to be written against the winner (no-spec baseline vs best MTP draft depth)._

## Action items / follow-ups

- [ ] Run the matrix; start with Case 01 (boot litmus), then Case 08 (first MTP litmus — watch R2/R3).
- [ ] Fill every `TBD` cell + the Findings / Detailed-results / Crash sections.
- [ ] Compare every number against the 0.5.12 TESTLOG (image delta) and the EP=4 sibling run.
- [ ] If MTP pays off, update the profile MTP block (it already has `speculative_enabled=true`)
      with the validated draft depth + accept_len; if not, document why and flip it back to false.
- [x] Driver/kernel/OS/K3s verified on spark1–4 (2026-06-16); driver 580.159.03 matches the `nv580.159` filename label. Still TBD: NCCL version (from the boot log of the actual run).
