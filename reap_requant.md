# REAP Requant: selective attention NVFP4 (W4A16) for glm-5.2-reap-504B-v2, plan

Plan, not yet started. Written 2026-07-16. Companions:
`../kikube/quantizer/configs/glm-5.2-reap-504b-v2-attn-selective_nvfp4.yaml`
(the quant-config draft with the same goal; THIS document is the operational
project plan with the go/no-go gates), `dsalogitrework.md` (p35 LIVE RESULT = the
profiling motivation), `DSA_speedup.md`, memory
`reference_glm52_dsa_indexer_deepgemm_sm121`.

## 0. Context for a fresh session (everything needed to start HERE)

**What's running:** `0xSero/glm-5.2-reap-504B-v2` (GlmMoeDsaForCausalLM = MLA + DSA,
NVFP4/modelopt, 168 REAP experts, 78 layers + 1 MTP) on the dgxarley cluster:
4× DGX Spark GB10 (**SM121**, consumer Blackwell, 128 GB unified, ARM64), TP4,
image `xomoxcc/dgx-spark-sglang:0.5.15-sm121`. Profile:
`roles/k8s_dgx/model_profiles/0xsero-glm-5.2-reap-504b-v2.yml`, which holds the
PATCH-ACTIVATION CONTRACT (attention_backend dsa + dsa_*_backend trtllm +
dsa_paged_mqa_logits_backend torch + dsa_indexer_triton). Deploy:
`ansible-playbook k8s_dgx.yml --tags sglang` (NEVER without explicit user
approval; `kubectl --context=ht@dgxarley`, local). Endpoint:
`https://sglang.dgx.elasticc.io/v1`.

**Patch stack (prior work, ESSENTIAL):** SGLang is patched at runtime via
source patches in `roles/k8s_dgx/files/sglang_patches/p<NN>_*.py`
(ConfigMap to `/patches`, runner in `sglang_launch.sh`, rules in `_patchlib.py`
plus `sglang_hy3_dsa_handoff.md`, **read the handoff first**). Relevant for THIS
project: `p21_mixed_nvfp4_dispatch.py` +
`p22_modelopt_mixed_nvfp4_variant.py` (the partial
MIXED_PRECISION/W4A16 support from the Qwen3.6 work, memory
`reference_qwen36_nvfp4_modelopt_mixed`; upstream references sglang PR #27906 +
#28099). p30/p35 (indexer) and p34 (native sparse attention) are the reason
the attention GEMVs are now the floor.

**GPU test environment without the cluster (standard method of all prior work):** spark5
(`ssh root@spark5.local`, NOT in k3s) has full GB10 access in podman:
`podman run --rm --device nvidia.com/gpu=all -v /root/patchtest:/patchtest
--entrypoint bash xomoxcc/dgx-spark-sglang:0.5.15-sm121 -c "python3 ..."`.
Under `/root/patchtest/` are the harness scripts of prior work as
templates (`validate_p34.sh`, `sm120_sparse_mla_test.py`, `sm120_perf.py`,
`triton_indexer.py`, `bench_indexer.py`, `run_idem.sh`). Apply the patch chain
before a test: copy the repo's `sglang_patches/` to spark5, then in the container
`for p in p[0-9][0-9]_*.py; do python3 $p; done`.

**Baselines (2026-07-16, measured live):** decode single-stream **8.4 tok/s**
(cuda-graph, DSA native; before MTP). GSM8K reference: 2-shot, n=20, concurrency 8,
temp 0, max_tokens 768 → **85% (p34) / 90% (p35)**, 0 errors. Harness:
simple OpenAI-client few-shot loop (`gsm8k_dsa.py`, env-driven:
N/NUM_SHOTS/CONCURRENCY/MAX_TOKENS/LABEL/BASE_URL, writes
`gsm8k_<LABEL>_summary.json`); last lived in the session tmp, rewritable in 10 min
if needed (datasets: `openai/gsm8k`, answer extraction via regex on
"answer is X").

**How the profiling numbers were produced (reproducible):** SGLang
`POST /start_profile {"output_dir":"/tmp/sglprof","num_steps":24,
"activities":["CPU","GPU"]}` against the head (port-forward on svc/sglang:8000),
generate load, parse the trace `*-TP-0.trace.json.gz` in the pod (sum kernel
events by name). GB10 quirk: `nvidia-smi` util is useless (memory
`reference_gb10_util_and_stuck_rank`).

**Checkpoint locations:** cluster HF cache on JuiceFS (`/mnt/jfs`,
USB-HDD backend, cold loads take hours, memory
`reference_juicefs_backend_usb_hdd`); recovery kit locally:
`~/hf_downloads/GLM-5.2-504B-REAP-recovery-kit/` (RECONSTRUCT.md = provenance
of all 0xSero-5.2 artifacts). Format ground truth for the modelopt-W4A16 packing:
`nvidia/Qwen3.6-35B-A3B-NVFP4` (a real MIXED_PRECISION export, read the
tensor/scale conventions from it).

**House rules that apply here:** no deploy/pod-delete without approval; no
HF/external push without approval; no GPU debug pod on spark1-4 while SGLang
serves (time-slicing → NCCL timeout); test pods `tail -f /dev/null`, never
label `app=sglang`; forward-fix, never image rollback.

## 1. Motivation (profiled, not estimated)

Live profiling of the decode step (2026-07-16, head, 22 steps, 88% GPU-busy,
~131 ms kernel time/token) after p34 (native sparse attention) + p35
(Triton indexer):

| ms/token | share | what |
|---|---|---|
| ~59 | ~45% | cuBLAS bf16 GEMV (166 µs × ~3.25/layer) + dsv3_fused_a_gemm + lm_head |
| ~12 | ~10% | small bf16 wmma GEMMs (kv_b / indexer projections) |
| ~27 | ~21% | NVFP4 MoE (grouped cutlass) |
| ~10 | ~8%  | NCCL AllReduce (TP4) |
| ~3  | ~2.5% | sparse attention (p34) |

The **unquantized bf16 MLA projections are the floor** (~71 ms/token,
pure weight bandwidth at bs=1). The byte chunks per layer: `o_proj`
(16384×6144 ≈ 100M params) and `q_b_proj` (2048×16384 ≈ 34M); `q_a` (12.6M),
`kv_a` (3.5M), `kv_b` (~15M) are small AND quantization-sensitive
(low-rank compressions of the MLA).

**Goal:** `o_proj` + `q_b_proj` (≈80% of the attention bytes) to NVFP4
**weight-only (W4A16)** → expected decode ~78 ms/token ≈ **~12-13 tok/s instead of
8.4 (~1.5×)**, multiplicative with MTP.

## 2. Why W4A16 instead of W4A4, and why directly on the published v2

- **W4A16 = data-free.** Only weight rounding (fp4 packed + block scales),
  NO calibration (that would only be needed for the activation quantization of
  W4A4). The decode gain comes 100% from weight bandwidth;
  leaving activations bf16 additionally avoids quality risk and is not worth it
  at bs=1 GEMV performance-wise either (quant overhead per step).
- **The BF16 attention in the published `0xSero/glm-5.2-reap-504B-v2` IS the
  KD recovery.** Recovery-kit finding (2026-07-16, local under
  `~/hf_downloads/GLM-5.2-504B-REAP-recovery-kit/`): v2 = `GLM-5.2-504B-Nvidia`
  (NVFP4 base) + router-KD-v2 gates + logit-KD-LoRA (r16/α32), and the LoRA
  targets exactly `q_a_proj, q_b_proj, kv_a_proj_with_mqa, kv_b_proj, o_proj`,
  merged into the BF16 attention weights. Any rebuild from a base
  would lose the LoRA (re-apply scripts are in the kit, unnecessary work).
  A public BF16 REAP base no longer exists as of 2026-07-16
  (`...BF16-REAP-keep168-unified` = 401 private/gone; `GLM-5.2-504B` is itself
  NVFP4 today, 309 GB).
- **No capacity problem:** the shard rewrite streams tensor by tensor, never the
  whole model in memory. Runs on spark5 or the workstation.

## 3. Target format

modelopt MIXED_PRECISION convention (precedent `nvidia/Qwen3.6-35B-A3B-NVFP4`
= W4A16_NVFP4; memory `reference_qwen36_nvfp4_modelopt_mixed`):

- Per quantized matrix: `weight` (fp4, 2 values/byte, uint8 container),
  `weight_scale` (fp8 block scales, block size 16), `weight_scale_2`
  (fp32 global). NO `input_scale` (W4A16).
- `config.json:quantization_config.ignore`: replace the per-layer globs
  `model.layers.N.self_attn*` with the explicit list WITHOUT
  `o_proj`/`q_b_proj` (i.e. still in: `q_a_proj`, `q_a_layernorm`,
  `kv_a_proj_with_mqa`, `kv_a_layernorm`, `kv_b_proj`, `*indexer*`).
  Layers 0-2 (dense MLP), `*shared_experts*`, `lm_head`, layer 78 (MTP)
  stay ignored UNCHANGED. `hf_quant_config.json` (if present) likewise.
- Regenerate `model.safetensors.index.json` (tensor names + sizes change
  in the affected shards).

## 4. Phases with go/no-go gates

### Phase 0 — W4A16 GEMV measurement on spark5 (THE decision gate)

Podman + GPU on spark5 (method like the p34/p35 validation, no cluster):

1. Identify which kernel SGLang dispatches for modelopt-W4A16 linear on SM121
   (the Qwen3.6 path; our p21/p22 patches are the entry point,
   upstream references PR #27906 + #28099).
2. Microbench at bs=1 (GEMV) with the real per-rank shapes (TP4):
   `o_proj` [6144 × 4096], `q_b` [4096 × 2048] (input/output dim depending on
   the shard axis), each W4A16-NVFP4 vs. bf16-cuBLAS baseline (the measured
   166 µs class calls).
3. **GO criterion: ≥2.5× vs. bf16 GEMV** (theoretically 4×; below 2.5× the
   dequant overhead eats the gain and the project isn't worth it → instead
   ask 0xSero for a selective export or drop it).
4. Document as a side finding: behavior at bs=8/32 (concurrency regime).

### Phase 0 — RESULT 2026-07-16: GO (measured on GB10)

Synthetic GEMV bench on spark5 (bf16-cuBLAS vs. the served NVFP4 path:
flashinfer.fp4_quantize weight offline + activation dynamically per call +
cutlass_scaled_fp4_mm; timing INCL. activation quant). Per-rank TP4 shapes:

| shape | bs=1 | bs=8 | bs=32 |
|---|---|---|---|
| o_proj (out 6144, in 4096) | **5.58x** | 3.15x | 3.86x |
| q_b (out 4096, in 2048) | **2.89x** | 0.91x | 1.22x |

**GO** (both bs=1 > 2.5x). Two qualifications: (1) the gain is
SINGLE-STREAM (bs=1 GEMV, memory-bound = our decode floor); q_b drops below
1x from bs=8 on (compute-bound + quant overhead), o_proj stays faster even at
batch. (2) mean_rel_err ~0.13 is on SYNTHETIC random weights with a
naive global scale, NOT a quality proof, the accuracy is only settled by the
GSM8K gate (Phase 3) on real, modelopt-calibrated weights. Bench script:
`$CLAUDE_JOB_DIR/tmp/gemv_phase0.py` (on spark5 under /root/quantwork/).

NB: the served path is W4A4 (activation dynamically fp4, no calibration
needed), not W4A16. At bs=1 the activation quant is a [1,in] tensor,
negligible, and the above already includes it. This simplifies Phase 1:
standard modelopt-NVFP4 on o_proj/q_b (dynamic input), no special W4A16 preset.

### Phase 0b — Kernel decision 2026-07-16: cutlass W4A4 (not Marlin W4A16)

After the GO, the kernel choice was measured (spark5, o_proj/q_b shapes, absolute
fp4 time over a bs sweep). Marlin W4A16 (the W4A16 path) collapses above
bs~64 (at bs=512 almost 3x slower than cutlass) and is unstable at bs=1;
cutlass W4A4 is stable and clearly superior from bs=128 on. → **cutlass W4A4
chosen** (user: don't optimize only bs=1).

Consequences: W4A4-dense supports NO dynamic (loader: "dynamic
quantization is not supported") → needs a static input_scale. Since 504B is too
large for full modelopt PTQ, **option 3 (user choice): HEURISTIC
input_scale**, no calibration run, GSM8K-gated. input_scale = generous
amax overestimate / (6*448) per module (overestimating = no clipping).

### Phase 1a — Format verified + script built 2026-07-16

The config path is SIMPLE: take o_proj/q_b out of quantization_config.ignore →
they fall into the existing W4A4 group_0 (same algo as the experts), NO
MIXED_PRECISION needed. Exact NVFP4 format verified against a real expert tensor:
weight uint8 [out,in//2], weight_scale float8_e4m3fn [out,in//16]
in LINEAR layout (is_sf_swizzled_layout=False, round-trip 0.089 vs swizzled
0.23), weight_scale_2 fp32 scalar = amax/(6*448). Data-free script:
`../kikube/quantizer/surgical_attn_nvfp4.py` (streams shards, quantizes only
o_proj+q_b, rest byte-identical, writes into a NEW directory; source is preserved).
NOT YET run (waits for rsync completion). Deploy + GSM8K = separate
approval-gated steps.

### Phase 1 — Shard rewrite script

`../kikube/quantizer/` (where the quant tooling lives): script
`surgical_attn_nvfp4.py`:

- Input: local snapshot of `0xSero/glm-5.2-reap-504B-v2` (lives in the
  cluster HF cache; copy to local disk for the rewrite).
- Streaming over the shards; for `model.layers.{3..77}.self_attn.{o_proj,q_b_proj}.weight`:
  NVFP4 block quant (16-blocks, amax-based, exactly the modelopt packing),
  replacing the bf16 tensor with the three quant tensors. Pass all other
  tensors through byte-identical.
- config/ignore/index updates (§3), plus a README stub with provenance.
- **Verification in the script:** per matrix a dequant round-trip against the
  bf16 original (log max relative error); spot-check comparison of the
  untouched tensors (hash).
- Output name suggestion: `glm-5.2-reap-504B-v2-attnq` (local; **no
  HF push without explicit approval**, house rule).
- Size expectation: ~294 GB → ~275 GB (o_proj+q_b: ~21 GB bf16 → ~5.3 GB).

### Phase 2 — Load/dispatch test on spark5 (without the cluster)

The full model doesn't fit on a single Spark → targeted loader test:

- Mini-harness in the container: parse `ModelOptFp4Config` with the new ignore
  list, instantiate ONE decoder layer (e.g. layer 10) and load its
  weights from the new shards → check param classes (o_proj/q_b as
  W4A16 quant params, q_a/kv_a/kv_b as bf16), forward on random data
  against the bf16 reference layer (tolerance: fp4 rounding level).
- If the mixed dispatch jams: extend p21/p22 (a known work area,
  Qwen3.6 precedent; possibly cherry-pick upstream PR #27906 fully).

### Phase 3 — Cluster deploy + validation (needs approval)

- Bring the model onto the cluster (mind the JuiceFS HDD cold load: ~275 GB ≈
  hours; plan a preload as with the original).
- New model profile (clone of `0xsero-glm-5.2-reap-504b-v2.yml`, own
  key, same DSA/MTP config).
- Yardstick, in this order:
  1. boot + smoke (coherence),
  2. **decode throughput vs. 8.4 tok/s** (or vs. the then-current
     MTP baseline, expectation ~1.5× on the non-MTP portion),
  3. **GSM8K 2-shot n=20 conc 8 vs. 85-90% baseline** (quality gate;
     on regression >5pp → abort, revert to the original),
  4. loop/attractor rate spot-check (REPORT.md guardrail: the pruned
     checkpoint has a 7.2% loop rate; `recommended_sampling` was tuned against
     dense attention and may need re-tuning).
- Rollback is trivial: profile back to the original repo (stays in the cache).

### Phase 3 — RESULT 2026-07-17: GO (deployed, gated, throughput confirmed)

Deploy + gates ran (4×GB10, TP4, DSA, MTP on). Two requant **config bugs**
had to be fixed first (both config-only, no re-quantization, below),
after which the model loads/serves cleanly (readyReplicas=1, 0 restarts):

- **GSM8K gate: PASS.** 5-shot, greedy, n=200: **93.0% flexible=strict** (186/200),
  **0 errors / 0 empty**. The heuristic input_scale holds (a broken W4A4 → <70%
  or garbage). Different harness cut than the 2-shot/n=20 reference (85-90%), so
  standalone go/no-go, not a paired A/B.
- **Decode throughput: higher, as intended.** Single-stream with MTP: **~18-29
  tok/s** (accept len ~3-4, prompt-dependent) vs. base reference ~11.7-12.4 tok/s
  (accept ~2.1). Requant-attributable (tok/s ÷ accept-len = raw forward-pass rate):
  **~15-18% faster forward pass** (o_proj/q_b W4A4 GEMV gain). The larger
  end-to-end delta includes higher MTP acceptance on the test prompts (base not
  re-measured on identical prompts) → ~15-18% is the clean requant-own number.
- **Loop/attractor gate (#4): still open** (recommended_sampling not re-swept).

**The two config bugs (SGLang loader traps; now fixed in `surgical_attn_nvfp4.py`
+ config.json patched on jfs/spark5/HF):**
1. `fused_qkv_a_proj_with_mqa`: SGLang fuses q_a_proj+kv_a_proj at runtime;
   its modelopt loader (`is_layer_excluded`) matches the `ignore` entries by
   **exact** set-membership on the last segment. A trailing `*` on
   `kv_a_proj_with_mqa*` misses the fused layer → it stays W4A4,
   checkpoint is BF16 → load assertion `[2624,3072]uint8` vs `[2624,6144]bf16`.
   Fix: emit the fused components WITHOUT `*`.
2. MTP layer 78: the rewrite also rewrote the non-requantized MTP layer
   (`--last-layer 77`) and set o_proj/q_b_proj to W4A4 →
   assertion `[4096,1024]uint8` vs `[4096,2048]bf16`. Fix: gate the rewrite on the
   requant range (out-of-range layers keep the wide `self_attn*` glob).
   Both validated deterministically against `is_layer_excluded` BEFORE the deploy (79
   layers, 0 mismatches).

## 5. Risks / open points

| risk | assessment |
|---|---|
| W4A16 GEMV kernel on SM121 slow/missing | THE gate (Phase 0); nothing further is worth it then |
| mixed-dispatch gaps in SGLang | known terrain (Qwen3.6, p21/p22); bounded effort |
| quality: quant rounding on KD-LoRA-bearing weights | limited to o_proj/q_b; GSM8K + loop gate in Phase 3 |
| modelopt packing details (scale layout, transposition) | verify in Phase 1 against a real modelopt-W4A16 export (Qwen3.6), don't guess |
| dsv3_fused_a_gemm (15 ms/token) stays bf16 | deliberate: that's the q_a/kv_a path (dense by design); remaining floor accounted for |
| hosting/distribution of the new checkpoint | local/cluster only; HF push only after approval |

## 6. Classification / ordering

First **evaluate MTP live** (running), then Phase 0. MTP and this project
multiply (~1.8× × ~1.5× ≈ 2.5-3× single-stream goal). Phase 0 is
half a spark5 day and decides everything further; Phases 1-2 are each
manageable; Phase 3 is a normal deploy cycle.
