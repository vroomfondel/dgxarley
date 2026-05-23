# SGLang Test Log — Qwen3.6 35B-A3B-NVFP4 (mmangkad / modelopt), 4 Nodes, TP=4 EP=1, v0.5.12

## Environment

| Component | Value                                                                       |
|-----------|-----------------------------------------------------------------------------|
| GPU       | NVIDIA GB10 (SM121/Blackwell-Consumer), 128 GB per node                     |
| Driver    | 580.159                                                                     |
| CUDA      | 13.2 host / 13.0 image (PR #21498)                                          |
| Kernel    | 6.17.0-1018-nvidia                                                          |
| OS        | Ubuntu 24.04 LTS (aarch64)                                                  |
| K3s       | v1.35.3+k3s1                                                                |
| Nodes     | spark1, spark2, spark3, spark4 (1 GPU each)                                 |
| Image     | `scitrera/dgx-spark-sglang:0.5.12` (Block A/B/E/F) / `xomoxcc/dgx-spark-sglang:0.5.12-cudnn` (cuDNN-FP4 cases per-case override) |
| Model     | `mmangkad/Qwen3.6-35B-A3B-NVFP4` (**modelopt_fp4** quantization)            |
| NCCL      | 2.29.7+cuda13.2 (dgxspark-3node-ring)                                       |
| Transport | **RoCE** via SR-IOV VF                                                      |
| AllReduce | Legacy (both `SGLANG_USE_JIT_ALL_REDUCE=0` + `SGLANG_OPT_..._V2=0`)         |

Matrix file: `kikube/matrixtest_matrices/sglang_nn4_tp4_ep1/qwen-3.6-35b-a3b-nvfp4-mmangkad/nv580.159_sglang-0.5.12_qwen-3.6-35b-a3b-nvfp4-mmangkad_n4_ep1.yaml`

Sister testlog: `TESTLOGS/sglang_nn4_tp4_ep1/qwen-3.6-35b-a3b-nvfp4/TESTLOG_nv580.159_sglang-0.5.12_qwen-3.6-35b-a3b-nvfp4_4n.md` (RedHatAI / `compressed-tensors` variant, 55 cases).

## Why this matrix exists — complementary kernel coverage

The RedHatAI sister matrix (`compressed-tensors` packaging) routes ALL four "non-trtllm" `moe_runner_backend` settings through the same dispatch at `compressed_tensors_w4a4_nvfp4_moe.py:apply_weights`:

```python
if self.use_flashinfer_trtllm: trtllm_fp4_block_scale_moe(...)
else:                          cutlass_moe_fp4(...)
```

The `create_moe_runner` call there is hardcoded to `MoeRunnerBackend.TRITON`, so `triton`, `flashinfer_cutlass`, `flashinfer_cutedsl`, and `cutlass` (direct) all end up at the same kernel (`cutlass_moe_fp4`). The RedHatAI matrix kept Blocks B/D mostly to confirm this empirically — they were dispatch-path duplicates, not separate kernels.

The mmangkad variant is **modelopt_fp4**-packed and routes through `modelopt_quant.py` → `MoeRunnerBackend.<actual>` — i.e. the `moe_runner_backend` knob actually selects distinct kernels here. **This is the matrix where `triton` vs `fi_cutlass` vs `fi_cutedsl` vs `cutlass` are real, non-aliased kernel paths.**

Coverage focus (13 cases vs RedHatAI's 55) is deliberately narrow:
- Drop no-cuda-graph variants (FP8 sweeps already proved eager is 2–3.5× slower)
- Drop MTP s=4/5 (FP8 + RedHatAI both put the sweet spot at s=2; regression at s=3+)
- Drop redundant cutlass-direct sweep (RedHatAI Block C covered it; one anchor here is enough)
- Pick up the two previously-cut cross-products: `fi_cutedsl × cuDNN-FP4` and `fi_trtllm × cuDNN-FP4` (both deemed "too experimental" originally — modelopt is where they have a chance of mattering)

## Image policy

Default image is `scitrera/dgx-spark-sglang:0.5.12`. Cases that set `fp4_gemm_backend: flashinfer_cudnn` (Tests 07–10) override per-case to `xomoxcc/dgx-spark-sglang:0.5.12-cudnn` — without this, the cuDNN-FP4 path would crash 4/4 at startup (the upstream scitrera image is cuDNN-less, confirmed by RedHatAI Block A Tests 07–12 + Block C Tests 31–36 = 12/12 crashes on the cuDNN-FP4 GEMM `RuntimeError: cuDNN is not available`).

## Configuration Matrix

All tests use: `tp=4, pp=1, ep=1, nccl_transport=roce, kv_cache_dtype=fp8_e4m3, mem_fraction_static=0.50, disable_deep_gemm=true, context_length=262144, num_experts=256, enable_eplb=false` unless noted. Quantization: `modelopt_fp4` (whereas RedHatAI sibling is `compressed-tensors`).

### Block A — REAL triton MoE kernel — Tests 01–02

Under `compressed-tensors`, `moe_runner_backend: triton` is hardcoded to `cutlass_moe_fp4`. Under modelopt, this is the actual triton MoE kernel.

| #  | moe_runner | attention | fp4_gemm     | dis_cuda_graph | dis_piecewise | Status | n=1 tok/s | n=4 peak | n=8 peak |
|----|------------|-----------|--------------|----------------|---------------|--------|----------:|---------:|---------:|
| 01 | triton     | fi        | fi_cutlass   | false          | false         | ok     |     66.35 |   232.05 |   394.85 |
| 02 | triton     | triton    | fi_cutlass   | false          | false         | ok     |     59.70 |   234.28 |   389.21 |

### Block B — REAL flashinfer_cutlass MoE kernel — Tests 03–04

Under `compressed-tensors`, `moe_runner_backend: flashinfer_cutlass` is an alias for `cutlass_moe_fp4` AND the SGLang pre-check whitelist rejected `compressed-tensors` upstream → 12/12 startup_crash on RedHatAI Block B. Under modelopt, the whitelist passes and a real fi_cutlass MoE implementation runs.

| #  | moe_runner  | attention | fp4_gemm     | dis_cuda_graph | dis_piecewise | Status | n=1 tok/s | n=4 peak | n=8 peak |
|----|-------------|-----------|--------------|----------------|---------------|--------|----------:|---------:|---------:|
| 03 | fi_cutlass  | fi        | fi_cutlass   | false          | false         | ok     |     67.98 |   248.43 |   405.04 |
| 04 | fi_cutlass  | triton    | fi_cutlass   | false          | false         | ok     |     63.10 |   248.64 |   406.68 |

### Block C — REAL flashinfer_cutedsl MoE × full fp4_gemm sweep — Tests 05–08

Includes the `fi_cutedsl × fi_cudnn-fp4` cross (previously cut as "too experimental" — modelopt is where it has a chance of mattering).

| #  | moe_runner  | attention | fp4_gemm     | dis_cuda_graph | dis_piecewise | Status      | n=1 | n=4 | n=8 |
|----|-------------|-----------|--------------|----------------|---------------|-------------|-----|-----|-----|
| 05 | fi_cutedsl  | fi        | fi_cutlass   | false          | false         | **crash A** | —   | —   | —   |
| 06 | fi_cutedsl  | triton    | fi_cutlass   | false          | false         | **crash A** | —   | —   | —   |
| 07 | fi_cutedsl  | fi        | fi_cudnn     | false          | false         | **crash A** | —   | —   | —   |
| 08 | fi_cutedsl  | triton    | fi_cudnn     | false          | false         | **crash A** | —   | —   | —   |

### Block D — fi_trtllm MoE × cuDNN-FP4 — Tests 09–10

The trtllm × cuDNN-FP4 cross was previously cut on the assumption that TRT-LLM consumes weight-scales via its own shuffled-FP4 pipeline. Tested here to confirm or refute.

| #  | moe_runner | attention | fp4_gemm  | dis_cuda_graph | dis_piecewise | Status | n=1 | n=4 | n=8 |
|----|------------|-----------|-----------|----------------|---------------|--------|-----|-----|-----|
| 09 | fi_trtllm  | fi        | fi_cudnn  | false          | false         | **crash C** | —   | —   | —   |
| 10 | fi_trtllm  | triton    | fi_cudnn  | false          | false         | **crash C** | —   | —   | —   |

### Block E — cutlass-direct MoE anchor (modelopt-native) — Test 11

Single case as direct A/B against RedHatAI Block C cutlass-direct (which routed through the same kernel via the dispatcher hack). Uses winner-shape from `nvidia/Qwen3.5-397B-A17B-NVFP4` 0.5.10 Test 28 (triton-attn + fi_cutlass-fp4 GEMM + CG on / piecewise off).

| #  | moe_runner | attention | fp4_gemm     | dis_cuda_graph | dis_piecewise | Status | n=1 | n=4 | n=8 |
|----|------------|-----------|--------------|----------------|---------------|--------|-----|-----|-----|
| 11 | cutlass    | triton    | fi_cutlass   | false          | true          | **crash B** | — | — | — |

### Block F — MTP NEXTN on hypothesized winner shape — Tests 12–13

Anchors on the cutlass-direct + triton-attn + fi_cutlass-fp4 + CG on shape (Block E baseline) with NEXTN s=2 and s=3. The interesting question: can modelopt + cutlass-direct + MTP beat the RedHatAI Test 45 winner (438.07 tok/s peak at n=8, triton-MoE + fi-attn + piecewise CG + MTP s=2)?

| #  | moe_runner | attention | fp4_gemm     | dis_cuda_graph | dis_piecewise | spec      | Status | n=1 | n=4 | n=8 |
|----|------------|-----------|--------------|----------------|---------------|-----------|--------|-----|-----|-----|
| 12 | cutlass    | triton    | fi_cutlass   | false          | true          | NEXTN s=2 | **crash B** | — | — | — |
| 13 | cutlass    | triton    | fi_cutlass   | false          | true          | NEXTN s=3 | **crash B** | — | — | — |

### Column Legend

| Column         | Description                                                                                                                    |
|----------------|--------------------------------------------------------------------------------------------------------------------------------|
| moe_runner     | `moe_runner_backend` — `triton`, `flashinfer_cutlass` (`fi_cutlass`), `flashinfer_cutedsl` (`fi_cutedsl`), `flashinfer_trtllm` (`fi_trtllm`), `cutlass` (direct) |
| attention      | `attention_backend` — `fi` = FlashInfer, `triton` = Triton                                                                     |
| fp4_gemm       | `fp4_gemm_backend` — `fi_cutlass` = `flashinfer_cutlass`, `fi_cudnn` = `flashinfer_cudnn`                                      |
| dis_cuda_graph | `disable_cuda_graph` — true = eager, false = capture CUDA graphs                                                               |
| dis_piecewise  | `disable_piecewise_cuda_graph` — true = fixed-BS graphs only, false = piecewise variable-length graphs                         |
| spec           | speculative decoding — `NEXTN s=N` = MTP with `speculative_num_steps=N`, `eagle_topk=1`, `num_draft_tokens=N+1`                |

### Crash Legend

- **crash A** (fi_cutedsl-JIT-architecture): `RuntimeError: No supported CUDA architectures found for major versions [10].` at `flashinfer/compilation_context.py:95` during `gen_moe_utils_module()`. FlashInfer Cute-DSL JIT asks for SM major version 10 (Blackwell-Datacenter family: sm_100/sm_103/sm_103a) and finds nothing — GB10 is **SM 12.1** (Blackwell-Consumer). Same architecture-level mismatch class as fi_trtllm's `sm100f` kernel (see RedHatAI sister testlog Block F). Not config-fixable from YAML; needs an upstream Cute-DSL codepath for SM121.
- **crash B** (modelopt-cutlass-not-implemented): `NotImplementedError: Unsupported runner backend: MoeRunnerBackend.CUTLASS` at `sglang/srt/layers/moe/moe_runner/runner.py:66`, called from `modelopt_quant.py:2014 create_moe_runner`. SGLang's modelopt MoE path does NOT implement `cutlass`-direct in its `MoeRunner` dispatch — `triton`, `flashinfer_cutlass`, `flashinfer_cutedsl`, `flashinfer_trtllm` are wired up, `cutlass` is missing. **This is an SGLang code-coverage gap, not hardware-related**: under `compressed-tensors` the dispatcher hack at `compressed_tensors_w4a4_nvfp4_moe.py:apply_weights` short-circuited `MoeRunner` entirely and routed straight to `cutlass_moe_fp4`, so `cutlass`-direct worked. Under modelopt there is no such short-circuit. Affects Tests 11, 12, 13 — kills the entire MTP / Block-F sweep on modelopt.
- **crash C** (fi_trtllm-sm100f-kernel): `RuntimeError: Error in function 'run' at /workspace/csrc/trtllm_batched_gemm_runner.cu:278: Error occurred when running GEMM! Kernel: bmm_E2m1_..._sm100f`. trtllm GEMM kernel selector picks an `sm100f`-suffixed Blackwell-Datacenter kernel with no SM121 fallback — same root cause as RedHatAI Block F (Tests 49–54). Tests 09/10 are the cuDNN-FP4 cross-product variant; the cuDNN-vs-fi_cutlass FP4 GEMM choice is irrelevant when trtllm itself can't dispatch.

---

## Results

**Matrix run complete (2026-05-23).** 13/13 cases attempted. **4 ok (Tests 01–04), 9 crash (Tests 05–13).**

### Completed `ok` cases

| #  | Config                                                  | n=1 tok/s | n=4 peak | n=8 peak | n=8 avg/req | n=8 ok | Finish reasons | n=8 TTR_min | Output  |
|----|---------------------------------------------------------|----------:|---------:|---------:|------------:|--------|----------------|------------:|---------|
| 01 | triton-moe + fi-attn + fi_cutlass-fp4 + piecewise       |     66.35 |   232.05 |   394.85 |       49.36 | 8/8    | length×8       |       0.733 | clean ✓ |
| 02 | triton-moe + triton-attn + fi_cutlass-fp4 + piecewise   |     59.70 |   234.28 |   389.21 |       48.65 | 8/8    | length×8       |       0.641 | clean ✓ |
| 03 | fi_cutlass-moe + fi-attn + fi_cutlass-fp4 + piecewise   |     67.98 |   248.43 |   405.04 |       50.63 | 8/8    | length×8       |       0.695 | clean ✓ |
| 04 | fi_cutlass-moe + triton-attn + fi_cutlass-fp4 + piecewise |   63.10 |   248.64 |   406.68 |       50.84 | 8/8    | length×8       |       0.645 | clean ✓ |

### Crash summary

| Block | Cases | Status | Root cause |
|-------|-------|--------|------------|
| C (fi_cutedsl MoE × FP4-GEMM sweep) | 05–08 | **4/4 crash A** | Cute-DSL JIT asks for sm major 10 (Blackwell-DC), GB10 is SM 12.1 |
| D (fi_trtllm × cuDNN-FP4) | 09–10 | **2/2 crash C** | trtllm GEMM picks `sm100f` (Blackwell-DC), no SM121 fallback. cuDNN-vs-fi_cutlass FP4 GEMM choice is moot — trtllm itself can't dispatch |
| E (cutlass-direct anchor) | 11 | **crash B** | `NotImplementedError: Unsupported runner backend: MoeRunnerBackend.CUTLASS` in `modelopt_quant.py:2014 → moe_runner/runner.py:66` — SGLang code gap on modelopt path |
| F (cutlass-direct + MTP s=2/3) | 12–13 | **2/2 crash B** | Same as Test 11. Kills the entire MTP sweep on modelopt — no MTP datapoint available |

### Findings

1. **Real `fi_cutlass` MoE kernel runs cleanly on modelopt — first such datapoint** (Tests 03/04). The path that crashed 12/12 on RedHatAI's pre-check whitelist (`Invalid quantization 'compressed-tensors'. FlashInfer Cutlass MOE supports only modelopt_fp4...`) now passes the check and runs end-to-end. **n=8 peak 405–407 → ~3 % faster than Block A (triton-MoE) at 389–395 tok/s**, consistent across both `fi-attn` and `triton-attn` pairings. Modest but real — and a kernel that simply could not be measured on the compressed-tensors path at all.
2. **`fi_cutedsl` MoE is dead on GB10 regardless of model packaging.** Cute-DSL's compilation context queries SM major 10 (Blackwell-Datacenter sm_100/103) and finds nothing → `RuntimeError: No supported CUDA architectures found for major versions [10].` 4/4 crash on modelopt, 6/6 crash on compressed-tensors (RedHatAI Block D). **Architecture-level mismatch** — needs an upstream SM121 codepath in `flashinfer/compilation_context.py` and `cute_dsl/moe_utils.py`. The Cute-DSL × cuDNN-FP4 cross (the previously-cut "too experimental" combination) was tested here for the first time — same crash, same fault, the cuDNN-FP4 choice never matters because fi_cutedsl can't even JIT.
3. **`fi_trtllm` × cuDNN-FP4 cross**: also dead — Tests 09/10 hit the same `sm100f` kernel mismatch as RedHatAI Block F. **The FP4-GEMM backend choice is irrelevant** when trtllm itself can't dispatch on this hardware, confirming the original matrix-design assumption that this cross would be a no-op.
4. **`cutlass`-direct MoE is unusable on the modelopt code path** — `NotImplementedError: Unsupported runner backend: MoeRunnerBackend.CUTLASS` at `sglang/srt/layers/moe/moe_runner/runner.py:66`. This is a **pure SGLang code-coverage gap**, not a hardware issue: under `compressed-tensors` the dispatcher hack at `compressed_tensors_w4a4_nvfp4_moe.py:apply_weights` short-circuits `MoeRunner` and routes directly to `cutlass_moe_fp4` — that's why RedHatAI Block C worked. Under modelopt that short-circuit doesn't exist; the `MoeRunner.__init__` `if/elif` chain handles `triton`, `flashinfer_cutlass`, `flashinfer_cutedsl`, `flashinfer_trtllm`, and falls through to `NotImplementedError` for `cutlass`. **Kills Tests 11/12/13 — there is no MTP datapoint on modelopt at all.**
5. **Best mmangkad/modelopt n=8 peak: Test 04 at 406.68 tok/s** (fi_cutlass-MoE + triton-attn + fi_cutlass-fp4 + piecewise CG). Compared to the active production champion RedHatAI Test 45 at 438.07 (triton-MoE + fi-attn + piecewise CG + **MTP s=2**) → **−7.1 %**. The Δ is not "modelopt is slower"; it's "modelopt couldn't run any MTP case". No-MTP-vs-no-MTP comparison: Test 04 (406.68) ≈ RedHatAI Test 01 (407.25) ≈ RedHatAI Test 06 (403.03). At parity, in other words.
6. **Output quality clean** across the 4 ok cases. TTR_min 0.641–0.733, all 8/8 `length`, no salad triggers. modelopt-packed `mmangkad` produces text at the same quality level as RedHatAI/compressed-tensors.

### Verdict

**Modelopt offers nothing usable that compressed-tensors doesn't.** Of the four MoE backends the modelopt path was supposed to unlock as "real, non-aliased kernels":

| MoE backend  | Modelopt | compressed-tensors | Comment |
|--------------|----------|--------------------|----------|
| `triton`       | works    | works (aliased)    | same speed |
| `fi_cutlass`   | works (NEW) | crashes pre-check | +3 % vs triton on modelopt — net win is small |
| `fi_cutedsl`   | broken (sm10 JIT) | broken (sm10 JIT) | architecture-level, packaging-agnostic |
| `cutlass`-direct | broken (code gap) | works via dispatcher hack | the critical loss — kills MTP cutlass-direct exploration |

And critically, modelopt **loses access to MTP** on `cutlass`-direct (the shape we wanted to test) because of the code gap. The MTP win on RedHatAI Test 45 (438.07) used `triton`-MoE — and on modelopt that triton path delivers ~394 vs RedHatAI's ~407, so even with MTP added on top, modelopt would likely land around 420, still below Test 45.

**Production profile stays on RedHatAI Test 45 shape.** `redhatai-qwen3.6-35b-a3b-nvfp4.yml` remains the right choice. The mmangkad matrix did its complementary-coverage job: it tells us the modelopt path is a dead end for further throughput exploration on this model + this hardware.

(Re-run via `kikube-bench matrix matrixtest_matrices/sglang_nn4_tp4_ep1/qwen-3.6-35b-a3b-nvfp4-mmangkad/nv580.159_sglang-0.5.12_qwen-3.6-35b-a3b-nvfp4-mmangkad_n4_ep1.yaml`.)

---

## Action items after the matrix run

- [ ] Fill the four block tables with actual results (Tests 08–13)
- [ ] Verify output quality on every `ok` case (pattern-grep + token-distribution + tail-eyeball)
- [ ] Compute Δ vs RedHatAI sister matrix on the matching shapes — pick the better-throughput model for the production NVFP4 profile
- [ ] If Block F (MTP on cutlass-direct) beats RedHatAI Test 45 (438.07): create a `mmangkad-qwen3.6-35b-a3b-nvfp4.yml` model profile entry with the Block F winner shape; otherwise keep `redhatai-qwen3.6-35b-a3b-nvfp4.yml` as the active NVFP4 profile
- [ ] Document fi_cutedsl SM121 incompatibility in CLAUDE.md (alongside the existing fi_trtllm `sm100f` note) so future matrices skip those Blocks on GB10 hardware
