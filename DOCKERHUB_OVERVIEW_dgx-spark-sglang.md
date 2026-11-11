<!-- short: SGLang for NVIDIA DGX Spark / GB10 (SM121): sm_121-pruned kernels, NVFP4, DSv4/GLM MTP, arm64 -->

# dgx-spark-sglang

Custom [SGLang](https://github.com/sgl-project/sglang) container images for the
**NVIDIA DGX Spark / ASUS Ascent GX10 (GB10, SM121, arm64)**.

Upstream SGLang / sgl-kernel binaries do not target `sm_121` and silently fall
back to JIT or to kernels that never run on a GB10. These images carry a stack
of patches against `sgl-kernel` that arch-prune the build to `sm_121` and strip
the Hopper-only kernels (FA3, sm90 targets, FlashMLA) it otherwise tries to
compile, plus per-release source patches for whatever the current SGLang tag
still gets wrong on this arch.

The original headline fix, the CUTLASS NVFP4 blockwise-MoE patch for the GB10's
101 KB shared-memory budget (device-side assert at
`nvfp4_blockwise_moe.cuh:78`), is **history as of `0.5.16-sm121`**: SGLang
[PR #30448](https://github.com/sgl-project/sglang/pull/30448) deleted that
kernel together with `cutlass_moe_fp4()` and now dispatches NVFP4 MoE to
FlashInfer-CUTLASS on SM100 and SM120/121 alike, which is the backend the model
profiles pin anyway. So the patch was not merged, it became unreachable, and it
is gated off (`APPLY_SGL_KERNEL_SM121=0`) on `0.5.16-sm121` and later. It is
still applied on `0.5.15.post1-sm121` and below.

For the DeepSeek-V4-Flash path the older tags also carry the then-unmerged
DeepSeek-V4 NVFP4 MoE support (upstream
[PR #25820](https://github.com/sgl-project/sglang/pull/25820), rebased onto
v0.5.13) so the `nvidia/DeepSeek-V4-Flash-NVFP4` checkpoint can be served on
SM121 at all; that one landed upstream and is native from v0.5.14 on. `sm_121`
remains a compute-capability tier the DeepGEMM / FlashMLA / cutlass kernel
ecosystem is only slowly shipping kernels for.

- **Source**: [github.com/vroomfondel/dgxarley](https://github.com/vroomfondel/dgxarley)
- **Hardware target**: NVIDIA GB10 / SM121 (DGX Spark, ASUS Ascent GX10) — arm64 only
- **License**: Apache-2.0 (same as SGLang)

## What's inside

- **SGLang** built from upstream tags (currently `v0.5.23`, which also covers v0.5.22)
- **sgl-kernel** with SM121 build patches: arch-prune to `sm_121` only, FA3 /
  sm90 targets / FlashMLA stripped (the bundled FlashMLA is Hopper-only). From
  `0.5.17-sm121` on all four are repathed, because SGLang RFC #29630 relocated
  the whole top-level `sgl-kernel/` tree to `python/sglang/kernels/aot/`, and
  `0.5.18-sm121` carries its own patch variant again: upstream reshaped the
  Blackwell gencode block in `aot/CMakeLists.txt` (the explicit `sm_103a`
  gencode is gone, replaced by `sm_100f` on CUDA 12.9+ with `sm_100a` as the
  pre-12.9 fallback), so the v0.5.17 arch-prune hunk no longer applies.
  `0.5.19-sm121` needs yet another variant, for a change that reads cosmetic and
  is not: upstream made the C++ standard a cached CMake variable, so every
  literal `-std=c++17` became `-std=c++${SGL_KERNEL_CXX_STANDARD}`, and that
  line is the trailing context of the arch-prune hunk, which therefore fails on
  v0.5.19. Through all of it the pruned result is unchanged: `sm_121a` on
  aarch64 plus `--compress-mode=size`.
  The CUTLASS NVFP4 blockwise-MoE patch (`StageCount<1>` +
  `KernelPtrArrayTmaWarpSpecialized`) is present only on `0.5.15.post1-sm121`
  and below, see above.
- **DeepSeek-V4 EAGLE-MTP marlin + TileLang indexer compat**, still patched on
  the current tags: `HybridFp8NvFp4Config.get_quant_method` has no marlin branch
  for the MTP/nextn draft-MoE layers, and TileLang's uint8 arg check still bites
  on the DSA indexer kernel.
- **DeepSeek-V4-Flash NVFP4 MoE (sm_121, experimental)** — the `0.5.13-sm121`
  tag carries upstream [PR #25820](https://github.com/sgl-project/sglang/pull/25820)
  ("DeepSeek-V4 NVFP4 MoE", unmerged at build time) rebased onto v0.5.13, so the
  `nvidia/DeepSeek-V4-Flash-NVFP4` checkpoint can be served on GB10. PR #25820
  is upstream-validated only on B200 (SM100) and default-routes to
  `flashinfer_trtllm_routed`, which is not runnable on SM121
  ([#26324](https://github.com/sgl-project/sglang/issues/26324)) — the model
  profile pins `flashinfer_cutlass` explicitly. **First-contact / unvalidated.**
  V4's sparse-decode path itself no longer needs a vendored kernel as of
  v0.5.13: upstream [PR #24692](https://github.com/sgl-project/sglang/pull/24692)
  ships a native SM120/121 Triton path (`major==12`, covers GB10), so the
  sm_121a-retargeted [`0xSero/deepseek-v4-flash-sm120`](https://github.com/0xSero/deepseek-v4-flash-sm120)
  kernel bake that earlier tags (`0.5.12.post1-sm121` and below) used is
  **dropped** here. Full wall-by-wall breakdown — DeepGEMM `hc_prenorm` +
  `paged_mqa_logits` torch fallbacks, `wo_a` fp8→bf16
  (`SGLANG_OPT_FP8_WO_A_GEMM=0`), `mem_fraction_static`, node swap for the load
  peak — in
  [`UPSTREAM_DSV4_BUGS.md`](https://github.com/vroomfondel/dgxarley/blob/main/UPSTREAM_DSV4_BUGS.md).
- **flashinfer pinned to `0.7.3`** on `0.5.23-sm121`, deliberately past upstream's
  own `0.7.2` pin and paired with **nvidia-cutlass-dsl `4.8.0`** (upstream main
  already pins that same 0.7.3 / 4.8.0 pair). `0.5.19-sm121` through
  `0.5.21-sm121` carry **`0.6.18.post1`** (`0.6.17` up to `0.5.18-sm121`). On
  `0.5.19-sm121` this pin is **not optional**: v0.5.19's own breaking-changes
  list flashinfer 0.6.18 as *required*, because
  [#36954](https://github.com/sgl-project/sglang/pull/36954),
  [#33237](https://github.com/sgl-project/sglang/pull/33237) and
  [#35120](https://github.com/sgl-project/sglang/pull/35120) removed the
  fallbacks for the pre-0.6.18 APIs behind `--dsa-topk-backend flashinfer` and
  the CuTe-DSL NVFP4 W4A16 mode. `.post1` is 0.6.18 plus one additive
  cherry-pick, i.e. the newest stable, which is the pin policy of this line.
  Paired with
  **nvidia-cutlass-dsl `4.6.2`** (`4.6.1` up to `0.5.17-sm121`; the bump follows
  upstream and its release notes credit it with fixing an FA4 startup
  regression on Blackwell). flashinfer only
  constrains cutlass-dsl `>=4.5.0` (open-ended), and NVIDIA has shipped
  internally-inconsistent `4.5.2`/`4.5.3` wheels that ICE on every fresh
  CuTe-DSL JIT compile at CUDA-graph warmup, so the explicit cutlass pin is not
  optional. What this pin buys on GB10:
  - [PR #3932](https://github.com/flashinfer-ai/flashinfer/pull/3932) fixes the
    **b12x FP4 quantization numerics** and adds `input_global_scale` to
    decouple weight from activation scale. Upstream's own summary is that W4A4
    serving on GB10 / SM120 / SM121 "now delivers the output quality its
    benchmark scores imply", with two NVFP4 quantization bugs fixed. This is
    the main reason for the `0.6.17` pin.
  - SM12x fused-MoE and FP4 GEMM resync:
    [#4253](https://github.com/flashinfer-ai/flashinfer/pull/4253) (`mm_fp4`
    SM120 NVFP4 dense GEMM to b12x HEAD),
    [#4130](https://github.com/flashinfer-ai/flashinfer/pull/4130) (cute SM120
    fused MoE, SwiGLU).
  - [#3897](https://github.com/flashinfer-ai/flashinfer/pull/3897) NVFP4
    attention enabled for SM121 and
    [#3960](https://github.com/flashinfer-ai/flashinfer/pull/3960) GDN CuteDSL
    as `sm_121a`, both inherited from the `0.6.16` pin, plus
    [#4117](https://github.com/flashinfer-ai/flashinfer/pull/4117) GDN WY
    decode on SM121 and
    [#3903](https://github.com/flashinfer-ai/flashinfer/pull/3903)
    `trtllm_allreduce` extended to SM12x.
  - Earlier in this line,
    [PR #3576](https://github.com/flashinfer-ai/flashinfer/pull/3576) added the
    `head_dim=512` dispatch for SM120/121. **Caveat for Gemma-4:** SGLang's own
    attention-backend allowlist hard-rejects `flashinfer` for the Gemma-4
    architecture (only `trtllm_mha | triton | ascend | intel_xpu` are
    accepted), so that fix stays moot for Gemma *attention*. The Gemma-4
    profiles still set `attention_backend=triton`, which is a SGLang allowlist
    constraint and not a flashinfer-version limitation.

  What `0.6.18` adds on top, all of it SM12x-relevant:
  [#4285](https://github.com/flashinfer-ai/flashinfer/pull/4285) (SM12x NVFP4
  fused-MoE kernels synced to b12x HEAD),
  [#4329](https://github.com/flashinfer-ai/flashinfer/pull/4329) (gated SM12x
  dynamic NVFP4 MoE),
  [#4318](https://github.com/flashinfer-ai/flashinfer/pull/4318) (wave+residue
  tile selection for the groupwise MoE GEMM),
  [#4380](https://github.com/flashinfer-ai/flashinfer/pull/4380) (DSV4 sparse
  MLA top-k 192/256 consolidated for SM120/121),
  [#4374](https://github.com/flashinfer-ai/flashinfer/pull/4374) (GDN cp host
  launch overhead on sm120), plus asymmetric VO-split NVFP4 paged prefill for
  **Gemma-4** on SM120/121 and native MXFP4 MoE on SM120/121 (no MXFP4
  checkpoint is served here, but the path now exists).

  **Read this before you benchmark `0.5.19-sm121` against an older tag:** the
  0.6.18 release notes state that the CUDA 13 **aarch64 wheels drop the native
  `sm_121a` cubins** and that DGX Spark "keeps running via SM120 family cubins".
  That is observable in the built image: `flashinfer_jit_cache` carries 907
  prebuilt modules, **zero** with `sm121` in the name and six with `sm120`
  (`nvfp4_attention_sm120`, `fp4_gemm_cutlass_sm120`, `gemm_sm120`, …). Nothing
  is broken by it, but it is a codegen change on exactly this hardware, so treat
  a throughput delta against `0.5.18-sm121` as a finding, not as noise.

  Rolling flashinfer back is only meaningful on `0.5.18-sm121` and older:
  `0.6.16.post3` (what the first `0.5.17-sm121` build shipped), then `0.6.16`
  (the `0.5.16-sm121` pin), then `0.6.15.post1` (live-validated on
  `0.5.15.post1-sm121`). On `0.5.19-sm121` there is no flashinfer rollback, roll
  the whole tag back instead.
- **transformers pinned to `5.19.0`** on `0.5.23-sm121` (needs tokenizers
  `0.23.2`, see below). Up to `0.5.21-sm121` it is pinned to `5.12.1` (exactly SGLang's own pyproject pin,
  unchanged from v0.5.15 through v0.5.19; was `5.8.1` on v0.5.13/v0.5.14), required for
  the Gemma-4 `*-assistant` drafter checkpoints used by NEXTN/MTP speculative
  decoding (`google/gemma-4-{26B-A4B,31B}-it-assistant`).
  Earlier transformers releases don't know the drafter's config subclass
  and the SGLang head exits with `Unrecognized configuration class` during
  drafter weight-loading. **Exception:** the `0.5.14-gemmadiffusion-sm121`
  image pins `5.11.0` instead — `diffusion_gemma` is an unregistered
  `model_type` before then (AutoConfig `KeyError`), and 5.11.0 is the
  version DiffusionGemma's upstream PR #28054 pins.
- **tokenizers pinned to `0.22.2`** (`0.5.19-sm121` through `0.5.21-sm121`;
  `0.23.2` on `0.5.23-sm121`, since transformers 5.19.0 requires `>=0.23.1`). v0.5.19 is the
  first SGLang ref to pin tokenizers at all, with the note that `0.23.0rc0` is
  incompatible with transformers' `CLIPTokenizer`, and the build does not
  arrive there on its own: the dependency completion step compares *presence*
  and not version, and the late transformers install runs with `--pre`, so pip
  is free to resolve the release candidate inside transformers' own
  `tokenizers>=0.22.0,<=0.23.0` range. The first `0.5.19-sm121` build shipped
  `0.23.0rc0` for exactly that reason; the pin is now an explicit build knob
  applied as the last pip action of the builder stage. Affects the CLIP-based
  multimodal / diffusion tokenizer paths rather than the NVFP4 text models.
- **Gemma-4 MTP (Frozen-KV) speculative-decoding patch** — the
  `0.5.11-gemma4-sm121` tag carries a cherry-pick of upstream
  [PR #24436](https://github.com/sgl-project/sglang/pull/24436)
  ("Gemma 4 — Adding MTP support", merged 2026-05-07, after the v0.5.11
  release tag). **Native in v0.5.12+** — the `0.5.12*` tags ship it from
  upstream and the cherry-pick is no longer applied. Adds the dedicated `Gemma4AssistantForCausalLM` model and a
  new `FROZEN_KV_MTP` speculative algorithm (recurrent hidden-state draft
  loop with frozen target KV cache). At runtime SGLang auto-promotes
  `--speculative-algorithm NEXTN → FROZEN_KV_MTP` once the drafter is
  detected. Without this patch the stock NEXTN/EAGLE worker crashes with
  `ValueError: No module or parameter named 'model.language_model' in
  TransformersMultiModalForCausalLM` during drafter weight-load.
  Verified working on the 4-node DGX Spark cluster — see the 31B-it
  TESTLOG, [Test 07 (`num_steps=2`, `num_draft_tokens=3`)](https://github.com/vroomfondel/dgxarley/blob/main/TESTLOGS/sglang_nn4_tp4_ep1/gemma-4-31b-it/TESTLOG_nv580.142_sglang-0.5.11_gemma-4-31b-it_4n.md#mtp-sweep-tests-711--partial-15-cases-done):
  **+98 % at n=1** (10.49 → 20.83 tok/s), **+76 % at n=4** (44.06 → 77.67
  tok/s), drafter acceptance rate median ~0.68, 5/5 requests stopped on
  natural EOS. The 26B-A4B MoE sibling's MTP sweep is still in progress
  ([TESTLOG](https://github.com/vroomfondel/dgxarley/blob/main/TESTLOGS/sglang_nn4_tp4_ep1/gemma-4-26b-a4b-it/TESTLOG_nv580.142_sglang-0.5.11_gemma-4-26b-a4b-it_4n.md)).
- **NemotronH MTP + radix cache (experimental)** — the
  `0.5.13-dev-nemotronh-mtp-sm121` tag carries upstream
  [PR #27998](https://github.com/sgl-project/sglang/pull/27998) (unmerged),
  which enables native MTP speculative decoding for
  `nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4` **without**
  `--disable-radix-cache`. It differs from the production `0.5.13-sm121` tag
  by exactly this one Python-only source patch so the MTP behaviour can be
  A/B'd against production. **First-contact / unvalidated** — validate
  `accept_len > 1` and no NaN logits (an NVFP4-MTP risk,
  [#27828](https://github.com/sgl-project/sglang/issues/27828)) on first boot.
- **DiffusionGemma dLLM (experimental, main-ahead)** — the
  `0.5.14-gemmadiffusion-sm121` tag is **the unified Gemma-4 image**: one
  build that serves all five Gemma-4 profiles — BF16 + FROZEN_KV_MTP,
  NVFP4, and the diffusion-LLM `nvidia/diffusiongemma-26B-A4B-it-NVFP4`. It
  is pinned forward to SGLang `main @ 3a1417a` (post-v0.5.13, 2026-06-12)
  rather than the v0.5.13 tag because (1)
  [PR #28081](https://github.com/sgl-project/sglang/pull/28081) fixes a
  short-lived broken `FrozenKVMTPCudaGraphRunner` that crashes Gemma-4 BF16
  MTP at boot on the v0.5.13 tag, and (2) the DiffusionGemma bake
  ([PR #28054](https://github.com/sgl-project/sglang/pull/28054), unmerged)
  applies far more cleanly main-vs-main. Carries the Gemma-4 GeGLU/FP4
  NaN-clamp source patch ([PR #22928](https://github.com/sgl-project/sglang/pull/22928))
  and uses the `-mainahead` sgl-kernel patch variants (one day of main drift
  shifted the mscclpp link lines). **First-contact / main-ahead, not a
  tagged release.**
- **Qwen4-Exp / Qwen3.8-Flash-Next (source patch on `0.5.18-sm121` ONLY, not in
  `0.5.19-sm121`; native upstream from `0.5.20-sm121` via
  [#37500](https://github.com/sgl-project/sglang/pull/37500))**: upstream
  [PR #36497](https://github.com/sgl-project/sglang/pull/36497) (still open,
  not in v0.5.18, v0.5.19 or `main`) applied to the source before install. It is
  the only implementation of `Qwen4ExpForConditionalGeneration` / `model_type
  qwen4_exp` anywhere; without it the image refuses
  `RadixArk/Qwen3.8-Flash-Next-NVFP4` at load with *"has no SGLang
  implementation"*. The build is only half of it: the two GB10/SM121 QSA
  decode fixes are deliberately **runtime** patches in the Ansible role, one of
  them a defensive veto against an upstream gate expression that was respelled
  three times in three days, one spelling of which silently corrupts
  long-context output on GB10. An image built from the source patch alone
  crashes at backend init on a Spark.
  **Why it is absent from `0.5.19-sm121`:** the patch is a squash of the
  upstream branch cut against a v0.5.18-era base and it does not rebase onto
  v0.5.19 by `patch(1)` (15 files with rejected hunks), while the upstream
  branch itself has moved on. So `0.5.19-sm121` refuses
  `RadixArk/Qwen3.8-Flash-Next-NVFP4` at load and `0.5.18-sm121` stays the tag
  for that model.
- **Nemotron-3.5-Lightning speculative decoding (`0.5.18-sm121`)**: upstream
  [PR #36186](https://github.com/sgl-project/sglang/pull/36186) (merged
  2026-08-25, three days after the v0.5.18 tag) backported together with its
  two DFlash2 prerequisites,
  [#35371](https://github.com/sgl-project/sglang/pull/35371) and
  [#35496](https://github.com/sgl-project/sglang/pull/35496). Enables all three
  published drafters for `nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4`
  (in-checkpoint MTP plus the external `-DFlash` / `-DSpark` drafts, the latter
  being NVIDIA's own DGX Spark recommendation). Stock v0.5.18 aborts DSPARK /
  DFLASH at backend setup with *"implements neither
  set_dspark_layers_to_capture nor set_dflash_layers_to_capture"*. Serving the
  target **unspeculated** does not need this patch. That expiry has **arrived**:
  v0.5.19 contains the merged commit, so `0.5.19-sm121` serves all three
  drafters natively and applies no backport.
- Built on a CUDA 13.2 + NCCL 2.30.7 base for the GB10 codegen path: PyTorch 2.14.1
  on `0.5.23-sm121`, PyTorch 2.13 on `0.5.19-sm121` through `0.5.21-sm121` (CUDA
  13.1 / PyTorch 2.10 fallback is ~45 % slower end-to-end). `0.5.23-sm121` uses the
  new **slim** base
  [`xomoxcc/dgx-spark-pytorch-dev:2.14.1-v2-cu132`](https://hub.docker.com/r/xomoxcc/dgx-spark-pytorch-dev)
  and removes the flashinfer trtllm-gen cubins (fmha / batched_gemm, ~7 GB, built
  for sm100/sm103/sm107 only and unloadable on sm121) in the builder stage. The
  image is 13.5 GB uncompressed in 9 layers; the previous build of the same tag
  on the fat base was 38.2 GB in 37 layers. **Known
  issue:** the NCCL 2.30.x NVLS path has a regression that can silently hang
  high-expert-count MoE weight loads (≥256 experts) on GB10/RoCE
  ([NVIDIA/nccl#2167](https://github.com/NVIDIA/nccl/issues/2167)) — set
  `NCCL_NVLS_ENABLE=0` when running these (free on non-NVLink hardware; the
  Ansible role does this for you)
- **SGLang v0.5.22 / v0.5.23 items relevant to GB10** (`0.5.23-sm121`):
  [#32779](https://github.com/sgl-project/sglang/pull/32779) SM120 Triton
  sparse-MLA prefill backend for DSA,
  [#32947](https://github.com/sgl-project/sglang/pull/32947) GLM5.2 DSA indexer
  fallback (opt-in env, no MTP), the DSA top-k overflow fixes
  [#37625](https://github.com/sgl-project/sglang/pull/37625) /
  [#37941](https://github.com/sgl-project/sglang/pull/37941), and the
  modelopt-nvfp4 gate/up global-scale order fixes
  [#45289](https://github.com/sgl-project/sglang/pull/45289) /
  [#44858](https://github.com/sgl-project/sglang/pull/44858) /
  [#45386](https://github.com/sgl-project/sglang/pull/45386)

## Runtime patches are part of the deal

The image is half the stack. A second set of patches is applied to the
installed `sglang` tree **at container start** by the Ansible role
([`roles/k8s_dgx/files/sglang_patches/`](https://github.com/vroomfondel/dgxarley/tree/main/roles/k8s_dgx/files/sglang_patches)),
because several GB10 fixes have to survive image bumps and because one
ConfigMap feeds instances pinned to *different* tags. Each patch gates itself,
never raises, and is idempotent; a drifted anchor logs `ANCHOR-DRIFT` and
degrades to unpatched SGLang rather than crash-looping the pod.

For `0.5.19-sm121` that set was replayed offline against the ref before the
build and five patches needed work, which is worth knowing if you run this tag
with your own patch set: v0.5.19 moved the arch predicates onto a
`srt/runtime_context` platform object (`is_sm100_supported()` →
`get_platform().is_sm100`), moved server-arg reads to the resolved exec context
(`server_args.<arg>` → `get_exec().<ns>.<arg>`), inlined the `*_CHOICES`
constants into their `Arg()` declarations, gave `DSATopKBackend` a `resolve()`
constructor, rebuilt the trtllm-MHA KV read around `use_fmha_v2`, moved the
fixed-q-len verify call into a helper, and split the one big
`arg_groups/overrides.py` into `arg_groups/model_overrides/<model>.py`. Two
older patches turned out to be absorbed upstream (the Qwen3.5 MTP KV-scale
mapper, and the HY3 NEXTN `final_layernorm` remap, that one already in v0.5.18)
and now self-gate off instead of reporting drift.

One deliberate refusal lives there too: v0.5.19's new **ragged verify layout**
calls the decode kernel without block scales or an XQA draft mask, and whether
that is correct with an NVFP4 KV cache is unverified. Since the failure mode of
guessing is silently corrupted output, that combination now raises an
actionable error instead.

## Tags

| Tag            | Notes                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     |
|----------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `0.5.23-sm121` | SGLang v0.5.23 (covers v0.5.22 too) + SM121 patches; **new slim PyTorch 2.14.1 base** (`xomoxcc/dgx-spark-pytorch-dev:2.14.1-v2-cu132`); flashinfer **0.7.3** (deliberately past upstream's 0.7.2 pin) + nvidia-cutlass-dsl 4.8.0; transformers 5.19.0 + tokenizers 0.23.2; flashinfer trtllm-gen cubins (sm100/sm103/sm107 only, ~7 GB, unloadable on sm121) removed in the builder stage, image 13.5 GB uncompressed in 9 layers (38.2 GB in 37 layers on the fat base). Patch-set acceptance gate clean; GPU smoke on a single DGX Spark only (Nemotron-3.5-Lightning-30B-A3B-NVFP4 + DSPARK draft, TP1, GSM8K 10/10, decode with CUDA graphs, no cubin download attempts). **Not yet validated multi-node (TP4 / RoCE) or with GLM-DSA / Qwen3.8 profiles. (current, first-contact)** |
| `0.5.21-sm121` | SGLang v0.5.21 + SM121 patches; quiet build side (only the cosmetic sgl-kernel patch suffix moves); xgrammar 0.2.7 + sentencepiece 0.2.1 pinned; speculative kernels moved from AOT to JIT upstream (#40033); PyTorch 2.13 base, flashinfer 0.6.18.post1 + cutlass-dsl 4.6.2. Deployed default on the private cluster since 2026-10-03                                                                                                                                                                                                                                                                                                                                                                                                                                                    |
| `0.5.20-sm121` | SGLang v0.5.20 + SM121 patches; Qwen3.8-Flash-Next is **native** (#37500), no 20k-line source patch needed; SM121 QSA pieces upstreamed (our p65 self-gates off, p66 only widens the TP contract); PyTorch 2.13 base, flashinfer 0.6.18.post1                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             |
| `0.5.19-sm121` | SGLang v0.5.19 + SM121 patches; PyTorch 2.13 base, flashinfer 0.6.18.post1; no Qwen4-Exp                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  |
| older tags     | `0.5.18-sm121` and older, incl. `-gemma4`/`-gemmadiffusion`/`-dev` variants: still pullable, notes in the [file history](https://github.com/vroomfondel/dgxarley/commits/main/DOCKERHUB_OVERVIEW_dgx-spark-sglang.md)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     |

All tags are **`linux/arm64` only** — these images are useless on x86_64 and on
non-GB10 NVIDIA hardware (the kernels are arch-pruned to `sm_121`).

## Build & deploy context

Everything that produces these images — Dockerfiles, sgl-kernel patches, recipe
files, the cross-arch podman build driver, and the Ansible roles that deploy
SGLang on a 4-node DGX Spark K3s cluster — lives in
[github.com/vroomfondel/dgxarley](https://github.com/vroomfondel/dgxarley).

Relevant entry points:

- [`scripts/build_sm121_image.sh`](https://github.com/vroomfondel/dgxarley/blob/main/scripts/build_sm121_image.sh)
  — remote-podman build driver (x86 control host → arm64 build runner)
- [`scripts/patches/sglang-0.5.23-sm121.recipe`](https://github.com/vroomfondel/dgxarley/blob/main/scripts/patches/sglang-0.5.23-sm121.recipe)
  — recipe pinned by the build (SGLang + flashinfer + cutlass-dsl + transformers
  pins, plus the per-release patch gates). One recipe per tag, they are kept
  rather than edited in place
- [`scripts/patches/sgl-kernel-sm121.patch`](https://github.com/vroomfondel/dgxarley/blob/main/scripts/patches/sgl-kernel-sm121.patch)
  — the core CUTLASS NVFP4 SM121 fix, applied up to `0.5.15.post1-sm121` and
  gated off since (`APPLY_SGL_KERNEL_SM121=0`)
- [`scripts/verify_sglang_image.sh`](https://github.com/vroomfondel/dgxarley/blob/main/scripts/verify_sglang_image.sh)
  — patch-set acceptance gate; the build driver runs it against the built image
  automatically (opt out with `--no-verify`), and no tag is pushed without it
- [`scripts/patches/sglang-dsv4-nvfp4-pr25820.patch`](https://github.com/vroomfondel/dgxarley/blob/main/scripts/patches/sglang-dsv4-nvfp4-pr25820.patch)
  — DeepSeek-V4 NVFP4 MoE support (upstream PR #25820, rebased onto v0.5.13)
- `CUTLASS_NVFP4_SM121_PRD.md` — NVFP4 root cause + fix rationale (in repo)
- `UPSTREAM_DSV4_BUGS.md` — DeepSeek-V4-Flash sm_121 boot chain, wall by wall (in repo)
- [`roles/k8s_dgx/`](https://github.com/vroomfondel/dgxarley/tree/main/roles/k8s_dgx)
  — Ansible role that deploys SGLang head + workers (Multus + RoCE-over-SR-IOV
  NCCL, HAProxy sidecar for the head's EADDRINUSE workaround, model profiles)

## Status / support

These images are built and exercised on a private 4-node DGX Spark cluster
(`spark1`–`spark4`). They are published in case someone else has the same
hardware and runs into the same SM121 crashes — there is no commercial support
and tags may be retagged or removed without notice. Open an issue on the
GitHub repo if something is broken.
