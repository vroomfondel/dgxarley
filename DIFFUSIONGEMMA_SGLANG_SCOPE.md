# Feasibility sketch: serving DiffusionGemma via SGLang on this cluster

As of 2026-06-19. Companion document to the profile
`roles/k8s_dgx/model_profiles/nvidia-diffusiongemma-26b-a4b-it-nvfp4.yml`.

**Question:** Can we run `nvidia/diffusiongemma-26B-A4B-it-NVFP4` (or its BF16 base
`google/diffusiongemma-26B-A4B-it`) on **SGLang** instead of vLLM?

**Short answer:** Technically yes, but **not today on any release or any of our
images** — the SGLang runtime code lives in an **open, unmerged PR**. It would
require a custom image bake **plus** a custom dLLM launch path. Until then the
**`vllm` tag** is the only working path.

---

## 0. Why it doesn't work today (PR status, verified 2026-06-19)

| PR | Content | Status |
|----|---------|--------|
| [#27824](https://github.com/sgl-project/sglang/pull/27824) | Mintlify cookbook **docs** | **MERGED 2026-06-10** (docs landed before the code) |
| [#27823](https://github.com/sgl-project/sglang/pull/27823) | Runtime/model **code** (1st attempt) | **CLOSED unmerged 2026-06-11** |
| [#28054](https://github.com/sgl-project/sglang/pull/28054) | Runtime/model **code** (reopened) | **OPEN** against `main`, not merged |

So the [cookbook page](https://docs.sglang.io/cookbook/autoregressive/Google/DiffusionGemma)
exists, but the code (`models/gemma4_diffusion.py`, the
`Gemma4Renoise`/EntropyBound sampler) is **in no tagged SGLang** — not even in
0.5.13. Our `0.5.13-gemma4-sm121` image cannot load the model.

**What the model is, per PR #28054:** an **encoder-decoder** uniform-state
(renoising) block-diffusion model (26B-A4B MoE) with a **Gemma-4 vision tower**.
Encoder = causal context; decoder = bidirectional fixed-length "canvas" (256
tokens), iteratively denoised over `max_denoising_steps` (default 48). This is a
**different runtime path** than the autoregressive scheduler that every other
SGLang profile in this repo builds on.

---

## 1. What would need to be done — file by file

### A) Image: bake in PR #28054

Same mechanism as the other unmerged-PR bakes (gemma4-NVFP4 #22928,
DSV4 #25820): source patch + Dockerfile patch, recipe-gated in
`scripts/build_sm121_image.sh::apply_patches`.

- [ ] `gh pr diff 28054 > scripts/patches/sglang-diffusiongemma-pr28054.patch`
      (rebased onto `v0.5.13`; the PR is against `main` → check context offsets,
      the in-container `patch --dry-run` catches drift).
- [ ] `dockerfile-diffusiongemma.patch` (COPY + `RUN patch -p1 < …` before
      `uv pip install ./python`). **Watch the Dockerfile region:** the
      gemma4-nvfp4 and dsv4-nvfp4 Dockerfile patches anchor on the **same region**
      per `build_sm121_image.sh` (comment 2b/2c). Since DiffusionGemma needs the
      gemma4 image as its base (vision tower + gemma4 patches), the new COPY+RUN
      block must co-exist with the gemma4-nvfp4 block → regenerate the context
      cleanly once.
- [ ] New recipe `sglang-0.5.13-gemma4-diffusion-sm121.recipe` (base:
      `sglang-0.5.13-gemma4-sm121.recipe`, plus `APPLY_DIFFUSIONGEMMA_PR28054=1`).
      Its own tag `xomoxcc/dgx-spark-sglang:0.5.13-gemma4-diffusion-sm121`, so the
      other four Gemma profiles stay on the leaner `0.5.13-gemma4`.
- [ ] Verify in the image: model `google/diffusiongemma-26B-A4B-it` loads,
      `--dllm-algorithm Gemma4Renoise` is recognized.

### B) Launch path — the actual crux

Today `roles/k8s_dgx/files/sglang_launch.sh:1117` builds a
`python3 -m sglang.launch_server --model-path … <autoregressive flags>` and
`exec`s the args array (line 1321). The dLLM path is different:

```
sglang serve --model-path google/diffusiongemma-26B-A4B-it \
  --dllm-algorithm Gemma4Renoise --trust-remote-code --host 0.0.0.0 --port 30000
```

- [ ] **OPEN QUESTION (verify against PR #28054):** Does
      `python3 -m sglang.launch_server` accept the `--dllm-algorithm` flag (same
      OpenAI server, just a different decode loop), or is `sglang serve` a
      **separate server entry**? Whether we just add an `if dllm_algorithm` branch
      to `sglang_launch.sh` or need a second launch path depends on this.
- [ ] Per the cookbook, dLLM **automatically forces**: Triton attention, eager
      mode, unchunked prefill. Our autoregressive knobs (`disable_cuda_graph`,
      `cuda_graph_max_bs`, `moe_runner_backend`, `fp4_gemm_backend`,
      `mem_fraction_static`, piecewise CG) are **inert or conflicting** there →
      suppress them in the launch script when `dllm_algorithm` is set.
- [ ] **Multi-node/TP unclear:** encoder-decoder block diffusion with TP=4 across
      4 nodes is documented nowhere (the cookbook shows single-node). Smoke-test
      single-node BF16 first, then TP.
- [ ] **Re-validate HAProxy sidecar + probes:** does the dLLM server bind the same
      way (pod-ip vs 0.0.0.0 → our "omit `--host` + HAProxy" logic)? Is `/health`
      present? Startup/denoising time for the `startupProbe`?
- [ ] **Streaming UX:** dLLM streams **one fully-denoised canvas per chunk** (not
      token-by-token). Re-check OpenWebUI/Hermes behavior.

### C) Profile schema

New dLLM knobs needed that the autoregressive schema doesn't know:

- [ ] `dllm_algorithm: "Gemma4Renoise"`, `max_denoising_steps: 48`,
      `entropy_bound: 0.1`, canvas/block size (256), temp schedule (`t_min`/`t_max`).
- [ ] `max_running_requests: 4` is already in the profile (card: `--max-num-seqs 4`).
- [ ] The autoregressive fields stay as schema placeholders (the profile won't
      validate otherwise) but are ignored by the launch script on the dLLM path
      (see B).

### D) NVFP4 — biggest substantive risk

- [ ] PR #28054 and the cookbook target the **BF16** base
      `google/diffusiongemma-26B-A4B-it`. Whether `nvidia/…-NVFP4` (modelopt_fp4)
      loads through `gemma4_diffusion.py` is **confirmed nowhere** — the NVIDIA
      card documents only vLLM for the NVFP4 quant. **Get BF16 working on SGLang
      first, NVFP4 as a separate experiment afterwards.**

---

## 2. Risks / unknowns

- **PR in flux:** #28054 is open, its predecessor #27823 was discarded ("new PR
  will be opened again soon"); a user reported model-load warnings. A bake today
  builds on a non-final state.
- **Eager-only on GB10:** no CUDA graph → unclear performance; the diffusion win
  (parallel decoding) has to more than offset the eager overhead. Benchmark
  against the vLLM baseline (peak, not aggregate).
- **NVFP4 path** (see above) — may not run under SGLang at all.
- **Multi-node** untested for an encoder-decoder diffusion model.

---

## 3. Recommended sequence

1. **Wait/monitor:** let PR #28054 merge + land in an SGLang release/nightly. Then
   the bake is a pinned patch instead of a moving target.
   Trigger: <https://github.com/sgl-project/sglang/pull/28054>.
2. If needed sooner: BF16 `google/diffusiongemma-26B-A4B-it`, **single-node**,
   image baked with #28054, dLLM launch branch in `sglang_launch.sh` → smoke-test
   (`/health`, one canvas output, coherence check).
3. Only after that: TP=4 multi-node, then NVFP4.
4. **Default recommendation until then:** the `vllm` tag (`scitrera/dgx-spark-vllm`)
   — the only documented + working path today.

---

## 4. Effort estimate (rough)

| Block | Effort | Blocker |
|-------|--------|---------|
| Image bake (A) | low — established patch mechanism | PR drift, Dockerfile-region merge |
| Launch path (B) | **medium-high** — new decode path, probes, HAProxy, MN | OPEN QUESTION `launch_server` vs `sglang serve` |
| Profile schema (C) | low | — |
| NVFP4 (D) | unknown — may not run | upstream BF16 only |

**Overall assessment:** feasible, but no quick win. The launch path (B) is the
actual work; everything else is routine. As long as #28054 isn't merged,
"wait + use vLLM" is the best bang for the buck.
