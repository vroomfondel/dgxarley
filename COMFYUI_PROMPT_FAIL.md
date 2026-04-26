# ComfyUI on sm121: Prompt Conditioning Detached from Input Text

**Status:** **RESOLVED.** Root cause is a numerically broken
`aten::_efficient_attention_forward` kernel path on Blackwell GB10 (sm121)
in PyTorch 2.10. The kernel returns garbage without crashing or producing
NaN, which silently corrupts every text-encoder forward in ComfyUI.
Workaround applied via a sitecustomize shim that wraps
`comfy.sd1_clip.SDClipModel.forward` with `sdpa_kernel([SDPBackend.MATH])`.
Verified end-to-end against SDXL (RealVisXL_V5.0) and Flux-schnell.

**First diagnosis session:** 2026-04-26
**Resolution session:** 2026-04-26

## Symptom

The RealVisXL-V5.0 workflow (SDXL + Fooocus-Inpaint) on the cluster
produced outputs that *changed per prompt* but bore no semantic relation
to the prompt text:

- **Local** (RTX 4090, default PyTorch stack): same workflow works
  cleanly, RealVis follows the prompt.
- **Cluster** (DGX Spark, Blackwell GB10 / sm121, custom image
  `xomoxcc/comfyui:sm121`, PyTorch 2.10): "complete garbage" — a
  "red apple" prompt rendered an interior of a designer loft, "blue cube"
  rendered a wooden tiki statue.

Conditioning was active (outputs reproducibly differed by prompt) but
semantically desynchronised — the embedding vector that `CLIPTextEncode`
produced did not represent the prompt content.

## Environment

```
ComfyUI version:  0.19.3
PyTorch:          2.10.0  (custom build, TORCH_CUDA_ARCH_LIST=8.0;12.1)
Python:           3.12.3
GPU:              NVIDIA GB10 (Blackwell, sm121)
Image:            xomoxcc/comfyui:sm121  (BUILDTIME 2026-04-25T15:07:41Z)
Pod node selector: spark-id=4 (pinned to spark4)
Launch flags:     main.py --listen 0.0.0.0 --port 8188
                  --use-sage-attention --highvram
External endpoint: https://comfyui.dgx.elasticc.io
ConfigMap source:  roles/k8s_dgx/templates/comfyui_launch.sh.j2
```

## Reproducer

A minimal txt2img workflow without any custom nodes — only Comfy core.
Two visually disjoint prompts with a fixed seed so any output difference
is conditioning-driven.

```json
{
  "3": {"inputs": {"seed": 12345, "steps": 25, "cfg": 5.0,
    "sampler_name": "dpmpp_2m", "scheduler": "karras",
    "denoise": 1.0, "model": ["4",0], "positive": ["6",0],
    "negative": ["7",0], "latent_image": ["5",0]}, "class_type": "KSampler"},
  "4": {"inputs": {"ckpt_name": "RealVisXL_V5.0_fp16.safetensors"},
        "class_type": "CheckpointLoaderSimple"},
  "5": {"inputs": {"width":1024,"height":1024,"batch_size":1},
        "class_type": "EmptyLatentImage"},
  "6": {"inputs": {"text":"<PROMPT>","clip":["4",1]}, "class_type":"CLIPTextEncode"},
  "7": {"inputs": {"text":"blurry, low quality, watermark, text, deformed",
                   "clip":["4",1]}, "class_type":"CLIPTextEncode"},
  "8": {"inputs": {"samples":["3",0],"vae":["4",2]}, "class_type":"VAEDecode"},
  "9": {"inputs": {"filename_prefix":"<PREFIX>","images":["8",0]},
        "class_type":"SaveImage"}
}
```

| Label | Positive prompt | Seed |
|---|---|---|
| A | `a vibrant red apple on a wooden table, professional photo, sharp focus` | 12345 |
| B | `a glossy blue cube on a marble surface, professional photo, sharp focus` | 12345 |

Driver: `/tmp/sage-test/run.py` (cluster) and `/tmp/sage-test/run_flux.py`
(Flux variant). PNGs are saved to `/tmp/sage-test/<mode>_<label>.png`.

## Diagnosis Chain

### Phase 1 — SageAttention ruled out

Hypothesis: `--use-sage-attention` (forced on sm121 because xformers'
FMHA dispatcher mis-handles cutlass kernels there) numerically drifts
the SDXL UNet cross-attention.

Removed the flag from the launch ConfigMap, restarted the pod, re-ran
the A/B test:

| Run | A (red apple) | B (blue cube) | bytes |
|---|---|---|---|
| with-sage | designer loft, stone wall, chandelier | wooden tiki, ivy wall | 1.68 / 1.96 MB |
| without-sage | photo of two men in T-shirts | photo of a man at a conference table | 1.56 / 1.42 MB |

**Both runs ignore the prompt.** Sage has a strong stylistic effect but
is not the source of the desync. Restored launch args, sage re-enabled.

### Phase 2 — Checkpoint integrity ruled out

`sha256sum /workspace/comfyui/models/checkpoints/RealVisXL_V5.0_fp16.safetensors`
on the cluster matched the local file:
`6a35a7855770ae9820a3c931d4964c3817b6d9e3c6f9c4dabb5b3a94e5643b80`.
Not a bad weight file.

### Phase 3 — Cross-model A/B (Flux-schnell)

If the bug were SDXL-specific (e.g. broken CLIP-G), Flux should be fine.
Same A/B test against `flux1-schnell-fp8.safetensors`, 4 steps, cfg=1.0,
euler/simple:

| Prompt | Output |
|---|---|
| A (red apple) | hand-written-looking card with pseudo-English text "P-bIT WIMG" |
| B (blue cube) | hand-written-looking card with pseudo-English text "PHERIS WILKINE" |

Cross-model bug confirmed. Flux is text-on-image–trained and falls back
to its "default text card" prior when conditioning fails. SDXL falls
back to "default scenes". The shared denominator is the text-encoder
forward, which on Flux is **T5-XXL** (SentencePiece) and on SDXL is
**CLIP-L + CLIP-G** (BPE) — different tokenisers, different model
families. Same-style failure in both rules out tokenizer-level bugs and
narrows it to the actual GPU forward pass.

### Phase 4 — Direct CLIP-L smoke test in the pod

Bypassed ComfyUI entirely. Loaded `openai/clip-vit-large-patch14` via
`transformers.CLIPTextModel` on CPU and GPU, encoded both prompts,
compared embeddings:

| Mode | norm | mean\|Δcpu\| | max\|Δcpu\| | ‖A−B‖ vs CPU ref (215.671) |
|---|---|---|---|---|
| cpu_fp32 (reference) | 295.8 | 0 | 0 | 215.671 |
| gpu_fp32 (1st pass) | 248.6 | 1.02 | **27.4** | 215.7 — but Δcpu massive |
| gpu_fp32 (2nd pass) | NaN | NaN | NaN | NaN |
| gpu_fp16 | NaN | NaN | NaN | NaN |
| gpu_bf16 | NaN | NaN | NaN | NaN |

GPU encoder produces NaN in fp16/bf16 (the dtype actually used by
ComfyUI), and even fp32 only sometimes returns numbers — but with
component-wise drift of std-magnitude (max\|Δ\|=27 at std≈1), which is
not floating-point noise; it is a kernel returning garbage. The first
fp32 pass happened to land on a viable kernel; the second pass cycled
into NaN territory.

The same FATAL kernel-table-probe spam we treat as cosmetic in
`COMFYUI_SM121_PATCHES.md` was streaming during the test, confirming
that PyTorch's CUTLASS dispatcher was active and rejecting kernels.

### Phase 5 — SDPA backend isolation

Re-ran the encoder against every available SDPA dispatch path,
controlled via `attn_implementation=` on the model and
`torch.nn.attention.sdpa_kernel([...])` as the kernel selector. CPU
fp32 reference: `‖A−B‖ = 215.671`.

| attn_impl | dtype | sdpa-backend | ‖A−B‖ | max\|Δcpu\| | Verdict |
|---|---|---|---|---|---|
| eager | fp32 | — | 215.7 | 0.0001 | OK |
| eager | fp16 | — | 216.0 | 0.26 | OK |
| eager | bf16 | — | 214.5 | 1.55 | OK (bf16 noise) |
| sdpa | fp32 | **MATH** | 215.7 | 0.0001 | OK |
| sdpa | fp32 | **EFFICIENT** | **128.0** | **25.9** | **broken** |
| sdpa | fp32 | FLASH | — | — | no sm121 SASS |
| sdpa | bf16 | **MATH** | 214.8 | 1.55 | OK |
| sdpa | bf16 | **EFFICIENT** | **101.2** | **25.6** | **broken** |
| sdpa | bf16 | FLASH | — | — | no sm121 SASS |

Single-cause finding: `aten::_efficient_attention_forward` on sm121.
MATH and eager backends both reproduce CPU-grade output. FLASH has no
sm121 kernel image (a clean failure, not a silent one). EFFICIENT is
viable from the dispatcher's view but produces results that pass
NaN/inf checks while being numerically unrelated to the input.

## Root Cause

PyTorch 2.10's CUTLASS-family efficient-attention forward kernel on
sm121 is selected by the dispatcher but executes on a kernel variant
whose output is uncorrelated with the input. The "FATAL: kernel
`fmha_cutlassF_*_sm80` is for sm80-sm100, but was built for sm121"
spam observed during VAE/CLIP forwards is the dispatcher rejecting
kernels in its priority table; the survivor it eventually settles on
is the one returning garbage. Because the output is finite and
non-NaN, every downstream check (logit norms, attention masks,
diffusion sampling) sees plausible tensors and continues running.

In ComfyUI specifically, every text encoder hits this path because of
`comfy/ldm/modules/attention.py:762`:

```python
def optimized_attention_for_device(device, mask=False, small_input=False):
    if small_input:
        if model_management.pytorch_attention_enabled():
            return attention_pytorch
        else:
            return attention_basic
    ...
```

`small_input=True` is hard-wired by every text-encoder caller (CLIP-L,
CLIP-G, T5-XXL, Llama, Qwen, Gemma, Mistral, ByT5 — every model
inheriting from `comfy.sd1_clip.SDClipModel`). `attention_pytorch`
unconditionally calls `F.scaled_dot_product_attention`, which dispatches
to EFFICIENT and silently corrupts. The `--use-sage-attention` flag
routes only the *diffusion* hot path through Sage — text encoders
bypass that switch entirely, which is why both with-sage and
without-sage runs in Phase 1 produced equally broken (but stylistically
different) outputs.

The xformers patches in `COMFYUI_SM121_PATCHES.md` likewise do not help
here — they teach xformers' own dispatcher to avoid CUTLASS and FA3 on
sm121, but ComfyUI's text encoders never go through xformers.

## Fix

A surgical wrapper around `SDClipModel.forward` forces the SDPA backend
to MATH for the duration of every text-encoder forward. Implemented as
a `sitecustomize.py` written into the hostPath workspace at pod start
and loaded via `PYTHONPATH=/workspace/comfyui` so Python's `site.py`
picks it up at interpreter init, before ComfyUI's `main.py` runs.

The shim installs an `importlib` meta-path finder that fires once on
the first import of `comfy.sd1_clip` and patches `SDClipModel.forward`
in place:

```python
@functools.wraps(orig)
def patched(self, *args, **kwargs):
    with sdpa_kernel([SDPBackend.MATH]):
        return orig(self, *args, **kwargs)
```

This single wrap covers every text-encoder model in ComfyUI because
every encoder class (CLIP-L, CLIP-G, T5-XXL, Llama, Qwen, Gemma, …)
inherits from `SDClipModel` and shares its forward. Idempotent via a
`cls._sm121_sdpa_patched` marker; survives ComfyUI `git pull`s because
no git-tracked file is modified.

The full implementation (with rationale comment block) lives in
[`roles/k8s_dgx/templates/comfyui_launch.sh.j2`](roles/k8s_dgx/templates/comfyui_launch.sh.j2)
under section §4c.

### Why MATH is acceptable here

The CLIP/T5 forwards in ComfyUI operate on 77-token sequences. MATH's
O(n²) memory and compute cost there is single-digit milliseconds per
workflow. Diffusion attention — the actual hot path — is unaffected:
SageAttention v2 continues to drive the UNet/DiT, and VAE attention
keeps using the xformers→fa2F-pt path established by the existing
`COMFYUI_SM121_PATCHES.md` patches. End-to-end wall-clock is
indistinguishable from before the patch.

## Verification

Pod redeployed 2026-04-26 with the §4c shim active. Pod startup log
confirms:

```
[comfyui] writing sm121 SDPA text-encoder shim
[sm121-sdpa-shim] installed lazy meta-path patcher for comfy.sd1_clip
[sm121-sdpa-shim] wrapped comfy.sd1_clip.SDClipModel.forward with SDPA-MATH (workaround for sm121 EFFICIENT_ATTENTION bug)
```

Re-ran both reproducers against the new pod (same seed=12345, same
prompts, no other changes):

| Run | A (red apple) | B (blue cube) |
|---|---|---|
| pre-fix with-sage | designer loft / stone wall | wooden tiki / ivy |
| pre-fix without-sage | two men in T-shirts | man at conference table |
| pre-fix Flux-schnell | hand-written card "P-bIT WIMG" | hand-written card "PHERIS WILKINE" |
| **post-fix SDXL** | **glossy red apple on wooden table** | **glossy blue cube on marble** |
| **post-fix Flux-schnell** | **red apple on wooden surface** | **blue glossy cube on marble** |

Wall-clock (warm pod, single Spark GPU under time-slicing):

| Workflow | First run (cold) | Warm |
|---|---|---|
| SDXL 25 steps 1024×1024 | 39.8 s | 10.7 s |
| Flux-schnell 4 steps 1024×1024 | 72.1 s | 7.6 s |

No measurable performance penalty.

## Related Files

| Path | Role |
|---|---|
| `roles/k8s_dgx/templates/comfyui_launch.sh.j2` §4c | Shim writer + `PYTHONPATH` export at launch |
| `COMFYUI_SM121_PATCHES.md` | xformers cutlass/FA3 patches — separate concern, complementary fix |
| `COMFYUI_ARM64_SM121.md` | Image build guide |
| `/tmp/sage-test/run.py` | SDXL A/B reproducer (workstation) |
| `/tmp/sage-test/run_flux.py` | Flux-schnell A/B reproducer (workstation) |
| `/tmp/sage-test/post-fix-{sdxl,flux}_*.png` | Verified-good output samples |

## Future Maintenance

- **PyTorch upgrade.** When PyTorch ships sm121-aware
  `_efficient_attention_forward` kernels, run the Phase 5 SDPA-isolation
  test again and, if EFFICIENT now produces ‖A−B‖≈215 and max\|Δcpu\|<2,
  drop the §4c shim.
- **ComfyUI upgrade.** The shim hooks one class
  (`comfy.sd1_clip.SDClipModel`). If upstream renames or refactors that
  base class, the meta-path finder will silently no-op (the marker check
  fires on a missing attribute, the warning lands in the pod log).
  Re-validate after every notable ComfyUI version bump by checking the
  `[sm121-sdpa-shim] wrapped …` log line is still emitted.
- **Diagnostic re-run.** The smoke test from Phase 4 (CPU vs GPU CLIP
  embedding distance for two prompts) is the cheapest correctness
  oracle; it runs in seconds and produces a numeric pass/fail signal.
  Useful as a post-deploy check whenever the image, PyTorch version, or
  GPU driver changes on the cluster.
