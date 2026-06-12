# SGLang Upstream Bugs / Gaps: DeepSeek-V4-Flash on SM121 (DGX Spark)

## Status (verified 2026-05-31, upstream re-checked 2026-06-08, re-verified 2026-06-11)

> **2026-06-08 — DSV4 ist nicht mehr der aktive Default.** `defaults/main.yml`
> hat auf `sglang_model: RedHatAI/Qwen3.6-35B-A3B-NVFP4` auf Image
> `scitrera/dgx-spark-sglang:0.5.12` zurückgestellt; das DSV4-Flash-FP8-Modell
> und das `xomoxcc/…0.5.12.post1-sm121`-Image sind jetzt **auskommentiert**.
> Dieses Dokument gilt, wenn DSV4-Flash (wieder) aktiviert wird. Upstream-Deltas
> seit 2026-05-31: **#26209 (FP4 Indexer for V4) ist MERGED** (2026-06-02, in
> `main`, noch nicht released — nach v0.5.12.post1); **#19589 ist CLOSED**;
> **DeepGEMM #317 ist CLOSED** (Maintainer abgelehnt — kein SM120-Hardware —
> und hat auf Community-PR #318 verwiesen, der noch offen/ungemergt ist). Der
> NVFP4-MoE-PR #25820 ist weiterhin offen, ein vollständiger NVFP4-V4-Pfad auf
> SGLang ist also in Arbeit, aber nicht verfügbar.
>
> **2026-06-11 — SGLang-Tag v0.5.13 heute geschnitten** (2026-06-11T08:09:52Z,
> noch kein GitHub-Release). `sglang 0.5.13` wurde am 2026-06-11T10:16Z auf PyPI
> veröffentlicht (~2 h nach dem Tag) — als reguläres Release installierbar, auch
> wenn kein GitHub-Release-Page existiert (Stand 2026-06-12). Enthält **PR
> #24692** (SM120-Support für DeepSeek-V4-Inference, merged 2026-06-01) sowie
> **#26209** (FP4 Indexer). Cluster-Images laufen weiterhin auf
> v0.5.12.post1-basierten Builds. PR #25820 (NVFP4 MoE) meldet laut letztem
> Kommentar vom 2026-06-11: „Flash NVFP4 is working now" (GSM8K 96.21% auf
> B200) — PR noch offen/ungemergt, kein SM120/121-Test erwähnt. Details zu den
> Wall-Auswirkungen von #24692 in §8.
>
> **Update 2026-06-12 — PR #25820 Status.** PR weiterhin offen; `/tag-and-rerun-ci`
> wurde 2026-06-12T07:18Z gepostet, `mergeable_state: blocked` (CI ausstehend),
> 13 Reviewer angefordert. Timeline: 2026-06-10 User-Report „Flash NVFP4 not
> working" (cuda_graph_runner exception) → 2026-06-10T17:28 trevor-m identifiziert
> veralteten quant_config im HF-Repo → 2026-06-11T11:50 „Flash NVFP4 is working
> now" nach HF-seitigem quant-config-Fix (GSM8K 96.21% auf B200; kein
> SM120/121-Test).
>
> **Update 2026-06-12 — `nvidia/DeepSeek-V4-Flash-NVFP4` existiert.** Das HF-Repo
> `nvidia/DeepSeek-V4-Flash-NVFP4` (createdAt 2026-05-18T00:02Z) existiert —
> die Aussage in §5 „kein nvidia/DeepSeek-V4-Flash-NVFP4" war falsch. Siehe §5
> für die Korrektur und die Implikation für die Kapazitätsabschätzung.

Summary of everything that blocks or constrains serving **DeepSeek-V4-Flash** on
our 4×GB10 / SM121 cluster under SGLang. Context (as written 2026-05-31): the
then-active model (`roles/k8s_dgx/defaults/main.yml`) was
**`sgl-project/DeepSeek-V4-Flash-FP8`** on image
**`xomoxcc/dgx-spark-sglang:0.5.12.post1-sm121`** (SGLang v0.5.12.post1) — see
the 2026-06-08 note above; this is no longer the default.
`DeepseekV4ForCausalLM` is mainline since PR #23882 (shipped in v0.5.12), but
Flash serving is still being stabilized upstream.

| # | Issue | State | Impact for us |
|---|-------|-------|---------------|
| 1 | `kv_lora_rank=None` strict-dataclass crash at config parse | **Worked around** (launch patch) | Blocks ANY V4-Flash checkpoint until patched |
| 2 | compressed-tensors `wqkv_a` vs `fused_wqa_wkv` target mismatch | **Open upstream** (#23724) | Makes RedHatAI / kylesayrs / canada-quant NVFP4 unloadable |
| 3 | NVFP4 MoE / FP4 indexer for V4 not implemented | **Partial** — FP4 indexer #26209 merged 2026-06-02 (main, in v0.5.13-Tag); NVFP4 MoE #25820 open (Flash NVFP4 working on B200 per 2026-06-11 comment, kein SM121-Test) | Kein nutzbarer NVFP4-Pfad auf SGLang für uns bis #25820 gemergt |
| 4 | NVFP4 runner instability on V4-Flash/Pro | **Open / partly closed** (#26324, #25704) | Even where NVFP4 loads, output is NaN/garbage except EAGLE |
| 5 | `nvidia/DeepSeek-V4-Pro-NVFP4` does not fit; `nvidia/DeepSeek-V4-Flash-NVFP4` needs verification | N/A (capacity) | Pro: 913 GB vs 512 GB — does not fit. Flash NVFP4 (~162B params) may fit TP=4 — unverified, see §5 |

**Net conclusion (Stand 2026-05-31, teilweise überholt):**
`sgl-project/DeepSeek-V4-Flash-FP8` (block-wise FP8) is the **only viable V4
path on SGLang** for this cluster today. NVFP4 of V4-Flash remained a dead end
on two independent axes — incomplete runtime support (#25820 offen) **and**
fehlender SGLang-nativer Checkpoint. ~~no loadable checkpoint~~ — **Update
2026-06-12:** `nvidia/DeepSeek-V4-Flash-NVFP4` existiert seit 2026-05-18 (§5).
Die Achse „kein Checkpoint" entfällt damit sobald #25820 gemergt ist; die Achse
„unvollständige Runtime" bleibt bis dahin bestehen. NVFP4 via vLLM (RedHatAI
checkpoint, vLLM-targeted) ist weiterhin eine Alternative.

---

## 1. `kv_lora_rank=None` — strict-dataclass crash at config parse (WORKED AROUND)

**Symptom** — head/worker exit immediately at startup:
```
huggingface_hub.errors.StrictDataclassFieldValidationError: Validation error for
field 'kv_lora_rank':
    TypeError: Field 'kv_lora_rank' expected int, got NoneType (value: None)
```

**Root cause:**
- DeepSeek-V4-**Flash** `config.json` has `kv_lora_rank: null` — legitimate: Flash
  uses q-LoRA + o-LoRA + GQA (`q_lora_rank=1024`, `o_lora_rank=1024`,
  `num_key_value_heads=1`), **no MLA KV compression**, so `kv_lora_rank` is null.
- SGLang parses `model_type=deepseek_v4` via
  `_DeepseekV4ConfigAlias(DeepseekV3Config)` in
  `sglang/srt/utils/hf_transformers/common.py` (V4 "reuses the V3 config schema").
  The field `kv_lora_rank: int = 512` is declared on transformers'
  `DeepseekV3Config` (`transformers/models/deepseek_v3/configuration_deepseek_v3.py:81`).
- Under **transformers 5.x**, `PreTrainedConfig` is a `huggingface_hub` `@strict`
  dataclass, so the validator built from that annotation rejects `None`. Under
  transformers <5 (non-strict) the null was silently tolerated — which is why
  external guidance says "pin transformers==4.57.1". **We can't**: our image pins
  transformers 5.8.0 for the Gemma-4 assistant drafter.

**Not the file you'd expect:** SGLang's own `sglang/srt/configs/deepseek_v4.py`
(`DeepSeekV4Config`) is a *runtime/modeling* class used by `models/deepseek_v4.py`
and the DSA indexer/compressor — it is **NOT** the class that parses `config.json`
for `model_type=deepseek_v4`. Editing it has no effect on this crash.

**Workaround (ours):** a launch-time source patch in
`roles/k8s_dgx/files/sglang_launch.sh` widens the annotation to `int | None`
**before** `python3 -m sglang.launch_server` (so `@strict` rebuilds a Union
validator that accepts None; pyc is timestamp-invalidated so the edit takes):
```
transformers/models/deepseek_v3/configuration_deepseek_v3.py:
    kv_lora_rank: int = 512   →   kv_lora_rank: int | None = 512
```
Safe widening — DeepSeek-V3 / V3.2 / Kimi-K2 (which share this config) always
supply an int; only V4-Flash's null now additionally passes. Verified in a
throwaway pod: `field.type` becomes `int | None` and `cls(kv_lora_rank=None)`
constructs cleanly. Log line on success:
`Patched DeepseekV3Config: kv_lora_rank now int|None (...)`.

**Upstream status:** not fixed — `kv_lora_rank: int = 512` is still present on
`main` and in v0.5.12.post1. No targeted upstream issue filed; related to the
general "V4 reuses V3 schema" design and the open Flash-serving tracker
(#25165 / #23743). Remove the launch patch once upstream types it Optional or
registers transformers' native `DeepseekV4Config` (which has no `kv_lora_rank`).

---

## 2. compressed-tensors module-naming mismatch — `wqkv_a` vs `fused_wqa_wkv` (OPEN)

**Symptom** — after the config parse succeeds, model build fails:
```
ValueError: Unable to find matching target for model.layers.0.self_attn.wqkv_a
            in the compressed-tensors config.
```
(`compressed_tensors/utils.py:find_matched_target` → no scheme, empty `ignore`.)

**Root cause:** third-party NVFP4 repackages (RedHatAI, kylesayrs, canada-quant)
are **compressed-tensors** format whose attention target regex is
`re:.*attn.*(fused_wqa_wkv|wq_b|wo_a|wo_b)$`. SGLang's V4 implementation names the
fused proj **`wqkv_a`** (`models/deepseek_v4.py:265`). 3 of 4 names match
(`wq_b`, `wo_a`, `wo_b`); only `wqkv_a` ≠ `fused_wqa_wkv` → no match → hard error.
SGLang's V4 loader expects deepseek-ai/sgl checkpoint naming
(`remap_weight_name_to_dpsk_hf_format`); the third-party repos follow
vLLM/transformers naming instead.

**Affected checkpoints (all hit this identically):**
- `RedHatAI/DeepSeek-V4-Flash-NVFP4-FP8` — target `fused_wqa_wkv`
- `kylesayrs/DeepSeek-V4-Flash-NVFP4-FP8-BLOCK` — target `fused_wqa_wkv`
- `canada-quant/DeepSeek-V4-Flash-NVFP4-FP8-MTP` — target `fused_wqa_wkv` (+ `compressor.fused_wkv_wgate`)
- None ship `hf_quant_config.json` → **none are ModelOpt format** (which would
  bypass the compressed-tensors matcher entirely).

**Upstream status:** compressed-tensors W4A16 support for V4 is an open feature —
[#23724](https://github.com/sgl-project/sglang/issues/23724). Not launch-patchable
in a robust way (matcher + weight-name remap + an NVFP4 scheme that doesn't exist
yet — see §3). **Fix = use a SGLang-native checkpoint**, not a patch.

---

## 3. NVFP4 MoE / FP4 indexer for DeepSeek-V4 — not implemented (OPEN)

Even with a correctly-named NVFP4 checkpoint, SGLang's V4 path lacks NVFP4
support:
- `models/deepseek_v4.py` has exactly one quant branch: `w4afp8` (line ~1401).
  No NVFP4 / ModelOpt path for the attention projections.
- [#25820](https://github.com/sgl-project/sglang/issues/25820) **[NVIDIA] Support
  NVFP4 MoE for DeepSeek-V4** — open.
- [#26209](https://github.com/sgl-project/sglang/issues/26209) **Add FP4 Indexer
  for DeepSeek V4** — **merged 2026-06-02** (in `main`, merged_at
  2026-06-02T07:14:39Z; enthalten im v0.5.13-Tag). Der DSA/Index-Attention-Pfad
  ist damit upstream verfügbar, aber noch kein GitHub Release.
- Roadmap: [#23602](https://github.com/sgl-project/sglang/issues/23602) DeepSeek V4
  Roadmap.

---

## 4. NVFP4 runner instability on V4 (OPEN / partly closed)

Where NVFP4 V4 does run (e.g. Pro on B200), it is buggy:
- [#26324](https://github.com/sgl-project/sglang/issues/26324) — `flashinfer_trtllm`
  MoE runner **asserts on DeepSeek-V4-Flash NVFP4 on B200** (and corrupts
  MiniMax-M2.7-NVFP4 output).
- [#25704](https://github.com/sgl-project/sglang/issues/25704) (closed) — DeepSeek-V4
  **Pro** NVFP4 on B200: non-speculative decode produces **NaN (TP=8) or garbage
  tokens** (DP+DeepEP); **only EAGLE works**.

On **SM121** specifically, our own `cutlass_moe_fp4` crashes apply on top (see
`CLAUDE.md` NVFP4 section, `TURBOQUANT.md`, `SGLANG_TP_EP_MOE_UPSTREAM_BUG.md`).

---

## 5. `nvidia/DeepSeek-V4-Pro-NVFP4` does not fit — `nvidia/DeepSeek-V4-Flash-NVFP4` needs verification (capacity, not a bug)

> **Update 2026-06-12:** Die ursprüngliche Aussage „there is no
> `nvidia/DeepSeek-V4-Flash-NVFP4`" und „the only NVIDIA ModelOpt-NVFP4 of V4
> is the Pro variant" waren **falsch**. Das HF-Repo
> `nvidia/DeepSeek-V4-Flash-NVFP4` existiert seit 2026-05-18T00:02Z (zuletzt
> geändert 2026-06-10T19:33Z, im Zuge des PR #25820 quant-config-Fix-Zyklus),
> 46 Safetensors-Shards (vs. 64 für Pro). `hf_quant_config.json` bestätigt:
> producer `modelopt`, version `dsv4-nvfp4-experts`, quant_algo
> `MIXED_PRECISION`, NVFP4 auf `layers.*.ffn.experts`, group_size 16 — gleiches
> Format wie das Pro-Checkpoint. Voraussetzung für die Nutzung: PR #25820 muss
> gemergt sein (Stand 2026-06-12: offen, CI-blocked, vgl. Status-Block oben).
> Kapazitäts-Implikation: Die §5-Abschätzung „does not fit" galt für Pro (913
> GB); Flash NVFP4 (~162 B params, NVFP4 experts, ~46 Shards) ist deutlich
> kleiner und könnte bei TP=4 in 4×128 GB passen — **muss noch verifiziert
> werden**, war bisher keine betrachtete Option.

**`nvidia/DeepSeek-V4-Pro-NVFP4`** — Pro-Variante (nicht lauffähig auf diesem Cluster):
- **913 GB** of weights on disk (64 safetensors), **~910 B params**, 61 layers,
  hidden 7168, 384 routed experts.
- ModelOpt mixed precision: NVFP4 on **experts only** (`hf_quant_config.json`,
  producer `dsv4-nvfp4-experts`, `awq_block_size 16`); attention/dense/shared/
  router/embeddings stay FP8/BF16 → net ~1 byte/param.

Cluster has **4×128 GB = 512 GB** total. At TP/EP=4 that is **~228 GB/GPU**
needed vs **128 GB** available — ~1.8× over capacity, before KV cache /
activations / CUDA context. Even hypothetical pure NVFP4 (~455 GB) leaves no
runtime headroom. **Pro does not fit.**

**`nvidia/DeepSeek-V4-Flash-NVFP4`** — Flash-Variante (~162 B params, 46
safetensors-Shards): Kapazitäts-Abschätzung steht aus. Bei TP=4 und ~1
byte/param sind ~162 GB Gewichte zu erwarten — nominal in 4×128 GB passend,
aber KV-Cache / Aktivierungen / CUDA-Kontext müssen gegengerechnet werden.
Erst sinnvoll zu testen, sobald PR #25820 gemergt ist.

---

## 6. SM121 (GB10) boot chain for `DeepSeek-V4-Flash-FP8` — the walls, in order

Bringing the block-FP8 checkpoint up on 4×GB10 / SM121 is a sequence of
first-contact failures. Each is fixed by a profile flag in
`sgl-project-deepseek-v4-flash-fp8.yml` (plumbed into BOTH serving and the
`sglang_shard` Job — the runtime-kernel and weight-dtype choices must match
between shard-save and serve). The walls span config-parse → weight load →
warmup → and (verified 2026-05-31) the **first real forward request**: model
load + `Uvicorn running` is NOT the finish line — walls 6–8 only fire when the
first prompt is actually decoded. The order they surface:

| # | Stage | Symptom | Fix |
|---|-------|---------|-----|
| 1 | config parse | `StrictDataclassFieldValidationError: kv_lora_rank ... NoneType` | §1 launch patch (int → int\|None) |
| 2 | weight load | `ValueError: Downcasting not allowed: target fp8, loaded bf16` (on `wo_a`) | `SGLANG_OPT_FP8_WO_A_GEMM=0` |
| 3 | mem pool | `RuntimeError: Not enough memory. Please try to increase --mem-fraction-static` | raise `mem_fraction_static` to **0.90** |
| 4 | kernel warmup | `tf32_hc_prenorm_gemm → InternalError ... hyperconnection.hpp:56 Unsupported architecture` | `SGLANG_OPT_DEEPGEMM_HC_PRENORM=0` |
| 5 | decode (pre-empted) | `paged_mqa_logits` DeepGEMM kernel, same SM121 gap | `SGLANG_FP8_PAGED_MQA_LOGITS_TORCH=1` |
| — | weight load (host RAM) | kernel **SIGKILL** (no CUDA trace) at end of load — MoE expert fusion in `process_weights_after_loading` doubles unified-memory footprint | node **swap** (see below) |
| 6 | first forward | `AssertionError: assert seq_lens.shape == (batch_size,)` in `dsv4/indexer.py:fp8_paged_mqa_logits_torch` | launch source-patch (squeeze trailing singleton) |
| 7 | first forward (decode) | `topk_v2.cuh:472 CUDA error: invalid argument` — TVM jit `topk_transform_512_v2` | `SGLANG_TOPK_TRANSFORM_512_TORCH=1` |
| 0 | first forward | `RuntimeError: Unsupported architecture for sparse decode fwd` (FlashMLA) | baked FlashMLA sm_121a kernel + `.pth` hook — see **§7** |

**(2) `wo_a` fp8→bf16 — `SGLANG_OPT_FP8_WO_A_GEMM` (#25181, default-on since 0.5.12).**
The opt keeps the o-LoRA `wo_a` projection fp8 and runs an fp8 GEMM. V4-Flash-FP8
ships `wo_a` as **bf16**, so with the opt on the param is created fp8 and
`copy_with_check` (parameter.py) refuses the bf16→fp8 downcast. `=0` dequants
`wo_a` to bf16 (`deepseek_v4.py:_dequant_fp8_wo_a`), matching the checkpoint.
Same error class as [#19589](https://github.com/sgl-project/sglang/issues/19589)
(Qwen3.5-FP8). Cost is trivial — `wo_a` is low-rank (o_lora_rank=1024): ~1.44 B
params total, **+1.44 GB bf16-vs-fp8 (~0.36 GB/rank at TP=4)**.

**(3) `mem_fraction_static`.** Not a bug — a sizing constraint. The flag governs
the post-weight-load KV reserve, NOT a vLLM-style fraction-of-total: KV pool =
`post_load_free − pre_load_free × (1 − mem_fraction_static)`. Weights are ~73.5 GB
≈ 0.61 of the 121 GiB node, so the old `0.50` left a *negative* KV pool → the
"Not enough memory" error. **Higher = more KV.** 0.90 → ~28 GB KV, safe because
the profile disables CUDA-graph capture (the held-back reserve only covers
activations; MQA `num_key_value_heads=1` → tiny per-token KV).

**(4)+(5) DeepGEMM has no sm_121 kernels for two V4 families**
([DeepGEMM #317](https://github.com/deepseek-ai/DeepGEMM/issues/317),
[SGLang #23657](https://github.com/sgl-project/sglang/issues/23657)):
`tf32_hc_prenorm_gemm` (Manifold-Constrained Hyper-Connections / MHC) and
`paged_mqa_logits` (DSA Lightning-Indexer) both assert *Unsupported architecture*
on SM120/121. SGLang has fallbacks gated by env, OFF by the wrong default for us:
- `SGLANG_OPT_DEEPGEMM_HC_PRENORM=0` → MHC prenorm uses the **TileLang** path
  (`mhc.py:mhc_pre_gemm_sqrsum_tilelang`). Hits at `kernel_warmup`, then every
  forward. **Independent of `disable_deep_gemm`** (that's `SGLANG_ENABLE_JIT_DEEPGEMM`,
  the JIT-MoE switch — it does NOT cover these two kernels).
- `SGLANG_FP8_PAGED_MQA_LOGITS_TORCH=1` → indexer uses the **torch** fallback.
  Surfaces at decode, not load — pre-emptively flipped.
Both fallbacks are slower than the kernels but correct; perf-tune later (cluster
is CPU-TCP-bound, so the delta may not matter). **Both are runtime-kernel choices
→ required at inference, not just sharding.**

**Host-RAM OOM during load (the swap story).** On GB10 unified memory (CPU+GPU
share ~121 GiB), `process_weights_after_loading` builds the fused MoE expert
tensors while the per-expert sources are still resident → a transient ~2× spike
that SIGKILLs at the end of load (host kernel OOM, no CUDA trace — proving it's
host-pageable, not device-pinned). Fixed with a disk-backed node swapfile
(KEP-2400). Per-pod control via QoS under kubelet `swapBehavior=LimitedSwap`:
the `sglang_shard` Job is **Burstable** (memory request) → gets swap → absorbs
the spike; the serving Deployment is **BestEffort** (GPU-limit only) → `memory.swap.max=0`
→ never swaps. This is *why* sharded_state matters operationally: the serve load
skips the fusion, so it fits in RAM without swap (and serving must not swap —
paging weights/activations would wreck latency). Swap device: `dgx_prepare`
tag `swap`; kubelet policy: `k3sserver` gate `k3s_node_swap_enabled`.

**(6) C4-indexer torch fallback asserts 1-D `seq_lens` (forward-time).** Enabling
`SGLANG_FP8_PAGED_MQA_LOGITS_TORCH=1` (wall 5) routes the indexer to
`dsv4/indexer.py:fp8_paged_mqa_logits_torch`, but its caller `forward_c4_indexer`
unconditionally unsqueezes `c4_seq_lens` to 2-D `(batch, 1)` for the deep_gemm /
tilelang kernels — and the torch fallback `assert seq_lens.shape == (batch_size,)`.
→ `AssertionError` on the **first multi-token forward** (any prompt, not just
EAGLE; speculative is off here). sglang's own gap. **Note:** the vendored
0xSero `_patch_sglang_indexer_fallbacks()` was meant to fix exactly this but
targets the OLD module paths (`nsa.tilelang_kernel`, `compressed.indexer`) — on
v0.5.12.post1 the indexer moved to `attention/dsv4/indexer.py`, so it patches
nothing. Fix = a launch-time source patch that squeezes a trailing singleton
before the assert (`sglang_launch.sh`, marker `assert seq_lens.shape ==
(batch_size,)`; no-op when already 1-D).

**(7) C4-indexer top-k transform TVM kernel dies on sm_121 (decode).** The next
wall, immediately after (6): `forward_c4_indexer` picks `topk_transform_512_v2`
(a TVM jit kernel, `sglang/jit_kernel/...`) when `SGLANG_OPT_USE_TOPK_V2`
(default-on) and no capture buffer. On GB10 it aborts with
`topk_v2.cuh:472 CUDA error: invalid argument` at decode. sglang ships a pure-
torch fallback `topk_transform_512_pytorch_vectorized`, selected by
`SGLANG_TOPK_TRANSFORM_512_TORCH=1` — and the torch branch is checked **before**
the v2 branch, so the flag alone wins (no need to also clear `OPT_USE_TOPK_V2`).
Profile flag `topk_transform_512_torch: true` → env, plumbed serving-side
(decode-time kernel; irrelevant to the shard Job). Same fallback family as (5).

**Post-load warmup latency.** With (4)/(5)/(7) on the torch/TileLang fallbacks,
the **first** request after startup takes ~2 min (one-time JIT/compile of the
TileLang MHC-prenorm path et al. on first real forward) — the request can outlast
a 120 s client timeout while the server still completes it (200 OK). Warm
requests are normal latency. Do not mistake the first-request stall for a hang.

**On `load_format`.** Verified 2026-05-31: `load_format: auto` (in-process MoE
fusion, no pre-sharding) loaded cleanly on the BestEffort serving Deployment
**without** triggering the host-RAM SIGKILL of the swap story above — the ~2×
fusion spike fit under 121 GiB for this checkpoint at TP=4. `sharded_state` (skip
fusion at serve) remains the swap-free design, but is not strictly required here.

---

## 7. FlashMLA sparse-decode kernel for sm_121a + the `.pth` hook (wall #0)

V4's attention backend (`deepseek_v4_backend.py`) hard-imports `flash_mla` and
calls `flash_mla.flash_mla_with_kvcache(...)` with **no fallback**, and upstream
FlashMLA ships no sm_120/sm_121 sparse-decode kernel → the first forward dies
`RuntimeError: Unsupported architecture for sparse decode fwd`. We bake a
sm_121a kernel into the image (vendored from `0xSero/deepseek-v4-flash-sm120`,
retargeted `sm_120a → sm_121a`) + stock `flash_mla` for the interface, and
monkey-patch `flash_mla_with_kvcache` at interpreter startup. Three sub-walls,
each a non-obvious trap (all hit during bringup 2026-05-31):

**(a) The hook never ran — Ubuntu shadows our `sitecustomize.py`.** The kernel
ships `sitecustomize_hook.py`; dropping it at
`/usr/local/lib/python3.12/dist-packages/sitecustomize.py` looks right but Python
imports only the **first** `sitecustomize` on `sys.path`, and Ubuntu's
`/usr/lib/python3.12/sitecustomize.py` (apport) sits earlier → ours never loaded,
patch never applied, stock FlashMLA crashed. Fix = a **`.pth`** file instead:
`site.py` executes the `import` line in **every** `.pth` across all site dirs (no
"first wins"), in main **and** every spawned sglang worker. Drop
`zz_dsv4_autopatch.pth` → `import dsv4_autopatch`.

**(b) Don't call `install()` — it loads tilelang's libcudart stub too early.**
The kernel's `patch_flash_mla()` (= `install()`) also runs
`_patch_sglang_indexer_fallbacks()`, which imports `sglang…nsa.tilelang_kernel`
→ loads `tilelang/lib/libcudart_stub.so`. If that stub loads **before**
`flashinfer.comm`, flashinfer's `find_loaded_library("libcudart")` grabs the stub
(missing `cudaDeviceReset`) → `AttributeError` at import — a HARD crash, NOT caught
by sglang's `except ImportError` (unlike the benign `cuda.tile` ModuleNotFoundError
from flashinfer 0.6.12, which sglang tolerates → allreduce-unavailable fallback).
Fix = the autopatch calls **only** `_patch_flash_mla_pkg()` (installs the wrapper,
no tilelang). sglang imports tilelang itself LATER, after flashinfer, in its
natural (proven-safe) order; bootstrapping it early is not our job. (And
`_patch_sglang_indexer_fallbacks()` is a no-op here anyway — see §6 wall 6.)

**(c) The wrapper is gated, inert for non-V4.** `_make_wrapper` only takes over
when `indices is not None` AND device-cap major == 12 (GB10 reports `(12, 1)`)
AND `is_fp8_kvcache` AND `q.element_size() == 2`; otherwise it defers to stock
flash_mla. So the `.pth` is safe to ship image-wide.

Artifacts: `scripts/patches/dockerfile-dsv4-flashmla.patch` (bakes kernel +
`.pth` at build), `sglang_launch.sh` (writes `dsv4_autopatch.py` + `.pth` at
runtime — no rebuild needed), recipe knobs `FLASH_MLA_*` / `DSV4_KERNEL_*` in
`scripts/patches/sglang-0.5.12-sm121.recipe`.

---

## What we actually run: `sgl-project/DeepSeek-V4-Flash-FP8`

SGLang's own pre-converted checkpoint ([#24111](https://github.com/sgl-project/sglang/issues/24111)):
- Plain **block-wise FP8** — `quant_method=fp8`, `weight_block_size=[128,128]`,
  `scale_fmt=ue8m0`. **Not compressed-tensors**, so §2 does not apply; quant is
  auto-detected (no explicit flag), same as `zai-org/GLM-4.7-FP8`.
- Profile: `roles/k8s_dgx/model_profiles/sgl-project-deepseek-v4-flash-fp8.yml`.
  SM121 FP8 kernel choices mirror GLM-4.7-FP8: `moe_runner_backend=triton`,
  `fp8_gemm_runner_backend=cutlass`, `disable_deep_gemm=true`,
  `attention_backend=flashinfer`.
- Recommended sampling (model card / `generation_config.json`): **temperature 1.0,
  top_p 1.0** (`do_sample=true`, no top_k).
- MTP: per the SGLang DeepSeek-V4 cookbook, V4-Flash MTP runs under the **EAGLE**
  framework and requires env `SGLANG_ENABLE_SPEC_V2=1`. Presets — balanced
  (steps=1, draft=2), low-latency (steps=3, draft=4), eagle-topk=1;
  max-throughput disables MTP. Left disabled for first-contact.
- **Still requires the §1 launch patch** — same architecture, same
  `kv_lora_rank: null`.

Why the image is `0.5.12.post1` (not `.12`): the .post1 cherry-picks touch DSv4
FP8 paths directly — **#25733/#26063** (single-token-decode garbled text via
`deep_gemm` ue8m0 scale-packing — our checkpoint is exactly ue8m0 block FP8) and
**#25646/#26072** (HiSparse GSM8K accuracy 0.825→0.960). See
`SGLANG_v0.5.12.post1_VERSION_CHANGES.md`.

**Status — end-to-end verified 2026-05-31.** With walls 1–7 + the §7 FlashMLA
hook in place, the 4×GB10 deployment loads, serves, and **decodes coherent
output** (correct factual answer, `200 OK`, clean token distribution) — confirmed
by a real `/v1/completions` request at TP=4. The earlier "expect issues past
model load" warning was correct: walls 6, 7 and §7 all surfaced only on the first
forward, past `Uvicorn running`. Subsequent first-contact risk is lower but not
zero — Flash serving is still stabilizing upstream
([#25165](https://github.com/sgl-project/sglang/issues/25165) main "broke" with
V4-Flash, [#23743](https://github.com/sgl-project/sglang/issues/23743) GB200
serving tracker, [#25526](https://github.com/sgl-project/sglang/issues/25526)
HiCache piecewise CUDA graph, [#26647](https://github.com/sgl-project/sglang/issues/26647)
Mooncake HiCache hybrid cache) — features we don't enable (HiCache, MTP/EAGLE,
PD-disagg) may surface new walls if turned on.

---

## Upstream references

| Ref | Title | State |
|-----|-------|-------|
| PR #23882 | DeepSeek-V4 day-0 support (`DeepseekV4ForCausalLM`) | merged (v0.5.12) |
| PR #24692 | feat: SM120 (Blackwell Desktop) support for DeepSeek-V4 inference | **merged 2026-06-01; im v0.5.13-Tag (2026-06-11) enthalten; auf PyPI als 0.5.13 seit 2026-06-11T10:16Z; noch kein GitHub-Release-Page (Stand 2026-06-12)** |
| #23602 | DeepSeek V4 Roadmap | open |
| #23724 | Support DeepSeek-V4 Compressed-tensor W4A16 | open |
| #25820 | [NVIDIA] Support NVFP4 MoE for DeepSeek-V4 | open (2026-06-11: Flash NVFP4 working auf B200, GSM8K 96.21%; kein SM120/121-Test) |
| #26209 | Add FP4 Indexer for DeepSeek V4 | **merged 2026-06-02 into main; im v0.5.13-Tag (2026-06-11) enthalten; auf PyPI als 0.5.13 seit 2026-06-11T10:16Z; noch kein GitHub-Release-Page (Stand 2026-06-12)** |
| #26324 | flashinfer_trtllm MoE runner asserts on DeepSeek-V4-Flash NVFP4 (B200) | open |
| #25704 | V4-Pro NVFP4 B200: NaN/garbage except EAGLE | closed |
| #25165 | main branch broke with deepseek v4 flash deployment | open |
| #23743 | [Tracking] DeepSeek V4 Flash GB200 serving fixes | open |
| #25526 | DSv4 Flash + HiCache breakable piecewise CUDA graph | open |
| #26647 | Mooncake HiCache fails with DeepSeek-V4-Flash hybrid cache | open |
| #24111 | About pre-converted FP8 checkpoints (sgl-project/DeepSeek-V4-Flash-FP8) | open |
| DeepGEMM #317 | DeepSeek-V4 on SM120: `tf32_hc_prenorm_gemm` + `paged_mqa_logits` kernels missing | **closed 2026-04-30 (declined, no SM120 HW; community PR #318 open)** |
| #23657 | DSv4 compressed attention: no SM120 fallback for Lightning Indexer | open |
| #25181 | `SGLANG_OPT_FP8_WO_A_GEMM` default-on | merged (v0.5.12) |
| #19589 | Qwen3.5 FP8 "Downcasting not allowed" (same error class as §6/2) | **closed 2026-05-02** |

## 8. PR #24692 — SM120-Support für DeepSeek-V4-Inference (update 2026-06-11)

**PR:** [#24692](https://github.com/sgl-project/sglang/pull/24692) „feat: SM120
(Blackwell Desktop) support for DeepSeek-V4 inference" — merged 2026-06-01,
enthalten im v0.5.13-Tag (2026-06-11). Cluster läuft noch auf
v0.5.12.post1-basierten Images; dieser Abschnitt dokumentiert, was sich mit
einem v0.5.13-Image ändert.

**Scope des PR:**
- `flash_mla_sm120.py` — nativer SM120-FlashMLA-Triton-Kernel, aktiviert via
  `_is_sm120=True` in `deepseek_v4_backend.py` (Check: `major==12`, gilt also
  auch für SM121/GB10).
- `fp8_paged_mqa_logits_torch_sm120` in `indexer.py` — SM120-Pfad für den
  FP8-Paged-MQA-Indexer; handhabt den `seq_lens`-Squeeze intern.
- MXFP4 MoE Triton-Fallback für SM120 (neu im V4-Modellpfad).

**Auswirkung auf unsere Walls (Stand v0.5.12.post1 → v0.5.13):**

| Wall | Titel | Status auf v0.5.12.post1 | Status ab v0.5.13 |
|------|-------|--------------------------|-------------------|
| 0 | FlashMLA sparse-decode (SM121) | 0xSero-vendored Kernel + `.pth`-Hook nötig (§7) | **Redundant** — nativ via `deepseek_v4_backend.py` (`_is_sm120=True`); Hook kann entfernt werden |
| 1 | `kv_lora_rank=None` strict-dataclass crash | Launch-Patch nötig (`int → int\|None`) | **Weiterhin nötig** — `kv_lora_rank: int = 512` in `configuration_deepseek_v3.py` unverändert; `_DeepseekV4ConfigAlias` in v0.5.13 überschreibt das Feld nicht |
| 2 | `wqkv_a` vs `fused_wqa_wkv` target mismatch | Nicht patchbar (§2) | **Unverändert** — kein Fix in #24692 |
| 3 | NVFP4 MoE / FP4-Indexer | FP4-Indexer merged (#26209), MoE-Pfad offen | **Teilweise** — #26209 in v0.5.13 (PyPI seit 2026-06-11); NVFP4 MoE (#25820) weiterhin offen/CI-blocked (Stand 2026-06-12) |
| 4 | DeepGEMM MHC-prenorm (SM121) | `SGLANG_OPT_DEEPGEMM_HC_PRENORM=0` nötig | **Weiterhin nötig** — `configurer.py` gated DeepGEMM bei exakt `sm_version==120`, nicht 121; `mhc.py` prüft nur `SGLANG_OPT_DEEPGEMM_HC_PRENORM`; kein Auto-Routing für SM121 in #24692 |
| 5 | `paged_mqa_logits` DeepGEMM SM121-Lücke | `SGLANG_FP8_PAGED_MQA_LOGITS_TORCH=1` nötig | **Weiterhin nötig** (Env-Var-Routing unverändert) |
| 6 | `seq_lens.shape`-Assert im torch-Fallback | Launch-Source-Patch nötig | **Redundant** — `fp8_paged_mqa_logits_torch_sm120` in v0.5.13 handhabt Squeeze intern; Patch kann entfernt werden |
| 7 | TVM `topk_transform_512_v2` auf SM121 | `SGLANG_TOPK_TRANSFORM_512_TORCH=1` nötig | **Weiterhin nötig** — `SGLANG_TOPK_TRANSFORM_512_TORCH` hat kein SM120/121-Auto-Routing in v0.5.13, Default weiterhin False |

**Zusammenfassung für den Image-Wechsel auf v0.5.13:**
- Walls 0 und 6 werden durch #24692 nativ gelöst → die entsprechenden
  Launch-Patches und der `.pth`-Hook können beim Wechsel auf v0.5.13 entfernt
  werden.
- Walls 1, 2, 4, 5, 7 bleiben unverändert und erfordern weiterhin die
  bestehenden Workarounds.
- Der 0xSero-vendored FlashMLA-Kernel im Image (`scripts/patches/`) wird mit
  v0.5.13 redundant; der Build-Patch kann entfernt werden.

---

## Local artifacts

- Launch patch: `roles/k8s_dgx/files/sglang_launch.sh` (DeepseekV3Config kv_lora_rank block)
- Model profile: `roles/k8s_dgx/model_profiles/sgl-project-deepseek-v4-flash-fp8.yml`
  — SM121 flags (§6): `opt_fp8_wo_a_gemm: false`, `opt_deepgemm_hc_prenorm: false`,
  `fp8_paged_mqa_logits_torch: true`, `mem_fraction_static: "0.90"`
- Env plumbing: `roles/k8s_dgx/tasks/sglang.yml` (serving ConfigMap) +
  `roles/k8s_dgx/tasks/sglang_shard.yml` (shard Job) — both read the same profile flags
- Node swap: device `roles/dgx_prepare/tasks/swap.yml` (tag `swap`, gate `k3s_node_swap_enabled`,
  size `dgx_swap_size`); kubelet policy `roles/k3sserver/templates/etc_rancher_k3s_kubelet-config.yaml.j2`
  (`failSwapOn: false`, `swapBehavior: LimitedSwap`)
- Active model: `roles/k8s_dgx/defaults/main.yml` (`sglang_model`)
- Image: `xomoxcc/dgx-spark-sglang:0.5.12.post1-sm121` — recipe `scripts/patches/sglang-0.5.12-sm121.recipe`
- Release notes: `SGLANG_v0.5.12.post1_VERSION_CHANGES.md`
