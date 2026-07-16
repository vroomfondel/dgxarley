# UPSTREAM PR: DSA sparse attention on SM120/SM121 via flashinfer's native sparse-MLA backend

Status: **NOT YET FILED** (drafted 2026-07-16). Local implementation =
`roles/k8s_dgx/files/sglang_patches/p34_dsa_trtllm_sparse_sm120.py`, live-proven
on the dgxarley cluster (4× DGX Spark GB10, SM121, TP4,
`0xSero/glm-5.2-reap-504B-v2` = GlmMoeDsaForCausalLM). After a merge lands in a
release we build on: DELETE p34 (its docstring carries the same re-sync rule).

Upstream state (checked 2026-07-16): `sglang/srt/layers/attention/dsa_backend.py`
on `main` still hardcodes `backend="trtllm-gen"` in `_forward_trtllm`; no cc-12x
branch exists anywhere in the DSA path.

## Proposed PR title

> [DSA] Enable sparse MLA decode+prefill on SM120/SM121 (consumer Blackwell) via
> flashinfer's packed sparse backend

## PR body draft (English)

### Problem

`attention_backend=dsa` with `dsa_decode_backend/dsa_prefill_backend=trtllm`
crashes on SM120/SM121 (DGX Spark GB10, RTX PRO Blackwell) with
`TllmGenFmhaRunner ... Unsupported architecture`: `_forward_trtllm` hardcodes
`backend="trtllm-gen"`, which only exists for datacenter Blackwell (SM100/103).

flashinfer (>= 0.6.x) already ships a native SM120/121 sparse-MLA implementation
(`flashinfer/mla/_sparse_mla_sm120.py`, `@supported_compute_capability([120, 121])`,
GLM_NSA/DSv3.2 model types, warp-spec decode kernels for num_tokens<=64 plus a
prefill orchestrator above that), and its dispatcher
`trtllm_batch_decode_with_kv_cache_mla` routes `cc==12 && sparse_mla_top_k>0`
to it automatically — but only with `backend="auto"`. Three small changes make
the whole DSA path (decode AND prefill) work on consumer Blackwell.

### Changes

1. **`model_runner_kv_cache_mixin.py::calculate_mla_kv_cache_dim`** — do not
   early-return the plain `kv_lora_rank + qk_rope_head_dim` layout for
   trtllm-backends on SM12x. The SM120 sparse kernel consumes the 656-byte
   packed inline-scale layout (512 fp8 nope + 4×fp32 tile scales + 64 bf16
   rope) that `quantize_k_cache` already writes; `dsa_kv_cache_store_fp8` then
   derives True automatically. SM100 keeps the early return (plain layout for
   trtllm-gen).
2. **`dsa_backend.py::_forward_trtllm`** — gate on
   `device_sm_major == 12 and self.dsa_kv_cache_store_fp8`:
   - `backend="auto"` instead of `"trtllm-gen"` (flashinfer picks `"sparse"`),
   - pass the KV cache as a `uint8` view (the sm120 checker requires it),
   - `kv_scale_format="arbitrary_fp32"`: sglang's `quantize_k_cache` writes
     amax/448 **arbitrary** fp32 tile scales, not pow2/ue8m0 — flashinfer's
     GLM_NSA scale semantics; the default "auto" (pow2) would misread them,
   - `skip_softmax_threshold_scale_factor=None` (unsupported by the sparse
     backend),
   - skip the fused rope+fp8-query-quantize branch (`mla_quantize_and_rope_for_fp8`)
     — the sparse kernel requires a **BF16 query** and dequants KV itself via
     the inline scales (first live deploy died at decode graph capture with
     "SM120 sparse MLA v32/GLM expects BF16 query, got torch.float8_e4m3fn").
3. **`deepseek_common/attention_forward_methods/forward_mla.py::_fuse_rope_for_trtllm_mla`**
   — return False for the dsa branch on SM12x so rope stays in
   `forward_absorb_prepare` and the query reaches `_forward_trtllm` in bf16
   (consequence of 2; the second call site — the extra cos_sin_cache args —
   is gated by the same function and stays consistent).

All conditionals keep the upstream values on the non-SM12x side: SM100/SM103
behaviour is byte-identical.

### Evidence (DGX Spark GB10, SM121, flashinfer 0.6.14)

- Kernel-level (GPU, real `quantize_k_cache` pool, torch dequant+softmax
  reference): decode bs=4 topk=2048 max|diff| 0.008 @ 0.072 ms/call; prefill
  2400 extend tokens max|diff| 0.016 @ 14.4 ms/layer-call; seq_lens>topk
  clamps safely (sglang passes UNCLIPPED cache_seqlens on decode); -1 padding
  skipped natively; cuda-graph capture+replay works directly.
- Live (GlmMoeDsa 504B NVFP4, TP4): boot clean, decode graph capture 18 s,
  decode 8.4 tok/s cuda-graph, prefill 873 tok/s input on a 7-seq/2240-token
  batch, GSM8K 2-shot n=20 conc 8 = 85%, 0 errors/restarts. MTP/NEXTN verify
  runs through the same kernel (accept ~2.1, +45% decode).

### Notes for reviewers

- Requires flashinfer >= 0.6.x with `mla/_sparse_mla_sm120.py` (prebuilt in
  current wheels).
- The layout coupling in `calculate_mla_kv_cache_dim` ("plain iff a backend is
  named trtllm") is the reason change 1 is needed at all; a cleaner long-term
  fix might key the layout on the actually-selected kernel rather than the
  backend name.
- Hardware for validation: any SM120/121 device (DGX Spark, RTX PRO/50-series
  Blackwell). Datacenter-Blackwell CI is unaffected by construction.
- Related context: DeepGEMM upstream declined SM12x (PR #318), so consumer
  Blackwell needs the flashinfer route; the DSA indexer needs its own fallback
  (companion PR draft: UPSTREAM_SGLANG_DSA_INDEXER_TORCH_TRITON_BACKEND.md).

## Local mapping

| local | upstream file |
|---|---|
| p34 mixin edit | `sglang/srt/model_executor/model_runner_kv_cache_mixin.py` |
| p34 edits 0-3 | `sglang/srt/layers/attention/dsa_backend.py` (`_forward_trtllm`) |
| p34 forward_mla edit | `sglang/srt/models/deepseek_common/attention_forward_methods/forward_mla.py` |

Full local chronology: `dsalogitrework.md` (PART 4 + LIVE-DEPLOY RESULT),
`DSA_speedup.md`, `dsa_cuda_graph_plan.md` §8.

## Submission checklist (recherchiert 2026-07-16)

**Unit-Tests: JA (Guide-Pflicht), aber der Kernel selbst braucht SM12x-Hardware
— Split-Strategie:**
1. **CI-lauffähig ohne SM12x** (`test/registered/unit/` spiegelt den Source-
   Tree): (a) `calculate_mla_kv_cache_dim`-Unit-Test mit gemocktem
   `get_device_capability` (12 vs. 10) und Fake-Config → 656/576-Matrix
   (Testlogik existiert schon: unsere spark5-Validierung `validate_p34.sh`
   prüft exakt diese drei Fälle); (b) `_forward_trtllm`-Kwarg-Selektion:
   flashinfer-Call monkeypatchen, Fake-Backend mit `device_sm_major=12` +
   `dsa_kv_cache_store_fp8=True` → assert `backend="auto"`,
   `kv_scale_format="arbitrary_fp32"`, uint8-View, `skip_softmax=None`, und
   auf dem 10er-Pfad Byte-Gleichheit der Upstream-Kwargs.
2. **SM12x-gebundener Kernel-Test** (skip-gated wie die bestehenden
   sm120-Quant-Tests, z.B. `test/registered/quant/test_nvfp4_gemm_sm120.py`):
   der Numerik-Test aus unserer spark5-Validierung (echtes
   `quantize_k_cache`-Pool, torch-Referenz, decode/prefill/overshoot/graph) —
   läuft in CI nur, wenn ein SM120/121-Runner existiert, sonst skip.
3. Hardware-Evidenz im PR-Body (GB10-Zahlen aus diesem Dokument), da die
   Maintainer vermutlich kein SM121 in CI haben (gleiche Lage wie DeepGEMM
   PR #318).

**Mechanik:** wie beim Companion-PR (Fork/Branch, echte Diffs gegen main —
Pfad-Restrukturierung in main beachten, `pre-commit run --all-files`,
`register_cuda_ci` für die CI-fähigen Tests, DCO beim ersten Push prüfen).
flashinfer-Mindestversion (>= 0.6.x mit `_sparse_mla_sm120`) im PR nennen und
ggf. als Import-Guard kodieren.
