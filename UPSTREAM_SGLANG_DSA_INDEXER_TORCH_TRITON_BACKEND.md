# UPSTREAM PR: `dsa_paged_mqa_logits_backend=torch` fallback (+ Triton fast path) for the DSA indexer

Status: **NOT YET FILED** (drafted 2026-07-16). Local implementation =
`roles/k8s_dgx/files/sglang_patches/p30_dsa_torch_backend.py` (torch backend,
incl. next_n>=2) + `p35_dsa_indexer_triton_logits.py` (Triton fast path),
live-proven on dgxarley (DGX Spark GB10, SM121). After a merge lands in a
release we build on: DELETE p30 + p35.

Upstream state (checked 2026-07-16): `DSAPagedMQALogitsBackend` on `main` has
only DEEPGEMM / CUTEDSL / AITER; no arch-independent fallback exists. Companion
draft: UPSTREAM_SGLANG_DSA_SM12X_NATIVE_SPARSE_MLA.md (the attention side).

## Proposed PR title

> [DSA] Add an arch-independent `torch` paged-MQA-logits backend with a fused
> Triton fast path (unblocks DSA models on SM120/SM121)

## PR body draft (English)

### Problem

The DSA indexer's paged-MQA-logits kernel has no backend that runs on consumer
Blackwell (or any arch outside the current three):

- DEEPGEMM (the `auto` default): compiled C++ `Unsupported architecture` assert
  on SM120/121; DeepGEMM upstream declined SM12x support (PR #318).
- CUTEDSL: gated `is_sm100_supported()`, and structurally SM100-only —
  `_setup_mma` uses `tcgen05.MmaF8F6F4Op`, a datacenter-Blackwell ISA op.
- AITER: ROCm.

So every DSA model (DeepSeek-V3.2 family, GLM `GlmMoeDsaForCausalLM`) crashes
on its first decode step on SM120/121. The dsv4 path already solved this for
DeepSeek-V4 (`fp8_paged_mqa_logits_torch_sm120`, PR #24692), but the generic
`dsa/dsa_indexer.py` path has no equivalent.

### Changes

1. **New backend value `torch`** in `DSAPagedMQALogitsBackend` + server args
   (`--dsa-paged-mqa-logits-backend torch`). Opt-in only, NOT selected by
   `auto` — archs where DeepGEMM/CuteDSL work are unaffected.
2. **New module `dsa/torch_paged_mqa_logits.py`**: a vectorized, cuda-graph-safe
   (no `.item()`, no data-dependent control flow) pure-torch port of the dsv4
   fallback for the generic DSA path. Discards the DeepGEMM schedule metadata
   (`_ = deep_gemm_metadata`) — the torch path does no SM-tiled scheduling, so
   the two eager `get_paged_mqa_logits_metadata` call sites in
   `dsa_backend.py::init_forward_metadata` (and the graph-replay refresh
   helper) are skipped for this backend.
3. **Dispatch in `dsa_indexer.py::_get_topk_paged`**: pure pass-through for ALL
   modes — decode, target_verify, draft_extend(_v2). No reshaping is needed:
   q/weights are already sliced per token (`[:q_offset]`), verify seqlens come
   from `get_seqlens_expanded()` (per-token), and `init_forward_metadata`
   already `repeat_interleave`s the page table to per-token rows for every
   multi-token mode. (We learned this live: an extra repeat in the dispatch
   DOUBLE-expands and trips the kernel's shape assert at the MTP warmup.)
4. **Triton fast path `dsa/triton_paged_mqa_logits.py`** (env-gated,
   `SGLANG_DSA_INDEXER_TRITON`, default on; falls back to torch when triton is
   unavailable or num_heads < 16): one program per (token, 64-KV-page), STATIC
   launch grid over the full page-table width (cuda-graph-safe) with per-block
   EARLY EXIT on the true `seq_lens[b]` read at replay time; fused fp8 load +
   inline-scale dequant + q·k dot + relu + weighted head-sum, no fp32 HBM
   intermediates. Motivation: under cuda-graph the page-table width is a
   CAPTURE constant, and the pure-torch kernel pays the full width every step
   — 1.476 ms/layer at width 131072 regardless of the true seq len (~116 ms of
   a ~119 ms token on a 78-layer model). The Triton kernel is bit-exact vs the
   torch path and 61x at that shape (0.024 ms).

### Evidence (DGX Spark GB10, SM121)

- torch path: numerically verified (dominant-KV-slot unit tests, masking,
  no NaN/all-zero) and live-proven across decode, prefill, target-verify and
  draft-extend (MTP/NEXTN: accept ~2.1, GSM8K 85% @ conc 8, 0 errors).
- Triton path: bit-exact vs torch (identical -inf masks; max|diff| 0.0 direct,
  3.6e-7 through the module) across bs 1/4/32, seq 300/2048/131072, width
  131072; graph capture+replay correct INCLUDING a seq-len change in the
  static buffer between capture and replay; folded verify batches (B=4,
  next_n=4, mixed ctx 64..1900) bit-exact vs a per-token-loop reference.

### Notes for reviewers

- The kernel hardcodes the DSA cache layout (head_dim=128, block 64,
  64*128 fp8 + 64 fp32 scales per block) and asserts it.
- `clean_logits` handling matches the existing backends (cleaning happens in
  topk_transform).
- The Triton kernel needs `num_heads >= 16` (tl.dot minimum); GLM=32,
  DSv3.2=64 — the guard falls back to torch below that.
- Perf floor context for SM121 reviewers: with this backend + the sparse-MLA
  routing PR, a 504B GLM-5.2 REAP serves at 8.4 tok/s single-stream (decode is
  then bounded by unquantized bf16 attention projections, not by DSA code).

## Local mapping

| local | upstream file |
|---|---|
| p30 enum/server_args edits | `dsa/paged_mqa_logits_backend.py`, `server_args.py` |
| p30 new file | `sglang/srt/layers/attention/dsa/torch_paged_mqa_logits.py` |
| p30 dsa_backend/dsa_indexer edits | `dsa_backend.py`, `dsa/dsa_indexer.py` |
| p35 new file + dispatch | `sglang/srt/layers/attention/dsa/triton_paged_mqa_logits.py` + the torch module |

Full local chronology: `dsalogitrework.md` (PART 1 = port plan, PHASE 2, p35
LIVE RESULT, MTP LIVE RESULT).

## Submission checklist (recherchiert 2026-07-16)

**Unit-Tests: JA, zwingend und mit direktem Präzedenzfall.** Die Contribution-
Guide (docs_new/docs/developer_guide/contribution_guide.mdx) verlangt Tests für
jedes Feature, und PR #24692 (der dsv4-Fallback, den p30 portiert) shippte
`test/registered/kernels/test_sm120_paged_mqa_logits.py` (314 Z.) — läuft auf
JEDER CUDA-GPU ("no SM120 hardware required"), registriert via
`register_cuda_ci(est_time=20, stage="base-b", runner_config="1-gpu-small")`.

**Unser Test** (neu, `test/registered/kernels/test_dsa_paged_mqa_logits.py`,
modelliert auf dem Präzedenzfall; Layout-Konstanten identisch — 64er-Pages,
head_dim 128, 8448 B/Page):
1. torch-Fn vs. loopy Referenz (Numerik + -inf-Masken),
2. Triton vs. torch **bit-exakt** (inkl. identischer -inf-Masken),
3. beide KV-dtype-Views (uint8 / float8_e4m3fn) — der historische
   Garbled-Output-Bug aus dem Präzedenztest,
4. variable per-Batch seq_lens + Masking-Semantik,
5. **per-Token-Shapes (verify/next_n>=2)**: expanded seqlens + per-Token-
   page_table-Zeilen (die Doppel-Expansions-Lektion als Regressionstest),
6. cuda-graph capture + replay, inkl. seq-len-Änderung im statischen Buffer
   zwischen Capture und Replay,
7. num_heads<16-Fallback-Guard (tl.dot-Minimum).
Alles CI-lauffähig auf H100-Runnern (torch + Triton sind arch-generisch);
Entwicklung/Vorvalidierung auf spark5 möglich (podman-Methode).

**Mechanik:**
- Fork + Branch; die p30/p35-Anker in ECHTE Diffs gegen main portieren.
  ACHTUNG: main hat die Kernel-Verzeichnisse restrukturiert
  (`python/sglang/kernels/ops/attention/...` statt `srt/layers/attention/...`
  für Teile) — Zielpfade gegen main verifizieren, nicht gegen v0.5.15.
- `pre-commit run --all-files` (Pflicht laut Guide; ggf. zweimal laufen lassen).
- CI-Registrierung des Tests via `register_cuda_ci` (Stage/Runner wie Präzedenz).
- PR-Body aus diesem Dokument; Companion-PR verlinken.
- Offen zu prüfen beim Einreichen: DCO/Sign-off-Pflicht (in der Guide nicht
  gesehen, beim ersten Push gegen die PR-Checks verifizieren).
