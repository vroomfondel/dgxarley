# SGLang Upstream Bug: `Fp8MoEMethod` raises `AttributeError` for `flashinfer_cutlass` MoE backend

## Status

**Resolved by fallback since SGLang v0.5.20 (`cebca698e2`, PR #39126): `flashinfer_cutlass`/`flashinfer_cutedsl` now fall back to triton for FP8 experts; no flashinfer_cutlass FP8 kernel exists (see Re-verified 2026-10-03). The text below describes v0.5.19 and earlier.**
Originally verified on SGLang `v0.5.11` ("The Tenacity Release", tagged 2026-05-05)
with the upstream image `scitrera/dgx-spark-sglang:0.5.11` (FlashInfer 0.6.10,
sgl-kernel 0.4.2). Reproduced on Qwen3.6-35B-A3B-FP8, 4×GB10 (SM12.0a), TP=4,
during the matrix run on 2026-05-10 (case `07_fi_cutlass-moe_fi-attn` in
`matrixtest/2026-05-10/results/sglang_nn4_tp4_ep1/qwen-3.6-35b-a3b-fp8/0.5.11/`).
Confirmed still present in **v0.5.12.post1** (the current default image
`xomoxcc/dgx-spark-sglang:0.5.12.post1-sm121`) by source inspection. Release
notes contain no `Fp8MoEMethod` + flashinfer_cutlass fix; PR #21872 still open
and unmerged (re-verified 2026-06-11, last update 2026-04-01); PR #22627 merged
2026-05-26 in v0.5.12.post1 but only patches `ModelOptFp8MoEMethod` — does NOT
fix the vanilla `Fp8MoEMethod` + `flashinfer_cutlass` bug described here. The
MoE-refactor series migrated related backends to `MoeRunner`: **PR #25525**
(merged 2026-05-17) migrated `flashinfer_cutedsl` + DeepEP to MoeRunner;
**PR #26489** ("[MoE Refactor] Migrate SM90 Cutlass W4A16 to MoeRunner", merged
2026-05-30) covers `flashinfer_mxfp4` (SM90 MXFP4). Both left `flashinfer_cutlass`
+ vanilla `Fp8MoEMethod` untouched — the `# TODO(cwan): refactor other backends`
comment in `fp8.py` is still there as of 0.5.12.post1. Issue #20719 remains open
(re-verified 2026-06-11).

On v0.5.19 and earlier (superseded by the Re-verified 2026-10-03 entry), the bug is plainly visible in the source: `Fp8MoEMethod.create_moe_runner`
ends with an explicit `# TODO(cwan): refactor other backends` for everything
that is not triton/aiter/deep_gemm/fi_trtllm.

**Update 2026-06-12:** A third party independently filed the issue and a
partial-fix PR on 2026-06-11:
[Issue #27951](https://github.com/sgl-project/sglang/issues/27951)
("`[Bug] --moe-runner-backend flashinfer_cutlass + FP8 weights crashes with
AttributeError: 'Fp8MoEMethod' object has no attribute 'runner'`", filed by
`gujialiang123`) and
[PR #27968](https://github.com/sgl-project/sglang/pull/27968)
("`fix(fp8/moe): raise clear error for unsupported MoE runner backend`", filed
by `Anai-Guo`, closes #27951). **PR #27968 does NOT implement
`flashinfer_cutlass` support for `Fp8MoEMethod`** — it raises a clean
`ValueError` for unsupported backends instead of the opaque `AttributeError`.
The SM12.0a hardware constraint (no SM90 CUTLASS FP8-block kernel) is
unchanged; the workaround (`moe_runner_backend: triton`) remains the correct
cluster configuration.

**Re-verified 2026-06-14:** SGLang **v0.5.13** was published as a full GitHub
Release on 2026-06-13 (previously only a bare git tag as of 2026-06-12). The
v0.5.13 release notes contain **no `Fp8MoEMethod` / `flashinfer_cutlass` fix**.
PR #27968 remains **open and unmerged**; Issue #27951 remains **open**;
PR #21872 remains **open** (last update 2026-04-01, stagnant). Bug is
unresolved in v0.5.13. `scitrera/dgx-spark-sglang:0.5.13` does not yet exist
on DockerHub (latest scitrera image remains `0.5.12`); cluster default stays
`xomoxcc/dgx-spark-sglang:0.5.12.post1-sm121`. Workaround (`moe_runner_backend:
triton`) stays in effect.

**Re-verifiziert 2026-06-19:** PR #27968 und Issue #27951 weiterhin **offen**,
kein Merge. Kein Fix in v0.5.13 (letztes SGLang-Release 2026-06-13). Workaround
(`moe_runner_backend: triton`) unverändert aktiv. Stand identisch zu 2026-06-14.

**Re-verified 2026-06-29:** v0.5.14 released 2026-06-26 (now the latest
release). No `Fp8MoEMethod`/`flashinfer_cutlass` fix in v0.5.14; PR #27968
still OPEN; Issue #27951 still OPEN; PR #21872 still OPEN; Issue #20719 still
OPEN. Workaround (`moe_runner_backend: triton`) unchanged. Note: v0.5.14 adds
a GB10-specific FP8 fused MoE Triton config (PR #25665), which tunes the
Triton workaround path on our hardware but does not fix the underlying dispatch
bug in `Fp8MoEMethod.create_moe_runner`.

**Re-verified 2026-07-23:** SGLang **v0.5.15 released 2026-07-10**, **v0.5.15.post1
released 2026-07-14** — no `Fp8MoEMethod`/`flashinfer_cutlass` fix in either release;
source-confirmed on current `main` that `create_moe_runner` still ends in the
`# TODO(cwan): refactor other backends` branch (vanilla `Fp8MoEMethod` never sets
`self.runner` for `flashinfer_cutlass`/`flashinfer_cutedsl`). PR #27968 still OPEN, no
activity since 2026-06-11; Issue #27951 still OPEN, no activity since 2026-06-19; PR
#21872 still OPEN, stagnant since 2026-04-01. Workaround (`moe_runner_backend: triton`)
unchanged.

**Re-verified 2026-07-28:** SGLang **v0.5.16 released 2026-07-25** (tag created
2026-07-24). Source-confirmed on the v0.5.16 tag: `Fp8MoEMethod.create_moe_runner` in
`fp8.py` (around line 2098) is unchanged, it still ends with `# TODO(cwan): refactor
other backends`, so vanilla `Fp8MoEMethod` still never sets `self.runner` for
`flashinfer_cutlass`/`flashinfer_cutedsl`, and the `AttributeError` at first forward
pass remains reproducible on this release. PR #27968 and Issue #27951 are still OPEN
with no comments since 2026-06-15. PR #21872 remains OPEN and stagnant. Workaround
(`moe_runner_backend: triton`) unchanged and still the correct cluster configuration.

**Re-verified 2026-08-15:** SGLang **v0.5.17** (tag `b6a09f38f`, 2026-08-08) is
still the latest release, no new release since the 2026-07-28 check.
Source-confirmed on the v0.5.17 tag: `Fp8MoEMethod.create_moe_runner` in
`fp8.py` still ends in the `# TODO(cwan): refactor other backends` branch
(comment now at line 2293, `self.runner = MoeRunner(...)` at line 2291);
vanilla `Fp8MoEMethod` still never sets `self.runner` for
`flashinfer_cutlass`/`flashinfer_cutedsl`. Bug unchanged, only line drift from
~2098 (v0.5.16) to ~2265-2293 (v0.5.17). PR #27968 is still OPEN, idle since
2026-06-11; Issue #27951 is still OPEN, idle since 2026-06-19; PR #21872 is
still OPEN, idle since 2026-04-01. Workaround (`moe_runner_backend: triton`)
unchanged. Cluster image is now `xomoxcc/dgx-spark-sglang:0.5.17-sm121` (repo
commit `59d7912`).

**Re-verified 2026-08-21:** SGLang v0.5.17 (tag `b6a09f38f`) remains the
latest release, no new release. Source-confirmed on upstream `main` (commit
`dad6fd0f04`, 2026-08-21): `Fp8MoEMethod.create_moe_runner` in `fp8.py` still
ends in the `# TODO(cwan): refactor other backends` branch (comment now at
line 2347, `self.runner = MoeRunner(...)` at line 2345, up from ~2291-2293 on
v0.5.17). The allowlist gained a new `moe_runner_backend.is_hpc_ops()` entry,
but `flashinfer_cutlass`/`flashinfer_cutedsl` remain excluded; vanilla
`Fp8MoEMethod` still never sets `self.runner` for them. Issue #27951 was
auto-closed by the stale bot on 2026-08-19 ("automatically closed due to
inactivity"), not by a fix; this is administrative housekeeping, not a
resolution, and the underlying `AttributeError` is unaffected. PR #27968
remains OPEN, idle since 2026-06-11 (its `closes #27951` link is now moot
since the issue self-closed independently of the PR merging). PR #21872
remains OPEN, idle since 2026-04-01. Workaround (`moe_runner_backend:
triton`) unchanged.

**Re-verified 2026-08-28:** SGLang **v0.5.18** was released 2026-08-22 (now
the latest release, superseding v0.5.17). No `Fp8MoEMethod`/`flashinfer_cutlass`
fix in it. Source-confirmed on the v0.5.18 tag and on upstream `main` (commit
`d56706459`, 2026-08-28): `create_moe_runner` in `fp8.py` still ends in the
`# TODO(cwan): refactor other backends` branch, comment now at line 2347
(v0.5.18 tag) / line 2360 (current main), up from line 2345/2347 previously
logged. One commit touched the file since v0.5.17, `27c36368b6` ("fix(moe):
guard FP8 delegate activation params", PR #36275, merged 2026-08-26): it adds
an `_owns_moe_runner` flag so the MXFP4 code path's borrowed `Fp8MoEMethod`
delegate skips a flashinfer_trtllm-only activation-prep call; it does not
touch the `flashinfer_cutlass`/`flashinfer_cutedsl` allowlist gap and changes
nothing for this bug. **PR #21872 was closed unmerged on
2026-08-26T01:32:58Z**, not by a fix but by the `github-actions` stale-PR-cap
bot ("you have 8 PRs open and none updated in 7 days, over our soft cap of 5"),
the same kind of administrative closure Issue #27951 got on 2026-08-19.
`mergedAt` is null; the author could reopen it, but as of today it carries
zero merged code, same as before. PR #27968 remains OPEN, idle since
2026-06-11; Issue #27951 remains CLOSED (stale, not fixed), unchanged since
2026-08-19. Workaround (`moe_runner_backend: triton`) unchanged.

**Re-verified 2026-09-02:** SGLang v0.5.18 (2026-08-22) remains the latest
release, no new release. `fp8.py` untouched on `main` since the 2026-08-28
check (270 commits total, none touching the file), so `create_moe_runner`
still ends in the `# TODO(cwan): refactor other backends` branch for
`flashinfer_cutlass`/`flashinfer_cutedsl`. PR #27968 remains OPEN, idle since
2026-06-11; Issue #27951 remains CLOSED (stale, not fixed); PR #21872 remains
CLOSED (stale-cap closure 2026-08-26, not fixed). **Context on the already
logged PR #36275:** Issue #36264 ("`AttributeError: 'Fp8MoEMethod' object has
no attribute 'moe_runner_config'`" with `flashinfer_trtllm_routed` MoE plus
speculative decoding on hybrid NVFP4 checkpoints), opened and closed
2026-08-26, is the report that PR #36275 (merged 2026-08-26, already noted in
the 2026-08-28 entry above as adding an `_owns_moe_runner` flag) actually
fixes. It is a different allowlist-gap bug in the same `Fp8MoEMethod` class
(missing `moe_runner_config` for the `flashinfer_trtllm_routed`-delegate
path under speculative decoding), not the `flashinfer_cutlass`/`self.runner`
bug tracked by this doc. Workaround (`moe_runner_backend: triton`) unchanged.

**Re-verified 2026-09-10:** SGLang **v0.5.19** released 2026-09-05 (now the
latest release, superseding v0.5.18). No `Fp8MoEMethod`/`flashinfer_cutlass`
fix in it. Source-confirmed on the v0.5.19 tag: `create_moe_runner` in
`fp8.py` still ends in the `# TODO(cwan): refactor other backends` branch,
comment now at line 2395 (`self.runner = MoeRunner(...)` at line 2392), up
from line 2347/2360 previously logged; vanilla `Fp8MoEMethod` still never
sets `self.runner` for `flashinfer_cutlass`/`flashinfer_cutedsl`. PR #27968
remains OPEN, idle since 2026-06-11; Issue #27951 remains CLOSED (stale,
not fixed), unchanged since 2026-08-19; PR #21872 remains CLOSED
(stale-cap closure 2026-08-26), unchanged. No new issue or PR touching this
allowlist gap found. Workaround (`moe_runner_backend: triton`) unchanged.

**Re-verified 2026-09-16:** No new SGLang release since v0.5.19 (2026-09-05,
still latest). PR/issue state change since 2026-09-10: PR
[#27968](https://github.com/sgl-project/sglang/pull/27968) ("fix(fp8/moe):
raise clear error for unsupported MoE runner backend") was **auto-closed by
the stale bot yesterday, 2026-09-15** ("no updates in 95 days"), the same
administrative closure Issue #27951 already got on 2026-08-19; not merged,
zero merged code. The bot's closing comment also flags several directory
renames the branch would need retargeting against before it could even be
reopened cleanly (`python/sglang/jit_kernel/` -> `python/sglang/kernels/jit/`,
`test/srt/` -> `test/registered/`, among others), consistent with the
`jit_kernel` rename already independently confirmed in
`SGLANG_NVFP4_SHUFFLE_ROWS_OOB_UPSTREAM_BUG.md`. Both upstream vehicles for
this bug (#27951 issue, #27968 PR) are now closed with no fix landed; no new
issue or PR found. PR #21872 remains CLOSED (stale-cap closure 2026-08-26),
unchanged. Workaround (`moe_runner_backend: triton`) remains the correct
cluster configuration. Local note: cluster image bumped from
`xomoxcc/dgx-spark-sglang:0.5.17-sm121` to
`xomoxcc/dgx-spark-sglang:0.5.19-sm121` (repo commits f36d51b/958c98d/
5c07a34, 2026-09-11..15); no local patch is involved for this bug (pure
profile pin), so the bump has no effect on this doc's conclusions.

**Re-verified 2026-09-22:** SGLang **v0.5.20** released 2026-09-18 (tag
commit `94602c9c2b7cbdb8efd5c52802dac6a1c180089e`), now the latest release,
superseding v0.5.19. No `Fp8MoEMethod`/`flashinfer_cutlass` fix in it.
Source-confirmed on the v0.5.20 tag: `create_moe_runner` in `fp8.py` (now
at line 2481, up from 2395 on v0.5.19) gained a new NPU/Ascend branch
(`self.runner = MoeRunner(MoeRunnerBackend.ASCEND, moe_runner_config)` at
line 2497, part of unrelated NPU-arch35 hardware-backend work added this
release), but still ends in the unchanged `# TODO(cwan): refactor other
backends` branch (comment at line 2538, `self.runner =
MoeRunner(moe_runner_backend, moe_runner_config)` at line 2535); vanilla
`Fp8MoEMethod` still never sets `self.runner` for
`flashinfer_cutlass`/`flashinfer_cutedsl`. A search for `flashinfer_cutlass
Fp8MoEMethod` turned up no new issue or PR since 2026-09-16 (only an
unrelated hit, PR #40173, a DeepSeek-V4 hybrid-NVFP4 MXFP4-draft-expert MoE
routing fix touching `modelopt_quant.py` only, not `fp8.py`). PR #27968
remains CLOSED (stale bot, 2026-09-15); Issue #27951 remains CLOSED (stale
bot, 2026-08-19); neither was reopened. PR #21872 remains CLOSED
(stale-cap closure, 2026-08-26), unchanged. Workaround (`moe_runner_backend:
triton`) remains the correct cluster configuration. Local note: cluster
image stays at `xomoxcc/dgx-spark-sglang:0.5.19-sm121`; a v0.5.20-sm121
image is being prepared (repo commits 63e72da/f3f29e1, 2026-09-22) but not
yet built or deployed; no local patch is involved for this bug (pure
profile pin), so neither the bump nor its eventual deployment changes this
doc's conclusions.

**Re-verified 2026-09-25:** No new SGLang release since v0.5.20 (2026-09-18,
still latest). PR #27968 remains CLOSED (stale bot, 2026-09-15, unchanged);
Issue #27951 remains CLOSED (stale bot, 2026-08-19, unchanged); PR #21872
remains CLOSED (stale-cap closure, 2026-08-26, unchanged); none reopened, no
new activity on any of the three. Local note: repo commit `3214ed2` (dated
2026-09-22 per `git log`) flipped `default_sglang_image` to
`xomoxcc/dgx-spark-sglang:0.5.20-sm121`; the live cluster is now confirmed
running that image on all sglang pods (checked today via `kubectl`). This
doc's 2026-09-22 entry already source-confirmed the `create_moe_runner` gap
directly against the v0.5.20 tag (new NPU/Ascend branch added, vanilla
`Fp8MoEMethod` behavior for `flashinfer_cutlass`/`flashinfer_cutedsl`
unchanged), so nothing new to re-verify now that v0.5.20 is live. No local
patch is involved for this bug (pure profile pin); the workaround
(`moe_runner_backend: triton`) remains the correct cluster configuration.

**Re-verified 2026-09-28:** No new SGLang release since v0.5.20 (2026-09-18,
still latest). `fp8.py` untouched on upstream `main` since the 2026-09-25
check (HEAD now `df378bca42`, 2026-09-28; the one intervening commit,
`38ec649048`, is a kernel-directory reorg that does not touch `fp8.py`).
PR #27968 remains CLOSED (stale bot, 2026-09-15, zero activity since); Issue
#27951 remains CLOSED (2026-08-19, zero activity since); PR #21872 remains
CLOSED (2026-08-26, zero activity since). No new issue or PR found searching
for `Fp8MoEMethod flashinfer_cutlass` opened after 2026-09-16. Workaround
(`moe_runner_backend: triton`) unchanged.

**Re-verified 2026-10-03: RESOLVED in v0.5.20 (missed by the 09-22/09-25/09-28 entries, which wrongly said "no fix").** SGLang **v0.5.21** was published 2026-10-02 (tag commit `e00930c548`, 2026-09-29), now the latest release. Source on the v0.5.20 and v0.5.21 tags: `Fp8MoEMethod.create_moe_runner` in `fp8.py` (v0.5.20 line ~2509-2521, v0.5.21 line ~2777-2790) has a new branch, added by `cebca698e2` (PR #39126, "[Qwen3.8] Enable NVIDIA NVFP4 on DGX Spark ...", merged 2026-09-13, first shipped in v0.5.20; absent in v0.5.17-v0.5.19). If the resolved backend is `flashinfer_cutlass` or `flashinfer_cutedsl`, it logs "Fp8MoEMethod has no %s path; using triton for its fp8 experts." and substitutes `MoeRunnerBackend.TRITON` before the allowlist, so `self.runner` is always set and the `AttributeError` no longer occurs. This is a fallback, not flashinfer_cutlass FP8 support: FP8 experts run on triton, identical to the `moe_runner_backend: triton` workaround. The `# TODO(cwan): refactor other backends` comment is still present (line 2538 on v0.5.20, 2800 on v0.5.21) but is unreachable for these two backends. The deployed `0.5.20-sm121` image already includes the fix. Issue #27951 (CLOSED 2026-08-19), PR #27968 (CLOSED 2026-09-15, unmerged) and PR #21872 (CLOSED 2026-08-26, unmerged) are unchanged and not reopened. No new issue or PR found for `Fp8MoEMethod flashinfer_cutlass` after 2026-09-28. upstream/main (HEAD `7be5e3473c`) keeps the fallback. Whether profiles may drop the `moe_runner_backend: triton` pin is a deployment decision, not re-tested on the cluster.

**Re-verified 2026-10-11:** Still resolved by fallback. No new SGLang release since v0.5.21 (2026-10-02, tag commit `e00930c548`). The `Fp8MoEMethod has no %s path; using triton for its fp8 experts.` fallback from `cebca698e2` (PR #39126) is present on the v0.5.21 tag (`fp8.py` line 2783) and on upstream/main (HEAD `ad8b1e1f85`, 2026-10-10; line 2857). The four `fp8.py` commits on main since 2026-10-03 (`10edb05e72`, `64c7f65eac`, `2d26823f94`, `2ff3e6dce1`) are a parallel-context refactor, an AMD aiter scale change and a DSV4.1 DeepGEMM change; none touch the MoE runner fallback. Issue #27951, PR #27968 and PR #21872 remain CLOSED and are not reopened; no new issue or PR found for `Fp8MoEMethod flashinfer_cutlass` after 2026-10-03. Correction to the 10-03 local note: the deployed image has been `xomoxcc/dgx-spark-sglang:0.5.21-sm121` since repo commit `5a42af7` (2026-10-03), not 0.5.20. The `moe_runner_backend: triton` pin in the FP8 profiles (qwen3.5-122b-a10b-fp8, qwen3.6-35b-a3b-fp8, glm-4.7-fp8) is unchanged; dropping it remains an untested deployment decision.

**Re-verified 2026-11-10:** Still resolved by fallback. SGLang v0.5.22 (tag commit `baa9befa38`, GitHub Release 2026-10-19) and v0.5.23 (tag commit `297e54fd10`, GitHub Release 2026-11-02, now the latest) both keep the `cebca698e2` fallback ("Fp8MoEMethod has no %s path; using triton for its fp8 experts.", `fp8.py` line 2857 on v0.5.22, 2981 on v0.5.23, 3173 on upstream/main HEAD `170804b40a`). Update to the 10-03 entry: the `# TODO(cwan): refactor other backends` comment no longer exists from v0.5.23 on; `4f8f06faf3` (PR #44993, 2026-10-20, "Derive layer contracts from the selected quant method") removed it and added an `else` that records `_moe_runner_backend` for direct kernels. The fallback and the `flashinfer_cutlass`/`flashinfer_cutedsl` -> triton substitution are untouched by it. Issue #27951, PR #27968 and PR #21872 remain CLOSED; no new issue or PR found for `Fp8MoEMethod flashinfer_cutlass` after 2026-10-03. The deployed image is still `0.5.21-sm121` (fallback present since v0.5.20); the `moe_runner_backend: triton` pin in the FP8 profiles is unchanged.

Adjacent open work:

- [PR #21872](https://github.com/sgl-project/sglang/pull/21872)
  ("Add FlashInfer CUTLASS fused MoE support for FP8 block-quantized models on SM90") —
  open since 2026-04-01, stagnant since 2026-04-01. Patches a **different
  class** (`CompressedTensorsW8A8Fp8MoEMethod` in
  `compressed_tensors/schemes/compressed_tensors_w8a8_fp8_moe.py`), not the
  vanilla `Fp8MoEMethod` we hit. Explicitly **SM90 only**.
- [PR #22627](https://github.com/sgl-project/sglang/pull/22627)
  ("Fix flashinfer_cutlass MoE crash when intermediate_size_per_partition is
  not 16-aligned") — **merged 2026-05-26 in v0.5.12.post1**. Patches
  `ModelOptFp8MoEMethod` in `modelopt_quant.py` and addresses a different
  symptom (alignment, not missing `runner`). Does **not** touch vanilla
  `Fp8MoEMethod`; the bug documented here is unaffected.
- [Issue #20719](https://github.com/sgl-project/sglang/issues/20719)
  ("CompressedTensorsW4A4Nvfp4MoE bypasses MoeRunner, hardcodes kernel
  dispatch in apply_weights") — open since 2026-03-16. **Correction (re-verified
  2026-07-28): this was NOT routine stale-bot housekeeping.** The issue was
  auto-closed by the stale bot twice (2026-05-16, then again 2026-07-16 after a
  reopen), but was subsequently reopened by `alexnails` and **genuinely closed as
  completed on 2026-07-25** via merged PR
  [#32248](https://github.com/sgl-project/sglang/pull/32248) ("Migrate
  CompressedTensorsW4A4Nvfp4MoE TRT-LLM path onto MoeRunner", merged
  2026-07-25T03:57:39Z). That PR migrates `CompressedTensorsW4A4Nvfp4MoE` onto the
  shared `MoeRunner(FLASHINFER_TRTLLM)` path, so the anti-pattern IS now fixed for
  that class. This does **not** touch `Fp8MoEMethod` (a different class in a
  different file), so the core claim of this doc is unaffected: names `Fp8MoEMethod`
  as the *correct* reference path, not as the bug, and remains true. No fix for the
  bug documented here.

None of these touch `quantization/fp8.py::Fp8MoEMethod`, which is the path
Qwen3.6-35B-A3B-FP8 (and every plain `--quantization fp8` MoE model) takes.

## Affected Configuration

- Quantization method: vanilla `Fp8MoEMethod`
  (`sglang/srt/layers/quantization/fp8.py`) — i.e. any FP8 MoE model whose HF
  config carries `quantization_config.quant_method == "fp8"` and is loaded
  through the standard `Fp8Config.get_quant_method` dispatch.
  Not affected: `CompressedTensorsW8A8Fp8MoEMethod` (PR #21872 territory),
  `ModelOptFp8MoEMethod` (PR #22627 territory).
- MoE runner backend: `--moe-runner-backend flashinfer_cutlass`
  (or the new `flashinfer_cutedsl` from PR #21339, same code-path gap).
- Tested models that hit this on our cluster:
  - `Qwen/Qwen3.6-35B-A3B-FP8` (256 experts, hybrid GDN, native 262K)
  - sibling FP8 MoE profiles in `roles/k8s_dgx/model_profiles/`:
    `qwen-qwen3-coder-30b-a3b-instruct-fp8`,
    `qwen-qwen3-next-80b-a3b-instruct-fp8`,
    `qwen-qwen3.5-35b-a3b-fp8`, `qwen-qwen3.5-122b-a10b-fp8`,
    `unsloth-qwen3-235b-a22b-instruct-2507-fp8`
    (all use vanilla `Fp8MoEMethod` and would crash identically).
- Hardware: confirmed on SM12.0a (NVIDIA GB10 / DGX Spark). Bug is
  hardware-independent at the Python level — would crash the same way on
  H100/H200/B200 if the user passed `--moe-runner-backend flashinfer_cutlass`
  with a vanilla FP8 MoE model. (See "Why this is also moot on SM120" for
  the deeper hardware constraint that makes any port to `Fp8MoEMethod` of
  questionable value on Blackwell-Edge.)

## The Bug

`Fp8MoEMethod.create_moe_runner`
(`sglang/srt/layers/quantization/fp8.py`, lines 1514–1545 on v0.5.11)
sets `self.runner` only for a closed allowlist of backends:

```python
def create_moe_runner(
    self, layer: torch.nn.Module, moe_runner_config: MoeRunnerConfig
):
    self.moe_runner_config = moe_runner_config
    moe_runner_backend = get_moe_runner_backend()

    if moe_runner_backend.is_auto():
        if self.is_deepgemm_moe_runner_backend_enabled():
            moe_runner_backend = MoeRunnerBackend.DEEP_GEMM
        elif (
            _is_hip
            and (_use_aiter or _use_hip_int4)
            and get_moe_a2a_backend().is_none()
        ):
            moe_runner_backend = MoeRunnerBackend.AITER
        else:
            moe_runner_backend = MoeRunnerBackend.TRITON

    if (
        moe_runner_backend.is_deep_gemm()
        or moe_runner_backend.is_triton()
        or moe_runner_backend.is_aiter()
        or moe_runner_backend.is_flashinfer_trtllm()
        or moe_runner_backend.is_flashinfer_trtllm_routed()
    ):
        self.runner = MoeRunner(moe_runner_backend, moe_runner_config)
    else:
        # TODO(cwan): refactor other backends
        pass
```

For `moe_runner_backend == flashinfer_cutlass` (and `flashinfer_cutedsl`)
the `else` branch is taken and `self.runner` is **never assigned**.

`Fp8MoEMethod.apply` (same file, lines 1566–1756) then unconditionally
dereferences `self.runner` from line 1605 onwards:

```python
and self.runner.runner_backend.is_aiter()      # 1605
...
return self.runner.run(dispatch_output, quant_info)  # 1612
...
if self.runner.runner_backend.is_deep_gemm():        # 1652
...
self.runner.runner_backend.is_flashinfer_trtllm()    # 1690
or self.runner.runner_backend.is_flashinfer_trtllm_routed()  # 1691
...
elif self.runner.runner_backend.is_triton():         # 1749
"Unsupported runner backend: %s" % self.runner.runner_backend  # 1753
return self.runner.run(dispatch_output, quant_info)  # 1756
```

The first dereference (line 1605, hit unconditionally on the standard
non-AITER path) raises `AttributeError`.

## Symptom

Server crashes deterministically during `Scheduler.__init__` →
`init_tp_model_worker` → `ModelRunner.__init__` →
`init_device_graphs` → CUDA-graph capture → first forward pass:

```
File "/usr/local/lib/python3.12/dist-packages/sglang/srt/managers/scheduler.py", line 432, in __init__
  self.init_model_worker()
...
File "/usr/local/lib/python3.12/dist-packages/sglang/srt/model_executor/cuda_graph_runner.py", line 884, in _capture_one_stream
  ) = self.capture_one_batch_size(bs, forward, stream_idx)
...
File "/usr/local/lib/python3.12/dist-packages/sglang/srt/models/qwen3_5.py", line 640, in forward
  hidden_states = self.mlp(...)
...
  return self.quant_method.apply(...)
File "/usr/local/lib/python3.12/dist-packages/sglang/srt/layers/quantization/fp8.py", line 1652, in apply
  if self.runner.runner_backend.is_deep_gemm():
AttributeError: 'Fp8MoEMethod' object has no attribute 'runner'
```

(Trace excerpt is from line 1652 because CUDA-graph capture took the
deep_gemm dispatch branch first; on `--disable-cuda-graph` it would
crash on line 1605 instead. Both lines fail for the same reason.)

The crash is reported by the bench matrix as
`outcome=startup_crash` with the head pod restart-looping until the
matrix driver scales the deployment to 0 and moves on to the next case.

## Reproducer

Standard 4-node setup, vanilla upstream image, the FP8 model and the
flag in question — no other special config required:

```bash
docker run --rm --gpus all scitrera/dgx-spark-sglang:0.5.11 \
  python3 -m sglang.launch_server \
    --model-path Qwen/Qwen3.6-35B-A3B-FP8 \
    --tp-size 4 --pp-size 1 --nnodes 4 --node-rank 0 \
    --nccl-init-addr <head>:50000 \
    --kv-cache-dtype fp8_e4m3 \
    --mem-fraction-static 0.50 \
    --attention-backend flashinfer \
    --fp8-gemm-backend cutlass \
    --moe-runner-backend flashinfer_cutlass \
    --mamba-scheduler-strategy extra_buffer
```

`--moe-runner-backend flashinfer_cutlass` is the only line that matters —
remove it (or set it to `triton`) and the same model launches and serves
fine. This was reproduced on 0.5.10 and 0.5.10.post1 in earlier matrix
sweeps; verified again on 0.5.11 in the 2026-05-10 run.

## Why this is also moot on SM12.0a (GB10 / DGX Spark)

Even after a hypothetical port of PR #21872 from
`CompressedTensorsW8A8Fp8MoEMethod` to vanilla `Fp8MoEMethod`, the
underlying kernel call would be:

```python
flashinfer.fused_moe.cutlass_fused_moe(use_deepseek_fp8_block_scale=True, ...)
```

That code-path in FlashInfer 0.6.x is a **SM90 CUTLASS kernel**
(see PR #21872 description: "Requirements: SM90 GPU (H100/H800)").
The CUTLASS templates for FP8-block fused MoE have not been ported to
SM12.0a — there is no kernel for our hardware to dispatch to even if
the Python-level dispatch were fixed. So for the GB10 cluster the only
correct outcome of this bug is "remain on `triton`".

PR #21872's own benchmark numbers also argue against pushing this on
H100, in the CUDA-graph case that we (and most production setups) run:

| Mode          | Triton | FlashInfer CUTLASS |           Δ |
| ------------- | -----: | -----------------: | ----------: |
| No CUDA graph |    152 |                163 |  **+7.4 %** |
| CUDA graph    |    869 |                749 | **−13.8 %** |

i.e. fi_cutlass FP8 only beats triton in eager mode, where overall
throughput is so far below the CUDA-graph path that the relative gain
doesn't matter (compare our case-02 eager run at 22 tok/s vs case-01
CUDA-graph run at 76 tok/s on the same model).

## Workaround

Pin `moe_runner_backend: triton` in the per-model SGLang profile.
For our cluster this is already the default in
`roles/k8s_dgx/model_profiles/qwen-qwen3.6-35b-a3b-fp8.yml` and the
sibling FP8 MoE profiles. The matrix sweep deliberately probes
`flashinfer_cutlass` to (re-)confirm the upstream gap.

For users who want to test other FP8 MoE models on this code-path,
the safe fallback is:

```yaml
moe_runner_backend: triton          # vanilla Fp8MoEMethod path that works
# moe_runner_backend: flashinfer_cutlass   # AttributeError, see this doc
# moe_runner_backend: flashinfer_cutedsl   # AttributeError (same code-path gap)
```

## Action items

1. ~~**File upstream issue** in `sgl-project/sglang` with the stack trace,
   reproducer command, and a pointer to the
   `# TODO(cwan): refactor other backends` line in `fp8.py` and to
   PRs #21872 / #22627 / Issue #20719 as the existing partial work.
   Title proposal: "`Fp8MoEMethod.create_moe_runner` does not handle
   `flashinfer_cutlass` / `flashinfer_cutedsl` → `AttributeError:
   'Fp8MoEMethod' object has no attribute 'runner'` at first forward".~~
   **Moot as of 2026-06-12** — Issue #27951 and PR #27968 filed independently
   by third parties on 2026-06-11. See Status section update above.
2. **Keep cluster profile pinned to `triton`** for all vanilla-FP8
   MoE models. Document in `CLAUDE.md` once filed.
3. **Stop re-running the fi_cutlass cases (07–12) of the v0.5.11
   correctness/throughput matrix for FP8 MoE models** until either
   (a) upstream lands a port of #21872 to `Fp8MoEMethod` *and*
   (b) FlashInfer ships an SM12.0a CUTLASS FP8-block fused-MoE
   kernel. Both are independent prerequisites; neither is in flight.
4. **Re-evaluate `flashinfer_cutedsl` (cases 15–20)** independently —
   that backend has the same `create_moe_runner` gap on FP8, but the
   underlying kernel (PR #21339) targets a different SM range and may
   matter for SM12.0a once the dispatch is fixed.
