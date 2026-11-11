# SGLang Upstream Bug: sharded_state + Speculative Decoding (NEXTN/EAGLE)

## Status

**Unreported** as of 2026-06-11 (re-verified: GitHub search for "sharded_state
speculative" / "speculative_draft_load_format sharded" in `sgl-project/sglang`
still returns no issues or PRs). Bug exists in SGLang v0.5.9, v0.5.10rc0, v0.5.10,
v0.5.10.post1, v0.5.11 (2026-05-05), v0.5.12 (2026-05-16), **v0.5.12.post1**
(released 2026-05-26 — a DeepSeek-V4/disaggregation point release that contains no
sharded_state + speculative fix), and **v0.5.13 (official GitHub Release published
2026-06-13)**. Commit search over the v0.5.12.post1..v0.5.13 delta found no fix for
the sharded_state + speculative draft-load bug; `scheduler.py:maybe_init_draft_worker`
is unchanged; still unreported upstream. The relevant code in v0.5.11
(`scheduler.py:669–675`) still only overrides `load_format` when
`speculative_draft_load_format` is explicitly set, leaving the draft model with
inherited `sharded_state` whenever the user doesn't pass the override flag — same
failure mode as in 0.5.10. The workaround in `sglang_launch.sh` (force
`--speculative-draft-load-format auto` when main `load_format=sharded_state`) is
therefore still required on v0.5.11 / v0.5.12 / v0.5.12.post1 / v0.5.13 / dev1 images.

> **Re-verified 2026-06-14:** v0.5.13 is an official GitHub Release since
> 2026-06-13. The sharded_state + speculative draft-load bug remains UNFIXED
> (source-confirmed: `scheduler.py:maybe_init_draft_worker` unchanged). Bug
> still unreported upstream. **New archaeological note:** Spec V1 was
> deprecated/removed in v0.5.13 via PR #25464 (merged 2026-06-08) —
> EAGLE/EAGLE3/MTP/NEXTN now run on the unified V2 worker. The traceback
> path `speculative/eagle_worker.py` no longer exists in v0.5.13; V2 uses
> a different worker class, but the root cause (draft model inheriting
> `sharded_state` from main model's ServerArgs when
> `speculative_draft_load_format` is unset) is upstream of algorithm
> selection and applies equally to the V2 worker. The workaround
> `--speculative-draft-load-format auto` is still valid and still required.

> **Note (2026-05-31):** SGLang v0.5.12 brings several *Spec-V2* reliability
> fixes that are accessible via this very workaround once it gets the draft
> model loaded — PR #23456 (overlap stale-state), #25204 (frozen-KV MTP crash
> when `bonus_tokens=None`), #24965 (ngram off-by-1), #25033 (Kimi-K2.5 MLA
> EAGLE + DP attention), all merged and in v0.5.12 / .post1. Re-benchmarking the
> MTP profiles on 0.5.12 is warranted (see `TODO_0.5.12.md` §5). This does not
> change the workaround requirement — it just makes the path it unblocks
> meaningfully better than on 0.5.9.

> **Note (2026-06-11):** SGLang v0.5.13 (official GitHub Release 2026-06-13) brings further Spec-V2 changes:
> Spec V1 deprecation (PR #25464), EAGLE draft kv_indices OOB fixes
> (#27338, #27460), FA3 EAGLE crash fix (#25077 "Fix(spec): Fix the crash
> issue in the FA3 backend when running with top-k > 1 and page_size > 1",
> merged 2026-06-09), and trtllm_mha draft-extend CUDA graph with v2
> semantics (#25002 "[spec_v2] Enable trtllm_mha draft-extend CUDA graph
> with v2 semantics", merged 2026-06-05). The sharded_state workaround is
> unaffected by these changes — the bug is upstream of the speculative
> algorithm selection and is still unreported.

> **Re-verified 2026-06-29:** SGLang **v0.5.14** released 2026-06-26. The v0.5.14
> speculative decoding changes (PRs #28854, #28782, #26312, #28516, #27469, #24955,
> #17260) contain no fix for `maybe_init_draft_worker` inheriting `load_format=sharded_state`
> when `speculative_draft_load_format` is unset. Bug still unreported upstream (GitHub
> search for "sharded_state speculative" / "speculative_draft_load_format sharded"
> in `sgl-project/sglang` returns no issues or PRs). The `--speculative-draft-load-format auto`
> workaround in `sglang_launch.sh` remains required and unchanged.

> **Re-verified 2026-07-23:** Bug confirmed **still present verbatim** in
> **v0.5.15** (2026-07-10) and **v0.5.15.post1** (2026-07-14) — source-checked
> both tags: `maybe_init_draft_worker` still only sets
> `self.server_args.load_format = self.server_args.speculative_draft_load_format`
> when that field `is not None`; unset (the default), the draft still
> silently inherits the main model's `load_format`. Still unreported upstream
> (no matching issue/PR). **Heads-up:** on `main` only (not yet in any
> release), a `RuntimeContext` config-namespace refactor (PRs #31813–#31818,
> merged/closed 2026-07-22) replaces the direct attribute mutation with
> `self.server_args.override("scheduler.draft_load_format", load_format=...)`.
> Same conditional, same root cause, just a different mechanism — the
> `scheduler.py:669–675` line references below will drift once this refactor
> ships in a release; re-check line numbers at that point. Also filed today
> and **related but distinct**: issue
> [#32202](https://github.com/sgl-project/sglang/issues/32202) ("Speculative
> draft auto load format does not resolve object-storage paths", opened
> 2026-07-23) — a different bug in the same draft-load-format area (S3 path
> resolution for `--speculative-draft-load-format auto`), not the
> sharded_state issue this doc tracks. The `--speculative-draft-load-format auto`
> + explicit `--speculative-draft-model-path` workaround remains required and unchanged.

> **Re-verified 2026-07-28:** SGLang **v0.5.16** (released 2026-07-25) is now
> the latest release, and it ships the `RuntimeContext` refactor that was
> flagged above as main-only on 2026-07-23. Source-diffed `scheduler.py` on
> the v0.5.16 tag against `main`, both identical: `maybe_init_draft_worker()`
> now calls `self.server_args.override("scheduler.draft_load_format",
> load_format=self.server_args.speculative_draft_load_format)` instead of the
> old direct assignment. The method moved from `scheduler.py:669-675` (the
> v0.5.11 line numbers quoted at the top of this doc) to roughly
> `scheduler.py:773-795` in v0.5.16 and on `main`. Root cause and our
> workaround are unchanged: the `override()` call is still reached only
> `if self.server_args.speculative_draft_load_format is not None`, so an
> unset value (the default) still leaves the draft silently inheriting the
> main model's `load_format`. **Upstream churn note:** PR
> [#32100](https://github.com/sgl-project/sglang/pull/32100) reverted part of
> this same refactor (#31813 through #31817), about 10.5 hours after they
> merged (2026-07-22, 08:17 to 18:52 UTC), citing three defects, one of them
> named "Draft load format ignored" (a bag-only write that never reached
> `LoadConfig.load_format`). The code that actually shipped in v0.5.16 is the
> correct, setattr-based variant (`ServerArgs.override()` does a real
> `setattr` on the field, confirmed by reading its implementation in
> `server_args.py`), so this near-miss regression did not make it into the
> release. Issue #32202 is still open with no new comments. The
> `--speculative-draft-load-format auto` and `--speculative-draft-model-path`
> workaround remains required and unchanged.

> **Re-verified 2026-08-03:** the tracked code moved again, this time on
> `main` only (no release yet; SGLang still at v0.5.16). A 5-part PR series
> merged 2026-08-03 04:22–04:24 UTC —
> [#33334](https://github.com/sgl-project/sglang/pull/33334) through
> [#33338](https://github.com/sgl-project/sglang/pull/33338) ("config: stop
> writing config onto the published ServerArgs…", "spec: build every draft
> worker from a draft ServerArgs copy", …) — relocates the mechanism:
> `Scheduler.maybe_init_draft_worker()` (now `scheduler.py:865`) calls a new
> helper `draft_server_args_copy()` in a new file
> `python/sglang/srt/speculative/draft_worker_common.py`. Source-verified on
> `main`: `_draft_load_format_fields()` (lines 64-68) still returns `{}` when
> `get_spec().speculative_draft_load_format is None` — the root-cause
> conditional gate is unchanged, just relocated. The draft still silently
> inherits `load_format=sharded_state` when the flag is unset; our
> `--speculative-draft-load-format auto` + `--speculative-draft-model-path`
> workaround remains required and unchanged. **One incidental fix:** PR
> #33335 resolves the latent in-place-mutation caveat flagged in the "Note"
> under "Our Workaround" below — `draft_server_args_copy()` now does
> `deepcopy(server_args)` first and overrides only the copy (docstring: "The
> target's own instance is untouched"), so the shared `server_args` object is
> no longer mutated on `main`.

> **Re-verified 2026-08-07:** relocated AGAIN — third move in a month, still
> `main`-only (SGLang still at v0.5.16, so "present in every release up to and
> including v0.5.16" stays true). A 6-unit "config close-out" stack merged
> 2026-08-06 02:28–02:32 UTC
> ([#33487](https://github.com/sgl-project/sglang/pull/33487)–[#33492](https://github.com/sgl-project/sglang/pull/33492);
> carrier [#33486](https://github.com/sgl-project/sglang/pull/33486) was a
> review/CI-only vehicle, closed unmerged by design). Units #33491/#33492
> **delete** `_draft_load_format_fields()` and `draft_server_args_copy()`
> from `draft_worker_common.py` entirely (the file keeps only
> `DraftWorkerBundle` & friends, no load-format logic);
> `maybe_init_draft_worker()` (now `scheduler.py:902`) passes
> `server_args=self.server_args` straight through, no copy/override. The
> authoritative location is now
> `python/sglang/srt/model_executor/model_runner.py`:
> `ModelRunner._resolve_draft_load_format()` (lines 1256-1267; returns `None`
> unless `get_spec().speculative_draft_load_format` is set) +
> `_load_format_scope()` (lines 1244-1254; `None` →
> `contextlib.nullcontext()`, i.e. the draft falls through to the main
> model's `load_format` incl. `sharded_state`), consumed in `__init__` (line
> 324) and the load paths (lines 674, 1066-1098). The root-cause gate is
> byte-for-byte the same conditional — and #33491 adds a unit test
> `test_an_unset_draft_load_format_leaves_the_load_config_alone` that pins
> the inherit-when-unset behaviour as INTENTIONAL upstream. The only
> auto-default for `speculative_draft_load_format` remains the
> `runai_streamer` object-storage case (`server_args.py:7271-7277`), nothing
> for `sharded_state`. Our `--speculative-draft-load-format auto` +
> `--speculative-draft-model-path` workaround remains required and unchanged.
> Issue #32202 still open, no new activity.

> **Re-verified 2026-08-09:** SGLang **v0.5.17** released 2026-08-08 (tag
> commit `b6a09f38f`, cut 2026-08-07T21:50 UTC; 582 PRs since v0.5.16). The
> bug ships in it unchanged — "present in every release up to and including
> v0.5.16" now reads **v0.5.17**. **Release-vs-main split:** the v0.5.17
> release branch did NOT pick up the 2026-08-06 relocation stack
> (#33487–#33492) — source-checked the v0.5.17 tag:
> `draft_worker_common.py` still has `_draft_load_format_fields()` (line 67)
> and `draft_server_args_copy()` (line 95), and `scheduler.py:901` still
> calls `draft_server_args_copy` — i.e. the released artifact has the
> 2026-08-03 code shape (PR stack #33334–#33338), while `main`'s
> authoritative location remains `model_runner.py` as described in the
> 2026-08-07 entry above. Line drift on `main` since then (unrelated DCP /
> shared-experts-fusion commits `07297049e`/`ce1b9f88b`/`b61a06921`):
> `_load_format_scope()` now :1242, `_resolve_draft_load_format()` now
> :1254, `__init__` consumption line 322, load paths 672 / 1064-1096 —
> logic verbatim. v0.5.17's Speculative-Decoding release notes contain no
> fix for this bug; no new PRs touch `speculative_draft_load_format`.
> Issue #32202 still open, 0 comments, untouched since 2026-07-23. Our
> `--speculative-draft-load-format auto` + `--speculative-draft-model-path`
> workaround remains required and unchanged.

> **Re-verified 2026-08-15:** SGLang still at **v0.5.17** (released 2026-08-08),
> no new release since the 2026-08-09 check. Bug unchanged, workaround
> unchanged. Issue #32202 still open, 0 comments, idle since 2026-07-23. Line
> numbers on `main` drifted again (+~55 lines each, from an unrelated
> in-flight config-bag refactor series, latest commits `1ab713c33` /
> `97279980c`, merged 2026-08-15, `main` HEAD `0c072235f` at
> 2026-08-15T09:20:30Z): `ModelRunner.__init__` consumption line 322 -> 331;
> remote-instance-transfer check 672 -> 676; load-path block 1064-1098 ->
> 1074-1106; `_load_format_scope()` :1242 -> :1297; `_resolve_draft_load_format()`
> :1254 -> :1309. Logic byte-for-byte identical. `draft_worker_common.py` had
> one unrelated touching commit (`fde9ad253`, 2026-08-11, Muse Glimmer model
> support) that does not affect the load-format logic. No new issues or PRs
> matching `draft_load_format` since 2026-08-08.

> **Re-verified 2026-08-21:** SGLang still at v0.5.17 (released 2026-08-08), no
> new release since the 2026-08-15 check. Bug unchanged, workaround unchanged.
> Issue #32202 still open, 0 comments, unchanged since 2026-07-23. No new
> issues or PRs matching `draft_load_format` / "sharded_state speculative"
> since 2026-08-08. Line numbers on `main` drifted again (`upstream/main` HEAD
> now `dad6fd0f04`, 2026-08-21T18:12:23+08:00; last touching commit to
> `model_runner.py` is `cba3c5d5ac`, 2026-08-17, "config: the per-instance
> families read the bags (#35026)"): `ModelRunner.__init__` consumption line
> 331 -> 330; remote-instance-transfer check 676 -> 670; load-path block
> 1074-1106 -> 1066-1098; `_load_format_scope()` :1297 -> :1289;
> `_resolve_draft_load_format()` :1309 -> :1301. Logic byte-for-byte
> identical: `_resolve_draft_load_format()` still returns `None` unless
> `speculative_draft_load_format` is explicitly set, so an unset value still
> falls through to the main model's `load_format`. `draft_worker_common.py`
> on `main` still holds only `DraftWorkerBundle` and worker-build helpers, no
> load-format logic (2026-08-07 relocation to `model_runner.py` stands).
> `scheduler.py:maybe_init_draft_worker()` now at line 926 (was 902 on
> 2026-08-07), behavior unchanged. The `--speculative-draft-load-format auto`
> + `--speculative-draft-model-path` workaround remains required and
> unchanged.

> **Re-verified 2026-08-28:** SGLang **v0.5.18** released 2026-08-22 (tag
> commit `71de97b264b04dcd514cf904003028aefe9775c8`, cut 2026-08-20T21:29
> UTC; ~710 PRs since v0.5.17). The bug ships in it unchanged; "present in
> every release up to and including v0.5.17" now reads **v0.5.18**.
> Source-checked the v0.5.18 tag: `ModelRunner.__init__` consumption at
> line 330, `_load_format_scope()` at line 1289, `_resolve_draft_load_format()`
> at line 1301 (logic unchanged: still returns `None` unless
> `speculative_draft_load_format` is explicitly set), `scheduler.py:
> maybe_init_draft_worker()` at line 923. `draft_worker_common.py` on the
> v0.5.18 tag still holds only `DraftWorkerBundle` and worker-build helpers,
> no load-format logic. v0.5.18's Speculative Decoding release-note section
> (17 PRs) contains no fix touching `speculative_draft_load_format`. Issue
> #32202 unchanged: still open, 0 comments, idle since 2026-07-23. Line
> numbers on `main` drifted again (`upstream/main` HEAD now
> `d56706459c8e52ec3ab1c41dae778e4fe03e0da3`, 2026-08-28):
> `ModelRunner.__init__` consumption line 330 -> 353; `_load_format_scope()`
> :1289 -> :1347; `_resolve_draft_load_format()` :1301 -> :1359;
> `scheduler.py:maybe_init_draft_worker()` line 926 -> 953. Logic
> byte-for-byte identical on both the v0.5.18 tag and `main`. **Adjacent
> but distinct, not our bug:** SGLang issue
> [#34622](https://github.com/sgl-project/sglang/issues/34622) ("Prevent
> Qwen3.5 MTP draft from inheriting GPTQ quantization", opened 2026-08-12,
> open) reports the same "draft silently inherits main-model config"
> failure class, but for `speculative_draft_model_quantization` (GPTQ), not
> `load_format`/`sharded_state`; even an explicit
> `--speculative-draft-model-quantization unquant` does not prevent it.
> Does not affect our tracked bug or workaround. The
> `--speculative-draft-load-format auto` + `--speculative-draft-model-path`
> workaround remains required and unchanged.

> **Re-verified 2026-09-02:** SGLang still at v0.5.18 (released 2026-08-22), no
> new release since the 2026-08-28 check. Bug unchanged, workaround unchanged.
> Issue #32202 still open, 0 comments, unchanged since 2026-07-23. PR #34622
> (the adjacent GPTQ-draft-quantization bug, not ours) still open, no new
> activity since 2026-08-18. Line numbers on `main` drifted again
> (`upstream/main` HEAD now `f5866545186f27a8a01c1a083e8c32809b5506f3`,
> 2026-09-02; 270 commits since the 2026-08-28 check, none touching
> `scheduler.py`, `model_runner.py`, or `draft_worker_common.py`):
> `ModelRunner.__init__` consumption line 353 -> 362; `_load_format_scope()`
> :1347 -> :1370; `_resolve_draft_load_format()` :1359 -> :1382;
> `scheduler.py:maybe_init_draft_worker()` line 953 -> 973. Logic
> byte-for-byte identical: `_resolve_draft_load_format()` still returns
> `None` unless `speculative_draft_load_format` is explicitly set. The
> `--speculative-draft-load-format auto` + `--speculative-draft-model-path`
> workaround remains required and unchanged.

> **Re-verified 2026-09-10:** SGLang **v0.5.19** released 2026-09-05 (tag
> commit `0bcd822377da7b5718e674eaf9c870d349424dd1`, cut 2026-09-03T14:22
> UTC), superseding v0.5.18. The bug ships in it unchanged: source-confirmed
> on the v0.5.19 tag, `ModelRunner._resolve_draft_load_format()`
> (`model_runner.py:1368`) still returns `None` unless
> `get_spec().speculative_draft_load_format` is explicitly set,
> `_load_format_scope()` at line 1356, consumption at line 354;
> `scheduler.py:maybe_init_draft_worker()` unchanged at line 973 (no drift
> from the 2026-09-02 check). v0.5.19's release notes (Speculative Decoding
> section, #31041/#36186/#34337/#36313/#35275 etc.) contain no fix touching
> `speculative_draft_load_format`. Issue #32202 still open, 0 comments, idle
> since 2026-07-23. PR #34622 (adjacent GPTQ-draft-quantization bug, not
> ours) still open, no activity since 2026-08-18. `upstream/main` HEAD now
> `908226fea2df861769e2720161a75649ae4c6f92` (2026-09-10);
> `_resolve_draft_load_format()` moved to line 1402,
> `maybe_init_draft_worker()` to line 1020, logic byte-for-byte identical.
> The `--speculative-draft-load-format auto` +
> `--speculative-draft-model-path` workaround remains required and
> unchanged.

> **Re-verified 2026-09-16:** No new SGLang release since v0.5.19 (2026-09-05,
> still latest). Issue #32202 unchanged: still open, 0 comments, idle since
> 2026-07-23. PR #34622 (adjacent GPTQ-draft-quantization bug, not ours)
> unchanged: still open, no activity since 2026-08-18. No new issue or PR
> found referencing `sharded_state`/`speculative_draft_load_format`. Local
> note: the cluster image was bumped from
> `xomoxcc/dgx-spark-sglang:0.5.17-sm121` to
> `xomoxcc/dgx-spark-sglang:0.5.19-sm121` (repo commits f36d51b/958c98d/5c07a34,
> 2026-09-11..15). This doc's 2026-09-10 entry already source-checked the bug
> directly on the v0.5.19 tag, so the image bump changes no conclusion. The
> `--speculative-draft-load-format auto` + `--speculative-draft-model-path`
> workaround remains required and unchanged.

> **Re-verified 2026-09-22:** SGLang **v0.5.20** released 2026-09-18 (tag
> commit `94602c9c2b7cbdb8efd5c52802dac6a1c180089e`, published
> 2026-09-18T22:41:33Z), now the latest release, superseding v0.5.19.
> Source-confirmed on the v0.5.20 tag: the bug is unchanged.
> `ModelRunner._resolve_draft_load_format()` (`model_runner.py:1405`) still
> returns `get_spec().speculative_draft_load_format` under the same
> `if not self.is_draft_worker: return None` gate, still `None` by default,
> so an unset value still falls through to the main model's `load_format`
> via `_load_format_scope()` (line 1393, `None` maps to
> `contextlib.nullcontext()`). Consumption in `__init__` is now at line 360
> (was 362 on v0.5.19), the load-path scope block at line 1202.
> `scheduler.py:maybe_init_draft_worker()` moved from line 973 (v0.5.19) to
> line 1024 on v0.5.20, a real shift of about 51 lines from unrelated
> refactor churn (renames spotted in the same diff: `is_cp_v2_active` to
> `is_cp_active`, `is_mla_prefill_cp_enabled` to `is_mla_cp_enabled`,
> `get_global_server_args` removed). Logic is byte-for-byte identical, only
> comments and docstrings were added around the two functions. Issue #32202
> (the adjacent S3-path-resolution bug, not this doc's bug) was
> **auto-closed by the stale bot today, 2026-09-22T00:28:30Z**
> ("automatically closed due to inactivity"), administrative, not fixed, no
> reopening. PR #34622 (adjacent GPTQ-draft-quantization bug, not ours)
> unchanged, still open, no activity since 2026-08-18. A GitHub search for
> `sharded_state speculative` / `speculative_draft_load_format sharded`
> still returns nothing new. The `--speculative-draft-load-format auto` +
> `--speculative-draft-model-path` workaround remains required and
> unchanged. Local note: repo commits 63e72da/f3f29e1 (2026-09-22) add
> SGLang v0.5.20 SM121 build patches and
> `scripts/patches/sglang-0.5.20-sm121.recipe`, but the image is not yet
> built or deployed; the cluster stays on
> `xomoxcc/dgx-spark-sglang:0.5.19-sm121`. This entry's source-check was run
> directly against the v0.5.20 tag, so once that image ships the workaround
> requirement is unaffected.

> **Re-verified 2026-09-25:** No new SGLang release since v0.5.20 (2026-09-18,
> still latest per `gh release list`/`gh api tags`). Issue #32202 stays closed
> (stale bot, 2026-09-22), no reopening, no new activity. PR #34622 unchanged,
> still open, no activity since 2026-08-18. Local note: repo commit `3214ed2`
> (dated 2026-09-22 per `git log`) flipped `default_sglang_image` to
> `xomoxcc/dgx-spark-sglang:0.5.20-sm121`, and the live cluster is now
> confirmed running that image on all sglang pods (`kubectl get pods -n
> sglang`, head plus 3 workers, checked today). This doc's 2026-09-22 entry
> already source-checked the bug directly against the v0.5.20 tag
> (`ModelRunner._resolve_draft_load_format()` and `maybe_init_draft_worker()`,
> logic byte-for-byte unchanged, only line-shifted), so nothing new to
> re-verify now that v0.5.20 is actually live. The
> `--speculative-draft-load-format auto` plus `--speculative-draft-model-path`
> workaround remains required and unchanged.

> **Re-verified 2026-09-28:** No new SGLang release since v0.5.20 (2026-09-18,
> still latest). Issue #32202 stays closed (stale bot, 2026-09-22), no
> reopening. PR #34622 remains OPEN, no activity since 2026-08-18. Nine
> commits touched `scheduler.py` / `model_runner.py` since 2026-09-25, none
> touching `maybe_init_draft_worker`, `_resolve_draft_load_format`,
> `_load_format_scope`, or `draft_load_format` (checked each commit's diff
> against these functions). Source-confirmed on `main` HEAD (`df378bca42`,
> 2026-09-28T16:59:59+08:00): the root-cause gate is unchanged,
> `_resolve_draft_load_format()` still returns `None` unless
> `speculative_draft_load_format` is explicitly set; the function moved to
> `model_runner.py:1447` (was 1405 on the v0.5.20 tag) and
> `maybe_init_draft_worker()` moved to `scheduler.py:1006` (was 1024 on
> v0.5.20), pure line drift from unrelated commits, logic byte-for-byte
> identical. The `--speculative-draft-load-format auto` plus
> `--speculative-draft-model-path` workaround remains required and unchanged.

> **Re-verified 2026-10-03:** SGLang v0.5.21 was released 2026-10-02. At the
> v0.5.21 tag, `_resolve_draft_load_format()` is at `model_runner.py:1447`
> and `maybe_init_draft_worker()` at `scheduler.py:1006` (same line numbers
> as `main` showed on 2026-09-28, so the tag was cut shortly after that).
> On `main` HEAD (`7be5e3473c`, 2026-10-03T02:21:55-0700) they drifted to
> `model_runner.py:1449` and `scheduler.py:1010`, pure line drift from
> stacked refactors (`a9c97c9f69`, `986cb8268b`, `499e86db98` in
> `model_runner.py`/`tp_worker.py`; `986cb8268b` and `499e86db98` checked:
> no `load_format` or `_init_model_config` change). The root-cause gate is
> unchanged: `_resolve_draft_load_format()` still returns `None` unless
> `speculative_draft_load_format` is explicitly set. Issue #32202 stays
> closed (2026-09-22). PR #34622 remains OPEN, no activity since
> 2026-08-18. The `--speculative-draft-load-format auto` plus
> `--speculative-draft-model-path` workaround remains required and
> unchanged.

> **Re-verified 2026-10-11:** No new SGLang release since v0.5.21
> (2026-10-02). On `main` HEAD (`ad8b1e1f85`, 2026-10-10)
> `_resolve_draft_load_format()` is at `model_runner.py:1483` and
> `maybe_init_draft_worker()` at `scheduler.py:1003`; the function body is
> byte-identical to 2026-10-03 (draft runner only, returns
> `get_spec().speculative_draft_load_format`, so `None` unless explicitly
> set). Eleven commits touched `model_runner.py` and fifteen `scheduler.py`
> since 2026-10-03; the only one near the load path, `6e8f3e09dd`
> (InstantTensor loader, #40453), adds a `load_group` argument but leaves the
> draft load-format resolution and its call sites unchanged. Issue #32202
> stays closed (2026-09-22). PR #34622 remains OPEN, no activity since
> 2026-08-18. The `--speculative-draft-load-format auto` plus `--speculative-
> draft-model-path` workaround remains required and unchanged.

> **Re-verified 2026-11-10:** SGLang v0.5.22 (2026-10-19) and v0.5.23 (2026-11-02, latest) are out. `_resolve_draft_load_format()` is at `model_runner.py:1514` / `scheduler.py` `maybe_init_draft_worker()` at `:1011` on v0.5.22, `:1538` / `:1035` on v0.5.23, and `:1626` / `:1097` on `main` HEAD (`170804b40a`, 2026-11-10); pure line drift. The body is unchanged: draft runner only, returns `get_spec().speculative_draft_load_format`, so `None` unless explicitly set. No commit touching `speculative_draft_load_format` on `main` since 2026-10-12. Issue #32202 stays closed (2026-09-22). PR #34622 remains OPEN, no activity since 2026-08-18. The `--speculative-draft-load-format auto` plus `--speculative-draft-model-path` workaround remains required and unchanged.

- File: `sglang/srt/managers/scheduler.py`, method `maybe_init_draft_worker()`
- Root cause in: `sglang/srt/managers/tp_worker.py`, method `_init_model_config()`

## Affected Configuration

- Load format: `sharded_state` (pre-sharded TP checkpoints)
- Speculative decoding: any algorithm (`NEXTN`, `EAGLE`, `EAGLE3`)
- Tested with: Qwen3.5-122B-A10B-FP8, TP=2, EP=2, NEXTN, SGLang 0.5.9

Models loaded with `--load-format auto` (default) are **not affected**.

## The Bug

When `load_format=sharded_state` is combined with speculative decoding, the draft
model's `ModelRunner` inherits the same `load_format` and `model_path` as the main
model. The `ShardedStateLoader` then attempts to load draft model weights from the
per-rank shard files, which only contain main model weight keys. Draft/MTP model
parameters that don't exist in the shard state dict cause a `KeyError` crash:

```
Scheduler hit an exception: Traceback (most recent call last):
  File ".../scheduler.py", line 3130, in run_scheduler_process
    scheduler = Scheduler(...)
  File ".../scheduler.py", line 368, in __init__
    self.init_model_worker()
  File ".../scheduler.py", line 565, in init_model_worker
    self.maybe_init_draft_worker()
  File ".../scheduler.py", line 561, in maybe_init_draft_worker
    self.draft_worker = DraftWorkerClass(**draft_worker_kwargs)
  File ".../speculative/eagle_worker.py", line 142, in __init__
    super().__init__(...)
  File ".../managers/tp_worker.py", line 247, in __init__
    self._init_model_runner()
  ...
  File ".../model_loader/loader.py", line 1426, in load_model
    param_data = state_dict[key].data
KeyError: 'model.layers.47.input_layernorm.weight'
```

The crash happens deterministically on every startup attempt. The main model loads
successfully (~5 min for 13 shards), CUDA graphs are captured, but then the Scheduler
subprocess dies when initializing the draft worker. Exit code 137 (SIGKILL from the
child's SIGQUIT propagation).

## Root Cause

The code path for draft model loading:

1. `Scheduler.maybe_init_draft_worker()` creates `draft_worker_kwargs` with
   `server_args=self.server_args` (same object as the main model's server_args).

2. If `speculative_draft_load_format` is set, `maybe_init_draft_worker` overrides
   `self.server_args.load_format` in-place. But if it's `None` (the default), the
   draft model inherits whatever `load_format` the main model uses — including
   `sharded_state`.

3. `TpModelWorker._init_model_config()` correctly selects
   `speculative_draft_model_path` for the draft model's `model_path`. But when
   `speculative_draft_model_path` is also `None` (default), `ServerArgs.__init__`
   sets it to `self.model_path` — i.e., the sharded directory.

4. The draft model's `ModelRunner` thus uses `ShardedStateLoader` pointing at the
   sharded directory. The shard files contain per-rank weight slices for the **main**
   model architecture. The **draft/MTP** model has a different architecture with
   different weight keys, causing `KeyError` on the first non-matching key.

The fundamental issue: `ServerArgs` defaults make the draft model silently inherit
both `load_format` and `model_path` from the main model, which is correct for
`auto` loading (the HF cache contains all weights) but broken for `sharded_state`
(the shard files only contain main model weights).

## Fix (upstream)

SGLang should either:

1. Default `speculative_draft_load_format` to `"auto"` when `load_format` is
   `"sharded_state"` (since pre-sharded checkpoints are never created with draft
   model awareness), or

2. Default `speculative_draft_model_path` to the original model ID/path (not the
   sharded directory) when `load_format` is `"sharded_state"`.

The infrastructure is already in place — `speculative_draft_load_format` and
`speculative_draft_model_path` exist and work correctly. The bug is purely in
the defaults not accounting for the `sharded_state` case.

## Our Workaround

No monkey-patch needed. SGLang already provides CLI flags for draft model overrides.
In `sglang_launch.sh`, when both speculative and sharded_state are active:

```bash
if [ "$SGLANG_SPECULATIVE_ENABLED" = "true" ]; then
  args+=(--speculative-algo "$SGLANG_SPECULATIVE_ALGO")
  # ... other speculative args ...

  # WORKAROUND: force auto load format for draft model when using sharded_state
  if [ "$SGLANG_LOAD_FORMAT" = "sharded_state" ]; then
    args+=(--speculative-draft-load-format auto)
    args+=(--speculative-draft-model-path "$SGLANG_MODEL")
  fi
fi
```

This forces the draft model to:
- Use `DefaultModelLoader` (`auto`) instead of `ShardedStateLoader`
- Load from the original HF model ID (resolved from HF cache) instead of the
  sharded directory

The main model continues to use `sharded_state` for fast loading. The draft model
loads from the HF cache, which adds a few seconds to startup but works correctly.

**Note (resolved upstream 2026-08-03, historical for ≤v0.5.16):**
`maybe_init_draft_worker` used to modify `self.server_args.load_format` in-place
(not a copy) when `speculative_draft_load_format` is set, mutating the shared
`server_args` object. Harmless in practice (the main model is already loaded at
that point), but a latent bug if SGLang ever re-read `load_format` after draft
worker initialization. Fixed on `main` by PR #33335 (2026-08-03): the draft
worker is now built from a `deepcopy` of `server_args` and only the copy is
overridden (see the 2026-08-03 re-verification entry above). Still present in
every release up to and including v0.5.16; **v0.5.17 (2026-08-08) is the first
release to ship the deepcopy fix** (its release branch carries the 2026-08-03
`draft_server_args_copy()` code shape, see the 2026-08-09 entry above).

## Impact

Without the workaround, speculative decoding (NEXTN/EAGLE) is completely unusable
with `--load-format sharded_state`. The pod enters a crash loop: each restart loads
the main model (~5 min), then immediately crashes on draft worker init. The worker
pod's NCCL connection breaks on each head restart, requiring a livenessProbe-driven
restart of the worker as well — doubling the blast radius.

## Related

- `SGLANG_TP_EP_MOE_UPSTREAM_BUG.md` — moe_wna16 qzeros + EP bug (same version)
- SGLang `ServerArgs.speculative_draft_load_format` — the escape hatch we use
- SGLang `TpModelWorker._init_model_config()` — correctly dispatches draft vs main
  model path, but relies on correct defaults upstream
