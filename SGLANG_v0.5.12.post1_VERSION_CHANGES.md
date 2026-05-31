# SGLang v0.5.12.post1 — Relevante Änderungen seit v0.5.12

Quelle: [Release Notes v0.5.12.post1](https://github.com/sgl-project/sglang/releases/tag/v0.5.12.post1) · Veröffentlicht 2026-05-26 · Diff: [`v0.5.12...v0.5.12.post1`](https://github.com/sgl-project/sglang/compare/v0.5.12...v0.5.12.post1)

Begleitdokument zu [`SGLANG_v0.5.12_VERSION_CHANGES.md`](./SGLANG_v0.5.12_VERSION_CHANGES.md) und [`TODO_0.5.12.md`](./TODO_0.5.12.md).

**TL;DR vorab (revidiert 2026-05-31):** `.post1` ist ein **reiner Stability-Patch** — ~12–15 Cherry-Picks auf den `release/v0.5.12`-Branch, **fast ausschließlich DeepSeek-V4- und PD-Disaggregation-Fixes**. Die ursprüngliche Einschätzung „für uns irrelevant" galt, solange wir DeepSeek V4 gar nicht fuhren. **Das hat sich geändert:** Der Default ist jetzt auf die kleine Variante **`sgl-project/DeepSeek-V4-Flash-FP8`** gesetzt (`roles/k8s_dgx/defaults/main.yml:31`) und das Image wurde bewusst auf `xomoxcc/dgx-spark-sglang:0.5.12.post1-sm121` gebumpt — **genau wegen der DSv4-Stability-Cherry-Picks unten**. ⚠️ **Wichtig:** V4-Flash-FP8 ist damit der per Default *versuchte* Pfad, aber das Modellprofil ist `UNTESTED / first-contact` — es ist **nicht** validiert, dass FP8-V4-Flash auf 4×GB10 bootet/coherent dekodiert. PD fahren wir weiterhin nicht disaggregiert (die PD-Items bleiben irrelevant). Das volle DeepSeek V4 (671B-Klasse) passt weiterhin nicht (siehe `SGLANG_v0.5.12_VERSION_CHANGES.md` §2) — nur die **Flash**-Variante ist der Versuch wert.

> Das Default-Image steht jetzt auf `xomoxcc/dgx-spark-sglang:0.5.12.post1-sm121` (`roles/k8s_dgx/defaults/main.yml`, Override + alle V4-Flash/DeepSeek-Profile). Der `.post1`-Bump ist also **erfolgt** — die DSv4-Single-Token-Decode- und Accuracy-Fixes (§1) sind für den V4-Flash-Boot-Versuch potenziell entscheidend.

---

## 1. Was tatsächlich drin ist (Cherry-Pick-Liste)

`.post1` cherry-picked Fixes onto `release/v0.5.12`. Die Release-Notes nennen die **Upstream-PR-Nummern**, der Diff zeigt die **Cherry-Pick-PRs auf dem Release-Branch** — daher Doppelnennungen.

### DeepSeek V4 (Mehrheit — jetzt relevant, seit V4-Flash-FP8 der Default ist)

> Diese Sektion war ursprünglich als „für uns nicht relevant" eingestuft. Seit `sgl-project/DeepSeek-V4-Flash-FP8` der Default-Versuch ist, sind v. a. die ersten beiden Items unmittelbar relevant für die Frage, ob V4-Flash überhaupt sauber dekodiert.

- **#25733 / #26063** — Garbled text beim Single-Token-Decode auf B200/B300; Fix im `deep_gemm` UE8M0 Scale-Packing (`fp8_einsum`-Input-Scale → ue8m0). **→ V4-Flash-FP8 nutzt ue8m0-Block-FP8 (`scale_fmt=ue8m0`); genau der Pfad, den dieser Fix anfasst. Direkt relevant für unseren Boot-Versuch.**
- **#25805 / #26078** — Crash um ~2000 Requests bei EAGLE/MTP in **Disagg-Decode** wegen stale Sliding-Window-Mappings (SWA double-free).
- **#25889 / #26079** — Stale Translation-Indices in HiCache → falsche Outputs (`cached_loc` invalidiert beim SWA-Mapping-Rebuild).
- **#25646 / #26072** — GSM8K-Accuracy von **0.825 → 0.960** mit HiSparse + `SGLANG_OPT_USE_COMPRESSOR_V2=1` wiederhergestellt.
- **#25396** — Scheduler-Crash beim Startup mit NSA-Prefill Context-Parallel im Disagg-Mode.
- **#25771 / #26076** — DSV4 PD-Disagg mit `pp_size > 1` freigeschaltet (stale `pp_size=1`-Assertion entfernt).
- **#25892 / #26075** — CUDA illegal memory access mit `--load-format dummy` + FlashInfer mxfp4.
- **#26154** — DeepSeek-V4 Context-Parallel-Error-Fix.
- **#26077** — `git gemm` Wrapper für `dispatch_bf16_fp32_backend`.

### PD-Disaggregation (für uns nicht relevant — wir fahren nicht disaggregiert)

- **#25699 / #25731** — Auxiliary-Data-Handling im PD/NIXL-Mode.

### CI / interne Hygiene (für uns irrelevant)

- **#26109** — base-* Suite-Namen, die durch Cherry-Picks reingezogen wurden.
- **#26111** — `run_all_tests` in `workflow_dispatch`-Inputs.
- **#26113** — Self-Heal für `$GITHUB_PATH`/`$GITHUB_ENV`-Writes.

---

## 2. Die drei Items, die uns *überhaupt* tangieren

### 2.1 ⚠️ `get_dp_buffer` — fehlendes `group`-Argument (#25585 / #26070)

Bug-Fix: fehlendes `group`-Argument in `get_dp_buffer`. Betrifft den **DP-Attention-Pfad**. Wir fahren auf den meisten MoE-Profilen DP-Attention (zusammen mit TP/EP), daher ist das der **einzige Fix mit potenziell direktem Bezug** zu unseren Standard-Deployments.

→ **Bewertung:** Niedriges Risiko, aber gut zu wissen. Falls auf 0.5.12 (ohne `.post1`) ein DP-Attention-Profil sporadisch crasht, ist das ein Kandidat. Kein proaktiver Handlungsbedarf — wenn unsere Profile auf 0.5.12 sauber laufen, sind wir nicht betroffen.

### 2.2 `nvidia-cutlass-dsl` → `[cu13]`-Extra für CUDA 13 (#25576 / #25931)

Dependency-Bump: `nvidia-cutlass-dsl` nutzt jetzt das `[cu13]`-Extra (nötig für sm_103 / B300 unter CUDA 13). Wir sind **SM121 / CUDA 13** und der Cute-DSL-FP4-Pfad (`flashinfer_cute_dsl`, Reland #23590 aus 0.5.12) ist genau einer unserer offenen Matrix-Test-Kandidaten (siehe `TODO_0.5.12.md` §4).

→ **Bewertung:** Wenn wir ohnehin ein `xomoxcc/dgx-spark-sglang:0.5.12-sm121` rebuilden (siehe `TODO_0.5.12.md`), beim Pinnen der `nvidia-cutlass-dsl`-Version das `[cu13]`-Extra mitnehmen. Bei `scitrera`-Base ist das transparent. Relevant **nur** für unseren Custom-Build.

### 2.3 Precompiled DeepGEMM-Branch + MHC-Prewarm (#25860, #25810 / #26071)

- **#25860** — Precompiled DeepGEMM-Branch reduziert Runtime-JIT-Compile-Kosten.
- **#25810 / #26071** — MHC Token-Count-Buckets beim Startup vorwärmen (eliminiert 20–40 s Cold-Bucket-Stalls). **DSv4-spezifisch** (MHC = Multi-Head-Compression-Pipeline aus dem DSv4-Pfad).

→ **Bewertung:** Der DeepGEMM-Precompile-Punkt *könnte* Startup-JIT auf MoE-Pfaden marginal entlasten — aber unklar, ob er außerhalb des DSv4-Pfads greift. MHC-Prewarm ist DSv4-only. Beides nicht messen wert, solange wir kein DSv4 fahren. Falls wir auf 0.5.12 lange Startup-Stalls beim ersten DeepGEMM-JIT sehen (separat von der bekannten ~7–8-min-Head-Startup-Zeit, siehe CLAUDE.md), wäre `.post1` einen Versuch wert.

---

## 3. Was *nicht* drin ist (Erwartungs-Check)

Damit klar ist, dass `.post1` unsere offenen 0.5.12-Punkte **nicht** löst:

- **`cutlass_moe_fp4`-Crash auf SM121** — weiterhin **nicht** adressiert. Die modellspezifischen Workarounds in den Profilen bleiben (Triton für Minimax-M2.5 PP=4; direktes `cutlass` für Qwen3.5-397B-NVFP4; `flashinfer_cutlass` als Default).
- **Eager-Mode auf `cutlass_moe_fp4`** (`!`-Token-Collapse) — kein Fix.
- **HAProxy-Sidecar `EADDRINUSE`-Workaround** — kein Scheduler-Bind-Fix; weiter nötig.
- **SM120/121-Gemma-4-NVFP4-PRs** (#22929, #22928, #22927, #22615) — **nicht** enthalten. Die NVFP4-Gemma-Modelle bleiben blockiert (siehe `TODO_0.5.12.md` §C und `SGLANG_GEMMA4_UPSTREAM_BUG.md`). `.post1` ändert daran nichts.

---

## 4. Empfehlung

1. **Bump ist erfolgt — und gerechtfertigt, sobald wir V4-Flash versuchen.** Ursprünglich als „kein dringender Bump" eingestuft, weil DSv4 außerhalb unseres Workloads lag. Mit `sgl-project/DeepSeek-V4-Flash-FP8` als Default ist die DSv4-Stabilisierung jetzt **im** Workload-Profil. Das Image steht entsprechend auf `0.5.12.post1-sm121`. Die NVFP4-MoE-Matrix-Tests (siehe `TODO_0.5.12.md`) bleiben für die *anderen* Profile der eigentliche Blocker — aber für V4-Flash sind die `.post1`-DSv4-Fixes (#25733/#26063 Single-Token-Decode, #25646/#26072 Accuracy) Voraussetzung, nicht Nice-to-have.
2. **Wenn wir 0.5.12 ohnehin gerade validieren**: direkt gegen `.post1` testen statt gegen `.12`, damit der DP-Attention-`get_dp_buffer`-Fix (#25585) mit drin ist — kostet nichts extra, schließt eine potenzielle Crash-Quelle aus. Voraussetzung: `scitrera` (oder unser Custom-Build) hat ein `.post1`-Tag.
3. **Beim SM121-Custom-Rebuild** das `nvidia-cutlass-dsl[cu13]`-Extra (#25576) berücksichtigen — passt zu unserem CUDA-13 / Cute-DSL-FP4-Matrix-Test.
4. **Ansonsten abwarten** auf die nächste Minor (0.5.13?) für die SM121-/NVFP4-relevanten Themen.

---

## Offene Fragen

- Baut `scitrera` ein `0.5.12.post1`-Tag? (`docker manifest inspect scitrera/dgx-spark-sglang:0.5.12.post1` prüfen, sobald wir bumpen wollen.)
- Greift der DeepGEMM-Precompile (#25860) auf unseren MoE-Pfaden (nicht-DSv4) und verkürzt er die Head-Startup-Zeit messbar? Nur testen, falls wir auf 0.5.12 auffällige erste-JIT-Stalls sehen.
