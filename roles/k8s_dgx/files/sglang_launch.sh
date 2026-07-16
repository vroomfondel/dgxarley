#!/bin/bash
set -e

# Runtime deps (tini/ping/ip/ethtool/net-tools/curl — used for ARP priming + the
# QSFP peer-wait) are NOT in the stock upstream sglang image and get apt-installed
# at boot. Our custom sm121 image bakes them in, so skip the whole apt bootstrap
# when either signal says the deps are already there:
#   1. DGXARLEY_RUNTIME_LIBS_BAKED=1 — explicit marker set in our image Dockerfile.
#   2. Every needed binary is already on PATH (works even without the marker).
# This keeps the stock image working (installs below) while our image starts
# instantly and never touches the mirror.
_skip_apt=true
if [ "${DGXARLEY_RUNTIME_LIBS_BAKED:-0}" = "1" ]; then
  echo "[launch] DGXARLEY_RUNTIME_LIBS_BAKED=1 — runtime deps baked into image, skipping apt bootstrap"
else
  for _b in tini ip ping ethtool netstat curl; do
    if ! command -v "$_b" >/dev/null 2>&1; then _skip_apt=false; break; fi
  done
  [ "$_skip_apt" = true ] && echo "[launch] runtime deps already on PATH — skipping apt bootstrap"
fi

# apt bootstrap for the stock image only. Canonical's ports.ubuntu.com is US-hosted
# behind a ~40-60% packet-loss Zayo transit path from this site (measured 2026-07-15:
# 60% ICMP loss from the workstation, path exits via Zayo → Boston/US; 1.1.1.1 is 0%
# loss), so a single apt attempt fails ~80% of the time and, under `set -e`,
# crashloops the pod with exit 100. Three mitigations: (1) force IPv4 (cluster is
# IPv4-only; belt to the CoreDNS no-AAAA change); (2) repoint apt at the well-peered
# German FAU Erlangen ports mirror (0% loss, ~29ms, full noble suites verified);
# (3) retry update+install in a bounded loop. Non-fatal on total failure.
if [ "$_skip_apt" != true ]; then
  echo 'Acquire::ForceIPv4 "true";' > /etc/apt/apt.conf.d/99force-ipv4
  sed -i 's|ports.ubuntu.com/ubuntu-ports|ftp.fau.de/ubuntu-ports|g' \
    /etc/apt/sources.list /etc/apt/sources.list.d/*.sources /etc/apt/sources.list.d/*.list 2>/dev/null || true

  apt_ok=false
  for i in $(seq 1 15); do
    apt-get update -qq 2>/dev/null || true
    if apt-get install -y -qq tini iproute2 iputils-ping net-tools curl ethtool >/dev/null 2>&1; then
      apt_ok=true
      break
    fi
    echo "[launch] apt install attempt ${i}/15 failed (lossy mirror), retrying in 5s..."
    sleep 5
  done
  [ "$apt_ok" = true ] || echo "[launch] WARNING: apt install failed after 15 attempts; continuing (ping-dependent steps may fail)"
fi

# accelerate: required by SGLang's ModelOptModelLoader
# (srt/model_loader/loader.py → _load_modelopt_base_model). Triggered by:
#   - GLM-5-NVFP4 (modelopt base model load path)
#   - EAGLE3/speculative decoding with a modelopt-quantized target model
#     (e.g. nvidia/Qwen3-235B-A22B-NVFP4 + lmsys EAGLE3 draft) — the draft
#     worker loads the target's embeddings via _load_modelopt_base_model
#     and hits ImportError if accelerate is missing.
# Upstream scitrera/dgx-spark-sglang image does NOT ship accelerate.
if [[ "$SGLANG_MODEL" == *"GLM-5"* ]] || [ "$SGLANG_SPECULATIVE_ENABLED" = "true" ]; then
  python3 -c "import accelerate" 2>/dev/null || pip install accelerate
fi

# GLM-5 specific: transformers upgrade + mem_get_info patch.
# Only needed for glm_moe_dsa models — skip for MiniMax, Qwen, etc.
if [[ "$SGLANG_MODEL" == *"GLM-5"* ]]; then
  echo "GLM-5 model detected — applying GLM-5 specific patches..."

  # transformers ≥5.3.0: required for glm_moe_dsa model type.
  # Must also pull huggingface_hub >=1.3.0 (transformers 5.3.0 dependency).
  python3 -c "from transformers.models.auto.configuration_auto import CONFIG_MAPPING; assert 'glm_moe_dsa' in CONFIG_MAPPING" 2>/dev/null \
    || pip install transformers==5.3.0 huggingface_hub==1.3.0

  # Patch _cuda_mem_fallback: transformers 5.x + huggingface_hub >=1.3.0
  # triggers a CUDA context init during import that breaks torch.cuda.mem_get_info()
  # on GB10 (cudaErrorMemoryAllocation). nvidia-smi also can't report memory on GB10.
  # Fix: fall back to /proc/meminfo (GB10 unified memory = system RAM).
  #
  # RE-ANCHORED 2026-07-16: upstream refactored the old inline "torch.cuda.mem_get_info()
  # failed -> raise RuntimeError" branch of get_nvgpu_memory_capacity() into a standalone
  # helper function ALSO named _cuda_mem_fallback() (name collision with our marker, not
  # our patch — it's upstream's own tier-1 nvidia-smi->mem_get_info() fallback, called from
  # 3 sites). Our tier-2 (mem_get_info() ALSO fails -> /proc/meminfo) now anchors inside
  # that function's except block. NOTE: in practice this tier is not exercised on this
  # cluster (mem_get_info() succeeds — see live logs: "Falling back to
  # torch.cuda.mem_get_info(). Reported total GPU memory per device (MiB): [124546]"), but
  # kept as defense-in-depth for the driver-stack edge case the comment above describes.
  # Also fixed here: the old anchor had no dedicated "already applied" marker in the
  # outer bash gate (it grepped the function NAME, which now collides with upstream's own
  # function) — re-running against an already-patched file would silently re-match part of
  # its own injected code and could double-patch on every pod restart. Now gated on the
  # marker string itself.
  COMMON_PY="/usr/local/lib/python3.12/dist-packages/sglang/srt/utils/common.py"
  if [ ! -f "$COMMON_PY" ]; then
    echo "ANCHOR-DRIFT: common.py: _sgl_cuda_mem_fallback_proc_meminfo_ target file missing (SGLang restructured/renamed?)"
  elif grep -q '_sgl_cuda_mem_fallback_proc_meminfo_' "$COMMON_PY" 2>/dev/null; then
    echo "common.py: _cuda_mem_fallback proc-meminfo tier already patched, skipping"
  elif grep -q 'def _cuda_mem_fallback' "$COMMON_PY" 2>/dev/null; then
    python3 << 'PATCH_MEM_FALLBACK_EOF'
f = "/usr/local/lib/python3.12/dist-packages/sglang/srt/utils/common.py"
with open(f) as fh:
    code = fh.read()
marker = "# [patch] _sgl_cuda_mem_fallback_proc_meminfo_"
old = '''    except (RuntimeError, ValueError, OSError) as e:
        raise RuntimeError(
            f"{reason} torch.cuda.mem_get_info() fallback also failed: {e}"
        ) from e'''
new = '''    except (RuntimeError, ValueError, OSError) as e:
        ''' + marker + ''' -- GB10 unified memory: try /proc/meminfo as a last resort
        # before giving up (rare case: transformers/huggingface_hub break
        # torch.cuda.mem_get_info() during import, cudaErrorMemoryAllocation).
        try:
            with open("/proc/meminfo") as _mf:
                _mem_mib = None
                for _line in _mf:
                    if _line.startswith("MemTotal:"):
                        _mem_mib = int(_line.split()[1]) // 1024  # kB -> MiB
                        break
            if _mem_mib is not None:
                logger.warning(
                    f"{reason} torch.cuda.mem_get_info() fallback also failed: {e}. "
                    f"Falling back to /proc/meminfo MemTotal: {_mem_mib} MiB."
                )
                return _mem_mib
        except OSError:
            pass
        raise RuntimeError(
            f"{reason} torch.cuda.mem_get_info() fallback also failed: {e}"
        ) from e'''
if marker in code:
    print("common.py: _cuda_mem_fallback proc-meminfo tier already patched, skipping")
elif old not in code:
    print("ANCHOR-DRIFT: common.py: _sgl_cuda_mem_fallback_proc_meminfo_ (SGLang version drift; re-check anchor)")
else:
    code = code.replace(old, new, 1)
    with open(f, 'w') as fh:
        fh.write(code)
    print("Patched common.py: _cuda_mem_fallback proc-meminfo tier added")
PATCH_MEM_FALLBACK_EOF
  else
    echo "ANCHOR-DRIFT: common.py: _cuda_mem_fallback function not found (SGLang version drift; re-check anchor)"
  fi
else
  echo "SKIPPING GLM-5 specific patches..."
fi

# ============================================================================
# Hunyuan (Hy3 / HYV3) special-token suffix backport — SGLang PR #29920
# ----------------------------------------------------------------------------
# This image predates PR #29920 (merged 2026-07-04, "resolve special-token
# suffix at runtime"). Its hunyuan tool-call AND reasoning detectors HARDCODE
# the bare structural tokens (<think>, <tool_calls>, <tool_call>, <tool_sep>,
# <arg_key>, <arg_value>). The shipping Hy3 checkpoints (vroomfondel/Hy3-NVFP4-
# W4A4, tencent/Hy3, ...) append a shared suffix to EVERY such token — the
# chat template's HYTK, mirrored by tokenizer_config.json "token_suffix", e.g.
# ":opensource" so the model emits <think:opensource>, <tool_calls:opensource>,
# ... (verified: these are single special tokens; the BARE forms are not tokens
# at all). With the bare-token detectors, the reasoning split never fires and NO
# tool calls are parsed → breaks honcho/hindsight function-calling.
#
# Faithful backport of PR #29920's resolve_hunyuan_tokens() into the two detector
# files. Upstream threads the tokenizer through ~8 caller files to feed the vocab
# to resolve_hunyuan_tokens(); this image threads no tokenizer, so we instead feed
# the resolved suffix via SGLANG_HUNYUAN_TOKEN_SUFFIX (read below from the model's
# tokenizer_config.json "token_suffix"). Empty suffix (preview checkpoints, or a
# non-Hy3 model) → bare tokens = unchanged upstream-preview behavior.
# RE-SYNC: when bumping to an image that already contains PR #29920, DELETE this
# block — the "resolve_hunyuan_tokens" grep guard already makes it a no-op then.
if [[ "$SGLANG_MODEL" == *"Hy3"* || "$SGLANG_MODEL" == *"Hunyuan"* \
      || "$SGLANG_TOOL_CALL_PARSER" == "hunyuan" || "$SGLANG_REASONING_PARSER" == "hunyuan" ]]; then
  export SGLANG_HUNYUAN_TOKEN_SUFFIX="$(python3 - <<'HY_SUFFIX_EOF'
import json, os
model = os.environ.get("SGLANG_MODEL", "")
suffix = ""
path = None
try:
    if os.path.isdir(model):
        cand = os.path.join(model, "tokenizer_config.json")
        path = cand if os.path.exists(cand) else None
    if path is None:
        try:
            from transformers.utils import cached_file
            path = cached_file(model, "tokenizer_config.json", local_files_only=True)
        except Exception:
            from huggingface_hub import hf_hub_download
            path = hf_hub_download(model, "tokenizer_config.json", local_files_only=True)
    if path:
        with open(path) as fh:
            suffix = json.load(fh).get("token_suffix", "") or ""
except Exception:
    suffix = ""
print(suffix, end="")
HY_SUFFIX_EOF
)"
  echo "Hunyuan token-suffix backport: SGLANG_HUNYUAN_TOKEN_SUFFIX='${SGLANG_HUNYUAN_TOKEN_SUFFIX}'"

  # --- 1) function_call/hunyuan_detector.py: resolve_hunyuan_tokens + suffixed
  #        tool-call tokens (bot/eot/tool_call/tool_sep/arg_key/arg_value + the
  #        two regexes + structure_info). ---
  python3 << 'PATCH_HUNYUAN_TOOL_EOF'
f = "/usr/local/lib/python3.12/dist-packages/sglang/srt/function_call/hunyuan_detector.py"
with open(f) as fh:
    code = fh.read()

if "resolve_hunyuan_tokens" in code:
    print("hunyuan_detector.py: resolve_hunyuan_tokens already present, skipping")
else:
    anchor = "logger = logging.getLogger(__name__)\n"
    helper = r'''
# [patch] _sgl_hunyuan_token_suffix_ — backport of SGLang PR #29920
import os as _os

_HUNYUAN_TOKEN_NAMES = (
    "tool_calls",
    "tool_call",
    "tool_sep",
    "arg_key",
    "arg_value",
    "think",
)

_HUNYUAN_TOKEN_RE = re.compile(
    r"^<(?P<name>" + "|".join(_HUNYUAN_TOKEN_NAMES) + r")(?::[^>]+)?>$"
)


def resolve_hunyuan_tokens(tokenizer=None):
    """Map bare token names to their real (possibly suffixed) strings.

    Prefers suffixed forms in the tokenizer vocab; when no tokenizer is threaded
    (this image predates PR #29920's caller plumbing), falls back to the
    launch-provided SGLANG_HUNYUAN_TOKEN_SUFFIX; finally to the bare literal.
    """
    tokens = {}
    vocab = None
    if tokenizer is not None:
        try:
            vocab = tokenizer.get_vocab()
        except Exception as e:
            logger.warning("Failed to read Hunyuan tokenizer vocab: %s", e)
            vocab = None
    if isinstance(vocab, dict):
        for tok in vocab:
            if not isinstance(tok, str):
                continue
            m = _HUNYUAN_TOKEN_RE.match(tok)
            if m:
                tokens[m.group("name")] = tok
    _suffix = _os.environ.get("SGLANG_HUNYUAN_TOKEN_SUFFIX", "")
    for name in _HUNYUAN_TOKEN_NAMES:
        tokens.setdefault(name, "<" + name + _suffix + ">")
    return tokens

'''
    old_init = r'''    def __init__(self):
        super().__init__()

        self.bot_token = "<tool_calls>"
        self.eot_token = "</tool_calls>"

        self.tool_call_start_token = "<tool_call>"
        self.tool_call_end_token = "</tool_call>"
        self.tool_sep_token = "<tool_sep>"

        self.arg_key_start_token = "<arg_key>"
        self.arg_key_end_token = "</arg_key>"
        self.arg_value_start_token = "<arg_value>"
        self.arg_value_end_token = "</arg_value>"

        self.tool_call_regex = re.compile(
            r"<tool_call>(.*?)<tool_sep>(.*?)</tool_call>", re.DOTALL
        )
        self.func_args_regex = re.compile(
            r"<arg_key>(.*?)</arg_key>\s*<arg_value>(.*?)</arg_value>", re.DOTALL
        )'''
    new_init = r'''    def __init__(self, tokenizer=None):
        super().__init__()

        t = resolve_hunyuan_tokens(tokenizer)
        tool_calls = t["tool_calls"]
        tool_call = t["tool_call"]
        tool_sep = t["tool_sep"]
        arg_key = t["arg_key"]
        arg_value = t["arg_value"]

        def _close(open_tok):
            return "</" + open_tok[1:] if open_tok.startswith("<") else open_tok

        self.bot_token = tool_calls
        self.eot_token = _close(tool_calls)
        self.tool_call_start_token = tool_call
        self.tool_call_end_token = _close(tool_call)
        self.tool_sep_token = tool_sep
        self.arg_key_start_token = arg_key
        self.arg_key_end_token = _close(arg_key)
        self.arg_value_start_token = arg_value
        self.arg_value_end_token = _close(arg_value)

        tc_end = _close(tool_call)
        ak_end = _close(arg_key)
        av_end = _close(arg_value)
        self.tool_call_regex = re.compile(
            re.escape(tool_call)
            + r"(.*?)"
            + re.escape(tool_sep)
            + r"(.*?)"
            + re.escape(tc_end),
            re.DOTALL,
        )
        self.func_args_regex = re.compile(
            re.escape(arg_key)
            + r"(.*?)"
            + re.escape(ak_end)
            + r"\s*"
            + re.escape(arg_value)
            + r"(.*?)"
            + re.escape(av_end),
            re.DOTALL,
        )'''
    old_si = r'''        return lambda name: StructureInfo(
            begin=f"<tool_calls>\n<tool_call>{name}<tool_sep>",
            end="</tool_call>\n</tool_calls>",
            trigger="<tool_calls>",
        )'''
    new_si = r'''        return lambda name: StructureInfo(
            begin=f"{self.bot_token}\n{self.tool_call_start_token}{name}{self.tool_sep_token}",
            end=f"{self.tool_call_end_token}\n{self.eot_token}",
            trigger=self.bot_token,
        )'''
    missing = [n for n, s in (("anchor", anchor), ("__init__", old_init), ("structure_info", old_si)) if s not in code]
    if missing:
        print("ANCHOR-DRIFT: hunyuan_detector.py: resolve_hunyuan_tokens backport, missing:", missing, "(SGLang version drift; re-check anchor)")
    else:
        code = code.replace(anchor, anchor + helper, 1)
        code = code.replace(old_init, new_init, 1)
        code = code.replace(old_si, new_si, 1)
        with open(f, "w") as fh:
            fh.write(code)
        print("Patched hunyuan_detector.py: resolve_hunyuan_tokens + suffixed tool-call tokens")
PATCH_HUNYUAN_TOOL_EOF

  # --- 2) parser/reasoning_parser.py: HunyuanDetector reasoning think/tool
  #        tokens resolved via the same backported helper. ---
  #
  # RE-CHECKED 2026-07-16: PR #29920 has LANDED upstream on this image — HunyuanDetector
  # now natively does `t = resolve_hunyuan_tokens(tokenizer)` and builds think_open/
  # think_close/tool_start_token from it (verified in the live source), i.e. exactly what
  # this sub-patch injects. The old "already applied" check only recognized OUR OWN
  # local-alias import (`resolve_hunyuan_tokens as _resolve_hunyuan_tokens`, hence the
  # leading-underscore substring match) and did not recognize upstream's unaliased native
  # call, so it fell through to the (now-stale) `old` anchor and reported a false
  # ANCHOR-DRIFT even though there is nothing left to patch. Fixed to match on the bare
  # function name, same idiom sub-patch (1) above already uses successfully.
  python3 << 'PATCH_HUNYUAN_REASON_EOF'
f = "/usr/local/lib/python3.12/dist-packages/sglang/srt/parser/reasoning_parser.py"
with open(f) as fh:
    code = fh.read()

if "resolve_hunyuan_tokens" in code:
    print("reasoning_parser.py: hunyuan suffix backport already present (native or patched), skipping")
else:
    old = r'''        super().__init__(
            "<think>",
            "</think>",
            force_reasoning=force_reasoning,
            stream_reasoning=stream_reasoning,
            tool_start_token="<tool_calls>",
            continue_final_message=continue_final_message,
            previous_content=previous_content,
        )'''
    new = r'''        # [patch] _sgl_hunyuan_token_suffix_ — backport of SGLang PR #29920
        from sglang.srt.function_call.hunyuan_detector import (
            resolve_hunyuan_tokens as _resolve_hunyuan_tokens,
        )

        _hy = _resolve_hunyuan_tokens()
        _think_open = _hy["think"]
        _think_close = (
            "</" + _think_open[1:] if _think_open.startswith("<") else _think_open
        )
        super().__init__(
            _think_open,
            _think_close,
            force_reasoning=force_reasoning,
            stream_reasoning=stream_reasoning,
            tool_start_token=_hy["tool_calls"],
            continue_final_message=continue_final_message,
            previous_content=previous_content,
        )'''
    if old not in code:
        print("ANCHOR-DRIFT: reasoning_parser.py: HunyuanDetector suffix backport (SGLang version drift; re-check anchor)")
    else:
        code = code.replace(old, new, 1)
        with open(f, "w") as fh:
            fh.write(code)
        print("Patched reasoning_parser.py: HunyuanDetector uses suffixed think/tool tokens")
PATCH_HUNYUAN_REASON_EOF

  # --- 3) models/hunyuan_v3.py: remap shared-expert weight names ---
  # HYV3 checkpoints (vroomfondel/Hy3-NVFP4-W4A4, tencent/Hy3) name the shared
  # expert `model.layers.N.mlp.shared_experts.*`, but SGLang's HYV3 model module
  # is `shared_mlp` (self.shared_mlp = HYV3FeedForward). load_weights has NO remap
  # for it (only router.gate → gate), so the shared-expert weights — which ARE
  # present and FP4-quantized — are silently skipped (`if name not in params_dict:
  # continue`) → shared_mlp stays zero-init → gate_up_proj outputs 0 → down_proj
  # FP4-quantizes a zero input → scale 448·6/0 degenerates → NaN at layer 1, first
  # forward. Localised via --debug-tensor-dump layer tracing (see QUANT_HY3_GOTCHAS).
  # Fix: remap .shared_experts. → .shared_mlp. at the TOP of the load loop, so the
  # existing gate_proj/up_proj → gate_up_proj stacking then applies correctly.
  python3 - <<'PATCH_HUNYUAN_SHARED_EOF'
f = "/usr/local/lib/python3.12/dist-packages/sglang/srt/models/hunyuan_v3.py"
with open(f) as fh:
    code = fh.read()
if 'replace(".shared_experts.", ".shared_mlp.")' in code:
    print("hunyuan_v3.py: shared_experts remap already present, skipping")
else:
    anchor = "        for name, loaded_weight in weights:\n"
    inject = anchor + (
        "            # [dgxarley] HYV3 checkpoints name the shared expert\n"
        "            # `mlp.shared_experts.*`; the SGLang model module is `shared_mlp`.\n"
        "            # Remap so the (real, FP4) shared-expert weights actually load —\n"
        "            # else silently skipped -> shared_mlp zero-init -> NaN at down_proj.\n"
        '            name = name.replace(".shared_experts.", ".shared_mlp.")\n'
    )
    if anchor in code:
        code = code.replace(anchor, inject, 1)
        with open(f, "w") as fh:
            fh.write(code)
        print("Patched hunyuan_v3.py: remap .shared_experts. -> .shared_mlp. in load_weights")
    else:
        print("ANCHOR-DRIFT: hunyuan_v3.py: shared_experts remap (SGLang version drift; re-check anchor)")
PATCH_HUNYUAN_SHARED_EOF
fi

# ── Llama-4 NVFP4 mllama4 loader patches (SGLang 0.5.15 / main gaps) ──────────
# A modelopt-NVFP4 Llama-4 (Llama4ForConditionalGeneration) does NOT load without
# these three fixes to models/mllama4.py. Verified against 0.5.15-sm121: all four
# anchors match and the model then produces coherent text. Upstream-PR-worthy;
# drop them on an image that ships the fixes (grep guards self-noop then).
# Full rationale + re-sync steps: UPSTREAM_SGLANG_LLAMA4_NVFP4_BUG.md.
#   1) _handle_expert_scale_params: NVFP4 ships a per-expert 3D block-scale
#      [num_experts, in_blocks, out] under one (name-less) key. The FP8-era code
#      broadcast a single 2D scale into every expert slot -> forces the whole 3D
#      tensor into each 2D slot -> expand([16,512,5120],[5120,512]) crash. Fix:
#      slice per expert + .T; 0-dim scalars / shared 2D scales keep broadcast.
#   2) permute_qk_weight_for_rotary: the permute view used config.hidden_size
#      (unpacked 5120), but the packed NVFP4 q/k weight last-dim is hidden/2 (2560)
#      -> "shape [8,64,2,5120] invalid for input of size 2621440". Fix: w.shape[-1].
#   3) permute_qk_weight_for_rotary: only .weight was row-permuted for rotary, the
#      per-output-row weight_scale was NOT -> scale<->weight desync -> wrong q/k
#      dequant. Fix: extend both branches to also permute the weight_scale.
# (An optional 4th patch, KV-scale name-remap k_proj.k_scale->attn.k_scale, is
#  quality-only; output stays coherent without it, so it is NOT included here.)
# Serving Llama-4 also needs moe_runner_backend=triton + attention_backend=triton
# in the model profile (flashinfer_cutlass asserts on apply_router_weight_on_input,
# flashinfer is arg-rejected); that is profile config, not a loader patch.
# NO model-name gate: mllama4.py is imported ONLY for the Llama4 arch, so this
# patch is inert for every other model. Applying it unconditionally also covers
# Llama-4 checkpoints served by a local path or a non-matching repo name, and the
# per-anchor guards below keep it idempotent and version-safe.
python3 - <<'PATCH_MLLAMA4_LOADER_EOF'
f = "/usr/local/lib/python3.12/dist-packages/sglang/srt/models/mllama4.py"
s = open(f).read()
edits = [
 ("expert-3d-scale",
"""        else:
            # No expert ID found - this is a single scale for all experts
            # Load the same scale for all experts
            for expert_id in range(num_experts):
                param.data[expert_id] = loaded_weight""",
"""        else:
            if loaded_weight.dim() == 3:
                for expert_id in range(num_experts):
                    param.data[expert_id] = loaded_weight[expert_id].T
            else:
                for expert_id in range(num_experts):
                    param.data[expert_id] = loaded_weight"""),
 ("permute-attn_out",
"            attn_out = self.language_model.config.hidden_size",
"            attn_out = w.shape[-1]"),
 ("permute-k-scale",
'        if ("wk" in modules or "k_proj" in modules) and modules[-1] == "weight":',
'        if ("wk" in modules or "k_proj" in modules) and modules[-1] in ("weight", "weight_scale"):'),
 ("permute-q-scale",
'        elif ("wq" in modules or "q_proj" in modules) and modules[-1] == "weight":',
'        elif ("wq" in modules or "q_proj" in modules) and modules[-1] in ("weight", "weight_scale"):'),
]
for tag, old, new in edits:
    if old in s:
        s = s.replace(old, new, 1); print(f">>> mllama4 patch {tag}: APPLIED")
    elif new.strip() in s:
        print(f">>> mllama4 patch {tag}: already applied")
    else:
        print(f"ANCHOR-DRIFT: mllama4.py: {tag} (SGLang version drift; re-check anchor)")
open(f, "w").write(s)
PATCH_MLLAMA4_LOADER_EOF

# ── Llama-4 NVFP4 KV-scale patches (QUALITY: load the baked FP8 KV scales) ────
# The modelopt-NVFP4 checkpoint bakes FP8 KV scales (...k_proj.k_scale /
# ...v_proj.v_scale). Without these two SGLang never loads them -> falls back to
# scale 1.0 ("less accurate results"). Verified on 0.5.15-sm121: 0 not-loaded
# warnings, and the triton attn backend uses the loaded scale (cache_k.div_(k_scale)).
# NOT a load-blocker (serves without them, just at scale 1.0). Both are needed
# (A alone is a no-op). Same no-gate rationale as the loader patches above:
# llama4.py / mllama4.py are imported only for the Llama4 arch.
#   A) llama4.py: Llama4Attention built RadixAttention() WITHOUT quant_config
#      (unlike llama.py / qwen2.py) -> create_weights never ran -> k_scale/v_scale
#      stayed plain None attrs, never in named_parameters(). Fix: pass quant_config.
#   B) mllama4.py: _handle_scale_remapping returned a bool but never COPIED the
#      remapped scale into the param (llama.py's loader does; mllama4's reimpl
#      dropped the copy). Fix: thread loaded_weight in + _handle_default_weight().
python3 - <<'PATCH_MLLAMA4_KVSCALE_EOF'
def patch(f, edits):
    s = open(f).read()
    for tag, old, new in edits:
        if old in s:
            s = s.replace(old, new, 1); print(f">>> kv patch {tag}: APPLIED")
        elif new.strip() in s:
            print(f">>> kv patch {tag}: already applied")
        else:
            print(f"ANCHOR-DRIFT: llama4/mllama4 kv-scale: {tag} (SGLang version drift; re-check anchor)")
    open(f, "w").write(s)

patch("/usr/local/lib/python3.12/dist-packages/sglang/srt/models/llama4.py", [
 ("A-quant_config",
'''        self.attn = RadixAttention(
            self.num_heads,
            self.head_dim,
            self.scaling,
            num_kv_heads=self.num_kv_heads,
            layer_id=layer_id,
            prefix=add_prefix("attn", prefix),
            use_irope=self.use_rope,
        )''',
'''        self.attn = RadixAttention(
            self.num_heads,
            self.head_dim,
            self.scaling,
            num_kv_heads=self.num_kv_heads,
            layer_id=layer_id,
            quant_config=quant_config,
            prefix=add_prefix("attn", prefix),
            use_irope=self.use_rope,
        )'''),
])
patch("/usr/local/lib/python3.12/dist-packages/sglang/srt/models/mllama4.py", [
 ("B-def",
'''    def _handle_scale_remapping(self, name: str, params_dict: dict) -> bool:
        """Handle scale parameter remapping. Returns True if handled."""
        if "scale" in name and "expert" not in name:
            remapped_name = maybe_remap_kv_scale_name(name, params_dict)
            return remapped_name != name
        return False''',
'''    def _handle_scale_remapping(
        self, name: str, loaded_weight: torch.Tensor, params_dict: dict
    ) -> bool:
        """Handle scale parameter remapping. Returns True if handled."""
        if "scale" in name and "expert" not in name:
            remapped_name = maybe_remap_kv_scale_name(name, params_dict)
            if remapped_name is None:
                return True
            if remapped_name != name:
                self._handle_default_weight(remapped_name, loaded_weight, params_dict)
                return True
            return False
        return False'''),
 ("B-call",
'''            if self._handle_scale_remapping(name, params_dict):''',
'''            if self._handle_scale_remapping(name, loaded_weight, params_dict):'''),
])
PATCH_MLLAMA4_KVSCALE_EOF

# ── HYV3 NEXTN/MTP head fixes — ONLY when EAGLE/MTP speculative decode is on ──
# The built-in NEXTN/MTP head (whole model.layers.80*) is BF16-EXCLUDED from NVFP4
# in hf_quant_config.json (verified: layer 80 = 0 scale tensors, all BF16), but
# SGLang's hunyuan_v3_nextn.py does NOT honour that exclude — it inherits the
# target's modelopt_fp4 quant_config → FusedMoE builds a packed NVFP4 buffer
# (hidden=2048) → boot crash vs the BF16 weight (hidden=4096) in _load_w13.
# --speculative-draft-model-quantization unquant does NOT help (SGLang normalizes
# "unquant"→None → re-auto-detects modelopt_fp4 from the shared checkpoint).
# Two source patches (neither upstream-merged as of 0.5.14). Drop on an image that
# ships PR #30331 + a NEXTN-side quant guard for hunyuan_v3_nextn.py.
if { [[ "$SGLANG_MODEL" == *"Hy3"* ]] || [[ "$SGLANG_MODEL" == *"Hunyuan"* ]]; } \
   && [ "$SGLANG_SPECULATIVE_ENABLED" = "true" ]; then

  # --- 1) hunyuan_v3_nextn.py: force the NEXTN/MTP head UNQUANTIZED (BF16) ---
  # Null quant_config at the HYV3ForCausalLMNextN.__init__ top so it covers the whole
  # draft (decoder layer-80 experts + shared_mlp + attention + lm_head — ALL BF16-
  # excluded). Same guard glm4_moe_nextn.py / qwen3_5_mtp.py already carry; HYV3 lacks it.
  python3 - <<'PATCH_HY3_NEXTN_BF16_EOF'
import pathlib
p = pathlib.Path("/usr/local/lib/python3.12/dist-packages/sglang/srt/models/hunyuan_v3_nextn.py")
if not p.exists():
    print("ANCHOR-DRIFT: hunyuan_v3_nextn.py: BF16-head override target file missing (SGLang restructured/renamed?)")
else:
    src = p.read_text()
    marker = "# [patch] _sgl_hy3_nextn_bf16_head_"
    anchor = (
        "        nn.Module.__init__(self)\n"
        "        self.config = config\n"
        "        self.quant_config = quant_config\n"
    )
    inject = (
        "        nn.Module.__init__(self)\n"
        "        self.config = config\n"
        "        " + marker + "\n"
        "        # layer-80 (NEXTN/MTP head) is BF16-excluded in hf_quant_config.json;\n"
        "        # drop the target's NVFP4 quant so create_weights allocates BF16 buffers.\n"
        "        if quant_config is not None and quant_config.get_name() in (\n"
        "            \"modelopt_fp4\",\n"
        "            \"modelopt_mixed\",\n"
        "        ):\n"
        "            quant_config = None\n"
        "        self.quant_config = quant_config\n"
    )
    if marker in src:
        print("hunyuan_v3_nextn.py: BF16-head override already patched, skipping")
    elif anchor not in src:
        print("ANCHOR-DRIFT: hunyuan_v3_nextn.py: BF16-head override (SGLang version drift; re-check anchor)")
    else:
        p.write_text(src.replace(anchor, inject, 1))
        print("Patched hunyuan_v3_nextn.py: NEXTN/MTP head forced unquantized (BF16)")
PATCH_HY3_NEXTN_BF16_EOF

  # --- 2) hunyuan_v3_nextn.py load_weights: remap the draft head's output norm ---
  # Checkpoint stores it as model.layers.80.final_layernorm.weight; the module is
  # model.shared_head.norm. Without this it falls into the generic else →
  # model.decoder.final_layernorm.weight (no such param) → silently dropped →
  # shared_head.norm stays default-init → accept-rate collapses. Upstream PR #30331.
  python3 - <<'PATCH_HY3_NEXTN_FINALNORM_EOF'
import pathlib
p = pathlib.Path("/usr/local/lib/python3.12/dist-packages/sglang/srt/models/hunyuan_v3_nextn.py")
if not p.exists():
    print("ANCHOR-DRIFT: hunyuan_v3_nextn.py: final_layernorm remap target file missing (SGLang restructured/renamed?)")
else:
    src = p.read_text()
    marker = "# [patch] _sgl_hy3_nextn_final_layernorm_"
    anchor = (
        "                if any(subname.startswith(s) for s in spec_weight_names):\n"
        "                    name = f\"model.{subname}\"\n"
        "                else:\n"
        "                    name = f\"model.decoder.{subname}\"\n"
    )
    inject = (
        "                if any(subname.startswith(s) for s in spec_weight_names):\n"
        "                    name = f\"model.{subname}\"\n"
        "                elif subname.startswith(\"final_layernorm\"):\n"
        "                    " + marker + "  # upstream PR #30331\n"
        "                    name = \"model.shared_head.norm.weight\"\n"
        "                else:\n"
        "                    name = f\"model.decoder.{subname}\"\n"
    )
    if marker in src:
        print("hunyuan_v3_nextn.py: final_layernorm remap already patched, skipping")
    elif anchor not in src:
        print("ANCHOR-DRIFT: hunyuan_v3_nextn.py: final_layernorm remap (SGLang version drift; re-check anchor)")
    else:
        p.write_text(src.replace(anchor, inject, 1))
        print("Patched hunyuan_v3_nextn.py: final_layernorm -> shared_head.norm (PR #30331)")
PATCH_HY3_NEXTN_FINALNORM_EOF
fi

# ── DeepSeek/GLM NEXTN: honour the checkpoint's per-module NVFP4 exclude on the
#    built-in MTP head (upstream blindly nulls the whole quant_config) ───────────
# GlmMoeDsaForCausalLM (0xSero/glm-5.2-reap-504B-v2) and plain DeepSeek-V3 route
# their built-in NEXTN/MTP head through deepseek_nextn.py, which UNCONDITIONALLY
# nulls a modelopt_fp4 quant_config for the ENTIRE MTP decoder layer. Correct when
# the whole MTP layer is BF16 (normal DeepSeek-V3), WRONG for this REAP export:
# its hf_quant_config 'ignore' keeps only attn/eh_proj/gate/shared_experts of
# layer N BF16 while the ROUTED EXPERTS stay NVFP4. Blindly nulling -> FusedMoE
# builds a BF16 (unpacked) w13 buffer -> the NVFP4-packed checkpoint tensor
# mismatches in _load_w13 ("size of tensor a (6144) must match b (3072)" = hidden
# 6144 unpacked vs 3072 packed, 2 fp4/byte) -> head crash-loop. The nextn decoder
# is built under the 'model.decoder.*' prefix, which never matches the checkpoint's
# 'model.layers.N.*' exclude entries, so is_layer_excluded can't discriminate per
# submodule on its own. Fix: when the checkpoint actually leaves the MTP experts
# quantized, KEEP the fp4 config and add 'model.decoder.*' aliases of the layer-N
# excludes -> attn/gate/shared build BF16, experts build NVFP4, exactly the mix the
# main model uses (only touches build-time quant choice, load path unchanged).
# Self-gated (is_layer_excluded + non-empty aliases) -> inert for all-BF16-MTP
# checkpoints, so it falls back to the upstream blanket null. Note:
# --speculative-draft-model-quantization=unquant does NOT reach this path
# (deepseek_nextn.py never reads that flag; only glm4_moe_nextn / qwen3_*_mtp do).
if [ "$SGLANG_SPECULATIVE_ENABLED" = "true" ]; then
  python3 - <<'PATCH_DSNEXTN_MIXED_MTP_EOF'
import pathlib
p = pathlib.Path("/usr/local/lib/python3.12/dist-packages/sglang/srt/models/deepseek_nextn.py")
if not p.exists():
    print("ANCHOR-DRIFT: deepseek_nextn.py: mixed-MTP quant override target file missing (SGLang restructured/renamed?)")
else:
    src = p.read_text()
    marker = "# [patch] _sgl_dsnextn_mixed_mtp_"
    anchor = (
        '        if quant_config is not None and quant_config.get_name() == "modelopt_fp4":\n'
        '            logger.warning(\n'
        '                "Overriding DeepseekV3ForCausalLMNextN quant config for modelopt_fp4 Deepseek model."\n'
        '            )\n'
        '            quant_config = None\n'
    )
    inject = (
        '        ' + marker + '  # honour per-module NVFP4 exclude on the MTP head\n'
        '        _dsnextn_kept_fp4 = False\n'
        '        if quant_config is not None and quant_config.get_name() == "modelopt_fp4":\n'
        '            _mtp_experts = f"model.layers.{config.num_hidden_layers}.mlp.experts"\n'
        '            _excl = getattr(quant_config, "exclude_modules", None)\n'
        '            _tag = f".layers.{config.num_hidden_layers}."\n'
        '            if (\n'
        '                isinstance(_excl, list)\n'
        '                and hasattr(quant_config, "is_layer_excluded")\n'
        '                and not quant_config.is_layer_excluded(_mtp_experts)\n'
        '            ):\n'
        '                _aliases = [\n'
        '                    e.replace(_tag, ".decoder.")\n'
        '                    for e in _excl\n'
        '                    if _tag in e and e.replace(_tag, ".decoder.") not in _excl\n'
        '                ]\n'
        '                if _aliases:\n'
        '                    quant_config.exclude_modules = _excl + _aliases\n'
        '                    _dsnextn_kept_fp4 = True\n'
        '                    logger.warning(\n'
        '                        "NextN modelopt_fp4: checkpoint keeps the MTP experts quantized; "\n'
        '                        "aliasing %d layer-%d excludes to model.decoder.* so attn/gate/shared "\n'
        '                        "stay BF16 while experts stay NVFP4.",\n'
        '                        len(_aliases),\n'
        '                        config.num_hidden_layers,\n'
        '                    )\n'
        '            if not _dsnextn_kept_fp4:\n'
        '                logger.warning(\n'
        '                    "Overriding DeepseekV3ForCausalLMNextN quant config for modelopt_fp4 Deepseek model."\n'
        '                )\n'
        '                quant_config = None\n'
    )
    if marker in src:
        print("deepseek_nextn.py: mixed-MTP quant override already patched, skipping")
    elif anchor not in src:
        print("ANCHOR-DRIFT: deepseek_nextn.py: mixed-MTP quant override (SGLang version drift; re-check anchor)")
    else:
        p.write_text(src.replace(anchor, inject, 1))
        print("Patched deepseek_nextn.py: NEXTN/MTP honours per-module NVFP4 exclude (mixed-precision MTP head)")
PATCH_DSNEXTN_MIXED_MTP_EOF
fi

# ── DSA paged-MQA-logits TORCH FALLBACK (GB10/SM121: DeepGEMM + CuteDSL both hard NO-GO) ──
# GlmMoeDsa / DeepSeek-V3.2-family DSA models score paged KV via a paged-MQA-logits kernel
# (sglang/srt/layers/attention/dsa/dsa_indexer.py::_get_topk_paged) before top-k selection.
# On GB10/SM121, BOTH hardware kernel routes are dead ends, not just unconfigured:
#   - DeepGEMM (the default): deep_gemm.get_paged_mqa_logits_metadata() throws a compiled
#     C++ "Unsupported architecture" assert on SM121. Upstream declined SM120/SM121 support
#     (DeepGEMM PR #318, maintainer cited lack of hardware/capacity) -- not a local gate.
#   - cutedsl: gated behind is_sm100_supported()==False on GB10, and even bypassing that
#     gate, the kernel's _setup_mma uses tcgen05.MmaF8F6F4Op, a datacenter-Blackwell-only
#     (SM100/SM103) tensor-core instruction that does not exist on consumer Blackwell
#     (SM121) -- a real ISA boundary, verified 2026-07-16 in a GPU debug pod.
# This crashes the TARGET model's every decode step under attention_backend="dsa" (not just
# the MTP/NEXTN draft, whose is_nextn indexer always computes topk_indices and needs this
# kernel too). Full investigation + design: DSA_speedup.md, dsalogitrework.md (repo root).
#
# Fix: port SGLang's own dsv4-side answer to this exact problem --
# fp8_paged_mqa_logits_torch_sm120 (dsv4/indexer.py, upstream PR #24692, merged 2026-06-01,
# in this image since v0.5.13) -- into the generic dsa/dsa_indexer.py path GlmMoeDsa uses,
# which has no equivalent. The torch path discards the DeepGEMM schedule metadata
# (`_ = deep_gemm_metadata`), so all 3 eager DeepGEMM-metadata call sites (2 in
# dsa_backend.py::init_forward_metadata + the shared _refresh_paged_mqa_schedule_metadata
# cuda-graph-replay helper, which alone covers 2 more replay call sites) can simply be
# skipped, not just the logits kernel itself -- verified live (numeric unit tests: dominant-
# KV-slot test matches a hand-derived reference exactly, masking correct, no NaN/all-zero/
# constant-across-seeds -- the failure signature a community SM120 fallback fork silently
# hit). New backend value "torch", OPT-IN ONLY via --dsa-paged-mqa-logits-backend torch (NOT
# selected by "auto", so archs where DeepGEMM/CuteDSL already work are unaffected).
#
# PHASE 1 ONLY (dsalogitrework.md Section 3): plain decode, next_n==1 -- unblocks ordinary
# target-model decode AND each individual MTP/NEXTN draft step (the draft decodes one token
# at a time). Phase 2 (target-verify / next_n>=2 multi-token batches) is NOT implemented;
# the dispatch raises NotImplementedError there rather than silently mis-routing.
#
# 5 files patched/written, unconditionally (no model-name gate): paged_mqa_logits_backend.py
# and server_args.py only add an opt-in enum value / CLI choice (zero behavior change unless
# explicitly selected); dsa_backend.py / dsa_indexer.py / the new torch_paged_mqa_logits.py
# are DSA-specific source files exercised only by actual DSA models (mirrors the mllama4.py
# precedent above: the files are inert for every other model by construction).

python3 - <<'PATCH_DSA_TORCH_ENUM_EOF'
import pathlib

# --- File 1: paged_mqa_logits_backend.py ---
f1 = pathlib.Path("/usr/local/lib/python3.12/dist-packages/sglang/srt/layers/attention/dsa/paged_mqa_logits_backend.py")
src1 = f1.read_text()
marker1 = "# [patch] _sgl_dsa_torch_fallback_enum_"
old1 = '''class DSAPagedMQALogitsBackend(Enum):
    DEEPGEMM = "deepgemm"
    CUTEDSL = "cutedsl"
    AITER = "aiter"

    def is_deepgemm(self) -> bool:
        return self == DSAPagedMQALogitsBackend.DEEPGEMM

    def is_cutedsl(self) -> bool:
        return self == DSAPagedMQALogitsBackend.CUTEDSL

    def is_aiter(self) -> bool:
        return self == DSAPagedMQALogitsBackend.AITER'''
new1 = f'''{marker1}
class DSAPagedMQALogitsBackend(Enum):
    DEEPGEMM = "deepgemm"
    CUTEDSL = "cutedsl"
    AITER = "aiter"
    TORCH = "torch"  # pure-torch fallback for archs DeepGEMM/CuteDSL don't cover (e.g. SM121/GB10)

    def is_deepgemm(self) -> bool:
        return self == DSAPagedMQALogitsBackend.DEEPGEMM

    def is_cutedsl(self) -> bool:
        return self == DSAPagedMQALogitsBackend.CUTEDSL

    def is_aiter(self) -> bool:
        return self == DSAPagedMQALogitsBackend.AITER

    def is_torch(self) -> bool:
        return self == DSAPagedMQALogitsBackend.TORCH'''
old1b = '''        if value == "auto" or value == "deepgemm":
            return DSAPagedMQALogitsBackend.DEEPGEMM
        if value == "aiter":
            raise ValueError("dsa_paged_mqa_logits_backend='aiter' requires ROCm.")
        if value == "cutedsl":
            if not is_sm100_supported():
                raise ValueError(
                    "dsa_paged_mqa_logits_backend='cutedsl' requires SM100 (Blackwell)."
                )
            return DSAPagedMQALogitsBackend.CUTEDSL
        raise ValueError(f"Unknown dsa_paged_mqa_logits_backend: {value!r}")'''
new1b = '''        if value == "auto" or value == "deepgemm":
            return DSAPagedMQALogitsBackend.DEEPGEMM
        if value == "aiter":
            raise ValueError("dsa_paged_mqa_logits_backend='aiter' requires ROCm.")
        if value == "cutedsl":
            if not is_sm100_supported():
                raise ValueError(
                    "dsa_paged_mqa_logits_backend='cutedsl' requires SM100 (Blackwell)."
                )
            return DSAPagedMQALogitsBackend.CUTEDSL
        if value == "torch":
            # No arch gate: that is the whole point of this backend. NOT selected by
            # "auto" (opt-in only) to avoid silently regressing perf on archs where
            # DeepGEMM/CuteDSL already work (see dsalogitrework.md Section 4.1).
            return DSAPagedMQALogitsBackend.TORCH
        raise ValueError(f"Unknown dsa_paged_mqa_logits_backend: {value!r}")'''

if marker1 in src1:
    print("paged_mqa_logits_backend.py: torch fallback enum already patched, skipping")
elif old1 not in src1:
    print("ANCHOR-DRIFT: paged_mqa_logits_backend.py: DSAPagedMQALogitsBackend enum anchor missing (SGLang version drift; re-check anchor)")
elif old1b not in src1:
    print("ANCHOR-DRIFT: paged_mqa_logits_backend.py: resolve() anchor missing (SGLang version drift; re-check anchor)")
else:
    src1 = src1.replace(old1, new1, 1)
    src1 = src1.replace(old1b, new1b, 1)
    f1.write_text(src1)
    print("Patched paged_mqa_logits_backend.py: added TORCH backend + resolve('torch')")
PATCH_DSA_TORCH_ENUM_EOF

python3 - <<'PATCH_DSA_TORCH_SERVERARGS_EOF'
import pathlib

f2 = pathlib.Path("/usr/local/lib/python3.12/dist-packages/sglang/srt/server_args.py")
src2 = f2.read_text()
marker2 = "# [patch] _sgl_dsa_torch_fallback_choice_"
old2a = 'DSA_PAGED_MQA_LOGITS_BACKEND_CHOICES = ["auto", "deepgemm", "cutedsl", "aiter"]'
new2a = (marker2 + '\n'
          'DSA_PAGED_MQA_LOGITS_BACKEND_CHOICES = ["auto", "deepgemm", "cutedsl", "aiter", "torch"]')
old2b = '''            help="DSA indexer paged MQA logits kernel backend. Options: 'auto' (default; DeepGEMM on CUDA, aiter on ROCm), 'deepgemm', 'cutedsl' (CuTe DSL kernel, SM 100 (Blackwell) only; wins at low batch size and long context), 'aiter' (ROCm only).",'''
new2b = '''            help="DSA indexer paged MQA logits kernel backend. Options: 'auto' (default; DeepGEMM on CUDA, aiter on ROCm), 'deepgemm', 'cutedsl' (CuTe DSL kernel, SM 100 (Blackwell) only; wins at low batch size and long context), 'aiter' (ROCm only), 'torch' (pure-torch fallback, any CUDA arch, e.g. SM120/SM121 where neither DeepGEMM nor CuteDSL run; slower, opt-in only).",'''

if marker2 in src2:
    print("server_args.py: dsa torch-backend choice already patched, skipping")
elif old2a not in src2:
    print("ANCHOR-DRIFT: server_args.py: DSA_PAGED_MQA_LOGITS_BACKEND_CHOICES anchor missing (SGLang version drift; re-check anchor)")
elif old2b not in src2:
    print("ANCHOR-DRIFT: server_args.py: dsa_paged_mqa_logits_backend help-string anchor missing (SGLang version drift; re-check anchor)")
else:
    src2 = src2.replace(old2a, new2a, 1)
    src2 = src2.replace(old2b, new2b, 1)
    f2.write_text(src2)
    print("Patched server_args.py: added 'torch' to DSA_PAGED_MQA_LOGITS_BACKEND_CHOICES")
PATCH_DSA_TORCH_SERVERARGS_EOF

python3 - <<'PATCH_DSA_TORCH_NEWFILE_EOF'
import pathlib
f3 = pathlib.Path("/usr/local/lib/python3.12/dist-packages/sglang/srt/layers/attention/dsa/torch_paged_mqa_logits.py")
marker3 = "fp8_paged_mqa_logits_torch_dsa"
new3 = '# SPDX-License-Identifier: Apache-2.0\n"""\nDSA paged-MQA-logits pure-torch fallback (Phase 1: plain decode, next_n == 1).\n\nPorted from sglang.srt.layers.attention.dsv4.indexer.fp8_paged_mqa_logits_torch_sm120\n(upstream SGLang PR #24692, merged 2026-06-01, first shipped v0.5.13) for the generic\nDSA indexer path (sglang.srt.layers.attention.dsa.dsa_indexer, used by GlmMoeDsa /\nDeepSeek-V3.2-family models), which has no equivalent fallback upstream. Copied rather\nthan imported cross-module: dsv4 and dsa are independent code paths in SGLang, and a\nshared import would create an unwanted coupling.\n\nWhy this exists: on GB10/SM121 (consumer Blackwell), deep_gemm.get_paged_mqa_logits_metadata\n/ fp8_paged_mqa_logits throw a compiled C++ "Unsupported architecture" assert (DeepGEMM\nupstream declined SM120/SM121 support, PR #318), and the CuteDSL alternative fails on a\nreal ISA boundary (tcgen05 MMA is datacenter-Blackwell-only, SM100/SM103). See\nDSA_speedup.md and dsalogitrework.md (repo root) for the full investigation.\n\nPhase 1 scope: plain decode only (next_n == 1), matching the call shape of\nsglang.jit_kernel.dsa.paged_mqa_logits.deepgemm_paged_mqa_logits_split. Phase 2\n(target-verify / next_n >= 2) is NOT implemented here; see dsalogitrework.md Section 3.\n\nCORRECTNESS WARNING (dsalogitrework.md Section 5): a community fork (kt-sglang) shipped\na similar-looking torch fallback for SM120 that ran without error but returned WRONG\nresults (all-zero/NaN logits). "Runs without crashing" is not sufficient verification of\nthis function — see the numeric checks required in dsalogitrework.md Section 5 before\ntrusting output from this path in production.\n"""\n\nfrom __future__ import annotations\n\nfrom typing import Any\n\nimport torch\nimport torch.nn.functional as F\n\nfrom sglang.srt.layers.quantization.fp8_kernel import is_fp8_fnuz\n\nFP8_DTYPE = torch.float8_e4m3fnuz if is_fp8_fnuz() else torch.float8_e4m3fn\n\n\ndef fp8_paged_mqa_logits_torch_dsa(\n    q_fp8: torch.Tensor,\n    kvcache_fp8: torch.Tensor,\n    weight: torch.Tensor,\n    seq_lens: torch.Tensor,\n    page_table: torch.Tensor,\n    deep_gemm_metadata: Any,\n    max_seq_len: int,\n    clean_logits: bool = True,\n) -> torch.Tensor:\n    """CUDA-graph-compatible FP8 paged MQA logits, pure torch (no DeepGEMM/CuteDSL).\n\n    Verbatim port of dsv4.indexer.fp8_paged_mqa_logits_torch_sm120 (vectorized,\n    no `.item()` / no data-dependent control flow -> CUDA-graph-capture-safe).\n    `deep_gemm_metadata` is accepted for call-site signature compatibility but\n    unused: this path does no SM-tiled scheduling (unlike DeepGEMM\'s kernel), so\n    it has no notion of a schedule to consume. Callers may pass None.\n    """\n    _ = deep_gemm_metadata\n    batch_size, _, num_heads, head_dim = q_fp8.shape\n    block_size = kvcache_fp8.shape[1]\n    device = q_fp8.device\n\n    assert head_dim == 128, "Vectorized torch impl hardcodes DSA indexer head_dim=128"\n    assert (\n        block_size == 64\n    ), "Vectorized torch impl hardcodes block_size=64 cache layout"\n    assert q_fp8.shape == (batch_size, 1, num_heads, head_dim)\n    assert kvcache_fp8.shape[1:] == (block_size, 1, head_dim + 4)\n    assert weight.shape == (batch_size, num_heads)\n    if seq_lens.dim() > 1:\n        seq_lens = seq_lens.squeeze(-1)\n    assert seq_lens.shape == (batch_size,)\n    assert page_table.shape[0] == batch_size\n    assert clean_logits == False\n\n    max_pages = (max_seq_len + block_size - 1) // block_size\n    max_padded_seq = max_pages * block_size\n\n    kvcache_flat = kvcache_fp8.view(-1, block_size * (head_dim + 4))\n    SCALE_OFFSET = block_size * head_dim\n\n    page_ids = page_table[:, :max_pages]\n    kvcache_gathered = kvcache_flat[page_ids]\n\n    kv_value_raw = kvcache_gathered[..., :SCALE_OFFSET]\n    kv_scale_raw = kvcache_gathered[..., SCALE_OFFSET:]\n\n    kv_value = kv_value_raw.contiguous().view(dtype=FP8_DTYPE).to(torch.float32)\n    kv_value = kv_value.view(batch_size, max_padded_seq, head_dim)\n\n    kv_scale = kv_scale_raw.contiguous().view(dtype=torch.float32)\n    kv_scale = kv_scale.view(batch_size, max_padded_seq)\n\n    q = q_fp8[:, 0].to(torch.float32)\n\n    score = torch.bmm(kv_value, q.transpose(1, 2))\n\n    score = F.relu(score)\n    score = score * weight.unsqueeze(1)\n    score = score.sum(dim=2)\n\n    score = score * kv_scale\n\n    out_width = min(max_padded_seq, max_seq_len)\n    logits = score.new_full((batch_size, max_seq_len), float("-inf"))\n    logits[:, :out_width] = score[:, :out_width]\n\n    positions = torch.arange(max_seq_len, device=device)\n    invalid_mask = positions.unsqueeze(0) >= seq_lens.unsqueeze(1)\n    logits.masked_fill_(invalid_mask, float("-inf"))\n\n    return logits\n'

if f3.exists() and marker3 in f3.read_text():
    print("dsa/torch_paged_mqa_logits.py: already written, skipping")
else:
    f3.write_text(new3)
    print("Wrote sglang/srt/layers/attention/dsa/torch_paged_mqa_logits.py: fp8_paged_mqa_logits_torch_dsa (phase 1)")
PATCH_DSA_TORCH_NEWFILE_EOF

python3 - <<'PATCH_DSA_TORCH_BACKEND_EOF'
import pathlib

f4 = pathlib.Path("/usr/local/lib/python3.12/dist-packages/sglang/srt/layers/attention/dsa_backend.py")
src4 = f4.read_text()
marker4 = "# [patch] _sgl_dsa_torch_fallback_backend_"

# --- 4a: import DSAPagedMQALogitsBackend ---
old4a = '''from sglang.srt.layers.attention.dsa.dsa_indexer import BaseIndexerMetadata
from sglang.srt.layers.attention.dsa.dsa_topk_backend import (
    DSATopKBackend,
    TopkTransformMethod,
)'''
new4a = f'''{marker4}
from sglang.srt.layers.attention.dsa.dsa_indexer import BaseIndexerMetadata
from sglang.srt.layers.attention.dsa.dsa_topk_backend import (
    DSATopKBackend,
    TopkTransformMethod,
)
from sglang.srt.layers.attention.dsa.paged_mqa_logits_backend import (
    DSAPagedMQALogitsBackend,
)'''

# --- 4b: resolve self.paged_mqa_logits_backend in __init__ ---
old4b = '''        self.dsa_topk_backend: DSATopKBackend = DSATopKBackend(
            model_runner.server_args.dsa_topk_backend
        )'''
new4b = '''        self.dsa_topk_backend: DSATopKBackend = DSATopKBackend(
            model_runner.server_args.dsa_topk_backend
        )
        # Independent resolve mirroring dsa_indexer.py's Indexer.__init__ (both must agree
        # on the backend so the eager metadata precompute here and the indexer's dispatch
        # don't disagree about whether DeepGEMM is being used).
        self.paged_mqa_logits_backend = DSAPagedMQALogitsBackend.resolve(
            model_runner.server_args.dsa_paged_mqa_logits_backend
        )'''

# --- 4c: gate init_forward_metadata call site A (~953-991, forward_batch.* variant) ---
old4c = '''        paged_mqa_schedule_metadata = None
        paged_mqa_ctx_lens_2d = None
        if is_cuda() and (
            forward_batch.forward_mode.is_decode_or_idle()
            or forward_batch.forward_mode.is_target_verify()
            or forward_batch.forward_mode.is_draft_extend_v2()
        ):
            paged_mqa_ctx_lens_2d = self._build_paged_mqa_schedule_2d_ctx_lens(
                forward_batch.forward_mode,
                cache_seqlens_int32,
                seqlens_expanded,
                forward_batch.batch_size,
            )
            # NOTE: block_kv arg must be 64 here — DG computes SPLIT_KV =
            # block_kv * 4 and both DG's and the indexer's compute kernels
            # require SPLIT_KV = 256; this is independent of the cache page size.
            paged_mqa_schedule_metadata = deep_gemm.get_paged_mqa_logits_metadata(
                paged_mqa_ctx_lens_2d, 64, deep_gemm.get_num_sms()
            )'''
new4c = '''        paged_mqa_schedule_metadata = None
        paged_mqa_ctx_lens_2d = None
        if is_cuda() and (
            forward_batch.forward_mode.is_decode_or_idle()
            or forward_batch.forward_mode.is_target_verify()
            or forward_batch.forward_mode.is_draft_extend_v2()
        ):
            paged_mqa_ctx_lens_2d = self._build_paged_mqa_schedule_2d_ctx_lens(
                forward_batch.forward_mode,
                cache_seqlens_int32,
                seqlens_expanded,
                forward_batch.batch_size,
            )
            # ctx_lens_2d is still needed unconditionally (consumed downstream as
            # seqlens_32_2d regardless of logits backend); only the DeepGEMM schedule
            # metadata call is skipped for the torch backend, which discards it anyway
            # (dsalogitrework.md Section 2: `_ = deep_gemm_metadata` in the torch fn).
            if not self.paged_mqa_logits_backend.is_torch():
                # NOTE: block_kv arg must be 64 here — DG computes SPLIT_KV =
                # block_kv * 4 and both DG's and the indexer's compute kernels
                # require SPLIT_KV = 256; this is independent of the cache page size.
                paged_mqa_schedule_metadata = deep_gemm.get_paged_mqa_logits_metadata(
                    paged_mqa_ctx_lens_2d, 64, deep_gemm.get_num_sms()
                )'''

# --- 4d: gate init_forward_metadata call site B (~1297-1320, forward_mode.* variant) ---
old4d = '''        paged_mqa_schedule_metadata = None
        paged_mqa_ctx_lens_2d = None
        if is_cuda() and (
            forward_mode.is_decode_or_idle()
            or forward_mode.is_target_verify()
            or forward_mode.is_draft_extend_v2()
        ):
            paged_mqa_ctx_lens_2d = self._build_paged_mqa_schedule_2d_ctx_lens(
                forward_mode, cache_seqlens_int32, seqlens_expanded, bs
            )
            paged_mqa_schedule_metadata = deep_gemm.get_paged_mqa_logits_metadata(
                paged_mqa_ctx_lens_2d, 64, deep_gemm.get_num_sms()
            )'''
new4d = '''        paged_mqa_schedule_metadata = None
        paged_mqa_ctx_lens_2d = None
        if is_cuda() and (
            forward_mode.is_decode_or_idle()
            or forward_mode.is_target_verify()
            or forward_mode.is_draft_extend_v2()
        ):
            paged_mqa_ctx_lens_2d = self._build_paged_mqa_schedule_2d_ctx_lens(
                forward_mode, cache_seqlens_int32, seqlens_expanded, bs
            )
            if not self.paged_mqa_logits_backend.is_torch():
                paged_mqa_schedule_metadata = deep_gemm.get_paged_mqa_logits_metadata(
                    paged_mqa_ctx_lens_2d, 64, deep_gemm.get_num_sms()
                )'''

# --- 4e: gate the shared graph-replay refresh helper (covers both replay call sites) ---
old4e = '''    def _refresh_paged_mqa_schedule_metadata(
        self,
        metadata: DSAMetadata,
        seqlens_32_2d: torch.Tensor,
    ) -> None:
        new_schedule = deep_gemm.get_paged_mqa_logits_metadata(
            seqlens_32_2d, 64, deep_gemm.get_num_sms()
        )
        if metadata.paged_mqa_schedule_metadata is None:
            object.__setattr__(metadata, "paged_mqa_schedule_metadata", new_schedule)
        else:
            metadata.paged_mqa_schedule_metadata.copy_(new_schedule)'''
new4e = '''    def _refresh_paged_mqa_schedule_metadata(
        self,
        metadata: DSAMetadata,
        seqlens_32_2d: torch.Tensor,
    ) -> None:
        # Torch backend: schedule metadata is unused (discarded by the torch fn) and
        # was never allocated (init_forward_metadata skips it too) -> nothing to refresh.
        # This single helper covers BOTH cuda-graph-replay refresh call sites, so gating
        # it here is sufficient without touching each call site separately.
        if self.paged_mqa_logits_backend.is_torch():
            return
        new_schedule = deep_gemm.get_paged_mqa_logits_metadata(
            seqlens_32_2d, 64, deep_gemm.get_num_sms()
        )
        if metadata.paged_mqa_schedule_metadata is None:
            object.__setattr__(metadata, "paged_mqa_schedule_metadata", new_schedule)
        else:
            metadata.paged_mqa_schedule_metadata.copy_(new_schedule)'''

edits = [("4a-import", old4a, new4a), ("4b-init", old4b, new4b), ("4c-site-A", old4c, new4c),
         ("4d-site-B", old4d, new4d), ("4e-refresh-helper", old4e, new4e)]

if marker4 in src4:
    print("dsa_backend.py: torch fallback wiring already patched, skipping")
else:
    missing = [tag for tag, old, new in edits if old not in src4]
    if missing:
        for tag in missing:
            print(f"ANCHOR-DRIFT: dsa_backend.py: torch fallback wiring anchor '{tag}' missing (SGLang version drift; re-check anchor)")
    else:
        for tag, old, new in edits:
            src4 = src4.replace(old, new, 1)
        f4.write_text(src4)
        print("Patched dsa_backend.py: torch backend resolve + gated 2 init_forward_metadata sites + graph-replay refresh helper")
PATCH_DSA_TORCH_BACKEND_EOF

python3 - <<'PATCH_DSA_TORCH_INDEXER_EOF'
import pathlib

f5 = pathlib.Path("/usr/local/lib/python3.12/dist-packages/sglang/srt/layers/attention/dsa/dsa_indexer.py")
src5 = f5.read_text()
marker5 = "# [patch] _sgl_dsa_torch_fallback_dispatch_"

# --- 5a: import the new function ---
old5a = '''from sglang.srt.layers.attention.dsa.paged_mqa_logits_backend import (
    DSAPagedMQALogitsBackend,
)'''
new5a = f'''{marker5}
from sglang.srt.layers.attention.dsa.paged_mqa_logits_backend import (
    DSAPagedMQALogitsBackend,
)
from sglang.srt.layers.attention.dsa.torch_paged_mqa_logits import (
    fp8_paged_mqa_logits_torch_dsa,
)'''

# --- 5b: use_dg_native must exclude torch backend (else next_n>=2 target-verify with
#     TORCH selected would still route to DeepGEMM's native path and crash on SM121). ---
old5b = '''        use_dg_native = (
            not use_cute_dsl
            and _is_cuda
            and forward_batch.forward_mode.is_target_verify()
            and next_n >= 2
            and ctx_2d is not None
            and ctx_2d.shape == (B, next_n)
        )'''
new5b = '''        use_dg_native = (
            not use_cute_dsl
            and not self.paged_mqa_logits_backend.is_torch()
            and _is_cuda
            and forward_batch.forward_mode.is_target_verify()
            and next_n >= 2
            and ctx_2d is not None
            and ctx_2d.shape == (B, next_n)
        )'''

# --- 5c: skip the DeepGEMM metadata fallback call when torch backend (metadata unused) ---
old5c = '''        if _is_cuda:
            if schedule_metadata is None:
                schedule_metadata = deep_gemm.get_paged_mqa_logits_metadata(
                    seqlens_32_2d, blocksize, self.sm_count
                )'''
new5c = '''        if _is_cuda and not self.paged_mqa_logits_backend.is_torch():
            if schedule_metadata is None:
                schedule_metadata = deep_gemm.get_paged_mqa_logits_metadata(
                    seqlens_32_2d, blocksize, self.sm_count
                )'''

# --- 5d: new dispatch branch, inserted before the final `else` (deepgemm_paged_mqa_logits_split) ---
old5d = '''        elif use_dg_native:
            logits = deepgemm_paged_mqa_logits_native(
                deep_gemm.fp8_paged_mqa_logits,
                q_fp8,
                kv_cache_fp8,
                weights,
                seqlens_32_2d,
                block_tables,
                schedule_metadata,
                max_seq_len,
                q_offset=q_offset,
                B=B,
                next_n=next_n,
            )
        else:
            logits = deepgemm_paged_mqa_logits_split(
                deep_gemm.fp8_paged_mqa_logits,
                q_fp8,
                kv_cache_fp8,
                weights,
                seqlens_32_2d,
                block_tables,
                schedule_metadata,
                max_seq_len,
                q_offset=q_offset,
            )'''
new5d = '''        elif use_dg_native:
            logits = deepgemm_paged_mqa_logits_native(
                deep_gemm.fp8_paged_mqa_logits,
                q_fp8,
                kv_cache_fp8,
                weights,
                seqlens_32_2d,
                block_tables,
                schedule_metadata,
                max_seq_len,
                q_offset=q_offset,
                B=B,
                next_n=next_n,
            )
        elif self.paged_mqa_logits_backend.is_torch():
            # Phase 1 (dsalogitrework.md): plain decode only (next_n==1), the shape
            # deepgemm_paged_mqa_logits_split below handles. next_n>=2 is the
            # target-verify multi-token batch (Phase 2, not implemented) -- raise
            # clearly instead of silently mis-routing into a wrong-shape call.
            if next_n >= 2:
                raise NotImplementedError(
                    "dsa_paged_mqa_logits_backend='torch' (phase 1) only supports "
                    "plain single-token decode (next_n==1); target-verify/multi-token "
                    "batches (next_n>=2) are not yet implemented. See dsalogitrework.md "
                    "Section 3 (Phase 2)."
                )
            logits = fp8_paged_mqa_logits_torch_dsa(
                q_fp8.unsqueeze(1)[:q_offset],
                kv_cache_fp8,
                weights[:q_offset],
                seqlens_32_2d,
                block_tables,
                None,  # schedule_metadata unused by the torch path
                max_seq_len,
                clean_logits=False,
            )
        else:
            logits = deepgemm_paged_mqa_logits_split(
                deep_gemm.fp8_paged_mqa_logits,
                q_fp8,
                kv_cache_fp8,
                weights,
                seqlens_32_2d,
                block_tables,
                schedule_metadata,
                max_seq_len,
                q_offset=q_offset,
            )'''

edits = [("5a-import", old5a, new5a), ("5b-use-dg-native", old5b, new5b),
         ("5c-fallback-metadata", old5c, new5c), ("5d-dispatch-branch", old5d, new5d)]

if marker5 in src5:
    print("dsa_indexer.py: torch fallback dispatch already patched, skipping")
else:
    missing = [tag for tag, old, new in edits if old not in src5]
    if missing:
        for tag in missing:
            print(f"ANCHOR-DRIFT: dsa_indexer.py: torch fallback dispatch anchor '{tag}' missing (SGLang version drift; re-check anchor)")
    else:
        for tag, old, new in edits:
            src5 = src5.replace(old, new, 1)
        f5.write_text(src5)
        print("Patched dsa_indexer.py: added torch-backend dispatch branch (phase 1, next_n==1) + gated DeepGEMM fallback + excluded dg_native")
PATCH_DSA_TORCH_INDEXER_EOF

# ── DSA attention decode: gather + reuse dense fa2 (GB10/SM121, gather+reuse, not a new kernel) ──
# Every DEDICATED DSA attention kernel is dead on GB10/SM121: trtllm-gen FMHA is a
# datacenter-Blackwell-only ISA (live crash: "Unsupported architecture" at
# TllmGenFmhaRunner autotune), flashmla_sparse/flashmla_kv's sgl_kernel extension is
# not built in this image, fa3 has a hard SM90/SM100-only gate, and tilelang compiles
# on SM121 but has a proven smem-vs-compile contradiction (no block_I both fits the
# ~99 KB budget and compiles). Full survey + verdict: DSA_speedup.md.
#
# Instead of a new kernel: gather the indexer's top-k selected KV (the "gather" prep
# ALREADY exists -- dsa_backend.py::forward_decode builds
# page_table_1 = transform_index_page_table_decode(page_table, topk_indices, page_size=1)
# for every backend) and run flashinfer's DENSE MLA decode (backend="fa2", the SAME
# kernel that already serves this model's dense baseline on SM121) over the small
# gathered subset. flashinfer's MLA wrapper rejects fp8 kv_data_type outside SM90/fa3,
# so the gathered KV must be dequantized to bf16 first -- reusing SGLang's OWN
# dequantize_k_cache_paged (already imported in dsa_backend.py, already used by the
# flashmla_sparse RAGGED-prefill path for exactly this purpose), not reinvented.
# Full design + numeric verification: dsalogitrework.md PART 2.
#
# New OPT-IN decode backend value "flashinfer_gather" (dsa_decode_backend), analogous
# to the paged-mqa-logits "torch" backend above: not selected by auto-detection, no
# arch gate, zero behavior change for every model/arch that doesn't explicitly select
# it. PHASE 1 ONLY: plain single-token decode. MTP/NEXTN target-verify (next_n>=2, via
# forward_extend's dsa_decode_impl reuse) is not wired here and falls through to
# forward_extend's existing "unsupported" handling -- moot for now since MTP stays off
# until this path is live-validated.
python3 - <<'PATCH_DSA_FLASHINFER_GATHER_EOF'
import pathlib

# --- File A: server_args.py -- add "flashinfer_gather" to DSA_CHOICES ---
fA = pathlib.Path("/usr/local/lib/python3.12/dist-packages/sglang/srt/server_args.py")
srcA = fA.read_text()
markerA = "# [patch] _sgl_dsa_flashinfer_gather_choice_"
oldA = '''DSA_CHOICES = [
    "flashmla_sparse",
    "flashmla_kv",
    "flashmla_auto",
    "fa3",
    "tilelang",
    "aiter",
    "trtllm",
]'''
newA = f'''{markerA}
DSA_CHOICES = [
    "flashmla_sparse",
    "flashmla_kv",
    "flashmla_auto",
    "fa3",
    "tilelang",
    "aiter",
    "trtllm",
    "flashinfer_gather",
]'''

if markerA in srcA:
    print("server_args.py: flashinfer_gather DSA choice already patched, skipping")
elif oldA not in srcA:
    print("ANCHOR-DRIFT: server_args.py: DSA_CHOICES anchor missing (SGLang version drift; re-check anchor)")
else:
    srcA = srcA.replace(oldA, newA, 1)
    fA.write_text(srcA)
    print("Patched server_args.py: added 'flashinfer_gather' to DSA_CHOICES")

# --- File B: dsa_backend.py ---
fB = pathlib.Path("/usr/local/lib/python3.12/dist-packages/sglang/srt/layers/attention/dsa_backend.py")
srcB = fB.read_text()
markerB_init = "# [patch] _sgl_dsa_flashinfer_gather_init_"

# B1: __init__ -- cache slot for the lazily-built wrapper.
old_init = '''        # Allocate global workspace buffer for TRT-LLM kernels (ragged attention on SM100/B200, or trtllm decode)
        if self.device_sm_major >= 10 or self.dsa_decode_impl == "trtllm":
            global global_workspace_buffer
            if global_workspace_buffer is None:
                global_workspace_buffer = torch.empty(
                    envs.SGLANG_FLASHINFER_WORKSPACE_SIZE.get(),
                    dtype=torch.uint8,
                    device=model_runner.device,
                )
            self.workspace_buffer = global_workspace_buffer
        else:
            self.workspace_buffer = None'''
new_init = old_init + f'''

        {markerB_init}
        # gather+dense-fa2 fallback (dsalogitrework.md PART 2): reuse the working
        # dense MLA decode kernel over the indexer's gathered+dequantized top-k KV,
        # since every dedicated DSA attention kernel is dead on GB10/SM121. Built
        # lazily on first use in _forward_flashinfer_gather (needs self.workspace_buffer,
        # set just above).
        self._flashinfer_gather_wrapper = None'''

# B2: forward_decode dispatch -- new elif branch before the final else/assert.
old_dispatch = '''elif self.dsa_decode_impl == "aiter":
            if q_all is None or not _is_hip:
                q_all = torch.cat([q_nope, q_rope], dim=-1)
            return self._forward_aiter(
                q_all=q_all,
                kv_cache=kv_cache,
                page_table_1=page_table_1,
                layer=layer,
                metadata=metadata,
                bs=forward_batch.batch_size,
            )

        else:
            assert False, f"Unsupported {self.dsa_decode_impl = }"'''
new_dispatch = f'''elif self.dsa_decode_impl == "aiter":
            if q_all is None or not _is_hip:
                q_all = torch.cat([q_nope, q_rope], dim=-1)
            return self._forward_aiter(
                q_all=q_all,
                kv_cache=kv_cache,
                page_table_1=page_table_1,
                layer=layer,
                metadata=metadata,
                bs=forward_batch.batch_size,
            )

        elif self.dsa_decode_impl == "flashinfer_gather":
            return self._forward_flashinfer_gather(
                q_nope=q_nope,
                q_rope=q_rope,
                kv_cache=kv_cache,
                page_table_1=page_table_1,
                sm_scale=layer.scaling,
                v_head_dim=layer.v_head_dim,
                metadata=metadata,
                k_scale=(
                    layer.k_scale_float
                    if getattr(layer, "k_scale_float", None) is not None
                    else 1.0
                ),
            )

        else:
            assert False, f"Unsupported {{self.dsa_decode_impl = }}"'''

# B3: new method, inserted right before _forward_fa3.
old_method_anchor = "    def _forward_fa3(\n"
new_method = '''    def _forward_flashinfer_gather(
        self,
        q_nope: torch.Tensor,
        q_rope: torch.Tensor,
        kv_cache: torch.Tensor,
        page_table_1: torch.Tensor,
        sm_scale: float,
        v_head_dim: int,
        metadata: "DSAMetadata",
        k_scale: float = 1.0,
    ) -> torch.Tensor:
        """Phase 1 (dsalogitrework.md PART 2, plain decode / next_n==1 only).

        Every dedicated DSA attention kernel is dead on GB10/SM121 (trtllm-gen
        FMHA = datacenter-only ISA, flashmla = extension not built in this
        image, fa3 = hard SM90/SM100 gate, tilelang = proven smem/compile
        contradiction). Instead of a new kernel: gather the indexer's top-k
        selected KV and run flashinfer's DENSE MLA decode (backend="fa2", the
        kernel that already serves this model's dense baseline on SM121) over
        the small gathered+dequantized subset.

        flashinfer's MLA wrapper rejects fp8 kv_data_type outside SM90/fa3
        (dsalogitrework.md Section 2 "THE blocker"), which is why the dequant
        to bf16 happens BEFORE the wrapper, not inside it.

        FIXED 2026-07-16 (live crash: "AssertionError: dim_quant: 576 != 656
        detected in dequantize_k_cache_paged"): the KV pool's byte layout is
        NOT always the 656-byte packed/block-quantized layout that
        dequantize_k_cache_paged hardcodes. Per
        model_runner_kv_cache_mixin.py::calculate_mla_kv_cache_dim, that packed
        layout (dsa_kv_cache_store_fp8=True, 512 fp8 nope + 16 scale bytes +
        128 bf16-rope bytes = 656) is used ONLY when dsa_prefill_backend and
        dsa_decode_backend are both NOT "trtllm" (and, on HIP, not
        tilelang/aiter). Our deployment keeps dsa_prefill_backend="trtllm", so
        the pool is ALWAYS the plain layout (dim = kv_lora_rank +
        qk_rope_head_dim = 576): nope and rope are simply cast to fp8_e4m3
        directly at write time (memory_pool.py set_mla_kv_buffer's "else"
        branch), no per-block scale stored -- a single scalar k_scale (mirrors
        _forward_trtllm's own bmm1_scale derivation) applies uniformly on
        dequant. Branch on self.dsa_kv_cache_store_fp8 so BOTH pool layouts are
        handled correctly (not just our deployment's config) -- the original
        dequantize_k_cache_paged path is kept for when the packed layout really
        is in use, per the parent investigation's "do not force the 656
        assumption" directive.

        Numerically verified (2026-07-16, GPU pod) against an independent
        manual gather + dequant + softmax reference in BOTH byte layouts (the
        formerly-tested 656 packed layout AND, after this fix, the real
        production 576 plain layout): max abs diff ~0.0008-0.0012 (bf16-level),
        no NaN/all-zero, seed-varying. Open point (dsalogitrework.md):
        kv_len_arr masking for requests with real context < topk is wired
        (clamp + flashinfer's own kv_len_arr) but not proven correct
        end-to-end, only plumbing-tested.
        """
        from flashinfer.mla import BatchMLAPagedAttentionWrapper

        if self._flashinfer_gather_wrapper is None:
            self._flashinfer_gather_wrapper = BatchMLAPagedAttentionWrapper(
                self.workspace_buffer, backend="fa2"
            )
        wrapper = self._flashinfer_gather_wrapper

        num_tokens_q = q_nope.shape[0]
        num_heads = q_nope.shape[1]
        topk = page_table_1.shape[-1]
        device = q_nope.device

        # (num_tokens_q * topk, 1, kv_lora_rank + qk_rope_head_dim), bf16.
        if self.dsa_kv_cache_store_fp8:
            # Packed block-quantized layout (656 bytes/token). See docstring.
            gathered = dequantize_k_cache_paged(kv_cache, page_table_1.reshape(-1))
        else:
            # Plain raw layout (576 = kv_lora_rank + qk_rope_head_dim here, but
            # derived from the buffer itself, not hardcoded): flat per-token
            # fp8 slots, gather by the same flattened page_table_1 index
            # dequantize_k_cache_paged would have used, then a single-scalar
            # fp8->bf16 dequant (no per-block scale to unpack).
            flat_kv_cache = kv_cache.view(-1, kv_cache.shape[-1])
            gathered_fp8 = flat_kv_cache[page_table_1.reshape(-1).long()]
            # fp32 intermediate for the scale multiply (matches the existing
            # _dequantize_k_cache_fast_kernel Triton convention: load+cast to
            # fp32, multiply by scale, THEN cast down to bf16) -- avoids
            # extra bf16 rounding when k_scale != 1.0. A no-op precision-wise
            # when k_scale == 1.0 (today's default; see docstring).
            gathered = (
                (gathered_fp8.to(torch.float32) * k_scale)
                .to(torch.bfloat16)
                .unsqueeze(1)
            )
        ckv = gathered[..., :v_head_dim].contiguous()
        kpe = gathered[..., v_head_dim:].contiguous()

        qo_indptr = torch.arange(0, num_tokens_q + 1, device=device, dtype=torch.int32)
        # page_size=1 post-gather: the freshly gathered/dequantized buffer is
        # already dense per request, so a plain sequential index is correct
        # (no second indirection into the original packed KV cache needed).
        kv_indptr = qo_indptr * topk
        kv_indices = torch.arange(
            0, num_tokens_q * topk, device=device, dtype=torch.int32
        )
        # Real per-request valid KV count (<=topk; short sequences leave a
        # padded tail in page_table_1 -- see dsalogitrework.md "Open points").
        kv_len_arr = metadata.dsa_cache_seqlens_int32.clamp(max=topk).to(torch.int32)

        wrapper.plan(
            qo_indptr,
            kv_indptr,
            kv_indices,
            kv_len_arr,
            num_heads,
            v_head_dim,
            kpe.shape[-1],
            1,
            True,
            sm_scale,
            q_nope.dtype,
            ckv.dtype,
        )
        return wrapper.run(q_nope, q_rope, ckv, kpe, return_lse=False)

    def _forward_fa3(\n'''

edits = [
    ("B1-init", old_init, new_init),
    ("B2-dispatch", old_dispatch, new_dispatch),
    ("B3-method", old_method_anchor, new_method),
]

if markerB_init in srcB:
    print("dsa_backend.py: flashinfer_gather wiring already patched, skipping")
else:
    missing = [tag for tag, old, new in edits if old not in srcB]
    if missing:
        for tag in missing:
            print(f"ANCHOR-DRIFT: dsa_backend.py: flashinfer_gather anchor '{tag}' missing (SGLang version drift; re-check anchor)")
    else:
        for tag, old, new in edits:
            srcB = srcB.replace(old, new, 1)
        fB.write_text(srcB)
        print("Patched dsa_backend.py: added flashinfer_gather decode backend (gather + dense fa2, phase 1)")
PATCH_DSA_FLASHINFER_GATHER_EOF

# ── DSA PREFILL fallback: reuse the SAME gather+dense-fa2 flashinfer_gather ────
# implementation as decode, above. Every dedicated DSA prefill kernel is dead on
# GB10/SM121 for the identical reason (trtllm-gen FMHA ISA wall -- LIVE crash:
# "TllmGenFmhaRunner ... Unsupported architecture" at forward_batch warmup/first
# request, since dsa_prefill_backend defaults/resolves to "trtllm" whenever
# unset). flashmla/fa3/tilelang/aiter are equally dead (DSA_speedup.md survey).
#
# KEY FINDING (source-verified, 2026-07-16): SGLANG_DSA_FUSE_TOPK defaults to
# TRUE, and with dsa_topk_backend="sgl-kernel" (our config),
# _get_fused_topk_page_table() just returns topk_indices UNCHANGED -- i.e. the
# page_table_1 handed to BOTH forward_decode's AND forward_extend's dispatch
# chain is the SAME kind of tensor in both modes: [num_query_tokens, topk]
# physical KV-cache-slot indices, -1-padded past the real (topk-clamped) context
# length. forward_decode's num_query_tokens == batch_size (1 query token per
# request); forward_extend's num_query_tokens == the total number of prefill/
# extend tokens across the batch (many query tokens per request, ragged). The
# ALREADY-VERIFIED-AND-DEPLOYED _forward_flashinfer_gather (decode patch above)
# is written generically against q_nope.shape[0]/page_table_1.shape[-1] -- it
# does not assume "one token per request" anywhere -- and its kv_len_arr source,
# metadata.dsa_cache_seqlens_int32, is ALSO populated per-query-token for EXTEND
# mode (dsa_backend.py's non-speculative-extend branch: dsa_cache_seqlens_int32
# = compute_dsa_seqlens(seqlens_expanded, topk), where seqlens_expanded already
# has one real-context-length entry per query token, matching decode's
# per-request semantics exactly when extend_len==1). Net effect: NO changes to
# _forward_flashinfer_gather are needed to reuse it for prefill -- only a new
# dispatch branch in forward_extend.
#
# There is no true "dense" code path on GB10 to fall back to: MHA_ONE_SHOT (the
# only actually-dense prefill impl in this backend) is gated to
# `device_sm == 90 or (100 <= device_sm < 110)` in set_dsa_prefill_impl -- SM121
# never qualifies, so self.use_mha is unconditionally False here regardless of
# sequence length. This is not a limitation for correctness: for a real context
# length <= topk (2048; true for every short prompt, e.g. GSM8K/smoke tests),
# the indexer's top-k selection has nothing to exclude and (by construction of
# compute_dsa_seqlens/the -1 padding) degenerates to selecting the FULL causal
# context, so gather+dense-fa2 computes the exact same result a true dense
# causal prefill would. Genuine sparse selection (real context > topk) reuses
# the identical code path but is UNVERIFIED this session (see docstring below)
# -- deliberately NOT hard-blocked with a NotImplementedError, since doing so
# safely would need extra per-request host-sync bookkeeping in a hot dispatch
# path that itself risks a new correctness bug; the existing decode fallback
# documents its own analogous open point (short-sequence kv_len_arr) the same
# way rather than gating it, and this follows that precedent.
python3 - <<'PATCH_DSA_FLASHINFER_GATHER_PREFILL_EOF'
import pathlib

f = pathlib.Path("/usr/local/lib/python3.12/dist-packages/sglang/srt/layers/attention/dsa_backend.py")
src = f.read_text()
marker = "# [patch] _sgl_dsa_flashinfer_gather_prefill_"
old = '''        elif dsa_impl == "aiter":
            if q_rope is not None:
                q_all = torch.cat([q_nope, q_rope], dim=-1)
            return self._forward_aiter_extend(
                q_all=q_all,
                kv_cache=kv_cache,
                page_table_1=page_table_1,
                layer=layer,
            )
        else:
            raise ValueError(
                f"Unsupported {dsa_impl = } for forward_extend. Consider using an other attention backend."
            )'''
new = '''        elif dsa_impl == "aiter":
            if q_rope is not None:
                q_all = torch.cat([q_nope, q_rope], dim=-1)
            return self._forward_aiter_extend(
                q_all=q_all,
                kv_cache=kv_cache,
                page_table_1=page_table_1,
                layer=layer,
            )

        elif dsa_impl == "flashinfer_gather":
            ''' + marker + '''
            # Reuses the decode implementation UNCHANGED: page_table_1 here is
            # the same [num_query_tokens, topk] fused-topk physical-slot-index
            # tensor (see the sglang_launch.sh patch comment above this class
            # for the source trace proving this), and metadata.dsa_cache_seqlens_int32
            # is populated per-query-token for EXTEND mode by the caller of
            # forward_extend, matching _forward_flashinfer_gather's existing
            # kv_len_arr derivation with no changes needed.
            return self._forward_flashinfer_gather(
                q_nope=q_nope,
                q_rope=q_rope,
                kv_cache=kv_cache,
                page_table_1=page_table_1,
                sm_scale=layer.scaling,
                v_head_dim=layer.v_head_dim,
                metadata=metadata,
                k_scale=(
                    layer.k_scale_float
                    if getattr(layer, "k_scale_float", None) is not None
                    else 1.0
                ),
            )

        else:
            raise ValueError(
                f"Unsupported {dsa_impl = } for forward_extend. Consider using an other attention backend."
            )'''

if marker in src:
    print("dsa_backend.py: flashinfer_gather PREFILL dispatch already patched, skipping")
elif old not in src:
    print("ANCHOR-DRIFT: dsa_backend.py: forward_extend final-else anchor missing (SGLang version drift; re-check anchor)")
else:
    src = src.replace(old, new, 1)
    f.write_text(src)
    print("Patched dsa_backend.py: added flashinfer_gather PREFILL dispatch (reuses decode's gather + dense fa2 impl, phase 1)")
PATCH_DSA_FLASHINFER_GATHER_PREFILL_EOF

# ── DSA flashinfer_gather CUDA-GRAPH plan/run-split (dsa_cuda_graph_plan.md) ───
# The decode/prefill flashinfer_gather patches above call wrapper.plan() INLINE in
# _forward_flashinfer_gather every forward. plan() does host-side stream sync /
# alloc that is NOT cuda-graph-recordable -> crash at decode-graph capture (the
# reason disable_cuda_graph was the eager workaround). This block splits it exactly
# like the native FlashInferMLAAttnBackend: build a per-bs wrapper(use_cuda_graph=
# True) OUT of the captured region, call the REAL .plan() once, monkeypatch
# wrapper.plan -> fast_mla_decode_plan (a module-level, causal-generic fast variant
# that skips the stream sync), then INSIDE the captured region call ONLY
# wrapper.run(). Validated bit-exact vs eager on synthetic tensors (scratchpad/
# microtest.py, incl. causal=True and short-seq kv_len) before deploy.
#
# INERT unless attention_backend=dsa + dsa_decode_backend=flashinfer_gather (the
# gated dispatch is only reached then). Eager path (prefill/extend, or
# disable_cuda_graph decode) keeps the original inline-plan behavior UNCHANGED.
python3 - <<'PATCH_DSA_FIG_GRAPH_SPLIT_EOF'
import pathlib

f = pathlib.Path("/usr/local/lib/python3.12/dist-packages/sglang/srt/layers/attention/dsa_backend.py")
src = f.read_text()
marker = "# [patch] _sgl_dsa_fig_graph_split_"

# S1: __init__ -- expand the single eager wrapper slot into the per-bs graph state.
s1_old = '''        self._flashinfer_gather_wrapper = None'''
s1_new = '''        self._flashinfer_gather_wrapper = None
        ''' + marker + '''
        # per-bs cuda-graph wrappers (plan/run split, dsa_cuda_graph_plan.md). The
        # eager slot above stays for prefill/extend + disable_cuda_graph decode
        # (inline plan); these back the CAPTURED-decode path (run-only in graph).
        self._flashinfer_gather_wrappers = {}   # bs -> BatchMLAPagedAttentionWrapper(use_cuda_graph=True)
        self._fig_static = {}                    # bs -> {qo_cpu, kv_indptr_cpu, kv_indices, kv_len_buf}
        self._fig_plan_params = None             # (num_heads, ckv_d, kpe_d, sm_scale, q_dtype, kv_dtype); model-wide const'''

# S2: signature -- add is_decode so the prefill dispatch can force the eager path.
s2_old = '''        metadata: "DSAMetadata",
        k_scale: float = 1.0,
    ) -> torch.Tensor:'''
s2_new = '''        metadata: "DSAMetadata",
        k_scale: float = 1.0,
        is_decode: bool = True,
    ) -> torch.Tensor:'''

# S3: head -- drop the build-eager-wrapper-at-top (moved into the eager branch).
s3_old = '''        from flashinfer.mla import BatchMLAPagedAttentionWrapper

        if self._flashinfer_gather_wrapper is None:
            self._flashinfer_gather_wrapper = BatchMLAPagedAttentionWrapper(
                self.workspace_buffer, backend="fa2"
            )
        wrapper = self._flashinfer_gather_wrapper

        num_tokens_q = q_nope.shape[0]'''
s3_new = '''        from flashinfer.mla import BatchMLAPagedAttentionWrapper

        num_tokens_q = q_nope.shape[0]'''

# S4: tail -- replace the inline plan()+run() with the graph(run-only)/eager(plan+run) split.
s4_old = '''        qo_indptr = torch.arange(0, num_tokens_q + 1, device=device, dtype=torch.int32)
        # page_size=1 post-gather: the freshly gathered/dequantized buffer is
        # already dense per request, so a plain sequential index is correct
        # (no second indirection into the original packed KV cache needed).
        kv_indptr = qo_indptr * topk
        kv_indices = torch.arange(
            0, num_tokens_q * topk, device=device, dtype=torch.int32
        )
        # Real per-request valid KV count (<=topk; short sequences leave a
        # padded tail in page_table_1 -- see dsalogitrework.md "Open points").
        kv_len_arr = metadata.dsa_cache_seqlens_int32.clamp(max=topk).to(torch.int32)

        wrapper.plan(
            qo_indptr,
            kv_indptr,
            kv_indices,
            kv_len_arr,
            num_heads,
            v_head_dim,
            kpe.shape[-1],
            1,
            True,
            sm_scale,
            q_nope.dtype,
            ckv.dtype,
        )
        return wrapper.run(q_nope, q_rope, ckv, kpe, return_lse=False)'''
s4_new = '''        bs = num_tokens_q
        # Stash the model-wide-constant plan params from the layer (available HERE,
        # not in init_forward_metadata_out_graph). MLA scaling/dims are identical
        # across all attention layers, so a one-time capture is correct.
        if self._fig_plan_params is None:
            self._fig_plan_params = (
                num_heads, v_head_dim, kpe.shape[-1], sm_scale, q_nope.dtype, ckv.dtype
            )

        # decode_cuda_graph_metadata[bs] exists ONLY for cuda-graph-captured decode
        # batch sizes (never for prefill/extend, never under disable_cuda_graph).
        # is_decode excludes the prefill dispatch (which reuses this method but must
        # always take the eager inline-plan path).
        graph_meta = getattr(self, "decode_cuda_graph_metadata", None)
        is_graph_decode = (
            is_decode
            and graph_meta is not None
            and graph_meta.get(bs, None) is not None
        )

        if is_graph_decode:
            # plan/run-split: the wrapper for this bs was planned OUT of the captured
            # region (either here during the uncaptured warmup run_once, or in
            # init_forward_metadata_out_graph at capture-prep). Inside the captured
            # region we call ONLY wrapper.run().
            if bs not in self._flashinfer_gather_wrappers:
                assert not torch.cuda.is_current_stream_capturing(), (
                    "flashinfer_gather graph wrapper missing for bs=%d inside capture "
                    "(warmup run_once / out_graph should have built it)" % bs
                )
                self._fig_build_graph_wrapper(
                    bs, num_heads, v_head_dim, kpe.shape[-1], sm_scale,
                    q_nope.dtype, ckv.dtype, metadata, topk,
                )
            wrapper = self._flashinfer_gather_wrappers[bs]
            return wrapper.run(q_nope, q_rope, ckv, kpe, return_lse=False)

        # Eager path (prefill/extend, or disable_cuda_graph decode): reusable
        # non-graph wrapper, plan() inline every call (original verified behavior).
        if self._flashinfer_gather_wrapper is None:
            self._flashinfer_gather_wrapper = BatchMLAPagedAttentionWrapper(
                self.workspace_buffer, backend="fa2"
            )
        wrapper = self._flashinfer_gather_wrapper
        qo_indptr = torch.arange(0, num_tokens_q + 1, device=device, dtype=torch.int32)
        # page_size=1 post-gather: the gathered/dequantized buffer is already dense
        # per request, so a plain sequential index is correct.
        kv_indptr = qo_indptr * topk
        kv_indices = torch.arange(
            0, num_tokens_q * topk, device=device, dtype=torch.int32
        )
        kv_len_arr = metadata.dsa_cache_seqlens_int32.clamp(max=topk).to(torch.int32)
        wrapper.plan(
            qo_indptr, kv_indptr, kv_indices, kv_len_arr,
            num_heads, v_head_dim, kpe.shape[-1], 1, True, sm_scale,
            q_nope.dtype, ckv.dtype,
        )
        return wrapper.run(q_nope, q_rope, ckv, kpe, return_lse=False)'''

# S5: two new methods, inserted right before _forward_flashinfer_gather.
s5_old = '''    def _forward_flashinfer_gather(
        self,
        q_nope: torch.Tensor,'''
s5_new = '''    def _fig_build_graph_wrapper(
        self, bs, num_heads, head_dim_ckv, head_dim_kpe,
        sm_scale, q_dtype, kv_dtype, metadata, topk,
    ):
        """Build + REAL-plan a per-bs cuda-graph flashinfer_gather wrapper ONCE, then
        monkeypatch its .plan to fast_mla_decode_plan (skips the non-graph stream sync
        on every subsequent replay). Mirrors FlashInferMLAAttnBackend's capture path.
        Post-gather addressing is fully static given (bs, topk); only kv_len_arr is
        dynamic (updated in place each replay by _fig_replan_graph)."""
        from functools import partial
        from flashinfer.mla import BatchMLAPagedAttentionWrapper
        from sglang.srt.layers.attention.flashinfer_mla_backend import fast_mla_decode_plan

        dev = metadata.dsa_cache_seqlens_int32.device
        qo_indptr = torch.arange(0, bs + 1, device=dev, dtype=torch.int32)
        kv_indptr = qo_indptr * topk
        kv_indices = torch.arange(0, bs * topk, device=dev, dtype=torch.int32)
        kv_len_buf = torch.empty(bs, device=dev, dtype=torch.int32)
        kv_len_buf.copy_(
            metadata.dsa_cache_seqlens_int32[:bs].clamp(max=topk).to(torch.int32)
        )
        wrapper = BatchMLAPagedAttentionWrapper(
            self.workspace_buffer, use_cuda_graph=True,
            qo_indptr=qo_indptr, kv_indptr=kv_indptr,
            kv_indices=kv_indices, kv_len_arr=kv_len_buf, backend="fa2",
        )
        # REAL plan once (populates wrapper._cached_module for the fast variant).
        wrapper.plan(
            qo_indptr, kv_indptr, kv_indices, kv_len_buf,
            num_heads, head_dim_ckv, head_dim_kpe, 1, True, sm_scale, q_dtype, kv_dtype,
        )
        wrapper.plan = partial(fast_mla_decode_plan, wrapper)
        self._flashinfer_gather_wrappers[bs] = wrapper
        self._fig_static[bs] = {
            "qo_cpu": qo_indptr.cpu(),
            "kv_indptr_cpu": kv_indptr.cpu(),
            "kv_indices": kv_indices,
            "kv_len_buf": kv_len_buf,
        }

    def _fig_replan_graph(self, bs, metadata):
        """Out-of-graph capture-prep / replay-prep: refresh kv_len (the one dynamic
        quantity) and re-run the FAST plan (no stream sync). Builds the wrapper lazily
        if the params are already stashed (capture-prep before the warmup run_once);
        no-op until either the wrapper exists or params are known."""
        if self.dsa_index_topk is None:
            return
        topk = self.dsa_index_topk
        if bs not in self._flashinfer_gather_wrappers:
            if self._fig_plan_params is None:
                return  # will be built by the uncaptured warmup run_once instead
            nh, ckv_d, kpe_d, sm, qd, kd = self._fig_plan_params
            self._fig_build_graph_wrapper(bs, nh, ckv_d, kpe_d, sm, qd, kd, metadata, topk)
            return
        nh, ckv_d, kpe_d, sm, qd, kd = self._fig_plan_params
        st = self._fig_static[bs]
        st["kv_len_buf"].copy_(
            metadata.dsa_cache_seqlens_int32[:bs].clamp(max=topk).to(torch.int32)
        )
        wrapper = self._flashinfer_gather_wrappers[bs]
        wrapper.plan(
            st["qo_cpu"], st["kv_indptr_cpu"], st["kv_indices"], st["kv_len_buf"].cpu(),
            nh, ckv_d, kpe_d, 1, True, sm, qd, kd,
        )

    def _forward_flashinfer_gather(
        self,
        q_nope: torch.Tensor,'''

# S6: init_forward_metadata_out_graph -- add the out-of-graph plan/replan hook.
s6_old = '''        self._apply_cuda_graph_metadata(
            bs=forward_batch.batch_size,
            req_pool_indices=forward_batch.req_pool_indices,
            seq_lens=forward_batch.seq_lens,
            seq_lens_cpu=seq_lens_cpu,
            forward_mode=forward_batch.forward_mode,
            spec_info=forward_batch.spec_info,
            out_cache_loc=getattr(forward_batch, "out_cache_loc", None),
            actual_forward_mode=getattr(forward_batch, "actual_forward_mode", None),
        )'''
s6_new = '''        self._apply_cuda_graph_metadata(
            bs=forward_batch.batch_size,
            req_pool_indices=forward_batch.req_pool_indices,
            seq_lens=forward_batch.seq_lens,
            seq_lens_cpu=seq_lens_cpu,
            forward_mode=forward_batch.forward_mode,
            spec_info=forward_batch.spec_info,
            out_cache_loc=getattr(forward_batch, "out_cache_loc", None),
            actual_forward_mode=getattr(forward_batch, "actual_forward_mode", None),
        )
        ''' + marker + '''
        # Out-of-graph plan/replan for the flashinfer_gather captured-decode wrapper
        # (dsa_cuda_graph_plan.md): builds it at capture-prep if params are known, and
        # fast-replans (fresh kv_len) before every replay. INERT for any other backend.
        if (
            self.dsa_decode_impl == "flashinfer_gather"
            and forward_batch.forward_mode.is_decode_or_idle()
        ):
            _fig_meta = self.decode_cuda_graph_metadata.get(forward_batch.batch_size)
            if _fig_meta is not None:
                self._fig_replan_graph(forward_batch.batch_size, _fig_meta)'''

# S7: prefill dispatch -- force the eager path (is_decode=False). Anchored on the
# forward_extend ValueError, which is UNIQUE to the prefill method (decode raises an
# assert), so this targets the prefill call, not the identical decode call.
s7_old = '''                metadata=metadata,
                k_scale=(
                    layer.k_scale_float
                    if getattr(layer, "k_scale_float", None) is not None
                    else 1.0
                ),
            )

        else:
            raise ValueError(
                f"Unsupported {dsa_impl = } for forward_extend. Consider using an other attention backend."
            )'''
s7_new = '''                metadata=metadata,
                is_decode=False,
                k_scale=(
                    layer.k_scale_float
                    if getattr(layer, "k_scale_float", None) is not None
                    else 1.0
                ),
            )

        else:
            raise ValueError(
                f"Unsupported {dsa_impl = } for forward_extend. Consider using an other attention backend."
            )'''

edits = [
    ("S1-init-state", s1_old, s1_new),
    ("S2-signature", s2_old, s2_new),
    ("S3-head", s3_old, s3_new),
    ("S4-tail-split", s4_old, s4_new),
    ("S5-new-methods", s5_old, s5_new),
    ("S6-out_graph-hook", s6_old, s6_new),
    ("S7-prefill-is_decode", s7_old, s7_new),
]

if marker in src:
    print("dsa_backend.py: flashinfer_gather cuda-graph plan/run-split already patched, skipping")
else:
    missing = [tag for tag, old, new in edits if old not in src]
    if missing:
        for tag in missing:
            print(f"ANCHOR-DRIFT: dsa_backend.py: fig-graph-split anchor '{tag}' missing (SGLang version drift; re-check anchor)")
    else:
        for tag, old, new in edits:
            src = src.replace(old, new, 1)
        f.write_text(src)
        print("Patched dsa_backend.py: flashinfer_gather CUDA-GRAPH plan/run-split (per-bs wrapper, real-plan-once + fast_mla_decode_plan, run-only in graph)")
PATCH_DSA_FIG_GRAPH_SPLIT_EOF



# Patch SGLang get_config() to convert dict sub_configs after loading (transformers 5.5.0 bug).
# Transformers 5.x auto-generates __init__ for PretrainedConfig subclasses with sub_configs,
# bypassing dict→config conversion. from_pretrained() also bypasses __post_init__.
# Both vision_config and text_config arrive as raw dicts → AttributeError on .hidden_size etc.
# Fix: patch get_config() to convert dict sub-configs after loading for any config with sub_configs.
#
# RE-ANCHORED 2026-07-16: sglang/srt/utils/hf_transformers_utils.py is now a bare
# backward-compat shim ("all code has moved to sglang.srt.utils.hf_transformers") — it
# has no get_config()/"return config" left, so the old anchor silently never matched on
# this image. The real get_config() lives in sglang/srt/utils/hf_transformers/config.py.
# The underlying transformers-5.x sub_configs bug is STILL UNSOLVED there: only the
# Mistral parser path calls the sglang-native _ensure_sub_configs() helper (for
# text_config/vision_config); the generic "hf" parser path (used by everything else,
# incl. GLM/DeepSeek/NemotronH) does not call it anywhere — verified by grepping the
# whole hf_transformers package. So this patch still applies, just at the new home.
HF_UTILS="/usr/local/lib/python3.12/dist-packages/sglang/srt/utils/hf_transformers/config.py"
if [ ! -f "$HF_UTILS" ]; then
  echo "ANCHOR-DRIFT: hf_transformers/config.py: get_config sub_configs fix target file missing (SGLang restructured/renamed?)"
elif grep -q 'sub_configs dict fix' "$HF_UTILS" 2>/dev/null; then
  echo "get_config() sub_configs dict fix: already applied, skipping"
elif grep -q 'def get_config' "$HF_UTILS" 2>/dev/null; then
  python3 << 'PATCH_GET_CONFIG_EOF'
f = "/usr/local/lib/python3.12/dist-packages/sglang/srt/utils/hf_transformers/config.py"
with open(f) as fh:
    code = fh.read()
# Find the final "return config" in get_config() and add sub_configs conversion before it.
# The gguf branch was refactored (model_type/config.update -> _set_architectures helper)
# since this patch was first written; anchor matches the current shape.
old = '''    if is_gguf:
        if config.model_type not in MODEL_FOR_CAUSAL_LM_MAPPING_NAMES:
            raise RuntimeError(f"Can't get gguf config for {config.model_type}.")
        _set_architectures(config, MODEL_FOR_CAUSAL_LM_MAPPING_NAMES[config.model_type])

    return config'''
new = '''    if is_gguf:
        if config.model_type not in MODEL_FOR_CAUSAL_LM_MAPPING_NAMES:
            raise RuntimeError(f"Can't get gguf config for {config.model_type}.")
        _set_architectures(config, MODEL_FOR_CAUSAL_LM_MAPPING_NAMES[config.model_type])

    # [patch] sub_configs dict fix — transformers 5.x from_pretrained() leaves sub-configs
    # as raw dicts instead of converting to their declared config classes.
    _sub_cfgs = getattr(config, "sub_configs", None)
    if _sub_cfgs:
        for _key, _cls in _sub_cfgs.items():
            _val = getattr(config, _key, None)
            if isinstance(_val, dict):
                try:
                    setattr(config, _key, _cls(**_val))
                except Exception:
                    pass  # non-critical: some sub-configs may not accept all dict keys

    # [patch] Qwen3.5 MoE: text_config lacks norm_topk_prob (Qwen2MoeSparseMoeBlock expects it).
    # Qwen3.5 uses softmax routing — renormalize=True is correct default.
    _tc = getattr(config, "text_config", None)
    if _tc is not None and not isinstance(_tc, dict) and not hasattr(_tc, "norm_topk_prob"):
        _tc.norm_topk_prob = True

    return config'''
if old in code:
    code = code.replace(old, new, 1)
    with open(f, 'w') as fh:
        fh.write(code)
    print("Patched hf_transformers/config.py get_config(): sub_configs dict→config conversion after loading")
else:
    print("ANCHOR-DRIFT: hf_transformers/config.py: get_config sub_configs fix (SGLang version drift; re-check anchor)")
PATCH_GET_CONFIG_EOF
else
  echo "ANCHOR-DRIFT: hf_transformers/config.py: get_config() function not found (SGLang version drift; re-check anchor)"
fi

# ─────────────────────────────────────────────────────────────────────────────
# [patch] _sgl_mistral_native_tokenizer_ — let get_tokenizer() load the shipped HF
# tokenizer.json OFFLINE for Mistral-native checkpoints (no config.json).
#
# Mistral-native repos (params.json + tekken.json + tokenizer.json, e.g.
# Mistral-Small-4-119B-NVFP4) ship NO config.json. SGLang's ModelConfig + get_processor
# already handle that via the name-triggered is_mistral_model()/load_mistral_config()
# path, BUT get_processor's TokenizersBackend reload (and any DIRECT get_tokenizer call)
# hits AutoTokenizer.from_pretrained → which loads config.json → offline OSError, crashing
# tokenizer init before any weight load. Upstream only special-cases BARE-tekken
# (tekken.json AND no tokenizer.json); these repos ALSO ship tokenizer.json, so the
# upstream is_bare_tekken_checkpoint branch is skipped and upstream main fails identically
# — an image bump does NOT fix this.
#
# Fix: inject the config built from params.json (load_mistral_config) into common_kwargs
# so AutoTokenizer (and _resolve_tokenizers_backend) skip the config.json fetch and load
# the shipped tokenizer.json as a standard HF TokenizersBackend. This keeps the full HF
# tokenizer API (add_special_tokens / apply_chat_template) that the pixtral MM processor
# requires — routing to MistralCommonTokenizer instead would need the ~119-line
# multimodal adaptation (add_special_tokens stub, pixtral markers) our image's
# patch_mistral_common_tokenizer lacks. Verified in a debug pod (0.5.14-sm121): loads
# offline, token IDs match the shipped tokenizer.json, pixtral add_special_tokens passes.
# See reference_sglang_mistral_native_support.
TOKPY="/usr/local/lib/python3.12/dist-packages/sglang/srt/utils/hf_transformers/tokenizer.py"
if [ ! -f "$TOKPY" ]; then
  echo "ANCHOR-DRIFT: tokenizer.py: _sgl_mistral_native_tokenizer_ target file missing (SGLang restructured/renamed?)"
elif grep -q '_sgl_mistral_native_tokenizer_' "$TOKPY" 2>/dev/null; then
  echo "tokenizer.py: _sgl_mistral_native_tokenizer_ already applied, skipping"
elif grep -q 'common_kwargs = dict(' "$TOKPY" 2>/dev/null; then
  python3 << 'PATCH_MISTRAL_NATIVE_TOK_EOF'
f = "/usr/local/lib/python3.12/dist-packages/sglang/srt/utils/hf_transformers/tokenizer.py"
with open(f) as fh:
    code = fh.read()
# Insert AFTER _resolve_tokenizer_name() (which applies _MISTRAL_TOKENIZER_REDIRECTS,
# e.g. Devstral) and BEFORE the AutoTokenizer try-block, so redirects still win.
old = '''    common_kwargs = dict(
        trust_remote_code=trust_remote_code,
        tokenizer_revision=tokenizer_revision,
        clean_up_tokenization_spaces=False,
        **kwargs,
    )

    try:'''
new = '''    common_kwargs = dict(
        trust_remote_code=trust_remote_code,
        tokenizer_revision=tokenizer_revision,
        clean_up_tokenization_spaces=False,
        **kwargs,
    )

    # [patch] _sgl_mistral_native_tokenizer_ — Mistral-native (params.json, tekken.json,
    # tokenizer.json, NO config.json) checkpoints make AutoTokenizer fetch config.json
    # -> offline OSError. Inject the config built from params.json so the shipped HF
    # tokenizer.json loads offline as a standard TokenizersBackend (full HF API the pixtral
    # MM processor needs). MistralCommonBackend would miss add_special_tokens here.
    from .mistral_utils import is_mistral_model, load_mistral_config
    if is_mistral_model(tokenizer_name) and "config" not in common_kwargs:
        common_kwargs["config"] = load_mistral_config(
            tokenizer_name, trust_remote_code=trust_remote_code, revision=tokenizer_revision
        )

    try:'''
if old in code:
    code = code.replace(old, new, 1)
    with open(f, 'w') as fh:
        fh.write(code)
    print("Patched tokenizer.py: Mistral-native config injection (offline tokenizer.json)")
else:
    print("ANCHOR-DRIFT: tokenizer.py: _sgl_mistral_native_tokenizer_ (SGLang version drift; re-check anchor)")
PATCH_MISTRAL_NATIVE_TOK_EOF
else
  echo "ANCHOR-DRIFT: tokenizer.py: common_kwargs anchor not found (SGLang version drift; re-check anchor)"
fi

# ─────────────────────────────────────────────────────────────────────────────
# Patch model_config.py: route NVFP4-bearing MIXED_PRECISION modelopt checkpoints
# to the modelopt_mixed loader regardless of architecture.
#
# SGLang 0.5.12/0.5.13's _parse_modelopt_quant_config() (configs/model_config.py,
# this function is byte-identical across both releases) maps a
# `quant_algo: MIXED_PRECISION` checkpoint to a quant method by ARCHITECTURE:
# NemotronH → modelopt_mixed, everything else → w4afp8. For
# Qwen3_5MoeForConditionalGeneration (nvidia/Qwen3.6-35B-A3B-NVFP4: W4A16_NVFP4 MoE
# experts + FP8 linear_attn, group_size 16) that fall-through is WRONG:
#   * quantization=modelopt_fp4/modelopt_mixed → _verify_quantization rejects the
#     mismatch (detected w4afp8 not in compatible[modelopt_*]=["modelopt"]).
#   * quantization=w4afp8 passes verify but the W4AFp8Config loader is the W4A8
#     (int4 + per-group fp8-scale, default group_size 128) path — the wrong weight
#     encoding for NVFP4 (e2m1) experts.
# The ModelOptMixedPrecisionConfig loader DOES compose per-layer NVFP4+FP8 correctly
# but is unreachable for this arch (its override_quantization_method only fires once
# detection already returned modelopt_mixed). Fix: also route to modelopt_mixed when
# any quantized layer is NVFP4 — discriminate on the real checkpoint format, not the
# arch name. NemotronH path preserved; pure-W4A8 MIXED_PRECISION still → w4afp8.
# Pairs with quantization: modelopt_mixed in model_profiles/nvidia-qwen3.6-35b-a3b-nvfp4.yml.
# ─────────────────────────────────────────────────────────────────────────────
python3 - <<'PATCH_MIXED_NVFP4_DISPATCH_EOF'
import pathlib
p = pathlib.Path("/usr/local/lib/python3.12/dist-packages/sglang/srt/configs/model_config.py")
if not p.exists():
    print("ANCHOR-DRIFT: model_config.py: _sgl_mixed_nvfp4_dispatch_ target file missing (SGLang restructured/renamed?)")
else:
    src = p.read_text()
    marker = "# [patch] _sgl_mixed_nvfp4_dispatch_"
    # RE-CHECKED 2026-07-16: upstream's _parse_modelopt_quant_config() now discriminates
    # MIXED_PRECISION checkpoints by scanning quantized_layers for NVFP4/W4A16_NVFP4
    # entries (has_modelopt_nvfp4_layers) and routes to modelopt_mixed generically -- the
    # NemotronH-only arch-name gate this patch targeted is GONE, replaced by exactly the
    # real-format discrimination this patch wanted (verified live, same semantics as our
    # own _has_nvfp4 check below, just upstream-native now). OBSOLETE: self-disable
    # instead of forcing a stale re-anchor onto code upstream already solved better.
    if marker in src:
        print("model_config.py: MIXED_PRECISION NVFP4 dispatch already patched, skipping")
    elif "has_modelopt_nvfp4_layers" in src:
        print("model_config.py: upstream already discriminates MIXED_PRECISION NVFP4 vs w4afp8 "
              "natively (has_modelopt_nvfp4_layers, no NemotronH-only gate), skipping "
              "_sgl_mixed_nvfp4_dispatch_ (obsolete)")
    else:
        old = (
            '        if quant_algo == "MIXED_PRECISION":\n'
            '            architectures = getattr(self.hf_config, "architectures", []) or []\n'
            '            if getattr(self.hf_config, "model_type", None) == "nemotron_h" or any(\n'
            '                arch.startswith("NemotronH") for arch in architectures\n'
            '            ):\n'
            '                return {"quant_method": "modelopt_mixed", "quant_algo": quant_algo}\n'
            '            return {"quant_method": "w4afp8", "quant_algo": quant_algo}\n'
        )
        new = (
            '        if quant_algo == "MIXED_PRECISION":\n'
            '            architectures = getattr(self.hf_config, "architectures", []) or []\n'
            '            ' + marker + '\n'
            '            # Route NVFP4-bearing MIXED_PRECISION checkpoints (e.g.\n'
            '            # nvidia/Qwen3.6-35B-A3B-NVFP4: W4A16_NVFP4 MoE + FP8 attn) to the\n'
            '            # modelopt_mixed loader. Upstream gates modelopt_mixed to NemotronH\n'
            '            # only; other archs fall through to w4afp8 (W4A8 int4), the wrong\n'
            '            # weight format for NVFP4 experts. Discriminate on real format.\n'
            '            _qlayers = json_quant_configs.get("quantized_layers", {}) or {}\n'
            '            _has_nvfp4 = any(\n'
            '                "NVFP4" in str(_li.get("quant_algo", "")).upper()\n'
            '                for _li in _qlayers.values()\n'
            '                if isinstance(_li, dict)\n'
            '            )\n'
            '            if (\n'
            '                getattr(self.hf_config, "model_type", None) == "nemotron_h"\n'
            '                or any(arch.startswith("NemotronH") for arch in architectures)\n'
            '                or _has_nvfp4\n'
            '            ):\n'
            '                return {"quant_method": "modelopt_mixed", "quant_algo": quant_algo}\n'
            '            return {"quant_method": "w4afp8", "quant_algo": quant_algo}\n'
        )
        if old not in src:
            print("ANCHOR-DRIFT: model_config.py: _sgl_mixed_nvfp4_dispatch_ (SGLang version drift; re-check anchor)")
        else:
            p.write_text(src.replace(old, new, 1))
            print("Patched model_config.py: MIXED_PRECISION NVFP4 -> modelopt_mixed dispatch")
PATCH_MIXED_NVFP4_DISPATCH_EOF

# ─────────────────────────────────────────────────────────────────────────────
# Patch modelopt_quant.py: treat "W4A16_NVFP4" (and any *NVFP4* variant) as NVFP4
# in the ModelOptMixedPrecisionConfig per-layer dispatch.
#
# modelopt 0.44 labels weight-only NVFP4 layers as quant_algo "W4A16_NVFP4"
# (4-bit NVFP4 weights, higher-precision activations), NOT plain "NVFP4". On the
# OLD image this patch targeted, SGLang's ModelOptMixedPrecisionConfig.get_quant_method()
# did exact `quant_algo == "NVFP4"` comparisons, so W4A16_NVFP4 experts/linears
# matched NEITHER the FP8 nor the NVFP4 branch → got NO quant method → were built
# as unquantized → load_weights crashed on the missing w*_weight_scale_2 params
# (KeyError model.layers.0.mlp.experts.w2_weight_scale_2). Normalizing the variant
# string to "NVFP4" routed them to the right loader (ModelOptNvFp4FusedMoEMethod /
# ModelOptFp4LinearMethod). Pairs with the _sgl_mixed_nvfp4_dispatch_ patch above.
# Target: nvidia/Qwen3.6-35B-A3B-NVFP4 (W4A16_NVFP4 MoE + shared-expert linears).
#
# SELF-DISABLING as of 2026-07-15/16: newer images resolve W4A16_NVFP4 natively —
# a DEDICATED dispatch branch (`quant_algo == "W4A16_NVFP4": return
# ModelOptNvFp4A16LinearMethod(self.nvfp4a16_config)` for both the Linear and
# FusedMoE arms, plus a separate `nvfp4a16_config` instance) that correctly keeps
# W4A16 (weight-only) distinct from full NVFP4 (weight+activation). On such an
# image, THIS patch's variant-collapse (`"NVFP4" in quant_algo -> quant_algo =
# "NVFP4"`) fires BEFORE that dedicated branch is ever reached and silently
# reroutes W4A16 layers into the wrong (full-quant) method/config — a REGRESSION,
# not a fix, on the current image. Guard below detects the native branch
# (`ModelOptNvFp4A16LinearMethod` / `nvfp4a16_config` in source) and skips the
# whole patch (both A and B) as an explicit no-op when present, so it only ever
# applies on a genuine pre-native-W4A16 image.
# ─────────────────────────────────────────────────────────────────────────────
python3 - <<'PATCH_MIXED_NVFP4_VARIANT_EOF'
import pathlib
p = pathlib.Path("/usr/local/lib/python3.12/dist-packages/sglang/srt/layers/quantization/modelopt_quant.py")
if not p.exists():
    print("ANCHOR-DRIFT: modelopt_quant.py: _sgl_mixed_nvfp4_variant_ target file missing (SGLang restructured/renamed?)")
else:
    src = p.read_text()
    if "ModelOptNvFp4A16LinearMethod" in src or "nvfp4a16_config" in src:
        print("modelopt_quant.py: upstream already handles W4A16_NVFP4 natively, skipping _sgl_mixed_nvfp4_variant_ (obsolete)")
        raise SystemExit
    marker = "# [patch] _sgl_mixed_nvfp4_variant_"
    # Patch A: normalize the resolved quant_algo in get_quant_method.
    old_a = (
        '        quant_algo = self._resolve_quant_algo(prefix)\n'
        '\n'
        '        if isinstance(layer, (LinearBase, ParallelLMHead)):\n'
    )
    new_a = (
        '        quant_algo = self._resolve_quant_algo(prefix)\n'
        '        ' + marker + '\n'
        '        # modelopt 0.44 labels weight-only NVFP4 as "W4A16_NVFP4"; the exact\n'
        '        # == "NVFP4" checks below would otherwise leave these layers unquantized.\n'
        '        # The checkpoint ships the full NVFP4 tensor set, so route any *NVFP4*\n'
        '        # variant through ModelOptNvFp4* / ModelOptFp4LinearMethod.\n'
        '        if quant_algo and "NVFP4" in quant_algo:\n'
        '            quant_algo = "NVFP4"\n'
        '\n'
        '        if isinstance(layer, (LinearBase, ParallelLMHead)):\n'
    )
    # Patch B: group_size probe in from_config (exact-match → substring).
    old_b = '            if layer_info.get("quant_algo", "").upper() == "NVFP4":\n'
    new_b = '            if "NVFP4" in layer_info.get("quant_algo", "").upper():\n'

    if marker in src:
        print("modelopt_quant.py: NVFP4-variant normalization already patched, skipping")
    elif old_a not in src:
        print("ANCHOR-DRIFT: modelopt_quant.py: _sgl_mixed_nvfp4_variant_ get_quant_method dispatch (SGLang version drift; re-check anchor)")
    else:
        src = src.replace(old_a, new_a, 1)
        if old_b in src:
            src = src.replace(old_b, new_b, 1)
            print("Patched modelopt_quant.py: NVFP4-variant dispatch + group_size probe")
        else:
            print("Patched modelopt_quant.py: NVFP4-variant dispatch (A only)")
            print("ANCHOR-DRIFT: modelopt_quant.py: _sgl_mixed_nvfp4_variant_ group_size probe (SGLang version drift; re-check anchor)")
        p.write_text(src)
PATCH_MIXED_NVFP4_VARIANT_EOF

# ─────────────────────────────────────────────────────────────────────────────
# Patch qwen3_5.py: allow QUANTIZED attention for uniform-NVFP4 modelopt_fp4
# checkpoints (e.g. 's vroomfondel/Ornith-1.0-35B-NVFP4-ModelOpt).
#
# qwen3_5.py hardcodes attention (both the full self_attn qkv/o_proj AND the
# GatedDeltaNet linear-attn in_proj_qkvz/in_proj_ba) to UNQUANTIZED (BF16) the
# moment quant_config.get_name() == "modelopt_fp4", at two byte-identical sites:
#     linear_attn_quant_config = (
#         None
#         if quant_config and quant_config.get_name() == "modelopt_fp4"
#         else quant_config)
# Every NVIDIA-published Qwen3.5/3.6 NVFP4 excludes attention (MoE-only NVFP4), so
# the shortcut matches THEIR modelopt_fp4 checkpoints. Distinct from the
# MIXED_PRECISION patches above: nvidia/Qwen3.6 (W4A16_NVFP4 MoE + FP8 attn) is
# quant_method modelopt_mixed, so this override never fires for it. But a UNIFORM
# W4A4 checkpoint that ALSO quantizes attention ('s deliberately-risky nvfp4
# recipe, vs the safe nvfp4_mlp_only) is modelopt_fp4 -> hits this override -> gets
# a plain BF16 param for qkv_proj/in_proj_qkvz, and the merged loader then copies a
# NVFP4-packed uint8 chunk (half input-dim width) into the full-width BF16 slot ->
# "assert param_data.shape == loaded_weight.shape" (linear.py Merged/QKV loader).
# Fix: replace the forced-None condition with False so the real quant_config passes
# through and ModelOptFp4Config.is_layer_excluded() decides per-prefix (the SAME
# dispatch that already keeps mlp.gate/conv1d/in_proj_a/in_proj_b BF16 for this
# checkpoint). create_weights() already supports merged quantized linears; the code
# was reachable, just unreached. INERT for NVIDIA MoE-only NVFP4 (excluded
# attention -> UnquantizedLinearMethod), for MIXED_PRECISION (modelopt_mixed, not
# modelopt_fp4), and for FP8/etc (the ternary already took the else branch). No
# model-name gate: qwen3_5.py is imported only for the Qwen3.5 arch.
# ─────────────────────────────────────────────────────────────────────────────
python3 - <<'PATCH_QWEN35_ATTN_QUANT_EOF'
import os
f = "/usr/local/lib/python3.12/dist-packages/sglang/srt/models/qwen3_5.py"
if not os.path.exists(f):
    print("ANCHOR-DRIFT: qwen3_5.py: attn-quant override target file missing (SGLang restructured/renamed?)")
else:
    s = open(f).read()
    old = '            if quant_config and quant_config.get_name() == "modelopt_fp4"'
    new = '            if False  # [patch]: let is_layer_excluded() decide (allow quantized attention)'
    n = s.count(old)
    if n:
        open(f, "w").write(s.replace(old, new))
        print(f">>> qwen3_5 attn-quant patch: APPLIED ({n} occurrence(s))")
    elif new.strip() in s:
        print(">>> qwen3_5 attn-quant patch: already applied")
    else:
        print("ANCHOR-DRIFT: qwen3_5.py: attn-quant override (SGLang version drift; re-check anchor)")
PATCH_QWEN35_ATTN_QUANT_EOF

# ─────────────────────────────────────────────────────────────────────────────
# Patch qwen3_5.py: load the baked FP8 KV scales (QUALITY, mirrors the Llama-4
# KV-scale patch A+B above, applied to qwen3_5).
#
# A modelopt-NVFP4 Qwen3.5 checkpoint with quantized attention bakes per-layer FP8
# KV scales (full-attn layers only: ...self_attn.k_proj.k_scale / ...v_proj.v_scale,
# F32 scalars). Two gaps in qwen3_5.py drop them -> SGLang logs "Using FP8 KV cache
# but no scaling factors provided. Defaulting to scaling factors of 1.0" and the
# flashinfer attn backend uses 1.0 (baked scales are ~0.01-0.04 -> 25-80x off, a
# real precision loss). NOT a load-blocker. Relevant only for checkpoints that
# quantize attention ( uniform-W4A4); NVIDIA MoE-only NVFP4 has no baked KV
# scales so both edits are inert there. No model-name gate (qwen3_5.py is imported
# only for this arch).
#   A) RadixAttention built WITHOUT quant_config -> FP8-KV quant method never runs
#      -> no k_scale/v_scale params. Fix: pass quant_config.
#   B) load_weights skips ...k_proj.k_scale/...v_proj.v_scale via ignore_suffixes and
#      never calls maybe_remap_kv_scale_name. Fix: remap onto the attn.k_scale/
#      v_scale params (which A registers) BEFORE the ignore-skip. Both needed.
# ─────────────────────────────────────────────────────────────────────────────
python3 - <<'PATCH_QWEN35_KVSCALE_EOF'
import os
f = "/usr/local/lib/python3.12/dist-packages/sglang/srt/models/qwen3_5.py"
if not os.path.exists(f):
    print("ANCHOR-DRIFT: qwen3_5.py: kv-scale target file missing (SGLang restructured/renamed?)")
else:
    s = open(f).read()
    edits = [
        ("A-radixattn-quant_config", "# [patch]  A: RadixAttention quant_config",
"""            num_kv_heads=self.num_kv_heads,
            layer_id=layer_id,
            prefix=f"{prefix}.attn",
        )""",
"""            num_kv_heads=self.num_kv_heads,
            layer_id=layer_id,
            quant_config=quant_config,  # [patch]  A: RadixAttention quant_config
            prefix=f"{prefix}.attn",
        )"""),
        ("B-kv-scale-remap", "# [patch]  B: kv-scale remap",
"""            if ".self_attn." in name:
                name = name.replace(".self_attn", "")
""",
"""            if ".self_attn." in name:
                name = name.replace(".self_attn", "")
            # [patch]  B: kv-scale remap -- load the baked FP8 KV scales.
            # qwen3_5 strips .self_attn (line above) then fuses k_proj/v_proj ->
            # qkv_proj in the stacked loop below, so the stock
            # maybe_remap_kv_scale_name (which keys off .self_attn.) never matches
            # and falls to a wrong generic branch. Map k_proj.k_scale/v_proj.v_scale
            # onto the RadixAttention attn.k_scale/attn.v_scale params that patch A
            # registers, BEFORE the fusion rename, so FP8 KV uses the calibrated
            # scale instead of the 1.0 fallback. Inert if A did not register the
            # param (remapped name not in params_dict -> name left unchanged ->
            # ignore-skipped as before).
            if name.endswith(".k_scale") or name.endswith(".v_scale"):
                _rm = name.replace(".k_proj.k_scale", ".attn.k_scale").replace(
                    ".v_proj.v_scale", ".attn.v_scale"
                )
                if _rm in params_dict:
                    # Load the scale DIRECTLY here and skip the rest of the loop. Just
                    # renaming (name = _rm) is not enough: the non-stacked else-branch
                    # below does not copy a remapped k/v_scale into its param, so the
                    # value stayed at the 1.0 fallback. Mirror the copy the Llama-4 KV
                    # patch B does explicitly.
                    _p = params_dict[_rm]
                    _wl = getattr(_p, "weight_loader", None)
                    if _wl is not None:
                        _wl(_p, loaded_weight)
                    else:
                        _p.data.copy_(loaded_weight.to(_p.dtype))
                    loaded_params.add(_rm)
                    continue
"""),
    ]
    for tag, marker, old, new in edits:
        if marker in s:
            print(f">>> qwen3_5 kv-scale patch {tag}: already applied")
        else:
            n = s.count(old)
            if n:
                s = s.replace(old, new)
                print(f">>> qwen3_5 kv-scale patch {tag}: APPLIED ({n} occurrence(s))")
            else:
                print(f"ANCHOR-DRIFT: qwen3_5.py: kv-scale {tag} (SGLang version drift; re-check anchor)")
    open(f, "w").write(s)
PATCH_QWEN35_KVSCALE_EOF

# ─────────────────────────────────────────────────────────────────────────────
# Patch linear.py: NVFP4 scalar-scale loading for fused/merged linears.
# Upstream fix PR #29151 / commit cfc3d0555e ("Fix ModelOpt NVFP4 scalar scales for
# merged linears"), merged 2026-07-13, MAIN-ONLY (not in the 0.5.15 image).
# MergedColumnParallelLinear / QKVParallelLinear weight_loader_v2 hardcode shard_id=0
# for the PerTensorScaleParameter (NVFP4 weight_scale_2 / input_scale scalars) when
# loaded_shard_id is None OR a tuple, leaving the other logical slots UNINITIALIZED;
# the shared ModelOpt NVFP4 path then .max()es over the slots, so a garbage slot
# becomes the runtime global alpha -> catastrophic dequant -> word-salad. Worst on
# GDN-hybrid models whose fused linear_attn.in_proj_qkv loads via a TUPLE shard_id
# (Ornith-1.0-9B: 24 GDN layers dominate -> total garbage). Fix: fill EVERY logical
# slot with the scalar. INERT for non-NVFP4 checkpoints (PerTensorScaleParameter path
# only taken for modelopt_fp4 scalar scales).
# ─────────────────────────────────────────────────────────────────────────────
python3 - <<'PATCH_LINEAR_NVFP4SCALE_EOF'
import os
f = "/usr/local/lib/python3.12/dist-packages/sglang/srt/layers/linear.py"
if not os.path.exists(f):
    print("ANCHOR-DRIFT: linear.py: nvfp4-scale target file missing (SGLang restructured/renamed?)")
else:
    s = open(f).read()
    edits = [
        ("D-merged-col-scalar-scale", "# [patch]: PR #29151 -- fill every logical slot",
"""            if isinstance(param, PerTensorScaleParameter):
                param.load_merged_column_weight(
                    loaded_weight=loaded_weight,
                    shard_id=0,
                    tp_rank=self.tp_rank,
                    tp_size=self.tp_size,
                )
                return
""",
"""            if isinstance(param, PerTensorScaleParameter):
                # [patch]: PR #29151 -- fill every logical slot with the
                # scalar NVFP4 scale instead of only slot 0, else the unfilled slots
                # feed garbage into the .max() global alpha (fused merged linears,
                # e.g. GDN in_proj_qkv with tuple shard_id).
                if loaded_weight.numel() != 1:
                    raise ValueError(
                        "Expected scalar scale for fused-in-checkpoint "
                        "merged-column checkpoint load, got shape "
                        f"{tuple(loaded_weight.shape)}"
                    )
                if loaded_shard_id is None:
                    shard_ids = range(param.data.shape[0])
                else:
                    shard_ids = loaded_shard_id
                for shard_id in shard_ids:
                    param.load_merged_column_weight(
                        loaded_weight=loaded_weight,
                        shard_id=shard_id,
                        tp_rank=self.tp_rank,
                        tp_size=self.tp_size,
                    )
                return
"""),
        ("D-qkv-scalar-scale", "# [patch]: PR #29151 -- fill all q/k/v slots",
"""            if isinstance(param, PerTensorScaleParameter):
                param.load_qkv_weight(loaded_weight=loaded_weight, shard_id=0)
                return
""",
"""            if isinstance(param, PerTensorScaleParameter):
                # [patch]: PR #29151 -- fill all q/k/v slots with the scalar scale
                if loaded_weight.numel() != 1:
                    raise ValueError(
                        "Expected scalar scale for fused-in-checkpoint QKV "
                        "checkpoint load when loaded_shard_id is None, got "
                        f"shape {tuple(loaded_weight.shape)}"
                    )
                for shard_id in param.qkv_idxs:
                    param.load_qkv_weight(
                        loaded_weight=loaded_weight, shard_id=shard_id
                    )
                return
"""),
    ]
    for tag, marker, old, new in edits:
        if marker in s:
            print(f">>> linear.py nvfp4-scale patch {tag}: already applied")
        else:
            n = s.count(old)
            if n:
                s = s.replace(old, new)
                print(f">>> linear.py nvfp4-scale patch {tag}: APPLIED ({n} occurrence(s))")
            else:
                print(f"ANCHOR-DRIFT: linear.py: nvfp4-scale {tag} (SGLang version drift; re-check anchor)")
    open(f, "w").write(s)
PATCH_LINEAR_NVFP4SCALE_EOF

# ─────────────────────────────────────────────────────────────────────────────
# Patch compressed_tensors/utils.py: keep VLM vision tower + multimodal projector
# in BF16 for compressed-tensors NVFP4 checkpoints (e.g. Mistral3/Pixtral such as
# RecViking/Mistral-Medium-3.5-128B-NVFP4).
#
# These checkpoints quantize ONLY the text transformer and list the entire
# vision_tower + multi_modal_projector in the `ignore` list (→ BF16). But
# should_ignore_layer() compares the checkpoint `ignore` names against the RUNTIME
# module names, and SGLang's Mistral3/Pixtral loader renames four families on load:
#   q/k/v_proj            → qkv_proj         (fusion; pixtral.py stacked_params_mapping)
#   gate/up_proj          → gate_up_proj     (fusion)
#   attention.o_proj      → attention.proj   (rename; pixtral.py)
#   model.vision_tower.X          → vision_tower.X           (model. strip; mistral.py/llava.py)
#   model.multi_modal_projector.X → multi_modal_projector.X  (model. strip; mistral.py/llava.py)
# (the inner Mistral3/Llava model mounts vision_tower + projector WITHOUT the
#  checkpoint's leading "model." — so the runtime prefix must be toggled to match)
# AND Pixtral/Mistral3ForConditionalGeneration define NO packed_modules_mapping, so
# the fused-decomposition branch never runs (fused_mapping is empty). Net effect:
# the vision tower is quantized to NVFP4 and crashes at warmup:
#   ValueError: mm_fp4 accepts 2d tensors, got torch.Size([1, 16, 832]) ...
# Upstream PR #24929 (OPEN) only fixes the unrelated ".linear" suffix variant.
#
# Fix: wrap should_ignore_layer to (1) inject the canonical fused mapping so the
# vision qkv_proj/gate_up_proj decompose to the checkpoint's unfused names, and
# (2) also test the inverse-remapped checkpoint-namespace candidates (model.-prefix
# for the projector, o_proj for attention.proj). The wrapper appends to utils.py so
# the `from .utils import should_ignore_layer` in compressed_tensors.py (executed
# later) binds to the patched function. The try/except keeps non-VLM compressed-
# tensors models (asymmetric fused-shard ignore) on the original behaviour.
# Verified at the matcher level against the real 340-entry ignore list (all vision +
# projector linears → BF16, all text linears → quantized).
# ─────────────────────────────────────────────────────────────────────────────
python3 - <<'PATCH_VLM_IGNORE_EOF'
import pathlib
p = pathlib.Path(
    "/usr/local/lib/python3.12/dist-packages/sglang/srt/layers/quantization/compressed_tensors/utils.py"
)
if not p.exists():
    print("ANCHOR-DRIFT: compressed_tensors/utils.py: _sgl_vlm_ignore_fix target file missing (SGLang restructured/renamed?)")
else:
    src = p.read_text()
    marker = "# [patch] _sgl_vlm_ignore_fix"
    block = '''

# [patch] _sgl_vlm_ignore_fix — keep VLM vision tower + multimodal projector in BF16.
# See sglang_launch.sh for the full rationale (4 weight-loader name remaps + missing
# packed_modules_mapping cause compressed-tensors NVFP4 VLMs to quantize the vision
# tower -> "mm_fp4 accepts 2d tensors" crash). Wrap should_ignore_layer to inject the
# canonical fused mapping and test inverse-remapped checkpoint-namespace candidates.
_sgl_orig_should_ignore_layer = should_ignore_layer
_sgl_vlm_fused_default = {
    "qkv_proj": ["q_proj", "k_proj", "v_proj"],
    "gate_up_proj": ["gate_proj", "up_proj"],
}


def _sgl_vlm_ignore_candidates(rt):
    base = {rt}
    # SGLang's Mistral3/Llava loader strips a leading "model." from the inner
    # multimodal submodels (vision_tower, multi_modal_projector) — they are mounted
    # at "vision_tower." / "multi_modal_projector." (no "model.") while the checkpoint
    # ignore list keeps the "model." prefix. Toggle it so names line up either way.
    if rt.startswith("model."):
        base.add(rt[len("model."):])
    else:
        base.add("model." + rt)
    names = set(base)
    for n in base:
        if n.endswith(".attention.proj"):  # vision attn output: runtime proj <- ckpt o_proj
            names.add(n[: -len(".attention.proj")] + ".attention.o_proj")
    return names


def should_ignore_layer(layer_name, ignore=tuple(), fused_mapping=None):
    base_fm = fused_mapping if fused_mapping else {}
    fm = dict(base_fm)
    if layer_name:
        _proj = layer_name.rsplit(".", 1)[-1]
        if _proj in _sgl_vlm_fused_default and _proj not in fm:
            fm[_proj] = _sgl_vlm_fused_default[_proj]
    try:
        return any(
            _sgl_orig_should_ignore_layer(c, ignore=ignore, fused_mapping=fm)
            for c in _sgl_vlm_ignore_candidates(layer_name or "")
        )
    except ValueError:  # mixed-scheme fused group with injected mapping -> defer to original
        return _sgl_orig_should_ignore_layer(
            layer_name, ignore=ignore, fused_mapping=base_fm
        )
'''
    if marker in src:
        print("compressed_tensors/utils.py: VLM ignore fix already patched, skipping")
    elif "def should_ignore_layer(" not in src:
        print("ANCHOR-DRIFT: compressed_tensors/utils.py: _sgl_vlm_ignore_fix should_ignore_layer (SGLang version drift; re-check anchor)")
    else:
        p.write_text(src + block)
        print("Patched compressed_tensors/utils.py: VLM vision/projector ignore-list fix")
PATCH_VLM_IGNORE_EOF

# ─────────────────────────────────────────────────────────────────────────────
# Patch NemotronH VL/Omni wrapper MoE routing — upstream PR #25024 (OPEN, not in
# v0.5.13), rebased onto v0.5.13. Three small Python edits across 3 files.
#
# Symptom (nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-NVFP4 on SM121/GB10):
#   cutlass_moe.py cutlass_moe_fp4 → AssertionError "mismatch in expected `n`"
#   (nx2_w1 == intermediate_size_per_partition * 2) during flashinfer autotune.
# Cause: the NemotronH MoE defaults hook only matched bare NemotronHForCausalLM, so
# the VL/Omni wrapper archs (NemotronH_Nano_VL_V2 / _Nano_Omni_Reasoning_V3) bypassed
# it. Their LLM sub-config nests under hf_config.llm_config (not top-level / text_config),
# so the MoE-config resolution was skipped → backend stayed AUTO → fell through to the
# sm_100-only cutlass_moe_fp4 with a mismatched intermediate size. Even an explicit
# moe_runner_backend=flashinfer_cutlass in the profile doesn't fix it — the hook also
# does the llm_config field resolution this needs.
# Fix (PR #25024): (1) server_args dispatch list includes the wrapper archs;
# (2) NemotronH overrides read fields from hf_config.llm_config for wrappers;
# (3) scheduler.init_moe_gemm_config resolves the LLM sub-config via hf_text_config.
# Drop this block once PR #25024 lands in a release tag we use (the grep guards make it
# a safe no-op if the targets are already changed, e.g. on a future baked image).
#
# RE-ANCHORED 2026-07-16: sub-patch (1) originally targeted the standalone
# arg_groups/nemotron_h_hook.py, which no longer exists on this image — upstream
# ABSORBED it into arg_groups/overrides.py::_nemotron_h_overrides (module docstring:
# "absorbed from the retired arg_groups/nemotron_h_hook.py"). BUT that absorption did
# NOT carry over the wrapper-arch fix: the @_register_for(...) decorator on
# _nemotron_h_overrides still only lists the two bare NemotronH archs, so the VL/Omni
# wrappers still dispatch to NO override function at all (verified: zero hits for
# NemotronH_Nano_VL_V2 / _Nano_Omni_Reasoning_V3 anywhere in overrides.py/server_args.py/
# scheduler.py on this image) — same bug, new home. Sub-patch (1) below now targets
# overrides.py: extends the decorator to the wrapper archs AND keeps the llm_config
# field resolution for mlp_hidden_act. Sub-patches (2)/(3) are unchanged (still apply).
# ─────────────────────────────────────────────────────────────────────────────
python3 - <<'PATCH_NEMOTRONH_OMNI_WRAPPER_EOF'
import pathlib
DIST = "/usr/local/lib/python3.12/dist-packages/sglang/srt"
marker = "# [patch] _sgl_nemotronh_omni_wrapper_"

# --- 1) arg_groups/overrides.py: _nemotron_h_overrides (formerly nemotron_h_hook.py) ---
p = pathlib.Path(DIST + "/arg_groups/overrides.py")
if not p.exists():
    print("ANCHOR-DRIFT: overrides.py: _sgl_nemotronh_omni_wrapper_ target file missing (SGLang restructured/renamed?)")
else:
    s = p.read_text()
    old1a = '@_register_for("NemotronHForCausalLM", "NemotronHPuzzleForCausalLM")\n'
    new1a = ('@_register_for(\n'
             '    "NemotronHForCausalLM",\n'
             '    "NemotronHPuzzleForCausalLM",\n'
             '    "NemotronH_Nano_VL_V2",\n'
             '    "NemotronH_Nano_Omni_Reasoning_V3",\n'
             ')\n')
    old1b = ('    model_arch = hf_config.architectures[0]\n'
             '    model_config = server_args.get_model_config()\n'
             '    overrides: Dict[str, Any] = {}\n'
             '\n'
             '    is_modelopt = model_config.quantization in [\n')
    new1b = ('    ' + marker + ' (PR #25024, re-anchored 2026-07-16 onto\n'
             '    # upstream-absorbed overrides.py::_nemotron_h_overrides; the decorator above\n'
             '    # was extended to also dispatch the VL/Omni wrapper archs, which upstream\n'
             '    # never registered here -> they fell through with NO overrides at all.\n'
             '    # NemotronH config fields live on inner llm_config for the wrappers, on\n'
             '    # hf_config for standalone.\n'
             '    model_arch = hf_config.architectures[0]\n'
             '    model_config = server_args.get_model_config()\n'
             '    nemotron_h_cfg = getattr(model_config.hf_config, "llm_config", model_config.hf_config)\n'
             '    overrides: Dict[str, Any] = {}\n'
             '\n'
             '    is_modelopt = model_config.quantization in [\n')
    old1c = '        assert model_config.hf_config.mlp_hidden_act == "relu2"\n'
    new1c = '        assert nemotron_h_cfg.mlp_hidden_act == "relu2"\n'
    if marker in s:
        print("overrides.py: nemotronh_omni_wrapper already patched, skipping")
    elif old1a not in s or old1b not in s or old1c not in s:
        missing = [n for n, v in (("decorator", old1a), ("body-header", old1b), ("assert", old1c)) if v not in s]
        print("ANCHOR-DRIFT: overrides.py: _sgl_nemotronh_omni_wrapper_ (SGLang version drift; missing:", missing, ")")
    else:
        s = s.replace(old1a, new1a, 1).replace(old1b, new1b, 1).replace(old1c, new1c, 1)
        p.write_text(s)
        print("Patched overrides.py: NemotronH VL/Omni wrapper archs registered + llm_config field resolution")

# --- 2) managers/scheduler.py: init_moe_gemm_config via hf_text_config ---
p = pathlib.Path(DIST + "/managers/scheduler.py")
if not p.exists():
    print("ANCHOR-DRIFT: scheduler.py: _sgl_nemotronh_omni_wrapper_ target file missing (SGLang restructured/renamed?)")
else:
    s = p.read_text()
    old2 = ('        # For the MM models, check the text_config for MoE settings\n'
            '        config_to_check = getattr(\n'
            '            self.model_config.hf_config, "text_config", self.model_config.hf_config\n'
            '        )\n')
    new2 = ('        ' + marker + ' (PR #25024) — resolve the LLM sub-config via\n'
            '        # hf_text_config so NemotronH VL/Omni wrappers (nest under llm_config) are\n'
            '        # not skipped (else initialize_moe_config is skipped -> MoE backend AUTO).\n'
            '        config_to_check = self.model_config.hf_text_config\n')
    if marker in s:
        print("scheduler.py: already patched, skipping")
    elif old2 not in s:
        print("ANCHOR-DRIFT: scheduler.py: _sgl_nemotronh_omni_wrapper_ init_moe_gemm_config (SGLang version drift; re-check anchor)")
    else:
        p.write_text(s.replace(old2, new2, 1))
        print("Patched scheduler.py: init_moe_gemm_config uses hf_text_config")

# --- 3) server_args.py: add wrapper archs to the NemotronH dispatch ---
p = pathlib.Path(DIST + "/server_args.py")
if not p.exists():
    print("ANCHOR-DRIFT: server_args.py: _sgl_nemotronh_omni_wrapper_ target file missing (SGLang restructured/renamed?)")
else:
    s = p.read_text()
    old3 = '        elif model_arch in ["NemotronHForCausalLM", "NemotronHPuzzleForCausalLM"]:\n'
    new3 = ('        ' + marker.lstrip() + ' (PR #25024): add VL/Omni wrapper archs\n'
            '        elif model_arch in ["NemotronHForCausalLM", "NemotronHPuzzleForCausalLM", "NemotronH_Nano_VL_V2", "NemotronH_Nano_Omni_Reasoning_V3"]:\n')
    if marker in s:
        print("server_args.py: NemotronH dispatch already patched, skipping")
    elif old3 not in s:
        print("ANCHOR-DRIFT: server_args.py: _sgl_nemotronh_omni_wrapper_ NemotronH dispatch (SGLang version drift; re-check anchor)")
    else:
        p.write_text(s.replace(old3, new3, 1))
        print("Patched server_args.py: NemotronH dispatch includes VL/Omni wrappers")
PATCH_NEMOTRONH_OMNI_WRAPPER_EOF

# Prime ARP table on the QSFP link before NCCL tries to connect.
# Without this, the first TCP SYNs get dropped until ARP resolves,
# causing ~230s delay in "Init torch distributed".
# Full-mesh: every node pings ALL other nodes so NCCL's ring/tree
# topology can communicate immediately over any path.
IFS=',' read -ra peers <<< "$QSFP_PEER_IPS"
pids=()
for peer in "${peers[@]}"; do
  (
    echo "Waiting for QSFP peer ${peer} ..."
    while true; do
      ping -c3 -W1 "$peer" 2>&1 | sed "s/^/[${peer}] /"
      [[ ${PIPESTATUS[0]} -eq 0 ]] && break
      sleep 1
    done
    echo "QSFP peer ${peer} reachable."
  ) &
  pids+=($!)
done
for pid in "${pids[@]}"; do wait "$pid"; done
echo "All ${#peers[@]} QSFP peers reachable."

# Patch CUTLASS BlockScaledMmaOp to support SM121 (DGX Spark GB10) for FP4 operations.
# Upstream CUTLASS restricts FP4 tensor ops to sm_100a only (issue NVIDIA/cutlass#2800).
# SM121 has native FP4 Tensor Core support but is not in admissible_archs → the JIT-compiled
# nvfp4_blockwise_moe kernel falls back to an incompatible code path → device-side assert.
# Fix: add sm_120a + sm_121a to admissible_archs in both CUTLASS DSL copies.
# External validation: BTankut/dgx-spark-sglang-moe-configs achieved 356 TFLOPS NVFP4 on GB10.
_mma_existing_any=false
for mma_py in \
  /usr/local/lib/python3.12/dist-packages/nvidia_cutlass_dsl/python_packages/cutlass/cute/nvgpu/tcgen05/mma.py \
  /usr/local/lib/python3.12/dist-packages/flashinfer/data/cutlass/python/CuTeDSL/cutlass/cute/nvgpu/tcgen05/mma.py; do
  if [ -f "$mma_py" ]; then
    _mma_existing_any=true
    if grep -q 'admissible_archs = \[' "$mma_py" 2>/dev/null; then
      if ! grep -q 'sm_121a' "$mma_py" 2>/dev/null; then
        sed -i 's/Arch\.sm_100a,/Arch.sm_100a, Arch.sm_120a, Arch.sm_121a,/' "$mma_py"
        echo "Patched $(basename $(dirname $(dirname $(dirname "$mma_py"))))/mma.py: added sm_120a + sm_121a to BlockScaledMmaOp.admissible_archs"
      else
        echo "$(basename "$mma_py"): sm_121a already present"
      fi
    else
      echo "ANCHOR-DRIFT: $(basename "$mma_py"): BlockScaledMmaOp admissible_archs anchor missing (CUTLASS version drift; re-check anchor)"
    fi
  fi
done
if [ "$_mma_existing_any" = false ]; then
  echo "ANCHOR-DRIFT: cutlass mma.py: neither CUTLASS DSL copy (nvidia_cutlass_dsl / flashinfer bundled) found (dep restructured/renamed?)"
fi

# NOTE: nvfp4_blockwise_moe.cuh SM121 tile fix was attempted but cannot be solved
# via runtime patching. The CUTLASS FP4 grouped GEMM kernel requires SM121-specific
# sgl-kernel build with proper TMA tile shapes — not achievable by editing the .cuh
# source at startup. See CUTLASS_NVFP4_SM121_PRD.md for full analysis.
# For NVFP4 models on SM121: use flashinfer_cutlass MoE runner (avoids cutlass_moe_fp4).

# ─────────────────────────────────────────────────────────────────────────────
# Patch sglang Transformers fallback: add Gemma-4 MoE config attribute names.
#
# Gemma-4 (Gemma4ForConditionalGeneration) has no native SGLang implementation
# and falls through to the Transformers backend. The MoEMixin.recursive_replace()
# method looks up top_k via ("num_experts_per_tok", "top_k") — Gemma-4 uses
# "top_k_experts" instead → AssertionError: Cannot determine top_k from config.
#
# Fix: add "top_k_experts" to the _getattr_first lookup tuple on line 1197.
#
# Note: SGLang v0.5.11 has native Gemma-4 support (PR #21952 + follow-ups
# #22079, #24048, #22842), so this patch is a no-op there (the grep guard
# inside makes it idempotent — pattern not found → skip).
# ─────────────────────────────────────────────────────────────────────────────
python3 - <<'PATCH_TRANSFORMERS_TOPK_EOF'
import pathlib
p = pathlib.Path("/usr/local/lib/python3.12/dist-packages/sglang/srt/models/transformers.py")
if not p.exists():
    print("ANCHOR-DRIFT: transformers.py: _sgl_gemma4_topk_ target file missing (SGLang restructured/renamed?)")
else:
    src = p.read_text()
    marker = "# [patch] _sgl_gemma4_topk_"
    old = '("num_experts_per_tok", "top_k")'
    new = '("num_experts_per_tok", "top_k", "top_k_experts")'
    if marker in src:
        print("sglang/srt/models/transformers.py: already patched (top_k_experts), skipping")
    elif old not in src:
        print("ANCHOR-DRIFT: transformers.py: _sgl_gemma4_topk_ (SGLang version drift; re-check anchor)")
    else:
        # Replace the tuple inline. Marker goes on a NEW line above to avoid
        # breaking the closing paren of _getattr_first(...).
        src = src.replace(old, new, 1)
        # Insert marker as a comment on the line before the patched line
        src = src.replace(
            "top_k = _getattr_first(text_config, " + new,
            marker + "\n        top_k = _getattr_first(text_config, " + new,
            1,
        )
        p.write_text(src)
        print("Patched sglang/srt/models/transformers.py: added top_k_experts to MoE config lookup")
PATCH_TRANSFORMERS_TOPK_EOF

# ─────────────────────────────────────────────────────────────────────────────
# Patch apply_fp8_linear: coerce a non-half/bf16 input to bf16 before the fp8 GEMM.
#
# Symptom (RecViking/Mistral-Medium-3.5-128B-NVFP4 + EAGLE MTP, fp8 draft, SM121):
#   eagle_draft_cuda_graph_runner → mistral_eagle.py:116 self.fc(torch.cat((embed,
#   target_hidden), -1)) → fp8.py Fp8LinearMethod.apply → fp8_utils.apply_fp8_linear
#   → fp8_scaled_mm(..., out_dtype=input.dtype) → "RuntimeError: out_dtype must be
#   Half or BFloat16".
#
# Why: the EAGLE fusion fc concatenates the draft's embedding with the NVFP4
#   target's previous hidden state. That fused tensor's dtype is neither fp16 nor
#   bf16 (float32 in the captured graph), and apply_fp8_linear passes
#   out_dtype=input.dtype straight into the sgl_kernel, which only accepts
#   half/bf16. NOT fixable via --speculative-draft-model-quantization unquant:
#   SGLang normalizes "unquant" to None → auto-detects the Mistral-native draft's
#   params.json qformat_weight fp8_e4m3 → the draft loads fp8 regardless.
#
# ROOT FIX is elsewhere: --dtype bfloat16 (profile `dtype: bfloat16` → SGLANG_MODEL_DTYPE)
#   stops the draft's fp32→fp16 fallback so the fc input is bf16 in the first place
#   (SGLang cookbook Mistral-Medium-3.5 §3.3). This source patch is a BELT-AND-
#   SUSPENDERS safety net behind that, kept deliberately: cast input to bf16 at the
#   top of apply_fp8_linear when its dtype is not already half/bf16. The input is
#   fp8-quantized immediately after anyway, so the f32→bf16 downcast is lossless in
#   effect. Harmless for every normal fp8 linear (their input is already half/bf16
#   → branch not taken). Idempotent via marker.
# ─────────────────────────────────────────────────────────────────────────────
python3 - <<'PATCH_FP8_OUT_DTYPE_EOF'
import pathlib
p = pathlib.Path("/usr/local/lib/python3.12/dist-packages/sglang/srt/layers/quantization/fp8_utils.py")
if not p.exists():
    print("ANCHOR-DRIFT: fp8_utils.py: _sgl_eagle_fp8_out_dtype_fix target file missing (SGLang restructured/renamed?)")
else:
    src = p.read_text()
    marker = "# [patch] _sgl_eagle_fp8_out_dtype_fix"
    anchor = "    # View input as 2D matrix for fp8 methods\n    input_2d = input.view(-1, input.shape[-1])"
    if marker in src:
        print("sglang fp8_utils.py: out_dtype fix already patched, skipping")
    elif anchor not in src:
        print("ANCHOR-DRIFT: fp8_utils.py: _sgl_eagle_fp8_out_dtype_fix (SGLang version drift; re-check anchor)")
    else:
        inject = (
            "    " + marker + " — EAGLE fc fuses NVFP4 target hidden states → input\n"
            "    # dtype is neither fp16 nor bf16; fp8_scaled_mm out_dtype=input.dtype\n"
            "    # then asserts Half/BFloat16. Cast to bf16 (input is fp8-quantized next).\n"
            "    if input.dtype not in (torch.float16, torch.bfloat16):\n"
            "        input = input.to(torch.bfloat16)\n"
            + anchor
        )
        src = src.replace(anchor, inject, 1)
        p.write_text(src)
        print("Patched sglang fp8_utils.py: apply_fp8_linear casts non-half/bf16 input to bf16")
PATCH_FP8_OUT_DTYPE_EOF

# ─────────────────────────────────────────────────────────────────────────────
# Patch flashinfer.jit.cpp_ext.get_cuda_version: avoid subprocess from inside a
# torch.compile/dynamo trace.
#
# Symptom (GLM-4.7-NVFP4 EP=1, piecewise CUDA graphs + fi_cudnn FP4):
#   flashinfer/quantization/fp4_quantization.py:170 build_and_load()
#   → gen_fp4_quantization_sm120f_module
#   → flashinfer/jit/cpp_ext.py:91 is_cuda_version_at_least("12.8")
#   → cpp_ext.py:73 subprocess.check_output([nvcc, "--version"])
#   → subprocess/threading.Lock() under torch/_dynamo/polyfills:392 getattr_and_trace
#   → child process sigquit → pod restart → startup_crash
#
# Why: the JIT build is triggered on the first forward pass, which for piecewise
# CUDA graphs happens inside a torch.compile trace. Dynamo can't polyfill
# subprocess.Popen (it does fork/threading internals), so the call blows up even
# though nvcc is present and works fine from a normal shell.
#
# Fix: short-circuit get_cuda_version() with torch.version.cuda, which is always
# available at import time and matches what nvcc reports for the same install.
# The original function already had this as a fallback path on exception — we
# just promote it to run first. This keeps the subprocess path as the fallback
# for pytorch builds without a CUDA version (none of ours).
# ─────────────────────────────────────────────────────────────────────────────
python3 - <<'PATCH_FI_CUDA_VER_EOF'
import pathlib
p = pathlib.Path("/usr/local/lib/python3.12/dist-packages/flashinfer/jit/cpp_ext.py")
if not p.exists():
    print("ANCHOR-DRIFT: flashinfer/jit/cpp_ext.py: _fi_cuda_ver_subprocess_bypass_ target file missing (SGLang restructured/renamed? flashinfer dep)")
else:
    src = p.read_text()
    marker = "# [patch] _fi_cuda_ver_subprocess_bypass_"
    if marker in src:
        print("flashinfer/jit/cpp_ext.py: get_cuda_version already patched, skipping")
    else:
        target = (
            "@functools.cache\n"
            "def get_cuda_version() -> Version:\n"
            "    # Try to query nvcc for CUDA version; if nvcc is unavailable, "
            "fall back to torch.version.cuda\n"
            "    try:"
        )
        replacement = (
            "@functools.cache\n"
            "def get_cuda_version() -> Version:\n"
            "    " + marker + "\n"
            "    # Short-circuit with torch.version.cuda to avoid spawning a `nvcc --version`\n"
            "    # subprocess from inside a torch.compile/dynamo trace context. See the\n"
            "    # sglang_launch.sh header block above this patch for the full rationale.\n"
            "    if torch.version.cuda is not None:\n"
            "        return Version(torch.version.cuda)\n"
            "    # Try to query nvcc for CUDA version; if nvcc is unavailable, "
            "fall back to torch.version.cuda\n"
            "    try:"
        )
        if target in src:
            p.write_text(src.replace(target, replacement, 1))
            print("Patched flashinfer/jit/cpp_ext.py:get_cuda_version to bypass subprocess")
        else:
            print("ANCHOR-DRIFT: flashinfer/jit/cpp_ext.py: _fi_cuda_ver_subprocess_bypass_ (SGLang version drift; re-check anchor)")
PATCH_FI_CUDA_VER_EOF

# ─────────────────────────────────────────────────────────────────────────────
# Patch flashinfer.quantization.fp4_quantization: register `fp4_quantize` as
# an opaque leaf op via `torch.compiler.allow_in_graph`, so dynamo emits a
# single FX graph node for the call without tracing its body.
#
# Symptom chain (GLM-4.7-NVFP4 EP=1, piecewise CUDA graphs):
#   sglang/srt/compilation/compile.py:183 _ensure_compiled
#   → torch._dynamo compile_wrapper
#   → sglang/.../modelopt_quant.py:1482 fp4_quantize(x, layer.input_scale_inv)
#   → flashinfer/quantization/fp4_quantization.py:700 fp4_quantize
#
# Dynamo hits a whole family of un-traceable things as it recurses into the
# FP4 path:
#   (a) get_fp4_quantization_module → JitSpec.is_aot → Path.exists → os.stat
#       → "Attempted to call function marked as skipped: posix.stat"
#   (b) fp4_quantize_sm100 (line 222) → `module.fp4_quantize(...)` is a
#       torch.autograd.Function → "Unsupported method call: Function.__call__"
#   (c) almost certainly more further in.
#
# Whack-a-mole fixes (functools.cache wrapping, pre-resolving the module
# lookup into a module-level constant, etc.) each unblock one layer and
# then hit the next. `@torch.compiler.disable` also doesn't work: sglang's
# piecewise compile path treats skipped calls as a hard error (gb0098:
# "Skip calling torch.compiler.disable()d function") rather than a normal
# graph break.
#
# Correct approach: `torch.compiler.allow_in_graph`. This tells dynamo to
# emit a single opaque call node for `fp4_quantize` in the FX graph without
# tracing through its body at all — the JIT lookup, os.stat, and
# autograd.Function.__call__ all execute at real runtime (outside any
# trace), which is completely fine for all of them.
#
# Contract for allow_in_graph: the function must take/return tensors
# (or pytrees of tensors) and must be deterministic in output dtype/shape
# given input dtype/shape. `fp4_quantize(input, global_scale)` returns
# (x_q, sf) — both tensors — and its output shapes are a deterministic
# function of input shape + sf_vec_size. Contract satisfied.
# ─────────────────────────────────────────────────────────────────────────────
python3 - <<'PATCH_FI_FP4_ALLOW_EOF'
import pathlib
p = pathlib.Path("/usr/local/lib/python3.12/dist-packages/flashinfer/quantization/fp4_quantization.py")
if not p.exists():
    print("ANCHOR-DRIFT: flashinfer/quantization/fp4_quantization.py: _fi_fp4_allow_in_graph_ target file missing (SGLang restructured/renamed? flashinfer dep)")
else:
    src = p.read_text()
    marker = "# [patch] _fi_fp4_allow_in_graph_"
    # Undo any remnants from earlier patch revisions in the same container
    # (e.g. after a crash loop re-execs sglang_launch.sh without a fresh
    # image) so the source stays clean.
    stale_const_call = "_SGLANG_FP4_MOD.fp4_quantize_sm100("
    if stale_const_call in src:
        src = src.replace(
            stale_const_call,
            'get_fp4_quantization_module(f"{major}{minor}").fp4_quantize_sm100(',
        )
    src = src.replace(
        "@torch.compiler.disable\n@flashinfer_api\ndef fp4_quantize(\n",
        "@flashinfer_api\ndef fp4_quantize(\n",
    )
    for old_marker in (
        "# [patch] _fi_fp4_cache_and_prewarm_",
        "# [patch] _fi_fp4_prewarm_const_",
        "# [patch] _fi_fp4_compiler_disable_",
    ):
        idx = src.find("\n\n# " + old_marker)
        if idx != -1:
            src = src[:idx].rstrip() + "\n"
    if marker in src:
        print("flashinfer/quantization/fp4_quantization.py: already patched (allow_in_graph), writing back only cleanup")
        p.write_text(src)
    elif "def fp4_quantize(" not in src:
        print("ANCHOR-DRIFT: flashinfer/quantization/fp4_quantization.py: _fi_fp4_allow_in_graph_ (SGLang version drift; re-check anchor)")
    else:
        # The registration must run at module import time AFTER fp4_quantize
        # has been defined. We append a trailing block that imports torch
        # and rebinds the module-level name through allow_in_graph. Rebinding
        # the name is safe because allow_in_graph returns a wrapper with the
        # exact same calling contract, and any subsequent `from flashinfer...
        # import fp4_quantize` inside sglang picks up the wrapped version.
        append_block = (
            "\n\n"
            "# " + marker + "\n"
            "# Appended by sglang_launch.sh runtime patch. Registers fp4_quantize\n"
            "# as an opaque leaf op so dynamo emits a single FX graph node for it\n"
            "# instead of tracing into the body (which hits os.stat during the JIT\n"
            "# lookup and torch.autograd.Function.__call__ inside fp4_quantize_sm100).\n"
            "# See sglang_launch.sh header for full rationale.\n"
            "try:\n"
            "    import torch as _sglang_t\n"
            "    fp4_quantize = _sglang_t.compiler.allow_in_graph(fp4_quantize)\n"
            "    import sys as _sglang_sys\n"
            "    print('[fp4_quantization] fp4_quantize registered via allow_in_graph', file=_sglang_sys.stderr)\n"
            "except Exception as _sglang_e:\n"
            "    import sys as _sglang_sys\n"
            "    print(f'[fp4_quantization] allow_in_graph registration failed: {_sglang_e}', file=_sglang_sys.stderr)\n"
        )
        p.write_text(src + append_block)
        print("Patched flashinfer/quantization/fp4_quantization.py: fp4_quantize → allow_in_graph")
PATCH_FI_FP4_ALLOW_EOF

# Version gate: warn if the container image changed — patches below may need review.
# Dev builds report __version__=0.0.0 (no setuptools-scm), so we check the image
# tag (injected as SGLANG_IMAGE env var by Ansible) instead of the Python version.
# The grep guards still prevent patching if the target code has changed.
# Current production line is the 0.5.11/0.5.12/0.5.13 -sm121 family (the old
# -sm121-dev1 dev tags were retired 2026-06-19). The grep guards on each patch
# still no-op safely if a specific target's source has moved.
SGLANG_EXPECTED_IMAGE_PATTERN="xomoxcc/dgx-spark-sglang:.*-sm121"
if [ -n "$SGLANG_IMAGE" ] && ! echo "$SGLANG_IMAGE" | grep -qE "^${SGLANG_EXPECTED_IMAGE_PATTERN}$"; then
  echo "WARNING: SGLang image does not match expected pattern ${SGLANG_EXPECTED_IMAGE_PATTERN} (got ${SGLANG_IMAGE})."
  echo "         Monkey-patches may no longer apply or may need updating."
fi

# When load_format is sharded_state, wait for the shard marker then use sharded path
model_path="$SGLANG_MODEL"
if [ "$SGLANG_LOAD_FORMAT" = "sharded_state" ]; then
  model_slug=$(echo "$SGLANG_MODEL" | sed 's|/|--|g')
  shard_suffix="sglang-TP${TP}"
  if [ -n "$EP" ] && [ "$EP" != "1" ]; then
    shard_suffix="${shard_suffix}-EP${EP}"
  fi
  if [ -n "$SGLANG_QUANTIZATION" ]; then
    shard_suffix="${shard_suffix}-${SGLANG_QUANTIZATION}"
  fi
  if [ -n "$SGLANG_MOE_RUNNER_BACKEND" ]; then
    shard_suffix="${shard_suffix}-${SGLANG_MOE_RUNNER_BACKEND}"
  fi
  sharded_path="/root/.cache/huggingface/sharded/${model_slug}-${shard_suffix}"
  marker="${sharded_path}/model.safetensors.index.json"
  echo "Waiting for sharded checkpoint at ${marker} ..."
  while [ ! -f "$marker" ]; do
    echo "  $(date '+%H:%M:%S') shard not ready yet, waiting 30s ..."
    sleep 30
  done
  model_path="$sharded_path"
  echo "Using pre-sharded model at ${model_path}"
fi

# Patch weight loading iterators to log progress per shard file.
# tqdm writes directly to sys.stderr in TP worker subprocesses — this output is
# NOT forwarded by SGLang's logger infrastructure, so it never appears in kubectl logs.
# Additionally, BAR_FORMAT lacks a trailing \n, so tqdm uses \r (carriage return)
# which is invisible in non-TTY kubectl logs.
# Fix: replace tqdm loops with logger.info() calls that go through the logging pipeline.
#
# v0.5.10: enable_multithread_load defaults to True, so the default code path is
# buffered_multi_thread_safetensors_weights_iterator (not the old single-thread one).
# We patch both to cover all cases.
WEIGHT_UTILS="/usr/local/lib/python3.12/dist-packages/sglang/srt/model_loader/weight_utils.py"
if [ ! -f "$WEIGHT_UTILS" ]; then
  echo "ANCHOR-DRIFT: weight_utils.py: safetensors tqdm->logger progress patches target file missing (SGLang restructured/renamed?)"
else
  python3 << 'PATCH_SAFETENSORS_TQDM_EOF'
import os
f = "/usr/local/lib/python3.12/dist-packages/sglang/srt/model_loader/weight_utils.py"
with open(f) as fh:
    code = fh.read()
patched = False
# Add logger import if missing
if "\nlogger = " not in code and "\nlogger=" not in code:
    code = code.replace(
        "from tqdm.auto import tqdm",
        "import logging\nfrom tqdm.auto import tqdm\nlogger = logging.getLogger(__name__)",
        1)

# --- Patch 1: single-thread safetensors_weights_iterator (v0.5.10rc0 path) ---
old_single = '''    for st_file in tqdm(
        hf_weights_files,
        desc="Loading safetensors checkpoint shards",
        disable=not enable_tqdm,
        bar_format=BAR_FORMAT,
        position=tqdm._get_free_pos(),
    ):'''
new_single = '''    _total = len(hf_weights_files)
    for _i, st_file in enumerate(hf_weights_files, 1):
        if enable_tqdm:
            logger.info(f"Loading safetensors shard {_i}/{_total}: {os.path.basename(st_file)}")'''
if old_single in code:
    code = code.replace(old_single, new_single, 1)
    patched = True
    print("Patched safetensors_weights_iterator: tqdm → logger.info")
elif "Loading safetensors shard {_i}/{_total}" in code:
    print("safetensors_weights_iterator: tqdm->logger patch already applied, skipping")
elif "def safetensors_weights_iterator(" in code:
    print("ANCHOR-DRIFT: weight_utils.py: safetensors_weights_iterator tqdm->logger anchor missing "
          "(SGLang version drift; re-check anchor) -- single-thread shard load will NOT log per-shard progress.")
else:
    print("ANCHOR-DRIFT: weight_utils.py: safetensors_weights_iterator function not found "
          "(SGLang version drift; re-check anchor)")

# --- Patch 2: buffered_multi_thread_safetensors_weights_iterator (v0.5.15) ---
# v0.5.15 stores (st_file, future) TUPLES in `pending` and does
# `st_file, future = pending.popleft()` (for the drop_cache_after_load feature).
# The old whole-block anchor (expecting `future = pending.popleft()`) silently
# MISSED on v0.5.15 -> the multi-thread NVFP4 weight load logged NOTHING (the tqdm
# bar writes \r to non-forwarded stderr; this is the default load path for the big
# NVFP4 models like glm-5.2-reap-504B). Minimal + drift-resistant fix: keep the
# (invisible) tqdm bar, add ONE logger.info per shard, anchored on the stable
# 2-line pop+result. See remote-control 2026-07-15.
old_buffered = '''                st_file, future = pending.popleft()
                state_dict = future.result()'''
new_buffered = '''                st_file, future = pending.popleft()
                state_dict = future.result()
                # [patch] _sgl_buf_iter_per_shard_log_
                if enable_tqdm:
                    logger.info(f"Loading shard (multi-thread): {os.path.basename(st_file)} ({len(state_dict)} tensors)")'''
# Idempotency (2026-07-16): key the already-applied check on a DEDICATED marker checked
# FIRST. old_buffered is a PREFIX of new_buffered, so `old_buffered in code` stays true
# after patching -> checking it first would re-inject a second logger.info on any re-run.
# The marker is also patch-specific: Patch 3 injects the same "Loading shard (multi-thread): "
# string, so keying the guard on that string would cross-contaminate between the two patches.
_buf_marker = "# [patch] _sgl_buf_iter_per_shard_log_"
if _buf_marker in code:
    print("buffered_multi_thread_safetensors_weights_iterator: per-shard logger.info already applied, skipping")
elif old_buffered in code:
    code = code.replace(old_buffered, new_buffered, 1)
    patched = True
    print("Patched buffered_multi_thread_safetensors_weights_iterator: +per-shard logger.info")
elif "def buffered_multi_thread_safetensors_weights_iterator(" in code:
    print("ANCHOR-DRIFT: weight_utils.py: buffered_multi_thread_safetensors_weights_iterator pop+result "
          "anchor missing (SGLang version drift; re-check anchor) -- the DEFAULT multi-thread NVFP4 weight "
          "load will NOT log per-shard progress (silent load, e.g. glm-5.2-reap-504B).")
else:
    print("ANCHOR-DRIFT: weight_utils.py: buffered_multi_thread_safetensors_weights_iterator function not "
          "found (SGLang version drift; re-check anchor)")

# --- Patch 3: multi_thread_safetensors_weights_iterator (non-buffered variant) ---
# RE-ANCHORED 2026-07-16: _load_file() now returns a (st_file, result) TUPLE (same
# shape change already handled for the buffered variant in Patch 2 above), so
# `future.result()` unpacks as `st_file, state_dict = future.result()`, not a bare
# state_dict — the old single-value anchor never matched. New anchor also covers the
# drop_cache_after_load tail so that logic is not silently dropped by the patch.
_mt_marker = "# [patch] _sgl_mt_iter_per_shard_log_"
old_mt = (
    "        for future in futures_iter:\n"
    "            st_file, state_dict = future.result()\n"
    "            for name, param in state_dict.items():\n"
    "                yield name, param\n"
    "            del state_dict\n"
)
new_mt = (
    "        " + _mt_marker + "\n"
    "        _mt_total = len(hf_weights_files)\n"
    "        _mt_done = 0\n"
    "        for future in futures_iter:\n"
    "            st_file, state_dict = future.result()\n"
    "            _mt_done += 1\n"
    "            if enable_tqdm:\n"
    "                logger.info(\n"
    "                    f\"Loading shard (multi-thread): {_mt_done}/{_mt_total} \"\n"
    "                    f\"{os.path.basename(st_file)} ({len(state_dict)} tensors)\"\n"
    "                )\n"
    "            for name, param in state_dict.items():\n"
    "                yield name, param\n"
    "            del state_dict\n"
)
if _mt_marker in code:
    print("multi_thread_safetensors_weights_iterator: per-shard logger.info already applied, skipping")
elif old_mt in code:
    code = code.replace(old_mt, new_mt, 1)
    patched = True
    print("Patched multi_thread_safetensors_weights_iterator: per-shard logger.info (st_file tuple aware)")
elif "def multi_thread_safetensors_weights_iterator(" in code:
    print("ANCHOR-DRIFT: weight_utils.py: multi_thread_safetensors_weights_iterator tqdm->logger anchor "
          "missing (SGLang version drift; re-check anchor) -- that multi-thread load path will NOT log "
          "per-shard progress.")
else:
    print("ANCHOR-DRIFT: weight_utils.py: multi_thread_safetensors_weights_iterator function not found "
          "(SGLang version drift; re-check anchor)")

if patched:
    if "import os" not in code:
        code = "import os\n" + code
    with open(f, 'w') as fh:
        fh.write(code)
# else: each of the 3 sub-patches above already printed its own precise
# already-applied / ANCHOR-DRIFT status; no generic catch-all needed here.
PATCH_SAFETENSORS_TQDM_EOF
fi

# Patch ShardedStateLoader to log progress per shard file (no progress bar by default)
LOADER="/usr/local/lib/python3.12/dist-packages/sglang/srt/model_loader/loader.py"
if [ ! -f "$LOADER" ]; then
  echo "ANCHOR-DRIFT: loader.py: ShardedStateLoader progress logging target file missing (SGLang restructured/renamed?)"
elif grep -q 'for path in filepaths:' "$LOADER" 2>/dev/null; then
  sed -i 's/for path in filepaths:/for _shard_i, path in enumerate(filepaths, 1):/' "$LOADER"
  sed -i '/_shard_i, path in enumerate(filepaths/a\                logger.info(f"Loading shard {_shard_i}\/{len(filepaths)}: {os.path.basename(path)}")' "$LOADER"
  echo "Patched ShardedStateLoader for per-file progress logging"
elif grep -q 'for _shard_i, path in enumerate(filepaths' "$LOADER" 2>/dev/null; then
  echo "ShardedStateLoader progress logging: already applied, skipping"
else
  echo "ANCHOR-DRIFT: loader.py: ShardedStateLoader 'for path in filepaths:' anchor missing (SGLang version drift; re-check anchor)"
fi

# Patch moe_wna16 weight loader for EP-aware qzeros handling (SGLang 0.5.9 bug).
# Two bugs in the w13_qzeros/w2_qzeros branches:
#   1. Uses raw global expert_id (0-127) instead of local EP index (0-63)
#   2. Uses global tp_rank for TP-slice, but moe_tp_size=tp/ep — need tp_rank % moe_tp_size
# Safe no-op when ep_size=1 (identity mapping, tp_rank unchanged).
MOE_WNA16="/usr/local/lib/python3.12/dist-packages/sglang/srt/layers/quantization/moe_wna16.py"
if [ ! -f "$MOE_WNA16" ]; then
  echo "ANCHOR-DRIFT: moe_wna16.py: EP-aware qzeros remap target file missing (SGLang restructured/renamed?)"
else
  python3 << 'PATCH_QZEROS_EOF'
import sys
f = "/usr/local/lib/python3.12/dist-packages/sglang/srt/layers/quantization/moe_wna16.py"
with open(f) as fh:
    code = fh.read()
old_w13 = '''            if "w13_qzeros" in weight_name:
                tensor = loaded_weight.view(
                    layer.moe_tp_size, -1, loaded_weight.size(1)
                )[tp_rank]
                if shard_id == "w1":
                    param.data[expert_id, : shard_size // 2] = tensor
                else:
                    param.data[expert_id, shard_size // 2 :] = tensor'''
new_w13 = '''            if "w13_qzeros" in weight_name:
                _local_id = layer._map_global_expert_id_to_local_expert_id(expert_id)
                if _local_id == -1:
                    return
                _moe_tp_rank = tp_rank % layer.moe_tp_size
                tensor = loaded_weight.view(
                    layer.moe_tp_size, -1, loaded_weight.size(1)
                )[_moe_tp_rank]
                if shard_id == "w1":
                    param.data[_local_id, : shard_size // 2] = tensor
                else:
                    param.data[_local_id, shard_size // 2 :] = tensor'''
old_w2 = '''            elif "w2_qzeros" in weight_name:
                param.data[expert_id] = loaded_weight.view(
                    loaded_weight.size(0), layer.moe_tp_size, -1
                )[:, tp_rank]'''
new_w2 = '''            elif "w2_qzeros" in weight_name:
                _local_id = layer._map_global_expert_id_to_local_expert_id(expert_id)
                if _local_id == -1:
                    return
                _moe_tp_rank = tp_rank % layer.moe_tp_size
                param.data[_local_id] = loaded_weight.view(
                    loaded_weight.size(0), layer.moe_tp_size, -1
                )[:, _moe_tp_rank]'''
changed = False
if new_w13 in code:
    print("w13_qzeros: already applied, skipping")
elif old_w13 not in code:
    print("ANCHOR-DRIFT: moe_wna16.py: w13_qzeros EP-aware remap anchor missing (SGLang version drift; re-check anchor)")
    sys.exit(0)
else:
    code = code.replace(old_w13, new_w13, 1)
    changed = True
if new_w2 in code:
    print("w2_qzeros: already applied, skipping")
elif old_w2 not in code:
    print("ANCHOR-DRIFT: moe_wna16.py: w2_qzeros EP-aware remap anchor missing (SGLang version drift; re-check anchor)")
    sys.exit(0)
else:
    code = code.replace(old_w2, new_w2, 1)
    changed = True
if changed:
    with open(f, 'w') as fh:
        fh.write(code)
    print("Patched moe_wna16.py: EP-aware expert_id + tp_rank remapping for qzeros")
PATCH_QZEROS_EOF
fi

# [removed 2026-07-12] quantization/utils.py dot-boundary is_layer_skipped patch (PR #23467)
# dropped: verified present in the 0.5.14-sm121 image (_module_path_match already defined in
# layers/quantization/utils.py) — the grep guard was already a no-op. Restore from git if an
# older base image is pinned again.

# Patch modelopt_quant.py: EP-aware input_scale slicing (SGLang 0.5.9 bug).
# In process_weights_after_loading(), the fallback else-branch computes
# w13_input_scale and w2_input_scale with shape (num_experts,) but multiplies
# them with w13_weight_scale_2 / w2_weight_scale_2 which are (num_local_experts,).
# With EP=2 on MiniMax-M2.5 (256 experts): (256,) * (128,) → RuntimeError.
# The cutedsl branch has _slice_scale() but the else-branch is missing it.
# Fix: slice input_scale to local experts in the else-branch.
MODELOPT_QUANT="/usr/local/lib/python3.12/dist-packages/sglang/srt/layers/quantization/modelopt_quant.py"
if [ ! -f "$MODELOPT_QUANT" ]; then
  echo "ANCHOR-DRIFT: modelopt_quant.py: EP-aware input_scale slicing target file missing (SGLang restructured/renamed?)"
elif grep -q 'w13_input_scale\.max(dim=-1)' "$MODELOPT_QUANT" 2>/dev/null; then
  python3 << 'PATCH_MODELOPT_EOF'
import sys
f = "/usr/local/lib/python3.12/dist-packages/sglang/srt/layers/quantization/modelopt_quant.py"
with open(f) as fh:
    code = fh.read()
old = '''        else:
            w13_input_scale = layer.w13_input_scale.max(dim=-1).values.to(torch.float32)
            w2_input_scale = layer.w2_input_scale'''
new = '''        else:
            w13_input_scale = layer.w13_input_scale.max(dim=-1).values.to(torch.float32)
            w2_input_scale = layer.w2_input_scale
            # EP-aware slicing: input_scale has shape (num_experts,) but must match
            # weight_scale_2 which is (num_local_experts,). No-op when ep_size=1.
            if layer.moe_ep_size > 1:
                _ep_start = layer.moe_ep_rank * layer.num_local_experts
                _ep_end = _ep_start + layer.num_local_experts
                w13_input_scale = w13_input_scale[_ep_start:_ep_end]
                w2_input_scale = w2_input_scale[_ep_start:_ep_end]'''
_already_marker = "# EP-aware slicing: input_scale has shape"
if _already_marker in code:
    print("modelopt_quant.py else-branch: already applied, skipping")
elif old not in code:
    print("ANCHOR-DRIFT: modelopt_quant.py: EP-aware input_scale slicing else-branch anchor missing (SGLang version drift; re-check anchor)")
    sys.exit(0)
else:
    code = code.replace(old, new, 1)
    with open(f, 'w') as fh:
        fh.write(code)
    print("Patched modelopt_quant.py: EP-aware input_scale slicing in else-branch")
PATCH_MODELOPT_EOF
else
  echo "ANCHOR-DRIFT: modelopt_quant.py: EP-aware input_scale slicing outer probe missing (SGLang version drift; re-check anchor)"
fi

# Patch modelopt_quant.py: CutlassMoEParams uses layer.num_experts (global=256)
# instead of layer.num_local_experts (128 with EP=2). The cutlass_moe_fp4 forward
# function then asserts num_experts == weight expert dim, which fails because
# weights are EP-sliced to num_local_experts. Fix: use num_local_experts.
if [ ! -f "$MODELOPT_QUANT" ]; then
  echo "ANCHOR-DRIFT: modelopt_quant.py: CutlassMoEParams num_local_experts EP fix target file missing (SGLang restructured/renamed?)"
elif grep -q 'num_experts=layer.num_experts,  # global num experts' "$MODELOPT_QUANT" 2>/dev/null; then
  sed -i 's/num_experts=layer\.num_experts,  # global num experts/num_experts=layer.num_local_experts,  # EP-aware: use local expert count/' "$MODELOPT_QUANT"
  sed -i 's/existing_params\.num_experts != layer\.num_experts/existing_params.num_experts != layer.num_local_experts/' "$MODELOPT_QUANT"
  echo "Patched modelopt_quant.py: CutlassMoEParams uses num_local_experts for EP"
elif grep -q 'num_experts=layer.num_local_experts,  # EP-aware: use local expert count' "$MODELOPT_QUANT" 2>/dev/null; then
  echo "modelopt_quant.py: CutlassMoEParams EP fix already applied, skipping"
else
  echo "ANCHOR-DRIFT: modelopt_quant.py: CutlassMoEParams num_local_experts EP fix anchor missing (SGLang version drift; re-check anchor)"
fi

# Patch cutlass_moe.py: cutlass_moe_fp4 EP correctness fix. Two independent
# bugs in the same function, both triggered by `topk_ids == -1` non-local
# sentinels that `StandardDispatcher.local_expert_mapping` writes under EP>1.
#
# BUG 1: a_map / c_map allocated with torch.empty (uninitialized). The native
# kernel prepare_moe_input.compute_arg_sorts iterates blockIdx.x over
# [0, num_experts) and only writes a_map[slot] / c_map[slot] where
# topk_ids[i] == blockIdx.x. The -1 entries match no block and leave those
# slots with torch.empty garbage. Downstream a.index_select(0, a_map) in
# _shuffle_rows_torch reads the garbage as row indices and trips torch's
# vectorized_gather_kernel bounds check — surfaces as device-side assert at
# nvfp4_blockwise_moe.cuh:78 (via next cudaMallocAsync sync point).
# Fix 1: zero-init. Zero is always a valid row index into `a`.
#
# BUG 2: topk_weights are NOT zeroed for -1 slots. Fix 1 alone eliminates the
# crash but produces garbage output. The dispatcher remaps topk_ids to local
# IDs + -1 sentinels, but leaves topk_weights carrying the original softmax
# weights. After the grouped GEMM path, shuffle_rows(c2, c_map, ...) at
# line 493 finds c_map[slot] == 0 for non-local slots (from our Fix 1
# zero-init) and reads c2[0] — the first ACTIVE expert's output for the
# first local token — into those slots. The non-local slots then go into
# `c2 * topk_weights.view(m, num_topk, 1)` carrying real-but-wrong finite
# values multiplied by real (non-zero) weights, and `sum(dim=1)` aggregates
# them into the output alongside the correct local-slot contributions.
# Fix 2: mask topk_weights where topk_ids < 0 at the start of
# cutlass_moe_fp4 — a .masked_fill before any math runs propagates through
# both the `apply_router_weight_on_input=False` path (line 496 final
# multiply) and the `=True` path (weights baked into input earlier).
#
# Two separate PATCH_*_EOF blocks so each patch's grep guard runs
# independently and failure of one doesn't prevent the other.
#
# Upstream PR #20869 is the adjacent work but only fixes the first two bugs
# in this chain (input-scale slicing + num_local_experts for CutlassMoEParams);
# it then sidesteps the third+fourth bugs here by auto-routing SM120 to
# flashinfer_cutlass. Our monkey-patches are the first real fix for the
# cutlass_moe_fp4 codepath under EP that we are aware of. See
# SGLANG_NVFP4_SHUFFLE_ROWS_OOB_UPSTREAM_BUG.md for the full debug ordeal.
CUTLASS_MOE="/usr/local/lib/python3.12/dist-packages/sglang/srt/layers/moe/cutlass_moe.py"
if [ ! -f "$CUTLASS_MOE" ]; then
  echo "ANCHOR-DRIFT: cutlass_moe.py: a_map/c_map zero-init target file missing (SGLang restructured/renamed?)"
elif grep -q 'EP-aware: mask topk_weights where topk_ids' "$CUTLASS_MOE" 2>/dev/null; then
  echo "cutlass_moe.py: EP-aware zero-init already applied, skipping"
elif grep -q 'a_map = torch.empty((topk_ids.numel())' "$CUTLASS_MOE" 2>/dev/null; then
  python3 << 'PATCH_CUTLASS_MOE_ZEROINIT_EOF'
import sys
f = "/usr/local/lib/python3.12/dist-packages/sglang/srt/layers/moe/cutlass_moe.py"
with open(f) as fh:
    code = fh.read()
# Only patch the cutlass_moe_fp4 call site (line ~436) — there is also a
# torch.empty at ~line 145 inside cutlass_fused_experts_fp8 which is the
# FP8 MoE path (not affected by this bug; leave it alone). The second
# occurrence is the one we need, discriminated by the surrounding
# num_topk = topk_ids.shape[1] line that exists only in cutlass_moe_fp4.
old = '''    num_topk = topk_ids.shape[1]
    device = a.device
    a_map = torch.empty((topk_ids.numel()), dtype=torch.int32, device=device)
    c_map = torch.empty((topk_ids.numel()), dtype=torch.int32, device=device)'''
new = '''    num_topk = topk_ids.shape[1]
    device = a.device
    # EP-aware: mask topk_weights where topk_ids == -1 (non-local sentinels
    # from StandardDispatcher.local_expert_mapping). Without this, non-local
    # slots carry real softmax weights into the final c2 * topk_weights
    # multiply and pollute the output reduction with the wrong expert's
    # values (see Fix 2 in the patch header).
    topk_weights = topk_weights.masked_fill(topk_ids < 0, 0)
    # EP-aware: zero-init instead of torch.empty. prepare_moe_input only
    # writes slots for non-(-1) topk_ids; -1 entries leave slots as garbage
    # and the downstream a.index_select(0, a_map) trips torch's bounds
    # check. Zero is a valid row index into `a`; the fake-gathered rows
    # then multiply by our newly-masked (zero) topk_weights above and
    # vanish in the reduction (see Fix 1 in the patch header).
    a_map = torch.zeros((topk_ids.numel()), dtype=torch.int32, device=device)
    c_map = torch.zeros((topk_ids.numel()), dtype=torch.int32, device=device)'''
if old not in code:
    print("ANCHOR-DRIFT: cutlass_moe.py: a_map/c_map zero-init + topk_weights mask anchor missing (SGLang version drift; re-check anchor)")
    sys.exit(0)
code = code.replace(old, new, 1)
with open(f, 'w') as fh:
    fh.write(code)
print("Patched cutlass_moe.py: a_map/c_map zero-init + topk_weights mask for EP")
PATCH_CUTLASS_MOE_ZEROINIT_EOF
else
  echo "ANCHOR-DRIFT: cutlass_moe.py: a_map/c_map zero-init anchor missing (SGLang version drift; re-check anchor)"
fi

# [removed 2026-07-12] ModelOptModelLoader sharded_state patch dropped: verified the
# 0.5.14-sm121 image already routes LoadFormat.SHARDED_STATE to ShardedStateLoader at loader
# selection (get_model_loader), before ModelOptModelLoader.load_model — dead code. Restore from
# git if a pre-v0.5.14 base image is pinned again. Ref: SGLANG_TP_EP_MOE_UPSTREAM_BUG.md.

# Patch MiniMaxM2ForCausalLM: add set_embed_and_head for NEXTN speculative decoding.
# The model has get_embed_and_head but is missing the setter, which eagle_worker.py
# calls to share the target model's embed/head weights with the draft model.
# Every other NEXTN-capable model (DeepSeek, GLM, Llama) has this method.
MINIMAX_M2="/usr/local/lib/python3.12/dist-packages/sglang/srt/models/minimax_m2.py"
if [ ! -f "$MINIMAX_M2" ]; then
  echo "ANCHOR-DRIFT: minimax_m2.py: set_embed_and_head NEXTN patch target file missing (SGLang restructured/renamed?)"
elif grep -q 'def set_embed_and_head' "$MINIMAX_M2"; then
  echo "MiniMax NEXTN patch: already applied, skipping"
elif grep -q 'def get_embed_and_head' "$MINIMAX_M2"; then
  python3 << 'PATCH_MINIMAX_NEXTN_EOF'
f = "/usr/local/lib/python3.12/dist-packages/sglang/srt/models/minimax_m2.py"
with open(f) as fh:
    code = fh.read()
old = "    def get_embed_and_head(self):"
new = """    def set_embed_and_head(self, embed, head):
        del self.model.embed_tokens.weight
        del self.lm_head.weight
        self.model.embed_tokens.weight = embed
        self.lm_head.weight = head
        import torch
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

    def get_embed_and_head(self):"""
code = code.replace(old, new, 1)
with open(f, 'w') as fh:
    fh.write(code)
print("Patched MiniMaxM2ForCausalLM: added set_embed_and_head for NEXTN speculative decoding")
PATCH_MINIMAX_NEXTN_EOF
else
  echo "ANCHOR-DRIFT: minimax_m2.py: get_embed_and_head anchor missing (SGLang version drift; re-check anchor)"
fi

# Patch DeepseekV3Config.kv_lora_rank to allow None for the DeepSeek-V4-Flash
# variant (config.json has kv_lora_rank: null — Flash uses q-LoRA + o-LoRA + GQA,
# NO MLA KV compression). The class sglang actually instantiates for model_type
# "deepseek_v4" is _DeepseekV4ConfigAlias in sglang/srt/utils/hf_transformers/
# common.py, which SUBCLASSES transformers' DeepseekV3Config — so the strict
# dataclass field `kv_lora_rank: int` (and its huggingface_hub @strict validator)
# is declared in transformers/models/deepseek_v3/configuration_deepseek_v3.py,
# NOT in sglang's own configs/deepseek_v4.py (that file is never used for this
# model_type). Under transformers 5.x the null value fails at startup with:
#   StrictDataclassFieldValidationError: Field 'kv_lora_rank' expected int,
#   got NoneType (value: None)
# Widening the annotation to `int | None` BEFORE import makes @strict build a
# Union validator that accepts None. This is a SAFE widening: DeepSeek-V3 / V3.2
# / Kimi-K2 (the other models sharing this config) always supply an int, so they
# are unaffected; only V4-Flash's null now passes. We keep None rather than
# coercing to an int (an int would push modeling onto the MLA KV-LoRA path the
# Flash weights don't have). pyc is timestamp-invalidated, so the edit takes on
# reimport. NOTE: clears the config-parse blocker only — Flash serving may still
# hit further upstream issues downstream (sglang #25165 / #23743).
DEEPSEEK_V3_CFG="/usr/local/lib/python3.12/dist-packages/transformers/models/deepseek_v3/configuration_deepseek_v3.py"
if [ ! -f "$DEEPSEEK_V3_CFG" ]; then
  echo "ANCHOR-DRIFT: configuration_deepseek_v3.py: kv_lora_rank Optional widening target file missing (transformers restructured/renamed?)"
elif grep -q 'kv_lora_rank: int = 512' "$DEEPSEEK_V3_CFG"; then
  python3 << 'PATCH_DSV4_KVLORA_EOF'
f = "/usr/local/lib/python3.12/dist-packages/transformers/models/deepseek_v3/configuration_deepseek_v3.py"
with open(f) as fh:
    code = fh.read()
old = "    kv_lora_rank: int = 512"
new = "    kv_lora_rank: int | None = 512"
if old not in code:
    print("ANCHOR-DRIFT: configuration_deepseek_v3.py: kv_lora_rank Optional widening (SGLang version drift; re-check anchor)")
else:
    code = code.replace(old, new, 1)
    with open(f, 'w') as fh:
        fh.write(code)
    print("Patched DeepseekV3Config: kv_lora_rank now int|None (DeepSeek-V4-Flash kv_lora_rank=null support)")
PATCH_DSV4_KVLORA_EOF
else
  echo "DeepseekV3Config kv_lora_rank patch: not needed or already applied, skipping"
fi

# [removed 2026-07-12] DSV4 indexer seq_lens 2-D patch dropped: verified fp8_paged_mqa_logits_torch
# in the 0.5.14-sm121 image already squeezes seq_lens to 1-D before the shape assert (upstream).
# Restore from git if a pre-v0.5.14 base image is pinned again. See UPSTREAM_DSV4_BUGS.md §6.

# fastsafetensors loader: make it usable on multi-node TP + no-GDS GB10 so the
# weight load STREAMS disk→device through a bounded bounce buffer instead of
# accumulating full shards in host memory (which is what swaps — confirmed by
# memray/smaps: top allocator _load_file weight_utils.py:1060, swapped mapping
# = safetensors-mmap). sglang's fastsafetensors_weights_iterator does a WORLD
# collective load (→ Gloo connectFullMesh timeout across our 4 nodes) onto
# cuda:{world_rank} (invalid on 1-GPU worker nodes). Rewrite to:
#   - SingleGroup() : each rank loads its files independently (no collective)
#   - device "cuda" : the local current device (not cuda:{world_rank})
#   - nogds=True    : 16 MB bounce-buffer streaming (no GPU Direct Storage here)
# TP slicing is unchanged — the per-param weight_loader slices the full tensors,
# exactly as the normal safetensors iterator yields them. Inert unless
# load_format=fastsafetensors. See UPSTREAM_DSV4_BUGS.md.
FST_F="/usr/local/lib/python3.12/dist-packages/sglang/srt/model_loader/weight_utils.py"
if [ ! -f "$FST_F" ]; then
  echo "ANCHOR-DRIFT: weight_utils.py: fastsafetensors target file missing (SGLang restructured/renamed?)"
elif grep -q 'device = torch.device(f"cuda:{rank}")' "$FST_F"; then
  python3 << 'PATCH_DSV4_FST_EOF'
f = "/usr/local/lib/python3.12/dist-packages/sglang/srt/model_loader/weight_utils.py"
code = open(f).read()
old1 = (
    '    if torch.distributed.is_initialized():\n'
    '        pg = torch.distributed.group.WORLD\n'
    '    else:\n'
    '        pg = SingleGroup()\n'
    '\n'
    '    try:\n'
    '        rank = pg.rank()\n'
    '    except Exception:\n'
    '        rank = 0\n'
    '\n'
    '    device = torch.device(f"cuda:{rank}")'
)
new1 = (
    '    # dgxarley: per-rank independent load (no WORLD collective → no Gloo\n'
    '    # connectFullMesh timeout across nodes) onto the LOCAL device (explicit\n'
    '    # index — fastsafetensors set_device rejects bare "cuda"), nogds\n'
    '    # bounce-buffer streaming (no GDS on GB10) → no host full-shard pileup.\n'
    '    pg = SingleGroup()\n'
    '    device = torch.device("cuda", torch.cuda.current_device())'
)
old2 = "        loader = SafeTensorsFileLoader(pg, device)"
new2 = "        loader = SafeTensorsFileLoader(pg, device, nogds=True)"
changed = False
if "torch.cuda.current_device())" in code and "pg = SingleGroup()\n    device" in code:
    print("fastsafetensors patch: already applied, skipping")
else:
    if old1 in code:
        code = code.replace(old1, new1, 1); changed = True
    else:
        print("ANCHOR-DRIFT: weight_utils.py: fastsafetensors pg/device (SGLang version drift; re-check anchor)")
    if old2 in code:
        code = code.replace(old2, new2, 1); changed = True
    else:
        print("ANCHOR-DRIFT: weight_utils.py: fastsafetensors SafeTensorsFileLoader (SGLang version drift; re-check anchor)")
    if changed:
        open(f, "w").write(code)
        print("Patched fastsafetensors_weights_iterator: SingleGroup + local device + nogds")
PATCH_DSV4_FST_EOF
else
  echo "fastsafetensors patch: not needed or already applied, skipping"
fi

# DeepSeek-V4-Flash FlashMLA sparse-decode hook activation (sm_121a / GB10).
# The image bakes deepseek_v4_kernel, but its sitecustomize.py is SHADOWED:
# Ubuntu ships /usr/lib/python3.12/sitecustomize.py (apport) earlier on sys.path,
# and Python imports only the FIRST sitecustomize it finds — so the hook never
# ran and V4-Flash died with "Unsupported architecture for sparse decode fwd".
# A .pth is immune: site.py runs EVERY import-line in EVERY .pth across ALL site
# dirs (no "first wins"), in main AND every spawned sglang worker.
#
# CRITICAL — install ONLY the flash_mla wrapper (_patch_flash_mla_pkg), NOT the
# kernel's patch_flash_mla()/install(). install() also runs
# _patch_sglang_indexer_fallbacks(), which imports sglang…nsa.tilelang_kernel →
# loads tilelang's libcudart_stub.so. At site-init that stub loads BEFORE
# flashinfer.comm, so flashinfer's find_loaded_library("libcudart") grabs the
# stub (no cudaDeviceReset) → hard AttributeError at import (NOT caught by
# sglang's `except ImportError`). sglang imports tilelang itself LATER (after
# flashinfer), so the indexer fallback is not ours to bootstrap. Guarded: a
# broken kernel just falls through to stock flash_mla. Idempotent (also written
# by dockerfile-dsv4-flashmla.patch on rebuilt images). See UPSTREAM_DSV4_BUGS.md §7.
DSV4_DP="/usr/local/lib/python3.12/dist-packages"
if [ -d "$DSV4_DP/deepseek_v4_kernel" ]; then
  cat > "$DSV4_DP/dsv4_autopatch.py" <<'DSV4_AUTOPATCH_EOF'
import os, sys
if os.environ.get("DSV4_KERNEL_DISABLE", "0") not in ("1", "true", "yes"):
    try:
        from deepseek_v4_kernel._patch import _patch_flash_mla_pkg
        _patch_flash_mla_pkg()
    except Exception as exc:
        print("[dsv4_autopatch] flash_mla patch skipped:", exc, file=sys.stderr)
DSV4_AUTOPATCH_EOF
  echo 'import dsv4_autopatch' > "$DSV4_DP/zz_dsv4_autopatch.pth"
  echo "Installed DSV4 FlashMLA autopatch (flash_mla wrapper only): $DSV4_DP/zz_dsv4_autopatch.pth"
else
  echo "DSV4 FlashMLA kernel not present in image, skipping autopatch"
fi

# DSV4 unified-memory load probe (diagnostic, gated by SGLANG_MEMPROBE=1).
# Copies /scripts/dsv4_memprobe.py into dist-packages and drops a .pth so every
# sglang worker arms it at startup (env DSV4_MEMPROBE=1, read by the module). It
# brackets ModelRunner.load_model / init_memory_pool / cuda-graph capture and the
# per-call cuda-alloc delta of Fp8(MoE|Linear)Method.process_weights_after_loading,
# plus a 0.2s ticker — to find which post-load action doubles GB10 unified memory.
# Output → stderr → Loki (grep "[memprobe"). Inert unless SGLANG_MEMPROBE=1.
if [ "${SGLANG_MEMPROBE:-0}" = "1" ] && [ -f /scripts/dsv4_memprobe.py ]; then
  cp /scripts/dsv4_memprobe.py "$DSV4_DP/dsv4_memprobe.py"
  echo 'import dsv4_memprobe' > "$DSV4_DP/zz_dsv4_memprobe.pth"
  export DSV4_MEMPROBE=1
  # memray for the HOST native+mmap allocation profiler (the probe uses it if
  # importable). Best-effort install; absence just disables host profiling.
  pip install -q memray >/dev/null 2>&1 && echo "memprobe: memray installed" || echo "memprobe: memray install failed (host profiling off)"
  echo "Installed DSV4 memprobe (DSV4_MEMPROBE=1): $DSV4_DP/zz_dsv4_memprobe.pth"
else
  rm -f "$DSV4_DP/zz_dsv4_memprobe.pth" "$DSV4_DP/dsv4_memprobe.py" 2>/dev/null || true
fi

# Manual pipeline-stage layer boundaries. SGLang reads SGLANG_PP_LAYER_PARTITION
# directly from the env (os.getenv in get_pp_indices), NOT as a CLI flag. We pass
# our own PP_LAYER_PARTITION and promote it ONLY when non-empty: an empty value
# would make SGLang parse int("") and crash. Empty = SGLang default even split.
if [ -n "${PP_LAYER_PARTITION:-}" ]; then
  export SGLANG_PP_LAYER_PARTITION="$PP_LAYER_PARTITION"
  echo "PP layer partition (manual): SGLANG_PP_LAYER_PARTITION=$SGLANG_PP_LAYER_PARTITION"
fi

args=(
  tini -s --
  python3 -m sglang.launch_server
  --model-path "$model_path"
  --context-length "$SGLANG_CONTEXT_LENGTH"
  --kv-cache-dtype "$SGLANG_KV_CACHE_DTYPE"
  --mem-fraction-static "$SGLANG_MEM_FRACTION"
  --tp-size "$TP"
  --pp-size "$PP"
  --nnodes "$NNODES"
  --node-rank "$NODE_RANK"
  --nccl-init-addr "${QSFP_IP_SPARK1}:${NCCL_PORT}"
  --port "$SGLANG_PORT"
)
# Model compute dtype (--dtype). Unset/"auto" → SGLang reads torch_dtype from the
# model config (correct for almost everything, incl. NVFP4 targets whose config
# already declares dtype bfloat16 — the FP4 weights are unaffected, --dtype only
# sets the compute/activation dtype). Set explicitly when a model needs it —
# notably EAGLE MTP with a Mistral-native draft: the draft params.json carries NO
# dtype field, so --dtype auto falls back to fp32→fp16 and collides with the bf16
# target's shared embed/head → "out_dtype must be Half or BFloat16" at draft graph
# capture. Forcing bfloat16 pins the draft to the target's dtype (SGLang cookbook
# Mistral-Medium-3.5 §3.3: "--dtype bfloat16 is required").
if [ -n "$SGLANG_MODEL_DTYPE" ] && [ "$SGLANG_MODEL_DTYPE" != "auto" ]; then
  args+=(--dtype "$SGLANG_MODEL_DTYPE")
fi
# Debug: per-layer forward-output tensor dump (localise the first inf/nan). OFF unless
# SGLANG_DEBUG_TENSOR_DUMP_OUTPUT_FOLDER is set (registers a forward hook on every layer
# → dumps outputs to the folder; SGLang model_runner.py register_forward_hook_for_model).
# Layers empty = all; else pass the id list through (space-separated for --debug-tensor-dump-layers).
if [ -n "${SGLANG_DEBUG_TENSOR_DUMP_OUTPUT_FOLDER:-}" ]; then
  args+=(--debug-tensor-dump-output-folder "$SGLANG_DEBUG_TENSOR_DUMP_OUTPUT_FOLDER")
  if [ -n "${SGLANG_DEBUG_TENSOR_DUMP_LAYERS:-}" ]; then
    args+=(--debug-tensor-dump-layers ${SGLANG_DEBUG_TENSOR_DUMP_LAYERS})
  fi
fi
# PP async micro-batching: overlap forward passes across pipeline stages.
if [ -n "$PP_ASYNC_BATCH_DEPTH" ] && [ "$PP_ASYNC_BATCH_DEPTH" != "0" ]; then
  args+=(--pp-async-batch-depth "$PP_ASYNC_BATCH_DEPTH")
fi
# Expert parallelism: partitions the TP group for MoE layers.
# EP=TP → MoE uses all-to-all, attention stays tensor-parallel.
if [ -n "$EP" ] && [ "$EP" != "1" ]; then
  args+=(--expert-parallel-size "$EP")
fi
if [ "$SGLANG_ENABLE_EPLB" = "true" ]; then
  args+=(--enable-eplb)
fi
if [ "$SGLANG_ENABLE_EXPERT_DISTRIBUTION_METRICS" = "true" ]; then
  args+=(--enable-expert-distribution-metrics)
fi
# Prometheus exporter: serves /metrics on the HTTP server port (only effective on
# the head; workers run no HTTP server). Gated per-instance via SGLANG_ENABLE_METRICS.
if [ "$SGLANG_ENABLE_METRICS" = "true" ]; then
  args+=(--enable-metrics)
fi
if [ -n "$SGLANG_HOST" ]; then
  args+=(--host "$SGLANG_HOST")
fi
if [ -n "$SGLANG_LOAD_FORMAT" ] && [ "$SGLANG_LOAD_FORMAT" != "auto" ]; then
  args+=(--load-format "$SGLANG_LOAD_FORMAT")
fi
# Diagnostic/tuning knob for the weight-load read-buffer concurrency. The default
# loader uses buffered_multi_thread_safetensors_weights_iterator with 8 workers
# (DEFAULT_NUM_THREADS) — up to 8 shards buffered at once on top of the resident
# weights. Pass e.g. {"enable_multithread_load": false} (no buffering pool) or
# {"num_threads": 1} to shrink the load-time source-buffer peak. NOTE: prefetch
# (weight_loader_prefetch_checkpoints) is OFF by default, so its num_threads is
# inert — THIS is the active buffering control.
if [ -n "$SGLANG_MODEL_LOADER_EXTRA_CONFIG" ]; then
  args+=(--model-loader-extra-config "$SGLANG_MODEL_LOADER_EXTRA_CONFIG")
fi
if [ -n "$SGLANG_QUANTIZATION" ]; then
  args+=(--quantization "$SGLANG_QUANTIZATION")
fi
if [ "$SGLANG_TRUST_REMOTE_CODE" = "true" ]; then
  args+=(--trust-remote-code)
fi
if [ -n "$SGLANG_JSON_MODEL_OVERRIDE_ARGS" ]; then
  args+=(--json-model-override-args "$SGLANG_JSON_MODEL_OVERRIDE_ARGS")
fi
if [ -n "$SGLANG_MOE_RUNNER_BACKEND" ]; then
  args+=(--moe-runner-backend "$SGLANG_MOE_RUNNER_BACKEND")
fi
if [ -n "$SGLANG_REASONING_PARSER" ]; then
  args+=(--reasoning-parser "$SGLANG_REASONING_PARSER")
fi
if [ -n "$SGLANG_TOOL_CALL_PARSER" ]; then
  args+=(--tool-call-parser "$SGLANG_TOOL_CALL_PARSER")
fi
# --chat-template: a built-in template NAME or a path to a .jinja file present in
# the container (from the profile's `chat_template` knob via SGLANG_CHAT_TEMPLATE).
# Empty -> no flag -> SGLang falls back to the model tokenizer's chat_template.
# Distinct from the --chat-template-kwargs handling further below.
if [ -n "$SGLANG_CHAT_TEMPLATE" ]; then
  args+=(--chat-template "$SGLANG_CHAT_TEMPLATE")
fi
if [ "$SGLANG_SPECULATIVE_ENABLED" = "true" ]; then
  args+=(--speculative-algo "$SGLANG_SPECULATIVE_ALGO")
  args+=(--speculative-num-steps "$SGLANG_SPECULATIVE_NUM_STEPS")
  args+=(--speculative-eagle-topk "$SGLANG_SPECULATIVE_EAGLE_TOPK")
  args+=(--speculative-num-draft-tokens "$SGLANG_SPECULATIVE_NUM_DRAFT_TOKENS")
  # External draft model (EAGLE/EAGLE3): use speculative_draft_model_path from profile.
  if [ -n "$SGLANG_SPECULATIVE_DRAFT_MODEL_PATH" ]; then
    args+=(--speculative-draft-model-path "$SGLANG_SPECULATIVE_DRAFT_MODEL_PATH")
  fi
  # Draft model quantization override. By default SGLang inherits the target's
  # quantization for the draft, which breaks when the target is modelopt-
  # quantized (NVFP4) but the draft ships as plain BF16 (typical for external
  # EAGLE3 drafts). Setting "unquant" forces the draft to load without
  # quantization, bypassing the modelopt loader and its Qwen3MoE state-dict
  # shape mismatch against the single-layer EAGLE3 checkpoint.
  if [ -n "$SGLANG_SPECULATIVE_DRAFT_MODEL_QUANTIZATION" ]; then
    args+=(--speculative-draft-model-quantization "$SGLANG_SPECULATIVE_DRAFT_MODEL_QUANTIZATION")
  fi
  # DSV4/SM121: the nextn draft MoE hardcodes an sm100 trtllm kernel that crashes
  # on GB10. Force marlin (SM80+) via this arg — requires the modelopt_quant
  # marlin-branch in sglang-dsv4-nvfp4-pr25820.patch (image rebuild).
  if [ -n "$SGLANG_SPECULATIVE_MOE_RUNNER_BACKEND" ]; then
    args+=(--speculative-moe-runner-backend "$SGLANG_SPECULATIVE_MOE_RUNNER_BACKEND")
  fi
  # WORKAROUND (SGLang 0.5.9): sharded_state + speculative decoding crash.
  # The draft model's ModelRunner inherits load_format=sharded_state from
  # server_args. ShardedStateLoader then fails with KeyError because the
  # per-rank shard files don't contain the draft/MTP model weight keys.
  # Fix: force auto load format for the draft model and point it to the
  # original HF model ID (resolved from HF cache) instead of the shard dir.
  # See SGLANG_SHARDED_SPECULATIVE_UPSTREAM_BUG.md for details.
  if [ "$SGLANG_LOAD_FORMAT" = "sharded_state" ]; then
    args+=(--speculative-draft-load-format auto)
    # Only override draft model path if not already set by profile
    if [ -z "$SGLANG_SPECULATIVE_DRAFT_MODEL_PATH" ]; then
      args+=(--speculative-draft-model-path "$SGLANG_MODEL")
    fi
  fi
  # Adaptive Spec V2 (SGLang ≥0.5.12, PR #23336). Dynamically retunes
  # num_steps / num_draft_tokens at runtime. Only meaningful with
  # EAGLE/EAGLE3 + speculative_eagle_topk=1 — SGLang silently disables
  # otherwise (adaptive_unsupported_reason() in
  # srt/speculative/adaptive_spec_params.py). NEXTN is NOT supported.
  if [ "$SGLANG_SPECULATIVE_ADAPTIVE" = "true" ]; then
    args+=(--speculative-adaptive)
    if [ -n "$SGLANG_SPECULATIVE_ADAPTIVE_CONFIG_JSON" ] \
        && [ "$SGLANG_SPECULATIVE_ADAPTIVE_CONFIG_JSON" != "{}" ] \
        && [ "$SGLANG_SPECULATIVE_ADAPTIVE_CONFIG_JSON" != "null" ]; then
      printf '%s' "$SGLANG_SPECULATIVE_ADAPTIVE_CONFIG_JSON" \
        > /tmp/speculative_adaptive_config.json
      args+=(--speculative-adaptive-config /tmp/speculative_adaptive_config.json)
    fi
  fi
fi
if [ -n "$SGLANG_MAMBA_SCHEDULER_STRATEGY" ]; then
  args+=(--mamba-scheduler-strategy "$SGLANG_MAMBA_SCHEDULER_STRATEGY")
fi
# Mamba state-cache pool sizing (hybrid SSM models). Empty = SGLang auto-fit.
# max_mamba_cache_size // mamba_ratio is the parallelism ceiling on hybrid models.
if [ -n "$SGLANG_MAMBA_FULL_MEMORY_RATIO" ]; then
  args+=(--mamba-full-memory-ratio "$SGLANG_MAMBA_FULL_MEMORY_RATIO")
fi
if [ -n "$SGLANG_MAX_MAMBA_CACHE_SIZE" ] && [ "$SGLANG_MAX_MAMBA_CACHE_SIZE" != "0" ]; then
  args+=(--max-mamba-cache-size "$SGLANG_MAX_MAMBA_CACHE_SIZE")
fi
if [ -n "$SGLANG_MAX_RUNNING_REQUESTS" ] && [ "$SGLANG_MAX_RUNNING_REQUESTS" != "0" ]; then
  args+=(--max-running-requests "$SGLANG_MAX_RUNNING_REQUESTS")
fi
# Absolute KV-cache pool cap (tokens). Unset/0 -> sized by mem_fraction_static.
# Used to pin a co-located instance's memory footprint deterministically.
if [ -n "$SGLANG_MAX_TOTAL_TOKENS" ] && [ "$SGLANG_MAX_TOTAL_TOKENS" != "0" ]; then
  args+=(--max-total-tokens "$SGLANG_MAX_TOTAL_TOKENS")
fi
if [ -n "$SGLANG_SCHEDULE_POLICY" ]; then
  args+=(--schedule-policy "$SGLANG_SCHEDULE_POLICY")
fi
if [ -n "$SGLANG_CHUNKED_PREFILL_SIZE" ] && [ "$SGLANG_CHUNKED_PREFILL_SIZE" != "0" ]; then
  args+=(--chunked-prefill-size "$SGLANG_CHUNKED_PREFILL_SIZE")
fi
if [ -n "$SGLANG_DIST_TIMEOUT" ]; then
  args+=(--dist-timeout "$SGLANG_DIST_TIMEOUT")
fi
if [ -n "$SGLANG_WATCHDOG_TIMEOUT" ]; then
  args+=(--watchdog-timeout "$SGLANG_WATCHDOG_TIMEOUT")
fi
if [ -n "$SGLANG_ATTENTION_BACKEND" ]; then
  args+=(--attention-backend "$SGLANG_ATTENTION_BACKEND")
fi
# DSA paged-MQA-logits backend (DeepSeek Sparse Attention indexer decode kernel).
# Empty → no flag → SGLang default ("auto" → DeepGEMM). Set "torch" on GB10/SM121 to use
# the _sgl_ torch fallback (DeepGEMM asserts on sm_121). Other choices: deepgemm/cutedsl/aiter.
if [ -n "$SGLANG_DSA_PAGED_MQA_LOGITS_BACKEND" ]; then
  args+=(--dsa-paged-mqa-logits-backend "$SGLANG_DSA_PAGED_MQA_LOGITS_BACKEND")
fi
# DSA decode attention backend (the MLA attention step over the indexer's top-k KV
# selection). Empty → no flag → SGLang default ("auto" → trtllm-gen FMHA, dead on
# SM121). Set "flashinfer_gather" on GB10/SM121 to use the _sgl_ gather+dense-fa2
# fallback patched above. Other choices: flashmla_sparse/flashmla_kv/flashmla_auto/
# fa3/tilelang/aiter/trtllm (all dead ends on SM121, see DSA_speedup.md).
if [ -n "$SGLANG_DSA_DECODE_BACKEND" ]; then
  args+=(--dsa-decode-backend "$SGLANG_DSA_DECODE_BACKEND")
fi
# DSA PREFILL attention backend (the MLA attention step for extend/prefill tokens).
# Empty → no flag → SGLang default ("auto" → trtllm-gen FMHA, dead on SM121, same
# crash class as the decode backend). Set "flashinfer_gather" on GB10/SM121 to reuse
# the SAME gather+dense-fa2 fallback as decode (patched above; no separate kernel).
# Other choices: flashmla_sparse/flashmla_kv/flashmla_auto/fa3/tilelang/aiter/trtllm
# (all dead ends on SM121, see DSA_speedup.md).
if [ -n "$SGLANG_DSA_PREFILL_BACKEND" ]; then
  args+=(--dsa-prefill-backend "$SGLANG_DSA_PREFILL_BACKEND")
fi
# Multimodal (vision/audio) attention backend. Empty → no flag → SGLang default.
# Choices (0.5.12): sdpa, fa3, fa4, triton_attn, ascend_attn, aiter_attn,
# flashinfer_cudnn. Only relevant for multimodal models (e.g. MiMoV2 omni).
if [ -n "$SGLANG_MM_ATTENTION_BACKEND" ]; then
  args+=(--mm-attention-backend "$SGLANG_MM_ATTENTION_BACKEND")
fi
# KV-cache page size (tokens per page). Empty/0 → no flag → SGLang default.
# Some attention backends / hybrid-SWA paths want a larger page (MiMoV2 card: 64).
if [ -n "$SGLANG_PAGE_SIZE" ] && [ "$SGLANG_PAGE_SIZE" != "0" ]; then
  args+=(--page-size "$SGLANG_PAGE_SIZE")
fi
# Hybrid-SWA KV budget: ratio of SWA-layer KV tokens to full-layer KV tokens
# (SGLang default 0.8). Empty → no flag → SGLang default. Only meaningful on
# hybrid sliding-window models (MiMoV2, Gemma, Llama4). For MiMoV2 + hierarchical
# cache SGLang internally resets this to 1.0.
if [ -n "$SGLANG_SWA_FULL_TOKENS_RATIO" ]; then
  args+=(--swa-full-tokens-ratio "$SGLANG_SWA_FULL_TOKENS_RATIO")
fi
# Diffusion-LLM (dLLM) decode path — when SGLANG_DLLM_ALGORITHM is set (e.g.
# "Gemma4Renoise" for DiffusionGemma) launch_server runs the block-diffusion
# scheduler instead of the autoregressive one. SGLang's _handle_dllm_inference
# auto-forces triton attention, eager mode (cuda graph disabled), and unchunked
# prefill for Gemma4Renoise, so the autoregressive cuda-graph / attention flags
# above are overridden internally. Empty for all autoregressive models → no flag
# is added, zero impact. Requires the 0.5.13-gemmadiffusion image (PR #28054
# baked); other images reject --dllm-algorithm for Gemma4.
if [ -n "$SGLANG_DLLM_ALGORITHM" ]; then
  args+=(--dllm-algorithm "$SGLANG_DLLM_ALGORITHM")
fi
# Server log level. Empty → no flag → SGLang's built-in default ('info'). Set
# SGLANG_LOG_LEVEL=debug (via sglang_log_level in defaults/main/sglang.yml or a model
# profile) to surface SGLang's logger.debug diagnostics — notably the Frozen-KV
# MTP draft-bind skip ("Draft model <class> does not implement ... skipping
# frozen-kv bind."), which names the class a Gemma-4 assistant draft loads as.
if [ -n "$SGLANG_LOG_LEVEL" ]; then
  args+=(--log-level "$SGLANG_LOG_LEVEL")
fi
if [ -n "$SGLANG_FP8_GEMM_RUNNER_BACKEND" ] && [ "$SGLANG_FP8_GEMM_RUNNER_BACKEND" != "auto" ]; then
  args+=(--fp8-gemm-backend "$SGLANG_FP8_GEMM_RUNNER_BACKEND")
fi
if [ -n "$SGLANG_FP4_GEMM_BACKEND" ] && [ "$SGLANG_FP4_GEMM_BACKEND" != "auto" ]; then
  args+=(--fp4-gemm-backend "$SGLANG_FP4_GEMM_BACKEND")
fi
if [ "$SGLANG_DISABLE_CUDA_GRAPH" = "true" ] || [ "$SGLANG_CUDA_GRAPH_MAX_BS" = "0" ]; then
  args+=(--disable-cuda-graph)
elif [ -n "$SGLANG_CUDA_GRAPH_MAX_BS" ] && [ "$SGLANG_CUDA_GRAPH_MAX_BS" != "256" ]; then
  args+=(--cuda-graph-max-bs "$SGLANG_CUDA_GRAPH_MAX_BS")
fi
if [ "$SGLANG_DISABLE_PIECEWISE_CUDA_GRAPH" = "true" ]; then
  args+=(--disable-piecewise-cuda-graph)
fi
if [ "$SGLANG_WEIGHT_LOADER_DISABLE_MMAP" = "true" ]; then
  args+=(--weight-loader-disable-mmap)
fi
if [ "$SGLANG_WEIGHT_LOADER_DROP_CACHE_AFTER_LOAD" = "true" ]; then
  args+=(--weight-loader-drop-cache-after-load)
fi
if [ "$SGLANG_DISABLE_OVERLAP_SCHEDULE" = "true" ]; then
  args+=(--disable-overlap-schedule)
fi
if [ "$SGLANG_DISABLE_FLASHINFER_CUTLASS_MOE_FP4_ALLGATHER" = "true" ]; then
  args+=(--disable-flashinfer-cutlass-moe-fp4-allgather)
fi
if [ -n "$SGLANG_SERVED_MODEL_NAME" ]; then
  args+=(--served-model-name "$SGLANG_SERVED_MODEL_NAME")
fi
# Chat template kwargs (enable_thinking, etc.)
# These control Jinja2 chat template rendering — NOT sampling parameters.
# Sampling defaults (temperature, top_p, ...) are set via generation_config.json
# overlay above, because SGLang has no CLI flags for individual sampling params.
# NOTE: thinking_budget does NOT go here — it uses SGLang's custom logit processor
# system (--enable-custom-logit-processor), not the chat template.
if [ -n "$SGLANG_CHAT_TEMPLATE_KWARGS" ] && [ "$SGLANG_CHAT_TEMPLATE_KWARGS" != "{}" ]; then
  # CAPABILITY GUARD: some SGLang builds (e.g. the 0.5.12/0.5.14-sm121 images) do
  # NOT expose a server-side --chat-template-kwargs flag — passing it aborts the
  # server at argparse ("unrecognized arguments: --chat-template-kwargs") and
  # crash-loops the head. The flag's `dest` (chat_template_kwargs) is present in
  # server_args.py ONLY on builds that support it, so grep that file to decide.
  # When the flag is UNsupported we skip it and rely on the LiteLLM extra_body
  # default (_sglang_extra_body_defaults, same profile key) — which still applies
  # the reasoning default to all LiteLLM-routed traffic. Direct-to-SGLang requests
  # get no server-side default on such images (pass reasoning_effort per request).
  SGLANG_ARGS_FILE="/usr/local/lib/python3.12/dist-packages/sglang/srt/server_args.py"
  if grep -q "chat_template_kwargs" "$SGLANG_ARGS_FILE" 2>/dev/null; then
    args+=(--chat-template-kwargs "$SGLANG_CHAT_TEMPLATE_KWARGS")
  else
    echo "[launch] NOTE: this SGLang build has no --chat-template-kwargs flag; skipping the" \
         "server-side chat_template_kwargs default ($SGLANG_CHAT_TEMPLATE_KWARGS)." \
         "LiteLLM extra_body still carries it for proxied requests."
  fi
fi
# Enable custom logit processors (required for per-request thinking_budget via
# Qwen3ThinkingBudgetLogitProcessor). Safe to always enable — no-op if unused.
args+=(--enable-custom-logit-processor)

# Echo the exact launch command + relevant ENV to stdout so the head/worker
# pod logs (and Loki) capture them verbatim. printf '%q ' produces a shell-safe,
# copy-pasteable form (handles spaces, quotes, JSON args). Logged before exec
# so it appears even if the server crashes during startup.
#
# ENV filter: SGLANG_*, NCCL_*, FLASHINFER_*, TORCH*, CUDA_*, HF_*, plus a few
# named knobs that gate behavior at runtime (mamba/spec/JIT). Excludes generic
# vars (PATH, HOME, K8S_*, KUBERNETES_*, POD_*) to keep output focused.
printf '=== sglang launch ENV (filtered, secrets redacted) ===\n'
env | grep -E '^(SGLANG_|NCCL_|FLASHINFER_|TORCH(_|INDUCTOR_)|CUDA_|HF_|GLOO_|UCX_|RDMAV_|MASTER_|RANK=|WORLD_SIZE=|LOCAL_RANK=|NODE_RANK=|NNODES=|DIST_INIT_ADDR=|MAMBA_|SPEC_V2)' \
  | grep -vE '^(SGLANG_EXPECTED_IMAGE_PATTERN=|HF_HUB_OFFLINE_PATH=)' \
  | sed -E 's/^([A-Z_0-9]*(TOKEN|SECRET|KEY|PASSWORD|PASS|API|CREDENTIAL)[A-Z_0-9]*)=.*/\1=***REDACTED***/' \
  | LC_ALL=C sort
printf '=== end sglang launch ENV ===\n'

printf '=== sglang launch command (%d args) ===\n' "${#args[@]}"
printf '%q ' "${args[@]}"
printf '\n=== end sglang launch command ===\n'

exec "${args[@]}"
