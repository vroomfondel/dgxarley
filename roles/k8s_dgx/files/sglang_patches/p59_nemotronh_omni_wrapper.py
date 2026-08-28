"""[dgxarley] NemotronH VL/Omni wrapper MoE routing — upstream PR #25024 (OPEN, not in
v0.5.13), rebased onto v0.5.13. Three small Python edits across 3 files.

Symptom (nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-NVFP4 on SM121/GB10):
  cutlass_moe.py cutlass_moe_fp4 -> AssertionError "mismatch in expected `n`"
  (nx2_w1 == intermediate_size_per_partition * 2) during flashinfer autotune.
Cause: the NemotronH MoE defaults hook only matched bare NemotronHForCausalLM, so
the VL/Omni wrapper archs (NemotronH_Nano_VL_V2 / _Nano_Omni_Reasoning_V3) bypassed
it. Their LLM sub-config nests under hf_config.llm_config (not top-level / text_config),
so the MoE-config resolution was skipped -> backend stayed AUTO -> fell through to the
sm_100-only cutlass_moe_fp4 with a mismatched intermediate size. Even an explicit
moe_runner_backend=flashinfer_cutlass in the profile doesn't fix it -- the hook also
does the llm_config field resolution this needs.
Fix (PR #25024): (1) server_args dispatch list includes the wrapper archs;
(2) NemotronH overrides read fields from hf_config.llm_config for wrappers;
(3) scheduler.init_moe_gemm_config resolves the LLM sub-config via hf_text_config.
Drop this block once PR #25024 lands in a release tag we use (the anchor-drift guards
make it a safe no-op if the targets are already changed, e.g. on a future baked image).

RE-ANCHORED 2026-07-16: sub-patch (1) originally targeted the standalone
arg_groups/nemotron_h_hook.py, which no longer exists on this image -- upstream
ABSORBED it into arg_groups/overrides.py::_nemotron_h_overrides (module docstring:
"absorbed from the retired arg_groups/nemotron_h_hook.py"). BUT that absorption did
NOT carry over the wrapper-arch fix: the @_register_for(...) decorator on
_nemotron_h_overrides still only lists the two bare NemotronH archs, so the VL/Omni
wrappers still dispatch to NO override function at all (verified: zero hits for
NemotronH_Nano_VL_V2 / _Nano_Omni_Reasoning_V3 anywhere in overrides.py/server_args.py/
scheduler.py on this image) -- same bug, new home. Sub-patch (1) below now targets
overrides.py: extends the decorator to the wrapper archs AND keeps the llm_config
field resolution for mlp_hidden_act. Sub-patches (2)/(3) are unchanged (still apply).

No model gate, no env gate: unconditional, same as the original heredoc (only the
per-file anchor-drift checks decide whether each sub-patch applies) -- with ONE
exception since 2026-08-28: sub-patch (3) is gated on the server_args.py dispatch
branch still existing, because the Nemotron-3.5 spec source patch (PR #36186,
applied by the 0.5.18 recipe) deletes it in favour of the override registry that
sub-patch (1) already extends. Gate false there = nothing to do, not drift.

Note on the shared MARKER and `Patch.replace`'s already-applied probe: the original
script checked `marker in s` ONCE, up front, over the whole sub-patch (3), before
touching any of the 3 anchors in that file. `_patchlib.Patch.replace` checks its
probe per-call instead, and MARKER text is only injected by the *second* of the 3
edits (the body-header rewrite) -- so only that call passes `marker=MARKER`
explicitly; the other two calls use the default probe (their own `new` text, which
is unique in the file), so a MARKER already present in the file from edit 2 does not
cause edits 1/3 to be misjudged as "already applied" before they've actually run.
"""

from _patchlib import Patch, target_contains

MARKER = "# [patch] _sgl_nemotronh_omni_wrapper_"

# --- 1) arg_groups/overrides.py: _nemotron_h_overrides (formerly nemotron_h_hook.py) ---
patch_overrides = Patch(
    name="NemotronH VL/Omni wrapper: overrides.py dispatch + llm_config resolution",
    target="sglang/srt/arg_groups/overrides.py",
)

OLD_1A = '@_register_for("NemotronHForCausalLM", "NemotronHPuzzleForCausalLM")\n'
NEW_1A = (
    "@_register_for(\n"
    '    "NemotronHForCausalLM",\n'
    '    "NemotronHPuzzleForCausalLM",\n'
    '    "NemotronH_Nano_VL_V2",\n'
    '    "NemotronH_Nano_Omni_Reasoning_V3",\n'
    ")\n"
)

OLD_1B = (
    "    model_arch = hf_config.architectures[0]\n"
    "    model_config = server_args.get_model_config()\n"
    "    overrides: Dict[str, Any] = {}\n"
    "\n"
    "    is_modelopt = model_config.quantization in [\n"
)
NEW_1B = (
    "    " + MARKER + " (PR #25024, re-anchored 2026-07-16 onto\n"
    "    # upstream-absorbed overrides.py::_nemotron_h_overrides; the decorator above\n"
    "    # was extended to also dispatch the VL/Omni wrapper archs, which upstream\n"
    "    # never registered here -> they fell through with NO overrides at all.\n"
    "    # NemotronH config fields live on inner llm_config for the wrappers, on\n"
    "    # hf_config for standalone.\n"
    "    model_arch = hf_config.architectures[0]\n"
    "    model_config = server_args.get_model_config()\n"
    '    nemotron_h_cfg = getattr(model_config.hf_config, "llm_config", model_config.hf_config)\n'
    "    overrides: Dict[str, Any] = {}\n"
    "\n"
    "    is_modelopt = model_config.quantization in [\n"
)

OLD_1C = '        assert model_config.hf_config.mlp_hidden_act == "relu2"\n'
NEW_1C = '        assert nemotron_h_cfg.mlp_hidden_act == "relu2"\n'


@patch_overrides.run
def apply_overrides(p: Patch) -> None:
    p.replace(OLD_1A, NEW_1A, what="decorator dispatch list (VL/Omni wrapper archs)")
    p.replace(OLD_1B, NEW_1B, marker=MARKER, what="body-header llm_config resolution")
    p.replace(OLD_1C, NEW_1C, what="mlp_hidden_act assert reads nemotron_h_cfg")


# --- 2) managers/scheduler.py: init_moe_gemm_config via hf_text_config ---
patch_scheduler = Patch(
    name="NemotronH VL/Omni wrapper: scheduler.py hf_text_config resolution",
    target="sglang/srt/managers/scheduler.py",
)

OLD_2 = (
    "        # For the MM models, check the text_config for MoE settings\n"
    "        config_to_check = getattr(\n"
    '            self.model_config.hf_config, "text_config", self.model_config.hf_config\n'
    "        )\n"
)
NEW_2 = (
    "        " + MARKER + " (PR #25024) — resolve the LLM sub-config via\n"
    "        # hf_text_config so NemotronH VL/Omni wrappers (nest under llm_config) are\n"
    "        # not skipped (else initialize_moe_config is skipped -> MoE backend AUTO).\n"
    "        config_to_check = self.model_config.hf_text_config\n"
)


@patch_scheduler.run
def apply_scheduler(p: Patch) -> None:
    p.replace(OLD_2, NEW_2, marker=MARKER, what="init_moe_gemm_config uses hf_text_config")


# --- 3) server_args.py: add wrapper archs to the NemotronH dispatch ---
_SERVER_ARGS = "sglang/srt/server_args.py"
# The dispatch branch this sub-patch widens. PR #36186 (the Nemotron-3.5 spec
# source patch this recipe applies, APPLY_NEMOTRON35_SPEC_PR36186) DELETES the
# whole branch and moves the NemotronH defaults into the override registry --
# which sub-patch (1) above already covers. So on such an image there is nothing
# left to widen here: gate on the branch instead of reporting ANCHOR-DRIFT, so
# the gate report stays a work list. Sub-patches (1)/(2) are unaffected.
_NEMOTRON_DISPATCH = '        elif model_arch in ["NemotronHForCausalLM", "NemotronHPuzzleForCausalLM"]:\n'

patch_server_args = Patch(
    name="NemotronH VL/Omni wrapper: server_args.py dispatch list",
    target=_SERVER_ARGS,
    when=target_contains(_SERVER_ARGS, _NEMOTRON_DISPATCH),
)

OLD_3 = _NEMOTRON_DISPATCH
NEW_3 = (
    "        " + MARKER.lstrip() + " (PR #25024): add VL/Omni wrapper archs\n"
    "        elif model_arch in ["
    '"NemotronHForCausalLM", "NemotronHPuzzleForCausalLM", '
    '"NemotronH_Nano_VL_V2", "NemotronH_Nano_Omni_Reasoning_V3"]:\n'
)


@patch_server_args.run
def apply_server_args(p: Patch) -> None:
    p.replace(OLD_3, NEW_3, marker=MARKER, what="NemotronH dispatch includes VL/Omni wrappers")
