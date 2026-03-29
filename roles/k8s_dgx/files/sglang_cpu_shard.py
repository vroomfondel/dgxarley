"""CPU-only AWQ model sharding for SGLang tensor-parallel loading.

Processes safetensors files iteratively on CPU via memory mapping.
No GPU, no NCCL, no multi-node coordination needed.
Each node runs independently as a K8s Job.

Target SGLang version: 0.5.9-t5

Supported architectures:
    - ``qwen2`` / ``qwen3`` (dense)
    - ``qwen2_moe`` / ``qwen3_moe`` (Mixture-of-Experts)

Environment variables:
    SGLANG_MODEL: Model ID (e.g. ``QuantTrio/Qwen3-235B-A22B-Instruct-2507-AWQ``).
    TP: Tensor parallel size (default: 2).
    EP: Expert parallel size (default: 1). Partitions the TP group for MoE
        layers. With ``EP=TP``, each rank receives whole experts instead of
        TP-split expert weight matrices.
    NODE_RANK: This node's rank (0 or 1).
    HF_TOKEN: HuggingFace token (optional).
    SHARD_OUTPUT_DIR: Override output directory (optional).
    DRY_RUN: If ``"1"``, print output tensor names without writing files.
"""

import json
import math
import os
import re
import shutil
import sys
from collections import defaultdict
from pathlib import Path

import torch
from safetensors import safe_open  # type: ignore[import-not-found]
from safetensors.torch import save_file  # type: ignore[import-not-found]

AWQ_SUFFIXES: tuple[str, ...] = ("qweight", "qzeros", "scales")
"""AWQ weight component suffixes present on every quantized linear layer."""

# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------

WeightMap = dict[str, str]
"""Maps a tensor name to the safetensors filename that contains it."""

OpenFiles = dict[str, object]
"""Maps a safetensors filename to its opened (memory-mapped) file handle."""

ArchParams = dict[str, str | int | bool]
"""Architecture parameters extracted from ``config.json``."""

TensorDict = dict[str, torch.Tensor]
"""Maps output tensor names to their (possibly sliced) tensor data."""


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------


def load_configs(
    model_path: Path,
) -> tuple[dict[str, object], dict[str, object], WeightMap]:
    """Load model configuration, quantization config, and safetensors index.

    Reads ``config.json``, ``quantize_config.json`` (or the embedded
    ``quantization_config`` key), and ``model.safetensors.index.json`` from
    the given model directory. Falls back to enumerating a single
    ``model.safetensors`` file if no index is present.

    Args:
        model_path: Local path to the downloaded HuggingFace model snapshot.

    Returns:
        A 3-tuple of ``(config, quant_config, weight_map)`` where *config*
        and *quant_config* are raw JSON dicts and *weight_map* maps every
        tensor name to the safetensors file that stores it.

    Raises:
        FileNotFoundError: If neither a quantization config nor any
            safetensors files can be found.
    """
    with open(model_path / "config.json") as f:
        config: dict[str, object] = json.load(f)

    quant_config_path = model_path / "quantize_config.json"
    if quant_config_path.exists():
        with open(quant_config_path) as f:
            quant_config: dict[str, object] = json.load(f)
    elif "quantization_config" in config:
        quant_config = config["quantization_config"]  # type: ignore[assignment]
    else:
        raise FileNotFoundError(
            f"No quantization config found: neither quantize_config.json "
            f"nor quantization_config in config.json in {model_path}"
        )

    index_path = model_path / "model.safetensors.index.json"
    if index_path.exists():
        with open(index_path) as f:
            index: dict[str, object] = json.load(f)
        weight_map: WeightMap = index["weight_map"]  # type: ignore[assignment]
    else:
        single = model_path / "model.safetensors"
        if not single.exists():
            raise FileNotFoundError(f"No safetensors files found in {model_path}")
        with safe_open(str(single), framework="pt", device="cpu") as f:
            weight_map = {name: "model.safetensors" for name in f.keys()}

    return config, quant_config, weight_map


def get_architecture_params(config: dict[str, object]) -> ArchParams:
    """Extract architecture parameters from the model's ``config.json``.

    Reads common transformer dimensions (hidden size, head counts, layer
    count, vocabulary size) and MoE-specific fields when applicable.

    Args:
        config: Parsed ``config.json`` dictionary.

    Returns:
        An :data:`ArchParams` dict with string, integer, and boolean values
        keyed by parameter name. MoE models additionally contain
        ``num_experts``, ``moe_intermediate_size``, and ``is_moe=True``.

    Raises:
        ValueError: If ``model_type`` is not one of the supported
            architectures (``qwen2``, ``qwen3``, ``qwen2_moe``,
            ``qwen3_moe``).
    """
    model_type = str(config.get("model_type", ""))
    hidden_size = int(config["hidden_size"])  # type: ignore[call-overload]
    num_attention_heads = int(config["num_attention_heads"])  # type: ignore[call-overload]
    num_kv_heads = int(config.get("num_key_value_heads", num_attention_heads))  # type: ignore[call-overload]
    num_hidden_layers = int(config["num_hidden_layers"])  # type: ignore[call-overload]
    vocab_size = int(config["vocab_size"])  # type: ignore[call-overload]
    head_dim = int(config.get("head_dim", hidden_size // num_attention_heads))  # type: ignore[call-overload]

    params: ArchParams = {
        "model_type": model_type,
        "hidden_size": hidden_size,
        "num_attention_heads": num_attention_heads,
        "num_key_value_heads": num_kv_heads,
        "num_hidden_layers": num_hidden_layers,
        "vocab_size": vocab_size,
        "head_dim": head_dim,
    }

    if model_type in ("qwen2_moe", "qwen3_moe"):
        params["num_experts"] = int(config["num_experts"])  # type: ignore[call-overload]
        params["moe_intermediate_size"] = int(config["moe_intermediate_size"])  # type: ignore[call-overload]
        params["intermediate_size"] = int(  # type: ignore[call-overload]
            config.get("shared_expert_intermediate_size", config["intermediate_size"])
        )
        params["is_moe"] = True
    elif model_type in ("qwen2", "qwen3"):
        params["intermediate_size"] = int(config["intermediate_size"])  # type: ignore[call-overload]
        params["is_moe"] = False
    else:
        raise ValueError(f"Unsupported model_type: {model_type!r}. " "Supported: qwen2, qwen2_moe, qwen3, qwen3_moe")

    return params


# ---------------------------------------------------------------------------
# Tensor grouping & file access
# ---------------------------------------------------------------------------


def group_tensors_by_layer(weight_map: WeightMap) -> dict[int | str, list[str]]:
    """Group tensor names by transformer layer index.

    Parses the ``model.layers.<N>.`` prefix from each tensor name. Tensors
    that do not belong to a numbered layer (e.g. ``model.embed_tokens``,
    ``model.norm``, ``lm_head``) are collected under the key ``"global"``.

    Args:
        weight_map: Tensor name to filename mapping.

    Returns:
        A dict mapping layer indices (``int``) or ``"global"`` to their
        list of tensor names.
    """
    groups: dict[int | str, list[str]] = defaultdict(list)
    layer_re = re.compile(r"model\.layers\.(\d+)\.")
    for name in weight_map:
        m = layer_re.match(name)
        if m:
            groups[int(m.group(1))].append(name)
        else:
            groups["global"].append(name)
    return groups


def open_safetensors_files(model_path: Path, weight_map: WeightMap) -> OpenFiles:
    """Open all unique safetensors files via memory mapping.

    Each file is opened once with ``safe_open(..., device="cpu")`` for
    zero-copy tensor access.

    Args:
        model_path: Directory containing the safetensors files.
        weight_map: Tensor name to filename mapping (only unique filenames
            are opened).

    Returns:
        A dict mapping each filename to its opened file handle.
    """
    files: OpenFiles = {}
    for filename in set(weight_map.values()):
        filepath = model_path / filename
        files[filename] = safe_open(str(filepath), framework="pt", device="cpu")
    return files


def get_tensor(name: str, weight_map: WeightMap, open_files: OpenFiles) -> torch.Tensor:
    """Load a single tensor by name from the appropriate safetensors file.

    Args:
        name: Fully qualified tensor name (e.g.
            ``model.layers.0.self_attn.q_proj.qweight``).
        weight_map: Tensor name to filename mapping.
        open_files: Opened safetensors file handles.

    Returns:
        The requested tensor on CPU.
    """
    filename = weight_map[name]
    return open_files[filename].get_tensor(name)  # type: ignore[attr-defined,no-any-return]


# ---------------------------------------------------------------------------
# TP split primitives
# ---------------------------------------------------------------------------


def split_column(tensor: torch.Tensor, tp: int, rank: int) -> torch.Tensor:
    """Column-parallel split: partition the last dimension.

    Used for weight matrices where the output features are distributed
    across TP ranks (e.g. ``q_proj``, ``gate_proj``, ``up_proj``).

    Args:
        tensor: Input tensor of shape ``(..., out_features)``.
        tp: Tensor parallel world size.
        rank: This rank's index within the TP group.

    Returns:
        A contiguous slice ``tensor[..., rank*chunk:(rank+1)*chunk]``.

    Raises:
        AssertionError: If the last dimension is not evenly divisible by *tp*.
    """
    size = tensor.shape[-1]
    assert size % tp == 0, f"Column split: dim={size} not divisible by tp={tp}"
    chunk = size // tp
    return tensor[..., rank * chunk : (rank + 1) * chunk].contiguous()


def split_row(tensor: torch.Tensor, tp: int, rank: int) -> torch.Tensor:
    """Row-parallel split: partition the packed input dimension.

    For 2-D tensors the split is along dim 0; for 3-D tensors (stacked
    expert weights) the split is along dim 1.

    Args:
        tensor: Input tensor of shape ``(in_features, ...)`` or
            ``(num_experts, in_features, ...)``.
        tp: Tensor parallel world size.
        rank: This rank's index within the TP group.

    Returns:
        A contiguous narrow slice of the input dimension.

    Raises:
        AssertionError: If the split dimension is not evenly divisible
            by *tp*.
    """
    dim = 0 if tensor.dim() == 2 else 1
    size = tensor.shape[dim]
    assert size % tp == 0, f"Row split: shape={list(tensor.shape)} dim={dim} " f"size={size} not divisible by tp={tp}"
    chunk = size // tp
    return torch.narrow(tensor, dim, rank * chunk, chunk).contiguous()


def split_vocab(tensor: torch.Tensor, tp: int, rank: int, vocab_size: int) -> torch.Tensor:
    """Vocabulary-parallel split: pad to a multiple of *tp*, then partition dim 0.

    Embedding and ``lm_head`` weight matrices may have a vocabulary
    dimension that is not divisible by *tp*. This function zero-pads to the
    next multiple before splitting.

    Args:
        tensor: Embedding tensor of shape ``(vocab_size, hidden_size)``.
        tp: Tensor parallel world size.
        rank: This rank's index within the TP group.
        vocab_size: Original (unpadded) vocabulary size.

    Returns:
        A contiguous slice of the (possibly padded) tensor along dim 0.
    """
    padded = math.ceil(vocab_size / tp) * tp
    if tensor.shape[0] < padded:
        pad = torch.zeros(padded - tensor.shape[0], *tensor.shape[1:], dtype=tensor.dtype)
        tensor = torch.cat([tensor, pad], dim=0)
    chunk = padded // tp
    return tensor[rank * chunk : (rank + 1) * chunk].contiguous()


# ---------------------------------------------------------------------------
# Layer processing — attention
# ---------------------------------------------------------------------------


def process_qkv(
    prefix: str,
    params: ArchParams,
    tp: int,
    rank: int,
    weight_map: WeightMap,
    open_files: OpenFiles,
) -> TensorDict:
    """Fuse Q/K/V projections into a single ``qkv_proj`` with GQA-aware TP split.

    SGLang expects a fused QKV weight. This function loads the separate
    ``q_proj``, ``k_proj``, ``v_proj`` AWQ components, slices each to this
    rank's share (accounting for grouped-query attention where
    ``num_kv_heads < num_attention_heads``), and concatenates them.

    Args:
        prefix: Layer name prefix (e.g. ``model.layers.0``).
        params: Architecture parameters containing head counts and dimensions.
        tp: Tensor parallel world size.
        rank: This rank's index.
        weight_map: Tensor name to filename mapping.
        open_files: Opened safetensors file handles.

    Returns:
        A dict mapping fused ``qkv_proj`` tensor names (and optional bias)
        to their sliced tensors.
    """
    output: TensorDict = {}
    head_dim = int(params["head_dim"])
    num_heads = int(params["num_attention_heads"])
    num_kv_heads = int(params["num_key_value_heads"])
    q_per_rank = (num_heads // tp) * head_dim
    kv_per_rank = (num_kv_heads // tp) * head_dim

    for suffix in AWQ_SUFFIXES:
        q_name = f"{prefix}.self_attn.q_proj.{suffix}"
        if q_name not in weight_map:
            continue
        q = get_tensor(q_name, weight_map, open_files)
        k = get_tensor(f"{prefix}.self_attn.k_proj.{suffix}", weight_map, open_files)
        v = get_tensor(f"{prefix}.self_attn.v_proj.{suffix}", weight_map, open_files)

        q_chunk = q[..., rank * q_per_rank : (rank + 1) * q_per_rank]
        k_chunk = k[..., rank * kv_per_rank : (rank + 1) * kv_per_rank]
        v_chunk = v[..., rank * kv_per_rank : (rank + 1) * kv_per_rank]

        output[f"{prefix}.self_attn.qkv_proj.{suffix}"] = torch.cat([q_chunk, k_chunk, v_chunk], dim=-1).contiguous()

    q_bias = f"{prefix}.self_attn.q_proj.bias"
    if q_bias in weight_map:
        qb = get_tensor(q_bias, weight_map, open_files)
        kb = get_tensor(f"{prefix}.self_attn.k_proj.bias", weight_map, open_files)
        vb = get_tensor(f"{prefix}.self_attn.v_proj.bias", weight_map, open_files)
        output[f"{prefix}.self_attn.qkv_proj.bias"] = torch.cat(
            [
                qb[rank * q_per_rank : (rank + 1) * q_per_rank],
                kb[rank * kv_per_rank : (rank + 1) * kv_per_rank],
                vb[rank * kv_per_rank : (rank + 1) * kv_per_rank],
            ]
        ).contiguous()

    return output


def process_o_proj(
    prefix: str,
    tp: int,
    rank: int,
    weight_map: WeightMap,
    open_files: OpenFiles,
) -> TensorDict:
    """Row-parallel split for the output projection (``o_proj``).

    The AWQ weight components are split along the input (row) dimension.
    Bias, if present, is replicated across all ranks.

    Args:
        prefix: Layer name prefix (e.g. ``model.layers.0``).
        tp: Tensor parallel world size.
        rank: This rank's index.
        weight_map: Tensor name to filename mapping.
        open_files: Opened safetensors file handles.

    Returns:
        A dict mapping ``o_proj`` tensor names to their row-split tensors.
    """
    output: TensorDict = {}
    for suffix in AWQ_SUFFIXES:
        name = f"{prefix}.self_attn.o_proj.{suffix}"
        if name in weight_map:
            output[name] = split_row(get_tensor(name, weight_map, open_files), tp, rank)

    bias_name = f"{prefix}.self_attn.o_proj.bias"
    if bias_name in weight_map:
        output[bias_name] = get_tensor(bias_name, weight_map, open_files)

    return output


# ---------------------------------------------------------------------------
# Layer processing — MLP (dense)
# ---------------------------------------------------------------------------


def process_dense_mlp(
    prefix: str,
    tp: int,
    rank: int,
    weight_map: WeightMap,
    open_files: OpenFiles,
) -> TensorDict:
    """Process a dense (non-MoE) MLP block.

    Fuses ``gate_proj`` and ``up_proj`` into a single ``gate_up_proj``
    (column-parallel split), and row-splits ``down_proj``.

    Args:
        prefix: Layer name prefix (e.g. ``model.layers.0``).
        tp: Tensor parallel world size.
        rank: This rank's index.
        weight_map: Tensor name to filename mapping.
        open_files: Opened safetensors file handles.

    Returns:
        A dict mapping fused ``gate_up_proj`` and split ``down_proj``
        tensor names to their tensors.
    """
    output: TensorDict = {}

    for suffix in AWQ_SUFFIXES:
        gate_name = f"{prefix}.mlp.gate_proj.{suffix}"
        if gate_name not in weight_map:
            continue
        gate = get_tensor(gate_name, weight_map, open_files)
        up = get_tensor(f"{prefix}.mlp.up_proj.{suffix}", weight_map, open_files)
        fused = torch.cat([gate, up], dim=-1)
        output[f"{prefix}.mlp.gate_up_proj.{suffix}"] = split_column(fused, tp, rank)

    for suffix in AWQ_SUFFIXES:
        name = f"{prefix}.mlp.down_proj.{suffix}"
        if name in weight_map:
            output[name] = split_row(get_tensor(name, weight_map, open_files), tp, rank)

    return output


# ---------------------------------------------------------------------------
# Layer processing — MoE experts
# ---------------------------------------------------------------------------


def process_moe_experts(
    prefix: str,
    params: ArchParams,
    tp: int,
    ep: int,
    rank: int,
    weight_map: WeightMap,
    open_files: OpenFiles,
) -> TensorDict:
    """Fuse per-expert gate+up into ``w13`` and stack down into ``w2``.

    Expert parallelism (EP) distributes whole experts across ranks. Within
    each EP group, remaining tensor parallelism splits expert weight
    matrices by dimension. With ``EP=TP`` (e.g. both 2), each rank receives
    half the experts without any dimension splitting.

    The output tensors are 3-D: ``(num_local_experts, packed_in, out)``.

    Args:
        prefix: Layer name prefix (e.g. ``model.layers.0``).
        params: Architecture parameters (must include ``num_experts``).
        tp: Tensor parallel world size.
        ep: Expert parallel size.
        rank: This rank's global index.
        weight_map: Tensor name to filename mapping.
        open_files: Opened safetensors file handles.

    Returns:
        A dict mapping ``w13_<suffix>`` and ``w2_<suffix>`` tensor names
        to their (possibly EP/TP-split) 3-D expert weight tensors.
    """
    output: TensorDict = {}
    num_experts = int(params["num_experts"])

    if ep > 1:
        experts_per_rank = num_experts // ep
        expert_start = rank * experts_per_rank
        expert_end = expert_start + experts_per_rank
        moe_tp = tp // ep
        moe_tp_rank = 0
    else:
        experts_per_rank = num_experts
        expert_start = 0
        expert_end = num_experts
        moe_tp = tp
        moe_tp_rank = rank

    for suffix in AWQ_SUFFIXES:
        first_gate = f"{prefix}.mlp.experts.{expert_start}.gate_proj.{suffix}"
        if first_gate not in weight_map:
            continue

        # Probe shapes from the first expert to pre-allocate output tensors
        ref = get_tensor(first_gate, weight_map, open_files)
        in_packed = ref.shape[0]
        gate_out = ref.shape[1]
        dtype = ref.dtype
        del ref

        ref_up = get_tensor(
            f"{prefix}.mlp.experts.{expert_start}.up_proj.{suffix}",
            weight_map,
            open_files,
        )
        up_out = ref_up.shape[1]
        del ref_up

        ref_down = get_tensor(
            f"{prefix}.mlp.experts.{expert_start}.down_proj.{suffix}",
            weight_map,
            open_files,
        )
        down_in_packed = ref_down.shape[0]
        down_out = ref_down.shape[1]
        del ref_down

        w13 = torch.empty(experts_per_rank, in_packed, gate_out + up_out, dtype=dtype)
        w2 = torch.empty(experts_per_rank, down_in_packed, down_out, dtype=dtype)

        for local_idx, global_idx in enumerate(range(expert_start, expert_end)):
            gate = get_tensor(
                f"{prefix}.mlp.experts.{global_idx}.gate_proj.{suffix}",
                weight_map,
                open_files,
            )
            up = get_tensor(
                f"{prefix}.mlp.experts.{global_idx}.up_proj.{suffix}",
                weight_map,
                open_files,
            )
            w13[local_idx, :, :gate_out] = gate
            w13[local_idx, :, gate_out:] = up
            del gate, up

            down = get_tensor(
                f"{prefix}.mlp.experts.{global_idx}.down_proj.{suffix}",
                weight_map,
                open_files,
            )
            w2[local_idx] = down
            del down

        if moe_tp > 1:
            output[f"{prefix}.mlp.experts.w13_{suffix}"] = split_column(w13, moe_tp, moe_tp_rank)
            output[f"{prefix}.mlp.experts.w2_{suffix}"] = split_row(w2, moe_tp, moe_tp_rank)
        else:
            output[f"{prefix}.mlp.experts.w13_{suffix}"] = w13
            output[f"{prefix}.mlp.experts.w2_{suffix}"] = w2
        del w13, w2

    return output


def process_shared_expert(
    prefix: str,
    tp: int,
    rank: int,
    weight_map: WeightMap,
    open_files: OpenFiles,
) -> TensorDict:
    """Process the shared expert in a MoE layer.

    Fuses ``shared_expert.gate_proj`` and ``shared_expert.up_proj`` into
    ``shared_expert.gate_up_proj`` (column-parallel), and row-splits
    ``shared_expert.down_proj``.

    Args:
        prefix: Layer name prefix (e.g. ``model.layers.0``).
        tp: Tensor parallel world size.
        rank: This rank's index.
        weight_map: Tensor name to filename mapping.
        open_files: Opened safetensors file handles.

    Returns:
        A dict mapping shared-expert tensor names to their split tensors.
    """
    output: TensorDict = {}

    for suffix in AWQ_SUFFIXES:
        gate_name = f"{prefix}.mlp.shared_expert.gate_proj.{suffix}"
        if gate_name not in weight_map:
            continue
        gate = get_tensor(gate_name, weight_map, open_files)
        up = get_tensor(
            f"{prefix}.mlp.shared_expert.up_proj.{suffix}",
            weight_map,
            open_files,
        )
        fused = torch.cat([gate, up], dim=-1)
        output[f"{prefix}.mlp.shared_expert.gate_up_proj.{suffix}"] = split_column(fused, tp, rank)

    for suffix in AWQ_SUFFIXES:
        name = f"{prefix}.mlp.shared_expert.down_proj.{suffix}"
        if name in weight_map:
            output[name] = split_row(get_tensor(name, weight_map, open_files), tp, rank)

    return output


# ---------------------------------------------------------------------------
# Full layer & global processing
# ---------------------------------------------------------------------------


def process_layer(
    layer_idx: int,
    layer_tensors: list[str],
    params: ArchParams,
    tp: int,
    ep: int,
    rank: int,
    weight_map: WeightMap,
    open_files: OpenFiles,
) -> TensorDict:
    """Process all tensors in a single transformer layer.

    Dispatches to attention (QKV fusion + O-proj split), MLP (dense or MoE),
    and replicates layer-norm weights. MoE detection is automatic based on
    the presence of ``.mlp.experts.`` tensor names.

    Args:
        layer_idx: Zero-based transformer layer index.
        layer_tensors: List of tensor names belonging to this layer.
        params: Architecture parameters.
        tp: Tensor parallel world size.
        ep: Expert parallel size.
        rank: This rank's global index.
        weight_map: Tensor name to filename mapping.
        open_files: Opened safetensors file handles.

    Returns:
        A dict mapping all output tensor names for this layer to their
        processed tensors.
    """
    output: TensorDict = {}
    prefix = f"model.layers.{layer_idx}"

    output.update(process_qkv(prefix, params, tp, rank, weight_map, open_files))
    output.update(process_o_proj(prefix, tp, rank, weight_map, open_files))

    has_experts = any(".mlp.experts." in name for name in layer_tensors)

    if has_experts:
        output.update(process_moe_experts(prefix, params, tp, ep, rank, weight_map, open_files))
        output.update(process_shared_expert(prefix, tp, rank, weight_map, open_files))
        for name in layer_tensors:
            if ".mlp.gate.weight" in name or ".mlp.shared_expert_gate.weight" in name:
                output[name] = get_tensor(name, weight_map, open_files)
    else:
        output.update(process_dense_mlp(prefix, tp, rank, weight_map, open_files))

    for name in layer_tensors:
        if "layernorm" in name or "layer_norm" in name:
            output[name] = get_tensor(name, weight_map, open_files)

    return output


def process_global_tensors(
    tensor_names: list[str],
    params: ArchParams,
    tp: int,
    rank: int,
    weight_map: WeightMap,
    open_files: OpenFiles,
) -> TensorDict:
    """Process non-layer tensors (embeddings, final norm, language model head).

    ``embed_tokens`` and ``lm_head`` are vocabulary-parallel split;
    ``model.norm.weight`` is replicated.

    Args:
        tensor_names: List of global (non-layer) tensor names.
        params: Architecture parameters (needs ``vocab_size``).
        tp: Tensor parallel world size.
        rank: This rank's index.
        weight_map: Tensor name to filename mapping.
        open_files: Opened safetensors file handles.

    Returns:
        A dict mapping global tensor names to their (possibly split) tensors.
    """
    output: TensorDict = {}
    for name in tensor_names:
        tensor = get_tensor(name, weight_map, open_files)
        if "embed_tokens" in name or "lm_head" in name:
            output[name] = split_vocab(tensor, tp, rank, int(params["vocab_size"]))
        else:
            output[name] = tensor
    return output


# ---------------------------------------------------------------------------
# Shard writer
# ---------------------------------------------------------------------------


class ShardWriter:
    """Accumulates tensors and flushes them to safetensors shard files.

    Tensors are buffered in memory until the cumulative size exceeds
    *max_size*, at which point they are written to a numbered shard file
    (``model-rank-{rank}-part-{N}.safetensors``).

    Attributes:
        output_dir: Directory where shard files are written.
        rank: TP rank index used in shard filenames.
        max_size: Byte threshold that triggers a flush (default 5 GiB).
        buffer: Currently accumulated tensors awaiting flush.
        buffer_size: Cumulative byte size of tensors in the buffer.
        part: Next shard part number (incremented after each flush).
    """

    def __init__(self, output_dir: Path, rank: int, max_size: int = 5 * 1024**3) -> None:
        """Initialize the shard writer.

        Args:
            output_dir: Target directory for shard files.
            rank: TP rank index embedded in filenames.
            max_size: Maximum buffer size in bytes before auto-flush.
        """
        self.output_dir = output_dir
        self.rank = rank
        self.max_size = max_size
        self.buffer: TensorDict = {}
        self.buffer_size = 0
        self.part = 0

    def add(self, tensors: TensorDict) -> None:
        """Add tensors to the buffer, flushing if the size threshold is exceeded.

        Args:
            tensors: Mapping of tensor names to tensor data to accumulate.
        """
        for name, tensor in tensors.items():
            self.buffer[name] = tensor
            self.buffer_size += tensor.numel() * tensor.element_size()
        if self.buffer_size >= self.max_size:
            self.flush()

    def flush(self) -> None:
        """Write all buffered tensors to a numbered safetensors shard file.

        Does nothing if the buffer is empty. Resets the buffer after writing.
        """
        if not self.buffer:
            return
        filename = f"model-rank-{self.rank}-part-{self.part}.safetensors"
        filepath = self.output_dir / filename
        size_gb = self.buffer_size / 1024**3
        print(
            f"  Writing {filename} ({len(self.buffer)} tensors, {size_gb:.1f} GB)",
            flush=True,
        )
        save_file(self.buffer, str(filepath))
        self.buffer = {}
        self.buffer_size = 0
        self.part += 1

    def finalize(self) -> None:
        """Flush any remaining tensors and print a summary.

        Must be called after all layers have been processed to ensure the
        last partial shard is written to disk.
        """
        self.flush()
        print(f"  Wrote {self.part} shard file(s)", flush=True)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    """Entry point for CPU-only AWQ model sharding.

    Reads configuration from environment variables, loads the model's
    safetensors files via memory mapping, processes each layer with the
    appropriate TP/EP splits, and writes numbered shard files to the output
    directory. A ``model.safetensors.index.json`` is written last as a
    completion marker.
    """
    model_id = os.environ.get("SGLANG_MODEL", "")
    tp = int(os.environ.get("TP", "2"))
    ep = int(os.environ.get("EP", "1"))
    rank = int(os.environ.get("NODE_RANK", "0"))
    dry_run = os.environ.get("DRY_RUN", "0") == "1"

    if not model_id:
        print("ERROR: SGLANG_MODEL not set", flush=True)
        sys.exit(1)

    model_slug = model_id.replace("/", "--")
    shard_suffix = f"TP{tp}"
    if ep > 1:
        shard_suffix += f"-EP{ep}"
    default_output = f"/root/.cache/huggingface/sharded/{model_slug}-{shard_suffix}"
    output_dir = Path(os.environ.get("SHARD_OUTPUT_DIR", default_output))

    print(f"[rank {rank}] CPU-only AWQ sharding", flush=True)
    print(f"[rank {rank}] Model:   {model_id}", flush=True)
    print(f"[rank {rank}] TP:      {tp}, EP: {ep}, rank: {rank}", flush=True)
    print(f"[rank {rank}] Output:  {output_dir}", flush=True)
    if dry_run:
        print(f"[rank {rank}] *** DRY RUN — no files will be written ***", flush=True)

    if (output_dir / "model.safetensors.index.json").exists() and not dry_run:
        print(f"[rank {rank}] Sharded checkpoint already exists (index.json found), skipping.", flush=True)
        sys.exit(0)

    from huggingface_hub import snapshot_download

    print(f"[rank {rank}] Resolving model path...", flush=True)
    local_path = Path(
        snapshot_download(
            repo_id=model_id,
            cache_dir="/root/.cache/huggingface/hub",
            local_files_only=True,
        )
    )
    print(f"[rank {rank}] Model at: {local_path}", flush=True)

    config, quant_config, weight_map = load_configs(local_path)
    params = get_architecture_params(config)

    bits = int(quant_config["bits"])  # type: ignore[call-overload]
    group_size = int(quant_config["group_size"])  # type: ignore[call-overload]
    pack_factor = 32 // bits

    print(f"[rank {rank}] Architecture: {params['model_type']}", flush=True)
    print(
        f"[rank {rank}] AWQ: {bits}-bit, group_size={group_size}, " f"pack_factor={pack_factor}",
        flush=True,
    )
    print(
        f"[rank {rank}] Layers: {params['num_hidden_layers']}, "
        f"heads: {params['num_attention_heads']}, "
        f"kv_heads: {params['num_key_value_heads']}",
        flush=True,
    )
    if params["is_moe"]:
        experts_local = int(params["num_experts"]) // ep if ep > 1 else int(params["num_experts"])
        print(
            f"[rank {rank}] MoE: {params['num_experts']} experts "
            f"({experts_local} local with EP={ep}), "
            f"moe_intermediate={params['moe_intermediate_size']}",
            flush=True,
        )

    groups = group_tensors_by_layer(weight_map)
    open_files = open_safetensors_files(local_path, weight_map)

    if dry_run:
        all_names: list[str] = []
        for layer_idx in sorted(k for k in groups if k != "global"):
            result = process_layer(
                int(layer_idx),
                groups[layer_idx],
                params,
                tp,
                ep,
                rank,
                weight_map,
                open_files,
            )
            all_names.extend(sorted(result.keys()))
        if "global" in groups:
            result = process_global_tensors(
                groups["global"],
                params,
                tp,
                rank,
                weight_map,
                open_files,
            )
            all_names.extend(sorted(result.keys()))
        print(f"\n[rank {rank}] Output tensor names ({len(all_names)}):", flush=True)
        for name in all_names:
            print(f"  {name}", flush=True)
        sys.exit(0)

    output_dir.mkdir(parents=True, exist_ok=True)
    writer = ShardWriter(output_dir, rank)
    total_tensors = 0

    for layer_idx in sorted(k for k in groups if k != "global"):
        print(
            f"[rank {rank}] Layer {layer_idx}/{int(params['num_hidden_layers']) - 1}",
            flush=True,
        )
        result = process_layer(
            int(layer_idx),
            groups[layer_idx],
            params,
            tp,
            ep,
            rank,
            weight_map,
            open_files,
        )
        total_tensors += len(result)
        writer.add(result)

    if "global" in groups:
        print(f"[rank {rank}] Global tensors", flush=True)
        result = process_global_tensors(
            groups["global"],
            params,
            tp,
            rank,
            weight_map,
            open_files,
        )
        total_tensors += len(result)
        writer.add(result)

    writer.finalize()
    print(f"[rank {rank}] Total output tensors: {total_tensors}", flush=True)

    print(f"[rank {rank}] Copying metadata...", flush=True)
    for item in local_path.iterdir():
        dst = output_dir / item.name
        if dst.exists():
            continue
        if item.suffix in (".bin", ".pt", ".safetensors"):
            continue
        if item.is_dir():
            shutil.copytree(item, dst)
        else:
            shutil.copy(item, dst)

    weight_map_out: dict[str, str] = {}
    for part_idx in range(writer.part):
        filename = f"model-rank-{rank}-part-{part_idx}.safetensors"
        filepath = output_dir / filename
        if filepath.exists():
            with safe_open(str(filepath), framework="pt") as f:
                for key in f.keys():
                    weight_map_out[key] = filename
    index_data: dict[str, object] = {
        "metadata": {"model": model_id, "tp": tp, "ep": ep, "rank": rank, "method": "cpu_shard"},
        "weight_map": weight_map_out,
    }
    (output_dir / "model.safetensors.index.json").write_text(json.dumps(index_data, indent=2))
    print(f"[rank {rank}] CPU sharding complete.", flush=True)


if __name__ == "__main__":
    main()
