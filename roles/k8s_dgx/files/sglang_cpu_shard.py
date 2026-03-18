"""CPU-only AWQ model sharding for SGLang TP loading.

Processes safetensors files iteratively on CPU via memory mapping.
No GPU, no NCCL, no multi-node coordination needed.
Each node runs independently as a K8s Job.

Target SGLang version: 0.5.9-t5

Supported architectures: qwen2 (dense), qwen2_moe (MoE)

Environment variables:
    SGLANG_MODEL          Model ID (e.g. QuantTrio/Qwen3-235B-A22B-Instruct-2507-AWQ)
    TP                    Tensor parallel size (default: 2)
    NODE_RANK             This node's rank (0 or 1)
    HF_TOKEN              HuggingFace token (optional)
    SHARD_OUTPUT_DIR      Override output directory (optional)
    DRY_RUN               If "1", print output tensor names without writing
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
from safetensors import safe_open
from safetensors.torch import save_file

AWQ_SUFFIXES = ("qweight", "qzeros", "scales")


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------

def load_configs(model_path: Path):
    """Load model config, quantize config, and safetensors index."""
    with open(model_path / "config.json") as f:
        config = json.load(f)

    quant_config_path = model_path / "quantize_config.json"
    if not quant_config_path.exists():
        raise FileNotFoundError(f"quantize_config.json not found in {model_path}")
    with open(quant_config_path) as f:
        quant_config = json.load(f)

    index_path = model_path / "model.safetensors.index.json"
    if index_path.exists():
        with open(index_path) as f:
            index = json.load(f)
        weight_map = index["weight_map"]
    else:
        # Single safetensors file — enumerate tensor names
        single = model_path / "model.safetensors"
        if not single.exists():
            raise FileNotFoundError(f"No safetensors files found in {model_path}")
        with safe_open(str(single), framework="pt", device="cpu") as f:
            weight_map = {name: "model.safetensors" for name in f.keys()}

    return config, quant_config, weight_map


def get_architecture_params(config: dict):
    """Extract architecture parameters from config.json."""
    model_type = config.get("model_type", "")

    params = {
        "model_type": model_type,
        "hidden_size": config["hidden_size"],
        "num_attention_heads": config["num_attention_heads"],
        "num_key_value_heads": config.get(
            "num_key_value_heads", config["num_attention_heads"]
        ),
        "num_hidden_layers": config["num_hidden_layers"],
        "vocab_size": config["vocab_size"],
        "head_dim": config.get(
            "head_dim",
            config["hidden_size"] // config["num_attention_heads"],
        ),
    }

    if model_type == "qwen2_moe":
        params["num_experts"] = config["num_experts"]
        params["moe_intermediate_size"] = config["moe_intermediate_size"]
        params["intermediate_size"] = config.get(
            "shared_expert_intermediate_size", config["intermediate_size"]
        )
        params["is_moe"] = True
    elif model_type == "qwen2":
        params["intermediate_size"] = config["intermediate_size"]
        params["is_moe"] = False
    else:
        raise ValueError(
            f"Unsupported model_type: {model_type!r}. "
            "Supported: qwen2, qwen2_moe"
        )

    return params


# ---------------------------------------------------------------------------
# Tensor grouping & file access
# ---------------------------------------------------------------------------

def group_tensors_by_layer(weight_map: dict):
    """Group tensor names by layer number. Non-layer tensors go to 'global'."""
    groups = defaultdict(list)
    layer_re = re.compile(r"model\.layers\.(\d+)\.")
    for name in weight_map:
        m = layer_re.match(name)
        if m:
            groups[int(m.group(1))].append(name)
        else:
            groups["global"].append(name)
    return groups


def open_safetensors_files(model_path: Path, weight_map: dict):
    """Open all unique safetensors files via mmap."""
    files = {}
    for filename in set(weight_map.values()):
        filepath = model_path / filename
        files[filename] = safe_open(str(filepath), framework="pt", device="cpu")
    return files


def get_tensor(name: str, weight_map: dict, open_files: dict) -> torch.Tensor:
    """Load a single tensor by name from the appropriate safetensors file."""
    filename = weight_map[name]
    return open_files[filename].get_tensor(name)


# ---------------------------------------------------------------------------
# TP split primitives
# ---------------------------------------------------------------------------

def split_column(tensor: torch.Tensor, tp: int, rank: int) -> torch.Tensor:
    """Column-parallel split: split last dimension (out_features)."""
    size = tensor.shape[-1]
    assert size % tp == 0, f"Column split: dim={size} not divisible by tp={tp}"
    chunk = size // tp
    return tensor[..., rank * chunk : (rank + 1) * chunk].contiguous()


def split_row(tensor: torch.Tensor, tp: int, rank: int) -> torch.Tensor:
    """Row-parallel split: split packed input dimension.

    2-D tensors: split dim 0.  3-D tensors (experts): split dim 1.
    """
    dim = 0 if tensor.dim() == 2 else 1
    size = tensor.shape[dim]
    assert size % tp == 0, (
        f"Row split: shape={list(tensor.shape)} dim={dim} "
        f"size={size} not divisible by tp={tp}"
    )
    chunk = size // tp
    return torch.narrow(tensor, dim, rank * chunk, chunk).contiguous()


def split_vocab(
    tensor: torch.Tensor, tp: int, rank: int, vocab_size: int
) -> torch.Tensor:
    """Vocab-parallel split: pad to multiple of tp, split dim 0."""
    padded = math.ceil(vocab_size / tp) * tp
    if tensor.shape[0] < padded:
        pad = torch.zeros(
            padded - tensor.shape[0], *tensor.shape[1:], dtype=tensor.dtype
        )
        tensor = torch.cat([tensor, pad], dim=0)
    chunk = padded // tp
    return tensor[rank * chunk : (rank + 1) * chunk].contiguous()


# ---------------------------------------------------------------------------
# Layer processing — attention
# ---------------------------------------------------------------------------

def process_qkv(
    prefix: str, params: dict, tp: int, rank: int,
    weight_map: dict, open_files: dict,
) -> dict:
    """Fuse q/k/v projections into qkv_proj with GQA-aware TP split."""
    output = {}
    head_dim = params["head_dim"]
    num_heads = params["num_attention_heads"]
    num_kv_heads = params["num_key_value_heads"]
    q_per_rank = (num_heads // tp) * head_dim
    kv_per_rank = (num_kv_heads // tp) * head_dim

    # AWQ packed tensors
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

        output[f"{prefix}.self_attn.qkv_proj.{suffix}"] = torch.cat(
            [q_chunk, k_chunk, v_chunk], dim=-1
        ).contiguous()

    # Bias (if present — Qwen2 has attention_bias)
    q_bias = f"{prefix}.self_attn.q_proj.bias"
    if q_bias in weight_map:
        qb = get_tensor(q_bias, weight_map, open_files)
        kb = get_tensor(f"{prefix}.self_attn.k_proj.bias", weight_map, open_files)
        vb = get_tensor(f"{prefix}.self_attn.v_proj.bias", weight_map, open_files)
        output[f"{prefix}.self_attn.qkv_proj.bias"] = torch.cat([
            qb[rank * q_per_rank : (rank + 1) * q_per_rank],
            kb[rank * kv_per_rank : (rank + 1) * kv_per_rank],
            vb[rank * kv_per_rank : (rank + 1) * kv_per_rank],
        ]).contiguous()

    return output


def process_o_proj(
    prefix: str, tp: int, rank: int,
    weight_map: dict, open_files: dict,
) -> dict:
    """Row-parallel split for o_proj.  Bias is replicated."""
    output = {}
    for suffix in AWQ_SUFFIXES:
        name = f"{prefix}.self_attn.o_proj.{suffix}"
        if name in weight_map:
            output[name] = split_row(
                get_tensor(name, weight_map, open_files), tp, rank
            )

    bias_name = f"{prefix}.self_attn.o_proj.bias"
    if bias_name in weight_map:
        output[bias_name] = get_tensor(bias_name, weight_map, open_files)

    return output


# ---------------------------------------------------------------------------
# Layer processing — MLP (dense)
# ---------------------------------------------------------------------------

def process_dense_mlp(
    prefix: str, tp: int, rank: int,
    weight_map: dict, open_files: dict,
) -> dict:
    """Fuse gate+up → gate_up_proj (column split), row-split down_proj."""
    output = {}

    for suffix in AWQ_SUFFIXES:
        gate_name = f"{prefix}.mlp.gate_proj.{suffix}"
        if gate_name not in weight_map:
            continue
        gate = get_tensor(gate_name, weight_map, open_files)
        up = get_tensor(f"{prefix}.mlp.up_proj.{suffix}", weight_map, open_files)
        fused = torch.cat([gate, up], dim=-1)
        output[f"{prefix}.mlp.gate_up_proj.{suffix}"] = split_column(
            fused, tp, rank
        )

    for suffix in AWQ_SUFFIXES:
        name = f"{prefix}.mlp.down_proj.{suffix}"
        if name in weight_map:
            output[name] = split_row(
                get_tensor(name, weight_map, open_files), tp, rank
            )

    return output


# ---------------------------------------------------------------------------
# Layer processing — MoE experts
# ---------------------------------------------------------------------------

def process_moe_experts(
    prefix: str, params: dict, tp: int, rank: int,
    weight_map: dict, open_files: dict,
) -> dict:
    """Fuse per-expert gate+up → w13, stack down → w2, TP-split."""
    output = {}
    num_experts = params["num_experts"]

    for suffix in AWQ_SUFFIXES:
        first_gate = f"{prefix}.mlp.experts.0.gate_proj.{suffix}"
        if first_gate not in weight_map:
            continue

        # Pre-allocate w13 and w2 to avoid holding all expert tensors at once
        ref = get_tensor(first_gate, weight_map, open_files)
        in_packed = ref.shape[0]
        gate_out = ref.shape[1]
        dtype = ref.dtype
        del ref

        ref_up = get_tensor(
            f"{prefix}.mlp.experts.0.up_proj.{suffix}", weight_map, open_files
        )
        up_out = ref_up.shape[1]
        del ref_up

        ref_down = get_tensor(
            f"{prefix}.mlp.experts.0.down_proj.{suffix}", weight_map, open_files
        )
        down_in_packed = ref_down.shape[0]
        down_out = ref_down.shape[1]
        del ref_down

        w13 = torch.empty(
            num_experts, in_packed, gate_out + up_out, dtype=dtype
        )
        w2 = torch.empty(
            num_experts, down_in_packed, down_out, dtype=dtype
        )

        for i in range(num_experts):
            gate = get_tensor(
                f"{prefix}.mlp.experts.{i}.gate_proj.{suffix}",
                weight_map, open_files,
            )
            up = get_tensor(
                f"{prefix}.mlp.experts.{i}.up_proj.{suffix}",
                weight_map, open_files,
            )
            w13[i, :, :gate_out] = gate
            w13[i, :, gate_out:] = up
            del gate, up

            down = get_tensor(
                f"{prefix}.mlp.experts.{i}.down_proj.{suffix}",
                weight_map, open_files,
            )
            w2[i] = down
            del down

        # w13 column-parallel, w2 row-parallel
        output[f"{prefix}.mlp.experts.w13_{suffix}"] = split_column(
            w13, tp, rank
        )
        output[f"{prefix}.mlp.experts.w2_{suffix}"] = split_row(w2, tp, rank)
        del w13, w2

    return output


def process_shared_expert(
    prefix: str, tp: int, rank: int,
    weight_map: dict, open_files: dict,
) -> dict:
    """Fuse shared_expert gate+up → gate_up_proj, row-split down_proj."""
    output = {}

    for suffix in AWQ_SUFFIXES:
        gate_name = f"{prefix}.mlp.shared_expert.gate_proj.{suffix}"
        if gate_name not in weight_map:
            continue
        gate = get_tensor(gate_name, weight_map, open_files)
        up = get_tensor(
            f"{prefix}.mlp.shared_expert.up_proj.{suffix}",
            weight_map, open_files,
        )
        fused = torch.cat([gate, up], dim=-1)
        output[f"{prefix}.mlp.shared_expert.gate_up_proj.{suffix}"] = (
            split_column(fused, tp, rank)
        )

    for suffix in AWQ_SUFFIXES:
        name = f"{prefix}.mlp.shared_expert.down_proj.{suffix}"
        if name in weight_map:
            output[name] = split_row(
                get_tensor(name, weight_map, open_files), tp, rank
            )

    return output


# ---------------------------------------------------------------------------
# Full layer & global processing
# ---------------------------------------------------------------------------

def process_layer(
    layer_idx: int, layer_tensors: list, params: dict,
    tp: int, rank: int, weight_map: dict, open_files: dict,
) -> dict:
    """Process all tensors in a single transformer layer."""
    output = {}
    prefix = f"model.layers.{layer_idx}"

    # Attention: QKV fusion + O proj split
    output.update(process_qkv(prefix, params, tp, rank, weight_map, open_files))
    output.update(process_o_proj(prefix, tp, rank, weight_map, open_files))

    # MLP: detect MoE vs dense by checking for expert tensors
    has_experts = any(".mlp.experts." in name for name in layer_tensors)

    if has_experts:
        output.update(
            process_moe_experts(prefix, params, tp, rank, weight_map, open_files)
        )
        output.update(
            process_shared_expert(prefix, tp, rank, weight_map, open_files)
        )
        # Replicated: router gate, shared_expert_gate
        for name in layer_tensors:
            if ".mlp.gate.weight" in name or ".mlp.shared_expert_gate.weight" in name:
                output[name] = get_tensor(name, weight_map, open_files)
    else:
        output.update(
            process_dense_mlp(prefix, tp, rank, weight_map, open_files)
        )

    # Replicated: layernorms
    for name in layer_tensors:
        if "layernorm" in name or "layer_norm" in name:
            output[name] = get_tensor(name, weight_map, open_files)

    return output


def process_global_tensors(
    tensor_names: list, params: dict, tp: int, rank: int,
    weight_map: dict, open_files: dict,
) -> dict:
    """Process non-layer tensors (embed_tokens, norm, lm_head)."""
    output = {}
    for name in tensor_names:
        tensor = get_tensor(name, weight_map, open_files)
        if "embed_tokens" in name or "lm_head" in name:
            output[name] = split_vocab(tensor, tp, rank, params["vocab_size"])
        else:
            # model.norm.weight — replicated
            output[name] = tensor
    return output


# ---------------------------------------------------------------------------
# Shard writer
# ---------------------------------------------------------------------------

class ShardWriter:
    """Accumulates tensors and flushes to safetensors files at a size threshold."""

    def __init__(self, output_dir: Path, rank: int, max_size: int = 5 * 1024**3):
        self.output_dir = output_dir
        self.rank = rank
        self.max_size = max_size
        self.buffer: dict[str, torch.Tensor] = {}
        self.buffer_size = 0
        self.part = 0

    def add(self, tensors: dict):
        for name, tensor in tensors.items():
            self.buffer[name] = tensor
            self.buffer_size += tensor.numel() * tensor.element_size()
        if self.buffer_size >= self.max_size:
            self.flush()

    def flush(self):
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

    def finalize(self):
        self.flush()
        print(f"  Wrote {self.part} shard file(s)", flush=True)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    model_id = os.environ.get("SGLANG_MODEL", "")
    tp = int(os.environ.get("TP", "2"))
    rank = int(os.environ.get("NODE_RANK", "0"))
    dry_run = os.environ.get("DRY_RUN", "0") == "1"

    if not model_id:
        print("ERROR: SGLANG_MODEL not set", flush=True)
        sys.exit(1)

    model_slug = model_id.replace("/", "--")
    default_output = f"/root/.cache/huggingface/sharded/{model_slug}-TP{tp}"
    output_dir = Path(os.environ.get("SHARD_OUTPUT_DIR", default_output))

    print(f"[rank {rank}] CPU-only AWQ sharding", flush=True)
    print(f"[rank {rank}] Model:   {model_id}", flush=True)
    print(f"[rank {rank}] TP:      {tp}, rank: {rank}", flush=True)
    print(f"[rank {rank}] Output:  {output_dir}", flush=True)
    if dry_run:
        print(f"[rank {rank}] *** DRY RUN — no files will be written ***", flush=True)

    # Check completion marker
    if (output_dir / "model.safetensors.index.json").exists() and not dry_run:
        print(f"[rank {rank}] Sharded checkpoint already exists (index.json found), skipping.", flush=True)
        sys.exit(0)

    # Resolve local model path (model-download initContainer already fetched it)
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

    # Load configs
    config, quant_config, weight_map = load_configs(local_path)
    params = get_architecture_params(config)

    bits = quant_config["bits"]
    group_size = quant_config["group_size"]
    pack_factor = 32 // bits

    print(f"[rank {rank}] Architecture: {params['model_type']}", flush=True)
    print(
        f"[rank {rank}] AWQ: {bits}-bit, group_size={group_size}, "
        f"pack_factor={pack_factor}",
        flush=True,
    )
    print(
        f"[rank {rank}] Layers: {params['num_hidden_layers']}, "
        f"heads: {params['num_attention_heads']}, "
        f"kv_heads: {params['num_key_value_heads']}",
        flush=True,
    )
    if params["is_moe"]:
        print(
            f"[rank {rank}] MoE: {params['num_experts']} experts, "
            f"moe_intermediate={params['moe_intermediate_size']}",
            flush=True,
        )

    # Group tensors by layer
    groups = group_tensors_by_layer(weight_map)

    # Open safetensors files (mmap, cheap)
    open_files = open_safetensors_files(local_path, weight_map)

    if dry_run:
        all_names = []
        for layer_idx in sorted(k for k in groups if k != "global"):
            result = process_layer(
                layer_idx, groups[layer_idx], params,
                tp, rank, weight_map, open_files,
            )
            all_names.extend(sorted(result.keys()))
        if "global" in groups:
            result = process_global_tensors(
                groups["global"], params, tp, rank, weight_map, open_files,
            )
            all_names.extend(sorted(result.keys()))
        print(f"\n[rank {rank}] Output tensor names ({len(all_names)}):", flush=True)
        for name in all_names:
            print(f"  {name}", flush=True)
        sys.exit(0)

    # Process and write shards
    output_dir.mkdir(parents=True, exist_ok=True)
    writer = ShardWriter(output_dir, rank)
    total_tensors = 0

    for layer_idx in sorted(k for k in groups if k != "global"):
        print(
            f"[rank {rank}] Layer {layer_idx}/{params['num_hidden_layers'] - 1}",
            flush=True,
        )
        result = process_layer(
            layer_idx, groups[layer_idx], params,
            tp, rank, weight_map, open_files,
        )
        total_tensors += len(result)
        writer.add(result)

    if "global" in groups:
        print(f"[rank {rank}] Global tensors", flush=True)
        result = process_global_tensors(
            groups["global"], params, tp, rank, weight_map, open_files,
        )
        total_tensors += len(result)
        writer.add(result)

    writer.finalize()
    print(f"[rank {rank}] Total output tensors: {total_tensors}", flush=True)

    # Copy metadata files
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

    # Write index file (serves as completion marker — written last after all parts)
    import json
    weight_map_out = {}
    for part_idx in range(writer.part):
        filename = f"model-rank-{rank}-part-{part_idx}.safetensors"
        filepath = output_dir / filename
        if filepath.exists():
            from safetensors import safe_open
            with safe_open(str(filepath), framework="pt") as f:
                for key in f.keys():
                    weight_map_out[key] = filename
    index_data = {"metadata": {"model": model_id, "tp": tp, "rank": rank, "method": "cpu_shard"}, "weight_map": weight_map_out}
    (output_dir / "model.safetensors.index.json").write_text(json.dumps(index_data, indent=2))
    print(f"[rank {rank}] CPU sharding complete.", flush=True)


if __name__ == "__main__":
    main()
