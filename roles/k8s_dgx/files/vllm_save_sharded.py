"""Pre-shard a vLLM model for fast multi-node TP loading.

Runs as a multi-node job (one process per TP rank). Each rank's LLM
loads only its portion of the model, then saves its shard locally.
After completion each node has its rank's files under the sharded path.

Architecture:
    LLM() on rank 1 (worker) BLOCKS — it runs the engine event loop
    in-process and never returns. The worker receives the
    save_sharded_state call via the model executor from the head.
    Therefore, only rank 0's script calls save_sharded_state(); rank 1
    just needs to start LLM() and stay alive.

Environment variables (set via ConfigMap):
    VLLM_MODEL            Model ID (e.g. QuantTrio/Qwen3-235B-A22B-Instruct-2507-AWQ)
    VLLM_QUANTIZATION     Quantization method (e.g. moe_wna16, gptq, or empty)
    TP                    Tensor parallel size (default: 2)
    EP                    Expert parallel size (default: 1). If > 1, enables
                          expert parallelism (boolean flag in vLLM).
    NNODES                Number of nodes (default: 2)
    NODE_RANK             This node's rank (0 = head, 1 = worker)
    QSFP_IP_SPARK1        NCCL init address (head IP)
    NCCL_PORT             NCCL bootstrap port
    HF_TOKEN              HuggingFace token (optional)
    SHARD_OUTPUT_DIR      Override output directory (optional)

After sharding, start vLLM with:
    --model <sharded-dir> --load-format sharded_state
"""

import os
import shutil
import sys
from pathlib import Path


def main() -> None:
    """Entry point for the vLLM multi-node sharding job.

    Reads configuration from environment variables, ensures the model is
    downloaded locally, then initialises a distributed vLLM ``LLM``
    instance across all nodes.  On rank 0 (head) ``save_sharded_state``
    is called to write shard files and model metadata to *output_dir*.
    On rank != 0 (workers) ``LLM()`` blocks indefinitely handling
    executor RPCs from the head; the process exits once the head
    disconnects and NCCL sends SIGQUIT.

    Environment variables consumed:
        VLLM_MODEL: HuggingFace model ID.
        VLLM_QUANTIZATION: Optional quantization method string.
        TP: Tensor-parallel size (default ``2``).
        EP: Expert-parallel size (default ``1``).
        NNODES: Total node count (default ``2``).
        NODE_RANK: Rank of this process (default ``0``).
        QSFP_IP_SPARK1: NCCL master address (default ``10.10.10.101``).
        NCCL_PORT: NCCL bootstrap port (default ``50000``).
        SHARD_OUTPUT_DIR: Override the computed output directory.
        SGLANG_ENABLE_JIT_DEEPGEMM: If ``"false"``, log that DeepGemm
            JIT is disabled (env passthrough, no functional effect here).

    Raises:
        SystemExit: With code 1 when ``VLLM_MODEL`` is not set; with
            code 0 on successful completion or when an existing sharded
            checkpoint is detected.
    """
    model_id = os.environ.get("VLLM_MODEL", "")
    quantization: str | None = os.environ.get("VLLM_QUANTIZATION", "") or None
    tp = int(os.environ.get("TP", "2"))
    ep = int(os.environ.get("EP", "1"))
    nnodes = int(os.environ.get("NNODES", "2"))
    node_rank = int(os.environ.get("NODE_RANK", "0"))
    master_addr = os.environ.get("QSFP_IP_SPARK1", "10.10.10.101")
    nccl_port = os.environ.get("NCCL_PORT", "50000")

    if not model_id:
        print("ERROR: VLLM_MODEL not set", flush=True)
        sys.exit(1)

    model_slug = model_id.replace("/", "--")
    shard_suffix = f"vllm-TP{tp}"
    if ep > 1:
        shard_suffix += f"-EP{ep}"
    if quantization:
        shard_suffix += f"-{quantization}"
    default_output = f"/root/.cache/huggingface/sharded/{model_slug}-{shard_suffix}"
    output_dir = os.environ.get("SHARD_OUTPUT_DIR", default_output)

    print(f"[rank {node_rank}] Model:        {model_id}", flush=True)
    print(f"[rank {node_rank}] Quantization: {quantization or '(none)'}", flush=True)
    print(f"[rank {node_rank}] TP size:      {tp}, EP size: {ep}, nnodes: {nnodes}, rank: {node_rank}", flush=True)
    print(f"[rank {node_rank}] NCCL init:    {master_addr}:{nccl_port}", flush=True)
    print(f"[rank {node_rank}] Output:       {output_dir}", flush=True)

    # Check if already sharded (index.json is written last by ShardedStateLoader)
    if (Path(output_dir) / "model.safetensors.index.json").exists():
        print(f"[rank {node_rank}] Sharded checkpoint already exists (index.json found), skipping.", flush=True)
        sys.exit(0)

    # Ensure the model is downloaded locally
    from huggingface_hub import snapshot_download

    print(f"[rank {node_rank}] Ensuring model is downloaded...", flush=True)
    local_path = snapshot_download(
        repo_id=model_id,
        cache_dir="/root/.cache/huggingface/hub",
    )
    print(f"[rank {node_rank}] Model cached at: {local_path}", flush=True)

    # Set MASTER_ADDR / MASTER_PORT for torch.distributed (vLLM uses these)
    os.environ["MASTER_ADDR"] = master_addr
    os.environ["MASTER_PORT"] = nccl_port

    # Create LLM with multi-node TP params
    from vllm import LLM  # type: ignore[import-not-found]

    llm_kwargs: dict[str, object] = {
        "model": local_path,
        "tensor_parallel_size": tp,
        # Multi-node without Ray requires "mp" executor backend
        "distributed_executor_backend": "mp",
        # Shard job only needs to load weights and save — no inference.
        # Use minimal context and disable CUDA graph to reduce memory pressure.
        "gpu_memory_utilization": 0.60,
        "max_model_len": 128,
        "enforce_eager": True,
    }
    if ep > 1:
        llm_kwargs["enable_expert_parallel"] = True
    if quantization:
        llm_kwargs["quantization"] = quantization

    if os.environ.get("SGLANG_ENABLE_JIT_DEEPGEMM", "").lower() == "false":
        print(f"[rank {node_rank}] DeepGemm JIT disabled (env passthrough)", flush=True)

    # Ensure output dir exists before LLM init
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    print(f"[rank {node_rank}] Initializing LLM...", flush=True)

    if node_rank != 0:
        # Worker: LLM() BLOCKS on rank != 0 — it runs the engine event loop
        # in-process and never returns. The model executor receives the
        # save_sharded_state call via broadcast from the head and writes the
        # rank-1 shard files to output_dir. When the head exits and NCCL
        # disconnects, vLLM sends SIGQUIT killing this process. The launch
        # shell script handles post-save cleanup (metadata copy + marker).
        llm = LLM(**llm_kwargs)
        # If LLM() ever returns on worker, just exit cleanly.
        del llm
        sys.exit(0)
    else:
        # Head: LLM() returns on rank 0, allowing us to call save.
        llm = LLM(**llm_kwargs)

        print(f"[rank {node_rank}] Saving sharded state to {output_dir}...", flush=True)
        llm.llm_engine.model_executor.save_sharded_state(
            path=output_dir,
            pattern=None,
            max_size=5 * 1024**3,  # 5 GB per shard file
        )
        print(f"[rank {node_rank}] save_sharded_state returned.", flush=True)

    # Copy metadata files (config.json, tokenizer, etc.)
    for file in os.listdir(local_path):
        src = os.path.join(local_path, file)
        dst = os.path.join(output_dir, file)
        if os.path.exists(dst):
            continue
        ext = os.path.splitext(file)[1]
        if ext in (".bin", ".pt", ".safetensors"):
            continue
        if os.path.isdir(src):
            shutil.copytree(src, dst)
        else:
            shutil.copy(src, dst)

    # model.safetensors.index.json is written by ShardedStateLoader.save_sharded_state
    # as the last step — serves as the completion marker.
    print(f"[rank {node_rank}] Sharding complete.", flush=True)


if __name__ == "__main__":
    main()
