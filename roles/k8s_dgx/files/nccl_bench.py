"""NCCL all-reduce benchmark using torch.distributed.

Initializes a 2-rank process group over NCCL, then measures all-reduce
throughput at various buffer sizes. Equivalent to nccl-tests all_reduce_perf
but uses torch.distributed for multi-node init (no MPI needed).

Environment variables (set by K8s pod spec):
  MASTER_ADDR, MASTER_PORT, RANK, WORLD_SIZE
  NCCL_SOCKET_IFNAME, NCCL_DEBUG
"""

import os
import time
import torch
import torch.distributed as dist

WARMUP_ITERS = 5
BENCH_ITERS = 20
MIN_BYTES = 1 * 1024 * 1024  # 1 MB
MAX_BYTES = 1024 * 1024 * 1024  # 1 GB
STEP_FACTOR = 2


def main() -> None:
    """Run the NCCL all-reduce benchmark across all distributed ranks.

    Reads RANK, WORLD_SIZE, MASTER_ADDR, and MASTER_PORT from environment
    variables to initialize the NCCL process group. Iterates over buffer
    sizes from MIN_BYTES to MAX_BYTES (doubling each step), runs WARMUP_ITERS
    warmup iterations followed by BENCH_ITERS timed iterations, and prints
    per-size throughput (algorithm bandwidth and bus bandwidth). Rank 0 also
    prints a final summary with peak and average bus bandwidth.
    """
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    dist.init_process_group(
        backend="nccl",
        init_method=f"tcp://{os.environ['MASTER_ADDR']}:{os.environ['MASTER_PORT']}",
        rank=rank,
        world_size=world_size,
    )

    device = torch.device("cuda:0")
    torch.cuda.set_device(device)

    if rank == 0:
        print(f"torch.distributed initialized: {world_size} ranks, backend=nccl")
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA: {torch.version.cuda}, PyTorch: {torch.__version__}")
        print()
        print(
            f"{'size':>12s}  {'count':>12s}  {'type':>6s}  {'redop':>5s}  {'time_us':>10s}  {'algbw_GBs':>10s}  {'busbw_GBs':>10s}"
        )

    results: list[tuple[int, int, float, float, float]] = []

    size = MIN_BYTES
    while size <= MAX_BYTES:
        count = size // 4  # float32 = 4 bytes
        buf = torch.randn(count, dtype=torch.float32, device=device)

        # warmup
        for _ in range(WARMUP_ITERS):
            dist.all_reduce(buf, op=dist.ReduceOp.SUM)
        torch.cuda.synchronize()

        # bench
        dist.barrier()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(BENCH_ITERS):
            dist.all_reduce(buf, op=dist.ReduceOp.SUM)
        torch.cuda.synchronize()
        t1 = time.perf_counter()

        avg_s = (t1 - t0) / BENCH_ITERS
        avg_us = avg_s * 1e6
        algbw = size / avg_s / 1e9  # GB/s
        busbw = algbw * 2 * (world_size - 1) / world_size  # bus bandwidth for all-reduce

        if rank == 0:
            results.append((size, count, avg_us, algbw, busbw))
            print(f"{size:12d}  {count:12d}  {'float':>6s}  {'sum':>5s}  {avg_us:10.1f}  {algbw:10.2f}  {busbw:10.2f}")

        del buf
        size *= STEP_FACTOR

    if rank == 0:
        peak_busbw = max(r[4] for r in results)
        peak_size = max(results, key=lambda r: r[4])
        avg_busbw = sum(r[4] for r in results) / len(results)
        print()
        print(
            f"RESULT: {world_size} ranks | peak busbw {peak_busbw:.2f} GB/s ({peak_busbw * 8:.1f} Gbit/s) @ {peak_size[0] // 1024 // 1024}MB | avg busbw {avg_busbw:.2f} GB/s ({avg_busbw * 8:.1f} Gbit/s)"
        )

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
