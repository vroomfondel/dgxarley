# DeepSeek-V4-Flash-FP8 — swap-free weight load on 4×GB10 (SM121)

**Status:** root cause identified by direct measurement; swap-free load achieved
and confirmed. Verified 2026-06-01 on `xomoxcc/dgx-spark-sglang:0.5.12.post1-sm121`.

## TL;DR

Loading `sgl-project/DeepSeek-V4-Flash-FP8` (TP=4) on the 4×GB10 unified-memory
cluster drove each node's memory to the ~127 GB ceiling **during the weight
load** and pushed ~50 GB onto disk swap. The swap was real and necessary —
without it (or with `weight_loader_disable_mmap`) the pod OOM-killed.

Ground-truth profiling (memray + `/proc/self/smaps` + `torch.cuda.memory`
snapshot, all wired into a probe) showed the swapped bytes were **the weight
checkpoint's host `safetensors`-mmap pages**, accumulated by sglang's default
buffered multi-thread loader — **not** CUDA/device memory and **not** a leak.

**The fix:** load the weights with `load_format=fastsafetensors`, with a small
launch-time patch to sglang's `fastsafetensors_weights_iterator` so it works on
our multi-node / no-GDS setup. It streams each shard **disk → 16 MB bounce
buffer → device**, so the full shard never sits in host memory. Result: swap
stays at the ~3 GB baseline (vs ~50 GB), no OOM, 0 restarts.

---

## 1. The problem

GB10 (Grace-Blackwell) has **one unified memory pool** (~127 GB) shared by CPU
and GPU — there is no separate device VRAM to migrate into. During the
`DeepSeek-V4-Flash-FP8` load:

- Per-rank weights are ~73 GB (device-managed, FP8).
- The default loader reads each `.safetensors` shard via **mmap** and copies the
  tensors out. The mmap'd shard pages stay resident as page cache.
- A `dist.monitored_barrier()` at the **end** of `load_model`
  (`model_runner.py:1593`, `wait_all_ranks=True`) makes every rank wait for the
  slowest rank for several minutes. During this wait the loaded weights sit idle
  **and** the shard page cache is still resident.
- Combined, the node hits ~127 GB. With `vm.swappiness=100` (intentional, set by
  `dgx_prepare` `dgx_swap_swappiness`) the kernel pages anonymous/file pages out
  to swap rather than stalling. Swap climbed to ~50–65 GB, then **emptied** once
  the barrier finished and `weight_loader_drop_cache_after_load` ran.

Symptoms observed earlier: kernel SIGKILL (exit 137) at end of load on the
first bring-up → swap was added (KEP-2400 disk-backed swap, Burstable serving
pod). With swap the load survives; the swap is transient (gone at steady state).

The open question the swap left unanswered: **what exactly doubles memory after
the weights are read, and can we avoid it instead of swapping?**

---

## 2. Method — measure, don't guess

We built a reusable, env-gated probe: `roles/k8s_dgx/files/dsv4_memprobe.py`
(activated via `-e sglang_memprobe=1`; copied to dist-packages + a `.pth` by
`sglang_launch.sh`, so it arms in the main launcher **and** every spawned TP
worker). All output → stderr → pod logs → Loki (`grep "[memprobe"`). It captures
**all three layers**, because we did not assume where the memory was:

1. **Per-1.5s memory categories** (`/proc/meminfo`): `cuda_alloc`/`cuda_resv`
   (torch allocator), `dev_used` = `torch.cuda.mem_get_info()` (whole-node on
   unified memory), `cached`, `anon`, `mapped`, `swapused`, plus `/proc/vmstat`
   reclaim/swap counters and the process's own `VmSwap`.
2. **Stack sampler** (`sys._current_frames()` every ~1.5s): a poor-man's
   profiler — **essential**, because sglang logs nothing during the multi-minute
   barrier; the most frequent live stack tells us what is actually running.
3. **`torch.cuda.memory._record_memory_history()` + `_dump_snapshot()`**: the
   authoritative CUDA/device allocation map (stacks + timeline).
4. **`memray.Tracker`** around `load_model` (installed at launch when probing):
   the native (malloc/mmap) allocation high-water-mark by call stack.
5. **`/proc/self/smaps` swap breakdown** when the process's swap first exceeds
   15 GB: *which mappings* are swapped (anon vs file vs `safetensors`-mmap vs
   `/dev/nvidia*`).

Auxiliary infra: query previous/historical pod logs from **Loki**
(`loki.loki.svc:3100`), not `kubectl logs --previous` (which only keeps the
immediately-prior container and is lost across a crash-loop).

---

## 3. What we ruled out (each by measurement)

| Hypothesis                                     | Test                                                   | Result                                                                                                                                                                           |
|------------------------------------------------|--------------------------------------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| CUDA / device memory doubles                   | `torch.cuda` snapshot                                  | 93 GB clean (weights+KV), **0 inactive/transient**. `cuda_alloc` flat at 73 GB through the whole swap window. **Not CUDA.**                                                      |
| Buffered loader never frees cache              | read `weight_utils.py`                                 | True code gap: `buffered_multi_thread_safetensors_weights_iterator` accepts `drop_cache_after_load` but **never calls it** (the single-thread + non-buffered iterators do). But… |
| …so single-thread (which *does* drop) fixes it | `-e ...extra_config={"enable_multithread_load":false}` | **Same** swap (~54 GB). Drop-cache gap is real but **not** the lever.                                                                                                            |
| Disable mmap (read into anon)                  | `weight_loader_disable_mmap=true`                      | **Worse — OOM** (`c10::Error`). Anon read buffers are non-reclaimable; mmap (file-backed, reclaimable/swappable) is the *survivable* path.                                       |
| Lower swappiness / make pod BestEffort         | (rejected)                                             | swappiness=100 is intentional; the swap is real demand, not elective — confirmed below.                                                                                          |
| `fastsafetensors` as-is                        | `load_format=fastsafetensors`                          | **Crash** — `Gloo connectFullMesh timeout`: sglang's wrapper does a `WORLD`-collective load onto `cuda:{world_rank}` (invalid on 1-GPU worker nodes). Needs patching (§5).       |

Also established: the 73 GB weights are **device-managed and do NOT appear in the
process's `/proc` RSS** (steady-state RSS ~3 GB). So the bytes that swap during
load are *host* memory, freed afterwards — a transient, not a steady leak.

---

## 4. Root cause (ground truth)

memray and smaps agreed, with no guessing left:

**`/proc/self/smaps` — what is swapped:**
```
safetensors-mmap = 13.1G   [anon] = 0.9G   [anon:mimalloc] = 0.7G   [heap] = 0.6G
```
The swapped bulk is the **`safetensors` mmap** — the weight checkpoint's host
file pages (growing well past 13 GB toward the peak).

**memray — who allocates (host high-water-mark):**
```
Peak memory usage: 301 GB   (incl. mmap mappings)
Top allocator:  _load_file → weight_utils.py:1060 → 588 GB cumulative   (#1 by far)
```
`weight_utils.py:1060` is exactly `result = {k: f.get_tensor(k) for k in f.keys()}`
inside the **buffered multi-thread** iterator — it mmaps each shard and
materialises its tensors. With the sliding window (≈ `max_workers+1` = 9 shards
in flight) and the long barrier holding everything resident, the host-side
checkpoint mmap accumulates and, under swappiness=100, is paged out.

**Conclusion:** on GB10's single unified pool, loading 73 GB of weights through
a **host** path needs the bytes transiently present **both** host-side (mmap)
**and** device-resident → exceeds 127 GB → must swap (mmap path survives) or OOM
(anon path). The swap was the correct mitigation; it was hiding a host-side
load transit, not a CUDA doubling.

---

## 5. The fix — stream straight to device

The only way to remove the host transit is to not stage full shards in host
memory. `fastsafetensors` does exactly this: in **nogds** mode (GPU Direct
Storage is **not** available on GB10 — `nvidia_fs` not loaded) it streams each
file **disk → 16 MB bounce buffer → device**. The full shard never lands in
host memory.

sglang already supports `load_format=fastsafetensors`, but its
`fastsafetensors_weights_iterator` is written for single-node multi-GPU and
breaks on our cluster. A launch-time source patch (in `sglang_launch.sh`,
alongside the existing kv_lora / indexer patches; inert unless
`load_format=fastsafetensors`) rewrites the three offending lines:

| sglang default (broken here)            | patched                                                                                                                                                                                                                       |
|-----------------------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `pg = torch.distributed.group.WORLD`    | `pg = SingleGroup()` — each rank loads its files **independently**; no cross-node collective → no `Gloo connectFullMesh` timeout                                                                                              |
| `device = torch.device(f"cuda:{rank}")` | `device = torch.device("cuda", torch.cuda.current_device())` — the **local** device with an explicit index (bare `"cuda"` is rejected by fastsafetensors' `set_device`; `cuda:{world_rank}` is invalid on 1-GPU worker nodes) |
| `SafeTensorsFileLoader(pg, device)`     | `SafeTensorsFileLoader(pg, device, nogds=True)` — bounce-buffer streaming (no GDS)                                                                                                                                            |

TP sharding is unchanged: each rank reads the full tensors and the existing
per-parameter `weight_loader` slices its TP portion — exactly as the normal
`safetensors` iterator yields full tensors. (Per-rank independent reads are
redundant on disk but bounded in memory, identical to the normal loader's I/O
pattern.)

---

## 6. Results

memprobe trace, default mmap loader vs patched fastsafetensors (head node):

| metric                     | default (mmap, buffered) | **patched fastsafetensors**         |
|----------------------------|--------------------------|-------------------------------------|
| `cached` (shard mmap)      | ~28 GB                   | **~10 GB**                          |
| `dev_used` peak            | ~128 GB                  | ~103–128 GB                         |
| **`swapused` during load** | **~50–65 GB**            | **~3 GB (baseline — no load swap)** |
| pod restarts               | (rode swap; transient)   | **0**                               |
| failure mode w/o swap      | OOM                      | n/a (no swap needed)                |

The swap line stays flat at the ~3 GB baseline across the **entire** load,
instead of climbing to ~50 GB. The host-side checkpoint pileup that memray/smaps
pinpointed simply never forms.

---

## 7. How to use

```bash
# swap-free load (the patch is inert unless this load_format is set)
ansible-playbook k8s_dgx.yml --tags sglang -e sglang_load_format=fastsafetensors

# to re-measure with the probe (memray + CUDA history + smaps + stack sampler):
ansible-playbook k8s_dgx.yml --tags sglang -e sglang_memprobe=1 -e sglang_load_format=fastsafetensors
# then in Loki:  {namespace="sglang"} |~ "[memprobe"
#   memray .bin / cuda .pickle land in the pod's /tmp/dsv4_*.{bin,pickle}
```

To make it the default for this model, set `load_format: fastsafetensors` in
`roles/k8s_dgx/model_profiles/sgl-project-deepseek-v4-flash-fp8.yml` (currently
`auto`). Swap (KEP-2400) can stay as a safety margin — with this loader it is
no longer exercised during load.

Prerequisite: `fastsafetensors` must be in the image (it is — v0.3.2).

---

## 8. Files

| File                                   | Change                                                                                                                                                                                                           |
|----------------------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `roles/k8s_dgx/files/sglang_launch.sh` | launch-time source patch of `fastsafetensors_weights_iterator` (SingleGroup + local device-with-index + `nogds=True`); also writes/activates the memprobe `.pth` + `pip install memray` when `SGLANG_MEMPROBE=1` |
| `roles/k8s_dgx/files/dsv4_memprobe.py` | the reusable probe (CUDA history + memray + smaps + stack sampler + meminfo categories), env-gated                                                                                                               |
| `roles/k8s_dgx/tasks/sglang.yml`       | `SGLANG_MEMPROBE` / `SGLANG_MODEL_LOADER_EXTRA_CONFIG` env, probe file in the launch ConfigMap, checksum wiring                                                                                                  |
| `roles/k8s_dgx/defaults/main.yml`      | `sglang_memprobe` default                                                                                                                                                                                        |

Reusable diagnostics worth keeping: the memprobe (both-layer + stack sampler +
smaps), and querying Loki for historical/crashed-container logs.

---

## 9. Why the obvious alternatives don't work here

- **Don't lower `vm.swappiness`** — it is intentional; and the swap was real
  demand (host transit exceeding RAM), not elective.
- **Don't use `weight_loader_disable_mmap`** — anon read buffers are not
  reclaimable; it OOMs (worse than mmap+swap).
- **Reducing load threads** (`enable_multithread_load:false`, `num_threads:1`)
  doesn't help — the host checkpoint pages accumulate regardless of concurrency.
- **`fastsafetensors` GDS** is unavailable on GB10 (no `nvidia_fs`); the **nogds**
  bounce-buffer path is what delivers the win.
- **Stock `fastsafetensors` load** crashes multi-node (WORLD collective +
  `cuda:{world_rank}`) — hence the §5 patch.
