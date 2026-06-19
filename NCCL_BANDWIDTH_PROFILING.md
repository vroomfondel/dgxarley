# NCCL Bandwidth Profiling on DGX Spark (GB10) Clusters

A portable recipe for measuring the **real** NCCL all-reduce bandwidth that a
multi-node SGLang deployment actually gets — including the small-message region
that TP decode depends on. Uses `torch.distributed`, **no MPI / no `mpirun`**.

This is the "how to measure" companion to `nccl-transport-test-plan.md` (which is
the cluster-specific RoCE-vs-socket debugging log).

---

## Why this matters: the interleaved QSFP / PCIe x4 trap

The DGX Spark / GB10 (and the MSI EdgeXpert, Dell, ASUS Ascent, … re-spins of the
same board) has **one ConnectX-7** driving **two QSFP ports**. They are **not two
independent rails** — they share a single CX7 behind a **PCIe Gen5 x4** link:

- Spanned correctly, NCCL pulls ~**185–190 Gbit/s** out of the pair.
- Mis-spanned, both links funnel through the one x4 and you cap at ~**92–95 Gbit/s** — roughly **half**.

A clean standalone `all_reduce_perf` busbw number (the big, asymptotic value) can
look healthy while the **serving path is at half rate**, because:

1. The benchmark may run with a *tuned* NCCL config that the SGLang pod does not inherit.
2. The asymptotic busbw saturates at 64 MB+ and **hides the small-message region**.

For **tensor parallelism (TP=N)**, every decoded token does an all-reduce across all
nodes, on the critical path, at **every concurrency level**. Per-GPU compute is tiny
(only the active experts), so the **small-message (KB-range, latency-bound) all-reduce
can dominate per-token decode time**. A halved effective bandwidth there shows up as a
roughly **flat throughput factor across all concurrency levels** — the classic symptom.

> **Key point:** profile the **8 KB – 1 MB** region, not just the headline busbw.
> The default sweep in `nccl_bench.py` starts at 1 MB and would miss exactly this.

---

## The tool: `roles/k8s_dgx/files/nccl_bench.py`

A `torch.distributed` all-reduce sweep (algbw + busbw per size, like
`nccl-tests all_reduce_perf`). Reads `RANK`, `WORLD_SIZE`, `MASTER_ADDR`,
`MASTER_PORT` from the environment and uses `cuda:0` (1 GPU per node).

Sweep range is **env-overridable** (defaults preserve the original 1 MB–1 GB sweep,
so `qsfp_nccl_test.yml` is unchanged):

| Env var | Default | Purpose |
|---------|---------|---------|
| `NCCL_BENCH_MIN_BYTES` | `1M` | sweep floor — **set to `8K` for decode profiling** |
| `NCCL_BENCH_MAX_BYTES` | `1G` | sweep ceiling |
| `NCCL_BENCH_STEP` | `2` | size multiplier per step |
| `NCCL_BENCH_WARMUP` | `5` | warmup iters per size |
| `NCCL_BENCH_ITERS` | `20` | timed iters per size |
| `NCCL_BENCH_SMALL_REGION_BYTES` | `1M` | sizes ≤ this count as the "small/decode" region in the summary |

(`*_BYTES` vars accept a `K`/`M`/`G` suffix, e.g. `8K`, `512M`.)

The summary prints a **`RATIO:`** line — small-region peak busbw as a fraction of the
overall peak. Below `0.4×` it flags `LOW: check NCCL port/channel spanning`.

---

## Method A — in-pod, inherits the real serving config (recommended, portable)

Runs the bench **inside the existing SGLang pods**, so it inherits the exact NCCL
config / interfaces / RoCE path the server actually uses. No MPI, no extra cluster.
Works on any Spark cluster (hand this to anyone running multi-node SGLang).

> ⚠️ **Stop the SGLang server process first** — GPUs are exclusive; running the bench
> alongside a live server contends or OOMs. Do this in a maintenance window. On a
> cluster with a `livenessProbe`/restartPolicy that relaunches the server, prefer
> **Method B** so you never touch the serving pods.

```bash
NS=sglang                              # this repo: {{ sglang_namespace }}
KC="kubectl --context=ht@dgxarley -n $NS"

# 1. Identify the 4 SGLang pods (head + 3 workers) and the head pod IP.
PODS=( $($KC get pods -l app=sglang -o jsonpath='{.items[*].metadata.name}') )   # adjust selector
HEAD=${PODS[0]}
HEAD_IP=$($KC get pod "$HEAD" -o jsonpath='{.status.podIP}')
PORT=29555                             # any free port for the torch rendezvous

# 2. Copy the bench into each pod.
for p in "${PODS[@]}"; do $KC cp roles/k8s_dgx/files/nccl_bench.py "$p":/tmp/nccl_bench.py; done

# 3. CRITICAL — capture the NCCL_* env the *running* server uses, not the pod-spec env.
#    kubectl exec inherits the image/pod-spec env, NOT vars the entrypoint exported
#    at runtime. Pull the real ones from the live process and re-supply them:
$KC exec "$HEAD" -- sh -c 'cat /proc/$(pgrep -f sglang | head -1)/environ | tr "\0" "\n" | grep ^NCCL_'
#    -> feed any NCCL_IB_HCA / NCCL_IB_GID_INDEX / NCCL_NET_GDR_LEVEL / NCCL_SOCKET_IFNAME
#       below if they are not already in the pod spec.

# 4. (stop the SGLang server process in each pod here)

# 5. Launch one rank per pod. Decode-focused range (8K-1G) + NCCL init/net debug.
for i in "${!PODS[@]}"; do
  $KC exec "${PODS[$i]}" -- \
    env MASTER_ADDR="$HEAD_IP" MASTER_PORT="$PORT" WORLD_SIZE=4 RANK="$i" LOCAL_RANK=0 \
        NCCL_BENCH_MIN_BYTES=8K NCCL_BENCH_MAX_BYTES=1G \
        NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=INIT,NET \
        python /tmp/nccl_bench.py &
done
wait

# 6. Read rank 0's table + RATIO line:
$KC logs "$HEAD"      # (or wherever rank 0 ran)
```

---

## Method B — dedicated test pods (this repo's framework, least disruptive)

If you have this repo's Ansible, the `nccl_test` framework deploys **separate** 4-pod
test sets (host / socket / roce) on **VF7**, so it runs **alongside** SGLang on VF0
without touching the serving pods:

```bash
ansible-playbook k8s_dgx.yml --tags nccl_test                           # all variants
ansible-playbook k8s_dgx.yml --tags nccl_test -e nccl_test_transport=roce
kubectl --context=ht@dgxarley -n sglang logs nccl-test-roce-rank0
```

See `roles/k8s_dgx/tasks/nccl_test.yml`. (Note: this path uses `nccl_test.py`, not
`nccl_bench.py`, and its own pod spec — extend there separately if you need the
small-message range in that flow.)

---

## The disambiguator — single-node TP=1 control

The cleanest way to separate **network** from **per-node hardware**: run a model that
fits **one** GB10 at **TP=1**. No inter-node collectives at all.

- TP=1 throughput is **also** low (~same factor) → it's **per-node** (power mode,
  memory clock, thermals — *not* the network). Check `nvidia-smi -q -d POWER,CLOCK`
  under load on all nodes: sustained TGP **and memory clock**, not just SM clock.
- TP=1 throughput is **full speed** → the deficit is in the **multi-node NCCL path** →
  Methods A/B above will show where (which ports/channels, small-message busbw).

---

## Interpreting the results

1. **`NCCL ... INFO` lines** (from `NCCL_DEBUG=INFO`): which HCAs/ports NCCL selected,
   how many channels (`nChannels`), and whether it used `NET/IB` (RoCE) or fell back to
   `NET/Socket`. Diff the in-pod selection against a known-good tuned run.
2. **The `RATIO:` line**: small-region peak busbw vs overall peak.
   - Healthy spanned dual-QSFP: small busbw a solid fraction of peak.
   - `≈ 0.5×` (or the `LOW` flag): the small/decode region is bandwidth-starved —
     consistent with both QSFP links sharing one PCIe Gen5 x4 (the ~92–95 Gbit/s trap),
     or NCCL using only one port. Fix the port/channel spanning (`NCCL_IB_HCA`,
     channel count) and re-measure.
3. **Absolute busbw**: ~185–190 Gbit/s peak ≈ correctly spanned; ~92–95 Gbit/s peak ≈
   mis-spanned x4; ~2 GB/s ≈ silent **socket fallback** (see `nccl-transport-test-plan.md`
   for the privileged / IB-char-device / GID prerequisites).
