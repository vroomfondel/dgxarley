# JuiceFS Migration HOWTO — moving the HuggingFace cache off per-node NVMe

Step-by-step guide for migrating the large, duplicated HuggingFace model cache
from per-node local NVMe (`hf_cache_path`, `/var/lib/hf-cache`) onto the shared
JuiceFS filesystem (`roles/juicefs_storage`), while keeping node-local JIT caches
local.

> Status of this doc's changes — the wiring is **already in the repo, gated OFF**:
> - `hf_hub_cache_on_juicefs: false` (`roles/k8s_dgx/defaults/main.yml`) — the master
>   switch. Flipping it to `true` routes the SGLang/vLLM HF cache to JuiceFS and
>   no-ops the rsync fan-out. Everything below is done via **this one flag**, not
>   manual file edits.
> - `juicefs_mount_master: false` (`roles/juicefs_storage/defaults/main.yml`) —
>   opt-in for mounting JuiceFS on the QSFP-less k3s master.
> - Phase 4b (`fa4-cute-dsl-cache` carve-out) is implemented unconditionally.
>
> So "migrating" = deploy the storage layer (Phase 1), seed (Phase 3), then flip
> the flag and redeploy (Phase 4). Reverting = flip back + redeploy.

---

## 1. Scope: what moves, what stays

Today every spark holds a full, independent ~1.2 TiB copy of the cache under
`/var/lib/hf-cache/`, distributed by manual `rsync` (`hf_preload.yml`,
`sglang_tune_moe.yml`). JuiceFS is a single shared namespace, so migrating the
big read-mostly data eliminates both the duplication **and** those rsync steps.

| Subdir               | → JuiceFS?                       | Rationale                                                                               |
|----------------------|----------------------------------|-----------------------------------------------------------------------------------------|
| `hub`                | ✅ **yes** (primary goal)         | The ~1.2 TiB weight cache; read-mostly after download; 4× duplicated today              |
| `xet`                | ✅ yes — **must move with `hub`** | HF `hf_xet` download-accel cache, same `HF_HOME` tree, coupled to `hub`                 |
| `modules`            | ✅ yes                            | Small `trust_remote_code` python cache; no write contention                             |
| `moe_configs`        | ✅ yes                            | Already collision-safe: `triton_<ver>/<gpu_slug>` naming (`sglang_tune_moe.sh:42-53`)   |
| `sharded`            | ✅ yes                            | Per-rank filenames; each node writes/reads only its own rank (`sglang_save_sharded.py`) |
| `flashinfer_cache`   | ❌ **stay local**                 | JIT-compiled kernels, hot path, concurrent per-node compiles → keep node-local          |
| `triton_cache`       | ❌ stay local                     | Same profile as flashinfer                                                              |
| `fa4_cute_dsl_cache` | ❌ stay local                     | Same profile; **was** implicit under the `hf-cache` mount → carved out in Phase 4b      |

**Why the three JIT caches stay local:** their total size is negligible vs
`hub`'s ~1.2 TiB (almost no space saved), but they sit on the inference hot path
and are written concurrently per-node — a shared FS adds latency and write-race
risk for no benefit.

---

## 2. Prerequisites

- The USB-SSD is physically attached to `juicefs_storage_node` (currently
  `spark1`; `roles/juicefs_storage/defaults/main.yml:17`) and mounted at
  `juicefs_disk_path` (`/mnt/jfs-usb`). The role refuses to run if it isn't a
  real mountpoint (`juicefs_require_disk_mount: true`).
- Real secrets set in `group_vars/all/vault.yml` (they are dummies in the role
  defaults): `juicefs_rustfs_access_key`, `juicefs_rustfs_secret_key`,
  `juicefs_valkey_password`.
- Free space check: each spark currently has only ~200 GiB free until the old
  local `hub` copy is removed (Phase 6). Size the cache accordingly (Phase 2).

---

## 3. Phase 1 — Deploy the JuiceFS storage layer

The role runs entirely as host-level systemd services (outside k3s). It is
currently **standalone / unconsumed** — deploying it changes nothing about the
running SGLang pods yet.

```bash
ansible-playbook storage.yml            # binary → Valkey+RustFS → format → mount → backup
```

Verify with the role's acceptance checklist (`roles/juicefs_storage/README.md`):

```bash
# on every mount node (dgxsparks):
mountpoint -q /mnt/jfs && echo OK
# cross-node visibility:
ssh root@spark1 'echo hi > /mnt/jfs/test'      # write on node A
ssh root@spark3 'cat /mnt/jfs/test'            # read on node B  → "hi"
```

---

## 4. Phase 2 — Size the per-node cache for the CURRENT free space

`juicefs_cache_size_mib` is the per-node NVMe read cache (MiB). Until Phase 6
frees the old `hub` copy, only ~200 GiB is free. Set it conservatively **now**,
raise it in Phase 6:

```yaml
# roles/juicefs_storage/defaults/main.yml (or host_vars/<host>/main.yml per node)
juicefs_cache_size_mib: 150_000   # ~146 GiB — safe under the ~200 GiB free today
```

Underscores are fine (`150_000` parses as int `150000`; the `--cache-size` flag
renders without the underscore). Per-node override works via
`host_vars/<host>/main.yml` (normal precedence: role defaults < host_vars).

---

## 5. Phase 3 — Seed the data into JuiceFS (once)

JuiceFS is one shared namespace — seed **once** from any spark that holds the
full copy. Choose the target layout now (see the Phase 4a gotcha below); this
guide maps the JuiceFS root directly to the HF cache dir:

```bash
# on ONE spark, one-time copy of the migrating subdirs into the shared FS:
rsync -a --info=progress2 \
  /var/lib/hf-cache/{hub,xet,modules,moe_configs,sharded} \
  /mnt/jfs/
# verify from a DIFFERENT node:
ssh root@spark3 'ls /mnt/jfs/hub | head'
```

Then warm each node's local cache (per the README):

```bash
# on each mount node:
juicefs warmup /mnt/jfs/hub
```

---

## 6. Phase 4 — Wire the SGLang pods to JuiceFS

### 4a. Flip the flag

The pod wiring is already in `roles/k8s_dgx/tasks/sglang_instance.yml`, gated by
`hf_hub_cache_on_juicefs`. Set it (globally or per-play):

```yaml
# roles/k8s_dgx/defaults/main.yml  (or -e hf_hub_cache_on_juicefs=true)
hf_hub_cache_on_juicefs: true
```

then redeploy the SGLang/vLLM instances (e.g. `ansible-playbook k8s_dgx.yml
--tags sglang`). The gate flips the `hf-cache` volume on every container:

| | `false` (default) | `true` |
|---|---|---|
| hostPath | `hf_cache_path` (`/var/lib/hf-cache`, local NVMe) | `hf_cache_juicefs_root` (`= juicefs_mount_path`, `/mnt/jfs`, JuiceFS) |
| `type` | `DirectoryOrCreate` | `Directory` (fail loudly if JuiceFS isn't mounted) |
| `mountPropagation` | `None` | `HostToContainer` (survive a JuiceFS FUSE remount) |

**Why these exact choices** (the two constraints that shaped the gate):

1. **Propagation safety.** JuiceFS is a `Restart=always` FUSE mount; if it
   remounts, a pod holding a stale bind gets `Transport endpoint is not
   connected`. `HostToContainer` (`README.md:48-66`) is set automatically when
   the flag is on.
2. **Download-script path coupling.** The model-download script hardcodes
   `cache_dir = "/root/.cache/huggingface/hub"`
   (`roles/k8s_dgx/files/hf_preload_download.py:26`, `:154`; a CLAUDE.md rule).
   So the container path `/root/.cache/huggingface` stays fixed and the JuiceFS
   **root** is mapped onto it → `hub` at `/mnt/jfs/hub` matches the hardcoded
   `cache_dir`. This is why **Phase 3 seeds to `/mnt/jfs/` (root)**.
   `hf_cache_juicefs_root` is DERIVED from `juicefs_mount_path` (promoted to
   `group_vars/all/main.yml`), so the HF-cache hostPath can never drift from the
   FUSE mountpoint — change `juicefs_mount_path` and both follow.

> **If you later want JuiceFS to host more than the HF cache**, switch to the
> `HF_HOME=/mnt/jfs/hf` layout instead — but then also update the hardcoded
> `cache_dir` in `hf_preload_download.py` (and the CLAUDE.md rule). The current
> root-mapping is the least-moving-parts choice for an HF-cache-only FS.

### 4b. Carve out `fa4_cute_dsl_cache` — ✅ ALREADY DONE

`fa4_cute_dsl_cache` had **no** volume — it rode inside the `hf-cache` mount via
`FLASH_ATTENTION_CUTE_DSL_CACHE_DIR=/root/.cache/huggingface/fa4_cute_dsl_cache`,
so a naïve Phase 4a would have dragged it onto JuiceFS. It now has its own
node-local volume (mirroring the flashinfer/triton pattern), in
`roles/k8s_dgx/tasks/sglang_instance.yml`:

- env → `FLASH_ATTENTION_CUTE_DSL_CACHE_DIR: "/root/.cache/fa4_cute_dsl_persistent"`
- volume `fa4-cute-dsl-cache` → hostPath `{{ hf_cache_path }}/fa4_cute_dsl_cache`
  (local NVMe), mounted at `/root/.cache/fa4_cute_dsl_persistent` on head+worker.

`flashinfer_cache` and `triton_cache` already had their own local volumes and
need **no change** — leave their hostPath at `{{ hf_cache_path }}/...`.

### 4c. rsync fan-out — auto-disabled by the same flag

With `hub`/`moe_configs` shared, the manual fan-out is redundant. Rather than
delete it (so the flag stays reversible), both tasks emit an **empty**
`RSYNC_TARGETS` when `hf_hub_cache_on_juicefs` is on, and the scripts already no-op
on an empty target list — **no script change**:

- `roles/k8s_dgx/tasks/hf_preload.yml` — `RSYNC_TARGETS` → `''`;
  `hf_preload_download.py:124/150` (`if rsync_targets:`) skips the rsync.
- `roles/k8s_dgx/tasks/sglang_tune_moe.yml` — `RSYNC_TARGETS` → `''`;
  `sglang_tune_moe.sh:63` (`if [ -n "$RSYNC_TARGETS" ]`) skips it.

`sglang_save_sharded.py` / `vllm_save_sharded.py` need **no** change — they write
under `/root/.cache/huggingface/sharded/...`, which now lands on JuiceFS
automatically.

---

## 7. Phase 5 — Cutover & verify (keep the old cache as a safety net)

1. Redeploy one model on one node first. Confirm the boot log reads weights from
   the JuiceFS-backed path and there is **no** re-download via xet (the
   mismatch symptom from the CLAUDE.md rule).
2. Serve a request; verify coherent output.
3. Only then roll out to all instances.
4. **Keep `/var/lib/hf-cache/{hub,xet,...}` in place for a few days** as rollback.

---

## 8. Phase 6 — Reclaim space, then raise the cache size

After a confirmed cutover, on each spark delete only the migrated subdirs
(NOT the three local JIT caches):

```bash
# on each spark — leave flashinfer_cache, triton_cache, fa4_cute_dsl_cache alone:
rm -rf /var/lib/hf-cache/{hub,xet,modules,moe_configs,sharded}
```

That frees ~1.2 TiB/node → now raise the per-node cache back up:

```yaml
juicefs_cache_size_mib: 600_000   # ~586 GiB, once the local hub copy is gone
```

---

## 9. Acceptance checklist

1. `mountpoint -q /mnt/jfs` true on every mount node.
2. A model directory written on node A is visible on node B under `/mnt/jfs/hub`.
3. New objects appear in the RustFS `juicefs` bucket after a download.
4. `juicefs warmup /mnt/jfs/hub` fills the local cache; a re-read produces no
   backend traffic.
5. A fresh SGLang boot does NOT re-download weights (no xet re-fetch in the log).
6. `flashinfer_cache` / `triton_cache` / `fa4_cute_dsl_cache` still resolve to
   `/var/lib/hf-cache/...` (local), not `/mnt/jfs`.
7. `juicefs dump` meta-backup lands in the rasnas bucket.

---

## 10. Rollback

Set `hf_hub_cache_on_juicefs: false` and redeploy. Because Phase 6 is deferred until
after a confirmed cutover, the full local copy is still on each spark, so
rollback is a flag flip + redeploy with no data loss.

---

## 11. Firewall

The host firewall (`roles/common/templates/iptables.sh.j2`) already trusts the
**entire QSFP subnet** (`qsfp_network` = `10.10.10.0/24`, blanket `RETURN` at
`:63-65`). All spark mount clients dial the storage node over QSFP, so Valkey
(6379) and RustFS (9000/9001) traffic between sparks is **already open — no
firewall change for the default (spark-only) migration.**

**Exception — `juicefs_mount_master: true`:** the master reaches the storage node
over the **mgmt/k3s VLAN**, where 6379/9000 are otherwise closed. This is now
**wired**: `roles/common/templates/iptables.sh.j2` emits, **only on the storage
node and only when `juicefs_mount_master` is on**, a rule opening 6379 + 9000
from `k3snodes`:

```
iptables -A HTSTUFFIN -m state --state NEW -p tcp -m multiport --dport 6379,9000 \
  -m set --match-set k3snodes src -j RETURN
```

To make this possible, `juicefs_storage_node` and `juicefs_mount_master` were
promoted to `group_vars/all/main.yml` (the `common` role needs cross-role
visibility). So enabling master-mount is: set `juicefs_mount_master: true`, then
re-run `common.yml` (firewall) **and** `storage.yml` (mount + dual-bind).

---

## 12. Mounting on the k3s master (optional, `juicefs_mount_master`)

Off by default and **not needed for the HF-cache use case** (SGLang runs only on
sparks). If you do want the master as a mount client:

1. Set `juicefs_mount_master: true` (`roles/juicefs_storage/defaults/main.yml`).
   This auto-engages: the master gets the mount (`tasks/main.yml` gate), resolves
   `juicefs_storage_address` to the storage node's k3s IP (per-client logic), and
   dual-homes Valkey on the storage node (`juicefs_valkey_bind_addresses`).
2. Add the firewall rule from §11 on the storage node.
3. Redeploy the storage layer: `ansible-playbook storage.yml`.

Caveat: the master has no QSFP, so it reaches the backend over the slower mgmt
VLAN, not the 200GbE mesh — fine for light use, not for streaming large weights.

---

## 13. Open decisions before starting

- **Storage node placement:** `juicefs_storage_node` (currently `spark1`) holds
  the USB-SSD and runs Valkey+RustFS — single point of failure for the metadata
  engine (mitigated by the meta-backup timer, not eliminated).
- **JuiceFS scope:** HF-cache-only (Phase 4a recommended path) vs. general
  shared FS (the README `HF_HOME` alternative). Decide before seeding, since it
  fixes the on-disk layout (`/mnt/jfs/hub` vs `/mnt/jfs/hf/hub`).
- **JIT caches later:** revisit moving `flashinfer_cache`/`triton_cache` to
  JuiceFS only under renewed space pressure — currently a poor effort/risk ratio.
