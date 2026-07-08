# JuiceFS Migration HOWTO — moving the HuggingFace cache off per-node NVMe

Step-by-step guide for migrating the large, duplicated HuggingFace model cache
from per-node local NVMe (`hf_cache_path`, `/var/lib/hf-cache`) onto the shared
JuiceFS filesystem (`roles/juicefs_storage`), while keeping node-local JIT caches
local.

> Status of this doc's changes — the wiring is **already in the repo, gated OFF**:
> - `hf_hub_cache_on_juicefs: false` (`roles/k8s_dgx/defaults/main/hf_cache.yml`) — the master
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

- The USB disk is physically attached to `juicefs_storage_node` and mounted at
  `juicefs_disk_path`. The real values for both live in
  `group_vars/all/vault/juicefs.yml` (public dummies: `spark1` in
  `group_vars/all/main/juicefs.yml`, `/mnt/jfs-usb` in the role defaults). The role
  refuses to run if the path isn't a real mountpoint
  (`juicefs_require_disk_mount: true`).
- Real secrets set in `group_vars/all/vault/juicefs.yml` (they are dummies in the role
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
# roles/k8s_dgx/defaults/main/hf_cache.yml  (or -e hf_hub_cache_on_juicefs=true)
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
   `group_vars/all/main/juicefs.yml`), so the HF-cache hostPath can never drift from the
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
promoted to `group_vars/all/main/juicefs.yml` (the `common` role needs cross-role
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

## 13. Moving the USB disk / storage node to another spark

Relocating the storage role from the current `juicefs_storage_node` (vault) to
another spark is **three moves, not one**, because the pieces live in different
places. Placeholders below: `<old-qsfp-ip>` / `<new-qsfp-ip>` = the QSFP IPs of
the old/new storage node, `<disk-path>` = `juicefs_disk_path` (vault),
`sparkN` = the target node.

| Piece | Lives where | Moves with the disk? |
|---|---|---|
| Object data (all chunks) | `{{ juicefs_disk_path }}/rustfs` on the USB disk | ✅ yes |
| **Metadata** (inodes, dir tree, chunk map) | Valkey — `/var/lib/valkey` on the storage **node's local disk** | ❌ **no** — must be dumped/loaded |
| Bucket endpoint URL | **inside the metadata** (baked at format time: `http://<old QSFP IP>:9000/juicefs`) | ❌ must be rewritten (`juicefs config`) |
| Valkey + RustFS services | systemd units on the storage node | ❌ redeployed by the role |
| Client mount units | every mount node, dialing the old node's QSFP IP | ❌ re-rendered by the role |

> ⚠️ **Never run a plain `ansible-playbook storage.yml` while the NEW node's
> Valkey is still empty but the bucket already holds data.** The format step is
> guarded only by `juicefs status` against Valkey — an empty Valkey looks
> "unformatted", so it would format a FRESH filesystem of the same name into the
> same bucket, orphaning/colliding with every existing object. During a move,
> restore the metadata dump FIRST; bring the backend up with
> `--tags juicefs_bin,juicefs_backend,juicefs_backup` only.

**Downtime scope (both variants):** everything under `/mnt/jfs` is unavailable
while the backend moves. With `hf_hub_cache_on_juicefs: true`, scale the
SGLang/vLLM deployments to 0 first (the pods hold the FUSE mount). Traffic
between sparks runs over the QSFP subnet, which the firewall blanket-trusts —
no iptables change. (Exception: with `juicefs_mount_master: true`, the
master-mount rule keys on `juicefs_storage_node` → also re-run `common.yml`.)

### 13.1 Variant A — second (initially empty) disk on the target spark

Downtime is short: the bulk copy happens over QSFP **while everything runs**,
only the delta + metadata dance needs the stop window.

1. **Prepare the new disk** on `sparkN`: partition/mkfs yourself (the role never
   touches block devices), add an fstab entry (`UUID=... <mountpoint> ext4
   defaults,nofail 0 2`), mount it. If the mountpoint differs from the current
   `juicefs_disk_path`, update that var in `group_vars/all/vault/juicefs.yml` later in
   step 5.
2. **Live pre-sync** of the object store over QSFP (chunks are write-once, so a
   running first pass is safe; repeat until the delta is small):
   ```bash
   # on sparkN:
   rsync -a --info=progress2 root@<old-qsfp-ip>:<disk-path>/rustfs/ <new-disk>/rustfs/
   ```
3. **Downtime starts** — stop consumers, then all mounts:
   ```bash
   # SGLang/vLLM down first if the HF cache is on JuiceFS, then on EVERY mount node:
   systemctl stop juicefs-dgxfs
   ```
4. **On the old storage node** — dump metadata, freeze objects, final delta:
   ```bash
   set -a; . /etc/juicefs/dgxfs.env; set +a
   juicefs dump --keep-secret-key redis://<old-qsfp-ip>:6379/1 /root/dgxfs-meta-move.json
   systemctl stop rustfs
   # final delta from sparkN (--delete: mirror exactly, drop since-deleted objects):
   #   rsync -a --delete root@<old-qsfp-ip>:<disk-path>/rustfs/ <new-disk>/rustfs/
   scp /root/dgxfs-meta-move.json root@sparkN:/root/
   systemctl disable --now rustfs valkey-server
   ```
   (`--keep-secret-key` keeps the S3 secret inside the dump so the FS is
   mountable right after load — delete the dump file once the move is verified.)
5. **Repo:** set `juicefs_storage_node: sparkN` (and `juicefs_disk_path` if it
   changed) in `group_vars/all/vault/juicefs.yml`.
6. **Backend bring-up on sparkN** (binaries + Valkey + RustFS + backup env —
   deliberately WITHOUT the format/mount tags, see the warning above):
   ```bash
   ansible-playbook storage.yml --tags juicefs_bin,juicefs_backend,juicefs_backup
   ```
   Then fix object ownership — the `rustfs` system user gets a **different UID**
   on each node, and rsync preserved the old one:
   ```bash
   # on sparkN:
   systemctl stop rustfs && chown -R rustfs:rustfs <new-disk>/rustfs && systemctl start rustfs
   ```
7. **Restore metadata + rewrite the bucket endpoint** (on sparkN; Valkey is
   empty after a fresh install — if a botched earlier attempt formatted it,
   `valkey-cli -a <pw> -n 1 flushdb` first):
   ```bash
   set -a; . /etc/juicefs/dgxfs.env; set +a
   juicefs load   redis://<sparkN-qsfp-ip>:6379/1 /root/dgxfs-meta-move.json
   juicefs config redis://<sparkN-qsfp-ip>:6379/1 --bucket http://<sparkN-qsfp-ip>:9000/juicefs
   juicefs status redis://<sparkN-qsfp-ip>:6379/1   # expect: name dgxfs, new bucket URL
   ```
8. **Full playbook run** — format is now correctly skipped (`status` succeeds),
   every client's mount unit re-renders to the new address and restarts, and the
   meta-backup cron migrates to sparkN (the role removes it elsewhere):
   ```bash
   ansible-playbook storage.yml
   ```
9. **Verify** (§9 checklist: `mountpoint`, cross-node read of a known model
   file, sessions in `juicefs status`). Then clean up: delete
   `/root/dgxfs-meta-move.json` on both nodes; keep the OLD disk + the old
   node's `/var/lib/valkey` untouched for a few days — together they are a full,
   consistent rollback (flip the vault var back + re-enable the old services).

### 13.2 Variant B — "switch JuiceFS off", re-plug the same disk

No second disk needed, but downtime lasts the whole move. "Switching JuiceFS
off" = stopping the client mounts everywhere, then the backend services —
in that order:

1. **Downtime starts** — SGLang/vLLM down first (if the HF cache is on
   JuiceFS), then on EVERY mount node:
   ```bash
   systemctl stop juicefs-dgxfs
   ```
2. **On the old storage node** — dump the metadata **onto the USB disk itself**
   (so it travels with the disk), then shut the backend down and release the disk:
   ```bash
   set -a; . /etc/juicefs/dgxfs.env; set +a
   juicefs dump --keep-secret-key redis://<old-qsfp-ip>:6379/1 <disk-path>/dgxfs-meta-move.json
   systemctl disable --now rustfs valkey-server
   umount <disk-path>
   sed -i '\|<disk-path>|d' /etc/fstab && systemctl daemon-reload
   ```
3. **Re-plug** the disk into sparkN and mount it (the filesystem UUID travels
   with the disk, so the fstab line is identical):
   ```bash
   # on sparkN:
   echo "UUID=<disk-uuid> <disk-path> ext4 defaults,nofail 0 2" >> /etc/fstab
   systemctl daemon-reload && mount <disk-path> && ls <disk-path>/rustfs
   ```
4. **Repo:** set `juicefs_storage_node: sparkN` in `group_vars/all/vault/juicefs.yml`
   (`juicefs_disk_path` stays unchanged).
5. **Backend bring-up on sparkN** — same as Variant A step 6, including the
   `chown -R rustfs:rustfs <disk-path>/rustfs` (different UID on the new node):
   ```bash
   ansible-playbook storage.yml --tags juicefs_bin,juicefs_backend,juicefs_backup
   systemctl stop rustfs && chown -R rustfs:rustfs <disk-path>/rustfs && systemctl start rustfs
   ```
6. **Restore metadata + rewrite the bucket endpoint** — same as Variant A
   step 7, with the dump read from `<disk-path>/dgxfs-meta-move.json`.
7. **Full playbook run** (`ansible-playbook storage.yml`) — mounts come back on
   all sparks, pointing at sparkN; backup cron migrates.
8. **Verify** (§9 checklist), then delete `<disk-path>/dgxfs-meta-move.json`.
   The old node's `/var/lib/valkey` remains an emergency metadata copy (as of
   the dump moment) — wipe it after a few healthy days, along with its leftover
   `/etc/rustfs` and `/etc/juicefs`.

**Rollback of a failed move (both variants):** the old node still has the full
Valkey data dir; Variant A additionally has the untouched old disk. Flip
`juicefs_storage_node` back in the vault, re-plug/re-mount the disk on the old
node (Variant B), `systemctl enable --now valkey-server rustfs` there, and run
`ansible-playbook storage.yml` — the clients re-render back. Nothing was
destroyed until you wiped the old copies in the last step.

---

## 14. Open decisions before starting

- **Storage node placement:** `juicefs_storage_node` (real value in vault)
  holds the USB disk and runs Valkey+RustFS — single point of failure for the
  metadata engine (mitigated by the meta-backup timer, not eliminated). §13
  documents how to relocate it.
- **JuiceFS scope:** HF-cache-only (Phase 4a recommended path) vs. general
  shared FS (the README `HF_HOME` alternative). Decide before seeding, since it
  fixes the on-disk layout (`/mnt/jfs/hub` vs `/mnt/jfs/hf/hub`).
- **JIT caches later:** revisit moving `flashinfer_cache`/`triton_cache` to
  JuiceFS only under renewed space pressure — currently a poor effort/risk ratio.
