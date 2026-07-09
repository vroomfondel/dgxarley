# juicefs_storage

Distributed JuiceFS storage layer for the DGX cluster, deployed **entirely as
host-level systemd services, outside k3s**. Pools large LLM weights across nodes,
caches hot data per-node on the NVMe, and persists chunks to a single USB-SSD via
an S3 object store.

## Why host-level (not a DaemonSet / CSI)

A FUSE mount inside a pod dies with the pod (image update, OOM, k3s upgrade,
eviction) → stale mount, `Transport endpoint is not connected` for every consumer,
plus the `mountPropagation: Bidirectional` risk. A native systemd mount is
isolated from the k3s lifecycle, restarts in seconds, and consumers only need the
safe `HostToContainer` propagation. JuiceFS has **no master**: every mount talks
directly to Valkey + RustFS, so nothing is routed through a coordinator.

## Components

| Component       | Where                 | What                                                                   |
|-----------------|-----------------------|------------------------------------------------------------------------|
| **Valkey**      | storage node          | metadata engine (apt `valkey-server`, AOF + `noeviction` + auth)       |
| **RustFS**      | storage node          | S3 object store (native binary, data dir on the USB-SSD)               |
| **format**      | storage node          | one-time idempotent `juicefs format` (guarded by `juicefs status`)     |
| **mount**       | `juicefs_mount_group` | native systemd FUSE mount, `Restart=always`, per-node cache            |
| **meta-backup** | backup node           | root cron job: `juicefs dump` → external backup S3 (`rustfs_rc_alias`) |

## Single-node vs. distributed object store

`juicefs_rustfs_distributed` (default **false**) selects the RustFS topology:

- **OFF (single-node):** RustFS runs only on `juicefs_storage_node`, one local
  volume, no erasure coding. Original behaviour — unchanged.
- **ON (distributed):** RustFS runs on **every** node in `juicefs_storage_nodes`,
  forming one erasure-coded S3 cluster. Object data is striped + made redundant
  across all their USB-SSDs, so no single USB bus carries every write.

`juicefs_storage_nodes` is a **structured, ordered** list — one entry per node,
each listing its USB disk(s) as dicts; heterogeneous paths and disk counts allowed:

```yaml
juicefs_storage_nodes:
  - node: spark1
    disks: [{ path: /mnt/intenso, uuid: "<blkid-UUID>", fstype: ext4 }]
  - node: spark2                                        # two disks on this node
    disks:
      - { path: /mnt/usb-a, uuid: "<uuid>", fstype: ext4 }
      - { path: /mnt/usb-b, uuid: "<uuid>", fstype: ext4 }
  - node: spark3
    disks: [{ path: /mnt/intenso, uuid: "<uuid>", fstype: ext4 }]
  - node: spark4
    disks: [{ path: /mnt/intenso, uuid: "<uuid>", fstype: ext4 }]
```

Per disk, `path` is always required (RustFS data dir = `<path>/rustfs`);
`uuid`/`fstype`/`options` are consulted only when `juicefs_manage_fstab` is on.
Order matters (positional erasure membership; `RUSTFS_VOLUMES` is rendered
identically on every node). The **primary** (`juicefs_storage_node`) must be a
member — it keeps running Valkey, does the one-time format, and is the single S3
endpoint the FUSE clients dial. Redundancy level via `juicefs_rustfs_raid_level`:

| `raid_level` | `EC:M` | usable (n drives) | tolerates |
|--------------|--------|-------------------|-----------|
| `raid0`      | `EC:0` | 100 %             | 0 (striping only) |
| `raid5`      | `EC:1` | (n−1)/n           | 1 node/disk |
| `raid6`      | `EC:2` | (n−2)/n           | 2 nodes/disks |

**Constraints (verified upstream):** ≥4 total drives; **Valkey stays single-node**
(distributing RustFS does NOT remove the metadata SPOF — the meta-backup cron is
the insurance); RAID6 needs ≥4 drives.

**Mounting the disks — `juicefs_manage_fstab`** (default **false**):

- **OFF:** you own `/etc/fstab` (mount each USB by hand, e.g.
  `UUID=… /mnt/intenso ext4 defaults,nofail 0 2`); the role only verifies the
  mountpoints and fails if one is missing. Original behaviour.
- **ON:** the role writes a **UUID-keyed** fstab entry (via `ansible.posix.mount`,
  `state: mounted`, opts default `defaults,nofail`) and mounts each disk that
  carries a `uuid` before the mountpoint check. UUID is mandatory — USB `/dev/sdX`
  names reorder across boots and a wrong name would drop a disk into the wrong
  erasure slot. Get the UUID with `blkid /dev/sdX1`. Single-node mode has no
  structured disk (only `juicefs_disk_path`), so it stays manual either way.

**⚠ Migration — enabling distributed is NOT in-place.** Single-node (SNSD) has no
erasure coding and its on-disk layout is incompatible with distributed mode.
Flipping `juicefs_rustfs_distributed` to true on a populated store means a **fresh
`juicefs format`** — the existing object data is abandoned. Move/re-download data
off the filesystem first, then wipe the old single-node data dir before deploying
the cluster.

## Key variables (see `defaults/main.yml`)

- `juicefs_storage_node` — **the placement knob**: which node has the USB-SSD and
  runs Valkey + RustFS (master or any spark).
- `juicefs_storage_address` — address mounts use; defaults to the storage node's
  QSFP IP (spark → 200GbE) or its k3s VLAN IP (master).
- `juicefs_disk_path` — USB-SSD mountpoint (you mount it; the role won't format it
  and refuses to run if it isn't a real mountpoint).
- `juicefs_cache_dir` / `juicefs_cache_size_mib` — per-node NVMe cache (MiB).
- Secrets (`juicefs_rustfs_access_key`/`_secret_key`, `juicefs_valkey_password`)
  are **dummies here** — put real values in `group_vars/all/vault/juicefs.yml`.

## Deploy

```bash
ansible-playbook storage.yml                      # full
ansible-playbook storage.yml --tags juicefs_mount # mounts only
```

Order is handled internally: binary → backend → format → mount → backup.

## Monitoring (gated by `juicefs_enabled` AND `juicefs_metrics_enabled`)

`juicefs_metrics_enabled` (pre-enabled) selects WHETHER the layer is monitored;
every consumer additionally couples to the `juicefs_enabled` master switch, so
nothing scrapes, alerts, or opens ports until the layer is actually deployed.
Deploy day is one flag flip (`juicefs_enabled: true`), then:
`storage.yml → common.yml --tags iptables → k8s_infra.yml --tags prometheus,grafana`:

- **Clients**: `--metrics <k3s_node_ip>:9567` on every FUSE mount unit (native
  `juicefs_*` Prometheus metrics; flipping the gate restarts the mounts once).
- **Valkey**: apt `prometheus-redis-exporter` on the storage node (`:9121`,
  reuses the requirepass; redis_exporter supports Valkey upstream).
- **RustFS**: has **no native `/metrics`** (OTel push only, rustfs#3154) — its
  health is monitored client-side via `juicefs_object_request_errors` (alert
  `JuiceFSObjectErrors`) and the object-request panels.
- **k8s_infra**: scrape jobs `juicefs`/`juicefs-valkey` + `juicefs-alerts`
  rules (`monitoring/prometheus.yml`), Grafana dashboards "JuiceFS Dashboard"
  (grafana.com 20794) and "Valkey (JuiceFS Metadata)" (763) via
  `download-dashboards.sh`; iptables opens both ports from `k3snodes` only.

## Consumer wiring (separate change, not done here)

Pods consume the shared FS by mounting the host path with one-way propagation:

```yaml
volumes:
  - name: jfs
    hostPath: { path: /mnt/jfs, type: Directory }
containers:
  - volumeMounts:
      - name: jfs
        mountPath: /mnt/jfs
        mountPropagation: HostToContainer
    env:
      - { name: HF_HOME, value: /mnt/jfs/hf }
```

After a model download, warm the per-node cache (runs per node):
`juicefs warmup /mnt/jfs/hf/<model-path>`.

## Acceptance

1. `mountpoint -q /mnt/jfs` true on every mount node.
2. A file written on node A under `/mnt/jfs/test` is readable on node B.
3. New objects appear in the RustFS `juicefs` bucket after writes.
4. `juicefs warmup` fills the local cache; a re-read produces no backend traffic.
5. `juicefs dump` runs and the backup lands in the external backup bucket.
