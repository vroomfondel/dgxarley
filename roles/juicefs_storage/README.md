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

| Component | Where | What |
|-----------|-------|------|
| **Valkey** | storage node | metadata engine (apt `valkey-server`, AOF + `noeviction` + auth) |
| **RustFS** | storage node | S3 object store (native binary, data dir on the USB-SSD) |
| **format** | storage node | one-time idempotent `juicefs format` (guarded by `juicefs status`) |
| **mount** | `juicefs_mount_group` | native systemd FUSE mount, `Restart=always`, per-node cache |
| **meta-backup** | backup node | root cron job: `juicefs dump` → external rasnas S3 |

## Key variables (see `defaults/main.yml`)

- `juicefs_storage_node` — **the placement knob**: which node has the USB-SSD and
  runs Valkey + RustFS (master or any spark).
- `juicefs_storage_address` — address mounts use; defaults to the storage node's
  QSFP IP (spark → 200GbE) or its k3s VLAN IP (master).
- `juicefs_disk_path` — USB-SSD mountpoint (you mount it; the role won't format it
  and refuses to run if it isn't a real mountpoint).
- `juicefs_cache_dir` / `juicefs_cache_size_mib` — per-node NVMe cache (MiB).
- Secrets (`juicefs_rustfs_access_key`/`_secret_key`, `juicefs_valkey_password`)
  are **dummies here** — put real values in `group_vars/all/vault.yml`.

## Deploy

```bash
ansible-playbook storage.yml                      # full
ansible-playbook storage.yml --tags juicefs_mount # mounts only
```

Order is handled internally: binary → backend → format → mount → backup.

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
5. `juicefs dump` runs and the backup lands in the rasnas bucket.
