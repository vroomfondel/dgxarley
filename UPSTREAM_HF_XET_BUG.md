# Upstream Bug: hf_xet Session-Download-Group — "Unable to parse string as hex hash value"

## Status

**Open upstream — workaround in place** (first diagnosed 2026-07-13). `hf_preload`
(and any container doing a fresh HuggingFace download) aborts on Xet-backed repos
with `RuntimeError: Task error: Unable to parse string as hex hash value`, thrown
from `hf_xet`'s **session-based** file-download-group API. The bug is present in
every `hf_xet >= 1.5.0` we can install, i.e. **latest stable `1.5.1`** and the
**newest pre-release `1.5.2rc0`** — so a plain image rebuild does NOT fix it (there
is nothing newer to pull). Downgrading `hf_xet` below 1.5.0 is blocked by
`huggingface_hub 1.23.0` (it requires `hf_xet >= 1.5.0`).

**Workaround (deployed):** set env `HF_HUB_DISABLE_XET=1` on every download
container → `snapshot_download` skips Xet and uses classic HTTPS/LFS. Wired as the
repo variable `hf_hub_disable_xet` (dummy `0` in `group_vars/all/main/huggingface.yml`,
real `1` in the vault copy), propagated to sglang head/worker, vllm, the shard Jobs,
`hf_preload`, `sglang_tune_moe`, comfyui and docling. To flip cluster-wide, change
only the vault value and redeploy.

## Symptom

```
Fetching 26 files:  27%|██▋  | 7/26 [00:00<00:00, 19.98it/s]
FAILED: nvidia/Llama-4-Scout-17B-16E-Instruct-NVFP4: Task error: Unable to parse string as hex hash value
```

The `hf_preload_download.py` catch-all turns this into `Failed models: [...]` →
`sys.exit(1)` → the whole K8s Job is marked `Failed`. Only a model doing a **fresh**
download trips it; already-cached models never exercise Xet (they "fetch" in 0:00),
which makes "only model X fails" misleading.

Full traceback (huggingface_hub 1.23.0):

```
_snapshot_download.py:513   thread_map( _inner_hf_hub_download ... )
_snapshot_download.py:493   _inner_hf_hub_download -> hf_hub_download(...)
file_download.py:1013       hf_hub_download -> _hf_hub_download_to_cache_dir
file_download.py:1236       -> _download_to_tmp_and_move
file_download.py:1920       -> xet_get(
file_download.py:563        xet_get -> session.new_file_download_group(
RuntimeError: Task error: Unable to parse string as hex hash value
```

## Affected versions

| Component | Version | Source |
|---|---|---|
| `hf_xet` | `1.5.1` | image `xomoxcc/dgx-spark-sglang:0.5.14-sm121` — **broken** |
| `hf_xet` | `1.5.2rc0` | image `xomoxcc/dgx-spark-sglang:0.5.15-sm121` — **broken** |
| `hf_xet` | `1.5.2` (stable, released 2026-07-16) | **confirmed still broken** — reported on `xet-core#895` 2026-07-23 (not yet tested in-cluster) |
| `huggingface_hub` | `1.23.0` | both images |
| `huggingface_hub` | `1.24.0` (released 2026-07-17) | **confirmed still broken** alongside `hf_xet 1.5.2` — same `xet-core#895` report, 2026-07-23 |

Neither `hf_xet` nor `huggingface_hub` is pinned in the SGLang build recipes
(`scripts/patches/sglang-0.5.1{4,5}-sm121.recipe`), so a rebuild pulls whatever pip
resolves at build time — currently the broken versions.

xet-core releases (checked 2026-07-13): `v1.5.2-rc0` (2026-07-09, pre-release, newest),
`v1.5.1` (2026-06-08, latest stable), `v1.5.0` (2026-05-06, **"Session based API"** —
where the failing `session.new_file_download_group` was introduced), `v1.4.3` (2026-03-31).

## Root cause (measured, not inferred)

The distinguishing variable is the **huggingface_hub download code path**, not the
data, the version, the cache, concurrency, disk, or the CAS service:

| Call | Result |
|---|---|
| `hf_hub_download(repo, filename=<shard>)` | ✅ reliably OK (shard1: 2/2, ~46s each) |
| `snapshot_download(repo, allow_patterns=[<shard>])` (what the Job uses) | ❌ always `hex hash` crash |

Both download the **same file** over the **same Xet CAS**, same clean cache, same
`hf_xet`. `snapshot_download` drives `hf_xet`'s new **session-based** download-group
API (added in hf_xet 1.5.0); `hf_hub_download` takes the per-file path. The crash is
in that session path.

The unparseable "hex hash" is **not** the file-level Xet hash: the `x-xet-hash`
resolve headers for all Scout shards are valid 64-hex (verified via
`curl -sI .../resolve/main/<shard>`), e.g. shard1
`b3bf96b94eefa65ef3c4bc393c3843e4fd41a8c29648d87b35798642025acf47`. The bad hash is
inside the CAS **reconstruction** response
(`cas-server.xethub.hf.co/v1/reconstructions/<hash>`) as consumed by the session
download group.

### Ruled out (with the test that ruled it out)

- **Auth / token** — no (partial files download fine with the `dgx_read_ALL` token).
- **Image bump 0.5.14 → 0.5.15** — no (0.5.15 = hf_xet 1.5.2rc0 fails identically).
- **Concurrency** — no (`snapshot_download(max_workers=1)` still fails).
- **Corrupt local Xet cache** — no (`rm -rf /mnt/jfs/xet`; recreated; still fails).
- **Disk full** — no (red herring; `/mnt/jfs` had 228 G free, shard is 5 G).
- **Server-side CAS flakiness** — no (`hf_hub_download` of a shard succeeds reliably, 2/2).
- **File-specific bad hash** — no (shard1 works via `hf_hub_download`, fails via
  `snapshot_download`; the file data is fine).

## Reproduction (faithful)

Run on spark2 (`192.168.191.202`, hostname-verify first), against the exact image,
with the shared JuiceFS HF cache mounted the way the Job mounts it:

```bash
podman run --rm --network host \
  -e HF_TOKEN="$HF_TOKEN" \
  -v /mnt/jfs:/root/.cache/huggingface \
  xomoxcc/dgx-spark-sglang:0.5.14-sm121 \
  python3 -c 'from huggingface_hub import snapshot_download; \
    snapshot_download(repo_id="nvidia/Llama-4-Scout-17B-16E-Instruct-NVFP4", \
      cache_dir="/root/.cache/huggingface/hub", \
      allow_patterns=["model-00002-of-00014.safetensors"], force_download=True)'
# -> RuntimeError: Task error: Unable to parse string as hex hash value
```

Add `-e HF_HUB_DISABLE_XET=1` and it downloads cleanly (classic HTTPS/LFS).
The truly faithful Job repro runs `roles/k8s_dgx/files/{hf_preload_run.sh,hf_preload_download.py}`
via `/bin/bash /scripts/run.sh` with the Job's env; single-file python one-liners
are apples-to-oranges (they take the `hf_hub_download` path, which works).

## Why a rebuild does not help

- Latest stable `hf_xet 1.5.1` is already the broken version; newest pre-release
  `1.5.2rc0` is also broken. There is no fixed release to pull.
- Pinning `hf_xet < 1.5.0` is blocked: with `huggingface_hub 1.23.0`, installing
  `hf_xet==1.4.3` yields `ValueError: To use optimized download using Xet storage,
  you need to install the hf_xet package ...` (hub 1.23.0 requires the ≥1.5.0
  session API). Restoring Xet would need a coordinated downgrade of **both**
  `huggingface_hub` and `hf_xet` — not worth it for a one-time cache warmup where
  Xet only buys download speed.

## Upstream tracking

- **Exact-string issue is now filed:** **[`huggingface/xet-core#895`](https://github.com/huggingface/xet-core/issues/895)**
  ("Download fails with 'Task error: Unable to parse string as hex hash
  value' (hf-xet 1.5.1)"), filed **2026-07-11** — it existed before this
  doc's original 2026-07-13 write-up, our search just missed it. **Still
  OPEN as of 2026-07-23.** Collaborator @seanses acknowledged on
  2026-07-16 that "huggingface_hub is passing incorrect file hash to
  hf-xet leading to this error", asking for the affected repo id.
  Maintainer @Wauplin followed up on **2026-07-23** (today) asking that
  the failing hash value be surfaced in the error message itself
  (`Task error: ... (got '<hash>')`) so the report can be narrowed down —
  no root cause identified, no fix merged. **TODO superseded:** no longer
  need to file our own issue; instead consider adding our reproduction
  details (faithful `snapshot_download` vs `hf_hub_download` split, see
  above) as a comment on #895 if it stays unresolved.
- Related closed reports (symptom cluster, same 1.5.x era, resolved
  independently of this bug):
  - huggingface/xet-core #358 — "errors became very common" with snapshot_download (closed)
  - huggingface/xet-core #399 — "Cannot Download XET Files" (closed)
  - huggingface/xet-core #483 — "Still can't download models" (closed)
  - huggingface/huggingface_hub #3960 — "Downloading not working with hf_xet" (still open, unconfirmed relation)
  - huggingface/huggingface_hub #3643 — snapshot_download blob checksum mismatch (XET) (closed)
- Watch: <https://github.com/huggingface/xet-core/issues/895> directly
  (now the actionable tracking issue), plus
  <https://github.com/huggingface/xet-core/releases> and the
  `huggingface_hub` changelog. `hf_xet 1.5.2` (2026-07-16) and
  `huggingface_hub 1.24.0` (2026-07-17) have both shipped since the
  original diagnosis — neither fixes this (see Affected versions table
  and Changelog below).

## How to know when to drop the workaround

When a `hf_xet > 1.5.2rc0` (or a huggingface_hub release noting a session
download-group fix) is out:

1. Rebuild an image OR test in a throwaway container with the new `hf_xet`.
2. Run the faithful reproduction above **without** `HF_HUB_DISABLE_XET`.
3. If shard 2 downloads cleanly → set `hf_hub_disable_xet: 0` in
   `group_vars/all/vault/huggingface.yml` and redeploy; delete this file or move it
   to `FIXED_UPSTREAM_HF_XET_BUG.md` with the fixing version recorded.

## Changelog

- **2026-07-13** — First diagnosis. Isolated to the `snapshot_download` →
  `session.new_file_download_group` path (hf_xet ≥ 1.5.0). Confirmed rebuild won't
  fix (1.5.1 / 1.5.2rc0 both broken; 1.4.3 blocked by hub 1.23.0). Workaround
  `HF_HUB_DISABLE_XET=1` wired as `hf_hub_disable_xet` across all download containers.
- **2026-07-23** — The bug is tracked upstream after all:
  `huggingface/xet-core#895` was filed 2026-07-11 (before this doc's first
  write-up) and remains **OPEN** — maintainer @Wauplin engaged today
  (2026-07-23) requesting more diagnostic detail, no fix yet. A commenter
  on the issue confirmed the identical failure on **hf_xet 1.5.2**
  (2026-07-16 stable) and **huggingface_hub 1.24.0** (2026-07-17) on
  2026-07-23, so the newer stable releases do not fix it either.
  `HF_HUB_DISABLE_XET=1` workaround unchanged and still required.
