# k3shelperstuff

Standalone helper scripts for operating the K3s cluster:

- [`update_local_k3s_keys.py`](#update_local_k3s_keyspy) — keep the local kubeconfig in sync with a remote K3s server
- [`keel_drift.py`](#keel_driftpy) — find Keel-tracked workloads whose running image lags behind its tag

## update_local_k3s_keys.py

K3s kubeconfig credential synchronization utility. Keeps local Kubernetes authentication credentials (`~/.kube/config`) in sync with a remote K3s server.

### What it does

Fetches the kubeconfig from a remote K3s server via SSH, compares user credentials and cluster CA data against the local kubeconfig, and interactively updates any differences.

- Extracts client certificates, client keys, and cluster CA data from both remote and local kubeconfig
- Shows truncated diffs without exposing full secrets
- Prompts before writing any changes
- Auto-detects remote host and context from the current-context in `~/.kube/config`

### Usage

```bash
k3s-keys-sync [OPTIONS]                                   # installed entry point
python -m dgxarley.k3shelperstuff.update_local_k3s_keys   # or as a module
```

| Option                    | Description                                                   |
|---------------------------|---------------------------------------------------------------|
| `-u`, `--user USER`       | SSH user (default: `root`)                                    |
| `-H`, `--host HOST`       | Remote host (auto-detected from kubeconfig server URL)        |
| `-c`, `--context CONTEXT` | Local kubeconfig context (auto-detected from current-context) |

The remote kubeconfig is read from `/etc/rancher/k3s/k3s.yaml` on the target host.

## keel_drift.py

Finds Keel-tracked workloads whose running image is older than the image its tag currently points at.

### Why it exists

On every poll, Keel compares the registry digest of right now against the digest it memorised during the previous poll. That memo lives in memory only and is seeded from the registry at startup. What actually runs in the cluster therefore never enters Keel's decision: if a tag is moved while Keel restarts, Keel sets its baseline to the new digest without ever touching the Deployment, and the change stays invisible until the next push.

This script performs exactly the comparison Keel does not: the digest of the running pod against the digest the tag currently points at.

### What it does

- Collects every Deployment, StatefulSet and DaemonSet carrying an active `keel.sh/policy` (annotations beat labels, `never` and empty count as inactive, the same order Keel itself uses)
- Reads the running digest per container from the `imageID` of the running pods
- Resolves the tag against the registry, accepting both the index digest and any per-platform manifest digest of a multi-arch tag
- Authenticates with the workload's `imagePullSecrets`, falling back to the local Docker login (`DOCKER_CONFIG` or `~/.docker/config.json`) so Docker Hub does not count against the anonymous 100/h per-IP limit
- Flags containers with `imagePullPolicy != Always`, since a restart cannot renew an unchanged tag there

### Usage

```bash
keel-drift [OPTIONS]                          # installed entry point
python -m dgxarley.k3shelperstuff.keel_drift  # or as a module
```

Needs the optional dependencies: `pip install 'dgxarley[k3s]'`.

| Option                   | Description                                                   |
|--------------------------|---------------------------------------------------------------|
| `-n`, `--namespace NS`   | Check only this namespace (default: all)                      |
| `--drift-only`           | Show only stale and unclear workloads                         |
| `--fix-command`          | Print the `kubectl rollout restart` commands to straighten out |
| `-q`, `--quiet`          | Suppress the table, print only the summary                    |
| `-v`, `--verbose`        | Log every namespace, workload and registry access             |
| `--no-local-credentials` | Ignore the local Docker login, query everything anonymously   |

The table goes to stdout, progress and diagnostics to stderr, so the output stays pipe-friendly.

### Exit codes

| Code | Meaning                                                      |
|------|--------------------------------------------------------------|
| `0`  | No workload is stale (or none is tracked by Keel at all)      |
| `1`  | At least one workload is stale, usable as a pipeline gate     |
| `2`  | Neither a kubeconfig nor an in-cluster context could be used  |

### Examples

```bash
keel-drift                          # every tracked workload
keel-drift --namespace somestuff    # a single namespace
keel-drift --drift-only --quiet     # drift only, terse
keel-drift --fix-command            # print rollout-restart commands
```
