# k3shelperstuff

K3s kubeconfig credential synchronization utility. Keeps local Kubernetes authentication credentials (`~/.kube/config`) in sync with a remote K3s server.

## What it does

`update_local_k3s_keys.py` fetches the kubeconfig from a remote K3s server via SSH, compares user credentials and cluster CA data against the local kubeconfig, and interactively updates any differences.

- Extracts client certificates, client keys, and cluster CA data from both remote and local kubeconfig
- Shows truncated diffs without exposing full secrets
- Prompts before writing any changes
- Auto-detects remote host and context from the current-context in `~/.kube/config`

## Usage

```bash
python update_local_k3s_keys.py [OPTIONS]
```

| Option | Description |
|---|---|
| `-u`, `--user USER` | SSH user (default: `root`) |
| `-H`, `--host HOST` | Remote host (auto-detected from kubeconfig server URL) |
| `-c`, `--context CONTEXT` | Local kubeconfig context (auto-detected from current-context) |

The remote kubeconfig is read from `/etc/rancher/k3s/k3s.yaml` on the target host.