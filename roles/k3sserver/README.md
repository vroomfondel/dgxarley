# K3s Server Role

Ansible role for deploying and configuring a K3s Kubernetes cluster with master/agent node support. Handles K3s installation, NFS shared storage, HAProxy load balancing, rsyslog for pod log collection, Helm chart extensions for bundled components, and scheduled maintenance tasks.

## Requirements

- Debian/Ubuntu target hosts
- Network access to `https://get.k3s.io`
- NFS server available (if `nfsmasterip` is defined)

## Role Variables

### Required Variables

These are typically set in `group_vars/k3sserver.yml` or `host_vars/`:

| Variable | Description |
|---|---|
| `k3smasterip` | IP address of the K3s master node |
| `nfsmasterip` | IP address of the NFS server for shared storage |
| `cluster_cidr` | Pod network CIDR (e.g., `10.42.0.0/16`) |
| `service_cidr` | Service network CIDR (e.g., `10.43.0.0/16`) |

### Host Variables

| Variable | Description |
|---|---|
| `isk3smaster: true` | Designates the K3s master (server) node |
| `isnfsmaster: true` | Host provides NFS storage |

## Task Files

### `main.yml`

Orchestrates all subtasks in order:

1. **Install packages** (`installbasicpackages.yml`)
2. **Mount NFS** -- creates `/mnt/nfs` and mounts `{{ nfsmasterip }}:/k3shared` (skipped if `nfsmasterip` is undefined)
3. **Install K3s** (`k3sinstall.yml`)
4. **Cron jobs** (`crontab.yml`)
5. **HAProxy** (`haproxy.yml`) -- master only
6. **Rsyslog** (`rsyslog.yml`)

### `installbasicpackages.yml`

Installs system packages on all cluster nodes.

**All nodes:** `nfs-common`, `curl`, `runc`

**Master only:** `python3-kubernetes`, `knxd`, `knxd-tools`, `golang-go`, `wakeonlan`, `haproxy`, `rsyslog`

### `k3sinstall.yml`

Handles the full K3s cluster bootstrap:

1. **Hosts file** -- adds `k3smaster` entry pointing to `{{ k3smasterip }}` on agents, or `127.0.0.1` on the master
2. **K3s config** -- deploys `/etc/rancher/k3s/config.yaml` from template (different content for master vs agent)
3. **Master installation** -- runs the K3s server installer script (version `v1.35.1+k3s1`), then copies Helm extension manifests to `/var/lib/rancher/k3s/server/manifests/`
4. **Token distribution** -- reads the node token from the master and distributes it to all agent nodes
5. **Agent installation** -- runs the K3s agent installer on non-master nodes using the distributed token

Installation is idempotent: scripts only run if the k3s service does not already exist.

### `haproxy.yml`

Configures HAProxy on the master node as a TCP proxy in front of Traefik. All frontends use PROXY protocol v2 to preserve client IPs.

| Frontend | Port | Backend Port | Purpose |
|---|---|---|---|
| HTTP | 80 | 8181 | Web traffic to Traefik |
| HTTPS | 443 | 4430 | TLS traffic to Traefik |
| MQTT | 1883 | 1882 | Mosquitto unencrypted |
| MQTT TLS | 8883 | 8882 | Mosquitto TLS |
| Stats | 8123 | -- | Admin UI (localhost only) |

The stats interface is accessible via SSH tunnel:
```bash
ssh -L 8123:127.0.0.1:8123 root@acerrevo.local
```

### `rsyslog.yml`

Configures rsyslog on all nodes to receive remote pod logs over UDP/TCP port 514 and forward them to systemd journal via `omjournal`. This allows pod logs shipped by Fluent Bit to appear in `journalctl`.

### `crontab.yml`

Deploys maintenance scripts and cron jobs on the master node (excluding `carneades` cluster).

| Schedule | Script / Command | Purpose |
|---|---|---|
| Daily 17:00 | `crictl rmi --prune` | Prune unused container images |
| Daily 22:00 | `dumpclusterstate.sh` | Backup cluster state and SQLite DB to RustFS (S3) |
| Daily 11:30 | `updatemosquittos.sh` | Sync Mosquitto TLS certs to relay instances |
| Daily 13:30 | `updatepiholetls.sh` | Sync Pi-hole TLS certs to relay instances |
| Sat 16:30 | `wakeonlan` | Wake `revo` machine |

## K3s Configuration

The master node `config.yaml` sets:

- **Disabled components:** `servicelb` (HAProxy is used instead)
- **Network policy:** disabled
- **Node port range:** `22-32767`
- **Flannel:** IPv6 masquerade disabled
- **Kubelet:** allows unsafe sysctls (`net.ipv4.*`, `net.ipv6.*`)
- **kube-proxy:** `nodeport-addresses=127.0.0.0/8` (NodePorts only on localhost)
- **Bind/node IP:** set to the host's default IPv4 address

Agent nodes set `node-ip`, `node-external-ip`, and the same `kube-proxy-arg: nodeport-addresses=127.0.0.0/8`.

## Helm Extension Files

These YAML files are placed in `/var/lib/rancher/k3s/server/manifests/` where K3s auto-deploys them as HelmChartConfig resources.

| File | Component | Key Settings |
|---|---|---|
| `traefik-extend.yaml` | Traefik ingress | 2 replicas with pod anti-affinity pinned to peekaboos, PROXY protocol on web/websecure/MQTT ports, fixed NodePorts, `/ping` health endpoint on `traefik` entryPoint (NodePort 30808), JSON access logs, dashboard enabled |
| `metrics-server-extend.yaml` | Metrics Server | Resource limits (50m-1000m CPU, 64-256Mi memory) |
| `local-path-provisioner-extend.yaml` | Local Path Provisioner | Resource limits (50m-200m CPU, 64-256Mi memory) |
| `coredns-extend.yaml` | CoreDNS | Resource limits (100m-300m CPU, 70-170Mi memory) |

### Traefik Port Mapping

Traefik listens on non-standard ports because HAProxy handles the standard ports. Each port has a fixed NodePort for use by peekaboo HAProxy instances (see edgeservices role). A `/ping` health endpoint is enabled on the default `traefik` entryPoint (8080, NodePort 30808) for keepalived tracking — this port has no PROXY protocol, so plain `curl` health checks work directly.

| Standard Port | Traefik Port | NodePort | Protocol |
|---|---|---|---|
| -- (ping/API) | 8080 | 30808 | HTTP (no PROXY protocol) |
| 80 (via HAProxy) | 8181 | 30080 | HTTP + PROXY protocol |
| 443 (via HAProxy) | 4430 | 30443 | HTTPS + PROXY protocol |
| 1883 (via HAProxy) | 1882 | 31882 | MQTT + PROXY protocol |
| 8883 (via HAProxy) | 8882 | 38882 | MQTT TLS + PROXY protocol |

NodePorts are restricted to localhost via `kube-proxy-arg: nodeport-addresses=127.0.0.0/8` (both master and agent nodes), so they are not reachable from LAN IPs. LoadBalancer and ClusterIP access is unaffected.

## Maintenance Scripts

### `dumpclusterstate.sh`

Backs up the full cluster state nightly:
1. Runs `kubectl cluster-info dump --all-namespaces`
2. Exports all secrets
3. Creates tarballs (cluster state + K3s SQLite DB)
4. Uploads to RustFS (S3) at `rasnas/acerrevo/<date>/`

### `updatemosquittos.sh`

Syncs Mosquitto TLS certificates and config from the K8s Mosquitto TLS secret to external relay instances. Compares MD5 hashes and only restarts the remote service when files change.

### `updatepiholetls.sh`

Syncs Pi-hole TLS certificates from the K8s Pi-hole TLS secret to external relay instances. Same hash-based change detection as the Mosquitto script.

## Tags

| Tag | Scope |
|---|---|
| `installbasicpackages` | Package installation only |
| `k3sinstall` | K3s server/agent installation |
| `crontab` | Cron job setup |
| `haproxy` | HAProxy configuration (master only) |
| `rsyslog` | Rsyslog configuration |

## Handlers

| Handler | Action |
|---|---|
| `restart knxd` | Restarts KNX daemon |
| `stop knxd and disable` | Stops and disables KNX daemon |
| `restart rsyslog` | Restarts rsyslog service |
| `restart haproxy` | Restarts HAProxy service |
| `sysctl reload` | Reloads sysctl settings (`sysctl -p`) |

## Playbook Usage

The role is applied to the `k3sserver` host group:

```yaml
# k3sserver.yml
- hosts: k3sserver
  roles:
    - k3sserver
```

```bash
# Full cluster setup
ansible-playbook k3sserver.yml

# Install/update K3s only
ansible-playbook k3sserver.yml --tags k3sinstall

# Reconfigure HAProxy
ansible-playbook k3sserver.yml --tags haproxy

# Target specific node
ansible-playbook k3sserver.yml --limit acerrevo.local
```

Also included in the site-wide playbook:
```bash
ansible-playbook site.yml
```

## Directory Structure

```
roles/k3sserver/
├── defaults/
│   └── main.yml
├── handlers/
│   └── main.yml                           # Service restart handlers
├── meta/
│   └── main.yml
├── tasks/
│   ├── main.yml                           # Orchestration
│   ├── installbasicpackages.yml           # System packages
│   ├── k3sinstall.yml                     # K3s master/agent setup
│   ├── haproxy.yml                        # HAProxy load balancer
│   ├── rsyslog.yml                        # Remote log collection
│   └── crontab.yml                        # Maintenance cron jobs
├── templates/
│   ├── etc_rancher_k3s_config.yaml.j2     # K3s node configuration
│   ├── k3s_install_server.sh.j2           # Master install script
│   ├── k3s_install_agent.sh.j2            # Agent install script
│   └── etc_haproxy_haproxy.cfg.j2         # HAProxy configuration
├── files/
│   ├── traefik-extend.yaml                # Traefik Helm overrides
│   ├── metrics-server-extend.yaml         # Metrics Server overrides
│   ├── local-path-provisioner-extend.yaml # Local Path Provisioner overrides
│   ├── coredns-extend.yaml                # CoreDNS overrides
│   ├── dumpclusterstate.sh                # Cluster backup script
│   ├── updatepihole.sh                    # Pi-hole command wrapper
│   ├── updatemosquittos.sh                # Mosquitto TLS sync
│   ├── updatepiholetls.sh                 # Pi-hole TLS sync
│   ├── etc_rsyslog.d_10-remote-to-journal.conf
│   └── etc_rsyslog.d_49-haproxy.conf
└── vars/
    └── main.yml
```
