#!/usr/bin/env python3
"""
Script to compare K3s kubeconfig from remote server with local ~/.kube/config
and optionally update local credentials for a specific context.

Supports --create to initially create the context/cluster/user entries,
and --yes for non-interactive (Ansible) usage.
"""

import argparse
import subprocess
import sys
from pathlib import Path
from urllib.parse import urlparse

import yaml


def get_remote_kubeconfig(host: str, remote_path: str) -> dict:
    """Fetch kubeconfig from remote host via SSH."""
    cmd = ["ssh", host, f"cat {remote_path}"]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        return yaml.safe_load(result.stdout)
    except subprocess.CalledProcessError as e:
        print(f"Error accessing {host} via SSH: {e.stderr}")
        sys.exit(1)


def load_local_kubeconfig(path: Path, allow_missing: bool = False) -> dict:
    """Load local kubeconfig file.

    If allow_missing is True, return an empty kubeconfig skeleton when the
    file does not exist (and create ~/.kube/ if needed).
    """
    if not path.exists():
        if allow_missing:
            path.parent.mkdir(parents=True, exist_ok=True)
            return {}
        print(f"Local kubeconfig not found: {path}")
        sys.exit(1)
    with open(path) as f:
        return yaml.safe_load(f) or {}


def find_context_user(kubeconfig: dict, context_name: str) -> str | None:
    """Find the user name associated with a context."""
    for ctx in kubeconfig.get("contexts", []):
        if ctx.get("name") == context_name:
            return ctx.get("context", {}).get("user")
    return None


def find_context_cluster(kubeconfig: dict, context_name: str) -> str | None:
    """Find the cluster name associated with a context."""
    for ctx in kubeconfig.get("contexts", []):
        if ctx.get("name") == context_name:
            return ctx.get("context", {}).get("cluster")
    return None


def get_user_credentials(kubeconfig: dict, user_name: str) -> dict:
    """Get credentials for a specific user."""
    for user in kubeconfig.get("users", []):
        if user.get("name") == user_name:
            return user.get("user", {})
    return {}


def get_cluster_ca(kubeconfig: dict, cluster_name: str) -> str | None:
    """Get certificate-authority-data for a specific cluster."""
    for cluster in kubeconfig.get("clusters", []):
        if cluster.get("name") == cluster_name:
            return cluster.get("cluster", {}).get("certificate-authority-data")
    return None


def compare_credentials(remote: dict, local: dict, remote_ca: str | None, local_ca: str | None) -> dict:
    """Compare remote and local credentials, return differences."""
    differences = {}

    remote_cert = remote.get("client-certificate-data")
    local_cert = local.get("client-certificate-data")
    if remote_cert != local_cert:
        differences["client-certificate-data"] = {
            "remote": remote_cert[:50] + "..." if remote_cert else None,
            "local": local_cert[:50] + "..." if local_cert else None,
        }

    remote_key = remote.get("client-key-data")
    local_key = local.get("client-key-data")
    if remote_key != local_key:
        differences["client-key-data"] = {
            "remote": remote_key[:50] + "..." if remote_key else None,
            "local": local_key[:50] + "..." if local_key else None,
        }

    if remote_ca != local_ca:
        differences["certificate-authority-data"] = {
            "remote": remote_ca[:50] + "..." if remote_ca else None,
            "local": local_ca[:50] + "..." if local_ca else None,
        }

    return differences


def update_local_kubeconfig(
    local_path: Path,
    kubeconfig: dict,
    user_name: str,
    cluster_name: str,
    remote_creds: dict,
    remote_ca: str,
) -> None:
    """Update local kubeconfig with remote credentials."""
    # Update user credentials
    for user in kubeconfig.get("users", []):
        if user.get("name") == user_name:
            user["user"]["client-certificate-data"] = remote_creds.get("client-certificate-data")
            user["user"]["client-key-data"] = remote_creds.get("client-key-data")
            break

    # Update cluster CA
    for cluster in kubeconfig.get("clusters", []):
        if cluster.get("name") == cluster_name:
            cluster["cluster"]["certificate-authority-data"] = remote_ca
            break

    # Write back
    with open(local_path, "w") as f:
        yaml.dump(kubeconfig, f, default_flow_style=False)

    print(f"Local kubeconfig updated: {local_path}")


def _upsert_named_entry(entries: list[dict], name: str, new_entry: dict) -> None:
    """Replace an existing entry by name, or append if not found."""
    for i, entry in enumerate(entries):
        if entry.get("name") == name:
            entries[i] = new_entry
            return
    entries.append(new_entry)


def create_context(
    local_path: Path,
    kubeconfig: dict,
    context_name: str,
    cluster_name: str,
    user_name: str,
    server_url: str,
    remote_creds: dict,
    remote_ca: str,
) -> None:
    """Create a new context/cluster/user in the local kubeconfig."""
    # Ensure base structure
    kubeconfig.setdefault("apiVersion", "v1")
    kubeconfig.setdefault("kind", "Config")
    kubeconfig.setdefault("preferences", {})
    kubeconfig.setdefault("clusters", [])
    kubeconfig.setdefault("users", [])
    kubeconfig.setdefault("contexts", [])

    # Add or update cluster entry
    cluster_entry = {
        "name": cluster_name,
        "cluster": {
            "certificate-authority-data": remote_ca,
            "server": server_url,
        },
    }
    _upsert_named_entry(kubeconfig["clusters"], cluster_name, cluster_entry)

    # Add or update user entry
    user_entry = {
        "name": user_name,
        "user": {
            "client-certificate-data": remote_creds.get("client-certificate-data"),
            "client-key-data": remote_creds.get("client-key-data"),
        },
    }
    _upsert_named_entry(kubeconfig["users"], user_name, user_entry)

    # Add or update context entry
    context_entry = {
        "name": context_name,
        "context": {
            "cluster": cluster_name,
            "user": user_name,
        },
    }
    _upsert_named_entry(kubeconfig["contexts"], context_name, context_entry)

    # Set as current-context if none is set
    if not kubeconfig.get("current-context"):
        kubeconfig["current-context"] = context_name

    # Write back
    with open(local_path, "w") as f:
        yaml.dump(kubeconfig, f, default_flow_style=False)

    print(f"Created context '{context_name}' (cluster={cluster_name}, user={user_name}) in {local_path}")


def derive_names_from_context(context_name: str) -> tuple[str, str]:
    """Derive user_name and cluster_name from context_name.

    User names must be globally unique across contexts to avoid collisions
    in the kubeconfig users list. We use the full context name as user name.

    'admin@mycluster' → user='admin@mycluster', cluster='mycluster'
    'mycluster'       → user='mycluster', cluster='mycluster'
    """
    if "@" in context_name:
        cluster_name = context_name.split("@", 1)[1]
        return context_name, cluster_name
    return context_name, context_name


def get_defaults_from_kubeconfig() -> tuple[str | None, str | None]:
    """Try to read default host and context from local kubeconfig's current-context."""
    local_path = Path.home() / ".kube" / "config"
    if not local_path.exists():
        return None, None

    try:
        with open(local_path) as f:
            config = yaml.safe_load(f)
    except Exception:
        return None, None

    context = config.get("current-context")
    host = None

    if context:
        cluster_name = find_context_cluster(config, context)
        if cluster_name:
            for cluster in config.get("clusters", []):
                if cluster.get("name") == cluster_name:
                    server = cluster.get("cluster", {}).get("server", "")
                    parsed = urlparse(server)
                    if parsed.hostname:
                        host = parsed.hostname

    return host, context


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    default_host, default_context = get_defaults_from_kubeconfig()

    parser = argparse.ArgumentParser(description="Compare K3s kubeconfig with local ~/.kube/config")
    parser.add_argument(
        "-u",
        "--user",
        default="root",
        help="SSH user for remote connection (default: root)",
    )
    parser.add_argument(
        "-H",
        "--host",
        default=default_host,
        help=f"Remote host (default: {default_host or 'from kubeconfig'})",
    )
    parser.add_argument(
        "-c",
        "--context",
        default=default_context,
        help=f"Local kubeconfig context (default: {default_context or 'current-context from kubeconfig'})",
    )
    parser.add_argument(
        "--create",
        action="store_true",
        help="Create context/cluster/user if context doesn't exist locally",
    )
    parser.add_argument(
        "-y",
        "--yes",
        action="store_true",
        help="Non-interactive mode (skip confirmation prompts)",
    )
    parser.add_argument(
        "--server",
        default=None,
        help="Override K3s API server URL (default: https://{host}:6443)",
    )

    args = parser.parse_args()

    if not args.host:
        print("Error: could not determine remote host from kubeconfig. Please specify with -H.")
        sys.exit(1)
    if not args.context:
        print("Error: could not determine context from kubeconfig. Please specify with -c.")
        sys.exit(1)

    return args


def main() -> None:
    args = parse_args()

    remote_host = f"{args.user}@{args.host}"
    remote_kubeconfig_path = "/etc/rancher/k3s/k3s.yaml"
    local_kubeconfig_path = Path.home() / ".kube" / "config"
    target_context = args.context

    print(f"Fetching kubeconfig from {remote_host}:{remote_kubeconfig_path}...")
    remote_kubeconfig = get_remote_kubeconfig(remote_host, remote_kubeconfig_path)

    print(f"Loading local kubeconfig: {local_kubeconfig_path}...")
    local_kubeconfig = load_local_kubeconfig(local_kubeconfig_path, allow_missing=args.create)

    # Extract remote credentials (first user/cluster from k3s.yaml)
    remote_users = remote_kubeconfig.get("users", [])
    if not remote_users:
        print("No users found in remote kubeconfig.")
        sys.exit(1)
    remote_creds = remote_users[0].get("user", {})

    remote_clusters = remote_kubeconfig.get("clusters", [])
    remote_ca = remote_clusters[0].get("cluster", {}).get("certificate-authority-data") if remote_clusters else None

    # Find context locally
    local_user = find_context_user(local_kubeconfig, target_context)
    local_cluster = find_context_cluster(local_kubeconfig, target_context)

    if local_user and local_cluster:
        # --- Existing context: compare/update flow ---
        print(f"Context: {target_context}")
        print(f"  User: {local_user}")
        print(f"  Cluster: {local_cluster}")

        local_creds = get_user_credentials(local_kubeconfig, local_user)
        local_ca = get_cluster_ca(local_kubeconfig, local_cluster)

        print("\nComparing certificate data...")
        differences = compare_credentials(remote_creds, local_creds, remote_ca, local_ca)

        if not differences:
            print("All certificate data matches.")
            sys.exit(0)

        print("\nDifferences found:")
        for key, vals in differences.items():
            print(f"  {key}:")
            print(f"    Remote: {vals['remote']}")
            print(f"    Local:  {vals['local']}")

        if args.yes:
            do_update = True
        else:
            print()
            response = input(f"Update local credentials for context '{target_context}'? [y/N]: ")
            do_update = response.lower() in ("y", "yes")

        if do_update:
            if not remote_ca:
                print("Warning: remote CA data is missing, skipping CA update.")
            update_local_kubeconfig(
                local_kubeconfig_path,
                local_kubeconfig,
                local_user,
                local_cluster,
                remote_creds,
                remote_ca or "",
            )
        else:
            print("No changes made.")

    elif args.create:
        # --- Create new context ---
        user_name, cluster_name = derive_names_from_context(target_context)
        server_url = args.server or f"https://{args.host}:6443"

        if not remote_ca:
            print("Error: remote CA data is missing, cannot create context.")
            sys.exit(1)

        create_context(
            local_kubeconfig_path,
            local_kubeconfig,
            target_context,
            cluster_name,
            user_name,
            server_url,
            remote_creds,
            remote_ca,
        )

    else:
        print(f"Context '{target_context}' not found in local kubeconfig.")
        print("Use --create to create it.")
        sys.exit(1)


if __name__ == "__main__":
    main()
