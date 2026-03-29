#!/usr/bin/env python3
"""Compare a K3s kubeconfig from a remote server with the local ``~/.kube/config``.

Fetches ``/etc/rancher/k3s/k3s.yaml`` from the K3s master via SSH, then either:

* **Update mode** (default): compares certificates/CA for an existing local context
  and offers to overwrite them with the remote values.
* **Create mode** (``--create``): bootstraps a brand-new context/cluster/user entry
  in the local kubeconfig from the remote data.

Pass ``--yes`` (or ``-y``) to skip interactive confirmation prompts, which makes
the script safe to call from Ansible without a TTY.

Example:
    Update the current-context credentials non-interactively::

        python update_local_k3s_keys.py --yes

    Create a new context for a freshly installed cluster::

        python update_local_k3s_keys.py --create --yes -H 192.168.191.10 -c ht@dgxarley
"""

import argparse
import subprocess
import sys
from pathlib import Path
from typing import cast
from urllib.parse import urlparse

import yaml

# A kubeconfig is a dict with string keys; values are strings, lists of dicts, etc.
# yaml.safe_load returns Any, so we cast at the boundary and use this alias throughout.
KubeConfig = dict[str, object]
KubeEntry = dict[str, object]


def _as_list(val: object) -> list[KubeEntry]:
    """Safely cast a kubeconfig value to a list of named entries.

    Args:
        val: An arbitrary value retrieved from a parsed kubeconfig dict.

    Returns:
        The value cast to ``list[KubeEntry]`` if it is already a list,
        otherwise an empty list.
    """
    if isinstance(val, list):
        return cast(list[KubeEntry], val)
    return []


def _as_dict(val: object) -> KubeEntry:
    """Safely cast a kubeconfig value to a dict.

    Args:
        val: An arbitrary value retrieved from a parsed kubeconfig dict.

    Returns:
        The value cast to ``KubeEntry`` if it is already a dict,
        otherwise an empty dict.
    """
    if isinstance(val, dict):
        return cast(KubeEntry, val)
    return {}


def _as_str(val: object) -> str | None:
    """Safely cast a kubeconfig value to ``str`` or ``None``.

    Args:
        val: An arbitrary value retrieved from a parsed kubeconfig dict.

    Returns:
        The value unchanged if it is a ``str``, otherwise ``None``.
    """
    if isinstance(val, str):
        return val
    return None


def get_remote_kubeconfig(host: str, remote_path: str) -> KubeConfig:
    """Fetch and parse a kubeconfig file from a remote host via SSH.

    Runs ``ssh <host> cat <remote_path>`` and parses the resulting YAML.
    Exits with code 1 on SSH or subprocess errors.

    Args:
        host: SSH target in ``user@hostname`` form.
        remote_path: Absolute path to the kubeconfig file on the remote host,
            e.g. ``/etc/rancher/k3s/k3s.yaml``.

    Returns:
        The parsed kubeconfig as a ``KubeConfig`` dict.  Returns an empty dict
        if the remote file is empty or contains only ``null``.
    """
    cmd = ["ssh", host, f"cat {remote_path}"]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        return cast(KubeConfig, yaml.safe_load(result.stdout)) or {}
    except subprocess.CalledProcessError as e:
        print(f"Error accessing {host} via SSH: {e.stderr}")
        sys.exit(1)


def load_local_kubeconfig(path: Path, allow_missing: bool = False) -> KubeConfig:
    """Load and parse a local kubeconfig file.

    When ``allow_missing`` is ``True`` and the file does not exist, the parent
    directory is created (``~/.kube/``) and an empty dict is returned so the
    caller can bootstrap a new kubeconfig from scratch.  When ``allow_missing``
    is ``False`` and the file is absent the script exits with code 1.

    Args:
        path: Filesystem path to the local kubeconfig (typically
            ``~/.kube/config``).
        allow_missing: If ``True``, tolerate a missing file and return an empty
            dict instead of exiting.

    Returns:
        The parsed kubeconfig as a ``KubeConfig`` dict, or an empty dict when
        the file is missing and ``allow_missing`` is ``True``.
    """
    if not path.exists():
        if allow_missing:
            path.parent.mkdir(parents=True, exist_ok=True)
            return {}
        print(f"Local kubeconfig not found: {path}")
        sys.exit(1)
    with open(path) as f:
        return cast(KubeConfig, yaml.safe_load(f)) or {}


def find_context_user(kubeconfig: KubeConfig, context_name: str) -> str | None:
    """Find the user name associated with a named context.

    Args:
        kubeconfig: A parsed kubeconfig dict to search.
        context_name: The ``name`` field of the context to look up.

    Returns:
        The ``user`` string from the context's inner dict, or ``None`` if the
        context does not exist or has no user field.
    """
    for ctx in _as_list(kubeconfig.get("contexts", [])):
        if ctx.get("name") == context_name:
            return _as_str(_as_dict(ctx.get("context", {})).get("user"))
    return None


def find_context_cluster(kubeconfig: KubeConfig, context_name: str) -> str | None:
    """Find the cluster name associated with a named context.

    Args:
        kubeconfig: A parsed kubeconfig dict to search.
        context_name: The ``name`` field of the context to look up.

    Returns:
        The ``cluster`` string from the context's inner dict, or ``None`` if
        the context does not exist or has no cluster field.
    """
    for ctx in _as_list(kubeconfig.get("contexts", [])):
        if ctx.get("name") == context_name:
            return _as_str(_as_dict(ctx.get("context", {})).get("cluster"))
    return None


def get_user_credentials(kubeconfig: KubeConfig, user_name: str) -> KubeEntry:
    """Retrieve the credential dict for a named user entry.

    Args:
        kubeconfig: A parsed kubeconfig dict to search.
        user_name: The ``name`` field of the user entry to look up.

    Returns:
        The inner ``user`` dict (containing ``client-certificate-data``,
        ``client-key-data``, etc.) for the matching entry, or an empty dict
        if no matching user is found.
    """
    for user in _as_list(kubeconfig.get("users", [])):
        if user.get("name") == user_name:
            return _as_dict(user.get("user", {}))
    return {}


def get_cluster_ca(kubeconfig: KubeConfig, cluster_name: str) -> str | None:
    """Retrieve the ``certificate-authority-data`` for a named cluster entry.

    Args:
        kubeconfig: A parsed kubeconfig dict to search.
        cluster_name: The ``name`` field of the cluster entry to look up.

    Returns:
        The base64-encoded CA certificate string, or ``None`` if the cluster
        entry is not found or the field is absent.
    """
    for cluster in _as_list(kubeconfig.get("clusters", [])):
        if cluster.get("name") == cluster_name:
            return _as_str(_as_dict(cluster.get("cluster", {})).get("certificate-authority-data"))
    return None


def compare_credentials(
    remote: KubeEntry,
    local: KubeEntry,
    remote_ca: str | None,
    local_ca: str | None,
) -> dict[str, dict[str, str | None]]:
    """Compare remote and local credentials and return any differences found.

    Checks three fields: ``client-certificate-data``, ``client-key-data``
    (from the user credential dicts), and ``certificate-authority-data``
    (passed separately because it lives in the cluster entry).  Values that
    differ are included in the result with a 50-character truncated preview.

    Args:
        remote: The remote user credential dict (``client-certificate-data``,
            ``client-key-data``, etc.).
        local: The local user credential dict to compare against.
        remote_ca: The remote cluster ``certificate-authority-data`` string,
            or ``None`` if absent.
        local_ca: The local cluster ``certificate-authority-data`` string,
            or ``None`` if absent.

    Returns:
        A dict mapping each differing field name to a sub-dict with ``remote``
        and ``local`` keys, each holding the first 50 characters of the
        respective value (suffixed with ``"..."``) or ``None``.  An empty dict
        means all compared fields are identical.
    """
    differences: dict[str, dict[str, str | None]] = {}

    remote_cert = _as_str(remote.get("client-certificate-data"))
    local_cert = _as_str(local.get("client-certificate-data"))
    if remote_cert != local_cert:
        differences["client-certificate-data"] = {
            "remote": remote_cert[:50] + "..." if remote_cert else None,
            "local": local_cert[:50] + "..." if local_cert else None,
        }

    remote_key = _as_str(remote.get("client-key-data"))
    local_key = _as_str(local.get("client-key-data"))
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
    kubeconfig: KubeConfig,
    user_name: str,
    cluster_name: str,
    remote_creds: KubeEntry,
    remote_ca: str,
) -> None:
    """Overwrite the credentials in an existing local kubeconfig context.

    Updates ``client-certificate-data`` on the matching user entry and
    ``certificate-authority-data`` on the matching cluster entry in-place,
    then serialises the modified kubeconfig back to disk.

    Args:
        local_path: Filesystem path of the local kubeconfig file to overwrite.
        kubeconfig: The already-parsed local kubeconfig dict (mutated in place).
        user_name: Name of the user entry to update (must already exist).
        cluster_name: Name of the cluster entry to update (must already exist).
        remote_creds: The remote user credential dict supplying the new
            ``client-certificate-data`` value.
        remote_ca: The new ``certificate-authority-data`` string for the cluster
            entry.
    """
    # Update user credentials
    for user in _as_list(kubeconfig.get("users", [])):
        if user.get("name") == user_name:
            user_data = _as_dict(user.get("user", {}))
            user_data["client-certificate-data"] = remote_creds.get("client-certificate-data")
            user["user"] = user_data
            break

    # Update cluster CA
    for cluster in _as_list(kubeconfig.get("clusters", [])):
        if cluster.get("name") == cluster_name:
            cluster_data = _as_dict(cluster.get("cluster", {}))
            cluster_data["certificate-authority-data"] = remote_ca
            cluster["cluster"] = cluster_data
            break

    # Write back
    with open(local_path, "w") as f:
        yaml.dump(kubeconfig, f, default_flow_style=False)

    print(f"Local kubeconfig updated: {local_path}")


def _upsert_named_entry(entries: list[KubeEntry], name: str, new_entry: KubeEntry) -> None:
    """Replace an existing named entry in a list, or append it if absent.

    Operates in place on ``entries``.  Matching is done by the ``"name"`` key
    of each element.

    Args:
        entries: A list of kubeconfig named-entry dicts (clusters, users, or
            contexts) to search and modify.
        name: The value of the ``"name"`` key to match.
        new_entry: The replacement entry dict.  Replaces the matched element, or
            is appended to ``entries`` when no match is found.
    """
    for i, entry in enumerate(entries):
        if entry.get("name") == name:
            entries[i] = new_entry
            return
    entries.append(new_entry)


def create_context(
    local_path: Path,
    kubeconfig: KubeConfig,
    context_name: str,
    cluster_name: str,
    user_name: str,
    server_url: str,
    remote_creds: KubeEntry,
    remote_ca: str,
) -> None:
    """Bootstrap a new context, cluster, and user entry in the local kubeconfig.

    Ensures the required top-level kubeconfig keys (``apiVersion``, ``kind``,
    ``preferences``, ``clusters``, ``users``, ``contexts``) are present, then
    upserts cluster, user, and context entries derived from the remote data.
    Sets ``current-context`` to ``context_name`` if no current context is
    defined yet.  Serialises the result back to ``local_path``.

    Args:
        local_path: Filesystem path of the local kubeconfig file to write.
        kubeconfig: The already-parsed local kubeconfig dict (mutated in place).
            May be an empty dict when bootstrapping from scratch.
        context_name: Name to assign to the new kubeconfig context.
        cluster_name: Name to assign to the new cluster entry.
        user_name: Name to assign to the new user entry.
        server_url: Full K3s API server URL, e.g. ``https://192.168.191.10:6443``.
        remote_creds: The remote user credential dict supplying
            ``client-certificate-data`` and ``client-key-data``.
        remote_ca: Base64-encoded CA certificate string for the cluster entry.
    """
    # Ensure base structure
    kubeconfig.setdefault("apiVersion", "v1")
    kubeconfig.setdefault("kind", "Config")
    kubeconfig.setdefault("preferences", {})
    kubeconfig.setdefault("clusters", [])
    kubeconfig.setdefault("users", [])
    kubeconfig.setdefault("contexts", [])

    # Add or update cluster entry
    cluster_entry: KubeEntry = {
        "name": cluster_name,
        "cluster": {
            "certificate-authority-data": remote_ca,
            "server": server_url,
        },
    }
    _upsert_named_entry(_as_list(kubeconfig["clusters"]), cluster_name, cluster_entry)

    # Add or update user entry
    user_entry: KubeEntry = {
        "name": user_name,
        "user": {
            "client-certificate-data": remote_creds.get("client-certificate-data"),
            "client-key-data": remote_creds.get("client-key-data"),
        },
    }
    _upsert_named_entry(_as_list(kubeconfig["users"]), user_name, user_entry)

    # Add or update context entry
    context_entry: KubeEntry = {
        "name": context_name,
        "context": {
            "cluster": cluster_name,
            "user": user_name,
        },
    }
    _upsert_named_entry(_as_list(kubeconfig["contexts"]), context_name, context_entry)

    # Set as current-context if none is set
    if not kubeconfig.get("current-context"):
        kubeconfig["current-context"] = context_name

    # Write back
    with open(local_path, "w") as f:
        yaml.dump(kubeconfig, f, default_flow_style=False)

    print(f"Created context '{context_name}' (cluster={cluster_name}, user={user_name}) in {local_path}")


def derive_names_from_context(context_name: str) -> tuple[str, str]:
    """Derive ``user_name`` and ``cluster_name`` from a context name string.

    User names must be globally unique across contexts to avoid collisions in
    the kubeconfig ``users`` list, so the full context name is used as the user
    name.

    Examples::

        "admin@mycluster"  ->  user="admin@mycluster", cluster="mycluster"
        "mycluster"        ->  user="mycluster",        cluster="mycluster"

    Args:
        context_name: A kubeconfig context name, optionally in
            ``<user>@<cluster>`` form.

    Returns:
        A ``(user_name, cluster_name)`` tuple.  When an ``@`` separator is
        present the part after the first ``@`` becomes the cluster name; the
        full string is always used as the user name.
    """
    if "@" in context_name:
        cluster_name = context_name.split("@", 1)[1]
        return context_name, cluster_name
    return context_name, context_name


def get_defaults_from_kubeconfig() -> tuple[str | None, str | None]:
    """Read default host and context from the local kubeconfig's current-context.

    Parses ``~/.kube/config``, follows the ``current-context`` to its cluster
    entry, and extracts the hostname from the cluster's ``server`` URL.  Used
    to provide sensible argument defaults so the script can be run without
    flags when the current-context is already configured correctly.

    Returns:
        A ``(host, context_name)`` tuple.  Either element may be ``None`` if
        ``~/.kube/config`` does not exist, cannot be parsed, or does not
        contain the required fields.
    """
    local_path = Path.home() / ".kube" / "config"
    if not local_path.exists():
        return None, None

    try:
        with open(local_path) as f:
            config = cast(KubeConfig, yaml.safe_load(f))
    except Exception:
        return None, None

    context = _as_str(config.get("current-context"))
    host: str | None = None

    if context:
        cluster_name = find_context_cluster(config, context)
        if cluster_name:
            for cluster in _as_list(config.get("clusters", [])):
                if cluster.get("name") == cluster_name:
                    server = _as_str(_as_dict(cluster.get("cluster", {})).get("server"))
                    if server:
                        parsed = urlparse(server)
                        if parsed.hostname:
                            host = parsed.hostname

    return host, context


def parse_args() -> argparse.Namespace:
    """Build and parse the command-line argument parser.

    Calls ``get_defaults_from_kubeconfig()`` to populate default values for
    ``--host`` and ``--context`` from the current local kubeconfig, so both
    flags are optional when a usable current-context already exists.  Exits
    with code 1 (via ``argparse`` or explicit checks) when mandatory values
    cannot be determined.

    Returns:
        A populated ``argparse.Namespace`` with attributes: ``user``, ``host``,
        ``context``, ``create``, ``yes``, and ``server``.
    """
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
    """Entry point: orchestrate credential comparison or context creation.

    Parses arguments, fetches the remote K3s kubeconfig, and branches into one
    of three paths:

    1. **Existing context** — compares remote vs. local certificates/CA,
       prints differences, and updates if confirmed (or ``--yes`` is set).
    2. **Create mode** (``--create``, context absent locally) — calls
       ``create_context`` to bootstrap the context from scratch.
    3. **Error** — context absent and ``--create`` not passed; exits with
       code 1.
    """
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
    remote_users = _as_list(remote_kubeconfig.get("users", []))
    if not remote_users:
        print("No users found in remote kubeconfig.")
        sys.exit(1)
    remote_creds = _as_dict(remote_users[0].get("user", {}))

    remote_clusters = _as_list(remote_kubeconfig.get("clusters", []))
    remote_ca = (
        _as_str(_as_dict(remote_clusters[0].get("cluster", {})).get("certificate-authority-data"))
        if remote_clusters
        else None
    )

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
