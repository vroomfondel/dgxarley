"""Mailboxes of the Hermes agents: one address domain on both mail backends, IMAP/SMTP hosts per backend.

``hermes_resolve_email`` fills every ``hermes_users[*].email`` block (address, hosts, ports) from the backend
settings, so tasks and templates keep reading ``item.email.*``. ``mail_domains_with_hermes`` adds the agents'
mailboxes to ``mail_domains`` when the cluster's own Postfix/Dovecot is the backend.
"""

import copy
import re
from collections.abc import Callable, Mapping, Sequence
from typing import Any

from ansible.errors import AnsibleFilterError  # type: ignore[import-untyped]

LOCAL_RE = re.compile(r"^[a-z0-9]([a-z0-9._-]*[a-z0-9])?$")
DOMAIN_RE = re.compile(r"^[a-z0-9]([a-z0-9.-]*[a-z0-9])?$")
HOST_KEYS = ("imap_host", "imap_port", "smtp_host", "smtp_port")


def _setting(settings: Mapping[str, Any], key: str) -> Any:
    if key not in settings:
        raise AnsibleFilterError(f"hermes_mail_settings: {key} missing")
    return settings[key]


def mail_domain_for(service_domain: str, prefix: str, replace: Sequence[str]) -> str:
    """Return the address domain for a service domain: ``<prefix>.<service domain with replace[0] -> replace[1]>``."""
    if len(replace) != 2 or not all(isinstance(r, str) for r in replace):
        raise AnsibleFilterError("hermes_mail_domain_replace must be [from, to]")
    old, new = replace
    base = service_domain.replace(old, new, 1) if old else service_domain
    return f"{prefix}.{base}" if prefix else base


def resolve_email(user: Mapping[str, Any], settings: Mapping[str, Any]) -> dict[str, Any]:
    """Return a copy of *user* whose ``email`` block carries the effective address, hosts and ports.

    Args:
        user: One ``hermes_users`` entry.
        settings: ``hermes_mail_settings`` (backend, backends, tenants, service_domain, domain_prefix,
            domain_replace, enabled_default).

    Returns:
        The user; without an ``email`` block unchanged, with email disabled only ``enabled`` and the address are set.

    Raises:
        AnsibleFilterError: Unknown backend, unset host, missing password, or a bad local part / domain.
    """
    out = dict(user)
    if not isinstance(user.get("email"), Mapping):
        return out
    email = dict(user["email"])
    email["enabled"] = bool(email.get("enabled", _setting(settings, "enabled_default")))
    out["email"] = email
    username = str(user.get("username", ""))
    where = f"hermes_users[{username}].email"

    # The address is the agent's identity (Outline, Seafile) even while the gateway is off.
    address = str(email.get("address", "") or "")
    if address:
        local, _, domain = address.rpartition("@")
    else:
        local = str(email.get("local", "") or username)
        tenant = user.get("tenant")
        tenants = _setting(settings, "tenants") or {}
        tenant_cfg = tenants.get(tenant, {}) if tenant else {}
        if tenant and not tenant_cfg:
            raise AnsibleFilterError(f"{where}: tenant {tenant!r} is not a key of tenants")
        domain = str(tenant_cfg.get("mail_domain", "") or "")
        if not domain:
            service_domain = str(tenant_cfg.get("domain") or _setting(settings, "service_domain"))
            domain = mail_domain_for(
                service_domain, str(_setting(settings, "domain_prefix")), _setting(settings, "domain_replace")
            )
    if not LOCAL_RE.match(local):
        raise AnsibleFilterError(f"{where}: local part {local!r} must match {LOCAL_RE.pattern}")
    if not DOMAIN_RE.match(domain):
        raise AnsibleFilterError(f"{where}: domain {domain!r} must match {DOMAIN_RE.pattern}")
    email.update(local=local, domain=domain, address=f"{local}@{domain}")
    if not email["enabled"]:
        return out

    backend = str(_setting(settings, "backend"))
    backends = _setting(settings, "backends")
    if backend not in backends:
        raise AnsibleFilterError(f"hermes_mail_backend {backend!r} must be one of {sorted(backends)}")
    hosts = backends[backend]
    if not str(email.get("password", "") or ""):
        raise AnsibleFilterError(f"{where}: password missing")
    email["backend"] = backend
    for key in HOST_KEYS:
        value = email.get(key, hosts.get(key))
        if value in (None, ""):
            raise AnsibleFilterError(
                f"{where}: {key} unset; set hermes_mail_{backend}.{key} (group_vars/all/vault/hermes.sops.yml) or email.{key}"
            )
        email[key] = value
    return out


def hermes_resolve_email(users: object, settings: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Return *users* with every ``email`` block resolved (see :func:`resolve_email`)."""
    if not isinstance(users, Sequence) or isinstance(users, str):
        raise AnsibleFilterError("hermes_resolve_email: hermes_users must be a list")
    return [resolve_email(u, settings) for u in users]


def mail_domains_with_hermes(
    mail_domains: Sequence[Mapping[str, Any]],
    users: Sequence[Mapping[str, Any]],
    settings: Mapping[str, Any],
    hermes_enabled: bool,
) -> list[dict[str, Any]]:
    """Return *mail_domains* plus the mailboxes of the active Hermes agents when the cluster is the backend.

    An explicit entry in *mail_domains* (same domain and local part) wins over a generated one. Generated
    mailboxes carry ``update_password: true`` so a rotated ``email.password`` reaches the mail DB.
    """
    out: list[dict[str, Any]] = [dict(d) for d in copy.deepcopy(list(mail_domains or []))]
    if not hermes_enabled or str(_setting(settings, "backend")) != "cluster":
        return out
    by_domain: dict[str, dict[str, Any]] = {str(d["domain"]): d for d in out}
    for user in hermes_resolve_email(users, settings):
        email = user.get("email") or {}
        if not (user.get("enabled", False) and email.get("enabled", False)):
            continue
        entry = by_domain.get(email["domain"])
        if entry is None:
            entry = {"domain": email["domain"], "mailboxes": []}
            out.append(entry)
            by_domain[email["domain"]] = entry
        mailboxes = entry.setdefault("mailboxes", [])
        if any(m.get("local") == email["local"] for m in mailboxes):
            continue
        mailboxes.append(
            {
                "local": email["local"],
                "name": f"Hermes Agent {user['username']}",
                "password": email["password"],
                "update_password": True,
            }
        )
    return out


class FilterModule:
    """Ansible entry point."""

    def filters(self) -> dict[str, Callable[..., Any]]:
        """Return the filters this plugin provides."""
        return {
            "hermes_resolve_email": hermes_resolve_email,
            "mail_domains_with_hermes": mail_domains_with_hermes,
        }
