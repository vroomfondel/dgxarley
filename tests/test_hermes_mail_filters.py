"""Tests for roles/k8s_infra/filter_plugins/hermes_mail.py (Hermes mailbox resolution and mail_domains generation)."""

import importlib.util
from pathlib import Path
from typing import Any

import pytest

REPO = Path(__file__).resolve().parent.parent
PLUGIN = REPO / "roles" / "k8s_infra" / "filter_plugins" / "hermes_mail.py"

_spec = importlib.util.spec_from_file_location("hermes_mail", PLUGIN)
assert _spec is not None and _spec.loader is not None
hermes_mail: Any = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(hermes_mail)

from ansible.errors import AnsibleFilterError  # type: ignore[import-untyped]  # noqa: E402


def settings(**over: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "backend": "external",
        "backends": {
            "external": {
                "imap_host": "imap.ext.example",
                "imap_port": 143,
                "smtp_host": "smtp.ext.example",
                "smtp_port": 587,
            },
            "cluster": {
                "imap_host": "hub.dgx.clients.example",
                "imap_port": 993,
                "smtp_host": "hub.dgx.clients.example",
                "smtp_port": 587,
            },
        },
        "tenants": {
            "acme": {"domain": "acme.clients.example"},
            "pinned": {"domain": "p.clients.example", "mail_domain": "mail.pinned.example"},
        },
        "service_domain": "dgx.clients.example",
        "domain_prefix": "hermes",
        "domain_replace": [".clients.", ".clientmail."],
        "enabled_default": True,
    }
    base.update(over)
    return base


def user(**over: Any) -> dict[str, Any]:
    base: dict[str, Any] = {"username": "alice", "enabled": True, "email": {"password": "s3cret-password"}}
    base.update(over)
    return base


def test_mail_domain_for() -> None:
    assert (
        hermes_mail.mail_domain_for("acme.clients.example", "hermes", [".clients.", ".clientmail."])
        == "hermes.acme.clientmail.example"
    )
    assert hermes_mail.mail_domain_for("acme.clients.example", "", ["", ""]) == "acme.clients.example"
    with pytest.raises(AnsibleFilterError):
        hermes_mail.mail_domain_for("x.example", "hermes", [".clients."])


def test_resolve_tenant_user_external_backend() -> None:
    r = hermes_mail.resolve_email(user(tenant="acme"), settings())
    assert r["email"]["address"] == "alice@hermes.acme.clientmail.example"
    assert r["email"]["imap_host"] == "imap.ext.example"
    assert r["email"]["imap_port"] == 143
    assert r["email"]["smtp_port"] == 587
    assert r["email"]["backend"] == "external"
    assert r["email"]["enabled"] is True


def test_resolve_cluster_backend_and_pinned_tenant_domain() -> None:
    r = hermes_mail.resolve_email(user(tenant="pinned"), settings(backend="cluster"))
    assert r["email"]["address"] == "alice@mail.pinned.example"
    assert r["email"]["imap_host"] == "hub.dgx.clients.example"
    assert r["email"]["imap_port"] == 993


def test_resolve_user_without_tenant_uses_service_domain() -> None:
    r = hermes_mail.resolve_email(user(), settings())
    assert r["email"]["address"] == "alice@hermes.dgx.clientmail.example"


def test_explicit_address_local_and_hosts_win() -> None:
    u = user(
        tenant="acme",
        email={
            "password": "s3cret-password",
            "address": "bot@other.example",
            "imap_host": "imap.other.example",
            "imap_port": 993,
        },
    )
    r = hermes_mail.resolve_email(u, settings())
    assert r["email"]["address"] == "bot@other.example"
    assert r["email"]["local"] == "bot"
    assert r["email"]["imap_host"] == "imap.other.example"
    assert r["email"]["imap_port"] == 993
    assert r["email"]["smtp_host"] == "smtp.ext.example"
    r2 = hermes_mail.resolve_email(user(email={"password": "s3cret-password", "local": "agent"}), settings())
    assert r2["email"]["address"] == "agent@hermes.dgx.clientmail.example"


def test_disabled_email_keeps_identity_only() -> None:
    no_block = user()
    del no_block["email"]
    assert hermes_mail.resolve_email(no_block, settings()) == no_block
    off = hermes_mail.resolve_email(user(tenant="acme", email={"enabled": False}), settings(backend="nope"))
    assert off["email"] == {
        "enabled": False,
        "local": "alice",
        "domain": "hermes.acme.clientmail.example",
        "address": "alice@hermes.acme.clientmail.example",
    }, "identity resolved, backend not even looked at"
    default_off = hermes_mail.resolve_email(user(email={"address": "x@y.example"}), settings(enabled_default=False))
    assert default_off["email"]["enabled"] is False
    assert default_off["email"]["address"] == "x@y.example"
    assert "imap_host" not in default_off["email"]
    inactive = hermes_mail.resolve_email(user(enabled=False, email={}), settings(backend="nope"))
    assert inactive["email"] == {
        "enabled": True,
        "local": "alice",
        "domain": "hermes.dgx.clientmail.example",
        "address": "alice@hermes.dgx.clientmail.example",
    }


@pytest.mark.parametrize(
    ("u", "s", "match"),
    [
        (user(), settings(backend="nope"), "hermes_mail_backend"),
        (
            user(),
            settings(backends={"external": {"imap_port": 143, "smtp_host": "s", "smtp_port": 587}, "cluster": {}}),
            "imap_host unset",
        ),
        (user(email={}), settings(), "password missing"),
        (user(username="Alice"), settings(), "local part"),
        (user(tenant="ghost"), settings(), "not a key of tenants"),
        (user(email={"password": "x", "address": "a@Bad_Domain"}), settings(), "domain"),
    ],
)
def test_resolve_errors(u: dict[str, Any], s: dict[str, Any], match: str) -> None:
    with pytest.raises(AnsibleFilterError, match=match):
        hermes_mail.resolve_email(u, s)


def test_hermes_resolve_email_list() -> None:
    users = [user(), user(username="bob", enabled=False)]
    out = hermes_mail.hermes_resolve_email(users, settings())
    assert [u["email"]["address"] for u in out] == [
        "alice@hermes.dgx.clientmail.example",
        "bob@hermes.dgx.clientmail.example",
    ]
    assert users[0]["email"] == {"password": "s3cret-password"}, "input must not be mutated"
    with pytest.raises(AnsibleFilterError):
        hermes_mail.hermes_resolve_email("alice", settings())


def test_mail_domains_untouched_for_external_backend_or_disabled_hermes() -> None:
    domains = [{"domain": "x.example", "mailboxes": [{"local": "a", "password": "p" * 12}]}]
    assert hermes_mail.mail_domains_with_hermes(domains, [user()], settings(), True) == domains
    assert hermes_mail.mail_domains_with_hermes(domains, [user()], settings(backend="cluster"), False) == domains
    assert hermes_mail.mail_domains_with_hermes(None, [], settings(backend="cluster"), True) == []


def test_mail_domains_with_hermes_generates_and_merges() -> None:
    domains = [
        {
            "domain": "hermes.acme.clientmail.example",
            "description": "kept",
            "mailboxes": [{"local": "alice", "password": "explicit-pw-1"}],
        },
    ]
    users = [
        user(tenant="acme"),
        user(username="bob", tenant="acme", email={"password": "bob-password-1"}),
        user(username="carol", email={"password": "carol-password"}),
        user(username="dave", enabled=False),
        user(username="erin", email={"enabled": False, "password": "x"}),
    ]
    out = hermes_mail.mail_domains_with_hermes(domains, users, settings(backend="cluster"), True)
    assert domains[0]["mailboxes"] == [{"local": "alice", "password": "explicit-pw-1"}], "input must not be mutated"
    acme = out[0]
    assert acme["description"] == "kept"
    assert [m["local"] for m in acme["mailboxes"]] == ["alice", "bob"]
    assert acme["mailboxes"][0]["password"] == "explicit-pw-1", "explicit mailbox wins"
    assert acme["mailboxes"][1] == {
        "local": "bob",
        "name": "Hermes Agent bob",
        "password": "bob-password-1",
        "update_password": True,
    }
    assert out[1] == {
        "domain": "hermes.dgx.clientmail.example",
        "mailboxes": [
            {"local": "carol", "name": "Hermes Agent carol", "password": "carol-password", "update_password": True}
        ],
    }
    assert len(out) == 2


def test_filter_module_exports() -> None:
    names = set(hermes_mail.FilterModule().filters())
    assert names == {"hermes_resolve_email", "mail_domains_with_hermes"}
