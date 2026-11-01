"""Static checks for the ops_wireguard role and its firewall integration (syntax, tags, templates, README)."""

import re
import shutil
import subprocess
from pathlib import Path
from typing import Any

import jinja2
import pytest
import yaml

REPO = Path(__file__).resolve().parent.parent
ROLE = REPO / "roles" / "ops_wireguard"
IPTABLES_TPL = REPO / "roles" / "common" / "templates" / "iptables.sh.j2"
VENV_BIN = REPO / ".venv" / "bin"

WG_DICT: dict[str, Any] = {
    "enabled": True,
    "interface": "ops-wg",
    "hub_endpoint": "hub.example.invalid:51820",
    "hub_public_key": "HUBPUBKEY=",
    "tunnel_address": "10.20.50.2/32",
    "ops_subnet": "10.20.50.0/24",
    "keepalive": 25,
    "route_mode": "full",
    "route_table": 51820,
    "rule_priority": 1000,
    "private_key_file": "/etc/wireguard/ops-wg.key",
    "forward_lan": True,
    "forward_lans": ["10.40.0.0/24"],
    "forward_masquerade": True,
}
WG_VARS: dict[str, Any] = {"ops_wireguard": WG_DICT}


def _conf_vars(**over: Any) -> dict[str, Any]:
    return {"_ops_wireguard": {**WG_DICT, **over}}


def _tool(name: str) -> str | None:
    venv = VENV_BIN / name
    if venv.exists():
        return str(venv)
    return shutil.which(name)


def _playbook(*args: str) -> subprocess.CompletedProcess[str]:
    exe = _tool("ansible-playbook")
    if exe is None:
        pytest.skip("ansible-playbook not available")
    return subprocess.run([exe, "common.yml", *args], cwd=REPO, capture_output=True, text=True)


def _render(template: Path, **variables: Any) -> str:
    env = jinja2.Environment(
        loader=jinja2.FileSystemLoader(str(template.parent)),
        undefined=jinja2.StrictUndefined,
        keep_trailing_newline=True,
        trim_blocks=True,
        lstrip_blocks=False,
    )
    # the few Ansible filters the templates use
    env.filters["quote"] = lambda s: "'" + str(s).replace("'", "'\"'\"'") + "'"
    env.filters["combine"] = lambda a, b: {**a, **b}
    env.filters["mandatory"] = lambda v: v
    env.filters["dict2items"] = lambda d: [{"key": k, "value": v} for k, v in d.items()]
    env.filters["bool"] = lambda v: v if isinstance(v, bool) else str(v).lower() in ("1", "true", "yes", "on")
    return env.get_template(template.name).render(**variables)


def _render_iptables(**extra: Any) -> str:
    base: dict[str, Any] = {
        "groups": {"k3sserver": []},
        "localnets": [],
        "group_names": [],
        "inventory_hostname": "k3smaster",
        "ansible_facts": {"default_ipv4": {"address": "192.0.2.10"}},
        "hub_ingress": {"enabled": False, "bind_address": "", "port": 8443, "source_ips": []},
    }
    return _render(IPTABLES_TPL, **{**base, **extra})


def test_syntax_check() -> None:
    res = _playbook("--syntax-check")
    assert res.returncode == 0, res.stdout + res.stderr


def test_tag_lists_role_tasks() -> None:
    res = _playbook("--tags", "ops_wireguard", "--list-tasks")
    assert res.returncode == 0, res.stdout + res.stderr
    assert "Ops WireGuard tunnel (box side)" in res.stdout
    for task in (
        "Assert WireGuard parameters",
        "Generate the WireGuard private key on the host (only when missing)",
        "Render /etc/wireguard/<iface>.conf (no secret inside)",
        "Wait for a handshake with the hub (non-fatal)",
        "Stop and disable wg-quick",
        "Remove the tunnel firewall chains",
    ):
        assert f"ops_wireguard : {task}" in res.stdout, task


def test_play_is_gated() -> None:
    plays = yaml.safe_load((REPO / "common.yml").read_text())
    play = next(p for p in plays if p["name"] == "Ops WireGuard tunnel (box side)")
    assert play["hosts"] == "all,!disabled,!switches"
    assert play["become"] is True
    role = play["roles"][0]
    assert role["role"] == "ops_wireguard"
    assert role["tags"] == ["ops_wireguard"]
    assert "ops_wireguard.enabled" in role["when"] and "ops_wireguard.teardown" in role["when"]
    # after the NUT play
    names = [p["name"] for p in plays]
    assert names.index("NUT UPS monitoring") < names.index(play["name"])


def test_defaults() -> None:
    d = yaml.safe_load((ROLE / "defaults" / "main.yml").read_text())["ops_wireguard_defaults"]
    assert d["enabled"] is False
    assert d["teardown"] is False
    assert d["interface"] == "ops-wg"
    assert d["keepalive"] == 25
    assert d["forward_lan"] is False
    assert d["forward_lans"] == []
    assert d["forward_masquerade"] is True
    assert d["private_key"] == ""
    assert d["handshake_wait"] == 30
    assert d["teardown_remove_key"] is False
    # the common firewall template carries the same interface default (role defaults are not loaded there)
    assert "kwg.interface | default('ops-wg')" in IPTABLES_TPL.read_text()


@pytest.mark.parametrize("forward", [True, False])
def test_conf_template_has_no_secret(forward: bool) -> None:
    out = _render(ROLE / "templates" / "etc_wireguard_iface.conf.j2", **_conf_vars(forward_lan=forward))
    assert not re.search(r"^PrivateKey\s*=", out, re.M)
    assert "PostUp = wg set %i private-key /etc/wireguard/ops-wg.key" in out
    assert "wg-forward.sh" not in out
    assert "AllowedIPs = 10.20.50.0/24" in out
    assert "PersistentKeepalive = 25" in out
    assert "Address = 10.20.50.2/32" in out
    assert "Endpoint = hub.example.invalid:51820" in out
    assert "PublicKey = HUBPUBKEY=" in out


def test_conf_route_modes() -> None:
    tpl = ROLE / "templates" / "etc_wireguard_iface.conf.j2"
    full = _render(tpl, **_conf_vars())
    assert "Table =" not in full and "ip -4 rule" not in full
    src = _render(tpl, **_conf_vars(route_mode="source", forward_lan=False))
    assert "Table = 51820" in src
    assert "PostUp = ip -4 rule add from 10.20.50.2 lookup 51820 priority 1000" in src
    assert "PostDown = ip -4 rule del from 10.20.50.2 lookup 51820 priority 1000" in src
    assert "PostUp = wg set %i private-key /etc/wireguard/ops-wg.key" in src
    assert not re.search(r"^PrivateKey\s*=", src, re.M)


def test_defaults_route_mode() -> None:
    d = yaml.safe_load((ROLE / "defaults" / "main.yml").read_text())["ops_wireguard_defaults"]
    assert d["route_mode"] == "full"
    assert d["route_table"] == 51820
    assert d["rule_priority"] == 1000


def test_forward_lan_requires_full_assert() -> None:
    tasks = (ROLE / "tasks" / "main.yml").read_text()
    assert "not _ops_wireguard.forward_lan | bool or _ops_wireguard.route_mode == 'full'" in tasks
    res = _playbook("--tags", "ops_wireguard", "--list-tasks")
    assert "ops_wireguard : Assert WireGuard parameters" in res.stdout
    assert "ops_wireguard : Summary for the hub (hub device stanza)" in res.stdout


def test_firewall_unchanged_when_flag_off() -> None:
    plain = _render_iptables()
    assert _render_iptables(ops_wireguard={"enabled": False, "forward_lan": True}) == plain
    assert "HTOPSWG" not in plain
    assert "ops_wireguard" not in plain.lower()


def test_firewall_rules_when_enabled() -> None:
    out = _render_iptables(**WG_VARS)
    assert "iptables -A HTSTUFFIN -i ops-wg -s 10.20.50.0/24 -j ACCEPT" in out
    assert "iptables -A HTOPSWG -i ops-wg -s 10.20.50.0/24 -d 10.40.0.0/24 -j ACCEPT" in out
    assert "iptables -A HTOPSWG -o ops-wg -m conntrack --ctstate ESTABLISHED,RELATED -j ACCEPT" in out
    assert "iptables -t nat -A HTOPSWGNAT -s 10.20.50.0/24 -d 10.40.0.0/24 ! -o ops-wg -j MASQUERADE" in out
    # INPUT rule sits before the generic localnets ssh rule and the chain's final DROP
    assert out.index("-i ops-wg -s 10.20.50.0/24 -j ACCEPT") < out.index("-m set --match-set xosrc src -j RETURN")


def test_firewall_without_forward_or_masquerade() -> None:
    no_fwd = _render_iptables(ops_wireguard={**WG_DICT, "forward_lan": False})
    assert "-i ops-wg -s 10.20.50.0/24 -j ACCEPT" in no_fwd
    assert "HTOPSWG" not in no_fwd
    no_masq = _render_iptables(ops_wireguard={**WG_DICT, "forward_masquerade": False})
    assert "HTOPSWG -i" in no_masq
    assert "MASQUERADE" not in no_masq


HUB_INGRESS: dict[str, Any] = {
    "enabled": True,
    "bind_address": "10.20.50.10",
    "port": 8443,
    "source_ips": ["10.20.50.1/32"],
}


def test_firewall_hub_ingress_drop_sits_between_hub_return_and_ops_accept() -> None:
    out = _render_iptables(**WG_VARS, hub_ingress=HUB_INGRESS, isk3smaster=True)
    hub = "iptables -A HTSTUFFIN -m state --state NEW -p tcp -m multiport --dports 8443 -s 10.20.50.1/32 -j RETURN"
    drop = "iptables -A HTSTUFFIN -i ops-wg -m state --state NEW -p tcp -m multiport --dports 8443 -j DROP"
    ops = "iptables -A HTSTUFFIN -i ops-wg -s 10.20.50.0/24 -j ACCEPT"
    assert out.index(hub) < out.index(drop) < out.index(ops)


@pytest.mark.parametrize(
    "extra",
    [
        {"hub_ingress": HUB_INGRESS},
        {"hub_ingress": {**HUB_INGRESS, "enabled": False}, "isk3smaster": True},
        {},
    ],
)
def test_firewall_no_hub_ingress_rules_unless_enabled_on_master(extra: dict[str, Any]) -> None:
    out = _render_iptables(**WG_VARS, **extra)
    assert "--dports 8443" not in out


def test_firewall_hub_ingress_drop_needs_the_ops_tunnel() -> None:
    out = _render_iptables(hub_ingress=HUB_INGRESS, isk3smaster=True)
    assert "--dports 8443 -s 10.20.50.1/32 -j RETURN" in out
    assert "--dports 8443 -j DROP" not in out


def test_firewall_hub_ingress_enabled_mail_gate_ports_join_return_and_drop() -> None:
    mail = {
        "smtp": {"enabled": True, "gate_port": 8025, "nodeport": 30025},
        "imaps": {"enabled": True, "gate_port": 8993, "nodeport": 30993},
        "imap": {"enabled": False, "gate_port": 8143, "nodeport": 30143},
    }
    out = _render_iptables(**WG_VARS, hub_ingress=HUB_INGRESS, hub_ingress_mail_services=mail, isk3smaster=True)
    assert "-m multiport --dports 8443,8025,8993 -s 10.20.50.1/32 -j RETURN" in out
    assert "-i ops-wg -m state --state NEW -p tcp -m multiport --dports 8443,8025,8993 -j DROP" in out
    assert "8143" not in out


def test_firewall_rendered_script_is_valid_bash() -> None:
    bash = shutil.which("bash")
    if bash is None:
        pytest.skip("bash not available")
    out = _render_iptables(**WG_VARS)
    res = subprocess.run([bash, "-n"], input=out, capture_output=True, text=True)
    assert res.returncode == 0, res.stderr


def test_readme_variables_table_aligned_and_in_sync() -> None:
    lines = (ROLE / "README.md").read_text().splitlines()
    start = next(i for i, ln in enumerate(lines) if ln.startswith("| Variable ") and "Default" in ln)
    table = []
    for ln in lines[start:]:
        if not ln.startswith("|"):
            break
        table.append(ln)
    assert len(table) > 5
    bars = [m.start() for m in re.finditer(r"\|", table[0])]
    for ln in table:
        assert [m.start() for m in re.finditer(r"\|", ln)] == bars, ln
    documented = {ln.split("|")[1].strip().strip("`") for ln in table[2:]}
    defaults = yaml.safe_load((ROLE / "defaults" / "main.yml").read_text())["ops_wireguard_defaults"]
    assert documented == {f"ops_wireguard.{k}" for k in defaults}


def test_root_readme_role_table_aligned() -> None:
    lines = (REPO / "README.md").read_text().splitlines()
    start = next(i for i, ln in enumerate(lines) if ln.startswith("| Role "))
    table = []
    for ln in lines[start:]:
        if not ln.startswith("|"):
            break
        table.append(ln)
    bars = [m.start() for m in re.finditer(r"\|", table[0])]
    for ln in table:
        assert [m.start() for m in re.finditer(r"\|", ln)] == bars, ln
    assert any(ln.startswith("| `ops_wireguard`") for ln in table)


VPN_INGRESS: dict[str, Any] = {
    "enabled": True,
    "target_host": "k3smaster",
    "tcp_ports": [443],
    "networks": ["10.20.30.128/26"],
}
VPN_ACCEPT = "iptables -A HTSTUFFIN -s 10.20.30.128/26 -m state --state NEW -p tcp -m multiport --dport 443 -j RETURN"
VPN_DROP = "iptables -A HTSTUFFIN -s 10.20.30.128/26 -m state --state NEW -j DROP"


def test_firewall_vpn_ingress_rules_precede_public_ports() -> None:
    out = _render_iptables(vpn_ingress=VPN_INGRESS, iptables_public_tcp_ports=[22, 6443])
    lines = out.splitlines()
    public = next(i for i, ln in enumerate(lines) if "multiport --dport 22,6443 -j RETURN" in ln)
    assert lines.index(VPN_ACCEPT) + 1 == lines.index(VPN_DROP) < public


@pytest.mark.parametrize(
    "extra",
    [
        {},
        {"vpn_ingress": {**VPN_INGRESS, "enabled": False}},
        {"vpn_ingress": {**VPN_INGRESS, "target_host": "other"}},
    ],
)
def test_firewall_no_vpn_ingress_rules_unless_enabled_on_target(extra: dict[str, Any]) -> None:
    out = _render_iptables(**extra)
    assert "10.20.30.128/26" not in out
    assert out == _render_iptables()


def test_firewall_vpn_ingress_batches_ports_by_15() -> None:
    ports = list(range(1000, 1016))
    out = _render_iptables(vpn_ingress={**VPN_INGRESS, "tcp_ports": ports})
    assert out.count("-s 10.20.30.128/26 -m state --state NEW -p tcp -m multiport") == 2
