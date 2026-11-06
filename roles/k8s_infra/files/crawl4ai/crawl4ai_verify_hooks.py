"""[dgxarley] Post-patch check for the Crawl4AI usermap hooks: exit 1 unless every hooked file parses and carries all hooks.

Usage: crawl4ai_verify_hooks.py <app-dir>. Independent of the patch library's output, so a patch that silently did
nothing is still caught.
"""

import ast
import sys
from pathlib import Path

IMPORT = "import crawl4ai_usermap as usermap"
EXPECTED: dict[str, dict[str, int]] = {
    "api.py": {"bind": 1, "precheck_llm": 3, "cache_mode": 2},
    "auth_gate.py": {"authenticate": 1, "call_app": 1},
    "llm_broker.py": {"llm_api_token": 1},
    "mcp_bridge.py": {"mcp_auth_headers": 1},
    "server.py": {"rate_limit_key": 1},
}


def calls(tree: ast.AST) -> dict[str, int]:
    found: dict[str, int] = {}
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "usermap"
        ):
            found[node.func.attr] = found.get(node.func.attr, 0) + 1
    return found


def check(app: Path) -> list[str]:
    problems: list[str] = []
    for name, want in EXPECTED.items():
        path = app / name
        try:
            text = path.read_text(encoding="utf-8")
            tree = ast.parse(text, filename=name)
        except (OSError, SyntaxError) as exc:
            problems.append(f"{name}: {exc}")
            continue
        if IMPORT not in text.splitlines():
            problems.append(f"{name}: missing '{IMPORT}'")
        have = calls(tree)
        for fn, count in want.items():
            if have.get(fn, 0) != count:
                problems.append(f"{name}: expected {count} call(s) of usermap.{fn}, found {have.get(fn, 0)}")
    if not (app / "crawl4ai_usermap.py").is_file():
        problems.append("crawl4ai_usermap.py missing")
    return problems


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print("usage: crawl4ai_verify_hooks.py <app-dir>", file=sys.stderr)
        return 2
    problems = check(Path(argv[1]))
    for line in problems:
        print(f"HOOK-MISSING: {line}")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
