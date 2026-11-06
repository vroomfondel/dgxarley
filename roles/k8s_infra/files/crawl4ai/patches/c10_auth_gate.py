"""[dgxarley] auth_gate.py: per-user token lookup and request identity for crawl4ai_usermap.

Three hooks: the import, usermap.authenticate(token) between the static operator token and the JWT check, and
usermap.call_app(...) in place of the final app call (sets the identity ContextVars). Inactive module = upstream.
"""

from _c4lib import IMPORT_LINE, Patch, edit, patch_for

patch = patch_for("auth_gate usermap hooks", "auth_gate.py")


@patch.run
def apply(p: Patch) -> None:
    edit(
        p,
        "from auth import constant_time_eq, decode_token\n",
        "from auth import constant_time_eq, decode_token\n" + IMPORT_LINE,
        IMPORT_LINE,
        "auth_gate import",
    )
    edit(
        p,
        "        # 2) HS256 JWT minted by this server\n",
        "        if (user := usermap.authenticate(token)) is not None:\n"
        "            return user\n\n"
        "        # 2) HS256 JWT minted by this server\n",
        "usermap.authenticate(token)",
        "auth_gate jwt step",
    )
    edit(
        p,
        '        state["principal"] = principal\n        await self.app(scope, receive, send)',
        '        state["principal"] = principal\n        await usermap.call_app(self.app, scope, receive, send)',
        "usermap.call_app(",
        "auth_gate app call",
    )
