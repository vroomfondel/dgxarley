"""[dgxarley] mcp_bridge.py: MCP tool calls reach the REST endpoints as the connected user.

Upstream mints an anonymous service JWT for the loopback call. usermap.mcp_auth_headers(default) returns the caller's own
bearer token when the connection was authenticated with a per-user token, else the upstream headers.
"""

from _c4lib import IMPORT_LINE, Patch, edit, patch_for

patch = patch_for("mcp_bridge usermap hooks", "mcp_bridge.py")


@patch.run
def apply(p: Patch) -> None:
    edit(
        p,
        "from mcp.server.models import InitializationOptions\n",
        "from mcp.server.models import InitializationOptions\n\n" + IMPORT_LINE,
        IMPORT_LINE,
        "mcp_bridge import",
    )
    edit(
        p,
        '    return {"Authorization": f"Bearer {token}"}',
        '    return usermap.mcp_auth_headers({"Authorization": f"Bearer {token}"})',
        "usermap.mcp_auth_headers(",
        "mcp_bridge headers",
    )
