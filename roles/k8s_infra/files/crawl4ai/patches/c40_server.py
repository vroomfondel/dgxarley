"""[dgxarley] server.py: per-user rate-limit key and early load of the users file.

Two hooks: the import and key_func=usermap.rate_limit_key(get_remote_address). The call runs at import time, so with
CRAWL4AI_USERMAP=on a missing or broken users file stops the server there.
"""

from _c4lib import IMPORT_LINE, Patch, edit, patch_for

patch = patch_for("server usermap hooks", "server.py")


@patch.run
def apply(p: Patch) -> None:
    edit(
        p,
        "from auth_gate import AuthGateMiddleware\n",
        "from auth_gate import AuthGateMiddleware\n" + IMPORT_LINE,
        IMPORT_LINE,
        "server import",
    )
    edit(
        p,
        "    key_func=get_remote_address,",
        "    key_func=usermap.rate_limit_key(get_remote_address),",
        "usermap.rate_limit_key(",
        "server limiter key_func",
    )
