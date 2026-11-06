"""[dgxarley] llm_broker.py: resolve_llm takes its api_token from crawl4ai_usermap.

The upstream value (llm.api_key from config, else None, which makes LLMConfig fall back to OPENAI_API_KEY) is passed in
as the default; with the usermap on, the module returns the caller's key, that default (fallback on) or a placeholder.
"""

from _c4lib import IMPORT_LINE, Patch, edit, patch_for

patch = patch_for("llm_broker usermap hooks", "llm_broker.py")


@patch.run
def apply(p: Patch) -> None:
    edit(
        p,
        "from typing import Dict, Optional\n",
        "from typing import Dict, Optional\n\n" + IMPORT_LINE,
        IMPORT_LINE,
        "llm_broker import",
    )
    edit(
        p,
        '"api_token": get_llm_api_key(config, provider),   # server credential',
        '"api_token": usermap.llm_api_token(config, provider, get_llm_api_key(config, provider)),',
        "usermap.llm_api_token(",
        "llm_broker api_token",
    )
