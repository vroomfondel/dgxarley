"""[dgxarley] api.py: LLM pre-checks (403), job identity binding and cache bypass for crawl4ai_usermap.

Hooks: import; usermap.bind(factory) in _enqueue_job; usermap.precheck_llm(config, provider) in handle_llm_qa, in the
LLM branch of handle_markdown_request and in create_new_task; usermap.cache_mode(...) at the two cache_mode lines.
"""

from _c4lib import IMPORT_LINE, Patch, edit, patch_for

patch = patch_for("api usermap hooks", "api.py")
CACHE_OLD = 'cache_mode = CacheMode.ENABLED if cache == "1" else CacheMode.WRITE_ONLY'
CACHE_INNER = 'CacheMode.ENABLED if cache == "1" else CacheMode.WRITE_ONLY'


@patch.run
def apply(p: Patch) -> None:
    edit(
        p,
        "from llm_broker import LLMProviderNotAllowed\n",
        "from llm_broker import LLMProviderNotAllowed\n" + IMPORT_LINE,
        IMPORT_LINE,
        "api import",
    )
    edit(
        p,
        "    from work_queue import get_job_queue, QueueFull, QuotaExceeded\n",
        "    factory = usermap.bind(factory)\n    from work_queue import get_job_queue, QueueFull, QuotaExceeded\n",
        "factory = usermap.bind(factory)",
        "api _enqueue_job",
    )
    edit(
        p,
        "    crawler = None\n    try:\n"
        "        if not url.startswith(('http://', 'https://')) and not url.startswith((\"raw:\", \"raw://\")):\n"
        "            url = 'https://' + url\n"
        "        validate_url_destination(url)\n        # Extract base URL",
        "    crawler = None\n    try:\n        usermap.precheck_llm(config, provider)\n"
        "        if not url.startswith(('http://', 'https://')) and not url.startswith((\"raw:\", \"raw://\")):\n"
        "            url = 'https://' + url\n"
        "        validate_url_destination(url)\n        # Extract base URL",
        "        usermap.precheck_llm(config, provider)\n        if not url.startswith",
        "api handle_llm_qa",
    )
    edit(
        p,
        "        if filter_type == FilterType.LLM:\n            is_valid, error_msg",
        "        if filter_type == FilterType.LLM:\n            usermap.precheck_llm(config, provider)\n            is_valid, error_msg",
        "            usermap.precheck_llm(config, provider)\n            is_valid, error_msg",
        "api handle_markdown_request",
    )
    edit(
        p,
        '    """Create and initialize a new task."""\n',
        '    """Create and initialize a new task."""\n    usermap.precheck_llm(config, provider)\n',
        '"""Create and initialize a new task."""\n    usermap.precheck_llm(config, provider)',
        "api create_new_task",
    )
    new_a = f"cache_mode = usermap.cache_mode({CACHE_INNER}, CacheMode.BYPASS)"
    edit(
        p,
        f"        {CACHE_OLD}\n\n        # Re-validate the destination",
        f"        {new_a}\n\n        # Re-validate the destination",
        new_a,
        "api process_llm_extraction cache",
    )
    new_b = f"cache_mode = usermap.cache_mode({CACHE_INNER}, CacheMode.BYPASS, filter_type == FilterType.LLM)"
    edit(
        p,
        f"        {CACHE_OLD}\n\n        from crawler_pool",
        f"        {new_b}\n\n        from crawler_pool",
        new_b,
        "api handle_markdown_request cache",
    )
