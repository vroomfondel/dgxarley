"""OpenWebUI / SGLang integration tests.

Exercises the full request path from a local Python script through OpenWebUI
(or directly against SGLang) using the model's recommended sampling parameters
read from the Ansible defaults file.

Sampling Presets:
    Recommended sampling parameters are loaded from the active model's
    ``recommended_sampling`` dict in ``roles/k8s_dgx/defaults/main.yml``
    (the Ansible single source of truth for model profiles).

    SGLang itself has no server-side default sampling flags — temperature,
    top_p, top_k etc. are always per-request via the OpenAI-compatible API.
    This script reads them from the Ansible defaults so every test call
    automatically uses the model author's recommended values.

    The ``recommended_sampling`` dict is always flat (keys like
    ``temperature``, ``top_p``, ``top_k`` directly). It is wrapped into
    a single preset called ``"default"``.

    Thinking vs. non-thinking is toggled per request:
    - API: ``extra_body={"chat_template_kwargs": {"enable_thinking": false}}``
    - Prompt shortcut: start message with ``/no_think`` or ``/think``

Architecture:
    ``LLMClient`` is the base class handling streaming, preset loading, and
    response parsing.  Two subclasses adapt to different backends:

    - ``OpenWebUIClient``: Auth via API key, ``extra_body`` wrapper,
      supports ``features`` (web_search).
    - ``SGLangClient``: No auth, flattens ``extra_body`` into top-level
      payload, no ``features`` support.

Usage::

    # Via OpenWebUI (default):
    python openwebui_integration_test.py thinking coding presets

    # Via direct SGLang:
    python sglang_integration_test.py thinking coding presets

    # All tests:
    python openwebui_integration_test.py all
"""

import json
import os
import sys
import time
from pathlib import Path

import requests
import yaml
from dgxarley import configure_logging, glogger, print_banner
from dgxarley.integration.thinking_parser import ThinkingParser

os.environ.setdefault("LOGURU_LEVEL", "DEBUG")
configure_logging()
glogger.enable("dgxarley")

try:
    from ascii_magic import AsciiArt
except Exception:
    AsciiArt = None

try:
    from PIL import Image
except Exception:
    Image = None  # type: ignore[assignment]

_VISION_AVAILABLE: bool = AsciiArt is not None and Image is not None

from loguru import logger

# Repo root: override via DGXARLEY_ROOT env var, otherwise 2 levels above __file__
_REPO_ROOT: Path = Path(os.environ.get("DGXARLEY_ROOT") or Path(__file__).resolve().parents[2]).resolve()

# Ansible defaults: override via DGXARLEY_DEFAULTS env var
_defaults_path: Path = Path(
    os.environ.get("DGXARLEY_DEFAULTS") or (_REPO_ROOT / "roles" / "k8s_dgx" / "defaults" / "main.yml")
).resolve()

# Load .env / .env.local from repo root (does not override existing env vars)
_env_files: list[Path] = [_REPO_ROOT / ".env", _REPO_ROOT / ".env.local"]
for _env_file in _env_files:
    if _env_file.is_file():
        for line in _env_file.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            key = key.strip()
            value = value.strip().strip("\"'")
            os.environ.setdefault(key, value)


# ---------------------------------------------------------------------------
# Model profiles & sampling presets (from Ansible defaults)
# ---------------------------------------------------------------------------

if not _defaults_path.is_file():
    logger.warning(
        f"Ansible defaults not found: {_defaults_path} " "(set DGXARLEY_ROOT or DGXARLEY_DEFAULTS to override)"
    )
    _dgx_defaults: dict[str, object] = {}
else:
    with open(_defaults_path) as _f:
        _dgx_defaults = yaml.safe_load(_f)
# YAML-loaded data has heterogeneous structure; annotated as dict[str, object]
_MODEL_PROFILES: dict[str, object] = _dgx_defaults.get("sglang_model_profiles", {})  # type: ignore[assignment]


def load_sampling_presets(
    model_id: str,
) -> dict[str, dict[str, str | int | float | bool | dict[str, str | int | float | bool | dict[str, bool]]]]:
    """Build sampling presets from a model's recommended_sampling in the Ansible defaults.

    Merges two layers:
      1. recommended_sampling — flat dict of values from the model creator.
      2. sampling_overrides   — local tuning on top (e.g. anti-repetition).

    Overrides win on key conflicts.

    Args:
        model_id: The model identifier string, as used in ``sglang_model_profiles``
            in the Ansible defaults file.

    Returns:
        A mapping with a single ``"default"`` preset containing sampling
        parameters ready to be merged into an OpenAI-compatible chat
        completion payload.  Returns an empty dict if the model has no
        sampling configuration.
    """
    profile: dict[str, object] = _MODEL_PROFILES.get(model_id, {})  # type: ignore[assignment]
    raw: dict[str, object] = profile.get("recommended_sampling", {})  # type: ignore[assignment]
    if not raw:
        return {}

    overrides: dict[str, object] = profile.get("sampling_overrides", {})  # type: ignore[assignment]
    merged: dict[str, object] = {**raw, **overrides}

    # OpenAI-compatible top-level keys
    _OPENAI_KEYS: set[str] = {"temperature", "top_p", "presence_penalty", "frequency_penalty", "repetition_penalty"}
    # SGLang/vLLM-native keys (sent via extra_body)
    _EXTRA_KEYS: set[str] = {"top_k", "min_p", "top_min_p", "min_tokens"}

    preset: dict[str, str | int | float | bool | dict[str, str | int | float | bool | dict[str, bool]]] = {}
    for k in _OPENAI_KEYS:
        if k in merged:
            preset[k] = merged[k]  # type: ignore[assignment]
    extra: dict[str, str | int | float | bool | dict[str, bool]] = {}
    for k in _EXTRA_KEYS:
        if k in merged:
            extra[k] = merged[k]  # type: ignore[assignment]
    if extra:
        preset["extra_body"] = extra
    return {"default": preset}


def pick_default_preset(
    presets: dict[str, dict[str, str | int | float | bool | dict[str, str | int | float | bool | dict[str, bool]]]],
) -> str | None:
    """Pick the default preset name from the available presets.

    Args:
        presets: Mapping of preset name to sampling parameter dict, as returned
            by :func:`load_sampling_presets`.

    Returns:
        ``"default"`` if present, otherwise the first key, or ``None`` if empty.
    """
    if "default" in presets:
        return "default"
    if presets:
        return list(presets)[0]
    return None


# ---------------------------------------------------------------------------
# LLMClient base class
# ---------------------------------------------------------------------------


class LLMClient:
    """Base class for streaming LLM chat completions with sampling presets.

    Subclasses must implement :meth:`_endpoint` and may override
    :meth:`_headers`, :meth:`_prepare_payload`, and :meth:`_supports_features`.

    Attributes:
        base_url: Base URL of the backend (trailing slash stripped).
        model_id: Model identifier string passed in every request.
        verbose: When ``True``, prints the full request payload before sending.
        presets: Sampling presets loaded from the Ansible defaults for this model.
        default_preset: Name of the preset selected by :func:`pick_default_preset`,
            or ``None`` if no presets are available.
    """

    base_url: str
    model_id: str
    verbose: bool
    presets: dict[str, dict[str, str | int | float | bool | dict[str, str | int | float | bool | dict[str, bool]]]]
    default_preset: str | None

    def __init__(self, base_url: str, model_id: str, verbose: bool = False) -> None:
        """Initialise the client and load sampling presets for the given model.

        Args:
            base_url: Base URL of the backend service.
            model_id: Model identifier string, used both in requests and for
                looking up sampling presets in the Ansible defaults.
            verbose: If ``True``, print the full request payload JSON before
                each request.
        """
        self.base_url = base_url.rstrip("/")
        self.model_id = model_id
        self.verbose = verbose
        self.presets = load_sampling_presets(model_id)
        self.default_preset = pick_default_preset(self.presets)

    # -- Subclass hooks --

    def _endpoint(self) -> str:
        """Return the full chat completions URL for this backend.

        Returns:
            The complete URL string to POST chat completion requests to.

        Raises:
            NotImplementedError: Always — subclasses must implement this method.
        """
        raise NotImplementedError

    def _headers(self) -> dict[str, str]:
        """Return HTTP request headers for this backend.

        Returns:
            A dict of header name to header value strings.  The base
            implementation returns only ``Content-Type: application/json``.
        """
        return {"Content-Type": "application/json"}

    def _prepare_payload(
        self,
        payload: dict[
            str,
            str
            | int
            | float
            | bool
            | list[dict[str, str | list[dict[str, str | dict[str, str]]]]]
            | dict[str, str | int | float | bool],
        ],
    ) -> dict[
        str,
        str
        | int
        | float
        | bool
        | list[dict[str, str | list[dict[str, str | dict[str, str]]]]]
        | dict[str, str | int | float | bool],
    ]:
        """Transform the payload before sending it to the backend.

        The base implementation returns the payload unchanged.  Subclasses
        may flatten ``extra_body`` or strip unsupported fields.

        Args:
            payload: The assembled OpenAI-compatible chat completion payload.

        Returns:
            The (potentially modified) payload dict ready to be serialised as
            JSON and sent to :meth:`_endpoint`.
        """
        return payload

    def _supports_features(self) -> bool:
        """Return whether this backend supports OpenWebUI ``features`` (e.g. web_search).

        Returns:
            ``True`` if the ``features`` key is accepted in the request payload,
            ``False`` otherwise.  The base implementation returns ``False``.
        """
        return False

    # -- Preset application --

    def apply_preset(
        self,
        payload: dict[
            str,
            str
            | int
            | float
            | bool
            | list[dict[str, str | list[dict[str, str | dict[str, str]]]]]
            | dict[str, str | int | float | bool],
        ],
        preset: "str | None | ellipsis" = ...,
        allow_fallback: bool = True,
    ) -> dict[
        str,
        str
        | int
        | float
        | bool
        | list[dict[str, str | list[dict[str, str | dict[str, str]]]]]
        | dict[str, str | int | float | bool],
    ]:
        """Merge a named sampling preset into the request payload.

        The ``preset`` parameter uses ``...`` (Ellipsis) as a sentinel meaning
        "use :attr:`default_preset`".  Pass ``None`` explicitly to skip preset
        application entirely.

        For ``non_thinking`` presets the last user message is also prefixed
        with ``/no_think`` so the directive works both through OpenWebUI
        (which ignores ``chat_template_kwargs``) and direct SGLang (which
        honours both, belt-and-suspenders).

        Args:
            payload: The assembled OpenAI-compatible chat completion payload.
                Modified in-place and returned.
            preset: Name of the preset to apply, ``None`` to skip, or ``...``
                (Ellipsis) to use :attr:`default_preset`.
            allow_fallback: If ``True`` and ``preset`` is not found, emit a
                warning and fall back to :attr:`default_preset`.  If ``False``,
                raise :class:`ValueError`.

        Returns:
            The payload dict with sampling parameters merged in.

        Raises:
            ValueError: If ``preset`` is not found and ``allow_fallback`` is
                ``False``.
        """
        if preset is ...:
            preset = self.default_preset
        if preset is None:
            return payload
        if preset not in self.presets:
            if allow_fallback:
                available: list[str] = list(self.presets) or ["(none)"]
                print(
                    f"  [WARN] Preset '{preset}' not available for {self.model_id} "
                    f"(available: {available}), using '{self.default_preset}'"
                )
                if self.default_preset is None:
                    return payload
                preset = self.default_preset
            else:
                raise ValueError(f"Unknown preset '{preset}'. Available: {list(self.presets)}")
        p = self.presets[preset]
        for k in ("temperature", "top_p", "presence_penalty", "frequency_penalty", "repetition_penalty"):
            if k in p:
                payload[k] = p[k]  # type: ignore[assignment]
        if "extra_body" in p:
            payload.setdefault("extra_body", {})
            payload["extra_body"].update(p["extra_body"])  # type: ignore[union-attr,arg-type]

        # For non_thinking presets: prepend /no_think to the last user message.
        # This works both via OpenWebUI (which ignores chat_template_kwargs) and
        # direct SGLang (which honours both, belt-and-suspenders).
        if preset.startswith("non_thinking"):
            messages = payload.get("messages", [])
            for msg in reversed(messages):  # type: ignore[arg-type]
                if msg.get("role") == "user":  # type: ignore[attr-defined]
                    content = msg.get("content", "")  # type: ignore[attr-defined]
                    if isinstance(content, str) and not content.startswith("/no_think"):
                        msg["content"] = f"/no_think\n{content}"  # type: ignore[index]
                    elif isinstance(content, list):
                        # Multimodal message — prepend to first text part
                        for part in content:
                            if part.get("type") == "text" and not part["text"].startswith("/no_think"):
                                part["text"] = f"/no_think\n{part['text']}"
                                break
                    break

        return payload

    # -- Streaming --

    def _niceprint_payload(
        self,
        payload: dict[
            str,
            str
            | int
            | float
            | bool
            | list[dict[str, str | list[dict[str, str | dict[str, str]]]]]
            | dict[str, str | int | float | bool],
        ],
    ) -> None:
        """Print a human-readable payload summary to stdout.

        Base64-encoded image data is replaced with a placeholder to keep the
        output readable.  String message contents longer than 200 characters
        are truncated.

        Args:
            payload: The request payload dict to summarise.
        """
        display: dict[str, object] = {}
        for k, v in payload.items():
            if k == "messages":
                # Summarize messages — truncate image data
                msgs: list[dict[str, object]] = []
                for m in v:  # type: ignore[union-attr]
                    content = m.get("content", "")  # type: ignore[union-attr]
                    if isinstance(content, list):
                        parts: list[dict[str, object]] = []
                        for p in content:
                            if p.get("type") == "image_url":
                                parts.append({"type": "image_url", "image_url": "(base64 omitted)"})
                            else:
                                parts.append(p)  # type: ignore[arg-type]
                        msgs.append({**m, "content": parts})  # type: ignore[dict-item]
                    elif isinstance(content, str) and len(content) > 200:
                        msgs.append({**m, "content": content[:200] + "..."})  # type: ignore[dict-item]
                    else:
                        msgs.append(m)  # type: ignore[arg-type]
                display[k] = msgs
            else:
                display[k] = v
        print(f"\033[2m[payload] {json.dumps(display, indent=2, ensure_ascii=False)}\n[/payload]\033[0m")

    def stream_chat(
        self,
        payload: dict[
            str,
            str
            | int
            | float
            | bool
            | list[dict[str, str | list[dict[str, str | dict[str, str]]]]]
            | dict[str, str | int | float | bool],
        ],
        print_thinking: bool = True,
    ) -> dict[str, int]:
        """Stream a chat completion request and print the response to stdout.

        Parses the SSE stream, printing ``reasoning_content`` deltas in dim
        ANSI style wrapped in ``<think>…</think>`` markers and ``content``
        deltas as plain text.

        Args:
            payload: A fully assembled, preset-applied chat completion payload.
                Will be passed through :meth:`_prepare_payload` before sending.
            print_thinking: If ``True``, reasoning (``reasoning_content``) deltas
                are printed to stdout.  If ``False``, they are consumed silently.

        Returns:
            The ``usage`` dict from the final SSE chunk (token counts), or an
            empty dict if the server did not emit usage data.

        Raises:
            requests.HTTPError: If the server returns a non-2xx status code.
        """
        payload = self._prepare_payload(payload)
        if self.verbose:
            self._niceprint_payload(payload)
        response = requests.post(
            self._endpoint(),
            headers=self._headers(),
            json=payload,
            stream=True,
            timeout=(10, 300),
        )
        response.raise_for_status()

        tp = ThinkingParser()
        in_thinking: bool = False
        usage: dict[str, int] = {}
        for raw_line in response.iter_lines():
            if not raw_line:
                continue
            decoded: str = raw_line.decode("utf-8")
            if not decoded.startswith("data: "):
                continue
            data: str = decoded[6:]
            if data == "[DONE]":
                break
            chunk: dict[str, object] = json.loads(data)
            if "usage" in chunk:
                usage = chunk["usage"]  # type: ignore[assignment]
            choices: list[dict[str, object]] = chunk.get("choices", [{}])  # type: ignore[assignment]
            delta: dict[str, str] = choices[0].get("delta", {})  # type: ignore[assignment]
            result = tp.feed(
                content=delta.get("content", ""),
                reasoning_content=delta.get("reasoning_content", ""),
            )
            if result.thinking:
                if print_thinking:
                    if not in_thinking:
                        print("\033[2m<think>", end="", flush=True)
                        in_thinking = True
                    print(result.thinking, end="", flush=True)
            if result.content:
                if in_thinking:
                    if print_thinking:
                        print("</think>\033[0m\n", end="", flush=True)
                    in_thinking = False
                print(result.content, end="", flush=True)
        if in_thinking:
            print("</think>\033[0m", end="", flush=True)
        print()

        # Token breakdown summary
        total_tok = usage.get("completion_tokens", tp.thinking_tokens_est + tp.content_tokens_est)
        prompt_tok = usage.get("prompt_tokens", 0)
        print(
            f"\033[2m  tokens: {total_tok} total"
            f" (think ~{tp.thinking_tokens_est} / content ~{tp.content_tokens_est})"
            f" | prompt: {prompt_tok}\033[0m"
        )
        return usage

    # -- Convenience helpers --

    def chat(
        self,
        messages: list[dict[str, str | list[dict[str, str | dict[str, str]]]]],
        preset: str | None = ...,  # type: ignore[assignment]
        print_thinking: bool = True,
        stream: bool = True,
        **extra_payload: str | int | float | bool,
    ) -> dict[str, int]:
        """Build a payload, apply a sampling preset, and stream the response.

        Convenience wrapper around :meth:`apply_preset` and
        :meth:`stream_chat`.

        The ``preset`` parameter uses ``...`` (Ellipsis) as a sentinel meaning
        "use :attr:`default_preset`".  Pass ``None`` explicitly to skip preset
        application.

        Args:
            messages: List of chat message dicts with ``"role"`` and
                ``"content"`` keys, following the OpenAI messages format.
            preset: Named preset to apply, ``None`` to skip, or ``...``
                (Ellipsis) to use :attr:`default_preset`.
            print_thinking: Whether to print reasoning deltas to stdout.
            stream: Whether to enable SSE streaming (should remain ``True``).
            **extra_payload: Additional top-level fields merged into the
                payload before preset application (e.g. ``max_tokens``).

        Returns:
            The ``usage`` dict from the final SSE chunk, or an empty dict.
        """
        payload: dict[str, object] = {"model": self.model_id, "messages": messages, "stream": stream, **extra_payload}
        payload = self.apply_preset(payload, preset)  # type: ignore[assignment,arg-type]
        return self.stream_chat(payload, print_thinking=print_thinking)  # type: ignore[arg-type]

    def explain_image(
        self,
        image: Image.Image,
        print_thinking: bool = True,
        preset: str | None = ...,  # type: ignore[assignment]
    ) -> None:
        """Send an image to the model and ask it to describe the content.

        The image is base64-encoded and embedded in a multimodal user message.
        The prompt is in German, as it is an intentional LLM prompt.

        Args:
            image: A Pillow ``Image`` object to describe.
            print_thinking: Whether to print reasoning deltas to stdout.
            preset: Named preset to apply, ``None`` to skip, or ``...``
                (Ellipsis) to use :attr:`default_preset`.
        """
        import base64
        from io import BytesIO

        buf = BytesIO()
        image.save(buf, format="PNG")
        b64: str = base64.b64encode(buf.getvalue()).decode("utf-8")

        self.chat(
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "Beschreibe dieses Bild. Was siehst du? Wenn es ein Comic ist, erkläre den Witz.",
                        },
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
                    ],
                }
            ],
            preset=preset,
            print_thinking=print_thinking,
        )

    def get_daily_briefing(
        self,
        print_thinking: bool = True,
        preset: str | None = ...,  # type: ignore[assignment]
    ) -> None:
        """Request a daily news briefing, optionally with live web search.

        Sends a German-language system prompt and user message (intentional LLM
        prompts).  If the backend supports the ``features`` key (OpenWebUI),
        ``web_search`` is enabled so the model can fetch current news.

        Args:
            print_thinking: Whether to print reasoning deltas to stdout.
            preset: Named preset to apply, ``None`` to skip, or ``...``
                (Ellipsis) to use :attr:`default_preset`.
        """
        payload: dict[str, object] = {
            "model": self.model_id,
            "messages": [
                {
                    "role": "system",
                    "content": "Erstelle eine übersicht über die geschehnisse der nacht. so im sinne eines daily briefings",
                },
                {"role": "user", "content": "Erstelle mir bitte das Daily Briefing für heute."},
            ],
            "stream": True,
        }
        if self._supports_features():
            payload["features"] = {"web_search": True}
        payload = self.apply_preset(payload, preset)  # type: ignore[assignment,arg-type]

        try:
            print("--- Daily Briefing ---")
            self.stream_chat(payload, print_thinking=print_thinking)  # type: ignore[arg-type]
        except requests.exceptions.RequestException as e:
            print(f"Request error: {e}")


# ---------------------------------------------------------------------------
# OpenWebUI client
# ---------------------------------------------------------------------------


class OpenWebUIClient(LLMClient):
    """LLM client routed via OpenWebUI.

    Adds Bearer token authentication and enables the ``features`` key so
    OpenWebUI can activate web search before forwarding the request to SGLang.

    Attributes:
        api_key: The OpenWebUI API key used in the ``Authorization`` header.
    """

    api_key: str

    def __init__(self, base_url: str, model_id: str, api_key: str, verbose: bool = False) -> None:
        """Initialise the OpenWebUI client.

        Args:
            base_url: Base URL of the OpenWebUI instance.
            model_id: Model identifier string.
            api_key: OpenWebUI API key (generate under Account -> API Keys).
            verbose: If ``True``, print the full request payload before sending.
        """
        super().__init__(base_url, model_id, verbose=verbose)
        self.api_key = api_key

    def _endpoint(self) -> str:
        """Return the OpenWebUI chat completions URL.

        Returns:
            The full URL string for the OpenWebUI chat completions endpoint.
        """
        return f"{self.base_url}/api/chat/completions"

    def _headers(self) -> dict[str, str]:
        """Return request headers including the Bearer token.

        Returns:
            A dict containing ``Authorization`` and ``Content-Type`` headers.
        """
        return {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}

    def _supports_features(self) -> bool:
        """Indicate that OpenWebUI supports the ``features`` payload key.

        Returns:
            Always ``True``.
        """
        return True


# ---------------------------------------------------------------------------
# SGLang direct client
# ---------------------------------------------------------------------------


class SGLangClient(LLMClient):
    """LLM client speaking directly to the SGLang server (no auth, flattened extra_body).

    SGLang expects SGLang-native keys (``top_k``, ``min_p``, etc.) and
    ``chat_template_kwargs`` as top-level fields in the JSON body rather than
    nested under ``extra_body``.  This client flattens the ``extra_body`` dict
    before sending.
    """

    def _endpoint(self) -> str:
        """Return the SGLang OpenAI-compatible chat completions URL.

        Returns:
            The full URL string for the SGLang chat completions endpoint.
        """
        return f"{self.base_url}/v1/chat/completions"

    def _prepare_payload(
        self,
        payload: dict[
            str,
            str
            | int
            | float
            | bool
            | list[dict[str, str | list[dict[str, str | dict[str, str]]]]]
            | dict[str, str | int | float | bool],
        ],
    ) -> dict[
        str,
        str
        | int
        | float
        | bool
        | list[dict[str, str | list[dict[str, str | dict[str, str]]]]]
        | dict[str, str | int | float | bool],
    ]:
        """Flatten ``extra_body`` into top-level fields for SGLang compatibility.

        SGLang expects keys such as ``top_k``, ``min_p``, and
        ``chat_template_kwargs`` at the top level of the JSON body.
        Also removes the ``features`` key, which SGLang does not support.

        Args:
            payload: The assembled chat completion payload before sending.

        Returns:
            The payload with ``extra_body`` contents merged to the top level
            and the ``features`` key removed.
        """
        extra = payload.pop("extra_body", None)
        if extra:
            payload.update(extra)  # type: ignore[arg-type]
        # SGLang does not support OpenWebUI features
        payload.pop("features", None)
        return payload


# ---------------------------------------------------------------------------
# XKCD helpers (backend-independent)
# ---------------------------------------------------------------------------


def get_random_xkcd_image_url() -> str:
    """Fetch a random XKCD comic and return its image URL.

    Follows the redirect from the XKCD random comic page, then fetches the
    comic's JSON metadata to extract the image URL.

    Returns:
        The direct URL of the comic's image (e.g. a PNG or JPEG).

    Raises:
        requests.HTTPError: If any HTTP request fails.
    """
    resp = requests.get("https://c.xkcd.com/random/comic/", timeout=10, allow_redirects=True)
    comic: dict[str, str | int] = requests.get(f"{resp.url}info.0.json", timeout=10).json()
    return comic["img"]  # type: ignore[return-value]


def get_random_xkcd_image(url: str) -> Image.Image:
    """Download an image from a URL and return it as a Pillow Image.

    Args:
        url: The direct URL of the image to download.

    Returns:
        A Pillow ``Image`` object with the downloaded content.

    Raises:
        requests.HTTPError: If the HTTP request fails.
    """
    from io import BytesIO

    resp = requests.get(url, timeout=10)
    resp.raise_for_status()
    return Image.open(BytesIO(resp.content))


def print_ascii_representation_of_image(image: Image.Image) -> None:
    """Render a Pillow image as ASCII art and print it to the terminal.

    The image is converted to RGB, then rendered at 120 columns with
    image enhancement enabled.

    Args:
        image: The Pillow ``Image`` object to render.
    """
    art = AsciiArt.from_pillow_image(image.convert("RGB"))
    art.to_terminal(columns=120, enhance_image=True)


# ---------------------------------------------------------------------------
# Test functions (operate on any LLMClient)
# ---------------------------------------------------------------------------


def test_thinking_mode(client: LLMClient, print_thinking: bool = True) -> None:
    """Run a basic arithmetic reasoning test using the ``thinking`` preset.

    Asks the model to sum the first 20 prime numbers and prints the elapsed
    time and token usage after the response completes.

    Args:
        client: The :class:`LLMClient` instance to use.
        print_thinking: Whether to print reasoning deltas to stdout.
    """
    print("\n=== Thinking Mode (default) ===")
    t0: float = time.monotonic()
    usage: dict[str, int] = client.chat(
        messages=[{"role": "user", "content": "What is the sum of the first 20 prime numbers?"}],
        preset="thinking",
        print_thinking=print_thinking,
    )
    elapsed: float = time.monotonic() - t0
    print(f"  [{elapsed:.1f}s, {usage}]")


def test_non_thinking_mode(client: LLMClient) -> None:
    """Run a simple factual query with thinking disabled.

    Uses the default preset but overrides ``enable_thinking`` to ``False``.

    Args:
        client: The :class:`LLMClient` instance to use.
    """
    print("\n=== Non-Thinking Mode ===")
    t0: float = time.monotonic()
    payload: dict[str, object] = {
        "model": client.model_id,
        "messages": [{"role": "user", "content": "What is the capital of France? Answer in one sentence."}],
        "stream": True,
        "extra_body": {"chat_template_kwargs": {"enable_thinking": False}},
    }
    payload = client.apply_preset(payload)  # type: ignore[assignment,arg-type]
    usage: dict[str, int] = client.stream_chat(payload, print_thinking=False)  # type: ignore[arg-type]
    elapsed: float = time.monotonic() - t0
    print(f"  [{elapsed:.1f}s, {usage}]")


def test_thinking_coding(client: LLMClient) -> None:
    """Run a code generation test with thinking enabled and lower temperature.

    Uses lower temperature (0.6) for more precise code generation.

    Args:
        client: The :class:`LLMClient` instance to use.
    """
    print("\n=== Thinking Mode (Coding) ===")
    t0: float = time.monotonic()
    usage: dict[str, int] = client.chat(
        messages=[
            {
                "role": "user",
                "content": "Write a Python function that checks if a string is a valid IPv4 address without using ipaddress module.",
            }
        ],
        print_thinking=True,
        temperature=0.6,
    )
    elapsed: float = time.monotonic() - t0
    print(f"  [{elapsed:.1f}s, {usage}]")


def test_sampling_params_passthrough(client: LLMClient) -> None:
    """Verify that temperature and top_p are actually applied by the backend.

    Sends the same single-word prompt three times each at low temperature
    (0.1) and high temperature (1.5) and prints the results.  Low temperature
    should produce near-identical outputs; high temperature should vary.

    Args:
        client: The :class:`LLMClient` instance to use.
    """
    prompt: str = "Give me a single random word."
    print("\n=== Sampling Parameter Passthrough Test ===")

    for label, temp, top_p in [("low temp (0.1)", 0.1, 0.5), ("high temp (1.5)", 1.5, 1.0)]:
        print(f"\n  --- {label} ---")
        for i in range(3):
            payload: dict[str, object] = {
                "model": client.model_id,
                "messages": [{"role": "user", "content": prompt}],
                "stream": True,
                "temperature": temp,
                "top_p": top_p,
                "max_tokens": 20,
                "extra_body": {"chat_template_kwargs": {"enable_thinking": False}},
            }
            print(f"    Run {i+1}: ", end="")
            client.stream_chat(client._prepare_payload(payload), print_thinking=False)  # type: ignore[arg-type]

    print("\n  (Low temp should produce similar/identical words, high temp should vary)")


def test_all_presets(client: LLMClient) -> None:
    """Run the same debugging question with the default sampling preset.

    Args:
        client: The :class:`LLMClient` instance to use.
    """
    prompt: str = "How would you approach debugging a memory leak in a Python web application?"
    print("\n" + "=" * 80)
    print("=== Default Sampling Preset ===")
    print("=" * 80)

    print(f"\n--- Preset: default ---")
    t0: float = time.monotonic()
    usage: dict[str, int] = client.chat(
        messages=[{"role": "user", "content": prompt}],
        print_thinking=False,
        max_tokens=512,
    )
    elapsed: float = time.monotonic() - t0
    print(f"  [default: {elapsed:.1f}s, {usage}]")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def create_openwebui_client(verbose: bool = False) -> OpenWebUIClient:
    """Create an :class:`OpenWebUIClient` from environment variables.

    Reads ``MODEL_ID``, ``OPEN_WEBUI_URL``, and ``OPENWEBUI_API_KEY`` (or
    ``API_KEY`` as a fallback) from the environment.  The ``.env`` and
    ``.env.local`` files in the repo root are loaded at module import time, so
    they are available here if present.

    Args:
        verbose: Passed through to :class:`OpenWebUIClient` to enable payload
            logging.

    Returns:
        A fully configured :class:`OpenWebUIClient` instance.

    Raises:
        ValueError: If neither ``OPENWEBUI_API_KEY`` nor ``API_KEY`` is set.
    """
    model_id: str = os.environ.get("MODEL_ID", "nvidia/MiniMax-M2.5-NVFP4")
    owui_url: str = os.environ.get("OPEN_WEBUI_URL", "https://openwebui.example.com")
    api_key: str = os.environ.get("OPENWEBUI_API_KEY", os.environ.get("API_KEY", ""))
    if not api_key:
        raise ValueError(
            "Set OPENWEBUI_API_KEY environment variable. " "Generate at: OpenWebUI -> User -> Account -> API Keys"
        )
    print(f"[OpenWebUI] {owui_url} model={model_id}")
    return OpenWebUIClient(owui_url, model_id, api_key, verbose=verbose)


def main() -> None:
    """Parse CLI arguments and run the selected integration tests.

    Supported test names: ``xkcd``, ``xkcd_non_thinking``, ``briefing``,
    ``briefing_non_thinking``, ``thinking``, ``non_thinking``, ``coding``,
    ``sampling``, ``presets``, ``all``.  The special name ``all`` expands to
    the full set.  Defaults to ``xkcd briefing``.
    """
    print_banner(module=Path(__file__).stem)
    import argparse

    parser = argparse.ArgumentParser(description="OpenWebUI / SGLang integration tests")
    parser.add_argument(
        "tests",
        nargs="*",
        default=["xkcd", "briefing"],
        help="Tests to run: xkcd, xkcd_non_thinking, briefing, briefing_non_thinking, "
        "thinking, non_thinking, coding, sampling, presets, all "
        "(default: xkcd briefing)",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Print full payload JSON before each request",
    )
    parser.add_argument(
        "-y",
        "--yes",
        action="store_true",
        help="Skip confirmation prompt",
    )
    args = parser.parse_args()

    client: OpenWebUIClient = create_openwebui_client(verbose=args.verbose)

    tests: set[str] = set(args.tests)
    if "all" in tests:
        tests = {
            "xkcd",
            "briefing",
            "xkcd_non_thinking",
            "briefing_non_thinking",
            "thinking",
            "non_thinking",
            "coding",
            "sampling",
            "presets",
        }

    # Show config summary and wait for confirmation
    from rich.console import Console
    from rich.panel import Panel
    from rich.syntax import Syntax

    config_summary = {
        "tests": sorted(tests),
        "verbose": args.verbose,
    }
    Console().print(
        Panel(
            Syntax(json.dumps(config_summary, indent=2, ensure_ascii=False), "json", theme="monokai"),
            title="[bold]Test Configuration[/]",
            border_style="cyan",
        )
    )
    if not args.yes:
        try:
            input("[Enter to run tests, Ctrl+C to abort] ")
        except KeyboardInterrupt:
            print("\nAborted.")
            sys.exit(0)

    if "xkcd" in tests:
        if not _VISION_AVAILABLE:
            glogger.warning("Skipping xkcd test — Pillow or ascii_magic not installed")
        else:
            image: Image.Image = get_random_xkcd_image(get_random_xkcd_image_url())
            print_ascii_representation_of_image(image)
            client.explain_image(image, print_thinking=True, preset="thinking")

    if "xkcd_non_thinking" in tests:
        if not _VISION_AVAILABLE:
            glogger.warning("Skipping xkcd_non_thinking test — Pillow or ascii_magic not installed")
        else:
            image = get_random_xkcd_image(get_random_xkcd_image_url())
            print_ascii_representation_of_image(image)
            client.explain_image(image, print_thinking=True, preset="non_thinking")

    if "briefing" in tests:
        print(f"\n{'*' * 80}")
        t0: float = time.monotonic()
        client.get_daily_briefing(print_thinking=True, preset="thinking")
        elapsed: float = time.monotonic() - t0
        print(f"\n--- Daily Briefing completed in {elapsed:.1f}s ---")

    if "briefing_non_thinking" in tests:
        print(f"\n{'*' * 80}")
        t0 = time.monotonic()
        client.get_daily_briefing(print_thinking=True, preset="non_thinking")
        elapsed = time.monotonic() - t0
        print(f"\n--- Daily Briefing completed in {elapsed:.1f}s ---")

    if "thinking" in tests:
        test_thinking_mode(client, print_thinking=True)

    if "non_thinking" in tests:
        test_non_thinking_mode(client)

    if "coding" in tests:
        test_thinking_coding(client)

    if "sampling" in tests:
        test_sampling_params_passthrough(client)

    if "presets" in tests:
        test_all_presets(client)


if __name__ == "__main__":
    main()
