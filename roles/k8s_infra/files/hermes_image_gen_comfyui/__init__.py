"""Local ComfyUI image generation backend for Hermes.

Routes the ``image_generate`` tool to an in-cluster (or any reachable) ComfyUI
server over its REST API, instead of a cloud provider (FAL / OpenAI / xAI).
Default workflow is FLUX.1-schnell (4-step, cfg 1.0) using the all-in-one fp8
checkpoint our ComfyUI deployment ships (``flux1-schnell-fp8.safetensors``).

This is a USER plugin (``~/.hermes/plugins/image_gen/comfyui/`` →
``$HERMES_HOME/plugins/image_gen/comfyui/``). It is OPT-IN: it does nothing
until it is both
  1. enabled   — add ``comfyui`` to ``plugins.enabled`` in config.yaml, and
  2. selected  — set ``image_gen.provider: comfyui`` in config.yaml.
See README.md for deploy + enable steps and the SM121 reliability caveat.

Server URL resolution (first hit wins):
  1. ``COMFYUI_URL`` env var
  2. ``image_gen.comfyui.url`` in config.yaml
  3. ``http://127.0.0.1:8188`` (default)

Flow (text-to-image only in v1):
  POST /prompt {prompt: <api-format graph>, client_id} -> {prompt_id}
  poll GET /history/{prompt_id} until outputs appear
  GET /view?filename=&subfolder=&type=output -> image bytes (cached locally)
"""

from __future__ import annotations

import logging
import os
import time
import uuid
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlencode

import requests

from agent.image_gen_provider import (
    DEFAULT_ASPECT_RATIO,
    ImageGenProvider,
    error_response,
    resolve_aspect_ratio,
    save_url_image,
    success_response,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Defaults / catalog
# ---------------------------------------------------------------------------

DEFAULT_URL = "http://127.0.0.1:8188"
DEFAULT_CHECKPOINT = "flux1-schnell-fp8.safetensors"
DEFAULT_STEPS = 4  # FLUX.1-schnell is a 4-step distilled model
DEFAULT_TIMEOUT_S = 300  # GB10/SM121 ComfyUI first-run can be slow

# FLUX likes dimensions that are multiples of 16.
_SIZES: Dict[str, Tuple[int, int]] = {
    "landscape": (1216, 832),
    "square": (1024, 1024),
    "portrait": (832, 1216),
}

# Single virtual model — the picker / image_gen.model can reference this id.
_MODELS: Dict[str, Dict[str, Any]] = {
    "flux-schnell": {
        "display": "FLUX.1-schnell (ComfyUI, local)",
        "speed": "~4 steps",
        "strengths": "Runs on the local cluster ComfyUI — no cloud, no API key.",
    },
}
DEFAULT_MODEL = "flux-schnell"


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


def _load_comfyui_config() -> Dict[str, Any]:
    """Read ``image_gen.comfyui`` from config.yaml (returns {} on any failure)."""
    try:
        from hermes_cli.config import load_config

        cfg = load_config()
        section = cfg.get("image_gen") if isinstance(cfg, dict) else None
        sub = section.get("comfyui") if isinstance(section, dict) else None
        return sub if isinstance(sub, dict) else {}
    except Exception as exc:  # pragma: no cover - defensive
        logger.debug("Could not load image_gen.comfyui config: %s", exc)
        return {}


def _resolve_url() -> str:
    env = os.environ.get("COMFYUI_URL")
    if env and env.strip():
        return env.strip().rstrip("/")
    cfg = _load_comfyui_config()
    url = cfg.get("url")
    if isinstance(url, str) and url.strip():
        return url.strip().rstrip("/")
    return DEFAULT_URL


def _resolve_int(key: str, default: int) -> int:
    cfg = _load_comfyui_config()
    value = cfg.get(key)
    try:
        return int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return default


def _resolve_checkpoint() -> str:
    cfg = _load_comfyui_config()
    ckpt = cfg.get("checkpoint")
    if isinstance(ckpt, str) and ckpt.strip():
        return ckpt.strip()
    return DEFAULT_CHECKPOINT


# ---------------------------------------------------------------------------
# Workflow (ComfyUI API format)
# ---------------------------------------------------------------------------


def _build_workflow(*, prompt: str, width: int, height: int, seed: int, checkpoint: str, steps: int) -> Dict[str, Any]:
    """Minimal FLUX.1-schnell text-to-image graph in ComfyUI API format.

    Uses the all-in-one fp8 checkpoint (MODEL+CLIP+VAE via
    CheckpointLoaderSimple). schnell: cfg=1.0, euler/simple, denoise=1.0.
    """
    return {
        "1": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": checkpoint}},
        "2": {"class_type": "CLIPTextEncode", "inputs": {"text": prompt, "clip": ["1", 1]}},
        # FLUX schnell ignores negatives at cfg=1.0, but KSampler needs the slot.
        "3": {"class_type": "CLIPTextEncode", "inputs": {"text": "", "clip": ["1", 1]}},
        "4": {
            "class_type": "EmptyLatentImage",
            "inputs": {"width": width, "height": height, "batch_size": 1},
        },
        "5": {
            "class_type": "KSampler",
            "inputs": {
                "seed": seed,
                "steps": steps,
                "cfg": 1.0,
                "sampler_name": "euler",
                "scheduler": "simple",
                "denoise": 1.0,
                "model": ["1", 0],
                "positive": ["2", 0],
                "negative": ["3", 0],
                "latent_image": ["4", 0],
            },
        },
        "6": {"class_type": "VAEDecode", "inputs": {"samples": ["5", 0], "vae": ["1", 2]}},
        "7": {"class_type": "SaveImage", "inputs": {"filename_prefix": "hermes", "images": ["6", 0]}},
    }


def _poll_history(base_url: str, prompt_id: str, timeout_s: int) -> Optional[Dict[str, Any]]:
    """Poll /history/<prompt_id> until outputs appear or timeout. Returns the
    history entry for prompt_id, or None on timeout."""
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        try:
            resp = requests.get(f"{base_url}/history/{prompt_id}", timeout=30)
            resp.raise_for_status()
            data = resp.json()
        except Exception as exc:  # transient — keep polling
            logger.debug("history poll error: %s", exc)
            time.sleep(1.5)
            continue
        entry = data.get(prompt_id) if isinstance(data, dict) else None
        if isinstance(entry, dict) and entry.get("outputs"):
            return entry
        time.sleep(1.5)
    return None


# ---------------------------------------------------------------------------
# Provider
# ---------------------------------------------------------------------------


class ComfyUIImageGenProvider(ImageGenProvider):
    """Local ComfyUI text-to-image backend (FLUX.1-schnell)."""

    @property
    def name(self) -> str:
        return "comfyui"

    @property
    def display_name(self) -> str:
        return "ComfyUI (local)"

    def is_available(self) -> bool:
        # A URL always resolves (default localhost); reachability is surfaced
        # at generate() time rather than via a network probe in the picker.
        return True

    def list_models(self) -> List[Dict[str, Any]]:
        return [
            {
                "id": model_id,
                "display": meta["display"],
                "speed": meta["speed"],
                "strengths": meta["strengths"],
                "price": "free (self-hosted)",
            }
            for model_id, meta in _MODELS.items()
        ]

    def default_model(self) -> Optional[str]:
        return DEFAULT_MODEL

    def get_setup_schema(self) -> Dict[str, Any]:
        return {
            "name": "ComfyUI (local)",
            "badge": "local",
            "tag": "Local ComfyUI (FLUX.1-schnell) — no cloud/key. Set COMFYUI_URL or image_gen.comfyui.url.",
            "env_vars": [
                {
                    "key": "COMFYUI_URL",
                    "prompt": "ComfyUI server URL (e.g. http://comfyui.comfyui.svc.cluster.local:8188)",
                    "url": "",
                },
            ],
        }

    def capabilities(self) -> Dict[str, Any]:
        # v1 is text-to-image only. Image-to-image would need an img2img
        # workflow (LoadImage + VAEEncode + denoise<1.0) — add later if needed.
        return {"modalities": ["text"], "max_reference_images": 0}

    def generate(
        self,
        prompt: str,
        aspect_ratio: str = DEFAULT_ASPECT_RATIO,
        *,
        image_url: Optional[str] = None,
        reference_image_urls: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        prompt = (prompt or "").strip()
        aspect = resolve_aspect_ratio(aspect_ratio)

        if not prompt:
            return error_response(
                error="Prompt is required and must be a non-empty string",
                error_type="invalid_argument",
                provider="comfyui",
                aspect_ratio=aspect,
            )

        base_url = _resolve_url()
        checkpoint = _resolve_checkpoint()
        steps = _resolve_int("steps", DEFAULT_STEPS)
        timeout_s = _resolve_int("timeout", DEFAULT_TIMEOUT_S)
        width, height = _SIZES.get(aspect, _SIZES["square"])
        # Deterministic-ish but varied seed; ComfyUI expects an int.
        seed = uuid.uuid4().int % (2**31)

        workflow = _build_workflow(
            prompt=prompt,
            width=width,
            height=height,
            seed=seed,
            checkpoint=checkpoint,
            steps=steps,
        )
        client_id = uuid.uuid4().hex

        # 1. Submit the prompt.
        try:
            resp = requests.post(
                f"{base_url}/prompt",
                json={"prompt": workflow, "client_id": client_id},
                timeout=60,
            )
            resp.raise_for_status()
            submit = resp.json()
        except requests.HTTPError as exc:
            body = exc.response.text[:500] if exc.response is not None else str(exc)
            return error_response(
                error=f"ComfyUI rejected the workflow ({getattr(exc.response, 'status_code', '?')}): {body}",
                error_type="api_error",
                provider="comfyui",
                model=DEFAULT_MODEL,
                prompt=prompt,
                aspect_ratio=aspect,
            )
        except requests.RequestException as exc:
            return error_response(
                error=f"Could not reach ComfyUI at {base_url}: {exc}",
                error_type="connection_error",
                provider="comfyui",
                model=DEFAULT_MODEL,
                prompt=prompt,
                aspect_ratio=aspect,
            )

        node_errors = submit.get("node_errors") if isinstance(submit, dict) else None
        if node_errors:
            return error_response(
                error=f"ComfyUI workflow has node errors: {node_errors}",
                error_type="invalid_workflow",
                provider="comfyui",
                model=DEFAULT_MODEL,
                prompt=prompt,
                aspect_ratio=aspect,
            )
        prompt_id = submit.get("prompt_id") if isinstance(submit, dict) else None
        if not prompt_id:
            return error_response(
                error=f"ComfyUI did not return a prompt_id: {submit}",
                error_type="empty_response",
                provider="comfyui",
                model=DEFAULT_MODEL,
                prompt=prompt,
                aspect_ratio=aspect,
            )

        # 2. Wait for completion.
        entry = _poll_history(base_url, prompt_id, timeout_s)
        if entry is None:
            return error_response(
                error=f"ComfyUI generation timed out after {timeout_s}s (prompt_id={prompt_id})",
                error_type="timeout",
                provider="comfyui",
                model=DEFAULT_MODEL,
                prompt=prompt,
                aspect_ratio=aspect,
            )

        # 3. Find the first output image across all SaveImage nodes.
        images: List[Dict[str, Any]] = []
        for node_out in (entry.get("outputs") or {}).values():
            if isinstance(node_out, dict):
                for img in node_out.get("images", []) or []:
                    if isinstance(img, dict) and img.get("filename"):
                        images.append(img)
        if not images:
            return error_response(
                error="ComfyUI completed but produced no images",
                error_type="empty_response",
                provider="comfyui",
                model=DEFAULT_MODEL,
                prompt=prompt,
                aspect_ratio=aspect,
            )

        first = images[0]
        query = urlencode(
            {
                "filename": first.get("filename", ""),
                "subfolder": first.get("subfolder", ""),
                "type": first.get("type", "output"),
            }
        )
        view_url = f"{base_url}/view?{query}"

        # 4. Materialise the bytes locally (the /view URL is only valid while
        # the server keeps the file; cache it so downstream consumers — chat,
        # Telegram send_photo, email — have a stable path).
        try:
            saved_path = save_url_image(view_url, prefix="comfyui_flux-schnell")
        except Exception as exc:
            return error_response(
                error=f"Could not fetch generated image from ComfyUI: {exc}",
                error_type="io_error",
                provider="comfyui",
                model=DEFAULT_MODEL,
                prompt=prompt,
                aspect_ratio=aspect,
            )

        return success_response(
            image=str(saved_path),
            model=DEFAULT_MODEL,
            prompt=prompt,
            aspect_ratio=aspect,
            provider="comfyui",
            modality="text",
            extra={"size": f"{width}x{height}", "checkpoint": checkpoint, "steps": steps},
        )


# ---------------------------------------------------------------------------
# Plugin entry point
# ---------------------------------------------------------------------------


def register(ctx) -> None:
    """Plugin entry point — wire ``ComfyUIImageGenProvider`` into the registry."""
    ctx.register_image_gen_provider(ComfyUIImageGenProvider())
