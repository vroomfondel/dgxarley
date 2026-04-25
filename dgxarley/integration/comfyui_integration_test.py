"""Integration tests for the local ComfyUI playground.

Covers basic reachability and API surface of the ComfyUI backend:
health (root), ``/system_stats``, ``/queue``, and ``/object_info`` — the
last one is also used to verify that at least one checkpoint has been
downloaded by ``comfyui_launch.sh`` (FLUX.1-schnell by default).

Usage::

    python comfyui_integration_test.py
    COMFYUI_URL=https://comfyui.example.com python comfyui_integration_test.py
"""

import copy
import os
import sys
import time
from io import BytesIO
from pathlib import Path
from typing import Any, Callable
from uuid import uuid4

import requests
from PIL import Image, ImageDraw

from dgxarley import configure_logging, glogger, print_banner

os.environ.setdefault("LOGURU_LEVEL", "DEBUG")
configure_logging()
glogger.enable("dgxarley")

# Repo root is 2 levels above __file__ (integration/ -> dgxarley/ -> repo-root)
_REPO_ROOT: Path = Path(__file__).resolve().parents[2]

# Load .env from repo root (does not override existing env vars)
_env_files: list[Path] = [_REPO_ROOT / ".env", _REPO_ROOT / ".env.local"]
for _env_file in _env_files:
    if _env_file.is_file():
        for _line in _env_file.read_text().splitlines():
            _line = _line.strip()
            if not _line or _line.startswith("#") or "=" not in _line:
                continue
            _key, _, _value = _line.partition("=")
            _key = _key.strip()
            _value = _value.strip().strip("\"'")
            os.environ.setdefault(_key, _value)

COMFYUI_URL: str = os.environ.get("COMFYUI_URL", "https://comfyui.example.com")
TIMEOUT: tuple[int, int] = (10, 60)

# Deadline for the full image-generation test (upload → queue → render → poll).
# SDXL + Fooocus inpaint on a single DGX Spark GPU takes ~30-90s in practice.
IMAGE_GEN_DEADLINE_S: int = 300

# Default prompt for the text2image tests; override via COMFYUI_PROMPT.
DEFAULT_PROMPT: str = os.environ.get(
    "COMFYUI_PROMPT",
    "a photo of a red apple on a wooden table, soft daylight, highly detailed",
)

# Sampler defaults per model family.
# RealVisXL = SDXL → klassisch viele steps, cfg ~5, dpmpp_2m+karras.
# FLUX.1-schnell ist distilliert → 4 steps reichen, cfg=1.0 (kein CFG-Guidance,
# der Distill-Loss ersetzt das), sampler=euler+simple. Der All-in-One fp8-
# Checkpoint (Comfy-Org/flux1-schnell) bündelt UNet+CLIP+VAE und lädt direkt
# über CheckpointLoaderSimple — kein separater UNETLoader-Pfad nötig.
SDXL_SAMPLER: dict[str, Any] = {
    "steps": 30,
    "cfg": 5.0,
    "sampler_name": "dpmpp_2m",
    "scheduler": "karras",
}
FLUX_SCHNELL_SAMPLER: dict[str, Any] = {
    "steps": 4,
    "cfg": 1.0,
    "sampler_name": "euler",
    "scheduler": "simple",
}
# FLUX.1 [dev] ist guidance-distilliert (nicht timestep-distilliert wie schnell):
# braucht ~20-25 steps, KSampler-cfg bleibt 1.0, aber die "echte" Guidance
# läuft über den separaten FluxGuidance-Node (typisch 3.5). Negative-Prompt
# wird ignoriert solange cfg=1.0. Same All-in-One-fp8-Format wie schnell,
# also CheckpointLoaderSimple genügt.
FLUX_DEV_SAMPLER: dict[str, Any] = {
    "steps": 25,
    "cfg": 1.0,
    "sampler_name": "euler",
    "scheduler": "simple",
}
FLUX_DEV_GUIDANCE: float = 3.5

# Mirror of
# /home/thiess/pythondev_workspace/GTBauprojekte/gtbauprojekte/gimpplugin/comfy-inpaint/workflow.json
# (RealVisXL_V5.0 + Fooocus inpaint + Crop-and-Stitch). Kept inline so the
# test is self-contained and doesn't depend on a sibling checkout.
INPAINT_WORKFLOW: dict[str, Any] = {
    "3": {
        "inputs": {
            "seed": 0,
            "steps": 35,
            "cfg": 5.0,
            "sampler_name": "dpmpp_2m",
            "scheduler": "karras",
            "denoise": 1.0,
            "model": ["17", 0],
            "positive": ["6", 0],
            "negative": ["7", 0],
            "latent_image": ["14", 0],
        },
        "class_type": "KSampler",
    },
    "4": {
        "inputs": {"ckpt_name": "RealVisXL_V5.0_fp16.safetensors"},
        "class_type": "CheckpointLoaderSimple",
    },
    "6": {
        "inputs": {"text": "a photo of a red apple, highly detailed", "clip": ["4", 1]},
        "class_type": "CLIPTextEncode",
    },
    "7": {
        "inputs": {"text": "blurry, low quality, watermark, text, bad anatomy, deformed", "clip": ["4", 1]},
        "class_type": "CLIPTextEncode",
    },
    "8": {"inputs": {"image": ""}, "class_type": "LoadImage"},
    "9": {"inputs": {"image": "", "channel": "red"}, "class_type": "LoadImageMask"},
    "10": {"inputs": {"pixels": ["16", 1], "vae": ["4", 2]}, "class_type": "VAEEncode"},
    "11": {"inputs": {"samples": ["3", 0], "vae": ["4", 2]}, "class_type": "VAEDecode"},
    "12": {
        "inputs": {"filename_prefix": "comfyui-integration-test", "images": ["18", 0]},
        "class_type": "SaveImage",
    },
    "14": {"inputs": {"samples": ["10", 0], "mask": ["16", 2]}, "class_type": "SetLatentNoiseMask"},
    "15": {
        "inputs": {"head": "fooocus_inpaint_head.pth", "patch": "inpaint_v26.fooocus.patch"},
        "class_type": "INPAINT_LoadFooocusInpaint",
    },
    "16": {
        "inputs": {
            "image": ["8", 0],
            "mask": ["9", 0],
            "downscale_algorithm": "bilinear",
            "upscale_algorithm": "bicubic",
            "preresize": False,
            "preresize_mode": "ensure minimum resolution",
            "preresize_min_width": 1024,
            "preresize_min_height": 1024,
            "preresize_max_width": 8192,
            "preresize_max_height": 8192,
            "mask_fill_holes": True,
            "mask_expand_pixels": 0,
            "mask_invert": False,
            "mask_blend_pixels": 32,
            "mask_hipass_filter": 0.1,
            "extend_for_outpainting": False,
            "extend_up_factor": 1.0,
            "extend_down_factor": 1.0,
            "extend_left_factor": 1.0,
            "extend_right_factor": 1.0,
            "context_from_mask_extend_factor": 1.2,
            "output_resize_to_target_size": True,
            "output_target_width": 1024,
            "output_target_height": 1024,
            "output_padding": "32",
            "device_mode": "gpu (much faster)",
        },
        "class_type": "InpaintCropImproved",
    },
    "17": {
        "inputs": {"model": ["4", 0], "patch": ["15", 0], "latent": ["14", 0]},
        "class_type": "INPAINT_ApplyFooocusInpaint",
    },
    "18": {
        "inputs": {"stitcher": ["16", 0], "inpainted_image": ["11", 0]},
        "class_type": "InpaintStitchImproved",
    },
}


_TMP_DIR: Path = Path("/tmp")


def _download_images_to_tmp(images: list[dict[str, Any]], test_name: str) -> list[str]:
    """Pull every generated image off the ComfyUI ``/view`` endpoint into /tmp.

    Names are prefixed with the test name so multiple workflows in one run
    don't clobber each other. Returns the list of paths actually written.
    """
    saved: list[str] = []
    for img in images:
        filename = img.get("filename")
        if not filename:
            continue
        params = {
            "filename": filename,
            "subfolder": img.get("subfolder", "") or "",
            "type": img.get("type", "output") or "output",
        }
        try:
            resp = requests.get(f"{COMFYUI_URL}/view", params=params, timeout=TIMEOUT)
            resp.raise_for_status()
        except Exception as e:
            glogger.warning(f"  view fetch failed for {filename}: {e}")
            continue
        out_path = _TMP_DIR / f"comfyui-it-{test_name}-{filename}"
        out_path.write_bytes(resp.content)
        saved.append(str(out_path))
        print(f"    saved {out_path} ({len(resp.content)} bytes)")
    return saved


def _upload_image(img: Image.Image, name: str) -> str:
    """Upload a PIL image to ComfyUI's input folder, return the server-side filename."""
    buf = BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    resp = requests.post(
        f"{COMFYUI_URL}/upload/image",
        files={"image": (name, buf, "image/png")},
        data={"overwrite": "true"},
        timeout=TIMEOUT,
    )
    resp.raise_for_status()
    return str(resp.json()["name"])


# At least one of these checkpoint filenames should be present in
# ``object_info`` once the launch script finished downloading models.
# Kept loose so the test still passes if the model list evolves.
EXPECTED_CHECKPOINT_SUBSTRINGS: list[str] = [
    "flux1-schnell",
    "RealVisXL",
]


class TestResult:
    """Result of a single integration test."""

    def __init__(self, name: str, passed: bool, duration: float, detail: str = "") -> None:
        self.name: str = name
        self.passed: bool = passed
        self.duration: float = duration
        self.detail: str = detail

    def __str__(self) -> str:
        status = "\033[32mPASS\033[0m" if self.passed else "\033[31mFAIL\033[0m"
        result = f"  [{status}] {self.name} ({self.duration:.2f}s)"
        if self.detail:
            result += f" — {self.detail}"
        return result


def test_health() -> TestResult:
    """Verify that the ComfyUI web UI is reachable."""
    t0 = time.monotonic()
    try:
        resp = requests.get(COMFYUI_URL, timeout=TIMEOUT)
        ok = resp.status_code == 200 and "ComfyUI" in resp.text
        return TestResult("health", ok, time.monotonic() - t0, f"status={resp.status_code}")
    except Exception as e:
        return TestResult("health", False, time.monotonic() - t0, str(e))


def test_system_stats() -> TestResult:
    """Verify that ``/system_stats`` returns Python + device info."""
    t0 = time.monotonic()
    try:
        resp = requests.get(f"{COMFYUI_URL}/system_stats", timeout=TIMEOUT)
        resp.raise_for_status()
        data = resp.json()
        devices = data.get("devices", [])
        system = data.get("system", {})
        ok = bool(devices) and "python_version" in system
        device_names = [d.get("name", "?") for d in devices]
        detail = f"python={system.get('python_version', '?').split()[0]}, devices={device_names}"
        return TestResult("system_stats", ok, time.monotonic() - t0, detail)
    except Exception as e:
        return TestResult("system_stats", False, time.monotonic() - t0, str(e))


def test_queue() -> TestResult:
    """Verify that ``/queue`` returns a well-formed queue state."""
    t0 = time.monotonic()
    try:
        resp = requests.get(f"{COMFYUI_URL}/queue", timeout=TIMEOUT)
        resp.raise_for_status()
        data = resp.json()
        ok = "queue_running" in data and "queue_pending" in data
        detail = f"running={len(data.get('queue_running', []))}, " f"pending={len(data.get('queue_pending', []))}"
        return TestResult("queue", ok, time.monotonic() - t0, detail)
    except Exception as e:
        return TestResult("queue", False, time.monotonic() - t0, str(e))


def test_object_info_checkpoints() -> TestResult:
    """Verify that ``/object_info`` exposes at least one expected checkpoint."""
    t0 = time.monotonic()
    try:
        resp = requests.get(f"{COMFYUI_URL}/object_info", timeout=TIMEOUT)
        resp.raise_for_status()
        data = resp.json()
        loader = data.get("CheckpointLoaderSimple", {})
        input_spec = loader.get("input", {}).get("required", {}).get("ckpt_name", [])
        checkpoints: list[str] = input_spec[0] if input_spec and isinstance(input_spec[0], list) else []
        matches = [c for c in checkpoints if any(s in c for s in EXPECTED_CHECKPOINT_SUBSTRINGS)]
        ok = len(matches) > 0
        detail = (
            f"{len(checkpoints)} checkpoints, matched {matches}"
            if ok
            else f"no expected checkpoint found (have: {checkpoints})"
        )
        return TestResult("object_info_checkpoints", ok, time.monotonic() - t0, detail)
    except Exception as e:
        return TestResult("object_info_checkpoints", False, time.monotonic() - t0, str(e))


def _submit_and_wait(workflow: dict[str, Any], test_name: str, t0: float) -> TestResult:
    """Queue a workflow, poll ``/history/<id>`` until completion, return a TestResult.

    Shared submission/polling path for all generation tests (inpaint + txt2img)
    so the queue/error-handling logic lives in exactly one place.
    """
    try:
        client_id = uuid4().hex
        resp = requests.post(
            f"{COMFYUI_URL}/prompt",
            json={"prompt": workflow, "client_id": client_id},
            timeout=TIMEOUT,
        )
        if resp.status_code >= 400:
            return TestResult(
                test_name, False, time.monotonic() - t0, f"queue rejected: {resp.status_code} {resp.text[:200]}"
            )
        prompt_id = resp.json()["prompt_id"]

        deadline = time.monotonic() + IMAGE_GEN_DEADLINE_S
        history: dict[str, Any] | None = None
        while time.monotonic() < deadline:
            hist_resp = requests.get(f"{COMFYUI_URL}/history/{prompt_id}", timeout=TIMEOUT)
            hist_resp.raise_for_status()
            entry = hist_resp.json().get(prompt_id)
            if entry is not None:
                status = entry.get("status", {}) or {}
                if status.get("status_str") == "error" or (status.get("completed") is False and status.get("messages")):
                    for mtype, mdata in status.get("messages", []):
                        if mtype == "execution_error":
                            return TestResult(test_name, False, time.monotonic() - t0, f"execution_error: {mdata}")
                if status.get("completed") or status.get("status_str") == "success":
                    history = entry
                    break
            time.sleep(2)

        if history is None:
            return TestResult(test_name, False, time.monotonic() - t0, f"timeout after {IMAGE_GEN_DEADLINE_S}s")

        outputs = history.get("outputs", {}) or {}
        images: list[dict[str, Any]] = []
        for node_out in outputs.values():
            images.extend(node_out.get("images", []) or [])
        ok = len(images) > 0
        saved = _download_images_to_tmp(images, test_name)
        if ok:
            detail = f"prompt_id={prompt_id[:8]}, saved={saved}"
        else:
            detail = f"no images in outputs={list(outputs)}"
        return TestResult(test_name, ok, time.monotonic() - t0, detail)
    except Exception as e:
        return TestResult(test_name, False, time.monotonic() - t0, str(e))


def _build_text2image_workflow(
    prompt: str,
    ckpt_name: str,
    *,
    steps: int,
    cfg: float,
    sampler_name: str,
    scheduler: str,
    width: int = 1024,
    height: int = 1024,
    negative: str = "",
    seed: int | None = None,
    flux_guidance: float | None = None,
) -> dict[str, Any]:
    """Minimaler txt2img-Graph: CheckpointLoader → CLIP/Empty-Latent → KSampler → VAEDecode → SaveImage.

    Funktioniert sowohl für SDXL-Single-File (RealVisXL) als auch für den
    FLUX-schnell-fp8-All-in-One-Checkpoint (Comfy-Org/flux1-schnell), weil
    beide UNet+CLIP+VAE bündeln und über CheckpointLoaderSimple geladen
    werden. Für FLUX-schnell ist cfg=1.0 + euler/simple Pflicht (Distill-
    Modell ohne klassisches CFG); SDXL nutzt die üblichen dpmpp_2m+karras.
    """
    if seed is None:
        seed = int.from_bytes(os.urandom(4), "big")
    workflow: dict[str, Any] = {
        "1": {"inputs": {"ckpt_name": ckpt_name}, "class_type": "CheckpointLoaderSimple"},
        "2": {"inputs": {"text": prompt, "clip": ["1", 1]}, "class_type": "CLIPTextEncode"},
        "3": {"inputs": {"text": negative, "clip": ["1", 1]}, "class_type": "CLIPTextEncode"},
        "4": {
            "inputs": {"width": width, "height": height, "batch_size": 1},
            "class_type": "EmptyLatentImage",
        },
        "5": {
            "inputs": {
                "seed": seed,
                "steps": steps,
                "cfg": cfg,
                "sampler_name": sampler_name,
                "scheduler": scheduler,
                "denoise": 1.0,
                "model": ["1", 0],
                "positive": ["2", 0],
                "negative": ["3", 0],
                "latent_image": ["4", 0],
            },
            "class_type": "KSampler",
        },
        "6": {"inputs": {"samples": ["5", 0], "vae": ["1", 2]}, "class_type": "VAEDecode"},
        "7": {
            "inputs": {"filename_prefix": "comfyui-it-text2image", "images": ["6", 0]},
            "class_type": "SaveImage",
        },
    }
    if flux_guidance is not None:
        # FLUX [dev]: distillierte Guidance separat als Conditioning-Wrapper
        # vor dem KSampler. KSampler-cfg bleibt 1.0 (echtes CFG aus).
        workflow["8"] = {
            "inputs": {"guidance": flux_guidance, "conditioning": ["2", 0]},
            "class_type": "FluxGuidance",
        }
        workflow["5"]["inputs"]["positive"] = ["8", 0]
    return workflow


def test_image_generation() -> TestResult:
    """Submit the Fooocus-inpaint workflow end-to-end and verify an output image is produced."""
    t0 = time.monotonic()
    try:
        # Input: horizontal color gradient so the sampler has something non-trivial to denoise.
        img = Image.new("RGB", (1024, 1024), (0, 0, 0))
        draw = ImageDraw.Draw(img)
        for x in range(1024):
            draw.line([(x, 0), (x, 1024)], fill=(x // 4, 120, 255 - x // 4))

        # Mask: red disc in the center (LoadImageMask reads the red channel).
        mask = Image.new("RGB", (1024, 1024), (0, 0, 0))
        ImageDraw.Draw(mask).ellipse([(320, 320), (704, 704)], fill=(255, 0, 0))

        token = uuid4().hex[:8]
        uploaded_img = _upload_image(img, f"it-input-{token}.png")
        uploaded_mask = _upload_image(mask, f"it-mask-{token}.png")

        workflow = copy.deepcopy(INPAINT_WORKFLOW)
        workflow["8"]["inputs"]["image"] = uploaded_img
        workflow["9"]["inputs"]["image"] = uploaded_mask
        workflow["3"]["inputs"]["seed"] = int.from_bytes(os.urandom(4), "big")
    except Exception as e:
        return TestResult("image_generation", False, time.monotonic() - t0, str(e))
    return _submit_and_wait(workflow, "image_generation", t0)


def test_text2image_realvisxl(prompt: str = DEFAULT_PROMPT) -> TestResult:
    """Generate an image from ``prompt`` with RealVisXL_V5.0 (SDXL)."""
    t0 = time.monotonic()
    workflow = _build_text2image_workflow(
        prompt,
        "RealVisXL_V5.0_fp16.safetensors",
        negative="blurry, low quality, watermark, text, bad anatomy, deformed",
        **SDXL_SAMPLER,
    )
    return _submit_and_wait(workflow, "text2image_realvisxl", t0)


def test_text2image_flux_schnell(prompt: str = DEFAULT_PROMPT) -> TestResult:
    """Generate an image from ``prompt`` with FLUX.1-schnell (fp8 all-in-one)."""
    t0 = time.monotonic()
    workflow = _build_text2image_workflow(
        prompt,
        "flux1-schnell-fp8.safetensors",
        **FLUX_SCHNELL_SAMPLER,
    )
    return _submit_and_wait(workflow, "text2image_flux_schnell", t0)


def test_text2image_flux_dev(prompt: str = DEFAULT_PROMPT) -> TestResult:
    """Generate an image from ``prompt`` with FLUX.1 [dev] (fp8 all-in-one)."""
    t0 = time.monotonic()
    workflow = _build_text2image_workflow(
        prompt,
        "flux1-dev-fp8.safetensors",
        flux_guidance=FLUX_DEV_GUIDANCE,
        **FLUX_DEV_SAMPLER,
    )
    return _submit_and_wait(workflow, "text2image_flux_dev", t0)


def main() -> None:
    """Run all ComfyUI integration tests and exit with an appropriate status code."""
    print_banner(module=Path(__file__).stem)
    print(f"ComfyUI integration tests — {COMFYUI_URL}\n")

    tests: list[Callable[[], TestResult]] = [
        test_health,
        test_system_stats,
        test_queue,
        test_object_info_checkpoints,
        test_image_generation,
        test_text2image_realvisxl,
        test_text2image_flux_schnell,
        test_text2image_flux_dev,
    ]

    results: list[TestResult] = []
    for test_fn in tests:
        result = test_fn()
        print(result)
        results.append(result)

    passed = sum(1 for r in results if r.passed)
    total = len(results)
    total_time = sum(r.duration for r in results)
    print(f"\n{passed}/{total} passed in {total_time:.1f}s")
    sys.exit(0 if passed == total else 1)


if __name__ == "__main__":
    main()
