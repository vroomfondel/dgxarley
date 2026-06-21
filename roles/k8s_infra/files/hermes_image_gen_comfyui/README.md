# Hermes image_gen backend: local ComfyUI

A custom Hermes `ImageGenProvider` that routes the `image_generate` tool to our
in-cluster **ComfyUI** (FLUX.1-schnell) instead of a cloud provider
(FAL / OpenAI / xAI). Written against the `agent.image_gen_provider` interface
of `nousresearch/hermes-agent` (pinned tag in `hermes_image_tag`).

## STATUS: created but DISABLED

These files are **inert** — nothing is wired into the Hermes Deployments
(no ConfigMap, no mount) and the provider is **not** in `plugins.enabled`.
So the running cluster is unaffected. The plugin only activates after the two
explicit steps in **Enabling** below. By design (user plugins are opt-in).

## Files

| File | Purpose |
|------|---------|
| `plugin.yaml` | Plugin manifest (`kind: backend`, `name: comfyui`, no required env). |
| `__init__.py` | `ComfyUIImageGenProvider` + `register(ctx)` entry point. |

## How it works

Text-to-image only (v1). On each `image_generate` call:
1. `POST {url}/prompt` with a FLUX.1-schnell graph (API format) — 4 steps, cfg 1.0,
   euler/simple, all-in-one fp8 checkpoint `flux1-schnell-fp8.safetensors`.
2. Poll `GET {url}/history/{prompt_id}` until outputs appear (timeout configurable).
3. `GET {url}/view?...` for the first image and cache the bytes under
   `$HERMES_HOME/cache/images/` (so chat / Telegram / email get a stable path).

Server URL resolution (first hit wins):
1. `COMFYUI_URL` env var
2. `image_gen.comfyui.url` in `config.yaml`
3. `http://127.0.0.1:8188` (default)

Other `image_gen.comfyui.*` config keys: `checkpoint` (default
`flux1-schnell-fp8.safetensors`), `steps` (default `4`), `timeout` (default `300`).

## Enabling (when you actually want it)

> Prerequisite: validate ComfyUI itself works first — it runs on ARM64/SM121
> (spark4) and has documented generation issues (`COMFYUI_PROMPT_FAIL.md`,
> `UPSTREAM_PYTORCH_SDPA_SM121.md`). Smoke-test a bare workflow against
> `comfyui.comfyui.svc.cluster.local:8188` before pointing Hermes at it.

1. **Ship the plugin dir** into each user's Hermes home at
   `$HERMES_HOME/plugins/image_gen/comfyui/` (= `/opt/data/plugins/image_gen/comfyui/`,
   the per-user NFS subdir hostPath). Not yet automated — to wire it into the
   playbook, mount these two files via a ConfigMap + copy them in the
   `merge-config` initContainer (mirror the `hermes-merge-config` pattern), or
   drop them in manually for a single user to try.
2. **Enable + select** in the per-user `config.yaml`
   (`roles/k8s_infra/templates/hermes/hermes_config.yaml.j2`):
   ```yaml
   plugins:
     enabled:
       - comfyui
   image_gen:
     provider: comfyui
     comfyui:
       url: "http://comfyui.comfyui.svc.cluster.local:8188"
       # checkpoint: "flux1-schnell-fp8.safetensors"
       # steps: 4
       # timeout: 300
   ```
   (Or set `COMFYUI_URL` in the per-user `.env` instead of `comfyui.url`.)
3. Roll the Hermes pods. The `image_generate` tool now renders via local ComfyUI.

Cross-namespace `hermes` → `comfyui` over ClusterIP DNS needs no extra wiring.

## Re-sync note

`agent.image_gen_provider` is upstream API. If you bump `hermes_image_tag`,
re-check the base class (`generate()` signature, `success_response` /
`save_url_image` helpers, `register(ctx)` contract) against the new tag and
re-apply any drift here. Reference siblings: `plugins/image_gen/{openai,xai,fal}`
in the hermes-agent repo.
