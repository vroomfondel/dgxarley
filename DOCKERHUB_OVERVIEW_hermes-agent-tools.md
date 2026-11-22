<!-- short: nousresearch/hermes-agent + OCR/PDF/document CLI tools and rootless pip/micromamba. amd64+arm64 -->

# hermes-agent-tools

**[`nousresearch/hermes-agent`](https://hub.docker.com/r/nousresearch/hermes-agent) with the command-line tools
the agent keeps reaching for already installed**, plus a way to add more packages **without root**.

The upstream image runs the agent's `terminal` and `execute_code` tools as the unprivileged `hermes` user (no
`sudo`), and the root filesystem resets on every restart. When the agent needs `tesseract` to OCR a scanned PDF,
`apt-get install` fails on the dpkg lock, and it falls back to slow vision-model calls page by page. This image
bakes in the usual tools and gives the agent a rootless install path that survives restarts.

- **Source / build script**: [github.com/vroomfondel/dgxarley](https://github.com/vroomfondel/dgxarley)
  (see [`scripts/hermes/`](https://github.com/vroomfondel/dgxarley/tree/main/scripts/hermes) and
  [`repo_scripts/build-hermes-image.sh`](https://github.com/vroomfondel/dgxarley/blob/main/repo_scripts/build-hermes-image.sh))
- **Platforms**: `linux/amd64`, `linux/arm64` (multi-arch manifest)
- **License**: the upstream licenses of `nousresearch/hermes-agent`, the Debian packages and micromamba

## What's inside

On top of the unchanged upstream image (same `ENTRYPOINT`, s6-overlay, UID remap via `HERMES_UID`/`HERMES_GID`):

- **OCR**: `tesseract-ocr` with `deu`, `eng`, `osd`; `ocrmypdf`, `unpaper`, `pngquant`
- **PDF**: `poppler-utils` (`pdftotext`, `pdftoppm`, `pdfimages`, ...), `qpdf`, `ghostscript`
- **Images / documents**: `imagemagick`, `pandoc`
- **Misc**: `jq`, `file`, `zip`/`unzip`, `p7zip-full` (`ffmpeg` already ships upstream)
- **`micromamba`** (pinned release, sha256-checked) for further CLI tools from conda-forge
- **`pip`** for further Python packages, importable from the agent's own Python

## Installing more at runtime (no root)

Everything lands on the persistent `HERMES_HOME` volume (`/opt/data`), so it survives pod restarts and stays per
user.

```sh
micromamba install -y -c conda-forge exiftool   # CLI tools -> /opt/data/.micromamba, already on PATH
pip install some-package                        # Python   -> /opt/data/.local/pip/python3.X
```

- `pip` is a small wrapper around the `uv` that ships in the base image (`uv pip install --target`). A `.pth` hook
  appends the target to `sys.path` of the Hermes venv, so packages are importable in `execute_code` right away.
  Console scripts are linked into `/opt/data/.local/bin`.
- Packages that Hermes itself ships keep their version: the install runs against a constraints file frozen at build
  time and fails with a resolver error on a conflict instead of shadowing Hermes' own dependencies. The target is
  appended **after** the venv's site-packages for the same reason.
- `pip list`, `freeze`, `show` and `uninstall` work; `--user`, `-y` and similar pip-only flags are ignored.
  `python -m pip` is not available, use `pip`.
- The target directory contains the Python minor version, so a base image with a newer Python starts with a clean
  one instead of loading incompatible wheels.
- `apt-get`/`apt install`, `update`, `remove` ... and `sudo` print a short hint (what is preinstalled, how to use
  micromamba/pip) and exit with 100, instead of failing on the dpkg lock. Root and read-only calls such as
  `apt list --installed` pass through unchanged.

## How to use

Swap the image name and keep the tag:

```yaml
image: xomoxcc/hermes-agent-tools:v0.23.1   # instead of nousresearch/hermes-agent:v0.23.1
```

Use it for every container that runs agent sessions (dashboard, gateway). Nothing else changes: same environment
variables, same volumes, same arguments.

## Tags

| Tag       | Notes                                                    |
|-----------|----------------------------------------------------------|
| `v0.23.1` | On `nousresearch/hermes-agent:v0.23.1`, amd64 + arm64    |
| `latest`  | Most recent build, same as the newest version tag        |

The tag follows the upstream tag it layers on; each upstream release needs its own rebuild. Pin a version tag in
deployments rather than `latest`.

## Status / support

Built for and run on a private k3s cluster. Published in case it is useful to someone else. No commercial support;
tags may be retagged or removed without notice. Open an issue on the GitHub repo if something is broken.
