# syntax=docker/dockerfile:1.7
#
# xomoxcc/hermes-agent-tools:<tag> — nousresearch/hermes-agent:<tag> plus the CLI tools the agent's terminal /
# execute_code reach for (OCR, PDF, images, documents), and rootless package paths (micromamba, pip) for the rest.
#
# The agent runs as the unprivileged `hermes` user (s6-setuidgid), so it can never apt-get at runtime; whatever it
# needs routinely has to be in the image; micromamba and pip install into the persistent /opt/data.
#
# Image contract of the base is kept untouched: no USER/ENTRYPOINT/CMD here, only additional files and packages.
# Tag = the base tag, so hermes.image_tag drives both images. Build: repo_scripts/build-hermes-image.sh (amd64 + arm64 manifest).

ARG BASE_TAG=v0.23.1
FROM docker.io/nousresearch/hermes-agent:${BASE_TAG}

ARG TARGETARCH
ARG MICROMAMBA_VERSION=2.10.0-0
ARG MICROMAMBA_SHA256_AMD64=701f39c06f8e6a670a0e186c8479b9e5c0502a2f0cda6cef65d83bdd2875e8bd
ARG MICROMAMBA_SHA256_ARM64=3bb56786e21b9b24024f599fe9185b4a59dc3f9f59937a42311ee583adbbb23c

USER root

RUN apt-get -o Acquire::Retries=3 update && \
    apt-get -o Acquire::Retries=3 install -y --no-install-recommends \
        tesseract-ocr tesseract-ocr-deu tesseract-ocr-eng tesseract-ocr-osd \
        ocrmypdf unpaper pngquant \
        poppler-utils qpdf ghostscript \
        imagemagick \
        pandoc \
        jq file unzip zip p7zip-full && \
    rm -rf /var/lib/apt/lists/*

RUN set -eu; \
    case "${TARGETARCH:-amd64}" in \
      amd64) arch=linux-64;      sha="${MICROMAMBA_SHA256_AMD64}" ;; \
      arm64) arch=linux-aarch64; sha="${MICROMAMBA_SHA256_ARM64}" ;; \
      *) echo "unsupported arch ${TARGETARCH}" >&2; exit 1 ;; \
    esac; \
    test -n "$sha" || { echo "MICROMAMBA_SHA256 for ${TARGETARCH} not set" >&2; exit 1; }; \
    curl -fsSL --retry 3 -o /usr/local/libexec/micromamba \
      "https://github.com/mamba-org/micromamba-releases/releases/download/${MICROMAMBA_VERSION}/micromamba-${arch}"; \
    echo "${sha}  /usr/local/libexec/micromamba" | sha256sum -c -; \
    chmod 0755 /usr/local/libexec/micromamba

COPY --chmod=0755 micromamba-wrapper.sh /usr/local/bin/micromamba
COPY --chmod=0755 no-root-hint.sh /usr/local/sbin/apt-get
COPY --chmod=0755 no-root-hint.sh /usr/local/sbin/apt
COPY --chmod=0755 no-root-hint.sh /usr/local/sbin/sudo

COPY --chmod=0755 pip-wrapper.sh /usr/local/sbin/pip
COPY --chmod=0755 pip-wrapper.sh /usr/local/sbin/pip3
COPY --chmod=0644 hermes_user_site.py /tmp/hermes_user_site.py
RUN set -eu; \
    uv="$(ls /opt/hermes/tools/uv-*/uv)"; \
    test "$(printf '%s\n' "$uv" | wc -l)" = 1; \
    ln -s "$uv" /usr/local/libexec/uv; \
    site="$(/opt/hermes/.venv/bin/python -c 'import sysconfig; print(sysconfig.get_path("purelib"))')"; \
    mv /tmp/hermes_user_site.py "$site/hermes_user_site.py"; \
    echo "import hermes_user_site" > "$site/zz_hermes_user_site.pth"; \
    mkdir -p /usr/local/share/hermes-pip; \
    /usr/local/libexec/uv pip freeze --python /opt/hermes/.venv/bin/python \
      | grep -v -E '^(-e |#)|@ file:' > /usr/local/share/hermes-pip/constraints.txt

ENV MAMBA_ROOT_PREFIX=/opt/data/.micromamba
ENV PATH="${PATH}:/opt/data/.micromamba/bin"

RUN tesseract --list-langs && ocrmypdf --version && pdftoppm -v && micromamba --version && \
    /opt/hermes/.venv/bin/python -c 'import hermes_user_site' && test -s /usr/local/share/hermes-pip/constraints.txt
