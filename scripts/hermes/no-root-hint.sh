#!/bin/sh
# Installed as apt-get/apt/sudo in /usr/local/sbin (ahead of /usr/bin in PATH); root and read-only calls pass through.
name="$(basename "$0")"

if [ "$(id -u)" = "0" ]; then
    [ "$name" = "sudo" ] && exec "$@"
    exec "/usr/bin/$name" "$@"
fi

if [ "$name" != "sudo" ]; then
    mutating=0
    for a in "$@"; do
        case "$a" in
            install|reinstall|remove|purge|upgrade|full-upgrade|dist-upgrade|update|autoremove|autopurge|build-dep|satisfy)
                mutating=1; break ;;
        esac
    done
    [ "$mutating" = "0" ] && exec "/usr/bin/$name" "$@"
fi

cat >&2 <<EOF
$name: not available. This container runs without root and has no sudo; system packages cannot be installed and
the root filesystem resets on every restart.

Already installed: tesseract (deu, eng, osd), ocrmypdf, unpaper, poppler-utils (pdftotext, pdftoppm, pdfimages),
qpdf, ghostscript, imagemagick (convert, magick), pandoc, ffmpeg, jq, file, zip/unzip, 7z.

For other CLI tools, install rootless into the persistent /opt/data (survives restarts, binaries land on PATH):
  micromamba install -y -c conda-forge <package>    # e.g. exiftool, graphviz, sqlite, gdal
  pip install <package>                             # Python packages, importable in execute_code right away
EOF
exit 100
