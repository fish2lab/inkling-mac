#!/bin/sh
# Render docs/figures/src/*.html to docs/figures/*.png at 2x with a headless Chromium (any Chrome-family binary).
#   CHROME=/path/to/chrome docs/figures/render.sh [name ...]
# Fonts come from Google Fonts, so rendering needs network access.
set -eu
HERE=$(cd "$(dirname "$0")" && pwd)
CHROME="${CHROME:-/Applications/Google Chrome.app/Contents/MacOS/Google Chrome}"
PY="${PY:-${INKLING_MAC_HOME:-$HOME/.local/share/inkling-mac}/venv/bin/python}"   # any Python with Pillow
prof=$(mktemp -d); trap 'rm -rf "$prof"' EXIT
names="${*:-overview decode prefill-memory splitk ablation}"
for n in $names; do
  h=$(sed -n 's/.*class="fig" style="height:\([0-9]*\)px.*/\1/p' "$HERE/src/$n.html" | head -1)
  "$CHROME" --headless=new --disable-gpu --user-data-dir="$prof" --no-first-run --hide-scrollbars \
    --force-device-scale-factor=2 --window-size=1400,$((h + 300)) --virtual-time-budget=8000 \
    --screenshot="$HERE/$n.png" "file://$HERE/src/$n.html" >/dev/null 2>&1
  # headless viewports are smaller than the window by an inconsistent margin: render oversized, crop to the figure
  "$PY" -c "import sys; from PIL import Image; p = sys.argv[1]; Image.open(p).crop((0, 0, 2400, int(sys.argv[2]) * 2)).save(p, optimize=True)" "$HERE/$n.png" "$h"
  echo "$n.png  1200x$h @2x"
done
