#!/bin/sh
# Install the patched LookAtMyQuant (lamq) MLX runtime into $INKLING_MAC_HOME/venv (default ~/.local/share/inkling-mac).
# Pins lookatmyquant 6170110 (M5 Split-K SDPA, FP8/Q4 global KV, zero-copy DSpark), applies lamq.patch
# (runtime refusal ablation in model.py + batch.py, PackedEmbed.inject for vision, 8192-step KV reserve).
set -eu
COMMIT=6170110d7f564f2abc825034d6dbb41894206e17
HERE=$(cd "$(dirname "$0")" && pwd)
DEST="${INKLING_MAC_HOME:-$HOME/.local/share/inkling-mac}"

command -v uv >/dev/null || { echo "uv is required: https://docs.astral.sh/uv/" >&2; exit 1; }
[ "$(uname -sm)" = "Darwin arm64" ] || { echo "Apple Silicon macOS only" >&2; exit 1; }

tmp=$(mktemp -d); trap 'rm -rf "$tmp"' EXIT
git -c advice.detachedHead=false clone -q --filter=blob:none https://github.com/ShinMegamiBoson/lookatmyquant "$tmp/src"
git -C "$tmp/src" -c advice.detachedHead=false checkout -q "$COMMIT"
git -C "$tmp/src" apply "$HERE/lamq.patch"

rm -rf "$DEST/venv"; mkdir -p "$DEST"
uv venv -q --python 3.13 "$DEST/venv"
uv pip install -q --python "$DEST/venv/bin/python" "$tmp/src" gguf==0.19.0 pillow==13.0.0
printf '%s\n' "$COMMIT" > "$DEST/LAMQ_COMMIT"
"$DEST/venv/bin/python" -c "import mlx.core as mx, runtime_mlx.inkling.model as m, runtime_mlx.inkling.batch as b; \
assert hasattr(m.Inkling2LoopPackedMLX, 'set_ablation'); print('lamq $COMMIT'[:12], 'mlx', mx.__version__, '->', '$DEST/venv')"
