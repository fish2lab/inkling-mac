#!/bin/sh
# Install the patched LookAtMyQuant (lamq) MLX runtime into $INKLING_MAC_HOME/venv (default ~/.local/share/inkling-mac).
# Pins a lookatmyquant commit, applies lamq.patch, and installs it non-editable into a uv venv; the source is thrown
# away afterwards. Needs git and uv (https://docs.astral.sh/uv/).
#
# What lamq.patch changes (see README for the measurements):
#   model.py         ablate(): y - b(b^T y) after o_proj, after every expert's down_proj and after the embedding
#                    lookup, all before the post-conv; Inkling2LoopPackedMLX.set_ablation(dirs) attaches the
#                    directions. PackedEmbed.inject: replace image placeholder tokens after embed_norm.
#   fast_kernels.py  split-K GQA decode attention (T < 32): one threadgroup per (KV head, key segment), the 3 query
#                    heads that share a KV head reuse each K/V load, and a second kernel merges the segments with
#                    log-sum-exp plus the attention sink. LAMQ_SPLITK=0 falls back to the stock kernel.
# To move to a newer lookatmyquant: change COMMIT; if the patch no longer applies, re-hook ablate/inject at the same spots.
set -eu
COMMIT=ae0ca83ed43f0b506384a220fc62903d60ca831b
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
uv pip install -q --python "$DEST/venv/bin/python" "$tmp/src" \
  mlx==0.32.4 mlx-lm==0.32.2 gguf==0.19.0 pillow==13.0.0
printf '%s\n' "$COMMIT" > "$DEST/LAMQ_COMMIT"
"$DEST/venv/bin/python" -c "import mlx.core as mx, runtime_mlx.inkling.fast_kernels as f, runtime_mlx.inkling.model as m; \
assert hasattr(m.Inkling2LoopPackedMLX, 'set_ablation') and f.SPLITK_ON; print('lamq $COMMIT'[:12], 'mlx', mx.__version__, '->', '$DEST/venv')"
