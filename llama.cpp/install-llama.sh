#!/bin/sh
# Build a pinned upstream llama.cpp (ggml-org) from source, plus unmerged PR commits, into
# $INKLING_MAC_HOME/llama.cpp/<name> with `current` pointing at it. llama.cpp/router uses current/llama-server.
#
# Why build from source: Inkling 2's inkling2_loop architecture (#32694) exists from b12600, and PR #32752 (chunked
# Inkling 2 attention-mask construction) isn't merged yet. Measured on M5 Pro (IQ3_S, ub128, KV q8_0): 80K compute
# buffer 2064 -> 1595 MiB; prefill with 16K/32K already in context +14% / +21%; decode and PPL unchanged.
# When the PR lands in a release: set TAG to that release and empty PICKS.
# cmake is pulled through uvx, nothing is installed system-wide. Build takes about a minute with -j18.
set -eu
TAG=b12628
TAG_SHA=8a97fa98ab1a454709bbf5ac979661feaeadec40
PR=32752
PICKS="37608cd0377ff226cde1765ee67a4eee46dbb419 56a4ac9eec559c295e84eb4af8730e7a1a889199"
DEST="${INKLING_MAC_HOME:-$HOME/.local/share/inkling-mac}/llama.cpp"
NAME="$TAG${PICKS:+-pr$PR}"
JOBS=$(sysctl -n hw.ncpu)

tmp=$(mktemp -d); trap 'rm -rf "$tmp"' EXIT
git -c advice.detachedHead=false clone -q --depth 1 --branch "$TAG" https://github.com/ggml-org/llama.cpp "$tmp/src"
cd "$tmp/src"
[ "$(git rev-parse HEAD)" = "$TAG_SHA" ] || { echo "tag $TAG is not the expected $TAG_SHA" >&2; exit 1; }
if [ -n "$PICKS" ]; then
  git fetch -q --depth 3 origin "pull/$PR/head"
  for c in $PICKS; do
    git -c user.name=local -c user.email=local@localhost cherry-pick "$c" >/dev/null
  done
fi
# LLAMA_USE_PREBUILT_UI=OFF: don't download the web UI at build time; only the API is used
log="$tmp/build.log"
{ uvx --from cmake cmake -S . -B build -G "Unix Makefiles" -DCMAKE_BUILD_TYPE=Release \
    -DGGML_METAL=ON -DGGML_METAL_EMBED_LIBRARY=ON -DBUILD_SHARED_LIBS=OFF -DLLAMA_CURL=OFF \
    -DLLAMA_USE_PREBUILT_UI=OFF -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF &&
  uvx --from cmake cmake --build build -j"$JOBS" \
    --target llama-server llama-bench llama-perplexity llama-fit-params; } > "$log" 2>&1 \
  || { tail -30 "$log" >&2; exit 1; }

rm -rf "${DEST:?}/$NAME"; mkdir -p "$DEST/$NAME"
cp build/bin/llama-server build/bin/llama-bench build/bin/llama-perplexity build/bin/llama-fit-params "$DEST/$NAME/"
ln -sfn "$DEST/$NAME" "$DEST/current"
# keep only the version `current` points at
for d in "$DEST"/*/; do d=${d%/}; [ "$d" = "$DEST/$NAME" ] || [ -L "$d" ] || rm -rf "$d"; done
"$DEST/current/llama-server" --version 2>&1 | head -1
