#!/bin/sh
# Download the model files into the Hugging Face cache, at the revisions this repo was measured with.
#   scripts/download-models.sh          lamq 8.9GB (8.3 GiB) + vision projector (0.6 GiB)
#   scripts/download-models.sh --llama  also AtomicChat's AD-IQ3_S GGUF (13.3 GB) for the llama.cpp backend
# Uses the hf CLI from the venv that lamq/install.sh creates. Xet high-performance mode with 16 fixed connections (32 was
# no faster on a ~50 Mbps link): the adaptive default
# tends to settle at low concurrency on home connections.
set -eu
HF="${INKLING_MAC_HOME:-$HOME/.local/share/inkling-mac}/venv/bin/hf"
[ -x "$HF" ] || { echo "run lamq/install.sh first" >&2; exit 1; }
export HF_XET_HIGH_PERFORMANCE=1 HF_XET_FIXED_DOWNLOAD_CONCURRENCY=16

"$HF" download shinmegamiboson/Inkling-2-Mini-LookAtMyQuant-8.9GB --revision 1df3390e09eae0f4a93d4c05db80d1f054da7dbd
AC=AtomicChat/Inkling-2-Mini-Uncensored-GGUF
AC_REV=40886ab636cabb4decf4998224f790b462ab048e
"$HF" download "$AC" --revision "$AC_REV" --include mmproj-Inkling-2-Mini-Uncensored-F16.gguf
if [ "${1:-}" = "--llama" ]; then
  "$HF" download "$AC" --revision "$AC_REV" --include Inkling-2-Mini-Uncensored-AD-IQ3_S.gguf
fi
