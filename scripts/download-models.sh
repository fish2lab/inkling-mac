#!/bin/sh
# Download the model files into the Hugging Face cache, at the revisions measured in this repo.
#   scripts/download-models.sh          lamq 11.2GB v5 + 4-bit DSpark drafter + vision projector
#   scripts/download-models.sh --splash also AtomicChat AD-IQ3_S GGUF + DFlash2 draft + BF16 mmproj for Splash
#   scripts/download-models.sh --llama  also AtomicChat AD-IQ3_S GGUF for llama.cpp
set -eu
HF="${INKLING_MAC_HOME:-$HOME/.local/share/inkling-mac}/venv/bin/hf"
[ -x "$HF" ] || HF=$(command -v hf)
export HF_XET_HIGH_PERFORMANCE=1 HF_XET_FIXED_DOWNLOAD_CONCURRENCY=16

"$HF" download shinmegamiboson/Inkling-2-Mini-LookAtMyQuant-11.2GB --revision 7191658901060d7c80871eb27413010c3e563b8c
"$HF" download shinmegamiboson/Inkling-2-Mini-LookAtMyQuant-drafter --revision 04479ce728ca19efd8e16eaafc704742c8a858da
AC=AtomicChat/Inkling-2-Mini-Uncensored-GGUF
"$HF" download "$AC" --revision 40886ab636cabb4decf4998224f790b462ab048e --include mmproj-Inkling-2-Mini-Uncensored-F16.gguf
if [ "${1:-}" = "--splash" ] || [ "${1:-}" = "--llama" ]; then
  "$HF" download "$AC" --revision c5667da36fe35da026fbee394d00547a921e98ab --include Inkling-2-Mini-Uncensored-AD-IQ3_S.gguf --include mmproj-Inkling-2-Mini-Uncensored-BF16.gguf
fi
if [ "${1:-}" = "--splash" ]; then
  "$HF" download incoai/Inkling-2-Mini-DFlash2 --revision 4f9088ea66c21f2b99ed657d1941c306b03e84d2
fi
