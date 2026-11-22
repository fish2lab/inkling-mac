# Third-party notices

This repository contains code written for it (MIT, see `LICENSE`) plus derived material from the projects below.
No model weights are redistributed; `scripts/download-models.sh` fetches them from their original repositories.

| Component | Where it is used here | Upstream | License |
|---|---|---|---|
| lookatmyquant (Shin Megami Boson) | `lamq/lamq.patch` is a patch against its MLX runtime; `lamq/install.sh` downloads the pinned commit | https://github.com/ShinMegamiBoson/lookatmyquant | MIT |
| mlx-vlm (Prince Canuma) | `lamq/inkling_vision.py` ports the Inkling 2 vision tower from `mlx_vlm/models/inkling2/vision.py` (commit f1618d6d) | https://github.com/Blaizzy/mlx-vlm | MIT |
| Inkling-2-Mini-Uncensored-GGUF (AtomicChat) | `lamq/ablate-dirs.npz` holds the `lora_b` vectors of their rank-1 refusal-ablation LoRA, extracted with `lamq/extract-dirs.py` and renormalized; the vision projector is downloaded from the same repository | https://huggingface.co/AtomicChat/Inkling-2-Mini-Uncensored-GGUF | Apache-2.0 |
| llama.cpp (ggml-org) | `llama.cpp/install-llama.sh` builds a pinned release plus commits from PR #32752; `models.ini.in` is a router preset | https://github.com/ggml-org/llama.cpp | MIT |
| Inkling 2 Mini | base model of everything above | see the model card of the weights you download | see model card |

Modifications to Apache-2.0 material: `ablate-dirs.npz` keeps only the per-layer `lora_b` direction vectors from
AtomicChat's LoRA GGUF (the `lora_a` factors are discarded; they equal `-W^T b` and are recomputed implicitly by
projecting the direction out at run time).
