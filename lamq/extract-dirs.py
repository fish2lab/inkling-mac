"""Extract the per-layer refusal directions from AtomicChat's rank-1 LoRA GGUF into an npz (layers [24, 2304], embd [2304]).

Usage: python extract-dirs.py <lora.gguf> <out.npz>
The LoRA comes from https://huggingface.co/AtomicChat/Inkling-2-Mini-Uncensored-GGUF (Apache-2.0).
Within a layer, attn_output and all 128 experts' ffn_down share one unit lora_b vector, and lora_a = -W^T b, i.e. the
edit is a full directional ablation W' = (I - b b^T) W. That's why only lora_b is kept here.
"""
import sys, numpy as np
from gguf import GGUFReader
r = GGUFReader(sys.argv[1]); T = {t.name: np.array(t.data, dtype=np.float32) for t in r.tensors}
B = np.stack([T[f"blk.{l}.attn_output.weight.lora_b"].reshape(-1) for l in range(24)])
emb = T["token_embd.weight.lora_b"].reshape(-1)
np.savez(sys.argv[2], layers=B, embd=emb)
C = np.abs(B @ B.T)
print("adjacent-layer |cos|:", np.round([C[i, i+1] for i in range(23)], 2))
print("|cos| to emb:", np.round(np.abs(B @ emb), 2))
for f in r.fields.values():
    if f.name.startswith(("adapter", "general")): 
        try: print(f.name, f.contents())
        except Exception: pass
