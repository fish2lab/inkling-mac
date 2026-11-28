"""Build a rank-1 refusal-ablation LoRA that is exact for one specific quantized GGUF.

  python make-exact-lora.py <base.gguf> <atomicchat-lora.gguf> <out.gguf>

AtomicChat's LoRA encodes W' = (I - b b^T) W as lora_b = b, lora_a = -b^T W, with W their unquantized weights. Applied
to another quantization Q(W) it leaves b^T (Q(W) - W) x along the refusal direction. This script keeps their unit
directions b (one per layer, plus one for the embedding) and recomputes lora_a = -b^T Q(W) from the dequantized base
tensors, so llama.cpp's runtime LoRA removes b exactly from what the quantized attn_output, every expert's
ffn_down_exps and the token embedding write. Peak RAM is one dequantized tensor (~1.4 GB for a layer's experts).
"""
import sys

import numpy as np
from gguf import GGUFReader, GGUFWriter
from gguf.quants import dequantize

base_path, lora_path, out_path = sys.argv[1:4]
base = {t.name: t for t in GGUFReader(base_path).tensors}
lora = GGUFReader(lora_path)
arch = next(f.contents() for f in lora.fields.values() if f.name == "general.architecture")

w = GGUFWriter(out_path, arch)
w.add_type("adapter")
w.add_string("adapter.type", "lora")
w.add_float32("adapter.lora_alpha", 1.0)
w.add_name("inkling-mac exact refusal ablation")

pairs = sorted({t.name[: -len(".lora_a")] for t in lora.tensors if t.name.endswith(".lora_a")})
src = {t.name: t for t in lora.tensors}
for name in pairs:
    b_raw = np.array(src[name + ".lora_b"].data, dtype=np.float32)
    W = dequantize(base[name].data, base[name].tensor_type).astype(np.float32)  # numpy: [.., out, in] / embd [vocab, H]
    if name == "token_embd.weight":
        b = b_raw.reshape(-1); b /= np.linalg.norm(b)                    # lora_b numpy (2304, 1)
        a = -(W @ b)[:, None]                                            # (vocab, 1): get_rows gives -b^T e_t
        w.add_tensor(name + ".lora_a", a.astype(np.float32).reshape(src[name + ".lora_a"].data.shape))
        w.add_tensor(name + ".lora_b", b[:, None].astype(np.float32))
        old = np.array(src[name + ".lora_a"].data, dtype=np.float32).reshape(-1)
    elif W.ndim == 3:  # experts: numpy (E, out, in), lora_b (E, out, 1), lora_a (E, 1, in)
        b = b_raw[0, :, 0] if b_raw.ndim == 3 else b_raw.reshape(-1)
        b = b / np.linalg.norm(b)
        a = -np.einsum("o,eoi->ei", b, W)[:, None, :]
        w.add_tensor(name + ".lora_a", a.astype(np.float32).reshape(src[name + ".lora_a"].data.shape))
        w.add_tensor(name + ".lora_b", np.broadcast_to(b[None, :, None], (W.shape[0], b.size, 1)).astype(np.float32).copy())
        old = np.array(src[name + ".lora_a"].data, dtype=np.float32).reshape(-1)
    else:  # attn_output: numpy (out, in), lora_b (out, 1), lora_a (1, in)
        b = b_raw.reshape(-1); b /= np.linalg.norm(b)
        a = -(b @ W)[None, :]
        w.add_tensor(name + ".lora_a", a.astype(np.float32).reshape(src[name + ".lora_a"].data.shape))
        w.add_tensor(name + ".lora_b", b[:, None].astype(np.float32))
        old = np.array(src[name + ".lora_a"].data, dtype=np.float32).reshape(-1)
    new = a.reshape(-1)
    cos = float(new @ old / (np.linalg.norm(new) * np.linalg.norm(old)))
    rel = float(np.linalg.norm(new - old) / np.linalg.norm(old))
    print(f"{name:32s} {base[name].tensor_type.name:7s} cos(new, atomicchat) {cos:.5f}  rel diff {rel:.4f}", flush=True)
    del W

w.write_header_to_file()
w.write_kv_data_to_file()
w.write_tensors_to_file()
w.close()
print("wrote", out_path)
