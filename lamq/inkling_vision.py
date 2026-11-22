"""Inkling 2 vision encoder for MLX. Reads weights straight from llama.cpp's mmproj GGUF and feeds lamq's text trunk.

The network is ported from mlx-vlm (commit f1618d6d, mlx_vlm/models/inkling2/vision.py, MIT).
Preprocessing matches llama.cpp's mtmd_image_preprocessor_inkling2v: native resolution, padded to a multiple of 40
(padding value -1 before normalization), scaled down only when the image exceeds 4096 image tokens.
Image features skip embed_norm and the refusal ablation; they replace the <|unused_200054|> placeholders in the prompt.
Verified against the mlx-vlm reference: min cosine similarity 0.99998 per token.
"""
from __future__ import annotations

import math
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from PIL import Image

IMAGE_TOKEN_ID = 200054
PATCH, MERGE, HIDDEN, HEADS, LAYERS, GRID = 20, 2, 1024, 16, 23, 32
CELL = PATCH * MERGE
MAX_IMAGE_TOKENS = 4096


def _rotate_pairs(x):
    pairs = x.reshape(*x.shape[:-1], -1, 2)
    return mx.stack([-pairs[..., 1], pairs[..., 0]], axis=-1).reshape(x.shape)


class _Attn(nn.Module):
    def __init__(self):
        super().__init__()
        for name in ("q_proj", "k_proj", "v_proj", "out_proj"):
            self[name] = nn.Linear(HIDDEN, HIDDEN)

    def __call__(self, x, cos, sin):
        hd = HIDDEN // HEADS
        q, k, v = [p(x).reshape(1, x.shape[0], HEADS, hd) for p in (self.q_proj, self.k_proj, self.v_proj)]
        rope = lambda a: a * cos[None, :, None] + _rotate_pairs(a) * sin[None, :, None]
        q, k = rope(q), rope(k)
        q, k, v = (a.transpose(0, 2, 1, 3) for a in (q, k, v))
        o = mx.fast.scaled_dot_product_attention(q, k, v, scale=hd ** -0.5)
        return self.out_proj(o.transpose(0, 2, 1, 3).reshape(x.shape))


class _Layer(nn.Module):
    def __init__(self):
        super().__init__()
        self.self_attn = _Attn()
        self.fc1 = nn.Linear(HIDDEN, 4 * HIDDEN)
        self.fc2 = nn.Linear(4 * HIDDEN, HIDDEN)
        self.ln1 = nn.LayerNorm(HIDDEN, eps=1e-5)
        self.ln2 = nn.LayerNorm(HIDDEN, eps=1e-5)
        self.ls1 = mx.ones((HIDDEN,))
        self.ls2 = mx.ones((HIDDEN,))

    def __call__(self, x, cos, sin):
        x = x + self.ls1 * self.self_attn(self.ln1(x), cos, sin)
        return x + self.ls2 * self.fc2(nn.gelu(self.fc1(self.ln2(x))))


def _bilinear(src: np.ndarray, out_h: int, out_w: int) -> np.ndarray:
    """F.interpolate(mode='bilinear', align_corners=False) for an (H, W, C) array."""
    def axis(n_in, n_out):
        s = np.maximum((np.arange(n_out) + 0.5) * n_in / n_out - 0.5, 0.0)
        i0 = np.minimum(np.floor(s).astype(int), n_in - 1)
        return i0, np.minimum(i0 + 1, n_in - 1), (s - i0)[:, None]
    y0, y1, wy = axis(src.shape[0], out_h)
    x0, x1, wx = axis(src.shape[1], out_w)
    rows = src[y0] * (1 - wy[..., None]) + src[y1] * wy[..., None]
    return rows[:, x0] * (1 - wx) + rows[:, x1] * wx


class InklingVision(nn.Module):
    def __init__(self):
        super().__init__()
        self.patch_embedding = nn.Linear(3 * PATCH * PATCH, HIDDEN, bias=False)
        self.position_embedding = mx.zeros((GRID * GRID, HIDDEN))
        self.class_embedding = mx.zeros((HIDDEN,))
        self.cls_position_embedding = mx.zeros((HIDDEN,))
        self.pre_ln = nn.LayerNorm(HIDDEN, eps=1e-5)
        self.layers = [_Layer() for _ in range(LAYERS)]
        self.proj_in = nn.Linear(MERGE * MERGE * HIDDEN, 4096, bias=False)
        self.proj_out = nn.Linear(4096, 2304, bias=False)

    @classmethod
    def from_mmproj(cls, path: str | Path) -> "InklingVision":
        from gguf import GGUFReader
        r = GGUFReader(str(path))
        T = {t.name: np.array(t.data) for t in r.tensors}
        w: dict[str, np.ndarray] = {
            # GGUF stores the conv2d layout (out, C, kh, kw); flatten to an HWC Linear
            "patch_embedding.weight": T["v.patch_embd.weight"].transpose(0, 2, 3, 1).reshape(HIDDEN, -1),
            "position_embedding": T["v.position_embd.weight"],
            "class_embedding": T["v.class_embd"],
            "cls_position_embedding": T["v.class_position_embd.weight"],
            "pre_ln.weight": T["v.pre_ln.weight"], "pre_ln.bias": T["v.pre_ln.bias"],
            "proj_in.weight": T["mm.1.weight"], "proj_out.weight": T["mm.2.weight"],
        }
        names = {"attn_q": "self_attn.q_proj", "attn_k": "self_attn.k_proj", "attn_v": "self_attn.v_proj",
                 "attn_out": "self_attn.out_proj", "ffn_up": "fc1", "ffn_down": "fc2", "ln1": "ln1", "ln2": "ln2"}
        for i in range(LAYERS):
            for g, m in names.items():
                for sfx in ("weight", "bias"):
                    w[f"layers.{i}.{m}.{sfx}"] = T[f"v.blk.{i}.{g}.{sfx}"]
            w[f"layers.{i}.ls1"] = T[f"v.blk.{i}.ls1.weight"]
            w[f"layers.{i}.ls2"] = T[f"v.blk.{i}.ls2.weight"]
        model = cls()
        # F16 weights to save memory, F32 activations (same as llama.cpp's F16 weight x F32 activation)
        model.load_weights([(k, mx.array(v.astype(np.float16 if v.ndim >= 2 else np.float32))) for k, v in w.items()], strict=True)
        mx.eval(model.parameters())
        model._pos_grid = T["v.position_embd.weight"].astype(np.float32).reshape(GRID, GRID, HIDDEN)
        return model

    @staticmethod
    def preprocess(image: Image.Image | str | Path) -> tuple[np.ndarray, int, int]:
        img = Image.open(image) if isinstance(image, (str, Path)) else image
        if img.mode in ("RGBA", "LA") or (img.mode == "P" and "transparency" in img.info):
            rgba = np.asarray(img.convert("RGBA")).astype(np.int32)
            yy, xx = np.indices(rgba.shape[:2])
            bg = np.where((yy // 8 + xx // 8) % 2 == 0, 204, 153)[..., None]
            a = rgba[..., 3:4]
            arr = ((rgba[..., :3] * a + bg * (255 - a) + 127) // 255).astype(np.uint8)
            img = Image.fromarray(arr)
        img = img.convert("RGB")
        w, h = img.size
        if math.ceil(w / CELL) * math.ceil(h / CELL) > MAX_IMAGE_TOKENS:
            s = math.sqrt(MAX_IMAGE_TOKENS * CELL * CELL / (w * h))
            w, h = max(CELL, int(w * s) // CELL * CELL), max(CELL, int(h * s) // CELL * CELL)
            img = img.resize((w, h), Image.BILINEAR)
        rows, cols = math.ceil(h / CELL), math.ceil(w / CELL)
        padded = np.full((rows * CELL, cols * CELL, 3), -1.0, np.float32)
        padded[:h, :w] = np.asarray(img, np.float32)
        padded = (padded / 255.0 - 0.5) / 0.5
        return padded, rows, cols

    def encode(self, image) -> mx.array:
        """Return (rows*cols, 2304) image features, row-major, one token per 40x40 pixel cell."""
        px, rows, cols = self.preprocess(image)
        count = rows * cols
        # (rows, m, p, cols, m, p, C) -> each token's m x m patches stored contiguously, HWC-flattened within a patch
        patches = px.reshape(rows, MERGE, PATCH, cols, MERGE, PATCH, 3).transpose(0, 3, 1, 4, 2, 5, 6)
        x = self.patch_embedding(mx.array(patches.reshape(count * MERGE * MERGE, -1)))
        pos = _bilinear(self._pos_grid, rows * MERGE, cols * MERGE)
        pos = pos.reshape(rows, MERGE, cols, MERGE, HIDDEN).transpose(0, 2, 1, 3, 4).reshape(-1, HIDDEN)
        x = x + mx.array(pos)
        cls = self.class_embedding + self.cls_position_embedding
        x = self.pre_ln(mx.concatenate([cls[None].astype(mx.float32), x.astype(mx.float32)], axis=0))
        yy, xx = np.meshgrid(np.arange(rows * MERGE), np.arange(cols * MERGE), indexing="ij")
        p2 = np.stack([xx, yy], -1).reshape(rows, MERGE, cols, MERGE, 2).transpose(0, 2, 1, 3, 4).reshape(-1, 2) + 1
        p2 = mx.array(np.concatenate([np.zeros((1, 2)), p2]), mx.float32)
        sd = HIDDEN // HEADS // 2
        inv = 10000.0 ** (-mx.arange(0, sd, 2) / sd)
        ang = mx.repeat((p2[..., None] * inv).reshape(x.shape[0], -1), 2, axis=-1)
        cos, sin = mx.cos(ang), mx.sin(ang)
        for layer in self.layers:
            x = layer(x, cos, sin)
        x = x[1:].reshape(count, MERGE * MERGE * HIDDEN)
        return self.proj_out(nn.gelu(self.proj_in(x)))


class ImageInjector:
    """Attached to PackedEmbed: replaces placeholder tokens with image features in order. The cursor carries across prefill chunks."""

    def __init__(self):
        self.feats: mx.array | None = None
        self.cursor = 0

    def load(self, feats: list[mx.array]):
        self.feats = mx.concatenate(feats, axis=0) if feats else None
        self.cursor = 0

    def __call__(self, ids, h: mx.array) -> mx.array:
        if self.feats is None or self.cursor >= self.feats.shape[0]:
            return h
        ids_np = np.asarray(ids if not isinstance(ids, mx.array) else np.array(ids)).reshape(-1)
        pos = np.nonzero(ids_np == IMAGE_TOKEN_ID)[0]
        if len(pos) == 0:
            return h
        n = len(pos)
        if self.cursor + n > self.feats.shape[0]:
            raise ValueError("more image placeholders than image features")
        f = self.feats[self.cursor:self.cursor + n].astype(h.dtype)
        self.cursor += n
        h = mx.array(h)
        h[0, mx.array(pos)] = f
        return h


def expand_placeholders(ids: list[int], counts: list[int]) -> list[int]:
    """The chat template emits one <|unused_200054|> per image; expand it to that image's token count."""
    out, k = [], 0
    for t in ids:
        if t == IMAGE_TOKEN_ID:
            if k >= len(counts):
                raise ValueError("more image placeholders than images")
            out.extend([IMAGE_TOKEN_ID] * counts[k]); k += 1
        else:
            out.append(t)
    if k != len(counts):
        raise ValueError("more images than placeholders")
    return out
