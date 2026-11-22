#!/usr/bin/env python3
"""OpenAI-compatible chat server for pi: LookAtMyQuant (lamq) Inkling-2-Mini, run on MLX, with runtime uncensoring and vision.

- Model: shinmegamiboson/Inkling-2-Mini-LookAtMyQuant-8.9GB, using lamq's own MLX runtime (pinned commit + lamq.patch; see install.sh).
- Uncensoring: the per-layer refusal directions from AtomicChat's rank-1 LoRA (ablate-dirs.npz). Each direction is
  projected out of every residual write at run time: the o_proj output, each expert's down_proj output, and the
  embedding, all before the post-conv. This removes the direction exactly from the quantized weights' real outputs,
  which merging the LoRA into the weights and then quantizing does not do.
  Measured 2026-11-22 (16 refusal probes / 8 harmless prompts): refusals 10/16 → 0/16; harmless teacher-forced
  KL 0.0117, top-1 96.3%; PPL 2.723 → 2.732.
- Vision: the original mmproj (llama.cpp GGUF) is loaded into inkling_vision.py; its features replace the
  <|unused_200054|> placeholders.
- Prefix cache: one live cache. A request that extends the cached tokens reuses them as-is. Otherwise the cache
  rolls back to the last checkpoint before the first differing token. The conv state can't be trimmed and the
  sliding window drops old tokens, so a checkpoint saves the conv state (by reference) plus a copy of the last
  1024 tokens of each sliding-window layer's KV (about 160 MB). The KV of the global layers is append-only, so
  restoring it just resets its length.
  Checkpoints are taken at the end of the system prompt / tool declarations, and at the prompt's second-to-last token.
- Decode: single-token serial decode with pipelined sampling (the GPU samples step k while Python queues step k+1).
  Attention uses the split-K kernel from lamq.patch. Measured 2026-11-22: decode 42 tok/s at short context,
  40 at 8K, 32 at 32K, 26–29 at 65K; prefill about 1000 tok/s at 1–5K and 580 averaged over 65K.
  The DSpark drafter isn't integrated: at pi's temp 1.0 it only clearly helps on code and math.

Endpoints: GET /health, GET /v1/models (status.value: unloaded|loading|loaded), POST /v1/chat/completions
(stream / non-stream, tools, image_url data URIs, chat_template_kwargs.reasoning_effort).
"""
from __future__ import annotations

import argparse
import base64
import gc
import hashlib
import io
import json
import select
import socket
import sys
import threading
import time
import uuid
from collections import OrderedDict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import mlx.core as mx
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from inkling_vision import IMAGE_TOKEN_ID, ImageInjector, InklingVision, expand_placeholders  # noqa: E402

MODEL_ID = "inkling-2-mini-lamq"
MSG_USER, MSG_MODEL, MSG_SYSTEM = 200000, 200001, 200002
C_TEXT, C_THINK, C_TOOL = 200004, 200008, 200049
END_MSG, EOS = 200010, 200006
SPECIAL_MIN = 199999
KINDS = {C_THINK: "reasoning_content", C_TEXT: "content", C_TOOL: "tool"}
MAX_CKPTS = 3
IMG_CACHE_TOKENS = 32768


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, file=sys.stderr, flush=True)


class ApiError(Exception):
    def __init__(self, status: int, message: str, etype: str = "invalid_request_error"):
        super().__init__(message)
        self.status, self.message, self.etype = status, message, etype


def common_prefix(a: list[int], b: list[int]) -> int:
    n = min(len(a), len(b))
    if n == 0:
        return 0
    diff = np.asarray(a[:n], np.int64) != np.asarray(b[:n], np.int64)
    i = int(np.argmax(diff))
    return i if diff[i] else n


# ----------------------------------------------------------------------------- prefix cache

class Ckpt:
    __slots__ = ("pos", "conv", "win")

    def __init__(self, pos, conv, win):
        self.pos, self.conv, self.win = pos, conv, win


class LiveCache:
    """An Inkling2Cache plus the token keys it holds (image tokens are replaced by the negated image hash) and rollback checkpoints."""

    def __init__(self, m):
        self.m = m
        self.cache = m.make_cache()
        assert self.cache._fast, "prefix rollback assumes lamq's fast-path FP16 KV"
        self.keys: list[int] = []
        self.ckpts: list[Ckpt] = []

    def reset(self):
        self.cache = None
        self.ckpts = []
        gc.collect()
        self.cache = self.m.make_cache()
        self.keys = []

    def checkpoint(self):
        c, W = self.cache, self.m.cfg.sliding_window_size
        assert c.seq_len == len(self.keys)
        win = {}
        for s, kv in enumerate(c.kv_f16):
            if kv is not None and c.kv_is_sliding[s]:
                n = min(c.kv_len[s], W)
                win[s] = mx.contiguous(kv[:, :, c.kv_len[s] - n:c.kv_len[s], :])
        conv = [list(x) for x in c.conv_states]
        mx.eval(list(win.values()), [x for row in conv for x in row if x is not None])
        self.ckpts = [k for k in self.ckpts if k.pos != c.seq_len]
        self.ckpts.append(Ckpt(c.seq_len, conv, win))
        self.ckpts.sort(key=lambda k: k.pos)
        while len(self.ckpts) > MAX_CKPTS:  # keep the earliest (system prompt), drop the next-oldest
            del self.ckpts[1]

    def restore(self, ck: Ckpt):
        c, p = self.cache, ck.pos
        for s, kv in enumerate(c.kv_f16):
            if kv is None:
                continue
            if c.kv_is_sliding[s]:
                w = ck.win[s]
                n = w.shape[2]
                cap = ((n + 2048 + 255) // 256) * 256
                buf = mx.zeros((2, 8, cap, 128), dtype=mx.float16)
                buf[:, :, :n, :] = w
                c.kv_f16[s], c.kv_cap[s], c.kv_start[s], c.kv_len[s] = buf, cap, p - n, n
            else:
                assert c.kv_start[s] == 0
                c.kv_len[s] = p
        c.conv_states = [list(x) for x in ck.conv]
        c.seq_len = p
        del self.keys[p:]
        self.ckpts = [k for k in self.ckpts if k.pos <= p]

    def plan(self, new: list[int]) -> tuple[int, int]:
        """Return (start position, length of the cache to discard) without changing anything."""
        L = common_prefix(self.keys, new)
        if L == len(self.keys) and L < len(new):
            return L, 0
        lim = min(L, len(new) - 1)
        best = max((k.pos for k in self.ckpts if k.pos <= lim), default=0)
        return best, len(self.keys) - best

    def prepare(self, new: list[int]) -> int:
        start, _ = self.plan(new)
        if start == len(self.keys):
            return start
        ck = next((k for k in self.ckpts if k.pos == start), None)
        if ck is None:
            self.reset()
            return 0
        self.restore(ck)
        return start


# ----------------------------------------------------------------------------- output parsing

class Parser:
    """Turn Inkling's message-framed output, <|message_model|>[recipient]<|content_KIND|>body<|end_message|>, into OpenAI deltas."""

    def __init__(self, tok):
        self.tok = tok
        self.state = "idle"
        self.kind = None
        self.head: list[int] = []
        self.body: list[int] = []
        self.pend: list[int] = []
        self.n_calls = 0

    def _text(self, ids):
        return self.tok.decode(ids) if ids else ""

    def feed(self, t: int) -> list[dict]:
        if t == EOS or t == MSG_MODEL or t == END_MSG:
            ev = self._close()
            self.state = "head" if t == MSG_MODEL else "idle"
            self.head = []
            return ev
        if self.state == "head":
            if t >= SPECIAL_MIN:
                self.kind = KINDS.get(t, "drop")
                self.state, self.body, self.pend = "body", [], []
            else:
                self.head.append(t)
            return []
        if t >= SPECIAL_MIN:
            return []
        if self.state == "idle":  # stray text outside a message: show it as content
            self.state, self.kind, self.pend, self.body = "body", "content", [], []
        if self.kind in ("reasoning_content", "content"):
            self.pend.append(t)
            s = self._text(self.pend)
            if s.endswith("�"):
                return []
            self.pend = []
            return [{self.kind: s}] if s else []
        if self.kind == "tool":
            self.body.append(t)
        return []

    def _close(self) -> list[dict]:
        ev = []
        if self.state == "body":
            if self.kind in ("reasoning_content", "content") and self.pend:
                ev.append({self.kind: self._text(self.pend)})
            elif self.kind == "tool":
                ev.append(self._tool_call())
        self.state, self.kind, self.pend, self.body = "idle", None, [], []
        return ev

    def _tool_call(self) -> dict:
        raw = self._text(self.body)
        name = self._text(self.head).strip()
        try:
            obj = json.loads(raw)
            name = obj.get("name") or name
            args = json.dumps(obj.get("args", {}), ensure_ascii=False)
        except (json.JSONDecodeError, AttributeError):
            args = raw  # pass it through as-is; pi reports the parse error to the model, which can retry
        call = {"index": self.n_calls, "id": f"call_{uuid.uuid4().hex[:24]}", "type": "function",
                "function": {"name": name, "arguments": args}}
        self.n_calls += 1
        return {"tool_calls": [call]}


# ----------------------------------------------------------------------------- engine

def decode_image_url(url) -> bytes:
    if isinstance(url, dict):
        url = url.get("url")
    if not isinstance(url, str) or not url.startswith("data:"):
        raise ApiError(400, "only base64 data: URIs are supported for images")
    try:
        return base64.b64decode(url.split(",", 1)[1])
    except Exception as e:  # noqa: BLE001
        raise ApiError(400, f"bad image data URI: {e}") from e


def normalize(messages) -> tuple[list[dict], list[bytes]]:
    """Fit OpenAI messages to the Inkling template: tool-call arguments become dicts, reasoning fields are unified, and image bytes are collected in template order."""
    if not isinstance(messages, list) or not messages:
        raise ApiError(400, "messages must be a non-empty list")
    out, blobs = [], []
    for m in messages:
        m = dict(m)
        if m.get("role") == "developer":
            m["role"] = "system"
        if not m.get("reasoning_content"):
            for k in ("reasoning", "reasoning_text"):
                if m.get(k):
                    m["reasoning_content"] = m[k]
                    break
        c = m.get("content")
        if isinstance(c, list):
            parts = []
            for p in c:
                if isinstance(p, dict) and p.get("type") in ("image_url", "input_image", "image"):
                    blobs.append(decode_image_url(p.get("image_url") or p.get("url")))
                    parts.append({"type": "image_url"})
                else:
                    parts.append(p)
            m["content"] = parts
        if m.get("tool_calls"):
            calls = []
            for tc in m["tool_calls"]:
                tc = json.loads(json.dumps(tc))
                fn = tc.setdefault("function", {})
                a = fn.get("arguments")
                if isinstance(a, str):
                    try:
                        a = json.loads(a) if a.strip() else {}
                    except json.JSONDecodeError:
                        a = {"_raw": a}
                if a is not None and not isinstance(a, dict):
                    a = {"_value": a}
                fn["arguments"] = a or {}
                calls.append(tc)
            m["tool_calls"] = calls
        out.append(m)
    return out, blobs


class Job:
    pass


class Engine:
    def __init__(self, a):
        self.a = a
        self.lock = threading.Lock()
        self.status = "unloaded"
        self.m = self.tok = self.vision = self.live = None
        self.inj = ImageInjector()
        self.img_cache: OrderedDict[str, mx.array] = OrderedDict()
        self.last_used = time.monotonic()
        threading.Thread(target=self._idle_loop, daemon=True).start()

    # -- lifecycle
    def load(self):
        if self.m is not None:
            return
        self.status = "loading"
        try:
            from runtime_mlx.inkling import load_inkling
            t = time.perf_counter()
            w, tok = load_inkling(self.a.model)
            m = w.inner
            if self.a.ablate:
                z = np.load(self.a.ablate)
                unit = lambda v: mx.array((v / np.linalg.norm(v)).astype(np.float32))  # noqa: E731
                m.set_ablation(([unit(v) for v in z["layers"]], unit(z["embd"])))
            m.embed.inject = self.inj
            mx.set_wired_limit(mx.device_info()["max_recommended_working_set_size"])
            # Warm up: JIT-compile the prefill and single-token decode kernels (otherwise the first request pays ~3 s)
            c = m.make_cache()
            mx.eval(m.forward_argmax([MSG_USER, C_TEXT] + [13225] * 300, cache=c)[0])
            mx.eval(m.forward_ordered_logits([END_MSG], cache=c)[0])
            del c
            self.m, self.tok, self.live = m, tok, LiveCache(m)
            self.status = "loaded"
            log(f"loaded in {time.perf_counter() - t:.1f}s, active {mx.get_active_memory() / 2**30:.2f} GiB, "
                f"ablation {'on' if self.a.ablate else 'off'}")
        except BaseException:
            self.status = "unloaded"
            raise

    def unload(self):
        self.m = self.tok = self.vision = self.live = None
        self.img_cache.clear()
        self.inj.load([])
        gc.collect()
        mx.clear_cache()
        mx.set_wired_limit(0)
        self.status = "unloaded"
        log(f"unloaded, active {mx.get_active_memory() / 2**30:.2f} GiB")

    def _idle_loop(self):
        while True:
            time.sleep(15)
            if self.m is None or time.monotonic() - self.last_used < self.a.idle:
                continue
            if self.lock.acquire(blocking=False):
                try:
                    if self.m is not None and time.monotonic() - self.last_used >= self.a.idle:
                        self.unload()
                finally:
                    self.lock.release()

    # -- request preparation (raises ApiError before any response bytes are sent)
    def _image_feats(self, blob: bytes) -> tuple[str, mx.array]:
        from PIL import Image
        h = hashlib.sha1(blob).hexdigest()
        if h in self.img_cache:
            self.img_cache.move_to_end(h)
            return h, self.img_cache[h]
        if self.vision is None:
            t = time.perf_counter()
            self.vision = InklingVision.from_mmproj(self.a.mmproj)
            log(f"vision loaded in {time.perf_counter() - t:.1f}s")
        try:
            img = Image.open(io.BytesIO(blob))
            img.load()
        except Exception as e:  # noqa: BLE001
            raise ApiError(400, f"cannot decode image: {e}") from e
        f = self.vision.encode(img).astype(mx.float16)
        mx.eval(f)
        self.img_cache[h] = f
        while sum(v.shape[0] for v in self.img_cache.values()) > IMG_CACHE_TOKENS and len(self.img_cache) > 1:
            self.img_cache.popitem(last=False)
        return h, f

    def prepare(self, req: dict) -> Job:
        self.load()
        msgs, blobs = normalize(req.get("messages"))
        kw = dict(req.get("chat_template_kwargs") or {})
        effort = kw.pop("reasoning_effort", None) or req.get("reasoning_effort")
        if effort is not None:
            kw["reasoning_effort"] = effort
        tools = req.get("tools") or None
        try:
            ids = self.tok.apply_chat_template(msgs, tools=tools, add_generation_prompt=True, **kw)
        except Exception as e:  # noqa: BLE001  (jinja raise_exception, bad roles, ...)
            raise ApiError(400, f"chat template error: {e}") from e
        if isinstance(ids, dict) or hasattr(ids, "input_ids"):
            ids = ids["input_ids"]
        ids = [int(x) for x in ids]
        feats, hashes = [], []
        for b in blobs:
            h, f = self._image_feats(b)
            hashes.append(h)
            feats.append(f)
        try:
            ids = expand_placeholders(ids, [f.shape[0] for f in feats])
        except ValueError as e:
            raise ApiError(400, str(e)) from e
        keys, spans, k = list(ids), [], 0
        i = 0
        while i < len(ids):
            if ids[i] == IMAGE_TOKEN_ID:
                n = feats[k].shape[0]
                key = -int(hashes[k][:15], 16)
                keys[i:i + n] = [key] * n
                spans.append((i, i + n, feats[k]))
                i += n
                k += 1
            else:
                i += 1
        n = len(ids)
        if n + 1 > self.a.ctx:
            raise ApiError(400, f"the request exceeds the available context size: {n} prompt tokens, context {self.a.ctx}",
                           "exceed_context_size_error")
        j = Job()
        j.ids, j.keys, j.spans = ids, keys, spans
        j.max_new = max(1, min(int(req.get("max_tokens") or req.get("max_completion_tokens") or 16384), self.a.ctx - n))
        t = req.get("temperature")
        j.temp = 1.0 if t is None else float(t)
        j.top_p = float(req.get("top_p") or 1.0)
        j.top_k = int(req.get("top_k") or 0)
        j.no_cache = req.get("cache_prompt") is False  # same convention as llama.cpp: start from an empty cache
        # Checkpoint at the end of the system prompt / tool declarations, i.e. just before the effort message that precedes the first user turn.
        j.sys_pos = 0
        if MSG_USER in ids:
            u = ids.index(MSG_USER)
            sys_starts = [p for p in range(u) if ids[p] == MSG_SYSTEM]
            if sys_starts and sys_starts[-1] >= 512:
                j.sys_pos = sys_starts[-1]
        return j

    # -- generation
    def run(self, j: Job, out) -> dict:
        m, n = self.m, len(j.ids)
        lc = self.live
        if j.no_cache:
            lc.reset()
        start, discard = lc.plan(j.keys)
        temp_cache = discard > 8192 and n - start < 8192 and common_prefix(lc.keys, j.keys) < n // 2
        if temp_cache:  # a small, unrelated request (title, side query): leave the large live cache alone
            lc = LiveCache(m)
            start = 0
        else:
            start = lc.prepare(j.keys)
        # Image features for positions >= start, in placeholder order (start may fall inside an image)
        self.inj.load([f[max(0, start - s):] for s, e, f in j.spans if e > start])
        mx.reset_peak_memory()
        t0 = time.perf_counter()
        cuts = sorted({p for p in (j.sys_pos, n - 1) if start < p < n} | {n})
        pos, logits, aborted = start, None, False
        try:
            for tgt in cuts:
                while pos < tgt:
                    e = min(pos + self.a.chunk, tgt)
                    seg = j.ids[pos:e]
                    if e == n:
                        logits, _ = m.forward_ordered_logits(seg, cache=lc.cache)
                        mx.eval(logits)
                    else:
                        o, _ = m.forward_argmax(seg, cache=lc.cache)
                        mx.eval(o)
                    lc.keys.extend(j.keys[pos:e])
                    pos = e
                    if e < n:
                        out.progress(pos, n)
                        if out.gone():
                            aborted = True
                            break
                if aborted:
                    break
                if tgt < n and not temp_cache:
                    lc.checkpoint()
            self.inj.load([])
            t1 = time.perf_counter()
            parser = Parser(self.tok)
            gen, finish = [], "abort" if aborted else "length"
            if not aborted:
                order, cap = m.lm_head.order_mx, m.lm_head.cap
                from runtime_mlx.inkling.generate import _sample_ordered_gpu

                def sample(lg):
                    return _sample_ordered_gpu(lg[0, -1], order, cap=cap, temp=j.temp, top_p=j.top_p, top_k=j.top_k)

                cur = sample(logits)
                mx.async_eval(cur)
                nxt = None
                try:
                    for i in range(j.max_new):
                        nxt = None
                        if i + 1 < j.max_new:
                            lg, _ = m.forward_ordered_logits(cur.reshape(1), cache=lc.cache)
                            nxt = sample(lg)
                            mx.async_eval(nxt)
                        t = int(cur)
                        if nxt is not None:
                            lc.keys.append(t)
                        gen.append(t)
                        for d in parser.feed(t):
                            out.delta(d)
                        if t == EOS:
                            finish = "stop"
                            break
                        if out.gone():
                            aborted, finish = True, "abort"
                            break
                        cur = nxt
                finally:
                    if nxt is not None:
                        mx.eval(nxt)
                for d in parser._close():
                    out.delta(d)
                if parser.n_calls and finish == "stop":
                    finish = "tool_calls"
            t2 = time.perf_counter()
        except BaseException:
            self.live = LiveCache(m) if self.m is not None else None  # cache state is unknown after a failure
            raise
        finally:
            self.inj.load([])
            mx.clear_cache()
            self.last_used = time.monotonic()
        nd = max(len(gen) - 1, 0)
        st = {"prompt_n": n - start, "cached": start, "prompt_ms": (t1 - t0) * 1e3, "predicted_n": len(gen),
              "predicted_ms": (t2 - t1) * 1e3, "finish": finish}
        log(f"prompt {n} (cached {start}{', temp cache' if temp_cache else ''}) prefill {(n - start) / max(t1 - t0, 1e-9):.0f} tok/s | "
            f"gen {len(gen)} at {nd / max(t2 - t1, 1e-9):.1f} tok/s | {finish} | active {mx.get_active_memory() / 2**30:.2f} GiB, "
            f"peak {mx.get_peak_memory() / 2**30:.2f} GiB"
            f" ckpts {[k.pos for k in self.live.ckpts]}")
        return st


# ----------------------------------------------------------------------------- HTTP

def client_gone(sock) -> bool:
    try:
        r, _, _ = select.select([sock], [], [], 0)
        return bool(r) and sock.recv(1, socket.MSG_PEEK) == b""
    except OSError:
        return True


class SSEOut:
    def __init__(self, h, rid, created):
        self.h, self.rid, self.created, self.dead = h, rid, created, False
        self.last_ping = time.monotonic()

    def _write(self, b: bytes):
        if self.dead:
            return
        try:
            self.h.wfile.write(b)
            self.h.wfile.flush()
        except (BrokenPipeError, ConnectionResetError, OSError):
            self.dead = True

    def send(self, obj):
        self._write(b"data: " + json.dumps(obj, ensure_ascii=False).encode() + b"\n\n")

    def chunk(self, delta, finish=None):
        self.send({"id": self.rid, "object": "chat.completion.chunk", "created": self.created, "model": MODEL_ID,
                   "choices": [{"index": 0, "delta": delta, "finish_reason": finish}]})

    def delta(self, d):
        self.chunk(d)

    def progress(self, pos, n):
        self._write(f": prefill {pos}/{n}\n\n".encode())  # SSE comment: keeps the connection alive; ignored by clients

    def gone(self):
        return self.dead or client_gone(self.h.connection)


class CollectOut:
    def __init__(self, h):
        self.h = h
        self.content, self.reasoning, self.calls = [], [], []

    def delta(self, d):
        if "content" in d:
            self.content.append(d["content"])
        if "reasoning_content" in d:
            self.reasoning.append(d["reasoning_content"])
        for c in d.get("tool_calls", []):
            self.calls.append({k: v for k, v in c.items() if k != "index"})

    def progress(self, pos, n):
        pass

    def gone(self):
        return client_gone(self.h.connection)


def usage(st):
    return {"prompt_tokens": st["prompt_n"] + st["cached"], "completion_tokens": st["predicted_n"],
            "total_tokens": st["prompt_n"] + st["cached"] + st["predicted_n"],
            "prompt_tokens_details": {"cached_tokens": st["cached"]}}


def timings(st):
    return {"cache_n": st["cached"], "prompt_n": st["prompt_n"], "prompt_ms": round(st["prompt_ms"], 1),
            "prompt_per_second": round(st["prompt_n"] / max(st["prompt_ms"], 1e-6) * 1e3, 1),
            "predicted_n": st["predicted_n"], "predicted_ms": round(st["predicted_ms"], 1),
            "predicted_per_second": round(max(st["predicted_n"] - 1, 0) / max(st["predicted_ms"], 1e-6) * 1e3, 1)}


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    engine: Engine

    def log_message(self, *a):
        pass

    def _json(self, code, obj):
        b = json.dumps(obj, ensure_ascii=False).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(b)))
        self.end_headers()
        self.wfile.write(b)

    def _error(self, code, msg, etype):
        self._json(code, {"error": {"code": code, "message": msg, "type": etype}})

    def do_GET(self):
        p = self.path.split("?")[0].rstrip("/")
        if p == "/health":
            self._json(200, {"status": "ok"})
        elif p in ("/v1/models", "/models"):
            self._json(200, {"object": "list", "data": [{"id": MODEL_ID, "object": "model", "created": 0, "owned_by": "lamq",
                                                          "status": {"value": self.engine.status}}]})
        else:
            self._error(404, "not found", "not_found_error")

    def do_POST(self):
        p = self.path.split("?")[0].rstrip("/")
        if p not in ("/v1/chat/completions", "/chat/completions"):
            return self._error(404, "not found", "not_found_error")
        try:
            req = json.loads(self.rfile.read(int(self.headers.get("Content-Length") or 0)) or b"{}")
        except json.JSONDecodeError as e:
            return self._error(400, f"bad JSON: {e}", "invalid_request_error")
        rid, created = f"chatcmpl-{uuid.uuid4().hex[:24]}", int(time.time())
        eng = self.engine
        with eng.lock:
            eng.last_used = time.monotonic()
            try:
                job = eng.prepare(req)
            except ApiError as e:
                log(f"400: {e.message}")
                return self._error(e.status, e.message, e.etype)
            except Exception as e:  # noqa: BLE001
                log(f"500 in prepare: {e!r}")
                return self._error(500, f"{type(e).__name__}: {e}", "server_error")
            if req.get("stream"):
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Cache-Control", "no-cache")
                self.send_header("Connection", "close")
                self.end_headers()
                self.close_connection = True
                out = SSEOut(self, rid, created)
                out.chunk({"role": "assistant", "content": ""})
                try:
                    st = eng.run(job, out)
                except Exception as e:  # noqa: BLE001
                    log(f"500 in run: {e!r}")
                    out.send({"error": {"message": f"{type(e).__name__}: {e}", "type": "server_error"}})
                    out._write(b"data: [DONE]\n\n")
                    return
                if st["finish"] != "abort":
                    out.chunk({}, st["finish"])
                    if (req.get("stream_options") or {}).get("include_usage"):
                        out.send({"id": rid, "object": "chat.completion.chunk", "created": created, "model": MODEL_ID,
                                  "choices": [], "usage": usage(st), "timings": timings(st)})
                    out._write(b"data: [DONE]\n\n")
            else:
                out = CollectOut(self)
                try:
                    st = eng.run(job, out)
                except Exception as e:  # noqa: BLE001
                    log(f"500 in run: {e!r}")
                    return self._error(500, f"{type(e).__name__}: {e}", "server_error")
                msg = {"role": "assistant", "content": "".join(out.content) or None}
                if out.reasoning:
                    msg["reasoning_content"] = "".join(out.reasoning)
                if out.calls:
                    msg["tool_calls"] = out.calls
                self._json(200, {"id": rid, "object": "chat.completion", "created": created, "model": MODEL_ID,
                                 "choices": [{"index": 0, "message": msg, "finish_reason": st["finish"]}],
                                 "usage": usage(st), "timings": timings(st)})


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--model", required=True, help="lamq packed model directory")
    ap.add_argument("--mmproj", required=True, help="Inkling 2 mmproj GGUF (llama.cpp)")
    ap.add_argument("--ablate", default="", help="refusal directions npz (layers [24,2304], embd); empty disables")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=9932)
    ap.add_argument("--ctx", type=int, default=81920)
    ap.add_argument("--chunk", type=int, default=2048)
    ap.add_argument("--idle", type=float, default=600, help="unload the model after this many idle seconds")
    ap.add_argument("--preload", action="store_true")
    a = ap.parse_args()
    eng = Engine(a)
    Handler.engine = eng
    if a.preload:
        with eng.lock:
            eng.load()
    srv = ThreadingHTTPServer((a.host, a.port), Handler)
    srv.daemon_threads = True
    log(f"listening on http://{a.host}:{a.port}/v1 (model {MODEL_ID}, ctx {a.ctx}, idle unload {a.idle:.0f}s)")
    srv.serve_forever()


if __name__ == "__main__":
    main()
