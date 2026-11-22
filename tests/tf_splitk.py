"""Teacher-forced decode KL: split-K vs the stock kernel, next to the stock runtime's own prefill-vs-decode gap.

  python tests/tf_splitk.py --corpus docs.txt [--ctx 20000] [--steps 200]
Greedy-decodes `steps` tokens with the stock kernel, then replays the same tokens (a) through split-K decode and
(b) through the prefill path, and prints KL / top-1 agreement for each pair.
"""
import argparse
import os
import sys

import mlx.core as mx
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "bench"))
from bench import DEFAULT_MODEL, corpus_ids  # noqa: E402
from runtime_mlx.inkling import fast_kernels as fk  # noqa: E402
from runtime_mlx.inkling import load_inkling  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--model", default=str(DEFAULT_MODEL)); ap.add_argument("--corpus", required=True)
ap.add_argument("--ctx", type=int, default=20000); ap.add_argument("--steps", type=int, default=200)
a = ap.parse_args()
w, tok = load_inkling(a.model); m = w.inner
ids = corpus_ids(tok, a.corpus, a.ctx)


def prefill(seq, c):
    for i in range(0, len(seq), 2048):
        mx.eval(m.forward_argmax(seq[i:i + 2048], cache=c)[0])


def decode(on, forced=None):
    fk.SPLITK_ON = on
    c = m.make_cache(); prefill(ids[:-1], c)
    cur, toks, lps = ids[-1], [], []
    for k in range(a.steps):
        lg, _ = m.forward_ordered_logits([cur], cache=c)
        lp = lg[0, -1].astype(mx.float32); lp = lp - mx.logsumexp(lp)
        lps.append(np.array(lp)); t = int(m.lm_head.order_mx[mx.argmax(lp)])
        toks.append(t); cur = forced[k] if forced else t
    return toks, np.stack(lps)


t0, l0 = decode(False)
_, l1 = decode(True, forced=t0)
fk.SPLITK_ON = False
c = m.make_cache(); prefill(ids[:-1], c)
lg, _ = m.forward_all_ordered_logits(ids[-1:] + t0[:-1], cache=c)
lp = lg[0].astype(mx.float32); l2 = np.array(lp - mx.logsumexp(lp, axis=-1, keepdims=True))
fk.SPLITK_ON = True
for name, x, y in (("stock decode vs stock prefill (runtime's own gap)", l0, l2), ("stock decode vs split-K decode", l0, l1)):
    kl = np.sum(np.exp(x) * (x - y), -1)
    print(f"ctx {a.ctx} {name}: KL mean {kl.mean():.2e} p99 {np.quantile(kl, .99):.2e} max {kl.max():.2e}, "
          f"top-1 {np.mean(x.argmax(-1) == y.argmax(-1)):.3f}")
