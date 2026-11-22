"""Split-K decode attention vs the stock lamq kernel on random inputs (no model needed).

  python tests/test_splitk.py
Checks max abs error and times one call for global/sliding layers across context lengths.
"""
import time

import mlx.core as mx
from runtime_mlx.inkling import fast_kernels as fk

mx.random.seed(0)


def case(T, kv_len, kv_start, is_sliding):
    kv_cap = ((kv_len + 255) // 256) * 256
    q = mx.random.normal((T, 24, 128)) * 3
    KV = (mx.random.normal((2, 8, kv_cap, 128)) * 2).astype(mx.float16)
    rel_st = mx.random.normal((T, 24, 16))
    rel_proj = (mx.random.normal((1024, 16)) * 0.5).astype(mx.float16)
    q_in = mx.random.normal((T, 2304))
    sink_w = (mx.random.normal((24, 2304)) * 0.02).astype(mx.float16)
    args = (q, KV, rel_st, rel_proj, q_in, sink_w, T, kv_len, kv_start, kv_start + kv_len - T, kv_cap, is_sliding)

    def run(on):
        fk.SPLITK_ON = on
        return fk.gqa_relpos_sinks_sdpa_fast(*args)

    ref, new = run(False), run(True)
    mx.eval(ref, new)
    err = float(mx.max(mx.abs(ref - new)))

    def bench(on, n=50):
        mx.eval([run(on) for _ in range(3)])
        t = time.perf_counter()
        for _ in range(n):
            mx.eval(run(on))
        return (time.perf_counter() - t) / n * 1e3

    print(f"T={T} kv_len={kv_len:6d} kv_start={kv_start:5d} sliding={int(is_sliding)}  max|err| {err:.1e}  "
          f"stock {bench(False):.3f} ms  split-K {bench(True):.3f} ms")
    assert err < 1e-3, err


for args in [(1, 37, 0, False), (1, 1000, 0, True), (1, 1300, 3000, True), (5, 1200, 7000, True), (1, 4096, 0, False),
             (1, 16384, 0, False), (1, 65536, 0, False), (5, 65536, 0, False)]:
    case(*args)
fk.SPLITK_ON = True
print("ok")
