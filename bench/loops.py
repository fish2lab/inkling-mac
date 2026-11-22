"""Degeneration (looping) test against any OpenAI-compatible server. Prints statistics only, never the generated text.

  python bench/loops.py --url http://127.0.0.1:9932 --model inkling-2-mini-lamq --corpus docs.txt --ctx-chars 200000 -n 4

Each sample puts --ctx-chars of the corpus in context (prefix-cached after the first sample) and asks for a long,
open-ended write-up at the model's default sampling (temp 1.0). Per sample it reports finish reason, length, and:
  rep    fraction of output words that sit inside a 12-word sequence already seen earlier in the same output
  tail   the same measure over the last 400 words only (a loop that started late shows here)
  loop   tail > 0.5, i.e. the output ended stuck in a repeat
Reasoning and content are scored together.
"""
import argparse
import json
import time
import urllib.request

ap = argparse.ArgumentParser()
ap.add_argument("--url", default="http://127.0.0.1:9932")
ap.add_argument("--model", default="inkling-2-mini-lamq")
ap.add_argument("--corpus", required=True)
ap.add_argument("--ctx-chars", type=int, default=200000)
ap.add_argument("-n", type=int, default=4)
ap.add_argument("--max-tokens", type=int, default=3000)
ap.add_argument("--effort", default="medium")
a = ap.parse_args()

ctx = open(a.corpus).read()[: a.ctx_chars]
ASK = ("\n\nUsing only the documentation above, write a complete, detailed guide to building an extension that adds a "
       "custom tool, a custom command and a custom UI widget, with full code for every step and an explanation of "
       "every API it touches.")


def rep(words, n=12):
    seen, hit = set(), [False] * len(words)
    for i in range(len(words) - n + 1):
        g = tuple(words[i:i + n])
        if g in seen:
            for k in range(i, i + n):
                hit[k] = True
        seen.add(g)
    return hit


def sample():
    body = {"model": a.model, "stream": True, "max_tokens": a.max_tokens, "stream_options": {"include_usage": True},
            "chat_template_kwargs": {"reasoning_effort": a.effort},
            "messages": [{"role": "user", "content": ctx + ASK}]}
    req = urllib.request.Request(a.url + "/v1/chat/completions", json.dumps(body).encode(), {"Content-Type": "application/json"})
    text, finish, usage, t = [], None, {}, time.time()
    for line in urllib.request.urlopen(req, timeout=3600):
        line = line.decode().strip()
        if not line.startswith("data: ") or line == "data: [DONE]":
            continue
        d = json.loads(line[6:])
        usage = d.get("usage") or usage
        for c in d.get("choices", []):
            dl = c.get("delta", {})
            text.append((dl.get("reasoning_content") or "") + (dl.get("content") or ""))
            finish = c.get("finish_reason") or finish
    words = "".join(text).split()
    hit = rep(words)
    tail = hit[-400:]
    r, tr = sum(hit) / max(len(hit), 1), sum(tail) / max(len(tail), 1)
    return {"finish": finish, "prompt": usage.get("prompt_tokens"), "gen": usage.get("completion_tokens"),
            "words": len(words), "rep": r, "tail": tr, "loop": tr > 0.5, "s": time.time() - t}


res = []
for i in range(a.n):
    s = sample()
    res.append(s)
    print(f"{i}: finish {s['finish']:10s} prompt {s['prompt']} gen {s['gen']:5d} words {s['words']:5d} "
          f"rep {s['rep']:.3f} tail {s['tail']:.3f} {'LOOP' if s['loop'] else ''} ({s['s']:.0f}s)", flush=True)
print(f"loops {sum(r['loop'] for r in res)}/{len(res)}, mean rep {sum(r['rep'] for r in res) / len(res):.3f}, "
      f"hit max_tokens {sum(r['finish'] == 'length' for r in res)}/{len(res)}")
