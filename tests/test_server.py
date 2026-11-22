"""End-to-end checks against a running lamq-server (stdlib only; Pillow, if importable, adds an image test).

  lamq/lamq-server &      # or let pi start it
  python3 tests/test_server.py [--url http://127.0.0.1:9932]

Covers: /health and /v1/models, streamed text with reasoning, tool calls, prefix-cache reuse across turns,
cache_prompt:false, an image round trip, the 400 context-overflow error, /metrics, and the Host / Content-Type guards.
"""
import argparse
import base64
import io
import json
import sys
import time
import urllib.error
import urllib.request

ap = argparse.ArgumentParser()
ap.add_argument("--url", default="http://127.0.0.1:9932")
URL = ap.parse_args().url.rstrip("/")
MODEL = "inkling-2-mini-lamq"


def get(path):
    with urllib.request.urlopen(URL + path) as r:
        return json.load(r)


def chat(messages, tools=None, effort="none", max_tokens=600, **extra):
    body = {"model": MODEL, "messages": messages, "stream": True, "max_tokens": max_tokens,
            "stream_options": {"include_usage": True}, "chat_template_kwargs": {"reasoning_effort": effort}, **extra}
    if tools:
        body["tools"] = tools
    req = urllib.request.Request(URL + "/v1/chat/completions", json.dumps(body).encode(), {"Content-Type": "application/json"})
    try:
        r = urllib.request.urlopen(req)
    except urllib.error.HTTPError as e:
        return {"error": e.code, "body": json.loads(e.read())}
    msg, calls, usage, finish = {"content": "", "reasoning_content": ""}, {}, None, None
    for line in r:
        line = line.decode().rstrip("\n")
        if not line.startswith("data: ") or line == "data: [DONE]":
            continue
        d = json.loads(line[6:])
        usage = d.get("usage") or usage
        for c in d["choices"]:
            dl = c["delta"]
            for k in msg:
                msg[k] += dl.get(k) or ""
            for tc in dl.get("tool_calls", []):
                slot = calls.setdefault(tc["index"], {"id": tc.get("id"), "name": "", "arguments": ""})
                slot["name"] += tc.get("function", {}).get("name") or ""
                slot["arguments"] += tc.get("function", {}).get("arguments") or ""
            finish = c.get("finish_reason") or finish
    return {"msg": msg, "calls": [calls[i] for i in sorted(calls)], "usage": usage, "finish": finish}


results = []


def check(name, cond, detail=""):
    results.append(cond)
    print(f"{'PASS' if cond else 'FAIL'}  {name}" + (f"  ({detail})" if detail else ""), flush=True)


assert get("/health")["status"] == "ok"
check("/v1/models lists the model", get("/v1/models")["data"][0]["id"] == MODEL)
m = get("/metrics")
check("/metrics reports memory", {"active_gib", "cache_gib", "peak_gib", "budget_gib"} <= m.keys(), f"budget {m.get('budget_gib')}")


def status(path, data=None, headers=None):
    try:
        return urllib.request.urlopen(urllib.request.Request(URL + path, data, headers or {})).status
    except urllib.error.HTTPError as e:
        return e.code


check("foreign Host rejected (DNS rebinding)", status("/health", headers={"Host": "attacker.example"}) == 403)
check("non-JSON POST rejected (CSRF)", status("/v1/chat/completions", b"{}", {"Content-Type": "text/plain"}) == 415)

t = time.time()
r = chat([{"role": "user", "content": "What is 17 * 23? Answer with just the number."}], effort="low")
check("text answer", "391" in r["msg"]["content"], f"{time.time() - t:.1f}s incl. load, finish={r['finish']}")
check("reasoning streamed separately", r["msg"]["reasoning_content"] != "" and "<|" not in r["msg"]["content"])

tools = [{"type": "function", "function": {"name": "get_weather", "description": "Current weather for a city",
          "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}}]
convo = [{"role": "system", "content": "You are a helpful assistant. Use tools when they help."},
         {"role": "user", "content": "What's the weather in Paris and in Tokyo right now?"}]
r = chat(convo, tools=tools)
names = [c["name"] for c in r["calls"]]
args = [json.loads(c["arguments"]) for c in r["calls"]]
check("tool calls", r["finish"] == "tool_calls" and names and set(names) == {"get_weather"}, f"{args}")

convo.append({"role": "assistant", "content": r["msg"]["content"], "reasoning_content": r["msg"]["reasoning_content"],
              "tool_calls": [{"id": c["id"], "type": "function", "function": {"name": c["name"], "arguments": c["arguments"]}}
                             for c in r["calls"]]})
for c, a in zip(r["calls"], args):
    convo.append({"role": "tool", "tool_call_id": c["id"], "content": json.dumps({"city": a.get("city"), "temp_c": 18})})
first_prompt = r["usage"]["prompt_tokens"]
r = chat(convo, tools=tools)
cached = r["usage"]["prompt_tokens_details"]["cached_tokens"]
check("prefix cache reused on the next turn", cached >= first_prompt - 8, f"{cached}/{r['usage']['prompt_tokens']} cached")
check("tool results used", "18" in r["msg"]["content"], r["msg"]["content"][:80].replace("\n", " "))

r = chat(convo, tools=tools, cache_prompt=False)
check("cache_prompt:false starts cold", r["usage"]["prompt_tokens_details"]["cached_tokens"] == 0)

try:
    from PIL import Image, ImageDraw
except ImportError:
    print("SKIP  image test (Pillow not importable)")
else:
    img = Image.new("RGB", (640, 480), "white")
    ImageDraw.Draw(img).ellipse((170, 90, 470, 390), fill=(220, 20, 20))
    buf = io.BytesIO(); img.save(buf, "PNG")
    uri = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()
    r = chat([{"role": "user", "content": [{"type": "text", "text": "What shape is in the image, and what color? One short sentence."},
                                           {"type": "image_url", "image_url": {"url": uri}}]}])
    out = r["msg"]["content"].lower()
    check("image understood", "red" in out and "circle" in out, out[:80])

r = chat([{"role": "user", "content": "hello " * 90000}])
check("context overflow -> 400", r.get("error") == 400 and "context size" in r["body"]["error"]["message"])

print(f"\n{sum(results)}/{len(results)} passed")
sys.exit(0 if all(results) else 1)
