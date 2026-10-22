#!/usr/bin/env python3
"""krait/bench_serve.py - the one-command benchmark of the Krait server on Nemotron 3.5 Super.

    krait/bench_serve.py [--url http://127.0.0.1:8095] [--out reports/serve_bench] [--max-tokens 256] [--conc 1,2,4]

Without --url it starts krait/krait (its own port, 8097), waits for /health and stops it at the end.
It sends the same six prompts every time (OpenAI /v1/chat/completions, streaming, temperature 0, thinking off, so every
run produces the same tokens), and measures:
  single  - one request at a time: time to first token (client clock), the engine's prompt and decode rates
            (timings.prompt_per_second / predicted_per_second: the engine's own DONE clocks) and MTP draft acceptance;
  conc N  - N requests at once (N = 1, 2, 4): aggregate tokens per second over the wall time from the first request
            sent to the last one finished, and each request's own rate.
Everything it measured goes to <out>/bench_serve.json, the answers to <out>/answers.jsonl, a table to stdout.
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import os
import signal
import statistics
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROMPTS = [
    "Explain how a Bloom filter works and when it gives false positives. Include a small worked example.",
    "Write a Python function that merges overlapping intervals, with a short explanation of its complexity.",
    "Summarize the causes of the 1929 stock market crash in a few paragraphs.",
    "What are the main differences between TCP and UDP? Give two use cases for each.",
    "Translate into French and then explain the grammar: 'The children would have gone to the park if it had not rained.'",
    "Describe the life cycle of a star like the Sun, from formation to its final state.",
]


def post_stream(url, body, timeout=1800):
    req = urllib.request.Request(url + "/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.monotonic()
    first, text, usage, timings, finish = None, [], None, None, None
    with urllib.request.urlopen(req, timeout=timeout) as r:
        for raw in r:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                break
            ev = json.loads(data)
            for ch in ev.get("choices") or []:
                d = ch.get("delta") or {}
                piece = (d.get("content") or "") + (d.get("reasoning_content") or "")
                if piece:
                    if first is None:
                        first = time.monotonic()
                    text.append(piece)
                if ch.get("finish_reason"):
                    finish = ch["finish_reason"]
            usage = ev.get("usage") or usage
            timings = ev.get("timings") or timings
    t1 = time.monotonic()
    return {"ttft_s": None if first is None else round(first - t0, 3), "wall_s": round(t1 - t0, 3),
            "text": "".join(text), "usage": usage, "timings": timings, "finish": finish}


def request(url, model, prompt, max_tokens):
    body = {"model": model, "stream": True, "stream_options": {"include_usage": True}, "temperature": 0,
            "max_tokens": max_tokens, "messages": [{"role": "user", "content": prompt}],
            "chat_template_kwargs": {"enable_thinking": False}}
    return post_stream(url, body)


def get_json(url, path, timeout=10):
    with urllib.request.urlopen(url + path, timeout=timeout) as r:
        return json.loads(r.read())


def wait_ready(url, proc, limit_s=900):
    t0 = time.monotonic()
    while time.monotonic() - t0 < limit_s:
        if proc is not None and proc.poll() is not None:
            raise SystemExit(f"the server exited ({proc.returncode}) before it was ready")
        try:
            h = get_json(url, "/health", timeout=5)
            if h.get("loaded", True) and h.get("status") == "ok":
                return h, round(time.monotonic() - t0, 1)
        except Exception:
            pass
        time.sleep(2)
    raise SystemExit("the server was not ready in time")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", help="a running server (default: start krait/krait on --port)")
    ap.add_argument("--port", type=int, default=8097)
    ap.add_argument("--out", default="reports/serve_bench")
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--conc", default="1,2,4")
    ap.add_argument("--model", default=None)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    proc, url = None, a.url
    if not url:
        url = f"http://127.0.0.1:{a.port}"
        logf = open(out / "server.log", "w")
        proc = subprocess.Popen([str(HERE / "krait")], env={**os.environ, "PORT": str(a.port)},
                                stdout=logf, stderr=subprocess.STDOUT, start_new_session=True)
    try:
        health, ready_s = wait_ready(url, proc)
        model = a.model or (get_json(url, "/v1/models").get("data") or [{}])[0].get("id")
        res = {"url": url, "health": health, "ready_s": ready_s if proc else None, "model": model,
               "max_tokens": a.max_tokens, "single": [], "conc": []}
        request(url, model, "Say hello.", 16)                      # warm-up, not counted
        answers = open(out / "answers.jsonl", "w")
        for i, p in enumerate(PROMPTS):
            r = request(url, model, p, a.max_tokens)
            t = r["timings"] or {}
            row = {"prompt": i, "ttft_s": r["ttft_s"], "wall_s": r["wall_s"], "finish": r["finish"],
                   "completion_tokens": (r["usage"] or {}).get("completion_tokens"),
                   "prompt_tokens": (r["usage"] or {}).get("prompt_tokens"),
                   "prompt_tok_s": t.get("prompt_per_second"), "decode_tok_s": t.get("predicted_per_second"),
                   "draft_n": t.get("draft_n"), "draft_n_accepted": t.get("draft_n_accepted")}
            res["single"].append(row)
            answers.write(json.dumps({"mode": "single", "prompt": i, "text": r["text"]}) + "\n")
            print(f"single p{i}: ttft {row['ttft_s']} s  prompt {row['prompt_tok_s']} tok/s  "
                  f"decode {row['decode_tok_s']} tok/s  tokens {row['completion_tokens']}", flush=True)
        for n in [int(x) for x in a.conc.split(",") if x.strip()]:
            jobs = [PROMPTS[k % len(PROMPTS)] for k in range(n)]
            t0 = time.monotonic()
            with cf.ThreadPoolExecutor(n) as ex:
                rs = list(ex.map(lambda p: request(url, model, p, a.max_tokens), jobs))
            wall = time.monotonic() - t0
            toks = sum((r["usage"] or {}).get("completion_tokens") or 0 for r in rs)
            row = {"n": n, "wall_s": round(wall, 3), "completion_tokens": toks, "agg_tok_s": round(toks / wall, 2),
                   "per_request_decode_tok_s": [(r["timings"] or {}).get("predicted_per_second") for r in rs],
                   "ttft_s": [r["ttft_s"] for r in rs]}
            res["conc"].append(row)
            for k, r in enumerate(rs):
                answers.write(json.dumps({"mode": f"conc{n}", "prompt": k % len(PROMPTS), "text": r["text"]}) + "\n")
            print(f"conc {n}: {toks} tokens in {wall:.1f} s = {row['agg_tok_s']} tok/s aggregate", flush=True)
        dec = [r["decode_tok_s"] for r in res["single"] if r["decode_tok_s"]]
        res["single_median_decode_tok_s"] = statistics.median(dec) if dec else None
        (out / "bench_serve.json").write_text(json.dumps(res, indent=1))
        print(f"single-stream median decode {res['single_median_decode_tok_s']} tok/s; written {out / 'bench_serve.json'}")
    finally:
        if proc is not None:
            os.killpg(proc.pid, signal.SIGTERM)
            try:
                proc.wait(timeout=120)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, signal.SIGKILL)


if __name__ == "__main__":
    main()
