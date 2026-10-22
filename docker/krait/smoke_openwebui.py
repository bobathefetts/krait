#!/usr/bin/env python3
"""Smoke test of the compose stack through Open WebUI's API (Python standard library only): makes (or signs in) an
account, lists the models, chats (plain and streaming), sends a picture, and does a document search (RAG) round trip:
uploads a text file, waits until Open WebUI has split and embedded it (through the llama.cpp embedder), asks a
question only that file answers, and checks the answer and the sources Open WebUI reports.
  KRAIT_SMOKE_PASSWORD=<12+ chars> python3 smoke_openwebui.py [--url http://127.0.0.1:3095] [--out DIR] [--image shapes.png]"""
import argparse, base64, json, os, sys, time, urllib.error, urllib.request, uuid
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--url", default="http://127.0.0.1:3095")
ap.add_argument("--email", default="smoke@krait.local")
ap.add_argument("--password", default=os.environ.get("KRAIT_SMOKE_PASSWORD"),
                help="the test account's password (or KRAIT_SMOKE_PASSWORD), at least 12 characters")
ap.add_argument("--out", default=".")
ap.add_argument("--image", help="a picture to ask about (optional)")
ap.add_argument("--image-question", default="What color is the circle? Answer in one word.")
ap.add_argument("--image-expect", default="blue")
ap.add_argument("--timeout", type=float, default=900)
a = ap.parse_args()
if not a.password or len(a.password) < 12:
    sys.exit("set --password or KRAIT_SMOKE_PASSWORD (at least 12 characters): the smoke test makes a real account")
out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
U = a.url.rstrip("/"); token = None; log = []

def req(method, path, body=None, raw=None, ctype=None, stream=False, timeout=600):
    h = {"Authorization": f"Bearer {token}"} if token else {}
    data = None
    if body is not None:
        data = json.dumps(body).encode(); h["Content-Type"] = "application/json"
    elif raw is not None:
        data = raw; h["Content-Type"] = ctype
    r = urllib.request.Request(U + path, data=data, method=method, headers=h)
    t = time.monotonic()
    try:
        with urllib.request.urlopen(r, timeout=timeout) as resp:
            if stream:
                lines = [l.decode("utf-8", "replace").rstrip("\n") for l in resp]
                res = {"status": resp.status, "lines": lines}
            else:
                txt = resp.read().decode("utf-8", "replace")
                try: res = {"status": resp.status, "json": json.loads(txt)}
                except ValueError: res = {"status": resp.status, "text": txt[:4000]}
    except urllib.error.HTTPError as e:
        res = {"status": e.code, "text": e.read().decode("utf-8", "replace")[:4000]}
    res["wall_s"] = round(time.monotonic() - t, 3)
    log.append({"method": method, "path": path, "status": res["status"], "wall_s": res["wall_s"]})
    return res

def content_of(r):
    try: return r["json"]["choices"][0]["message"]["content"] or ""
    except (KeyError, IndexError, TypeError): return ""

results = {"url": U, "cases": []}
def case(name, ok, **kw):
    results["cases"].append({"name": name, "ok": bool(ok), **kw}); print(("PASS " if ok else "FAIL ") + name, flush=True)

t0 = time.monotonic()
while True:
    try:
        with urllib.request.urlopen(U + "/health", timeout=5) as r:
            if r.status == 200: break
    except Exception: pass
    if time.monotonic() - t0 > a.timeout: sys.exit("open webui not up")
    time.sleep(3)
r = req("POST", "/api/v1/auths/signup", {"name": "Smoke Test", "email": a.email, "password": a.password})
if r["status"] != 200:
    r = req("POST", "/api/v1/auths/signin", {"email": a.email, "password": a.password})
token = (r.get("json") or {}).get("token")
case("account", token, status=r["status"], role=(r.get("json") or {}).get("role"))
if not token: (out / "smoke.json").write_text(json.dumps(results, indent=1)); sys.exit(1)

model = None
while time.monotonic() - t0 < a.timeout:   # Open WebUI lists Krait's model once Krait answers /v1/models
    r = req("GET", "/api/models")
    ids = [m.get("id") for m in (r.get("json") or {}).get("data", [])]
    if ids: model = ids[0]; break
    time.sleep(5)
case("models", model, models=ids if model else None)
if not model: (out / "smoke.json").write_text(json.dumps(results, indent=1)); sys.exit(1)

r = req("POST", "/api/chat/completions", {"model": model, "stream": False, "temperature": 0, "max_tokens": 400,
        "messages": [{"role": "user", "content": "What is the capital of France? Answer with the city name only."}]})
c = content_of(r); case("chat", r["status"] == 200 and "paris" in c.lower(), status=r["status"], content=c, wall_s=r["wall_s"],
                        usage=(r.get("json") or {}).get("usage"))

r = req("POST", "/api/chat/completions", {"model": model, "stream": True, "temperature": 0, "max_tokens": 400,
        "messages": [{"role": "user", "content": "Count from 1 to 5, separated by commas."}]}, stream=True)
chunks, text, reasoning = 0, "", ""
for l in r.get("lines", []):
    if l.startswith("data: ") and l[6:].strip() != "[DONE]":
        try: d = json.loads(l[6:])
        except ValueError: continue
        for ch in d.get("choices") or []:
            dl = ch.get("delta") or {}; text += dl.get("content") or ""; reasoning += dl.get("reasoning_content") or ""
            chunks += 1
case("chat_stream", r["status"] == 200 and chunks > 3 and "1" in text and "5" in text, status=r["status"], chunks=chunks,
     content=text, reasoning_chars=len(reasoning), wall_s=r["wall_s"])

if a.image:
    p = Path(a.image); mime = "image/png" if p.suffix.lower() == ".png" else "image/jpeg"
    r = req("POST", "/api/chat/completions", {"model": model, "stream": False, "temperature": 0, "max_tokens": 600,
            "messages": [{"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": f"data:{mime};base64," + base64.b64encode(p.read_bytes()).decode()}},
                {"type": "text", "text": a.image_question}]}]})
    c = content_of(r); case("image", r["status"] == 200 and a.image_expect.lower() in c.lower(), status=r["status"], content=c,
                            wall_s=r["wall_s"], usage=(r.get("json") or {}).get("usage"))

# RAG: the fact is made up, so only the uploaded document can answer it; the other sections are filler on other topics
code = "KESTREL-" + uuid.uuid4().hex[:6].upper()
filler = [
    "Section {i}. The cafeteria on the third floor serves soup on Mondays and Thursdays. The menu changes every season, "
    "and the vegetarian option is listed on the board next to the elevators. Questions about catering for meetings go to "
    "the facilities desk, which answers within two working days. " * 3
] 
sections = [filler[0].format(i=i) for i in range(1, 9)]
fact = (f"Section 5. Calibration of the test rig. The quarterly calibration of the vibration test rig uses a reference mass "
        f"of 7413 grams. It is carried out by technician Ilona Varga on the second Tuesday of the quarter, and the "
        f"calibration record is filed under the code {code}. ")
sections.insert(4, fact * 2)
doc = "Facilities handbook (internal)\n\n" + "\n\n".join(sections) + "\n"
(out / "rag_document.txt").write_text(doc)
bnd = "----krait" + uuid.uuid4().hex
body = (f"--{bnd}\r\nContent-Disposition: form-data; name=\"file\"; filename=\"facilities_handbook.txt\"\r\n"
        f"Content-Type: text/plain\r\n\r\n").encode() + doc.encode() + f"\r\n--{bnd}--\r\n".encode()
r = req("POST", "/api/v1/files/", raw=body, ctype=f"multipart/form-data; boundary={bnd}")
fid = (r.get("json") or {}).get("id")
case("rag_upload", r["status"] == 200 and fid, status=r["status"], file_id=fid, doc_chars=len(doc), text=r.get("text"))
st = None
if fid:
    for _ in range(120):
        s = req("GET", f"/api/v1/files/{fid}/process/status")
        st = (s.get("json") or {}).get("status") if s["status"] == 200 else f"http {s['status']}"
        if st in ("completed", "failed") or s["status"] == 404: break
        time.sleep(2)
    case("rag_processed", st == "completed", status=st, detail=s.get("json") or s.get("text"))
    q = "According to the facilities handbook, what reference mass is used to calibrate the vibration test rig, who does it, and what is the record code?"
    r = req("POST", "/api/chat/completions", {"model": model, "stream": False, "temperature": 0, "max_tokens": 800,
            "messages": [{"role": "user", "content": q}], "files": [{"type": "file", "id": fid}]})
    j = r.get("json") or {}; c = content_of(r)
    srcs = j.get("sources") or []
    flat = json.dumps(srcs)
    case("rag_answer", r["status"] == 200 and "7413" in c.replace(",", "").replace(" ", "") and "varga" in c.lower() and code.lower() in c.lower(),
         status=r["status"], question=q, content=c, code=code, n_sources=len(srcs),
         sources_contain_fact=("7413" in flat), wall_s=r["wall_s"], usage=j.get("usage"), raw_keys=sorted(j.keys()))
results["requests"] = log
(out / "smoke.json").write_text(json.dumps(results, indent=1))
bad = [c["name"] for c in results["cases"] if not c["ok"]]
print("failed:", bad if bad else "none")
sys.exit(1 if bad else 0)
