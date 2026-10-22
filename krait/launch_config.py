#!/usr/bin/env python3
"""krait/launch_config.py - krait/krait runs this before the server: it checks the install (Python packages, the
built binaries, the model files, the tokenizer pack, the API key and the bind address) with a clear message for each
problem, and writes the config the server runs with: krait/config.json with the model paths under KRAIT_MODELS, the
machine's own engine flags (KRAIT_LOCAL_CONFIG / KRAIT_ENGINE_ARGS) and the thread count filled in.
  launch_config.py REPO KRAIT_ROOT OUT_JSON [server options...]"""
from __future__ import annotations

import ipaddress
import json
import os
import re
import subprocess
import sys
from pathlib import Path


def fail(msg: str) -> None:
    print(f"krait: {msg}", file=sys.stderr)
    sys.exit(2)


def physical_cores() -> int:
    try:
        out = subprocess.run(["lscpu", "-p=socket,core"], capture_output=True, text=True, timeout=10).stdout
        cores = {line for line in out.splitlines() if line and not line.startswith("#")}
        if cores:
            return len(cores)
    except (OSError, subprocess.SubprocessError):
        pass
    return max(1, (os.cpu_count() or 2) // 2)


def arg_value(args: list[str], flag: str) -> str | None:
    """The last value given for flag (--flag v or --flag=v), or None."""
    val = None
    for i, a in enumerate(args):
        if a == flag and i + 1 < len(args):
            val = args[i + 1]
        elif a.startswith(flag + "="):
            val = a.split("=", 1)[1]
    return val


def merge_flags(args: list[str], extra: list[str]) -> list[str]:
    """extra flags over args: a flag already in args gets the new value, a new one is appended."""
    out = list(args)
    i = 0
    while i < len(extra):
        flag = extra[i]
        if not flag.startswith("--"):
            fail(f"engine flags must start with --: {flag!r} (KRAIT_ENGINE_ARGS / the local config)")
        val = extra[i + 1] if i + 1 < len(extra) and not extra[i + 1].startswith("--") else None
        i += 2 if val is not None else 1
        if flag in out:
            j = out.index(flag)
            if val is not None:
                if j + 1 < len(out) and not out[j + 1].startswith("--"):
                    out[j + 1] = val
                else:
                    out.insert(j + 1, val)
        else:
            out += [flag] + ([val] if val is not None else [])
    return out


def is_loopback(host: str) -> bool:
    if host.lower() == "localhost":
        return True
    try:
        return ipaddress.ip_address(host.strip("[]")).is_loopback
    except ValueError:
        return False


def shards_of(first: Path) -> list[Path]:
    m = re.match(r"(.*-)(\d+)-of-(\d+)\.gguf$", first.name)
    if not m:
        return [first]
    width, n = len(m.group(2)), int(m.group(3))
    return [first.with_name(f"{m.group(1)}{i:0{width}d}-of-{m.group(3)}.gguf") for i in range(1, n + 1)]


def main() -> None:
    repo, root, out = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
    server_args = sys.argv[4:]
    missing = []
    for mod in ("jinja2", "regex", "numpy", "yaml"):   # what the server needs to start (requirements.txt has the rest)
        try:
            __import__(mod)
        except ImportError:
            missing.append(mod)
    if missing:
        fail(f"Python packages missing ({', '.join(missing)}) for {sys.executable}: run\n"
             f"  python3 -m venv {repo}/.venv && {repo}/.venv/bin/pip install -r {repo}/requirements.txt\n"
             "  (krait/krait uses that .venv by itself; or point KRAIT_PYTHON at a Python that has them)")
    cfg_path = Path(os.environ.get("KRAIT_CONFIG") or repo / "krait" / "config.json")
    try:
        cfg = json.loads(cfg_path.read_text())
    except (OSError, ValueError) as e:
        fail(f"cannot read the config {cfg_path}: {e}")
    models = Path(os.environ.get("KRAIT_MODELS") or root / "models").expanduser().absolute()

    def mpath(p: str) -> str:
        return str(models / p[len("models/"):]) if isinstance(p, str) and p.startswith("models/") else p

    args = [mpath(a) for a in cfg["args"]]
    extra: list[str] = []
    local = Path(os.environ.get("KRAIT_LOCAL_CONFIG") or root / "krait.local.json")
    if local.is_file():
        try:
            lc = json.loads(local.read_text())
        except ValueError as e:
            fail(f"cannot read the local config {local}: {e}")
        extra += [str(x) for x in lc.get("engine_args", [])]
        for key in ("max_queue", "queue_timeout", "max_tokens_cap", "request_timeout"):
            if key in lc:
                cfg[key] = lc[key]
    elif os.environ.get("KRAIT_LOCAL_CONFIG"):
        fail(f"KRAIT_LOCAL_CONFIG names {local}, which does not exist")
    extra += os.environ.get("KRAIT_ENGINE_ARGS", "").split()
    args = merge_flags(args, extra)
    threads = os.environ.get("KRAIT_THREADS") or str(physical_cores())
    if not threads.isdigit() or int(threads) < 1:
        fail(f"KRAIT_THREADS must be a positive whole number, not {threads!r}")
    args = [threads if a == "auto" and i > 0 and args[i - 1] == "--threads" else a for i, a in enumerate(args)]
    cfg["args"] = args
    vision = cfg.get("vision")
    if vision:
        for k in ("mmproj", "model"):
            if k in vision:
                vision[k] = mpath(vision[k])
        if vision.get("threads") == "auto":
            vision["threads"] = int(threads)
    if os.environ.get("KRAIT_IMAGE_FILES") == "1":
        cfg["image_files"] = True
    if os.environ.get("KRAIT_IMAGE_URLS") == "1":
        cfg["image_urls"] = True

    exe = root / cfg["exe"]
    if not exe.is_file():
        fail(f"{exe} is missing: build it with  KRAIT_ROOT={root} {repo}/krait/build.sh")
    if vision and not (root / vision["exe"]).is_file():
        fail(f"{root / vision['exe']} is missing: build it with  KRAIT_ROOT={root} {repo}/krait/build.sh")
    first = Path(args[args.index("--model") + 1])
    files = shards_of(first) + ([Path(vision["mmproj"])] if vision and vision.get("mmproj") else [])
    absent = [str(p) for p in files if not p.is_file()]
    if absent:
        fail("model files missing:\n  " + "\n  ".join(absent) +
             f"\n  set KRAIT_MODELS to the directory that holds UD-Q4_K_XL/ and mmproj-BF16.gguf (now {models})")
    tok = root / cfg["tokenizer"] / "tokenizer.json"
    if not tok.is_file():
        fail(f"the tokenizer pack is missing ({tok}); make it with\n"
             f"  {sys.executable} {repo}/tools/strata_tokenizer.py --gguf {first} --out {(root / cfg['tokenizer']).parent}")

    host = arg_value(server_args, "--host") or os.environ.get("HOST") or cfg.get("host") or "127.0.0.1"
    key = arg_value(server_args, "--api-key") or os.environ.get("KRAIT_API_KEY") or os.environ.get("STRATA_API_KEY") \
        or cfg.get("api_key") or ""
    if isinstance(key, list):
        key = ",".join(str(k) for k in key)
    keys = [k.strip() for k in str(key).split(",") if k.strip()]
    if any(k.lower().startswith("change-me") for k in keys):
        fail("the API key is still a placeholder (change-me...): set a real one, e.g. KRAIT_API_KEY=$(openssl rand -hex 16)")
    no_key = not keys or (len(keys) == 1 and keys[0].lower() == "none") or "--no-api-key" in server_args
    if not is_loopback(host) and no_key and os.environ.get("KRAIT_ALLOW_OPEN") != "1":
        fail(f"HOST={host} serves on the network but no API key is set: set KRAIT_API_KEY (or run with "
             "KRAIT_ALLOW_OPEN=1 to serve without one, which lets anyone who can reach the port use it)")
    out.write_text(json.dumps(cfg, indent=2) + "\n")


if __name__ == "__main__":
    main()
