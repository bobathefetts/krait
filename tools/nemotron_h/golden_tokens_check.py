"""tools/nemotron_h/golden_tokens_check.py - render the golden set with Strata's own chat-template path
(serve/frontend.py ChatTemplate, the GGUF's tokenizer.chat_template) and tokenizer (tools/strata_tokenizer.py, tekken
pre-tokenizer, ignore_merges, no BOS), then compare text and token ids with the llama.cpp reference
(bench/golden/rendered/<id>.txt and tokens/<id>.i32 in the port's workspace).

The messages are given to the template the way llama.cpp b11806 hands them to its jinja path (common/chat.cpp,
common_chat_msgs_to_json_oaicompat / common_chat_tools_to_json_oaicompat): OpenAI form, tool calls as
{"type": "function", "id", "function": {"name", "arguments"}} with the arguments as an object (this template iterates
them with |items), tools as {"type": "function", "function": {...}}; add_generation_prompt=true and enable_thinking per
prompt.  The rendered text is tokenized with parse_special (llama.cpp: add_special=false, parse_special=true).

Usage: python tools/nemotron_h/golden_tokens_check.py --model SHARD1.gguf --golden DIR [--out report.json]
Exit 0 only when every prompt is token-identical.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys
import tempfile

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE.parent.parent))
from strata_tokenizer import Tokenizer  # noqa: E402
from gguf_reader import GGUFFile  # noqa: E402
from serve.frontend import ChatTemplate  # noqa: E402


def to_oai(messages: list[dict]) -> list[dict]:
    out = []
    for m in messages:
        o = {"role": m["role"], "content": m.get("content", "")}
        if m.get("tool_calls"):
            o["tool_calls"] = [{"type": "function", "id": tc.get("id", ""),
                                "function": {"name": tc["name"], "arguments": tc["arguments"]}}
                               for tc in m["tool_calls"]]
        if "tool_call_id" in m:
            o["tool_call_id"] = m["tool_call_id"]
        if "name" in m:
            o["name"] = m["name"]
        out.append(o)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--golden", required=True)
    ap.add_argument("--out")
    a = ap.parse_args()
    gd = pathlib.Path(a.golden)
    md = GGUFFile(pathlib.Path(a.model)).metadata
    tok = Tokenizer.from_gguf(a.model)
    if md.get("tokenizer.ggml.add_bos_token"):
        raise SystemExit("this check assumes add_bos_token=False (llama-vocab.cpp:2918)")
    with tempfile.NamedTemporaryFile("w", suffix=".jinja", delete=False, encoding="utf-8") as f:
        f.write(md["tokenizer.chat_template"])
        tpl_path = f.name
    tpl = ChatTemplate(tpl_path)
    prompts = json.loads((gd / "prompts.json").read_text(encoding="utf-8"))["prompts"]
    rows, n_tok, n_match, n_ident, n_text = [], 0, 0, 0, 0
    for p in prompts:
        tools = [{"type": "function", "function": t} for t in p["tools"]] if p.get("tools") else None
        text = tpl.render(to_oai(p["messages"]), tools=tools, add_generation_prompt=True,
                          enable_thinking=bool(p.get("enable_thinking", False)))
        ref_text = (gd / "rendered" / (p["id"] + ".txt")).read_text(encoding="utf-8")
        ref = np.fromfile(gd / "tokens" / (p["id"] + ".i32"), dtype="<i4").tolist()
        ids = tok.encode(text, parse_special=True)
        same_text = text == ref_text
        k = next((i for i, (x, y) in enumerate(zip(ids, ref)) if x != y), min(len(ids), len(ref)))
        ident = ids == ref
        pos_match = sum(1 for x, y in zip(ids, ref) if x == y) if len(ids) == len(ref) else k
        n_tok += len(ref)
        n_match += pos_match
        n_ident += ident
        n_text += same_text
        row = {"id": p["id"], "text_identical": same_text, "n_ref": len(ref), "n_strata": len(ids),
               "identical": ident, "first_divergence": None if ident else k}
        if not same_text:
            c = next((i for i, (x, y) in enumerate(zip(text, ref_text)) if x != y), min(len(text), len(ref_text)))
            row["text_first_diff"] = c
            row["text_ctx_strata"] = text[max(0, c - 40):c + 40]
            row["text_ctx_ref"] = ref_text[max(0, c - 40):c + 40]
        if not ident and k < min(len(ids), len(ref)):
            row["tok_ctx_strata"] = [(i, tok.decode([i])) for i in ids[max(0, k - 3):k + 4]]
            row["tok_ctx_ref"] = [(i, tok.decode([i])) for i in ref[max(0, k - 3):k + 4]]
        rows.append(row)
        print("%-24s text %-5s tokens %6d/%6d %s" % (p["id"], same_text, len(ids), len(ref),
                                                    "IDENTICAL" if ident else "DIFF at %d" % k))
    summary = {"prompts": len(prompts), "text_identical": n_text, "token_identical_prompts": n_ident,
               "ref_tokens": n_tok, "matching_tokens": n_match, "pass": n_ident == len(prompts)}
    print(json.dumps(summary))
    if a.out:
        pathlib.Path(a.out).write_text(json.dumps({"summary": summary, "per_prompt": rows}, indent=1,
                                                  ensure_ascii=False), encoding="utf-8")
    return 0 if summary["pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
