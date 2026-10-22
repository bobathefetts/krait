"""serve/test_nemotron_frontend.py - the server pieces the Nemotron 3.5 Super port (arch nemotron_h_moe) changed, without
the model: the reasoning-effort mapping onto the template's medium_effort, the model card sampling and the family name.

    python -m pytest serve/test_nemotron_frontend.py
"""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from serve.frontend import ChatTemplate, effort_kwargs
from serve.server import FAMILY_MODEL_NAMES, MODEL_CARD_SAMPLING

# the part of Nemotron 3.5 Super's template that the effort levels touch (the GGUF's tokenizer.chat_template)
TEMPLATE = """{%- set enable_thinking = enable_thinking if enable_thinking is defined else True %}
{%- set medium_effort = medium_effort if medium_effort is defined else False %}
{%- for message in messages %}{{- '<|im_start|>' + message.role + '\\n' }}
{%- if message.role == "user" and loop.last and medium_effort %}{{- message.content + '\\n\\n{reasoning effort: efficient}' }}
{%- else %}{{- message.content }}{%- endif %}{{- '<|im_end|>\\n' }}{%- endfor %}
{%- if add_generation_prompt %}{%- if enable_thinking %}{{- '<|im_start|>assistant\\n<think>\\n' }}
{%- else %}{{- '<|im_start|>assistant\\n<think></think>' }}{%- endif %}{%- endif %}"""


class NemotronFrontend(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        p = Path(self.dir.name) / "chat_template.jinja"
        p.write_text(TEMPLATE, encoding="utf-8")
        self.t = ChatTemplate(str(p))
        self.m = [{"role": "user", "content": "Hi"}]

    def tearDown(self):
        self.dir.cleanup()

    def render(self, effort):
        return self.t.render(self.m, add_generation_prompt=True, **effort_kwargs(effort))

    def test_efforts(self):
        self.assertTrue(self.render(None).endswith("Hi<|im_end|>\n<|im_start|>assistant\n<think>\n"))
        self.assertTrue(self.render("high").endswith("Hi<|im_end|>\n<|im_start|>assistant\n<think>\n"))
        self.assertTrue(self.render("none").endswith("<|im_start|>assistant\n<think></think>"))
        for level in ("low", "medium"):
            self.assertIn("Hi\n\n{reasoning effort: efficient}<|im_end|>", self.render(level))

    def test_explicit_medium_effort_wins(self):
        out = self.t.render(self.m, add_generation_prompt=True, reasoning_effort="low", medium_effort=False)
        self.assertNotIn("{reasoning effort: efficient}", out)

    def test_card_and_family(self):
        self.assertEqual(FAMILY_MODEL_NAMES["nemotron_h_moe"], "nemotron-3.5-super-120b-a12b")
        for mode in ("thinking", "instruct"):
            self.assertEqual(MODEL_CARD_SAMPLING["nemotron_h_moe"][mode]["temperature"], 1.0)
            self.assertEqual(MODEL_CARD_SAMPLING["nemotron_h_moe"][mode]["top_p"], 0.95)


if __name__ == "__main__":
    unittest.main()
