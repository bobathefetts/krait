"""serve/test_krait_launch.py - Krait's launcher checks (krait/launch_config.py) and the server's picture-source and
request-deadline guards, without the model or a GPU.

    python -m unittest serve.test_krait_launch
"""
from __future__ import annotations

import base64
import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path

from serve import server
from serve.server import IMAGE_SOURCES, RequestDeadline, Vision, public_url_host, _CheckedRedirect

REPO = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("launch_config", REPO / "krait" / "launch_config.py")
lc = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(lc)


class ImageSources(unittest.TestCase):
    def setUp(self):
        self.saved = dict(IMAGE_SOURCES)

    def tearDown(self):
        IMAGE_SOURCES.update(self.saved)

    def test_non_public_addresses_refused(self):
        for url in ("http://127.0.0.1/x.png", "http://10.1.2.3/x.png", "http://192.168.0.1/x.png",
                    "http://169.254.169.254/latest/meta-data", "http://[::1]/x.png", "http://[::ffff:127.0.0.1]/x.png",
                    "http://[fe80::1]/x.png", "http://0.0.0.0/x.png", "http://224.0.0.1/x.png", "http://localhost/x.png",
                    "ftp://8.8.8.8/x.png", "file:///etc/passwd"):
            with self.subTest(url=url), self.assertRaises(ValueError):
                public_url_host(url)

    def test_public_address_allowed(self):
        public_url_host("http://8.8.8.8/x.png")          # a numeric address: no DNS needed

    def test_redirect_to_private_refused(self):
        with self.assertRaises(ValueError):
            _CheckedRedirect().redirect_request(None, None, 302, "Found", {}, "http://127.0.0.1:8095/v1/models")

    def test_off_by_default_in_config(self):
        cfg = json.loads((REPO / "krait" / "config.json").read_text())
        self.assertIs(cfg["image_files"], False)
        self.assertIs(cfg["image_urls"], False)

    def test_files_and_urls_refused_when_off(self):
        IMAGE_SOURCES.update(files=False, urls=False)
        with tempfile.NamedTemporaryFile(suffix=".png") as f:
            with self.assertRaises(ValueError) as e:
                Vision.load(f.name)
            self.assertIn("--image-files", str(e.exception))
        with self.assertRaises(ValueError) as e:
            Vision.load("http://8.8.8.8/x.png")
        self.assertIn("--image-urls", str(e.exception))
        self.assertEqual(Vision.load("data:image/png;base64," + base64.b64encode(b"abc").decode()), b"abc")

    def test_files_when_on(self):
        IMAGE_SOURCES.update(files=True)
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            f.write(b"xyz")
        try:
            self.assertEqual(Vision.load(f.name), b"xyz")
        finally:
            os.unlink(f.name)


class Deadline(unittest.TestCase):
    def test_deadline(self):
        d = RequestDeadline(0.05)
        self.assertFalse(d.is_set())
        time.sleep(0.1)
        self.assertTrue(d.is_set())
        n = RequestDeadline(None)
        self.assertFalse(n.is_set())
        n.set()
        self.assertTrue(n.is_set())


class LaunchConfig(unittest.TestCase):
    def test_merge_flags(self):
        base = ["--model", "m", "--threads", "auto", "--pin", "--batch", "4"]
        self.assertEqual(lc.merge_flags(base, ["--batch", "2", "--vram-limit-mib", "29346", "--no-mmap"]),
                         ["--model", "m", "--threads", "auto", "--pin", "--batch", "2", "--vram-limit-mib", "29346",
                          "--no-mmap"])

    def test_loopback(self):
        for h in ("127.0.0.1", "localhost", "::1", "[::1]", "127.0.0.2"):
            self.assertTrue(lc.is_loopback(h), h)
        for h in ("0.0.0.0", "192.168.1.5", "::", "example.com"):
            self.assertFalse(lc.is_loopback(h), h)

    def test_shards(self):
        s = lc.shards_of(Path("/m/X-00001-of-00003.gguf"))
        self.assertEqual([p.name for p in s], ["X-00001-of-00003.gguf", "X-00002-of-00003.gguf", "X-00003-of-00003.gguf"])

    def run_lc(self, root, env=None, *args):
        e = {k: v for k, v in os.environ.items() if not k.startswith(("KRAIT_", "STRATA_API_KEY", "HOST"))}
        e.update(env or {})
        return subprocess.run([sys.executable, str(REPO / "krait" / "launch_config.py"), str(REPO), str(root),
                               str(Path(root) / "run.json"), *args], capture_output=True, text=True, env=e, timeout=60)

    def make_root(self, d):
        root = Path(d)
        for b in ("build/krait/krait-engine", "build/vision/krait-vision", "pack/nemotron/tokenizer/tokenizer.json"):
            (root / b).parent.mkdir(parents=True, exist_ok=True)
            (root / b).write_text("x")
        m = root / "models" / "UD-Q4_K_XL"
        m.mkdir(parents=True)
        for i in (1, 2, 3):
            (m / f"NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-UD-Q4_K_XL-0000{i}-of-00003.gguf").write_text("x")
        (root / "models" / "mmproj-BF16.gguf").write_text("x")
        return root

    def test_missing_binary(self):
        with tempfile.TemporaryDirectory() as d:
            r = self.run_lc(d)
            self.assertEqual(r.returncode, 2)
            self.assertIn("krait/build.sh", r.stderr)

    def test_missing_models_names_variable(self):
        with tempfile.TemporaryDirectory() as d:
            root = self.make_root(d)
            r = self.run_lc(root, {"KRAIT_MODELS": str(root / "nowhere")})
            self.assertEqual(r.returncode, 2)
            self.assertIn("KRAIT_MODELS", r.stderr)

    def test_ok_and_local_override(self):
        with tempfile.TemporaryDirectory() as d:
            root = self.make_root(d)
            (root / "krait.local.json").write_text(json.dumps({"engine_args": ["--vram-limit-mib", "29346"]}))
            r = self.run_lc(root, {"KRAIT_THREADS": "7", "KRAIT_ENGINE_ARGS": "--batch 2"})
            self.assertEqual(r.returncode, 0, r.stderr)
            cfg = json.loads((root / "run.json").read_text())
            a = cfg["args"]
            self.assertEqual(a[a.index("--threads") + 1], "7")
            self.assertEqual(a[a.index("--batch") + 1], "2")
            self.assertEqual(a[a.index("--vram-limit-mib") + 1], "29346")
            self.assertTrue(a[a.index("--model") + 1].startswith(str(root / "models")))
            self.assertEqual(cfg["vision"]["threads"], 7)

    def test_shipped_config_has_no_vram_cap(self):
        cfg = json.loads((REPO / "krait" / "config.json").read_text())
        self.assertNotIn("--vram-limit-mib", cfg["args"])

    def test_key_rules(self):
        with tempfile.TemporaryDirectory() as d:
            root = self.make_root(d)
            r = self.run_lc(root, {}, "--host", "0.0.0.0")
            self.assertEqual(r.returncode, 2)
            self.assertIn("no API key", r.stderr)
            r = self.run_lc(root, {"KRAIT_API_KEY": "change-me"}, "--host", "127.0.0.1")
            self.assertEqual(r.returncode, 2)
            self.assertIn("placeholder", r.stderr)
            r = self.run_lc(root, {"KRAIT_API_KEY": "k" * 32}, "--host", "0.0.0.0")
            self.assertEqual(r.returncode, 0, r.stderr)
            r = self.run_lc(root, {"KRAIT_ALLOW_OPEN": "1"}, "--host", "0.0.0.0")
            self.assertEqual(r.returncode, 0, r.stderr)


if __name__ == "__main__":
    unittest.main()
