"""serve/test_nemotron_vision.py - the server's picture path for Nemotron 3.5 Super (arch nemotron_h_moe), without the
model: <image> -> <img> + one <image> per picture row + </img> (llama.cpp mtmd's nemotron_omni layout), the count
checks, and the vision section's flash_attn key reaching the encoder's command line.

    python -m unittest serve.test_nemotron_vision
"""
from __future__ import annotations

import unittest

from serve.server import NEMOTRON_IMAGE, Vision, expand_nemotron_pictures

IMG, BEG, END = 18, [19], [20]   # <image>, <img>, </img> in Nemotron 3.5 Super's vocabulary


class NemotronPictures(unittest.TestCase):
    def test_marker(self):
        self.assertEqual(NEMOTRON_IMAGE, "<image>")

    def test_one_picture(self):
        ids = [1, 2, IMG, 3]
        self.assertEqual(expand_nemotron_pictures(ids, IMG, BEG, END, [3]), [1, 2, 19, IMG, IMG, IMG, 20, 3])

    def test_two_pictures_in_order(self):
        out = expand_nemotron_pictures([IMG, 5, IMG], IMG, BEG, END, [2, 1])
        self.assertEqual(out, [19, IMG, IMG, 20, 5, 19, IMG, 20])

    def test_no_pictures(self):
        self.assertEqual(expand_nemotron_pictures([7, 8], IMG, BEG, END, []), [7, 8])

    def test_counts_must_match(self):
        with self.assertRaises(ValueError):
            expand_nemotron_pictures([IMG, IMG], IMG, BEG, END, [4])     # a marker without a picture
        with self.assertRaises(ValueError):
            expand_nemotron_pictures([IMG], IMG, BEG, END, [4, 4])       # a picture without a marker

    def test_rows_of_real_sizes(self):
        # 640x480 -> 300 rows and 640x400 -> 260 rows (measured with the nemotron_omni projector, mmproj-BF16)
        out = expand_nemotron_pictures([1, IMG, 2], IMG, BEG, END, [300])
        self.assertEqual(len(out), 2 + 300 + 2)
        self.assertEqual(out[1], 19)
        self.assertEqual(out[-2], 20)


class VisionArgs(unittest.TestCase):
    CFG = {"exe": "strata-vision", "mmproj": "mmproj.gguf", "model": "model.gguf", "threads": 16}

    def test_flash_attn_passed(self):
        args = Vision.reader_args(dict(self.CFG, flash_attn="on"))
        self.assertEqual(args[args.index("--flash-attn") + 1], "on")
        self.assertEqual(args[args.index("--threads") + 1], "16")

    def test_flash_attn_absent_or_bad(self):
        self.assertNotIn("--flash-attn", Vision.reader_args(dict(self.CFG)))
        self.assertNotIn("--flash-attn", Vision.reader_args(dict(self.CFG, flash_attn="yes")))


if __name__ == "__main__":
    unittest.main()
