import os
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from hardsub_extractor.ocr.line_splitter import split_subtitle_lines
from hardsub_extractor.ocr.opencv_recognizer import (
    ctc_decode_logits,
    load_ppocr_dictionary,
    preprocess_ppocr_line,
)


def font_path():
    windir = Path(os.environ.get("WINDIR", r"C:\\Windows"))
    candidates = (
        windir / "Fonts" / "malgun.ttf",
        windir / "Fonts" / "malgunsl.ttf",
        windir / "Fonts" / "gulim.ttc",
        windir / "Fonts" / "batang.ttc",
        Path("/usr/share/fonts/truetype/nanum/NanumGothic.ttf"),
        Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"),
        Path("/usr/share/fonts/truetype/noto/NotoSans-Regular.ttf"),
    )
    for p in candidates:
        if p.exists():
            return str(p)
    raise RuntimeError("Korean test font missing")


def render(lines):
    img = Image.new("RGB", (640, 180), (64, 72, 85))
    d = ImageDraw.Draw(img)
    f = ImageFont.truetype(font_path(), 31)
    heights = []
    for line in lines:
        b = d.textbbox((0, 0), line, font=f, stroke_width=3)
        heights.append(b[3] - b[1])
    total = sum(heights) + 10 * (len(lines) - 1)
    y = (180 - total) // 2
    for line, hh in zip(lines, heights):
        b = d.textbbox((0, 0), line, font=f, stroke_width=3)
        x = (640 - (b[2] - b[0])) // 2
        d.text((x, y), line, font=f, fill="white", stroke_width=3, stroke_fill="black")
        y += hh + 10
    return cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)


class OpenCVRecognizerCoreTests(unittest.TestCase):
    def test_dictionary_ctc_layout(self):
        with TemporaryDirectory() as td:
            p = Path(td) / "dict.txt"
            p.write_text("가\n나\n다\n", encoding="utf-8")
            chars = load_ppocr_dictionary(p)
            self.assertEqual(chars[:5], ["blank", "가", "나", "다", " "])

            logits = np.zeros((1, 7, 5), dtype=np.float32)
            ids = [0, 1, 1, 0, 2, 2, 3]
            for t, idx in enumerate(ids):
                logits[0, t, idx] = 9.0
            text, confidence = ctc_decode_logits(logits, chars)
            self.assertEqual(text, "가나다")
            self.assertGreater(confidence, 0.95)

    def test_preprocess_shape_and_range(self):
        image = np.full((40, 250, 3), 255, dtype=np.uint8)
        blob = preprocess_ppocr_line(image)
        self.assertEqual(blob.shape[0:3], (1, 3, 48))
        self.assertEqual(blob.shape[3] % 32, 0)
        self.assertGreaterEqual(blob.shape[3], 320)
        self.assertLessEqual(float(blob.max()), 1.01)
        self.assertGreaterEqual(float(blob.min()), -1.01)

    def test_split_one_line(self):
        crops = split_subtitle_lines(render(["괜찮아"]))
        self.assertEqual(len(crops), 1)
        self.assertGreater(crops[0].image.shape[1], 80)

    def test_split_two_lines(self):
        crops = split_subtitle_lines(render(["두 줄 자막도", "정확히 잡아야 해"]))
        self.assertEqual(len(crops), 2)
        self.assertLess(crops[0].y0, crops[1].y0)


if __name__ == "__main__":
    unittest.main()
