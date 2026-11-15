"""Genuine pinned RapidOCR inference against reproducible multilingual subtitles.

This is a bounded *synthetic* OCR acceptance, not a real anime/video quality
claim. Runs using the same adapter shipped in the frozen executable, no video,
website, credentials or VPN access. Reports character error rate and timing.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from hardsub_extractor.models import Frame
from hardsub_extractor.ocr.rapidocr_adapter import RapidOCRAdapter
from hardsub_extractor.ocr.base import comparison_key

FIXTURES = {
    "ko": ("오늘은 정말 좋은 날이에요", "잠시만 기다려 주세요"),
    "en": ("Where are you going tomorrow?", "I will be there in five minutes."),
    "ja": ("今日は何をしていますか", "明日は学校へ行きます"),
    "zh": ("今天晚上一起去看电影吧", "今天晚上一起去看電影吧"),
}
WINDOWS_FONT_NAMES = {
    "ko": ("malgun.ttf", "malgunbd.ttf"),
    "en": ("arial.ttf", "segoeui.ttf"),
    "ja": ("meiryo.ttc", "msgothic.ttc", "YuGothM.ttc"),
    "zh": ("msyh.ttc", "simhei.ttf", "simsun.ttc", "malgun.ttf"),
}


def choose_font(language: str) -> Path:
    roots = (Path("C:/Windows/Fonts"), Path("C:/WINDOWS/Fonts"))
    for name in WINDOWS_FONT_NAMES[language]:
        for root in roots:
            candidate = root / name
            if candidate.is_file():
                return candidate
    raise RuntimeError("a permitted installed Windows font is unavailable for this language")


def edit_distance(a: str, b: str) -> int:
    previous = list(range(len(b) + 1))
    for row, left in enumerate(a, 1):
        current = [row]
        for col, right in enumerate(b, 1):
            current.append(min(
                previous[col] + 1,
                current[-1] + 1,
                previous[col - 1] + (left != right),
            ))
        previous = current
    return previous[-1]


def render_subtitle(text: str, font_file: Path) -> np.ndarray:
    """Render lower-middle white outlined subtitle; same BGR format as video."""
    width, height = 1280, 720
    image = Image.new("RGB", (width, height), (36, 43, 59))
    draw = ImageDraw.Draw(image)
    size = 58
    while True:
        font = ImageFont.truetype(str(font_file), size=size)
        bbox = draw.textbbox((0, 0), text, font=font, stroke_width=2)
        text_width = bbox[2] - bbox[0]
        if text_width <= width * 0.85 or size <= 30:
            break
        size -= 3
    if text_width > width * 0.90:
        raise RuntimeError("synthetic reference text does not fit OCR frame")
    x = (width - text_width) / 2 - bbox[0]
    y = 525 - bbox[1]
    draw.text(
        (x, y), text, font=font, fill="white",
        stroke_width=2, stroke_fill="black",
    )
    return cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)


def inspect(language: str, directory: Path, manifest: Path) -> list[dict]:
    model = RapidOCRAdapter(
        model_dir=str(directory), manifest_path=str(manifest),
        language=language,
    )
    font = choose_font(language)
    results = []
    for i, expected in enumerate(FIXTURES[language]):
        frame = Frame(pts=1.0+i, image=render_subtitle(expected, font))
        begin = time.perf_counter()
        observed = model.recognize(frame)
        elapsed = time.perf_counter() - begin
        truth_key = comparison_key(expected)
        observed_key = comparison_key(observed.text)
        cer = edit_distance(truth_key, observed_key) / max(1, len(truth_key))
        row = {
            "language": language,
            "case": i + 1,
            "model_verified": True,
            "cer": round(cer, 4),
            "recognized": bool(observed_key),
            "confidence": round(float(observed.confidence), 4),
            "elapsed_seconds": round(elapsed, 2),
            "reference_chars": len(truth_key),
            "hypothesis_chars": len(observed_key),
            # Synthetic data only; print codepoint-escaped output to diagnose
            # handwriting/spacing/CJK errors without exposing private videos.
            "expected": expected,
            "observed": observed.text,
        }
        print("HSE_MULTILANG_OCR " + json.dumps(row, ensure_ascii=True))
        results.append(row)
    return results


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", default=str(ROOT / "models" / "runtime"))
    ap.add_argument("--manifest", default=str(ROOT / "models" / "manifest.json"))
    ap.add_argument("--languages", default="ko,en,ja,zh")
    ap.add_argument("--max-cer", type=float, default=0.35)
    args = ap.parse_args(argv)
    languages = [value.strip() for value in args.languages.split(",") if value.strip()]
    if len(languages) != len(set(languages)) or any(x not in FIXTURES for x in languages):
        ap.error("select unique ko/en/ja/zh")
    if not 0.0 <= args.max_cer <= 0.7:
        ap.error("CER gate must be in 0.0..0.7")
    all_results = []
    for lang in languages:
        try:
            all_results.extend(inspect(lang, Path(args.model_dir), Path(args.manifest)))
        except Exception as exc:
            print("HSE_MULTILANG_OCR_ERROR " + json.dumps({
                "language": lang, "error_type": type(exc).__name__,
                "message": str(exc)[:240],
            }, ensure_ascii=True))
            return 3
    failed = [r for r in all_results if r["cer"] > args.max_cer or not r["recognized"]]
    summary = {
        "total_cases": len(all_results),
        "failed_cases": len(failed),
        "threshold_max_cer": args.max_cer,
        "max_observed_cer": max((r["cer"] for r in all_results), default=None),
    }
    print("HSE_MULTILANG_OCR_SUMMARY " + json.dumps(summary, sort_keys=True))
    return 0 if not failed and all_results else 4


if __name__ == "__main__":
    raise SystemExit(main())
