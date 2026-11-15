"""Offline actual-model OCR of reproducible transparent BDN subtitle images.

This is a DIAGNOSTIC, not proof of real anime transcription accuracy.
It calls the production BDN reader and source-language SHA-pinned RapidOCR.
No user video, browser, provider, network, or video-frame reference leakage.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import time
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from PIL import Image, ImageDraw, ImageFont
import numpy as np
import cv2

from hardsub_extractor.models import Frame
from hardsub_extractor.local_bitmap_bdn_ocr import _image_to_bgr

from hardsub_extractor.local_bitmap_bdn_ocr import BitmapBdnOcrError, recognize_bdn_images
from hardsub_extractor.ocr.adaptive import AdaptiveOCREngine
from hardsub_extractor.ocr.base import comparison_key, normalize_text
from hardsub_extractor.ocr.rapidocr_adapter import RapidOCRAdapter
from hardsub_extractor.subtitle_formats import parse_subtitle_bytes
from hardsub_extractor.writers import write_subtitle_pair_exclusive
from verify_multilang_ocr_synthetic import FIXTURES, choose_font, edit_distance

LANGUAGES = ("ko", "en", "ja", "zh")
IMAGE_SIZE = (1280, 180)
TIMES = ((1, 3), (4, 6))


def _draw(path: Path, rows: tuple[str, ...], font_path: Path) -> None:
    image = Image.new("RGBA", IMAGE_SIZE, (0, 0, 0, 0))
    draw = ImageDraw.Draw(image)
    for size in range(50, 27, -2):
        font = ImageFont.truetype(str(font_path), size)
        boxes = [draw.textbbox((0, 0), row, font=font, stroke_width=2) for row in rows]
        if all(box[2] - box[0] < IMAGE_SIZE[0] - 64 for box in boxes):
            break
    else:
        raise AssertionError("reference text cannot fit bitmap")
    ys = (61,) if len(rows) == 1 else (8, 99)
    for text, box, top in zip(rows, boxes, ys):
        width = box[2] - box[0]
        draw.text(
            ((IMAGE_SIZE[0] - width) // 2 - box[0], top - box[1]),
            text, font=font, fill=(255, 255, 255, 255),
            stroke_width=2, stroke_fill=(0, 0, 0, 255),
        )
    image.save(path, format="PNG", optimize=True)
    if not 0 < path.stat().st_size <= 8 * 1024 * 1024:
        raise AssertionError("bitmap image size invalid")


def _index(folder: Path) -> None:
    root = ET.Element("BDN", Version="0.93")
    desc = ET.SubElement(root, "Description")
    ET.SubElement(desc, "Format", VideoFormat="1920x1080", FrameRate="25", DropFrame="false")
    events = ET.SubElement(root, "Events")
    for i, (start, end) in enumerate(TIMES, 1):
        event = ET.SubElement(
            events, "Event", InTC=f"00:00:{start:02d}:00",
            OutTC=f"00:00:{end:02d}:00", Forced="false",
        )
        graphic = ET.SubElement(
            event, "Graphic", Width=str(IMAGE_SIZE[0]), Height=str(IMAGE_SIZE[1]),
            X="320", Y="800",
        )
        graphic.text = f"{i:04d}.png"
    ET.ElementTree(root).write(folder / "index.xml", encoding="utf-8", xml_declaration=True)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _inspect(lang: str, models: Path, manifest: Path, folder: Path, *, composite_study: bool = False) -> dict:
    references = ((FIXTURES[lang][0],), FIXTURES[lang])
    images = folder / "bitmap"
    images.mkdir()
    font = choose_font(lang)
    for n, rows in enumerate(references, 1):
        _draw(images / f"{n:04d}.png", rows, font)
    _index(images)
    ocr = AdaptiveOCREngine(RapidOCRAdapter(
        model_dir=str(models), manifest_path=str(manifest), language=lang,
    ))
    begin = time.perf_counter()
    try:
        cues = recognize_bdn_images(images, ocr)
    except BitmapBdnOcrError as exc:
        if "no recognized bitmap subtitle text" not in str(exc):
            raise
        cues = []
    elapsed = round(time.perf_counter() - begin, 3)
    by_start = {round(cue.start): cue for cue in cues}
    if len(by_start) != len(cues):
        raise AssertionError("multiple OCR results for one source image")
    cases = []
    for i, ((start, end), rows) in enumerate(zip(TIMES, references), 1):
        expected = "\n".join(rows)
        result = by_start.get(start)
        got = result.text if result else ""
        key_a, key_b = comparison_key(expected), comparison_key(got)
        cer = edit_distance(key_a, key_b) / max(1, len(key_a))
        if result and (abs(result.start - start) > .02 or abs(result.end - end) > .02):
            raise AssertionError("BDN original timeline not preserved")
        cases.append({
            "case": i,
            "detected": bool(got),
            "lines_expected": len(rows),
            "lines_found": len(got.splitlines()) if got else 0,
            "cer": round(cer, 4),
            "exact_text": normalize_text(expected) == normalize_text(got),
            "exact_layout": len(got.splitlines()) == len(rows),
            "observed_synthetic_text": got if composite_study else None,
            "confidence": round(float(result.confidence), 4) if result else 0.0,
        })
    experiments = []
    if composite_study:
        # Only 4 affected synthetic examples; independently measure image
        # polarities, never leak reference words to candidate selection.
        for i, rows in enumerate(references, 1):
            png = images / f"{i:04d}.png"
            expected_key = comparison_key("\n".join(rows))
            black = _image_to_bgr(png, white=False)
            # BDN PNGs can contain huge transparent borders. The detector
            # may undersample a small text region. Trim by the alpha mask
            # only; neither reference labels nor OCR hypotheses are inputs.
            with Image.open(png) as picture:
                rgba = np.asarray(picture.convert("RGBA"))
                bbox = picture.getchannel("A").getbbox()
            if bbox is None:
                raise AssertionError("synthetic glyph bitmap has no alpha")
            alpha = rgba[:, :, 3:4].astype(np.float32) / 255.0
            background_bgr = np.array([59, 43, 36], dtype=np.float32)
            neutral = np.clip(
                black.astype(np.float32)
                + background_bgr[None, None, :] * (1 - alpha),
                0, 255,
            ).astype(np.uint8)
            left, top, right, bottom = bbox
            margin = 20
            x0 = max(0, left - margin)
            y0 = max(0, top - margin)
            x1 = min(black.shape[1], right + margin)
            y1 = min(black.shape[0], bottom + margin)
            if x1 - x0 < 20 or y1 - y0 < 20:
                raise AssertionError("unexpected empty alpha crop")
            def cropped(value):
                return np.ascontiguousarray(value[y0:y1, x0:x1])
            # Place the same source PNG on a 720p neutral matte, as in the
            # previously successful synthetic video OCR reference. This
            # tests input geometry and decoder behavior separately; it
            # never uses the expected words to choose model output.
            screen = Image.new("RGB", (1280, 720), (36, 43, 59))
            with Image.open(png) as original:
                art = original.convert("RGBA")
                screen.paste(art, (0, 505), mask=art.getchannel("A"))
            screen_bgr = cv2.cvtColor(np.asarray(screen), cv2.COLOR_RGB2BGR)
            variants = (
                ("neutral_720_bitmap", screen_bgr),
                ("neutral_720_scene", screen_bgr),
            )
            for name, pixels in variants:
                recognized = (
                    ocr.recognize(Frame(0.0, pixels))
                    if name.endswith("_scene")
                    else ocr.base.recognize_cropped_bitmap(Frame(0.0, pixels))
                )
                got = recognized.text.strip()
                experiments.append({
                    "case": i,
                    "view": name,
                    "cer": round(edit_distance(
                        expected_key, comparison_key(got),
                    ) / max(1, len(expected_key)), 4),
                    "lines_expected": len(rows),
                    "lines_found": len(got.splitlines()) if got else 0,
                    "confidence": round(float(recognized.confidence), 4),
                    "observed_synthetic_text": got,
                    "input_width": int(pixels.shape[1]),
                    "input_height": int(pixels.shape[0]),
                })
    if cues:
        base = folder / "output" / "caption"
        srt, ass = write_subtitle_pair_exclusive(cues, base)
        srt_fmt, srt_rows = parse_subtitle_bytes(srt.read_bytes(), source_url="caption.srt")
        ass_fmt, ass_rows = parse_subtitle_bytes(ass.read_bytes(), source_url="caption.ass")
        if (srt_fmt, ass_fmt) != ("srt", "ass") or len(cues) != len(srt_rows) or len(cues) != len(ass_rows):
            raise AssertionError("source bitmap OCR SRT and ASS mismatch")
        for source, sr, ar in zip(cues, srt_rows, ass_rows):
            if (source.text != sr.text or sr.text != ar.text
                    or abs(sr.start - ar.start) > .02 or abs(sr.end - ar.end) > .02):
                raise AssertionError("original bitmap OCR text or timebase lost")
        hashes = (_sha(srt), _sha(ass))
        try:
            write_subtitle_pair_exclusive(cues, base)
        except FileExistsError:
            pass
        else:
            raise AssertionError("existing subtitles overwritten on second publish")
        if hashes != (_sha(srt), _sha(ass)):
            raise AssertionError("existing exported subtitle files changed")
    return {
        "language": lang,
        "source": "generated_transparent_BDN_PNG",
        "source_events": len(references),
        "recognized_events": len(cues),
        "cases": cases,
        "composite_study": experiments,
        "ocr_seconds": elapsed,
        "sha_pinned_real_ppocr_model": True,
        "real_anime_accuracy_proven": False,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", type=Path, required=True)
    ap.add_argument("--manifest", type=Path, default=ROOT / "models" / "manifest.json")
    ap.add_argument("--composite-study", action="store_true")
    args = ap.parse_args()
    languages = ("ko", "zh") if args.composite_study else LANGUAGES
    results = []
    with tempfile.TemporaryDirectory(prefix="hse-bdn-four-lang-") as tmp:
        root = Path(tmp)
        for lang in languages:
            folder = root / lang
            folder.mkdir()
            result = _inspect(
                lang, args.model_dir, args.manifest, folder,
                composite_study=args.composite_study,
            )
            results.append(result)
            print("HSE_BDN_LANGUAGE=" + json.dumps(result, sort_keys=True, ensure_ascii=True), flush=True)
    summary = {
        "languages": list(languages),
        "composite_study": args.composite_study,
        "total_cases": sum(len(x["cases"]) for x in results),
        "recognized_events": sum(x["recognized_events"] for x in results),
        "exact_text_cases": sum(sum(y["exact_text"] for y in x["cases"]) for x in results),
        "exact_layout_cases": sum(sum(y["exact_layout"] for y in x["cases"]) for x in results),
        "max_cer": max(y["cer"] for x in results for y in x["cases"]),
        "quality_gate": "DIAGNOSTIC_ONLY_NOT_RELEASE_APPROVAL",
        "user_media_used": False,
    }
    print("HSE_BDN_SUMMARY=" + json.dumps(summary, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
