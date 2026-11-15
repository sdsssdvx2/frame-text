"""Reproducible animated-background subtitle quality benchmark.

Generates its own moving, high-contrast illustrated-style background and
genuine encoded MP4 clips with source-language white outlined captions.
Exercises the *production* 8fps streaming detector, PP-OCRv5 recognizer,
timed SRT/ASS publishing, and exact known synthetic reference text.

No real anime, website, credentials, network, browser, or user media is read.
This tests difficult moving visuals; it is NOT ground-truth real anime CER.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from hardsub_extractor.ocr.adaptive import AdaptiveOCRConfig, AdaptiveOCREngine
from hardsub_extractor.ocr.rapidocr_adapter import RapidOCRAdapter
from hardsub_extractor.ocr.base import comparison_key
from hardsub_extractor.pipeline import extract
from hardsub_extractor.runtime_paths import resolve_binary
from hardsub_extractor.subtitle_formats import parse_subtitle_bytes
from verify_multilang_ocr_synthetic import choose_font, edit_distance
from verify_multilang_video_pipeline import VIDEO_TEXT, SEGMENTS, EXPECTED_SPANS

WIDTH, HEIGHT, FPS = 1280, 720, 8
MAX_VIDEO_BYTES = 32 * 1024 * 1024


def _multiline_reference(text: str) -> list[str]:
    """Fixture-only two-line split preserving every character in the source."""
    chunks = text.split(" ")
    if len(chunks) >= 3:
        half = max(1, len(chunks) // 2)
        return [" ".join(chunks[:half]), " ".join(chunks[half:])]
    index = max(1, len(text) // 2)
    return [text[:index], text[index:]]


def subtitle_overlay(
    text: str, font_path: Path, *,
    style: str = "white_single",
) -> np.ndarray:
    """Render independently annotated text, not OCR-derived guessed strings."""
    if style not in {"white_single", "white_two_line", "white_two_line_high", "yellow_single"}:
        raise ValueError("unsupported synthetic OCR stress style")
    image = Image.new("RGBA", (WIDTH, HEIGHT), (0, 0, 0, 0))
    draw = ImageDraw.Draw(image)
    lines = (
        _multiline_reference(text)
        if style in {"white_two_line", "white_two_line_high"} else [text]
    )
    size = 51 if len(lines) == 2 else 58
    while True:
        font = ImageFont.truetype(str(font_path), size=size)
        boxes = [
            draw.textbbox((0, 0), line, font=font, stroke_width=3)
            for line in lines
        ]
        widths = [box[2] - box[0] for box in boxes]
        if max(widths) <= WIDTH * 0.85 or size <= 29:
            break
        size -= 3
    if max(widths) > WIDTH * 0.90:
        raise ValueError("reference text is too long for the video frame")
    fill = (
        (255, 230, 5, 255) if style == "yellow_single"
        else (255, 255, 255, 255)
    )
    for n, (line, box, width) in enumerate(zip(lines, boxes, widths)):
        # y=525 preserves the original white-single video exactly.
        # Explicit high-position research fixture: the top glyph row sits
        # above the historical y=0.58 crop, while the second row remains
        # inside it. The normal fixtures are pixel-identical by default.
        baseline = (
            355 + n * 80 if style == "white_two_line_high"
            else 470 + n * 70 if len(lines) == 2 else 525
        )
        draw.text(
            ((WIDTH - width) / 2 - box[0], baseline - box[1]),
            line, font=font, fill=fill,
            stroke_width=3, stroke_fill=(1, 1, 1, 255),
        )
    return cv2.cvtColor(np.asarray(image), cv2.COLOR_RGBA2BGRA)


def animated_frame(index: int, base: np.ndarray, caption: np.ndarray | None) -> np.ndarray:
    """Illustrated moving color fields, silhouettes, occluding light, and lines.

    Bright moving areas intentionally cross the subtitles' normal backdrop.
    The text layer is composited afterward, as with actual baked-in captions.
    """
    image = base.copy()
    t = index / FPS
    cv2.rectangle(image, (0, 0), (WIDTH, 255),
                  (88 + (index * 2) % 65, 105, 156), -1)
    cv2.ellipse(
        image, (225 + (index * 8) % 1120, 300 + (index % 13)),
        (245, 355), 25, 0, 360, (170, 125, 87), -1,
    )
    cv2.ellipse(
        image, (1000 - index * 7, 410), (210, 360), -15,
        0, 360, (63, 82, 126), -1,
    )
    # Stylized moving "foreground" shapes and rim highlights produce
    # realistic edge/contrast motion without any facial identity.
    light = int(160 + 60 * abs(np.sin(t * 0.65)))
    cv2.rectangle(
        image, (130 + (index * 13) % 950, 425),
        (310 + (index * 13) % 950, 615),
        (light, light, min(255, light + 32)), -1,
    )
    cv2.line(
        image, (0, (index * 7) % HEIGHT),
        (WIDTH, (340 + index * 7) % HEIGHT),
        (200, 215, 235), 7, cv2.LINE_AA,
    )
    cv2.line(
        image, (0, 498 + index % 6), (WIDTH, 496 + index % 6),
        (29, 36, 48), 3, cv2.LINE_AA,
    )
    if caption is not None:
        a = caption[:, :, 3:4].astype(np.uint16)
        image = (
            (image.astype(np.uint16) * (255 - a)
             + caption[:, :, :3].astype(np.uint16) * a + 127) // 255
        ).astype(np.uint8)
    return image


def encode_video(
    dest: Path, language: str, *,
    include_subtitles: bool = True, style: str = "white_single",
) -> None:
    overlays = (
        tuple(subtitle_overlay(text, choose_font(language), style=style)
              for text in VIDEO_TEXT[language])
        if include_subtitles else ()
    )
    # A fixed non-solid linear gradient is the animation's immutable layer.
    x = np.arange(WIDTH, dtype=np.uint16)[None, :]
    y = np.arange(HEIGHT, dtype=np.uint16)[:, None]
    base = np.empty((HEIGHT, WIDTH, 3), dtype=np.uint8)
    base[:, :, 0] = ((x // 4 + y // 6) % 93 + 42).astype(np.uint8)
    base[:, :, 1] = ((x // 5 + y // 3) % 77 + 53).astype(np.uint8)
    base[:, :, 2] = ((x // 7 + y // 4) % 95 + 64).astype(np.uint8)

    cmd = [
        resolve_binary("ffmpeg"), "-nostdin", "-hide_banner", "-loglevel",
        "error", "-f", "rawvideo", "-pixel_format", "bgr24",
        "-video_size", f"{WIDTH}x{HEIGHT}", "-framerate", str(FPS),
        "-i", "pipe:0", "-an", "-c:v", "mpeg4", "-q:v", "4",
        "-pix_fmt", "yuv420p", "-movflags", "+faststart", "-y", str(dest),
    ]
    process = subprocess.Popen(
        cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        if process.stdin is None:
            raise RuntimeError("unable to open pinned encoder stdin")
        previous = 0
        for start, end, cue in SEGMENTS:
            if start != previous:
                raise RuntimeError("noncontiguous reference animation timeline")
            for index in range(start, end):
                frame = animated_frame(
                    index, base, overlays[cue] if include_subtitles and cue is not None else None,
                )
                process.stdin.write(frame.tobytes())
            previous = end
        process.stdin.close()
        if process.wait(timeout=90) != 0 or not dest.is_file():
            raise RuntimeError("pinned animated MP4 encoder failed")
    finally:
        if process.poll() is None:
            process.kill()
        process.wait(timeout=10)
    if not (1024 <= dest.stat().st_size <= MAX_VIDEO_BYTES):
        raise RuntimeError("animated video exceeds the bounded test fixture size")


def check_language(
    root: Path, lang: str, models: Path, manifest: Path,
    *, retry_box_threshold: float = 0.90,
    style: str = "white_single",
    diagnostics=None,
    detector_config=None,
    source_override: Path | None = None,
) -> dict:
    source = Path(source_override) if source_override is not None else root / (lang + ".mp4")
    if source_override is None:
        encode_video(source, lang, style=style)
    engine = AdaptiveOCREngine(
        RapidOCRAdapter(
            language=lang, model_dir=str(models), manifest_path=str(manifest),
        ),
        config=AdaptiveOCRConfig(
            retry_below_box_confidence=retry_box_threshold,
        ),
    )
    began = time.monotonic()
    result = extract(
        str(source), engine, root / (lang + ".subtitles"),
        sample_fps=8.0, streaming=True, diagnostics=diagnostics,
        detector=detector_config,
    )
    elapsed = time.monotonic() - began
    _, srt = parse_subtitle_bytes(
        result.srt_path.read_bytes(), source_url="animated.srt",
    )
    _, ass = parse_subtitle_bytes(
        result.ass_path.read_bytes(), source_url="animated.ass",
    )
    aligned = []
    for index, expected in enumerate(VIDEO_TEXT[lang]):
        if index >= len(srt):
            aligned.append({"cue": index + 1, "present": False})
            continue
        truth = comparison_key(expected)
        hypothesis = comparison_key(srt[index].text)
        start, end = EXPECTED_SPANS[index]
        aligned.append({
            "cue": index + 1,
            "present": True,
            "exact_text": srt[index].text == expected,
            "cer": round(edit_distance(truth, hypothesis) / max(1, len(truth)), 4),
            # This is an entirely generated text fixture, not user media.
            "reference_synthetic": expected,
            "observed_synthetic": srt[index].text,
            # Synthetic-only media; bounded timestamps contain no user data.
            # Capture the observed edge too, so a delayed switch can be
            # distinguished from an over-extended preceding caption.
            "observed_start_seconds": round(srt[index].start, 3),
            "observed_end_seconds": round(srt[index].end, 3),
            "start_error": round(abs(srt[index].start - start), 3),
            "end_error": round(abs(srt[index].end - end), 3),
        })
    matching_output = (
        len(srt) == len(ass) == len(VIDEO_TEXT[lang])
        and all(
            a.text == b.text and abs(a.start - b.start) <= 0.02
            and abs(a.end - b.end) <= 0.02
            for a, b in zip(srt, ass)
        )
    )
    return {
        "language": lang,
        "style": style,
        "frame_count": SEGMENTS[-1][1],
        "cues_detected": len(srt),
        "cues_expected": len(VIDEO_TEXT[lang]),
        "matched_srt_ass": matching_output,
        "elapsed_seconds": round(elapsed, 2),
        "ocr_calls": engine.total_calls,
        "ocr_retry_calls": engine.retry_calls,
        "retry_below_box_confidence": retry_box_threshold,
        "reference_comparison": aligned,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--manifest", default=str(ROOT / "models" / "manifest.json"))
    ap.add_argument("--languages", default="ko,en,ja,zh")
    ap.add_argument("--retry-box-threshold", type=float, default=0.90)
    ap.add_argument("--max-cer", type=float, default=0.15)
    ap.add_argument("--max-time-error", type=float, default=0.65)
    opts = ap.parse_args()
    langs = [v.strip() for v in opts.languages.split(",") if v.strip()]
    if langs != ["ko", "en", "ja", "zh"]:
        ap.error("the full animated test covers ko,en,ja,zh in order")
    if not (0.65 <= opts.retry_box_threshold <= 0.95):
        ap.error("bounded retry threshold required")
    if not (0 <= opts.max_cer <= 0.5 and 0 <= opts.max_time_error <= 1.0):
        ap.error("quality bounds must remain strict")
    samples = []
    try:
        with tempfile.TemporaryDirectory(prefix="hse-animated-video-ocr-") as td:
            for lang in langs:
                samples.append(check_language(
                    Path(td), lang, Path(opts.model_dir), Path(opts.manifest),
                    retry_box_threshold=opts.retry_box_threshold,
                ))
    except Exception as exc:
        print("HSE_ANIMATED_VIDEO_OCR_ERROR " + json.dumps({
            "error_class": type(exc).__name__,
        }))
        return 3
    failures = []
    for sample in samples:
        lang = sample["language"]
        if not sample["matched_srt_ass"] or sample["cues_detected"] != 2:
            failures.append(lang + ":cue_count_or_format")
        for item in sample["reference_comparison"]:
            if not item["present"]:
                failures.append(lang + ":missing_cue")
            elif item["cer"] > opts.max_cer:
                failures.append(lang + ":cue" + str(item["cue"]) + ":cer")
            elif max(item["start_error"], item["end_error"]) > opts.max_time_error:
                failures.append(lang + ":cue" + str(item["cue"]) + ":timing")
    report = {
        "schema": 1,
        "scope": "procedural_moving_animation_background_and_real_encoded_mp4",
        "real_anime_accuracy_verified": False,
        "actual_website_tested": False,
        "languages": langs,
        "retry_below_box_confidence": opts.retry_box_threshold,
        "exact_matches": sum(
            item.get("exact_text", False)
            for sample in samples for item in sample["reference_comparison"]
        ),
        "tested_cues": 8,
        "failures": failures,
        "results": samples,
    }
    print("HSE_ANIMATED_VIDEO_OCR_SUMMARY " + json.dumps(report, ensure_ascii=True))
    return 0 if not failures else 4


if __name__ == "__main__":
    raise SystemExit(main())
