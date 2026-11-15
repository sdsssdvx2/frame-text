"""Bounded genuine multilingual hard-subtitle OCR through FFmpeg -> SRT/ASS.

Synthetic frames are encoded as a real MPEG-4 Part 2 MP4 with pinned LGPL FFmpeg, then
read by the production frame detector and native RapidOCR. This verifies
timed cue discovery, recognizing each explicit language, and subtitle export.
Temporary input/output files are deleted. No browser, provider, credentials,
video download, user profile or installed program are involved.
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

from hardsub_extractor.diagnostics import DiagnosticRecorder
from hardsub_extractor.ocr.adaptive import AdaptiveOCREngine
from hardsub_extractor.ocr.rapidocr_adapter import RapidOCRAdapter
from hardsub_extractor.ocr.base import comparison_key
from hardsub_extractor.pipeline import extract
from hardsub_extractor.runtime_paths import resolve_binary
from hardsub_extractor.subtitle_formats import parse_subtitle_bytes
from verify_multilang_ocr_synthetic import (
    FIXTURES, choose_font, edit_distance, render_subtitle,
)

# Two distinct cues, with an actual blank interval, to distinguish subtitle
# transition detection from simply recognizing a static screenshot. zh
# exercises both simplified and traditional Chinese in the same video.
VIDEO_TEXT = {
    **FIXTURES,
    "zh": ("今天晚上一起去看电影吧", "請你告訴我真正的名字"),
}
FPS = 8
SEGMENTS = (
    (0, 12, None),  # 0.00-1.50s, background calibration
    (12, 29, 0),   # 1.50-3.625s, first subtitle
    (29, 34, None),# 3.625-4.25s, genuinely blank
    (34, 51, 1),   # 4.25-6.375s, second subtitle
    (51, 57, None),# 6.375-7.125s, blank
)
EXPECTED_SPANS = ((1.50, 3.625), (4.25, 6.375))


def encode_sample(video: Path, language: str) -> None:
    font = choose_font(language)
    text_frames = tuple(render_subtitle(text, font) for text in VIDEO_TEXT[language])
    height, width = text_frames[0].shape[:2]
    blank = np.zeros_like(text_frames[0])
    blank[:, :] = (59, 43, 36)  # PIL RGB(36,43,59) decoded BGR
    images = {None: blank, 0: text_frames[0], 1: text_frames[1]}
    command = [
        resolve_binary("ffmpeg"),
        "-nostdin", "-hide_banner", "-loglevel", "error",
        "-f", "rawvideo", "-pixel_format", "bgr24",
        "-video_size", f"{width}x{height}", "-framerate", str(FPS),
        "-i", "pipe:0", "-an",
        "-c:v", "mpeg4", "-q:v", "4",
        "-pix_fmt", "yuv420p", "-movflags", "+faststart",
        "-y", str(video),
    ]
    process = subprocess.Popen(
        command, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        assert process.stdin is not None
        frames_written = 0
        for start, end, index in SEGMENTS:
            payload = images[index].tobytes()
            for _ in range(start, end):
                try:
                    process.stdin.write(payload)
                except BrokenPipeError as exc:
                    process.wait(timeout=10)
                    raise RuntimeError("pinned FFmpeg encoder terminated before finishing the temporary MP4") from exc
                frames_written += 1
        process.stdin.close()
        if process.wait(timeout=90) != 0:
            raise RuntimeError("temporary FFmpeg sample encoding failed")
        if frames_written != SEGMENTS[-1][1] or not video.is_file():
            raise RuntimeError("temporary MP4 fixture has invalid frame count")
        if video.stat().st_size > 32 * 1024 * 1024:
            raise RuntimeError("temporary MP4 fixture exceeds bounded size")
    finally:
        if process.poll() is None:
            process.kill()
        process.wait(timeout=10)


def inspect(language: str, root: Path, model_dir: Path, manifest: Path) -> dict:
    source = root / f"{language}.mp4"
    encode_sample(source, language)
    ocr = AdaptiveOCREngine(RapidOCRAdapter(
        model_dir=str(model_dir), manifest_path=str(manifest),
        language=language,
    ))
    diagnostic = (
        DiagnosticRecorder(root / f"{language}.synthetic-observations")
        if language in {"en", "zh"} else None
    )
    start = time.perf_counter()
    result = extract(
        str(source), ocr, root / f"{language}.result",
        diagnostics=diagnostic,
    )
    elapsed = time.perf_counter() - start
    # Only synthetic sentences are recorded here; the same diagnostics
    # structure is used in production but no user or provider content enters
    # this isolated reproducible test. All JPEG data stays temporary.
    if diagnostic is not None:
        for record in diagnostic.records[:10]:
            sample_rows = []
            for sample in record["samples"]:
                observed = sample["ocr"]
                if not observed["text"]:
                    continue
                detections = (observed.get("metadata") or {}).get("detections", [])
                sample_rows.append({
                    "pts": sample["pts"],
                    "text": observed["text"],
                    "boxes": [
                        {
                            "text": item.get("text", ""),
                            "cx": item.get("cx"),
                            "width": item.get("width"),
                        }
                        for item in detections[:6]
                    ],
                })
            if sample_rows:
                print("HSE_VIDEO_OBSERVATIONS " + json.dumps({
                    "language": language,
                    "segment": record["index"],
                    "start": record["start"], "end": record["end"],
                    "samples": sample_rows,
                    "voted_text": record["voted"]["text"],
                }, ensure_ascii=True))
    if not (result.srt_path.is_file() and result.ass_path.is_file()):
        raise AssertionError("both original-timeline subtitle formats are required")
    _, srt = parse_subtitle_bytes(result.srt_path.read_bytes(), source_url=result.srt_path.name)
    _, ass = parse_subtitle_bytes(result.ass_path.read_bytes(), source_url=result.ass_path.name)
    if len(srt) != len(ass):
        raise AssertionError("SRT and ASS cue counts disagree")
    checks = []
    for i, expected in enumerate(VIDEO_TEXT[language]):
        if i >= len(srt):
            break
        reference = comparison_key(expected)
        hypothesis = comparison_key(srt[i].text)
        cer = edit_distance(reference, hypothesis) / max(1, len(reference))
        span = EXPECTED_SPANS[i]
        checks.append({
            "cue": i+1,
            "cer": round(cer, 4),
            "start_error": round(abs(srt[i].start - span[0]), 4),
            "end_error": round(abs(srt[i].end - span[1]), 4),
            "ass_start_error": round(abs(srt[i].start - ass[i].start), 4),
            "ass_end_error": round(abs(srt[i].end - ass[i].end), 4),
            "exact_text": srt[i].text == expected,
            "reference": expected,
            "recognized": srt[i].text,
        })
    info = {
        "language": language,
        "cue_count": len(srt),
        "frame_count": SEGMENTS[-1][1],
        "decode_encode_pipeline": "ffmpeg-mp4-mpeg4-yuv420p-local",
        "elapsed_seconds": round(elapsed, 2),
        "ocr_calls": ocr.total_calls,
        "ocr_retry_calls": ocr.retry_calls,
        "checks": checks,
    }
    print("HSE_VIDEO_OCR " + json.dumps(info, ensure_ascii=True))
    return info


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", default=str(ROOT / "models" / "runtime"))
    parser.add_argument("--manifest", default=str(ROOT / "models" / "manifest.json"))
    parser.add_argument("--languages", default="ko,en,ja,zh")
    parser.add_argument("--max-cer", type=float, default=0.20)
    parser.add_argument("--max-timing-error", type=float, default=0.65)
    parser.add_argument("--require-exact-text", action="store_true",
                        help="Demand exact visible cue text, including English word spaces")
    options = parser.parse_args(argv)
    languages = [x.strip() for x in options.languages.split(",") if x.strip()]
    if not languages or len(languages) != len(set(languages)) or any(x not in VIDEO_TEXT for x in languages):
        parser.error("select distinct ko/en/ja/zh")
    if not (0 <= options.max_cer <= 0.5 and 0 <= options.max_timing_error <= 2.0):
        parser.error("invalid bounded quality thresholds")
    cases = []
    with tempfile.TemporaryDirectory(prefix="hse-multilingual-video-") as td:
        for language in languages:
            try:
                cases.append(inspect(
                    language, Path(td), Path(options.model_dir),
                    Path(options.manifest),
                ))
            except Exception as exc:
                print("HSE_VIDEO_OCR_ERROR " + json.dumps({
                    "language": language,
                    "type": type(exc).__name__,
                    "description": str(exc)[:160],
                }, ensure_ascii=True))
                return 3
    failures = []
    for case in cases:
        if case["cue_count"] != 2 or len(case["checks"]) != 2:
            failures.append(f"{case['language']}:cue_count")
        for check in case["checks"]:
            if options.require_exact_text and not check["exact_text"]:
                failures.append(f"{case['language']}:cue{check['cue']}:surface_text")
            if check["cer"] > options.max_cer:
                failures.append(f"{case['language']}:cue{check['cue']}:cer")
            if max(check["start_error"], check["end_error"]) > options.max_timing_error:
                failures.append(f"{case['language']}:cue{check['cue']}:timing")
            if max(check["ass_start_error"], check["ass_end_error"]) > 0.02:
                failures.append(f"{case['language']}:cue{check['cue']}:srt_ass_timeline")
    print("HSE_VIDEO_OCR_SUMMARY " + json.dumps({
        "languages": languages, "cues_expected": len(cases) * 2,
        "cases": len(cases), "failures": failures,
    }, ensure_ascii=True))
    return 4 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
