"""Headless end-to-end acceptance using OCR models INSIDE the packaged EXE.

Only four small synthetic locally encoded MP4s are permitted. The command
never accesses a website, external URL, browser profile, source credentials,
VPN, original user videos, or installed HSE configuration. All intermediate
SRT/ASS files live in an auto-deleted temporary directory; only a privacy-
preserving JSON result is written to the explicitly requested report path.
"""
from __future__ import annotations

from pathlib import Path
import json
import os
import sys
import tempfile

from .model_assets import resolve_model_paths
from .runtime_paths import bundled_manifest_path, bundled_model_dir
from .ocr.adaptive import AdaptiveOCREngine
from .ocr.rapidocr_adapter import RapidOCRAdapter
from .pipeline import extract
from .subtitle_formats import parse_subtitle_bytes

EXPECTED_TEXT = {
    "ko": ("오늘은 정말 좋은 날이에요", "잠시만 기다려 주세요"),
    "en": ("Where are you going tomorrow?", "I will be there in five minutes."),
    "ja": ("今日は何をしていますか", "明日は学校へ行きます"),
    "zh": ("今天晚上一起去看电影吧", "請你告訴我真正的名字"),
}
EXPECTED_SPANS = ((1.5, 3.625), (4.25, 6.375))
LANGUAGES = tuple(EXPECTED_TEXT)


def _bounded_local_fixture(directory: Path, language: str) -> Path:
    if not directory.is_dir() or directory.is_symlink():
        raise ValueError("not an isolated fixture directory")
    base = directory.resolve(strict=True)
    video = directory / (language + ".mp4")
    if video.is_symlink() or not video.is_file():
        raise ValueError("required synthetic video is missing")
    if video.resolve(strict=True).parent != base:
        raise ValueError("synthetic video escapes fixture directory")
    size = video.stat().st_size
    if not (1024 <= size <= 32 * 1024 * 1024):
        raise ValueError("synthetic MP4 file size is out of bounds")
    with video.open("rb") as f:
        prefix = f.read(12)
    if prefix[4:8] != b"ftyp":
        raise ValueError("fixture is not a local ISO BMFF video")
    return video


def _case(
    language: str,
    fixture: Path,
    temp_output: Path,
    model_dir: Path,
    manifest: Path,
    *,
    inference_engine: str = "onnxruntime",
) -> dict:
    # A model reference is never resolved with verify=False in acceptance.
    resolve_model_paths(
        model_dir, manifest_path=manifest, language=language, verify=True,
    )
    adapter = RapidOCRAdapter(
        language=language, model_dir=str(model_dir), manifest_path=str(manifest),
        inference_engine=inference_engine,
    )
    engine = AdaptiveOCREngine(adapter)
    report = extract(
        str(fixture), engine, temp_output / (language + ".subtitles"),
    )
    if report.cue_count != 2 or not report.srt_path.is_file() or not report.ass_path.is_file():
        raise AssertionError("incorrect real-video subtitle count")
    _, srt = parse_subtitle_bytes(
        report.srt_path.read_bytes(), source_url="synthetic.srt",
    )
    _, ass = parse_subtitle_bytes(
        report.ass_path.read_bytes(), source_url="synthetic.ass",
    )
    if len(srt) != 2 or len(ass) != 2:
        raise AssertionError("parsed SRT/ASS cue count does not match reference")

    max_error = 0.0
    max_cross_format_error = 0.0
    for i, expected in enumerate(EXPECTED_TEXT[language]):
        cue = srt[i]
        counterpart = ass[i]
        if cue.text != expected or counterpart.text != expected:
            raise AssertionError("foreign-language original text is not exact")
        span = EXPECTED_SPANS[i]
        error = max(abs(cue.start - span[0]), abs(cue.end - span[1]))
        cross_error = max(
            abs(cue.start - counterpart.start),
            abs(cue.end - counterpart.end),
        )
        max_error = max(max_error, error)
        max_cross_format_error = max(max_cross_format_error, cross_error)
        if error > 0.65 or cross_error > 0.02:
            raise AssertionError("real-video cue timing exceeds tolerance")
    return {
        "language": language,
        "requested_inference_engine": inference_engine,
        "inference_engine": getattr(adapter, "inference_engine", inference_engine),
        "backend_fallback_reason": getattr(adapter, "backend_fallback_reason", None),
        "verified_model": True,
        "real_ffmpeg_video_frames": True,
        "original_text_exact_cues": 2,
        "srt_ass_verified": True,
        "max_timing_error_seconds": round(max_error, 4),
        "max_srt_ass_timing_delta_seconds": round(max_cross_format_error, 4),
        "video_ocr_calls": engine.total_calls,
        "adaptive_retry_calls": engine.retry_calls,
    }


def run_frozen_multilingual_ocr_acceptance(
    fixture_dir: str | Path,
    report_path: str | Path,
    *,
    inference_engine: str = "onnxruntime",
) -> int:
    """Run all four exact video fixtures, emit no user/media text or file paths."""
    if inference_engine not in {"auto", "onnxruntime", "openvino"}:
        raise ValueError("unsupported frozen OCR backend")
    result: dict = {
        "schema": 1,
        "type": "frozen_multilingual_ocr_video_acceptance",
        "is_frozen": bool(getattr(sys, "frozen", False)),
        "requested_inference_engine": inference_engine,
        "inference_engine": None,
        "ok": False,
        "languages": [],
        "original_text_exact_total": 0,
        "no_external_requests": True,
        "temporary_outputs_removed": False,
    }
    code = 3
    try:
        directory = Path(fixture_dir)
        sources = {code: _bounded_local_fixture(directory, code) for code in LANGUAGES}
        manifest = bundled_manifest_path()
        models = bundled_model_dir()
        with tempfile.TemporaryDirectory(prefix="hse-frozen-ocr-acceptance-") as td:
            temporary = Path(td)
            for language in LANGUAGES:
                row = _case(
                    language, sources[language], temporary, models, manifest,
                    inference_engine=inference_engine,
                )
                result["languages"].append(row)
                result["original_text_exact_total"] += row["original_text_exact_cues"]
        actual_backends = {x["inference_engine"] for x in result["languages"]}
        result["inference_engine"] = (
            next(iter(actual_backends)) if len(actual_backends) == 1 else "mixed"
        )
        result["temporary_outputs_removed"] = True
        result["ok"] = (
            len(result["languages"]) == 4
            and result["original_text_exact_total"] == 8
            and all(x["srt_ass_verified"] for x in result["languages"])
        )
        code = 0 if result["ok"] else 4
    except Exception as exc:
        # Unexpected exception messages may contain video paths/codec commands.
        # Only record bounded class names, never the source filename, URL,
        # OCR text, actual exception message or user profile location.
        result["failure_type"] = type(exc).__name__
    finally:
        # TemporaryDirectory deletes even after a per-language extraction
        # fails. The flag means all four finished and cleanup was reached.
        target = Path(report_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(
            json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8",
        )
    return code
