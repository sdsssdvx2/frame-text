"""Real PP-OCRv5 and pinned VSF integration for the streaming timed OCR core.

One four-language study would redundantly encode and infer the same film
again. This targeted acceptance tests a single JA case that previously
missed one strict joint cue, on genuine licensed moving-film frames with
a separately annotated two-row added caption and a no-added-caption
control. This does NOT install or bundle GPL VSF into the product.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import tempfile

from compare_external_engine_exports import Cue as ScoreCue, evaluate
from compare_sintel_real_film_hard_ocr import make_original_motion_pair
from compare_vsf_timing_research import check_native_version
from hardsub_extractor.native_vsf_timing import detect_native_vsf_timings
from probe_sintel_original_animation import (
    EXPECTED_SOURCE_SHA256, fetch_source, probe_duration_and_dimensions,
)
from verify_animated_background_ocr_quality import _multiline_reference
from verify_multilang_video_pipeline import EXPECTED_SPANS, VIDEO_TEXT
from hardsub_extractor.native_vsf_assets import ensure_official_vsf
from hardsub_extractor.ocr.rapidocr_adapter import RapidOCRAdapter
from hardsub_extractor.subtitle_formats import parse_subtitle_bytes
from hardsub_extractor.timed_ocr import extract_timed_ocr

LANGUAGE = "ja"


def run_real_streaming_trial(model_dir: Path, manifest: Path) -> dict:
    with tempfile.TemporaryDirectory(prefix="hse-sintel-timed-ocr-") as temp:
        root = Path(temp)
        original = root / "sintel.mp4"
        sha, original_bytes = fetch_source(
            original, required_hash=EXPECTED_SOURCE_SHA256,
        )
        media = probe_duration_and_dimensions(original)
        positive, negative = root / "caption.mp4", root / "no-added-caption.mp4"
        sources = make_original_motion_pair(
            original, positive, negative, LANGUAGE,
        )
        native, official_exe_sha = ensure_official_vsf(
            consent=True, cache_base=root / "official-native-cache",
        )
        if ensure_official_vsf(
            consent=True, cache_base=root / "official-native-cache",
        ) != (native, official_exe_sha):
            raise RuntimeError("native official cache reuse identity failed")
        check_native_version(native)
        positive_timing = detect_native_vsf_timings(
            positive, native, pinned_executable_sha256=official_exe_sha,
        )
        negative_timing = detect_native_vsf_timings(
            negative, native, pinned_executable_sha256=official_exe_sha,
        )
        intervals = positive_timing.intervals
        detect_seconds = positive_timing.detection_seconds
        negative_intervals = negative_timing.intervals
        adapter = RapidOCRAdapter(
            language=LANGUAGE, model_dir=str(model_dir),
            manifest_path=str(manifest),
        )
        report = extract_timed_ocr(
            str(positive), intervals, adapter,
            root / "streamed-output", sample_fps=8,
        )
        _, output = parse_subtitle_bytes(
            report.srt_path.read_bytes(), source_url="output.srt",
        )
        _, ass = parse_subtitle_bytes(
            report.ass_path.read_bytes(), source_url="output.ass",
        )
        if (
            len(output) != len(ass)
            or any(
                x.text != y.text or abs(x.start - y.start) > 0.015
                or abs(x.end - y.end) > 0.015
                for x, y in zip(output, ass)
            )
        ):
            raise RuntimeError("timed OCR SRT and ASS export disagree")
        reference = [
            ScoreCue(start, end, "\n".join(_multiline_reference(text)))
            for (start, end), text in zip(
                EXPECTED_SPANS, VIDEO_TEXT[LANGUAGE],
            )
        ]
        score = evaluate(
            reference,
            [ScoreCue(item.start, item.end, item.text) for item in output],
            negative_intervals,
        )
        if (
            hashlib.sha256(original.read_bytes()).hexdigest() != sha
            or hashlib.sha256(positive.read_bytes()).hexdigest()
               != sources["positive_sha256"]
            or hashlib.sha256(negative.read_bytes()).hexdigest()
               != sources["negative_sha256"]
        ):
            raise RuntimeError("benchmark input media changed during inference")
        if (
            report.detected_intervals != len(intervals)
            or report.recognized_intervals != len(output)
            or report.maximum_retained_frames > 5
            or report.ocr_calls > len(intervals) * 5
            or report.missed_intervals != 0
            or len(negative_intervals) != 0
            or len(output) != len(reference)
            or score["strict_both_edges_250ms_and_text_wrap"] < 1
        ):
            raise RuntimeError("real streaming subtitle integration quality gate failed")
        return {
            "schema": 1,
            "original": "W3C Sintel trailer, Blender Foundation CC BY 3.0",
            "source_sha256_verified": True,
            "source_byte_count": original_bytes,
            "source_duration_seconds": media["duration_seconds"],
            "language": LANGUAGE,
            "constructed_subtitles_over_published_animation": True,
            "natural_original_hard_subs_verified": False,
            "official_vsf_native_executable_verified": True,
            "official_archive_cache_sha256_verified": True,
            "official_native_cache_reused_without_download": True,
            "source_identical_for_native_detector_and_hse_ocr": True,
            "real_ffmpeg_pts_decoded": True,
            "native_intervals": len(intervals),
            "native_negative_intervals": len(negative_intervals),
            "native_positive_detection_seconds": detect_seconds,
            "real_model_calls": report.ocr_calls,
            "maximum_retained_cropped_images": report.maximum_retained_frames,
            "positive_cues": len(output),
            "strict_exact_text_physical_wrap_and_both_edges_250ms":
                score["strict_both_edges_250ms_and_text_wrap"],
            "character_error_rate": score["aggregate_character_error_rate"],
            "srt_ass_match": True,
            "no_user_files_changed": True,
            "study_media_and_native_binary_retained": False,
            "release_gate_passed": False,
        }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", type=Path, required=True)
    ap.add_argument("--manifest", type=Path, default=Path("models/manifest.json"))
    args = ap.parse_args()
    if len(EXPECTED_SOURCE_SHA256) != 64:
        print("HSE_SINTEL_STREAM_ERROR " + json.dumps({
            "error_class": "SourceSHAUnpinned",
        }), flush=True)
        return 2
    try:
        summary = run_real_streaming_trial(args.model_dir, args.manifest)
    except Exception as exc:
        print("HSE_SINTEL_STREAM_ERROR " + json.dumps({
            "error_class": type(exc).__name__,
            "successful_proof": False,
        }), flush=True)
        return 3
    print("HSE_SINTEL_STREAM_PASS " + json.dumps(summary, sort_keys=True),
          flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
