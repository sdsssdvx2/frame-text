"""Actual licensed animation background with artificial source-language captions.

RESEARCH ONLY. Original Sintel trailer footage is downloaded from a fixed
W3C HTTPS URL with exact SHA pin and deleted with all derived media. This is
NOT a naturally distributed hard-sub title or an end-user anime accuracy gate.

Compare (1) the current guarded HSE detector plus PP-OCRv5 and (2) a pinned
official isolated native VSF temporal detector plus the SAME PP-OCRv5 model.
Both operate on byte-identical independently encoded positive MP4s. Blank
controls have identical REAL motion and NO ADDED CAPTIONS; native movie title
cards/credits are not separately masked or mislabeled as totally text-free.
Reference text, line breaks and intervals are never passed to inference.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile
import time

import numpy as np

from compare_external_engine_exports import Cue as ScoreCue, evaluate
from compare_quiet_segment_caps import check_blank_control
from compare_vsf_hse_ocr_research import (
    original_pts_frames, predict_from_frames, validated_output_score,
)
from compare_vsf_timing_research import run_vsf, check_native_version
from probe_sintel_original_animation import (
    EXPECTED_SOURCE_SHA256, fetch_source, probe_duration_and_dimensions,
)
from verify_animated_background_ocr_quality import (
    FPS, HEIGHT, WIDTH, _multiline_reference, subtitle_overlay, check_language,
)
from verify_multilang_video_pipeline import VIDEO_TEXT, SEGMENTS, EXPECTED_SPANS
from verify_multilang_ocr_synthetic import choose_font
from verify_vsf_cli_research import download_pinned_archive, unpack_pinned_archive

from hardsub_extractor.detector import DetectorConfig
from hardsub_extractor.ocr.rapidocr_adapter import RapidOCRAdapter
from hardsub_extractor.runtime_paths import resolve_binary
from hardsub_extractor.subtitle_formats import parse_subtitle_bytes

LANGUAGES = ("ko", "en", "ja", "zh")
STUDY_START_SECONDS = 12.0
MAX_ENCODED_BYTES = 32 * 1024 * 1024
FRAME_COUNT = SEGMENTS[-1][1]
FRAME_BYTES = WIDTH * HEIGHT * 3


def _encoded_command(destination: Path) -> list[str]:
    if destination.exists() or destination.is_symlink():
        raise RuntimeError("preexisting derived video")
    return [
        resolve_binary("ffmpeg"), "-nostdin", "-hide_banner",
        "-loglevel", "error",
        "-f", "rawvideo", "-pixel_format", "bgr24",
        "-video_size", f"{WIDTH}x{HEIGHT}",
        "-framerate", str(FPS), "-i", "pipe:0", "-an",
        "-c:v", "mpeg4", "-q:v", "4", "-pix_fmt", "yuv420p",
        "-movflags", "+faststart", "-y", str(destination),
    ]


def _decode_command(original: Path) -> list[str]:
    return [
        resolve_binary("ffmpeg"), "-nostdin", "-hide_banner",
        "-loglevel", "error", "-protocol_whitelist", "file,pipe",
        "-ss", str(STUDY_START_SECONDS),
        "-i", str(original), "-t", str(FRAME_COUNT / FPS),
        "-an", "-vf",
        f"fps={FPS},scale={WIDTH}:{HEIGHT}:force_original_aspect_ratio=decrease,"
        f"pad={WIDTH}:{HEIGHT}:(ow-iw)/2:(oh-ih)/2",
        "-frames:v", str(FRAME_COUNT), "-f", "rawvideo",
        "-pix_fmt", "bgr24", "pipe:1",
    ]


def _read_frame(stream) -> bytes:
    remaining = FRAME_BYTES
    chunks = []
    while remaining:
        part = stream.read(remaining)
        if not part:
            raise RuntimeError("not enough real-film decoded frames")
        chunks.append(part)
        remaining -= len(part)
    return b"".join(chunks)


def make_original_motion_pair(
    original: Path, positive: Path, negative: Path, language: str,
) -> dict:
    """Encode 57 matching REAL-animation frames, adding text to one only.

    Memory is bounded to one decoded image and two temporary render buffers.
    Use independent source timestamps and caption reference made before OCR.
    """
    if language not in LANGUAGES:
        raise ValueError("unsupported language")
    overlays = [
        subtitle_overlay(text, choose_font(language), style="white_two_line")
        for text in VIDEO_TEXT[language]
    ]
    references = {}
    for a, b, label in SEGMENTS:
        references.update({i: label for i in range(a, b)})
    if len(references) != FRAME_COUNT:
        raise RuntimeError("noncontiguous fixed original caption schedule")
    decoder = subprocess.Popen(
        _decode_command(original), stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
    )
    encoders = [
        subprocess.Popen(
            _encoded_command(path), stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        for path in (positive, negative)
    ]
    try:
        if decoder.stdout is None or any(p.stdin is None for p in encoders):
            raise RuntimeError("unable to stream real original animation")
        for index in range(FRAME_COUNT):
            raw = _read_frame(decoder.stdout)
            original_frame = np.frombuffer(raw, dtype=np.uint8).reshape(
                (HEIGHT, WIDTH, 3)
            )
            overlay_index = references[index]
            if overlay_index is None:
                captioned = raw
            else:
                overlay = overlays[overlay_index]
                alpha = overlay[:, :, 3:4].astype(np.uint16)
                captioned = ((
                    original_frame.astype(np.uint16) * (255 - alpha)
                    + overlay[:, :, :3].astype(np.uint16) * alpha + 127
                ) // 255).astype(np.uint8).tobytes()
            encoders[0].stdin.write(captioned)
            encoders[1].stdin.write(raw)
        if decoder.stdout.read(1):
            raise RuntimeError("real-film frame limit was exceeded")
        decoder.stdout.close()
        for encoder in encoders:
            encoder.stdin.close()
        if decoder.wait(timeout=40) != 0:
            raise RuntimeError("pinned original animation decoder failed")
        for encoder in encoders:
            if encoder.wait(timeout=100) != 0:
                raise RuntimeError("derived animation encoder failed")
    finally:
        if decoder.poll() is None:
            decoder.kill()
        for encoder in encoders:
            if encoder.poll() is None:
                encoder.kill()
        decoder.wait(timeout=12)
        for encoder in encoders:
            encoder.wait(timeout=12)
    for output in (positive, negative):
        if not output.is_file() or not (
            1024 <= output.stat().st_size <= MAX_ENCODED_BYTES
        ):
            raise RuntimeError("real-film derived video size invalid")
    return {
        "frames_same_original_real_animated_footage": FRAME_COUNT,
        "caption_source_independent_of_inference": True,
        "no_added_caption_control": True,
        "positive_sha256": hashlib.sha256(positive.read_bytes()).hexdigest(),
        "negative_sha256": hashlib.sha256(negative.read_bytes()).hexdigest(),
    }


def run_case(
    folder: Path, original: Path, vsf: Path, language: str,
    models: Path, manifest: Path,
) -> dict:
    folder.mkdir()
    positive = folder / "caption.mp4"
    negative = folder / "no-added-caption.mp4"
    fingerprint = make_original_motion_pair(
        original, positive, negative, language,
    )
    reference = [
        ScoreCue(start, end, "\n".join(_multiline_reference(text)))
        for (start, end), text in zip(EXPECTED_SPANS, VIDEO_TEXT[language])
    ]
    detected, native_seconds = run_vsf(vsf, positive, folder / "vsf-positive")
    no_caption_detected, blank_seconds = run_vsf(
        vsf, negative, folder / "vsf-negative",
    )
    # Use interval list produced by official native executable, not truth.
    selected = original_pts_frames(positive, detected)
    if detected and not all(selected):
        raise RuntimeError("native interval lacks original decoded PTS frames")
    adapter = RapidOCRAdapter(
        language=language, model_dir=str(models),
        manifest_path=str(manifest),
    )
    native_start = time.monotonic()
    predicted, calls = predict_from_frames(detected, selected, adapter)
    score_native = validated_output_score(
        predicted, reference, folder / "vsf-ocr",
        no_caption_detected,
    )
    native_ocr_seconds = round(time.monotonic() - native_start, 2)
    del adapter
    gc.collect()

    baseline_dir = folder / "hse"
    baseline_dir.mkdir()
    try:
        # Always use the same PTS decoder, image bytes and model file.
        baseline = check_language(
            baseline_dir, language, models, manifest,
            source_override=positive, style="white_two_line",
            detector_config=DetectorConfig(
                max_quiet_segment_seconds=0.625,
                max_pending_change_seconds=0.375,
            ),
        )
    except RuntimeError as exc:
        if not str(exc).startswith("인식된 자막이 없습니다."):
            raise
        baseline = {
            "cues_detected": 0,
            "ocr_calls": None,
            "elapsed_seconds": None,
            "no_recognized_cues": True,
        }
        output_cues = []
    else:
        srt = baseline_dir / (language + ".srt")
        _, parsed = parse_subtitle_bytes(srt.read_bytes(), source_url="real-film.srt")
        output_cues = [ScoreCue(c.start, c.end, c.text) for c in parsed]
    baseline_blank_dir = baseline_dir / "blank"
    baseline_blank_dir.mkdir()
    blank = check_blank_control(
        baseline_blank_dir, language, negative, models, manifest, 0.625, 0.375,
    )
    if not blank.get("executed"):
        raise RuntimeError("baseline negative real-film video was not processed")
    score_baseline = evaluate(reference, output_cues, [])
    score_baseline["blank_video_false_cues"] = blank["false_cues"]
    score_baseline["no_caption_control_pass"] = blank["false_cues"] == 0
    score_baseline["all_reference_cues_exact_both_edges_250ms"] &= (
        blank["false_cues"] == 0
    )
    # The two encodes are known exact via hashes, not video names alone.
    if any(
        hashlib.sha256(path.read_bytes()).hexdigest() != fingerprint[which]
        for path, which in (
            (positive, "positive_sha256"), (negative, "negative_sha256")
        )
    ):
        raise RuntimeError("comparison input media were modified")
    return {
        "language": language,
        "study_scope": "real_original_sintel_animation_plus_constructed_subtitles",
        "encoded_frames": FRAME_COUNT,
        "verified_same_positive_media_for_both_methods": True,
        "matched_negative_video": True,
        "native_detection_seconds": native_seconds,
        "native_negative_detection_seconds": blank_seconds,
        "native_detected_intervals": len(detected),
        "native_negative_intervals": len(no_caption_detected),
        "methods": {
            "vsf_plus_hse_v5": {
                "source_corrected_score": score_native,
                "actual_ocr_calls": calls,
                "ocr_and_export_seconds": native_ocr_seconds,
                "native_interval_source": "independent official native VSF",
            },
            "guarded_hse_v5": {
                "source_corrected_score": score_baseline,
                "actual_ocr_calls": baseline.get("ocr_calls"),
                "ocr_and_export_seconds": baseline.get("elapsed_seconds"),
                "negative_media_ocr_ran": True,
            },
        },
        "no_auth_drm_or_browser_used": True,
        "production_changed": False,
        "release_gate_passed": False,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", type=Path, required=True)
    ap.add_argument("--manifest", type=Path, default=Path("models/manifest.json"))
    args = ap.parse_args()
    if len(EXPECTED_SOURCE_SHA256) != 64:
        print("HSE_SINTEL_FILM_ERROR " + json.dumps({
            "error_class": "OriginalSHAUnpinned", "benchmarked": False,
        }), flush=True)
        return 3
    cases = []
    try:
        with tempfile.TemporaryDirectory(prefix="hse-real-sintel-vsf-ocr-") as td:
            root = Path(td)
            original = root / "original-sintel.mp4"
            sha, size = fetch_source(
                original, required_hash=EXPECTED_SOURCE_SHA256,
            )
            media = probe_duration_and_dimensions(original)
            archive = download_pinned_archive(root / "vsf.zip")
            exe = unpack_pinned_archive(archive, root / "vsf")
            check_native_version(exe)
            for language in LANGUAGES:
                row = run_case(
                    root / language, original, exe, language,
                    args.model_dir, args.manifest,
                )
                cases.append(row)
                print(
                    "HSE_SINTEL_FILM_CASE " + json.dumps(row, sort_keys=True),
                    flush=True,
                )
            if hashlib.sha256(original.read_bytes()).hexdigest() != sha:
                raise RuntimeError("original licensed animation bytes changed")
    except Exception as exc:
        print("HSE_SINTEL_FILM_ERROR " + json.dumps({
            "error_class": type(exc).__name__,
            "completed_languages": [item["language"] for item in cases],
            "benchmark_completed": False,
        }), flush=True)
        return 4
    print("HSE_SINTEL_FILM_SUMMARY " + json.dumps({
        "schema": 1,
        "source": "W3C-hosted Blender Foundation Sintel trailer",
        "source_sha256_verified": True,
        "original_bytes": size,
        "source_metadata": media,
        "tested_languages": [x["language"] for x in cases],
        "constructed_subtitles_on_real_film": True,
        "original_natural_hard_subtitles": False,
        "same_source_bytes_compared_per_case": True,
        "no_added_caption_moving_controls": True,
        "external_native_source_verified": True,
        "native_binary_bundled": False,
        "media_and_models_retained": False,
        "release_gate_passed": False,
    }, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
