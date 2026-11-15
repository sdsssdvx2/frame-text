"""VideoSubFinder 6.10.5 timing + real HSE PP-OCRv5/v6 text: isolated CJK A/B.

The native detector chooses intervals without ever seeing expected dialogue.
Both OCR engines see exactly the same original encoded MP4 PTS-selected
frames; the v6 test keeps v5 detector, changes only official SHA-pinned
recognition model and dictionary. No human transcript enters selection,
voting, native executable or OCR engine. Results include real SRT/ASS files
saved exclusively inside a fresh disposable directory, removed on exit.
NOT the production HSE pipeline, NOT a legal real-anime benchmark.
"""
from __future__ import annotations

import argparse
from contextlib import closing
import hashlib
import json
import math
from pathlib import Path
import tempfile
import time

from compare_cjk_v5_v6_recognition import (
    V6_RECOGNIZER, V6_DICTIONARY, retrieve_verified, make_v6_recognizer,
)
from compare_external_engine_exports import evaluate as grade_exports, Cue as ScoreCue
from compare_pending_timeout_ab import LANGUAGES
from compare_quiet_segment_caps import check_blank_control
from compare_vsf_timing_research import (
    run_vsf, check_native_version, timing_report,
)
from verify_vsf_cli_research import download_pinned_archive, unpack_pinned_archive
from verify_animated_background_ocr_quality import encode_video, check_language, _multiline_reference
from verify_multilang_video_pipeline import EXPECTED_SPANS, VIDEO_TEXT
from hardsub_extractor.detector import DetectorConfig
from hardsub_extractor.frame_source import FFmpegPTSFrameSource
from hardsub_extractor.models import Frame, Cue
from hardsub_extractor.ocr.base import vote_observations
from hardsub_extractor.ocr.rapidocr_adapter import RapidOCRAdapter
from hardsub_extractor.writers import write_subtitle_pair_exclusive
from hardsub_extractor.subtitle_formats import parse_subtitle_bytes

FRACTIONS = (0.15, 0.325, 0.50, 0.675, 0.85)
MAX_TIMED_CUES = 8
MAX_VIDEO_BYTES = 32 * 1024 * 1024
MAX_SELECTED_FRAMES = 60


def selection_targets(cue: ScoreCue) -> tuple[float, ...]:
    if not (
        math.isfinite(cue.start) and math.isfinite(cue.end)
        and cue.end > cue.start
    ):
        raise ValueError("invalid timed interval")
    return tuple(
        cue.start + (cue.end - cue.start) * fraction for fraction in FRACTIONS
    )


def original_pts_frames(
    video: Path, intervals: list[ScoreCue],
) -> list[list[Frame]]:
    """Select nearest independent FFmpeg video PTS without any truth labels."""
    if len(intervals) > MAX_TIMED_CUES:
        raise ValueError("timing interval count out of bounded study scope")
    targets = [selection_targets(cue) for cue in intervals]
    retained = [[None] * len(FRACTIONS) for _ in intervals]
    distances = [[float("inf")] * len(FRACTIONS) for _ in intervals]
    with closing(iter(FFmpegPTSFrameSource(str(video), sample_fps=8.0))) as frames:
        for frame in frames:
            pts = frame.pts
            for index, (start_end, points) in enumerate(zip(intervals, targets)):
                if not start_end.start <= pts < start_end.end:
                    continue
                for j, pos in enumerate(points):
                    distance = abs(pts - pos)
                    if distance < distances[index][j] and distance <= 0.17:
                        retained[index][j] = frame
                        distances[index][j] = distance
    result = []
    for group in retained:
        used = set()
        selected = []
        for frame in group:
            if frame is not None and frame.pts not in used:
                used.add(frame.pts)
                selected.append(frame)
        result.append(selected)
    if sum(map(len, result)) > MAX_SELECTED_FRAMES:
        raise ValueError("too many retained research OCR frames")
    return result


def predict_from_frames(
    intervals: list[ScoreCue], frames: list[list[Frame]], adapter,
) -> tuple[list[Cue], int]:
    if len(intervals) != len(frames):
        raise ValueError("prediction intervals and frames differ")
    predicted = []
    calls = 0
    for slot, original_frames in zip(intervals, frames):
        observations = []
        for frame in original_frames:
            # No script language correction, expected phrase or timing
            # reference is passed into this actual ONNX inference.
            observations.append(adapter.recognize(frame))
            calls += 1
        voted = vote_observations(observations)
        if voted.text:
            predicted.append(Cue(
                slot.start, slot.end, voted.text, voted.confidence,
            ))
    return predicted, calls


def validated_output_score(cues: list[Cue], reference: list[ScoreCue], out_base: Path,
                           blank_intervals: list[ScoreCue]) -> dict:
    srt, ass = write_subtitle_pair_exclusive(cues, out_base)
    if cues:
        _, actual_srt = parse_subtitle_bytes(srt.read_bytes(), source_url="export.srt")
        _, actual_ass = parse_subtitle_bytes(ass.read_bytes(), source_url="export.ass")
        if len(actual_srt) != len(actual_ass) or len(actual_srt) != len(cues):
            raise RuntimeError("SRT/ASS cue count mismatch")
        if any(
            a.text != b.text or abs(a.start - b.start) > .011
            or abs(a.end - b.end) > .011
            for a, b in zip(actual_srt, actual_ass)
        ):
            raise RuntimeError("SRT and ASS output disagree")
        candidate = [
            ScoreCue(v.start, v.end, v.text) for v in actual_srt
        ]
    else:
        candidate = []
        if srt.stat().st_size or not ass.is_file():
            raise RuntimeError("zero-cue export mismatch")
    return grade_exports(reference, candidate, blank_intervals)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", required=True, type=Path)
    parser.add_argument("--manifest", default="models/manifest.json", type=Path)
    opt = parser.parse_args()
    reports = []
    try:
        with tempfile.TemporaryDirectory(prefix="hse-vsf-plus-ocr-") as tmp:
            root = Path(tmp)
            exe = unpack_pinned_archive(
                download_pinned_archive(root / "vsf-official.zip"), root / "vsf-bin",
            )
            check_native_version(exe)
            official_model_root = root / "v6-model"
            official_model_root.mkdir()
            rec = retrieve_verified(V6_RECOGNIZER, official_model_root)
            dictionary = retrieve_verified(V6_DICTIONARY, official_model_root)
            for lang in LANGUAGES:
                case_root = root / lang
                case_root.mkdir()
                src, neg = case_root / "caption.mp4", case_root / "blank.mp4"
                encode_video(src, lang, style="white_two_line")
                encode_video(neg, lang, style="white_two_line", include_subtitles=False)
                if max(src.stat().st_size, neg.stat().st_size) > MAX_VIDEO_BYTES:
                    raise RuntimeError("generated MP4 outside study size bound")
                file_hashes = {
                    path: hashlib.sha256(path.read_bytes()).hexdigest()
                    for path in (src, neg)
                }
                observed, native_seconds = run_vsf(exe, src, case_root / "timing")
                false, blank_seconds = run_vsf(exe, neg, case_root / "blank-timing")
                reference = [
                    ScoreCue(start, end, "\n".join(_multiline_reference(text)))
                    for (start, end), text in zip(EXPECTED_SPANS, VIDEO_TEXT[lang])
                ]
                anchors = original_pts_frames(src, observed)
                if observed and not all(anchors):
                    raise RuntimeError("native interval produced no source anchor frames")
                v5 = RapidOCRAdapter(
                    language=lang, model_dir=str(opt.model_dir),
                    manifest_path=str(opt.manifest),
                )
                v6 = RapidOCRAdapter(
                    language=lang, model_dir=str(opt.model_dir),
                    manifest_path=str(opt.manifest),
                )
                v6.engine = make_v6_recognizer(
                    opt.model_dir, opt.manifest, lang, rec, dictionary,
                )
                native_blank = [
                    ScoreCue(x.start, x.end, x.text) for x in false
                ]
                comparisons = {}
                for name, adapter in (("vsf_v5", v5), ("vsf_v6", v6)):
                    started = time.monotonic()
                    predicted, calls = predict_from_frames(observed, anchors, adapter)
                    scores = validated_output_score(
                        predicted, reference, case_root / name,
                        native_blank,
                    )
                    comparisons[name] = {
                        "scores": scores,
                        "original_source_pts_sampled": sum(map(len, anchors)),
                        "native_blank_ocr_calls": 0,
                        "model_calls": calls,
                        "inference_and_export_seconds": round(
                            time.monotonic() - started, 2,
                        ),
                        "produced_real_srt_ass": True,
                    }
                baseline_root = case_root / "hse-guarded"
                baseline_root.mkdir()
                baseline = check_language(
                    baseline_root, lang, opt.model_dir, opt.manifest,
                    source_override=src, style="white_two_line",
                    detector_config=DetectorConfig(
                        max_quiet_segment_seconds=.625,
                        max_pending_change_seconds=.375,
                    ),
                )
                base_srt = baseline_root / (lang + ".srt")
                base_ass = baseline_root / (lang + ".ass")
                if not base_srt.exists() or not base_ass.exists():
                    raise RuntimeError("guarded HSE baseline failed to export")
                _, hse_cues = parse_subtitle_bytes(
                    base_srt.read_bytes(), source_url="baseline.srt",
                )
                _, hse_ass = parse_subtitle_bytes(
                    base_ass.read_bytes(), source_url="baseline.ass",
                )
                if len(hse_cues) != len(hse_ass):
                    raise RuntimeError("baseline SRT/ASS structure differs")
                blank_root = baseline_root / "no-caption"
                blank_root.mkdir()
                blank_result = check_blank_control(
                    blank_root, lang, neg, opt.model_dir,
                    opt.manifest, .625, .375,
                )
                if not blank_result.get("executed"):
                    raise RuntimeError("guarded HSE blank-control execution failed")
                # A blank control's count is measured by the existing
                # independent test. No fake placeholder timings are invented.
                guarded_scores = grade_exports(reference, [
                    ScoreCue(x.start, x.end, x.text) for x in hse_cues
                ], [])
                guarded_scores["blank_video_false_cues"] = blank_result["false_cues"]
                guarded_scores["no_caption_control_pass"] = (
                    blank_result["false_cues"] == 0
                )
                guarded_scores["all_reference_cues_exact_both_edges_250ms"] = (
                    guarded_scores["all_reference_cues_exact_both_edges_250ms"]
                    and blank_result["false_cues"] == 0
                )
                comparisons["hse_guarded_v5"] = {
                    "scores": guarded_scores,
                    "model_calls": baseline["ocr_calls"],
                    "inference_and_export_seconds": baseline["elapsed_seconds"],
                    "negative_video_ocr_calls": blank_result.get("ocr_calls"),
                    "produced_real_srt_ass": True,
                }
                if any(
                    hashlib.sha256(path.read_bytes()).hexdigest() != digest
                    for path, digest in file_hashes.items()
                ):
                    raise RuntimeError("original MP4 was unexpectedly modified")
                row = {
                    "language": lang,
                    "source_same_mp4_for_native_and_hse": True,
                    "native_release_sha_and_version_verified": True,
                    "v6_model_and_dictionary_verified": True,
                    "original_pts_selection_no_reference_text": True,
                    "reference_used_only_after_inference": True,
                    "native_timing": timing_report(observed),
                    "native_elapsed_seconds": native_seconds,
                    "native_negative_elapsed_seconds": blank_seconds,
                    "native_negative_false_intervals": len(false),
                    "methods": comparisons,
                }
                reports.append(row)
                print("HSE_VSF_OCR_CASE " + json.dumps(row, sort_keys=True), flush=True)
    except Exception as exc:
        print("HSE_VSF_OCR_ERROR " + json.dumps({
            "error_class": type(exc).__name__,
        }), flush=True)
        return 4
    print("HSE_VSF_OCR_SUMMARY " + json.dumps({
        "schema": 1,
        "complete": len(reports) == len(LANGUAGES),
        "languages": list(LANGUAGES),
        "method_names": ["vsf_v5", "vsf_v6", "hse_guarded_v5"],
        "synthetic_not_actual_anime": True,
        "third_party_binary_bundled": False,
        "production_models_modified": False,
        "user_installation_modified": False,
        "release_gate_passed": False,
    }, sort_keys=True), flush=True)
    return 0 if len(reports) == len(LANGUAGES) else 4


if __name__ == "__main__":
    raise SystemExit(main())
