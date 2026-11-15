"""Detector-only cause isolation: show bounded segments on same encoded JA/ZH MP4.

NO OCR model, reference texts, site connections or user media. Ground-truth
cue times may label POST-detection sample positions in the output, but
do not influence segmentation. All generated MP4s are temporary.
"""
from __future__ import annotations

import argparse
from contextlib import closing
import hashlib
import json
import math
from pathlib import Path
import tempfile

from hardsub_extractor.detector import DetectorConfig, iter_segment_frames
from hardsub_extractor.frame_source import FFmpegPTSFrameSource
from verify_animated_background_ocr_quality import encode_video
from verify_multilang_video_pipeline import EXPECTED_SPANS

PROFILES = (
    ("default", None, None),
    ("quiet_625", 0.625, None),
    ("pending_375", 0.625, 0.375),
    ("pending_500", 0.625, 0.500),
)
LANGUAGES = ("ja", "zh")


def stage_for_pts(pts):
    """Ground-truth timing is used only to label completed detector output."""
    if not isinstance(pts, (int, float)) or not math.isfinite(pts):
        return "unknown"
    if EXPECTED_SPANS[0][0] <= pts < EXPECTED_SPANS[0][1]:
        return "first_caption"
    if EXPECTED_SPANS[1][0] <= pts < EXPECTED_SPANS[1][1]:
        return "second_caption"
    if EXPECTED_SPANS[0][1] <= pts < EXPECTED_SPANS[1][0]:
        return "caption_gap"
    return "other_background"


def summarize_segments(segments):
    rows = []
    for segment in segments:
        if segment.end <= 3.0 or segment.start >= 6.625:
            continue
        pts = [sample.pts for sample in segment.samples]
        diag = segment.diagnostics or {}
        rows.append({
            "start_seconds": round(segment.start, 3),
            "end_seconds": round(segment.end, 3),
            "selected_pts": [round(value, 3) for value in pts],
            "posthoc_sample_stage": [stage_for_pts(value) for value in pts],
            "boundary_kind": (
                "quiet_cap" if diag.get("quiet_duration_cap")
                else diag.get("settled_by")
                or ("pending_eof" if diag.get("pending_eof") else "eof")
            ),
            "sample_count": len(pts),
        })
    return rows


def run_source(video, cap, pending_timeout=None):
    events = []
    def record_state(event):
        if len(events) >= 256:
            raise RuntimeError("detector state trace exceeded hard bound")
        events.append(event)
    with closing(iter(FFmpegPTSFrameSource(str(video), sample_fps=8.0))) as frames:
        with closing(iter_segment_frames(
            frames, DetectorConfig(
                max_quiet_segment_seconds=cap,
                max_pending_change_seconds=pending_timeout,
            ),
            trace=record_state,
        )) as segments:
            windows = summarize_segments(segments)
    return windows, [
        e for e in events if 3.0 <= e["at_pts"] <= 6.625
    ][:96]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--languages", default="ja,zh")
    args = ap.parse_args()
    chosen = tuple(a.strip() for a in args.languages.split(","))
    if not chosen or len(set(chosen)) != len(chosen) or any(
        a not in LANGUAGES for a in chosen
    ):
        ap.error("languages must be distinct and subset of ja,zh")
    results = []
    try:
        with tempfile.TemporaryDirectory(prefix="hse-detector-window-audit-") as td:
            for language in chosen:
                video = Path(td) / (language+".mp4")
                encode_video(video, language, style="white_two_line")
                original_sha = hashlib.sha256(video.read_bytes()).hexdigest()
                for name, cap, timeout in PROFILES:
                    windows, state_events = run_source(video, cap, timeout)
                    if hashlib.sha256(video.read_bytes()).hexdigest() != original_sha:
                        raise RuntimeError("source MP4 modified across profiles")
                    row = {
                        "language": language,
                        "profile": name,
                        "cap_seconds": cap,
                        "pending_timeout_seconds": timeout,
                        "source_same_bytes_for_all_profiles": True,
                        "detector_runs_without_OCR": True,
                        "selected_window_count": len(windows),
                        "sample_windows": windows[:28],
                        "detector_state_events": state_events,
                        "pending_timeout_observed": any(
                            e["kind"] == "cap_blocked_by_pending"
                            for e in state_events
                        ),
                    }
                    results.append(row)
                    print("HSE_DETECTOR_WINDOW_ROW " +
                          json.dumps(row, sort_keys=True), flush=True)
    except Exception as exc:
        print("HSE_DETECTOR_WINDOW_ERROR "+
              json.dumps({"exception_class": type(exc).__name__}))
        return 3
    completed = len(results) == len(chosen)*len(PROFILES)
    print("HSE_DETECTOR_WINDOW_SUMMARY " + json.dumps({
        "schema": 1,
        "study_complete": completed,
        "language_coverage": list(chosen),
        "reference_used_for_detector_inputs": False,
        "real_anime_tested": False,
        "production_modified": False,
        "candidate_promoted": False,
    }, sort_keys=True))
    return 0 if completed else 4


if __name__ == "__main__":
    raise SystemExit(main())
