"""Measure real production streaming OCR on independently generated two-line MP4s.

This intentionally uses the CURRENT extractor, not the timing experiment's
known-midpoint OCR oracle. A CI success certifies that measurement completed,
not that every cue was recognized. No video, transcript, image or user profile
is retained after the run. Production files are never modified by this script.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import tempfile

from hardsub_extractor.ocr.base import comparison_key, normalize_text
from verify_animated_background_ocr_quality import (
    _multiline_reference, check_language,
)
from verify_multilang_video_pipeline import VIDEO_TEXT

LANGUAGES = ("ko", "en", "ja", "zh")

from verify_multilang_video_pipeline import EXPECTED_SPANS

class InMemorySyntheticSegmentAudit:
    """Capture only bounded per-sample text/times; no video frames or file writes."""

    def __init__(self):
        self.segments = []

    def record_segment(self, index, segment, observations, voted):
        if len(self.segments) >= 200:
            raise RuntimeError("synthetic segment diagnostic exceeded bounded size")
        self.segments.append((
            segment.start, segment.end,
            tuple((obs.pts, obs.text, obs.metadata or {}) for obs in observations),
            voted.text,
        ))


def summarize_synthetic_evidence(audit, language):
    """Compare OCR samples independently of final SRT, omit all transcript text."""
    summary = []
    for index, span in enumerate(EXPECTED_SPANS):
        truth = VIDEO_TEXT[language][index]
        upper, lower = _multiline_reference(truth)
        upper_key, lower_key = comparison_key(upper), comparison_key(lower)
        full_key = comparison_key(truth)
        samples = [
            (pts, text, metadata)
            for _start, _end, group, _voted in audit.segments
            for pts, text, metadata in group
            if span[0] <= pts < span[1]
        ]
        upper_hits = sum(
            bool(upper_key and upper_key in comparison_key(text))
            for _, text, _ in samples
        )
        lower_hits = sum(
            bool(lower_key and lower_key in comparison_key(text))
            for _, text, _ in samples
        )
        both_hits = sum(
            comparison_key(text) == full_key and len(text.splitlines()) == 2
            for _, text, _ in samples
        )
        upper_boxes = sum(
            isinstance(record, dict)
            and isinstance(record.get("cy"), (float, int))
            and record["cy"] < 0.30
            for _, _, meta in samples
            for record in meta.get("detections", [])
        )
        windows_with_overlap = [
            (start, end, voted)
            for start, end, _group, voted in audit.segments
            if start < span[1] and end > span[0]
        ]
        summaries_with_overlap = [voted for _, _, voted in windows_with_overlap]
        summary.append({
            "sample_count": len(samples),
            "recognized_upper_line_samples": upper_hits,
            "recognized_lower_line_samples": lower_hits,
            "both_lines_exact_samples": both_hits,
            "upper_ocr_detection_boxes": upper_boxes,
            "overlapping_segment_votes": len(summaries_with_overlap),
            "synthetic_segment_windows": [
                {"start": round(start, 3), "end": round(end, 3),
                 "two_line_voted": len(voted.splitlines()) == 2}
                for start, end, voted in windows_with_overlap[:12]
            ],
            "two_line_vote_count": sum(
                len(t.splitlines()) == 2 for t in summaries_with_overlap
            ),
            "first_both_lines_pts": round(min(
                (pts for pts, text, _ in samples
                 if comparison_key(text) == full_key
                 and len(text.splitlines()) == 2),
                default=-1.0,
            ), 3),
        })
    return summary


def score_case(row: dict, language: str) -> dict:
    """Independent known text + two visible lines + correct video-time gate."""
    expected = VIDEO_TEXT[language]
    observations = row.get("reference_comparison", [])
    checks = []
    for index, truth in enumerate(expected):
        observed = observations[index] if index < len(observations) else {}
        normalized = str(observed.get("observed_synthetic", ""))
        expected_lines = _multiline_reference(truth)
        line_match = len(normalized.splitlines()) == len(expected_lines)
        # Existing CER and comparison_key disregard whitespace/punctuation
        # and fold case, so the historic "exact" metric is only normalized
        # text equality. Do not silently equate it with the literal visible
        # caption, including punctuation and the intended physical wrap.
        content_match = comparison_key(normalized) == comparison_key(truth)
        reference_surface = normalize_text("\n".join(expected_lines))
        observed_surface = normalize_text(normalized)
        strict_visible_match = bool(
            observed_surface and observed_surface == reference_surface
        )
        exact_line_wrap = bool(
            len(normalized.splitlines()) == len(expected_lines)
            and all(normalize_text(a) == normalize_text(b) for a, b in
                    zip(normalized.splitlines(), expected_lines))
        )
        cer = observed.get("cer")
        start_error = observed.get("start_error")
        end_error = observed.get("end_error")
        # A quality gate must reject missing, negative, boolean, NaN and
        # infinite timing observations. Python's max(0.0, nan) can return
        # 0.0 and falsely pass an invalid end boundary.
        times_acceptable = (
            type(start_error) in (int, float)
            and type(end_error) in (int, float)
            and math.isfinite(start_error)
            and math.isfinite(end_error)
            and 0.0 <= start_error <= 0.65
            and 0.0 <= end_error <= 0.65
        )
        checks.append({
            "reference_has_two_lines": len(expected_lines) == 2,
            "observed_two_lines": line_match,
            "observed_start_seconds": observed.get("observed_start_seconds"),
            "observed_end_seconds": observed.get("observed_end_seconds"),
            "start_error_seconds": start_error if type(start_error) in (int, float)
                                 and math.isfinite(start_error) else None,
            "end_error_seconds": end_error if type(end_error) in (int, float)
                               and math.isfinite(end_error) else None,
            # Backward compatible name, with explicit normalization caveat.
            "exact_reference_characters": content_match,
            "normalized_reference_text_match": content_match,
            "strict_visible_text_match": strict_visible_match,
            "exact_physical_line_wrap": exact_line_wrap,
            "character_error_rate": cer,
            "timing_acceptable": times_acceptable,
            "joint_acceptable": bool(
                observed.get("present")
                and line_match and content_match and times_acceptable
            ),
            "strict_joint_acceptable": bool(
                observed.get("present")
                and strict_visible_match and exact_line_wrap
                and times_acceptable
            ),
        })
    return {
        "language": language,
        "real_encoded_mp4_and_production_streaming_ocr": True,
        "reported_cues": row["cues_detected"],
        "expected_cues": len(expected),
        "srt_ass_consistent": row["matched_srt_ass"],
        "ocr_calls": row["ocr_calls"],
        "retry_calls": row["ocr_retry_calls"],
        "checks": checks,
        "all_cues_complete": (
            row["cues_detected"] == len(expected)
            and row["matched_srt_ass"]
            and all(x["joint_acceptable"] for x in checks)
        ),
        "all_cues_strict_visible": (
            row["cues_detected"] == len(expected)
            and row["matched_srt_ass"]
            and all(x["strict_joint_acceptable"] for x in checks)
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--manifest", default="models/manifest.json")
    parser.add_argument("--languages", default="ko,en,ja,zh",
                        help="research-only subset; never claims full release qualification")
    args = parser.parse_args()
    selected = tuple(s.strip() for s in args.languages.split(","))
    if not selected or len(set(selected)) != len(selected) or any(
        language not in LANGUAGES for language in selected
    ):
        parser.error("invalid distinct OCR language subset")
    rows = []
    try:
        with tempfile.TemporaryDirectory(prefix="hse-true-streaming-two-line-") as td:
            for lang in selected:
                path = Path(td) / lang
                path.mkdir()
                try:
                    audit = InMemorySyntheticSegmentAudit()
                    source = check_language(
                        path, lang, Path(args.model_dir),
                        Path(args.manifest), style="white_two_line",
                        diagnostics=audit,
                    )
                    row = score_case(source, lang)
                    row["synthetic_sample_evidence"] = summarize_synthetic_evidence(
                        audit, lang,
                    )
                except Exception as exc:
                    # Only bounded exception class; no source text or path.
                    row = {
                        "language": lang,
                        "full_path_executed": False,
                        "error_class": type(exc).__name__,
                        "all_cues_complete": False,
                    }
                rows.append(row)
                print("HSE_TWO_LINE_PIPELINE_CASE " +
                      json.dumps(row, sort_keys=True), flush=True)
    except Exception as exc:
        print("HSE_TWO_LINE_PIPELINE_FATAL " +
              json.dumps({"error_class": type(exc).__name__}))
        return 3
    successes = sum(bool(x.get("all_cues_complete")) for x in rows)
    strict_successes = sum(bool(x.get("all_cues_strict_visible")) for x in rows)
    result = {
        "schema": 1,
        "study_completed": len(rows) == len(selected),
        "tested_languages": list(selected),
        "full_four_language_coverage": len(selected) == len(LANGUAGES),
        "model_paths_successful": sum(
            bool(x.get("real_encoded_mp4_and_production_streaming_ocr"))
            for x in rows
        ),
        "fully_accepted_languages": successes,
        "fully_accepted_cues": sum(
            sum(bool(c["joint_acceptable"]) for c in x.get("checks", []))
            for x in rows
        ),
        "fully_strict_visible_languages": strict_successes,
        "fully_strict_visible_cues": sum(
            sum(bool(c["strict_joint_acceptable"]) for c in x.get("checks", []))
            for x in rows
        ),
        "normalized_metric_ignores_punctuation_case_and_exact_wrap": True,
        "total_expected_cues": 2 * len(selected),
        "production_modified": False,
        "real_anime_quality_verified": False,
        "user_media_opened": False,
        "known_midpoint_ocr_oracle_used": False,
        # Even 8/8 synthetic success is NOT a release approval. The
        # independent naturally hard-subtitled licensed-media evidence,
        # blank-scene controls, live provider transport and frozen EXE
        # acceptance are separate, currently unmet requirements.
        "synthetic_fidelity_gate_passed": (
            len(selected) == len(LANGUAGES)
            and strict_successes == len(LANGUAGES)
        ),
        "release_gate_passed": False,
        "release_requirements_missing": [
            "independent_naturally_burned_in_video_reference",
            "no_caption_false_positive_acceptance",
            "live_supported_site_transport_acceptance",
            "frozen_exe_acceptance_for_this_commit",
        ],
    }
    print("HSE_TWO_LINE_PIPELINE_SUMMARY " +
          json.dumps(result, sort_keys=True))
    # Research-only: return success only for completed *execution*, never
    # let this status masquerade as release_gate_passed.
    return 0 if (
        result["study_completed"]
        and result["model_paths_successful"] == len(selected)
    ) else 4


if __name__ == "__main__":
    raise SystemExit(main())
