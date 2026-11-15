"""A/B detector sample budgets on the SAME real moving two-line MP4 bytes.

Research ONLY. Measure whether more automatically selected samples recover
missing lines and true cue times without fabricating success. The deployed
3-sample profile is not touched. Model and generated files stay isolated.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import tempfile

from hardsub_extractor.detector import DetectorConfig
from verify_animated_background_ocr_quality import check_language, encode_video
from verify_multiline_streaming_ocr_quality import score_case

LANGUAGES = ("ko", "en", "ja", "zh")
SAMPLE_BUDGETS = (3, 5, 7)


def summarize_case(source, language, budget):
    measured = score_case(source, language)
    comparisons = source["reference_comparison"]
    # Per-index exact matches are not a valid full export when the stream
    # emits extra/missing cues or the SRT/ASS contract itself has failed.
    # An earlier research report counted a Japanese 3-cue output as one
    # exact cue against a 2-cue reference. Preserve that raw observation
    # separately, but NEVER promote it into the valid-export count.
    aligned_matches = sum(x["joint_acceptable"] for x in measured["checks"])
    export_shape_valid = (
        measured["reported_cues"] == measured["expected_cues"]
        and measured["srt_ass_consistent"]
    )
    return {
        "language": language,
        "max_samples_per_segment": budget,
        "joint_exact_cues": aligned_matches if export_shape_valid else 0,
        "aligned_reference_joint_matches": aligned_matches,
        "export_shape_valid": export_shape_valid,
        "two_line_cues": sum(x["observed_two_lines"] for x in measured["checks"]),
        "character_exact_cues": sum(
            x["exact_reference_characters"] for x in measured["checks"]
        ),
        "cue_count": measured["reported_cues"],
        "srt_ass_consistent": measured["srt_ass_consistent"],
        "max_cer": max((c.get("cer") or 0.0) for c in comparisons),
        "max_start_error": max((c.get("start_error") or 0.0) for c in comparisons),
        "max_end_error": max((c.get("end_error") or 0.0) for c in comparisons),
        "ocr_calls": measured["ocr_calls"],
        "retry_calls": measured["retry_calls"],
        "all_complete": measured["all_cues_complete"],
        "elapsed_seconds": source["elapsed_seconds"],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--manifest", default="models/manifest.json")
    args = parser.parse_args()
    results = []
    try:
        with tempfile.TemporaryDirectory(prefix="hse-multiline-sample-ab-") as td:
            root = Path(td)
            for language in LANGUAGES:
                inp = root / language
                inp.mkdir()
                video = inp / "identical-encoded-source.mp4"
                encode_video(video, language, style="white_two_line")
                original_sha = hashlib.sha256(video.read_bytes()).hexdigest()
                for budget in SAMPLE_BUDGETS:
                    folder = inp / ("budget-" + str(budget))
                    folder.mkdir()
                    try:
                        source = check_language(
                            folder, language, Path(args.model_dir),
                            Path(args.manifest), style="white_two_line",
                            source_override=video,
                            detector_config=DetectorConfig(
                                max_samples_per_segment=budget,
                            ),
                        )
                        row = summarize_case(source, language, budget)
                        row["complete_trial"] = True
                    except Exception as exc:
                        row = {
                            "language": language,
                            "max_samples_per_segment": budget,
                            "complete_trial": False,
                            "error_class": type(exc).__name__,
                        }
                    # No extra media downloads, edits or source rewrites.
                    if hashlib.sha256(video.read_bytes()).hexdigest() != original_sha:
                        raise RuntimeError("identical MP4 input was modified by OCR trial")
                    results.append(row)
                    print("HSE_SAMPLE_BUDGET_CASE " +
                          json.dumps(row, sort_keys=True), flush=True)
    except Exception as exc:
        print("HSE_SAMPLE_BUDGET_FATAL " +
              json.dumps({"error_class": type(exc).__name__}))
        return 3
    complete = all(row.get("complete_trial") for row in results)
    summaries = {}
    for budget in SAMPLE_BUDGETS:
        trials = [row for row in results if row["max_samples_per_segment"] == budget]
        summaries[str(budget)] = {
            "tested_languages": len(trials),
            "joint_exact_cues": sum(row.get("joint_exact_cues", 0) for row in trials),
            "aligned_reference_joint_matches": sum(
                row.get("aligned_reference_joint_matches", 0) for row in trials
            ),
            "valid_export_languages": sum(
                bool(row.get("export_shape_valid")) for row in trials
            ),
            "invalid_export_languages": [
                row["language"] for row in trials
                if not row.get("export_shape_valid")
            ],
            "two_line_cues": sum(row.get("two_line_cues", 0) for row in trials),
            "ocr_calls": sum(row.get("ocr_calls", 0) for row in trials),
        }
    result = {
        "schema": 1,
        "experiment_completed": complete and len(results) == 12,
        "same_encoded_video_bytes_for_all_budgets_per_language": True,
        "model_oracle_used_for_keyframe_selection": False,
        "existing_three_sample_profile_preserved": True,
        "production_modified": False,
        "real_anime_verified": False,
        "blank_control_completed": False,
        "candidate_deployed": False,
        "results": summaries,
    }
    print("HSE_SAMPLE_BUDGET_SUMMARY " + json.dumps(result, sort_keys=True))
    return 0 if result["experiment_completed"] else 4


if __name__ == "__main__":
    raise SystemExit(main())
