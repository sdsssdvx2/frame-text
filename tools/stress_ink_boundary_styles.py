"""Independent real-model boundary study on unseen subtitle layout/color.

Two generated 7.125-second encoded videos per language: two separate
white-on-black outlined lines and saturated yellow-on-black-outlined
single-line cues. Each has an identical no-caption moving-background
control. The original baseline is unchanged, never released.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import tempfile

from experiment_ink_boundary_refinement import run_language


STYLES = ("white_two_line", "yellow_single")
LANGUAGES = ("ko", "en", "ja", "zh")


def strict(result: dict) -> bool:
    return bool(
        result.get("anchor_verified")
        and result.get("false_on_blank_control") == 0
        and result.get("false_non_cue_frames") == 0
        and result.get("start_error_seconds") is not None
        and result.get("end_error_seconds") is not None
        and max(
            result["start_error_seconds"],
            result["end_error_seconds"],
        ) <= 0.25
    )


def summarize(style: str, language: str, result: dict) -> dict:
    cues = []
    for item in result["cues"]:
        hyst = item.get("comparisons", {}).get("hysteresis_low_0_62", {})
        cues.append({
            "ocr_geometry_found": bool(item.get("anchor_verified")),
            "fixed_threshold_0_76_strict": strict(item),
            "temporal_hysteresis_strict": strict({
                "anchor_verified": item.get("anchor_verified"),
                **hyst,
            }),
            "hysteresis_start_error": hyst.get("start_error_seconds"),
            "hysteresis_end_error": hyst.get("end_error_seconds"),
            "hysteresis_blank_false_frames": hyst.get("false_on_blank_control"),
            "hysteresis_outside_false_frames": hyst.get("false_non_cue_frames"),
        })
    return {
        "style": style, "language": language,
        "source_is_independent_generated_encoded_mp4": True,
        "cue_count": len(cues), "cues": cues,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--manifest", default="models/manifest.json")
    args = parser.parse_args()
    reports = []
    try:
        with tempfile.TemporaryDirectory(prefix="hse-ink-style-stress-") as td:
            root = Path(td)
            for style in STYLES:
                for language in LANGUAGES:
                    sub = root / style / language
                    sub.mkdir(parents=True, exist_ok=False)
                    output = run_language(
                        sub, language, Path(args.model_dir),
                        Path(args.manifest), style=style,
                    )
                    row = summarize(style, language, output)
                    reports.append(row)
                    print("HSE_INK_STYLE_ROW " + json.dumps(row, sort_keys=True),
                          flush=True)
    except Exception as exc:
        print("HSE_INK_STYLE_STUDY_ERROR " + json.dumps({
            "exception_class": type(exc).__name__,
        }))
        return 3

    strict_counts = {}
    for style in STYLES:
        current = [c for row in reports if row["style"] == style
                   for c in row["cues"]]
        strict_counts[style] = {
            "tested": len(current),
            "ocr_geometry_found": sum(c["ocr_geometry_found"] for c in current),
            "fixed_strict": sum(c["fixed_threshold_0_76_strict"] for c in current),
            "hysteresis_strict": sum(c["temporal_hysteresis_strict"] for c in current),
            "hysteresis_blank_false_frames": sum(
                c["hysteresis_blank_false_frames"] or 0 for c in current
            ),
        }
    summary = {
        "schema": 1,
        "study_completed": len(reports) == len(STYLES) * len(LANGUAGES),
        "source_cue_count": 16,
        "source_is_synthetic_video": True,
        "production_changed": False,
        "real_anime_ground_truth_verified": False,
        "candidate_fully_validated": all(
            item["hysteresis_strict"] == 8
            and item["hysteresis_blank_false_frames"] == 0
            for item in strict_counts.values()
        ),
        "styles": strict_counts,
    }
    print("HSE_INK_STYLE_SUMMARY " + json.dumps(summary, sort_keys=True))
    return 0 if summary["study_completed"] else 4


if __name__ == "__main__":
    raise SystemExit(main())
