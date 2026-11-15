"""Paired moving-background OCR retry policy study, NOT a production change.

Compares a pinned 0.90 box-confidence retry threshold to an exploratory 0.80
threshold on exactly the same generated English and Chinese encoded animated
video/text fixtures. The study exits success only when the experiment itself
completes, never from a candidate quality claim. Production keeps 0.90 until
independent evidence demonstrates no accuracy regression.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import tempfile

from verify_animated_background_ocr_quality import check_language


def _metrics(row: dict) -> dict:
    cues = row.get("reference_comparison", [])
    present = (
        row.get("cues_detected") == 2
        and row.get("matched_srt_ass") is True
        and len(cues) == 2
        and all(cue.get("present", False) for cue in cues)
    )
    return {
        "valid_structure": bool(present),
        "exact": sum(bool(cue.get("exact_text", False)) for cue in cues),
        "max_cer": max((cue.get("cer", 1.0) for cue in cues), default=1.0),
        "max_clock_error": max((
            max(cue.get("start_error", 99), cue.get("end_error", 99))
            for cue in cues
        ), default=99),
        "calls": row.get("ocr_calls", -1),
        "retry_calls": row.get("ocr_retry_calls", -1),
        "seconds": row.get("elapsed_seconds", -1.0),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--manifest", default="models/manifest.json")
    args = ap.parse_args()
    findings = []
    with tempfile.TemporaryDirectory(prefix="hse-ocr-retry-comparison-") as td:
        root = Path(td)
        for lang in ("en", "zh"):
            metrics = {}
            for label, value in (("baseline", 0.90), ("candidate", 0.80)):
                try:
                    working = root / label
                    working.mkdir(parents=True, exist_ok=True)
                    row = check_language(
                        working, lang,
                        Path(args.model_dir), Path(args.manifest),
                        retry_box_threshold=value,
                    )
                    metrics[label] = _metrics(row)
                except Exception as exc:
                    metrics[label] = {
                        "valid_structure": False,
                        "error_class": type(exc).__name__,
                    }
            a, b = metrics["baseline"], metrics["candidate"]
            complete = a["valid_structure"] and b["valid_structure"]
            no_accuracy_regression = (
                complete
                and b["exact"] >= a["exact"]
                and b["max_cer"] <= a["max_cer"]
                and b["max_clock_error"] <= a["max_clock_error"]
            )
            faster = (
                complete and b["calls"] <= int(a["calls"] * 0.90)
            )
            report = {
                "language": lang,
                "baseline": a,
                "candidate": b,
                "candidate_fewer_calls": bool(faster),
                "candidate_no_accuracy_regression": bool(no_accuracy_regression),
                "candidate_safe_to_consider": bool(faster and no_accuracy_regression),
            }
            findings.append(report)
            print("HSE_ANIMATED_RETRY_PAIR " + json.dumps(
                report, sort_keys=True, ensure_ascii=True,
            ), flush=True)
    tested = len(findings) == 2
    all_baselines_pass = tested and all(
        f["baseline"].get("valid_structure")
        and f["baseline"].get("max_cer", 1.0) <= 0.15
        and f["baseline"].get("max_clock_error", 2.0) <= 0.65
        for f in findings
    )
    summary = {
        "scope": "moving_synthetic_video_real_ocr_only",
        "test_completed": bool(tested and all_baselines_pass),
        "production_config_changed": False,
        "real_anime_accuracy_verified": False,
        "candidate_adoption_recommended": bool(
            tested and all(f["candidate_safe_to_consider"] for f in findings)
        ),
        "studied_languages": ["en", "zh"],
        "results": findings,
    }
    print("HSE_ANIMATED_RETRY_SUMMARY " + json.dumps(
        summary, sort_keys=True, ensure_ascii=True,
    ))
    return 0 if summary["test_completed"] else 4


if __name__ == "__main__":
    raise SystemExit(main())
