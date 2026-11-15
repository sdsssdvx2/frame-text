"""Real PP-OCRv5 matched-video pending-change timeout A/B (research only).

Compare unchanged production defaults with two bounded opt-in candidate
timeouts on the exact same generated moving JA/ZH MP4 source and matching
no-caption controls. Synthetic references enter *only* post-OCR grading.
No source frames, transcripts, credentials or paths are emitted to logs.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import tempfile

from hardsub_extractor.detector import DetectorConfig
from compare_quiet_segment_caps import check_blank_control, summarize_trial
from verify_animated_background_ocr_quality import check_language, encode_video

LANGUAGES = ("ja", "zh")
PROFILES = (
    ("default", None, None),
    ("pending_375ms", 0.625, 0.375),
    ("pending_500ms", 0.625, 0.500),
)
MAX_VIDEO_BYTES = 32 * 1024 * 1024


def assess_trial(result, language, name, cap, timeout):
    row = summarize_trial(result, language, cap)
    row["profile"] = name
    row["pending_timeout_seconds"] = timeout
    row["tested_identical_original_video"] = True
    return row


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--manifest", default="models/manifest.json")
    args = ap.parse_args()
    results = []
    controls = []
    try:
        with tempfile.TemporaryDirectory(prefix="hse-pending-timeout-ab-") as td:
            for language in LANGUAGES:
                root = Path(td) / language
                root.mkdir()
                sample = root / "caption.mp4"
                empty = root / "no-caption.mp4"
                encode_video(sample, language, style="white_two_line")
                encode_video(
                    empty, language, include_subtitles=False,
                    style="white_two_line",
                )
                if max(sample.stat().st_size, empty.stat().st_size) > MAX_VIDEO_BYTES:
                    raise RuntimeError("generated MP4 exceeds hard test bound")
                hashes = {
                    item: hashlib.sha256(item.read_bytes()).hexdigest()
                    for item in (sample, empty)
                }
                for name, cap, timeout in PROFILES:
                    out = root / name
                    out.mkdir()
                    try:
                        measured = check_language(
                            out, language, Path(args.model_dir),
                            Path(args.manifest), style="white_two_line",
                            source_override=sample,
                            detector_config=DetectorConfig(
                                max_quiet_segment_seconds=cap,
                                max_pending_change_seconds=timeout,
                            ),
                        )
                        row = assess_trial(
                            measured, language, name, cap, timeout,
                        )
                    except Exception as exc:
                        row = {
                            "language": language, "profile": name,
                            "trial_executed": False,
                            "error_class": type(exc).__name__,
                        }
                    results.append(row)
                    print("HSE_PENDING_AB_CASE " +
                          json.dumps(row, sort_keys=True), flush=True)

                    negroot = out / "no-caption"
                    negroot.mkdir()
                    try:
                        neg = check_blank_control(
                            negroot, language, empty, Path(args.model_dir),
                            Path(args.manifest), cap, timeout,
                        )
                        neg.update({"language": language, "profile": name})
                    except Exception as exc:
                        neg = {
                            "language": language, "profile": name,
                            "executed": False,
                            "error_class": type(exc).__name__,
                        }
                    controls.append(neg)
                    print("HSE_PENDING_AB_BLANK " +
                          json.dumps(neg, sort_keys=True), flush=True)
                    if any(
                        hashlib.sha256(path.read_bytes()).hexdigest() != digest
                        for path, digest in hashes.items()
                    ):
                        raise RuntimeError("identical MP4 input bytes changed")
    except Exception as exc:
        print("HSE_PENDING_AB_FATAL " +
              json.dumps({"error_class": type(exc).__name__}))
        return 3

    all_executed = (
        len(results) == len(controls) == len(LANGUAGES)*len(PROFILES)
        and all(r.get("trial_executed") for r in results)
        and all(c.get("executed") for c in controls)
    )
    clean = all_executed and all(c.get("false_cues") == 0 for c in controls)
    aggregate = {}
    for name, cap, timeout in PROFILES:
        subset = [r for r in results if r.get("profile") == name]
        blank = [c for c in controls if c.get("profile") == name]
        aggregate[name] = {
            "cap_seconds": cap,
            "pending_timeout_seconds": timeout,
            "languages": len(subset),
            "historical_normalized_cues": sum(
                r.get("joint_exact_cues", 0) for r in subset
            ),
            "strict_visible_cues": sum(
                r.get("strict_visible_joint_cues", 0) for r in subset
            ),
            "strict_250ms_cues": sum(
                r.get("strict_visible_tight_250ms_cues", 0) for r in subset
            ),
            "ocr_calls": sum(r.get("ocr_calls", 0) for r in subset),
            "no_caption_false_cues": (
                sum(c["false_cues"] for c in blank)
                if all(c.get("executed") for c in blank) else None
            ),
        }
    summary = {
        "schema": 1,
        "study_completed": all_executed,
        "blank_controls_clean": clean,
        "same_mp4_bytes_all_profiles": True,
        "ground_truth_used_to_choose_boundaries": False,
        "inert_without_explicit_config": True,
        "real_anime_tested": False,
        "released_build_modified": False,
        "release_gate_passed": False,
        "aggregate": aggregate,
    }
    print("HSE_PENDING_AB_SUMMARY " +
          json.dumps(summary, sort_keys=True), flush=True)
    return 0 if all_executed else 4


if __name__ == "__main__":
    raise SystemExit(main())
