"""Same-MP4 PP-OCRv5 timing study with optional quiet segment caps.

The default production detector is UNCHANGED; all test caps are explicit.
Known reference intervals are used only for scoring, not selecting frames,
forced cap boundaries or source OCR inputs. Emit no recognized transcripts,
screenshots, paths or customer media; all experiment files are ephemeral.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import tempfile

from hardsub_extractor.detector import DetectorConfig
from verify_animated_background_ocr_quality import check_language, encode_video
from verify_multiline_streaming_ocr_quality import score_case

LANGUAGES = ("ja", "zh")
CAPS = (None, 1.0, 1.25)


def parse_caps(raw: str) -> tuple[float | None, ...]:
    """Exact bounded non-overlapping settings for an opt-in research pass."""
    tokens = [part.strip().lower() for part in raw.split(",")]
    if not 1 <= len(tokens) <= 5 or any(not token for token in tokens):
        raise ValueError("provide 1..5 nonempty cap values")
    caps = []
    for token in tokens:
        if token == "none":
            cap = None
        else:
            try:
                cap = float(token)
            except (ValueError, OverflowError):
                raise ValueError("unrecognized quiet cap") from None
            if not math.isfinite(cap) or not 0.50 <= cap <= 30.0:
                raise ValueError("cap must be finite and in 0.50..30.0 seconds")
        if cap in caps:
            raise ValueError("duplicate cap is not an independent trial")
        caps.append(cap)
    return tuple(caps)


def check_blank_control(root, language, video, model_dir, manifest, cap, pending_timeout=None):
    """Real pipeline, real OCR, same detector; no-output RuntimeError is expected.

    Every other error is a failed study, not evidence of a clean blank.
    """
    out_base = root / (language + ".subtitles")
    try:
        result = check_language(
            root, language, model_dir, manifest, style="white_two_line",
            source_override=video,
            detector_config=DetectorConfig(
                max_quiet_segment_seconds=cap,
                max_pending_change_seconds=pending_timeout,
            ),
        )
    except RuntimeError as exc:
        if not str(exc).startswith("인식된 자막이 없습니다."):
            raise
        if out_base.with_suffix(".srt").exists() or out_base.with_suffix(".ass").exists():
            raise RuntimeError("unexpected partial output in empty-caption control")
        return {"executed": True, "false_cues": 0}
    return {"executed": True, "false_cues": result["cues_detected"]}



def summarize_trial(source, language, cap):
    scored = score_case(source, language)
    shape_ok = (
        scored["reported_cues"] == scored["expected_cues"]
        and scored["srt_ass_consistent"]
    )
    checks = scored["checks"]
    strict_tight = sum(
        bool(item["strict_joint_acceptable"])
        and item.get("start_error_seconds") is not None
        and item.get("end_error_seconds") is not None
        and 0 <= item["start_error_seconds"] <= 0.25
        and 0 <= item["end_error_seconds"] <= 0.25
        for item in checks
    )
    return {
        "language": language,
        "quiet_cap_seconds": cap,
        "trial_executed": True,
        "valid_export_shape": shape_ok,
        "cue_count": scored["reported_cues"],
        "srt_ass_consistent": scored["srt_ass_consistent"],
        "joint_exact_cues": (
            sum(x["joint_acceptable"] for x in checks) if shape_ok else 0
        ),
        "aligned_reference_joint_matches": sum(
            x["joint_acceptable"] for x in checks
        ),
        "strict_visible_joint_cues": (
            sum(x["strict_joint_acceptable"] for x in checks)
            if shape_ok else 0
        ),
        "strict_visible_tight_250ms_cues": (
            strict_tight if shape_ok else 0
        ),
        "all_cues_strict_visible": scored["all_cues_strict_visible"],
        "strict_visible_tight_250ms_complete": (
            shape_ok and strict_tight == len(checks)
            and scored["all_cues_strict_visible"]
        ),
        "reference_cue_edges": [
            {
                "observed_start_seconds": x.get("observed_start_seconds"),
                "observed_end_seconds": x.get("observed_end_seconds"),
                "start_error_seconds": x.get("start_error_seconds"),
                "end_error_seconds": x.get("end_error_seconds"),
                "source_characters_match": x["exact_reference_characters"],
                "two_lines_present": x["observed_two_lines"],
                "joint_acceptable": x["joint_acceptable"],
                "strict_visible_joint": x["strict_joint_acceptable"],
                "strict_physical_line_wrap": x["exact_physical_line_wrap"],
            }
            for x in checks
        ],
        "all_cues_complete": scored["all_cues_complete"],
        "ocr_calls": scored["ocr_calls"],
        "retry_calls": scored["retry_calls"],
        "elapsed_seconds": source["elapsed_seconds"],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--manifest", default="models/manifest.json")
    ap.add_argument("--caps", default="none,1.0,1.25")
    ap.add_argument("--blank-controls", action="store_true")
    args = ap.parse_args()
    try:
        caps = parse_caps(args.caps)
    except ValueError as exc:
        ap.error(str(exc))
    trials = []
    controls = []
    try:
        with tempfile.TemporaryDirectory(prefix="hse-quiet-cap-ab-") as td:
            for lang in LANGUAGES:
                source_root = Path(td) / lang
                source_root.mkdir()
                video = source_root / "fixed-source.mp4"
                encode_video(video, lang, style="white_two_line")
                file_sha = hashlib.sha256(video.read_bytes()).hexdigest()
                blank = source_root / "fixed-no-caption.mp4"
                blank_sha = None
                if args.blank_controls:
                    encode_video(blank, lang, style="white_two_line",
                                 include_subtitles=False)
                    blank_sha = hashlib.sha256(blank.read_bytes()).hexdigest()
                for cap in caps:
                    trial_root = source_root / (
                        "production-default" if cap is None
                        else "quiet-cap-" + str(cap)
                    )
                    trial_root.mkdir()
                    try:
                        data = check_language(
                            trial_root, lang, Path(args.model_dir),
                            Path(args.manifest), style="white_two_line",
                            source_override=video,
                            detector_config=DetectorConfig(
                                max_quiet_segment_seconds=cap,
                            ),
                        )
                        row = summarize_trial(data, lang, cap)
                    except Exception as exc:
                        row = {
                            "language": lang, "quiet_cap_seconds": cap,
                            "trial_executed": False,
                            "error_class": type(exc).__name__,
                        }
                    if hashlib.sha256(video.read_bytes()).hexdigest() != file_sha:
                        raise RuntimeError("identical encoded source modified during trial")
                    trials.append(row)
                    print("HSE_QUIET_CAP_CASE " +
                          json.dumps(row, sort_keys=True), flush=True)
                    if args.blank_controls:
                        control_root = trial_root / "no-caption"
                        control_root.mkdir()
                        try:
                            control = check_blank_control(
                                control_root, lang, blank,
                                Path(args.model_dir), Path(args.manifest), cap,
                            )
                            control.update({"language": lang, "quiet_cap_seconds": cap})
                        except Exception as exc:
                            control = {
                                "language": lang, "quiet_cap_seconds": cap,
                                "executed": False,
                                "error_class": type(exc).__name__,
                            }
                        if hashlib.sha256(blank.read_bytes()).hexdigest() != blank_sha:
                            raise RuntimeError("blank video bytes changed during trial")
                        controls.append(control)
                        print("HSE_QUIET_CAP_BLANK " +
                              json.dumps(control, sort_keys=True), flush=True)
    except Exception as exc:
        print("HSE_QUIET_CAP_FATAL " +
              json.dumps({"error_class": type(exc).__name__}))
        return 3
    complete = (
        len(trials) == len(LANGUAGES)*len(caps)
        and all(t.get("trial_executed") for t in trials)
        and (
            not args.blank_controls or (
                len(controls) == len(trials)
                and all(c.get("executed") for c in controls)
            )
        )
    )
    aggregate = {
        str(cap): {
            "joint_exact_cues": sum(
                row.get("joint_exact_cues", 0) for row in trials
                if row["quiet_cap_seconds"] == cap
            ),
            "strict_visible_joint_cues": sum(
                row.get("strict_visible_joint_cues", 0) for row in trials
                if row["quiet_cap_seconds"] == cap
            ),
            "strict_visible_tight_250ms_cues": sum(
                row.get("strict_visible_tight_250ms_cues", 0)
                for row in trials if row["quiet_cap_seconds"] == cap
            ),
            "ocr_calls": sum(
                row.get("ocr_calls", 0) for row in trials
                if row["quiet_cap_seconds"] == cap
            ),
            "fully_complete_languages": sum(
                bool(row.get("all_cues_complete")) for row in trials
                if row["quiet_cap_seconds"] == cap
            ),
            "blank_false_cues": (
                sum(int(row["false_cues"]) for row in controls
                    if row["quiet_cap_seconds"] == cap and row.get("executed"))
                if args.blank_controls else None
            ),
        } for cap in caps
    }
    report = {
        "schema": 1,
        "experiment_completed": complete,
        "original_production_default_preserved": True,
        "reference_oracle_used_for_segmenting": False,
        "same_mp4_sha_for_all_caps_per_language": True,
        "real_anime_tested": False,
        "no_caption_control_completed": (
            args.blank_controls and len(controls) == len(trials)
            and all(c.get("executed") for c in controls)
        ),
        "no_caption_control_zero_false_cues": (
            args.blank_controls and len(controls) == len(trials)
            and all(c.get("executed") and c.get("false_cues") == 0
                    for c in controls)
        ),
        "tested_caps": list(caps),
        "candidate_deployed": False,
        "results": aggregate,
    }
    print("HSE_QUIET_CAP_SUMMARY " + json.dumps(report, sort_keys=True))
    return 0 if complete else 4


if __name__ == "__main__":
    raise SystemExit(main())
