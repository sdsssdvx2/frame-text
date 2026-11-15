"""Opt-in real PP-OCRv5 A/B for CJK raised white subtitles over moving video.

Exactly identical encoded caption/control MP4 inputs per JA/ZH language.
Baseline lower65 and research-only persistent-white signature are compared
through the existing streaming detector, native OCR, SRT/ASS and false-cue
evaluator. No production defaults, accounts, user media or paid APIs change.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import tempfile

from compare_tab_roi_high_multilang import (
    LANGUAGES, STYLE, _extract_once, _hash, _reference,
)
from compare_external_engine_exports import evaluate
from hardsub_extractor.detector import DetectorConfig
from verify_animated_background_ocr_quality import encode_video


def extract_profile(video, dest, language, model_dir, manifest, *, persistent):
    # Exercise the same opt-in DetectorConfig path as browser offline_ocr;
    # monkey-patching the global signature would bypass the real wiring.
    cfg = (DetectorConfig(signature_mode="persistent_white_3")
           if persistent else None)
    return _extract_once(
        video, dest, language, "lower65", model_dir, manifest,
        detector_config=cfg,
    )


def study(model_dir: Path, manifest: Path) -> dict:
    rows = []
    comparisons = {}
    for language in LANGUAGES:
        with tempfile.TemporaryDirectory(prefix="hse-white-cjk-" + language + "-") as td:
            root = Path(td)
            caption = root / "raised-captions.mp4"
            clean = root / "matched-no-caption.mp4"
            encode_video(caption, language, style=STYLE)
            encode_video(clean, language, style=STYLE,
                         include_subtitles=False)
            pins = (_hash(caption), _hash(clean))
            if pins[0] == pins[1]:
                raise RuntimeError("positive_negative_source_identical")
            cases = {}
            # This pair is essential: previous independent experiments did
            # not provide a same-input A/B for the new persistence feature.
            for profile in ("current_lower65", "persistent_white_3"):
                folder = root / profile
                folder.mkdir()
                is_persistent = profile == "persistent_white_3"
                positive, pos_info = extract_profile(
                    caption, folder / "caption", language,
                    model_dir, manifest, persistent=is_persistent,
                )
                negative, neg_info = extract_profile(
                    clean, folder / "blank", language,
                    model_dir, manifest, persistent=is_persistent,
                )
                if pins != (_hash(caption), _hash(clean)):
                    raise RuntimeError("source_video_changed_during_ab")
                quality = evaluate(_reference(language), positive, negative)
                case = {
                    "language": language, "profile": profile,
                    "real_ocr": True, "same_media_pinned": True,
                    "caption": pos_info, "blank": neg_info,
                    "quality": quality,
                    "source": "synthetic_animated_mp4",
                }
                cases[profile] = case
                rows.append(case)
                print("HSE_PERSISTENT_WHITE_OCR_CASE " +
                      json.dumps(case, sort_keys=True), flush=True)
            base = cases["current_lower65"]
            candidate = cases["persistent_white_3"]
            q0, q1 = base["quality"], candidate["quality"]
            comparisons[language] = {
                "extra_exact_visible_cues": (
                    q1["exact_visible_text_ignoring_wrap"] -
                    q0["exact_visible_text_ignoring_wrap"]
                ),
                "recovered_reference_cues": (
                    q0["missed_reference_cues"] -
                    q1["missed_reference_cues"]
                ),
                "character_error_rate_reduction": round(
                    q0["aggregate_character_error_rate"] -
                    q1["aggregate_character_error_rate"], 6,
                ),
                "absolute_edge_error_reduction_seconds": (
                    round(q0["mean_absolute_edge_error_seconds"] -
                          q1["mean_absolute_edge_error_seconds"], 4)
                    if q0["mean_absolute_edge_error_seconds"] is not None
                    and q1["mean_absolute_edge_error_seconds"] is not None
                    else None
                ),
                "extra_blank_false_cues": (
                    q1["blank_video_false_cues"] -
                    q0["blank_video_false_cues"]
                ),
                "extra_ocr_calls_caption": (
                    candidate["caption"]["ocr_calls"] -
                    base["caption"]["ocr_calls"]
                ),
                "extra_ocr_calls_blank": (
                    candidate["blank"]["ocr_calls"] -
                    base["blank"]["ocr_calls"]
                ),
                "better_without_new_blank_false_cues": (
                    q1["exact_visible_text_ignoring_wrap"] >
                    q0["exact_visible_text_ignoring_wrap"]
                    and q1["blank_video_false_cues"] <=
                    q0["blank_video_false_cues"]
                ),
                "recovered_without_new_blank_false_cues": (
                    q1["missed_reference_cues"] <
                    q0["missed_reference_cues"]
                    and q1["aggregate_character_error_rate"] <
                    q0["aggregate_character_error_rate"]
                    and q1["blank_video_false_cues"] <=
                    q0["blank_video_false_cues"]
                ),
                "no_auto_promotion": True,
            }
    return {
        "schema": 1,
        "study": "real_CJK_OCR_persistent_white_stable_pixels",
        "case_count": len(rows),
        "complete": len(rows) == len(LANGUAGES) * 2,
        "real_anime": False, "live_tab": False,
        "paid_api_calls": 0, "production_changed": False,
        "language_results": comparisons, "rows": rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", required=True, type=Path)
    parser.add_argument("--manifest", default=Path("models/manifest.json"),
                        type=Path)
    args = parser.parse_args()
    try:
        result = study(args.model_dir, args.manifest)
    except Exception as exc:
        print("HSE_PERSISTENT_WHITE_OCR_ERROR " +
              json.dumps({"error_class": type(exc).__name__}), flush=True)
        return 3
    print("HSE_PERSISTENT_WHITE_OCR_RESULT " +
          json.dumps(result, sort_keys=True), flush=True)
    return 0 if result["complete"] else 4


if __name__ == "__main__":
    raise SystemExit(main())
