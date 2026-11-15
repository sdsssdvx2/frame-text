"""Real publisher-authored open-caption/clean-video A/B for all HSE local ROI presets.

Separate official captioned and editorial clean originals, and an independent
publisher SRT, are used ONLY inside disposable local scratch. Each ROI runs the
same free English PP-OCRv5, 8fps, production detector and SRT/ASS writer. No
manual transcript, captions or model output influence inference. There are
no private samples, GUI installation, default changes, network upload or EXE
release approvals. Results are numeric, NOT generalized anime accuracy.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import tempfile
import time

from compare_external_engine_exports import evaluate
from compare_tab_roi_high_multilang import _extract_once
from verify_commons_natural_hardsub import (
    _get_media, _get_reference, _probe, _cue_error_diagnostics,
)
from hardsub_extractor.tab_capture import CAPTION_REGIONS, roi_for_region


REGIONS = ("bottom40", "bottom25", "lower65", "lower80")


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def run_study(model_dir: Path, manifest: Path) -> dict:
    if not model_dir.is_dir() or not manifest.is_file():
        raise ValueError("verified free model input missing")
    if not all(region in CAPTION_REGIONS for region in REGIONS):
        raise ValueError("study ROI not in product selection")
    if len(set(REGIONS)) != 4 or REGIONS[0] != "bottom40":
        raise ValueError("production baseline not first")
    for region in REGIONS:
        geometry = roi_for_region(region)
        if not (
            abs(geometry.x - .04) < 1e-9 and
            abs(geometry.w - .92) < 1e-9 and
            abs((geometry.y + geometry.h) - .98) < 1e-9
        ):
            raise ValueError("ROI geometry not source-aligned")
    reference, reference_sha = _get_reference()
    with tempfile.TemporaryDirectory(prefix="hse-publisher-roi-natural-") as td:
        root = Path(td)
        media = {
            name: root / (name + ".webm")
            for name in ("caption", "clean")
        }
        pins = {
            name: _get_media(media[name], name)
            for name in ("caption", "clean")
        }
        details = {name: _probe(media[name]) for name in media}
        if abs(details["caption"]["seconds"] - details["clean"]["seconds"]) > .1:
            raise RuntimeError("publisher positive and control timebase diverged")
        rows: list[dict] = []
        for region in REGIONS:
            region_folder = root / ("experiment-" + region)
            region_folder.mkdir()
            start = time.monotonic()
            caption, positive_stats = _extract_once(
                media["caption"], region_folder / "positive",
                "en", region, model_dir, manifest,
            )
            clean, negative_stats = _extract_once(
                media["clean"], region_folder / "control",
                "en", region, model_dir, manifest,
            )
            if any(_sha(media[name]) != pin for name, pin in pins.items()):
                raise RuntimeError("published original modified by OCR")
            score = evaluate(reference, caption, clean)
            row = {
                "roi": region,
                "roi_y": roi_for_region(region).y,
                "roi_h": roi_for_region(region).h,
                "publisher_exact_cue_reference_count": len(reference),
                "positive": positive_stats,
                "caption_free_editorial_control": negative_stats,
                "quality": score,
                "cue_error_diagnostics": _cue_error_diagnostics(
                    reference, caption, clean
                ),
                "elapsed_seconds": round(time.monotonic() - start, 2),
                "reference_not_used_in_ocr": True,
                "same_model_same_media": True,
                "SRT_ASS_exact_rendered_text_and_timeline_agreement": (
                    positive_stats["srt_ass_equivalent"] and
                    negative_stats["srt_ass_equivalent"]
                ),
            }
            if not row["SRT_ASS_exact_rendered_text_and_timeline_agreement"]:
                raise RuntimeError("SRT/ASS publication not identical")
            rows.append(row)
            # Only numeric metrics; no raw subtitles, frame pixels or paths.
            print("HSE_PUBLISHER_REAL_ROI_CASE " + json.dumps(
                row, sort_keys=True, ensure_ascii=True,
            ), flush=True)

        controls = rows[0]["quality"]
        comparisons = {}
        for row in rows:
            metric = row["quality"]
            comparisons[row["roi"]] = {
                "exact_text_and_wrap_vs_baseline": (
                    metric["exact_visible_text_and_wrap"] -
                    controls["exact_visible_text_and_wrap"]
                ),
                "strict_both_edges_250ms_vs_baseline": (
                    metric["strict_both_edges_250ms_and_text_wrap"] -
                    controls["strict_both_edges_250ms_and_text_wrap"]
                ),
                "clean_video_false_cues_vs_baseline": (
                    metric["blank_video_false_cues"] -
                    controls["blank_video_false_cues"]
                ),
                "missed_publisher_cues_vs_baseline": (
                    metric["missed_reference_cues"] -
                    controls["missed_reference_cues"]
                ),
                "maximally_conservative_one_source_nonregression": (
                    metric["exact_visible_text_and_wrap"] >=
                    controls["exact_visible_text_and_wrap"] and
                    metric["strict_both_edges_250ms_and_text_wrap"] >=
                    controls["strict_both_edges_250ms_and_text_wrap"] and
                    metric["blank_video_false_cues"] <=
                    controls["blank_video_false_cues"] and
                    metric["missed_reference_cues"] <=
                    controls["missed_reference_cues"]
                ),
                "not_a_model_or_default_promotion": True,
            }
        return {
            "schema": 1,
            "real_publisher_burned_in_subtitles": True,
            "matching_caption_free_editorial_control": True,
            "human_pixel_corrected_ground_truth": False,
            "external_publisher_TimedText_fixed_revision": 1117709522,
            "reference_sha256": reference_sha,
            "publisher_video_sha256": pins,
            "original_media_unchanged": True,
            "experiment": "manual_local_OCR_ROI_choices_not_autoselect",
            "regions": list(REGIONS),
            "real_English_PP_OCRv5_not_CJK_anime": True,
            "no_private_media_or_subtitle_text_logged": True,
            "comparison": comparisons,
            "rows": rows,
            "default_bottom40_changed": False,
            "model_calls_billable": False,
            "release_gate_passed": False,
        }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path,
                        default=Path("models/manifest.json"))
    parser.add_argument("--accept-public-download", action="store_true")
    args = parser.parse_args()
    if not args.accept_public_download:
        print("HSE_PUBLISHER_REAL_ROI=EXPLICIT_PUBLIC_SOURCE_APPROVAL_REQUIRED")
        return 2
    try:
        output = run_study(args.model_dir, args.manifest)
    except Exception as exc:
        print("HSE_PUBLISHER_REAL_ROI_ERROR " + json.dumps({
            "error_class": type(exc).__name__,
            "model_quality_measured": False,
            "not_released": True,
        }, sort_keys=True), flush=True)
        return 3
    print("HSE_PUBLISHER_REAL_ROI_RESULT " + json.dumps(
        output, sort_keys=True, ensure_ascii=True,
    ), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
