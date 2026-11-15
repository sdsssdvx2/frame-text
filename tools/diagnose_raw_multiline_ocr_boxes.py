"""Isolated real-model diagnostic: OCR's raw upper-row boxes vs accepted rows.

Unlike a known-midpoint anchor test, this uses ONLY the production detector's
automatically chosen sample frames. Ground-truth cue times and synthetic text
enter after decoding and recognition, for scoring only. No customer media,
paths, screenshots, transcripts, or model outputs are written or logged.
Production OCR filters and detector are never patched by this experiment.
"""
from __future__ import annotations

import argparse
from contextlib import closing
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace

import numpy as np

from hardsub_extractor.detector import iter_segment_frames
from hardsub_extractor.frame_source import FFmpegPTSFrameSource
from hardsub_extractor.ocr.base import comparison_key
from hardsub_extractor.ocr.rapidocr_adapter import (
    RapidOCRAdapter, _box_record, _script_anchor,
    _join_visual_line, _subtitle_box_lines,
)
from verify_animated_background_ocr_quality import (
    _multiline_reference, encode_video,
)
from verify_multilang_video_pipeline import EXPECTED_SPANS, VIDEO_TEXT

LANGUAGES = ("ko", "en", "ja", "zh")
MAX_MODEL_FRAMES = 150


def _box_row(box, height):
    """Report only the rough normalized detected box location."""
    try:
        arr = np.asarray(box, dtype=float)
        if arr.ndim != 2 or arr.shape[0] < 2 or arr.shape[1] < 2:
            return None
        if not np.all(np.isfinite(arr[:, :2])):
            return None
        center = (float(arr[:, 1].min()) + float(arr[:, 1].max())) / (2.0*height)
    except (TypeError, ValueError, OverflowError, IndexError):
        return None
    if not 0 <= center <= 1:
        return None
    return "upper" if center < 0.30 else "lower"



def raw_upper_filter_geometry(texts, boxes, image_shape, upper_indices, lower_indices, language):
    """Numeric, independent input gates; diagnostic only, not a filter rewrite.

    Compare accepted upper-row text with the raw model result to locate the
    Korean y/x/script gate before re-running the expensive OCR models.
    Report normalized dimensions, never raw text, frame pixels or paths.
    """
    height, width = image_shape[:2]
    if height <= 0 or width <= 0 or not upper_indices or not lower_indices:
        return {}
    upper = [_box_record(i, boxes[i]) for i in upper_indices]
    lower = [_box_record(i, boxes[i]) for i in lower_indices]
    if any(rec is None for rec in upper + lower):
        return {"box_contract_ok": False}
    ucx = (min(r["x0"] for r in upper) + max(r["x1"] for r in upper)) / (2*width)
    lcx = (min(r["x0"] for r in lower) + max(r["x1"] for r in lower)) / (2*width)
    ucy = float(np.median([r["cy"] for r in upper])) / height
    lcy = float(np.median([r["cy"] for r in lower])) / height
    uh = float(np.median([r["height"] for r in upper]))
    lh = float(np.median([r["height"] for r in lower]))
    upper_candidates = [
        0.18 <= rec["cx"]/width <= 0.82
        and rec["cy"]/height >= 0.10
        and _script_anchor(texts[index], require_hangul=(language=="ko"),
                           language=language if language!="ko" else None)
        for index, rec in zip(upper_indices, upper)
    ]
    vertical_gap_norm = (
        min(r["y0"] for r in lower) - max(r["y1"] for r in upper)
    ) / height
    current_allowed_overhang = min(0.08, min(uh, lh)*0.32/height)
    row_pair_possible = (
        0.18 <= ucx <= 0.82
        and 0.18 <= lcx <= 0.82
        and 0.10 <= ucy < 0.30 <= lcy
        and abs(lcx-ucx) <= 0.16
        and 0.07 <= lcy-ucy <= 0.34
        and 0.55 <= uh/max(lh, 1e-6) <= 1.8
        and min(r["y0"] for r in lower)
        >= max(r["y1"] for r in upper) - height * current_allowed_overhang
    )
    return {
        "box_contract_ok": True,
        "upper_center_y": round(ucy, 3),
        "lower_center_y": round(lcy, 3),
        "row_x_delta": round(abs(ucx-lcx), 3),
        "row_y_delta": round(lcy-ucy, 3),
        "row_height_ratio": round(uh/max(lh, 1e-6), 3),
        "row_vertical_gap_norm": round(vertical_gap_norm, 3),
        "bottom_row_center_x": round(lcx, 3),
        "top_row_center_x": round(ucx, 3),
        "row_edge_gap_original_pass": vertical_gap_norm >= -0.035,
        "row_edge_gap_current_pass": vertical_gap_norm >= -current_allowed_overhang,
        "current_allowed_overhang_norm": round(current_allowed_overhang, 3),
        "row_edge_gap_relaxed_010_pass": vertical_gap_norm >= -0.10,
        "row_center_conditions_pass": bool(
            0.18 <= ucx <= 0.82 and 0.18 <= lcx <= 0.82
            and 0.10 <= ucy < 0.30 <= lcy
            and abs(lcx - ucx) <= 0.16
            and 0.07 <= lcy - ucy <= 0.34
            and 0.55 <= uh/max(lh, 1e-6) <= 1.8
        ),
        "upper_boxes_eligible": sum(upper_candidates),
        "upper_boxes_total": len(upper),
        "upper_reject_before_group_y": sum(r["cy"]/height < 0.10 for r in upper),
        "upper_reject_before_group_x": sum(
            not 0.18 <= r["cx"]/width <= 0.82 for r in upper
        ),
        "upper_reject_before_group_script": sum(
            not _script_anchor(texts[index],
                               require_hangul=(language=="ko"),
                               language=language if language!="ko" else None)
            for index in upper_indices
        ),
        "raw_pair_geometry_possible": bool(row_pair_possible),
    }


def analyze_raw_result(result, image_shape, language, expected):
    """Score source-authored text after RAW model results already exist.

    The measured raw/kept pairs make a model miss distinguishable from a
    post-model subtitle box filter rejection. Transcript strings stay in RAM.
    """
    texts = [str(t) for t in (getattr(result, "txts", None) or ())]
    boxes = getattr(result, "boxes", None)
    n = len(texts)
    if boxes is None or len(boxes) != n:
        return {"raw_box_contract_ok": False}
    raw_by_row = {"upper": [], "lower": []}
    for index, box in enumerate(boxes):
        row = _box_row(box, image_shape[0])
        if row:
            raw_by_row[row].append(index)

    def row_key(indices):
        # Horizontal ordering only, within an independently located row.
        def left(idx):
            box = np.asarray(boxes[idx], dtype=float)
            return float(box[:, 0].min())
        return comparison_key("".join(texts[i] for i in sorted(indices, key=left)))

    truth_upper, truth_lower = _multiline_reference(expected)
    ref_keys = [comparison_key(truth_upper), comparison_key(truth_lower)]
    raw_keys = [row_key(raw_by_row[name]) for name in ("upper", "lower")]

    lines = _subtitle_box_lines(
        texts, boxes, image_shape, require_hangul=(language == "ko"),
        language=language if language != "ko" else None,
    )
    kept_text = "\n".join(
        _join_visual_line(texts, boxes, line, language=language)
        for line in lines
    )
    kept_lines = kept_text.splitlines()
    kept_key = comparison_key(kept_text)
    raw_upper = bool(ref_keys[0] and ref_keys[0] in raw_keys[0])
    raw_lower = bool(ref_keys[1] and ref_keys[1] in raw_keys[1])
    geometry = (
        raw_upper_filter_geometry(
            texts, boxes, image_shape, raw_by_row["upper"],
            raw_by_row["lower"], language,
        ) if raw_upper and raw_lower else {}
    )
    kept_upper = bool(ref_keys[0] and ref_keys[0] in kept_key)
    kept_lower = bool(ref_keys[1] and ref_keys[1] in kept_key)
    return {
        "raw_box_contract_ok": True,
        "raw_box_count": n,
        "raw_upper_box_count": len(raw_by_row["upper"]),
        "raw_lower_box_count": len(raw_by_row["lower"]),
        "kept_line_count": len(kept_lines),
        "raw_upper_text_match": raw_upper,
        "raw_lower_text_match": raw_lower,
        "kept_upper_text_match": kept_upper,
        "kept_lower_text_match": kept_lower,
        "both_lines_kept_exact": (
            len(kept_lines) == 2
            and comparison_key(kept_text) == comparison_key(expected)
        ),
        "raw_two_row_filter_geometry": geometry,
    }


def evaluate_samples(samples, language):
    """Time references do not influence which video frames were recognized."""
    results = []
    for start, end in EXPECTED_SPANS:
        source_truth = VIDEO_TEXT[language][len(results)]
        current = [
            sample["diagnosis"] for sample in samples
            if start <= sample["pts"] < end
        ]
        valid = [row for row in current if row.get("raw_box_contract_ok")]
        def count(key):
            return sum(bool(row.get(key)) for row in valid)
        results.append({
            "selected_frames": len(current),
            "valid_raw_geometry_samples": len(valid),
            "raw_upper_box_samples": sum(
                bool(x.get("raw_upper_box_count")) for x in valid
            ),
            "raw_upper_text_exact_samples": count("raw_upper_text_match"),
            "kept_upper_text_exact_samples": count("kept_upper_text_match"),
            "raw_lower_text_exact_samples": count("raw_lower_text_match"),
            "kept_lower_text_exact_samples": count("kept_lower_text_match"),
            "both_lines_kept_exact_samples": count("both_lines_kept_exact"),
            "upper_detected_but_filtered": sum(
                bool(x.get("raw_upper_text_match"))
                and not bool(x.get("kept_upper_text_match"))
                for x in valid
            ),
            "raw_upper_preselection_y_rejections": sum(
                x.get("raw_two_row_filter_geometry", {}).get(
                    "upper_reject_before_group_y", 0
                ) for x in valid
            ),
            "raw_upper_preselection_x_rejections": sum(
                x.get("raw_two_row_filter_geometry", {}).get(
                    "upper_reject_before_group_x", 0
                ) for x in valid
            ),
            "raw_upper_preselection_script_rejections": sum(
                x.get("raw_two_row_filter_geometry", {}).get(
                    "upper_reject_before_group_script", 0
                ) for x in valid
            ),
            "raw_upper_pair_geometry_possible_samples": sum(
                bool(x.get("raw_two_row_filter_geometry", {}).get(
                    "raw_pair_geometry_possible", False
                )) for x in valid
            ),
            "raw_upper_pair_geometry_examples": [
                x["raw_two_row_filter_geometry"]
                for x in valid
                if x.get("raw_upper_text_match")
                and not x.get("kept_upper_text_match")
                and x.get("raw_two_row_filter_geometry")
            ][:2],
            "reference_label_used_only_for_scoring": bool(source_truth),
        })
    return results


def run_language(root, language, models, manifest):
    video = root / (language + ".mp4")
    encode_video(video, language, style="white_two_line")
    engine = RapidOCRAdapter(
        language=language, model_dir=str(models),
        manifest_path=str(manifest),
    )
    samples = []
    with closing(iter(FFmpegPTSFrameSource(str(video), sample_fps=8.0))) as frames:
        with closing(iter_segment_frames(frames)) as segments:
            for segment in segments:
                for sample in segment.samples:
                    if len(samples) >= MAX_MODEL_FRAMES:
                        raise RuntimeError("detector generated too many model frames")
                    raw = engine.engine(
                        sample.image, use_det=True, use_cls=False,
                        use_rec=True, text_score=engine.text_score,
                    )
                    # Reference is looked up only once the model has run.
                    span_index = next(
                        (i for i, (start, end) in enumerate(EXPECTED_SPANS)
                         if start <= sample.pts < end), None,
                    )
                    diagnosis = (
                        analyze_raw_result(
                            raw, sample.image.shape, language,
                            VIDEO_TEXT[language][span_index],
                        ) if span_index is not None
                        else {"outside_reference": True}
                    )
                    samples.append({"pts": sample.pts, "diagnosis": diagnosis})
    return {
        "language": language,
        "model_calls": len(samples),
        "source_encoded_video": True,
        "reference_chosen_samples": False,
        "source_intervals": evaluate_samples(samples, language),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model-dir", required=True)
    p.add_argument("--manifest", default="models/manifest.json")
    p.add_argument("--languages", default="ko,ja",
                   help="comma-delimited real OCR model profiles to inspect")
    opts = p.parse_args()
    selected = tuple(v.strip() for v in opts.languages.split(","))
    if not selected or any(v not in LANGUAGES for v in selected) or len(set(selected)) != len(selected):
        p.error("languages must be distinct supported OCR profiles")
    reports = []
    try:
        with tempfile.TemporaryDirectory(prefix="hse-raw-ocr-lines-") as td:
            for language in selected:
                folder = Path(td)/language
                folder.mkdir()
                try:
                    row = run_language(folder, language,
                                       Path(opts.model_dir), Path(opts.manifest))
                except Exception as exc:
                    row = {"language": language, "error_class": type(exc).__name__,
                           "source_encoded_video": False}
                reports.append(row)
                print("HSE_RAW_MULTILINE_ROW " +
                      json.dumps(row, sort_keys=True), flush=True)
    except Exception as exc:
        print("HSE_RAW_MULTILINE_FATAL " +
              json.dumps({"error_class": type(exc).__name__}))
        return 3
    complete = len(reports) == len(selected) and all(
        r.get("source_encoded_video") and len(r.get("source_intervals", [])) == 2
        for r in reports
    )
    summary = {
        "schema": 1,
        "measurement_complete": complete,
        "tested_languages": list(selected),
        "all_four_languages_tested": len(selected) == 4,
        "raw_model_vs_filter_isolation": True,
        "reference_used_for_sample_selection": False,
        "production_modified": False,
        "real_anime_tested": False,
        "candidate_deployed": False,
        "model_calls": sum(x.get("model_calls", 0) for x in reports),
        "raw_upper_text_exact_samples": sum(
            cue["raw_upper_text_exact_samples"]
            for r in reports for cue in r.get("source_intervals", [])
        ),
        "upper_detected_but_filtered_samples": sum(
            cue["upper_detected_but_filtered"]
            for r in reports for cue in r.get("source_intervals", [])
        ),
    }
    print("HSE_RAW_MULTILINE_SUMMARY " + json.dumps(summary, sort_keys=True))
    return 0 if complete else 4


if __name__ == "__main__":
    raise SystemExit(main())
