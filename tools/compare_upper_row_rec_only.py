"""Research-only, no-reference-selected OCR retry on independently located upper text.

Use the ORIGINAL production detector's automatically selected frames.
The source-authored transcript and cue times enter ONLY the evaluator.
This is not a production fallback, licensed-episode benchmark or accuracy
claim. All generated MP4s and model scratch files are temporary; the
console emits no recognized words, boxes with text, screenshots or paths.
"""
from __future__ import annotations

import argparse
from contextlib import closing
import json
import math
from pathlib import Path
import tempfile

import cv2
import numpy as np

from hardsub_extractor.detector import iter_segment_frames
from hardsub_extractor.frame_source import FFmpegPTSFrameSource
from hardsub_extractor.ocr.base import comparison_key, normalize_text
from hardsub_extractor.ocr.rapidocr_adapter import RapidOCRAdapter, _box_record, _script_anchor
from hardsub_extractor.ocr.adaptive import _clahe_bgr, _upscale
from verify_animated_background_ocr_quality import (
    _multiline_reference, encode_video,
)
from verify_multilang_video_pipeline import EXPECTED_SPANS, VIDEO_TEXT

LANGUAGES = ("ja", "zh")
VARIANTS = ("raw_detected_upper", "recognize_crop", "recognize_upscaled",
            "recognize_clahe_upscaled", "unsupervised_highest_confidence")
MAX_SAMPLES = 96
MAX_CROP_FRACTION = 0.50


def _safe_confidence(value):
    try:
        score = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return score if math.isfinite(score) and 0 <= score <= 1 else None


def _upper_boxes(result, shape):
    """Select upper-row boxes from OCR geometry only; never from ground truth."""
    height, width = shape[:2]
    texts = list(getattr(result, "txts", None) or ())
    scores = list(getattr(result, "scores", None) or ())
    boxes = getattr(result, "boxes", None)
    if (boxes is None or len(boxes) != len(texts)
            or len(scores) != len(texts)):
        return [], texts, scores
    recs = []
    for idx, box in enumerate(boxes):
        rec = _box_record(idx, box)
        if rec is None:
            continue
        cy = rec["cy"]/height
        cx = rec["cx"]/width
        if (0.08 <= cy < 0.30 and 0.12 <= cx <= 0.88
                and 9 <= rec["height"] <= height * 0.36
                and rec["width"] >= 8 and texts[idx]):
            recs.append(rec)
    if not recs:
        return [], texts, scores
    recs.sort(key=lambda x: (x["cy"], x["x0"]))
    # A raw text result may include a genuine two-line caption plus
    # independent scene lettering. Never synthesize an extra line:
    # collect only boxes spatially consistent with the first upper row.
    first = recs[0]
    group = [r for r in recs
             if abs(r["cy"]-first["cy"]) <= max(8, first["height"]*0.45)]
    return sorted(group, key=lambda x: x["x0"]), texts, scores


def _has_paired_lower_anchor(raw, upper_records, shape, language):
    """Reject persistent scene headings before a costly upper-row OCR retry.

    The candidate upper text and its neighboring lower subtitle are located
    using only existing raw OCR boxes, language script and geometry. There
    is no source transcript, truth timestamp or reference-dependent crop.
    This study gate never modifies the production subtitle filter.
    """
    if language not in LANGUAGES or not upper_records:
        return False
    height, width = shape[:2]
    texts = list(getattr(raw, "txts", None) or ())
    boxes = getattr(raw, "boxes", None)
    if height <= 0 or width <= 0 or boxes is None or len(boxes) != len(texts):
        return False

    upper_cx = (
        min(rec["x0"] for rec in upper_records)
        + max(rec["x1"] for rec in upper_records)
    ) / (2 * width)
    upper_cy = float(np.median([rec["cy"] for rec in upper_records])) / height
    upper_h = float(np.median([rec["height"] for rec in upper_records]))
    upper_bottom = max(rec["y1"] for rec in upper_records)
    if not (0.12 <= upper_cx <= 0.88 and 0.08 <= upper_cy < 0.30):
        return False

    for index, text in enumerate(texts):
        rec = _box_record(index, boxes[index])
        if rec is None:
            continue
        lower_cx = rec["cx"] / width
        lower_cy = rec["cy"] / height
        if not (
            0.18 <= lower_cx <= 0.82
            and lower_cy >= 0.30
            and abs(lower_cx - upper_cx) <= 0.16
            and 0.07 <= lower_cy - upper_cy <= 0.34
            and 0.55 <= upper_h / rec["height"] <= 1.80
            and rec["y0"] >= upper_bottom - min(
                height * 0.08, min(upper_h, rec["height"]) * 0.32
            )
        ):
            continue
        if _script_anchor(
            normalize_text(str(text)), require_hangul=False,
            language=language,
        ):
            return True
    return False


def _single_recognition(engine, crop):
    if crop.shape[0] < 18 or crop.shape[1] < 18:
        return ("", None)
    try:
        result = engine(crop, use_det=False, use_cls=False,
                        use_rec=True, text_score=0.45)
        texts = list(getattr(result, "txts", None) or ())
        scores = list(getattr(result, "scores", None) or ())
    except Exception:
        return ("", None)
    if len(texts) != 1 or len(scores) != 1:
        return ("", None)
    confidence = _safe_confidence(scores[0])
    if confidence is None:
        return ("", None)
    return (str(texts[0]), confidence)


def variants_for_raw_frame(
    raw, image, engine, *, language="ja", independent_recognizer=None,
):
    """Return actual observed strings only; no generative or reference repair."""
    records, texts, scores = _upper_boxes(raw, image.shape)
    if not records or not _has_paired_lower_anchor(
        raw, records, image.shape, language,
    ):
        return {}
    # An upper OCR row with many widely distributed fragments is not
    # automatically one word: each detection remains an observed raw read.
    raw_text = "".join(texts[r["index"]] for r in records)
    raw_scores = [_safe_confidence(scores[r["index"]]) for r in records]
    raw_score = min((v for v in raw_scores if v is not None), default=None)
    if raw_score is None:
        return {}
    height, width = image.shape[:2]
    x0 = max(0, int(min(r["x0"] for r in records))-14)
    x1 = min(width, int(max(r["x1"] for r in records))+15)
    y0 = max(0, int(min(r["y0"] for r in records))-10)
    y1 = min(height, int(max(r["y1"] for r in records))+11)
    if (x1-x0 < 18 or y1-y0 < 18
            or (y1-y0) > height * MAX_CROP_FRACTION):
        return {"raw_detected_upper": (raw_text, raw_score)}
    crop = image[y0:y1, x0:x1]
    findings = {"raw_detected_upper": (raw_text, raw_score)}
    findings["recognize_crop"] = _single_recognition(engine, crop)
    findings["recognize_upscaled"] = _single_recognition(
        engine, _upscale(crop, 1.75),
    )
    findings["recognize_clahe_upscaled"] = _single_recognition(
        engine, _upscale(_clahe_bgr(crop), 1.75),
    )
    # Independent PP-OCRv6 recognition is a separately SHA-pinned research
    # verifier, never a replacement for the existing V5 detector/recognizer.
    # A failed or conflicting verification is an abstention, not new text.
    if independent_recognizer is not None:
        findings["independent_v6"] = _single_recognition(
            independent_recognizer, crop,
        )
    options = [findings[key] for key in VARIANTS[:-1]
               if key in findings and findings[key][0]
               and findings[key][1] is not None]
    if options:
        # This choice NEVER observes the known source transcript.
        # Lexical correctness is independently evaluated afterwards.
        findings["unsupervised_highest_confidence"] = max(
            options, key=lambda item: item[1],
        )
    return findings



def _independent_variant_agreement(findings):
    """Require two independent recognizer-only enhancements, not confidence.

    A single high-confidence false line is never sufficient. The resulting
    text is an ACTUAL model observation, never merged/generated content.
    """
    raw_key = comparison_key(
        findings.get("raw_detected_upper", ("", None))[0]
    )
    if not raw_key:
        return None
    by_key = {}
    for variant in ("recognize_crop", "recognize_upscaled",
                    "recognize_clahe_upscaled"):
        text, score = findings.get(variant, ("", None))
        key = comparison_key(text)
        if not key or score is None or score < 0.70:
            continue
        if not (0.60 <= len(key)/len(raw_key) <= 1.50):
            continue
        by_key.setdefault(key, []).append((text, score))
    choices = [
        (key, items) for key, items in by_key.items() if len(items) >= 2
    ]
    if not choices:
        return None
    key, items = max(
        choices,
        key=lambda pair: (
            len(pair[1]),
            sum(score for _, score in pair[1]) / len(pair[1]),
        ),
    )
    return max(items, key=lambda item: item[1])


def temporal_agreement(selected):
    """No truth times/text: corroborate across independently sampled frames.

    Samples are grouped by the actual visual detector segment IDs, not
    authored subtitle intervals. Two different frames each containing >=2
    agreeing recognizer-only variants are required; blank flashes and one
    isolated OCR hallucination cannot create a replacement.
    """
    groups = {}
    for segment_id, pts, findings in selected:
        if not isinstance(pts, (float, int)) or not math.isfinite(pts):
            continue
        candidate = _independent_variant_agreement(findings)
        if candidate is not None:
            groups.setdefault(segment_id, []).append((float(pts), *candidate))
    decisions = {}
    for segment_id, readings in groups.items():
        by_key = {}
        for pts, text, score in readings:
            key = comparison_key(text)
            # Multiple results from the SAME source timestamp are a single
            # observation, even if a detector yields duplicate sample frames.
            by_key.setdefault(key, {})
            old = by_key[key].get(pts)
            if old is None or score > old[1]:
                by_key[key][pts] = (text, score)
        viable = [
            list(pts_map.values()) for pts_map in by_key.values()
            if len(pts_map) >= 2
        ]
        if not viable:
            continue
        best = max(
            viable,
            key=lambda items: (
                len(items),
                sum(score for _, score in items)/len(items),
            ),
        )
        # Return an actually OCR-observed rendering, not generated text.
        text, confidence = max(best, key=lambda item: item[1])
        decisions[segment_id] = {
            "text": text,
            "confidence": confidence,
            "support_frames": len(best),
        }
    return decisions



def cross_model_temporal_agreement(selected):
    """Optional v5+v6 recognition corroboration, with conservative abstention.

    The two original v5 crop variants are CORRELATED, not independent
    models. This gate additionally requires >=2 distinct source timestamps
    whose V6 recognizer sees that exact candidate on the SAME frames.
    A confident conflicting V6 result on a supporting frame abstains.
    No source-authored transcript or grading cue times are consulted.
    """
    baseline = temporal_agreement(selected)
    validated = {}
    for segment_id, item in baseline.items():
        wanted = comparison_key(item["text"])
        matched_pts = set()
        conflict = False
        for sid, pts, findings in selected:
            if sid != segment_id or not isinstance(pts, (float, int)):
                continue
            if not math.isfinite(pts):
                continue
            local = _independent_variant_agreement(findings)
            if local is None or comparison_key(local[0]) != wanted:
                continue
            independent_text, independent_conf = findings.get(
                "independent_v6", ("", None),
            )
            conf = _safe_confidence(independent_conf)
            other = comparison_key(independent_text)
            if conf is None or conf < 0.70 or not other:
                continue
            if other == wanted:
                matched_pts.add(float(pts))
            else:
                conflict = True
        if not conflict and len(matched_pts) >= 2:
            validated[segment_id] = {
                **item,
                "independent_model_support_frames": len(matched_pts),
            }
    return validated


def _process_source(path, language, engine, *, independent_recognizer=None):
    selected = []
    frames_seen = 0
    with closing(iter(FFmpegPTSFrameSource(str(path), sample_fps=8.0))) as frames:
        with closing(iter_segment_frames(frames)) as segments:
            for segment_index, segment in enumerate(segments):
                for frame in segment.samples:
                    if frames_seen >= MAX_SAMPLES:
                        raise RuntimeError("unbounded OCR research frame selection")
                    frames_seen += 1
                    raw = engine.engine(
                        frame.image, use_det=True, use_cls=False,
                        use_rec=True, text_score=engine.text_score,
                    )
                    findings = variants_for_raw_frame(
                        raw, frame.image, engine.engine, language=language,
                        independent_recognizer=independent_recognizer,
                    )
                    selected.append((segment_index, frame.pts, findings))
    return selected


def evaluate_known_source(selected, language):
    # Reference cue times are strictly POST-recognition scoring labels;
    # they cannot influence detection, row crops, variants or model inputs.
    reports = []
    consensus = temporal_agreement(selected)
    independently_confirmed = cross_model_temporal_agreement(selected)
    for index, (start, end) in enumerate(EXPECTED_SPANS):
        expected_upper, _ = _multiline_reference(VIDEO_TEXT[language][index])
        truth = comparison_key(expected_upper)
        matching = [(seg, v) for seg, pts, v in selected if start <= pts < end]
        results = [v for _, v in matching]
        matched_segment_ids = {seg for seg, _ in matching}
        produced = [consensus[seg] for seg in matched_segment_ids
                    if seg in consensus]
        verified = [
            independently_confirmed[seg] for seg in matched_segment_ids
            if seg in independently_confirmed
        ]
        per = {
            "two_variant_two_frame_consensus_segments": len(produced),
            "two_variant_two_frame_correct_segments": sum(
                comparison_key(p["text"]) == truth for p in produced
            ),
            "cross_model_verified_segments": len(verified),
            "cross_model_verified_correct_segments": sum(
                comparison_key(p["text"]) == truth for p in verified
            ),
            "scored_selected_frames": len(results),
            "recognized_upper_candidate_frames": sum(bool(x) for x in results),
        }
        for variant in VARIANTS:
            available = [v[variant][0] for v in results
                         if variant in v and v[variant][0]]
            per[variant+"_samples"] = len(available)
            per[variant+"_correct"] = sum(
                comparison_key(x) == truth for x in available
            )
        reports.append(per)
    return reports


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--manifest", default="models/manifest.json")
    ap.add_argument(
        "--cross-model", action="store_true",
        help="opt-in independent, pinned PP-OCRv6 recognition-only verification",
    )
    opts = ap.parse_args()
    cases = []
    try:
        with tempfile.TemporaryDirectory(prefix="hse-upper-recognizer-ab-") as td:
            v6_rec = v6_dict = None
            if opts.cross_model:
                from compare_cjk_v5_v6_recognition import (
                    V6_RECOGNIZER, V6_DICTIONARY, retrieve_verified,
                    make_v6_recognizer,
                )
                official = Path(td)/"official-v6-only"
                official.mkdir()
                v6_rec = retrieve_verified(V6_RECOGNIZER, official)
                v6_dict = retrieve_verified(V6_DICTIONARY, official)
            for language in LANGUAGES:
                try:
                    root = Path(td)/language
                    root.mkdir()
                    with_text = root/"with.mp4"
                    no_text = root/"blank.mp4"
                    encode_video(with_text, language, style="white_two_line")
                    encode_video(no_text, language, style="white_two_line",
                                 include_subtitles=False)
                    ocr = RapidOCRAdapter(
                        language=language, model_dir=str(opts.model_dir),
                        manifest_path=str(opts.manifest),
                    )
                    verifier = (
                        make_v6_recognizer(
                            Path(opts.model_dir), Path(opts.manifest),
                            language, v6_rec, v6_dict,
                        ) if opts.cross_model else None
                    )
                    recognized = _process_source(
                        with_text, language, ocr,
                        independent_recognizer=verifier,
                    )
                    blank = _process_source(
                        no_text, language, ocr,
                        independent_recognizer=verifier,
                    )
                    row = {
                        "language": language,
                        "executed": True,
                        "independent_v6_enabled": opts.cross_model,
                        "independent_v6_official_sha_verified": opts.cross_model,
                        "reference_not_used_for_frame_or_crop_selection": True,
                        "source_frame_count": len(recognized),
                        "blank_control_frames": len(blank),
                        "blank_upper_candidate_frames": sum(
                            bool(v) for _, _, v in blank
                        ),
                        "blank_unsupervised_readings": sum(
                            bool(v.get("unsupervised_highest_confidence"))
                            for _, _, v in blank
                        ),
                        "blank_two_variant_two_frame_consensus_segments": len(
                            temporal_agreement(blank)
                        ),
                        "blank_cross_model_verified_segments": len(
                            cross_model_temporal_agreement(blank)
                        ),
                        "source_two_variant_two_frame_consensus_segments": len(
                            temporal_agreement(recognized)
                        ),
                        "source_cross_model_verified_segments": len(
                            cross_model_temporal_agreement(recognized)
                        ),
                        "reference_cues": evaluate_known_source(recognized, language),
                    }
                except Exception as exc:
                    row = {"language": language, "executed": False,
                           "error_class": type(exc).__name__}
                cases.append(row)
                print("HSE_UPPER_REC_ONLY_ROW " +
                      json.dumps(row, sort_keys=True), flush=True)
    except Exception as exc:
        print("HSE_UPPER_REC_ONLY_FATAL " +
              json.dumps({"error_class": type(exc).__name__}))
        return 3
    completed = len(cases) == len(LANGUAGES) and all(
        row.get("executed") and len(row.get("reference_cues", [])) == 2
        for row in cases
    )
    summary = {
        "schema": 1,
        "experiment_complete": completed,
        "languages": list(LANGUAGES),
        "independent_blank_control_included": True,
        "independent_v6_recognizer_used": opts.cross_model,
        "reference_oracle_used_for_model_inputs": False,
        "production_changed": False,
        "real_anime_tested": False,
        "candidate_deployed": False,
    }
    print("HSE_UPPER_REC_ONLY_SUMMARY "+
          json.dumps(summary, sort_keys=True))
    return 0 if completed else 4


if __name__ == "__main__":
    raise SystemExit(main())
