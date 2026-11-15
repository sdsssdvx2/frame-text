"""Experimental color/outline-aware OCR geometry masks; NEVER production.

Uses independently detected OCR rectangles and synthetic reference midpoints.
Success does not establish real-episode discovery, accuracy, or deployment safety.
"""
from __future__ import annotations

import argparse
from contextlib import closing
from dataclasses import dataclass
import json
from pathlib import Path
import tempfile

import cv2
import numpy as np

from experiment_ink_boundary_refinement import _ocr_box, _evaluation, build_anchor
from hardsub_extractor.frame_source import FFmpegPTSFrameSource
from hardsub_extractor.models import Frame
from hardsub_extractor.ocr.rapidocr_adapter import RapidOCRAdapter
from hardsub_extractor.ocr.base import comparison_key
from verify_animated_background_ocr_quality import encode_video
from verify_multilang_video_pipeline import EXPECTED_SPANS, VIDEO_TEXT
from verify_multilang_ocr_synthetic import edit_distance

STYLES = ("white_single", "white_two_line", "yellow_single")
LANGUAGES = ("ko", "en", "ja", "zh")


@dataclass(frozen=True)
class PaletteAnchor:
    box: tuple[int, int, int, int]
    geometry: np.ndarray
    stable_pixels: np.ndarray
    palette: str
    hue: int | None


def _rectangle_geometry(shape, metadata, box):
    """Only use boxes returned by OCR, not blank union-box gaps."""
    h, w = shape[:2]
    x0, y0, x1, y1 = box
    keep = np.zeros((y1-y0, x1-x0), dtype=bool)
    for rec in metadata.get("detections", ()):
        if not isinstance(rec, dict):
            continue
        try:
            cx, cy, bw, bh, score = (
                float(rec[key]) for key in ("cx", "cy", "width", "height", "score")
            )
        except (KeyError, TypeError, ValueError, OverflowError):
            continue
        if not all(np.isfinite(v) for v in (cx, cy, bw, bh, score)):
            continue
        if not (0 < cx < 1 and 0 < cy < 1 and 0 < bw <= 1
                and 0 < bh <= 0.8 and score >= 0.45):
            continue
        xa = max(x0, int((cx-bw/2)*w)-2)
        xb = min(x1, int((cx+bw/2)*w)+3)
        ya = max(y0, int((cy-bh/2)*h)-2)
        yb = min(y1, int((cy+bh/2)*h)+3)
        if xb > xa and yb > ya:
            keep[ya-y0:yb-y0, xa-x0:xb-x0] = True
    return keep


def _base_highlights(crop, geometry):
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    white = (np.min(crop, axis=2) >= 208) & (
        np.max(crop, axis=2).astype(np.int16)
        - np.min(crop, axis=2).astype(np.int16) <= 46
    )
    colored = (hsv[:, :, 1] >= 95) & (hsv[:, :, 2] >= 175)
    # An outlined-glyph experiment, not a universal rule: unoutlined text
    # and other colors can fail this discriminant.
    dark = (np.max(crop, axis=2) <= 105).astype(np.uint8)
    near_dark = cv2.dilate(dark, np.ones((9, 9), np.uint8)) > 0
    return hsv, white & near_dark & geometry, colored & near_dark & geometry


def _palette_mask(crop, geometry, palette, hue):
    hsv, white, colored = _base_highlights(crop, geometry)
    if palette == "white":
        return white
    if palette != "colored" or hue is None:
        raise ValueError("invalid calibrated palette")
    distance = np.abs(hsv[:, :, 0].astype(np.int16) - hue)
    distance = np.minimum(distance, 180-distance)
    return colored & (distance <= 12)


def build_palette_anchor(reference: Frame, nearby, metadata, *, min_pixels=24):
    outcome = _ocr_box(metadata, reference.image.shape)
    if outcome is None:
        return None
    box, _ = outcome
    x0, y0, x1, y1 = box
    geom = _rectangle_geometry(reference.image.shape, metadata, box)
    crop = reference.image[y0:y1, x0:x1]
    hsv, _, colored = _base_highlights(crop, geom)
    hue = None
    if np.count_nonzero(colored) >= min_pixels:
        histogram = np.bincount(hsv[:, :, 0][colored] // 6, minlength=30)
        hue = int(histogram.argmax()) * 6 + 3
    candidates = []
    for palette, color in (("white", None), ("colored", hue)):
        if palette == "colored" and hue is None:
            continue
        ref_mask = _palette_mask(crop, geom, palette, color)
        if np.count_nonzero(ref_mask) < min_pixels:
            continue
        stable = np.zeros_like(ref_mask)
        observations = 0
        for frame in nearby:
            if frame.image.shape != reference.image.shape:
                raise ValueError("inconsistent source frame dimensions")
            if not (0 < abs(frame.pts-reference.pts) <= 0.65):
                continue
            candidate = _palette_mask(
                frame.image[y0:y1, x0:x1], geom, palette, color,
            )
            candidate = cv2.dilate(candidate.astype(np.uint8),
                                   np.ones((3, 3), np.uint8)) > 0
            stable |= ref_mask & candidate
            observations += 1
        support = np.count_nonzero(stable)
        if observations >= 2 and support >= min_pixels and support >= 0.003*geom.size:
            candidates.append((int(support), palette, color, stable))
    if not candidates:
        return None
    _, palette, color, stable = max(candidates, key=lambda x: x[0])
    return PaletteAnchor(box, geom, stable, palette, color)


def palette_score(anchor: PaletteAnchor, frame: Frame) -> float:
    x0, y0, x1, y1 = anchor.box
    if (x1 > frame.image.shape[1] or y1 > frame.image.shape[0]
            or anchor.stable_pixels.shape != (y1-y0, x1-x0)):
        raise ValueError("invalid source-frame shape")
    sample = _palette_mask(
        frame.image[y0:y1, x0:x1], anchor.geometry,
        anchor.palette, anchor.hue,
    )
    sample = cv2.dilate(sample.astype(np.uint8),
                        np.ones((3, 3), np.uint8)) > 0
    support = int(np.count_nonzero(anchor.stable_pixels))
    if support == 0:
        raise ValueError("empty anchored template")
    return float(np.count_nonzero(anchor.stable_pixels & sample) / support)


def presence(frames, anchor, *, high=0.76, low=0.62):
    """Same bounded two-threshold policy as original white experiment."""
    if not 0.45 <= low < high <= 1.0:
        raise ValueError("invalid thresholds")
    scores = [palette_score(anchor, x) for x in frames]
    weak = [s >= low for s in scores]
    initial = list(weak)
    for i in range(1, len(frames)-1):
        if (not initial[i] and initial[i-1] and initial[i+1]
                and frames[i+1].pts-frames[i-1].pts <= 0.30):
            weak[i] = True
    selected = []
    start = None
    for i in range(len(frames)+1):
        if i < len(frames) and weak[i]:
            if start is None:
                start = i
        elif start is not None:
            strong = [j for j in range(start, i) if scores[j] >= high]
            if len(strong) >= 2 and frames[strong[-1]].pts > frames[strong[0]].pts:
                selected.extend(x.pts for x in frames[start:i])
            start = None
    return selected


def evaluate(frames, blank, anchor, span):
    start, end = span
    found = presence(frames, anchor)
    controls = presence(blank, anchor)
    within = [t for t in found if start <= t < end]
    return {
        "anchor_found": True,
        "palette": anchor.palette,
        "true_recalled": len(within),
        "false_outside": len(found)-len(within),
        "false_blank": len(controls),
        "start_error": round(abs(within[0]-start), 3) if within else None,
        "end_error": round(abs(end-within[-1]-0.125), 3) if within else None,
    }


def strict(entry):
    return bool(entry.get("anchor_found")
                and entry.get("false_blank") == 0
                and entry.get("false_outside") == 0
                and entry.get("start_error") is not None
                and entry.get("end_error") is not None
                and max(entry["start_error"], entry["end_error"]) <= 0.25)


def text_fidelity(observed, expected: str, style: str) -> dict:
    """Score independent synthetic text, including two-line completeness."""
    expected_key = comparison_key(expected)
    observed_key = comparison_key(observed.text)
    expected_lines = 2 if style == "white_two_line" else 1
    observed_lines = len([line for line in observed.text.splitlines() if line.strip()])
    return {
        "exact_key": observed_key == expected_key,
        "cer": round(edit_distance(expected_key, observed_key)
                     / max(1, len(expected_key)), 4),
        "reference_lines": expected_lines,
        "observed_lines": observed_lines,
        "line_count_match": expected_lines == observed_lines,
    }


def run_language(root, language, style, model_dir, manifest):
    source = root / "with.mp4"
    blank = root / "blank.mp4"
    encode_video(source, language, style=style)
    encode_video(blank, language, include_subtitles=False, style=style)
    with closing(iter(FFmpegPTSFrameSource(str(source), sample_fps=8))) as frames_iter:
        frames = list(frames_iter)
    with closing(iter(FFmpegPTSFrameSource(str(blank), sample_fps=8))) as frames_iter:
        controls = list(frames_iter)
    if len(frames) != len(controls) or len(frames) < 50:
        raise RuntimeError("encoded frames differ from blank control")
    ocr = RapidOCRAdapter(language=language, model_dir=str(model_dir),
                          manifest_path=str(manifest))
    cues = []
    for cue_idx, span in enumerate(EXPECTED_SPANS):
        mid = sum(span)/2
        ref = min(frames, key=lambda f: abs(f.pts-mid))
        nearby = [f for f in frames if 0 < abs(f.pts-ref.pts) <= 0.26]
        observation = ocr.recognize(ref)
        transcription = text_fidelity(observation, VIDEO_TEXT[language][cue_idx], style)
        old = build_anchor(ref, nearby, observation.metadata)
        new = build_palette_anchor(ref, nearby, observation.metadata)
        original = (_evaluation(frames, controls, old, span, low_threshold=0.62)
                    if old is not None else {})
        prior = {
            "anchor_found": old is not None,
            "false_blank": original.get("false_on_blank_control"),
            "false_outside": original.get("false_non_cue_frames"),
            "start_error": original.get("start_error_seconds"),
            "end_error": original.get("end_error_seconds"),
        }
        candidate = evaluate(frames, controls, new, span) if new is not None else {}
        cues.append({"old_strict": strict(prior),
                     "candidate_strict": strict(candidate),
                     "text_fidelity": transcription,
                     "joint_text_and_timing_strict": (
                         strict(candidate) and transcription["exact_key"]
                         and transcription["line_count_match"]
                     ),
                     "candidate": candidate,
                     "old": prior,
                     "ocr_text_nonempty": bool(observation.text)})
    return {"style": style, "language": language, "cues": cues}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--manifest", default="models/manifest.json")
    args = parser.parse_args()
    results = []
    try:
        with tempfile.TemporaryDirectory(prefix="hse-palette-exp-") as temp:
            for style in STYLES:
                for language in LANGUAGES:
                    folder = Path(temp) / style / language
                    folder.mkdir(parents=True)
                    row = run_language(folder, language, style,
                                       Path(args.model_dir), Path(args.manifest))
                    results.append(row)
                    print("HSE_PALETTE_ROW " + json.dumps(row, sort_keys=True), flush=True)
    except Exception as exc:
        print("HSE_PALETTE_ERROR " + json.dumps({"exception_class": type(exc).__name__}))
        return 3
    by_style = {}
    for style in STYLES:
        entries = [cue for item in results if item["style"] == style
                   for cue in item["cues"]]
        by_style[style] = {
            "tested": len(entries),
            "old_strict": sum(c["old_strict"] for c in entries),
            "candidate_strict": sum(c["candidate_strict"] for c in entries),
            "text_exact": sum(c["text_fidelity"]["exact_key"] for c in entries),
            "line_count_match": sum(c["text_fidelity"]["line_count_match"] for c in entries),
            "joint_text_and_timing_strict": sum(c["joint_text_and_timing_strict"] for c in entries),
            "max_cer": max(c["text_fidelity"]["cer"] for c in entries),
            "blank_false": sum((c["candidate"].get("false_blank") or 0) for c in entries),
            "missing_candidate": sum(not c["candidate"] for c in entries),
        }
    report = {"schema": 1, "complete": len(results) == 12,
              "reference_anchor_oracle_used": True,
              "real_anime_validated": False, "production_changed": False,
              "candidate_deployed": False, "by_style": by_style}
    print("HSE_PALETTE_SUMMARY " + json.dumps(report, sort_keys=True))
    return 0 if report["complete"] else 4


if __name__ == "__main__":
    raise SystemExit(main())
