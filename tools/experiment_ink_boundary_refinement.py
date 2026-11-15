"""Experimental OCR-seeded cue-boundary refinement on real encoded frames.

The production app is NOT modified or connected to this experiment.
A confirmed OCR cue supplies a geometry anchor (normalized PP-OCRv5 boxes).
A few nearby frames establish a stable bright text-ink template. Then the
source video is re-read and ink survival is measured on the original frame
times, independent of broad scene motion in the remaining subtitle ROI.

Careful: the known synthetic reference interval selects each test anchor.
The experiment measures only whether recovery is feasible, not unsupervised
discovery, real-anime accuracy or a user-facing production feature.
"""
from __future__ import annotations

import argparse
from contextlib import closing
from dataclasses import dataclass
import json
from pathlib import Path
import sys
import tempfile
from typing import Iterable

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hardsub_extractor.frame_source import FFmpegPTSFrameSource
from hardsub_extractor.models import Frame
from hardsub_extractor.ocr.rapidocr_adapter import RapidOCRAdapter
from verify_animated_background_ocr_quality import encode_video
from verify_multilang_video_pipeline import EXPECTED_SPANS


@dataclass(frozen=True)
class InkAnchor:
    box: tuple[int, int, int, int]
    stable_white_pixels: np.ndarray
    source_pts: float
    boxes_used: int


def _white_ink(frame_crop: np.ndarray) -> np.ndarray:
    """Conservative *image* filter; the known box supplies semantic location."""
    if frame_crop.ndim != 3 or frame_crop.shape[2] != 3:
        raise ValueError("expected a real BGR subtitle crop")
    bgr = frame_crop.astype(np.int16)
    bright = np.min(bgr, axis=2) >= 208
    low_chroma = (
        np.max(bgr, axis=2) - np.min(bgr, axis=2)
    ) <= 46
    return np.logical_and(bright, low_chroma)


def _ocr_box(metadata: dict | None, shape) -> tuple[tuple[int, int, int, int], int] | None:
    if not isinstance(metadata, dict) or not isinstance(metadata.get("detections"), list):
        return None
    height, width = shape[:2]
    rectangles = []
    for record in metadata["detections"]:
        if not isinstance(record, dict):
            continue
        try:
            cx = float(record["cx"])
            cy = float(record["cy"])
            bw = float(record["width"])
            bh = float(record["height"])
            score = float(record.get("score", 0))
        except (ValueError, TypeError, KeyError, OverflowError):
            continue
        if not all(np.isfinite(x) for x in (cx, cy, bw, bh, score)):
            continue
        if not (0.0 < cx < 1.0 and 0.0 < cy < 1.0 and
                0.0 < bw <= 1.0 and 0.0 < bh <= 0.8 and score >= 0.45):
            continue
        x0 = (cx - bw / 2.0) * width
        x1 = (cx + bw / 2.0) * width
        y0 = (cy - bh / 2.0) * height
        y1 = (cy + bh / 2.0) * height
        if x1 <= x0 or y1 <= y0:
            continue
        rectangles.append((x0, y0, x1, y1))
    if not rectangles:
        return None
    margin_x = max(2, round(width * 0.004))
    margin_y = max(2, round(height * 0.012))
    x0 = max(0, int(min(x[0] for x in rectangles)) - margin_x)
    y0 = max(0, int(min(x[1] for x in rectangles)) - margin_y)
    x1 = min(width, int(max(x[2] for x in rectangles) + 1) + margin_x)
    y1 = min(height, int(max(x[3] for x in rectangles) + 1) + margin_y)
    if (x1 - x0 < 15 or y1 - y0 < 12 or
            (x1 - x0) * (y1 - y0) >= 0.5 * width * height):
        return None
    return (x0, y0, x1, y1), len(rectangles)


def build_anchor(reference: Frame, nearby: Iterable[Frame],
                 ocr_metadata: dict | None, *,
                 min_pixels: int = 24) -> InkAnchor | None:
    """Use *independent OCR geometry* and co-occurring stable white glyphs.

    Do not manufacture an anchor if OCR did not actually detect a rectangle.
    Motion across the background is removed using agreement with at least one
    nearby frame; no transcript is synthesized from visual similarity.
    """
    outcome = _ocr_box(ocr_metadata, reference.image.shape)
    if outcome is None:
        return None
    box, boxes_used = outcome
    x0, y0, x1, y1 = box
    ref = _white_ink(reference.image[y0:y1, x0:x1])
    observations = 0
    stable = np.zeros_like(ref, dtype=bool)
    kernel = np.ones((3, 3), dtype=np.uint8)
    for candidate in nearby:
        if candidate.image.shape != reference.image.shape:
            raise ValueError("mismatched frame dimensions")
        if abs(candidate.pts - reference.pts) > 0.65:
            continue
        if candidate.pts == reference.pts:
            continue
        ink = _white_ink(candidate.image[y0:y1, x0:x1])
        dilated = cv2.dilate(ink.astype(np.uint8), kernel, iterations=1) > 0
        stable |= np.logical_and(ref, dilated)
        observations += 1
    if observations < 2:
        return None
    count = int(np.count_nonzero(stable))
    if count < min_pixels or count < (stable.size * 0.003):
        return None
    return InkAnchor(box, stable, reference.pts, boxes_used)


def ink_support(anchor: InkAnchor, candidate: Frame) -> float:
    x0, y0, x1, y1 = anchor.box
    if (y1 > candidate.image.shape[0] or x1 > candidate.image.shape[1]
            or anchor.stable_white_pixels.shape != (y1 - y0, x1 - x0)):
        raise ValueError("anchor geometry does not match source frame")
    test = _white_ink(candidate.image[y0:y1, x0:x1])
    test = cv2.dilate(test.astype(np.uint8), np.ones((3, 3), np.uint8), iterations=1)
    return float(np.count_nonzero(anchor.stable_white_pixels & (test > 0))
                 / np.count_nonzero(anchor.stable_white_pixels))


def infer_presence(frames: list[Frame], anchor: InkAnchor, *,
                   threshold: float = 0.76) -> list[float]:
    if not 0.5 <= threshold <= 1.0:
        raise ValueError("invalid bounded matching threshold")
    return [frame.pts for frame in frames
            if ink_support(anchor, frame) >= threshold]


def hysteresis_presence(
    frames: list[Frame], anchor: InkAnchor, *,
    high_threshold: float = 0.76,
    low_threshold: float = 0.55,
    bridge_gap_frames: int = 1,
) -> list[float]:
    """Contiguous candidate spans must contain >=2 independent strong frames.

    Once the same subtitle is strongly supported, weaker *neighboring* frames
    may extend its boundaries. This rejects isolated bright motion bursts.
    No subtitle text or language is invented, and an unseen cue cannot begin
    without strong OCR-seeded visual confirmation.
    """
    if not (0.45 <= low_threshold < high_threshold <= 1.0):
        raise ValueError("invalid hysteresis confidence band")
    if bridge_gap_frames not in (0, 1):
        raise ValueError("only bounded one-frame gap bridging supported")
    if not frames:
        return []
    scores = [ink_support(anchor, frame) for frame in frames]
    low = [value >= low_threshold for value in scores]
    if bridge_gap_frames:
        # Base decisions only: do not let two adjacent errors bridge to each
        # other or fabricate a long unobserved interval.
        base = list(low)
        for idx in range(1, len(base) - 1):
            if (not base[idx] and base[idx - 1] and base[idx + 1]
                    and frames[idx + 1].pts - frames[idx - 1].pts <= 0.30):
                low[idx] = True
    selected = []
    start = None
    for idx in range(len(frames) + 1):
        present = idx < len(frames) and low[idx]
        if present and start is None:
            start = idx
        elif not present and start is not None:
            strong = [
                j for j in range(start, idx)
                if scores[j] >= high_threshold
            ]
            if len(strong) >= 2 and (
                frames[strong[-1]].pts > frames[strong[0]].pts
            ):
                selected.extend(frame.pts for frame in frames[start:idx])
            start = None
    return selected


def _evaluation(frames: list[Frame], no_caption: list[Frame], anchor: InkAnchor,
                reference_span: tuple[float, float], *,
                threshold: float = 0.76,
                low_threshold: float | None = None) -> dict:
    if low_threshold is None:
        present = infer_presence(frames, anchor, threshold=threshold)
        blank = infer_presence(no_caption, anchor, threshold=threshold)
    else:
        present = hysteresis_presence(
            frames, anchor, high_threshold=threshold,
            low_threshold=low_threshold,
        )
        blank = hysteresis_presence(
            no_caption, anchor, high_threshold=threshold,
            low_threshold=low_threshold,
        )
    start, end = reference_span
    true_samples = [x.pts for x in frames if start <= x.pts < end]
    recovered_true = [x for x in present if start <= x < end]
    false_times = [x for x in present if not (start <= x < end)]
    first_error = (
        round(abs(recovered_true[0] - start), 3) if recovered_true else None
    )
    last_error = (
        round(abs(end - recovered_true[-1] - 0.125), 3)
        if recovered_true else None
    )
    return {
        "anchor_box_count": anchor.boxes_used,
        "anchor_ink_pixels": int(np.count_nonzero(anchor.stable_white_pixels)),
        "true_frames": len(true_samples),
        "recovered_true_frames": len(recovered_true),
        "false_non_cue_frames": len(false_times),
        "false_on_blank_control": len(blank),
        "start_error_seconds": first_error,
        "end_error_seconds": last_error,
        "recovered_samples": len(present),
    }


def run_language(root: Path, language: str, model_dir: Path,
                 manifest: Path, *, style: str = "white_single") -> dict:
    source = root / f"{language}-caption.mp4"
    control = root / f"{language}-no-caption.mp4"
    encode_video(source, language, style=style)
    encode_video(control, language, include_subtitles=False, style=style)
    with closing(iter(FFmpegPTSFrameSource(str(source), sample_fps=8.0))) as seq:
        frames = list(seq)
    with closing(iter(FFmpegPTSFrameSource(str(control), sample_fps=8.0))) as seq:
        no_caption = list(seq)
    if len(frames) != len(no_caption) or len(frames) < 50:
        raise RuntimeError("source and blank control timeline do not match")
    engine = RapidOCRAdapter(
        language=language, model_dir=str(model_dir), manifest_path=str(manifest),
    )
    cues = []
    for span in EXPECTED_SPANS:
        midpoint = sum(span) / 2
        # The oracle picks a stable center *only in the benchmark*; production
        # would select an OCR-confirmed anchor from its existing segment.
        reference = min(frames, key=lambda x: abs(x.pts - midpoint))
        nearby = [f for f in frames if 0 < abs(f.pts - reference.pts) <= 0.26]
        observation = engine.recognize(reference)
        anchor = build_anchor(reference, nearby, observation.metadata)
        if anchor is None:
            cues.append({
                "anchor_ocr_has_text": bool(observation.text),
                "anchor_verified": False,
                "reason": "no_stable_ocr_geometry_or_ink",
            })
            continue
        entry = _evaluation(frames, no_caption, anchor, span)
        entry["comparisons"] = {
            "raw_threshold_0_65": _evaluation(
                frames, no_caption, anchor, span, threshold=0.65,
            ),
            "hysteresis_low_0_55": _evaluation(
                frames, no_caption, anchor, span,
                threshold=0.76, low_threshold=0.55,
            ),
            "hysteresis_low_0_62": _evaluation(
                frames, no_caption, anchor, span,
                threshold=0.76, low_threshold=0.62,
            ),
        }
        entry["anchor_ocr_has_text"] = bool(observation.text)
        entry["anchor_verified"] = True
        cues.append(entry)
    return {
        "language": language, "style": style, "cues": cues,
        "decoded_frames": len(frames),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--manifest", default=str(ROOT / "models" / "manifest.json"))
    args = parser.parse_args()
    try:
        with tempfile.TemporaryDirectory(prefix="hse-ink-boundary-exp-") as td:
            root = Path(td)
            summary = [
                run_language(root, lang, Path(args.model_dir), Path(args.manifest))
                for lang in ("ko", "en", "ja", "zh")
            ]
    except Exception as exc:
        print("HSE_INK_BOUNDARY_STUDY_ERROR " + json.dumps({
            "error_class": type(exc).__name__,
        }))
        return 3
    results = [item for language in summary for item in language["cues"]]
    successes = [
        row for row in results if row.get("anchor_verified") is True
        and row.get("false_on_blank_control") == 0
        and row.get("false_non_cue_frames") == 0
        and row.get("start_error_seconds", 2) <= 0.25
        and row.get("end_error_seconds", 2) <= 0.25
    ]
    candidates = ("raw_threshold_0_65", "hysteresis_low_0_55",
                  "hysteresis_low_0_62")
    def strict(row: dict) -> bool:
        return (
            row.get("false_on_blank_control") == 0
            and row.get("false_non_cue_frames") == 0
            and row.get("start_error_seconds") is not None
            and row["start_error_seconds"] <= 0.25
            and row.get("end_error_seconds") is not None
            and row["end_error_seconds"] <= 0.25
        )
    candidate_results = {
        name: {
            "strict_cues": sum(
                strict(row.get("comparisons", {}).get(name, {}))
                for row in results
            ),
            "blank_control_false_frames": sum(
                row.get("comparisons", {}).get(name, {}).get(
                    "false_on_blank_control", 0,
                )
                for row in results
            ),
        }
        for name in candidates
    }
    report = {
        "schema": 1,
        "experiment_complete": len(results) == 8,
        "production_changed": False,
        "licensed_real_anime_tested": False,
        "actual_website_tested": False,
        "reference_anchors_come_from_known_synthetic_cue_centers": True,
        "precise_boundary_candidates": len(successes),
        "candidate_comparisons": candidate_results,
        "candidate_deployed": False,
        "tested_anchors": 8,
        "languages": summary,
    }
    print("HSE_INK_BOUNDARY_STUDY " + json.dumps(report, sort_keys=True))
    # A successful job shows that the study *completed*, not that all eight
    # captions passed. Never silently promote failure into production.
    return 0 if report["experiment_complete"] else 4


if __name__ == "__main__":
    raise SystemExit(main())
