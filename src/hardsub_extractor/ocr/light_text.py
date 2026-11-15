"""Research-only white/light subtitle glyph polarity filter.

No extra OCR model calls. When explicitly enabled, accept OCR lines only
when the already recognized glyph boxes contain bright text against a
darker local background. This rejects black article/credit text on bright
pages, but WILL reject dark/colored real subtitles. Never enable by default.
"""
from __future__ import annotations

import math

import cv2
import numpy as np

from ..models import OCRObservation
from .base import OCREngine


def _white_on_darker_patch(image: np.ndarray, row: dict) -> bool:
    """Compare glyph highlights with local median, not language or words."""
    if not isinstance(row, dict) or image.ndim != 3:
        return False
    height, width = image.shape[:2]
    if height < 20 or width < 20:
        return False
    keys = ("cx", "cy", "width", "height")
    numbers = [row.get(k) for k in keys]
    if any(type(x) not in (float, int) or not math.isfinite(x) for x in numbers):
        return False
    cx, cy, w, h = map(float, numbers)
    if not (0 <= cx <= 1 and 0 <= cy <= 1 and 0 < w <= 1 and 0 < h <= 1):
        return False
    x0 = max(0, min(width, int((cx - w * 0.5) * width)))
    x1 = max(0, min(width, int(math.ceil((cx + w * 0.5) * width))))
    y0 = max(0, min(height, int((cy - h * 0.5) * height)))
    y1 = max(0, min(height, int(math.ceil((cy + h * 0.5) * height))))
    if x1 - x0 < 4 or y1 - y0 < 5 or (x1 - x0) * (y1 - y0) > 2_000_000:
        return False
    luminance = cv2.cvtColor(image[y0:y1, x0:x1], cv2.COLOR_BGR2GRAY)
    median = float(np.median(luminance))
    bright = float(np.percentile(luminance, 95))
    # Black-on-white article glyphs have a near-white background median;
    # white-on-gray subtitle cards or white outlined dialogue have a
    # noticeably lower median with visible white highlights.
    return bool(bright >= 215 and median <= 201 and bright - median >= 28)


def select_light_caption_lines(
    observation: OCRObservation, image: np.ndarray,
) -> OCRObservation:
    """All lines are opt-in candidates; empty OCR evidence stays empty.

    Require >=75% recognized characters in each output visual row to have
    light-on-dark box evidence, permitting small punctuation-box variation.
    On missing/invalid position metadata, the explicit light-text-only
    option fails closed rather than treating arbitrary black webpage
    text as a valid subtitle. Default HSE never calls this selector.
    """
    if not observation.text:
        return observation
    original_lines = observation.text.splitlines()
    metadata = observation.metadata or {}
    records = metadata.get("detections") if isinstance(metadata, dict) else None
    if not isinstance(records, list) or len(records) > 256 or not original_lines:
        return OCRObservation("", 0.0, observation.pts, {"light_text_evidence": "missing"})

    totals = [0] * len(original_lines)
    bright = [0] * len(original_lines)
    accepted_records = []
    for row in records:
        if not isinstance(row, dict) or type(row.get("line")) is not int:
            return OCRObservation("", 0.0, observation.pts, {"light_text_evidence": "invalid"})
        line = row["line"]
        if not 0 <= line < len(original_lines):
            return OCRObservation("", 0.0, observation.pts, {"light_text_evidence": "invalid"})
        chars = len(str(row.get("text") or "").strip())
        if chars == 0:
            continue
        totals[line] += chars
        if _white_on_darker_patch(image, row):
            bright[line] += chars
            accepted_records.append(row)

    keep = [
        i for i in range(len(original_lines))
        if totals[i] > 0 and bright[i] >= math.ceil(totals[i] * .75)
    ]
    if not keep:
        return OCRObservation("", 0.0, observation.pts, {"light_text_evidence": "absent"})
    if len(keep) == len(original_lines):
        return observation
    indices = {original: j for j, original in enumerate(keep)}
    new_metadata = dict(metadata)
    kept_records = [
        {**row, "line": indices[row["line"]]}
        for row in accepted_records
        if row["line"] in indices
    ]
    new_metadata["detections"] = kept_records
    new_metadata["light_text_discarded_rows"] = len(original_lines) - len(keep)
    # A high-scoring black title/credit may have dominated the raw sentence
    # confidence. After discarding it, never let its score certify weak real
    # subtitle text. Character-weight the ONLY retained bright OCR boxes and
    # lower confidence only with complete, valid retained-row evidence.
    confidence = observation.confidence
    weighted_scores = []
    if (
        kept_records
        and {row["line"] for row in kept_records} == set(range(len(keep)))
    ):
        for record in kept_records:
            score = record.get("score")
            chars = len(str(record.get("text") or "").strip())
            if (
                type(score) not in (int, float)
                or not math.isfinite(score) or not 0 <= score <= 1
                or chars <= 0
            ):
                weighted_scores = []
                break
            weighted_scores.append((float(score), chars))
    if weighted_scores:
        retained_mean = (
            sum(score * chars for score, chars in weighted_scores)
            / sum(chars for _, chars in weighted_scores)
        )
        if retained_mean < confidence:
            confidence = retained_mean
            new_metadata["light_text_retained_confidence_bounded"] = True
    return OCRObservation(
        "\n".join(original_lines[i] for i in keep),
        confidence, observation.pts, new_metadata,
    )


class LightCaptionEngine(OCREngine):
    """Explicit local OCR style mode, not the baseline or auto-release."""

    def __init__(self, inner: OCREngine):
        self.inner = inner
        self.discarded_rows = 0
        self.candidate_frames = 0

    def recognize(self, frame) -> OCRObservation:
        # Bright subtitles must participate in adaptive retry/voting BEFORE
        # a strong black webpage title can win the result. Use the untouched
        # video frame for every sample's polarity check: normalized detection
        # boxes still refer to the same content after optional upscale/CLAHE,
        # while enhancement itself must not repaint dark source glyphs white.
        # This is experimental-only. The default AdaptiveOCREngine is unchanged.
        from .adaptive import AdaptiveOCREngine

        had_text = False

        def light_only(observation: OCRObservation) -> OCRObservation:
            nonlocal had_text
            if not observation.text:
                return observation
            had_text = True
            filtered = select_light_caption_lines(observation, frame.image)
            self.discarded_rows += max(
                0,
                len(observation.text.splitlines())
                - len(filtered.text.splitlines()),
            )
            return filtered

        if isinstance(self.inner, AdaptiveOCREngine):
            result = self.inner.recognize_with_filter(frame, light_only)
        else:
            result = light_only(self.inner.recognize(frame))
        # Keep a per-input-frame counter instead of treating model retries
        # as extra video frames.
        if had_text:
            self.candidate_frames += 1
        return result

    def __getattr__(self, key: str):
        return getattr(self.inner, key)
