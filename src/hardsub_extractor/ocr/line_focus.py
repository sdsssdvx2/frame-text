"""Explicit experimental last-two-row focus for VISUAL hard-subtitle OCR.

Only observations already produced by the selected local RapidOCR engine
are filtered. Never run another model, infer missing words, change timings,
select a model, or modify the source subtitle-first decision. Three-line
dialogue can lose genuine text. Keep OFF unless the user explicitly opts in.
"""
from __future__ import annotations

import math

from ..models import Frame, OCRObservation
from .base import OCREngine


class BottomTwoLineEngine(OCREngine):
    """Reusable form of the independently measured publisher-caption study."""

    def __init__(self, base: OCREngine):
        if not callable(getattr(base, "recognize", None)):
            raise TypeError("visual OCR engine must provide recognize")
        self.base = base
        self.modified_observations = 0
        self.dropped_rows = 0

    @property
    def total_calls(self) -> int:
        return getattr(self.base, "total_calls", 0)

    @property
    def retry_calls(self) -> int:
        return getattr(self.base, "retry_calls", 0)

    def recognize(self, frame: Frame) -> OCRObservation:
        # Filter every adaptive reading, not only its already-voted result.
        # Retain the original adaptive instance and its live call counters.
        from .adaptive import AdaptiveOCREngine
        if isinstance(self.base, AdaptiveOCREngine):
            return self.base.recognize_with_filter(frame, self._filter_observation)
        return self._filter_observation(self.base.recognize(frame))

    def _filter_observation(self, observation: OCRObservation) -> OCRObservation:
        if not isinstance(observation, OCRObservation):
            raise TypeError("invalid OCR observation")
        lines = observation.text.splitlines()
        if len(lines) <= 2:
            return observation

        omitted = len(lines) - 2
        metadata = dict(observation.metadata or {})
        detections = metadata.get("detections")
        retained_confidence = observation.confidence
        if isinstance(detections, list):
            # Strip dropped-row OCR strings from the frame metadata too; a
            # subsequent adaptive/diagnostic stage must not silently use
            # rejected upper words as if they were accepted dialogue.
            retained = [
                {**record, "line": record["line"] - omitted}
                for record in detections
                if isinstance(record, dict)
                and type(record.get("line")) is int
                and omitted <= record["line"] < len(lines)
            ]
            metadata["detections"] = retained
            # RapidOCR's original confidence averages ALL accepted boxes,
            # including discarded top-row credits. Do not let strong omitted
            # scene text certify weak retained dialogue. Only lower the final
            # confidence when both kept lines have valid score evidence.
            # Adaptive consensus may already be more conservative: never
            # raise its confidence using scores from just one observation.
            if (
                {record["line"] for record in retained} == {0, 1}
                and all(
                    type(record.get("score")) in (int, float)
                    and math.isfinite(record["score"])
                    and 0.0 <= record["score"] <= 1.0
                    for record in retained
                )
            ):
                kept_mean = sum(record["score"] for record in retained) / len(retained)
                if kept_mean < retained_confidence:
                    retained_confidence = kept_mean
                    metadata["experimental_bottom_two_confidence_bounded"] = True
        metadata["experimental_bottom_two_lines_only"] = True
        metadata["research_bottom_two_lines_only"] = True
        metadata["research_discarded_line_count"] = omitted
        self.modified_observations += 1
        self.dropped_rows += omitted
        return OCRObservation(
            "\n".join(lines[-2:]), retained_confidence,
            observation.pts, metadata,
        )
