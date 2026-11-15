from __future__ import annotations

from dataclasses import dataclass
from typing import Callable
import math
import cv2
import numpy as np

from .base import OCREngine, vote_observations, normalize_text
from ..models import Frame, OCRObservation
from ..extraction_progress import ExtractionCancelled


@dataclass(slots=True)
class AdaptiveOCRConfig:
    retry_below_confidence: float = 0.72
    upscale: float = 1.75
    enable_clahe_retry: bool = True
    retry_below_box_confidence: float = 0.90


def _upscale(image: np.ndarray, scale: float) -> np.ndarray:
    if scale <= 1.0:
        return image
    return cv2.resize(
        image,
        None,
        fx=scale,
        fy=scale,
        interpolation=cv2.INTER_CUBIC,
    )


def _clahe_bgr(image: np.ndarray) -> np.ndarray:
    # Preserve chroma while improving local luminance contrast. This is less
    # destructive than hard binarisation for coloured/outlined anime subtitles.
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l2 = clahe.apply(l)
    return cv2.cvtColor(cv2.merge((l2, a, b)), cv2.COLOR_LAB2BGR)


class AdaptiveOCREngine(OCREngine):
    """
    Low-cost first pass + selective retries.

    Normal high-confidence frames run OCR once. Empty/low-confidence results
    receive an upscale retry and, optionally, a CLAHE+upscale retry. The
    observed strings are combined using the same confidence-weighted medoid
    voting used across neighbouring frames.
    """

    def __init__(
        self,
        base: OCREngine,
        config: AdaptiveOCRConfig | None = None,
    ):
        self.base = base
        self.config = config or AdaptiveOCRConfig()
        self.total_calls = 0
        self.retry_calls = 0

    def _call(self, frame: Frame) -> OCRObservation:
        self.total_calls += 1
        return self.base.recognize(frame)

    def recognize(self, frame: Frame) -> OCRObservation:
        # Production/default OCR retains its original unfiltered decision.
        return self._recognize(frame, transform=None)

    def recognize_with_filter(
        self,
        frame: Frame,
        transform: Callable[[OCRObservation], OCRObservation],
    ) -> OCRObservation:
        """Explicit opt-in: filter each model reading before retry and voting.

        Ignore dropped credits when judging weak text; never use more than the
        existing primary + two bounded optional model calls.
        """
        if not callable(transform):
            raise TypeError("observation filter must be callable")
        return self._recognize(frame, transform=transform)

    def _recognize(
        self,
        frame: Frame,
        *,
        transform: Callable[[OCRObservation], OCRObservation] | None,
    ) -> OCRObservation:
        def observed(sample: Frame) -> OCRObservation:
            result = self._call(sample)
            if transform is not None:
                result = transform(result)
                if not isinstance(result, OCRObservation):
                    raise TypeError("observation filter returned invalid OCR result")
            return result

        first = observed(frame)
        # A sentence average can hide one weakly recognized word among several
        # near-perfect boxes. Inspect accepted boxes before deciding to skip
        # enhancement; engines without geometry retain the existing fallback.
        weak_box = False
        for detection in (first.metadata or {}).get("detections", []):
            score = detection.get("score")
            if isinstance(score, (int, float)) and math.isfinite(score):
                weak_box |= score < self.config.retry_below_box_confidence
        if (first.text.strip() and first.confidence >= self.config.retry_below_confidence
                and not weak_box):
            return first

        observations = [first]
        optional_retry_failures = 0

        # Optional enhancement can fail on a particular frame even after
        # primary inference returned a useful reading. Keep that reading
        # instead of aborting the whole video, but do not hide the failure.
        # Primary inference errors and cancellation always propagate.
        if self.config.upscale > 1.0:
            self.retry_calls += 1
            try:
                observations.append(observed(Frame(
                    pts=frame.pts,
                    image=_upscale(frame.image, self.config.upscale),
                )))
            except ExtractionCancelled:
                raise
            except Exception:
                optional_retry_failures += 1

        if self.config.enable_clahe_retry:
            self.retry_calls += 1
            try:
                enhanced = _clahe_bgr(frame.image)
                enhanced = _upscale(enhanced, self.config.upscale)
                observations.append(observed(
                    Frame(pts=frame.pts, image=enhanced),
                ))
            except ExtractionCancelled:
                raise
            except Exception:
                optional_retry_failures += 1

        result = vote_observations(observations)
        if transform is not None and result.text:
            # A medoid vote can tie on the same text even when its first
            # observation has much weaker OCR geometry. In this experimental
            # filter path, retain the most confident *exact-text* box evidence
            # for subsequent pixel timing, without inflating the deliberately
            # conservative consensus confidence or changing the chosen text.
            matching_boxes = [
                observation for observation in observations
                if normalize_text(observation.text) == result.text
                and isinstance(observation.metadata, dict)
                and isinstance(observation.metadata.get("detections"), list)
                and observation.metadata["detections"]
            ]
            if matching_boxes:
                strongest = max(matching_boxes, key=lambda obs: obs.confidence)
                if strongest.metadata is not result.metadata:
                    result = OCRObservation(
                        result.text, result.confidence, result.pts,
                        strongest.metadata,
                    )
        if optional_retry_failures:
            # Record the number, not any private frame data or error strings.
            metadata = dict(result.metadata or {})
            metadata["optional_ocr_retry_failures"] = optional_retry_failures
            return OCRObservation(
                result.text, result.confidence, result.pts, metadata,
            )
        return result

