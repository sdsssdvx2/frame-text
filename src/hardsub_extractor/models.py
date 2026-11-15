from __future__ import annotations
from dataclasses import dataclass
from typing import Any
import numpy as np

@dataclass(slots=True)
class Frame:
    pts: float
    image: np.ndarray

@dataclass(slots=True)
class OCRObservation:
    text: str
    confidence: float
    pts: float
    metadata: dict[str, Any] | None = None

@dataclass(slots=True)
class Cue:
    start: float
    end: float
    text: str
    confidence: float = 1.0
    def duration(self) -> float:
        return max(0.0, self.end - self.start)

@dataclass(slots=True)
class VisualSegment:
    start: float
    end: float
    samples: list[Frame]
    diagnostics: dict[str, Any] | None = None
