"""Optional, content-free progress and cooperative cancellation for extraction.

Only stage names, integer counts and media clock seconds cross this boundary:
never frame pixels, OCR text, URLs, browser state or output paths. When omitted,
the existing extraction/OCR path is untouched.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Callable, Iterable, Iterator

from .models import Frame, OCRObservation
from .ocr.base import OCREngine


class ExtractionCancelled(Exception):
    """A user-requested interruption, not a failed recognition."""


@dataclass(frozen=True, slots=True)
class ProgressEvent:
    stage: str
    completed: int = 0
    total: int | None = None
    media_seconds: float | None = None


ProgressCallback = Callable[[ProgressEvent], None]
CancelPredicate = Callable[[], bool]


class ExtractionMonitor:
    """Keep user-interaction decisions outside the actual OCR algorithm."""

    _STAGES = frozenset({
        "resolving", "capturing", "captured", "recognizing",
        "exporting", "syncing", "completed",
    })

    def __init__(
        self,
        on_progress: ProgressCallback | None = None,
        *,
        is_cancelled: CancelPredicate | None = None,
        frame_interval: int = 100,
        ocr_interval: int = 8,
        exclusive_outputs: bool = False,
    ) -> None:
        if frame_interval < 1 or ocr_interval < 1:
            raise ValueError("progress reporting intervals must be positive")
        self.on_progress = on_progress
        self.is_cancelled = is_cancelled
        self.frame_interval = frame_interval
        self.ocr_interval = ocr_interval
        self.exclusive_outputs = bool(exclusive_outputs)

    def check_cancelled(self) -> None:
        if self.is_cancelled is not None and self.is_cancelled():
            raise ExtractionCancelled("자막 추출을 취소했습니다.")

    def report(
        self,
        stage: str,
        completed: int = 0,
        total: int | None = None,
        media_seconds: float | None = None,
    ) -> None:
        self.check_cancelled()
        if stage not in self._STAGES:
            raise ValueError("invalid extraction progress stage")
        if (
            not isinstance(completed, int)
            or completed < 0
            or total is not None
            and (not isinstance(total, int) or total < completed)
            or media_seconds is not None
            and (not math.isfinite(media_seconds) or media_seconds < 0)
        ):
            raise ValueError("invalid extraction progress counters")
        if self.on_progress is not None:
            self.on_progress(ProgressEvent(stage, completed, total, media_seconds))

    def finish(self, cue_count: int) -> None:
        # Writing both subtitle files is the commit point. If a cancel request
        # arrives during those synchronous writes, completion wins rather than
        # falsely reporting that output files were not created.
        if self.on_progress is not None:
            self.on_progress(ProgressEvent("completed", cue_count, cue_count))

    def frames(
        self, source: Iterable[Frame], *, report_progress: bool = True,
    ) -> Iterator[Frame]:
        """Pass through exact frame objects; close upstream on early exit."""
        iterator = iter(source)
        count = 0
        last_pts: float | None = None
        try:
            if report_progress:
                self.report("capturing")
            for frame in iterator:
                self.check_cancelled()
                count += 1
                last_pts = float(frame.pts)
                if report_progress and (count == 1 or count % self.frame_interval == 0):
                    self.report("capturing", count, media_seconds=last_pts)
                yield frame
            if report_progress:
                self.report("captured", count, media_seconds=last_pts)
        finally:
            close = getattr(iterator, "close", None)
            if callable(close):
                close()

    def ocr(self, engine: OCREngine, *, total: int | None) -> OCREngine:
        self.report("recognizing", 0, total)
        return _MonitoredOCR(engine, self, total)


class _MonitoredOCR(OCREngine):
    """Preserve OCR results exactly; only count completed observations."""

    def __init__(self, inner: OCREngine, monitor: ExtractionMonitor, total: int | None):
        self.inner = inner
        self.monitor = monitor
        self.total = total
        self.count = 0

    def recognize(self, frame: Frame) -> OCRObservation:
        self.monitor.check_cancelled()
        observation = self.inner.recognize(frame)
        self.monitor.check_cancelled()
        self.count += 1
        if self.count % self.monitor.ocr_interval == 0 or (self.total is not None and self.count == self.total):
            self.monitor.report(
                "recognizing", self.count, self.total,
                media_seconds=(frame.pts if self.total is None else None),
            )
        return observation
