"""One explicit, read-only original-video OCR preview BEFORE full extraction.

No browser, external network, cached frame, subtitle output or user file change.
A single local HSE ffmpeg frame and the same selected-language OCR engine are
used. This is a manual visual sample, NOT a whole-video accuracy assertion.
"""
from __future__ import annotations

from contextlib import closing
from dataclasses import dataclass
import math
from pathlib import Path
from typing import Callable

import numpy as np

from .frame_source import FFmpegPTSFrameSource, NormalizedROI, probe_video
from .models import Frame, OCRObservation

_VIDEO_EXTENSIONS = frozenset({".mkv", ".mp4", ".webm", ".mov", ".avi", ".m4v", ".ts"})
_MAX_VIDEO_BYTES = 100 * 1024 ** 3
_MAX_PREVIEW_SECOND = 6 * 3600


@dataclass(slots=True)
class LocalOCRPreview:
    image: np.ndarray
    requested_second: float
    actual_second: float
    text: str
    confidence: float
    preview_mode: str = "default"
    region_mode: str = "bottom40"
    comparison_text: str | None = None
    comparison_confidence: float | None = None


def parse_preview_second(raw: str) -> float:
    """Parse SS, MM:SS or HH:MM:SS; bounded and no localized ambiguity."""
    if not isinstance(raw, str) or not 1 <= len(raw) <= 20:
        raise ValueError("invalid preview time")
    fields = raw.strip().split(":")
    if len(fields) not in (1, 2, 3):
        raise ValueError("invalid preview clock")
    if not all(x and x.replace(".", "", 1).isdigit() for x in fields):
        raise ValueError("invalid preview clock")
    if any("." in x for x in fields[:-1]):
        raise ValueError("fractional clock field not last")
    parts = [float(x) for x in fields]
    if not all(math.isfinite(x) for x in parts):
        raise ValueError("invalid preview number")
    if len(parts) > 1 and (parts[-1] >= 60 or (len(parts) == 3 and parts[1] >= 60)):
        raise ValueError("invalid preview minute or second")
    value = sum(x * 60 ** i for i, x in enumerate(reversed(parts)))
    if value < 0 or value > _MAX_PREVIEW_SECOND:
        raise ValueError("preview outside six-hour limit")
    return value


def inspect_local_ocr_preview(
    video: str | Path,
    second: float,
    make_ocr: Callable[[], object],
    *,
    check_cancelled: Callable[[], None] | None = None,
    cancel_requested: Callable[[], bool] | None = None,
    preview_mode: str = "default",
    region_mode: str = "bottom40",
    compare_make_ocr: Callable[[], object] | None = None,
) -> LocalOCRPreview:
    """One bounded local decode, one chosen-style OCR, no output or retained video.

    The factory owns the selected style; this marker only labels the result.
    Never apply a different post-hoc filter to an already voted result.
    """
    if preview_mode not in {
        "default", "bottom-two-lines", "light-caption", "caption-islands",
        "compare-bright",
    }:
        raise ValueError("unsupported local OCR preview mode")
    if (preview_mode == "compare-bright") != (compare_make_ocr is not None):
        raise ValueError("comparison factory requires explicit matching mode")
    if compare_make_ocr is not None and not callable(compare_make_ocr):
        raise TypeError("comparison OCR factory must be callable")
    from .tab_capture import roi_for_region
    roi = roi_for_region(region_mode)
    path = Path(video).expanduser()
    if (
        path.is_symlink() or not path.is_file()
        or path.suffix.lower() not in _VIDEO_EXTENSIONS
        or not 0 < path.stat().st_size <= _MAX_VIDEO_BYTES
    ):
        raise ValueError("invalid local preview media")
    if type(second) not in (float, int) or not math.isfinite(second) or not 0 <= second <= _MAX_PREVIEW_SECOND:
        raise ValueError("invalid requested preview second")
    if check_cancelled:
        check_cancelled()
    path = path.resolve(strict=True)
    # Do not follow remote media references or network protocols hidden
    # inside an otherwise local container. FFmpeg's file/pipe-only input
    # protocol fence applies independently to the metadata probe and decode.
    local_protocols = ["-protocol_whitelist", "file,pipe"]
    info = probe_video(
        str(path), extra_input_args=local_protocols, timeout_seconds=25,
    )
    if (
        info.width < 2 or info.height < 2
        or info.width * info.height > 20_000_000
        or info.duration is not None and (
            not math.isfinite(info.duration) or second >= info.duration
        )
    ):
        raise ValueError("invalid preview video dimensions or duration")
    # Reuse the metadata from the bounded FFprobe, avoiding a second,
    # previously unbounded metadata probe inside the FFmpeg frame iterator.
    # In particular, only ONE ffprobe process runs for a single preview.
    source_stat = path.stat()
    seek = float(math.floor(second))
    duration = second - seek + 0.5
    best: Frame | None = None
    best_distance = float("inf")
    stream = iter(FFmpegPTSFrameSource(
        str(path), roi=roi, sample_fps=8.0,
        start_seconds=seek, max_duration_seconds=duration,
        extra_input_args=local_protocols,
        preprobed_info=info,
        decode_timeout_seconds=30.0,
        cancel_requested=cancel_requested,
    ))
    with closing(stream):
        for index, item in enumerate(stream):
            if check_cancelled:
                check_cancelled()
            if index >= 16:
                break
            actual = seek + item.pts
            if not math.isfinite(actual) or actual < seek - 1e-6:
                raise ValueError("invalid preview frame timestamp")
            gap = abs(actual - second)
            if gap < best_distance:
                best = item
                best_distance = gap
            if actual > second + 0.125:
                break
    # If the owner cancelled immediately before an empty decoder EOF,
    # preserve cancellation rather than reporting a misleading media error.
    if check_cancelled:
        check_cancelled()
    if best is None:
        raise RuntimeError("no preview image decoded")
    # Refuse to infer from a frame if the owner replaced or changed the
    # video during the probe/decode handoff; never write or repair the video.
    current_stat = path.stat()
    if (current_stat.st_size != source_stat.st_size
            or current_stat.st_mtime_ns != source_stat.st_mtime_ns
            or current_stat.st_ino != source_stat.st_ino):
        raise RuntimeError("local preview source changed during read")
    actual_second = best.pts + seek
    # Exactly one bounded FFmpeg decode. Both local OCR alternatives see
    # independent copies of the SAME pixels and never mutate the source.
    result = make_ocr().recognize(Frame(actual_second, best.image.copy()))
    if check_cancelled:
        check_cancelled()
    if not isinstance(result, OCRObservation):
        raise TypeError("invalid preview OCR result")
    alternative = None
    if compare_make_ocr is not None:
        alternative = compare_make_ocr().recognize(
            Frame(actual_second, best.image.copy()),
        )
        if check_cancelled:
            check_cancelled()
        if not isinstance(alternative, OCRObservation):
            raise TypeError("invalid comparison OCR result")
    # Both readings stay in RAM; neither appears in ordinary logs or files.
    return LocalOCRPreview(
        best.image.copy(), float(second), float(actual_second),
        result.text, float(result.confidence), preview_mode, region_mode,
        alternative.text if alternative is not None else None,
        float(alternative.confidence) if alternative is not None else None,
    )
