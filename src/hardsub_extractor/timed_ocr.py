"""Bounded-memory OCR from independently detected subtitle time intervals.

Research-only optional transport-independent building block for an alternative
temporal detector. It takes TIMINGS, not raw output folders or a third-party
executable. Never downloads, invokes, bundles or trusts native binaries, and
never reads supplied subtitle transcripts into the recognition process.

The old research held decoded frames for all intervals and capped the study
at eight cues. This reader only retains up to five cropped frames per cue,
and can seek across long subtitle-free stretches in bounded decode windows.
Source/video read-only; SRT and ASS are published exclusively together.
"""
from __future__ import annotations

from contextlib import closing
from dataclasses import dataclass
from fractions import Fraction
import math
from statistics import median
from pathlib import Path
from typing import Callable, Iterable

from .frame_source import FFmpegPTSFrameSource, NormalizedROI
from .models import Cue, Frame, OCRObservation
from .ocr.base import (
    OCREngine, vote_observations, comparison_key, text_similarity,
)
from .writers import write_subtitle_pair_exclusive

# Default fractions are chosen before seeing any reference or model output.
_SAMPLE_FRACTIONS = (0.15, 0.325, 0.50, 0.675, 0.85)
_MAX_INTERVALS = 3000
_MAX_TIMELINE_SECONDS = 21600.0
_MAX_NEAREST_DISTANCE = 0.17
# Spawn a new decoder only when a large blank gap justifies its overhead.
_SEEK_GAP_SECONDS = 30.0
_SEEK_LEAD_SECONDS = 30.0
_SEEK_PADDING_SECONDS = 1.0
_MAX_SEEK_WINDOWS = 64
_MAX_REVIEW_WINDOWS = 8
# Recover only native intervals that received ZERO frames on the 8fps grid.
# At most sixteen bounded local decodes per video, which may each cover
# several nearby native subtitle intervals. Individual OCR keeps <=5 frames.
_MAX_SHORT_FRAME_RESCUES = 16
_SHORT_FRAME_RESCUE_FPS = 24.0
_SHORT_RESCUE_MAX_GAP = 2.0
_SHORT_RESCUE_MAX_SPAN = 24.0


def format_missed_windows(windows: tuple[tuple[float, float], ...]) -> str:
    """Format only source timestamps, never OCR text or video paths."""
    def stamp(value: float) -> str:
        # Round milliseconds BEFORE splitting into clock fields; otherwise
        # 59.9996s could be printed as the invalid 00:00:60.000.
        milliseconds = max(0, round(value * 1000))
        hours, remainder = divmod(milliseconds, 3_600_000)
        minutes, remainder = divmod(remainder, 60_000)
        seconds, milliseconds = divmod(remainder, 1000)
        return f"{hours:02}:{minutes:02}:{seconds:02}.{milliseconds:03}"

    return ", ".join(
        f"{stamp(start)}–{stamp(end)}"
        for start, end in windows[:_MAX_REVIEW_WINDOWS]
    )


@dataclass(frozen=True, slots=True)
class TimedOCRReport:
    detected_intervals: int
    recognized_intervals: int
    missed_intervals: int
    ocr_calls: int
    maximum_retained_frames: int
    srt_path: Path
    ass_path: Path
    # Review-only: first eight native detected intervals with no OCR text.
    missed_windows: tuple[tuple[float, float], ...] = ()
    exported_cues: int = 0
    split_source_intervals: int = 0
    # First output cue number of each recovered pair (1-based, up to 8).
    # UI can direct review without exposing OCR text or source paths.
    split_cue_numbers: tuple[int, ...] = ()
    short_frame_rescue_attempted: int = 0
    short_frame_rescue_recognized: int = 0
    short_frame_rescue_deferred: int = 0
    short_frame_rescue_errors: int = 0


@dataclass(frozen=True, slots=True)
class _Interval:
    start: float
    end: float



def _stable_cue_pair(
    readings: list[tuple[float, OCRObservation]], interval: _Interval,
) -> tuple[Cue, Cue] | None:
    """Opt-in evidence-only temporal split: two stable observations per side.

    No transcript source, model retry, file access or image is introduced.
    Do not split without reliable recognition and aligned OCR box geometry.
    """
    if len(readings) not in (4, 5):
        return None
    centers: list[tuple[float, float]] = []
    for pts, observation in readings:
        if (
            not math.isfinite(pts)
            or not observation.text.strip()
            or not math.isfinite(observation.confidence)
            or not 0.88 <= observation.confidence <= 1.0
        ):
            return None
        locations = []
        for detection in (observation.metadata or {}).get("detections", ()):
            if not isinstance(detection, dict):
                continue
            x, y = detection.get("cx"), detection.get("cy")
            if (
                type(x) in (int, float) and type(y) in (int, float)
                and math.isfinite(x) and math.isfinite(y)
                and 0 <= x <= 1 and 0 <= y <= 1
            ):
                locations.append((x, y))
        if not locations:
            return None
        centers.append((
            median(x for x, _ in locations),
            median(y for _, y in locations),
        ))
    if (
        max(x for x, _ in centers) - min(x for x, _ in centers) > .18
        or max(y for _, y in centers) - min(y for _, y in centers) > .12
    ):
        return None
    candidates: list[tuple[Cue, Cue]] = []
    for index in range(2, len(readings) - 1):
        before, after = readings[:index], readings[index:]
        left = vote_observations([obs for _, obs in before])
        right = vote_observations([obs for _, obs in after])
        if (
            len(comparison_key(left.text)) < 4
            or len(comparison_key(right.text)) < 4
            or text_similarity(left.text, right.text) > .46
            or any(text_similarity(obs.text, left.text) < .86 for _, obs in before)
            or any(text_similarity(obs.text, right.text) < .86 for _, obs in after)
        ):
            continue
        gap = after[0][0] - before[-1][0]
        boundary = (before[-1][0] + after[0][0]) / 2
        if (
            not 0 < gap <= .75
            or boundary - interval.start < .4
            or interval.end - boundary < .4
        ):
            continue
        candidates.append((
            Cue(interval.start, boundary, left.text, left.confidence),
            Cue(boundary, interval.end, right.text, right.confidence),
        ))
    return candidates[0] if len(candidates) == 1 else None


def _validate_intervals(items: Iterable[object]) -> list[_Interval]:
    """Fail closed before FFmpeg/model initialization or result publication."""
    answer: list[_Interval] = []
    for item in items:
        if len(answer) >= _MAX_INTERVALS:
            raise ValueError("too many external timing intervals")
        start, end = getattr(item, "start", None), getattr(item, "end", None)
        if (
            type(start) not in (float, int)
            or type(end) not in (float, int)
            or not math.isfinite(start) or not math.isfinite(end)
            or not 0 <= start < end <= _MAX_TIMELINE_SECONDS
            or (answer and start < answer[-1].end)
        ):
            raise ValueError("invalid, overlapping or unordered timed interval")
        answer.append(_Interval(float(start), float(end)))
    if not answer:
        raise ValueError("empty external timing interval list")
    return answer



def _decoder_windows(
    spans: list[_Interval], *, sample_fps: float = 8.0,
) -> tuple[tuple[float, float], ...]:
    """Group nearby cues; merge cheapest gaps if seek windows exceed budget.

    A one-second margin protects nearest-frame voting at seek boundaries.
    The seek start MUST lie on the source-wide output-fps grid: FFmpeg's
    fps filter restarts its time origin at every input -ss, so unaligned
    seeking selects different pixels than full-length decoding.

    For fractional output rates, a source seek that is merely on that grid
    may still select a different underlying frame near a rounding tie.
    Use integral-second seeks that are also exact rational-fps boundaries.
    If the rate has no bounded stable rational grid, use one decoder from 0
    instead of repeatedly decoding overlapping windows or mixing pixels.
    Returned pairs are (absolute source seek, bounded decode duration).
    """
    groups: list[tuple[float, float]] = []
    first, last = spans[0].start, spans[0].end
    for span in spans[1:]:
        if span.start - last > _SEEK_GAP_SECONDS:
            groups.append((first, last))
            first = span.start
        last = span.end
    groups.append((first, last))
    if len(groups) > _MAX_SEEK_WINDOWS:
        # Previously we collapsed the entire season-length span into ONE
        # decoder once the disjoint group count reached 65. On sparse
        # episodes that re-decoded almost every subtitle-free frame.
        #
        # Each removed boundary between adjacent groups costs exactly the
        # blank-time gap between them. Keep the 64-window process cap while
        # merging only the cheapest gaps, with stable early-boundary ties.
        # This minimizes total decoded time for the fixed window budget and
        # keeps the original sparse source PTS grid and cue ordering.
        merges_needed = len(groups) - _MAX_SEEK_WINDOWS
        cheapest = sorted(
            (groups[i + 1][0] - groups[i][1], i)
            for i in range(len(groups) - 1)
        )[:merges_needed]
        merged_boundaries = {index for _, index in cheapest}
        compact: list[tuple[float, float]] = []
        first, last = groups[0]
        for boundary, (next_first, next_last) in enumerate(groups[1:]):
            if boundary in merged_boundaries:
                last = next_last
            else:
                compact.append((first, last))
                first, last = next_first, next_last
        compact.append((first, last))
        groups = compact

    fractional = sample_fps != math.floor(sample_fps)
    seek_step = 0
    if fractional:
        # E.g. 7.8fps = 39/5: seek at 40s, not 40.256410...s.
        # Both the decoded source and the fps filter then start on a
        # stable whole-second / whole-output-frame boundary.
        rational = Fraction(str(sample_fps)).limit_denominator(1000)
        if (abs(float(rational) - sample_fps) > 1e-10
                or rational.denominator > 30):
            stop = min(_MAX_TIMELINE_SECONDS,
                       spans[-1].end + _SEEK_PADDING_SECONDS)
            return ((0.0, stop),)
        seek_step = rational.denominator

    windows = []
    for first, last in groups:
        requested = max(0.0, first - _SEEK_PADDING_SECONDS)
        # Floor, never round up: preserve the leading sample margin.
        if fractional:
            seek = float(math.floor(requested / seek_step) * seek_step)
        else:
            # Preserve exact current default behavior for integer rates.
            seek = math.floor(requested * sample_fps) / sample_fps
        if seek < _SEEK_LEAD_SECONDS:
            seek = 0.0
        stop = min(_MAX_TIMELINE_SECONDS, last + _SEEK_PADDING_SECONDS)
        windows.append((seek, stop - seek))
    return tuple(windows)


def _group_short_rescues(
    candidates: list[_Interval],
) -> tuple[tuple[_Interval, ...], ...]:
    """Pack only nearby missing short intervals into bounded reread windows.

    Keep every cue separate for OCR voting and final SRT identity. Groups
    cap decoder wall time and memory; 16 groups are NOT 16 total subtitles.
    """
    groups: list[tuple[_Interval, ...]] = []
    current: list[_Interval] = []
    for candidate in candidates:
        if current and (
            candidate.start - current[-1].end > _SHORT_RESCUE_MAX_GAP
            or candidate.end - current[0].start > _SHORT_RESCUE_MAX_SPAN
        ):
            groups.append(tuple(current))
            current = []
        current.append(candidate)
    if current:
        groups.append(tuple(current))
    return tuple(groups)


def _interval_frames(
    source: str,
    spans: list[_Interval],
    *,
    roi: NormalizedROI | None,
    sample_fps: float,
    check_cancelled: Callable[[], None] | None,
) -> Iterable[Frame]:
    """Restore seek-local FFmpeg showinfo PTS to the original media timeline.

    Each decoder is closed on cancellation or error; the caller remains
    responsible for flushing every input timing, even beyond video EOF.
    """
    for seek, duration in _decoder_windows(spans, sample_fps=sample_fps):
        if check_cancelled:
            check_cancelled()
        frames = iter(FFmpegPTSFrameSource(
            source, roi=roi, sample_fps=sample_fps,
            start_seconds=seek, max_duration_seconds=duration,
        ))
        with closing(frames):
            for frame in frames:
                yield Frame(frame.pts + seek, frame.image) if seek else frame


def extract_timed_ocr(
    source: str,
    intervals: Iterable[object],
    ocr: OCREngine,
    out_base: str | Path,
    *,
    sample_fps: float = 8.0,
    roi: NormalizedROI | None = None,
    check_cancelled: Callable[[], None] | None = None,
    split_stable_text: bool = False,
) -> TimedOCRReport:
    """Bounded decode windows; at most 5 cropped images retained at a time.

    A detected interval with zero successful text observations is counted as
    missed, not converted to guessed dialogue. Empty results raise an error
    and never create a misleading SRT; existing SRT/ASS are not replaced.
    """
    if type(split_stable_text) is not bool:
        raise ValueError("native subtitle split opt-in must be boolean")
    if (
        type(sample_fps) not in (float, int)
        or not math.isfinite(sample_fps)
        or not 1.0 <= sample_fps <= 12.0
    ):
        raise ValueError("invalid timed OCR sampling rate")
    spans = _validate_intervals(intervals)
    base = Path(out_base)
    if any(
        p.exists() or p.is_symlink()
        for p in (base.with_suffix(".srt"), base.with_suffix(".ass"))
    ):
        raise FileExistsError("existing subtitle result cannot be overwritten")
    if check_cancelled:
        check_cancelled()

    recognized: list[Cue] = []
    source_recognized = 0
    split_source_intervals = 0
    split_cue_numbers: list[int] = []
    missed_windows: list[tuple[float, float]] = []
    short_frame_candidates: list[_Interval] = []
    calls = 0
    max_retained = 0
    slot = 0
    targets = tuple(
        spans[0].start + (spans[0].end - spans[0].start) * x
        for x in _SAMPLE_FRACTIONS
    )
    nearest: list[Frame | None] = [None] * len(targets)
    distances = [float("inf")] * len(targets)
    previous_pts = -1.0

    def flush() -> None:
        nonlocal calls, nearest, source_recognized, split_source_intervals
        current = spans[slot]
        # The same single video frame may be nearest to multiple fractions
        # for a short cue; it must never be counted or OCR'd repeatedly.
        unique = {frame.pts: frame for frame in nearest if frame is not None}
        readings: list[tuple[float, OCRObservation]] = []
        for frame in sorted(unique.values(), key=lambda f: f.pts):
            if check_cancelled:
                check_cancelled()
            readings.append((frame.pts, ocr.recognize(frame)))
            calls += 1
        voted = vote_observations([observation for _, observation in readings])
        if voted.text:
            source_recognized += 1
            pair = _stable_cue_pair(readings, current) if split_stable_text else None
            if pair is not None:
                if len(split_cue_numbers) < _MAX_REVIEW_WINDOWS:
                    split_cue_numbers.append(len(recognized) + 1)
                recognized.extend(pair)
                split_source_intervals += 1
            else:
                recognized.append(Cue(
                    current.start, current.end, voted.text, voted.confidence,
                ))
        elif len(missed_windows) < _MAX_REVIEW_WINDOWS:
            # This may be a native detector false positive, not missing dialogue.
            missed_windows.append((current.start, current.end))
        # Native timing can outlive the actual visible glyphs: a single
        # 8fps frame near the fading edge can produce EMPTY OCR even though
        # another frame inside a <250ms caption contains real text.
        # Revisit only missed short spans with zero sample (<1/fps) or
        # ONE blank sample (<2/fps). Never retry a recognized sentence.
        # The existing 16-group cap bounds CPU and extra model calls.
        duration = current.end - current.start
        if not voted.text and (
            (not unique and duration < 1.0 / sample_fps)
            or (len(unique) == 1 and duration < 2.0 / sample_fps)
        ):
            short_frame_candidates.append(current)
        # Drop the previous interval's entire pixel image collection before
        # moving into the next interval, independent of video length.
        nearest = []

    frame_stream = iter(_interval_frames(
        str(source), spans, roi=roi, sample_fps=float(sample_fps),
        check_cancelled=check_cancelled,
    ))
    with closing(frame_stream):
        for frame in frame_stream:
            if check_cancelled:
                check_cancelled()
            if not math.isfinite(frame.pts) or frame.pts < previous_pts:
                raise RuntimeError("video PTS is not finite and monotonic")
            previous_pts = frame.pts

            while slot < len(spans) and frame.pts >= spans[slot].end:
                flush()
                slot += 1
                if slot < len(spans):
                    targets = tuple(
                        spans[slot].start + (spans[slot].end - spans[slot].start) * x
                        for x in _SAMPLE_FRACTIONS
                    )
                    nearest = [None] * len(targets)
                    distances = [float("inf")] * len(targets)
            if slot >= len(spans):
                break

            span = spans[slot]
            if span.start <= frame.pts < span.end:
                for index, target in enumerate(targets):
                    gap = abs(frame.pts - target)
                    if gap <= _MAX_NEAREST_DISTANCE and gap < distances[index]:
                        nearest[index] = frame
                        distances[index] = gap
                max_retained = max(
                    max_retained, len({f.pts for f in nearest if f is not None}),
                )

    # This final flush also reports intervals past the input video duration
    # as missing, rather than inventing caption content.
    while slot < len(spans):
        if check_cancelled:
            check_cancelled()
        flush()
        slot += 1
        if slot < len(spans):
            nearest = [None] * len(_SAMPLE_FRACTIONS)

    short_groups = _group_short_rescues(short_frame_candidates)
    short_attempted = 0
    short_recovered = 0
    short_errors = 0
    for group in short_groups[:_MAX_SHORT_FRAME_RESCUES]:
        if check_cancelled:
            check_cancelled()
        # Count native intervals attempted, not decoder processes. One local
        # 24fps window can recover many nearby 80ms captions.
        short_attempted += len(group)
        # Whole-second seek is aligned to the 24fps frame grid. Group length
        # is limited to 24 seconds before its small decoder margins.
        seek = float(max(0, math.floor(group[0].start - 0.5)))
        stop = min(_MAX_TIMELINE_SECONDS, group[-1].end + 0.5)
        current_slot = 0
        candidates: list[Frame] = []

        def finish_short() -> None:
            nonlocal candidates, calls, source_recognized, short_recovered
            current = group[current_slot]
            if candidates:
                observations: list[OCRObservation] = []
                for sample in candidates:
                    if check_cancelled:
                        check_cancelled()
                    observations.append(ocr.recognize(sample))
                    calls += 1
                winner = vote_observations(observations)
                if winner.text:
                    recognized.append(Cue(
                        current.start, current.end, winner.text, winner.confidence,
                    ))
                    source_recognized += 1
                    short_recovered += 1
            # Never retain previous cue images while examining another cue.
            candidates = []

        try:
            short_stream = iter(FFmpegPTSFrameSource(
                str(source), roi=roi, sample_fps=_SHORT_FRAME_RESCUE_FPS,
                start_seconds=seek, max_duration_seconds=stop - seek,
            ))
            with closing(short_stream):
                previous_short_pts = -1.0
                for frame in short_stream:
                    if check_cancelled:
                        check_cancelled()
                    absolute = frame.pts + seek
                    if not math.isfinite(absolute) or absolute < previous_short_pts:
                        raise RuntimeError("invalid short interval decoder PTS")
                    previous_short_pts = absolute
                    while (
                        current_slot < len(group)
                        and absolute >= group[current_slot].end
                    ):
                        finish_short()
                        current_slot += 1
                    if current_slot == len(group):
                        break
                    current = group[current_slot]
                    if current.start <= absolute < current.end and len(candidates) < 5:
                        candidates.append(Frame(absolute, frame.image))
                        max_retained = max(max_retained, len(candidates))
                while current_slot < len(group):
                    if check_cancelled:
                        check_cancelled()
                    finish_short()
                    current_slot += 1
        except (OSError, RuntimeError, ValueError):
            # Preserve any earlier successful cues, including successful
            # samples in THIS window; only unprocessed members are errors.
            # Cooperative owner cancellation propagates unchanged.
            if check_cancelled:
                check_cancelled()
            short_errors += len(group) - current_slot

    # Recovery is late and may precede already extracted cues on the source
    # timeline. Resort and derive review identities from final SRT positions.
    split_starts = {
        recognized[index - 1].start for index in split_cue_numbers
    }
    recognized.sort(key=lambda cue: (cue.start, cue.end))
    split_cue_numbers = [
        index for index, cue in enumerate(recognized, 1)
        if cue.start in split_starts
    ]
    recognized_starts = {cue.start for cue in recognized}
    missed_windows = [
        (span.start, span.end) for span in spans
        if span.start not in recognized_starts
    ][:_MAX_REVIEW_WINDOWS]

    if not recognized:
        raise RuntimeError("no recognized timed subtitles")
    if check_cancelled:
        check_cancelled()
    srt, ass = write_subtitle_pair_exclusive(recognized, base)
    return TimedOCRReport(
        detected_intervals=len(spans),
        recognized_intervals=source_recognized,
        missed_intervals=len(spans) - source_recognized,
        ocr_calls=calls,
        maximum_retained_frames=max_retained,
        srt_path=srt,
        ass_path=ass,
        missed_windows=tuple(missed_windows),
        exported_cues=len(recognized),
        split_source_intervals=split_source_intervals,
        split_cue_numbers=tuple(split_cue_numbers),
        short_frame_rescue_attempted=short_attempted,
        short_frame_rescue_recognized=short_recovered,
        short_frame_rescue_deferred=sum(
            len(group) for group in short_groups[_MAX_SHORT_FRAME_RESCUES:]
        ),
        short_frame_rescue_errors=short_errors,
    )
