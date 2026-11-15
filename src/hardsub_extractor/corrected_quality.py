"""Offline cue-level comparison of HSE output with a separately edited SRT.

The edited SRT is NOT certified ground truth. Only numeric quality evidence
and cue numbers are returned; text and user paths never enter the report.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
import re
import unicodedata

from .models import Cue
from .subtitle_formats import decode_subtitle_bytes, parse_subtitle_bytes


class CorrectedQualityError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class QualityComparison:
    baseline_cues: int
    corrected_cues: int
    aligned_cues: int
    corrected_only: int
    baseline_only: int
    exact_visible_cues: int
    exact_physical_lines: int
    precise_250ms_cues: int
    strict_joint_cues: int
    reference_characters: int
    character_errors: int
    character_error_rate: float
    max_start_drift_seconds: float
    max_end_drift_seconds: float
    first_review_cue_numbers: tuple[int, ...]
    provenance: str = "OWNER_EDITED_NOT_INDEPENDENTLY_VERIFIED"


def _text(value: str) -> str:
    return unicodedata.normalize(
        "NFC", value.replace("\r\n", "\n").replace("\r", "\n")
    )


def _validate(cues: list[Cue]) -> None:
    if not 1 <= len(cues) <= 10_000:
        raise CorrectedQualityError("subtitle count outside supported range")
    last_start, total = -1.0, 0
    for cue in cues:
        if (
            not math.isfinite(cue.start) or not math.isfinite(cue.end)
            or not 0 <= cue.start < cue.end <= 6 * 3600
            or cue.start < last_start or not isinstance(cue.text, str)
        ):
            raise CorrectedQualityError("invalid or unsorted subtitle timeline")
        content = _text(cue.text)
        if not content.strip() or len(content) > 1024:
            raise CorrectedQualityError("invalid subtitle text length")
        total += len(content)
        last_start = cue.start
    if total > 1_000_000:
        raise CorrectedQualityError("subtitle transcript exceeds bounded comparison")


def _physical_line_break_positions(text: str) -> tuple[int, ...]:
    """Visible-character offsets of line breaks, not merely line count.

    Equal numbers of newline characters can still represent different
    subtitle layouts. Compare ordered boundaries relative to NFC text
    without newlines; do not use OCR words to align separate cues.
    """
    offsets: list[int] = []
    visible_chars = 0
    for ch in text:
        if ch == "\n":
            offsets.append(visible_chars)
        else:
            visible_chars += 1
    return tuple(offsets)


def _pair_cost(a: Cue, b: Cue) -> float | None:
    """Match using time evidence, never agreement with the corrected words."""
    ds, de = abs(a.start - b.start), abs(a.end - b.end)
    overlap = max(0.0, min(a.end, b.end) - max(a.start, b.start))
    smaller = min(a.end - a.start, b.end - b.start)
    if not ((ds <= 2.0 and de <= 2.0) or overlap >= .25 * smaller):
        return None
    shared = min(1.0, overlap / max(smaller, 1e-9))
    return .02 + .25 * min(ds, 2.0) + .25 * min(de, 2.0) + .10 * (1 - shared)


def _align(a: list[Cue], b: list[Cue]) -> list[tuple[int | None, int | None]]:
    """Bounded monotone alignment, never text-driven and never quadratic RAM."""
    n, m = len(a), len(b)
    band = max(64, abs(n - m) + 8)
    if band > 256:
        raise CorrectedQualityError("too many inserted or deleted cues to align")
    previous: list[float] = []
    previous_low = 0
    offsets: list[int] = []
    choices: list[bytearray] = []
    infinity = float("inf")
    for i in range(n + 1):
        low, high = max(0, i - band), min(m, i + band)
        cost = [infinity] * (high - low + 1)
        trace = bytearray(high - low + 1)
        for j in range(low, high + 1):
            k = j - low
            if i == 0 and j == 0:
                cost[k] = 0.0
                continue
            best, direction = infinity, 0
            if i and j and previous_low <= j - 1 < previous_low + len(previous):
                pair = _pair_cost(a[i - 1], b[j - 1])
                if pair is not None:
                    best = previous[j - 1 - previous_low] + pair
                    direction = 1
            if i and previous_low <= j < previous_low + len(previous):
                deletion = previous[j - previous_low] + 1.0
                if deletion < best:
                    best, direction = deletion, 2
            if j > low and cost[k - 1] < infinity:
                insertion = cost[k - 1] + 1.0
                if insertion < best:
                    best, direction = insertion, 3
            cost[k], trace[k] = best, direction
        offsets.append(low)
        choices.append(trace)
        previous, previous_low = cost, low
    if (
        not previous_low <= m < previous_low + len(previous)
        or not math.isfinite(previous[m - previous_low])
    ):
        raise CorrectedQualityError("no valid bounded timeline alignment")
    i, j = n, m
    aligned: list[tuple[int | None, int | None]] = []
    while i or j:
        direction = choices[i][j - offsets[i]]
        if direction == 1:
            aligned.append((i - 1, j - 1))
            i, j = i - 1, j - 1
        elif direction == 2:
            aligned.append((i - 1, None))
            i -= 1
        elif direction == 3:
            aligned.append((None, j - 1))
            j -= 1
        else:
            raise CorrectedQualityError("broken timestamp alignment")
    return aligned[::-1]


def _edit_distance(a: str, b: str) -> int:
    if a == b:
        return 0
    if len(a) < len(b):
        a, b = b, a
    previous = list(range(len(b) + 1))
    for i, left in enumerate(a, start=1):
        row = [i]
        for j, right in enumerate(b, start=1):
            row.append(min(
                row[-1] + 1, previous[j] + 1,
                previous[j - 1] + (left != right),
            ))
        previous = row
    return previous[-1]


def compare_corrected_cues(
    baseline: list[Cue], corrected: list[Cue],
) -> QualityComparison:
    """Exact Unicode punctuation, line wraps and 250ms edges are separate."""
    _validate(baseline)
    _validate(corrected)
    pairs = _align(baseline, corrected)
    matched = original_only = new_only = exact = wrap = precise = joint = errors = 0
    max_start = max_end = 0.0
    first_review: list[int] = []
    ref_chars = sum(len(_text(x.text)) for x in corrected)
    for i, j in pairs:
        if i is None:
            new_only += 1
            errors += len(_text(corrected[j].text))
            if len(first_review) < 16:
                first_review.append(j + 1)
            continue
        if j is None:
            original_only += 1
            errors += len(_text(baseline[i].text))
            if len(first_review) < 16:
                first_review.append(i + 1)
            continue
        matched += 1
        raw, reference = _text(baseline[i].text), _text(corrected[j].text)
        same_text = raw == reference
        same_wrap = (
            _physical_line_break_positions(raw)
            == _physical_line_break_positions(reference)
        )
        start = abs(baseline[i].start - corrected[j].start)
        end = abs(baseline[i].end - corrected[j].end)
        close = start <= .25 + 1e-8 and end <= .25 + 1e-8
        exact += same_text
        wrap += same_wrap
        precise += close
        joint += same_text and close
        errors += _edit_distance(raw, reference)
        max_start = max(max_start, start)
        max_end = max(max_end, end)
        if not (same_text and close) and len(first_review) < 16:
            first_review.append(j + 1)
    return QualityComparison(
        baseline_cues=len(baseline), corrected_cues=len(corrected),
        aligned_cues=matched, corrected_only=new_only,
        baseline_only=original_only, exact_visible_cues=exact,
        exact_physical_lines=wrap, precise_250ms_cues=precise,
        strict_joint_cues=joint, reference_characters=ref_chars,
        character_errors=errors,
        character_error_rate=round(errors / max(1, ref_chars), 6),
        max_start_drift_seconds=round(max_start, 3),
        max_end_drift_seconds=round(max_end, 3),
        first_review_cue_numbers=tuple(first_review),
    )


def audit_proofread_srt(original_srt: Path) -> QualityComparison:
    """Read-only local comparison; no OCR, download, editor or output files."""
    original = Path(original_srt).expanduser()
    corrected = original.with_name(original.stem + "-proofread.srt")
    for source in (original, corrected):
        if (
            source.suffix.lower() != ".srt" or source.is_symlink()
            or not source.is_file() or source.stat().st_nlink != 1
            or not 0 < source.stat().st_size <= 8 * 1024 * 1024
        ):
            raise CorrectedQualityError("missing or unsafe local SRT")
    if original.samefile(corrected):
        raise CorrectedQualityError("original SRT and corrected SRT are linked")
    lists = []
    for source in (original, corrected):
        first = source.stat()
        data = source.read_bytes()
        last = source.stat()
        if (
            len(data) != first.st_size or first.st_ino != last.st_ino
            or first.st_mtime_ns != last.st_mtime_ns
            or first.st_size != last.st_size
        ):
            raise CorrectedQualityError("SRT changed during comparison")
        try:
            kind, cues = parse_subtitle_bytes(data, source_url=str(source))
        except (UnicodeError, ValueError) as exc:
            raise CorrectedQualityError("invalid SRT format") from exc
        if kind != "srt":
            raise CorrectedQualityError("not SRT")
        # The legacy parser skips invalid blocks; do not silently score
        # a partly malformed human-edited transcript as a complete reference.
        text = decode_subtitle_bytes(data).replace("\r\n", "\n").replace("\r", "\n").strip()
        if len(re.split(r"\n[ \t]*\n+", text)) != len(cues):
            raise CorrectedQualityError("malformed SRT blocks skipped by parser")
        lists.append(cues)
    return compare_corrected_cues(*lists)
