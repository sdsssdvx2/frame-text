"""Read-only, reference-aware subtitle quality evidence without retaining text.

This module never calls OCR, reads browser sessions, or changes production cues.
Unmatched/aligned metrics are *comparison signals*, not verified transcription
accuracy: subtitle translations, timing offsets, and split/merged cues vary.
"""
from __future__ import annotations

from dataclasses import dataclass
from difflib import SequenceMatcher
import math
from bisect import bisect_left, bisect_right
import re
import statistics
import unicodedata

from .models import Cue
from .subtitle_language import normalize_language


@dataclass(frozen=True, slots=True)
class AuditOptions:
    alignment_tolerance_seconds: float = 1.5
    reference_shift_seconds: float = 0.0
    fast_reading_cps: float = 24.0
    long_gap_seconds: float = 35.0
    max_review_locations: int = 30
    # Display a bounded page *after* this original 1-based cue number.
    # Aggregate warning counts still span the entire validated subtitle.
    review_locations_after_cue: int = 0
    requested_language: str = "ko"
    # Optional independently known input-media duration, NEVER inferred from
    # the final SRT cue (which may itself be a persistent watermark).
    video_duration_seconds: float | None = None


def _text_key(value: str) -> str:
    # Comparison-only; retain original characters in memory for distance.
    normalized = unicodedata.normalize("NFKC", value or "").lower()
    return "".join(ch for ch in normalized if ch.isalnum())


def _edit_distance(a: str, b: str) -> int:
    """Exact character edit distance using linear auxiliary space."""
    if len(a) > len(b):
        a, b = b, a
    previous = list(range(len(a) + 1))
    for row_idx, y in enumerate(b, 1):
        current = [row_idx]
        for col_idx, x in enumerate(a, 1):
            current.append(min(
                current[col_idx - 1] + 1,
                previous[col_idx] + 1,
                previous[col_idx - 1] + (x != y),
            ))
        previous = current
    return previous[-1]


def _validate(cues: list[Cue], label: str) -> None:
    if not cues:
        raise ValueError(f"{label}_empty")
    # Keep a faulty/hostile single caption from triggering a multi-million
    # cell edit-distance calculation. Both caps are far above normal dialogue.
    if len(cues) > 10_000:
        raise ValueError(f"{label}_cue_budget_exceeded")
    previous_start = -1.0
    for cue in cues:
        if not isinstance(cue.text, str) or not cue.text.strip():
            raise ValueError(f"{label}_empty_text")
        if len(cue.text) > 4096:
            raise ValueError(f"{label}_text_budget_exceeded")
        if not (
            math.isfinite(cue.start)
            and math.isfinite(cue.end)
            and 0 <= cue.start < cue.end
        ):
            raise ValueError(f"{label}_invalid_timeline")
        if cue.start < previous_start:
            raise ValueError(f"{label}_out_of_order")
        previous_start = cue.start


def _script_warning(text: str, language: str) -> str | None:
    """Script presence is a review signal, not a language detector.

    Japanese may be Kanji-only; Chinese can mix Latin; subtitles may quote
    other languages. Never discard a cue based on character coverage.
    """
    if not any(ch.isalpha() for ch in text):
        return None
    normalized = unicodedata.normalize("NFKC", text)
    names = [unicodedata.name(ch, "") for ch in normalized if ch.isalpha()]
    han = any(name.startswith(("CJK UNIFIED IDEOGRAPH", "CJK COMPATIBILITY IDEOGRAPH"))
              for name in names)
    if language == "ko":
        present = any("HANGUL" in name for name in names)
        warning = "no_hangul_review"
    elif language == "en":
        present = any("LATIN" in name for name in names)
        warning = "no_latin_review"
    elif language == "ja":
        present = han or any(
            "HIRAGANA" in name or "KATAKANA" in name for name in names
        )
        warning = "no_japanese_script_review"
    else:
        present = han
        warning = "no_han_review"
    return None if present else warning



def _repeated_full_line_overlays(candidate: list[Cue]):
    """Yield advisory-only locations of a recurring, full OCR text line.

    A single-line logo is indistinguishable from real dialogue using an SRT
    alone, as is an unchanged two-line caption. Require >=3 consecutive
    MULTILINE cues with the same complete first/last line, >=3 different
    other-line combinations and short inter-cue gaps. Never infer absence
    of a legitimate subtitle, rewrite the text, or expose its contents.
    This is a bounded O(cues * text length) post-export review heuristic.
    """
    for side in (0, -1):
        run_start = None
        run_key = None
        companions: set[tuple[str, ...]] = set()
        run_count = 0
        last_end = -1.0
        for index in range(len(candidate) + 1):
            cue = candidate[index] if index < len(candidate) else None
            lines = (
                [_text_key(line) for line in cue.text.splitlines()
                 if _text_key(line)]
                if cue is not None else []
            )
            key = lines[side] if len(lines) >= 2 else ""
            other = (
                tuple(lines[1:]) if side == 0 else tuple(lines[:-1])
            ) if key else ()
            continuing = bool(
                key and len(key) >= 8 and key == run_key
                and cue is not None and cue.start - last_end <= 8.0
            )
            if not continuing:
                if run_count >= 3 and len(companions) >= 3:
                    assert run_start is not None
                    yield run_start + 1, candidate[run_start].start
                run_start = index if key and len(key) >= 8 else None
                run_key = key if run_start is not None else None
                companions = set()
                run_count = 0
            if run_start is not None and cue is not None:
                companions.add(other)
                run_count += 1
                last_end = cue.end


def _review(candidate: list[Cue], options: AuditOptions) -> dict[str, object]:
    issues: dict[str, list[dict[str, object]]] = {}
    counts: dict[str, int] = {}

    def mark(code: str, cue_index: int, start: float) -> None:
        counts[code] = counts.get(code, 0) + 1
        bucket = issues.setdefault(code, [])
        if (cue_index > options.review_locations_after_cue
                and len(bucket) < options.max_review_locations):
            bucket.append({"cue": cue_index, "start_seconds": round(start, 3)})

    language = normalize_language(options.requested_language)
    previous: Cue | None = None
    for idx, cue in enumerate(candidate, 1):
        duration = cue.end - cue.start
        key = _text_key(cue.text)
        script_warning = _script_warning(
            cue.text, "zh" if language.startswith("zh") else language,
        )
        if duration < 0.15:
            mark("very_short_duration_review", idx, cue.start)
        if len(key) / duration > options.fast_reading_cps:
            mark("fast_reading_review", idx, cue.start)
        # A hard-subtitle OCR cue hundreds of characters long or spanning
        # many physical lines is almost always a scene-text/credits capture,
        # not ordinary dialogue. This remains review-only: never truncate,
        # rewrite or delete user-facing text without source-frame inspection.
        physical_lines = [line for line in cue.text.splitlines() if line.strip()]
        if len(cue.text) > 600 or len(physical_lines) > 8:
            mark("oversized_ocr_cue_review", idx, cue.start)
        if script_warning is not None:
            mark(script_warning, idx, cue.start)
        if (options.video_duration_seconds is not None
                and cue.end > options.video_duration_seconds + 0.5):
            mark("cue_exceeds_video_duration_review", idx, cue.start)
        if previous is not None:
            if previous.end > cue.start + 0.02:
                mark("overlapping_cues_review", idx, cue.start)
            if cue.start - previous.end > options.long_gap_seconds:
                mark("long_gap_review", idx, previous.end)
            if key and _text_key(previous.text) == key and cue.start - previous.end <= 0.25:
                mark("nearby_duplicate_text_review", idx, cue.start)
        previous = cue

    # Repeated OCR-only overlay rows may appear alongside different real
    # subtitles. Mark the first candidate in each run, never remove it.
    # Reviewers must inspect source frames before deciding what was shown.
    for first_index, start_time in _repeated_full_line_overlays(candidate):
        mark("repeated_multiline_overlay_review", first_index, start_time)

    # A full-length logo with no dialogue may be indistinguishable from a
    # single legitimate full-length caption. Flag for review, NEVER remove.
    # This requires independently known media length; the last OCR cue end
    # MUST NOT be used as a substitute for actual video duration.
    video_duration = options.video_duration_seconds
    if video_duration is not None and len(candidate) == 1:
        only = candidate[0]
        if (
            only.start <= 0.12 * video_duration
            and only.end >= 0.88 * video_duration
            and only.end - only.start >= 0.75 * video_duration
        ):
            mark("near_full_runtime_single_cue_review", 1, only.start)

    return {
        "cue_count": len(candidate),
        "requested_language": language,
        "script_warnings_are_heuristic_only": True,
        "independent_video_duration_supplied": video_duration is not None,
        "last_end_seconds": round(candidate[-1].end, 3),
        "warnings_are_not_verified_ocr_errors": True,
        "warning_counts": counts,
        "review_locations": issues,
        "review_locations_limited_per_code": options.max_review_locations,
        "review_locations_after_cue": options.review_locations_after_cue,
    }


def _temporal_overlap(a: Cue, b: Cue, shift: float = 0.0) -> float:
    return max(0.0, min(a.end, b.end + shift) - max(a.start, b.start + shift))


def _pair_score(candidate: Cue, reference: Cue, options: AuditOptions) -> tuple[float, float] | None:
    shifted_start = reference.start + options.reference_shift_seconds
    shifted_end = reference.end + options.reference_shift_seconds
    overlap = _temporal_overlap(candidate, reference, options.reference_shift_seconds)
    candidate_duration = candidate.end - candidate.start
    reference_duration = reference.end - reference.start
    shortest = min(candidate_duration, reference_duration)
    longest = max(candidate_duration, reference_duration)
    start_error = abs(candidate.start - shifted_start)
    end_error = abs(candidate.end - shifted_end)
    midpoint_error = abs((candidate.start + candidate.end - shifted_start - shifted_end) * 0.5)
    tolerance = options.alignment_tolerance_seconds
    # An endpoint near a very long cue must not create a spurious 1:1 match
    # against a short reference subtitle. True one-to-many merges/splits are
    # reported separately as overlapping, unmatched cues.
    #
    # Similarly, two short adjacent captions can have both endpoint errors
    # below a generous time tolerance while sharing no visible interval.
    # Requiring positive meaningful overlap avoids inflating alignment count.
    if (
        shortest / longest < 0.25
        or overlap <= 0
        or overlap / shortest < 0.20
        or min(start_error, end_error, midpoint_error) > tolerance
    ):
        return None
    a, b = _text_key(candidate.text), _text_key(reference.text)
    similarity = SequenceMatcher(None, a, b, autojunk=False).ratio() if a or b else 1.0
    overlap_ratio = min(1.0, overlap / shortest) if shortest else 0.0
    # Positive match rewards maximize paired cues before choosing the best
    # among ambiguous local alignments. Text is only a tie-break, never a
    # replacement for temporal evidence.
    return 2.0 + similarity + 0.5 * overlap_ratio, similarity


def _align(candidate: list[Cue], reference: list[Cue], options: AuditOptions) -> list[tuple[int, int, float]]:
    """Monotonic one-to-one maximum-weight dynamic programming alignment."""
    n, m = len(candidate), len(reference)
    scores = [[0.0] * (m + 1) for _ in range(n + 1)]
    back = [bytearray(m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            best = scores[i - 1][j]
            direction = 1  # skip candidate
            if scores[i][j - 1] > best + 1e-12:
                best = scores[i][j - 1]
                direction = 2  # skip reference
            pair = _pair_score(candidate[i - 1], reference[j - 1], options)
            if pair is not None:
                score, similarity = pair
                if scores[i - 1][j - 1] + score > best + 1e-12:
                    best = scores[i - 1][j - 1] + score
                    direction = 3
            scores[i][j] = best
            back[i][j] = direction
    matches = []
    i, j = n, m
    while i > 0 and j > 0:
        direction = back[i][j]
        if direction == 3:
            pair = _pair_score(candidate[i - 1], reference[j - 1], options)
            assert pair is not None
            matches.append((i - 1, j - 1, pair[1]))
            i -= 1
            j -= 1
        elif direction == 1:
            i -= 1
        else:
            j -= 1
    matches.reverse()
    return matches


# Bound plausible time-overlap work independent of n*m. A pathological
# collection of thousands of all-overlapping cues still fails closed rather
# than consuming quadratic CPU or RAM. Typical episode dialogue produces
# only a few plausible reference candidates per cue.
_MAX_SPARSE_PAIR_VISITS = 100_000


def _align_sparse(
    candidate: list[Cue], reference: list[Cue], options: AuditOptions,
) -> list[tuple[int, int, float]]:
    """Exact maximum-weight monotonic matching without an n*m DP matrix.

    Candidate and reference starts were validated as nondecreasing. Even when
    ends overlap or a reference cue extends unusually far forward, prefix
    max-ends safely prune only genuinely impossible time-overlap pairs.

    A Fenwick tree retrieves the best chain using strictly earlier reference
    and candidate positions; all updates for one candidate are deferred to
    the end of its row. With equal total weights, prefer earlier *last*
    (candidate, reference) positions, recursively: the same right-to-left
    skip-on-tie preference as _align's DP traceback. Existing <=2M pair
    callers continue to run the original dense implementation unchanged.
    """
    m = len(reference)
    shift = options.reference_shift_seconds
    starts = [r.start + shift for r in reference]
    max_end: list[float] = []
    latest = float("-inf")
    for item in reference:
        latest = max(latest, item.end + shift)
        max_end.append(latest)

    # record: (total weight, previous node, candidate index,
    #          reference index, comparison text similarity)
    records: list[tuple[float, int, int, int, float]] = []
    tree: list[int] = [-1] * (m + 1)
    eps = 1e-12

    def better(left: int, right: int) -> bool:
        if left < 0:
            return False
        if right < 0:
            return True
        left_score, right_score = records[left][0], records[right][0]
        if left_score > right_score + eps:
            return True
        if right_score > left_score + eps:
            return False
        # Reconstruct neither transcript nor a full path array. Tie-break
        # using immutable predecessor nodes (constant auxiliary memory).
        while left >= 0 and right >= 0:
            li, lj = records[left][2:4]
            ri, rj = records[right][2:4]
            if li != ri:
                return li < ri
            if lj != rj:
                return lj < rj
            left, right = records[left][1], records[right][1]
        return left < 0 <= right

    def query(last_reference_exclusive: int) -> int:
        best = -1
        while last_reference_exclusive:
            entry = tree[last_reference_exclusive]
            if better(entry, best):
                best = entry
            last_reference_exclusive -= last_reference_exclusive & -last_reference_exclusive
        return best

    def update(reference_index: int, node: int) -> None:
        position = reference_index + 1
        while position <= m:
            if better(node, tree[position]):
                tree[position] = node
            position += position & -position

    possible_visited = 0
    for i, cue in enumerate(candidate):
        # Only start positions strictly before cue.end and possible end
        # positions strictly after cue.start can overlap in source time.
        left = bisect_right(max_end, cue.start)
        right = bisect_left(starts, cue.end)
        if left >= right:
            continue
        possible_visited += right - left
        if possible_visited > _MAX_SPARSE_PAIR_VISITS:
            raise ValueError("quality_audit_reference_pair_budget_exceeded")

        pending: list[tuple[int, int]] = []
        for j in range(left, right):
            if reference[j].end + shift <= cue.start:
                continue
            match = _pair_score(cue, reference[j], options)
            if match is None:
                continue
            score, similarity = match
            prior = query(j)
            total = score + (records[prior][0] if prior >= 0 else 0.0)
            records.append((total, prior, i, j, similarity))
            pending.append((j, len(records) - 1))
        # Two reference positions cannot consume the same OCR cue.
        for j, node in pending:
            update(j, node)

    matches: list[tuple[int, int, float]] = []
    node = query(m)
    while node >= 0:
        _, previous, i, j, similarity = records[node]
        matches.append((i, j, similarity))
        node = previous
    matches.reverse()
    return matches


def _quantile_abs(values: list[float], ratio: float) -> float | None:
    if not values:
        return None
    ordered = sorted(abs(v) for v in values)
    return round(ordered[min(len(ordered) - 1, math.ceil(len(ordered) * ratio) - 1)], 3)


def _reference_audit(
    candidate: list[Cue], reference: list[Cue], options: AuditOptions,
) -> dict[str, object]:
    matches = (
        _align(candidate, reference, options)
        if len(candidate) * len(reference) <= 2_000_000
        else _align_sparse(candidate, reference, options)
    )
    paired_candidate = {i for i, _, _ in matches}
    paired_reference = {j for _, j, _ in matches}
    unmatched_c = [i for i in range(len(candidate)) if i not in paired_candidate]
    unmatched_r = [j for j in range(len(reference)) if j not in paired_reference]
    start_errors = []
    end_errors = []
    total_char_errors = 0
    total_reference_characters = 0
    exact = 0
    similarities = []
    for ci, ri, similarity in matches:
        a, b = candidate[ci], reference[ri]
        akey, bkey = _text_key(a.text), _text_key(b.text)
        total_char_errors += _edit_distance(akey, bkey)
        total_reference_characters += len(bkey)
        exact += int(akey == bkey)
        similarities.append(similarity)
        start_errors.append(a.start - (b.start + options.reference_shift_seconds))
        end_errors.append(a.end - (b.end + options.reference_shift_seconds))

    def overlapping_unmatched(indices: list[int], own: list[Cue], other: list[Cue], own_shift: float, other_shift: float) -> int:
        # A prefix-max end-time index answers each unmatched time-overlap
        # query in O(log m), even if one long reference subtitle overlaps
        # many short OCR cues. Existing split/merge count semantics remain.
        shifted_starts = [c.start + other_shift for c in other]
        latest = float("-inf")
        shifted_prefix_ends: list[float] = []
        for item in other:
            latest = max(latest, item.end + other_shift)
            shifted_prefix_ends.append(latest)
        count = 0
        for i in indices:
            finish = own[i].end + own_shift
            start = own[i].start + own_shift
            upto = bisect_left(shifted_starts, finish)
            if upto > 0 and shifted_prefix_ends[upto - 1] > start:
                count += 1
        return count

    return {
        "reference_supplied": True,
        "reference_is_verified_exact_translation": False,
        "reference_cue_count": len(reference),
        "candidate_cue_count": len(candidate),
        "reference_shift_seconds": options.reference_shift_seconds,
        "alignment_tolerance_seconds": options.alignment_tolerance_seconds,
        "one_to_one_aligned_cues": len(matches),
        "reference_alignment_fraction": round(len(matches) / len(reference), 4),
        "candidate_alignment_fraction": round(len(matches) / len(candidate), 4),
        "text_exact_match_count_aligned_only": exact,
        "average_text_similarity_aligned_only": round(statistics.mean(similarities), 4) if similarities else None,
        "character_error_rate_aligned_only": round(total_char_errors / total_reference_characters, 4) if total_reference_characters else None,
        "median_absolute_start_error_seconds_aligned_only": round(statistics.median(abs(e) for e in start_errors), 3) if start_errors else None,
        "p95_absolute_start_error_seconds_aligned_only": _quantile_abs(start_errors, 0.95),
        "median_absolute_end_error_seconds_aligned_only": round(statistics.median(abs(e) for e in end_errors), 3) if end_errors else None,
        "p95_absolute_end_error_seconds_aligned_only": _quantile_abs(end_errors, 0.95),
        "unmatched_candidate_count": len(unmatched_c),
        "unmatched_reference_count": len(unmatched_r),
        "unmatched_candidate_with_any_time_overlap_count": overlapping_unmatched(unmatched_c, candidate, reference, 0.0, options.reference_shift_seconds),
        "unmatched_reference_with_any_time_overlap_count": overlapping_unmatched(unmatched_r, reference, candidate, options.reference_shift_seconds, 0.0),
        "unmatched_candidate_cue_numbers_first_30": [x + 1 for x in unmatched_c[:30]],
        "unmatched_reference_cue_numbers_first_30": [x + 1 for x in unmatched_r[:30]],
        "comparison_is_not_ground_truth_accuracy": True,
        "one_to_many_split_merge_may_affect_unmatched_counts": True,
    }


def audit_subtitles(
    candidate: list[Cue],
    *,
    reference: list[Cue] | None = None,
    options: AuditOptions | None = None,
) -> dict[str, object]:
    """Calculate text-free, privacy-preserving metrics; do not alter input cues."""
    options = options or AuditOptions()
    if (
        not math.isfinite(options.alignment_tolerance_seconds)
        or not 0 < options.alignment_tolerance_seconds <= 30.0
        or not math.isfinite(options.reference_shift_seconds)
        or abs(options.reference_shift_seconds) > 3600.0
        or not math.isfinite(options.fast_reading_cps)
        or options.fast_reading_cps <= 0
        or not math.isfinite(options.long_gap_seconds)
        or options.long_gap_seconds <= 0
        or type(options.max_review_locations) is not int
        or not 0 <= options.max_review_locations <= 500
        or type(options.review_locations_after_cue) is not int
        or not 0 <= options.review_locations_after_cue <= 10000
        or (options.video_duration_seconds is not None and (
            type(options.video_duration_seconds) not in (int, float)
            or not math.isfinite(options.video_duration_seconds)
            or not 0.1 <= options.video_duration_seconds <= 21600.0
        ))
    ):
        raise ValueError("quality_audit_invalid_options")
    try:
        normalize_language(options.requested_language)
    except ValueError as exc:
        raise ValueError("quality_audit_invalid_options") from exc
    _validate(candidate, "candidate")
    if reference is not None:
        _validate(reference, "reference")
        # Small inputs retain the original DP exactly; long-episode inputs
        # switch to time-indexed matching with a bounded pair budget.
    report: dict[str, object] = {
        "schema": 1,
        "audit_type": "subtitle_comparison_not_ocr_ground_truth",
        "accuracy_verified": False,
        "contains_subtitle_text": False,
        "contains_source_file_paths": False,
        "candidate": _review(candidate, options),
        "reference_comparison": (
            _reference_audit(candidate, reference, options)
            if reference is not None
            else {"reference_supplied": False, "comparison_not_performed": True}
        ),
    }
    return report
