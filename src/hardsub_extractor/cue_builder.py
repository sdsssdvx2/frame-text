from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

from .models import Cue, VisualSegment
from .ocr.base import (
    OCREngine,
    comparison_key,
    normalize_text,
    text_similarity,
    vote_observations,
)


@dataclass(slots=True)
class CueRecognitionConfig:
    min_confidence: float = 0.20
    support_similarity: float = 0.82
    min_supporters: int = 2

    # A genuinely very short subtitle can have only one sampled frame.
    # Accept a single observation only if it is extremely confident and the
    # visual segment itself is short.
    single_frame_confidence: float = 0.94
    single_frame_max_duration: float = 0.32

    # If every available sample agrees but there is only one physical sample,
    # this threshold still applies.
    min_text_length: int = 1

    # The visual detector can deliberately over-segment a subtitle when the
    # animation underneath it is busy. Track OCR text across those tiny
    # segments so one persistent subtitle is not emitted as many fragments.
    temporal_blank_confirm: float = 0.35
    temporal_strong_similarity: float = 0.80
    temporal_weak_similarity: float = 0.65
    temporal_partial_similarity: float = 0.58
    temporal_change_support_similarity: float = 0.75
    temporal_change_min_supporters: int = 3
    temporal_change_min_support_duration: float = 0.30

    # A real new subtitle can be textually similar to the previous one.
    # Require a coherent weak-alternative streak before splitting it.
    temporal_weak_change_similarity: float = 0.88
    temporal_weak_change_min_supporters: int = 3
    temporal_weak_change_min_support_duration: float = 0.45


@dataclass(slots=True)
class _SegmentEvidence:
    segment: VisualSegment
    text: str
    confidence: float
    supporters: int
    local_accepted: bool

    @property
    def start(self) -> float:
        return self.segment.start

    @property
    def end(self) -> float:
        return self.segment.end

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)


def _supporting_observations(voted_text: str, observations, threshold: float):
    voted_text = normalize_text(voted_text)
    if not voted_text:
        return []
    supporters = []
    for obs in observations:
        text = normalize_text(obs.text)
        if text and text_similarity(voted_text, text) >= threshold:
            supporters.append(obs)
    return supporters


def _support_count(voted_text: str, observations, threshold: float) -> int:
    return len(_supporting_observations(voted_text, observations, threshold))


def _character_multiset_similarity(a: str, b: str) -> float:
    """Layout-insensitive character overlap used only for scene-cut repair."""
    ka, kb = comparison_key(a), comparison_key(b)
    if not ka and not kb:
        return 1.0
    if not ka or not kb:
        return 0.0
    common = sum((Counter(ka) & Counter(kb)).values())
    return (2.0 * common) / (len(ka) + len(kb))


def _layout_reorder_match(a: str, b: str) -> bool:
    ordered = text_similarity(a, b)
    return ordered >= 0.70 and _character_multiset_similarity(a, b) >= 0.93


def _representative(
    items: list[_SegmentEvidence],
    *,
    support_similarity: float = 0.82,
) -> _SegmentEvidence | None:
    """Pick semantic content first, then the best equivalent surface form.

    Visual motion may split one stable subtitle into many tiny segments. A raw
    fragment count therefore lets a burst of short variants outvote a longer
    stable observation. A hard fuzzy-similarity cutoff has the opposite edge
    case: a slightly wrong OCR spelling can happen to include one more segment
    just above the threshold and win despite being less central to the whole
    observation set.

    The first stage is the duration-weighted semantic medoid used to reject
    real OCR spelling errors. Candidates whose soft scores are within one
    percent of total observed duration are resolved by hard support, local
    support, weighted centrality and confidence. Only after a semantic winner
    is chosen may spaces, punctuation or line breaks be recovered from other
    candidates, and only from candidates with the exact same comparison key.
    This keeps repeated spacing such as ``소원 수리함`` without allowing a
    longer misspelling such as an inserted Hangul syllable to outrank the
    semantically better reading.
    """
    candidates = [item for item in items if item.text]
    if not candidates:
        return None

    total_duration = sum(item.duration for item in candidates)
    scored = []
    for candidate in candidates:
        support = 0
        support_duration = 0.0
        soft_support_duration = 0.0
        weighted = candidate.confidence
        for other in candidates:
            similarity = text_similarity(candidate.text, other.text)
            layout_match = _layout_reorder_match(candidate.text, other.text)
            effective_similarity = max(similarity, 0.99 if layout_match else 0.0)
            soft_support_duration += other.duration * (effective_similarity ** 2)
            if similarity >= support_similarity or layout_match:
                support += 1
                support_duration += other.duration
            if other is not candidate:
                weighted += similarity * max(0.20, other.confidence)

        scored.append(
            (
                candidate,
                soft_support_duration,
                support_duration,
                support,
                weighted,
            )
        )

    max_soft_support = max(row[1] for row in scored)
    soft_tie_tolerance = max(1e-9, total_duration * 0.01)
    near_best = [
        row for row in scored
        if row[1] >= max_soft_support - soft_tie_tolerance
    ]

    semantic_best, *_ = max(
        near_best,
        key=lambda row: (
            row[2],
            row[3],
            row[0].supporters,
            row[4],
            row[0].confidence,
            min(row[0].duration, 0.75),
            -row[0].start,
        ),
    )

    semantic_key = comparison_key(semantic_best.text)
    same_semantic = [
        item for item in candidates if comparison_key(item.text) == semantic_key
    ]
    if len(same_semantic) <= 1:
        return semantic_best

    surface_stats: dict[str, tuple[float, int]] = {}
    for item in same_semantic:
        surface = normalize_text(item.text)
        duration, count = surface_stats.get(surface, (0.0, 0))
        surface_stats[surface] = (duration + item.duration, count + 1)

    return max(
        same_semantic,
        key=lambda item: (
            surface_stats[normalize_text(item.text)][0],
            surface_stats[normalize_text(item.text)][1],
            item.supporters,
            item.confidence,
            min(item.duration, 0.75),
            -item.start,
        ),
    )


def _temporal_relation(a: str, b: str, cfg: CueRecognitionConfig) -> str:
    similarity = text_similarity(a, b)
    if similarity >= cfg.temporal_strong_similarity or _layout_reorder_match(a, b):
        return "strong"

    ka, kb = comparison_key(a), comparison_key(b)
    if min(len(ka), len(kb)) >= 6 and similarity >= cfg.temporal_weak_similarity:
        return "weak"

    if ka and kb:
        length_ratio = min(len(ka), len(kb)) / max(len(ka), len(kb))
        if (
            min(len(ka), len(kb)) >= 4
            and length_ratio >= 0.55
            and similarity >= cfg.temporal_partial_similarity
        ):
            return "weak"
    return "different"


def _coherent_group(
    items: list[_SegmentEvidence],
    *,
    similarity_threshold: float,
    min_supporters: int,
    min_support_duration: float,
) -> tuple[bool, _SegmentEvidence | None]:
    valid = [item for item in items if item.text]
    representative = _representative(valid)
    if representative is None:
        return False, None

    supporters = []
    for item in valid:
        similarity = text_similarity(representative.text, item.text)
        if similarity >= similarity_threshold or _layout_reorder_match(
            representative.text, item.text
        ):
            supporters.append(item)

    support_duration = sum(item.duration for item in supporters)
    return (
        len(supporters) >= min_supporters
        and support_duration >= min_support_duration
    ), representative


def _temporal_track(
    evidence: list[_SegmentEvidence],
    cfg: CueRecognitionConfig,
) -> list[Cue]:
    """Convert noisy visual segments into persistent subtitle text states."""
    out: list[Cue] = []
    current: list[_SegmentEvidence] = []
    current_start: float | None = None
    pending: list[_SegmentEvidence] = []
    seed: list[_SegmentEvidence] = []
    weak_streak: list[_SegmentEvidence] = []
    weak_anchor: _SegmentEvidence | None = None
    blank_start: float | None = None
    blank_end: float | None = None

    def emit(
        items: list[_SegmentEvidence],
        start: float,
        end: float | None = None,
    ) -> None:
        representative = _representative(items)
        if representative is None:
            return
        cue_end = end if end is not None else items[-1].end
        if cue_end <= start:
            return
        out.append(
            Cue(
                start,
                cue_end,
                representative.text,
                max(item.confidence for item in items),
            )
        )

    def pending_confirmed(items: list[_SegmentEvidence]) -> bool:
        if not items:
            return False
        if any(
            item.local_accepted
            and item.duration >= 0.45
            and item.supporters >= cfg.min_supporters
            for item in items
        ):
            return True
        confirmed, _ = _coherent_group(
            items,
            similarity_threshold=cfg.temporal_change_support_similarity,
            min_supporters=cfg.temporal_change_min_supporters,
            min_support_duration=cfg.temporal_change_min_support_duration,
        )
        return confirmed

    def weak_change_confirmed(
        items: list[_SegmentEvidence],
        previous: _SegmentEvidence,
    ) -> bool:
        confirmed, representative = _coherent_group(
            items,
            similarity_threshold=cfg.temporal_weak_change_similarity,
            min_supporters=cfg.temporal_weak_change_min_supporters,
            min_support_duration=cfg.temporal_weak_change_min_support_duration,
        )
        return bool(
            confirmed
            and representative is not None
            and text_similarity(previous.text, representative.text)
            < cfg.temporal_strong_similarity
        )

    for item in evidence:
        valid = (
            bool(item.text)
            and len(item.text) >= cfg.min_text_length
            and item.confidence >= cfg.min_confidence
        )

        if not valid:
            seed.clear()
            weak_streak.clear()
            weak_anchor = None

            if pending and current:
                if pending_confirmed(pending):
                    emit(current, current_start, pending[0].start)
                    current = list(pending)
                    current_start = pending[0].start
                elif any(candidate.local_accepted for candidate in pending):
                    emit(current, current_start, pending[0].start)
                    accepted = [
                        candidate for candidate in pending if candidate.local_accepted
                    ]
                    if accepted:
                        emit(accepted, accepted[0].start, accepted[-1].end)
                    current = []
                    current_start = None
                pending.clear()

            if current:
                if blank_start is None:
                    blank_start = item.start
                blank_end = item.end
                if blank_end - blank_start >= cfg.temporal_blank_confirm:
                    emit(current, current_start, blank_start)
                    current = []
                    current_start = None
                    blank_start = None
                    blank_end = None
            continue

        if not current:
            blank_start = None
            blank_end = None
            if item.local_accepted:
                if seed:
                    seed_rep = _representative(seed)
                    relation = (
                        _temporal_relation(seed_rep.text, item.text, cfg)
                        if seed_rep is not None
                        else "different"
                    )
                    if relation != "different":
                        seed.append(item)
                        current = list(seed)
                        current_start = seed[0].start
                        seed.clear()
                    else:
                        current = [item]
                        current_start = item.start
                        seed.clear()
                else:
                    current = [item]
                    current_start = item.start
            else:
                seed.append(item)
                if pending_confirmed(seed):
                    current = list(seed)
                    current_start = seed[0].start
                    seed.clear()
            continue

        assert current_start is not None

        if blank_start is not None:
            previous = _representative(current)
            relation = (
                _temporal_relation(previous.text, item.text, cfg)
                if previous is not None
                else "different"
            )
            blank_duration = (blank_end or blank_start) - blank_start
            if relation != "different" and blank_duration < cfg.temporal_blank_confirm:
                current.append(item)
            else:
                emit(current, current_start, blank_start)
                current = [item] if item.local_accepted else []
                current_start = item.start if current else None
                seed = [] if current else [item]
            pending.clear()
            weak_streak.clear()
            weak_anchor = None
            blank_start = None
            blank_end = None
            continue

        previous = _representative(current)
        relation = (
            _temporal_relation(previous.text, item.text, cfg)
            if previous is not None
            else "different"
        )

        if pending:
            pending_rep = _representative(pending)
            pending_relation = (
                _temporal_relation(pending_rep.text, item.text, cfg)
                if pending_rep is not None
                else "different"
            )

            if relation != "different":
                pending.clear()
                weak_streak.clear()
                weak_anchor = None
                current.append(item)
                continue

            if (
                len(pending) == 1
                and pending[0].local_accepted
                and pending[0].duration <= cfg.single_frame_max_duration
                and pending_relation == "different"
            ):
                emit(current, current_start, pending[0].start)
                emit(pending, pending[0].start, pending[0].end)
                current = []
                current_start = None
                pending = []
                seed = [item]
                weak_streak.clear()
                weak_anchor = None
                if item.local_accepted:
                    current = [item]
                    current_start = item.start
                    seed.clear()
                continue

            pending.append(item)
            if pending_confirmed(pending):
                emit(current, current_start, pending[0].start)
                current = list(pending)
                current_start = pending[0].start
                pending.clear()
                weak_streak.clear()
                weak_anchor = None
            continue

        if relation == "strong":
            weak_streak.clear()
            weak_anchor = None
            current.append(item)
            continue

        if relation == "weak":
            if not weak_streak:
                weak_anchor = previous
            weak_streak.append(item)
            current.append(item)
            if (
                weak_anchor is not None
                and weak_change_confirmed(weak_streak, weak_anchor)
            ):
                split_at = weak_streak[0].start
                weak_ids = {id(candidate) for candidate in weak_streak}
                old_items = [
                    candidate for candidate in current if id(candidate) not in weak_ids
                ]
                emit(old_items, current_start, split_at)
                current = list(weak_streak)
                current_start = split_at
                weak_streak.clear()
                weak_anchor = None
            continue

        weak_streak.clear()
        weak_anchor = None
        pending = [item]
        if pending_confirmed(pending):
            emit(current, current_start, pending[0].start)
            current = list(pending)
            current_start = pending[0].start
            pending.clear()

    if current:
        assert current_start is not None
        if pending:
            if pending_confirmed(pending):
                emit(current, current_start, pending[0].start)
                current = list(pending)
                current_start = pending[0].start
            else:
                accepted = [
                    candidate for candidate in pending if candidate.local_accepted
                ]
                if accepted:
                    emit(current, current_start, pending[0].start)
                    emit(accepted, accepted[0].start, accepted[-1].end)
                    current = []
        if current:
            emit(current, current_start)

    return out


def recognize_segments(
    segments: list[VisualSegment],
    ocr: OCREngine,
    *,
    min_confidence: float = 0.20,
    diagnostics=None,
    config: CueRecognitionConfig | None = None,
) -> list[Cue]:
    cfg = config or CueRecognitionConfig(min_confidence=min_confidence)
    evidence: list[_SegmentEvidence] = []

    for index, segment in enumerate(segments):
        observations = [ocr.recognize(frame) for frame in segment.samples]
        voted = vote_observations(observations)

        if diagnostics is not None:
            diagnostics.record_segment(index, segment, observations, voted)

        text = normalize_text(voted.text)
        supporting = _supporting_observations(
            text, observations, cfg.support_similarity
        )
        supporters = len(supporting)
        duration = segment.end - segment.start
        enough_consensus = supporters >= cfg.min_supporters
        short_high_confidence_exception = (
            supporters >= 1
            and voted.confidence >= cfg.single_frame_confidence
            and duration <= cfg.single_frame_max_duration
        )
        local_accepted = bool(
            text
            and len(text) >= cfg.min_text_length
            and voted.confidence >= cfg.min_confidence
            and segment.end > segment.start
            and (enough_consensus or short_high_confidence_exception)
        )

        # A long visual segment can stay visually quiet while a subtitle first
        # appears near its tail.  In that case the OCR vote is supported only
        # by late samples.  Starting the cue at segment.start can move dialogue
        # seconds early (observed in live EpSite QA).  Keep the conservative
        # visual end, but never start before the first sample that actually
        # supports the voted text.
        evidence_segment = segment
        if supporting:
            first_supported_pts = max(segment.start, min(obs.pts for obs in supporting))
            first_sample_supports = bool(
                observations
                and normalize_text(observations[0].text)
                and text_similarity(text, observations[0].text)
                >= cfg.support_similarity
            )
            if not first_sample_supports and first_supported_pts < segment.end:
                evidence_segment = VisualSegment(
                    first_supported_pts,
                    segment.end,
                    segment.samples,
                    segment.diagnostics,
                )

        evidence.append(
            _SegmentEvidence(
                segment=evidence_segment,
                text=text,
                confidence=voted.confidence,
                supporters=supporters,
                local_accepted=local_accepted,
            )
        )

    return _temporal_track(evidence, cfg)


def merge_cues(
    cues: list[Cue],
    *,
    similarity_threshold: float = 0.94,
    max_gap: float = 0.20,
    min_duration: float = 0.08,
    layout_reorder_similarity: float = 0.96,
    layout_reorder_ordered_floor: float = 0.72,
) -> list[Cue]:
    out: list[Cue] = []
    for cue in sorted(cues, key=lambda c: (c.start, c.end)):
        cue.text = normalize_text(cue.text)
        if not cue.text:
            continue

        if out:
            prev = out[-1]
            gap = cue.start - prev.end
            ordered_similarity = text_similarity(prev.text, cue.text)
            normal_match = ordered_similarity >= similarity_threshold
            layout_reorder_match = (
                ordered_similarity >= layout_reorder_ordered_floor
                and _character_multiset_similarity(prev.text, cue.text)
                >= layout_reorder_similarity
            )
            if gap <= max_gap and (normal_match or layout_reorder_match):
                if normal_match and cue.confidence > prev.confidence:
                    prev.text = cue.text
                prev.end = max(prev.end, cue.end)
                prev.confidence = max(prev.confidence, cue.confidence)
                continue

        if cue.duration() >= min_duration:
            out.append(cue)
    return out
