from __future__ import annotations

import re
from dataclasses import replace
from typing import Iterable

from .cue_builder import (
    CueRecognitionConfig,
    _SegmentEvidence,
    _supporting_observations,
    _temporal_relation,
    _temporal_track,
)
from .models import Cue, OCRObservation, VisualSegment
from .ocr.base import (
    OCREngine,
    comparison_key,
    normalize_text,
    text_similarity,
    vote_observations,
)


# Anime subtitle fonts often append hollow circles or similar ornaments. OCR can
# turn those shapes into compatibility jamo (ㅇ), latin o, or, after terminal
# punctuation, a single Hangul syllable. Only trim shapes in deliberately narrow
# contexts; ordinary Korean text and normal punctuation are left untouched.
_TRAILING_CIRCLE_LIKE = re.compile(r"(?<=[가-힣])[ㄱ-ㅎㅏ-ㅣoO0○◯]{1,3}$")
_TERMINAL_DECORATIVE_SYLLABLE = re.compile(
    r"^(?P<head>.*[가-힣]{3,}[!?！？]{2,})[가-힣ㄱ-ㅎㅏ-ㅣ]$"
)


# PP-OCRv5 (seen with the OpenVINO backend on a real Korean run, 2026-11-15)
# sometimes reads the gap between two Hangul words as a vertical bar:
# "약사의|혼잣말은". A bar between Hangul syllables is never dialogue.
_BAR_BETWEEN_HANGUL = re.compile(r"(?<=[가-힣])\s*[|｜¦]\s*(?=[가-힣])")


def _clean_visual_line(line: str) -> str:
    line = _BAR_BETWEEN_HANGUL.sub(" ", line)
    line = _TRAILING_CIRCLE_LIKE.sub("", line)
    match = _TERMINAL_DECORATIVE_SYLLABLE.match(line)
    if match:
        line = match.group("head")
    return line


def clean_visual_text(text: str) -> str:
    """Remove narrowly identifiable decorative-tail OCR contamination."""
    normalized = normalize_text(text)
    if not normalized:
        return ""
    cleaned = [_clean_visual_line(line) for line in normalized.split("\n")]
    return normalize_text("\n".join(cleaned))


def _clean_observation(observation: OCRObservation) -> OCRObservation:
    raw = normalize_text(observation.text)
    cleaned = clean_visual_text(raw)
    if cleaned == raw:
        return OCRObservation(
            cleaned,
            observation.confidence,
            observation.pts,
            observation.metadata,
        )

    metadata = dict(observation.metadata or {})
    metadata.setdefault("raw_text", raw)
    metadata["visual_tail_cleanup"] = True
    return OCRObservation(cleaned, observation.confidence, observation.pts, metadata)


def _stabilize_vote_from_recent_consensus(
    voted: OCRObservation,
    observations: list[OCRObservation],
    prior_evidence: list[_SegmentEvidence],
    segment: VisualSegment,
    *,
    lookback_seconds: float = 2.0,
    min_similarity: float = 0.90,
    min_confidence_advantage: float = 0.03,
    min_observation_gain: float = 0.05,
) -> OCRObservation:
    """Repair a one-off near-miss only when independent evidence corroborates it.

    A visually busy subtitle can be split into many detector segments. One long
    segment can then dominate the duration-weighted temporal representative even
    when two independent earlier segments agree on a slightly different reading.
    Do not blindly prefer repetition: the repeated reading must be very close to
    the current vote, have clearly stronger average confidence, and a raw sample
    inside the current segment must itself be materially closer to that repeated
    reading than to the current vote. This keeps the correction evidence-based
    and avoids a phrase dictionary or language-specific substitution table.
    """
    voted_text = normalize_text(voted.text)
    if not voted_text or not observations or not prior_evidence:
        return voted

    lower_bound = segment.start - max(0.0, lookback_seconds)
    recent = [
        item
        for item in prior_evidence
        if item.text and item.end >= lower_bound and item.start <= segment.start
    ]
    if len(recent) < 2:
        return voted

    groups: dict[str, list[_SegmentEvidence]] = {}
    for item in recent:
        key = comparison_key(item.text)
        if key:
            groups.setdefault(key, []).append(item)

    voted_key = comparison_key(voted_text)
    contenders: list[tuple[tuple[int, float, float, float], _SegmentEvidence]] = []
    for key, group in groups.items():
        if key == voted_key or len(group) < 2:
            continue

        representative = max(
            group,
            key=lambda item: (item.confidence, item.supporters, item.end),
        )
        candidate_text = normalize_text(representative.text)
        candidate_similarity = text_similarity(voted_text, candidate_text)
        if candidate_similarity < min_similarity:
            continue

        average_confidence = sum(item.confidence for item in group) / len(group)
        if average_confidence < voted.confidence + min_confidence_advantage:
            continue

        corroborated = False
        best_gain = float("-inf")
        for observation in observations:
            observation_text = normalize_text(observation.text)
            if not observation_text:
                continue
            candidate_fit = text_similarity(observation_text, candidate_text)
            voted_fit = text_similarity(observation_text, voted_text)
            gain = candidate_fit - voted_fit
            best_gain = max(best_gain, gain)
            if candidate_fit >= min_similarity and gain >= min_observation_gain:
                corroborated = True

        if not corroborated:
            continue

        contenders.append(
            (
                (
                    len(group),
                    average_confidence,
                    candidate_similarity,
                    best_gain,
                ),
                representative,
            )
        )

    if not contenders:
        return voted

    _, winner = max(contenders, key=lambda pair: pair[0])
    stabilized_text = normalize_text(winner.text)
    metadata = dict(voted.metadata or {})
    metadata.setdefault("pre_temporal_consensus_text", voted_text)
    metadata["temporal_consensus_text"] = stabilized_text
    metadata["temporal_consensus_repair"] = True
    return OCRObservation(
        stabilized_text,
        voted.confidence,
        voted.pts,
        metadata,
    )


def _accepted(
    text: str,
    confidence: float,
    supporters: int,
    duration: float,
    cfg: CueRecognitionConfig,
    *,
    inferred_single_anchor: bool = False,
) -> bool:
    if not text or len(text) < cfg.min_text_length or confidence < cfg.min_confidence:
        return False
    if duration <= 0:
        return False
    if supporters >= cfg.min_supporters:
        return True
    if (
        supporters >= 1
        and confidence >= cfg.single_frame_confidence
        and duration <= cfg.single_frame_max_duration
    ):
        return True
    # When a high-confidence head observation is followed much later by two
    # mutually supporting observations of a different subtitle, that head is
    # not merely a one-frame flash: it owns the interval until the unresolved
    # transition. Treat the interval as temporal corroboration without claiming
    # that there were two physical OCR samples.
    return bool(
        inferred_single_anchor
        and supporters >= 1
        and confidence >= 0.88
        and duration >= 0.45
    )


def _distinct_head_tail_evidence(
    segment: VisualSegment,
    observations: list[OCRObservation],
    voted: OCRObservation,
    cfg: CueRecognitionConfig,
) -> list[_SegmentEvidence] | None:
    tail_text = normalize_text(voted.text)
    if not tail_text or len(observations) < 3:
        return None

    tail_indices = [
        index
        for index, observation in enumerate(observations)
        if normalize_text(observation.text)
        and text_similarity(tail_text, observation.text) >= cfg.support_similarity
    ]
    if len(tail_indices) < cfg.min_supporters:
        return None

    first_tail_index = min(tail_indices)
    if first_tail_index <= 0:
        return None

    prefix = observations[:first_tail_index]
    prefix_nonempty = [obs for obs in prefix if normalize_text(obs.text)]
    if not prefix_nonempty:
        # Blank -> subtitle must remain conservative. This preserves the live
        # "2장" regression: never invent an earlier start before OCR support.
        return None

    head_vote = vote_observations(prefix_nonempty)
    head_text = normalize_text(head_vote.text)
    if not head_text or head_vote.confidence < 0.88:
        return None
    if _temporal_relation(head_text, tail_text, cfg) != "different":
        return None

    head_support = _supporting_observations(
        head_text, prefix_nonempty, cfg.support_similarity
    )
    if not head_support:
        return None

    first_tail = observations[first_tail_index]
    last_head_pts = max(obs.pts for obs in head_support)
    if first_tail.pts <= last_head_pts:
        return None

    # The exact switch lies between the last confirmed old state and the first
    # confirmed new state. Bias the estimate toward the first new observation:
    # this improves an otherwise unreadably short tail cue while remaining
    # conservative about backdating the new subtitle.
    transition = last_head_pts + (first_tail.pts - last_head_pts) * 0.75
    transition = max(segment.start, min(segment.end, transition))
    head_duration = transition - segment.start
    tail_duration = segment.end - transition
    if head_duration <= 0 or tail_duration <= 0:
        return None

    head_supporters = len(head_support)
    head_accepted = _accepted(
        head_text,
        head_vote.confidence,
        head_supporters,
        head_duration,
        cfg,
        inferred_single_anchor=True,
    )
    if not head_accepted:
        return None

    tail_support = _supporting_observations(
        tail_text, observations[first_tail_index:], cfg.support_similarity
    )
    tail_supporters = len(tail_support)
    tail_accepted = _accepted(
        tail_text,
        voted.confidence,
        tail_supporters,
        tail_duration,
        cfg,
    )
    if not tail_accepted:
        return None

    # _temporal_track uses the supporter count to distinguish a durable local
    # state from an uncorroborated pending blip. For the inferred head interval,
    # the elapsed interval itself supplies that corroboration.
    tracked_head_supporters = max(head_supporters, cfg.min_supporters)

    return [
        _SegmentEvidence(
            segment=VisualSegment(
                segment.start,
                transition,
                segment.samples,
                segment.diagnostics,
            ),
            text=head_text,
            confidence=head_vote.confidence,
            supporters=tracked_head_supporters,
            local_accepted=True,
        ),
        _SegmentEvidence(
            segment=VisualSegment(
                transition,
                segment.end,
                segment.samples,
                segment.diagnostics,
            ),
            text=tail_text,
            confidence=voted.confidence,
            supporters=tail_supporters,
            local_accepted=True,
        ),
    ]


def recognize_segments(
    segments: Iterable[VisualSegment],
    ocr: OCREngine,
    *,
    min_confidence: float = 0.20,
    diagnostics=None,
    config: CueRecognitionConfig | None = None,
    release_frames: bool = False,
) -> list[Cue]:
    """Production recognition with sparse intra-segment transition repair.

    The core temporal tracker remains the one in cue_builder. This wrapper adds
    live-QA evidence that the visual detector cannot represent on its own: a
    confirmed subtitle change inside one quiet visual segment, narrow cleanup
    for decorative glyphs swallowed by OCR, and conservative temporal consensus
    repair for a one-off near-miss contradicted by repeated stronger readings.
    """
    cfg = config or CueRecognitionConfig(min_confidence=min_confidence)
    evidence: list[_SegmentEvidence] = []

    for index, segment in enumerate(segments):
        observations = [
            _clean_observation(ocr.recognize(frame)) for frame in segment.samples
        ]
        voted = vote_observations(observations)
        voted = _stabilize_vote_from_recent_consensus(
            voted,
            observations,
            evidence,
            segment,
        )

        if diagnostics is not None:
            diagnostics.record_segment(index, segment, observations, voted)

        split = _distinct_head_tail_evidence(segment, observations, voted, cfg)
        if split is not None:
            if release_frames:
                split = [
                    replace(item, segment=replace(item.segment, samples=[]))
                    for item in split
                ]
            evidence.extend(split)
            continue

        text = normalize_text(voted.text)
        supporting = _supporting_observations(
            text, observations, cfg.support_similarity
        )
        supporters = len(supporting)
        duration = segment.end - segment.start
        local_accepted = _accepted(
            text,
            voted.confidence,
            supporters,
            duration,
            cfg,
        )

        # Preserve the conservative blank->subtitle rule from cue_builder: a
        # tail-only hit may not begin before the first observation that actually
        # supports it.
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

        if release_frames:
            # Temporal tracking only needs spans/text/confidence; retaining
            # the original image samples in each evidence record can pin
            # gigabytes of video frames over a full episode.
            evidence_segment = replace(evidence_segment, samples=[])

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
