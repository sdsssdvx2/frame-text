"""Optional spatial OCR caption-island selection, preserving unlimited caption rows.

Uses only geometry from the already executed and SHA-pinned OCR engine.
No new model call, no external service, no video data, no reference
transcript. This is deliberately NOT a production-default filter until
independent multi-language, natural-video and no-caption comparison.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
import statistics

from ..models import OCRObservation
from .base import OCREngine


@dataclass(frozen=True, slots=True)
class _LineBox:
    index: int
    left: float
    right: float
    top: float
    bottom: float
    middle_x: float
    glyph_height: float


def _layout(observation: OCRObservation) -> list[_LineBox] | None:
    """Return all original visual lines or no decision when metadata is sparse."""
    lines = observation.text.split("\n")
    if len(lines) < 2 or len(lines) > 64 or not all(lines):
        return None
    metadata = observation.metadata
    detections = metadata.get("detections") if isinstance(metadata, dict) else None
    if not isinstance(detections, list) or len(detections) > 256:
        return None

    buckets: list[list[tuple[float, float, float, float, float]]] = [
        [] for _ in lines
    ]
    for row in detections:
        if not isinstance(row, dict):
            return None
        line = row.get("line")
        if type(line) is not int or not 0 <= line < len(lines):
            return None
        coordinates = [row.get(key) for key in ("cx", "cy", "width", "height")]
        if any(
            type(value) not in (float, int)
            or not math.isfinite(value)
            for value in coordinates
        ):
            return None
        cx, cy, w, h = map(float, coordinates)
        if not (
            0 <= cx <= 1 and 0 <= cy <= 1
            and 0 < w <= 1 and 0 < h <= 1
        ):
            return None
        score = row.get("score", 0.0)
        if type(score) not in (float, int) or not math.isfinite(score):
            return None
        buckets[line].append((cx, cy, w, h, float(score)))
    if not all(buckets):
        return None
    positions: list[_LineBox] = []
    for index, recs in enumerate(buckets):
        left = min(cx - w / 2 for cx, _, w, _, _ in recs)
        right = max(cx + w / 2 for cx, _, w, _, _ in recs)
        top = min(cy - h / 2 for _, cy, _, h, _ in recs)
        bottom = max(cy + h / 2 for _, cy, _, h, _ in recs)
        glyph_height = statistics.median(h for _, _, _, h, _ in recs)
        positions.append(_LineBox(
            index=index, left=left, right=right, top=top, bottom=bottom,
            middle_x=(left + right) / 2, glyph_height=glyph_height,
        ))
    # Confused model metadata should not silently discard dialogue.
    if any(a.top > b.top for a, b in zip(positions, positions[1:])):
        return None
    return positions


def _same_typographic_block(upper: _LineBox, lower: _LineBox) -> bool:
    short = min(upper.glyph_height, lower.glyph_height)
    tall = max(upper.glyph_height, lower.glyph_height)
    gap = lower.top - upper.bottom
    return bool(
        short > 0
        and tall / short <= 1.8
        and -0.65 * short <= gap <= max(0.05, 1.45 * short)
        and abs(upper.middle_x - lower.middle_x) <= 0.20
    )


def filter_caption_island(observation: OCRObservation) -> OCRObservation:
    """Select one visually coherent subtitle block, never blindly last-two.

    A three/four-line caption with consistent typography is kept complete.
    If there is no geometry, a single block, or ambiguous metadata, leave
    the OCR output unchanged. Distinct title/logo lines can be excluded
    from a separate dialogue block. Multiple coherent blocks remain
    ambiguous: keep the original observation unless exactly one multiline
    block has independent visual support. This is a research policy;
    users' original SRT output remains available through the default.
    """
    positions = _layout(observation)
    if positions is None:
        return observation
    blocks: list[list[_LineBox]] = [[positions[0]]]
    for current in positions[1:]:
        if _same_typographic_block(blocks[-1][-1], current):
            blocks[-1].append(current)
        else:
            blocks.append([current])
    if len(blocks) == 1:
        return observation

    # Two disconnected single-line regions have no uniquely supported
    # subtitle block. Their relative width is NOT proof: a top channel
    # watermark may be wider than legitimate one-line dialogue.
    # Likewise, two plausible multiline groups may be simultaneous captions
    # (e.g. bilingual subtitles). Never pick one by text width or y-position.
    # Preserve all rows in those ambiguous cases; default OCR is unchanged.
    plausible = [block for block in blocks if len(block) >= 2]
    if len(plausible) != 1:
        return observation
    best = plausible[0]
    order = {row.index: new for new, row in enumerate(best)}
    retained = [observation.text.split("\n")[row.index] for row in best]
    if not retained or len(retained) == len(positions):
        return observation

    metadata = dict(observation.metadata or {})
    metadata["detections"] = [
        {**row, "line": order[row["line"]]}
        for row in metadata["detections"]
        if row["line"] in order
    ]
    metadata["caption_island_original_lines"] = len(positions)
    metadata["caption_island_kept_lines"] = len(best)
    metadata["caption_island_discarded_lines"] = len(positions) - len(best)
    # The high score of a removed title/logo must not certify the retained
    # subtitle. Use only complete, valid retained OCR box evidence, and
    # never inflate the existing (possibly consensus-bounded) confidence.
    confidence = observation.confidence
    accepted = metadata["detections"]
    weighted = []
    if accepted and {rec["line"] for rec in accepted} == set(range(len(best))):
        for rec in accepted:
            score = rec.get("score")
            chars = len(str(rec.get("text") or "").strip())
            if (
                type(score) not in (int, float)
                or not math.isfinite(score)
                or not 0 <= score <= 1
                or chars <= 0
            ):
                weighted = []
                break
            weighted.append((float(score), chars))
    if weighted:
        remaining = (
            sum(score * chars for score, chars in weighted)
            / sum(chars for _, chars in weighted)
        )
        if remaining < confidence:
            confidence = remaining
            metadata["caption_island_retained_confidence_bounded"] = True
    return OCRObservation(
        "\n".join(retained), confidence,
        observation.pts, metadata,
    )


class CaptionIslandEngine(OCREngine):
    """Experimental per-attempt block selector with bounded adaptive retries."""

    def __init__(self, engine: OCREngine):
        self.engine = engine
        self.dropped_rows = 0
        self.modified_observations = 0

    def _filter_observation(self, original: OCRObservation) -> OCRObservation:
        filtered = filter_caption_island(original)
        if filtered is not original:
            self.modified_observations += 1
            self.dropped_rows += len(original.text.splitlines()) - len(filtered.text.splitlines())
        return filtered

    def recognize(self, frame):
        # Each adaptive attempt must select its own coherent caption block
        # BEFORE retry decisions and the vote. Otherwise already-discarded
        # credits can still decide which recognition wins.
        from .adaptive import AdaptiveOCREngine
        if isinstance(self.engine, AdaptiveOCREngine):
            return self.engine.recognize_with_filter(frame, self._filter_observation)
        return self._filter_observation(self.engine.recognize(frame))

    def __getattr__(self, key):
        return getattr(self.engine, key)

class LightCaptionIslandEngine(OCREngine):
    """Opt-in bright-caption polarity AND coherent-block filter per OCR attempt.

    Reuse the existing independent selectors and the existing 1+2 adaptive
    retry budget. Never run a second model or filter only after the medoid
    vote, where removed scene text could already have won.
    Not a safe universal mode: dark, color or multiple-language blocks can
    be rejected, so this must not become the default without external QA.
    """

    def __init__(self, engine: OCREngine):
        from .adaptive import AdaptiveOCREngine
        if not isinstance(engine, AdaptiveOCREngine):
            raise TypeError("combined filter requires a single adaptive OCR owner")
        self.inner = engine
        self.candidate_frames = 0
        self.discarded_rows = 0
        self.dropped_rows = 0
        self.modified_observations = 0

    def recognize(self, frame):
        from .light_text import select_light_caption_lines
        had_text = False

        def both_filters(observation: OCRObservation) -> OCRObservation:
            nonlocal had_text
            had_text |= bool(observation.text)
            if not observation.text:
                return observation
            bright = select_light_caption_lines(observation, frame.image)
            self.discarded_rows += max(
                0, len(observation.text.splitlines()) - len(bright.text.splitlines()),
            )
            if not bright.text:
                return bright
            selected = filter_caption_island(bright)
            if selected is not bright:
                self.modified_observations += 1
                removed = max(
                    0, len(bright.text.splitlines()) - len(selected.text.splitlines()),
                )
                self.dropped_rows += removed
            return selected

        result = self.inner.recognize_with_filter(frame, both_filters)
        if had_text:
            self.candidate_frames += 1
        return result

    def __getattr__(self, name):
        return getattr(self.inner, name)
