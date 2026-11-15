"""Opt-in local-video timing refinement from actual subtitle *pixels*.

The scene-change detector provides coarse visual segments, not true caption
edges. After existing OCR is finished, use a compact in-memory glyph-mask
witness from its ALREADY computed accepted boxes to independently locate
the first and final video frames that display that glyph layout.

No model inference, text-generation, network connection, reference SRT,
duration guessing, image export, or new executable is involved. A conservative
failure keeps the original cue. Only supported light-on-dark local captions
are candidates; default OCR and browser capture are never altered.
"""
from __future__ import annotations

from bisect import bisect_left, bisect_right
from contextlib import closing
from dataclasses import dataclass
import math
import subprocess
import numpy as np
import cv2

from ..frame_source import FFmpegPTSFrameSource, probe_video
from ..models import Cue
from .base import OCREngine, comparison_key, text_similarity

BOX_W = 96
BOX_H = 24
FRAME_FPS = 8.0
MAX_EDGE_SEARCH = 1.85
MIN_SAMPLES = 3
MIN_MATCH = 0.72
MAX_WITNESSES = 12000
# Keep several independent visual anchors for a 2-second dialogue while
# avoiding 8 nearly identical masks per second. Repetition within a caption
# need not displace evidence for another caption near the end of the video.
MIN_STABLE_WITNESS_INTERVAL_SECONDS = 0.50
# 2-4 CJK letters need two genuinely distinct source frames for the strict
# pixel match. Only the *second* per-caption witness uses this shorter gap;
# otherwise 0.50s dedup would silently disable sub-half-second CJK support.
MIN_SHORT_CJK_WITNESS_INTERVAL_SECONDS = 0.18
MAX_CUES = 3000
MAX_BOXES = 6
MAX_SCAN_FRAMES = 200
MAX_ANCHOR_ROWS = 256
MAX_ANCHORS_PER_CUE = 4
# Optical evidence may momentarily disappear at a cut or compression glitch.
# Bridging a truly absent caption for a full second would fabricate dialogue.
MAX_UNVERIFIED_BRIDGE_SECONDS = .375
MAX_BATCH_SPAN_SECONDS = 24.0
MAX_BATCH_CUES = 8


@dataclass(frozen=True)
class _GlyphBox:
    x: float
    y: float
    w: float
    h: float
    packed_mask: bytes


@dataclass(frozen=True)
class _Witness:
    pts: float
    key: str
    text: str
    boxes: tuple[_GlyphBox, ...]


def _rect(image: np.ndarray, record: dict) -> tuple[int, int, int, int] | None:
    if not isinstance(record, dict):
        return None
    fields = [record.get(p) for p in ("cx", "cy", "width", "height")]
    if any(type(p) not in (int, float) or not math.isfinite(p) for p in fields):
        return None
    cx, cy, w, h = (float(p) for p in fields)
    if not (0 <= cx <= 1 and 0 <= cy <= 1 and 0 < w <= 1 and 0 < h <= 1):
        return None
    height, width = image.shape[:2]
    x0 = max(0, min(width, round((cx - w * .5) * width)))
    x1 = max(0, min(width, round((cx + w * .5) * width)))
    y0 = max(0, min(height, round((cy - h * .5) * height)))
    y1 = max(0, min(height, round((cy + h * .5) * height)))
    if x1 - x0 < 12 or y1 - y0 < 8:
        return None
    return (x0, y0, x1, y1)


def _bright_map(image: np.ndarray, record: dict) -> np.ndarray | None:
    coords = _rect(image, record)
    if coords is None:
        return None
    a, b, c, d = coords
    gray = cv2.cvtColor(image[b:d, a:c], cv2.COLOR_BGR2GRAY)
    # Use area-averaging to preserve glyph stroke occupancy under changes in
    # original decode dimensions; this is NOT OCR or language-dependent.
    scaled = cv2.resize(gray, (BOX_W, BOX_H), interpolation=cv2.INTER_AREA)
    return (scaled >= 202).astype(np.uint8)


def _capture_witness(frame, observation) -> _Witness | None:
    if not observation.text or observation.metadata is None:
        return None
    rows = observation.metadata.get("detections")
    if not isinstance(rows, list) or not 1 <= len(rows) <= MAX_ANCHOR_ROWS:
        return None
    if not math.isfinite(frame.pts) or frame.pts < 0:
        return None
    # The OCR engine has already restricted rows to geometrically accepted
    # light-on-dark caption candidates. Do not copy whole decoded frames.
    image = frame.image
    if image.ndim != 3 or image.shape[2] != 3:
        return None
    scored = []
    for record in rows:
        if not isinstance(record, dict):
            return None
        n = len(str(record.get("text") or ""))
        if n < 2:
            continue
        mask = _bright_map(image, record)
        if mask is None:
            continue
        pixels = int(np.count_nonzero(mask))
        density = pixels / mask.size
        # A uniform-bright page/background is not a glyph template.
        if not 12 <= pixels <= 700 or not .015 <= density <= .45:
            continue
        score = record.get("score", 0)
        if type(score) not in (int, float) or not math.isfinite(score):
            continue
        item = _GlyphBox(
            float(record["cx"]), float(record["cy"]),
            float(record["width"]), float(record["height"]),
            np.packbits(mask.reshape(-1)).tobytes(),
        )
        scored.append((min(n, 64), float(score), item))
    if not scored:
        return None
    # More distinct long glyph sequences provide a stronger match than
    # selecting punctuation; bounded to six OCR boxes per observation.
    scored.sort(key=lambda v: (v[0], v[1]), reverse=True)
    boxes = tuple(item for _, _, item in scored[:MAX_BOXES])
    return _Witness(
        frame.pts, comparison_key(observation.text), observation.text, boxes,
    )


def _presence(frame, anchor: _Witness) -> float:
    image = frame.image
    values = []
    for box in anchor.boxes:
        record = {
            "cx": box.x, "cy": box.y,
            "width": box.w, "height": box.h,
        }
        observed = _bright_map(image, record)
        if observed is None or observed.shape != (BOX_H, BOX_W):
            return 0.0
        # Bit-packed only in memory: up to 8x smaller than per-pixel uint8
        # witnesses across a full episode, no private image ever written.
        expected = np.unpackbits(
            np.frombuffer(box.packed_mask, dtype=np.uint8),
            count=BOX_H * BOX_W,
        ).reshape(BOX_H, BOX_W)
        intersection = int(np.count_nonzero(observed & expected))
        actual = int(np.count_nonzero(observed))
        original = int(np.count_nonzero(expected))
        dice = 2 * intersection / (actual + original) if actual + original else 0.0
        # More bright background may appear after a scene cut *behind*
        # unchanged white dialogue. A pure Dice ratio undercounts the
        # genuinely persisting anchor glyphs because the denominator grows.
        # Permit a foreground-recall path only if the selected box does NOT
        # look like an almost-white webpage or fullscreen credit card.
        density = actual / observed.size
        recall = intersection / original if original else 0.0
        if density > .52:
            values.append(0.0)
        elif density <= .36:
            values.append(max(dice, .90 * recall))
        else:
            values.append(dice)
    if not values:
        return 0.0
    # A single box is more easily spoofed by animated background/overlays;
    # require all selected glyph words to agree on foreground presence.
    return float(np.median(values))



def _witness_time_index(
    witnesses: list[_Witness],
) -> tuple[list[tuple[float, int, _Witness]], list[float]]:
    """Sort only existing witness references, not their image templates."""
    indexed = sorted(
        ((float(w.pts), i, w) for i, w in enumerate(witnesses)
         if math.isfinite(w.pts)),
        key=lambda row: (row[0], row[1]),
    )
    return indexed, [row[0] for row in indexed]


def _witnesses_near_cue(
    cue: Cue,
    indexed: list[tuple[float, int, _Witness]],
    times: list[float],
) -> list[_Witness]:
    """Bound by time before the existing strict 0.60-second anchor test.

    Restore incoming acquisition order to preserve old tie-break behavior.
    """
    left = bisect_left(times, cue.start - .60)
    right = bisect_right(times, cue.end + .60)
    return [
        row[2] for row in sorted(indexed[left:right], key=lambda x: x[1])
    ]



def _is_short_cjk_caption(key: str) -> bool:
    # Do not change the existing high-safety path for very short Latin UI
    # labels such as "OK", "HP" or "TV". Two independent OCR observations
    # are required only for actual Hangul/kana/ideographic caption text.
    if not 2 <= len(key) <= 4:
        return False
    return any(
        "\u3040" <= char <= "\u30ff"
        or "\u3400" <= char <= "\u9fff"
        or "\uac00" <= char <= "\ud7a3"
        for char in key
    )


def _short_cjk_pixel_match(first: _Witness, second: _Witness) -> bool:
    """Strict same-glyph evidence from two independent OCR timestamps.

    A 2-character OCR phrase is much easier to confuse with interface text
    than a 6-character line. Require both spatial alignment and near-identical
    bit-packed actual foreground masks; text equality alone is insufficient.
    """
    if not first.boxes or len(first.boxes) != len(second.boxes):
        return False
    for a, b in zip(first.boxes, second.boxes):
        if any(
            abs(left - right) > .035
            for left, right in zip(
                (a.x, a.y, a.w, a.h),
                (b.x, b.y, b.w, b.h),
            )
        ):
            return False
        # Bit packing is already produced by _capture_witness; no OCR/model,
        # full-frame copy or external I/O is involved in comparing witnesses.
        left = np.unpackbits(np.frombuffer(a.packed_mask, dtype=np.uint8))
        right = np.unpackbits(np.frombuffer(b.packed_mask, dtype=np.uint8))
        if left.size != BOX_W * BOX_H or right.size != left.size:
            return False
        total = int(np.count_nonzero(left)) + int(np.count_nonzero(right))
        if not total:
            return False
        overlap = int(np.count_nonzero(left & right))
        if (2.0 * overlap) / total < .78:
            return False
    return True


def _short_cjk_anchors(
    cue: Cue, witnesses: list[_Witness], key: str,
) -> tuple[_Witness, ...]:
    """Conservatively admit a 2-4 CJK character cue with corroborated pixels.

    No single-frame short caption is ever corrected. Select from bounded
    observations near cue midpoint; a same-text label at another point
    without matching pixel geometry does not supply a false timing anchor.
    """
    middle = (cue.start + cue.end) * .5
    exact = [w for w in witnesses if w.key == key and w.boxes]
    if len(exact) < 2:
        return ()
    nearby = sorted(exact, key=lambda w: abs(w.pts - middle))[:32]
    nearby.sort(key=lambda w: w.pts)
    for i, earlier in enumerate(nearby):
        for later in nearby[i + 1:]:
            elapsed = later.pts - earlier.pts
            if elapsed > 2.0:
                break
            if elapsed < MIN_SHORT_CJK_WITNESS_INTERVAL_SECONDS:
                continue
            if _short_cjk_pixel_match(earlier, later):
                return (earlier, later)
    return ()


def _choose_anchors(cue: Cue, witnesses: list[_Witness]) -> tuple[_Witness, ...]:
    """Use independently observed early/middle/late glyphs, not one scene.

    Exact text may move during a scene change. One late glyph-box snapshot
    must never silently erase an earlier matching OCR observation.
    """
    target = comparison_key(cue.text)
    if len(target) < 5:
        if not _is_short_cjk_caption(target):
            return ()
        # Short CJK subtitles were previously excluded outright. Accept
        # only independently matching text AND real glyph pixel evidence.
        # Latin UI/watermark labels and one-frame CJK reads stay excluded.
        candidates = [
            item for item in witnesses
            if item.key == target
            and abs(
                item.pts - max(cue.start, min(item.pts, cue.end))
            ) < .60
        ]
        return _short_cjk_anchors(cue, candidates, target)
    candidates = [
        item for item in witnesses
        if item.key and
        abs(item.pts - max(cue.start, min(item.pts, cue.end))) < .60
        and text_similarity(item.text, cue.text) >= .92
    ]
    if not candidates:
        return ()
    ranked = sorted(
        candidates,
        key=lambda w: (len(w.boxes), -abs(w.pts - (cue.start + cue.end) * .5)),
        reverse=True,
    )
    selected = []
    for item in (min(candidates, key=lambda w: w.pts),
                 max(candidates, key=lambda w: w.pts),
                 *ranked):
        if not any(abs(item.pts - other.pts) < .18 for other in selected):
            selected.append(item)
        if len(selected) >= MAX_ANCHORS_PER_CUE:
            break
    return tuple(selected)


def _choose_anchor(cue: Cue, witnesses: list[_Witness]) -> _Witness | None:
    """Compatibility for old single-anchor tests and consumers."""
    selected = _choose_anchors(cue, witnesses)
    return selected[0] if selected else None


def _single_refined(
    cue: Cue,
    anchor: _Witness | tuple[_Witness, ...],
    frames,
) -> tuple[float, float] | None:
    anchors = (anchor,) if isinstance(anchor, _Witness) else anchor
    if not anchors or len(anchors) > MAX_ANCHORS_PER_CUE:
        return None
    samples = []
    for frame in frames:
        if len(samples) >= MAX_SCAN_FRAMES:
            return None
        if not math.isfinite(frame.pts):
            return None
        samples.append((
            float(frame.pts),
            max(_presence(frame, a) for a in anchors),
        ))
    return _refine_scored_samples(cue, anchors, samples)


def _refine_scored_samples(
    cue: Cue,
    anchors: tuple[_Witness, ...],
    samples: list[tuple[float, float]],
) -> tuple[float, float] | None:
    """Match a bounded set of already-computed pixel evidence, not OCR words."""
    if not anchors or len(samples) < MIN_SAMPLES:
        return None
    # Anchor itself must be found inside an observed matched island.
    # Never infer a long duration merely by filling a gap in OCR.
    groups = []
    current = []
    missing = 0
    for at, score in samples:
        if score >= MIN_MATCH:
            if current and at - current[-1] > .265:
                groups.append(current)
                current = []
            current.append(at)
            missing = 0
        elif current:
            missing += 1
            if missing >= 2:
                groups.append(current)
                current = []
                missing = 0
    if current:
        groups.append(current)
    valid = [
        group for group in groups
        if len(group) >= MIN_SAMPLES
        and group[-1] - group[0] >= 0.245
        and any(group[0] - .20 <= a.pts <= group[-1] + .20
                for a in anchors)
    ]
    if not valid:
        return None
    if len(valid) == 1:
        group = valid[0]
        start, last = group[0], group[-1]
    else:
        # Do not silently fill an evidence-free interval merely because
        # the same OCR text appears on BOTH sides. A dialogue can repeat
        # after an intentionally blank shot. Only tolerate a single brief
        # optical glitch (< ~three 8-fps frame intervals).
        valid.sort(key=lambda group: group[0])
        if (len(anchors) < 2 or any(
            right[0] - left[-1] > MAX_UNVERIFIED_BRIDGE_SECONDS
            for left, right in zip(valid, valid[1:])
        )):
            return None
        start, last = valid[0][0], valid[-1][-1]
    end = last + 1 / FRAME_FPS
    if (not 0 <= start < end
            or abs(start - cue.start) > MAX_EDGE_SEARCH + .30
            or abs(end - cue.end) > MAX_EDGE_SEARCH + .30):
        return None
    return (start, end)


@dataclass(frozen=True)
class _ScanPlan:
    index: int
    cue: Cue
    anchors: tuple[_Witness, ...]
    seek: float
    until: float


def _merge_scan_windows(plans: list[_ScanPlan]) -> list[list[_ScanPlan]]:
    """Share one decoder only for nearby, bounded, already-pinned video windows."""
    groups: list[list[_ScanPlan]] = []
    group_start = group_end = 0.0
    for plan in sorted(plans, key=lambda item: (item.seek, item.index)):
        if (
            groups and len(groups[-1]) < MAX_BATCH_CUES
            and plan.seek <= group_end + .125
            and max(plan.until, group_end) - group_start <= MAX_BATCH_SPAN_SECONDS
        ):
            groups[-1].append(plan)
            group_end = max(group_end, plan.until)
        else:
            groups.append([plan])
            group_start = plan.seek
            group_end = plan.until
    return groups



def _ocr_box_layout(observation) -> tuple[tuple[float, float, float, float], ...] | None:
    """Cheap bounded layout witness to protect moving caption text.

    A title can keep identical OCR words through a scene cut while its
    rendered position changes substantially. Do not suppress a fresh glyph
    mask simply because the words and time delta match.
    """
    metadata = observation.metadata
    if not isinstance(metadata, dict):
        return None
    rows = metadata.get("detections")
    if not isinstance(rows, list) or not 1 <= len(rows) <= MAX_ANCHOR_ROWS:
        return None
    layout = []
    for row in rows:
        if not isinstance(row, dict):
            return None
        coords = tuple(row.get(k) for k in ("cx", "cy", "width", "height"))
        if any(
            type(value) not in (int, float) or not math.isfinite(value)
            for value in coords
        ):
            return None
        layout.append(tuple(float(v) for v in coords))
    return tuple(sorted(layout))


def _layout_changed(
    previous: tuple[tuple[float, float, float, float], ...] | None,
    current: tuple[tuple[float, float, float, float], ...] | None,
) -> bool:
    if previous is None or current is None:
        return False
    return (
        len(previous) != len(current)
        or any(
            abs(a - b) > .035
            for before, after in zip(previous, current)
            for a, b in zip(before, after)
        )
    )


def _run_sample_rank(run_id: int) -> int:
    """Stable 64-bit mixing for a bounded sample of >12k caption runs.

    Hash-priority sampling spreads surviving transitions over a whole episode
    instead of deterministic modulo compaction preserving only the beginning
    and the last few minutes. No random seed, user text or disk state needed.
    """
    mask = (1 << 64) - 1
    z = (int(run_id) + 0x9E3779B97F4A7C15) & mask
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & mask
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & mask
    return z ^ (z >> 31)


class LightTemporalWitnessEngine(OCREngine):
    """Wrap the already-selected light caption OCR mode, no extra inference."""

    def __init__(self, inner: OCREngine):
        self.inner = inner
        self.witnesses: list[_Witness] = []
        # A fixed "take the first 12,000" ceiling silently disables optical
        # timing for late episodes/long movies. Retain a deterministic,
        # episode-spanning sparse sample when memory reaches that ceiling.
        self._witness_ordinals: list[int] = []
        self._witness_run_ids: list[int] = []
        self._current_witness_run = 0
        self._witness_seen = 0
        self.witness_sample_stride = 1
        self.witness_compactions = 0
        self.witness_duplicate_skips = 0
        self._last_observed_key: str | None = None
        self._last_captured_key: str | None = None
        self._last_captured_pts: float | None = None
        self._last_captured_layout = None
        # A new subtitle (or a relocated identical one) may initially lack
        # valid OCR box pixels. Keep its priority until the first usable
        # witness is actually stored, even under sparse post-cap sampling.
        self._pending_run_anchor = False
        self._current_run_saved_witnesses = 0
        self.candidate_cues = 0
        self.refined_cues = 0
        self.skipped_cues = 0
        self.scanned_frames = 0
        self.decoder_launches = 0
        self.failed_decoder_batches = 0

    def _compact_witnesses(self) -> None:
        """Prioritize distinct subtitle runs within a fixed image-mask budget.

        Previous modulo-of-frame-ordinal eviction retained as few as 49
        witnesses in each early quarter of a 100k-transition 12k-cap video,
        while retaining almost all of the most recent quarter. The new
        deterministic run-priority sample preserves first/last pixel evidence
        for many distinct captions and spreads unavoidable losses across
        the whole video instead of solely erasing its middle.
        """
        target = max(1, MAX_WITNESSES // 2)
        if not self.witnesses:
            return
        by_run: dict[int, list[int]] = {}
        for index, run_id in enumerate(self._witness_run_ids):
            by_run.setdefault(run_id, []).append(index)
        if len(by_run) >= target:
            latest_run = next(reversed(by_run))
            if target == 1:
                chosen = [by_run[latest_run][-1]]
            else:
                historical = sorted(
                    (run for run in by_run if run != latest_run),
                    key=_run_sample_rank, reverse=True,
                )[:target - 1]
                chosen = [by_run[run][0] for run in historical]
                chosen.append(by_run[latest_run][-1])
        else:
            # One early anchor per subtitle run wins over duplicate frames.
            chosen = [frames[0] for frames in by_run.values()]
            chosen_set = set(chosen)
            for position in (1.0, .5, .25):
                remaining = target - len(chosen)
                if remaining <= 0:
                    break
                extras = []
                for frames in by_run.values():
                    candidate = frames[round((len(frames) - 1) * position)]
                    if candidate not in chosen_set:
                        extras.append(candidate)
                if len(extras) > remaining:
                    extras.sort(
                        key=lambda index: _run_sample_rank(
                            self._witness_run_ids[index] * 5
                            + self._witness_ordinals[index]
                        ),
                        reverse=True,
                    )
                    extras = extras[:remaining]
                chosen.extend(extras)
                chosen_set.update(extras)
        ordered = sorted(chosen)
        self.witnesses = [self.witnesses[index] for index in ordered]
        self._witness_ordinals = [
            self._witness_ordinals[index] for index in ordered
        ]
        self._witness_run_ids = [
            self._witness_run_ids[index] for index in ordered
        ]
        self.witness_sample_stride *= 2
        self.witness_compactions += 1

    def recognize(self, frame):
        result = self.inner.recognize(frame)
        if not result.text:
            # The same words appearing after a blank frame are a new visual
            # caption run, not a duplicate of the previous appearance.
            self._last_observed_key = None
            self._pending_run_anchor = False
            return result
        key = comparison_key(result.text)
        if not key:
            self._last_observed_key = None
            self._pending_run_anchor = False
            return result
        new_run = key != self._last_observed_key
        self._last_observed_key = key
        # Repeating an unchanged subtitle at every 8fps sample consumes the
        # long-video memory budget before later dialogue is reached. Retain
        # periodic independent anchors, but NEVER skip the first observation
        # of a different caption (including after a real blank).
        pts = float(frame.pts)
        layout = _ocr_box_layout(result)
        # In the post-cap sparse path, a scene cut may move the *same*
        # caption. Treat verified layout changes like a first caption sample
        # at BOTH the capture gate and the append gate, otherwise the new
        # mask is computed and silently discarded on non-stride ordinals.
        changed_layout = (
            key == self._last_captured_key
            and _layout_changed(self._last_captured_layout, layout)
        )
        # A transition must not be considered captured until a valid glyph
        # mask is appended. Otherwise a failed first-frame box is followed
        # by a sparse-stride rejection of the ONLY usable later frame.
        # Count a moving-caption run only once while a capture is pending.
        if new_run or (changed_layout and not self._pending_run_anchor):
            self._current_witness_run += 1
            self._current_run_saved_witnesses = 0
            self._pending_run_anchor = True
        preserve_transition = (
            new_run or changed_layout or self._pending_run_anchor
        )
        # The regular .50s duplicate gate suppresses the second observation
        # of a 0.25-0.40s Korean/Japanese/Chinese dialogue. Without this
        # narrowly-scoped exception, _short_cjk_anchors can never obtain its
        # required two independent samples from the real recognize() path.
        # Never relax one-letter CJK or short Latin labels, and never take
        # unlimited 8-fps duplicate masks from the same short phrase.
        second_short_cjk = (
            not preserve_transition
            and self._current_run_saved_witnesses == 1
            and _is_short_cjk_caption(key)
            and key == self._last_captured_key
            and self._last_captured_pts is not None
            and math.isfinite(pts)
            and MIN_SHORT_CJK_WITNESS_INTERVAL_SECONDS
                <= pts - self._last_captured_pts
                < MIN_STABLE_WITNESS_INTERVAL_SECONDS
        )
        if (
            not (preserve_transition or second_short_cjk)
            and key == self._last_captured_key
            and self._last_captured_pts is not None
            and math.isfinite(pts)
            and pts >= self._last_captured_pts
            and pts - self._last_captured_pts < MIN_STABLE_WITNESS_INTERVAL_SECONDS
        ):
            self.witness_duplicate_skips += 1
            return result
        ordinal = self._witness_seen
        self._witness_seen += 1
        # After global memory compaction the ordinary sparse stride is used
        # only for a stable repeated caption; a new caption gets first chance.
        if (
            ordinal % self.witness_sample_stride
            and not (preserve_transition or second_short_cjk)
        ):
            return result
        witness = _capture_witness(frame, result)
        if witness is None:
            return result
        if len(self.witnesses) >= MAX_WITNESSES:
            self._compact_witnesses()
        if (
            ordinal % self.witness_sample_stride == 0
            or preserve_transition or second_short_cjk
        ):
            self._witness_ordinals.append(ordinal)
            self._witness_run_ids.append(self._current_witness_run)
            self.witnesses.append(witness)
            self._last_captured_key = key
            self._last_captured_pts = pts
            self._last_captured_layout = layout
            self._current_run_saved_witnesses += 1
            self._pending_run_anchor = False
        return result

    def __getattr__(self, key):
        return getattr(self.inner, key)

    def refine_timings(
        self, cues: list[Cue], source: str, roi, *,
        check_cancelled=None,
        cancel_requested=None,
    ) -> list[Cue]:
        """Refine by local pixels with one shared bounded video decode per window.

        No original cue is edited. No source is downloaded. A missing glyph,
        decoder failure, overlapping corrected cue, or unsupported long
        optical gap conservatively keeps the original times unchanged.
        """
        if check_cancelled is not None:
            check_cancelled()
        if cancel_requested is not None and not callable(cancel_requested):
            raise TypeError("cancel_requested must be callable")
        if len(cues) > MAX_CUES:
            return cues
        plans = []
        indexed, pts_index = _witness_time_index(self.witnesses)
        for index, cue in enumerate(cues):
            nearby = _witnesses_near_cue(cue, indexed, pts_index)
            anchors = _choose_anchors(cue, nearby)
            if not anchors:
                continue
            self.candidate_cues += 1
            start = max(0., min(cue.start, min(a.pts for a in anchors))
                        - MAX_EDGE_SEARCH)
            until = max(cue.end, max(a.pts for a in anchors)) + MAX_EDGE_SEARCH
            seek = math.floor(start * FRAME_FPS) / FRAME_FPS
            if seek < 3.0:
                seek = 0.0
            if until <= seek or until - seek > MAX_SCAN_FRAMES / FRAME_FPS:
                continue
            plans.append(_ScanPlan(index, cue, anchors, seek, until))

        # Many cue windows belong to the SAME local source. Avoid one new
        # metadata-probe process per optical window.
        groups = _merge_scan_windows(plans)
        source_info = None
        if groups:
            if check_cancelled is not None:
                check_cancelled()
            try:
                source_info = probe_video(
                    source,
                    extra_input_args=["-protocol_whitelist", "file,pipe"],
                    timeout_seconds=25.0,
                )
            except (
                OSError, RuntimeError, ValueError, TypeError, IndexError,
                KeyError, subprocess.SubprocessError,
            ):
                if check_cancelled is not None:
                    check_cancelled()
                # A failed optional shared probe cannot erase first-pass OCR.
                self.failed_decoder_batches += 1
                self.skipped_cues += len(cues)
                return cues
            if check_cancelled is not None:
                check_cancelled()

        proposals: dict[int, tuple[float, float]] = {}
        anchor_by_index = {plan.index: plan.anchors for plan in plans}
        for group in groups:
            if check_cancelled is not None:
                check_cancelled()
            begin = min(item.seek for item in group)
            until = max(item.until for item in group)
            samples: dict[int, list[tuple[float, float]]] = {
                item.index: [] for item in group
            }
            failed = set()
            try:
                # This optional second pass must not hang after the first
                # OCR pass successfully recognized the user's subtitles.
                # Use the already-tested FFmpeg decoder watchdog, bounded
                # metadata probe and local file-only transport. Failed
                # optical evidence preserves the original cue boundaries.
                stream = iter(FFmpegPTSFrameSource(
                    source, roi=roi, sample_fps=FRAME_FPS,
                    start_seconds=begin, max_duration_seconds=until - begin,
                    extra_input_args=["-protocol_whitelist", "file,pipe"],
                    preprobed_info=source_info,
                    decode_timeout_seconds=90.0,
                    cancel_requested=cancel_requested,
                ))
                self.decoder_launches += 1
                with closing(stream):
                    for frame in stream:
                        if check_cancelled is not None:
                            check_cancelled()
                        self.scanned_frames += 1
                        at = float(frame.pts + begin)
                        if not math.isfinite(at):
                            raise ValueError("non-finite frame PTS")
                        for item in group:
                            if not item.seek - .001 <= at <= item.until + .001:
                                continue
                            rows = samples[item.index]
                            if len(rows) >= MAX_SCAN_FRAMES:
                                failed.add(item.index)
                                continue
                            rows.append((
                                at,
                                max(_presence(frame, anchor)
                                    for anchor in item.anchors),
                            ))
                for item in group:
                    if item.index not in failed:
                        candidate = _refine_scored_samples(
                            item.cue, item.anchors, samples[item.index],
                        )
                        if candidate is not None:
                            proposals[item.index] = candidate
            except (OSError, RuntimeError, ValueError, subprocess.SubprocessError):
                if check_cancelled is not None:
                    check_cancelled()
                # A broken batch is not a reason to guess missing subtitle
                # boundaries or overwrite the user's prior subtitles.
                self.failed_decoder_batches += 1
                continue

        result = []
        for index, cue in enumerate(cues):
            if check_cancelled is not None:
                check_cancelled()
            revised = proposals.get(index)
            anchors = anchor_by_index.get(index, ()) if revised else ()
            if revised is None:
                self.skipped_cues += 1
                result.append(cue)
                continue
            a, b = revised
            # Every chosen witness was an accepted OCR observation, not a
            # foreign publisher transcript or model-generated hallucination.
            a = min(a, min(item.pts for item in anchors))
            b = max(b, max(item.pts for item in anchors) + 1 / FRAME_FPS)
            if (abs(a - cue.start) > MAX_EDGE_SEARCH + .30
                    or abs(b - cue.end) > MAX_EDGE_SEARCH + .30
                    or (index > 0 and a < result[-1].end - .005)
                    or (index + 1 < len(cues)
                        and b > cues[index + 1].start + .005)):
                self.skipped_cues += 1
                result.append(cue)
                continue
            self.refined_cues += 1
            result.append(Cue(a, b, cue.text, cue.confidence))
        return result
