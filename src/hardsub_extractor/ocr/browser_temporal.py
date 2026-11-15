"""Bounded, in-RAM glyph timing evidence for browser-rendered hard subtitles.

A protected/browser-only video cannot be reopened as an ffmpeg input. This
explicit research option snapshots ONLY a low-resolution 1-bit light-glyph
mask while the existing Chromium player already renders the video. No user
cookies, URL, browser network requests, screenshots, transcript or JPEG is
written, uploaded or passed to a separate model. When the bounded memory
allowance is exceeded, all journal data is immediately discarded and
the original unrefined subtitle cues remain intact.
"""
from __future__ import annotations

from bisect import bisect_left, bisect_right
from contextlib import ExitStack
from dataclasses import dataclass
import math

import cv2
import numpy as np

from ..models import Frame, Cue
from .light_temporal import (
    BOX_W, BOX_H, FRAME_FPS, MAX_SCAN_FRAMES, MAX_EDGE_SEARCH,
    MIN_MATCH, _choose_anchors, _refine_scored_samples,
)

MASK_W, MASK_H = 640, 160
BYTES_PER_MASK = MASK_W * MASK_H // 8
MAX_BROWSER_FRAME_COUNT = 16000
MAX_JOURNAL_BYTES = 200 * 1024 * 1024


@dataclass(frozen=True, slots=True)
class _MaskFrame:
    pts: float
    packed: bytes


def _shrink_binary(image: np.ndarray) -> bytes:
    if not isinstance(image, np.ndarray) or image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("non-BGR browser pixel evidence")
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (MASK_W, MASK_H), interpolation=cv2.INTER_AREA)
    bits = np.packbits((resized >= 202).astype(np.uint8).ravel()).tobytes()
    if len(bits) != BYTES_PER_MASK:
        raise ValueError("packed light glyph frame size invalid")
    return bits


def _score_box(mask: np.ndarray, box) -> float:
    x0 = max(0, min(MASK_W, int(round((box.x - box.w / 2) * MASK_W))))
    x1 = max(0, min(MASK_W, int(round((box.x + box.w / 2) * MASK_W))))
    y0 = max(0, min(MASK_H, int(round((box.y - box.h / 2) * MASK_H))))
    y1 = max(0, min(MASK_H, int(round((box.y + box.h / 2) * MASK_H))))
    if x1 - x0 < 2 or y1 - y0 < 2:
        return 0.
    patch = mask[y0:y1, x0:x1]
    if patch.size == 0:
        return 0.
    # Preserve subpixel coverage instead of rounding a 0/1 uint8 map
    # back to integers before measuring thin subtitle strokes.
    scaled = cv2.resize(
        patch.astype(np.float32), (BOX_W, BOX_H),
        interpolation=cv2.INTER_AREA,
    )
    observed = (scaled >= .40).astype(np.uint8)
    reference = np.unpackbits(
        np.frombuffer(box.packed_mask, dtype=np.uint8),
        count=BOX_W * BOX_H,
    ).reshape(BOX_H, BOX_W)
    overlap = int(np.count_nonzero(reference & observed))
    truth = int(np.count_nonzero(reference))
    actual = int(np.count_nonzero(observed))
    if not truth or not actual:
        return 0.
    density = actual / (BOX_W * BOX_H)
    if density > .52:
        return 0.
    dice = 2 * overlap / (truth + actual)
    recall = overlap / truth
    return max(dice, .90 * recall) if density <= .36 else dice


def _presence_from_mask(mask: np.ndarray, witness) -> float:
    values = [_score_box(mask, box) for box in witness.boxes]
    return float(np.median(values)) if values else 0.


class BrowserGlyphJournal:
    """RAM-only read-once frame witness, never a temporary video copy."""

    def __init__(self):
        self.frames: list[_MaskFrame] = []
        self.bytes_kept = 0
        self.incomplete = False
        self.scored_frames = 0
        self.refined_cues = 0
        self.skipped_cues = 0

    def clear(self):
        self.frames.clear()
        self.bytes_kept = 0

    def capture(self, frames):
        # BrowserJPEGFrame.image decodes on every access. Convert it into a
        # regular Frame once so detection/OCR reuse those same decoded pixels.
        # Only streaming mode may use this adapter; no full-frame arrays are
        # retained beyond the detector's bounded current segment samples.
        with ExitStack() as stack:
            iterator = iter(frames)
            # The production browser source is a closable generator,
            # whereas replay-free offline clients may pass a plain iterator.
            # Close only when supported, and still close on cancellation.
            close_iterator = getattr(iterator, "close", None)
            if callable(close_iterator):
                stack.callback(close_iterator)
            last_pts = -1.
            for frame in iterator:
                current = float(frame.pts)
                image = frame.image
                if not math.isfinite(current) or current <= last_pts:
                    self.incomplete = True
                    self.clear()
                last_pts = current
                if not self.incomplete:
                    try:
                        payload = _shrink_binary(image)
                        if (
                            len(self.frames) >= MAX_BROWSER_FRAME_COUNT
                            or self.bytes_kept + len(payload) > MAX_JOURNAL_BYTES
                        ):
                            self.incomplete = True
                            self.clear()
                        else:
                            self.frames.append(_MaskFrame(current, payload))
                            self.bytes_kept += len(payload)
                    except (ValueError, cv2.error):
                        self.incomplete = True
                        self.clear()
                yield Frame(current, image)

    def refine(self, engine, cues: list[Cue]) -> list[Cue]:
        """Use only original in-memory pixels; uncertain cues are unchanged."""
        if self.incomplete or not self.frames or len(cues) > 3000:
            return cues
        times = [s.pts for s in self.frames]
        proposals = {}
        anchors_by_index = {}
        for index, cue in enumerate(cues):
            anchors = _choose_anchors(cue, engine.witnesses)
            if not anchors:
                continue
            anchors_by_index[index] = anchors
            begin = max(0., min(cue.start, min(a.pts for a in anchors))
                        - MAX_EDGE_SEARCH)
            until = max(cue.end, max(a.pts for a in anchors)) + MAX_EDGE_SEARCH
            left = bisect_left(times, begin)
            right = bisect_right(times, until)
            if right - left < 3 or right - left > MAX_SCAN_FRAMES:
                continue
            pairs = []
            for frame in self.frames[left:right]:
                pixels = np.unpackbits(
                    np.frombuffer(frame.packed, dtype=np.uint8),
                    count=MASK_W * MASK_H,
                ).reshape(MASK_H, MASK_W)
                score = max(_presence_from_mask(pixels, anchor)
                            for anchor in anchors)
                pairs.append((frame.pts, score))
                self.scored_frames += 1
            candidate = _refine_scored_samples(cue, anchors, pairs)
            if candidate is not None:
                proposals[index] = candidate
        result = []
        for index, cue in enumerate(cues):
            revised = proposals.get(index)
            anchors = anchors_by_index.get(index, ())
            if revised is None or not anchors:
                result.append(cue)
                self.skipped_cues += 1
                continue
            start, end = revised
            start = min(start, min(a.pts for a in anchors))
            end = max(end, max(a.pts for a in anchors) + 1. / FRAME_FPS)
            if (
                abs(start - cue.start) > MAX_EDGE_SEARCH + .30
                or abs(end - cue.end) > MAX_EDGE_SEARCH + .30
                or (index > 0 and start < result[-1].end - .005)
                or (index + 1 < len(cues) and end > cues[index + 1].start + .005)
            ):
                result.append(cue)
                self.skipped_cues += 1
            else:
                result.append(Cue(start, end, cue.text, cue.confidence))
                self.refined_cues += 1
        return result
