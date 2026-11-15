from __future__ import annotations

from dataclasses import dataclass
import cv2
import numpy as np


@dataclass(frozen=True, slots=True)
class TextLineCrop:
    image: np.ndarray
    y0: int
    y1: int
    score: float


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    start = None
    for i, value in enumerate(mask.tolist() + [False]):
        if value and start is None:
            start = i
        elif not value and start is not None:
            out.append((start, i))
            start = None
    return out


def split_subtitle_lines(
    image: np.ndarray,
    *,
    max_lines: int = 2,
    min_band_height: int = 12,
    min_width_fraction: float = 0.15,
    margin_y: int = 5,
    margin_x: int = 10,
) -> list[TextLineCrop]:
    """
    Heuristic line splitter for an already-limited subtitle ROI.

    It does not try to OCR the whole anime frame. It combines:
    - high-frequency edge density,
    - bright/dark stroke contrast,
    - central horizontal occupancy,
    then extracts at most two strongest vertical bands.

    If the heuristic is uncertain, callers should keep the full-ROI fallback.
    """
    if image is None or image.size == 0:
        return []

    h, w = image.shape[:2]
    if h < min_band_height or w < 32:
        return []

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (3, 3), 0)

    # White/yellow subtitles with dark outline are common, but don't require
    # a specific colour. Combine local gradient and extreme luminance pixels.
    gx = cv2.Sobel(blur, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(blur, cv2.CV_32F, 0, 1, ksize=3)
    mag = cv2.magnitude(gx, gy)
    grad_mask = mag >= max(28.0, float(np.percentile(mag, 78)))

    bright = blur >= 205
    dark = blur <= 48

    # Prefer the central 90% where dialogue subtitles normally live.
    x0 = int(round(w * 0.05))
    x1 = int(round(w * 0.95))
    if x1 <= x0:
        x0, x1 = 0, w

    edge_profile = grad_mask[:, x0:x1].mean(axis=1)
    bright_profile = bright[:, x0:x1].mean(axis=1)
    dark_profile = dark[:, x0:x1].mean(axis=1)

    profile = (
        0.62 * edge_profile
        + 0.23 * bright_profile
        + 0.15 * np.minimum(dark_profile, 0.35)
    ).astype(np.float32)

    # Smooth over a few pixels; subtitles form sustained horizontal bands.
    k = max(3, min(11, (h // 18) | 1))
    kernel = np.ones(k, dtype=np.float32) / k
    smooth = np.convolve(profile, kernel, mode="same")

    # Use a robust adaptive threshold. The absolute floor avoids pure
    # background texture being selected in flat scenes.
    med = float(np.median(smooth))
    p75 = float(np.percentile(smooth, 75))
    p90 = float(np.percentile(smooth, 90))
    threshold = max(0.018, med + (p90 - med) * 0.46, p75 * 0.80)
    active = smooth >= threshold

    # Close tiny vertical gaps inside one subtitle line.
    close_len = max(3, min(9, h // 28))
    kernel_v = np.ones((close_len, 1), np.uint8)
    active_u8 = active.astype(np.uint8)[:, None] * 255
    active = (
        cv2.morphologyEx(active_u8, cv2.MORPH_CLOSE, kernel_v)[:, 0] > 0
    )

    bands = []
    for y0, y1 in _runs(active):
        if y1 - y0 < min_band_height:
            continue
        local = grad_mask[y0:y1, x0:x1]
        xs = np.where(local.any(axis=0))[0]
        if xs.size == 0:
            continue
        span = int(xs[-1] - xs[0] + 1)
        if span < int(w * min_width_fraction):
            continue
        score = float(np.mean(smooth[y0:y1])) * (1.0 + span / max(1, w))
        bands.append((score, y0, y1, int(xs[0] + x0), int(xs[-1] + 1 + x0)))

    if not bands:
        return []

    # Keep strongest bands, then restore top-to-bottom order.
    bands = sorted(bands, reverse=True)[:max_lines]
    bands.sort(key=lambda b: b[1])

    result: list[TextLineCrop] = []
    for score, y0, y1, bx0, bx1 in bands:
        yy0 = max(0, y0 - margin_y)
        yy1 = min(h, y1 + margin_y)
        xx0 = max(0, bx0 - margin_x)
        xx1 = min(w, bx1 + margin_x)
        crop = image[yy0:yy1, xx0:xx1]
        if crop.size:
            result.append(TextLineCrop(crop.copy(), yy0, yy1, score))
    return result
