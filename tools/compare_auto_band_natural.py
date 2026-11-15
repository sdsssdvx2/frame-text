"""Research-only automatic subtitle-band A/B on a real publisher pair.

The auto-band estimator never reads the publisher reference, OCR transcript,
or clean-control video. It observes only geometry/confidence of OCR boxes from
8 fixed times in the captioned original's lower 45%, then derives a bounded
vertical ROI from rows that recur across samples. The resulting ROI is scored
*afterward* against the same independent 4-cue TimedText and clean editorial
control already used by HSE natural-video QA.

This is inspired by the public MIT hardsub-extractor auto-area strategy, but
implemented independently on HSE's existing FFmpeg/RapidOCR stack. No user
media, cloud OCR, paid API, automatic production selection, or output rewrite.
"""
from __future__ import annotations

import argparse
from contextlib import closing
import hashlib
import json
import math
from pathlib import Path
import tempfile
import time

import numpy as np

from compare_external_engine_exports import evaluate
from verify_commons_natural_hardsub import (
    _get_media, _get_reference, _hse_cues, _probe,
)
from hardsub_extractor.frame_source import (
    FFmpegPTSFrameSource, NormalizedROI, probe_video,
)
from hardsub_extractor.ocr.adaptive import AdaptiveOCREngine
from hardsub_extractor.ocr.rapidocr_adapter import RapidOCRAdapter

SAMPLE_FRACTIONS = tuple(0.06 + 0.88 * i / 7 for i in range(8))
CALIBRATION_ROI = NormalizedROI(0.0, 0.55, 1.0, 0.43)
MIN_BOX_SCORE = 0.60
MIN_CENTER_X = 0.10
MAX_CENTER_X = 0.90


def _sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _raw_box_rows(adapter: RapidOCRAdapter, image: np.ndarray) -> list[tuple[int, int]]:
    """Return only accepted y spans; never expose recognized OCR strings."""
    result = adapter.engine(
        image, use_det=True, use_cls=False, use_rec=True,
        text_score=adapter.text_score,
    )
    boxes = getattr(result, "boxes", None)
    scores = list(getattr(result, "scores", None) or ())
    if boxes is None:
        return []
    rows: list[tuple[int, int]] = []
    height, width = image.shape[:2]
    for index, box in enumerate(boxes):
        if index >= len(scores):
            continue
        try:
            score = float(scores[index])
            arr = np.asarray(box, dtype=float)
        except (TypeError, ValueError, OverflowError):
            continue
        if (
            not math.isfinite(score) or score < MIN_BOX_SCORE
            or arr.ndim != 2 or arr.shape[0] < 2 or arr.shape[1] < 2
            or not np.isfinite(arr[:, :2]).all()
        ):
            continue
        xs, ys = arr[:, 0], arr[:, 1]
        x0, x1 = float(xs.min()), float(xs.max())
        y0, y1 = float(ys.min()), float(ys.max())
        if x1 <= x0 or y1 <= y0:
            continue
        cx = ((x0 + x1) * 0.5) / max(width, 1)
        if not MIN_CENTER_X <= cx <= MAX_CENTER_X:
            continue
        top = max(0, min(height - 1, int(math.floor(y0))))
        bottom = max(top, min(height - 1, int(math.ceil(y1))))
        rows.append((top, bottom))
    return rows


def estimate_auto_band(video: Path, models: Path, manifest: Path) -> tuple[NormalizedROI, dict]:
    info = probe_video(
        str(video), extra_input_args=["-protocol_whitelist", "file,pipe"],
        timeout_seconds=30,
    )
    if (
        info.duration is None or not math.isfinite(info.duration)
        or info.duration < 2 or info.width < 160 or info.height < 120
    ):
        raise ValueError("unsupported auto-band source geometry")
    adapter = RapidOCRAdapter(
        language="en", model_dir=str(models), manifest_path=str(manifest),
    )
    x0, y_cut, crop_w, crop_h = CALIBRATION_ROI.pixels(info.width, info.height)
    del x0  # horizontal coordinates remain full-frame; only y density is scored.
    row_hits = np.zeros(info.height, dtype=np.int16)
    successful = 0
    accepted_boxes = 0

    for fraction in SAMPLE_FRACTIONS:
        start = min(max(0.0, info.duration * fraction), max(0.0, info.duration - 0.30))
        source = FFmpegPTSFrameSource(
            str(video), roi=CALIBRATION_ROI,
            extra_input_args=["-protocol_whitelist", "file,pipe"],
            start_seconds=start, max_duration_seconds=0.30,
            sample_fps=8.0, preprobed_info=info,
            decode_timeout_seconds=20.0,
        )
        with closing(iter(source)) as frames:
            try:
                frame = next(frames)
            except StopIteration:
                continue
        spans = _raw_box_rows(adapter, frame.image)
        # Count each y row at most once per sampled frame. This makes the
        # persistence score mean "seen in N distinct times", not "many boxes
        # happened to overlap one row in a single title card".
        covered = np.zeros(info.height, dtype=bool)
        for top, bottom in spans:
            abs_top = max(0, min(info.height - 1, y_cut + top))
            abs_bottom = max(abs_top, min(info.height - 1, y_cut + bottom))
            covered[abs_top:abs_bottom + 1] = True
        row_hits += covered.astype(np.int16)
        successful += 1
        accepted_boxes += len(spans)

    if successful < 6:
        raise RuntimeError("too few stable auto-band calibration frames")
    threshold = max(2, int(math.ceil(successful * 0.25)))
    eligible = np.flatnonzero(row_hits >= threshold)
    if not len(eligible):
        raise RuntimeError("no persistent lower-frame text band")

    # Find a dense core between 6% and 25% of the full frame. No transcript,
    # clean control, or expected subtitle timing participates in this choice.
    min_h = max(8, int(round(info.height * 0.06)))
    max_h = max(min_h, int(round(info.height * 0.25)))
    best: tuple[float, int, int] | None = None
    for window_h in range(min_h, max_h + 1, 2):
        limit = info.height - window_h
        for top in range(y_cut, limit + 1):
            score = float(row_hits[top:top + window_h].mean())
            # Exact tie -> lower band, a source-only prior for burned captions.
            candidate = (score, top, window_h)
            if best is None or score > best[0] + 1e-12 or (
                abs(score - best[0]) <= 1e-12 and top > best[1]
            ):
                best = candidate
    if best is None or best[0] <= 0:
        raise RuntimeError("auto-band core not measurable")

    _, core_top, core_h = best
    peak = int(row_hits.max())
    up_threshold = max(threshold, int(math.ceil(peak * 0.60)))
    down_threshold = threshold
    max_total = max(core_h, int(round(info.height * 0.32)))
    gap_tolerance = max(4, int(round(info.height * 0.015)))
    top = core_top
    bottom = core_top + core_h - 1

    gap = 0
    for y in range(core_top - 1, y_cut - 1, -1):
        if row_hits[y] >= up_threshold and bottom - y + 1 <= max_total:
            top, gap = y, 0
        else:
            gap += 1
            if gap > gap_tolerance:
                break
    gap = 0
    for y in range(core_top + core_h, info.height):
        if row_hits[y] >= down_threshold and y - top + 1 <= max_total:
            bottom, gap = y, 0
        else:
            gap += 1
            if gap > gap_tolerance:
                break

    pad = max(6, int(round(info.height * 0.02)))
    max_bottom = max(2, int(math.floor(info.height * 0.98)))
    top = max(y_cut, top - pad)
    bottom = min(max_bottom - 1, bottom + pad)
    if bottom <= top:
        raise RuntimeError("auto-band collapsed")
    fraction_y = top / info.height
    fraction_h = (bottom + 1 - top) / info.height
    if not (0.50 <= fraction_y < 0.94 and 0.06 <= fraction_h <= 0.36
            and fraction_y + fraction_h <= 0.981):
        raise RuntimeError("auto-band outside conservative product bounds")
    roi = NormalizedROI(0.04, fraction_y, 0.92, fraction_h)

    return roi, {
        "sample_slots": len(SAMPLE_FRACTIONS),
        "successful_samples": successful,
        "accepted_centered_boxes": accepted_boxes,
        "persistence_threshold_frames": threshold,
        "peak_persistent_samples": peak,
        "core_y_fraction": round(core_top / info.height, 5),
        "core_h_fraction": round(core_h / info.height, 5),
        "roi_y": round(roi.y, 5),
        "roi_h": round(roi.h, 5),
        "roi_bottom": round(roi.y + roi.h, 5),
        "reference_or_clean_control_used_for_selection": False,
        "recognized_text_used_for_selection": False,
    }


def _extract_pair(root: Path, media: dict[str, Path], models: Path,
                  manifest: Path, roi: NormalizedROI | None, label: str,
                  reference) -> dict:
    outputs = {}
    calls = retries = segments = 0
    began = time.perf_counter()
    for source_name in ("caption", "clean"):
        adaptive = AdaptiveOCREngine(RapidOCRAdapter(
            language="en", model_dir=str(models), manifest_path=str(manifest),
        ))
        cues, count = _hse_cues(
            media[source_name], root / f"{label}-{source_name}",
            adaptive, roi=roi,
        )
        outputs[source_name] = cues
        calls += adaptive.total_calls
        retries += adaptive.retry_calls
        segments += count
    score = evaluate(reference, outputs["caption"], outputs["clean"])
    return {
        "score": score,
        "model_calls": int(calls),
        "retry_calls": int(retries),
        "segments": int(segments),
        "elapsed_seconds": round(time.perf_counter() - began, 2),
        "positive_cues": len(outputs["caption"]),
        "clean_control_cues": len(outputs["clean"]),
    }


def run(root: Path, models: Path, manifest: Path) -> dict:
    if root.exists() or root.is_symlink() or not models.is_dir() or not manifest.is_file():
        raise ValueError("fresh scratch and pinned model bundle required")
    root.mkdir(mode=0o700)
    reference, ref_sha = _get_reference()
    media = {name: root / f"{name}.webm" for name in ("caption", "clean")}
    pins = {name: _get_media(media[name], name) for name in media}
    details = {name: _probe(media[name]) for name in media}
    if abs(details["caption"]["seconds"] - details["clean"]["seconds"]) > .1:
        raise RuntimeError("publisher caption/control timing diverged")

    auto_roi, calibration = estimate_auto_band(
        media["caption"], models, manifest,
    )
    baseline = _extract_pair(
        root, media, models, manifest, None, "bottom40", reference,
    )
    candidate = _extract_pair(
        root, media, models, manifest, auto_roi, "auto-band", reference,
    )
    for name, path in media.items():
        if _sha(path) != pins[name]:
            raise RuntimeError("publisher source changed during auto-band A/B")

    b, c = baseline["score"], candidate["score"]
    nonregression = (
        c["exact_visible_text_and_wrap"] >= b["exact_visible_text_and_wrap"]
        and c["strict_both_edges_250ms_and_text_wrap"] >=
            b["strict_both_edges_250ms_and_text_wrap"]
        and c["missed_reference_cues"] <= b["missed_reference_cues"]
        and c["blank_video_false_cues"] <= b["blank_video_false_cues"]
    )
    full_reference_preserved = (
        c["reference_cues"] == 4
        and c["missed_reference_cues"] == 0
        and c["exact_visible_text_and_wrap"] == 4
        and c["strict_both_edges_250ms_and_text_wrap"] == 4
        and c["blank_video_false_cues"] == 0
    )
    return {
        "schema": 1,
        "experiment": "caption-source-only recurring OCR-box auto-band",
        "source": "Wikimedia publisher-authored 15-second caption/clean pair",
        "reference_sha256": ref_sha,
        "reference_cues": len(reference),
        "real_publisher_burned_subtitles": True,
        "matching_clean_editorial_control": True,
        "selection": calibration,
        "baseline_bottom40": baseline,
        "candidate_auto_band": candidate,
        "candidate_nonregression_on_this_source": bool(nonregression),
        "candidate_preserves_all_four_exact_250ms_cues": bool(full_reference_preserved),
        "candidate_worth_second_natural_source": bool(
            nonregression and full_reference_preserved
            and auto_roi.h < 0.40
        ),
        "source_sha256_unchanged": True,
        "reference_used_only_after_roi_and_ocr": True,
        "production_default_modified": False,
        "automatic_roi_enabled": False,
        "paid_model_requests": 0,
        "user_media_read": False,
        "release": "HOLD",
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scratch", type=Path, required=True)
    ap.add_argument("--model-dir", type=Path, required=True)
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--approve-public-sources", action="store_true")
    args = ap.parse_args()
    if not args.approve_public_sources:
        print("HSE_AUTO_BAND=PUBLIC_SOURCE_NOT_APPROVED")
        return 2
    try:
        result = run(args.scratch, args.model_dir, args.manifest)
        print("HSE_AUTO_BAND_RESULT " + json.dumps(result, sort_keys=True), flush=True)
        return 0
    except Exception as exc:
        print("HSE_AUTO_BAND_HOLD " + json.dumps({
            "error_class": type(exc).__name__,
            "auto_retry": False,
            "quality_proven": False,
        }, sort_keys=True), flush=True)
        return 3


if __name__ == "__main__":
    raise SystemExit(main())
