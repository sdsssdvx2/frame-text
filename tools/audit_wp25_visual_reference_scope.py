"""WP25 visual-reference scope audit using the exact burned/clean publisher pair.

This does NOT run OCR and does NOT score text. It checks whether the separately
published TimedText intervals correspond to real, added bright pixels in the
burned edition relative to the clean edition. Reference timing selects audit
frames, so this is scope/provenance evidence only, never an extraction score.

The two exact publisher files and immutable reference are already pinned by
verify_wp25_exact_pair_ocr. No frame, text, or media artifact is published.
"""
from __future__ import annotations

import argparse
from contextlib import closing
import hashlib
import json
import math
from pathlib import Path

import cv2
import numpy as np

from hardsub_extractor.frame_source import FFmpegPTSFrameSource, NormalizedROI, probe_video
from verify_wp25_exact_pair_ocr import (
    SLICE_SECONDS, SOURCES, _clip, _download, _pair_alignment, _reference,
)

FULL = NormalizedROI(0.0, 0.0, 1.0, 1.0)
FPS = 8.0
MAX_FRAME_GAP = 0.19
_CURRENT_STAGE = "not_started"


def _stage(value: str) -> None:
    global _CURRENT_STAGE
    _CURRENT_STAGE = value



def _sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _nearest(path: Path, second: float):
    seek = float(max(0, math.floor(second)))
    source = iter(FFmpegPTSFrameSource(
        str(path), roi=FULL, sample_fps=FPS,
        start_seconds=seek, max_duration_seconds=min(1.4, second-seek+.6),
    ))
    best = None
    gap = float("inf")
    with closing(source):
        for index, frame in enumerate(source):
            if index >= 16:
                break
            actual = seek + frame.pts
            delta = abs(actual-second)
            if delta < gap:
                best = (actual, frame.image.copy())
                gap = delta
            if actual > second + .16:
                break
    if best is None or gap > MAX_FRAME_GAP:
        raise RuntimeError("no bounded paired frame near audit timestamp")
    return best


def _visual_delta(caption: np.ndarray, clean: np.ndarray) -> dict:
    if caption.shape != clean.shape or caption.ndim != 3 or caption.shape[2] != 3:
        raise ValueError("publisher pair frame geometry mismatch")
    # A tiny blur suppresses codec-block noise but keeps subtitle strokes.
    a = cv2.GaussianBlur(cv2.cvtColor(caption, cv2.COLOR_BGR2GRAY), (3,3), 0)
    b = cv2.GaussianBlur(cv2.cvtColor(clean, cv2.COLOR_BGR2GRAY), (3,3), 0)
    delta = a.astype(np.int16) - b.astype(np.int16)
    height = delta.shape[0]
    upper = delta[:round(height*.55)]
    lower = delta[round(height*.55):]
    def stats(region):
        positive = region.astype(np.float32)
        return {
            "bright_gt24_fraction": round(float(np.mean(positive > 24)), 6),
            "bright_gt40_fraction": round(float(np.mean(positive > 40)), 6),
            "positive_mean": round(float(np.mean(np.maximum(positive, 0))), 4),
            "positive_p99": round(float(np.percentile(np.maximum(positive, 0), 99)), 3),
        }
    top = stats(upper)
    bottom = stats(lower)
    return {
        "top55": top,
        "lower45": bottom,
        "lower_minus_top_bright_gt24": round(
            bottom["bright_gt24_fraction"] - top["bright_gt24_fraction"], 6
        ),
        "lower_minus_top_bright_gt40": round(
            bottom["bright_gt40_fraction"] - top["bright_gt40_fraction"], 6
        ),
    }


def _gap_sample_times(start: float, end: float) -> list[float]:
    duration = end - start
    if duration < .35:
        return []
    # Stay away from subtitle edges/fades. Dense gaps get two independent
    # controls; long gaps get at most four. This is bounded to avoid turning
    # a provenance audit into a video scan.
    count = 1 if duration < .60 else 2 if duration < 1.20 else 3 if duration < 2.0 else 4
    margin = min(.20, duration * .25)
    return [
        float(x) for x in np.linspace(start + margin, end - margin, count)
    ]


def _off_times(cues) -> list[float]:
    times = []
    left = 0.0
    for cue in cues:
        times.extend(_gap_sample_times(left, cue.start))
        left = max(left, cue.end)
    times.extend(_gap_sample_times(left, SLICE_SECONDS))
    if len(times) > 20:
        indexes = np.linspace(0, len(times)-1, 20).round().astype(int)
        times = [times[i] for i in sorted(set(indexes.tolist()))]
    return times


def _summarize(rows) -> dict:
    if not rows:
        return {
            "frames": 0,
            "median_lower_excess_gt24": None,
            "min_lower_excess_gt24": None,
            "median_lower_excess_gt40": None,
            "min_lower_excess_gt40": None,
        }
    values24 = [r["lower_minus_top_bright_gt24"] for r in rows]
    values40 = [r["lower_minus_top_bright_gt40"] for r in rows]
    return {
        "frames": len(rows),
        "median_lower_excess_gt24": round(float(np.median(values24)), 6),
        "min_lower_excess_gt24": round(float(np.min(values24)), 6),
        "median_lower_excess_gt40": round(float(np.median(values40)), 6),
        "min_lower_excess_gt40": round(float(np.min(values40)), 6),
    }


def run(root: Path) -> dict:
    if root.exists() or root.is_symlink():
        raise ValueError("fresh audit scratch required")
    root.mkdir(mode=0o700)
    _stage("reference")
    cues, ref_sha = _reference()
    originals = {k: root/f"{k}.webm" for k in SOURCES}
    source_sha = {}
    for key in SOURCES:
        _stage(f"download:{key}")
        source_sha[key] = _download(originals[key], SOURCES[key])
    clips = {k: root/f"{k}-40s.webm" for k in SOURCES}
    clip_sha = {}
    for key in SOURCES:
        _stage(f"clip:{key}")
        clip_sha[key] = _clip(originals[key], clips[key])
    _stage("pair_alignment")
    alignment = _pair_alignment(clips["caption"], clips["clean"])

    _stage("scope_times")
    active_times = [
        [
            cue.start + (cue.end - cue.start) * fraction
            for fraction in (.25, .50, .75)
        ]
        for cue in cues
    ]
    inactive_times = _off_times(cues)
    if len(active_times) != 13 or any(len(x) != 3 for x in active_times):
        raise RuntimeError("unexpected fixed reference audit grid")

    def audit(second):
        ca, a = _nearest(clips["caption"], second)
        cb, b = _nearest(clips["clean"], second)
        if abs(ca-cb) > .08:
            raise RuntimeError("paired audit frames not time aligned")
        row = _visual_delta(a, b)
        row["pair_time_gap_seconds"] = round(abs(ca-cb), 4)
        return row

    _stage("active_multi_sample")
    active_groups = [[audit(t) for t in times] for times in active_times]
    active = [row for group in active_groups for row in group]
    _stage("inactive_multi_sample")
    inactive = [audit(t) for t in inactive_times]
    active_summary = _summarize(active)
    inactive_summary = _summarize(inactive)

    # Cue-level evidence: at least one of three interior frames has a
    # materially brighter lower band in the burned edition. Thresholds are
    # pixel-fraction provenance gates, not OCR/text thresholds.
    supported_numbers = []
    cue_strengths = []
    for index, group in enumerate(active_groups, 1):
        best24 = max(row["lower_minus_top_bright_gt24"] for row in group)
        best40 = max(row["lower_minus_top_bright_gt40"] for row in group)
        cue_strengths.append((best24, best40))
        if best24 >= .0005 and best40 >= .0002:
            supported_numbers.append(index)
    inactive_control_sufficient = inactive_summary["frames"] >= 3
    inactive_max24 = max(
        (row["lower_minus_top_bright_gt24"] for row in inactive), default=None
    )
    inactive_max40 = max(
        (row["lower_minus_top_bright_gt40"] for row in inactive), default=None
    )
    inactive_discriminates = bool(
        inactive_control_sufficient
        and inactive_max24 is not None and inactive_max40 is not None
        and inactive_max24 < .0005 and inactive_max40 < .0002
    )
    all_reference_cues_visual = len(supported_numbers) == len(cues)
    visual_scope_supported = bool(
        all_reference_cues_visual
        and inactive_control_sufficient
        and inactive_discriminates
    )

    for key in SOURCES:
        if _sha(originals[key]) != source_sha[key]:
            raise RuntimeError("publisher original changed during scope audit")
        if _sha(clips[key]) != clip_sha[key]:
            raise RuntimeError("publisher clip changed during scope audit")

    return {
        "schema": 1,
        "source": "Wikimedia Foundation WP25 alternate cut exact burned/clean pair",
        "reference_sha256": ref_sha,
        "reference_cues": len(cues),
        "publisher_pair_alignment": alignment,
        "active_reference_interior_frames": active_summary,
        "active_reference_cues_with_visual_evidence": len(supported_numbers),
        "active_reference_cue_numbers_with_visual_evidence": supported_numbers,
        "all_reference_cues_show_burned_text_somewhere_inside": all_reference_cues_visual,
        "inactive_gap_frames": inactive_summary,
        "inactive_gap_max_lower_excess_gt24": (
            round(inactive_max24, 6) if inactive_max24 is not None else None
        ),
        "inactive_gap_max_lower_excess_gt40": (
            round(inactive_max40, 6) if inactive_max40 is not None else None
        ),
        "inactive_gap_control_sufficient": inactive_control_sufficient,
        "inactive_gap_discriminates": inactive_discriminates,
        "visual_hardsub_scope_supported": visual_scope_supported,
        "reference_text_used": False,
        "reference_timing_used_only_to_select_scope_audit_frames": True,
        "ocr_model_calls": 0,
        "source_sha256_preserved": True,
        "paid_model_requests": 0,
        "user_media_read": False,
        "product_default_modified": False,
        "release": "HOLD",
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scratch", type=Path, required=True)
    ap.add_argument("--approve-public-sources", action="store_true")
    args = ap.parse_args()
    if not args.approve_public_sources:
        print("HSE_WP25_SCOPE=PUBLIC_SOURCE_NOT_APPROVED")
        return 2
    try:
        result = run(args.scratch)
        print("HSE_WP25_SCOPE_RESULT " + json.dumps(result, sort_keys=True), flush=True)
        return 0
    except Exception as exc:
        print("HSE_WP25_SCOPE_HOLD " + json.dumps({
            "error_class": type(exc).__name__,
            "stage": _CURRENT_STAGE,
            "auto_retry": False,
        }, sort_keys=True), flush=True)
        return 3


if __name__ == "__main__":
    raise SystemExit(main())
