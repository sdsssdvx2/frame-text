"""Robust source-only learned-band candidate for pathological WP25 cues.

Product-side candidate generation uses ONLY normal OCR geometry from the burned
caption source. The exact publisher clean pair is decoded only after the
candidate is produced, as a visual-delta witness that the tighter band still
covers added subtitle pixels. It never contributes text, timing, band choice,
OCR input, or replacement decisions.

The original SRT/ASS remain byte-identical. A separate candidate pair is
produced only for measurement; no product default is changed.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from contextlib import closing
from dataclasses import replace
import hashlib
import json
import math
from pathlib import Path
import statistics

import cv2
import numpy as np

from hardsub_extractor.frame_source import FFmpegPTSFrameSource
from hardsub_extractor.local_ocr_quality_signals import local_ocr_quality_signals
from hardsub_extractor.models import OCRObservation
from hardsub_extractor.ocr.adaptive import AdaptiveOCREngine
from hardsub_extractor.ocr.rapidocr_adapter import RapidOCRAdapter
from hardsub_extractor.pipeline import extract
from hardsub_extractor.subtitle_formats import parse_subtitle_bytes
from hardsub_extractor.tab_capture import roi_for_region
from hardsub_extractor.writers import write_subtitle_pair_exclusive

from probe_wp25_learned_band import (
    BIN_WIDTH, MAX_CHARS, MAX_LINES, MIN_CALIBRATION_OBSERVATIONS,
    MIN_PERSISTENT_OBSERVATIONS, NormalGeometryCollector, _filter_to_band,
    _learn_band, _oversized, _sha, _valid_detection,
)
from probe_wp25_targeted_recovery import _targeted_recovery
from verify_wp25_exact_pair_ocr import (
    SOURCES, _AggregateObservationRecorder, _clip, _download,
)

LOCAL_PROTOCOLS = ["-protocol_whitelist", "file,pipe"]
DELTA_THRESHOLDS = (24, 40)


def _percentile(values: list[float], fraction: float) -> float:
    if not values:
        raise ValueError("empty percentile input")
    if not 0.0 <= fraction <= 1.0:
        raise ValueError("invalid percentile fraction")
    ordered = sorted(float(x) for x in values)
    if len(ordered) == 1:
        return ordered[0]
    pos = fraction * (len(ordered) - 1)
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return ordered[lo]
    weight = pos - lo
    return ordered[lo] * (1.0 - weight) + ordered[hi] * weight


def _learn_robust_band(collector: NormalGeometryCollector):
    if collector.normal_observations < MIN_CALIBRATION_OBSERVATIONS:
        raise RuntimeError("too few normal observations for robust band")
    bins: dict[int, set[int]] = defaultdict(set)
    by_observation: dict[int, list[dict]] = defaultdict(list)
    for observation_id, raw in collector.normal_detections:
        det = _valid_detection(raw)
        if det is None:
            continue
        index = max(0, min(19, int(det["cy"] / BIN_WIDTH)))
        bins[index].add(observation_id)
        by_observation[observation_id].append(det)
    if not bins:
        raise RuntimeError("no valid normal geometry for robust band")

    peak = max(bins, key=lambda idx: (len(bins[idx]), idx))
    peak_count = len(bins[peak])
    if peak_count < MIN_PERSISTENT_OBSERVATIONS:
        raise RuntimeError("robust band peak is not persistent")
    neighbor_threshold = max(
        MIN_PERSISTENT_OBSERVATIONS,
        int(math.ceil(peak_count * 0.65)),
    )
    core_bins = {peak}
    for neighbor in (peak - 1, peak + 1):
        if 0 <= neighbor <= 19 and len(bins.get(neighbor, set())) >= neighbor_threshold:
            core_bins.add(neighbor)

    per_obs_top = []
    per_obs_bottom = []
    heights = []
    used_observations = 0
    for observation_id, detections in by_observation.items():
        core = [
            d for d in detections
            if max(0, min(19, int(d["cy"] / BIN_WIDTH))) in core_bins
        ]
        if not core:
            continue
        used_observations += 1
        per_obs_top.append(min(d["cy"] - 0.70 * d["height"] for d in core))
        per_obs_bottom.append(max(d["cy"] + 0.70 * d["height"] for d in core))
        heights.extend(d["height"] for d in core)

    if used_observations < MIN_PERSISTENT_OBSERVATIONS or not heights:
        raise RuntimeError("robust band core lacks observation support")
    median_height = statistics.median(heights)
    pad = max(0.025, min(0.060, median_height * 0.55))
    top = max(0.08, _percentile(per_obs_top, 0.10) - pad)
    bottom = min(0.96, _percentile(per_obs_bottom, 0.90) + pad)
    if not 0.12 <= bottom - top <= 0.40:
        raise RuntimeError("robust band outside conservative bounds")

    return (top, bottom), {
        "peak_bin": peak,
        "peak_observations": peak_count,
        "core_bins": sorted(core_bins),
        "neighbor_threshold": neighbor_threshold,
        "supported_observations": used_observations,
        "median_detection_height": round(median_height, 5),
        "band_top": round(top, 5),
        "band_bottom": round(bottom, 5),
        "band_height": round(bottom - top, 5),
        "quantile_top": 0.10,
        "quantile_bottom": 0.90,
        "recognized_text_used_for_band": False,
        "reference_text_or_timing_used_for_band": False,
        "clean_control_used_for_band": False,
    }


def _quality(path: Path, video: Path) -> dict:
    kind, cues = parse_subtitle_bytes(path.read_bytes(), source_url=path.name)
    if kind != "srt":
        raise RuntimeError("candidate quality input is not SRT")
    lengths = [len(c.text or "") for c in cues]
    lines = [len([x for x in (c.text or "").splitlines() if x.strip()]) for c in cues]
    signals = local_ocr_quality_signals(video, path, language="en")
    return {
        "cues": len(cues),
        "max_cue_chars": max(lengths, default=0),
        "max_cue_lines": max(lines, default=0),
        "cues_over_600_chars": sum(x > MAX_CHARS for x in lengths),
        "cues_over_8_lines": sum(x > MAX_LINES for x in lines),
        "quality_status": signals.get("status"),
        "quality_signal_codes": sorted(signals.get("signal_codes", [])),
        "review_signal_count": int(signals.get("review_signal_count", 0)),
    }


def _paired_delta_witness(caption: Path, clean: Path, cues, bands: dict[str, tuple[float, float]]):
    roi = roi_for_region("bottom40")
    totals = {
        name: {
            str(threshold): {"delta_pixels": 0, "inside_pixels": 0}
            for threshold in DELTA_THRESHOLDS
        } for name in bands
    }
    paired_frames = 0
    max_gap = 0.0

    flagged = [cue for cue in cues if _oversized(
        OCRObservation(cue.text, cue.confidence, cue.start)
    )]
    if not flagged:
        raise RuntimeError("no pathological cue windows for visual witness")

    for cue in flagged:
        duration = cue.duration()
        a_source = FFmpegPTSFrameSource(
            str(caption), roi=roi, extra_input_args=LOCAL_PROTOCOLS,
            sample_fps=4.0, start_seconds=max(0.0, cue.start),
            max_duration_seconds=max(0.10, duration),
            decode_timeout_seconds=45.0, probe_timeout_seconds=25.0,
        )
        b_source = FFmpegPTSFrameSource(
            str(clean), roi=roi, extra_input_args=LOCAL_PROTOCOLS,
            sample_fps=4.0, start_seconds=max(0.0, cue.start),
            max_duration_seconds=max(0.10, duration),
            decode_timeout_seconds=45.0, probe_timeout_seconds=25.0,
        )
        with closing(iter(a_source)) as aa, closing(iter(b_source)) as bb:
            for fa, fb in zip(aa, bb):
                if fa.image.shape != fb.image.shape:
                    raise RuntimeError("paired visual witness frame shape mismatch")
                gap = abs(fa.pts - fb.pts)
                max_gap = max(max_gap, gap)
                if gap > 0.08:
                    raise RuntimeError("paired visual witness time mismatch")
                diff = cv2.absdiff(fa.image, fb.image)
                gray = diff.max(axis=2)
                height, width = gray.shape
                paired_frames += 1
                for threshold in DELTA_THRESHOLDS:
                    mask = gray > threshold
                    delta = int(mask.sum())
                    for name, (top, bottom) in bands.items():
                        y0 = max(0, min(height, int(math.floor(top * height))))
                        y1 = max(y0 + 1, min(height, int(math.ceil(bottom * height))))
                        x0 = max(0, min(width, int(math.floor(0.10 * width))))
                        x1 = max(x0 + 1, min(width, int(math.ceil(0.90 * width))))
                        inside = int(mask[y0:y1, x0:x1].sum())
                        totals[name][str(threshold)]["delta_pixels"] += delta
                        totals[name][str(threshold)]["inside_pixels"] += inside

    if paired_frames < 2:
        raise RuntimeError("too few paired pathological frames")
    result = {}
    for name, band in bands.items():
        row = {
            "band_top": round(band[0], 5),
            "band_bottom": round(band[1], 5),
            "band_height": round(band[1] - band[0], 5),
        }
        for threshold in DELTA_THRESHOLDS:
            item = totals[name][str(threshold)]
            row[f"coverage_gt{threshold}"] = round(
                item["inside_pixels"] / max(1, item["delta_pixels"]), 5
            )
            row[f"delta_pixels_gt{threshold}"] = item["delta_pixels"]
        result[name] = row
    return {
        "paired_frames": paired_frames,
        "max_pair_time_gap_seconds": round(max_gap, 4),
        "caption_clean_used_only_after_candidate_generation": True,
        "reference_text_used": False,
        "bands": result,
    }


def run(root: Path, models: Path, manifest: Path) -> dict:
    if root.exists() or root.is_symlink() or not models.is_dir() or not manifest.is_file():
        raise ValueError("fresh scratch and pinned model bundle required")
    root.mkdir(mode=0o700)

    caption_original = root / "caption-original.webm"
    clean_original = root / "clean-original.webm"
    caption_sha = _download(caption_original, SOURCES["caption"])
    clean_sha = _download(clean_original, SOURCES["clean"])
    caption_clip = root / "caption-40s.webm"
    clean_clip = root / "clean-40s.webm"
    caption_clip_sha = _clip(caption_original, caption_clip)
    clean_clip_sha = _clip(clean_original, clean_clip)

    adaptive = AdaptiveOCREngine(RapidOCRAdapter(
        language="en", model_dir=str(models), manifest_path=str(manifest),
        inference_engine="openvino",
    ))
    collector = NormalGeometryCollector(adaptive)
    recorder = _AggregateObservationRecorder(collector)
    baseline_report = extract(
        str(caption_clip), recorder, root / "baseline", streaming=True,
    )
    current_band, current_calibration = _learn_band(collector)
    robust_band, robust_calibration = _learn_robust_band(collector)

    kind, cues = parse_subtitle_bytes(
        baseline_report.srt_path.read_bytes(), source_url=baseline_report.srt_path.name,
    )
    if kind != "srt" or len(cues) != baseline_report.cue_count:
        raise RuntimeError("baseline subtitle parse mismatch")
    baseline_quality = _quality(baseline_report.srt_path, caption_clip)
    original_srt_sha = hashlib.sha256(baseline_report.srt_path.read_bytes()).hexdigest()
    original_ass_sha = hashlib.sha256(baseline_report.ass_path.read_bytes()).hexdigest()

    # Candidate generation sees only caption source + robust source-only band.
    recovered, targeted = _targeted_recovery(
        caption_clip, cues, robust_band, models, manifest,
    )
    if len(recovered) != len(cues) or any(
        a.start != b.start or a.end != b.end for a, b in zip(cues, recovered)
    ):
        raise RuntimeError("robust recovery changed cue topology")
    candidate_srt, candidate_ass = write_subtitle_pair_exclusive(
        recovered, root / "robust-recovery-candidate",
    )
    candidate_quality = _quality(candidate_srt, caption_clip)

    if (
        hashlib.sha256(baseline_report.srt_path.read_bytes()).hexdigest() != original_srt_sha
        or hashlib.sha256(baseline_report.ass_path.read_bytes()).hexdigest() != original_ass_sha
    ):
        raise RuntimeError("baseline output changed")
    if len(parse_subtitle_bytes(candidate_ass.read_bytes(), source_url=candidate_ass.name)[1]) != len(recovered):
        raise RuntimeError("candidate ASS count mismatch")

    # Only now decode the clean pair as independent geometry evidence.
    witness = _paired_delta_witness(
        caption_clip, clean_clip, cues,
        {"current": current_band, "robust": robust_band},
    )

    for path, expected in (
        (caption_original, caption_sha), (clean_original, clean_sha),
        (caption_clip, caption_clip_sha), (clean_clip, clean_clip_sha),
    ):
        if _sha(path) != expected:
            raise RuntimeError("publisher source bytes changed")

    current_w = witness["bands"]["current"]
    robust_w = witness["bands"]["robust"]
    geometry_non_regression = (
        robust_w["coverage_gt40"] >= current_w["coverage_gt40"] - 0.03
        and robust_w["coverage_gt24"] >= current_w["coverage_gt24"] - 0.03
    )
    practical = (
        targeted["replaced_cues"] > 0
        and candidate_quality["cues_over_600_chars"] == 0
        and candidate_quality["cues_over_8_lines"] == 0
        and candidate_quality["max_cue_chars"] < baseline_quality["max_cue_chars"]
        and candidate_quality["max_cue_lines"] <= baseline_quality["max_cue_lines"]
        and geometry_non_regression
    )
    return {
        "schema": 1,
        "experiment": "robust source-only learned band for pathological cue sidecar",
        "source": "Wikimedia WP25 exact burned-clean publisher pair",
        "current_calibration": current_calibration,
        "robust_calibration": robust_calibration,
        "baseline": baseline_quality,
        "candidate": candidate_quality,
        "targeted": targeted,
        "visual_delta_witness": witness,
        "robust_band_smaller": (
            robust_band[1] - robust_band[0] < current_band[1] - current_band[0]
        ),
        "visual_geometry_non_regression": bool(geometry_non_regression),
        "candidate_sidecar_practical": bool(practical),
        "candidate_is_not_transcription_ground_truth": True,
        "clean_control_used_for_candidate": False,
        "reference_text_or_timing_used": False,
        "cue_count_and_timing_preserved": True,
        "original_srt_ass_bytes_preserved": True,
        "production_default_modified": False,
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
        print("HSE_WP25_ROBUST_BAND=PUBLIC_SOURCES_NOT_APPROVED")
        return 2
    try:
        result = run(args.scratch, args.model_dir, args.manifest)
        print("HSE_WP25_ROBUST_BAND_RESULT " + json.dumps(result, sort_keys=True), flush=True)
        return 0
    except Exception as exc:
        print("HSE_WP25_ROBUST_BAND_HOLD " + json.dumps({
            "error_class": type(exc).__name__,
            "auto_retry": False,
            "original_output_preserved": True,
        }, sort_keys=True), flush=True)
        return 3


if __name__ == "__main__":
    raise SystemExit(main())
