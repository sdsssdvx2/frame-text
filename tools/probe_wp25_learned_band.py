"""Research-only learned subtitle-band fallback for oversized WP25 OCR.

Pass 1 runs unchanged HSE OCR and learns a recurring vertical text band only
from *normal* final OCR observations. No publisher transcript, clean-control
video, expected cue time, recognized string content, or reference answer is
used to select the band. Pass 2 leaves every normal OCR attempt unchanged and
applies the learned geometry only when an adaptive attempt is already
pathological (>600 chars or >8 physical lines).

The candidate is fail-closed: weak/ambiguous calibration or filtering keeps the
original oversized observation so the existing source-frame review signal still
fires. This study does NOT claim transcription accuracy without a valid visual
ground truth. No user media, cloud OCR, paid API, or product default change.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import statistics
import time

from hardsub_extractor.local_ocr_quality_signals import local_ocr_quality_signals
from hardsub_extractor.models import OCRObservation
from hardsub_extractor.ocr.adaptive import AdaptiveOCREngine
from hardsub_extractor.ocr.rapidocr_adapter import RapidOCRAdapter
from hardsub_extractor.pipeline import extract
from hardsub_extractor.subtitle_formats import parse_subtitle_bytes
from verify_wp25_exact_pair_ocr import SOURCES, _clip, _download, _AggregateObservationRecorder

MAX_CHARS = 600
MAX_LINES = 8
NORMAL_MAX_CHARS = 240
NORMAL_MAX_LINES = 4
MIN_SCORE = 0.72
BIN_WIDTH = 0.05
MIN_CALIBRATION_OBSERVATIONS = 12
MIN_PERSISTENT_OBSERVATIONS = 5


def _oversized(observation: OCRObservation) -> bool:
    text = observation.text or ""
    lines = [line for line in text.splitlines() if line.strip()]
    return len(text) > MAX_CHARS or len(lines) > MAX_LINES


def _normal_candidate(observation: OCRObservation) -> bool:
    text = observation.text or ""
    lines = [line for line in text.splitlines() if line.strip()]
    detections = (observation.metadata or {}).get("detections")
    return bool(
        text.strip()
        and len(text) <= NORMAL_MAX_CHARS
        and 1 <= len(lines) <= NORMAL_MAX_LINES
        and isinstance(detections, list)
        and detections
    )


def _valid_detection(det: object) -> dict | None:
    if not isinstance(det, dict):
        return None
    try:
        score = float(det.get("score", 0.0))
        cx = float(det["cx"])
        cy = float(det["cy"])
        width = float(det["width"])
        height = float(det["height"])
        line = int(det.get("line", 0))
    except (KeyError, TypeError, ValueError, OverflowError):
        return None
    if not all(math.isfinite(x) for x in (score, cx, cy, width, height)):
        return None
    if (
        score < MIN_SCORE
        or not 0.06 <= cx <= 0.94
        or not 0.04 <= cy <= 0.98
        or not 0.006 <= width <= 0.95
        or not 0.008 <= height <= 0.35
        or not 0 <= line <= 64
    ):
        return None
    text = det.get("text")
    if not isinstance(text, str) or not text.strip():
        return None
    return {
        "text": text.strip(),
        "score": score,
        "cx": cx,
        "cy": cy,
        "width": width,
        "height": height,
        "line": line,
    }


class NormalGeometryCollector:
    """Pass-through wrapper collecting geometry from final normal observations."""

    def __init__(self, inner: AdaptiveOCREngine):
        self.inner = inner
        self.normal_observations = 0
        self.normal_detections: list[tuple[int, dict]] = []
        self.total_observations = 0
        self.oversized_observations = 0

    def recognize(self, frame):
        obs = self.inner.recognize(frame)
        self.total_observations += 1
        if _oversized(obs):
            self.oversized_observations += 1
            return obs
        if _normal_candidate(obs):
            accepted = [
                item for item in (
                    _valid_detection(x)
                    for x in (obs.metadata or {}).get("detections", [])
                )
                if item is not None
            ]
            if accepted:
                key = self.normal_observations
                self.normal_observations += 1
                self.normal_detections.extend((key, item) for item in accepted)
        return obs


def _learn_band(collector: NormalGeometryCollector) -> tuple[tuple[float, float], dict]:
    if collector.normal_observations < MIN_CALIBRATION_OBSERVATIONS:
        raise RuntimeError("too few normal observations for learned-band calibration")
    if not collector.normal_detections:
        raise RuntimeError("no normal OCR geometry")

    # Count each y-bin at most once per observation so a title card with many
    # boxes cannot dominate a recurring dialogue location.
    bins: dict[int, set[int]] = defaultdict(set)
    by_bin: dict[int, list[dict]] = defaultdict(list)
    for observation_id, det in collector.normal_detections:
        index = max(0, min(19, int(det["cy"] / BIN_WIDTH)))
        bins[index].add(observation_id)
        by_bin[index].append(det)

    ranked = sorted(
        bins,
        key=lambda idx: (len(bins[idx]), idx),  # tie -> lower frame
        reverse=True,
    )
    if not ranked:
        raise RuntimeError("no repeated subtitle band")
    peak = ranked[0]
    peak_count = len(bins[peak])
    if peak_count < MIN_PERSISTENT_OBSERVATIONS:
        raise RuntimeError("subtitle band not persistent enough")

    # Expand only into neighbouring rows observed in a meaningful fraction of
    # distinct normal OCR observations. This is source-only geometry.
    threshold = max(3, int(math.ceil(peak_count * 0.45)))
    lo = hi = peak
    while lo > 0 and len(bins.get(lo - 1, set())) >= threshold:
        lo -= 1
    while hi < 19 and len(bins.get(hi + 1, set())) >= threshold:
        hi += 1

    cluster = [
        det for idx in range(lo, hi + 1)
        for det in by_bin.get(idx, [])
    ]
    if len(cluster) < MIN_PERSISTENT_OBSERVATIONS:
        raise RuntimeError("learned subtitle band cluster too small")

    median_height = statistics.median(x["height"] for x in cluster)
    raw_top = min(x["cy"] - x["height"] * 0.75 for x in cluster)
    raw_bottom = max(x["cy"] + x["height"] * 0.75 for x in cluster)
    # Enough padding for a second subtitle row, but never trust nearly the
    # entire crop. Geometry is relative to HSE's existing bottom40 ROI.
    pad = max(0.04, min(0.12, median_height * 1.4))
    top = max(0.08, raw_top - pad)
    bottom = min(0.96, raw_bottom + pad)
    if not (0.12 <= bottom - top <= 0.62):
        raise RuntimeError("learned band outside conservative bounds")

    return (top, bottom), {
        "normal_observations": collector.normal_observations,
        "normal_detections": len(collector.normal_detections),
        "peak_bin": peak,
        "peak_observations": peak_count,
        "neighbor_threshold": threshold,
        "band_top": round(top, 5),
        "band_bottom": round(bottom, 5),
        "band_height": round(bottom - top, 5),
        "median_detection_height": round(median_height, 5),
        "recognized_text_used_for_band": False,
        "reference_text_or_timing_used_for_band": False,
        "clean_control_used_for_band": False,
    }


def _join_line(items: list[dict]) -> str:
    """Conservative English-only reconstruction for an oversized attempt."""
    parts: list[str] = []
    for item in sorted(items, key=lambda x: x["cx"]):
        word = item["text"].strip()
        if not word:
            continue
        if not parts:
            parts.append(word)
        elif word[0] in ".,!?;:%)]}" or parts[-1].endswith(("(", "[", "{")):
            parts[-1] += word
        else:
            parts.append(" " + word)
    return "".join(parts).strip()


def _filter_to_band(observation: OCRObservation, band: tuple[float, float]) -> OCRObservation:
    if not _oversized(observation):
        return observation
    detections = (observation.metadata or {}).get("detections")
    if not isinstance(detections, list) or not detections:
        return observation
    top, bottom = band
    kept = []
    for raw in detections:
        det = _valid_detection(raw)
        if det is None:
            continue
        if top <= det["cy"] <= bottom and 0.10 <= det["cx"] <= 0.90:
            kept.append(det)
    if not kept:
        return observation

    grouped: dict[int, list[dict]] = defaultdict(list)
    for det in kept:
        grouped[det["line"]].append(det)
    lines = []
    for _, row in sorted(
        grouped.items(),
        key=lambda pair: statistics.median(x["cy"] for x in pair[1]),
    ):
        line = _join_line(row)
        if line:
            lines.append(line)
    candidate_text = "\n".join(lines).strip()
    before_text = observation.text or ""
    candidate = OCRObservation(
        candidate_text,
        sum(x["score"] for x in kept) / len(kept),
        observation.pts,
        {
            "detections": kept,
            "oversized_learned_band_fallback": True,
            "learned_band_top": round(top, 5),
            "learned_band_bottom": round(bottom, 5),
        },
    )
    if (
        not candidate_text
        or len(candidate_text) >= len(before_text)
        or _oversized(candidate)
        or len(lines) > MAX_LINES
    ):
        return observation
    return candidate


class OversizedLearnedBandFallback:
    """Apply a source-learned band only to already-pathological attempts."""

    def __init__(self, inner: AdaptiveOCREngine, band: tuple[float, float]):
        self.inner = inner
        self.band = band
        self.oversized_attempts = 0
        self.modified_attempts = 0
        self.unresolved_attempts = 0
        self.removed_characters = 0

    def _filter(self, observation: OCRObservation) -> OCRObservation:
        if not _oversized(observation):
            return observation
        self.oversized_attempts += 1
        candidate = _filter_to_band(observation, self.band)
        if candidate is observation:
            self.unresolved_attempts += 1
            return observation
        self.modified_attempts += 1
        self.removed_characters += len(observation.text or "") - len(candidate.text or "")
        return candidate

    def recognize(self, frame):
        return self.inner.recognize_with_filter(frame, self._filter)


def _sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _metrics(report, recorder, adaptive, output: Path, video: Path, elapsed: float) -> dict:
    kind, cues = parse_subtitle_bytes(
        report.srt_path.read_bytes(), source_url=report.srt_path.name,
    )
    if kind != "srt" or len(cues) != report.cue_count:
        raise RuntimeError("product SRT parse/count mismatch")
    lengths = [len(c.text or "") for c in cues]
    lines = [len([x for x in (c.text or "").splitlines() if x.strip()]) for c in cues]
    signals = local_ocr_quality_signals(video, report.srt_path, language="en")
    return {
        "cues": len(cues),
        "segments": int(report.segment_count),
        "ocr_calls": int(adaptive.total_calls),
        "retry_calls": int(adaptive.retry_calls),
        "observations": recorder.observations,
        "max_frame_ocr_chars": recorder.max_chars,
        "max_frame_ocr_lines": recorder.max_lines,
        "frame_ocr_over_600_chars": recorder.over_600_chars,
        "max_cue_chars": max(lengths, default=0),
        "max_cue_lines": max(lines, default=0),
        "cues_over_600_chars": sum(x > MAX_CHARS for x in lengths),
        "cues_over_8_lines": sum(x > MAX_LINES for x in lines),
        "quality_status": signals.get("status"),
        "quality_signal_codes": sorted(signals.get("signal_codes", [])),
        "review_signal_count": int(signals.get("review_signal_count", 0)),
        "elapsed_seconds": round(elapsed, 2),
    }


def run(root: Path, models: Path, manifest: Path) -> dict:
    if root.exists() or root.is_symlink() or not models.is_dir() or not manifest.is_file():
        raise ValueError("fresh scratch and pinned models required")
    root.mkdir(mode=0o700)
    original = root / "caption-original.webm"
    source_sha = _download(original, SOURCES["caption"])
    clip = root / "caption-40s.webm"
    clip_sha = _clip(original, clip)

    # Pass 1: unchanged product behavior, only collecting final normal geometry.
    baseline_adaptive = AdaptiveOCREngine(RapidOCRAdapter(
        language="en", model_dir=str(models), manifest_path=str(manifest),
        inference_engine="openvino",
    ))
    collector = NormalGeometryCollector(baseline_adaptive)
    baseline_recorder = _AggregateObservationRecorder(collector)
    began = time.perf_counter()
    baseline_report = extract(
        str(clip), baseline_recorder, root / "baseline", streaming=True,
    )
    baseline_elapsed = time.perf_counter() - began
    band, calibration = _learn_band(collector)
    baseline = _metrics(
        baseline_report, baseline_recorder, baseline_adaptive,
        baseline_report.srt_path, clip, baseline_elapsed,
    )

    # Pass 2: same model/backend; only oversized adaptive attempts can change.
    candidate_adaptive = AdaptiveOCREngine(RapidOCRAdapter(
        language="en", model_dir=str(models), manifest_path=str(manifest),
        inference_engine="openvino",
    ))
    fallback = OversizedLearnedBandFallback(candidate_adaptive, band)
    candidate_recorder = _AggregateObservationRecorder(fallback)
    began = time.perf_counter()
    candidate_report = extract(
        str(clip), candidate_recorder, root / "candidate", streaming=True,
    )
    candidate_elapsed = time.perf_counter() - began
    candidate = _metrics(
        candidate_report, candidate_recorder, candidate_adaptive,
        candidate_report.srt_path, clip, candidate_elapsed,
    )

    if _sha(original) != source_sha or _sha(clip) != clip_sha:
        raise RuntimeError("public source changed during learned-band trial")

    improved = (
        candidate["cues_over_600_chars"] < baseline["cues_over_600_chars"]
        or candidate["cues_over_8_lines"] < baseline["cues_over_8_lines"]
    )
    safe_scope = (
        fallback.modified_attempts > 0
        and fallback.modified_attempts <= fallback.oversized_attempts
        and collector.normal_observations >= MIN_CALIBRATION_OBSERVATIONS
    )
    return {
        "schema": 1,
        "experiment": "normal-observation learned band, oversized attempts only",
        "source": "Wikimedia WP25 publisher-authored burned-subtitle alternate cut",
        "study_seconds": 40,
        "backend": "openvino",
        "calibration": calibration,
        "baseline": baseline,
        "candidate": candidate,
        "oversized_attempts": fallback.oversized_attempts,
        "modified_oversized_attempts": fallback.modified_attempts,
        "unresolved_oversized_attempts": fallback.unresolved_attempts,
        "removed_characters_aggregate": fallback.removed_characters,
        "normal_attempts_modified_by_rule": False,
        "oversized_quality_improved": bool(improved),
        "scope_guard_satisfied": bool(safe_scope),
        "candidate_promoted": False,
        "reference_text_or_timing_used": False,
        "clean_control_used_for_calibration": False,
        "recognized_text_used_for_calibration": False,
        "transcription_accuracy_verified": False,
        "production_default_modified": False,
        "paid_model_requests": 0,
        "user_media_read": False,
        "source_sha256_preserved": True,
        "release": "HOLD",
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scratch", type=Path, required=True)
    ap.add_argument("--model-dir", type=Path, required=True)
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--approve-public-source", action="store_true")
    args = ap.parse_args()
    if not args.approve_public_source:
        print("HSE_WP25_LEARNED_BAND=PUBLIC_SOURCE_NOT_APPROVED")
        return 2
    try:
        result = run(args.scratch, args.model_dir, args.manifest)
        print("HSE_WP25_LEARNED_BAND_RESULT " + json.dumps(result, sort_keys=True), flush=True)
        return 0
    except Exception as exc:
        print("HSE_WP25_LEARNED_BAND_HOLD " + json.dumps({
            "error_class": type(exc).__name__,
            "auto_retry": False,
            "quality_proven": False,
        }, sort_keys=True), flush=True)
        return 3


if __name__ == "__main__":
    raise SystemExit(main())
