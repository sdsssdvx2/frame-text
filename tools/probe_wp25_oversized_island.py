"""Probe a fail-safe geometry fallback only on pathological WP25 OCR frames.

Production currently preserves oversized OCR and routes it to source-frame
review. This research asks a narrower question: when an individual adaptive OCR
attempt is >600 characters or >8 physical lines, can HSE's existing
CaptionIsland geometry isolate exactly one coherent multiline subtitle block?

Normal observations are returned byte-logically unchanged. An oversized
observation is replaced only if the existing geometry filter returns a
non-empty, strictly smaller result that is itself <=600 chars and <=8 lines.
Ambiguous/failed filtering preserves the original oversized observation, so
the existing product review path still catches it. No transcript/reference,
clean control, semantic model, text correction, or paid API is used.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import tempfile
import time

from hardsub_extractor.local_ocr_quality_signals import local_ocr_quality_signals
from hardsub_extractor.models import OCRObservation
from hardsub_extractor.ocr.adaptive import AdaptiveOCREngine
from hardsub_extractor.ocr.caption_islands import filter_caption_island
from hardsub_extractor.ocr.rapidocr_adapter import RapidOCRAdapter
from hardsub_extractor.pipeline import extract
from hardsub_extractor.subtitle_formats import parse_subtitle_bytes
from verify_wp25_exact_pair_ocr import SOURCES, _clip, _download, _AggregateObservationRecorder

MAX_CHARS = 600
MAX_LINES = 8


def _oversized(observation: OCRObservation) -> bool:
    text = observation.text or ""
    lines = [line for line in text.splitlines() if line.strip()]
    return len(text) > MAX_CHARS or len(lines) > MAX_LINES


class OversizedIslandFallback:
    """Use existing island geometry only for already-pathological attempts."""

    def __init__(self, inner: AdaptiveOCREngine):
        self.inner = inner
        self.oversized_attempts = 0
        self.modified_attempts = 0
        self.unresolved_attempts = 0
        self.removed_characters = 0
        self.removed_lines = 0

    def _filter(self, observation: OCRObservation) -> OCRObservation:
        if not _oversized(observation):
            return observation
        self.oversized_attempts += 1
        before_text = observation.text or ""
        before_lines = len([x for x in before_text.splitlines() if x.strip()])
        candidate = filter_caption_island(observation)
        after_text = candidate.text or ""
        after_lines = len([x for x in after_text.splitlines() if x.strip()])
        if (
            candidate is observation
            or not after_text.strip()
            or _oversized(candidate)
            or len(after_text) >= len(before_text)
        ):
            self.unresolved_attempts += 1
            return observation
        self.modified_attempts += 1
        self.removed_characters += len(before_text) - len(after_text)
        self.removed_lines += max(0, before_lines - after_lines)
        metadata = dict(candidate.metadata or {})
        metadata["oversized_caption_island_fallback"] = True
        return OCRObservation(
            candidate.text, candidate.confidence, candidate.pts, metadata,
        )

    def recognize(self, frame):
        # Filter each primary/upscale/CLAHE attempt before confidence voting so
        # unrelated dense scene text cannot dominate the adaptive medoid.
        return self.inner.recognize_with_filter(frame, self._filter)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(root: Path, models: Path, manifest: Path) -> dict:
    if root.exists() or root.is_symlink() or not models.is_dir() or not manifest.is_file():
        raise ValueError("fresh scratch and pinned model bundle required")
    root.mkdir(mode=0o700)
    original = root / "caption-original.webm"
    source_sha = _download(original, SOURCES["caption"])
    clip = root / "caption-40s.webm"
    clip_sha = _clip(original, clip)

    adaptive = AdaptiveOCREngine(RapidOCRAdapter(
        language="en", model_dir=str(models), manifest_path=str(manifest),
    ))
    fallback = OversizedIslandFallback(adaptive)
    recorder = _AggregateObservationRecorder(fallback)
    out = root / "candidate"
    began = time.perf_counter()
    report = extract(str(clip), recorder, out, streaming=True)
    elapsed = time.perf_counter() - began

    kind, cues = parse_subtitle_bytes(
        report.srt_path.read_bytes(), source_url=report.srt_path.name,
    )
    if kind != "srt" or len(cues) != report.cue_count:
        raise RuntimeError("candidate product SRT parse/count mismatch")
    source_before = _sha(clip)
    srt_before = _sha(report.srt_path)
    signals = local_ocr_quality_signals(
        clip, report.srt_path, language="en",
    )
    if _sha(clip) != source_before or _sha(report.srt_path) != srt_before:
        raise RuntimeError("read-only quality check mutated candidate output")
    if _sha(original) != source_sha or _sha(clip) != clip_sha:
        raise RuntimeError("public source changed during fallback probe")

    cue_lengths = [len(c.text or "") for c in cues]
    cue_lines = [len([x for x in (c.text or "").splitlines() if x.strip()]) for c in cues]
    return {
        "schema": 1,
        "source": "Wikimedia WP25 publisher-authored burned-subtitle alternate cut",
        "study_seconds": 40,
        "candidate": "oversized-only existing CaptionIsland fallback",
        "normal_observations_modified_by_rule": False,
        "oversized_attempts": fallback.oversized_attempts,
        "modified_oversized_attempts": fallback.modified_attempts,
        "unresolved_oversized_attempts": fallback.unresolved_attempts,
        "removed_characters_aggregate": fallback.removed_characters,
        "removed_lines_aggregate": fallback.removed_lines,
        "final_observations": recorder.observations,
        "max_final_frame_ocr_chars": recorder.max_chars,
        "max_final_frame_ocr_lines": recorder.max_lines,
        "max_final_frame_detection_boxes": recorder.max_detection_boxes,
        "final_frame_ocr_over_600_chars": recorder.over_600_chars,
        "final_frame_ocr_over_4_lines": recorder.over_4_lines,
        "cues": len(cues),
        "max_cue_chars": max(cue_lengths, default=0),
        "max_cue_lines": max(cue_lines, default=0),
        "cues_over_600_chars": sum(x > MAX_CHARS for x in cue_lengths),
        "cues_over_8_lines": sum(x > MAX_LINES for x in cue_lines),
        "quality_status": signals.get("status"),
        "quality_signal_count": int(signals.get("review_signal_count", 0)),
        "quality_signal_codes": sorted(signals.get("signal_codes", [])),
        "oversized_quality_signal_remains": (
            "oversized_ocr_cue_review" in signals.get("signal_codes", [])
        ),
        "ocr_calls": int(adaptive.total_calls),
        "retry_calls": int(adaptive.retry_calls),
        "segments": int(report.segment_count),
        "elapsed_seconds": round(elapsed, 2),
        "source_and_srt_unchanged_by_audit": True,
        "reference_text_or_timing_used": False,
        "automatic_product_change": False,
        "candidate_promoted": False,
        "paid_model_requests": 0,
        "user_media_read": False,
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
        print("HSE_WP25_OVERSIZED_ISLAND=PUBLIC_SOURCE_NOT_APPROVED")
        return 2
    try:
        result = run(args.scratch, args.model_dir, args.manifest)
        print("HSE_WP25_OVERSIZED_ISLAND_RESULT " + json.dumps(result, sort_keys=True), flush=True)
        return 0
    except Exception as exc:
        print("HSE_WP25_OVERSIZED_ISLAND_HOLD " + json.dumps({
            "error_class": type(exc).__name__,
            "auto_retry": False,
            "quality_proven": False,
        }, sort_keys=True), flush=True)
        return 3


if __name__ == "__main__":
    raise SystemExit(main())
