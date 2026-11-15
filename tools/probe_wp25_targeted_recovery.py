"""Targeted non-destructive recovery candidate for pathological WP25 cues.

This experiment preserves the original HSE SRT/ASS. It learns recurring subtitle
geometry from normal OCR observations produced by the unchanged first pass, then
re-opens ONLY cue windows already exported as pathological (>600 chars or >8
physical lines). A bounded number of local frames are re-OCRed with the same
OpenVINO-backed PP-OCRv5 model; only source-learned geometry may filter those
observations. Candidate text is emitted into a separate SRT/ASS pair only when
it becomes non-pathological and shorter. No transcript, clean control, expected
text or reference timing participates. Candidate accuracy is therefore NOT
claimed; this measures whether a safe sidecar recovery workflow is practical.
"""
from __future__ import annotations

import argparse
from contextlib import closing
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import time

from hardsub_extractor.frame_source import FFmpegPTSFrameSource
from hardsub_extractor.local_ocr_quality_signals import local_ocr_quality_signals
from hardsub_extractor.models import OCRObservation
from hardsub_extractor.ocr.adaptive import AdaptiveOCREngine
from hardsub_extractor.ocr.base import vote_observations
from hardsub_extractor.ocr.rapidocr_adapter import RapidOCRAdapter
from hardsub_extractor.pipeline import extract
from hardsub_extractor.subtitle_formats import parse_subtitle_bytes
from hardsub_extractor.tab_capture import roi_for_region
from hardsub_extractor.writers import write_subtitle_pair_exclusive

from probe_wp25_learned_band import (
    MAX_CHARS, MAX_LINES, NormalGeometryCollector, _filter_to_band,
    _learn_band, _oversized, _sha,
)
from verify_wp25_exact_pair_ocr import (
    SOURCES, _AggregateObservationRecorder, _clip, _download,
)

MAX_FLAGGED_CUES = 8
MAX_FRAMES_PER_CUE = 7
TARGET_SAMPLE_FPS = 4.0
LOCAL_PROTOCOLS = ["-protocol_whitelist", "file,pipe"]
_STAGE = "init"


def _stage(name: str) -> None:
    global _STAGE
    _STAGE = name



def _cue_is_pathological(cue) -> bool:
    return _oversized(OCRObservation(cue.text, cue.confidence, cue.start))


def _quality(path: Path, video: Path) -> dict:
    kind, cues = parse_subtitle_bytes(path.read_bytes(), source_url=path.name)
    if kind != "srt":
        raise RuntimeError("recovery quality input is not SRT")
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


def _targeted_recovery(video: Path, cues, band, models: Path, manifest: Path):
    _stage("targeted_identify")
    flagged = [index for index, cue in enumerate(cues) if _cue_is_pathological(cue)]
    print("HSE_WP25_TARGETED_PHASE " + json.dumps({
        "stage": "targeted_identify", "flagged_cues": len(flagged),
    }, sort_keys=True), flush=True)
    if not flagged or len(flagged) > MAX_FLAGGED_CUES:
        raise RuntimeError("unexpected pathological cue count")

    adaptive = AdaptiveOCREngine(RapidOCRAdapter(
        language="en",
        model_dir=str(models),
        manifest_path=str(manifest),
        inference_engine="openvino",
    ))
    roi = roi_for_region("bottom40")
    replacements = {}
    frame_counts = {}
    began = time.perf_counter()

    for index in flagged:
        cue = cues[index]
        duration = cue.duration()
        if not 0.08 <= duration <= 30.0:
            continue
        observations = []
        _stage("targeted_decode")
        source = FFmpegPTSFrameSource(
            str(video),
            roi=roi,
            extra_input_args=LOCAL_PROTOCOLS,
            sample_fps=TARGET_SAMPLE_FPS,
            start_seconds=max(0.0, cue.start),
            max_duration_seconds=max(0.10, duration),
            decode_timeout_seconds=45.0,
            probe_timeout_seconds=25.0,
        )
        count = 0
        with closing(iter(source)) as frames:
            for frame in frames:
                if count >= MAX_FRAMES_PER_CUE:
                    break
                count += 1
                _stage("targeted_ocr")
                observed = adaptive.recognize(frame)
                candidate = _filter_to_band(observed, band)
                if (
                    candidate is not observed
                    and candidate.text.strip()
                    and not _oversized(candidate)
                ):
                    observations.append(candidate)
        frame_counts[str(index + 1)] = count
        print("HSE_WP25_TARGETED_CUE " + json.dumps({
            "cue_number": index + 1, "decoded_frames": count,
            "usable_filtered_observations": len(observations),
        }, sort_keys=True), flush=True)
        if not observations:
            continue
        _stage("targeted_vote")
        voted = vote_observations(observations)
        if (
            not voted.text.strip()
            or _oversized(voted)
            or len(voted.text) >= len(cue.text or "")
        ):
            continue
        replacements[index] = voted

    _stage("targeted_apply_in_memory")
    recovered = list(cues)
    for index, observation in replacements.items():
        recovered[index] = replace(
            recovered[index],
            text=observation.text,
            confidence=min(recovered[index].confidence, observation.confidence),
        )

    return recovered, {
        "flagged_cues": len(flagged),
        "replaced_cues": len(replacements),
        "unchanged_flagged_cues": len(flagged) - len(replacements),
        "sampled_frames_by_cue": frame_counts,
        "targeted_model_calls": int(adaptive.total_calls),
        "targeted_retry_calls": int(adaptive.retry_calls),
        "targeted_elapsed_seconds": round(time.perf_counter() - began, 2),
        "reference_or_clean_control_used": False,
        "original_cue_text_used_for_model_inference": False,
    }


def run(root: Path, models: Path, manifest: Path) -> dict:
    if (
        root.exists() or root.is_symlink()
        or not models.is_dir() or not manifest.is_file()
    ):
        raise ValueError("fresh scratch and pinned model bundle required")
    root.mkdir(mode=0o700)

    _stage("download_source")
    original = root / "caption-original.webm"
    source_sha = _download(original, SOURCES["caption"])
    _stage("make_clip")
    clip = root / "caption-40s.webm"
    clip_sha = _clip(original, clip)

    _stage("baseline_engine")
    # First pass is ordinary product output plus read-only normal-geometry
    # collection. No filtering occurs here.
    adaptive = AdaptiveOCREngine(RapidOCRAdapter(
        language="en",
        model_dir=str(models),
        manifest_path=str(manifest),
        inference_engine="openvino",
    ))
    collector = NormalGeometryCollector(adaptive)
    recorder = _AggregateObservationRecorder(collector)
    _stage("baseline_extract")
    baseline_report = extract(
        str(clip), recorder, root / "baseline", streaming=True,
    )
    _stage("learn_band")
    band, calibration = _learn_band(collector)
    srt_path = baseline_report.srt_path
    ass_path = baseline_report.ass_path
    srt_before = hashlib.sha256(srt_path.read_bytes()).hexdigest()
    ass_before = hashlib.sha256(ass_path.read_bytes()).hexdigest()

    _stage("parse_baseline")
    kind, cues = parse_subtitle_bytes(
        srt_path.read_bytes(), source_url=srt_path.name,
    )
    if kind != "srt" or len(cues) != baseline_report.cue_count:
        raise RuntimeError("baseline SRT parse/count mismatch")
    baseline_quality = _quality(srt_path, clip)
    if not (
        baseline_quality["cues_over_600_chars"] > 0
        or baseline_quality["cues_over_8_lines"] > 0
    ):
        raise RuntimeError("pinned pathological source no longer reproduces")

    _stage("targeted_start")
    recovered, targeted = _targeted_recovery(
        clip, cues, band, models, manifest,
    )
    if len(recovered) != len(cues):
        raise RuntimeError("recovery changed cue count")
    if any(
        a.start != b.start or a.end != b.end
        for a, b in zip(cues, recovered)
    ):
        raise RuntimeError("recovery changed cue timing")

    _stage("write_candidate")
    candidate_base = root / "oversized-recovery-candidate"
    candidate_srt, candidate_ass = write_subtitle_pair_exclusive(
        recovered, candidate_base,
    )
    _stage("candidate_quality")
    candidate_quality = _quality(candidate_srt, clip)
    _stage("candidate_ass_verify")
    _, candidate_ass_cues = parse_subtitle_bytes(
        candidate_ass.read_bytes(), source_url=candidate_ass.name,
    )
    if len(candidate_ass_cues) != len(recovered) or any(
        a.text != b.text
        or abs(a.start - b.start) > .015
        or abs(a.end - b.end) > .015
        for a, b in zip(recovered, candidate_ass_cues)
    ):
        raise RuntimeError("candidate SRT/ASS mismatch")

    _stage("verify_original_unchanged")
    original_unchanged = (
        hashlib.sha256(srt_path.read_bytes()).hexdigest() == srt_before
        and hashlib.sha256(ass_path.read_bytes()).hexdigest() == ass_before
    )
    if not original_unchanged:
        raise RuntimeError("original subtitle pair changed")
    if _sha(original) != source_sha or _sha(clip) != clip_sha:
        raise RuntimeError("source media changed")

    improved = (
        candidate_quality["cues_over_600_chars"]
        < baseline_quality["cues_over_600_chars"]
        or candidate_quality["cues_over_8_lines"]
        < baseline_quality["cues_over_8_lines"]
    )
    usable_sidecar = bool(
        improved
        and targeted["replaced_cues"] > 0
        and candidate_quality["cues_over_600_chars"] == 0
        and candidate_quality["cues_over_8_lines"] == 0
    )

    return {
        "schema": 1,
        "experiment": "post-export targeted oversized cue recovery sidecar",
        "source": "Wikimedia WP25 publisher-authored burned-subtitle alternate cut",
        "study_seconds": 40,
        "backend": "openvino",
        "calibration": calibration,
        "baseline": baseline_quality,
        "candidate": candidate_quality,
        "targeted": targeted,
        "cue_count_and_timing_preserved": True,
        "original_srt_ass_bytes_preserved": True,
        "source_sha256_preserved": True,
        "candidate_pair_published_separately": True,
        "candidate_is_not_transcription_ground_truth": True,
        "candidate_sidecar_practical": usable_sidecar,
        "automatic_replacement_of_original": False,
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
    ap.add_argument("--approve-public-source", action="store_true")
    args = ap.parse_args()
    if not args.approve_public_source:
        print("HSE_WP25_TARGETED_RECOVERY=PUBLIC_SOURCE_NOT_APPROVED")
        return 2
    try:
        result = run(args.scratch, args.model_dir, args.manifest)
        print("HSE_WP25_TARGETED_RECOVERY_RESULT " +
              json.dumps(result, sort_keys=True), flush=True)
        return 0
    except Exception as exc:
        print("HSE_WP25_TARGETED_RECOVERY_HOLD " + json.dumps({
            "error_class": type(exc).__name__,
            "stage": _STAGE,
            "auto_retry": False,
            "original_output_preserved": True,
        }, sort_keys=True), flush=True)
        return 3


if __name__ == "__main__":
    raise SystemExit(main())
