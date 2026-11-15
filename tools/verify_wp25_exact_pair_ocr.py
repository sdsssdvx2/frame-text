"""Exact-publisher-pair natural hard-subtitle benchmark (research HOLD).

Wikimedia Foundation published the same 2025 WP25 alternate cut both with
burned-in English subtitles and without burned subtitles, plus a separate
English TimedText SRT. This study pins all three official resources, verifies
that the non-caption upper frame regions remain the same editorial cut, then
compares HSE production OCR with the existing light-island temporal candidate.

Reference text/times are read only after OCR for scoring. They never select
frames, ROI, thresholds, detector state, retries, or text. No media/text/frame
artifact leaves the disposable scratch. No user media or paid service.
"""
from __future__ import annotations

import argparse
from contextlib import closing
import hashlib
import json
import math
import re
from pathlib import Path
import subprocess
import tempfile
import time
import urllib.request

import cv2
import numpy as np

from compare_external_engine_exports import evaluate, parse_strict_srt
from hardsub_extractor.frame_source import FFmpegPTSFrameSource, NormalizedROI, probe_video
from hardsub_extractor.ocr.adaptive import AdaptiveOCREngine
from hardsub_extractor.ocr.caption_islands import LightCaptionIslandEngine
from hardsub_extractor.ocr.light_temporal import LightTemporalWitnessEngine
from hardsub_extractor.ocr.rapidocr_adapter import RapidOCRAdapter
from hardsub_extractor.pipeline import extract
from hardsub_extractor.models import OCRObservation
from hardsub_extractor.runtime_paths import resolve_binary
from hardsub_extractor.subtitle_formats import parse_subtitle_bytes

SOURCES = {
    "caption": {
        "url": "https://upload.wikimedia.org/wikipedia/commons/3/37/WP25_Anthem_Video_-_alternate_cut_with_subtitles.webm",
        "size": 71_335_214,
        "sha1": "a99b904f98dfed52198282a2df81a5e69c608747",
    },
    "clean": {
        "url": "https://upload.wikimedia.org/wikipedia/commons/1/15/WP25_Anthem_video_-_alternate_cut.webm",
        "size": 31_642_622,
        "sha1": "80349a4028c5a3dfc5dc25354ecdbf200478432b",
    },
}
REFERENCE_URL = (
    "https://commons.wikimedia.org/w/index.php?"
    "title=TimedText:WP25_Anthem_video_-_alternate_cut.webm.en.srt"
    "&oldid=1152949843&action=raw"
)
REFERENCE_OLDID = 1152949843
FULL_REFERENCE_CUES = 27
SLICE_REFERENCE_CUES = 13
SLICE_SECONDS = 40.0
_CURRENT_STAGE = "not_started"


def _stage(value: str) -> None:
    global _CURRENT_STAGE
    _CURRENT_STAGE = value


MAX_REFERENCE = 64_000
TOP_ALIGNMENT_ROI = NormalizedROI(0.0, 0.0, 1.0, 0.55)


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise ValueError("publisher resource redirected")


def _opener():
    return urllib.request.build_opener(NoRedirect())


def _download(path: Path, spec: dict) -> str:
    if path.exists() or path.is_symlink():
        raise FileExistsError("public media scratch exists")
    partial = path.with_suffix(".partial")
    h1, h256 = hashlib.sha1(), hashlib.sha256()
    count = 0
    request = urllib.request.Request(
        spec["url"], headers={
            "User-Agent": "HSE-WP25-exact-pair-research/1.0",
            "Accept-Encoding": "identity",
        },
    )
    try:
        with _opener().open(request, timeout=90) as response, partial.open("xb") as out:
            if response.status != 200 or response.geturl() != spec["url"]:
                raise ValueError("unexpected publisher media response")
            declared = response.headers.get("Content-Length")
            if declared is not None and (
                not declared.isdecimal() or int(declared) != spec["size"]
            ):
                raise ValueError("publisher media length changed")
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                count += len(chunk)
                if count > spec["size"]:
                    raise ValueError("publisher media exceeded exact bound")
                h1.update(chunk)
                h256.update(chunk)
                out.write(chunk)
        if count != spec["size"] or h1.hexdigest() != spec["sha1"]:
            raise ValueError("publisher media SHA-1/size pin failed")
        partial.rename(path)
        return h256.hexdigest()
    except BaseException:
        partial.unlink(missing_ok=True)
        raise


def _reference():
    request = urllib.request.Request(
        REFERENCE_URL, headers={
            "User-Agent": "HSE-WP25-exact-pair-research/1.0",
            "Accept-Encoding": "identity",
        },
    )
    with _opener().open(request, timeout=45) as response:
        if response.status != 200 or response.geturl() != REFERENCE_URL:
            raise ValueError("unexpected immutable TimedText response")
        raw = response.read(MAX_REFERENCE + 1)
    if not 500 < len(raw) <= MAX_REFERENCE:
        raise ValueError("publisher TimedText size changed")
    all_cues = parse_strict_srt(raw)
    if len(all_cues) != FULL_REFERENCE_CUES:
        raise ValueError("publisher TimedText cue count changed")
    selected = [c for c in all_cues if c.start < SLICE_SECONDS and c.end <= SLICE_SECONDS]
    if (
        len(selected) != SLICE_REFERENCE_CUES
        or abs(selected[0].start) > .001
        or abs(selected[-1].end - 39.916) > .001
    ):
        raise ValueError("fixed first-40s reference scope changed")
    return selected, hashlib.sha256(raw).hexdigest()


def _clip(source: Path, target: Path) -> str:
    if target.exists() or target.is_symlink():
        raise FileExistsError("study clip already exists")
    cmd = [
        resolve_binary("ffmpeg"), "-nostdin", "-hide_banner", "-loglevel", "error",
        "-protocol_whitelist", "file,pipe", "-i", str(source),
        "-t", str(SLICE_SECONDS), "-map", "0:v:0", "-an", "-sn", "-dn",
        "-c:v", "copy", "-f", "webm", str(target),
    ]
    p = subprocess.run(
        cmd, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL, timeout=120, check=False,
    )
    if p.returncode != 0 or not target.is_file():
        raise RuntimeError("publisher first-40s stream-copy failed")
    info = probe_video(str(target))
    if (
        info.width != 1920 or info.height != 1080 or info.duration is None
        or not 39.5 <= info.duration <= 40.2
    ):
        raise ValueError("fixed publisher study clip metadata changed")
    h = hashlib.sha256()
    with target.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _pair_alignment(caption: Path, clean: Path) -> dict:
    def frames(path):
        return FFmpegPTSFrameSource(
            str(path), roi=TOP_ALIGNMENT_ROI, sample_fps=1.0,
            max_duration_seconds=39.5,
        )
    correlations = []
    time_gaps = []
    with closing(iter(frames(caption))) as a, closing(iter(frames(clean))) as b:
        for fa, fb in zip(a, b):
            if len(correlations) >= 40:
                break
            time_gaps.append(abs(fa.pts - fb.pts))
            aa = cv2.resize(
                cv2.cvtColor(fa.image, cv2.COLOR_BGR2GRAY), (96, 54),
                interpolation=cv2.INTER_AREA,
            ).astype(np.float32).ravel()
            bb = cv2.resize(
                cv2.cvtColor(fb.image, cv2.COLOR_BGR2GRAY), (96, 54),
                interpolation=cv2.INTER_AREA,
            ).astype(np.float32).ravel()
            aa -= aa.mean(); bb -= bb.mean()
            denom = float(np.linalg.norm(aa) * np.linalg.norm(bb))
            corr = float(np.dot(aa, bb) / denom) if denom > 1e-6 else 1.0
            if not math.isfinite(corr):
                raise RuntimeError("non-finite publisher-pair correlation")
            correlations.append(corr)
    median = float(np.median(correlations)) if correlations else None
    p10 = float(np.percentile(correlations, 10)) if correlations else None
    diagnostic = {
        "aligned_samples": len(correlations),
        "max_sample_time_gap_seconds": round(max(time_gaps, default=99), 4),
        "top55_median_correlation": round(median, 5) if median is not None else None,
        "top55_p10_correlation": round(p10, 5) if p10 is not None else None,
    }
    print("HSE_WP25_ALIGNMENT_DIAG " + json.dumps(diagnostic, sort_keys=True), flush=True)
    if len(correlations) < 35 or max(time_gaps, default=99) > .08:
        raise RuntimeError("publisher pair sample clocks are not aligned")
    if median < .92 or p10 < .78:
        raise RuntimeError("publisher clean/burned pair is not the same visual edit")
    return {
        "aligned_samples": len(correlations),
        "top55_median_correlation": round(median, 5),
        "top55_p10_correlation": round(p10, 5),
        "max_sample_time_gap_seconds": round(max(time_gaps), 4),
    }


def _run_mode(root: Path, clips: dict, models: Path, manifest: Path, mode: str, truth):
    measured = {}
    began = time.perf_counter()
    for label in ("caption", "clean"):
        _stage(f"{mode}:{label}:engine_init")
        base = AdaptiveOCREngine(RapidOCRAdapter(
            language="en", model_dir=str(models),
            manifest_path=str(manifest),
        ))
        if mode == "production":
            engine, refiner = base, None
        elif mode == "light-island-temporal":
            engine = LightTemporalWitnessEngine(LightCaptionIslandEngine(base))
            refiner = engine.refine_timings
        else:
            raise ValueError("unknown fixed research mode")
        out = root / f"{mode}-{label}"
        _stage(f"{mode}:{label}:extract")
        report = extract(
            str(clips[label]), engine, out, streaming=True,
            timing_refiner=refiner,
        )
        _stage(f"{mode}:{label}:parse_srt")
        observed = parse_strict_srt(report.srt_path.read_bytes())
        _stage(f"{mode}:{label}:parsed")
        measured[label] = {
            "cues": observed,
            "segments": int(report.segment_count),
            "ocr_calls": int(base.total_calls),
            "retry_calls": int(base.retry_calls),
            "optical_scanned_frames": int(getattr(engine, "scanned_frames", 0)),
            "optical_refined_cues": int(getattr(engine, "refined_cues", 0)),
        }
    elapsed = time.perf_counter() - began
    _stage(f"{mode}:score")
    score = evaluate(truth, measured["caption"]["cues"], measured["clean"]["cues"])
    return {
        "elapsed_seconds": round(elapsed, 2),
        "positive_segments": measured["caption"]["segments"],
        "control_segments": measured["clean"]["segments"],
        "model_calls": measured["caption"]["ocr_calls"] + measured["clean"]["ocr_calls"],
        "retry_calls": measured["caption"]["retry_calls"] + measured["clean"]["retry_calls"],
        "optical_scanned_frames": measured["caption"]["optical_scanned_frames"],
        "optical_refined_cues": measured["caption"]["optical_refined_cues"],
        "score": score,
    }


STRICT_SRT_FAILURES = {
    "SRT must be UTF-8": "encoding",
    "reference/positive SRT must contain cues": "empty",
    "truncated SRT cue": "truncated_block",
    "cue without timing or text": "missing_timing_or_text",
    "non-SRT block detected; silently dropping cues forbidden": "non_srt_block",
    "non-positive SRT duration": "non_positive_duration",
    "missing or invalid SRT cue body": "invalid_body",
    "subtitle cue timestamps not ordered": "unordered_start_times",
    "too many subtitle cues": "too_many_cues",
}


def _srt_structure_stats(data: bytes, report, base) -> dict:
    """Aggregate-only diagnostics; never log OCR words."""
    try:
        text = data.decode("utf-8-sig")
        decode_ok = True
    except UnicodeDecodeError:
        text = ""
        decode_ok = False
    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    blocks = [
        block for block in re.split(r"\n[ \t]*\n", normalized.strip())
        if block
    ]
    body_lengths = []
    for block in blocks:
        lines = block.split("\n")
        if lines and lines[0].strip().isdigit():
            lines = lines[1:]
        if len(lines) >= 2:
            body_lengths.append(len("\n".join(x.strip() for x in lines[1:]).strip()))
    return {
        "bytes": len(data),
        "utf8_decode_ok": decode_ok,
        "nonempty_blocks": len(blocks),
        "max_body_chars_loose": max(body_lengths, default=0),
        "reported_cues": int(report.cue_count),
        "reported_segments": int(report.segment_count),
        "ocr_calls": int(base.total_calls),
        "retry_calls": int(base.retry_calls),
    }


class _AggregateObservationRecorder:
    """Pass through OCR while retaining only aggregate geometry/text sizes.

    No OCR text, image, box coordinates, or publisher content is logged.
    This isolates whether one pathological cue is already oversized at
    frame-level OCR or is created later by temporal cue tracking.
    """
    def __init__(self, inner):
        self.inner = inner
        self.observations = 0
        self.nonempty = 0
        self.max_chars = 0
        self.max_lines = 0
        self.max_detection_boxes = 0
        self.over_240_chars = 0
        self.over_600_chars = 0
        self.over_4_lines = 0

    def recognize(self, frame):
        observed = self.inner.recognize(frame)
        if not isinstance(observed, OCRObservation):
            raise TypeError("unexpected OCR observation type")
        self.observations += 1
        text = observed.text or ""
        if text.strip():
            self.nonempty += 1
        chars = len(text)
        lines = len(text.splitlines()) if text else 0
        boxes = (observed.metadata or {}).get("detections", [])
        box_count = len(boxes) if isinstance(boxes, list) else 0
        self.max_chars = max(self.max_chars, chars)
        self.max_lines = max(self.max_lines, lines)
        self.max_detection_boxes = max(self.max_detection_boxes, box_count)
        self.over_240_chars += int(chars > 240)
        self.over_600_chars += int(chars > 600)
        self.over_4_lines += int(lines > 4)
        return observed


def _production_caption_probe(
    root: Path, clip: Path, models: Path, manifest: Path,
) -> dict:
    """Reproduce only the first production/burned branch, then stop.

    This intentionally avoids clean-control and candidate OCR so a pre-score
    failure can be localized without repeating the full four-path benchmark.
    No reference text or timings enter extraction or validation.
    """
    _stage("production:caption:engine_init")
    base = AdaptiveOCREngine(RapidOCRAdapter(
        language="en", model_dir=str(models), manifest_path=str(manifest),
    ))
    recorder = _AggregateObservationRecorder(base)
    out = root / "production-caption-probe"
    _stage("production:caption:extract")
    began = time.perf_counter()
    report = extract(str(clip), recorder, out, streaming=True)
    elapsed = time.perf_counter() - began
    _stage("production:caption:parse_srt")
    raw_srt = report.srt_path.read_bytes()
    try:
        observed = parse_strict_srt(raw_srt)
    except ValueError as exc:
        stats = _srt_structure_stats(raw_srt, report, base)
        stats.update({
            "ocr_observations": recorder.observations,
            "nonempty_ocr_observations": recorder.nonempty,
            "max_frame_ocr_chars": recorder.max_chars,
            "max_frame_ocr_lines": recorder.max_lines,
            "max_frame_detection_boxes": recorder.max_detection_boxes,
            "frame_ocr_over_240_chars": recorder.over_240_chars,
            "frame_ocr_over_600_chars": recorder.over_600_chars,
            "frame_ocr_over_4_lines": recorder.over_4_lines,
        })
        # Also characterize written cues without exposing their text. A frame
        # can be clean while temporal tracking still creates an oversized cue.
        try:
            permissive_kind, permissive_cues = parse_subtitle_bytes(
                raw_srt, source_url=report.srt_path.name,
            )
        except Exception:
            permissive_kind, permissive_cues = "unreadable", []
        cue_lengths = [len(c.text or "") for c in permissive_cues]
        cue_lines = [len((c.text or "").splitlines()) for c in permissive_cues]
        stats.update({
            "permissive_format": permissive_kind,
            "permissive_cues": len(permissive_cues),
            "cue_bodies_over_240_chars": sum(x > 240 for x in cue_lengths),
            "cue_bodies_over_600_chars": sum(x > 600 for x in cue_lengths),
            "cue_bodies_over_4_lines": sum(x > 4 for x in cue_lines),
            "max_cue_body_chars": max(cue_lengths, default=0),
            "max_cue_body_lines": max(cue_lines, default=0),
        })
        stats["failure_category"] = STRICT_SRT_FAILURES.get(
            str(exc), "other_value_error"
        )
        print(
            "HSE_WP25_SRT_STRUCTURE " + json.dumps(stats, sort_keys=True),
            flush=True,
        )
        raise
    _stage("production:caption:parsed")
    return {
        "schema": 1,
        "probe": "production_burned_only",
        "cues": len(observed),
        "segments": int(report.segment_count),
        "ocr_calls": int(base.total_calls),
        "retry_calls": int(base.retry_calls),
        "elapsed_seconds": round(elapsed, 2),
        "srt_strict_parse_pass": True,
        "ocr_observations": recorder.observations,
        "max_frame_ocr_chars": recorder.max_chars,
        "max_frame_ocr_lines": recorder.max_lines,
        "max_frame_detection_boxes": recorder.max_detection_boxes,
        "reference_text_or_timing_used": False,
        "clean_control_ocr_run": False,
        "candidate_ocr_run": False,
        "production_default_modified": False,
        "release": "HOLD",
    }


def _production_quality_review(
    root: Path, clip: Path, models: Path, manifest: Path,
) -> dict:
    """Exercise the actual product audit/review path on this natural source.

    No strict benchmark parser or reference transcript is involved. The
    product-generated SRT is audited read-only, then only its quality-review
    cues are decoded into a private local review pack. Neither source nor SRT
    may change.
    """
    from hardsub_extractor.local_ocr_quality_signals import local_ocr_quality_signals
    from hardsub_extractor.review_pack import create_pack

    _stage("production:quality_review:engine_init")
    base = AdaptiveOCREngine(RapidOCRAdapter(
        language="en", model_dir=str(models), manifest_path=str(manifest),
    ))
    out = root / "production-quality-review"
    source_pin = hashlib.sha256(clip.read_bytes()).hexdigest()
    _stage("production:quality_review:extract")
    began = time.perf_counter()
    report = extract(str(clip), base, out, streaming=True)
    elapsed = time.perf_counter() - began
    srt = report.srt_path
    srt_pin = hashlib.sha256(srt.read_bytes()).hexdigest()

    _stage("production:quality_review:audit")
    signals = local_ocr_quality_signals(clip, srt, language="en")
    codes = tuple(signals.get("signal_codes", ()))
    numbers = tuple(signals.get("review_cue_numbers", ()))
    if (
        signals.get("status") != "REVIEW_SUGGESTED"
        or "oversized_ocr_cue_review" not in codes
        or not numbers
    ):
        raise RuntimeError("actual oversized product output did not reach quality review")

    review_dir = root / "private-quality-review"
    _stage("production:quality_review:frames")
    review = create_pack(
        clip, srt, review_dir, mode="quality", max_cues=8, language="en",
    )
    if (
        review.get("status") != "CREATED"
        or not 1 <= int(review.get("cue_count", 0)) <= 8
        or int(review.get("frame_count", 0)) != 2 * int(review.get("cue_count", 0))
        or not (review_dir / "review.html").is_file()
        or not (review_dir / "contact-sheet.jpg").is_file()
        or not (review_dir / "index.json").is_file()
    ):
        raise RuntimeError("actual oversized review pack incomplete")
    index = json.loads((review_dir / "index.json").read_text(encoding="utf-8"))
    selected_codes = {
        str(code)
        for cue in index.get("review_cues", [])
        for code in cue.get("warning_codes", [])
    }
    if "oversized_ocr_cue_review" not in selected_codes:
        raise RuntimeError("oversized cue was not prioritized into first quality review page")
    if (
        hashlib.sha256(clip.read_bytes()).hexdigest() != source_pin
        or hashlib.sha256(srt.read_bytes()).hexdigest() != srt_pin
    ):
        raise RuntimeError("quality review mutated product source or SRT")
    return {
        "schema": 1,
        "probe": "actual_product_quality_review",
        "cues": int(report.cue_count),
        "segments": int(report.segment_count),
        "ocr_calls": int(base.total_calls),
        "retry_calls": int(base.retry_calls),
        "elapsed_seconds": round(elapsed, 2),
        "review_signal_count": int(signals["review_signal_count"]),
        "bounded_review_cue_numbers": len(numbers),
        "oversized_signal_detected": True,
        "oversized_signal_selected_in_first_page": True,
        "private_review_cues": int(review["cue_count"]),
        "private_review_frames": int(review["frame_count"]),
        "source_and_srt_sha256_unchanged": True,
        "strict_benchmark_parser_used": False,
        "reference_text_or_timing_used": False,
        "production_output_rewritten": False,
        "paid_model_requests": 0,
        "release": "HOLD",
    }


def run(
    root: Path, models: Path, manifest: Path, *,
    production_caption_probe: bool = False,
    production_quality_review: bool = False,
) -> dict:
    if root.exists() or root.is_symlink() or not models.is_dir() or not manifest.is_file():
        raise ValueError("fresh scratch and pinned English models required")
    root.mkdir(mode=0o700)
    _stage("reference")
    truth, reference_sha = _reference()
    originals = {k: root / f"{k}-original.webm" for k in SOURCES}
    original_sha = {}
    for key in SOURCES:
        _stage(f"download:{key}")
        original_sha[key] = _download(originals[key], SOURCES[key])
    clips = {k: root / f"{k}-40s.webm" for k in SOURCES}
    clip_sha = {}
    for key in SOURCES:
        _stage(f"clip:{key}")
        clip_sha[key] = _clip(originals[key], clips[key])
    _stage("pair_alignment")
    alignment = _pair_alignment(clips["caption"], clips["clean"])
    if production_quality_review:
        result = _production_quality_review(
            root, clips["caption"], models, manifest,
        )
        result["visual_pair_alignment"] = alignment
        result["source_exact_size_sha1_verified"] = True
        result["source_sha256_preserved"] = all(
            hashlib.sha256(originals[key].read_bytes()).hexdigest()
            == original_sha[key]
            for key in SOURCES
        )
        return result
    if production_caption_probe:
        result = _production_caption_probe(
            root, clips["caption"], models, manifest,
        )
        result["visual_pair_alignment"] = alignment
        result["source_exact_size_sha1_verified"] = True
        result["source_sha256_preserved"] = all(
            hashlib.sha256(originals[key].read_bytes()).hexdigest()
            == original_sha[key]
            for key in SOURCES
        )
        return result

    modes = {}
    for mode in ("production", "light-island-temporal"):
        row = _run_mode(root, clips, models, manifest, mode, truth)
        modes[mode] = row
        print("HSE_WP25_PAIR_CASE " + json.dumps({
            "mode": mode,
            "elapsed_seconds": row["elapsed_seconds"],
            "model_calls": row["model_calls"],
            "score": row["score"],
        }, sort_keys=True), flush=True)
        for key in SOURCES:
            h = hashlib.sha256(originals[key].read_bytes()).hexdigest()
            if h != original_sha[key]:
                raise RuntimeError("publisher original changed during OCR study")

    baseline = modes["production"]["score"]
    candidate = modes["light-island-temporal"]["score"]
    better = (
        candidate["missed_reference_cues"] <= baseline["missed_reference_cues"]
        and candidate["blank_video_false_cues"] <= baseline["blank_video_false_cues"]
        and candidate["exact_visible_text_and_wrap"] >= baseline["exact_visible_text_and_wrap"]
        and candidate["strict_both_edges_250ms_and_text_wrap"] >=
            baseline["strict_both_edges_250ms_and_text_wrap"]
        and (
            candidate["missed_reference_cues"] < baseline["missed_reference_cues"]
            or candidate["blank_video_false_cues"] < baseline["blank_video_false_cues"]
            or candidate["exact_visible_text_and_wrap"] > baseline["exact_visible_text_and_wrap"]
            or candidate["strict_both_edges_250ms_and_text_wrap"] >
                baseline["strict_both_edges_250ms_and_text_wrap"]
        )
    )
    return {
        "schema": 1,
        "source": "Wikimedia Foundation WP25 alternate cut, CC BY-SA 4.0",
        "reference_oldid": REFERENCE_OLDID,
        "reference_sha256": reference_sha,
        "reference_cues_first_40s": len(truth),
        "publisher_burned_subtitle_version": True,
        "publisher_clean_same_alternate_cut": True,
        "publisher_clean_timedtext_reference": True,
        "source_exact_size_sha1_verified": True,
        "source_sha256_preserved": True,
        "derived_clip_sha256": clip_sha,
        "visual_pair_alignment": alignment,
        "synthetic_caption_overlay": False,
        "reference_used_only_after_inference": True,
        "same_model_roi_detector_both_modes": True,
        "ocr_backend": "onnxruntime",
        "modes": modes,
        "candidate_nonregression_and_improvement_on_this_source": bool(better),
        "candidate_auto_promoted": False,
        "production_default_modified": False,
        "paid_model_requests": 0,
        "user_media_read": False,
        "release": "HOLD",
    }


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--scratch", type=Path, required=True)
    p.add_argument("--model-dir", type=Path, required=True)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--approve-public-sources", action="store_true")
    p.add_argument("--production-caption-probe", action="store_true")
    p.add_argument("--production-quality-review", action="store_true")
    a = p.parse_args()
    if a.production_caption_probe and a.production_quality_review:
        print("HSE_WP25_PAIR=SELECT_ONE_PROBE")
        return 2
    if not a.approve_public_sources:
        print("HSE_WP25_PAIR=PUBLIC_SOURCE_NOT_APPROVED")
        return 2
    try:
        result = run(
            a.scratch, a.model_dir, a.manifest,
            production_caption_probe=a.production_caption_probe,
            production_quality_review=a.production_quality_review,
        )
        marker = (
            "HSE_WP25_QUALITY_REVIEW_RESULT "
            if a.production_quality_review
            else "HSE_WP25_CAPTION_PROBE_RESULT "
            if a.production_caption_probe else "HSE_WP25_PAIR_RESULT "
        )
        print(marker + json.dumps(result, sort_keys=True), flush=True)
        return 0
    except Exception as exc:
        print("HSE_WP25_PAIR_HOLD " + json.dumps({
            "error_class": type(exc).__name__,
            "stage": _CURRENT_STAGE,
            "auto_retry": False,
            "quality_proven": False,
        }, sort_keys=True), flush=True)
        return 3


if __name__ == "__main__":
    raise SystemExit(main())
