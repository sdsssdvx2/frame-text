"""Independent original Japanese OGV throughput vs subtitle preservation experiment.

Same *single* original SHA-pinned, publisher-authored public Japanese footage,
losslessly converted into one identical 60-second FFV1 visual input for each
trial. No reference transcript exists, so compare only against HSE's DEFAULT
production extraction output; parity is not the same as human transcription
accuracy. Same model/ROI/detector/renderer for all trials. A poor candidate
is measured and rejected, not fixed using the baseline text as an oracle.
No product default changes, no screenshots or text printed, no user media.
"""
from __future__ import annotations

import argparse
from hashlib import sha256
import json
import math
from pathlib import Path
import subprocess
import sys
import time

from hardsub_extractor.corrected_quality import compare_corrected_cues
from hardsub_extractor.ocr.adaptive import AdaptiveOCREngine
from hardsub_extractor.ocr.rapidocr_adapter import RapidOCRAdapter
from hardsub_extractor.pipeline import extract
from hardsub_extractor.quality_audit_io import _probe_local_video_duration
from hardsub_extractor.runtime_paths import resolve_binary
from hardsub_extractor.subtitle_formats import parse_subtitle_bytes
from verify_real_japanese_ogv import _download_original

# Fixed before measurement; no ground-truth timestamps choose the segment.
START_SECONDS = 45
LENGTH_SECONDS = 60
FPS_VARIANTS = (None, 12.0, 8.0)


def _sha(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _fixed_lossless_clip(original: Path, clip: Path) -> None:
    if clip.exists() or clip.is_symlink():
        raise ValueError("owned trial clip already exists")
    # Decode exactly one range; FFV1 lossless preserves its decoded pixels.
    # Both study branches read the same output bytes. No audio/metadata tracks.
    cmd = [
        resolve_binary("ffmpeg"), "-nostdin", "-hide_banner",
        "-loglevel", "error", "-protocol_whitelist", "file,pipe",
        "-ss", str(START_SECONDS), "-i", str(original),
        "-t", str(LENGTH_SECONDS), "-map", "0:v:0",
        "-an", "-sn", "-dn", "-c:v", "ffv1", "-pix_fmt", "yuv420p",
        "-f", "matroska", str(clip),
    ]
    completed = subprocess.run(
        cmd, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL, timeout=120, check=False,
    )
    if (
        completed.returncode != 0 or not clip.is_file()
        or not 100000 < clip.stat().st_size < 300_000_000
    ):
        raise RuntimeError("lossless independent study clip failed")
    seconds = _probe_local_video_duration(clip)
    if not 59 <= seconds <= 61:
        raise RuntimeError("clipped natural source duration is unexpected")


def _cues(path: Path) -> list:
    if not path.is_file() or not 1 <= path.stat().st_size <= 8_000_000:
        raise RuntimeError("missing bounded subtitle result")
    kind, cues = parse_subtitle_bytes(path.read_bytes(), source_url=path.name)
    if kind != "srt" or not cues:
        raise RuntimeError("no readable natural SRT cues")
    return cues


def _trial(root: Path, clip: Path, model_dir: Path, fps: float | None):
    name = "default" if fps is None else f"fps-{int(fps)}"
    target = root / name
    if target.with_suffix(".srt").exists():
        raise RuntimeError("trial output already exists")
    ocr = AdaptiveOCREngine(RapidOCRAdapter(
        model_dir=str(model_dir), language="ja",
    ))
    start = time.perf_counter()
    report = extract(
        str(clip), ocr, target, streaming=True,
        **({"sample_fps": fps} if fps is not None else {}),
    )
    elapsed = time.perf_counter() - start
    srt, ass = _cues(target.with_suffix(".srt")), target.with_suffix(".ass")
    fmt, ass_cues = parse_subtitle_bytes(ass.read_bytes(), source_url=ass.name)
    if (
        fmt != "ass" or len(srt) != len(ass_cues)
        or any(
            x.text != y.text or abs(x.start - y.start) > .011
            or abs(x.end - y.end) > .011
            for x, y in zip(srt, ass_cues)
        )
    ):
        raise RuntimeError("trial SRT/ASS lost text or timing")
    return {
        "fps": fps, "cues": len(srt),
        "segments": report.segment_count,
        "model_calls": int(ocr.total_calls),
        "retry_calls": int(ocr.retry_calls),
        "elapsed_seconds": round(elapsed, 2),
    }, srt


def run(root: Path, models: Path) -> dict:
    if root.exists() or root.is_symlink() or not models.is_dir():
        raise ValueError("new isolated scratch and pinned model assets required")
    root.mkdir(mode=0o700, parents=False)
    original, clip = root / "source.ogv", root / "same-pixels.mkv"
    source_sha = _download_original(original)
    _fixed_lossless_clip(original, clip)
    trial_sha = _sha(clip)
    results = []
    baseline = None
    for fps in FPS_VARIANTS:
        row, cues = _trial(root, clip, models, fps)
        if baseline is None:
            baseline = cues
            row["same_default_text_and_timing"] = True
            row["missing_baseline_cues"] = 0
            row["new_cues_not_in_default"] = 0
            row["baseline_exact_text_cues"] = len(cues)
            row["within_250ms_cues"] = len(cues)
            row["candidate_for_promotion"] = False
        else:
            diff = compare_corrected_cues(baseline, cues)
            # This is DEFAULT HSE baseline comparison, not independent truth!
            row.update({
                "baseline_cues": len(baseline),
                "missing_baseline_cues": int(diff.baseline_only),
                "new_cues_not_in_default": int(diff.corrected_only),
                "aligned_cues": int(diff.aligned_cues),
                "baseline_exact_text_cues": int(diff.exact_visible_cues),
                "within_250ms_cues": int(diff.precise_250ms_cues),
                "joint_text_and_250ms_cues": int(diff.strict_joint_cues),
                "model_calls_saved": results[0]["model_calls"] - row["model_calls"],
                "wall_time_saved_seconds": round(
                    results[0]["elapsed_seconds"] - row["elapsed_seconds"], 2,
                ),
            })
            parity = (
                row["cues"] == len(baseline)
                and diff.baseline_only == diff.corrected_only == 0
                and diff.exact_visible_cues == len(baseline)
                and diff.strict_joint_cues == len(baseline)
            )
            faster = (
                row["elapsed_seconds"] <= results[0]["elapsed_seconds"] * .85
                and row["model_calls"] <= results[0]["model_calls"] * .95
            )
            row["same_default_text_and_timing"] = bool(parity)
            row["candidate_for_promotion"] = bool(parity and faster)
        if _sha(clip) != trial_sha or _sha(original) != source_sha:
            raise RuntimeError("source bytes changed during visual comparison")
        results.append(row)
        print("HSE_JA_FPS_CASE " + json.dumps(row, sort_keys=True), flush=True)
    if len(results) != len(FPS_VARIANTS) or baseline is None:
        raise RuntimeError("incomplete original Japanese OCR comparative trial")
    summary = {
        "test_type": "original authored Japanese, decoded lossless 60-sec fixed slice",
        "published_source_sha_pinned": True,
        "source_frames_sha_preserved": True,
        "same_exact_input_clip_sha_all_trials": True,
        "same_local_PP_OCRv5_config": True,
        "production_default_modified": False,
        "automatic_speed_opt_in": False,
        "verified_transcription_truth": False,
        "verified_original_subtitle_timing_truth": False,
        "reference_transcript_used_in_inference": False,
        "user_media_read": False,
        "external_media_text_export": False,
        "paid_model_requests": 0,
        "variants": results,
        "release": "HOLD",
    }
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scratch", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--approve-cc-by-download", action="store_true")
    a = parser.parse_args()
    if not a.approve_cc_by_download:
        print("HSE_JA_FPS_RESEARCH=SOURCE_DOWNLOAD_NOT_APPROVED")
        return 2
    try:
        report = run(a.scratch, a.model_dir)
    except Exception as exc:
        print("HSE_JA_FPS_HOLD " + json.dumps({
            "error_class": type(exc).__name__,
            "auto_retry": False,
            "original_operating_source_untouched": True,
        }, sort_keys=True))
        return 3
    print("HSE_JA_FPS_REPORT " + json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
