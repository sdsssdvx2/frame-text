"""Real VideoSubFinder 6.10.5 Windows CLI temporal detector research only.

This study runs the official SHA-verified isolated native CPU binary on the
same original encoded JA/ZH two-line moving video and no-caption controls.
It NEVER asks the upstream detector to recognize reference text. It never
runs on user media. Results are timing-only, NOT OCR accuracy proof.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import tempfile
import time

from compare_external_engine_exports import align_cues, parse_strict_srt, Cue
from verify_animated_background_ocr_quality import encode_video
from verify_multilang_video_pipeline import EXPECTED_SPANS
from verify_vsf_cli_research import download_pinned_archive, unpack_pinned_archive

LANGUAGES = ("ja", "zh")
MAX_CUE_COUNT = 150
MAX_VIDEO_BYTES = 32 * 1024 * 1024
MAX_RUN_SECONDS = 200


def _call_bounded(command, cwd, timeout):
    t0 = time.monotonic()
    try:
        result = subprocess.run(
            [str(i) for i in command], cwd=str(cwd),
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            timeout=timeout, check=False,
        )
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError("native research command timed out") from exc
    return result, round(time.monotonic() - t0, 2)


def check_native_version(exe: Path) -> bool:
    r, _ = _call_bounded([exe, "--version"], exe.parent, 15)
    value = (r.stdout + r.stderr).decode("utf-8", "replace")
    if r.returncode != 0 or "6.10.5" not in value:
        raise RuntimeError("official native binary version mismatch")
    return True


def run_vsf(exe: Path, video: Path, destination: Path) -> tuple[list[Cue], float]:
    """No -c (deletion) and output is a newly created isolated directory."""
    if destination.exists() or destination.is_symlink():
        raise RuntimeError("research output must be unique")
    destination.mkdir(parents=True)
    srt = destination / "timing.srt"
    command = [
        exe, "-r", "-i", video, "-o", destination,
        "-ces", srt, "-te", "0.44", "-be", "0",
        "-le", "0", "-re", "1", "-nthr", "4",
    ]
    result, elapsed = _call_bounded(command, exe.parent, MAX_RUN_SECONDS)
    if result.returncode != 0:
        raise RuntimeError("isolated VideoSubFinder CLI failed")
    if not srt.is_file():
        raise RuntimeError("native timing SRT not emitted")
    if srt.stat().st_size > 512 * 1024:
        raise RuntimeError("native timing SRT exceeds safe bound")
    cues = parse_strict_srt(srt.read_bytes(), allow_empty=True)
    if len(cues) > MAX_CUE_COUNT:
        raise RuntimeError("research native produced implausible cue count")
    return cues, elapsed


def timing_report(cues: list[Cue]) -> dict:
    truth = [Cue(start, end, "") for start, end in EXPECTED_SPANS]
    pairs = align_cues(truth, cues)
    errors = [
        (
            abs(truth[i].start - cues[j].start),
            abs(truth[i].end - cues[j].end),
        )
        for i, j in pairs
    ]
    return {
        "detected_timing_cues": len(cues),
        "matched_reference_intervals": len(pairs),
        "missed_reference_intervals": len(truth) - len(pairs),
        "extra_timing_intervals": len(cues) - len(pairs),
        "both_edges_within_250ms": sum(
            max(x) <= 0.25000001 for x in errors
        ),
        "both_edges_within_650ms": sum(
            max(x) <= 0.65000001 for x in errors
        ),
        "max_absolute_matched_edge_error_seconds": (
            round(max(max(x) for x in errors), 4) if errors else None
        ),
    }


def main() -> int:
    rows = []
    version_verified = False
    try:
        with tempfile.TemporaryDirectory(prefix="hse-vsf-research-") as td:
            root = Path(td)
            source = download_pinned_archive(root / "official.zip")
            exe = unpack_pinned_archive(source, root / "native")
            version_verified = check_native_version(exe)
            for lang in LANGUAGES:
                target = root / lang
                target.mkdir()
                caption = target / "caption.mp4"
                blank = target / "blank.mp4"
                encode_video(caption, lang, style="white_two_line")
                encode_video(
                    blank, lang, style="white_two_line",
                    include_subtitles=False,
                )
                if max(caption.stat().st_size, blank.stat().st_size) > MAX_VIDEO_BYTES:
                    raise RuntimeError("research source video too large")
                original_hash = hashlib.sha256(caption.read_bytes()).hexdigest()
                control_hash = hashlib.sha256(blank.read_bytes()).hexdigest()
                positive, elapsed = run_vsf(exe, caption, target / "positive")
                negative, blank_elapsed = run_vsf(exe, blank, target / "negative")
                if (
                    hashlib.sha256(caption.read_bytes()).hexdigest() != original_hash
                    or hashlib.sha256(blank.read_bytes()).hexdigest() != control_hash
                ):
                    raise RuntimeError("test video bytes changed across native experiment")
                row = {
                    "language": lang,
                    "native_cli_version": "6.10.5",
                    "source_encoded_mp4": True,
                    "same_original_bytes_for_any_future_comparison": True,
                    "reference_used_for_native_detection": False,
                    "source_is_synthetic_not_real_anime": True,
                    "rendered_background_negative_control": True,
                    "native_executed_with_pinned_sha_and_version": version_verified,
                    "timing": timing_report(positive),
                    "no_caption_false_timing_cues": len(negative),
                    "elapsed_positive_seconds": elapsed,
                    "elapsed_blank_seconds": blank_elapsed,
                    "ocr_recognition_executed": False,
                }
                rows.append(row)
                print("HSE_VSF_TIMING_CASE " + json.dumps(row, sort_keys=True), flush=True)
    except Exception as exc:
        print("HSE_VSF_TIMING_ERROR " + json.dumps({
            "error_class": type(exc).__name__,
        }), flush=True)
        return 4
    complete = len(rows) == len(LANGUAGES)
    print("HSE_VSF_TIMING_SUMMARY " + json.dumps({
        "schema": 1, "all_languages_complete": complete,
        "native_version_verified": version_verified,
        "languages": list(LANGUAGES),
        "upstream_binary_added_to_hse": False,
        "user_installed_app_changed": False,
        "no_private_video_used": True,
        "false_cues_total": sum(
            x["no_caption_false_timing_cues"] for x in rows
        ),
        "real_anime_ocr_accuracy_verified": False,
        "release_gate_passed": False,
    }, sort_keys=True), flush=True)
    return 0 if complete else 4


if __name__ == "__main__":
    raise SystemExit(main())
