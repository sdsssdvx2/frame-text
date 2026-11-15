"""ONE targeted real-media research A/B, not a production release gate.

Use publisher-authored burned-caption video + separately published clean video.
The original AV1 source is pinned. Both originals are re-encoded with the same
deterministic MPEG4/MP4 ffmpeg settings for compatibility with the already
validated native VSF CLI. Both HSE baseline and native-gated HSE receive the
*same derived MP4 bytes*. The publisher TimedText is used only at evaluation.
No user videos, browsers, accounts, credentials, text dumps or persistent
installation. External program is already vetted pinned VideoSubFinder v6.10.5.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import subprocess
import tempfile
import time

from compare_external_engine_exports import Cue as ScoreCue, evaluate
from compare_vsf_timing_research import run_vsf, check_native_version
from verify_vsf_cli_research import download_pinned_archive, unpack_pinned_archive
from verify_commons_natural_hardsub import (
    PUBLISHED, _get_media, _get_reference, _probe, _hse_cues,
)
from hardsub_extractor.ocr.adaptive import AdaptiveOCREngine
from hardsub_extractor.ocr.rapidocr_adapter import RapidOCRAdapter
from hardsub_extractor.runtime_paths import resolve_binary
from hardsub_extractor.subtitle_formats import parse_subtitle_bytes
from hardsub_extractor.timed_ocr import extract_timed_ocr

# Pins independently collected from earlier *official-source* natural study.
ORIGINAL_SHA256 = {
    "caption": "01ffc4c35508dda62edfa4dff4e6541e03cfc769a5f76cd71d35cb61713a7f14",
    "clean": "3dc8afa5687766f0cc0a70a1721b05737d5c8ebed815b7c37015cf12b13708ed",
}
MAX_15S_NATIVE_INTERVALS = 24
MAX_ENCODED_BYTES = 120 * 1024 * 1024
MAX_ORIGINAL_DURATION_DELTA = 0.15
_CURRENT_STAGE = "not_started"


def _stage(value: str) -> None:
    """Finite, non-sensitive phase markers; never include video names or text."""
    global _CURRENT_STAGE
    _CURRENT_STAGE = value



def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _derived_mpeg4(original: Path, output: Path, expected_duration: float) -> dict:
    if output.exists() or output.is_symlink():
        raise FileExistsError("transcode target already exists")
    # This is a *re-encode*, NOT original-byte-identity. All methods
    # independently process the same resulting MP4 to keep the A/B fair.
    cmd = [
        resolve_binary("ffmpeg"), "-hide_banner", "-loglevel", "error",
        "-nostdin", "-n", "-i", str(original),
        "-map", "0:v:0", "-an", "-sn", "-dn",
        "-c:v", "mpeg4", "-q:v", "2", "-fps_mode", "passthrough",
        "-movflags", "+faststart", str(output),
    ]
    try:
        result = subprocess.run(
            cmd, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL, timeout=110, check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise RuntimeError("pinned publisher media transcode failed") from exc
    if result.returncode != 0:
        raise RuntimeError("MPEG4 transcode unavailable")
    if output.is_symlink() or not output.is_file():
        raise RuntimeError("derived media missing or linked")
    if not 1 <= output.stat().st_size <= MAX_ENCODED_BYTES:
        raise RuntimeError("derived media outside size bound")
    probe = subprocess.run(
        [
            resolve_binary("ffprobe"), "-v", "error",
            "-protocol_whitelist", "file,pipe",
            "-select_streams", "v:0",
            "-show_entries", "stream=codec_name,width,height:format=duration",
            "-of", "json", str(output),
        ],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
        timeout=25, check=False,
    )
    if probe.returncode or len(probe.stdout) > 16384:
        raise RuntimeError("derived video cannot be validated")
    obj = json.loads(probe.stdout)
    stream = obj["streams"][0]
    duration = float(obj["format"]["duration"])
    if (
        stream["codec_name"] != "mpeg4"
        or int(stream["width"]) != 1920
        or int(stream["height"]) != 1080
        or not math.isfinite(duration)
        or abs(duration - expected_duration) > MAX_ORIGINAL_DURATION_DELTA
    ):
        raise RuntimeError("derived media is not same-timeline content")
    return {
        "codec": "mpeg4",
        "bytes": output.stat().st_size,
        "sha256": _sha256(output),
        "duration": round(duration, 3),
        "reencoded_original": True,
        "not_identical_pixels_or_original_bytes": True,
    }


def _decode_timed_output(base: Path) -> list[ScoreCue]:
    srt = base.with_suffix(".srt")
    ass = base.with_suffix(".ass")
    if not srt.is_file() or not ass.is_file():
        raise RuntimeError("timed OCR result did not publish both subtitle formats")
    _, first = parse_subtitle_bytes(srt.read_bytes(), source_url="candidate.srt")
    _, second = parse_subtitle_bytes(ass.read_bytes(), source_url="candidate.ass")
    if len(first) != len(second):
        raise RuntimeError("native-gated SRT and ASS disagree")
    if any(
        x.text != y.text or abs(x.start - y.start) > 0.011
        or abs(x.end - y.end) > 0.011
        for x, y in zip(first, second)
    ):
        raise RuntimeError("native-gated SRT/ASS content inconsistent")
    return [ScoreCue(c.start, c.end, c.text) for c in first]


def _native_gated(
    media: Path,
    intervals: list[ScoreCue],
    engine: AdaptiveOCREngine,
    base: Path,
) -> tuple[list[ScoreCue], dict]:
    if not intervals:
        return [], {
            "native_intervals": 0, "recognized_intervals": 0,
            "model_calls": 0, "maximum_retained_frames": 0,
        }
    if len(intervals) > MAX_15S_NATIVE_INTERVALS:
        raise RuntimeError("excess native intervals in a 15-second public video")
    before = engine.total_calls
    try:
        report = extract_timed_ocr(
            str(media), intervals, engine, base, sample_fps=8.0,
        )
    except RuntimeError as exc:
        if str(exc) != "no recognized timed subtitles":
            raise
        if base.with_suffix(".srt").exists() or base.with_suffix(".ass").exists():
            raise RuntimeError("empty OCR unexpectedly published subtitles")
        return [], {
            "native_intervals": len(intervals), "recognized_intervals": 0,
            "model_calls": engine.total_calls - before,
            "maximum_retained_frames": 0,
        }
    cues = _decode_timed_output(base)
    if len(cues) != report.recognized_intervals:
        raise RuntimeError("timed OCR report and output counts diverged")
    return cues, {
        "native_intervals": len(intervals),
        "recognized_intervals": report.recognized_intervals,
        "model_calls": engine.total_calls - before,
        "maximum_retained_frames": report.maximum_retained_frames,
    }


def run(model_dir: Path, manifest: Path) -> dict:
    with tempfile.TemporaryDirectory(prefix="hse-vsf-publisher-natural-") as temp:
        root = Path(temp)
        _stage("public_reference")
        reference, reference_sha = _get_reference()
        original = {}
        transformed = {}
        transformed_info = {}
        for label in ("caption", "clean"):
            source = root / (label + ".webm")
            _stage("public_media_" + label)
            pin = _get_media(source, label)
            if pin != ORIGINAL_SHA256[label]:
                raise RuntimeError("original publisher SHA256 was not pinned")
            _stage("probe_original_" + label)
            metadata = _probe(source)
            if source.stat().st_size != PUBLISHED[label][1]:
                raise RuntimeError("original publisher video length changed")
            original[label] = source
            output = root / (label + ".mp4")
            _stage("transcode_" + label)
            transformed_info[label] = _derived_mpeg4(
                source, output, metadata["seconds"],
            )
            transformed[label] = output

        # The pinned native detector has been run and verified independently
        # on *synthetic* Windows samples. This is its first natural-content
        # comparison; no reference transcript is given to the detector.
        _stage("official_native_archive")
        exe = unpack_pinned_archive(
            download_pinned_archive(root / "vsf-official.zip"),
            root / "vsf-bin",
        )
        _stage("native_version")
        check_native_version(exe)

        selected = {}
        timing = {}
        for label in ("caption", "clean"):
            start = time.monotonic()
            _stage("native_detection_" + label)
            selected[label], elapsed = run_vsf(
                exe, transformed[label], root / (label + "-native-detection")
            )
            if len(selected[label]) > MAX_15S_NATIVE_INTERVALS:
                raise RuntimeError("native detector produced implausible interval count")
            timing[label] = {
                "detected_intervals": len(selected[label]),
                "elapsed_seconds": elapsed,
            }

        # Use *exactly the same* pair of decoded MP4 files for HSE baseline
        # and native-gated OCR, including the control with scene text.
        _stage("baseline_model_initialization")
        standard_ocr = AdaptiveOCREngine(RapidOCRAdapter(
            language="en", model_dir=str(model_dir),
            manifest_path=str(manifest),
        ))
        baseline = {}
        baseline_results = {}
        for label in ("caption", "clean"):
            before = standard_ocr.total_calls
            started = time.monotonic()
            _stage("baseline_ocr_" + label)
            baseline_results[label], segments = _hse_cues(
                transformed[label], root / (label + "-hse-baseline"),
                standard_ocr,
            )
            baseline[label] = {
                "cues": len(baseline_results[label]),
                "segments": segments,
                "ocr_calls": standard_ocr.total_calls - before,
                "elapsed_seconds": round(time.monotonic() - started, 2),
            }

        _stage("gated_model_initialization")
        gated_ocr = AdaptiveOCREngine(RapidOCRAdapter(
            language="en", model_dir=str(model_dir),
            manifest_path=str(manifest),
        ))
        gated = {}
        gated_results = {}
        for label in ("caption", "clean"):
            started = time.monotonic()
            _stage("gated_ocr_" + label)
            gated_results[label], gated[label] = _native_gated(
                transformed[label], selected[label], gated_ocr,
                root / (label + "-native-gated"),
            )
            gated[label]["ocr_elapsed_seconds"] = round(
                time.monotonic() - started, 2,
            )
            gated[label]["timing_elapsed_seconds"] = timing[label]["elapsed_seconds"]

        if any(
            _sha256(transformed[label]) != transformed_info[label]["sha256"]
            or _sha256(original[label]) != ORIGINAL_SHA256[label]
            for label in ("caption", "clean")
        ):
            raise RuntimeError("original or derived input was modified")
        _stage("scoring")
        return {
            "schema": 1,
            "published_original_content_and_independent_reference": True,
            "publisher_timedtext_reference_provisional_not_pixel_audited": True,
            "same_derived_video_bytes_for_both_methods": True,
            "derived_media_transcoded_not_byte_identical_to_publisher": True,
            "source_media_sha256_verified": True,
            "source_media_bytes_and_sha1_publisher_verified": True,
            "source_reference_sha256": reference_sha,
            "derived_video": transformed_info,
            "reference_cues": len(reference),
            "native_official_archive_sha256_and_version_verified": True,
            "vendor_not_given_reference_or_transcript": True,
            "native_detections": timing,
            "comparison": {
                "hse_default": {
                    "scores": evaluate(
                        reference, baseline_results["caption"],
                        baseline_results["clean"],
                    ),
                    "performance": baseline,
                },
                "vsf_gated_hse": {
                    "scores": evaluate(
                        reference, gated_results["caption"],
                        gated_results["clean"],
                    ),
                    "performance": gated,
                },
            },
            "real_native_exe_and_hse_model_inference": True,
            "no_auto_promotion_no_vendor_binary_bundled": True,
            "not_ko_ja_zh_natural_video_proof": True,
            "no_user_data_or_production_installation_change": True,
            "study_artifacts_discarded": True,
            "release_gate_passed": False,
        }


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--model-dir", type=Path, required=True)
    p.add_argument("--manifest", type=Path, default=Path("models/manifest.json"))
    p.add_argument("--accept-public-source", action="store_true")
    args = p.parse_args(argv)
    if not args.accept_public_source:
        print("HSE_VSF_NATURAL=PUBLIC_MEDIA_APPROVAL_REQUIRED")
        return 2
    try:
        _stage("starting")
        data = run(args.model_dir, args.manifest)
    except Exception as exc:
        print("HSE_VSF_NATURAL_ERROR " + json.dumps({
            "reason_class": type(exc).__name__,
            "stage": _CURRENT_STAGE,
            "no_score_on_incomplete_execution": True,
        }), flush=True)
        return 4
    print("HSE_VSF_NATURAL_RESULT " + json.dumps(data, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
