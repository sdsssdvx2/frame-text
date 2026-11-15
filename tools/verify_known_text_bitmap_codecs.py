"""Actual seconv text -> PGS/VobSub -> MKV -> HSE source-image OCR.

Eight offline source-media experiments (four languages, two bitmap codecs).
Unlike an isolated synthetic PNG test this crosses the real PGS and DVD
subpicture encoders, Matroska muxer, pinned decoder and the production
language-specific PP-OCR pipeline. Expected text is used ONLY for evaluation.
The resulting CER is a DIAGNOSTIC, not natural-anime quality certification.
No customer files, browser, translation API, installed software or VPN.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hardsub_extractor.local_image_subtitles import acquire_local_embedded_image
from hardsub_extractor.models import Cue
from hardsub_extractor.ocr.adaptive import AdaptiveOCREngine
from hardsub_extractor.ocr.rapidocr_adapter import RapidOCRAdapter
from hardsub_extractor.ocr.base import comparison_key, normalize_text
from hardsub_extractor.subtitle_formats import parse_subtitle_bytes
from hardsub_extractor.runtime_paths import resolve_binary
from hardsub_extractor.writers import write_srt
from verify_multilang_ocr_synthetic import FIXTURES, edit_distance

FONTS = {
    "ko": "Malgun Gothic",
    "en": "Arial",
    "ja": "Yu Gothic",
    "zh": "Microsoft YaHei",
}
LANGUAGE_TAGS = {"ko": "kor", "en": "eng", "ja": "jpn", "zh": "chi"}
CODECS = (("pgs", "bluraysup", ".sup", "hdmv_pgs_subtitle"),
          ("vobsub", "vobsub", ".idx", "dvd_subtitle"))


def _sha(file: Path) -> str:
    result = hashlib.sha256()
    with file.open("rb") as stream:
        for block in iter(lambda: stream.read(128 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def _run(command: list[str], *, timeout: int) -> bytes:
    result = subprocess.run(
        command, stdin=subprocess.DEVNULL, capture_output=True,
        timeout=timeout, check=False,
    )
    if result.returncode:
        raise RuntimeError("offline codec conversion failed")
    return result.stdout


def _generate(language: str, kind: str, target: Path, seconv: Path) -> tuple[Path, tuple[Cue, ...]]:
    src = target / "source.srt"
    one, two = FIXTURES[language]
    # The second bitmap contains both subtitle lines, not a video scene.
    references = (
        Cue(1.0, 3.0, one),
        Cue(4.0, 6.0, one + "\n" + two),
    )
    write_srt(list(references), src)
    image_dir = target / "bitstream"
    image_dir.mkdir()
    output_format = "bluraysup" if kind == "pgs" else "vobsub"
    extension = ".sup" if kind == "pgs" else ".idx"
    raw = _run([
        str(seconv), str(src), output_format,
        f"--output-folder:{image_dir}", "--resolution:1920x1080",
        f"--font-name:{FONTS[language]}", "--font-size:58",
        "--font-color:white", "--outline-color:black",
        "--json",
    ], timeout=90)
    try:
        receipt = json.loads(raw.decode("utf-8-sig"))
        if not isinstance(receipt, dict) or receipt.get("success") is not True:
            raise ValueError("converter did not confirm image export")
    except (ValueError, UnicodeError) as exc:
        raise RuntimeError("seconv image export receipt invalid") from exc
    binaries = sorted(image_dir.glob("*" + extension))
    if len(binaries) != 1 or not 1024 <= binaries[0].stat().st_size <= 32 * 1024 * 1024:
        raise RuntimeError("expected exactly one bounded source subtitle bitstream")
    if kind == "vobsub":
        companion = binaries[0].with_suffix(".sub")
        if not companion.is_file() or companion.stat().st_size < 1024:
            raise RuntimeError("VobSub original pair missing")
    source_video = target / "source.mkv"
    _run([
        resolve_binary("ffmpeg"),
        "-nostdin", "-hide_banner", "-loglevel", "error",
        "-protocol_whitelist", "file,pipe",
        # Copy the timestamps when remuxing generated PGS. Otherwise
        # FFmpeg may zero-base the first packet and change the source
        # timeline before HSE ever receives the MKV.
        *(["-copyts"] if kind == "pgs" else []),
        "-i", str(binaries[0]), "-map", "0:s:0",
        "-c:s", "copy", "-metadata:s:s:0",
        "language=" + LANGUAGE_TAGS[language], str(source_video),
    ], timeout=90)
    if not source_video.is_file() or not 1024 <= source_video.stat().st_size <= 32 * 1024 * 1024:
        raise RuntimeError("expected one bounded MKV with real bitmap track")
    return source_video, references


def _inspect(language: str, kind: str, model: AdaptiveOCREngine, directory: Path, seconv: Path) -> dict:
    video, references = _generate(language, kind, directory, seconv)
    original_sha = _sha(video)
    result = acquire_local_embedded_image(
        video, directory / "export" / "caption",
        language=language, ocr_factory=lambda: model,
    )
    if result is None:
        raise AssertionError("HSE did not select an actual image subtitle track")
    expected_codec = "hdmv_pgs_subtitle" if kind == "pgs" else "dvd_subtitle"
    fmt_s, actual = parse_subtitle_bytes(
        result.srt_path.read_bytes(), source_url="caption.srt",
    )
    fmt_a, ass = parse_subtitle_bytes(
        result.ass_path.read_bytes(), source_url="caption.ass",
    )
    if fmt_s != "srt" or fmt_a != "ass" or len(actual) != len(ass):
        raise AssertionError("image subtitle SRT ASS export parity failed")
    for s, a in zip(actual, ass):
        if s.text != a.text or abs(s.start - a.start) > .02 or abs(s.end - a.end) > .02:
            raise AssertionError("original text or timeline changed between SRT and ASS")
    meta = json.loads(result.provenance_path.read_text("utf-8"))
    coverage = meta.get("bitmap_ocr_event_coverage")
    if (result.mode != "local-embedded-image-ocr"
            or meta.get("source_codec") != expected_codec
            or meta.get("original_text_verified") is not False
            or not isinstance(coverage, dict)
            or coverage.get("recognized_bitmap_events") != len(actual)
            or coverage.get("source_bitmap_events", 0) < len(actual)
            or coverage.get("unrecognized_bitmap_events")
               != coverage["source_bitmap_events"] - len(actual)):
        raise AssertionError("real encoded image subtitle provenance invalid")
    cases = []
    # Only an event-complete two-cue track can be aligned by source order.
    # This separates recognizer errors from timestamp shifting introduced
    # by a generated test media remux. Incomplete tracks still require
    # a source-time match; never fabricate a dropped original cue.
    ordinal_safe = (
        len(actual) == len(references)
        and coverage["unrecognized_bitmap_events"] == 0
        and coverage["source_bitmap_events"] == len(references)
    )
    for i, expected in enumerate(references):
        cue = (
            actual[i] if ordinal_safe else
            min(actual, key=lambda c: abs(c.start - expected.start))
            if actual else None
        )
        if not ordinal_safe and cue is not None and abs(cue.start - expected.start) > .75:
            cue = None
        hypothesis = cue.text if cue else ""
        reference_key, actual_key = comparison_key(expected.text), comparison_key(hypothesis)
        cases.append({
            "character_error_rate": round(
                edit_distance(reference_key, actual_key) / max(1, len(reference_key)), 4,
            ),
            "exact_normalized_text": normalize_text(expected.text) == normalize_text(hypothesis),
            "source_lines": len(expected.text.splitlines()),
            "recognized_lines": len(hypothesis.splitlines()) if hypothesis else 0,
            "source_ordinal_alignment": ordinal_safe,
            "observed_start_s": round(cue.start, 3) if cue else None,
            "observed_end_s": round(cue.end, 3) if cue else None,
            "start_error_s": round(abs(expected.start - cue.start), 3) if cue else None,
            "end_error_s": round(abs(expected.end - cue.end), 3) if cue else None,
        })
    output_files = [
        result.srt_path, result.ass_path, result.extracted_track_path,
        result.provenance_path,
    ]
    original_outputs = {_sha(p) for p in output_files}
    try:
        acquire_local_embedded_image(
            video, directory / "export" / "caption",
            language=language, ocr_factory=lambda: model,
        )
    except FileExistsError:
        pass
    else:
        raise AssertionError("duplicate source image export was not rejected")
    if {_sha(p) for p in output_files} != original_outputs or _sha(video) != original_sha:
        raise AssertionError("existing subtitles or source media changed")
    return {
        "language": language, "codec": expected_codec,
        "source_events": coverage["source_bitmap_events"],
        "recognized_events": coverage["recognized_bitmap_events"],
        "unrecognized_events": coverage["unrecognized_bitmap_events"],
        "cases": cases,
        "real_encoder_and_MKV": True,
        "original_media_and_outputs_unchanged": True,
        "natural_video_quality_proven": False,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scratch", type=Path, required=True)
    ap.add_argument("--seconv", type=Path, required=True)
    ap.add_argument("--model-dir", type=Path, required=True)
    ap.add_argument("--manifest", type=Path, required=True)
    args = ap.parse_args()
    if not args.seconv.is_file() or not args.model_dir.is_dir() or not args.scratch.is_dir():
        raise ValueError("offline pinned model/seconv or isolated scratch unavailable")
    results = []
    failures = []
    with tempfile.TemporaryDirectory(
        prefix="hse-known-bitmap-", dir=str(args.scratch),
    ) as temporary:
        root = Path(temporary)
        for language in ("ko", "en", "ja", "zh"):
            model = AdaptiveOCREngine(RapidOCRAdapter(
                model_dir=str(args.model_dir),
                manifest_path=str(args.manifest), language=language,
            ))
            for kind, _format, _ext, _codec in CODECS:
                folder = root / (language + "-" + kind)
                folder.mkdir()
                try:
                    row = _inspect(language, kind, model, folder, args.seconv)
                except Exception as exc:
                    failures.append({
                        "language": language, "codec": kind,
                        "error_type": type(exc).__name__,
                    })
                else:
                    results.append(row)
                    print("HSE_PGS_VOBSUB_ORACLE_CASE=" + json.dumps(
                        row, sort_keys=True, ensure_ascii=True,
                    ), flush=True)
    summary = {
        "scope": "OFFLINE_KNOWN_SOURCE_TEXT_ACTUAL_PGS_VOBSUB",
        "total_experiments": 8,
        "completed": len(results),
        "failures": failures,
        "exact_cases": sum(
            int(item["exact_normalized_text"])
            for row in results for item in row["cases"]
        ),
        "total_cues_measured": sum(len(row["cases"]) for row in results),
        "zero_cer_cases": sum(
            int(item["character_error_rate"] == 0.0)
            for row in results for item in row["cases"]
        ),
        "timing_within_0p5s_cases": sum(
            int(item["start_error_s"] is not None
                and item["end_error_s"] is not None
                and item["start_error_s"] <= .5
                and item["end_error_s"] <= .5)
            for row in results for item in row["cases"]
        ),
        "max_character_error_rate": max(
            (item["character_error_rate"]
             for row in results for item in row["cases"]), default=None,
        ),
        "diagnostic_not_release_approval": True,
        "actual_anime_footage_not_used": True,
    }
    print("HSE_PGS_VOBSUB_ORACLE_SUMMARY=" + json.dumps(
        summary, sort_keys=True, ensure_ascii=True,
    ), flush=True)
    # Finish the measurement even if some languages/codecs failed. A runner
    # success means the study completed, never that OCR quality passed.
    return 0 if len(results) + len(failures) == 8 else 3


if __name__ == "__main__":
    raise SystemExit(main())
