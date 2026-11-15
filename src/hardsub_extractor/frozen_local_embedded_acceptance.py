"""Frozen Windows EXE embedded-text subtitle end-to-end acceptance.

All input media is newly generated synthetic local material inside an
auto-deleted temporary directory. Never read user videos, browser profiles,
VPN settings, or network URLs. The report deliberately omits text and paths.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile

from .local_soft_subtitles import acquire_local_embedded_text
from .runtime_paths import resolve_binary
from .subtitle_formats import parse_subtitle_bytes

_TEXT = {
    "ko": ("안녕하세요", "오늘도 좋은 하루입니다"),
    "en": ("Hello there!", "Have a nice day."),
}
_SPANS = ((1.0, 2.5), (3.0, 4.5))


def _create_fixture(root: Path) -> Path:
    captions = []
    for lang, rows in _TEXT.items():
        path = root / (lang + ".srt")
        path.write_text(
            "1\n00:00:01,000 --> 00:00:02,500\n" + rows[0]
            + "\n\n2\n00:00:03,000 --> 00:00:04,500\n" + rows[1] + "\n",
            encoding="utf-8-sig",
        )
        captions.append(path)
    video = root / "frozen-local-soft-fixture.mkv"
    args = [
        resolve_binary("ffmpeg"),
        "-nostdin", "-hide_banner", "-loglevel", "error",
        "-f", "lavfi", "-i", "color=c=black:s=640x360:r=10:d=6",
        "-sub_charenc", "UTF-8", "-f", "srt", "-i", str(captions[0]),
        "-sub_charenc", "UTF-8", "-f", "srt", "-i", str(captions[1]),
        "-map", "0:v:0", "-map", "1:s:0", "-map", "2:s:0",
        "-c:v", "mpeg4", "-q:v", "4", "-c:s", "srt",
        "-metadata:s:s:0", "language=kor",
        "-metadata:s:s:1", "language=eng",
        "-t", "6", "-y", str(video),
    ]
    process = subprocess.run(
        args, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL, timeout=90, check=False,
    )
    if process.returncode != 0 or not video.is_file():
        raise RuntimeError("synthetic local video generation failed")
    if not 1024 <= video.stat().st_size <= 32 * 1024 * 1024:
        raise RuntimeError("synthetic local video size invalid")
    return video


def _verify_real_product(root: Path) -> list[dict[str, object]]:
    source = _create_fixture(root)
    original_hash = hashlib.sha256(source.read_bytes()).hexdigest()
    rows = []
    for language in ("ko", "en"):
        result = acquire_local_embedded_text(
            source, root / ("result-" + language), language=language,
        )
        if result is None or result.mode != "local-embedded-text" or result.cue_count != 2:
            raise AssertionError("frozen product did not select embedded text")
        if not result.extracted_track_path.is_file() or not result.provenance_path.is_file():
            raise AssertionError("missing embedded text provenance")
        metadata = json.loads(result.provenance_path.read_text("utf-8"))
        if metadata.get("ocr_applied") is not False or metadata.get("cue_count") != 2:
            raise AssertionError("frozen product provenance inconsistent")
        if metadata.get("source_stream_export") != "ffmpeg_decoded_srt_not_original_container_bytes":
            raise AssertionError("frozen product claimed impossible original stream bytes")
        maximum_timing_error = 0.0
        for ext, dest in (("srt", result.srt_path), ("ass", result.ass_path)):
            _, cues = parse_subtitle_bytes(dest.read_bytes(), source_url="local." + ext)
            if len(cues) != 2:
                raise AssertionError("frozen product lost embedded subtitles")
            for index, reference in enumerate(_TEXT[language]):
                if cues[index].text != reference:
                    raise AssertionError("frozen product output is not exact")
                start, end = _SPANS[index]
                maximum_timing_error = max(
                    maximum_timing_error,
                    abs(cues[index].start - start), abs(cues[index].end - end),
                )
        if maximum_timing_error > 0.08:
            raise AssertionError("frozen product subtitle timeline drift")
        rows.append({
            "language": language,
            "exact_cue_count": 2,
            "original_timeline_max_error_seconds": round(maximum_timing_error, 4),
            "srt_ass_verified": True,
        })

    if acquire_local_embedded_text(
        source, root / "wrong-language", language="ja",
    ) is not None:
        raise AssertionError("frozen product accepted unrelated track")
    if (root / "wrong-language.srt").exists():
        raise AssertionError("frozen product published a false translation")

    original = b"EXISTING_SUBTITLE_NEVER_REPLACE"
    (root / "occupied.srt").write_bytes(original)
    try:
        acquire_local_embedded_text(source, root / "occupied", language="ko")
    except FileExistsError:
        pass
    else:
        raise AssertionError("frozen product replaced existing outputs")
    if (root / "occupied.srt").read_bytes() != original:
        raise AssertionError("frozen product modified existing outputs")

    if hashlib.sha256(source.read_bytes()).hexdigest() != original_hash:
        raise AssertionError("frozen product modified original video")
    return rows


def run_frozen_local_embedded_acceptance(report_path: str | Path) -> int:
    """Run only on the packaged EXE; no outside inputs or subtitle text logs."""
    frozen = bool(getattr(sys, "frozen", False))
    result: dict[str, object] = {
        "schema": 1,
        "type": "frozen_local_embedded_video_acceptance",
        "is_frozen": frozen,
        "ok": False,
        "languages": [],
        "exact_cue_count": 0,
        "wrong_language_blocked": False,
        "existing_outputs_preserved": False,
        "original_video_unchanged": False,
        "no_external_requests": True,
        "temporary_outputs_removed": False,
    }
    try:
        if not frozen:
            raise RuntimeError("acceptance is only valid inside packaged executable")
        with tempfile.TemporaryDirectory(prefix="hse-frozen-local-embedded-") as td:
            rows = _verify_real_product(Path(td))
            result["languages"] = rows
            result["exact_cue_count"] = sum(int(x["exact_cue_count"]) for x in rows)
            result["wrong_language_blocked"] = True
            result["existing_outputs_preserved"] = True
            result["original_video_unchanged"] = True
        result["temporary_outputs_removed"] = True
        result["ok"] = (
            len(result["languages"]) == 2
            and result["exact_cue_count"] == 4
            and all(row["srt_ass_verified"] for row in result["languages"])
        )
    except Exception as exc:
        # No absolute source/temporary path or subtitle text is ever persisted.
        result["failure_type"] = type(exc).__name__

    dest = Path(report_path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    with dest.open("x", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
        f.write("\n")
    return 0 if result["ok"] else 3
