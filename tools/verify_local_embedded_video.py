"""Real isolated FFmpeg MKV -> language-tagged text -> SRT/ASS product gate.

No user videos, browser state, VPN, network subtitle fetches, OCR-model
download, installation or source/profile modifications. Synthetic temp-only
fixture and outputs are always removed; stdout contains no media paths/text.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hardsub_extractor.local_soft_subtitles import acquire_local_embedded_text
from hardsub_extractor.runtime_paths import resolve_binary
from hardsub_extractor.subtitle_formats import parse_subtitle_bytes

TEXT = {
    "ko": ("안녕하세요", "오늘도 좋은 하루입니다"),
    "en": ("Hello there!", "Have a nice day."),
}


def encode_fixture(root: Path) -> Path:
    subtitles = []
    for language, (first, second) in TEXT.items():
        dest = root / (language + ".srt")
        dest.write_text(
            f"1\n00:00:01,000 --> 00:00:02,500\n{first}\n\n"
            f"2\n00:00:03,000 --> 00:00:04,500\n{second}\n",
            encoding="utf-8-sig",
        )
        subtitles.append(dest)
    video = root / "다국어 내장 자막.mkv"
    cmd = [
        resolve_binary("ffmpeg"),
        "-nostdin", "-hide_banner", "-loglevel", "error",
        "-f", "lavfi", "-i", "color=c=black:s=640x360:r=10:d=6",
        "-sub_charenc", "UTF-8", "-f", "srt", "-i", str(subtitles[0]),
        "-sub_charenc", "UTF-8", "-f", "srt", "-i", str(subtitles[1]),
        "-map", "0:v:0", "-map", "1:s:0", "-map", "2:s:0",
        "-c:v", "mpeg4", "-q:v", "4", "-c:s", "srt",
        "-metadata:s:s:0", "language=kor",
        "-metadata:s:s:1", "language=eng",
        "-t", "6", "-y", str(video),
    ]
    run = subprocess.run(
        cmd, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL, timeout=90, check=False,
    )
    if run.returncode != 0 or not video.is_file() or video.stat().st_size < 1024:
        raise RuntimeError("bounded synthetic multilingual MKV encoding failed")
    return video


def verify(root: Path) -> dict:
    source = encode_fixture(root)
    source_digest = hashlib.sha256(source.read_bytes()).hexdigest()
    results = []
    for language in ("ko", "en"):
        result = acquire_local_embedded_text(
            source, root / ("output-" + language), language=language,
        )
        if result is None or result.cue_count != 2:
            raise AssertionError("expected two cues from embedded text stream")
        if not result.extracted_track_path.is_file() or not result.provenance_path.is_file():
            raise AssertionError("source stream provenance is missing")
        _, srt = parse_subtitle_bytes(result.srt_path.read_bytes(), source_url="x.srt")
        _, ass = parse_subtitle_bytes(result.ass_path.read_bytes(), source_url="x.ass")
        for variant in (srt, ass):
            if len(variant) != 2:
                raise AssertionError("expected same cue count in SRT and ASS")
            for index, expected in enumerate(TEXT[language]):
                if variant[index].text != expected:
                    raise AssertionError("real embedded text differs from reference")
                boundaries = ((1, 2.5), (3, 4.5))[index]
                if max(
                    abs(variant[index].start - boundaries[0]),
                    abs(variant[index].end - boundaries[1]),
                ) > 0.08:
                    raise AssertionError("original timeline exceeded expected tolerance")
        meta = json.loads(result.provenance_path.read_text("utf-8"))
        if meta.get("ocr_applied") is not False or meta.get("source_video_modified") is not False:
            raise AssertionError("incorrect embedded source provenance")
        if str(root) in result.provenance_path.read_text("utf-8"):
            raise AssertionError("source location leaked in provenance")
        results.append({"language": language, "exact_cues": 2, "timeline_verified": True})

    if acquire_local_embedded_text(
        source, root / "output-japanese", language="ja",
    ) is not None:
        raise AssertionError("Japanese mode incorrectly selected Korean/English track")
    if any((root / ("output-japanese" + suffix)).exists() for suffix in (".srt", ".ass")):
        raise AssertionError("unknown-language outputs were falsely published")

    protected = root / "protect.srt"
    protected.write_bytes(b"USER_DATA_NOT_TO_TOUCH")
    try:
        acquire_local_embedded_text(source, root / "protect", language="ko")
    except FileExistsError:
        pass
    else:
        raise AssertionError("output collision did not fail closed")
    if protected.read_bytes() != b"USER_DATA_NOT_TO_TOUCH":
        raise AssertionError("previous subtitle content was modified")

    if hashlib.sha256(source.read_bytes()).hexdigest() != source_digest:
        raise AssertionError("original source video was modified")

    return {
        "schema": 1,
        "source": "temporary synthetic FFmpeg MKV, not user media",
        "results": results,
        "missing_language_fail_closed": True,
        "existing_output_preserved": True,
        "source_video_unchanged": True,
        "ocr_or_model_calls": 0,
    }


def main() -> int:
    try:
        with tempfile.TemporaryDirectory(prefix="hse-local-text-acceptance-") as tmp:
            report = verify(Path(tmp))
        print("HSE_LOCAL_EMBEDDED_TEXT_GATE " + json.dumps(report, sort_keys=True))
        return 0
    except Exception as exc:
        print("HSE_LOCAL_EMBEDDED_TEXT_GATE_FAILED type=" + type(exc).__name__)
        return 3


if __name__ == "__main__":
    raise SystemExit(main())
