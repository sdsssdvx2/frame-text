"""Genuine PP-OCRv5 video OCR through the new caption-absence fallback.

A deterministic local MP4 stands in for the *media transport only* so no
public streaming site, user content, browser profile, token, VPN or proxy is
touched. All video-frame recognition and subtitle publishing use real
production engines, FFmpeg and the normal fallback orchestration.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hardsub_extractor.extraction_progress import ExtractionMonitor
from hardsub_extractor.ocr.adaptive import AdaptiveOCREngine
from hardsub_extractor.ocr.rapidocr_adapter import RapidOCRAdapter
from hardsub_extractor.pipeline import extract
from hardsub_extractor.public_video_ocr import resolve_missing_public_ocr
from hardsub_extractor.subtitle_acquisition import CaptionResult
from hardsub_extractor.subtitle_formats import parse_subtitle_bytes

from verify_multilang_video_pipeline import VIDEO_TEXT, EXPECTED_SPANS, encode_sample

YOUTUBE = "https://www.youtube.com/watch?v=synthetic-acceptance"


def accept_one(root: Path, language: str, models: Path, manifest: Path) -> dict:
    fixture = root / ("source-" + language + ".mp4")
    encode_sample(fixture, language)
    if not 1024 <= fixture.stat().st_size <= 32 * 1024 * 1024:
        raise AssertionError("synthetic public-video fixture is not bounded")

    downloaded: list[Path] = []
    original = fixture.read_bytes()
    def fake_media_fetcher(url, folder, **kwargs):
        if url != YOUTUBE or kwargs.get("proxy_url") is not None:
            raise AssertionError("unexpected media route")
        if len(downloaded) != 0:
            raise AssertionError("same episode media downloaded twice")
        output = folder / "video.mp4"
        shutil.copyfile(fixture, output)
        downloaded.append(output)
        return output

    def actual_ocr(video, base, selected_language, monitor):
        engine = AdaptiveOCREngine(RapidOCRAdapter(
            language=selected_language,
            model_dir=str(models), manifest_path=str(manifest),
        ))
        return extract(
            str(video), engine, base,
            monitor=ExtractionMonitor(exclusive_outputs=True),
        )

    output = root / ("results-" + language)
    already_captioned = CaptionResult(
        "youtube", "ko" if language != "ko" else "en",
        "downloaded", video_id="known-track",
        cue_count=300,
    )
    missing = CaptionResult(
        "youtube", language, "needs_ocr", video_id="synthetic",
    )
    outputs = resolve_missing_public_ocr(
        YOUTUBE, output, [already_captioned, missing],
        media_fetcher=fake_media_fetcher,
        ocr_extractor=actual_ocr,
    )
    if len(outputs) != 2 or outputs[0] != already_captioned:
        raise AssertionError("existing public soft-caption success was mutated")
    item = outputs[1]
    if (item.status != "hard_ocr" or item.cue_count != 2
            or not item.srt.is_file() or not item.ass.is_file()):
        raise AssertionError("fallback did not publish expected OCR outputs")
    if len(downloaded) != 1 or downloaded[0].exists():
        raise AssertionError("temporary public video was not cleaned")
    if fixture.read_bytes() != original:
        raise AssertionError("synthetic original video changed")

    _, srt = parse_subtitle_bytes(item.srt.read_bytes(), source_url="result.srt")
    _, ass = parse_subtitle_bytes(item.ass.read_bytes(), source_url="result.ass")
    if len(srt) != 2 or len(ass) != 2:
        raise AssertionError("OCR cue count differs across formats")
    max_error = 0.0
    for index, expected in enumerate(VIDEO_TEXT[language]):
        if srt[index].text != expected or ass[index].text != expected:
            raise AssertionError("fallback OCR text not identical to reference")
        start, end = EXPECTED_SPANS[index]
        max_error = max(
            max_error,
            abs(srt[index].start - start),
            abs(srt[index].end - end),
        )
        if (abs(srt[index].start - ass[index].start) > 0.02
                or abs(srt[index].end - ass[index].end) > 0.02):
            raise AssertionError("SRT and ASS timing diverged")
    if max_error > 0.65:
        raise AssertionError("fallback media timeline drifted")
    metadata = json.loads(item.manifest.read_text(encoding="utf-8"))
    if not all((
        metadata.get("track_origin") == "visual_ocr",
        metadata.get("ocr_applied") is True,
        metadata.get("original_text_verified") is False,
        metadata.get("machine_translation_applied") is False,
        metadata.get("temporary_video_retained") is False,
        metadata.get("cue_count") == 2,
    )):
        raise AssertionError("fallback OCR provenance is incorrect")
    if YOUTUBE in item.manifest.read_text(encoding="utf-8"):
        raise AssertionError("public URL leaked into provenance")
    return {
        "language": language,
        "exact_cues": 2,
        "max_timeline_error_seconds": round(max_error, 4),
        "one_temporary_video": True,
        "original_source_unchanged": True,
        "soft_caption_preserved": True,
        "original_text_verified": True,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--manifest", default=str(ROOT / "models" / "manifest.json"))
    ap.add_argument("--languages", default="ko,en,ja,zh")
    opts = ap.parse_args()
    languages = [s.strip() for s in opts.languages.split(",") if s.strip()]
    if languages != ["ko", "en", "ja", "zh"]:
        ap.error("acceptance must test exact four-language contract")
    try:
        with tempfile.TemporaryDirectory(prefix="hse-public-visual-fallback-") as td:
            root = Path(td)
            rows = [
                accept_one(root, language, Path(opts.model_dir), Path(opts.manifest))
                for language in languages
            ]
        if len(rows) != 4 or sum(x["exact_cues"] for x in rows) != 8:
            raise AssertionError("public fallback evidence is incomplete")
    except Exception as exc:
        print("HSE_PUBLIC_VISUAL_FALLBACK=FAIL type=" + type(exc).__name__)
        return 3
    print("HSE_PUBLIC_VISUAL_FALLBACK=" + json.dumps({
        "status": "PASS", "scope": "isolated real FFmpeg MP4 and RapidOCR, fake public transport",
        "exact_cues": 8, "all_temporary_outputs_removed": True,
        "results": rows,
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
