"""Packaged EXE acceptance: caption absence -> real local video OCR.

The *transport* is a one-video local fixture copy, not public network access.
The packaged frozen EXE itself runs the actual multilingual PP-OCRv5 and
FFmpeg pipeline, source-timecode export, provenance and cleanup. No browser,
VPN, site login, credentials, source text logs or user videos.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil
import sys
import tempfile

from .frozen_multilingual_ocr_acceptance import (
    EXPECTED_SPANS, EXPECTED_TEXT, LANGUAGES, _bounded_local_fixture,
)
from .public_video_ocr import resolve_missing_public_ocr
from .public_output_audit import audit_soft_output, audit_visual_output
from .subtitle_acquisition import CaptionResult, _commit_caption
from .subtitle_language import SubtitleTrack
from .subtitle_formats import parse_subtitle_bytes

_SYNTHETIC_URL = "https://www.youtube.com/watch?v=frozen-synthetic-public-ocr"


def _case(language: str, source: Path, root: Path) -> dict:
    before = hashlib.sha256(source.read_bytes()).hexdigest()
    copied: list[Path] = []
    def fake_public_transport(url, folder, **kwargs):
        if url != _SYNTHETIC_URL or kwargs.get("proxy_url") is not None:
            raise AssertionError("frozen acceptance transport was not isolated")
        if copied:
            raise AssertionError("duplicate synthetic video transfer")
        dest = folder / "video.mp4"
        shutil.copyfile(source, dest)
        copied.append(dest)
        return dest

    existing = CaptionResult(
        "youtube", "en" if language == "ko" else "ko",
        "downloaded", video_id="synthetic", cue_count=1,
    )
    missing = CaptionResult(
        "youtube", language, "needs_ocr", video_id="synthetic",
    )
    returned = resolve_missing_public_ocr(
        _SYNTHETIC_URL, root / ("outputs-" + language),
        [existing, missing],
        media_fetcher=fake_public_transport,
    )
    if len(returned) != 2 or returned[0] != existing:
        raise AssertionError("existing public subtitles were modified")
    item = returned[1]
    if (item.status != "hard_ocr" or item.cue_count != 2
            or item.srt is None or item.ass is None or item.manifest is None):
        raise AssertionError("frozen visual OCR was not published")
    if len(copied) != 1 or copied[0].exists():
        raise AssertionError("temporary downloaded video was not removed")

    _, srt = parse_subtitle_bytes(item.srt.read_bytes(), source_url="frozen.srt")
    _, ass = parse_subtitle_bytes(item.ass.read_bytes(), source_url="frozen.ass")
    if len(srt) != 2 or len(ass) != 2:
        raise AssertionError("frozen fallback did not produce two real cues")
    max_clock = 0.0
    max_cross = 0.0
    for index, expected in enumerate(EXPECTED_TEXT[language]):
        if srt[index].text != expected or ass[index].text != expected:
            raise AssertionError("frozen fallback text differs from locked reference")
        start, end = EXPECTED_SPANS[index]
        max_clock = max(
            max_clock,
            abs(srt[index].start - start),
            abs(srt[index].end - end),
        )
        max_cross = max(
            max_cross,
            abs(srt[index].start - ass[index].start),
            abs(srt[index].end - ass[index].end),
        )
    if max_clock > 0.65 or max_cross > 0.02:
        raise AssertionError("frozen fallback timeline drifted")

    provenance_raw = item.manifest.read_text(encoding="utf-8")
    provenance = json.loads(provenance_raw)
    if (provenance.get("ocr_applied") is not True
            or provenance.get("machine_translation_applied") is not False
            or provenance.get("original_text_verified") is not False
            or provenance.get("temporary_video_retained") is not False
            or provenance.get("track_origin") != "visual_ocr"
            or _SYNTHETIC_URL in provenance_raw):
        raise AssertionError("frozen fallback provenance invalid")
    if hashlib.sha256(source.read_bytes()).hexdigest() != before:
        raise AssertionError("fixture source was modified")

    # The frozen executable must also prove the new *read-only resume* path,
    # not merely freshly exported OCR. An already complete result may not
    # download the source video or load OCR again, even when no matching
    # provider captions were advertised.
    before_files = {
        str(p): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in (item.srt, item.ass, item.manifest)
    }
    def forbid_network_or_model(*_args, **_kwargs):
        raise AssertionError("read-only resume unexpectedly needed media or OCR")
    reopened = resolve_missing_public_ocr(
        _SYNTHETIC_URL, root / ("outputs-" + language), [missing],
        media_fetcher=forbid_network_or_model,
        ocr_extractor=forbid_network_or_model,
    )
    if (len(reopened) != 1 or reopened[0].status != "existing_verified"
            or reopened[0].cue_count != item.cue_count
            or reopened[0].srt != item.srt or reopened[0].ass != item.ass
            or reopened[0].manifest != item.manifest
            or len(copied) != 1):
        raise AssertionError("frozen visual OCR resume audit failed")
    verified_visual = audit_visual_output(
        root / ("outputs-" + language) /
        ("youtube-synthetic-" + language + "-hard-ocr"),
        provider="youtube", video_id="synthetic", language=language,
    )
    if verified_visual.state != "verified" or verified_visual.cue_count != 2:
        raise AssertionError("frozen OCR output structure was not verified")
    if before_files != {
        str(p): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in (item.srt, item.ass, item.manifest)
    }:
        raise AssertionError("visual resume audit modified published results")

    # A real original VTT, normalized SRT/ASS and metadata are also published
    # through the production writer and checked WITHOUT video or network.
    soft_root = root / ("soft-" + language)
    soft_text = EXPECTED_TEXT[language][0]
    sample_vtt = (
        "WEBVTT\n\n00:00:01.000 --> 00:00:02.500\n"
        + soft_text + "\n"
    ).encode("utf-8")
    soft = _commit_caption(
        sample_vtt, SubtitleTrack(language, "manual", "vtt", language),
        video_id="synthetic_resume", provider="youtube",
        target_directory=soft_root, original_extension=".vtt",
    )
    soft_files = (soft.original, soft.srt, soft.ass, soft.manifest)
    soft_before = [
        hashlib.sha256(p.read_bytes()).hexdigest() for p in soft_files
    ]
    soft_audit = audit_soft_output(
        soft_root / ("youtube-synthetic_resume-" + language + "-manual"),
        provider="youtube", video_id="synthetic_resume",
        requested=language, provided=language, origin="manual",
    )
    if (soft_audit.state != "verified" or soft_audit.cue_count != 1
            or soft_audit.original != soft.original
            or [hashlib.sha256(p.read_bytes()).hexdigest() for p in soft_files]
            != soft_before):
        raise AssertionError("frozen public soft-caption restart audit failed")

    return {
        "language": language,
        "exact_cues": 2,
        "real_encoded_mp4_and_ocr": True,
        "max_timeline_error_seconds": round(max_clock, 4),
        "max_srt_ass_delta_seconds": round(max_cross, 4),
        "single_video_transfer": True,
        "soft_caption_preserved": True,
        "temporary_video_removed": True,
        "read_only_visual_resume_verified": True,
        "read_only_soft_resume_verified": True,
    }


def run_frozen_public_video_ocr_acceptance(
    fixture_dir: str | Path,
    report_path: str | Path,
) -> int:
    result: dict = {
        "schema": 1,
        "type": "frozen_public_video_ocr_fallback_acceptance",
        "is_frozen": bool(getattr(sys, "frozen", False)),
        "ok": False,
        "exact_total": 0,
        "languages": [],
        "transport": "synthetic_local_no_network",
        "raw_media_retained": False,
        "temporary_outputs_removed": False,
    }
    try:
        if not result["is_frozen"]:
            raise RuntimeError("frozen EXE is required")
        folder = Path(fixture_dir)
        fixtures = {
            lang: _bounded_local_fixture(folder, lang) for lang in LANGUAGES
        }
        with tempfile.TemporaryDirectory(prefix="hse-frozen-public-fallback-") as td:
            scratch = Path(td)
            for lang in LANGUAGES:
                row = _case(lang, fixtures[lang], scratch)
                result["languages"].append(row)
                result["exact_total"] += row["exact_cues"]
        result["temporary_outputs_removed"] = True
        result["ok"] = (
            len(result["languages"]) == 4
            and result["exact_total"] == 8
            and all(row["temporary_video_removed"] for row in result["languages"])
            and all(row["read_only_visual_resume_verified"] for row in result["languages"])
            and all(row["read_only_soft_resume_verified"] for row in result["languages"])
        )
    except Exception as exc:
        # Never emit source paths, URL parameters, model/OCR text, headers,
        # temporary media paths or untrusted native stderr.
        result["failure_type"] = type(exc).__name__
    destination = Path(report_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("x", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
        f.write("\n")
    return 0 if result["ok"] else 3
