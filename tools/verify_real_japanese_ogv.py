"""One real, publisher-burned-in Japanese caption source through HSE (HOLD).

Source: Wikimedia Commons, "Wanna Work Together? (Japanese subtitled).ogv"
Original animation: Creative Commons Attribution 2.5 (Ryan Junell et al.)
Japanese subtitles: Donbraco (public domain per file page).
https://commons.wikimedia.org/wiki/File:Wanna_Work_Together%3F_(Japanese_subtitled).ogv

The independently published SHA-1 and byte count pin the ORIGINAL resource;
the video and every subtitle/thumbnail are discarded in a temporary owner-only
folder. No downloaded media or text is uploaded to GitHub, no user media is
opened, no API or account is used, and this does NOT establish transcription
accuracy without a frame-verified independent Japanese transcript.
"""
from __future__ import annotations

import argparse
from contextlib import redirect_stdout
import hashlib
import io
import json
from pathlib import Path
import sys
import urllib.request
from unittest import mock

from hardsub_extractor.cli import main as extract_cli
from hardsub_extractor.quality_audit_io import _probe_local_video_duration
from hardsub_extractor.review_pack import create_pack
from hardsub_extractor.subtitle_formats import parse_subtitle_bytes

SOURCE_URL = (
    "https://upload.wikimedia.org/wikipedia/commons/2/28/"
    "Wanna_Work_Together%3F_%28Japanese_subtitled%29.ogv"
)
SOURCE_BYTES = 10_055_325
SOURCE_SHA1 = "5d649faac3d7179b796de8f18687ef8f4502de09"


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise RuntimeError("pinned natural media source redirected")


def _download_original(destination: Path) -> str:
    if destination.exists() or destination.is_symlink():
        raise ValueError("owned Japanese source path already exists")
    request = urllib.request.Request(
        SOURCE_URL,
        headers={"User-Agent": "HSE-free-JA-original-visual-QA/1.0",
                 "Accept-Encoding": "identity"},
    )
    opener = urllib.request.build_opener(NoRedirect())
    sha1, sha256 = hashlib.sha1(), hashlib.sha256()
    size = 0
    with opener.open(request, timeout=90) as response, destination.open("xb") as output:
        if response.status != 200 or response.geturl() != SOURCE_URL:
            raise RuntimeError("unverified original source response")
        while True:
            chunk = response.read(1024 * 1024)
            if not chunk:
                break
            size += len(chunk)
            if size > SOURCE_BYTES:
                raise RuntimeError("real Japanese source too large")
            sha1.update(chunk)
            sha256.update(chunk)
            output.write(chunk)
    if size != SOURCE_BYTES or sha1.hexdigest() != SOURCE_SHA1:
        raise RuntimeError("real Japanese original metadata does not match Commons")
    return sha256.hexdigest()


def run(root: Path, model_dir: Path) -> dict:
    if root.exists() or root.is_symlink() or not model_dir.is_dir():
        raise ValueError("new isolated scratch and verified free models required")
    root.mkdir(mode=0o700, parents=False)
    original = root / "source.ogv"
    source_sha256 = _download_original(original)
    duration = _probe_local_video_duration(original)
    if not 178 <= duration <= 183:
        raise RuntimeError("original Japanese movie duration unexpected")

    result_base = root / "captions"
    args = [
        "hard-sub-extract", str(original),
        "--ocr-language", "ja", "--model-dir", str(model_dir),
        "--out", str(result_base),
    ]
    with mock.patch.object(sys, "argv", args), redirect_stdout(io.StringIO()) as captured:
        extract_cli()
    # A legitimate embedded text track would be preferred to OCR. This specific
    # published OGV is hard-subtitled; reject an accidental soft-only result.
    mode_line = captured.getvalue()
    if "mode=hard-ocr " not in mode_line:
        raise RuntimeError("published source did not exercise hard OCR fallback")

    srt = result_base.with_suffix(".srt")
    ass = result_base.with_suffix(".ass")
    if not srt.is_file() or not ass.is_file():
        raise RuntimeError("hard-OCR subtitle files missing")
    srt_format, srt_cues = parse_subtitle_bytes(srt.read_bytes(), source_url=srt.name)
    ass_format, ass_cues = parse_subtitle_bytes(ass.read_bytes(), source_url=ass.name)
    if not (srt_format == "srt" and ass_format == "ass"
            and 1 <= len(srt_cues) == len(ass_cues) <= 1000):
        raise RuntimeError("source video produced invalid SRT/ASS pair")
    if not all(
        x.text == y.text
        and abs(x.start - y.start) <= .011
        and abs(x.end - y.end) <= .011
        and 0 <= x.start < x.end <= duration + .25
        for x, y in zip(srt_cues, ass_cues)
    ):
        raise RuntimeError("real Japanese OCR SRT and ASS disagree")
    # This is merely Japanese-script presence, not a truth-aligned CER score.
    recognized_script = sum(
        any(
            "\u3040" <= char <= "\u30ff" or "\u3400" <= char <= "\u9fff"
            for char in cue.text
        ) for cue in srt_cues
    )
    if recognized_script < 1:
        raise RuntimeError("no Japanese-script OCR evidence")

    private = root / "review"
    evidence = create_pack(
        original, srt, private, mode="all", max_cues=2, language="ja",
    )
    if not (
        evidence["status"] == "CREATED"
        and evidence["cue_count"] in (1, 2)
        and evidence["frame_count"] == evidence["cue_count"] * 2
        and (private / "contact-sheet.jpg").is_file()
        and (private / "review.html").is_file()
    ):
        raise RuntimeError("source-frame review export incomplete")
    if hashlib.sha256(original.read_bytes()).hexdigest() != source_sha256:
        raise RuntimeError("source video modified by HSE")

    return {
        "source": "Wikimedia Commons CC BY 2.5 with public-domain Japanese subtitles",
        "official_original_length_sha1_verified": True,
        "source_sha256_unchanged": True,
        "real_original_ogv_theora": True,
        "source_seconds": round(duration, 3),
        "default_original_subtitle_first_then_hard_OCR": True,
        "real_local_PP_OCRv5_Japanese": True,
        "result_cues": len(srt_cues),
        "japanese_script_cues": recognized_script,
        "SRT_ASS_text_and_time_agree": True,
        "local_source_frame_review_cues": evidence["cue_count"],
        "actual_transcription_accuracy_verified": False,
        "reference_transcript_or_timecodes_available": False,
        "original_user_files_modified": False,
        "media_text_and_images_uploaded": False,
        "paid_model_calls": 0,
        "release": "HOLD",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scratch", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--approve-cc-by-download", action="store_true")
    opts = parser.parse_args()
    if not opts.approve_cc_by_download:
        print("HSE_REAL_JA_OGV=SOURCE_DOWNLOAD_NOT_APPROVED")
        return 2
    try:
        answer = run(opts.scratch, opts.model_dir)
        print("HSE_REAL_JA_OGV_RESULT " + json.dumps(answer, sort_keys=True))
        return 0
    except Exception as exc:
        print("HSE_REAL_JA_OGV_HOLD " + json.dumps({
            "error_class": type(exc).__name__,
            "source_user_media_opened": False,
            "natural_accuracy_verified": False,
            "automatic_retry": False,
        }, sort_keys=True))
        return 3


if __name__ == "__main__":
    raise SystemExit(main())
