"""Strict opt-in *real public* YouTube video-only transport smoke.

The 19-second reference item is publicly available CC-BY-3.0 material
(Me at the zoo, jNQXAC9IVRw). No screenshot, transcript, downloaded video,
network URL, browser state, site cookies, or private local content is saved
to runner artifacts or printed to logs. Use this only for explicit CI runs.

Proves the exact bundled yt-dlp video-download flags, real network access,
temporary size/deadline, verified local video container, and cleanup.
It does NOT claim the video contains any burned-in subtitles, nor that a
real WeTV URL is accessible, nor that OCR text quality has been tested here.
"""
from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
import time

from hardsub_extractor.public_video_ocr import (
    PublicVideoOCRError, download_public_video_for_ocr,
)
from hardsub_extractor.subtitle_acquisition import find_caption_downloader

REFERENCE = "https://www.youtube.com/watch?v=jNQXAC9IVRw"


def main() -> int:
    result = {
        "schema": 1,
        "stage": "public_youtube_video_only_transport",
        "ok": False,
        "media_bytes": 0,
        "local_media_probed": False,
        "single_temporary_video": False,
        "temporary_media_removed": False,
        "credential_use": False,
        "live_source": "public_cc_by_reference",
    }
    saved = None
    started = time.monotonic()
    try:
        with tempfile.TemporaryDirectory(prefix="hse-public-video-live-") as td:
            root = Path(td)
            saved = download_public_video_for_ocr(
                REFERENCE, root,
                executable=find_caption_downloader(),
                byte_limit=16 * 1024 * 1024,
                timeout_seconds=180.0,
            )
            result["media_bytes"] = saved.stat().st_size
            result["local_media_probed"] = True
            result["single_temporary_video"] = (
                len([p for p in root.iterdir() if p.is_file()]) == 1
            )
            result["ok"] = (
                1024 <= result["media_bytes"] <= 16 * 1024 * 1024
                and result["single_temporary_video"]
            )
        result["temporary_media_removed"] = bool(saved) and not saved.exists()
        result["ok"] = bool(result["ok"] and result["temporary_media_removed"])
    except (PublicVideoOCRError, OSError, ValueError) as exc:
        # Never persist provider stderr, manifests, signed URLs, paths,
        # network tokens, subtitles, frames, or exception messages.
        result["failure_class"] = type(exc).__name__
    result["elapsed_seconds"] = round(time.monotonic() - started, 2)
    print("HSE_LIVE_PUBLIC_VIDEO_TRANSPORT=" + json.dumps(result, sort_keys=True))
    return 0 if result["ok"] else 4


if __name__ == "__main__":
    sys.exit(main())
