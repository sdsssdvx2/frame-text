"""Generate four tiny local MP4s for the *frozen* product acceptance.

This is a pre-executable fixture creation step, NOT source-Python OCR.
The real frozen application must independently decode/recognize/export these
videos and must validate the original subtitle text against a locked fixture
contract. A unique CI scratch directory is created and later deleted.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from verify_multilang_video_pipeline import VIDEO_TEXT, encode_sample


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    folder = Path(args.out)
    folder.mkdir(parents=True, exist_ok=False)
    for language in ("ko", "en", "ja", "zh"):
        if len(VIDEO_TEXT[language]) != 2:
            raise ValueError("synthetic reference video contract changed")
        video = folder / f"{language}.mp4"
        encode_sample(video, language)
        size = video.stat().st_size
        if not (1024 <= size <= 32 * 1024 * 1024):
            raise RuntimeError("generated MP4 size invalid")
        print(f"HSE_FROZEN_OCR_FIXTURE language={language} bytes={size}")
    print("HSE_FROZEN_OCR_FIXTURES=4")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
