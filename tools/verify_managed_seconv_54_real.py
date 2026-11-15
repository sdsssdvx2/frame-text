"""One-shot real official SeConv 5.4 private-cache admission on Windows.

This tests the actual production user-cache loader, not a staging script.
It is intentionally separate from user-installed HSE and real media. No OCR
models, browser, personal documents or other application settings are used.
"""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import sys

from hardsub_extractor.managed_seconv import ensure_official_seconv_54


def _sha(path):
    hasher=hashlib.sha256()
    with path.open("rb") as source:
        for part in iter(lambda:source.read(128*1024),b""):
            hasher.update(part)
    return hasher.hexdigest()


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--scratch",type=Path,required=True)
    args=p.parse_args()
    if sys.platform!="win32":
        raise RuntimeError("official SeConv 5.4 real runtime acceptance requires Windows")
    root=args.scratch.resolve(strict=True)
    cache=root/"isolated-SeConv-cache"
    if cache.exists() or cache.is_symlink():
        raise RuntimeError("existing external converter cache cannot be changed")
    first=ensure_official_seconv_54(consent=True,cache_base=cache)
    if not first.is_file():
        raise RuntimeError("official SeConv 5.4 did not publish")
    identity=_sha(first)
    second=ensure_official_seconv_54(consent=True,cache_base=cache)
    if first!=second or _sha(second)!=identity:
        raise RuntimeError("cached official converter identity drifted")
    if not (cache/"SeConv-v5.4.0"/"official.zip").is_file():
        raise RuntimeError("pinned official archive unavailable for provenance")
    print("HSE_SECONV54_LIVE_CACHE=OFFICIAL_SHA_PIN_VERIFIED_AND_REAL_VERSION_SMOKE_PASS;"
          "SECOND_USE_BYTE_REVERIFIED=TRUE;UPSTREAM_DOWNLOAD_ONCE=TRUE;"
          "USER_MEDIA=NO;USER_HSE_INSTALL=UNCHANGED;PAID_API_CALLS=0")
    return 0


if __name__=="__main__":
    raise SystemExit(main())
