"""Subtitle-first public-video batch runner with safe, optional OCR fallback.

Usage:
  python -m hardsub_extractor.subtitle_batch_cli --url https://www.youtube.com/watch?v=... --languages ko,en
  python -m hardsub_extractor.subtitle_batch_cli --urls-file episodes.txt --languages zh-Hans,en --video-ocr

Without --video-ocr the CLI never downloads video or invokes OCR. Explicit
fallback uses bounded transient media after the matching captions are absent.
Read-only resume auditing never replaces partial or existing output files.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
from hardsub_extractor.subtitle_batch import load_episode_urls
from hardsub_extractor.subtitle_proxy import CaptionProxyRoute, parse_site_proxy_specs
from hardsub_extractor.public_video_ocr import PublicVideoOCRError, resolve_missing_public_ocr

from hardsub_extractor.subtitle_acquisition import (
    CaptionAccessError,
    PublicCaptionCollector,
)


def _load_urls(named: list[str], url_file: str | None) -> list[str]:
    """Backward-compatible CLI wrapper for shared GUI batch validation."""
    return load_episode_urls(named, url_file)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Subtitle-first public batch: YouTube, WeTV, Vimeo and Dailymotion; optional --video-ocr",
    )
    parser.add_argument("--url", action="append", default=[], help="Repeat for multiple videos")
    parser.add_argument("--urls-file", help="Text file, one episode URL per line")
    parser.add_argument("--languages", default="ko", help="ko,en,ja,zh,zh-Hans,zh-Hant")
    parser.add_argument("--out", default="public-subtitles", help="Never-overwritten output folder")
    parser.add_argument("--yt-dlp", default=None, help="Optional installed yt-dlp executable path")
    parser.add_argument(
        "--video-ocr", action="store_true",
        help="Only when NO matching manual/auto captions exist, fetch a bounded temporary public video and OCR ko/en/ja/zh (max 512MiB)",
    )
    parser.add_argument(
        "--proxy-mode", choices=("direct", "required", "fallback"), default="direct",
        help="Default DIRECT; required enforces proxy, fallback retries missing captions",
    )
    parser.add_argument(
        "--proxy", default="", help="Unauthenticated HTTP/SOCKS5/SOCKS5H proxy URL",
    )
    parser.add_argument(
        "--site-proxy", action="append", default=[], metavar="SITE=URL",
        help="Require an approved site proxy, e.g. wetv=socks5h://127.0.0.1:17920",
    )
    parser.add_argument(
        "--site-proxy-fallback", action="append", default=[], metavar="SITE=URL",
        help="Use DIRECT, then configured proxy only for this site's failures",
    )
    args = parser.parse_args(argv)
    try:
        urls = _load_urls(args.url, args.urls_file)
        languages = [value.strip() for value in args.languages.split(",") if value.strip()]
        if not languages:
            raise ValueError("at least one subtitle language must be selected")
        default_route = CaptionProxyRoute(args.proxy_mode, args.proxy)
        by_site = parse_site_proxy_specs(
            required=args.site_proxy, fallback=args.site_proxy_fallback,
        )
        collector = PublicCaptionCollector(
            args.yt_dlp, proxy_route=default_route, site_proxy_routes=by_site,
        )
        total_success = 0
        total_verified_existing = 0
        total_pending = 0
        for index, url in enumerate(urls, 1):
            try:
                statuses = collector.collect(
                    url, args.out, languages=languages, skip_existing=True,
                )
                if args.video_ocr and any(item.status == "needs_ocr" for item in statuses):
                    try:
                        statuses = resolve_missing_public_ocr(
                            url, args.out, statuses, executable=args.yt_dlp,
                            proxy_route=default_route, site_proxy_routes=by_site,
                            skip_existing=True,
                        )
                    except PublicVideoOCRError:
                        # Preserve successful soft-track accounting, even when
                        # a later video/probe fails in this same episode.
                        statuses = [
                            replace(item, status="ocr_unavailable")
                            if item.status == "needs_ocr" else item
                            for item in statuses
                        ]
            except FileExistsError:
                print(f"ITEM={index} STATUS=existing_files_preserved")
                total_pending += 1
                continue
            except (CaptionAccessError, ValueError, OSError, PublicVideoOCRError):
                print(f"ITEM={index} STATUS=access_unavailable")
                total_pending += 1
                continue
            for status in statuses:
                # No URLs, playlist IDs, filenames, subtitle contents, bearer
                # headers, tokens, or site stderr are printed into terminal logs.
                print(
                    f"ITEM={index} LANG={status.requested_language}"
                    f" SITE={status.source} STATUS={status.status}"
                    f" CUES={status.cue_count} ROUTE={status.connection_route}"
                )
                if status.status in {"downloaded", "hard_ocr"}:
                    total_success += 1
                elif status.status == "existing_verified":
                    total_verified_existing += 1
                else:
                    total_pending += 1
        # Preserve the historical summary when no verified resume item was
        # present. Audit successes are never mislabeled as new downloads.
        verified_suffix = (
            f" verified={total_verified_existing}" if total_verified_existing else ""
        )
        print(f"SUMMARY downloaded={total_success} pending={total_pending}{verified_suffix}")
        return 0 if total_pending == 0 else 1
    except (ValueError, OSError):
        parser.error("invalid source list or language options")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
