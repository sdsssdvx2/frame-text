"""Opt-in, no-login public caption E2E diagnostic.

Network is used only for the explicitly provided public episode URLs.
Only provenance/count/timeline checks appear in the report; raw tracks,
signed metadata, URLs and subtitle text are discarded in TemporaryDirectory.
No video, OCR, browser profile or cookies are used.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import tempfile
import time

from hardsub_extractor.subtitle_acquisition import (
    CaptionAccessError, PublicCaptionCollector, supported_public_site,
)
from hardsub_extractor.subtitle_formats import parse_subtitle_bytes
from hardsub_extractor.subtitle_language import candidate_tracks, language_matches, normalize_language


class ProbeDeadline(Exception):
    """The bounded live diagnostic was intentionally stopped."""


class ObservedCollector(PublicCaptionCollector):
    def __init__(self, executable: str):
        super().__init__(executable)
        self.candidate_count: dict[str, int] = {"manual": 0, "automatic": 0}
        self.requested_language = "en"

    def _metadata(
        self, url: str, *, check_cancelled=None, proxy_url: str | None = None,
    ) -> dict:
        meta = super()._metadata(
            url, check_cancelled=check_cancelled, proxy_url=proxy_url,
        )
        tracks = candidate_tracks(
            self.requested_language,
            meta.get("subtitles") or {},
            meta.get("automatic_captions") or {},
        )
        self.candidate_count = {
            origin: sum(track.origin == origin for track in tracks)
            for origin in ("manual", "automatic")
        }
        return meta


def _verify_download(result) -> int:
    """Verify all four files, without disclosing or preserving their contents."""
    paths = (result.original, result.srt, result.ass, result.manifest)
    if any(path is None or not Path(path).is_file() for path in paths):
        raise AssertionError("downloaded caption is missing an output file")
    raw_format, raw = parse_subtitle_bytes(
        result.original.read_bytes(), source_url=result.original.name,
    )
    _, srt = parse_subtitle_bytes(result.srt.read_bytes(), source_url=result.srt.name)
    _, ass = parse_subtitle_bytes(result.ass.read_bytes(), source_url=result.ass.name)
    manifest = json.loads(result.manifest.read_text(encoding="utf-8"))
    if not all((
        len(raw) == len(srt) == len(ass) == result.cue_count,
        result.cue_count > 0,
        manifest.get("cue_count") == result.cue_count,
        manifest.get("original_format") == raw_format,
        manifest.get("site") == result.source,
        manifest.get("track_origin") == result.track_origin,
        manifest.get("original_timeline") == "source_video",
        manifest.get("output_timeline") == "source_video",
        manifest.get("ocr_applied") is False,
        manifest.get("machine_translation_applied") is False,
        language_matches(result.requested_language, manifest.get("language_provided")),
    )):
        raise AssertionError("subtitle structure or provenance is inconsistent")
    # A caption set with the right cue count but the wrong clock is not
    # usable. SRT stores 1 ms precision; ASS stores 10 ms precision.
    for source_cue, srt_cue, ass_cue in zip(raw, srt, ass):
        if (
            abs(source_cue.start - srt_cue.start) > 0.002
            or abs(source_cue.end - srt_cue.end) > 0.002
            or abs(source_cue.start - ass_cue.start) > 0.015
            or abs(source_cue.end - ass_cue.end) > 0.015
        ):
            raise AssertionError("normalized subtitle timeline differs from original")
    if any(key in manifest for key in ("url", "cookies", "authorization", "subtitle_text")):
        raise AssertionError("unexpected sensitive manifest field")
    return result.cue_count


def probe_one(url: str, language: str, collector, *, max_seconds: int = 180) -> dict:
    """Run one public item; report a real download only after verification."""
    site = supported_public_site(url)
    if site not in {"youtube", "wetv", "vimeo", "dailymotion"}:
        raise ValueError("only explicitly approved public HTTPS video URLs are supported")
    language = normalize_language(language)
    started = time.monotonic()
    collector.requested_language = language
    # Never carry successful metadata counts into the next site's failure.
    collector.candidate_count = {"manual": 0, "automatic": 0}
    report = {"site": site, "language": language, "status": "unverified",
              "cue_count": 0, "files_verified": False,
              "track_origin": "", "language_provided": ""}

    def check_deadline() -> None:
        if time.monotonic() - started > max_seconds:
            raise ProbeDeadline("live diagnostic deadline reached")

    with tempfile.TemporaryDirectory(prefix="hse-public-live-") as temporary:
        try:
            results = collector.collect(
                url, Path(temporary), languages=(language,),
                check_cancelled=check_deadline,
            )
            if len(results) != 1:
                raise AssertionError("one-language probe returned an unexpected result")
            result = results[0]
            report["status"] = result.status
            if result.status == "downloaded":
                # A malformed downloaded track is an *integrity failure*,
                # not an ordinary site-access error.
                try:
                    cue_count = _verify_download(result)
                except (AssertionError, ValueError, OSError, UnicodeError):
                    report["status"] = "integrity_failed"
                else:
                    report["cue_count"] = cue_count
                    report["files_verified"] = True
                    report["track_origin"] = result.track_origin
                    report["language_provided"] = result.track_language
        except ProbeDeadline:
            report["status"] = "deadline"
        except CaptionAccessError:
            report["status"] = "access_unavailable"
        except (OSError, ValueError):
            report["status"] = "adapter_error"
        except AssertionError:
            report["status"] = "integrity_failed"
        finally:
            # Explicitly limit the diagnostic to public aggregate counts.
            report["manual_candidates"] = int(getattr(collector, "candidate_count", {}).get("manual", 0))
            report["automatic_candidates"] = int(getattr(collector, "candidate_count", {}).get("automatic", 0))
            report["elapsed_seconds"] = round(time.monotonic() - started, 2)
    # All temporary originals and normalized subtitles have been removed here.
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Temporary no-login public caption diagnostic")
    parser.add_argument("--url", action="append", required=True, help="Public episode URL; never printed")
    parser.add_argument("--languages", default="en")
    parser.add_argument("--yt-dlp", required=True, help="Verified isolated yt-dlp binary")
    parser.add_argument("--max-seconds", type=int, default=180)
    parser.add_argument("--report", required=True)
    parser.add_argument("--require-download", action="store_true")
    args = parser.parse_args(argv)
    if not 10 <= args.max_seconds <= 240:
        parser.error("max-seconds must be between 10 and 240")
    if len(args.url) > 2 or not args.url:
        parser.error("probe one or two public URLs")
    try:
        language = normalize_language(args.languages)
        if any(supported_public_site(url) is None for url in args.url):
            raise ValueError("unsupported URL")
        collector = ObservedCollector(args.yt_dlp)
        findings = [probe_one(url, language, collector, max_seconds=args.max_seconds) for url in args.url]
        report = {"schema": 1, "type": "public_captions_live_probe",
                  "no_credentials": True, "no_video_download": True,
                  "raw_subtitles_retained": False, "results": findings}
        destination = Path(args.report)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        for item in findings:
            print("HSE_PUBLIC_PROBE site={site} lang={language} status={status} cues={cue_count} verified={files_verified} origin={track_origin} provided={language_provided} manual={manual_candidates} auto={automatic_candidates} seconds={elapsed_seconds}".format(**item))
        # Live access failures are observations, never product acceptance.
        # An invalid verified-output invariant *is* a diagnostic failure.
        if any(item["status"] == "integrity_failed" for item in findings):
            return 2
        if args.require_download and any(
            item["status"] != "downloaded" or not item["files_verified"]
            for item in findings
        ):
            return 4
        return 0
    except (ValueError, OSError):
        parser.error("public diagnostic input or output unavailable")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
