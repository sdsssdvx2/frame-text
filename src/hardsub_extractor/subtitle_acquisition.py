"""Opt-in public caption-first acquisition. Never download the video.

Uses a separately installed yt-dlp executable only for public metadata and
available caption tracks. Does not import browser profiles, cookies, tokens,
or DRM keys. Does not synthesize hidden media URLs or perform OCR fallback
implicitly. Existing Korean recognition remains independent.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Callable, Mapping, Sequence
from urllib.parse import urlsplit
import json
import re
import shutil
import subprocess
import tempfile
import time

from .subtitle_formats import SubtitleFormatError, parse_subtitle_bytes, subtitle_cues_have_korean
from .subtitle_language import SubtitleTrack, normalize_language, candidate_tracks
from .writers import write_subtitle_pair_exclusive
from .runtime_paths import bundle_root
from .external_subprocess import external_popen
from .subtitle_proxy import CaptionProxyRoute, route_for_site
from .public_output_audit import audit_soft_output, audit_visual_output

_MAX_METADATA = 12 * 1024 * 1024
_MAX_SUBTITLE = 20 * 1024 * 1024
_VALID_TRACK_TAG = re.compile(r"[A-Za-z0-9_-]{2,28}\Z")
_VIDEO_ID = re.compile(r"[^A-Za-z0-9_-]+")


class CaptionAccessError(RuntimeError):
    """Public metadata or subtitle access could not be verified."""


@dataclass(frozen=True, slots=True)
class CaptionResult:
    source: str
    requested_language: str
    status: str  # downloaded / existing_verified / existing_preserved / needs_ocr / unavailable / proxy_unavailable / unsupported_site
    video_id: str = ""
    track_language: str = ""
    track_origin: str = ""
    cue_count: int = 0
    original: Path | None = None
    srt: Path | None = None
    ass: Path | None = None
    manifest: Path | None = None
    connection_route: str = "direct"


def supported_public_site(url: str) -> str | None:
    """Explicit HTTPS-only entry URL policy; block local and credential URLs."""
    try:
        parsed = urlsplit(url)
        host = (parsed.hostname or "").lower().rstrip(".")
        port = parsed.port
    except ValueError:
        return None
    if (parsed.scheme != "https" or port not in (None, 443)
            or parsed.username or parsed.password
            or not host or not (parsed.path or parsed.query)):
        return None
    if host in {"youtube.com", "www.youtube.com", "m.youtube.com", "youtu.be"}:
        return "youtube"
    if host in {"wetv.vip", "www.wetv.vip"}:
        return "wetv"
    # Explicit public-video URL profiles only: no arbitrary hostname,
    # generic extractor, local server, account or search URL.
    if host in {"vimeo.com", "www.vimeo.com"}:
        if re.fullmatch(r"/[0-9]{1,15}(?:/[A-Za-z0-9]{1,64})?/?", parsed.path):
            return "vimeo"
    if host == "player.vimeo.com":
        if re.fullmatch(r"/video/[0-9]{1,15}/?", parsed.path):
            return "vimeo"
    if host in {"dailymotion.com", "www.dailymotion.com"}:
        if re.fullmatch(r"/(?:video|embed/video)/[A-Za-z0-9]{2,32}/?", parsed.path):
            return "dailymotion"
    if host == "dai.ly":
        if re.fullmatch(r"/[A-Za-z0-9]{2,32}/?", parsed.path):
            return "dailymotion"
    return None


def _safe_video_id(raw: object) -> str:
    ident = _VIDEO_ID.sub("_", str(raw or ""))[:64].strip("_-")
    if not ident:
        raise CaptionAccessError("public extractor did not return a usable video ID")
    return ident


def _safe_track_tag(raw: str) -> str:
    if not _VALID_TRACK_TAG.fullmatch(raw):
        raise CaptionAccessError("invalid subtitle language tag from provider")
    return raw


def _run_silent(
    command: Sequence[str], *, timeout_seconds: float,
    max_stdout_bytes: int,
    check_cancelled: Callable[[], None] | None = None,
) -> bytes:
    """Bounded-memory, abortable external call; never retain provider stderr.

    yt-dlp may stall on a remote site. A GUI cancel or close must terminate
    the child promptly instead of waiting for its full network timeout.
    stdout may contain signed media URLs, so it is never sent to logs.
    """
    if check_cancelled is not None:
        check_cancelled()
    try:
        with tempfile.TemporaryFile() as captured:
            proc = external_popen(
                list(command),
                stdin=subprocess.DEVNULL,
                stdout=captured,
                stderr=subprocess.DEVNULL,
            )
            try:
                deadline = time.monotonic() + timeout_seconds
                while proc.poll() is None:
                    if check_cancelled is not None:
                        check_cancelled()
                    if time.monotonic() >= deadline:
                        raise subprocess.TimeoutExpired(command, timeout_seconds)
                    time.sleep(min(0.2, max(0.0, deadline - time.monotonic())))
                if check_cancelled is not None:
                    check_cancelled()
                if proc.returncode != 0:
                    raise CaptionAccessError(
                        "public subtitle access unavailable for this item"
                    )
                captured.seek(0, 2)
                if captured.tell() > max_stdout_bytes:
                    raise CaptionAccessError(
                        "public subtitle metadata is unexpectedly large"
                    )
                captured.seek(0)
                return captured.read(max_stdout_bytes + 1)
            except BaseException:
                if proc.poll() is None:
                    proc.kill()
                proc.wait()
                raise
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise CaptionAccessError(
            "public subtitle adapter unavailable or timed out"
        ) from exc


def _commit_caption(
    raw: bytes,
    track: SubtitleTrack,
    *,
    video_id: str,
    provider: str,
    target_directory: Path,
    original_extension: str | None = None,
    proxy_active: bool = False,
) -> CaptionResult:
    # Parse before creating output. Original bytes are preserved unmodified.
    if len(raw) > _MAX_SUBTITLE:
        raise SubtitleFormatError("subtitle file is unexpectedly large")
    fmt, cues = parse_subtitle_bytes(raw, source_url="caption." + track.extension)
    if track.requested_language == "ko" and not subtitle_cues_have_korean(cues):
        raise SubtitleFormatError("track labeled Korean lacks Korean text")
    language = _safe_track_tag(track.language)
    if provider not in {"youtube", "wetv", "vimeo", "dailymotion"}:
        raise ValueError("unsupported caption provenance site")
    name = f"{provider}-{_safe_video_id(video_id)}-{language}-{track.origin}"
    base = target_directory / name
    actual_extension = (original_extension or fmt).lower().lstrip(".")
    if actual_extension not in {"vtt", "srt", "ass", "ssa"}:
        raise SubtitleFormatError("unsupported raw subtitle extension")
    original = target_directory / f"{name}.original.{actual_extension}"
    report = target_directory / f"{name}.source.json"
    srt = base.with_suffix(".srt")
    ass = base.with_suffix(".ass")
    for dest in (original, report, srt, ass):
        if dest.exists() or dest.is_symlink():
            raise FileExistsError(
                "기존 자막이 있습니다. 새 저장 폴더를 선택하여 원본 파일을 보존하세요."
            )
    target_directory.mkdir(parents=True, exist_ok=True)
    created: list[Path] = []
    try:
        with original.open("xb") as stream:
            created.append(original)
            stream.write(raw)
        new_srt, new_ass = write_subtitle_pair_exclusive(cues, base)
        created.extend((new_srt, new_ass))
        metadata = {
            "schema": 1,
            "site": provider,
            "video_id": _safe_video_id(video_id),
            "language_requested": track.requested_language,
            "language_provided": language,
            "track_origin": track.origin,
            "language_evidence": "provider_metadata_and_hangul" if track.requested_language == "ko" else "provider_metadata_only",
            "original_format": fmt,
            "original_timeline": "source_video",
            "output_timeline": "source_video",
            "network_route": "proxy" if proxy_active else "direct",
            "ocr_applied": False,
            "machine_translation_applied": False,  # Not translated by HSE.
            "provider_translation_status": (
                "original" if track.origin == "automatic" and track.language.lower().endswith("-orig")
                else "unknown" if track.origin == "automatic" else "not_applicable"
            ),
            "cue_count": len(cues),
            "original_file": original.name,
            "srt_file": new_srt.name,
            "ass_file": new_ass.name,
        }
        with report.open("x", encoding="utf-8") as stream:
            created.append(report)
            json.dump(metadata, stream, ensure_ascii=False, indent=2)
    except BaseException:
        # Only files created by this invocation are eligible for rollback.
        for path in reversed(created):
            try:
                path.unlink(missing_ok=True)
            except OSError:
                pass
        raise
    return CaptionResult(
        provider, track.requested_language, "downloaded",
        video_id=_safe_video_id(video_id),
        track_language=track.language, track_origin=track.origin,
        cue_count=len(cues), original=original, srt=new_srt,
        ass=new_ass, manifest=report,
        connection_route="proxy" if proxy_active else "direct",
    )


def find_caption_downloader() -> str | None:
    """Prefer the verified companion executable extracted with a frozen HSE.

    Never import a browser profile or user credentials. In source checkouts,
    also accept an existing manually installed yt-dlp on PATH, but the
    production onefile EXE includes its own SHA-pinned copy.
    """
    # Opt-in, SHA-checked versioned free downloader cache. An invalid or
    # disabled managed release never hides the verified bundled fallback.
    from .managed_caption_downloader import active_downloader
    managed = active_downloader()
    if managed is not None:
        return managed
    candidate = bundle_root() / "bin" / "yt-dlp.exe"
    if candidate.is_file():
        return str(candidate)
    return shutil.which("yt-dlp")


def _has_existing_caption_output(
    root: Path, *, provider: str, video_id: str, track: SubtitleTrack,
) -> bool:
    """Find complete or partial results without opening or changing them."""
    tag = _safe_track_tag(track.language)
    stem = f"{provider}-{_safe_video_id(video_id)}-{tag}-{track.origin}"
    suffixes = (
        ".source.json", ".srt", ".ass",
        ".original.vtt", ".original.srt", ".original.ass", ".original.ssa",
    )
    return any(
        (root / (stem + suffix)).exists()
        or (root / (stem + suffix)).is_symlink()
        for suffix in suffixes
    )


class PublicCaptionCollector:
    """Read public metadata then acquire selected subtitle tracks only.

    yt-dlp is optional and external to HSE; nothing is installed automatically.
    """

    def __init__(
        self, executable: str | None = None,
        *,
        command_runner: Callable[..., bytes] = _run_silent,
        proxy_route: CaptionProxyRoute | None = None,
        site_proxy_routes: Mapping[str, CaptionProxyRoute] | None = None,
    ) -> None:
        self.executable = executable or find_caption_downloader()
        self.command_runner = command_runner
        self.proxy_route = proxy_route if proxy_route is not None else CaptionProxyRoute()
        self.site_proxy_routes = dict(site_proxy_routes or {})

    def _run(
        self, args: list[str], *, timeout: float, max_stdout: int,
        check_cancelled: Callable[[], None] | None = None,
        proxy_url: str | None = None,
    ) -> bytes:
        if not self.executable:
            raise CaptionAccessError("yt-dlp is not installed or configured")
        kwargs = {"timeout_seconds": timeout, "max_stdout_bytes": max_stdout}
        # Preserve the original injected fake-runner API used by offline tests.
        # The production subprocess runner additionally supports live cancel.
        if self.command_runner is _run_silent:
            kwargs["check_cancelled"] = check_cancelled
        return self.command_runner(
            [self.executable, "--ignore-config", "--no-plugin-dirs",
             "--no-playlist", "--skip-download", "--no-warnings", "--no-progress",
             "--proxy", proxy_url or "", *args],
            **kwargs,
        )

    def _metadata(
        self, url: str, *,
        check_cancelled: Callable[[], None] | None = None,
        proxy_url: str | None = None,
    ) -> dict:
        output = self._run(
            ["--dump-single-json", "--", url],
            timeout=75, max_stdout=_MAX_METADATA,
            check_cancelled=check_cancelled,
            proxy_url=proxy_url,
        )
        try:
            meta = json.loads(output)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise CaptionAccessError("public subtitle metadata was invalid") from exc
        if not isinstance(meta, dict) or meta.get("_type") in {"playlist", "multi_video"}:
            raise CaptionAccessError("single episode URL required, not a playlist")
        return meta

    def collect(
        self,
        url: str,
        out_dir: str | Path,
        *,
        languages: Sequence[str] = ("ko",),
        check_cancelled: Callable[[], None] | None = None,
        skip_existing: bool = False,
    ) -> list[CaptionResult]:
        """Use one explicit route per attempt, shared by metadata and bytes.

        Fallback is opt-in. It first uses DIRECT and retries only missing
        language tracks through the configured proxy. Required-proxy mode
        never attempts DIRECT even after a proxy network failure.
        """
        site = supported_public_site(url) or ""
        route = route_for_site(site, self.proxy_route, self.site_proxy_routes)

        def attempt(langs: Sequence[str], proxy: str | None) -> list[CaptionResult]:
            return self._collect_once(
                url, out_dir, languages=langs,
                check_cancelled=check_cancelled,
                skip_existing=skip_existing, proxy_url=proxy,
            )

        if route.mode == "direct":
            return attempt(languages, None)
        if route.mode == "required":
            return attempt(languages, route.url)

        try:
            direct = attempt(languages, None)
        except CaptionAccessError:
            # The metadata fetch can fail before any file is written.
            return attempt(languages, route.url)

        missing = [
            result.requested_language for result in direct
            if result.status in {"needs_ocr", "unavailable"}
        ]
        if not missing:
            return direct
        try:
            proxied = attempt(missing, route.url)
        except CaptionAccessError as exc:
            # Retain successfully committed DIRECT tracks. Record proxy
            # failure on just the missing languages instead of discarding
            # correct partial results from a multi-language job.
            if any(
                item.status in {"downloaded", "existing_verified", "existing_preserved"}
                for item in direct
            ):
                return [
                    replace(item, status="proxy_unavailable", connection_route="proxy")
                    if item.status in {"needs_ocr", "unavailable"} else item
                    for item in direct
                ]
            raise CaptionAccessError("configured proxy retry unavailable") from exc

        by_language = {result.requested_language: result for result in proxied}
        return [
            by_language.get(result.requested_language, result)
            if result.status in {"needs_ocr", "unavailable"} else result
            for result in direct
        ]

    def _collect_once(
        self,
        url: str,
        out_dir: str | Path,
        *,
        languages: Sequence[str] = ("ko",),
        check_cancelled: Callable[[], None] | None = None,
        skip_existing: bool = False,
        proxy_url: str | None = None,
    ) -> list[CaptionResult]:
        if check_cancelled is not None:
            check_cancelled()
        requested = list(dict.fromkeys(normalize_language(lang) for lang in languages))
        if not requested or len(requested) > 6:
            raise ValueError("request one to six supported caption languages")
        provider = supported_public_site(url)
        if provider is None:
            return [
                CaptionResult("unsupported", language, "unsupported_site")
                for language in requested
            ]
        meta = self._metadata(
            url, check_cancelled=check_cancelled, proxy_url=proxy_url,
        )
        if check_cancelled is not None:
            check_cancelled()
        video_id = _safe_video_id(meta.get("id"))
        manual = meta.get("subtitles") or {}
        automatic = meta.get("automatic_captions") or {}
        root = Path(out_dir)
        results: list[CaptionResult] = []

        for language in requested:
            if check_cancelled is not None:
                check_cancelled()
            tracks = candidate_tracks(language, manual, automatic)
            if not tracks:
                # Even with --video-ocr disabled, a previous successfully
                # exported visual subtitle is still a completed result.
                # Inspect ONLY the known local output stem; no video download
                # or OCR/model initialization is needed for restart verification.
                if skip_existing and language in {"ko", "en", "ja", "zh"}:
                    visual_base = root / f"{provider}-{video_id}-{language}-hard-ocr"
                    saved = audit_visual_output(
                        visual_base, provider=provider, video_id=video_id,
                        language=language,
                    )
                    if saved.state != "absent":
                        results.append(CaptionResult(
                            provider, language,
                            "existing_verified" if saved.state == "verified"
                            else "existing_preserved",
                            video_id=video_id, track_language=language,
                            track_origin="visual_ocr", cue_count=saved.cue_count,
                            srt=saved.srt, ass=saved.ass, manifest=saved.manifest,
                            connection_route=saved.network_route
                            if saved.state == "verified"
                            else ("proxy" if proxy_url else "direct"),
                        ))
                        continue
                results.append(CaptionResult(
                    provider, language, "needs_ocr", video_id=video_id,
                    connection_route="proxy" if proxy_url else "direct",
                ))
                continue

            if skip_existing:
                # Scan only the bounded provider-advertised candidates.
                # Prefer an independently verified complete output even if an
                # earlier manual candidate has an incomplete user-owned file.
                # Never write into any existing candidate, valid or partial.
                verified = None
                partial = None
                for existing in tracks:
                    if not _has_existing_caption_output(
                        root, provider=provider, video_id=video_id, track=existing,
                    ):
                        continue
                    base = root / (
                        f"{provider}-{video_id}-{_safe_track_tag(existing.language)}-{existing.origin}"
                    )
                    audit = audit_soft_output(
                        base, provider=provider, video_id=video_id,
                        requested=language, provided=existing.language,
                        origin=existing.origin,
                    )
                    if audit.state == "verified":
                        verified = (existing, audit)
                        break
                    if partial is None:
                        partial = existing
                if verified is not None:
                    existing, audit = verified
                    results.append(CaptionResult(
                        provider, language, "existing_verified", video_id=video_id,
                        track_language=existing.language, track_origin=existing.origin,
                        cue_count=audit.cue_count, original=audit.original,
                        srt=audit.srt, ass=audit.ass, manifest=audit.manifest,
                        connection_route=audit.network_route,
                    ))
                    continue
                if partial is not None:
                    results.append(CaptionResult(
                        provider, language, "existing_preserved", video_id=video_id,
                        track_language=partial.language, track_origin=partial.origin,
                    ))
                    continue

            downloaded: CaptionResult | None = None
            # Prefer matching manual alternatives, then matching automatic
            # alternatives. All attempts are bounded by candidate_tracks.
            for track in tracks:
                if check_cancelled is not None:
                    check_cancelled()
                tag = _safe_track_tag(track.language)
                flag = "--write-subs" if track.origin == "manual" else "--write-auto-subs"
                try:
                    root.mkdir(parents=True, exist_ok=True)
                    with tempfile.TemporaryDirectory(prefix=".hse-captions-", dir=root) as td:
                        output_template = str(Path(td) / "caption.%(ext)s")
                        self._run([
                            flag, "--sub-langs", "^" + re.escape(tag) + "$",
                            "--sub-format", track.extension,
                            "--output", output_template, "--", url,
                        ], timeout=90, max_stdout=512 * 1024,
                            check_cancelled=check_cancelled, proxy_url=proxy_url)
                        matches = [
                            path for path in Path(td).iterdir()
                            if path.is_file() and not path.is_symlink()
                            and path.suffix.lower() in {".vtt", ".srt", ".ass", ".ssa"}
                            and f".{tag.lower()}." in path.name.lower()
                        ]
                        if len(matches) != 1:
                            raise CaptionAccessError(
                                "requested subtitle format was not published"
                            )
                        if matches[0].stat().st_size > _MAX_SUBTITLE:
                            raise CaptionAccessError("downloaded subtitle exceeds limit")
                        raw = matches[0].read_bytes()
                        if check_cancelled is not None:
                            check_cancelled()
                        if len(raw) > _MAX_SUBTITLE:
                            raise CaptionAccessError("downloaded subtitle exceeds limit")
                        downloaded = _commit_caption(
                            raw, track, video_id=video_id, provider=provider,
                            target_directory=root,
                            original_extension=matches[0].suffix,
                            proxy_active=bool(proxy_url),
                        )
                    break
                except (CaptionAccessError, SubtitleFormatError):
                    continue
                except FileExistsError:
                    if not skip_existing:
                        raise
                    audit = audit_soft_output(
                        root / (f"{provider}-{video_id}-{_safe_track_tag(track.language)}-{track.origin}"),
                        provider=provider, video_id=video_id,
                        requested=language, provided=track.language, origin=track.origin,
                    )
                    downloaded = CaptionResult(
                        provider, language,
                        "existing_verified" if audit.state == "verified" else "existing_preserved",
                        video_id=video_id, track_language=track.language,
                        track_origin=track.origin,
                        cue_count=audit.cue_count, original=audit.original,
                        srt=audit.srt, ass=audit.ass, manifest=audit.manifest,
                        connection_route=audit.network_route if audit.state == "verified"
                        else ("proxy" if proxy_url else "direct"),
                    )
                    break
            if downloaded is not None:
                results.append(downloaded)
            else:
                results.append(CaptionResult(
                    provider, language, "unavailable", video_id=video_id,
                    connection_route="proxy" if proxy_url else "direct",
                ))
        return results
