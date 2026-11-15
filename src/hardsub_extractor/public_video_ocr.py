"""Bounded, optional public video OCR after real manual/automatic subtitle absence.

Only explicitly supported HTTPS provider-video URLs; no credentials, browser profile,
DRM support, protected media requests or persistent downloaded video.
An explicit no-track result triggers a temporary video download with resource
caps. Failed advertised subtitle downloads and blocked media do not trigger
visual OCR. Actual OCR remains original-language, not machine translation.
"""
from __future__ import annotations

from dataclasses import replace
import json
import math
from pathlib import Path
import subprocess
import tempfile
import time
from typing import Callable, Mapping, Sequence

from .external_subprocess import external_popen
from .extraction_progress import ExtractionMonitor
from .model_assets import normalize_ocr_language
from .subtitle_acquisition import (
    CaptionResult, _safe_video_id, supported_public_site, find_caption_downloader,
)
from .subtitle_proxy import CaptionProxyRoute, route_for_site
from .runtime_paths import bundled_model_dir, bundled_manifest_path, resolve_binary
from .public_output_audit import audit_visual_output

MAX_PUBLIC_VIDEO_BYTES = 512 * 1024 * 1024
MAX_PUBLIC_VIDEO_SECONDS = 3600.0
MAX_DOWNLOAD_SECONDS = 900.0
_VIDEO_EXTENSIONS = frozenset({".mp4", ".mkv", ".webm", ".mov", ".m4v", ".ts", ".flv"})
_OCR_LANGUAGES = frozenset({"ko", "en", "ja", "zh"})


class PublicVideoOCRError(RuntimeError):
    """Public visual OCR unavailable or exceeded bounded resource limits."""


def _scan_video_directory(folder: Path, byte_limit: int) -> None:
    size = 0
    for entry in folder.iterdir():
        if entry.is_symlink() or not entry.is_file():
            raise PublicVideoOCRError("unexpected media download output")
        size += entry.stat().st_size
        if size > byte_limit:
            raise PublicVideoOCRError("public OCR video exceeds temporary size limit")


def _video_file(folder: Path, byte_limit: int) -> Path:
    _scan_video_directory(folder, byte_limit)
    choices = [
        item for item in folder.iterdir()
        if item.is_file() and not item.is_symlink()
        and item.name.startswith("video.")
        and item.suffix.lower() in _VIDEO_EXTENSIONS
        and item.stat().st_size > 0
    ]
    if len(choices) != 1:
        raise PublicVideoOCRError("public downloader did not produce one usable video")
    return choices[0]


def _probe_local_video_for_ocr(path: Path) -> None:
    """Require finite media duration/resolution before loading OCR models."""
    command = [
        resolve_binary("ffprobe"), "-v", "error", "-protocol_whitelist",
        "file,pipe", "-select_streams", "v:0",
        "-show_entries", "stream=width,height:format=duration",
        "-of", "json", "-i", str(path),
    ]
    try:
        probe = subprocess.run(
            command, input=b"", stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
            timeout=40, check=False,
        )
        if probe.returncode or len(probe.stdout) > 128 * 1024:
            raise PublicVideoOCRError("temporary video probe failed")
        parsed = json.loads(probe.stdout)
        stream = parsed["streams"][0]
        width, height = int(stream["width"]), int(stream["height"])
        duration = float(parsed["format"]["duration"])
    except (OSError, subprocess.TimeoutExpired, KeyError, ValueError, IndexError,
            TypeError, OverflowError) as exc:
        raise PublicVideoOCRError("temporary video metadata unavailable") from exc
    if not (16 <= width <= 3840 and 16 <= height <= 2160
            and math.isfinite(duration) and 0 < duration <= MAX_PUBLIC_VIDEO_SECONDS):
        raise PublicVideoOCRError("temporary video duration or resolution exceeds OCR limit")


def download_public_video_for_ocr(
    url: str,
    folder: Path,
    *,
    executable: str | None = None,
    proxy_url: str | None = None,
    check_cancelled: Callable[[], None] | None = None,
    byte_limit: int = MAX_PUBLIC_VIDEO_BYTES,
    timeout_seconds: float = MAX_DOWNLOAD_SECONDS,
) -> Path:
    """Fetch public video only in a new isolated temporary folder, bounded."""
    if supported_public_site(url) not in {"youtube", "wetv", "vimeo", "dailymotion"}:
        raise ValueError("public OCR supports explicitly approved HTTPS video URLs only")
    if not (1 <= byte_limit <= MAX_PUBLIC_VIDEO_BYTES):
        raise ValueError("invalid bounded video capacity")
    if not (1 <= timeout_seconds <= MAX_DOWNLOAD_SECONDS):
        raise ValueError("invalid bounded download deadline")
    if not folder.is_dir() or any(folder.iterdir()):
        raise ValueError("video download requires an empty temporary folder")
    binary = executable or find_caption_downloader()
    if not binary:
        raise PublicVideoOCRError("public video downloader unavailable")
    if check_cancelled is not None:
        check_cancelled()
    cmd = [
        str(binary), "--ignore-config", "--no-plugin-dirs", "--no-playlist",
        "--no-warnings", "--no-progress", "--no-simulate",
        "--no-write-subs", "--no-write-auto-subs", "--no-write-thumbnail",
        "--no-write-description", "--no-allow-unplayable-formats",
        "--no-overwrites", "--no-mtime", "--proxy", proxy_url or "",
        "--format", "bv[height<=720]/b[height<=720]",
        "--max-filesize", str(byte_limit),
        "--output", str(folder / "video.%(ext)s"), "--", url,
    ]
    try:
        process = external_popen(
            cmd, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except OSError as exc:
        raise PublicVideoOCRError("public video downloader could not start") from exc
    try:
        deadline = time.monotonic() + timeout_seconds
        while process.poll() is None:
            if check_cancelled is not None:
                check_cancelled()
            _scan_video_directory(folder, byte_limit)
            if time.monotonic() >= deadline:
                raise PublicVideoOCRError("public video download deadline exceeded")
            time.sleep(min(0.2, max(0.0, deadline - time.monotonic())))
        if check_cancelled is not None:
            check_cancelled()
        if process.returncode != 0:
            raise PublicVideoOCRError("public video access refused or unavailable")
        media = _video_file(folder, byte_limit)
        _probe_local_video_for_ocr(media)
        return media
    except BaseException:
        if process.poll() is None:
            process.kill()
        process.wait()
        raise


def _output_base(root: Path, result: CaptionResult) -> Path:
    if result.source not in {"youtube", "wetv", "vimeo", "dailymotion"}:
        raise ValueError("unknown public OCR site")
    language = result.requested_language
    if language not in _OCR_LANGUAGES:
        raise ValueError("source-language OCR supports only ko/en/ja/zh")
    return root / (
        f"{result.source}-{_safe_video_id(result.video_id)}-{language}-hard-ocr"
    )


def _output_files(base: Path) -> tuple[Path, Path, Path]:
    return (
        base.with_suffix(".srt"), base.with_suffix(".ass"),
        base.with_name(base.name + ".source.json"),
    )


def _existing(base: Path) -> bool:
    return any(p.exists() or p.is_symlink() for p in _output_files(base))


def _existing_result(base: Path, original: CaptionResult) -> CaptionResult:
    audit = audit_visual_output(
        base, provider=original.source,
        video_id=_safe_video_id(original.video_id),
        language=original.requested_language,
    )
    if audit.state == "verified":
        return replace(
            original, status="existing_verified",
            track_language=original.requested_language,
            track_origin="visual_ocr",
            cue_count=audit.cue_count,
            srt=audit.srt, ass=audit.ass, manifest=audit.manifest,
            connection_route=audit.network_route,
        )
    return replace(original, status="existing_preserved")


def _ocr_engine(language: str):
    from .ocr.rapidocr_adapter import RapidOCRAdapter
    from .ocr.adaptive import AdaptiveOCREngine

    lang = normalize_ocr_language(language)
    model_dir = bundled_model_dir()
    if not model_dir.is_dir():
        raise RuntimeError("public video OCR model bundle unavailable")
    return AdaptiveOCREngine(RapidOCRAdapter(
        model_dir=str(model_dir),
        manifest_path=str(bundled_manifest_path()), language=lang,
    ))


def _extract_ocr(
    media: Path, base: Path, language: str,
    monitor: ExtractionMonitor | None,
):
    from .pipeline import extract
    active_monitor = monitor or ExtractionMonitor(exclusive_outputs=True)
    if not active_monitor.exclusive_outputs:
        raise ValueError("public video OCR requires exclusive subtitle publication")
    return extract(
        str(media), _ocr_engine(language), base, monitor=active_monitor,
        sample_fps=8.0, streaming=True,
    )


def _finalize_ocr_result(original: CaptionResult, extracted, base: Path) -> CaptionResult:
    srt, ass, manifest = _output_files(base)
    if (extracted.srt_path != srt or extracted.ass_path != ass
            or extracted.cue_count <= 0):
        raise PublicVideoOCRError("unexpected visual OCR outputs")
    metadata = {
        "schema": 1,
        "site": original.source,
        "video_id": _safe_video_id(original.video_id),
        "language_requested": original.requested_language,
        "language_provided": "ocr_visual_candidate",
        "track_origin": "visual_ocr",
        "ocr_applied": True,
        "original_text_verified": False,
        "machine_translation_applied": False,
        "original_timeline": "source_video",
        "output_timeline": "source_video",
        "network_route": original.connection_route,
        "cue_count": extracted.cue_count,
        "srt_file": srt.name,
        "ass_file": ass.name,
        "temporary_video_retained": False,
    }
    try:
        with manifest.open("x", encoding="utf-8") as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)
            f.write("\n")
    except BaseException:
        # Extract published both files exclusively within this invocation.
        srt.unlink(missing_ok=True)
        ass.unlink(missing_ok=True)
        raise
    return replace(
        original, status="hard_ocr", track_language=original.requested_language,
        track_origin="visual_ocr", cue_count=extracted.cue_count,
        srt=srt, ass=ass, manifest=manifest,
    )


def resolve_missing_public_ocr(
    url: str,
    out_dir: str | Path,
    results: Sequence[CaptionResult],
    *,
    proxy_route: CaptionProxyRoute | None = None,
    site_proxy_routes: Mapping[str, CaptionProxyRoute] | None = None,
    executable: str | None = None,
    media_fetcher: Callable[..., Path] = download_public_video_for_ocr,
    ocr_extractor: Callable[..., object] = _extract_ocr,
    monitor: ExtractionMonitor | None = None,
    skip_existing: bool = True,
    max_video_bytes: int = MAX_PUBLIC_VIDEO_BYTES,
) -> list[CaptionResult]:
    """Replace only verified absent soft tracks; share one temp video per route."""
    site = supported_public_site(url)
    if site not in {"youtube", "wetv", "vimeo", "dailymotion"}:
        raise ValueError("public OCR URL is not on the verified site allowlist")
    original = list(results)
    if not original:
        return []
    if any(item.source != site for item in original):
        raise ValueError("public OCR results do not match episode provider")
    if any(item.requested_language not in {
        "ko", "en", "ja", "zh", "zh-hans", "zh-hant",
    } for item in original):
        raise ValueError("unvalidated public OCR language")
    if monitor is not None:
        monitor.check_cancelled()
    root = Path(out_dir).expanduser()
    output = list(original)
    pending_by_route: dict[str, list[tuple[int, CaptionResult, Path]]] = {}
    for index, result in enumerate(original):
        if result.status != "needs_ocr" or result.requested_language not in _OCR_LANGUAGES:
            continue
        base = _output_base(root, result)
        if _existing(base):
            if not skip_existing:
                raise FileExistsError("visual OCR output already exists")
            output[index] = _existing_result(base, result)
            continue
        if result.connection_route not in {"direct", "proxy"}:
            raise ValueError("unverified public OCR connection route")
        pending_by_route.setdefault(result.connection_route, []).append((index, result, base))
    if not pending_by_route:
        return output

    route = route_for_site(site, proxy_route or CaptionProxyRoute(), site_proxy_routes)
    if route.mode == "required" and "direct" in pending_by_route:
        raise ValueError("required-proxy mode forbids direct video OCR")
    if route.mode == "direct" and "proxy" in pending_by_route:
        raise ValueError("direct mode cannot claim proxy video OCR")

    root.mkdir(parents=True, exist_ok=True)
    for route_label, items in pending_by_route.items():
        if monitor is not None:
            monitor.check_cancelled()
        requested_proxy = route.url if route_label == "proxy" else None
        with tempfile.TemporaryDirectory(prefix=".hse-visual-ocr-", dir=root) as td:
            folder = Path(td)
            try:
                try:
                    media = media_fetcher(
                        url, folder, executable=executable, proxy_url=requested_proxy,
                        check_cancelled=monitor.check_cancelled if monitor else None,
                        byte_limit=max_video_bytes,
                    )
                    actual_route = route_label
                except PublicVideoOCRError:
                    if route.mode != "fallback" or route_label != "direct":
                        raise
                    # Retry is opt-in. Reset only contents created in THIS temp
                    # directory; the required-proxy route must never try DIRECT.
                    for child in folder.iterdir():
                        if child.is_symlink() or not child.is_file():
                            raise PublicVideoOCRError("download temp output untrusted")
                        child.unlink()
                    media = media_fetcher(
                        url, folder, executable=executable, proxy_url=route.url,
                        check_cancelled=monitor.check_cancelled if monitor else None,
                        byte_limit=max_video_bytes,
                    )
                    actual_route = "proxy"
            except PublicVideoOCRError:
                for index, result, _ in items:
                    output[index] = replace(result, status="ocr_unavailable")
                continue

            if not media.is_file() or media.is_symlink() or media.parent != folder:
                raise PublicVideoOCRError("OCR media must be a temporary video file")
            for index, result, base in items:
                if monitor is not None:
                    monitor.check_cancelled()
                if _existing(base):
                    if not skip_existing:
                        raise FileExistsError("visual OCR output appeared before recognition")
                    output[index] = _existing_result(base, result)
                    continue
                try:
                    extracted = ocr_extractor(
                        media, base, result.requested_language, monitor,
                    )
                except RuntimeError as exc:
                    # Keep later languages/episodes alive when this one has
                    # no visible subtitles or a model/decoder fails. Preserve
                    # unexpected partially published files as an explicit
                    # error: we cannot distinguish them safely from a
                    # concurrent user's output and must never delete them.
                    if _existing(base):
                        raise
                    no_text = str(exc).startswith("인식된 자막이 없습니다")
                    output[index] = replace(
                        result, status="ocr_unavailable" if no_text else "ocr_failed",
                    )
                    continue
                updated = replace(result, connection_route=actual_route)
                output[index] = _finalize_ocr_result(updated, extracted, base)
    return output
