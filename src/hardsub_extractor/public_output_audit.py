"""Read-only restart audit for existing public subtitle and OCR results.

Only independently constructed, expected sibling paths are opened. Metadata
is *checked*, never used as a file locator. No network calls, file writes,
deletes, renames, repairs, auth state or OCR model initialization.

This is a structural consistency audit, NOT a cryptographic seal: schema 1
historical outputs do not have pinned content hashes. Never call a partial
result complete merely because one .srt or .source.json file exists.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import stat

from .subtitle_formats import SubtitleFormatError, parse_subtitle_bytes
from .subtitle_language import language_matches

_MAX_SUBTITLE_BYTES = 20 * 1024 * 1024
_MAX_PROVENANCE_BYTES = 64 * 1024
_RAW_EXTS = ("vtt", "srt", "ass", "ssa")


@dataclass(frozen=True, slots=True)
class ExistingOutputAudit:
    state: str  # absent / verified / preserved
    cue_count: int = 0
    srt: Path | None = None
    ass: Path | None = None
    manifest: Path | None = None
    original: Path | None = None
    network_route: str = "direct"


def _safe_read(path: Path, max_bytes: int) -> bytes:
    """Bound input even if a file changes between stat and read."""
    entry = path.lstat()
    if not stat.S_ISREG(entry.st_mode) or path.is_symlink():
        raise ValueError("existing result is not a regular non-symlink file")
    if not 0 < entry.st_size <= max_bytes:
        raise ValueError("existing result size is invalid")
    with path.open("rb") as stream:
        raw = stream.read(max_bytes + 1)
    post = path.lstat()
    if (not stat.S_ISREG(post.st_mode) or path.is_symlink()
            or post.st_size != entry.st_size or post.st_mtime_ns != entry.st_mtime_ns
            or post.st_ino != entry.st_ino or len(raw) != entry.st_size):
        raise ValueError("existing result changed during read")
    return raw


def _present(path: Path) -> bool:
    return path.exists() or path.is_symlink()


def _parse(path: Path):
    return parse_subtitle_bytes(
        _safe_read(path, _MAX_SUBTITLE_BYTES), source_url=path.name,
    )


def _manifest(path: Path) -> dict:
    raw = _safe_read(path, _MAX_PROVENANCE_BYTES)
    value = json.loads(raw.decode("utf-8"))
    if not isinstance(value, dict):
        raise ValueError("existing subtitle metadata is not an object")
    return value


def _matching_cues(original, srt, ass) -> bool:
    if not original or len(original) != len(srt) or len(original) != len(ass):
        return False
    for a, b, c in zip(original, srt, ass):
        if not a.text.strip() or a.text != b.text or a.text != c.text:
            return False
        if (abs(a.start - b.start) > 0.002
                or abs(a.end - b.end) > 0.002
                or abs(a.start - c.start) > 0.015
                or abs(a.end - c.end) > 0.015):
            return False
    return True


def audit_soft_output(
    base: Path,
    *,
    provider: str,
    video_id: str,
    requested: str,
    provided: str,
    origin: str,
) -> ExistingOutputAudit:
    """Verify one known manual/automatic provider track without changes."""
    manifest = base.with_name(base.name + ".source.json")
    srt, ass = base.with_suffix(".srt"), base.with_suffix(".ass")
    originals = [base.with_name(base.name + f".original.{ext}") for ext in _RAW_EXTS]
    candidates = (manifest, srt, ass, *originals)
    if not any(_present(p) for p in candidates):
        return ExistingOutputAudit("absent")
    preserved = ExistingOutputAudit("preserved")
    try:
        present_originals = [p for p in originals if _present(p)]
        if (len(present_originals) != 1 or
                not all(_present(p) for p in (manifest, srt, ass))):
            return preserved
        original = present_originals[0]
        metadata = _manifest(manifest)
        raw_format, raw_cues = _parse(original)
        srt_format, srt_cues = _parse(srt)
        ass_format, ass_cues = _parse(ass)
        if srt_format != "srt" or ass_format != "ass":
            return preserved
        if not _matching_cues(raw_cues, srt_cues, ass_cues):
            return preserved
        if (type(metadata.get("schema")) is not int or metadata["schema"] != 1
                or metadata.get("site") != provider
                or metadata.get("video_id") != video_id
                or metadata.get("language_requested") != requested
                or metadata.get("language_provided") != provided
                or not language_matches(requested, provided)
                or metadata.get("track_origin") != origin
                or metadata.get("original_format") != raw_format
                or metadata.get("original_file") != original.name
                or metadata.get("srt_file") != srt.name
                or metadata.get("ass_file") != ass.name
                or metadata.get("original_timeline") != "source_video"
                or metadata.get("output_timeline") != "source_video"
                or metadata.get("ocr_applied") is not False
                or metadata.get("machine_translation_applied") is not False
                or metadata.get("network_route") not in {"direct", "proxy"}
                or type(metadata.get("cue_count")) is not int
                or metadata["cue_count"] != len(raw_cues)):
            return preserved
        return ExistingOutputAudit(
            "verified", len(raw_cues), srt, ass, manifest, original,
            metadata["network_route"],
        )
    except (OSError, ValueError, TypeError, UnicodeError, RecursionError,
            SubtitleFormatError):
        return preserved


def audit_visual_output(
    base: Path,
    *,
    provider: str,
    video_id: str,
    language: str,
) -> ExistingOutputAudit:
    """Verify existing source-language visual OCR SRT/ASS and sidecar."""
    manifest = base.with_name(base.name + ".source.json")
    srt, ass = base.with_suffix(".srt"), base.with_suffix(".ass")
    paths = (manifest, srt, ass)
    if not any(_present(p) for p in paths):
        return ExistingOutputAudit("absent")
    preserved = ExistingOutputAudit("preserved")
    try:
        if not all(_present(p) for p in paths):
            return preserved
        metadata = _manifest(manifest)
        srt_format, srt_cues = _parse(srt)
        ass_format, ass_cues = _parse(ass)
        if (srt_format != "srt" or ass_format != "ass"
                or not _matching_cues(srt_cues, srt_cues, ass_cues)):
            return preserved
        if (type(metadata.get("schema")) is not int or metadata["schema"] != 1
                or metadata.get("site") != provider
                or metadata.get("video_id") != video_id
                or metadata.get("language_requested") != language
                or metadata.get("language_provided") != "ocr_visual_candidate"
                or metadata.get("track_origin") != "visual_ocr"
                or metadata.get("original_timeline") != "source_video"
                or metadata.get("output_timeline") != "source_video"
                or metadata.get("ocr_applied") is not True
                or metadata.get("original_text_verified") is not False
                or metadata.get("machine_translation_applied") is not False
                or metadata.get("temporary_video_retained") is not False
                or metadata.get("network_route") not in {"direct", "proxy"}
                or metadata.get("srt_file") != srt.name
                or metadata.get("ass_file") != ass.name
                or type(metadata.get("cue_count")) is not int
                or metadata["cue_count"] != len(srt_cues)):
            return preserved
        return ExistingOutputAudit(
            "verified", len(srt_cues), srt, ass, manifest, None,
            metadata["network_route"],
        )
    except (OSError, ValueError, TypeError, UnicodeError, RecursionError,
            SubtitleFormatError):
        return preserved
