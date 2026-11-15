"""Shared, offline-only SRT/ASS/VTT validation and local media metadata.

These helpers are used by both the standalone audit and frozen GUI frame
review. Never silently drop malformed subtitle blocks and never enable
network-backed FFprobe protocols.
"""
from __future__ import annotations
import json
import math
import re
from pathlib import Path
import stat
import subprocess

from .runtime_paths import resolve_binary
from .subtitle_formats import decode_subtitle_bytes, parse_subtitle_bytes

_SRT_CLOCK = r"\d{1,2}:\d{2}:\d{2}[,.]\d{1,3}"
_VTT_CLOCK = r"(?:\d{1,2}:)?\d{2}:\d{2}\.\d{1,3}"
_SRT_TIMING = re.compile(
    rf"^\s*{_SRT_CLOCK}\s*-->\s*{_SRT_CLOCK}\s*$"
)
_VTT_TIMING = re.compile(
    rf"^\s*{_VTT_CLOCK}\s*-->\s*{_VTT_CLOCK}(?:\s+.*)?$"
)


def _validate_all_cue_blocks(payload: bytes, suffix: str, parsed_count: int) -> None:
    """Reject silently dropped cue blocks without counting arrows in dialogue.

    The reusable subtitle parsers intentionally tolerate non-cue material.
    A *quality audit* is fail-closed: one malformed cue must not appear as a
    smaller but otherwise successful comparison. Do not modify global parsers.
    """
    text = decode_subtitle_bytes(payload).replace("\r\n", "\n").replace("\r", "\n")
    if suffix == ".srt":
        blocks = [block.splitlines() for block in re.split(r"\n\s*\n", text.strip()) if block.strip()]
        if len(blocks) != parsed_count:
            raise ValueError("quality_audit_srt_unparsed_blocks")
        for block in blocks:
            timing_index = 1 if block[0].strip().isdigit() else 0
            if (
                timing_index >= len(block)
                or not _SRT_TIMING.fullmatch(block[timing_index])
                or not any(line.strip() for line in block[timing_index + 1:])
            ):
                raise ValueError("quality_audit_srt_malformed_block")
    elif suffix == ".vtt":
        blocks = [block.splitlines() for block in re.split(r"\n\s*\n", text.strip()) if block.strip()]
        if not blocks or not blocks[0][0].lstrip("\ufeff").startswith("WEBVTT"):
            raise ValueError("quality_audit_vtt_header_missing")
        expected = 0
        for block in blocks[1:]:
            first = block[0].strip()
            if re.match(r"^(?:NOTE|STYLE|REGION)(?:\s|$)", first):
                continue
            timing_index = 0 if "-->" in first else 1
            if (
                timing_index >= len(block)
                or not _VTT_TIMING.fullmatch(block[timing_index].strip())
                or not any(line.strip() for line in block[timing_index + 1:])
            ):
                raise ValueError("quality_audit_vtt_malformed_block")
            expected += 1
        if expected != parsed_count:
            raise ValueError("quality_audit_vtt_unparsed_blocks")
    elif suffix in {".ass", ".ssa"}:
        expected = len(re.findall(r"(?im)^\s*Dialogue:", text))
        if expected != parsed_count:
            raise ValueError("quality_audit_ass_unparsed_blocks")


def _read_subtitles(path: Path):
    if path.suffix.lower() not in {".srt", ".ass", ".ssa", ".vtt"}:
        raise ValueError("unsupported_subtitle_extension")
    if not path.is_file():
        raise ValueError("subtitle_file_missing")
    if path.stat().st_size > 20 * 1024 * 1024:
        raise ValueError("subtitle_file_too_large")
    payload = path.read_bytes()
    _, cues = parse_subtitle_bytes(payload, source_url=path.name)
    _validate_all_cue_blocks(payload, path.suffix.lower(), len(cues))
    return cues


def _probe_local_video_duration(path: Path, *, binary_resolver=resolve_binary) -> float:
    """Read only bounded local media metadata; never fetch linked streams.

    A local playlist can reference network URLs. The FFprobe protocol
    whitelist is intentionally restricted to file and pipe, and its output
    is bounded before JSON parsing. No source path is emitted on failure.
    """
    if path.is_symlink() or not path.is_file():
        raise ValueError("quality_audit_invalid_local_video")
    item = path.stat()
    if not stat.S_ISREG(item.st_mode) or item.st_size <= 0:
        raise ValueError("quality_audit_invalid_local_video")
    if path.suffix.lower() not in {
        ".mp4", ".mkv", ".webm", ".ogv", ".avi", ".mov", ".m4v", ".ts",
    }:
        raise ValueError("quality_audit_invalid_video_extension")
    command = [
        binary_resolver("ffprobe"),
        "-v", "error",
        "-protocol_whitelist", "file,pipe",
        "-select_streams", "v:0",
        "-show_entries", "format=duration",
        "-of", "json",
        "-i", str(path),
    ]
    try:
        result = subprocess.run(
            command, stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
            timeout=35, check=False,
        )
        if result.returncode or len(result.stdout) > 32768:
            raise ValueError("quality_audit_video_probe_failed")
        info = json.loads(result.stdout)
        seconds = float(info["format"]["duration"])
    except (OSError, subprocess.TimeoutExpired, TypeError, KeyError,
            IndexError, OverflowError, UnicodeError, ValueError) as exc:
        raise ValueError("quality_audit_video_duration_unavailable") from exc
    if not math.isfinite(seconds) or not 0.1 <= seconds <= 21600.0:
        raise ValueError("quality_audit_invalid_video_duration")
    return seconds


