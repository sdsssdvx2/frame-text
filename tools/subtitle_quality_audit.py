"""Read-only, local SRT/ASS/VTT evidence audit; never prints subtitle text.

Usage:
  python tools/subtitle_quality_audit.py --candidate exported.srt
  python tools/subtitle_quality_audit.py --candidate exported.srt --reference known.srt --json local-audit.json

A reference must be the same *translation* and aligned video cut for character
error to represent OCR error. Otherwise text comparisons are only diagnostics.
"""
from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
import stat
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from hardsub_extractor.quality_audit import AuditOptions, audit_subtitles
from hardsub_extractor.subtitle_language import normalize_language
from hardsub_extractor.runtime_paths import resolve_binary
from hardsub_extractor.subtitle_formats import (
    decode_subtitle_bytes,
    parse_subtitle_bytes,
)


from hardsub_extractor.quality_audit_io import (
    _validate_all_cue_blocks,
    _read_subtitles,
    _probe_local_video_duration as _shared_probe_local_video_duration,
)


def _probe_local_video_duration(path: Path) -> float:
    """Preserve the public CLI's binary override for existing diagnostics."""
    return _shared_probe_local_video_duration(path, binary_resolver=resolve_binary)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Private subtitle timeline and reference comparison; no OCR or network")
    ap.add_argument("--candidate", type=Path, required=True)
    ap.add_argument("--reference", type=Path)
    ap.add_argument("--reference-shift-seconds", type=float, default=0.0)
    ap.add_argument("--tolerance-seconds", type=float, default=1.5)
    ap.add_argument(
        "--language", default="ko",
        help="Expected subtitle language (ko/en/ja/zh/zh-Hans/zh-Hant); script warnings only",
    )
    duration_input = ap.add_mutually_exclusive_group()
    duration_input.add_argument(
        "--video-duration-seconds", type=float,
        help="Independently measured video duration; never inferred from SRT",
    )
    duration_input.add_argument(
        "--video", type=Path,
        help="Optional actual local source video; ffprobe reads duration only, without network access",
    )
    ap.add_argument("--json", type=Path, help="Explicitly opt in to local JSON output")
    args = ap.parse_args(argv)
    try:
        if args.json is not None:
            protected = {args.candidate.resolve()}
            if args.reference is not None:
                protected.add(args.reference.resolve())
            if args.json.resolve() in protected:
                raise ValueError("quality_audit_output_overwrites_input")
        candidate = _read_subtitles(args.candidate)
        reference = _read_subtitles(args.reference) if args.reference else None
        duration = (
            _probe_local_video_duration(args.video)
            if args.video is not None else args.video_duration_seconds
        )
        options = AuditOptions(
            alignment_tolerance_seconds=args.tolerance_seconds,
            reference_shift_seconds=args.reference_shift_seconds,
            requested_language=normalize_language(args.language),
            video_duration_seconds=duration,
        )
        report = audit_subtitles(candidate, reference=reference, options=options)
        report_json = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
        if args.json:
            args.json.parent.mkdir(parents=True, exist_ok=True)
            # Exclusive creation fails on existing files, symlinks, and
            # hardlinks. An audit must never silently truncate prior evidence
            # or a user's existing input file, even via a second pathname.
            with args.json.open("x", encoding="utf-8") as stream:
                stream.write(report_json)
        # Console output is bounded, text-free, and has no paths or raw error
        # details. In particular, avoid echoing source filenames on failure.
        print(report_json.encode("ascii", "backslashreplace").decode("ascii"), end="")
        return 0
    except (OSError, UnicodeError, ValueError) as exc:
        # External subtitle text/paths are never printed, even on parse errors.
        print(f"HSE_QUALITY_AUDIT_FAILED={type(exc).__name__}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
