"""Optional official native VideoSubFinder 6.10.5 timing bridge.

The core DOES NOT download, install, bundle or run GPL software by default.
The caller must explicitly select an already SHA-verified official binary.
All detection runs in an ephemeral scratch directory and supplies only
independently observed timing intervals to HSE's own bounded PP-OCR reader.
The executable's output text is not used as a subtitle transcript.

The upstream CLI is GPL-2.0, not part of HSE's MIT Python code or release
package. Bundling it in a future release needs independent redistribution,
third-party dependency, matching-source and NOTICE review.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
from pathlib import Path
import re
import subprocess
import tempfile
import time
from typing import Callable

from .external_subprocess import external_popen

_NATIVE_VERSION = b"6.10.5"
_MAX_TIMING_BYTES = 512 * 1024
_MAX_TIMINGS = 3000
_MAX_DURATION = 21600.0
_MAX_MEDIA_BYTES = 100 * 1024 ** 3
_RE_CLOCK = re.compile(
    r"(\d{1,3}):([0-5]\d):([0-5]\d)[,.](\d{3})\Z"
)
_RE_TIME_LINE = re.compile(
    r"\s*(\S+)\s+-->\s+(\S+)\s*\Z"
)
_MEDIA_SUFFIXES = frozenset({
    ".mp4", ".mkv", ".webm", ".mov", ".m4v", ".ts", ".avi",
})


@dataclass(frozen=True, slots=True)
class NativeTiming:
    start: float
    end: float


@dataclass(frozen=True, slots=True)
class NativeTimingResult:
    intervals: tuple[NativeTiming, ...]
    detection_seconds: float


class NativeTimingError(RuntimeError):
    """Untrusted executable, timing output or bounded native process failure."""


def _clock(value: str) -> float:
    match = _RE_CLOCK.fullmatch(value)
    if match is None:
        raise NativeTimingError("invalid native timing clock")
    hh, mm, ss, ms = map(int, match.groups())
    result = hh * 3600 + mm * 60 + ss + ms / 1000
    if not 0 <= result <= _MAX_DURATION:
        raise NativeTimingError("native timing exceeds bounded source duration")
    return result


def parse_native_timing_srt(data: bytes) -> tuple[NativeTiming, ...]:
    """Fail closed on malformed, overlapping, excessive or unordered timings.

    Only the timestamps are returned; native body text is never retained or
    promoted to model-produced subtitles.
    """
    if len(data) > _MAX_TIMING_BYTES:
        raise NativeTimingError("native timing report too large")
    try:
        text = data.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise NativeTimingError("native timing report is not UTF-8") from exc
    text = text.replace("\r\n", "\n").replace("\r", "\n").strip()
    if not text:
        return ()
    parts = re.split(r"\n[ \t]*\n", text)
    if len(parts) > _MAX_TIMINGS:
        raise NativeTimingError("native interval count exceeds bounded limit")
    spans: list[NativeTiming] = []
    for index, part in enumerate(parts, 1):
        lines = part.split("\n")
        if len(lines) < 3 or lines[0].strip() != str(index):
            raise NativeTimingError("native timing index invalid")
        match = _RE_TIME_LINE.fullmatch(lines[1])
        if match is None or not "\n".join(lines[2:]).strip():
            raise NativeTimingError("native interval missing timing or placeholder")
        start, end = _clock(match.group(1)), _clock(match.group(2))
        if not (0 <= start < end <= _MAX_DURATION):
            raise NativeTimingError("native interval has invalid duration")
        if spans and start < spans[-1].end:
            raise NativeTimingError("overlapping or disordered native intervals")
        spans.append(NativeTiming(start, end))
    return tuple(spans)


def _vsf_roi_edges(roi) -> tuple[str, str, str, str]:
    """Translate HSE top-left ROI into VSF bottom-origin region flags.

    Preserve the previously accepted broad native search for default HSE
    regions. A consciously different ROI must reach *both* interval detection
    and FFmpeg OCR, otherwise selecting a high/side caption silently misses it.
    Coordinates are finite normalized fractions, not external shell text.
    """
    from .frame_source import NormalizedROI

    if roi is None:
        return ("0.44", "0", "0", "1")
    if not isinstance(roi, NormalizedROI):
        raise ValueError("native VSF region must be a normalized HSE ROI")
    values = (roi.x, roi.y, roi.w, roi.h)
    if any(type(v) not in (int, float) or not math.isfinite(v) for v in values):
        raise ValueError("native VSF region has invalid coordinate type")
    x, y, w, h = (float(v) for v in values)
    if (x < 0 or y < 0 or w <= 0 or h <= 0
            or x + w > 1 or y + h > 1):
        raise ValueError("native VSF region exceeds normalized image")

    # Old CLI passed the standard default ROI for OCR but the wider
    # native crop was intentional; keep default detected coverage unchanged.
    if values == (0.04, 0.58, 0.92, 0.40):
        return ("0.44", "0", "0", "1")

    top_from_bottom = 1 - y
    bottom_from_bottom = 1 - (y + h)
    result = tuple(
        format(v, ".6f")
        for v in (top_from_bottom, bottom_from_bottom, x, x + w)
    )
    if (float(result[0]) <= float(result[1])
            or float(result[2]) >= float(result[3])):
        raise ValueError("native VSF selected region is subpixel after rounding")
    return result


def _hash_file(path: Path) -> str:
    hashvalue = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            hashvalue.update(chunk)
    return hashvalue.hexdigest()


def _spawn_wait(
    argv: list[str], directory: Path,
    timeout_seconds: float,
    check_cancelled: Callable[[], None] | None,
) -> None:
    child = external_popen(
        argv, cwd=str(directory), stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    try:
        deadline = time.monotonic() + timeout_seconds
        while child.poll() is None:
            if check_cancelled is not None:
                check_cancelled()
            if time.monotonic() >= deadline:
                raise NativeTimingError("native detector exceeded runtime limit")
            time.sleep(0.2)
        if check_cancelled is not None:
            check_cancelled()
        # Reap the completed process even if poll() already saw exit 0.
        child.wait(timeout=15)
        if child.returncode != 0:
            raise NativeTimingError("native detector terminated without success")
    except BaseException:
        if child.poll() is None:
            child.kill()
        child.wait(timeout=15)
        raise


def detect_native_vsf_timings(
    video: str | Path,
    executable: str | Path,
    *,
    pinned_executable_sha256: str,
    max_seconds: float = 900.0,
    roi=None,
    check_cancelled: Callable[[], None] | None = None,
) -> NativeTimingResult:
    """Run exact-pinned native VSF once, then delete only its own scratch.

    The executable path and trusted SHA pin are mandatory, *never* inferred
    from an untrusted runtime output or caller-chosen text. The official
    release archive must have been verified separately before this call.
    A no-subtitle video may legitimately produce an empty interval list.
    """
    if (
        type(max_seconds) not in (int, float)
        or not math.isfinite(max_seconds)
        or not 1 <= max_seconds <= 3600
    ):
        raise ValueError("invalid native timing deadline")
    if (
        not isinstance(pinned_executable_sha256, str)
        or re.fullmatch(r"[0-9a-f]{64}", pinned_executable_sha256) is None
    ):
        raise ValueError("trusted native executable SHA-256 pin required")
    te, be, le, right_edge = _vsf_roi_edges(roi)
    src = Path(video)
    binary = Path(executable)
    if (
        src.is_symlink() or not src.is_file()
        or src.suffix.lower() not in _MEDIA_SUFFIXES
        or not 0 < src.stat().st_size <= _MAX_MEDIA_BYTES
        or binary.is_symlink() or not binary.is_file()
        or binary.suffix.lower() != ".exe"
    ):
        raise NativeTimingError("native detector requires local regular video and executable")
    if check_cancelled:
        check_cancelled()
    if _hash_file(binary) != pinned_executable_sha256:
        raise NativeTimingError("native detector executable hash does not match official pin")
    source = src.resolve(strict=True)
    native = binary.resolve(strict=True)
    # The executable must have passed the SHA pin before any CreateProcess.
    begin = time.monotonic()
    with tempfile.TemporaryDirectory(prefix="hse-vsf-timings-") as root:
        scratch = Path(root)
        report = scratch / "timings.srt"
        # -dsi forbids unnecessary RGB frame files. The OCR decoder will
        # sample the same original video in a forward pass independently.
        argv = [
            str(native), "-r", "-dsi", "-i", str(source),
            "-o", str(scratch), "-ces", str(report),
            "-te", te, "-be", be, "-le", le, "-re", right_edge,
            "-nthr", "4",
        ]
        _spawn_wait(argv, native.parent, float(max_seconds), check_cancelled)
        if report.is_symlink() or not report.is_file():
            raise NativeTimingError("native detector did not publish timing SRT")
        if report.stat().st_size > _MAX_TIMING_BYTES:
            raise NativeTimingError("native detector timing output exceeds budget")
        spans = parse_native_timing_srt(report.read_bytes())
    return NativeTimingResult(spans, round(time.monotonic() - begin, 3))


@dataclass(frozen=True, slots=True)
class NativeVsfOCRExtraction:
    timing: NativeTimingResult
    report: object  # TimedOCRReport; lazy import avoids loading FFmpeg/OCR at module import


def extract_with_native_vsf(
    video: str | Path,
    out_base: str | Path,
    *,
    executable: str | Path,
    pinned_executable_sha256: str,
    ocr_factory: Callable[[], object],
    sample_fps: float = 8.0,
    roi=None,
    check_cancelled: Callable[[], None] | None = None,
    split_stable_text: bool = False,
) -> NativeVsfOCRExtraction:
    """Explicit opt-in CLI/backend route; original subtitle-first stays default.

    Never initializes expensive recognition if no trustworthy native timing
    interval exists. Never calls a model, downloader or external native exe
    for an already occupied result path. No reference/source subtitle text
    enters either recognition or interval choice.
    """
    from .timed_ocr import extract_timed_ocr
    base = Path(out_base).expanduser()
    if any(
        path.is_symlink() or path.exists()
        for path in (base.with_suffix(".srt"), base.with_suffix(".ass"))
    ):
        raise FileExistsError("original subtitle result must not be overwritten")
    timing = detect_native_vsf_timings(
        video, executable,
        pinned_executable_sha256=pinned_executable_sha256,
        roi=roi,
        check_cancelled=check_cancelled,
    )
    if not timing.intervals:
        raise NativeTimingError("native detector found no hardcoded subtitle intervals")
    if check_cancelled is not None:
        check_cancelled()
    report = extract_timed_ocr(
        str(video), timing.intervals, ocr_factory(), base,
        sample_fps=sample_fps, roi=roi, check_cancelled=check_cancelled,
        split_stable_text=split_stable_text,
    )
    return NativeVsfOCRExtraction(timing=timing, report=report)
