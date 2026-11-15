"""Original-subtitle-first batch extraction for 1–50 explicitly selected local videos.

No directory crawl, background scheduling, URL, browser, network request, user
media write, transcription upload, overwrite or automatic deletion. Sources
are validated as a complete bounded set BEFORE the first output or OCR call.
The chosen output directory may exist, but each result uses an independently
reserved deterministic stem. On interruption or retry a source-backed, structurally verified original
text result can be skipped without model calls. Incomplete or unmanifested
OCR results are preserved, never guessed complete from mere file existence.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import partial
import hashlib
import os
from pathlib import Path
import re
import unicodedata
from typing import Callable, Iterable

from .extraction_progress import ExtractionMonitor
from .local_sidecar_subtitles import acquire_local_sidecar_text
from .local_output_audit import audit_local_output
from .local_soft_subtitles import acquire_local_embedded_text
from .local_image_subtitles import acquire_local_embedded_image
from .model_assets import normalize_ocr_language
from .pipeline import extract as extract_hard_subtitles

_MAX_LIST_BYTES = 128 * 1024
_VIDEO_EXTENSIONS = frozenset({
    ".mp4", ".mkv", ".webm", ".ogv", ".mov", ".avi", ".m4v", ".ts",
})
# Windows disallows these characters in filenames. Retain Korean, Japanese
# and Chinese characters in results rather than replacing non-ASCII names.
_INVALID_FILENAME = re.compile(r'[<>:"/\\|?*\x00-\x1f]')
_WHITESPACE = re.compile(r"\s+")


@dataclass(frozen=True, slots=True)
class LocalBatchItem:
    source: Path
    out_base: Path


@dataclass(frozen=True, slots=True)
class LocalBatchResult:
    index: int
    mode: str  # local-embedded-text/local-sidecar-text/hard_ocr/existing_verified/existing_preserved/failed
    cues: int
    # Sensitive video names and paths never have to be logged.
    failure_class: str = ""
    # Number of source intervals with no recognized text in optional native
    # timing mode. Zero is not a claim of correct transcription.
    missed_events: int = 0
    missed_windows: tuple[tuple[float, float], ...] = ()
    split_cue_numbers: tuple[int, ...] = ()
    short_frame_rescue_attempted: int = 0
    short_frame_rescue_recognized: int = 0
    short_frame_rescue_deferred: int = 0
    short_frame_rescue_errors: int = 0
    optical_timing_candidates: int = 0
    optical_timing_refined: int = 0
    # >1 means bounded OCR pixel evidence was thinned over a long video.
    # The transcript is not claimed verified even when this is 1.
    optical_witness_stride: int = 1


def _safe_source(value: str | Path) -> Path:
    source = Path(value).expanduser()
    if (
        source.is_symlink() or not source.is_file()
        or source.suffix.lower() not in _VIDEO_EXTENSIONS
    ):
        raise ValueError("invalid explicit local video source")
    raw_stat = source.stat()
    if not raw_stat.st_size:
        raise ValueError("empty local video")
    return source.resolve(strict=True)


def _output_basename(source: Path) -> str:
    clean = unicodedata.normalize("NFC", source.stem)
    clean = _WHITESPACE.sub(" ", _INVALID_FILENAME.sub("_", clean))
    # Path.with_suffix(".srt") replaces everything after the final dot.
    # Typical episode names like show.S01E02.1080p MUST NOT drop both
    # their season marker and our path-identity digest during publishing.
    clean = clean.replace(".", "_")
    clean = clean.rstrip(" ")[:64].rstrip(" ") or "video"
    # Include every directory, not just the filename. Different episodes
    # from distinct folders may legitimately share names. The path digest is
    # not claimed as a cryptographic fingerprint of the video's CONTENT.
    identity = os.fsencode(str(source))
    discriminator = hashlib.sha256(identity).hexdigest()[:16]
    return f"{clean}-{discriminator}"


def load_local_batch(
    named: Iterable[str | Path], file_list: str | Path | None, out_dir: str | Path,
) -> list[LocalBatchItem]:
    """Validate entire list, dedupe paths and prevent output-base collisions."""
    submitted: list[Path] = [Path(p).expanduser() for p in named if str(p).strip()]
    if file_list:
        list_path = Path(file_list).expanduser()
        if list_path.is_symlink() or not list_path.is_file():
            raise ValueError("local video list must be a regular file")
        if not (0 < list_path.stat().st_size <= _MAX_LIST_BYTES):
            raise ValueError("invalid local video list size")
        with list_path.open("rb") as stream:
            raw = stream.read(_MAX_LIST_BYTES + 1)
        if len(raw) > _MAX_LIST_BYTES:
            raise ValueError("oversized local video list")
        try:
            entries = raw.decode("utf-8-sig").splitlines()
        except UnicodeDecodeError as exc:
            raise ValueError("video list must be UTF-8") from exc
        if len(entries) > 500:
            raise ValueError("local video list has too many lines")
        for line in entries:
            value = line.strip()
            if not value or value.startswith("#"):
                continue
            item = Path(value).expanduser()
            submitted.append(
                item if item.is_absolute() else list_path.parent / item
            )
    if not 1 <= len(submitted) <= 50:
        raise ValueError("local batch requires 1 to 50 explicit video paths")

    directory = Path(out_dir).expanduser()
    if directory.exists() and (
        not directory.is_dir() or directory.is_symlink()
    ):
        raise ValueError("result destination must be a local directory")
    seen_sources: set[str] = set()
    seen_outputs: set[str] = set()
    items: list[LocalBatchItem] = []
    for path in submitted:
        source = _safe_source(path)
        key = os.path.normcase(str(source))
        if key in seen_sources:
            continue
        seen_sources.add(key)
        out = directory / _output_basename(source)
        folded = out.name.casefold()
        if folded in seen_outputs:
            raise ValueError("ambiguous output filenames in local batch")
        seen_outputs.add(folded)
        items.append(LocalBatchItem(source, out))
    if not items:
        raise ValueError("local batch contains no unique videos")
    return items


def run_local_batch(
    items: list[LocalBatchItem],
    *,
    language: str = "ko",
    ocr_factory: Callable[[], object],
    native_timing_factory: Callable[[], tuple[Path, str]] | None = None,
    native_split_opt_in: bool = False,
    focus_bottom_two_lines: bool = False,
    caption_islands: bool = False,
    light_caption_temporal: bool = False,
    pixel_timing_only: bool = False,
    region_mode: str = "bottom40",
    image_converter_factory: Callable[[], Path] | None = None,
    check_cancelled: Callable[[], None] | None = None,
    monitor: ExtractionMonitor | None = None,
    on_result: Callable[[LocalBatchResult], None] | None = None,
) -> list[LocalBatchResult]:
    """Process validated video items; only load an OCR model when necessary."""
    chosen = normalize_ocr_language(language)
    from .tab_capture import roi_for_region
    crop = roi_for_region(region_mode)
    if region_mode != "bottom40" and native_timing_factory is not None:
        raise ValueError("ROI incompatible with independent native detector")
    if not 1 <= len(items) <= 50:
        raise ValueError("invalid local batch item count")
    if len({x.out_base for x in items}) != len(items):
        raise ValueError("duplicate local batch output target")
    if len({x.source for x in items}) != len(items):
        raise ValueError("duplicate local batch source")
    if native_timing_factory is not None and not callable(native_timing_factory):
        raise ValueError("optional native detector must be a callable factory")
    if type(native_split_opt_in) is not bool or (
        native_split_opt_in and native_timing_factory is None
    ):
        raise ValueError("stable subtitle split requires native detector opt-in")
    if (type(focus_bottom_two_lines) is not bool
            or focus_bottom_two_lines and native_timing_factory is not None):
        raise ValueError("bottom-two-line focus requires standard local visual OCR")
    if type(light_caption_temporal) is not bool or (
        light_caption_temporal and (
            native_timing_factory is not None or focus_bottom_two_lines
        )
    ):
        raise ValueError("bright-caption optical timing requires standard visual OCR")
    if type(caption_islands) is not bool or (
        caption_islands and (
            native_timing_factory is not None or focus_bottom_two_lines
            or light_caption_temporal
        )
    ):
        raise ValueError("caption-island filter requires standard visual OCR only")
    if type(pixel_timing_only) is not bool or (
        pixel_timing_only and (
            light_caption_temporal or caption_islands or focus_bottom_two_lines
            or native_timing_factory is not None
        )
    ):
        raise ValueError("pixel-only optical timing requires default visual OCR")
    if image_converter_factory is not None and not callable(image_converter_factory):
        raise ValueError("optional image converter must be a callable factory")
    if monitor is not None and not monitor.exclusive_outputs:
        raise ValueError("local batch requires fail-closed exclusive subtitle writers")
    for item in items:
        if (
            item.source.is_symlink() or not item.source.is_file()
            or item.source.suffix.lower() not in _VIDEO_EXTENSIONS
        ):
            raise ValueError("video disappeared before extraction")
    # Do not create an empty output directory when every item is invalid.
    ocr = None
    native_tool: tuple[Path, str] | None = None
    native_tool_failed = False

    def acquire_native_tool() -> tuple[Path, str]:
        nonlocal native_tool, native_tool_failed
        if native_timing_factory is None:
            raise RuntimeError("native detector was not enabled")
        if native_tool is None:
            # First use happens only AFTER original subtitle inventory for
            # the current episode. Never fetch once per video or on a clean
            # 50-episode original-text collection.
            if native_tool_failed:
                raise RuntimeError("previous optional native setup failed")
            try:
                pair = native_timing_factory()
                if (
                    not isinstance(pair, tuple) or len(pair) != 2
                    or not isinstance(pair[0], (Path, str))
                    or not isinstance(pair[1], str)
                ):
                    raise ValueError("optional native detector identity invalid")
                native_tool = (Path(pair[0]), pair[1])
            except (OSError, RuntimeError, ValueError):
                # Missing/offline/corrupted optional native tools fail closed.
                # Retain original subtitle priority for other episodes, but
                # never retry an unsuccessful download for each next episode.
                native_tool_failed = True
                raise
        return native_tool

    def acquire_ocr():
        nonlocal ocr
        if ocr is None:
            ocr = ocr_factory()
        return ocr

    results: list[LocalBatchResult] = []
    for index, item in enumerate(items, 1):
        if check_cancelled:
            check_cancelled()
        existing = audit_local_output(
            item.source, item.out_base, language=chosen,
        )
        if existing.state == "verified":
            record = LocalBatchResult(
                index, "existing_verified", existing.cue_count,
            )
        elif existing.state == "preserved":
            record = LocalBatchResult(index, "existing_preserved", 0)
        else:
            try:
                # A detected text track is sufficient; avoid fetching model
                # assets or consuming any OCR inference on that episode.
                soft = acquire_local_embedded_text(
                    item.source, item.out_base, language=chosen,
                    check_cancelled=check_cancelled,
                )
                if soft is None:
                    soft = acquire_local_sidecar_text(
                        item.source, item.out_base, language=chosen,
                        check_cancelled=check_cancelled,
                    )
                if soft is None:
                    # Optional portable free seconv conversion of an embedded
                    # PGS/VobSub image track. Distinct from visible-frame OCR.
                    soft = acquire_local_embedded_image(
                        item.source, item.out_base, language=chosen,
                        check_cancelled=check_cancelled,
                        ocr_factory=acquire_ocr,
                        **({"converter_factory": image_converter_factory}
                           if image_converter_factory is not None else {}),
                    )
                if soft is not None:
                    source_missed = getattr(soft, "missed_events", 0)
                    if (
                        type(source_missed) is not int or source_missed < 0
                    ):
                        raise ValueError("invalid source bitmap OCR event count")
                    soft_mode = (
                        "local-embedded-image-ocr-partial"
                        if soft.mode == "local-embedded-image-ocr"
                        and source_missed > 0
                        else soft.mode
                    )
                    record = LocalBatchResult(
                        index, soft_mode, soft.cue_count,
                        missed_events=source_missed,
                    )
                else:
                    if check_cancelled:
                        check_cancelled()
                    if native_timing_factory is not None:
                        # No silent fallthrough to a different OCR strategy
                        # if the user explicitly selected the native detector.
                        # This would hide incomplete subtitles on long batches.
                        from .native_vsf_timing import extract_with_native_vsf
                        exe, digest = acquire_native_tool()
                        native = extract_with_native_vsf(
                            item.source, item.out_base,
                            executable=exe, pinned_executable_sha256=digest,
                            ocr_factory=acquire_ocr,
                            check_cancelled=check_cancelled,
                            **({"split_stable_text": True}
                               if native_split_opt_in else {}),
                        )
                        report = native.report
                        # Recognized text is not automatically a complete
                        # transcript. Preserve the files, but never count
                        # source intervals skipped by OCR as a full episode.
                        mode = (
                            "hard_ocr_native_timing_partial"
                            if report.missed_intervals else
                            "hard_ocr_native_timing"
                        )
                        record = LocalBatchResult(
                            index, mode,
                            getattr(report, "exported_cues", report.recognized_intervals),
                            missed_events=report.missed_intervals,
                            missed_windows=tuple(
                                getattr(report, "missed_windows", ())
                            )[:8],
                            split_cue_numbers=tuple(
                                getattr(report, "split_cue_numbers", ())
                            )[:8],
                            short_frame_rescue_attempted=getattr(
                                report, "short_frame_rescue_attempted", 0
                            ),
                            short_frame_rescue_recognized=getattr(
                                report, "short_frame_rescue_recognized", 0
                            ),
                            short_frame_rescue_deferred=getattr(
                                report, "short_frame_rescue_deferred", 0
                            ),
                            short_frame_rescue_errors=getattr(
                                report, "short_frame_rescue_errors", 0
                            ),
                        )
                    else:
                        ocr = acquire_ocr()
                        selected_ocr = ocr
                        light = None
                        if caption_islands:
                            from .ocr.caption_islands import CaptionIslandEngine
                            # A separate wrapper per episode, shared lazy base
                            # recognizer across all 1–50 episodes.
                            selected_ocr = CaptionIslandEngine(ocr)
                        run_monitor = monitor or ExtractionMonitor(
                            exclusive_outputs=True,
                        )
                        if light_caption_temporal or pixel_timing_only:
                            from .ocr.light_temporal import LightTemporalWitnessEngine
                            # Reuse the OCR model but never share source pixel
                            # evidence between episodes. Pixel-only preserves
                            # default recognition without a brightness filter.
                            if light_caption_temporal:
                                from .ocr.light_text import LightCaptionEngine
                                selected_ocr = LightCaptionEngine(ocr)
                            light = LightTemporalWitnessEngine(selected_ocr)
                            selected_ocr = light
                        # Original frame sampling and source-first path stay intact.
                        report = extract_hard_subtitles(
                            str(item.source), selected_ocr, item.out_base,
                            streaming=True, monitor=run_monitor,
                            **({"roi": crop} if region_mode != "bottom40" else {}),
                            **({"focus_bottom_two_lines": True}
                               if focus_bottom_two_lines else {}),
                            **({"timing_refiner": partial(
                                light.refine_timings,
                                check_cancelled=run_monitor.check_cancelled,
                                cancel_requested=run_monitor.is_cancelled,
                            )} if light is not None else {}),
                        )
                        record = LocalBatchResult(
                            index, "hard_ocr", report.cue_count,
                            optical_timing_candidates=(
                                light.candidate_cues if light is not None else 0
                            ),
                            optical_timing_refined=(
                                light.refined_cues if light is not None else 0
                            ),
                            optical_witness_stride=(
                                light.witness_sample_stride
                                if light is not None else 1
                            ),
                        )
            except FileExistsError:
                # A concurrent writer's result is never removed or replaced.
                record = LocalBatchResult(index, "existing_preserved", 0)
            except (ValueError, OSError, RuntimeError) as exc:
                if check_cancelled:
                    check_cancelled()
                # Never log source video names, user captions, paths,
                # exception messages, or media credentials.
                record = LocalBatchResult(index, "failed", 0, type(exc).__name__)
        results.append(record)
        if on_result:
            on_result(record)
    return results
