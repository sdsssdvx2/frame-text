"""Read-only, reference-free quality signals after local HSE OCR output.

Reuses the existing original-duration FFprobe and subtitle quality audit.
No additional OCR, network, automatic text removal or result file writes.
An absent warning is never a claim of transcription accuracy.
"""
from __future__ import annotations

from pathlib import Path

from .quality_audit import AuditOptions, audit_subtitles
from .quality_audit_io import _probe_local_video_duration, _read_subtitles
from .subtitle_language import normalize_language

# These high-signal issues merit optional source-frame review. Routine
# 35-second dialogue gaps, quoted foreign scripts and reading speed alone
# do not automatically mark an episode's transcription as defective.
REVIEW_CODES = frozenset({
    "near_full_runtime_single_cue_review",
    "repeated_multiline_overlay_review",
    "cue_exceeds_video_duration_review",
    "nearby_duplicate_text_review",
    "very_short_duration_review",
    "overlapping_cues_review",
    "oversized_ocr_cue_review",
})


def local_ocr_quality_signals(
    video: Path, subtitle: Path, *, language: str,
) -> dict[str, object]:
    """Return sanitized review cues and counts; never change SRT/ASS.

    No independent transcript is supplied, so accuracy_verified always
    remains False. Source duration is never guessed from extracted cues.
    Exceptions are for callers to handle as *audit unavailable*, not a
    reason to discard a successfully extracted subtitle.
    """
    language = normalize_language(language)
    cues = _read_subtitles(Path(subtitle))
    duration = _probe_local_video_duration(Path(video))
    report = audit_subtitles(cues, options=AuditOptions(
        requested_language=language,
        video_duration_seconds=duration,
        # Small read-only cue index sample; count warnings over ALL cues.
        max_review_locations=8,
    ))
    candidate = report["candidate"]
    counts = candidate["warning_counts"]
    major = {
        key: value for key, value in counts.items()
        if key in REVIEW_CODES and type(value) is int and value > 0
    }
    flagged_numbers = sorted({
        int(entry["cue"])
        for key, entries in candidate["review_locations"].items()
        if key in major
        for entry in entries
    })[:8]
    return {
        "review_cue_numbers": tuple(flagged_numbers),
        "status": "REVIEW_SUGGESTED" if major else "NO_MAJOR_HEURISTIC_WARNING",
        "cue_count": len(cues),
        "review_signal_count": sum(major.values()),
        "signal_codes": tuple(sorted(major)),
        "accuracy_verified": False,
        "independent_duration_verified": True,
        "source_and_subtitle_unchanged": True,
    }
