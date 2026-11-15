"""Conservative non-deployable persistent-text quarantine experiment.

Only inspects *already produced* per-frame OCR metadata and already-produced
SRT cues. No new inference, no use of reference transcripts, no video read and
no output file mutations. Candidate lines must keep a stable position over
most of a video AND coexist with distinct, temporally separated moving text.
A stationary single real subtitle with no other text is NOT deleted.

This module is advisory only. It never deletes/substitutes subtitles in
production; a user must explicitly open the separate manual review UI.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING
import math
import statistics
import unicodedata

if TYPE_CHECKING:
    from .models import Cue


@dataclass(frozen=True)
class LineTrace:
    pts: float
    text: str
    cx: float
    cy: float


def key(text: str) -> str:
    return unicodedata.normalize("NFC", text).strip().casefold()


def fold_whitespace(text: str) -> str:
    """Alternative research variant: fold visual word spacing, not characters.

    Never confound this with a trained language model or meaning correction.
    Unicode script, digits and punctuation remain exact.
    """
    return "".join(key(text).split())


def collect_lines(segment_audit: object) -> list[LineTrace]:
    """Read the audited HSE box geometry, not the independently known truth."""
    lines: list[LineTrace] = []
    seen = set()
    segments = getattr(segment_audit, "segments", ())
    if len(segments) > 200:
        raise ValueError("bounded per-segment research audit required")
    for segment in segments:
        if not isinstance(segment, tuple) or len(segment) != 4:
            raise ValueError("malformed bounded segment observation")
        for pts, text, metadata in segment[2]:
            if not isinstance(pts, (int, float)) or not math.isfinite(pts):
                continue
            if not isinstance(text, str) or not isinstance(metadata, dict):
                continue
            surfaces = text.splitlines()
            groups = defaultdict(list)
            for rec in metadata.get("detections", ()):
                if not isinstance(rec, dict):
                    continue
                index, cx, cy = rec.get("line"), rec.get("cx"), rec.get("cy")
                if (
                    type(index) is not int or index < 0 or index >= len(surfaces)
                    or type(cx) not in (int, float) or type(cy) not in (int, float)
                    or not math.isfinite(cx) or not math.isfinite(cy)
                    or not (0 <= cx <= 1 and 0 <= cy <= 1)
                ):
                    continue
                groups[index].append((float(cx), float(cy)))
            for index, points in groups.items():
                surface = key(surfaces[index])
                if len(surface) < 3:
                    continue
                cx = statistics.median(x for x, _ in points)
                cy = statistics.median(y for _, y in points)
                # Approximate position quantization avoids exact-PTS metadata
                # duplicates from segmentation retries creating fake support.
                item_id = (round(float(pts), 4), surface)
                if item_id in seen:
                    continue
                seen.add(item_id)
                lines.append(LineTrace(float(pts), surface, cx, cy))
            if len(lines) > 10000:
                raise ValueError("bounded OCR observation record exceeded")
    return lines


def find_candidates(
    traces: list[LineTrace], video_duration: float,
    *, flank_policy: str = "strict",
) -> set[str]:
    """Require long-lived geometry AND independently changing co-occurring text.

    Never use language-specific known watermark words, any reference subtitles,
    or a blank-frame oracle. 'Uncertain' always results in unchanged output.
    """
    if (not isinstance(video_duration, (int, float))
            or not math.isfinite(video_duration)
            or not 1.0 <= video_duration <= 3600):
        raise ValueError("invalid bounded video duration")
    if flank_policy not in {"strict", "three_windows", "baseline"}:
        raise ValueError("unsupported fixed research-only flank policy")
    if len(traces) > 10000:
        raise ValueError("excessive OCR audit observations")
    groups = defaultdict(list)
    co_visible = defaultdict(set)
    for line in traces:
        if not (0.0 <= line.pts <= video_duration + 0.5):
            continue
        groups[line.text].append(line)
        co_visible[round(line.pts, 4)].add(line.text)
    candidates = set()
    for text, items in groups.items():
        pts = sorted({round(item.pts, 4) for item in items})
        if len(pts) < 6 or pts[-1] - pts[0] < 0.70 * video_duration:
            continue
        xvalues, yvalues = [p.cx for p in items], [p.cy for p in items]
        if max(xvalues) - min(xvalues) > 0.08:
            continue
        if max(yvalues) - min(yvalues) > 0.06:
            continue
        other_early = set()
        other_late = set()
        for moment in pts:
            others = co_visible[moment] - {text}
            if moment < video_duration * 0.5:
                other_early.update(others)
            else:
                other_late.update(others)
        # Deliberately refuse to strip text in a single-caption video.
        # Independent other subtitles must be seen in both temporal halves.
        if not (other_early and other_late and len(other_early | other_late) >= 2):
            continue
        # Require multiple *independent observed frames* with this line
        # alone at both ends of the video. A repeated genuine caption line
        # limited to mid-video dialogue exchanges must not be quarantined
        # just because different second lines accompany it.
        early_only = sum(
            moment <= video_duration * 0.12
            and co_visible[moment] == {text}
            for moment in pts
        )
        late_only = sum(
            moment >= video_duration * 0.88
            and co_visible[moment] == {text}
            for moment in pts
        )
        middle_only = sum(
            video_duration * 0.40 <= moment <= video_duration * 0.60
            and co_visible[moment] == {text}
            for moment in pts
        )
        # Fixed preregistered alternatives: do not secretly choose the
        # friendliest policy from the measured reference transcript.
        if flank_policy == "strict" and (early_only < 2 or late_only < 2):
            continue
        if flank_policy == "three_windows" and (
            early_only < 1 or middle_only < 1 or late_only < 1
        ):
            continue
        candidates.add(text)
    return candidates


def anonymous_support_summary(
    traces: list[LineTrace], video_duration: float,
) -> dict:
    """Expose only aggregate track statistics, never caption text or hashes."""
    if not 1.0 <= video_duration <= 3600 or not math.isfinite(video_duration):
        raise ValueError("invalid video length for anonymized study")
    groups = defaultdict(list)
    visible = defaultdict(set)
    for record in traces:
        if 0 <= record.pts <= video_duration + 0.5:
            groups[record.text].append(record)
            visible[round(record.pts, 4)].add(record.text)
    largest = []
    for text, observations in groups.items():
        pts = sorted({round(x.pts, 4) for x in observations})
        cx, cy = [x.cx for x in observations], [x.cy for x in observations]
        single = [p for p in pts if visible[p] == {text}]
        largest.append({
            "distinct_sample_times": len(pts),
            "span_video_fraction": round((pts[-1] - pts[0]) / video_duration, 3),
            "x_position_range": round(max(cx) - min(cx), 3),
            "y_position_range": round(max(cy) - min(cy), 3),
            "independent_intro_samples": sum(p <= video_duration * 0.12 for p in single),
            "independent_middle_samples": sum(
                video_duration * 0.40 <= p <= video_duration * 0.60 for p in single
            ),
            "independent_outro_samples": sum(p >= video_duration * 0.88 for p in single),
        })
    largest.sort(key=lambda g: (g["distinct_sample_times"], g["span_video_fraction"]), reverse=True)
    return {"distinct_text_tracks": len(groups), "top_support_without_text": largest[:3]}


def quarantine(cues: list[Cue], candidates: set[str], *, normalizer=key) -> tuple[list[Cue], dict]:
    """Prototype only; preserve timestamps and never modify input objects/files."""
    out = []
    removed_lines = dropped_cues = 0
    for cue in cues:
        surfaces = cue.text.splitlines()
        retained = [line for line in surfaces if normalizer(line) not in candidates]
        removed_lines += len(surfaces) - len(retained)
        if retained:
            out.append(replace(cue, text="\n".join(retained)))
        else:
            dropped_cues += 1
    return out, {
        "candidate_stable_lines": len(candidates),
        "removed_lines": removed_lines,
        "dropped_all_candidate_cues": dropped_cues,
        "unchanged_if_no_candidate": not candidates,
        "files_changed": False,
        "release_gate_passed": False,
    }


class TemporalReviewRecorder:
    """Bounded in-RAM segment metadata for a single *explicitly selected* local OCR.

    No extra model call and no per-frame image, network transfer or disk audit.
    A missing/overflowed observation becomes INCOMPLETE, never false proof
    of absence of a persistent watermark.
    """
    def __init__(self, *, max_lines: int = 10000) -> None:
        if type(max_lines) is not int or not 1 <= max_lines <= 10000:
            raise ValueError("invalid bounded OCR audit size")
        self.max_lines = max_lines
        self.traces: list[LineTrace] = []
        self.incomplete = False

    def record_segment(self, index, segment, observations, voted) -> None:
        if self.incomplete:
            return
        try:
            from types import SimpleNamespace
            sample = SimpleNamespace(segments=[(
                float(segment.start), float(segment.end),
                tuple((o.pts, o.text, o.metadata or {}) for o in observations),
                voted.text,
            )])
            lines = collect_lines(sample)
            if len(self.traces) + len(lines) > self.max_lines:
                self.incomplete = True
                self.traces.clear()
                return
            self.traces.extend(lines)
        except (AttributeError, KeyError, TypeError, ValueError, OverflowError):
            # Diagnostics are optional: malformed metadata cannot abort a
            # previously successful OCR/SRT operation.
            self.incomplete = True
            self.traces.clear()

    def cue_numbers(self, cues: list[Cue], duration: float) -> tuple[int, ...]:
        """Return immutable 1-based review hints, NOT proposed deletions."""
        if self.incomplete or not self.traces:
            raise ValueError("temporal_review_evidence_incomplete")
        folded = [LineTrace(x.pts, fold_whitespace(x.text), x.cx, x.cy)
                  for x in self.traces]
        names = find_candidates(folded, duration,
                                flank_policy="three_windows")
        if not names:
            return ()
        # No transcript material crosses the UI boundary. The caller receives
        # only cue positions for the existing human-only frame review screen.
        return tuple(i for i, cue in enumerate(cues, 1) if any(
            fold_whitespace(line) in names
            for line in cue.text.splitlines()
        ))


class ChainedSegmentAudits:
    """Send one OCR observation to optional private diagnostics + RAM hints."""
    def __init__(self, *audits):
        self.audits = tuple(a for a in audits if a is not None)

    def record_segment(self, index, segment, observations, voted):
        for audit in self.audits:
            audit.record_segment(index, segment, observations, voted)
