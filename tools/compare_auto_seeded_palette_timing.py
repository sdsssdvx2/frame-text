"""Unsupervised OCR-seed -> source-frame boundary timing research.

A REAL automatically selected OCR observation must supply the anchor.
The independent synthetic reference is used ONLY AFTER all candidate
intervals have been generated, for scoring. This is not product code.
No source video, screenshot, user transcript, credentials or private data
is written outside an automatically removed TemporaryDirectory.
"""
from __future__ import annotations

import argparse
from contextlib import closing
from dataclasses import dataclass
import json
from pathlib import Path
import statistics
import tempfile

from hardsub_extractor.frame_source import FFmpegPTSFrameSource
from hardsub_extractor.ocr.base import comparison_key, text_similarity
from experiment_palette_ink_boundary import (
    build_palette_anchor, presence,
)
from verify_animated_background_ocr_quality import (
    check_language, encode_video, _multiline_reference,
)
from verify_multilang_video_pipeline import VIDEO_TEXT, EXPECTED_SPANS
from verify_multiline_streaming_ocr_quality import InMemorySyntheticSegmentAudit

LANGUAGES = ("ko", "en", "ja", "zh")


@dataclass(frozen=True)
class ProposedInterval:
    start: float
    end: float
    seeded_at: float
    text: str
    frames_supported: int
    blank_false_frames: int


def group_timestamps(points: list[float], frame_period: float) -> list[list[float]]:
    """Group consecutive frame supports with at most one missing sample."""
    if not 0.04 <= frame_period <= 0.50:
        raise ValueError("invalid bounded sample interval")
    groups = []
    for pts in sorted(set(points)):
        if groups and pts - groups[-1][-1] <= frame_period * 2.1:
            groups[-1].append(pts)
        else:
            groups.append([pts])
    return groups


def overlap(a: ProposedInterval, b: ProposedInterval) -> float:
    return max(0.0, min(a.end, b.end) - max(a.start, b.start))


def deduplicate(proposals: list[ProposedInterval]) -> list[ProposedInterval]:
    """Vote for an ACTUALLY observed reading within each visual interval.

    One strong-looking but incorrect early OCR crop must not outrank
    several independently sampled correct two-line readings. First group
    overlapping visual intervals *without using any reference transcript*;
    then select the repeated normalized OCR reading, with a soft medoid
    tiebreak. Never synthesize a word or combine fragments into new text.
    This is still research-only: real adjacent scene text can overlap a cue.
    """
    groups: list[list[ProposedInterval]] = []
    for candidate in sorted(proposals, key=lambda p: p.seeded_at):
        if candidate.end <= candidate.start:
            continue
        matched = None
        for group in groups:
            representative = group[0]
            if (
                overlap(candidate, representative)
                >= 0.60 * min(candidate.end - candidate.start,
                             representative.end - representative.start)
                and abs(
                    (candidate.start+candidate.end)/2
                    - (representative.start+representative.end)/2
                ) <= 0.65
            ):
                matched = group
                break
        if matched is None:
            groups.append([candidate])
        else:
            matched.append(candidate)

    selected = []
    for group in groups:
        occurrences = {}
        for item in group:
            key = comparison_key(item.text)
            if key:
                occurrences[key] = occurrences.get(key, 0) + 1
        if not occurrences:
            continue
        # Repeated independent OCR readings outrank a single false high-
        # coverage anchor; all ties still choose an existing observation.
        selected.append(max(
            group,
            key=lambda item: (
                occurrences.get(comparison_key(item.text), 0),
                sum(text_similarity(item.text, other.text)
                    for other in group),
                item.frames_supported,
                -item.blank_false_frames,
                -item.seeded_at,
            ),
        ))
    return sorted(selected, key=lambda p: (p.start, p.end))


def automatic_candidates(
    frames, blank, audit: InMemorySyntheticSegmentAudit
) -> tuple[list[ProposedInterval], dict]:
    """No ground-truth cue time or text is passed into this function."""
    if len(frames) != len(blank) or len(frames) < 10:
        raise ValueError("mismatched encoded source/control frames")
    period = statistics.median(
        b.pts - a.pts for a, b in zip(frames, frames[1:])
    )
    if not 0.04 <= period <= 0.50:
        raise ValueError("unusable source frame cadence")
    candidates = 0
    anchors = 0
    proposals = []
    for _start, _end, observations, _voted in audit.segments:
        for pts, text, metadata in observations:
            if len(text.splitlines()) != 2:
                continue
            if not isinstance(metadata, dict) or not metadata.get("detections"):
                continue
            candidates += 1
            index = min(range(len(frames)),
                        key=lambda i: abs(frames[i].pts-pts))
            if abs(frames[index].pts-pts) > period * 0.35:
                continue
            nearby = [
                frame for frame in frames
                if 0 < abs(frame.pts-pts) <= period*2.05
            ]
            anchor = build_palette_anchor(
                frames[index], nearby, metadata,
            )
            if anchor is None:
                continue
            anchors += 1
            support = presence(frames, anchor)
            blank_support = presence(blank, anchor)
            matches = [
                group for group in group_timestamps(support, period)
                if len(group) >= 3
                and group[0]-period <= pts <= group[-1]+period
            ]
            if not matches:
                continue
            winner = max(matches, key=len)
            proposals.append(ProposedInterval(
                start=round(winner[0], 3),
                end=round(winner[-1]+period, 3),
                seeded_at=round(pts, 3),
                text=text,
                frames_supported=len(winner),
                blank_false_frames=len(blank_support),
            ))
    unique = deduplicate(proposals)
    return unique, {
        "ocr_selected_two_line_samples": candidates,
        "usable_palette_anchors": anchors,
        "deduplicated_proposals": len(unique),
        "blank_control_false_frames": sum(p.blank_false_frames for p in unique),
    }


def evaluate(proposals: list[ProposedInterval], language: str) -> list[dict]:
    """Reference may influence scoring only, never seed or candidate creation."""
    out = []
    for start, end in EXPECTED_SPANS:
        matches = [
            p for p in proposals if min(p.end, end) > max(p.start, start)
        ]
        if not matches:
            out.append({"matched": False, "strict": False})
            continue
        best = max(
            matches,
            key=lambda p: (
                min(p.end, end)-max(p.start, start),
                p.frames_supported,
            ),
        )
        reference = VIDEO_TEXT[language][len(out)]
        exact_key = comparison_key(best.text) == comparison_key(reference)
        rows = len(best.text.splitlines()) == len(_multiline_reference(reference))
        start_error = round(abs(best.start-start), 3)
        end_error = round(abs(best.end-end), 3)
        out.append({
            "matched": True,
            "source_characters_match": exact_key,
            "both_rows_preserved": rows,
            "start_error_seconds": start_error,
            "end_error_seconds": end_error,
            "control_false_frames": best.blank_false_frames,
            "strict": bool(
                exact_key and rows and best.blank_false_frames == 0
                and max(start_error, end_error) <= 0.25
            ),
        })
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--manifest", default="models/manifest.json")
    args = ap.parse_args()
    results = []
    try:
        with tempfile.TemporaryDirectory(prefix="hse-auto-seed-palette-") as td:
            for language in LANGUAGES:
                root = Path(td) / language
                root.mkdir()
                audit = InMemorySyntheticSegmentAudit()
                try:
                    check_language(
                        root, language, Path(args.model_dir),
                        Path(args.manifest), style="white_two_line",
                        diagnostics=audit,
                    )
                    source = root / (language+".mp4")
                    blank_file = root / "blank.mp4"
                    encode_video(
                        blank_file, language,
                        style="white_two_line", include_subtitles=False,
                    )
                    with closing(iter(
                        FFmpegPTSFrameSource(str(source), sample_fps=8)
                    )) as it:
                        frames = list(it)
                    with closing(iter(
                        FFmpegPTSFrameSource(str(blank_file), sample_fps=8)
                    )) as it:
                        blank = list(it)
                    proposals, diagnostics = automatic_candidates(
                        frames, blank, audit,
                    )
                    checks = evaluate(proposals, language)
                    row = {
                        "language": language,
                        "executed": True,
                        "proposals_without_reference_oracle": len(proposals),
                        "diagnostics": diagnostics,
                        "reference_cues": checks,
                        "strict_reference_cues": sum(x["strict"] for x in checks),
                    }
                except Exception as exc:
                    row = {
                        "language": language, "executed": False,
                        "error_class": type(exc).__name__,
                    }
                results.append(row)
                print("HSE_AUTO_SEED_ROW "+
                      json.dumps(row, sort_keys=True), flush=True)
    except Exception as exc:
        print("HSE_AUTO_SEED_ERROR "+
              json.dumps({"error_class":type(exc).__name__}))
        return 3
    report = {
        "schema": 1,
        "experiment_complete": len(results)==4 and all(r["executed"] for r in results),
        "ground_truth_used_to_select_anchors": False,
        "ground_truth_used_only_in_evaluation": True,
        "real_anime_or_website_tested": False,
        "production_modified": False,
        "candidate_deployed": False,
        "true_complete_cues": sum(
            r.get("strict_reference_cues", 0) for r in results
        ),
        "expected_cues": 8,
    }
    print("HSE_AUTO_SEED_SUMMARY "+json.dumps(report, sort_keys=True))
    return 0 if report["experiment_complete"] else 4


if __name__ == "__main__":
    raise SystemExit(main())
