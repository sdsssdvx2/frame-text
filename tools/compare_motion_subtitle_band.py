"""OCR-free, source-frame-matched visual segmentation comparison.

Observes actual production FFmpeg decoding and detector signatures over
synthetically animated *moving* MP4. The experiment measures extra sampling,
blank control false-segment proxies, and known subtitle interval coverage.
It never invokes OCR or claims actual transcript accuracy. Moving/background
isolation is motivated by VideoSubFinder/VSE approaches but copies NO code.
No new video source, backend, package, profile or production setting is added.
"""
from __future__ import annotations

import argparse
from contextlib import closing
import json
from pathlib import Path
import sys
import tempfile
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hardsub_extractor import detector
from hardsub_extractor.frame_source import FFmpegPTSFrameSource

from verify_animated_background_ocr_quality import encode_video
from verify_multilang_video_pipeline import EXPECTED_SPANS

# These are *internal* fractional zones of the already cropped lower 40%.
# They do not discard or overwrite any source video or subtitle frame.
PROFILES = {
    "current_lower_roi": None,
    "focus_band_22_78": (0.22, 0.78),
    "focus_band_28_70": (0.28, 0.70),
}


def focused_signature(image, *, band: tuple[float, float] | None,
                      original_signature, width: int = 320):
    """Apply exactly one sub-band before the production signature operation."""
    if band is None:
        return original_signature(image, width)
    h = image.shape[0]
    y0, y1 = (int(round(h * b)) for b in band)
    if not (0 <= y0 < y1 <= h and y1 - y0 >= 16):
        raise ValueError("invalid bounded detection band")
    return original_signature(image[y0:y1], width)


def observations(media: Path, profile: str, *, expected_cues: bool) -> dict:
    band = PROFILES[profile]
    original_signature = detector.subtitle_signature
    def signature(image, width=320):
        return focused_signature(
            image, band=band, original_signature=original_signature,
            width=width,
        )

    with patch.object(detector, "subtitle_signature", new=signature):
        source = FFmpegPTSFrameSource(str(media), sample_fps=8.0)
        with closing(iter(source)) as frames:
            segments = list(detector.iter_segment_frames(frames))

    if not segments:
        raise AssertionError("source decoder yielded no frames or segments")
    pts = [f.pts for segment in segments for f in segment.samples]
    if not pts:
        raise AssertionError("no sampled frames for detector")
    detected_cues = []
    if expected_cues:
        for start, end in EXPECTED_SPANS:
            in_cue = sorted({p for p in pts if start <= p < end})
            detected_cues.append({
                "observations": len(in_cue),
                "first_offset_seconds": (
                    round(in_cue[0] - start, 3) if in_cue else None
                ),
                "tail_offset_seconds": (
                    round(end - in_cue[-1], 3) if in_cue else None
                ),
                "at_least_two_observations": len(in_cue) >= 2,
            })

    return {
        "profile": profile,
        "synthetic_subtitles_present": expected_cues,
        "segments": len(segments),
        "sample_observations": len(pts),
        "distinct_sample_times": len(set(pts)),
        "max_samples_in_one_segment": max(len(x.samples) for x in segments),
        "expected_cues": detected_cues,
    }


def compare_language(root: Path, language: str) -> dict:
    with_subtitles = root / f"{language}-sub.mp4"
    without_subtitles = root / f"{language}-blank.mp4"
    encode_video(with_subtitles, language)
    encode_video(without_subtitles, language, include_subtitles=False)
    profiles = {}
    for profile in PROFILES:
        profiles[profile] = {
            "text": observations(with_subtitles, profile, expected_cues=True),
            "blank": observations(without_subtitles, profile, expected_cues=False),
        }
    old = profiles["current_lower_roi"]
    candidates = {}
    for name, pair in profiles.items():
        if name == "current_lower_roi":
            continue
        good_coverage = all(
            x["at_least_two_observations"]
            and x["first_offset_seconds"] <= 0.65
            and x["tail_offset_seconds"] <= 0.65
            for x in pair["text"]["expected_cues"]
        )
        no_more_blank_observations = (
            pair["blank"]["sample_observations"]
            <= old["blank"]["sample_observations"]
        )
        fewer_text_observations = (
            pair["text"]["sample_observations"]
            <= old["text"]["sample_observations"] * 0.85
        )
        candidates[name] = {
            "subtitle_coverage_provisionally_preserved": good_coverage,
            "blank_motion_samples_not_increased": no_more_blank_observations,
            "at_least_15pct_fewer_text_samples": fewer_text_observations,
            "candidate_worth_real_ocr_ab": bool(
                good_coverage and no_more_blank_observations and
                fewer_text_observations
            ),
        }
    return {
        "language": language,
        "files_are_generated_encoded_mp4": True,
        "profiles": profiles,
        "candidate_filters": candidates,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--languages", default="ko,en,ja,zh")
    args = ap.parse_args()
    languages = [x.strip() for x in args.languages.split(",") if x.strip()]
    if languages != ["ko", "en", "ja", "zh"]:
        ap.error("select the full four-language synthetic benchmark")
    try:
        with tempfile.TemporaryDirectory(prefix="hse-detector-band-study-") as td:
            root = Path(td)
            result = [compare_language(root, lang) for lang in languages]
    except Exception as exc:
        print("HSE_DETECTOR_BAND_STUDY_ERROR " +
              json.dumps({"error_class": type(exc).__name__}))
        return 3
    report = {
        "schema": 1,
        "scope": "synthetic_motion_actual_mp4_production_detector_only",
        "production_changed": False,
        "real_anime_ground_truth_verified": False,
        "model_accuracy_compared": False,
        "video_count": 8,
        "profiles_per_video": len(PROFILES),
        "test_completed": len(result) == 4,
        "candidates_ready_for_optional_real_ocr_ab": [
            name for name in list(PROFILES)[1:]
            if all(
                row["candidate_filters"][name]["candidate_worth_real_ocr_ab"]
                for row in result
            )
        ],
        "languages": result,
    }
    print("HSE_DETECTOR_BAND_STUDY " +
          json.dumps(report, ensure_ascii=True, sort_keys=True))
    return 0 if report["test_completed"] else 4


if __name__ == "__main__":
    raise SystemExit(main())
