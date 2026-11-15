"""Real encoded-MP4 watermark persistence stress study; research only.

Only generated synthetic background/video/dialogue: no websites, real anime,
user material or external engines. The source-side truth enters scoring ONLY
after production FFmpeg -> segment detector -> RapidOCR -> SRT/ASS has run.
This records, but does not claim to fix, watermark-induced false positives.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from compare_external_engine_exports import Cue, evaluate, parse_strict_srt
from hardsub_extractor.ocr.adaptive import AdaptiveOCREngine
from hardsub_extractor.ocr.rapidocr_adapter import RapidOCRAdapter
from hardsub_extractor.pipeline import extract
from hardsub_extractor.runtime_paths import resolve_binary
from hardsub_extractor.subtitle_formats import parse_subtitle_bytes
from verify_animated_background_ocr_quality import (
    WIDTH, HEIGHT, FPS, MAX_VIDEO_BYTES, _multiline_reference,
    animated_frame, subtitle_overlay,
)
from verify_multilang_ocr_synthetic import choose_font
from verify_multilang_video_pipeline import VIDEO_TEXT, SEGMENTS, EXPECTED_SPANS
from verify_multiline_streaming_ocr_quality import InMemorySyntheticSegmentAudit
from study_temporal_overlay_quarantine import (
    LineTrace, anonymous_support_summary, collect_lines,
    find_candidates, fold_whitespace, quarantine,
)

LANGUAGES = ("ko", "zh")
CASES = ("clean", "persistent_watermark", "watermark_only")
WATERMARKS = {"ko": "방송 안내", "zh": "视频标识"}
FRAME_COUNT = SEGMENTS[-1][1]


def reference_cues(language: str) -> list[Cue]:
    if language not in LANGUAGES:
        raise ValueError("unsupported research language")
    return [
        Cue(start, end, "\n".join(_multiline_reference(text)))
        for (start, end), text in zip(EXPECTED_SPANS, VIDEO_TEXT[language])
    ]


def watermark_layer(language: str) -> np.ndarray:
    if language not in WATERMARKS:
        raise ValueError("unsupported watermark language")
    layer = Image.new("RGBA", (WIDTH, HEIGHT), (0, 0, 0, 0))
    draw = ImageDraw.Draw(layer)
    font = ImageFont.truetype(str(choose_font(language)), size=38)
    text = WATERMARKS[language]
    bounds = draw.textbbox((0, 0), text, font=font, stroke_width=2)
    draw.text(
        (320 - bounds[0], 622 - bounds[1]), text, font=font,
        fill=(250, 248, 245, 230), stroke_width=2,
        stroke_fill=(3, 3, 3, 245),
    )
    return cv2.cvtColor(np.asarray(layer), cv2.COLOR_RGBA2BGRA)


def blend(frame: np.ndarray, overlay: np.ndarray) -> np.ndarray:
    if frame.shape != (HEIGHT, WIDTH, 3) or overlay.shape != (HEIGHT, WIDTH, 4):
        raise ValueError("invalid comparison frame geometry")
    a = overlay[:, :, 3:4].astype(np.uint16)
    return (
        (frame.astype(np.uint16) * (255 - a)
         + overlay[:, :, :3].astype(np.uint16) * a + 127) // 255
    ).astype(np.uint8)


def gradient() -> np.ndarray:
    x = np.arange(WIDTH, dtype=np.uint16)[None, :]
    y = np.arange(HEIGHT, dtype=np.uint16)[:, None]
    base = np.empty((HEIGHT, WIDTH, 3), dtype=np.uint8)
    base[:, :, 0] = ((x // 4 + y // 6) % 93 + 42).astype(np.uint8)
    base[:, :, 1] = ((x // 5 + y // 3) % 77 + 53).astype(np.uint8)
    base[:, :, 2] = ((x // 7 + y // 4) % 95 + 64).astype(np.uint8)
    return base


def decoded_frame_count(video: Path) -> int:
    result = subprocess.run([
        resolve_binary("ffprobe"), "-v", "error",
        "-select_streams", "v:0", "-count_frames",
        "-show_entries", "stream=nb_read_frames",
        "-of", "json", str(video),
    ], stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, timeout=60)
    if result.returncode or len(result.stdout) > 32768:
        raise RuntimeError("independent decoder frame count unavailable")
    streams = json.loads(result.stdout)["streams"]
    if len(streams) != 1:
        raise RuntimeError("invalid fixture stream count")
    count = int(streams[0]["nb_read_frames"])
    if not 1 <= count <= FRAME_COUNT:
        raise RuntimeError("invalid fixture decoded frame count")
    return count


def make_video(dest: Path, language: str, profile: str) -> dict:
    if language not in LANGUAGES or profile not in CASES:
        raise ValueError("unsupported paired video profile")
    if dest.exists() or dest.is_symlink():
        raise RuntimeError("will not overwrite any video")
    captions = profile != "watermark_only"
    logo = None if profile == "clean" else watermark_layer(language)
    overlays = tuple(
        subtitle_overlay(text, choose_font(language), style="white_two_line")
        for text in VIDEO_TEXT[language]
    ) if captions else ()
    base = gradient()
    cmd = [
        resolve_binary("ffmpeg"), "-nostdin", "-hide_banner", "-loglevel",
        "error", "-f", "rawvideo", "-pixel_format", "bgr24",
        "-video_size", f"{WIDTH}x{HEIGHT}", "-framerate", str(FPS),
        "-i", "pipe:0", "-an", "-c:v", "mpeg4", "-q:v", "4",
        "-pix_fmt", "yuv420p", "-movflags", "+faststart", "-n", str(dest),
    ]
    process = subprocess.Popen(
        cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    written = 0
    try:
        if process.stdin is None:
            raise RuntimeError("encoded source pipe unavailable")
        for start, end, cue in SEGMENTS:
            if start != written:
                raise RuntimeError("reference intervals not contiguous")
            for index in range(start, end):
                image = animated_frame(
                    index, base,
                    overlays[cue] if captions and cue is not None else None,
                )
                if logo is not None:
                    image = blend(image, logo)
                process.stdin.write(image.tobytes())
                written += 1
        process.stdin.close()
        if process.wait(timeout=120) != 0:
            raise RuntimeError("synthetic MP4 encoder failed")
    finally:
        if process.poll() is None:
            process.kill()
        process.wait(timeout=10)
    if (written != FRAME_COUNT or not dest.is_file()
            or not 1024 <= dest.stat().st_size <= MAX_VIDEO_BYTES):
        raise RuntimeError("bounded synthetic video unavailable")
    count = decoded_frame_count(dest)
    if count != FRAME_COUNT:
        raise RuntimeError("independent decoded-frame verification failed")
    return {
        "sha256": hashlib.sha256(dest.read_bytes()).hexdigest(),
        "source_frames": written,
        "decoded_frames": count,
        "caption_present": captions,
        "persistent_watermark_present": logo is not None,
    }


def is_expected_no_cue(exc: RuntimeError, base: Path) -> bool:
    if not str(exc).startswith("인식된 자막이 없습니다."):
        return False
    if any(base.with_suffix("." + ext).exists() for ext in ("srt", "ass")):
        raise RuntimeError("no-cue extraction unexpectedly created output")
    return True


def run_hse(video: Path, base: Path, engine, *, diagnostics=None) -> tuple[list[Cue], int]:
    if any(base.with_suffix("." + ext).exists() for ext in ("srt", "ass")):
        raise RuntimeError("existing subtitle output is protected")
    try:
        result = extract(str(video), engine, base, sample_fps=8.0, streaming=True,
                         diagnostics=diagnostics)
    except RuntimeError as exc:
        if is_expected_no_cue(exc, base):
            return [], 0
        raise
    srt = parse_strict_srt(result.srt_path.read_bytes())
    _, ass = parse_subtitle_bytes(result.ass_path.read_bytes(), source_url="study.ass")
    if len(srt) != len(ass) or len(srt) != result.cue_count:
        raise RuntimeError("exported SRT and ASS counts inconsistent")
    if any(
        a.text != b.text or abs(a.start - b.start) > 0.011
        or abs(a.end - b.end) > 0.011
        for a, b in zip(srt, ass)
    ):
        raise RuntimeError("exported SRT and ASS text/timing inconsistent")
    return srt, result.segment_count


def score_profiles(cues: dict[str, list[Cue]], language: str) -> dict:
    if set(cues) != set(CASES):
        raise ValueError("all three intervention profiles must be measured")
    truth = reference_cues(language)
    clean = evaluate(truth, cues["clean"], [])
    marked = evaluate(truth, cues["persistent_watermark"], [])
    empty_control = evaluate([], [], cues["watermark_only"])
    return {
        "clean": clean,
        "persistent_watermark": marked,
        "watermark_only": {
            "false_positive_cues": empty_control["blank_video_false_cues"],
            "no_caption_control_pass": empty_control["no_caption_control_pass"],
        },
        "watermark_minus_clean": {
            "missed_reference_cues_delta":
                marked["missed_reference_cues"] - clean["missed_reference_cues"],
            "extra_extracted_cues_delta":
                marked["extra_extracted_cues"] - clean["extra_extracted_cues"],
            "character_error_rate_delta": round(
                marked["aggregate_character_error_rate"]
                - clean["aggregate_character_error_rate"], 5,
            ),
            "strict_250ms_cues_delta":
                marked["strict_both_edges_250ms_and_text_wrap"]
                - clean["strict_both_edges_250ms_and_text_wrap"],
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=ROOT / "models/manifest.json")
    parser.add_argument("--languages", default="ko,zh")
    args = parser.parse_args()
    languages = tuple(part.strip() for part in args.languages.split(","))
    if (not languages or len(set(languages)) != len(languages)
            or any(language not in LANGUAGES for language in languages)):
        parser.error("distinct subset of ko,zh only")
    completed_rows = []
    started = time.monotonic()
    try:
        with tempfile.TemporaryDirectory(prefix="hse-watermark-mp4-") as td:
            root = Path(td)
            for lang in languages:
                engine = AdaptiveOCREngine(RapidOCRAdapter(
                    language=lang, model_dir=str(args.model_dir),
                    manifest_path=str(args.manifest),
                ))
                cues, sources, observed, audits = {}, {}, {}, {}
                for profile in CASES:
                    media = root / f"{lang}-{profile}.mp4"
                    sources[profile] = make_video(media, lang, profile)
                    previous_calls = engine.total_calls
                    previous_retries = engine.retry_calls
                    before = time.monotonic()
                    audits[profile] = InMemorySyntheticSegmentAudit()
                    cues[profile], segments = run_hse(
                        media, root / f"{lang}-{profile}.subtitles", engine,
                        diagnostics=audits[profile],
                    )
                    observed[profile] = {
                        "segments": segments,
                        "ocr_calls": engine.total_calls - previous_calls,
                        "retry_calls": engine.retry_calls - previous_retries,
                        "elapsed_seconds": round(time.monotonic() - before, 2),
                    }
                if len({p["sha256"] for p in sources.values()}) != len(CASES):
                    raise RuntimeError("video intervention source fingerprints collided")
                # Candidate selection has NO reference-text access.
                # This only scores a hypothetical postfilter in memory.
                prototypes = {}
                for profile in CASES:
                    traces = collect_lines(audits[profile])
                    folded = [
                        LineTrace(t.pts, fold_whitespace(t.text), t.cx, t.cy)
                        for t in traces
                    ]
                    variants = {}
                    # All six policies use the SAME OCR observations and
                    # original user-protected SRT cues; no fresh inference.
                    for name, records, normalizer, policy in (
                        ("exact_baseline", traces, None, "baseline"),
                        ("folded_baseline", folded, fold_whitespace, "baseline"),
                        ("exact_strict", traces, None, "strict"),
                        ("folded_strict", folded, fold_whitespace, "strict"),
                        ("exact_three_windows", traces, None, "three_windows"),
                        ("folded_three_windows", folded, fold_whitespace, "three_windows"),
                    ):
                        candidates = find_candidates(
                            records, FRAME_COUNT / FPS, flank_policy=policy,
                        )
                        if normalizer is None:
                            proposed, status = quarantine(cues[profile], candidates)
                        else:
                            proposed, status = quarantine(
                                cues[profile], candidates, normalizer=normalizer,
                            )
                        if profile == "watermark_only":
                            grading = {
                                "remaining_false_positive_cues": len(proposed),
                                "no_caption_control_pass": not proposed,
                            }
                        else:
                            graded = evaluate(reference_cues(lang), proposed, [])
                            grading = {
                                "cues": graded["extracted_cues"],
                                "missed": graded["missed_reference_cues"],
                                "extra": graded["extra_extracted_cues"],
                                "cer": graded["aggregate_character_error_rate"],
                                "strict_250ms_text_wrap":
                                    graded["strict_both_edges_250ms_and_text_wrap"],
                            }
                        variants[name] = {
                            "candidate_lines": status["candidate_stable_lines"],
                            "removed_lines": status["removed_lines"],
                            "dropped_cues": status["dropped_all_candidate_cues"],
                            "in_memory_only": True,
                            "proposed_grading": grading,
                        }
                    prototypes[profile] = {
                        "audited_ocr_line_traces": len(traces),
                        "raw_track_support": anonymous_support_summary(
                            traces, FRAME_COUNT / FPS,
                        ),
                        "folded_track_support": anonymous_support_summary(
                            folded, FRAME_COUNT / FPS,
                        ),
                        "variants": variants,
                    }
                row = {
                    "language": lang,
                    "sources": sources,
                    "metrics": score_profiles(cues, lang),
                    "production_pipeline": observed,
                    "conservative_overlay_prototype": prototypes,
                }
                completed_rows.append(row)
                print("HSE_WATERMARK_MP4_CASE " + json.dumps(
                    row, sort_keys=True, ensure_ascii=True,
                ), flush=True)
    except Exception as exc:
        print("HSE_WATERMARK_MP4_ERROR " + json.dumps({
            "error_class": type(exc).__name__,
            "completed_languages": len(completed_rows),
        }), flush=True)
        return 4
    print("HSE_WATERMARK_MP4_SUMMARY " + json.dumps({
        "schema": 1,
        "complete": len(completed_rows) == len(languages),
        "languages": list(languages),
        "profiles": list(CASES),
        "frames_per_video": FRAME_COUNT,
        "real_mp4_actual_ffmpeg_and_rapidocr": True,
        "reference_text_not_used_by_ocr": True,
        "subtitle_free_watermark_negative_control_included": True,
        "production_settings_unmodified": True,
        "experimental_quarantine_in_memory_only": True,
        "temporary_generated_data_only": True,
        "existing_user_data_unchanged": True,
        "real_anime_quality_tested": False,
        "release_gate_passed": False,
        "wall_seconds": round(time.monotonic() - started, 2),
    }, sort_keys=True), flush=True)
    return 0 if len(completed_rows) == len(languages) else 4


if __name__ == "__main__":
    raise SystemExit(main())
