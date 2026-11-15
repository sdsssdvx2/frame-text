"""Publisher-authored open-caption video OCR acceptance; research ONLY.

This measures real published burned-in subtitles, rather than caption text
artificially overlaid by an HSE benchmark. The separately published official
TimedText is a provisional independent reference, not a frame-verified
transcription. No model output is fed into the reference or vice versa.
All media, model runtime outputs and subtitle files stay in a temporary folder.
This is not a natural-anime, multilingual, or production-release quality claim.

Sources: Wikimedia Foundation, "Kin Wikipedia 25 Anthem 15" (2025).
Media: CC BY-SA 4.0; see Wikimedia Commons file pages for full attributions.
The media are downloaded only during the explicit isolated study and never
distributed, packaged, uploaded as Actions artifacts, or installed for users.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import subprocess
import tempfile
import urllib.request

from compare_external_engine_exports import (
    Cue, align_cues, edit_distance, evaluate, parse_strict_srt,
)

# Official Wikimedia Commons original-file links; SHA-1 and exact byte lengths
# independently published in the file pages' Structured Data metadata.
# HTTPS transport + exact size + official SHA-1 is checked before decode.
# SHA-256 is independently calculated and logged for subsequent stronger pinning.
PUBLISHED = {
    "caption": (
        "https://upload.wikimedia.org/wikipedia/commons/4/40/"
        "Kin_Wikipedia_25_Anthem_15%22_16x9_Caption.webm",
        2_931_703, "ed23611d5372156ee8865292c76ac25262448537",
    ),
    "clean": (
        "https://upload.wikimedia.org/wikipedia/commons/3/39/"
        "Kin_Wikipedia_25_Anthem_15%22_16x9.webm",
        2_640_311, "ed69a42b592fdced4104ac862f83758180966ee6",
    ),
}
# Frozen, publicly visible TimedText source revision for the clean edit.
# It is independent of every model inference and post-hoc OCR output.
REFERENCE_URL = (
    "https://commons.wikimedia.org/w/index.php?title=TimedText:"
    "Kin_Wikipedia_25_Anthem_15%22_16x9.webm.en.srt"
    "&oldid=1117709522&action=raw"
)
EXPECTED_TIMES = (
    (0.000, 3.250), (3.416, 7.750),
    (8.041, 10.500), (10.583, 13.208),
)
EXPECTED_LINES = (2, 2, 1, 2)
# SHA-256 of the four separately published TimedText cue bodies joined by
# ASCII Record Separator (U+001E), after strict SRT parsing and trimming.
# Independently recorded from the fixed public page revision; pin the text,
# not just timings, without embedding/transmitting the original dialogue.
EXPECTED_TEXT_SHA256 = "e165e168c72c442024708639fbcd49e85f04cb3280f4dcfb64caafd6088400d4"
MAX_MEDIA_BYTES = 4_000_000
MAX_REFERENCE_BYTES = 16_384


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise ValueError("unapproved network redirect")


def _opener():
    # No cookie jar, authentication, browser profile, or user session.
    return urllib.request.build_opener(_NoRedirect())


def _get_media(target: Path, label: str, *, opener=None) -> str:
    if label not in PUBLISHED:
        raise ValueError("unrecognized reference source")
    url, expected_bytes, expected_sha1 = PUBLISHED[label]
    if target.exists() or target.is_symlink():
        raise FileExistsError("preexisting reference video")
    part = target.with_suffix(".partial")
    if part.exists() or part.is_symlink():
        raise FileExistsError("preexisting download scratch")
    opener = _opener() if opener is None else opener
    sha1, sha256 = hashlib.sha1(), hashlib.sha256()
    size = 0
    request = urllib.request.Request(
        url, headers={
            "User-Agent": "HSE-Published-Caption-Research/1.0",
            "Accept-Encoding": "identity",
        },
    )
    try:
        with opener.open(request, timeout=45) as response:
            if response.geturl() != url or response.status != 200:
                raise ValueError("unexpected media response")
            declared = response.headers.get("Content-Length")
            if declared is not None and (
                not declared.isdecimal() or int(declared) != expected_bytes
            ):
                raise ValueError("public media length changed")
            with part.open("xb") as stream:
                while True:
                    chunk = response.read(64 * 1024)
                    if not chunk:
                        break
                    size += len(chunk)
                    if size > MAX_MEDIA_BYTES or size > expected_bytes:
                        raise ValueError("public media exceeded pinned bounds")
                    sha1.update(chunk)
                    sha256.update(chunk)
                    stream.write(chunk)
        if size != expected_bytes or sha1.hexdigest() != expected_sha1:
            raise ValueError("official original media checksum mismatch")
        part.rename(target)
        return sha256.hexdigest()
    except BaseException:
        part.unlink(missing_ok=True)
        raise


def _get_reference(*, opener=None) -> tuple[list[Cue], str]:
    opener = _opener() if opener is None else opener
    request = urllib.request.Request(
        REFERENCE_URL, headers={
            "User-Agent": "HSE-Published-Caption-Research/1.0",
            "Accept-Encoding": "identity",
        },
    )
    with opener.open(request, timeout=30) as response:
        if response.geturl() != REFERENCE_URL or response.status != 200:
            raise ValueError("unexpected publisher transcript response")
        raw = response.read(MAX_REFERENCE_BYTES + 1)
    if not 80 < len(raw) <= MAX_REFERENCE_BYTES:
        raise ValueError("reference transcript size changed")
    cues = parse_strict_srt(raw)
    if len(cues) != 4 or any(
        abs(cue.start - start) > 0.001
        or abs(cue.end - end) > 0.001
        or len(cue.text.splitlines()) != EXPECTED_LINES[index]
        for index, (cue, (start, end)) in enumerate(zip(cues, EXPECTED_TIMES))
    ):
        raise ValueError("publisher TimedText cue boundaries or layout changed")
    observed_text_hash = hashlib.sha256(
        "\x1e".join(cue.text for cue in cues).encode("utf-8")
    ).hexdigest()
    if observed_text_hash != EXPECTED_TEXT_SHA256:
        raise ValueError("published TimedText exact source text changed")
    return cues, hashlib.sha256(raw).hexdigest()


def _probe(video: Path) -> dict:
    from hardsub_extractor.runtime_paths import resolve_binary
    result = subprocess.run(
        [
            resolve_binary("ffprobe"), "-v", "error",
            "-protocol_whitelist", "file,pipe",
            "-select_streams", "v:0",
            "-show_entries", "stream=codec_name,width,height:format=duration",
            "-of", "json", str(video),
        ],
        stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL, timeout=35, check=False,
    )
    if result.returncode or len(result.stdout) > 32768:
        raise RuntimeError("pinned public clip cannot be probed")
    data = json.loads(result.stdout)
    videos = data.get("streams", [])
    duration = float(data["format"]["duration"])
    if (
        len(videos) != 1 or videos[0].get("codec_name") != "av1"
        or videos[0].get("width") != 1920
        or videos[0].get("height") != 1080
        or not math.isfinite(duration)
        or not 14.9 < duration < 15.2
    ):
        raise ValueError("unrecognized published clip metadata")
    return {"codec": "av1", "width": 1920, "height": 1080,
            "seconds": round(duration, 3)}


def _hse_cues(
    video: Path, output: Path, engine, *, detector=None,
    timing_refiner=None, roi=None,
) -> tuple[list[Cue], int]:
    from hardsub_extractor.pipeline import extract
    from hardsub_extractor.subtitle_formats import parse_subtitle_bytes

    if any(output.with_suffix(ext).exists() for ext in (".srt", ".ass")):
        raise FileExistsError("study output already exists")
    try:
        report = extract(
            str(video), engine, output, sample_fps=8.0, streaming=True,
            detector=detector, timing_refiner=timing_refiner,
            **({"roi": roi} if roi is not None else {}),
        )
    except RuntimeError as exc:
        if not str(exc).startswith("인식된 자막이 없습니다."):
            raise
        if any(output.with_suffix(ext).exists() for ext in (".srt", ".ass")):
            raise RuntimeError("missing-cue path changed outputs") from None
        return [], 0

    detected = parse_strict_srt(report.srt_path.read_bytes())
    _, ass = parse_subtitle_bytes(
        report.ass_path.read_bytes(), source_url="published-test.ass",
    )
    if len(detected) != len(ass) or len(detected) != report.cue_count or any(
        a.text != b.text or abs(a.start - b.start) > 0.015
        or abs(a.end - b.end) > 0.015
        for a, b in zip(detected, ass)
    ):
        raise RuntimeError("HSE SRT/ASS output does not agree")
    return detected, report.segment_count



def _cue_error_diagnostics(
    reference: list[Cue], positive: list[Cue], control: list[Cue],
) -> dict:
    """Locate OCR/timing failures without printing publisher or OCR text.

    The matching algorithm uses timing alone, not similarity with the answer.
    Each extracted cue is reported at most once. There is no raw OCR text,
    frame, image or user input in the diagnostic record.
    """
    if len(positive) > 200 or len(control) > 200:
        raise ValueError("public 15-second study produced implausibly many cues")
    pairs = dict(align_cues(reference, positive))
    paired_extracted = set(pairs.values())

    def no_text(cue: Cue) -> dict:
        return {
            "start_ms": round(cue.start * 1000),
            "end_ms": round(cue.end * 1000),
            "text_chars_excluding_newline": len(cue.text.replace("\n", "")),
        }

    references = []
    for i, truth in enumerate(reference):
        record = {
            "reference_index": i + 1,
            "reference_chars_excluding_newline": len(truth.text.replace("\n", "")),
        }
        j = pairs.get(i)
        if j is None:
            record["matched"] = False
        else:
            guess = positive[j]
            t = truth.text.replace("\n", "")
            h = guess.text.replace("\n", "")
            record.update({
                "matched": True,
                "extracted_index": j + 1,
                "character_edits": edit_distance(t, h),
                "exact_visible_text": t == h,
                "exact_original_line_wrap": truth.text == guess.text,
                "recognized_chars_excluding_newline": len(h),
                "start_signed_offset_ms": round((guess.start - truth.start) * 1000),
                "end_signed_offset_ms": round((guess.end - truth.end) * 1000),
            })
        references.append(record)
    return {
        "reference_cue_diagnostics": references,
        "unmatched_extracted_cues": [
            {"extracted_index": i + 1, **no_text(cue)}
            for i, cue in enumerate(positive) if i not in paired_extracted
        ],
        "non_caption_control_detections": [
            {"control_index": i + 1, **no_text(cue)}
            for i, cue in enumerate(control)
        ],
        "raw_subtitle_or_ocr_text_logged": False,
        "alignment_uses_reference_transcript_text": False,
    }


def _study_detector(mode: str):
    """Opt-in research boundary variant; never touches production defaults.

    Two bounded temporal safeguards together can rescue a dialogue change
    masked by background motion. Both can also split legitimate persistent
    text and increase false positives; their efficacy must be measured on
    the exact same published caption/no-caption pair.
    """
    if mode == "production":
        return None
    if mode == "bounded-1p5s":
        from hardsub_extractor.detector import DetectorConfig
        return DetectorConfig(
            max_quiet_segment_seconds=1.5,
            max_pending_change_seconds=0.75,
        )
    if mode == "caption-band-white":
        from hardsub_extractor.detector import DetectorConfig
        # Exact opt-in implementation measured on the second published pair.
        # Keep full baseline OCR crop; change only lower-band transition signal.
        return DetectorConfig(signature_mode="caption_band_white_3")
    raise ValueError("unsupported opt-in research detector")



# Use the exact same implementation as the opt-in local product path.
# The independently pinned publisher-caption benchmark now measures its
# actual code rather than a duplicate research-only copy.
from hardsub_extractor.ocr.line_focus import BottomTwoLineEngine as _BottomTwoLineResearch


def run_study(
    model_dir: Path, manifest: Path, *, opener=None,
    detector_mode: str = "production",
    ocr_mode: str = "production",
    compare_production: bool = False,
) -> dict:
    detector = _study_detector(detector_mode)
    if ocr_mode not in {"production", "bottom-two-lines", "coherent-caption-islands", "light-text-only", "light-text-temporal", "light-island-temporal", "pixel-timing-only"}:
        raise ValueError("unsupported opt-in research OCR row limit")
    if type(compare_production) is not bool or (compare_production and ocr_mode == "production"):
        raise ValueError("paired baseline comparison requires an explicit non-production OCR mode")
    from hardsub_extractor.ocr.adaptive import AdaptiveOCREngine
    from hardsub_extractor.ocr.rapidocr_adapter import RapidOCRAdapter

    with tempfile.TemporaryDirectory(prefix="hse-commons-real-hardsub-") as temp:
        root = Path(temp)
        # Verify the inexpensive immutable publisher transcript FIRST. If its
        # exact text, cue geometry or timestamp differs, do not even download
        # the two original videos or initialize expensive OCR models.
        reference, reference_sha256 = _get_reference(opener=opener)
        paths = {k: root / (k + ".webm") for k in PUBLISHED}
        pins = {
            name: _get_media(paths[name], name, opener=opener)
            for name in ("caption", "clean")
        }
        metadata = {label: _probe(paths[label]) for label in ("caption", "clean")}
        if abs(metadata["caption"]["seconds"] - metadata["clean"]["seconds"]) > 0.1:
            raise RuntimeError("published positive/control durations disagree")
        engine = AdaptiveOCREngine(RapidOCRAdapter(
            language="en", model_dir=str(model_dir),
            manifest_path=str(manifest),
        ))
        if ocr_mode == "bottom-two-lines":
            engine = _BottomTwoLineResearch(engine)
        elif ocr_mode == "coherent-caption-islands":
            from hardsub_extractor.ocr.caption_islands import CaptionIslandEngine
            engine = CaptionIslandEngine(engine)
        elif ocr_mode == "light-text-only":
            from hardsub_extractor.ocr.light_text import LightCaptionEngine
            engine = LightCaptionEngine(engine)
        elif ocr_mode == "light-text-temporal":
            from hardsub_extractor.ocr.light_text import LightCaptionEngine
            from hardsub_extractor.ocr.light_temporal import LightTemporalWitnessEngine
            engine = LightTemporalWitnessEngine(LightCaptionEngine(engine))
        elif ocr_mode == "light-island-temporal":
            from hardsub_extractor.ocr.caption_islands import LightCaptionIslandEngine
            from hardsub_extractor.ocr.light_temporal import LightTemporalWitnessEngine
            engine = LightTemporalWitnessEngine(LightCaptionIslandEngine(engine))
        elif ocr_mode == "pixel-timing-only":
            from hardsub_extractor.ocr.light_temporal import LightTemporalWitnessEngine
            # Same unfiltered AdaptiveOCREngine as production. Only the
            # optical cue boundary witness is added, not text filtering.
            engine = LightTemporalWitnessEngine(engine)
        timing_refiner = (
            engine.refine_timings
            if ocr_mode in {"light-text-temporal", "light-island-temporal", "pixel-timing-only"}
            else None
        )
        actual, positive_segments = _hse_cues(
            paths["caption"], root / "caption-result", engine,
            detector=detector, timing_refiner=timing_refiner,
        )
        positive_refined = getattr(engine, "refined_cues", 0)
        positive_scanned = getattr(engine, "scanned_frames", 0)
        positive_decoders = getattr(engine, "decoder_launches", 0)
        positive_decoder_errors = getattr(engine, "failed_decoder_batches", 0)
        negative_engine = engine
        negative_refiner = timing_refiner
        if ocr_mode in {"light-text-temporal", "light-island-temporal"}:
            engine.witnesses.clear()
        elif ocr_mode == "pixel-timing-only":
            # Do not reuse pixel evidence from the positive movie on the
            # independent clean edit. Both share the same base OCR model,
            # but NEVER share optical witness state across source videos.
            negative_engine = LightTemporalWitnessEngine(engine.inner)
            negative_refiner = negative_engine.refine_timings
        negative, negative_segments = _hse_cues(
            paths["clean"], root / "control-result", negative_engine,
            detector=detector, timing_refiner=negative_refiner,
        )
        if any(
            hashlib.sha256(paths[name].read_bytes()).hexdigest() != pins[name]
            for name in pins
        ):
            raise RuntimeError("published original reference modified during OCR")
        score = evaluate(reference, actual, negative)
        # A/B on the exact same independently pinned original captioned and
        # clean videos. This bypasses flaky Windows localhost/Chrome capture
        # transport so that style-specific OCR quality can be assessed on its
        # own merits. No captions or OCR strings are written to the report.
        baseline_score = None
        baseline_calls = None
        if compare_production:
            baseline = AdaptiveOCREngine(RapidOCRAdapter(
                language="en", model_dir=str(model_dir),
                manifest_path=str(manifest),
            ))
            baseline_actual, _ = _hse_cues(
                paths["caption"], root / "baseline-caption", baseline,
                detector=detector,
            )
            baseline_control, _ = _hse_cues(
                paths["clean"], root / "baseline-clean", baseline,
                detector=detector,
            )
            if any(
                hashlib.sha256(paths[name].read_bytes()).hexdigest() != pins[name]
                for name in pins
            ):
                raise RuntimeError("publisher original changed during paired comparison")
            baseline_score = evaluate(reference, baseline_actual, baseline_control)
            baseline_calls = baseline.total_calls
        report = {
            "schema": 1,
            "research_detector": detector_mode,
            "research_ocr_row_selector": ocr_mode,
            "research_dropped_ocr_rows": getattr(engine, "dropped_rows", 0),
            "research_modified_ocr_observations": getattr(
                engine, "modified_observations", 0,
            ),
            "research_light_text_candidate_frames": getattr(
                engine, "candidate_frames", 0,
            ),
            "research_light_text_discarded_rows": getattr(
                engine, "discarded_rows", 0,
            ),
            "research_temporal_refined_positive_cues": positive_refined,
            "research_temporal_scanned_positive_frames": positive_scanned,
            "research_temporal_decoder_launches_positive": positive_decoders,
            "research_temporal_decoder_failures_positive": positive_decoder_errors,
            "research_temporal_decoder_launches_control": (
                getattr(negative_engine, "decoder_launches", 0)
                if ocr_mode == "pixel-timing-only" else
                getattr(engine, "decoder_launches", 0) - positive_decoders
            ),
            "research_temporal_refined_control_cues": (
                getattr(negative_engine, "refined_cues", 0)
                if ocr_mode == "pixel-timing-only" else
                getattr(engine, "refined_cues", 0) - positive_refined
            ),
            "controlled_detector_change_only": detector is not None,
            "source": "Wikimedia Foundation public 2025 video, CC BY-SA 4.0",
            "natural_publisher_burned_in_subtitles": True,
            "not_artificial_benchmark_overlay": True,
            "publisher_original_and_clean_editorial_pair": True,
            "publisher_reference_revision": 1117709522,
            "reference_is_external_timedtext_not_pixel_verified": True,
            "reference_cues": len(reference),
            "reference_sha256": reference_sha256,
            "video_sha256": pins,
            "original_source_bytes_and_sha1_verified": True,
            "metadata": metadata,
            "real_english_model_inference": True,
            "positive_segments": positive_segments,
            "control_segments": negative_segments,
            "ocr_calls": engine.total_calls,
            "retry_calls": engine.retry_calls,
            "scores": score,
            "cue_error_diagnostics": _cue_error_diagnostics(
                reference, actual, negative,
            ),
            "negative_control_contains_other_possible_scene_text": True,
            "hse_default_model_or_detection_changed": False,
            "no_user_content_or_media_redistribution": True,
            "media_models_and_outputs_retained": False,
            "only_15s_english_not_general_anime_accuracy": True,
            "release_gate_passed": False,
        }
        if compare_production:
            if ocr_mode == "pixel-timing-only":
                # The baseline is re-extracted from the independently pinned
                # publisher originals. Published TimedText is scoring-only.
                if (
                    len(actual) != len(baseline_actual)
                    or len(negative) != len(baseline_control)
                    or any(a.text != b.text for a, b in zip(actual, baseline_actual))
                    or any(a.text != b.text for a, b in zip(negative, baseline_control))
                ):
                    raise RuntimeError("pixel timing altered real OCR cue text or count")
                boundaries = [
                    max(abs(a.start-b.start), abs(a.end-b.end))
                    for a, b in zip(actual, baseline_actual)
                ]
                control_boundaries = [
                    max(abs(a.start-b.start), abs(a.end-b.end))
                    for a, b in zip(negative, baseline_control)
                ]
                base_mae = baseline_score["mean_absolute_edge_error_seconds"]
                candidate_mae = score["mean_absolute_edge_error_seconds"]
                report.update({
                    "pixel_only_text_and_count_preserved_positive": True,
                    "pixel_only_text_and_count_preserved_control": True,
                    "pixel_only_modified_positive_times": sum(
                        delta > .001 for delta in boundaries
                    ),
                    "pixel_only_max_positive_boundary_delta_seconds": round(
                        max(boundaries, default=0.), 3
                    ),
                    "pixel_only_modified_control_times": sum(
                        delta > .001 for delta in control_boundaries
                    ),
                    "pixel_only_independent_publisher_reference": True,
                    "pixel_only_publisher_timing_not_frame_verified": True,
                    "pixel_only_strict_250ms_change": (
                        score["strict_both_edges_250ms_and_text_wrap"] -
                        baseline_score["strict_both_edges_250ms_and_text_wrap"]
                    ),
                    "pixel_only_mean_absolute_edge_error_change_seconds": (
                        round(candidate_mae-base_mae, 5)
                        if candidate_mae is not None and base_mae is not None
                        else None
                    ),
                    "pixel_only_no_default_promotion": True,
                })
            report.update({
                "paired_comparison": True,
                "paired_production_model": "same_pinned_english_rapidocr",
                "paired_production_detector_unchanged": True,
                "paired_production_video_sha256": pins,
                "paired_production_scores": baseline_score,
                "paired_production_ocr_calls": baseline_calls,
                "comparison_not_release_approval": True,
            })
        return report


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument(
        "--manifest", type=Path, default=Path("models/manifest.json"),
    )
    parser.add_argument("--accept-public-download", action="store_true")
    parser.add_argument("--compare-production", action="store_true",
                        help="Use identical pinned public source and clean control for optional OCR A/B")
    parser.add_argument(
        "--detector-research",
        choices=("production", "bounded-1p5s", "caption-band-white"),
        default="production",
    )
    parser.add_argument(
        "--ocr-research",
        choices=("production", "bottom-two-lines", "coherent-caption-islands", "light-text-only", "light-text-temporal", "light-island-temporal", "pixel-timing-only"),
        default="production",
    )
    args = parser.parse_args(argv)
    if not args.accept_public_download:
        print("HSE_NATURAL_HARDSUB=DOWNLOAD_NOT_APPROVED")
        return 2
    try:
        report = run_study(
            args.model_dir, args.manifest,
            detector_mode=args.detector_research,
            ocr_mode=args.ocr_research,
            compare_production=args.compare_production,
        )
    except Exception as exc:
        # No raw frames, transcript, candidate text, or user/private paths.
        print("HSE_NATURAL_HARDSUB_ERROR " + json.dumps({
            "error_class": type(exc).__name__, "quality_measured": False,
        }), flush=True)
        return 3
    print("HSE_NATURAL_HARDSUB_RESULT " + json.dumps(
        report, ensure_ascii=False, sort_keys=True,
    ), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
