"""Second, independent publisher-authored natural hard-subtitle OCR A/B (HOLD).

Wikimedia Foundation / Victor Grigas, Open Letter for Free Access to Wikipedia
(CC BY-SA 3.0). Two genuine 2013 public editorial video masters (one with
burned-in English subtitles, one without) and the separately published, pinned
2015 TimedText are compared in a private, disposable 40-second research slice.

This is neither synthetic subtitle compositing nor a frame-verified transcript.
No text, frames, copyrighted media, model files or user files are published.
Never promote an OCR option from this study alone. No paid model/API calls.
"""
from __future__ import annotations

import argparse
import hashlib
import math
import json
import re
from pathlib import Path
import subprocess
import tempfile
import time
import urllib.request

from compare_external_engine_exports import Cue, evaluate
from verify_commons_natural_hardsub import _hse_cues, _cue_error_diagnostics
from hardsub_extractor.ocr.adaptive import AdaptiveOCREngine
from hardsub_extractor.ocr.rapidocr_adapter import RapidOCRAdapter
from hardsub_extractor.ocr.light_text import LightCaptionEngine
from hardsub_extractor.ocr.light_temporal import LightTemporalWitnessEngine
from hardsub_extractor.runtime_paths import resolve_binary

# Official MediaWiki Commons action=query&prop=imageinfo observed 2026-11-13.
# A source changes? Fail-closed on bytes and SHA-1 BEFORE media decode.
PUBLIC_FILES = {
    "caption": (
        "https://upload.wikimedia.org/wikipedia/commons/a/ae/"
        "Open_Letter_for_Free_Access_to_Wikipedia_%28EN_subtitles%29.webm",
        109_377_579, "95af36a55d4db5f51407ab585c63e23fe340a70a",
    ),
    "clean": (
        "https://upload.wikimedia.org/wikipedia/commons/e/ee/"
        "Open_Letter_for_Free_Access_to_Wikipedia.webm",
        106_031_257, "0ab31daa28383f2cf4c95a96344628c6204cc9dc",
    ),
}
REFERENCE_URL = (
    "https://commons.wikimedia.org/w/api.php?"
    "action=query&format=json&prop=revisions&revids=167024147"
    "&rvprop=ids%7Csha1%7Ccontent&rvslots=main"
)
REFERENCE_REVISION = 167024147
REFERENCE_SHA1 = "3198744afec0e7dcf044e9fef4f90c564bd2b192"
RESEARCH_START = 15.0
RESEARCH_LENGTH = 40.0
EXPECTED_SOURCE_CUES = 34
EXPECTED_SLICE_CUES = 13
MAX_REFERENCE_BYTES = 48_000
MAX_SOURCE_BYTES = 115_000_000
MAX_DERIVED_BYTES = 80_000_000
MAX_NATIVE_DERIVED_BYTES = 220_000_000  # 1920x1080/8fps MPEG4 40s upper bound
CLOCK_RE = re.compile(
    r"^(\d\d):(\d\d):(\d\d),(\d{1,3})\s+-->\s+"
    r"(\d\d):(\d\d):(\d\d),(\d{1,3})$"
)


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise ValueError("source redirected outside exact official pinned URL")


def https_opener():
    # No cookies, user profile, authenticated browser or request credentials.
    return urllib.request.build_opener(NoRedirect())


def read_reference(*, opener=None) -> tuple[list[Cue], str]:
    opener = opener or https_opener()
    req = urllib.request.Request(REFERENCE_URL, headers={
        "User-Agent": "HSE-CC-BY-SA-OptIn-Benchmark/1.0",
        "Accept-Encoding": "identity",
    })
    with opener.open(req, timeout=40) as response:
        if response.status != 200 or response.geturl() != REFERENCE_URL:
            raise ValueError("untrusted official reference response")
        raw = response.read(MAX_REFERENCE_BYTES + 1)
    if not 300 < len(raw) <= MAX_REFERENCE_BYTES:
        raise ValueError("publisher reference size invalid")
    data = json.loads(raw)
    pages = data["query"]["pages"]
    if len(pages) != 1:
        raise ValueError("ambiguous publisher subtitle page")
    revisions = next(iter(pages.values()))["revisions"]
    if len(revisions) != 1:
        raise ValueError("publisher revision count invalid")
    revision = revisions[0]
    if (
        revision["revid"] != REFERENCE_REVISION
        or revision["sha1"] != REFERENCE_SHA1
    ):
        raise ValueError("publisher subtitle revision changed")
    content = revision["slots"]["main"]["*"]
    if (
        not isinstance(content, str)
        or hashlib.sha1(content.encode("utf-8")).hexdigest() != REFERENCE_SHA1
    ):
        raise ValueError("publisher subtitle content failed independent SHA-1")
    cues = []
    for block in re.split(r"\n[ \t]*\n", content.replace("\r\n", "\n")):
        parts = block.splitlines()
        if len(parts) < 3 or not parts[0].strip().isdigit():
            continue
        match = CLOCK_RE.fullmatch(parts[1].strip())
        if match is None:
            raise ValueError("malformed official SRT time")
        m = match.groups()
        def clock(h, minute, sec, fraction):
            if int(minute) >= 60 or int(sec) >= 60:
                raise ValueError("out-of-range published subtitle time")
            return int(h)*3600 + int(minute)*60 + int(sec) + int(fraction)/(10**len(fraction))
        start = clock(*m[:4])
        end = clock(*m[4:])
        text = "\n".join(line.strip() for line in parts[2:]).strip()
        if not text or not 0 <= start < end <= 122:
            raise ValueError("invalid published source subtitle cue")
        if cues and start < cues[-1].start:
            raise ValueError("unordered official subtitle reference")
        cues.append(Cue(start, end, text))
    if len(cues) != EXPECTED_SOURCE_CUES:
        raise ValueError("official publisher subtitle count changed")
    selected = [
        Cue(c.start-RESEARCH_START, c.end-RESEARCH_START, c.text)
        for c in cues
        if c.start >= RESEARCH_START and c.end <= RESEARCH_START+RESEARCH_LENGTH
    ]
    if len(selected) != EXPECTED_SLICE_CUES:
        raise ValueError("frozen public source study cue count changed")
    return selected, REFERENCE_SHA1


def get_original(target: Path, label: str, *, opener=None) -> str:
    opener = opener or https_opener()
    url, expected_size, expected_sha1 = PUBLIC_FILES[label]
    if target.exists() or target.is_symlink():
        raise FileExistsError("existing public media scratch")
    staging = target.with_suffix(".partial")
    if staging.exists() or staging.is_symlink():
        raise FileExistsError("existing public media partial")
    h1 = hashlib.sha1()
    h256 = hashlib.sha256()
    count = 0
    req = urllib.request.Request(url, headers={
        "User-Agent": "HSE-CC-BY-SA-OptIn-Benchmark/1.0",
        "Accept-Encoding": "identity",
    })
    try:
        with opener.open(req, timeout=60) as response:
            if response.status != 200 or response.geturl() != url:
                raise ValueError("public source redirect or status changed")
            declared = response.headers.get("Content-Length")
            if declared is not None and (
                not declared.isdecimal() or int(declared) != expected_size
            ):
                raise ValueError("public media length header mismatched")
            with staging.open("xb") as stream:
                while True:
                    data = response.read(1024*1024)
                    if not data:
                        break
                    count += len(data)
                    if count > MAX_SOURCE_BYTES or count > expected_size:
                        raise ValueError("public source exceeded exact size limit")
                    h1.update(data)
                    h256.update(data)
                    stream.write(data)
        if count != expected_size or h1.hexdigest() != expected_sha1:
            raise ValueError("public media failed immutable official SHA-1 pin")
        staging.rename(target)
        return h256.hexdigest()
    except BaseException:
        staging.unlink(missing_ok=True)
        raise


def make_video_slice(source: Path, target: Path, *, native_resolution: bool = False) -> str:
    if type(native_resolution) is not bool:
        raise ValueError('invalid explicit resolution policy')
    if target.exists() or target.is_symlink():
        raise FileExistsError("existing target slice")
    cmd = [
        resolve_binary("ffmpeg"), "-nostdin", "-hide_banner", "-loglevel", "error",
        "-protocol_whitelist", "file,pipe", "-threads", "2", "-i", str(source),
        "-ss", str(RESEARCH_START), "-t", str(RESEARCH_LENGTH),
        "-map", "0:v:0", "-an", "-sn", "-dn",
        "-vf", ("fps=8" if native_resolution else "fps=8,scale=1280:720:flags=lanczos"),
        "-c:v", "mpeg4", "-q:v", "3", "-pix_fmt", "yuv420p",
        "-f", "mp4", str(target),
    ]
    proc = subprocess.run(
        cmd, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE, timeout=360, check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError("verified public source video trim failed")
    size = target.stat().st_size
    allowed = MAX_NATIVE_DERIVED_BYTES if native_resolution else MAX_DERIVED_BYTES
    if not 10_000 < size <= allowed:
        raise ValueError("derived isolated study media size invalid")
    h = hashlib.sha256()
    with target.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b""):
            h.update(chunk)
    return h.hexdigest()



def _paired_raw_frames(source: Path) -> "np.ndarray":
    """Decode 1 frame/s of already SHA-pinned original; no SRT/score in input.

    The public source pairs are *editorial versions*, not proven lossless
    pixel-identical masters. Only qualified near-aligned frames may be used.
    No image or decoded text is logged or preserved on disk.
    """
    import numpy as np
    width, height = 640, 360
    maximum_frames = 42
    command = [
        resolve_binary("ffmpeg"), "-nostdin", "-hide_banner", "-loglevel", "error",
        "-threads", "2", "-protocol_whitelist", "file,pipe",
        "-i", str(source), "-ss", str(RESEARCH_START),
        "-t", str(RESEARCH_LENGTH), "-map", "0:v:0", "-an", "-sn", "-dn",
        "-vf", "fps=1,scale=640:360:flags=lanczos",
        "-pix_fmt", "bgr24", "-f", "rawvideo", "pipe:1",
    ]
    proc = subprocess.run(
        command, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
        stderr=subprocess.PIPE, timeout=180, check=False,
    )
    expected_frame_bytes = width * height * 3
    if proc.returncode or len(proc.stdout) > maximum_frames * expected_frame_bytes:
        raise RuntimeError("publisher frame-only decoder failed bounded read")
    if len(proc.stdout) % expected_frame_bytes:
        raise RuntimeError("publisher raw frame stream truncated")
    count = len(proc.stdout) // expected_frame_bytes
    if not 35 <= count <= maximum_frames:
        raise RuntimeError("unexpected publisher source frame count")
    return np.frombuffer(proc.stdout, dtype=np.uint8).reshape(
        count, height, width, 3
    )


def compare_publisher_overlay_geometry() -> dict:
    """Find probable bright editorial overlays by subtraction only.

    The reference SRT is verified, but is NEVER used to choose regions,
    set thresholds or influence image subtraction. This measures placement,
    not OCR accuracy or hard-subtitle ground truth. Differing shot edits
    and compression can produce false visual differences; retain caution.
    """
    import numpy as np
    import cv2

    reference, ref_pin = read_reference()
    with tempfile.TemporaryDirectory(prefix="hse-editorial-difference-") as tmp:
        folder = Path(tmp)
        paths = {k: folder / (k+".webm") for k in PUBLIC_FILES}
        pins = {k: get_original(paths[k], k) for k in PUBLIC_FILES}
        caption = _paired_raw_frames(paths["caption"])
        clean = _paired_raw_frames(paths["clean"])
        if caption.shape != clean.shape:
            raise ValueError("publisher versions differ in sample grid")
        frame_count, height, width, channels = caption.shape
        if channels != 3:
            raise ValueError("unexpected video color channels")
        all_row_counts = np.zeros(height, dtype=np.int64)
        all_col_counts = np.zeros(width, dtype=np.int64)
        per_frame = []
        aligned = 0
        selected = 0
        reference_active = 0
        reference_active_witnessed = 0
        for k in range(frame_count):
            a = cv2.cvtColor(caption[k], cv2.COLOR_BGR2GRAY).astype(np.int16)
            b = cv2.cvtColor(clean[k], cv2.COLOR_BGR2GRAY).astype(np.int16)
            absolute = np.abs(a-b)
            median_difference = float(np.median(absolute))
            time_at = RESEARCH_START + k + 0.5
            has_published_caption = any(
                c.start <= time_at-RESEARCH_START <= c.end
                for c in reference
            )
            reference_active += int(has_published_caption)
            # Reject likely shifted editing/camera cuts before using any
            # appearance differences. Compression is not a subtitle mask.
            acceptable = median_difference <= 23.0
            ink = (
                (a-b >= 42) & (a >= 160) & (b <= 215)
            ) if acceptable else np.zeros((height, width), dtype=np.bool_)
            count = int(np.count_nonzero(ink))
            if acceptable:
                aligned += 1
                if count >= 20:
                    selected += 1
                    all_row_counts += np.count_nonzero(ink, axis=1)
                    all_col_counts += np.count_nonzero(ink, axis=0)
                    reference_active_witnessed += int(has_published_caption)
            per_frame.append({
                "source_second": int(RESEARCH_START+k),
                "pair_median_luma_difference": round(median_difference, 2),
                "aligned_pair": acceptable,
                "bright_added_pixels": count,
                "independent_srt_on_screen": has_published_caption,
            })
        total = int(all_row_counts.sum())
        band_count = 12
        counts = []
        for band in range(band_count):
            y0 = height * band // band_count
            y1 = height * (band+1) // band_count
            counts.append(int(all_row_counts[y0:y1].sum()))
        def share(y0, y1):
            return round(
                float(all_row_counts[int(y0*height):int(y1*height)].sum())
                / total, 5
            ) if total else None
        row_cdf = np.cumsum(all_row_counts)
        def quantile(q):
            return (
                round(float(np.searchsorted(row_cdf, total*q)) / height, 4)
                if total else None
            )
        spatial_rows = [
            {"normalized_y_band": f"{i}/{band_count}..{i+1}/{band_count}",
             "bright_added_pixels": count}
            for i, count in enumerate(counts)
        ]
        return {
            "research_schema": 1,
            "research": "publisher_captioned_minus_editorial_clean_bright_overlay_geometry",
            "original_sources_sha1_and_size_verified": True,
            "original_sources_calculated_sha256": pins,
            "reference_revision_sha1_verified": ref_pin == REFERENCE_SHA1,
            "reference_cues": len(reference),
            "frame_pair_count": frame_count,
            "aligned_frame_pairs": aligned,
            "pairs_with_bright_added_evidence": selected,
            "bright_added_pixel_count": total,
            "bright_added_pixel_y_p10": quantile(.1),
            "bright_added_pixel_y_p50": quantile(.5),
            "bright_added_pixel_y_p90": quantile(.9),
            "bright_added_share_inside_default_bottom40": share(.58, .98),
            "bright_added_share_above_default_bottom40": share(0, .58),
            "bright_added_share_inside_lower65": share(.33, .98),
            "vertical_bands": spatial_rows,
            "frame_diagnostics": per_frame,
            "reference_active_sample_frames": reference_active,
            "reference_active_with_added_ink": reference_active_witnessed,
            "independent_ground_truth_subtitle_mask": False,
            "editorial_pair_exact_pixel_identity_unproven": True,
            "positive_residual_alone_does_not_prove_subtitle": True,
            "text_or_frames_logged": False,
            "model_calls": 0,
            "default_roi_or_ocr_changed": False,
            "release": "HOLD",
        }


def run(
    model_dir: Path,
    manifest: Path,
    *,
    native_resolution: bool = False,
    research_quiet_cap_seconds: float | None = None,
    research_region: str = "bottom40",
    research_signature_mode: str = "edges",
    research_ocr_mode: str = "light-text-temporal",
) -> dict:
    # Same production image/OCR code, with only the pre-existing optional
    # detector quiet-segment cap changed by explicit research approval.
    if research_quiet_cap_seconds is not None and (
        type(research_quiet_cap_seconds) is not float
        or not math.isfinite(research_quiet_cap_seconds)
        or not 0.5 <= research_quiet_cap_seconds <= 4.0
    ):
        raise ValueError("research cap must be a finite float 0.5..4 seconds")
    if type(native_resolution) is not bool:
        raise ValueError('resolution selection must be explicit bool')
    if type(research_region) is not str or research_region not in ("bottom40", "bottom25"):
        raise ValueError("unapproved real-media ROI comparison")
    if (
        type(research_signature_mode) is not str
        or research_signature_mode not in ("edges", "caption_band_white_3")
        or (research_signature_mode != "edges" and
            (research_region != "bottom40" or research_quiet_cap_seconds is not None))
    ):
        raise ValueError("caption band is explicitly opt-in on unchanged bottom40")
    if research_ocr_mode not in ("light-text-temporal", "light-island-temporal"):
        raise ValueError("unapproved combined OCR research mode")
    if not model_dir.is_dir() or not manifest.is_file():
        raise ValueError("SHA-pinned local OCR model bundle missing")
    # Make reference integrity failure cheap: do not fetch either video until
    # the independent caption witness passes.
    reference, ref_sha1 = read_reference()
    with tempfile.TemporaryDirectory(prefix="hse-second-natural-publisher-") as temp:
        root = Path(temp)
        input_files = {key: root / (key+".webm") for key in PUBLIC_FILES}
        source_sha256 = {
            key: get_original(input_files[key], key) for key in PUBLIC_FILES
        }
        slices = {key: root / (key+".mp4") for key in PUBLIC_FILES}
        slice_sha256 = {
            key: make_video_slice(
                input_files[key], slices[key], native_resolution=native_resolution,
            )
            for key in PUBLIC_FILES
        }
        from hardsub_extractor.frame_source import probe_video
        durations = [probe_video(str(p)).duration for p in slices.values()]
        if any(abs(x-RESEARCH_LENGTH) > .5 for x in durations):
            raise ValueError("positive and negative source timing not synchronized")
        from hardsub_extractor.detector import DetectorConfig
        from hardsub_extractor.tab_capture import roi_for_region
        # Default retains its original crop. Experimental region is already
        # user-selectable in HSE: no production feature or model change.
        study_roi = roi_for_region(research_region) if research_region != "bottom40" else None
        detector = (
            DetectorConfig(
                max_quiet_segment_seconds=research_quiet_cap_seconds,
                signature_mode=research_signature_mode,
            )
            if research_quiet_cap_seconds is not None
            or research_signature_mode != "edges"
            else None
        )
        measured = {}
        for mode in ("production", research_ocr_mode):
            scores = {}
            calls = 0
            positive = []
            control = []
            scanned = 0
            refined = 0
            for label in ("caption", "clean"):
                base = AdaptiveOCREngine(RapidOCRAdapter(
                    language="en", model_dir=str(model_dir),
                    manifest_path=str(manifest),
                ))
                if mode == "light-text-temporal":
                    engine = LightTemporalWitnessEngine(LightCaptionEngine(base))
                    refiner = engine.refine_timings
                elif mode == "light-island-temporal":
                    from hardsub_extractor.ocr.caption_islands import LightCaptionIslandEngine
                    engine = LightTemporalWitnessEngine(LightCaptionIslandEngine(base))
                    refiner = engine.refine_timings
                else:
                    engine, refiner = base, None
                out = root / (mode+"-"+label)
                extracted, segments = _hse_cues(
                    slices[label], out, engine, detector=detector,
                    timing_refiner=refiner, roi=study_roi,
                )
                if label == "caption":
                    positive = extracted
                    scanned = getattr(engine, "scanned_frames", 0)
                    refined = getattr(engine, "refined_cues", 0)
                    scores["positive_segments"] = segments
                else:
                    control = extracted
                    scores["control_segments"] = segments
                calls += base.total_calls
            scores["score"] = evaluate(reference, positive, control)
            scores["diagnostics"] = _cue_error_diagnostics(reference, positive, control)
            scores["local_model_calls"] = calls
            scores["optical_scanned_frames"] = scanned
            scores["optical_refined_cues"] = refined
            measured[mode] = scores
        for key in PUBLIC_FILES:
            h = hashlib.sha256()
            with input_files[key].open("rb") as handle:
                for data in iter(lambda: handle.read(1024*1024), b""):
                    h.update(data)
            if h.hexdigest() != source_sha256[key]:
                raise RuntimeError("official original changed during comparison")
        original = measured["production"]["score"]
        candidate = measured[research_ocr_mode]["score"]
        return {
            "research_schema": 2 if research_quiet_cap_seconds is not None else 1,
            "detector_quiet_cap_seconds": research_quiet_cap_seconds,
            "detector_pending_change_timeout_seconds": None,
            "research_signature_mode": research_signature_mode,
            "research_ocr_mode": research_ocr_mode,
            "OCR_input_crop_unchanged_for_caption_band": research_signature_mode == "caption_band_white_3",
            "product_detector_defaults_changed": False,
            "reference_timings_used_only_for_scoring": True,
            "source": "Wikimedia Foundation 2013 Open Letter, CC BY-SA 3.0",
            "original_source_is_author_burned_hardsubs": True,
            "real_editorial_clean_original": True,
            "separately_published_2015_reference": True,
            "reference_not_independently_frame_verified": True,
            "exact_reference_sha1": ref_sha1,
            "original_files_exact_official_sha1_size_verified": True,
            "original_files_own_sha256": source_sha256,
            "fixed_40s_derived_video_sha256": slice_sha256,
            "exact_same_burned_and_clean_video_pixels": False,
            "model": "same_official_English_PP_OCRv5_both_modes",
            "reference_cues": len(reference),
            "study_seconds": RESEARCH_LENGTH,
            "research_region": research_region,
            "existing_official_gui_roi_only": True,
            "roi_choice_not_based_on_reference_ocr_text": True,
            "derived_video_dimensions": "1920x1080" if native_resolution else "1280x720",
            "native_source_pixel_dimensions_preserved": native_resolution,
            "reencoded_sample_not_byte_identical_to_source": True,
            "original": measured["production"],
            "candidate": measured[research_ocr_mode],
            "changes_vs_baseline": {
                "exact_text_wrap": (
                    candidate["exact_visible_text_and_wrap"] -
                    original["exact_visible_text_and_wrap"]
                ),
                "strict_250ms_and_text": (
                    candidate["strict_both_edges_250ms_and_text_wrap"] -
                    original["strict_both_edges_250ms_and_text_wrap"]
                ),
                "false_clean_cues": (
                    candidate["blank_video_false_cues"] -
                    original["blank_video_false_cues"]
                ),
                "missed_reference_cues": (
                    candidate["missed_reference_cues"] -
                    original["missed_reference_cues"]
                ),
            },
            "manual_transcript_validation": False,
            "no_personal_media_or_credentials": True,
            "paid_model_requests": 0,
            "no_operating_files_modified": True,
            "release": "HOLD",
        }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--approve-public-sources", action="store_true")
    parser.add_argument("--reference-only", action="store_true")
    parser.add_argument("--geometry-only", action="store_true",
                        help="Measure pinned captioned/clean pixel differences without model inference")
    parser.add_argument("--research-region", choices=("bottom40", "bottom25"),
                        default="bottom40", help="Existing GUI local ROI; opt-in study only")
    parser.add_argument("--research-signature-mode",
                        choices=("edges", "caption_band_white_3"),
                        default="edges", help="Use narrow lower-band motion evidence while keeping full OCR pixels")
    parser.add_argument("--research-ocr-mode",
                        choices=("light-text-temporal", "light-island-temporal"),
                        default="light-text-temporal", help="Optional per-attempt compositional filter, not product default")
    parser.add_argument("--native-resolution", action="store_true",
                        help="Compare exact same 40s licensed source at native 1920x1080 instead of downscaled 720p")
    parser.add_argument("--research-quiet-cap-seconds", type=float, default=None,
                        help="Explicit experimental per-segment cap, 0.5..4 seconds; never changes product defaults")
    args = parser.parse_args()
    if not args.approve_public_sources:
        print("HSE_SECOND_NATURAL=PUBLIC_SOURCE_DOWNLOAD_NOT_APPROVED", flush=True)
        return 2
    try:
        if args.geometry_only:
            if args.reference_only or args.native_resolution or args.research_quiet_cap_seconds is not None or args.research_region != "bottom40" or args.research_signature_mode != "edges":
                raise ValueError("geometry study must not combine OCR or other modes")
            measured = compare_publisher_overlay_geometry()
            print("HSE_SECOND_PUBLISHER_GEOMETRY " + json.dumps(
                measured, sort_keys=True, ensure_ascii=True,
            ), flush=True)
            return 0
        if args.reference_only:
            reference, sha = read_reference()
            print("HSE_SECOND_NATURAL_REFERENCE " + json.dumps({
                "source_revision": REFERENCE_REVISION,
                "source_sha1_verified": sha == REFERENCE_SHA1,
                "cues_in_40s_slice": len(reference),
                "media_downloads": 0,
                "reference_is_independent": True,
                "not_frame_verified": True,
            }, sort_keys=True), flush=True)
            return 0
        if args.model_dir is None or args.manifest is None:
            raise ValueError("model directory and manifest required")
        summary = run(
            args.model_dir, args.manifest,
            native_resolution=args.native_resolution,
            research_quiet_cap_seconds=args.research_quiet_cap_seconds,
            research_region=args.research_region,
            research_signature_mode=args.research_signature_mode,
            research_ocr_mode=args.research_ocr_mode,
        )
        print("HSE_SECOND_NATURAL_RESULT " + json.dumps(
            summary, sort_keys=True, ensure_ascii=True,
        ), flush=True)
        return 0
    except Exception as exc:
        print("HSE_SECOND_NATURAL_HOLD " + json.dumps({
            "error_class": type(exc).__name__, "quality_verified": False,
            "operating_files_changed": False, "auto_retry": False,
        }, sort_keys=True), flush=True)
        return 3


if __name__ == "__main__":
    raise SystemExit(main())
