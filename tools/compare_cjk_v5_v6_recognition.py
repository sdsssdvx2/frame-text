"""Research-only PP-OCRv5 vs PP-OCRv6-small CJK recognizer differential.

Original v5 detector pixels, selected temporal samples, image shapes,
source file bytes and subtitle filtering are *identical*. The only A/B
variable is the recognition model/dictionary. Only synthetic ground truth
for grading after BOTH engines have already produced outputs is used.
All downloads are fixed official URLs with pinned SHA-256 and bounded size.
No source frame, decoded word, customer data, or credential is logged.
"""
from __future__ import annotations

import argparse
from contextlib import closing
import hashlib
import json
from pathlib import Path
import tempfile
import urllib.request

from hardsub_extractor.detector import iter_segment_frames
from hardsub_extractor.frame_source import FFmpegPTSFrameSource
from hardsub_extractor.model_assets import resolve_model_paths
from hardsub_extractor.ocr.rapidocr_adapter import (
    RapidOCRAdapter, _subtitle_box_lines,
)
from diagnose_raw_multiline_ocr_boxes import (
    analyze_raw_result, evaluate_samples, MAX_MODEL_FRAMES,
)
from verify_animated_background_ocr_quality import encode_video
from verify_multilang_video_pipeline import EXPECTED_SPANS, VIDEO_TEXT

LANGUAGES = ("ja", "zh")
ROOT_URL = "https://www.modelscope.cn/models/RapidAI/RapidOCR/resolve/v3.9.2/"
V6_RECOGNIZER = {
    "path": "onnx/PP-OCRv6/rec/PP-OCRv6_rec_small.onnx",
    "sha256": "6f327246b50388f3c176ae304bd95767ea6dc0c9ae92153ef8cbe210b3c14884",
    "max_bytes": 80 * 1024 * 1024,
}
V6_DICTIONARY = {
    "path": "paddle/PP-OCRv6/rec/PP-OCRv6_rec_small/ppocrv6_dict.txt",
    # Independently recorded PP-OCRv6 dictionary checksum: must still
    # match exact official tagged ModelScope bytes before any inference.
    "sha256": "b5f2bfe2bdd9448429e3e82b51c789775d9b42f2403d082b00662eb77e401c5d",
    "max_bytes": 2 * 1024 * 1024,
}


def verify_official_resource(spec, content: bytes) -> bool:
    return bool(
        content and len(content) <= spec["max_bytes"]
        and hashlib.sha256(content).hexdigest() == spec["sha256"]
    )


def retrieve_verified(spec, root: Path) -> Path:
    if (not spec["path"] or ".." in spec["path"]
            or spec["path"].startswith("/") or chr(92) in spec["path"]):
        raise ValueError("unsafe official model asset path")
    output = root / Path(spec["path"]).name
    if output.exists() or output.is_symlink():
        raise RuntimeError("isolated model asset path already exists")
    request = urllib.request.Request(
        ROOT_URL + spec["path"],
        headers={"User-Agent": "HSE-isolated-OCR-recognition-research/1"},
    )
    hasher = hashlib.sha256()
    total = 0
    partial = output.with_suffix(output.suffix + ".part")
    try:
        with urllib.request.urlopen(request, timeout=60) as response, partial.open("xb") as file:
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                total += len(chunk)
                if total > spec["max_bytes"]:
                    raise RuntimeError("research asset exceeds hard size limit")
                hasher.update(chunk)
                file.write(chunk)
        if not 0 < total <= spec["max_bytes"] or hasher.hexdigest() != spec["sha256"]:
            raise RuntimeError("official model SHA-256 verification failed")
        partial.replace(output)
    except Exception:
        partial.unlink(missing_ok=True)
        raise
    return output


def make_v6_recognizer(models, manifest, language, v6_rec, v6_dict):
    """Use verified V5 detection unchanged, switching ONLY CJK recognition."""
    from rapidocr import EngineType, LangRec, ModelType, OCRVersion, RapidOCR
    base = resolve_model_paths(
        models, manifest_path=manifest, language=language, verify=True,
    )
    if not v6_rec.is_file() or not v6_dict.is_file():
        raise RuntimeError("verified PP-OCRv6 research assets missing")
    params = {
        "Det.engine_type": EngineType.ONNXRUNTIME,
        "Det.model_type": ModelType.MOBILE,
        "Det.ocr_version": OCRVersion.PPOCRV5,
        "Det.model_path": str(base.detector),
        "Rec.engine_type": EngineType.ONNXRUNTIME,
        "Rec.lang_type": LangRec.CH,
        "Rec.model_type": ModelType.SMALL,
        "Rec.ocr_version": OCRVersion.PPOCRV6,
        "Rec.model_path": str(v6_rec),
        "Rec.rec_keys_path": str(v6_dict),
        "Global.use_cls": False,
    }
    return RapidOCR(params=params)


def same_frame_model_result(engine, image):
    return engine(
        image, use_det=True, use_cls=False, use_rec=True,
        text_score=0.45,
    )


def sample_model_difference(source, language, engines):
    rows = {name: [] for name in engines}
    model_calls = {name: 0 for name in engines}
    with closing(iter(FFmpegPTSFrameSource(str(source), sample_fps=8.0))) as frames:
        with closing(iter_segment_frames(frames)) as segments:
            for segment in segments:
                for sample in segment.samples:
                    if max(model_calls.values()) >= MAX_MODEL_FRAMES:
                        raise RuntimeError("excessive OCR model frames")
                    # V5/V6 are invoked before ground-truth matching.
                    image = sample.image
                    observations = {
                        name: same_frame_model_result(engine, image)
                        for name, engine in engines.items()
                    }
                    reference_index = next((
                        i for i, (start, end) in enumerate(EXPECTED_SPANS)
                        if start <= sample.pts < end
                    ), None)
                    for name, output in observations.items():
                        model_calls[name] += 1
                        if reference_index is None:
                            diag = {"outside_reference": True}
                        else:
                            diag = analyze_raw_result(
                                output, image.shape, language,
                                VIDEO_TEXT[language][reference_index],
                            )
                        rows[name].append({
                            "pts": sample.pts, "diagnosis": diag,
                        })
    return {
        name: {
            "model_calls": model_calls[name],
            "reference_cues": evaluate_samples(rows[name], language),
        } for name in engines
    }


def blank_false_positive_counts(source, language, engines):
    counts = {name: {"any_filtered_line": 0, "two_filtered_lines": 0, "frames": 0}
              for name in engines}
    with closing(iter(FFmpegPTSFrameSource(str(source), sample_fps=8.0))) as frames:
        with closing(iter_segment_frames(frames)) as segments:
            for segment in segments:
                for sample in segment.samples:
                    if max(x["frames"] for x in counts.values()) >= MAX_MODEL_FRAMES:
                        raise RuntimeError("no-caption OCR research bound exceeded")
                    pixels = sample.image
                    for name, engine in engines.items():
                        output = same_frame_model_result(engine, pixels)
                        texts = list(getattr(output, "txts", None) or ())
                        boxes = getattr(output, "boxes", None)
                        lines = _subtitle_box_lines(
                            texts, boxes, pixels.shape, require_hangul=False,
                            language=language,
                        )
                        counts[name]["frames"] += 1
                        counts[name]["any_filtered_line"] += bool(lines)
                        counts[name]["two_filtered_lines"] += len(lines) >= 2
    return counts



def compare_pinned_film_frame_groundtruth(model_dir, manifest, v6_rec, v6_dict):
    """Judge V5 and V6 on the SAME 48 pinned JA/ZH frames and exact text.

    Use the 72-frame dataset's existing verified manifest/image downloader;
    exclude Korean because V6 Korean recognition isn't a validated profile.
    No transcript enters inference; no model text or images leave the scratch.
    Both model families use HSE's unchanged V5 detector, geometry filter,
    bottom-40% ROI, adaptive OCR and optional light-caption postfilter.
    """
    import copy
    import gc
    import time
    from collections import defaultdict

    import cv2
    import numpy as np

    from hardsub_extractor.models import Frame
    from hardsub_extractor.tab_capture import roi_for_region
    from hardsub_extractor.ocr.adaptive import AdaptiveOCREngine
    from hardsub_extractor.ocr.light_text import LightCaptionEngine
    from verify_geeklink_cjk_frame_groundtruth import (
        DATASET_COMMIT, MANIFEST_BLOB, SAMPLES, TOTAL_FRAMES,
        MAX_IMAGE_PIXELS, _norm, _read_manifest, _pull_verified_images,
    )
    from compare_external_engine_exports import edit_distance

    by_id = _read_manifest()
    image_bytes = _pull_verified_images()
    if len(image_bytes) != TOTAL_FRAMES:
        raise RuntimeError("incomplete independently pinned CJK image corpus")
    roi = roi_for_region("bottom40")
    metrics = defaultdict(
        lambda: {"n": 0, "characters": 0, "edits": 0, "exact": 0, "blank": 0},
    )
    measured_calls = []
    for language in LANGUAGES:
        pool = tuple(sample for sample in SAMPLES if sample["lang"] == language)
        if len(pool) != 24 or sum(bool(s["watermark"]) for s in pool) != 12:
            raise RuntimeError("CJK frozen strata changed")
        v5_adapter = RapidOCRAdapter(
            language=language, model_dir=str(model_dir),
            manifest_path=str(manifest),
        )
        # Copy the already-verified adapter configuration; replace only the
        # official recognition engine. The ONNX detector remains pinned V5.
        v6_adapter = copy.copy(v5_adapter)
        v6_adapter.engine = make_v6_recognizer(
            model_dir, manifest, language, v6_rec, v6_dict,
        )
        for model_name, adapter in (
            ("v5_mobile", v5_adapter), ("v6_small", v6_adapter),
        ):
            for mode in ("baseline", "light"):
                adaptive = AdaptiveOCREngine(adapter)
                ocr = (
                    adaptive if mode == "baseline"
                    else LightCaptionEngine(adaptive)
                )
                began = time.monotonic()
                for sample in pool:
                    packed = np.frombuffer(
                        image_bytes[sample["id"]], dtype=np.uint8,
                    )
                    pixels = cv2.imdecode(packed, cv2.IMREAD_COLOR)
                    if (
                        pixels is None or pixels.ndim != 3
                        or pixels.shape[2] != 3
                        or pixels.shape[0] * pixels.shape[1] > MAX_IMAGE_PIXELS
                    ):
                        raise ValueError("verified CJK film frame decode invalid")
                    height, width = pixels.shape[:2]
                    x, y, w, h = roi.pixels(width, height)
                    cropped = np.ascontiguousarray(pixels[y:y+h, x:x+w])
                    # Independent input copies: a model must not modify
                    # the pixels seen by another model or filter.
                    observation = ocr.recognize(Frame(0.0, cropped.copy()))
                    if not isinstance(observation.text, str):
                        raise TypeError("CJK recognition returned invalid text")
                    actual = _norm(observation.text)
                    expected = _norm(by_id[sample["id"]]["ground_truth"])
                    if not expected:
                        raise ValueError("empty independent annotation")
                    for subset in (
                        "all", "watermark" if sample["watermark"] else "clean",
                    ):
                        bucket = metrics[(language, model_name, mode, subset)]
                        bucket["n"] += 1
                        bucket["characters"] += len(expected)
                        bucket["edits"] += edit_distance(actual, expected)
                        bucket["exact"] += int(actual == expected)
                        bucket["blank"] += int(not actual)
                if not 0 < adaptive.total_calls <= 3 * len(pool):
                    raise RuntimeError("CJK per-mode adaptive inference exceeded limit")
                measured_calls.append({
                    "language": language, "model": model_name, "mode": mode,
                    "ocr_calls": int(adaptive.total_calls),
                    "elapsed_seconds": round(time.monotonic() - began, 2),
                })
                del ocr, adaptive
        del v5_adapter, v6_adapter
        gc.collect()
    if len(metrics) != len(LANGUAGES) * 2 * 2 * 3:
        raise RuntimeError("CJK matched model/stratum measurements incomplete")
    result = {
        "schema": 1,
        "source": "GeekLinkDev/geeklink-ocr-benchmark",
        "pinned_dataset_commit": DATASET_COMMIT,
        "pinned_manifest_git_blob": MANIFEST_BLOB,
        "verified_source_images": TOTAL_FRAMES,
        "judged_independent_images": 48,
        "languages": list(LANGUAGES),
        "model_variants": ["v5_mobile", "v6_small"],
        "postfilter_variants": ["baseline", "light"],
        "same_input_roi": "bottom40",
        "unchanged_sha_pinned_v5_detector": True,
        "actual_publisher_burned_in_dialogue": False,
        "real_film_background_synthetic_caption": True,
        "actual_natural_video_timing_untested": True,
        "model_default_unchanged": True,
        "reference_only_used_after_recognition": True,
        "srt_ass_or_media_exported": False,
        "user_media_or_credentials_used": False,
        "paid_api_calls": 0,
        "release": "HOLD",
        "model_calls": measured_calls,
        "metrics": [
            {
                "language": language, "model": model, "mode": mode,
                "subset": subset,
                "n": value["n"], "characters": value["characters"],
                "character_edits": value["edits"], "exact": value["exact"],
                "blank": value["blank"],
                "cer": round(value["edits"] / value["characters"], 5),
            }
            for (language, model, mode, subset), value in sorted(metrics.items())
        ],
    }
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--manifest", default="models/manifest.json")
    ap.add_argument(
        "--pinned-groundtruth-ja-zh", action="store_true",
        help="Only compare JA/ZH model variants on existing pinned film-frame ground truth",
    )
    ap.add_argument(
        "--approve-cc-dataset-download", action="store_true",
        help="Explicitly approve existing CC0/CC film-frame research download",
    )
    opts = ap.parse_args()
    if opts.pinned_groundtruth_ja_zh and not opts.approve_cc_dataset_download:
        print("HSE_V6_CJK_FILM=PUBLIC_DATASET_NOT_APPROVED", flush=True)
        return 2
    reports = []
    try:
        with tempfile.TemporaryDirectory(prefix="hse-ppocr-v6-cjk-ab-") as td:
            root = Path(td)
            v6root = root / "official-v6"
            v6root.mkdir()
            rec = retrieve_verified(V6_RECOGNIZER, v6root)
            dictionary = retrieve_verified(V6_DICTIONARY, v6root)
            if opts.pinned_groundtruth_ja_zh:
                result = compare_pinned_film_frame_groundtruth(
                    Path(opts.model_dir), Path(opts.manifest), rec, dictionary,
                )
                print("HSE_V6_CJK_FILM_RESULT " +
                      json.dumps(result, sort_keys=True), flush=True)
                return 0
            for lang in LANGUAGES:
                srcroot = root / lang
                srcroot.mkdir()
                video = srcroot / "source.mp4"
                empty = srcroot / "empty.mp4"
                encode_video(video, lang, style="white_two_line")
                encode_video(empty, lang, style="white_two_line",
                             include_subtitles=False)
                original = hashlib.sha256(video.read_bytes()).hexdigest()
                no_text_hash = hashlib.sha256(empty.read_bytes()).hexdigest()
                old = RapidOCRAdapter(
                    language=lang, model_dir=str(opts.model_dir),
                    manifest_path=str(opts.manifest),
                )
                new = make_v6_recognizer(
                    opts.model_dir, opts.manifest, lang, rec, dictionary,
                )
                engines = {"v5_mobile": old.engine, "v6_small": new}
                measure = sample_model_difference(video, lang, engines)
                controls = blank_false_positive_counts(empty, lang, engines)
                if (hashlib.sha256(video.read_bytes()).hexdigest() != original
                        or hashlib.sha256(empty.read_bytes()).hexdigest() != no_text_hash):
                    raise RuntimeError("source MP4 changed during model A/B")
                row = {
                    "language": lang,
                    "source_encoded_video": True,
                    "same_v5_detector_and_selected_frames": True,
                    "v6_model_and_dictionary_official_bytes_sha_verified": True,
                    "references_used_only_after_model_inference": True,
                    "model_results": measure,
                    "no_caption_candidate_counts": controls,
                }
                reports.append(row)
                print("HSE_V6_CJK_MODEL_CASE "+
                      json.dumps(row, sort_keys=True), flush=True)
    except Exception as exc:
        print("HSE_V6_CJK_MODEL_ERROR "+
              json.dumps({"error_class": type(exc).__name__}))
        return 4
    complete = len(reports) == len(LANGUAGES)
    print("HSE_V6_CJK_MODEL_SUMMARY "+json.dumps({
        "schema": 1,
        "experiment_complete": complete,
        "unchanged_model_v5_detector": True,
        "source_same_pixels_and_auto_selected_frames": True,
        "actual_windows_onnx_inference": True,
        "real_anime_tested": False,
        "production_manifest_or_models_modified": False,
        "candidate_deployed": False,
        "release_gate_passed": False,
        "languages": list(LANGUAGES),
    }, sort_keys=True))
    return 0 if complete else 4


if __name__ == "__main__":
    raise SystemExit(main())
