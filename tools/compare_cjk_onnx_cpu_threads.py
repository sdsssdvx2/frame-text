"""Same pinned 48 JA/ZH film frames: ONNX CPU automatic vs 4/2 threads.

Model bits, detector, recognizer, language dictionary, ROI, adaptive retries,
and light-caption policy are IDENTICAL across variants. Only intra-op CPU
thread count differs. Ground truth is used exclusively for post-inference
CER, never for deciding OCR frames or configuring model parameters.
This is a static synthetic-caption movie-background benchmark, not natural
published Japanese dialogue, long-video timing or Windows frozen EXE QA.
No user data, GPU/provider change, remote model/API, or output SRT edits.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import gc
import json
from pathlib import Path
import time

import cv2
import numpy as np

from hardsub_extractor.models import Frame
from hardsub_extractor.ocr.adaptive import AdaptiveOCREngine
from hardsub_extractor.ocr.light_text import LightCaptionEngine
from hardsub_extractor.ocr.rapidocr_adapter import RapidOCRAdapter
from hardsub_extractor.tab_capture import roi_for_region
from compare_external_engine_exports import edit_distance
from verify_geeklink_cjk_frame_groundtruth import (
    DATASET_COMMIT, MANIFEST_BLOB, SAMPLES, TOTAL_FRAMES,
    MAX_IMAGE_PIXELS, _read_manifest, _pull_verified_images, _norm,
)

LANGUAGES = ("ja", "zh")
MODES = ("baseline", "light")
THREADS = (None, 4, 2)


def run(models: Path, manifest: Path) -> dict:
    if not models.is_dir() or not manifest.is_file():
        raise ValueError("pinned models and manifest required")
    annotations = _read_manifest()
    images = _pull_verified_images()
    if len(images) != TOTAL_FRAMES or TOTAL_FRAMES != 72:
        raise RuntimeError("independent source frame corpus incomplete")
    roi = roi_for_region("bottom40")
    prepared = {}
    for language in LANGUAGES:
        samples = [sample for sample in SAMPLES if sample["lang"] == language]
        if len(samples) != 24 or sum(s["watermark"] for s in samples) != 12:
            raise RuntimeError("pinned CJK strata unexpectedly changed")
        pairs = []
        for sample in samples:
            packed = np.frombuffer(images[sample["id"]], dtype=np.uint8)
            pixels = cv2.imdecode(packed, cv2.IMREAD_COLOR)
            if (
                pixels is None or pixels.ndim != 3 or pixels.shape[2] != 3
                or pixels.shape[0] * pixels.shape[1] > MAX_IMAGE_PIXELS
            ):
                raise ValueError("CJK pinned image decode invalid")
            height, width = pixels.shape[:2]
            x, y, w, h = roi.pixels(width, height)
            crop = np.ascontiguousarray(pixels[y:y+h, x:x+w])
            truth = _norm(annotations[sample["id"]]["ground_truth"])
            if not truth:
                raise RuntimeError("empty independent truth is not measurable")
            pairs.append((sample, crop, truth))
        prepared[language] = pairs

    records = []
    raw = {}
    for language in LANGUAGES:
        for mode in MODES:
            # Reverse order for the second language, so OS warm cache
            # does not systematically favor the tuned settings.
            order = THREADS if language == "ja" else tuple(reversed(THREADS))
            for threads in order:
                adapter = RapidOCRAdapter(
                    language=language, model_dir=str(models),
                    manifest_path=str(manifest), onnx_intra_threads=threads,
                )
                adaptive = AdaptiveOCREngine(adapter)
                engine = adaptive if mode == "baseline" else LightCaptionEngine(adaptive)
                predictions = []
                edits = characters = exact = blanks = 0
                clock = time.perf_counter()
                for sample, crop, truth in prepared[language]:
                    result = engine.recognize(Frame(0.0, crop.copy()))
                    if not isinstance(result.text, str):
                        raise TypeError("invalid CJK OCR result type")
                    observed = _norm(result.text)
                    predictions.append(observed)
                    edits += edit_distance(observed, truth)
                    characters += len(truth)
                    exact += int(observed == truth)
                    blanks += int(not observed)
                seconds = time.perf_counter() - clock
                if not (1 <= adaptive.total_calls <= 3 * len(predictions)):
                    raise RuntimeError("ONNX CPU thread study exceeded OCR budget")
                key = "auto" if threads is None else str(threads)
                raw[(language, mode, key)] = tuple(predictions)
                row = {
                    "language": language,
                    "mode": mode,
                    "threads": key,
                    "images": len(predictions),
                    "characters": characters,
                    "edits": edits,
                    "cer": round(edits / characters, 5),
                    "exact": exact,
                    "blank": blanks,
                    "ocr_calls": int(adaptive.total_calls),
                    "retry_calls": int(adaptive.retry_calls),
                    "elapsed_seconds": round(seconds, 2),
                }
                records.append(row)
                del engine, adaptive, adapter
                gc.collect()

    eligible = {}
    for threads in (4, 2):
        key = str(threads)
        eligible[key] = True
        for language in LANGUAGES:
            for mode in MODES:
                expected = raw[(language, mode, "auto")]
                actual = raw[(language, mode, key)]
                same = sum(x == y for x, y in zip(expected, actual))
                selected = next(
                    x for x in records if x["language"] == language
                    and x["mode"] == mode and x["threads"] == key
                )
                baseline = next(
                    x for x in records if x["language"] == language
                    and x["mode"] == mode and x["threads"] == "auto"
                )
                selected["identical_text_vs_auto"] = same
                selected["full_text_parity"] = same == len(expected)
                selected["elapsed_saved_seconds"] = round(
                    baseline["elapsed_seconds"] - selected["elapsed_seconds"], 2,
                )
                selected["inference_calls_delta"] = (
                    selected["ocr_calls"] - baseline["ocr_calls"]
                )
                eligible[key] &= (
                    same == len(expected)
                    and selected["ocr_calls"] == baseline["ocr_calls"]
                )
        total_default = sum(x["elapsed_seconds"] for x in records if x["threads"] == "auto")
        total_tuned = sum(x["elapsed_seconds"] for x in records if x["threads"] == key)
        eligible[key] &= total_tuned <= .85 * total_default

    result = {
        "schema": 1,
        "experiment": "ORT CPU intra-op threads with exactly identical source model and crop",
        "pinned_source_commit": DATASET_COMMIT,
        "pinned_manifest_blob_sha": MANIFEST_BLOB,
        "verified_licensed_cjk_images": TOTAL_FRAMES,
        "independent_reference_frames": 48,
        "natural_publisher_subtitle_video": False,
        "video_timing_measured": False,
        "cpu_only_gpu_provider_unchanged": True,
        "all_original_local_media_untouched": True,
        "production_thread_default_modified": False,
        "actual_frozen_Windows_GUI_measured": False,
        "reference_in_inference_or_configuration": False,
        "paid_model_requests": 0,
        "no_text_or_source_frames_logged": True,
        "variants": sorted(records, key=lambda x: (
            x["language"], x["mode"], x["threads"],
        )),
        "promotion_candidate_in_stills_only": eligible,
        "release": "HOLD",
    }
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--approve-cc-dataset-download", action="store_true")
    args = parser.parse_args()
    if not args.approve_cc_dataset_download:
        print("HSE_ORT_THREADS=PINNED_DATASET_DOWNLOAD_NOT_APPROVED")
        return 2
    try:
        result = run(args.model_dir, args.manifest)
        print("HSE_ORT_THREADS_RESULT " + json.dumps(result, sort_keys=True), flush=True)
        return 0
    except Exception as exc:
        print("HSE_ORT_THREADS_HOLD " + json.dumps({
            "error_class": type(exc).__name__,
            "auto_retry": False,
            "actual_quality_proven": False,
        }, sort_keys=True), flush=True)
        return 3


if __name__ == "__main__":
    raise SystemExit(main())
