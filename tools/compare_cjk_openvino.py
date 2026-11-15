"""Compare ONNX Runtime CPU vs OpenVINO CPU on the same pinned CJK frames.

The verified PP-OCRv5 detector, recognizer, dictionary, bottom40 crop,
AdaptiveOCR retry policy, and optional light-caption filter are identical.
Only RapidOCR's inference backend changes. Ground truth is used only after
inference for CER/exact scoring. This still-image benchmark does not prove
natural-video timing or frozen-Windows performance. No user content is read.
"""
from __future__ import annotations

import argparse
import gc
from importlib import metadata
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
BACKENDS = ("onnxruntime", "openvino")
EXPECTED_OPENVINO = "2026.4.1"


def _engine(language: str, mode: str, models: Path, manifest: Path, backend: str):
    adaptive = AdaptiveOCREngine(RapidOCRAdapter(
        language=language,
        model_dir=str(models),
        manifest_path=str(manifest),
        inference_engine=backend,
    ))
    return adaptive, adaptive if mode == "baseline" else LightCaptionEngine(adaptive)


def run(models: Path, manifest: Path) -> dict:
    if not models.is_dir() or not manifest.is_file():
        raise ValueError("pinned model files and manifest required")
    if metadata.version("openvino") != EXPECTED_OPENVINO:
        raise RuntimeError("unexpected OpenVINO runtime version")

    annotations = _read_manifest()
    images = _pull_verified_images()
    if len(images) != TOTAL_FRAMES or TOTAL_FRAMES != 72:
        raise RuntimeError("licensed CJK corpus incomplete")

    roi = roi_for_region("bottom40")
    prepared = {}
    for language in LANGUAGES:
        selected = [s for s in SAMPLES if s["lang"] == language]
        if len(selected) != 24:
            raise RuntimeError("unexpected language sample count")
        rows = []
        for sample in selected:
            packed = np.frombuffer(images[sample["id"]], dtype=np.uint8)
            pixels = cv2.imdecode(packed, cv2.IMREAD_COLOR)
            if (
                pixels is None or pixels.ndim != 3 or pixels.shape[2] != 3
                or pixels.shape[0] * pixels.shape[1] > MAX_IMAGE_PIXELS
            ):
                raise ValueError("pinned image decode invalid")
            height, width = pixels.shape[:2]
            x, y, w, h = roi.pixels(width, height)
            crop = np.ascontiguousarray(pixels[y:y+h, x:x+w])
            truth = _norm(annotations[sample["id"]]["ground_truth"])
            if not truth:
                raise RuntimeError("independent truth unexpectedly blank")
            rows.append((crop, truth))
        prepared[language] = rows

    variants = []
    raw = {}
    for language in LANGUAGES:
        # Alternate order to reduce systematic warm-cache bias.
        backend_order = BACKENDS if language == "ja" else tuple(reversed(BACKENDS))
        for mode in MODES:
            for backend in backend_order:
                adaptive, engine = _engine(language, mode, models, manifest, backend)
                predictions = []
                edits = chars = exact = blank = 0
                began = time.perf_counter()
                for crop, truth in prepared[language]:
                    observed = engine.recognize(Frame(0.0, crop.copy()))
                    actual = _norm(observed.text)
                    predictions.append(actual)
                    edits += edit_distance(actual, truth)
                    chars += len(truth)
                    exact += int(actual == truth)
                    blank += int(not actual)
                elapsed = time.perf_counter() - began
                if not (1 <= adaptive.total_calls <= 72):
                    raise RuntimeError("OCR call budget exceeded")
                raw[(language, mode, backend)] = tuple(predictions)
                variants.append({
                    "language": language,
                    "mode": mode,
                    "backend": backend,
                    "images": len(predictions),
                    "characters": chars,
                    "edits": edits,
                    "cer": round(edits / chars, 5),
                    "exact": exact,
                    "blank": blank,
                    "ocr_calls": int(adaptive.total_calls),
                    "retry_calls": int(adaptive.retry_calls),
                    "elapsed_seconds": round(elapsed, 2),
                })
                del engine, adaptive
                gc.collect()

    parity = True
    for language in LANGUAGES:
        for mode in MODES:
            base = raw[(language, mode, "onnxruntime")]
            alt = raw[(language, mode, "openvino")]
            same = sum(a == b for a, b in zip(base, alt))
            onnx_row = next(
                x for x in variants if x["language"] == language
                and x["mode"] == mode and x["backend"] == "onnxruntime"
            )
            openvino_row = next(
                x for x in variants if x["language"] == language
                and x["mode"] == mode and x["backend"] == "openvino"
            )
            openvino_row["identical_text_vs_onnx"] = same
            openvino_row["full_text_parity"] = same == len(base)
            openvino_row["elapsed_saved_seconds"] = round(
                onnx_row["elapsed_seconds"] - openvino_row["elapsed_seconds"], 2
            )
            openvino_row["ocr_calls_delta"] = (
                openvino_row["ocr_calls"] - onnx_row["ocr_calls"]
            )
            parity &= (
                same == len(base)
                and openvino_row["ocr_calls"] == onnx_row["ocr_calls"]
            )

    onnx_total = sum(x["elapsed_seconds"] for x in variants if x["backend"] == "onnxruntime")
    openvino_total = sum(x["elapsed_seconds"] for x in variants if x["backend"] == "openvino")
    promotion = bool(parity and openvino_total <= onnx_total * 0.85)

    return {
        "schema": 1,
        "experiment": "same PP-OCRv5 model bits, ONNX Runtime CPU vs OpenVINO CPU",
        "rapidocr_version": metadata.version("rapidocr"),
        "onnxruntime_version": metadata.version("onnxruntime"),
        "openvino_version": metadata.version("openvino"),
        "pinned_source_commit": DATASET_COMMIT,
        "pinned_manifest_blob_sha": MANIFEST_BLOB,
        "verified_licensed_images": TOTAL_FRAMES,
        "independent_reference_frames": 48,
        "same_detector_recognizer_dictionary": True,
        "same_roi_and_adaptive_policy": True,
        "full_text_parity_all_variants": bool(parity),
        "onnx_elapsed_seconds": round(onnx_total, 2),
        "openvino_elapsed_seconds": round(openvino_total, 2),
        "openvino_speedup_ratio": round(
            onnx_total / openvino_total, 3
        ) if openvino_total else None,
        "promotion_candidate_in_stills_only": promotion,
        "natural_publisher_video_timing_measured": False,
        "frozen_windows_exe_measured": False,
        "production_default_modified": False,
        "paid_model_requests": 0,
        "user_media_read": False,
        "variants": sorted(
            variants, key=lambda x: (x["language"], x["mode"], x["backend"])
        ),
        "release": "HOLD",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--approve-cc-dataset-download", action="store_true")
    args = parser.parse_args()
    if not args.approve_cc_dataset_download:
        print("HSE_OPENVINO_CJK=PINNED_DATASET_DOWNLOAD_NOT_APPROVED")
        return 2
    try:
        result = run(args.model_dir, args.manifest)
        print("HSE_OPENVINO_CJK_RESULT " + json.dumps(result, sort_keys=True), flush=True)
        return 0
    except Exception as exc:
        print("HSE_OPENVINO_CJK_HOLD " + json.dumps({
            "error_class": type(exc).__name__,
            "auto_retry": False,
            "quality_proven": False,
        }, sort_keys=True), flush=True)
        return 3


if __name__ == "__main__":
    raise SystemExit(main())
