"""Japanese recognizer A/B: current PP-OCRv5 Chinese vs PP-OCRv4 Japanese.

The HSE PP-OCRv5 detector, exact 24 pinned Japanese film frames, bottom40 ROI,
AdaptiveOCR retry policy, and optional light-caption filter remain identical.
Only recognition model/language changes. Independent annotation is used only
after inference for scoring. The v4 Japanese model is an exact RapidOCR 3.10.0
official ONNX asset with SHA-256 pin and embedded character metadata verified
before inference. No product model/default is modified.
"""
from __future__ import annotations

import argparse
import copy
import gc
import hashlib
import json
from pathlib import Path
import time
import urllib.request

import cv2
import numpy as np
import onnxruntime as ort

from compare_external_engine_exports import edit_distance
from hardsub_extractor.model_assets import resolve_model_paths
from hardsub_extractor.models import Frame
from hardsub_extractor.ocr.adaptive import AdaptiveOCREngine
from hardsub_extractor.ocr.light_text import LightCaptionEngine
from hardsub_extractor.ocr.rapidocr_adapter import RapidOCRAdapter
from hardsub_extractor.tab_capture import roi_for_region
from verify_geeklink_cjk_frame_groundtruth import (
    DATASET_COMMIT, MANIFEST_BLOB, MAX_IMAGE_PIXELS, SAMPLES, TOTAL_FRAMES,
    _norm, _pull_verified_images, _read_manifest,
)

MODEL_URL = (
    "https://www.modelscope.cn/models/RapidAI/RapidOCR/resolve/v3.10.0/"
    "onnx/PP-OCRv4/rec/japan_PP-OCRv4_rec_mobile.onnx"
)
MODEL_SHA256 = "e1075a67dba758ecfc7ebc78a10ae61c95ac8fb66a9c86fab5541e33f085cb7a"
MAX_MODEL_BYTES = 32 * 1024 * 1024


def _download_model(path: Path) -> None:
    if path.exists() or path.is_symlink():
        raise FileExistsError("Japanese research model path exists")
    request = urllib.request.Request(
        MODEL_URL,
        headers={"User-Agent": "HSE-pinned-JA-v4-research/1.0",
                 "Accept-Encoding": "identity"},
    )
    sha = hashlib.sha256()
    total = 0
    partial = path.with_suffix(".part")
    try:
        with urllib.request.urlopen(request, timeout=90) as response, partial.open("xb") as out:
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                total += len(chunk)
                if total > MAX_MODEL_BYTES:
                    raise RuntimeError("Japanese model exceeded hard size bound")
                sha.update(chunk)
                out.write(chunk)
        if not total or sha.hexdigest() != MODEL_SHA256:
            raise RuntimeError("Japanese model SHA-256 pin mismatch")
        partial.replace(path)
    except BaseException:
        partial.unlink(missing_ok=True)
        raise

    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    meta = session.get_modelmeta().custom_metadata_map
    characters = meta.get("character", "")
    if not isinstance(characters, str) or len(characters.splitlines()) < 1000:
        raise RuntimeError("Japanese ONNX character metadata missing")


def _v4_japanese_engine(models: Path, manifest: Path, v4_rec: Path):
    from rapidocr import EngineType, LangRec, ModelType, OCRVersion, RapidOCR
    base = resolve_model_paths(
        models, manifest_path=manifest, language="ja", verify=True,
    )
    return RapidOCR(params={
        "Det.engine_type": EngineType.ONNXRUNTIME,
        "Det.model_type": ModelType.MOBILE,
        "Det.ocr_version": OCRVersion.PPOCRV5,
        "Det.model_path": str(base.detector),
        "Rec.engine_type": EngineType.ONNXRUNTIME,
        "Rec.lang_type": LangRec.JAPAN,
        "Rec.model_type": ModelType.MOBILE,
        "Rec.ocr_version": OCRVersion.PPOCRV4,
        "Rec.model_path": str(v4_rec),
        "Global.use_cls": False,
    })


def run(models: Path, manifest: Path, scratch: Path) -> dict:
    if not models.is_dir() or not manifest.is_file():
        raise ValueError("pinned HSE model bundle required")
    if scratch.exists() or scratch.is_symlink():
        raise ValueError("fresh model scratch required")
    scratch.mkdir(mode=0o700)
    v4 = scratch / "japan-v4.onnx"
    _download_model(v4)

    annotations = _read_manifest()
    images = _pull_verified_images()
    if len(images) != TOTAL_FRAMES or TOTAL_FRAMES != 72:
        raise RuntimeError("licensed CJK corpus incomplete")
    samples = [x for x in SAMPLES if x["lang"] == "ja"]
    if len(samples) != 24 or sum(bool(x["watermark"]) for x in samples) != 12:
        raise RuntimeError("Japanese benchmark strata changed")

    roi = roi_for_region("bottom40")
    prepared = []
    for sample in samples:
        pixels = cv2.imdecode(
            np.frombuffer(images[sample["id"]], dtype=np.uint8),
            cv2.IMREAD_COLOR,
        )
        if (
            pixels is None or pixels.ndim != 3 or pixels.shape[2] != 3
            or pixels.shape[0] * pixels.shape[1] > MAX_IMAGE_PIXELS
        ):
            raise ValueError("pinned Japanese image decode invalid")
        h0, w0 = pixels.shape[:2]
        x, y, w, h = roi.pixels(w0, h0)
        truth = _norm(annotations[sample["id"]]["ground_truth"])
        prepared.append((sample, np.ascontiguousarray(pixels[y:y+h, x:x+w]), truth))

    v5_adapter = RapidOCRAdapter(
        language="ja", model_dir=str(models), manifest_path=str(manifest),
    )
    v4_adapter = copy.copy(v5_adapter)
    v4_adapter.engine = _v4_japanese_engine(models, manifest, v4)

    variants = []
    raw = {}
    for name, adapter in (("v5_chinese", v5_adapter), ("v4_japanese", v4_adapter)):
        for mode in ("baseline", "light"):
            adaptive = AdaptiveOCREngine(adapter)
            engine = adaptive if mode == "baseline" else LightCaptionEngine(adaptive)
            predictions = []
            edits = chars = exact = blank = 0
            began = time.perf_counter()
            for sample, crop, truth in prepared:
                observed = engine.recognize(Frame(0.0, crop.copy()))
                actual = _norm(observed.text)
                predictions.append(actual)
                edits += edit_distance(actual, truth)
                chars += len(truth)
                exact += int(actual == truth)
                blank += int(not actual)
            if not 1 <= adaptive.total_calls <= 72:
                raise RuntimeError("Japanese OCR call budget exceeded")
            raw[(name, mode)] = tuple(predictions)
            variants.append({
                "model": name,
                "mode": mode,
                "images": len(predictions),
                "characters": chars,
                "edits": edits,
                "cer": round(edits / chars, 5),
                "exact": exact,
                "blank": blank,
                "ocr_calls": int(adaptive.total_calls),
                "retry_calls": int(adaptive.retry_calls),
                "elapsed_seconds": round(time.perf_counter() - began, 2),
            })
            del engine, adaptive
            gc.collect()

    for mode in ("baseline", "light"):
        base = raw[("v5_chinese", mode)]
        alt = raw[("v4_japanese", mode)]
        same = sum(a == b for a, b in zip(base, alt))
        row = next(x for x in variants if x["model"] == "v4_japanese" and x["mode"] == mode)
        row["identical_text_vs_v5"] = same

    v5_light = next(x for x in variants if x["model"] == "v5_chinese" and x["mode"] == "light")
    v4_light = next(x for x in variants if x["model"] == "v4_japanese" and x["mode"] == "light")
    candidate = (
        v4_light["cer"] < v5_light["cer"]
        and v4_light["exact"] >= v5_light["exact"]
        and v4_light["blank"] <= v5_light["blank"]
    )
    return {
        "schema": 1,
        "experiment": "same V5 detector, V5 Chinese recognizer vs V4 Japanese recognizer",
        "pinned_dataset_commit": DATASET_COMMIT,
        "pinned_manifest_blob_sha": MANIFEST_BLOB,
        "official_v4_japanese_model_sha256": MODEL_SHA256,
        "verified_licensed_source_images": TOTAL_FRAMES,
        "judged_japanese_images": 24,
        "watermark_images": 12,
        "same_detector_roi_adaptive_policy": True,
        "reference_used_only_after_inference": True,
        "natural_publisher_video_timing_measured": False,
        "production_default_modified": False,
        "paid_model_requests": 0,
        "user_media_read": False,
        "promotion_candidate_in_stills_only": bool(candidate),
        "variants": variants,
        "release": "HOLD",
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", type=Path, required=True)
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--scratch", type=Path, required=True)
    ap.add_argument("--approve-cc-dataset-download", action="store_true")
    args = ap.parse_args()
    if not args.approve_cc_dataset_download:
        print("HSE_JA_V4=PINNED_DATASET_DOWNLOAD_NOT_APPROVED")
        return 2
    try:
        result = run(args.model_dir, args.manifest, args.scratch)
        print("HSE_JA_V4_RESULT " + json.dumps(result, sort_keys=True), flush=True)
        return 0
    except Exception as exc:
        print("HSE_JA_V4_HOLD " + json.dumps({
            "error_class": type(exc).__name__,
            "auto_retry": False,
            "quality_proven": False,
        }, sort_keys=True), flush=True)
        return 3


if __name__ == "__main__":
    raise SystemExit(main())
