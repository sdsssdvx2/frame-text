"""PP-OCRv5 mobile vs server recognizer on the same pinned JA/ZH truth.

HSE's existing PP-OCRv5 mobile detector, exact 48 licensed film-background
caption frames, bottom40 ROI, AdaptiveOCR policy and optional light filter are
held constant. Only the official PP-OCRv5 recognition model changes from
mobile to server. Ground truth is used after inference only. No product model,
manifest, user data or operating files are modified.
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
    "onnx/PP-OCRv5/rec/ch_PP-OCRv5_rec_server.onnx"
)
MODEL_SHA256 = "e09385400eaaaef34ceff54aeb7c4f0f1fe014c27fa8b9905d4709b65746562a"
MAX_MODEL_BYTES = 128 * 1024 * 1024
LANGUAGES = ("ja", "zh")


def _download_model(path: Path) -> int:
    if path.exists() or path.is_symlink():
        raise FileExistsError("server recognizer scratch exists")
    req = urllib.request.Request(
        MODEL_URL,
        headers={"User-Agent": "HSE-pinned-v5-server-research/1.0",
                 "Accept-Encoding": "identity"},
    )
    sha = hashlib.sha256()
    count = 0
    partial = path.with_suffix(".part")
    try:
        with urllib.request.urlopen(req, timeout=120) as response, partial.open("xb") as out:
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                count += len(chunk)
                if count > MAX_MODEL_BYTES:
                    raise RuntimeError("server recognizer exceeded size bound")
                sha.update(chunk)
                out.write(chunk)
        if count <= 1_000_000 or sha.hexdigest() != MODEL_SHA256:
            raise RuntimeError("official server recognizer SHA-256 mismatch")
        partial.replace(path)
    except BaseException:
        partial.unlink(missing_ok=True)
        raise
    # RapidOCR ONNX recognition should carry its own character metadata.
    meta = ort.InferenceSession(
        str(path), providers=["CPUExecutionProvider"],
    ).get_modelmeta().custom_metadata_map
    chars = meta.get("character", "")
    if not isinstance(chars, str) or len(chars.splitlines()) < 1000:
        raise RuntimeError("server recognizer character metadata missing")
    return count


def _server_engine(models: Path, manifest: Path, language: str, server: Path):
    from rapidocr import EngineType, LangRec, ModelType, OCRVersion, RapidOCR
    base = resolve_model_paths(
        models, manifest_path=manifest, language=language, verify=True,
    )
    return RapidOCR(params={
        "Det.engine_type": EngineType.ONNXRUNTIME,
        "Det.model_type": ModelType.MOBILE,
        "Det.ocr_version": OCRVersion.PPOCRV5,
        "Det.model_path": str(base.detector),
        "Rec.engine_type": EngineType.ONNXRUNTIME,
        "Rec.lang_type": LangRec.CH,
        "Rec.model_type": ModelType.SERVER,
        "Rec.ocr_version": OCRVersion.PPOCRV5,
        "Rec.model_path": str(server),
        "Global.use_cls": False,
    })


def run(models: Path, manifest: Path, scratch: Path) -> dict:
    if not models.is_dir() or not manifest.is_file():
        raise ValueError("pinned HSE CJK model bundle required")
    if scratch.exists() or scratch.is_symlink():
        raise ValueError("fresh server-model scratch required")
    scratch.mkdir(mode=0o700)
    server_path = scratch / "v5-server.onnx"
    server_bytes = _download_model(server_path)

    annotations = _read_manifest()
    images = _pull_verified_images()
    if len(images) != TOTAL_FRAMES or TOTAL_FRAMES != 72:
        raise RuntimeError("licensed benchmark corpus incomplete")
    roi = roi_for_region("bottom40")

    prepared = {}
    for language in LANGUAGES:
        pool = [s for s in SAMPLES if s["lang"] == language]
        if len(pool) != 24 or sum(bool(s["watermark"]) for s in pool) != 12:
            raise RuntimeError("CJK benchmark strata changed")
        rows = []
        for sample in pool:
            pixels = cv2.imdecode(
                np.frombuffer(images[sample["id"]], dtype=np.uint8),
                cv2.IMREAD_COLOR,
            )
            if (
                pixels is None or pixels.ndim != 3 or pixels.shape[2] != 3
                or pixels.shape[0] * pixels.shape[1] > MAX_IMAGE_PIXELS
            ):
                raise ValueError("pinned CJK frame decode invalid")
            h0, w0 = pixels.shape[:2]
            x, y, w, h = roi.pixels(w0, h0)
            truth = _norm(annotations[sample["id"]]["ground_truth"])
            rows.append((
                np.ascontiguousarray(pixels[y:y+h, x:x+w]), truth,
            ))
        prepared[language] = rows

    variants = []
    raw = {}
    for language in LANGUAGES:
        mobile = RapidOCRAdapter(
            language=language, model_dir=str(models),
            manifest_path=str(manifest),
        )
        server = copy.copy(mobile)
        server.engine = _server_engine(models, manifest, language, server_path)
        for name, adapter in (("v5_mobile", mobile), ("v5_server", server)):
            for mode in ("baseline", "light"):
                adaptive = AdaptiveOCREngine(adapter)
                engine = adaptive if mode == "baseline" else LightCaptionEngine(adaptive)
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
                if not 1 <= adaptive.total_calls <= 72:
                    raise RuntimeError("CJK OCR call budget exceeded")
                raw[(language, name, mode)] = tuple(predictions)
                variants.append({
                    "language": language,
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

    wins = {}
    for language in LANGUAGES:
        for mode in ("baseline", "light"):
            base = raw[(language, "v5_mobile", mode)]
            alt = raw[(language, "v5_server", mode)]
            row = next(x for x in variants if
                       x["language"] == language and
                       x["model"] == "v5_server" and x["mode"] == mode)
            row["identical_text_vs_mobile"] = sum(a == b for a, b in zip(base, alt))
            mobile_row = next(x for x in variants if
                              x["language"] == language and
                              x["model"] == "v5_mobile" and x["mode"] == mode)
            wins[(language, mode)] = (
                row["cer"] < mobile_row["cer"]
                and row["exact"] >= mobile_row["exact"]
                and row["blank"] <= mobile_row["blank"]
            )

    # Baseline is the product path; light is secondary evidence only.
    candidate = all(wins[(lang, "baseline")] for lang in LANGUAGES)
    return {
        "schema": 1,
        "experiment": "same V5 detector/policy, mobile vs server V5 recognizer",
        "official_server_model_sha256": MODEL_SHA256,
        "official_server_model_bytes": server_bytes,
        "pinned_dataset_commit": DATASET_COMMIT,
        "pinned_manifest_blob_sha": MANIFEST_BLOB,
        "verified_licensed_source_images": TOTAL_FRAMES,
        "judged_images": 48,
        "same_detector_roi_adaptive_policy": True,
        "reference_used_only_after_inference": True,
        "natural_publisher_video_timing_measured": False,
        "production_default_modified": False,
        "promotion_candidate_in_stills_only": bool(candidate),
        "paid_model_requests": 0,
        "user_media_read": False,
        "variants": sorted(
            variants, key=lambda x: (x["language"], x["mode"], x["model"])
        ),
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
        print("HSE_V5_SERVER=PINNED_DATASET_DOWNLOAD_NOT_APPROVED")
        return 2
    try:
        result = run(args.model_dir, args.manifest, args.scratch)
        print("HSE_V5_SERVER_RESULT " + json.dumps(result, sort_keys=True), flush=True)
        return 0
    except Exception as exc:
        print("HSE_V5_SERVER_HOLD " + json.dumps({
            "error_class": type(exc).__name__,
            "auto_retry": False,
            "quality_proven": False,
        }, sort_keys=True), flush=True)
        return 3


if __name__ == "__main__":
    raise SystemExit(main())
