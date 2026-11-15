"""PP-OCRv6-medium recognizer quality check on pinned JA/ZH film frames.

Same HSE PP-OCRv5 detector, bottom40 ROI, AdaptiveOCR policy, independent
24 JA + 24 ZH frames, and optional light-caption filter as prior V5/V6-small
study. Only the recognizer switches to the exact official RapidOCR 3.10.0
PP-OCRv6-medium ONNX model. Reference text is used only after inference.
No product model/default change, no user media, no paid service.
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

from compare_external_engine_exports import edit_distance
from compare_cjk_v5_v6_recognition import V6_DICTIONARY, retrieve_verified
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

MEDIUM_SPEC = {
    "path": "onnx/PP-OCRv6/rec/PP-OCRv6_rec_medium.onnx",
    "sha256": "eef444829dbbe18d7fea59a3f6eb75647518d2b3a9568d27c92e42940204894b",
    "max_bytes": 320 * 1024 * 1024,
}
ROOT_URL = "https://www.modelscope.cn/models/RapidAI/RapidOCR/resolve/v3.10.0/"


def _retrieve_medium(root: Path) -> Path:
    output = root / Path(MEDIUM_SPEC["path"]).name
    if output.exists() or output.is_symlink():
        raise RuntimeError("medium model scratch collision")
    req = urllib.request.Request(
        ROOT_URL + MEDIUM_SPEC["path"],
        headers={"User-Agent": "HSE-pinned-V6-medium-research/1.0",
                 "Accept-Encoding": "identity"},
    )
    sha = hashlib.sha256()
    total = 0
    partial = output.with_suffix(".part")
    try:
        with urllib.request.urlopen(req, timeout=120) as response, partial.open("xb") as f:
            while True:
                block = response.read(1024 * 1024)
                if not block:
                    break
                total += len(block)
                if total > MEDIUM_SPEC["max_bytes"]:
                    raise RuntimeError("medium model exceeds hard bound")
                sha.update(block)
                f.write(block)
        if not total or sha.hexdigest() != MEDIUM_SPEC["sha256"]:
            raise RuntimeError("medium model SHA-256 mismatch")
        partial.replace(output)
    except BaseException:
        partial.unlink(missing_ok=True)
        raise
    return output


def _medium_engine(models: Path, manifest: Path, language: str, model: Path, dictionary: Path):
    from rapidocr import EngineType, LangRec, ModelType, OCRVersion, RapidOCR
    base = resolve_model_paths(
        models, manifest_path=manifest, language=language, verify=True,
    )
    lang = LangRec.JAPAN if language == "ja" else LangRec.CH
    return RapidOCR(params={
        "Det.engine_type": EngineType.ONNXRUNTIME,
        "Det.model_type": ModelType.MOBILE,
        "Det.ocr_version": OCRVersion.PPOCRV5,
        "Det.model_path": str(base.detector),
        "Rec.engine_type": EngineType.ONNXRUNTIME,
        "Rec.lang_type": lang,
        "Rec.model_type": ModelType.MEDIUM,
        "Rec.ocr_version": OCRVersion.PPOCRV6,
        "Rec.model_path": str(model),
        "Rec.rec_keys_path": str(dictionary),
        "Global.use_cls": False,
    })


def run(models: Path, manifest: Path, scratch: Path) -> dict:
    if not models.is_dir() or not manifest.is_file() or scratch.exists():
        raise ValueError("fresh scratch and pinned V5 bundle required")
    scratch.mkdir(mode=0o700)
    medium = _retrieve_medium(scratch)
    # PP-OCRv6 small/medium share the official multilingual dictionary.
    dictionary = retrieve_verified(V6_DICTIONARY, scratch)

    annotations = _read_manifest()
    image_bytes = _pull_verified_images()
    if len(image_bytes) != TOTAL_FRAMES or TOTAL_FRAMES != 72:
        raise RuntimeError("pinned film corpus incomplete")
    roi = roi_for_region("bottom40")
    variants = []

    for language in ("ja", "zh"):
        pool = [x for x in SAMPLES if x["lang"] == language]
        if len(pool) != 24:
            raise RuntimeError("language stratum changed")
        prepared = []
        for sample in pool:
            pixels = cv2.imdecode(
                np.frombuffer(image_bytes[sample["id"]], dtype=np.uint8),
                cv2.IMREAD_COLOR,
            )
            if (
                pixels is None or pixels.ndim != 3 or pixels.shape[2] != 3
                or pixels.shape[0] * pixels.shape[1] > MAX_IMAGE_PIXELS
            ):
                raise ValueError("verified frame decode invalid")
            h0, w0 = pixels.shape[:2]
            x, y, w, h = roi.pixels(w0, h0)
            prepared.append((
                np.ascontiguousarray(pixels[y:y+h, x:x+w]),
                _norm(annotations[sample["id"]]["ground_truth"]),
            ))

        v5 = RapidOCRAdapter(
            language=language, model_dir=str(models), manifest_path=str(manifest),
        )
        v6 = copy.copy(v5)
        v6.engine = _medium_engine(models, manifest, language, medium, dictionary)

        for model_name, adapter in (("v5_mobile", v5), ("v6_medium", v6)):
            for mode in ("baseline", "light"):
                adaptive = AdaptiveOCREngine(adapter)
                engine = adaptive if mode == "baseline" else LightCaptionEngine(adaptive)
                edits = chars = exact = blank = 0
                began = time.perf_counter()
                for crop, truth in prepared:
                    observed = engine.recognize(Frame(0.0, crop.copy()))
                    actual = _norm(observed.text)
                    edits += edit_distance(actual, truth)
                    chars += len(truth)
                    exact += int(actual == truth)
                    blank += int(not actual)
                variants.append({
                    "language": language,
                    "model": model_name,
                    "mode": mode,
                    "images": len(prepared),
                    "characters": chars,
                    "character_edits": edits,
                    "cer": round(edits / chars, 5),
                    "exact": exact,
                    "blank": blank,
                    "ocr_calls": int(adaptive.total_calls),
                    "retry_calls": int(adaptive.retry_calls),
                    "elapsed_seconds": round(time.perf_counter() - began, 2),
                })
                del engine, adaptive
                gc.collect()

    def row(language, model, mode):
        return next(x for x in variants if
                    x["language"] == language and x["model"] == model and x["mode"] == mode)
    gains = {}
    promoted = True
    for language in ("ja", "zh"):
        base = row(language, "v5_mobile", "light")
        candidate = row(language, "v6_medium", "light")
        gains[language] = {
            "cer_delta": round(candidate["cer"] - base["cer"], 5),
            "exact_delta": candidate["exact"] - base["exact"],
            "blank_delta": candidate["blank"] - base["blank"],
        }
        promoted &= (
            candidate["cer"] < base["cer"]
            and candidate["exact"] >= base["exact"]
            and candidate["blank"] <= base["blank"]
        )
    return {
        "schema": 1,
        "experiment": "V5 mobile vs official PP-OCRv6 medium recognizer",
        "official_medium_sha256": MEDIUM_SPEC["sha256"],
        "pinned_dataset_commit": DATASET_COMMIT,
        "pinned_manifest_blob_sha": MANIFEST_BLOB,
        "verified_source_images": TOTAL_FRAMES,
        "judged_images": 48,
        "same_v5_detector_roi_adaptive_policy": True,
        "reference_used_only_after_inference": True,
        "natural_video_timing_measured": False,
        "production_default_modified": False,
        "paid_model_requests": 0,
        "user_media_read": False,
        "promotion_candidate_in_stills_only": bool(promoted),
        "light_mode_deltas": gains,
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
        print("HSE_V6_MEDIUM=DATASET_DOWNLOAD_NOT_APPROVED")
        return 2
    try:
        result = run(args.model_dir, args.manifest, args.scratch)
        print("HSE_V6_MEDIUM_RESULT " + json.dumps(result, sort_keys=True), flush=True)
        return 0
    except Exception as exc:
        print("HSE_V6_MEDIUM_HOLD " + json.dumps({
            "error_class": type(exc).__name__,
            "auto_retry": False,
            "quality_proven": False,
        }, sort_keys=True), flush=True)
        return 3


if __name__ == "__main__":
    raise SystemExit(main())
