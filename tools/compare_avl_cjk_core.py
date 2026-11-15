"""Compare HSE current light OCR with pinned AVL subtitle detector/recognizer.

Research only. AVL-SubtitleOCR is Apache-2.0 and installed from one immutable
Git commit in CI with --no-deps so HSE's pinned ONNX Runtime is not replaced.
Both candidates see the identical bottom40 pixels and use HSE's current
subtitle-box geometry/order filter; the independent text reference is used only
after both engines have returned text. No user media, account, API, or paid
service is used.
"""
from __future__ import annotations

import argparse
from importlib import metadata
import json
from pathlib import Path
import time

import cv2
import numpy as np

from compare_external_engine_exports import edit_distance
from hardsub_extractor.models import Frame
from hardsub_extractor.ocr.adaptive import AdaptiveOCREngine
from hardsub_extractor.ocr.light_text import LightCaptionEngine
from hardsub_extractor.ocr.rapidocr_adapter import (
    RapidOCRAdapter, _join_visual_line, _subtitle_box_lines,
)
from hardsub_extractor.tab_capture import roi_for_region
from verify_geeklink_cjk_frame_groundtruth import (
    DATASET_COMMIT, MANIFEST_BLOB, MAX_IMAGE_PIXELS, SAMPLES, TOTAL_FRAMES,
    _norm, _pull_verified_images, _read_manifest,
)

AVL_COMMIT = "ee0269df692719b658ccae0e75106e9b84af40ec"
AVL_PACKAGE_VERSION = "0.1.0"
AVL_MODEL_FILES = {
    "res18-db-damo_mixed.onnx": (24_375_935, "40d48be015e9fcc85e85786f50999cd6321c2a8c"),
    "paddle_v6_rec_small.onnx": (21_159_378, "7c052944eb4f25cd3a38e4dbef8794430dc40f6b"),
    "paddle_v6_dict.txt": (74_947, "270082aa29aec1ff3ac74d0d575c90399216f74d"),
}
LANGUAGES = ("ja", "zh")


def _avl_text(backend, crop: np.ndarray, language: str) -> str:
    detections = list(backend.detect_and_recognize(crop.copy()))
    if not detections:
        return ""
    txts = [str(x.text) for x in detections]
    boxes = []
    for item in detections:
        if item.polygon:
            boxes.append(item.polygon)
        else:
            x1, y1, x2, y2 = item.bbox
            boxes.append(((x1, y1), (x2, y1), (x2, y2), (x1, y2)))
    lines = _subtitle_box_lines(
        txts, boxes, crop.shape, require_hangul=False, language=language,
    )
    return "\n".join(
        _join_visual_line(txts, boxes, indices, language=language)
        for indices in lines
        if indices
    ).strip()


def _verify_avl_package() -> dict:
    import avl_open_ocr
    root = Path(avl_open_ocr.__file__).resolve().parent / "model_files"
    if metadata.version("avl-open-ocr") != AVL_PACKAGE_VERSION:
        raise RuntimeError("unexpected AVL package version")
    import hashlib
    measured = {}
    for name, (expected_size, expected_blob) in AVL_MODEL_FILES.items():
        path = root / name
        if not path.is_file() or path.is_symlink():
            raise RuntimeError("pinned AVL model package mismatch")
        raw = path.read_bytes()
        canonical = raw.replace(b"\r\n", b"\n") if name.endswith(".txt") else raw
        header = b"blob " + str(len(canonical)).encode("ascii") + b"\0"
        blob = hashlib.sha1(header + canonical).hexdigest()
        if len(canonical) != expected_size or blob != expected_blob:
            raise RuntimeError("pinned AVL model package mismatch")
        measured[name] = {"bytes": len(canonical), "git_blob_sha1": blob}
    return measured


def run(models: Path, manifest: Path) -> dict:
    if not models.is_dir() or not manifest.is_file():
        raise ValueError("pinned HSE model bundle required")
    print("HSE_AVL_STAGE=VERIFY_PACKAGE_START", flush=True)
    model_sizes = _verify_avl_package()
    print("HSE_AVL_STAGE=VERIFY_PACKAGE_PASS", flush=True)
    from avl_open_ocr import create_subtle_backend

    print("HSE_AVL_STAGE=PINNED_DATA_START", flush=True)
    annotations = _read_manifest()
    images = _pull_verified_images()
    if len(images) != TOTAL_FRAMES or TOTAL_FRAMES != 72:
        raise RuntimeError("licensed subtitle-frame corpus incomplete")
    print("HSE_AVL_STAGE=PINNED_DATA_PASS", flush=True)
    roi = roi_for_region("bottom40")

    variants = []
    raw = {}
    for language in LANGUAGES:
        selected = [s for s in SAMPLES if s["lang"] == language]
        if len(selected) != 24:
            raise RuntimeError("CJK benchmark stratum changed")
        prepared = []
        for sample in selected:
            packed = np.frombuffer(images[sample["id"]], dtype=np.uint8)
            pixels = cv2.imdecode(packed, cv2.IMREAD_COLOR)
            if (
                pixels is None or pixels.ndim != 3 or pixels.shape[2] != 3
                or pixels.shape[0] * pixels.shape[1] > MAX_IMAGE_PIXELS
            ):
                raise ValueError("pinned frame decode invalid")
            h0, w0 = pixels.shape[:2]
            x, y, w, h = roi.pixels(w0, h0)
            crop = np.ascontiguousarray(pixels[y:y+h, x:x+w])
            truth = _norm(annotations[sample["id"]]["ground_truth"])
            prepared.append((crop, truth))

        print("HSE_AVL_STAGE=LANG_INIT_"+language, flush=True)
        current = LightCaptionEngine(AdaptiveOCREngine(RapidOCRAdapter(
            language=language, model_dir=str(models),
            manifest_path=str(manifest),
        )))
        avl = create_subtle_backend(
            language="japan" if language == "ja" else "ch",
            device="cpu", recognizer="v6_small",
        )
        print("HSE_AVL_STAGE=LANG_READY_"+language, flush=True)
        for name in ("hse_light", "avl_subtle_v6"):
            print("HSE_AVL_STAGE=RUN_"+language+"_"+name, flush=True)
            edits = chars = exact = blank = 0
            predictions = []
            began = time.perf_counter()
            for crop, truth in prepared:
                if name == "hse_light":
                    text = current.recognize(Frame(0.0, crop.copy())).text
                else:
                    text = _avl_text(avl, crop, language)
                actual = _norm(text)
                predictions.append(actual)
                edits += edit_distance(actual, truth)
                chars += len(truth)
                exact += int(actual == truth)
                blank += int(not actual)
            row = {
                "language": language,
                "engine": name,
                "images": len(prepared),
                "characters": chars,
                "character_edits": edits,
                "cer": round(edits / chars, 5),
                "exact": exact,
                "blank": blank,
                "elapsed_seconds": round(time.perf_counter() - began, 2),
            }
            variants.append(row)
            raw[(language, name)] = tuple(predictions)

    candidate = True
    for language in LANGUAGES:
        hse = next(x for x in variants if x["language"] == language
                   and x["engine"] == "hse_light")
        avl = next(x for x in variants if x["language"] == language
                   and x["engine"] == "avl_subtle_v6")
        same = sum(
            a == b for a, b in zip(
                raw[(language, "hse_light")],
                raw[(language, "avl_subtle_v6")],
            )
        )
        avl["identical_text_vs_hse"] = same
        avl["cer_delta_vs_hse"] = round(avl["cer"] - hse["cer"], 5)
        avl["exact_delta_vs_hse"] = avl["exact"] - hse["exact"]
        candidate &= (
            avl["cer"] < hse["cer"]
            and avl["exact"] >= hse["exact"]
            and avl["blank"] <= hse["blank"]
        )

    return {
        "schema": 1,
        "experiment": "same bottom40 pixels and HSE geometry: HSE light vs AVL DamoDBNet+v6-small",
        "avl_repository": "allvoicelab/AVL-SubtitleOCR",
        "avl_commit": AVL_COMMIT,
        "avl_license": "Apache-2.0",
        "avl_package_version": metadata.version("avl-open-ocr"),
        "avl_bundled_model_sizes": model_sizes,
        "pinned_dataset_commit": DATASET_COMMIT,
        "pinned_manifest_blob_sha": MANIFEST_BLOB,
        "verified_source_images": TOTAL_FRAMES,
        "judged_images": 48,
        "same_bottom40_pixels": True,
        "same_hse_geometry_order_filter": True,
        "reference_used_only_after_inference": True,
        "avl_directml_used": False,
        "natural_video_timing_measured": False,
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
    ap.add_argument("--approve-cc-dataset-download", action="store_true")
    args = ap.parse_args()
    if not args.approve_cc_dataset_download:
        print("HSE_AVL_CJK=PINNED_DATASET_DOWNLOAD_NOT_APPROVED")
        return 2
    try:
        result = run(args.model_dir, args.manifest)
        print("HSE_AVL_CJK_RESULT " + json.dumps(result, sort_keys=True), flush=True)
        return 0
    except Exception as exc:
        message = str(exc)
        if "AVL model package mismatch" in message:
            category = "AVL_MODEL_PACKAGE_MISMATCH"
        elif "unexpected AVL package version" in message:
            category = "AVL_PACKAGE_VERSION_MISMATCH"
        elif "licensed subtitle-frame corpus incomplete" in message:
            category = "PINNED_CORPUS_INCOMPLETE"
        elif "CJK benchmark stratum changed" in message:
            category = "PINNED_STRATUM_CHANGED"
        else:
            category = "UNCLASSIFIED_RUNTIME"
        print("HSE_AVL_CJK_HOLD " + json.dumps({
            "error_class": type(exc).__name__,
            "error_category": category,
            "auto_retry": False,
            "quality_proven": False,
        }, sort_keys=True), flush=True)
        return 3


if __name__ == "__main__":
    raise SystemExit(main())
