"""Research-only Japanese recognizer candidate on HSE-selected subtitle crops.

Current HSE PP-OCRv5 + LightCaptionEngine first performs all detection,
geometry/polarity filtering and adaptive retry. Only the accepted subtitle-box
union is then given to the independent Apache-2.0 manga-ocr recognizer.
Ground truth is used after both outputs for scoring only. This isolates text
recognition quality and prevents manga-ocr from seeing whole scene frames.

The candidate model is the safetensors repack of kha-white/manga-ocr-base at
immutable Hugging Face revision 095f762..., with the large weight file SHA-256
verified before loading. No remote code, user media, API, or paid service.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import time

import cv2
import numpy as np
from PIL import Image

from compare_external_engine_exports import edit_distance
from hardsub_extractor.models import Frame
from hardsub_extractor.ocr.adaptive import AdaptiveOCREngine
from hardsub_extractor.ocr.light_text import LightCaptionEngine
from hardsub_extractor.ocr.rapidocr_adapter import RapidOCRAdapter
from hardsub_extractor.tab_capture import roi_for_region
from verify_geeklink_cjk_frame_groundtruth import (
    DATASET_COMMIT, MANIFEST_BLOB, MAX_IMAGE_PIXELS, SAMPLES, TOTAL_FRAMES,
    _norm, _pull_verified_images, _read_manifest,
)

HF_REPO = "mayocream/manga-ocr"
HF_REVISION = "095f76269c525275bd4add4375317788d25f6bf5"
MODEL_SHA256 = "75cbff89aa1c03330b966b24867fd656a953a7df7e8f938a7a061d2b0eb4eb67"
MODEL_FILES = (
    "config.json", "preprocessor_config.json", "special_tokens_map.json",
    "tokenizer_config.json", "vocab.txt", "model.safetensors",
)
MAX_MODEL_BYTES = 500 * 1024 * 1024


def _download_model(root: Path) -> Path:
    if root.exists() or root.is_symlink():
        raise ValueError("fresh manga OCR model scratch required")
    root.mkdir(mode=0o700)
    from huggingface_hub import hf_hub_download
    for name in MODEL_FILES:
        hf_hub_download(
            repo_id=HF_REPO, filename=name, revision=HF_REVISION,
            local_dir=str(root),
        )
    model = root / "model.safetensors"
    if not model.is_file() or not 100_000_000 < model.stat().st_size <= MAX_MODEL_BYTES:
        raise RuntimeError("manga OCR model size invalid")
    digest = hashlib.sha256()
    with model.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    if digest.hexdigest() != MODEL_SHA256:
        raise RuntimeError("manga OCR safetensors SHA-256 mismatch")
    if any(not (root/name).is_file() for name in MODEL_FILES):
        raise RuntimeError("manga OCR immutable snapshot incomplete")
    return root


def _accepted_crop(image: np.ndarray, observation) -> Image.Image | None:
    metadata = observation.metadata if isinstance(observation.metadata, dict) else {}
    rows = metadata.get("detections")
    if not observation.text or not isinstance(rows, list) or not rows:
        return None
    height, width = image.shape[:2]
    boxes = []
    for row in rows:
        if not isinstance(row, dict):
            return None
        values = [row.get(k) for k in ("cx", "cy", "width", "height")]
        if any(type(v) not in (int, float) or not np.isfinite(v) for v in values):
            return None
        cx, cy, w, h = map(float, values)
        if not (0 <= cx <= 1 and 0 <= cy <= 1 and 0 < w <= 1 and 0 < h <= 1):
            return None
        boxes.append((
            (cx-w*.5)*width, (cy-h*.5)*height,
            (cx+w*.5)*width, (cy+h*.5)*height,
        ))
    x0=min(b[0] for b in boxes); y0=min(b[1] for b in boxes)
    x1=max(b[2] for b in boxes); y1=max(b[3] for b in boxes)
    pad_x=max(8, round((x1-x0)*.08)); pad_y=max(6, round((y1-y0)*.18))
    a=max(0,int(np.floor(x0))-pad_x); b=max(0,int(np.floor(y0))-pad_y)
    c=min(width,int(np.ceil(x1))+pad_x); d=min(height,int(np.ceil(y1))+pad_y)
    if c-a < 24 or d-b < 16 or (c-a)*(d-b) > width*height*.85:
        return None
    rgb=cv2.cvtColor(image[b:d,a:c], cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb)


def run(models: Path, manifest: Path, model_root: Path) -> dict:
    if not models.is_dir() or not manifest.is_file():
        raise ValueError("pinned HSE models required")
    snapshot=_download_model(model_root)
    # No network after immutable model acquisition.
    os.environ["HF_HUB_OFFLINE"]="1"
    os.environ["TRANSFORMERS_OFFLINE"]="1"
    from manga_ocr import MangaOcr
    manga=MangaOcr(pretrained_model_name_or_path=str(snapshot), force_cpu=True)

    annotations=_read_manifest()
    images=_pull_verified_images()
    if len(images)!=TOTAL_FRAMES or TOTAL_FRAMES!=72:
        raise RuntimeError("licensed benchmark corpus incomplete")
    samples=[x for x in SAMPLES if x["lang"]=="ja"]
    if len(samples)!=24:
        raise RuntimeError("Japanese benchmark sample count changed")

    hse=LightCaptionEngine(AdaptiveOCREngine(RapidOCRAdapter(
        language="ja", model_dir=str(models), manifest_path=str(manifest),
    )))
    roi=roi_for_region("bottom40")
    hse_edits=manga_edits=chars=hse_exact=manga_exact=0
    crop_count=0
    hse_seconds=manga_seconds=0.0
    for sample in samples:
        pixels=cv2.imdecode(
            np.frombuffer(images[sample["id"]],dtype=np.uint8),cv2.IMREAD_COLOR,
        )
        if (
            pixels is None or pixels.ndim!=3 or pixels.shape[2]!=3
            or pixels.shape[0]*pixels.shape[1]>MAX_IMAGE_PIXELS
        ):
            raise ValueError("pinned Japanese image invalid")
        h0,w0=pixels.shape[:2]; x,y,w,h=roi.pixels(w0,h0)
        frame=np.ascontiguousarray(pixels[y:y+h,x:x+w])
        truth=_norm(annotations[sample["id"]]["ground_truth"])
        if not truth:
            raise RuntimeError("independent Japanese truth blank")

        began=time.perf_counter()
        observed=hse.recognize(Frame(0.0,frame.copy()))
        hse_seconds+=time.perf_counter()-began
        current=_norm(observed.text)
        hse_edits+=edit_distance(current,truth)
        hse_exact+=int(current==truth)

        crop=_accepted_crop(frame,observed)
        if crop is None:
            candidate=""
        else:
            crop_count+=1
            began=time.perf_counter()
            candidate=_norm(manga(crop))
            manga_seconds+=time.perf_counter()-began
        manga_edits+=edit_distance(candidate,truth)
        manga_exact+=int(candidate==truth)
        chars+=len(truth)

    if chars<=0 or crop_count!=24:
        raise RuntimeError("HSE did not supply all fixed Japanese subtitle crops")
    hse_cer=hse_edits/chars; manga_cer=manga_edits/chars
    candidate=bool(manga_cer < hse_cer and manga_exact >= hse_exact)
    return {
        "schema":1,
        "experiment":"HSE-selected Japanese subtitle crops: PP-OCRv5 light vs manga-ocr-base",
        "pinned_dataset_commit":DATASET_COMMIT,
        "pinned_manifest_blob_sha":MANIFEST_BLOB,
        "verified_source_images":TOTAL_FRAMES,
        "judged_japanese_images":len(samples),
        "same_HSE_detection_geometry_and_light_filter_for_crop":True,
        "candidate_never_sees_full_scene":True,
        "candidate_model_repo":HF_REPO,
        "candidate_model_revision":HF_REVISION,
        "candidate_model_safetensors_sha256":MODEL_SHA256,
        "candidate_license":"Apache-2.0",
        "candidate_model_bytes":(snapshot/"model.safetensors").stat().st_size,
        "reference_used_only_after_inference":True,
        "current_hse":{"character_edits":hse_edits,"characters":chars,
                       "cer":round(hse_cer,5),"exact":hse_exact,
                       "elapsed_seconds":round(hse_seconds,2)},
        "manga_ocr":{"character_edits":manga_edits,"characters":chars,
                     "cer":round(manga_cer,5),"exact":manga_exact,
                     "elapsed_seconds":round(manga_seconds,2),
                     "accepted_crops":crop_count},
        "promotion_candidate_in_stills_only":candidate,
        "natural_video_timing_measured":False,
        "no_caption_false_positive_measured":False,
        "production_default_modified":False,
        "paid_model_requests":0,"user_media_read":False,"release":"HOLD",
    }


def main()->int:
    p=argparse.ArgumentParser()
    p.add_argument("--model-dir",type=Path,required=True)
    p.add_argument("--manifest",type=Path,required=True)
    p.add_argument("--candidate-model-dir",type=Path,required=True)
    p.add_argument("--approve-cc-dataset-download",action="store_true")
    a=p.parse_args()
    if not a.approve_cc_dataset_download:
        print("HSE_MANGA_OCR_JA=PINNED_DATASET_DOWNLOAD_NOT_APPROVED");return 2
    try:
        result=run(a.model_dir,a.manifest,a.candidate_model_dir)
        print("HSE_MANGA_OCR_JA_RESULT "+json.dumps(result,sort_keys=True),flush=True)
        return 0
    except Exception as exc:
        print("HSE_MANGA_OCR_JA_HOLD "+json.dumps({
          "error_class":type(exc).__name__,"auto_retry":False,
          "quality_proven":False},sort_keys=True),flush=True)
        return 3

if __name__=="__main__":
    raise SystemExit(main())
