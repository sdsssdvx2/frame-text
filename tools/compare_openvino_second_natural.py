"""Natural publisher hard-subtitle backend comparison: ONNX vs OpenVINO.

Uses the same pinned 40-second Wikimedia publisher-authored burned-caption
video, separately published 2015 SRT reference, clean editorial control,
8fps derived inputs, English PP-OCRv5 model bits, ROI, detector and AdaptiveOCR
policy. Only RapidOCR inference backend differs. Reference text/timing is used
after inference for scoring only. No user files or paid services.
"""
from __future__ import annotations

import argparse
from hashlib import sha256
from importlib import metadata
import json
from pathlib import Path
import tempfile
import time

from compare_external_engine_exports import evaluate
from verify_commons_natural_hardsub import _hse_cues
from verify_wikimedia_second_natural_ocr import (
    PUBLIC_FILES, RESEARCH_LENGTH, get_original, make_video_slice, read_reference,
)
from hardsub_extractor.frame_source import probe_video
from hardsub_extractor.ocr.adaptive import AdaptiveOCREngine
from hardsub_extractor.ocr.rapidocr_adapter import RapidOCRAdapter

EXPECTED_OPENVINO = "2026.4.1"
BACKENDS = ("onnxruntime", "openvino")


def _digest(path: Path) -> str:
    h = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def run(model_dir: Path, manifest: Path) -> dict:
    if not model_dir.is_dir() or not manifest.is_file():
        raise ValueError("pinned English model assets required")
    if metadata.version("openvino") != EXPECTED_OPENVINO:
        raise RuntimeError("unexpected OpenVINO runtime version")
    reference, ref_sha1 = read_reference()

    with tempfile.TemporaryDirectory(prefix="hse-openvino-natural-") as td:
        root = Path(td)
        originals = {label: root / f"{label}.webm" for label in PUBLIC_FILES}
        source_sha = {
            label: get_original(originals[label], label)
            for label in PUBLIC_FILES
        }
        slices = {label: root / f"{label}.mp4" for label in PUBLIC_FILES}
        slice_sha = {
            label: make_video_slice(originals[label], slices[label])
            for label in PUBLIC_FILES
        }
        if any(abs(probe_video(str(p)).duration - RESEARCH_LENGTH) > .5
               for p in slices.values()):
            raise RuntimeError("fixed natural study slices are not synchronized")

        measured = {}
        raw_text = {}
        for backend in BACKENDS:
            positive = []
            control = []
            calls = retries = 0
            began = time.perf_counter()
            for label in ("caption", "clean"):
                adaptive = AdaptiveOCREngine(RapidOCRAdapter(
                    language="en",
                    model_dir=str(model_dir),
                    manifest_path=str(manifest),
                    inference_engine=backend,
                ))
                cues, segments = _hse_cues(
                    slices[label], root / f"{backend}-{label}", adaptive,
                )
                calls += adaptive.total_calls
                retries += adaptive.retry_calls
                if label == "caption":
                    positive = cues
                    positive_segments = segments
                else:
                    control = cues
                    control_segments = segments
            elapsed = time.perf_counter() - began
            score = evaluate(reference, positive, control)
            measured[backend] = {
                "elapsed_seconds": round(elapsed, 2),
                "local_model_calls": int(calls),
                "retry_calls": int(retries),
                "positive_segments": int(positive_segments),
                "control_segments": int(control_segments),
                "score": score,
            }
            raw_text[backend] = tuple(
                (round(c.start, 3), round(c.end, 3), c.text) for c in positive
            )

        # Exact backend parity is diagnostic only; independent reference score
        # is the release-quality criterion for this natural public source.
        same = sum(
            a == b for a, b in zip(
                raw_text["onnxruntime"], raw_text["openvino"]
            )
        )
        onnx_score = measured["onnxruntime"]["score"]
        ov_score = measured["openvino"]["score"]
        non_regression = (
            ov_score["missed_reference_cues"] <= onnx_score["missed_reference_cues"]
            and ov_score["blank_video_false_cues"] <= onnx_score["blank_video_false_cues"]
            and ov_score["exact_visible_text_and_wrap"] >= onnx_score["exact_visible_text_and_wrap"]
            and ov_score["strict_both_edges_250ms_and_text_wrap"] >=
                onnx_score["strict_both_edges_250ms_and_text_wrap"]
        )
        speedup = (
            measured["onnxruntime"]["elapsed_seconds"]
            / measured["openvino"]["elapsed_seconds"]
        )
        candidate = bool(non_regression and speedup >= 1.15)

        for label in PUBLIC_FILES:
            if _digest(originals[label]) != source_sha[label]:
                raise RuntimeError("publisher original changed during study")
            if _digest(slices[label]) != slice_sha[label]:
                raise RuntimeError("derived fixed source changed during study")

        return {
            "schema": 1,
            "source": "Wikimedia Foundation Open Letter 2013, CC BY-SA 3.0",
            "reference_revision_sha1": ref_sha1,
            "reference_cues": len(reference),
            "study_seconds": RESEARCH_LENGTH,
            "publisher_burned_hardsubs": True,
            "separately_published_reference": True,
            "reference_not_independently_frame_verified": True,
            "same_exact_derived_video_bytes_both_backends": True,
            "same_detector_recognizer_dictionary": True,
            "same_detector_roi_adaptive_policy": True,
            "source_sha256_preserved": True,
            "onnxruntime_version": metadata.version("onnxruntime"),
            "openvino_version": metadata.version("openvino"),
            "rapidocr_version": metadata.version("rapidocr"),
            "positive_exact_cue_tuple_parity_count": same,
            "onnx_positive_cues": len(raw_text["onnxruntime"]),
            "openvino_positive_cues": len(raw_text["openvino"]),
            "natural_reference_non_regression": bool(non_regression),
            "openvino_speedup_ratio": round(speedup, 3),
            "promotion_candidate_natural_english_only": candidate,
            "production_default_modified": False,
            "frozen_windows_exe_measured": False,
            "natural_cjk_video_truth_measured": False,
            "paid_model_requests": 0,
            "user_media_read": False,
            "measured": measured,
            "release": "HOLD",
        }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", type=Path, required=True)
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--approve-public-sources", action="store_true")
    args = ap.parse_args()
    if not args.approve_public_sources:
        print("HSE_OPENVINO_NATURAL=PUBLIC_SOURCE_NOT_APPROVED")
        return 2
    try:
        result = run(args.model_dir, args.manifest)
        print("HSE_OPENVINO_NATURAL_RESULT " + json.dumps(result, sort_keys=True), flush=True)
        return 0
    except Exception as exc:
        print("HSE_OPENVINO_NATURAL_HOLD " + json.dumps({
            "error_class": type(exc).__name__,
            "auto_retry": False,
            "quality_proven": False,
        }, sort_keys=True), flush=True)
        return 3


if __name__ == "__main__":
    raise SystemExit(main())
