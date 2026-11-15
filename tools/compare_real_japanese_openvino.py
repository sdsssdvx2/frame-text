"""Real publisher-authored Japanese OGV: ONNX Runtime vs OpenVINO CPU.

Uses the exact same SHA-pinned original, the same lossless 60-second slice,
the same PP-OCRv5 models, detector, ROI, AdaptiveOCR policy and frame stream.
Only RapidOCR's inference backend changes. There is no independent transcript
for this source, so this measures backend parity and speed, NOT OCR accuracy.
No user media, paid APIs, screenshots or extracted text leave the scratch.
"""
from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
import time

from hardsub_extractor.corrected_quality import compare_corrected_cues
from hardsub_extractor.ocr.adaptive import AdaptiveOCREngine
from hardsub_extractor.ocr.rapidocr_adapter import RapidOCRAdapter
from hardsub_extractor.pipeline import extract
from hardsub_extractor.subtitle_formats import parse_subtitle_bytes
from compare_real_japanese_fps import (
    _cues, _download_original, _fixed_lossless_clip,
)

BACKENDS = ("onnxruntime", "openvino")


def _sha(path: Path) -> str:
    h = sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _trial(root: Path, clip: Path, models: Path, backend: str):
    adapter = RapidOCRAdapter(
        language="ja",
        model_dir=str(models),
        inference_engine=backend,
    )
    adaptive = AdaptiveOCREngine(adapter)
    out = root / backend
    began = time.perf_counter()
    report = extract(str(clip), adaptive, out, streaming=True)
    elapsed = time.perf_counter() - began
    srt = _cues(out.with_suffix(".srt"))
    kind, ass = parse_subtitle_bytes(
        out.with_suffix(".ass").read_bytes(),
        source_url=out.with_suffix(".ass").name,
    )
    if (
        kind != "ass" or len(srt) != len(ass)
        or any(
            a.text != b.text
            or abs(a.start - b.start) > .011
            or abs(a.end - b.end) > .011
            for a, b in zip(srt, ass)
        )
    ):
        raise RuntimeError("backend SRT/ASS mismatch")
    return {
        "backend": backend,
        "elapsed_seconds": round(elapsed, 2),
        "cues": len(srt),
        "segments": int(report.segment_count),
        "model_calls": int(adaptive.total_calls),
        "retry_calls": int(adaptive.retry_calls),
    }, srt


def run(root: Path, models: Path) -> dict:
    if root.exists() or root.is_symlink() or not models.is_dir():
        raise ValueError("fresh scratch and pinned models required")
    root.mkdir(mode=0o700)
    original = root / "source.ogv"
    clip = root / "same-pixels.mkv"
    source_sha = _download_original(original)
    _fixed_lossless_clip(original, clip)
    clip_sha = _sha(clip)

    measured = {}
    cues = {}
    # ONNX first deliberately matches established product baseline.
    for backend in BACKENDS:
        row, output = _trial(root, clip, models, backend)
        measured[backend] = row
        cues[backend] = output
        if _sha(original) != source_sha or _sha(clip) != clip_sha:
            raise RuntimeError("source bytes changed during backend comparison")
        print("HSE_JA_OPENVINO_CASE " + json.dumps(row, sort_keys=True), flush=True)

    diff = compare_corrected_cues(cues["onnxruntime"], cues["openvino"])
    onnx = measured["onnxruntime"]
    ov = measured["openvino"]
    exact_tuple_parity = (
        len(cues["onnxruntime"]) == len(cues["openvino"])
        and diff.baseline_only == 0
        and diff.corrected_only == 0
        and diff.exact_visible_cues == len(cues["onnxruntime"])
        and diff.strict_joint_cues == len(cues["onnxruntime"])
    )
    speedup = onnx["elapsed_seconds"] / ov["elapsed_seconds"]
    return {
        "schema": 1,
        "source": "Wikimedia Commons Wanna Work Together Japanese subtitled OGV",
        "publisher_authored_natural_japanese": True,
        "same_exact_lossless_clip_bytes": True,
        "source_sha256_preserved": True,
        "same_model_detector_roi_adaptive_policy": True,
        "independent_transcript_available": False,
        "ocr_accuracy_verified": False,
        "onnx_cues": onnx["cues"],
        "openvino_cues": ov["cues"],
        "baseline_only_cues": int(diff.baseline_only),
        "openvino_only_cues": int(diff.corrected_only),
        "exact_visible_text_cues": int(diff.exact_visible_cues),
        "strict_text_and_250ms_cues": int(diff.strict_joint_cues),
        "exact_backend_tuple_parity": bool(exact_tuple_parity),
        "openvino_speedup_ratio": round(speedup, 3),
        "promotion_candidate_for_backend_parity": bool(
            exact_tuple_parity and speedup >= 1.15
        ),
        "production_default_modified": False,
        "paid_model_requests": 0,
        "user_media_read": False,
        "measured": measured,
        "release": "HOLD",
    }


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--scratch", type=Path, required=True)
    p.add_argument("--model-dir", type=Path, required=True)
    p.add_argument("--approve-cc-by-download", action="store_true")
    a = p.parse_args()
    if not a.approve_cc_by_download:
        print("HSE_JA_OPENVINO=SOURCE_DOWNLOAD_NOT_APPROVED")
        return 2
    try:
        result = run(a.scratch, a.model_dir)
        print("HSE_JA_OPENVINO_RESULT " + json.dumps(result, sort_keys=True), flush=True)
        return 0
    except Exception as exc:
        print("HSE_JA_OPENVINO_HOLD " + json.dumps({
            "error_class": type(exc).__name__,
            "auto_retry": False,
            "backend_parity_proven": False,
        }, sort_keys=True), flush=True)
        return 3


if __name__ == "__main__":
    raise SystemExit(main())
