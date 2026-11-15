"""Bounded, isolated study of small leading English glyph in compressed MP4.

Uses the same pinned English PP-OCRv5 engine and synthetic sentence as HSE
quality QA. Never processes a user video. Deliberately does NOT change
production OCR: compare detector thresholds with recognition-only crops.
"""
from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
from hardsub_extractor.frame_source import FFmpegPTSFrameSource
from hardsub_extractor.ocr.rapidocr_adapter import RapidOCRAdapter
from verify_multilang_video_pipeline import encode_sample, VIDEO_TEXT


def collect_detections(result) -> dict:
    texts = list(result.txts or ())
    scores = list(result.scores or ())
    boxes = getattr(result, "boxes", None)
    detections = []
    for i, value in enumerate(texts[:16]):
        bounds = None
        if boxes is not None and i < len(boxes):
            points = np.asarray(boxes[i], dtype=float)
            bounds = [
                round(float(points[:, 0].min()), 1),
                round(float(points[:, 1].min()), 1),
                round(float(points[:, 0].max()), 1),
                round(float(points[:, 1].max()), 1),
            ]
        detections.append({
            "text": str(value),
            "score": round(float(scores[i]), 4) if i < len(scores) else None,
            "bounds": bounds,
        })
    return {"detections": detections}


def main(argv=None) -> int:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--manifest", default=str(ROOT / "models" / "manifest.json"))
    args = parser.parse_args(argv)

    with tempfile.TemporaryDirectory(prefix="hse-english-glyph-") as td:
        video = Path(td) / "english.mp4"
        encode_sample(video, "en")
        frames = FFmpegPTSFrameSource(str(video))
        try:
            selected = None
            for frame in frames:
                if 4.60 <= frame.pts <= 4.75:
                    selected = frame
                    break
            if selected is None:
                raise RuntimeError("expected second synthetic cue frame is missing")
        finally:
            frames = None
        adapter = RapidOCRAdapter(
            language="en", model_dir=args.model_dir, manifest_path=args.manifest,
        )
        image = selected.image
        print("HSE_GLYPH_IMAGE " + json.dumps({
            "shape": list(image.shape),
            "truth": VIDEO_TEXT["en"][1],
            "pts": selected.pts,
        }, ensure_ascii=True))
        normal = adapter.engine(
            image, use_det=True, use_cls=False,
            use_rec=True, text_score=0.45,
        )
        print("HSE_GLYPH_DETECT_DEFAULT " + json.dumps(
            collect_detections(normal), ensure_ascii=True,
        ))
        # Candidate detector threshold: no production use until independently
        # tested against false positives and bounds.
        relaxed = adapter.engine(
            image, use_det=True, use_cls=False,
            use_rec=True, text_score=0.20, box_thresh=0.2,
        )
        print("HSE_GLYPH_DETECT_RELAXED " + json.dumps(
            collect_detections(relaxed), ensure_ascii=True,
        ))
        candidate_boxes = getattr(normal, "boxes", None)
        if candidate_boxes is None or len(candidate_boxes) == 0:
            raise RuntimeError("normal English OCR has no bounds for bounded crops")
        coordinates = np.concatenate(
            [np.asarray(coords, dtype=float) for coords in candidate_boxes],
            axis=0,
        )
        left, top = coordinates[:, 0].min(), coordinates[:, 1].min()
        right, bottom = coordinates[:, 0].max(), coordinates[:, 1].max()
        h, w = image.shape[:2]
        for margin in (50, 120, 220):
            x0 = max(0, int(left) - margin)
            x1 = min(w, int(right) + 36)
            y0 = max(0, int(top) - 12)
            y1 = min(h, int(bottom) + 12)
            crop = image[y0:y1, x0:x1]
            if crop.size == 0:
                raise RuntimeError("invalid bounded synthetic OCR crop")
            output = adapter.engine(
                crop, use_det=False, use_cls=False,
                use_rec=True, text_score=0.20,
            )
            print("HSE_GLYPH_REC_ONLY " + json.dumps({
                "margin_left": margin,
                "crop": [x0, y0, x1, y1],
                "txts": list(output.txts or ()),
                "scores": [
                    round(float(x), 5) for x in (output.scores or ())
                ],
            }, ensure_ascii=True))
    print("HSE_GLYPH_DIAGNOSTIC=COMPLETE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
