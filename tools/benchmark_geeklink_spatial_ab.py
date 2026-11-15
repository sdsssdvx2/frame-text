"""Pinned public-film subtitle crop A/B: aggregate only, no product changes.

Compares current HSE lower-frame ROI against two *precommitted* narrower
windows on exactly the same 24 Git-SHA-verified image samples. Captions
are constructed on real movie scenes; this neither proves natural anime OCR
nor authorizes automatic crop selection based on a ground-truth label.
No raw images, predictions, reference text, or source paths are persisted.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from benchmark_geeklink_ocr import (
    BenchmarkError,
    LANGUAGES,
    _blob,
    _decode,
    _distance,
    _score,
    source_preflight,
)

# Freeze variants before looking at outcomes. One production control plus
# two geometry-only alternatives; do not use reference text to choose crops.
ROI_CANDIDATES = {
    "production_roi": (0.04, 0.58, 0.92, 0.40),
    "center_bottom_32": (0.08, 0.66, 0.84, 0.32),
    "center_bottom_26": (0.12, 0.72, 0.76, 0.26),
}


def _crop(image, coordinates):
    from hardsub_extractor.frame_source import NormalizedROI

    if image.ndim != 3 or image.shape[2] != 3:
        raise BenchmarkError("invalid_image_shape")
    roi = NormalizedROI(*coordinates)
    x, y, width, height = roi.pixels(image.shape[1], image.shape[0])
    if min(width, height) <= 0 or x + width > image.shape[1] or y + height > image.shape[0]:
        raise BenchmarkError("invalid_research_crop")
    return image[y:y + height, x:x + width].copy()


def _group_summary(rows, predictions):
    """Compute CER, exactness and improvement/harm WITHOUT leaking text."""
    required = set(ROI_CANDIDATES)
    if set(predictions) != required or len(rows) != 24:
        raise BenchmarkError("comparison_incomplete")
    ids = [r["id"] for r in rows]
    if len(set(ids)) != len(ids) or any(set(predictions[k]) != set(ids) for k in required):
        raise BenchmarkError("comparison_id_mismatch")
    scores = {}
    for name in ROI_CANDIDATES:
        compared = [(r["lang"], r["has_watermark"],
                     predictions[name][r["id"]], r["ground_truth"]) for r in rows]
        scores[name] = _score(compared)
        scores[name]["exact_image_count"] = sum(
            predictions[name][r["id"]] == r["ground_truth"] for r in rows
        )

    effects = {}
    for candidate in ROI_CANDIDATES:
        if candidate == "production_roi":
            continue
        parts = {}
        for group in ("all", "clean", "watermark"):
            subset = [r for r in rows if (
                group == "all" or r["has_watermark"] == (
                    "True" if group == "watermark" else "False"
                )
            )]
            counts = {"improved": 0, "worsened": 0, "unchanged": 0, "n": len(subset)}
            for r in subset:
                sid, truth = r["id"], r["ground_truth"]
                base = _distance(predictions["production_roi"][sid], truth)
                proposed = _distance(predictions[candidate][sid], truth)
                key = "improved" if proposed < base else "worsened" if proposed > base else "unchanged"
                counts[key] += 1
            parts[group] = counts
        effects[candidate] = parts
    return scores, effects


def measure(model_dir: Path, manifest: Path) -> dict:
    from hardsub_extractor.models import Frame
    from hardsub_extractor.ocr.rapidocr_adapter import RapidOCRAdapter

    paths, source, selected, _baseline = source_preflight()
    if len(selected) != 24:
        raise BenchmarkError("sample_count_drift")
    output = {name: {} for name in ROI_CANDIDATES}
    inference_counts = {language: 0 for language in LANGUAGES}
    # Model init per language, one decoded file in memory per sample.
    for lang in LANGUAGES:
        engine = RapidOCRAdapter(
            model_dir=str(model_dir), manifest_path=str(manifest), language=lang,
        )
        for r in selected:
            if r["lang"] != lang:
                continue
            raw = _blob(r["image"], paths, 250_000)
            image = _decode(raw)
            for name, window in ROI_CANDIDATES.items():
                obs = engine.recognize(Frame(0.0, _crop(image, window)))
                if not isinstance(obs.text, str):
                    raise BenchmarkError("unexpected_model_text")
                output[name][r["id"]] = obs.text
                inference_counts[lang] += 1
    scores, effects = _group_summary(selected, output)
    if any(n != 18 for n in inference_counts.values()):
        raise BenchmarkError("model_inference_count_drift")
    return {
        "schema": 1,
        "study": "independent_pinned_public_film_spatial_roi_ab",
        "dataset_commit": "baa9ccbb7ffad36d56cc0ea115256a09bdf427f4",
        "sample_count": 24,
        "per_language": 6,
        "watermark_samples": 12,
        "filmed_background_constructed_subtitles": True,
        "natural_subtitles_or_video_timing_tested": False,
        "no_model_or_release_change": True,
        "quality_improvement_verified": False,
        "captions_images_and_paths_not_reported": True,
        "no_image_redistribution": True,
        "paid_calls": 0,
        "variants": {k: list(v) for k, v in ROI_CANDIDATES.items()},
        "inference_calls_by_language": inference_counts,
        "cer": scores,
        "pairwise_errors_vs_production": effects,
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", type=Path, required=True)
    ap.add_argument("--model-manifest", type=Path, required=True)
    ap.add_argument("--report", type=Path, required=True)
    ap.add_argument("--accept-public-download", action="store_true")
    args = ap.parse_args(argv)
    if not args.accept_public_download:
        print("HSE_SPATIAL_AB=PUBLIC_DOWNLOAD_NOT_APPROVED")
        return 2
    try:
        outcome = measure(args.model_dir, args.model_manifest)
        with args.report.open("x", encoding="utf-8") as stream:
            json.dump(outcome, stream, sort_keys=True, ensure_ascii=False, indent=2)
            stream.write("\n")
    except Exception as exc:
        # Never print raw subtitle content or local file paths.
        print("HSE_SPATIAL_AB=FAILED;ERROR_CLASS=" + type(exc).__name__, file=sys.stderr)
        return 3
    print("HSE_SPATIAL_AB=COMPLETE;SAMPLES=24;ROI_METHODS=3;RELEASE_CHANGE=NONE")
    for name, bucket in outcome["cer"].items():
        print(f"HSE_SPATIAL_ROI={name};ALL={bucket['all']['cer']:.4f};"
              f"CLEAN={bucket['clean']['cer']:.4f};"
              f"WATERMARK={bucket['watermark']['cer']:.4f};"
              f"EXACT_IMAGES={bucket['exact_image_count']}")
    for name, groups in outcome["pairwise_errors_vs_production"].items():
        for group, effect in groups.items():
            print(f"HSE_SPATIAL_HARM={name};GROUP={group};"
                  f"IMPROVED={effect['improved']};WORSENED={effect['worsened']};"
                  f"UNCHANGED={effect['unchanged']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
