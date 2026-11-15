"""Bounded, pinned, real-film-background subtitle OCR generalization study.

Uses an upstream *published* open-image benchmark, not real-anime videos.
The source text is original and annotated, film frames have mixed licenses;
never redistribute images or transcriptions. Pinned upstream Git commit is
verified separately by the CI workflow. No network requests in this script.
Compare actual HSE OCR observations against publisher-provided baseline CSVs:
their OCR engines are NOT executed here and timing is not comparable.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import re
import time
import unicodedata

from compare_external_engine_exports import edit_distance

DATASET_COMMIT = "baa9ccbb7ffad36d56cc0ea115256a09bdf427f4"
LANGUAGES = ("ko", "en", "ja", "zh")
EXTERNAL_BASELINES = {
    "published_geeklink_onnx": "baselines/geeklink.csv",
    "published_paddleocrv6_small": "external_baselines/paddleocr_small.csv",
}
MAX_MANIFEST_BYTES = 512 * 1024
MAX_BASELINE_BYTES = 384 * 1024
MAX_IMAGE_BYTES = 5 * 1024 * 1024
MAX_IMAGE_PIXELS = 6_000_000
EXPECTED_MANIFEST_ROWS = 1140
EXPECTED_FOUR_LANGUAGE_ROWS = 761
IMAGE_PATTERN = re.compile(r"^data/images/[A-Za-z0-9_-]+[.]jpe?g$")
ID_PATTERN = re.compile(r"^[A-Za-z0-9_-]{1,64}$")


def safe_input(root: Path, relative: str, maximum: int, *, image: bool = False) -> Path:
    if (
        not isinstance(relative, str)
        or (image and not IMAGE_PATTERN.fullmatch(relative))
        or not relative or "\\" in relative or ":" in relative
        or ".." in relative.split("/")
        or relative.startswith("/")
    ):
        raise ValueError("unsafe public benchmark relative input")
    path = root.joinpath(*relative.split("/"))
    if path.is_symlink() or any(
        part.is_symlink() for part in path.parents if part != root and part != root.parent
    ):
        raise ValueError("symlinked benchmark input")
    resolved = path.resolve(strict=True)
    if not resolved.is_relative_to(root) or not resolved.is_file():
        raise ValueError("benchmark input outside pinned root")
    if not 0 < resolved.stat().st_size <= maximum:
        raise ValueError("oversized or empty benchmark file")
    return resolved


def load_csv(root: Path, relative: str, max_bytes: int) -> list[dict]:
    path = safe_input(root, relative, max_bytes)
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def read_reference(root: Path) -> list[dict]:
    rows = load_csv(root, "data/manifest.csv", MAX_MANIFEST_BYTES)
    if len(rows) != EXPECTED_MANIFEST_ROWS:
        raise ValueError("unexpected benchmark manifest length")
    ids = set()
    for row in rows:
        if (
            not isinstance(row, dict)
            or not ID_PATTERN.fullmatch(row.get("id") or "")
            or row["id"] in ids
            or row.get("lang") not in {"en", "es", "ja", "ko", "zh", "el"}
            or row.get("has_watermark") not in {"True", "False"}
            or not IMAGE_PATTERN.fullmatch(row.get("image") or "")
            or not isinstance(row.get("ground_truth"), str)
            or not row["ground_truth"]
        ):
            raise ValueError("malformed or duplicate reference row")
        ids.add(row["id"])
    return rows


def select_stratified(rows: list[dict], per_group: int) -> list[dict]:
    if isinstance(per_group, bool) or not isinstance(per_group, int) or not 1 <= per_group <= 20:
        raise ValueError("bounded samples per language/condition required")
    selected = []
    for lang in LANGUAGES:
        for watermark in ("False", "True"):
            bucket = [
                r for r in rows
                if r.get("lang") == lang and r.get("has_watermark") == watermark
            ]
            # Deterministic, opaque stable ordering; never handpick easy samples.
            bucket.sort(key=lambda r: hashlib.sha256(
                (DATASET_COMMIT + ":" + r["id"]).encode("ascii"),
            ).digest())
            if len(bucket) < per_group:
                raise ValueError("insufficient independently published samples")
            selected.extend(bucket[:per_group])
    return selected


def select_all_four_languages(rows: list[dict]) -> list[dict]:
    """Precommitted full population: no choosing frames after seeing predictions."""
    selection = [row for row in rows if row["lang"] in LANGUAGES]
    if len(selection) != EXPECTED_FOUR_LANGUAGE_ROWS:
        raise ValueError("unexpected size of independently published cohort")
    for lang in LANGUAGES:
        subset = [r for r in selection if r["lang"] == lang]
        if (
            len(subset) < 180
            or sum(r["has_watermark"] == "True" for r in subset) < 30
            or sum(r["has_watermark"] == "False" for r in subset) < 100
        ):
            raise ValueError("public benchmark subgroup unexpectedly incomplete")
    return selection


def load_published_predictions(root: Path, selected: list[dict]) -> dict[str, dict[str, str]]:
    ids = {r["id"] for r in selected}
    outputs = {}
    for name, rel in EXTERNAL_BASELINES.items():
        rows = load_csv(root, rel, MAX_BASELINE_BYTES)
        predictions = {}
        for row in rows:
            sid = row.get("id")
            pred = row.get("prediction")
            if not isinstance(sid, str) or not isinstance(pred, str) or sid in predictions:
                raise ValueError("invalid or duplicate published prediction")
            predictions[sid] = pred
        if not ids.issubset(predictions):
            raise ValueError("published baseline lacks selected samples")
        outputs[name] = predictions
    return outputs


def char_distance(reference: str, prediction: str) -> tuple[int, int]:
    # Retain punctuation, spacing, line breaks and exact script;
    # do not flatten out an OCR engine's inconvenient formatting.
    a = unicodedata.normalize("NFC", reference)
    b = unicodedata.normalize("NFC", prediction)
    return edit_distance(a, b), max(1, len(a))


def summarize(selected: list[dict], predictions: dict[str, dict[str, str]]) -> list[dict]:
    result = []
    for lang in LANGUAGES:
        for watermark in ("False", "True"):
            subset = [
                r for r in selected
                if r["lang"] == lang and r["has_watermark"] == watermark
            ]
            scores = {}
            for engine, values in predictions.items():
                errors = total = zero = empty = 0
                for row in subset:
                    dist, length = char_distance(
                        row["ground_truth"], values[row["id"]],
                    )
                    errors += dist
                    total += length
                    zero += dist == 0
                    empty += not values[row["id"]].strip()
                scores[engine] = {
                    "cer": round(errors / total, 5),
                    "exact_samples": zero,
                    "empty_predictions": empty,
                }
            # Paired per-image comparison of the SAME actual model call.
            # Both paths see exactly identical decoded images, which makes
            # the effect of HSE's subtitle selection inspectable without
            # exposing any recognized dialogue or individual image IDs.
            paired = None
            if {"hse_real_ocr", "same_inference_raw_ocr"}.issubset(predictions):
                filtered_better = raw_better = tied = 0
                filtered_empty_raw_nonempty = raw_empty_filtered_nonempty = 0
                total_filtered_errors = total_raw_errors = 0
                for row in subset:
                    fid = row["id"]
                    ftext = predictions["hse_real_ocr"][fid]
                    rtext = predictions["same_inference_raw_ocr"][fid]
                    ferr, _ = char_distance(row["ground_truth"], ftext)
                    rerr, _ = char_distance(row["ground_truth"], rtext)
                    total_filtered_errors += ferr
                    total_raw_errors += rerr
                    filtered_better += ferr < rerr
                    raw_better += rerr < ferr
                    tied += ferr == rerr
                    filtered_empty_raw_nonempty += not ftext.strip() and bool(rtext.strip())
                    raw_empty_filtered_nonempty += not rtext.strip() and bool(ftext.strip())
                if filtered_better + raw_better + tied != len(subset):
                    raise RuntimeError("incomplete paired OCR audit")
                paired = {
                    "filtered_better_samples": filtered_better,
                    "raw_better_samples": raw_better,
                    "equal_error_samples": tied,
                    "filtered_empty_while_raw_nonempty": filtered_empty_raw_nonempty,
                    "raw_empty_while_filtered_nonempty": raw_empty_filtered_nonempty,
                    "total_character_edit_errors_filtered": total_filtered_errors,
                    "total_character_edit_errors_raw": total_raw_errors,
                    "identical_model_first_call_comparison": True,
                    "line_break_and_punctuation_sensitive": True,
                }
            coherent_pair = None
            if {"hse_real_ocr", "hse_caption_islands_research"}.issubset(predictions):
                original_better = candidate_better = tied = dropped_rows = 0
                original_errors = candidate_errors = 0
                for row in subset:
                    ident = row["id"]
                    before = predictions["hse_real_ocr"][ident]
                    after = predictions["hse_caption_islands_research"][ident]
                    old_err, _ = char_distance(row["ground_truth"], before)
                    new_err, _ = char_distance(row["ground_truth"], after)
                    original_errors += old_err
                    candidate_errors += new_err
                    original_better += old_err < new_err
                    candidate_better += new_err < old_err
                    tied += old_err == new_err
                    dropped_rows += max(0, len(before.splitlines()) - len(after.splitlines()))
                coherent_pair = {
                    "candidate_improved_images": candidate_better,
                    "candidate_worsened_images": original_better,
                    "equal_character_edit_images": tied,
                    "original_total_character_edits": original_errors,
                    "candidate_total_character_edits": candidate_errors,
                    "rows_removed_by_candidate": dropped_rows,
                    "same_first_model_inference": True,
                    "may_preserve_three_or_more_coherent_rows": True,
                    "not_video_timing_or_false_cue_proof": True,
                }
            row_selector_pair = None
            if {"hse_real_ocr", "hse_bottom_two_research"}.issubset(predictions):
                original_better = candidate_better = tied = 0
                original_errors = candidate_errors = dropped_rows = 0
                for row in subset:
                    fid = row["id"]
                    original = predictions["hse_real_ocr"][fid]
                    candidate = predictions["hse_bottom_two_research"][fid]
                    old_error, _ = char_distance(row["ground_truth"], original)
                    new_error, _ = char_distance(row["ground_truth"], candidate)
                    original_errors += old_error
                    candidate_errors += new_error
                    dropped_rows += max(0, len(original.splitlines()) - 2)
                    original_better += old_error < new_error
                    candidate_better += new_error < old_error
                    tied += old_error == new_error
                if original_better + candidate_better + tied != len(subset):
                    raise RuntimeError("research OCR row A/B images missing")
                row_selector_pair = {
                    "same_model_inference": True,
                    "candidate_preserves_at_most_two_original_visual_lines": True,
                    "no_reference_guided_selection": True,
                    "candidate_improved_images": candidate_better,
                    "candidate_worsened_images": original_better,
                    "equal_character_edit_images": tied,
                    "original_total_character_edits": original_errors,
                    "candidate_total_character_edits": candidate_errors,
                    "rows_removed_by_candidate": dropped_rows,
                }
            result.append({
                "language": lang,
                "watermark": watermark == "True",
                "sample_count": len(subset),
                "scores": scores,
                "same_call_raw_vs_hse_filtered": paired,
                "research_bottom_two_vs_unmodified": row_selector_pair,
                "research_caption_islands_vs_unmodified": coherent_pair,
            })
    return result


class FirstCallCapture:
    """Intercept the exact first full-frame model result without reinference.

    HSE can make extra targeted rec calls (English), but the first call is
    the unfiltered full-frame OCR that also drove the production adapter.
    """
    def __init__(self, engine):
        self.inner = engine
        self.first = None
        self.calls = 0

    def reset(self):
        self.first = None
        self.calls = 0

    def __call__(self, *args, **kwargs):
        observed = self.inner(*args, **kwargs)
        if self.calls == 0:
            self.first = observed
        self.calls += 1
        return observed

    def __getattr__(self, name):
        return getattr(self.inner, name)


def run(root: Path, model_root: Path, manifest: Path,
        per_group: int, *, all_available: bool = False,
        research_last_two_lines: bool = False,
        research_caption_islands: bool = False) -> dict:
    import cv2
    import numpy as np
    from hardsub_extractor.models import Frame
    from hardsub_extractor.ocr.rapidocr_adapter import RapidOCRAdapter
    from hardsub_extractor.ocr.caption_islands import filter_caption_island

    root = root.resolve(strict=True)
    if not root.is_dir():
        raise ValueError("public dataset not a directory")
    rows = read_reference(root)
    selected = (
        select_all_four_languages(rows) if all_available
        else select_stratified(rows, per_group)
    )
    published = load_published_predictions(root, selected)
    observed = {}
    bottom_two = {}
    coherent = {}
    raw_first = {}
    calls_by_lang = {}
    total_seconds = 0.0
    for lang in LANGUAGES:
        # Constructor verifies the independently SHA-pinned language models.
        engine = RapidOCRAdapter(
            language=lang, model_dir=str(model_root),
            manifest_path=str(manifest),
        )
        captured = FirstCallCapture(engine.engine)
        engine.engine = captured
        count = 0
        calls = 0
        for row in selected:
            if row["lang"] != lang:
                continue
            path = safe_input(root, row["image"], MAX_IMAGE_BYTES, image=True)
            data = path.read_bytes()
            decoded = cv2.imdecode(np.frombuffer(data, dtype=np.uint8),
                                   cv2.IMREAD_COLOR)
            if (
                decoded is None or decoded.ndim != 3
                or decoded.shape[2] != 3
                or decoded.shape[0] * decoded.shape[1] > MAX_IMAGE_PIXELS
            ):
                raise ValueError("unsupported benchmark source image")
            captured.reset()
            before = time.perf_counter()
            original_observation = engine.recognize(Frame(0.0, decoded))
            prediction = original_observation.text
            if research_caption_islands:
                coherent[row["id"]] = filter_caption_island(original_observation).text
            if research_last_two_lines:
                # A/B on the SAME OCR observation; do NOT re-run any model.
                candidate_rows = prediction.splitlines()
                bottom_two[row["id"]] = (
                    "\n".join(candidate_rows[-2:])
                    if len(candidate_rows) > 2 else prediction
                )
            total_seconds += time.perf_counter() - before
            if captured.calls < 1 or captured.first is None:
                raise RuntimeError("OCR capture missed the original full-frame inference")
            raw_fragments = getattr(captured.first, "txts", None) or []
            if not all(isinstance(fragment, str) for fragment in raw_fragments):
                raise RuntimeError("OCR returned unexpected raw text fragments")
            raw_first[row["id"]] = "\n".join(raw_fragments)
            calls += captured.calls
            if not isinstance(prediction, str):
                raise RuntimeError("OCR returned a non-string observation")
            observed[row["id"]] = prediction
            count += 1
        calls_by_lang[lang] = calls
        if count != sum(row["lang"] == lang for row in selected):
            raise RuntimeError("missing language observations")
    all_preds = {"hse_real_ocr": observed,
                 "same_inference_raw_ocr": raw_first, **published}
    if research_last_two_lines:
        all_preds["hse_bottom_two_research"] = bottom_two
    if research_caption_islands:
        all_preds["hse_caption_islands_research"] = coherent
    groups = summarize(selected, all_preds)
    if (
        len(groups) != 8
        or sum(g["sample_count"] for g in groups) != len(selected)
        or (not all_available and any(g["sample_count"] != per_group for g in groups))
    ):
        raise RuntimeError("benchmark incomplete")
    return {
        "dataset_commit": DATASET_COMMIT,
        "languages": list(LANGUAGES),
        "selected_images": len(selected),
        "sampling": (
            "complete pinned four-language public population"
            if all_available else "fixed SHA256-order equal clean+watermark strata"
        ),
        "all_1140_public_reference_rows_validated": True,
        "hse_model_actual_inference_executed": True,
        "research_bottom_two_lines_enabled": research_last_two_lines,
        "research_caption_islands_enabled": research_caption_islands,
        "research_uses_same_hse_model_call_without_reinference": True,
        "same_first_model_result_compared_before_vs_after_subtitle_filter": True,
        "model_calls_by_language": calls_by_lang,
        "publisher_baselines_loaded_not_executed": True,
        "same_public_images_for_every_engine_prediction": True,
        "published_raw_vs_hse_filtered_not_product_level_parity": True,
        "no_user_data_no_media_redistribution": True,
        "not_video_timing_or_real_anime_accuracy": True,
        "release_gate_passed": False,
        "hse_inference_seconds_excluding_model_load": round(total_seconds, 2),
        "scores_by_language_and_watermark": groups,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", type=Path, required=True)
    ap.add_argument("--model-dir", type=Path, required=True)
    ap.add_argument("--manifest", type=Path, default=Path("models/manifest.json"))
    ap.add_argument("--per-group", type=int, default=12)
    ap.add_argument("--all-available", action="store_true",
                    help="Run all 761 pinned ko/en/ja/zh images, without sampling")
    ap.add_argument("--research-last-two-lines", action="store_true",
                    help="Score opt-in last-two-OCR-row candidate on same model output")
    ap.add_argument("--research-caption-islands", action="store_true",
                    help="Same OCR calls, coherent multi-row caption grouping")
    args = ap.parse_args()
    try:
        answer = run(args.dataset, args.model_dir, args.manifest,
                     args.per_group, all_available=args.all_available,
                     research_last_two_lines=args.research_last_two_lines,
                     research_caption_islands=args.research_caption_islands)
    except (ValueError, OSError, RuntimeError, TypeError, KeyError) as exc:
        # No personal paths, URLs, raw captions, model internals or images.
        print("HSE_GEEKLINK_ERROR " + json.dumps({
            "error_class": type(exc).__name__,
        }))
        return 4
    print("HSE_GEEKLINK_BENCH " + json.dumps(answer, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
