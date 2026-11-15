"""Pinned public subtitle-image benchmark; no paid calls or image redistribution.

GeekLink's CC0 annotations cover subtitles rendered onto three public/open
film sources, but some source frames require CC BY-SA 3.0 attribution.
This is a real-background SYNTHETIC-caption image pilot, not a proof of
natural subtitle recognition or episode time boundaries.

Selection is exactly 4 languages x 3 films x 2 watermark conditions;
recognition uses the existing HSE PP-OCRv5 production models and ROI.
No images, captions, raw model predictions, or user files are saved.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
from io import BytesIO, StringIO
import json
from pathlib import Path
import re
import sys
import urllib.request
from urllib.parse import urlsplit

REPO = "GeekLinkDev/geeklink-ocr-benchmark"
COMMIT = "baa9ccbb7ffad36d56cc0ea115256a09bdf427f4"
TREE = "8cb62b5d6c2c40033e062cebe2f62d6feda45e63"
MANIFEST = "data/manifest.csv"
MANIFEST_BLOB = "9e38415f5cba1663f8ad29fe75acfbd7e91f7c85"
BASELINE = "baselines/geeklink.csv"
BASELINE_BLOB = "71a81b23e586a1f305d0ed8d54775c18f9c73f9f"
LANGUAGES = ("ko", "en", "ja", "zh")
FILMS = ("v01", "v02", "v03")
COLUMNS = ("id", "video_id", "lang", "image", "ground_truth", "has_watermark")


class BenchmarkError(ValueError):
    pass


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *args, **kwargs):
        raise BenchmarkError("redirect_denied")


def _fetch(url: str, host: str, limit: int) -> bytes:
    parsed = urlsplit(url)
    if (parsed.scheme != "https" or parsed.netloc != host
            or parsed.username or parsed.password or parsed.fragment
            or parsed.query not in ("", "recursive=1")):
        raise BenchmarkError("untrusted_url")
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "HSE-public-OCR-evaluation"})
        with urllib.request.build_opener(NoRedirect()).open(req, timeout=20) as res:
            if res.status != 200 or int(res.headers.get("Content-Length") or 0) > limit:
                raise BenchmarkError("upstream_http_or_size")
            data = res.read(limit + 1)
    except BenchmarkError:
        raise
    except urllib.error.HTTPError as exc:
        raise BenchmarkError("upstream_http_" + str(exc.code)) from exc
    except Exception as exc:
        raise BenchmarkError("upstream_unavailable") from exc
    if not data or len(data) > limit:
        raise BenchmarkError("upstream_body_outside_budget")
    return data


def _git_blob_verify(data: bytes, sha: str) -> None:
    if not re.fullmatch("[0-9a-f]{40}", sha):
        raise BenchmarkError("bad_blob_pin")
    body = ("blob %d\0" % len(data)).encode("ascii") + data
    if hashlib.sha1(body).hexdigest() != sha:
        raise BenchmarkError("source_blob_integrity_failed")


def _tree() -> dict[str, tuple[str, int]]:
    url = f"https://api.github.com/repos/{REPO}/git/trees/{TREE}?recursive=1"
    data = json.loads(_fetch(url, "api.github.com", 500_000))
    if data.get("sha") != TREE or data.get("truncated") is not False:
        raise BenchmarkError("source_tree_integrity_failed")
    files = {}
    for entry in data["tree"]:
        if entry.get("type") == "blob" and entry.get("mode") == "100644":
            files[entry["path"]] = (entry["sha"], entry["size"])
    for path, pin in ((MANIFEST, MANIFEST_BLOB), (BASELINE, BASELINE_BLOB)):
        if files.get(path, ("", 0))[0] != pin:
            raise BenchmarkError("source_manifest_pin_mismatch")
    return files


def _blob(path: str, files: dict[str, tuple[str, int]], limit: int) -> bytes:
    if (not re.fullmatch("[a-zA-Z0-9_./-]+", path)
            or path not in files or files[path][1] > limit):
        raise BenchmarkError("source_path_not_allowed")
    url = f"https://raw.githubusercontent.com/{REPO}/{COMMIT}/{path}"
    data = _fetch(url, "raw.githubusercontent.com", limit)
    if len(data) != files[path][1]:
        raise BenchmarkError("source_length_mismatch")
    _git_blob_verify(data, files[path][0])
    return data


def _csv(data: bytes, columns: tuple[str, ...]) -> list[dict]:
    try:
        reader = csv.DictReader(StringIO(data.decode("utf-8-sig")))
        if tuple(reader.fieldnames or ()) != columns:
            raise BenchmarkError("source_schema_mismatch")
        rows = list(reader)
    except (UnicodeError, csv.Error) as exc:
        raise BenchmarkError("source_csv_invalid") from exc
    if any(None in row or any(value is None for value in row.values()) for row in rows):
        raise BenchmarkError("source_csv_row_invalid")
    return rows


def _manifest(raw: bytes, count: int = 1140) -> list[dict]:
    rows = _csv(raw, COLUMNS)
    if len(rows) != count:
        raise BenchmarkError("source_count_drift")
    ids = set()
    for row in rows:
        sid = row["id"]
        if (not re.fullmatch(r"v0[1-3]_[0-9]{4}", sid)
                or sid in ids or row["video_id"] not in FILMS
                or not sid.startswith(row["video_id"] + "_")
                or row["image"] != "data/images/" + sid + ".jpg"
                or row["lang"] not in LANGUAGES + ("es", "el")
                or row["has_watermark"] not in ("True", "False")
                or not row["ground_truth"] or len(row["ground_truth"]) > 500):
            raise BenchmarkError("source_row_drift")
        ids.add(sid)
    return rows


def _select(rows: list[dict]) -> list[dict]:
    picked = []
    for lang in LANGUAGES:
        for mark in ("False", "True"):
            for film in FILMS:
                cell = sorted((row for row in rows
                               if row["lang"] == lang
                               and row["has_watermark"] == mark
                               and row["video_id"] == film),
                              key=lambda row: row["id"])
                if not cell:
                    raise BenchmarkError("stratum_missing")
                picked.append(cell[len(cell) // 2])
    if len({row["id"] for row in picked}) != 24:
        raise BenchmarkError("stratified_sample_not_unique")
    return picked


def _distance(a: str, b: str) -> int:
    if len(a) > 4096 or len(b) > 500:
        raise BenchmarkError("text_length_budget")
    if len(a) > len(b):
        a, b = b, a
    previous = list(range(len(a) + 1))
    for i, char in enumerate(b, 1):
        current = [i]
        for j, other in enumerate(a, 1):
            current.append(min(current[-1] + 1, previous[j] + 1,
                               previous[j - 1] + (char != other)))
        previous = current
    return previous[-1]


def _score(rows: list[tuple[str, str, str, str]]) -> dict:
    buckets = {"all": []}
    for language, watermark, prediction, reference in rows:
        for key in (language, "watermark" if watermark == "True" else "clean"):
            buckets.setdefault(key, []).append((prediction, reference))
        buckets["all"].append((prediction, reference))
    return {
        name: {"n": len(items), "cer": round(
            sum(_distance(p, r) for p, r in items)
            / sum(max(len(r), 1) for _, r in items), 4,
        )}
        for name, items in buckets.items()
    }


def _decode(data: bytes):
    import numpy as np
    from PIL import Image
    try:
        with Image.open(BytesIO(data)) as image:
            if image.format != "JPEG" or not 240 <= image.width <= 1920 or not 120 <= image.height <= 1080:
                raise BenchmarkError("source_image_shape")
            image.load()
            rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
    except (ValueError, OSError) as exc:
        raise BenchmarkError("source_image_invalid") from exc
    return rgb[:, :, ::-1].copy()


def source_preflight():
    """Network and source pin admission, without installing or loading OCR."""
    files = _tree()
    source = _blob(MANIFEST, files, 200_000)
    selected = _select(_manifest(source))
    published = _csv(_blob(BASELINE, files, 200_000), ("id", "prediction"))
    baseline = {row["id"]: row["prediction"] for row in published}
    if len(baseline) != 1140 or not all(row["id"] in baseline for row in selected):
        raise BenchmarkError("upstream_baseline_incomplete")
    _blob(selected[0]["image"], files, 250_000)
    return files, source, selected, baseline


def measure(model_dir: Path, model_manifest: Path) -> dict:
    from hardsub_extractor.frame_source import NormalizedROI
    from hardsub_extractor.models import Frame
    from hardsub_extractor.ocr.rapidocr_adapter import RapidOCRAdapter
    files, source, selected, baseline = source_preflight()
    recognizers = {
        lang: RapidOCRAdapter(language=lang, model_dir=str(model_dir),
                              manifest_path=str(model_manifest))
        for lang in LANGUAGES
    }
    output = {"hse_production_roi": [], "hse_full_frame": [],
              "geeklink_public_onnx_raw": []}
    for row in selected:
        image = _decode(_blob(row["image"], files, 250_000))
        height, width = image.shape[:2]
        x, y, w, h = NormalizedROI().pixels(width, height)
        roi = image[y:y+h, x:x+w].copy()
        engine = recognizers[row["lang"]]
        predictions = {
            "hse_production_roi": engine.recognize(Frame(0.0, roi)).text,
            "hse_full_frame": engine.recognize(Frame(0.0, image)).text,
            "geeklink_public_onnx_raw": baseline[row["id"]],
        }
        for name, text in predictions.items():
            output[name].append((row["lang"], row["has_watermark"],
                                 text, row["ground_truth"]))
    return {
        "schema": 1, "type": "pinned_public_film_subtitle_ocr_pilot",
        "upstream_commit": COMMIT, "upstream_tree": TREE,
        "manifest_sha256": hashlib.sha256(source).hexdigest(),
        "subject": "constructed_subtitles_on_real_open_film_frames",
        "not_a_natural_video_timing_benchmark": True,
        "license_scope": "CC0 annotations; some images require CC BY-SA 3.0",
        "images_redistributed": False, "paid_api_calls": 0,
        "quality_acceptance_claimed": False,
        "total_images": len(selected), "watermark_images": 12,
        "scores": {name: _score(rows) for name, rows in output.items()},
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", type=Path)
    ap.add_argument("--manifest", type=Path)
    ap.add_argument("--out", type=Path)
    ap.add_argument("--source-only", action="store_true")
    ap.add_argument("--accept-public-download", action="store_true")
    args = ap.parse_args(argv)
    if not args.accept_public_download:
        print("HSE_GEEKLINK_PILOT=NO_DOWNLOAD_CONSENT")
        return 2
    try:
        if args.source_only:
            source_preflight()
            print("HSE_GEEKLINK_SOURCE=PIN_VERIFIED;SAMPLES=24;IMAGES_SAVED=0")
            return 0
        if args.model_dir is None or args.manifest is None or args.out is None:
            raise BenchmarkError("missing_explicit_model_or_output")
        result = measure(args.model_dir, args.manifest)
        with args.out.open("x", encoding="utf-8") as out:
            json.dump(result, out, indent=2)
            out.write("\n")
    except Exception as exc:
        reason = (str(exc) if isinstance(exc, BenchmarkError)
                  else type(exc).__name__)
        print("HSE_GEEKLINK_PILOT=FAILED;SAFE_REASON=" + reason, file=sys.stderr)
        return 3
    print("HSE_GEEKLINK_PILOT=MEASURED;IMAGES=24;WATERMARK=12;UPLOADS=0;PAID=0")
    for name, scores in result["scores"].items():
        print(f"HSE_GEEKLINK_CER={name};ALL={scores['all']['cer']:.4f}"
              f";CLEAN={scores['clean']['cer']:.4f}"
              f";WATERMARK={scores['watermark']['cer']:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
