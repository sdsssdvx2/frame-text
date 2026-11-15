"""Isolated public upstream model-integrity reconnaissance.

No model is installed, bundled, trusted or enabled by this probe.
Only official RapidOCR v3.10.0 model assets are downloaded into a temporary
directory and SHA256/byte counts are reported. A recognizer is accepted only
when its SHA matches the hash published in RapidOCR's default_models.yaml.
Dictionary hashes must be independently pinned before production use.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tempfile
import urllib.request

ORIGIN = "https://www.modelscope.cn/models/RapidAI/RapidOCR/resolve/v3.10.0"
ASSETS = {
    "ch_rec": (
        "onnx/PP-OCRv5/rec/ch_PP-OCRv5_rec_mobile.onnx",
        "5825fc7ebf84ae7a412be049820b4d86d77620f204a041697b0494669b1742c5",
    ),
    "ch_dict": (
        "paddle/PP-OCRv5/rec/ch_PP-OCRv5_rec_mobile/ppocrv5_dict.txt",
        None,
    ),
    "en_rec": (
        "onnx/PP-OCRv5/rec/en_PP-OCRv5_rec_mobile.onnx",
        "c3461add59bb4323ecba96a492ab75e06dda42467c9e3d0c18db5d1d21924be8",
    ),
    "en_dict": (
        "paddle/PP-OCRv5/rec/en_PP-OCRv5_rec_mobile/ppocrv5_en_dict.txt",
        None,
    ),
}
CAP = 48 * 1024 * 1024


def inspect_asset(name: str, path: str, expected: str | None) -> dict:
    """Bound each download, verify upstream-recognizer checksum and discard."""
    req = urllib.request.Request(
        ORIGIN + "/" + path,
        headers={"User-Agent": "HSE-isolated-model-integrity-probe/1"},
    )
    sha = hashlib.sha256()
    count = 0
    with tempfile.TemporaryFile() as dest:
        with urllib.request.urlopen(req, timeout=75) as response:
            while True:
                chunk = response.read(256 * 1024)
                if not chunk:
                    break
                count += len(chunk)
                if count > CAP:
                    raise ValueError("model asset exceeds bounded size")
                sha.update(chunk)
                dest.write(chunk)
        if not count:
            raise ValueError("empty upstream model asset")
        if expected is not None and sha.hexdigest() != expected:
            raise ValueError("official RapidOCR recognizer SHA mismatch")
        if name.endswith("_dict"):
            dest.seek(0)
            chars = dest.read().decode("utf-8-sig").splitlines()
            if len(chars) < 50 or len(chars) > 50000:
                raise ValueError("recognition dictionary has invalid token count")
            if not all(chars):
                raise ValueError("recognition dictionary contains empty token")
            vocabulary = len(chars)
        else:
            vocabulary = None
        return {
            "asset": name,
            "sha256": sha.hexdigest(),
            "bytes": count,
            "upstream_recognizer_pin_matched": expected is not None,
            "dictionary_tokens": vocabulary,
        }


def main() -> int:
    results = []
    for name, (path, expected) in ASSETS.items():
        try:
            row = inspect_asset(name, path, expected)
            results.append(row)
            print("HSE_MODEL_PROBE " + json.dumps(row, sort_keys=True))
        except Exception as exc:
            # Never print URLs or network exception strings (may contain
            # credentials or unexpected upstream response content).
            print(f"HSE_MODEL_PROBE asset={name} status=error type={type(exc).__name__}")
            return 1
    print(f"HSE_MODEL_PROBE_COMPLETE verified_downloads={len(results)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
