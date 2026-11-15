from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hardsub_extractor.model_assets import (
    download_verified_models,
    load_manifest,
    sha256_file,
)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "models" / "runtime"))
    ap.add_argument("--manifest", default=str(ROOT / "models" / "manifest.json"))
    ap.add_argument(
        "--languages", default="ko,en,zh",
        help="Pinned OCR model profiles to bundle; ja and zh share a recognizer",
    )
    ap.add_argument("--prefer-mirrors", action="store_true")
    ap.add_argument("--timeout-seconds", type=float, default=60.0)
    ap.add_argument("--attempts", type=int, default=3)
    args = ap.parse_args()
    if not 0 < args.timeout_seconds <= 120 or not 1 <= args.attempts <= 3:
        ap.error("bounded model download options required")

    manifest = load_manifest(args.manifest)
    languages = [s.strip() for s in args.languages.split(",") if s.strip()]
    if not languages:
        ap.error("at least one OCR language is required")
    profiles = {}
    for language in languages:
        paths = download_verified_models(
            args.out, manifest_path=args.manifest, language=language,
            timeout=args.timeout_seconds, attempts=args.attempts,
            prefer_mirrors=args.prefer_mirrors,
        )
        profiles[language] = {
            key: {
                "path": str(value),
                "sha256": sha256_file(value),
            }
            for key, value in (
                ("detector", paths.detector),
                ("recognizer", paths.recognizer),
                ("dictionary", paths.dictionary),
            )
        }
    result = {
        "profile": manifest["profile"],
        "model_dir": str(Path(args.out).resolve()),
        "files": profiles["ko"] if "ko" in profiles else {},
        "language_profiles": profiles,
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
