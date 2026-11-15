"""Compatibility CLI for the app-packaged offline frame review module."""
from __future__ import annotations

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from hardsub_extractor.review_pack import main

if __name__ == "__main__":
    raise SystemExit(main())
