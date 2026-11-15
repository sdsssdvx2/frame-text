"""Backwards-compatible command-line entrypoint for public caption-only batch.

Keep the executable script path stable; the implementation lives inside the
installed hardsub_extractor package so full test discovery imports it safely.
"""
from hardsub_extractor.subtitle_batch_cli import _load_urls, main


if __name__ == "__main__":
    raise SystemExit(main())
