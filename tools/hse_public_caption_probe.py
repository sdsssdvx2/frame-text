"""Source-checkout entry point for the packaged public caption diagnostic.

The same implementation runs inside the frozen Windows EXE. Both paths
discard raw subtitle content and return only anonymous aggregate results.
"""
from hardsub_extractor.public_caption_diagnostics import main, probe_one


if __name__ == "__main__":
    raise SystemExit(main())
