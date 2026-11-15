"""Backward-compatible developer entrypoint: HSE package owns the engine."""
from hardsub_extractor.tab_capture import (
    PAGE, MAX_CHUNK, MAX_TOTAL, CaptureSession, _Server, _Handler,
    offline_ocr, serve, main,
)
if __name__ == "__main__":
    raise SystemExit(main())
