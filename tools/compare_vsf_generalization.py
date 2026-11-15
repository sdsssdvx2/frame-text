"""Research-only upstream VideoSubFinder timing generalization matrix.

Four languages x two typography/layout styles x matched moving blank sources.
All media generated independently (actual MPEG4 decoded by native VSF). Native
timings never see ground truth; reference labels used ONLY in post-run grade.
This is not OCR, not real anime, not a production backend and not a release gate.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tempfile

from compare_vsf_timing_research import (
    check_native_version, run_vsf, timing_report,
)
from verify_vsf_cli_research import (
    download_pinned_archive, unpack_pinned_archive,
)
from verify_animated_background_ocr_quality import encode_video

LANGUAGES = ("ko", "en", "ja", "zh")
STYLES = ("white_two_line", "yellow_single")
MAX_VIDEO_BYTES = 32 * 1024 * 1024


def main() -> int:
    case_rows = []
    try:
        with tempfile.TemporaryDirectory(prefix="hse-vsf-matrix-") as td:
            root = Path(td)
            exe = unpack_pinned_archive(
                download_pinned_archive(root / "official.zip"),
                root / "native",
            )
            check_native_version(exe)
            for language in LANGUAGES:
                for style in STYLES:
                    folder = root / language / style
                    folder.mkdir(parents=True)
                    positive = folder / "caption.mp4"
                    blank = folder / "blank.mp4"
                    encode_video(positive, language, style=style)
                    encode_video(
                        blank, language, include_subtitles=False,
                        style=style,
                    )
                    if not 0 < min(positive.stat().st_size, blank.stat().st_size):
                        raise RuntimeError("empty encoded source")
                    if max(positive.stat().st_size, blank.stat().st_size) > MAX_VIDEO_BYTES:
                        raise RuntimeError("oversized research source")
                    source_sha = hashlib.sha256(positive.read_bytes()).hexdigest()
                    blank_sha = hashlib.sha256(blank.read_bytes()).hexdigest()
                    cues, elapsed = run_vsf(exe, positive, folder / "positive")
                    no_cues, blank_elapsed = run_vsf(exe, blank, folder / "negative")
                    if (
                        hashlib.sha256(positive.read_bytes()).hexdigest() != source_sha
                        or hashlib.sha256(blank.read_bytes()).hexdigest() != blank_sha
                    ):
                        raise RuntimeError("research source bytes changed")
                    row = {
                        "language": language,
                        "style": style,
                        "genuine_encoded_mp4": True,
                        "actual_sha_verified_native_windows_executable": True,
                        "matched_moving_blank_control": True,
                        "reference_not_used_for_native_selection": True,
                        "timing": timing_report(cues),
                        "no_caption_false_intervals": len(no_cues),
                        "elapsed_detection_seconds": elapsed,
                        "elapsed_blank_seconds": blank_elapsed,
                        "native_ocr_words_read": False,
                    }
                    case_rows.append(row)
                    print(
                        "HSE_VSF_MATRIX_CASE " + json.dumps(row, sort_keys=True),
                        flush=True,
                    )
    except Exception as exc:
        print("HSE_VSF_MATRIX_ERROR " + json.dumps({
            "error_class": type(exc).__name__,
            "cases_completed": len(case_rows),
        }), flush=True)
        return 3
    results = {
        "schema": 1,
        "cases": len(case_rows),
        "languages": list(LANGUAGES),
        "styles": list(STYLES),
        "matched_intervals": sum(
            r["timing"]["matched_reference_intervals"] for r in case_rows
        ),
        "expected_intervals": len(case_rows) * 2,
        "both_edges_250ms": sum(
            r["timing"]["both_edges_within_250ms"] for r in case_rows
        ),
        "missed_intervals": sum(
            r["timing"]["missed_reference_intervals"] for r in case_rows
        ),
        "extra_intervals": sum(
            r["timing"]["extra_timing_intervals"] for r in case_rows
        ),
        "blank_false_intervals": sum(
            r["no_caption_false_intervals"] for r in case_rows
        ),
        "synthetic_only": True,
        "third_party_ocr_executed": False,
        "upstream_bundle_added_to_product": False,
        "real_anime_release_gate_passed": False,
    }
    print("HSE_VSF_MATRIX_SUMMARY " + json.dumps(results, sort_keys=True), flush=True)
    return 0 if len(case_rows) == len(LANGUAGES) * len(STYLES) else 4


if __name__ == "__main__":
    raise SystemExit(main())
