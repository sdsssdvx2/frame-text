"""Isolated real native timing A/B for the old and new official VSF CLI.

No OCR models, user videos, browser, credentials or upstream network calls
except downloading the two explicitly pinned official releases into runner
scratch. No HSE production runtime or operating app is altered.
"""
from __future__ import annotations

import json
from pathlib import Path
import subprocess
import tempfile
import time

from hardsub_extractor.native_vsf_assets import ensure_official_vsf
from hardsub_extractor.native_vsf_timing import detect_native_vsf_timings
from verify_animated_background_ocr_quality import encode_video


def main() -> int:
    findings = {}
    with tempfile.TemporaryDirectory(prefix="hse-vsf-side-by-side-") as scratch:
        root = Path(scratch)
        cache = root / "pinned-release-cache"
        releases = {}
        for version in ("v6.10.5", "v6.10.7"):
            executable, digest = ensure_official_vsf(
                consent=True, cache_base=cache, release_version=version,
            )
            check = subprocess.run(
                [str(executable), "--version"], cwd=executable.parent,
                stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                stdin=subprocess.DEVNULL, timeout=20, check=False,
            )
            response = (check.stdout + check.stderr)[:1024].decode(
                "utf-8", "replace"
            )
            if check.returncode != 0 or version.removeprefix("v") not in response:
                raise RuntimeError("official pinned native version did not pass")
            releases[version] = (executable, digest)
        if releases["v6.10.5"][0] == releases["v6.10.7"][0]:
            raise RuntimeError("older official VSF release was replaced")
        print("HSE_NATIVE_6107_RELEASE=PINNED_OFFICIAL_SHA_AND_VERSION_PASS")
        for lang in ("ja", "zh"):
            target = root / lang
            target.mkdir()
            source = target / "caption.mp4"
            negative = target / "blank.mp4"
            encode_video(source, lang, style="white_two_line")
            encode_video(
                negative, lang, include_subtitles=False, style="white_two_line"
            )
            for variant, path in (("caption", source), ("blank", negative)):
                for version, (binary, digest) in releases.items():
                    start = time.monotonic()
                    result = detect_native_vsf_timings(
                        path, binary, pinned_executable_sha256=digest,
                        max_seconds=180,
                    )
                    findings[f"{lang}/{variant}/{version}"] = {
                        "interval_count": len(result.intervals),
                        "seconds": round(time.monotonic()-start, 3),
                    }
        # A native detector can miss cues even when the executable is healthy.
        # Compare numeric observations, do not claim accuracy or promote the
        # candidate automatically. A blank control must be reported honestly.
        old = [findings[k]["interval_count"] for k in findings if k.endswith("v6.10.5")]
        new = [findings[k]["interval_count"] for k in findings if k.endswith("v6.10.7")]
        print("HSE_VSF_6107_AB=" + json.dumps({
            "version_checks_passed": True,
            "original_user_media_access": False,
            "upstream_paid_api_calls": 0,
            "count_comparison": findings,
            "old_total_intervals": sum(old),
            "new_total_intervals": sum(new),
            "candidate_approved_for_production": False,
            "output_text_or_private_paths_logged": False,
        }, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
