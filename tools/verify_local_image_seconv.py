"""Real published Matroska PGS/VobSub -> portable seconv -> HSE SRT/ASS.

The caller prepares SHA/git-blob-pinned, public-domain-or-upstream-test
fixture(s) and free portable converter in an isolated CI temp directory.
This script only reads its named inputs and emits sanitized numeric metrics;
it never downloads, logs dialogue, touches personal videos or installs apps.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile

from hardsub_extractor.local_image_subtitles import acquire_local_embedded_image
from hardsub_extractor.matroska_track_headers import read_image_track_headers
from hardsub_extractor.runtime_paths import resolve_binary
from hardsub_extractor.subtitle_formats import parse_subtitle_bytes


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(128 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--scratch", type=Path, required=True)
    args = parser.parse_args()
    source = args.fixture.resolve(strict=True)
    scratch = args.scratch.resolve(strict=True)
    if not source.is_file() or source.suffix.lower() != ".mkv":
        raise ValueError("published image-track fixture must be an MKV")
    source_pin = sha(source)
    results = []
    with tempfile.TemporaryDirectory(prefix="image-track-real-", dir=scratch) as tmp:
        tmpdir = Path(tmp)
        # The same upstream MKV has one VobSub and one PGS track. Remux
        # WITHOUT re-encoding the subtitle bitstreams. Label ONLY the
        # selected track English for the free pinned Latin nOCR database;
        # the unrelated image track is explicitly Japanese and must not
        # be invoked or contaminate that original-language result.
        for english_track_ordinal in (0, 1):
            selected = tmpdir / f"test-{english_track_ordinal}.mkv"
            cmd = [
                resolve_binary("ffmpeg"), "-nostdin", "-hide_banner",
                "-loglevel", "error", "-protocol_whitelist", "file,pipe",
                "-i", str(source),
                "-map", "0:s", "-c:s", "copy",
                "-metadata:s:s:0",
                f"language={'eng' if english_track_ordinal == 0 else 'jpn'}",
                "-metadata:s:s:1",
                f"language={'eng' if english_track_ordinal == 1 else 'jpn'}",
                str(selected),
            ]
            proc = subprocess.run(
                cmd, stdin=subprocess.DEVNULL, capture_output=True,
                timeout=45, check=False,
            )
            if proc.returncode or not selected.exists() or not selected.stat().st_size:
                raise AssertionError("remux of public bitmap subtitle fixture failed")
            remux_pin = sha(selected)
            original_tracks = read_image_track_headers(selected)
            if len(original_tracks) != 2:
                raise AssertionError("fixture missing its two separate bitmap tracks")
            track_numbers = {x.track_number for x in original_tracks}
            if len(track_numbers) != 2 or None in track_numbers:
                raise AssertionError("Matroska track-number evidence ambiguous")
            english = [x for x in original_tracks if x.language.lower() == "eng"]
            if len(english) != 1:
                raise AssertionError("explicit English bitmap track was lost")
            dest = tmpdir / "outputs" / f"track-{english_track_ordinal}"
            result = acquire_local_embedded_image(
                selected, dest, language="en",
            )
            if result is None:
                raise AssertionError("portable free image OCR engine unavailable")
            if result.track_number != english[0].track_number:
                raise AssertionError("wrong Matroska subtitle image track selected")
            sf, sc = parse_subtitle_bytes(
                result.srt_path.read_bytes(), source_url="x.srt",
            )
            af, ac = parse_subtitle_bytes(
                result.ass_path.read_bytes(), source_url="x.ass",
            )
            if sf != "srt" or af != "ass" or not sc or len(sc) != len(ac):
                raise AssertionError("bitmap-OCR SRT/ASS mismatch")
            for a, b in zip(sc, ac):
                if (a.text != b.text or abs(a.start-b.start) > .02
                        or abs(a.end-b.end) > .02):
                    raise AssertionError("bitmap-OCR text/timing export not preserved")
            meta = json.loads(result.provenance_path.read_text("utf-8"))
            if (meta.get("ocr_applied") is not True
                    or meta.get("original_text_verified") is not False
                    or meta.get("image_track_number") != english[0].track_number):
                raise AssertionError("image OCR falsely labelled original text/track")
            existing = {
                path: sha(path) for path in (
                    result.srt_path, result.ass_path,
                    result.extracted_track_path, result.provenance_path,
                )
            }
            try:
                acquire_local_embedded_image(selected, dest, language="en")
            except FileExistsError:
                pass
            else:
                raise AssertionError("second run overwrote existing OCR output")
            if any(sha(path) != value for path, value in existing.items()):
                raise AssertionError("existing subtitles were modified")
            if sha(selected) != remux_pin or sha(source) != source_pin:
                raise AssertionError("input media modified")
            results.append({
                "codec": english[0].codec,
                "track_number": english[0].track_number,
                "cues": len(sc),
            })
    codecs = {x["codec"] for x in results}
    if codecs != {"dvd_subtitle", "hdmv_pgs_subtitle"}:
        raise AssertionError("did not exercise both upstream image types")
    # Log NUMERIC evidence only. Never emit subtitle text or actual paths.
    print("HSE_BITMAP_TRACK_SECONV_REAL=" + json.dumps({
        "source_upstream_public_fixture": True,
        "independent_bitmap_codecs": sorted(codecs),
        "ocr_generated_cues_by_codec": {
            item["codec"]: item["cues"] for item in results
        },
        "separate_language_selected_tracks": len(results),
        "track_numbers_from_bounded_Matroska_EBML": [
            item["track_number"] for item in results
        ],
        "srt_ass_equal_text_and_timing": True,
        "old_files_preserved": True,
        "source_media_unchanged": True,
        "source_original_text_verified": False,
        "model_or_site_accuracy_verified": False,
        "user_files_changed": False,
    }, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
