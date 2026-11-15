"""Strict public-media-only HSE vs third-party CLI research.

prepare: fetch two publisher-original media files and independent TimedText,
         pin their exact bytes, and write no-clobber source/reference records.
score: read only a fresh third-party experiment's SRTs, verify original media
       pins again, and publish aggregate metrics without leaking any text.

The external engine MUST be executed independently in an isolated sandbox;
this script does not install, import or execute AVL's untrusted source.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import unicodedata

from compare_external_engine_exports import (
    Cue, evaluate, parse_strict_srt,
)
from verify_commons_natural_hardsub import (
    EXPECTED_LINES, EXPECTED_TEXT_SHA256, EXPECTED_TIMES,
    PUBLISHED, _get_media, _get_reference,
)

AVL_COMMIT = "ee0269df692719b658ccae0e75106e9b84af40ec"
SOURCE_SHA256 = {
    "caption": "01ffc4c35508dda62edfa4dff4e6541e03cfc769a5f76cd71d35cb61713a7f14",
    "clean": "3dc8afa5687766f0cc0a70a1721b05737d5c8ebed815b7c37015cf12b13708ed",
}
MAX_EXPORT_BYTES = 2 * 1024 * 1024
MAX_OUTPUT_CUES = 200
HSE_BASELINE = {
    "baseline_run": 42421046563,
    "bottom_two_run": 42422623555,
    "baseline_exact_text_wrap": 3,
    "baseline_strict_250ms": 0,
    "baseline_no_caption_false_cues": 2,
    "bottom_two_exact_text_wrap": 4,
    "bottom_two_strict_250ms": 1,
    "bottom_two_no_caption_false_cues": 2,
}


def _sha256(path: Path) -> str:
    with path.open("rb") as stream:
        digest = hashlib.sha256()
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
        return digest.hexdigest()


def _existing_regular_file(path: Path, maximum: int, *, allow_empty=False) -> bytes:
    if path.is_symlink() or not path.is_file():
        raise ValueError("missing_or_symlinked_trial_file")
    n = path.stat().st_size
    if n > maximum or (not allow_empty and n == 0):
        raise ValueError("trial_file_size_invalid")
    with path.open("rb") as stream:
        payload = stream.read(maximum + 1)
    if len(payload) > maximum:
        raise ValueError("trial_file_grew")
    return payload


def _timestamp(seconds: float) -> str:
    total = round(seconds * 1000)
    hours, rem = divmod(total, 3600000)
    minutes, rem = divmod(rem, 60000)
    whole, ms = divmod(rem, 1000)
    return f"{hours:02d}:{minutes:02d}:{whole:02d},{ms:03d}"


def _srt_payload(cues: list[Cue]) -> bytes:
    lines = []
    for i, cue in enumerate(cues, 1):
        lines.extend([
            str(i),
            f"{_timestamp(cue.start)} --> {_timestamp(cue.end)}",
            cue.text,
            "",
        ])
    return ("\n".join(lines) + "\n").encode("utf-8")


def _reference_sha(cues: list[Cue]) -> str:
    return hashlib.sha256(
        "\x1e".join(cue.text for cue in cues).encode("utf-8"),
    ).hexdigest()


def _check_public_cue_source(cues: list[Cue]) -> None:
    if len(cues) != len(EXPECTED_TIMES) or _reference_sha(cues) != EXPECTED_TEXT_SHA256:
        raise ValueError("published_reference_pin_changed")
    for cue, (start, end), line_count in zip(cues, EXPECTED_TIMES, EXPECTED_LINES):
        if (
            abs(cue.start - start) > 0.001
            or abs(cue.end - end) > 0.001
            or len(cue.text.splitlines()) != line_count
        ):
            raise ValueError("published_reference_timing_or_layout_changed")


def prepare(root: Path) -> dict:
    if root.exists() or root.is_symlink():
        raise FileExistsError("study_source_directory_already_exists")
    root.mkdir(mode=0o700)
    try:
        reference, raw_sha = _get_reference()
        _check_public_cue_source(reference)
        video_pins = {}
        for label in ("caption", "clean"):
            video_pins[label] = _get_media(root / (label + ".webm"), label)
            if video_pins[label] != SOURCE_SHA256[label]:
                raise ValueError("public_source_sha256_changed")
            if (root / (label + ".webm")).stat().st_size != PUBLISHED[label][1]:
                raise ValueError("published_media_size_changed")
        reference_bytes = _srt_payload(reference)
        with (root / "reference.srt").open("xb") as stream:
            stream.write(reference_bytes)
        _check_public_cue_source(parse_strict_srt(reference_bytes))
        attestation = {
            "schema": 1,
            "avl_source_commit": AVL_COMMIT,
            "published_video_sha256": video_pins,
            "reference_text_sha256": EXPECTED_TEXT_SHA256,
            "reference_raw_publisher_sha256": raw_sha,
        }
        with (root / "sources.json").open("x", encoding="utf-8") as stream:
            json.dump(attestation, stream, sort_keys=True)
        return {
            "schema": 1,
            "media_source_sha256_verified": True,
            "reference_text_sha256_verified": True,
            "third_party_ocr_executed": False,
            "reference_cue_count": len(reference),
        }
    except BaseException:
        # Clean up ONLY the new private study root on failed preparation;
        # user sources and existing application artifacts are not in scope.
        import shutil
        shutil.rmtree(root)
        raise



def _verify_export_json(raw: bytes, observed: list[Cue]) -> None:
    """Require AVL's independent JSON cue export to agree with its SRT.

    Never accept an apparent SRT pass if the engine's companion structured
    output has different text/count/timing; no proprietary model internals
    or reference transcript need to be accessed by the vendor.
    """
    def unique_pairs(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("duplicate_avl_json_field")
            result[key] = value
        return result

    try:
        payload = json.loads(
            raw, object_pairs_hook=unique_pairs,
            parse_constant=lambda _: (_ for _ in ()).throw(
                ValueError("invalid_nonfinite_avl_json")
            ),
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("malformed_avl_json") from exc
    if not isinstance(payload, list) or len(payload) != len(observed):
        raise ValueError("avl_srt_json_cue_count_mismatch")
    for structured, cue in zip(payload, observed):
        if not isinstance(structured, dict):
            raise ValueError("invalid_avl_json_cue_record")
        start, end = structured.get("start_ms"), structured.get("end_ms")
        body = structured.get("text")
        if (
            not isinstance(start, (int, float)) or isinstance(start, bool)
            or not isinstance(end, (int, float)) or isinstance(end, bool)
            or not isinstance(body, str)
        ):
            raise ValueError("avl_srt_json_cue_schema_mismatch")
        import math
        if (
            not math.isfinite(start) or not math.isfinite(end)
            or abs(round(start) / 1000.0 - cue.start) > 0.0005
            or abs(round(end) / 1000.0 - cue.end) > 0.0005
            or unicodedata.normalize("NFC", body) != cue.text
        ):
            raise ValueError("avl_srt_json_cue_content_mismatch")


def score(root: Path, output_root: Path) -> dict:
    if root.is_symlink() or not root.is_dir():
        raise ValueError("public_trial_source_not_directory")
    if output_root.is_symlink() or not output_root.is_dir():
        raise ValueError("third_party_output_not_directory")
    info = json.loads(_existing_regular_file(root / "sources.json", 8192))
    if (info.get("schema") != 1
            or info.get("avl_source_commit") != AVL_COMMIT
            or info.get("published_video_sha256") != SOURCE_SHA256
            or info.get("reference_text_sha256") != EXPECTED_TEXT_SHA256):
        raise ValueError("research_source_metadata_invalid")
    refs = parse_strict_srt(
        _existing_regular_file(root / "reference.srt", 8192)
    )
    _check_public_cue_source(refs)
    source_sizes = {}
    for label in ("caption", "clean"):
        source = root / (label + ".webm")
        if source.is_symlink() or not source.is_file():
            raise ValueError("source_video_missing_or_symlinked")
        if source.stat().st_size != PUBLISHED[label][1]:
            raise ValueError("source_video_size_invalid")
        if _sha256(source) != SOURCE_SHA256[label]:
            raise ValueError("source_video_corrupted")
        source_sizes[label] = source.stat().st_size
    observed = {}
    counts = {}
    for label in ("caption", "clean"):
        text = _existing_regular_file(
            output_root / (label + "-result.srt"),
            MAX_EXPORT_BYTES, allow_empty=True,
        )
        parsed = parse_strict_srt(text, allow_empty=True)
        if len(parsed) > MAX_OUTPUT_CUES:
            raise ValueError("implausible_third_party_cue_count")
        observed[label] = parsed
        counts[label] = len(parsed)
        structured = _existing_regular_file(
            output_root / (label + "-result.json"),
            MAX_EXPORT_BYTES,
        )
        _verify_export_json(structured, parsed)
        _existing_regular_file(
            output_root / (label + "-result.meta.json"),
            MAX_EXPORT_BYTES,
        )
    result = evaluate(refs, observed["caption"], observed["clean"])
    return {
        "schema": 1,
        "avl_commit": AVL_COMMIT,
        "same_pinned_public_videos_as_hse_baseline": True,
        "same_four_publisher_reference_cues_as_hse_baseline": True,
        "third_party_engine_score_from_actual_output_srt": True,
        "independent_publisher_reference_not_passed_to_ocr": True,
        "third_party_source_and_installation_pinned_by_workflow": True,
        "github_hosted_sandbox_offline_and_without_user_data": True,
        "not_automatic_release_gate": True,
        "not_four_language_or_real_anime_proof": True,
        "source_lengths": source_sizes,
        "observed_cue_counts": counts,
        "hse_comparison_reference": HSE_BASELINE,
        "score": result,
    }


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("operation", choices=("prepare", "score"))
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--output", type=Path)
    args = p.parse_args()
    try:
        if args.operation == "prepare":
            report = prepare(args.source)
            print("HSE_AVL_PREPARE " + json.dumps(report, sort_keys=True))
        else:
            if args.output is None:
                raise ValueError("trial_output_required")
            report = score(args.source, args.output)
            print("HSE_AVL_NATURAL_RESULT " + json.dumps(report, sort_keys=True))
        return 0
    except (ValueError, FileExistsError, OSError, TypeError, KeyError) as exc:
        # Do not log video title, raw captions, paths, model internals or URLs.
        print("HSE_AVL_TRIAL_ERROR " + json.dumps({
            "error_type": type(exc).__name__,
        }))
        return 4


if __name__ == "__main__":
    raise SystemExit(main())
