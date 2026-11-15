"""Independent official PyPI RapidVideOCR v3.1.1 versus original CJK MP4s.

This exercises a mature third-party OCR/exporter unchanged except for the
required versioned RapidOCR PP-OCRv5 model parameters. The unsafe
RapidVideOCR VideoSubFinder process launcher is NOT invoked. Only the
standalone, SHA-verified third-party native detector generates RGBImages,
in a fresh isolated disposable directory. Data from decoded source video
is never sent to a cloud service. Official third-party outputs are only
read, inspected for structural quality, and then deleted with the scratch.
NOT production, NOT real-anime proof and NOT a release quality gate.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tempfile
import time

from compare_vsf_timing_research import check_native_version, run_vsf
from verify_vsf_cli_research import download_pinned_archive, unpack_pinned_archive
from compare_external_engine_exports import parse_strict_srt, evaluate, Cue as ScoreCue
from verify_animated_background_ocr_quality import encode_video, _multiline_reference
from verify_multilang_video_pipeline import EXPECTED_SPANS, VIDEO_TEXT
from hardsub_extractor.model_assets import resolve_model_paths
from hardsub_extractor.subtitle_formats import parse_subtitle_bytes
from rapidocr import EngineType, LangRec, ModelType, OCRVersion

PACKAGE = "rapid-videocr"
VERSION = "3.1.1"
OFFICIAL_PYPI_WHEEL_SHA256 = (
    "5ccfa46f9911eb02c21a4e60946e2fd4d1c282ddbdfcd4f8a780e867816e56b2"
)
LANGUAGES = ("ja", "zh")  # Legacy two-language reproducibility baseline
EXTENDED_LANGUAGES = ("ko", "en", "ja", "zh")
MAX_RGB_IMAGES = 60


def official_rec_language(language: str):
    """Use the same explicit model language routing as HSE's vetted adapter."""
    mapping = {
        "ko": LangRec.KOREAN, "en": LangRec.EN,
        "ja": LangRec.CH, "zh": LangRec.CH,
    }
    if language not in mapping:
        raise ValueError("unsupported recognition language")
    return mapping[language]


def parse_research_languages(value: str) -> tuple[str, ...]:
    """Reject implicit, duplicate or unsupported language profiles."""
    if not isinstance(value, str):
        raise ValueError("invalid research languages")
    codes = tuple(item.strip() for item in value.split(","))
    if (
        not codes or len(codes) != len(set(codes))
        or any(item not in EXTENDED_LANGUAGES for item in codes)
    ):
        raise ValueError("supported distinct explicit ko/en/ja/zh languages only")
    return codes


def official_v5_params(model_root: Path, manifest: Path, language: str) -> dict:
    paths = resolve_model_paths(
        model_root, manifest_path=manifest, language=language,
        verify=True,
    )
    return {
        "Det.engine_type": EngineType.ONNXRUNTIME,
        "Det.model_type": ModelType.MOBILE,
        "Det.ocr_version": OCRVersion.PPOCRV5,
        "Det.model_path": str(paths.detector),
        "Rec.engine_type": EngineType.ONNXRUNTIME,
        "Rec.lang_type": official_rec_language(language),
        "Rec.model_type": ModelType.MOBILE,
        "Rec.ocr_version": OCRVersion.PPOCRV5,
        "Rec.model_path": str(paths.recognizer),
        "Rec.rec_keys_path": str(paths.dictionary),
        "Global.use_cls": False,
    }


def grade_srt_ass(srt: Path, ass: Path, reference: list[ScoreCue],
                  native_blank_intervals: int) -> dict:
    if not srt.is_file() or not ass.is_file():
        raise RuntimeError("official RapidVideOCR did not create SRT and ASS")
    if srt.stat().st_size > 512 * 1024 or ass.stat().st_size > 512 * 1024:
        raise RuntimeError("official RapidVideOCR output exceeds research bound")
    valid_srt = False
    actual = []
    try:
        actual = parse_strict_srt(srt.read_bytes(), allow_empty=True)
        valid_srt = True
    except ValueError:
        pass
    consistent = False
    if valid_srt:
        try:
            _, srt_cues = parse_subtitle_bytes(
                srt.read_bytes(), source_url="output.srt",
            )
            _, ass_cues = parse_subtitle_bytes(
                ass.read_bytes(), source_url="output.ass",
            )
            consistent = (
                len(srt_cues) == len(ass_cues) == len(actual)
                and all(
                    a.text == b.text
                    and abs(a.start - b.start) <= 0.011
                    and abs(a.end - b.end) <= 0.011
                    for a, b in zip(srt_cues, ass_cues)
                )
            )
        except ValueError:
            consistent = False
    scores = evaluate(
        reference, actual,
        [ScoreCue(9 + i, 10 + i, "[vsf false interval]")
         for i in range(native_blank_intervals)],
    ) if valid_srt else None
    return {
        "strict_srt_valid": valid_srt,
        "valid_ass_matching_srt": consistent,
        "scores": scores,
        "release_gate_passed": False,
    }



def grade_lossless_hse_reexport(
    official_srt: Path, official_ass: Path, separate_base: Path,
    reference: list[ScoreCue], native_blank_intervals: int,
) -> dict:
    """Research-only proof that the *official SRT* can be losslessly repaired.

    The external ASS remains unchanged, including any multiline formatting
    defect; regenerated SRT/ASS are clearly FIRST-PARTY exports in a fresh
    temporary directory. They are not mislabelled as official RapidVideOCR.
    Existing paths are protected by the exclusive subtitle-pair publisher.
    """
    from hardsub_extractor.models import Cue as ExportCue
    from hardsub_extractor.writers import write_subtitle_pair_exclusive

    original_srt = official_srt.read_bytes()
    original_ass = official_ass.read_bytes()
    cues = parse_strict_srt(original_srt)
    if not cues:
        raise RuntimeError("cannot regenerate subtitle pair from empty upstream SRT")
    converted = [
        ExportCue(cue.start, cue.end, cue.text) for cue in cues
    ]
    generated_srt, generated_ass = write_subtitle_pair_exclusive(
        converted, separate_base,
    )
    if parse_strict_srt(generated_srt.read_bytes()) != cues:
        raise RuntimeError("first-party SRT normalization altered original cues")
    inspected = grade_srt_ass(
        generated_srt, generated_ass, reference, native_blank_intervals,
    )
    if not (
        inspected["strict_srt_valid"]
        and inspected["valid_ass_matching_srt"]
    ):
        raise RuntimeError("first-party safe ASS normalization lost subtitle data")
    if (
        official_srt.read_bytes() != original_srt
        or official_ass.read_bytes() != original_ass
    ):
        raise RuntimeError("upstream exports changed during normalization")
    return {
        "performed_by_hse_not_official_upstream": True,
        "new_files_in_disposable_scratch_only": True,
        "original_official_srt_ass_unchanged": True,
        "both_formats_preserved_exact_text_and_wrap": True,
        "scores": inspected["scores"],
        "release_gate_passed": False,
    }


def main(
    languages: tuple[str, ...] = LANGUAGES,
    *,
    model_root: Path,
    manifest: Path,
) -> int:
    rows = []
    try:
        import importlib.metadata
        from rapid_videocr.main import RapidVideOCR, RapidVideOCRInput

        if importlib.metadata.version(PACKAGE) != VERSION:
            raise RuntimeError("official PyPI package version mismatch")

        with tempfile.TemporaryDirectory(prefix="hse-vsf-rv-official-") as td:
            root = Path(td)
            exe = unpack_pinned_archive(
                download_pinned_archive(root / "official-vsf.zip"), root / "native",
            )
            check_native_version(exe)
            for language in languages:
                folder = root / language
                folder.mkdir()
                pos, neg = folder / "caption.mp4", folder / "blank.mp4"
                encode_video(pos, language, style="white_two_line")
                encode_video(neg, language, style="white_two_line",
                             include_subtitles=False)
                src_sha = hashlib.sha256(pos.read_bytes()).hexdigest()
                blank_sha = hashlib.sha256(neg.read_bytes()).hexdigest()
                native, native_time = run_vsf(exe, pos, folder / "native-positive")
                blank, _ = run_vsf(exe, neg, folder / "native-negative")
                rgb = folder / "native-positive" / "RGBImages"
                if not rgb.is_dir():
                    raise RuntimeError("native VSF RGBImages directory missing")
                images = [
                    x for x in rgb.iterdir()
                    if x.is_file() and x.suffix.lower()
                    in {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
                ]
                if not 1 <= len(images) <= MAX_RGB_IMAGES:
                    raise RuntimeError("upstream keyframe-image count outside bound")
                output = folder / "official-export"
                output.mkdir()
                params = RapidVideOCRInput(
                    is_batch_rec=False, ocr_params=official_v5_params(
                        model_root, manifest, language,
                    ),
                )
                before = time.monotonic()
                upstream = RapidVideOCR(params)
                upstream(rgb, output, save_name="candidate")
                elapsed = time.monotonic() - before
                scores = grade_srt_ass(
                    output / "candidate.srt", output / "candidate.ass",
                    [
                        ScoreCue(start, end, "\n".join(_multiline_reference(text)))
                        for (start, end), text in zip(
                            EXPECTED_SPANS, VIDEO_TEXT[language],
                        )
                    ],
                    len(blank),
                )
                repaired = grade_lossless_hse_reexport(
                    output / "candidate.srt", output / "candidate.ass",
                    output / "hse-safe-normalized",
                    [
                        ScoreCue(start, end, "\n".join(_multiline_reference(text)))
                        for (start, end), text in zip(
                            EXPECTED_SPANS, VIDEO_TEXT[language],
                        )
                    ],
                    len(blank),
                )
                if (
                    hashlib.sha256(pos.read_bytes()).hexdigest() != src_sha
                    or hashlib.sha256(neg.read_bytes()).hexdigest() != blank_sha
                ):
                    raise RuntimeError("video inputs mutated by third-party OCR")
                row = {
                    "language": language,
                    "official_rapid_videocr_version": VERSION,
                    "pypi_wheel_sha256_pinned": True,
                    "native_vsf_version_and_sha_pinned": True,
                    "native_intervals": len(native),
                    "native_false_blank_intervals": len(blank),
                    "source_frames_from_actual_encoded_video": True,
                    "vsf_rgb_images_processed": len(images),
                    "official_rapid_videocr_ocr_seconds": round(elapsed, 2),
                    "native_timing_seconds": native_time,
                    "third_party_ocr_executed": True,
                    "upstream_vsf_process_wrapper_executed": False,
                    "upstream_output_only_inside_disposable_scratch": True,
                    "scoring": scores,
                    "hse_safe_reexport_from_official_srt": repaired,
                }
                rows.append(row)
                print("HSE_RAPIDVIDOCR_CASE " + json.dumps(row, sort_keys=True), flush=True)
    except Exception as exc:
        print("HSE_RAPIDVIDOCR_ERROR " + json.dumps({
            "error_class": type(exc).__name__,
            "cases_completed": len(rows),
        }), flush=True)
        return 4
    print("HSE_RAPIDVIDOCR_SUMMARY " + json.dumps({
        "schema": 1, "all_cases_completed": len(rows) == len(languages),
        "languages": list(languages),
        "official_package_version": VERSION,
        "source_caption_text_used_as_model_input": False,
        "actual_real_anime_tested": False,
        "no_existing_user_srt_overwritten": True,
        "no_production_files_or_models_modified": True,
        "upstream_native_launcher_disabled": True,
        "release_gate_passed": False,
    }, sort_keys=True), flush=True)
    return 0 if len(rows) == len(languages) else 4


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path,
                        default=Path("models/manifest.json"))
    parser.add_argument("--languages", default="ja,zh",
                        help="Explicit research subset: ko,en,ja,zh")
    options = parser.parse_args()
    try:
        chosen_languages = parse_research_languages(options.languages)
    except ValueError as exc:
        parser.error(str(exc))
    raise SystemExit(main(
        chosen_languages,
        model_root=options.model_dir,
        manifest=options.manifest,
    ))
