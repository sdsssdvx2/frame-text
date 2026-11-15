"""Offline safety and execution contracts for genuine frozen multilingual video QA."""
from __future__ import annotations

import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest import mock

from hardsub_extractor import gui_entry
from hardsub_extractor.frozen_multilingual_ocr_acceptance import (
    EXPECTED_TEXT, LANGUAGES, EXPECTED_SPANS, _bounded_local_fixture,
    _case, run_frozen_multilingual_ocr_acceptance,
)
from hardsub_extractor.model_assets import ModelAssetError
from hardsub_extractor.models import Cue
from hardsub_extractor.writers import write_ass, write_srt


def make_fake_mp4(path: Path):
    path.write_bytes(b"\x00\x00\x00\x18ftypisom" + b"\0" * 2048)


class FrozenMultilingualAcceptanceTests(unittest.TestCase):
    def test_only_local_existing_small_iso_bmff_files_are_allowed(self):
        with TemporaryDirectory() as td:
            root = Path(td)
            make_fake_mp4(root / "ja.mp4")
            self.assertEqual(_bounded_local_fixture(root, "ja").name, "ja.mp4")
            with self.assertRaises(ValueError):
                _bounded_local_fixture(root, "en")
            (root / "en.mp4").write_bytes(b"https://example.invalid/stream")
            with self.assertRaisesRegex(ValueError, "out of bounds"):
                _bounded_local_fixture(root, "en")
            (root / "en.mp4").write_bytes(b"x" * 2200)
            with self.assertRaisesRegex(ValueError, "ISO BMFF"):
                _bounded_local_fixture(root, "en")

    def test_validation_rejects_symlink_input_when_supported(self):
        with TemporaryDirectory() as td:
            root = Path(td)
            make_fake_mp4(root / "ko.mp4")
            link = root / "ja.mp4"
            try:
                link.symlink_to(root / "ko.mp4")
            except (OSError, NotImplementedError):
                self.skipTest("Windows runner does not permit file symlinks")
            with self.assertRaises(ValueError):
                _bounded_local_fixture(root, "ja")

    def test_success_report_tracks_four_languages_without_original_text_or_paths(self):
        with TemporaryDirectory() as td:
            root = Path(td)
            for language in LANGUAGES:
                make_fake_mp4(root / (language + ".mp4"))
            report = root / "report.json"
            def fake_case(
                lang, fixture, out, models, manifest, *,
                inference_engine="onnxruntime",
            ):
                self.assertEqual(fixture.name, lang + ".mp4")
                self.assertTrue(out.is_dir())
                return {
                    "language": lang,
                    "inference_engine": inference_engine,
                    "verified_model": True,
                    "real_ffmpeg_video_frames": True,
                    "original_text_exact_cues": 2, "srt_ass_verified": True,
                }
            with mock.patch(
                "hardsub_extractor.frozen_multilingual_ocr_acceptance._case",
                side_effect=fake_case,
            ), mock.patch.object(sys, "frozen", True, create=True):
                rc = run_frozen_multilingual_ocr_acceptance(root, report)
            self.assertEqual(rc, 0)
            data = json.loads(report.read_text(encoding="utf-8"))
            self.assertTrue(data["ok"])
            self.assertTrue(data["is_frozen"])
            self.assertTrue(data["temporary_outputs_removed"])
            self.assertEqual(data["original_text_exact_total"], 8)
            self.assertEqual([x["language"] for x in data["languages"]], list(LANGUAGES))
            self.assertTrue(data["no_external_requests"])
            contents = report.read_text(encoding="utf-8")
            self.assertNotIn(str(root), contents)
            for cues in EXPECTED_TEXT.values():
                for text in cues:
                    self.assertNotIn(text, contents)

    def test_error_is_sanitized_and_failed_run_never_reports_success(self):
        with TemporaryDirectory() as td:
            root = Path(td)
            for language in LANGUAGES:
                make_fake_mp4(root / (language + ".mp4"))
            report = root / "result.json"
            with mock.patch(
                "hardsub_extractor.frozen_multilingual_ocr_acceptance._case",
                side_effect=RuntimeError("private URL https://example.invalid/secret"),
            ):
                rc = run_frozen_multilingual_ocr_acceptance(root, report)
            self.assertNotEqual(rc, 0)
            content = report.read_text(encoding="utf-8")
            data = json.loads(content)
            self.assertFalse(data["ok"])
            self.assertEqual(data["failure_type"], "RuntimeError")
            self.assertNotIn("private URL", content)
            self.assertNotIn("secret", content)

    def test_video_acceptance_computes_real_original_text_and_srt_ass_timing(self):
        with TemporaryDirectory() as td:
            root = Path(td)
            def fake_extract(_input, _ocr, base):
                cues = [
                    Cue(*EXPECTED_SPANS[i], EXPECTED_TEXT["en"][i])
                    for i in range(2)
                ]
                srt = write_srt(cues, base.with_suffix(".srt"))
                ass = write_ass(cues, base.with_suffix(".ass"))
                return SimpleNamespace(cue_count=2, srt_path=srt, ass_path=ass)
            adapter = object()
            engine = SimpleNamespace(total_calls=33, retry_calls=18)
            with mock.patch(
                "hardsub_extractor.frozen_multilingual_ocr_acceptance.resolve_model_paths"
            ) as verify, mock.patch(
                "hardsub_extractor.frozen_multilingual_ocr_acceptance.RapidOCRAdapter",
                return_value=adapter,
            ), mock.patch(
                "hardsub_extractor.frozen_multilingual_ocr_acceptance.AdaptiveOCREngine",
                return_value=engine,
            ), mock.patch(
                "hardsub_extractor.frozen_multilingual_ocr_acceptance.extract",
                side_effect=fake_extract,
            ):
                result = _case("en", root / "en.mp4", root, root, root / "manifest.json")
            self.assertEqual(result["original_text_exact_cues"], 2)
            self.assertEqual(result["max_timing_error_seconds"], 0)
            self.assertLessEqual(result["max_srt_ass_timing_delta_seconds"], 0.005)
            self.assertEqual(verify.call_args.kwargs["language"], "en")
            self.assertTrue(verify.call_args.kwargs["verify"])

    def test_wrong_text_is_rejected_not_counted_as_success(self):
        with TemporaryDirectory() as td:
            root = Path(td)
            def wrong_extract(_source, _ocr, base):
                cues = [
                    Cue(*EXPECTED_SPANS[0], "A completely different caption"),
                    Cue(*EXPECTED_SPANS[1], EXPECTED_TEXT["en"][1]),
                ]
                return SimpleNamespace(
                    cue_count=2,
                    srt_path=write_srt(cues, base.with_suffix(".srt")),
                    ass_path=write_ass(cues, base.with_suffix(".ass")),
                )
            with mock.patch(
                "hardsub_extractor.frozen_multilingual_ocr_acceptance.resolve_model_paths"
            ), mock.patch(
                "hardsub_extractor.frozen_multilingual_ocr_acceptance.RapidOCRAdapter"
            ), mock.patch(
                "hardsub_extractor.frozen_multilingual_ocr_acceptance.AdaptiveOCREngine"
            ), mock.patch(
                "hardsub_extractor.frozen_multilingual_ocr_acceptance.extract",
                side_effect=wrong_extract,
            ):
                with self.assertRaisesRegex(AssertionError, "not exact"):
                    _case("en", root / "en.mp4", root, root, root / "manifest.json")

    def test_missing_model_fails_closed_before_video_extraction(self):
        with TemporaryDirectory() as td:
            root = Path(td)
            with mock.patch(
                "hardsub_extractor.frozen_multilingual_ocr_acceptance.resolve_model_paths",
                side_effect=ModelAssetError("missing verified Japanese OCR"),
            ), mock.patch(
                "hardsub_extractor.frozen_multilingual_ocr_acceptance.extract",
            ) as video:
                with self.assertRaises(ModelAssetError):
                    _case("ja", root / "ja.mp4", root, root, root / "manifest.json")
            video.assert_not_called()

    def test_frozen_headless_dispatch_never_starts_gui(self):
        with TemporaryDirectory() as td:
            report = str(Path(td) / "acceptance.json")
            with mock.patch.object(
                sys, "argv",
                ["HardSubtitleExtractor.exe", "--multilingual-ocr-acceptance",
                 "--ocr-fixture-dir", td, "--ocr-acceptance-report", report],
            ), mock.patch(
                "hardsub_extractor.frozen_multilingual_ocr_acceptance.run_frozen_multilingual_ocr_acceptance",
                return_value=0,
            ) as acceptance, mock.patch("hardsub_extractor.gui.main") as gui:
                with self.assertRaises(SystemExit) as ended:
                    gui_entry.main()
            self.assertEqual(ended.exception.code, 0)
            acceptance.assert_called_once_with(
                td, report, inference_engine="onnxruntime",
            )
            gui.assert_not_called()

    def test_fatal_error_entry_preserves_only_exception_class(self):
        with TemporaryDirectory() as td:
            report = Path(td) / "fatal.json"
            with mock.patch(
                "hardsub_extractor.frozen_multilingual_ocr_acceptance.run_frozen_multilingual_ocr_acceptance",
                side_effect=RuntimeError("file:///hidden-profile/secret"),
            ):
                rc = gui_entry._run_cli_multilingual_ocr_acceptance([
                    "--multilingual-ocr-acceptance",
                    "--ocr-fixture-dir", td,
                    "--ocr-acceptance-report", str(report),
                ])
            self.assertEqual(rc, 3)
            data = json.loads(report.read_text(encoding="utf-8"))
            self.assertFalse(data["ok"])
            self.assertEqual(data["failure_type"], "RuntimeError")
            self.assertNotIn("hidden-profile", report.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
