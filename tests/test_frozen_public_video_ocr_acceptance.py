"""Frozen source-language OCR fallback acceptance contracts; no actual network."""
from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

from hardsub_extractor import gui_entry
from hardsub_extractor.frozen_public_video_ocr_acceptance import (
    run_frozen_public_video_ocr_acceptance,
)


class FrozenPublicFallbackAcceptanceTests(unittest.TestCase):
    def test_source_python_cannot_report_frozen_product_success(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "report.json"
            with mock.patch.object(sys, "frozen", False, create=True), mock.patch(
                "hardsub_extractor.frozen_public_video_ocr_acceptance._case",
            ) as case:
                code = run_frozen_public_video_ocr_acceptance(
                    Path(td) / "missing", path,
                )
            self.assertEqual(code, 3)
            case.assert_not_called()
            report = json.loads(path.read_text("utf-8"))
            self.assertFalse(report["is_frozen"])
            self.assertFalse(report["ok"])

    def test_frozen_four_language_reports_no_text_or_paths(self):
        with tempfile.TemporaryDirectory() as td:
            report_path = Path(td) / "report.json"
            row = {
                "language": "ko", "exact_cues": 2,
                "temporary_video_removed": True,
                "read_only_visual_resume_verified": True,
                "read_only_soft_resume_verified": True,
            }
            with mock.patch.object(sys, "frozen", True, create=True), mock.patch(
                "hardsub_extractor.frozen_public_video_ocr_acceptance._bounded_local_fixture",
                side_effect=lambda folder, language: folder / (language + ".mp4"),
            ), mock.patch(
                "hardsub_extractor.frozen_public_video_ocr_acceptance._case",
                side_effect=lambda language, source, output: {
                    **row, "language": language,
                },
            ) as case:
                self.assertEqual(
                    run_frozen_public_video_ocr_acceptance(td, report_path),
                    0,
                )
            self.assertEqual(case.call_count, 4)
            raw = report_path.read_text("utf-8")
            report = json.loads(raw)
            self.assertTrue(report["ok"])
            self.assertTrue(report["is_frozen"])
            self.assertEqual(report["exact_total"], 8)
            self.assertEqual(len(report["languages"]), 4)
            self.assertTrue(report["temporary_outputs_removed"])
            self.assertNotIn(str(Path(td)), raw)
            self.assertNotIn("안녕하세요", raw)
            self.assertEqual(report["transport"], "synthetic_local_no_network")

    def test_frozen_gate_requires_verified_resume_for_each_language(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "gate.json"
            def fake_case(language, source, output):
                return {
                    "language": language,
                    "exact_cues": 2,
                    "temporary_video_removed": True,
                    "read_only_visual_resume_verified": language != "zh",
                    "read_only_soft_resume_verified": True,
                }
            with mock.patch.object(sys, "frozen", True, create=True), mock.patch(
                "hardsub_extractor.frozen_public_video_ocr_acceptance._bounded_local_fixture",
                side_effect=lambda folder, language: folder / (language + ".mp4"),
            ), mock.patch(
                "hardsub_extractor.frozen_public_video_ocr_acceptance._case",
                side_effect=fake_case,
            ):
                code = run_frozen_public_video_ocr_acceptance(td, path)
            self.assertEqual(code, 3)
            report = json.loads(path.read_text("utf-8"))
            self.assertFalse(report["ok"])
            self.assertEqual(report["exact_total"], 8)
            self.assertFalse(report["languages"][-1]["read_only_visual_resume_verified"])

    def test_failure_serialization_never_leaks_sensitive_media_url(self):
        with tempfile.TemporaryDirectory() as td:
            report_path = Path(td) / "report.json"
            secret = "https://video.private.example/path?auth=DO_NOT_PRINT"
            with mock.patch.object(sys, "frozen", True, create=True), mock.patch(
                "hardsub_extractor.frozen_public_video_ocr_acceptance._bounded_local_fixture",
                side_effect=lambda folder, language: folder / (language + ".mp4"),
            ), mock.patch(
                "hardsub_extractor.frozen_public_video_ocr_acceptance._case",
                side_effect=RuntimeError(secret),
            ):
                self.assertEqual(run_frozen_public_video_ocr_acceptance(td, report_path), 3)
            raw = report_path.read_text("utf-8")
            self.assertNotIn(secret, raw)
            self.assertEqual(json.loads(raw)["failure_type"], "RuntimeError")

    def test_existing_result_report_is_never_overwritten(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "report.json"
            path.write_bytes(b"USER_PRIVATE_REPORT")
            with mock.patch.object(sys, "frozen", False, create=True):
                with self.assertRaises(FileExistsError):
                    run_frozen_public_video_ocr_acceptance(td, path)
            self.assertEqual(path.read_bytes(), b"USER_PRIVATE_REPORT")

    def test_headless_dispatch_runs_only_on_exact_acceptance_flag(self):
        with mock.patch(
            "hardsub_extractor.frozen_public_video_ocr_acceptance.run_frozen_public_video_ocr_acceptance",
            return_value=0,
        ) as run:
            self.assertIsNone(gui_entry._run_cli_frozen_public_video_ocr([]))
            self.assertIsNone(gui_entry._run_cli_frozen_public_video_ocr([
                "--public-ocr-report", "ignored.json",
            ]))
            rc = gui_entry._run_cli_frozen_public_video_ocr([
                "--public-video-ocr-acceptance", "--ocr-fixture-dir",
                "synthetic-fixtures", "--public-ocr-report", "summary.json",
            ])
        self.assertEqual(rc, 0)
        run.assert_called_once_with("synthetic-fixtures", "summary.json")

    def test_headless_dispatch_exception_safely_preserves_existing_report(self):
        with tempfile.TemporaryDirectory() as td:
            report = Path(td) / "report.json"
            report.write_bytes(b"USER_REPORT")
            with mock.patch(
                "hardsub_extractor.frozen_public_video_ocr_acceptance.run_frozen_public_video_ocr_acceptance",
                side_effect=RuntimeError("secret-URL-should-not-print"),
            ):
                rc = gui_entry._run_cli_frozen_public_video_ocr([
                    "--public-video-ocr-acceptance", "--ocr-fixture-dir",
                    "synthetic-fixtures", "--public-ocr-report", str(report),
                ])
            self.assertEqual(rc, 3)
            self.assertEqual(report.read_bytes(), b"USER_REPORT")


if __name__ == "__main__":
    unittest.main()
