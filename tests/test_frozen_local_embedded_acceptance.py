"""Offline contracts for the headless frozen local-embedded acceptance gate."""
from __future__ import annotations

from pathlib import Path
import json
import sys
import tempfile
import unittest
from unittest import mock

from hardsub_extractor import gui_entry
from hardsub_extractor.frozen_local_embedded_acceptance import (
    run_frozen_local_embedded_acceptance,
)


class FrozenLocalEmbeddedAcceptanceTests(unittest.TestCase):
    def test_nonfrozen_source_is_never_reported_as_product_success(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "result.json"
            with mock.patch.object(sys, "frozen", False, create=True), mock.patch(
                "hardsub_extractor.frozen_local_embedded_acceptance._create_fixture",
            ) as fixture:
                code = run_frozen_local_embedded_acceptance(path)
            self.assertEqual(code, 3)
            fixture.assert_not_called()
            data = json.loads(path.read_text("utf-8"))
            self.assertFalse(data["is_frozen"])
            self.assertFalse(data["ok"])
            self.assertFalse(data["temporary_outputs_removed"])
            self.assertFalse(data["wrong_language_blocked"])
            self.assertEqual(data["exact_cue_count"], 0)

    def test_frozen_acceptance_collects_only_metadata_and_cleans_outputs(self):
        with tempfile.TemporaryDirectory() as td:
            target = Path(td) / "frozen.json"
            rows = [
                {"language": "ko", "exact_cue_count": 2, "srt_ass_verified": True},
                {"language": "en", "exact_cue_count": 2, "srt_ass_verified": True},
            ]
            with mock.patch.object(sys, "frozen", True, create=True), mock.patch(
                "hardsub_extractor.frozen_local_embedded_acceptance._verify_real_product",
                return_value=rows,
            ):
                self.assertEqual(run_frozen_local_embedded_acceptance(target), 0)
            body = target.read_text("utf-8")
            data = json.loads(body)
            self.assertTrue(data["is_frozen"])
            self.assertTrue(data["ok"])
            self.assertEqual(data["exact_cue_count"], 4)
            self.assertTrue(data["temporary_outputs_removed"])
            self.assertTrue(data["wrong_language_blocked"])
            self.assertNotIn(str(Path(td)), body)
            self.assertNotIn("안녕하세요", body)

    def test_frozen_failure_report_does_not_leak_private_exception(self):
        with tempfile.TemporaryDirectory() as td:
            target = Path(td) / "result.json"
            secret = "https://secret.example.local/path?token=DO_NOT_LOG"
            with mock.patch.object(sys, "frozen", True, create=True), mock.patch(
                "hardsub_extractor.frozen_local_embedded_acceptance._verify_real_product",
                side_effect=RuntimeError(secret),
            ):
                self.assertEqual(run_frozen_local_embedded_acceptance(target), 3)
            body = target.read_text("utf-8")
            self.assertNotIn(secret, body)
            self.assertEqual(json.loads(body)["failure_type"], "RuntimeError")

    def test_existing_acceptance_report_is_not_replaced(self):
        with tempfile.TemporaryDirectory() as td:
            target = Path(td) / "result.json"
            target.write_bytes(b"USER_REPORT_DO_NOT_REPLACE")
            with mock.patch.object(sys, "frozen", False, create=True):
                with self.assertRaises(FileExistsError):
                    run_frozen_local_embedded_acceptance(target)
            self.assertEqual(target.read_bytes(), b"USER_REPORT_DO_NOT_REPLACE")

    def test_gui_entry_dispatch_does_not_open_tk_or_browser(self):
        with mock.patch(
            "hardsub_extractor.frozen_local_embedded_acceptance.run_frozen_local_embedded_acceptance",
            return_value=0,
        ) as run:
            code = gui_entry._run_cli_local_embedded_acceptance([
                "--local-embedded-acceptance",
                "--local-embedded-report", "temporary-report.json",
            ])
        self.assertEqual(code, 0)
        run.assert_called_once_with("temporary-report.json")

    def test_dispatch_without_exact_flag_leaves_normal_gui_routing_alone(self):
        with mock.patch(
            "hardsub_extractor.frozen_local_embedded_acceptance.run_frozen_local_embedded_acceptance",
        ) as run:
            self.assertIsNone(gui_entry._run_cli_local_embedded_acceptance([]))
            self.assertIsNone(gui_entry._run_cli_local_embedded_acceptance([
                "--local-embedded-report", "irrelevant.json",
            ]))
            run.assert_not_called()

    def test_dispatch_exception_preserves_preexisting_report(self):
        with tempfile.TemporaryDirectory() as td:
            report = Path(td) / "report.json"
            report.write_bytes(b"PRESERVE_ORIGINAL")
            with mock.patch(
                "hardsub_extractor.frozen_local_embedded_acceptance.run_frozen_local_embedded_acceptance",
                side_effect=RuntimeError("secret media url"),
            ):
                code = gui_entry._run_cli_local_embedded_acceptance([
                    "--local-embedded-acceptance",
                    "--local-embedded-report", str(report),
                ])
            self.assertEqual(code, 3)
            self.assertEqual(report.read_bytes(), b"PRESERVE_ORIGINAL")


if __name__ == "__main__":
    unittest.main()
