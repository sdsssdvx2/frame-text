"""The packaged local-frame acceptance is headless, private and fail-closed."""
from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

from hardsub_extractor import gui_entry
from hardsub_extractor.frozen_frame_review_acceptance import (
    run_frozen_frame_review_acceptance,
)

_GOOD = {
    "gui_worker_invoked": True, "flagged_cues": 1, "resumed_cues": 2,
    "jpeg_frames_verified": 6, "contact_sheets_verified": 2,
    "private_offline_html_reports_verified": 2,
    "trusted_windows_notepad_present": True,
    "notepad_process_started": False,
    "proofread_source_validated_without_editing": True,
    "originals_unchanged": True, "existing_evidence_preserved": True,
}


class FrozenFrameReviewAcceptanceTests(unittest.TestCase):
    def test_private_offline_report_rejects_external_or_missing_frame_markup(self):
        from hardsub_extractor.frozen_frame_review_acceptance import (
            _inspect_private_review_html,
        )
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            row = {"review_cues": [{"cue": 1, "frames": [
                {"image": "cue-00001-early.jpg"},
                {"image": "cue-00001-late.jpg"},
            ]}]}
            valid = (
                '<!doctype html><html><head><meta charset="utf-8">'
                '<meta http-equiv="Content-Security-Policy" content="'
                "default-src 'none'; script-src 'none'; connect-src 'none';"
                '"></head><body><section><h2>자막 1번</h2>'
                "固定電視台台標文字第一句對話"
                '<img src="cue-00001-early.jpg"><img src="cue-00001-late.jpg">'
                '</section></body></html>'
            )
            (root / "review.html").write_text(
                valid + (" " * 200), encoding="utf-8"
            )
            _inspect_private_review_html(root, (1,), row)
            for altered in (
                valid.replace("script-src 'none'", "script-src *"),
                valid.replace("cue-00001-late.jpg", "https://example.invalid"),
                valid.replace("第一句對話", ""),
                valid.replace('<img src="cue-00001-early.jpg">', ""),
            ):
                (root / "review.html").write_text(
                    altered + (" " * 200), encoding="utf-8"
                )
                with self.assertRaises(AssertionError):
                    _inspect_private_review_html(root, (1,), row)

    def test_nonfrozen_fails_without_creating_input_media(self):
        with tempfile.TemporaryDirectory() as td:
            report = Path(td) / "no.json"
            with mock.patch.object(sys, "frozen", False, create=True), mock.patch(
                "hardsub_extractor.frozen_frame_review_acceptance._fixture",
            ) as fixture:
                code = run_frozen_frame_review_acceptance(report)
            self.assertEqual(code, 3)
            fixture.assert_not_called()
            data = json.loads(report.read_text("utf-8"))
            self.assertFalse(data["is_frozen"])
            self.assertFalse(data["ok"])
            self.assertEqual(data["jpeg_frames_verified"], 0)

    def test_success_is_metadata_only_and_cleans_temporary_media(self):
        with tempfile.TemporaryDirectory() as td:
            report = Path(td) / "result.json"
            with mock.patch.object(sys, "frozen", True, create=True), mock.patch(
                "hardsub_extractor.frozen_frame_review_acceptance._verify",
                return_value=dict(_GOOD),
            ):
                self.assertEqual(run_frozen_frame_review_acceptance(report), 0)
            raw = report.read_text("utf-8")
            data = json.loads(raw)
            self.assertTrue(data["ok"])
            self.assertTrue(data["temporary_inputs_removed"])
            self.assertEqual(data["jpeg_frames_verified"], 6)
            self.assertEqual(data["private_offline_html_reports_verified"], 2)
            self.assertTrue(data["trusted_windows_notepad_present"])
            self.assertFalse(data["notepad_process_started"])
            self.assertNotIn(str(Path(td)), raw)

    def test_private_exception_contents_never_enter_report(self):
        with tempfile.TemporaryDirectory() as td:
            report = Path(td) / "error.json"
            with mock.patch.object(sys, "frozen", True, create=True), mock.patch(
                "hardsub_extractor.frozen_frame_review_acceptance._verify",
                side_effect=RuntimeError("DO_NOT_LEAK_PRIVATE_FILENAME"),
            ):
                self.assertEqual(run_frozen_frame_review_acceptance(report), 3)
            raw = report.read_text("utf-8")
            self.assertNotIn("DO_NOT_LEAK_PRIVATE_FILENAME", raw)
            self.assertEqual(json.loads(raw)["failure_type"], "RuntimeError")

    def test_preexisting_report_is_never_overwritten(self):
        with tempfile.TemporaryDirectory() as td:
            report = Path(td) / "existing.json"
            report.write_bytes(b"ORIGINAL_USER_FILE")
            with mock.patch.object(sys, "frozen", False, create=True):
                with self.assertRaises(FileExistsError):
                    run_frozen_frame_review_acceptance(report)
            self.assertEqual(report.read_bytes(), b"ORIGINAL_USER_FILE")

    def test_headless_dispatch_only_on_explicit_flag(self):
        with mock.patch(
            "hardsub_extractor.frozen_frame_review_acceptance.run_frozen_frame_review_acceptance",
            return_value=0,
        ) as run:
            self.assertIsNone(gui_entry._run_cli_frozen_frame_review([]))
            self.assertEqual(gui_entry._run_cli_frozen_frame_review([
                "--frozen-frame-review-acceptance", "--review-acceptance-report", "test.json"
            ]), 0)
            run.assert_called_once_with("test.json")

    def test_entry_skips_tk_when_frozen_review_requested(self):
        with mock.patch(
            "hardsub_extractor.frozen_frame_review_acceptance.run_frozen_frame_review_acceptance",
            return_value=0,
        ), mock.patch.object(sys, "argv", [
            "HardSubtitleExtractor.exe", "--frozen-frame-review-acceptance",
            "--review-acceptance-report", "result.json",
        ]), mock.patch("hardsub_extractor.gui.main") as gui:
            with self.assertRaises(SystemExit) as outcome:
                gui_entry.main()
            self.assertEqual(outcome.exception.code, 0)
            gui.assert_not_called()


if __name__ == "__main__":
    unittest.main()
