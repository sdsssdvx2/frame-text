"""Frozen caption acceptance dispatch contracts, offline and without Tk."""
from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

from hardsub_extractor import gui_entry

PUBLIC = "https://www.youtube.com/watch?v=unit_example"


class FrozenCaptionEntryTests(unittest.TestCase):
    def _runtime(self, root: Path) -> Path:
        path = root / "bin" / "yt-dlp.exe"
        path.parent.mkdir()
        path.write_bytes(b"placeholder for injected command runner")
        return path

    def test_no_flag_leaves_existing_gui_entry_unchanged(self):
        self.assertIsNone(gui_entry._run_cli_caption_acceptance([]))

    def test_dispatch_uses_bundled_executable_and_never_opens_gui(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            binary = self._runtime(root)
            report = root / "report.json"
            argv = [
                "--caption-acceptance", "--caption-url", PUBLIC,
                "--caption-language", "ko", "--caption-report", str(report),
            ]
            with mock.patch("hardsub_extractor.runtime_paths.bundle_root", return_value=root), \
                 mock.patch("hardsub_extractor.public_caption_diagnostics.main", return_value=0) as diagnostic, \
                 mock.patch("hardsub_extractor.gui.main") as gui, \
                 mock.patch.object(sys, "argv", ["HardSubtitleExtractor.exe", *argv]):
                with self.assertRaises(SystemExit) as ended:
                    gui_entry.main()
            self.assertEqual(ended.exception.code, 0)
            gui.assert_not_called()
            args = diagnostic.call_args.args[0]
            self.assertEqual(args[args.index("--yt-dlp") + 1], str(binary))
            self.assertIn("--require-download", args)
            self.assertEqual(args[args.index("--languages") + 1], "ko")

    def test_fatal_provider_detail_is_redacted_from_report(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            self._runtime(root)
            report = root / "result.json"
            with mock.patch("hardsub_extractor.runtime_paths.bundle_root", return_value=root), \
                 mock.patch(
                     "hardsub_extractor.public_caption_diagnostics.main",
                     side_effect=RuntimeError("sensitive media URL must not be logged"),
                 ):
                rc = gui_entry._run_cli_caption_acceptance([
                    "--caption-acceptance", "--caption-url", PUBLIC,
                    "--caption-language", "en", "--caption-report", str(report),
                ])
            self.assertEqual(rc, 3)
            output = report.read_text(encoding="utf-8")
            self.assertEqual(json.loads(output)["status"], "diagnostic_failed")
            self.assertNotIn("sensitive", output)
            self.assertNotIn(PUBLIC, output)

    def test_missing_bundle_reports_failure_without_probing(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            report = root / "failure.json"
            with mock.patch("hardsub_extractor.runtime_paths.bundle_root", return_value=root), \
                 mock.patch("hardsub_extractor.public_caption_diagnostics.main") as diagnostic:
                rc = gui_entry._run_cli_caption_acceptance([
                    "--caption-acceptance", "--caption-url", PUBLIC,
                    "--caption-language", "ko", "--caption-report", str(report),
                ])
            self.assertEqual(rc, 3)
            diagnostic.assert_not_called()
            self.assertEqual(json.loads(report.read_text())["files_verified"], False)


if __name__ == "__main__":
    unittest.main()
