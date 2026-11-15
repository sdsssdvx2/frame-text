"""No-network regressions for early output collision rejection."""
from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from hardsub_extractor import cli
from hardsub_extractor.extraction_progress import ExtractionMonitor
from hardsub_extractor.pipeline import extract
from hardsub_extractor.writers import ensure_subtitle_pair_available


class EarlyOutputCollisionTests(unittest.TestCase):
    def test_fresh_output_does_not_create_any_file(self):
        with tempfile.TemporaryDirectory() as td:
            base = Path(td) / "fresh"
            ensure_subtitle_pair_available(base)
            self.assertEqual(list(Path(td).iterdir()), [])

    def test_existing_either_subtitle_is_preserved(self):
        for extension in (".srt", ".ass"):
            with self.subTest(extension=extension), tempfile.TemporaryDirectory() as td:
                base = Path(td) / "result"
                file = base.with_suffix(extension)
                file.write_bytes(b"original user data")
                with self.assertRaises(FileExistsError):
                    ensure_subtitle_pair_available(base)
                self.assertEqual(file.read_bytes(), b"original user data")
                self.assertEqual(len(list(Path(td).iterdir())), 1)

    def test_dangling_link_is_not_mistaken_for_fresh_output(self):
        with tempfile.TemporaryDirectory() as td:
            base = Path(td) / "result"
            link = base.with_suffix(".srt")
            try:
                link.symlink_to(Path(td) / "not-there")
            except (OSError, NotImplementedError):
                self.skipTest("creating symlinks is not permitted here")
            self.assertTrue(link.is_symlink())
            self.assertFalse(link.exists())
            with self.assertRaises(FileExistsError):
                ensure_subtitle_pair_available(base)
            self.assertTrue(link.is_symlink())

    def test_pipeline_stops_before_first_decoder_frame(self):
        with tempfile.TemporaryDirectory() as td:
            base = Path(td) / "result"
            original = base.with_suffix(".ass")
            original.write_bytes(b"original")
            with mock.patch(
                "hardsub_extractor.pipeline.FFmpegPTSFrameSource",
                side_effect=AssertionError("should not decode"),
            ):
                with self.assertRaises(FileExistsError):
                    extract("source.mkv", object(), base, streaming=True,
                            monitor=ExtractionMonitor(exclusive_outputs=True))
            self.assertEqual(original.read_bytes(), b"original")
            self.assertFalse(base.with_suffix(".srt").exists())

    def test_cli_stops_before_model_initialization(self):
        with tempfile.TemporaryDirectory() as td:
            base = Path(td) / "result"
            original = base.with_suffix(".srt")
            original.write_bytes(b"original")
            argv = ["hard-sub-extract", "source.mkv", "--hard-only",
                    "--out", str(base)]
            with mock.patch.object(sys, "argv", argv), mock.patch.object(
                cli, "_ocr", side_effect=AssertionError("should not load OCR")
            ), mock.patch.object(
                cli, "extract", side_effect=AssertionError("should not decode")
            ):
                with self.assertRaises(FileExistsError):
                    cli.main()
            self.assertEqual(original.read_bytes(), b"original")
            self.assertFalse(base.with_suffix(".ass").exists())


if __name__ == "__main__":
    unittest.main()
