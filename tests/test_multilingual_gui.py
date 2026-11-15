"""Headless GUI language-selection contracts; no network, model load or Tk root."""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest import mock
import queue
import tempfile
import unittest

from hardsub_extractor.gui import ExtractorGUI


class Var:
    def __init__(self, value):
        self.value = value

    def get(self):
        return self.value


class MultilingualGuiTests(unittest.TestCase):
    @staticmethod
    def gui(root):
        gui = ExtractorGUI.__new__(ExtractorGUI)
        gui.source_mode_var = Var("local")
        gui.url_var = Var(str(root / "video.mp4"))
        gui.target_var = Var("")
        gui.out_var = Var(str(root / "output" / "subtitles"))
        gui.ocr_language_var = Var("ja")
        gui.events = queue.Queue()
        return gui

    def test_local_video_explicit_ocr_language_is_stored_before_work(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root / "video.mp4").write_bytes(b"synthetic")
            gui = self.gui(root)
            self.assertEqual(gui._validate()[0], "local")
            self.assertEqual(gui._requested_ocr_language, "ja")

    def test_invalid_foreign_ocr_language_rejected_before_output_created(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root / "video.mp4").write_bytes(b"synthetic")
            gui = self.gui(root)
            gui.ocr_language_var = Var("../../auto")
            with self.assertRaises(ValueError):
                gui._validate()
            self.assertFalse((root / "output").exists())

    def test_local_video_uses_foreign_recognizer_without_epsite_network(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root / "video.mp4").write_bytes(b"synthetic-existing-video")
            gui = self.gui(root)
            output = SimpleNamespace(
                cue_count=2, srt_path=root / "a.srt", ass_path=root / "a.ass",
            )
            with mock.patch(
                "hardsub_extractor.gui.acquire_local_embedded_text", return_value=None
            ), mock.patch(
                "hardsub_extractor.gui.acquire_local_sidecar_text", return_value=None
            ), mock.patch(
                "hardsub_extractor.gui.acquire_local_embedded_image", return_value=None
            ), mock.patch(
                "hardsub_extractor.gui.bundled_model_dir", return_value=root
            ), mock.patch(
                "hardsub_extractor.gui.RapidOCRAdapter", return_value="model"
            ) as recog, mock.patch(
                "hardsub_extractor.gui.AdaptiveOCREngine", return_value="ocr"
            ), mock.patch(
                "hardsub_extractor.gui.extract_local_video", return_value=output
            ) as extract, mock.patch(
                "hardsub_extractor.gui.AutoEpSiteClient"
            ) as anime:
                gui._worker(
                    "local", str(root / "video.mp4"), None, root / "out",
                    False, None, "ja",
                )
            self.assertEqual(recog.call_args.kwargs["language"], "ja")
            self.assertEqual(
                recog.call_args.kwargs["model_dir"], str(root),
            )
            extract.assert_called_once()
            anime.assert_not_called()
            self.assertTrue(any(kind == "done" for kind, _ in gui.events.queue))

    def test_epsite_loads_foreign_model_only_if_soft_track_missing(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            gui = self.gui(root)
            output = SimpleNamespace(
                mode="soft", srt_path=root / "a.srt", ass_path=root / "a.ass",
            )
            with mock.patch(
                "hardsub_extractor.gui.AutoEpSiteClient"
            ), mock.patch(
                "hardsub_extractor.gui.extract_epsite", return_value=output
            ) as source, mock.patch(
                "hardsub_extractor.gui.RapidOCRAdapter"
            ) as recog:
                gui._worker(
                    "epsite", "https://episode.example/episode/1",
                    None, root / "out", False, None, "zh",
                )
                factory = source.call_args.kwargs["ocr_factory"]
                self.assertTrue(callable(factory))
                recog.assert_not_called()
            # The lazy factory is passed through uninvoked when a soft track
            # is available; the GUI is not allowed to download OCR models.

    def test_original_korean_worker_defaults_remain_unchanged(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            source = root / "legacy.mp4"
            source.write_bytes(b"synthetic-existing-video")
            gui = self.gui(root)
            output = SimpleNamespace(
                cue_count=1, srt_path=root / "a.srt", ass_path=root / "a.ass",
            )
            with mock.patch(
                "hardsub_extractor.gui.acquire_local_embedded_text", return_value=None
            ), mock.patch(
                "hardsub_extractor.gui.acquire_local_sidecar_text", return_value=None
            ), mock.patch(
                "hardsub_extractor.gui.acquire_local_embedded_image", return_value=None
            ), mock.patch(
                "hardsub_extractor.gui.bundled_model_dir", return_value=root
            ), mock.patch(
                "hardsub_extractor.gui.RapidOCRAdapter"
            ) as recog, mock.patch(
                "hardsub_extractor.gui.AdaptiveOCREngine"
            ), mock.patch(
                "hardsub_extractor.gui.extract_local_video", return_value=output
            ):
                gui._worker(
                    "local", str(source), None, root / "out", False,
                )
            self.assertEqual(recog.call_args.kwargs["language"], "ko")


if __name__ == "__main__":
    unittest.main()
