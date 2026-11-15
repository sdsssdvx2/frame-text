"""GUI/CLI local sidecar routing must stay subtitle-first and model-free."""
from __future__ import annotations

from contextlib import redirect_stdout
import io
from pathlib import Path
import queue
from types import SimpleNamespace
import unittest
from unittest import mock

from hardsub_extractor import cli
from hardsub_extractor.gui import ExtractorGUI


def result(mode):
    return SimpleNamespace(
        mode=mode,
        cue_count=2,
        srt_path=Path("output.srt"),
        ass_path=Path("output.ass"),
        provenance_path=Path("output.source.json"),
    )


class LocalSidecarRoutingTests(unittest.TestCase):
    def gui(self):
        gui = ExtractorGUI.__new__(ExtractorGUI)
        gui.events = queue.Queue()
        gui._cancel_event = None
        return gui

    def test_gui_prefers_embedded_without_scanning_sidecar(self):
        gui = self.gui()
        with mock.patch(
            "hardsub_extractor.gui.acquire_local_embedded_text",
            return_value=result("local-embedded-text"),
        ), mock.patch("hardsub_extractor.gui.acquire_local_sidecar_text") as sidecar, mock.patch(
            "hardsub_extractor.gui.RapidOCRAdapter"
        ) as ocr:
            gui._worker("local", "video.mkv", None, Path("out"), False, ocr_language="en")
        sidecar.assert_not_called()
        ocr.assert_not_called()
        self.assertEqual([v for k, v in list(gui.events.queue) if k == "done"],
                         [("output.srt", "output.ass")])

    def test_gui_sidecar_prevents_ocr_when_no_matching_embedded_track(self):
        gui = self.gui()
        with mock.patch(
            "hardsub_extractor.gui.acquire_local_embedded_text", return_value=None,
        ), mock.patch(
            "hardsub_extractor.gui.acquire_local_sidecar_text",
            return_value=result("local-sidecar-text"),
        ) as sidecar, mock.patch("hardsub_extractor.gui.RapidOCRAdapter") as ocr:
            gui._worker("local", "video.mkv", None, Path("out"), False, ocr_language="ja")
        sidecar.assert_called_once_with("video.mkv", Path("out"), language="ja")
        ocr.assert_not_called()
        self.assertIn(("done", ("output.srt", "output.ass")), list(gui.events.queue))

    def test_cli_sidecar_before_ocr(self):
        out = io.StringIO()
        with mock.patch("sys.argv", ["hse", "video.mkv", "--out", "output", "--ocr-language", "en"]), mock.patch(
            "hardsub_extractor.cli.acquire_local_embedded_text", return_value=None
        ), mock.patch(
            "hardsub_extractor.cli.acquire_local_sidecar_text",
            return_value=result("local-sidecar-text"),
        ) as sidecar, mock.patch("hardsub_extractor.cli._ocr") as ocr, redirect_stdout(out):
            cli.main()
        sidecar.assert_called_once_with("video.mkv", Path("output"), language="en")
        ocr.assert_not_called()
        self.assertIn("mode=local-sidecar-text cues=2", out.getvalue())

    def test_gui_explicit_path_wins_over_embedded_track_and_does_not_ocr(self):
        gui = self.gui()
        with mock.patch(
            "hardsub_extractor.gui.acquire_local_embedded_text"
        ) as embedded, mock.patch(
            "hardsub_extractor.gui.acquire_local_sidecar_text",
            return_value=result("local-selected-text"),
        ) as sidecar, mock.patch("hardsub_extractor.gui.RapidOCRAdapter") as ocr:
            gui._worker("local", "video.mkv", "downloaded.srt",
                        Path("out"), False, ocr_language="ko")
        embedded.assert_not_called()
        sidecar.assert_called_once_with(
            "video.mkv", Path("out"), language="ko",
            selected_path="downloaded.srt",
        )
        ocr.assert_not_called()
        self.assertIn(("done", ("output.srt", "output.ass")), list(gui.events.queue))

    def test_gui_invalid_explicit_path_is_failure_without_any_ocr(self):
        gui = self.gui()
        with mock.patch(
            "hardsub_extractor.gui.acquire_local_embedded_text"
        ) as embedded, mock.patch(
            "hardsub_extractor.gui.acquire_local_sidecar_text",
            side_effect=ValueError("invalid manually chosen subtitle"),
        ), mock.patch("hardsub_extractor.gui.RapidOCRAdapter") as ocr:
            gui._worker("local", "video.mkv", "broken.srt",
                        Path("out"), False, ocr_language="en")
        embedded.assert_not_called()
        ocr.assert_not_called()
        self.assertTrue(any(key == "error" for key, _ in list(gui.events.queue)))
        self.assertFalse(any(key == "done" for key, _ in list(gui.events.queue)))

    def test_gui_explicit_picker_and_preflight(self):
        class Var:
            def __init__(self, value):
                self.value = value
            def get(self):
                return self.value
            def set(self, value):
                self.value = value
        from tempfile import TemporaryDirectory
        with TemporaryDirectory() as td:
            root = Path(td)
            video = root / "영상.mp4"
            subtitle = root / "받아둔 자막.vtt"
            video.write_bytes(b"temporary video")
            subtitle.write_bytes(b"WEBVTT\\n\\n00:00:01.000 --> 00:00:02.000\\nHello\\n")
            gui = self.gui()
            gui.source_mode_var = Var("local")
            gui.url_var = Var(str(video))
            gui.out_var = Var(str(root / "export" / "out"))
            gui.target_var = Var("old-target-do-not-use")
            gui.sidecar_var = Var("")
            gui.ocr_language_var = Var("en")
            with mock.patch(
                "hardsub_extractor.gui.filedialog.askopenfilename",
                return_value=str(subtitle),
            ) as picker:
                gui._pick_sidecar()
            self.assertIn("*.vtt", picker.call_args.kwargs["filetypes"][0][1])
            mode, source, target, output = gui._validate()
            self.assertEqual((mode, source, target), ("local", str(video), str(subtitle)))
            self.assertEqual(output, root / "export" / "out")
            gui.sidecar_var.set("missing.srt")
            with self.assertRaises(ValueError):
                gui._validate()

    def test_cli_explicit_file_never_scans_embedded_or_initializes_ocr(self):
        out = io.StringIO()
        with mock.patch("sys.argv", [
            "hse", "video.mkv", "--out", "output", "--ocr-language", "ja",
            "--subtitle-file", "picked.vtt",
        ]), mock.patch(
            "hardsub_extractor.cli.acquire_local_embedded_text"
        ) as embedded, mock.patch(
            "hardsub_extractor.cli.acquire_local_sidecar_text",
            return_value=result("local-selected-text"),
        ) as sidecar, mock.patch(
            "hardsub_extractor.cli._ocr"
        ) as ocr, redirect_stdout(out):
            cli.main()
        embedded.assert_not_called()
        sidecar.assert_called_once_with(
            "video.mkv", Path("output"), language="ja",
            selected_path="picked.vtt",
        )
        ocr.assert_not_called()
        self.assertIn("mode=local-selected-text", out.getvalue())

    def test_cli_rejects_conflicting_hard_only_and_explicit_file(self):
        with mock.patch("sys.argv", [
            "hse", "video.mkv", "--hard-only",
            "--subtitle-file", "picked.vtt",
        ]), mock.patch("hardsub_extractor.cli._ocr") as ocr, mock.patch(
            "hardsub_extractor.cli.acquire_local_sidecar_text"
        ) as sidecar:
            with self.assertRaises(SystemExit) as caught, mock.patch(
                "sys.stderr", io.StringIO(),
            ):
                cli.main()
        self.assertEqual(caught.exception.code, 2)
        sidecar.assert_not_called()
        ocr.assert_not_called()

    def test_cli_hard_only_skips_both_original_subtitle_sources(self):
        with mock.patch("sys.argv", ["hse", "video.mkv", "--hard-only"]), mock.patch(
            "hardsub_extractor.cli.acquire_local_embedded_text"
        ) as embedded, mock.patch(
            "hardsub_extractor.cli.acquire_local_sidecar_text"
        ) as sidecar, mock.patch(
            "hardsub_extractor.cli._ocr", return_value="model"
        ), mock.patch(
            "hardsub_extractor.cli.extract",
            return_value=SimpleNamespace(segment_count=1,cue_count=1,srt_path=Path("out.srt"),ass_path=Path("out.ass")),
        ) as extractor, redirect_stdout(io.StringIO()):
            cli.main()
        embedded.assert_not_called()
        sidecar.assert_not_called()
        # Keep the local CLI's full original frame sampling and OCR, but
        # never hold an entire episode of decoded video in memory.
        self.assertTrue(extractor.call_args.kwargs["streaming"])
        self.assertNotIn("sample_fps", extractor.call_args.kwargs)
        self.assertTrue(extractor.call_args.kwargs["monitor"].exclusive_outputs)


if __name__ == "__main__":
    unittest.main()
