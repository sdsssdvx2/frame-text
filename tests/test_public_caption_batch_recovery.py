"""50-episode public-caption / visual-OCR batch recovery without network.

A deterministic fake collector writes one existing soft track and the fake
visual stage conditionally publishes another language. Repeating the batch
must never replace a byte of a previously successful subtitle, and a failed
video probe on one episode must not erase a successful soft track or stop
later episodes. No user media, browser profile, login or live network used.
"""
from __future__ import annotations

import contextlib
import io
from pathlib import Path
import queue
import tempfile
import threading
import unittest
from unittest import mock

from hardsub_extractor.gui import ExtractorGUI
from hardsub_extractor.public_video_ocr import PublicVideoOCRError
from hardsub_extractor.subtitle_acquisition import CaptionResult
from hardsub_extractor.subtitle_batch_cli import main as cli_main

EPISODES = [
    f"https://www.youtube.com/watch?v=synthetic{index:03d}"
    for index in range(1, 51)
]


class BatchFixture:
    def __init__(self):
        self.collect_calls = []
        self.ocr_calls = []
        self.once_failed = {13, 37}
        self.source_bytes = b"LOCKED_SOFT_SUBTITLE_BYTES"
        self.hard_bytes = b"LOCKED_OCR_SUBTITLE_BYTES"

    @staticmethod
    def number(url):
        return int(url[-3:])

    def collect(self, url, out_dir, *, languages, **kwargs):
        index = self.number(url)
        self.collect_calls.append(index)
        if tuple(languages) != ("ko", "en"):
            raise AssertionError("batch language order was not preserved")
        if not kwargs.get("skip_existing", True):
            raise AssertionError("existing user subtitles must be preserved")
        root = Path(out_dir)
        root.mkdir(parents=True, exist_ok=True)
        locked = root / f"soft-{index:03d}.srt"
        if locked.is_file():
            if locked.read_bytes() != self.source_bytes:
                raise AssertionError("successful soft track changed")
            status = "existing_preserved"
        else:
            with locked.open("xb") as f:
                f.write(self.source_bytes)
            status = "downloaded"
        return [
            CaptionResult("youtube", "ko", status, video_id=f"episode{index:03d}", cue_count=1),
            CaptionResult("youtube", "en", "needs_ocr", video_id=f"episode{index:03d}"),
        ]

    def video(self, url, out_dir, results, **kwargs):
        index = self.number(url)
        self.ocr_calls.append(index)
        if [x.requested_language for x in results] != ["ko", "en"]:
            raise AssertionError("OCR received a mutated language list")
        root = Path(out_dir)
        file = root / f"hard-{index:03d}.srt"
        if file.exists():
            if file.read_bytes() != self.hard_bytes:
                raise AssertionError("published hard OCR changed")
            status = "existing_preserved"
        elif index in self.once_failed:
            self.once_failed.remove(index)
            raise PublicVideoOCRError("private-proxy-query-DO-NOT-LEAK")
        else:
            with file.open("xb") as f:
                f.write(self.hard_bytes)
            status = "hard_ocr"
        return [
            results[0],
            CaptionResult(
                "youtube", "en", status, video_id=f"episode{index:03d}",
                cue_count=1 if status == "hard_ocr" else 0,
            ),
        ]


class FiftyEpisodeRecoveryTests(unittest.TestCase):
    def _urls_file(self, root):
        path = root / "episodes.txt"
        path.write_text("\n".join(EPISODES) + "\n", encoding="utf-8")
        return path

    def _gui(self):
        g = ExtractorGUI.__new__(ExtractorGUI)
        g.events = queue.Queue()
        g._cancel_event = threading.Event()
        g._requested_video_ocr_enabled = True
        g._progress_stage = None
        return g

    def _gui_outcome(self, gui):
        return [payload for label, payload in gui.events.queue
                if label == "captions_done"]

    def test_gui_50_episodes_first_and_resume_preserve_every_soft_result(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            url_file = self._urls_file(root)
            out = root / "export"
            fixture = BatchFixture()
            gui = self._gui()
            with mock.patch(
                "hardsub_extractor.gui.PublicCaptionCollector",
            ) as collector, mock.patch(
                "hardsub_extractor.gui.resolve_missing_public_ocr",
                side_effect=fixture.video,
            ):
                collector.return_value.collect.side_effect = fixture.collect
                gui._worker("captions", "", str(url_file), out, False, ("ko", "en"))
                self.assertEqual(
                    self._gui_outcome(gui), [(98, 0, 2, 0, str(out))],
                )
                self.assertEqual(fixture.collect_calls, list(range(1, 51)))
                self.assertEqual(fixture.ocr_calls, list(range(1, 51)))
                self.assertFalse(any(label == "error" for label, _ in gui.events.queue))
                self.assertFalse(any(label == "cancelled" for label, _ in gui.events.queue))
                self.assertEqual(len(list(out.glob("soft-*.srt"))), 50)
                self.assertEqual(len(list(out.glob("hard-*.srt"))), 48)
                self.assertNotIn("private-proxy-query", repr(list(gui.events.queue)))
                self.assertNotIn("synthetic013", repr(list(gui.events.queue)))

                gui.events = queue.Queue()
                gui._worker("captions", "", str(url_file), out, False, ("ko", "en"))
            self.assertEqual(
                self._gui_outcome(gui), [(2, 0, 0, 98, str(out))],
            )
            self.assertEqual(len(fixture.collect_calls), 100)
            self.assertEqual(fixture.ocr_calls, list(range(1, 51)) * 2)
            self.assertEqual(len(list(out.glob("soft-*.srt"))), 50)
            self.assertEqual(len(list(out.glob("hard-*.srt"))), 50)
            self.assertTrue(all(
                item.read_bytes() == fixture.source_bytes
                for item in out.glob("soft-*.srt")
            ))
            self.assertTrue(all(
                item.read_bytes() == fixture.hard_bytes
                for item in out.glob("hard-*.srt")
            ))
            self.assertFalse(any(label == "error" for label, _ in gui.events.queue))

    def test_cli_50_episodes_first_and_resume_retains_partial_success(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            url_file = self._urls_file(root)
            out = root / "export"
            fixture = BatchFixture()
            output = io.StringIO()
            with mock.patch(
                "hardsub_extractor.subtitle_batch_cli.PublicCaptionCollector",
            ) as collector, mock.patch(
                "hardsub_extractor.subtitle_batch_cli.resolve_missing_public_ocr",
                side_effect=fixture.video,
            ), contextlib.redirect_stdout(output):
                collector.return_value.collect.side_effect = fixture.collect
                code1 = cli_main([
                    "--urls-file", str(url_file), "--languages", "ko,en",
                    "--out", str(out), "--video-ocr",
                ])
                code2 = cli_main([
                    "--urls-file", str(url_file), "--languages", "ko,en",
                    "--out", str(out), "--video-ocr",
                ])
            # Existing output is never claimed as a verified new success:
            # the CLI deliberately reports preserved outputs as pending.
            self.assertEqual((code1, code2), (1, 1))
            lines = output.getvalue().splitlines()
            summaries = [line for line in lines if line.startswith("SUMMARY ")]
            self.assertEqual(
                summaries,
                ["SUMMARY downloaded=98 pending=2",
                 "SUMMARY downloaded=2 pending=98"],
            )
            self.assertNotIn("private-proxy-query", output.getvalue())
            self.assertNotIn("synthetic013", output.getvalue())
            self.assertEqual(len(list(out.glob("soft-*.srt"))), 50)
            self.assertEqual(len(list(out.glob("hard-*.srt"))), 50)
            self.assertEqual(len(fixture.collect_calls), 100)
            self.assertEqual(len(fixture.ocr_calls), 100)

    def test_bad_video_validation_does_not_hide_successful_soft_caption(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            fixture = BatchFixture()
            with mock.patch(
                "hardsub_extractor.subtitle_batch_cli.PublicCaptionCollector",
            ) as collector, mock.patch(
                "hardsub_extractor.subtitle_batch_cli.resolve_missing_public_ocr",
                side_effect=PublicVideoOCRError("private-media-path"),
            ), contextlib.redirect_stdout(io.StringIO()) as output:
                collector.return_value.collect.side_effect = fixture.collect
                code = cli_main([
                    "--url", EPISODES[0], "--languages", "ko,en",
                    "--out", str(root), "--video-ocr",
                ])
            self.assertEqual(code, 1)
            self.assertIn("LANG=ko SITE=youtube STATUS=downloaded", output.getvalue())
            self.assertIn("LANG=en SITE=youtube STATUS=ocr_unavailable", output.getvalue())
            self.assertIn("SUMMARY downloaded=1 pending=1", output.getvalue())
            self.assertNotIn("private-media-path", output.getvalue())


if __name__ == "__main__":
    unittest.main()
