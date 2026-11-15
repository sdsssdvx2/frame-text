"""GUI, CLI and candidate-order integration for read-only resume audits."""
from __future__ import annotations

import contextlib
import io
import json
from pathlib import Path
import queue
import tempfile
import threading
import unittest
from unittest import mock

from hardsub_extractor.gui import ExtractorGUI
from hardsub_extractor.subtitle_acquisition import CaptionResult, PublicCaptionCollector, _commit_caption
from hardsub_extractor.subtitle_batch_cli import main as cli_main
from hardsub_extractor.subtitle_language import SubtitleTrack

VIDEO = "https://www.youtube.com/watch?v=audit"
RAW = b"WEBVTT\n\n00:00:01.000 --> 00:00:02.500\nHello world\n"


def gui():
    window = ExtractorGUI.__new__(ExtractorGUI)
    window.events = queue.Queue()
    window._cancel_event = threading.Event()
    window._requested_video_ocr_enabled = True
    window._progress_stage = None
    window._pending_close = False
    window._verified_existing_count = 0
    window.progress_bar = mock.Mock()
    window.start_button = mock.Mock()
    window.cancel_button = mock.Mock()
    window.status_var = mock.Mock()
    # A real initialized GUI always has source_mode_var; this headless
    # fixture must represent an active captions-mode window as well.
    window.source_mode_var = mock.Mock()
    window.source_mode_var.get.return_value = "captions"
    window.root = mock.Mock()
    window._log = mock.Mock()
    return window


class PublicResumeIntegrationTests(unittest.TestCase):
    def test_gui_explicitly_distinguishes_new_verified_and_unverified(self):
        with tempfile.TemporaryDirectory() as td:
            window = gui()
            outputs = [
                CaptionResult("youtube", "en", "existing_verified", cue_count=2),
                CaptionResult("youtube", "ja", "existing_preserved"),
                CaptionResult("youtube", "ko", "downloaded", cue_count=3),
            ]
            with mock.patch(
                "hardsub_extractor.gui.PublicCaptionCollector",
            ) as collector, mock.patch(
                "hardsub_extractor.gui.messagebox.showinfo",
            ) as notice:
                collector.return_value.collect.return_value = outputs
                window._worker(
                    "captions", VIDEO, None, Path(td), False, ("en", "ja", "ko"),
                )
                self.assertIn(
                    ("caption_verified", 1), list(window.events.queue),
                )
                self.assertIn(
                    ("captions_done", (1, 0, 0, 1, td)), list(window.events.queue),
                )
                window._poll_events()
            self.assertEqual(notice.call_count, 1)
            summary = notice.call_args.args[1]
            self.assertIn("신규 확보: 1개", summary)
            self.assertIn("기존 결과 구조 검증: 1개", summary)
            self.assertIn("검증 불가 파일 보존: 1개", summary)
            self.assertNotIn("https://", summary)

    def test_gui_only_complete_existing_results_are_marked_finished(self):
        with tempfile.TemporaryDirectory() as td:
            window = gui()
            with mock.patch("hardsub_extractor.gui.PublicCaptionCollector") as collector:
                collector.return_value.collect.return_value = [
                    CaptionResult("youtube", "en", "existing_verified", cue_count=2),
                ]
                window._worker("captions", VIDEO, None, Path(td), False, ("en",))
            self.assertIn(("captions_done", (0, 0, 0, 0, td)), list(window.events.queue))
            self.assertIn(("caption_verified", 1), list(window.events.queue))

    def test_cli_exits_zero_for_complete_verified_resume(self):
        output = io.StringIO()
        with mock.patch(
            "hardsub_extractor.subtitle_batch_cli.PublicCaptionCollector",
        ) as collector, contextlib.redirect_stdout(output):
            collector.return_value.collect.return_value = [
                CaptionResult("youtube", "en", "existing_verified", cue_count=2),
                CaptionResult("youtube", "ko", "existing_verified", cue_count=1),
            ]
            code = cli_main([
                "--url", VIDEO, "--languages", "en,ko",
            ])
        self.assertEqual(code, 0)
        self.assertIn("STATUS=existing_verified", output.getvalue())
        self.assertIn("SUMMARY downloaded=0 pending=0 verified=2", output.getvalue())
        self.assertNotIn(VIDEO, output.getvalue())

    def test_cli_preserved_partial_prevents_all_success_claim(self):
        output = io.StringIO()
        with mock.patch(
            "hardsub_extractor.subtitle_batch_cli.PublicCaptionCollector",
        ) as collector, contextlib.redirect_stdout(output):
            collector.return_value.collect.return_value = [
                CaptionResult("youtube", "en", "existing_verified", cue_count=2),
                CaptionResult("youtube", "ko", "existing_preserved"),
            ]
            code = cli_main(["--url", VIDEO, "--languages", "en,ko"])
        self.assertEqual(code, 1)
        self.assertIn("SUMMARY downloaded=0 pending=1 verified=1", output.getvalue())

    def test_cli_real_production_collector_audits_saved_files_on_resume(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            saved = _commit_caption(
                RAW, SubtitleTrack("en", "manual", "vtt", "en"),
                provider="youtube", video_id="episode_id",
                target_directory=root, original_extension=".vtt",
            )
            before = {
                p.name: p.read_bytes() for p in root.iterdir()
            }
            commands = []
            def fake_metadata(args, **kwargs):
                commands.append(args)
                if "--dump-single-json" not in args:
                    raise AssertionError("complete resume must never download any caption")
                return json.dumps({
                    "id": "episode_id",
                    "subtitles": {"en": [{"ext": "vtt"}]},
                }).encode()
            collector = PublicCaptionCollector("test-pinned", command_runner=fake_metadata)
            output = io.StringIO()
            with mock.patch(
                "hardsub_extractor.subtitle_batch_cli.PublicCaptionCollector",
                return_value=collector,
            ), contextlib.redirect_stdout(output):
                rc = cli_main([
                    "--url", VIDEO, "--languages", "en", "--out", str(root),
                ])
            self.assertEqual(rc, 0)
            self.assertIn("STATUS=existing_verified CUES=1", output.getvalue())
            self.assertIn("SUMMARY downloaded=0 pending=0 verified=1", output.getvalue())
            self.assertEqual(len(commands), 1)
            self.assertEqual(
                {p.name: p.read_bytes() for p in root.iterdir()}, before,
            )
            self.assertTrue(saved.original.is_file())

    def test_cli_real_production_collector_preserves_partial_on_resume(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            leftover = root / "youtube-episode_id-en-manual.srt"
            leftover.write_bytes(b"USER_UNFINISHED_FILE")
            commands = []
            def fake_metadata(args, **kwargs):
                commands.append(args)
                if "--dump-single-json" not in args:
                    raise AssertionError("partial resume must not download/overwrite")
                return json.dumps({
                    "id": "episode_id",
                    "subtitles": {"en": [{"ext": "vtt"}]},
                }).encode()
            collector = PublicCaptionCollector("test-pinned", command_runner=fake_metadata)
            output = io.StringIO()
            with mock.patch(
                "hardsub_extractor.subtitle_batch_cli.PublicCaptionCollector",
                return_value=collector,
            ), contextlib.redirect_stdout(output):
                rc = cli_main([
                    "--url", VIDEO, "--languages", "en", "--out", str(root),
                ])
            self.assertEqual(rc, 1)
            self.assertIn("STATUS=existing_preserved CUES=0", output.getvalue())
            self.assertIn("SUMMARY downloaded=0 pending=1", output.getvalue())
            self.assertEqual(len(commands), 1)
            self.assertEqual(leftover.read_bytes(), b"USER_UNFINISHED_FILE")

    def test_cli_fifty_actual_saved_exports_are_audited_without_new_writes(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            export = root / "export"
            urls = []
            for index in range(1, 51):
                video_id = f"resume{index:03d}"
                urls.append("https://www.youtube.com/watch?v=" + video_id)
                _commit_caption(
                    RAW, SubtitleTrack("en", "manual", "vtt", "en"),
                    provider="youtube", video_id=video_id,
                    target_directory=export, original_extension=".vtt",
                )
            url_file = root / "episodes.txt"
            url_file.write_text("\n".join(urls) + "\n", encoding="utf-8")
            locked = {
                p.name: p.read_bytes() for p in export.iterdir()
                if p.is_file()
            }
            self.assertEqual(len(locked), 200)
            commands = []
            def metadata_only(args, **kwargs):
                commands.append(args)
                if "--dump-single-json" not in args:
                    raise AssertionError("restart must not redownload any caption")
                episode = args[-1].split("v=", 1)[-1]
                return json.dumps({
                    "id": episode,
                    "subtitles": {"en": [{"ext": "vtt"}]},
                }).encode()
            collector = PublicCaptionCollector(
                "fake-pinned", command_runner=metadata_only,
            )
            stdout = io.StringIO()
            with mock.patch(
                "hardsub_extractor.subtitle_batch_cli.PublicCaptionCollector",
                return_value=collector,
            ), contextlib.redirect_stdout(stdout):
                rc = cli_main([
                    "--urls-file", str(url_file), "--languages", "en",
                    "--out", str(export),
                ])
            self.assertEqual(rc, 0)
            self.assertEqual(len(commands), 50)
            self.assertEqual(stdout.getvalue().count("STATUS=existing_verified"), 50)
            self.assertIn("SUMMARY downloaded=0 pending=0 verified=50", stdout.getvalue())
            self.assertNotIn(urls[0], stdout.getvalue())
            self.assertEqual(
                {p.name: p.read_bytes() for p in export.iterdir() if p.is_file()},
                locked,
            )

    def test_valid_auto_track_wins_over_earlier_partial_manual(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            manual = root / "youtube-episode_id-en-manual.srt"
            manual.write_bytes(b"PARTIAL_MANUAL_DO_NOT_MODIFY")
            automatic = _commit_caption(
                RAW,
                SubtitleTrack("en", "automatic", "vtt", "en"),
                provider="youtube", video_id="episode_id",
                target_directory=root, original_extension=".vtt",
            )
            existing = {
                p.name: p.read_bytes()
                for p in root.iterdir()
            }
            commands = []
            def runner(args, **kwargs):
                commands.append(args)
                if "--dump-single-json" not in args:
                    raise AssertionError("no caption track should be redownloaded")
                return json.dumps({
                    "id": "episode_id",
                    "subtitles": {"en": [{"ext": "vtt"}]},
                    "automatic_captions": {"en": [{"ext": "vtt"}]},
                }).encode()
            collector = PublicCaptionCollector("test-pinned", command_runner=runner)
            result = collector.collect(
                VIDEO, root, languages=("en",), skip_existing=True,
            )[0]
            self.assertEqual(result.status, "existing_verified")
            self.assertEqual(result.track_origin, "automatic")
            self.assertEqual(result.cue_count, 1)
            self.assertEqual(result.original, automatic.original)
            self.assertEqual(len(commands), 1)
            self.assertEqual(
                {p.name: p.read_bytes() for p in root.iterdir()}, existing,
            )
            self.assertEqual(manual.read_bytes(), b"PARTIAL_MANUAL_DO_NOT_MODIFY")


if __name__ == "__main__":
    unittest.main()
