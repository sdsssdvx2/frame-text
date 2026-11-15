"""CLI batch routing is offline; no video/requests or yt-dlp subprocess runs."""
from __future__ import annotations

import contextlib
import io
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from hardsub_extractor.subtitle_acquisition import CaptionAccessError, CaptionResult
from hardsub_extractor.subtitle_batch_cli import _load_urls, main


class SubtitleBatchCliTests(unittest.TestCase):
    def test_input_file_loads_multiple_episodes_without_plaintext_logs(self):
        with tempfile.TemporaryDirectory() as td:
            src = Path(td) / "episodes.txt"
            src.write_text(
                "# authorized public episodes\n"
                "https://www.youtube.com/watch?v=one\n"
                "\nhttps://www.youtube.com/watch?v=two\n",
                encoding="utf-8",
            )
            self.assertEqual(len(_load_urls([], str(src))), 2)

    def test_duplicate_episode_urls_are_deduplicated_without_reordering(self):
        first = "https://www.youtube.com/watch?v=first"
        second = "https://wetv.vip/en/play/second"
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "episodes.txt"
            path.write_text(first + "\n" + second + "\n" + first + "\n", encoding="utf-8")
            self.assertEqual(_load_urls([second, first], str(path)), [second, first])

    def test_empty_url_values_are_discarded_before_network(self):
        self.assertEqual(
            _load_urls(["", "  ", "https://www.youtube.com/watch?v=one"], None),
            ["https://www.youtube.com/watch?v=one"],
        )

    def test_reject_more_than_fifty_episodes_before_any_network(self):
        with self.assertRaisesRegex(ValueError, "1 to 50"):
            _load_urls([f"https://www.youtube.com/watch?v=unique{i}" for i in range(51)], None)

    def test_success_does_not_log_url_or_video_id(self):
        out = io.StringIO()
        result = CaptionResult(
            source="youtube", requested_language="ja",
            status="downloaded", video_id="SECRET_VIDEO_ID",
            cue_count=12,
        )
        with mock.patch(
            "hardsub_extractor.subtitle_batch_cli.PublicCaptionCollector"
        ) as adapter, contextlib.redirect_stdout(out):
            adapter.return_value.collect.return_value = [result]
            code = main([
                "--url", "https://www.youtube.com/watch?v=PRIVATE_TOKEN",
                "--languages", "ja",
                "--out", "unused",
            ])
        self.assertEqual(code, 0)
        self.assertIn("STATUS=downloaded", out.getvalue())
        self.assertNotIn("PRIVATE_TOKEN", out.getvalue())
        self.assertNotIn("SECRET_VIDEO_ID", out.getvalue())

    def test_missing_caption_has_distinct_needs_ocr_status(self):
        out = io.StringIO()
        result = CaptionResult(
            source="youtube", requested_language="ko",
            status="needs_ocr",
        )
        with mock.patch(
            "hardsub_extractor.subtitle_batch_cli.PublicCaptionCollector"
        ) as adapter, contextlib.redirect_stdout(out):
            adapter.return_value.collect.return_value = [result]
            code = main(["--url", "https://www.youtube.com/watch?v=x"])
        self.assertEqual(code, 1)
        self.assertIn("STATUS=needs_ocr", out.getvalue())
        self.assertIn("downloaded=0 pending=1", out.getvalue())

    def test_video_ocr_explicit_flag_upgrades_missing_caption(self):
        out = io.StringIO()
        missing = CaptionResult("youtube", "ja", "needs_ocr", video_id="internal-secret")
        converted = CaptionResult("youtube", "ja", "hard_ocr", cue_count=7)
        with mock.patch(
            "hardsub_extractor.subtitle_batch_cli.PublicCaptionCollector"
        ) as soft, mock.patch(
            "hardsub_extractor.subtitle_batch_cli.resolve_missing_public_ocr",
            return_value=[converted],
        ) as video, contextlib.redirect_stdout(out):
            soft.return_value.collect.return_value = [missing]
            code = main([
                "--url", "https://www.youtube.com/watch?v=SECRET",
                "--languages", "ja", "--video-ocr",
            ])
        self.assertEqual(code, 0)
        self.assertEqual(video.call_count, 1)
        self.assertTrue(video.call_args.kwargs["skip_existing"])
        self.assertEqual(video.call_args.args[2], [missing])
        self.assertIn("STATUS=hard_ocr", out.getvalue())
        self.assertIn("downloaded=1 pending=0", out.getvalue())
        self.assertNotIn("SECRET", out.getvalue())
        self.assertNotIn("internal-secret", out.getvalue())

    def test_caption_success_never_downloads_video_with_ocr_flag(self):
        out = io.StringIO()
        with mock.patch(
            "hardsub_extractor.subtitle_batch_cli.PublicCaptionCollector"
        ) as soft, mock.patch(
            "hardsub_extractor.subtitle_batch_cli.resolve_missing_public_ocr"
        ) as video, contextlib.redirect_stdout(out):
            soft.return_value.collect.return_value = [
                CaptionResult("youtube", "ko", "downloaded", cue_count=4)
            ]
            code = main([
                "--url", "https://www.youtube.com/watch?v=x",
                "--video-ocr",
            ])
        self.assertEqual(code, 0)
        video.assert_not_called()

    def test_ocr_video_access_failure_is_reported_not_downloaded(self):
        out = io.StringIO()
        with mock.patch(
            "hardsub_extractor.subtitle_batch_cli.PublicCaptionCollector"
        ) as soft, mock.patch(
            "hardsub_extractor.subtitle_batch_cli.resolve_missing_public_ocr",
            return_value=[CaptionResult("youtube", "ko", "ocr_unavailable")],
        ), contextlib.redirect_stdout(out):
            soft.return_value.collect.return_value = [
                CaptionResult("youtube", "ko", "needs_ocr", video_id="x")
            ]
            code = main([
                "--url", "https://www.youtube.com/watch?v=x",
                "--video-ocr",
            ])
        self.assertEqual(code, 1)
        self.assertIn("STATUS=ocr_unavailable", out.getvalue())
        self.assertIn("downloaded=0 pending=1", out.getvalue())

    def test_permission_error_fails_closed_without_private_exception_text(self):
        out = io.StringIO()
        with mock.patch(
            "hardsub_extractor.subtitle_batch_cli.PublicCaptionCollector"
        ) as adapter, contextlib.redirect_stdout(out):
            adapter.return_value.collect.side_effect = CaptionAccessError(
                "private-token-NEVER-PRINT"
            )
            code = main(["--url", "https://wetv.vip/en/play/private"])
        self.assertEqual(code, 1)
        self.assertIn("STATUS=access_unavailable", out.getvalue())
        self.assertNotIn("private-token-NEVER-PRINT", out.getvalue())

    def test_old_files_collision_never_silently_overwrites(self):
        out = io.StringIO()
        with mock.patch(
            "hardsub_extractor.subtitle_batch_cli.PublicCaptionCollector"
        ) as adapter, contextlib.redirect_stdout(out):
            adapter.return_value.collect.side_effect = FileExistsError(
                "do not reveal private paths"
            )
            code = main(["--url", "https://www.youtube.com/watch?v=x"])
        self.assertEqual(code, 1)
        self.assertIn("STATUS=existing_files_preserved", out.getvalue())
        self.assertNotIn("private paths", out.getvalue())


if __name__ == "__main__":
    unittest.main()
