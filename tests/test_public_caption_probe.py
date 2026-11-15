"""Offline contracts for temporary public-site E2E probe. No network."""
from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from hardsub_extractor.models import Cue
from hardsub_extractor.subtitle_acquisition import CaptionAccessError, CaptionResult
from hardsub_extractor.writers import write_subtitle_pair_exclusive
from unittest import mock
from hardsub_extractor.public_caption_diagnostics import main as probe_main, probe_one, ObservedCollector
from hardsub_extractor.subtitle_proxy import CaptionProxyRoute

YOUTUBE = "https://www.youtube.com/watch?v=unit_test"


class FakeCollector:
    def __init__(self, outcome="downloaded"):
        self.outcome = outcome
        self.requested_language = ""
        self.candidate_count = {"manual": 1, "automatic": 0}
        self.temporary_path = None

    def collect(self, url, out_dir, *, languages, check_cancelled):
        check_cancelled()
        root = Path(out_dir)
        self.temporary_path = root
        if self.outcome == "unavailable":
            raise CaptionAccessError("signed URL must never appear in reports")
        if self.outcome == "needs_ocr":
            return [CaptionResult("youtube", languages[0], "needs_ocr")]
        raw = b"WEBVTT\n\n00:00:01.000 --> 00:00:02.000\nhello\n"
        original = root / "unit.original.vtt"
        original.write_bytes(raw)
        srt, ass = write_subtitle_pair_exclusive([Cue(1.0, 2.0, "hello")], root / "unit")
        manifest = root / "unit.source.json"
        manifest.write_text(json.dumps({
            "site": "youtube", "cue_count": 1,
            "original_format": "vtt", "track_origin": "manual",
            "language_provided": "en",
            "original_timeline": "source_video",
            "output_timeline": "source_video", "ocr_applied": False,
            "machine_translation_applied": False,
        }), encoding="utf-8")
        return [CaptionResult(
            "youtube", languages[0], "downloaded",
            track_language="en", track_origin="manual", cue_count=1,
            original=original, srt=srt, ass=ass, manifest=manifest,
        )]


class PublicCaptionProbeTests(unittest.TestCase):
    def test_observed_collector_forwards_proxy_to_both_metadata_and_download(self):
        """Regression: diagnostic override must accept the new route keyword."""
        proxy = "socks5h://127.0.0.1:17920"
        commands = []

        def fake_yt_dlp(command, *, timeout_seconds, max_stdout_bytes):
            commands.append(list(command))
            if "--dump-single-json" in command:
                return json.dumps({
                    "id": "probe_proxy",
                    "subtitles": {"en": [{"ext": "vtt"}]},
                }).encode()
            dest = command[command.index("--output") + 1]
            Path(dest.replace("%(ext)s", "en.vtt")).write_bytes(
                b"WEBVTT\n\n00:00:01.000 --> 00:00:02.000\nhello\n"
            )
            return b""

        collector = ObservedCollector("synthetic-yt-dlp")
        collector.command_runner = fake_yt_dlp
        collector.proxy_route = CaptionProxyRoute("required", proxy)
        result = probe_one(YOUTUBE, "en", collector, max_seconds=10)
        self.assertEqual(result["status"], "downloaded")
        self.assertTrue(result["files_verified"])
        self.assertEqual(result["manual_candidates"], 1)
        self.assertEqual(len(commands), 2)
        self.assertEqual(
            [cmd[cmd.index("--proxy") + 1] for cmd in commands],
            [proxy, proxy],
        )

    def test_provenance_and_format_verified_and_original_discarded(self):
        collector = FakeCollector()
        result = probe_one(YOUTUBE, "en", collector, max_seconds=10)
        self.assertEqual(result["status"], "downloaded")
        self.assertTrue(result["files_verified"])
        self.assertEqual(result["cue_count"], 1)
        self.assertFalse(collector.temporary_path.exists())
        self.assertNotIn("hello", repr(result))

    def test_real_track_origin_is_reported_only_after_verification(self):
        collector = FakeCollector()
        result = probe_one(YOUTUBE, "en", collector, max_seconds=10)
        self.assertEqual(result["track_origin"], "manual")
        self.assertEqual(result["language_provided"], "en")

    def test_broken_download_is_an_integrity_failure(self):
        class BrokenResult(FakeCollector):
            def collect(self, *args, **kwargs):
                records = super().collect(*args, **kwargs)
                records[0].original.write_bytes(b"<html>not subtitles</html>")
                return records

        collector = BrokenResult()
        result = probe_one(YOUTUBE, "en", collector, max_seconds=10)
        self.assertEqual(result["status"], "integrity_failed")
        self.assertFalse(result["files_verified"])
        self.assertEqual(result["track_origin"], "")
        self.assertFalse(collector.temporary_path.exists())

    def test_same_cue_count_with_shifted_srt_timing_fails_integrity(self):
        class ShiftedSrt(FakeCollector):
            def collect(self, *args, **kwargs):
                records = super().collect(*args, **kwargs)
                path = records[0].srt
                content = path.read_text(encoding="utf-8")
                self.assert_marker = "00:00:01,000 --> 00:00:02,000" in content
                path.write_text(
                    content.replace(
                        "00:00:01,000 --> 00:00:02,000",
                        "00:01:01,000 --> 00:01:02,000",
                    ), encoding="utf-8",
                )
                return records

        collector = ShiftedSrt()
        result = probe_one(YOUTUBE, "en", collector, max_seconds=10)
        self.assertTrue(collector.assert_marker)
        self.assertEqual(result["status"], "integrity_failed")
        self.assertFalse(result["files_verified"])
        self.assertFalse(collector.temporary_path.exists())

    def test_metadata_counters_do_not_leak_across_site_failures(self):
        collector = FakeCollector("unavailable")
        collector.candidate_count = {"manual": 3, "automatic": 2}
        result = probe_one(YOUTUBE, "en", collector, max_seconds=10)
        self.assertEqual(result["manual_candidates"], 0)
        self.assertEqual(result["automatic_candidates"], 0)

    def test_access_failure_is_not_falsely_reported_as_downloaded(self):
        collector = FakeCollector("unavailable")
        result = probe_one(YOUTUBE, "en", collector, max_seconds=10)
        self.assertEqual(result["status"], "access_unavailable")
        self.assertFalse(result["files_verified"])
        self.assertFalse(collector.temporary_path.exists())
        self.assertNotIn("signed URL", repr(result))

    def test_missing_caption_remains_pending_ocr(self):
        collector = FakeCollector("needs_ocr")
        result = probe_one(YOUTUBE, "en", collector, max_seconds=10)
        self.assertEqual(result["status"], "needs_ocr")
        self.assertEqual(result["cue_count"], 0)
        self.assertFalse(collector.temporary_path.exists())

    def test_required_download_fails_if_only_ocr_is_possible(self):
        import contextlib
        import io
        collector = FakeCollector("needs_ocr")
        with tempfile.TemporaryDirectory() as td:
            report = Path(td) / "summary.json"
            with mock.patch(
                "hardsub_extractor.public_caption_diagnostics.ObservedCollector",
                return_value=collector,
            ), contextlib.redirect_stdout(io.StringIO()):
                code = probe_main([
                    "--url", YOUTUBE, "--languages", "en",
                    "--yt-dlp", "mock-binary", "--report", str(report),
                    "--require-download",
                ])
            self.assertEqual(code, 4)
            self.assertTrue(report.exists())
            self.assertEqual(
                json.loads(report.read_text(encoding="utf-8"))["results"][0]["status"],
                "needs_ocr",
            )
        self.assertFalse(collector.temporary_path.exists())

    def test_required_download_passes_for_verified_caption(self):
        import contextlib
        import io
        collector = FakeCollector()
        with tempfile.TemporaryDirectory() as td:
            with mock.patch(
                "hardsub_extractor.public_caption_diagnostics.ObservedCollector",
                return_value=collector,
            ), contextlib.redirect_stdout(io.StringIO()):
                code = probe_main([
                    "--url", YOUTUBE, "--languages", "en",
                    "--yt-dlp", "mock-binary",
                    "--report", str(Path(td) / "report.json"),
                    "--require-download",
                ])
            self.assertEqual(code, 0)

    def test_unsupported_site_is_rejected_before_collector_call(self):
        collector = FakeCollector()
        with self.assertRaises(ValueError):
            probe_one("https://example.org/episode", "en", collector)
        self.assertIsNone(collector.temporary_path)


if __name__ == "__main__":
    unittest.main()
