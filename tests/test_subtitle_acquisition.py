"""No-network contract tests for multilingual subtitle-first batch acquisition."""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from hardsub_extractor.subtitle_acquisition import (
    CaptionAccessError,
    PublicCaptionCollector,
    _commit_caption,
    _run_silent,
    supported_public_site,
)
from hardsub_extractor.subtitle_language import (
    SubtitleTrack,
    candidate_tracks,
    language_matches,
    normalize_language,
    select_track,
)

YOUTUBE = "https://www.youtube.com/watch?v=synthetic_test"
WETV = "https://wetv.vip/en/play/synthetic-episode"
VTT = b"WEBVTT\n\n00:00:01.000 --> 00:00:02.500\n" + "중국어 테스트 자막".encode()


class FakeCaptionRunner:
    def __init__(self, meta, *, missing=False):
        self.meta = meta
        self.missing = missing
        self.commands = []

    def __call__(self, command, *, timeout_seconds, max_stdout_bytes):
        self.commands.append(list(command))
        if "--dump-single-json" in command:
            return json.dumps(self.meta).encode()
        if self.missing:
            return b""
        target = command[command.index("--output") + 1]
        requested_language = command[command.index("--sub-langs") + 1]
        if not (requested_language.startswith("^") and requested_language.endswith("$")):
            raise AssertionError("subtitle language selector must be exact")
        requested_language = requested_language[1:-1].replace(r"\-", "-")
        ext = command[command.index("--sub-format") + 1]
        dest = Path(target.replace("%(ext)s", requested_language + "." + ext))
        dest.write_bytes(VTT)
        return b""


class LanguageTrackTests(unittest.TestCase):
    def test_language_options_are_explicit_and_not_machine_translated(self):
        self.assertEqual(normalize_language("ZH_hANS"), "zh-hans")
        for invalid in ("", "all", "ru", "en.*", "ko;rm", "../en", "--write-subs"):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                normalize_language(invalid)

    def test_region_specific_english_and_orig_tag_are_recognized(self):
        self.assertTrue(language_matches("en", "en-orig"))
        self.assertTrue(language_matches("en", "en-GB"))
        self.assertFalse(language_matches("en", "fr-en"))
        self.assertFalse(language_matches("ko", "ja"))

    def test_japanese_and_chinese_cannot_be_inferred_from_same_han_glyphs(self):
        self.assertFalse(language_matches("ja", "zh-Hans"))
        self.assertFalse(language_matches("zh", "ja"))
        self.assertTrue(language_matches("zh-Hans", "zh-CN"))
        self.assertTrue(language_matches("zh-Hant", "zh-TW"))
        self.assertFalse(language_matches("zh-Hant", "zh-Hans"))

    def test_manual_english_wins_over_automatic_english(self):
        selected = select_track(
            "en", {"en": [{"ext": "vtt"}]},
            {"en-orig": [{"ext": "vtt"}]},
        )
        self.assertEqual(selected.origin, "manual")
        self.assertEqual(selected.language, "en")

    def test_original_language_auto_track_preferred_over_auto_translation(self):
        track = select_track(
            "en", {},
            {"en": [{"ext": "vtt"}], "en-orig": [{"ext": "vtt"}]},
        )
        self.assertEqual(track.language, "en-orig")
        track_zh = select_track(
            "zh-Hans", {}, {"zh-Hans-orig": [{"ext": "vtt"}]},
        )
        self.assertEqual(track_zh.language, "zh-Hans-orig")

    def test_automatic_requested_language_only_when_manual_missing(self):
        selected = select_track(
            "ko", {"en": [{"ext": "vtt"}]},
            {"ko-orig": [{"ext": "vtt"}]},
        )
        self.assertEqual(selected.origin, "automatic")
        self.assertEqual(selected.language, "ko-orig")

    def test_vtt_is_selected_over_unconvertible_json3_track(self):
        selected = select_track(
            "ja", {"ja": [{"ext": "json3"}, {"ext": "vtt"}]},
            {},
        )
        self.assertEqual(selected.extension, "vtt")

    def test_only_wrong_language_returns_no_track_not_a_false_success(self):
        self.assertIsNone(select_track(
            "ko", {"en": [{"ext": "vtt"}]}, {},
        ))

    def test_malformed_provider_tracks_are_ignored(self):
        self.assertIsNone(select_track(
            "ko", {"ko": "not a valid track list"}, {},
        ))

    def test_multiple_matching_manual_formats_before_automatic(self):
        tracks = candidate_tracks(
            "en", {"en": [{"ext": "vtt"}, {"ext": "srt"}, {"ext": "ass"}]},
            {"en-orig": [{"ext": "vtt"}]},
        )
        self.assertEqual(
            [(track.origin, track.extension) for track in tracks],
            [("manual", "vtt"), ("manual", "srt"),
             ("manual", "ass"), ("automatic", "vtt")],
        )

    def test_duplicate_provider_offers_are_not_retried(self):
        tracks = candidate_tracks(
            "en", {"en": [{"ext": "vtt"}, {"ext": "vtt"}, {"ext": "srt"}]},
            {},
        )
        self.assertEqual([track.extension for track in tracks], ["vtt", "srt"])

    def test_candidate_retry_count_is_bounded(self):
        tracks = candidate_tracks(
            "zh", {"zh": [{"ext": "vtt"}, {"ext": "srt"}, {"ext": "ass"}, {"ext": "ssa"}],
                   "zh-CN": [{"ext": "vtt"}]},
            {"zh-orig": [{"ext": "vtt"}, {"ext": "srt"}, {"ext": "ass"}]},
        )
        self.assertLessEqual(len(tracks), 5)


class PublicCaptionAdapterTests(unittest.TestCase):
    def test_https_host_allowlist_prevents_local_and_spoof_urls(self):
        self.assertEqual(supported_public_site(YOUTUBE), "youtube")
        self.assertEqual(supported_public_site(WETV), "wetv")
        for url in (
            "http://www.youtube.com/watch?v=1",
            "https://www.youtube.com.evil.test/watch?v=1",
            "https://127.0.0.1/watch?v=1",
            "file:///etc/passwd",
            "https://user:pass@www.youtube.com/watch?v=1",
            "https://www.youtube.com:444/watch?v=1",
            "https://unlisted-video.example/show/a",
        ):
            with self.subTest(url=url):
                self.assertIsNone(supported_public_site(url))

    def test_bundled_downloader_is_preferred_over_existing_path(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            downloader = root / "bin" / "yt-dlp.exe"
            downloader.parent.mkdir()
            downloader.write_bytes(b"synthetic executable marker")
            with (
                mock.patch(
                    "hardsub_extractor.subtitle_acquisition.bundle_root",
                    return_value=root,
                ),
                mock.patch(
                    "hardsub_extractor.subtitle_acquisition.shutil.which",
                    return_value="untrusted-user-path-yt-dlp",
                ),
            ):
                from hardsub_extractor.subtitle_acquisition import PublicCaptionCollector
                self.assertEqual(
                    PublicCaptionCollector().executable, str(downloader)
                )

    def test_cancellation_prevents_caption_commit_even_after_download(self):
        runner = FakeCaptionRunner({
            "id": "cancel_test", "subtitles": {"en": [{"ext": "vtt"}]},
        })
        collector = PublicCaptionCollector("yt-dlp-test", command_runner=runner)
        callback_count = 0

        def cancel_after_download():
            nonlocal callback_count
            callback_count += 1
            if callback_count >= 5:
                raise KeyboardInterrupt("user requested cancel")

        with tempfile.TemporaryDirectory() as td:
            with self.assertRaises(KeyboardInterrupt):
                collector.collect(
                    YOUTUBE, td, languages=("en",),
                    check_cancelled=cancel_after_download,
                )
            self.assertEqual(list(Path(td).iterdir()), [])
        self.assertEqual(len(runner.commands), 2)

    def test_unsupported_site_is_reported_without_any_external_call(self):
        runner = FakeCaptionRunner({})
        collector = PublicCaptionCollector("yt-dlp-test", command_runner=runner)
        result = collector.collect(
            "https://unlisted-video.example/movie", "never-created", languages=("en",),
        )
        self.assertEqual(result[0].status, "unsupported_site")
        self.assertEqual(runner.commands, [])

    def test_no_available_track_marks_need_for_ocr_without_downloading(self):
        runner = FakeCaptionRunner({
            "id": "test_id", "subtitles": {"en": [{"ext": "vtt"}]},
            "automatic_captions": {},
        })
        collector = PublicCaptionCollector("yt-dlp-test", command_runner=runner)
        with tempfile.TemporaryDirectory() as td:
            result = collector.collect(YOUTUBE, td, languages=("ko",))
            self.assertFalse(list(Path(td).iterdir()))
        self.assertEqual(result[0].status, "needs_ocr")
        self.assertEqual(len(runner.commands), 1)

    def test_yt_dlp_never_downloads_video_or_imports_browser_credentials(self):
        runner = FakeCaptionRunner({
            "id": "safe_test", "subtitles": {"ko": [{"ext": "vtt"}]},
        })
        collector = PublicCaptionCollector("yt-dlp-test", command_runner=runner)
        with tempfile.TemporaryDirectory() as td:
            result = collector.collect(YOUTUBE, td, languages=("ko",))
            self.assertEqual(result[0].status, "downloaded")
        self.assertEqual(len(runner.commands), 2)
        for command in runner.commands:
            self.assertIn("--ignore-config", command)
            self.assertIn("--no-plugin-dirs", command)
            self.assertIn("--skip-download", command)
            self.assertIn("--no-playlist", command)
            self.assertNotIn("--cookies-from-browser", command)
            self.assertNotIn("--username", command)
            self.assertNotIn("--password", command)
            self.assertNotIn("--allow-unplayable-formats", command)

    def test_original_vtt_preserved_alongside_korean_srt_ass(self):
        runner = FakeCaptionRunner({
            "id": "test_001", "subtitles": {"ko": [{"ext": "vtt"}]},
        })
        collector = PublicCaptionCollector("yt-dlp-test", command_runner=runner)
        with tempfile.TemporaryDirectory() as td:
            result = collector.collect(YOUTUBE, td, languages=("ko",))[0]
            self.assertEqual(result.status, "downloaded")
            self.assertEqual(result.cue_count, 1)
            self.assertEqual(result.original.read_bytes(), VTT)
            self.assertIn("중국어 테스트 자막", result.srt.read_text(encoding="utf-8"))
            self.assertTrue(result.ass.is_file())
            info = json.loads(result.manifest.read_text(encoding="utf-8"))
            self.assertEqual(info["original_timeline"], "source_video")
            self.assertFalse(info["ocr_applied"])
            self.assertFalse(info["machine_translation_applied"])
            self.assertEqual(info["provider_translation_status"], "not_applicable")
            self.assertEqual(info["language_provided"], "ko")
            self.assertNotIn("https://", result.manifest.read_text(encoding="utf-8"))

    def test_new_public_caption_never_overwrites_old_one(self):
        runner = FakeCaptionRunner({
            "id": "test_001", "subtitles": {"en": [{"ext": "vtt"}]},
        })
        collector = PublicCaptionCollector("yt-dlp-test", command_runner=runner)
        with tempfile.TemporaryDirectory() as td:
            first = collector.collect(YOUTUBE, td, languages=("en",))[0]
            with self.assertRaises(FileExistsError):
                collector.collect(YOUTUBE, td, languages=("en",))
            self.assertEqual(first.original.read_bytes(), VTT)
            self.assertTrue(first.srt.is_file())
            self.assertTrue(first.manifest.is_file())

    def test_opt_in_resume_preserves_existing_english_and_continues_korean(self):
        runner = FakeCaptionRunner({
            "id": "resume_vid",
            "subtitles": {"en": [{"ext": "vtt"}], "ko": [{"ext": "vtt"}]},
        })
        collector = PublicCaptionCollector("yt-dlp-test", command_runner=runner)
        with tempfile.TemporaryDirectory() as td:
            prior = collector.collect(YOUTUBE, td, languages=("en",))[0]
            preserved = prior.original.read_bytes()
            count = len(runner.commands)
            outcome = collector.collect(
                YOUTUBE, td, languages=("en", "ko"), skip_existing=True,
            )
            self.assertEqual([r.status for r in outcome], [
                "existing_verified", "downloaded",
            ])
            self.assertEqual(outcome[0].cue_count, 1)
            self.assertEqual(outcome[0].original, prior.original)
            self.assertEqual(outcome[0].srt, prior.srt)
            self.assertTrue(outcome[1].srt.is_file())
            self.assertEqual(prior.original.read_bytes(), preserved)
            self.assertEqual(len(runner.commands) - count, 2)
            self.assertEqual(
                runner.commands[-1][runner.commands[-1].index("--sub-langs") + 1],
                "^ko$",
            )

    def test_opt_in_resume_never_modifies_partial_outputs(self):
        runner = FakeCaptionRunner({
            "id": "partial_vid",
            "subtitles": {"en": [{"ext": "vtt"}], "ko": [{"ext": "vtt"}]},
        })
        collector = PublicCaptionCollector("yt-dlp-test", command_runner=runner)
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            existing = root / "youtube-partial_vid-en-manual.srt"
            existing.write_bytes(b"KEEP EXISTING BYTES")
            outcomes = collector.collect(
                YOUTUBE, td, languages=("en", "ko"), skip_existing=True,
            )
            self.assertEqual([r.status for r in outcomes], [
                "existing_preserved", "downloaded",
            ])
            self.assertEqual(existing.read_bytes(), b"KEEP EXISTING BYTES")
            self.assertFalse((root / "youtube-partial_vid-en-manual.ass").exists())

    def test_non_resume_default_preserves_previous_collision_behavior(self):
        runner = FakeCaptionRunner({
            "id": "previous", "subtitles": {"en": [{"ext": "vtt"}]},
        })
        collector = PublicCaptionCollector("yt-dlp-test", command_runner=runner)
        with tempfile.TemporaryDirectory() as td:
            collector.collect(YOUTUBE, td, languages=("en",))
            with self.assertRaises(FileExistsError):
                collector.collect(YOUTUBE, td, languages=("en",))

    def test_automatic_caption_does_not_claim_manual_origin(self):
        runner = FakeCaptionRunner({
            "id": "vid", "subtitles": {},
            "automatic_captions": {"en-orig": [{"ext": "vtt"}]},
        })
        collector = PublicCaptionCollector("yt-dlp-test", command_runner=runner)
        with tempfile.TemporaryDirectory() as td:
            first = collector.collect(YOUTUBE, td, languages=("en",))[0]
            self.assertEqual(first.status, "downloaded")
            self.assertEqual(first.track_origin, "automatic")
            info = json.loads(first.manifest.read_text(encoding="utf-8"))
            self.assertEqual(info["provider_translation_status"], "original")
        self.assertIn("--write-auto-subs", runner.commands[1])
        self.assertNotIn("--write-subs", runner.commands[1])

    def test_failed_manual_caption_falls_back_to_original_auto_caption(self):
        metadata = {
            "id": "fallback_vid",
            "subtitles": {"en": [{"ext": "vtt"}]},
            "automatic_captions": {"en-orig": [{"ext": "vtt"}]},
        }

        class ManualFails(FakeCaptionRunner):
            def __call__(self, command, *, timeout_seconds, max_stdout_bytes):
                if "--write-subs" in command:
                    self.commands.append(list(command))
                    raise CaptionAccessError("manual subtitle not accessible")
                return super().__call__(
                    command, timeout_seconds=timeout_seconds,
                    max_stdout_bytes=max_stdout_bytes,
                )

        runner = ManualFails(metadata)
        collector = PublicCaptionCollector("yt-dlp-test", command_runner=runner)
        with tempfile.TemporaryDirectory() as td:
            result = collector.collect(YOUTUBE, td, languages=("en",))[0]
            self.assertEqual(result.status, "downloaded")
            self.assertEqual(result.track_language, "en-orig")
            self.assertEqual(result.track_origin, "automatic")
            self.assertEqual(result.original.read_bytes(), VTT)
        self.assertEqual(len(runner.commands), 3)
        self.assertIn("--write-subs", runner.commands[1])
        self.assertIn("--write-auto-subs", runner.commands[2])

    def test_failed_first_manual_format_tries_second_manual_before_auto(self):
        class BrokenVtt(FakeCaptionRunner):
            def __call__(self, command, *, timeout_seconds, max_stdout_bytes):
                if "--sub-format" in command and command[command.index("--sub-format") + 1] == "vtt":
                    self.commands.append(list(command))
                    raise CaptionAccessError("broken subtitle format")
                return super().__call__(
                    command, timeout_seconds=timeout_seconds,
                    max_stdout_bytes=max_stdout_bytes,
                )

        runner = BrokenVtt({
            "id": "manual_alternatives",
            "subtitles": {"en": [{"ext": "vtt"}, {"ext": "srt"}]},
            "automatic_captions": {"en-orig": [{"ext": "vtt"}]},
        })
        collector = PublicCaptionCollector("yt-dlp-test", command_runner=runner)
        with tempfile.TemporaryDirectory() as td:
            result = collector.collect(YOUTUBE, td, languages=("en",))[0]
            self.assertEqual(result.status, "downloaded")
            self.assertEqual(result.track_origin, "manual")
            self.assertTrue(result.srt.exists())
        self.assertEqual(len(runner.commands), 3)
        self.assertEqual(
            [command[command.index("--sub-format") + 1] for command in runner.commands[1:]],
            ["vtt", "srt"],
        )

    def test_real_child_process_can_be_cancelled_without_waiting_for_timeout(self):
        callbacks = 0

        def cancel():
            nonlocal callbacks
            callbacks += 1
            if callbacks >= 4:
                raise KeyboardInterrupt("cancel test")

        started = time.monotonic()
        with self.assertRaisesRegex(KeyboardInterrupt, "cancel test"):
            _run_silent(
                [sys.executable, "-c", "import time; time.sleep(5)"],
                timeout_seconds=10, max_stdout_bytes=100,
                check_cancelled=cancel,
            )
        self.assertLess(time.monotonic() - started, 4)

    def test_real_child_stdout_has_an_enforced_size_limit(self):
        with self.assertRaisesRegex(CaptionAccessError, "unexpectedly large"):
            _run_silent(
                [sys.executable, "-c", "print('A' * 1000)"],
                timeout_seconds=5, max_stdout_bytes=32,
            )

    def test_missing_download_is_unavailable_not_downloaded(self):
        runner = FakeCaptionRunner({
            "id": "test", "subtitles": {"en": [{"ext": "vtt"}]},
        }, missing=True)
        collector = PublicCaptionCollector("yt-dlp-test", command_runner=runner)
        with tempfile.TemporaryDirectory() as td:
            result = collector.collect(YOUTUBE, td, languages=("en",))[0]
            self.assertFalse(list(Path(td).iterdir()))
        self.assertEqual(result.status, "unavailable")

    def test_best_effort_wetv_uses_same_public_track_contract(self):
        runner = FakeCaptionRunner({
            "id": "wetv_id", "subtitles": {"zh-Hans": [{"ext": "vtt"}]},
        })
        collector = PublicCaptionCollector("yt-dlp-test", command_runner=runner)
        with tempfile.TemporaryDirectory() as td:
            result = collector.collect(WETV, td, languages=("zh-Hans",))[0]
            self.assertEqual(result.source, "wetv")
            self.assertEqual(result.track_language, "zh-Hans")
            self.assertEqual(result.status, "downloaded")
        # This synthetic test does NOT prove yt-dlp currently supports WeTV.

    def test_unknown_metadata_or_invalid_json_is_not_used_for_ocr(self):
        runner = FakeCaptionRunner({"_type": "playlist", "entries": [], "id": "playlist"})
        collector = PublicCaptionCollector("yt-dlp-test", command_runner=runner)
        with tempfile.TemporaryDirectory() as td:
            with self.assertRaises(CaptionAccessError):
                collector.collect(YOUTUBE, td)
        self.assertEqual(len(runner.commands), 1)

    def test_reject_bad_subtitle_language_from_provider(self):
        runner = FakeCaptionRunner({
            "id": "x", "subtitles": {"en-orig;rm": [{"ext": "vtt"}]},
        })
        collector = PublicCaptionCollector("yt-dlp-test", command_runner=runner)
        with tempfile.TemporaryDirectory() as td:
            result = collector.collect(YOUTUBE, td, languages=("en",))[0]
        self.assertEqual(result.status, "needs_ocr")

    def test_mislabeled_korean_track_is_not_silently_saved(self):
        with tempfile.TemporaryDirectory() as td:
            wrong_language_vtt = (
                "WEBVTT\n\n00:00:01.000 --> 00:00:02.000\n"
                "Entirely English subtitle only\n"
            ).encode("utf-8")
            track = SubtitleTrack("ko", "manual", "vtt", "ko")
            with self.assertRaisesRegex(ValueError, "lacks Korean text"):
                _commit_caption(
                    wrong_language_vtt, track, video_id="id",
                    provider="youtube", target_directory=Path(td),
                )
            self.assertFalse(list(Path(td).iterdir()))

    def test_preserve_actual_downloaded_extension_when_provider_format_differs(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            track = SubtitleTrack("en", "manual", "srt", "en")
            result = _commit_caption(
                VTT, track, video_id="test", provider="youtube",
                target_directory=root, original_extension=".vtt",
            )
            self.assertEqual(result.original.suffix, ".vtt")
            self.assertTrue(result.original.name.endswith(".original.vtt"))
            self.assertEqual(result.original.read_bytes(), VTT)
            self.assertEqual(result.srt.suffix, ".srt")
            info = json.loads(result.manifest.read_text(encoding="utf-8"))
            self.assertEqual(info["original_format"], "vtt")

    def test_independent_youtube_and_wetv_ids_do_not_collide(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            track = SubtitleTrack("en", "manual", "vtt", "en")
            youtube = _commit_caption(
                VTT, track, video_id="episode_1",
                provider="youtube", target_directory=root,
            )
            wetv = _commit_caption(
                VTT, track, video_id="episode_1",
                provider="wetv", target_directory=root,
            )
            self.assertNotEqual(youtube.original, wetv.original)
            self.assertNotEqual(youtube.srt, wetv.srt)
            self.assertEqual(youtube.original.read_bytes(), VTT)
            self.assertEqual(wetv.original.read_bytes(), VTT)
            self.assertTrue(youtube.original.name.startswith("youtube-"))
            self.assertTrue(wetv.original.name.startswith("wetv-"))

    def test_invalid_provenance_site_cannot_escape_output_directory(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            track = SubtitleTrack("en", "manual", "vtt", "en")
            with self.assertRaisesRegex(ValueError, "unsupported caption provenance"):
                _commit_caption(
                    VTT, track, video_id="episode_1",
                    provider="../../outside", target_directory=root,
                )
            self.assertEqual(list(root.iterdir()), [])

    def test_existing_raw_caption_collision_causes_no_change(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            original = root / "youtube-id-ko-manual.original.vtt"
            original.write_text("KEEP ORIGINAL", encoding="utf-8")
            track = SubtitleTrack("ko", "manual", "vtt", "ko")
            with self.assertRaises(FileExistsError):
                _commit_caption(VTT, track, video_id="id", provider="youtube", target_directory=root)
            self.assertEqual(original.read_text(encoding="utf-8"), "KEEP ORIGINAL")
            self.assertFalse((root / "youtube-id-ko-manual.srt").exists())


if __name__ == "__main__":
    unittest.main()
