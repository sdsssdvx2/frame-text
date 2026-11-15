"""Offline, fail-closed local embedded text subtitle acquisition contracts."""
from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from hardsub_extractor.local_soft_subtitles import (
    LocalSoftSubtitleError, _probe_streams,
    acquire_local_embedded_text,
)


KOREAN = (
    "1\n00:00:01,000 --> 00:00:02,500\n안녕하세요\n\n"
    "2\n00:00:03,000 --> 00:00:04,000\n다음 자막\n"
).encode("utf-8")
ENGLISH = (
    "1\n00:00:01,000 --> 00:00:02,500\nHello there!\n\n"
    "2\n00:00:03,000 --> 00:00:04,000\nHave a nice day.\n"
).encode("utf-8")


def inventory(*streams):
    return json.dumps({"streams": [
        {"index": 0, "codec_type": "video", "codec_name": "mpeg4"},
        *streams,
    ]}).encode("utf-8")


def subtitle(index, language, *, codec="subrip", default=0, forced=0):
    return {
        "index": index,
        "codec_type": "subtitle",
        "codec_name": codec,
        "tags": {"language": language},
        "disposition": {"default": default, "forced": forced},
    }


class LocalEmbeddedSubtitleTests(unittest.TestCase):
    def fixture(self, root):
        source = root / "한글 경로 [01].mkv"
        source.write_bytes(b"original-local-video-do-not-modify")
        return source

    def collect(self, source, base, answer, *exports, language="ko"):
        with mock.patch(
            "hardsub_extractor.local_soft_subtitles.resolve_binary",
            side_effect=lambda name: "PINNED_" + name,
        ), mock.patch(
            "hardsub_extractor.local_soft_subtitles._run_bounded",
            side_effect=[answer, *exports],
        ) as runner:
            result = acquire_local_embedded_text(source, base, language=language)
        return result, runner

    def test_matching_korean_embedded_track_preferred_and_provenance_retained(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            source = self.fixture(root)
            original = source.read_bytes()
            base = root / "outputs" / "episode"
            result, runner = self.collect(
                source, base, inventory(
                    subtitle(1, "eng"), subtitle(2, "kor", codec="ass", default=1),
                ), KOREAN,
            )
            self.assertEqual(result.mode, "local-embedded-text")
            self.assertEqual(result.track_language, "kor")
            self.assertEqual(result.stream_index, 2)
            self.assertEqual(result.cue_count, 2)
            self.assertIn("안녕하세요", result.srt_path.read_text("utf-8"))
            self.assertIn("안녕하세요", result.ass_path.read_text("utf-8"))
            self.assertEqual(result.extracted_track_path.read_bytes(), KOREAN)
            metadata = json.loads(result.provenance_path.read_text("utf-8"))
            self.assertFalse(metadata["ocr_applied"])
            self.assertEqual(metadata["source_stream_export"], "ffmpeg_decoded_srt_not_original_container_bytes")
            self.assertNotIn(str(root), result.provenance_path.read_text("utf-8"))
            self.assertEqual(source.read_bytes(), original)
            self.assertEqual(runner.call_count, 2)
            args = runner.call_args.args[0]
            self.assertIn("0:2", args)
            self.assertIn("file,pipe", args)
            self.assertNotIn("--enable-url", args)

    def test_english_stream_is_not_labeled_korean(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            source = self.fixture(root)
            result, runner = self.collect(
                source, root / "out", inventory(subtitle(1, "eng")),
            )
            self.assertIsNone(result)
            self.assertEqual(runner.call_count, 1)
            self.assertEqual(sorted(p.name for p in root.iterdir()),
                             [source.name])

    def test_english_uses_only_matching_language_track(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            source = self.fixture(root)
            result, runner = self.collect(
                source, root / "english", inventory(
                    subtitle(1, "kor"), subtitle(2, "eng"),
                ), ENGLISH, language="en",
            )
            self.assertEqual(result.cue_count, 2)
            self.assertEqual(result.track_language, "eng")
            self.assertIn("0:2", runner.call_args.args[0])

    def test_unknown_language_is_not_guessed_as_japanese_or_chinese(self):
        for language in ("ja", "zh"):
            with self.subTest(language=language), tempfile.TemporaryDirectory() as td:
                root = Path(td)
                source = self.fixture(root)
                result, runner = self.collect(
                    source, root / "out",
                    inventory(subtitle(1, "und")), language=language,
                )
                self.assertIsNone(result)
                self.assertEqual(runner.call_count, 1)

    def test_bitmap_only_track_does_not_fake_text_soft_subtitles(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            source = self.fixture(root)
            result, runner = self.collect(
                source, root / "out",
                inventory(subtitle(1, "kor", codec="hdmv_pgs_subtitle")),
            )
            self.assertIsNone(result)
            self.assertEqual(runner.call_count, 1)

    def test_prefers_nonforced_default_over_forced_caption(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            source = self.fixture(root)
            result, runner = self.collect(
                source, root / "out", inventory(
                    subtitle(1, "kor", forced=1, default=1),
                    subtitle(2, "kor", default=1),
                ), KOREAN,
            )
            self.assertEqual(result.stream_index, 2)
            self.assertIn("0:2", runner.call_args.args[0])

    def test_mislabeled_korean_track_is_rejected_before_writing(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            source = self.fixture(root)
            result, runner = self.collect(
                source, root / "out", inventory(subtitle(1, "kor")),
                ENGLISH,
            )
            self.assertIsNone(result)
            self.assertEqual(runner.call_count, 2)
            self.assertFalse((root / "out.srt").exists())

    def test_bad_first_matching_track_tries_good_second_without_partial_output(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            source = self.fixture(root)
            result, runner = self.collect(
                source, root / "out", inventory(
                    subtitle(1, "kor"), subtitle(2, "kor"),
                ), b"not valid srt", KOREAN,
            )
            self.assertEqual(result.stream_index, 2)
            self.assertEqual(runner.call_count, 3)

    def test_existing_result_guard_fails_before_ffprobe_and_does_not_mutate(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            source = self.fixture(root)
            base = root / "out"
            original = b"important original subtitles"
            (root / "out.srt").write_bytes(original)
            with mock.patch("hardsub_extractor.local_soft_subtitles._run_bounded") as runner:
                with self.assertRaises(FileExistsError):
                    acquire_local_embedded_text(source, base)
            runner.assert_not_called()
            self.assertEqual((root / "out.srt").read_bytes(), original)
            self.assertFalse((root / "out.ass").exists())

    def test_existing_extracted_provenance_result_is_also_protected(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            source = self.fixture(root)
            original = b"existing provenance"
            report = root / "out.source.json"
            report.write_bytes(original)
            with self.assertRaises(FileExistsError):
                acquire_local_embedded_text(source, root / "out")
            self.assertEqual(report.read_bytes(), original)

    def test_invalid_probe_inventory_fails_closed_not_ocr_fallback(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            source = self.fixture(root)
            with mock.patch(
                "hardsub_extractor.local_soft_subtitles.resolve_binary",
                return_value="ffprobe",
            ), mock.patch(
                "hardsub_extractor.local_soft_subtitles._run_bounded",
                return_value=b'{"unexpected":"missing streams"}',
            ):
                with self.assertRaisesRegex(LocalSoftSubtitleError, "inventory"):
                    acquire_local_embedded_text(source, root / "out")
            self.assertFalse((root / "out.srt").exists())

    def test_duplicate_stream_indices_are_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            source = self.fixture(root)
            with mock.patch(
                "hardsub_extractor.local_soft_subtitles.resolve_binary",
                return_value="ffprobe",
            ), mock.patch(
                "hardsub_extractor.local_soft_subtitles._run_bounded",
                return_value=inventory(subtitle(1, "kor"), subtitle(1, "eng")),
            ):
                with self.assertRaisesRegex(LocalSoftSubtitleError, "index"):
                    _probe_streams(source)

    def test_invalid_path_does_not_attempt_ocr_or_network(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            with mock.patch("hardsub_extractor.local_soft_subtitles._run_bounded") as runner:
                with self.assertRaises(ValueError):
                    acquire_local_embedded_text(root / "not-a-video", root / "out")
            runner.assert_not_called()

    def test_atomic_commit_rollback_if_provenance_creation_collides(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            source = self.fixture(root)
            original = b"source-metadata-must-survive"
            base = root / "out"
            raw = inventory(subtitle(1, "kor"))
            # Simulate a separate process creating .source.json AFTER our
            # initial exclusive collision preflight but BEFORE the publish.
            def after_export(command, **kwargs):
                if command[0] == "PINNED_ffmpeg":
                    (root / "out.source.json").write_bytes(original)
                    return KOREAN
                return raw
            with mock.patch("hardsub_extractor.local_soft_subtitles.resolve_binary",
                            side_effect=lambda name: "PINNED_" + name), mock.patch(
                "hardsub_extractor.local_soft_subtitles._run_bounded",
                side_effect=after_export,
            ):
                with self.assertRaises(FileExistsError):
                    acquire_local_embedded_text(source, base)
            self.assertEqual((root / "out.source.json").read_bytes(), original)
            self.assertFalse((root / "out.srt").exists())
            self.assertFalse((root / "out.ass").exists())
            self.assertFalse((root / "out.embedded.extracted.srt").exists())


if __name__ == "__main__":
    unittest.main()
