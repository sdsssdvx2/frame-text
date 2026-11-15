"""Offline post-export evidence; no model or user media handling."""
from __future__ import annotations

from pathlib import Path
from unittest import TestCase, mock

from hardsub_extractor.local_ocr_quality_signals import (
    REVIEW_CODES, local_ocr_quality_signals,
)
from hardsub_extractor.models import Cue


class LocalPostExportReviewTests(TestCase):
    def test_suspicious_persistent_single_cue_is_review_not_deletion(self):
        cues = [Cue(0.2, 99.1, "WATERMARK", 0.7)]
        with (
            mock.patch(
                "hardsub_extractor.local_ocr_quality_signals._read_subtitles",
                return_value=cues,
            ) as read,
            mock.patch(
                "hardsub_extractor.local_ocr_quality_signals._probe_local_video_duration",
                return_value=100.0,
            ) as probe,
        ):
            result = local_ocr_quality_signals(
                Path("source.mp4"), Path("output.srt"), language="ja",
            )
        self.assertEqual(result["status"], "REVIEW_SUGGESTED")
        self.assertGreaterEqual(result["review_signal_count"], 1)
        self.assertIn("near_full_runtime_single_cue_review", result["signal_codes"])
        self.assertEqual(result["review_cue_numbers"], (1,))
        self.assertFalse(result["accuracy_verified"])
        read.assert_called_once_with(Path("output.srt"))
        probe.assert_called_once_with(Path("source.mp4"))
        self.assertEqual(cues[0].text, "WATERMARK")

    def test_oversized_cue_is_promoted_to_source_frame_review(self):
        cues = [Cue(1, 3, "A" * 601, .8)]
        with (
            mock.patch(
                "hardsub_extractor.local_ocr_quality_signals._read_subtitles",
                return_value=cues,
            ),
            mock.patch(
                "hardsub_extractor.local_ocr_quality_signals._probe_local_video_duration",
                return_value=10.0,
            ),
        ):
            result = local_ocr_quality_signals(
                Path("source.mp4"), Path("output.srt"), language="en",
            )
        self.assertEqual(result["status"], "REVIEW_SUGGESTED")
        self.assertIn("oversized_ocr_cue_review", result["signal_codes"])
        self.assertEqual(result["review_cue_numbers"], (1,))
        self.assertEqual(cues[0].text, "A" * 601)

    def test_normal_dialogue_is_not_claimed_verified(self):
        cues = [Cue(1, 2.5, "Hello there"), Cue(4, 6, "Second line")]
        with (
            mock.patch(
                "hardsub_extractor.local_ocr_quality_signals._read_subtitles",
                return_value=cues,
            ),
            mock.patch(
                "hardsub_extractor.local_ocr_quality_signals._probe_local_video_duration",
                return_value=15.0,
            ),
        ):
            result = local_ocr_quality_signals(
                Path("source.mp4"), Path("output.srt"), language="en",
            )
        self.assertEqual(result["status"], "NO_MAJOR_HEURISTIC_WARNING")
        self.assertEqual(result["review_signal_count"], 0)
        self.assertFalse(result["accuracy_verified"])
        self.assertTrue(result["source_and_subtitle_unchanged"])

    def test_long_episode_shows_bounded_exact_review_indices_not_text(self):
        cues = [Cue(float(i), float(i) + .1, "PRIVATE CHARACTER TEXT", .9)
                for i in range(20)]
        with (
            mock.patch(
                "hardsub_extractor.local_ocr_quality_signals._read_subtitles",
                return_value=cues,
            ),
            mock.patch(
                "hardsub_extractor.local_ocr_quality_signals._probe_local_video_duration",
                return_value=30.0,
            ),
        ):
            result = local_ocr_quality_signals(
                Path("private_video.mp4"), Path("private_result.srt"), language="en",
            )
        self.assertGreaterEqual(result["review_signal_count"], 20)
        self.assertEqual(result["review_cue_numbers"], tuple(range(1, 9)))
        self.assertNotIn("PRIVATE CHARACTER TEXT", repr(result))
        self.assertNotIn("private_video.mp4", repr(result))

    def test_no_duration_witness_fails_unavailable_not_fake_success(self):
        with (
            mock.patch(
                "hardsub_extractor.local_ocr_quality_signals._read_subtitles",
                return_value=[Cue(1, 3, "Hello")],
            ),
            mock.patch(
                "hardsub_extractor.local_ocr_quality_signals._probe_local_video_duration",
                side_effect=ValueError("duration unavailable"),
            ),
        ):
            with self.assertRaises(ValueError):
                local_ocr_quality_signals(
                    Path("source.mp4"), Path("output.srt"), language="en",
                )

    def test_major_signal_list_does_not_include_ordinary_dialogue_gaps(self):
        self.assertNotIn("long_gap_review", REVIEW_CODES)
        self.assertNotIn("fast_reading_review", REVIEW_CODES)
        self.assertNotIn("no_japanese_script_review", REVIEW_CODES)
