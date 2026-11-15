"""Generalization comparison matrix static and temporal contracts."""
from __future__ import annotations

from pathlib import Path
import sys
import unittest
from unittest import mock
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import compare_vsf_generalization as target
from compare_external_engine_exports import Cue
from compare_vsf_timing_research import timing_report


class NativeDetectorGeneralizationTests(unittest.TestCase):
    def test_languages_cover_requested_four(self):
        self.assertEqual(target.LANGUAGES, ("ko", "en", "ja", "zh"))

    def test_styles_cover_both_two_lines_and_color(self):
        self.assertEqual(target.STYLES, ("white_two_line", "yellow_single"))

    def test_eight_source_and_eight_blank_runs_required(self):
        self.assertEqual(
            len(target.LANGUAGES) * len(target.STYLES), 8,
        )

    def test_expected_source_duration_is_independent(self):
        from verify_multilang_video_pipeline import EXPECTED_SPANS
        self.assertEqual(len(EXPECTED_SPANS), 2)
        self.assertEqual(timing_report(
            [Cue(a, b, "?") for a,b in EXPECTED_SPANS]
        )["both_edges_within_250ms"], 2)

    def test_no_overlap_control_fails_labeled_timing(self):
        s = timing_report([Cue(99, 100, "?")])
        self.assertEqual(s["matched_reference_intervals"], 0)
        self.assertEqual(s["extra_timing_intervals"], 1)
        self.assertEqual(s["missed_reference_intervals"], 2)

    def test_different_caption_color_is_not_a_reference_oracle(self):
        # Feature test: reference enters only timing_report, not native.
        from inspect import getsource
        code = getsource(target.main)
        self.assertNotIn("VIDEO_TEXT", code)
        self.assertIn("include_subtitles=False", code)
        self.assertIn("timing_report(cues)", code)
        self.assertIn("run_vsf(exe, positive", code)

    def test_invalid_style_not_silently_accepted(self):
        from verify_animated_background_ocr_quality import subtitle_overlay
        with self.assertRaises(ValueError):
            subtitle_overlay("x", Path("fake.ttf"), style="unknown")

    def test_sha_verified_native_loader_used(self):
        from inspect import getsource
        code = getsource(target.main)
        self.assertIn("download_pinned_archive", code)
        self.assertIn("unpack_pinned_archive", code)
        self.assertIn("check_native_version", code)
        self.assertIn("source_sha", code)
        self.assertIn("blank_sha", code)


if __name__ == "__main__":
    unittest.main()
