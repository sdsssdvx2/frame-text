"""Isolated tests of VSF timing + real adapter candidate wiring."""
from __future__ import annotations

from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import compare_vsf_hse_ocr_research as t
from compare_external_engine_exports import Cue as ScoreCue
from hardsub_extractor.models import Frame, OCRObservation
from verify_multilang_video_pipeline import EXPECTED_SPANS, VIDEO_TEXT
from verify_animated_background_ocr_quality import _multiline_reference


class VSFAndRealOCRContracts(unittest.TestCase):
    def test_sample_fractions_are_fixed_before_ground_truth(self):
        cue = ScoreCue(1.5, 3.625, "[placeholder]")
        points = t.selection_targets(cue)
        self.assertEqual(len(points), 5)
        self.assertTrue(all(cue.start < p < cue.end for p in points))
        self.assertEqual(points[2], (cue.start + cue.end) / 2)

    def test_invalid_native_interval_rejected(self):
        for start, end in ((1, 1), (float("nan"), 2), (2, 1)):
            with self.subTest(start=start, end=end):
                with self.assertRaises(ValueError):
                    t.selection_targets(ScoreCue(start, end, "x"))

    def test_mismatched_frame_counts_fatal(self):
        with self.assertRaises(ValueError):
            t.predict_from_frames([ScoreCue(1, 2, "x")], [], object())

    def test_adapter_voting_ignores_truth_text(self):
        class Fixed:
            def __init__(self):
                self.calls = 0
            def recognize(self, frame):
                self.calls += 1
                return OCRObservation("字幕\n第二行", .99, frame.pts)
        adapter = Fixed()
        intervals = [ScoreCue(1, 3, "[sub_duration]")]
        images = [[Frame(1.5, np.zeros((30, 30, 3), dtype=np.uint8)),
                   Frame(2.1, np.zeros((30, 30, 3), dtype=np.uint8))]]
        cues, calls = t.predict_from_frames(intervals, images, adapter)
        self.assertEqual(calls, 2)
        self.assertEqual(adapter.calls, 2)
        self.assertEqual([(x.start, x.end) for x in cues], [(1, 3)])
        self.assertEqual(cues[0].text, "字幕\n第二行")

    def test_blank_ocr_observations_produce_measured_miss(self):
        class Blank:
            def recognize(self, frame):
                return OCRObservation("", 0, frame.pts)
        intervals = [ScoreCue(1, 3, "[sub_duration]")]
        images = [[Frame(1.5, np.zeros((5, 5, 3), dtype=np.uint8))]]
        cues, calls = t.predict_from_frames(intervals, images, Blank())
        self.assertEqual(calls, 1)
        self.assertEqual(cues, [])
        with tempfile.TemporaryDirectory() as td:
            score = t.validated_output_score(
                cues, [ScoreCue(1, 3, "정답")],
                Path(td) / "result", [],
            )
        self.assertEqual(score["missed_reference_cues"], 1)
        self.assertEqual(score["aggregate_character_error_rate"], 1)

    def test_generated_srt_and_ass_same_exact_two_lines(self):
        with tempfile.TemporaryDirectory() as td:
            ref = [
                ScoreCue(x[0], x[1], "\n".join(_multiline_reference(y)))
                for x, y in zip(EXPECTED_SPANS, VIDEO_TEXT["ja"])
            ]
            cues = [t.Cue(a.start, a.end, a.text) for a in ref]
            result = t.validated_output_score(
                cues, ref, Path(td) / "original", [],
            )
        self.assertEqual(result["strict_both_edges_250ms_and_text_wrap"], 2)
        self.assertTrue(result["all_reference_cues_exact_both_edges_250ms"])

    def test_rendered_reference_preserves_two_physical_rows(self):
        expected = "\n".join(_multiline_reference(VIDEO_TEXT["ja"][0]))
        self.assertEqual(len(expected.splitlines()), 2)
        self.assertNotEqual(expected, VIDEO_TEXT["ja"][0])
        with tempfile.TemporaryDirectory() as td:
            ref = [ScoreCue(1.5, 3.625, expected)]
            good = t.validated_output_score(
                [t.Cue(1.5, 3.625, expected)], ref,
                Path(td) / "correct", [],
            )
            wrong = t.validated_output_score(
                [t.Cue(1.5, 3.625, VIDEO_TEXT["ja"][0])], ref,
                Path(td) / "wrong", [],
            )
        self.assertEqual(good["strict_both_edges_250ms_and_text_wrap"], 1)
        self.assertEqual(wrong["strict_both_edges_250ms_and_text_wrap"], 0)

    def test_nonmatching_text_does_not_pass_srt_ass(self):
        with tempfile.TemporaryDirectory() as td:
            result = t.validated_output_score(
                [t.Cue(1.5, 3.625, "誤字\n第二行")],
                [ScoreCue(1.5, 3.625, "正字\n第二行")],
                Path(td) / "test", [],
            )
            self.assertEqual(result["strict_both_edges_250ms_and_text_wrap"], 0)
            self.assertGreater(result["aggregate_character_error_rate"], 0)

    def test_no_caption_native_intervals_are_counted(self):
        with tempfile.TemporaryDirectory() as td:
            result = t.validated_output_score(
                [t.Cue(1.5, 3.625, "正字\n第二行")],
                [ScoreCue(1.5, 3.625, "正字\n第二行")],
                Path(td) / "test",
                [ScoreCue(2, 3, "[sub_duration]")],
            )
            self.assertEqual(result["blank_video_false_cues"], 1)
            self.assertFalse(result["all_reference_cues_exact_both_edges_250ms"])

    def test_source_frames_selected_by_video_pts_and_native_intervals(self):
        from verify_multilang_video_pipeline import EXPECTED_SPANS
        intervals = [ScoreCue(a, b, "[placeholder]") for a, b in EXPECTED_SPANS]
        def frames(*_, **__):
            for i in range(8 * 8):
                pts = i / 8
                yield Frame(pts, np.zeros((2, 3, 3), dtype=np.uint8))
        with mock.patch.object(t, "FFmpegPTSFrameSource", side_effect=frames):
            groups = t.original_pts_frames(Path("not-read"), intervals)
        self.assertEqual(len(groups), 2)
        self.assertTrue(all(len(x) == 5 for x in groups))
        for group, interval in zip(groups, intervals):
            self.assertTrue(all(interval.start <= f.pts < interval.end for f in group))

    def test_maliciously_excessive_intervals_rejected(self):
        excessive = [ScoreCue(float(i), float(i + 1), "x")
                     for i in range(t.MAX_TIMED_CUES + 1)]
        with self.assertRaises(ValueError):
            t.original_pts_frames(Path("not-read"), excessive)


if __name__ == "__main__":
    unittest.main()
