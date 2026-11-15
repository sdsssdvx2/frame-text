"""Detector-only segment audit: reference labels never affect segment generation."""
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from diagnose_quiet_detector_windows import stage_for_pts, summarize_segments, PROFILES
from hardsub_extractor.models import Frame, VisualSegment
from hardsub_extractor.detector import DetectorConfig, iter_segment_frames


class DetectorWindowAuditTests(unittest.TestCase):
    def test_pending_detector_profiles_are_explicit_and_bounded(self):
        self.assertEqual(PROFILES[0], ("default", None, None))
        self.assertEqual(len(PROFILES), len(set(PROFILES)))
        self.assertEqual(
            {t for _, _, t in PROFILES if t is not None},
            {0.375, 0.5},
        )
        self.assertTrue(
            all(cap is not None for _, cap, timeout in PROFILES
                if timeout is not None)
        )

    def test_reference_labels_are_posthoc_only(self):
        self.assertEqual(stage_for_pts(1.5), "first_caption")
        self.assertEqual(stage_for_pts(3.625), "caption_gap")
        self.assertEqual(stage_for_pts(4.25), "second_caption")
        self.assertEqual(stage_for_pts(6.375), "other_background")
        self.assertEqual(stage_for_pts(float("nan")), "unknown")

    def test_segment_summary_keeps_boundary_kind_and_bounded_sample_times(self):
        def frame(t):
            return Frame(t, np.zeros((8, 8, 3), dtype=np.uint8))
        segments = [
            VisualSegment(0.0, 3.625, [frame(3.375)], {"settled_by": "old_baseline"}),
            VisualSegment(3.625, 4.25, [frame(3.625), frame(4.0)],
                          {"quiet_duration_cap": True}),
            VisualSegment(4.25, 4.875, [frame(4.25), frame(4.625)],
                          {"quiet_duration_cap": True}),
            VisualSegment(4.875, 6.375, [frame(5.0), frame(6.25)],
                          {"eof": True}),
        ]
        result = summarize_segments(segments)
        self.assertEqual([r["boundary_kind"] for r in result],
                         ["old_baseline", "quiet_cap", "quiet_cap", "eof"])
        self.assertEqual(result[1]["posthoc_sample_stage"],
                         ["caption_gap", "caption_gap"])
        self.assertEqual(result[2]["posthoc_sample_stage"],
                         ["second_caption", "second_caption"])
        self.assertEqual(result[1]["selected_pts"], [3.625, 4.0])

    def test_opt_in_detector_trace_is_bounded_numeric_only(self):
        frames = [
            Frame(i * 0.125, np.zeros((8, 8, 3), dtype=np.uint8))
            for i in range(25)
        ]
        events = []
        intervals = list(iter_segment_frames(
            frames,
            DetectorConfig(max_quiet_segment_seconds=0.625),
            trace=events.append,
        ))
        self.assertGreater(len(intervals), 1)
        self.assertTrue(any(e["kind"] == "quiet_cap" for e in events))
        for e in events:
            self.assertTrue({"kind", "at_pts", "current_start",
                             "pending_start"}.issubset(e))
            self.assertFalse(any(
                forbidden in repr(e).lower()
                for forbidden in ("source_path", "frame_image",
                                  "transcript", "url", "token")
            ))

    def test_detector_trace_absent_by_default_and_same_boundaries(self):
        frames = [
            Frame(i * 0.125, np.zeros((8, 8, 3), dtype=np.uint8))
            for i in range(25)
        ]
        plain = list(iter_segment_frames(
            frames, DetectorConfig(max_quiet_segment_seconds=0.75)
        ))
        events = []
        traced = list(iter_segment_frames(
            frames, DetectorConfig(max_quiet_segment_seconds=0.75),
            trace=events.append,
        ))
        self.assertEqual(
            [(r.start, r.end, [f.pts for f in r.samples]) for r in plain],
            [(r.start, r.end, [f.pts for f in r.samples]) for r in traced],
        )

    def test_no_cues_or_source_text_are_in_audit(self):
        segment = VisualSegment(
            4.25, 4.75, [Frame(4.5, np.zeros((8, 8, 3), dtype=np.uint8))],
            {"quiet_duration_cap": True},
        )
        report = summarize_segments([segment])
        self.assertFalse(any("text" in key or "frame" in key
                             for key in report[0]))


if __name__ == "__main__":
    unittest.main()
