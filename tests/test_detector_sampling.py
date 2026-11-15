import unittest
from unittest import mock

import cv2
import numpy as np

from hardsub_extractor.detector import (
    DetectorConfig, _append_sample, iter_segment_frames, persistent_white_signature,
)
from hardsub_extractor.models import Frame


def frame(pts: float) -> Frame:
    return Frame(pts, np.zeros((8, 8, 3), dtype=np.uint8))


class DetectorSamplingTests(unittest.TestCase):
    def test_three_sample_budget_keeps_first_and_two_recent_samples(self):
        cfg = DetectorConfig(max_samples_per_segment=3, sample_spacing=0.18)
        samples = [frame(0.0)]

        for pts in (0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0):
            _append_sample(samples, frame(pts), cfg)

        self.assertEqual(len(samples), 3)
        self.assertAlmostEqual(samples[0].pts, 0.0)
        self.assertAlmostEqual(samples[-1].pts, 2.0)
        self.assertGreaterEqual(samples[1].pts, 1.6)
        self.assertLess(samples[1].pts, samples[-1].pts)

    def test_late_slots_roll_forward_in_long_segment(self):
        cfg = DetectorConfig(max_samples_per_segment=3, sample_spacing=0.18)
        samples = [frame(12.8)]

        for pts in (13.0, 13.2, 13.4, 13.6, 13.8, 14.0, 14.2, 14.4, 14.6, 14.8, 15.0, 15.13):
            _append_sample(samples, frame(pts), cfg)

        self.assertEqual(len(samples), 3)
        self.assertAlmostEqual(samples[0].pts, 12.8)
        self.assertAlmostEqual(samples[-1].pts, 15.13)
        self.assertGreaterEqual(samples[1].pts, 14.8)

    def test_spacing_still_controls_initial_sample_fill(self):
        cfg = DetectorConfig(max_samples_per_segment=3, sample_spacing=0.18)
        samples = [frame(0.0)]
        _append_sample(samples, frame(0.05), cfg)
        _append_sample(samples, frame(0.10), cfg)
        self.assertEqual([sample.pts for sample in samples], [0.0])

    def test_true_tail_is_kept_even_when_last_frame_is_close(self):
        cfg = DetectorConfig(max_samples_per_segment=3, sample_spacing=0.18)
        samples = [frame(0.0), frame(0.2), frame(0.4)]
        _append_sample(samples, frame(0.51), cfg)
        self.assertAlmostEqual(samples[-1].pts, 0.51)

    def test_extended_budgets_roll_middle_slots_instead_of_freezing_old_frames(self):
        for limit in (4, 5, 7):
            with self.subTest(limit=limit):
                cfg = DetectorConfig(
                    max_samples_per_segment=limit, sample_spacing=0.18,
                )
                samples = [frame(0.0)]
                for tick in range(1, 41):
                    _append_sample(samples, frame(tick * 0.25), cfg)
                self.assertEqual(len(samples), limit)
                self.assertAlmostEqual(samples[0].pts, 0.0)
                self.assertAlmostEqual(samples[-1].pts, 10.0)
                self.assertTrue(
                    all(a.pts < b.pts for a, b in zip(samples, samples[1:])),
                )
                # The usable non-anchor budget contains *recent* samples,
                # not 0.25, 0.5, ... fixed at the start of the segment.
                self.assertGreaterEqual(
                    samples[1].pts, 10.0 - (limit - 1) * 0.25,
                )

    def test_extended_budget_is_bounded_when_frames_come_faster_than_spacing(self):
        cfg = DetectorConfig(max_samples_per_segment=7, sample_spacing=0.18)
        samples = [frame(0.0)]
        for tick in range(1, 1001):
            _append_sample(samples, frame(tick * 0.125), cfg)
            self.assertLessEqual(len(samples), 7)
        self.assertEqual(len(samples), 7)
        self.assertAlmostEqual(samples[0].pts, 0.0)
        self.assertAlmostEqual(samples[-1].pts, 125.0)
        self.assertTrue(
            all(x.pts < y.pts for x, y in zip(samples, samples[1:])),
        )
        self.assertGreaterEqual(samples[1].pts, 123.0)

    def test_opt_in_quiet_cap_bounds_segments_without_duplicate_frames(self):
        frames = [frame(i * 0.125) for i in range(40)]
        capped = list(iter_segment_frames(
            frames, DetectorConfig(max_quiet_segment_seconds=1.0),
        ))
        self.assertEqual(len(capped), 5)
        for i, segment in enumerate(capped):
            with self.subTest(segment=i):
                self.assertLessEqual(segment.end-segment.start, 1.001)
                self.assertLessEqual(len(segment.samples), 3)
                self.assertTrue(all(
                    segment.start <= sample.pts < segment.end
                    for sample in segment.samples
                ))
                if i:
                    self.assertAlmostEqual(segment.start, capped[i-1].end)
        self.assertEqual(
            sum(bool(s.diagnostics.get("quiet_duration_cap")) for s in capped),
            4,
        )
        self.assertAlmostEqual(capped[-1].end, 5.0)

    def test_default_quiet_cap_is_inert_on_identical_source(self):
        frames = [frame(i * 0.125) for i in range(40)]
        normal = list(iter_segment_frames(frames))
        explicit = list(iter_segment_frames(
            frames, DetectorConfig(max_quiet_segment_seconds=None),
        ))
        self.assertEqual(len(normal), 1)
        self.assertEqual(
            [(s.start, s.end, [x.pts for x in s.samples]) for s in normal],
            [(s.start, s.end, [x.pts for x in s.samples]) for s in explicit],
        )

    def test_nonfinite_or_unsafe_opt_in_quiet_caps_are_rejected(self):
        frames = [frame(i * 0.125) for i in range(3)]
        bad = (float("nan"), float("inf"), -1.0, 0.49, 30.01,
               True, False, "1.25")
        for setting in bad:
            with self.subTest(setting=repr(setting)), self.assertRaises(ValueError):
                list(iter_segment_frames(
                    frames, DetectorConfig(max_quiet_segment_seconds=setting),
                ))

    def test_optional_pending_timeout_splits_unsettled_scene_change(self):
        # 6 quiet frames establish an old zero-change regime; a hard
        # moving-background change then persists and cannot stabilize
        # against that regime before the explicitly requested deadline.
        frames = [frame(i * 0.125) for i in range(17)]
        scores = [0.0] * 6 + [0.20] * 10
        recorder = []
        with mock.patch(
            "hardsub_extractor.detector.signature_change", side_effect=scores
        ):
            intervals = list(iter_segment_frames(
                frames,
                DetectorConfig(max_pending_change_seconds=0.375),
                trace=recorder.append,
            ))
        timeout = [e for e in recorder
                   if e["kind"] == "pending_confirm"
                   and e["reason"] == "pending_timeout"]
        self.assertGreaterEqual(len(timeout), 1)
        self.assertEqual(timeout[0]["boundary"], 0.875)
        self.assertEqual(timeout[0]["at_pts"], 1.25)
        self.assertEqual(intervals[0].end, 0.875)
        self.assertLessEqual(max(len(s.samples) for s in intervals), 3)
        self.assertTrue(all(
            a.end <= b.start for a, b in zip(intervals, intervals[1:])
        ))

    def test_pending_timeout_is_inert_on_blank_moving_control(self):
        frames = [frame(i * 0.125) for i in range(40)]
        records = []
        plain = list(iter_segment_frames(frames))
        optional = list(iter_segment_frames(
            frames, DetectorConfig(max_pending_change_seconds=0.375),
            trace=records.append,
        ))
        self.assertEqual(
            [(s.start, s.end, [f.pts for f in s.samples]) for s in plain],
            [(s.start, s.end, [f.pts for f in s.samples]) for s in optional],
        )
        self.assertFalse(any(
            event["kind"] == "pending_confirm" for event in records
        ))

    def test_invalid_pending_timeout_settings_fail_closed(self):
        frames = [frame(i * 0.125) for i in range(3)]
        for bad in (float("nan"), float("inf"), -1.0, 0.249,
                    5.01, True, False, "0.375", None if False else ""):
            with self.subTest(setting=repr(bad)), self.assertRaises(ValueError):
                list(iter_segment_frames(
                    frames,
                    DetectorConfig(max_pending_change_seconds=bad),
                ))

    def test_opt_in_caption_band_sees_lower_white_overlay_without_cropping_ocr_frame(self):
        image = np.zeros((240, 320, 3), dtype=np.uint8)
        cv2.putText(
            image, "CAPTION", (40, 194),
            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2,
        )
        original = persistent_white_signature()
        opt_in = persistent_white_signature(band_start=0.38, band_end=0.90)
        original_masks = [original(image) for _ in range(3)]
        opt_in_masks = [opt_in(image) for _ in range(3)]
        self.assertFalse(any(mask.any() for mask in original_masks))
        self.assertFalse(any(mask.any() for mask in opt_in_masks[:2]))
        self.assertTrue(opt_in_masks[-1].any())
        frames = [Frame(i * 0.125, image) for i in range(16)]
        result = list(iter_segment_frames(
            frames, DetectorConfig(signature_mode="caption_band_white_3"),
        ))
        self.assertTrue(result)
        self.assertEqual(len(result[0].samples[0].image.shape), 3)
        self.assertEqual(result[0].samples[0].image.shape, (240, 320, 3))

    def test_three_sample_default_remains_exactly_unchanged(self):
        cfg = DetectorConfig(max_samples_per_segment=3, sample_spacing=0.18)
        samples = [frame(0.0)]
        for tick in range(1, 10):
            _append_sample(samples, frame(tick * 0.25), cfg)
        self.assertEqual([x.pts for x in samples], [0.0, 2.0, 2.25])



if __name__ == "__main__":
    unittest.main()
