"""Deterministic checks for OCR-seeded visual boundary recovery experiment.

No model downloads, credentials, media streaming or production changes.
"""
from __future__ import annotations

from pathlib import Path
import sys
import unittest
from unittest import mock

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import experiment_ink_boundary_refinement as study
from experiment_ink_boundary_refinement import (
    _ocr_box, _evaluation, build_anchor, ink_support, infer_presence,
    hysteresis_presence,
)
from hardsub_extractor.models import Frame


def rendered(pts: float, *, caption: bool, moving: int = 0) -> Frame:
    img = np.zeros((130, 340, 3), dtype=np.uint8)
    img[:, :] = (71, 85, 103)
    # Large, bright moving scene object above the caption geometry.
    cv2.rectangle(
        img, (20 + moving * 2, 2), (95 + moving * 2, 44),
        (250, 250, 250), -1,
    )
    cv2.line(img, (0, (moving * 11) % 48),
             (339, (moving * 11 + 25) % 48),
             (255, 255, 255), 3)
    if caption:
        cv2.putText(
            img, "TEST CAPTION", (40, 101), cv2.FONT_HERSHEY_SIMPLEX,
            0.85, (254, 254, 254), 2, lineType=cv2.LINE_AA,
        )
    return Frame(pts, img)


def metadata():
    # Normalized *independently derived OCR box* of subtitle pixel area.
    # Includes the full known TEST CAPTION without covering moving scene object.
    return {"detections": [{
        "cx": 0.50, "cy": 0.67, "width": 0.80,
        "height": 0.29, "score": 0.98, "text": "TEST CAPTION",
    }]}


class InkBoundaryExperimentTests(unittest.TestCase):
    def test_geometry_anchor_tracks_real_white_text_and_rejects_moving_blank(self):
        ref = rendered(2.25, caption=True, moving=4)
        nearby = [
            rendered(2.125, caption=True, moving=3),
            rendered(2.375, caption=True, moving=5),
        ]
        anchor = build_anchor(ref, nearby, metadata())
        self.assertIsNotNone(anchor)
        self.assertGreater(anchor.stable_white_pixels.sum(), 70)
        self.assertGreater(ink_support(anchor, rendered(2.5, caption=True, moving=9)), 0.8)
        self.assertLess(ink_support(anchor, rendered(2.5, caption=False, moving=9)), 0.3)

    def test_known_appearance_and_disappearance_are_not_backdated(self):
        frames = [
            rendered(i * 0.125, caption=8 <= i <= 23, moving=i % 20)
            for i in range(34)
        ]
        anchor = build_anchor(frames[16], [frames[15], frames[17]], metadata())
        self.assertIsNotNone(anchor)
        points = infer_presence(frames, anchor)
        self.assertTrue(points)
        self.assertAlmostEqual(points[0], 1.0)
        self.assertAlmostEqual(points[-1], 23 * 0.125)
        self.assertFalse([p for p in points if p < 1.0 or p >= 3.0])
        report = _evaluation(
            frames, [rendered(i*0.125, caption=False, moving=i%20)
                     for i in range(34)],
            anchor, (1.0, 3.0),
        )
        self.assertEqual(report["false_non_cue_frames"], 0)
        self.assertEqual(report["false_on_blank_control"], 0)
        self.assertEqual(report["start_error_seconds"], 0)
        self.assertEqual(report["end_error_seconds"], 0)
        self.assertEqual(report["recovered_true_frames"], 16)

    def test_two_threshold_tracking_extends_contiguous_weak_edges_without_spike(self):
        frames = [rendered(i * 0.125, caption=False) for i in range(17)]
        scores = [
            0.0, 0.0, 0.0, 0.0,
            0.60, 0.90, 0.92, 0.89,
            0.40, 0.68, 0.57,
            0.02, 0.0, 0.0, 0.99, 0.0, 0.0,
        ]
        fake_anchor = object()
        with mock.patch.object(study, "ink_support", side_effect=lambda a,f: scores[round(f.pts * 8)]):
            selected = hysteresis_presence(frames, fake_anchor)
        # One weak gap is bridged only between two supported states.
        self.assertEqual(selected, [i * 0.125 for i in range(4, 11)])
        # The isolated bright flash at frame 14 has no second strong seed.
        self.assertNotIn(14 * 0.125, selected)

    def test_hysteresis_never_creates_text_from_low_support_only(self):
        frames = [rendered(i * 0.125, caption=False) for i in range(20)]
        with mock.patch.object(study, "ink_support", return_value=0.63):
            self.assertEqual(hysteresis_presence(frames, object()), [])
        with mock.patch.object(study, "ink_support", side_effect=[
            0.91, 0.0, 0.0, 0.92, *([0.0] * 16),
        ]):
            self.assertEqual(hysteresis_presence(frames, object()), [])

    def test_hysteresis_rejects_unbounded_bridge_and_invalid_confidence(self):
        frames = [rendered(i * 0.125, caption=False) for i in range(5)]
        for lo, hi, bridge in (
            (0.2, 0.8, 1), (0.81, 0.8, 1),
            (0.6, 1.01, 1), (0.55, 0.76, 2),
        ):
            with self.subTest(params=(lo,hi,bridge)), self.assertRaises(ValueError):
                hysteresis_presence(
                    frames, object(), low_threshold=lo,
                    high_threshold=hi, bridge_gap_frames=bridge,
                )

    def test_unknown_ocr_geometry_cannot_produce_an_anchor(self):
        frame = rendered(1.0, caption=True)
        nearby = [rendered(0.875, caption=True),
                  rendered(1.125, caption=True)]
        for invalid in (
            None, {}, {"detections":[]}, {"detections":[{}]},
            {"detections":[{"cx": 0.5, "cy": 0.5, "width": 0.7,
                             "height": 0.2, "score": 0.01}]},
            {"detections":[{"cx": float("nan"), "cy": 0.5,
                             "width": 0.7, "height": 0.2, "score": 0.9}]},
        ):
            with self.subTest(value=invalid):
                self.assertIsNone(build_anchor(frame, nearby, invalid))

    def test_no_false_anchor_when_text_does_not_exist(self):
        no = rendered(2.0, caption=False, moving=4)
        nearby = [rendered(1.875, caption=False, moving=3),
                  rendered(2.125, caption=False, moving=5)]
        self.assertIsNone(build_anchor(no, nearby, metadata()))

    def test_short_or_out_of_window_companions_do_not_authenticate_anchor(self):
        ref = rendered(2.0, caption=True)
        far = [rendered(0.125, caption=True), rendered(4.0, caption=True)]
        self.assertIsNone(build_anchor(ref, far, metadata()))
        self.assertIsNone(build_anchor(ref, far[:1], metadata()))

    def test_suspicious_box_dimensions_and_bad_matching_threshold_rejected(self):
        image = rendered(2.0, caption=True)
        for box in (
            {"cx":0.5, "cy":0.5, "width":1.0, "height":0.8,"score":0.9},
            {"cx":-0.3, "cy":0.5, "width":0.3,"height":0.2,"score":0.9},
            {"cx":0.5, "cy":0.5, "width":0,"height":0.2,"score":0.9},
        ):
            self.assertIsNone(_ocr_box({"detections":[box]}, image.image.shape))
        anchor=build_anchor(
            image, [rendered(1.875,caption=True),
                    rendered(2.125,caption=True)], metadata(),
        )
        for threshold in (0.0, 0.49, 1.01):
            with self.assertRaises(ValueError):
                infer_presence([image], anchor, threshold=threshold)


if __name__ == "__main__":
    unittest.main()
