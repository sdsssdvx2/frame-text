"""Offline regression for experimental OCR-rectangle palette timing only.

No network, OCR model, browser or user media. Production is not imported
or altered by the palette candidate.
"""
from __future__ import annotations

from pathlib import Path
import sys
import unittest
from unittest import mock

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import experiment_palette_ink_boundary as palette
from hardsub_extractor.models import Frame


def synthetic(pts, *, caption=False, color=(255, 255, 255), moving=0):
    image = np.zeros((130, 340, 3), dtype=np.uint8)
    image[:] = (85, 75, 62)
    cv2.rectangle(image, ((moving*5)%310, 5),
                  (((moving*5)%310)+20, 45), (255, 255, 255), -1)
    if caption:
        cv2.putText(image, "TEST", (110, 100), cv2.FONT_HERSHEY_SIMPLEX,
                    1.3, (0, 0, 0), 8, cv2.LINE_AA)
        cv2.putText(image, "TEST", (110, 100), cv2.FONT_HERSHEY_SIMPLEX,
                    1.3, color, 3, cv2.LINE_AA)
    return Frame(pts, image)


def independent_metadata():
    # Approximate OCR-detected glyph rectangle, *not* pixel template.
    return {"detections": [{
        "cx": 0.5, "cy": 0.72, "width": 0.52,
        "height": 0.30, "score": 0.95,
    }]}


class PaletteBoundaryResearchTests(unittest.TestCase):
    def test_both_white_and_yellow_with_black_outline_recover_boundaries(self):
        for color, expected_kind in (
            ((255, 255, 255), "white"), ((0, 230, 255), "colored"),
        ):
            with self.subTest(color=color):
                frames = [
                    synthetic(i * 0.125, caption=(10 <= i <= 30),
                              color=color, moving=i)
                    for i in range(45)
                ]
                anchor = palette.build_palette_anchor(
                    frames[20], [frames[19], frames[21]],
                    independent_metadata(),
                )
                self.assertIsNotNone(anchor)
                self.assertEqual(anchor.palette, expected_kind)
                recovered = palette.presence(frames, anchor)
                self.assertEqual(recovered[0], 10 * 0.125)
                self.assertEqual(recovered[-1], 30 * 0.125)
                self.assertEqual(len(recovered), 21)
                blank = [
                    synthetic(f.pts, caption=False, moving=i)
                    for i, f in enumerate(frames)
                ]
                self.assertEqual(palette.presence(blank, anchor), [])

    def test_correct_timing_cannot_hide_missing_second_line(self):
        from hardsub_extractor.models import OCRObservation
        good = OCRObservation("First line\nSecond line", 0.99, 2.0)
        partial = OCRObservation("First line", 0.99, 2.0)
        merged = OCRObservation("First line Second line", 0.99, 2.0)
        self.assertEqual(
            palette.text_fidelity(good, "First line Second line",
                                  "white_two_line")["line_count_match"], True,
        )
        for observation in (partial, merged):
            metrics = palette.text_fidelity(
                observation, "First line Second line", "white_two_line",
            )
            self.assertFalse(
                metrics["exact_key"] and metrics["line_count_match"]
            )
        self.assertGreater(
            palette.text_fidelity(
                partial, "First line Second line", "white_two_line",
            )["cer"], 0,
        )

    def test_missing_ocr_boxes_fail_closed(self):
        ref = synthetic(1.25, caption=True)
        neighbors = [synthetic(1.125, caption=True),
                     synthetic(1.375, caption=True)]
        for metadata in (None, {}, {"detections": []}):
            with self.subTest(metadata=metadata):
                self.assertIsNone(
                    palette.build_palette_anchor(ref, neighbors, metadata),
                )

    def test_ocr_rectangles_do_not_include_wide_union_box_gap(self):
        meta = {"detections": [
            {"cx": 0.5, "cy": 0.45, "width": 0.5,
             "height": 0.1, "score": 0.95},
            {"cx": 0.5, "cy": 0.8, "width": 0.5,
             "height": 0.1, "score": 0.95},
        ]}
        mask = palette._rectangle_geometry(
            (130, 340, 3), meta, (0, 0, 340, 130),
        )
        self.assertFalse(mask[80, 170])
        self.assertTrue(mask[58, 170])
        self.assertTrue(mask[105, 170])

    def test_no_single_strong_flash_accepted_and_invalid_thresholds_rejected(self):
        frames = [synthetic(i * 0.125) for i in range(18)]
        scores = [0.0] * len(frames)
        scores[7] = 0.99
        scores[6] = 0.65
        scores[8] = 0.65
        with mock.patch.object(
            palette, "palette_score",
            side_effect=lambda _anchor, f: scores[round(f.pts * 8)],
        ):
            self.assertEqual(palette.presence(frames, object()), [])
        for lo, hi in ((0.2, 0.8), (0.81, 0.8), (0.7, 1.5)):
            with self.subTest(thresholds=(lo, hi)), self.assertRaises(ValueError):
                palette.presence([], object(), low=lo, high=hi)


if __name__ == "__main__":
    unittest.main()
