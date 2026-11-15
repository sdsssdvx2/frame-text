"""Source-only subtitle-band signature study, no OCR/media/side-effects."""
from __future__ import annotations

import sys
from pathlib import Path
import unittest

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from compare_motion_subtitle_band import (  # noqa: E402
    PROFILES, focused_signature,
)
from hardsub_extractor.detector import (  # noqa: E402
    signature_change, subtitle_signature,
)


class SignatureFocusStudies(unittest.TestCase):
    def test_baseline_signature_remains_byte_identical(self):
        image = np.zeros((180, 320, 3), dtype=np.uint8)
        cv2.putText(
            image, "Subtitle sample", (20, 110),
            cv2.FONT_HERSHEY_SIMPLEX, 0.85, (255, 255, 255), 2,
        )
        baseline = subtitle_signature(image, 320)
        focused = focused_signature(
            image, band=None, original_signature=subtitle_signature,
            width=320,
        )
        np.testing.assert_array_equal(baseline, focused)

    def test_upper_moving_edges_suppressed_but_caption_edges_remain(self):
        base = np.zeros((180, 320, 3), dtype=np.uint8)
        upper_movement = base.copy()
        cv2.rectangle(
            upper_movement, (32, 4), (159, 34),
            (255, 255, 255), 2,
        )
        caption = base.copy()
        cv2.putText(
            caption, "Subtitle here", (12, 115),
            cv2.FONT_HERSHEY_SIMPLEX, 0.65,
            (255, 255, 255), 2,
        )
        old_base = subtitle_signature(base)
        old_movement = subtitle_signature(upper_movement)
        self.assertGreater(signature_change(old_base, old_movement), 0.0)
        for label, band in PROFILES.items():
            with self.subTest(profile=label):
                blank = focused_signature(
                    base, band=band,
                    original_signature=subtitle_signature,
                )
                changed = focused_signature(
                    caption, band=band,
                    original_signature=subtitle_signature,
                )
                self.assertGreater(signature_change(blank, changed), 0)
                if band is not None:
                    moved = focused_signature(
                        upper_movement, band=band,
                        original_signature=subtitle_signature,
                    )
                    self.assertEqual(signature_change(blank, moved), 0.0)

    def test_bad_crops_fail_closed_not_silent(self):
        image = np.zeros((100, 200, 3), dtype=np.uint8)
        for band in ((-0.1, 0.8), (0.7, 0.3), (0.8, 1.1), (0.5, 0.51)):
            with self.subTest(band=band):
                with self.assertRaises(ValueError):
                    focused_signature(
                        image, band=band,
                        original_signature=subtitle_signature,
                    )


if __name__ == "__main__":
    unittest.main()
