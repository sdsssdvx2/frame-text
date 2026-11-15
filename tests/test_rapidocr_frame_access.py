from __future__ import annotations

import unittest
from types import SimpleNamespace

import numpy as np

from hardsub_extractor.ocr.rapidocr_adapter import RapidOCRAdapter


class _CountingBrowserFrame:
    """Model a browser JPEG frame: each .image access decodes the JPEG anew."""

    def __init__(self):
        self.pts = 12.5
        self._image = np.zeros((400, 1800, 3), dtype=np.uint8)
        self.image_reads = 0

    @property
    def image(self):
        self.image_reads += 1
        return self._image


def _box(x0, y0, x1, y1):
    return np.array(
        [[x0, y0], [x1, y0], [x1, y1], [x0, y1]], dtype=float
    )


class RapidOCRFrameAccessTests(unittest.TestCase):
    def test_one_image_access_per_browser_frame_with_multiple_detected_boxes(self):
        frame = _CountingBrowserFrame()
        observed_images = []
        mock_result = SimpleNamespace(
            txts=["첫 줄", "둘째 줄"],
            scores=[0.99, 0.98],
            boxes=[
                _box(700, 210, 1100, 265),
                _box(685, 292, 1110, 355),
            ],
        )

        def fake_engine(image, **kwargs):
            observed_images.append(image)
            self.assertTrue(kwargs["use_det"])
            self.assertTrue(kwargs["use_rec"])
            return mock_result

        # Avoid heavyweight model download or initialization: only exercise
        # the exact production recognize() path and its image access contract.
        adapter = RapidOCRAdapter.__new__(RapidOCRAdapter)
        adapter.engine = fake_engine
        adapter.text_score = 0.45
        adapter.korean_subtitle_mode = True

        observed = adapter.recognize(frame)

        self.assertEqual(observed.text, "첫 줄\n둘째 줄")
        self.assertAlmostEqual(observed.pts, 12.5)
        self.assertEqual(len(observed.metadata["detections"]), 2)
        self.assertEqual(len(observed_images), 1)
        self.assertIs(observed_images[0], frame._image)
        # BrowserJPEGFrame.image deliberately does not cache decoded ndarray;
        # redundant shape queries used to decode again for every OCR box.
        self.assertEqual(frame.image_reads, 1)


if __name__ == "__main__":
    unittest.main()
