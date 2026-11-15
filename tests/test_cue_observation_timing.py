import unittest

import numpy as np

from hardsub_extractor.cue_builder import recognize_segments
from hardsub_extractor.models import Frame, OCRObservation, VisualSegment
from hardsub_extractor.ocr.base import OCREngine


class SequenceOCR(OCREngine):
    def __init__(self, outputs):
        self.outputs = list(outputs)
        self.index = 0

    def recognize(self, frame):
        text, confidence = self.outputs[self.index]
        self.index += 1
        return OCRObservation(text, confidence, frame.pts)


def frame(pts):
    return Frame(pts, np.zeros((30, 120, 3), dtype=np.uint8))


class CueObservationTimingTests(unittest.TestCase):
    def test_late_tail_hit_does_not_backdate_to_long_segment_start(self):
        # Reproduces the live EpSite QA shape that previously emitted "2장"
        # from 16.600 even though the first supporting OCR sample was 19.811.
        segments = [
            VisualSegment(
                16.600,
                19.853,
                [frame(16.600), frame(16.808), frame(19.811)],
            ),
            VisualSegment(19.853, 19.978, [frame(19.853)]),
            VisualSegment(19.978, 20.020, [frame(19.978)]),
        ]
        ocr = SequenceOCR([
            ("", 0.0),
            ("", 0.0),
            ("2장", 0.9974),
            ("2장", 0.9998),
            ("", 0.0),
        ])

        cues = recognize_segments(segments, ocr)

        self.assertEqual(len(cues), 1)
        self.assertEqual(cues[0].text, "2장")
        self.assertAlmostEqual(cues[0].start, 19.811, places=3)
        self.assertAlmostEqual(cues[0].end, 19.978, places=3)

    def test_two_late_samples_preserve_new_tail_subtitle(self):
        # Generic regression for a visually quiet long segment whose subtitle
        # changes near the tail. The detector's rolling three-sample policy
        # should provide [old, new, new]; cue recognition must then emit the
        # new text from its first supporting observation rather than letting a
        # stale early reading own the whole visual segment.
        segments = [
            VisualSegment(
                12.796,
                15.173,
                [frame(12.796), frame(14.965), frame(15.132)],
            ),
            VisualSegment(15.173, 15.600, [frame(15.173)]),
        ]
        ocr = SequenceOCR([
            ("이전 문장", 0.997),
            ("새 문장", 0.970),
            ("새 문장", 0.955),
            ("", 0.0),
        ])

        cues = recognize_segments(segments, ocr)

        self.assertEqual(len(cues), 1)
        self.assertEqual(cues[0].text, "새 문장")
        self.assertAlmostEqual(cues[0].start, 14.965, places=3)
        self.assertAlmostEqual(cues[0].end, 15.173, places=3)


if __name__ == "__main__":
    unittest.main()
