import unittest

import numpy as np

from hardsub_extractor.cue_builder import _SegmentEvidence
from hardsub_extractor.models import Frame, OCRObservation, VisualSegment
from hardsub_extractor.ocr.base import OCREngine
from hardsub_extractor.refined_recognition import (
    _stabilize_vote_from_recent_consensus,
    clean_visual_text,
    recognize_segments,
)


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


class RefinedRecognitionTests(unittest.TestCase):
    def test_blank_to_tail_never_backdates_before_first_support(self):
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

    def test_distinct_head_and_confirmed_tail_both_survive_one_visual_segment(self):
        segments = [
            VisualSegment(
                12.796,
                15.173,
                [frame(12.796), frame(15.090), frame(15.132)],
            ),
            VisualSegment(15.173, 15.600, [frame(15.173)]),
        ]
        ocr = SequenceOCR([
            ("이전 문장", 0.998),
            ("새 문장", 0.970),
            ("새 문장", 0.955),
            ("", 0.0),
        ])

        cues = recognize_segments(segments, ocr)

        transition = 12.796 + (15.090 - 12.796) * 0.75
        self.assertEqual([cue.text for cue in cues], ["이전 문장", "새 문장"])
        self.assertAlmostEqual(cues[0].start, 12.796, places=3)
        self.assertAlmostEqual(cues[0].end, transition, places=3)
        self.assertAlmostEqual(cues[1].start, transition, places=3)
        self.assertAlmostEqual(cues[1].end, 15.173, places=3)
        self.assertGreater(cues[1].duration(), 0.60)

    def test_low_confidence_head_does_not_create_false_split(self):
        segments = [
            VisualSegment(
                12.796,
                15.173,
                [frame(12.796), frame(15.090), frame(15.132)],
            ),
            VisualSegment(15.173, 15.600, [frame(15.173)]),
        ]
        ocr = SequenceOCR([
            ("잡음", 0.70),
            ("새 문장", 0.970),
            ("새 문장", 0.955),
            ("", 0.0),
        ])

        cues = recognize_segments(segments, ocr)

        self.assertEqual(len(cues), 1)
        self.assertEqual(cues[0].text, "새 문장")
        self.assertAlmostEqual(cues[0].start, 15.090, places=3)

    def test_repeated_stronger_reading_repairs_one_long_near_miss(self):
        # Two independent detector segments agree on one reading. A later long
        # segment votes for a one-syllable near-miss, but one of its own raw OCR
        # samples is materially closer to the repeated reading. The production
        # rule must use those generic signals rather than a phrase dictionary.
        segments = [
            VisualSegment(0.000, 0.125, [frame(0.000)]),
            VisualSegment(0.125, 0.250, [frame(0.125)]),
            VisualSegment(
                0.250,
                1.300,
                [frame(0.250), frame(0.900), frame(1.250)],
            ),
        ]
        ocr = SequenceOCR([
            ("표현은 느낌으로다가", 0.96),
            ("표현은 느낌으로다가", 0.92),
            ("표현은 느낌으로다게", 0.89),
            ("표현은 스낌으로다가", 0.80),
            ("표현은 느낌으로다게", 0.84),
        ])

        cues = recognize_segments(segments, ocr)

        self.assertEqual(len(cues), 1)
        self.assertEqual(cues[0].text, "표현은 느낌으로다가")

    def test_repeated_reading_does_not_override_without_current_sample_support(self):
        prior = [
            _SegmentEvidence(
                VisualSegment(0.000, 0.125, []),
                "표현은 느낌으로다가",
                0.96,
                1,
                True,
            ),
            _SegmentEvidence(
                VisualSegment(0.125, 0.250, []),
                "표현은 느낌으로다가",
                0.92,
                1,
                False,
            ),
        ]
        segment = VisualSegment(0.250, 1.300, [])
        observations = [
            OCRObservation("표현은 느낌으로다게", 0.89, 0.250),
            OCRObservation("표현은 느낌으로다게", 0.86, 0.900),
            OCRObservation("표현은 느낌으로다게", 0.84, 1.250),
        ]
        voted = OCRObservation("표현은 느낌으로다게", 0.86, 0.250)

        stabilized = _stabilize_vote_from_recent_consensus(
            voted,
            observations,
            prior,
            segment,
        )

        self.assertEqual(stabilized.text, voted.text)
        self.assertFalse((stabilized.metadata or {}).get("temporal_consensus_repair"))

    def test_decorative_circle_like_tails_are_cleaned_narrowly(self):
        self.assertEqual(clean_visual_text("소원수리함ㅇㅇ"), "소원수리함")
        self.assertEqual(clean_visual_text("소원수리함ㅇo"), "소원수리함")
        self.assertEqual(clean_visual_text("성적으로!?이"), "성적으로!?")

        # Normal punctuation and a short, potentially legitimate continuation
        # must not be rewritten by the decorative-tail heuristic.
        self.assertEqual(clean_visual_text("소원 수리함?"), "소원 수리함?")
        self.assertEqual(clean_visual_text("뭐!?야"), "뭐!?야")

    def test_bar_between_hangul_words_becomes_a_space(self):
        # Real PP-OCRv5 Korean output (OpenVINO backend, synthetic hard-sub video).
        self.assertEqual(clean_visual_text("약사의|혼잣말은 재미있는 애니입니다."), "약사의 혼잣말은 재미있는 애니입니다.")
        self.assertEqual(clean_visual_text("오늘 | 날씨"), "오늘 날씨")
        # Bars next to latin text or digits are left alone.
        self.assertEqual(clean_visual_text("A|B 테스트"), "A|B 테스트")
        self.assertEqual(clean_visual_text("1|2화"), "1|2화")


if __name__ == "__main__":
    unittest.main()
