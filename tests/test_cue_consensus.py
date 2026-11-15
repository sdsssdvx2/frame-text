import unittest
import numpy as np

from hardsub_extractor.cue_builder import CueRecognitionConfig, recognize_segments
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


def segment(start=1.0, end=2.0, count=3):
    samples = [
        Frame(
            start + (end-start) * i / max(1, count-1),
            np.zeros((30,120,3), dtype=np.uint8),
        )
        for i in range(count)
    ]
    return VisualSegment(start, end, samples)


class CueConsensusTests(unittest.TestCase):
    def test_isolated_false_hit_is_rejected(self):
        ocr = SequenceOCR([
            ("", .0),
            ("방송국", .96),
            ("", .0),
        ])
        cues = recognize_segments([segment()], ocr)
        self.assertEqual(cues, [])

    def test_two_similar_frames_are_accepted(self):
        ocr = SequenceOCR([
            ("괜찮아", .91),
            ("괜찮아", .94),
            ("", .0),
        ])
        cues = recognize_segments([segment()], ocr)
        self.assertEqual(len(cues), 1)
        self.assertEqual(cues[0].text, "괜찮아")

    def test_one_bad_spelling_plus_two_good_frames_votes_good(self):
        ocr = SequenceOCR([
            ("괜찮0ㅏ", .55),
            ("괜찮아", .91),
            ("괜찮아", .93),
        ])
        cues = recognize_segments([segment()], ocr)
        self.assertEqual(len(cues), 1)
        self.assertEqual(cues[0].text, "괜찮아")

    def test_single_sample_short_high_confidence_is_kept(self):
        ocr = SequenceOCR([("앗!", .98)])
        cues = recognize_segments(
            [segment(1.0, 1.20, count=1)],
            ocr,
        )
        self.assertEqual(len(cues), 1)
        self.assertEqual(cues[0].text, "앗!")

    def test_single_sample_long_segment_is_rejected(self):
        ocr = SequenceOCR([("간판", .99)])
        cues = recognize_segments(
            [segment(1.0, 3.0, count=1)],
            ocr,
        )
        self.assertEqual(cues, [])

    def test_single_sample_low_conf_short_is_rejected(self):
        ocr = SequenceOCR([("앗!", .70)])
        cues = recognize_segments(
            [segment(1.0, 1.20, count=1)],
            ocr,
        )
        self.assertEqual(cues, [])

    def test_busy_scene_fragments_vote_back_to_one_subtitle(self):
        segments = [
            segment(0.000, 0.125, 1),
            segment(0.125, 0.250, 1),
            segment(0.250, 0.375, 1),
            segment(0.375, 0.500, 1),
            segment(0.500, 0.625, 1),
            segment(0.625, 0.750, 1),
            segment(0.750, 0.875, 1),
            segment(0.875, 1.000, 1),
            segment(1.000, 1.125, 1),
        ]
        ocr = SequenceOCR([
            ("잔력 와있을까", .81),
            ("잔뜩\n라있을까", .90),
            ("라있을까", .99),
            ("잔뜩\n와있을까", .82),
            ("잔특 와있을까", .84),
            ("잔독와있을까", .91),
            ("잔득 와있을까", .81),
            ("잔똑와 있을까", .90),
            ("잔똑와있을까.", .89),
        ])
        cues = recognize_segments(segments, ocr)
        self.assertEqual(len(cues), 1)
        self.assertEqual(cues[0].text, "잔뜩\n와있을까")
        self.assertAlmostEqual(cues[0].start, 0.0, places=3)
        self.assertAlmostEqual(cues[0].end, 1.125, places=3)

    def test_transient_misread_does_not_split_stable_subtitle(self):
        segments = [
            segment(0.0, 0.60, 3),
            segment(0.60, 0.85, 1),
            segment(0.85, 0.975, 1),
            segment(0.975, 1.225, 1),
        ]
        ocr = SequenceOCR([
            ("매달\n기대하고\n있어", .91),
            ("매달\n기대하고\n있어", .92),
            ("매달\n기대하고\n있어", .91),
            ("미달기대하고\n있어", .91),
            ("기대하고\n매달\n있어", .95),
            ("매달기대하고\n있어", .91),
        ])
        cues = recognize_segments(segments, ocr)
        self.assertEqual(len(cues), 1)
        self.assertEqual(cues[0].text, "매달\n기대하고\n있어")
        self.assertAlmostEqual(cues[0].end, 1.225, places=3)

    def test_persistent_different_text_confirms_boundary(self):
        segments = [
            segment(0.0, 0.60, 3),
            segment(0.60, 0.725, 1),
            segment(0.725, 0.850, 1),
            segment(0.850, 0.975, 1),
            segment(0.975, 1.100, 1),
        ]
        ocr = SequenceOCR([
            ("첫 번째 대사", .96),
            ("첫 번째 대사", .97),
            ("첫 번째 대사", .96),
            ("에잇", .99),
            ("에잇", .99),
            ("에잇", .99),
            ("에잇", .99),
        ])
        cues = recognize_segments(segments, ocr)
        self.assertEqual([cue.text for cue in cues], ["첫 번째 대사", "에잇"])
        self.assertAlmostEqual(cues[0].end, .60, places=3)
        self.assertAlmostEqual(cues[1].start, .60, places=3)

    def test_persistent_similar_new_subtitle_is_not_overmerged(self):
        segments = [
            segment(0.0, 0.60, 3),
            segment(0.60, 0.80, 1),
            segment(0.80, 1.00, 1),
            segment(1.00, 1.20, 1),
        ]
        ocr = SequenceOCR([
            ("오늘은 정말 좋아", .96),
            ("오늘은 정말 좋아", .97),
            ("오늘은 정말 좋아", .96),
            ("오늘은 정말 싫어", .97),
            ("오늘은 정말 싫어", .97),
            ("오늘은 정말 싫어", .97),
        ])
        cues = recognize_segments(segments, ocr)
        self.assertEqual(
            [cue.text for cue in cues],
            ["오늘은 정말 좋아", "오늘은 정말 싫어"],
        )
        self.assertAlmostEqual(cues[0].end, .60, places=3)
        self.assertAlmostEqual(cues[1].start, .60, places=3)


if __name__=="__main__":
    unittest.main()
