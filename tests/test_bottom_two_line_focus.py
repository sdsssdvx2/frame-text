"""Source-only proof of optional visual OCR focus and unchanged defaults."""
from __future__ import annotations

import unittest
from unittest import mock
import numpy as np

from hardsub_extractor.models import Frame, OCRObservation
from hardsub_extractor.ocr.line_focus import BottomTwoLineEngine


def sample():
    return Frame(1.25, np.zeros((16, 24, 3), dtype=np.uint8))


class Base:
    total_calls = 0
    retry_calls = 0

    def __init__(self, text, meta=None):
        self.text = text
        self.meta = meta
        self.calls = 0

    def recognize(self, frame):
        self.calls += 1
        self.total_calls += 1
        return OCRObservation(self.text, 0.91, frame.pts, self.meta)


class OptionalBottomTwoLinesTests(unittest.TestCase):
    def test_single_and_double_line_pass_through_without_metadata_mutation(self):
        for text in ("", "안녕하세요", "東京\nありがとう"):
            with self.subTest(text=text):
                base = Base(text, {"detections": [{"line": 0, "text": text}]})
                engine = BottomTwoLineEngine(base)
                result = engine.recognize(sample())
                self.assertEqual(result.text, text)
                self.assertEqual(engine.dropped_rows, 0)
                self.assertEqual(engine.modified_observations, 0)
                self.assertEqual(base.calls, 1)

    def test_natural_caption_credits_above_two_line_dialogue(self):
        base = Base(
            "UPPER SCENE CREDIT\nFirst dialogue row.\nSecond dialogue row?",
            {"detections": [
                {"line": 0, "text": "UPPER SCENE CREDIT", "score": 0.91},
                {"line": 1, "text": "First dialogue row.", "score": 0.97},
                {"line": 2, "text": "Second dialogue row?", "score": 0.96},
            ], "other_count": 4},
        )
        engine = BottomTwoLineEngine(base)
        result = engine.recognize(sample())
        self.assertEqual(
            result.text, "First dialogue row.\nSecond dialogue row?",
        )
        self.assertEqual([d["line"] for d in result.metadata["detections"]], [0, 1])
        self.assertEqual(
            [d["text"] for d in result.metadata["detections"]],
            ["First dialogue row.", "Second dialogue row?"],
        )
        self.assertEqual((engine.modified_observations,engine.dropped_rows), (1,1))
        self.assertEqual((engine.total_calls,engine.retry_calls), (1,0))
        self.assertEqual(base.calls, 1)
        self.assertEqual(result.pts, 1.25)
        self.assertEqual(result.confidence, 0.91)
        self.assertEqual(result.metadata["other_count"], 4)

    def test_discarded_high_score_credit_cannot_boost_dialogue(self):
        evidence = {"detections": [
            {"line": 0, "text": "SCENE CREDIT", "score": 0.99},
            {"line": 1, "text": "첫째", "score": 0.30},
            {"line": 2, "text": "둘째", "score": 0.32},
        ]}
        base = Base("SCENE CREDIT\n첫째\n둘째", evidence)
        result = BottomTwoLineEngine(base).recognize(sample())
        self.assertEqual(result.text, "첫째\n둘째")
        self.assertAlmostEqual(result.confidence, 0.31)
        self.assertEqual([d["line"] for d in result.metadata["detections"]], [0, 1])
        self.assertTrue(result.metadata["experimental_bottom_two_confidence_bounded"])
        self.assertEqual(base.meta, evidence)
        self.assertEqual(len(base.meta["detections"]), 3)

    def test_retained_box_scores_cannot_inflate_adaptive_vote(self):
        base = Base("CREDIT\nA\nB", {"detections": [
            {"line": 0, "score": 0.10},
            {"line": 1, "score": 0.98},
            {"line": 2, "score": 0.97},
        ]})
        result = BottomTwoLineEngine(base).recognize(sample())
        self.assertEqual(result.confidence, 0.91)
        self.assertNotIn("experimental_bottom_two_confidence_bounded",
                         result.metadata)

    def test_missing_or_invalid_retained_score_does_not_guess_confidence(self):
        cases = (
            [{"line": 0, "score": 0.99}, {"line": 1, "score": 0.20}],
            [{"line": 0, "score": 0.99}, {"line": 1, "score": 0.20},
             {"line": 2, "score": float("nan")}],
            [{"line": 0, "score": 0.99}, {"line": 1, "score": 0.20},
             {"line": 2, "score": "low"}],
            [],
        )
        for detections in cases:
            with self.subTest(detections=detections):
                result = BottomTwoLineEngine(Base(
                    "CREDIT\nA\nB", {"detections": detections},
                )).recognize(sample())
                self.assertEqual(result.confidence, 0.91)
                self.assertNotIn("experimental_bottom_two_confidence_bounded",
                                 result.metadata)

    def test_three_real_dialogue_lines_can_be_lost_documented_risk(self):
        base = Base("실제 첫째줄\n실제 둘째줄\n실제 셋째줄")
        actual = BottomTwoLineEngine(base).recognize(sample())
        self.assertEqual(actual.text, "실제 둘째줄\n실제 셋째줄")
        self.assertNotIn("실제 첫째줄", actual.text)

    def test_no_new_ocr_calls_or_alternate_model(self):
        base = Base("a\nb\nc\nd")
        engine = BottomTwoLineEngine(base)
        outputs = [engine.recognize(sample()) for _ in range(3)]
        self.assertEqual(base.calls, 3)
        self.assertEqual(engine.dropped_rows, 6)
        self.assertEqual([x.text for x in outputs], ["c\nd"]*3)

    def test_false_single_line_background_is_not_auto_removed(self):
        base = Base("SCENE TITLE")
        result = BottomTwoLineEngine(base).recognize(sample())
        self.assertEqual(result.text, "SCENE TITLE")
        # This option alone cannot certify no-caption false positives.
        self.assertEqual(base.calls, 1)

    def test_adaptive_focus_votes_on_retained_dialogue_not_strong_scene_credit(self):
        from hardsub_extractor.ocr.adaptive import AdaptiveOCREngine, AdaptiveOCRConfig

        readings = [
            ("PRODUCTION CREDITS\n잘못 읽음\n끝", .96, {"detections": [
                {"line": 0, "text": "PRODUCTION CREDITS", "score": .99},
                {"line": 1, "text": "잘못 읽음", "score": .40},
                {"line": 2, "text": "끝", "score": .50},
            ]}),
            ("PRODUCTION CREDITS\n정확한 첫째\n정확한 둘째", .95, {"detections": [
                {"line": 0, "text": "PRODUCTION CREDITS", "score": .99},
                {"line": 1, "text": "정확한 첫째", "score": .96},
                {"line": 2, "text": "정확한 둘째", "score": .94},
            ]}),
        ]

        class Sequence:
            def __init__(self):
                self.calls = 0

            def recognize(self, frame):
                text, confidence, metadata = readings[min(self.calls, 1)]
                self.calls += 1
                return OCRObservation(text, confidence, frame.pts, metadata)

        config = AdaptiveOCRConfig(upscale=1.5, enable_clahe_retry=False)
        original = Sequence()
        old_vote = AdaptiveOCREngine(original, config).recognize(sample())
        old_filtered = BottomTwoLineEngine(Base(old_vote.text)).recognize(sample())
        self.assertEqual((old_filtered.text, original.calls), ("잘못 읽음\n끝", 2))

        raw = Sequence()
        focused = BottomTwoLineEngine(AdaptiveOCREngine(raw, config))
        result = focused.recognize(sample())
        self.assertEqual(result.text, "정확한 첫째\n정확한 둘째")
        self.assertEqual((raw.calls, focused.total_calls, focused.retry_calls), (2, 2, 1))
        self.assertEqual((focused.modified_observations, focused.dropped_rows), (2, 2))
        self.assertEqual([box["line"] for box in result.metadata["detections"]], [0, 1])
        self.assertGreater(result.confidence, .9)

    def test_adaptive_focus_does_not_retry_discarded_weak_credit(self):
        from hardsub_extractor.ocr.adaptive import AdaptiveOCREngine
        metadata = {"detections": [
            {"line": 0, "text": "CREDIT", "score": .10},
            {"line": 1, "text": "첫째 대사", "score": .98},
            {"line": 2, "text": "둘째 대사", "score": .97},
        ]}
        old = AdaptiveOCREngine(Base("CREDIT\n첫째 대사\n둘째 대사", metadata))
        old.recognize(sample())
        self.assertEqual(old.total_calls, 3)

        base = Base("CREDIT\n첫째 대사\n둘째 대사", metadata)
        focused = BottomTwoLineEngine(AdaptiveOCREngine(base))
        result = focused.recognize(sample())
        self.assertEqual(result.text, "첫째 대사\n둘째 대사")
        self.assertEqual((base.calls, focused.total_calls, focused.retry_calls), (1, 1, 0))
        self.assertEqual(len(metadata["detections"]), 3)

    def test_pipeline_end_to_end_explicit_opt_in_keeps_default(self):
        from pathlib import Path
        import tempfile
        from hardsub_extractor.pipeline import extract
        from hardsub_extractor.subtitle_formats import parse_subtitle_bytes

        frames = lambda: iter([
            Frame(float(i) * 0.25, np.zeros((18, 24, 3), dtype=np.uint8))
            for i in range(4)
        ])
        with tempfile.TemporaryDirectory() as td, mock.patch(
            "hardsub_extractor.pipeline.FFmpegPTSFrameSource",
            side_effect=lambda *args, **kwargs: frames(),
        ) as source:
            original = Base("제작사 이름\n두 줄 첫째\n두 줄 둘째")
            plain = extract("only-test.mkv", original, Path(td)/"default",
                            streaming=True)
            focused = Base("제작사 이름\n두 줄 첫째\n두 줄 둘째")
            adjusted = extract(
                "only-test.mkv", focused, Path(td)/"focused",
                streaming=True, focus_bottom_two_lines=True,
            )
            _, normal = parse_subtitle_bytes(
                plain.srt_path.read_bytes(), source_url="plain.srt",
            )
            _, filtered = parse_subtitle_bytes(
                adjusted.srt_path.read_bytes(), source_url="focused.srt",
            )
            self.assertEqual(plain.cue_count, 1)
            self.assertEqual(adjusted.cue_count, 1)
            self.assertEqual(normal[0].text,
                             "제작사 이름\n두 줄 첫째\n두 줄 둘째")
            self.assertEqual(filtered[0].text, "두 줄 첫째\n두 줄 둘째")
            self.assertEqual(source.call_count, 2)
            self.assertEqual(original.calls, focused.calls)
            _, normal_ass = parse_subtitle_bytes(
                plain.ass_path.read_bytes(), source_url="normal.ass",
            )
            _, filtered_ass = parse_subtitle_bytes(
                adjusted.ass_path.read_bytes(), source_url="filtered.ass",
            )
            self.assertEqual(normal[0].text, normal_ass[0].text)
            self.assertEqual(filtered[0].text, filtered_ass[0].text)

    def test_pipeline_rejects_nonboolean_focus_before_decode(self):
        from hardsub_extractor.pipeline import extract
        with mock.patch(
            "hardsub_extractor.pipeline.FFmpegPTSFrameSource",
            side_effect=AssertionError("decoded invalid opt-in"),
        ) as decoder:
            with self.assertRaises(ValueError):
                extract(
                    "test.mp4", Base("x"), "output",
                    focus_bottom_two_lines="yes",
                )
            decoder.assert_not_called()

    def test_reject_non_engine_and_bad_result(self):
        with self.assertRaises(TypeError):
            BottomTwoLineEngine(object())
        class Invalid:
            def recognize(self, frame):
                return "not an observation"
        with self.assertRaises(TypeError):
            BottomTwoLineEngine(Invalid()).recognize(sample())


if __name__ == "__main__":
    unittest.main()
