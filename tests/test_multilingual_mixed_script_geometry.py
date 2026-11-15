"""Mixed-script OCR subtitle fidelity and geometry-only overlay protection."""
import unittest

import numpy as np

from hardsub_extractor.ocr.rapidocr_adapter import (
    _join_visual_line, _subtitle_box_lines,
)

SHAPE = (400, 1800, 3)


def rect(x0, y0, x1, y1):
    return np.array(
        [[x0, y0], [x1, y0], [x1, y1], [x0, y1]], dtype=float,
    )


class MixedScriptLineTests(unittest.TestCase):
    def check_line(self, language, texts, boxes, expected_indices, expected_text):
        lines = _subtitle_box_lines(texts, boxes, SHAPE, language=language)
        self.assertEqual(lines, [expected_indices])
        self.assertEqual(
            _join_visual_line(texts, boxes, lines[0], language=language),
            expected_text,
        )

    def test_japanese_embedded_latin_and_punctuation(self):
        self.check_line(
            "ja", ["君は", "OK", "?"], [
                rect(600, 280, 775, 345),
                rect(787, 281, 880, 345),
                rect(885, 281, 905, 345),
            ], [0, 1, 2], "君はOK?",
        )

    def test_chinese_numeric_fragment_between_han_words(self):
        self.check_line(
            "zh", ["明天", "7", "点见"], [
                rect(600, 280, 750, 345),
                rect(759, 281, 792, 345),
                rect(806, 282, 954, 345),
            ], [0, 1, 2], "明天7点见",
        )

    def test_japanese_english_phrase_preserves_internal_spaces(self):
        self.check_line(
            "ja", ["君", "I", "love", "you", "だ"], [
                rect(600, 280, 665, 345),
                rect(675, 281, 695, 345),
                rect(710, 282, 800, 345),
                rect(815, 281, 880, 345),
                rect(898, 281, 950, 345),
            ], [0, 1, 2, 3, 4], "君I love youだ",
        )

    def test_english_numbers_and_trailing_punctuation(self):
        self.check_line(
            "en", ["Hello", "!", "2026"], [
                rect(600, 280, 800, 345),
                rect(804, 280, 823, 345),
                rect(835, 280, 900, 345),
            ], [0, 1, 2], "Hello! 2026",
        )

    def test_nearly_coincident_number_box_is_ocr_noise_not_caption(self):
        # A detector can propose two almost identical boxes for one visual
        # region; script-neutral text must not duplicate the anchor pixels.
        for language, anchor in (("en", "Hello"), ("ja", "こんにちは"), ("zh", "你好")):
            with self.subTest(language=language):
                self.check_line(
                    language, [anchor, "123"], [
                        rect(620, 280, 950, 345),
                        rect(625, 281, 944, 345),
                    ], [0], anchor,
                )

    def test_unanchored_latin_or_number_overlay_is_never_caption(self):
        for language in ("ja", "zh"):
            with self.subTest(language=language):
                self.assertEqual(
                    _subtitle_box_lines(
                        ["Credits", "2026", "!"], [
                            rect(600, 280, 780, 345),
                            rect(790, 280, 900, 345),
                            rect(905, 280, 920, 345),
                        ], SHAPE, language=language,
                    ), [],
                )

    def test_distant_counter_is_not_pulled_into_anchored_caption(self):
        self.check_line(
            "ja", ["君は", "2026"], [
                rect(600, 280, 775, 345),
                rect(1050, 281, 1190, 345),
            ], [0], "君は",
        )

    def test_two_lines_have_independent_anchors_and_attachments(self):
        lines = _subtitle_box_lines(
            ["今日は", "OK", "明日", "2026", "字幕監修"], [
                rect(600, 155, 830, 220),
                rect(839, 155, 910, 220),
                rect(600, 280, 770, 345),
                rect(780, 280, 930, 345),
                rect(1510, 280, 1710, 345),
            ], SHAPE, language="ja",
        )
        self.assertEqual(lines, [[0, 1], [2, 3]])

    def test_spurious_bracket_near_english_dialogue_is_not_recovered(self):
        # Real MP4 QA: a false positive opening bracket appeared just before
        # "in five" and disrupted the high-confidence leading-I repair.
        texts = ["will be", "there", "[", "in five", "minutes."]
        boxes = [
            rect(600, 280, 780, 345),
            rect(790, 280, 900, 345),
            rect(905, 280, 925, 345),
            rect(930, 280, 1060, 345),
            rect(1070, 280, 1240, 345),
        ]
        self.check_line(
            "en", texts, boxes, [0, 1, 3, 4],
            "will be there in five minutes.",
        )

    def test_only_sentence_punctuation_not_bracket_noise_is_attached(self):
        self.check_line(
            "ja", ["君は", "OK", "?", "[", "]"], [
                rect(600, 280, 775, 345),
                rect(787, 280, 880, 345),
                rect(885, 280, 905, 345),
                rect(910, 280, 928, 345),
                rect(930, 280, 950, 345),
            ], [0, 1, 2], "君はOK?",
        )

    def test_korean_legacy_filter_does_not_acquire_latin_fragments(self):
        self.assertEqual(
            _subtitle_box_lines(
                ["안녕", "OK", "!"], [
                    rect(600, 280, 775, 345),
                    rect(787, 280, 880, 345),
                    rect(885, 280, 905, 345),
                ], SHAPE,
            ), [[0]],
        )


if __name__ == "__main__":
    unittest.main()
