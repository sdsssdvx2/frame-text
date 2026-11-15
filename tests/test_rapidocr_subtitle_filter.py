import unittest

import numpy as np

from hardsub_extractor.ocr.rapidocr_adapter import (
    _join_visual_line,
    _subtitle_box_indices,
    _subtitle_box_lines,
)


def box(x0, y0, x1, y1):
    return np.asarray([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], dtype=float)


class KoreanSubtitleBoxFilterTests(unittest.TestCase):
    SHAPE = (400, 1800, 3)

    def test_keeps_central_lower_korean_and_drops_right_credits(self):
        texts = ["Kay Yu", "mockingbird", "소원 수리함?"]
        boxes = [
            box(1420, 80, 1680, 130),
            box(1410, 250, 1710, 300),
            box(690, 285, 1110, 355),
        ]
        self.assertEqual(
            _subtitle_box_indices(texts, boxes, self.SHAPE),
            [2],
        )

    def test_requires_precomposed_hangul_in_korean_mode(self):
        texts = ["7=X-93V70Pa-1-", "/V", "Z3"]
        boxes = [
            box(700, 250, 1100, 310),
            box(800, 300, 980, 350),
            box(830, 315, 970, 370),
        ]
        self.assertEqual(_subtitle_box_indices(texts, boxes, self.SHAPE), [])

    def test_keeps_mixed_number_hangul_dialogue(self):
        texts = ["2장"]
        boxes = [box(760, 275, 1010, 350)]
        self.assertEqual(_subtitle_box_indices(texts, boxes, self.SHAPE), [0])

    def test_rejects_upper_central_scene_text(self):
        texts = ["소원수리함", "진짜 대사"]
        boxes = [
            box(500, 20, 1000, 90),
            box(700, 285, 1100, 350),
        ]
        self.assertEqual(_subtitle_box_indices(texts, boxes, self.SHAPE), [1])

    def test_rejects_lower_edge_korean_false_positive(self):
        texts = ["제작진", "대사"]
        boxes = [
            box(1510, 300, 1740, 370),
            box(760, 300, 1040, 365),
        ]
        self.assertEqual(_subtitle_box_indices(texts, boxes, self.SHAPE), [1])

    def test_fallback_without_boxes_still_requires_hangul(self):
        texts = ["mockingbird", "맞아", "123"]
        self.assertEqual(
            _subtitle_box_indices(texts, None, self.SHAPE),
            [1],
        )

    def test_non_korean_mode_preserves_all_detections_without_boxes(self):
        texts = ["English only", "123"]
        self.assertEqual(
            _subtitle_box_indices(texts, None, self.SHAPE, require_hangul=False),
            [0, 1],
        )

    def test_same_visual_line_is_sorted_left_to_right_not_by_top_edge(self):
        # Regression from real EpSite QA: three boxes were one rendered line,
        # but slightly different box heights made y0 sorting produce
        # "야시시하다 / 하나 / 해야" instead of natural reading order.
        texts = ["야시시하다", "하나", "해야"]
        boxes = [
            box(560, 275, 870, 360),
            box(995, 278, 1190, 360),
            box(868, 280, 1010, 356),
        ]
        lines = _subtitle_box_lines(texts, boxes, self.SHAPE)
        self.assertEqual(lines, [[0, 2, 1]])
        self.assertEqual(_join_visual_line(texts, boxes, lines[0]), "야시시하다 해야 하나")

    def test_heavily_overlapping_fragments_rejoin_single_word(self):
        texts = ["성적", "인"]
        boxes = [
            box(570, 280, 705, 355),
            box(685, 282, 765, 354),
        ]
        lines = _subtitle_box_lines(texts, boxes, self.SHAPE)
        self.assertEqual(lines, [[0, 1]])
        self.assertEqual(_join_visual_line(texts, boxes, lines[0]), "성적인")

    def test_overlapping_boxes_do_not_duplicate_shared_edge_syllable(self):
        # Real EpSite regression: detector emitted "확인을" and "을해야"
        # over the same rendered 을 glyph. The subtitle itself reads
        # "확인을 해야 하나", so the shared edge glyph must be de-duplicated.
        texts = ["확인을", "을해야", "하나"]
        boxes = [
            box(535, 278, 770, 360),
            box(735, 280, 980, 358),
            box(1000, 279, 1160, 360),
        ]
        lines = _subtitle_box_lines(texts, boxes, self.SHAPE)
        self.assertEqual(lines, [[0, 1, 2]])
        self.assertEqual(_join_visual_line(texts, boxes, lines[0]), "확인을 해야 하나")

    def test_fully_duplicated_overlapping_box_is_dropped(self):
        texts = ["문제는", "는"]
        boxes = [
            box(620, 280, 840, 355),
            box(815, 282, 860, 353),
        ]
        lines = _subtitle_box_lines(texts, boxes, self.SHAPE)
        self.assertEqual(_join_visual_line(texts, boxes, lines[0]), "문제는")

    def test_minor_overlap_between_regular_words_keeps_space(self):
        # Slight detector-box jitter between two real words must not turn them
        # into one token.  8% overlap stays below the fragment-repair gate.
        texts = ["오늘은", "정말"]
        boxes = [
            box(620, 280, 820, 355),
            box(804, 282, 1004, 354),
        ]
        lines = _subtitle_box_lines(texts, boxes, self.SHAPE)
        self.assertEqual(lines, [[0, 1]])
        self.assertEqual(_join_visual_line(texts, boxes, lines[0]), "오늘은 정말")

    def test_raised_two_line_dialogue_is_recovered_in_all_four_languages(self):
        # Real PP-OCRv5 video crop is bottom 40% (288px at 720p).
        # The upper row's center lies ABOVE the old y=0.30 cutoff.
        shape = (288, 1280, 3)
        texts_by_language = {
            "ko": ("오늘은 정말", "좋은 날이에요"),
            "en": ("Where are you", "going tomorrow"),
            "ja": ("今日は何を", "していますか"),
            "zh": ("今天晚上", "一起去看电影吧"),
        }
        for language, texts in texts_by_language.items():
            with self.subTest(language=language):
                boxes = [
                    box(290, 44, 990, 95),
                    box(280, 124, 1000, 177),
                ]
                lines = _subtitle_box_lines(
                    texts, boxes, shape,
                    language=language if language != "ko" else None,
                )
                self.assertEqual(lines, [[0], [1]])

    def test_bounded_ocr_box_overhang_keeps_genuine_two_rows(self):
        # Two independent text rows can have PP-OCRv5 bounding rectangles
        # whose edges overlap slightly. Only the conservative paired row
        # gate is relaxed; isolated scene lettering is still rejected.
        shape = (288, 1280, 3)
        fixtures = {
            "ko": ("오늘은 정말", "좋은 날이에요"),
            "en": ("Where are you", "going tomorrow"),
            "ja": ("今日は何を", "していますか"),
            "zh": ("今天晚上", "一起去看电影吧"),
        }
        for language, strings in fixtures.items():
            with self.subTest(language=language):
                top = box(300, 30, 980, 110)
                bottom = box(300, 95, 980, 175)   # 15px edge overlap
                accepted = _subtitle_box_lines(
                    strings, [top, bottom], shape,
                    language=language if language != "ko" else None,
                )
                self.assertEqual(accepted, [[0], [1]])

    def test_upper_title_overlapping_too_far_is_not_promoted(self):
        shape = (288, 1280, 3)
        strings = ["영화 제목", "실제 자막"]
        # 40px overhang is larger than both the 8%-crop cap and 32% of
        # the shorter box. Centered same-script lettering alone is not
        # a sufficient justification to reinterpret this as dialogue.
        boxes = [
            box(300, 15, 980, 120),
            box(300, 80, 980, 185),
        ]
        self.assertEqual(
            _subtitle_box_lines(strings, boxes, shape), [[1]],
        )

    def test_an_offcenter_upper_credit_stays_excluded_after_overlap_fix(self):
        shape = (288, 1280, 3)
        texts = ["제작진", "실제 자막"]
        boxes = [
            box(25, 30, 250, 110),
            box(300, 95, 980, 175),
        ]
        self.assertEqual(
            _subtitle_box_lines(texts, boxes, shape), [[1]],
        )

    def test_upper_scene_sign_is_not_accepted_without_matching_lower_row(self):
        shape = (288, 1280, 3)
        # An isolated central heading cannot be silently included as
        # dialogue just because a distant lower subtitle also exists.
        self.assertEqual(
            _subtitle_box_lines(
                ["장면 글자", "실제 자막"],
                [box(310, 45, 980, 90), box(295, 217, 985, 268)],
                shape,
            ),
            [[1]],
        )
        # Different script and off-center upper text cannot be paired.
        self.assertEqual(
            _subtitle_box_lines(
                ["ENGLISH", "실제 자막"],
                [box(320, 45, 900, 97), box(295, 124, 985, 177)],
                shape,
            ),
            [[1]],
        )
        self.assertEqual(
            _subtitle_box_lines(
                ["영화 제목", "실제 자막"],
                [box(40, 45, 340, 95), box(295, 124, 985, 177)],
                shape,
            ),
            [[1]],
        )

    def test_two_subtitle_rows_remain_separate_lines(self):
        texts = ["앗또", "성적인", "느낌으로다가"]
        boxes = [
            box(770, 205, 890, 255),
            box(570, 285, 780, 365),
            box(775, 282, 1160, 360),
        ]
        lines = _subtitle_box_lines(texts, boxes, self.SHAPE)
        self.assertEqual(lines, [[0], [1, 2]])
        rendered = "\n".join(_join_visual_line(texts, boxes, line) for line in lines)
        self.assertEqual(rendered, "앗또\n성적인 느낌으로다가")


if __name__ == "__main__":
    unittest.main()
