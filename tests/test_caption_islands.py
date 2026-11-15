"""Targeted behavior only: preserve real 3+ line captions and prevent unsafe deletion."""
from __future__ import annotations

import unittest
import cv2
import numpy as np
from hardsub_extractor.models import Frame, OCRObservation
from hardsub_extractor.ocr.base import OCREngine
from hardsub_extractor.ocr.adaptive import AdaptiveOCREngine
from hardsub_extractor.ocr.caption_islands import (
    CaptionIslandEngine, LightCaptionIslandEngine, filter_caption_island,
)


def obs(text, boxes):
    return OCRObservation(
        text, 0.95, 4.0,
        {"detections": [
            {"line": line, "cx": x, "cy": y, "width": w,
             "height": h, "score": 0.95, "text": "synthetic"}
            for line, x, y, w, h in boxes
        ]},
    )


class CaptionIslandTests(unittest.TestCase):
    def test_separated_upper_title_three_real_caption_rows_preserved(self):
        o = obs(
            "SCENE TITLE\nfirst subtitle\nsecond subtitle\nthird subtitle",
            [
                (0, .5, .14, .30, .06),
                (1, .5, .56, .50, .07),
                (2, .5, .65, .52, .07),
                (3, .5, .74, .55, .07),
            ],
        )
        output = filter_caption_island(o)
        self.assertEqual(
            output.text,
            "first subtitle\nsecond subtitle\nthird subtitle",
        )
        self.assertEqual(output.pts, o.pts)
        self.assertEqual(output.metadata["caption_island_kept_lines"], 3)
        self.assertEqual(
            [entry["line"] for entry in output.metadata["detections"]],
            [0, 1, 2],
        )
        self.assertEqual(o.text.splitlines()[0], "SCENE TITLE")

    def test_two_line_dialogue_preferred_to_single_bottom_watermark(self):
        o = obs(
            "dialogue row 1\ndialogue row 2\nwatermark",
            [(0,.48,.43,.44,.08),(1,.50,.53,.52,.08),(2,.85,.94,.1,.025)],
        )
        self.assertEqual(
            filter_caption_island(o).text,
            "dialogue row 1\ndialogue row 2",
        )

    def test_legitimate_four_rows_or_single_line_never_arbitrarily_trimmed(self):
        four = obs(
            "A\nB\nC\nD",
            [(i,.5,.45+i*.09,.40,.075) for i in range(4)],
        )
        single = obs("single dialogue", [(0,.5,.65,.4,.08)])
        self.assertIs(filter_caption_island(four), four)
        self.assertIs(filter_caption_island(single), single)

    def test_long_single_line_watermark_cannot_displace_real_one_line_dialogue(self):
        original = obs(
            "LONG BROADCAST LOGO WATERMARK\n짧은 진짜 대사",
            [
                (0, .5, .15, .8, .06),
                (1, .5, .82, .28, .065),
            ],
        )
        self.assertIs(filter_caption_island(original), original)
        self.assertEqual(original.text.splitlines()[-1], "짧은 진짜 대사")

    def test_two_separate_multiline_blocks_preserve_dual_caption_content(self):
        original = obs(
            "TOP TITLE A\nTOP TITLE B\nspoken subtitle A\nspoken subtitle B",
            [
                (0, .5, .18, .55, .055),
                (1, .5, .25, .55, .055),
                (2, .5, .70, .49, .065),
                (3, .5, .80, .52, .065),
            ],
        )
        self.assertIs(filter_caption_island(original), original)
        self.assertEqual(len(original.text.splitlines()), 4)

    def test_no_geometry_or_malformed_group_keeps_original(self):
        o = OCRObservation("one\ntwo\nthree", 0.9, 1.0, None)
        self.assertIs(filter_caption_island(o), o)
        mismatched = obs("one\ntwo\nthree", [(0,.5,.3,.5,.05),(2,.5,.7,.5,.05)])
        self.assertIs(filter_caption_island(mismatched), mismatched)
        invalid = obs("one\ntwo", [(0,.5,.3,.5,.05),(1,float("nan"),.6,.5,.05)])
        self.assertIs(filter_caption_island(invalid), invalid)

    def test_removed_high_confidence_title_cannot_certify_weak_dialogue(self):
        original = obs(
            "TITLE\nfirst subtitle\nsecond subtitle",
            [(0,.5,.12,.40,.06),(1,.5,.62,.40,.07),(2,.5,.72,.40,.07)],
        )
        boxes = original.metadata["detections"]
        boxes[0]["score"] = .99
        boxes[1]["score"] = .50
        boxes[2]["score"] = .60
        result = filter_caption_island(original)
        self.assertEqual(result.text, "first subtitle\nsecond subtitle")
        self.assertAlmostEqual(result.confidence, .55)
        self.assertTrue(result.metadata["caption_island_retained_confidence_bounded"])
        self.assertEqual(original.confidence, .95)
        self.assertEqual(original.metadata["detections"][0]["line"], 0)

    def test_no_confidence_inflation_and_incomplete_box_scores_do_not_guess(self):
        original = obs(
            "TITLE\nfirst subtitle\nsecond subtitle",
            [(0,.5,.12,.40,.06),(1,.5,.62,.40,.07),(2,.5,.72,.40,.07)],
        )
        original.confidence = .65
        result = filter_caption_island(original)
        self.assertEqual(result.confidence, .65)
        self.assertNotIn("caption_island_retained_confidence_bounded",result.metadata)
        original.metadata["detections"][1].pop("score")
        result = filter_caption_island(original)
        self.assertEqual(result.confidence, .65)
        self.assertNotIn("caption_island_retained_confidence_bounded",result.metadata)

    def test_combined_bright_and_islands_uses_one_adaptive_read_and_original_pixels(self):
        image = np.full((240, 320, 3), 250, dtype=np.uint8)
        rows = [
            ("BRIGHT LOGO", .13, True),
            ("DARK ARTICLE", .34, False),
            ("caption line one", .65, True),
            ("caption line two", .76, True),
        ]
        boxes = []
        for index, (label, cy, bright) in enumerate(rows):
            y0 = round(cy * 240) - 10
            y1 = y0 + 20
            if bright:
                image[y0:y1, 45:275] = 105
                cv2.rectangle(image, (55, y0 + 5), (245, y0 + 8), (255,255,255), -1)
            else:
                cv2.rectangle(image, (55, y0 + 5), (245, y0 + 8), (0,0,0), -1)
            boxes.append({
                "line": index, "cx": .5, "cy": cy, "width": .72,
                "height": 20/240, "score": .98, "text": label,
            })

        class SingleRead:
            calls = 0
            def recognize(self, frame):
                self.calls += 1
                return OCRObservation(
                    "\n".join(row[0] for row in rows),
                    .98, frame.pts, {"detections": boxes},
                )

        raw = SingleRead()
        engine = LightCaptionIslandEngine(AdaptiveOCREngine(raw))
        actual = engine.recognize(Frame(3.0, image))
        self.assertEqual(actual.text, "caption line one\ncaption line two")
        self.assertEqual([row["line"] for row in actual.metadata["detections"]], [0,1])
        self.assertEqual((raw.calls, engine.total_calls, engine.retry_calls), (1,1,0))
        self.assertEqual(engine.candidate_frames, 1)
        self.assertEqual((engine.discarded_rows, engine.dropped_rows), (1,1))
        self.assertEqual(engine.modified_observations, 1)
        self.assertLessEqual(actual.confidence, .98)

    def test_adaptive_filters_each_model_reading_before_vote(self):
        def reading(dialogue, score):
            result = obs(
                "CREDIT\n" + dialogue + "\nsecond subtitle",
                [(0,.5,.12,.40,.06),(1,.5,.62,.40,.07),(2,.5,.72,.40,.07)],
            )
            result.confidence = .98
            result.metadata["detections"][0]["score"] = .99
            result.metadata["detections"][1]["score"] = score
            result.metadata["detections"][2]["score"] = score
            return result

        class ThreeReadings(OCREngine):
            def __init__(self):
                self.calls = 0
                self.data = [
                    reading("wrong subtitle", .45),
                    reading("correct subtitle", .98),
                    reading("correct subtitle", .98),
                ]

            def recognize(self, frame):
                result = self.data[min(self.calls, 2)]
                self.calls += 1
                return OCRObservation(
                    result.text, result.confidence, frame.pts, result.metadata
                )

        base = ThreeReadings()
        adaptive = AdaptiveOCREngine(base)
        selected = CaptionIslandEngine(adaptive)
        result = selected.recognize(Frame(1.0,np.zeros((80,240,3),dtype=np.uint8)))
        self.assertEqual(result.text, "correct subtitle\nsecond subtitle")
        self.assertEqual(base.calls, 3)
        self.assertEqual((adaptive.total_calls, adaptive.retry_calls), (3,2))
        self.assertEqual(selected.modified_observations, 3)
        self.assertEqual(selected.dropped_rows, 3)
        self.assertEqual([x["line"] for x in result.metadata["detections"]],[0,1])


if __name__ == "__main__":
    unittest.main()
