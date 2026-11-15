"""Only new light-text polarity logic; no model calls, network, or user media."""
import unittest
import numpy as np
import cv2
from hardsub_extractor.models import Frame, OCRObservation
from hardsub_extractor.ocr.light_text import (
    LightCaptionEngine, select_light_caption_lines,
)


def fixture(*, first_bright=True, second_bright=False):
    img = np.full((200, 400, 3), 250, dtype=np.uint8)
    records = []
    for i, bright in enumerate((first_bright, second_bright)):
        x0, x1, y0, y1 = 90, 310, 80 + i * 50, 105 + i * 50
        if bright:
            img[y0:y1, x0:x1] = 105
            cv2.rectangle(img, (x0+9,y0+8), (x0+170,y0+10), (255,255,255), -1)
        else:
            cv2.rectangle(img, (x0+9,y0+8), (x0+170,y0+10), (4,4,4), -1)
        records.append({
            "line": i, "cx": (x0+x1)/800, "cy": (y0+y1)/400,
            "width": (x1-x0)/400, "height": (y1-y0)/200,
            "score": .98, "text": ("subtitle" if i == 0 else "webpage"),
        })
    return Frame(3.5, img), OCRObservation(
        "subtitle\nwebpage", .97, 3.5, {"detections": records}
    )


class LightSubtitleContracts(unittest.TestCase):
    def test_only_bright_subtitle_row_survives_black_page_text(self):
        frame, raw = fixture()
        out = select_light_caption_lines(raw, frame.image)
        self.assertEqual(out.text, "subtitle")
        self.assertEqual(out.pts, raw.pts)
        self.assertEqual(out.metadata["light_text_discarded_rows"], 1)
        self.assertEqual([r["line"] for r in out.metadata["detections"]], [0])
        self.assertEqual(raw.text, "subtitle\nwebpage")

    def test_discarded_black_credit_cannot_certify_weak_bright_caption(self):
        frame, reading = fixture(first_bright=True, second_bright=False)
        # A black page title originally boosted the complete OCR sentence.
        reading.confidence = .96
        reading.metadata["detections"][0]["score"] = .31
        reading.metadata["detections"][1]["score"] = .99
        result = select_light_caption_lines(reading, frame.image)
        self.assertEqual(result.text, "subtitle")
        self.assertAlmostEqual(result.confidence, .31)
        self.assertEqual(result.metadata["light_text_discarded_rows"], 1)
        self.assertTrue(result.metadata["light_text_retained_confidence_bounded"])
        self.assertEqual(len(reading.metadata["detections"]), 2)
        self.assertAlmostEqual(reading.confidence, .96)

    def test_kept_box_score_must_not_inflate_original_adaptive_confidence(self):
        frame, reading = fixture(first_bright=True, second_bright=False)
        reading.confidence = .72
        reading.metadata["detections"][0]["score"] = .98
        result = select_light_caption_lines(reading, frame.image)
        self.assertEqual(result.text, "subtitle")
        self.assertEqual(result.confidence, .72)
        self.assertNotIn(
            "light_text_retained_confidence_bounded", result.metadata,
        )

    def test_bright_caption_low_confidence_is_retried_before_vote(self):
        from hardsub_extractor.ocr.adaptive import AdaptiveOCREngine, AdaptiveOCRConfig
        frame, reading = fixture(first_bright=True, second_bright=False)
        reading.confidence = .99
        reading.metadata["detections"][0]["score"] = .32
        reading.metadata["detections"][1]["score"] = .99
        strong = {
            "detections": [{
                **reading.metadata["detections"][0],
                "line": 0, "score": .98,
            }],
        }

        class Sequence:
            calls = 0
            def recognize(self, frame):
                self.calls += 1
                if self.calls == 1:
                    return reading
                return OCRObservation("subtitle", .98, frame.pts, strong)

        base = Sequence()
        engine = LightCaptionEngine(AdaptiveOCREngine(
            base, AdaptiveOCRConfig(enable_clahe_retry=False),
        ))
        result = engine.recognize(frame)
        self.assertEqual(base.calls, 2)
        self.assertEqual(result.text, "subtitle")
        # The existing HSE consensus deliberately does NOT turn one strong
        # retry into near-certainty: matching .32/.98 readings average .65.
        self.assertAlmostEqual(result.confidence, .65)
        self.assertEqual(result.metadata["detections"][0]["score"], .98)
        self.assertEqual(engine.retry_calls, 1)
        self.assertEqual(engine.candidate_frames, 1)

    def test_two_dark_rows_fail_closed_without_guessing(self):
        frame, raw = fixture(first_bright=False, second_bright=False)
        self.assertEqual(select_light_caption_lines(raw, frame.image).text, "")

    def test_two_light_rows_preserve_original_and_multiline(self):
        frame, raw = fixture(first_bright=True, second_bright=True)
        self.assertIs(select_light_caption_lines(raw, frame.image), raw)

    def test_missing_geometry_never_passes_as_light_subtitle(self):
        frame, raw = fixture()
        raw.metadata = None
        self.assertEqual(select_light_caption_lines(raw, frame.image).text, "")

    def test_adaptive_vote_discards_black_credit_before_retrying_real_caption(self):
        from hardsub_extractor.ocr.adaptive import AdaptiveOCREngine, AdaptiveOCRConfig

        frame, original = fixture(first_bright=True, second_bright=False)
        bright = {**original.metadata["detections"][0], "line": 0}
        dark = {**original.metadata["detections"][1], "line": 0}

        class TwoReadings:
            def __init__(self):
                self.calls = 0
                self.shapes = []

            def recognize(self, sample):
                self.shapes.append(sample.image.shape)
                self.calls += 1
                if self.calls == 1:
                    # Raw winner without polarity filtering is a dark scene
                    # credit and is erroneously considered fully confident.
                    return OCRObservation("webpage", .99, sample.pts,
                                          {"detections": [dark]})
                return OCRObservation("subtitle", .97, sample.pts,
                                      {"detections": [bright]})

        defaults = TwoReadings()
        standard = AdaptiveOCREngine(
            defaults,
            AdaptiveOCRConfig(enable_clahe_retry=False),
        )
        self.assertEqual(standard.recognize(frame).text, "webpage")
        self.assertEqual(defaults.calls, 1)

        raw = TwoReadings()
        selected = LightCaptionEngine(AdaptiveOCREngine(
            raw, AdaptiveOCRConfig(enable_clahe_retry=False),
        ))
        result = selected.recognize(frame)
        self.assertEqual(result.text, "subtitle")
        self.assertEqual(raw.calls, 2)
        self.assertGreater(raw.shapes[1][1], raw.shapes[0][1])
        self.assertEqual(selected.candidate_frames, 1)
        self.assertEqual(selected.discarded_rows, 1)
        self.assertEqual(result.metadata["detections"][0]["text"], "subtitle")

    def test_adaptive_light_caption_keeps_bright_first_pass_to_one_model_call(self):
        from hardsub_extractor.ocr.adaptive import AdaptiveOCREngine
        frame, original = fixture(first_bright=True, second_bright=False)
        bright = {**original.metadata["detections"][0], "line": 0}

        class Bright:
            calls = 0
            def recognize(self, sample):
                self.calls += 1
                return OCRObservation("subtitle", .98, sample.pts,
                                      {"detections": [bright]})

        base = Bright()
        selected = LightCaptionEngine(AdaptiveOCREngine(base))
        self.assertEqual(selected.recognize(frame).text, "subtitle")
        self.assertEqual((base.calls, selected.candidate_frames), (1, 1))
        self.assertEqual(selected.discarded_rows, 0)

    def test_dark_credit_does_not_become_subtitle_after_adaptive_votes(self):
        from hardsub_extractor.ocr.adaptive import AdaptiveOCREngine, AdaptiveOCRConfig
        frame, original = fixture(first_bright=False, second_bright=False)
        dark = {**original.metadata["detections"][1], "line": 0}

        class BlackOnly:
            calls = 0
            def recognize(self, sample):
                self.calls += 1
                return OCRObservation("webpage", .99, sample.pts,
                                      {"detections": [dark]})

        base = BlackOnly()
        result = LightCaptionEngine(AdaptiveOCREngine(
            base, AdaptiveOCRConfig(enable_clahe_retry=False),
        )).recognize(frame)
        self.assertEqual(result.text, "")
        self.assertEqual(base.calls, 2)

    def test_retry_color_classification_uses_original_source_pixels(self):
        from hardsub_extractor.ocr.adaptive import AdaptiveOCREngine, AdaptiveOCRConfig
        frame, original = fixture(first_bright=True, second_bright=False)
        bright = {**original.metadata["detections"][0], "line": 0}
        dark = {**original.metadata["detections"][1], "line": 0}

        class ArtificialEnhancement:
            calls = 0
            def recognize(self, sample):
                self.calls += 1
                if self.calls == 1:
                    return OCRObservation("webpage", .99, sample.pts,
                                          {"detections": [dark]})
                # Force the upscaled sample's pixels white; only a classifier
                # using the *original* video frame can retain the actual
                # bright subtitle instead of trusting altered retry pixels.
                sample.image[:] = 255
                return OCRObservation("subtitle", .98, sample.pts,
                                      {"detections": [bright]})

        base = ArtificialEnhancement()
        result = LightCaptionEngine(AdaptiveOCREngine(
            base, AdaptiveOCRConfig(enable_clahe_retry=False),
        )).recognize(frame)
        self.assertEqual(result.text, "subtitle")
        self.assertEqual(base.calls, 2)

    def test_wrapper_calls_inner_once(self):
        frame, raw = fixture()
        class CountingEngine:
            calls = 0
            def recognize(self, supplied):
                self.calls += 1
                self.last = supplied
                return raw
        base = CountingEngine()
        mode = LightCaptionEngine(base)
        self.assertEqual(mode.recognize(frame).text, "subtitle")
        self.assertEqual(base.calls, 1)
        self.assertEqual(mode.discarded_rows, 1)


if __name__ == "__main__":
    unittest.main()
