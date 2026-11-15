import unittest
import numpy as np

from hardsub_extractor.models import Frame, OCRObservation
from hardsub_extractor.ocr.base import OCREngine
from hardsub_extractor.ocr.adaptive import AdaptiveOCREngine, AdaptiveOCRConfig
from hardsub_extractor.extraction_progress import ExtractionCancelled


class SequenceOCR(OCREngine):
    def __init__(self, outputs):
        self.outputs=list(outputs)
        self.calls=0
    def recognize(self, frame):
        item=self.outputs[min(self.calls,len(self.outputs)-1)]
        self.calls+=1
        return OCRObservation(item[0],item[1],frame.pts,item[2] if len(item)>2 else None)


class AdaptiveOCRTests(unittest.TestCase):
    def frame(self):
        return Frame(1.0,np.zeros((80,240,3),dtype=np.uint8))

    def test_high_confidence_uses_one_call(self):
        base=SequenceOCR([("정확한 자막",.95)])
        ocr=AdaptiveOCREngine(base)
        got=ocr.recognize(self.frame())
        self.assertEqual(got.text,"정확한 자막")
        self.assertEqual(base.calls,1)

    def test_low_confidence_retries_and_votes(self):
        base=SequenceOCR([
            ("괜찮0ㅏ",.40),
            ("괜찮아",.91),
            ("괜찮아",.88),
        ])
        ocr=AdaptiveOCREngine(
            base,
            AdaptiveOCRConfig(retry_below_confidence=.72,upscale=1.5,enable_clahe_retry=True)
        )
        got=ocr.recognize(self.frame())
        self.assertEqual(got.text,"괜찮아")
        self.assertEqual(base.calls,3)

    def test_high_sentence_average_does_not_hide_one_weak_word(self):
        weak = {'detections': [{'score': .55}, {'score': .99}, {'score': .99}]}
        strong = {'detections': [{'score': .97}, {'score': .99}]}
        base = SequenceOCR([
            ('다른 읽기', .96, weak),
            ('정확한 읽기', .98, strong),
            ('정확한 읽기', .97, strong),
        ])
        got = AdaptiveOCREngine(base).recognize(self.frame())
        self.assertEqual(base.calls, 3)
        self.assertEqual(got.text, '정확한 읽기')
        self.assertEqual(got.metadata, strong)

    def test_opt_in_same_text_uses_strongest_boxes_but_preserves_vote_confidence(self):
        def sequence():
            return SequenceOCR([
                ("같은 대사", .32, {
                    "detections": [{"line": 0, "score": .32, "tag": "first"}],
                }),
                ("같은 대사", .98, {
                    "detections": [{"line": 0, "score": .98, "tag": "retry"}],
                }),
            ])
        config = AdaptiveOCRConfig(enable_clahe_retry=False)
        plain_source = sequence()
        plain = AdaptiveOCREngine(plain_source, config).recognize(self.frame())
        filtered_source = sequence()
        filtered_engine = AdaptiveOCREngine(filtered_source, config)
        filtered = filtered_engine.recognize_with_filter(
            self.frame(), lambda observation: observation,
        )
        self.assertEqual(plain_source.calls, 2)
        self.assertEqual(filtered_source.calls, 2)
        self.assertEqual((plain.text, filtered.text), ("같은 대사", "같은 대사"))
        self.assertAlmostEqual(plain.confidence, .65)
        self.assertAlmostEqual(filtered.confidence, plain.confidence)
        # The default path remains byte-for-byte behavior compatible. Only
        # the explicitly filtered path upgrades matched text-box evidence.
        self.assertEqual(plain.metadata["detections"][0]["tag"], "first")
        self.assertEqual(filtered.metadata["detections"][0]["tag"], "retry")
        self.assertEqual((filtered_engine.total_calls, filtered_engine.retry_calls), (2, 1))

    def test_strong_word_boxes_skip_extra_inference(self):
        base = SequenceOCR([('정확한 읽기', .98,
                             {'detections': [{'score': .95}, {'score': .99}]})])
        AdaptiveOCREngine(base).recognize(self.frame())
        self.assertEqual(base.calls, 1)


    def test_failed_optional_upscale_keeps_primary_and_attempts_clahe(self):
        class Flaky(OCREngine):
            calls = 0
            def recognize(self, frame):
                self.calls += 1
                if self.calls == 2:
                    raise RuntimeError("upscale failed")
                return OCRObservation("정상 대사", .61 if self.calls == 1 else .97, frame.pts)
        base = Flaky()
        adaptive = AdaptiveOCREngine(base)
        got = adaptive.recognize(self.frame())
        self.assertEqual((got.text, base.calls, adaptive.retry_calls), ("정상 대사", 3, 2))
        self.assertEqual(got.metadata["optional_ocr_retry_failures"], 1)

    def test_two_optional_failures_do_not_discard_primary_text(self):
        class Flaky(OCREngine):
            calls = 0
            def recognize(self, frame):
                self.calls += 1
                if self.calls > 1:
                    raise RuntimeError("enhancement failed")
                return OCRObservation("처음 인식한 문장", .60, frame.pts)
        base = Flaky()
        got = AdaptiveOCREngine(base).recognize(self.frame())
        self.assertEqual((got.text, base.calls), ("처음 인식한 문장", 3))
        self.assertEqual(got.metadata["optional_ocr_retry_failures"], 2)

    def test_primary_model_exception_still_propagates(self):
        class Broken(OCREngine):
            def recognize(self, frame):
                raise RuntimeError("primary failed")
        with self.assertRaisesRegex(RuntimeError, "primary failed"):
            AdaptiveOCREngine(Broken()).recognize(self.frame())


    def test_cancel_during_optional_retry_propagates(self):
        class UserCancelled(OCREngine):
            calls = 0
            def recognize(self, frame):
                self.calls += 1
                if self.calls == 2:
                    raise ExtractionCancelled("user cancelled")
                return OCRObservation("취소 전 자막", .60, frame.pts)
        engine = UserCancelled()
        with self.assertRaises(ExtractionCancelled):
            AdaptiveOCREngine(engine).recognize(self.frame())
        self.assertEqual(engine.calls, 2)


if __name__=="__main__":
    unittest.main()

