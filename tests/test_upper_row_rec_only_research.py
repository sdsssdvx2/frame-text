"""No-oracle recognizer-only upper-row research contracts; no OCR model needed."""
from __future__ import annotations

from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"tools"))
import compare_upper_row_rec_only as study
from verify_animated_background_ocr_quality import _multiline_reference
from verify_multilang_video_pipeline import VIDEO_TEXT


def box(x0, y0, x1, y1):
    return np.asarray([[x0, y0], [x1, y0],
                       [x1, y1], [x0, y1]], dtype=float)


class UpperRowRecognitionResearchTests(unittest.TestCase):
    IMAGE = np.zeros((288, 1280, 3), dtype=np.uint8)

    def test_upper_box_is_selected_without_caption_time_or_reference(self):
        raw = SimpleNamespace(
            txts=["上方字幕", "下面的话"],
            scores=[0.95, 0.95],
            boxes=[box(290, 40, 990, 96), box(290, 125, 990, 180)],
        )
        items, texts, scores = study._upper_boxes(raw, self.IMAGE.shape)
        self.assertEqual([x["index"] for x in items], [0])
        self.assertEqual(len(texts), len(scores))

    def test_no_upper_box_means_no_unsupervised_text_invention(self):
        raw = SimpleNamespace(
            txts=["下面的话"], scores=[0.98],
            boxes=[box(290, 125, 990, 180)],
        )
        calls = []
        def recognizer(*args, **kwargs):
            calls.append(1)
            return SimpleNamespace(txts=["invented"], scores=[1.0])
        found = study.variants_for_raw_frame(raw, self.IMAGE, recognizer)
        self.assertEqual(found, {})
        self.assertEqual(calls, [])

    def test_real_crop_variants_only_use_detector_observed_upper_region(self):
        raw = SimpleNamespace(
            txts=["错误上行", "下面的话"],
            scores=[0.73, 0.99],
            boxes=[box(290, 40, 990, 96), box(290, 125, 990, 180)],
        )
        invocations = []
        def recognizer(crop, **kwargs):
            self.assertFalse(kwargs["use_det"])
            self.assertTrue(kwargs["use_rec"])
            invocations.append(tuple(crop.shape))
            return SimpleNamespace(txts=["正确上行"], scores=[0.98])
        result = study.variants_for_raw_frame(raw, self.IMAGE, recognizer)
        self.assertEqual(result["raw_detected_upper"], ("错误上行", 0.73))
        self.assertEqual(result["recognize_crop"], ("正确上行", 0.98))
        self.assertEqual(result["unsupervised_highest_confidence"],
                         ("正确上行", 0.98))
        self.assertEqual(len(invocations), 3)
        self.assertLess(invocations[0][0], self.IMAGE.shape[0])
        self.assertLess(invocations[0][1], self.IMAGE.shape[1])

    def test_persistent_heading_without_lower_dialogue_is_not_a_candidate(self):
        raw = SimpleNamespace(
            txts=["电影标题"], scores=[0.99],
            boxes=[box(290, 40, 990, 96)],
        )
        def forbidden_recognizer(*args, **kwargs):
            self.fail("unpaired heading must never launch another OCR call")
        for language in ("ja", "zh"):
            with self.subTest(language=language):
                self.assertEqual(
                    study.variants_for_raw_frame(
                        raw, self.IMAGE, forbidden_recognizer,
                        language=language,
                    ), {},
                )

    def test_distant_and_offcenter_lower_text_cannot_enable_upper_retry(self):
        raw = SimpleNamespace(
            txts=["错误上行", "下面的话"], scores=[0.93, 0.98],
            boxes=[box(290, 40, 990, 96), box(5, 198, 260, 259)],
        )
        self.assertEqual(
            study.variants_for_raw_frame(
                raw, self.IMAGE, lambda *_a, **_k: self.fail(
                    "unpaired box should never be retried"
                ), language="zh",
            ), {},
        )

    def test_consensus_requires_unique_source_timestamps(self):
        item = {
            "raw_detected_upper": ("錯字", 0.97),
            "recognize_crop": ("正確", 0.92),
            "recognize_upscaled": ("正確", 0.95),
        }
        self.assertEqual(
            study.temporal_agreement([
                (3, 1.625, dict(item)),
                (3, 1.625, dict(item)),
            ]), {},
        )
        self.assertIn(3, study.temporal_agreement([
            (3, 1.625, dict(item)),
            (3, 1.875, dict(item)),
        ]))


    def test_cross_model_verifier_never_promotes_unpaired_scene_text(self):
        raw = SimpleNamespace(
            txts=["电影标题"], scores=[0.99],
            boxes=[box(290, 40, 990, 96)],
        )
        calls = []
        def verifier(*args, **kwargs):
            calls.append(1)
            return SimpleNamespace(txts=["赝品"], scores=[0.99])
        self.assertEqual(study.variants_for_raw_frame(
            raw, self.IMAGE, verifier, language="zh",
            independent_recognizer=verifier,
        ), {})
        self.assertEqual(calls, [])

    def test_same_frame_two_recognizer_models_are_not_two_temporal_frames(self):
        item = {
            "raw_detected_upper": ("错误", 0.92),
            "recognize_crop": ("正確字", 0.88),
            "recognize_upscaled": ("正確字", 0.91),
            "independent_v6": ("正確字", 0.99),
        }
        self.assertEqual(study.cross_model_temporal_agreement([
            (2, 1.625, item), (2, 1.625, item),
        ]), {})
        result = study.cross_model_temporal_agreement([
            (2, 1.625, item), (2, 1.875, item),
        ])
        self.assertEqual(result[2]["text"], "正確字")
        self.assertEqual(result[2]["independent_model_support_frames"], 2)

    def test_cross_model_conflict_and_missing_result_abstain(self):
        base = {
            "raw_detected_upper": ("错字", 0.85),
            "recognize_crop": ("真正文字", 0.91),
            "recognize_upscaled": ("真正文字", 0.90),
        }
        good = {**base, "independent_v6": ("真正文字", 0.89)}
        disagree = {**base, "independent_v6": ("其它文字", 0.98)}
        missing = {**base, "independent_v6": ("", None)}
        self.assertEqual(study.cross_model_temporal_agreement([
            (4, 2.125, good), (4, 2.375, disagree), (4, 2.625, good),
        ]), {})
        self.assertEqual(study.cross_model_temporal_agreement([
            (4, 2.125, missing), (4, 2.375, good),
        ]), {})

    def test_independent_verifier_cannot_override_source_recognizer(self):
        raw = SimpleNamespace(
            txts=["错误上行", "下面的话"],
            scores=[0.75, 0.99],
            boxes=[box(290, 40, 990, 96), box(290, 125, 990, 180)],
        )
        def v5(crop, **kwargs):
            return SimpleNamespace(txts=["原读"], scores=[0.93])
        def v6(crop, **kwargs):
            return SimpleNamespace(txts=["独立读"], scores=[0.99])
        got=study.variants_for_raw_frame(
            raw, self.IMAGE, v5, language="zh",
            independent_recognizer=v6,
        )
        self.assertEqual(got["recognize_crop"][0], "原读")
        self.assertEqual(got["independent_v6"][0], "独立读")
        self.assertEqual(
            got["unsupervised_highest_confidence"][0], "原读",
        )

    def test_consensus_requires_two_real_variants_on_two_distinct_frames(self):
        agree = {
            "raw_detected_upper": ("真正錯字", 0.97),
            "recognize_crop": ("真正的字", 0.86),
            "recognize_upscaled": ("真正的字", 0.93),
            "recognize_clahe_upscaled": ("別的錯字", 0.99),
        }
        votes = [
            (7, 1.625, dict(agree)),
            (7, 1.875, dict(agree)),
        ]
        result = study.temporal_agreement(votes)
        self.assertIn(7, result)
        self.assertEqual(result[7]["text"], "真正的字")
        self.assertEqual(result[7]["support_frames"], 2)

    def test_single_blank_flash_is_not_temporal_consensus(self):
        hallucination = {
            "raw_detected_upper": ("錯字", 0.96),
            "recognize_crop": ("假字", 0.92),
            "recognize_upscaled": ("假字", 0.95),
            "recognize_clahe_upscaled": ("假字", 0.99),
        }
        self.assertEqual(
            study.temporal_agreement([(5, 0.5, hallucination)]), {},
        )

    def test_rec_only_disagreement_cannot_override_raw_reading(self):
        disagreement = {
            "raw_detected_upper": ("錯字", 0.93),
            "recognize_crop": ("甲甲", 0.92),
            "recognize_upscaled": ("乙乙", 0.94),
            "recognize_clahe_upscaled": ("丙丙", 0.95),
        }
        self.assertEqual(
            study.temporal_agreement([(2, 1.5, disagreement),
                                      (2, 1.75, disagreement)]), {},
        )

    def test_invalid_confidences_are_not_promoted(self):
        for invalid in (None, float("nan"), float("inf"), -1.0, 1.1):
            with self.subTest(value=invalid):
                self.assertIsNone(study._safe_confidence(invalid))
        self.assertAlmostEqual(study._safe_confidence(0.89), 0.89)

    def test_post_recognition_evaluator_checks_reference_but_not_input(self):
        lang = "zh"
        actual_upper = _multiline_reference(VIDEO_TEXT[lang][0])[0]
        selected = [(
            0, 1.625, {
                "raw_detected_upper": ("错字", 0.80),
                "recognize_upscaled": (actual_upper, 0.92),
                "unsupervised_highest_confidence": (actual_upper, 0.92),
            }
        )]
        report = study.evaluate_known_source(selected, lang)
        self.assertEqual(report[0]["raw_detected_upper_correct"], 0)
        self.assertEqual(report[0]["recognize_upscaled_correct"], 1)
        self.assertEqual(report[0]["unsupervised_highest_confidence_correct"], 1)
        self.assertEqual(report[1]["recognized_upper_candidate_frames"], 0)


if __name__ == "__main__":
    unittest.main()
