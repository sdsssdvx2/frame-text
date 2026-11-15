"""Synthetic model-output contracts for raw OCR vs accepted subtitle rows."""
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"tools"))
from diagnose_raw_multiline_ocr_boxes import analyze_raw_result, evaluate_samples
from verify_animated_background_ocr_quality import _multiline_reference
from verify_multilang_video_pipeline import VIDEO_TEXT

SHAPE = (288, 1280, 3)


def rectangle(y0, y1):
    return np.asarray([[300, y0], [980, y0],
                       [980, y1], [300, y1]], dtype=float)


class RawVsFilteredOCRTests(unittest.TestCase):
    def test_both_rows_survive_four_language_geometry(self):
        for language in ("ko", "en", "ja", "zh"):
            with self.subTest(language=language):
                text = VIDEO_TEXT[language][0]
                first, second = _multiline_reference(text)
                raw = SimpleNamespace(
                    txts=[first, second],
                    boxes=[rectangle(40, 94), rectangle(125, 180)],
                )
                metrics = analyze_raw_result(raw, SHAPE, language, text)
                self.assertTrue(metrics["raw_box_contract_ok"])
                self.assertTrue(metrics["raw_upper_text_match"])
                self.assertTrue(metrics["raw_lower_text_match"])
                self.assertTrue(metrics["kept_upper_text_match"])
                self.assertTrue(metrics["kept_lower_text_match"])
                self.assertTrue(metrics["both_lines_kept_exact"])

    def test_raw_upper_detection_can_be_lost_when_lower_is_absent(self):
        text = VIDEO_TEXT["ko"][0]
        upper, _ = _multiline_reference(text)
        raw = SimpleNamespace(txts=[upper], boxes=[rectangle(40, 94)])
        result = analyze_raw_result(raw, SHAPE, "ko", text)
        self.assertTrue(result["raw_upper_text_match"])
        self.assertFalse(result["kept_upper_text_match"])
        self.assertEqual(result["kept_line_count"], 0)

    def test_real_missing_upper_is_not_mislabeled_filter_loss(self):
        text = VIDEO_TEXT["ko"][0]
        _, lower = _multiline_reference(text)
        raw = SimpleNamespace(txts=[lower], boxes=[rectangle(125, 180)])
        result = analyze_raw_result(raw, SHAPE, "ko", text)
        self.assertFalse(result["raw_upper_text_match"])
        self.assertFalse(result["kept_upper_text_match"])
        self.assertTrue(result["raw_lower_text_match"])
        self.assertTrue(result["kept_lower_text_match"])

    def test_rejected_upper_y_is_classified_without_rewriting_filter(self):
        reference = VIDEO_TEXT["ko"][0]
        first, second = _multiline_reference(reference)
        result = analyze_raw_result(
            SimpleNamespace(
                txts=[first, second],
                boxes=[rectangle(0, 25), rectangle(125, 180)],
            ),
            SHAPE, "ko", reference,
        )
        self.assertTrue(result["raw_upper_text_match"])
        self.assertFalse(result["kept_upper_text_match"])
        geom = result["raw_two_row_filter_geometry"]
        self.assertEqual(geom["upper_reject_before_group_y"], 1)
        self.assertFalse(geom["raw_pair_geometry_possible"])
        self.assertNotIn(reference, repr(geom))

    def test_rejected_upper_x_is_classified_even_if_text_exact(self):
        reference = VIDEO_TEXT["ko"][0]
        first, second = _multiline_reference(reference)
        left = np.asarray(
            [[5, 45], [120, 45], [120, 95], [5, 95]], dtype=float
        )
        result = analyze_raw_result(
            SimpleNamespace(txts=[first, second],
                            boxes=[left, rectangle(125, 180)]),
            SHAPE, "ko", reference,
        )
        self.assertTrue(result["raw_upper_text_match"])
        self.assertFalse(result["kept_upper_text_match"])
        self.assertEqual(
            result["raw_two_row_filter_geometry"][
                "upper_reject_before_group_x"
            ], 1,
        )

    def test_reason_counters_do_not_inflate_missing_model_text(self):
        reference = VIDEO_TEXT["zh"][0]
        _, lower = _multiline_reference(reference)
        result = analyze_raw_result(
            SimpleNamespace(txts=[lower], boxes=[rectangle(125, 180)]),
            SHAPE, "zh", reference,
        )
        self.assertEqual(result["raw_two_row_filter_geometry"], {})
        self.assertFalse(result["raw_upper_text_match"])

    def test_row_edges_overlap_while_centers_are_distinct(self):
        reference = VIDEO_TEXT["ko"][0]
        first, second = _multiline_reference(reference)
        raw = SimpleNamespace(
            txts=[first, second],
            boxes=[rectangle(30, 110), rectangle(95, 175)],
        )
        result = analyze_raw_result(raw, SHAPE, "ko", reference)
        self.assertTrue(result["raw_upper_text_match"])
        self.assertTrue(result["kept_upper_text_match"])
        geometry = result["raw_two_row_filter_geometry"]
        self.assertTrue(geometry["row_center_conditions_pass"])
        self.assertFalse(geometry["row_edge_gap_original_pass"])
        self.assertTrue(geometry["row_edge_gap_current_pass"])
        self.assertTrue(geometry["row_edge_gap_relaxed_010_pass"])
        self.assertAlmostEqual(geometry["row_vertical_gap_norm"], -0.052, places=3)

    def test_distant_overlap_still_rejected_by_stricter_pairing(self):
        reference = VIDEO_TEXT["ko"][0]
        first, second = _multiline_reference(reference)
        result = analyze_raw_result(
            SimpleNamespace(
                txts=[first, second],
                boxes=[rectangle(15, 120), rectangle(80, 185)],
            ), SHAPE, "ko", reference,
        )
        geometry = result["raw_two_row_filter_geometry"]
        self.assertTrue(geometry["row_center_conditions_pass"])
        self.assertFalse(geometry["row_edge_gap_original_pass"])
        self.assertFalse(geometry["row_edge_gap_current_pass"])
        self.assertFalse(geometry["row_edge_gap_relaxed_010_pass"])
        self.assertFalse(result["kept_upper_text_match"])

    def test_geometry_mismatch_fails_closed(self):
        raw = SimpleNamespace(txts=["hello"], boxes=[])
        metrics = analyze_raw_result(raw, SHAPE, "en", VIDEO_TEXT["en"][0])
        self.assertFalse(metrics["raw_box_contract_ok"])

    def test_scoring_is_in_memory_counts_without_raw_text(self):
        text = VIDEO_TEXT["ko"][0]
        upper, lower = _multiline_reference(text)
        sample_metrics = analyze_raw_result(
            SimpleNamespace(txts=[upper, lower],
                            boxes=[rectangle(40, 94), rectangle(125, 180)]),
            SHAPE, "ko", text,
        )
        samples = [
            {"pts": 1.5, "diagnosis": sample_metrics},
            {"pts": 4.25, "diagnosis": {"raw_box_contract_ok": False}},
        ]
        summary = evaluate_samples(samples, "ko")
        self.assertEqual(summary[0]["selected_frames"], 1)
        self.assertEqual(summary[0]["raw_upper_text_exact_samples"], 1)
        self.assertEqual(summary[0]["upper_detected_but_filtered"], 0)
        self.assertEqual(summary[1]["valid_raw_geometry_samples"], 0)
        self.assertNotIn(text, str(summary))


if __name__ == "__main__":
    unittest.main()
