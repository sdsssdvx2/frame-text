"""A valid OCR output must not count matched indices if it emitted extra cues."""
from __future__ import annotations

from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from compare_multiline_sampling_budgets import summarize_case
from verify_animated_background_ocr_quality import _multiline_reference
from verify_multilang_video_pipeline import VIDEO_TEXT


def sample_source(language="en", *, cue_count=2, formats_ok=True, first_wrong=False):
    observations = []
    for idx, truth in enumerate(VIDEO_TEXT[language]):
        text = "\n".join(_multiline_reference(truth))
        if first_wrong and idx == 0:
            text = text + "X"
        observations.append({
            "present": True, "observed_synthetic": text,
            "cer": 0.04 if first_wrong and idx == 0 else 0.0,
            "start_error": 0.0, "end_error": 0.0,
        })
    return {
        "reference_comparison": observations,
        "cues_detected": cue_count,
        "matched_srt_ass": formats_ok,
        "ocr_calls": 40,
        "ocr_retry_calls": 15,
        "elapsed_seconds": 45.0,
    }


class SampleBudgetMetricGateTests(unittest.TestCase):
    def test_complete_two_cue_exact_export_counts_both(self):
        row = summarize_case(sample_source(), "en", 3)
        self.assertEqual(row["joint_exact_cues"], 2)
        self.assertEqual(row["aligned_reference_joint_matches"], 2)
        self.assertTrue(row["export_shape_valid"])
        self.assertTrue(row["all_complete"])

    def test_three_cue_output_never_gets_false_success_from_first_two(self):
        row = summarize_case(sample_source(cue_count=3, formats_ok=False), "en", 7)
        self.assertEqual(row["aligned_reference_joint_matches"], 2)
        self.assertEqual(row["joint_exact_cues"], 0)
        self.assertFalse(row["export_shape_valid"])
        self.assertFalse(row["all_complete"])

    def test_extra_cue_even_when_formats_flag_is_wrongly_true_fails(self):
        row = summarize_case(sample_source(cue_count=3), "en", 7)
        self.assertEqual(row["joint_exact_cues"], 0)
        self.assertFalse(row["export_shape_valid"])

    def test_exact_two_cues_but_inconsistent_formats_fail(self):
        row = summarize_case(sample_source(formats_ok=False), "en", 5)
        self.assertEqual(row["aligned_reference_joint_matches"], 2)
        self.assertEqual(row["joint_exact_cues"], 0)

    def test_missing_cue_never_counts_reference_matches_as_full_export(self):
        row = summarize_case(sample_source(cue_count=1), "en", 5)
        self.assertEqual(row["joint_exact_cues"], 0)
        self.assertFalse(row["all_complete"])

    def test_valid_export_with_wrong_text_keeps_only_individual_exact_cue(self):
        row = summarize_case(sample_source(first_wrong=True), "en", 3)
        self.assertEqual(row["aligned_reference_joint_matches"], 1)
        self.assertEqual(row["joint_exact_cues"], 1)
        self.assertTrue(row["export_shape_valid"])
        self.assertFalse(row["all_complete"])


if __name__ == "__main__":
    unittest.main()
