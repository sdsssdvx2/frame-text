"""Fail-closed quiet-cap AB scoring without contacting OCR/FFmpeg/models."""
from __future__ import annotations

from pathlib import Path
import sys
import unittest
from unittest import mock
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"tools"))
from compare_quiet_segment_caps import summarize_trial, parse_caps, check_blank_control
from verify_multilang_video_pipeline import VIDEO_TEXT
from verify_animated_background_ocr_quality import _multiline_reference


def source(language="ja", cue_count=2, formats=True, timing_error=0.0):
    observations = []
    for index, original in enumerate(VIDEO_TEXT[language]):
        observations.append({
            "present": True,
            "observed_synthetic": "\n".join(_multiline_reference(original)),
            "observed_start_seconds": (1.5 if index == 0 else 4.25),
            "observed_end_seconds": (3.625 if index == 0 else 6.375),
            "start_error": timing_error if index == 1 else 0.0,
            "end_error": 0.0,
        })
    return {
        "cues_detected": cue_count,
        "matched_srt_ass": formats,
        "reference_comparison": observations,
        "ocr_calls": 38,
        "ocr_retry_calls": 18,
        "elapsed_seconds": 40.5,
    }


class QuietCapScoringTests(unittest.TestCase):
    def test_opt_in_cap_argument_bounds_and_duplicate_rejection(self):
        self.assertEqual(parse_caps("none,0.625,0.75"), (None, 0.625, 0.75))
        for invalid in ("none,none", "1.0,1", "0.49", "30.1",
                        "nan", "inf", "", "none,", "none,foo",
                        "none,0.5,0.75,1,1.25,1.5"):
            with self.subTest(setting=invalid), self.assertRaises(ValueError):
                parse_caps(invalid)

    def test_no_caption_control_expected_empty_error_is_not_a_failure(self):
        with tempfile.TemporaryDirectory() as td, mock.patch(
            "compare_quiet_segment_caps.check_language",
            side_effect=RuntimeError("인식된 자막이 없습니다. 자막 없음"),
        ):
            path = Path(td)
            result = check_blank_control(path, "zh", path / "blank.mp4",
                                         path / "models", path / "manifest.json",
                                         0.625)
            self.assertEqual(result, {"executed": True, "false_cues": 0})

    def test_no_caption_control_detects_real_ghost_cues(self):
        with tempfile.TemporaryDirectory() as td, mock.patch(
            "compare_quiet_segment_caps.check_language",
            return_value={"cues_detected": 2},
        ):
            path = Path(td)
            result = check_blank_control(path, "ja", path / "blank.mp4",
                                         path / "models", path / "manifest.json",
                                         0.75)
            self.assertEqual(result["false_cues"], 2)

    def test_no_caption_control_must_not_mask_other_runtime_failures(self):
        with tempfile.TemporaryDirectory() as td, mock.patch(
            "compare_quiet_segment_caps.check_language",
            side_effect=RuntimeError("FFmpeg decode unexpectedly failed"),
        ):
            path = Path(td)
            with self.assertRaisesRegex(RuntimeError, "FFmpeg"):
                check_blank_control(path, "ja", path / "blank.mp4",
                                    path / "models", path / "manifest.json",
                                    0.75)

    def test_empty_control_with_partial_output_is_not_clean(self):
        with tempfile.TemporaryDirectory() as td, mock.patch(
            "compare_quiet_segment_caps.check_language",
            side_effect=RuntimeError("인식된 자막이 없습니다. 없음"),
        ):
            path = Path(td)
            (path / "ja.srt").write_text("unexpected")
            with self.assertRaisesRegex(RuntimeError, "partial output"):
                check_blank_control(path, "ja", path / "blank.mp4",
                                    path / "models", path / "manifest.json",
                                    0.75)

    def test_valid_source_cues_pass_independently_of_cap(self):
        for cap in (None, 1.0, 1.25):
            with self.subTest(cap=cap):
                result = summarize_trial(source(), "ja", cap)
                self.assertEqual(result["joint_exact_cues"], 2)
                self.assertTrue(result["all_cues_complete"])
                self.assertTrue(result["all_cues_strict_visible"])
                self.assertEqual(result["strict_visible_joint_cues"], 2)
                self.assertEqual(result["strict_visible_tight_250ms_cues"], 2)
                self.assertTrue(result["strict_visible_tight_250ms_complete"])
                self.assertEqual(result["quiet_cap_seconds"], cap)
                self.assertEqual(len(result["reference_cue_edges"]), 2)

    def test_punctuation_loss_is_not_strictly_correct(self):
        trial = source("en")
        trial["reference_comparison"][0]["observed_synthetic"] = (
            trial["reference_comparison"][0]["observed_synthetic"]
            .replace("?", "")
        )
        result = summarize_trial(trial, "en", 0.625)
        self.assertEqual(result["joint_exact_cues"], 2)
        self.assertEqual(result["strict_visible_joint_cues"], 1)
        self.assertEqual(result["strict_visible_tight_250ms_cues"], 1)
        self.assertFalse(result["all_cues_strict_visible"])

    def test_tight_250ms_quality_is_independent_of_legacy_time_gate(self):
        trial = source("ja")
        trial["reference_comparison"][0]["start_error"] = 0.625
        result = summarize_trial(trial, "ja", 0.75)
        self.assertEqual(result["joint_exact_cues"], 2)
        self.assertEqual(result["strict_visible_joint_cues"], 2)
        self.assertEqual(result["strict_visible_tight_250ms_cues"], 1)
        self.assertFalse(result["strict_visible_tight_250ms_complete"])

    def test_extra_or_missing_cues_do_not_claim_valid_export(self):
        for count in (1, 3):
            with self.subTest(count=count):
                result = summarize_trial(source(cue_count=count), "ja", 1.0)
                self.assertEqual(result["joint_exact_cues"], 0)
                self.assertFalse(result["valid_export_shape"])

    def test_mismatched_srt_and_ass_are_not_reference_pass(self):
        result = summarize_trial(source(formats=False), "ja", 1.25)
        self.assertEqual(result["joint_exact_cues"], 0)
        self.assertFalse(result["all_cues_complete"])

    def test_bad_timing_rejected_even_with_correct_two_line_text(self):
        result = summarize_trial(source(timing_error=1.875), "ja", None)
        self.assertEqual(result["joint_exact_cues"], 1)
        self.assertFalse(result["all_cues_complete"])
        self.assertEqual(
            result["reference_cue_edges"][1]["start_error_seconds"], 1.875,
        )

    def test_report_does_not_include_synthetic_source_dialogue(self):
        result = summarize_trial(source("zh"), "zh", 1.0)
        for truth in VIDEO_TEXT["zh"]:
            self.assertNotIn(truth, repr(result))


if __name__ == "__main__":
    unittest.main()
