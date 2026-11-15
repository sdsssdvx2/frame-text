"""Never accept a visually timed two-line caption when text is incomplete."""
from __future__ import annotations

from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from verify_multiline_streaming_ocr_quality import (score_case, InMemorySyntheticSegmentAudit, summarize_synthetic_evidence)
from verify_multilang_video_pipeline import VIDEO_TEXT
from verify_animated_background_ocr_quality import _multiline_reference


def report(observations):
    return {
        "cues_detected": 2,
        "matched_srt_ass": True,
        "ocr_calls": 15,
        "ocr_retry_calls": 2,
        "reference_comparison": observations,
    }


def observation(text, error=0.0):
    return {
        "present": True, "observed_synthetic": text,
        "cer": error, "start_error": 0.125,
        "end_error": 0.25,
    }


class TwoLineStreamMeasurementTests(unittest.TestCase):
    def test_genuine_two_line_all_languages(self):
        for lang in ("ko", "en", "ja", "zh"):
            with self.subTest(lang=lang):
                rows = [
                    observation("\n".join(_multiline_reference(text)))
                    for text in VIDEO_TEXT[lang]
                ]
                result = score_case(report(rows), lang)
                self.assertTrue(result["all_cues_complete"])
                self.assertTrue(result["all_cues_strict_visible"])
                self.assertEqual(len(result["checks"]), 2)
                self.assertEqual(result["reported_cues"], 2)

    def test_release_text_requires_original_punctuation_and_case(self):
        truth = VIDEO_TEXT["en"]
        samples = [
            observation("\n".join(_multiline_reference(text))) for text in truth
        ]
        valid = score_case(report(samples), "en")
        self.assertTrue(valid["all_cues_strict_visible"])
        self.assertTrue(all(c["strict_joint_acceptable"] for c in valid["checks"]))
        for corrupt in (
            samples[0]["observed_synthetic"].replace("?", ""),
            samples[0]["observed_synthetic"].lower(),
        ):
            with self.subTest(corrupt_repr=repr(corrupt)):
                bad = [dict(x) for x in samples]
                bad[0]["observed_synthetic"] = corrupt
                measured = score_case(report(bad), "en")
                # Historical normalized-only metric misses this damage.
                self.assertTrue(measured["checks"][0]["joint_acceptable"])
                self.assertFalse(measured["checks"][0]["strict_joint_acceptable"])
                self.assertFalse(measured["all_cues_strict_visible"])
                self.assertFalse(measured["checks"][0]["strict_visible_text_match"])

    def test_two_rows_must_preserve_correct_physical_line_wrap(self):
        truth = VIDEO_TEXT["en"]
        samples = [
            observation("\n".join(_multiline_reference(text))) for text in truth
        ]
        first = samples[0]["observed_synthetic"]
        new_text = first.replace("\n", " ", 1)
        words = new_text.split(" ")
        samples[0]["observed_synthetic"] = words[0] + "\n" + " ".join(words[1:])
        result = score_case(report(samples), "en")
        self.assertTrue(result["checks"][0]["joint_acceptable"])
        self.assertFalse(result["checks"][0]["exact_physical_line_wrap"])
        self.assertFalse(result["checks"][0]["strict_joint_acceptable"])
        self.assertFalse(result["all_cues_strict_visible"])

    def test_missing_row_is_not_hidden_by_exact_boundaries(self):
        truth = VIDEO_TEXT["en"]
        samples = [
            observation(_multiline_reference(truth[0])[0], error=0.5),
            observation("\n".join(_multiline_reference(truth[1]))),
        ]
        result = score_case(report(samples), "en")
        self.assertFalse(result["all_cues_complete"])
        self.assertFalse(result["checks"][0]["observed_two_lines"])
        self.assertTrue(result["checks"][0]["timing_acceptable"])

    def test_flattening_both_lines_is_insufficient(self):
        truth = VIDEO_TEXT["ko"]
        samples = [observation(text) for text in truth]
        result = score_case(report(samples), "ko")
        self.assertFalse(result["all_cues_complete"])
        self.assertTrue(all(x["exact_reference_characters"] for x in result["checks"]))
        self.assertTrue(all(not x["observed_two_lines"] for x in result["checks"]))

    def test_bad_text_even_with_two_lines_fails(self):
        truth = VIDEO_TEXT["ja"]
        samples = [
            observation("\n".join(_multiline_reference(truth[0])) + "X", error=0.04),
            observation("\n".join(_multiline_reference(truth[1]))),
        ]
        result = score_case(report(samples), "ja")
        self.assertFalse(result["checks"][0]["joint_acceptable"])
        self.assertTrue(result["checks"][1]["joint_acceptable"])

    def test_missing_cue_and_inconsistent_srt_ass_fail(self):
        truth = VIDEO_TEXT["zh"]
        row = report([observation("\n".join(_multiline_reference(truth[0])))])
        row["cues_detected"] = 1
        row["matched_srt_ass"] = False
        result = score_case(row, "zh")
        self.assertFalse(result["all_cues_complete"])
        self.assertFalse(result["checks"][1]["joint_acceptable"])

    def test_timing_tolerance_enforced_separately(self):
        truth = VIDEO_TEXT["ko"]
        row = report([observation("\n".join(_multiline_reference(t))) for t in truth])
        row["reference_comparison"][1]["start_error"] = 0.875
        result = score_case(row, "ko")
        self.assertFalse(result["all_cues_complete"])
        self.assertFalse(result["checks"][1]["timing_acceptable"])


    def test_malformed_time_errors_never_pass_quality_gate(self):
        # These values can escape naïve max(error_a, error_b) tests:
        # max(0.0, nan) evaluates to 0.0 in Python.
        truth = VIDEO_TEXT["en"]
        source = [
            observation("\n".join(_multiline_reference(t))) for t in truth
        ]
        invalid = (float("nan"), float("inf"), float("-inf"),
                   -0.01, True, False, None, "0")
        for field in ("start_error", "end_error"):
            for value in invalid:
                with self.subTest(field=field, value=repr(value)):
                    samples = [dict(x) for x in source]
                    samples[1][field] = value
                    result = score_case(report(samples), "en")
                    self.assertFalse(result["all_cues_complete"])
                    self.assertFalse(result["checks"][1]["timing_acceptable"])
                    self.assertFalse(result["checks"][1]["joint_acceptable"])

    def test_exact_timing_boundary_is_permitted(self):
        truth = VIDEO_TEXT["ko"]
        samples = [
            observation("\n".join(_multiline_reference(t))) for t in truth
        ]
        samples[0]["start_error"] = 0.65
        samples[0]["end_error"] = 0.0
        result = score_case(report(samples), "ko")
        self.assertTrue(result["all_cues_complete"])


    def test_synthetic_timing_report_preserves_absolute_observed_edges(self):
        truth = VIDEO_TEXT["ja"]
        source = report([
            observation("\n".join(_multiline_reference(t)))
            for t in truth
        ])
        source["reference_comparison"][0]["observed_start_seconds"] = 1.625
        source["reference_comparison"][0]["observed_end_seconds"] = 3.750
        result = score_case(source, "ja")
        self.assertAlmostEqual(
            result["checks"][0]["observed_start_seconds"], 1.625,
        )
        self.assertAlmostEqual(
            result["checks"][0]["observed_end_seconds"], 3.750,
        )
        self.assertAlmostEqual(
            result["checks"][0]["start_error_seconds"], 0.125,
        )
        self.assertAlmostEqual(
            result["checks"][0]["end_error_seconds"], 0.25,
        )
        self.assertNotIn(truth[0], repr(result))

    def test_diagnostic_is_bounded_and_reports_counts_without_text(self):
        from hardsub_extractor.models import OCRObservation, VisualSegment
        audit = InMemorySyntheticSegmentAudit()
        expected = VIDEO_TEXT["ko"][0]
        rows = _multiline_reference(expected)
        full = "\n".join(rows)
        # Only a synthetic OCR transcript enters the bounded in-memory audit.
        obs = [
            OCRObservation(full, 0.99, 2.0, {"detections": [{"cy": 0.25}]}),
            OCRObservation(rows[1], 0.92, 2.125, {"detections": [{"cy": 0.6}]}),
        ]
        audit.record_segment(0, VisualSegment(1.9, 2.5, []), obs, obs[0])
        result = summarize_synthetic_evidence(audit, "ko")[0]
        self.assertEqual(result["sample_count"], 2)
        self.assertEqual(result["both_lines_exact_samples"], 1)
        self.assertEqual(result["recognized_lower_line_samples"], 2)
        self.assertEqual(result["upper_ocr_detection_boxes"], 1)
        self.assertEqual(result["first_both_lines_pts"], 2.0)
        self.assertEqual(
            result["synthetic_segment_windows"],
            [{"start": 1.9, "end": 2.5, "two_line_voted": True}],
        )
        self.assertNotIn(expected, repr(result))


if __name__ == "__main__":
    unittest.main()
