"""Research-only pending-timeout A/B output contracts."""
from __future__ import annotations

from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"tools"))
from compare_pending_timeout_ab import PROFILES, LANGUAGES, assess_trial
from verify_animated_background_ocr_quality import _multiline_reference
from verify_multilang_video_pipeline import VIDEO_TEXT


def make_measurement(lang):
    originals = VIDEO_TEXT[lang]
    return {
        "cues_detected": 2,
        "matched_srt_ass": True,
        "ocr_calls": 36,
        "ocr_retry_calls": 17,
        "elapsed_seconds": 30.0,
        "reference_comparison": [
            {
                "present": True,
                "observed_synthetic": "\n".join(_multiline_reference(text)),
                "start_error": 0.125,
                "end_error": 0.125,
                "observed_start_seconds": (1.5 if i == 0 else 4.25),
                "observed_end_seconds": (3.625 if i == 0 else 6.375),
            }
            for i, text in enumerate(originals)
        ],
    }


class PendingTimeoutABTests(unittest.TestCase):
    def test_profiles_include_true_default_and_distinct_enabled_timeouts(self):
        self.assertEqual(PROFILES[0], ("default", None, None))
        self.assertEqual(len(PROFILES), len(set(PROFILES)))
        self.assertEqual(set(LANGUAGES), {"ja", "zh"})
        self.assertTrue(all(
            0.25 <= timeout <= 5.0 and cap is not None
            for _, cap, timeout in PROFILES[1:]
        ))

    def test_candidate_does_not_use_transcript_to_grade_before_ocr(self):
        row = assess_trial(make_measurement("ja"), "ja",
                           "pending_375ms", 0.625, 0.375)
        self.assertEqual(row["strict_visible_joint_cues"], 2)
        self.assertEqual(row["strict_visible_tight_250ms_cues"], 2)
        self.assertTrue(row["strict_visible_tight_250ms_complete"])
        self.assertEqual(row["pending_timeout_seconds"], 0.375)
        for raw in VIDEO_TEXT["ja"]:
            self.assertNotIn(raw, repr(row))

    def test_bad_time_or_wrong_visible_text_fail_closed(self):
        measurement = make_measurement("zh")
        measurement["reference_comparison"][0]["end_error"] = 1.25
        measurement["reference_comparison"][1]["observed_synthetic"] += "X"
        row = assess_trial(measurement, "zh",
                           "pending_500ms", 0.625, 0.5)
        self.assertEqual(row["strict_visible_joint_cues"], 0)
        self.assertEqual(row["strict_visible_tight_250ms_cues"], 0)
        self.assertFalse(row["all_cues_strict_visible"])


if __name__ == "__main__":
    unittest.main()
