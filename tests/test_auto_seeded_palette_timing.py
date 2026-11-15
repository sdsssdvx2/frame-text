"""Source-only controls for the unsupervised OCR-anchor timing experiment."""
from pathlib import Path
import sys
import unittest
from unittest import mock

import numpy as np
from hardsub_extractor.models import Frame
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import compare_auto_seeded_palette_timing as study
from verify_multilang_video_pipeline import VIDEO_TEXT
from verify_animated_background_ocr_quality import _multiline_reference


def frames():
    return [Frame(i * 0.125, np.zeros((4, 6, 3), dtype=np.uint8))
            for i in range(57)]


class AutomaticSeedTimingStudyTests(unittest.TestCase):
    def test_contiguous_group_allows_one_missing_video_frame(self):
        groups = study.group_timestamps(
            [1.5, 1.625, 1.875, 2.0, 3.25, 3.375], 0.125
        )
        self.assertEqual(groups, [[1.5, 1.625, 1.875, 2.0],
                                  [3.25, 3.375]])

    def test_reference_not_required_to_form_candidate(self):
        actual, blank = frames(), frames()
        audit = study.InMemorySyntheticSegmentAudit()
        script = "\n".join(_multiline_reference(VIDEO_TEXT["en"][0]))
        audit.segments = [(
            1.5, 3.625,
            ((2.375, script, {"detections": [{"cy": 0.2}]}),),
            script,
        )]
        candidate_times = [i * 0.125 for i in range(12, 29)]
        with mock.patch.object(study, "build_palette_anchor",
                               return_value=object()) as anchor, mock.patch.object(
            study, "presence", side_effect=[candidate_times, []],
        ) as detector:
            proposals, report = study.automatic_candidates(
                actual, blank, audit,
            )
        self.assertEqual(len(proposals), 1)
        self.assertAlmostEqual(proposals[0].start, 1.5)
        self.assertAlmostEqual(proposals[0].end, 3.625)
        self.assertEqual(report["usable_palette_anchors"], 1)
        self.assertEqual(report["blank_control_false_frames"], 0)
        self.assertEqual(anchor.call_count, 1)
        self.assertEqual(detector.call_count, 2)

    def test_single_row_without_two_line_ocr_evidence_never_seeded(self):
        actual, blank = frames(), frames()
        audit = study.InMemorySyntheticSegmentAudit()
        audit.segments = [(
            1.5, 3.625,
            ((2.0, "Only bottom line", {"detections": [{"cy": 0.55}]}),),
            "Only bottom line",
        )]
        with mock.patch.object(study, "build_palette_anchor") as anchor:
            proposed, stats = study.automatic_candidates(actual, blank, audit)
        self.assertEqual(proposed, [])
        self.assertEqual(stats["ocr_selected_two_line_samples"], 0)
        anchor.assert_not_called()

    def test_same_text_proposals_deduplicated_without_ground_truth(self):
        proposals = [
            study.ProposedInterval(1.5, 3.625, 2.0, "First\nSecond", 15, 0),
            study.ProposedInterval(1.625, 3.625, 2.375, "First\nSecond", 12, 0),
            study.ProposedInterval(4.25, 6.375, 5.0, "Other\nSecond", 17, 0),
        ]
        chosen = study.deduplicate(proposals)
        self.assertEqual(len(chosen), 2)
        self.assertEqual([x.start for x in chosen], [1.5, 4.25])

    def test_majority_of_independently_observed_ocr_lines_beats_early_misread(self):
        # A single wrong early OCR image once won merely by appearing first.
        readings = [
            study.ProposedInterval(
                1.5, 3.625, 1.75, "Mistakn\ncaption", 17, 0,
            ),
            study.ProposedInterval(
                1.5, 3.625, 2.0, "Correct\ncaption", 16, 0,
            ),
            study.ProposedInterval(
                1.5, 3.625, 2.375, "Correct\ncaption", 17, 0,
            ),
            study.ProposedInterval(
                1.5, 3.625, 2.75, "Correct\ncaption", 16, 0,
            ),
        ]
        chosen = study.deduplicate(readings)
        self.assertEqual(len(chosen), 1)
        self.assertEqual(chosen[0].text, "Correct\ncaption")
        self.assertIn(chosen[0], readings)

    def test_repeated_same_text_in_nonoverlapping_intervals_is_kept_twice(self):
        first = study.ProposedInterval(
            1.5, 3.625, 2.0, "Again\nplease", 17, 0,
        )
        second = study.ProposedInterval(
            4.25, 6.375, 5.0, "Again\nplease", 17, 0,
        )
        selected = study.deduplicate([second, first])
        self.assertEqual(selected, [first, second])

    def test_empty_or_degenerate_proposals_cannot_be_voted_into_a_cue(self):
        invalid = [
            study.ProposedInterval(1.5, 1.5, 1.5, "A\nB", 1, 0),
            study.ProposedInterval(2.0, 3.0, 2.5, "", 4, 0),
        ]
        self.assertEqual(study.deduplicate(invalid), [])

    def test_scoring_rejects_partial_words_or_false_blank_frames(self):
        correct = "\n".join(_multiline_reference(VIDEO_TEXT["ko"][0]))
        correct_proposal = study.ProposedInterval(
            1.5, 3.625, 2.125, correct, 17, 0,
        )
        report = study.evaluate([correct_proposal], "ko")
        self.assertTrue(report[0]["strict"])
        self.assertFalse(report[1]["matched"])
        wrong = study.ProposedInterval(
            1.5, 3.625, 2.125, correct + "X", 17, 0,
        )
        self.assertFalse(study.evaluate([wrong], "ko")[0]["strict"])
        contaminated = study.ProposedInterval(
            1.5, 3.625, 2.125, correct, 17, 5,
        )
        self.assertFalse(study.evaluate([contaminated], "ko")[0]["strict"])

    def test_defensive_time_cadence_rejected(self):
        with self.assertRaises(ValueError):
            study.group_timestamps([1, 2], 0.0)


if __name__ == "__main__":
    unittest.main()
