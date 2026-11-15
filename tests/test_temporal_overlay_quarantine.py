"""Stdlib-only regression controls for safe research-only overlay quarantine."""
import math
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from compare_external_engine_exports import Cue
from study_temporal_overlay_quarantine import (
    LineTrace, anonymous_support_summary, collect_lines,
    find_candidates, fold_whitespace, quarantine,
)


class TemporalOverlayQuarantineTests(unittest.TestCase):
    @staticmethod
    def tracks(*, watermark=True, changing=True, moving=False, duration=7.0):
        samples = []
        for i in range(29):
            pts = i * duration / 28
            if watermark:
                samples.append(LineTrace(
                    pts, "방송 안내", 0.36 + (0.20 if moving and i > 14 else 0),
                    0.76,
                ))
            if changing:
                if 1.5 <= pts <= 3.5:
                    samples.append(LineTrace(pts, "첫 번째 대사", 0.5, 0.3))
                if 4.3 <= pts <= 6.3:
                    samples.append(LineTrace(pts, "두 번째 대사", 0.5, 0.3))
        return samples

    def test_persistent_overlay_with_two_temporally_separate_dialogues(self):
        tracks = self.tracks()
        self.assertEqual(find_candidates(tracks, 7.0), {"방송 안내"})

    def test_clean_dialogue_does_not_become_watermark(self):
        self.assertEqual(find_candidates(self.tracks(watermark=False), 7.0), set())

    def test_one_long_unchanged_real_dialogue_is_not_removed(self):
        self.assertEqual(find_candidates(self.tracks(changing=False), 7.0), set())

    def test_off_center_moving_overlay_is_not_automatically_removed(self):
        self.assertEqual(find_candidates(self.tracks(moving=True), 7.0), set())

    def test_one_half_alone_is_insufficient_independent_evidence(self):
        tracks = [
            t for t in self.tracks()
            if t.text == "방송 안내" or t.pts < 3.5
        ]
        self.assertEqual(find_candidates(tracks, 7.0), set())

    def test_genuine_shared_caption_line_during_two_dialogues_is_not_quarantined(self):
        # This is a genuine fixed first dialogue line (NOT a watermark).
        # It remains displayed through two changing lower lines, but is
        # absent in introductory/outro caption-free frames.
        traces = [p for p in self.tracks()
                  if p.text != "방송 안내" or 1.0 <= p.pts <= 6.0]
        self.assertEqual(find_candidates(traces, 7.0), set())

    def test_non_singleton_video_flanks_cannot_prove_watermark(self):
        traces = self.tracks()
        for i, trace in enumerate(traces[:]):
            if trace.text == "방송 안내" and (trace.pts < 0.85 or trace.pts > 6.15):
                traces.append(LineTrace(trace.pts, "화면 설명", 0.51, 0.20))
        # Multiple independent different texts must *not* turn a shared
        # caption into a watermark; at both flanks it is not seen alone.
        self.assertEqual(find_candidates(traces, 7.0), set())

    def test_one_frame_at_each_video_flank_is_insufficient(self):
        traces = [p for p in self.tracks() if p.text != "방송 안내"
                  or (0.85 <= p.pts <= 6.15 or p.pts in (0.0, 7.0))]
        self.assertEqual(find_candidates(traces, 7.0), set())

    def test_three_windows_retains_overlays_with_sparse_flanks(self):
        # The raw 8fps segment sampler can supply only one observation
        # at each end even if a watermark is truly continuous.
        observations = [p for p in self.tracks()
                        if p.text != "방송 안내"
                        or (0.85 <= p.pts <= 6.15 or p.pts in (0.0, 7.0))]
        self.assertEqual(find_candidates(observations, 7.0), set())
        self.assertEqual(find_candidates(
            observations, 7.0, flank_policy="three_windows",
        ), {"방송 안내"})

    def test_three_windows_never_drops_genuine_midvideo_repeated_caption(self):
        traces = [p for p in self.tracks()
                  if p.text != "방송 안내" or 1.0 <= p.pts <= 6.0]
        self.assertEqual(
            find_candidates(traces, 7.0, flank_policy="three_windows"), set(),
        )

    def test_three_windows_requires_middle_gap_evidence(self):
        observations = [p for p in self.tracks()
                        if p.text != "방송 안내"
                        or p.pts < 2.85 or p.pts > 4.20]
        self.assertEqual(
            find_candidates(observations, 7.0, flank_policy="three_windows"),
            set(),
        )
        self.assertEqual(
            find_candidates(observations, 7.0, flank_policy="baseline"),
            {"방송 안내"},
        )

    def test_reject_unrecognized_policy_instead_of_changing_heuristics(self):
        with self.assertRaises(ValueError):
            find_candidates(self.tracks(), 7.0, flank_policy="auto")

    def test_singleton_flanks_support_a_real_stable_overlay(self):
        traces = self.tracks()
        self.assertEqual(find_candidates(traces, 7.0), {"방송 안내"})

    def test_frame_observations_are_not_faked_by_repeated_same_pts(self):
        traces = ([LineTrace(2.0, "방송 안내", 0.36, 0.76)] * 100
                  + [LineTrace(2.0, "대사", 0.5, 0.3)])
        self.assertEqual(find_candidates(traces, 7.0), set())

    def test_only_exact_candidate_line_discarded_without_touching_objects(self):
        cues = [
            Cue(1.25, 3.5, "첫 번째 대사\n방송 안내"),
            Cue(3.6, 4.2, "방송 안내"),
            Cue(4.25, 6.0, "두 번째 대사\n방송 안내"),
        ]
        result, audit = quarantine(cues, {"방송 안내"})
        self.assertEqual(len(result), 2)
        self.assertEqual([x.text for x in result], ["첫 번째 대사", "두 번째 대사"])
        self.assertEqual([(x.start, x.end) for x in result], [(1.25, 3.5), (4.25, 6.0)])
        self.assertEqual([x.text for x in cues], [
            "첫 번째 대사\n방송 안내", "방송 안내", "두 번째 대사\n방송 안내",
        ])
        self.assertEqual((audit["removed_lines"], audit["dropped_all_candidate_cues"]), (3, 1))
        self.assertFalse(audit["files_changed"])

    def test_no_candidate_never_changes_any_cue_or_text(self):
        cues = [Cue(0.0, 7.0, "an actual long subtitle")]
        output, meta = quarantine(cues, set())
        self.assertEqual(output, cues)
        self.assertTrue(meta["unchanged_if_no_candidate"])

    def test_invalid_duration_is_fail_closed(self):
        for value in (float("nan"), float("inf"), 0, -1, 10000, "7"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                find_candidates([], value)

    def test_extract_actual_metadata_grouped_by_line_and_video_pts(self):
        records = [
            (0.0, 1.0, (
                (0.5, "한국어 대사\n방송 안내", {
                    "detections": [
                        {"line": 0, "cx": 0.5, "cy": 0.3},
                        {"line": 1, "cx": 0.35, "cy": 0.76},
                    ]
                }),
                (0.5, "한국어 대사\n방송 안내", {
                    "detections": [{"line": 1, "cx": 0.35, "cy": 0.76}]
                }),
            ), "대사")
        ]
        audit = type("Audit", (), {"segments": records})()
        lines = collect_lines(audit)
        self.assertEqual(len(lines), 2)
        self.assertEqual({x.text for x in lines}, {"한국어 대사", "방송 안내"})
        self.assertEqual([x.pts for x in lines], [0.5, 0.5])

    def test_malformed_segment_rejected_not_partially_silently_scored(self):
        audit = type("Audit", (), {"segments": [("bad",)]})()
        with self.assertRaises(ValueError):
            collect_lines(audit)


    def test_folded_whitespace_does_not_alter_script_or_punctuation(self):
        self.assertEqual(fold_whitespace("방송  안내"), fold_whitespace("방송안내"))
        self.assertEqual(fold_whitespace(" 방송\t안내 "), "방송안내")
        self.assertNotEqual(fold_whitespace("방송!안내"), fold_whitespace("방송안내"))

    def test_folded_variant_recovers_spacing_variation_across_time_halves(self):
        original = [
            LineTrace(x.pts,
                      ("방송 안내" if x.pts < 3.5 else "방송안내")
                      if x.text == "방송 안내" else x.text,
                      x.cx, x.cy)
            for x in self.tracks()
        ]
        self.assertEqual(find_candidates(original, 7.0), set())
        folded = [LineTrace(x.pts, fold_whitespace(x.text), x.cx, x.cy)
                  for x in original]
        self.assertEqual(find_candidates(folded, 7.0), {"방송안내"})
        outputs, status = quarantine([
            Cue(1.0, 2.0, "첫 번째 대사\n방송 안내"),
            Cue(3.0, 4.0, "방송안내"),
        ], {"방송안내"}, normalizer=fold_whitespace)
        self.assertEqual(len(outputs), 1)
        self.assertEqual(outputs[0].text, "첫 번째 대사")
        self.assertEqual(status["removed_lines"], 2)

    def test_support_diagnostics_never_contain_ocr_surface_or_hash(self):
        summary = anonymous_support_summary(self.tracks(), 7.0)
        self.assertGreater(summary["distinct_text_tracks"], 1)
        self.assertNotIn("방송", str(summary))
        self.assertLessEqual(len(summary["top_support_without_text"]), 3)
        top = summary["top_support_without_text"][0]
        self.assertGreaterEqual(top["independent_intro_samples"], 1)
        self.assertGreaterEqual(top["independent_middle_samples"], 1)
        self.assertGreaterEqual(top["independent_outro_samples"], 1)


if __name__ == "__main__":
    unittest.main()
