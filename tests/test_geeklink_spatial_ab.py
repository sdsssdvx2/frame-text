"""No-network contracts for pinned spatial HSE OCR quality research."""
from __future__ import annotations

import io
from pathlib import Path
import sys
import unittest
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import benchmark_geeklink_spatial_ab as trial


def fixtures():
    rows = []
    guesses = {name: {} for name in trial.ROI_CANDIDATES}
    for language in trial.LANGUAGES:
        for marked in ("False", "True"):
            for movie in ("v01", "v02", "v03"):
                sid = f"{movie}-{language}-{marked}"
                rows.append({
                    "id": sid,
                    "lang": language,
                    "video_id": movie,
                    "has_watermark": marked,
                    "ground_truth": "a b c",
                })
                guesses["production_roi"][sid] = "a b c"
                guesses["center_bottom_32"][sid] = "a b c"
                guesses["center_bottom_26"][sid] = "a b c"
    return rows, guesses


class PinnedSpatialStudyTests(unittest.TestCase):
    def test_crops_are_static_and_inside_video(self):
        self.assertEqual(len(trial.ROI_CANDIDATES), 3)
        self.assertEqual(trial.ROI_CANDIDATES["production_roi"],
                         (0.04, 0.58, 0.92, 0.40))
        for name, (x, y, w, h) in trial.ROI_CANDIDATES.items():
            with self.subTest(name=name):
                self.assertTrue(0 <= x and 0 <= y and x + w <= 1 and y + h <= 1)
                self.assertTrue(w >= 0.5 and h >= 0.20)

    def test_aggregate_zero_cer_is_correct_and_hides_text(self):
        rows, guesses = fixtures()
        scores, harms = trial._group_summary(rows, guesses)
        for name in trial.ROI_CANDIDATES:
            self.assertEqual(scores[name]["all"]["n"], 24)
            self.assertEqual(scores[name]["all"]["cer"], 0.0)
            self.assertEqual(scores[name]["clean"]["n"], 12)
            self.assertEqual(scores[name]["watermark"]["n"], 12)
            self.assertEqual(scores[name]["exact_image_count"], 24)
        for name, groups in harms.items():
            self.assertEqual(groups["all"]["unchanged"], 24)
            self.assertEqual(groups["clean"]["worsened"], 0)
        self.assertNotIn("a b c", str(scores))

    def test_aggregate_tracks_clean_harm_and_watermark_gain(self):
        rows, guesses = fixtures()
        watermark = next(r for r in rows if r["has_watermark"] == "True")["id"]
        clean = next(r for r in rows if r["has_watermark"] == "False")["id"]
        guesses["production_roi"][watermark] = "a b d"
        guesses["center_bottom_32"][clean] = "a b d"
        scores, harms = trial._group_summary(rows, guesses)
        self.assertEqual(harms["center_bottom_32"]["watermark"]["improved"], 1)
        self.assertEqual(harms["center_bottom_32"]["clean"]["worsened"], 1)
        self.assertGreater(scores["production_roi"]["watermark"]["cer"], 0)

    def test_mismatched_ids_and_incomplete_methods_fail_closed(self):
        rows, guesses = fixtures()
        guesses["production_roi"].pop(rows[0]["id"])
        with self.assertRaises(trial.BenchmarkError):
            trial._group_summary(rows, guesses)
        rows, guesses = fixtures()
        del guesses["center_bottom_26"]
        with self.assertRaises(trial.BenchmarkError):
            trial._group_summary(rows, guesses)

    def test_unapproved_network_is_disabled_before_any_source_fetch(self):
        with mock.patch.object(trial, "measure") as measured:
            result = trial.main([
                "--model-dir", "missing", "--model-manifest", "missing",
                "--report", "missing.json",
            ])
        self.assertEqual(result, 2)
        measured.assert_not_called()

    def test_report_does_not_overwrite_user_file(self):
        from tempfile import TemporaryDirectory
        with TemporaryDirectory() as td:
            dest = Path(td) / "prior.json"
            dest.write_bytes(b"leave-alone")
            with mock.patch.object(trial, "measure", return_value={
                "cer": {}, "pairwise_errors_vs_production": {},
            }):
                result = trial.main([
                    "--model-dir", "test", "--model-manifest", "test",
                    "--report", str(dest), "--accept-public-download",
                ])
            self.assertEqual(result, 3)
            self.assertEqual(dest.read_bytes(), b"leave-alone")


if __name__ == "__main__":
    unittest.main()
