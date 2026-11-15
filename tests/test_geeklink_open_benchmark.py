"""No-network, no-image, stdlib-only safety and scoring checks."""
from __future__ import annotations

import hashlib
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import benchmark_geeklink_open_images as target


class PinnedPublicImageOCRContracts(unittest.TestCase):
    @staticmethod
    def records(n=24):
        return [
            {
                "id": f"{lang}-{flag}-{i}",
                "lang": lang,
                "has_watermark": flag,
                "ground_truth": "日本語." if lang == "ja" else "Hello.",
                "image": f"data/images/{lang}_{i}.jpg",
            }
            for lang in target.LANGUAGES for flag in ("False", "True")
            for i in range(n)
        ]

    def test_first_full_frame_ocr_capture_no_duplicate_inference(self):
        class Dummy:
            version = "pinned"
            def __init__(self):
                self.called = 0
            def __call__(self, value):
                self.called += 1
                return {"sample": value, "call": self.called}
        inner = Dummy()
        captured = target.FirstCallCapture(inner)
        a = captured("full-frame")
        b = captured("targeted-crop")
        self.assertEqual(captured.first, a)
        self.assertEqual(captured.calls, 2)
        self.assertEqual(inner.called, 2)
        self.assertEqual(captured.version, "pinned")
        captured.reset()
        self.assertIsNone(captured.first)
        c = captured("next-frame")
        self.assertEqual(captured.first, c)
        self.assertEqual(captured.calls, 1)
        self.assertEqual(inner.called, 3)

    def test_public_source_commit_pinned(self):
        self.assertEqual(
            target.DATASET_COMMIT,
            "baa9ccbb7ffad36d56cc0ea115256a09bdf427f4",
        )
        self.assertEqual(target.EXPECTED_MANIFEST_ROWS, 1140)

    def test_four_languages_even_clean_and_watermarked(self):
        selected = target.select_stratified(self.records(), 12)
        self.assertEqual(len(selected), 96)
        self.assertEqual(len({r["id"] for r in selected}), 96)
        self.assertEqual(selected, target.select_stratified(self.records(), 12))

    def test_strata_membership_stable_independent_of_manifest_order(self):
        rows = self.records()
        a = target.select_stratified(rows, 12)
        b = target.select_stratified(list(reversed(rows)), 12)
        self.assertEqual([x["id"] for x in a], [x["id"] for x in b])

    def test_complete_four_language_population_is_not_cherry_picked(self):
        quotas = {"ko": 191, "en": 191, "ja": 191, "zh": 188}
        all_rows = [
            {"id": f"{lang}-{i}", "lang": lang,
             "has_watermark": "True" if i < 35 else "False"}
            for lang, quota in quotas.items() for i in range(quota)
        ]
        complete = target.select_all_four_languages(all_rows)
        self.assertEqual(len(complete), 761)
        self.assertEqual(complete, all_rows)
        with self.assertRaises(ValueError):
            target.select_all_four_languages(all_rows[:-1])

    def test_sample_budget_and_missing_strata_fail_closed(self):
        for invalid in (0, -1, 21, True, 1.0):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                target.select_stratified(self.records(), invalid)
        with self.assertRaises(ValueError):
            target.select_stratified(self.records(5), 12)

    def test_raw_character_errors_preserve_punctuation_and_wrapping(self):
        self.assertEqual(target.char_distance("Hallo?", "Hallo!"), (1, 6))
        self.assertEqual(target.char_distance("A\nB", "AB"), (1, 3))
        self.assertEqual(target.char_distance("一二三", "一二三"), (0, 3))

    def test_strict_path_traversal_and_external_links_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d).resolve()
            folder = root / "data" / "images"
            folder.mkdir(parents=True)
            image = folder / "sample.jpg"
            image.write_bytes(b"safe")
            self.assertEqual(target.safe_input(
                root, "data/images/sample.jpg", 100, image=True,
            ), image.resolve())
            for path in ("../sample.jpg", "data/images/../sample.jpg",
                         "C:/data/images/sample.jpg", "data/images/a.png",
                         "/data/images/sample.jpg"):
                with self.subTest(path=path), self.assertRaises(ValueError):
                    target.safe_input(root, path, 100, image=True)
            with self.assertRaises(ValueError):
                target.safe_input(root, "data/images/sample.jpg", 2, image=True)

    def test_published_baseline_requires_all_predictions(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            (root / "baselines").mkdir()
            (root / "external_baselines").mkdir()
            (root / "baselines" / "geeklink.csv").write_text(
                "id,prediction\none,ok\n", encoding="utf8",
            )
            (root / "external_baselines" / "paddleocr_small.csv").write_text(
                "id,prediction\none,ok\n", encoding="utf8",
            )
            good = [{"id": "one"}]
            self.assertEqual(len(target.load_published_predictions(root.resolve(), good)), 2)
            with self.assertRaises(ValueError):
                target.load_published_predictions(root.resolve(), [{"id": "missing"}])

    def test_same_call_filter_audit_counts_better_worse_tied_and_dropped(self):
        selected = target.select_stratified(self.records(), 2)
        truth = {r["id"]: r["ground_truth"] for r in selected}
        raw = dict(truth)
        filtered = dict(truth)
        ko_clean = [r for r in selected if r["lang"] == "ko"
                    and r["has_watermark"] == "False"]
        self.assertEqual(len(ko_clean), 2)
        raw[ko_clean[0]["id"]] = "unrelated background watermark"
        filtered[ko_clean[1]["id"]] = ""
        out = target.summarize(selected, {
            "same_inference_raw_ocr": raw, "hse_real_ocr": filtered,
        })
        pair = next(r["same_call_raw_vs_hse_filtered"] for r in out
                    if r["language"] == "ko" and not r["watermark"])
        self.assertEqual(pair["filtered_better_samples"], 1)
        self.assertEqual(pair["raw_better_samples"], 1)
        self.assertEqual(pair["equal_error_samples"], 0)
        self.assertEqual(pair["filtered_empty_while_raw_nonempty"], 1)
        self.assertEqual(pair["raw_empty_while_filtered_nonempty"], 0)
        self.assertGreater(pair["total_character_edit_errors_raw"], 0)
        self.assertGreater(pair["total_character_edit_errors_filtered"], 0)

    def test_without_raw_capture_pair_is_explicitly_unavailable(self):
        selected = target.select_stratified(self.records(), 1)
        truth = {r["id"]: r["ground_truth"] for r in selected}
        out = target.summarize(selected, {"hse_real_ocr": truth})
        self.assertTrue(all(r["same_call_raw_vs_hse_filtered"] is None for r in out))

    def test_research_bottom_two_paired_gains_and_losses_are_both_visible(self):
        selected = target.select_stratified(self.records(), 2)
        original = {row["id"]: row["ground_truth"] for row in selected}
        candidate = dict(original)
        ko_clean = [r for r in selected if r["lang"] == "ko"
                    and r["has_watermark"] == "False"]
        self.assertEqual(len(ko_clean), 2)
        first, second = ko_clean
        first["ground_truth"] = "X\nHello."
        original[first["id"]] = "EXTRA\nX\nHello."
        candidate[first["id"]] = "X\nHello."
        second["ground_truth"] = "T\nU\nV"
        original[second["id"]] = "T\nU\nV"
        candidate[second["id"]] = "U\nV"
        groups = target.summarize(selected, {
            "hse_real_ocr": original,
            "hse_bottom_two_research": candidate,
        })
        pair = next(g["research_bottom_two_vs_unmodified"] for g in groups
                    if g["language"] == "ko" and not g["watermark"])
        self.assertTrue(pair["same_model_inference"])
        self.assertEqual(pair["candidate_improved_images"], 1)
        self.assertEqual(pair["candidate_worsened_images"], 1)
        self.assertEqual(pair["equal_character_edit_images"], 0)
        self.assertEqual(pair["rows_removed_by_candidate"], 2)
        self.assertTrue(all(
            g["research_bottom_two_vs_unmodified"] is not None for g in groups
        ))

    def test_research_row_selector_is_off_unless_requested(self):
        selected = target.select_stratified(self.records(), 1)
        truth = {r["id"]: r["ground_truth"] for r in selected}
        scored = target.summarize(selected, {"hse_real_ocr": truth})
        self.assertTrue(all(
            g["research_bottom_two_vs_unmodified"] is None for g in scored
        ))

    def test_group_scoring_exposes_watermark_regression(self):
        selected = target.select_stratified(self.records(), 2)
        baseline = {row["id"]: row["ground_truth"] for row in selected}
        bad = dict(baseline)
        for row in selected:
            if row["lang"] == "ko" and row["has_watermark"] == "True":
                bad[row["id"]] = "wrong"
        out = target.summarize(selected, {"hse_real_ocr": bad, "published": baseline})
        ko_wm = [r for r in out if r["language"] == "ko" and r["watermark"]][0]
        ko_clean = [r for r in out if r["language"] == "ko" and not r["watermark"]][0]
        self.assertGreater(ko_wm["scores"]["hse_real_ocr"]["cer"], 0)
        self.assertEqual(ko_clean["scores"]["hse_real_ocr"]["cer"], 0)
        self.assertEqual(ko_wm["scores"]["published"]["cer"], 0)


if __name__ == "__main__":
    unittest.main()
