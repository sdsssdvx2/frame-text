"""Offline no-download controls for the official RapidVideOCR PyPI candidate."""
from __future__ import annotations

from pathlib import Path
import importlib.util
import sys
import tempfile
import unittest
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import compare_official_rapid_videocr as rv
from compare_external_engine_exports import Cue

class OfficialRapidVideoOCRContracts(unittest.TestCase):
    def test_pypi_dist_version_exact(self):
        self.assertEqual(rv.PACKAGE, "rapid-videocr")
        self.assertEqual(rv.VERSION, "3.1.1")

    def test_pypi_official_wheel_digest_fixed(self):
        self.assertEqual(
            rv.OFFICIAL_PYPI_WHEEL_SHA256,
            "5ccfa46f9911eb02c21a4e60946e2fd4d1c282ddbdfcd4f8a780e867816e56b2",
        )

    def test_only_japanese_and_chinese_initial_comparison(self):
        self.assertEqual(rv.LANGUAGES, ("ja", "zh"))

    def test_explicit_full_product_language_matrix(self):
        from rapidocr import LangRec
        self.assertEqual(
            rv.parse_research_languages("ko,en,ja,zh"), rv.EXTENDED_LANGUAGES,
        )
        self.assertEqual(rv.official_rec_language("ko"), LangRec.KOREAN)
        self.assertEqual(rv.official_rec_language("en"), LangRec.EN)
        self.assertEqual(rv.official_rec_language("ja"), LangRec.CH)
        self.assertEqual(rv.official_rec_language("zh"), LangRec.CH)

    def test_language_selection_fails_closed(self):
        for invalid in ("", "ko,ko", "ja,ru", "zh,", " KO ", "ja,,zh"):
            with self.subTest(invalid=invalid):
                with self.assertRaises(ValueError):
                    rv.parse_research_languages(invalid)
        with self.assertRaises(ValueError):
            rv.official_rec_language("ru")

    def test_identical_reference_wrap_applied(self):
        from verify_animated_background_ocr_quality import _multiline_reference
        from verify_multilang_video_pipeline import VIDEO_TEXT
        y = "\n".join(_multiline_reference(VIDEO_TEXT["ja"][0]))
        self.assertEqual(len(y.splitlines()), 2)

    def test_reject_no_output_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with self.assertRaises(RuntimeError):
                rv.grade_srt_ass(
                    root / "none.srt", root / "none.ass",
                    [Cue(1, 2, "字")], 0,
                )

    def test_valid_first_party_srt_ass_measured(self):
        from hardsub_extractor.writers import write_subtitle_pair_exclusive
        from hardsub_extractor.models import Cue as HSECue
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "case"
            srt, ass = write_subtitle_pair_exclusive(
                [HSECue(1.5, 3.625, "字幕\n第二行")], p,
            )
            scores = rv.grade_srt_ass(
                srt, ass, [Cue(1.5, 3.625, "字幕\n第二行")], 0,
            )
            self.assertTrue(scores["strict_srt_valid"])
            self.assertTrue(scores["valid_ass_matching_srt"])
            self.assertEqual(
                scores["scores"]["strict_both_edges_250ms_and_text_wrap"], 1,
            )

    def test_invalid_srt_no_false_pass(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root/"one.srt").write_text("invalid SRT", encoding="utf8")
            (root/"one.ass").write_text("invalid ASS", encoding="utf8")
            out = rv.grade_srt_ass(
                root/"one.srt", root/"one.ass", [Cue(1, 2, "x")], 0,
            )
            self.assertFalse(out["strict_srt_valid"])
            self.assertFalse(out["valid_ass_matching_srt"])
            self.assertIsNone(out["scores"])

    def test_blank_native_control_is_not_ignored(self):
        from hardsub_extractor.writers import write_subtitle_pair_exclusive
        from hardsub_extractor.models import Cue as HSECue
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "case"
            srt, ass = write_subtitle_pair_exclusive(
                [HSECue(1.5, 3.625, "字幕\n第二行")], p,
            )
            scores = rv.grade_srt_ass(
                srt, ass, [Cue(1.5, 3.625, "字幕\n第二行")], 1,
            )
            self.assertFalse(scores["scores"]["all_reference_cues_exact_both_edges_250ms"])
            self.assertEqual(scores["scores"]["blank_video_false_cues"], 1)

    def test_no_unsafe_raw_subprocess_launcher_used(self):
        from inspect import getsource
        s = getsource(rv.main)
        self.assertNotIn("VideoSubFinderInput", s)
        self.assertNotIn("subprocess.run", s)
        self.assertIn("RapidVideOCR(params)", s)

    def test_package_scratch_output_is_unique_before_call(self):
        from inspect import getsource
        s = getsource(rv.main)
        self.assertIn("output.mkdir()", s)
        self.assertIn("TemporaryDirectory", s)

    @unittest.skipUnless(
        importlib.util.find_spec("rapid_videocr") is not None,
        "official pinned rapid-videocr lives only in isolated upstream research CI",
    )
    def test_actual_upstream_multiline_ass_drops_second_row(self):
        """Pin the actual v3.1.1 exporter behavior; do not pretend broken ASS is valid.

        The source SRT is correct, but the official ASS dialogue builder embeds
        a literal physical newline instead of ASS's escaped \\N delimiter.
        This test uses the actual installed upstream exporter, without OCR.
        """
        from rapid_videocr.export import SrtExportStrategy, AssExportStrategy
        from rapid_videocr.ocr_processor import OCRProcessor

        samples = [[
            0, "00:00:01,500 --> 00:00:02,625",
            "上段\\n下段".replace("\\n", "\n"),
            "0:00:01.50,0:00:02.62",
        ]]
        srt_rows = OCRProcessor._generate_srt_results(samples)
        ass_rows = OCRProcessor._generate_ass_results(samples)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            SrtExportStrategy().export(root, "official", srt_rows, ass_rows, [])
            AssExportStrategy().export(root, "official", srt_rows, ass_rows, [])
            inspected = rv.grade_srt_ass(
                root / "official.srt", root / "official.ass",
                [Cue(1.5, 2.625, "上段\n下段")], 0,
            )
            self.assertTrue(inspected["strict_srt_valid"])
            self.assertEqual(
                inspected["scores"]["strict_both_edges_250ms_and_text_wrap"], 1,
            )
            # The second physical text row is dropped by ASS dialogue parsing.
            self.assertFalse(inspected["valid_ass_matching_srt"])

    @unittest.skipUnless(
        importlib.util.find_spec("rapid_videocr") is not None,
        "official pinned rapid-videocr lives only in isolated upstream research CI",
    )
    def test_verified_upstream_srt_can_safely_reexport_lossless_ass(self):
        """Read original SRT, write to NEW paths, preserve exact wrap and cues."""
        from rapid_videocr.export import SrtExportStrategy, AssExportStrategy
        from rapid_videocr.ocr_processor import OCRProcessor
        from hardsub_extractor.subtitle_formats import parse_subtitle_bytes
        from hardsub_extractor.writers import write_subtitle_pair_exclusive

        rows = [[
            0, "00:00:01,500 --> 00:00:02,625",
            "上段\n下段", "0:00:01.50,0:00:02.62",
        ]]
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            SrtExportStrategy().export(
                root, "official", OCRProcessor._generate_srt_results(rows),
                OCRProcessor._generate_ass_results(rows), [],
            )
            AssExportStrategy().export(
                root, "official", OCRProcessor._generate_srt_results(rows),
                OCRProcessor._generate_ass_results(rows), [],
            )
            original_srt = root / "official.srt"
            original_ass = root / "official.ass"
            before = (original_srt.read_bytes(), original_ass.read_bytes())
            _, valid_cues = parse_subtitle_bytes(
                original_srt.read_bytes(), source_url="official.srt",
            )
            candidate = root / "separate-safe-output"
            repaired_srt, repaired_ass = write_subtitle_pair_exclusive(
                valid_cues, candidate,
            )
            checked = rv.grade_srt_ass(
                repaired_srt, repaired_ass, [Cue(1.5, 2.625, "上段\n下段")], 0,
            )
            self.assertTrue(checked["strict_srt_valid"])
            self.assertTrue(checked["valid_ass_matching_srt"])
            self.assertEqual(
                checked["scores"]["strict_both_edges_250ms_and_text_wrap"], 1,
            )
            self.assertIn(b"\\N", repaired_ass.read_bytes())
            self.assertEqual(
                (original_srt.read_bytes(), original_ass.read_bytes()), before,
            )
            with self.assertRaises(FileExistsError):
                write_subtitle_pair_exclusive(valid_cues, candidate)
            self.assertEqual(
                (original_srt.read_bytes(), original_ass.read_bytes()), before,
            )


if __name__ == "__main__":
    unittest.main()
