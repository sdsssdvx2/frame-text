"""Offline, no-network contracts for the temporary watermark MP4 study."""
from pathlib import Path
from contextlib import redirect_stdout
import io
import json
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from compare_external_engine_exports import Cue
import verify_temporal_watermark_video_ocr as study


class TemporalWatermarkContracts(unittest.TestCase):
    def test_fixed_population_before_any_ocr(self):
        self.assertEqual(study.LANGUAGES, ("ko", "zh"))
        self.assertEqual(study.CASES, (
            "clean", "persistent_watermark", "watermark_only",
        ))
        self.assertEqual(study.FRAME_COUNT, 57)
        for lang in study.LANGUAGES:
            cues = study.reference_cues(lang)
            self.assertEqual(len(cues), 2)
            self.assertEqual(
                [(c.start, c.end) for c in cues], list(study.EXPECTED_SPANS),
            )
            self.assertTrue(all(len(c.text.splitlines()) == 2 for c in cues))
        with self.assertRaises(ValueError):
            study.reference_cues("unsupported")

    def test_output_reference_is_not_used_as_ocr_input(self):
        # The only call to reference_cues in production study is the grader;
        # actual study.run_hse receives only generated MP4, base, and engine.
        import ast
        tree = ast.parse(Path(study.__file__).read_text(encoding="utf-8"))
        grader = next(n for n in tree.body
                      if isinstance(n, ast.FunctionDef) and n.name == "score_profiles")
        run = next(n for n in tree.body
                   if isinstance(n, ast.FunctionDef) and n.name == "run_hse")
        self.assertIn("reference_cues(", ast.unparse(grader))
        self.assertNotIn("reference_cues(", ast.unparse(run))

    def test_only_known_profiles_accepted_without_encoding(self):
        with self.assertRaises(ValueError):
            study.make_video(Path("uncreated.mp4"), "ko", "unrecognised")
        with self.assertRaises(ValueError):
            study.make_video(Path("uncreated.mp4"), "en", "clean")

    def test_compositor_keeps_input_frames_immutable(self):
        frame = np.full((study.HEIGHT, study.WIDTH, 3), 47, np.uint8)
        copy = frame.copy()
        empty = np.zeros((study.HEIGHT, study.WIDTH, 4), np.uint8)
        self.assertTrue(np.array_equal(study.blend(frame, empty), frame))
        layer = empty.copy()
        layer[610:620, 320:340] = [125, 135, 145, 255]
        result = study.blend(frame, layer)
        self.assertTrue(np.array_equal(frame, copy))
        self.assertTrue(np.array_equal(result[400:600], frame[400:600]))
        self.assertTrue(np.array_equal(result[650:], frame[650:]))
        self.assertTrue(np.array_equal(result[612, 323], [125, 135, 145]))
        with self.assertRaises(ValueError):
            study.blend(frame, empty[:10])

    def test_observed_false_cue_on_blank_control_is_not_hidden(self):
        truth = study.reference_cues("ko")
        graded = study.score_profiles({
            "clean": truth,
            "persistent_watermark": [
                Cue(truth[0].start, truth[0].end, "wrong text"),
                truth[1],
            ],
            "watermark_only": [Cue(0.0, 7.0, "NOT DIALOGUE")],
        }, "ko")
        self.assertEqual(graded["watermark_only"]["false_positive_cues"], 1)
        self.assertFalse(graded["watermark_only"]["no_caption_control_pass"])
        self.assertEqual(graded["clean"]["strict_both_edges_250ms_and_text_wrap"], 2)
        self.assertEqual(
            graded["persistent_watermark"]["strict_both_edges_250ms_and_text_wrap"], 1,
        )
        self.assertEqual(graded["watermark_minus_clean"]["strict_250ms_cues_delta"], -1)

    def test_missing_first_cue_is_not_shifted_to_next(self):
        truth = study.reference_cues("zh")
        graded = study.score_profiles({
            "clean": truth, "persistent_watermark": [truth[1]], "watermark_only": [],
        }, "zh")
        self.assertEqual(graded["persistent_watermark"]["matched_cues"], 1)
        self.assertEqual(graded["persistent_watermark"]["missed_reference_cues"], 1)
        self.assertEqual(graded["watermark_minus_clean"]["missed_reference_cues_delta"], 1)
        self.assertTrue(graded["watermark_only"]["no_caption_control_pass"])

    def test_study_fails_closed_if_any_source_profile_is_missing(self):
        truth = study.reference_cues("ko")
        for partial in ({}, {"clean": truth},
                        {"clean": truth, "watermark_only": []}):
            with self.subTest(count=len(partial)), self.assertRaises(ValueError):
                study.score_profiles(partial, "ko")

    def test_no_cues_only_accepted_with_no_output_files(self):
        with tempfile.TemporaryDirectory() as td:
            base = Path(td) / "new"
            expected = RuntimeError("인식된 자막이 없습니다. 선택한 OCR 언어")
            self.assertTrue(study.is_expected_no_cue(expected, base))
            self.assertFalse(study.is_expected_no_cue(RuntimeError("unexpected"), base))
            base.with_suffix(".srt").write_text("partial", encoding="utf8")
            with self.assertRaises(RuntimeError):
                study.is_expected_no_cue(expected, base)

    def test_unknown_runtime_error_is_never_caught_as_empty_cue(self):
        with tempfile.TemporaryDirectory() as td, mock.patch.object(
            study, "extract", side_effect=RuntimeError("decoder failed"),
        ):
            with self.assertRaisesRegex(RuntimeError, "decoder failed"):
                study.run_hse(Path(td) / "source.mp4", Path(td) / "output", object())

    def test_actual_pipeline_explicit_empty_result_is_measured_as_zero(self):
        with tempfile.TemporaryDirectory() as td, mock.patch.object(
            study, "extract", side_effect=RuntimeError("인식된 자막이 없습니다. 없음"),
        ):
            self.assertEqual(
                study.run_hse(Path(td) / "source.mp4", Path(td) / "output", object()),
                ([], 0),
            )


    def test_main_two_languages_complete_with_fully_mocked_inference(self):
        """A nested candidate-loop variable must never replace completed rows.

        This reproduces the CI #42355269582 false 'complete=false' exit:
        both genuine per-language reports existed but the list variable was
        overwritten by an inner OCR trace iteration.
        """
        class FakeEngine:
            def __init__(self, *_args, **_kwargs):
                self.total_calls = 0
                self.retry_calls = 0

        def fake_video(path, language, profile):
            return {
                "sha256": language + "/" + profile,
                "caption_present": profile != "watermark_only",
            }

        def fake_pipeline(media, _base, _engine, *, diagnostics=None):
            language = media.name.split("-", 1)[0]
            if media.stem.endswith("-watermark_only"):
                return [], 1
            return study.reference_cues(language), 2

        output = io.StringIO()
        with mock.patch.object(study, "RapidOCRAdapter", return_value=object()), \
             mock.patch.object(study, "AdaptiveOCREngine", FakeEngine), \
             mock.patch.object(study, "make_video", side_effect=fake_video), \
             mock.patch.object(study, "run_hse", side_effect=fake_pipeline), \
             mock.patch.object(study, "InMemorySyntheticSegmentAudit"), \
             mock.patch("sys.argv", [
                 "verify_temporal_watermark_video_ocr.py",
                 "--model-dir", "inert", "--languages", "ko,zh",
             ]), redirect_stdout(output):
            code = study.main()

        self.assertEqual(code, 0)
        rows = [
            json.loads(line.partition("HSE_WATERMARK_MP4_CASE ")[2])
            for line in output.getvalue().splitlines()
            if line.startswith("HSE_WATERMARK_MP4_CASE ")
        ]
        summaries = [
            json.loads(line.partition("HSE_WATERMARK_MP4_SUMMARY ")[2])
            for line in output.getvalue().splitlines()
            if line.startswith("HSE_WATERMARK_MP4_SUMMARY ")
        ]
        self.assertEqual([row["language"] for row in rows], ["ko", "zh"])
        self.assertEqual(len(summaries), 1)
        self.assertTrue(summaries[0]["complete"])
        self.assertEqual(summaries[0]["languages"], ["ko", "zh"])


if __name__ == "__main__":
    unittest.main()
