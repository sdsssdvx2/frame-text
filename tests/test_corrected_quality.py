import unittest
import tempfile
from pathlib import Path
from hardsub_extractor.models import Cue
from hardsub_extractor.corrected_quality import (
    compare_corrected_cues, CorrectedQualityError, audit_proofread_srt,
)


def c(start, end, text):
    return Cue(start, end, text)


class CorrectedQualityTests(unittest.TestCase):
    def test_identical(self):
        rows = [c(1, 2, "안녕?\n네!"), c(3, 4, "東京。")]
        result = compare_corrected_cues(rows, rows)
        self.assertEqual(
            (result.strict_joint_cues, result.character_error_rate,
             result.aligned_cues), (2, 0, 2),
        )

    def test_punctuation(self):
        result = compare_corrected_cues(
            [c(1, 2, "안녕하세요")], [c(1, 2, "안녕하세요!")]
        )
        self.assertEqual((result.character_errors,
                          result.strict_joint_cues), (1, 0))

    def test_wrap(self):
        result = compare_corrected_cues(
            [c(1, 2, "두 줄의 자막")], [c(1, 2, "두 줄의\n자막")]
        )
        self.assertEqual(
            (result.exact_physical_lines, result.exact_visible_cues), (0, 0)
        )

    def test_same_number_of_lines_but_different_breaks_are_not_equal(self):
        result = compare_corrected_cues(
            [c(1, 2, "안녕\n하세요")], [c(1, 2, "안\n녕하세요")]
        )
        self.assertEqual(result.aligned_cues, 1)
        self.assertEqual(result.precise_250ms_cues, 1)
        self.assertEqual(result.exact_physical_lines, 0)
        self.assertEqual(result.exact_visible_cues, 0)
        self.assertEqual(result.strict_joint_cues, 0)

    def test_same_break_position_remains_comparable_with_one_ocr_typo(self):
        result = compare_corrected_cues(
            [c(1, 2, "첫째\n대사")], [c(1, 2, "첫차\n대사")]
        )
        self.assertEqual(result.exact_physical_lines, 1)
        self.assertEqual(result.exact_visible_cues, 0)
        self.assertEqual(result.character_errors, 1)

    def test_three_line_boundaries_are_position_sensitive(self):
        result = compare_corrected_cues(
            [c(1, 2, "a\nbc\nd")], [c(1, 2, "ab\nc\nd")]
        )
        self.assertEqual(result.exact_physical_lines, 0)

    def test_time_drift(self):
        result = compare_corrected_cues(
            [c(1, 2, "같음")], [c(1.4, 2.4, "같음")]
        )
        self.assertEqual(
            (result.aligned_cues, result.strict_joint_cues,
             result.character_errors), (1, 0, 0),
        )

    def test_missing(self):
        rows = [c(1, 2, "첫"), c(2.5, 3, "둘"), c(4, 5, "셋")]
        result = compare_corrected_cues(rows, [rows[0], rows[2]])
        self.assertEqual(
            (result.aligned_cues, result.baseline_only,
             result.corrected_only), (2, 1, 0),
        )

    def test_added(self):
        rows = [c(1, 2, "첫"), c(4, 5, "셋")]
        result = compare_corrected_cues(
            rows, [rows[0], c(2.5, 3, "둘"), rows[1]]
        )
        self.assertEqual(
            (result.aligned_cues, result.baseline_only,
             result.corrected_only), (2, 0, 1),
        )

    def test_far_shift(self):
        result = compare_corrected_cues(
            [c(1, 2, "다른")], [c(20, 21, "다른")]
        )
        self.assertEqual(
            (result.aligned_cues, result.corrected_only,
             result.baseline_only), (0, 1, 1),
        )

    def test_unsorted(self):
        with self.assertRaises(CorrectedQualityError):
            compare_corrected_cues(
                [c(5, 6, "a"), c(2, 3, "b")], [c(1, 2, "a")]
            )

    def test_large_episode(self):
        rows = [c(i * 2, i * 2 + 1, "字") for i in range(3000)]
        result = compare_corrected_cues(rows, rows)
        self.assertEqual(result.aligned_cues, 3000)

    def test_too_many_insertions_refused(self):
        rows = [c(i * 2, i * 2 + 1, "b") for i in range(280)]
        with self.assertRaises(CorrectedQualityError):
            compare_corrected_cues([c(1, 2, "a")], rows)

    def test_actual_srt_files_are_not_modified(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            original = root / "v.srt"
            corrected = root / "v-proofread.srt"
            original.write_text(
                "1\n00:00:01,000 --> 00:00:02,000\n짧은 대사\n",
                encoding="utf-8",
            )
            corrected.write_text(
                "1\n00:00:01,000 --> 00:00:02,000\n짧은 대사!\n",
                encoding="utf-8",
            )
            source = original.read_bytes()
            revised = corrected.read_bytes()
            result = audit_proofread_srt(original)
            self.assertEqual(result.character_errors, 1)
            self.assertEqual((original.read_bytes(), corrected.read_bytes()),
                             (source, revised))

    def test_gui_renders_only_numbers_not_dialogue_in_log(self):
        from hardsub_extractor.gui import ExtractorGUI
        from unittest import mock

        class Var:
            def __init__(self, value):
                self.value = value

            def get(self):
                return self.value

            def set(self, value):
                self.value = value

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            original = root / "episode.srt"
            revised = root / "episode-proofread.srt"
            original.write_text(
                "1\n00:00:01,000 --> 00:00:02,000\nPRIVATE_DIALOGUE\n",
                encoding="utf-8",
            )
            revised.write_text(
                "1\n00:00:01,000 --> 00:00:02,000\nEDITED_SECRET\n",
                encoding="utf-8",
            )
            app = ExtractorGUI.__new__(ExtractorGUI)
            app._pending_close = False
            app.worker = None
            app._tab_capture_thread = None
            app.source_mode_var = Var("local")
            app.out_var = Var(str(root / "episode"))
            app.status_var = Var("")
            app._log = mock.Mock()
            with mock.patch("hardsub_extractor.gui.messagebox.showinfo") as popup:
                app._compare_corrected_quality()
            self.assertEqual(popup.call_count, 1)
            shown = popup.call_args.args[1]
            self.assertIn("CER", shown)
            self.assertNotIn("PRIVATE_DIALOGUE", shown)
            self.assertNotIn("EDITED_SECRET", shown)
            self.assertNotIn("PRIVATE_DIALOGUE", repr(app._log.call_args_list))
            self.assertEqual(original.stat().st_size, len(original.read_bytes()))
            self.assertEqual(revised.stat().st_size, len(revised.read_bytes()))

    def test_owner_edit_is_not_independent_ground_truth(self):
        result = compare_corrected_cues(
            [c(1, 2, "a")], [c(1, 2, "b")]
        )
        self.assertEqual(
            result.provenance,
            "OWNER_EDITED_NOT_INDEPENDENTLY_VERIFIED",
        )
