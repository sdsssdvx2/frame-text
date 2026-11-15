"""Offline adversarial regression tests for external OCR comparison scoring."""
from __future__ import annotations

from contextlib import redirect_stdout
import hashlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import compare_external_engine_exports as target

REFERENCE = (
    "1\n00:00:01,500 --> 00:00:03,625\n"
    "자막 첫째\n자막 둘째\n\n"
    "2\n00:00:04,250 --> 00:00:06,375\n말씨 둘째\n\n"
)


def study(root):
    (root / "source.mp4").write_bytes(b"source-bytes" * 100)
    (root / "no-caption.mp4").write_bytes(b"negative-bytes" * 100)
    a = hashlib.sha256((root / "source.mp4").read_bytes()).hexdigest()
    b = hashlib.sha256((root / "no-caption.mp4").read_bytes()).hexdigest()
    (root / "reference.srt").write_text(REFERENCE, encoding="utf8")
    entries = []
    for eid in ("hse", "external"):
        (root / (eid + ".srt")).write_text(REFERENCE, encoding="utf8")
        (root / (eid + ".blank.srt")).write_bytes(b"")
        entries.append({
            "id": eid, "input_video_sha256": a,
            "blank_video_sha256": b,
            "output_srt": eid + ".srt", "blank_srt": eid + ".blank.srt",
        })
    doc = {
        "schema": 1, "language": "ko", "reference_scope": "synthetic",
        "source_video": "source.mp4", "blank_video": "no-caption.mp4",
        "source_sha256": a, "blank_sha256": b,
        "reference_srt": "reference.srt", "engines": entries,
    }
    (root / "study.json").write_text(json.dumps(doc), encoding="utf8")
    return doc


class ExternalEngineABTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.doc = study(self.root)

    def tearDown(self):
        self.tmp.cleanup()

    def save(self):
        (self.root / "study.json").write_text(
            json.dumps(self.doc), encoding="utf8",
        )

    def score(self):
        return target.grade_directory(self.root)["comparisons"]["external"]

    def test_perfect_different_tools_same_source(self):
        out = target.grade_directory(self.root)
        self.assertFalse(out["source_used_by_engine_independently_attested"])
        self.assertFalse(out["release_gate_passed"])
        self.assertEqual(list(out["comparisons"]), ["hse", "external"])
        for scores in out["comparisons"].values():
            self.assertEqual(
                scores["strict_both_edges_250ms_and_text_wrap"], 2,
            )
            self.assertEqual(scores["aggregate_character_error_rate"], 0)
            self.assertEqual(scores["blank_video_false_cues"], 0)

    def test_timing_degradation_not_hidden_by_exact_words(self):
        (self.root / "external.srt").write_text(
            REFERENCE.replace("00:00:03,625", "00:00:04,125"), encoding="utf8",
        )
        score = self.score()
        self.assertEqual(score["exact_visible_text_ignoring_wrap"], 2)
        self.assertEqual(
            score["strict_both_edges_250ms_and_text_wrap"], 1,
        )
        self.assertEqual(score["both_edges_650ms_and_text_wrap"], 2)

    def test_line_wrap_preserved(self):
        (self.root / "external.srt").write_text(
            REFERENCE.replace("자막 첫째\n자막 둘째", "자막 첫째자막 둘째"),
            encoding="utf8",
        )
        score = self.score()
        self.assertEqual(score["exact_visible_text_ignoring_wrap"], 2)
        self.assertEqual(score["exact_visible_text_and_wrap"], 1)

    def test_punctuation_counts_as_character_error(self):
        (self.root / "external.srt").write_text(
            REFERENCE.replace("말씨 둘째", "말씨, 둘째"), encoding="utf8",
        )
        score = self.score()
        self.assertGreater(score["aggregate_character_error_rate"], 0)
        self.assertEqual(score["exact_visible_text_ignoring_wrap"], 1)

    def test_missed_cues_not_shifted_to_another_reference(self):
        (self.root / "external.srt").write_text(
            "1\n00:00:04,250 --> 00:00:06,375\n말씨 둘째\n",
            encoding="utf8",
        )
        score = self.score()
        self.assertEqual(score["matched_cues"], 1)
        self.assertEqual(score["missed_reference_cues"], 1)
        self.assertGreater(score["aggregate_character_error_rate"], 0)

    def test_extra_cue_detected(self):
        (self.root / "external.srt").write_text(
            REFERENCE + "3\n00:00:07,500 --> 00:00:08,500\n허상 자막\n",
            encoding="utf8",
        )
        score = self.score()
        self.assertEqual(score["extra_extracted_cues"], 1)
        self.assertGreater(score["aggregate_character_error_rate"], 0)

    def test_blank_false_cue_detected(self):
        (self.root / "external.blank.srt").write_text(
            "1\n00:00:01,000 --> 00:00:03,000\n가짜 자막\n",
            encoding="utf8",
        )
        score = self.score()
        self.assertEqual(score["blank_video_false_cues"], 1)
        self.assertFalse(score["no_caption_control_pass"])

    def test_source_sha_required(self):
        self.doc["source_sha256"] = "a" * 64
        self.save()
        with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
            target.grade_directory(self.root)

    def test_engine_attestation_required(self):
        self.doc["engines"][0]["input_video_sha256"] = "a" * 64
        self.save()
        with self.assertRaises(ValueError):
            target.grade_directory(self.root)

    def test_missing_blank_control_is_not_clean(self):
        (self.root / "external.blank.srt").unlink()
        with self.assertRaises((ValueError, FileNotFoundError)):
            target.grade_directory(self.root)

    def test_invalid_srt_fails_instead_of_dropping(self):
        (self.root / "external.srt").write_text(
            REFERENCE + "nonsense", encoding="utf8",
        )
        with self.assertRaises(ValueError):
            target.grade_directory(self.root)

    def test_duplicate_engine_id_is_fatal(self):
        self.doc["engines"][1]["id"] = "hse"
        self.save()
        with self.assertRaises(ValueError):
            target.grade_directory(self.root)

    def test_directory_traversal_rejected(self):
        self.doc["reference_srt"] = "../outside.srt"
        self.save()
        with self.assertRaises(ValueError):
            target.grade_directory(self.root)

    def test_symlink_rejected(self):
        link = self.root / "ext-link.srt"
        self.doc["engines"][1]["output_srt"] = "ext-link.srt"
        self.save()
        try:
            link.symlink_to(self.root / "external.srt")
        except OSError as exc:
            # Unprivileged Windows accounts cannot create file symlinks
            # (WinError 1314). Keep the path guard tested without granting
            # Windows privileges or weakening the production rejection.
            if getattr(exc, "winerror", None) != 1314:
                raise
            with mock.patch.object(
                target.Path, "is_symlink", autospec=True,
                side_effect=lambda path: path.name == "ext-link.srt",
            ):
                with self.assertRaises(ValueError):
                    target.grade_directory(self.root)
        else:
            with self.assertRaises(ValueError):
                target.grade_directory(self.root)

    def test_duplicate_json_field_rejected(self):
        (self.root / "study.json").write_text(
            '{"schema":1,"schema":1}', encoding="utf8",
        )
        with self.assertRaisesRegex(ValueError, "duplicate JSON"):
            target.grade_directory(self.root)

    def test_no_user_transcript_or_path_in_error_report(self):
        self.doc["reference_srt"] = "../秘密-phrase.srt"
        self.save()
        with redirect_stdout(io.StringIO()) as capture:
            with mock.patch(
                "sys.argv", ["score", "--study-dir", str(self.root)],
            ):
                rc = target.main()
        self.assertEqual(rc, 3)
        self.assertNotIn("秘密", capture.getvalue())
        self.assertNotIn(str(self.root), capture.getvalue())

    def test_malformed_srt_timestamps(self):
        for bad in (
            b"1\n00:00:61,000 --> 00:00:62,000\nx\n",
            b"1\n00:00:01,000 --> 00:00:00,000\nx\n",
            b"<html>oops</html>",
        ):
            with self.subTest(bad=bad):
                with self.assertRaises(ValueError):
                    target.parse_strict_srt(bad)

    def test_unaligned_cues_not_forced_matched_by_equal_text(self):
        score = target.evaluate(
            [target.Cue(0, 1, "same")],
            [target.Cue(5, 6, "same")], [],
        )
        self.assertEqual(score["matched_cues"], 0)
        self.assertEqual(score["missed_reference_cues"], 1)
        self.assertEqual(score["extra_extracted_cues"], 1)

    def test_zero_positive_cues_is_measured_as_total_miss(self):
        (self.root / "external.srt").write_bytes(b"")
        score = self.score()
        self.assertEqual(score["extracted_cues"], 0)
        self.assertEqual(score["missed_reference_cues"], 2)
        self.assertEqual(score["aggregate_character_error_rate"], 1.0)

    def test_reference_srt_cannot_masquerade_as_candidate(self):
        self.doc["engines"][1]["output_srt"] = "reference.srt"
        self.save()
        with self.assertRaisesRegex(ValueError, "independently named"):
            target.grade_directory(self.root)

    def test_two_engine_exports_must_be_distinct(self):
        self.doc["engines"][1]["output_srt"] = "hse.srt"
        self.save()
        with self.assertRaisesRegex(ValueError, "independently named"):
            target.grade_directory(self.root)

    def test_blank_and_caption_srt_must_be_distinct(self):
        self.doc["engines"][1]["blank_srt"] = "external.srt"
        self.save()
        with self.assertRaisesRegex(ValueError, "independently named"):
            target.grade_directory(self.root)

    def test_srt_unicode_nfc_equivalence(self):
        cues = target.parse_strict_srt(
            "1\n00:00:01,000 --> 00:00:02,000\ne\u0301\n".encode("utf8"),
        )
        self.assertEqual(cues[0].text, "\u00e9")


if __name__ == "__main__":
    unittest.main()
