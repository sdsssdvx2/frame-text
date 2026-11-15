"""No-network contracts for official VideoSubFinder CLI timing research."""
from __future__ import annotations

from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import verify_vsf_cli_research as ver
import compare_vsf_timing_research as target
from compare_external_engine_exports import Cue


class VSFTimingResearchContracts(unittest.TestCase):
    def test_release_url_is_fixed_https_asset(self):
        self.assertEqual(
            ver.URL,
            "https://github.com/eritpchy/videosubfinder-cli/releases/"
            "download/v6.10.5/videosubfinder-cli-windows-x64.zip",
        )
        self.assertEqual(len(ver.EXPECTED_SHA256), 64)

    def test_empty_archive_rejected(self):
        self.assertFalse(ver.archive_is_official(b""))

    def test_other_archive_rejected(self):
        self.assertFalse(ver.archive_is_official(b"dummy zip content"))

    def test_oversized_archive_rejected(self):
        self.assertFalse(ver.archive_is_official(b"x" * (ver.MAX_ARCHIVE_BYTES + 1)))

    def test_research_version_rejects_wrong_executable(self):
        with mock.patch.object(
            target, "_call_bounded",
            return_value=(SimpleNamespace(returncode=0, stdout=b"6.10.4",
                                          stderr=b""), 0.1),
        ):
            with self.assertRaises(RuntimeError):
                target.check_native_version(Path("no.exe"))

    def test_research_version_accepts_exact_release(self):
        with mock.patch.object(
            target, "_call_bounded",
            return_value=(SimpleNamespace(returncode=0, stdout=b"6.10.5",
                                          stderr=b""), 0.1),
        ):
            self.assertTrue(target.check_native_version(Path("VSF.exe")))

    def test_full_detection_cues_have_exact_timing(self):
        from verify_multilang_video_pipeline import EXPECTED_SPANS
        cues = [Cue(start, end, "[sub_duration]") for start,end in EXPECTED_SPANS]
        report = target.timing_report(cues)
        self.assertEqual(report["both_edges_within_250ms"], len(EXPECTED_SPANS))
        self.assertEqual(report["missed_reference_intervals"], 0)
        self.assertEqual(report["extra_timing_intervals"], 0)

    def test_false_timing_intervals_cannot_replace_reference_cues(self):
        report = target.timing_report([Cue(10, 11, "[sub_duration]")])
        self.assertEqual(report["matched_reference_intervals"], 0)
        self.assertEqual(report["missed_reference_intervals"], 2)
        self.assertEqual(report["extra_timing_intervals"], 1)

    def test_delayed_time_fails_strict_edge(self):
        from verify_multilang_video_pipeline import EXPECTED_SPANS
        a, b = EXPECTED_SPANS[0]
        report = target.timing_report([Cue(a, b + .6, "[sub_duration]")])
        self.assertEqual(report["both_edges_within_250ms"], 0)
        self.assertEqual(report["both_edges_within_650ms"], 1)

    def test_missing_output_is_not_reported_as_clean(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            with mock.patch.object(
                target, "_call_bounded",
                return_value=(SimpleNamespace(returncode=0), 0.0),
            ):
                with self.assertRaises(RuntimeError):
                    target.run_vsf(Path("native-vsf.exe"), Path("src.mp4"), root/"out")

    def test_existing_output_directory_is_protected(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root/"out").mkdir()
            with self.assertRaises(RuntimeError):
                target.run_vsf(Path("native-vsf.exe"), Path("src.mp4"), root/"out")

    def test_research_never_passes_native_clear_deletion_option(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            def fake_run(cmd, cwd, timeout):
                self.assertIn("-r", cmd)
                self.assertNotIn("-c", cmd)
                self.assertNotIn("--clear_dirs", cmd)
                out = Path(str(cmd[cmd.index("-ces") + 1]))
                out.write_text(
                    "1\n00:00:01,500 --> 00:00:03,625\n[sub_duration]\n",
                    encoding="utf-8",
                )
                return SimpleNamespace(returncode=0), 0.2
            with mock.patch.object(target, "_call_bounded", side_effect=fake_run):
                cues, dur = target.run_vsf(
                    Path("native-vsf.exe"), Path("source.mp4"), root/"out",
                )
            self.assertEqual(len(cues), 1)
            self.assertEqual(dur, 0.2)

    def test_unsafe_archive_members_rejected(self):
        for value in ("../a.exe", "C:/a.exe", "/root/a.exe", "./a.exe", "a\\b.exe", "a//b.exe"):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    ver._safe_zip_member(value)

    def test_safe_member_name_not_absolute(self):
        self.assertEqual(ver._safe_zip_member("licenses/MIT.txt").parts, ("licenses","MIT.txt"))


if __name__ == "__main__":
    unittest.main()
