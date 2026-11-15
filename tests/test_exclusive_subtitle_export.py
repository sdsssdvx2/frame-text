"""No-model tests for exclusive SRT/ASS publication and sync report rollback."""
from __future__ import annotations

import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

from hardsub_extractor.models import Cue
from hardsub_extractor.subtitle_formats import normalize_subtitle_bytes
from hardsub_extractor.sync_workflow import sync_srt_to_target
from hardsub_extractor.writers import (
    write_ass,
    write_srt,
    write_subtitle_pair_exclusive,
)


def sample_cues():
    return [Cue(1.0, 2.5, "비공개 한국어\n두 번째 줄", 0.95)]


class ExclusiveSubtitleExportTests(unittest.TestCase):
    def test_exclusive_srt_and_ass_have_identical_bytes_to_legacy_writers(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            legacy = root / "legacy"
            fresh = root / "fresh"
            reference_srt = write_srt(sample_cues(), legacy.with_suffix(".srt"))
            reference_ass = write_ass(sample_cues(), legacy.with_suffix(".ass"))
            result_srt, result_ass = write_subtitle_pair_exclusive(sample_cues(), fresh)
            self.assertEqual(result_srt.read_bytes(), reference_srt.read_bytes())
            self.assertEqual(result_ass.read_bytes(), reference_ass.read_bytes())
            self.assertEqual(
                sorted(x.name for x in root.iterdir()),
                ["fresh.ass", "fresh.srt", "legacy.ass", "legacy.srt"],
            )

    def test_existing_srt_is_never_overwritten(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            base = root / "existing"
            existing = base.with_suffix(".srt")
            existing.write_text("ORIGINAL USER SRT", encoding="utf-8")
            with self.assertRaises(FileExistsError):
                write_subtitle_pair_exclusive(sample_cues(), base)
            self.assertEqual(existing.read_text(encoding="utf-8"), "ORIGINAL USER SRT")
            self.assertFalse(base.with_suffix(".ass").exists())

    def test_existing_ass_is_preserved_and_new_srt_is_rolled_back(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            base = root / "existing"
            existing = base.with_suffix(".ass")
            existing.write_text("ORIGINAL USER ASS", encoding="utf-8")
            with self.assertRaises(FileExistsError):
                write_subtitle_pair_exclusive(sample_cues(), base)
            self.assertEqual(existing.read_text(encoding="utf-8"), "ORIGINAL USER ASS")
            self.assertFalse(base.with_suffix(".srt").exists())

    def test_hardlink_alias_of_other_input_is_never_truncated(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            base = root / "output"
            original = root / "precious.srt"
            original.write_text("KEEP THIS SOURCE", encoding="utf-8")
            os.link(original, base.with_suffix(".srt"))
            with self.assertRaises(FileExistsError):
                write_subtitle_pair_exclusive(sample_cues(), base)
            self.assertEqual(original.read_text(encoding="utf-8"), "KEEP THIS SOURCE")
            self.assertEqual(base.with_suffix(".srt").read_text(encoding="utf-8"), "KEEP THIS SOURCE")

    def test_partial_write_error_removes_only_new_file(self):
        with tempfile.TemporaryDirectory() as td:
            base = Path(td) / "output"
            import shutil

            def fail_copy(source, destination, *args, **kwargs):
                destination.write(b"PARTIALLY_WRITTEN")
                raise OSError("simulated failure")

            with mock.patch.object(shutil, "copyfileobj", side_effect=fail_copy):
                with self.assertRaisesRegex(OSError, "simulated failure"):
                    write_subtitle_pair_exclusive(sample_cues(), base)
            self.assertFalse(base.with_suffix(".srt").exists())
            self.assertFalse(base.with_suffix(".ass").exists())
            self.assertFalse(list(Path(td).glob(".hse-subtitles-*")))

    def test_exclusive_soft_track_normalization_still_uses_original_text(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            input_sub = (
                "1\n00:00:01,000 --> 00:00:02,000\n한국어 원문 자막\n"
            ).encode("utf-8")
            result = normalize_subtitle_bytes(
                input_sub, root / "output", source_url="source.srt",
                require_korean=True, exclusive_outputs=True,
            )
            self.assertEqual(result.cue_count, 1)
            original = result.srt_path.read_bytes()
            with self.assertRaises(FileExistsError):
                normalize_subtitle_bytes(
                    input_sub, root / "output", source_url="source.srt",
                    require_korean=True, exclusive_outputs=True,
                )
            self.assertEqual(result.srt_path.read_bytes(), original)
            self.assertIn("한국어 원문 자막", result.srt_path.read_text(encoding="utf-8"))

    def test_existing_sync_report_cannot_be_overwritten(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            source = write_srt(sample_cues(), root / "source.srt")
            base = root / "target"
            report = base.with_suffix(".sync.json")
            report.write_text("PREEXISTING SYNC REPORT", encoding="utf-8")
            sync = SimpleNamespace(segments=[], anchors=[], sample_step=0.25)
            with (
                mock.patch(
                    "hardsub_extractor.sync_workflow.estimate_piecewise_offsets",
                    return_value=sync,
                ),
                mock.patch(
                    "hardsub_extractor.sync_workflow.apply_piecewise_offsets",
                    return_value=sample_cues(),
                ),
            ):
                with self.assertRaises(FileExistsError):
                    sync_srt_to_target(
                        source, "source.mp4", "target.mp4", base,
                        exclusive_outputs=True,
                    )
            self.assertEqual(
                report.read_text(encoding="utf-8"), "PREEXISTING SYNC REPORT"
            )
            self.assertFalse(base.with_suffix(".srt").exists())
            self.assertFalse(base.with_suffix(".ass").exists())

    def test_late_sync_report_write_failure_rolls_back_both_new_subtitles(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            source = write_srt(sample_cues(), root / "source.srt")
            base = root / "target"
            report = base.with_suffix(".sync.json")
            sync = SimpleNamespace(segments=[], anchors=[], sample_step=0.25)
            original_open = Path.open

            def fail_only_report(path, *args, **kwargs):
                if path == report and args and args[0] == "x":
                    raise OSError("simulated report failure")
                return original_open(path, *args, **kwargs)

            with (
                mock.patch(
                    "hardsub_extractor.sync_workflow.estimate_piecewise_offsets",
                    return_value=sync,
                ),
                mock.patch(
                    "hardsub_extractor.sync_workflow.apply_piecewise_offsets",
                    return_value=sample_cues(),
                ),
                mock.patch.object(Path, "open", fail_only_report),
            ):
                with self.assertRaisesRegex(OSError, "simulated report failure"):
                    sync_srt_to_target(
                        source, "source.mp4", "target.mp4", base,
                        exclusive_outputs=True,
                    )
            self.assertTrue(source.exists())
            self.assertFalse(report.exists())
            self.assertFalse(base.with_suffix(".srt").exists())
            self.assertFalse(base.with_suffix(".ass").exists())


if __name__ == "__main__":
    unittest.main()
