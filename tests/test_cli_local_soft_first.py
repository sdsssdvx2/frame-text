"""CLI local-video soft-first and opt-in hard-only backward compatibility."""
from __future__ import annotations

from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace
import io
import unittest
from unittest import mock

from hardsub_extractor import cli


class LocalCliRoutingTests(unittest.TestCase):
    def _run(self, args, soft_result):
        stdout = io.StringIO()
        with mock.patch("sys.argv", ["hard-sub-extract", *args]), mock.patch(
            "hardsub_extractor.cli.acquire_local_embedded_text",
            return_value=soft_result,
        ) as soft, mock.patch(
            "hardsub_extractor.cli.acquire_local_sidecar_text", return_value=None,
        ) as sidecar, mock.patch(
            "hardsub_extractor.cli.acquire_local_embedded_image", return_value=None,
        ) as bitmap, mock.patch("hardsub_extractor.cli._ocr", return_value="ocr") as ocr, mock.patch(
            "hardsub_extractor.cli.extract",
            return_value=SimpleNamespace(segment_count=3, cue_count=1,
                                         srt_path=Path("out.srt"), ass_path=Path("out.ass")),
        ) as hard, redirect_stdout(stdout):
            cli.main()
        return stdout.getvalue(), soft, ocr, hard

    def test_soft_track_never_loads_hard_ocr_models(self):
        result = SimpleNamespace(
            mode="local-embedded-text", cue_count=2,
            provenance_path=Path("out.source.json"),
            srt_path=Path("out.srt"), ass_path=Path("out.ass"),
        )
        output, soft, ocr, hard = self._run(
            ["episode.mkv", "--out", "out", "--ocr-language", "en"], result,
        )
        soft.assert_called_once_with("episode.mkv", Path("out"), language="en")
        ocr.assert_not_called()
        hard.assert_not_called()
        self.assertIn("mode=local-embedded-text", output)

    def test_no_matching_soft_track_lazily_falls_back_to_ocr(self):
        output, soft, ocr, hard = self._run(
            ["episode.mkv", "--out", "out", "--ocr-language", "ja"], None,
        )
        soft.assert_called_once_with("episode.mkv", Path("out"), language="ja")
        # A missing embedded text track must check sidecar and bitmap
        # candidates before the true hard-video OCR fallback.
        ocr.assert_called_once()
        hard.assert_called_once()
        self.assertNotIn("focus_bottom_two_lines", hard.call_args.kwargs)
        self.assertIn("mode=hard-ocr", output)

    def test_hard_only_bypasses_embedded_track_for_explicit_comparison(self):
        output, soft, ocr, hard = self._run(
            ["episode.mkv", "--out", "out", "--hard-only"], None,
        )
        soft.assert_not_called()
        ocr.assert_called_once()
        hard.assert_called_once()
        self.assertIn("mode=hard-ocr", output)

    def test_bottom_two_focus_opt_in_reaches_actual_local_hard_ocr(self):
        output, soft, ocr, hard = self._run(
            ["episode.mkv", "--out", "out", "--hard-only",
             "--focus-bottom-two-lines"], None,
        )
        soft.assert_not_called()
        ocr.assert_called_once()
        hard.assert_called_once()
        self.assertIs(hard.call_args.kwargs["focus_bottom_two_lines"], True)
        self.assertTrue(hard.call_args.kwargs["monitor"].exclusive_outputs)
        self.assertIn("mode=hard-ocr", output)

    def test_bottom_two_focus_never_runs_for_existing_embedded_text(self):
        source = SimpleNamespace(
            mode="local-embedded-text", cue_count=1,
            provenance_path=Path("original.json"),
            srt_path=Path("original.srt"), ass_path=Path("original.ass"),
        )
        _, soft, ocr, hard = self._run(
            ["episode.mkv", "--out", "out", "--focus-bottom-two-lines"], source,
        )
        soft.assert_called_once()
        ocr.assert_not_called()
        hard.assert_not_called()

    def test_bottom_two_focus_rejects_other_modes_before_loading_models(self):
        combinations = (
            ["--native-vsf-official"],
            ["--caption-islands"],
            ["--light-caption-only"],
            ["--opencv-recognizer"],
            ["--epsite"],
        )
        for extra in combinations:
            with self.subTest(extra=extra), mock.patch(
                "sys.argv", ["hard-sub-extract", "video.mkv",
                             "--focus-bottom-two-lines", *extra],
            ), mock.patch.object(cli, "_ocr") as model, mock.patch(
                "hardsub_extractor.cli.extract",
            ) as hard:
                with self.assertRaises(SystemExit) as exc:
                    cli.main()
                self.assertEqual(exc.exception.code, 2)
                model.assert_not_called()
                hard.assert_not_called()

    def test_preflight_failure_does_not_fall_back_and_hide_errors(self):
        with mock.patch(
            "sys.argv", ["hard-sub-extract", "video.mkv", "--out", "out"],
        ), mock.patch(
            "hardsub_extractor.cli.acquire_local_embedded_text",
            side_effect=ValueError("invalid local media"),
        ), mock.patch("hardsub_extractor.cli._ocr") as ocr:
            with self.assertRaisesRegex(ValueError, "invalid local media"):
                cli.main()
        ocr.assert_not_called()


if __name__ == "__main__":
    unittest.main()
