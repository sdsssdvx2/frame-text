"""Research-only opt-in native timing flags must preserve local soft-first."""
from __future__ import annotations

from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

from hardsub_extractor import site_adapter
from hardsub_extractor.site_adapter import site_adapter_available
from unittest import mock

from hardsub_extractor import cli


class NativeVsfCLITests(unittest.TestCase):
    def test_missing_sha_or_executable_fails_before_model_and_media(self):
        for args in (
            ["--native-vsf-executable", "vsf.exe"],
            ["--native-vsf-sha256", "a" * 64],
        ):
            with self.subTest(args=args), mock.patch(
                "sys.argv", ["hard-sub-extract", "original.mkv", *args],
            ), mock.patch.object(cli, "_ocr") as model:
                with self.assertRaises(SystemExit) as failure:
                    cli.main()
                self.assertEqual(failure.exception.code, 2)
                model.assert_not_called()

    @unittest.skipUnless(site_adapter_available(), "private site adapter not in this build")
    def test_rejects_native_binary_for_remote_epsite(self):
        with mock.patch(
            "sys.argv", [
                "hard-sub-extract", site_adapter.SITE_ADAPTER.SAMPLE_EPISODE_URL,
                "--native-vsf-executable", "vsf.exe",
                "--native-vsf-sha256", "a"*64,
            ],
        ), mock.patch.object(cli, "_ocr") as model:
            with self.assertRaises(SystemExit) as failure:
                cli.main()
            self.assertEqual(failure.exception.code, 2)
            model.assert_not_called()

    def test_hard_only_explicit_native_uses_bounded_source_timing(self):
        report = SimpleNamespace(
            detected_intervals=2, recognized_intervals=2,
            missed_intervals=0, ocr_calls=10,
            srt_path=Path("result.srt"), ass_path=Path("result.ass"),
        )
        with (
            mock.patch("sys.argv", [
                "hard-sub-extract", "video.mkv",
                "--hard-only", "--native-vsf-executable", "pinned.exe",
                "--native-vsf-sha256", "f"*64,
            ]),
            mock.patch("hardsub_extractor.native_vsf_timing.extract_with_native_vsf",
                       return_value=SimpleNamespace(report=report)) as native,
            mock.patch.object(cli, "_ocr", return_value="lazy-ocr") as make_ocr,
            mock.patch("builtins.print") as print_result,
        ):
            cli.main()
        self.assertEqual(native.call_count, 1)
        self.assertEqual(native.call_args.kwargs["pinned_executable_sha256"], "f"*64)
        self.assertEqual(native.call_args.kwargs["executable"], "pinned.exe")
        # The model factory is passed lazily, not called by CLI if native
        # detection later finds no time intervals.
        make_ocr.assert_not_called()
        self.assertTrue(any("mode=hard-ocr-native-timing" in str(c.args[0])
                            for c in print_result.call_args_list))

    def test_short_frame_rescue_counts_are_visible_without_subtitle_text(self):
        report = SimpleNamespace(
            detected_intervals=2, recognized_intervals=2,
            missed_intervals=0, ocr_calls=12,
            short_frame_rescue_attempted=1,
            short_frame_rescue_recognized=1,
            short_frame_rescue_deferred=0,
            short_frame_rescue_errors=0,
            srt_path=Path("captions.srt"), ass_path=Path("captions.ass"),
        )
        with (
            mock.patch("sys.argv", [
                "hard-sub-extract", "video.mkv",
                "--hard-only", "--native-vsf-executable", "native.exe",
                "--native-vsf-sha256", "f"*64,
            ]),
            mock.patch("hardsub_extractor.native_vsf_timing.extract_with_native_vsf",
                       return_value=SimpleNamespace(report=report)),
            mock.patch("builtins.print") as prints,
        ):
            cli.main()
        lines = [str(x.args[0]) for x in prints.call_args_list]
        self.assertTrue(any(
            "short_frame_rescue_attempted=1 recovered=1" in line
            for line in lines
        ))
        self.assertFalse(any("SECRET_OCR_WORDS" in line for line in lines))

    def test_stable_split_rejects_no_native_detector(self):
        with (
            mock.patch("sys.argv", [
                "hard-sub-extract", "video.mkv",
                "--native-vsf-split-stable",
            ]),
            mock.patch.object(cli, "_ocr") as model,
        ):
            with self.assertRaises(SystemExit) as outcome:
                cli.main()
            self.assertEqual(outcome.exception.code, 2)
            model.assert_not_called()

    def test_official_opt_in_is_lazy_when_original_text_track_exists(self):
        source = SimpleNamespace(
            mode="local-embedded-text", cue_count=2,
            provenance_path=Path("origin.json"),
            srt_path=Path("origin.srt"), ass_path=Path("origin.ass"),
        )
        with (
            mock.patch("sys.argv", [
                "hard-sub-extract", "episode.mkv", "--native-vsf-official",
            ]),
            mock.patch.object(cli, "acquire_local_embedded_text", return_value=source),
            mock.patch("hardsub_extractor.native_vsf_assets.ensure_official_vsf") as fetch,
            mock.patch("hardsub_extractor.native_vsf_timing.extract_with_native_vsf") as run,
            mock.patch.object(cli, "_ocr") as make_ocr,
            mock.patch("builtins.print"),
        ):
            cli.main()
        fetch.assert_not_called()
        run.assert_not_called()
        make_ocr.assert_not_called()

    def test_opt_in_official_download_deferred_until_no_original_track(self):
        report = SimpleNamespace(
            detected_intervals=2, recognized_intervals=2,
            missed_intervals=0, ocr_calls=10,
            srt_path=Path("subtitle.srt"), ass_path=Path("subtitle.ass"),
        )
        # This opt-in now intentionally requires an actual LOCAL media file
        # before downloading its optional third-party helper. Test that
        # prerequisite without decoding, uploading or changing user media.
        with tempfile.TemporaryDirectory() as td:
            source = Path(td) / "episode.mkv"
            source.write_bytes(b"synthetic local video placeholder for CLI preflight")
            with (
                mock.patch("sys.argv", [
                    "hard-sub-extract", str(source),
                    "--hard-only", "--native-vsf-official",
                ]),
                mock.patch("hardsub_extractor.native_vsf_assets.ensure_official_vsf",
                           return_value=(Path("cached-VSF.exe"), "1"*64)) as fetch,
                mock.patch("hardsub_extractor.native_vsf_timing.extract_with_native_vsf",
                           return_value=SimpleNamespace(report=report)) as run,
                mock.patch.object(cli, "_ocr") as make_ocr,
                mock.patch("builtins.print"),
            ):
                cli.main()
            self.assertEqual(
                source.read_bytes(), b"synthetic local video placeholder for CLI preflight",
            )
        fetch.assert_called_once_with(consent=True)
        self.assertEqual(run.call_args.kwargs["pinned_executable_sha256"], "1"*64)
        self.assertEqual(run.call_args.kwargs["executable"], Path("cached-VSF.exe"))
        make_ocr.assert_not_called()

    def test_official_and_manual_flags_cannot_be_combined(self):
        with mock.patch("sys.argv", [
            "hard-sub-extract", "episode.mkv",
            "--native-vsf-official", "--native-vsf-executable", "native.exe",
            "--native-vsf-sha256", "f"*64,
        ]), mock.patch.object(cli, "_ocr") as model:
            with self.assertRaises(SystemExit) as error:
                cli.main()
            self.assertEqual(error.exception.code, 2)
            model.assert_not_called()

    def test_official_flag_rejects_remote_input_before_any_download(self):
        with (
            mock.patch("sys.argv", [
                "hard-sub-extract", "https://example.org/video.mkv",
                "--hard-only", "--native-vsf-official",
            ]),
            mock.patch("hardsub_extractor.native_vsf_assets.ensure_official_vsf") as fetch,
            mock.patch.object(cli, "_ocr") as ocr,
        ):
            with self.assertRaises(SystemExit) as failure:
                cli.main()
        self.assertEqual(failure.exception.code, 2)
        fetch.assert_not_called()
        ocr.assert_not_called()

    def test_official_flag_preserves_existing_result_without_download(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            video = root / "clip.mkv"
            video.write_bytes(b"LOCAL SYNTHETIC PROBE INPUT")
            output = root / "captions"
            user_srt = output.with_suffix(".srt")
            user_srt.write_bytes(b"ORIGINAL USER SUBTITLE")
            with (
                mock.patch("sys.argv", [
                    "hard-sub-extract", str(video), "--hard-only",
                    "--native-vsf-official", "--out", str(output),
                ]),
                mock.patch("hardsub_extractor.native_vsf_assets.ensure_official_vsf") as fetch,
                mock.patch.object(cli, "_ocr") as ocr,
            ):
                with self.assertRaises(SystemExit) as failure:
                    cli.main()
            self.assertEqual(failure.exception.code, 2)
            fetch.assert_not_called()
            ocr.assert_not_called()
            self.assertEqual(user_srt.read_bytes(), b"ORIGINAL USER SUBTITLE")

    def test_source_first_keeps_valid_soft_original_without_native_process(self):
        source = SimpleNamespace(
            mode="local-embedded-text",
            cue_count=2,
            provenance_path=Path("origin.json"),
            srt_path=Path("origin.srt"),
            ass_path=Path("origin.ass"),
        )
        with (
            mock.patch("sys.argv", [
                "hard-sub-extract", "episode.mkv",
                "--native-vsf-executable", "pinned.exe",
                "--native-vsf-sha256", "f"*64,
            ]),
            mock.patch.object(cli, "acquire_local_embedded_text", return_value=source),
            mock.patch("hardsub_extractor.native_vsf_timing.extract_with_native_vsf") as native,
            mock.patch.object(cli, "_ocr") as make_ocr,
            mock.patch("builtins.print"),
        ):
            cli.main()
        native.assert_not_called()
        make_ocr.assert_not_called()


if __name__ == "__main__":
    unittest.main()
