import queue
import tempfile
import unittest

from hardsub_extractor import site_adapter
from hardsub_extractor.site_adapter import site_adapter_available
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from hardsub_extractor.gui import ExtractorGUI, _user_error_summary


class Var:
    def __init__(self, value):
        self.value = value
    def get(self):
        return self.value


class LocalInputGuiTests(unittest.TestCase):
    def gui(self, root, mode="local"):
        gui = ExtractorGUI.__new__(ExtractorGUI)
        gui.source_mode_var = Var(mode)
        gui.url_var = Var(str(root / "한글 영상.mkv"))
        gui.target_var = Var(str(root / "missing-old-target.mkv"))
        gui.out_var = Var(str(root / "output" / "subtitles"))
        gui.events = queue.Queue()
        return gui

    def messages(self, gui):
        return list(gui.events.queue)

    def test_switch_to_local_enables_video_picker_and_disables_target(self):
        with tempfile.TemporaryDirectory() as td:
            gui = self.gui(Path(td), "local")
            gui.url_var = mock.Mock()
            gui.source_label = mock.Mock()
            gui.source_picker = mock.Mock()
            gui.target_entry = mock.Mock()
            gui.target_picker = mock.Mock()
            gui._source_mode_changed()
            gui.source_picker.configure.assert_called_once_with(state="normal", text="영상 선택")
            gui.target_entry.configure.assert_called_once_with(state="disabled")
            gui.target_picker.configure.assert_any_call(state="disabled")
            gui.source_label.configure.assert_called_once_with(
                text="자막이 포함된 영상 파일"
            )
            gui.url_var.set.assert_called_once_with("")

    def test_switch_to_epsite_disables_file_picker(self):
        with tempfile.TemporaryDirectory() as td:
            gui = self.gui(Path(td), "epsite")
            gui.url_var = mock.Mock()
            gui.source_label = mock.Mock()
            gui.source_picker = mock.Mock()
            gui.target_entry = mock.Mock()
            gui.target_picker = mock.Mock()
            gui._source_mode_changed()
            gui.source_picker.configure.assert_called_once_with(state="disabled", text="영상 선택")
            gui.target_entry.configure.assert_called_once_with(state="normal")
            gui.target_picker.configure.assert_any_call(state="normal")
            gui.url_var.set.assert_called_once_with("")

    def test_validate_local_unicode_path_and_ignore_old_target(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root / "한글 영상.mkv").write_bytes(b"fixture")
            gui = self.gui(root)
            mode, source, target, out = gui._validate()
            self.assertEqual(mode, "local")
            self.assertEqual(source, str(root / "한글 영상.mkv"))
            self.assertIsNone(target)
            self.assertTrue(out.parent.is_dir())

    def test_existing_subtitles_are_not_overwritten(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root / "한글 영상.mkv").write_bytes(b"fixture")
            gui = self.gui(root)
            out = root / "output" / "subtitles.srt"
            out.parent.mkdir()
            out.write_text("existing subtitles", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "이미"):
                gui._validate()
            self.assertEqual(out.read_text(encoding="utf-8"), "existing subtitles")

    @unittest.skipUnless(site_adapter_available(), "private site adapter not in this build")
    def test_epsite_existing_srt_is_preserved_before_browser_start(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            gui = self.gui(root, "epsite")
            gui.url_var = Var(site_adapter.SITE_ADAPTER.SAMPLE_EPISODE_URL)
            gui.target_var = Var("")
            output = root / "output" / "subtitles.srt"
            output.parent.mkdir()
            output.write_text("DO NOT ERASE", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "이미"):
                gui._validate()
            self.assertEqual(output.read_text(encoding="utf-8"), "DO NOT ERASE")

    @unittest.skipUnless(site_adapter_available(), "private site adapter not in this build")
    def test_epsite_target_sync_preserves_source_timeline_files(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            gui = self.gui(root, "epsite")
            gui.url_var = Var(site_adapter.SITE_ADAPTER.SAMPLE_EPISODE_URL)
            target = root / "target.mkv"
            target.write_bytes(b"synthetic")
            gui.target_var = Var(str(target))
            source = root / "output" / "subtitles-epsite-source.srt"
            source.parent.mkdir()
            source.write_text("ORIGINAL TIMEBASE", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "이미"):
                gui._validate()
            self.assertEqual(
                source.read_text(encoding="utf-8"), "ORIGINAL TIMEBASE"
            )

    @unittest.skipUnless(site_adapter_available(), "private site adapter not in this build")
    def test_epsite_target_sync_preserves_existing_sync_report(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            gui = self.gui(root, "epsite")
            gui.url_var = Var(site_adapter.SITE_ADAPTER.SAMPLE_EPISODE_URL)
            target = root / "target.mkv"
            target.write_bytes(b"synthetic")
            gui.target_var = Var(str(target))
            report = root / "output" / "subtitles.sync.json"
            report.parent.mkdir()
            report.write_text("existing-sync-report", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "이미"):
                gui._validate()
            self.assertEqual(
                report.read_text(encoding="utf-8"), "existing-sync-report"
            )

    @unittest.skipUnless(site_adapter_available(), "private site adapter not in this build")
    def test_epsite_output_guard_rejects_dangling_symlink(self):
        # Windows symlinks may require developer mode; simulate the filesystem
        # predicate rather than creating a privileged link on the runner.
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            gui = self.gui(root, "epsite")
            gui.url_var = Var(site_adapter.SITE_ADAPTER.SAMPLE_EPISODE_URL)
            gui.target_var = Var("")
            with mock.patch.object(Path, "is_symlink", return_value=True):
                with self.assertRaisesRegex(ValueError, "이미"):
                    gui._validate()

    def test_missing_local_file_does_not_create_output(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            gui = self.gui(root)
            with self.assertRaises(ValueError):
                gui._validate()
            self.assertFalse((root / "output").exists())

    def test_epsite_mode_rejects_unrelated_or_suffix_spoof_hosts(self):
        with tempfile.TemporaryDirectory() as td:
            gui = self.gui(Path(td), "epsite")
            for source in ("https://example.com/x", "https://lookalike.evil.example/x"):
                gui.url_var = Var(source)
                with self.subTest(source=source), self.assertRaises(ValueError):
                    gui._validate()

    def test_single_local_crop_passes_exact_ffmpeg_roi_only_after_source_priority(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            movie = root / "source.mkv"
            movie.write_bytes(b"source unchanged")
            gui = self.gui(root)
            result = SimpleNamespace(
                cue_count=2, srt_path=root / "out.srt", ass_path=root / "out.ass",
            )
            with (mock.patch("hardsub_extractor.gui.acquire_local_embedded_text",
                             return_value=None) as embedded,
                  mock.patch("hardsub_extractor.gui.acquire_local_sidecar_text",
                             return_value=None),
                  mock.patch("hardsub_extractor.gui.acquire_local_embedded_image",
                             return_value=None),
                  mock.patch("hardsub_extractor.gui.bundled_model_dir",
                             return_value=root),
                  mock.patch("hardsub_extractor.gui.RapidOCRAdapter",
                             return_value=mock.Mock()),
                  mock.patch("hardsub_extractor.gui.extract_local_video",
                             return_value=result) as actual):
                gui._worker_local(str(movie), root / "out", False,
                                  region_mode="lower65")
            embedded.assert_called_once()
            selected = actual.call_args.kwargs["roi"]
            self.assertEqual((selected.x, selected.y, selected.w, selected.h),
                             (.04, .33, .92, .65))
            self.assertEqual(movie.read_bytes(), b"source unchanged")
            self.assertIn(("done", (str(result.srt_path), str(result.ass_path))),
                          self.messages(gui))

    def test_custom_local_crop_rejects_incompatible_native_before_source(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            gui = self.gui(root)
            with (mock.patch("hardsub_extractor.gui.acquire_local_embedded_text") as embedded,
                  self.assertRaises(ValueError)):
                gui._worker_local(str(root / "movie.mkv"), root / "out", False,
                                  region_mode="bottom25", native_vsf_opt_in=True)
            embedded.assert_not_called()
            self.assertFalse((root / "out").exists())

    def test_local_worker_uses_local_pipeline_without_http_client(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            source = root / "source.mkv"
            source.write_bytes(b"synthetic-existing-video")
            gui = self.gui(root)
            result = SimpleNamespace(cue_count=2, srt_path=root / "out.srt", ass_path=root / "out.ass")
            with mock.patch("hardsub_extractor.gui.acquire_local_embedded_text", return_value=None), mock.patch("hardsub_extractor.gui.acquire_local_sidecar_text", return_value=None), mock.patch("hardsub_extractor.gui.acquire_local_embedded_image", return_value=None), mock.patch("hardsub_extractor.gui.bundled_model_dir", return_value=root), mock.patch(
                "hardsub_extractor.gui.RapidOCRAdapter", return_value="base"
            ), mock.patch("hardsub_extractor.gui.AdaptiveOCREngine", return_value="ocr"), mock.patch(
                "hardsub_extractor.gui.extract_local_video", return_value=result
            ) as extract, mock.patch("hardsub_extractor.gui.AutoEpSiteClient") as network:
                gui._worker("local", str(source), None, root / "out", False)
            network.assert_not_called()
            extract.assert_called_once_with(str(source), "ocr", root / "out", diagnostics=None, streaming=True)
            self.assertIn(("done", (str(result.srt_path), str(result.ass_path))), self.messages(gui))

    def test_caption_islands_local_worker_wraps_only_visual_fallback(self):
        from hardsub_extractor.ocr.caption_islands import CaptionIslandEngine
        from hardsub_extractor.ocr.adaptive import AdaptiveOCREngine
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            source = root / "video.mkv"
            source.write_bytes(b"unchanged")
            gui = self.gui(root)
            done = SimpleNamespace(
                cue_count=2, srt_path=root / "out.srt", ass_path=root / "out.ass",
            )
            with (
                mock.patch("hardsub_extractor.gui.acquire_local_embedded_text",
                           return_value=None),
                mock.patch("hardsub_extractor.gui.acquire_local_sidecar_text",
                           return_value=None),
                mock.patch("hardsub_extractor.gui.acquire_local_embedded_image",
                           return_value=None),
                mock.patch("hardsub_extractor.gui.bundled_model_dir",
                           return_value=root),
                mock.patch("hardsub_extractor.gui.RapidOCRAdapter",
                           return_value=mock.Mock()),
                mock.patch("hardsub_extractor.gui.extract_local_video",
                           return_value=done) as visual,
            ):
                gui._worker_local(str(source), root / "out", False,
                                  caption_islands=True)
            visual.assert_called_once()
            selected = visual.call_args.args[1]
            self.assertIsInstance(selected, CaptionIslandEngine)
            self.assertIsInstance(selected.engine, AdaptiveOCREngine)
            self.assertTrue(visual.call_args.kwargs["streaming"])
            self.assertEqual(source.read_bytes(), b"unchanged")
            self.assertIn(
                ("done", (str(done.srt_path), str(done.ass_path))), self.messages(gui),
            )

    def test_caption_islands_local_soft_track_skips_model(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            gui = self.gui(root)
            original = SimpleNamespace(
                mode="local-embedded-text", cue_count=2,
                srt_path=root / "original.srt", ass_path=root / "original.ass",
                provenance_path=root / "original.source.json",
            )
            with (
                mock.patch("hardsub_extractor.gui.acquire_local_embedded_text",
                           return_value=original),
                mock.patch("hardsub_extractor.gui.RapidOCRAdapter") as recognizer,
                mock.patch("hardsub_extractor.gui.extract_local_video") as hard,
            ):
                gui._worker_local("video.mkv", root / "out", False,
                                  caption_islands=True)
            recognizer.assert_not_called()
            hard.assert_not_called()
            self.assertIn(
                ("done", (str(original.srt_path), str(original.ass_path))),
                self.messages(gui),
            )

    def test_caption_islands_worker_rejects_conflicting_modes_before_probe(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            gui = self.gui(root)
            for opts in (
                {"caption_islands": True, "native_vsf_opt_in": True},
                {"caption_islands": True, "focus_bottom_two_lines": True},
                {"caption_islands": True, "light_caption_temporal": True},
                {"caption_islands": "yes"},
            ):
                with (
                    self.subTest(opts=opts),
                    mock.patch("hardsub_extractor.gui.acquire_local_embedded_text") as soft,
                ):
                    with self.assertRaises(ValueError):
                        gui._worker_local("video.mkv", root / "out", False, **opts)
                    soft.assert_not_called()

    def test_gui_light_caption_timing_passed_to_local_ocr(self):
        from hardsub_extractor.ocr.light_text import LightCaptionEngine
        from hardsub_extractor.ocr.light_temporal import LightTemporalWitnessEngine

        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            source = root / "local.mkv"
            source.write_bytes(b"test-media")
            gui = self.gui(root)
            completed = SimpleNamespace(
                cue_count=2, srt_path=root / "result.srt",
                ass_path=root / "result.ass",
            )
            with (
                mock.patch("hardsub_extractor.gui.acquire_local_embedded_text",
                           return_value=None),
                mock.patch("hardsub_extractor.gui.acquire_local_sidecar_text",
                           return_value=None),
                mock.patch("hardsub_extractor.gui.acquire_local_embedded_image",
                           return_value=None),
                mock.patch("hardsub_extractor.gui.bundled_model_dir",
                           return_value=root),
                mock.patch("hardsub_extractor.gui.RapidOCRAdapter",
                           return_value=mock.Mock()),
                mock.patch("hardsub_extractor.gui.extract_local_video",
                           return_value=completed) as visual,
            ):
                gui._worker_local(
                    str(source), root / "result", False,
                    light_caption_temporal=True,
                )
            visual.assert_called_once()
            wrapped = visual.call_args.args[1]
            self.assertIsInstance(wrapped, LightTemporalWitnessEngine)
            self.assertIsInstance(wrapped.inner, LightCaptionEngine)
            self.assertIs(visual.call_args.kwargs["timing_refiner"].func.__self__, wrapped)
            self.assertTrue(visual.call_args.kwargs["streaming"])
            self.assertIn(("done", (str(completed.srt_path), str(completed.ass_path))),
                          self.messages(gui))

    def test_local_embedded_soft_avoids_ocr_and_network(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            gui = self.gui(root)
            result = SimpleNamespace(
                mode="local-embedded-text", cue_count=2,
                srt_path=root / "embedded.srt", ass_path=root / "embedded.ass",
                provenance_path=root / "embedded.source.json",
            )
            with mock.patch(
                "hardsub_extractor.gui.acquire_local_embedded_text",
                return_value=result,
            ) as soft, mock.patch("hardsub_extractor.gui.RapidOCRAdapter") as ocr, mock.patch(
                "hardsub_extractor.gui.AutoEpSiteClient"
            ) as network, mock.patch("hardsub_extractor.gui.extract_local_video") as hard:
                gui._worker("local", "source.mkv", None, root / "out", False, ocr_language="en")
            soft.assert_called_once_with("source.mkv", root / "out", language="en")
            ocr.assert_not_called()
            hard.assert_not_called()
            network.assert_not_called()
            self.assertIn(("done", (str(result.srt_path), str(result.ass_path))),
                          self.messages(gui))

    def test_local_soft_probe_failure_does_not_falsely_start_ocr(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            gui = self.gui(root)
            with mock.patch(
                "hardsub_extractor.gui.acquire_local_embedded_text",
                side_effect=ValueError("untrusted embedded inventory"),
            ), mock.patch("hardsub_extractor.gui.RapidOCRAdapter") as ocr:
                gui._worker("local", "source.mkv", None, root / "out", False)
            ocr.assert_not_called()
            self.assertFalse(any(kind == "done" for kind, _ in self.messages(gui)))
            self.assertTrue(any(kind == "error" for kind, _ in self.messages(gui)))

    def test_local_reserved_provenance_is_rejected_in_preflight(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            source = root / "한글 영상.mkv"
            source.write_bytes(b"fixture")
            gui = self.gui(root)
            out = root / "output" / "subtitles.source.json"
            out.parent.mkdir()
            out.write_text("DO NOT REPLACE", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "이미"):
                gui._validate()
            self.assertEqual(out.read_text("utf-8"), "DO NOT REPLACE")

    def test_local_worker_failure_is_error_without_done(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            source = root / "source.mkv"
            source.write_bytes(b"synthetic-existing-video")
            gui = self.gui(root)
            with mock.patch("hardsub_extractor.gui.acquire_local_embedded_text", return_value=None), mock.patch("hardsub_extractor.gui.acquire_local_sidecar_text", return_value=None), mock.patch("hardsub_extractor.gui.acquire_local_embedded_image", return_value=None), mock.patch("hardsub_extractor.gui.bundled_model_dir", return_value=root / "missing"):
                gui._worker("local", str(source), None, root / "out", False)
            messages = self.messages(gui)
            self.assertFalse(any(kind == "done" for kind, _ in messages))
            error = next(value for kind, value in messages if kind == "error")
            self.assertIn("모델", error["summary"])

    def test_zero_cues_is_not_reported_as_success(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            gui = self.gui(root)
            result = SimpleNamespace(cue_count=0)
            with mock.patch("hardsub_extractor.gui.acquire_local_embedded_text", return_value=None), mock.patch("hardsub_extractor.gui.bundled_model_dir", return_value=root), mock.patch(
                "hardsub_extractor.gui.RapidOCRAdapter"
            ), mock.patch("hardsub_extractor.gui.AdaptiveOCREngine"), mock.patch(
                "hardsub_extractor.gui.extract_local_video", return_value=result
            ):
                gui._worker("local", "source.mkv", None, root / "out", False)
            messages = self.messages(gui)
            self.assertFalse(any(kind == "done" for kind, _ in messages))
            self.assertTrue(any(kind == "error" for kind, _ in messages))

    def test_partial_embedded_bitmap_is_not_claimed_complete(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            gui = self.gui(root)
            result = SimpleNamespace(
                mode="local-embedded-image-ocr", cue_count=1,
                missed_events=2, provenance_path=root / "bitmap.source.json",
                srt_path=root / "bitmap.srt", ass_path=root / "bitmap.ass",
            )
            with (
                mock.patch("hardsub_extractor.gui.acquire_local_embedded_text",
                           return_value=None),
                mock.patch("hardsub_extractor.gui.acquire_local_sidecar_text",
                           return_value=None),
                mock.patch("hardsub_extractor.gui.acquire_local_embedded_image",
                           return_value=result),
                mock.patch("hardsub_extractor.gui.RapidOCRAdapter") as model,
            ):
                gui._worker("local", "video.mkv", None, root / "out", False)
            model.assert_not_called()
            messages = self.messages(gui)
            self.assertIn(("partial_done", (
                str(result.srt_path), str(result.ass_path), 2,
            )), messages)
            self.assertFalse(any(kind == "done" for kind, _ in messages))

    def test_partial_native_timing_is_not_claimed_complete(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            gui = self.gui(root)
            report = SimpleNamespace(
                detected_intervals=3, recognized_intervals=2,
                missed_intervals=1, ocr_calls=10,
                srt_path=root / "native.srt", ass_path=root / "native.ass",
            )
            with (
                mock.patch("hardsub_extractor.gui.acquire_local_embedded_text", return_value=None),
                mock.patch("hardsub_extractor.gui.acquire_local_sidecar_text", return_value=None),
                mock.patch("hardsub_extractor.gui.acquire_local_embedded_image", return_value=None),
                mock.patch("hardsub_extractor.gui.bundled_model_dir", return_value=root),
                mock.patch("hardsub_extractor.native_vsf_assets.ensure_official_vsf",
                           return_value=(root / "native.exe", "d"*64)),
                mock.patch("hardsub_extractor.native_vsf_timing.extract_with_native_vsf",
                           return_value=SimpleNamespace(report=report)),
            ):
                # _worker() is the thread dispatcher; the opt-in keyword
                # belongs on its _worker_local() backend entrypoint.
                gui._worker_local(
                    "video.mkv", root / "out", False,
                    native_vsf_opt_in=True,
                )
            messages = self.messages(gui)
            self.assertIn(("partial_done", (
                str(report.srt_path), str(report.ass_path), 1,
            )), messages)
            self.assertFalse(any(kind == "done" for kind, _ in messages))

    def test_partial_terminal_event_sets_warning_not_success(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            gui = self.gui(root)
            gui._pending_close = False
            gui.progress_bar = mock.Mock()
            gui.cancel_button = mock.Mock()
            gui.start_button = mock.Mock()
            gui.root = mock.Mock()
            gui.status_var = mock.Mock()
            gui._log = mock.Mock()
            gui.events.put(("partial_done", ("subtitle.srt", "subtitle.ass", 2)))
            with mock.patch("hardsub_extractor.gui.messagebox.showwarning") as warn:
                gui._poll_events()
            gui.status_var.set.assert_called_with("일부 미확보")
            warn.assert_called_once()
            self.assertIn("2개", warn.call_args.args[1])
            gui.root.after.assert_called_once()

    def test_native_opt_in_still_uses_embedded_text_without_download_or_ocr(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            gui = self.gui(root)
            gui._active_native_vsf_opt_in = True
            source = SimpleNamespace(
                mode="local-embedded-text", cue_count=2,
                srt_path=root / "original.srt", ass_path=root / "original.ass",
                provenance_path=root / "original.source.json",
            )
            with (
                mock.patch("hardsub_extractor.gui.acquire_local_embedded_text",
                           return_value=source),
                mock.patch("hardsub_extractor.native_vsf_assets.ensure_official_vsf") as fetch,
                mock.patch("hardsub_extractor.native_vsf_timing.extract_with_native_vsf") as native,
                mock.patch("hardsub_extractor.gui.RapidOCRAdapter") as recognizer,
            ):
                gui._worker("local", "episode.mkv", None, root / "out", False)
            fetch.assert_not_called()
            native.assert_not_called()
            recognizer.assert_not_called()
            self.assertIn(("done", (str(source.srt_path), str(source.ass_path))),
                          self.messages(gui))

    def test_native_opt_in_uses_official_cache_after_original_absence(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            gui = self.gui(root)
            gui._active_native_vsf_opt_in = True
            result = SimpleNamespace(
                detected_intervals=2, recognized_intervals=2,
                missed_intervals=0, ocr_calls=10,
                srt_path=root / "out.srt", ass_path=root / "out.ass",
            )
            with (
                mock.patch("hardsub_extractor.gui.acquire_local_embedded_text", return_value=None),
                mock.patch("hardsub_extractor.gui.acquire_local_sidecar_text", return_value=None),
                mock.patch("hardsub_extractor.gui.acquire_local_embedded_image", return_value=None),
                mock.patch("hardsub_extractor.gui.bundled_model_dir", return_value=root),
                mock.patch("hardsub_extractor.native_vsf_assets.ensure_official_vsf",
                           return_value=(root / "official.exe", "a"*64)) as fetch,
                mock.patch("hardsub_extractor.native_vsf_timing.extract_with_native_vsf",
                           return_value=SimpleNamespace(report=result)) as native,
                mock.patch("hardsub_extractor.gui.RapidOCRAdapter") as recognizer,
            ):
                gui._worker("local", "episode.mkv", None, root / "out", False)
            fetch.assert_called_once_with(consent=True, check_cancelled=None)
            self.assertEqual(native.call_args.kwargs["pinned_executable_sha256"], "a"*64)
            self.assertEqual(native.call_args.kwargs["executable"], root / "official.exe")
            recognizer.assert_not_called()
            self.assertIn(("done", (str(result.srt_path), str(result.ass_path))),
                          self.messages(gui))

    def test_declined_native_download_never_starts_background_worker(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            gui = self.gui(root)
            gui.worker = None
            gui._pending_close = False
            gui.native_vsf_var = mock.Mock()
            gui.native_vsf_var.get.return_value = True
            with (
                mock.patch.object(gui, "_validate", return_value=(
                    "local", "episode.mkv", None, root / "out",
                )),
                mock.patch("hardsub_extractor.gui.messagebox.askyesno",
                           return_value=False) as approval,
                mock.patch("hardsub_extractor.gui.threading.Thread") as thread,
            ):
                gui._start()
            approval.assert_called_once()
            thread.assert_not_called()
            self.assertFalse(getattr(gui, "_active_native_vsf_opt_in", False))

    def test_epsite_worker_delegates_resolution_to_central_extractor(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            gui = self.gui(root, "epsite")
            client = mock.Mock(transport_name="urllib")
            result = SimpleNamespace(mode="soft", srt_path=root / "out.srt", ass_path=root / "out.ass")
            with mock.patch("hardsub_extractor.gui.AutoEpSiteClient", return_value=client), mock.patch(
                "hardsub_extractor.gui.extract_epsite", return_value=result
            ) as extract, mock.patch("hardsub_extractor.gui.RapidOCRAdapter") as ocr:
                gui._worker("epsite", "https://episode.example/e/x", None, root / "out", False)

            client.resolve.assert_not_called()
            extract.assert_called_once()
            args = extract.call_args.args
            kwargs = extract.call_args.kwargs
            self.assertEqual(args, ("https://episode.example/e/x", root / "out"))
            self.assertIs(kwargs["client"], client)
            self.assertIsNone(kwargs["ocr"])
            self.assertTrue(callable(kwargs["ocr_factory"]))
            self.assertIsNone(kwargs["diagnostics"])
            self.assertTrue(kwargs["streaming"])
            ocr.assert_not_called()
            self.assertTrue(any(kind == "done" for kind, _ in self.messages(gui)))


if __name__ == "__main__":
    unittest.main()
