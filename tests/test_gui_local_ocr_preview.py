"""Headless optional pre-extraction GUI OCR preview integration.

No real window is created, no network/model is run, no original media touched.
"""
from __future__ import annotations

from pathlib import Path
import queue
import tempfile
import threading
import unittest
from unittest import mock

import numpy as np

from hardsub_extractor import gui as gui_module
from hardsub_extractor.local_ocr_preview import LocalOCRPreview


class Var:
    def __init__(self, value):
        self.value = value

    def get(self):
        return self.value

    def set(self, value):
        self.value = value


def fake_preview_gui(video: Path):
    gui = gui_module.ExtractorGUI.__new__(gui_module.ExtractorGUI)
    gui.source_mode_var = Var("local")
    gui.url_var = Var(str(video))
    gui.ocr_language_var = Var("ja")
    gui.preview_time_var = Var("00:01:02")
    gui.caption_islands_var = Var(False)
    gui.status_var = Var("준비됨")
    gui.worker = None
    gui._pending_close = False
    gui._cancel_event = threading.Event()
    gui.events = queue.Queue()
    gui.start_button = mock.Mock()
    gui.cancel_button = mock.Mock()
    gui.ocr_preview_button = mock.Mock()
    gui.ocr_compare_button = mock.Mock()
    gui.review_button = mock.Mock()
    gui.progress_bar = mock.Mock()
    gui.root = mock.Mock()
    gui._log = mock.Mock()
    return gui


class GuiLocalOCRPreviewTests(unittest.TestCase):
    def test_one_frame_preview_freezes_real_crop_with_gui_changes(self):
        with tempfile.TemporaryDirectory() as td:
            movie = Path(td) / "movie.mp4"
            movie.write_bytes(b"private original")
            gui = fake_preview_gui(movie)
            gui.local_ocr_region_var = Var("좁게 · 하단 25% (윗줄 유실 가능)")
            with mock.patch.object(gui_module.threading, "Thread") as thread:
                gui._start_local_ocr_preview()
            self.assertEqual(thread.call_args.kwargs["args"], (movie, 62., "ja"))
            self.assertEqual(
                thread.call_args.kwargs["kwargs"],
                {"region_mode": "bottom25", "inference_engine": "onnxruntime"},
            )
            gui.local_ocr_region_var.set("넓게 · 하단 80% (오검출 주의)")
            self.assertEqual(
                thread.call_args.kwargs["kwargs"],
                {"region_mode": "bottom25", "inference_engine": "onnxruntime"},
            )
            self.assertEqual(movie.read_bytes(), b"private original")

    def test_invalid_custom_roi_never_starts_worker(self):
        with tempfile.TemporaryDirectory() as td:
            movie = Path(td) / "movie.mp4"
            movie.write_bytes(b"original")
            gui = fake_preview_gui(movie)
            gui.local_ocr_region_var = Var("untrusted option")
            with (mock.patch.object(gui_module.threading, "Thread") as thread,
                  mock.patch.object(gui_module.messagebox, "showerror")):
                gui._start_local_ocr_preview()
            thread.assert_not_called()

    def test_explicit_click_snapshots_source_second_language_one_worker(self):
        with tempfile.TemporaryDirectory() as td:
            video = Path(td) / "original.mkv"
            video.write_bytes(b"private video unchanged")
            gui = fake_preview_gui(video)
            with mock.patch.object(gui_module.threading, "Thread") as thread:
                gui._start_local_ocr_preview()
            thread.assert_called_once()
            self.assertEqual(
                thread.call_args.kwargs["args"], (video, 62.0, "ja"),
            )
            self.assertIs(
                thread.call_args.kwargs["target"].__self__, gui,
            )
            self.assertEqual(
                thread.call_args.kwargs["target"].__name__,
                "_local_ocr_preview_worker",
            )
            thread.return_value.start.assert_called_once()
            self.assertEqual(video.read_bytes(), b"private video unchanged")
            self.assertEqual(list(Path(td).iterdir()), [video])

    def test_explicit_one_click_ab_compare_snapshots_one_local_frame(self):
        with tempfile.TemporaryDirectory() as td:
            movie = Path(td) / "video.mkv"
            movie.write_bytes(b"private original")
            gui = fake_preview_gui(movie)
            with mock.patch.object(gui_module.threading, "Thread") as worker:
                gui._start_local_ocr_preview(compare_bright=True)
            self.assertEqual(worker.call_args.kwargs["args"],
                             (movie, 62., "ja", "compare-bright"))
            gui.ocr_compare_button.configure.assert_any_call(state="disabled")
            self.assertFalse(gui.caption_islands_var.get())
            self.assertEqual(movie.read_bytes(), b"private original")

    def test_comparison_worker_reuses_same_adapter_for_both_ocr_modes(self):
        from hardsub_extractor.ocr.adaptive import AdaptiveOCREngine
        from hardsub_extractor.ocr.light_text import LightCaptionEngine
        with tempfile.TemporaryDirectory() as td:
            movie = Path(td) / "video.mkv"
            movie.write_bytes(b"original")
            models = Path(td) / "models"
            models.mkdir()
            manifest = Path(td) / "manifest.json"
            manifest.write_text("{}", encoding="utf-8")
            gui = fake_preview_gui(movie)
            result = LocalOCRPreview(
                np.zeros((10,10,3), np.uint8), 62., 62.,
                "private all", .90, "compare-bright", "bottom40",
                "private bright", .91,
            )
            def inspect(source, second, make_ocr, *, check_cancelled,
                        cancel_requested, preview_mode, compare_make_ocr):
                self.assertEqual(preview_mode, "compare-bright")
                a = make_ocr()
                b = compare_make_ocr()
                self.assertIsInstance(a, AdaptiveOCREngine)
                self.assertIsInstance(b, LightCaptionEngine)
                self.assertIsInstance(b.inner, AdaptiveOCREngine)
                self.assertIs(a.base, b.inner.base)
                check_cancelled()
                return result
            with (
                mock.patch.object(gui_module, "bundled_model_dir", return_value=models),
                mock.patch.object(gui_module, "bundled_manifest_path", return_value=manifest),
                mock.patch.object(gui_module, "RapidOCRAdapter") as adapter,
                mock.patch("hardsub_extractor.local_ocr_preview.inspect_local_ocr_preview",
                           side_effect=inspect),
            ):
                gui._local_ocr_preview_worker(movie, 62., "ja", "compare-bright")
            adapter.assert_called_once()
            self.assertEqual(gui.events.get_nowait(), ("local_ocr_preview_ready", result))
            self.assertTrue(gui.events.empty())
            self.assertNotIn("private bright", repr(gui._log.call_args_list))
            self.assertEqual(movie.read_bytes(), b"original")

    def test_preview_click_snapshots_selected_experimental_style(self):
        with tempfile.TemporaryDirectory() as td:
            video = Path(td) / "original.mkv"
            video.write_bytes(b"owned video")
            for choice, selected in (
                ("focus", "bottom-two-lines"),
                ("light", "light-caption"),
                ("islands", "caption-islands"),
            ):
                with self.subTest(choice=choice):
                    gui = fake_preview_gui(video)
                    gui.focus_bottom_two_var = Var(choice == "focus")
                    gui.light_caption_temporal_var = Var(choice == "light")
                    gui.caption_islands_var = Var(choice == "islands")
                    gui.native_vsf_var = Var(False)
                    with mock.patch.object(gui_module.threading, "Thread") as thread:
                        gui._start_local_ocr_preview()
                    self.assertEqual(
                        thread.call_args.kwargs["args"],
                        (video, 62.0, "ja", selected),
                    )
                    thread.return_value.start.assert_called_once()
            self.assertEqual(video.read_bytes(), b"owned video")

    def test_preview_rejects_incompatible_filter_options_before_work(self):
        with tempfile.TemporaryDirectory() as td:
            video = Path(td) / "source.mkv"
            video.write_bytes(b"original")
            for focus, light, native in (
                (True, True, False), (True, False, True), (False, True, True),
            ):
                gui = fake_preview_gui(video)
                gui.focus_bottom_two_var = Var(focus)
                gui.light_caption_temporal_var = Var(light)
                gui.native_vsf_var = Var(native)
                with (
                    self.subTest(focus=focus, light=light, native=native),
                    mock.patch.object(gui_module.threading, "Thread") as thread,
                    mock.patch.object(gui_module.messagebox, "showerror") as error,
                ):
                    gui._start_local_ocr_preview()
                    thread.assert_not_called()
                    error.assert_called_once()
            self.assertEqual(video.read_bytes(), b"original")

    def test_preview_factory_matches_selected_visual_engine_without_new_model(self):
        from hardsub_extractor.ocr.adaptive import AdaptiveOCREngine
        from hardsub_extractor.ocr.line_focus import BottomTwoLineEngine
        from hardsub_extractor.ocr.light_text import LightCaptionEngine
        from hardsub_extractor.ocr.caption_islands import CaptionIslandEngine

        with tempfile.TemporaryDirectory() as td:
            video = Path(td) / "source.mkv"
            video.write_bytes(b"private")
            models = Path(td) / "models"
            models.mkdir()
            manifest = Path(td) / "manifest.json"
            manifest.write_text("{}", encoding="utf-8")
            for selected, engine_type in (
                ("bottom-two-lines", BottomTwoLineEngine),
                ("light-caption", LightCaptionEngine),
                ("caption-islands", CaptionIslandEngine),
            ):
                gui = fake_preview_gui(video)
                result = LocalOCRPreview(
                    np.zeros((5, 10, 3), np.uint8),
                    62., 62., "private text", .7, selected,
                )
                def inspect(source, second, make_ocr, *,
                            check_cancelled, cancel_requested, preview_mode):
                    self.assertEqual(preview_mode, selected)
                    chosen = make_ocr()
                    self.assertIsInstance(chosen, engine_type)
                    wrapped = (
                        chosen.base if selected == "bottom-two-lines"
                        else chosen.engine if selected == "caption-islands"
                        else chosen.inner
                    )
                    self.assertIsInstance(wrapped, AdaptiveOCREngine)
                    return result
                with (
                    mock.patch.object(gui_module, "bundled_model_dir", return_value=models),
                    mock.patch.object(gui_module, "bundled_manifest_path", return_value=manifest),
                    mock.patch.object(gui_module, "RapidOCRAdapter") as adapter,
                    mock.patch(
                        "hardsub_extractor.local_ocr_preview.inspect_local_ocr_preview",
                        side_effect=inspect,
                    ) as observed,
                    self.subTest(selected=selected),
                ):
                    gui._local_ocr_preview_worker(video, 62., "ja", selected)
                    self.assertEqual(observed.call_count, 1)
                    adapter.assert_called_once()
                self.assertEqual(
                    gui.events.get_nowait(), ("local_ocr_preview_ready", result),
                )
                self.assertTrue(gui.events.empty())
            self.assertEqual(video.read_bytes(), b"private")

    def test_preview_rejects_caption_islands_with_other_modes(self):
        with tempfile.TemporaryDirectory() as td:
            video = Path(td) / "original.mkv"
            video.write_bytes(b"unchanged")
            for incompatible in ("focus", "light", "native"):
                gui = fake_preview_gui(video)
                gui.caption_islands_var.set(True)
                gui.focus_bottom_two_var = Var(incompatible == "focus")
                gui.light_caption_temporal_var = Var(incompatible == "light")
                gui.native_vsf_var = Var(incompatible == "native")
                with (
                    self.subTest(incompatible=incompatible),
                    mock.patch.object(gui_module.threading, "Thread") as worker,
                    mock.patch.object(gui_module.messagebox, "showerror") as error,
                ):
                    gui._start_local_ocr_preview()
                    worker.assert_not_called()
                    error.assert_called_once()
            self.assertEqual(video.read_bytes(), b"unchanged")

    def test_preview_click_is_ignored_during_active_tab_capture(self):
        with tempfile.TemporaryDirectory() as td:
            video = Path(td) / "original.mkv"
            video.write_bytes(b"source")
            gui = fake_preview_gui(video)
            gui._tab_capture_thread = mock.Mock()
            gui._tab_capture_thread.is_alive.return_value = True
            with mock.patch.object(gui_module.threading, "Thread") as thread:
                gui._start_local_ocr_preview()
            thread.assert_not_called()
            self.assertEqual(video.read_bytes(), b"source")

    def test_regular_job_completion_reenables_preview_button(self):
        with tempfile.TemporaryDirectory() as td:
            video = Path(td) / "original.mkv"
            video.write_bytes(b"source")
            gui = fake_preview_gui(video)
            gui.events.put(("error", {
                "summary": "Unable to open input", "traceback": "error type",
            }))
            with mock.patch.object(gui_module.messagebox, "showerror"):
                gui._poll_events()
            gui.ocr_preview_button.configure.assert_any_call(state="normal")

    def test_invalid_source_or_time_never_creates_worker(self):
        with tempfile.TemporaryDirectory() as td:
            video = Path(td) / "original.mkv"
            video.write_bytes(b"source")
            gui = fake_preview_gui(video)
            for mode, time_text in (
                ("captions", "00:00:01"), ("local", "12:60"),
            ):
                gui.source_mode_var.set(mode)
                gui.preview_time_var.set(time_text)
                with (
                    mock.patch.object(gui_module.threading, "Thread") as thread,
                    mock.patch.object(gui_module.messagebox, "showerror") as err,
                ):
                    gui._start_local_ocr_preview()
                thread.assert_not_called()
                err.assert_called_once()
            self.assertEqual(video.read_bytes(), b"source")

    def test_worker_ocr_raw_text_only_in_private_result_event(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            video = root / "original.mkv"
            video.write_bytes(b"source")
            models = root / "models"
            models.mkdir()
            manifest = root / "manifest.json"
            manifest.write_text("{}", encoding="utf-8")
            gui = fake_preview_gui(video)
            result = LocalOCRPreview(
                np.zeros((4, 8, 3), np.uint8),
                62.0, 62.0, "SECRET_私用日本語", 0.78,
            )
            def inspect(
                source, second, make_ocr, *, check_cancelled,
                cancel_requested,
            ):
                check_cancelled()
                self.assertFalse(cancel_requested())
                make_ocr()
                return result
            with (
                mock.patch.object(gui_module, "bundled_model_dir",
                                  return_value=models),
                mock.patch.object(gui_module, "bundled_manifest_path",
                                  return_value=manifest),
                mock.patch.object(gui_module, "RapidOCRAdapter") as adapter,
                mock.patch(
                    "hardsub_extractor.local_ocr_preview.inspect_local_ocr_preview",
                    side_effect=inspect,
                ) as called,
            ):
                gui._local_ocr_preview_worker(video, 62, "ja")
            called.assert_called_once()
            self.assertEqual(adapter.call_args.kwargs["language"], "ja")
            self.assertEqual(
                gui.events.get_nowait(), ("local_ocr_preview_ready", result),
            )
            self.assertTrue(gui.events.empty())
            self.assertNotIn("SECRET_", repr(gui._log.call_args_list))
            self.assertEqual(video.read_bytes(), b"source")

    def test_worker_exception_only_emits_type_not_private_text(self):
        with tempfile.TemporaryDirectory() as td:
            video = Path(td) / "private-s1.mkv"
            video.write_bytes(b"source")
            gui = fake_preview_gui(video)
            with mock.patch(
                "hardsub_extractor.local_ocr_preview.inspect_local_ocr_preview",
                side_effect=RuntimeError("private-s1.mkv SECRET_字幕"),
            ):
                gui._local_ocr_preview_worker(video, 20, "ja")
            self.assertEqual(
                gui.events.get_nowait(), ("local_ocr_preview_error", "RuntimeError"),
            )
            self.assertTrue(gui.events.empty())

    def test_gui_event_displays_in_main_thread_and_keeps_polling_after_failure(self):
        with tempfile.TemporaryDirectory() as td:
            video = Path(td) / "original.mkv"
            video.write_bytes(b"source")
            gui = fake_preview_gui(video)
            result = LocalOCRPreview(
                np.zeros((4, 8, 3), np.uint8), 10., 10.,
                "not-for-log", .9,
            )
            gui.events.put(("local_ocr_preview_ready", result))
            with (
                mock.patch.object(gui, "_show_local_ocr_preview",
                    side_effect=RuntimeError("bad dialog SECRET_字幕")) as show,
                mock.patch.object(gui_module.messagebox, "showerror") as error,
            ):
                gui._poll_events()
            show.assert_called_once_with(result)
            error.assert_called_once()
            gui.root.after.assert_called_once()
            self.assertNotIn("SECRET_字幕", repr(gui._log.call_args_list))
            self.assertNotIn("not-for-log", repr(gui._log.call_args_list))
            self.assertEqual(gui.status_var.get(), "미리보기 화면 표시 실패")


if __name__ == "__main__":
    unittest.main()
