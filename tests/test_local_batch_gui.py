"""Local video batch GUI contracts: source selection, lazy model, privacy.

No actual Tk root, video, browser, network, installer or user media required.
The source-list is the same bounded module used by the worker and the GUI.
"""
from __future__ import annotations

from pathlib import Path
import queue
import tempfile
import threading
import unittest
from unittest import mock

from hardsub_extractor.gui import ExtractorGUI
from hardsub_extractor.local_batch import LocalBatchItem, LocalBatchResult


class Var:
    def __init__(self, value=""):
        self.value = value

    def get(self):
        return self.value

    def set(self, value):
        self.value = value


class LocalBatchGuiTests(unittest.TestCase):
    def gui(self, root):
        obj = ExtractorGUI.__new__(ExtractorGUI)
        obj.source_mode_var = Var("local-batch")
        obj.url_var = Var("")
        obj.target_var = Var("old-hq-video-must-not-be-used")
        obj.sidecar_var = Var("stale-subtitle-must-not-be-used")
        obj.out_var = Var(str(root / "results"))
        obj.ocr_language_var = Var("ja")
        obj.events = queue.Queue()
        obj._cancel_event = threading.Event()
        obj._pending_close = False
        obj._progress_stage = None
        return obj

    def event(self, gui, name):
        return [v for k, v in list(gui.events.queue) if k == name]

    def test_local_batch_txt_preflight_and_output_directory(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            clip = root / "第01話.mkv"
            clip.write_bytes(b"not decoded in GUI test")
            lst = root / "videos.txt"
            lst.write_text(clip.name + "\n", encoding="utf-8")
            gui = self.gui(root)
            gui.url_var.set(str(lst))
            mode, source, target, out = gui._validate()
            self.assertEqual((mode, source, target), ("local-batch", str(lst), None))
            self.assertEqual(out, root / "results")
            self.assertEqual(gui._requested_ocr_language, "ja")
            self.assertFalse(out.exists())

    def test_one_bad_entry_blocks_start_before_creating_outputs(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root / "one.mp4").write_bytes(b"safe")
            file = root / "videos.txt"
            file.write_text("one.mp4\nmissing.mp4\n", "utf-8")
            gui = self.gui(root)
            gui.url_var.set(str(file))
            with self.assertRaises(ValueError):
                gui._validate()
            self.assertFalse((root / "results").exists())

    def test_local_batch_mode_clears_stale_target_sidecar_and_hides_both(self):
        with tempfile.TemporaryDirectory() as td:
            gui = self.gui(Path(td))
            for field in (
                "source_label", "source_picker", "target_label",
                "target_entry", "target_picker", "sidecar_label",
                "sidecar_entry", "sidecar_picker", "out_label",
                "languages_entry", "proxy_entry", "video_ocr_check",
                "ocr_language_entry",
            ):
                setattr(gui, field, mock.Mock())
            gui._source_mode_changed()
            self.assertEqual(gui.url_var.get(), "")
            self.assertEqual(gui.target_var.get(), "")
            self.assertEqual(gui.sidecar_var.get(), "")
            gui.source_picker.configure.assert_called_once_with(
                state="normal", text="영상 여러 개 선택",
            )
            self.assertEqual(gui.sidecar_entry.grid_remove.call_count, 1)
            self.assertEqual(gui.target_entry.grid_remove.call_count, 1)
            gui.out_label.configure.assert_called_once_with(
                text="자막 저장 폴더 (기존 결과 보존)"
            )
            gui.languages_entry.configure.assert_called_once_with(
                state="disabled",
            )
            gui.ocr_language_entry.configure.assert_called_once_with(
                state="readonly",
            )

    def test_explorer_multi_select_needs_no_manual_list_and_shows_count_only(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            videos = [root / "제01화.mkv", root / "제02화.mp4"]
            for item in videos:
                item.write_bytes(b"media not decoded in contract")
            gui = self.gui(root)
            with mock.patch(
                "hardsub_extractor.gui.filedialog.askopenfilenames",
                return_value=tuple(str(item) for item in videos),
            ) as picker, mock.patch(
                "hardsub_extractor.gui.filedialog.askopenfilename",
                side_effect=AssertionError("manual path list should not open"),
            ):
                gui._pick_source()
            self.assertIn("*.mkv", picker.call_args.kwargs["filetypes"][0][1])
            self.assertEqual(len(gui._local_batch_selection), 2)
            display = gui.url_var.get()
            self.assertIn("2개", display)
            self.assertNotIn(str(root), display)
            self.assertNotIn(videos[0].stem, display)
            mode, source, target, out = gui._validate()
            self.assertEqual((mode, source, target),
                             ("local-batch", display, None))
            self.assertEqual(out, root / "results")
            self.assertEqual(gui._requested_local_batch_paths,
                             tuple(str(item) for item in videos))
            self.assertFalse(out.exists())
            with mock.patch("hardsub_extractor.gui.filedialog.askdirectory",
                            return_value=td):
                gui._pick_output()
            self.assertEqual(gui.out_var.get(), td)

    def test_existing_utf8_text_list_remains_supported_without_dialog(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            video = root / "유니코드 1편.mkv"
            video.write_bytes(b"sample")
            lst = root / "my-videos.txt"
            lst.write_text(video.name + "\n", encoding="utf-8")
            gui = self.gui(root)
            gui.url_var.set(str(lst))
            mode, source, _, _ = gui._validate()
            self.assertEqual((mode, source), ("local-batch", str(lst)))
            self.assertEqual(gui._requested_local_batch_paths, ())

    def test_picker_cancel_and_invalid_51_video_selection_are_inert(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            gui = self.gui(root)
            gui.url_var.set("original-text")
            with mock.patch("hardsub_extractor.gui.filedialog.askopenfilenames",
                            return_value=()):
                gui._pick_source()
            self.assertEqual(gui.url_var.get(), "original-text")
            videos = [root / f"episode{i:02}.mkv" for i in range(51)]
            for item in videos:
                item.write_bytes(b"sample")
            with mock.patch(
                "hardsub_extractor.gui.filedialog.askopenfilenames",
                return_value=tuple(map(str, videos)),
            ), mock.patch("hardsub_extractor.gui.messagebox.showerror") as error:
                gui._pick_source()
            error.assert_called_once()
            self.assertEqual(gui.url_var.get(), "original-text")
            self.assertEqual(getattr(gui, "_local_batch_selection", ()), ())
            self.assertFalse((root / "results").exists())

    def test_typed_text_invalidates_selected_batch_display(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            item = root / "episode.mkv"
            item.write_bytes(b"sample")
            gui = self.gui(root)
            with mock.patch(
                "hardsub_extractor.gui.filedialog.askopenfilenames",
                return_value=(str(item),),
            ):
                gui._pick_source()
            gui.url_var.set("선택한 동영상 50개 (목록 파일 불필요)")
            with self.assertRaises(ValueError):
                gui._validate()
            self.assertFalse((root / "results").exists())

    def test_start_captures_immutable_selection_before_thread_launch(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            a = root / "a.mkv"
            b = root / "b.mkv"
            a.write_bytes(b"sample")
            b.write_bytes(b"sample")
            gui = self.gui(root)
            gui.worker = None
            gui.start_button = mock.Mock()
            gui.cancel_button = mock.Mock()
            gui.progress_bar = mock.Mock()
            gui.status_var = Var("ready")
            gui.diag_var = Var(False)
            gui._log = mock.Mock()
            with mock.patch(
                "hardsub_extractor.gui.filedialog.askopenfilenames",
                return_value=(str(a), str(b)),
            ):
                gui._pick_source()
            with mock.patch(
                "hardsub_extractor.gui.threading.Thread",
            ) as thread:
                gui._start()
            thread.assert_called_once()
            captured = thread.call_args.kwargs["args"]
            self.assertEqual(captured[0], "local-batch")
            self.assertEqual(captured[-1], (str(a), str(b)))
            gui._local_batch_selection = ()
            gui.url_var.set("different input edited while running")
            self.assertEqual(captured[-1], (str(a), str(b)))
            with (
                mock.patch("hardsub_extractor.gui.load_local_batch",
                           return_value=[]) as load,
                mock.patch("hardsub_extractor.gui.run_local_batch", return_value=[]),
            ):
                gui._worker(
                    "local-batch", captured[1], None, captured[3],
                    False, None, "ja", captured[-1],
                )
            load.assert_called_once_with((str(a), str(b)), None, root / "results")
            self.assertNotIn(str(a), repr(list(gui.events.queue)))

    def test_dispatch_runs_local_batch_not_site_client(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            gui = self.gui(root)
            result = [
                LocalBatchResult(1, "local-embedded-text", 2),
                LocalBatchResult(2, "hard_ocr", 3),
                LocalBatchResult(3, "existing_verified", 4),
                LocalBatchResult(4, "existing_preserved", 0),
                LocalBatchResult(5, "failed", 0, "RuntimeError"),
            ]
            # Production post-export review needs source and output identities.
            # Plain object() previously raised AttributeError before captions_done.
            fake_items = [
                LocalBatchItem(root / f'episode{i}.mkv', root / f'output{i}')
                for i in range(1, len(result) + 1)
            ]
            def runner(items, **kwargs):
                self.assertEqual(items, fake_items)
                for item in result:
                    kwargs["on_result"](item)
                return result
            with (
                mock.patch("hardsub_extractor.gui.load_local_batch",
                           return_value=fake_items) as loader,
                mock.patch("hardsub_extractor.gui.run_local_batch",
                           side_effect=runner) as batch,
                mock.patch("hardsub_extractor.gui.AutoEpSiteClient") as anime,
                mock.patch("hardsub_extractor.gui.PublicCaptionCollector") as web,
                mock.patch("hardsub_extractor.gui.RapidOCRAdapter") as model,
                mock.patch(
                    "hardsub_extractor.local_ocr_quality_signals.local_ocr_quality_signals",
                    return_value={"review_signal_count": 0},
                ) as review,
            ):
                gui._worker(
                    "local-batch", str(root / "episodes.txt"),
                    None, root / "result", False, None, "ja",
                )
            loader.assert_called_once()
            batch.assert_called_once()
            anime.assert_not_called()
            web.assert_not_called()
            model.assert_not_called()
            review.assert_called_once_with(
                fake_items[1].source, fake_items[1].out_base.with_suffix(".srt"),
                language="ja",
            )
            self.assertEqual(self.event(gui, "caption_progress"),
                             [(0, 5), (1, 5), (2, 5), (3, 5), (4, 5), (5, 5)])
            self.assertEqual(self.event(gui, "captions_done"),
                             [(2, 0, 1, 1, str(root / "result"))])
            self.assertEqual(self.event(gui, "caption_verified"), [1])
            self.assertNotIn("episodes.txt", repr(list(gui.events.queue)))
            self.assertNotIn("RuntimeError", repr(list(gui.events.queue)))

    def test_batch_opt_in_lazy_native_factory_and_partial_count_in_log(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            gui = self.gui(root)
            gui._active_native_vsf_opt_in = True
            fake = LocalBatchResult(1, "hard_ocr_native_timing_partial", 2, missed_events=1)
            def run(items, **kw):
                self.assertTrue(callable(kw["native_timing_factory"]))
                self.assertFalse(fetch.called)
                self.assertEqual(kw["native_timing_factory"](), (root/"official.exe", "a"*64))
                kw["on_result"](fake)
                return [fake]
            with (
                mock.patch("hardsub_extractor.gui.load_local_batch", return_value=[
                    LocalBatchItem(root / "episode.mkv", root / "output")
                ]),
                mock.patch("hardsub_extractor.gui.run_local_batch", side_effect=run),
                mock.patch("hardsub_extractor.native_vsf_assets.ensure_official_vsf",
                           return_value=(root/"official.exe", "a"*64)) as fetch,
                mock.patch(
                    "hardsub_extractor.local_ocr_quality_signals.local_ocr_quality_signals",
                    return_value={"review_signal_count": 0},
                ) as review,
            ):
                gui._worker("local-batch", str(root/"items.txt"), None,
                            root/"output", False, None, "ja")
            fetch.assert_called_once()
            self.assertIs(fetch.call_args.kwargs["consent"], True)
            self.assertTrue(callable(fetch.call_args.kwargs["check_cancelled"]))
            self.assertEqual(self.event(gui, "caption_progress"), [(0, 1), (1, 1)])
            review.assert_called_once_with(
                root / "episode.mkv", (root / "output").with_suffix(".srt"),
                language="ja",
            )
            self.assertIn("미인식 구간 1개", repr(self.event(gui, "log")))
            self.assertEqual(self.event(gui, "captions_done"),
                             [(0, 1, 0, 0, str(root/"output"))])

    def test_batch_native_option_enabled_but_rejected_outside_local_modes(self):
        with tempfile.TemporaryDirectory() as td:
            gui = self.gui(Path(td))
            gui.native_vsf_var = Var(True)
            gui.native_vsf_check = mock.Mock()
            gui.proxy_mode_var = Var("direct")
            for field in ("source_label", "source_picker", "target_label",
                          "target_entry", "target_picker", "sidecar_label",
                          "sidecar_entry", "sidecar_picker", "out_label",
                          "languages_entry", "proxy_entry", "video_ocr_check",
                          "ocr_language_entry"):
                setattr(gui, field, mock.Mock())
            gui._source_mode_changed()
            gui.native_vsf_check.configure.assert_called_with(state="normal")
            self.assertTrue(gui.native_vsf_var.get())
            gui.source_mode_var.set("captions")
            gui._source_mode_changed()
            self.assertFalse(gui.native_vsf_var.get())
            gui.native_vsf_check.configure.assert_called_with(state="disabled")

    def test_model_factory_lazy_and_uses_correct_original_language(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            gui = self.gui(root)
            factory = {}
            def runner(items, **kwargs):
                factory["make"] = kwargs["ocr_factory"]
                return []
            with (
                mock.patch("hardsub_extractor.gui.load_local_batch",
                           return_value=[]),
                mock.patch("hardsub_extractor.gui.run_local_batch",
                           side_effect=runner),
                mock.patch("hardsub_extractor.gui.bundled_model_dir",
                           return_value=root),
                mock.patch("hardsub_extractor.gui.bundled_manifest_path",
                           return_value=root / "manifest.json"),
                mock.patch("hardsub_extractor.gui.RapidOCRAdapter",
                           return_value="native-model") as native,
                mock.patch("hardsub_extractor.gui.AdaptiveOCREngine",
                           return_value="ocr"),
            ):
                gui._worker(
                    "local-batch", str(root / "episodes.txt"),
                    None, root / "out", False, None, "zh",
                )
                native.assert_not_called()
                self.assertEqual(factory["make"](), "ocr")
                self.assertEqual(native.call_args.kwargs["language"], "zh")
                self.assertEqual(native.call_args.kwargs["manifest_path"],
                                 str(root / "manifest.json"))


if __name__ == "__main__":
    unittest.main()
