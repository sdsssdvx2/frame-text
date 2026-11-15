"""GUI caption-first mode contracts; no real Tk, video or network."""
from __future__ import annotations
import queue
import tempfile
import threading
import unittest
from pathlib import Path
from unittest import mock

from hardsub_extractor.gui import ExtractorGUI
from hardsub_extractor.extraction_progress import ExtractionCancelled
from hardsub_extractor.subtitle_acquisition import CaptionAccessError, CaptionResult

YOUTUBE="https://www.youtube.com/watch?v=synthetic"
WETV="https://wetv.vip/en/play/synthetic"


class Var:
    def __init__(self, value):
        self.value=value
    def get(self):
        return self.value
    def set(self, value):
        self.value=value


class CaptionGuiTests(unittest.TestCase):
    def gui(self, root):
        gui=ExtractorGUI.__new__(ExtractorGUI)
        gui.source_mode_var=Var("captions")
        gui.url_var=Var(YOUTUBE)
        gui.target_var=Var("")
        gui.out_var=Var(str(root / "captions"))
        gui.languages_var=Var("ko,en,ja,zh-Hans")
        gui.events=queue.Queue()
        gui._cancel_event=threading.Event()
        gui._pending_close=False
        gui._progress_stage=None
        return gui

    def events(self, gui, kind):
        return [value for label,value in list(gui.events.queue) if label == kind]

    def test_one_episode_languages_validate_without_writing_user_files(self):
        with tempfile.TemporaryDirectory() as td:
            gui=self.gui(Path(td))
            mode,url,batch,root=gui._validate()
            self.assertEqual(mode,"captions")
            self.assertEqual(url,YOUTUBE)
            self.assertIsNone(batch)
            self.assertEqual(gui._requested_languages,("ko","en","ja","zh-hans"))
            self.assertEqual(root, Path(td)/"captions")
            self.assertFalse(root.exists())

    def test_batch_file_only_is_accepted_without_video_path(self):
        with tempfile.TemporaryDirectory() as td:
            gui=self.gui(Path(td))
            gui.url_var=Var("")
            batch=Path(td)/"series.txt"
            batch.write_text(YOUTUBE+"\n"+WETV+"\n",encoding="utf-8")
            gui.target_var=Var(str(batch))
            mode,url,file_path,_=gui._validate()
            self.assertEqual((mode,url,file_path),("captions","",str(batch)))

    def test_unsupported_or_insecure_urls_rejected_before_network(self):
        with tempfile.TemporaryDirectory() as td:
            gui=self.gui(Path(td))
            for url in ("https://unlisted-video.invalid/show", "http://www.youtube.com/watch?v=x",
                        "https://127.0.0.1/private", "https://youtube.com.evil.invalid/x"):
                gui.url_var=Var(url)
                with self.subTest(url=url), self.assertRaises(ValueError):
                    gui._validate()
            self.assertFalse((Path(td)/"captions").exists())

    def test_duplicated_languages_rejected_before_any_download(self):
        with tempfile.TemporaryDirectory() as td:
            gui=self.gui(Path(td))
            gui.languages_var=Var("ko,KO")
            with self.assertRaisesRegex(ValueError,"중복"):
                gui._validate()

    def test_destination_existing_file_not_treated_as_folder(self):
        with tempfile.TemporaryDirectory() as td:
            gui=self.gui(Path(td))
            file=Path(td)/"captions"
            file.write_bytes(b"precious")
            with self.assertRaisesRegex(ValueError,"폴더"):
                gui._validate()
            self.assertEqual(file.read_bytes(),b"precious")

    def test_caption_mode_disables_video_picker_and_enables_url_list(self):
        with tempfile.TemporaryDirectory() as td:
            gui=self.gui(Path(td))
            for name in ("source_label","source_picker","target_entry",
                         "target_picker","target_label","out_label","languages_entry"):
                setattr(gui,name,mock.Mock())
            gui._source_mode_changed()
            self.assertEqual(gui.url_var.get(),"")
            self.assertEqual(gui.target_var.get(),"")
            gui.source_picker.configure.assert_called_once_with(state="disabled", text="영상 선택")
            gui.target_entry.configure.assert_called_once_with(state="normal")
            gui.languages_entry.configure.assert_called_once_with(state="normal")
            gui.target_picker.configure.assert_any_call(text="목록 선택")
            gui.out_label.configure.assert_called_once_with(
                text="자막 저장 폴더 (기존 결과 보존)"
            )

    def test_caption_mode_output_folder_picker_does_not_append_base_filename(self):
        with tempfile.TemporaryDirectory() as td:
            gui=self.gui(Path(td))
            with mock.patch(
                "hardsub_extractor.gui.filedialog.askdirectory",return_value=td,
            ):
                gui._pick_output()
            self.assertEqual(gui.out_var.get(),td)

    def test_caption_mode_list_picker_uses_text_file_dialog(self):
        with tempfile.TemporaryDirectory() as td:
            gui=self.gui(Path(td))
            with mock.patch(
                "hardsub_extractor.gui.filedialog.askopenfilename",
                return_value=str(Path(td)/"episodes.txt"),
            ) as dialog:
                gui._pick_target()
            self.assertEqual(gui.target_var.get(),str(Path(td)/"episodes.txt"))
            self.assertEqual(dialog.call_args.kwargs["filetypes"][0][1],"*.txt")

    def test_caption_mode_dispatch_never_constructs_epsite_client_or_ocr(self):
        with tempfile.TemporaryDirectory() as td:
            gui=self.gui(Path(td))
            found=[CaptionResult("youtube","ko","downloaded",cue_count=2)]
            with (
                mock.patch("hardsub_extractor.gui.PublicCaptionCollector") as collector,
                mock.patch("hardsub_extractor.gui.AutoEpSiteClient") as anime,
                mock.patch("hardsub_extractor.gui.RapidOCRAdapter") as ocr,
            ):
                collector.return_value.collect.return_value=found
                gui._worker("captions",YOUTUBE,None,Path(td),False,("ko",))
            self.assertEqual(self.events(gui,"captions_done"),[(1,0,0,0,td)])
            self.assertEqual(self.events(gui,"caption_progress"),[(0,1),(1,1)])
            anime.assert_not_called()
            ocr.assert_not_called()

    def test_failed_episode_does_not_block_remaining_batch(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td)
            gui=self.gui(root)
            file=root/"episodes.txt"
            file.write_text(YOUTUBE+"\n"+WETV+"\n",encoding="utf-8")
            responses=[
                CaptionAccessError("private signed URL SECRET"),
                [CaptionResult("wetv","en","downloaded",cue_count=3)],
            ]
            with mock.patch("hardsub_extractor.gui.PublicCaptionCollector") as c:
                c.return_value.collect.side_effect=responses
                gui._worker("captions","",str(file),root/"out",False,("en",))
            self.assertEqual(self.events(gui,"captions_done"),[(1,0,1,0,str(root/"out"))])
            self.assertFalse(self.events(gui,"error"))
            self.assertNotIn("SECRET",repr(list(gui.events.queue)))

    def test_missing_public_caption_reports_ocr_needed_but_never_starts_ocr(self):
        with tempfile.TemporaryDirectory() as td:
            gui=self.gui(Path(td))
            results=[CaptionResult("youtube","ja","needs_ocr")]
            with (mock.patch("hardsub_extractor.gui.PublicCaptionCollector") as c,
                  mock.patch("hardsub_extractor.gui.RapidOCRAdapter") as ocr):
                c.return_value.collect.return_value=results
                gui._worker("captions",YOUTUBE,None,Path(td),False,("ja",))
            self.assertEqual(self.events(gui,"captions_done"),[(0,1,0,0,td)])
            ocr.assert_not_called()

    def test_gui_caption_absence_routes_to_bounded_video_ocr_once(self):
        with tempfile.TemporaryDirectory() as td:
            gui = self.gui(Path(td))
            gui._requested_video_ocr_enabled = True
            missing = [CaptionResult("youtube", "ja", "needs_ocr", video_id="item")]
            hard = [CaptionResult("youtube", "ja", "hard_ocr", video_id="item", cue_count=4)]
            with mock.patch(
                "hardsub_extractor.gui.PublicCaptionCollector"
            ) as soft, mock.patch(
                "hardsub_extractor.gui.resolve_missing_public_ocr",
                return_value=hard,
            ) as video, mock.patch(
                "hardsub_extractor.gui.AutoEpSiteClient"
            ) as anime:
                soft.return_value.collect.return_value = missing
                gui._worker("captions", YOUTUBE, None, Path(td), False, ("ja",))
            self.assertEqual(video.call_count, 1)
            args = video.call_args.args
            self.assertEqual(args, (YOUTUBE, Path(td), missing))
            self.assertTrue(video.call_args.kwargs["skip_existing"])
            self.assertIsNotNone(video.call_args.kwargs["monitor"])
            self.assertEqual(self.events(gui, "captions_done"), [(1, 0, 0, 0, td)])
            anime.assert_not_called()

    def test_gui_caption_success_never_calls_video_ocr_even_when_enabled(self):
        with tempfile.TemporaryDirectory() as td:
            gui = self.gui(Path(td))
            gui._requested_video_ocr_enabled = True
            with mock.patch(
                "hardsub_extractor.gui.PublicCaptionCollector"
            ) as soft, mock.patch(
                "hardsub_extractor.gui.resolve_missing_public_ocr"
            ) as video:
                soft.return_value.collect.return_value = [
                    CaptionResult("youtube", "ko", "downloaded", cue_count=2)
                ]
                gui._worker("captions", YOUTUBE, None, Path(td), False, ("ko",))
            video.assert_not_called()
            self.assertEqual(self.events(gui, "captions_done"), [(1, 0, 0, 0, td)])

    def test_gui_inaccessible_ocr_status_is_not_false_success(self):
        with tempfile.TemporaryDirectory() as td:
            gui = self.gui(Path(td))
            gui._requested_video_ocr_enabled = True
            with mock.patch(
                "hardsub_extractor.gui.PublicCaptionCollector"
            ) as soft, mock.patch(
                "hardsub_extractor.gui.resolve_missing_public_ocr",
                return_value=[CaptionResult("youtube", "ko", "ocr_unavailable")],
            ):
                soft.return_value.collect.return_value = [
                    CaptionResult("youtube", "ko", "needs_ocr", video_id="item")
                ]
                gui._worker("captions", YOUTUBE, None, Path(td), False, ("ko",))
            self.assertEqual(self.events(gui, "captions_done"), [(0, 0, 1, 0, td)])

    def test_gui_video_ocr_option_is_captured_on_validation(self):
        with tempfile.TemporaryDirectory() as td:
            gui = self.gui(Path(td))
            gui.video_ocr_var = Var(True)
            gui._validate()
            self.assertTrue(gui._requested_video_ocr_enabled)
            gui.video_ocr_var = Var(False)
            gui._validate()
            self.assertFalse(gui._requested_video_ocr_enabled)

    def test_existing_subtitle_is_accounted_separately_from_new_download(self):
        with tempfile.TemporaryDirectory() as td:
            gui = self.gui(Path(td))
            with mock.patch("hardsub_extractor.gui.PublicCaptionCollector") as adapter:
                adapter.return_value.collect.return_value = [
                    CaptionResult("youtube", "en", "existing_preserved"),
                    CaptionResult("youtube", "ko", "downloaded", cue_count=3),
                ]
                gui._worker("captions", YOUTUBE, None, Path(td), False, ("en", "ko"))
                self.assertTrue(adapter.return_value.collect.call_args.kwargs["skip_existing"])
            self.assertEqual(self.events(gui, "captions_done"), [(1, 0, 0, 1, td)])

    def test_access_failure_completes_episode_progress(self):
        with tempfile.TemporaryDirectory() as td:
            gui = self.gui(Path(td))
            with mock.patch("hardsub_extractor.gui.PublicCaptionCollector") as adapter:
                adapter.return_value.collect.side_effect = CaptionAccessError("offline")
                gui._worker("captions", YOUTUBE, None, Path(td), False, ("en",))
            self.assertEqual(self.events(gui, "caption_progress"), [(0, 1), (1, 1)])

    def test_cancelled_collector_stops_before_any_second_episode(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td)
            gui=self.gui(root)
            file=root/"episodes.txt"
            file.write_text(YOUTUBE+"\n"+WETV+"\n",encoding="utf-8")
            def cancel_first(*args, **kwargs):
                gui._cancel_event.set()
                kwargs["check_cancelled"]()
            with mock.patch("hardsub_extractor.gui.PublicCaptionCollector") as c:
                c.return_value.collect.side_effect=cancel_first
                gui._worker("captions","",str(file),root/"out",False,("ko",))
            self.assertEqual(c.return_value.collect.call_count,1)
            self.assertEqual(self.events(gui,"cancelled"),[None])
            self.assertFalse(self.events(gui,"captions_done"))


if __name__ == "__main__":
    unittest.main()
