"""GUI review acceptance without user files, external media or Tk display."""
from __future__ import annotations

from pathlib import Path
import queue
import tempfile
import threading
import unittest
from unittest import mock

from hardsub_extractor.gui import ExtractorGUI


class Var:
    def __init__(self, value=""):
        self.value = value

    def get(self):
        return self.value

    def set(self, value):
        self.value = value


def fake_gui(video, base, mode="local", scope="경고 자막만", after="0"):
    gui = ExtractorGUI.__new__(ExtractorGUI)
    for name, value in {
        "source_mode_var": mode, "url_var": str(video),
        "out_var": str(base), "review_mode_var": scope,
        "review_after_var": after, "ocr_language_var": "zh",
        "status_var": "준비됨",
    }.items():
        setattr(gui, name, Var(value))
    gui._pending_close = False
    gui._cancel_event = threading.Event()
    gui.worker = None
    gui.events = queue.Queue()
    gui.progress_bar = mock.Mock()
    gui.start_button = mock.Mock()
    gui.cancel_button = mock.Mock()
    gui.review_button = mock.Mock()
    gui.root = mock.Mock()
    gui._log = mock.Mock()
    return gui


def setup_files(path):
    video = path / "private-video.mp4"
    video.write_bytes(b"unchanged-video")
    base = path / "captions"
    srt = path / "captions.srt"
    srt.write_text(
        "1\n00:00:01,000 --> 00:00:03,000\n"
        "비공개 영상 자막\n", encoding="utf-8",
    )
    return video, base, srt


class FrameReviewGUITests(unittest.TestCase):
    def test_only_local_source_mode_is_eligible(self):
        with tempfile.TemporaryDirectory() as td:
            video, base, _ = setup_files(Path(td))
            gui = fake_gui(video, base, mode="captions")
            with (mock.patch("hardsub_extractor.gui.messagebox.showerror") as err,
                  mock.patch("hardsub_extractor.gui.messagebox.askyesno") as ask,
                  mock.patch("hardsub_extractor.gui.threading.Thread") as spawn):
                gui._start_frame_review()
            err.assert_called_once()
            ask.assert_not_called()
            spawn.assert_not_called()

    def test_existing_srt_is_required_and_extraction_not_restarted(self):
        with tempfile.TemporaryDirectory() as td:
            video, base, srt = setup_files(Path(td))
            srt.unlink()
            gui = fake_gui(video, base)
            with (mock.patch("hardsub_extractor.gui.messagebox.showerror") as err,
                  mock.patch("hardsub_extractor.gui.threading.Thread") as spawn):
                gui._start_frame_review()
            err.assert_called_once()
            spawn.assert_not_called()

    def test_declined_privacy_consent_starts_no_job_or_file(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            video, base, srt = setup_files(root)
            before = video.read_bytes(), srt.read_bytes()
            gui = fake_gui(video, base)
            with (mock.patch("hardsub_extractor.gui.messagebox.askyesno",return_value=False),
                  mock.patch("hardsub_extractor.gui.threading.Thread") as spawn):
                gui._start_frame_review()
            spawn.assert_not_called()
            self.assertEqual((video.read_bytes(),srt.read_bytes()), before)
            self.assertEqual(len(list(root.iterdir())),2)

    def test_split_review_requires_current_source_and_exact_srt_bytes(self):
        from hashlib import sha256
        with tempfile.TemporaryDirectory() as td:
            video, base, subtitle = setup_files(Path(td))
            gui = fake_gui(video, base, scope="연속 대사 분리 검토")
            stat = video.stat()
            gui._native_split_review_hints = (
                video, subtitle, sha256(subtitle.read_bytes()).digest(),
                stat.st_size, stat.st_mtime_ns, (1,),
            )
            with (
                mock.patch("hardsub_extractor.gui.messagebox.askyesno",
                           return_value=True) as consent,
                mock.patch("hardsub_extractor.gui.threading.Thread") as spawn,
            ):
                gui._start_frame_review()
            args = spawn.call_args.kwargs["args"]
            self.assertEqual(args[3:], ("split", 0, "zh", (1,)))
            self.assertEqual(consent.call_count, 1)
            spawn.return_value.start.assert_called_once()
            gui.worker = None
            subtitle.write_text(
                subtitle.read_text("utf-8") + "\n", encoding="utf-8"
            )
            with (
                mock.patch("hardsub_extractor.gui.messagebox.showerror") as error,
                mock.patch("hardsub_extractor.gui.messagebox.askyesno") as consent,
                mock.patch("hardsub_extractor.gui.threading.Thread") as spawn,
            ):
                gui._start_frame_review()
            error.assert_called_once()
            consent.assert_not_called()
            spawn.assert_not_called()

    def test_native_split_worker_attests_exact_export_before_enabling_review(self):
        from hashlib import sha256
        from types import SimpleNamespace
        from hardsub_extractor import gui as gui_module
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            video, base, srt = setup_files(root)
            srt.write_text(
                "1\n00:00:01,000 --> 00:00:02,000\n앞 자막\n\n"
                "2\n00:00:02,000 --> 00:00:03,000\n뒤 자막\n",
                encoding="utf-8",
            )
            models = root / "models"
            models.mkdir()
            engine = SimpleNamespace(
                report=SimpleNamespace(
                    detected_intervals=1, recognized_intervals=1,
                    missed_intervals=0, ocr_calls=5,
                    exported_cues=2, split_source_intervals=1,
                    split_cue_numbers=(1,), missed_windows=(),
                    srt_path=srt, ass_path=base.with_suffix(".ass"),
                )
            )
            gui = fake_gui(video, base)
            with (
                mock.patch.object(gui_module, "acquire_local_embedded_text",
                                  return_value=None),
                mock.patch.object(gui_module, "acquire_local_sidecar_text",
                                  return_value=None),
                mock.patch.object(gui_module, "acquire_local_embedded_image",
                                  return_value=None),
                mock.patch.object(gui_module, "bundled_model_dir",
                                  return_value=models),
                mock.patch("hardsub_extractor.native_vsf_assets.ensure_official_vsf",
                           return_value=(root / "trusted.exe", "a"*64)),
                mock.patch("hardsub_extractor.native_vsf_timing.extract_with_native_vsf",
                           return_value=engine) as backend,
                mock.patch.object(gui, "_emit_local_ocr_quality"),
            ):
                gui._worker_local(
                    str(video), base, False, ocr_language="ko",
                    native_vsf_opt_in=True, native_split_opt_in=True,
                )
            self.assertTrue(backend.call_args.kwargs["split_stable_text"])
            events = []
            while not gui.events.empty():
                events.append(gui.events.get_nowait())
            proof = [value for name, value in events
                     if name == "native_split_review_ready"]
            self.assertEqual(len(proof), 1)
            stat = video.stat()
            self.assertEqual(proof[0], (
                video, srt, sha256(srt.read_bytes()).digest(),
                stat.st_size, stat.st_mtime_ns, (1,),
            ))
            self.assertEqual(events[-1][0], "done")
            self.assertNotIn("앞 자막", str(events))
            self.assertNotIn("뒤 자막", str(events))

    def test_split_review_worker_forwards_only_selected_cue_numbers(self):
        with tempfile.TemporaryDirectory() as td:
            video, base, subtitle = setup_files(Path(td))
            gui = fake_gui(video, base)
            with mock.patch(
                "hardsub_extractor.review_pack.create_pack",
                return_value={"status": "NO_REVIEW_LOCATIONS", "last_cue": 0,
                              "cue_count": 0, "frame_count": 0},
            ) as call:
                gui._review_worker(
                    video, subtitle, base.parent / "unused",
                    "split", 0, "zh", (2,),
                )
            self.assertEqual(call.call_args.kwargs["mode"], "split")
            self.assertEqual(call.call_args.kwargs["split_cue_numbers"], (2,))
            self.assertEqual(gui.events.get_nowait()[0], "review_done")

    def test_approved_mode_has_frozen_worker_arguments(self):
        with tempfile.TemporaryDirectory() as td:
            video, base, srt = setup_files(Path(td))
            gui = fake_gui(video, base, scope="모든 자막 순서대로", after="7")
            with (mock.patch("hardsub_extractor.gui.messagebox.askyesno",return_value=True),
                  mock.patch("hardsub_extractor.gui.threading.Thread") as spawn):
                gui._start_frame_review()
            a=spawn.call_args.kwargs["args"]
            self.assertEqual(a[:2],(video,srt))
            self.assertEqual(a[3:],("all",7,"zh"))
            self.assertEqual(a[2].name,"captions.review-all-after-00007")
            spawn.return_value.start.assert_called_once()
            gui.review_button.configure.assert_called_with(state="disabled")

    def test_existing_private_folder_is_never_overwritten(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            video, base, _ = setup_files(root)
            folder=root/"captions.review-flagged-after-00000"
            folder.mkdir()
            marker=folder/"keep.txt"
            marker.write_text("KEEP","utf-8")
            gui=fake_gui(video,base)
            with (mock.patch("hardsub_extractor.gui.messagebox.showerror") as err,
                  mock.patch("hardsub_extractor.gui.messagebox.askyesno") as ask,
                  mock.patch("hardsub_extractor.gui.threading.Thread") as spawn):
                gui._start_frame_review()
            err.assert_called_once()
            ask.assert_not_called()
            spawn.assert_not_called()
            self.assertEqual(marker.read_text("utf-8"),"KEEP")

    def test_invalid_resume_or_scope_never_spawns_worker(self):
        with tempfile.TemporaryDirectory() as td:
            video, base, _ = setup_files(Path(td))
            for after, scope in (("-1","경고 자막만"),
                                 ("invalid","경고 자막만"),
                                 ("10001","경고 자막만"),
                                 ("0","unknown")):
                gui=fake_gui(video,base,scope=scope,after=after)
                with (mock.patch("hardsub_extractor.gui.messagebox.showerror") as err,
                      mock.patch("hardsub_extractor.gui.threading.Thread") as spawn):
                    gui._start_frame_review()
                err.assert_called_once()
                spawn.assert_not_called()

    def test_review_worker_calls_pack_without_network_or_rewriting_subtitles(self):
        with tempfile.TemporaryDirectory() as td:
            video, base, srt = setup_files(Path(td))
            gui=fake_gui(video,base)
            result={"status":"CREATED","cue_count":2,"frame_count":4,"last_cue":12}
            with mock.patch("hardsub_extractor.review_pack.create_pack",return_value=result) as pack:
                gui._review_worker(video,srt,base.parent/"evidence","all",0,"zh")
            self.assertEqual(pack.call_args.kwargs["mode"],"all")
            self.assertEqual(pack.call_args.kwargs["max_cues"],8)
            self.assertEqual(pack.call_args.kwargs["after_cue"],0)
            pack.call_args.kwargs["check_cancelled"]()
            self.assertEqual(gui.events.get_nowait()[0],"review_done")
            self.assertTrue(gui.events.empty())

    def test_review_cancel_produces_no_success_event(self):
        with tempfile.TemporaryDirectory() as td:
            video,base,srt=setup_files(Path(td))
            gui=fake_gui(video,base)
            gui._cancel_event.set()
            def aborted(*args,**kwargs):
                kwargs["check_cancelled"]()
            with mock.patch("hardsub_extractor.review_pack.create_pack",side_effect=aborted):
                gui._review_worker(video,srt,base.parent/"evidence","all",0,"zh")
            self.assertEqual(gui.events.get_nowait()[0],"cancelled")
            self.assertTrue(gui.events.empty())

    def test_sensitive_exception_text_is_absent_from_worker_events(self):
        with tempfile.TemporaryDirectory() as td:
            video,base,srt=setup_files(Path(td))
            gui=fake_gui(video,base)
            with mock.patch("hardsub_extractor.review_pack.create_pack",
                            side_effect=RuntimeError("SECRET_USER_FILE_PATH")):
                gui._review_worker(video,srt,base.parent/"evidence","all",0,"zh")
            kind,payload=gui.events.get_nowait()
            self.assertEqual(kind,"error")
            self.assertNotIn("SECRET_USER_FILE_PATH",str(payload))
            self.assertIn("RuntimeError",str(payload))

    def test_complete_event_advances_cursor_only_after_success(self):
        gui=fake_gui("source.mp4","captions")
        gui.events.put(("review_done",({
            "status":"CREATED","cue_count":2,"frame_count":4,"last_cue":12,
        },"private-review-directory")))
        with mock.patch("hardsub_extractor.gui.messagebox.showinfo") as info:
            gui._poll_events()
        self.assertEqual(gui.review_after_var.get(),"12")
        self.assertEqual(gui.status_var.get(),"원본 화면 검토 완료")
        gui.review_button.configure.assert_called_with(state="normal")
        gui.cancel_button.configure.assert_called_with(state="disabled")
        info.assert_called_once()

    def test_review_complete_opens_private_page_once_but_empty_review_never_does(self):
        gui = fake_gui("source.mp4", "captions")
        gui.events.put(("review_done", ({
            "status": "CREATED", "cue_count": 1, "frame_count": 2, "last_cue": 4
        }, "private-review-dir")))
        with (mock.patch("hardsub_extractor.gui.messagebox.showinfo"),
              mock.patch("hardsub_extractor.gui._open_owner_review_page", return_value=True) as open_page):
            gui._poll_events()
        open_page.assert_called_once_with("private-review-dir")
        self.assertEqual(gui.review_after_var.get(), "4")
        gui.events.put(("review_done", ({
            "status": "NO_REVIEW_LOCATIONS", "cue_count": 0, "frame_count": 0, "last_cue": 4
        }, "unused")))
        with (mock.patch("hardsub_extractor.gui.messagebox.showinfo"),
              mock.patch("hardsub_extractor.gui._open_owner_review_page") as open_page):
            gui._poll_events()
        open_page.assert_not_called()

    @unittest.skipUnless(__import__("os").name == "nt", "real Windows ShellExecute boundary")
    def test_review_page_opener_never_launches_missing_or_symlinked(self):
        from hardsub_extractor import gui as module
        with tempfile.TemporaryDirectory() as td:
            folder = Path(td) / "private"
            folder.mkdir()
            with mock.patch.object(module.os, "startfile") as invoke:
                self.assertFalse(module._open_owner_review_page(str(folder)))
                invoke.assert_not_called()
                (folder / "review.html").write_text("private", "utf-8")
                self.assertTrue(module._open_owner_review_page(str(folder)))
                invoke.assert_called_once_with(str(folder / "review.html"))

    def test_no_warning_keeps_original_resume_position(self):
        gui=fake_gui("source.mp4","captions",after="9")
        gui.events.put(("review_done",({
            "status":"NO_REVIEW_LOCATIONS","cue_count":0,
            "frame_count":0,"last_cue":9,
        },"no-directory")))
        with mock.patch("hardsub_extractor.gui.messagebox.showinfo"):
            gui._poll_events()
        self.assertEqual(gui.review_after_var.get(),"9")
        self.assertEqual(gui.status_var.get(),"검토 경고 없음")


if __name__ == "__main__":
    unittest.main()
