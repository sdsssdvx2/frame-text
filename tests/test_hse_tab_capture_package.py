"""The source GUI and CLI use the same free local tab capture engine."""
from __future__ import annotations
import inspect
import unittest
from hardsub_extractor import tab_capture
from hardsub_extractor.gui import ExtractorGUI
from tools.hse_tab_capture_fallback import CaptureSession as LegacyCaptureSession


class BrowserTabGuiContract(unittest.TestCase):
    def test_no_second_implementation(self):
        self.assertIs(tab_capture.CaptureSession, LegacyCaptureSession)
        self.assertEqual(inspect.signature(tab_capture.serve).parameters["ocr_language"].default, "ko")

    def test_explicit_taller_ocr_roi_preserves_baseline_and_reaches_pipeline(self):
        from pathlib import Path
        from tempfile import TemporaryDirectory
        from unittest.mock import patch
        from hardsub_extractor.frame_source import NormalizedROI
        from hardsub_extractor.runtime_paths import bundled_model_dir
        self.assertEqual(
            tab_capture.roi_for_region("bottom40").pixels(1920,1080),
            NormalizedROI().pixels(1920,1080))
        actual=[
            tab_capture.roi_for_region(k).pixels(1920,1080)
            for k in ("bottom40","lower65","lower80")
        ]
        # Same horizontal width and bottom coordinate; y moves upward.
        self.assertEqual(len({(x,w) for x,y,w,h in actual}),1)
        self.assertEqual(len({y+h for x,y,w,h in actual}),1)
        self.assertGreater(actual[0][1],actual[1][1])
        self.assertGreater(actual[1][1],actual[2][1])
        for bad in (None,True,"full-screen","all","lower100",42):
            with self.subTest(bad=bad),self.assertRaises(ValueError):
                tab_capture.roi_for_region(bad)
        with TemporaryDirectory(prefix="hse-roi-pipeline-") as d:
            models=Path(d)/"models"
            models.mkdir()
            manifest=Path(d)/"manifest.json"
            manifest.write_text("{}",encoding="utf-8")
            expected=object()
            with (patch("hardsub_extractor.runtime_paths.bundled_model_dir",
                        return_value=models),
                  patch("hardsub_extractor.runtime_paths.bundled_manifest_path",
                        return_value=manifest),
                  patch("hardsub_extractor.ocr.rapidocr_adapter.RapidOCRAdapter"),
                  patch("hardsub_extractor.pipeline.extract",return_value=expected)
                        as extracted):
                output=tab_capture.offline_ocr(
                    Path(d)/"recorded.webm",Path(d)/"result",
                    language="ja",caption_mode="all",
                    region_mode="lower65")
                self.assertIs(output,expected)
                selected=extracted.call_args.kwargs["roi"]
                self.assertEqual(selected.pixels(1920,1080),actual[1])
                self.assertEqual(extracted.call_args.kwargs["sample_fps"],8.0)
                self.assertTrue(extracted.call_args.kwargs["streaming"])
                extracted.reset_mock()
                tab_capture.offline_ocr(
                    Path(d)/"recorded.webm",Path(d)/"legacy",
                    language="ko")
                self.assertNotIn("roi",extracted.call_args.kwargs)

    def test_immediate_stop_before_browser_open_does_not_leave_server(self):
        import tempfile
        import threading
        from pathlib import Path
        with tempfile.TemporaryDirectory(prefix="hse-gui-tab-stop-") as tmp:
            stopper=threading.Event()
            stopper.set()
            ready=[]
            finished=[]
            tab_capture.serve(
                Path(tmp), open_browser=False, stop_event=stopper,
                on_ready=ready.append, on_finished=finished.append,
            )
            self.assertEqual(ready, [])
            self.assertEqual(len(finished), 1)
            self.assertEqual(finished[0]["state"], "recording")
            self.assertNotIn("srt_path", finished[0])
            self.assertEqual(list(Path(tmp).iterdir()), [])

    def test_completed_tab_auto_releases_gui_server_and_preserves_exports(self):
        # Test the real serve() lifecycle, but not Windows HTTP, a browser,
        # external models or private media. No user stop button is pressed.
        import tempfile
        import threading
        from pathlib import Path
        from types import SimpleNamespace
        from unittest.mock import patch
        with tempfile.TemporaryDirectory(prefix="hse-tab-autoclose-") as tmp:
            root=Path(tmp)
            keep=root/"existing.srt"
            keep.write_bytes(b"KEEP_ORIGINAL")
            created={}
            class OwnedSession(tab_capture.CaptureSession):
                def __init__(self, *args, **kwargs):
                    super().__init__(*args, **kwargs)
                    created["session"]=self
            def fake_ocr(video, base, **kwargs):
                if video.read_bytes()[:4]!=bytes.fromhex("1a45dfa3"):
                    raise ValueError("bad_test_webm")
                base.with_suffix(".srt").write_text(
                    "1\\n00:00:01,000 --> 00:00:02,000\\nCAPTION\\n",
                    encoding="utf-8")
                base.with_suffix(".ass").write_text(
                    "[Script Info]\\nTitle: local\\n",encoding="utf-8")
                return SimpleNamespace(cue_count=1)
            ready=threading.Event()
            results=[]
            stop=threading.Event()
            def on_ready(url):
                self.assertTrue(url.startswith("http://127.0.0.1:"))
                ready.set()
            with (patch.object(tab_capture,"CaptureSession",OwnedSession),
                  patch.object(tab_capture,"offline_ocr",side_effect=fake_ocr)):
                server_thread=threading.Thread(
                    target=tab_capture.serve,
                    args=(root,),
                    kwargs={"open_browser":False,"stop_event":stop,
                            "on_ready":on_ready,"on_finished":results.append},
                    daemon=True,
                )
                server_thread.start()
                try:
                    self.assertTrue(ready.wait(timeout=3))
                    session=created["session"]
                    session.append(0,bytes.fromhex("1a45dfa3")+bytes(8192))
                    session.finish()
                    session.worker.join(timeout=4)
                    self.assertFalse(session.worker.is_alive())
                    self.assertEqual(session.snapshot()["state"],"completed")
                    # The GUI did not request stop. The one-recording helper
                    # should deliver the local result after a short grace.
                    self.assertFalse(stop.is_set())
                    server_thread.join(timeout=13)
                    self.assertFalse(server_thread.is_alive(),
                        "finished tab server required a second manual close")
                    self.assertEqual(len(results),1)
                    self.assertEqual(results[0]["state"],"completed")
                    self.assertEqual(results[0]["cue_count"],1)
                    self.assertTrue(Path(results[0]["srt_path"]).is_file())
                    self.assertTrue(Path(results[0]["ass_path"]).is_file())
                    self.assertEqual(keep.read_bytes(),b"KEEP_ORIGINAL")
                finally:
                    stop.set()
                    server_thread.join(timeout=11)

    def test_real_local_result_after_in_memory_subtitle_export(self):
        # Focused result callback contract: no Windows loopback dependency.
        from pathlib import Path
        from tempfile import TemporaryDirectory
        from types import SimpleNamespace
        with TemporaryDirectory(prefix="hse-tab-result-") as d:
            root=Path(d)
            scratch=root/"private";scratch.mkdir()
            outputs=root/"result";outputs.mkdir()
            keep=outputs/"existing.srt";keep.write_bytes(b"ORIGINAL")
            def processor(video,base):
                base.with_suffix(".srt").write_text("new srt")
                base.with_suffix(".ass").write_text("new ass")
                return SimpleNamespace(cue_count=2)
            session=tab_capture.CaptureSession(scratch,outputs,processor)
            self.assertEqual(session.local_result()["state"],"recording")
            self.assertNotIn("srt_path",session.local_result())
            session.append(0,bytes.fromhex("1a45dfa3")+bytes(8192))
            session.finish()
            session.worker.join(timeout=4)
            self.assertFalse(session.worker.is_alive())
            result=session.local_result()
            self.assertEqual(result["state"],"completed")
            self.assertEqual(result["cue_count"],2)
            self.assertEqual(Path(result["srt_path"]).read_text(),"new srt")
            self.assertEqual(Path(result["ass_path"]).read_text(),"new ass")
            self.assertNotIn("srt_path",session.snapshot())
            self.assertEqual(keep.read_bytes(),b"ORIGINAL")
            session.close()

    def test_gui_explicit_methods(self):
        self.assertTrue(callable(ExtractorGUI._start_tab_capture))
        self.assertTrue(callable(ExtractorGUI._stop_tab_capture))
        self.assertTrue(callable(ExtractorGUI._on_close))

    def test_languages(self):
        from hardsub_extractor.model_assets import normalize_ocr_language
        self.assertEqual([normalize_ocr_language(v) for v in ("ko","en","ja","zh")],
                         ["ko","en","ja","zh"])


if __name__=="__main__":
    unittest.main()
