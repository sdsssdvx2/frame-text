"""Pure stdlib HTTP/stream acceptance. No browser, user media or model calls."""
from __future__ import annotations
import hashlib
import http.client
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import threading
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"tools"))
from hse_tab_capture_fallback import CaptureSession,_Server,MAX_TOTAL,PAGE


class TabCaptureBridgeTests(unittest.TestCase):
    def setUp(self):
        self.t=tempfile.TemporaryDirectory()
        self.root=Path(self.t.name)
        self.source=self.root/"scratch"
        self.source.mkdir()
        self.out=self.root/"exports"
        self.out.mkdir()
        self.captured=[]
        def fake_processor(video,base):
            self.captured.append((video.stat().st_size,video.read_bytes()[:4]))
            base.with_suffix(".srt").write_text("1\n00:00:01,000 --> 00:00:02,000\n자막\n")
            base.with_suffix(".ass").write_text("fake ass")
            return SimpleNamespace(cue_count=1)
        self.s=CaptureSession(self.source,self.out,fake_processor)
        self.http=_Server(self.s)
        self.thread=threading.Thread(target=self.http.serve_forever,daemon=True)
        self.thread.start()
        self.origin=self.http.url

    def tearDown(self):
        self.http.shutdown()
        self.http.server_close()
        self.thread.join(timeout=3)
        if self.s.worker:self.s.worker.join(timeout=3)
        self.s.close()
        self.t.cleanup()

    def request(self,path,body=None,token=None,origin=None,host=None,method=None,content_type="application/octet-stream"):
        c=http.client.HTTPConnection("127.0.0.1",self.http.server_port,timeout=3)
        h={"Host":host or "127.0.0.1:"+str(self.http.server_port)}
        if body is not None:
            h.update({"X-HSE-Token":token if token is not None else self.s.token,
                      "Origin":origin if origin is not None else self.origin,
                      "Content-Type":content_type})
        elif token is not None:
            h["X-HSE-Token"]=token
        c.request(method or ("POST" if body is not None else "GET"),path,body=body,headers=h)
        try:
            resp=c.getresponse()
            data=resp.read()
            result=(resp.status,resp.getheaders(),data)
        except (ConnectionResetError, ConnectionAbortedError, BrokenPipeError):
            # A Windows loopback firewall may terminate rejected HTTP
            # connections without relaying their 4xx bodies. An abort is
            # fail-closed, never evidence that a request was accepted.
            result=(0,[],b"")
        finally:
            c.close()
        return result

    def test_control_page_has_user_gesture_without_network_or_extension(self):
        # Windows loopback may reset a completed read-only GET response during
        # host admission. This is safe to repeat: GET / does not write media,
        # start OCR, grant screen capture or change the one-use token.
        # Only a lost transport ACK (status 0) is retried, never a real HTTP
        # rejection or malformed page.
        for attempt in range(3):
            code,headers,raw=self.request("/")
            if code != 0:
                break
            if attempt != 2:
                time.sleep(0.12)
        self.assertEqual(code,200)
        self.assertIn(b"navigator.mediaDevices.getDisplayMedia",raw)
        self.assertIn(b"MediaRecorder",raw)
        self.assertIn(b"rec.onerror=()=>fail(",raw)
        self.assertIn(b"if(!seq){",raw)
        self.assertIn(b"__TOKEN__",PAGE.encode())
        self.assertNotIn(b"__TOKEN__",raw)
        self.assertIn(b"Content-Security-Policy",str(headers).encode())
        self.assertNotIn(b"https://",raw)

    def test_cross_origin_host_and_missing_token_blocked(self):
        self.assertIn(self.request("/",host="evil.test")[0],(0,403))
        self.assertIn(self.request("/status",token="wrong")[0],(0,403))
        self.assertIn(self.request("/chunk?i=0",b"data",origin="https://evil.test")[0],(0,403))
        self.assertEqual(self.s.next_index,0)

    def test_chunk_sequence_bounds_and_no_overwrite(self):
        self.assertIn(self.request("/chunk?i=0",b"bad-webm")[0],(0,409))
        self.assertEqual(self.s.next_index,0)
        chunk=bytes.fromhex("1a45dfa3")+bytes(8192)
        self.assertIn(self.request("/chunk?i=1",chunk)[0],(0,409))
        outcomes=[self.request("/chunk?i=0",chunk)[0]]
        if outcomes[-1]!=200:
            outcomes.append(self.request("/chunk?i=0",chunk)[0])
        if outcomes[-1]!=200:
            outcomes.append(self.request("/chunk?i=0",chunk)[0])
        # Windows can lose every accepted POST ACK. That is recoverable
        # only if the host proves the exact byte sequence is committed.
        if 200 not in outcomes:
            status=self.s.snapshot()
            self.assertEqual(status["last_chunk_index"],0)
            self.assertEqual(status["last_chunk_size"],len(chunk))
            self.assertEqual(status["last_chunk_sha256"],
                             hashlib.sha256(chunk).hexdigest())
        self.assertEqual(self.s.next_index,1)
        self.assertEqual(self.s.total,len(chunk))
        self.assertEqual(self.s.snapshot()["last_chunk_sha256"],
                         hashlib.sha256(chunk).hexdigest())
        self.assertIn(self.request("/chunk?i=0",chunk+b"other")[0],(0,409))
        self.assertEqual(self.s.total,len(chunk))
        self.assertIn(self.request("/chunk?i=1",b"x",content_type="text/plain")[0],(0,415))
        self.assertIn(self.request("/chunk?i=1",b"x"*2,host="wrong.invalid")[0],(0,403))
        self.assertEqual(self.s.total,len(chunk))

    def test_byte_identical_replay_is_idempotent_but_conflict_fails(self):
        chunk=bytes.fromhex("1a45dfa3")+bytes(8192)
        a=self.s.append(0,chunk)
        b=self.s.append(0,chunk)
        self.assertFalse(a["replayed"])
        self.assertTrue(b["replayed"])
        self.assertEqual(a["chunks"],b["chunks"])
        self.assertEqual(self.s.total,len(chunk))
        with self.assertRaises(ValueError):
            self.s.append(0,chunk+b"x")
        self.assertEqual(self.s.total,len(chunk))

    def test_finish_calls_local_extractor_once_and_preserves_originals(self):
        unchanged=self.out/"owner.srt"
        unchanged.write_bytes(b"original")
        first=bytes.fromhex("1a45dfa3")+bytes(10000)
        def accepted(path, payload):
            for _ in range(3):
                if self.request(path, payload)[0] == 200:
                    return
            self.fail("bounded identical HTTP replay never acknowledged")

        # Windows may drop an ACK AFTER a chunk or finish was committed.
        # Replay identical bytes and require one successful acknowledgment,
        # with no duplicate disk data or duplicate OCR worker.
        accepted("/chunk?i=0", first)
        self.assertEqual(self.s.total, len(first))
        accepted("/finish", b"")
        worker = self.s.worker
        accepted("/finish", b"")
        self.assertIs(self.s.worker, worker)
        worker.join(timeout=3)
        accepted("/finish", b"")
        self.assertIs(self.s.worker, worker)
        self.assertEqual(self.s.snapshot()["state"],"completed")
        self.assertEqual(self.s.snapshot()["cue_count"],1)
        report=self.s.local_result()
        self.assertEqual(report["state"],"completed")
        self.assertEqual(report["cue_count"],1)
        self.assertTrue(Path(report["srt_path"]).is_file())
        self.assertTrue(Path(report["ass_path"]).is_file())
        self.assertNotIn("srt_path",self.s.snapshot())
        self.assertEqual(self.captured,[(len(first),bytes.fromhex("1a45dfa3"))])
        self.assertEqual(unchanged.read_bytes(),b"original")
        self.assertEqual(len(list(self.out.glob("hse-tab-*.srt"))),1)
        self.assertEqual(len(list(self.out.glob("hse-tab-*.ass"))),1)

    def test_finish_replay_does_not_start_duplicate_OCR(self):
        chunk=bytes.fromhex("1a45dfa3")+bytes(8192)
        self.s.append(0,chunk)
        first=self.s.finish()
        worker=self.s.worker
        replay=self.s.finish()
        self.assertEqual(first["state"],"processing")
        self.assertFalse(first["replayed"])
        self.assertTrue(replay["replayed"])
        self.assertIs(self.s.worker,worker)
        worker.join(timeout=3)
        self.assertFalse(worker.is_alive())
        self.assertEqual(self.s.snapshot()["state"],"completed")
        self.assertEqual(self.s.finish()["state"],"processing")
        self.assertIs(self.s.worker,worker)
        self.assertEqual(len(self.captured),1)

    def test_selected_region_pinned_on_first_finish_and_replay(self):
        # No live HTTP, model, media or user data. A lost acknowledgement
        # must not select a different reading region or start another OCR.
        chunk=bytes.fromhex("1a45dfa3")+bytes(8192)
        self.s.append(0,chunk)
        with self.assertRaises(ValueError):
            self.s.finish(0,"all","entire-screen")
        self.assertEqual(self.s.state,"recording")
        first=self.s.finish(0,"all","lower65")
        self.assertEqual(first["state"],"processing")
        self.assertEqual(self.s.region_mode,"lower65")
        worker=self.s.worker
        replay=self.s.finish(0,"all","lower65")
        self.assertTrue(replay["replayed"])
        self.assertIs(worker,self.s.worker)
        with self.assertRaises(ValueError):
            self.s.finish(0,"all","lower80")
        self.assertIs(worker,self.s.worker)
        worker.join(timeout=4)
        self.assertFalse(worker.is_alive())
        self.assertEqual(self.s.snapshot()["state"],"completed")
        self.assertEqual(len(self.captured),1)
        self.assertNotIn("region_mode",json.dumps(self.s.snapshot()))

    def test_persistent_caption_detection_is_explicit_and_replay_pinned(self):
        chunk=bytes.fromhex("1a45dfa3")+bytes(8192)
        self.s.append(0,chunk)
        with self.assertRaises(ValueError):
            self.s.finish(0,"all","bottom40","persistent-white-3")
        with self.assertRaises(ValueError):
            self.s.finish(0,"light","lower65","persistent-white-3")
        self.assertEqual(self.s.state,"recording")
        first=self.s.finish(0,"all","lower65","persistent-white-3")
        self.assertEqual(first["state"],"processing")
        self.assertEqual(self.s.detection_mode,"persistent-white-3")
        worker=self.s.worker
        self.assertTrue(self.s.finish(
            0,"all","lower65","persistent-white-3")["replayed"])
        with self.assertRaises(ValueError):
            self.s.finish(0,"all","lower65","edges")
        self.assertIs(self.s.worker,worker)
        worker.join(timeout=4)
        self.assertFalse(worker.is_alive())
        self.assertEqual(len(self.captured),1)
        self.assertNotIn("detection_mode",json.dumps(self.s.snapshot()))
        self.assertIn('id="detection-mode"',PAGE)

    def test_caption_islands_pinned_on_one_explicit_finish(self):
        self.s.append(0,bytes.fromhex("1a45dfa3")+bytes(8192))
        with self.assertRaises(ValueError):
            self.s.finish(0,"islands","bottom40","persistent-white-3")
        self.assertEqual(self.s.state,"recording")
        first=self.s.finish(0,"islands","lower65","persistent-white-3")
        self.assertEqual(first["state"],"processing")
        worker=self.s.worker
        self.assertEqual(self.s.caption_mode,"islands")
        self.assertTrue(self.s.finish(
            0,"islands","lower65","persistent-white-3")["replayed"])
        with self.assertRaises(ValueError):
            self.s.finish(0,"all","lower65","persistent-white-3")
        self.assertIs(self.s.worker,worker)
        worker.join(timeout=4)
        self.assertEqual(self.s.snapshot()["state"],"completed")
        self.assertEqual(len(self.captured),1)
        self.assertNotIn("islands",json.dumps(self.s.snapshot()))

    def test_caption_islands_only_wraps_opted_in_processor(self):
        # The historical local HTTP contract intentionally runs with stdlib
        # Python only. Its browser/finish checks run there unchanged; the
        # optional OCR-wiring test runs in a separately installed free OCR
        # scratch environment, never on the global runner Python.
        try:
            import numpy  # noqa: F401
            import cv2  # noqa: F401
        except ImportError:
            self.skipTest("optional native OCR libraries unavailable in stdlib admission")
        from unittest.mock import patch
        from hardsub_extractor import tab_capture
        from hardsub_extractor.ocr.caption_islands import CaptionIslandEngine
        from hardsub_extractor.ocr.adaptive import AdaptiveOCREngine
        models=self.root/"mock-models"
        models.mkdir()
        manifest=self.root/"mock-manifest.json"
        manifest.write_text("{}")
        observed=[]
        def fake_extract(source,engine,base,**kwargs):
            observed.append((engine,kwargs))
            return SimpleNamespace(cue_count=1)
        with (patch("hardsub_extractor.runtime_paths.bundled_model_dir",
                    return_value=models),
              patch("hardsub_extractor.runtime_paths.bundled_manifest_path",
                    return_value=manifest),
              patch("hardsub_extractor.ocr.rapidocr_adapter.RapidOCRAdapter",
                    return_value=object()),
              patch("hardsub_extractor.pipeline.extract",
                    side_effect=fake_extract)):
            tab_capture.offline_ocr(
                self.root/"mock.webm",self.root/"island-result",
                language="ja",region_mode="lower65",
                caption_mode="islands",detection_mode="persistent-white-3")
            tab_capture.offline_ocr(
                self.root/"mock.webm",self.root/"default-result",
                language="ja")
        self.assertIsInstance(observed[0][0],CaptionIslandEngine)
        self.assertIsInstance(observed[0][0].engine,AdaptiveOCREngine)
        self.assertEqual(
            observed[0][1]["detector"].signature_mode,"persistent_white_3")
        self.assertIsInstance(observed[1][0],AdaptiveOCREngine)
        self.assertNotIn("detector",observed[1][1])
        self.assertNotIn("roi",observed[1][1])

    def test_ocr_progress_is_count_only_and_hidden_after_processing(self):
        # Progress is derived from the already running local OCR monitor;
        # no page pixels, recognized captions or private file paths cross HTTP.
        self.assertIsNone(self.s.snapshot()["ocr_progress"])
        with self.s.lock:
            self.s.state="processing"
        self.s.mark_progress(SimpleNamespace(
            stage="recognizing",completed=8,media_seconds=1.25,
            text="PRIVATE_SUBTITLE_DO_NOT_EXPOSE"))
        progress=self.s.snapshot()["ocr_progress"]
        self.assertEqual(progress,{
            "stage":"recognizing","completed":8,"media_seconds":1.25})
        self.s.mark_progress(SimpleNamespace(
            stage="recognizing",completed=4,media_seconds=.5))
        self.assertEqual(self.s.snapshot()["ocr_progress"]["completed"],8)
        self.assertEqual(self.s.snapshot()["ocr_progress"]["media_seconds"],1.25)
        self.s.mark_progress(SimpleNamespace(
            stage="exporting",completed=2,media_seconds=None))
        self.assertEqual(self.s.snapshot()["ocr_progress"]["stage"],"exporting")
        self.assertNotIn("PRIVATE_SUBTITLE_DO_NOT_EXPOSE",
                         json.dumps(self.s.snapshot()))
        self.assertNotIn(str(self.root),json.dumps(self.s.snapshot()))
        self.assertIn("검사 ",PAGE)
        self.assertIn("SRT/ASS 파일 작성 중",PAGE)
        with self.s.lock:
            self.s.state="failed"
        self.assertIsNone(self.s.snapshot()["ocr_progress"])
        with self.s.lock:
            self.s.state="recording"
        self.assertIsNone(self.s.snapshot()["ocr_progress"])

    def test_direct_finish_offset_commits_once_without_http_transport(self):
        # The contaminated Windows loopback route can strip headers or drop
        # acknowledgements. Prove the session's one-time offset binding
        # without pretending that its real HTTP/Chromium path passed.
        self.s.append(0, bytes.fromhex("1a45dfa3") + bytes(8192))
        for invalid in (-1, 99 * 3600 * 1000 + 1, True, 1.5):
            with self.subTest(offset=invalid), self.assertRaises(ValueError):
                self.s.finish(offset_ms=invalid)
        self.assertEqual(self.s.state, "recording")
        self.assertIsNone(self.s.worker)
        first = self.s.finish(offset_ms=1201234)
        self.assertFalse(first["replayed"])
        self.assertEqual(self.s.output_offset_ms, 1201234)
        worker = self.s.worker
        self.assertIsNotNone(worker)
        replay = self.s.finish(offset_ms=1201234)
        self.assertTrue(replay["replayed"])
        self.assertIs(self.s.worker, worker)
        with self.assertRaisesRegex(ValueError, "capture_finish_offset_replay_mismatch"):
            self.s.finish(offset_ms=1201235)
        worker.join(timeout=3)
        self.assertFalse(worker.is_alive())
        self.assertEqual(self.s.state, "completed")
        self.assertEqual(len(self.captured), 1)
        self.assertEqual(len(list(self.out.glob("*.srt"))), 1)
        self.assertEqual(len(list(self.out.glob("*.ass"))), 1)

    def test_original_start_offset_is_pinned_on_finish_retry(self):
        self.s.append(0,bytes.fromhex("1a45dfa3")+bytes(8192))
        # A corrupt or ambiguous offset must never start OCR.
        invalid=(
            "/finish?offset_ms=-1",
            "/finish?offset_ms=356400001",
            "/finish?offset_ms=12&offset_ms=13",
            "/finish?offset_ms=12&unsafe=1",
            "/finish?offset_ms=1.5",
            "/finish?offset_ms=",
        )
        for path in invalid:
            self.assertIn(self.request(path,b"")[0],(0,409))
        self.assertEqual(self.s.state,"recording")
        self.assertIsNone(self.s.worker)
        self.assertEqual(self.s.output_offset_ms,0)

        # 20m 01.234s is frozen before the OCR worker starts.
        for _ in range(3):
            if self.request("/finish?offset_ms=1201234",b"")[0]==200:
                break
        else:
            self.fail("offset finish was never acknowledged")
        worker=self.s.worker
        self.assertEqual(self.s.output_offset_ms,1201234)
        self.assertIsNotNone(worker)
        for _ in range(3):
            if self.request("/finish?offset_ms=1201234",b"")[0]==200:
                break
        else:
            self.fail("identical offset finish replay was never acknowledged")
        self.assertIs(self.s.worker,worker)
        self.assertIn(self.request("/finish?offset_ms=0",b"")[0],(0,409))
        self.assertIn(self.request("/finish",b"")[0],(0,409))
        self.assertIs(self.s.worker,worker)
        worker.join(timeout=3)
        self.assertEqual(len(self.captured),1)
        self.assertEqual(self.s.local_result()["state"],"completed")
        # Ordinary browser status must not reveal owner paths or playback position.
        state=json.dumps(self.s.snapshot())
        self.assertNotIn("output_offset_ms",state)
        self.assertNotIn("1201234",state)

    def test_source_offset_form_has_clear_user_gesture(self):
        self.assertIn('id="offset" value="00:00:00.000"',PAGE)
        self.assertIn("parseOffsetMillis(offsetInput.value)",PAGE)
        self.assertIn("sourceOffsetMillis=parseOffsetMillis",PAGE)
        self.assertIn('"/finish?offset_ms="+sourceOffsetMillis',PAGE)
        self.assertIn("offsetInput.disabled=true",PAGE)

    def test_multi_hour_limit_respects_disk_reserve_without_writing(self):
        from hardsub_extractor import tab_capture as tab
        sample=bytes.fromhex("1a45dfa3")+bytes(8192)
        self.assertGreaterEqual(MAX_TOTAL,8*1024*1024*1024)
        with patch.object(tab.shutil,"disk_usage",
                          return_value=SimpleNamespace(
                              free=tab.MIN_FREE_BYTES+len(sample)-1)):
            self.assertIn(self.request("/chunk?i=0",sample)[0],(0,507))
        self.assertEqual(self.s.next_index,0)
        self.assertEqual(self.s.total,0)
        self.assertEqual(self.s.state,"recording")

        with patch.object(tab,"MAX_TOTAL",len(sample)-1):
            self.assertIn(self.request("/chunk?i=0",sample)[0],(0,413))
        self.assertEqual(self.s.total,0)
        self.assertFalse(list(self.out.iterdir()))

        self.s.append(0,sample)
        report=self.s.snapshot()
        self.assertEqual(report["recording_bytes"],len(sample))
        self.assertEqual(report["recording_limit_bytes"],MAX_TOTAL)
        self.assertNotIn(str(self.root),json.dumps(report))

    def test_short_disk_write_marks_recording_unusable_not_replayable(self):
        sample=bytes.fromhex("1a45dfa3")+bytes(8192)
        self.s.writer.close()
        class PartialWriter:
            closed=False
            def write(self, payload):
                return len(payload)-1
            def close(self):
                self.closed=True
        self.s.writer=PartialWriter()
        with self.assertRaises(OSError):
            self.s.append(0,sample)
        self.assertEqual(self.s.state,"failed")
        self.assertEqual(self.s.snapshot()["error_type"],"OSError")
        self.assertEqual(self.s.next_index,0)
        self.assertEqual(self.s.total,0)
        self.assertTrue(self.s.writer.closed)
        with self.assertRaises(ValueError):
            self.s.append(0,sample)
        self.assertFalse(list(self.out.iterdir()))

    def test_short_capture_does_not_claim_completion(self):
        tiny=bytes.fromhex("1a45dfa3")+b"x"
        # Windows may drop the HTTP ACK after this valid chunk is committed.
        # Retry the exact same bytes; the sink must not duplicate them.
        codes=[self.request("/chunk?i=0",tiny)[0]]
        for _ in range(2):
            if codes[-1]==200:
                break
            codes.append(self.request("/chunk?i=0",tiny)[0])
        self.assertIn(200,codes,"bounded identical replay never acknowledged")
        self.assertEqual(self.s.next_index,1)
        self.assertEqual(self.s.total,len(tiny))
        self.assertIn(self.request("/finish",b"")[0],(0,409))
        self.assertEqual(self.s.snapshot()["state"],"recording")
        self.assertEqual(len(list(self.out.iterdir())),0)


class BrowserTimeOffsetWriterTests(unittest.TestCase):
    def test_real_srt_ass_writers_receive_identical_original_timeline(self):
        # This is a true pipeline->real SRT/ASS publication test, using
        # synthetic Cue objects to avoid media, model, browser and network.
        if (importlib.util.find_spec("numpy") is None
                or importlib.util.find_spec("cv2") is None):
            self.skipTest("optional native OCR dependencies unavailable")
        from hardsub_extractor import pipeline
        from hardsub_extractor.models import Cue
        from hardsub_extractor.extraction_progress import ExtractionMonitor

        with tempfile.TemporaryDirectory() as scratch:
            base=Path(scratch)/"shifted"
            orig=Cue(1.25,2.50,"VISIBLE SUBTITLE",.92)
            def output(name, offset=None):
                with (patch.object(pipeline,"FFmpegPTSFrameSource",return_value=iter(())),
                      patch.object(pipeline,"segment_frames",return_value=[]),
                      patch.object(pipeline,"recognize_segments",return_value=[orig]),
                      patch.object(pipeline,"merge_cues",side_effect=list)):
                    kw={} if offset is None else {"output_time_offset_ms":offset}
                    return pipeline.extract(
                        "unused-local-video",object(),Path(scratch)/name,
                        monitor=ExtractionMonitor(exclusive_outputs=True),**kw)

            report=output("shifted",20*60*1000)
            self.assertEqual(report.cue_count,1)
            srt=report.srt_path.read_text(encoding="utf-8")
            ass=report.ass_path.read_text(encoding="utf-8")
            self.assertIn("00:20:01,250 --> 00:20:02,500",srt)
            self.assertIn("Dialogue: 0,0:20:01.25,0:20:02.50",ass)
            self.assertIn("VISIBLE SUBTITLE",srt)
            self.assertIn("VISIBLE SUBTITLE",ass)
            self.assertEqual((orig.start,orig.end),(1.25,2.5))

            legacy=output("legacy")
            self.assertIn("00:00:01,250 --> 00:00:02,500",
                          legacy.srt_path.read_text(encoding="utf-8"))
            self.assertIn("Dialogue: 0,0:00:01.25,0:00:02.50",
                          legacy.ass_path.read_text(encoding="utf-8"))
            for bad in (-1,99*3600*1000+1,True,1.25,"3000"):
                with self.subTest(bad=bad):
                    with self.assertRaises(ValueError):
                        output("invalid",bad)
            self.assertFalse((Path(scratch)/"invalid.srt").exists())
            self.assertFalse((Path(scratch)/"invalid.ass").exists())



if __name__=="__main__":
    unittest.main()
