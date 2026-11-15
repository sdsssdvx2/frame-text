"""Single-pass temporal watermark suggestions -> existing private GUI frame review.

Real local OCR inference remains in separate existing Windows acceptance; these
tests drive real diagnosis/review logic with synthetic OCR metadata and a
synthetic local file, never download models or user media.
"""
from __future__ import annotations

from hashlib import sha256
from io import BytesIO
import json
from pathlib import Path
from types import SimpleNamespace
from unittest import mock
import tempfile
import unittest

from PIL import Image

from hardsub_extractor.models import Cue
from hardsub_extractor.temporal_review import (
    TemporalReviewRecorder, ChainedSegmentAudits,
)
from hardsub_extractor import review_pack, gui as gui_module
from test_gui_frame_review import fake_gui, setup_files


def samples(recorder, *, include_logo=True, dialogue=True):
    for i in range(29):
        pts=i*7.0/28
        text=[]
        detections=[]
        if dialogue and 1.5<=pts<=3.5:
            text.append("첫 번째 대사")
            detections.append({"line":0,"cx":.51,"cy":.30})
        if dialogue and 4.3<=pts<=6.3:
            text.append("두 번째 대사")
            detections.append({"line":0,"cx":.51,"cy":.30})
        if include_logo:
            detections.append({"line":len(text),"cx":.36,"cy":.76})
            text.append("방송 안내")
        if not text:
            continue
        observation=SimpleNamespace(pts=pts,text="\n".join(text),
                                    metadata={"detections":detections})
        segment=SimpleNamespace(start=pts,end=pts+.1)
        recorder.record_segment(i,segment,[observation],
                                SimpleNamespace(text=observation.text))


def baseline_cues():
    return [
        Cue(0.0,1.0,"방송 안내"),
        Cue(1.5,3.5,"첫 번째 대사\n방송 안내"),
        Cue(4.3,6.3,"두 번째 대사\n방송 안내"),
        Cue(6.3,7.0,"방송 안내"),
    ]


def subtitle_text():
    return (
        "1\n00:00:00,000 --> 00:00:01,000\n방송 안내\n\n"
        "2\n00:00:01,500 --> 00:00:03,500\n"
        "첫 번째 대사\n방송 안내\n\n"
        "3\n00:00:04,300 --> 00:00:06,300\n"
        "두 번째 대사\n방송 안내\n\n"
        "4\n00:00:06,300 --> 00:00:07,000\n방송 안내\n"
    )


class TemporalReviewGUITests(unittest.TestCase):
    def test_reuses_existing_three_window_policy_without_editing_cues(self):
        collector=TemporalReviewRecorder()
        original=baseline_cues()
        values=tuple((x.start,x.end,x.text) for x in original)
        samples(collector)
        self.assertEqual(collector.cue_numbers(original,7.0),(1,2,3,4))
        self.assertEqual(tuple((x.start,x.end,x.text) for x in original),values)
        clean=TemporalReviewRecorder()
        samples(clean,include_logo=False)
        self.assertEqual(clean.cue_numbers(original,7.0),())

    def test_bounded_overflow_cannot_be_mistaken_for_no_suspects(self):
        collector=TemporalReviewRecorder(max_lines=5)
        samples(collector)
        self.assertTrue(collector.incomplete)
        self.assertEqual(collector.traces,[])
        with self.assertRaisesRegex(ValueError,"incomplete"):
            collector.cue_numbers(baseline_cues(),7.0)

    def test_existing_diagnostics_and_new_ram_collector_both_observe_once(self):
        original=mock.Mock()
        collector=TemporalReviewRecorder()
        combined=ChainedSegmentAudits(original,collector)
        samples(combined)
        self.assertEqual(original.record_segment.call_count,29)
        self.assertEqual(collector.cue_numbers(baseline_cues(),7.0),
                         (1,2,3,4))

    def test_temporal_review_pack_uses_only_pinned_cue_numbers_and_private_images(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td)
            video,srt=root/"video.mp4",root/"captions.srt"
            video.write_bytes(b"not a real user video")
            srt.write_text(subtitle_text(),encoding="utf-8")
            original=(video.read_bytes(),srt.read_bytes())
            pic=BytesIO()
            Image.new("RGB",(320,180),(25,50,90)).save(pic,"JPEG")
            destination=root/"private-timeline-review"
            with (mock.patch.object(review_pack,"_probe_local_video_duration",
                                    return_value=7.0),
                  mock.patch.object(review_pack,"extract_frame",
                                    return_value=pic.getvalue()) as draw):
                result=review_pack.create_pack(
                    video,srt,destination,mode="temporal",
                    language="ko",max_cues=2,after_cue=1,
                    temporal_cue_numbers=(1,2,3,4))
            self.assertEqual((result["cue_count"],result["last_cue"]),(2,3))
            self.assertEqual(draw.call_count,4)
            index=json.loads((destination/"index.json").read_text("utf-8"))
            self.assertEqual(index["mode"],"temporal")
            self.assertEqual([x["cue"] for x in index["review_cues"]],[2,3])
            self.assertTrue(all(
                "temporal_persistent_overlay_review" in x["warning_codes"]
                for x in index["review_cues"]))
            self.assertNotIn("방송 안내",json.dumps(index,ensure_ascii=False))
            self.assertEqual((video.read_bytes(),srt.read_bytes()),original)

    def test_temporal_review_requires_exact_current_video_and_srt(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td)
            video,base,srt=setup_files(root)
            gui=fake_gui(video,base,scope="반복 워터마크 후보")
            stat=video.stat()
            gui._temporal_review_hints=(
                video,srt,sha256(srt.read_bytes()).digest(),
                stat.st_size,stat.st_mtime_ns,(1,))
            with (mock.patch.object(gui_module.messagebox,"askyesno",
                                    return_value=True) as allow,
                  mock.patch.object(gui_module.threading,"Thread") as launch):
                gui._start_frame_review()
            self.assertEqual(launch.call_args.kwargs["args"][-1],(1,))
            self.assertEqual(launch.call_args.kwargs["args"][-4:-1],
                             ("temporal",0,"zh"))
            self.assertEqual(allow.call_count,1)
            launch.return_value.start.assert_called_once()

            # The first mocked thread is deliberately not scheduled. Simulate
            # its normal completion before checking the next user gesture.
            gui.worker=None
            # Stale SRT can never consume the previous video's candidates.
            srt.write_text(srt.read_text("utf-8")+"\n",encoding="utf-8")
            with (mock.patch.object(gui_module.messagebox,"showerror") as error,
                  mock.patch.object(gui_module.messagebox,"askyesno") as allow,
                  mock.patch.object(gui_module.threading,"Thread") as launch):
                gui._start_frame_review()
            error.assert_called_once()
            allow.assert_not_called()
            launch.assert_not_called()

    def test_absent_candidate_proof_requires_no_fresh_ocr_or_output(self):
        with tempfile.TemporaryDirectory() as td:
            video,base,srt=setup_files(Path(td))
            gui=fake_gui(video,base,scope="반복 워터마크 후보")
            with (mock.patch.object(gui_module.messagebox,"showerror") as error,
                  mock.patch.object(gui_module.threading,"Thread") as launch):
                gui._start_frame_review()
            error.assert_called_once()
            launch.assert_not_called()
            stat=video.stat()
            gui._temporal_review_hints=(
                video,srt,sha256(srt.read_bytes()).digest(),
                stat.st_size,stat.st_mtime_ns,())
            with (mock.patch.object(gui_module.messagebox,"showinfo") as info,
                  mock.patch.object(gui_module.threading,"Thread") as launch):
                gui._start_frame_review()
            info.assert_called_once()
            launch.assert_not_called()

    def test_local_ocr_worker_emits_bound_hints_using_one_model_pass(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td)
            video=root/"video.mp4"
            video.write_bytes(b"fake")
            base=root/"captions"
            gui=fake_gui(video,base,scope="반복 워터마크 후보")
            gui._active_temporal_review_opt_in=True
            models=root/"models"
            models.mkdir()
            class Report:
                cue_count=4
                srt_path=base.with_suffix(".srt")
                ass_path=base.with_suffix(".ass")
            def fake_extract(source,engine,out,**kwargs):
                self.assertEqual(str(video),source)
                self.assertTrue(kwargs["streaming"])
                self.assertIsInstance(kwargs["diagnostics"],
                                      ChainedSegmentAudits)
                samples(kwargs["diagnostics"])
                Report.srt_path.write_text(subtitle_text(),encoding="utf-8")
                Report.ass_path.write_text("ASS output",encoding="utf-8")
                return Report()
            with (mock.patch.object(gui_module,"acquire_local_embedded_text",
                                    return_value=None),
                  mock.patch.object(gui_module,"acquire_local_sidecar_text",
                                    return_value=None),
                  mock.patch.object(gui_module,"acquire_local_embedded_image",
                                    return_value=None),
                  mock.patch.object(gui_module,"bundled_model_dir",
                                    return_value=models),
                  mock.patch.object(gui_module,"bundled_manifest_path",
                                    return_value=root/"manifest.json"),
                  mock.patch.object(gui_module,"RapidOCRAdapter",
                                    return_value=mock.Mock()),
                  mock.patch.object(gui_module,"AdaptiveOCREngine",
                                    return_value=mock.Mock()),
                  mock.patch.object(gui_module,"extract_local_video",
                                    side_effect=fake_extract) as extract,
                  mock.patch("hardsub_extractor.frame_source.probe_video",
                             return_value=SimpleNamespace(duration=7.0))):
                gui._worker_local(str(video),base,False,ocr_language="ko")
            self.assertEqual(extract.call_count,1)
            events=[]
            while not gui.events.empty():
                events.append(gui.events.get_nowait())
            found=[data for name,data in events
                   if name=="temporal_review_ready"]
            self.assertEqual(len(found),1)
            self.assertEqual(found[0][-1],(1,2,3,4))
            self.assertEqual(found[0][2],sha256(Report.srt_path.read_bytes()).digest())
            self.assertEqual(events[-1][0],"done")


if __name__=="__main__":
    unittest.main()
