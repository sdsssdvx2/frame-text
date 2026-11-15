"""Post-export *major* cue review matches GUI hints and can page forward.

No real media or credentials. Source SRT files are never edited by review;
source-frame images only arise after explicit user consent.
"""
from __future__ import annotations

from hashlib import sha256
from pathlib import Path
import queue
import tempfile
import threading
import unittest
from unittest import mock

from hardsub_extractor import review_pack as review
from hardsub_extractor.gui import ExtractorGUI
from hardsub_extractor.models import Cue
from hardsub_extractor.quality_audit import AuditOptions, audit_subtitles


class Var:
    def __init__(self, value):
        self.value = value

    def get(self):
        return self.value

    def set(self, value):
        self.value = value


def review_gui(video, base):
    gui = ExtractorGUI.__new__(ExtractorGUI)
    for name, value in {
        "source_mode_var": "local",
        "url_var": str(video),
        "out_var": str(base),
        "review_mode_var": "추출 후 검토 권장",
        "review_after_var": "0",
        "ocr_language_var": "zh",
        "status_var": "준비됨",
    }.items():
        setattr(gui, name, Var(value))
    gui.worker = None
    gui._pending_close = False
    gui._cancel_event = threading.Event()
    gui.events = queue.Queue()
    gui.progress_bar = mock.Mock()
    gui.start_button = mock.Mock()
    gui.cancel_button = mock.Mock()
    gui.review_button = mock.Mock()
    gui._log = mock.Mock()
    return gui


class PostExportQualityHandoffTests(unittest.TestCase):
    def test_does_not_mix_script_only_warning_with_major_recommendation(self):
        cues = [
            Cue(0.5, 1.5, "SOME ENGLISH TITLE"),
            Cue(2, 3, "中文字幕"),
            Cue(4, 4.1, "字幕A"),
            Cue(5, 5.1, "字幕B"),
        ]
        report = audit_subtitles(cues, options=AuditOptions(
            requested_language="zh", video_duration_seconds=8,
        ))
        flagged = review.select_cues(cues, report, "flagged", 8)
        quality = review.select_cues(cues, report, "quality", 8)
        self.assertIn(1, [i for i, _ in flagged])
        self.assertEqual([i for i, _ in quality], [3, 4])
        self.assertEqual(
            review.select_cues(cues, report, "quality", 8, after_cue=3)[0][0],
            4,
        )
        self.assertTrue(all(
            codes == ["very_short_duration_review"] for _, codes in quality
        ))

    def test_private_frame_pack_opens_only_major_quality_cues(self):
        import json
        from io import BytesIO
        from PIL import Image
        image_bytes = BytesIO()
        Image.new("RGB", (160, 90)).save(image_bytes, "JPEG")
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            video, srt, dest = root / "source.mp4", root / "captions.srt", root / "private-frames"
            video.write_bytes(b"source unmodified")
            srt.write_text(
                "1\n00:00:00,500 --> 00:00:01,500\nENGLISH SCENE TITLE\n\n"
                "2\n00:00:02,000 --> 00:00:03,000\n中文字幕\n\n"
                "3\n00:00:04,000 --> 00:00:04,100\n字幕文本\n",
                encoding="utf-8",
            )
            original = (video.read_bytes(), srt.read_bytes())
            with (
                mock.patch.object(review, "_probe_local_video_duration", return_value=9.0),
                mock.patch.object(review, "extract_frame", return_value=image_bytes.getvalue()) as frames,
            ):
                result = review.create_pack(
                    video, srt, dest, mode="quality", max_cues=8, language="zh",
                )
            self.assertEqual(result["cue_count"], 1)
            self.assertEqual(frames.call_count, 2)
            record = json.loads((dest / "index.json").read_text("utf-8"))
            self.assertEqual(record["mode"], "quality")
            self.assertEqual([x["cue"] for x in record["review_cues"]], [3])
            self.assertEqual(
                record["review_cues"][0]["warning_codes"], ["very_short_duration_review"]
            )
            self.assertNotIn("ENGLISH SCENE TITLE", str(record))
            self.assertEqual((video.read_bytes(), srt.read_bytes()), original)

    def test_exact_exported_srt_bound_to_video_before_private_review(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            video, base = root / "source.mp4", root / "result"
            srt = base.with_suffix(".srt")
            video.write_bytes(b"private source video preserved")
            srt.write_text(
                "1\n00:00:01,000 --> 00:00:03,000\n"
                "private dialogue\n", encoding="utf-8",
            )
            original = video.read_bytes(), srt.read_bytes()
            gui = review_gui(video, base)
            with mock.patch(
                "hardsub_extractor.local_ocr_quality_signals.local_ocr_quality_signals",
                return_value={
                    "review_signal_count": 1,
                    "review_cue_numbers": (1,),
                },
            ):
                gui._emit_local_ocr_quality(video, srt, "zh")
            events = list(gui.events.queue)
            self.assertIn(("ocr_quality_signal", 1), events)
            self.assertIn(("ocr_quality_cue_numbers", (1,)), events)
            proof = [payload for kind, payload in events if kind == "ocr_quality_bound_hints"]
            self.assertEqual(len(proof), 1)
            self.assertEqual(proof[0][0:2], (video, srt))
            self.assertEqual(proof[0][2], sha256(original[1]).digest())
            gui._quality_review_hints = proof[0]
            gui.review_after_var.set("0")
            with (
                mock.patch("hardsub_extractor.gui.messagebox.askyesno", return_value=True),
                mock.patch("hardsub_extractor.gui.threading.Thread") as thread,
            ):
                gui._start_frame_review()
            thread.return_value.start.assert_called_once()
            args = thread.call_args.kwargs["args"]
            self.assertEqual(args[:2], (video, srt))
            self.assertEqual(args[3:6], ("quality", 0, "zh"))
            self.assertEqual(args[2].name, "result.review-quality-after-00000")
            self.assertEqual((video.read_bytes(), srt.read_bytes()), original)
            self.assertFalse(args[2].exists())

            gui.worker = None
            srt.write_text(srt.read_text(encoding="utf-8") + "\n", encoding="utf-8")
            with (
                mock.patch("hardsub_extractor.gui.messagebox.showerror") as error,
                mock.patch("hardsub_extractor.gui.messagebox.askyesno") as allow,
                mock.patch("hardsub_extractor.gui.threading.Thread") as thread,
            ):
                gui._start_frame_review()
            error.assert_called_once()
            allow.assert_not_called()
            thread.assert_not_called()

    def test_quality_review_without_prior_ocr_proof_never_starts_decoding(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            video, base = root / "input.mp4", root / "captions"
            video.write_bytes(b"unmodified")
            base.with_suffix(".srt").write_bytes(b"private subtitle")
            gui = review_gui(video, base)
            with (
                mock.patch("hardsub_extractor.gui.messagebox.showerror") as error,
                mock.patch("hardsub_extractor.gui.threading.Thread") as thread,
                mock.patch("hardsub_extractor.gui.messagebox.askyesno") as allow,
            ):
                gui._start_frame_review()
            error.assert_called_once()
            allow.assert_not_called()
            thread.assert_not_called()


if __name__ == "__main__":
    unittest.main()
