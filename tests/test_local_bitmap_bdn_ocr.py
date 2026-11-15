"""Bounded BDN PNG timed-text parser and reuse of HSE multilingual OCR.

No real media, network or model assets; this is a targeted source-level
contract. The separate pinned real-video Windows run establishes that
seconv produced genuine PGS/VobSub source bitmaps.
"""
from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from PIL import Image
import numpy as np
from types import SimpleNamespace

from hardsub_extractor.ocr.rapidocr_adapter import RapidOCRAdapter
from hardsub_extractor.models import Frame
from hardsub_extractor import local_image_subtitles as adapter
from hardsub_extractor.local_bitmap_bdn_ocr import (
    BitmapBdnOcrError, recognize_bdn_images, _image_to_bgr,
)
from hardsub_extractor.matroska_track_headers import MatroskaImageTrack
from hardsub_extractor.models import OCRObservation
from hardsub_extractor.subtitle_formats import parse_subtitle_bytes


def make_bdn(root: Path, names=("0001.png", "0002.png"), *, fps="23.976"):
    root.mkdir(parents=True, exist_ok=True)
    for name in names:
        if "/" in name or ".." in name:
            continue
        Image.new("RGBA", (160, 35), (255, 255, 255, 255)).save(root / name)
    event = "\n".join(
        '<Event InTC="00:00:%02d:00" OutTC="00:00:%02d:00" Forced="false">'
        '<Graphic Width="160" Height="35" X="100" Y="200">%s</Graphic></Event>'
        % (i+1, i+2, name) for i, name in enumerate(names)
    )
    xml = (
        '<?xml version="1.0" encoding="utf-8"?>\n'
        '<BDN Version="0.93"><Description>'
        '<Format VideoFormat="1920x1080" FrameRate="%s" DropFrame="false"/>'
        '</Description><Events>%s</Events></BDN>'
    ) % (fps, event)
    (root / "index.xml").write_text(xml, encoding="utf-8")
    return root


class StubOCR:
    def __init__(self, texts):
        self.texts = iter(texts)
        self.calls = 0
        self.pts = []

    def recognize(self, frame):
        self.calls += 1
        self.pts.append(frame.pts)
        self.shape = frame.image.shape
        return OCRObservation(next(self.texts), 0.97, frame.pts)


class BitmapBdnOcrTests(unittest.TestCase):
    def test_neutral_720_bitmap_canvas_contains_glyphs_and_preserves_source(self):
        with tempfile.TemporaryDirectory() as td:
            source = Path(td) / "caption.png"
            original = Image.new("RGBA", (640, 100), (0, 0, 0, 0))
            from PIL import ImageDraw
            ImageDraw.Draw(original).rectangle(
                (90, 26, 290, 72), fill=(255, 255, 255, 255),
            )
            original.save(source)
            before = source.read_bytes()
            canvas = _image_to_bgr(source, white=False, neutral_canvas=True)
            self.assertEqual(canvas.shape, (720, 1280, 3))
            self.assertEqual(canvas[0, 0].tolist(), [59, 43, 36])
            self.assertEqual(canvas[620, 420].tolist(), [255, 255, 255])
            self.assertEqual(source.read_bytes(), before)

    def test_reads_two_original_bitmap_events_and_timestamps(self):
        with tempfile.TemporaryDirectory() as td:
            directory = make_bdn(Path(td) / "image-bitmap")
            ocr = StubOCR(["한글 자막", "日本語 자막"])
            cues = recognize_bdn_images(directory, ocr)
            self.assertEqual([x.text for x in cues], ["한글 자막", "日本語 자막"])
            self.assertEqual(len(cues), 2)
            self.assertEqual(ocr.shape, (35, 160, 3))
            self.assertEqual(ocr.calls, 2)
            self.assertAlmostEqual(cues[0].start, 1.001, places=3)
            self.assertAlmostEqual(cues[0].end, 2.002, places=3)
            self.assertEqual(ocr.pts, [x.start for x in cues])

    def test_bitmap_source_event_coverage_tracks_missing_ocr_without_fabricating(self):
        with tempfile.TemporaryDirectory() as td:
            directory = make_bdn(Path(td) / "source-images")
            audit = {"stale": 999}
            # The second graphic remains a real input event even when OCR
            # returns blank on both source background combinations.
            ocr = StubOCR(["한글 자막", "", ""])
            cues = recognize_bdn_images(directory, ocr, audit=audit)
            self.assertEqual([cue.text for cue in cues], ["한글 자막"])
            self.assertEqual(ocr.calls, 3)
            self.assertEqual(audit, {
                "source_bitmap_events": 2,
                "recognized_bitmap_events": 1,
                "unrecognized_bitmap_events": 1,
            })
            self.assertTrue((directory / "0002.png").is_file())

    def test_bitmap_full_coverage_audit_reports_all_source_events(self):
        with tempfile.TemporaryDirectory() as td:
            directory = make_bdn(Path(td) / "source-images")
            audit = {}
            cues = recognize_bdn_images(
                directory, StubOCR(["第一行", "第二行"]), audit=audit,
            )
            self.assertEqual(len(cues), 2)
            self.assertEqual(audit["source_bitmap_events"], 2)
            self.assertEqual(audit["recognized_bitmap_events"], 2)
            self.assertEqual(audit["unrecognized_bitmap_events"], 0)

    def test_rejects_path_traversal_entity_and_malformed_xml(self):
        with tempfile.TemporaryDirectory() as td:
            directory = make_bdn(Path(td) / "image-bitmap")
            orig = (directory / "index.xml").read_text("utf-8")
            for data in (
                orig.replace("0001.png", "../outside.png"),
                '<!DOCTYPE foo [<!ENTITY x SYSTEM "file:///secret">]>' + orig,
                orig.replace("<Events>", "<Events Invalid>"),
            ):
                with self.subTest(data=data[:35]):
                    (directory / "index.xml").write_text(data, "utf-8")
                    with self.assertRaises(BitmapBdnOcrError):
                        recognize_bdn_images(directory, StubOCR(["unexpected"]))
            self.assertFalse((Path(td) / "outside.png").exists())

    def test_rejects_duplicate_frames_and_wrong_frame_rate(self):
        with tempfile.TemporaryDirectory() as td:
            directory = make_bdn(Path(td) / "images",
                                 ("0001.png", "0001.png"))
            with self.assertRaises(BitmapBdnOcrError):
                recognize_bdn_images(directory, StubOCR(["unexpected"]))
            make_bdn(directory, fps="0")
            with self.assertRaises(BitmapBdnOcrError):
                recognize_bdn_images(directory, StubOCR(["unexpected"]))

    def test_blank_ocr_never_fabricates_cues(self):
        with tempfile.TemporaryDirectory() as td:
            directory = make_bdn(Path(td) / "images", ("0001.png",))
            with self.assertRaisesRegex(BitmapBdnOcrError, "no recognized"):
                recognize_bdn_images(directory, StubOCR(["", ""]))
            self.assertTrue((directory / "index.xml").exists())

    def test_cropped_bitmap_avoids_video_scene_position_filter(self):
        adapter_obj = RapidOCRAdapter.__new__(RapidOCRAdapter)
        adapter_obj.language = "ja"
        adapter_obj.korean_subtitle_mode = False
        adapter_obj.text_score = 0.45
        adapter_obj.engine = mock.Mock(return_value=SimpleNamespace(
            txts=["字幕です"], scores=[0.97],
            boxes=[[[90, 2], [190, 2], [190, 19], [90, 19]]],
        ))
        frame = Frame(1.25, np.zeros((90, 280, 3), dtype=np.uint8))
        # Global video-scene filters intentionally refuse upper text without
        # a second lower dialogue row; do not weaken those filters.
        self.assertEqual(adapter_obj.recognize(frame).text, "")
        bitmap = adapter_obj.recognize_cropped_bitmap(frame)
        self.assertEqual(bitmap.text, "字幕です")
        self.assertEqual(bitmap.pts, frame.pts)
        self.assertTrue(bitmap.metadata["bitmap_only"])

    def test_only_blank_bdn_event_uses_cropped_image_full_text_fallback(self):
        class SceneGuard:
            def __init__(self):
                self.calls = 0
                self.fallbacks = 0

            def recognize(self, frame):
                self.calls += 1
                return OCRObservation("", 0.0, frame.pts)

            def recognize_cropped_bitmap(self, frame):
                self.fallbacks += 1
                return OCRObservation("字幕です", 0.92, frame.pts)

        class Wrapped:
            def __init__(self):
                self.base = SceneGuard()

            def recognize(self, frame):
                return self.base.recognize(frame)

        with tempfile.TemporaryDirectory() as td:
            directory = make_bdn(Path(td) / "images", ("0001.png",))
            engine = Wrapped()
            cues = recognize_bdn_images(directory, engine)
            self.assertEqual([x.text for x in cues], ["字幕です"])
            self.assertEqual(engine.base.calls, 0)
            self.assertEqual(engine.base.fallbacks, 1)

    def test_bitmap_reader_preserves_both_rows_when_video_filter_keeps_only_lower(self):
        class PartialSceneOCR:
            def __init__(self):
                self.scene_calls = 0
                self.bitmap_calls = 0

            def recognize(self, frame):
                self.scene_calls += 1
                return OCRObservation("아래 줄", 0.99, frame.pts)

            def recognize_cropped_bitmap(self, frame):
                self.bitmap_calls += 1
                return OCRObservation("윗줄\n아래 줄", 0.96, frame.pts)

        with tempfile.TemporaryDirectory() as td:
            directory = make_bdn(Path(td) / "image-bitmap", ("0001.png",))
            engine = PartialSceneOCR()
            cues = recognize_bdn_images(directory, engine)
            self.assertEqual([cue.text for cue in cues], ["윗줄\n아래 줄"])
            self.assertEqual(engine.bitmap_calls, 1)
            self.assertEqual(engine.scene_calls, 0)
            self.assertAlmostEqual(cues[0].start, 1.001, places=3)

    def test_blank_bitmap_composites_still_allow_legacy_scene_rescue(self):
        class EmptyBitmapOCR:
            def __init__(self):
                self.bitmap_calls = 0
                self.scene_calls = 0

            def recognize_cropped_bitmap(self, frame):
                self.bitmap_calls += 1
                return OCRObservation("", 0.0, frame.pts)

            def recognize(self, frame):
                self.scene_calls += 1
                return OCRObservation("원본 자막", 0.91, frame.pts)

        with tempfile.TemporaryDirectory() as td:
            directory = make_bdn(Path(td) / "image-bitmap", ("0001.png",))
            engine = EmptyBitmapOCR()
            cues = recognize_bdn_images(directory, engine)
            self.assertEqual([cue.text for cue in cues], ["원본 자막"])
            # First normalized 720p, then original black and white
            # composites, before the existing scene OCR rescue.
            self.assertEqual(engine.bitmap_calls, 3)
            self.assertEqual(engine.scene_calls, 1)

    def test_multilingual_model_factory_is_lazy_and_outputs_provenance(self):
        with tempfile.TemporaryDirectory() as td:
            directory = Path(td)
            video = directory / "source.mkv"
            video.write_bytes(b"not decoded video in test")
            output = directory / "final" / "one-123456"
            ocr = StubOCR(["한국어 자막", "두 번째 한국어 자막"])
            created = []

            def new_ocr():
                created.append(True)
                return ocr

            def fake_seconv(command, scratch, env_patch, cancel):
                self.assertEqual(command[2], "bdnxml")
                self.assertIn("--track-number:5", command)
                self.assertEqual(command.count("--json"), 1)
                make_bdn(scratch / "image-bitmap")
                return json.dumps({"success": True}).encode("utf-8")

            source_track = MatroskaImageTrack(
                track_number=5, codec="hdmv_pgs_subtitle",
                language="kor", default=True, forced=False,
            )
            with (
                mock.patch.object(adapter, "_converter_executable",
                                  return_value=Path("seconv.exe")),
                mock.patch.object(adapter, "_track_candidates",
                                  return_value=[source_track]),
                mock.patch.object(adapter, "_ocr_backend",
                                  side_effect=AssertionError("separate Tesseract forbidden")),
                mock.patch.object(adapter, "_invoke", side_effect=fake_seconv),
            ):
                result = adapter.acquire_local_embedded_image(
                    video, output, language="ko", ocr_factory=new_ocr,
                )
            self.assertEqual(created, [True])
            self.assertEqual(result.mode, "local-embedded-image-ocr")
            self.assertEqual(result.cue_count, 2)
            self.assertEqual(result.missed_events, 0)
            self.assertEqual(video.read_bytes(), b"not decoded video in test")
            fmt, actual = parse_subtitle_bytes(result.srt_path.read_bytes(),
                                               source_url="result.srt")
            self.assertEqual(fmt, "srt")
            self.assertEqual([c.text for c in actual],
                             ["한국어 자막", "두 번째 한국어 자막"])
            self.assertTrue(result.ass_path.is_file())
            metadata = json.loads(result.provenance_path.read_text("utf-8"))
            self.assertEqual(metadata["ocr_engine"], "hse_multilingual_ppocr")
            self.assertEqual(metadata["bitmap_timing_precision"],
                             "BDN_frame_quantized")
            self.assertTrue(metadata["ocr_applied"])
            self.assertFalse(metadata["original_text_verified"])
            coverage = metadata["bitmap_ocr_event_coverage"]
            self.assertEqual(coverage["source_bitmap_events"], 2)
            self.assertEqual(coverage["recognized_bitmap_events"], 2)
            self.assertEqual(coverage["unrecognized_bitmap_events"], 0)
            self.assertTrue(coverage["all_source_events_recognized"])
            self.assertIn("not_text_accuracy", coverage["method"])

    def test_partial_bitmap_result_is_explicitly_marked_in_provenance(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            video = root / "source.mkv"
            video.write_bytes(b"unchanged original video")
            output = root / "exports" / "candidate"

            def seconv(command, scratch, env, cancel):
                make_bdn(scratch / "image-bitmap")
                return b'{"success":true}'

            selected = MatroskaImageTrack(
                track_number=5, codec="hdmv_pgs_subtitle",
                language="eng", default=True, forced=False,
            )
            with (
                mock.patch.object(adapter, "_converter_executable",
                                  return_value=Path("seconv.exe")),
                mock.patch.object(adapter, "_track_candidates",
                                  return_value=[selected]),
                mock.patch.object(adapter, "_invoke", side_effect=seconv),
            ):
                result = adapter.acquire_local_embedded_image(
                    video, output, language="en",
                    ocr_factory=lambda: StubOCR(["First subtitle", "", ""]),
                )
            self.assertEqual(result.cue_count, 1)
            self.assertEqual(result.missed_events, 1)
            self.assertEqual(video.read_bytes(), b"unchanged original video")
            metadata = json.loads(result.provenance_path.read_text("utf-8"))
            coverage = metadata["bitmap_ocr_event_coverage"]
            self.assertEqual(coverage["source_bitmap_events"], 2)
            self.assertEqual(coverage["recognized_bitmap_events"], 1)
            self.assertEqual(coverage["unrecognized_bitmap_events"], 1)
            self.assertFalse(coverage["all_source_events_recognized"])
            self.assertFalse(metadata["original_text_verified"])

    def test_scratch_budget_rejects_growth_and_symlinks(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root / "0001.png").write_bytes(b"12345")
            with mock.patch.object(adapter, "_MAX_SCRATCH_BYTES", 4):
                with self.assertRaisesRegex(
                    adapter.LocalImageSubtitleError, "storage budget",
                ):
                    adapter._check_scratch_budget(root)
            with mock.patch.object(adapter, "_MAX_SCRATCH_FILES", 0):
                with self.assertRaisesRegex(
                    adapter.LocalImageSubtitleError, "storage budget",
                ):
                    adapter._check_scratch_budget(root)
            try:
                (root / "escape.png").symlink_to(root / "0001.png")
            except OSError:
                return
            with self.assertRaisesRegex(
                adapter.LocalImageSubtitleError, "symlink",
            ):
                adapter._check_scratch_budget(root)

    def test_no_matching_track_does_not_initialize_expensive_model(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td)
            src = p / "sample.mkv"
            src.write_bytes(b"source")
            with (
                mock.patch.object(adapter, "_converter_executable",
                                  return_value=Path("seconv.exe")),
                mock.patch.object(adapter, "_track_candidates",
                                  return_value=[]),
                mock.patch.object(adapter, "_invoke") as invoked,
            ):
                self.assertIsNone(adapter.acquire_local_embedded_image(
                    src, p / "out", language="ja",
                    ocr_factory=lambda: self.fail("OCR loaded without source"),
                ))
                invoked.assert_not_called()


if __name__ == "__main__":
    unittest.main()
