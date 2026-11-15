"""Deterministic bounded-memory streaming OCR regression, no FFmpeg/network.

The legacy local/EpSite path remains materialized and unchanged. These tests
compare its exact evidence, cue text and timestamp output against the streaming
public video fallback. Frame arrays are tracked by weak reference to prove
that they are no longer all retained before recognition or in the evidence.
"""
from __future__ import annotations

from pathlib import Path
import tempfile
import unittest
from unittest import mock
from weakref import WeakValueDictionary

import numpy as np

from hardsub_extractor.detector import (
    DetectorConfig, iter_segment_frames, segment_frames,
)
from hardsub_extractor.extraction_progress import ExtractionMonitor
from hardsub_extractor.models import Frame, OCRObservation
from hardsub_extractor.pipeline import extract


def frames(count=800):
    for i in range(count):
        # One phase change each 10 input frames. The patched signature below
        # isolates the segment/event logic from irrelevant native CV2 details.
        image = np.full((64, 128, 3), 255 if (i // 10) % 2 else 0, dtype=np.uint8)
        yield Frame(i / 8.0, image)


def signature(image, width=320):
    return np.full((4, 4), image[0, 0, 0] > 0, dtype=np.bool_)


class TrackingOCR:
    def __init__(self, live=None, fail=False):
        self.live = live
        self.max_alive = 0
        self.calls = 0
        self.fail = fail

    def recognize(self, frame):
        self.calls += 1
        if self.live is not None:
            self.max_alive = max(self.max_alive, len(self.live))
        if self.fail:
            raise RuntimeError("simulated recognition failure")
        return OCRObservation(
            "Long-form caption reference", 0.99, frame.pts,
        )


class StreamingBoundedMemoryTests(unittest.TestCase):
    def test_streaming_detector_matches_original_evidence_exactly(self):
        with mock.patch("hardsub_extractor.detector.subtitle_signature", new=signature):
            original = segment_frames(frames(480))
            streaming = list(iter_segment_frames(frames(480)))
        self.assertGreater(len(original), 12)
        self.assertEqual(len(original), len(streaming))
        for a, b in zip(original, streaming):
            self.assertEqual((a.start, a.end, a.diagnostics),
                             (b.start, b.end, b.diagnostics))
            self.assertEqual([x.pts for x in a.samples], [x.pts for x in b.samples])
            for fa, fb in zip(a.samples, b.samples):
                np.testing.assert_array_equal(fa.image, fb.image)

    def test_streaming_pipeline_exactly_matches_materialized_srt_ass(self):
        with tempfile.TemporaryDirectory() as td:
            with mock.patch("hardsub_extractor.detector.subtitle_signature", new=signature), mock.patch(
                "hardsub_extractor.pipeline.FFmpegPTSFrameSource",
                side_effect=lambda *args, **kwargs: frames(480),
            ):
                legacy = extract("ignored", TrackingOCR(), Path(td) / "old")
                new = extract(
                    "ignored", TrackingOCR(), Path(td) / "new",
                    streaming=True,
                )
            self.assertEqual(legacy.segment_count, new.segment_count)
            self.assertEqual(legacy.cue_count, new.cue_count)
            self.assertGreater(legacy.cue_count, 0)
            self.assertEqual(legacy.srt_path.read_bytes(), new.srt_path.read_bytes())
            self.assertEqual(legacy.ass_path.read_bytes(), new.ass_path.read_bytes())

    def test_800_frames_do_not_accumulate_in_public_streaming_ocr(self):
        live = WeakValueDictionary()
        def generated():
            for index in range(800):
                image = np.full(
                    (64, 128, 3), 255 if (index // 10) % 2 else 0,
                    dtype=np.uint8,
                )
                live[index] = image
                yield Frame(index / 8.0, image)

        engine = TrackingOCR(live=live)
        with tempfile.TemporaryDirectory() as td:
            with mock.patch("hardsub_extractor.detector.subtitle_signature", new=signature), mock.patch(
                "hardsub_extractor.pipeline.FFmpegPTSFrameSource",
                side_effect=lambda *args, **kwargs: generated(),
            ):
                monitor = ExtractionMonitor(exclusive_outputs=True)
                result = extract(
                    "ignored", engine, Path(td) / "bounded",
                    streaming=True, monitor=monitor,
                )
            self.assertGreater(result.segment_count, 25)
            self.assertGreater(engine.calls, 60)
            self.assertLessEqual(engine.max_alive, 24)
            self.assertTrue(result.srt_path.is_file())
            self.assertTrue(result.ass_path.is_file())
        self.assertLessEqual(len(live), 1)

    def test_streaming_model_error_closes_decoder_and_preserves_files(self):
        # First, the exclusive preflight MUST reject existing user output
        # before initializing the frame iterator or touching the OCR engine.
        # Separately, on a free destination, model failure MUST close the
        # decoder and leave no partial SRT/ASS. These are distinct contracts.
        closed = []
        decoder_created = []
        def generated():
            try:
                yield from frames(800)
            finally:
                closed.append(True)
        def source(*args, **kwargs):
            decoder_created.append(True)
            return generated()
        with tempfile.TemporaryDirectory() as td:
            out = Path(td) / "public"
            prior = out.with_suffix(".srt")
            prior.write_bytes(b"USER_SUBTITLE")
            with mock.patch("hardsub_extractor.detector.subtitle_signature", new=signature), mock.patch(
                "hardsub_extractor.pipeline.FFmpegPTSFrameSource", side_effect=source,
            ):
                occupied_ocr = TrackingOCR(fail=True)
                with self.assertRaises(FileExistsError):
                    extract(
                        "ignored", occupied_ocr, out,
                        streaming=True, monitor=ExtractionMonitor(exclusive_outputs=True),
                    )
                self.assertEqual(decoder_created, [])
                self.assertEqual(occupied_ocr.calls, 0)
                self.assertEqual(closed, [])
                self.assertEqual(prior.read_bytes(), b"USER_SUBTITLE")
                self.assertFalse(out.with_suffix(".ass").exists())

                empty_out = Path(td) / "separate-model-error"
                failing_ocr = TrackingOCR(fail=True)
                with self.assertRaisesRegex(RuntimeError, "simulated recognition"):
                    extract(
                        "ignored", failing_ocr, empty_out,
                        streaming=True, monitor=ExtractionMonitor(exclusive_outputs=True),
                    )
                self.assertGreater(failing_ocr.calls, 0)
                self.assertEqual(decoder_created, [True])
                self.assertEqual(closed, [True])
                self.assertFalse(empty_out.with_suffix(".srt").exists())
                self.assertFalse(empty_out.with_suffix(".ass").exists())
            self.assertEqual(prior.read_bytes(), b"USER_SUBTITLE")

    def test_open_ended_progress_reports_no_guessed_total(self):
        events = []
        monitor = ExtractionMonitor(events.append, ocr_interval=2)
        wrapped = monitor.ocr(TrackingOCR(), total=None)
        frame = next(frames(1))
        for _ in range(4):
            wrapped.recognize(frame)
        self.assertEqual(
            [(x.stage, x.completed, x.total) for x in events],
            [
                ("recognizing", 0, None),
                ("recognizing", 2, None),
                ("recognizing", 4, None),
            ],
        )

    def test_streaming_on_empty_source_emits_no_artifacts(self):
        with tempfile.TemporaryDirectory() as td:
            with mock.patch(
                "hardsub_extractor.pipeline.FFmpegPTSFrameSource",
                return_value=iter(()),
            ):
                with self.assertRaisesRegex(RuntimeError, "인식된 자막이 없습니다"):
                    extract(
                        "ignored", TrackingOCR(), Path(td) / "empty",
                        streaming=True,
                    )
            self.assertFalse(list(Path(td).iterdir()))


if __name__ == "__main__":
    unittest.main()
