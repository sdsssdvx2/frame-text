"""No-network proof of the alternative timed-OCR streaming core.

These tests use independent frame timestamps, not a third-party detector,
media file, real OCR model, browser session or packaged application. They
cover dozens/hundreds of intervals to prevent regressing to the previous
eight-interval / all-decoded-frames research limitation.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import tempfile
import unittest
import math
from unittest import mock

import numpy as np

from hardsub_extractor.models import Frame, OCRObservation
from hardsub_extractor.subtitle_formats import parse_subtitle_bytes
from hardsub_extractor import timed_ocr


@dataclass(frozen=True)
class Interval:
    start: float
    end: float
    text: str = "reference must never reach OCR"


class Recognizer:
    def __init__(self, recognize=lambda f: "국문\n둘째 줄"):
        self.calls = []
        self.recognize_text = recognize

    def recognize(self, frame):
        self.calls.append(frame.pts)
        return OCRObservation(self.recognize_text(frame), 0.99, frame.pts)


def sample_frames(count: int, *, fps=8.0, finalizer=None):
    try:
        for n in range(count):
            yield Frame(n / fps, np.zeros((6, 8, 3), np.uint8))
    finally:
        if finalizer is not None:
            finalizer.append(True)


class TimedOCRStreamingTests(unittest.TestCase):
    def run_fake(self, frames, intervals, adapter, base, **kwargs):
        with mock.patch.object(
            timed_ocr, "FFmpegPTSFrameSource", return_value=frames,
        ) as decoder:
            report = timed_ocr.extract_timed_ocr(
                "fixture.mp4", intervals, adapter, base, **kwargs,
            )
        self.assertEqual(decoder.call_count, 1)
        self.assertEqual(decoder.call_args.kwargs["sample_fps"], 8.0)
        return report

    def test_two_intervals_exact_source_timeline_and_physical_two_rows(self):
        spans = [Interval(1.5, 3.625), Interval(4.25, 6.375)]
        adapter = Recognizer(
            lambda f: "첫 대사\n둘째 줄" if f.pts < 4 else "다음 대사\n두 번째 줄"
        )
        with tempfile.TemporaryDirectory() as td:
            base = Path(td) / "captions"
            report = self.run_fake(sample_frames(57), spans, adapter, base)
            fmt, srt = parse_subtitle_bytes(
                report.srt_path.read_bytes(), source_url="test.srt",
            )
            fmt_ass, ass = parse_subtitle_bytes(
                report.ass_path.read_bytes(), source_url="test.ass",
            )
            self.assertEqual((fmt, fmt_ass), ("srt", "ass"))
            self.assertEqual(len(srt), len(ass))
            self.assertEqual([(c.start, c.end) for c in srt],
                             [(1.5, 3.625), (4.25, 6.375)])
            self.assertEqual([c.text for c in srt],
                             ["첫 대사\n둘째 줄", "다음 대사\n두 번째 줄"])
            self.assertEqual([c.text for c in srt], [c.text for c in ass])
            self.assertEqual(report.recognized_intervals, 2)
            self.assertEqual(report.missed_intervals, 0)
            self.assertEqual(report.ocr_calls, len(adapter.calls))
            self.assertLessEqual(report.maximum_retained_frames, 5)
            self.assertGreater(report.ocr_calls, 2)

    def test_missed_interval_time_windows_are_review_only_and_bounded(self):
        spans = [
            Interval(i + .1, i + .8, "DO NOT PRINT PRIVATE REFERENCE")
            for i in range(12)
        ]
        with tempfile.TemporaryDirectory() as td:
            report = self.run_fake(
                sample_frames(110), spans,
                Recognizer(lambda f: "첫 자막" if f.pts < 1 else ""),
                Path(td) / "first-only",
            )
            self.assertEqual((report.detected_intervals,
                              report.recognized_intervals,
                              report.missed_intervals), (12, 1, 11))
            self.assertEqual(
                report.missed_windows,
                tuple((i + .1, i + .8) for i in range(1, 9)),
            )
            summary = timed_ocr.format_missed_windows(report.missed_windows)
            self.assertEqual(len(summary.split(", ")), 8)
            self.assertIn("00:00:01.100–00:00:01.800", summary)
            self.assertNotIn("00:00:09.", summary)
            self.assertNotIn("PRIVATE", summary)
            self.assertIn("첫 자막", report.srt_path.read_text("utf-8"))

    def test_opt_in_stable_double_caption_without_more_ocr_calls(self):
        class DoubleCaptionOCR:
            def __init__(self):
                self.calls = 0
            def recognize(self, frame):
                self.calls += 1
                text = ("차례대로 여기에 온다" if frame.pts < 1.9
                        else "저녁에는 음악을 듣자")
                return OCRObservation(
                    text, .99, frame.pts,
                    {"detections": [{"cx": .5, "cy": .75}]},
                )
        with tempfile.TemporaryDirectory() as td:
            first, second = DoubleCaptionOCR(), DoubleCaptionOCR()
            before = self.run_fake(
                sample_frames(25), [Interval(1, 3)], first,
                Path(td) / "before",
            )
            after = self.run_fake(
                sample_frames(25), [Interval(1, 3)], second,
                Path(td) / "after", split_stable_text=True,
            )
            _, before_cues = parse_subtitle_bytes(
                before.srt_path.read_bytes(), source_url="before.srt",
            )
            _, after_cues = parse_subtitle_bytes(
                after.srt_path.read_bytes(), source_url="after.srt",
            )
            _, after_ass = parse_subtitle_bytes(
                after.ass_path.read_bytes(), source_url="after.ass",
            )
            self.assertEqual(len(before_cues), 1)
            self.assertEqual([x.text for x in after_cues],
                             ["차례대로 여기에 온다", "저녁에는 음악을 듣자"])
            self.assertEqual([x.text for x in after_ass],
                             [x.text for x in after_cues])
            self.assertEqual((after.recognized_intervals, after.missed_intervals,
                              after.exported_cues, after.split_source_intervals),
                             (1, 0, 2, 1))
            self.assertEqual(before.ocr_calls, after.ocr_calls)
            self.assertEqual(first.calls, second.calls)
            self.assertEqual(before.split_cue_numbers, ())
            self.assertEqual(after.split_cue_numbers, (1,))
            self.assertLess(abs(after_cues[0].end - 1.8125), .002)
            self.assertEqual(after_cues[0].end, after_cues[1].start)

    def test_split_review_number_uses_actual_output_index_after_other_cues(self):
        class MixedOCR:
            def recognize(self, frame):
                if frame.pts < 1.0:
                    text = "먼저 등장하는 대사"
                elif frame.pts < 1.9:
                    text = "차례대로 여기에 온다"
                elif frame.pts < 3.0:
                    text = "저녁에는 음악을 듣자"
                else:
                    text = ""
                return OCRObservation(
                    text, .99, frame.pts,
                    {"detections": [{"cx": .5, "cy": .75}]} if text else None,
                )
        with tempfile.TemporaryDirectory() as td:
            report = self.run_fake(
                sample_frames(40),
                [Interval(.125, .875), Interval(1, 3), Interval(3.5, 4.25)],
                MixedOCR(), Path(td) / "mixed",
                split_stable_text=True,
            )
            self.assertEqual(report.detected_intervals, 3)
            self.assertEqual(report.recognized_intervals, 2)
            self.assertEqual(report.missed_intervals, 1)
            self.assertEqual(report.exported_cues, 3)
            self.assertEqual(report.split_source_intervals, 1)
            self.assertEqual(report.split_cue_numbers, (2,))
            self.assertEqual(report.missed_windows, ((3.5, 4.25),))
            _, actual = parse_subtitle_bytes(
                report.srt_path.read_bytes(), source_url="mixed.srt",
            )
            self.assertEqual(len(actual), 3)
            self.assertEqual(
                [c.text for c in actual],
                ["먼저 등장하는 대사", "차례대로 여기에 온다",
                 "저녁에는 음악을 듣자"],
            )

    def test_stable_split_rejects_glitch_missing_geometry_low_confidence(self):
        span = timed_ocr._Interval(1.0, 3.0)
        times = (1.25, 1.625, 2.0, 2.375, 2.75)
        a, b = "차례대로 여기에 온다", "저녁에는 음악을 듣자"
        def readings(texts, geometry=True, confidence=.99):
            return [(pts, OCRObservation(
                text, confidence, pts,
                {"detections": [{"cx": .5, "cy": .75}]} if geometry else None,
            )) for pts, text in zip(times, texts)]
        self.assertIsNone(timed_ocr._stable_cue_pair(
            readings([a, a, b, a, a]), span,
        ))
        self.assertIsNone(timed_ocr._stable_cue_pair(
            readings([a, a, b, b, b], geometry=False), span,
        ))
        self.assertIsNone(timed_ocr._stable_cue_pair(
            readings([a, a, b, b, b], confidence=.8), span,
        ))
        with self.assertRaises(ValueError):
            timed_ocr.extract_timed_ocr(
                "fixture.mp4", [Interval(1, 3)], Recognizer(),
                Path("unwritten"), split_stable_text="yes",
            )

    def test_short_interval_same_frame_deduplicated(self):
        with tempfile.TemporaryDirectory() as td:
            adapter = Recognizer()
            report = self.run_fake(
                sample_frames(12), [Interval(.24, .27)],
                adapter, Path(td) / "result",
            )
            self.assertEqual(report.ocr_calls, 1)
            self.assertEqual(report.maximum_retained_frames, 1)
            self.assertEqual(adapter.calls, [.25])

    def test_short_interval_between_8fps_frames_recovers_with_bounded_24fps(self):
        """A native 80ms subtitle is invisible on 8fps grid but has real pixels."""
        windows = []
        def source(*args, **kwargs):
            windows.append(kwargs)
            if kwargs["sample_fps"] == 8.0:
                return sample_frames(40)
            self.assertEqual(kwargs["sample_fps"], 24.0)
            seek = kwargs["start_seconds"]
            def short_frames():
                yield Frame(1.041667 - seek, np.zeros((6, 8, 3), np.uint8))
                yield Frame(1.083333 - seek, np.zeros((6, 8, 3), np.uint8))
            return short_frames()

        with tempfile.TemporaryDirectory() as td, mock.patch.object(
            timed_ocr, "FFmpegPTSFrameSource", side_effect=source,
        ) as decoder:
            report = timed_ocr.extract_timed_ocr(
                "fixture.mp4",
                [Interval(1.01, 1.09), Interval(2, 2.75)],
                Recognizer(lambda f: "짧은 원본 대사" if f.pts < 1.5 else "일반 대사"),
                Path(td) / "short-rescued",
            )
            _, cues = parse_subtitle_bytes(
                report.srt_path.read_bytes(), source_url="recovered.srt",
            )
            self.assertEqual(
                [(c.start, c.end, c.text) for c in cues],
                [(1.01, 1.09, "짧은 원본 대사"), (2, 2.75, "일반 대사")],
            )
            self.assertEqual(
                (report.detected_intervals, report.recognized_intervals,
                 report.missed_intervals, report.short_frame_rescue_attempted,
                 report.short_frame_rescue_recognized,
                 report.short_frame_rescue_deferred),
                (2, 2, 0, 1, 1, 0),
            )
            self.assertEqual(decoder.call_count, 2)
            self.assertEqual(report.missed_windows, ())
            self.assertLessEqual(report.maximum_retained_frames, 5)

    def test_one_blank_8fps_frame_gets_short_24fps_check_without_retries_of_success(self):
        """A lone sampled frame can be real but blank during fade-out."""
        def source(*args, **kwargs):
            if kwargs["sample_fps"] == 8.0:
                return sample_frames(40)
            seek = kwargs["start_seconds"]
            # Production FFmpeg reader yields a closeable generator.
            # Do not substitute a list_iterator: contextlib.closing enforces
            # decoder cleanup on every path.
            return (frame for frame in [
                Frame(1.041667 - seek, np.zeros((6, 8, 3), np.uint8)),
                Frame(1.083333 - seek, np.zeros((6, 8, 3), np.uint8)),
            ])

        def transcript(frame):
            if frame.pts < 1.10:
                return "아주 짧은 첫 대사"
            if frame.pts < 2:
                return ""
            return "그다음 보통 길이 대사"

        with tempfile.TemporaryDirectory() as td, mock.patch.object(
            timed_ocr, "FFmpegPTSFrameSource", side_effect=source,
        ) as decoder:
            ocr = Recognizer(transcript)
            report = timed_ocr.extract_timed_ocr(
                "fixture.mp4", [Interval(1.01, 1.20), Interval(2, 2.75)],
                ocr, Path(td) / "blank-one",
            )
            self.assertEqual(decoder.call_count, 2)
            self.assertEqual(
                (report.short_frame_rescue_attempted,
                 report.short_frame_rescue_recognized, report.missed_intervals),
                (1, 1, 0),
            )
            self.assertIn(1.125, ocr.calls)
            _, cues = parse_subtitle_bytes(
                report.srt_path.read_bytes(), source_url="blank-one.srt",
            )
            self.assertEqual(
                [(cue.start, cue.end, cue.text) for cue in cues],
                [(1.01, 1.20, "아주 짧은 첫 대사"),
                 (2, 2.75, "그다음 보통 길이 대사")],
            )

        with tempfile.TemporaryDirectory() as td, mock.patch.object(
            timed_ocr, "FFmpegPTSFrameSource", side_effect=source,
        ) as decoder:
            report = timed_ocr.extract_timed_ocr(
                "fixture.mp4", [Interval(1.01, 1.20)],
                Recognizer(lambda frame: "이미 읽은 정상 대사"),
                Path(td) / "already-recognized",
            )
            self.assertEqual(decoder.call_count, 1)
            self.assertEqual(report.short_frame_rescue_attempted, 0)
            self.assertEqual(report.recognized_intervals, 1)

    def test_actual_ffmpeg_short_caption_pixel_recovers_with_native_timing(self):
        """Actual lossless 30fps video: 8fps misses the timed frame, 24fps sees it.

        The white overlay is constructed separately from OCR output and only
        exists from 1.01 through 1.09s. No native executable, real OCR model,
        network, user video or credential is used.
        """
        import subprocess
        from hardsub_extractor.frame_source import (
            FFmpegPTSFrameSource, NormalizedROI,
        )
        from hardsub_extractor.runtime_paths import resolve_binary

        try:
            ffmpeg = resolve_binary("ffmpeg")
            resolve_binary("ffprobe")
        except (OSError, RuntimeError, FileNotFoundError):
            self.skipTest("FFmpeg unavailable")
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            video = root / "timed-80ms.mkv"
            subprocess.run([
                ffmpeg, "-nostdin", "-hide_banner", "-loglevel", "error",
                "-f", "lavfi", "-i",
                "color=c=black:s=96x64:r=30:d=3",
                "-vf",
                "drawbox=x=18:y=30:w=60:h=12:color=white:t=fill:"
                "enable='gte(t,1.01)*lt(t,1.09)'",
                "-c:v", "ffv1", "-pix_fmt", "yuv420p", str(video),
            ], check=True, timeout=30, stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
            roi = NormalizedROI(0, 0, 1, 1)
            low = list(FFmpegPTSFrameSource(
                str(video), roi=roi, sample_fps=8.0,
            ))
            self.assertFalse(any(1.01 <= f.pts < 1.09 for f in low))

            class WhiteRegionRecognizer:
                def recognize(self, frame):
                    white = int(np.count_nonzero(frame.image[:, :, 0] > 180))
                    return OCRObservation(
                        "짧은 자막" if white >= 100 else "",
                        .99 if white >= 100 else 0,
                        frame.pts,
                    )

            report = timed_ocr.extract_timed_ocr(
                str(video), [Interval(1.01, 1.09)],
                WhiteRegionRecognizer(), root / "result",
                roi=roi,
            )
            self.assertEqual(
                (report.recognized_intervals, report.missed_intervals,
                 report.short_frame_rescue_attempted,
                 report.short_frame_rescue_recognized),
                (1, 0, 1, 1),
            )
            self.assertLessEqual(report.maximum_retained_frames, 5)
            _, cues = parse_subtitle_bytes(
                report.srt_path.read_bytes(), source_url="80ms.srt",
            )
            self.assertEqual(
                [(cue.start, cue.end, cue.text) for cue in cues],
                [(1.01, 1.09, "짧은 자막")],
            )
            self.assertEqual(len(list(root.glob("*.mkv"))), 1)

    def test_short_frame_rescue_batches_nearby_intervals_one_decoder(self):
        """Twenty 80ms intervals share one extra 24fps decoder, not 16."""
        def source(*args, **kwargs):
            if kwargs["sample_fps"] == 8.0:
                return sample_frames(184)
            seek = kwargs["start_seconds"]
            stop = seek + kwargs["max_duration_seconds"]
            def short_frames():
                for i in range(20):
                    absolute = i + .05
                    if seek <= absolute < stop:
                        yield Frame(
                            absolute - seek, np.zeros((6, 8, 3), np.uint8),
                        )
            return short_frames()

        intervals = [Interval(i + .01, i + .09) for i in range(20)]
        intervals.append(Interval(21, 22))
        with tempfile.TemporaryDirectory() as td, mock.patch.object(
            timed_ocr, "FFmpegPTSFrameSource", side_effect=source,
        ) as decoder:
            report = timed_ocr.extract_timed_ocr(
                "fixture.mp4", intervals, Recognizer(),
                Path(td) / "batched",
            )
            self.assertEqual(decoder.call_count, 2)
            self.assertEqual(report.short_frame_rescue_attempted, 20)
            self.assertEqual(report.short_frame_rescue_recognized, 20)
            self.assertEqual(report.short_frame_rescue_deferred, 0)
            self.assertEqual(report.recognized_intervals, 21)
            self.assertEqual(report.missed_intervals, 0)
            self.assertEqual(report.exported_cues, 21)
            self.assertEqual(report.missed_windows, ())
            self.assertLessEqual(report.maximum_retained_frames, 5)

    def test_actual_ffmpeg_twenty_short_caption_intervals_in_one_decoder(self):
        """Read real 30fps FFV1 pixels; no OCR model, network or VSF executable."""
        import subprocess
        from hardsub_extractor.frame_source import NormalizedROI
        from hardsub_extractor.runtime_paths import resolve_binary
        try:
            ffmpeg = resolve_binary("ffmpeg")
            resolve_binary("ffprobe")
        except (OSError, RuntimeError, FileNotFoundError):
            self.skipTest("FFmpeg unavailable")
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            video = root / "twenty-80ms.mkv"
            subprocess.run([
                ffmpeg, "-nostdin", "-hide_banner", "-loglevel", "error",
                "-f", "lavfi", "-i", "color=c=black:s=96x64:r=30:d=22",
                "-vf", "drawbox=x=18:y=30:w=60:h=12:color=white:t=fill:"
                       "enable='lt(mod(t,1),0.09)*gt(mod(t,1),0.01)'",
                "-c:v", "ffv1", "-pix_fmt", "yuv420p", str(video),
            ], check=True, timeout=40, stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)

            class PixelEvidence:
                def recognize(self, frame):
                    count = int(np.count_nonzero(frame.image[:, :, 0] > 180))
                    return OCRObservation(
                        "짧은 대사" if count > 100 else "",
                        0.99 if count > 100 else 0, frame.pts,
                    )
            real_decoder = timed_ocr.FFmpegPTSFrameSource
            with mock.patch.object(
                timed_ocr, "FFmpegPTSFrameSource", wraps=real_decoder,
            ) as decoders:
                report = timed_ocr.extract_timed_ocr(
                    str(video),
                    [Interval(i + .01, i + .09) for i in range(20)],
                    PixelEvidence(), root / "result",
                    roi=NormalizedROI(0, 0, 1, 1),
                )
            self.assertEqual(decoders.call_count, 2)
            self.assertEqual(
                (report.recognized_intervals, report.missed_intervals,
                 report.short_frame_rescue_attempted,
                 report.short_frame_rescue_recognized),
                (20, 0, 20, 20),
            )
            self.assertLessEqual(report.maximum_retained_frames, 5)
            _, cues = parse_subtitle_bytes(
                report.srt_path.read_bytes(), source_url="twenty.srt",
            )
            self.assertEqual(
                [(round(c.start, 3), round(c.end, 3), c.text)
                 for c in cues],
                [(round(i + .01, 3), round(i + .09, 3), "짧은 대사")
                 for i in range(20)],
            )

    def test_batched_short_decode_failure_preserves_earlier_success(self):
        def source(*args, **kwargs):
            if kwargs["sample_fps"] == 8.0:
                return sample_frames(30)
            seek = kwargs["start_seconds"]
            def interrupted():
                yield Frame(.05 - seek, np.zeros((6, 8, 3), np.uint8))
                yield Frame(.125 - seek, np.zeros((6, 8, 3), np.uint8))
                raise RuntimeError("a later local frame cannot be decoded")
            return interrupted()

        with tempfile.TemporaryDirectory() as td, mock.patch.object(
            timed_ocr, "FFmpegPTSFrameSource", side_effect=source,
        ):
            report = timed_ocr.extract_timed_ocr(
                "fixture.mp4", [
                    Interval(.01, .09), Interval(1.01, 1.09),
                    Interval(2, 2.75),
                ], Recognizer(), Path(td) / "partial",
            )
            self.assertEqual(report.short_frame_rescue_attempted, 2)
            self.assertEqual(report.short_frame_rescue_recognized, 1)
            self.assertEqual(report.short_frame_rescue_errors, 1)
            self.assertEqual(report.missed_windows, ((1.01, 1.09),))
            _, cues = parse_subtitle_bytes(
                report.srt_path.read_bytes(), source_url="partial.srt",
            )
            self.assertEqual([c.start for c in cues], [.01, 2.0])
            self.assertLessEqual(report.maximum_retained_frames, 5)

    def test_short_frame_rescue_sixteen_decoder_windows_still_bounded(self):
        """Distant 80ms intervals remain limited to sixteen extra processes."""
        intervals = [
            Interval(i * 40 + .01, i * 40 + .09) for i in range(18)
        ]
        intervals.append(Interval(800.0, 801.0))
        spans = timed_ocr._validate_intervals(intervals)
        normal_windows = len(timed_ocr._decoder_windows(spans))
        def source(*args, **kwargs):
            if kwargs["sample_fps"] == 8.0:
                return sample_frames(
                    math.ceil(kwargs["max_duration_seconds"] * 8),
                )
            seek = kwargs["start_seconds"]
            original_time = seek + kwargs["max_duration_seconds"] - .54
            # Supply a closeable generator rather than a list_iterator.
            return (frame for frame in [Frame(
                original_time - seek, np.zeros((6, 8, 3), np.uint8),
            )])

        with tempfile.TemporaryDirectory() as td, mock.patch.object(
            timed_ocr, "FFmpegPTSFrameSource", side_effect=source,
        ) as decoder:
            report = timed_ocr.extract_timed_ocr(
                "fixture.mp4", intervals, Recognizer(),
                Path(td) / "bounded",
            )
            self.assertEqual(decoder.call_count, normal_windows + 16)
            self.assertEqual(report.short_frame_rescue_attempted, 16)
            self.assertEqual(report.short_frame_rescue_recognized, 16)
            self.assertEqual(report.short_frame_rescue_deferred, 2)
            self.assertEqual(report.recognized_intervals, 17)
            self.assertEqual(report.missed_intervals, 2)
            self.assertEqual(report.exported_cues, 17)
            self.assertEqual(
                report.missed_windows,
                tuple((i * 40 + .01, i * 40 + .09)
                      for i in range(16, 18)),
            )

    def test_short_frame_rescue_failure_preserves_other_original_cues(self):
        def source(*args, **kwargs):
            if kwargs["sample_fps"] == 8.0:
                return sample_frames(40)
            raise RuntimeError("optional tiny reread unavailable")
        with tempfile.TemporaryDirectory() as td, mock.patch.object(
            timed_ocr, "FFmpegPTSFrameSource", side_effect=source,
        ):
            report = timed_ocr.extract_timed_ocr(
                "fixture.mp4", [Interval(1.01, 1.09), Interval(2, 2.75)],
                Recognizer(), Path(td) / "optional-failed",
            )
            self.assertEqual(report.recognized_intervals, 1)
            self.assertEqual(report.missed_intervals, 1)
            self.assertEqual(report.short_frame_rescue_errors, 1)
            self.assertEqual(report.missed_windows, ((1.01, 1.09),))
            self.assertTrue(report.srt_path.is_file())

    def test_missed_window_clock_rounds_over_minute_boundary(self):
        self.assertEqual(
            timed_ocr.format_missed_windows(((3599.9996, 3600.042),)),
            "01:00:00.000–01:00:00.042",
        )

    def test_long_episode_many_intervals_without_retaining_frames(self):
        spans = [Interval(float(i), float(i) + .75)
                 for i in range(0, 500, 2)]
        with tempfile.TemporaryDirectory() as td:
            report = self.run_fake(
                sample_frames(4008), spans,
                Recognizer(), Path(td) / "long",
            )
            self.assertEqual(report.detected_intervals, 250)
            self.assertEqual(report.recognized_intervals, 250)
            self.assertEqual(report.missed_intervals, 0)
            self.assertLessEqual(report.maximum_retained_frames, 5)
            self.assertLessEqual(report.ocr_calls, 250 * 5)


    def test_sparse_long_gap_seeks_and_restores_absolute_pts(self):
        """Only two short decode windows, with original cue times and text."""
        calls = []
        adapter = Recognizer(
            lambda frame: "BEFORE" if frame.pts < 20 else "AFTER"
        )

        def source(*args, **kwargs):
            calls.append(kwargs)
            # Model actual FFmpeg seek-local showinfo timestamps.
            return sample_frames(int(kwargs["max_duration_seconds"] * 8))

        with tempfile.TemporaryDirectory() as td, mock.patch.object(
            timed_ocr, "FFmpegPTSFrameSource", side_effect=source,
        ):
            report = timed_ocr.extract_timed_ocr(
                "fixture.mp4", [Interval(1, 2), Interval(41, 42)],
                adapter, Path(td) / "sparse",
            )
            _, cues = parse_subtitle_bytes(
                report.srt_path.read_bytes(), source_url="sparse.srt",
            )
            self.assertEqual([(c.start, c.end, c.text) for c in cues],
                             [(1, 2, "BEFORE"), (41, 42, "AFTER")])
        self.assertEqual([x["start_seconds"] for x in calls], [0.0, 40.0])
        self.assertLess(sum(x["max_duration_seconds"] for x in calls), 10)
        self.assertEqual(report.recognized_intervals, 2)
        self.assertEqual(report.missed_intervals, 0)
        self.assertTrue(any(41 <= x < 42 for x in adapter.calls))
        self.assertLessEqual(report.maximum_retained_frames, 5)


    def test_fractional_window_seek_remains_on_output_fps_grid(self):
        """Previously a 40.333s seek shifted every 8fps selected pixel."""
        spans = timed_ocr._validate_intervals([
            Interval(1.333, 2.208), Interval(41.333, 42.208),
        ])
        for fps in (8.0, 10.0, 7.7, 7.8):
            with self.subTest(fps=fps):
                windows = timed_ocr._decoder_windows(spans, sample_fps=fps)
                self.assertEqual(len(windows), 2)
                seek, duration = windows[1]
                self.assertLessEqual(seek, 41.333 - 1.0)
                self.assertTrue(
                    math.isclose(seek * fps, round(seek * fps), abs_tol=1e-8)
                )
                self.assertGreater(duration, 2)
                self.assertLess(duration, 5)
        self.assertEqual(
            timed_ocr._decoder_windows(spans, sample_fps=8.0)[1][0], 40.25,
        )
        for fps in (7.7, 7.8):
            self.assertEqual(
                timed_ocr._decoder_windows(spans, sample_fps=fps)[1][0], 40.0,
            )

    def test_fractional_unstable_grid_uses_one_nonoverlapping_decoder(self):
        spans = timed_ocr._validate_intervals([
            Interval(1.0, 2.0), Interval(81.0, 82.0),
            Interval(161.0, 162.0),
        ])
        windows = timed_ocr._decoder_windows(spans, sample_fps=7.77)
        self.assertEqual(windows, ((0.0, 163.0),))

    def test_real_ffmpeg_fractional_seek_pixels_match_full_decode(self):
        """An actual 30fps video detects source-frame ties missed by pure math.

        This is the transport acceptance test: no OCR model, native VSF,
        network connection or user media is needed.
        """
        import subprocess
        from hardsub_extractor.frame_source import FFmpegPTSFrameSource
        from hardsub_extractor.runtime_paths import resolve_binary

        try:
            ffmpeg = resolve_binary("ffmpeg")
            resolve_binary("ffprobe")
        except (OSError, RuntimeError, FileNotFoundError):
            self.skipTest("FFmpeg binaries unavailable")
        with tempfile.TemporaryDirectory() as td:
            movie = Path(td) / "synthetic.mkv"
            subprocess.run([
                ffmpeg, "-nostdin", "-hide_banner", "-loglevel", "error",
                "-f", "lavfi", "-i",
                "testsrc2=size=64x64:rate=30:duration=45",
                "-c:v", "ffv1", "-pix_fmt", "yuv420p", str(movie),
            ], check=True, timeout=45, stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE)
            spans = timed_ocr._validate_intervals([
                Interval(1.0, 2.0), Interval(41.333, 42.208),
            ])
            seek, duration = timed_ocr._decoder_windows(
                spans, sample_fps=7.8,
            )[1]
            self.assertEqual(seek, 40.0)
            full = list(FFmpegPTSFrameSource(
                str(movie), sample_fps=7.8,
            ))
            partial = list(FFmpegPTSFrameSource(
                str(movie), sample_fps=7.8,
                start_seconds=seek, max_duration_seconds=duration,
            ))
            self.assertGreater(len(partial), 12)
            for frame in partial:
                full_index = round((seek + frame.pts) * 7.8)
                self.assertAlmostEqual(
                    seek + frame.pts, full[full_index].pts, delta=2e-5,
                )
                np.testing.assert_array_equal(
                    frame.image, full[full_index].image,
                )

    def test_sparse_second_window_cancellation_closes_decoder_no_output(self):
        """Cancel during seeked decoding, never leak a process or export."""
        finals = []
        count = [0]

        class Stop(Exception):
            pass

        def check():
            count[0] += 1
            if count[0] > 36:
                raise Stop()

        def source(*args, **kwargs):
            return sample_frames(
                int(kwargs["max_duration_seconds"] * 8), finalizer=finals,
            )

        with tempfile.TemporaryDirectory() as td, mock.patch.object(
            timed_ocr, "FFmpegPTSFrameSource", side_effect=source,
        ) as decoder:
            base = Path(td) / "cancelled"
            with self.assertRaises(Stop):
                timed_ocr.extract_timed_ocr(
                    "fixture.mp4", [Interval(1, 2), Interval(41, 42)],
                    Recognizer(), base, check_cancelled=check,
                )
            self.assertEqual(decoder.call_count, 2)
            self.assertEqual(len(finals), 2)
            self.assertFalse(base.with_suffix(".srt").exists())
            self.assertFalse(base.with_suffix(".ass").exists())

    def test_65_distant_intervals_keep_64_small_decoders_not_whole_movie(self):
        # Previously this 43-minute sparse subtitle source reverted to one
        # 2,562-second decoder, wasting nearly all decoded frames. Preserve
        # the original 64-process cap and avoid processing most blank film.
        spans = [Interval(float(i * 40), float(i * 40 + 1))
                 for i in range(65)]
        windows = timed_ocr._decoder_windows(
            timed_ocr._validate_intervals(spans),
        )
        self.assertEqual(len(windows), 64)
        self.assertEqual(windows[0][0], 0.0)
        self.assertLess(sum(duration for _, duration in windows), 250)
        self.assertLess(
            sum(duration for _, duration in windows)
            / (spans[-1].end - spans[0].start), 0.10,
        )
        for start, duration in windows:
            self.assertGreater(duration, 0)
            self.assertLessEqual(start, spans[-1].end)
            self.assertTrue(math.isclose(
                start * 8.0, round(start * 8.0), abs_tol=1e-8,
            ))

    def test_overflow_prefers_shortest_gaps_and_maintains_64_window_cap(self):
        # All gaps are >30 seconds, but only two are relatively cheap.
        # Stable minimum-extra-decode merging must choose these two rather
        # than reverting to 1 decoder or joining arbitrary distant scenes.
        starts = [0.0]
        for index in range(1, 66):
            starts.append(starts[-1] + (32.0 if index in (12, 45) else 48.0))
        spans = [Interval(start, start + 1) for start in starts]
        windows = timed_ocr._decoder_windows(
            timed_ocr._validate_intervals(spans),
        )
        self.assertEqual(len(windows), 64)
        self.assertEqual(len([duration for _, duration in windows
                              if duration > 30.0]), 2)
        self.assertLess(max(duration for _, duration in windows), 36)
        self.assertLess(sum(duration for _, duration in windows), 300)
        for index in (11, 44):
            self.assertTrue(any(
                seek <= starts[index] and
                seek + duration >= starts[index + 1] + 1
                for seek, duration in windows
            ))

    def test_65_sparse_intervals_live_frame_loop_uses_exact_64_decoders(self):
        spans = [Interval(float(i * 40), float(i * 40 + 1))
                 for i in range(65)]
        sampled = []
        adapter = Recognizer(lambda frame: f"OCR AT {int(frame.pts // 40)}")
        def source(*args, **kwargs):
            sampled.append(kwargs)
            return sample_frames(math.ceil(kwargs["max_duration_seconds"] * 8))
        with tempfile.TemporaryDirectory() as td, mock.patch.object(
            timed_ocr, "FFmpegPTSFrameSource", side_effect=source,
        ):
            base = Path(td) / "sparse"
            report = timed_ocr.extract_timed_ocr(
                "fixture.mp4", spans, adapter, base,
            )
            _, cues = parse_subtitle_bytes(
                report.srt_path.read_bytes(), source_url="sparse.srt",
            )
            self.assertEqual(len(cues), 65)
            self.assertEqual([(c.start, c.end) for c in cues],
                             [(s.start, s.end) for s in spans])
            self.assertEqual(report.recognized_intervals, 65)
            self.assertEqual(report.missed_intervals, 0)
            self.assertFalse(any("reference" in c.text.lower() for c in cues))
        self.assertEqual(len(sampled), 64)
        self.assertLess(sum(call["max_duration_seconds"]
                            for call in sampled), 250)
        self.assertLessEqual(report.maximum_retained_frames, 5)
        self.assertLessEqual(report.ocr_calls, 65 * 5)

    def test_invalid_intervals_fail_before_decoder(self):
        invalid = [
            [],
            [Interval(1, 1)],
            [Interval(float("nan"), 2)],
            [Interval(3, 2)],
            [Interval(2, 3), Interval(2.9, 5)],
            [Interval(2, 3), Interval(1, 2)],
            [Interval(0, 21601)],
            [Interval(True, 5)],
            [Interval(float(i), float(i) + .5)
             for i in range(3001)],
        ]
        with tempfile.TemporaryDirectory() as td, mock.patch.object(
            timed_ocr, "FFmpegPTSFrameSource",
        ) as decoder:
            for items in invalid:
                with self.subTest(size=len(items)):
                    with self.assertRaises(ValueError):
                        timed_ocr.extract_timed_ocr(
                            "fixture.mp4", items, Recognizer(),
                            Path(td) / "out",
                        )
            decoder.assert_not_called()
            self.assertFalse((Path(td) / "out.srt").exists())

    def test_invalid_sampling_fails_without_decoder(self):
        with mock.patch.object(timed_ocr, "FFmpegPTSFrameSource") as decoder:
            for fps in (0, 13, float("nan"), float("inf"), True):
                with self.subTest(fps=fps), self.assertRaises(ValueError):
                    timed_ocr.extract_timed_ocr(
                        "fixture.mp4", [Interval(0, 1)], Recognizer(),
                        "never-created", sample_fps=fps,
                    )
            decoder.assert_not_called()

    def test_existing_result_preserved_and_nothing_decoded(self):
        with tempfile.TemporaryDirectory() as td:
            base = Path(td) / "out"
            existing = base.with_suffix(".srt")
            existing.write_bytes(b"original user text")
            with mock.patch.object(timed_ocr, "FFmpegPTSFrameSource") as decoder:
                with self.assertRaises(FileExistsError):
                    timed_ocr.extract_timed_ocr(
                        "fixture.mp4", [Interval(0, 1)], Recognizer(), base,
                    )
                decoder.assert_not_called()
            self.assertEqual(existing.read_bytes(), b"original user text")
            self.assertFalse(base.with_suffix(".ass").exists())

    def test_invalid_video_pts_does_not_publish_and_closes_decoder(self):
        final = []
        def frames():
            try:
                yield Frame(.5, np.zeros((4, 4, 3), np.uint8))
                yield Frame(.25, np.zeros((4, 4, 3), np.uint8))
            finally:
                final.append(True)
        with tempfile.TemporaryDirectory() as td:
            with mock.patch.object(timed_ocr, "FFmpegPTSFrameSource",
                                   return_value=frames()):
                with self.assertRaisesRegex(RuntimeError, "PTS"):
                    timed_ocr.extract_timed_ocr(
                        "fixture.mp4", [Interval(0, 1)], Recognizer(),
                        Path(td) / "out",
                    )
            self.assertTrue(final)
            self.assertFalse((Path(td) / "out.srt").exists())

    def test_cancelled_while_ocr_closes_decoder_and_preserves_outputs(self):
        class Cancel(Exception):
            pass
        final = []
        checks = [0]
        def stop():
            checks[0] += 1
            if checks[0] > 19:
                raise Cancel
        with tempfile.TemporaryDirectory() as td:
            with mock.patch.object(
                timed_ocr, "FFmpegPTSFrameSource",
                return_value=sample_frames(50, finalizer=final),
            ):
                with self.assertRaises(Cancel):
                    timed_ocr.extract_timed_ocr(
                        "fixture.mp4", [Interval(0, 2), Interval(3, 4)],
                        Recognizer(), Path(td) / "out", check_cancelled=stop,
                    )
            self.assertTrue(final)
            self.assertFalse((Path(td) / "out.srt").exists())

    def test_empty_ocr_result_is_failure_not_empty_successful_export(self):
        final = []
        with tempfile.TemporaryDirectory() as td:
            with self.assertRaisesRegex(RuntimeError, "no recognized"):
                self.run_fake(
                    sample_frames(22, finalizer=final),
                    [Interval(1, 2)], Recognizer(lambda f: ""),
                    Path(td) / "out",
                )
            self.assertTrue(final)
            self.assertFalse((Path(td) / "out.srt").exists())

    def test_last_interval_end_causes_decoder_to_close_early(self):
        final = []
        def frames():
            try:
                for n in range(1000000):
                    yield Frame(n / 8, np.zeros((2, 3, 3), np.uint8))
            finally:
                final.append(True)
        adapter = Recognizer()
        with tempfile.TemporaryDirectory() as td:
            report = self.run_fake(
                frames(), [Interval(.3, 1.2)], adapter,
                Path(td) / "out",
            )
            self.assertTrue(final)
            self.assertEqual(report.recognized_intervals, 1)
            self.assertLess(max(adapter.calls), 1.2)

    def test_all_text_must_come_from_model_not_interval_reference(self):
        with tempfile.TemporaryDirectory() as td:
            adapter = Recognizer(lambda f: "OCR ONLY")
            report = self.run_fake(
                sample_frames(24),
                [Interval(.5, 1.5, "FORBIDDEN TRUTH WORDS")],
                adapter, Path(td) / "out",
            )
            text = report.srt_path.read_text("utf-8")
            self.assertIn("OCR ONLY", text)
            self.assertNotIn("FORBIDDEN", text)


if __name__ == "__main__":
    unittest.main()
