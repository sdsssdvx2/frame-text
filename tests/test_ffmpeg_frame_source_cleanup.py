"""No real video, network, subprocess or OCR is used in this lifecycle test."""
from __future__ import annotations

import io
import subprocess
import threading
import time
import unittest
from unittest import mock

import numpy as np

from hardsub_extractor.extraction_progress import ExtractionCancelled, ExtractionMonitor
from hardsub_extractor.frame_source import FFmpegPTSFrameSource, NormalizedROI, VideoInfo

DATA = np.arange(8 * 8 * 3, dtype=np.uint8).tobytes()


class FakeFFmpeg:
    def __init__(self, raw=DATA, pts=(1.0,), code=0, timeout_once=False):
        self.stdout = io.BytesIO(raw)
        self.stderr = io.BytesIO(b"".join(
            f"[Parsed_showinfo_0] n: {i} pts_time:{t:.3f}\n".encode()
            for i, t in enumerate(pts)
        ))
        self.returncode = None
        self.code = code
        self.timeout_once = timeout_once
        self.terminated = 0
        self.killed = 0
        self.wait_times = []

    def poll(self):
        return self.returncode

    def terminate(self):
        self.terminated += 1
        self.returncode = -15

    def kill(self):
        self.killed += 1
        self.returncode = -9

    def wait(self, timeout=None):
        self.wait_times.append(timeout)
        if self.timeout_once:
            self.timeout_once = False
            raise subprocess.TimeoutExpired("ffmpeg", timeout)
        if self.returncode is None:
            self.returncode = self.code
        return self.returncode


class FFmpegCleanupTests(unittest.TestCase):
    def source(self, decoder):
        # Patch on the live module to test the exact production generator.
        return (
            mock.patch(
                "hardsub_extractor.frame_source.probe_video",
                return_value=VideoInfo(8, 8, 30.0, 5.0),
            ),
            mock.patch(
                "hardsub_extractor.frame_source.resolve_binary",
                return_value="fake-ffmpeg",
            ),
            mock.patch(
                "hardsub_extractor.frame_source.subprocess.Popen",
                return_value=decoder,
            ),
        )

    def frames(self):
        return FFmpegPTSFrameSource(
            "test-only.mp4", roi=NormalizedROI(0.0, 0.0, 1.0, 1.0)
        )

    def test_successful_eof_preserves_frames_and_checks_decoder_exit(self):
        fake = FakeFFmpeg(raw=DATA * 2, pts=(1.0, 1.4))
        a, b, c = self.source(fake)
        with a, b, c:
            frames = list(self.frames())
        self.assertEqual([f.pts for f in frames], [1.0, 1.4])
        self.assertEqual(frames[0].image.tobytes(), DATA)
        self.assertEqual(fake.terminated, 0)
        self.assertEqual(fake.killed, 0)
        self.assertEqual(fake.wait_times, [10])

    def test_opt_in_preprobed_info_avoids_second_metadata_probe(self):
        fake = FakeFFmpeg(raw=DATA, pts=(1.0,))
        with (
            mock.patch(
                "hardsub_extractor.frame_source.probe_video",
                side_effect=AssertionError("metadata probed again"),
            ) as probe,
            mock.patch("hardsub_extractor.frame_source.resolve_binary",
                       return_value="fake-ffmpeg"),
            mock.patch("hardsub_extractor.frame_source.subprocess.Popen",
                       return_value=fake),
        ):
            frames = list(FFmpegPTSFrameSource(
                "only-a-test.mkv",
                roi=NormalizedROI(0.0, 0.0, 1.0, 1.0),
                preprobed_info=VideoInfo(8, 8, 30.0, 5.0),
            ))
        probe.assert_not_called()
        self.assertEqual(len(frames), 1)
        self.assertEqual(frames[0].image.tobytes(), DATA)
        self.assertEqual(fake.wait_times, [10])

    def test_preview_watchdog_times_out_blocked_stdout_and_reaps_process(self):
        class BlockedPipe:
            def __init__(self):
                self.release = threading.Event()
                self.closed = False

            def read(self, count):
                self.release.wait(3)
                return b""

            def close(self):
                self.closed = True
                self.release.set()

        class StalledFFmpeg(FakeFFmpeg):
            def __init__(self):
                super().__init__()
                self.stdout = BlockedPipe()

            def terminate(self):
                super().terminate()
                self.stdout.release.set()

            def kill(self):
                super().kill()
                self.stdout.release.set()

        fake = StalledFFmpeg()
        a, b, c = self.source(fake)
        before = time.monotonic()
        with a, b, c:
            with self.assertRaisesRegex(RuntimeError, "preview decoder exceeded time limit"):
                list(FFmpegPTSFrameSource(
                    "test-only.mkv",
                    preprobed_info=VideoInfo(8, 8, 30, 10),
                    roi=NormalizedROI(0, 0, 1, 1),
                    decode_timeout_seconds=0.15,
                ))
        self.assertLess(time.monotonic() - before, 2.5)
        self.assertEqual(fake.terminated, 1)
        self.assertEqual(fake.killed, 0)
        self.assertTrue(fake.stdout.closed)

    def test_opt_in_preview_watchdog_keeps_fast_normal_eof_successful(self):
        fake = FakeFFmpeg(raw=DATA, pts=(1.0,))
        a, b, c = self.source(fake)
        with a, b, c:
            result = list(FFmpegPTSFrameSource(
                "test-only.mkv",
                preprobed_info=VideoInfo(8, 8, 30, 10),
                roi=NormalizedROI(0, 0, 1, 1),
                decode_timeout_seconds=1.0,
                cancel_requested=lambda: False,
            ))
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].image.tobytes(), DATA)
        self.assertEqual(fake.terminated, 0)
        self.assertEqual(fake.wait_times, [10])

    def test_preview_owner_cancel_interrupts_blocked_stdout(self):
        class BlockedPipe:
            def __init__(self):
                self.release = threading.Event()

            def read(self, count):
                self.release.wait(3)
                return b""

            def close(self):
                self.release.set()

        class StalledFFmpeg(FakeFFmpeg):
            def __init__(self):
                super().__init__()
                self.stdout = BlockedPipe()

            def terminate(self):
                super().terminate()
                self.stdout.release.set()

        fake = StalledFFmpeg()
        stop = threading.Event()
        a, b, c = self.source(fake)
        signal = threading.Thread(target=lambda: (time.sleep(.07), stop.set()))
        signal.start()
        try:
            with a, b, c:
                with self.assertRaises(ExtractionCancelled):
                    list(FFmpegPTSFrameSource(
                        "test-only.mkv",
                        preprobed_info=VideoInfo(8, 8, 30, 10),
                        roi=NormalizedROI(0, 0, 1, 1),
                        decode_timeout_seconds=2,
                        cancel_requested=stop.is_set,
                    ))
        finally:
            signal.join(timeout=2)
        self.assertEqual(fake.terminated, 1)
        self.assertEqual(fake.killed, 0)

    def test_opt_in_metadata_probe_timeout_reaches_actual_probe(self):
        fake = FakeFFmpeg(raw=DATA, pts=(1.0,))
        a, b, c = self.source(fake)
        with a as probe, b, c:
            frames = list(FFmpegPTSFrameSource(
                "test-only.mp4",
                roi=NormalizedROI(0, 0, 1, 1),
                probe_timeout_seconds=17.0,
            ))
        probe.assert_called_once_with(
            "test-only.mp4", extra_input_args=[], timeout_seconds=17.0,
        )
        self.assertEqual(len(frames), 1)
        self.assertEqual(frames[0].image.tobytes(), DATA)
        self.assertEqual(fake.terminated, 0)

    def test_preview_rejects_invalid_opt_in_watchdog_inputs(self):
        for bad in (0, -1, float("nan"), float("inf"), 121):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                FFmpegPTSFrameSource("test-only.mp4", decode_timeout_seconds=bad)
        for bad in (0, -1, float("nan"), float("inf"), 121):
            with self.subTest(probe_timeout=bad), self.assertRaises(ValueError):
                FFmpegPTSFrameSource("test-only.mp4", probe_timeout_seconds=bad)
        with self.assertRaises(TypeError):
            FFmpegPTSFrameSource("test-only.mp4", cancel_requested=True)

    def test_natural_eof_with_nonzero_code_is_not_silently_accepted(self):
        fake = FakeFFmpeg(code=7)
        a, b, c = self.source(fake)
        with a, b, c:
            with self.assertRaisesRegex(RuntimeError, r"FFmpeg decode failed \(7\)"):
                list(self.frames())
        self.assertEqual(fake.terminated, 0)

    def test_early_generator_close_terminates_ffmpeg_without_decode_error(self):
        fake = FakeFFmpeg(raw=DATA * 3, pts=(1.0, 1.5, 2.0), code=9)
        a, b, c = self.source(fake)
        with a, b, c:
            gen = iter(self.frames())
            next(gen)
            gen.close()
        self.assertEqual(fake.terminated, 1)
        self.assertEqual(fake.killed, 0)
        self.assertTrue(fake.stdout.closed)

    def test_cancelled_monitor_keeps_cancel_reason_and_cleans_decoder(self):
        fake = FakeFFmpeg(raw=DATA * 3, pts=(1.0, 1.5, 2.0))
        stop = threading.Event()
        a, b, c = self.source(fake)
        with a, b, c:
            wrapped = ExtractionMonitor(is_cancelled=stop.is_set).frames(self.frames())
            first = next(wrapped)
            self.assertEqual(first.pts, 1.0)
            stop.set()
            with self.assertRaises(ExtractionCancelled):
                next(wrapped)
        self.assertEqual(fake.terminated, 1)
        self.assertEqual(fake.killed, 0)

    def test_cancelled_ffmpeg_escalates_to_kill_only_after_wait_timeout(self):
        fake = FakeFFmpeg(raw=DATA * 2, pts=(1.0, 1.5), timeout_once=True)
        a, b, c = self.source(fake)
        with a, b, c:
            gen = iter(self.frames())
            next(gen)
            gen.close()
        self.assertEqual(fake.terminated, 1)
        self.assertEqual(fake.killed, 1)
        self.assertEqual(fake.wait_times, [3, 5])

    def test_never_exiting_decoder_is_not_reported_as_clean_cancel(self):
        class NeverExits(FakeFFmpeg):
            def wait(self, timeout=None):
                self.wait_times.append(timeout)
                raise subprocess.TimeoutExpired("ffmpeg", timeout)

        fake = NeverExits(raw=DATA * 2, pts=(1.0, 1.5))
        a, b, c = self.source(fake)
        with a, b, c:
            gen = iter(self.frames())
            next(gen)
            with self.assertRaisesRegex(
                RuntimeError, "cleanup could not confirm process exit"
            ):
                gen.close()
        self.assertEqual(fake.terminated, 1)
        self.assertEqual(fake.killed, 1)
        self.assertEqual(fake.wait_times, [3, 5])

    def test_truncated_raw_frame_stays_a_real_error(self):
        fake = FakeFFmpeg(raw=DATA + b"abc", pts=(1.0, 1.5))
        a, b, c = self.source(fake)
        with a, b, c:
            with self.assertRaisesRegex(RuntimeError, "FFmpeg raw frame truncated"):
                list(self.frames())
        self.assertEqual(fake.terminated, 1)


if __name__ == "__main__":
    unittest.main()
