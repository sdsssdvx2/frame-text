"""No-network native temporal detector bridge contract, research only."""
from __future__ import annotations

import hashlib
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from hardsub_extractor import native_vsf_timing as bridge


class NativeVsfTimingTests(unittest.TestCase):
    def test_valid_native_srt_timings_never_return_native_text(self):
        raw = (
            b"1\r\n00:00:01,500 --> 00:00:03,625\r\n"
            b"[sub_duration]\r\n\r\n"
            b"2\r\n00:00:04,250 --> 00:00:06,375\r\n"
            b"UNTRUSTED NON-OCR BODY\r\n"
        )
        spans = bridge.parse_native_timing_srt(raw)
        self.assertEqual([(x.start, x.end) for x in spans],
                         [(1.5, 3.625), (4.25, 6.375)])
        self.assertFalse(hasattr(spans[0], "text"))

    def test_empty_timing_report_is_valid_no_subtitle_control(self):
        self.assertEqual(bridge.parse_native_timing_srt(b""), ())

    def test_rejects_malformed_overlapping_oversized_and_nonlinear_intervals(self):
        cases = (
            b"1\ninvalid\n[sub_duration]",
            b"1\n00:00:05,000 --> 00:00:04,000\nx",
            b"1\n00:00:01,000 --> 00:00:03,000\nx\n\n"
            b"2\n00:00:02,000 --> 00:00:04,000\ny",
            b"1\n00:00:01,000 --> 00:00:02,000\n",
            b"1\n00:00:01,000 --> 00:00:02,000\nx\n\n"
            b"3\n00:00:03,000 --> 00:00:04,000\ny",
            b"1\n00:00:01,000 --> 00:00:02,000\nx\n\n"
            b"2\n00:00:00,000 --> 00:00:03,000\ny",
            b"x" * (bridge._MAX_TIMING_BYTES + 1),
        )
        for raw in cases:
            with self.subTest(raw=raw[:20]), self.assertRaises(bridge.NativeTimingError):
                bridge.parse_native_timing_srt(raw)

    def test_native_roi_coordinate_mapping_preserves_default_and_accepts_top(self):
        from hardsub_extractor.frame_source import NormalizedROI
        self.assertEqual(
            bridge._vsf_roi_edges(None), ("0.44", "0", "0", "1"),
        )
        self.assertEqual(
            bridge._vsf_roi_edges(NormalizedROI()),
            ("0.44", "0", "0", "1"),
        )
        # HSE is top-left based; native VSF measures y from image bottom.
        self.assertEqual(
            bridge._vsf_roi_edges(NormalizedROI(0.15, 0.12, 0.70, 0.28)),
            ("0.880000", "0.600000", "0.150000", "0.850000"),
        )
        for roi in (
            NormalizedROI(-0.01, 0.20, 0.5, 0.3),
            NormalizedROI(0.30, 0.90, 0.4, 0.3),
            NormalizedROI(0.2, 0.2, 0.0, 0.3),
            NormalizedROI(float("nan"), 0.2, 0.4, 0.3),
            NormalizedROI(0.2, 0.2, True, 0.3),
        ):
            with self.subTest(roi=roi), self.assertRaises(ValueError):
                bridge._vsf_roi_edges(roi)

    def test_rejects_missing_pin_before_createprocess(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            src = root / "a.mp4"
            exe = root / "VideoSubFinderCli.exe"
            src.write_bytes(b"fake video")
            exe.write_bytes(b"official-binary-fixture")
            with mock.patch.object(bridge, "external_popen") as popen:
                with self.assertRaises(ValueError):
                    bridge.detect_native_vsf_timings(
                        src, exe, pinned_executable_sha256="",
                    )
                with self.assertRaises(bridge.NativeTimingError):
                    bridge.detect_native_vsf_timings(
                        src, exe, pinned_executable_sha256="0"*64,
                    )
                popen.assert_not_called()

    def test_native_spawn_uses_isolated_paths_no_clear_and_disposes_temporary(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            src = root / "a.mp4"
            exe = root / "VideoSubFinderCli.exe"
            src.write_bytes(b"source unchanged")
            exe.write_bytes(b"trusted pinned executable")
            known = hashlib.sha256(exe.read_bytes()).hexdigest()
            seen = []
            class Process:
                returncode = 0
                def poll(self): return self.returncode
                def wait(self, timeout=None): return self.returncode
            def spawn(command, **kwargs):
                self.assertIn("-r", command)
                self.assertIn("-dsi", command)
                self.assertNotIn("-c", command)
                self.assertNotIn("--clear_dirs", command)
                self.assertEqual(kwargs["cwd"], str(root.resolve()))
                self.assertEqual(kwargs["stdin"], bridge.subprocess.DEVNULL)
                out = Path(command[command.index("-ces")+1])
                self.assertNotEqual(out.parent, src.parent)
                out.write_text(
                    "1\n00:00:01,000 --> 00:00:03,000\n[sub_duration]\n",
                    encoding="utf-8",
                )
                seen.append(out)
                return Process()
            with mock.patch.object(bridge, "external_popen", side_effect=spawn):
                result = bridge.detect_native_vsf_timings(
                    src, exe, pinned_executable_sha256=known,
                )
            self.assertEqual(len(result.intervals), 1)
            self.assertTrue(result.detection_seconds >= 0)
            self.assertEqual(src.read_bytes(), b"source unchanged")
            self.assertFalse(seen[0].exists())

    def test_cancellation_kills_native_child_and_cannot_publish(self):
        class Stop(Exception): pass
        class Process:
            returncode = None
            killed = False
            waited = False
            def poll(self): return self.returncode
            def kill(self):
                self.killed = True
                self.returncode = -1
            def wait(self, timeout=None): self.waited = True
        child = Process()
        with mock.patch.object(bridge, "external_popen", return_value=child):
            with self.assertRaises(Stop):
                bridge._spawn_wait(
                    ["fake.exe"], Path("."), 30.0,
                    lambda: (_ for _ in ()).throw(Stop()),
                )
        self.assertTrue(child.killed)
        self.assertTrue(child.waited)

    def test_executable_symlink_and_bad_deadline_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            exe = root / "VSF.exe"
            exe.write_bytes(b"native exe")
            fake = root / "symlink.exe"
            src = root / "video.mp4"
            src.write_bytes(b"video")
            try:
                fake.symlink_to(exe)
            except OSError:
                pass
            else:
                with self.assertRaises(bridge.NativeTimingError):
                    bridge.detect_native_vsf_timings(
                        src, fake,
                        pinned_executable_sha256=hashlib.sha256(exe.read_bytes()).hexdigest(),
                    )
            for seconds in (0, True, -2, float("nan"), float("inf"), 3601):
                with self.subTest(seconds=seconds), self.assertRaises(ValueError):
                    bridge.detect_native_vsf_timings(
                        src, exe, pinned_executable_sha256="0"*64,
                        max_seconds=seconds,
                    )



    def test_no_native_intervals_never_loads_model_or_writes(self):
        with tempfile.TemporaryDirectory() as td:
            base = Path(td) / "result"
            no_cues = bridge.NativeTimingResult((), 0.25)
            with mock.patch.object(
                bridge, "detect_native_vsf_timings", return_value=no_cues,
            ), mock.patch(
                "hardsub_extractor.timed_ocr.extract_timed_ocr",
            ) as timed:
                with self.assertRaisesRegex(bridge.NativeTimingError, "no hardcoded"):
                    bridge.extract_with_native_vsf(
                        "fixture.mp4", base, executable="vsf.exe",
                        pinned_executable_sha256="a"*64,
                        ocr_factory=lambda: self.fail("model loaded for no-caption input"),
                    )
            timed.assert_not_called()
            self.assertFalse(base.with_suffix(".srt").exists())

    def test_preexisting_user_srt_blocks_native_and_model(self):
        with tempfile.TemporaryDirectory() as td:
            base = Path(td) / "result"
            base.with_suffix(".srt").write_bytes(b"USER ORIGINAL SUBTITLE")
            with mock.patch.object(bridge, "detect_native_vsf_timings") as det:
                with self.assertRaises(FileExistsError):
                    bridge.extract_with_native_vsf(
                        "fixture.mp4", base, executable="vsf.exe",
                        pinned_executable_sha256="a"*64,
                        ocr_factory=lambda: self.fail("unexpected OCR"),
                    )
            det.assert_not_called()
            self.assertEqual(base.with_suffix(".srt").read_bytes(), b"USER ORIGINAL SUBTITLE")

    def test_native_ocr_bridges_only_source_timings_to_one_pass(self):
        with tempfile.TemporaryDirectory() as td:
            intervals = (bridge.NativeTiming(1.5, 3.625),)
            native = bridge.NativeTimingResult(intervals, 1.25)
            fake = mock.Mock(
                detected_intervals=1, recognized_intervals=1,
                missed_intervals=0, ocr_calls=5,
            )
            engine = object()
            from hardsub_extractor.frame_source import NormalizedROI
            selected_roi = NormalizedROI(0.15, 0.12, 0.70, 0.28)
            with mock.patch.object(bridge, "detect_native_vsf_timings",
                                   return_value=native) as native_call, mock.patch(
                "hardsub_extractor.timed_ocr.extract_timed_ocr",
                return_value=fake,
            ) as ocr_call:
                result = bridge.extract_with_native_vsf(
                    "fixture.mp4", Path(td) / "out",
                    executable="trusted.exe",
                    pinned_executable_sha256="f"*64,
                    ocr_factory=lambda: engine, roi=selected_roi,
                    split_stable_text=True,
                )
            self.assertIs(result.report, fake)
            self.assertIs(result.timing, native)
            self.assertEqual(native_call.call_count, 1)
            self.assertIs(native_call.call_args.kwargs["roi"], selected_roi)
            args = ocr_call.call_args
            self.assertIs(args.kwargs["roi"], selected_roi)
            self.assertEqual(args.args[1], intervals)
            self.assertIs(args.args[2], engine)
            self.assertEqual(args.kwargs["sample_fps"], 8.0)
            self.assertTrue(args.kwargs["split_stable_text"])


if __name__ == "__main__":
    unittest.main()
