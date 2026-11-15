"""Offline actual-child FFprobe cancellation/cleanup tests. No model or network."""
from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from unittest import mock

from hardsub_extractor.extraction_progress import ExtractionCancelled
from hardsub_extractor.frame_source import (
    FFmpegPTSFrameSource, NormalizedROI, VideoInfo, probe_video,
)


class CancelableProbeTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="hse-probe-test-")
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def sleeper(self, delay=2., *, failure=False):
        path = self.root / "probe.py"
        path.write_text(
            "import sys,time,os,json\n"
            "time.sleep("+str(delay)+")\n"
            + ('print("bad-probe",file=sys.stderr)\nsys.exit(13)\n' if failure else
                'print(json.dumps({"streams":[{"width":8,"height":8,"avg_frame_rate":"8/1"}],"format":{"duration":"5"}}))\n'),
            encoding="utf8",
        )
        return [sys.executable, "-u", str(path)]

    def test_cancel_before_probe_has_zero_subprocesses(self):
        with mock.patch("hardsub_extractor.frame_source.subprocess.Popen") as launch:
            with self.assertRaises(ExtractionCancelled):
                probe_video("unused", cancel_requested=lambda: True)
            launch.assert_not_called()

    def test_cancel_during_real_sleeping_probe_reaps_child_before_decoder(self):
        stop = threading.Event()
        trigger = threading.Thread(target=lambda: (time.sleep(.12), stop.set()))
        trigger.start()
        # Spy on the actual OS subprocess, not a fake-Popen object.
        spawned = []
        real_spawn = subprocess.Popen
        def record_child(*args, **kwargs):
            proc = real_spawn(*args, **kwargs)
            spawned.append(proc)
            return proc
        try:
            with (
                mock.patch("hardsub_extractor.frame_source._build_ffprobe_command", return_value=self.sleeper()),
                mock.patch("hardsub_extractor.frame_source._build_ffmpeg_decode_command", side_effect=AssertionError("no decoder after cancel")) as decoder,
                mock.patch("hardsub_extractor.frame_source.subprocess.Popen", side_effect=record_child),
            ):
                start=time.monotonic()
                with self.assertRaises(ExtractionCancelled):
                    list(FFmpegPTSFrameSource(
                        "dummy-local", roi=NormalizedROI(0.,0.,1.,1.),
                        probe_timeout_seconds=4., cancel_requested=stop.is_set,
                    ))
                self.assertLess(time.monotonic()-start,1.5)
                decoder.assert_not_called()
        finally:
            trigger.join(timeout=3)
        self.assertEqual(len(spawned), 1)
        self.assertIsNotNone(spawned[0].poll())
        self.assertTrue(spawned[0].stdout.closed)
        self.assertTrue(spawned[0].stderr.closed)

    def test_explicit_probe_timeout_reaps_sleeper(self):
        spawned = []
        real_spawn = subprocess.Popen
        def record_child(*args, **kwargs):
            proc = real_spawn(*args, **kwargs)
            spawned.append(proc)
            return proc
        with (
            mock.patch("hardsub_extractor.frame_source._build_ffprobe_command", return_value=self.sleeper()),
            mock.patch("hardsub_extractor.frame_source.subprocess.Popen", side_effect=record_child),
        ):
            start=time.monotonic()
            with self.assertRaises(subprocess.TimeoutExpired):
                probe_video("test", timeout_seconds=.20, cancel_requested=lambda: False)
            self.assertLess(time.monotonic()-start,1.5)
        self.assertEqual(len(spawned), 1)
        self.assertIsNotNone(spawned[0].poll())
        self.assertTrue(spawned[0].stdout.closed)
        self.assertTrue(spawned[0].stderr.closed)

    def test_cancelable_normal_probe_matches_original_metadata(self):
        with mock.patch("hardsub_extractor.frame_source._build_ffprobe_command", return_value=self.sleeper(delay=0.01)):
            got=probe_video("test",timeout_seconds=4,cancel_requested=lambda: False)
        self.assertEqual(got,VideoInfo(8,8,8.,5.))

    def test_opt_in_nonzero_ffprobe_still_raises_called_process_error(self):
        with mock.patch("hardsub_extractor.frame_source._build_ffprobe_command", return_value=self.sleeper(delay=.01,failure=True)):
            with self.assertRaises(subprocess.CalledProcessError) as err:
                probe_video("test",timeout_seconds=3,cancel_requested=lambda: False)
            self.assertEqual(err.exception.returncode,13)

    def test_legacy_non_cancel_probe_still_invokes_subprocess_run(self):
        result=subprocess.CompletedProcess(["ffprobe"],0,stdout='{"streams":[{"width":8,"height":8,"avg_frame_rate":"8/1"}],"format":{"duration":"5"}}')
        with (
            mock.patch("hardsub_extractor.frame_source.subprocess.run", return_value=result) as old,
            mock.patch("hardsub_extractor.frame_source.subprocess.Popen") as new,
        ):
            result=probe_video("dummy",timeout_seconds=2)
        self.assertEqual(result,VideoInfo(8,8,8.,5.))
        new.assert_not_called()
        self.assertEqual(old.call_args.kwargs["timeout"],2)

    def test_invalid_owner_callback_is_rejected_before_launch(self):
        with mock.patch("hardsub_extractor.frame_source.subprocess.Popen") as launch:
            with self.assertRaises(TypeError):
                probe_video("unused",cancel_requested=True)
            launch.assert_not_called()

if __name__=='__main__':unittest.main()
