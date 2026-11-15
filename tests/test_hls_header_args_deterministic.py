"""Network-free regressions for FFmpeg/ffprobe input-header placement."""

from __future__ import annotations

import unittest
from unittest import mock

from hardsub_extractor.frame_source import (
    _build_ffmpeg_decode_command,
    _build_ffprobe_command,
)


SOURCE = "https://media.example/stream.m3u8"
UA = "HSE-Test-UA/1.0"
REFERER = "https://player.example/embed"
INPUT_ARGS = ["-user_agent", UA, "-referer", REFERER]


class HLSHeaderCommandTests(unittest.TestCase):
    def _assert_input_headers_before(self, cmd: list[str], boundary: int) -> None:
        ua_index = cmd.index("-user_agent")
        referer_index = cmd.index("-referer")
        self.assertLess(ua_index, boundary)
        self.assertLess(referer_index, boundary)
        self.assertEqual(cmd[ua_index + 1], UA)
        self.assertEqual(cmd[referer_index + 1], REFERER)

    @mock.patch("hardsub_extractor.frame_source.resolve_binary")
    def test_ffprobe_places_user_agent_and_referer_before_input_url(self, resolve):
        resolve.return_value = "ffprobe-test.exe"
        cmd = _build_ffprobe_command(SOURCE, extra_input_args=INPUT_ARGS)

        self.assertEqual(cmd[0], "ffprobe-test.exe")
        source_index = cmd.index(SOURCE)
        self._assert_input_headers_before(cmd, source_index)
        self.assertEqual(source_index, len(cmd) - 1)

    @mock.patch("hardsub_extractor.frame_source.resolve_binary")
    def test_ffmpeg_places_headers_and_seek_before_input_duration_after(self, resolve):
        resolve.return_value = "ffmpeg-test.exe"
        cmd = _build_ffmpeg_decode_command(
            SOURCE,
            extra_input_args=INPUT_ARGS,
            start_seconds=12.5,
            max_duration_seconds=20.0,
            video_filter="crop=640:240:0:300,fps=2.0,showinfo",
        )

        self.assertEqual(cmd[0], "ffmpeg-test.exe")
        input_flag = cmd.index("-i")
        source_index = input_flag + 1
        self.assertEqual(cmd[source_index], SOURCE)
        self._assert_input_headers_before(cmd, input_flag)
        self.assertLess(cmd.index("-ss"), input_flag)
        self.assertEqual(cmd[cmd.index("-ss") + 1], "12.5")
        self.assertGreater(cmd.index("-t"), source_index)
        self.assertEqual(cmd[cmd.index("-t") + 1], "20.0")


if __name__ == "__main__":
    unittest.main()
