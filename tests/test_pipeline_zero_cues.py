import tempfile
import unittest
from pathlib import Path
from unittest import mock

from hardsub_extractor.pipeline import extract


class PipelineZeroCueTests(unittest.TestCase):
    def test_zero_cues_fail_before_writing_srt_or_ass(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            out = root / "result" / "subtitles"

            with mock.patch(
                "hardsub_extractor.pipeline.FFmpegPTSFrameSource",
                return_value="frames",
            ), mock.patch(
                "hardsub_extractor.pipeline.segment_frames",
                return_value=[],
            ), mock.patch(
                "hardsub_extractor.pipeline.recognize_segments",
                return_value=[],
            ), mock.patch(
                "hardsub_extractor.pipeline.merge_cues",
                return_value=[],
            ), mock.patch(
                "hardsub_extractor.pipeline.write_srt"
            ) as write_srt, mock.patch(
                "hardsub_extractor.pipeline.write_ass"
            ) as write_ass:
                with self.assertRaisesRegex(RuntimeError, "인식된 자막이 없습니다"):
                    extract("source.mkv", object(), out)

            write_srt.assert_not_called()
            write_ass.assert_not_called()
            self.assertFalse(out.with_suffix(".srt").exists())
            self.assertFalse(out.with_suffix(".ass").exists())
            self.assertFalse(out.parent.exists())


if __name__ == "__main__":
    unittest.main()
