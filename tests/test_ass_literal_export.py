"""Real ASS/SRT literal subtitle export regression; no OCR, accounts or media."""
from __future__ import annotations

from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

from hardsub_extractor.models import Cue
from hardsub_extractor.subtitle_formats import parse_ass
from hardsub_extractor.writers import (
    _ass_plain_text, write_ass, write_srt, write_subtitle_pair_exclusive,
)


class ASSLiteralExportTests(unittest.TestCase):
    def test_round_trip_literal_glyphs_including_escaped_ass_tags(self):
        examples = [
            "첫 줄\n둘째 줄",
            "안녕 {세계}",
            r"HELLO {\an8} END",
            r"HELLO {\pos(50,20)} END",
            r"HELLO \N world",
            r"HELLO \h world",
            r"C:\folder\name",
            r"\{escaped}",
            "}{",
            "a\\\u2060b",
            "こんにちは{世界}",
            "你好{有趣}",
        ]
        with tempfile.TemporaryDirectory() as td:
            output = Path(td) / "test.ass"
            for phrase in examples:
                with self.subTest(phrase=phrase):
                    write_ass([Cue(.5, 2., phrase)], output)
                    actual = parse_ass(output.read_text(encoding="utf-8"))
                    self.assertEqual(len(actual), 1)
                    self.assertEqual(actual[0].text, phrase)
                    self.assertAlmostEqual(actual[0].start, .5)
                    self.assertAlmostEqual(actual[0].end, 2.)

    def test_genuine_ass_override_tags_are_still_not_displayed_as_text(self):
        source = """[Script Info]
ScriptType: v4.00+
[Events]
Format: Layer,Start,End,Style,Name,MarginL,MarginR,MarginV,Effect,Text
Dialogue: 0,0:00:01.00,0:00:02.00,Default,,0,0,0,,{\\an8}안녕\\N세상{\\r}
"""
        self.assertEqual([x.text for x in parse_ass(source)], ["안녕\n세상"])

    def test_writer_makes_only_intended_newlines_not_literal_slash_n(self):
        line = r"\N and \h are literal subtitles"
        rendered = _ass_plain_text(line)
        self.assertIn("\\\u2060N", rendered)
        self.assertIn("\\\u2060h", rendered)
        self.assertNotIn(r"\N", rendered)
        self.assertIn(r"\N", _ass_plain_text("첫째\n둘째"))

    def test_plain_cjk_and_multiline_bytes_still_use_legacy_content(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "plain.ass"
            write_ass([Cue(1., 2., "한글\n한 줄")], path)
            data = path.read_text("utf-8")
            self.assertIn("한글\\N한 줄", data)
            self.assertEqual([x.text for x in parse_ass(data)], ["한글\n한 줄"])

    def test_exclusive_pair_retains_literal_content_and_original_srt(self):
        words = r"안녕 {\an8} 안녕 \N"
        with tempfile.TemporaryDirectory() as td:
            srt, ass = write_subtitle_pair_exclusive(
                [Cue(.5, 2., words)], Path(td) / "literal"
            )
            self.assertIn(words, srt.read_text(encoding="utf-8"))
            self.assertEqual(parse_ass(ass.read_text("utf-8"))[0].text, words)
            self.assertIn(r"\{{}", ass.read_text("utf-8"))
            with self.assertRaises(FileExistsError):
                write_subtitle_pair_exclusive(
                    [Cue(.5, 2., words)], Path(td) / "literal"
                )
            self.assertIn(words, srt.read_text(encoding="utf-8"))


@unittest.skipUnless(shutil.which("ffmpeg"), "FFmpeg unavailable")
class ASSLiteralRenderedPixelTests(unittest.TestCase):
    def _render(self, folder: Path, phrase: str, stem: str) -> tuple[int, int, int]:
        ass = folder / (stem + ".ass")
        write_ass([Cue(0., 1., phrase)], ass)
        result = subprocess.run(
            [
                shutil.which("ffmpeg"), "-nostdin", "-hide_banner",
                "-loglevel", "error", "-f", "lavfi", "-i",
                "color=c=black:s=640x360:d=1:r=1",
                "-vf", "ass=" + ass.name,
                "-frames:v", "1", "-pix_fmt", "gray",
                "-f", "rawvideo", "pipe:1",
            ],
            cwd=folder, capture_output=True, timeout=20,
        )
        if result.returncode:
            if b"No such filter: 'ass'" in result.stderr:
                self.skipTest("FFmpeg built without libass")
            self.fail("ASS render failure: " +
                      result.stderr.decode("utf-8", "replace")[-600:])
        frame = result.stdout
        self.assertEqual(len(frame), 640 * 360)
        positions = [i for i, pixel in enumerate(frame) if pixel > 45]
        self.assertTrue(positions, "rendered subtitle must be visible")
        return len(positions), min(positions) // 640, max(positions) // 640

    def test_literal_position_tag_is_visible_text_not_active_top_alignment(self):
        with tempfile.TemporaryDirectory() as td:
            count, y0, y1 = self._render(
                Path(td), r"HELLO {\an8} END", "dangerous_tag"
            )
            self.assertGreater(count, 250)
            self.assertGreater(y0, 250, "user text must stay at bottom")
            self.assertGreater(y1, y0)

    def test_literal_curly_text_is_not_discarded_by_renderer(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td)
            full, _, _ = self._render(path, "HELLO {world} END", "braced")
            stripped, _, _ = self._render(path, "HELLO  END", "stripped")
            self.assertGreater(full, stripped * 1.15)


if __name__ == "__main__":
    unittest.main()
