import unicodedata
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from hardsub_extractor.subtitle_formats import (
    SubtitleFormatError,
    normalize_subtitle_bytes,
    parse_ass,
    parse_srt,
    parse_vtt,
)


class SubtitleFormatTests(unittest.TestCase):
    def test_srt_cp949_to_utf8_outputs(self):
        src="1\n00:00:01,000 --> 00:00:02,500\n안녕하세요\n"
        with TemporaryDirectory() as td:
            r=normalize_subtitle_bytes(src.encode("cp949"),Path(td)/"out",source_url="x.srt")
            self.assertEqual(r.source_format,"srt")
            self.assertEqual(r.cue_count,1)
            self.assertIn("안녕하세요",r.srt_path.read_text(encoding="utf-8"))
            self.assertIn("안녕하세요",r.ass_path.read_text(encoding="utf-8"))

    def test_require_korean_accepts_short_dense_korean_track(self):
        src="1\n00:00:01,000 --> 00:00:02,500\n안녕하세요\n"
        with TemporaryDirectory() as td:
            base=Path(td)/"nested"/"out"
            r=normalize_subtitle_bytes(
                src.encode("utf-8"),
                base,
                source_url="x.srt",
                require_korean=True,
            )
            self.assertEqual(r.cue_count,1)
            self.assertTrue(base.with_suffix(".srt").is_file())
            self.assertTrue(base.with_suffix(".ass").is_file())

    def test_require_korean_accepts_nfd_hangul_without_rewriting_output(self):
        decomposed = unicodedata.normalize("NFD", "안녕하세요")
        self.assertNotEqual(decomposed, "안녕하세요")
        src=f"1\n00:00:01,000 --> 00:00:02,500\n{decomposed}\n"
        with TemporaryDirectory() as td:
            base=Path(td)/"out"
            r=normalize_subtitle_bytes(
                src.encode("utf-8"),
                base,
                source_url="x.srt",
                require_korean=True,
            )
            written=r.srt_path.read_text(encoding="utf-8")
            self.assertIn(decomposed,written)
            self.assertNotIn("안녕하세요",written)

    def test_require_korean_rejects_english_before_any_output_is_created(self):
        src="1\n00:00:01,000 --> 00:00:02,500\nEnglish subtitle only\n"
        with TemporaryDirectory() as td:
            base=Path(td)/"nested"/"out"
            with self.assertRaisesRegex(SubtitleFormatError,"Korean"):
                normalize_subtitle_bytes(
                    src.encode("utf-8"),
                    base,
                    source_url="x.srt",
                    require_korean=True,
                )
            self.assertFalse(base.parent.exists())
            self.assertFalse(base.with_suffix(".srt").exists())
            self.assertFalse(base.with_suffix(".ass").exists())

    def test_require_korean_rejects_incidental_korean_word_in_english_track(self):
        src=(
            "1\n00:00:01,000 --> 00:00:02,500\n"
            "This is a long English subtitle line mentioning 한국 once but otherwise English.\n"
            "\n2\n00:00:03,000 --> 00:00:04,000\n"
            "Another English sentence continues without Korean dialogue.\n"
        )
        with TemporaryDirectory() as td:
            base=Path(td)/"out"
            with self.assertRaisesRegex(SubtitleFormatError,"Korean"):
                normalize_subtitle_bytes(
                    src.encode("utf-8"),
                    base,
                    source_url="x.srt",
                    require_korean=True,
                )
            self.assertFalse(base.with_suffix(".srt").exists())
            self.assertFalse(base.with_suffix(".ass").exists())

    def test_require_korean_rejects_korean_note_inside_long_foreign_track(self):
        src=(
            "1\n00:00:01,000 --> 00:00:02,500\n"
            "This remains an English subtitle with a translator note 한국어한국어한국어 only.\n"
            "\n2\n00:00:03,000 --> 00:00:04,000\n"
            "The rest of this subtitle continues in English for a long enough sample.\n"
            "\n3\n00:00:05,000 --> 00:00:06,000\n"
            "There is still no Korean dialogue in the actual track content here.\n"
        )
        with TemporaryDirectory() as td:
            base=Path(td)/"out"
            with self.assertRaisesRegex(SubtitleFormatError,"Korean"):
                normalize_subtitle_bytes(
                    src.encode("utf-8"),
                    base,
                    source_url="x.srt",
                    require_korean=True,
                )
            self.assertFalse(base.with_suffix(".srt").exists())
            self.assertFalse(base.with_suffix(".ass").exists())

    def test_invalid_clock_range_is_rejected_before_writing(self):
        src="1\n00:99:01,000 --> 00:99:02,000\n안녕하세요\n"
        with TemporaryDirectory() as td:
            base=Path(td)/"out"
            with self.assertRaisesRegex(SubtitleFormatError,"invalid timestamp"):
                normalize_subtitle_bytes(src.encode("utf-8"),base,source_url="x.srt")
            self.assertFalse(base.with_suffix(".srt").exists())
            self.assertFalse(base.with_suffix(".ass").exists())

    def test_non_positive_cue_duration_is_rejected_before_writing(self):
        src="1\n00:00:03,000 --> 00:00:02,000\n안녕하세요\n"
        with TemporaryDirectory() as td:
            base=Path(td)/"out"
            with self.assertRaisesRegex(SubtitleFormatError,"invalid cue timeline"):
                normalize_subtitle_bytes(src.encode("utf-8"),base,source_url="x.srt")
            self.assertFalse(base.with_suffix(".srt").exists())
            self.assertFalse(base.with_suffix(".ass").exists())

    def test_vtt_to_srt(self):
        src="WEBVTT\n\n00:01.000 --> 00:02.000\n첫 줄\n\n00:03.000 --> 00:04.500\n둘째 줄\n"
        cues=parse_vtt(src)
        self.assertEqual(len(cues),2)
        self.assertEqual(cues[0].text,"첫 줄")

    def test_vtt_preserves_rendered_lines_and_literal_escaped_tags(self):
        source = (
            "WEBVTT\n\n"
            "00:00:01.250 --> 00:00:02.875 align:start\n"
            "<v Speaker><c.bg_transparent>첫 줄<br>둘째 &amp; 셋째</c></v>\n"
            "마지막 &lt;br&gt; 는 문자 그대로\n"
            "\n"
            "00:00:04.000 --> 00:00:05.000\n"
            "<ruby>漢<rt>かん</rt></ruby> <00:00:04.500><i>字</i> &quot;인용&quot;\n"
        )
        cues = parse_vtt(source)
        self.assertEqual(len(cues), 2)
        self.assertEqual(cues[0].start, 1.25)
        self.assertEqual(cues[0].end, 2.875)
        self.assertEqual(
            cues[0].text,
            "첫 줄\n둘째 & 셋째\n마지막 <br> 는 문자 그대로",
        )
        self.assertEqual(cues[1].text, '漢 字 "인용"')

    def test_vtt_metadata_blocks_never_become_cues(self):
        source = (
            "WEBVTT\n\n"
            "NOTE diagnostic payload\n"
            "00:00:00.000 --> 00:00:01.000\n"
            "이 문장은 주석입니다\n\n"
            "STYLE\n"
            "::cue { color: lime }\n"
            "00:00:01.000 --> 00:00:02.000\n"
            "not a cue\n\n"
            "REGION\n"
            "id:caption-area\n"
            "00:00:02.000 --> 00:00:03.000\n"
            "also not a cue\n\n"
            "cue-id\n"
            "00:00:03.000 --> 00:00:04.500 line:90%\n"
            "실제 자막만\n"
        )
        cues = parse_vtt(source)
        self.assertEqual(len(cues), 1)
        self.assertEqual(cues[0].text, "실제 자막만")
        self.assertEqual((cues[0].start, cues[0].end), (3.0, 4.5))

    def test_vtt_note_only_is_rejected_without_false_cues(self):
        with self.assertRaisesRegex(SubtitleFormatError, "no VTT cues"):
            parse_vtt(
                "WEBVTT\n\nNOTE contains timestamp\n"
                "00:00:00.000 --> 00:00:01.000\n"
                "이건 자막이 아님\n"
            )

    def test_vtt_multiline_conversion_keeps_srt_ass_and_timestamps(self):
        source = (
            "WEBVTT\n\n"
            "00:00:01.250 --> 00:00:02.875\n"
            "<c>첫 줄<br />둘째 &amp; 셋째</c>\n"
        ).encode("utf-8")
        with TemporaryDirectory() as td:
            result = normalize_subtitle_bytes(
                source, Path(td) / "converted", source_url="x.vtt",
                require_korean=True,
            )
            srt = result.srt_path.read_text(encoding="utf-8")
            ass = result.ass_path.read_text(encoding="utf-8")
            self.assertEqual(result.cue_count, 1)
            self.assertIn("00:00:01,250 --> 00:00:02,875", srt)
            self.assertIn("첫 줄\n둘째 & 셋째", srt)
            self.assertIn(r"첫 줄\N둘째 & 셋째", ass)

    def test_ass_dialogue_to_plain_cues(self):
        src="""[Script Info]
Title: x
[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
Dialogue: 0,0:00:01.00,0:00:02.00,Default,,0,0,0,,{\\b1}한 줄\\N두 줄
"""
        cues=parse_ass(src)
        self.assertEqual(len(cues),1)
        self.assertEqual(cues[0].text,"한 줄\n두 줄")


if __name__=="__main__":
    unittest.main()
