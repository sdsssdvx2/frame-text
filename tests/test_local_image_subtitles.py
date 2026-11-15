"""Bounded offline PGS/VobSub image-track adapter; no real media or downloads.

A verified target document is always SRT/ASS on the original timebase,
with distinct bitmap-OCR provenance (never labelled source text).
"""
from __future__ import annotations
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from hardsub_extractor import local_image_subtitles as image_adapter
from hardsub_extractor.matroska_track_headers import (
    MatroskaImageTrack, read_image_track_headers,
)
from hardsub_extractor.subtitle_formats import parse_subtitle_bytes

SRT = (
    "1\n00:00:01,000 --> 00:00:02,500\nHello world\n\n"
    "2\n00:00:03,000 --> 00:00:04,500\nSubtitle frame\n"
).encode("utf-8")


def track(language="eng", number=2, codec="hdmv_pgs_subtitle"):
    return MatroskaImageTrack(
        track_number=number, codec=codec, language=language,
        default=True, forced=False,
    )


class BitmapTrackAdapterTests(unittest.TestCase):
    def root(self, folder):
        base = Path(folder)
        source = base / "episode.01.mkv"
        source.write_bytes(b"not modified")
        return source, base / "outputs" / "episode_01-12345678"

    def mock_runtime(self, raw=SRT, *, language="eng", track_number=2):
        def convert(args, scratch, env, cancelled):
            self.assertEqual(args[1], str(self.video))
            self.assertIn("--track-number:2", args)
            self.assertEqual(args.count("--json"), 1)
            self.assertTrue(any(x.startswith("--output-folder:") for x in args))
            self.assertTrue(any(x.startswith("--ocr-engine:") for x in args))
            (scratch / "image-original.srt").write_bytes(raw)
            return b'{"success":true,"totalFiles":1,"files":[],"errors":[]}'

        self.video = self.current_source
        return (
            mock.patch.object(
                image_adapter, "_converter_executable",
                return_value=Path("/fake/pinned-seconv.exe"),
            ),
            mock.patch.object(
                image_adapter, "_ocr_backend",
                return_value=(["--ocr-engine:nocr", "--ocr-db:Latin.nocr"], {}),
            ),
            mock.patch.object(
                image_adapter, "read_image_track_headers",
                return_value=[track(language, track_number)],
            ),
            mock.patch.object(image_adapter, "_invoke", side_effect=convert),
        )

    def test_bounded_mkv_ebml_reads_nonconsecutive_real_track_ids_and_language(self):
        def element(eid, payload):
            size = len(payload)
            if size < 127:
                vint = bytes([0x80 | size])
            elif size < 16383:
                vint = (0x4000 | size).to_bytes(2, "big")
            else:
                raise ValueError("test fixture oversize")
            return eid.to_bytes((eid.bit_length() + 7) // 8, "big") + vint + payload

        def entry(number, codec, language, forced=False):
            payload = b"".join((
                element(0xD7, bytes([number])),
                element(0x83, b"\x11"),
                element(0x86, codec.encode("ascii")),
                element(0x22B59C, language.encode("ascii")),
                element(0x55AA, b"\x01" if forced else b"\x00"),
            ))
            return element(0xAE, payload)

        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "tiny.mkv"
            tracks = element(0x1654AE6B, b"".join((
                entry(2, "S_HDMV/PGS", "eng"),
                entry(5, "S_VOBSUB", "jpn", forced=True),
            )))
            # Actual Matroska Segment can be unknown-sized. A fixed-size
            # Segment spanning metadata also must work.
            root = element(0x1A45DFA3, b"") + element(0x18538067, tracks)
            path.write_bytes(root)
            parsed = read_image_track_headers(path)
            self.assertEqual([t.track_number for t in parsed], [2, 5])
            self.assertEqual([t.codec for t in parsed],
                             ["hdmv_pgs_subtitle", "dvd_subtitle"])
            self.assertEqual([t.language for t in parsed], ["eng", "jpn"])
            self.assertEqual([t.forced for t in parsed], [False, True])
            # This must NOT derive 1/2 from FFprobe stream.index.
            self.assertNotEqual(parsed[1].track_number, 2)

    def test_ebml_rejects_truncated_track_element_without_fabricated_identity(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "truncated.mkv"
            path.write_bytes(b"\x1a\x45\xdf\xa3\x80\x18\x53\x80\x67\xff"
                             b"\x16\x54\xae\x6b\xac\xae\x85\xD7\x81")
            with self.assertRaises(ValueError):
                read_image_track_headers(path)

    def test_source_track_ocr_real_writer_srt_ass_and_distinct_provenance(self):
        with tempfile.TemporaryDirectory() as td:
            src, base = self.root(td)
            self.current_source = src
            p1,p2,p3,p4 = self.mock_runtime()
            with p1,p2,p3,p4:
                result = image_adapter.acquire_local_embedded_image(src, base, language="en")
            self.assertEqual(result.mode, "local-embedded-image-ocr")
            self.assertEqual(result.cue_count, 2)
            self.assertEqual(result.track_number, 2)
            self.assertEqual(src.read_bytes(), b"not modified")
            srt_fmt,srt_cues = parse_subtitle_bytes(
                result.srt_path.read_bytes(),source_url="subtitle.srt")
            ass_fmt,ass_cues = parse_subtitle_bytes(
                result.ass_path.read_bytes(),source_url="subtitle.ass")
            self.assertEqual((srt_fmt,ass_fmt),("srt","ass"))
            self.assertEqual([(x.start,x.end,x.text) for x in srt_cues],
                             [(x.start,x.end,x.text) for x in ass_cues])
            self.assertEqual(result.extracted_track_path.read_bytes(), SRT)
            meta=json.loads(result.provenance_path.read_text("utf-8"))
            self.assertEqual(meta["mode"],"local-embedded-image-ocr")
            self.assertTrue(meta["ocr_applied"])
            self.assertFalse(meta["original_text_verified"])
            self.assertEqual((meta["image_track_number"],meta["source_codec"]),(2,"hdmv_pgs_subtitle"))
            self.assertNotIn("stream_index", meta)
            self.assertEqual(meta["source_stream_export"],"seconv_bitmap_OCR_not_original_bitmap_bytes")

    def test_existing_result_blocks_converter_before_any_change(self):
        with tempfile.TemporaryDirectory() as td:
            source, base = self.root(td)
            base.parent.mkdir()
            old=base.with_suffix(".srt")
            old.write_bytes(b"DO NOT TOUCH")
            with mock.patch.object(image_adapter,"_converter_executable") as converter:
                with self.assertRaises(FileExistsError):
                    image_adapter.acquire_local_embedded_image(source,base,language="en")
            converter.assert_not_called()
            self.assertEqual(old.read_bytes(),b"DO NOT TOUCH")

    def test_converter_absent_is_soft_fallback_without_media_decode(self):
        with tempfile.TemporaryDirectory() as td:
            source,base=self.root(td)
            with (
                mock.patch.object(image_adapter,"_converter_executable",return_value=None),
                mock.patch.object(image_adapter,"read_image_track_headers") as probe,
            ):
                self.assertIsNone(image_adapter.acquire_local_embedded_image(source,base,language="ko"))
            probe.assert_not_called()
            self.assertFalse(base.with_suffix(".srt").exists())

    def test_unknown_language_and_non_mkv_never_claim_image_track(self):
        with tempfile.TemporaryDirectory() as td:
            src,base=self.root(td)
            self.current_source=src
            p1,p2,p3,p4=self.mock_runtime(language="und")
            with p1,p2,p3,p4:
                self.assertIsNone(image_adapter.acquire_local_embedded_image(src,base,language="en"))
            self.assertFalse(base.with_suffix(".srt").exists())
            webm=src.with_suffix(".webm")
            webm.write_bytes(b"no img track implementation")
            with p1,p2,p3,p4:
                self.assertIsNone(image_adapter.acquire_local_embedded_image(webm,base,language="en"))

    def test_missing_track_id_is_not_guessed_from_stream_index(self):
        with tempfile.TemporaryDirectory() as td:
            src,base=self.root(td)
            self.current_source=src
            p1,p2,p3,p4=self.mock_runtime(track_number=None)
            with p1,p2,p3,p4:
                self.assertIsNone(image_adapter.acquire_local_embedded_image(src,base,language="en"))
            self.assertFalse(base.with_suffix(".srt").exists())

    def test_converter_empty_or_invalid_text_never_published(self):
        for raw in (b"", b"not a subtitle", b"1\nbroken time\nHello"):
            with self.subTest(sample=raw),tempfile.TemporaryDirectory() as td:
                src,base=self.root(td)
                self.current_source=src
                p1,p2,p3,p4=self.mock_runtime(raw=raw)
                with p1,p2,p3,p4,self.assertRaises(image_adapter.LocalImageSubtitleError):
                    image_adapter.acquire_local_embedded_image(src,base,language="en")
                self.assertFalse(base.with_suffix(".srt").exists())
                self.assertFalse(base.with_suffix(".ass").exists())

    def test_converter_unsuccessful_receipt_fails_closed(self):
        with tempfile.TemporaryDirectory() as td:
            src,base=self.root(td)
            self.current_source=src
            p1,p2,p3,p4=self.mock_runtime()
            with p1,p2,p3, mock.patch.object(image_adapter,"_invoke",return_value=b'{"success":false}'):
                with self.assertRaises(image_adapter.LocalImageSubtitleError):
                    image_adapter.acquire_local_embedded_image(src,base,language="en")
            self.assertFalse(base.with_suffix(".srt").exists())

    def test_ko_requires_korean_original_ocr_evidence(self):
        with tempfile.TemporaryDirectory() as td:
            src,base=self.root(td)
            self.current_source=src
            p1,p2,p3,p4=self.mock_runtime(language="kor")
            with p1,p2,p3,p4,self.assertRaises(image_adapter.LocalImageSubtitleError):
                image_adapter.acquire_local_embedded_image(src,base,language="ko")
            self.assertFalse(base.with_suffix(".srt").exists())

    def test_extracted_image_raw_orphan_blocks_all_three_sources(self):
        with tempfile.TemporaryDirectory() as td:
            src,base=self.root(td)
            base.parent.mkdir()
            raw=base.with_name(base.name+".embedded.image-ocr.srt")
            raw.write_bytes(b"old partial result")
            with self.assertRaises(FileExistsError):
                image_adapter.acquire_local_embedded_image(src,base,language="en")
            self.assertEqual(raw.read_bytes(),b"old partial result")

    def test_backend_declines_ko_if_only_english_no_cr_model_exists(self):
        with mock.patch.dict("os.environ",{
            "HSE_TESSERACT":"missing.exe","HSE_NOCR_DB":"missing.nocr",
        }):
            self.assertIsNone(image_adapter._ocr_backend("ko"))


    def test_converter_uses_frozen_safe_external_spawn(self):
        # A frozen PyInstaller process may have overridden SetDllDirectory.
        # The shared external launcher resets it only for CreateProcess.
        # This guards seconv as well as the already-isolated yt-dlp path.
        with tempfile.TemporaryDirectory() as td:
            scratch = Path(td)
            called = []

            class CompletedChild:
                returncode = 0

                def poll(self):
                    return self.returncode

                def wait(self):
                    return self.returncode

            def spawn(args, **kwargs):
                called.append((args, kwargs))
                kwargs["stdout"].write(b'{"success":true}')
                return CompletedChild()

            with mock.patch.object(
                image_adapter, "external_popen", side_effect=spawn,
            ):
                receipt = image_adapter._invoke(
                    ["seconv.exe", "sample.mkv", "bdnxml"],
                    scratch, {}, None,
                )
            self.assertEqual(receipt, b'{"success":true}')
            self.assertEqual(len(called), 1)
            self.assertEqual(called[0][0][0], "seconv.exe")
            self.assertEqual(called[0][1]["cwd"], str(scratch))
            self.assertEqual(called[0][1]["stdin"], image_adapter.subprocess.DEVNULL)
            self.assertEqual(called[0][1]["env"]["NO_PROXY"], "*")
            self.assertNotIn("HTTP_PROXY", called[0][1]["env"])


if __name__ == "__main__":
    unittest.main()
