"""Opt-in private frame evidence: bounded, no source modification."""
from __future__ import annotations

from io import BytesIO, StringIO
from contextlib import redirect_stderr
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

from PIL import Image

import hardsub_extractor.review_pack as review


def jpeg():
    output = BytesIO()
    Image.new("RGB", (320, 180), (10, 50, 80)).save(output, "JPEG")
    return output.getvalue()


def source_files(root, subtitles):
    video, srt = root / "source.mp4", root / "subtitle.srt"
    video.write_bytes(b"source remains unchanged")
    srt.write_text(subtitles, encoding="utf-8")
    return video, srt


def dialogue():
    return (
        "1\n00:00:01,000 --> 00:00:03,000\n"
        "固定電視台台標文字\n第一句對話\n\n"
        "2\n00:00:04,000 --> 00:00:06,000\n"
        "固定電視台台標文字\n第二句對話\n\n"
        "3\n00:00:07,000 --> 00:00:09,000\n"
        "固定電視台台標文字\n第三句對話\n"
    )


class ReviewPackTest(unittest.TestCase):
    def setUp(self):
        self.frame = jpeg()

    def run_pack(self, root, **kwargs):
        movie, srt = source_files(root, dialogue())
        original = movie.read_bytes(), srt.read_bytes()
        dest = root / "private-review"
        with (mock.patch.object(review, "_probe_local_video_duration", return_value=12.0),
              mock.patch.object(review, "extract_frame", return_value=self.frame) as calls):
            status = review.create_pack(movie, srt, dest, language="zh", **kwargs)
        self.assertEqual((movie.read_bytes(), srt.read_bytes()), original)
        return status, dest, calls

    def test_ogv_review_uses_real_ffprobe_contract_without_video_rewrite(self):
        from hardsub_extractor.quality_audit_io import _probe_local_video_duration
        with tempfile.TemporaryDirectory() as td:
            movie = Path(td) / "public-japanese.ogv"
            movie.write_bytes(b"synthetic container fixture, not decoded")
            with mock.patch(
                "hardsub_extractor.quality_audit_io.subprocess.run",
                return_value=mock.Mock(
                    returncode=0, stdout=b'{"format":{"duration":"180.323"}}'
                ),
            ) as probe:
                seconds = _probe_local_video_duration(
                    movie, binary_resolver=lambda _: "ffprobe",
                )
            self.assertAlmostEqual(seconds, 180.323)
            cmd = probe.call_args.args[0]
            self.assertIn("-protocol_whitelist", cmd)
            self.assertEqual(cmd[cmd.index("-protocol_whitelist") + 1], "file,pipe")
            self.assertEqual(movie.read_bytes(), b"synthetic container fixture, not decoded")

    def test_warned_cue_frame_pair_and_text_free_manifest(self):
        with tempfile.TemporaryDirectory() as td:
            r, dest, calls = self.run_pack(Path(td))
            self.assertEqual((r["cue_count"], r["frame_count"]), (1, 2))
            self.assertEqual(calls.call_count, 2)
            data = json.loads((dest / "index.json").read_text("utf-8"))
            self.assertEqual(data["review_cues"][0]["cue"], 1)
            self.assertEqual(data["review_cues"][0]["warning_codes"], [
                "repeated_multiline_overlay_review",
            ])
            self.assertEqual([x["second"] for x in data["review_cues"][0]["frames"]],
                             [1.4, 2.6])
            self.assertTrue((dest / "contact-sheet.jpg").is_file())
            self.assertTrue((dest / "cue-00001-early.jpg").is_file())
            self.assertNotIn("固定電視台台標文字", str(data))
            self.assertNotIn(str(Path(td)), str(data))

    def test_owner_local_html_pairs_escaped_ocr_and_source_frames(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            hostile = "OCR <script src='https://bad.example/x'></script> & 실제 대사"
            video, source = source_files(
                root, "1\n00:00:01,000 --> 00:00:03,000\n" + hostile + "\n",
            )
            unchanged = video.read_bytes(), source.read_bytes()
            dest = root / "private"
            with (mock.patch.object(review, "_probe_local_video_duration", return_value=4.0),
                  mock.patch.object(review, "extract_frame", return_value=self.frame)):
                status = review.create_pack(video, source, dest, mode="all", language="ko")
            self.assertEqual(status["status"], "CREATED")
            html = (dest / "review.html").read_text("utf-8")
            self.assertIn("OCR &lt;script", html)
            self.assertIn("&lt;/script&gt; &amp; 실제 대사", html)
            self.assertNotIn("<script src=", html)
            self.assertNotIn('<img src="https:', html)
            self.assertIn("script-src 'none'", html)
            self.assertIn("connect-src 'none'", html)
            self.assertIn('src="cue-00001-early.jpg"', html)
            self.assertIn('src="cue-00001-late.jpg"', html)
            self.assertNotIn(hostile, (dest / "index.json").read_text("utf-8"))
            self.assertEqual((video.read_bytes(), source.read_bytes()), unchanged)

    def test_long_ocr_text_is_bounded_in_private_report(self):
        from hardsub_extractor.models import Cue
        page = review.private_review_page(
            [Cue(0.0, 2.0, "<b>" + "자" * 30000 + "</b>")], [(1, [])]
        )
        self.assertLessEqual(len(page), 128 * 1024)
        self.assertIn("긴 OCR 문장은 일부만 표시", page.decode("utf-8"))
        self.assertNotIn("<b>", page.decode("utf-8"))

    def test_native_split_pair_review_uses_pinned_pairs_and_pagination(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            created, folder, decoded = self.run_pack(
                root, mode="split", split_cue_numbers=(1,),
                max_cues=1,
            )
            self.assertEqual((created["cue_count"], created["last_cue"]), (1, 1))
            self.assertEqual(decoded.call_count, 2)
            manifest = json.loads((folder / "index.json").read_text("utf-8"))
            self.assertEqual(manifest["mode"], "split")
            self.assertEqual([x["cue"] for x in manifest["review_cues"]], [1])
            self.assertIn("native_split_pair_review",
                          manifest["review_cues"][0]["warning_codes"])
            self.assertNotIn("第一句對話", str(manifest))
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            page, folder, decoded = self.run_pack(
                root, mode="split", split_cue_numbers=(1,),
                after_cue=1, max_cues=1,
            )
            self.assertEqual((page["cue_count"], page["last_cue"]), (1, 2))
            self.assertEqual(decoded.call_count, 2)
            manifest = json.loads((folder / "index.json").read_text("utf-8"))
            self.assertEqual([x["cue"] for x in manifest["review_cues"]], [2])

    def test_native_split_pair_rejects_invalid_stale_number_before_images(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            video, subtitle = source_files(root, dialogue())
            output = root / "no-private-frames"
            with (
                mock.patch.object(review, "_probe_local_video_duration",
                                  return_value=12.0),
                mock.patch.object(review, "extract_frame") as decode,
            ):
                for pairs in (None, (3,), (1, 1), (1, 2),
                              ("1",), (True,), tuple(range(1, 10))):
                    with self.subTest(pairs=pairs), self.assertRaises(ValueError):
                        review.create_pack(
                            video, subtitle, output,
                            mode="split", split_cue_numbers=pairs,
                        )
                self.assertFalse(output.exists())
                decode.assert_not_called()

    def test_all_mode_is_capped_and_existing_output_is_not_overwritten(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            r, dest, _ = self.run_pack(root, mode="all", max_cues=2)
            self.assertEqual((r["cue_count"], r["frame_count"]), (2, 4))
            old = (dest / "index.json").read_bytes()
            with self.assertRaisesRegex(ValueError, "review_output_exists"):
                review.create_pack(root/"source.mp4", root/"subtitle.srt",
                                   dest, language="zh", mode="all")
            self.assertEqual((dest / "index.json").read_bytes(), old)

    def test_long_video_review_can_resume_at_an_exact_original_cue(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            r, out, calls = self.run_pack(
                root, mode="all", after_cue=1, max_cues=2,
            )
            self.assertEqual((r["cue_count"], r["last_cue"]), (2, 3))
            self.assertEqual(calls.call_count, 4)
            index = json.loads((out / "index.json").read_text("utf-8"))
            self.assertEqual(index["after_cue"], 1)
            self.assertEqual([row["cue"] for row in index["review_cues"]],
                             [2, 3])
            self.assertFalse((out / "cue-00001-early.jpg").exists())

    def test_flagged_resume_skips_earlier_warning_without_creating_files(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            movie, srt = source_files(root, dialogue())
            out = root / "no-new-warnings"
            with (mock.patch.object(review, "_probe_local_video_duration", return_value=12.0),
                  mock.patch.object(review, "extract_frame") as ff):
                result = review.create_pack(
                    movie, srt, out, language="zh", after_cue=1,
                )
            self.assertEqual(result["status"], "NO_REVIEW_LOCATIONS")
            self.assertEqual(result["last_cue"], 1)
            self.assertFalse(out.exists())
            ff.assert_not_called()

    def test_flagged_mode_finds_late_warnings_beyond_default_audit_cap(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            lines = []
            for n in range(1, 41):
                lines.append(
                    f"{n}\n00:00:{n:02d},000 --> 00:00:{n:02d},900"
                    "\nLatin-only scene text\n"
                )
            video, srt = source_files(root, "\n".join(lines))
            output = root / "late-private-review"
            original = video.read_bytes(), srt.read_bytes()
            with (mock.patch.object(review, "_probe_local_video_duration",
                                    return_value=60.0),
                  mock.patch.object(review, "extract_frame",
                                    return_value=self.frame) as decoded):
                result = review.create_pack(
                    video, srt, output, language="zh",
                    mode="flagged", after_cue=35, max_cues=2,
                )
            self.assertEqual(
                (result["cue_count"], result["frame_count"], result["last_cue"]),
                (2, 4, 37),
            )
            self.assertEqual(decoded.call_count, 4)
            manifest = json.loads((output / "index.json").read_text("utf-8"))
            self.assertEqual(
                [row["cue"] for row in manifest["review_cues"]], [36, 37],
            )
            self.assertEqual((video.read_bytes(), srt.read_bytes()), original)
            self.assertNotIn("Latin-only scene text", str(manifest))

    def test_out_of_range_resume_cue_is_rejected_before_image_decode(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            movie, srt = source_files(root, dialogue())
            out = root / "not-written"
            with (mock.patch.object(review, "_probe_local_video_duration", return_value=12.0),
                  mock.patch.object(review, "extract_frame") as ff):
                for bad in (-1, 4, 1.0, True):
                    with self.subTest(bad=bad), self.assertRaises(ValueError):
                        review.create_pack(
                            movie, srt, out, mode="all", after_cue=bad,
                        )
            self.assertFalse(out.exists())
            ff.assert_not_called()

    def test_no_warning_means_no_output_and_no_ffmpeg(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            movie, srt = source_files(root,
                "1\n00:00:01,000 --> 00:00:03,000\n좋은 아침입니다\n")
            dest = root / "absent"
            with (mock.patch.object(review, "_probe_local_video_duration", return_value=5.0),
                  mock.patch.object(review, "extract_frame") as decoding):
                r = review.create_pack(movie, srt, dest)
            self.assertEqual(r["status"], "NO_REVIEW_LOCATIONS")
            self.assertFalse(dest.exists())
            decoding.assert_not_called()

    def test_untrusted_playlist_is_rejected_before_decoder(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            _, srt = source_files(root, dialogue())
            media = root / "untrusted.m3u8"
            media.write_text("#EXTM3U\nhttps://example.org/video")
            with mock.patch.object(review, "extract_frame") as decoder:
                with self.assertRaises(ValueError):
                    review.create_pack(media, srt, root/"private", language="zh")
                decoder.assert_not_called()

    def test_decoder_does_not_allow_network_input_or_write_to_media(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            movie, _ = source_files(root, dialogue())
            result = mock.Mock(returncode=0, stdout=self.frame)
            with (mock.patch.object(review, "resolve_binary", return_value="ffmpeg"),
                  mock.patch.object(review.subprocess, "run", return_value=result) as proc):
                self.assertEqual(review.extract_frame(movie, 1.25), self.frame)
            cmd = proc.call_args.args[0]
            self.assertEqual(cmd[cmd.index("-protocol_whitelist")+1], "file,pipe")
            self.assertIn("-nostdin", cmd)
            self.assertNotIn("-y", cmd)
            self.assertIs(proc.call_args.kwargs["stdin"], review.subprocess.DEVNULL)
            self.assertIs(proc.call_args.kwargs["stderr"], review.subprocess.DEVNULL)

    def test_invalid_decoder_or_timeout_do_not_save_any_output(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            movie, _ = source_files(root, dialogue())
            bad = mock.Mock(returncode=0, stdout=b"notjpeg" * 30)
            with (mock.patch.object(review, "resolve_binary", return_value="ffmpeg"),
                  mock.patch.object(review.subprocess, "run", return_value=bad)):
                with self.assertRaises(ValueError):
                    review.extract_frame(movie, 1.25)
            with (mock.patch.object(review, "resolve_binary", return_value="ffmpeg"),
                  mock.patch.object(review.subprocess, "run",
                                    side_effect=review.subprocess.TimeoutExpired("private", 35))):
                with self.assertRaises(ValueError):
                    review.extract_frame(movie, 1.25)
            self.assertEqual(sorted(x.name for x in root.iterdir()),
                             ["source.mp4", "subtitle.srt"])

    def test_second_frame_failure_leaves_no_partial_folder(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            movie, srt = source_files(root, dialogue())
            dest = root / "new"
            with (mock.patch.object(review, "_probe_local_video_duration", return_value=12.0),
                  mock.patch.object(review, "extract_frame",
                                    side_effect=[self.frame, ValueError("fail")])):
                with self.assertRaises(ValueError):
                    review.create_pack(movie, srt, dest, language="zh")
            self.assertFalse(dest.exists())

    def test_cue_beyond_media_eof_cannot_create_output(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            movie, srt = source_files(root,
                "1\n00:00:20,000 --> 00:00:22,000\n원본 확인할 자막\n")
            out = root/"private"
            with (mock.patch.object(review, "_probe_local_video_duration", return_value=5.0),
                  mock.patch.object(review, "extract_frame") as decoder):
                with self.assertRaises(ValueError):
                    review.create_pack(movie, srt, out, mode="all")
            self.assertFalse(out.exists())
            decoder.assert_not_called()


if __name__ == "__main__":
    unittest.main()
