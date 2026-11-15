from __future__ import annotations

from contextlib import redirect_stdout, redirect_stderr
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from hardsub_extractor.models import Cue


CLI = Path(__file__).resolve().parents[1] / 'tools' / 'subtitle_quality_audit.py'
spec = importlib.util.spec_from_file_location('subtitle_quality_audit', CLI)
assert spec and spec.loader
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class QualityAuditCLITests(unittest.TestCase):
    def test_candidate_no_reference_never_prints_subtitle_text_or_path(self):
        with tempfile.TemporaryDirectory() as td:
            srt = Path(td)/'비공개-자막.srt'
            srt.write_text('1\n00:00:01,000 --> 00:00:02,000\n비공개 내용\n',encoding='utf-8')
            out = io.StringIO()
            with redirect_stdout(out):
                code=module.main(['--candidate',str(srt)])
            self.assertEqual(code,0)
            self.assertFalse((Path(td)/'audit.json').exists())
            report=json.loads(out.getvalue())
            self.assertEqual(report['candidate']['cue_count'],1)
            self.assertNotIn('비공개',out.getvalue())
            self.assertNotIn(str(srt),out.getvalue())

    def test_same_input_output_path_is_rejected_without_mutation(self):
        with tempfile.TemporaryDirectory() as td:
            path=Path(td)/'do-not-overwrite.srt'
            original='1\n00:00:01,000 --> 00:00:02,000\n중요한 기록\n'
            path.write_text(original,encoding='utf-8')
            err=io.StringIO()
            with redirect_stderr(err):
                code=module.main(['--candidate',str(path),'--json',str(path)])
            self.assertEqual(code,2)
            self.assertEqual(path.read_text(encoding='utf-8'),original)
            self.assertNotIn('중요한 기록',err.getvalue())

    def test_malformed_srt_cue_is_not_silently_dropped(self):
        with tempfile.TemporaryDirectory() as td:
            srt=Path(td)/'broken.srt'
            srt.write_text('1\n00:00:01,000 --> 00:00:02,000\n정상\n\n2\nbroken --> time\n누락 의심\n',encoding='utf-8')
            err=io.StringIO()
            with redirect_stderr(err):
                code=module.main(['--candidate',str(srt)])
            self.assertEqual(code,2)
            self.assertNotIn('누락 의심',err.getvalue())

    def test_opt_in_json_is_scalar_only_and_utf8(self):
        with tempfile.TemporaryDirectory() as td:
            candidate=Path(td)/'extracted.srt'
            candidate.write_text('1\n00:00:01,000 --> 00:00:02,000\n비밀 자막\n',encoding='utf-8')
            report=Path(td)/'only-if-requested.json'
            with redirect_stdout(io.StringIO()):
                code=module.main(['--candidate',str(candidate),'--json',str(report)])
            self.assertEqual(code,0)
            payload=report.read_text(encoding='utf-8')
            self.assertEqual(json.loads(payload)['candidate']['cue_count'],1)
            self.assertNotIn('비밀',payload)
            self.assertNotIn(str(candidate),payload)

    def test_literal_arrows_in_dialogue_are_not_counted_as_cue_blocks(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "arrow.srt"
            path.write_text(
                "1\n00:00:01,000 --> 00:00:03,000\n"
                "화살표 --> 는 일반 대사입니다.\n", encoding="utf-8"
            )
            with redirect_stdout(io.StringIO()) as out:
                result = module.main(["--candidate", str(path)])
            self.assertEqual(result, 0)
            self.assertEqual(json.loads(out.getvalue())["candidate"]["cue_count"], 1)

    def test_malformed_srt_block_without_arrow_fails_closed(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "bad.srt"
            path.write_text(
                "1\n00:00:01,000 --> 00:00:02,000\n첫 대사\n\n"
                "2\n00:00:03,000 -> 00:00:04,000\n두 번째 대사\n",
                encoding="utf-8",
            )
            with redirect_stderr(io.StringIO()):
                code = module.main(["--candidate", str(path)])
            self.assertEqual(code, 2)

    def test_valid_vtt_with_note_metadata_and_cue_identifier(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "valid.vtt"
            path.write_text(
                "WEBVTT\n\nNOTE annotation\nNot a caption\n\n"
                "sub-01\n00:00:01.000 --> 00:00:02.500 align:start\n한국어 대사\n",
                encoding="utf-8",
            )
            with redirect_stdout(io.StringIO()) as out:
                code = module.main(["--candidate", str(path)])
            self.assertEqual(code, 0)
            self.assertEqual(json.loads(out.getvalue())["candidate"]["cue_count"], 1)

    def test_vtt_invalid_second_cue_is_not_silently_ignored(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "invalid.vtt"
            path.write_text(
                "WEBVTT\n\n00:00:01.000 --> 00:00:02.000\n첫 대사\n\n"
                "00:00:03.000 -> 00:00:04.000\n누락된 대사\n",
                encoding="utf-8",
            )
            with redirect_stderr(io.StringIO()):
                code = module.main(["--candidate", str(path)])
            self.assertEqual(code, 2)

    def test_legacy_cp949_srt_validation_still_works(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "cp949.srt"
            path.write_bytes(
                "1\n00:00:01,000 --> 00:00:02,000\n옛날 자막\n".encode("cp949")
            )
            with redirect_stdout(io.StringIO()) as out:
                code = module.main(["--candidate", str(path)])
            self.assertEqual(code, 0)
            self.assertEqual(json.loads(out.getvalue())["candidate"]["cue_count"], 1)

    def test_existing_report_is_never_overwritten(self):
        with tempfile.TemporaryDirectory() as td:
            source = Path(td) / "source.srt"
            source.write_text(
                "1\n00:00:01,000 --> 00:00:02,000\n한국어 대사\n",
                encoding="utf-8"
            )
            report = Path(td) / "quality.json"
            original = "THIS FILE MUST SURVIVE"
            report.write_text(original, encoding="utf-8")
            with redirect_stderr(io.StringIO()):
                code = module.main([
                    "--candidate", str(source), "--json", str(report)
                ])
            self.assertEqual(code, 2)
            self.assertEqual(report.read_text(encoding="utf-8"), original)

    def test_report_hardlink_to_subtitle_is_never_truncated(self):
        with tempfile.TemporaryDirectory() as td:
            import os
            source = Path(td) / "source.srt"
            original = "1\n00:00:01,000 --> 00:00:02,000\n비공개 대사\n"
            source.write_text(original, encoding="utf-8")
            report = Path(td) / "hardlink.json"
            os.link(source, report)
            with redirect_stderr(io.StringIO()):
                code = module.main([
                    "--candidate", str(source), "--json", str(report)
                ])
            self.assertEqual(code, 2)
            self.assertEqual(source.read_text(encoding="utf-8"), original)

    def test_invalid_ass_dialogue_must_not_disappear(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "invalid.ass"
            path.write_text(
                "[Script Info]\nScriptType: v4.00+\n\n[Events]\n"
                "Format: Layer,Start,End,Style,Name,MarginL,MarginR,MarginV,Effect,Text\n"
                "Dialogue: 0,0:00:01.00,0:00:02.00,Default,,0,0,0,,첫 문장\n"
                "Dialogue: MISSING_FIELDS\n",
                encoding="utf-8",
            )
            with redirect_stderr(io.StringIO()):
                code = module.main(["--candidate", str(path)])
            self.assertEqual(code, 2)

    def test_bad_user_options_fail_closed_without_writing_report(self):
        with tempfile.TemporaryDirectory() as td:
            candidate=Path(td)/'extracted.srt'
            candidate.write_text('1\n00:00:01,000 --> 00:00:02,000\n자막\n',encoding='utf-8')
            report=Path(td)/'audit.json'
            with redirect_stderr(io.StringIO()):
                code=module.main(['--candidate',str(candidate),'--tolerance-seconds','-1','--json',str(report)])
            self.assertEqual(code,2)
            self.assertFalse(report.exists())


    def test_multilingual_cli_script_warning_is_not_always_hangul(self):
        with tempfile.TemporaryDirectory() as td:
            source = Path(td) / "private-japanese.srt"
            source.write_text(
                "1\n00:00:01,000 --> 00:00:03,000\n今日は学校へ行きます\n",
                encoding="utf-8",
            )
            for language in ("ja", "zh", "en"):
                with self.subTest(language=language), redirect_stdout(io.StringIO()) as out:
                    code = module.main([
                        "--candidate", str(source), "--language", language,
                    ])
                self.assertEqual(code, 0)
                payload = json.loads(out.getvalue())
                self.assertEqual(payload["candidate"]["requested_language"], language)
                self.assertNotIn(
                    "no_hangul_review", payload["candidate"]["warning_counts"]
                )
                self.assertNotIn("今日は", out.getvalue())
                self.assertNotIn(str(source), out.getvalue())

    def test_cli_duration_review_is_explicit_opt_in_with_no_side_effect(self):
        with tempfile.TemporaryDirectory() as td:
            source = Path(td) / "synthetic-mark.srt"
            source.write_text(
                "1\n00:00:00,000 --> 00:00:07,125\n방송 안내\n",
                encoding="utf-8",
            )
            original = source.read_bytes()
            output = io.StringIO()
            with redirect_stdout(output):
                result = module.main([
                    "--candidate", str(source),
                    "--language", "ko",
                    "--video-duration-seconds", "7.125",
                ])
            self.assertEqual(result, 0)
            payload = json.loads(output.getvalue())
            self.assertEqual(
                payload["candidate"]["warning_counts"][
                    "near_full_runtime_single_cue_review"
                ], 1,
            )
            self.assertEqual(source.read_bytes(), original)
            self.assertFalse(list(Path(td).glob("*.json")))

    def test_cli_rejects_untrusted_language_and_duration_without_writing(self):
        with tempfile.TemporaryDirectory() as td:
            source = Path(td) / "private.srt"
            source.write_text(
                "1\n00:00:01,000 --> 00:00:02,000\n비공개 문장\n",
                encoding="utf-8",
            )
            for params in (
                ["--language", "xx"],
                ["--video-duration-seconds", "nan"],
                ["--video-duration-seconds", "-1"],
                ["--video-duration-seconds", "inf"],
            ):
                report = Path(td) / "audit.json"
                with self.subTest(params=params), redirect_stderr(io.StringIO()) as err:
                    code = module.main([
                        "--candidate", str(source), "--json", str(report), *params,
                    ])
                self.assertEqual(code, 2)
                self.assertFalse(report.exists())
                self.assertNotIn("비공개", err.getvalue())



    def test_local_video_duration_probe_is_network_disabled_and_read_only(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            video = root / "user-private-video.mp4"
            video.write_bytes(b"test-double-not-an-actual-video")
            subtitle = root / "private-caption.srt"
            subtitle.write_text(
                "1\n00:00:00,000 --> 00:00:07,125\n비공개 자막\n",
                encoding="utf-8",
            )
            before = (video.read_bytes(), subtitle.read_bytes())
            fake = mock.Mock(
                returncode=0, stdout=b'{"format":{"duration":"7.125"}}',
            )
            result_buffer = io.StringIO()
            with mock.patch.object(
                module, "resolve_binary", return_value="ffprobe-test",
            ), mock.patch.object(
                module.subprocess, "run", return_value=fake,
            ) as probe, redirect_stdout(result_buffer):
                status = module.main([
                    "--candidate", str(subtitle), "--video", str(video),
                    "--language", "ko",
                ])
            self.assertEqual(status, 0)
            report = json.loads(result_buffer.getvalue())
            self.assertEqual(
                report["candidate"]["warning_counts"].get(
                    "near_full_runtime_single_cue_review"
                ), 1,
            )
            self.assertTrue(report["candidate"]["independent_video_duration_supplied"])
            args, kwargs = probe.call_args
            self.assertIn("-protocol_whitelist", args[0])
            self.assertIn("file,pipe", args[0])
            self.assertNotIn("http,https", args[0])
            self.assertNotIn("-headers", args[0])
            self.assertEqual(args[0][-2], "-i")
            self.assertEqual(args[0][-1], str(video))
            self.assertIs(kwargs["stdin"], module.subprocess.DEVNULL)
            self.assertIs(kwargs["stderr"], module.subprocess.DEVNULL)
            self.assertFalse(kwargs["check"])
            self.assertEqual(kwargs["timeout"], 35)
            self.assertEqual((video.read_bytes(), subtitle.read_bytes()), before)
            self.assertNotIn(str(video), result_buffer.getvalue())
            self.assertNotIn(str(subtitle), result_buffer.getvalue())
            self.assertNotIn("비공개", result_buffer.getvalue())
            self.assertFalse(list(root.glob("*.json")))

    def test_video_duration_input_modes_are_mutually_exclusive(self):
        with redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit) as caught:
                module.main([
                    "--candidate", "ignored.srt",
                    "--video", "video.mp4",
                    "--video-duration-seconds", "7.125",
                ])
        self.assertEqual(caught.exception.code, 2)

    def test_malformed_probe_never_writes_a_quality_report(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            video = root / "local.mp4"
            video.write_bytes(b"untrusted format data")
            caption = root / "source.srt"
            original = "1\n00:00:01,000 --> 00:00:02,000\n원본 대사\n"
            caption.write_text(original, encoding="utf-8")
            cases = [
                mock.Mock(returncode=0, stdout=b'not-json'),
                mock.Mock(returncode=0, stdout=b'{"format":{}}'),
                mock.Mock(returncode=0, stdout=b'{"format":{"duration":"nan"}}'),
                mock.Mock(returncode=0, stdout=b'{"format":{"duration":"inf"}}'),
                mock.Mock(returncode=0, stdout=b'{"format":{"duration":"0"}}'),
                mock.Mock(returncode=0, stdout=b'{"format":{"duration":"50000"}}'),
                mock.Mock(returncode=0, stdout=b'x' * 32769),
                mock.Mock(returncode=1, stdout=b'{"format":{"duration":"7"}}'),
            ]
            for fake in cases:
                out = root / "review.json"
                with self.subTest(output=fake.stdout[:20]), mock.patch.object(
                    module, "resolve_binary", return_value="ffprobe-test",
                ), mock.patch.object(
                    module.subprocess, "run", return_value=fake,
                ), redirect_stderr(io.StringIO()) as err:
                    code = module.main([
                        "--candidate", str(caption), "--video", str(video),
                        "--json", str(out),
                    ])
                self.assertEqual(code, 2)
                self.assertFalse(out.exists())
                self.assertNotIn("원본", err.getvalue())
                self.assertNotIn(str(video), err.getvalue())
                self.assertEqual(caption.read_text(encoding="utf-8"), original)

    def test_invalid_local_video_extension_fails_before_spawning_probe(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            video = root / "media.m3u8"
            video.write_bytes(b"local playlist with internet references")
            caption = root / "source.srt"
            caption.write_text(
                "1\n00:00:01,000 --> 00:00:02,000\n대사\n",
                encoding="utf-8",
            )
            with mock.patch.object(module.subprocess, "run") as probe, redirect_stderr(io.StringIO()):
                result = module.main([
                    "--candidate", str(caption), "--video", str(video),
                ])
            self.assertEqual(result, 2)
            probe.assert_not_called()

    def test_probe_timeout_fails_closed_without_raw_video_path(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            video = root / "video.mp4"
            video.write_bytes(b"fake")
            subtitle = root / "source.srt"
            subtitle.write_text(
                "1\n00:00:01,000 --> 00:00:02,000\n민감한 텍스트\n",
                encoding="utf-8",
            )
            with mock.patch.object(
                module, "resolve_binary", return_value="ffprobe-test",
            ), mock.patch.object(
                module.subprocess, "run",
                side_effect=module.subprocess.TimeoutExpired("private command", 35),
            ), redirect_stderr(io.StringIO()) as err:
                result = module.main([
                    "--candidate", str(subtitle), "--video", str(video),
                ])
            self.assertEqual(result, 2)
            self.assertNotIn(str(video), err.getvalue())
            self.assertNotIn("민감한", err.getvalue())



if __name__=='__main__':
    unittest.main()
