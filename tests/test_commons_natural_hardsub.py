"""Offline contracts for publisher-authored, naturally burned-in study.

No network, video download, external executable or model calls in this suite.
Reference text is synthetic test text, not copied from the publisher.
"""
from __future__ import annotations

import hashlib
import io
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import verify_commons_natural_hardsub as trial


class Response(io.BytesIO):
    def __init__(self, body, url, *, declared=None, status=200, final=None):
        super().__init__(body)
        self._url = final if final is not None else url
        self.status = status
        self.headers = (
            {"Content-Length": str(declared)}
            if declared is not None else {}
        )

    def geturl(self):
        return self._url


class FakeOpener:
    def __init__(self, response):
        self.response = response
        self.calls = 0

    def open(self, request, timeout):
        self.calls += 1
        return self.response


def fake_reference(first_start="00:00:00,000", *, last_lines=2):
    expected = trial.EXPECTED_TIMES
    lines = []
    for index, ((start, end), count) in enumerate(
        zip(expected, trial.EXPECTED_LINES), 1
    ):
        def stamp(value):
            minute = int(value // 60)
            seconds = int(value % 60)
            ms = round((value - int(value)) * 1000)
            return f"00:{minute:02d}:{seconds:02d},{ms:03d}"
        a = first_start if index == 1 else stamp(start)
        b = stamp(end)
        n = last_lines if index == 4 else count
        text = "\n".join(f"synthetic line {j}" for j in range(n))
        lines.append(f"{index}\n{a} --> {b}\n{text}")
    return ("\n\n".join(lines) + "\n").encode("utf-8")


class NaturalCaptionContractTests(unittest.TestCase):
    def test_real_media_source_metadata_is_frozen_with_official_sha1(self):
        self.assertEqual(set(trial.PUBLISHED), {"caption", "clean"})
        for url, size, digest in trial.PUBLISHED.values():
            self.assertTrue(url.startswith(
                "https://upload.wikimedia.org/wikipedia/commons/"
            ))
            self.assertTrue(url.endswith(".webm"))
            self.assertTrue(size > 2_000_000)
            self.assertEqual(len(digest), 40)
            int(digest, 16)
        self.assertIn("oldid=1117709522", trial.REFERENCE_URL)
        self.assertIn("action=raw", trial.REFERENCE_URL)

    def test_approval_required_before_any_network_or_model(self):
        with mock.patch.object(trial, "run_study") as run:
            code = trial.main([
                "--model-dir", "not-used", "--manifest", "not-used",
            ])
        self.assertEqual(code, 2)
        run.assert_not_called()

    def test_media_fetch_verified_sha_and_new_file_only(self):
        data = b"TEST VIDEO DATA" * 130
        url = trial.PUBLISHED["caption"][0]
        pin = (url, len(data), hashlib.sha1(data).hexdigest())
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "caption.webm"
            opener = FakeOpener(Response(data, url, declared=len(data)))
            with mock.patch.dict(trial.PUBLISHED, {"caption": pin}):
                sha256 = trial._get_media(target, "caption", opener=opener)
            self.assertEqual(sha256, hashlib.sha256(data).hexdigest())
            self.assertEqual(target.read_bytes(), data)
            self.assertEqual(opener.calls, 1)
            self.assertFalse(target.with_suffix(".partial").exists())

    def test_failed_hash_never_publishes_partial_media(self):
        data = b"real media cannot be replaced" * 100
        url, _, digest = trial.PUBLISHED["caption"]
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "new.webm"
            with mock.patch.dict(
                trial.PUBLISHED, {"caption": (url, len(data), digest)},
            ):
                with self.assertRaises(ValueError):
                    trial._get_media(
                        target, "caption",
                        opener=FakeOpener(Response(data, url)),
                    )
            self.assertFalse(target.exists())
            self.assertFalse(target.with_suffix(".partial").exists())

    def test_media_rejects_redirect_and_length_drift(self):
        data = b"a" * 2048
        url = trial.PUBLISHED["caption"][0]
        pin = (url, len(data), hashlib.sha1(data).hexdigest())
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "caption.webm"
            with mock.patch.dict(trial.PUBLISHED, {"caption": pin}):
                with self.assertRaises(ValueError):
                    trial._get_media(
                        target, "caption", opener=FakeOpener(Response(
                            data, url, final="https://invalid.example/file.webm",
                        )),
                    )
                self.assertFalse(target.exists())
                with self.assertRaises(ValueError):
                    trial._get_media(
                        target, "caption", opener=FakeOpener(Response(
                            data, url, declared=len(data) + 1,
                        )),
                    )
                self.assertFalse(target.with_suffix(".partial").exists())

    def test_media_rejects_existing_user_file_before_network(self):
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "keep.webm"
            target.write_bytes(b"do not replace")
            opener = mock.Mock()
            with self.assertRaises(FileExistsError):
                trial._get_media(target, "caption", opener=opener)
            opener.open.assert_not_called()
            self.assertEqual(target.read_bytes(), b"do not replace")

    def test_reference_failure_precedes_all_video_fetches_and_inference(self):
        # Public media files must not be downloaded if independently pinned
        # publisher TimedText has already drifted.
        with (
            mock.patch.object(
                trial, "_get_reference",
                side_effect=ValueError("publisher transcript drift"),
            ),
            mock.patch.object(trial, "_get_media") as media,
            mock.patch.object(trial, "_probe") as probe,
        ):
            with self.assertRaisesRegex(ValueError, "publisher transcript drift"):
                trial.run_study(Path("not-used"), Path("not-used"))
            media.assert_not_called()
            probe.assert_not_called()

    def test_reference_is_independent_four_cue_publisher_revision(self):
        raw = fake_reference()
        opener = FakeOpener(Response(raw, trial.REFERENCE_URL))
        # Synthetic text is intentionally different from the real published
        # transcript. Only the fixture's hash is temporarily injected.
        synthetic_cues = trial.parse_strict_srt(raw)
        synthetic_sha = hashlib.sha256(
            "\x1e".join(cue.text for cue in synthetic_cues).encode("utf-8")
        ).hexdigest()
        with mock.patch.object(trial, "EXPECTED_TEXT_SHA256", synthetic_sha):
            cues, sha256 = trial._get_reference(opener=opener)
        self.assertEqual(len(cues), 4)
        self.assertEqual([len(c.text.splitlines()) for c in cues],
                         [2, 2, 1, 2])
        self.assertEqual(sha256, hashlib.sha256(raw).hexdigest())
        self.assertEqual(opener.calls, 1)

    def test_unchanged_times_and_layout_but_changed_text_are_rejected(self):
        original = fake_reference()
        original_cues = trial.parse_strict_srt(original)
        approved_hash = hashlib.sha256(
            "\x1e".join(cue.text for cue in original_cues).encode("utf-8")
        ).hexdigest()
        changed = original.replace(b"synthetic line 0", b"changed meaning 0")
        # In the same source revision, text is part of the witness. Merely
        # matching four timestamp pairs and row counts is not sufficient.
        with mock.patch.object(trial, "EXPECTED_TEXT_SHA256", approved_hash):
            with self.assertRaisesRegex(ValueError, "exact source text"):
                trial._get_reference(
                    opener=FakeOpener(Response(changed, trial.REFERENCE_URL))
                )

    def test_reference_wrong_timing_and_layout_rejected(self):
        for body in (
            fake_reference("00:00:00,250"),
            fake_reference(last_lines=1),
            b"<html>not a subtitle</html>",
        ):
            with self.subTest(length=len(body)), self.assertRaises(ValueError):
                trial._get_reference(
                    opener=FakeOpener(Response(body, trial.REFERENCE_URL)),
                )

    def test_reference_unapproved_redirect_rejected(self):
        raw = fake_reference()
        with self.assertRaises(ValueError):
            trial._get_reference(opener=FakeOpener(Response(
                raw, trial.REFERENCE_URL,
                final="https://somewhere.example/raw",
            )))

    def test_no_cue_control_is_not_silently_fabricated(self):
        from hardsub_extractor import pipeline
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with mock.patch.object(
                pipeline, "extract",
                side_effect=RuntimeError("인식된 자막이 없습니다. 영상에 자막 없음"),
            ):
                cues, count = trial._hse_cues(
                    root / "source.webm", root / "clean-result", object(),
                )
            self.assertEqual((cues, count), ([], 0))
            self.assertEqual(list(root.iterdir()), [])

    def test_paired_quality_requires_explicit_candidate_before_media_or_model(self):
        # The paired benchmark is opt-in and never auto-runs in production
        # or downloads publisher/user content merely to check its arguments.
        with mock.patch.object(trial, "_get_reference",
                               side_effect=AssertionError("network forbidden")):
            for invalid in (
                {"ocr_mode": "production", "compare_production": True},
                {"ocr_mode": "light-text-temporal", "compare_production": "yes"},
            ):
                with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                    trial.run_study(Path("not-used"),Path("not-used"),
                                    **invalid)

    def test_temporal_research_variant_is_explicit_and_preserves_defaults(self):
        from hardsub_extractor.detector import DetectorConfig
        self.assertIsNone(trial._study_detector("production"))
        standard = DetectorConfig()
        self.assertIsNone(standard.max_quiet_segment_seconds)
        self.assertIsNone(standard.max_pending_change_seconds)
        candidate = trial._study_detector("bounded-1p5s")
        self.assertEqual(candidate.max_quiet_segment_seconds, 1.5)
        self.assertEqual(candidate.max_pending_change_seconds, 0.75)
        with self.assertRaisesRegex(ValueError, "unsupported"):
            trial._study_detector("auto-promote")

    def test_temporal_variant_reaches_actual_streaming_extraction(self):
        from hardsub_extractor import pipeline
        candidate = trial._study_detector("bounded-1p5s")
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with mock.patch.object(
                pipeline, "extract",
                side_effect=RuntimeError("인식된 자막이 없습니다."),
            ) as called:
                observed, count = trial._hse_cues(
                    root / "source.webm", root / "testout",
                    object(), detector=candidate,
                )
            self.assertEqual((observed, count), ([], 0))
            self.assertIs(called.call_args.kwargs["detector"], candidate)
            self.assertEqual(called.call_args.kwargs["sample_fps"], 8.0)
            self.assertTrue(called.call_args.kwargs["streaming"])

    def test_research_row_selector_preserves_short_real_dialogue(self):
        from hardsub_extractor.models import OCRObservation
        base = mock.Mock(total_calls=7, retry_calls=3)
        original = OCRObservation(
            "first dialog line\nsecond dialog line", 0.92, 5.0,
            {"detections": [{"line": 0}, {"line": 1}]},
        )
        base.recognize.return_value = original
        wrapper = trial._BottomTwoLineResearch(base)
        frame = object()
        self.assertIs(wrapper.recognize(frame), original)
        self.assertEqual(wrapper.dropped_rows, 0)
        self.assertEqual(wrapper.modified_observations, 0)
        self.assertEqual(wrapper.total_calls, 7)
        self.assertEqual(wrapper.retry_calls, 3)
        base.recognize.assert_called_once_with(frame)

    def test_research_row_selector_discards_top_rows_without_ocr_oracle(self):
        from hardsub_extractor.models import OCRObservation
        base = mock.Mock(total_calls=9, retry_calls=4)
        base.recognize.return_value = OCRObservation(
            "credits\nmore credits\nsubtitle top\nsubtitle bottom",
            0.89, 2.5, {"detections": [
                {"line": 0, "score": 0.9, "text": "credits"},
                {"line": 1, "score": 0.9, "text": "more credits"},
                {"line": 2, "score": 0.9, "text": "subtitle top"},
                {"line": 3, "score": 0.9, "text": "subtitle bottom"},
            ]},
        )
        wrapper = trial._BottomTwoLineResearch(base)
        out = wrapper.recognize(object())
        self.assertEqual(out.text, "subtitle top\nsubtitle bottom")
        self.assertEqual(out.pts, 2.5)
        self.assertEqual(out.confidence, 0.89)
        self.assertEqual([x["line"] for x in out.metadata["detections"]], [0, 1])
        self.assertTrue(out.metadata["research_bottom_two_lines_only"])
        self.assertEqual(out.metadata["research_discarded_line_count"], 2)
        self.assertEqual(wrapper.dropped_rows, 2)
        self.assertEqual(wrapper.modified_observations, 1)
        self.assertEqual(wrapper.total_calls, 9)
        self.assertEqual(wrapper.retry_calls, 4)

    def test_diagnostics_locate_error_without_returning_public_transcript(self):
        reference = [
            trial.Cue(1.0, 3.0, "Alpha\nBravo"),
            trial.Cue(4.0, 6.0, "Charlie"),
        ]
        recognized = [
            trial.Cue(1.125, 2.875, "Alpha Bravo"),
            trial.Cue(7.0, 8.0, "unexpected"),
        ]
        control = [trial.Cue(8.0, 9.0, "logo")]
        info = trial._cue_error_diagnostics(reference, recognized, control)
        rows = info["reference_cue_diagnostics"]
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0]["reference_chars_excluding_newline"], 10)
        self.assertEqual(rows[0]["character_edits"], 1)
        self.assertFalse(rows[0]["exact_visible_text"])
        self.assertFalse(rows[0]["exact_original_line_wrap"])
        self.assertEqual(rows[0]["start_signed_offset_ms"], 125)
        self.assertEqual(rows[0]["end_signed_offset_ms"], -125)
        self.assertFalse(rows[1]["matched"])
        self.assertEqual(info["unmatched_extracted_cues"][0]["start_ms"], 7000)
        self.assertEqual(info["non_caption_control_detections"][0]["end_ms"], 9000)
        self.assertNotIn("Alpha", str(info))
        self.assertNotIn("logo", str(info))
        self.assertFalse(info["alignment_uses_reference_transcript_text"])

    def test_diagnostics_fail_closed_on_implausible_cue_volume(self):
        reference = [trial.Cue(1, 2, "original")]
        junk = [trial.Cue(0, 1, "x")] * 201
        with self.assertRaisesRegex(ValueError, "implausibly many cues"):
            trial._cue_error_diagnostics(reference, junk, [])
        with self.assertRaisesRegex(ValueError, "implausibly many cues"):
            trial._cue_error_diagnostics(reference, [], junk)

    def test_ffprobe_wrong_codec_is_rejected(self):
        output = b'{"format":{"duration":"15.026"},"streams":[{"codec_name":"h264","width":1920,"height":1080}]}'
        result = mock.Mock(returncode=0, stdout=output)
        with (
            mock.patch(
                "hardsub_extractor.runtime_paths.resolve_binary",
                return_value="fake-ffprobe.exe",
            ),
            mock.patch.object(trial.subprocess, "run", return_value=result),
        ):
            with self.assertRaises(ValueError):
                trial._probe(Path("never-opened.webm"))

    def test_correct_probe_metadata_is_purely_readonly(self):
        output = b'{"format":{"duration":"15.026"},"streams":[{"codec_name":"av1","width":1920,"height":1080}]}'
        result = mock.Mock(returncode=0, stdout=output)
        with (
            mock.patch(
                "hardsub_extractor.runtime_paths.resolve_binary",
                return_value="fake-ffprobe.exe",
            ),
            mock.patch.object(trial.subprocess, "run", return_value=result) as call,
        ):
            meta = trial._probe(Path("never-opened.webm"))
        self.assertEqual(meta["seconds"], 15.026)
        self.assertEqual(call.call_args.kwargs["timeout"], 35)
        self.assertIn("file,pipe", call.call_args.args[0])


if __name__ == "__main__":
    unittest.main()
