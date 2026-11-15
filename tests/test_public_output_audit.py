"""Real saved SRT/ASS/provenance read-only resume audit, no network or OCR.

Fixtures are written by the production soft-caption publisher and OCR exporter.
Audit may open only expected siblings and may never repair, overwrite, delete
or otherwise change an existing user file. All error assertions are aggregate;
no private caption text or paths are logged by production code.
"""
from __future__ import annotations

import contextlib
import io
from dataclasses import replace
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest import mock

from hardsub_extractor.models import Cue
from hardsub_extractor.public_output_audit import (
    audit_soft_output, audit_visual_output,
)
from hardsub_extractor.public_video_ocr import (
    _finalize_ocr_result, resolve_missing_public_ocr,
)
from hardsub_extractor.subtitle_acquisition import (
    CaptionResult, PublicCaptionCollector, _commit_caption,
)
from hardsub_extractor.subtitle_language import SubtitleTrack
from hardsub_extractor.subtitle_batch_cli import main as cli_main
from hardsub_extractor.writers import write_subtitle_pair_exclusive

URL = "https://www.youtube.com/watch?v=resume-audit"
VTT = (b"WEBVTT\n\n"
       b"00:00:01.000 --> 00:00:02.500\nHello world\n\n"
       b"00:00:03.000 --> 00:00:05.000\nSecond line\n")


def soft_fixture(root: Path, *, video_id="resume_vid", site="youtube"):
    result = _commit_caption(
        VTT, SubtitleTrack("en", "manual", "vtt", "en"),
        video_id=video_id, provider=site, target_directory=root,
        original_extension=".vtt",
    )
    base = root / f"{site}-{video_id}-en-manual"
    return result, base


def check_soft(base: Path, *, video_id="resume_vid", site="youtube"):
    return audit_soft_output(
        base, provider=site, video_id=video_id,
        requested="en", provided="en", origin="manual",
    )


def visual_fixture(root: Path, *, lang="ja", video_id="resume_vid"):
    result = CaptionResult("youtube", lang, "needs_ocr", video_id=video_id)
    base = root / f"youtube-{video_id}-{lang}-hard-ocr"
    cues = [
        Cue(1.25, 3.5, "Sample line"),
        Cue(5.25, 7.8, "Second line"),
    ]
    srt, ass = write_subtitle_pair_exclusive(cues, base)
    saved = _finalize_ocr_result(
        result,
        SimpleNamespace(srt_path=srt, ass_path=ass, cue_count=2),
        base,
    )
    return saved, base


def saved_hashes(paths):
    return {
        str(p): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in paths
    }


class ReadOnlySoftOutputAuditTests(unittest.TestCase):
    def test_published_soft_captions_verify_without_touching_any_file(self):
        with tempfile.TemporaryDirectory() as td:
            result, base = soft_fixture(Path(td))
            originals = (result.original, result.srt, result.ass, result.manifest)
            before = saved_hashes(originals)
            audit = check_soft(base)
            self.assertEqual(audit.state, "verified")
            self.assertEqual(audit.cue_count, 2)
            self.assertEqual(audit.original, result.original)
            self.assertEqual(audit.srt, result.srt)
            self.assertEqual(audit.ass, result.ass)
            self.assertEqual(audit.manifest, result.manifest)
            self.assertEqual(audit.network_route, "direct")
            self.assertEqual(saved_hashes(originals), before)

    def test_missing_or_partial_files_are_never_assumed_verified(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            base = root / "youtube-resume_vid-en-manual"
            self.assertEqual(check_soft(base).state, "absent")
            base.with_suffix(".srt").write_bytes(b"USER_OLD_DATA")
            self.assertEqual(check_soft(base).state, "preserved")
            self.assertEqual(base.with_suffix(".srt").read_bytes(), b"USER_OLD_DATA")

    def test_corrupt_or_incomplete_real_published_sets_are_preserved(self):
        for variant in ("missing-original", "missing-srt", "wrong-srt",
                        "shifted-srt-clock", "bad-ass", "missing-manifest",
                        "bad-json", "oversized-json", "wrong-cue-count",
                        "wrong-site", "wrong-video-id", "wrong-language",
                        "wrong-track-origin", "wrong-original-file",
                        "wrong-srt-file", "wrong-ass-file",
                        "ocr-claim", "machine-translation-claim",
                        "wrong-original-timeline", "wrong-output-timeline"):
            with self.subTest(variant=variant), tempfile.TemporaryDirectory() as td:
                result, base = soft_fixture(Path(td))
                metadata = json.loads(result.manifest.read_text("utf-8"))
                if variant == "missing-original":
                    result.original.unlink()
                elif variant == "missing-srt":
                    result.srt.unlink()
                elif variant == "wrong-srt":
                    result.srt.write_bytes(b"BROKEN")
                elif variant == "shifted-srt-clock":
                    text = result.srt.read_text("utf-8")
                    result.srt.write_text(
                        text.replace("00:00:01,000", "00:00:12,000"), encoding="utf-8",
                    )
                elif variant == "bad-ass":
                    result.ass.write_bytes(b"[Events]\n")
                elif variant == "missing-manifest":
                    result.manifest.unlink()
                elif variant == "bad-json":
                    result.manifest.write_bytes(b"{invalid :")
                elif variant == "oversized-json":
                    result.manifest.write_bytes(b"X" * (64 * 1024 + 1))
                else:
                    field, val = {
                        "wrong-cue-count": ("cue_count", 999),
                        "wrong-site": ("site", "wetv"),
                        "wrong-video-id": ("video_id", "other"),
                        "wrong-language": ("language_provided", "ja"),
                        "wrong-track-origin": ("track_origin", "automatic"),
                        "wrong-original-file": ("original_file", "../../secrets.txt"),
                        "wrong-srt-file": ("srt_file", "OTHER.srt"),
                        "wrong-ass-file": ("ass_file", "OTHER.ass"),
                        "ocr-claim": ("ocr_applied", True),
                        "machine-translation-claim": ("machine_translation_applied", True),
                        "wrong-original-timeline": ("original_timeline", "shifted"),
                        "wrong-output-timeline": ("output_timeline", "shifted"),
                    }[variant]
                    metadata[field] = val
                    result.manifest.write_text(json.dumps(metadata), encoding="utf-8")
                existing = [p for p in (
                    result.original, result.srt, result.ass, result.manifest,
                ) if p.exists()]
                before = saved_hashes(existing)
                self.assertEqual(check_soft(base).state, "preserved")
                self.assertEqual(saved_hashes(existing), before)

    def test_wrong_provider_or_episode_never_validates_foreign_outputs(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            result, base = soft_fixture(root)
            before = saved_hashes((result.original, result.srt, result.ass, result.manifest))
            self.assertEqual(check_soft(base, site="wetv").state, "preserved")
            self.assertEqual(check_soft(base, video_id="wrong").state, "preserved")
            self.assertEqual(saved_hashes((result.original, result.srt, result.ass, result.manifest)), before)

    def test_symlinked_expected_result_is_not_followed(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            result, base = soft_fixture(root)
            with mock.patch.object(
                Path, "is_symlink",
                autospec=True,
                side_effect=lambda self: self == result.srt,
            ):
                self.assertEqual(check_soft(base).state, "preserved")
            self.assertEqual(result.srt.read_bytes()[:1], b"1")

    def test_complete_resume_skips_downloader_and_reports_own_verified_bytes(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            result, _ = soft_fixture(root)
            before = saved_hashes((result.original, result.srt, result.ass, result.manifest))
            commands = []
            def fake_runner(args, **kwargs):
                commands.append(args)
                if "--dump-single-json" not in args:
                    raise AssertionError("complete resume must not download a track")
                return json.dumps({
                    "id": "resume_vid",
                    "subtitles": {"en": [{"ext": "vtt"}]},
                }).encode()
            collector = PublicCaptionCollector("test-pinned", command_runner=fake_runner)
            result2 = collector.collect(
                URL, root, languages=("en",), skip_existing=True,
            )[0]
            self.assertEqual(result2.status, "existing_verified")
            self.assertEqual(result2.cue_count, 2)
            self.assertEqual(result2.original, result.original)
            self.assertEqual(len(commands), 1)
            self.assertEqual(saved_hashes((result.original, result.srt, result.ass, result.manifest)), before)

    def test_partial_resume_skips_network_download_and_remains_pending(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            path = root / "youtube-resume_vid-en-manual.srt"
            path.write_bytes(b"PARTIAL_KEEP")
            def fake_runner(args, **kwargs):
                if "--dump-single-json" not in args:
                    raise AssertionError("partial user-owned result must not be overwritten")
                return json.dumps({
                    "id": "resume_vid", "subtitles": {"en": [{"ext": "vtt"}]},
                }).encode()
            collector = PublicCaptionCollector("test-pinned", command_runner=fake_runner)
            returned = collector.collect(
                URL, root, languages=("en",), skip_existing=True,
            )[0]
            self.assertEqual(returned.status, "existing_preserved")
            self.assertEqual(returned.cue_count, 0)
            self.assertEqual(path.read_bytes(), b"PARTIAL_KEEP")


class ReadOnlyVisualOutputAuditTests(unittest.TestCase):
    def test_no_available_soft_tracks_can_verify_existing_visual_without_ocr(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            saved, base = visual_fixture(root, lang="ja")
            original_bytes = saved_hashes((saved.srt, saved.ass, saved.manifest))
            queries = []
            def metadata_only(args, **kwargs):
                queries.append(args)
                if "--dump-single-json" not in args:
                    raise AssertionError("no media/track download permitted for complete resume")
                return json.dumps({"id": "resume_vid", "subtitles": {},
                                   "automatic_captions": {}}).encode()
            collector = PublicCaptionCollector("yt-dlp-pinned", command_runner=metadata_only)
            result = collector.collect(
                URL, root, languages=("ja",), skip_existing=True,
            )[0]
            self.assertEqual(result.status, "existing_verified")
            self.assertEqual(result.cue_count, 2)
            self.assertEqual(result.track_origin, "visual_ocr")
            self.assertEqual(result.ass, saved.ass)
            self.assertEqual(len(queries), 1)
            self.assertEqual(saved_hashes((saved.srt, saved.ass, saved.manifest)), original_bytes)

    def test_no_track_with_partial_visual_is_preserved_and_not_restarted(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            saved, base = visual_fixture(root)
            saved.ass.unlink()
            def metadata_only(args, **kwargs):
                if "--dump-single-json" not in args:
                    raise AssertionError("partial OCR result must never be redownloaded")
                return json.dumps({"id": "resume_vid"}).encode()
            collector = PublicCaptionCollector("yt-dlp-pinned", command_runner=metadata_only)
            result = collector.collect(
                URL, root, languages=("ja",), skip_existing=True,
            )[0]
            self.assertEqual(result.status, "existing_preserved")
            self.assertEqual(result.track_origin, "visual_ocr")
            self.assertTrue(saved.srt.is_file())
            self.assertFalse(saved.ass.exists())

    def test_default_subtitle_only_cli_can_count_audited_old_visual_export(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            saved, base = visual_fixture(root)
            bytes_before = saved_hashes((saved.srt, saved.ass, saved.manifest))
            def metadata_only(args, **kwargs):
                if "--dump-single-json" not in args:
                    raise AssertionError("default CLI must not download a video")
                return json.dumps({"id": "resume_vid"}).encode()
            collector = PublicCaptionCollector("yt-dlp-pinned", command_runner=metadata_only)
            stdout = io.StringIO()
            with mock.patch(
                "hardsub_extractor.subtitle_batch_cli.PublicCaptionCollector",
                return_value=collector,
            ), contextlib.redirect_stdout(stdout):
                rc = cli_main([
                    "--url", URL, "--languages", "ja", "--out", str(root),
                ])
            self.assertEqual(rc, 0)
            self.assertIn("STATUS=existing_verified", stdout.getvalue())
            self.assertIn("SUMMARY downloaded=0 pending=0 verified=1", stdout.getvalue())
            self.assertNotIn(URL, stdout.getvalue())
            self.assertEqual(saved_hashes((saved.srt, saved.ass, saved.manifest)), bytes_before)

    def test_complete_real_visual_output_verifies_without_model_or_video(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            saved, base = visual_fixture(root)
            paths = (saved.srt, saved.ass, saved.manifest)
            before = saved_hashes(paths)
            audit = audit_visual_output(
                base, provider="youtube", video_id="resume_vid", language="ja",
            )
            self.assertEqual(audit.state, "verified")
            self.assertEqual(audit.cue_count, 2)
            self.assertEqual(saved_hashes(paths), before)

    def test_second_run_never_fetches_existing_complete_visual_video(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            saved, base = visual_fixture(root)
            before = saved_hashes((saved.srt, saved.ass, saved.manifest))
            fetcher = mock.Mock(side_effect=AssertionError("no media should download"))
            reader = mock.Mock(side_effect=AssertionError("no OCR should load"))
            returned = resolve_missing_public_ocr(
                URL, root, [CaptionResult(
                    "youtube", "ja", "needs_ocr", video_id="resume_vid",
                )], media_fetcher=fetcher, ocr_extractor=reader,
            )[0]
            self.assertEqual(returned.status, "existing_verified")
            self.assertEqual(returned.cue_count, 2)
            self.assertEqual(returned.track_origin, "visual_ocr")
            self.assertEqual(returned.srt, saved.srt)
            self.assertEqual(saved_hashes((saved.srt, saved.ass, saved.manifest)), before)
            fetcher.assert_not_called()
            reader.assert_not_called()

    def test_visual_partial_or_corrupted_metadata_is_not_verified(self):
        for variant in ("missing-ass", "missing-srt", "wrong-clock",
                        "wrong-text", "missing-provenance", "bad-json",
                        "wrong-site", "wrong-episode", "wrong-language",
                        "wrong-origin", "fake-verification", "wrong-cue-count",
                        "false-no-retention", "bad-file-name"):
            with self.subTest(variant=variant), tempfile.TemporaryDirectory() as td:
                root = Path(td)
                saved, base = visual_fixture(root)
                metadata = json.loads(saved.manifest.read_text("utf-8"))
                if variant == "missing-ass":
                    saved.ass.unlink()
                elif variant == "missing-srt":
                    saved.srt.unlink()
                elif variant == "wrong-clock":
                    saved.ass.write_text(
                        saved.ass.read_text("utf-8").replace("0:00:01.25", "0:00:41.25"),
                        encoding="utf-8",
                    )
                elif variant == "wrong-text":
                    saved.srt.write_text(
                        saved.srt.read_text("utf-8").replace("Sample line", "Wrong"),
                        encoding="utf-8",
                    )
                elif variant == "missing-provenance":
                    saved.manifest.unlink()
                elif variant == "bad-json":
                    saved.manifest.write_bytes(b"[invalid")
                else:
                    field, value = {
                        "wrong-site": ("site", "wetv"),
                        "wrong-episode": ("video_id", "unrelated"),
                        "wrong-language": ("language_requested", "zh"),
                        "wrong-origin": ("track_origin", "manual"),
                        "fake-verification": ("original_text_verified", True),
                        "wrong-cue-count": ("cue_count", 100),
                        "false-no-retention": ("temporary_video_retained", True),
                        "bad-file-name": ("ass_file", "../../outside.ass"),
                    }[variant]
                    metadata[field] = value
                    saved.manifest.write_text(json.dumps(metadata), encoding="utf-8")
                present = [p for p in (saved.srt, saved.ass, saved.manifest) if p.exists()]
                before = saved_hashes(present)
                audit = audit_visual_output(
                    base, provider="youtube", video_id="resume_vid", language="ja",
                )
                self.assertEqual(audit.state, "preserved")
                self.assertEqual(saved_hashes(present), before)

    def test_read_only_pending_status_never_starts_model_or_fetch_for_partial(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            saved, base = visual_fixture(root)
            saved.ass.unlink()
            downloader = mock.Mock(side_effect=AssertionError("no download on collision"))
            ocr = mock.Mock(side_effect=AssertionError("no OCR on collision"))
            returned = resolve_missing_public_ocr(
                URL, root, [CaptionResult(
                    "youtube", "ja", "needs_ocr", video_id="resume_vid",
                )], media_fetcher=downloader, ocr_extractor=ocr,
            )
            self.assertEqual(returned[0].status, "existing_preserved")
            downloader.assert_not_called()
            ocr.assert_not_called()

    def test_bad_language_does_not_claim_regional_chinese_verification(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            saved, base = visual_fixture(root, lang="zh")
            self.assertEqual(
                audit_visual_output(base, provider="youtube",
                                    video_id="resume_vid", language="zh-hant").state,
                "preserved",
            )
            self.assertEqual(
                audit_visual_output(base, provider="youtube",
                                    video_id="resume_vid", language="zh").state,
                "verified",
            )


if __name__ == "__main__":
    unittest.main()
