"""No-network contracts for gated public-video OCR of absent soft tracks."""
from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from types import SimpleNamespace
from unittest import mock

from hardsub_extractor.extraction_progress import ExtractionCancelled, ExtractionMonitor
from hardsub_extractor.public_video_ocr import (
    MAX_PUBLIC_VIDEO_BYTES, PublicVideoOCRError,
    _probe_local_video_for_ocr, _extract_ocr, download_public_video_for_ocr,
    resolve_missing_public_ocr,
)
from hardsub_extractor.subtitle_acquisition import CaptionResult
from hardsub_extractor.subtitle_proxy import CaptionProxyRoute

YOUTUBE = "https://www.youtube.com/watch?v=example"
WETV = "https://wetv.vip/en/play/example"
PROXY = "socks5h://127.0.0.1:17920"


def absent(lang="ko", site="youtube", route="direct", id="sample"):
    return CaptionResult(
        site, lang, "needs_ocr", video_id=id, connection_route=route,
    )


class FakeVideoAndOCR:
    def __init__(self, fail=False):
        self.downloads = []
        self.ocr = []
        self.fail = fail
        self.temp_files = []

    def fetch(self, url, folder, **kwargs):
        self.downloads.append((url, kwargs.get("proxy_url")))
        if self.fail:
            raise PublicVideoOCRError("mock inaccessible video")
        file = folder / "video.mp4"
        file.write_bytes(b"synthetic-media-content-no-network")
        self.temp_files.append(file)
        return file

    def extract(self, video, base, lang, monitor):
        self.ocr.append((lang, video))
        self_arg = SimpleNamespace(
            srt_path=base.with_suffix(".srt"),
            ass_path=base.with_suffix(".ass"),
            cue_count=3,
        )
        self_arg.srt_path.write_bytes(b"unit-test-SRT")
        self_arg.ass_path.write_bytes(b"unit-test-ASS")
        return self_arg


class PublicVisualFallbackTests(unittest.TestCase):
    def run_pipeline(self, td, statuses, **kwargs):
        fake = FakeVideoAndOCR()
        outputs = resolve_missing_public_ocr(
            YOUTUBE, Path(td), statuses,
            media_fetcher=fake.fetch,
            ocr_extractor=fake.extract,
            **kwargs,
        )
        return outputs, fake

    def test_downloaded_manual_caption_never_fetches_video_or_loads_ocr(self):
        with tempfile.TemporaryDirectory() as td:
            original = CaptionResult("youtube", "ko", "downloaded", cue_count=9)
            results, fake = self.run_pipeline(td, [original])
            self.assertEqual(results, [original])
            self.assertFalse(fake.downloads)
            self.assertFalse(fake.ocr)
            self.assertFalse(list(Path(td).iterdir()))

    def test_missing_manual_and_auto_triggers_only_target_language(self):
        with tempfile.TemporaryDirectory() as td:
            original = CaptionResult("youtube", "ko", "downloaded", cue_count=9)
            results, fake = self.run_pipeline(td, [
                original, absent("en", id="video-one"),
            ])
            self.assertEqual(results[0], original)
            self.assertEqual(results[1].status, "hard_ocr")
            self.assertEqual(results[1].cue_count, 3)
            self.assertEqual(results[1].track_origin, "visual_ocr")
            self.assertEqual([x[0] for x in fake.ocr], ["en"])
            self.assertEqual(len(fake.downloads), 1)
            self.assertTrue(results[1].srt.is_file())
            self.assertTrue(results[1].ass.is_file())
            report = json.loads(results[1].manifest.read_text("utf-8"))
            self.assertTrue(report["ocr_applied"])
            self.assertFalse(report["original_text_verified"])
            self.assertFalse(report["machine_translation_applied"])
            self.assertFalse(report["temporary_video_retained"])
            self.assertNotIn(YOUTUBE, results[1].manifest.read_text("utf-8"))
            self.assertTrue(all(not path.exists() for path in fake.temp_files))

    def test_two_missing_languages_share_one_video_download(self):
        with tempfile.TemporaryDirectory() as td:
            results, fake = self.run_pipeline(td, [
                absent("ja"), absent("en"), absent("zh"),
            ])
            self.assertEqual([x.status for x in results], ["hard_ocr"] * 3)
            self.assertEqual([x[0] for x in fake.ocr], ["ja", "en", "zh"])
            self.assertEqual(len(fake.downloads), 1)
            self.assertTrue(all(x.srt.is_file() for x in results))

    def test_regional_chinese_scripts_never_mislabelled_from_general_model(self):
        with tempfile.TemporaryDirectory() as td:
            results, fake = self.run_pipeline(td, [
                absent("zh-hans"), absent("zh-hant"),
            ])
            self.assertEqual([x.status for x in results], ["needs_ocr"] * 2)
            self.assertFalse(fake.downloads)
            self.assertFalse(fake.ocr)

    def test_failed_manual_track_does_not_imply_hard_sub(self):
        with tempfile.TemporaryDirectory() as td:
            results, fake = self.run_pipeline(td, [
                replace(absent(), status="unavailable"),
                replace(absent("en"), status="proxy_unavailable"),
            ])
            self.assertFalse(fake.downloads)
            self.assertEqual([x.status for x in results], [
                "unavailable", "proxy_unavailable",
            ])

    def test_mismatched_site_fails_before_any_download(self):
        with tempfile.TemporaryDirectory() as td:
            fake = FakeVideoAndOCR()
            with self.assertRaisesRegex(ValueError, "provider"):
                resolve_missing_public_ocr(
                    YOUTUBE, Path(td), [absent(site="wetv")],
                    media_fetcher=fake.fetch,
                )
            self.assertFalse(fake.downloads)

    def test_insecure_or_loopback_url_rejected_before_any_download(self):
        with tempfile.TemporaryDirectory() as td:
            for source in ("http://www.youtube.com/watch?v=x",
                           "https://127.0.0.1/video",
                           "https://youtube.com.evil.example/x"):
                with self.subTest(source=source):
                    with self.assertRaises(ValueError):
                        resolve_missing_public_ocr(
                            source, Path(td), [absent()],
                        )

    def test_existing_output_is_preserved_without_download(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            previous = root / "youtube-sample-ko-hard-ocr.srt"
            previous.write_bytes(b"DO_NOT_ERASE")
            results, fake = self.run_pipeline(td, [absent()])
            self.assertEqual(results[0].status, "existing_preserved")
            self.assertFalse(fake.downloads)
            self.assertEqual(previous.read_bytes(), b"DO_NOT_ERASE")

    def test_existing_output_with_no_skip_fails_closed_before_download(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root / "youtube-sample-ko-hard-ocr.source.json").write_bytes(b"preserve")
            fake = FakeVideoAndOCR()
            with self.assertRaises(FileExistsError):
                resolve_missing_public_ocr(
                    YOUTUBE, root, [absent()],
                    media_fetcher=fake.fetch, skip_existing=False,
                )
            self.assertFalse(fake.downloads)

    def test_missing_stream_public_media_denial_is_not_falsely_downloaded(self):
        with tempfile.TemporaryDirectory() as td:
            fake = FakeVideoAndOCR(fail=True)
            result = resolve_missing_public_ocr(
                YOUTUBE, Path(td), [absent()],
                media_fetcher=fake.fetch, ocr_extractor=fake.extract,
            )
            self.assertEqual(result[0].status, "ocr_unavailable")
            self.assertEqual(len(fake.downloads), 1)
            self.assertFalse(fake.ocr)
            self.assertFalse(list(Path(td).iterdir()))

    def test_required_proxy_never_attempts_direct(self):
        with tempfile.TemporaryDirectory() as td:
            fake = FakeVideoAndOCR()
            route = CaptionProxyRoute("required", PROXY)
            output = resolve_missing_public_ocr(
                YOUTUBE, td, [absent(route="proxy")],
                proxy_route=route,
                media_fetcher=fake.fetch, ocr_extractor=fake.extract,
            )
            self.assertEqual(output[0].status, "hard_ocr")
            self.assertEqual(fake.downloads, [(YOUTUBE, PROXY)])
            self.assertEqual(output[0].connection_route, "proxy")

    def test_required_proxy_forbids_direct_claim_even_with_absent_subtitles(self):
        with tempfile.TemporaryDirectory() as td:
            fake = FakeVideoAndOCR()
            with self.assertRaisesRegex(ValueError, "forbids direct"):
                resolve_missing_public_ocr(
                    YOUTUBE, td, [absent(route="direct")],
                    proxy_route=CaptionProxyRoute("required", PROXY),
                    media_fetcher=fake.fetch, ocr_extractor=fake.extract,
                )
            self.assertFalse(fake.downloads)

    def test_fallback_proxy_retries_video_once_after_direct_denial(self):
        with tempfile.TemporaryDirectory() as td:
            fake = FakeVideoAndOCR()
            def retry(url, folder, **kwargs):
                fake.downloads.append((url, kwargs["proxy_url"]))
                if kwargs["proxy_url"] is None:
                    (folder / "video.mp4.part").write_bytes(b"temp-fragment")
                    raise PublicVideoOCRError("direct unavailable")
                file = folder / "video.mp4"
                file.write_bytes(b"synthetic-video")
                fake.temp_files.append(file)
                return file

            result = resolve_missing_public_ocr(
                YOUTUBE, td, [absent()],
                proxy_route=CaptionProxyRoute("fallback", PROXY),
                media_fetcher=retry, ocr_extractor=fake.extract,
            )
            self.assertEqual(fake.downloads, [(YOUTUBE, None), (YOUTUBE, PROXY)])
            self.assertEqual(result[0].status, "hard_ocr")
            self.assertEqual(result[0].connection_route, "proxy")

    def test_direct_route_never_retries_after_media_denial(self):
        with tempfile.TemporaryDirectory() as td:
            fake = FakeVideoAndOCR(fail=True)
            result = resolve_missing_public_ocr(
                YOUTUBE, td, [absent()],
                proxy_route=CaptionProxyRoute(),
                media_fetcher=fake.fetch, ocr_extractor=fake.extract,
            )
            self.assertEqual(result[0].status, "ocr_unavailable")
            self.assertEqual(fake.downloads, [(YOUTUBE, None)])

    def test_cancel_before_video_fetch_propagates_without_output(self):
        with tempfile.TemporaryDirectory() as td:
            fake = FakeVideoAndOCR()
            monitor = ExtractionMonitor(
                is_cancelled=lambda: True, exclusive_outputs=True,
            )
            with self.assertRaises(ExtractionCancelled):
                resolve_missing_public_ocr(
                    YOUTUBE, td, [absent()], monitor=monitor,
                    media_fetcher=fake.fetch, ocr_extractor=fake.extract,
                )
            self.assertFalse(fake.downloads)

    def test_video_without_visible_subtitles_is_not_false_success(self):
        with tempfile.TemporaryDirectory() as td:
            fake = FakeVideoAndOCR()
            def no_text(video, base, lang, monitor):
                raise RuntimeError("인식된 자막이 없습니다. 하드자막이 없습니다.")
            results = resolve_missing_public_ocr(
                YOUTUBE, td, [absent("en"), absent("ko")],
                media_fetcher=fake.fetch, ocr_extractor=no_text,
            )
            self.assertEqual([r.status for r in results], [
                "ocr_unavailable", "ocr_unavailable",
            ])
            self.assertEqual(len(fake.downloads), 1)
            self.assertFalse(list(Path(td).iterdir()))

    def test_one_language_runtime_model_failure_does_not_cancel_rest(self):
        with tempfile.TemporaryDirectory() as td:
            fake = FakeVideoAndOCR()
            def sometimes_fails(media, base, lang, monitor):
                if lang == "ja":
                    raise RuntimeError("OCR model missing PRIVATE-URL-DO-NOT-LOG")
                return fake.extract(media, base, lang, monitor)
            outputs = resolve_missing_public_ocr(
                YOUTUBE, td, [absent("ja"), absent("ko")],
                media_fetcher=fake.fetch, ocr_extractor=sometimes_fails,
            )
            self.assertEqual([x.status for x in outputs], ["ocr_failed", "hard_ocr"])
            self.assertEqual([x[0] for x in fake.ocr], ["ko"])
            self.assertEqual(len(fake.downloads), 1)
            self.assertFalse(list(Path(td).glob(".hse-visual-ocr-*")))
            self.assertTrue(outputs[1].srt.is_file())

    def test_partial_files_from_failed_extractor_are_preserved_and_reported(self):
        with tempfile.TemporaryDirectory() as td:
            fake = FakeVideoAndOCR()
            base = Path(td) / "youtube-sample-ko-hard-ocr.srt"
            def interrupted(media, dest, lang, monitor):
                base.write_bytes(b"UNVERIFIED_OUTPUT_DO_NOT_DELETE")
                raise RuntimeError("decoder exited abnormally")
            with self.assertRaisesRegex(RuntimeError, "decoder"):
                resolve_missing_public_ocr(
                    YOUTUBE, td, [absent("ko")],
                    media_fetcher=fake.fetch, ocr_extractor=interrupted,
                )
            self.assertEqual(base.read_bytes(), b"UNVERIFIED_OUTPUT_DO_NOT_DELETE")
            self.assertFalse(list(Path(td).glob(".hse-visual-ocr-*")))
            self.assertEqual(len(fake.downloads), 1)

    def test_public_ocr_limits_sample_rate_with_real_pipeline_api(self):
        with mock.patch(
            "hardsub_extractor.public_video_ocr._ocr_engine",
            return_value=object(),
        ), mock.patch(
            "hardsub_extractor.pipeline.extract",
            return_value=object(),
        ) as extract:
            monitor = ExtractionMonitor(exclusive_outputs=True)
            result = _extract_ocr(
                Path("synthetic.mp4"), Path("output"), "en", monitor,
            )
        self.assertIsNotNone(result)
        self.assertEqual(extract.call_args.kwargs["sample_fps"], 8.0)
        self.assertTrue(extract.call_args.kwargs["streaming"])
        self.assertIs(extract.call_args.kwargs["monitor"], monitor)

    def test_nonexclusive_monitor_rejected_before_ocr_model_initialization(self):
        with mock.patch("hardsub_extractor.public_video_ocr._ocr_engine") as model:
            with self.assertRaisesRegex(ValueError, "exclusive"):
                _extract_ocr(
                    Path("synthetic.mp4"), Path("output"), "en",
                    ExtractionMonitor(exclusive_outputs=False),
                )
        model.assert_not_called()

    def test_ocr_provenance_failure_removes_only_new_srt_and_ass(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            fake = FakeVideoAndOCR()
            base = root / "youtube-sample-ko-hard-ocr"
            provenance = root / "youtube-sample-ko-hard-ocr.source.json"
            def collide(video, dest, lang, monitor):
                result = fake.extract(video, dest, lang, monitor)
                provenance.write_bytes(b"CONCURRENT_USER_SOURCE")
                return result
            with self.assertRaises(FileExistsError):
                resolve_missing_public_ocr(
                    YOUTUBE, td, [absent()],
                    media_fetcher=fake.fetch, ocr_extractor=collide,
                )
            self.assertEqual(provenance.read_bytes(), b"CONCURRENT_USER_SOURCE")
            self.assertFalse(base.with_suffix(".srt").exists())
            self.assertFalse(base.with_suffix(".ass").exists())


class BoundedDownloaderTests(unittest.TestCase):
    def test_public_download_command_never_includes_cookies_or_plugins(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            process = mock.Mock()
            process.poll.return_value = 0
            process.returncode = 0
            commands = []
            def launch(args, **kwargs):
                commands.append((args, kwargs))
                (root / "video.mp4").write_bytes(b"synthetic-stream")
                return process
            with mock.patch(
                "hardsub_extractor.public_video_ocr.external_popen",
                side_effect=launch,
            ), mock.patch(
                "hardsub_extractor.public_video_ocr._probe_local_video_for_ocr",
            ) as probe:
                result = download_public_video_for_ocr(
                    YOUTUBE, root, executable="pinned-downloader.exe",
                )
            self.assertEqual(result, root / "video.mp4")
            probe.assert_called_once()
            command, opts = commands[0]
            self.assertIn("--ignore-config", command)
            self.assertIn("--no-plugin-dirs", command)
            self.assertIn("--no-allow-unplayable-formats", command)
            self.assertIn("--max-filesize", command)
            self.assertEqual(command[command.index("--proxy") + 1], "")
            self.assertEqual(command[-2:], ["--", YOUTUBE])
            self.assertEqual(opts["stderr"], subprocess.DEVNULL)
            self.assertNotIn("--cookies", command)

    def test_empty_folder_required_before_process_start(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root / "DO_NOT_TOUCH").write_bytes(b"precious")
            with mock.patch("hardsub_extractor.public_video_ocr.external_popen") as p:
                with self.assertRaises(ValueError):
                    download_public_video_for_ocr(YOUTUBE, root, executable="pinned")
            p.assert_not_called()
            self.assertEqual((root / "DO_NOT_TOUCH").read_bytes(), b"precious")

    def test_video_bytes_above_bound_kills_child_process(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            process = mock.Mock()
            process.poll.side_effect = [None, None]
            def launch(*args, **kwargs):
                (root / "video.mp4.part").write_bytes(b"0123456789ABCDEF")
                return process
            with mock.patch(
                "hardsub_extractor.public_video_ocr.external_popen",
                side_effect=launch,
            ):
                with self.assertRaisesRegex(PublicVideoOCRError, "size limit"):
                    download_public_video_for_ocr(
                        YOUTUBE, root, executable="pinned", byte_limit=8,
                    )
            process.kill.assert_called_once()
            process.wait.assert_called_once()

    def test_untrusted_symlink_like_output_refused(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            process = mock.Mock()
            process.poll.return_value = 0
            process.returncode = 0
            def launch(*args, **kwargs):
                (root / "video.mp4").write_bytes(b"video")
                return process
            with mock.patch(
                "hardsub_extractor.public_video_ocr.external_popen",
                side_effect=launch,
            ), mock.patch.object(
                Path, "is_symlink", return_value=True,
            ):
                with self.assertRaisesRegex(PublicVideoOCRError, "unexpected"):
                    download_public_video_for_ocr(YOUTUBE, root, executable="pinned")

    def test_bad_video_probe_rejects_huge_or_nonfinite_duration(self):
        with tempfile.TemporaryDirectory() as td:
            file = Path(td) / "video.mp4"
            file.write_bytes(b"fixture")
            fake = SimpleNamespace(returncode=0, stdout=json.dumps({
                "streams": [{"width": 1920, "height": 1080}],
                "format": {"duration": "nan"},
            }).encode())
            with mock.patch(
                "hardsub_extractor.public_video_ocr.resolve_binary",
                return_value="ffprobe",
            ), mock.patch(
                "hardsub_extractor.public_video_ocr.subprocess.run",
                return_value=fake,
            ):
                with self.assertRaisesRegex(PublicVideoOCRError, "duration"):
                    _probe_local_video_for_ocr(file)

    def test_untrusted_download_url_rejected_before_process_start(self):
        with tempfile.TemporaryDirectory() as td:
            with mock.patch("hardsub_extractor.public_video_ocr.external_popen") as p:
                for url in ("file:///C:/private", "http://www.youtube.com/w",
                            "https://youtube.com.evil.invalid/video"):
                    with self.subTest(url=url), self.assertRaises(ValueError):
                        download_public_video_for_ocr(
                            url, Path(td), executable="pinned",
                        )
            p.assert_not_called()


if __name__ == "__main__":
    unittest.main()
