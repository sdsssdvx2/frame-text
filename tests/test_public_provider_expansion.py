"""Cross-provider subtitle-first and optional OCR extension, fully offline.

All network, yt-dlp and OCR execution are replaced by explicitly local fakes.
This verifies exactly approved public video URLs, language handling, proxy
routing, no-overwrite, source provenance, and fail-closed hostile URL input.
An offline success does not claim live sites or private media work.
"""
from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace

from hardsub_extractor.subtitle_acquisition import (
    PublicCaptionCollector, CaptionResult, supported_public_site,
)
from hardsub_extractor.subtitle_batch import load_episode_urls
from hardsub_extractor.subtitle_proxy import CaptionProxyRoute, parse_site_proxy_specs, route_for_site
from hardsub_extractor.public_video_ocr import resolve_missing_public_ocr, download_public_video_for_ocr

VIMEO = "https://vimeo.com/76979871"
DAILY = "https://www.dailymotion.com/video/x9ab12c"
EN_VTT = b"WEBVTT\n\n00:00:01.000 --> 00:00:02.500\nEnglish public captions.\n"


class ExplicitPublicSiteTests(unittest.TestCase):
    def test_exact_public_video_profiles_and_embedding_aliases(self):
        aliases = (
            (VIMEO, "vimeo"),
            ("https://www.vimeo.com/76979871", "vimeo"),
            ("https://player.vimeo.com/video/76979871?h=8272103f6e", "vimeo"),
            (DAILY, "dailymotion"),
            ("https://dailymotion.com/embed/video/x9ab12c", "dailymotion"),
            ("https://dai.ly/x9ab12c", "dailymotion"),
        )
        for url, expected in aliases:
            with self.subTest(host=expected, alias=url[:21]):
                self.assertEqual(supported_public_site(url), expected)

    def test_host_prefix_credentials_ports_and_nonspecific_urls_are_denied(self):
        disallowed = (
            "http://vimeo.com/76979871",
            "https://vimeo.com.evil.test/76979871",
            "https://evil.test/vimeo.com/76979871",
            "https://user:pass@vimeo.com/76979871",
            "https://vimeo.com:8080/76979871",
            "https://vimeo.com/manage/videos/76979871",
            "https://player.vimeo.com/video/../../../etc",
            "https://player.vimeo.com/video/%2e%2e%2f",
            "https://vimeo.com/127.0.0.1",
            "https://dailymotion.com.evil.test/video/x9abc123",
            "http://www.dailymotion.com/video/x9abc123",
            "https://dailymotion.com:8443/video/x9abc123",
            "https://dailymotion.com/playlist/x9abc123",
            "https://dailymotion.com/video/%2e%2e",
            "https://dailymotion.com/video/x9abc123/../private",
            "https://user@dai.ly/x9abc123",
            "https://127.0.0.1/video/x9abc123",
            "https://unlisted-video.invalid/watch",
        )
        for url in disallowed:
            with self.subTest(sample=url[:44]):
                self.assertIsNone(supported_public_site(url))

    def test_batch_accepts_mixed_providers_but_rejects_bad_member_atomically(self):
        urls = load_episode_urls([VIMEO, DAILY, VIMEO], None)
        self.assertEqual(urls, [VIMEO, DAILY])
        for bad in ("http://vimeo.com/76979871", "https://unlisted-video.invalid/watch"):
            with self.subTest(sample=bad):
                with self.assertRaisesRegex(ValueError, "approved"):
                    load_episode_urls([VIMEO, bad, DAILY], None)

    def test_proxy_profiles_independently_support_new_provider_domains(self):
        specs = parse_site_proxy_specs(
            required=["vimeo=socks5h://127.0.0.1:17920"],
            fallback=["dailymotion=http://127.0.0.1:8080"],
        )
        self.assertEqual(route_for_site("vimeo", CaptionProxyRoute(), specs).mode, "required")
        self.assertEqual(route_for_site("dailymotion", CaptionProxyRoute(), specs).mode, "fallback")
        self.assertEqual(route_for_site("youtube", CaptionProxyRoute(), specs).mode, "direct")


class CaptionFakeRunner:
    def __init__(self):
        self.commands = []

    def __call__(self, cmd, *, timeout_seconds, max_stdout_bytes):
        self.commands.append(list(cmd))
        if "--dump-single-json" in cmd:
            return json.dumps({
                "id": "video76979871",
                "subtitles": {"en": [{"ext": "vtt"}]},
                "automatic_captions": {},
            }).encode("utf-8")
        if "--write-subs" not in cmd:
            raise AssertionError("collector must not silently switch to video download")
        templated = cmd[cmd.index("--output")+1]
        tag = cmd[cmd.index("--sub-langs")+1]
        if tag != "^en$" or cmd[cmd.index("--sub-format")+1] != "vtt":
            raise AssertionError("wrong language or format")
        target = Path(templated.replace("%(ext)s", "en.vtt"))
        target.write_bytes(EN_VTT)
        return b""


class CrossProviderEndToEndContracts(unittest.TestCase):
    def test_identical_caption_first_path_and_raw_original_for_each_provider(self):
        for url, site in ((VIMEO, "vimeo"), (DAILY, "dailymotion")):
            with self.subTest(site=site), tempfile.TemporaryDirectory() as td:
                root = Path(td)
                runner = CaptionFakeRunner()
                collector = PublicCaptionCollector(
                    "yt-dlp-synthetic", command_runner=runner,
                )
                results = collector.collect(url, root, languages=("en",), skip_existing=True)
                self.assertEqual(len(results), 1)
                result = results[0]
                self.assertEqual(result.status, "downloaded")
                self.assertEqual(result.source, site)
                self.assertEqual(result.track_origin, "manual")
                self.assertEqual(result.cue_count, 1)
                self.assertEqual(result.original.read_bytes(), EN_VTT)
                self.assertTrue(result.srt.is_file())
                self.assertTrue(result.ass.is_file())
                metadata = json.loads(result.manifest.read_text("utf-8"))
                self.assertEqual(metadata["site"], site)
                self.assertEqual(metadata["original_timeline"], "source_video")
                self.assertFalse(metadata["ocr_applied"])
                self.assertEqual(len(runner.commands), 2)
                self.assertTrue(all("--skip-download" in cmd for cmd in runner.commands))
                self.assertTrue(all("--no-plugin-dirs" in cmd for cmd in runner.commands))
                self.assertTrue(all("--proxy" in cmd for cmd in runner.commands))
                self.assertNotIn(url, result.manifest.read_text("utf-8"))
                written = {p: p.read_bytes() for p in root.iterdir()}
                again = collector.collect(url, root, languages=("en",), skip_existing=True)
                self.assertEqual(again[0].status, "existing_verified")
                self.assertEqual({p: p.read_bytes() for p in root.iterdir()}, written)
                # Resume is metadata-only: no second subtitle fetch/download.
                self.assertEqual(len(runner.commands), 3)

    def test_no_matching_caption_remains_needs_ocr_without_video_download(self):
        for url, site in ((VIMEO, "vimeo"), (DAILY, "dailymotion")):
            with self.subTest(site=site), tempfile.TemporaryDirectory() as td:
                class Missing:
                    def __init__(self):
                        self.calls = []
                    def __call__(self, cmd, *, timeout_seconds, max_stdout_bytes):
                        self.calls.append(list(cmd))
                        return json.dumps({
                            "id": "n" + site,
                            "subtitles": {"fr": [{"ext": "vtt"}]},
                            "automatic_captions": {},
                        }).encode()
                runner = Missing()
                collector = PublicCaptionCollector("yt-dlp-fake", command_runner=runner)
                records = collector.collect(url, td, languages=("en",))
                self.assertEqual([r.status for r in records], ["needs_ocr"])
                self.assertEqual(len(runner.calls), 1)
                self.assertFalse(list(Path(td).iterdir()))

    def test_video_ocr_opt_in_reuses_single_temporary_download_and_language(self):
        for url, site in ((VIMEO, "vimeo"), (DAILY, "dailymotion")):
            with self.subTest(site=site), tempfile.TemporaryDirectory() as td:
                root = Path(td)
                fetched = []
                def fetch(url_, folder, **kwargs):
                    fetched.append(kwargs.get("proxy_url"))
                    fake = folder / "video.mp4"
                    fake.write_bytes(b"dummy offline media bytes")
                    return fake
                extracted = []
                def ocr(media, out, language, monitor):
                    extracted.append(language)
                    target_srt = out.with_suffix(".srt")
                    target_ass = out.with_suffix(".ass")
                    target_srt.write_text(
                        "1\n00:00:01,000 --> 00:00:02,000\nHello\n",
                        encoding="utf-8",
                    )
                    target_ass.write_text(
                        "[Script Info]\nScriptType: v4.00+\n\n[Events]\n"
                        "Format: Layer,Start,End,Style,Name,MarginL,MarginR,MarginV,Effect,Text\n"
                        "Dialogue: 0,0:00:01.00,0:00:02.00,Default,,0,0,0,,Hello\n",
                        encoding="utf-8",
                    )
                    return SimpleNamespace(srt_path=target_srt, ass_path=target_ass, cue_count=1)
                source = CaptionResult(site, "en", "needs_ocr", video_id="test123")
                rows = resolve_missing_public_ocr(
                    url, root, [source],
                    media_fetcher=fetch, ocr_extractor=ocr,
                )
                self.assertEqual(rows[0].status, "hard_ocr")
                self.assertEqual(rows[0].source, site)
                self.assertEqual(fetched, [None])
                self.assertEqual(extracted, ["en"])
                self.assertTrue(rows[0].manifest.is_file())
                self.assertFalse(list(root.glob("**/video.mp4")))
                self.assertEqual(
                    resolve_missing_public_ocr(
                        url, root, [source],
                        media_fetcher=fetch, ocr_extractor=ocr,
                    )[0].status, "existing_verified",
                )
                self.assertEqual(len(fetched), 1)

    def test_ocr_source_mismatch_denied_before_local_download(self):
        with tempfile.TemporaryDirectory() as td:
            wrong = CaptionResult("dailymotion", "en", "needs_ocr", video_id="x")
            def never(*args, **kwargs):
                raise AssertionError("mismatched site must fail before network")
            with self.assertRaisesRegex(ValueError, "provider"):
                resolve_missing_public_ocr(
                    VIMEO, td, [wrong], media_fetcher=never, ocr_extractor=never,
                )
