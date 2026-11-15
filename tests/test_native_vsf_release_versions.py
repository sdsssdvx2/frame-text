"""Offline V6.10.7 side-by-side official binary release and consent contracts.

No GitHub release is fetched or executed by this test; all bytes are synthetic.
The default V6.10.5 path must remain exactly compatible with historical tests.
"""
from __future__ import annotations

import hashlib
import io
from pathlib import Path
import tempfile
import unittest
from unittest import mock
import zipfile

from hardsub_extractor import native_vsf_assets as assets


def fixture(label: bytes):
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w", compression=zipfile.ZIP_DEFLATED) as z:
        z.writestr("VideoSubFinderCli.exe", b"MZ-synthetic-" + label)
        z.writestr("support.dll", b"support-" + label)
        z.writestr("LICENSE.txt", b"GPL-2.0 fixture")
    return stream.getvalue()


class PinnedOfficialReleases(unittest.TestCase):
    def test_known_official_6107_from_frozen_release_digest(self):
        url, digest = assets._release_details("v6.10.7")
        self.assertEqual(url, "https://github.com/eritpchy/videosubfinder-cli/"
                              "releases/download/v6.10.7/videosubfinder-cli-windows-x64.zip")
        self.assertEqual(digest,
                         "a8036cada008fa24c88fa4d0b45aaf2eaad584d8c12cfa8b60d7a27496e372ba")
        self.assertEqual(assets.OFFICIAL_DEFAULT_VERSION, "v6.10.5")
        self.assertEqual(assets.OFFICIAL_CANDIDATE_VERSION, "v6.10.7")

    def test_disallowed_selector_cannot_touch_filesystem_or_network(self):
        for value in ("latest", "v6.10.8", "../v6.10.7", "", None, 6107):
            with self.subTest(value=value), tempfile.TemporaryDirectory() as td:
                root = Path(td) / "cache"
                with mock.patch.object(assets, "download_verified_archive") as download:
                    with self.assertRaises(assets.OfficialVsfAssetError):
                        assets.ensure_official_vsf(
                            consent=True, release_version=value, cache_base=root)
                    download.assert_not_called()
                self.assertFalse(root.exists())

    def test_default_and_candidate_are_separate_and_sha_checked(self):
        old, new = fixture(b"6105"), fixture(b"6107")
        with tempfile.TemporaryDirectory() as td:
            root = Path(td) / "cache"
            observed = []

            def download(path, cancel, **options):
                observed.append(options)
                path.write_bytes(new if options else old)

            with (
                mock.patch.object(assets, "OFFICIAL_ARCHIVE_SHA256",
                                  hashlib.sha256(old).hexdigest()),
                mock.patch.dict(assets.PINNED_OFFICIAL_VERSIONS,
                                {"v6.10.7": hashlib.sha256(new).hexdigest()}),
                mock.patch.object(assets, "download_verified_archive",
                                  side_effect=download),
            ):
                earlier, earlier_sha = assets.ensure_official_vsf(
                    consent=True, cache_base=root)
                candidate, candidate_sha = assets.ensure_official_vsf(
                    consent=True, cache_base=root, release_version="v6.10.7")
                self.assertNotEqual(earlier, candidate)
                self.assertIn("VideoSubFinderCLI-v6.10.5", str(earlier))
                self.assertIn("VideoSubFinderCLI-v6.10.7", str(candidate))
                self.assertEqual(earlier_sha,
                                 hashlib.sha256(b"MZ-synthetic-6105").hexdigest())
                self.assertEqual(candidate_sha,
                                 hashlib.sha256(b"MZ-synthetic-6107").hexdigest())
                # Reuse exact candidate files: no second network activity.
                second, second_sha = assets.ensure_official_vsf(
                    consent=True, cache_base=root, release_version="v6.10.7")
                self.assertEqual((second, second_sha), (candidate, candidate_sha))
                self.assertEqual(len(observed), 2)
                self.assertEqual(observed[0], {})
                self.assertEqual(observed[1]["expected_sha256"],
                                 hashlib.sha256(new).hexdigest())
                # Candidate ZIP corruption never touches verified previous release.
                (candidate.parent / "support.dll").write_bytes(b"corrupted")
                with self.assertRaises(assets.OfficialVsfAssetError):
                    assets.ensure_official_vsf(
                        consent=True, cache_base=root, release_version="v6.10.7")
                old_again, old_again_sha = assets.ensure_official_vsf(
                    consent=True, cache_base=root)
                self.assertEqual((old_again, old_again_sha),
                                 (earlier, earlier_sha))
                self.assertEqual(len(observed), 2)
                self.assertEqual(list(root.glob(".vsf-new-*")), [])

    def test_cli_explicit_candidate_version_is_forwarded_only_after_local_preflight(self):
        from types import SimpleNamespace
        from hardsub_extractor import cli
        with tempfile.TemporaryDirectory() as td:
            video = Path(td) / "local-video.mkv"
            video.write_bytes(b"synthetic LOCAL file; no media decoder invoked")
            result = SimpleNamespace(
                report=SimpleNamespace(
                    detected_intervals=1, recognized_intervals=1,
                    missed_intervals=0, ocr_calls=1,
                    srt_path=Path(td) / "candidate.srt",
                    ass_path=Path(td) / "candidate.ass",
                ),
            )
            with (
                mock.patch("sys.argv", [
                    "hard-sub-extract", str(video), "--hard-only",
                    "--native-vsf-official", "--native-vsf-official-version",
                    "v6.10.7",
                ]),
                mock.patch(
                    "hardsub_extractor.native_vsf_assets.ensure_official_vsf",
                    return_value=(Path("pinned-6107.exe"), "a" * 64),
                ) as fetch,
                mock.patch(
                    "hardsub_extractor.native_vsf_timing.extract_with_native_vsf",
                    return_value=result,
                ) as extractor,
                mock.patch.object(cli, "_ocr"),
                mock.patch("builtins.print"),
            ):
                cli.main()
            fetch.assert_called_once_with(
                consent=True, release_version="v6.10.7")
            self.assertEqual(extractor.call_args.kwargs["executable"],
                             Path("pinned-6107.exe"))

    def test_cli_candidate_without_official_consent_is_rejected(self):
        from hardsub_extractor import cli
        with tempfile.TemporaryDirectory() as td:
            source = Path(td) / "file.mkv"
            source.write_bytes(b"synthetic")
            with (
                mock.patch("sys.argv", [
                    "hard-sub-extract", str(source),
                    "--native-vsf-official-version", "v6.10.7",
                ]),
                mock.patch(
                    "hardsub_extractor.native_vsf_assets.ensure_official_vsf",
                ) as fetch,
            ):
                with self.assertRaises(SystemExit):
                    cli.main()
                fetch.assert_not_called()

    def test_candidate_bad_sha_fails_before_cache_promotion(self):
        new = fixture(b"untrusted")
        with tempfile.TemporaryDirectory() as td:
            root = Path(td) / "cache"
            with mock.patch.object(
                assets, "download_verified_archive",
                side_effect=lambda path, cb, **options: path.write_bytes(new),
            ):
                with self.assertRaises(assets.OfficialVsfAssetError):
                    assets.ensure_official_vsf(
                        consent=True, cache_base=root, release_version="v6.10.7")
            self.assertFalse((root / "VideoSubFinderCLI-v6.10.7").exists())
            self.assertEqual(list(root.glob(".vsf-new-*")), [])

    def test_candidate_requires_user_consent_before_cache_or_network(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td) / "cache"
            with mock.patch.object(assets, "download_verified_archive") as download:
                with self.assertRaises(PermissionError):
                    assets.ensure_official_vsf(
                        consent=False, cache_base=root, release_version="v6.10.7")
                download.assert_not_called()
            self.assertFalse(root.exists())


if __name__ == "__main__":
    unittest.main()
