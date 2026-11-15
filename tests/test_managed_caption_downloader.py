"""Offline tests for first-party, opt-in yt-dlp updating and rollback."""
from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from hardsub_extractor import managed_caption_downloader as update


class _Reply:
    def __init__(self, body: bytes, url: str, status=200):
        self.body, self.url, self.status = io.BytesIO(body), url, status
        self.headers = {"Content-Length": str(len(body))}

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def geturl(self):
        return self.url

    def read(self, size=-1):
        return self.body.read(size)


class _FakeOpener:
    def __init__(self, version: str, binary: bytes, *, digest=None, url=None):
        self.version, self.binary = version, binary
        self.digest = digest if digest is not None else hashlib.sha256(binary).hexdigest()
        self.asset_url = url or (
            "https://github.com/yt-dlp/yt-dlp/releases/download/" + version + "/yt-dlp.exe"
        )
        self.requests = []

    def open(self, request, timeout=0):
        url = request.full_url
        self.requests.append(url)
        if url == update._API:
            doc = {
                "tag_name": self.version, "draft": False, "prerelease": False,
                "assets": [{
                    "name": "yt-dlp.exe", "size": len(self.binary),
                    "state": "uploaded", "digest": "sha256:" + self.digest,
                    "browser_download_url": self.asset_url,
                }],
            }
            return _Reply(json.dumps(doc).encode(), url)
        if url == self.asset_url:
            return _Reply(self.binary, url)
        raise AssertionError("untrusted URL attempted")


class ManagedDownloaderTests(unittest.TestCase):
    def test_no_consent_and_zero_network(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "cache"
            opener = _FakeOpener("2026.11.09", b"MZ" + b"0" * update._MIN_BYTES)
            self.assertEqual(update.refresh(cache_root=root, opener=opener), "disabled")
            self.assertEqual(opener.requests, [])
            self.assertFalse(root.exists())
            self.assertIsNone(update.active_downloader(cache_root=root))

    def test_consent_is_explicit_and_reversible(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "cache"
            update.set_auto_update_enabled(True, cache_root=root)
            self.assertTrue(update.auto_update_enabled(cache_root=root))
            update.set_auto_update_enabled(False, cache_root=root)
            self.assertFalse(update.auto_update_enabled(cache_root=root))
            self.assertFalse((root / "versions").exists())

    def test_official_download_validated_smoked_and_cached(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "cache"
            update.set_auto_update_enabled(True, cache_root=root)
            binary = b"MZ" + b"a" * update._MIN_BYTES
            opener = _FakeOpener("2026.11.09", binary)
            with mock.patch.object(update, "_smoke") as smoke:
                self.assertEqual(
                    update.refresh(cache_root=root, opener=opener), "updated")
                smoke.assert_called_once()
            path = update.active_downloader(cache_root=root)
            self.assertIsNotNone(path)
            self.assertEqual(Path(path).read_bytes(), binary)
            self.assertEqual(len(opener.requests), 2)
            with mock.patch.object(update, "_smoke") as smoke:
                self.assertEqual(
                    update.refresh(cache_root=root, opener=opener), "current")
                smoke.assert_not_called()
            self.assertEqual(len(opener.requests), 3)
            self.assertEqual(
                update.refresh(auto=True, cache_root=root, opener=opener), "not_due")
            self.assertEqual(len(opener.requests), 3)

    def test_non_executable_asset_never_launched(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "cache"
            update.set_auto_update_enabled(True, cache_root=root)
            op = _FakeOpener("2026.11.09", b"ZZ" + b"a" * update._MIN_BYTES)
            with mock.patch.object(update, "_smoke") as smoke:
                with self.assertRaises(update.ManagedDownloaderError):
                    update.refresh(cache_root=root, opener=op)
                smoke.assert_not_called()
            self.assertIsNone(update.active_downloader(cache_root=root))

    def test_bad_archive_digest_never_promoted(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "cache"
            update.set_auto_update_enabled(True, cache_root=root)
            op = _FakeOpener("2026.11.09", b"MZ" + b"0" * update._MIN_BYTES,
                             digest="f" * 64)
            with self.assertRaises(update.ManagedDownloaderError):
                update.refresh(cache_root=root, opener=op)
            self.assertIsNone(update.active_downloader(cache_root=root))
            self.assertEqual(list((root / "versions").glob(".new-*")), [])

    def test_untrusted_download_origin_rejected_before_execution(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "cache"
            update.set_auto_update_enabled(True, cache_root=root)
            op = _FakeOpener("2026.11.09", b"MZ" + b"0" * update._MIN_BYTES,
                             url="https://attacker.example.invalid/yt-dlp.exe")
            with mock.patch.object(update, "_smoke") as smoke:
                with self.assertRaises(update.ManagedDownloaderError):
                    update.refresh(cache_root=root, opener=op)
                smoke.assert_not_called()

    def test_runtime_corruption_falls_back_without_deleting_user_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "cache"
            update.set_auto_update_enabled(True, cache_root=root)
            binary = b"MZ" + b"a" * update._MIN_BYTES
            with mock.patch.object(update, "_smoke"):
                update.refresh(cache_root=root,
                               opener=_FakeOpener("2026.11.09", binary))
            active = Path(update.active_downloader(cache_root=root))
            active.write_bytes(b"MZ" + b"b" * update._MIN_BYTES)
            self.assertIsNone(update.active_downloader(cache_root=root))
            self.assertTrue(active.is_file())
            self.assertEqual(active.stat().st_size, len(binary))

    def test_downloader_path_selection_uses_verified_version_and_bundled_fallback(self):
        from hardsub_extractor.subtitle_acquisition import find_caption_downloader
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "cache"
            bundle = Path(tmp) / "bundle"
            binary = bundle / "bin" / "yt-dlp.exe"
            binary.parent.mkdir(parents=True)
            binary.write_bytes(b"bundled-test-only")
            with mock.patch("hardsub_extractor.managed_caption_downloader._root",
                            return_value=root), mock.patch(
                "hardsub_extractor.subtitle_acquisition.bundle_root",
                return_value=bundle,
            ):
                self.assertEqual(find_caption_downloader(), str(binary))
                update.set_auto_update_enabled(True, cache_root=root)
                with mock.patch.object(update, "_smoke"):
                    update.refresh(cache_root=root, opener=_FakeOpener(
                        "2026.11.09", b"MZ" + b"a" * update._MIN_BYTES))
                active = update.active_downloader(cache_root=root)
                self.assertEqual(find_caption_downloader(), active)
                Path(active).write_bytes(b"MZ" + b"b" * update._MIN_BYTES)
                self.assertEqual(find_caption_downloader(), str(binary))

    def test_disabling_auto_selection_restores_bundled_selection(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "cache"
            update.set_auto_update_enabled(True, cache_root=root)
            with mock.patch.object(update, "_smoke"):
                update.refresh(cache_root=root, opener=_FakeOpener(
                    "2026.11.09", b"MZ" + b"a" * update._MIN_BYTES))
            self.assertIsNotNone(update.active_downloader(cache_root=root))
            update.set_auto_update_enabled(False, cache_root=root)
            self.assertIsNone(update.active_downloader(cache_root=root))

    def test_invalid_state_and_symlink_fail_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "cache"
            root.mkdir()
            (root / "settings.json").write_text('{"auto": "true"}')
            self.assertFalse(update.auto_update_enabled(cache_root=root))
            self.assertIsNone(update.active_downloader(cache_root=root))
            with self.assertRaises(update.ManagedDownloaderError):
                update.refresh(cache_root=root)

    def test_unsafe_version_and_downgrade_denied(self):
        self.assertLess(update._version_key("2026.10.26"), update._version_key("2026.11.09"))
        for version in ("../../evil", "v2026.11.09", "2026.13.01", "2026.11.99"):
            with self.subTest(version=version):
                with self.assertRaises(update.ManagedDownloaderError):
                    update._version_key(version)


if __name__ == "__main__":
    unittest.main()
