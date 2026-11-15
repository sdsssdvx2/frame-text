"""Entirely offline official free native-tool cache safety tests."""
from __future__ import annotations

import hashlib
import io
from pathlib import Path
import stat
import tempfile
import unittest
from unittest import mock
import zipfile

from hardsub_extractor import native_vsf_assets as assets


def make_zip(entries=None):
    mem = io.BytesIO()
    with zipfile.ZipFile(mem, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for name, data in (entries or {
            "VideoSubFinderCli.exe": b"official-FAKE-test-only-binary",
            "support.dll": b"read-only-test-runtime",
            "LICENSE.txt": b"GPL-2.0 fixture license",
        }).items():
            z.writestr(name, data)
    return mem.getvalue()


class NativeVsfAssetTests(unittest.TestCase):
    def test_no_consent_no_directory_or_network(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td) / "cache"
            with mock.patch.object(assets, "download_verified_archive") as dl:
                with self.assertRaises(PermissionError):
                    assets.ensure_official_vsf(consent=False, cache_base=root)
                dl.assert_not_called()
            self.assertFalse(root.exists())

    def test_official_release_is_exact_https_and_fixed_SHA(self):
        self.assertEqual(
            assets.OFFICIAL_URL,
            "https://github.com/eritpchy/videosubfinder-cli/releases/"
            "download/v6.10.5/videosubfinder-cli-windows-x64.zip",
        )
        self.assertEqual(assets.OFFICIAL_ARCHIVE_SHA256,
                         "60c17f0fa7dde94a51adb468898a7ed4ac33b250fd68af3d736b19989942cb35")

    def test_verified_once_then_cached_without_network(self):
        contents = make_zip()
        with tempfile.TemporaryDirectory() as td:
            root = Path(td) / "cache"
            with mock.patch.object(assets, "OFFICIAL_ARCHIVE_SHA256",
                                   hashlib.sha256(contents).hexdigest()), mock.patch.object(
                assets, "download_verified_archive", side_effect=lambda path, cb: path.write_bytes(contents),
            ) as dl:
                exe, digest = assets.ensure_official_vsf(consent=True, cache_base=root)
                self.assertTrue(exe.is_file())
                self.assertEqual(hashlib.sha256(exe.read_bytes()).hexdigest(), digest)
                self.assertTrue((exe.parent / "support.dll").is_file())
                other, other_digest = assets.ensure_official_vsf(
                    consent=True, cache_base=root,
                )
                self.assertEqual((exe, digest), (other, other_digest))
                self.assertEqual(dl.call_count, 1)
                self.assertEqual(list(root.glob(".vsf-new-*")), [])

    def test_archive_digest_failure_does_not_publish(self):
        contents = make_zip()
        with tempfile.TemporaryDirectory() as td:
            root = Path(td) / "cache"
            with mock.patch.object(assets, "download_verified_archive",
                                   side_effect=lambda path, cb: path.write_bytes(contents)):
                with self.assertRaises(assets.OfficialVsfAssetError):
                    assets.ensure_official_vsf(consent=True, cache_base=root)
            self.assertFalse((root / "VideoSubFinderCLI-v6.10.5").exists())
            self.assertEqual(list(root.glob(".vsf-new-*")), [])

    def test_corrupted_cached_dll_fails_closed_with_no_repair(self):
        contents = make_zip()
        with tempfile.TemporaryDirectory() as td:
            root = Path(td) / "cache"
            with mock.patch.object(assets, "OFFICIAL_ARCHIVE_SHA256",
                                   hashlib.sha256(contents).hexdigest()), mock.patch.object(
                assets, "download_verified_archive", side_effect=lambda path, cb: path.write_bytes(contents),
            ) as dl:
                exe, digest = assets.ensure_official_vsf(consent=True, cache_base=root)
                support = exe.parent / "support.dll"
                support.write_bytes(b"tampered-but-still-present")
                with self.assertRaises(assets.OfficialVsfAssetError):
                    assets.ensure_official_vsf(consent=True, cache_base=root)
                self.assertEqual(dl.call_count, 1)
                self.assertEqual(support.read_bytes(), b"tampered-but-still-present")

    def test_unsafe_archive_entry_and_symlink_rejected(self):
        for entry in ("../escape", "/tmp/escape", "a\\..\\file", "C:/other.exe", "a//bad", "CON", "folder/../evil"):
            with self.subTest(entry=entry):
                with self.assertRaises(assets.OfficialVsfAssetError):
                    assets._safe_member(entry)
        mem = io.BytesIO()
        with zipfile.ZipFile(mem, "w") as z:
            s = zipfile.ZipInfo("VideoSubFinderCli.exe")
            s.create_system = 3
            s.external_attr = (stat.S_IFLNK | 0o777) << 16
            z.writestr(s, "other/path")
        with zipfile.ZipFile(io.BytesIO(mem.getvalue())) as z:
            with self.assertRaises(assets.OfficialVsfAssetError):
                assets._archive_members(z)

    def test_duplicate_casing_and_extra_dll_rejected(self):
        with zipfile.ZipFile(io.BytesIO(make_zip({
            "VideoSubFinderCli.exe": b"one",
            "videosubfindercli.EXE": b"two",
        }))) as z:
            with self.assertRaises(assets.OfficialVsfAssetError):
                assets._archive_members(z)

    def test_extra_file_in_private_cache_rejected_without_erasing(self):
        blob = make_zip()
        with tempfile.TemporaryDirectory() as td:
            root = Path(td) / "cache"
            with mock.patch.object(
                assets, "OFFICIAL_ARCHIVE_SHA256",
                hashlib.sha256(blob).hexdigest(),
            ), mock.patch.object(
                assets, "download_verified_archive",
                side_effect=lambda path, cb: path.write_bytes(blob),
            ):
                exe, _ = assets.ensure_official_vsf(consent=True, cache_base=root)
                extra = exe.parent / "unknown-private.txt"
                extra.write_bytes(b"preserve me")
                with self.assertRaises(assets.OfficialVsfAssetError):
                    assets.ensure_official_vsf(consent=True, cache_base=root)
                self.assertEqual(extra.read_bytes(), b"preserve me")

    def test_consent_cancel_preserves_cached_data(self):
        class Cancel(Exception): pass
        fixture = make_zip()
        with tempfile.TemporaryDirectory() as td:
            root = Path(td) / "cache"
            with mock.patch.object(assets, "OFFICIAL_ARCHIVE_SHA256",
                                   hashlib.sha256(fixture).hexdigest()), mock.patch.object(
                assets, "download_verified_archive",
                side_effect=lambda path, cb: path.write_bytes(fixture),
            ):
                exe, _ = assets.ensure_official_vsf(consent=True, cache_base=root)
                with self.assertRaises(Cancel):
                    assets.ensure_official_vsf(
                        consent=True, cache_base=root,
                        check_cancelled=lambda: (_ for _ in ()).throw(Cancel()),
                    )
                self.assertTrue(exe.is_file())

    def test_posix_race_never_overwrites_concurrent_empty_folder(self):
        if assets.os.name != "posix":
            self.skipTest("POSIX exclusive publish only")
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            stage, target = root / "stage", root / "target"
            stage.mkdir()
            target.mkdir()
            (stage / "marker").write_bytes(b"untouched")
            with self.assertRaises(assets.OfficialVsfAssetError):
                assets._publish_exclusive(stage, target)
            self.assertEqual((stage / "marker").read_bytes(), b"untouched")
            self.assertTrue(target.is_dir())

    def test_unsafe_trailing_dot_and_space_zip_names_rejected(self):
        for value in ("foo./VideoSubFinderCli.exe", "foo /file.dll",
                      "NUL.txt", "LPT1.cfg"):
            with self.subTest(value=value), self.assertRaises(assets.OfficialVsfAssetError):
                assets._safe_member(value)

    def test_existing_non_directory_cache_not_overwritten(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td) / "cache"
            root.mkdir()
            target = root / "VideoSubFinderCLI-v6.10.5"
            target.write_bytes(b"unrelated private data")
            with self.assertRaises(assets.OfficialVsfAssetError):
                assets.ensure_official_vsf(consent=True, cache_base=root)
            self.assertEqual(target.read_bytes(), b"unrelated private data")


if __name__ == "__main__":
    unittest.main()
