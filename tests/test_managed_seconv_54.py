"""No-network safety and source-first routing for optional official SeConv 5.4."""
from __future__ import annotations

import hashlib
import io
from pathlib import Path
import stat
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock
import zipfile

from hardsub_extractor import managed_seconv, local_image_subtitles as image


def _fake_archive():
    buffer=io.BytesIO()
    with zipfile.ZipFile(buffer,"w",compression=zipfile.ZIP_DEFLATED) as z:
        z.writestr("seconv.exe",b"MZ-fake-version-executable")
        z.writestr("seconv.dll",b"fake-self-contained-runtime")
        z.writestr("seconv.deps.json",b'{"mock":true}')
    return buffer.getvalue()


class ManagedSeconvContracts(unittest.TestCase):
    def test_pinned_actual_stable_asset_has_fixed_identity(self):
        self.assertEqual(managed_seconv.OFFICIAL_VERSION,"v5.4.0")
        self.assertEqual(managed_seconv.OFFICIAL_ARCHIVE_BYTES,43026089)
        self.assertEqual(managed_seconv.OFFICIAL_ARCHIVE_SHA256,
            "aea19664f460fcfb84e8ba8eca72419091818c6005d4e063d9950c41ea81062d")
        self.assertEqual(managed_seconv.OFFICIAL_URL,
            "https://github.com/SubtitleEdit/subtitleedit/releases/download/v5.4.0/SeConv-Windows-x64.zip")

    def test_without_consent_no_network_or_cache(self):
        with tempfile.TemporaryDirectory() as td:
            dest=Path(td)/"cache"
            with mock.patch.object(managed_seconv,"_download") as download:
                with self.assertRaises(PermissionError):
                    managed_seconv.ensure_official_seconv_54(
                        consent=False,cache_base=dest)
                download.assert_not_called()
            self.assertFalse(dest.exists())

    def test_verified_staged_published_and_second_use_reverifies_all_bytes(self):
        archive=_fake_archive()
        with tempfile.TemporaryDirectory() as td:
            root=Path(td)/"cache"
            with (
                mock.patch.object(managed_seconv,"OFFICIAL_ARCHIVE_BYTES",len(archive)),
                mock.patch.object(managed_seconv,"OFFICIAL_ARCHIVE_SHA256",hashlib.sha256(archive).hexdigest()),
                mock.patch.object(managed_seconv,"_download",
                    side_effect=lambda p,cb:p.write_bytes(archive)) as download,
                mock.patch.object(managed_seconv,"_smoke") as smoke,
            ):
                a=managed_seconv.ensure_official_seconv_54(consent=True,cache_base=root)
                b=managed_seconv.ensure_official_seconv_54(consent=True,cache_base=root)
                self.assertEqual(a,b)
                self.assertEqual(download.call_count,1)
                self.assertEqual(smoke.call_count,2)
                self.assertEqual(a.read_bytes(),b"MZ-fake-version-executable")
                self.assertTrue((a.parent/"seconv.dll").is_file())
                self.assertEqual(list(root.glob(".seconv-new-*")),[])
                (a.parent/"seconv.dll").write_bytes(b"modified")
                with self.assertRaises(managed_seconv.OfficialSeconvError):
                    managed_seconv.ensure_official_seconv_54(consent=True,cache_base=root)
                self.assertEqual(download.call_count,1)
                self.assertEqual((a.parent/"seconv.dll").read_bytes(),b"modified")

    def test_bad_release_digest_not_published_and_no_smoke(self):
        data=_fake_archive()
        with tempfile.TemporaryDirectory() as td:
            root=Path(td)/"cache"
            with (
                mock.patch.object(managed_seconv,"OFFICIAL_ARCHIVE_BYTES",len(data)),
                mock.patch.object(managed_seconv,"_download",
                    side_effect=lambda path,cb:path.write_bytes(data)),
                mock.patch.object(managed_seconv,"_smoke") as smoke,
            ):
                with self.assertRaises(managed_seconv.OfficialSeconvError):
                    managed_seconv.ensure_official_seconv_54(consent=True,cache_base=root)
                smoke.assert_not_called()
                self.assertFalse((root/"SeConv-v5.4.0").exists())
                self.assertEqual(list(root.glob(".seconv-new-*")),[])

    def test_archive_path_escape_symlink_and_case_collision_blocked(self):
        cases=[
            {"../outside.exe":b"x","seconv.exe":b"x"},
            {"seconv.exe":b"x","SeConv.EXE":b"x"},
        ]
        for members in cases:
            with self.subTest(members=list(members)):
                data=io.BytesIO()
                with zipfile.ZipFile(data,"w") as z:
                    for name,blob in members.items():
                        z.writestr(name,blob)
                with zipfile.ZipFile(io.BytesIO(data.getvalue())) as z:
                    with self.assertRaises(managed_seconv.OfficialSeconvError):
                        managed_seconv._members(z)
        stream=io.BytesIO()
        with zipfile.ZipFile(stream,"w") as z:
            entry=zipfile.ZipInfo("seconv.exe")
            entry.create_system=3
            entry.external_attr=(stat.S_IFLNK|0o777)<<16
            z.writestr(entry,"other")
        with zipfile.ZipFile(io.BytesIO(stream.getvalue())) as z:
            with self.assertRaises(managed_seconv.OfficialSeconvError):
                managed_seconv._members(z)

    def test_matching_original_text_does_not_call_seconv_candidate(self):
        from hardsub_extractor import cli
        result=SimpleNamespace(
            mode="local-embedded-text",cue_count=1,
            provenance_path=Path("source.json"),
            srt_path=Path("original.srt"),ass_path=Path("original.ass"),
        )
        with (
            mock.patch("sys.argv",["hard-sub-extract","sample.mkv","--seconv-official-54"]),
            mock.patch.object(cli,"acquire_local_embedded_text",return_value=result),
            mock.patch.object(cli,"acquire_local_embedded_image") as embedded,
            mock.patch("hardsub_extractor.managed_seconv.ensure_official_seconv_54") as managed,
            mock.patch("builtins.print"),
        ):
            cli.main()
            embedded.assert_not_called()
            managed.assert_not_called()

    def test_no_matching_bitmap_track_never_downloads_candidate(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td)
            video=root/"video.mkv"
            video.write_bytes(b"local-original")
            factory=mock.Mock(return_value=root/"seconv.exe")
            with (
                mock.patch.object(image,"_converter_executable",return_value=None),
                mock.patch.object(image,"_track_candidates",return_value=[]),
            ):
                self.assertIsNone(image.acquire_local_embedded_image(
                    video,root/"output",language="en",
                    ocr_factory=lambda:object(),converter_factory=factory))
            factory.assert_not_called()
            self.assertEqual(video.read_bytes(),b"local-original")

    def test_candidate_requested_only_after_actual_bitmap_track(self):
        from hardsub_extractor.matroska_track_headers import MatroskaImageTrack
        with tempfile.TemporaryDirectory() as td:
            root=Path(td)
            video=root/"video.mkv"
            video.write_bytes(b"original")
            binary=root/"candidate-seconv.exe"
            binary.write_bytes(b"private-mock-executable")
            factory=mock.Mock(return_value=binary)
            track=MatroskaImageTrack(track_number=2,codec="hdmv_pgs_subtitle",
                                    language="eng",default=True,forced=False)
            with (
                mock.patch.object(image,"_converter_executable",return_value=None),
                mock.patch.object(image,"_track_candidates",return_value=[track]),
                mock.patch.object(image,"_invoke",side_effect=RuntimeError("mock stop before publish")) as invoke,
            ):
                with self.assertRaisesRegex(RuntimeError,"mock stop"):
                    image.acquire_local_embedded_image(
                        video,root/"output",language="en",
                        ocr_factory=lambda:object(),converter_factory=factory)
                factory.assert_called_once()
                self.assertEqual(invoke.call_args.args[0][0],str(binary))
                self.assertEqual(invoke.call_args.args[0][2],"bdnxml")
            self.assertEqual(video.read_bytes(),b"original")
            self.assertFalse((root/"output.srt").exists())


if __name__=="__main__":
    unittest.main()
