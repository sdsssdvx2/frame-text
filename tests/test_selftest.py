import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch

from hardsub_extractor.selftest import run_selftest


class SelfTestTests(unittest.TestCase):
    def test_soft_only_accepts_no_models_when_binaries_work(self):
        def fake_resolve(name: str) -> str:
            return f"fake-{name}"

        def fake_version(binary: str) -> str:
            return f"{binary} test-version"

        with patch("hardsub_extractor.selftest.resolve_binary", side_effect=fake_resolve), patch(
            "hardsub_extractor.selftest._version", side_effect=fake_version
        ):
            r = run_selftest(require_models=False)
            self.assertTrue(r.ok, r.to_json())


    def test_free_portable_bitmap_tool_frozen_version_and_license_gate(self):
        # Source-only small contract. A full EXE still needs an independent
        # real runtime extraction and DLL loading acceptance.
        class Process:
            returncode = 0
            def __init__(self, command):
                self.command = command
            def communicate(self, timeout):
                if self.command[0].endswith("seconv.exe"):
                    return b"seconv 5.3.0\n", b""
                return b"2026.10.26\n", b""

        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            converter = root / "bin" / "seconv" / "seconv.exe"
            converter.parent.mkdir(parents=True)
            converter.write_bytes(b"only a fake binary for unit contracts")
            caption_tool = root / "bin" / "yt-dlp.exe"
            caption_tool.write_bytes(b"fake downloader")
            license_file = root / "licenses" / "seconv-MIT-LICENSE.txt"
            license_file.parent.mkdir()
            license_file.write_text("MIT license\n", encoding="utf-8")
            def proc(command, **kwargs):
                return Process(command)
            with (
                patch("hardsub_extractor.selftest.sys.frozen", True, create=True),
                patch("hardsub_extractor.selftest.bundle_root", return_value=root),
                patch("hardsub_extractor.selftest.resolve_binary",
                      side_effect=lambda name: name),
                patch("hardsub_extractor.selftest._version",
                      return_value="synthetic-binary"),
                patch("hardsub_extractor.selftest.external_popen",
                      side_effect=proc),
                patch("hardsub_extractor.subtitle_acquisition.find_caption_downloader",
                      return_value=str(caption_tool)),
            ):
                report = run_selftest(require_models=False)
                checks = {entry.name: entry for entry in report.checks}
                self.assertTrue(checks["seconv"].ok, report.to_json())
                self.assertEqual(checks["seconv"].detail,
                                 "5.3.0; local; free; MIT notice")
                self.assertTrue(checks["yt-dlp"].ok)
                license_file.unlink()
                incomplete = run_selftest(require_models=False)
                self.assertFalse(next(
                    entry for entry in incomplete.checks
                    if entry.name == "seconv"
                ).ok)


if __name__ == "__main__":
    unittest.main()
