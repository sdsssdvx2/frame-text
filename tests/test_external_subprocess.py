"""Standalone executable launch contracts; no binary or network required."""
from __future__ import annotations

import os
import sys
import subprocess
import unittest
from unittest import mock

from hardsub_extractor.external_subprocess import external_popen, _external_env


class ExternalExecutableIsolationTests(unittest.TestCase):
    def test_source_launch_does_not_modify_user_process_environment(self):
        with mock.patch.object(sys, "frozen", False, create=True), mock.patch(
            "hardsub_extractor.external_subprocess.subprocess.Popen",
        ) as launcher, mock.patch(
            "hardsub_extractor.external_subprocess._set_windows_dll_directory",
        ) as set_dll:
            external_popen(["yt-dlp-test", "--version"], stdin=subprocess.DEVNULL)
        args, kwargs = launcher.call_args
        self.assertEqual(args[0], ["yt-dlp-test", "--version"])
        self.assertNotIn("env", kwargs)
        set_dll.assert_not_called()

    def test_frozen_child_resets_nested_pyinstaller_env_and_restores_dll(self):
        with mock.patch.object(sys, "frozen", True, create=True), mock.patch.object(
            sys, "_MEIPASS", "X:\\TEMP\\HSE-ONEFILE", create=True,
        ), mock.patch.object(sys, "platform", "win32"), mock.patch.object(
            subprocess, "CREATE_NO_WINDOW", 0x08000000, create=True,
        ), mock.patch(
            "hardsub_extractor.external_subprocess._set_windows_dll_directory",
        ) as set_dll, mock.patch(
            "hardsub_extractor.external_subprocess.subprocess.Popen",
        ) as launcher, mock.patch.dict(
            os.environ, {"HSE_EXTERNAL_TEST_ONLY": "preserve"}, clear=False,
        ):
            external_popen(["C:\\verified\\yt-dlp.exe", "--version"], stdout=subprocess.PIPE)
        self.assertEqual(set_dll.call_args_list, [
            mock.call(None), mock.call("X:\\TEMP\\HSE-ONEFILE"),
        ])
        options = launcher.call_args.kwargs
        self.assertEqual(options["env"]["PYINSTALLER_RESET_ENVIRONMENT"], "1")
        self.assertEqual(options["env"]["HSE_EXTERNAL_TEST_ONLY"], "preserve")
        self.assertEqual(options["creationflags"] & 0x08000000, 0x08000000)

    def test_frozen_custom_env_is_preserved_except_bootloader_reset(self):
        with mock.patch.object(sys, "frozen", True, create=True), mock.patch.object(
            sys, "_MEIPASS", None, create=True,
        ), mock.patch(
            "hardsub_extractor.external_subprocess.subprocess.Popen"
        ) as launcher:
            external_popen(["yt-dlp.exe"], env={
                "CUSTOM_KEY": "unchanged", "PYINSTALLER_RESET_ENVIRONMENT": "0",
            })
        environment = launcher.call_args.kwargs["env"]
        self.assertEqual(environment["CUSTOM_KEY"], "unchanged")
        self.assertEqual(environment["PYINSTALLER_RESET_ENVIRONMENT"], "1")

    def test_createprocess_failure_always_restores_windows_dll_path(self):
        with mock.patch.object(sys, "frozen", True, create=True), mock.patch.object(
            sys, "_MEIPASS", "X:\\HSE", create=True,
        ), mock.patch.object(sys, "platform", "win32"), mock.patch.object(
            subprocess, "CREATE_NO_WINDOW", 0x08000000, create=True,
        ), mock.patch(
            "hardsub_extractor.external_subprocess._set_windows_dll_directory"
        ) as set_dll, mock.patch(
            "hardsub_extractor.external_subprocess.subprocess.Popen",
            side_effect=OSError("simulated failure"),
        ):
            with self.assertRaisesRegex(OSError, "simulated failure"):
                external_popen(["yt-dlp.exe"])
        self.assertEqual(set_dll.call_args_list, [
            mock.call(None), mock.call("X:\\HSE"),
        ])

    def test_restore_failure_terminates_child_instead_of_leaking_process(self):
        child = mock.Mock()
        with mock.patch.object(sys, "frozen", True, create=True), mock.patch.object(
            sys, "_MEIPASS", "X:\\HSE", create=True,
        ), mock.patch.object(sys, "platform", "win32"), mock.patch.object(
            subprocess, "CREATE_NO_WINDOW", 0x08000000, create=True,
        ), mock.patch(
            "hardsub_extractor.external_subprocess._set_windows_dll_directory",
            side_effect=[None, OSError("simulated DLL restore failure")],
        ), mock.patch(
            "hardsub_extractor.external_subprocess.subprocess.Popen",
            return_value=child,
        ):
            with self.assertRaisesRegex(OSError, "DLL restore"):
                external_popen(["yt-dlp.exe"])
        child.kill.assert_called_once()
        child.wait.assert_called_once_with(timeout=5)

    def test_frozen_selftest_ytdlp_uses_isolated_launcher_and_verified_version(self):
        """The source-mode frozen selftest now requires BOTH portable tools."""
        import tempfile
        from pathlib import Path
        from hardsub_extractor.selftest import run_selftest
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            executable = root / "bin" / "yt-dlp.exe"
            executable.parent.mkdir()
            executable.write_bytes(b"unit-test-verified-downloader")
            converter = root / "bin" / "seconv" / "seconv.exe"
            converter.parent.mkdir()
            converter.write_bytes(b"unit-test-verified-portable-converter")
            notice = root / "licenses" / "seconv-MIT-LICENSE.txt"
            notice.parent.mkdir()
            notice.write_text("MIT synthetic test fixture", encoding="utf-8")
            downloader = mock.Mock(returncode=0)
            downloader.communicate.return_value = (b"2026.10.26\n", b"")
            seconv = mock.Mock(returncode=0)
            seconv.communicate.return_value = (b"5.3.0\n", b"")
            with mock.patch.object(sys, "frozen", True, create=True), mock.patch(
                "hardsub_extractor.selftest.resolve_binary",
                side_effect=lambda name: name,
            ), mock.patch(
                "hardsub_extractor.selftest._version", return_value="version",
            ), mock.patch(
                "hardsub_extractor.subtitle_acquisition.bundle_root",
                return_value=root,
            ), mock.patch(
                "hardsub_extractor.selftest.bundle_root",
                return_value=root,
            ), mock.patch(
                "hardsub_extractor.selftest.external_popen",
                side_effect=[downloader, seconv],
            ) as launcher:
                report = run_selftest(require_models=False)
            self.assertTrue(report.ok, report.to_json())
            self.assertEqual(launcher.call_count, 2)
            self.assertEqual(
                launcher.call_args_list[0].args[0],
                [str(executable), "--version"],
            )
            self.assertEqual(
                launcher.call_args_list[1].args[0],
                [str(converter), "--version"],
            )
            for call in launcher.call_args_list:
                self.assertIs(call.kwargs["stdin"], subprocess.DEVNULL)
                self.assertIs(call.kwargs["stderr"], subprocess.DEVNULL)

    def test_frozen_selftest_ytdlp_failure_records_safe_exit_only(self):
        import tempfile
        from pathlib import Path
        from hardsub_extractor.selftest import run_selftest
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            executable = root / "bin" / "yt-dlp.exe"
            executable.parent.mkdir()
            executable.write_bytes(b"unit-test-downloader")
            process = mock.Mock()
            process.communicate.return_value = (b"bad-output-SECRET", b"")
            process.returncode = 93
            with mock.patch.object(sys, "frozen", True, create=True), mock.patch(
                "hardsub_extractor.selftest.resolve_binary",
                side_effect=lambda name: name,
            ), mock.patch(
                "hardsub_extractor.selftest._version", return_value="version",
            ), mock.patch(
                "hardsub_extractor.subtitle_acquisition.bundle_root",
                return_value=root,
            ), mock.patch(
                "hardsub_extractor.selftest.bundle_root",
                return_value=root,
            ), mock.patch(
                "hardsub_extractor.selftest.external_popen", return_value=process,
            ):
                report = run_selftest(require_models=False)
            self.assertFalse(report.ok)
            row = next(check for check in report.checks if check.name == "yt-dlp")
            self.assertIn("exit_code=93", row.detail)
            self.assertIn("stdout_bytes=17", row.detail)
            self.assertNotIn("SECRET", report.to_json())

    def test_frozen_environment_is_only_prepared_not_installed_globally(self):
        with mock.patch.object(sys, "frozen", True, create=True), mock.patch.dict(
            os.environ,
            {"PYINSTALLER_RESET_ENVIRONMENT": "0"},
            clear=False,
        ):
            prepared = _external_env()
            self.assertEqual(prepared["PYINSTALLER_RESET_ENVIRONMENT"], "1")
            self.assertEqual(os.environ["PYINSTALLER_RESET_ENVIRONMENT"], "0")


if __name__ == "__main__":
    unittest.main()
