"""Contracts for the per-user Windows installer release (packaging/windows)."""
from __future__ import annotations

import re
import subprocess
import sys
import tomllib
import unittest
import zipfile
from pathlib import Path
from tempfile import TemporaryDirectory

ROOT = Path(__file__).resolve().parents[1]
PKG = ROOT / "packaging" / "windows"


def _release_module():
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "hse_make_user_release", ROOT / "tools" / "make_user_release.py",
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _pins(path: Path) -> dict[str, str]:
    pins = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            name, version = line.split("==")
            pins[name.lower().replace("_", "-")] = version
    return pins


class UserReleasePackagingTests(unittest.TestCase):
    def test_requirements_match_pyproject_pins(self):
        pins = _pins(PKG / "requirements-windows.txt")
        project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
        declared = list(project["project"]["dependencies"])
        for extra in ("ocr", "accelerated"):
            declared += project["project"]["optional-dependencies"][extra]
        for spec in declared:
            name = re.split(r"[<>=!~ ]", spec, maxsplit=1)[0].lower()
            with self.subTest(name=name):
                self.assertIn(name, pins)
                if "==" in spec:
                    self.assertEqual(pins[name], spec.split("==", 1)[1])
        # Windows-only transitive dependencies must be pinned explicitly.
        self.assertIn("colorama", pins)

    def test_installer_binary_pins_match_ci_preparers(self):
        installer = (PKG / "install.ps1").read_text(encoding="utf-8-sig")
        ffmpeg = (ROOT / "tools" / "prepare_ffmpeg_ci.ps1").read_text(encoding="utf-8")
        ytdlp = (ROOT / "tools" / "prepare_yt_dlp_ci.ps1").read_text(encoding="utf-8")
        seconv = (ROOT / "tools" / "prepare_seconv_ci.ps1").read_text(encoding="utf-8")
        for source in (ffmpeg, ytdlp, seconv):
            digest = re.search(r"'([0-9a-f]{64})'|\"([0-9a-f]{64})\"", source)
            self.assertIsNotNone(digest)
            self.assertIn(digest.group(1) or digest.group(2), installer)
        for name, digest in _release_module().PINNED_BINARIES.items():
            self.assertIn(digest, installer, name)

    def test_installer_never_deletes_user_data_and_backs_up(self):
        installer = (PKG / "install.ps1").read_text(encoding="utf-8-sig")
        self.assertIn("Move-Item -LiteralPath $InstallDir -Destination $BackupDir", installer)
        self.assertNotIn("Remove-Item -LiteralPath $InstallDir", installer)
        self.assertIn("$DryRun", installer)
        uninstall = (PKG / "uninstall.ps1").read_text(encoding="utf-8-sig")
        self.assertIn("launch_hse.pyw", uninstall)  # guarded delete
        self.assertNotIn("HardSubtitleExtractor\\managed", uninstall)

    def test_release_zip_launcher_runs_self_test_from_installed_layout(self):
        make_user_release = _release_module()
        with TemporaryDirectory() as td:
            out = Path(td) / "hse.zip"
            make_user_release.build(out, "test", None, None)
            with zipfile.ZipFile(out) as z:
                names = set(z.namelist())
                self.assertIn("HSE-test/install.ps1", names)
                self.assertIn("HSE-test/app/src/hardsub_extractor/gui_entry.py", names)
                self.assertIn("HSE-test/app/models/manifest.json", names)
                self.assertFalse(any("__pycache__" in n for n in names))
                z.extractall(td)
            # Reproduce what install.ps1 does: app payload + launcher.
            installed = Path(td) / "installed"
            (Path(td) / "HSE-test" / "app").rename(installed)
            (Path(td) / "HSE-test" / "launch_hse.pyw").rename(installed / "launch_hse.pyw")
            report = installed / "selftest.json"
            proc = subprocess.run(
                [sys.executable, str(installed / "launch_hse.pyw"), "--self-test-json", str(report)],
                capture_output=True, text=True, timeout=180,
                env={"PATH": "", "SYSTEMROOT": ""},
            )
            # No models/ffmpeg in this layout: the self-test must run from the
            # installed tree and fail closed (exit 2), never crash (exit 3).
            self.assertEqual(proc.returncode, 2, proc.stderr[-2000:])
            import json

            payload = json.loads(report.read_text(encoding="utf-8"))
            self.assertFalse(payload["ok"])
            self.assertIn("checks", payload)


if __name__ == "__main__":
    unittest.main()
