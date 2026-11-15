from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]


class PackagingContractTests(unittest.TestCase):
    def test_only_headless_opencv_is_declared(self):
        text = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
        self.assertIn('opencv-python-headless>=4.10', text)
        self.assertNotIn('"opencv-python>=', text)

    def test_browser_render_runtime_is_pinned_and_frozen(self):
        project = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
        spec = (ROOT / "HardSubtitleExtractor.spec").read_text(encoding="utf-8")
        self.assertIn('"playwright==1.64.1"', project)
        self.assertIn('"playwright"', spec)
        self.assertIn("collect_all", spec)

    def test_windows_build_runs_frozen_selftest(self):
        text = (ROOT / ".github" / "workflows" / "windows-build.yml").read_text(encoding="utf-8")
        self.assertIn("HardSubtitleExtractor.exe", text)
        self.assertIn('"--self-test"', text)
        self.assertIn("frozen-selftest.json", text)
        self.assertIn("frozen-selftest-progress.jsonl", text)
        self.assertIn("runtime-build-inventory.json", text)
        self.assertIn("release-manifest.json", text)
        self.assertIn("THIRD_PARTY_NOTICES.md", text)
        self.assertIn("Start-Process", text)
        self.assertIn("-PassThru", text)
        self.assertIn("AddMinutes(5)", text)
        self.assertIn("Stop-Process", text)
        self.assertIn("windows-2025", text)
        # GitHub-hosted runners only: never a self-hosted label or variable.
        self.assertIn("runs-on: windows-2025", text)
        self.assertNotIn("HSE_WINDOWS_RUNNER", text)
        self.assertNotIn("self-hosted", text)
        self.assertIn('python-version: "3.12"', text)

    def test_release_manifest_records_tested_and_source_heads(self):
        text = (ROOT / ".github" / "workflows" / "windows-build.yml").read_text(encoding="utf-8")
        self.assertIn("git rev-parse HEAD", text)
        self.assertIn("tested_sha", text)
        self.assertIn("source_head_sha", text)
        self.assertIn("github.sha", text)

    def test_runtime_inventory_records_release_relevant_versions(self):
        text = (ROOT / ".github" / "workflows" / "windows-build.yml").read_text(encoding="utf-8")
        for package in ("python", "opencv", "rapidocr", "onnxruntime", "numpy", "pillow", "pyinstaller"):
            self.assertIn(f"{package} = $packages.{package}", text)

    def test_ocr_acceptance_runner_is_pinned(self):
        text = (ROOT / ".github" / "workflows" / "ocr-acceptance.yml").read_text(encoding="utf-8")
        self.assertIn("windows-2025", text)
        # GitHub-hosted runners only: never a self-hosted label or variable.
        self.assertIn("runs-on: windows-2025", text)
        self.assertNotIn("HSE_WINDOWS_RUNNER", text)
        self.assertNotIn("self-hosted", text)
        self.assertIn('python-version: "3.12"', text)

    def test_workflows_use_hosted_runners_and_safe_triggers(self):
        import re
        for path in sorted((ROOT / ".github" / "workflows").glob("*.yml")):
            text = path.read_text(encoding="utf-8")
            with self.subTest(workflow=path.name):
                self.assertNotIn("pull_request_target", text)
                self.assertNotRegex(text, r"(?m)^  pull_request:")
                for runs_on in re.findall(r"(?m)^\s*runs-on:\s*(.+)$", text):
                    self.assertRegex(runs_on.strip(), r"^(windows|ubuntu|macos)-(latest|\d{2,4}(\.\d+)?)$")
                for _ in re.findall(r"actions/upload-artifact@", text):
                    self.assertIn("retention-days:", text)

    def test_selftest_forces_real_rapidocr_inference(self):
        text = (ROOT / "src" / "hardsub_extractor" / "selftest.py").read_text(encoding="utf-8")
        self.assertIn('"rapidocr-inference"', text)
        self.assertIn("engine.recognize", text)
        self.assertIn("progress_path", text)

    def test_pyinstaller_uses_package_import_wrapper_and_notices(self):
        spec = (ROOT / "HardSubtitleExtractor.spec").read_text(encoding="utf-8")
        self.assertIn('ROOT / "tools" / "windows_entry.py"', spec)
        self.assertNotIn('SRC / "hardsub_extractor" / "gui_entry.py"', spec)
        self.assertIn('ROOT / "THIRD_PARTY_NOTICES.md"', spec)
        wrapper = (ROOT / "tools" / "windows_entry.py").read_text(encoding="utf-8")
        self.assertIn("from hardsub_extractor.gui_entry import main", wrapper)

    def test_ffmpeg_distribution_input_is_pinned_lgpl(self):
        script = (ROOT / "tools" / "prepare_ffmpeg_ci.ps1").read_text(encoding="utf-8")
        self.assertIn("win64-lgpl-9.0.zip", script)
        self.assertIn("expectedArchiveSha256", script)
        self.assertIn("--enable-gpl", script)
        self.assertIn('license_variant = "LGPL"', script)

    def test_third_party_notices_cover_major_bundled_components(self):
        notices = (ROOT / "THIRD_PARTY_NOTICES.md").read_text(encoding="utf-8")
        for name in (
            "FFmpeg",
            "RapidOCR",
            "PaddleOCR",
            "ONNX Runtime",
            "OpenCV",
            "NumPy",
            "Pillow",
            "Playwright",
            "PyInstaller",
        ):
            self.assertIn(name, notices)
        self.assertIn("Apache License 2.0", notices)
        self.assertIn("does not bundle a Playwright-downloaded browser binary", notices)

    def test_readme_tracks_verified_state_without_stale_test_count_contract(self):
        # The detailed verification log is kept in the private build only; the
        # public README is a short user guide and must not carry stale claims.
        log = ROOT / "docs" / "private-research-log.md"
        readme = (ROOT / "README.md").read_text(encoding="utf-8")
        if log.exists():
            detail = log.read_text(encoding="utf-8")
            self.assertNotIn("실제 PP-OCRv5 Korean inference는\n실행하지 못했습니다", detail)
            self.assertIn("실제 EpSite", detail)
            self.assertIn("regression suite PASS", detail)
            self.assertIn("release-manifest.json", detail)
        self.assertNotIn("77/77 PASS", readme)
        self.assertNotIn("77개 regression test", readme)


if __name__ == "__main__":
    unittest.main()
