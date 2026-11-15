"""No-network contracts for the pinned external subtitle OCR pilot."""
from __future__ import annotations
import csv
from hashlib import sha1
from io import StringIO
from pathlib import Path
import sys
import unittest
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import benchmark_geeklink_ocr as study


def rows():
    result = []
    for film in study.FILMS:
        for language in study.LANGUAGES:
            for mark in ("False", "True"):
                name = f"{film}_{len(result):04d}"
                result.append(dict(id=name, video_id=film, lang=language,
                                   image=f"data/images/{name}.jpg",
                                   ground_truth="안녕", has_watermark=mark))
    return result


class PinnedGeekLinkBenchmarkTests(unittest.TestCase):
    def test_all_git_source_pins_have_exact_40_hex_characters(self):
        import re
        for pin in (study.COMMIT, study.TREE, study.MANIFEST_BLOB,
                    study.BASELINE_BLOB):
            self.assertRegex(pin, r"^[0-9a-f]{40}$")

    def test_24_stratified_images_cover_every_language_film_and_watermark(self):
        picked = study._select(rows())
        self.assertEqual(len(picked), 24)
        self.assertEqual(len(set(item["id"] for item in picked)), 24)
        for language in study.LANGUAGES:
            self.assertEqual(sum(x["lang"] == language for x in picked), 6)
        self.assertEqual(sum(x["has_watermark"] == "True" for x in picked), 12)

    def test_manifest_integrity_and_no_path_traversal(self):
        source = rows()
        output = StringIO()
        writer = csv.DictWriter(output, fieldnames=study.COLUMNS)
        writer.writeheader()
        writer.writerows(source)
        data = output.getvalue().encode()
        self.assertEqual(len(study._manifest(data, count=24)), 24)
        with self.assertRaises(study.BenchmarkError):
            study._manifest(data.replace(source[0]["image"].encode(),
                                         b"../../private.jpg"), count=24)

    def test_pinned_git_blob_detects_content_changes(self):
        data = b"safe-public-benchmark-test"
        digest = sha1(b"blob " + str(len(data)).encode() + b"\0" + data).hexdigest()
        study._git_blob_verify(data, digest)
        with self.assertRaises(study.BenchmarkError):
            study._git_blob_verify(data + b"altered", digest)

    def test_cer_uses_exact_characters_and_watermark_subsets(self):
        metrics = study._score([
            ("ko", "False", "안녕", "안녕"),
            ("zh", "True", "你好", "你们好"),
        ])
        self.assertEqual(metrics["all"], {"n": 2, "cer": 0.2})
        self.assertEqual(metrics["watermark"], {"n": 1, "cer": 0.3333})

    def test_preflight_only_is_dependency_free_and_checks_explicit_consent(self):
        with mock.patch.object(study, "source_preflight", return_value=None) as preflight:
            result = study.main(["--source-only", "--accept-public-download"])
        self.assertEqual(result, 0)
        preflight.assert_called_once()

    def test_unapproved_cli_never_calls_model_or_network(self):
        with mock.patch.object(study, "measure") as measured:
            code = study.main(["--model-dir", "missing",
                               "--manifest", "missing", "--out", "missing.json"])
        self.assertEqual(code, 2)
        measured.assert_not_called()

    def test_network_rejects_untrusted_host_before_connection(self):
        with mock.patch.object(study.urllib.request, "build_opener") as opener:
            with self.assertRaises(study.BenchmarkError):
                study._fetch("https://evil.invalid/image.jpg", "api.github.com", 200)
            with self.assertRaises(study.BenchmarkError):
                study._fetch("https://api.github.com/path?auth=token", "api.github.com", 200)
            opener.assert_not_called()


if __name__ == "__main__":
    unittest.main()
