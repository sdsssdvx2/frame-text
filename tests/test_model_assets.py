import hashlib
import io
import json
import unittest
import urllib.error
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from hardsub_extractor.model_assets import (
    ModelAssetError,
    download_verified_models,
    load_manifest,
    resolve_model_paths,
    verify_asset,
)


class ModelAssetTests(unittest.TestCase):
    def test_manifest_has_pinned_primary_assets(self):
        m = load_manifest()
        self.assertEqual(m["rapidocr_version"], "3.10.0")
        self.assertEqual(m["onnxruntime_version"], "1.31.1")
        self.assertEqual(
            m["assets"]["detector"]["sha256"],
            "4d97c44a20d30a81aad087d6a396b08f786c4635742afc391f6621f5c6ae78ae",
        )
        self.assertEqual(
            m["assets"]["recognizer"]["sha256"],
            "cd6e2ea50f6943ca7271eb8c56a877a5a90720b7047fe9c41a2e541a25773c9b",
        )
        self.assertEqual(
            m["assets"]["dictionary"]["sha256"],
            "a88071c68c01707489baa79ebe0405b7beb5cca229f4fc94cc3ef992328802d7",
        )

    def test_verify_rejects_modified_bytes(self):
        with TemporaryDirectory() as td:
            p = Path(td) / "x.bin"
            p.write_bytes(b"abc")
            good = hashlib.sha256(b"abc").hexdigest()
            verify_asset(p, good, 3)
            p.write_bytes(b"abd")
            with self.assertRaises(ModelAssetError):
                verify_asset(p, good, 3)

    def test_resolve_checks_all_assets(self):
        with TemporaryDirectory() as td:
            root = Path(td)
            contents = {
                "det": b"detector",
                "rec": b"recognizer",
                "dict": "가\n나\n".encode("utf-8"),
            }
            specs = {
                "detector": ("det.onnx", contents["det"]),
                "recognizer": ("rec.onnx", contents["rec"]),
                "dictionary": ("dict.txt", contents["dict"]),
            }
            assets = {}
            for key, (name, data) in specs.items():
                (root / name).write_bytes(data)
                assets[key] = {
                    "filename": name,
                    "url": "https://example.invalid/" + name,
                    "sha256": hashlib.sha256(data).hexdigest(),
                    "size_bytes": len(data),
                }
            manifest = root / "manifest.json"
            manifest.write_text(
                json.dumps({"schema": 1, "assets": assets}),
                encoding="utf-8",
            )
            paths = resolve_model_paths(root, manifest_path=manifest)
            self.assertTrue(paths.detector.is_file())
            self.assertTrue(paths.recognizer.is_file())
            self.assertTrue(paths.dictionary.is_file())

    def test_download_retries_transient_http_500_then_verifies(self):
        with TemporaryDirectory() as td:
            root = Path(td)
            payloads = {
                "detector": b"detector-bytes",
                "recognizer": b"recognizer-bytes",
                "dictionary": "가\n나\n".encode("utf-8"),
            }
            assets = {}
            for key, data in payloads.items():
                suffix = ".txt" if key == "dictionary" else ".onnx"
                assets[key] = {
                    "filename": key + suffix,
                    "url": f"https://example.invalid/{key}{suffix}",
                    "sha256": hashlib.sha256(data).hexdigest(),
                    "size_bytes": len(data),
                }
            manifest = root / "manifest.json"
            manifest.write_text(
                json.dumps({"schema": 1, "assets": assets}),
                encoding="utf-8",
            )
            transient = urllib.error.HTTPError(
                assets["detector"]["url"], 500, "temporary", None, None
            )
            responses = [
                transient,
                io.BytesIO(payloads["detector"]),
                io.BytesIO(payloads["recognizer"]),
                io.BytesIO(payloads["dictionary"]),
            ]
            with patch(
                "hardsub_extractor.model_assets.urllib.request.urlopen",
                side_effect=responses,
            ) as open_mock, patch("hardsub_extractor.model_assets.time.sleep") as sleep_mock:
                paths = download_verified_models(
                    root / "runtime",
                    manifest_path=manifest,
                    attempts=3,
                    retry_delay=0.01,
                )
            self.assertEqual(open_mock.call_count, 4)
            sleep_mock.assert_called_once()
            self.assertTrue(paths.detector.is_file())
            self.assertTrue(paths.recognizer.is_file())
            self.assertTrue(paths.dictionary.is_file())

    def test_download_does_not_retry_integrity_failure(self):
        with TemporaryDirectory() as td:
            root = Path(td)
            good = b"expected"
            assets = {
                key: {
                    "filename": key + (".txt" if key == "dictionary" else ".onnx"),
                    "url": f"https://example.invalid/{key}",
                    "sha256": hashlib.sha256(good).hexdigest(),
                    "size_bytes": len(good),
                }
                for key in ("detector", "recognizer", "dictionary")
            }
            manifest = root / "manifest.json"
            manifest.write_text(
                json.dumps({"schema": 1, "assets": assets}),
                encoding="utf-8",
            )
            with patch(
                "hardsub_extractor.model_assets.urllib.request.urlopen",
                return_value=io.BytesIO(b"corrupt!"),
            ) as open_mock:
                with self.assertRaises(ModelAssetError):
                    download_verified_models(
                        root / "runtime",
                        manifest_path=manifest,
                        attempts=3,
                        retry_delay=0,
                    )
            self.assertEqual(open_mock.call_count, 1)


if __name__ == "__main__":
    unittest.main()
