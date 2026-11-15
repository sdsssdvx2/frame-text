import hashlib
import io
import json
import unittest
import urllib.error
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from hardsub_extractor.model_assets import download_verified_models


class ModelMirrorTests(unittest.TestCase):
    def test_transient_primary_failure_uses_mirror_same_round(self):
        with TemporaryDirectory() as td:
            root = Path(td)
            payloads = {
                "detector": b"detector-bytes",
                "recognizer": b"recognizer-bytes",
                "dictionary": b"dictionary-bytes",
            }
            assets = {}
            for key, data in payloads.items():
                suffix = ".txt" if key == "dictionary" else ".onnx"
                spec = {
                    "filename": key + suffix,
                    "url": f"https://primary.invalid/{key}{suffix}",
                    "sha256": hashlib.sha256(data).hexdigest(),
                    "size_bytes": len(data),
                }
                if key == "detector":
                    spec["mirrors"] = [f"https://mirror.invalid/{key}{suffix}"]
                assets[key] = spec

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

            requested = [call.args[0].full_url for call in open_mock.call_args_list]
            self.assertEqual(requested[0], assets["detector"]["url"])
            self.assertEqual(requested[1], assets["detector"]["mirrors"][0])
            self.assertEqual(open_mock.call_count, 4)
            sleep_mock.assert_not_called()
            self.assertEqual(paths.detector.read_bytes(), payloads["detector"])
            self.assertEqual(paths.recognizer.read_bytes(), payloads["recognizer"])
            self.assertEqual(paths.dictionary.read_bytes(), payloads["dictionary"])


if __name__ == "__main__":
    unittest.main()
