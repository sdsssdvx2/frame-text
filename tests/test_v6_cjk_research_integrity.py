"""Offline integrity contracts for research-only PP-OCRv6 CJK study."""
from __future__ import annotations

import hashlib
import io
from pathlib import Path
import sys
import tempfile
from unittest import mock
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"tools"))
import compare_cjk_v5_v6_recognition as trial


class PinnedV6ResearchTests(unittest.TestCase):
    def test_official_model_url_is_fixed_and_checksum_pinned(self):
        self.assertEqual(
            trial.ROOT_URL,
            "https://www.modelscope.cn/models/RapidAI/RapidOCR/resolve/v3.9.2/",
        )
        self.assertTrue(all(
            len(spec["sha256"]) == 64
            and all(c in "0123456789abcdef" for c in spec["sha256"])
            and spec["path"].startswith(
                ("onnx/PP-OCRv6/", "paddle/PP-OCRv6/")
            )
            for spec in (trial.V6_RECOGNIZER, trial.V6_DICTIONARY)
        ))

    def test_good_and_bad_verified_bytes(self):
        payload = b"official-research-fixture"
        spec = {
            "sha256": hashlib.sha256(payload).hexdigest(),
            "max_bytes": len(payload),
        }
        self.assertTrue(trial.verify_official_resource(spec, payload))
        self.assertFalse(trial.verify_official_resource(spec, b"changed"))
        self.assertFalse(trial.verify_official_resource(spec, b""))
        self.assertFalse(trial.verify_official_resource(
            {**spec, "max_bytes": len(payload)-1}, payload,
        ))

    def test_retrieval_checks_sha_and_stays_in_isolated_folder(self):
        payload = b"cjk-model-fixture"
        spec = {
            "path": "onnx/PP-OCRv6/rec/model.onnx",
            "sha256": hashlib.sha256(payload).hexdigest(),
            "max_bytes": 1024,
        }
        with tempfile.TemporaryDirectory() as td, mock.patch(
            "compare_cjk_v5_v6_recognition.urllib.request.urlopen",
            return_value=io.BytesIO(payload),
        ):
            root = Path(td)
            path = trial.retrieve_verified(spec, root)
            self.assertEqual(path.read_bytes(), payload)
            self.assertEqual(path.parent, root)
            self.assertFalse((root/"model.onnx.part").exists())

    def test_bad_sha_fails_closed_without_leaving_partial_model(self):
        spec = {
            "path": "onnx/PP-OCRv6/rec/model.onnx",
            "sha256": "0"*64,
            "max_bytes": 1024,
        }
        with tempfile.TemporaryDirectory() as td, mock.patch(
            "compare_cjk_v5_v6_recognition.urllib.request.urlopen",
            return_value=io.BytesIO(b"other"),
        ):
            root = Path(td)
            with self.assertRaisesRegex(RuntimeError, "SHA-256"):
                trial.retrieve_verified(spec, root)
            self.assertFalse((root/"model.onnx").exists())
            self.assertFalse((root/"model.onnx.part").exists())

    def test_unsafe_model_filename_and_collision_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            for filename in ("../secret", "/root", "folder\\item", ""):
                with self.subTest(filename=filename), self.assertRaises(ValueError):
                    trial.retrieve_verified({
                        "path": filename,
                        "sha256": "0"*64,
                        "max_bytes": 1024,
                    }, root)
            (root/"model.onnx").write_bytes(b"protected")
            with self.assertRaises(RuntimeError):
                trial.retrieve_verified({
                    "path": "onnx/model.onnx",
                    "sha256": "0"*64,
                    "max_bytes": 1024,
                }, root)
            self.assertEqual((root/"model.onnx").read_bytes(), b"protected")

    def test_reference_selection_never_changes_model_input(self):
        self.assertEqual(trial.LANGUAGES, ("ja", "zh"))
        self.assertLessEqual(trial.MAX_MODEL_FRAMES, 150)


if __name__ == "__main__":
    unittest.main()
