"""Isolated HSE ONNX CPU threading selection: no OCR model or network calls."""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest import TestCase, mock
import sys

from hardsub_extractor.ocr.rapidocr_adapter import RapidOCRAdapter


class FakeRapid:
    def __init__(self, *, params):
        self.params = dict(params)


MOCK_RAPID = SimpleNamespace(
    RapidOCR=FakeRapid,
    EngineType=SimpleNamespace(ONNXRUNTIME="onnxruntime", OPENVINO="openvino"),
    LangRec=SimpleNamespace(KOREAN="korean", EN="english", CH="chinese"),
    ModelType=SimpleNamespace(MOBILE="mobile"),
    OCRVersion=SimpleNamespace(PPOCRV5="v5"),
)
MODELS = SimpleNamespace(
    detector=Path("root/detector.onnx"),
    recognizer=Path("root/recognizer.onnx"),
    dictionary=Path("root/dict.txt"),
)


class OnnxCPUThreadOptions(TestCase):
    def adapter(self, value=None, language="ja"):
        with (
            mock.patch.dict(sys.modules, {"rapidocr": MOCK_RAPID}),
            mock.patch(
                "hardsub_extractor.model_assets.resolve_model_paths",
                return_value=MODELS,
            ) as pin,
        ):
            adapter = RapidOCRAdapter(
                language=language, model_dir="root",
                onnx_intra_threads=value,
            )
        pin.assert_called_once()
        self.assertEqual(adapter.engine.params["Det.model_path"], str(MODELS.detector))
        self.assertEqual(adapter.engine.params["Rec.model_path"], str(MODELS.recognizer))
        return adapter.engine.params

    def test_existing_default_model_config_stays_byte_logically_identical(self):
        p = self.adapter()
        self.assertNotIn("EngineConfig.onnxruntime.intra_op_num_threads", p)
        self.assertEqual(p["Rec.lang_type"], "chinese")
        self.assertEqual(p["Det.ocr_version"], "v5")

    def test_explicit_two_or_four_threads_reach_both_same_onnx_models(self):
        for value in (2, 4):
            with self.subTest(threads=value):
                p = self.adapter(value)
                self.assertEqual(
                    p["EngineConfig.onnxruntime.intra_op_num_threads"], value,
                )
                self.assertEqual(p["Det.engine_type"], "onnxruntime")
                self.assertEqual(p["Rec.engine_type"], "onnxruntime")
                self.assertEqual(p["Global.use_cls"], False)

    def test_explicit_openvino_keeps_same_pinned_models_and_switches_only_engine(self):
        p = self.adapter()
        with (
            mock.patch.dict(sys.modules, {"rapidocr": MOCK_RAPID}),
            mock.patch(
                "hardsub_extractor.model_assets.resolve_model_paths",
                return_value=MODELS,
            ),
        ):
            adapter = RapidOCRAdapter(
                language="ja", model_dir="root",
                inference_engine="openvino",
            )
        p2 = adapter.engine.params
        self.assertEqual(p2["Det.engine_type"], "openvino")
        self.assertEqual(p2["Rec.engine_type"], "openvino")
        self.assertEqual(p2["Det.model_path"], p["Det.model_path"])
        self.assertEqual(p2["Rec.model_path"], p["Rec.model_path"])
        self.assertEqual(p2["Rec.rec_keys_path"], p["Rec.rec_keys_path"])
        self.assertNotIn("EngineConfig.onnxruntime.intra_op_num_threads", p2)

    def test_openvino_rejects_onnx_thread_override(self):
        with mock.patch.dict(sys.modules, {"rapidocr": MOCK_RAPID}):
            with self.assertRaises(ValueError):
                RapidOCRAdapter(
                    language="ja", model_dir="root",
                    inference_engine="openvino", onnx_intra_threads=4,
                )

    def test_unknown_inference_engine_fails_before_model_import(self):
        with mock.patch.dict(sys.modules, {"rapidocr": MOCK_RAPID}):
            with self.assertRaises(ValueError):
                RapidOCRAdapter(
                    language="ja", model_dir="root",
                    inference_engine="magic",
                )

    def test_invalid_unsafe_or_implicit_values_fail_before_model_import(self):
        for value in (-1, 0, 17, 3.5, "4", True, False):
            with self.subTest(value=value), mock.patch.dict(
                sys.modules, {"rapidocr": MOCK_RAPID},
            ):
                with self.assertRaises(ValueError):
                    RapidOCRAdapter(
                        language="ja", model_dir="root", onnx_intra_threads=value,
                    )


if __name__ == "__main__":
    import unittest
    unittest.main()
