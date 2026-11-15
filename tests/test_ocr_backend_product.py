import sys
from types import SimpleNamespace
from unittest import TestCase, mock

from hardsub_extractor import cli, gui
from hardsub_extractor.ocr.rapidocr_adapter import (
    RapidOCRAdapter, openvino_runtime_available,
)



class _FakeRapid:
    fail_openvino = False

    def __init__(self, *, params):
        if self.fail_openvino and params["Det.engine_type"] == "openvino":
            raise RuntimeError("simulated OpenVINO init failure")
        self.params = dict(params)


_FAKE_RAPID_MODULE = SimpleNamespace(
    RapidOCR=_FakeRapid,
    EngineType=SimpleNamespace(ONNXRUNTIME="onnxruntime", OPENVINO="openvino"),
    LangRec=SimpleNamespace(KOREAN="korean", EN="english", CH="chinese"),
    ModelType=SimpleNamespace(MOBILE="mobile"),
    OCRVersion=SimpleNamespace(PPOCRV5="v5"),
)



class OCRBackendProductTests(TestCase):
    def test_openvino_packaging_probe_is_side_effect_free(self):
        with mock.patch(
            "hardsub_extractor.ocr.rapidocr_adapter.importlib.util.find_spec",
            return_value=object(),
        ):
            self.assertTrue(openvino_runtime_available())
        with mock.patch(
            "hardsub_extractor.ocr.rapidocr_adapter.importlib.util.find_spec",
            return_value=None,
        ):
            self.assertFalse(openvino_runtime_available())

    def test_missing_openvino_runtime_fails_before_rapidocr_import(self):
        with mock.patch(
            "hardsub_extractor.ocr.rapidocr_adapter.openvino_runtime_available",
            return_value=False,
        ):
            with self.assertRaisesRegex(RuntimeError, "OpenVINO"):
                RapidOCRAdapter(inference_engine="openvino")

    def test_auto_prefers_openvino_but_falls_back_to_onnx_on_init_failure(self):
        with (
            mock.patch.dict(sys.modules, {"rapidocr": _FAKE_RAPID_MODULE}),
            mock.patch(
                "hardsub_extractor.ocr.rapidocr_adapter.openvino_runtime_available",
                return_value=True,
            ),
        ):
            _FakeRapid.fail_openvino = False
            fast = RapidOCRAdapter(inference_engine="auto")
            self.assertEqual(fast.requested_inference_engine, "auto")
            self.assertEqual(fast.inference_engine, "openvino")
            self.assertIsNone(fast.backend_fallback_reason)

            _FakeRapid.fail_openvino = True
            fallback = RapidOCRAdapter(inference_engine="auto")
            self.assertEqual(fallback.requested_inference_engine, "auto")
            self.assertEqual(fallback.inference_engine, "onnxruntime")
            self.assertEqual(fallback.backend_fallback_reason, "RuntimeError")
            self.assertEqual(
                fallback.engine.params["Det.engine_type"], "onnxruntime",
            )
        _FakeRapid.fail_openvino = False

    def test_auto_openvino_first_inference_failure_switches_to_onnx_once(self):
        import numpy as np
        from hardsub_extractor.models import Frame

        calls = []

        class _LazyFailRapid(_FakeRapid):
            def __call__(self, image, **kwargs):
                calls.append(self.params["Det.engine_type"])
                if self.params["Det.engine_type"] == "openvino":
                    raise RuntimeError("simulated OpenVINO compile failure")
                return SimpleNamespace(txts=(), scores=(), boxes=None)

        module = SimpleNamespace(**{**vars(_FAKE_RAPID_MODULE), "RapidOCR": _LazyFailRapid})
        with (
            mock.patch.dict(sys.modules, {"rapidocr": module}),
            mock.patch(
                "hardsub_extractor.ocr.rapidocr_adapter.openvino_runtime_available",
                return_value=True,
            ),
        ):
            adapter = RapidOCRAdapter(inference_engine="auto")
            self.assertEqual(adapter.inference_engine, "openvino")
            frame = Frame(image=np.zeros((32, 64, 3), dtype=np.uint8), pts=1.0)
            observation = adapter.recognize(frame)
            self.assertEqual(observation.text, "")
            self.assertEqual(adapter.inference_engine, "onnxruntime")
            self.assertEqual(adapter.backend_fallback_reason, "RuntimeError")
            adapter.recognize(frame)
        self.assertEqual(calls, ["openvino", "onnxruntime", "onnxruntime"])

    def test_explicit_openvino_inference_failure_is_not_hidden(self):
        import numpy as np
        from hardsub_extractor.models import Frame

        class _AlwaysFailRapid(_FakeRapid):
            def __call__(self, image, **kwargs):
                raise RuntimeError("simulated OpenVINO failure")

        module = SimpleNamespace(**{**vars(_FAKE_RAPID_MODULE), "RapidOCR": _AlwaysFailRapid})
        with (
            mock.patch.dict(sys.modules, {"rapidocr": module}),
            mock.patch(
                "hardsub_extractor.ocr.rapidocr_adapter.openvino_runtime_available",
                return_value=True,
            ),
        ):
            adapter = RapidOCRAdapter(inference_engine="openvino")
            frame = Frame(image=np.zeros((32, 64, 3), dtype=np.uint8), pts=1.0)
            with self.assertRaisesRegex(RuntimeError, "simulated"):
                adapter.recognize(frame)
        self.assertEqual(adapter.inference_engine, "openvino")

    def test_auto_without_openvino_uses_existing_onnx_path(self):
        with (
            mock.patch.dict(sys.modules, {"rapidocr": _FAKE_RAPID_MODULE}),
            mock.patch(
                "hardsub_extractor.ocr.rapidocr_adapter.openvino_runtime_available",
                return_value=False,
            ),
        ):
            adapter = RapidOCRAdapter(inference_engine="auto")
        self.assertEqual(adapter.inference_engine, "onnxruntime")
        self.assertIsNone(adapter.backend_fallback_reason)

    def test_cli_explicit_and_auto_backend_reach_rapidocr(self):
        base = SimpleNamespace(
            ocr_language="ko",
            opencv_recognizer=False,
            recognizer_model=None,
            recognizer_dict=None,
            model_dir=None,
            text_score=.45,
            caption_islands=False,
            light_caption_only=False,
            light_caption_temporal=False,
            ocr_backend="openvino",
        )
        with (
            mock.patch("hardsub_extractor.cli.RapidOCRAdapter") as adapter,
            mock.patch(
                "hardsub_extractor.cli.AdaptiveOCREngine",
                side_effect=lambda value: value,
            ),
        ):
            cli._ocr(base)
        self.assertEqual(adapter.call_args.kwargs["inference_engine"], "openvino")

        base.ocr_backend = "onnxruntime"
        with (
            mock.patch("hardsub_extractor.cli.RapidOCRAdapter") as adapter,
            mock.patch(
                "hardsub_extractor.cli.AdaptiveOCREngine",
                side_effect=lambda value: value,
            ),
        ):
            cli._ocr(base)
        self.assertEqual(adapter.call_args.kwargs["inference_engine"], "onnxruntime")

        base.ocr_backend = "auto"
        with (
            mock.patch("hardsub_extractor.cli.RapidOCRAdapter") as adapter,
            mock.patch(
                "hardsub_extractor.cli.AdaptiveOCREngine",
                side_effect=lambda value: value,
            ),
        ):
            cli._ocr(base)
        self.assertEqual(adapter.call_args.kwargs["inference_engine"], "auto")

    def test_gui_worker_factory_uses_frozen_backend_snapshot(self):
        app = gui.ExtractorGUI.__new__(gui.ExtractorGUI)
        app._active_ocr_backend = "openvino"
        with mock.patch("hardsub_extractor.gui.RapidOCRAdapter") as adapter:
            app._make_rapidocr_adapter(
                model_dir="models",
                manifest_path="manifest.json",
                language="ja",
            )
        self.assertEqual(adapter.call_args.kwargs["inference_engine"], "openvino")

        del app._active_ocr_backend
        with mock.patch("hardsub_extractor.gui.RapidOCRAdapter") as adapter:
            app._make_rapidocr_adapter(
                model_dir="models",
                manifest_path="manifest.json",
                language="ja",
            )
        self.assertEqual(adapter.call_args.kwargs["inference_engine"], "onnxruntime")


if __name__ == "__main__":
    import unittest
    unittest.main()
