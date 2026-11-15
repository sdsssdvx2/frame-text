"""Offline contracts for four-language hard-subtitle model routing and text fidelity."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np

from hardsub_extractor.model_assets import (
    ModelAssetError, load_manifest, normalize_ocr_language,
    resolve_model_paths, download_verified_models,
)
from hardsub_extractor.models import Frame
from hardsub_extractor.ocr.rapidocr_adapter import (
    RapidOCRAdapter, _subtitle_box_indices, _join_visual_line,
    _recover_english_leading_word,
)
from hardsub_extractor.gui import ExtractorGUI


def rect(x0, y0, x1, y1):
    return np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], dtype=float)


def dummy_manifest(root: Path) -> Path:
    def asset(name, content, *, source="https://example.invalid/"):
        (root / name).write_bytes(content)
        return {
            "filename": name,
            "url": source + name,
            "sha256": hashlib.sha256(content).hexdigest(),
            "size_bytes": len(content),
        }

    official = "https://www.modelscope.cn/models/RapidAI/RapidOCR/resolve/v3.10.0/"
    manifest = {
        "schema": 1,
        "assets": {
            "detector": asset("det.onnx", b"det"),
            "recognizer": asset("ko.onnx", b"korean"),
            "dictionary": asset("ko.txt", b"ko"),
        },
        "additional_profiles": {
            "en": {
                "recognizer": asset("en.onnx", b"english", source=official),
                "dictionary": asset("en.txt", b"en", source=official),
            },
            "cjk": {
                "recognizer": asset("cjk.onnx", b"chinese+japanese", source=official),
                "dictionary": asset("cjk.txt", b"cjk", source=official),
            },
        },
    }
    target = root / "manifest.json"
    target.write_text(json.dumps(manifest), encoding="utf-8")
    return target


class ModelProfileTests(unittest.TestCase):
    def test_official_model_manifest_pins_english_and_cjk(self):
        manifest = load_manifest()
        self.assertEqual(
            manifest["additional_profiles"]["en"]["recognizer"]["sha256"],
            "c3461add59bb4323ecba96a492ab75e06dda42467c9e3d0c18db5d1d21924be8",
        )
        self.assertEqual(
            manifest["additional_profiles"]["cjk"]["recognizer"]["sha256"],
            "5825fc7ebf84ae7a412be049820b4d86d77620f204a041697b0494669b1742c5",
        )
        self.assertEqual(
            manifest["additional_profiles"]["cjk"]["dictionary"]["sha256"],
            "d1979e9f794c464c0d2e0b70a7fe14dd978e9dc644c0e71f14158cdf8342af1b",
        )

    def test_only_four_explicit_ocr_languages_accepted(self):
        for raw, code in (
            ("ko", "ko"), ("EN", "en"), ("JA", "ja"),
            ("zh", "zh"), ("zh-Hans", "zh"), ("zh_Hant", "zh"),
        ):
            with self.subTest(raw=raw):
                self.assertEqual(normalize_ocr_language(raw), code)
        for raw in ("", "auto", "all", "zh-TW-guess", "ja;rm", "../../", "ru", "en,ja"):
            with self.subTest(raw=raw), self.assertRaises(ValueError):
                normalize_ocr_language(raw)

    def test_japanese_and_chinese_share_verified_recognizer_not_korean(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            manifest = dummy_manifest(root)
            ja = resolve_model_paths(root, manifest_path=manifest, language="ja")
            zh = resolve_model_paths(root, manifest_path=manifest, language="zh-Hant")
            en = resolve_model_paths(root, manifest_path=manifest, language="en")
            ko = resolve_model_paths(root, manifest_path=manifest, language="ko")
            self.assertEqual(ja.recognizer, zh.recognizer)
            self.assertEqual(ja.dictionary, zh.dictionary)
            self.assertNotEqual(ja.recognizer, ko.recognizer)
            self.assertNotEqual(ja.recognizer, en.recognizer)
            self.assertEqual(ja.detector, en.detector)
            self.assertEqual(ja.detector, ko.detector)

    def test_missing_or_modified_foreign_assets_fail_closed(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            manifest = dummy_manifest(root)
            (root / "cjk.onnx").write_bytes(b"corrupted")
            with self.assertRaisesRegex(ModelAssetError, "size mismatch"):
                resolve_model_paths(root, manifest_path=manifest, language="ja")
            with self.assertRaisesRegex(ModelAssetError, "size mismatch"):
                resolve_model_paths(root, manifest_path=manifest, language="zh")
            # Korean remains usable even when a different language is bad.
            self.assertTrue(resolve_model_paths(
                root, manifest_path=manifest, language="ko",
            ).recognizer.is_file())
            (root / "en.txt").unlink()
            with self.assertRaisesRegex(ModelAssetError, "missing"):
                resolve_model_paths(root, manifest_path=manifest, language="en")

    def test_invalid_optional_profile_pin_is_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            manifest = dummy_manifest(root)
            content = json.loads(manifest.read_text())
            content["additional_profiles"]["en"]["recognizer"]["filename"] = "../evil.onnx"
            manifest.write_text(json.dumps(content))
            with self.assertRaises(ModelAssetError):
                load_manifest(manifest)

    def test_english_download_reuses_detector_and_never_fetches_korean(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            manifest = dummy_manifest(root)
            files = {f.name: f.read_bytes() for f in root.iterdir() if f.is_file()}
            destination = root / "runtime"

            def open_asset(request, timeout=None):
                from io import BytesIO
                name = request.full_url.rsplit("/", 1)[-1]
                return BytesIO(files[name])

            with mock.patch(
                "hardsub_extractor.model_assets.urllib.request.urlopen",
                side_effect=open_asset,
            ) as urlopen:
                en = download_verified_models(
                    destination, manifest_path=manifest, language="en",
                )
            self.assertEqual(en.recognizer.name, "en.onnx")
            self.assertEqual(en.dictionary.name, "en.txt")
            self.assertEqual(urlopen.call_count, 3)
            self.assertFalse((destination / "ko.onnx").exists())


class LanguageGeometryTests(unittest.TestCase):
    SHAPE = (400, 1800, 3)

    def test_scripts_do_not_reuse_korean_hangul_filter(self):
        texts = ["안녕하세요", "Hello there", "你好", "こんにちは", "123", "©"]
        boxes = [rect(600 + (i % 2) * 50, 250, 1050, 330) for i in range(len(texts))]
        self.assertEqual(
            _subtitle_box_indices(texts, boxes, self.SHAPE, language="en"),
            [1],
        )
        self.assertEqual(
            _subtitle_box_indices(texts, boxes, self.SHAPE, language="zh"),
            [2],
        )
        self.assertEqual(
            _subtitle_box_indices(texts, boxes, self.SHAPE, language="ja"),
            [2, 3],
        )
        self.assertEqual(
            _subtitle_box_indices(texts, boxes, self.SHAPE), [0],
        )

    def test_cjk_fragments_join_without_spurious_spaces(self):
        chunks = ["君", "は", "誰", "？"]
        boxes = [rect(600+i*75, 290, 680+i*75, 355) for i in range(4)]
        self.assertEqual(
            _join_visual_line(chunks, boxes, [0, 1, 2, 3], language="ja"),
            "君は誰?",
        )
        self.assertEqual(
            _join_visual_line(["今", "天", "气", "好"], boxes, [0,1,2,3], language="zh"),
            "今天气好",
        )

    def test_long_english_caption_keeps_short_edge_word(self):
        boxes = [
            rect(150, 280, 190, 350),    # "I" left of the old 0.18 ROI
            rect(260, 282, 660, 350),    # central dialog
            rect(1450, 281, 1700, 349), # right-side credits on same baseline
        ]
        texts = ["I", "will be there", "Credits"]
        indices = _subtitle_box_indices(
            texts, boxes, self.SHAPE, language="en",
        )
        self.assertEqual(indices, [0, 1])
        self.assertEqual(
            _join_visual_line(texts, boxes, indices, language="en"),
            "I will be there",
        )

    def test_non_korean_lone_edge_overlay_remains_excluded(self):
        for language, text in (
            ("en", "Credits"), ("ja", "字幕監修"), ("zh", "片尾字幕"),
        ):
            with self.subTest(language=language):
                self.assertEqual(
                    _subtitle_box_indices(
                        [text], [rect(1510, 300, 1730, 350)],
                        self.SHAPE, language=language,
                    ),
                    [],
                )

    def test_original_korean_roi_remains_strict(self):
        self.assertEqual(
            _subtitle_box_indices(
                ["한국어", "대사"], [
                    rect(150, 280, 190, 350),
                    rect(400, 282, 680, 350),
                ], self.SHAPE,
            ),
            [1],
        )

    def test_cjk_overlapping_duplicate_glyph_is_removed_before_join(self):
        # Encoded video OCR may split overlapping text and re-recognize 一.
        boxes = [rect(600, 280, 865, 355), rect(830, 281, 1080, 354)]
        self.assertEqual(
            _join_visual_line(
                ["今天晚上一", "一起去"], boxes, [0, 1], language="zh",
            ),
            "今天晚上一起去",
        )

    def test_genuine_repeated_han_characters_are_not_deleted(self):
        boxes = [rect(600, 280, 685, 355), rect(710, 280, 795, 355)]
        self.assertEqual(
            _join_visual_line(["天", "天"], boxes, [0, 1], language="zh"),
            "天天",
        )

    def test_english_word_breaks_and_korean_regressions_remain(self):
        boxes = [rect(600,290,760,355),rect(790,290,980,355)]
        self.assertEqual(
            _join_visual_line(["Hello","world"], boxes, [0,1], language="en"),
            "Hello world",
        )
        self.assertEqual(
            _join_visual_line(["좋은","하루"], boxes, [0,1]),
            "좋은 하루",
        )


class LeadingEnglishGlyphTests(unittest.TestCase):
    @staticmethod
    def _scenario():
        image = np.zeros((288, 1178, 3), dtype=np.uint8)
        boxes = [
            rect(244, 101, 425, 159),
            rect(420, 102, 570, 159),
            rect(563, 99, 742, 161),
            rect(732, 101, 953, 160),
        ]
        primary = "will be there in five minutes."
        return image, boxes, primary

    def test_real_model_observed_edge_glyph_is_recovered_by_pixels_not_guess(self):
        image, boxes, primary = self._scenario()
        engine = mock.Mock(return_value=SimpleNamespace(
            txts=["I will be there in five minutes."],
            scores=[0.99592],
        ))
        text, confidence = _recover_english_leading_word(
            engine, image, boxes, [0, 1, 2, 3],
            primary, text_score=0.45,
        )
        self.assertEqual(text, "I will be there in five minutes.")
        self.assertAlmostEqual(confidence, 0.99592)
        args, kwargs = engine.call_args
        self.assertEqual(args[0].shape[2], 3)
        self.assertTrue(kwargs["use_rec"])
        self.assertFalse(kwargs["use_det"])
        self.assertGreater(args[0].shape[1], 700)
        self.assertLess(args[0].shape[1], 900)

    def test_fallback_never_invents_prefix_from_mismatched_recognition(self):
        image, boxes, primary = self._scenario()
        for candidate, score in (
            ("She will be there in five minutes.", 0.999),
            ("I might be there in five minutes.", 0.999),
            ("I will be there in five minutes.", 0.89),
            ("Iwill be there in five minutes.", 0.999),
            ("B will be there in five minutes!", 0.999),
        ):
            with self.subTest(candidate=candidate):
                engine = mock.Mock(return_value=SimpleNamespace(
                    txts=[candidate], scores=[score],
                ))
                recovered, confidence = _recover_english_leading_word(
                    engine, image, boxes, [0, 1, 2, 3],
                    primary, text_score=0.45,
                )
                self.assertEqual(recovered, primary)
                self.assertIsNone(confidence)

    def test_no_extra_ocr_for_full_width_or_lone_box_or_failed_fallback(self):
        image, boxes, primary = self._scenario()
        engine = mock.Mock(side_effect=RuntimeError("optional OCR failed"))
        for indices, adjusted_boxes in (
            ([0], boxes),
            ([0, 1, 2, 3], [
                rect(90, 101, 425, 159), *boxes[1:],
            ]),
            ([0, 1, 2, 3], boxes),
        ):
            recovered, confidence = _recover_english_leading_word(
                engine, image, adjusted_boxes, indices,
                primary, text_score=0.45,
            )
            self.assertEqual(recovered, primary)
            self.assertIsNone(confidence)
        self.assertEqual(engine.call_count, 1)

    def test_adapter_records_evidence_and_keeps_one_original_image_access(self):
        image, boxes, primary = self._scenario()
        class CounterFrame:
            pts = 6.5
            image_reads = 0

            @property
            def image(self):
                self.image_reads += 1
                return image

        frame = CounterFrame()
        normal = SimpleNamespace(
            txts=["will be", "there", "in five", "minutes."],
            scores=[0.998, 0.998, 0.996, 0.993],
            boxes=boxes,
        )
        recover = SimpleNamespace(
            txts=["I will be there in five minutes."],
            scores=[0.99592],
        )
        def run_ocr(_image, **kwargs):
            return normal if kwargs["use_det"] else recover

        adapter = RapidOCRAdapter.__new__(RapidOCRAdapter)
        adapter.engine = mock.Mock(side_effect=run_ocr)
        adapter.text_score = 0.45
        adapter.language = "en"
        adapter.korean_subtitle_mode = False
        output = adapter.recognize(frame)
        self.assertEqual(output.text, "I will be there in five minutes.")
        self.assertEqual(output.metadata["english_leading_glyph_recognition_only"], 1)
        self.assertEqual(frame.image_reads, 1)
        self.assertEqual(adapter.engine.call_count, 2)


class AdapterLanguageTests(unittest.TestCase):
    def _fake_rapidocr(self, engine_factory):
        enum = SimpleNamespace(KOREAN="korean", CH="ch", EN="en")
        return SimpleNamespace(
            EngineType=SimpleNamespace(ONNXRUNTIME="onnx"),
            ModelType=SimpleNamespace(MOBILE="mobile"),
            OCRVersion=SimpleNamespace(PPOCRV5="v5"),
            LangRec=enum,
            RapidOCR=engine_factory,
        )

    def test_verified_model_language_is_passed_to_rapidocr_engine(self):
        routes = {"ko": "korean", "en": "en", "zh": "ch", "ja": "ch"}
        for language, expected_route in routes.items():
            with self.subTest(language=language):
                fake = mock.Mock()
                fake.return_value = lambda image, **kwargs: SimpleNamespace(
                    txts=[], scores=[], boxes=[],
                )
                paths = SimpleNamespace(
                    detector=Path("det.onnx"), recognizer=Path("rec.onnx"),
                    dictionary=Path("dict.txt"),
                )
                with mock.patch.dict(sys.modules, {
                    "rapidocr": self._fake_rapidocr(fake),
                }), mock.patch(
                    "hardsub_extractor.model_assets.resolve_model_paths",
                    return_value=paths,
                ) as resolve:
                    adapter = RapidOCRAdapter(
                        language=language,
                        model_dir="verified_model_folder",
                        manifest_path="verified_manifest.json",
                    )
                self.assertEqual(resolve.call_args.kwargs["language"], language)
                params = fake.call_args.kwargs["params"]
                self.assertEqual(params["Rec.lang_type"], expected_route)
                self.assertEqual(params["Rec.model_path"], str(paths.recognizer))
                self.assertEqual(params["Rec.rec_keys_path"], str(paths.dictionary))
                self.assertEqual(adapter.korean_subtitle_mode, language == "ko")

    def test_non_korean_never_uses_implicit_rapidocr_download(self):
        with mock.patch.dict(sys.modules, {"rapidocr": mock.Mock()}):
            for lang in ("en", "ja", "zh"):
                with self.subTest(lang=lang), self.assertRaises(ModelAssetError):
                    RapidOCRAdapter(language=lang)

    def test_foreign_model_cannot_be_labeled_korean_only(self):
        with self.assertRaises(ValueError):
            RapidOCRAdapter(language="ja", korean_subtitle_mode=True, model_dir="models")

    def test_real_recognizer_path_controls_cjk_scripting_and_ocr_lines(self):
        adapter = RapidOCRAdapter.__new__(RapidOCRAdapter)
        adapter.text_score = 0.4
        adapter.language = "ja"
        adapter.korean_subtitle_mode = False
        adapter.engine = lambda image, **kwargs: SimpleNamespace(
            txts=["今日は", "晴れ", "안녕", "TOP TITLE"],
            scores=[0.99, 0.95, 0.91, 0.99],
            boxes=[
                rect(620, 250, 820, 325),
                rect(825, 250, 1080, 325),
                rect(600, 250, 790, 325),
                rect(600, 10, 1000, 50),
            ],
        )
        observed = adapter.recognize(Frame(pts=4, image=np.zeros((400, 1800, 3), dtype=np.uint8)))
        self.assertEqual(observed.text, "今日は晴れ")
        self.assertEqual(len(observed.metadata["detections"]), 2)


if __name__ == "__main__":
    unittest.main()
