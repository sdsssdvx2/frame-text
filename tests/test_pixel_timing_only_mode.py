"""Isolated wiring regressions for opt-in OCR text-preserving pixel timing.

No real media, models, browser, network, output publication or source edit.
The existing pipeline separately enforces unchanged cue count and wording.
"""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

from hardsub_extractor.ocr.adaptive import AdaptiveOCREngine
from hardsub_extractor.ocr.light_temporal import LightTemporalWitnessEngine
from hardsub_extractor.ocr.light_text import LightCaptionEngine
from hardsub_extractor.tab_capture import CAPTION_MODES, offline_ocr
from hardsub_extractor.local_batch import LocalBatchItem, run_local_batch


class PixelTimingPreservationTests(unittest.TestCase):
    def test_browser_pixel_only_retains_unfiltered_default_engine(self):
        self.assertIn("pixel-timing", CAPTION_MODES)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            models = root / "models"
            models.mkdir()
            manifest = root / "manifest.json"
            manifest.write_text("{}", encoding="utf-8")
            seen = []
            def capture(source, engine, output, **options):
                seen.append((engine, options["timing_refiner"], options))
                return SimpleNamespace(cue_count=2)
            with (
                patch("hardsub_extractor.runtime_paths.bundled_model_dir", return_value=models),
                patch("hardsub_extractor.runtime_paths.bundled_manifest_path", return_value=manifest),
                patch("hardsub_extractor.ocr.rapidocr_adapter.RapidOCRAdapter", return_value=object()),
                patch("hardsub_extractor.pipeline.extract", side_effect=capture),
            ):
                for mode in ("all", "pixel-timing", "light-temporal"):
                    report = offline_ocr(root / "input.mp4", root / mode, language="ja",
                                         caption_mode=mode)
                    self.assertEqual(report.cue_count, 2)
            baseline, pixel, light = seen
            self.assertIsInstance(baseline[0], AdaptiveOCREngine)
            self.assertIsNone(baseline[1])
            self.assertIsInstance(pixel[0], LightTemporalWitnessEngine)
            self.assertIsInstance(pixel[0].inner, AdaptiveOCREngine)
            self.assertIs(pixel[1].__self__, pixel[0])
            self.assertIsInstance(light[0].inner, LightCaptionEngine)
            self.assertIs(light[1].__self__, light[0])
            self.assertTrue(all(row[2]["streaming"] for row in seen))

    def test_batch_pixel_only_reuses_base_ocr_and_separates_witnesses(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            video1 = root / "one.mp4"
            video2 = root / "two.mp4"
            video1.write_bytes(b"scratch-first-video")
            video2.write_bytes(b"scratch-second-video")
            items = [LocalBatchItem(video1, root / "out1"),
                     LocalBatchItem(video2, root / "out2")]
            baseline_ocr = object()
            seen = []
            def capture(source, engine, output, **options):
                seen.append((engine, options.get("timing_refiner")))
                return SimpleNamespace(cue_count=3)
            with (
                patch("hardsub_extractor.local_batch.audit_local_output",
                      return_value=SimpleNamespace(state="missing")),
                patch("hardsub_extractor.local_batch.acquire_local_embedded_text",
                      return_value=None),
                patch("hardsub_extractor.local_batch.acquire_local_sidecar_text",
                      return_value=None),
                patch("hardsub_extractor.local_batch.acquire_local_embedded_image",
                      return_value=None),
                patch("hardsub_extractor.local_batch.extract_hard_subtitles",
                      side_effect=capture),
            ):
                rows = run_local_batch(items, ocr_factory=lambda: baseline_ocr,
                                       pixel_timing_only=True)
            self.assertEqual([r.mode for r in rows], ["hard_ocr", "hard_ocr"])
            self.assertEqual([r.cues for r in rows], [3, 3])
            self.assertEqual(len(seen), 2)
            self.assertIsNot(seen[0][0], seen[1][0])
            for engine, callback in seen:
                self.assertIsInstance(engine, LightTemporalWitnessEngine)
                self.assertIs(engine.inner, baseline_ocr)
                self.assertIs(callback.func.__self__, engine)

    def test_batch_rejects_incompatible_combinations(self):
        item = LocalBatchItem(Path("missing.mp4"), Path("out"))
        for flag in (
            {"caption_islands": True},
            {"light_caption_temporal": True},
            {"focus_bottom_two_lines": True},
            {"native_timing_factory": lambda: None},
        ):
            with self.subTest(flag=tuple(flag)):
                with self.assertRaises(ValueError):
                    run_local_batch([item], ocr_factory=object,
                                    pixel_timing_only=True, **flag)
        with self.assertRaises(ValueError):
            run_local_batch([item], ocr_factory=object,
                            pixel_timing_only=1)


if __name__ == "__main__":
    unittest.main()
