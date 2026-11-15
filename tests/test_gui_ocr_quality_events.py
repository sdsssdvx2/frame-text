"""Targeted zero-OCR GUI evidence display for single/batch local videos.

The existing media source/SRT are never opened by these UI event tests.
"""
from __future__ import annotations

import queue
from types import SimpleNamespace
from unittest import TestCase, mock

from hardsub_extractor.gui import ExtractorGUI


class LocalQualityEventTests(TestCase):
    def _gui(self, mode):
        gui = ExtractorGUI.__new__(ExtractorGUI)
        gui.events = queue.Queue()
        gui.root = mock.Mock()
        gui.source_mode_var = mock.Mock()
        gui.source_mode_var.get.return_value = mode
        gui._active_source_mode = mode
        gui._pending_close = False
        gui.review_button = mock.Mock()
        gui.progress_bar = mock.Mock()
        gui.start_button = mock.Mock()
        gui.cancel_button = mock.Mock()
        gui.status_var = mock.Mock()
        gui.worker = None
        gui._log = mock.Mock()
        return gui

    def test_single_ocr_warning_explained_without_overwrite(self):
        gui = self._gui("local")
        gui.events.put(("ocr_quality_signal", 2))
        gui.events.put(("ocr_quality_cue_numbers", (3, 9)))
        gui.events.put(("done", ("existing.srt", "existing.ass")))
        with mock.patch("hardsub_extractor.gui.messagebox.showinfo") as info:
            gui._poll_events()
        gui.status_var.set.assert_called_with("완료 · 검토 권장")
        self.assertEqual(info.call_count, 1)
        message = info.call_args.args[1]
        self.assertIn("2건", message)
        self.assertIn("원본 화면 검토", message)
        self.assertIn("참고 자막 번호: 3, 9", message)
        self.assertIn("정확성을 증명", message)
        self.assertEqual(gui._last_local_ocr_quality_signals, 0)
        gui.root.after.assert_called_once()

    def test_batch_report_preserves_existing_completion_contract(self):
        gui = self._gui("local-batch")
        gui._verified_existing_count = 1
        gui.events.put(("batch_ocr_quality_item", (5, 3, (1, 11))))
        gui.events.put(("batch_ocr_quality", (3, 2, 1)))
        gui.events.put(("captions_done", (3, 0, 0, 0, "outputs")))
        with mock.patch("hardsub_extractor.gui.messagebox.showinfo") as info:
            gui._poll_events()
        gui.status_var.set.assert_called_with("완료 · 검토 권장")
        msg = info.call_args.args[1]
        self.assertIn("검토 권장: 2개", msg)
        self.assertIn("품질 확인 불가: 1개", msg)
        self.assertIn("인식", msg)
        self.assertTrue(any(
            "영상 5번" in str(call) and "자막 번호 1, 11" in str(call)
            for call in gui._log.call_args_list
        ))
        self.assertFalse(any("private" in str(call) for call in gui._log.call_args_list))
        self.assertEqual(gui._last_batch_quality_review, (0, 0, 0))
        self.assertEqual(gui._verified_existing_count, 0)

    def test_regular_source_original_does_not_acquire_ocr_warning_state(self):
        gui = self._gui("local")
        gui.events.put(("done", ("original.srt", "original.ass")))
        with mock.patch("hardsub_extractor.gui.messagebox.showinfo") as info:
            gui._poll_events()
        gui.status_var.set.assert_called_with("완료")
        self.assertNotIn("검토 권장 신호", info.call_args.args[1])

    def test_after_export_audit_failure_does_not_falsely_fail_extraction(self):
        gui = self._gui("local")
        with mock.patch(
            "hardsub_extractor.local_ocr_quality_signals.local_ocr_quality_signals",
            side_effect=ValueError("private path hidden"),
        ):
            gui._emit_local_ocr_quality(
                SimpleNamespace(), SimpleNamespace(), "ko",
            )
        self.assertEqual(gui.events.get_nowait(), ("ocr_quality_unavailable", None))


if __name__ == "__main__":
    import unittest
    unittest.main()
