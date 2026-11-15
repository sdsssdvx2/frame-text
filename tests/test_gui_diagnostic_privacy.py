from __future__ import annotations

import queue
import unittest
from unittest import mock

from hardsub_extractor.gui import ExtractorGUI, _safe_gui_error_text, _user_error_summary


class GuiDiagnosticPrivacyTests(unittest.TestCase):
    def test_diagnostics_default_to_opt_in_without_initializing_tk(self):
        # Check the opt-in defaults, not the total number of GUI switches:
        # unrelated local OCR experiments may add BooleanVars over time.
        def fake_boolean_var(*, value=False):
            variable = mock.Mock()
            variable.get.return_value = value
            return variable

        with (
            mock.patch("hardsub_extractor.gui.tk.StringVar"),
            mock.patch(
                "hardsub_extractor.gui.tk.BooleanVar",
                side_effect=fake_boolean_var,
            ),
            mock.patch("hardsub_extractor.gui.auto_update_enabled", return_value=False),
            mock.patch.object(ExtractorGUI, "_build"),
            mock.patch.object(ExtractorGUI, "_refresh_runtime_status"),
        ):
            gui = ExtractorGUI(mock.Mock())

        self.assertTrue(gui.video_ocr_var.get())  # public-video OCR allowed
        for name in (
            "diag_var",  # private diagnostics require explicit consent
            "native_vsf_var",
            "native_split_var",
            "focus_bottom_two_var",
            "caption_islands_var",
            "light_caption_temporal_var",
            "seconv54_var",
            "managed_caption_updates_var",
        ):
            with self.subTest(variable=name):
                self.assertFalse(getattr(gui, name).get())

        self.assertFalse(gui._managed_caption_auto_on_start)
        self.assertFalse(gui._active_native_split_opt_in)
        self.assertFalse(gui._active_caption_islands)
        self.assertFalse(gui._active_light_caption_temporal)
        self.assertFalse(gui._active_seconv54_opt_in)
        self.assertTrue(gui._requested_video_ocr_enabled)

    def test_error_redacts_full_media_url_query_and_sensitive_headers(self):
        unsafe = (
            "network failed https://player.example/watch?token=SENSITIVE_KEY&x=1\n"
            "Authorization: Bearer SENSITIVE_BEARER\n"
            "Cookie: session=SENSITIVE_COOKIE\n"
            "x-api-key: SENSITIVE_API\n"
            "ordinary context remains"
        )
        redacted = _safe_gui_error_text(unsafe)
        for secret in ("SENSITIVE_KEY", "SENSITIVE_BEARER", "SENSITIVE_COOKIE", "SENSITIVE_API"):
            self.assertNotIn(secret, redacted)
        self.assertNotIn("player.example", redacted)
        self.assertIn("ordinary context remains", redacted)
        self.assertIn("<url>", redacted)
        self.assertIn("<redacted-header>", redacted)

        summary = _user_error_summary(RuntimeError(unsafe))
        self.assertNotIn("SENSITIVE", summary)
        self.assertIn("network failed", summary)

    def test_worker_failure_does_not_place_private_url_in_gui_event(self):
        gui = ExtractorGUI.__new__(ExtractorGUI)
        gui.events = queue.Queue()
        with mock.patch(
            "hardsub_extractor.gui.AutoEpSiteClient",
            side_effect=RuntimeError("site failed https://video.example/watch?auth=SENSITIVE_QUERY"),
        ):
            gui._worker("epsite", "https://episode.example/episode", None, mock.Mock(), False)

        items = list(gui.events.queue)
        self.assertEqual(items[0][0], "log")
        errors = [payload for kind, payload in items if kind == "error"]
        self.assertEqual(len(errors), 1)
        self.assertNotIn("SENSITIVE_QUERY", str(errors[0]))
        self.assertNotIn("video.example", str(errors[0]))
        self.assertIn("<url>", errors[0]["summary"])
        self.assertIn("<url>", errors[0]["traceback"])


if __name__ == "__main__":
    unittest.main()
