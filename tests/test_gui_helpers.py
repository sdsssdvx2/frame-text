import unittest

from hardsub_extractor.gui import _user_error_summary


class GuiErrorSummaryTests(unittest.TestCase):
    def test_preserves_normal_exception_message(self):
        self.assertEqual(
            _user_error_summary(RuntimeError("connection reset")),
            "connection reset",
        )

    def test_uses_exception_type_when_message_is_empty(self):
        self.assertEqual(_user_error_summary(RuntimeError()), "RuntimeError")

    def test_truncates_very_long_message(self):
        result = _user_error_summary(RuntimeError("x" * 1000), limit=20)
        self.assertEqual(result, "x" * 20 + "...")


if __name__ == "__main__":
    unittest.main()
