from pathlib import Path
import importlib.util
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("rapidocr_acceptance", ROOT / "tools" / "rapidocr_acceptance.py")
MOD = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
SPEC.loader.exec_module(MOD)


class AcceptanceHelperTests(unittest.TestCase):
    def test_parse_srt_retains_text_and_timing(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "x.srt"
            p.write_text(
                "1\n00:00:01,250 --> 00:00:02,875\n첫 줄\n둘째 줄\n",
                encoding="utf-8",
            )
            got = MOD.parse_srt(p)
        self.assertEqual(len(got), 1)
        self.assertEqual(got[0]["text"], "첫 줄\n둘째 줄")
        self.assertAlmostEqual(got[0]["start"], 1.25, places=3)
        self.assertAlmostEqual(got[0]["end"], 2.875, places=3)

    def test_srt_seconds_supports_hour_boundary(self):
        self.assertAlmostEqual(MOD._srt_seconds("01:02:03,004"), 3723.004, places=3)


if __name__ == "__main__":
    unittest.main()
