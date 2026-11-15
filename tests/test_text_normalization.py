import unittest

from hardsub_extractor.ocr.base import normalize_text


class TextNormalizationTests(unittest.TestCase):
    def test_standalone_compatibility_jamo_stays_readable(self):
        self.assertEqual(normalize_text("소원수리함ㅇㅇ"), "소원수리함ㅇㅇ")

    def test_conjoining_standalone_jamo_is_rendered_as_compatibility_jamo(self):
        self.assertEqual(normalize_text("소원수리함ᄋᄋ"), "소원수리함ㅇㅇ")

    def test_real_decomposed_hangul_syllable_is_composed_before_jamo_cleanup(self):
        self.assertEqual(normalize_text("가나"), "가나")


if __name__ == "__main__":
    unittest.main()
