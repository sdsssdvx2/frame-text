import unittest

from hardsub_extractor.cue_builder import _SegmentEvidence, _representative
from hardsub_extractor.models import VisualSegment


class RepresentativeSoftSupportTests(unittest.TestCase):
    def test_duration_weighted_medoid_beats_threshold_cliff(self):
        # Shape captured from a real QA failure: several OCR spellings describe
        # one persistent subtitle. A hard 0.82 cutoff gave the wrong near-match
        # slightly more binary support duration even though the correct reading
        # was more central to the full observation set. Production code must
        # resolve this generically; no phrase correction dictionary is used.
        rows = [
            (11.720044, 11.845167, "잔득 와있을까", 0.826120),
            (11.845167, 11.970289, "잔뜩 와있을까", 0.869157),
            (11.970289, 12.095422, "라있을까", 0.989710),
            (12.095422, 12.220544, "잔뜩와있을까", 0.775268),
            (12.220544, 12.345667, "잔득 와있을까", 0.862137),
            (12.345667, 12.470789, "잔뜩 와있을까", 0.856681),
            (12.470789, 12.595922, "라있을까", 0.987900),
            (12.595922, 12.721044, "잔독와있을까", 0.908080),
            (12.721044, 12.846167, "잔득 와있을까", 0.815280),
            (12.846167, 12.971289, "잔뜩 와있을까", 0.872077),
            (12.971289, 13.179833, "와있을까", 0.981710),
            (13.179833, 13.304956, "찬똑와있을까", 0.917840),
            (13.304956, 13.430089, "잔똑와 있을까", 0.905750),
            (13.430089, 13.555211, "잔똑와 있을까.", 0.884153),
        ]
        evidence = [
            _SegmentEvidence(
                segment=VisualSegment(start, end, []),
                text=text,
                confidence=confidence,
                supporters=1,
                local_accepted=False,
            )
            for start, end, text, confidence in rows
        ]

        representative = _representative(evidence)

        self.assertIsNotNone(representative)
        self.assertEqual(representative.text, "잔뜩 와있을까")

    def test_exact_surface_duration_preserves_repeated_spacing(self):
        # Semantic comparison intentionally ignores whitespace so spacing OCR
        # variants remain one cue. When two variants are semantically identical,
        # the surface form observed for longer should survive instead of a
        # shorter, slightly higher-confidence collapsed spelling.
        evidence = [
            _SegmentEvidence(
                segment=VisualSegment(5.631, 6.632, []),
                text="소원 수리함?",
                confidence=0.9842,
                supporters=3,
                local_accepted=True,
            ),
            _SegmentEvidence(
                segment=VisualSegment(6.632, 7.090, []),
                text="소원수리함?",
                confidence=0.9960,
                supporters=3,
                local_accepted=True,
            ),
        ]

        representative = _representative(evidence)

        self.assertIsNotNone(representative)
        self.assertEqual(representative.text, "소원 수리함?")

    def test_surface_duration_cannot_promote_different_ocr_spelling(self):
        # Live EpSite regression: a longer segment contained one inserted OCR
        # syllable ("확인을음"), while the surrounding observations agreed on
        # "확인을". Surface-form duration must only decide spacing/layout among
        # candidates with identical semantic comparison keys; it must not make
        # a genuine character error outrank the medoid.
        evidence = [
            _SegmentEvidence(
                segment=VisualSegment(13.113456, 14.573256, []),
                text="어떻게\n확인을음해야 하나",
                confidence=0.930324,
                supporters=3,
                local_accepted=True,
            ),
            _SegmentEvidence(
                segment=VisualSegment(14.573256, 14.698378, []),
                text="어떻게\n확인을 해야하나.",
                confidence=0.944121,
                supporters=1,
                local_accepted=True,
            ),
            _SegmentEvidence(
                segment=VisualSegment(14.698378, 15.532544, []),
                text="어떻게\n확인을 해야하나",
                confidence=0.958174,
                supporters=3,
                local_accepted=True,
            ),
        ]

        representative = _representative(evidence)

        self.assertIsNotNone(representative)
        self.assertEqual(representative.text, "어떻게\n확인을 해야하나")


if __name__ == "__main__":
    unittest.main()
