"""Offline-only safety and scoring contracts for pinned AVL independent trial."""
from __future__ import annotations

from contextlib import ExitStack
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))

import verify_avl_public_natural as target


def _digest(body: bytes):
    return hashlib.sha256(body).hexdigest()


def _reference():
    return [
        target.Cue(0.0, 3.25, "first\nsubtitle"),
        target.Cue(3.416, 7.75, "second\nsubtitle"),
        target.Cue(8.041, 10.5, "third"),
        target.Cue(10.583, 13.208, "fourth\nsubtitle"),
    ]


class PinnedAVLTrialContracts(unittest.TestCase):
    def fixture(self, source: Path, output: Path):
        pos = b"public-caption-test"
        neg = b"public-uncaptioned-test"
        pins = {"caption": _digest(pos), "clean": _digest(neg)}
        media = {
            "caption": ("https://invalid.example/public-caption", len(pos), "mock"),
            "clean": ("https://invalid.example/public-clean", len(neg), "mock"),
        }
        refs = _reference()
        text_hash = target._reference_sha(refs)

        def fake_get_media(path, label):
            payload = pos if label == "caption" else neg
            path.write_bytes(payload)
            return pins[label]

        with ExitStack() as stack:
            stack.enter_context(mock.patch.object(target, "SOURCE_SHA256", pins))
            stack.enter_context(mock.patch.object(target, "PUBLISHED", media))
            stack.enter_context(mock.patch.object(target, "EXPECTED_TEXT_SHA256", text_hash))
            stack.enter_context(mock.patch.object(target, "_get_reference", return_value=(refs, "fake-source-revision")))
            stack.enter_context(mock.patch.object(target, "_get_media", side_effect=fake_get_media))
            report = target.prepare(source)
            self.assertTrue(report["media_source_sha256_verified"])
            self.assertFalse(report["third_party_ocr_executed"])
            output.mkdir()
            self._output(output, "caption", refs)
            self._output(output, "clean", [])
            yield pins, media, text_hash, refs

    @staticmethod
    def _output(out, label, cues):
        (out / (label + "-result.srt")).write_bytes(
            target._srt_payload(cues) if cues else b""
        )
        (out / (label + "-result.json")).write_text(
            json.dumps([
                {"start_ms": cue.start * 1000,
                 "end_ms": cue.end * 1000,
                 "text": cue.text}
                for cue in cues
            ], ensure_ascii=False),
            encoding="utf-8",
        )
        (out / (label + "-result.meta.json")).write_bytes(b"{}")

    def test_valid_paired_exports_match_exact_four_cues_and_blank(self):
        with tempfile.TemporaryDirectory() as d:
            src, out = Path(d) / "media", Path(d) / "results"
            for pins, media, text_hash, refs in self.fixture(src, out):
                with ExitStack() as stack:
                    stack.enter_context(mock.patch.object(target, "SOURCE_SHA256", pins))
                    stack.enter_context(mock.patch.object(target, "PUBLISHED", media))
                    stack.enter_context(mock.patch.object(target, "EXPECTED_TEXT_SHA256", text_hash))
                    result = target.score(src, out)
                self.assertEqual(result["score"]["exact_visible_text_and_wrap"], 4)
                self.assertEqual(result["score"]["strict_both_edges_250ms_and_text_wrap"], 4)
                self.assertEqual(result["score"]["blank_video_false_cues"], 0)
                self.assertTrue(result["same_pinned_public_videos_as_hse_baseline"])
                self.assertTrue(result["third_party_engine_score_from_actual_output_srt"])

    def test_preexisting_source_never_emptied_or_replaced(self):
        with tempfile.TemporaryDirectory() as d:
            src = Path(d) / "media"
            src.mkdir()
            keep = src / "keep.txt"
            keep.write_text("owner data", encoding="utf-8")
            with self.assertRaises(FileExistsError):
                target.prepare(src)
            self.assertEqual(keep.read_text(encoding="utf-8"), "owner data")

    def test_corrupt_media_rejected_before_accepting_export(self):
        with tempfile.TemporaryDirectory() as d:
            src, out = Path(d) / "media", Path(d) / "results"
            for pins, media, text_hash, _ in self.fixture(src, out):
                (src / "caption.webm").write_bytes(b"malicious substitution")
                with ExitStack() as stack:
                    stack.enter_context(mock.patch.object(target, "SOURCE_SHA256", pins))
                    stack.enter_context(mock.patch.object(target, "PUBLISHED", media))
                    stack.enter_context(mock.patch.object(target, "EXPECTED_TEXT_SHA256", text_hash))
                    with self.assertRaisesRegex(ValueError, "source_video"):
                        target.score(src, out)

    def test_missing_or_symlinked_trial_export_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            src, out = Path(d) / "media", Path(d) / "results"
            for pins, media, text_hash, _ in self.fixture(src, out):
                export = out / "caption-result.srt"
                export.unlink()
                with ExitStack() as stack:
                    stack.enter_context(mock.patch.object(target, "SOURCE_SHA256", pins))
                    stack.enter_context(mock.patch.object(target, "PUBLISHED", media))
                    stack.enter_context(mock.patch.object(target, "EXPECTED_TEXT_SHA256", text_hash))
                    with self.assertRaisesRegex(ValueError, "missing_or_symlinked"):
                        target.score(src, out)
                    try:
                        export.symlink_to(src / "reference.srt")
                    except (OSError, NotImplementedError):
                        # Windows without Developer Mode may deny symlink
                        # creation. Missing-file rejection already ran;
                        # never weaken the production symlink guard.
                        return
                    with self.assertRaisesRegex(ValueError, "missing_or_symlinked"):
                        target.score(src, out)

    def test_reference_timing_modified_without_text_change_is_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            src, out = Path(d) / "media", Path(d) / "results"
            for pins, media, text_hash, refs in self.fixture(src, out):
                changed = list(refs)
                changed[0] = target.Cue(0.5, refs[0].end, refs[0].text)
                (src / "reference.srt").write_bytes(target._srt_payload(changed))
                with ExitStack() as stack:
                    stack.enter_context(mock.patch.object(target, "SOURCE_SHA256", pins))
                    stack.enter_context(mock.patch.object(target, "PUBLISHED", media))
                    stack.enter_context(mock.patch.object(target, "EXPECTED_TEXT_SHA256", text_hash))
                    with self.assertRaisesRegex(ValueError, "timing_or_layout"):
                        target.score(src, out)

    def test_export_json_disagreement_never_passes_subtitle_score(self):
        with tempfile.TemporaryDirectory() as d:
            src, out = Path(d) / "media", Path(d) / "results"
            for pins, media, text_hash, refs in self.fixture(src, out):
                export = out / "caption-result.json"
                altered = [
                    {"start_ms": cue.start * 1000,
                     "end_ms": cue.end * 1000,
                     "text": cue.text}
                    for cue in refs
                ]
                altered[0]["text"] = "wrong output"
                export.write_text(json.dumps(altered), encoding="utf-8")
                with ExitStack() as stack:
                    stack.enter_context(mock.patch.object(target, "SOURCE_SHA256", pins))
                    stack.enter_context(mock.patch.object(target, "PUBLISHED", media))
                    stack.enter_context(mock.patch.object(target, "EXPECTED_TEXT_SHA256", text_hash))
                    with self.assertRaisesRegex(ValueError, "avl_srt_json_cue_content_mismatch"):
                        target.score(src, out)

    def test_malformed_or_oversized_srt_fails_closed(self):
        with tempfile.TemporaryDirectory() as d:
            src, out = Path(d) / "media", Path(d) / "results"
            for pins, media, text_hash, _ in self.fixture(src, out):
                export = out / "caption-result.srt"
                with ExitStack() as stack:
                    stack.enter_context(mock.patch.object(target, "SOURCE_SHA256", pins))
                    stack.enter_context(mock.patch.object(target, "PUBLISHED", media))
                    stack.enter_context(mock.patch.object(target, "EXPECTED_TEXT_SHA256", text_hash))
                    export.write_bytes(b"not a subtitle file")
                    with self.assertRaises((ValueError, UnicodeError)):
                        target.score(src, out)
                    export.write_bytes(b"1" * (target.MAX_EXPORT_BYTES + 1))
                    with self.assertRaisesRegex(ValueError, "trial_file_size_invalid"):
                        target.score(src, out)


if __name__ == "__main__":
    unittest.main()
