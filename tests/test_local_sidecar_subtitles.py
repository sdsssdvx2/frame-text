"""No-network integration contracts for language-tagged local sidecar priority."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from hardsub_extractor.local_sidecar_subtitles import acquire_local_sidecar_text
from hardsub_extractor.writers import write_subtitle_pair_exclusive

KO = b"1\n00:00:01,000 --> 00:00:02,250\n" + "안녕하세요".encode() + b"\n"
EN = b"1\n00:00:01,000 --> 00:00:02,250\nHello there\n"
VTT = b"WEBVTT\n\n00:00:01.000 --> 00:00:02.250\nHello there\n"


class LocalSidecarTextTests(unittest.TestCase):
    def fixture(self, root: Path):
        video = root / "에피소드 01.mkv"
        video.write_bytes(b"UNMODIFIED-VIDEO")
        return video

    def test_korean_sidecar_before_ocr_and_preserve_raw_provenance(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            video = self.fixture(root)
            raw = video.with_name(video.stem + ".ko.srt")
            raw.write_bytes(KO)
            out = root / "output" / "subtitle"
            result = acquire_local_sidecar_text(video, out, language="ko")
            self.assertEqual(result.mode, "local-sidecar-text")
            self.assertEqual(result.cue_count, 1)
            self.assertEqual(result.track_language, "ko")
            self.assertEqual(result.extracted_track_path.read_bytes(), KO)
            self.assertIn("안녕하세요", result.srt_path.read_text("utf-8"))
            self.assertIn("안녕하세요", result.ass_path.read_text("utf-8"))
            report = json.loads(result.provenance_path.read_text("utf-8"))
            self.assertEqual(report["original_sha256"], hashlib.sha256(KO).hexdigest())
            self.assertTrue(report["original_bytes_retained"])
            self.assertFalse(report["ocr_applied"])
            self.assertEqual(report["source_sidecar_name"], raw.name)
            self.assertNotIn(str(root), result.provenance_path.read_text("utf-8"))
            self.assertEqual(video.read_bytes(), b"UNMODIFIED-VIDEO")
            self.assertEqual(raw.read_bytes(), KO)

    def test_webvtt_original_preserved_with_line_timing(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            video = self.fixture(root)
            video.with_name(video.stem + ".en.vtt").write_bytes(VTT)
            result = acquire_local_sidecar_text(video, root / "out", language="en")
            self.assertEqual(result.track_language, "en")
            self.assertEqual(result.extracted_track_path.suffix, ".vtt")
            self.assertEqual(result.extracted_track_path.read_bytes(), VTT)
            self.assertIn("00:00:01,000 --> 00:00:02,250", result.srt_path.read_text("utf-8"))

    def test_language_tag_does_not_make_wrong_language_valid(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            video = self.fixture(root)
            video.with_name(video.stem + ".ko.srt").write_bytes(EN)
            self.assertIsNone(acquire_local_sidecar_text(video, root / "out", language="ko"))
            self.assertFalse((root / "out.srt").exists())

    def test_no_untagged_or_different_language_guess(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            video = self.fixture(root)
            video.with_suffix(".srt").write_bytes(EN)
            video.with_name(video.stem + ".en.srt").write_bytes(EN)
            self.assertIsNone(acquire_local_sidecar_text(video, root / "out", language="ja"))
            self.assertFalse((root / "out.source.json").exists())

    def test_existing_outputs_preserved_before_read(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            video = self.fixture(root)
            video.with_name(video.stem + ".ko.srt").write_bytes(KO)
            protected = root / "out.source.json"
            protected.write_bytes(b"PROTECTED")
            with self.assertRaises(FileExistsError):
                acquire_local_sidecar_text(video, root / "out", language="ko")
            self.assertEqual(protected.read_bytes(), b"PROTECTED")
            self.assertFalse((root / "out.srt").exists())

    def test_existing_raw_extension_preserved_even_for_other_source_format(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            video = self.fixture(root)
            video.with_name(video.stem + ".en.vtt").write_bytes(VTT)
            protected = root / "out.sidecar.original.srt"
            protected.write_bytes(b"DO-NOT-REPLACE")
            with self.assertRaises(FileExistsError):
                acquire_local_sidecar_text(video, root / "out", language="en")
            self.assertEqual(protected.read_bytes(), b"DO-NOT-REPLACE")
            self.assertFalse((root / "out.ass").exists())

    def test_partial_provenance_collision_rolls_back_only_new_exports(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            video = self.fixture(root)
            video.with_name(video.stem + ".ko.srt").write_bytes(KO)
            out = root / "out"
            provenance = root / "out.source.json"
            def concurrent_owner(cues, base):
                pair = write_subtitle_pair_exclusive(cues, base)
                provenance.write_bytes(b"CONCURRENT-OWNER")
                return pair
            with mock.patch(
                "hardsub_extractor.local_sidecar_subtitles.write_subtitle_pair_exclusive",
                side_effect=concurrent_owner,
            ):
                with self.assertRaises(FileExistsError):
                    acquire_local_sidecar_text(video, out, language="ko")
            self.assertEqual(provenance.read_bytes(), b"CONCURRENT-OWNER")
            self.assertFalse((root / "out.srt").exists())
            self.assertFalse((root / "out.ass").exists())
            self.assertFalse((root / "out.sidecar.original.srt").exists())

    def test_embedded_priority_rejects_existing_sidecar_output_before_probe(self):
        from hardsub_extractor.local_soft_subtitles import acquire_local_embedded_text
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            video = self.fixture(root)
            user_raw = root / "out.sidecar.original.vtt"
            user_raw.write_bytes(b"USER_SIDECAR_BYTES")
            with mock.patch("hardsub_extractor.local_soft_subtitles._run_bounded") as proc:
                with self.assertRaises(FileExistsError):
                    acquire_local_embedded_text(video, root / "out", language="en")
            proc.assert_not_called()
            self.assertEqual(user_raw.read_bytes(), b"USER_SIDECAR_BYTES")

    def test_gui_preflight_rejects_existing_sidecar_before_start(self):
        from hardsub_extractor.gui import ExtractorGUI
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            video = self.fixture(root)
            output = root / "output"
            raw = root / "output.sidecar.original.srt"
            raw.write_bytes(b"DO_NOT_TOUCH")
            gui = ExtractorGUI.__new__(ExtractorGUI)
            class Var:
                def __init__(self, value):
                    self.value = value
                def get(self):
                    return self.value
            gui.source_mode_var = Var("local")
            gui.url_var = Var(str(video))
            gui.out_var = Var(str(output))
            gui.ocr_language_var = Var("ko")
            with self.assertRaises(ValueError):
                gui._validate()
            self.assertEqual(raw.read_bytes(), b"DO_NOT_TOUCH")
            self.assertFalse((root / "output.srt").exists())

    def test_oversized_sidecar_is_rejected_without_opening_it(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            video = self.fixture(root)
            huge = video.with_name(video.stem + ".en.srt")
            with huge.open("wb") as stream:
                stream.seek(20 * 1024 * 1024)
                stream.write(b"x")
            with mock.patch.object(Path, "open", side_effect=AssertionError("size cap bypassed")):
                result = acquire_local_sidecar_text(video, root / "out", language="en")
            self.assertIsNone(result)
            self.assertFalse((root / "out.source.json").exists())

    def test_user_selected_untagged_file_outside_video_folder_wins(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            video = self.fixture(root)
            selected = root / "downloaded" / "한국어 자막.srt"
            selected.parent.mkdir()
            selected.write_bytes(KO)
            result = acquire_local_sidecar_text(
                video, root / "output" / "captions", language="ko",
                selected_path=selected,
            )
            self.assertEqual(result.mode, "local-selected-text")
            self.assertEqual(result.cue_count, 1)
            self.assertEqual(result.extracted_track_path.read_bytes(), KO)
            report = json.loads(result.provenance_path.read_text("utf-8"))
            self.assertEqual(report["selection"], "explicit")
            self.assertEqual(report["source_sidecar_name"], "한국어 자막.srt")
            self.assertEqual(report["original_sha256"], hashlib.sha256(KO).hexdigest())
            self.assertNotIn(str(root), result.provenance_path.read_text("utf-8"))
            self.assertEqual(selected.read_bytes(), KO)

    def test_explicit_invalid_source_never_falls_back_to_auto_or_writes(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            video = self.fixture(root)
            video.with_name(video.stem + ".ko.srt").write_bytes(KO)
            damaged = root / "not-a-subtitle.srt"
            damaged.write_bytes(b"not a subtitle")
            other = root / "wrong-language.srt"
            other.write_bytes(EN)
            wrong_ext = root / "not-a-subtitle.txt"
            wrong_ext.write_bytes(KO)
            for selected in (damaged, other, wrong_ext, root / "missing.srt"):
                with self.subTest(selected=selected.name):
                    with self.assertRaises(ValueError):
                        acquire_local_sidecar_text(
                            video, root / "out", language="ko",
                            selected_path=selected,
                        )
                    self.assertFalse((root / "out.srt").exists())
                    self.assertFalse((root / "out.source.json").exists())

    def test_explicit_select_supports_untagged_english_webvtt(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            video = self.fixture(root)
            selected = root / "captions.vtt"
            selected.write_bytes(VTT)
            result = acquire_local_sidecar_text(
                video, root / "out", language="en", selected_path=selected,
            )
            self.assertEqual(result.mode, "local-selected-text")
            self.assertIn("Hello there", result.srt_path.read_text("utf-8"))

    def test_unsupported_language_rejected_without_output(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            video = self.fixture(root)
            with self.assertRaises(ValueError):
                acquire_local_sidecar_text(video, root / "out", language="xx")
            self.assertFalse((root / "out.srt").exists())


if __name__ == "__main__":
    unittest.main()
