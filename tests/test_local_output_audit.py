"""Read-only structural resume of existing local-original subtitle outputs.

No model/network/FFmpeg calls. Every fixture lives in a disposable directory;
actual SRT/ASS/raw tracks are published by production format writers.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from hardsub_extractor.local_batch import load_local_batch, run_local_batch
from hardsub_extractor.local_output_audit import audit_local_output
from hardsub_extractor.local_sidecar_subtitles import (
    _outputs, acquire_local_sidecar_text,
)
from hardsub_extractor.models import Cue
from hardsub_extractor.writers import write_subtitle_pair_exclusive


SRT = (
    "1\n00:00:01,000 --> 00:00:02,500\n첫 번째 대사\n\n"
    "2\n00:00:03,000 --> 00:00:04,500\n두 번째 대사\n"
).encode("utf-8")


class LocalRestartAuditTests(unittest.TestCase):
    def source(self, root):
        video = root / "영상01.mkv"
        video.write_bytes(b"fake local video, no decoding")
        sidecar = root / "영상01.ko.srt"
        sidecar.write_bytes(SRT)
        out = root / "saved" / "ep"
        out.parent.mkdir()
        result = acquire_local_sidecar_text(video, out, language="ko")
        self.assertIsNotNone(result)
        return video, sidecar, out

    def sha(self, paths):
        return {
            path: hashlib.sha256(path.read_bytes()).hexdigest()
            for path in paths if path.exists() and path.is_file()
        }

    def test_complete_original_sidecar_is_verified_and_skips_all_work(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            video, sidecar, out = self.source(root)
            pins = self.sha((*_outputs(out), video, sidecar))
            audit = audit_local_output(video, out, language="ko")
            self.assertEqual(
                (audit.state, audit.mode, audit.cue_count),
                ("verified", "local-sidecar-text", 2),
            )
            selected = load_local_batch([video], None, root / "batch")
            # Reuse one output namespace explicitly to exercise restart.
            selected[0] = type(selected[0])(video, out)
            with (
                mock.patch("hardsub_extractor.local_batch.acquire_local_embedded_text") as embedded,
                mock.patch("hardsub_extractor.local_batch.acquire_local_sidecar_text") as side,
                mock.patch("hardsub_extractor.local_batch.extract_hard_subtitles") as hard,
            ):
                results = run_local_batch(
                    selected, language="ko",
                    ocr_factory=lambda: self.fail("OCR model initialized"),
                )
            embedded.assert_not_called()
            side.assert_not_called()
            hard.assert_not_called()
            self.assertEqual(
                (results[0].mode, results[0].cues), ("existing_verified", 2),
            )
            self.assertEqual(self.sha((*_outputs(out), video, sidecar)), pins)

    def test_source_sidecar_changed_keeps_old_export_unmodified_pending(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            video, sidecar, out = self.source(root)
            pins = self.sha(_outputs(out))
            sidecar.write_bytes("1\n00:00:01,000 --> 00:00:02,500\n바뀐 내용\n".encode("utf-8"))
            self.assertEqual(audit_local_output(video, out).state, "preserved")
            saved = load_local_batch([video], None, root / "unrelated")[0]
            saved = type(saved)(saved.source, out)
            with mock.patch(
                "hardsub_extractor.local_batch.acquire_local_embedded_text",
            ) as embedded:
                result = run_local_batch([saved], ocr_factory=lambda: None)
            self.assertEqual(result[0].mode, "existing_preserved")
            embedded.assert_not_called()
            self.assertEqual(self.sha(_outputs(out)), pins)

    def test_missing_source_sidecar_is_not_claimed_verified(self):
        with tempfile.TemporaryDirectory() as td:
            video, sidecar, out = self.source(Path(td))
            sidecar.unlink()
            self.assertEqual(audit_local_output(video, out).state, "preserved")

    def test_partial_manifest_and_corrupt_srt_preserved(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            video, sidecar, out = self.source(root)
            manifest = out.with_name(out.name + ".source.json")
            manifest.unlink()
            self.assertEqual(audit_local_output(video, out).state, "preserved")
            raw = out.with_suffix(".srt")
            raw.write_bytes(b"invalid subtitle timestamp, no fabricated completeness")
            self.assertEqual(audit_local_output(video, out).state, "preserved")

    def test_unsafe_manifest_locator_is_not_followed(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            video, sidecar, out = self.source(root)
            manifest = out.with_name(out.name + ".source.json")
            obj = json.loads(manifest.read_text("utf-8"))
            obj["extracted_stream_file"] = "../../outside-private.srt"
            obj["source_sidecar_name"] = "../../outside-private.srt"
            manifest.write_text(json.dumps(obj), "utf-8")
            with mock.patch(
                "hardsub_extractor.local_output_audit._safe_read",
                wraps=__import__(
                    "hardsub_extractor.public_output_audit", fromlist=["_safe_read"],
                )._safe_read,
            ) as read:
                result = audit_local_output(video, out)
            self.assertEqual(result.state, "preserved")
            self.assertTrue(all(".." not in str(call.args[0]) for call in read.call_args_list))

    def test_unmanifested_hard_ocr_pair_is_never_claimed_complete(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            video = root / "hard.mkv"
            video.write_bytes(b"synthetic")
            out = root / "result"
            write_subtitle_pair_exclusive([
                Cue(1, 2.5, "하드자막", 0.99),
            ], out)
            self.assertEqual(audit_local_output(video, out).state, "preserved")
            self.assertTrue(out.with_suffix(".srt").is_file())

    def test_embedded_schema1_structurally_verified_not_source_sha(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            video = root / "multi-language.mkv"
            video.write_bytes(b"container")
            out = root / "export"
            cues = [
                Cue(1, 2.5, "첫 번째 대사", 1),
                Cue(3, 4.5, "두 번째 대사", 1),
            ]
            srt, ass = write_subtitle_pair_exclusive(cues, out)
            raw = out.with_name(out.name + ".embedded.extracted.srt")
            raw.write_bytes(SRT)
            manifest = out.with_name(out.name + ".source.json")
            manifest.write_text(json.dumps({
                "schema": 1, "mode": "local-embedded-text",
                "language_requested": "ko", "language_tag": "kor",
                "stream_index": 1, "source_codec": "subrip",
                "is_default": True, "is_forced": False,
                "cue_count": 2, "source_video_modified": False,
                "ocr_applied": False,
                "source_stream_export":
                    "ffmpeg_decoded_srt_not_original_container_bytes",
                "output_timeline": "source_video",
                "extracted_stream_file": raw.name,
                "srt_file": srt.name, "ass_file": ass.name,
            }, ensure_ascii=False), encoding="utf-8")
            audit = audit_local_output(video, out)
            self.assertEqual(
                (audit.state, audit.mode, audit.cue_count),
                ("verified", "local-embedded-text", 2),
            )
            # Legacy schema1 DOES NOT contain a source-video SHA. Thus this
            # only proves structured output parity; no media version proof.
            self.assertFalse("source_sha256" in manifest.read_text("utf-8"))

    def test_conflicting_raw_modes_fail_closed(self):
        with tempfile.TemporaryDirectory() as td:
            video, sidecar, out = self.source(Path(td))
            out.with_name(out.name + ".embedded.extracted.srt").write_bytes(SRT)
            self.assertEqual(audit_local_output(video, out).state, "preserved")

    def test_dangling_symlink_does_not_turn_into_verified_or_absent(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            video, sidecar, out = self.source(root)
            manifest = out.with_name(out.name + ".source.json")
            manifest.unlink()
            try:
                manifest.symlink_to(root / "missing-private-metadata")
            except OSError:
                self.skipTest("symlink not permitted by Windows test host")
            self.assertEqual(audit_local_output(video, out).state, "preserved")


if __name__ == "__main__":
    unittest.main()
