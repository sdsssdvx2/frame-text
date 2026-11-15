"""Local-only optional existing-editor handoff, no real processes."""
from __future__ import annotations

from pathlib import Path
import tempfile
import unittest
from unittest import mock

from hardsub_extractor.subtitle_edit_handoff import (
    find_existing_subtitle_edit, launch_subtitle_edit_proofreading,
    validate_handoff, ProofreadHandoffError,
)


SRT = ("1\n00:00:01,000 --> 00:00:02,000\n안녕하세요\n\n"
       "2\n00:00:03,000 --> 00:00:05,000\n두 줄\n두 번째 줄\n\n").encode()


class SubtitleEditProofreadHandoffTests(unittest.TestCase):
    def inputs(self, root):
        video = root / "one movie.mkv"
        subtitle = root / "one movie.srt"
        exe = root / "SubtitleEdit.exe"
        video.write_bytes(b"private-media-intact")
        subtitle.write_bytes(SRT)
        exe.write_bytes(b"synthetic editor placeholder, never executed")
        return video, subtitle, exe

    def test_notepad_fallback_creates_only_its_first_owner_proofread_copy(self):
        from hardsub_extractor.subtitle_edit_handoff import launch_notepad_proofreading
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            video, original, _ = self.inputs(root)
            editor = root / "notepad.exe"
            editor.write_bytes(b"never executed test executable")
            spawned = mock.Mock()
            first = launch_notepad_proofreading(
                video, original, executable=editor, opener=spawned,
            )
            self.assertTrue(first.newly_created)
            self.assertEqual(first.working_copy.read_bytes(), SRT)
            spawned.assert_called_once_with(
                [str(editor.resolve()), str(root / "one movie-proofread.srt")],
                cwd=str(root), shell=False,
            )
            corrected = SRT.replace("안녕하세요".encode(), "수정한 대사".encode())
            first.working_copy.write_bytes(corrected)
            second = launch_notepad_proofreading(
                video, original, executable=editor, opener=mock.Mock(),
            )
            self.assertFalse(second.newly_created)
            self.assertEqual(second.working_copy.read_bytes(), corrected)
            self.assertEqual(original.read_bytes(), SRT)
            self.assertEqual(video.read_bytes(), b"private-media-intact")

    def test_notepad_refuses_other_executable_and_existing_hardlinked_copy(self):
        from hardsub_extractor.subtitle_edit_handoff import (
            launch_notepad_proofreading, ProofreadHandoffError,
        )
        import os
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            video, original, _ = self.inputs(root)
            other = root / "unknown.exe"
            other.write_bytes(b"untrusted")
            with self.assertRaises(ProofreadHandoffError):
                launch_notepad_proofreading(video, original, executable=other,
                                           opener=mock.Mock())
            self.assertFalse((root / "one movie-proofread.srt").exists())
            notepad = root / "notepad.exe"
            notepad.write_bytes(b"test")
            original_copy = root / "one movie-proofread.srt"
            try:
                os.link(original, original_copy)
            except (OSError, NotImplementedError):
                self.skipTest("hardlink unsupported")
            with self.assertRaises(ProofreadHandoffError):
                launch_notepad_proofreading(
                    video, original, executable=notepad, opener=mock.Mock()
                )
            self.assertEqual(original.read_bytes(), SRT)

    def test_notepad_launch_failure_only_removes_new_copy(self):
        from hardsub_extractor.subtitle_edit_handoff import launch_notepad_proofreading
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            video, original, _ = self.inputs(root)
            editor = root / "notepad.exe"
            editor.write_bytes(b"synthetic")
            with self.assertRaises(OSError):
                launch_notepad_proofreading(
                    video, original, executable=editor,
                    opener=mock.Mock(side_effect=OSError("launch failed")),
                )
            self.assertFalse((root / "one movie-proofread.srt").exists())
            existing = root / "one movie-proofread.srt"
            existing.write_bytes(b"human text not to be discarded")
            with self.assertRaises(OSError):
                launch_notepad_proofreading(
                    video, original, executable=editor,
                    opener=mock.Mock(side_effect=OSError("launch failed")),
                )
            self.assertEqual(existing.read_bytes(), b"human text not to be discarded")
            self.assertEqual(original.read_bytes(), SRT)

    def test_gui_falls_back_to_notepad_only_after_approval(self):
        from hardsub_extractor.gui import ExtractorGUI
        from hardsub_extractor.subtitle_edit_handoff import ProofreadHandoff
        class Var:
            def __init__(self, value): self.value = value
            def get(self): return self.value
            def set(self, value): self.value = value
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            video, original, _ = self.inputs(root)
            editor = root / "notepad.exe"
            editor.write_bytes(b"test")
            gui = ExtractorGUI.__new__(ExtractorGUI)
            gui._pending_close = False
            gui.worker = None
            gui._tab_capture_thread = None
            gui.source_mode_var = Var("local")
            gui.url_var = Var(str(video))
            gui.out_var = Var(str(original.with_suffix("")))
            gui.status_var = Var("")
            gui._log = mock.Mock()
            with (mock.patch(
                    "hardsub_extractor.subtitle_edit_handoff.find_existing_subtitle_edit",
                    return_value=None,
                  ),
                  mock.patch(
                    "hardsub_extractor.subtitle_edit_handoff.find_windows_notepad",
                    return_value=editor,
                  ),
                  mock.patch("hardsub_extractor.gui.messagebox.askyesno",
                             return_value=False) as approved,
                  mock.patch(
                    "hardsub_extractor.subtitle_edit_handoff.launch_notepad_proofreading"
                  ) as opened):
                gui._start_external_proofread()
            self.assertIn("메모장", approved.call_args.args[0])
            opened.assert_not_called()
            receipt = ProofreadHandoff(video, original,
                       root / "one movie-proofread.srt", editor, True)
            with (mock.patch(
                    "hardsub_extractor.subtitle_edit_handoff.find_existing_subtitle_edit",
                    return_value=None,
                  ),
                  mock.patch(
                    "hardsub_extractor.subtitle_edit_handoff.find_windows_notepad",
                    return_value=editor,
                  ),
                  mock.patch(
                    "hardsub_extractor.subtitle_edit_handoff.launch_notepad_proofreading",
                    return_value=receipt,
                  ) as opened,
                  mock.patch("hardsub_extractor.gui.messagebox.askyesno",
                             return_value=True)):
                gui._start_external_proofread()
            opened.assert_called_once_with(video.resolve(), original.resolve(),
                                           executable=editor)
            self.assertIn("메모장", gui.status_var.get())
            self.assertEqual(original.read_bytes(), SRT)
            self.assertNotIn(str(root), repr(gui._log.call_args_list))

    def test_existing_subtitle_edit_is_preferred_to_notepad(self):
        from hardsub_extractor.gui import ExtractorGUI
        from hardsub_extractor.subtitle_edit_handoff import ProofreadHandoff
        class Var:
            def __init__(self, value): self.value = value
            def get(self): return self.value
            def set(self, value): self.value = value
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            video, original, editor = self.inputs(root)
            gui = ExtractorGUI.__new__(ExtractorGUI)
            gui._pending_close = False
            gui.worker = None
            gui._tab_capture_thread = None
            gui.source_mode_var = Var("local")
            gui.url_var = Var(str(video))
            gui.out_var = Var(str(original.with_suffix("")))
            gui.status_var = Var("")
            gui._log = mock.Mock()
            receipt = ProofreadHandoff(video, original,
                root / "one movie-proofread.srt", editor, True)
            with (mock.patch(
                    "hardsub_extractor.subtitle_edit_handoff.find_existing_subtitle_edit",
                    return_value=editor,
                  ),
                  mock.patch(
                    "hardsub_extractor.subtitle_edit_handoff.find_windows_notepad"
                  ) as fallback,
                  mock.patch(
                    "hardsub_extractor.subtitle_edit_handoff.launch_subtitle_edit_proofreading",
                    return_value=receipt,
                  ) as launch,
                  mock.patch("hardsub_extractor.gui.messagebox.askyesno",
                             return_value=True)):
                gui._start_external_proofread()
            fallback.assert_not_called()
            launch.assert_called_once_with(video.resolve(), original.resolve(),
                                           executable=editor.resolve())
            self.assertIn("Subtitle Edit", gui.status_var.get())

    def test_existing_installed_editor_found_without_network(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            exe = root / "Program Files" / "Subtitle Edit" / "SubtitleEdit.exe"
            exe.parent.mkdir(parents=True)
            exe.write_bytes(b"dummy")
            found = find_existing_subtitle_edit(
                env={"ProgramFiles": str(root / "Program Files"), "PATH": ""},
                home=root,
            )
            self.assertEqual(found, exe.resolve())

    def test_missing_editor_has_no_implicit_download(self):
        with tempfile.TemporaryDirectory() as tmp:
            with mock.patch(
                "hardsub_extractor.subtitle_edit_handoff.shutil.which",
                return_value=None,
            ):
                found = find_existing_subtitle_edit(
                    env={"PATH": ""}, home=Path(tmp)
                )
            self.assertIsNone(found)

    def test_uses_official_startup_args_and_keeps_original_untouched(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            video, subtitle, exe = self.inputs(root)
            opener = mock.Mock()
            r = launch_subtitle_edit_proofreading(
                video, subtitle, executable=exe, opener=opener,
            )
            opener.assert_called_once_with(
                [str(exe.resolve()),
                 str(root / "one movie-proofread.srt"),
                 "/video:" + str(video.resolve())],
                cwd=str(exe.resolve().parent), shell=False,
            )
            self.assertTrue(r.newly_created)
            self.assertEqual(r.working_copy.read_bytes(), SRT)
            self.assertEqual(subtitle.read_bytes(), SRT)
            self.assertEqual(video.read_bytes(), b"private-media-intact")
            self.assertEqual(
                sorted(p.name for p in root.iterdir()),
                ["SubtitleEdit.exe", "one movie-proofread.srt",
                 "one movie.mkv", "one movie.srt"],
            )

    def test_reopens_existing_human_edits_without_overwrite(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            video, subtitle, exe = self.inputs(root)
            edited = root / "one movie-proofread.srt"
            edited.write_text("user corrected already", encoding="utf-8")
            r = launch_subtitle_edit_proofreading(
                video, subtitle, executable=exe, opener=mock.Mock(),
            )
            self.assertFalse(r.newly_created)
            self.assertEqual(edited.read_text(encoding="utf-8"),
                             "user corrected already")
            self.assertEqual(subtitle.read_bytes(), SRT)

    def test_existing_hardlink_copy_is_not_opened_for_editing(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            video, subtitle, exe = self.inputs(root)
            review = root / "one movie-proofread.srt"
            import os
            try:
                os.link(subtitle, review)
            except (OSError, NotImplementedError):
                self.skipTest("hardlinks unsupported")
            opener = mock.Mock()
            with self.assertRaises(ProofreadHandoffError):
                launch_subtitle_edit_proofreading(
                    video, subtitle, executable=exe, opener=opener,
                )
            opener.assert_not_called()
            self.assertEqual(subtitle.read_bytes(), SRT)

    def test_corrected_srt_to_ass_preserves_both_originals(self):
        from hardsub_extractor.subtitle_edit_handoff import export_proofread_ass
        from hardsub_extractor.subtitle_formats import parse_subtitle_bytes
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            video, original, exe = self.inputs(root)
            working = root / "one movie-proofread.srt"
            working.write_bytes(SRT.replace("안녕하세요".encode(), "교정된 자막".encode()))
            baseline = original.read_bytes()
            receipt = export_proofread_ass(original)
            self.assertFalse(receipt.already_present)
            self.assertEqual(receipt.cue_count, 2)
            self.assertTrue(receipt.ass_path.name.startswith("one movie-proofread-"))
            self.assertEqual(receipt.ass_path.suffix, ".ass")
            self.assertEqual(original.read_bytes(), baseline)
            self.assertEqual(working.read_bytes(), SRT.replace(
                "안녕하세요".encode(), "교정된 자막".encode()))
            fmt, cues = parse_subtitle_bytes(
                receipt.ass_path.read_bytes(), source_url="corrected.ass"
            )
            self.assertEqual(fmt, "ass")
            self.assertEqual([cue.text for cue in cues],
                             ["교정된 자막", "두 줄\n두 번째 줄"])

    def test_same_correction_is_idempotent_and_does_not_spawn_versions(self):
        from hardsub_extractor.subtitle_edit_handoff import export_proofread_ass
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _, original, _ = self.inputs(root)
            draft = root / "one movie-proofread.srt"
            draft.write_bytes(SRT)
            first = export_proofread_ass(original)
            second = export_proofread_ass(original)
            self.assertTrue(second.already_present)
            self.assertEqual(first.ass_path, second.ass_path)
            self.assertEqual(len(list(root.glob("*-proofread-*.ass"))), 1)

    def test_new_human_correction_creates_distinct_ass_without_overwrite(self):
        from hardsub_extractor.subtitle_edit_handoff import export_proofread_ass
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _, original, _ = self.inputs(root)
            draft = root / "one movie-proofread.srt"
            draft.write_bytes(SRT)
            first = export_proofread_ass(original)
            saved = first.ass_path.read_bytes()
            draft.write_bytes(SRT.replace("안녕하세요".encode(), "다시 교정".encode()))
            second = export_proofread_ass(original)
            self.assertNotEqual(first.ass_path, second.ass_path)
            self.assertEqual(first.ass_path.read_bytes(), saved)
            self.assertEqual(original.read_bytes(), SRT)

    def test_incomplete_edited_srt_block_refused_without_any_ass(self):
        from hardsub_extractor.subtitle_edit_handoff import (
            export_proofread_ass, ProofreadHandoffError,
        )
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _, original, _ = self.inputs(root)
            draft = root / "one movie-proofread.srt"
            draft.write_bytes(SRT + b"\n3\nBROKEN WITHOUT TIMECODE\nTEXT\n")
            with self.assertRaises(ProofreadHandoffError):
                export_proofread_ass(original)
            self.assertFalse(list(root.glob("*.ass")))
            self.assertEqual(original.read_bytes(), SRT)

    def test_same_hash_destination_with_different_bytes_refused(self):
        from hardsub_extractor.subtitle_edit_handoff import (
            export_proofread_ass, ProofreadHandoffError,
        )
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _, original, _ = self.inputs(root)
            draft = root / "one movie-proofread.srt"
            draft.write_bytes(SRT)
            first = export_proofread_ass(original)
            first.ass_path.write_bytes(b"existing user-authored ASS")
            with self.assertRaises(ProofreadHandoffError):
                export_proofread_ass(original)
            self.assertEqual(first.ass_path.read_bytes(),
                             b"existing user-authored ASS")

    def test_output_publication_failure_never_touches_original_or_draft(self):
        from hardsub_extractor.subtitle_edit_handoff import export_proofread_ass
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _, original, _ = self.inputs(root)
            draft = root / "one movie-proofread.srt"
            draft.write_bytes(SRT)
            with mock.patch(
                "hardsub_extractor.subtitle_edit_handoff.os.link",
                side_effect=OSError("filesystem rejects hardlinks"),
            ):
                with self.assertRaises(OSError):
                    export_proofread_ass(original)
            self.assertEqual(original.read_bytes(), SRT)
            self.assertEqual(draft.read_bytes(), SRT)
            self.assertFalse(list(root.glob("*.ass")))

    def test_ass_export_keeps_literal_braces_as_visible_text(self):
        # The actual ASS writer now escapes literal opening braces, and
        # export_proofread_ass independently PARSES the generated ASS before
        # publishing. Reject text loss; do not reject safe escaping itself.
        from hardsub_extractor.subtitle_edit_handoff import export_proofread_ass
        from hardsub_extractor.subtitle_formats import parse_subtitle_bytes
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _, original, _ = self.inputs(root)
            draft = root / "one movie-proofread.srt"
            corrected = SRT.replace(
                "안녕하세요".encode(), "{not a style}안녕하세요".encode(),
            )
            draft.write_bytes(corrected)
            receipt = export_proofread_ass(original)
            self.assertFalse(receipt.already_present)
            fmt, parsed = parse_subtitle_bytes(
                receipt.ass_path.read_bytes(), source_url="literal-braces.ass",
            )
            self.assertEqual(fmt, "ass")
            self.assertEqual(parsed[0].text, "{not a style}안녕하세요")
            self.assertEqual(parsed[1].text, "두 줄\n두 번째 줄")
            self.assertEqual(original.read_bytes(), SRT)
            self.assertEqual(draft.read_bytes(), corrected)

    def test_gui_exports_ass_without_running_editor_or_ocr(self):
        from hardsub_extractor.gui import ExtractorGUI
        from hardsub_extractor.subtitle_edit_handoff import ProofreadASSExport

        class Var:
            def __init__(self, value):
                self.value = value

            def get(self):
                return self.value

            def set(self, value):
                self.value = value

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _, original, _ = self.inputs(root)
            draft = root / "one movie-proofread.srt"
            draft.write_bytes(SRT)
            gui = ExtractorGUI.__new__(ExtractorGUI)
            gui._pending_close = False
            gui.worker = None
            gui._tab_capture_thread = None
            gui.source_mode_var = Var("local")
            gui.out_var = Var(str(root / "one movie"))
            gui.status_var = Var("")
            gui._log = mock.Mock()
            expected = ProofreadASSExport(
                draft, root / "one movie-proofread-abc.ass", 2, False,
            )
            with (
                mock.patch(
                    "hardsub_extractor.subtitle_edit_handoff.export_proofread_ass",
                    return_value=expected,
                ) as converter,
                mock.patch("hardsub_extractor.gui.messagebox.showinfo") as info,
            ):
                gui._export_external_proofread_ass()
            converter.assert_called_once_with(original)
            info.assert_called_once()
            self.assertIn("교정 ASS 새로 생성 완료", gui.status_var.get())
            self.assertNotIn(str(root), repr(gui._log.call_args_list))
            self.assertEqual(original.read_bytes(), SRT)
            self.assertEqual(draft.read_bytes(), SRT)

    def test_gui_refuses_export_while_background_extraction_runs(self):
        from hardsub_extractor.gui import ExtractorGUI

        class Busy:
            def is_alive(self):
                return True

        gui = ExtractorGUI.__new__(ExtractorGUI)
        gui._pending_close = False
        gui.worker = Busy()
        with mock.patch(
            "hardsub_extractor.subtitle_edit_handoff.export_proofread_ass",
        ) as converter:
            gui._export_external_proofread_ass()
        converter.assert_not_called()

    def test_invalid_srt_never_launches_or_creates_review_copy(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            video, subtitle, exe = self.inputs(root)
            subtitle.write_bytes(b"not a subtitle")
            opener = mock.Mock()
            with self.assertRaises(ProofreadHandoffError):
                launch_subtitle_edit_proofreading(
                    video, subtitle, executable=exe, opener=opener,
                )
            opener.assert_not_called()
            self.assertFalse((root / "one movie-proofread.srt").exists())

    def test_launch_error_cleans_only_newly_created_copy(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            video, subtitle, exe = self.inputs(root)
            with self.assertRaises(OSError):
                launch_subtitle_edit_proofreading(
                    video, subtitle, executable=exe,
                    opener=mock.Mock(side_effect=OSError("launch failed")),
                )
            self.assertFalse((root / "one movie-proofread.srt").exists())
            self.assertEqual(subtitle.read_bytes(), SRT)

    def test_external_failure_never_deletes_existing_human_copy(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            video, subtitle, exe = self.inputs(root)
            draft = root / "one movie-proofread.srt"
            draft.write_bytes(b"human changes")
            with self.assertRaises(OSError):
                launch_subtitle_edit_proofreading(
                    video, subtitle, executable=exe,
                    opener=mock.Mock(side_effect=OSError("launch failed")),
                )
            self.assertEqual(draft.read_bytes(), b"human changes")

    def test_symlinks_and_wrong_app_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            video, subtitle, exe = self.inputs(root)
            wrong_exe = root / "other.exe"
            wrong_exe.write_bytes(b"untrusted")
            with self.assertRaises(ProofreadHandoffError):
                validate_handoff(video, subtitle, executable=wrong_exe)
            linked = root / "redirect.srt"
            try:
                linked.symlink_to(subtitle)
            except (OSError, NotImplementedError):
                self.skipTest("symlinks unavailable")
            with self.assertRaises(ProofreadHandoffError):
                validate_handoff(video, linked, executable=exe)


if __name__ == "__main__":
    unittest.main()
