"""Actual Windows Tk GUI -> background worker -> FFmpeg embedded SRT/ASS acceptance.

Research-only script. It creates an encoded two-language MKV in disposable
temporary storage, invokes REAL Tk radio and Start widgets, pumps Tk's event
queue, and checks produced files/metadata and original media hashes.

No user media, login/browser profiles, live web, OCR inference or installation.
Modal dialogs are intercepted solely to prevent an unattended CI runner from
blocking. All extracted text and absolute paths remain in local temporary
memory/files and are never printed in logs. This is not a frozen EXE or human
mouse test; those remain separately gated.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import tkinter as tk
from tkinter import ttk
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools"))

from hardsub_extractor.gui import ExtractorGUI
from hardsub_extractor.local_batch import load_local_batch
from hardsub_extractor.subtitle_formats import parse_subtitle_bytes
from verify_local_embedded_video import TEXT, encode_fixture

EXPECTED_SPANS = ((1.0, 2.5), (3.0, 4.5))
TERMINAL_STATUSES = {"완료", "실패", "취소됨"}
MAX_PER_CASE_SECONDS = 90.0


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _all_widgets(root):
    for child in root.winfo_children():
        yield child
        yield from _all_widgets(child)


def _choose_local_mode(root, gui: ExtractorGUI):
    controls = [
        child for child in _all_widgets(root)
        if isinstance(child, ttk.Radiobutton)
        and str(child.cget("value")) == "local"
    ]
    if len(controls) != 1:
        raise AssertionError("missing or duplicate real local-mode Tk radio")
    if str(gui.source_picker.cget("state")) != "disabled":
        raise AssertionError("unexpected initial GUI source picker")
    controls[0].invoke()
    if gui.source_mode_var.get() != "local":
        raise AssertionError("real Tk local-mode callback failed")
    if str(gui.source_picker.cget("state")) != "normal":
        raise AssertionError("local video picker was not activated")
    if str(gui.ocr_language_entry.cget("state")) != "readonly":
        raise AssertionError("multilingual local OCR selector unavailable")


def _start_and_poll(root, gui, dest: Path, video: Path, language: str, desired: str):
    if gui.worker is not None and gui.worker.is_alive():
        raise AssertionError("previous worker was not joined")
    if str(gui.start_button.cget("state")) != "normal":
        raise AssertionError("new extraction was not enabled")
    gui.url_var.set(str(video))
    gui.out_var.set(str(dest))
    gui.ocr_language_var.set(language)
    gui.diag_var.set(False)
    if str(gui.start_button.invoke()) not in ("", "None"):
        # Tk's returned Tcl value can be empty or None for a Python callback.
        raise AssertionError("unexpected start button return value")
    if gui.worker is None:
        raise AssertionError("real Start Tk callback never created a worker")
    if str(gui.start_button.cget("state")) != "disabled":
        raise AssertionError("start button not disabled during active work")
    deadline = time.monotonic() + MAX_PER_CASE_SECONDS
    while time.monotonic() < deadline:
        root.update()
        state = gui.status_var.get()
        if state in TERMINAL_STATUSES and not gui.worker.is_alive():
            if state != desired:
                raise AssertionError("unexpected real Tk worker terminal status")
            if str(gui.start_button.cget("state")) != "normal":
                raise AssertionError("GUI was not re-enabled after worker result")
            if str(gui.cancel_button.cget("state")) != "disabled":
                raise AssertionError("cancel button remained enabled after completion")
            return
        time.sleep(0.025)
    gui._cancel_event.set()
    raise TimeoutError("real Tk background worker never reached a terminal result")


def _assert_export(out: Path, language: str):
    raw = out.with_name(out.name + ".embedded.extracted.srt")
    metadata = out.with_name(out.name + ".source.json")
    srt = out.with_suffix(".srt")
    ass = out.with_suffix(".ass")
    if not all(path.is_file() and path.stat().st_size > 0
               for path in (srt, ass, raw, metadata)):
        raise AssertionError("real Tk event path did not publish all embedded outputs")
    manifest = json.loads(metadata.read_text(encoding="utf-8"))
    if (
        manifest.get("ocr_applied") is not False
        or manifest.get("source_video_modified") is not False
        or manifest.get("language_requested") != language
        or manifest.get("cue_count") != 2
        or manifest.get("source_stream_export")
        != "ffmpeg_decoded_srt_not_original_container_bytes"
    ):
        raise AssertionError("embedded source provenance not correctly recorded")
    for suffix, path in (("srt", srt), ("ass", ass), ("srt", raw)):
        fmt, cues = parse_subtitle_bytes(path.read_bytes(), source_url="source." + suffix)
        if fmt != suffix or len(cues) != 2:
            raise AssertionError("real Tk export mismatched format or cue count")
        for i, cue in enumerate(cues):
            if cue.text != TEXT[language][i]:
                raise AssertionError("real Tk export subtitle text differs from independent source")
            if max(abs(cue.start - EXPECTED_SPANS[i][0]),
                   abs(cue.end - EXPECTED_SPANS[i][1])) > 0.08:
                raise AssertionError("real Tk export drifted from original source timeline")
    if str(out.parent) in metadata.read_text(encoding="utf-8"):
        raise AssertionError("absolute source path leaked into provenance")
    return [srt, ass, raw, metadata]



def _encode_no_embedded_tracks(root: Path, source: Path) -> tuple[Path, dict]:
    """Keep the same real encoded video but remove embedded soft tracks."""
    from hardsub_extractor.runtime_paths import resolve_binary
    video = root / "사이드카만 있는 영상.mkv"
    cmd = [
        resolve_binary("ffmpeg"), "-nostdin", "-hide_banner", "-loglevel", "error",
        "-protocol_whitelist", "file,pipe", "-i", str(source),
        "-map", "0:v:0", "-c:v", "copy", "-y", str(video),
    ]
    result = subprocess.run(
        cmd, input=b"", stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        timeout=60, check=False,
    )
    if result.returncode != 0 or not video.is_file() or video.stat().st_size < 1024:
        raise AssertionError("could not create separate video with no embedded text")
    sidecars = {}
    for language in ("ko", "en"):
        suffix = "srt" if language == "ko" else "vtt"
        track = video.with_name(video.stem + f".{language}.{suffix}")
        if suffix == "vtt":
            lines = ["WEBVTT", ""]
            for (start, end), phrase in zip(EXPECTED_SPANS, TEXT[language]):
                lines.extend((f"00:00:{start:06.3f} --> 00:00:{end:06.3f}", phrase, ""))
        else:
            lines = []
            for i, ((start, end), phrase) in enumerate(zip(EXPECTED_SPANS, TEXT[language]), 1):
                def clock(value: float) -> str:
                    seconds = int(value)
                    millis = round((value - seconds) * 1000)
                    return f"00:00:{seconds:02d},{millis:03d}"
                lines.extend((str(i), f"{clock(start)} --> {clock(end)}", phrase, ""))
        track.write_text("\n".join(lines), encoding="utf-8")
        sidecars[language] = track
    return video, sidecars


def _assert_sidecar_export(out: Path, language: str, original: Path, *, expected_mode: str = "local-sidecar-text"):
    extension = original.suffix
    raw = out.with_name(out.name + ".sidecar.original" + extension)
    metadata = out.with_name(out.name + ".source.json")
    srt = out.with_suffix(".srt")
    ass = out.with_suffix(".ass")
    if not all(path.is_file() and path.stat().st_size > 0
               for path in (raw, metadata, srt, ass)):
        raise AssertionError("real Tk sidecar failed to publish source and normalized outputs")
    if raw.read_bytes() != original.read_bytes():
        raise AssertionError("original sidecar bytes were changed during import")
    manifest = json.loads(metadata.read_text("utf-8"))
    if (manifest.get("mode") != expected_mode
        or manifest.get("language_requested") != language
        or manifest.get("cue_count") != 2
        or manifest.get("ocr_applied") is not False
        or manifest.get("original_bytes_retained") is not True
        or manifest.get("source_sidecar_name") != original.name
        or manifest.get("original_sha256") != _hash(original)):
        raise AssertionError("real Tk sidecar provenance differs from original source")
    for fmt, path in (("srt", srt), ("ass", ass)):
        _, cues = parse_subtitle_bytes(path.read_bytes(), source_url="x." + fmt)
        if len(cues) != 2:
            raise AssertionError("real Tk sidecar SRT/ASS count differs")
        for i, cue in enumerate(cues):
            if cue.text != TEXT[language][i]:
                raise AssertionError("real Tk sidecar original text changed")
            if max(abs(cue.start - EXPECTED_SPANS[i][0]),
                   abs(cue.end - EXPECTED_SPANS[i][1])) > 0.01:
                raise AssertionError("real Tk sidecar output timeline drift")
    return [raw, metadata, srt, ass]


def verify() -> dict:
    with tempfile.TemporaryDirectory(prefix="hse-real-tk-end-to-end-") as td:
        scratch = Path(td)
        video = encode_fixture(scratch)
        original_hash = _hash(video)
        input_sources = {
            path: _hash(path) for path in (scratch / "ko.srt", scratch / "en.srt")
        }
        root = None
        info = []
        errors = []
        forbidden_calls = []
        try:
            root = tk.Tk()
            root.withdraw()
            with mock.patch("hardsub_extractor.gui.messagebox.showinfo",
                            side_effect=lambda *a, **k: info.append(a[0])), \
                 mock.patch("hardsub_extractor.gui.messagebox.showerror",
                            side_effect=lambda *a, **k: errors.append(a[0])), \
                 mock.patch("hardsub_extractor.gui.RapidOCRAdapter",
                            side_effect=AssertionError("should never run OCR")) as ocr, \
                 mock.patch("hardsub_extractor.gui.extract_local_video",
                            side_effect=AssertionError("should never OCR video")) as hard, \
                 mock.patch("hardsub_extractor.gui.PublicCaptionCollector",
                            side_effect=AssertionError("should never fetch sites")) as network:
                gui = ExtractorGUI(root)
                _choose_local_mode(root, gui)
                root.update_idletasks()
                for language in ("ko", "en"):
                    dest = scratch / f"real-tk-{language}"
                    before = len(info)
                    _start_and_poll(root, gui, dest, video, language, "완료")
                    if len(info) != before + 1 or info[-1] != "완료" or errors:
                        raise AssertionError("GUI completion popup did not match completed export")
                    files = _assert_export(dest, language)
                    forbidden_calls.extend(files)

                # A real encoded video without internal text tracks must use
                # correctly tagged sibling SRT/VTT instead of activating OCR.
                plain, sidecars = _encode_no_embedded_tracks(scratch, video)
                plain_pin = _hash(plain)
                sidecar_pins = {p: _hash(p) for p in sidecars.values()}
                for language, sidecar in sidecars.items():
                    dest = scratch / f"real-tk-sidecar-{language}"
                    before = len(info)
                    _start_and_poll(root, gui, dest, plain, language, "완료")
                    if len(info) != before + 1 or info[-1] != "완료" or errors:
                        raise AssertionError("real Tk sidecar GUI completion was not successful")
                    forbidden_calls.extend(_assert_sidecar_export(dest, language, sidecar))
                if _hash(plain) != plain_pin or any(
                    _hash(p) != pin for p, pin in sidecar_pins.items()
                ):
                    raise AssertionError("real video or original sidecar was modified")

                # Real Tk: select TWO local video files in one Explorer
                # dialog, then Start the actual background batch worker.
                # This tests more than a file-list mock: every selected video
                # is actually parsed by FFprobe/FFmpeg, original subtitle
                # bytes are preserved, and no OCR/network engine is started.
                batch_radios = [
                    item for item in _all_widgets(root)
                    if isinstance(item, ttk.Radiobutton)
                    and str(item.cget("value")) == "local-batch"
                ]
                if len(batch_radios) != 1:
                    raise AssertionError("actual Tk local batch mode is not unique")
                batch_radios[0].invoke()
                batch_dest = scratch / "real-tk-local-multi-select"
                gui.out_var.set(str(batch_dest))
                with mock.patch(
                    "hardsub_extractor.gui.filedialog.askopenfilenames",
                    return_value=(str(video), str(plain)),
                ):
                    gui.source_picker.invoke()
                display = gui.url_var.get()
                if "2개" not in display or str(scratch) in display:
                    raise AssertionError("batch chooser leaked paths or rejected selection")
                planned = load_local_batch(
                    [video, plain], None, batch_dest,
                )
                if len(planned) != 2:
                    raise AssertionError("two independently selected inputs were lost")
                before = len(info)
                _start_and_poll(
                    root, gui, batch_dest, display, "ko", "완료",
                )
                if (
                    len(info) != before + 1
                    or info[-1] != "로컬 영상 일괄 처리 결과" or errors
                ):
                    raise AssertionError("real Tk local batch worker did not finish")
                forbidden_calls.extend(_assert_export(
                    planned[0].out_base, "ko",
                ))
                forbidden_calls.extend(_assert_sidecar_export(
                    planned[1].out_base, "ko", sidecars["ko"],
                ))
                if any(_hash(p) != pin for p, pin in sidecar_pins.items()):
                    raise AssertionError("multi-picker changed original sidecar")
                # Resume the SAME inputs without scanning new media/model
                # sources. Both must be source-backed, structurally verified
                # and all existing subtitle bytes remain identical.
                pins = {_path: _hash(_path) for _path in forbidden_calls}
                before = len(info)
                _start_and_poll(root, gui, batch_dest, display, "ko", "완료")
                if len(info) != before + 1 or errors:
                    raise AssertionError("real Tk original-text resume failed")
                if "기존 원본 자막 구조 검증 2개" not in gui.log.get("1.0", "end"):
                    raise AssertionError("existing original subtitle resume not audited")
                if any(_hash(path) != digest for path, digest in pins.items()):
                    raise AssertionError("read-only batch resume modified source or result")
                # Remaining local one-video and explicit manually chosen
                # subtitle acceptance must still work after batch mode.
                local_radios = [
                    item for item in _all_widgets(root)
                    if isinstance(item, ttk.Radiobutton)
                    and str(item.cget("value")) == "local"
                ]
                if len(local_radios) != 1:
                    raise AssertionError("local input mode missing after batch")
                local_radios[0].invoke()
                if gui.source_mode_var.get() != "local":
                    raise AssertionError("batch mode could not switch back")

                # A user-selected file may have an unrelated name and live
                # outside the video folder. It wins over EMBEDDED captions,
                # and an invalid selection must not launch OCR.
                explicit = scratch / "downloaded-unlabeled.vtt"
                explicit.write_bytes(sidecars["en"].read_bytes())
                explicit_pin = _hash(explicit)
                gui.sidecar_var.set(str(explicit))
                dest = scratch / "real-tk-selected-en"
                before = len(info)
                _start_and_poll(root, gui, dest, video, "en", "완료")
                if len(info) != before + 1 or info[-1] != "완료" or errors:
                    raise AssertionError("explicit selected subtitle did not finish")
                forbidden_calls.extend(_assert_sidecar_export(
                    dest, "en", explicit, expected_mode="local-selected-text",
                ))
                if _hash(explicit) != explicit_pin:
                    raise AssertionError("manually chosen source subtitle mutated")
                broken = scratch / "invalid-selected.srt"
                broken.write_bytes(b"invalid timestamp data")
                gui.sidecar_var.set(str(broken))
                before = len(errors)
                failed = scratch / "real-tk-selected-invalid"
                _start_and_poll(root, gui, failed, video, "en", "실패")
                if len(errors) != before + 1 or errors[-1] != "작업 실패":
                    raise AssertionError("invalid selected original was not rejected")
                if any(p.exists() for p in (
                    failed.with_suffix(".srt"), failed.with_suffix(".ass"),
                    failed.with_name(failed.name + ".source.json"),
                )):
                    raise AssertionError("invalid selected subtitle produced a false result")
                gui.sidecar_var.set("")

                # Clicking Start again on an occupied destination must never
                # change a byte of a user-protected SRT or provenance file.
                dest = scratch / "real-tk-ko"
                pins = {_path: _hash(_path) for _path in forbidden_calls}
                before = len(errors)
                prior_worker = gui.worker
                gui.ocr_language_var.set("ko")
                gui.url_var.set(str(video))
                gui.out_var.set(str(dest))
                gui.start_button.invoke()
                root.update()
                if len(errors) != before + 1 or errors[-1] != "입력 확인":
                    raise AssertionError("real Tk preflight failed to reject existing result")
                if gui.worker is not prior_worker:
                    raise AssertionError("occupied GUI output improperly spawned a new worker")
                if any(_hash(path) != digest for path, digest in pins.items()):
                    raise AssertionError("existing user-like subtitles changed during GUI rerun")

                # A third language is NOT available in the actual MKV.
                # Explicitly deny model availability so missing-language
                # handling is checked without launching expensive/unsafe OCR.
                missing = scratch / "forbidden-ocr-model-dir"
                with mock.patch("hardsub_extractor.gui.bundled_model_dir",
                                return_value=missing):
                    before = len(errors)
                    ja_out = scratch / "real-tk-ja"
                    _start_and_poll(root, gui, ja_out, video, "ja", "실패")
                    if len(errors) != before + 1 or errors[-1] != "작업 실패":
                        raise AssertionError("no-track JA did not show a real GUI failure")
                    if any(p.exists() for p in (
                        ja_out.with_suffix(".srt"), ja_out.with_suffix(".ass"),
                        ja_out.with_name(ja_out.name + ".source.json"),
                    )):
                        raise AssertionError("missing JA track falsely published a subtitle")

                ocr.assert_not_called()
                hard.assert_not_called()
                network.assert_not_called()
                if _hash(video) != original_hash:
                    raise AssertionError("source MKV was changed by real GUI operations")
                if any(_hash(path) != digest for path, digest in input_sources.items()):
                    raise AssertionError("original fixture SRT source was changed")
        finally:
            if root is not None:
                try:
                    root.destroy()
                except tk.TclError:
                    pass
    return {
        "schema": 1, "real_tk_callbacks": True,
        "real_worker_thread_and_after_event_loop": True,
        "real_mkv_ffmpeg_english_korean": True,
        "languages_with_exact_srt_ass_and_source_provenance": ["ko", "en"],
        "total_independently_verified_cues": 14,
        "real_tk_explorer_two_video_batch_start_and_worker": True,
        "real_tk_local_resume_two_original_tracks_verified_readonly": True,
        "batch_embedded_first_and_sidecar_without_ocr": True,
        "batch_no_manual_path_list_required": True,
        "real_tk_explicit_untagged_text_precedence": True,
        "invalid_explicit_source_never_falls_back_to_ocr": True,
        "real_tk_original_sidecar_languages": ["ko", "en"],
        "real_tk_sidecar_original_bytes_preserved": True,
        "srt_ass_timing_within_80ms": True,
        "preexisting_output_unchanged": True,
        "nonexistent_language_rejected_without_ocr": True,
        "ocr_model_calls": 0,
        "site_requests": 0,
        "user_files_modified": False,
        "frozen_exe_gui_verified": False,
        "real_site_or_anime_accuracy_verified": False,
        "release_gate_passed": False,
    }


def main() -> int:
    try:
        result = verify()
        print("HSE_REAL_TK_LOCAL_E2E " + json.dumps(result, sort_keys=True))
        return 0
    except Exception as exc:
        print("HSE_REAL_TK_LOCAL_E2E_ERROR " + json.dumps({
            "failure_type": type(exc).__name__,
            "real_tk_pipeline_complete": False,
            "release_gate_passed": False,
        }, sort_keys=True))
        return 3


if __name__ == "__main__":
    raise SystemExit(main())
