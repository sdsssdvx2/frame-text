"""Real Windows Tk widget smoke: no video, browser, network or settings changes.

The programmatic ttk Radiobutton invoke exercises actual GUI callbacks and
widget state. It is not a simulated human mouse click.
"""
from __future__ import annotations

from pathlib import Path
import tempfile
import tkinter as tk
from tkinter import ttk
from unittest import mock

from hardsub_extractor.gui import ExtractorGUI


def _radios(widget):
    for child in widget.winfo_children():
        if isinstance(child, ttk.Radiobutton):
            yield child
        yield from _radios(child)


def _click_mode(root, label):
    # A local video and a local batch BOTH begin "내 영상". Select the
    # semantic control value, never an ambiguous localized substring.
    modes = {
        "내 영상": "local",
        "내 영상 일괄": "local-batch",
        "웹 자막 우선": "captions",
        "사이트 회차": "epsite",
    }
    expected = modes[label]
    candidates = [
        item for item in _radios(root)
        if str(item.cget("value")) == expected
    ]
    if len(candidates) != 1:
        raise AssertionError("expected exactly one source-mode radio control")
    candidates[0].invoke()


def _require_state(widget, state):
    if str(widget.cget("state")) != state:
        raise AssertionError("unexpected Tk widget state")


def main():
    root = None
    try:
        root = tk.Tk()
        root.withdraw()
        gui = ExtractorGUI(root)
        root.update_idletasks()
        if gui.source_mode_var.get() != "epsite":
            raise AssertionError("unexpected default input mode")
        _require_state(gui.source_picker, "disabled")
        _require_state(gui.review_button, "disabled")
        _require_state(gui.local_ocr_region_choice, "disabled")
        if gui.local_ocr_region_var.get() != "기본 · 하단 40%":
            raise AssertionError("existing 40-percent video crop not preserved")

        _click_mode(root, "내 영상")
        if gui.source_mode_var.get() != "local":
            raise AssertionError("local video mode selection failed")
        _require_state(gui.source_picker, "normal")
        _require_state(gui.review_button, "normal")
        _require_state(gui.target_entry, "disabled")
        _require_state(gui.languages_entry, "disabled")
        _require_state(gui.ocr_language_entry, "readonly")
        _require_state(gui.video_ocr_check, "disabled")
        _require_state(gui.caption_islands_check, "normal")
        _require_state(gui.local_ocr_region_choice, "readonly")
        gui.local_ocr_region_choice.set("좁게 · 하단 25% (윗줄 유실 가능)")
        if gui.local_ocr_region_var.get() != "좁게 · 하단 25% (윗줄 유실 가능)":
            raise AssertionError("actual Tk ROI combobox selection failed")
        if gui.caption_islands_var.get():
            raise AssertionError("experimental caption islands must default OFF")
        gui.caption_islands_check.invoke()
        if not gui.caption_islands_var.get():
            raise AssertionError("real Tk experimental checkbox did not respond")
        with tempfile.TemporaryDirectory(prefix="hse-local-multilang-tk-") as td:
            local_video = Path(td) / "synthetic.mp4"
            local_video.write_bytes(b"synthetic no network or OCR")
            gui.url_var.set(str(local_video))
            gui.out_var.set(str(Path(td) / "subtitles"))
            gui.ocr_language_var.set("ja")
            mode, _source, _target, _out = gui._validate()
            if mode != "local" or gui._requested_ocr_language != "ja":
                raise AssertionError("local multilingual OCR selection failed")

        _click_mode(root, "내 영상 일괄")
        if gui.source_mode_var.get() != "local-batch":
            raise AssertionError("local batch mode selection failed")
        _require_state(gui.caption_islands_check, "normal")
        _require_state(gui.local_ocr_region_choice, "readonly")
        if gui.local_ocr_region_var.get() != "좁게 · 하단 25% (윗줄 유실 가능)":
            raise AssertionError("local-to-batch ROI selection not preserved")
        if not gui.caption_islands_var.get():
            raise AssertionError("local-to-batch opt-in unexpectedly lost")
        _require_state(gui.source_picker, "normal")
        _require_state(gui.review_button, "disabled")
        _require_state(gui.target_entry, "disabled")
        _require_state(gui.languages_entry, "disabled")
        _require_state(gui.ocr_language_entry, "readonly")
        _require_state(gui.video_ocr_check, "disabled")
        if gui.source_picker.cget("text") != "영상 여러 개 선택":
            raise AssertionError("local episode multi-video chooser missing")
        with tempfile.TemporaryDirectory(prefix="hse-local-batch-tk-") as td:
            directory = Path(td)
            video = directory / "제01화.mp4"
            video.write_bytes(b"validated only; not decoded")
            second = directory / "제02화.mkv"
            second.write_bytes(b"validated only; not decoded")
            gui.out_var.set(str(directory / "out"))
            gui.ocr_language_var.set("zh")
            with mock.patch(
                "hardsub_extractor.gui.filedialog.askopenfilenames",
                return_value=(str(video), str(second)),
            ):
                gui.source_picker.invoke()
            mode, source, target, out = gui._validate()
            if (mode, target) != ("local-batch", None):
                raise AssertionError("local batch multi-video GUI dispatch incorrect")
            if ("2개" not in source or str(directory) in source
                    or gui._requested_local_batch_paths !=
                    (str(video), str(second))):
                raise AssertionError("Tk Explorer selection was not private and exact")
            if gui._requested_ocr_language != "zh" or out.exists():
                raise AssertionError("local batch language or dry-run output check failed")
            # Legacy UTF-8 user-authored lists still work with typed path.
            list_file = directory / "batch.txt"
            list_file.write_text("제01화.mp4\n", encoding="utf-8")
            gui.url_var.set(str(list_file))
            mode, source, target, out = gui._validate()
            if (mode, source, target) != ("local-batch", str(list_file), None):
                raise AssertionError("legacy UTF-8 local batch lost compatibility")
            if gui._requested_local_batch_paths:
                raise AssertionError("typed list unexpectedly reused stale selection")

        _click_mode(root, "웹 자막 우선")
        if gui.source_mode_var.get() != "captions":
            raise AssertionError("caption mode selection failed")
        _require_state(gui.caption_islands_check, "disabled")
        _require_state(gui.local_ocr_region_choice, "disabled")
        if gui.local_ocr_region_var.get() != "기본 · 하단 40%":
            raise AssertionError("custom ROI leaked into unrelated web mode")
        if gui.caption_islands_var.get():
            raise AssertionError("experimental local OCR leaked into public captions")
        _require_state(gui.source_picker, "disabled")
        _require_state(gui.review_button, "disabled")
        _require_state(gui.target_entry, "normal")
        _require_state(gui.languages_entry, "normal")
        _require_state(gui.ocr_language_entry, "disabled")
        _require_state(gui.video_ocr_check, "normal")
        if not gui.video_ocr_var.get():
            raise AssertionError("bounded video OCR must be visibly enabled by default")
        if gui.target_picker.cget("text") != "목록 선택":
            raise AssertionError("caption episode list picker is not available")
        with tempfile.TemporaryDirectory(prefix="hse-gui-widget-") as td:
            gui.url_var.set("https://www.youtube.com/watch?v=public_smoke")
            gui.languages_var.set("ko,en,zh-Hans")
            gui.out_var.set(str(Path(td) / "captions"))
            mode, source, url_file, out = gui._validate()
            if not (
                mode == "captions" and source.startswith("https://")
                and url_file is None and not out.exists()
                and gui._requested_languages == ("ko", "en", "zh-hans")
            ):
                raise AssertionError("caption form validation failed")
            if not gui._requested_video_ocr_enabled:
                raise AssertionError("selected bounded OCR setting was not validated")
            gui.video_ocr_var.set(False)
            gui._validate()
            if gui._requested_video_ocr_enabled:
                raise AssertionError("turning off web video OCR was ignored")
            gui.video_ocr_var.set(True)
            gui._validate()

        _click_mode(root, "사이트 회차")
        if gui.source_mode_var.get() != "epsite":
            raise AssertionError("EpSite mode selection failed")
        _require_state(gui.source_picker, "disabled")
        _require_state(gui.review_button, "disabled")
        _require_state(gui.languages_entry, "disabled")
        _require_state(gui.ocr_language_entry, "readonly")
        _require_state(gui.video_ocr_check, "disabled")
        _require_state(gui.target_entry, "normal")

        print("HSE_TK_WIDGET_SMOKE=PASS MODES=4 VALIDATION=PASS OCR_LANGUAGES=4")
        return 0
    except (tk.TclError, AssertionError, OSError) as exc:
        print("HSE_TK_WIDGET_SMOKE=FAIL REASON=" + type(exc).__name__)
        return 1
    finally:
        if root is not None:
            root.destroy()


if __name__ == "__main__":
    raise SystemExit(main())
