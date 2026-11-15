"""Actual Windows Tk GUI visual-OCR fallback through real encoded MP4 and PP-OCR.

Runs entirely inside a disposable CI workspace. It does NOT use user media,
remote websites, VPN, credentials, real anime or an installed/frozen EXE.
Only modal popup functions and provider network entrypoints are stubbed;
Tk controls, extraction worker, FFmpeg, subtitle detector and actual OCR
models are used as shipped. All source text stays in temporary memory.
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
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools"))

from hardsub_extractor.gui import ExtractorGUI
from hardsub_extractor.runtime_paths import resolve_binary
from hardsub_extractor.subtitle_formats import parse_subtitle_bytes
from verify_multilang_video_pipeline import encode_sample, VIDEO_TEXT, EXPECTED_SPANS
from verify_real_gui_local_subtitles import _all_widgets, _choose_local_mode, _start_and_poll

SUPPORTED_LANGUAGES = ("ko", "en", "ja", "zh")
DEFAULT_LANGUAGES = ("ko", "en")


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _assert_hard_output(base: Path, language: str) -> dict:
    srt = base.with_suffix(".srt")
    ass = base.with_suffix(".ass")
    if not (srt.is_file() and ass.is_file()):
        raise AssertionError("real GUI hard OCR did not create both subtitle formats")
    _, primary = parse_subtitle_bytes(srt.read_bytes(), source_url="test.srt")
    _, secondary = parse_subtitle_bytes(ass.read_bytes(), source_url="test.ass")
    if len(primary) != 2 or len(secondary) != 2:
        raise AssertionError("synthetic visible dialogue cue count not exact")
    largest_error = 0.0
    for i, (a, b) in enumerate(zip(primary, secondary)):
        expected = VIDEO_TEXT[language][i]
        if a.text != expected or b.text != expected:
            raise AssertionError("synthetic visible dialogue OCR text differs from independent oracle")
        largest_error = max(
            largest_error,
            abs(a.start - EXPECTED_SPANS[i][0]),
            abs(a.end - EXPECTED_SPANS[i][1]),
            abs(b.start - EXPECTED_SPANS[i][0]),
            abs(b.end - EXPECTED_SPANS[i][1]),
        )
        if (a.text != b.text
                or abs(a.start - b.start) > 0.011
                or abs(a.end - b.end) > 0.011):
            raise AssertionError("actual GUI SRT/ASS text and timing parity broken")
    # This checks bounded source timeline, not proof of real-anime accuracy.
    if largest_error > 0.65:
        raise AssertionError("real GUI hard OCR synthetic timing exceeds 650 ms")
    return {
        "language": language,
        "exact_text_cues": len(primary),
        "max_source_edge_error_seconds": round(largest_error, 4),
        "srt_ass_parity": True,
    }


def _create_blank_clip(dest: Path):
    command = [
        resolve_binary("ffmpeg"), "-nostdin", "-hide_banner", "-loglevel",
        "error", "-f", "lavfi", "-i",
        "color=c=black:s=640x360:r=8:d=7.125",
        "-an", "-c:v", "mpeg4", "-q:v", "4",
        "-pix_fmt", "yuv420p", "-movflags", "+faststart",
        "-n", str(dest),
    ]
    done = subprocess.run(
        command, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL, timeout=90, check=False,
    )
    if done.returncode != 0 or not dest.is_file() or dest.stat().st_size < 1024:
        raise RuntimeError("bounded blank hard-OCR negative media unavailable")



def _accept_real_one_click_ocr_comparison(
    root: tk.Tk, gui: ExtractorGUI, video: Path, language: str,
) -> dict:
    """Real Tk click -> one FFmpeg frame -> two genuine local PP-OCR readings.

    Existing fixture and verified model assets are reused. Do not inspect
    private user videos, save OCR content or silently enable a filter.
    """
    from hardsub_extractor.local_ocr_preview import LocalOCRPreview

    if str(gui.ocr_compare_button.cget("state")) != "normal":
        raise AssertionError("real Tk one-click OCR compare is unavailable")
    gui.url_var.set(str(video))
    gui.ocr_language_var.set(language)
    gui.preview_time_var.set("00:00:02")
    old_light = bool(gui.light_caption_temporal_var.get())
    before = _sha(video)
    if old_light:
        raise AssertionError("experimental bright OCR unexpectedly enabled by default")
    opened = None
    try:
        with mock.patch.object(
            gui, "_show_local_ocr_preview",
            wraps=gui._show_local_ocr_preview,
        ) as show:
            gui.ocr_compare_button.invoke()
            if gui.worker is None:
                raise AssertionError("real Tk OCR comparison did not start a worker")
            deadline = time.monotonic() + 90.0
            while time.monotonic() < deadline:
                root.update()
                if show.called and not gui.worker.is_alive():
                    opened = show.call_args.args[0]
                    break
                time.sleep(0.025)
        if not isinstance(opened, LocalOCRPreview):
            raise AssertionError("actual OCR did not reach the real popup")
        if (
            gui.status_var.get() != "한 프레임 OCR 확인 완료"
            or opened.preview_mode != "compare-bright"
            or opened.region_mode != "bottom40"
            or abs(opened.actual_second - 2.0) > 0.25
            or opened.comparison_text is None
            or opened.comparison_confidence is None
        ):
            raise AssertionError("actual OCR comparison GUI result is incomplete")
        if VIDEO_TEXT[language][0] not in opened.text:
            raise AssertionError("actual preview missed the known source subtitle text")
        windows = [
            child for child in root.winfo_children()
            if isinstance(child, tk.Toplevel)
        ]
        if len(windows) != 1:
            raise AssertionError("expected exactly one real comparison popup")
        fields = [
            child for child in _all_widgets(windows[0])
            if isinstance(child, tk.Text)
        ]
        if (
            len(fields) != 2
            or fields[0].get("1.0", "end-1c") != opened.text
            or fields[1].get("1.0", "end-1c") != (
                opened.comparison_text or "(감지된 글자 없음)"
            )
        ):
            raise AssertionError("the real comparison popup does not show both readings")
        if (
            bool(gui.light_caption_temporal_var.get()) != old_light
            or _sha(video) != before
            or str(gui.start_button.cget("state")) != "normal"
            or str(gui.cancel_button.cget("state")) != "disabled"
        ):
            raise AssertionError("read-only OCR comparison changed extraction state")
        return {
            "real_tk_compare_button": True,
            "single_decoded_frame": True,
            "actual_ppocr_comparison": True,
            "baseline_exact_first_cue": True,
            "two_real_text_fields": True,
            "filter_settings_unchanged": True,
            "original_video_sha256_unchanged": True,
            "original_ocr_text_logged": False,
        }
    finally:
        for child in root.winfo_children():
            if isinstance(child, tk.Toplevel):
                child.destroy()
        if gui.worker is not None and gui.worker.is_alive():
            gui._cancel_event.set()

def verify(
    models: Path, manifest: Path, languages: tuple[str, ...],
    *, compare_caption_islands: bool = False,
) -> dict:
    if not models.is_dir() or not manifest.is_file():
        raise ValueError("required pinned OCR models/manifest absent")
    results = []
    experimental = []
    with tempfile.TemporaryDirectory(prefix="hse-gui-real-hard-ocr-") as tmp:
        rootdir = Path(tmp)
        sources = {}
        for lang in languages:
            source = rootdir / f"{lang}-source.mp4"
            encode_sample(source, lang)
            sources[source] = _sha(source)
        blank = rootdir / "blank-no-dialogue.mp4"
        _create_blank_clip(blank)
        sources[blank] = _sha(blank)

        tk_root = None
        messages = []
        errors = []
        try:
            tk_root = tk.Tk()
            tk_root.withdraw()
            with mock.patch("hardsub_extractor.gui.messagebox.showinfo",
                            side_effect=lambda *args, **kwargs: messages.append(args[0])), \
                 mock.patch("hardsub_extractor.gui.messagebox.showerror",
                            side_effect=lambda *args, **kwargs: errors.append(args[0])), \
                 mock.patch("hardsub_extractor.gui.PublicCaptionCollector",
                            side_effect=AssertionError("web provider invoked in local OCR")), \
                 mock.patch("hardsub_extractor.gui.AutoEpSiteClient",
                            side_effect=AssertionError("EpSite provider invoked in local OCR")), \
                 mock.patch("hardsub_extractor.gui.bundled_model_dir", return_value=models), \
                 mock.patch("hardsub_extractor.gui.bundled_manifest_path", return_value=manifest):
                gui = ExtractorGUI(tk_root)
                _choose_local_mode(tk_root, gui)
                preview_evidence = _accept_real_one_click_ocr_comparison(
                    tk_root, gui, rootdir / f"{languages[0]}-source.mp4", languages[0],
                )
                for language in languages:
                    source = rootdir / f"{language}-source.mp4"
                    base = rootdir / f"{language}-result"
                    before = len(messages)
                    _start_and_poll(
                        tk_root, gui, base, source, language, "완료",
                    )
                    if len(messages) != before + 1 or messages[-1] != "완료" or errors:
                        raise AssertionError("GUI failed to show successful OCR publication")
                    results.append(_assert_hard_output(base, language))
                    if compare_caption_islands:
                        # Invoke the REAL Tk checkbutton and real Start worker
                        # on the SAME source, model, expected text and timebase.
                        # This is a controlled synthetic A/B, not evidence of
                        # general real-anime quality improvement.
                        if (str(gui.caption_islands_check.cget("state")) != "normal"
                                or gui.caption_islands_var.get()):
                            raise AssertionError("experimental GUI checkbox not ready/default off")
                        gui.caption_islands_check.invoke()
                        try:
                            if not gui.caption_islands_var.get():
                                raise AssertionError("experimental Tk opt-in did not latch")
                            candidate_base = rootdir / f"{language}-caption-islands"
                            before = len(messages)
                            _start_and_poll(
                                tk_root, gui, candidate_base, source, language, "완료",
                            )
                            if len(messages) != before + 1 or errors:
                                raise AssertionError("experimental Tk OCR did not complete")
                            experimental.append(_assert_hard_output(candidate_base, language))
                        finally:
                            gui.caption_islands_check.invoke()
                        if gui.caption_islands_var.get():
                            raise AssertionError("experimental Tk opt-in not reset")

                # Independently verified no-dialogue clip must never be
                # counted as a successful subtitle publication.
                # Negative controls run under both configurations. Never
                # claim a caption-free clip as successful output.
                for candidate_mode in ((False, True) if compare_caption_islands else (False,)):
                    if candidate_mode:
                        gui.caption_islands_check.invoke()
                    try:
                        negative_base = rootdir / (
                            "no-dialogue-islands" if candidate_mode
                            else "no-dialogue-result"
                        )
                        failures = len(errors)
                        _start_and_poll(
                            tk_root, gui, negative_base, blank, languages[0], "실패",
                        )
                        if len(errors) != failures + 1 or errors[-1] != "작업 실패":
                            raise AssertionError("caption-free MP4 did not return a GUI error")
                        if (negative_base.with_suffix(".srt").exists()
                                or negative_base.with_suffix(".ass").exists()):
                            raise AssertionError("caption-free MP4 published spurious output")
                    finally:
                        if candidate_mode:
                            gui.caption_islands_check.invoke()
                if any(_sha(path) != expected for path, expected in sources.items()):
                    raise AssertionError("real GUI OCR altered source MP4 bytes")
        finally:
            if tk_root is not None:
                try:
                    tk_root.destroy()
                except tk.TclError:
                    pass
    return {
        "schema": 1,
        "actual_tk_start_worker_event_completion": True,
        "source": "disposable_encoded_mp4",
        "real_ffmpeg_detector_and_pinned_ppocr": True,
        "languages": list(languages),
        "real_one_click_comparison": preview_evidence,
        "results": results,
        "caption_islands_opt_in_executed": compare_caption_islands,
        "caption_islands_results": experimental,
        "caption_islands_exact_independent_text_cues": sum(
            r["exact_text_cues"] for r in experimental
        ),
        "caption_islands_comparison_same_source_and_models": compare_caption_islands,
        "exact_independent_text_cues": sum(r["exact_text_cues"] for r in results),
        "blank_clip_false_positive_cues": 0,
        "caption_islands_blank_clip_false_positive_cues": (
            0 if compare_caption_islands else None
        ),
        "blank_clip_rejected_as_success": True,
        "original_media_sha256_unchanged": True,
        "provider_requests": 0,
        "no_user_inputs_modified": True,
        "real_anime_quality_verified": False,
        "frozen_exe_gui_verified": False,
        "release_gate_passed": False,
    }


def main() -> int:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--languages", default=",".join(DEFAULT_LANGUAGES),
                        help="Distinct local OCR languages (ko,en,ja,zh)")
    parser.add_argument("--manifest", type=Path, default=ROOT / "models/manifest.json")
    parser.add_argument(
        "--compare-caption-islands", action="store_true",
        help="Explicit same-source actual Tk/PP-OCR caption-island opt-in A/B",
    )
    args = parser.parse_args()
    try:
        selected = tuple(item.strip() for item in args.languages.split(",") if item.strip())
        if (not selected or len(selected) != len(set(selected))
                or any(item not in SUPPORTED_LANGUAGES for item in selected)):
            parser.error("select distinct OCR languages from ko,en,ja,zh")
        results = verify(
            args.model_dir, args.manifest, selected,
            compare_caption_islands=args.compare_caption_islands,
        )
        print("HSE_REAL_TK_HARD_OCR " + json.dumps(results, sort_keys=True))
        return 0
    except Exception as exc:
        print("HSE_REAL_TK_HARD_OCR_ERROR " + json.dumps({
            "error_class": type(exc).__name__,
            "release_gate_passed": False,
        }, sort_keys=True))
        return 3


if __name__ == "__main__":
    raise SystemExit(main())
