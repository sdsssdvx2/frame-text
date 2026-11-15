"""Frozen Windows executable original-frame evidence, using only synthetic media."""
from __future__ import annotations

import hashlib
from io import BytesIO
import json
from pathlib import Path
import queue
import subprocess
import sys
import tempfile
import threading

from PIL import Image
from .runtime_paths import resolve_binary

_UPPER = "固定電視台台標文字"
_LOWER = ("第一句對話", "第二句對話", "第三句對話")
_SPANS = ((0.5, 1.5), (2.0, 3.0), (4.0, 5.0))


def _fixture(root: Path) -> tuple[Path, Path]:
    video = root / "synthetic.mp4"
    process = subprocess.run([
        resolve_binary("ffmpeg"), "-nostdin", "-hide_banner", "-loglevel", "error",
        "-f", "lavfi", "-i", "testsrc2=size=640x360:rate=10:duration=6",
        "-map", "0:v:0", "-an", "-sn", "-dn", "-c:v", "mpeg4",
        "-q:v", "5", "-t", "6", "-y", str(video),
    ], stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
       stderr=subprocess.DEVNULL, timeout=75, check=False)
    if process.returncode or not video.is_file() or not 1024 < video.stat().st_size < 40000000:
        raise RuntimeError("synthetic_media_invalid")
    subtitle = root / "original.srt"
    blocks = []
    def clock(t: float) -> str:
        millis = round(t * 1000)
        return f"00:00:{millis // 1000:02d},{millis % 1000:03d}"
    for index, ((start, end), lower) in enumerate(zip(_SPANS, _LOWER), 1):
        blocks.append(f"{index}\n{clock(start)} --> {clock(end)}\n{_UPPER}\n{lower}\n")
    subtitle.write_text("\n".join(blocks), encoding="utf-8")
    return video, subtitle


def _gui_review(video: Path, subtitle: Path, destination: Path, *,
                mode: str, after: int):
    # Invoke the actual packaged GUI worker without opening Tk or a browser.
    from .gui import ExtractorGUI
    gui = ExtractorGUI.__new__(ExtractorGUI)
    gui.events = queue.Queue()
    gui._cancel_event = threading.Event()
    gui._review_worker(video, subtitle, destination, mode, after, "zh")
    kind, result = gui.events.get_nowait()
    if kind != "review_done" or not gui.events.empty():
        raise AssertionError("frozen_review_worker_failed")
    summary, path = result
    if path != str(destination) or summary["status"] != "CREATED":
        raise AssertionError("frozen_review_worker_result_invalid")
    return summary


def _inspect_private_review_html(dest: Path, expected: tuple[int, ...],
                                 index_data: dict) -> None:
    """Attest the new local OCR+source-frame page in the packaged EXE itself.

    Do not print the source text, image bytes or user filesystem locations.
    A full, actual Chrome rendering was separately proven by Windows run
    43373723775; this is the frozen EXE's structural/source ownership gate.
    """
    page = dest / "review.html"
    if page.is_symlink() or not page.is_file():
        raise AssertionError("frozen_private_html_missing")
    raw = page.read_bytes()
    if not 500 <= len(raw) <= 128 * 1024:
        raise AssertionError("frozen_private_html_outside_bounds")
    html = raw.decode("utf-8", errors="strict")
    if (html.count("<section>") != len(expected)
            or "<script" in html.lower()
            or "script-src 'none'" not in html
            or "connect-src 'none'" not in html
            or "<meta charset=\"utf-8\">" not in html
            or "https://" in html or "http://" in html):
        raise AssertionError("frozen_private_html_not_offline_safe")
    # Index is deliberately text-free, while this separate private page
    # contains the original OCR text and fixed local image filenames.
    for number in expected:
        if html.count(f"자막 {number}번") != 1:
            raise AssertionError("frozen_private_html_missing_cue_label")
        if _UPPER not in html or _LOWER[number - 1] not in html:
            raise AssertionError("frozen_private_html_cue_text_missing")
    n = 0
    for cue in index_data["review_cues"]:
        for frame in cue["frames"]:
            exact = f'src="{frame["image"]}"'
            if html.count(exact) != 1:
                raise AssertionError("frozen_private_html_image_link_invalid")
            n += 1
    if html.count("<img ") != n:
        raise AssertionError("frozen_private_html_unexpected_image_reference")


def _inspect_images(dest: Path, expected: tuple[int, ...]) -> int:
    info = (dest / "index.json").read_bytes()
    if _UPPER.encode() in info or any(row.encode() in info for row in _LOWER):
        raise AssertionError("private_subtitle_leaked_in_index")
    data = json.loads(info)
    if not data.get("review_only") or not data.get("no_caption_text_embedded"):
        raise AssertionError("private_review_metadata_invalid")
    if tuple(x["cue"] for x in data["review_cues"]) != expected:
        raise AssertionError("private_review_cue_mismatch")
    hashes = set()
    count = 0
    for cue in data["review_cues"]:
        if len(cue["frames"]) != 2:
            raise AssertionError("missing_early_or_late_frame")
        for f in cue["frames"]:
            path = dest / f["image"]
            if path.name != f["image"] or not path.name.startswith(f"cue-{cue['cue']:05d}-"):
                raise AssertionError("private_review_frame_name_invalid")
            raw = path.read_bytes()
            digest = hashlib.sha256(raw).hexdigest()
            if digest != f["sha256"]:
                raise AssertionError("private_review_frame_corruption")
            with Image.open(BytesIO(raw)) as image:
                image.verify()
                if image.format != "JPEG" or image.width > 960:
                    raise AssertionError("private_review_frame_format_invalid")
            hashes.add(digest)
            count += 1
    if len(hashes) < 2:
        raise AssertionError("real_moving_video_frames_not_distinct")
    with Image.open(dest / "contact-sheet.jpg") as sheet:
        sheet.verify()
        if sheet.format != "JPEG" or sheet.width != 960:
            raise AssertionError("private_contact_sheet_missing")
    _inspect_private_review_html(dest, expected, data)
    return count


def _verify(root: Path) -> dict:
    from .review_pack import create_pack
    from .subtitle_edit_handoff import (
        find_windows_notepad, validate_proofread_source,
    )
    video, srt = _fixture(root)
    # Prove the zero-install corrector is present in this frozen build without
    # opening Notepad, or ever editing the original source files.
    no_install_editor = find_windows_notepad()
    if no_install_editor is None or no_install_editor.name.lower() != "notepad.exe":
        raise AssertionError("trusted_system_notepad_unavailable")
    checked_video, checked_srt = validate_proofread_source(video, srt)
    if checked_video != video.resolve() or checked_srt != srt.resolve():
        raise AssertionError("frozen_original_correction_source_validation_failed")
    before = (
        hashlib.sha256(video.read_bytes()).hexdigest(),
        hashlib.sha256(srt.read_bytes()).hexdigest(),
    )
    flagged = root / "flagged"
    first = _gui_review(video, srt, flagged, mode="flagged", after=0)
    if (first["cue_count"], first["frame_count"], first["last_cue"]) != (1, 2, 1):
        raise AssertionError("frozen_review_warning_selection_invalid")
    if _inspect_images(flagged, (1,)) != 2:
        raise AssertionError("frozen_review_warning_images_invalid")
    more = root / "resumed"
    second = _gui_review(video, srt, more, mode="all", after=1)
    if (second["cue_count"], second["frame_count"], second["last_cue"]) != (2, 4, 3):
        raise AssertionError("frozen_review_resume_invalid")
    if _inspect_images(more, (2, 3)) != 4:
        raise AssertionError("frozen_review_resume_images_invalid")
    existing = (flagged / "index.json").read_bytes()
    try:
        create_pack(video, srt, flagged, mode="all", language="zh")
    except ValueError:
        pass
    else:
        raise AssertionError("frozen_review_unsafe_overwrite")
    if (flagged / "index.json").read_bytes() != existing:
        raise AssertionError("frozen_review_existing_evidence_changed")
    after = (
        hashlib.sha256(video.read_bytes()).hexdigest(),
        hashlib.sha256(srt.read_bytes()).hexdigest(),
    )
    if after != before:
        raise AssertionError("frozen_review_originals_modified")
    return {
        "gui_worker_invoked": True,
        "flagged_cues": 1, "resumed_cues": 2,
        "jpeg_frames_verified": 6, "contact_sheets_verified": 2,
        "private_offline_html_reports_verified": 2,
        "trusted_windows_notepad_present": True,
        "notepad_process_started": False,
        "proofread_source_validated_without_editing": True,
        "originals_unchanged": True, "existing_evidence_preserved": True,
    }


def run_frozen_frame_review_acceptance(destination: str | Path) -> int:
    frozen = bool(getattr(sys, "frozen", False))
    result = {
        "schema": 1, "type": "frozen_frame_review_acceptance",
        "is_frozen": frozen, "ok": False,
        "synthetic_local_only": True, "network_requests": 0,
        "frame_uploads": 0, "jpeg_frames_verified": 0,
        "temporary_inputs_removed": False,
    }
    try:
        if not frozen:
            raise RuntimeError("packaged_executable_required")
        with tempfile.TemporaryDirectory(prefix="hse-frozen-review-") as td:
            result.update(_verify(Path(td)))
        result["temporary_inputs_removed"] = True
        result["ok"] = (
            result["gui_worker_invoked"]
            and result["flagged_cues"] == 1
            and result["resumed_cues"] == 2
            and result["jpeg_frames_verified"] == 6
            and result["contact_sheets_verified"] == 2
            and result["private_offline_html_reports_verified"] == 2
            and result["trusted_windows_notepad_present"]
            and not result["notepad_process_started"]
            and result["proofread_source_validated_without_editing"]
            and result["originals_unchanged"]
            and result["existing_evidence_preserved"]
        )
    except Exception as exc:
        result["failure_type"] = type(exc).__name__
    path = Path(destination)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    return 0 if result["ok"] else 3
