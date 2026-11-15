"""One-track Matroska bitmap -> BDN PNG -> HSE's existing multilingual OCR.

The pinned free seconv performs *image-to-image* pass-through: no bundled
Tesseract, extra PaddleOCR process, language pack download, or network.
The PNG + index.xml are ephemeral. Only parsed subtitle cues are published
by the caller, with explicit OCR provenance (never as original source text).

BDN time codes quantize to frames. This is NOT an original 1ms PTS proof.
Source subtitles, XML data, image files and helper stdout are treated as
untrusted and processed in an isolated directory under strict limits.
"""
from __future__ import annotations

import math
from pathlib import Path
import re
import xml.etree.ElementTree as ET

import cv2
import numpy as np
from PIL import Image, UnidentifiedImageError

from .models import Cue, Frame

_MAX_INDEX = 4 * 1024 * 1024
_MAX_EVENTS = 6000
_MAX_PNG_BYTES = 8 * 1024 * 1024
_MAX_PIXELS = 12_000_000
_MAX_WIDTH = 6000
_MAX_HEIGHT = 4000
_MAX_CUE_SECONDS = 30 * 60
_MAX_DURATION_SECONDS = 6 * 3600
_IMAGE_NAME = re.compile(r"[0-9]{4,8}\.png\Z", re.I)
_TC = re.compile(r"([0-9]{2}):([0-5][0-9]):([0-5][0-9]):([0-9]{2})\Z")


class BitmapBdnOcrError(RuntimeError):
    """Untrusted image extraction or OCR result failed closed."""


def _timecode(tc: str, fps: float) -> float:
    match = _TC.fullmatch(tc)
    if match is None:
        raise BitmapBdnOcrError("invalid image subtitle BDN timecode")
    hours, minutes, seconds, frame = map(int, match.groups())
    if hours > 6 or frame >= math.ceil(fps):
        raise BitmapBdnOcrError("out of range image subtitle timecode")
    v = hours * 3600 + minutes * 60 + seconds + frame / fps
    # ExportHandlerBdnXml.ToHHMMSSFF in Subtitle Edit 5.3 divides
    # fractional-rate source milliseconds by 1.001 before formatting.
    # This is a lossy, frame-quantized timestamp, not source exact PTS.
    if not fps.is_integer():
        v *= 1.001
    return v


def _parse_index(directory: Path) -> list[tuple[float, float, Path]]:
    index = directory / "index.xml"
    if index.is_symlink() or not index.is_file():
        raise BitmapBdnOcrError("no isolated BDN image index")
    if not 0 < index.stat().st_size <= _MAX_INDEX:
        raise BitmapBdnOcrError("oversized BDN image index")
    raw = index.read_bytes()
    if (len(raw) > _MAX_INDEX or b"<!DOCTYPE" in raw.upper()
            or b"<!ENTITY" in raw.upper()):
        raise BitmapBdnOcrError("unsafe BDN image XML")
    try:
        root = ET.fromstring(raw)
        if root.tag != "BDN":
            raise ValueError("unexpected BDN XML root")
        description = root.find("Description")
        format_tag = description.find("Format") if description is not None else None
        if format_tag is None or format_tag.get("DropFrame") != "false":
            raise ValueError("unsupported bitmap subtitle timecode standard")
        fps = float(format_tag.get("FrameRate", "0"))
        if not math.isfinite(fps) or not 1.0 <= fps <= 120.0:
            raise ValueError("bad frame rate")
        events = root.find("Events")
        if events is None or events.get("Type") is not None:
            raise ValueError("invalid image subtitle events")
        items = list(events)
        if not 1 <= len(items) <= _MAX_EVENTS:
            raise ValueError("invalid image subtitle event count")
        out: list[tuple[float, float, Path]] = []
        prior = -1.0
        seen: set[str] = set()
        for event in items:
            if event.tag != "Event":
                raise ValueError("invalid image event type")
            start = _timecode(event.attrib["InTC"], fps)
            end = _timecode(event.attrib["OutTC"], fps)
            if (not math.isfinite(start) or not math.isfinite(end)
                    or start < 0 or end <= start or end > _MAX_DURATION_SECONDS
                    or end - start > _MAX_CUE_SECONDS or start < prior):
                raise ValueError("invalid image event times")
            graphic = list(event)
            if len(graphic) != 1 or graphic[0].tag != "Graphic":
                raise ValueError("ambiguous image event geometry")
            name = graphic[0].text or ""
            if not _IMAGE_NAME.fullmatch(name) or name in seen:
                raise ValueError("invalid or duplicate image filename")
            seen.add(name)
            image = directory / name
            if image.is_symlink() or not image.is_file():
                raise ValueError("missing isolated bitmap file")
            if not 0 < image.stat().st_size <= _MAX_PNG_BYTES:
                raise ValueError("oversized bitmap file")
            # Geometry must match the stored PNG; otherwise the source
            # converted images and declared timestamps could be unrelated.
            for key in ("Width", "Height", "X", "Y"):
                if key not in graphic[0].attrib:
                    raise ValueError("incomplete image geometry")
                value = int(graphic[0].attrib[key])
                if not 0 <= value <= 10000:
                    raise ValueError("untrusted image geometry")
            out.append((start, end, image))
            prior = start
        return out
    except (ET.ParseError, KeyError, TypeError, ValueError, OverflowError) as exc:
        raise BitmapBdnOcrError("invalid isolated BDN image index") from exc


def _image_to_bgr(path: Path, *, white: bool, neutral_canvas: bool = False) -> np.ndarray:
    if path.is_symlink():
        raise BitmapBdnOcrError("symlink in image subtitle conversion")
    try:
        with Image.open(path) as original:
            if original.format != "PNG":
                raise ValueError("not a PNG")
            width, height = original.size
            if (width < 2 or height < 2 or width > _MAX_WIDTH
                    or height > _MAX_HEIGHT or width * height > _MAX_PIXELS):
                raise ValueError("image dimensions not bounded")
            glyphs = original.convert("RGBA")
        if neutral_canvas:
            # Pinned models are calibrated on full video-sized frames. Direct
            # inference on transparent BDN crops can confidently omit glyphs
            # or whole lines. Restore a bounded neutral 720p geometry WITHOUT
            # video access, user data, language guesses, or oracle text.
            # Existing source timing and the actual glyph pixels are retained.
            maximum_width, maximum_height = 1280, 685
            scale = min(
                1.0, maximum_width / width, maximum_height / height,
            )
            if scale < 1.0:
                glyphs = glyphs.resize((
                    max(2, round(width * scale)),
                    max(2, round(height * scale)),
                ), resample=Image.Resampling.LANCZOS)
            canvas = Image.new("RGB", (1280, 720), (36, 43, 59))
            offset = (
                (1280 - glyphs.width) // 2,
                720 - glyphs.height - 35,
            )
            canvas.paste(glyphs, offset, mask=glyphs.getchannel("A"))
            return cv2.cvtColor(np.asarray(canvas), cv2.COLOR_RGB2BGR)
        rgba = np.asarray(glyphs, dtype=np.uint8)
        alpha = rgba[:, :, 3:4].astype(np.float32) / 255.0
        bg = 255.0 if white else 0.0
        rgb = (rgba[:, :, :3] * alpha + bg * (1.0 - alpha)).astype(np.uint8)
        return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    except (OSError, ValueError, UnidentifiedImageError, Image.DecompressionBombError) as exc:
        raise BitmapBdnOcrError("invalid converted PNG subtitle") from exc


def recognize_bdn_images(
    directory: Path, ocr, check_cancelled=None,
    *, audit: dict[str, int] | None = None,
) -> list[Cue]:
    """Read source-exact bitmap positions and frame-quantized timing.

    At most one bitmap plus OCR outputs are retained per event; no all-image
    memory accumulation even with thousands of original subtitle events.
    No captions fabricated for images where the existing HSE OCR is blank.
    """
    entries = _parse_index(directory)
    cues: list[Cue] = []
    for start, end, image in entries:
        if check_cancelled:
            check_cancelled()
        # Do not decode each PNG a second time when normalized OCR succeeds.
        # For long episodes this avoids a redundant image read per event.
        frame = None
        # Source bitmap pictures contain subtitle glyphs already separated
        # from the video. Video-scene filters can accept the lower dialogue
        # row while silently discarding an upper row; a nonblank *partial*
        # result previously prevented the full-bitmap reader from running.
        # Prefer the existing pinned-language bitmap reader in this path
        # only. Normal hardcoded-video OCR remains unchanged.
        base = getattr(ocr, "base", ocr)
        bitmap_reader = getattr(base, "recognize_cropped_bitmap", None)
        if callable(bitmap_reader):
            # Genuine pinned KO/ZH model studies found raw BDN crops can
            # produce highly confident garbage, while neutral 720p geometry
            # recovered all four affected synthetic source-text references.
            # The visual reference NEVER participates in actual OCR.
            selected = bitmap_reader(Frame(
                start, _image_to_bgr(
                    image, white=False, neutral_canvas=True,
                ),
            ))
            if not selected.text.strip():
                # Guard uncertain media: retain the original black/white
                # image readers and legacy scene reader as fallbacks.
                frame = Frame(start, _image_to_bgr(image, white=False))
                selected = bitmap_reader(frame)
            if not selected.text.strip():
                selected = bitmap_reader(Frame(
                    start, _image_to_bgr(image, white=True),
                ))
            if not selected.text.strip():
                selected = ocr.recognize(frame)
                if not selected.text.strip():
                    selected = ocr.recognize(Frame(
                        start, _image_to_bgr(image, white=True),
                    ))
        else:
            frame = Frame(start, _image_to_bgr(image, white=False))
            selected = ocr.recognize(frame)
            if not selected.text.strip():
                selected = ocr.recognize(Frame(
                    start, _image_to_bgr(image, white=True),
                ))
        if selected.text.strip():
            cues.append(Cue(
                start, end, selected.text.strip(),
                max(0.0, min(1.0, float(selected.confidence))),
            ))
    if check_cancelled:
        check_cancelled()
    if not cues:
        raise BitmapBdnOcrError("no recognized bitmap subtitle text")
    if audit is not None:
        # Complete only after the entire source image track has been read.
        # This counts events with nonblank OCR; it is NOT transcription
        # accuracy, and missing source events must not be silently hidden.
        audit.clear()
        audit.update({
            "source_bitmap_events": len(entries),
            "recognized_bitmap_events": len(cues),
            "unrecognized_bitmap_events": len(entries) - len(cues),
        })
    return cues
