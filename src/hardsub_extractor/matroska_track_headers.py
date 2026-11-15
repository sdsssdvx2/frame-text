"""Read-only bounded Matroska EBML TRACK HEADER reader (not media decoder).

Only authoritative TrackNumber, TrackType, CodecID and explicit language
metadata are used. No cue packets, frames, tags, attachments, subtitles,
network, user configuration or media writes are read. Unknown/truncated
layouts fail closed, never guessed from FFprobe stream index.

This small parser is necessary because FFprobe stream.id is often empty or
duplicated on Matroska even though the on-disk TrackNumber is defined.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

_MAX_HEADER_BYTES = 16 * 1024 * 1024
_MAX_TRACKS_BYTES = 2 * 1024 * 1024
_MAX_TRACK_ENTRY = 256 * 1024
_MAX_TRACKS = 128
_SEGMENT = 0x18538067
_TRACKS = 0x1654AE6B
_TRACK_ENTRY = 0xAE
_NUMBER = 0xD7
_TYPE = 0x83
_CODEC_ID = 0x86
_LANGUAGE = 0x22B59C
_LANGUAGE_IETF = 0x22B59D
_DEFAULT = 0x88
_FORCED = 0x55AA
_SUBTITLE_TYPE = 0x11
_CODECS = {
    "S_HDMV/PGS": "hdmv_pgs_subtitle",
    "S_VOBSUB": "dvd_subtitle",
}


@dataclass(frozen=True, slots=True)
class MatroskaImageTrack:
    track_number: int
    codec: str
    language: str
    default: bool
    forced: bool


def _header(buf: bytes, offset: int, *, identifier: bool) -> tuple[int | None, int]:
    if offset >= len(buf):
        raise ValueError("truncated Matroska header")
    first = buf[offset]
    marker, size = 0x80, 1
    while marker and not (first & marker):
        marker >>= 1
        size += 1
    if marker == 0 or size > (4 if identifier else 8):
        raise ValueError("invalid EBML variable integer length")
    if offset + size > len(buf):
        raise ValueError("truncated EBML variable integer")
    value = first if identifier else first & (marker - 1)
    for byte in buf[offset + 1:offset + size]:
        value = (value << 8) | byte
    if not identifier and value == (1 << (7 * size)) - 1:
        return None, offset + size  # Unknown size: valid for Segment only.
    return value, offset + size


def _elements(buf: bytes, start: int, end: int, *, allow_incomplete=False):
    if start < 0 or end < start:
        raise ValueError("invalid EBML element bounds")
    pos = start
    while pos < end:
        if pos >= len(buf):
            if allow_incomplete:
                return
            raise ValueError("Matroska tracks past bounded read")
        element_id, after_id = _header(buf, pos, identifier=True)
        size, body_start = _header(buf, after_id, identifier=False)
        if size is None:
            # The outer Segment alone may be unknown-size.
            if element_id != _SEGMENT:
                raise ValueError("unknown-sized non-Segment EBML element")
            body_end = end
        else:
            body_end = body_start + size
        if body_end > end:
            if element_id == _SEGMENT and allow_incomplete:
                # The Segment commonly spans an entire multi-GB video.
                # Its children (Tracks) still fit near the beginning.
                body_end = end
            elif element_id == _TRACKS:
                raise ValueError("incomplete or oversized Matroska Tracks metadata")
            elif allow_incomplete:
                return
            else:
                raise ValueError("Matroska element extends beyond bounded section")
        yield element_id, body_start, body_end
        pos = body_end


def _number(buf: bytes, start: int, end: int) -> int:
    if end <= start or end - start > 8:
        raise ValueError("invalid EBML unsigned integer")
    return int.from_bytes(buf[start:end], "big")


def _text(buf: bytes, start: int, end: int, maxlen: int) -> str:
    if end < start or end - start > maxlen:
        raise ValueError("invalid EBML string length")
    return buf[start:end].decode("utf-8", errors="strict").rstrip("\x00")


def _parse_track_entry(buf: bytes, start: int, end: int) -> MatroskaImageTrack | None:
    if end - start > _MAX_TRACK_ENTRY:
        raise ValueError("oversized Matroska TrackEntry")
    fields: dict[int, tuple[int, int]] = {}
    for element_id, body_start, body_end in _elements(buf, start, end):
        if element_id in (_NUMBER, _TYPE, _CODEC_ID, _LANGUAGE,
                          _LANGUAGE_IETF, _DEFAULT, _FORCED):
            if element_id in fields:
                raise ValueError("duplicate critical Matroska track metadata")
            fields[element_id] = body_start, body_end
    if _NUMBER not in fields or _TYPE not in fields or _CODEC_ID not in fields:
        raise ValueError("incomplete Matroska TrackEntry")
    track_number = _number(buf, *fields[_NUMBER])
    track_type = _number(buf, *fields[_TYPE])
    codec = _text(buf, *fields[_CODEC_ID], 64)
    if track_type != _SUBTITLE_TYPE or codec not in _CODECS:
        return None
    if not 1 <= track_number <= 4096:
        raise ValueError("invalid Matroska TrackNumber")
    # IMPORTANT: default-language "eng" is not inferred. An actual
    # Language or LanguageIETF element must exist for an exact match.
    if _LANGUAGE_IETF in fields:
        language = _text(buf, *fields[_LANGUAGE_IETF], 32)
    elif _LANGUAGE in fields:
        language = _text(buf, *fields[_LANGUAGE], 32)
    else:
        language = ""
    default = _number(buf, *fields[_DEFAULT]) != 0 if _DEFAULT in fields else True
    forced = _number(buf, *fields[_FORCED]) != 0 if _FORCED in fields else False
    return MatroskaImageTrack(track_number, _CODECS[codec], language, default, forced)


def _read_tracks(buf: bytes, start: int, end: int) -> list[MatroskaImageTrack]:
    if end - start > _MAX_TRACKS_BYTES:
        raise ValueError("oversized Matroska Tracks element")
    result: list[MatroskaImageTrack] = []
    seen: set[int] = set()
    for element_id, body_start, body_end in _elements(buf, start, end):
        if element_id != _TRACK_ENTRY:
            continue
        if len(seen) >= _MAX_TRACKS:
            raise ValueError("Matroska TrackEntry budget exceeded")
        track = _parse_track_entry(buf, body_start, body_end)
        if track is not None:
            if track.track_number in seen:
                raise ValueError("duplicate image TrackNumber")
            seen.add(track.track_number)
            result.append(track)
    return result


def read_image_track_headers(source: Path) -> list[MatroskaImageTrack]:
    """Get real MKV TrackNumbers from headers only; return [] if no tracks.

    A malformed/truncated header is a ValueError. One large container read
    does not mean all video content was ingested; max 16MiB independent
    of total video length, no decode or seeking arbitrary user paths.
    """
    if not source.is_file() or source.suffix.lower() != ".mkv":
        return []
    with source.open("rb") as file:
        buf = file.read(_MAX_HEADER_BYTES)
    if len(buf) < 12 or buf[:4] != b"\x1a\x45\xdf\xa3":
        return []
    for element_id, body_start, body_end in _elements(
        buf, 0, len(buf), allow_incomplete=True,
    ):
        if element_id != _SEGMENT:
            continue
        for nested_id, nested_start, nested_end in _elements(
            buf, body_start, body_end, allow_incomplete=True,
        ):
            if nested_id != _TRACKS:
                continue
            if nested_end > len(buf):
                raise ValueError("Matroska Tracks not within bounded header")
            return _read_tracks(buf, nested_start, nested_end)
        break
    return []
