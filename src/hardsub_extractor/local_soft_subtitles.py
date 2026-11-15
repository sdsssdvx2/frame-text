"""Conservative local-video embedded *text* subtitle priority.

Read the user's existing local container (never edit it).  Select only an
explicitly language-tagged text subtitle stream; unknown-language tracks,
bitmap subtitles and unrelated languages do not silently masquerade as the
requested OCR language.  Decode selected text through bundled ffprobe/ffmpeg,
preserve the demuxed SRT representation and provenance, then export SRT/ASS
exclusively.  A missing/invalid matching text track returns None so the
caller may lazily run burned-in frame OCR.

A muxed subtitle's original on-disk byte representation cannot be promised:
the retained .embedded.extracted.srt is FFmpeg's decoded/exported stream, not
a byte-exact copy of the container's internal track.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable
import json
import subprocess
import tempfile
import time

from .model_assets import normalize_ocr_language
from .runtime_paths import resolve_binary
from .subtitle_formats import (
    SubtitleFormatError, parse_subtitle_bytes, subtitle_cues_have_korean,
)
from .subtitle_language import language_matches
from .writers import write_subtitle_pair_exclusive
from .local_sidecar_subtitles import _outputs as _all_local_outputs

_MAX_PROBE_BYTES = 1024 * 1024
_MAX_TEXT_BYTES = 20 * 1024 * 1024
_MAX_STREAMS = 128
_TEXT_CODECS = frozenset({"subrip", "ass", "ssa", "webvtt", "mov_text", "text"})
_OLD_ISO_TAGS = {"eng": "en", "jpn": "ja", "kor": "ko", "chi": "zh", "zho": "zh"}


class LocalSoftSubtitleError(RuntimeError):
    """A local-container preflight or decoding operation was not trustworthy."""


@dataclass(frozen=True, slots=True)
class EmbeddedSubtitleStream:
    index: int
    codec: str
    language: str
    default: bool
    forced: bool


@dataclass(frozen=True, slots=True)
class LocalSoftSubtitleResult:
    mode: str
    cue_count: int
    srt_path: Path
    ass_path: Path
    extracted_track_path: Path
    provenance_path: Path
    stream_index: int
    track_language: str


def _run_bounded(
    command: list[str],
    *,
    max_bytes: int,
    timeout_seconds: float,
    check_cancelled: Callable[[], None] | None = None,
) -> bytes:
    """Cap output memory, deadline and child lifetime without stderr leakage."""
    if check_cancelled is not None:
        check_cancelled()
    try:
        with tempfile.TemporaryFile() as captured:
            process = subprocess.Popen(
                command,
                stdin=subprocess.DEVNULL,
                stdout=captured,
                stderr=subprocess.DEVNULL,
            )
            try:
                deadline = time.monotonic() + timeout_seconds
                while process.poll() is None:
                    if check_cancelled is not None:
                        check_cancelled()
                    if time.monotonic() >= deadline:
                        raise LocalSoftSubtitleError("local subtitle probe or decode timed out")
                    time.sleep(min(0.2, max(0.0, deadline - time.monotonic())))
                if check_cancelled is not None:
                    check_cancelled()
                if process.returncode != 0:
                    raise LocalSoftSubtitleError("local subtitle probe or decode failed")
                captured.seek(0, 2)
                if captured.tell() > max_bytes:
                    raise LocalSoftSubtitleError("local embedded subtitle output exceeds size limit")
                captured.seek(0)
                return captured.read(max_bytes + 1)
            except BaseException:
                if process.poll() is None:
                    process.kill()
                process.wait()
                raise
    except OSError as exc:
        raise LocalSoftSubtitleError("local FFmpeg/FFprobe process unavailable") from exc


def _local_file(source: str | Path) -> Path:
    path = Path(source).expanduser()
    if not path.is_file():
        raise ValueError("local subtitle source must be an existing video file")
    return path.resolve(strict=True)


def _requested_matches(chosen: str, tag: str) -> bool:
    tag = str(tag).strip().lower().replace("_", "-")
    tag = _OLD_ISO_TAGS.get(tag, tag)
    return language_matches(chosen, tag)


def _probe_streams(
    source: Path,
    *,
    check_cancelled: Callable[[], None] | None = None,
) -> list[EmbeddedSubtitleStream]:
    cmd = [
        resolve_binary("ffprobe"), "-v", "error",
        "-protocol_whitelist", "file,pipe",
        "-show_entries",
        "stream=index,codec_name,codec_type:stream_tags=language:stream_disposition=default,forced",
        "-of", "json", "-i", str(source),
    ]
    raw = _run_bounded(
        cmd, max_bytes=_MAX_PROBE_BYTES, timeout_seconds=40,
        check_cancelled=check_cancelled,
    )
    try:
        parsed = json.loads(raw)
        streams = parsed["streams"]
        if not isinstance(streams, list) or len(streams) > _MAX_STREAMS:
            raise ValueError("invalid stream inventory")
    except (ValueError, KeyError, TypeError) as exc:
        raise LocalSoftSubtitleError("invalid local subtitle stream inventory") from exc

    output = []
    seen: set[int] = set()
    for item in streams:
        if not isinstance(item, dict):
            raise LocalSoftSubtitleError("invalid local subtitle stream record")
        index = item.get("index")
        if not isinstance(index, int) or isinstance(index, bool) or index < 0 or index > 4096 or index in seen:
            raise LocalSoftSubtitleError("invalid local subtitle stream index")
        seen.add(index)
        if item.get("codec_type") != "subtitle":
            continue
        codec = str(item.get("codec_name") or "").lower()
        if codec not in _TEXT_CODECS:
            continue  # PGS/VobSub/DVB/image-based tracks cannot be read as text.
        tag = (item.get("tags") or {}).get("language", "")
        if not isinstance(tag, str) or len(tag) > 32:
            continue
        disposition = item.get("disposition") or {}
        if not isinstance(disposition, dict):
            disposition = {}
        output.append(EmbeddedSubtitleStream(
            index=index, codec=codec, language=tag,
            default=disposition.get("default") == 1,
            forced=disposition.get("forced") == 1,
        ))
    return output


def _paths(out_base: Path) -> tuple[Path, Path, Path, Path]:
    return (
        out_base.with_suffix(".srt"),
        out_base.with_suffix(".ass"),
        out_base.with_name(out_base.name + ".embedded.extracted.srt"),
        out_base.with_name(out_base.name + ".source.json"),
    )


def _guard_outputs(out_base: Path) -> None:
    # One output namespace is shared by embedded and external original tracks.
    # A sidecar provenance/raw orphan must prevent an embedded stream from
    # publishing a contradictory result under the same output base.
    for path in _all_local_outputs(out_base):
        if path.exists() or path.is_symlink():
            raise FileExistsError(
                "기존 자막/내장자막 출처 결과가 있습니다. 다른 결과 경로를 지정하세요."
            )


def _export_stream(
    source: Path,
    stream: EmbeddedSubtitleStream,
    *,
    check_cancelled: Callable[[], None] | None = None,
) -> bytes:
    return _run_bounded(
        [
            resolve_binary("ffmpeg"),
            "-nostdin", "-hide_banner", "-loglevel", "error",
            "-protocol_whitelist", "file,pipe",
            "-i", str(source), "-map", "0:" + str(stream.index),
            "-vn", "-an", "-c:s", "srt", "-f", "srt", "pipe:1",
        ],
        max_bytes=_MAX_TEXT_BYTES,
        timeout_seconds=180,
        check_cancelled=check_cancelled,
    )


def acquire_local_embedded_text(
    source: str | Path,
    out_base: str | Path,
    *,
    language: str = "ko",
    check_cancelled: Callable[[], None] | None = None,
) -> LocalSoftSubtitleResult | None:
    """Return a verified embedded-text result, otherwise None for frame OCR.

    No OCR/model loading/network requests. Unknown-language text never wins by
    guessing from Chinese/Japanese Han characters. Existing outputs are not
    replaced or partially claimed on errors, including races.
    """
    chosen = normalize_ocr_language(language)
    video = _local_file(source)
    base = Path(out_base).expanduser()
    _guard_outputs(base)
    tracks = [
        x for x in _probe_streams(video, check_cancelled=check_cancelled)
        if _requested_matches(chosen, x.language)
    ]
    # A forced/signs-only subtitle should lose to a normal/default full track.
    tracks.sort(key=lambda x: (x.forced, not x.default, x.index))
    for track in tracks:
        if check_cancelled is not None:
            check_cancelled()
        try:
            raw = _export_stream(video, track, check_cancelled=check_cancelled)
            fmt, cues = parse_subtitle_bytes(raw, source_url="embedded.srt")
            if fmt != "srt" or not cues:
                continue
            if chosen == "ko" and not subtitle_cues_have_korean(cues):
                continue
        except (LocalSoftSubtitleError, SubtitleFormatError):
            # Another matching stream can still be usable. If none is, the
            # caller may attempt genuine visible hard-subtitle OCR.
            continue

        if check_cancelled is not None:
            check_cancelled()
        _guard_outputs(base)
        srt_path, ass_path, raw_path, provenance_path = _paths(base)
        created: list[Path] = []
        try:
            srt_path, ass_path = write_subtitle_pair_exclusive(cues, base)
            created.extend((srt_path, ass_path))
            with raw_path.open("xb") as output:
                created.append(raw_path)
                output.write(raw)
            report = {
                "schema": 1,
                "mode": "local-embedded-text",
                "language_requested": chosen,
                "language_tag": track.language,
                "stream_index": track.index,
                "source_codec": track.codec,
                "is_default": track.default,
                "is_forced": track.forced,
                "cue_count": len(cues),
                "source_video_modified": False,
                "ocr_applied": False,
                "source_stream_export": "ffmpeg_decoded_srt_not_original_container_bytes",
                "output_timeline": "source_video",
                "extracted_stream_file": raw_path.name,
                "srt_file": srt_path.name,
                "ass_file": ass_path.name,
            }
            with provenance_path.open("x", encoding="utf-8") as output:
                created.append(provenance_path)
                json.dump(report, output, ensure_ascii=False, indent=2)
                output.write("\n")
        except BaseException:
            for dest in reversed(created):
                try:
                    dest.unlink(missing_ok=True)
                except OSError:
                    pass
            raise
        return LocalSoftSubtitleResult(
            mode="local-embedded-text",
            cue_count=len(cues),
            srt_path=srt_path,
            ass_path=ass_path,
            extracted_track_path=raw_path,
            provenance_path=provenance_path,
            stream_index=track.index,
            track_language=track.language,
        )
    return None
