"""Import an explicitly language-tagged subtitle beside a local video before OCR.

Automatic selection considers only video.<language>.{srt,vtt,ass,ssa} in the same
directory. An explicitly selected existing file may have any name/location.
Untagged files are never auto-selected; guessing their language could make
a Japanese/Chinese sidecar masquerade as a requested Korean subtitle. The
original subtitle bytes and the user's video are never modified.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Callable

from .model_assets import normalize_ocr_language
from .subtitle_formats import SubtitleFormatError, parse_subtitle_bytes, subtitle_cues_have_korean
from .writers import write_subtitle_pair_exclusive

_MAX_SIDECAR_BYTES = 20 * 1024 * 1024
_FORMATS = ("srt", "vtt", "ass", "ssa")
_TAGS = {
    "ko": ("ko", "ko-KR", "kor"),
    "en": ("en", "en-US", "en-GB", "eng"),
    "ja": ("ja", "ja-JP", "jpn"),
    "zh": ("zh", "zh-Hans", "zh-Hant", "zh-CN", "zh-TW", "chi", "zho"),
}


@dataclass(frozen=True, slots=True)
class LocalSidecarSubtitleResult:
    mode: str
    cue_count: int
    srt_path: Path
    ass_path: Path
    extracted_track_path: Path
    provenance_path: Path
    track_language: str


def _outputs(base: Path) -> tuple[Path, ...]:
    return (
        base.with_suffix(".srt"),
        base.with_suffix(".ass"),
        base.with_name(base.name + ".source.json"),
        base.with_name(base.name + ".embedded.extracted.srt"),
        base.with_name(base.name + ".embedded.image-ocr.srt"),
        *(base.with_name(base.name + ".sidecar.original." + ext) for ext in _FORMATS),
    )


def _check_outputs(base: Path) -> None:
    if any(path.exists() or path.is_symlink() for path in _outputs(base)):
        raise FileExistsError("기존 자막/출처 결과가 있습니다. 다른 저장 경로를 선택하세요.")


def _read_unchanged(path: Path) -> bytes | None:
    """Refuse symlinks, oversized inputs and files changing during our read."""
    if path.is_symlink() or not path.is_file():
        return None
    before = path.stat()
    if not (0 < before.st_size <= _MAX_SIDECAR_BYTES):
        return None
    # A file may grow after stat(): enforce the size cap during the actual read.
    with path.open("rb") as stream:
        raw = stream.read(_MAX_SIDECAR_BYTES + 1)
    after = path.stat()
    if (len(raw) > _MAX_SIDECAR_BYTES or len(raw) != before.st_size or before.st_size != after.st_size
            or before.st_mtime_ns != after.st_mtime_ns
            or before.st_ino != after.st_ino):
        return None
    return raw


def acquire_local_sidecar_text(
    video_path: str | Path,
    out_base: str | Path,
    *,
    language: str = "ko",
    check_cancelled: Callable[[], None] | None = None,
    selected_path: str | Path | None = None,
) -> LocalSidecarSubtitleResult | None:
    """Use a matching local sidecar, if valid; otherwise let caller try OCR.

    No directory-wide search, browser, network, model or media decoder calls.
    The selected source is copied byte-for-byte into exclusive output and its
    normalized SRT/ASS share the exact source timeline. An explicitly selected
    file has priority and MUST fail closed when invalid, never trigger OCR.
    """
    chosen = normalize_ocr_language(language)
    video = Path(video_path).expanduser()
    explicit = selected_path is not None
    if not video.is_file():
        if explicit:
            raise ValueError("선택한 원본 영상 파일이 없습니다.")
        return None
    base = Path(out_base).expanduser()
    _check_outputs(base)
    if explicit:
        selected = Path(selected_path).expanduser()
        if selected.suffix.lower().lstrip(".") not in _FORMATS:
            raise ValueError("선택한 원본 자막은 SRT/VTT/ASS/SSA 파일이어야 합니다.")
        candidates = [(selected, chosen)]
    else:
        candidates = [
            (video.with_name(f"{video.stem}.{tag}.{ext}"), tag)
            for tag in _TAGS[chosen] for ext in _FORMATS
        ]
    for candidate, tag in candidates:
            if check_cancelled is not None:
                check_cancelled()
            ext = candidate.suffix.lower().lstrip(".")
            raw = _read_unchanged(candidate)
            if raw is None:
                if explicit:
                    raise ValueError("선택한 원본 자막을 안전하게 읽을 수 없습니다.")
                continue
            try:
                fmt, cues = parse_subtitle_bytes(raw, source_url=f"sidecar.{ext}")
            except SubtitleFormatError as exc:
                if explicit:
                    raise ValueError("선택한 원본 자막 형식이 잘못되었습니다.") from exc
                continue
            if fmt != ("ass" if ext == "ssa" else ext) or not cues:
                if explicit:
                    raise ValueError("선택한 원본 자막의 내용과 확장자가 일치하지 않습니다.")
                continue
            if chosen == "ko" and not subtitle_cues_have_korean(cues):
                if explicit:
                    raise ValueError("선택한 자막에 한국어 텍스트가 충분하지 않습니다.")
                continue

            if check_cancelled is not None:
                check_cancelled()
            _check_outputs(base)
            raw_dest = base.with_name(base.name + f".sidecar.original.{ext}")
            provenance = base.with_name(base.name + ".source.json")
            created: list[Path] = []
            try:
                srt, ass = write_subtitle_pair_exclusive(cues, base)
                created.extend((srt, ass))
                with raw_dest.open("xb") as output:
                    created.append(raw_dest)
                    output.write(raw)
                report = {
                    "schema": 1,
                    "mode": "local-selected-text" if explicit else "local-sidecar-text",
                    "language_requested": chosen,
                    "language_tag": tag,
                    "selection": "explicit" if explicit else "adjacent",
                    "original_format": fmt,
                    "source_sidecar_name": candidate.name,
                    "original_sha256": hashlib.sha256(raw).hexdigest(),
                    "original_bytes_retained": True,
                    "source_video_modified": False,
                    "source_sidecar_modified": False,
                    "ocr_applied": False,
                    "output_timeline": "source_video",
                    "cue_count": len(cues),
                    "extracted_stream_file": raw_dest.name,
                    "srt_file": srt.name,
                    "ass_file": ass.name,
                }
                with provenance.open("x", encoding="utf-8") as output:
                    created.append(provenance)
                    json.dump(report, output, ensure_ascii=False, indent=2)
                    output.write("\n")
            except BaseException:
                for path in reversed(created):
                    try:
                        path.unlink(missing_ok=True)
                    except OSError:
                        pass
                raise
            return LocalSidecarSubtitleResult(
                "local-selected-text" if explicit else "local-sidecar-text",
                len(cues), srt, ass, raw_dest, provenance, tag,
            )
    return None
