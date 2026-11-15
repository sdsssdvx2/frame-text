"""Read-only local 1–50 episode restart audit for ORIGINAL text subtitles.

This is structural integrity, not real-anime OCR ground truth or proof that
a same-named media file is byte-for-byte identical to a historic input.
Legacy local text provenance schema 1 contains no source-video SHA-256.
Never infer that an unmanifested SRT/ASS-only OCR pair is verified.

No video decoding, network, OCR engine, filesystem writes, automatic cleanup,
user-provided output names, or manifest-derived path traversal. Every path
read here is independently computed from the requested output base.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path

from .local_sidecar_subtitles import _FORMATS, _TAGS, _outputs
from .local_soft_subtitles import _TEXT_CODECS, _requested_matches
from .model_assets import normalize_ocr_language
from .public_output_audit import (
    _manifest, _matching_cues, _parse, _present, _safe_read,
    _MAX_SUBTITLE_BYTES,
)
from .subtitle_formats import SubtitleFormatError


@dataclass(frozen=True, slots=True)
class LocalOutputAudit:
    state: str  # absent / verified / preserved
    mode: str = ""
    cue_count: int = 0


def audit_local_output(
    video: str | Path, base: str | Path, *, language: str = "ko",
) -> LocalOutputAudit:
    """Return verified only for complete, internally consistent soft tracks.

    Uses strictly bounded read-only files; a legacy hard-OCR result without
    a source-binding manifest must remain preserved, not verified.
    """
    chosen = normalize_ocr_language(language)
    source = Path(video)
    out = Path(base)
    paths = _outputs(out)
    if not any(_present(path) for path in paths):
        return LocalOutputAudit("absent")
    preserved = LocalOutputAudit("preserved")
    srt = out.with_suffix(".srt")
    ass = out.with_suffix(".ass")
    manifest = out.with_name(out.name + ".source.json")
    embedded = out.with_name(out.name + ".embedded.extracted.srt")
    sidecars = {
        ext: out.with_name(out.name + f".sidecar.original.{ext}")
        for ext in _FORMATS
    }

    try:
        if not all(_present(p) for p in (srt, ass, manifest)):
            return preserved
        meta = _manifest(manifest)
        if (type(meta.get("schema")) is not int or meta["schema"] != 1
                or meta.get("language_requested") != chosen
                or meta.get("output_timeline") != "source_video"
                or meta.get("source_video_modified") is not False
                or meta.get("ocr_applied") is not False
                or meta.get("srt_file") != srt.name
                or meta.get("ass_file") != ass.name
                or type(meta.get("cue_count")) is not int
                or meta["cue_count"] <= 0):
            return preserved
        srt_format, srt_cues = _parse(srt)
        ass_format, ass_cues = _parse(ass)
        if (srt_format != "srt" or ass_format != "ass"
                or len(srt_cues) != meta["cue_count"]):
            return preserved

        mode = meta.get("mode")
        raw_file = None
        if mode == "local-embedded-text":
            if not _present(embedded) or any(
                _present(other) for other in sidecars.values()
            ):
                return preserved
            if (meta.get("extracted_stream_file") != embedded.name
                    or meta.get("source_stream_export")
                    != "ffmpeg_decoded_srt_not_original_container_bytes"
                    or meta.get("source_codec") not in _TEXT_CODECS
                    or type(meta.get("stream_index")) is not int
                    or not 0 <= meta["stream_index"] < 128
                    or not _requested_matches(
                        chosen, meta.get("language_tag", ""),
                    )
                    or type(meta.get("is_default")) is not bool
                    or type(meta.get("is_forced")) is not bool):
                return preserved
            raw_file = embedded

        elif mode == "local-sidecar-text":
            if _present(embedded):
                return preserved
            existing_raws = [
                ext for ext, path in sidecars.items() if _present(path)
            ]
            if len(existing_raws) != 1:
                return preserved
            ext = existing_raws[0]
            raw_file = sidecars[ext]
            tag = meta.get("language_tag")
            if (tag not in _TAGS[chosen]
                    or meta.get("selection") != "adjacent"
                    or meta.get("extracted_stream_file") != raw_file.name
                    or meta.get("source_sidecar_name")
                    != f"{source.stem}.{tag}.{ext}"
                    or meta.get("original_bytes_retained") is not True
                    or meta.get("source_sidecar_modified") is not False
                    or not isinstance(meta.get("original_sha256"), str)
                    or len(meta["original_sha256"]) != 64):
                return preserved
            # The existing user's tagged sidecar must still match the
            # byte-identical raw track copied at first extraction.
            original = source.with_name(f"{source.stem}.{tag}.{ext}")
            source_bytes = _safe_read(original, _MAX_SUBTITLE_BYTES)
            raw_bytes = _safe_read(raw_file, _MAX_SUBTITLE_BYTES)
            if (raw_bytes != source_bytes
                    or hashlib.sha256(raw_bytes).hexdigest()
                    != meta["original_sha256"]):
                return preserved
        else:
            # Explicitly selected subtitles are not batch-owned originals.
            # Existing OCR SRT/ASS has no sufficiently verified local
            # source provenance; neither may be guessed complete here.
            return preserved

        raw_format, raw_cues = _parse(raw_file)
        if (mode == "local-embedded-text" and raw_format != "srt"
                or mode == "local-sidecar-text" and
                raw_format != ("ass" if ext == "ssa" else ext)
                or mode == "local-sidecar-text" and
                meta.get("original_format") != raw_format):
            return preserved
        if not _matching_cues(raw_cues, srt_cues, ass_cues):
            return preserved
        if len(raw_cues) != meta["cue_count"]:
            return preserved
        return LocalOutputAudit("verified", mode, len(srt_cues))
    except (OSError, ValueError, TypeError, UnicodeError, RecursionError,
            AttributeError, SubtitleFormatError):
        return preserved
