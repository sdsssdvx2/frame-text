"""Optional offline OCR for a *separate embedded bitmap subtitle track*.

Neither a visible hard-subtitle frame nor an ordinary text track. The free
Subtitle Edit 5.3 seconv helper is used only when explicitly supplied as a
standalone portable executable (HSE_SECONV), or when present in the future
free bundled bin/seconv.exe. No download, browser, login, user settings,
site networking, package installation, or heuristic language guessing.

All helper output is created in a separate temporary directory. Its stdout
and stderr never expose private dialogue or source names in HSE diagnostics.
The original MKV and any existing user files are always read-only.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
import subprocess
import tempfile
import time
from typing import Callable

from .local_sidecar_subtitles import _outputs
from .local_soft_subtitles import (
    _guard_outputs, _local_file, _requested_matches,
)
from .matroska_track_headers import (
    MatroskaImageTrack, read_image_track_headers,
)
from .model_assets import normalize_ocr_language
from .runtime_paths import bundle_root
from .subtitle_formats import SubtitleFormatError, parse_subtitle_bytes, subtitle_cues_have_korean
from .writers import write_subtitle_pair_exclusive, write_srt
from .local_bitmap_bdn_ocr import recognize_bdn_images
from .external_subprocess import external_popen

_MAX_CONVERTER_LOG = 128 * 1024
_MAX_SRT_BYTES = 20 * 1024 * 1024
_MAX_CUES = 10_000
_MAX_OCR_SECONDS = 3600
_MAX_SCRATCH_BYTES = 512 * 1024 * 1024
_MAX_SCRATCH_FILES = 6500
_TESSERACT_LANG = {"ko": "kor", "en": "eng", "ja": "jpn", "zh": "chi_sim"}


class LocalImageSubtitleError(RuntimeError):
    """A selected image track could not be safely OCR-converted."""


@dataclass(frozen=True, slots=True)
class LocalImageSubtitleResult:
    mode: str
    cue_count: int
    srt_path: Path
    ass_path: Path
    extracted_track_path: Path
    provenance_path: Path
    track_number: int
    track_language: str
    # Recognition coverage, not transcription accuracy. This is computed
    # from ALL source BDN events when the production PP-OCR path is used.
    missed_events: int = 0


def _regular_file(path: Path) -> bool:
    return path.is_file() and not path.is_symlink()


def _converter_executable() -> Path | None:
    configured = os.environ.get("HSE_SECONV")
    if configured:
        file = Path(configured)
        if not _regular_file(file):
            raise LocalImageSubtitleError("invalid explicitly configured image subtitle converter")
        return file.resolve(strict=True)
    # No PATH search, no execution from an arbitrary user-installed app.
    # Portable release is a .NET self-contained application: all runtime
    # and native dependencies must remain next to seconv.exe, not flattened
    # into HSE's general bin directory.
    candidate = bundle_root() / "bin" / "seconv" / "seconv.exe"
    return candidate if _regular_file(candidate) else None


def _ocr_backend(language: str) -> tuple[list[str], dict[str, str]] | None:
    """Prefer pinned local Tesseract; free Latin nOCR is EN-only fallback.

    Never allow seconv's default lookup to discover arbitrary Tesseract
    installations or a network-backed vision model.
    """
    data_dir = Path(os.environ.get(
        "HSE_TESSDATA", str(bundle_root() / "bin" / "tesseract" / "tessdata"),
    ))
    tess_bin = os.environ.get("HSE_TESSERACT")
    tess = Path(tess_bin) if tess_bin else bundle_root() / "bin" / "tesseract" / "tesseract.exe"
    lang_code = _TESSERACT_LANG[language]
    trained = data_dir / (lang_code + ".traineddata")
    if _regular_file(tess) and _regular_file(trained) and data_dir.is_dir() and not data_dir.is_symlink():
        return (
            ["--ocr-engine:tesseract", f"--ocr-language:{lang_code}"],
            {"PATH_PREFIX": str(tess.parent), "TESSDATA_PREFIX": str(data_dir)},
        )
    if language == "en":
        database = Path(os.environ.get(
            "HSE_NOCR_DB", str(bundle_root() / "bin" / "ocr" / "Latin.nocr"),
        ))
        if _regular_file(database):
            return (
                ["--ocr-engine:nocr", f"--ocr-db:{str(database.resolve(strict=True))}"],
                {},
            )
    return None


def _check_scratch_budget(scratch: Path) -> None:
    # Converted bitmap PNGs can accumulate before the helper exits.
    # Bound disk usage even for a long or malformed track; never traverse
    # symlinks or write outside the one disposable scratch folder.
    count = 0
    total = 0
    for root, dirs, files in os.walk(scratch, followlinks=False):
        for name in dirs:
            if (Path(root) / name).is_symlink():
                raise LocalImageSubtitleError("bitmap scratch symlink rejected")
        for name in files:
            path = Path(root) / name
            if path.is_symlink():
                raise LocalImageSubtitleError("bitmap scratch symlink rejected")
            stat = path.stat()
            if not path.is_file():
                raise LocalImageSubtitleError("bitmap scratch file type rejected")
            count += 1
            total += stat.st_size
            if count > _MAX_SCRATCH_FILES or total > _MAX_SCRATCH_BYTES:
                raise LocalImageSubtitleError(
                    "bitmap scratch storage budget exceeded"
                )


def _invoke(
    args: list[str], scratch: Path, env_patch: dict[str, str],
    check_cancelled: Callable[[], None] | None,
) -> bytes:
    if check_cancelled:
        check_cancelled()
    env = os.environ.copy()
    env["DOTNET_CLI_HOME"] = str(scratch)
    env["APPDATA"] = str(scratch)
    env["LOCALAPPDATA"] = str(scratch)
    env["HOME"] = str(scratch)
    env["USERPROFILE"] = str(scratch)
    env["XDG_CONFIG_HOME"] = str(scratch)
    env["XDG_DATA_HOME"] = str(scratch)
    env["NO_PROXY"] = "*"
    for key in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy",
                "https_proxy", "all_proxy"):
        env.pop(key, None)
    if "PATH_PREFIX" in env_patch:
        env["PATH"] = env_patch["PATH_PREFIX"] + os.pathsep + env.get("PATH", "")
    if "TESSDATA_PREFIX" in env_patch:
        env["TESSDATA_PREFIX"] = env_patch["TESSDATA_PREFIX"]
    with tempfile.TemporaryFile() as output:
        try:
            creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
            child = external_popen(
                args, cwd=str(scratch), stdin=subprocess.DEVNULL,
                stdout=output, stderr=subprocess.DEVNULL,
                env=env, creationflags=creationflags,
            )
            try:
                deadline = time.monotonic() + _MAX_OCR_SECONDS
                next_budget_check = time.monotonic()
                while child.poll() is None:
                    if check_cancelled:
                        check_cancelled()
                    now = time.monotonic()
                    if now >= deadline:
                        raise LocalImageSubtitleError("image subtitle OCR timed out")
                    if now >= next_budget_check:
                        _check_scratch_budget(scratch)
                        next_budget_check = now + 2.0
                    time.sleep(0.2)
                _check_scratch_budget(scratch)
                if check_cancelled:
                    check_cancelled()
                if child.returncode:
                    raise LocalImageSubtitleError("image subtitle converter returned failure")
                output.seek(0, os.SEEK_END)
                size = output.tell()
                if not 0 < size <= _MAX_CONVERTER_LOG:
                    raise LocalImageSubtitleError("invalid converter JSON size")
                output.seek(0)
                return output.read(_MAX_CONVERTER_LOG + 1)
            finally:
                if child.poll() is None:
                    child.kill()
                child.wait()
        except OSError as exc:
            raise LocalImageSubtitleError("portable image subtitle OCR executable unavailable") from exc


def _track_candidates(
    video: Path,
    language: str,
    check_cancelled: Callable[[], None] | None,
) -> list[MatroskaImageTrack]:
    # FFprobe's stream.index is unrelated to MKV TrackNumber; stream.id
    # may be absent in real media. Read the Matroska EBML Tracks element
    # directly, bounded to 16 MiB, without touching frames or payload.
    if video.suffix.lower() != ".mkv":
        return []
    if check_cancelled:
        check_cancelled()
    tracks = [
        t for t in read_image_track_headers(video)
        if _requested_matches(language, t.language)
        and t.track_number is not None
    ]
    tracks.sort(key=lambda t: (t.forced, not t.default, t.track_number))
    return tracks


def acquire_local_embedded_image(
    source: str | Path,
    out_base: str | Path,
    *,
    language: str = "ko",
    check_cancelled: Callable[[], None] | None = None,
    ocr_factory: Callable[[], object] | None = None,
    converter_factory: Callable[[], Path] | None = None,
) -> LocalImageSubtitleResult | None:
    """OCR one explicitly language-matched PGS/VobSub track, if supported.

    Return None ONLY when there is no eligible image track or no bundled/
    explicitly supplied free engine. If an eligible supported track exists
    but conversion fails, do not quietly substitute video visual OCR and
    misrepresent the source.
    """
    chosen = normalize_ocr_language(language)
    base = Path(out_base).expanduser()
    _guard_outputs(base)
    converter = _converter_executable()
    # If a pinned HSE multilingual OCR factory was supplied, prefer the
    # existing on-device PP-OCR model rather than shipping another runtime
    # or requiring Tesseract language packs for KO/JA/ZH. The factory is
    # called ONLY after an exact matching image track has been found.
    backend = None if ocr_factory is not None else (
        _ocr_backend(chosen) if converter else None
    )
    # No helper bundled => zero additional probing or inference; preserve
    # existing local-video route exactly. Legacy optional text conversion
    # still requires a verified offline seconv OCR backend.
    if (converter is None and converter_factory is None) or (ocr_factory is None and backend is None):
        return None
    video = _local_file(source)
    tracks = _track_candidates(video, chosen, check_cancelled)
    if not tracks:
        return None
    # An optional official version is fetched/verified ONLY after matching
    # PGS/VobSub track identity, output guard, and original-text priority.
    if converter_factory is not None:
        converter = Path(converter_factory())
        if not _regular_file(converter):
            raise LocalImageSubtitleError("approved official converter unavailable")

    if backend is not None:
        args_backend, env_patch = backend
    else:
        args_backend, env_patch = [], {}
    with tempfile.TemporaryDirectory(prefix="hse-image-subtitle-") as tmp:
        scratch = Path(tmp)
        # Source directory is never searched or used for output by seconv.
        # A dedicated, one-track request avoids mixing English, Japanese
        # and Chinese subtitles in the same MKV.
        for track in tracks:
            if check_cancelled:
                check_cancelled()
            file = scratch / "image-original.srt"
            if file.exists() or file.is_symlink():
                raise LocalImageSubtitleError("ambiguous image track scratch result")
            using_hse_ocr = ocr_factory is not None
            bitmap_audit: dict[str, int] | None = {} if using_hse_ocr else None
            image_dir = scratch / "image-bitmap"
            if using_hse_ocr and image_dir.exists():
                raise LocalImageSubtitleError("ambiguous source bitmap scratch folder")
            cmd = [
                str(converter), str(video),
                "bdnxml" if using_hse_ocr else "subrip",
                f"--track-number:{track.track_number}",
                *args_backend,
                f"--output-folder:{scratch}",
                "--output-filename:image-bitmap"
                if using_hse_ocr else "--output-filename:image-original.srt",
                "--json",
            ]
            if not using_hse_ocr:
                cmd.insert(-1, "--no-language-suffix")
            raw_log = _invoke(cmd, scratch, env_patch, check_cancelled)
            try:
                receipt = json.loads(raw_log.decode("utf-8-sig"))
                if not isinstance(receipt, dict) or receipt.get("success") is not True:
                    raise ValueError("converter did not confirm success")
                if using_hse_ocr:
                    if not image_dir.is_dir() or image_dir.is_symlink():
                        raise ValueError("missing isolated BDN image folder")
                    cues = recognize_bdn_images(
                        image_dir, ocr_factory(), check_cancelled=check_cancelled,
                        audit=bitmap_audit,
                    )
                    if not 0 < len(cues) <= _MAX_CUES:
                        raise ValueError("invalid multilingual image OCR cue count")
                    # Preserve normalized HSE OCR text separately from the
                    # original bitmap: do not mislabel it as original text.
                    write_srt(cues, file)
                    raw = file.read_bytes()
                else:
                    if (not _regular_file(file)
                            or not 0 < file.stat().st_size <= _MAX_SRT_BYTES):
                        raise ValueError("converter produced no valid SRT")
                    raw = file.read_bytes()
                    fmt, cues = parse_subtitle_bytes(raw, source_url="image-original.srt")
                    if fmt != "srt" or not 0 < len(cues) <= _MAX_CUES:
                        raise ValueError("image OCR SRT content invalid")
                if len(raw) > _MAX_SRT_BYTES:
                    raise ValueError("bitmap OCR transcript too large")
                if chosen == "ko" and not subtitle_cues_have_korean(cues):
                    raise ValueError("Korean image track OCR did not contain Korean text")
            except (ValueError, OSError, UnicodeError, TypeError, SubtitleFormatError) as exc:
                raise LocalImageSubtitleError("image subtitle OCR output invalid") from exc

            if check_cancelled:
                check_cancelled()
            _guard_outputs(base)
            srt = base.with_suffix(".srt")
            ass = base.with_suffix(".ass")
            raw_dest = base.with_name(base.name + ".embedded.image-ocr.srt")
            provenance = base.with_name(base.name + ".source.json")
            created: list[Path] = []
            try:
                srt, ass = write_subtitle_pair_exclusive(cues, base)
                created.extend((srt, ass))
                with raw_dest.open("xb") as output:
                    created.append(raw_dest)
                    output.write(raw)
                metadata = {
                    "schema": 1,
                    "mode": "local-embedded-image-ocr",
                    "language_requested": chosen,
                    "language_tag": track.language,
                    "source_codec": track.codec,
                    "image_track_number": track.track_number,
                    "ocr_engine": (
                        "hse_multilingual_ppocr"
                        if using_hse_ocr else
                        ("tesseract" if args_backend[0].endswith("tesseract") else "nocr")
                    ),
                    "bitmap_timing_precision": (
                        "BDN_frame_quantized" if using_hse_ocr
                        else "seconv_native_text_OCR"
                    ),
                    "ocr_applied": True,
                    "original_text_verified": False,
                    "source_video_modified": False,
                    "source_stream_export": (
                        "seconv_BDN_bitmaps_then_HSE_OCR_not_original_text"
                        if using_hse_ocr else
                        "seconv_bitmap_OCR_not_original_bitmap_bytes"
                    ),
                    "output_timeline": "source_video",
                    "cue_count": len(cues),
                    "extracted_stream_file": raw_dest.name,
                    "srt_file": srt.name,
                    "ass_file": ass.name,
                }
                if bitmap_audit is not None:
                    # Distinguish nonblank OCR event coverage from accuracy.
                    # Source image events without text are exported as
                    # incomplete, never claimed to be a full transcript.
                    source_count = bitmap_audit["source_bitmap_events"]
                    recognized_count = bitmap_audit["recognized_bitmap_events"]
                    skipped_count = bitmap_audit["unrecognized_bitmap_events"]
                    if not (
                        0 < recognized_count == len(cues) <= source_count
                        and skipped_count == source_count - recognized_count
                    ):
                        raise LocalImageSubtitleError(
                            "inconsistent bitmap source event coverage"
                        )
                    metadata["bitmap_ocr_event_coverage"] = {
                        **bitmap_audit,
                        "all_source_events_recognized": skipped_count == 0,
                        "method": "nonblank_OCR_per_source_bitmap_event_not_text_accuracy",
                    }
                with provenance.open("x", encoding="utf-8") as output:
                    created.append(provenance)
                    json.dump(metadata, output, ensure_ascii=False, indent=2)
                    output.write("\n")
            except BaseException:
                for path in reversed(created):
                    try:
                        path.unlink(missing_ok=True)
                    except OSError:
                        pass
                raise
            return LocalImageSubtitleResult(
                "local-embedded-image-ocr", len(cues), srt, ass,
                raw_dest, provenance, track.track_number,
                track.language,
                missed_events=(
                    bitmap_audit["unrecognized_bitmap_events"]
                    if bitmap_audit is not None else 0
                ),
            )
    return None
