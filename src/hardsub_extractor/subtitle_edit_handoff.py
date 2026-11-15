"""Opt-in handoff to an EXISTING free Subtitle Edit installation.

Never installs, downloads or modifies user source SRT/ASS/video. A separate
SRT is exclusively created once for proofreading. Repeated clicks open
the existing proofread copy, never silently replacing human corrections.
"""
from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile

from .subtitle_formats import decode_subtitle_bytes, parse_subtitle_bytes
from .writers import write_ass


class ProofreadHandoffError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class ProofreadHandoff:
    video: Path
    original: Path
    working_copy: Path
    executable: Path
    newly_created: bool


def find_existing_subtitle_edit(
    *,
    env: dict[str, str] | None = None,
    home: Path | None = None,
) -> Path | None:
    """Find installed/portable editor in known roots; never search all disks."""
    env = dict(os.environ if env is None else env)
    home = Path.home() if home is None else Path(home)
    candidates: list[Path] = []
    for key in ("ProgramFiles", "ProgramFiles(x86)"):
        if env.get(key):
            candidates.extend([
                Path(env[key]) / "Subtitle Edit" / "SubtitleEdit.exe",
                Path(env[key]) / "SubtitleEdit" / "SubtitleEdit.exe",
            ])
    if env.get("LOCALAPPDATA"):
        candidates.extend([
            Path(env["LOCALAPPDATA"]) / "Programs" / "Subtitle Edit"
            / "SubtitleEdit.exe",
            Path(env["LOCALAPPDATA"]) / "Programs" / "SubtitleEdit"
            / "SubtitleEdit.exe",
        ])
    # Common opt-in portable locations, checked without recursive scanning.
    candidates.extend([
        home / "Apps" / "SubtitleEdit" / "SubtitleEdit.exe",
        home / "Applications" / "SubtitleEdit" / "SubtitleEdit.exe",
    ])
    for drive in ("C:", "D:"):
        for folder in ("SubtitleEdit", "Subtitle Edit"):
            candidates.append(
                Path(drive + "/") / "Utility" / folder / "SubtitleEdit.exe"
            )
    found = shutil.which("SubtitleEdit.exe", path=env.get("PATH", ""))
    if found:
        candidates.append(Path(found))
    for candidate in candidates:
        if (candidate.name.lower() == "subtitleedit.exe"
                and candidate.is_file() and not candidate.is_symlink()):
            return candidate.resolve()
    return None


def validate_proofread_source(
    video: Path, subtitle: Path,
) -> tuple[Path, Path]:
    """Check original local media and SRT before either free editor can run."""
    video, subtitle = Path(video), Path(subtitle)
    if (
        video.is_symlink() or not video.is_file()
        or subtitle.is_symlink() or not subtitle.is_file()
        or video.suffix.lower() not in {
            ".mkv", ".mp4", ".mov", ".webm", ".avi", ".m4v", ".ts"
        }
        or subtitle.suffix.lower() != ".srt"
        or not 0 < video.stat().st_size <= 100 * 1024 ** 3
        or not 0 < subtitle.stat().st_size <= 8 * 1024 ** 2
    ):
        raise ProofreadHandoffError("invalid local proofread input")
    try:
        kind, cues = parse_subtitle_bytes(
            subtitle.read_bytes(), source_url=str(subtitle)
        )
    except (UnicodeError, ValueError) as exc:
        raise ProofreadHandoffError("invalid original subtitle") from exc
    if kind != "srt" or not 1 <= len(cues) <= 10000:
        raise ProofreadHandoffError("unsupported source subtitle contents")
    previous = -1.0
    for cue in cues:
        if not cue.text.strip() or cue.start < previous:
            raise ProofreadHandoffError("unordered or empty subtitle content")
        previous = cue.start
    return video.resolve(strict=True), subtitle.resolve(strict=True)


def validate_handoff(
    video: Path,
    subtitle: Path,
    *,
    executable: Path,
) -> tuple[Path, Path, Path]:
    """Preserve the original strict Subtitle Edit-only external contract."""
    video, subtitle = validate_proofread_source(video, subtitle)
    executable = Path(executable)
    if (
        executable.is_symlink() or not executable.is_file()
        or executable.name.lower() != "subtitleedit.exe"
    ):
        raise ProofreadHandoffError("invalid Subtitle Edit executable")
    return video, subtitle, executable.resolve(strict=True)


def find_windows_notepad() -> Path | None:
    """Read the actual Windows System32 directory, not PATH or a user override."""
    if os.name != "nt":
        return None
    import ctypes
    buf = ctypes.create_unicode_buffer(32768)
    length = ctypes.windll.kernel32.GetSystemDirectoryW(buf, len(buf))
    if not 0 < length < len(buf):
        return None
    candidate = Path(buf.value) / "notepad.exe"
    if (
        candidate.name.lower() != "notepad.exe"
        or candidate.is_symlink() or not candidate.is_file()
    ):
        return None
    return candidate.resolve(strict=True)


def _prepare_proofread_copy(video: Path, original: Path) -> tuple[Path, bool]:
    """Exclusive first copy and never overwrite a user's prior text corrections."""
    target = original.with_name(original.stem + "-proofread.srt")
    if target.is_symlink() or target.is_dir():
        raise ProofreadHandoffError("review copy path unsafe")
    created = False
    if not target.exists():
        try:
            with original.open("rb") as inp, target.open("xb") as out:
                created = True
                shutil.copyfileobj(inp, out, length=128 * 1024)
                out.flush()
                os.fsync(out.fileno())
        except BaseException:
            if created:
                target.unlink(missing_ok=True)
            raise
    else:
        stat = target.stat()
        if (
            not 0 < stat.st_size <= 8 * 1024 ** 2
            or stat.st_nlink != 1
            or target.samefile(original)
            or target.samefile(video)
        ):
            raise ProofreadHandoffError("existing review copy unsafe")
    return target, created


def launch_notepad_proofreading(
    video: Path, subtitle: Path, *,
    executable: Path | None = None,
    opener=None,
) -> ProofreadHandoff:
    """Explicit owner-approved, no-install fallback when Subtitle Edit is absent.

    Notepad edits plain SRT without video-sync preview. Only the separate
    -proofread.srt is opened; no original media, SRT or ASS is modified.
    """
    video, original = validate_proofread_source(video, subtitle)
    program = find_windows_notepad() if executable is None else Path(executable)
    if (
        program is None or program.is_symlink() or not program.is_file()
        or program.name.lower() != "notepad.exe"
    ):
        raise ProofreadHandoffError("trusted Windows Notepad not found")
    program = program.resolve(strict=True)
    working, created = _prepare_proofread_copy(video, original)
    opener = subprocess.Popen if opener is None else opener
    try:
        opener([str(program), str(working)], cwd=str(program.parent), shell=False)
    except BaseException:
        if created:
            working.unlink(missing_ok=True)
        raise
    return ProofreadHandoff(
        video=video, original=original, working_copy=working,
        executable=program, newly_created=created,
    )


def launch_subtitle_edit_proofreading(
    video: Path,
    subtitle: Path,
    *,
    executable: Path,
    opener=None,
) -> ProofreadHandoff:
    """Launch only from direct user action; never shell-expand any media path."""
    video, original, executable = validate_handoff(
        video, subtitle, executable=executable
    )
    target, created = _prepare_proofread_copy(video, original)
    # Do not create/replace an .ass counterpart: the user edits SRT in the
    # external editor and may explicitly export an ASS from there.
    opener = subprocess.Popen if opener is None else opener
    try:
        opener(
            [str(executable), str(target), "/video:" + str(video)],
            cwd=str(executable.parent),
            shell=False,
        )
    except BaseException:
        if created:
            target.unlink(missing_ok=True)
        raise
    return ProofreadHandoff(
        video=video, original=original, working_copy=target,
        executable=executable, newly_created=created,
    )


@dataclass(frozen=True, slots=True)
class ProofreadASSExport:
    working_copy: Path
    ass_path: Path
    cue_count: int
    already_present: bool


_SRT_TIMESTAMP = (
    r"\d{1,2}:\d{2}:\d{2}[,.]\d{1,3}"
    r"\s*-->\s*\d{1,2}:\d{2}:\d{2}[,.]\d{1,3}"
)


def _read_checked_proofread_srt(path: Path):
    """Strict all-block check: never silently drop a malformed edited cue."""
    stat_before = path.stat()
    if not 0 < stat_before.st_size <= 8 * 1024 * 1024:
        raise ProofreadHandoffError("invalid proofread SRT size")
    raw = path.read_bytes()
    if len(raw) != stat_before.st_size:
        raise ProofreadHandoffError("proofread SRT changed while reading")
    try:
        text = decode_subtitle_bytes(raw).replace("\r\n", "\n").replace("\r", "\n").strip()
        fmt, cues = parse_subtitle_bytes(raw, source_url=str(path))
    except (UnicodeError, ValueError) as exc:
        raise ProofreadHandoffError("invalid proofread SRT") from exc
    if fmt != "srt" or not 1 <= len(cues) <= 10000:
        raise ProofreadHandoffError("unsupported proofread SRT")
    blocks = re.split(r"\n[ \t]*\n+", text)
    if len(blocks) != len(cues):
        raise ProofreadHandoffError("malformed SRT blocks cannot be skipped")
    for block in blocks:
        lines = block.splitlines()
        if (
            len(lines) < 3 or not lines[0].strip().isdigit()
            or re.fullmatch(_SRT_TIMESTAMP, lines[1].strip()) is None
            or not any(line.strip() for line in lines[2:])
        ):
            raise ProofreadHandoffError("malformed edited SRT block")
    previous_start = -1.0
    for cue in cues:
        if cue.start < previous_start:
            raise ProofreadHandoffError("edited SRT is not chronological")
        previous_start = cue.start
    return raw, cues, stat_before


def export_proofread_ass(original_srt: Path) -> ProofreadASSExport:
    """Export an idempotent, content-pinned ASS beside the corrected SRT.

    Does not invoke external programs and cannot overwrite an original
    SRT/ASS, human edits or any already published ASS content. A changed
    corrected SRT creates a *different* ASS; an identical draft reuses
    the exact previously published bytes.
    """
    original = Path(original_srt).expanduser()
    if (
        original.suffix.lower() != ".srt" or original.is_symlink()
        or not original.is_file()
    ):
        raise ProofreadHandoffError("original HSE SRT missing or unsafe")
    working = original.with_name(original.stem + "-proofread.srt")
    if (
        working.is_symlink() or not working.is_file()
        or working.stat().st_nlink != 1 or working.samefile(original)
    ):
        raise ProofreadHandoffError("corrected SRT missing or linked")
    raw, cues, before = _read_checked_proofread_srt(working)
    digest = sha256(raw).hexdigest()[:12]
    destination = working.with_name(working.stem + "-" + digest + ".ass")
    # Only this invocation's temporary staging file is ever removed.
    with tempfile.TemporaryDirectory(
        prefix=".hse-proofread-export-", dir=working.parent
    ) as scratch:
        staged = Path(scratch) / "draft.ass"
        write_ass(cues, staged)
        # ASS must preserve every displayed cue and physical line count.
        # If legacy ASS syntax would alter any user-corrected text, refuse
        # rather than silently export a misleading subtitle.
        try:
            fmt, roundtrip = parse_subtitle_bytes(
                staged.read_bytes(), source_url="proofread.ass"
            )
        except (UnicodeError, ValueError) as exc:
            raise ProofreadHandoffError("generated ASS cannot be parsed") from exc
        if (
            fmt != "ass" or len(roundtrip) != len(cues)
            or any(
                x.text != y.text
                or abs(x.start - y.start) > .011
                or abs(x.end - y.end) > .011
                for x, y in zip(cues, roundtrip)
            )
        ):
            raise ProofreadHandoffError("ASS export would change corrected text")
        # Protect against a replaced or concurrently saved edited SRT.
        after = working.stat()
        if (
            after.st_ino != before.st_ino
            or after.st_size != before.st_size
            or after.st_mtime_ns != before.st_mtime_ns
            or after.st_nlink != 1
        ):
            raise ProofreadHandoffError("corrected SRT changed during export")
        if destination.is_symlink():
            raise ProofreadHandoffError("ASS destination is a symlink")
        if destination.exists():
            if (
                not destination.is_file()
                or destination.stat().st_nlink != 1
                or destination.read_bytes() != staged.read_bytes()
            ):
                raise ProofreadHandoffError("previous ASS output conflicts")
            return ProofreadASSExport(
                working, destination, len(cues), already_present=True
            )
        # Same directory => same volume, and NTFS hardlink publication is
        # an atomic CREATE-EXCLUSIVE operation with no partially visible ASS.
        try:
            os.link(staged, destination)
        except FileExistsError as exc:
            raise ProofreadHandoffError("ASS output appeared concurrently") from exc
    return ProofreadASSExport(
        working, destination, len(cues), already_present=False
    )
