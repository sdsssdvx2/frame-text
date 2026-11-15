"""Explicitly approved, SHA-pinned official free VideoSubFinder CLI cache.

Never runs during default HSE extraction or package import. First use requires
an explicit user action; no automatic download, system installation, service,
PATH change, browser session, user video access or overwrite of existing data.

The GPL-2.0 third-party executable remains a separate user-fetched optional
tool. It is NOT added to HSE's build and is not permission to redistribute
its binaries without third-party source/license compliance review.
"""
from __future__ import annotations

import hashlib
import os
from pathlib import Path, PurePosixPath
import shutil
import stat
import time
from typing import Callable
import urllib.error
import urllib.parse
import urllib.request
import uuid
import zipfile

OFFICIAL_URL = (
    "https://github.com/eritpchy/videosubfinder-cli/releases/download/"
    "v6.10.5/videosubfinder-cli-windows-x64.zip"
)
OFFICIAL_ARCHIVE_SHA256 = (
    "60c17f0fa7dde94a51adb468898a7ed4ac33b250fd68af3d736b19989942cb35"
)
# Separate, independently pinned upstream stable versions. Never derive a
# downloaded binary or hash from an untrusted runtime selector or moving URL.
# Old explicit opt-in defaults continue to use the original v6.10.5 bytes.
PINNED_OFFICIAL_VERSIONS = {
    "v6.10.5": OFFICIAL_ARCHIVE_SHA256,
    "v6.10.7": "a8036cada008fa24c88fa4d0b45aaf2eaad584d8c12cfa8b60d7a27496e372ba",
}
OFFICIAL_DEFAULT_VERSION = "v6.10.5"
OFFICIAL_CANDIDATE_VERSION = "v6.10.7"


def _release_details(version: str) -> tuple[str, str]:
    # No arbitrary version tags, URLs, local archive names, or unattended
    # auto-upgrades. Every candidate has a public official upstream SHA pin.
    if type(version) is not str or version not in PINNED_OFFICIAL_VERSIONS:
        raise OfficialVsfAssetError("unsupported unapproved official VSF release")
    digest = (OFFICIAL_ARCHIVE_SHA256 if version == OFFICIAL_DEFAULT_VERSION
              else PINNED_OFFICIAL_VERSIONS[version])
    url = ("https://github.com/eritpchy/videosubfinder-cli/releases/download/"
           + version + "/videosubfinder-cli-windows-x64.zip")
    return url, digest
_MAX_DOWNLOAD_BYTES = 16 * 1024 * 1024
_MAX_EXPANDED_BYTES = 96 * 1024 * 1024
_MAX_MEMBERS = 120
_ALLOWED_REDIRECT_HOSTS = frozenset({
    "github.com", "release-assets.githubusercontent.com",
    "objects.githubusercontent.com",
})


class OfficialVsfAssetError(RuntimeError):
    """Official executable unavailable, untrusted or not safely cacheable."""


class _SafeReleaseRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, request, fp, code, msg, headers, newurl):
        parsed = urllib.parse.urlparse(newurl)
        if (parsed.scheme.lower() != "https"
                or (parsed.hostname or "").lower() not in _ALLOWED_REDIRECT_HOSTS
                or parsed.username or parsed.password):
            raise OfficialVsfAssetError("official HTTPS release redirect rejected")
        return super().redirect_request(request, fp, code, msg, headers, newurl)


def _sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _is_link_or_junction(path: Path) -> bool:
    return path.is_symlink() or bool(
        getattr(path, "is_junction", lambda: False)()
    )


def _safe_member(name: str) -> PurePosixPath:
    if (
        not name or len(name) > 220
        or "\\" in name or ":" in name or "\0" in name
    ):
        raise OfficialVsfAssetError("unsafe release ZIP path")
    p = PurePosixPath(name)
    if (
        p.is_absolute()
        or any(part in {"", ".", ".."} for part in name.rstrip("/").split("/"))
        or any(part.endswith((" ", ".")) for part in p.parts)
        or any(part.rstrip(" .").upper().split(".")[0] in {
            "CON", "PRN", "NUL", "AUX", *("COM" + str(i) for i in range(1, 10)),
            *("LPT" + str(i) for i in range(1, 10)),
        } for part in p.parts)
    ):
        raise OfficialVsfAssetError("release ZIP path would escape private cache")
    return p


def _archive_members(zipfile_handle: zipfile.ZipFile):
    files = []
    seen = set()
    exe = None
    total = 0
    members = zipfile_handle.infolist()
    if not 1 <= len(members) <= _MAX_MEMBERS:
        raise OfficialVsfAssetError("unexpected native release archive entries")
    for info in members:
        p = _safe_member(info.filename)
        canonical = tuple(x.casefold() for x in p.parts)
        if canonical in seen:
            raise OfficialVsfAssetError("case-folding file collision in release")
        seen.add(canonical)
        if (info.external_attr >> 16) & 0o170000 == stat.S_IFLNK:
            raise OfficialVsfAssetError("links forbidden in official native archive")
        if info.file_size < 0 or info.file_size > _MAX_EXPANDED_BYTES:
            raise OfficialVsfAssetError("oversized native release archive member")
        total += info.file_size
        if total > _MAX_EXPANDED_BYTES:
            raise OfficialVsfAssetError("native archive extraction budget exceeded")
        if not info.is_dir():
            files.append((p, info))
            if p.name.lower() == "videosubfindercli.exe":
                if exe is not None:
                    raise OfficialVsfAssetError("ambiguous official native executable")
                exe = p
    if exe is None:
        raise OfficialVsfAssetError("official VideoSubFinder executable missing")
    return files, exe


def _verified_archive(path: Path, *, expected_sha256: str | None = None) -> None:
    if (
        _is_link_or_junction(path) or not path.is_file()
        or not 0 < path.stat().st_size <= _MAX_DOWNLOAD_BYTES
        or _sha(path) != (expected_sha256 if expected_sha256 is not None else OFFICIAL_ARCHIVE_SHA256)
    ):
        raise OfficialVsfAssetError("official archive SHA-256 mismatch")


def download_verified_archive(
    dest: Path, check_cancelled: Callable[[], None] | None,
    *, url: str | None = None, expected_sha256: str | None = None,
) -> None:
    """HTTPS official release only, bounded streaming, no response/log exposure."""
    if dest.exists() or dest.is_symlink():
        raise OfficialVsfAssetError("release destination must be empty")
    request = urllib.request.Request(
        url if url is not None else OFFICIAL_URL,
        headers={"User-Agent": "HSE-optional-free-VSF/version-pinned"},
    )
    opener = urllib.request.build_opener(_SafeReleaseRedirect())
    count = 0
    h = hashlib.sha256()
    deadline = time.monotonic() + 240.0
    try:
        with opener.open(request, timeout=75) as response, dest.open("xb") as output:
            resolved = urllib.parse.urlparse(response.geturl())
            if (
                resolved.scheme.lower() != "https"
                or (resolved.hostname or "").lower() not in _ALLOWED_REDIRECT_HOSTS
            ):
                raise OfficialVsfAssetError("unsupported official release origin")
            while True:
                if check_cancelled:
                    check_cancelled()
                if time.monotonic() >= deadline:
                    raise OfficialVsfAssetError("official release acquisition deadline exceeded")
                chunk = response.read(256 * 1024)
                if not chunk:
                    break
                count += len(chunk)
                if count > _MAX_DOWNLOAD_BYTES:
                    raise OfficialVsfAssetError("official download exceeds size ceiling")
                h.update(chunk)
                output.write(chunk)
        trusted = expected_sha256 if expected_sha256 is not None else OFFICIAL_ARCHIVE_SHA256
        if count == 0 or h.hexdigest() != trusted:
            raise OfficialVsfAssetError("downloaded official release failed SHA-256 pin")
    except BaseException:
        dest.unlink(missing_ok=True)
        raise


def _check_member_bytes(
    expected_stream, actual_path: Path, size: int,
    check_cancelled: Callable[[], None] | None,
) -> None:
    if _is_link_or_junction(actual_path) or not actual_path.is_file():
        raise OfficialVsfAssetError("missing/linked cached official native file")
    info = actual_path.stat()
    if info.st_size != size or info.st_nlink > 1:
        raise OfficialVsfAssetError("native release member size/link changed")
    with actual_path.open("rb") as read:
        while True:
            if check_cancelled:
                check_cancelled()
            a = expected_stream.read(256 * 1024)
            b = read.read(256 * 1024)
            if a != b:
                raise OfficialVsfAssetError("cached native DLL or exe differs from official release")
            if not a:
                break


def _verify_cached(
    folder: Path, check_cancelled: Callable[[], None] | None,
    *, expected_sha256: str | None = None,
) -> tuple[Path, str]:
    archive = folder / "official.zip"
    _verified_archive(archive, expected_sha256=expected_sha256)
    native = folder / "native"
    if _is_link_or_junction(native) or not native.is_dir():
        raise OfficialVsfAssetError("verified release cache directory missing")
    try:
        with zipfile.ZipFile(archive) as zf:
            files, executable = _archive_members(zf)
            expected_files = {tuple(part.casefold() for part in p.parts)
                              for p, _ in files}
            observed_files = set()
            for path in native.rglob("*"):
                if _is_link_or_junction(path):
                    raise OfficialVsfAssetError("unexpected linked native file")
                if not path.is_file():
                    continue
                rel = path.relative_to(native)
                canonical = tuple(part.casefold() for part in rel.parts)
                if canonical in observed_files:
                    raise OfficialVsfAssetError("native cache case collision")
                observed_files.add(canonical)
            if observed_files != expected_files:
                raise OfficialVsfAssetError("native cache contents differ from official release")
            for p, info in files:
                with zf.open(info) as zstream:
                    _check_member_bytes(
                        zstream, native.joinpath(*p.parts),
                        info.file_size, check_cancelled,
                    )
            exe = native.joinpath(*executable.parts)
            return exe, _sha(exe)
    except (zipfile.BadZipFile, OSError, EOFError, RuntimeError) as exc:
        if isinstance(exc, OfficialVsfAssetError):
            raise
        raise OfficialVsfAssetError("official native cache validation failed") from exc


def _unpack(
    archive: Path, native: Path,
    check_cancelled: Callable[[], None] | None,
    *, expected_sha256: str | None = None,
) -> None:
    _verified_archive(archive, expected_sha256=expected_sha256)
    if native.exists() or _is_link_or_junction(native):
        raise OfficialVsfAssetError("native unpack destination is occupied")
    native.mkdir(mode=0o700)
    try:
        with zipfile.ZipFile(archive) as zf:
            files, _ = _archive_members(zf)
            for rel, info in files:
                if check_cancelled:
                    check_cancelled()
                dest = native.joinpath(*rel.parts)
                dest.parent.mkdir(parents=True, exist_ok=True)
                with zf.open(info) as source, dest.open("xb") as output:
                    length = 0
                    while data := source.read(256 * 1024):
                        if check_cancelled:
                            check_cancelled()
                        length += len(data)
                        if length > info.file_size:
                            raise OfficialVsfAssetError("native member decompression overrun")
                        output.write(data)
                    if length != info.file_size:
                        raise OfficialVsfAssetError("native member decompression truncated")
    except BaseException:
        shutil.rmtree(native, ignore_errors=True)
        raise


def _publish_exclusive(stage: Path, target: Path) -> None:
    """Atomically publish a new directory without replacing concurrent data.

    Windows rename already rejects existing destinations. On Linux, plain
    rename replaces empty directories; require renameat2 NOREPLACE instead.
    Other POSIX platforms fail closed until an audited API is available.
    """
    if os.name == "nt":
        stage.rename(target)
        return
    if os.name != "posix":
        raise OfficialVsfAssetError("atomic native cache publish unsupported")
    import ctypes
    import errno
    libc = ctypes.CDLL(None, use_errno=True)
    renamer = getattr(libc, "renameat2", None)
    if renamer is None:
        raise OfficialVsfAssetError("non-replacing directory publish unavailable")
    renamer.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int,
                        ctypes.c_char_p, ctypes.c_uint]
    renamer.restype = ctypes.c_int
    if renamer(-100, os.fsencode(stage), -100, os.fsencode(target), 1) != 0:
        if ctypes.get_errno() == errno.EEXIST:
            raise OfficialVsfAssetError("concurrent native cache already exists")
        raise OfficialVsfAssetError("native cache atomic publication failed")


def default_cache_base() -> Path:
    """Private per-user data directory; never touch program files or HSE EXE."""
    if os.name == "nt":
        base = os.environ.get("LOCALAPPDATA")
        if not base:
            raise OfficialVsfAssetError("private Windows LocalAppData unavailable")
        return Path(base) / "HardSubtitleExtractor" / "optional-native-tools"
    return (
        Path(os.environ["XDG_CACHE_HOME"])
        if os.environ.get("XDG_CACHE_HOME") else
        Path.home() / ".cache"
    ) / "HardSubtitleExtractor" / "optional-native-tools"


def ensure_official_vsf(
    *,
    consent: bool,
    cache_base: Path | None = None,
    check_cancelled: Callable[[], None] | None = None,
    release_version: str = OFFICIAL_DEFAULT_VERSION,
) -> tuple[Path, str]:
    """Return separately pinned executable and digest only after explicit opt-in.

    An existing pinned cache is verified byte-for-byte against the *official*
    ZIP at each use, including adjacent DLLs. An incomplete or corrupted
    directory is NEVER deleted, replaced, launched or silently repaired.
    """
    if consent is not True:
        raise PermissionError("official native download requires explicit opt-in")
    trusted_url, trusted_sha = _release_details(release_version)
    candidate_release = release_version != OFFICIAL_DEFAULT_VERSION
    verification_kwargs = ({"expected_sha256": trusted_sha} if candidate_release else {})
    root = Path(cache_base) if cache_base is not None else default_cache_base()
    if _is_link_or_junction(root):
        raise OfficialVsfAssetError("native cache root must not be linked")
    if check_cancelled:
        check_cancelled()
    target = root / ("VideoSubFinderCLI-" + release_version)
    if target.exists() or _is_link_or_junction(target):
        if _is_link_or_junction(target) or not target.is_dir():
            raise OfficialVsfAssetError("existing native cache is not a directory")
        return _verify_cached(target, check_cancelled, **verification_kwargs)
    root.mkdir(parents=True, exist_ok=True, mode=0o700)
    stage = root / (".vsf-new-" + uuid.uuid4().hex)
    stage.mkdir(mode=0o700)
    try:
        archive = stage / "official.zip"
        if candidate_release:
            download_verified_archive(
                archive, check_cancelled,
                url=trusted_url, expected_sha256=trusted_sha,
            )
        else:
            download_verified_archive(archive, check_cancelled)
        _unpack(archive, stage / "native", check_cancelled, **verification_kwargs)
        result = _verify_cached(stage, check_cancelled, **verification_kwargs)
        if check_cancelled:
            check_cancelled()
        if target.exists() or _is_link_or_junction(target):
            # Concurrent preparation is never allowed to replace a folder.
            return _verify_cached(target, check_cancelled, **verification_kwargs)
        _publish_exclusive(stage, target)
        return target / "native" / result[0].relative_to(stage / "native"), result[1]
    finally:
        if stage.exists():
            shutil.rmtree(stage, ignore_errors=True)
