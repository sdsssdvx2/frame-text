"""Opt-in, no-purchase, side-by-side official yt-dlp updater for HSE.

Only a user-owned cache is changed. The PyInstaller bundled executable is
never edited. Upstream release metadata and asset SHA-256 must agree before
an executable is launched for offline compatibility checks. No public video,
browser session, API key or credential is used in the acceptance phase.

This is NOT a signature verification or general-purpose software updater:
GitHub repository/release integrity and HTTPS remain trust assumptions.
"""
from __future__ import annotations

from datetime import date
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import urllib.parse
import urllib.request
from uuid import uuid4

from .external_subprocess import external_popen
from .runtime_paths import bundle_root

_API = "https://api.github.com/repos/yt-dlp/yt-dlp/releases/latest"
_PROJECT = "yt-dlp/yt-dlp"
_RELEASE_RE = re.compile(r"20[0-9]{2}\.(?:0?[1-9]|1[0-2])\.(?:0?[1-9]|[12][0-9]|3[01])$")
_SHA_RE = re.compile(r"[0-9a-f]{64}$")
_MIN_BYTES = 1024 * 1024
_MAX_BYTES = 85 * 1024 * 1024
_MAX_JSON = 256 * 1024
_CURRENT_BUNDLED = "2026.10.26"
_REDIRECT_HOSTS = frozenset((
    "github.com", "release-assets.githubusercontent.com",
    "objects.githubusercontent.com",
))


class ManagedDownloaderError(RuntimeError):
    pass


class _ReleaseRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        parsed = urllib.parse.urlparse(newurl)
        if (parsed.scheme != "https" or parsed.username or parsed.password
                or (parsed.hostname or "").lower() not in _REDIRECT_HOSTS):
            raise ManagedDownloaderError("unsafe official release redirect")
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def _opener():
    # Ignore user/system HTTP_PROXY when checking a public official release.
    return urllib.request.build_opener(
        urllib.request.ProxyHandler({}), _ReleaseRedirect(),
    )


def _linked(path: Path) -> bool:
    return path.is_symlink() or bool(getattr(path, "is_junction", lambda: False)())


def _sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(256 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _root() -> Path:
    if os.name == "nt":
        base = os.environ.get("LOCALAPPDATA")
        if not base:
            raise ManagedDownloaderError("Windows private app data unavailable")
        return Path(base) / "HardSubtitleExtractor" / "managed-tools" / "yt-dlp"
    return (
        Path(os.environ["XDG_CACHE_HOME"]) if os.environ.get("XDG_CACHE_HOME")
        else Path.home() / ".cache"
    ) / "HardSubtitleExtractor" / "managed-tools" / "yt-dlp"


def _read_settings(root: Path) -> dict:
    file = root / "settings.json"
    if _linked(root) or _linked(file):
        raise ManagedDownloaderError("linked updater state not accepted")
    if not file.is_file():
        return {"auto": False}
    if file.stat().st_size > 4096:
        raise ManagedDownloaderError("updater state size invalid")
    obj = json.loads(file.read_text(encoding="utf-8"))
    if not isinstance(obj, dict) or type(obj.get("auto")) is not bool:
        raise ManagedDownloaderError("updater state format invalid")
    return obj


def _write_settings(root: Path, obj: dict) -> None:
    if _linked(root):
        raise ManagedDownloaderError("linked updater directory not accepted")
    root.mkdir(parents=True, exist_ok=True)
    target = root / "settings.json"
    if _linked(target):
        raise ManagedDownloaderError("linked updater state not accepted")
    temp = root / (".settings-" + uuid4().hex + ".tmp")
    try:
        with temp.open("x", encoding="utf-8") as file:
            json.dump(obj, file, sort_keys=True, separators=(",", ":"))
            file.flush()
            os.fsync(file.fileno())
        os.replace(temp, target)
    finally:
        temp.unlink(missing_ok=True)


def auto_update_enabled(*, cache_root: Path | None = None) -> bool:
    try:
        return _read_settings(Path(cache_root) if cache_root else _root())["auto"] is True
    except (OSError, ValueError, TypeError, ManagedDownloaderError):
        return False


def set_auto_update_enabled(enabled: bool, *, cache_root: Path | None = None) -> None:
    if type(enabled) is not bool:
        raise TypeError("update permission must be boolean")
    root = Path(cache_root) if cache_root else _root()
    obj = _read_settings(root)
    obj["auto"] = enabled
    _write_settings(root, obj)


def _release_meta(opener) -> tuple[str, int, str, str]:
    req = urllib.request.Request(_API, headers={
        "Accept": "application/vnd.github+json",
        "User-Agent": "HardSubtitleExtractor-Free-EngineUpdater/1",
    })
    with opener.open(req, timeout=20) as response:
        if response.status != 200 or response.geturl() != _API:
            raise ManagedDownloaderError("unexpected official release metadata")
        if int(response.headers.get("Content-Length", "0") or 0) > _MAX_JSON:
            raise ManagedDownloaderError("release metadata oversized")
        raw = response.read(_MAX_JSON + 1)
    if len(raw) > _MAX_JSON:
        raise ManagedDownloaderError("release metadata oversized")
    obj = json.loads(raw)
    if not isinstance(obj, dict) or obj.get("draft") is not False or obj.get("prerelease") is not False:
        raise ManagedDownloaderError("not a stable official release")
    version = obj.get("tag_name")
    if type(version) is not str or not _RELEASE_RE.fullmatch(version):
        raise ManagedDownloaderError("unsupported official release version")
    assets = obj.get("assets")
    if not isinstance(assets, list):
        raise ManagedDownloaderError("official asset listing missing")
    matches = [a for a in assets if isinstance(a, dict) and a.get("name") == "yt-dlp.exe"]
    if len(matches) != 1:
        raise ManagedDownloaderError("ambiguous official Windows asset")
    asset = matches[0]
    raw_digest, size = asset.get("digest"), asset.get("size")
    url = asset.get("browser_download_url")
    expected_url = (
        "https://github.com/yt-dlp/yt-dlp/releases/download/"
        + version + "/yt-dlp.exe"
    )
    if (asset.get("state") != "uploaded" or type(size) is not int
            or not _MIN_BYTES <= size <= _MAX_BYTES
            or type(raw_digest) is not str
            or not raw_digest.startswith("sha256:")
            or not _SHA_RE.fullmatch(raw_digest[7:])
            or url != expected_url):
        raise ManagedDownloaderError("official Windows asset has no usable release digest")
    return version, size, raw_digest[7:], url


def _version_key(version: str) -> tuple[int, int, int]:
    if not _RELEASE_RE.fullmatch(version):
        raise ManagedDownloaderError("invalid official version")
    return tuple(int(x) for x in version.split("."))


def _verified_active(root: Path, state: dict) -> Path | None:
    if state.get("auto") is not True:
        return None
    active = state.get("active")
    if not isinstance(active, dict):
        return None
    version, digest = active.get("version"), active.get("sha256")
    if (type(version) is not str or not _RELEASE_RE.fullmatch(version)
            or type(digest) is not str or not _SHA_RE.fullmatch(digest)):
        return None
    versions = root / "versions"
    folder = versions / version
    executable = folder / "yt-dlp.exe"
    if any(_linked(p) for p in (root, versions, folder, executable)):
        return None
    if not executable.is_file() or not _MIN_BYTES <= executable.stat().st_size <= _MAX_BYTES:
        return None
    if _sha(executable) != digest:
        return None
    return executable


def active_downloader(*, cache_root: Path | None = None) -> str | None:
    """Find an explicitly enabled, byte-checked managed tool; never download."""
    try:
        root = Path(cache_root) if cache_root else _root()
        path = _verified_active(root, _read_settings(root))
        return str(path) if path else None
    except (OSError, ValueError, ManagedDownloaderError):
        return None


def _download_asset(opener, url: str, destination: Path, size: int, digest: str) -> None:
    req = urllib.request.Request(url, headers={"User-Agent": "HardSubtitleExtractor-Free-EngineUpdater/1"})
    h = hashlib.sha256()
    total = 0
    with opener.open(req, timeout=75) as response:
        parsed = urllib.parse.urlparse(response.geturl())
        if (response.status != 200 or parsed.scheme != "https"
                or (parsed.hostname or "").lower() not in _REDIRECT_HOSTS):
            raise ManagedDownloaderError("unexpected executable download origin")
        with destination.open("xb") as file:
            while True:
                chunk = response.read(256 * 1024)
                if not chunk:
                    break
                total += len(chunk)
                if total > size:
                    raise ManagedDownloaderError("official executable size exceeded")
                h.update(chunk)
                file.write(chunk)
    if total != size or h.hexdigest() != digest:
        raise ManagedDownloaderError("official executable checksum did not match release")


def _smoke(executable: Path, version: str) -> None:
    # Executables are deliberately never invoked before official SHA comparison.
    # Restrict subprocess environment to non-secret Windows runtime essentials.
    allowed = (
        "SYSTEMROOT", "WINDIR", "PATH", "PATHEXT", "TEMP", "TMP",
        "USERPROFILE", "HOMEDRIVE", "HOMEPATH", "APPDATA", "LOCALAPPDATA",
    )
    env = {k: os.environ[k] for k in allowed if k in os.environ}
    child = external_popen(
        [str(executable), "--ignore-config", "--no-plugin-dirs", "--version"],
        stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL, env=env,
    )
    try:
        stdout, _ = child.communicate(timeout=30)
    except subprocess.TimeoutExpired:
        child.kill()
        child.communicate()
        raise ManagedDownloaderError("official downloader startup timed out")
    if child.returncode != 0 or len(stdout) > 256 or stdout.decode("ascii", "replace").strip() != version:
        raise ManagedDownloaderError("updated downloader version self-check failed")
    child = external_popen(
        [str(executable), "--ignore-config", "--no-plugin-dirs", "--list-extractors"],
        stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL, env=env,
    )
    try:
        child.wait(timeout=40)
    except subprocess.TimeoutExpired:
        child.kill()
        child.wait()
        raise ManagedDownloaderError("official downloader extractor smoke timed out")
    if child.returncode:
        raise ManagedDownloaderError("official downloader extractor smoke failed")


def _publish_folder(stage: Path, target: Path) -> None:
    # Reuse the existing audited no-replace directory publication from
    # the separately verified optional native-tool cache.
    from .native_vsf_assets import _publish_exclusive
    _publish_exclusive(stage, target)


def refresh(*, auto: bool = False, cache_root: Path | None = None, opener=None) -> str:
    """Stage + smoke official stable Windows yt-dlp; promote only after checks.

    First call requires explicit persisted consent. Once opted in, GUI launches
    at most one check per local calendar day; manual calls can force a check.
    Failure leaves both the bundled and any previously selected release alone.
    """
    root = Path(cache_root) if cache_root else _root()
    state = _read_settings(root)
    if state.get("auto") is not True:
        return "disabled"
    today = date.today().isoformat()
    if auto and state.get("last_attempt_day") == today:
        return "not_due"
    # Save cadence before a possibly long network request, to prevent
    # repeated background fetches after a temporary offline failure.
    state["last_attempt_day"] = today
    _write_settings(root, state)
    opener = opener if opener is not None else _opener()
    version, size, digest, url = _release_meta(opener)
    current = state.get("active")
    if isinstance(current, dict) and _verified_active(root, state) is not None:
        known_version = current["version"]
    else:
        known_version = _CURRENT_BUNDLED
    if _version_key(version) <= _version_key(known_version):
        return "current"

    versions = root / "versions"
    if _linked(versions):
        raise ManagedDownloaderError("linked version cache")
    versions.mkdir(parents=True, exist_ok=True)
    target = versions / version
    candidate = target / "yt-dlp.exe"
    if target.exists() or _linked(target):
        if (_linked(target) or _linked(candidate) or not candidate.is_file()
                or candidate.stat().st_size != size or _sha(candidate) != digest):
            raise ManagedDownloaderError("existing version cache differs from official release")
        _smoke(candidate, version)
    else:
        stage = versions / (".new-" + uuid4().hex)
        stage.mkdir()
        try:
            downloaded = stage / "yt-dlp.exe"
            _download_asset(opener, url, downloaded, size, digest)
            with downloaded.open("rb") as stream:
                if stream.read(2) != b"MZ":
                    raise ManagedDownloaderError("official Windows executable header invalid")
            if _sha(downloaded) != digest:
                raise ManagedDownloaderError("staged downloader changed before execution")
            _smoke(downloaded, version)
            if target.exists() or _linked(target):
                raise ManagedDownloaderError("concurrent version target appeared")
            _publish_folder(stage, target)
        finally:
            if stage.exists():
                shutil.rmtree(stage, ignore_errors=True)

    # Only the active selection changes. The previous complete version is
    # kept for fallback/rollback; the bundled baseline remains untouched.
    latest_state = _read_settings(root)
    if latest_state.get("auto") is not True:
        return "disabled"
    latest_state["active"] = {"version": version, "sha256": digest}
    latest_state["last_attempt_day"] = today
    _write_settings(root, latest_state)
    return "updated"
