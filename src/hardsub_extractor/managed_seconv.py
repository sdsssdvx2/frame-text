"""Opt-in SHA-pinned official SeConv 5.4 portable image-track helper.

The bundled, verified 5.3.0 seconv stays the default. Only after an eligible
PGS/VobSub track has been selected, an explicit GUI/CLI opt-in may obtain the
official free 5.4.0 release in a separate private user cache. Source videos,
existing result files, application installs, registries and HSE's EXE are not
changed. Never claims OCR transcription accuracy or source timing perfection.
"""
from __future__ import annotations

import hashlib
import os
from pathlib import Path
import shutil
import stat
import subprocess
import time
from typing import Callable
import urllib.parse
import urllib.request
from uuid import uuid4
import zipfile

from .external_subprocess import external_popen
from .native_vsf_assets import (
    _is_link_or_junction, _publish_exclusive, _safe_member,
    OfficialVsfAssetError,
)

OFFICIAL_VERSION = "v5.4.0"
OFFICIAL_URL = (
    "https://github.com/SubtitleEdit/subtitleedit/releases/download/"
    "v5.4.0/SeConv-Windows-x64.zip"
)
OFFICIAL_ARCHIVE_SHA256 = (
    "aea19664f460fcfb84e8ba8eca72419091818c6005d4e063d9950c41ea81062d"
)
OFFICIAL_ARCHIVE_BYTES = 43026089
_MAX_EXPANDED = 256 * 1024 * 1024
_MAX_FILES = 3000
_ALLOWED_HOSTS = frozenset({
    "github.com", "release-assets.githubusercontent.com",
    "objects.githubusercontent.com",
})


class OfficialSeconvError(RuntimeError):
    """Candidate external converter cannot be admitted without changing state."""


def _sha(path: Path) -> str:
    digest=hashlib.sha256()
    with path.open("rb") as stream:
        for part in iter(lambda:stream.read(256*1024),b""):
            digest.update(part)
    return digest.hexdigest()


class _SafeRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, url):
        parsed=urllib.parse.urlparse(url)
        if (
            parsed.scheme!="https" or parsed.username or parsed.password
            or (parsed.hostname or "").lower() not in _ALLOWED_HOSTS
        ):
            raise OfficialSeconvError("untrusted official release redirect")
        return super().redirect_request(req, fp, code, msg, headers, url)


def _download(target: Path, check_cancelled: Callable[[], None] | None) -> None:
    if target.exists() or _is_link_or_junction(target):
        raise OfficialSeconvError("official archive destination already exists")
    opener=urllib.request.build_opener(
        urllib.request.ProxyHandler({}),_SafeRedirect(),
    )
    req=urllib.request.Request(
        OFFICIAL_URL,headers={"User-Agent":"HSE-official-SeConv/5.4.0"}
    )
    length=0
    sha=hashlib.sha256()
    deadline=time.monotonic()+240.0
    try:
        with opener.open(req,timeout=75) as response:
            final=urllib.parse.urlparse(response.geturl())
            if response.status!=200 or final.scheme!="https" or (
                final.hostname or ""
            ).lower() not in _ALLOWED_HOSTS:
                raise OfficialSeconvError("official archive origin invalid")
            with target.open("xb") as out:
                while True:
                    if check_cancelled:
                        check_cancelled()
                    if time.monotonic()>deadline:
                        raise OfficialSeconvError("official archive deadline exceeded")
                    part=response.read(256*1024)
                    if not part:
                        break
                    length+=len(part)
                    if length>OFFICIAL_ARCHIVE_BYTES:
                        raise OfficialSeconvError("official archive oversized")
                    sha.update(part)
                    out.write(part)
        if length!=OFFICIAL_ARCHIVE_BYTES or sha.hexdigest()!=OFFICIAL_ARCHIVE_SHA256:
            raise OfficialSeconvError("official archive bytes do not match SHA-256")
    except BaseException:
        target.unlink(missing_ok=True)
        raise


def _members(zf: zipfile.ZipFile):
    items=zf.infolist()
    if not 1<=len(items)<=_MAX_FILES:
        raise OfficialSeconvError("official converter archive entry count invalid")
    files=[]
    seen=set()
    total=0
    exe=None
    try:
        for member in items:
            p=_safe_member(member.filename)
            collision=tuple(x.casefold() for x in p.parts)
            if collision in seen:
                raise OfficialSeconvError("official converter archive path collision")
            seen.add(collision)
            kind=(member.external_attr>>16)&0o170000
            if kind==stat.S_IFLNK:
                raise OfficialSeconvError("official converter link forbidden")
            if member.flag_bits & 1:
                raise OfficialSeconvError("encrypted upstream file forbidden")
            if member.is_dir():
                continue
            if member.file_size<0:
                raise OfficialSeconvError("invalid native archive entry size")
            total+=member.file_size
            if total>_MAX_EXPANDED:
                raise OfficialSeconvError("official converter expanded size too large")
            files.append((p,member))
            if p.name.casefold()=="seconv.exe":
                if exe is not None:
                    raise OfficialSeconvError("ambiguous official converter executable")
                exe=p
    except OfficialVsfAssetError as exc:
        raise OfficialSeconvError("official converter archive path invalid") from exc
    if exe is None or not files:
        raise OfficialSeconvError("missing self-contained official seconv executable")
    return files,exe


def _verify_cache(folder: Path) -> tuple[Path,str]:
    if _is_link_or_junction(folder) or not folder.is_dir():
        raise OfficialSeconvError("official converter cache directory invalid")
    archive=folder/"official.zip"
    native=folder/"native"
    if any(_is_link_or_junction(p) for p in (archive,native)):
        raise OfficialSeconvError("linked official converter cache forbidden")
    if (not archive.is_file()
        or archive.stat().st_size!=OFFICIAL_ARCHIVE_BYTES
        or _sha(archive)!=OFFICIAL_ARCHIVE_SHA256
        or not native.is_dir()):
        raise OfficialSeconvError("official converter archive changed")
    try:
        with zipfile.ZipFile(archive) as zf:
            entries,rel_exe=_members(zf)
            expected={tuple(x.casefold() for x in p.parts) for p,_ in entries}
            actual=set()
            for file in native.rglob("*"):
                if _is_link_or_junction(file):
                    raise OfficialSeconvError("linked converter cache entry")
                if not file.is_file():
                    continue
                folded=tuple(x.casefold() for x in file.relative_to(native).parts)
                if folded in actual:
                    raise OfficialSeconvError("case-colliding converter cached file")
                actual.add(folded)
            if expected!=actual:
                raise OfficialSeconvError("converter portable files changed")
            for rel,info in entries:
                dest=native.joinpath(*rel.parts)
                if not dest.is_file() or dest.stat().st_size!=info.file_size or (
                    dest.stat().st_nlink>1
                ):
                    raise OfficialSeconvError("converter runtime file changed")
                with zf.open(info) as source, dest.open("rb") as read:
                    while True:
                        a=source.read(256*1024)
                        b=read.read(256*1024)
                        if a!=b:
                            raise OfficialSeconvError("converter runtime bytes differ from signed archive digest")
                        if not a:
                            break
            exe=native.joinpath(*rel_exe.parts)
            return exe,_sha(exe)
    except (OSError,zipfile.BadZipFile,RuntimeError) as exc:
        if isinstance(exc,OfficialSeconvError):
            raise
        raise OfficialSeconvError("converter cache validation failed") from exc


def _unpack(archive: Path, folder: Path) -> None:
    if folder.exists() or _is_link_or_junction(folder):
        raise OfficialSeconvError("official converter extraction target occupied")
    folder.mkdir()
    with zipfile.ZipFile(archive) as zf:
        files,_=_members(zf)
        for rel,info in files:
            dest=folder.joinpath(*rel.parts)
            dest.parent.mkdir(parents=True,exist_ok=True)
            with zf.open(info) as source,dest.open("xb") as out:
                copied=0
                while True:
                    data=source.read(256*1024)
                    if not data:
                        break
                    copied+=len(data)
                    if copied>info.file_size:
                        raise OfficialSeconvError("official converter entry expanded too far")
                    out.write(data)
                if copied!=info.file_size:
                    raise OfficialSeconvError("official converter truncated file")


def _smoke(exe: Path) -> None:
    env_keys=("SYSTEMROOT","WINDIR","PATH","PATHEXT","TEMP","TMP",
              "USERPROFILE","LOCALAPPDATA","APPDATA","DOTNET_ROOT")
    env={k:os.environ[k] for k in env_keys if k in os.environ}
    child=external_popen(
        [str(exe),"--version"],cwd=str(exe.parent),
        stdout=subprocess.PIPE,stderr=subprocess.DEVNULL,
        stdin=subprocess.DEVNULL,env=env,
    )
    try:
        out,_=child.communicate(timeout=30)
    except subprocess.TimeoutExpired:
        child.kill()
        child.communicate()
        raise OfficialSeconvError("new converter startup timed out")
    if (child.returncode!=0 or len(out)>2048 or b"5.4.0" not in out):
        raise OfficialSeconvError("new converter version check failed")


def default_cache_base() -> Path:
    if os.name=="nt":
        base=os.environ.get("LOCALAPPDATA")
        if not base:
            raise OfficialSeconvError("private Windows app data unavailable")
        return Path(base)/"HardSubtitleExtractor"/"optional-native-tools"
    base=Path(os.environ["XDG_CACHE_HOME"]) if os.environ.get("XDG_CACHE_HOME") else Path.home()/".cache"
    return base/"HardSubtitleExtractor"/"optional-native-tools"


def ensure_official_seconv_54(
    *,consent:bool,cache_base:Path|None=None,
    check_cancelled:Callable[[],None]|None=None,
) -> Path:
    """Return the verified official 5.4 exe, never replacing 5.3.0."""
    if consent is not True:
        raise PermissionError("explicit approved optional SeConv 5.4 required")
    root=Path(cache_base) if cache_base is not None else default_cache_base()
    if _is_link_or_junction(root):
        raise OfficialSeconvError("linked user cache root forbidden")
    if check_cancelled:
        check_cancelled()
    target=root/"SeConv-v5.4.0"
    if target.exists() or _is_link_or_junction(target):
        exe,_=_verify_cache(target)
        if check_cancelled:
            check_cancelled()
        _smoke(exe)
        return exe
    root.mkdir(parents=True,exist_ok=True,mode=0o700)
    stage=root/(".seconv-new-"+uuid4().hex)
    stage.mkdir(mode=0o700)
    try:
        archive=stage/"official.zip"
        _download(archive,check_cancelled)
        _unpack(archive,stage/"native")
        exe,_=_verify_cache(stage)
        if check_cancelled:
            check_cancelled()
        _smoke(exe)
        if target.exists() or _is_link_or_junction(target):
            existing,_=_verify_cache(target)
            _smoke(existing)
            return existing
        _publish_exclusive(stage,target)
        return target/"native"/exe.relative_to(stage/"native")
    finally:
        if stage.exists():
            shutil.rmtree(stage,ignore_errors=True)
