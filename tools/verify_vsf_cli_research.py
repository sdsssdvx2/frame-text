"""Isolated, official SHA-pinned VideoSubFinder CLI 6.10.5 research unpacker.

This module only supports the fixed upstream Windows x64 archive. NO user
installation and NO fetched setup scripts. All extracted archive members remain
under a fresh temporary directory. Bundling the GPL binary is NOT permitted
by this research step, and it is NOT incorporated in HSE release builds.
"""
from __future__ import annotations

import hashlib
import os
from pathlib import Path, PurePosixPath
import shutil
import stat
import urllib.request
import zipfile

URL = (
    "https://github.com/eritpchy/videosubfinder-cli/releases/download/"
    "v6.10.5/videosubfinder-cli-windows-x64.zip"
)
EXPECTED_SHA256 = "60c17f0fa7dde94a51adb468898a7ed4ac33b250fd68af3d736b19989942cb35"
MAX_ARCHIVE_BYTES = 16 * 1024 * 1024
MAX_TOTAL_EXPANDED = 96 * 1024 * 1024
MAX_MEMBERS = 120


def archive_is_official(data: bytes) -> bool:
    return (
        bool(data) and len(data) <= MAX_ARCHIVE_BYTES
        and hashlib.sha256(data).hexdigest() == EXPECTED_SHA256
    )


def download_pinned_archive(destination: Path) -> Path:
    if destination.exists() or destination.is_symlink():
        raise RuntimeError("research archive path must be new")
    partial = destination.with_suffix(".part")
    if partial.exists() or partial.is_symlink():
        raise RuntimeError("research partial file path must be new")
    h = hashlib.sha256()
    size = 0
    try:
        request = urllib.request.Request(
            URL, headers={"User-Agent": "HSE-isolated-vsf-6.10.5-research/1"},
        )
        with urllib.request.urlopen(request, timeout=75) as response, partial.open("xb") as f:
            while chunk := response.read(1024 * 1024):
                size += len(chunk)
                if size > MAX_ARCHIVE_BYTES:
                    raise RuntimeError("official archive size ceiling exceeded")
                h.update(chunk)
                f.write(chunk)
        if not size or h.hexdigest() != EXPECTED_SHA256:
            raise RuntimeError("upstream archive digest mismatch")
        partial.rename(destination)
    except BaseException:
        partial.unlink(missing_ok=True)
        raise
    return destination


def _safe_zip_member(name: str) -> PurePosixPath:
    if not name or len(name) > 220 or chr(92) in name or ":" in name or "\x00" in name:
        raise ValueError("unsafe archive member name")
    p = PurePosixPath(name)
    if (
        p.is_absolute()
        or any(part in {"", ".", ".."} for part in name.strip("/").split("/"))
    ):
        raise ValueError("unsafe archive member path")
    return p


def unpack_pinned_archive(archive: Path, target: Path) -> Path:
    """Verify *archive bytes* and ZIP metadata before extracting ANY member."""
    if target.exists() or target.is_symlink():
        raise RuntimeError("isolation directory must be new")
    data = archive.read_bytes()
    if not archive_is_official(data):
        raise RuntimeError("official binary archive digest mismatch")
    with zipfile.ZipFile(archive) as zf:
        members = zf.infolist()
        if not 1 <= len(members) <= MAX_MEMBERS:
            raise ValueError("archive member count rejected")
        total = 0
        executable = []
        paths = set()
        for info in members:
            p = _safe_zip_member(info.filename)
            norm = tuple(part.casefold() for part in p.parts)
            if norm in paths:
                raise ValueError("duplicate zip paths")
            paths.add(norm)
            if (info.external_attr >> 16) & 0o170000 == stat.S_IFLNK:
                raise ValueError("archive links not permitted")
            if info.file_size < 0 or info.file_size > MAX_TOTAL_EXPANDED:
                raise ValueError("oversized archive member")
            total += info.file_size
            if total > MAX_TOTAL_EXPANDED:
                raise ValueError("expanded archive exceeds bound")
            if p.name.lower() == "videosubfindercli.exe":
                executable.append(p)
        if len(executable) != 1:
            raise ValueError("expected exactly one VideoSubFinderCli.exe")
        target.mkdir(parents=True)
        try:
            for info in members:
                p = _safe_zip_member(info.filename)
                dest = target.joinpath(*p.parts)
                if info.is_dir():
                    dest.mkdir(parents=True, exist_ok=True)
                    continue
                dest.parent.mkdir(parents=True, exist_ok=True)
                with zf.open(info, "r") as src, dest.open("xb") as out:
                    shutil.copyfileobj(src, out, 1024 * 1024)
                    if out.tell() != info.file_size:
                        raise RuntimeError("truncated zip member")
            exe = target.joinpath(*executable[0].parts).resolve(strict=True)
            if not exe.is_relative_to(target.resolve(strict=True)):
                raise RuntimeError("executable escaped research extraction")
            if not exe.is_file() or exe.stat().st_size == 0:
                raise RuntimeError("research executable not found")
        except BaseException:
            shutil.rmtree(target, ignore_errors=True)
            raise
    return exe
