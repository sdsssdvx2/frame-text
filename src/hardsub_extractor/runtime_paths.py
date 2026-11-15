from __future__ import annotations

from pathlib import Path
import os
import shutil
import sys


def bundle_root() -> Path:
    """
    Root for bundled data.

    - PyInstaller onefile: sys._MEIPASS
    - source checkout/editable install: project root
    """
    frozen_root = getattr(sys, "_MEIPASS", None)
    if frozen_root:
        return Path(frozen_root)
    return Path(__file__).resolve().parents[2]


def bundled_model_dir() -> Path:
    return bundle_root() / "models" / "runtime"


def bundled_manifest_path() -> Path:
    return bundle_root() / "models" / "manifest.json"


def resolve_binary(name: str) -> str:
    env_key = f"HSE_{name.upper()}"
    explicit = os.environ.get(env_key)
    if explicit:
        p = Path(explicit)
        if p.is_file():
            return str(p)
        raise FileNotFoundError(f"{env_key} points to a missing file: {p}")

    candidates = []
    suffixes = [".exe", ""] if os.name == "nt" else ["", ".exe"]
    for suffix in suffixes:
        candidates.append(bundle_root() / "bin" / f"{name}{suffix}")

    for p in candidates:
        if p.is_file():
            return str(p)

    found = shutil.which(name)
    if found:
        return found

    raise FileNotFoundError(
        f"{name} was not found. Expected bundled bin/{name} or PATH."
    )
