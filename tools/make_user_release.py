"""Assemble the user-facing Windows release ZIP (source runtime + installer).

Layout of the ZIP::

    HSE-<tag>/
      Install-HSE.cmd, install.ps1, Uninstall-HSE.cmd, Restore-Previous-HSE.cmd,
      uninstall.ps1, launch_hse.pyw, requirements-windows.txt, README-KO.md
      bin/   ffmpeg.exe ffprobe.exe yt-dlp.exe   (optional, SHA-256 verified)
      app/   src/ models/manifest.json tools/prepare_models.py
             THIRD_PARTY_NOTICES.md third_party/ pyproject.toml

Only files tracked by git (at HEAD) are copied from the repository, so local
scratch files, models or outputs can never leak into the release.

    python tools/make_user_release.py --out dist/HSE.zip --tag Claude-Final-20261115 \
        --bin-dir /path/with/verified/exes --readme README-KO.md
"""
from __future__ import annotations

import argparse
import hashlib
import subprocess
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PACKAGING = ROOT / "packaging" / "windows"
PINNED_BINARIES = {
    "ffmpeg.exe": "a4073d6e566266c8bc240a191bedfca66132592f5ae033313bbbe1b8d2de0bb5",
    "ffprobe.exe": "efe458c219a4f0ed53b50c65abc53e49ebfff94029e1cf9504d920ae7af78463",
    "yt-dlp.exe": "f5389d3698ecde4ef06f4e6b1ce0294fe9ac1396228c8990b8d3416d6d177a7d",
}
APP_PREFIXES = ("src/", "third_party/")
APP_FILES = (
    "models/manifest.json",
    "tools/prepare_models.py",
    "THIRD_PARTY_NOTICES.md",
    "pyproject.toml",
)
TOP_FILES = (
    "Install-HSE.cmd",
    "install.ps1",
    "Uninstall-HSE.cmd",
    "Restore-Previous-HSE.cmd",
    "uninstall.ps1",
    "launch_hse.pyw",
    "requirements-windows.txt",
)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def tracked_files() -> list[str]:
    out = subprocess.run(
        ["git", "-C", str(ROOT), "ls-files", "-z"],
        check=True, capture_output=True,
    ).stdout.decode("utf-8")
    return [p for p in out.split("\0") if p]


def build(out: Path, tag: str, bin_dir: Path | None, readme: Path | None) -> list[str]:
    base = f"HSE-{tag}/"
    files = tracked_files()
    added: list[str] = []
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        raise SystemExit(f"refusing to overwrite existing {out}")
    with zipfile.ZipFile(out, "x", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for name in TOP_FILES:
            src = PACKAGING / name
            if not src.is_file():
                raise SystemExit(f"missing packaging file {src}")
            z.write(src, base + name)
            added.append(name)
        for rel in files:
            if rel.startswith(APP_PREFIXES) or rel in APP_FILES:
                if "__pycache__" in rel:
                    continue
                z.write(ROOT / rel, base + "app/" + rel)
                added.append("app/" + rel)
        if readme is not None:
            z.write(readme, base + "README-KO.md")
            added.append("README-KO.md")
        if bin_dir is not None:
            for name, expected in PINNED_BINARIES.items():
                src = bin_dir / name
                if not src.is_file():
                    continue
                actual = _sha256(src)
                if actual != expected:
                    raise SystemExit(f"{name} SHA-256 mismatch: {actual}")
                z.write(src, base + "bin/" + name)
                added.append("bin/" + name)
    return added


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--bin-dir")
    ap.add_argument("--readme")
    a = ap.parse_args()
    added = build(
        Path(a.out), a.tag,
        Path(a.bin_dir) if a.bin_dir else None,
        Path(a.readme) if a.readme else None,
    )
    print(f"{len(added)} files -> {a.out}")
    for name in added:
        if not name.startswith("app/src/"):
            print("  " + name)


if __name__ == "__main__":
    main()
