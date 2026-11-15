# -*- mode: python ; coding: utf-8 -*-
from pathlib import Path
from PyInstaller.utils.hooks import collect_all

ROOT = Path(SPECPATH)
SRC = ROOT / "src"

datas = []
binaries = []
hiddenimports = [
    "hardsub_extractor.review_pack",
    "hardsub_extractor.temporal_review",
    "hardsub_extractor.quality_audit_io",
    "hardsub_extractor.frozen_frame_review_acceptance",
]

for package in ("rapidocr", "onnxruntime", "openvino", "playwright"):
    d, b, h = collect_all(package)
    datas += d
    binaries += b
    hiddenimports += h

manifest = ROOT / "models" / "manifest.json"
runtime_models = ROOT / "models" / "runtime"
notices = ROOT / "THIRD_PARTY_NOTICES.md"
if manifest.is_file():
    datas.append((str(manifest), "models"))
if runtime_models.is_dir():
    datas.append((str(runtime_models), "models/runtime"))
if notices.is_file():
    datas.append((str(notices), "."))
else:
    raise SystemExit(f"Missing third-party notices: {notices}")

for name in ("ffmpeg.exe", "ffprobe.exe", "yt-dlp.exe"):
    p = ROOT / "bin" / name
    if not p.is_file():
        raise SystemExit(f"Missing bundled binary: {p}")
    binaries.append((str(p), "bin"))

# Keep self-contained .NET seconv's adjacent DLLs/runtimeconfig/assets intact.
# Never require the user to install .NET, Tesseract or paid OCR products.
# Source/release builders must stage only the independently SHA-pinned
# official free portable archive before creating a distributable EXE.
seconv = ROOT / "bin" / "seconv"
if not (seconv / "seconv.exe").is_file():
    raise SystemExit("Missing verified portable seconv; run tools/prepare_seconv_ci.ps1")
datas.append((str(seconv), "bin/seconv"))
seconv_license = ROOT / "third_party" / "seconv-MIT-LICENSE.txt"
if not seconv_license.is_file():
    raise SystemExit("Missing mandatory upstream seconv MIT license notice")
datas.append((str(seconv_license), "licenses"))

a = Analysis(
    [str(ROOT / "tools" / "windows_entry.py")],
    pathex=[str(SRC)],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name="HardSubtitleExtractor",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
