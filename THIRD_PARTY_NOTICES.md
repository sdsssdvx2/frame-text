# Hard Subtitle Extractor — Third-Party Notices

This application bundles or depends on open-source components. The list below records the major runtime components used by the Windows build. It is informational and does not replace the license text or notices supplied by each upstream project.

## FFmpeg / FFprobe

- Project: FFmpeg
- Upstream: https://ffmpeg.org/
- Source: https://git.ffmpeg.org/ffmpeg.git
- Bundled build provider: https://github.com/BtbN/FFmpeg-Builds
- Pinned provider release: `autobuild-2026-11-01-12-49`
- Pinned asset: `ffmpeg-n9.0.2-42-gd7643a0fe3-win64-lgpl-9.0.zip`
- FFmpeg revision in that asset: `n9.0.2-42-gd7643a0fe3`
- Archive SHA-256: `187fef3a0221148eaf9c342c82309209277d84b2257efcac7d230687b77f2862`
- Variant used by this project: LGPL build; the build gate rejects a configuration containing `--enable-gpl`.
- License information and source instructions: https://ffmpeg.org/legal.html and https://ffmpeg.org/download.html#get-sources

The Windows artifact also contains `ffmpeg-provenance.json` with the exact binary hashes, version strings, and build configuration captured at build time.

## yt-dlp (bundled public caption-only extractor)

- Project: yt-dlp
- Upstream: https://github.com/yt-dlp/yt-dlp
- Pinned Windows x64 standalone release: `2026.10.26`
- Official binary: `https://github.com/yt-dlp/yt-dlp/releases/download/2026.10.26/yt-dlp.exe`
- Official GitHub release asset SHA-256: `f5389d3698ecde4ef06f4e6b1ce0294fe9ac1396228c8990b8d3416d6d177a7d`
- Main yt-dlp project license: Unlicense (public domain dedication), https://github.com/yt-dlp/yt-dlp/blob/2026.10.26/LICENSE
- Official packaged executable includes independently licensed bundled components; consult the official release `THIRD_PARTY_LICENSES.txt` and `https://github.com/yt-dlp/yt-dlp/releases/tag/2026.10.26` for their notices. This HSE summary does not replace upstream third-party license terms.
- HSE uses it exclusively for subtitle metadata/downloads via a separately spawned binary with `--ignore-config --no-plugin-dirs --no-playlist --skip-download`; HSE neither imports browser cookies nor attempts to decrypt protected video.
- The CI build verifies the exact SHA-256 before including the binary; the frozen self-test verifies its version.

## Subtitle Edit 5.3 / seconv (free portable embedded-image subtitle converter)

- Project: Subtitle Edit 5, `seconv` headless Windows x64 converter.
- Official source and license: https://github.com/SubtitleEdit/subtitleedit/blob/v5.3.0/LICENSE
- License: MIT (Copyright (c) 2026 Nikolaj Olsson). The complete upstream MIT license text is also included in the frozen app as `licenses/seconv-MIT-LICENSE.txt`.
- Version and archive: `v5.3.0`, `SeConv-Windows-x64.zip` from https://github.com/SubtitleEdit/subtitleedit/releases/tag/v5.3.0
- Archive SHA-256: `1bcafcd5fd84d445b92d33dbfef3b95d268e5d36df34272c003d83d1f37b72f7`.
- Bundle includes the complete official portable application directory and its adjacent self-contained .NET runtime dependencies. No purchase, subscription, system-level installation or user-supplied .NET runtime is required. Any independent third-party licenses/notices present in that official portable directory remain included.
- HSE uses it only for an explicitly language-matched PGS or VobSub image track in a local Matroska video. For multi-language OCR, the converter preserves original subpicture bitmaps into isolated BDN PNGs, while HSE's existing offline PP-OCR model performs the text recognition. It never acts as a browser/VPN/DRM helper.
- HSE's runtime and CI validate the official archive hash before bundling it; `seconv-provenance.json` records exact staged runtime-file SHA-256 values in the Windows build evidence.
- PGS/VobSub original bitmap OCR is not exact original text transcription, and BDN timestamps are quantized to frame precision; see the original-output provenance and acceptance documentation.

## RapidOCR

- Runtime version pinned by this project: `3.10.0`
- Upstream: https://github.com/RapidAI/RapidOCR
- License: Apache License 2.0

## PaddleOCR / PP-OCRv5 model assets

- Model family: PP-OCRv5 Korean detector/recognizer/dictionary
- Upstream: https://github.com/PaddlePaddle/PaddleOCR
- License: Apache License 2.0
- This project pins the model files by SHA-256 in `models/manifest.json` and accepts mirrors only when the bytes match the pinned digest.

## ONNX Runtime

- Runtime version pinned by this project: `1.31.1`
- Upstream: https://github.com/microsoft/onnxruntime
- License: MIT

## OpenVINO

- Runtime version pinned by the accelerated Windows build: `2026.4.1`
- Upstream: https://github.com/openvinotoolkit/openvino
- License: Apache License 2.0
- HSE uses OpenVINO only as an explicitly selected free CPU inference backend for the same SHA-pinned OCR models. The established ONNX Runtime path remains available and unchanged.
- The accelerated frozen Windows build bundles this runtime so the user does not need a separate purchase, subscription, or OpenVINO installation.

## OpenCV

- Windows validation version: OpenCV `5.0.0` (`opencv-python-headless` wheel `5.0.0.93` in the verified build)
- Upstream: https://opencv.org/
- License: Apache License 2.0 for OpenCV 4.5.0 and later

## NumPy

- Windows validation version: `2.5.4`
- Upstream: https://numpy.org/
- License: BSD-3-Clause for NumPy itself. Binary wheels can contain separately licensed runtime components; retain the metadata/notices shipped with the wheel when redistributing them separately.

## Pillow

- Windows validation version: `13.0.0`
- Upstream: https://python-pillow.github.io/
- License: MIT-CMU

## Playwright

- Python runtime version pinned by this project: `1.64.1`
- Upstream: https://github.com/microsoft/playwright-python and https://github.com/microsoft/playwright
- License: Apache License 2.0
- The Windows artifact bundles the Playwright client/runtime used to control an already installed Chrome, Edge, or Chromium executable. HSE does not bundle a Playwright-downloaded browser binary and does not import an existing user browser profile, cookies, login state, extensions, or history.

## PyInstaller bootloader

- Build-time version used by the verified Windows artifact: `6.22.3`
- Upstream: https://pyinstaller.org/ and https://github.com/pyinstaller/pyinstaller
- License: GPL-2.0-or-later WITH the PyInstaller Bootloader Exception. The exception permits distributing applications containing the PyInstaller bootloader without applying the GPL to the application as a whole, subject to the exception terms.

## Other Python runtime dependencies

RapidOCR, ONNX Runtime, and Playwright bring additional permissively licensed Python packages into the frozen application, including packages such as Shapely, PyYAML, Requests, OmegaConf, pyclipper, protobuf, flatbuffers, charset-normalizer, idna, urllib3, certifi, tqdm, colorlog, six, pyee, greenlet, and typing-extensions. Their upstream package metadata and license files remain the authoritative source for their exact terms.

For the exact versions in a particular Windows artifact, use `runtime-build-inventory.json` and `release-manifest.json` shipped with that artifact.
