from __future__ import annotations

from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
import json
import subprocess
import sys

import numpy as np

from .model_assets import resolve_model_paths
from .external_subprocess import external_popen
from .runtime_paths import (
    bundled_manifest_path,
    bundled_model_dir,
    bundle_root,
    resolve_binary,
)


@dataclass(slots=True)
class Check:
    name: str
    ok: bool
    detail: str


@dataclass(slots=True)
class SelfTestReport:
    checks: list[Check]

    @property
    def ok(self) -> bool:
        return all(x.ok for x in self.checks)

    def to_json(self) -> str:
        return json.dumps(
            {"ok": self.ok, "checks": [asdict(x) for x in self.checks]},
            ensure_ascii=False,
            indent=2,
        )


def _record_progress(progress_path: str | Path | None, stage: str, detail: str = "") -> None:
    if not progress_path:
        return
    path = Path(progress_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    row = {
        "time": datetime.now(timezone.utc).isoformat(),
        "stage": stage,
        "detail": detail,
    }
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _version(binary: str) -> str:
    p = subprocess.run(
        [binary, "-version"],
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )
    first = (p.stdout or p.stderr).splitlines()
    if p.returncode != 0 or not first:
        raise RuntimeError(f"exit={p.returncode}")
    return first[0].strip()


def run_selftest(
    *,
    require_models: bool = True,
    progress_path: str | Path | None = None,
) -> SelfTestReport:
    checks: list[Check] = []
    _record_progress(progress_path, "start")

    for name in ("ffmpeg", "ffprobe"):
        _record_progress(progress_path, f"{name}:start")
        try:
            path = resolve_binary(name)
            checks.append(Check(name, True, f"{path} | {_version(path)}"))
        except Exception as exc:
            checks.append(Check(name, False, str(exc)))
        _record_progress(progress_path, f"{name}:done", str(checks[-1].ok))

    # The developer/source checkout may intentionally omit yt-dlp, but the
    # frozen GUI must never claim standalone caption support without the
    # pinned companion executable included in its own extracted bundle.
    if getattr(sys, "frozen", False) or getattr(sys, "_MEIPASS", None):
        _record_progress(progress_path, "yt-dlp:start")
        version_exit: int | None = None
        version_size: int | None = None
        try:
            # Always verify the immutable bundled fallback, regardless of
            # which separately vetted managed version will serve requests.
            pinned = bundle_root() / "bin" / "yt-dlp.exe"
            actual = str(pinned)
            if not pinned.is_file():
                raise RuntimeError("bundled caption downloader is missing")
            # A nested official yt-dlp onefile executable is independent of
            # HSE's own PyInstaller bootloader and bundled DLL directory.
            # Always provide usable stdin/stdout handles in a windowed EXE.
            proc = external_popen(
                [actual, "--version"],
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
            )
            try:
                output, _ = proc.communicate(timeout=20)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.communicate()
                raise
            version_exit = proc.returncode
            version_size = len(output)
            version = output.decode("utf-8", "replace").strip()
            if version_exit != 0 or version != "2026.10.26":
                raise RuntimeError("bundled caption downloader version mismatch")
            checks.append(Check("yt-dlp", True, version))
        except Exception as exc:
            # Only numeric process outcome is safe to record; never raw
            # yt-dlp stdout/stderr, which may contain user/device details.
            detail = type(exc).__name__
            if version_exit is not None:
                detail += f"; exit_code={version_exit}; stdout_bytes={version_size}"
            checks.append(Check("yt-dlp", False, detail))
        _record_progress(progress_path, "yt-dlp:done", str(checks[-1].ok))

    # A distributable onefile HSE must include the complete free portable
    # bitmap-subtitle converter (seconv + its adjacent .NET runtime) rather
    # than silently requiring the user to install a separate app.
    if getattr(sys, "frozen", False) or getattr(sys, "_MEIPASS", None):
        _record_progress(progress_path, "seconv:start")
        outcome: int | None = None
        try:
            binary = bundle_root() / "bin" / "seconv" / "seconv.exe"
            notice = bundle_root() / "licenses" / "seconv-MIT-LICENSE.txt"
            if not binary.is_file() or not notice.is_file():
                raise RuntimeError("free portable bitmap converter or MIT license missing")
            proc = external_popen(
                [str(binary), "--version"],
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
            )
            try:
                output, _ = proc.communicate(timeout=20)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.communicate()
                raise
            outcome = proc.returncode
            import re
            if outcome != 0 or not re.search(
                r"(?<![0-9])5\.3\.0(?![0-9])",
                output[:1024].decode("utf-8", "replace"),
            ):
                raise RuntimeError("bundled free bitmap converter version mismatch")
            checks.append(Check("seconv", True, "5.3.0; local; free; MIT notice"))
        except Exception as exc:
            checks.append(Check(
                "seconv", False, f"{type(exc).__name__}"
                + (f"; exit_code={outcome}" if outcome is not None else ""),
            ))
        _record_progress(progress_path, "seconv:done", str(checks[-1].ok))

    model_dir = bundled_model_dir()
    manifest = bundled_manifest_path()
    if require_models:
        _record_progress(progress_path, "ocr-models:start")
        try:
            paths = resolve_model_paths(
                model_dir,
                manifest_path=manifest,
                verify=True,
            )
            checks.append(
                Check(
                    "ocr-models",
                    True,
                    f"det={paths.detector.name}, rec={paths.recognizer.name}, dict={paths.dictionary.name}",
                )
            )
        except Exception as exc:
            checks.append(Check("ocr-models", False, str(exc)))
        _record_progress(progress_path, "ocr-models:done", str(checks[-1].ok))
        # Each selectable OCR language must have its own verified model path.
        # The existing Korean model is still validated above unchanged.
        for language in ("en", "ja", "zh"):
            name = f"ocr-models-{language}"
            _record_progress(progress_path, name + ":start")
            try:
                selected = resolve_model_paths(
                    model_dir, manifest_path=manifest,
                    verify=True, language=language,
                )
                checks.append(Check(
                    name, True,
                    f"rec={selected.recognizer.name}, dict={selected.dictionary.name}",
                ))
            except Exception as exc:
                checks.append(Check(name, False, type(exc).__name__))
            _record_progress(progress_path, name + ":done", str(checks[-1].ok))
    else:
        checks.append(
            Check(
                "ocr-models",
                True,
                "not required for soft-subtitle-only operation",
            )
        )

    _record_progress(progress_path, "opencv:start")
    try:
        import cv2
        checks.append(Check("opencv", True, cv2.__version__))
    except Exception as exc:
        checks.append(Check("opencv", False, str(exc)))
    _record_progress(progress_path, "opencv:done", str(checks[-1].ok))

    if require_models:
        _record_progress(progress_path, "rapidocr-import:start")
        try:
            import rapidocr
            checks.append(
                Check(
                    "rapidocr",
                    True,
                    str(getattr(rapidocr, "__version__", "import ok")),
                )
            )
        except Exception as exc:
            checks.append(Check("rapidocr", False, str(exc)))
        _record_progress(progress_path, "rapidocr-import:done", str(checks[-1].ok))

        _record_progress(progress_path, "onnxruntime-import:start")
        try:
            import onnxruntime
            checks.append(Check("onnxruntime", True, onnxruntime.__version__))
        except Exception as exc:
            checks.append(Check("onnxruntime", False, str(exc)))
        _record_progress(progress_path, "onnxruntime-import:done", str(checks[-1].ok))

        # Force the lazy-loaded RapidOCR models through one real ONNX inference.
        # A blank frame is intentional: this checks that detector/recognizer model
        # files, ONNX Runtime providers, and bundled runtime paths all execute in
        # the frozen application without making text accuracy part of self-test.
        _record_progress(progress_path, "rapidocr-inference:start")
        try:
            from .models import Frame
            from .ocr.rapidocr_adapter import RapidOCRAdapter

            engine = RapidOCRAdapter(
                model_dir=str(model_dir),
                manifest_path=str(manifest),
            )
            obs = engine.recognize(
                Frame(pts=0.0, image=np.full((192, 640, 3), 255, dtype=np.uint8))
            )
            checks.append(
                Check(
                    "rapidocr-inference",
                    True,
                    f"executed; text_len={len(obs.text)}, confidence={obs.confidence:.3f}",
                )
            )
        except Exception as exc:
            checks.append(Check("rapidocr-inference", False, str(exc)))
        _record_progress(progress_path, "rapidocr-inference:done", str(checks[-1].ok))
        # A frozen app must initialize each foreign recognizer with the actual
        # verified ONNX and dictionary bytes, not merely ship file names.
        for language in ("en", "ja", "zh"):
            name = f"rapidocr-inference-{language}"
            _record_progress(progress_path, name + ":start")
            try:
                foreign = RapidOCRAdapter(
                    model_dir=str(model_dir), manifest_path=str(manifest),
                    language=language,
                )
                observed = foreign.recognize(Frame(
                    pts=0.0, image=np.full((192, 640, 3), 255, dtype=np.uint8),
                ))
                checks.append(Check(
                    name, True,
                    f"executed; text_len={len(observed.text)}, confidence={observed.confidence:.3f}",
                ))
            except Exception as exc:
                checks.append(Check(name, False, type(exc).__name__))
            _record_progress(progress_path, name + ":done", str(checks[-1].ok))

    report = SelfTestReport(checks)
    _record_progress(progress_path, "done", str(report.ok))
    return report


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--soft-only", action="store_true")
    ap.add_argument("--progress-jsonl")
    args = ap.parse_args()
    report = run_selftest(
        require_models=not args.soft_only,
        progress_path=args.progress_jsonl,
    )
    print(report.to_json())
    raise SystemExit(0 if report.ok else 2)


if __name__ == "__main__":
    main()
