"""Start Hard Subtitle Extractor from an installed source tree (no console).

The installer places this file next to ``src``, ``models`` and ``bin`` and
runs it with the private virtual environment's ``pythonw.exe``. Paths are
resolved relative to this file, so the install folder can be renamed or
restored from a backup without breaking the launcher.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def _main() -> None:
    try:
        from hardsub_extractor.gui_entry import main
    except Exception as exc:  # pragma: no cover - only for broken installs
        try:
            import tkinter
            from tkinter import messagebox

            root = tkinter.Tk()
            root.withdraw()
            messagebox.showerror(
                "Hard Subtitle Extractor",
                "프로그램 파일을 불러오지 못했습니다. 설치 스크립트를 다시 실행하세요.\n\n"
                f"{type(exc).__name__}: {exc}",
            )
        finally:
            raise SystemExit(3)
    main()


if __name__ == "__main__":
    _main()
