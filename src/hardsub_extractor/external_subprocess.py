"""Launch a separately packaged external executable from a frozen HSE.

PyInstaller's Windows bootloader changes the process-wide DLL search path.
That setting is inherited by children, including the independently packaged
yt-dlp.exe. When HSE runs as a windowed onefile EXE, the nested executable
must not resolve the parent's DLLs or inherit its bootloader state.

Apply the documented reset only around CreateProcess, then restore HSE's
own bundled DLL search directory immediately. Do not change user settings,
system PATH, persistent environment, browser state or the installed EXE.
"""
from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys
import threading
from typing import Sequence

_DLL_SPAWN_LOCK = threading.RLock()


def _external_env() -> dict[str, str] | None:
    """Preserve all user proxy/environment choices unless HSE is frozen."""
    if not getattr(sys, "frozen", False):
        return None
    env = dict(os.environ)
    # Ensure a separately packaged PyInstaller child starts as a new app.
    env["PYINSTALLER_RESET_ENVIRONMENT"] = "1"
    return env


def _set_windows_dll_directory(value: str | None) -> None:
    import ctypes

    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    setter = kernel.SetDllDirectoryW
    setter.argtypes = [ctypes.c_wchar_p]
    setter.restype = ctypes.c_int
    if not setter(value):
        raise OSError("could not isolate external executable DLL search")


def external_popen(
    command: Sequence[str], **kwargs,
) -> subprocess.Popen:
    """Spawn immediately and restore the original DLL path before returning.

    The process-wide SetDllDirectoryW modification lasts only until
    CreateProcess returns. A lock serializes HSE's external launches; the
    subprocess itself is never held under the DLL lock.
    """
    opts = dict(kwargs)
    env = _external_env()
    if env is not None:
        custom = opts.pop("env", None)
        if custom is not None:
            env.update(custom)
        env["PYINSTALLER_RESET_ENVIRONMENT"] = "1"
        opts["env"] = env

    # A bundled command-line helper must not flash a console window when
    # invoked from HSE's Windows GUI, even on the normal caption path.
    if sys.platform == "win32":
        opts["creationflags"] = opts.get("creationflags", 0) | subprocess.CREATE_NO_WINDOW

    # Non-frozen Python and non-Windows systems must not touch DLL search.
    root = getattr(sys, "_MEIPASS", None)
    if not (getattr(sys, "frozen", False) and sys.platform == "win32" and root):
        return subprocess.Popen(list(command), **opts)

    with _DLL_SPAWN_LOCK:
        _set_windows_dll_directory(None)
        child = None
        try:
            child = subprocess.Popen(list(command), **opts)
        finally:
            try:
                _set_windows_dll_directory(str(Path(root)))
            except BaseException:
                # A failed restore is unsafe for HSE's own native modules.
                if child is not None:
                    try:
                        child.kill()
                        child.wait(timeout=5)
                    except (OSError, subprocess.TimeoutExpired):
                        pass
                raise
    return child
