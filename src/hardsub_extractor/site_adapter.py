"""Optional site-specific episode adapter.

The public Hard Subtitle Extractor core contains no adapter for any particular
streaming website. A private build may add ``_site_adapter_private.py`` (and
the modules it imports); when it is importable, the GUI and CLI expose the
extra "site episode URL" source mode. Without it, that mode is hidden and the
CLI rejects site URLs with a clear message.
"""
from __future__ import annotations

from pathlib import Path


def site_adapter_available() -> bool:
    return SITE_ADAPTER is not None


def is_site_episode_url(value: str) -> bool:
    if SITE_ADAPTER is None:
        return False
    return bool(SITE_ADAPTER.is_episode_url(value))


def source_subtitle_base(out_base: str | Path) -> Path:
    """Source-timeline base must NEVER resolve to target-timeline SRT/ASS.

    Path.with_suffix(".srt") replaces (not appends) a preceding suffix, so a
    naive ``name + ".x-source"`` would collide with the final ``name.srt``.
    """
    base = Path(out_base)
    base_without_ext = base.with_suffix("")
    stem = "".join(
        char if char.isalnum() or char in "-_" else "_"
        for char in base_without_ext.name
    )
    if not stem:
        raise ValueError("invalid subtitle output base")
    extracted_base = base.parent / (stem + "-epsite-source")
    if extracted_base.with_suffix(".srt") == base.with_suffix(".srt"):
        raise ValueError("source and synced subtitle paths must differ")
    return extracted_base


# Loaded last: the private adapter imports source_subtitle_base from here.
def _load():
    import importlib

    name = __package__ + "._site_adapter_private"  # private builds only
    try:
        adapter = importlib.import_module(name)
    except ModuleNotFoundError as exc:
        # Only "the private module is not shipped" means public build; any other
        # import failure inside a private build must surface, not hide the mode.
        if exc.name != name:
            raise
        return None
    return adapter


SITE_ADAPTER = _load()
