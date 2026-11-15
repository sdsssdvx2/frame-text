"""Bounded, deterministic URL lists shared by GUI and CLI.

No site access and no user file modification. Reads a plaintext list only
when the user explicitly selects it, preserving input order and deduping.
"""
from __future__ import annotations

from pathlib import Path
from .subtitle_acquisition import supported_public_site


def load_episode_urls(named: list[str], url_file: str | None) -> list[str]:
    urls = [value.strip() for value in named if value.strip()]
    if url_file:
        path = Path(url_file).expanduser()
        if not path.is_file() or path.stat().st_size > 128 * 1024:
            raise ValueError("episode URL input file is missing or too large")
        for line in path.read_text(encoding="utf-8-sig").splitlines():
            line = line.strip()
            if line and not line.startswith("#"):
                urls.append(line)
    urls = list(dict.fromkeys(urls))
    if not urls or len(urls) > 50:
        raise ValueError("supply 1 to 50 individual episode URLs")
    # Reject unknown, local, non-HTTPS and credential-bearing URLs before
    # *any* external network call, rather than mix rejected items into a batch.
    if any(supported_public_site(url) is None for url in urls):
        raise ValueError("all URLs must be approved single public-video HTTPS URLs")
    return urls
