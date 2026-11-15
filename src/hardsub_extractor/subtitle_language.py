"""Explicit language preferences for multilingual soft subtitle acquisition.

Source language tags are *metadata*, not a guess from Han glyph shapes. We do
not infer Chinese vs Japanese solely from shared CJK characters. Old Korean OCR
and Korean-only EpSite selection remain unchanged.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

_LANGUAGES = frozenset({"ko", "en", "ja", "zh", "zh-hans", "zh-hant"})


def normalize_language(code: str) -> str:
    if not isinstance(code, str):
        raise ValueError("subtitle language must be a string")
    value = code.strip().replace("_", "-").lower()
    if value not in _LANGUAGES:
        raise ValueError("unsupported language; use ko, en, ja, zh, zh-Hans or zh-Hant")
    return value


def language_matches(requested: str, provided: str) -> bool:
    """Match known regional tags without mixing unrelated translations."""
    target = normalize_language(requested)
    tag = str(provided or "").replace("_", "-").lower().strip()
    if tag.endswith("-orig"):
        tag = tag[:-5]
    if not tag:
        return False
    if target == "zh-hans":
        return tag in {"zh-hans", "zh-cn", "zh-sg"}
    if target == "zh-hant":
        return tag in {"zh-hant", "zh-tw", "zh-hk", "zh-mo"}
    if target == "zh":
        return tag in {"zh", "zh-hans", "zh-hant", "zh-cn", "zh-sg", "zh-tw", "zh-hk", "zh-mo"}
    if target == "ko":
        return tag in {"ko", "ko-kr", "ko-orig"}
    if target == "ja":
        return tag in {"ja", "ja-jp", "ja-orig"}
    if target == "en":
        return tag in {"en", "en-us", "en-gb", "en-ca", "en-au", "en-ie", "en-in", "en-orig"}
    return False


@dataclass(frozen=True, slots=True)
class SubtitleTrack:
    language: str
    origin: str  # "manual" or "automatic"
    extension: str
    requested_language: str

    def __post_init__(self) -> None:
        if self.origin not in {"manual", "automatic"}:
            raise ValueError("invalid subtitle origin")
        if self.extension not in {"vtt", "srt", "ass", "ssa"}:
            raise ValueError("unsupported subtitle extension")


def candidate_tracks(
    requested_language: str,
    manual_tracks: Mapping[str, object],
    automatic_tracks: Mapping[str, object],
) -> list[SubtitleTrack]:
    """Return bounded alternatives, manual before automatic.

    A bad or unavailable manual track should not discard another matching
    manual format or regional variant. The collector tries these in rank
    order, then automatic tracks of the SAME language. Duplicate
    language/format offers need only one yt-dlp attempt.
    """
    requested = normalize_language(requested_language)
    fmt_rank = {"vtt": 0, "srt": 1, "ass": 2, "ssa": 3}
    selected: list[SubtitleTrack] = []
    for origin, source, max_attempts in (
        ("manual", manual_tracks, 3),
        ("automatic", automatic_tracks, 2),
    ):
        if not isinstance(source, Mapping):
            continue
        candidates: list[tuple[tuple[int, int, int, int, str], SubtitleTrack]] = []
        for raw_tag, entries in source.items():
            tag = str(raw_tag)
            if not language_matches(requested, tag):
                continue
            if not isinstance(entries, Sequence) or isinstance(entries, (str, bytes)):
                continue
            for entry in entries:
                if not isinstance(entry, Mapping):
                    continue
                extension = str(entry.get("ext") or "").lower().lstrip(".")
                if extension not in fmt_rank:
                    continue
                canonical = tag.lower().replace("_", "-")
                is_orig = canonical.endswith("-orig")
                without_orig = canonical[:-5] if is_orig else canonical
                origin_penalty = int(origin == "automatic" and not is_orig)
                exact_penalty = int(without_orig != requested)
                regional_penalty = (
                    0 if without_orig in {"zh-hans", "zh-cn"} else 1
                ) if requested == "zh" else 0
                order = (
                    origin_penalty,
                    exact_penalty,
                    regional_penalty,
                    fmt_rank[extension],
                    tag,
                )
                candidates.append((order, SubtitleTrack(
                    tag, origin, extension, requested,
                )))
        seen: set[tuple[str, str]] = set()
        for _, track in sorted(candidates, key=lambda item: item[0]):
            identity = (track.language.lower(), track.extension)
            if identity in seen:
                continue
            seen.add(identity)
            selected.append(track)
            if len(seen) >= max_attempts:
                break
    return selected


def select_track(
    requested_language: str,
    manual_tracks: Mapping[str, object],
    automatic_tracks: Mapping[str, object],
) -> SubtitleTrack | None:
    """Return the preferred track, retained for simple one-track callers."""
    tracks = candidate_tracks(requested_language, manual_tracks, automatic_tracks)
    return tracks[0] if tracks else None
