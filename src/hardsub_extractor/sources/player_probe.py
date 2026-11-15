from __future__ import annotations

from dataclasses import dataclass
import html as html_lib
import json
import re
from urllib.parse import urljoin, urlparse


@dataclass(frozen=True, slots=True)
class SubtitleCandidate:
    url: str
    source: str
    kind: str | None = None
    label: str | None = None


@dataclass(frozen=True, slots=True)
class MediaCandidate:
    url: str
    source: str
    media_type: str | None = None
    label: str | None = None
    default: bool = False
    with_credentials: bool = False


_PLAYERJS_RE = re.compile(
    r'''playerjsSubtitle\s*=\s*(?P<q>["'])(?P<value>.*?)(?P=q)''',
    re.IGNORECASE | re.DOTALL,
)
_URL_RE = re.compile(
    r'''(?:\[[^\]]*\])?\s*(https?://[^,\s"'<>]+|(?:/|\.\.?/)[^,\s"'<>]+)''',
    re.IGNORECASE,
)
_HTML_TRACK_RE = re.compile(
    r'''<track\b(?P<attrs>[^>]+)>''',
    re.IGNORECASE,
)
_ATTR_RE = re.compile(
    r'''(?P<name>[\w:-]+)\s*=\s*(?P<q>["'])(?P<value>.*?)(?P=q)''',
    re.IGNORECASE | re.DOTALL,
)
_SETUP_RE = re.compile(
    r"\bjwplayer\s*\([^)]*\)\s*\.\s*setup\s*\(",
    re.IGNORECASE,
)

_JS_OBJECT_RE = re.compile(r'''\{(?P<body>[^{}]{0,1800})\}''', re.DOTALL)
_JS_KV_RE = re.compile(
    r'''(?P<key>kind|file|src|label)\s*:\s*(?P<q>["'])(?P<value>.*?)(?P=q)''',
    re.IGNORECASE | re.DOTALL,
)


def _dedupe(items: list[SubtitleCandidate]) -> list[SubtitleCandidate]:
    result: list[SubtitleCandidate] = []
    seen: set[str] = set()
    for item in items:
        key = item.url.strip()
        if not key or key in seen:
            continue
        seen.add(key)
        result.append(item)
    return result


def _dedupe_media(items: list[MediaCandidate]) -> list[MediaCandidate]:
    """Deduplicate URLs without discarding stricter protection metadata."""
    result: list[MediaCandidate] = []
    positions: dict[str, int] = {}
    for item in items:
        key = item.url.strip()
        if not key:
            continue
        position = positions.get(key)
        if position is None:
            positions[key] = len(result)
            result.append(item)
            continue

        current = result[position]
        result[position] = MediaCandidate(
            url=current.url,
            source=current.source,
            media_type=current.media_type or item.media_type,
            label=current.label or item.label,
            default=current.default or item.default,
            # A duplicate may come from runtime allSources or the item-level
            # shortcut with stricter credential requirements. Never let an
            # earlier unprotected-looking duplicate erase that signal.
            with_credentials=current.with_credentials or item.with_credentials,
        )
    return result


def _web_media_url(value: object, base_url: str) -> str | None:
    text = str(value or "").strip()
    if not text:
        return None
    resolved = urljoin(base_url, text)
    parsed = urlparse(resolved)
    if (
        parsed.scheme.lower() not in {"http", "https"}
        or not parsed.hostname
        or parsed.username
        or parsed.password
    ):
        return None
    return resolved


def _media_type(explicit: object, url: str) -> str | None:
    value = str(explicit or "").strip().lower()
    if value:
        if "mpegurl" in value or value in {"hls", "m3u8"}:
            return "hls"
        if "dash" in value or value == "mpd":
            return "dash"
        if value in {"mp4", "video/mp4"}:
            return "mp4"
        return value
    path = (urlparse(url).path or "").lower()
    if path.endswith(".m3u8"):
        return "hls"
    if path.endswith(".mpd"):
        return "dash"
    if path.endswith((".mp4", ".m4v", ".mov", ".webm")):
        return path.rsplit(".", 1)[-1]
    return None


def _boolean_flag(value: object, *, unknown: bool = False) -> bool:
    """Normalize JSON/JWPlayer boolean-ish values without Python truthiness traps."""
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    if isinstance(value, (int, float)) and value in {0, 1}:
        return bool(value)
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "1", "yes"}:
            return True
        if lowered in {"false", "0", "no", ""}:
            return False
    return unknown


def _balanced_object(text: str, start: int) -> str | None:
    if start >= len(text) or text[start] != "{":
        return None
    depth = 0
    quote: str | None = None
    escaped = False
    for index in range(start, len(text)):
        char = text[index]
        if quote is not None:
            if escaped:
                escaped = False
                continue
            if char == "\\":
                escaped = True
                continue
            if char == quote:
                quote = None
            continue
        if char in {'"', "'"}:
            quote = char
            continue
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[start : index + 1]
            if depth < 0:
                return None
    return None


def strict_json_jwplayer_configs(html: str) -> list[dict]:
    """Read strict-JSON objects passed directly to ``jwplayer().setup()``.

    This deliberately does not evaluate JavaScript or normalize JavaScript-only
    object syntax. Unquoted keys, functions, comments, single-quoted strings,
    or other non-JSON constructs fail closed and are ignored.
    """
    text = str(html or "")
    configs: list[dict] = []
    for match in _SETUP_RE.finditer(text):
        start = match.end()
        while start < len(text) and text[start].isspace():
            start += 1
        fragment = _balanced_object(text, start)
        if fragment is None:
            continue
        try:
            payload = json.loads(fragment)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            configs.append(payload)
    return configs


def playerjs_subtitle_candidates(
    html: str,
    base_url: str,
) -> list[SubtitleCandidate]:
    match = _PLAYERJS_RE.search(html or "")
    if not match:
        return []
    value = html_lib.unescape(match.group("value"))
    found: list[SubtitleCandidate] = []
    for item in _URL_RE.finditer(value):
        found.append(
            SubtitleCandidate(
                url=urljoin(base_url, item.group(1)),
                source="playerjsSubtitle",
            )
        )
    return _dedupe(found)


def html_track_candidates(
    html: str,
    base_url: str,
) -> list[SubtitleCandidate]:
    found: list[SubtitleCandidate] = []
    for match in _HTML_TRACK_RE.finditer(html or ""):
        attrs = {
            m.group("name").lower(): html_lib.unescape(m.group("value"))
            for m in _ATTR_RE.finditer(match.group("attrs"))
        }
        kind = attrs.get("kind", "").lower()
        src = attrs.get("src") or attrs.get("file")
        if kind not in {"captions", "subtitles"} or not src:
            continue
        found.append(
            SubtitleCandidate(
                url=urljoin(base_url, src),
                source="html-track",
                kind=kind,
                label=attrs.get("label"),
            )
        )
    return _dedupe(found)


def jwplayer_html_candidates(
    html: str,
    base_url: str,
) -> list[SubtitleCandidate]:
    found: list[SubtitleCandidate] = []
    for obj in _JS_OBJECT_RE.finditer(html or ""):
        fields: dict[str, str] = {}
        for kv in _JS_KV_RE.finditer(obj.group("body")):
            fields[kv.group("key").lower()] = html_lib.unescape(kv.group("value"))
        kind = fields.get("kind", "").lower()
        file = fields.get("file") or fields.get("src")
        if kind not in {"captions", "subtitles"} or not file:
            continue
        found.append(
            SubtitleCandidate(
                url=urljoin(base_url, file),
                source="jwplayer-html",
                kind=kind,
                label=fields.get("label"),
            )
        )
    return _dedupe(found)


def _jw_items(playlist: dict | list | None) -> list[dict]:
    if isinstance(playlist, list):
        return [item for item in playlist if isinstance(item, dict)]
    if not isinstance(playlist, dict):
        return []
    nested = playlist.get("playlist")
    if isinstance(nested, list):
        return [item for item in nested if isinstance(item, dict)]
    return [playlist]


def jwplayer_media_candidates(
    playlist: dict | list | None,
    base_url: str,
) -> list[MediaCandidate]:
    """Read media URLs from documented JWPlayer playlist structures only.

    This consumes structured playlist/runtime data rather than scraping
    arbitrary JavaScript. It understands the documented item-level ``file``
    shortcut plus ``sources`` and runtime ``allSources`` arrays. Rich source
    metadata is kept in preference to a duplicate ``file`` shortcut. Non-web
    schemes and embedded credentials are ignored. ``withCredentials`` is
    represented explicitly so the resolver can fail closed.
    """
    found: list[MediaCandidate] = []
    for item in _jw_items(playlist):
        item_with_credentials = (
            _boolean_flag(item.get("withCredentials"), unknown=True)
            if "withCredentials" in item
            else False
        )
        for field_name in ("sources", "allSources"):
            sources = item.get(field_name)
            if not isinstance(sources, list):
                continue
            for source in sources:
                if not isinstance(source, dict):
                    continue
                media_url = _web_media_url(
                    source.get("file") or source.get("src"),
                    base_url,
                )
                if not media_url:
                    continue
                source_with_credentials = (
                    _boolean_flag(source.get("withCredentials"), unknown=True)
                    if "withCredentials" in source
                    else item_with_credentials
                )
                found.append(
                    MediaCandidate(
                        url=media_url,
                        source=f"jwplayer-{field_name}",
                        media_type=_media_type(source.get("type"), media_url),
                        label=str(source.get("label") or "") or None,
                        default=_boolean_flag(source.get("default")),
                        with_credentials=source_with_credentials,
                    )
                )

        direct_url = _web_media_url(item.get("file"), base_url)
        if direct_url:
            found.append(
                MediaCandidate(
                    url=direct_url,
                    source="jwplayer-file",
                    media_type=_media_type(item.get("type"), direct_url),
                    label=str(item.get("label") or "") or None,
                    default=_boolean_flag(item.get("default")),
                    with_credentials=item_with_credentials,
                )
            )
    return _dedupe_media(found)


def jwplayer_subtitle_candidates(
    playlist: dict | list | None,
    base_url: str,
) -> list[SubtitleCandidate]:
    items = _jw_items(playlist)

    tracks: list[dict] = []
    for item in items:
        direct = item.get("tracks")
        if isinstance(direct, list):
            tracks.extend(x for x in direct if isinstance(x, dict))

        sources = item.get("sources")
        if isinstance(sources, list):
            for source in sources:
                if isinstance(source, dict) and isinstance(source.get("tracks"), list):
                    tracks.extend(
                        x for x in source["tracks"] if isinstance(x, dict)
                    )

        nested_playlist = item.get("playlist")
        if isinstance(nested_playlist, list):
            for nested in nested_playlist:
                if isinstance(nested, dict) and isinstance(nested.get("tracks"), list):
                    tracks.extend(
                        x for x in nested["tracks"] if isinstance(x, dict)
                    )

    found: list[SubtitleCandidate] = []
    for track in tracks:
        kind = str(track.get("kind") or "").lower()
        file = str(track.get("file") or track.get("src") or "")
        if kind not in {"captions", "subtitles"} or not file:
            continue
        found.append(
            SubtitleCandidate(
                url=urljoin(base_url, file),
                source="jwplayer",
                kind=kind,
                label=str(track.get("label") or "") or None,
            )
        )
    return _dedupe(found)


def discover_soft_subtitles(
    *,
    player_html: str = "",
    jw_playlist: dict | list | None = None,
    base_url: str,
) -> list[SubtitleCandidate]:
    return _dedupe(
        playerjs_subtitle_candidates(player_html, base_url)
        + html_track_candidates(player_html, base_url)
        + jwplayer_html_candidates(player_html, base_url)
        + jwplayer_subtitle_candidates(jw_playlist, base_url)
    )
