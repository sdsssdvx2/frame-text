from __future__ import annotations

from dataclasses import dataclass
from urllib.parse import urlparse

from .player_probe import (
    MediaCandidate,
    SubtitleCandidate,
    jwplayer_media_candidates,
    jwplayer_subtitle_candidates,
    strict_json_jwplayer_configs,
)


class JWPlayerPublicAdapterError(RuntimeError):
    """The public player HTML does not expose a supported direct HLS setup."""


@dataclass(frozen=True, slots=True)
class PublicJWPlayerSetup:
    hls_url: str
    subtitle_candidates: list[SubtitleCandidate]
    strict_setup_count: int
    media_candidate_count: int


def _safe_web_url(url: str, *, allow_localhost_http: bool = False) -> bool:
    try:
        parsed = urlparse(url)
    except ValueError:
        return False
    if parsed.username or parsed.password or not parsed.hostname:
        return False
    if parsed.scheme.lower() == "https":
        return True
    return (
        allow_localhost_http
        and parsed.scheme.lower() == "http"
        and parsed.hostname in {"127.0.0.1", "localhost"}
    )


def _dedupe_subtitles(
    items: list[SubtitleCandidate],
    *,
    allow_localhost_http: bool = False,
) -> list[SubtitleCandidate]:
    result: list[SubtitleCandidate] = []
    seen: set[str] = set()
    for item in items:
        if item.url in seen or not _safe_web_url(
            item.url,
            allow_localhost_http=allow_localhost_http,
        ):
            continue
        seen.add(item.url)
        result.append(item)
    return result


def _config_items(config: dict) -> list[dict]:
    playlist = config.get("playlist")
    if isinstance(playlist, list):
        return [item for item in playlist if isinstance(item, dict)]
    return [config]


def parse_public_jwplayer_setup(
    player_html: str,
    player_url: str,
    *,
    allow_localhost_http: bool = False,
) -> PublicJWPlayerSetup:
    """Extract a direct HLS source only from documented strict-JSON JWPlayer setup.

    This helper performs no network requests and no JavaScript execution. It
    does not infer hidden endpoints or normalize JavaScript object literals.
    Only media already exposed in a strict JSON object passed directly to
    ``jwplayer(...).setup({...})`` is considered.

    Sources requiring ``withCredentials`` are deliberately rejected because a
    reusable URL outside the player runtime has not been established. DASH and
    progressive files are also left unsupported here; HSE's current remote
    decode path is validated around HLS.

    Production URLs are HTTPS-only. Plain HTTP is accepted only for explicit
    localhost test fixtures, matching the direct resolver's URL policy.

    Subtitle tracks stay bound to the same strict JWPlayer playlist item as the
    selected HLS source. If more than one playlist item exposes HLS, static HTML
    alone does not prove which item is currently playing, so the adapter fails
    closed instead of using source ``default`` as an episode selector. Within a
    single item, ``default`` may still select among HLS quality/source variants.
    """
    configs = strict_json_jwplayer_configs(player_html)
    if not configs:
        raise JWPlayerPublicAdapterError(
            "player HTML did not expose a strict-JSON JWPlayer setup"
        )

    hls_items: list[tuple[list[MediaCandidate], list[SubtitleCandidate]]] = []
    media_candidate_count = 0
    for config in configs:
        for item in _config_items(config):
            media = jwplayer_media_candidates(item, player_url)
            subtitles = jwplayer_subtitle_candidates(item, player_url)
            media_candidate_count += len(media)
            item_hls = [
                candidate
                for candidate in media
                if candidate.media_type == "hls"
                and _safe_web_url(
                    candidate.url,
                    allow_localhost_http=allow_localhost_http,
                )
            ]
            if item_hls:
                hls_items.append((item_hls, subtitles))

    if not hls_items:
        raise JWPlayerPublicAdapterError(
            "strict-JSON JWPlayer setup did not expose a reusable HLS source"
        )
    if len(hls_items) > 1:
        raise JWPlayerPublicAdapterError(
            "multiple JWPlayer playlist items expose HLS; static setup is ambiguous"
        )

    hls_sources, subtitles = hls_items[0]
    eligible_hls = [
        item
        for item in hls_sources
        if not item.with_credentials
    ]
    if not eligible_hls:
        raise JWPlayerPublicAdapterError(
            "JWPlayer HLS source requires withCredentials and is not reusable "
            "by the direct HSE input path"
        )

    selected = next(
        (item for item in eligible_hls if item.default),
        eligible_hls[0],
    )
    return PublicJWPlayerSetup(
        hls_url=selected.url,
        subtitle_candidates=_dedupe_subtitles(
            subtitles,
            allow_localhost_http=allow_localhost_http,
        ),
        strict_setup_count=len(configs),
        media_candidate_count=media_candidate_count,
    )
