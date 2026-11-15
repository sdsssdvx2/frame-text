"""Opt-in, provider-neutral proxy routing for public caption acquisition.

No VPN management, credentials, browser profiles, tunnel probing or DRM support.
All routes require an explicit proxy flag for both yt-dlp metadata and captions;
the direct route explicitly overrides inherited proxy environment settings.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping
from urllib.parse import urlsplit
import re

MODES = frozenset({"direct", "required", "fallback"})
_SITES = frozenset({"youtube", "wetv", "vimeo", "dailymotion"})
_HOST = re.compile(r"[a-zA-Z0-9_.-]{1,253}\Z")


def validate_proxy_url(raw: str) -> str:
    """Allow standard unauthenticated HTTP and SOCKS5 proxy endpoints.

    Credentials are deliberately rejected: process arguments, crash reports
    and diagnostics must not contain passwords. SOCKS5H performs remote DNS.
    No path/query/fragment/userinfo is passed to yt-dlp.
    """
    value = str(raw).strip()
    if not value or len(value) > 256 or any(c.isspace() for c in value):
        raise ValueError("proxy must be an HTTP/SOCKS5 URL with host and port")
    try:
        parts = urlsplit(value)
        port = parts.port
        host = parts.hostname
    except ValueError as exc:
        raise ValueError("proxy address or port is invalid") from exc
    if (parts.scheme.lower() not in {"http", "socks5", "socks5h"}
            or not host or not (1 <= (port or 0) <= 65535)
            or parts.username is not None or parts.password is not None
            or parts.path not in {"", "/"} or parts.query or parts.fragment
            or "@" in parts.netloc):
        raise ValueError("proxy must use http/socks5/socks5h with host:port and no credentials")
    # Accept bracketed IPv6; urlsplit has already validated IPv6 brackets.
    if ":" not in host and not _HOST.fullmatch(host):
        raise ValueError("proxy host is invalid")
    return value


@dataclass(frozen=True, slots=True)
class CaptionProxyRoute:
    mode: str = "direct"  # direct / required / fallback
    url: str = ""

    def __post_init__(self) -> None:
        if self.mode not in MODES:
            raise ValueError("invalid proxy routing mode")
        if self.mode == "direct":
            if self.url:
                raise ValueError("direct mode cannot carry a proxy URL")
        else:
            object.__setattr__(self, "url", validate_proxy_url(self.url))


def parse_site_proxy_specs(
    required: list[str] | tuple[str, ...] = (),
    fallback: list[str] | tuple[str, ...] = (),
) -> dict[str, CaptionProxyRoute]:
    """Parse site=url options; absent sites retain the job default route."""
    result: dict[str, CaptionProxyRoute] = {}
    for mode, entries in (("required", required), ("fallback", fallback)):
        for spec in entries:
            site, sep, url = spec.partition("=")
            site = site.strip().lower()
            if sep != "=" or site not in _SITES or site in result:
                raise ValueError("site proxy must be unique for a supported public provider")
            result[site] = CaptionProxyRoute(mode, url.strip())
    return result


def route_for_site(
    site: str,
    default: CaptionProxyRoute,
    by_site: Mapping[str, CaptionProxyRoute] | None,
) -> CaptionProxyRoute:
    if by_site is None:
        return default
    return by_site.get(site, default)
