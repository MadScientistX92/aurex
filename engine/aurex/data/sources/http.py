"""Shared HTTP session: honest identification, timeouts, and courtesy delays.

Aurex scrapes public data from servers nobody is paying to run. Every request
identifies itself, respects a per-host delay, and gives up quickly rather than
holding a connection open.
"""

from __future__ import annotations

import threading
import time
from urllib.parse import urlparse
from urllib.robotparser import RobotFileParser

import requests

from aurex.config import HTTP_COURTESY_DELAY, HTTP_TIMEOUT, USER_AGENT

_last_request_at: dict[str, float] = {}
_lock = threading.Lock()
_robots_cache: dict[str, RobotFileParser | None] = {}


def _throttle(host: str) -> None:
    """Sleep just long enough that we never hit one host faster than the delay."""
    with _lock:
        last = _last_request_at.get(host)
        now = time.monotonic()
        if last is not None:
            wait = HTTP_COURTESY_DELAY - (now - last)
            if wait > 0:
                time.sleep(wait)
                now = time.monotonic()
        _last_request_at[host] = now


def robots_allows(url: str) -> bool:
    """Check ``robots.txt`` for this URL.

    Fails open on an unreachable ``robots.txt`` — a missing file is not a
    prohibition — but fails closed if the file exists and disallows the path.
    """
    parsed = urlparse(url)
    origin = f"{parsed.scheme}://{parsed.netloc}"

    with _lock:
        seen = origin in _robots_cache
        parser = _robots_cache.get(origin)

    if not seen:
        # A cached value of None means "we looked and could not read it", which is
        # why presence in the dict is checked rather than truthiness.
        fetched: RobotFileParser | None = RobotFileParser()
        assert fetched is not None
        fetched.set_url(f"{origin}/robots.txt")
        try:
            fetched.read()
        except Exception:
            # An unreachable or malformed robots.txt is not a prohibition, so fail
            # open. A robots.txt that loads and disallows is honoured below.
            fetched = None
        with _lock:
            _robots_cache[origin] = fetched
        parser = fetched

    if parser is None:
        return True
    return bool(parser.can_fetch(USER_AGENT, url))


def get(url: str, *, check_robots: bool = True, timeout: float = HTTP_TIMEOUT) -> requests.Response:
    """GET ``url`` politely, raising for any non-2xx status.

    Raises:
        PermissionError: ``robots.txt`` disallows this path.
    """
    if check_robots and not robots_allows(url):
        raise PermissionError(f"robots.txt disallows {url}")

    _throttle(urlparse(url).netloc)
    response = requests.get(
        url,
        headers={"User-Agent": USER_AGENT, "Accept-Encoding": "gzip, deflate"},
        timeout=timeout,
    )
    response.raise_for_status()
    return response
