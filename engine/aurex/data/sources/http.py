"""Shared HTTP session: honest identification, timeouts, and courtesy delays.

Aurex scrapes public data from servers nobody is paying to run. Every request
identifies itself, respects a per-host delay, and gives up quickly rather than
holding a connection open.
"""

from __future__ import annotations

import threading
import time
from typing import Any, Final
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


def _read_robots(origin: str) -> RobotFileParser | None:
    """Fetch and parse ``origin``'s ``robots.txt`` with *Aurex's* client.

    ``RobotFileParser.read()`` is not used, and that is the whole point of this
    function. It fetches through ``urllib``, which sends ``Python-urllib/3.x`` and none
    of Aurex's headers, so the file it parses is the file the origin serves *that*
    client rather than the one it serves this one. Measured on 2026-08-17: stooq.com
    returns ``User-agent: * / Disallow: /`` to Aurex's client and **404** to urllib's,
    and the parser maps a 404 to "allow everything" — so the check returned True for a
    path the site forbids. A guard that fails open silently is the shape §1 warns
    about, and here it fails open against somebody else's stated wishes.

    Returns the parser to consult, or ``None`` for "there are no rules to apply".
    """
    url = f"{origin}/robots.txt"
    _throttle(urlparse(origin).netloc)
    try:
        response = requests.get(
            url,
            headers={"User-Agent": USER_AGENT, "Accept-Encoding": "gzip, deflate"},
            timeout=HTTP_TIMEOUT,
        )
    except requests.RequestException:
        # Unreachable. Not a prohibition, and unchanged from the previous behaviour.
        return None

    if response.status_code in (401, 403):
        # The file exists and we are not allowed to read it. Read that as a refusal
        # rather than as silence: this is the one 4xx that says something about us
        # rather than about the file. prices.lbma.org.uk answers 401 here, so this is
        # the branch its loader steps around with ``check_robots=False``.
        #
        # Expressed as a synthesised total disallow rather than by setting
        # ``parser.disallow_all``: that attribute exists at runtime but not in typeshed,
        # so ``mypy --strict`` rejects it, and reaching for a private-in-practice
        # attribute to say something robots.txt has its own syntax for is the wrong
        # trade. Same behaviour, public API, and it reads as what it means.
        parser = RobotFileParser()
        parser.set_url(url)
        parser.parse(["User-agent: *", "Disallow: /"])
        return parser

    if not response.ok:
        # Any other 4xx is a missing file, and a 5xx is a broken one. Neither is a
        # prohibition. Also unchanged.
        return None

    parser = RobotFileParser()
    parser.set_url(url)
    # `errors="replace"` because a malformed byte in somebody's robots.txt must not
    # decide whether Aurex fetches: an exception here would land in the caller's
    # "unreachable, so allow" branch, which is the wrong direction to guess in.
    parser.parse(response.content.decode("utf-8", "replace").splitlines())
    return parser


def robots_allows(url: str) -> bool:
    """Check ``robots.txt`` for this URL, reading it as Aurex rather than as urllib.

    Fails open on an unreachable or absent ``robots.txt`` — a missing file is not a
    prohibition — but fails closed if the file exists and disallows the path, and on a
    401 or 403, which says the file exists and is not ours to read.
    """
    parsed = urlparse(url)
    origin = f"{parsed.scheme}://{parsed.netloc}"

    with _lock:
        seen = origin in _robots_cache
        parser = _robots_cache.get(origin)

    if not seen:
        # A cached value of None means "we looked and there was nothing to apply",
        # which is why presence in the dict is checked rather than truthiness.
        parser = _read_robots(origin)
        with _lock:
            _robots_cache[origin] = parser

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


#: Response headers worth carrying into a failure message. The Cloudflare three are
#: here because the failure that motivated this function happens behind Cloudflare and
#: ``cf-mitigated`` is what a challenge sets; the rest identify the responder at all.
_DIAGNOSTIC_HEADERS: Final[tuple[str, ...]] = (
    "content-type",
    "content-length",
    "server",
    "cf-ray",
    "cf-mitigated",
    "cf-cache-status",
)

#: Bytes of body to quote back. Enough to tell an HTML challenge page from an empty
#: body or a truncated write, short enough to sit in a skip record.
_BODY_EXCERPT_BYTES: Final[int] = 200


class NonJsonResponseError(ValueError):
    """A 2xx response whose body would not decode as JSON.

    Carries what the body *was* — status, the headers that identify the responder, the
    byte count and the leading bytes — because the bare ``JSONDecodeError`` does not.
    "Expecting value: line 1 column 1 (char 0)" is emitted identically by an empty body
    and by an HTML challenge page, and those two need opposite fixes: one is a
    transient worth retrying, the other means the host will not serve this client at
    all and a different host is required. Four public skip records were written without
    that distinction being recoverable from any of them.
    """


def describe_response(response: requests.Response) -> str:
    """One line identifying what came back, safe to put in a log or an artifact.

    The body excerpt is ``repr``'d rather than interpolated: it is attacker-adjacent
    text of unknown encoding heading for a JSON file, and a raw newline or quote in the
    middle of a skip record is a second problem on top of the one being recorded.
    """
    body = response.content
    parts = [f"HTTP {response.status_code}", f"{len(body)} bytes"]
    parts += [
        f"{name}={response.headers[name]}"
        for name in _DIAGNOSTIC_HEADERS
        if name in response.headers
    ]
    parts.append(f"body starts {body[:_BODY_EXCERPT_BYTES]!r}")
    return "; ".join(parts)


def get_json(url: str, *, check_robots: bool = True, timeout: float = HTTP_TIMEOUT) -> Any:
    """GET ``url`` and decode it as JSON, describing the body when that fails.

    Raises:
        PermissionError: ``robots.txt`` disallows this path.
        NonJsonResponseError: The response was 2xx but did not decode as JSON.
    """
    response = get(url, check_robots=check_robots, timeout=timeout)
    try:
        return response.json()
    except ValueError as exc:
        raise NonJsonResponseError(f"{exc} — {describe_response(response)}") from exc
