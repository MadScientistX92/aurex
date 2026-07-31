"""Filesystem layout and tunables.

Paths resolve relative to the engine package so the CLI behaves the same no matter
which directory it is invoked from.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Final

ENGINE_ROOT: Final[Path] = Path(__file__).resolve().parent.parent
REPO_ROOT: Final[Path] = ENGINE_ROOT.parent

CACHE_DIR: Final[Path] = Path(os.environ.get("AUREX_CACHE_DIR", ENGINE_ROOT / ".cache"))
SCHEDULE_DIR: Final[Path] = Path(__file__).resolve().parent / "data" / "schedules"
PUBLIC_DATA_DIR: Final[Path] = REPO_ROOT / "public-data"

#: Troy ounces to grams. Fixed by definition, not a measurement.
GRAMS_PER_TROY_OUNCE: Final[float] = 31.1034768

#: Identify Aurex honestly to every server we touch, with a contact route.
USER_AGENT: Final[str] = (
    "aurex-research/0.1 (+https://github.com/aurex-engine/aurex; open-source gold research)"
)

#: Per-request timeout, seconds. Nightly job must fail fast rather than hang.
HTTP_TIMEOUT: Final[float] = 20.0

#: Politeness delay between successive requests to the same host, seconds.
HTTP_COURTESY_DELAY: Final[float] = 1.0


def fred_api_key() -> str | None:
    """FRED API key, if the user set one.

    Aurex does not require it — the ``fredgraph.csv`` endpoint serves the same
    observations without authentication. The key is only used when present.
    """
    key = os.environ.get("FRED_API_KEY", "").strip()
    return key or None
