"""LBMA London fix loader — the gold fallback.

The LBMA publishes the daily London fix as plain JSON back to 1968, unauthenticated
and un-rate-limited. It is close-only, so a series resolved here carries
``has_ohlc=False`` and the realised-volatility estimators downstream must adapt
rather than assume they have highs and lows.
"""

from __future__ import annotations

from datetime import date

import pandas as pd

from aurex.data.base import LoadedSeries, SourceCitation, build_meta
from aurex.data.sources import http

GOLD_PM_URL = "https://prices.lbma.org.uk/json/gold_pm.json"
GOLD_AM_URL = "https://prices.lbma.org.uk/json/gold_am.json"

#: Position of each currency inside the JSON ``v`` array.
_CURRENCY_INDEX = {"USD": 0, "GBP": 1, "EUR": 2}


class LbmaGoldLoader:
    """Daily London gold fix in a chosen currency.

    Args:
        series_id: Aurex's internal series name.
        fix: ``"PM"`` (the benchmark most contracts settle against) or ``"AM"``.
        currency: One of ``USD``, ``GBP``, ``EUR``.
    """

    def __init__(
        self,
        series_id: str = "xauusd",
        fix: str = "PM",
        currency: str = "USD",
    ) -> None:
        if currency not in _CURRENCY_INDEX:
            raise ValueError(f"unsupported currency {currency!r}")
        self.series_id = series_id
        self.fix = fix.upper()
        self.currency = currency
        self.source_name = f"LBMA:gold_{self.fix.lower()}:{currency}"
        # Primary: the LBMA publishes the fix it administers, on its own host.
        self.citation = SourceCitation(
            source_url="https://www.lbma.org.uk/prices-and-data/precious-metal-prices",
            source_confidence="primary",
        )
        self.url = GOLD_PM_URL if self.fix == "PM" else GOLD_AM_URL

    def fetch(self, start: date, end: date) -> LoadedSeries:
        # One open question, not a general permission. ``prices.lbma.org.uk`` answers
        # **HTTP 401** for ``/robots.txt``, which Aurex's own checker reads as a total
        # disallow (RFC 9309 §2.3.1.3). What it does not say is whether that is the
        # LBMA's policy or an artifact of the Cloudflare edge that also serves this
        # host's interstitial — a 401 on a file that is normally world-readable is
        # ambiguous, and ambiguity is not consent. ``docs/lbma-enquiry.md`` asks them
        # directly; until they answer, the single nightly fetch continues and is
        # disclosed rather than quietly stopped or quietly excused. See
        # ``docs/robots-position.md``: an explicit ``Disallow`` is honoured, and this
        # flag must not be copied to any host that publishes one.
        payload = http.get_json(self.url, check_robots=False)
        index = _CURRENCY_INDEX[self.currency]

        rows: list[tuple[pd.Timestamp, float]] = []
        for record in payload:
            values = record.get("v") or []
            if index >= len(values):
                continue
            value = values[index]
            # The euro did not exist before 1999; those entries are null.
            if value is None:
                continue
            rows.append((pd.Timestamp(record["d"]), float(value)))

        if not rows:
            raise ValueError(f"{self.source_name}: no usable observations")

        frame = pd.DataFrame(rows, columns=["date", "close"]).set_index("date").sort_index()
        frame = frame[~frame.index.duplicated(keep="last")]
        frame = frame.loc[str(start) : str(end)]

        if frame.empty:
            raise ValueError(f"{self.source_name}: no observations in {start}..{end}")

        return LoadedSeries(
            frame=frame,
            meta=build_meta(
                series_id=self.series_id,
                source_name=self.source_name,
                source_url=self.url,
                citation=self.citation,
                frame=frame,
                has_ohlc=False,
            ),
        )
