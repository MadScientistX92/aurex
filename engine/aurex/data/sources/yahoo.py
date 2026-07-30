"""Yahoo Finance loader via ``yfinance``.

The spec names this as the preferred source for gold, USD/INR and VIX. It is kept
first in every chain that mentions it — but Yahoo rate-limits aggressively (HTTP 429
throughout this build), so it is never the only source. See
:class:`~aurex.data.chain.SourceChain`.
"""

from __future__ import annotations

from datetime import date, timedelta

import pandas as pd

from aurex.data.base import LoadedSeries, build_meta

_OHLC = ["Open", "High", "Low", "Close"]


class YahooLoader:
    """Load one Yahoo ticker as daily OHLC.

    Args:
        series_id: Aurex's internal series name.
        ticker: Yahoo symbol, e.g. ``GC=F``.
    """

    def __init__(self, series_id: str, ticker: str) -> None:
        self.series_id = series_id
        self.ticker = ticker
        self.source_name = f"yfinance:{ticker}"

    @property
    def url(self) -> str:
        return f"https://finance.yahoo.com/quote/{self.ticker}/history"

    def fetch(self, start: date, end: date) -> LoadedSeries:
        import yfinance as yf

        raw = yf.download(
            self.ticker,
            start=start.isoformat(),
            # yfinance treats `end` as exclusive.
            end=(end + timedelta(days=1)).isoformat(),
            interval="1d",
            auto_adjust=False,
            progress=False,
            threads=False,
        )
        if raw is None or raw.empty:
            raise ValueError(f"{self.ticker}: yfinance returned no rows (rate-limited?)")

        # Recent yfinance returns a (field, ticker) MultiIndex even for one symbol.
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.get_level_values(0)

        missing = [c for c in _OHLC if c not in raw.columns]
        if missing:
            raise ValueError(f"{self.ticker}: missing columns {missing}")

        frame = raw[_OHLC].copy()
        frame.columns = ["open", "high", "low", "close"]
        frame.index = pd.DatetimeIndex(frame.index).tz_localize(None)
        frame.index.name = "date"
        frame = frame[~frame.index.duplicated(keep="last")].sort_index().dropna(how="all")

        if frame.empty:
            raise ValueError(f"{self.ticker}: no usable rows after cleaning")

        return LoadedSeries(
            frame=frame,
            meta=build_meta(
                series_id=self.series_id,
                source_name=self.source_name,
                source_url=self.url,
                frame=frame,
                has_ohlc=True,
            ),
        )
