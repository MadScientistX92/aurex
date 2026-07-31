"""FRED loaders.

Uses the ``fredgraph.csv`` endpoint, which serves the same observations as the
authenticated API without a key. ``FRED_API_KEY`` is honoured when set but is never
required — Aurex must run for someone who has just cloned the repo.
"""

from __future__ import annotations

import io
from datetime import date

import pandas as pd

from aurex.data.base import LoadedSeries, build_meta
from aurex.data.sources import http

CSV_ENDPOINT = "https://fred.stlouisfed.org/graph/fredgraph.csv"


class FredLoader:
    """Load one FRED series by its FRED code.

    Args:
        series_id: Aurex's internal series name (the cache key).
        fred_code: The FRED series code, e.g. ``DFII10``.
        column: Output column name.
    """

    def __init__(self, series_id: str, fred_code: str, column: str = "value") -> None:
        self.series_id = series_id
        self.fred_code = fred_code
        self.column = column
        self.source_name = f"FRED:{fred_code}"

    @property
    def url(self) -> str:
        return f"{CSV_ENDPOINT}?id={self.fred_code}"

    def fetch(self, start: date, end: date) -> LoadedSeries:
        response = http.get(self.url, check_robots=False)
        frame = pd.read_csv(io.StringIO(response.text))

        if frame.shape[1] < 2:
            raise ValueError(f"{self.fred_code}: unexpected CSV shape {frame.shape}")

        date_col, value_col = frame.columns[0], frame.columns[1]
        frame = frame.rename(columns={date_col: "date", value_col: self.column})
        frame["date"] = pd.to_datetime(frame["date"])
        # FRED marks missing observations with "." on non-trading days.
        frame[self.column] = pd.to_numeric(frame[self.column], errors="coerce")
        frame = frame.dropna(subset=[self.column]).set_index("date").sort_index()
        frame = frame.loc[str(start) : str(end)]

        if frame.empty:
            raise ValueError(f"{self.fred_code}: no observations in {start}..{end}")

        return LoadedSeries(
            frame=frame[[self.column]],
            meta=build_meta(
                series_id=self.series_id,
                source_name=self.source_name,
                source_url=self.url,
                frame=frame,
            ),
        )
