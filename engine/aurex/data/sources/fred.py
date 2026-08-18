"""FRED loaders.

Uses the ``fredgraph.csv`` endpoint, which serves the same observations as the
authenticated API without a key. ``FRED_API_KEY`` is honoured when set but is never
required — Aurex must run for someone who has just cloned the repo.
"""

from __future__ import annotations

import io
from datetime import date

import pandas as pd

from aurex.data.base import LoadedSeries, SourceCitation, build_meta
from aurex.data.sources import http

CSV_ENDPOINT = "https://fred.stlouisfed.org/graph/fredgraph.csv"
SERIES_PAGE = "https://fred.stlouisfed.org/series"


class FredLoader:
    """Load one FRED series by its FRED code.

    Every series reached this way is ``secondary`` by construction and the constant is
    not a parameter: FRED redistributes observations computed by somebody else — the
    Treasury, the EIA, the OECD, an exchange — and which one is stated on the series
    page rather than in the CSV this loader reads. Naming an originator here would be
    asserting a citation from memory, so the citation points at the page that carries
    it instead.

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
        self.citation = SourceCitation(
            source_url=f"{SERIES_PAGE}/{fred_code}",
            source_confidence="secondary",
        )

    @property
    def url(self) -> str:
        return f"{CSV_ENDPOINT}?id={self.fred_code}"

    def fetch(self, start: date, end: date) -> LoadedSeries:
        # The robots guard runs here like anywhere else. It used to be bypassed with
        # ``check_robots=False`` and no reason recorded at the call site — the widest
        # bypass in the codebase, covering five series' primary route and two more on
        # fallback. Measured 2026-08-17 with Aurex's own client:
        # ``fred.stlouisfed.org/robots.txt`` gives ``User-agent: *`` a ``Crawl-delay: 1``
        # and six specific ``Disallow``s, none of which matches ``/graph/fredgraph.csv``
        # — ``/graph/fredgraph.png`` is barred and the CSV beside it is not, which is a
        # line somebody drew deliberately. So the flag was buying nothing, and a bypass
        # that is unnecessary is indistinguishable from the outside from one that is
        # load-bearing.
        response = http.get(self.url)
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
                citation=self.citation,
                frame=frame,
            ),
        )
