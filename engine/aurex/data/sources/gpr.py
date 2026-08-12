"""Caldara-Iacoviello geopolitical risk index — the daily series.

This is the factor the driver set has declared and gone without since it was written,
and going without it is not a neutral omission. Every remaining regressor reaches
escalation through a channel that pushes the metal *down*: crude rises, expected
inflation rises, policy turns hawkish, the real yield rises, and a loading fitted on
real yields alone will duly report that escalation is bearish. The safe-haven bid that
actually moves the price on the day of an escalation has no regressor to load on, so it
is absorbed into the residual — and the wrong sign arrives attached to honestly
estimated coefficients and a clean causal story, which is what makes it dangerous. It
passes the hand-typed-view check because nothing was hand-typed.

**The series.** The authors publish the index themselves, monthly and daily, free and
unauthenticated, under CC BY. The daily file is the one wired here: attribution runs on
weekly returns, and a monthly index cannot say which week of October the escalation
landed in. The published monthly index is *not* the mean of these daily values — it is
computed from monthly article shares — so anything in this repository that needs a
monthly figure aggregates the daily series and says that it did, rather than implying
the authors' monthly number.

**Format.** A legacy ``.xls`` workbook, which is why ``xlrd`` is a dependency. The sheet
carries its own data dictionary in two trailing columns (``var_name``, ``var_label``)
alongside the observations; those are dropped, and the columns actually read are named
explicitly so a reshuffled workbook fails loudly instead of silently loading whatever
sat in position three.
"""

from __future__ import annotations

import io
from datetime import date

import pandas as pd

from aurex.data.base import LoadedSeries, SourceCitation, build_meta
from aurex.data.sources import http

#: The authors' page. Their own request is that users cite both the paper and the site.
INDEX_PAGE = "https://www.matteoiacoviello.com/gpr.htm"
DAILY_XLS_URL = "https://www.matteoiacoviello.com/gpr_files/data_gpr_daily_recent.xls"

CITE_AS = (
    "Caldara, Dario and Matteo Iacoviello (2022), 'Measuring Geopolitical Risk', "
    "American Economic Review, April, 112(4), pp. 1194-1225. Data downloaded from "
    "https://www.matteoiacoviello.com/gpr.htm"
)

#: Stated on the index page: all material there is open access under CC BY.
LICENCE = "CC BY 4.0"

#: Published column -> the name Aurex serves it under. The threats/acts split is loaded
#: beside the headline index because they are the two halves of the escalation story a
#: scenario axis would otherwise have to assert: threats are risk priced in advance,
#: acts are the event itself.
COLUMNS: dict[str, str] = {
    "GPRD": "gpr",
    "GPRD_THREAT": "gpr_threats",
    "GPRD_ACT": "gpr_acts",
}

#: The index is a share-of-articles measure normalised to 100 over 1985-2019, so it is
#: dimensionless and its base period is a fact about the series rather than a choice
#: Aurex made. Recorded here because a reader seeing "154" needs to know 100 is average.
BASE_PERIOD = "1985-2019 = 100"


def parse_daily(raw: pd.DataFrame) -> pd.DataFrame:
    """The published sheet as a dated frame of the three series Aurex reads.

    Separate from the fetch so the parse is testable without either the network or a
    binary fixture, and so a workbook that changes shape produces a message naming the
    column that went missing.
    """
    missing = [name for name in ("date", *COLUMNS) if name not in raw.columns]
    if missing:
        raise ValueError(f"GPR daily sheet is missing columns {missing}")

    frame = raw[["date", *COLUMNS]].rename(columns=COLUMNS)
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    for column in COLUMNS.values():
        frame[column] = pd.to_numeric(frame[column], errors="coerce")

    # The data dictionary sits in trailing *columns* beside the first dozen observations
    # rather than in rows of its own, so selecting by name is all it takes to drop it.
    # The dropna is for the export's own trailing blanks and for any row whose date did
    # not parse — a row Aurex cannot date is a row it cannot align to a return.
    frame = frame.dropna(subset=["date", "gpr"]).set_index("date").sort_index()
    frame = frame[~frame.index.duplicated(keep="last")]
    frame.index.name = "date"
    return frame


class GprDailyLoader:
    """Daily geopolitical risk, from the authors' own workbook.

    Args:
        series_id: Aurex's internal series name (the cache key).
    """

    source_name = "Caldara-Iacoviello:GPRD"

    #: Primary: the file is published by the authors who construct the index, on their
    #: own site, and they state how it should be cited. That statement is why
    #: ``cite_as`` is filled here and empty on the redistributed series.
    citation = SourceCitation(
        source_url=INDEX_PAGE,
        source_confidence="primary",
        cite_as=CITE_AS,
        licence=LICENCE,
    )

    def __init__(self, series_id: str = "gpr") -> None:
        self.series_id = series_id

    @property
    def url(self) -> str:
        return DAILY_XLS_URL

    def fetch(self, start: date, end: date) -> LoadedSeries:
        response = http.get(self.url)
        raw = pd.read_excel(io.BytesIO(response.content), engine="xlrd")

        frame = parse_daily(raw).loc[str(start) : str(end)]
        if frame.empty:
            raise ValueError(f"{self.source_name}: no observations in {start}..{end}")

        return LoadedSeries(
            frame=frame,
            meta=build_meta(
                series_id=self.series_id,
                source_name=self.source_name,
                # The workbook, not the landing page: this is the URL that was called,
                # and the citation above is where the series is published.
                source_url=self.url,
                citation=self.citation,
                frame=frame,
            ),
        )
