"""Loader protocol and provenance types.

Every series in Aurex arrives through a :class:`Loader` and carries its provenance
with it. Nothing downstream is permitted to see a number without also being able to
see which source answered, when, and at what URL.

A URL alone stopped being enough once a series arrived that a reader might mistake for
a modelling choice. The dated schedules have carried ``source_url`` *and*
``source_confidence`` per entry since the duty table existed, on the reasoning that a
citation and a link are different claims. Series were the last external facts in this
repository not held to that rule, so :class:`SourceCitation` applies it here, reusing
the schedules' own ``VALID_CONFIDENCE`` rather than declaring a second vocabulary that
could drift from it.

It is a member of the :class:`Loader` protocol rather than something read back with
``getattr``, because the failure mode is a series that publishes with no confidence
recorded and nothing anywhere saying so.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import date, datetime
from typing import Any, Protocol, runtime_checkable

import pandas as pd

from aurex.data.schedules.provenance import VALID_CONFIDENCE, Confidence


class DataUnavailableError(RuntimeError):
    """Every source in a chain failed and no cached copy exists."""


@dataclass(frozen=True, slots=True)
class SourceCitation:
    """Who published a series, and how directly this loader reaches them.

    ``primary`` is the publisher's own file — the LBMA's fix, the authors' own index.
    ``secondary`` is a redistributor: FRED serves observations originated by the
    Treasury, the EIA and the OECD, and Yahoo serves exchange data it did not compute.
    Neither label is a quality judgement. It records how many hands the number passed
    through, which is the thing a reader cannot recover from a URL.

    ``cite_as`` is filled only where the publisher states how they wish to be cited.
    An empty one is not an oversight to be papered over with a plausible-looking
    reference: a citation nobody supplied is exactly the kind of claim this repository
    refuses to imply.
    """

    #: Where the series is published, which is not always the URL a loader called.
    source_url: str
    source_confidence: Confidence
    cite_as: str | None = None
    licence: str | None = None

    def __post_init__(self) -> None:
        if not self.source_url:
            raise ValueError("a citation without a source_url cites nothing")
        if self.source_confidence not in VALID_CONFIDENCE:
            raise ValueError(
                f"source_confidence {self.source_confidence!r} not in {sorted(VALID_CONFIDENCE)}"
            )

    def describe(self) -> dict[str, Any]:
        return {
            "source_url": self.source_url,
            "source_confidence": self.source_confidence,
            "cite_as": self.cite_as,
            "licence": self.licence,
        }


@dataclass(frozen=True, slots=True)
class SeriesMeta:
    """Provenance for one loaded series.

    ``source_name`` is the source that actually answered, which is not necessarily
    the preferred one — see :class:`~aurex.data.chain.SourceChain`. ``source_url`` is
    the URL that call actually used; ``citation`` is where the series is published and
    how directly this route reaches it, which for a dated report are not the same URL.
    """

    series_id: str
    source_name: str
    source_url: str
    citation: SourceCitation
    fetched_at: datetime
    rows: int
    start: date | None
    end: date | None
    #: True when the source supplied OHLC rather than close-only. Realised-volatility
    #: estimators need to know: a close-only fallback silently degrades them.
    has_ohlc: bool = False
    #: Sources tried before this one, with why each declined. Empty when the
    #: preferred source answered first.
    fallbacks: tuple[str, ...] = field(default_factory=tuple)

    def to_dict(self) -> dict[str, Any]:
        """JSON-ready form for the published artifact."""
        out: dict[str, Any] = asdict(self)
        out["citation"] = self.citation.describe()
        out["fetched_at"] = self.fetched_at.isoformat()
        out["start"] = self.start.isoformat() if self.start else None
        out["end"] = self.end.isoformat() if self.end else None
        out["fallbacks"] = list(self.fallbacks)
        return out


@dataclass(frozen=True, slots=True)
class LoadedSeries:
    """A dataframe and the provenance that justifies it."""

    frame: pd.DataFrame
    meta: SeriesMeta

    def __post_init__(self) -> None:
        if not isinstance(self.frame.index, pd.DatetimeIndex):
            raise TypeError(f"{self.meta.series_id}: index must be a DatetimeIndex")
        if not self.frame.index.is_monotonic_increasing:
            raise ValueError(f"{self.meta.series_id}: index must be sorted ascending")
        if self.frame.index.has_duplicates:
            raise ValueError(f"{self.meta.series_id}: index has duplicate dates")


@runtime_checkable
class Loader(Protocol):
    """One way of obtaining one series.

    Implementations raise any exception on failure; :class:`SourceChain` catches it
    and moves to the next source, recording the reason.
    """

    #: Stable identifier, also the cache filename stem.
    series_id: str
    #: Human-readable source label recorded in the artifact.
    source_name: str
    #: Where this route's data is published, and how directly it reaches the publisher.
    #: A protocol member so a loader that forgets one is rejected at the boundary
    #: rather than serving numbers with no recorded confidence behind them.
    citation: SourceCitation

    def fetch(self, start: date, end: date) -> LoadedSeries:
        """Retrieve ``[start, end]`` inclusive. Network access is expected here."""
        ...


def price_column(frame: pd.DataFrame) -> pd.Series:
    """Pick the column carrying the price, tolerating OHLC and close-only shapes.

    Shared rather than private to the pipeline because the freshness guard has to
    measure staleness on the *same* column the pipeline will price from. Two
    definitions of "the price" would let a series pass the guard on one column and
    publish from another.
    """
    for candidate in ("close", "value"):
        if candidate in frame.columns:
            return frame[candidate]
    numeric = frame.select_dtypes("number")
    if numeric.empty:
        raise ValueError(f"no numeric column in {list(frame.columns)}")
    return numeric.iloc[:, 0]


def build_meta(
    *,
    series_id: str,
    source_name: str,
    source_url: str,
    citation: SourceCitation,
    frame: pd.DataFrame,
    has_ohlc: bool = False,
    fetched_at: datetime | None = None,
) -> SeriesMeta:
    """Derive :class:`SeriesMeta` from a freshly-fetched frame.

    ``citation`` is required rather than defaulted. A default here would be a
    table-level default by another name, and the whole point of the rule is that one
    verified citation must never end up standing behind an unverified series.
    """
    from datetime import UTC

    index = frame.index
    return SeriesMeta(
        series_id=series_id,
        source_name=source_name,
        source_url=source_url,
        citation=citation,
        fetched_at=fetched_at or datetime.now(UTC),
        rows=len(frame),
        start=index[0].date() if len(frame) else None,
        end=index[-1].date() if len(frame) else None,
        has_ohlc=has_ohlc,
    )
