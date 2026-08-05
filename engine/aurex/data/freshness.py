"""Publication freshness, and the refusal it licenses.

A nightly runner is a fresh machine with no cache and no memory of last night. Yahoo
rate-limits, IBJA does not publish on Indian holidays, and
:class:`~aurex.data.chain.SourceChain` is *built* to fall back rather than fail —
which is the right behaviour for a human at a terminal and the wrong one for an
unattended job at 02:00 UTC. Those three combine into a run that quietly resolves
week-old prices, simulates from them, and commits a forecast dated today.

That is the one error this repository cannot tolerate, and it is worth being precise
about why. A night the job skipped is a *visible* hole: the index in
:mod:`aurex.record` lists the date and says what happened. A night the job fabricated
is not visible at all — it has the same shape, the same fields and the same timestamp
as a real one, and nothing downstream can tell them apart. Every claim in §0 rests on
the dated log being what the engine actually said on the day it is dated, so a
fabricated night does not merely add a bad forecast; it removes the reason to believe
any of the others.

So each series declares how far behind the run date it is allowed to fall, and why
that number rather than another. Two rules follow, and both fail closed:

- Where a **blocking** series exceeds its tolerance, the run publishes nothing and
  exits non-zero. Blocking means the price series and the exchange rate of any lens
  that would be published — the inputs a fabricated price would come from.
- Where a blocking series has **no declared tolerance**, the run refuses on the same
  terms. An undeclared policy is not a permissive one, and a series added without one
  must fail in CI rather than pass silently at 02:00.

Non-blocking series are measured and reported but never block. A stale factor changes
a fitted loading, which is a different and lesser problem than a fabricated price, and
`dxy` runs a week behind by FRED's own publication schedule in the ordinary case.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import date
from typing import Any

import pandas as pd

from aurex.data.base import LoadedSeries, price_column


class StaleDataError(RuntimeError):
    """A blocking series does not reach the run date within its declared tolerance.

    Raised in preference to returning a flag because the caller that ignores a flag
    publishes the fabrication this module exists to prevent.
    """


@dataclass(frozen=True, slots=True)
class SeriesFreshness:
    """How stale one series may be before a forecast built on it is a fabrication.

    The rationale is not decoration. These tolerances are judgements about publication
    calendars rather than vendor guarantees, and a reader deciding whether to trust a
    published forecast needs to see the reasoning that licensed it, not just the
    number that passed.
    """

    #: Maximum calendar days the last observation may sit behind the run date.
    max_lag_days: int
    #: The publication rhythm the number was derived from.
    calendar: str
    #: Why this tolerance and not a tighter or looser one.
    rationale: str

    def __post_init__(self) -> None:
        if self.max_lag_days < 0:
            raise ValueError(f"max_lag_days must be non-negative, got {self.max_lag_days}")
        if not self.calendar.strip() or not self.rationale.strip():
            raise ValueError("a freshness tolerance must declare both calendar and rationale")

    def describe(self) -> dict[str, Any]:
        return {
            "max_lag_days": self.max_lag_days,
            "calendar": self.calendar,
            "rationale": self.rationale,
        }


#: Verdicts a single series can carry. Only ``fresh`` permits publication of a
#: blocking series; the rest are distinguished because they need different fixes.
FRESH = "fresh"
STALE = "stale"
UNDECLARED = "undeclared"
UNAVAILABLE = "unavailable"
EMPTY = "empty"


@dataclass(frozen=True, slots=True)
class SeriesAge:
    """One series, how far behind it is, and whether that is permitted."""

    series_id: str
    blocking: bool
    verdict: str
    last_observation: date | None
    lag_days: int | None
    tolerance: SeriesFreshness | None
    detail: str

    @property
    def blocks_publication(self) -> bool:
        return self.blocking and self.verdict != FRESH

    def describe(self) -> dict[str, Any]:
        return {
            "series_id": self.series_id,
            "blocking": self.blocking,
            "verdict": self.verdict,
            "last_observation": (
                self.last_observation.isoformat() if self.last_observation else None
            ),
            "lag_days": self.lag_days,
            "tolerance": self.tolerance.describe() if self.tolerance else None,
            "detail": self.detail,
        }


@dataclass(frozen=True, slots=True)
class FreshnessVerdict:
    """Every series measured against its tolerance, and the publish/refuse decision."""

    run_date: date
    ages: tuple[SeriesAge, ...]

    @property
    def failures(self) -> tuple[SeriesAge, ...]:
        return tuple(age for age in self.ages if age.blocks_publication)

    @property
    def publishable(self) -> bool:
        return not self.failures

    def raise_if_stale(self) -> None:
        """Refuse loudly. The nightly job's only correct response to a failure."""
        if self.publishable:
            return
        lines = [
            f"refusing to publish a forecast dated {self.run_date.isoformat()}: "
            f"{len(self.failures)} blocking series did not reach it"
        ]
        lines += [f"  - {age.detail}" for age in self.failures]
        lines.append(
            "  A forecast dated today, built on prices from before today, is "
            "fabrication with a timestamp on it. A skipped night is a visible hole in "
            "the track record; this would not be visible at all."
        )
        raise StaleDataError("\n".join(lines))

    def describe(self) -> dict[str, Any]:
        """The block the artifact carries, so a published forecast cites its tolerance."""
        return {
            "run_date": self.run_date.isoformat(),
            "publishable": self.publishable,
            "policy": (
                "A blocking series — the price series, and the exchange rate of any "
                "published lens — must reach the run date within its declared "
                "tolerance or the run publishes nothing and exits non-zero. Tolerances "
                "are calendar days and are stated per series with the publication "
                "calendar they were derived from. Non-blocking series are measured and "
                "reported here but never block: a stale factor moves a fitted loading, "
                "which is a smaller problem than a fabricated price."
            ),
            "series": [age.describe() for age in self.ages],
        }


def _last_observation(frame: pd.DataFrame, *, strict: bool) -> date | None:
    """The last date this series actually carries a number for.

    ``strict`` measures the price column specifically, which is what a blocking series
    is judged on: an index reaching today over a NaN close is not an observation, and
    it is exactly the shape a partial fetch leaves behind. The generous reading — any
    non-null column — is used for reporting on series that do not block.
    """
    if frame.empty:
        return None
    if strict:
        try:
            column = price_column(frame)
        except ValueError:
            return None
        index = column.dropna().index
    else:
        index = frame.dropna(how="all").index
    if len(index) == 0:
        return None
    return pd.Timestamp(index[-1]).date()


def assess(
    *,
    series: Mapping[str, LoadedSeries],
    unavailable: Mapping[str, str],
    policies: Mapping[str, SeriesFreshness | None],
    blocking: frozenset[str],
    run_date: date,
) -> FreshnessVerdict:
    """Measure every resolved series against its declared tolerance.

    ``policies`` covers every series the run attempted, resolved or not; a blocking
    series missing from it is reported ``undeclared`` and refuses, because a tolerance
    nobody wrote down is not a tolerance anybody agreed to.
    """
    ages: list[SeriesAge] = []

    for series_id in sorted(set(policies) | set(series) | set(unavailable)):
        blocks = series_id in blocking
        tolerance = policies.get(series_id)
        loaded = series.get(series_id)

        if loaded is None:
            reason = unavailable.get(series_id, "no source answered and nothing is cached")
            ages.append(
                SeriesAge(
                    series_id=series_id,
                    blocking=blocks,
                    verdict=UNAVAILABLE,
                    last_observation=None,
                    lag_days=None,
                    tolerance=tolerance,
                    detail=f"{series_id}: unavailable — {reason}",
                )
            )
            continue

        last = _last_observation(loaded.frame, strict=blocks)
        if last is None:
            ages.append(
                SeriesAge(
                    series_id=series_id,
                    blocking=blocks,
                    verdict=EMPTY,
                    last_observation=None,
                    lag_days=None,
                    tolerance=tolerance,
                    detail=f"{series_id}: resolved but carries no observation",
                )
            )
            continue

        lag = (run_date - last).days

        if tolerance is None:
            ages.append(
                SeriesAge(
                    series_id=series_id,
                    blocking=blocks,
                    verdict=UNDECLARED,
                    last_observation=last,
                    lag_days=lag,
                    tolerance=None,
                    detail=(
                        f"{series_id}: no staleness tolerance is declared for this "
                        f"series, so it cannot be judged fresh. Declare one where its "
                        f"loaders are declared."
                    ),
                )
            )
            continue

        stale = lag > tolerance.max_lag_days
        ages.append(
            SeriesAge(
                series_id=series_id,
                blocking=blocks,
                verdict=STALE if stale else FRESH,
                last_observation=last,
                lag_days=lag,
                tolerance=tolerance,
                detail=(
                    f"{series_id}: last observation {last.isoformat()} is {lag} days "
                    f"behind the run date, tolerance is {tolerance.max_lag_days} "
                    f"({tolerance.calendar})"
                ),
            )
        )

    return FreshnessVerdict(run_date=run_date, ages=tuple(ages))
