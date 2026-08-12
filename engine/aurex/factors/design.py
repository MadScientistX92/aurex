"""Weekly design matrix, built from what an asset declares and nothing else.

This module knows that a driver has a series, a column, an aggregation, a transform
and a lag. It does not know what any of them are, and it must not: the guard in
``tests/test_asset_abstraction.py`` fails the build if a driver's name appears here.

**Weekly, anchored on Friday.** Daily returns are dominated by noise the macro series
cannot resolve, and monthly throws away most of the sample. Friday because the series
involved are business-daily and a Friday anchor makes "the last observation of the
week" the natural reading for almost every week of the sample.

**Alignment is where attribution quietly becomes forecasting, or quietly becomes
cheating.** Two rules, both enforced here rather than left to a caller:

- A driver enters at its declared lag. Zero is contemporaneous, which is what
  attribution means: how much of the move that *happened* do these drivers account for.
- A driver built from the asset's own price series must declare a positive lag. At zero
  it would be the target under another name, and the fit would report an R-squared near
  one that means nothing at all. :func:`build` refuses rather than fitting it.

**A required driver that cannot be built is fatal.** Optional drivers drop out with a
reason. Required ones do not, because the whole reason a driver is marked required is
that the fit without it is not a smaller version of the fit with it — it is a different
and possibly sign-inverted answer, arrived at honestly.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from aurex.assets.base import Asset, FactorSpec
from aurex.data.base import price_column

#: Weekly anchor. Friday, and stated once rather than threaded through as a parameter,
#: because two parts of this package resampling on different anchors would produce
#: loadings and a chain that cannot be read against each other.
WEEK_RULE = "W-FRI"

#: How much of the required-driver sample an optional driver must leave standing to be
#: admitted. Set at four fifths: an optional driver is worth some sample, since it is in
#: the set because somebody thought it mattered, but a driver that costs a fifth of the
#: history is buying its own inclusion with the evidence behind every other loading.
MIN_OPTIONAL_COVERAGE = 0.8

VALID_TRANSFORMS = frozenset({"diff", "pct_change", "level"})
VALID_AGGREGATIONS = frozenset({"last", "mean"})


class DesignError(ValueError):
    """The design matrix cannot be built as declared."""


@dataclass(frozen=True, slots=True)
class WeeklyDesign:
    """Aligned weekly target and regressors, with the reason for anything missing."""

    #: Weekly returns of the asset's price series, in the asset's own transform space.
    target: pd.Series
    #: One column per driver that made it in, in declaration order.
    frame: pd.DataFrame
    #: Driver id -> why it is not a column here.
    dropped: dict[str, str]
    #: Drivers that made it in, in column order.
    specs: tuple[FactorSpec, ...]
    #: Whichever kept driver's own history starts latest, and therefore decides where
    #: the sample begins. ``None`` for an empty design.
    binding_factor: str | None = None

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(self.frame.columns)

    def matrix(self) -> np.ndarray:
        out: np.ndarray = self.frame.to_numpy(dtype=float)
        return out

    def values(self) -> np.ndarray:
        out: np.ndarray = self.target.to_numpy(dtype=float)
        return out

    def describe(self) -> dict[str, Any]:
        return {
            "frequency": WEEK_RULE,
            "observations": len(self.frame),
            "first_week": self.frame.index[0].date().isoformat() if len(self.frame) else None,
            "last_week": self.frame.index[-1].date().isoformat() if len(self.frame) else None,
            "regressors": [
                {
                    "id": spec.id,
                    "series_id": spec.series_id,
                    "column": spec.column,
                    "transform": spec.transform,
                    "aggregation": spec.aggregation,
                    "lag_weeks": spec.lag,
                    "description": spec.description,
                }
                for spec in self.specs
            ],
            "dropped": dict(self.dropped),
            "sample_binding_factor": self.binding_factor,
            "alignment": (
                "Weekly observations anchored on Friday. A regressor at lag zero is "
                "contemporaneous with the return it helps account for, which is what "
                "attribution means and is not a forecast. Rows are kept only where the "
                "target and every regressor are observed, so the sample is the "
                "intersection rather than a padded union."
            ),
        }


def _column_of(frame: pd.DataFrame, spec: FactorSpec) -> pd.Series:
    if spec.column is not None:
        if spec.column not in frame.columns:
            raise DesignError(
                f"{spec.id}: series {spec.series_id!r} has no column {spec.column!r}; "
                f"it carries {sorted(frame.columns)}"
            )
        return frame[spec.column]
    return price_column(frame)


def to_weekly(daily: pd.Series, *, aggregation: str) -> pd.Series:
    """Collapse a dated series to one observation per week."""
    if aggregation not in VALID_AGGREGATIONS:
        raise DesignError(
            f"unknown aggregation {aggregation!r}; expected one of {sorted(VALID_AGGREGATIONS)}"
        )
    grouped = daily.dropna().resample(WEEK_RULE)
    weekly = grouped.last() if aggregation == "last" else grouped.mean()
    return weekly.dropna()


def apply_transform(weekly: pd.Series, *, transform: str) -> pd.Series:
    """Turn a weekly level into the regressor the driver declares."""
    if transform not in VALID_TRANSFORMS:
        raise DesignError(
            f"unknown transform {transform!r}; expected one of {sorted(VALID_TRANSFORMS)}"
        )
    if transform == "level":
        return weekly
    if transform == "diff":
        return weekly.diff()
    # A proportional change is undefined through zero and explodes near it, which is not
    # hypothetical for a series that can be negative. Rows where the base is zero become
    # NaN and drop out with the rest, rather than becoming an infinity that survives
    # standardisation as the largest observation in the sample.
    base = weekly.shift(1)
    changed = (weekly - base) / base.where(base != 0.0)
    return changed.replace([np.inf, -np.inf], np.nan)


def build(
    asset: Asset,
    frames: dict[str, pd.DataFrame],
    *,
    unavailable: dict[str, str] | None = None,
) -> WeeklyDesign:
    """Assemble the weekly design for ``asset`` from resolved series.

    Args:
        asset: Supplies the driver declarations and the return transform.
        frames: Resolved series by id, including the asset's own price series.
        unavailable: Series id -> why it did not resolve, for the dropped reasons.
    """
    reasons = dict(unavailable or {})

    price = frames.get(asset.price_series_id)
    if price is None:
        raise DesignError(
            f"cannot build a design without the price series {asset.price_series_id!r}: "
            f"{reasons.get(asset.price_series_id, 'it did not resolve')}"
        )

    weekly_price = to_weekly(price_column(price), aggregation="last")
    target = asset.return_transform.to_returns(weekly_price)

    columns: dict[str, pd.Series] = {}
    specs: list[FactorSpec] = []
    dropped: dict[str, str] = {}

    for spec in asset.factor_set:
        if spec.series_id == asset.price_series_id and spec.lag < 1:
            raise DesignError(
                f"{spec.id}: a driver built from the asset's own price series must "
                f"declare a lag of at least one week. At lag zero it is the target "
                f"under another name, and the fit would report an R-squared near one."
            )

        frame = frames.get(spec.series_id)
        if frame is None:
            reason = reasons.get(spec.series_id, "the series did not resolve")
            if spec.required:
                raise DesignError(
                    f"{spec.id} is a required driver and {spec.series_id!r} is "
                    f"unavailable ({reason}). Fitting without it would not be a smaller "
                    f"answer than fitting with it; it would be a different one."
                )
            dropped[spec.id] = reason
            continue

        try:
            raw = _column_of(frame, spec)
        except (DesignError, ValueError) as exc:
            if spec.required:
                raise
            dropped[spec.id] = str(exc)
            continue

        weekly = apply_transform(
            to_weekly(raw, aggregation=spec.aggregation), transform=spec.transform
        )
        if spec.lag:
            weekly = weekly.shift(spec.lag)

        usable = int(weekly.notna().sum())
        if usable == 0:
            reason = "the series resolved but carries no usable weekly observation"
            if spec.required:
                raise DesignError(f"{spec.id} is required and {reason}")
            dropped[spec.id] = reason
            continue

        columns[spec.id] = weekly
        specs.append(spec)

    if not columns:
        raise DesignError("no driver could be built; there is nothing to attribute to")

    return _join(target, columns, specs, dropped)


def _rows(target: pd.Series, columns: dict[str, pd.Series]) -> int:
    return len(pd.concat([target, *columns.values()], axis=1, sort=False).dropna())


def _join(
    target: pd.Series,
    columns: dict[str, pd.Series],
    specs: list[FactorSpec],
    dropped: dict[str, str],
) -> WeeklyDesign:
    """Intersect the target with the drivers, refusing to let an optional one gut it.

    Every row must be observed on every column, so a driver with three weeks of history
    does not shorten the sample — it *replaces* it. That is not hypothetical: a flow
    proxy scraped from a daily report accumulates a few days per fetch, and on a fresh
    clone it carries three weeks against the target's twenty years. Joining naively took
    a design of 1,080 weeks down to 2, and every downstream number was still computed,
    still formatted, and still wrong.

    So a required driver may bind the sample — that is what required means, and which
    driver binds it is reported — while an optional one is admitted only if it leaves
    most of the sample standing, and dropped with both counts named if it does not.
    """
    required = {spec.id: columns[spec.id] for spec in specs if spec.required}
    baseline = _rows(target, required) if required else len(target.dropna())

    kept: dict[str, pd.Series] = dict(required)
    kept_specs: list[FactorSpec] = [spec for spec in specs if spec.required]

    for spec in specs:
        if spec.required:
            continue
        candidate = dict(kept)
        candidate[spec.id] = columns[spec.id]
        rows = _rows(target, candidate)
        if baseline and rows < MIN_OPTIONAL_COVERAGE * baseline:
            dropped[spec.id] = (
                f"including it would cut the sample from {baseline} weeks to {rows}, "
                f"below the {MIN_OPTIONAL_COVERAGE:.0%} of the required-driver sample an "
                f"optional driver has to leave standing. Its own history is the "
                f"constraint, not the driver's usefulness."
            )
            continue
        kept, kept_specs = candidate, [*kept_specs, spec]

    ordered = [spec for spec in specs if spec.id in kept]
    joined = pd.concat(
        [target.rename("__target__"), pd.DataFrame({s.id: kept[s.id] for s in ordered})],
        axis=1,
        sort=False,
    ).dropna()

    # Which driver's own history decides where the sample starts. Reported rather than
    # left to be inferred from a start date, because "the sample begins in 2006" and
    # "one regressor begins in 2006" are the same fact and only one of them is visible.
    binding = None
    if len(joined):
        starts = {
            spec.id: kept[spec.id].dropna().index[0]
            for spec in ordered
            if kept[spec.id].notna().any()
        }
        if starts:
            binding = max(starts, key=lambda name: starts[name])

    return WeeklyDesign(
        target=joined["__target__"],
        frame=joined.drop(columns="__target__"),
        dropped=dropped,
        specs=tuple(ordered),
        binding_factor=binding,
    )
