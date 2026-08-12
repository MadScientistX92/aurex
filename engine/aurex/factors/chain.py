"""Composing a declared chain, and refusing to let it be added to anything.

Three things are estimated here and the relationship between them is the whole point.

**Link by link.** Each arrow gets its own local projection with its own band. The bands
are per-horizon and no identifying assumption is imposed, so a link with nothing in it
says so rather than inheriting significance from the link beside it.

**Compounded, with a band that keeps the links' dependence.** The chain-implied response
is the product of the per-link responses, and the band around that product is *not* the
product of the bands. Multiplying three intervals treats the links as independent, and
they are not — they are estimated from overlapping windows of the same economy, so a
month that makes the first link steep tends to make the second link steep too. The band
comes from a moving-block bootstrap that resamples *months* once per replicate and
re-estimates every link on the same resampled months, so whatever dependence the sample
has survives into the product.

**Directly, and again with the overlapping channel held constant.** The same shock is
already a driver in the weekly factor set, so the direct estimate and the chain measure
overlapping things through different amounts of noise. Two rules follow and both are
enforced here rather than left to a reader:

- They are published as *alternative decompositions*, never summed. There is no field in
  this module's output that adds a chain response to a factor loading, and there is no
  total for a reader to mistake for one.
- The direct estimate is published twice: raw, and with the quote-currency price of the
  asset held constant. The controlled one is the part of the path that runs through the
  local economy — the tax stack, the exchange rate, the domestic premium — with the
  global price channel the factor loadings already carry taken out. That is the explicit
  orthogonalisation, and it is what makes the comparison with the chain meaningful.

**The last arrow is not estimated.** A chain that ends at a price expressed through a
currency lens ends in an identity: the lens computes the local price from the quote and
the rate by arithmetic, so the elasticity of the local price to the rate is one by
construction. Estimating it would produce a number near one with a band around it and
invite a reader to treat sampling noise in an identity as evidence about the world. It
is asserted, and measured only as a diagnostic that the lens is doing what it claims.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Any

import numpy as np
import pandas as pd

from aurex.assets.base import ChainControl, ChainLink, TransmissionChain
from aurex.data.base import price_column
from aurex.factors.projections import Projection, coefficients_only, project

#: Monthly, anchored at the start of each month. The published statistical series in a
#: chain like this are monthly averages already, so this is the frequency the data has
#: rather than a choice; the daily series are averaged into it rather than sampled, to
#: match how a monthly index is constructed.
MONTH_RULE = "MS"

#: Horizons in months. Zero is the impact response, and twelve is where a monthly sample
#: of a couple of hundred observations stops being able to say anything at all.
DEFAULT_HORIZONS: tuple[int, ...] = (0, 1, 3, 6, 12)

DEFAULT_DRAWS = 2_000
DEFAULT_BLOCK = 12


class ChainError(ValueError):
    """The chain cannot be estimated as declared."""


@dataclass(frozen=True, slots=True)
class Compounded:
    """The product of the links at one horizon, and the band the bootstrap gives it."""

    horizon: int
    point: float
    interval: tuple[float, float]
    draws: int
    #: Per-link point estimates at this horizon, in link order, so a reader can see
    #: which arrow the product is being carried by.
    contributions: tuple[float, ...]

    @property
    def spans_zero(self) -> bool:
        return self.interval[0] <= 0.0 <= self.interval[1]

    def describe(self) -> dict[str, Any]:
        return {
            "horizon_months": self.horizon,
            "compounded": round(self.point, 8),
            "interval_95": [round(self.interval[0], 8), round(self.interval[1], 8)],
            "spans_zero": self.spans_zero,
            "link_contributions": [round(value, 8) for value in self.contributions],
            "draws": self.draws,
        }


@dataclass(frozen=True, slots=True)
class ChainEstimate:
    """Everything the chain layer publishes for one declared chain."""

    chain_id: str
    label: str
    links: tuple[Projection, ...]
    compounded: tuple[Compounded, ...]
    direct: Projection
    direct_orthogonalised: Projection
    controlled: Projection | None
    control_months: int
    control_moves: tuple[str, ...]
    mechanical_check: dict[str, Any]
    months: int
    first_month: str
    last_month: str
    note: str

    @property
    def every_horizon_spans_zero(self) -> bool:
        return all(entry.spans_zero for entry in self.compounded)

    def describe(self) -> dict[str, Any]:
        return {
            "chain": self.chain_id,
            "label": self.label,
            "frequency": MONTH_RULE,
            "months": self.months,
            "first_month": self.first_month,
            "last_month": self.last_month,
            "links": [link.describe() for link in self.links],
            "compounded": {
                "method": "product_of_link_responses",
                "band": "joint_moving_block_bootstrap_over_months",
                "every_horizon_spans_zero": self.every_horizon_spans_zero,
                "note": (
                    "The band is not the product of the links' own bands. Multiplying "
                    "intervals would assume the links are independent, and they are "
                    "estimated from overlapping windows of one economy. Each replicate "
                    "resamples months once and re-estimates every link on the same "
                    "months, so the dependence survives into the product."
                ),
                "horizons": [entry.describe() for entry in self.compounded],
            },
            "direct": {
                "raw": self.direct.describe(),
                "orthogonalised": self.direct_orthogonalised.describe(),
                "note": (
                    "Two estimates of the same arrow. The raw one carries every channel "
                    "from the shock to the local price, including the global price "
                    "channel the weekly factor loadings already measure. The "
                    "orthogonalised one holds the quote-currency price constant, so its "
                    "coefficient is the part running through the local economy alone."
                ),
            },
            "administered_control": {
                "months_covered": self.control_months,
                "moves_in_sample": list(self.control_moves),
                "estimate": None if self.controlled is None else self.controlled.describe(),
                "note": (
                    "The first link re-estimated with the administered rate held "
                    "constant, on the months where the schedule knows that rate. It is a "
                    "separate estimate rather than the headline because the schedule "
                    "does not cover the whole sample, and running the headline on the "
                    "covered months alone would trade a control for most of the "
                    "history without saying so."
                ),
            },
            "mechanical_link": self.mechanical_check,
            "double_counting": (
                "The shock this chain starts from is also a driver in the weekly factor "
                "set, so the direct loading and this chain measure overlapping things "
                "through different amounts of noise. They are alternative "
                "decompositions and are never summed; nothing in this artifact adds a "
                "chain response to a factor loading, and there is no total that would "
                "invite it."
            ),
            "note": self.note,
        }


def _monthly(series: pd.Series, *, transform: str) -> pd.Series:
    """Average into months, then difference as the link declares."""
    monthly = series.dropna().resample(MONTH_RULE).mean().dropna()
    if transform == "log_diff":
        # Non-positive values have no logarithm and are not hypothetical for a series
        # that can settle below zero; they become NaN and drop out rather than raising.
        positive = monthly.where(monthly > 0.0)
        changed: pd.Series = np.log(positive).diff().dropna()
        return changed
    if transform == "diff":
        differenced: pd.Series = monthly.diff().dropna()
        return differenced
    raise ChainError(f"unknown chain transform {transform!r}; expected log_diff or diff")


def _series_from(frames: dict[str, pd.DataFrame], series_id: str, column: str | None) -> pd.Series:
    frame = frames.get(series_id)
    if frame is None:
        raise ChainError(f"the chain needs series {series_id!r} and it did not resolve")
    if column is not None:
        if column not in frame.columns:
            raise ChainError(
                f"{series_id!r} has no column {column!r}; it carries "
                f"{sorted(str(name) for name in frame.columns)}"
            )
        return frame[column]
    return price_column(frame)


def _link_inputs(link: ChainLink, frames: dict[str, pd.DataFrame]) -> tuple[pd.Series, pd.Series]:
    shock = _monthly(
        _series_from(frames, link.source_series, link.source_column),
        transform=link.source_transform,
    )
    response = _monthly(
        _series_from(frames, link.target_series, link.target_column),
        transform=link.target_transform,
    )
    return shock, response


def control_frame(control: ChainControl, index: pd.DatetimeIndex) -> pd.Series:
    """The administered rate as a monthly regressor, NaN where the schedule is silent."""
    resolved = pd.Series(
        [control.resolver(stamp.date()) for stamp in index], index=index, dtype="float64"
    )
    if control.transform == "diff":
        # Differenced *without* filling: a NaN month makes its neighbour's difference NaN
        # too, which is correct — a change across an unknown level is an unknown change.
        return resolved.diff()
    return resolved


def moves_in(control: ChainControl, index: pd.DatetimeIndex) -> tuple[str, ...]:
    """Months where the administered rate moved, for the artifact to flag."""
    changes = control_frame(control, index).fillna(0.0)
    return tuple(stamp.date().isoformat() for stamp in index[changes != 0.0])


def _bootstrap_product(
    inputs: list[tuple[pd.Series, pd.Series]],
    *,
    horizons: tuple[int, ...],
    draws: int,
    block: int,
    seed: int,
) -> dict[int, np.ndarray]:
    """Resample months once per replicate and re-estimate every link on the same months.

    The links are aligned onto one shared monthly index first. That alignment is what
    makes a joint resample meaningful: drawing a block of months has to draw the *same*
    months out of every link, or the product is composed from three different samples and
    the dependence the band exists to capture is destroyed by the resampling itself.
    """
    aligned = pd.concat(
        [
            pd.concat([shock.rename(f"s{i}"), response.rename(f"r{i}")], axis=1)
            for i, (shock, response) in enumerate(inputs)
        ],
        axis=1,
    ).dropna()
    n = len(aligned)
    if n < block * 3:
        return {}

    rng = np.random.default_rng(seed)
    span = min(block, n)
    count = int(np.ceil(n / span))
    products: dict[int, list[float]] = {horizon: [] for horizon in horizons}

    for _ in range(draws):
        starts = rng.integers(0, n, size=count)
        picks = (starts[:, None] + np.arange(span)[None, :]) % n
        index = picks.reshape(-1)[:n]
        resampled = aligned.iloc[index].reset_index(drop=True)
        # A resampled block series has no calendar, so it is re-indexed onto a synthetic
        # monthly one. The projections only need ordering and spacing, and the ordering
        # inside each block is what carries the serial dependence.
        stamps = pd.date_range("2000-01-01", periods=n, freq=MONTH_RULE)
        resampled.index = stamps

        per_link: dict[int, list[float]] = {horizon: [] for horizon in horizons}
        for i in range(len(inputs)):
            fitted = coefficients_only(resampled[f"s{i}"], resampled[f"r{i}"], horizons=horizons)
            for horizon in horizons:
                if horizon in fitted:
                    per_link[horizon].append(fitted[horizon])

        for horizon in horizons:
            if len(per_link[horizon]) == len(inputs):
                products[horizon].append(float(np.prod(per_link[horizon])))

    return {horizon: np.array(values) for horizon, values in products.items() if len(values) > 1}


def _mechanical_check(frames: dict[str, pd.DataFrame], chain: TransmissionChain) -> dict[str, Any]:
    """Confirm the terminal arrow is the identity it is asserted to be.

    Not an estimate of the last link — the lens computes the local price from the quote
    and the rate by arithmetic, so its elasticities are one by construction. This is a
    check that the arithmetic in the artifact is the arithmetic claimed, and it is
    reported as a diagnostic rather than as a coefficient with a band.
    """
    local = _monthly(_series_from(frames, chain.terminal_series_id, None), transform="log_diff")
    parts = {
        series_id: _monthly(_series_from(frames, series_id, None), transform="log_diff")
        for series_id in chain.direct_controls
    }
    last_link = chain.links[-1]
    parts[last_link.target_series] = _monthly(
        _series_from(frames, last_link.target_series, last_link.target_column),
        transform="log_diff",
    )

    joined = pd.concat([local.rename("__local__"), pd.DataFrame(parts)], axis=1).dropna()
    if len(joined) < 24:
        return {"available": False, "reason": "too few aligned months to check the identity"}

    design = np.column_stack(
        [np.ones(len(joined)), joined.drop(columns="__local__").to_numpy(dtype=float)]
    )
    target = joined["__local__"].to_numpy(dtype=float)
    fitted = np.linalg.lstsq(design, target, rcond=None)[0]
    residual = target - design @ fitted
    total = float(((target - target.mean()) ** 2).sum())

    return {
        "available": True,
        "asserted_elasticity": 1.0,
        "measured": {
            name: round(float(value), 4)
            for name, value in zip(
                joined.drop(columns="__local__").columns, fitted[1:], strict=True
            )
        },
        "r_squared": round(1.0 - float((residual**2).sum()) / total, 5) if total else None,
        "months": len(joined),
        "note": (
            "The local price is computed from the quote and the rate by the lens, so "
            "these elasticities are one by construction and this is a check on the "
            "arithmetic rather than a measurement of the world. A coefficient away from "
            "one means the tax stack moved inside the sample, which is a fact about the "
            "schedule and not about transmission."
        ),
    }


def estimate(
    chain: TransmissionChain,
    frames: dict[str, pd.DataFrame],
    *,
    horizons: tuple[int, ...] = DEFAULT_HORIZONS,
    draws: int = DEFAULT_DRAWS,
    block: int = DEFAULT_BLOCK,
    seed: int = 11,
) -> ChainEstimate:
    """Estimate every link, compound them, and estimate the whole thing directly."""
    if not chain.links:
        raise ChainError(f"{chain.id}: a chain with no links has nothing to estimate")

    inputs = [_link_inputs(link, frames) for link in chain.links]
    projections = tuple(
        project(
            shock,
            response,
            link_id=link.id,
            horizons=horizons,
            note=link.description,
        )
        for link, (shock, response) in zip(chain.links, inputs, strict=True)
    )

    replicates = _bootstrap_product(inputs, horizons=horizons, draws=draws, block=block, seed=seed)
    compounded: list[Compounded] = []
    for horizon in horizons:
        per_link = [projection.at(horizon) for projection in projections]
        if any(response is None for response in per_link):
            continue
        contributions = tuple(response.coefficient for response in per_link if response)
        values = replicates.get(horizon)
        if values is None or values.size < 2:
            continue
        compounded.append(
            Compounded(
                horizon=horizon,
                point=float(np.prod(contributions)),
                interval=(
                    float(np.quantile(values, 0.025)),
                    float(np.quantile(values, 0.975)),
                ),
                draws=int(values.size),
                contributions=contributions,
            )
        )

    shock = _monthly(_series_from(frames, chain.direct_source_series, None), transform="log_diff")
    local = _monthly(_series_from(frames, chain.terminal_series_id, None), transform="log_diff")
    orthogonalising = pd.DataFrame(
        {
            series_id: _monthly(_series_from(frames, series_id, None), transform="log_diff")
            for series_id in chain.direct_controls
        }
    )

    direct = project(
        shock,
        local,
        link_id=f"{chain.id}_direct",
        horizons=horizons,
        note="The whole path in one regression, every channel included.",
    )
    orthogonalised = project(
        shock,
        local,
        link_id=f"{chain.id}_direct_orthogonalised",
        horizons=horizons,
        controls=orthogonalising if not orthogonalising.empty else None,
        note=(
            "The same regression with the quote-currency price held constant, so the "
            "coefficient is the local-economy part of the path and does not double count "
            "the channel the weekly loadings already carry."
        ),
    )

    controlled: Projection | None = None
    control_months = 0
    control_moves: tuple[str, ...] = ()
    if chain.controls:
        first_shock, first_response = inputs[0]
        index = pd.DatetimeIndex(first_shock.index)
        columns = {control.id: control_frame(control, index) for control in chain.controls}
        control_moves = tuple(
            month for control in chain.controls for month in moves_in(control, index)
        )
        frame = pd.DataFrame(columns).dropna()
        control_months = len(frame)
        if control_months >= 36:
            controlled = project(
                first_shock,
                first_response,
                link_id=f"{chain.links[0].id}_administered_control",
                horizons=horizons,
                controls=frame,
                note=(
                    "The first link with the administered rate held constant, on the "
                    "months the schedule covers."
                ),
            )

    aligned = pd.concat([shock, local], axis=1).dropna()
    return ChainEstimate(
        chain_id=chain.id,
        label=chain.label,
        links=projections,
        compounded=tuple(compounded),
        direct=direct,
        direct_orthogonalised=orthogonalised,
        controlled=controlled,
        control_months=control_months,
        control_moves=control_moves,
        mechanical_check=_mechanical_check(frames, chain),
        months=len(aligned),
        first_month=aligned.index[0].date().isoformat() if len(aligned) else "",
        last_month=aligned.index[-1].date().isoformat() if len(aligned) else "",
        note=chain.note,
    )


def local_price_series(
    prices: pd.Series, rate: pd.Series, *, duty_on: Any = None, at: date | None = None
) -> pd.Series:
    """Deliberately absent. The lens computes the local price; this module never does.

    Kept as a named refusal rather than a missing function because a second implementation
    of the lens arithmetic is exactly how a tax stack drifts away from the schedule that
    carries its citations. Callers pass the lens's own output in under the chain's
    ``terminal_series_id``.
    """
    raise NotImplementedError(
        "the currency lens computes the local price; pass its output in as a series"
    )
