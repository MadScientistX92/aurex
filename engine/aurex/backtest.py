"""Composition: an asset, the model it declares, and the scoring harness.

The third composition module, alongside :mod:`aurex.forecast` and
:mod:`aurex.pipeline`, and it exists for the same reason they do — :mod:`aurex.score`
must not know what it is scoring, and an asset must not know how it will be graded.
This is the only file that holds both.

**The null is built from the same parts as the model.** Both forecasters get the
asset's own return transform and its venue's session limit, so the random walk is not
a straw man assembled from different plumbing: it differs from the model in exactly one
respect, which is that it has no conditional variance. That is the comparison §0 asks
for.

**A backtest scores the asset's own quote, not a lens.** The distributions this grades
are the ones the engine publishes for the native view. A currency lens composes the
base paths with an exchange rate through a copula, so scoring it means walking that
joint simulation forward too; that is a larger object and it is not scored here yet
rather than being approximated by fitting the converted price directly, which would
grade something the engine never publishes.
"""

from __future__ import annotations

import dataclasses
from typing import Any

import pandas as pd

from aurex.assets.base import Asset
from aurex.score.forecasters import ModelForecaster, RandomWalkForecaster
from aurex.score.walkforward import (
    DEFAULT_MIN_OBSERVATIONS,
    WalkForwardRequest,
    WalkForwardResult,
    walk_forward,
)
from aurex.vol import model_for
from aurex.vol.har import parkinson_variance

#: Paths per simulated date. Lower than a published forecast's ensemble because a
#: backtest runs hundreds of them; the Monte Carlo error this leaves in a single CRPS
#: averages out across dates, and the fair estimator removes the ensemble-size bias
#: that would otherwise favour whichever side simulated more.
DEFAULT_BACKTEST_PATHS = 4_000


def backtest_asset(
    asset: Asset,
    *,
    prices: pd.Series,
    ohlc: pd.DataFrame | None = None,
    breaks: tuple[pd.Timestamp, ...] = (),
    request: WalkForwardRequest | None = None,
    model_id: str | None = None,
    n_paths: int = DEFAULT_BACKTEST_PATHS,
    block_length: int = 10,
) -> WalkForwardResult:
    """Walk the asset's declared model forward against the random walk."""
    defaults = asset.vol_defaults
    ask = request or WalkForwardRequest(
        # The harness must not start before the model can be fitted at all, or every
        # early date is a recorded skip rather than a forecast.
        min_observations=max(DEFAULT_MIN_OBSERVATIONS, defaults.min_observations)
    )

    model = model_for(
        model_id or defaults.default_model,
        min_observations=defaults.min_observations,
        **dict(defaults.model_options),
    )

    realised = None
    if ohlc is not None and {"high", "low"} <= set(ohlc.columns):
        realised = parkinson_variance(ohlc)

    subject = ModelForecaster(
        model=model,
        transform=asset.return_transform,
        session_limit=defaults.session_limit,
        block_length=block_length,
        n_paths=n_paths,
        breaks=breaks if defaults.break_aware else (),
        realised_variance=realised,
    )
    baseline = RandomWalkForecaster(
        transform=asset.return_transform,
        session_limit=defaults.session_limit,
        n_paths=n_paths,
        min_observations=defaults.min_observations,
    )
    # The second null carries the sample's own drift. Filtered historical simulation
    # resamples empirical standardised residuals, whose mean is not zero in a sample
    # that trended, so the model is not driftless either — and scoring it only against
    # a demeaned null would credit that drift to its volatility work. The difference
    # between the two skill scores is how much of the win was drift.
    drift_matched = dataclasses.replace(baseline, demean=False)

    return walk_forward(
        prices,
        subject=subject,
        baseline=baseline,
        request=ask,
        extra_baselines=(drift_matched,),
    )


def describe_backtest(asset: Asset, result: WalkForwardResult) -> dict[str, Any]:
    """Artifact block: which asset was scored, and how it did."""
    return {
        "asset": {
            "id": asset.id,
            "label": asset.label,
            "quote_currency": asset.quote_currency,
            "base_unit": asset.base_unit,
            "price_series_id": asset.price_series_id,
        },
        "calibration": result.calibration().describe(),
        "scope": (
            "The asset's own quote. A currency lens composes these paths with an "
            "exchange rate through a copula and is not scored here."
        ),
    }
