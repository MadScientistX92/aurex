"""Composition: an asset, the model it declares, and the scoring harness.

The third composition module, alongside :mod:`aurex.forecast` and
:mod:`aurex.pipeline`, and it exists for the same reason they do — :mod:`aurex.score`
must not know what it is scoring, and an asset must not know how it will be graded.
This is the only file that holds both.

**The null is built from the same parts as the model.** Both forecasters get the
asset's own return transform, its venue's session limit and the *same drift policy*, so
the random walk is not a straw man assembled from different plumbing: it differs from
the model in exactly one respect, which is that it has no conditional variance. That is
the comparison §0 asks for.

That last part was not true until it was fixed, and the failure is worth recording
because it is easy to reproduce. The model resampled an undemeaned residual pool while
the null had its mean removed by construction, so the model was scored carrying a drift
its null was denied — worth up to +4.6% of CRPS skill at a quarter on the first asset
measured, all of it eleven years of appreciation rather than anything the volatility
layer did. The drift policy is now one flag that both sides read, and it also picks
which null is the like-for-like one rather than leaving the caller to remember. A
comparison whose fairness depends on someone remembering is not one.

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
from aurex.dist.fhs import DEMEAN_BY_DEFAULT
from aurex.score.events import BinaryEvent, default_events
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
    demean_residuals: bool = DEMEAN_BY_DEFAULT,
    extra_events: tuple[BinaryEvent, ...] = (),
) -> WalkForwardResult:
    """Walk the asset's declared model forward against the random walk.

    ``demean_residuals`` sets the drift policy for *both* sides. Turning it off scores
    a drift-continuation model, which is a legitimate thing to measure and is not the
    default; the required baseline stays §0's driftless walk either way, and the
    drift-matched walk rides along as the second null so both readings are always in
    the artifact.

    ``extra_events`` is how the hurdle events reach the scorer. They are built by the
    caller from a route and a jurisdiction and arrive here as ordinary binary events
    carrying a number, so this module never learns what friction is — the same reason
    the scorer does not. Passing none scores the price events alone, which is a complete
    run rather than a degraded one.
    """
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
        demean_residuals=demean_residuals,
    )
    baseline = RandomWalkForecaster(
        transform=asset.return_transform,
        session_limit=defaults.session_limit,
        n_paths=n_paths,
        min_observations=defaults.min_observations,
    )
    # The second null carries the sample's own drift, and it stays even now that the
    # model is demeaned by default — it is what makes the drift visible. With a
    # driftless model it is a *hindsight* benchmark: a walk handed the sample mean it
    # could not have known, which is why losing to it is not a finding about the model
    # and beating it is a strong one.
    drift_matched = dataclasses.replace(baseline, demean=False)

    return walk_forward(
        prices,
        subject=subject,
        baseline=baseline,
        request=ask,
        events=(*default_events(ask.reference_moves), *extra_events),
        extra_baselines=(drift_matched,),
    )


def describe_backtest(asset: Asset, result: WalkForwardResult) -> dict[str, Any]:
    """Artifact block: which asset was scored, and how it did."""
    # Indexed rather than fetched with a default: a subject that stopped declaring its
    # drift policy must break the artifact loudly, because the alternative is a report
    # that names the wrong null as the fair one and reads perfectly well doing it.
    like_for_like = result.subject["like_for_like_null"]
    return {
        # The window every number below came from. Published at the top level rather
        # than only per horizon, because the per-horizon first/last as-of dates describe
        # where forecasts happened to land, not what the run was allowed to see.
        "sample": None if result.sample is None else result.sample.describe(),
        "asset": {
            "id": asset.id,
            "label": asset.label,
            "quote_currency": asset.quote_currency,
            "base_unit": asset.base_unit,
            "price_series_id": asset.price_series_id,
        },
        "calibration": result.calibration().describe(),
        "like_for_like_null": {
            "null": like_for_like,
            "why": (
                "The subject and this null share a drift policy, so the difference "
                "between their scores is conditional variance and nothing else. The "
                "other null differs in drift as well, and its skill score should be "
                "read as the size of that difference rather than as a verdict."
            ),
        },
        "scope": (
            "The asset's own quote. A currency lens composes these paths with an "
            "exchange rate through a copula and is not scored here."
        ),
    }
