"""Something that can be asked for a distribution *as of* a past date.

The walk-forward needs to re-ask the engine what it would have said on a day twenty
years ago, and it needs to ask the null the same question in the same words. That is
what this protocol is for, and having exactly one of it is what makes the comparison
structurally fair: the model and the baseline go through the same bootstrap, the same
price-space walk and the same session-limit handling, so any difference between their
scores is the conditional variance and nothing else.

**No lookahead is structural, not procedural.** A forecaster receives a price history
and never learns the index it was cut from, so there is nothing later than the as-of
date for it to reach for. The harness slices; the forecaster cannot unslice.

**The null is driftless by construction.** :class:`RandomWalkForecaster` removes the
sample mean from the increments it resamples rather than trusting a long sample to
average out. Twenty years of any appreciating series carries a large positive mean, and
a null that quietly inherited it would be a directional forecast — one that any model
beats for the wrong reason in a flat sample, and that beats every model for a worse one
in a rising sample.

**And so is the model, now.** Both forecasters declare their drift through the same
:class:`~aurex.dist.fhs.ResidualPool`, and both default to demeaned. That is not tidying:
the model used to resample an undemeaned pool, which made it a drift-continuation
forecast scored against a null that had been denied one, and the difference showed up as
CRPS skill that belonged to neither. Two forecasters that differ in their drift policy
are not comparable, so the policy is the same object on both sides and the comparison is
back to conditional variance and nothing else.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

import numpy as np
import pandas as pd

from aurex.assets.transforms import ReturnTransform
from aurex.dist.fhs import (
    DEMEAN_BY_DEFAULT,
    SimulationSpec,
    bootstrap_shocks,
    paths_from_returns,
    residual_pool,
    simulate,
)
from aurex.dist.paths import PathEnsemble
from aurex.vol.base import VolatilityModel, require_observations
from aurex.vol.limits import SessionLimit


@runtime_checkable
class AsOfForecaster(Protocol):
    """Produces the distributions one would have published on the last day it can see."""

    @property
    def label(self) -> str: ...

    def forecast(
        self, history: pd.Series, *, horizons: tuple[int, ...], seed: int
    ) -> dict[int, PathEnsemble]:
        """One ensemble per horizon, anchored on the final observation of ``history``.

        All horizons at once because the expensive part — the fit — does not depend on
        the horizon, and refitting per horizon would triple a backtest for nothing.
        Implementations raise :class:`~aurex.vol.base.InsufficientDataError` when the
        history is too short, so the harness can record a skip with a reason rather
        than a silent gap.
        """
        ...

    def describe(self) -> dict[str, Any]: ...


@dataclass(frozen=True, slots=True)
class ModelForecaster:
    """The engine's own answer: fit the declared volatility model, simulate by FHS."""

    model: VolatilityModel
    transform: ReturnTransform
    session_limit: SessionLimit | None = None
    block_length: int = 10
    n_paths: int = 4_000
    breaks: tuple[pd.Timestamp, ...] = ()
    #: Realised variance over the *whole* history, sliced to the as-of date on every
    #: call. Range-based models need it and return-only models ignore it; supplying it
    #: here rather than recomputing per date keeps the OHLC estimator out of the loop.
    realised_variance: pd.Series | None = None
    #: Resample the standardised residuals with their mean removed. Default, and the
    #: same default the null carries, so the two are comparable by construction.
    demean_residuals: bool = DEMEAN_BY_DEFAULT

    @property
    def label(self) -> str:
        return self.model.id

    @property
    def like_for_like_null(self) -> str:
        """Which null this forecaster's drift policy makes the fair comparison.

        A demeaned model belongs against the demeaned walk, which is §0's null and the
        one every run is required to carry. A model resampling a pool that still holds
        the sample's drift has to be read against the drift-matched walk instead, or the
        skill score is measuring the drift. Derived rather than configured, so the two
        cannot be set inconsistently.
        """
        return "random_walk" if self.demean_residuals else "random_walk_drift_matched"

    def forecast(
        self, history: pd.Series, *, horizons: tuple[int, ...], seed: int
    ) -> dict[int, PathEnsemble]:
        returns = self.transform.to_returns(history)
        as_of = history.index[-1]
        realised = None if self.realised_variance is None else self.realised_variance.loc[:as_of]
        fit = self.model.fit(returns, realised_variance=realised, exclude=self.breaks)
        pool = residual_pool(fit.standardized_residuals, demean=self.demean_residuals)

        anchor = float(history.iloc[-1])
        return {
            horizon: simulate(
                fit,
                transform=self.transform,
                anchor=anchor,
                spec=SimulationSpec(
                    horizon_days=horizon,
                    n_paths=self.n_paths,
                    block_length=self.block_length,
                    seed=seed + horizon,
                    demean_residuals=self.demean_residuals,
                ),
                session_limit=self.session_limit,
                pool=pool,
            )
            for horizon in horizons
        }

    def describe(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "model": self.model.describe(),
            "transform": self.transform.describe(),
            "n_paths": self.n_paths,
            "block_length": self.block_length,
            "breaks_excluded": len(self.breaks),
            "residual_pool": {
                "demeaned": self.demean_residuals,
                "drift": (
                    "none: the resampled pool is centred, so the simulated median sits "
                    "at spot and the distribution makes no directional claim"
                    if self.demean_residuals
                    else "the sample's own, carried by the resampled residuals rather "
                    "than fitted as a mean"
                ),
            },
            "like_for_like_null": self.like_for_like_null,
        }


@dataclass(frozen=True, slots=True)
class RandomWalkForecaster:
    """The null: increments drawn iid from the history, no conditional variance.

    Empirical increments rather than a fitted normal, so the null inherits the same fat
    tails the model does and the comparison is about *conditioning* — whether knowing
    today's volatility helps — rather than about tail shape, which would be a much
    easier contest to win.

    **Two nulls, because they answer different questions.** With ``demean`` the null is
    driftless, which is what §0 names, and beating it means beating "no drift and no
    conditioning" together. Without it the null carries whatever drift the history had,
    which isolates the conditioning. Filtered historical simulation resamples empirical
    standardised residuals, and in a sample that rose those have a positive mean — so
    the *model* is not driftless either, and scoring it only against the demeaned null
    credits the model's drift to its volatility work. Run both.
    """

    transform: ReturnTransform
    session_limit: SessionLimit | None = None
    n_paths: int = 4_000
    #: One means iid, which is what a random walk means. Raising it would import serial
    #: structure into the null and quietly make it a model.
    block_length: int = 1
    min_observations: int = 250
    #: Remove the sample mean from the resampled increments. See the class docstring.
    demean: bool = True

    @property
    def label(self) -> str:
        return "random_walk" if self.demean else "random_walk_drift_matched"

    def forecast(
        self, history: pd.Series, *, horizons: tuple[int, ...], seed: int
    ) -> dict[int, PathEnsemble]:
        returns = self.transform.to_returns(history).dropna()
        require_observations(int(returns.size), self.min_observations, self.label)

        # The same pool type the model uses. The null's increments are raw returns
        # rather than standardised residuals, but "does this pool carry a drift" is the
        # same question, and asking it through one object is what keeps the two sides
        # of the comparison honest.
        pool = residual_pool(returns, demean=self.demean)
        anchor = float(history.iloc[-1])

        ensembles: dict[int, PathEnsemble] = {}
        for horizon in horizons:
            spec = SimulationSpec(
                horizon_days=horizon,
                n_paths=self.n_paths,
                block_length=self.block_length,
                seed=seed + horizon,
                demean_residuals=self.demean,
            )
            drawn = bootstrap_shocks(
                pool,
                n_paths=spec.n_paths,
                horizon=spec.horizon_days,
                block_length=spec.block_length,
                rng=np.random.default_rng(spec.seed),
            )
            prices, diagnostics = paths_from_returns(
                self.transform, anchor, drawn, session_limit=self.session_limit
            )
            ensembles[horizon] = PathEnsemble(
                prices=prices,
                anchor=anchor,
                diagnostics=diagnostics
                | {
                    "simulation": spec.describe(),
                    "vol_model": {"model": self.label, "conditional_variance": False},
                    "residual_sample_size": int(pool.residuals.size),
                    "residual_pool": pool.describe(),
                    "transform": self.transform.describe(),
                },
            )
        return ensembles

    def describe(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "model": {
                "id": self.label,
                "increments": (
                    "iid draws from the demeaned empirical return distribution"
                    if self.demean
                    else "iid draws from the raw empirical return distribution"
                ),
                "drift": (
                    "zero by construction, not by sample average"
                    if self.demean
                    else "the sample's own, so the comparison isolates conditioning"
                ),
            },
            "transform": self.transform.describe(),
            "n_paths": self.n_paths,
            "block_length": self.block_length,
        }
