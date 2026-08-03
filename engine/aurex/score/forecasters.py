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
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

import numpy as np
import pandas as pd

from aurex.assets.transforms import ReturnTransform
from aurex.dist.fhs import SimulationSpec, bootstrap_shocks, paths_from_returns, simulate
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

    @property
    def label(self) -> str:
        return self.model.id

    def forecast(
        self, history: pd.Series, *, horizons: tuple[int, ...], seed: int
    ) -> dict[int, PathEnsemble]:
        returns = self.transform.to_returns(history)
        as_of = history.index[-1]
        realised = None if self.realised_variance is None else self.realised_variance.loc[:as_of]
        fit = self.model.fit(returns, realised_variance=realised, exclude=self.breaks)

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
                ),
                session_limit=self.session_limit,
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
        returns = self.transform.to_returns(history).dropna().to_numpy(dtype=float)
        require_observations(int(returns.size), self.min_observations, self.label)

        increments = returns - float(np.mean(returns)) if self.demean else returns
        anchor = float(history.iloc[-1])

        ensembles: dict[int, PathEnsemble] = {}
        for horizon in horizons:
            spec = SimulationSpec(
                horizon_days=horizon,
                n_paths=self.n_paths,
                block_length=self.block_length,
                seed=seed + horizon,
            )
            drawn = bootstrap_shocks(
                increments,
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
                    "residual_sample_size": int(increments.size),
                    "transform": self.transform.describe(),
                    "drift": (
                        "removed from the resampled increments"
                        if self.demean
                        else "left in the resampled increments"
                    ),
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
