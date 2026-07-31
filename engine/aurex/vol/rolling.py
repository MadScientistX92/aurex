"""Rolling-window standard deviation — the model everything else has to beat.

It has no parameters to overfit, no optimiser to fail, and no memory beyond its
window. Included as a working model rather than a placeholder because a conditional
variance model that cannot beat a 60-day standard deviation out of sample has not
earned its extra parameters, and §6's benchmark table needs the honest baseline
sitting in the same interface as the models it is judging.

The window is strictly backward-looking: sigma for day *t* is estimated from days
before *t*. Including the day's own return in its own conditional variance would
shrink the standardised residuals toward zero and make every simulated path from
them too narrow.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from aurex.vol.base import (
    MeanSpec,
    conditional_mean,
    excluded_mask,
    require_observations,
)


@dataclass(frozen=True, slots=True)
class RollingStdFit:
    """A fitted rolling-window volatility estimate."""

    model_id: str
    mu: float
    window: int
    conditional_sigma: pd.Series
    standardized_residuals: pd.Series
    #: Sigma carried forward to every future step; this model has no dynamics.
    next_sigma: float
    n_observations: int
    n_excluded: int
    mean_spec: MeanSpec

    def forward_sigma(self, horizon: int) -> np.ndarray:
        if horizon < 1:
            raise ValueError(f"horizon must be positive, got {horizon}")
        return np.full(horizon, self.next_sigma, dtype=float)

    def propagate(self, shocks: np.ndarray) -> np.ndarray:
        """Constant sigma across the horizon — no recursion, by construction.

        Paths from this model differ only in their shocks, so the ensemble misses the
        variance clustering a leveraged position actually meets. That is the model's
        honest limitation, not a bug to paper over.
        """
        if shocks.ndim != 2:
            raise ValueError(f"shocks must be (n_paths, horizon), got {shocks.shape}")
        return self.mu + self.next_sigma * shocks

    def describe(self) -> dict[str, Any]:
        return {
            "model": self.model_id,
            "window": self.window,
            "mean": self.mean_spec,
            "mu": self.mu,
            "observations": self.n_observations,
            "excluded_observations": self.n_excluded,
            "dynamics": "none: sigma is held flat across the horizon",
        }


@dataclass(frozen=True, slots=True)
class RollingStd:
    """Backward-looking rolling standard deviation."""

    window: int = 60
    mean: MeanSpec = "zero"
    #: Carried so every model answers to the asset's declared minimum, even one whose
    #: own binding constraint is its window.
    min_observations: int = 0
    id: str = "rolling_std"

    def fit(
        self,
        returns: pd.Series,
        *,
        realised_variance: pd.Series | None = None,
        exclude: Iterable[pd.Timestamp] = (),
    ) -> RollingStdFit:
        del realised_variance

        clean = returns.dropna().astype(float)
        usable = excluded_mask(clean.index, exclude)
        n_excluded = int((~usable).sum())
        # Dropped outright rather than masked in place: a rolling window over a hole
        # would silently shorten itself.
        kept = clean[usable]
        require_observations(len(kept), max(self.window + 1, self.min_observations), self.id)

        mu = conditional_mean(kept.to_numpy(), self.mean)
        sigma = kept.shift(1).rolling(self.window, min_periods=self.window).std(ddof=1)
        valid = sigma.notna() & (sigma > 0.0)
        require_observations(int(valid.sum()), 1, self.id)

        sigma = sigma[valid]
        standardized = (kept[valid] - mu) / sigma

        return RollingStdFit(
            model_id=self.id,
            mu=mu,
            window=self.window,
            conditional_sigma=sigma.rename("sigma"),
            standardized_residuals=standardized.rename("z"),
            next_sigma=float(kept.iloc[-self.window :].std(ddof=1)),
            n_observations=int(valid.sum()),
            n_excluded=n_excluded,
            mean_spec=self.mean,
        )

    def describe(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "window": self.window,
            "mean": self.mean,
            "min_observations": self.min_observations,
        }
