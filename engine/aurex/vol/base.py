"""The volatility interface, and the two decisions every model here inherits.

**Models are fitted in transform space and report in it.** A fit knows nothing about
prices; it produces a conditional sigma series and the standardised residuals that
filtered historical simulation resamples. Anything a reader sees is mapped back to
price space first, by the asset's transform, because a shifted-log transform silently
rescales any percentage quoted out of transform space — see
:mod:`aurex.assets.transforms` for the measurement that makes this non-negotiable.

**The mean is zero unless someone insists otherwise.** A fitted constant mean is a
drift, and a drift over a 20-year sample is a directional forecast wearing a mean's
clothes: it puts the median terminal price above spot and every path-dependent
statistic downstream inherits it. §0 says the null hypothesis is the random walk, so
that is the default. ``mean="constant"`` exists, records itself loudly in
:meth:`describe`, and is nobody's default.

**Breaks are excluded, not absorbed.** A policy step is a mechanical jump in the
price, not information about volatility. Fitting through one lets a scheduled
discontinuity inflate the variance process for weeks afterwards. Excluded dates
contribute nothing to the likelihood and enter the recursion as a zero shock, so the
variance decays across the break rather than spiking at it.

**There is no annualisation helper here, deliberately.** Same reasoning as
:mod:`aurex.assets.transforms`: an annualised sigma computed in transform space is a
percentage that a shifted-log constant can halve, and the only reliable way to stop
one being printed is to make it inconvenient to produce. Report price-space
quantiles instead.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any, Literal, Protocol, runtime_checkable

import numpy as np
import pandas as pd

#: How the conditional mean is handled. See the module docstring.
MeanSpec = Literal["zero", "constant"]


class InsufficientDataError(ValueError):
    """Fewer usable observations than the model needs to be meaningful."""


@runtime_checkable
class FittedVol(Protocol):
    """A fitted conditional-volatility process, ready to simulate forward."""

    @property
    def model_id(self) -> str: ...

    @property
    def mu(self) -> float:
        """Conditional mean in transform space. Zero unless the fit was told not to be."""
        ...

    @property
    def conditional_sigma(self) -> pd.Series:
        """In-sample conditional sigma, indexed like the fitted returns."""
        ...

    @property
    def standardized_residuals(self) -> pd.Series:
        """``(r - mu) / sigma`` — what filtered historical simulation resamples.

        Excluded observations are absent rather than zero: resampling a residual that
        was never fitted would feed a policy step back in as if it were a shock.
        """
        ...

    def forward_sigma(self, horizon: int) -> np.ndarray:
        """Expected sigma path over ``horizon`` steps, from the end of the sample.

        Reporting only. Simulation uses :meth:`propagate`, which lets each path carry
        its own variance — collapsing that to its expectation is exactly the mistake
        §18 objects to.
        """
        ...

    def propagate(self, shocks: np.ndarray) -> np.ndarray:
        """Run the model's own variance recursion forward over standardised shocks.

        ``shocks`` is ``(n_paths, horizon)`` with unit variance; the return is
        ``(n_paths, horizon)`` of simulated returns in transform space.
        """
        ...

    def describe(self) -> dict[str, Any]: ...


class DeterministicVarianceError(ValueError):
    """A model whose paths share one variance trajectory was asked for a leveraged view."""


@runtime_checkable
class VolatilityModel(Protocol):
    """A fittable volatility specification."""

    @property
    def id(self) -> str: ...

    @property
    def per_path_variance(self) -> bool:
        """Does each simulated path carry its own variance trajectory?

        A protocol member rather than a lookup table somewhere downstream, because the
        answer is a property of the recursion and only the model knows it. A model that
        iterates its variance deterministically produces an ensemble in which every
        path sees the same volatility on the same day — fine for a terminal
        distribution, wrong for anything that reads the path. Barrier probabilities and
        liquidation statistics are exactly that, and a leveraged position is scored on
        the path because it is closed out on the path.

        This is the flag :func:`~aurex.vol.require_per_path_variance` reads.
        """
        ...

    def fit(
        self,
        returns: pd.Series,
        *,
        realised_variance: pd.Series | None = None,
        exclude: Iterable[pd.Timestamp] = (),
    ) -> FittedVol:
        """Fit to ``returns``, ignoring ``exclude`` dates in the likelihood.

        ``realised_variance`` is supplied by asset code that has an OHLC series to
        estimate it from; models that do not use it ignore it, and models that
        require it say so rather than substituting the squared return.
        """
        ...

    def describe(self) -> dict[str, Any]: ...


def excluded_mask(index: pd.Index, exclude: Iterable[pd.Timestamp]) -> np.ndarray:
    """Boolean mask, ``True`` where an observation is usable.

    Dates are compared at day resolution: a break is a calendar fact, and the series
    it lands in may carry a timestamp, a timezone, or neither.
    """
    excluded = {pd.Timestamp(stamp).normalize() for stamp in exclude}
    if not excluded:
        return np.ones(len(index), dtype=bool)
    normalized = pd.DatetimeIndex(index).tz_localize(None).normalize()
    return ~np.isin(normalized.to_numpy(), np.array(sorted(excluded), dtype="datetime64[ns]"))


def require_observations(usable: int, minimum: int, model_id: str) -> None:
    if usable < minimum:
        raise InsufficientDataError(
            f"{model_id} needs at least {minimum} usable observations, got {usable}"
        )


def conditional_mean(returns: np.ndarray, mean: MeanSpec) -> float:
    """The fitted mean, which is zero unless explicitly asked for. See the docstring."""
    return float(np.mean(returns)) if mean == "constant" else 0.0
