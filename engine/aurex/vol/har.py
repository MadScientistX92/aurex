"""HAR-RV: heterogeneous autoregression on realised variance.

Corsi's model regresses tomorrow's realised variance on yesterday's, the last week's
and the last month's, as a reduced-form stand-in for traders who look at different
horizons. It is a linear regression, not a filter, so it needs a *measured* variance
series rather than the return series alone.

**Where the measurement comes from.** With daily bars and no intraday data, realised
variance is estimated from the session's range. Parkinson uses the high-low range and
Garman-Klass adds the open-close move; both are several times more efficient than a
squared close-to-close return, which is why the range estimators are worth the
dependency on an OHLC series at all. A close-only series cannot support this model,
and :meth:`HarRv.fit` says so rather than substituting squared returns and calling
the result realised variance.

**Two honest limitations, both recorded in :meth:`describe`.**

* Range estimators assume continuous trading within the session. They understate
  variance across an overnight gap, which is where a substantial share of a
  cash-settled contract's move can arrive.
* The forecast is iterated deterministically over the horizon, because a simulated
  path has no high and no low to measure a range from. Every path in an ensemble
  built on this model therefore shares one variance trajectory, so the ensemble
  carries no variance-of-variance. Where path dependence is the question — a
  leveraged position, §18 — the recursive model is the one to use.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from aurex.vol.base import (
    InsufficientDataError,
    MeanSpec,
    conditional_mean,
    excluded_mask,
    require_observations,
)

#: Corsi's cascade: daily, weekly, monthly.
WEEKLY_LAG = 5
MONTHLY_LAG = 22


def _unmeasurable_is_absent(variance: pd.Series) -> pd.Series:
    """A session with no range is missing, not a measurement of near-zero volatility.

    This used to clip to a floor of ``1e-12`` so a zero range could not put ``-inf``
    into a log regression, and the floor is what made the model unusable. Roughly 7% of
    the futures sessions in this repository's own cache print ``high == low`` — a stale
    or synthetic quote, not a session the asset did not move in. Flooring them recorded
    each one as a log-variance of -27.6 against a series whose real mean is -10.5, which
    tripled the residual spread of the HAR regression, and the retransformation term is
    half that spread inside an exponential: the fitted smearing came out at 9.87, a
    multiplier of about 19,000, and the variance recursion diverged within two steps.

    Absence is a first-class state everywhere else in this engine, and it is the correct
    reading here too. A session that cannot be measured is dropped, and the count of what
    was dropped is reported by the fit rather than absorbed.
    """
    return variance.where(variance > 0.0).rename("realised_variance")


def parkinson_variance(frame: pd.DataFrame) -> pd.Series:
    """Range-based daily variance from high and low."""
    _require_columns(frame, ("high", "low"))
    log_range = np.log(frame["high"].astype(float) / frame["low"].astype(float))
    variance: pd.Series = log_range**2 / (4.0 * np.log(2.0))
    return _unmeasurable_is_absent(variance)


def garman_klass_variance(frame: pd.DataFrame) -> pd.Series:
    """Range-based daily variance using the open-close move as well."""
    _require_columns(frame, ("open", "high", "low", "close"))
    high_low = np.log(frame["high"].astype(float) / frame["low"].astype(float))
    close_open = np.log(frame["close"].astype(float) / frame["open"].astype(float))
    variance: pd.Series = 0.5 * high_low**2 - (2.0 * np.log(2.0) - 1.0) * close_open**2
    return _unmeasurable_is_absent(variance)


def _require_columns(frame: pd.DataFrame, columns: tuple[str, ...]) -> None:
    missing = [c for c in columns if c not in frame.columns]
    if missing:
        raise InsufficientDataError(
            f"realised variance needs {list(columns)}; missing {missing}. A close-only "
            "series cannot support a range estimator."
        )


@dataclass(frozen=True, slots=True)
class HarRvFit:
    """A fitted HAR-RV regression on log realised variance."""

    model_id: str
    mu: float
    #: ``const``, ``daily``, ``weekly``, ``monthly`` on the log-variance scale.
    coefficients: dict[str, float]
    #: Half the residual variance of the log regression: the retransformation term.
    smearing: float
    conditional_sigma: pd.Series
    standardized_residuals: pd.Series
    #: Realised variance history the forecast recursion starts from, oldest first.
    recent_variance: np.ndarray
    r_squared: float
    n_observations: int
    n_excluded: int
    mean_spec: MeanSpec

    def forward_sigma(self, horizon: int) -> np.ndarray:
        """Iterate the cascade, applying the retransformation once per *report*.

        The smearing term converts a forecast of ``log RV`` into a forecast of ``RV``,
        and it belongs on the number that leaves this method rather than on the state
        that feeds the next step. Adding it to the state compounds it: the recursion's
        fixed point moves up by ``smearing / (1 - sum of the lag coefficients)``, which
        on this repository's own cached sample is about ten log units. Measured against
        the realised standard deviation of h-session returns, compounding it overshot by
        51% at a week and 250% at a quarter; applying it once lands within 25% at a week
        and within a percent at a month.

        What remains is a real and declared bias in the other direction. A deterministic
        path has no dispersion, so the weekly and monthly regressors — which are logs of
        *arithmetic* means in the fitted data — lose the Jensen gap that the estimated
        constant absorbed, and the forecast drifts low as the horizon grows: about 20%
        short at forty-two sessions and 32% at sixty-three. That is a property of
        iterating a log cascade deterministically rather than simulating it, which is the
        limitation this model already declares, and it is recorded here so the number is
        read with the bias in front of it.
        """
        if horizon < 1:
            raise ValueError(f"horizon must be positive, got {horizon}")

        history = list(self.recent_variance)
        out = []
        for _ in range(horizon):
            log_next = self._next_log_variance(history)
            out.append(np.sqrt(np.exp(log_next + self.smearing)))
            history.append(float(np.exp(log_next)))
        return np.array(out, dtype=float)

    def _next_log_variance(self, history: list[float]) -> float:
        """One step of the cascade, on the log scale it was estimated on."""
        recent = np.asarray(history, dtype=float)
        return float(
            self.coefficients["const"]
            + self.coefficients["daily"] * np.log(recent[-1])
            + self.coefficients["weekly"] * np.log(recent[-WEEKLY_LAG:].mean())
            + self.coefficients["monthly"] * np.log(recent[-MONTHLY_LAG:].mean())
        )

    def propagate(self, shocks: np.ndarray) -> np.ndarray:
        """Every path shares one variance trajectory. See the module docstring."""
        if shocks.ndim != 2:
            raise ValueError(f"shocks must be (n_paths, horizon), got {shocks.shape}")
        sigma = self.forward_sigma(shocks.shape[1])
        simulated: np.ndarray = self.mu + sigma[np.newaxis, :] * shocks
        return simulated

    def describe(self) -> dict[str, Any]:
        return {
            "model": self.model_id,
            "mean": self.mean_spec,
            "mu": self.mu,
            "coefficients": dict(self.coefficients),
            "r_squared": self.r_squared,
            "observations": self.n_observations,
            "excluded_observations": self.n_excluded,
            "scale": "fitted on log realised variance; forecasts carry a smearing term",
            "limitations": [
                "Range estimators assume continuous sessions and understate overnight gaps.",
                "The variance path is iterated deterministically, so an ensemble "
                "built on this model carries no variance-of-variance.",
            ],
        }


@dataclass(frozen=True, slots=True)
class HarRv:
    """The HAR-RV specification. Requires a measured realised-variance series."""

    mean: MeanSpec = "zero"
    min_observations: int = 250
    id: str = "har_rv"
    #: The cascade is iterated deterministically — a simulated path has no intraday
    #: range to measure realised variance from — so every path in the ensemble shares
    #: one variance trajectory.
    per_path_variance: bool = False

    def fit(
        self,
        returns: pd.Series,
        *,
        realised_variance: pd.Series | None = None,
        exclude: Iterable[pd.Timestamp] = (),
    ) -> HarRvFit:
        if realised_variance is None:
            raise InsufficientDataError(
                f"{self.id} needs a realised-variance series estimated from OHLC; "
                "squared close-to-close returns are not a substitute for it"
            )

        # Dropped rather than floored, and dropped *before* the lags are built, so the
        # cascade averages five and twenty-two sessions it could actually measure. A
        # non-positive value surviving here came from a caller's own series rather than
        # from the estimators above, and it gets the same treatment.
        variance = realised_variance.dropna().astype(float)
        measurable = variance[variance > 0.0]
        if measurable.empty:
            raise InsufficientDataError(
                f"{self.id}: no session in the realised-variance series has a "
                "measurable range, so there is nothing to regress"
            )
        design, target, index = _har_design(measurable)
        usable = excluded_mask(index, exclude)
        require_observations(int(usable.sum()), self.min_observations, self.id)

        coefficients, fitted, r_squared = _ordinary_least_squares(
            design[usable], target[usable], design
        )
        residual_variance = float(np.var(target[usable] - fitted[usable], ddof=len(coefficients)))
        smearing = residual_variance / 2.0

        # exp(E[log RV]) is a median, not a mean; the smearing term is the
        # lognormal retransformation that makes the forecast a variance again.
        sigma = pd.Series(np.sqrt(np.exp(fitted + smearing)), index=index, name="sigma")

        clean_returns = returns.dropna().astype(float)
        aligned = clean_returns.reindex(sigma.index).dropna()
        if aligned.empty:
            raise InsufficientDataError(
                f"{self.id}: the return series and the realised-variance series do "
                "not overlap after aligning"
            )
        mu = conditional_mean(aligned.to_numpy(), self.mean)
        standardized = ((aligned - mu) / sigma.reindex(aligned.index)).rename("z")

        return HarRvFit(
            model_id=self.id,
            mu=mu,
            coefficients=coefficients,
            smearing=smearing,
            conditional_sigma=sigma,
            standardized_residuals=standardized[excluded_mask(standardized.index, exclude)],
            # From the measurable series, so the recursion cannot start from a session
            # that was dropped for having no range.
            recent_variance=measurable.to_numpy()[-MONTHLY_LAG:],
            r_squared=r_squared,
            n_observations=int(usable.sum()),
            n_excluded=int((~usable).sum()),
            mean_spec=self.mean,
        )

    def describe(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "mean": self.mean,
            "min_observations": self.min_observations,
            "cascade_days": [1, WEEKLY_LAG, MONTHLY_LAG],
        }


def _har_design(variance: pd.Series) -> tuple[np.ndarray, np.ndarray, pd.Index]:
    """Regressors are strictly lagged, so nothing predicts itself."""
    log_variance = np.log(variance)
    daily = log_variance.shift(1)
    weekly = np.log(variance.rolling(WEEKLY_LAG).mean()).shift(1)
    monthly = np.log(variance.rolling(MONTHLY_LAG).mean()).shift(1)

    frame = pd.concat(
        {"target": log_variance, "daily": daily, "weekly": weekly, "monthly": monthly},
        axis=1,
    ).dropna()
    if frame.empty:
        raise InsufficientDataError("realised-variance series is too short for HAR lags")

    design = np.column_stack(
        [
            np.ones(len(frame)),
            frame["daily"].to_numpy(),
            frame["weekly"].to_numpy(),
            frame["monthly"].to_numpy(),
        ]
    )
    return design, frame["target"].to_numpy(), frame.index


def _ordinary_least_squares(
    design: np.ndarray, target: np.ndarray, full_design: np.ndarray
) -> tuple[dict[str, float], np.ndarray, float]:
    beta, *_ = np.linalg.lstsq(design, target, rcond=None)
    fitted_in_sample = design @ beta
    total = float(np.sum((target - target.mean()) ** 2))
    residual = float(np.sum((target - fitted_in_sample) ** 2))
    r_squared = 1.0 - residual / total if total > 0.0 else 0.0

    names = ("const", "daily", "weekly", "monthly")
    coefficients = {name: float(value) for name, value in zip(names, beta, strict=True)}
    return coefficients, full_design @ beta, r_squared
