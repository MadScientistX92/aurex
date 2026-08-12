"""Local projections. One regression per horizon, no identifying assumptions imposed.

§19 asks for local projections rather than a VAR, and the reason is worth stating because
it decides what the output means. A VAR estimates a small system and then *derives* every
horizon from it by iterating the same coefficients forward, so a misspecification at one
lag compounds into every impulse response, and the bands at long horizons inherit a
structure nobody tested. A local projection fits horizon ``h`` directly — regress the
cumulative move from ``t`` to ``t + h`` on the shock at ``t`` — so each horizon carries
its own coefficient, its own residual and its own band, and a horizon with nothing in it
says so instead of borrowing significance from a horizon that does.

The cost is efficiency and overlapping residuals: the response at ``h`` and at ``h + 1``
are estimated from windows that share all but one period, which is a moving-average
dependence of order ``h`` by construction rather than by accident. So the standard error
is Newey-West with a truncation of at least ``h``. That is the Jordà convention and it is
not optional here: at ``h = 12`` on two hundred monthly observations, the uncorrected
error is roughly half the right one, which is the difference between a band that spans
zero and a finding.

Nothing in this module knows what any series is. It takes a shock, a response, optional
controls, and returns a coefficient per horizon with the band around it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

#: Own lags of the shock and the response included as controls at every horizon. Two,
#: which is enough to absorb the first-order persistence in monthly macro series without
#: spending a tenth of a two-hundred-month sample on nuisance parameters.
DEFAULT_LAGS = 2

#: 95%, and the normal quantile rather than a t: the HAC covariance is an asymptotic
#: object, so a small-sample reference would claim a precision the estimator lacks.
Z_95 = 1.959963984540054


class ProjectionError(ValueError):
    """A projection cannot be estimated as asked."""


@dataclass(frozen=True, slots=True)
class Response:
    """The response at one horizon, with the band around it."""

    horizon: int
    coefficient: float
    std_error: float
    observations: int
    hac_lag: int

    @property
    def interval(self) -> tuple[float, float]:
        return (
            self.coefficient - Z_95 * self.std_error,
            self.coefficient + Z_95 * self.std_error,
        )

    @property
    def spans_zero(self) -> bool:
        low, high = self.interval
        return low <= 0.0 <= high

    @property
    def p_value(self) -> float:
        if self.std_error <= 0.0:
            return float("nan")
        return float(2.0 * stats.norm.sf(abs(self.coefficient) / self.std_error))

    def describe(self) -> dict[str, Any]:
        low, high = self.interval
        return {
            "horizon_months": self.horizon,
            "coefficient": round(self.coefficient, 6),
            "std_error": round(self.std_error, 6),
            "interval_95": [round(low, 6), round(high, 6)],
            "spans_zero": self.spans_zero,
            "p_value": round(self.p_value, 4),
            "observations": self.observations,
            "hac_lag": self.hac_lag,
        }


@dataclass(frozen=True, slots=True)
class Projection:
    """One link: a shock, a response, and what it does over the horizons."""

    link_id: str
    responses: tuple[Response, ...]
    controls: tuple[str, ...]
    lags: int
    note: str

    def at(self, horizon: int) -> Response | None:
        return next((r for r in self.responses if r.horizon == horizon), None)

    @property
    def every_horizon_spans_zero(self) -> bool:
        return all(response.spans_zero for response in self.responses)

    def describe(self) -> dict[str, Any]:
        return {
            "link": self.link_id,
            "estimator": "local_projection",
            "controls": list(self.controls),
            "own_lags": self.lags,
            "variance_estimator": "newey_west_bartlett",
            "hac_rule": "max(horizon, floor(4 (n/100)^(2/9)))",
            "every_horizon_spans_zero": self.every_horizon_spans_zero,
            "note": self.note,
            "responses": [response.describe() for response in self.responses],
        }


def hac_errors(
    design: np.ndarray, target: np.ndarray, *, lag: int
) -> tuple[np.ndarray, np.ndarray]:
    """OLS coefficients and Newey-West standard errors, intercept first."""
    n = len(target)
    with_const = np.column_stack([np.ones(n), design])
    coefficients = np.linalg.lstsq(with_const, target, rcond=None)[0]
    residual = target - with_const @ coefficients

    scores = with_const * residual[:, None]
    meat = scores.T @ scores
    for order in range(1, min(lag, n - 1) + 1):
        gamma = scores[order:].T @ scores[:-order]
        meat = meat + (1.0 - order / (lag + 1.0)) * (gamma + gamma.T)

    bread = np.linalg.pinv(with_const.T @ with_const)
    covariance = bread @ meat @ bread
    return coefficients, np.sqrt(np.clip(np.diag(covariance), 0.0, None))


def _automatic_lag(n: int) -> int:
    return int(np.floor(4.0 * (n / 100.0) ** (2.0 / 9.0))) if n > 0 else 0


def project(
    shock: pd.Series,
    response: pd.Series,
    *,
    link_id: str,
    horizons: tuple[int, ...],
    controls: pd.DataFrame | None = None,
    lags: int = DEFAULT_LAGS,
    note: str = "",
) -> Projection:
    """Regress the cumulative response over ``h`` periods on the shock at ``t``.

    ``shock`` and ``response`` are already-differenced series — this function does not
    transform anything, because which transform a series needs is a property of the
    series and belongs where the series is declared.
    """
    if not horizons:
        raise ProjectionError(f"{link_id}: no horizons to project over")

    frame = pd.concat([shock.rename("__shock__"), response.rename("__response__")], axis=1)
    if controls is not None and not controls.empty:
        frame = pd.concat([frame, controls], axis=1)
    control_names = [name for name in frame.columns if not name.startswith("__")]

    for order in range(1, lags + 1):
        frame[f"__shock_lag{order}__"] = frame["__shock__"].shift(order)
        frame[f"__response_lag{order}__"] = frame["__response__"].shift(order)

    regressors = [
        "__shock__",
        *(
            f"__{name}_lag{order}__"
            for name in ("shock", "response")
            for order in range(1, lags + 1)
        ),
        *control_names,
    ]

    responses: list[Response] = []
    for horizon in horizons:
        # The cumulative move from t to t+h. The response series is already a per-period
        # change, so the cumulative response is its forward-looking rolling sum — which
        # is what makes the residual an MA(h) and the HAC truncation below mandatory.
        cumulative = (
            frame["__response__"].rolling(horizon).sum().shift(-horizon)
            if horizon > 0
            else frame["__response__"]
        )
        block = frame[regressors].assign(__y__=cumulative).dropna()
        if len(block) <= len(regressors) + 2:
            continue

        design = block[regressors].to_numpy(dtype=float)
        target = block["__y__"].to_numpy(dtype=float)
        truncation = max(horizon, _automatic_lag(len(block)))
        coefficients, errors = hac_errors(design, target, lag=truncation)

        responses.append(
            Response(
                horizon=horizon,
                # Index 1: index 0 is the intercept, and the shock is the first regressor.
                coefficient=float(coefficients[1]),
                std_error=float(errors[1]),
                observations=len(block),
                hac_lag=truncation,
            )
        )

    if not responses:
        raise ProjectionError(
            f"{link_id}: no horizon had enough overlapping observations to estimate"
        )

    return Projection(
        link_id=link_id,
        responses=tuple(responses),
        controls=tuple(control_names),
        lags=lags,
        note=note,
    )


def coefficients_only(
    shock: pd.Series,
    response: pd.Series,
    *,
    horizons: tuple[int, ...],
    controls: pd.DataFrame | None = None,
    lags: int = DEFAULT_LAGS,
) -> dict[int, float]:
    """Point estimates per horizon and nothing else — the bootstrap's inner loop.

    Split from :func:`project` because a bootstrap replicate needs the coefficient and
    not the standard error, and computing a HAC covariance two thousand times to throw
    it away is most of the cost of the compounded band.
    """
    frame = pd.concat([shock.rename("__shock__"), response.rename("__response__")], axis=1)
    if controls is not None and not controls.empty:
        frame = pd.concat([frame, controls], axis=1)
    control_names = [name for name in frame.columns if not name.startswith("__")]

    for order in range(1, lags + 1):
        frame[f"__shock_lag{order}__"] = frame["__shock__"].shift(order)
        frame[f"__response_lag{order}__"] = frame["__response__"].shift(order)

    regressors = [
        "__shock__",
        *(
            f"__{name}_lag{order}__"
            for name in ("shock", "response")
            for order in range(1, lags + 1)
        ),
        *control_names,
    ]

    out: dict[int, float] = {}
    for horizon in horizons:
        cumulative = (
            frame["__response__"].rolling(horizon).sum().shift(-horizon)
            if horizon > 0
            else frame["__response__"]
        )
        block = frame[regressors].assign(__y__=cumulative).dropna()
        if len(block) <= len(regressors) + 2:
            continue
        design = np.column_stack([np.ones(len(block)), block[regressors].to_numpy(dtype=float)])
        fitted = np.linalg.lstsq(design, block["__y__"].to_numpy(dtype=float), rcond=None)[0]
        out[horizon] = float(fitted[1])
    return out
