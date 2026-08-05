"""GJR-GARCH(1,1,1): asymmetric conditional variance, fitted by bounded MLE.

.. code-block::

    sigma2_t = omega + (alpha + gamma * 1[eps_{t-1} < 0]) * eps2_{t-1} + beta * sigma2_{t-1}

``gamma`` is the asymmetry: the extra variance a negative shock contributes over a
positive one of the same size. It is estimated, never assumed — for some assets it
is the largest term in the model and for others it is indistinguishable from zero,
and which of those is true is a finding rather than an input. The fit reports it with
its standard error so a reader can see when it is not identified.

**Innovations are Student-t by default.** Standardised residuals of daily financial
returns are not normal, and a normal likelihood both understates the tails and biases
the variance parameters to compensate. The degrees of freedom are estimated jointly.
Note that filtered historical simulation does not resample from the fitted
distribution at all — it resamples the empirical residuals — so the innovation choice
affects the fitted parameters, not the shape of the simulated tail.

**Numerics.** Returns are scaled by 100 inside the fit, which is standard practice
for this likelihood: an unscaled daily variance is ~1e-4 and the optimiser's default
tolerances are not built for it. Parameters are reported unscaled. The in-sample
recursion is a first-order linear filter in ``sigma2`` driven by a known input, so it
runs through :func:`scipy.signal.lfilter` rather than a Python loop; the simulation
recursion cannot be, because there the input depends on the path.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np
import pandas as pd
from scipy import optimize, signal

from aurex.vol.base import (
    MeanSpec,
    conditional_mean,
    excluded_mask,
    require_observations,
)

#: Innovation distribution for the likelihood.
Innovation = Literal["t", "normal"]

#: Returns are multiplied by this inside the optimiser. See the module docstring.
_SCALE = 100.0

#: Persistence is bounded strictly below one: at one the unconditional variance does
#: not exist, and the simulation would have nothing to revert to.
_MAX_PERSISTENCE = 0.9995

#: Student-t below this is barely integrable and usually signals a fit failure.
_MIN_DF = 2.1
_MAX_DF = 60.0

#: A bounded optimiser can stop just short of its ceiling, so "at the bound" has to
#: mean "near enough that the distinction is numerical" — otherwise the flag reads
#: false at 59.99 and quietly implies evidence of fat tails that the fit never found.
_DF_BOUND_TOLERANCE = 1.0


@dataclass(frozen=True, slots=True)
class GjrGarchFit:
    """A fitted GJR-GARCH process. Parameters are in unscaled return units."""

    model_id: str
    mu: float
    omega: float
    alpha: float
    gamma: float
    beta: float
    innovation: Innovation
    df: float | None
    conditional_sigma: pd.Series
    standardized_residuals: pd.Series
    #: One-step-ahead conditional variance, the seed for every simulated path.
    next_variance: float
    log_likelihood: float
    n_observations: int
    n_excluded: int
    converged: bool
    mean_spec: MeanSpec
    standard_errors: dict[str, float] = field(default_factory=dict)

    @property
    def persistence(self) -> float:
        """``alpha + gamma/2 + beta`` — the decay rate of a variance shock.

        The ``gamma/2`` assumes shocks are symmetric about zero, which is what makes
        this the persistence of the *symmetrised* process rather than of any
        particular path.
        """
        return self.alpha + self.gamma / 2.0 + self.beta

    @property
    def unconditional_variance(self) -> float:
        return self.omega / (1.0 - self.persistence)

    def forward_sigma(self, horizon: int) -> np.ndarray:
        """Expected sigma path — reporting only, never the basis of a simulation."""
        if horizon < 1:
            raise ValueError(f"horizon must be positive, got {horizon}")
        long_run = self.unconditional_variance
        steps = np.arange(horizon)
        variances = long_run + self.persistence**steps * (self.next_variance - long_run)
        sigma: np.ndarray = np.sqrt(np.maximum(variances, 0.0))
        return sigma

    def propagate(self, shocks: np.ndarray) -> np.ndarray:
        """Variance recursion forward, one path per row of ``shocks``."""
        if shocks.ndim != 2:
            raise ValueError(f"shocks must be (n_paths, horizon), got {shocks.shape}")

        n_paths, horizon = shocks.shape
        variance = np.full(n_paths, self.next_variance, dtype=float)
        out = np.empty((n_paths, horizon), dtype=float)

        for step in range(horizon):
            eps = np.sqrt(variance) * shocks[:, step]
            out[:, step] = self.mu + eps
            asymmetry = self.alpha + self.gamma * (eps < 0.0)
            variance = self.omega + asymmetry * eps**2 + self.beta * variance

        return out

    def describe(self) -> dict[str, Any]:
        return {
            "model": self.model_id,
            "innovation": self.innovation,
            "df": self.df,
            # At the ceiling the t is numerically a normal: the data carried no
            # evidence of fat tails, which is worth saying rather than implying by
            # printing a large number.
            "df_at_bound": self.df is not None and self.df >= _MAX_DF - _DF_BOUND_TOLERANCE,
            "mean": self.mean_spec,
            "mu": self.mu,
            "params": {
                "omega": self.omega,
                "alpha": self.alpha,
                "gamma": self.gamma,
                "beta": self.beta,
            },
            "standard_errors": dict(self.standard_errors),
            "persistence": self.persistence,
            "log_likelihood": self.log_likelihood,
            "observations": self.n_observations,
            "excluded_observations": self.n_excluded,
            "converged": self.converged,
            "asymmetry_note": (
                "gamma is the extra variance contributed by a negative shock. It is "
                "estimated; compare it against its standard error before reading "
                "anything into it."
            ),
        }


@dataclass(frozen=True, slots=True)
class GjrGarch:
    """The GJR-GARCH(1,1,1) specification."""

    innovation: Innovation = "t"
    mean: MeanSpec = "zero"
    #: Below this the likelihood surface is too flat for the parameters to mean much.
    min_observations: int = 250
    id: str = "gjr_garch"
    #: Each path runs its own variance recursion over its own shocks, so two paths
    #: that diverge see different volatility from then on. The only model here that
    #: does, and therefore the only one usable where the path is the question.
    per_path_variance: bool = True

    def fit(
        self,
        returns: pd.Series,
        *,
        realised_variance: pd.Series | None = None,
        exclude: Iterable[pd.Timestamp] = (),
    ) -> GjrGarchFit:
        del realised_variance  # This model reads the return series alone.

        clean = returns.dropna().astype(float)
        usable = excluded_mask(clean.index, exclude)
        require_observations(int(usable.sum()), self.min_observations, self.id)

        values = clean.to_numpy() * _SCALE
        mu = conditional_mean(values[usable], self.mean)
        eps = values - mu
        # Excluded observations enter the recursion as no news at all, so a policy
        # step decays out of the variance instead of being read as a shock.
        news_source = np.where(usable, eps, 0.0)
        backcast = float(np.var(eps[usable]))

        theta, log_likelihood, converged = _maximise(
            news_source, eps, usable, backcast, self.innovation
        )
        omega, alpha, gamma, beta = theta[:4]
        df = float(theta[4]) if self.innovation == "t" else None

        variance = _variance_path(news_source, theta[:4], backcast)
        sigma = np.sqrt(variance[:-1])
        standardized = eps / sigma

        errors = _standard_errors(theta, news_source, eps, usable, backcast, self.innovation)

        index = clean.index
        return GjrGarchFit(
            model_id=self.id,
            mu=mu / _SCALE,
            omega=omega / _SCALE**2,
            alpha=alpha,
            gamma=gamma,
            beta=beta,
            innovation=self.innovation,
            df=df,
            conditional_sigma=pd.Series(sigma / _SCALE, index=index, name="sigma"),
            # Excluded dates are dropped rather than kept at zero: FHS resamples this
            # series, and a residual that was never fitted is not a shock that happened.
            standardized_residuals=pd.Series(standardized, index=index, name="z")[usable],
            next_variance=float(variance[-1]) / _SCALE**2,
            log_likelihood=log_likelihood,
            n_observations=int(usable.sum()),
            n_excluded=int((~usable).sum()),
            converged=converged,
            mean_spec=self.mean,
            standard_errors=errors,
        )

    def describe(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "innovation": self.innovation,
            "mean": self.mean,
            "min_observations": self.min_observations,
            "equation": (
                "sigma2_t = omega + (alpha + gamma*1[eps_{t-1}<0])*eps2_{t-1} + beta*sigma2_{t-1}"
            ),
        }


def _variance_path(
    news_source: np.ndarray, params: Sequence[float] | np.ndarray, backcast: float
) -> np.ndarray:
    """Conditional variances for every observation, plus the one-step-ahead forecast.

    Length is ``n + 1``: element ``t`` is the variance for observation ``t``, and the
    last element is the forecast for the first unobserved period.
    """
    omega, alpha, gamma, beta = params
    # news[k] is the shock entering sigma2_{k+1}, so the last element produces the
    # one-step-ahead forecast and the array is one longer than the sample.
    news = (alpha + gamma * (news_source < 0.0)) * news_source**2

    # sigma2_t = (omega + news_t) + beta * sigma2_{t-1} is an AR(1) in sigma2 with a
    # known input, so the whole in-sample path is one linear filter pass.
    driven: np.ndarray = signal.lfilter(
        [1.0], [1.0, -beta], omega + news, zi=np.array([beta * backcast])
    )[0]
    return np.concatenate(([backcast], driven))


def _negative_log_likelihood(
    theta: np.ndarray,
    news_source: np.ndarray,
    eps: np.ndarray,
    usable: np.ndarray,
    backcast: float,
    innovation: Innovation,
) -> float:
    variance = _variance_path(news_source, theta[:4], backcast)[:-1]
    if not np.all(np.isfinite(variance)) or np.any(variance <= 0.0):
        return 1e12

    z2 = eps[usable] ** 2 / variance[usable]
    log_var = np.log(variance[usable])

    if innovation == "normal":
        total = -0.5 * np.sum(np.log(2.0 * np.pi) + log_var + z2)
    else:
        df = float(theta[4])
        const = (
            math.lgamma((df + 1.0) / 2.0)
            - math.lgamma(df / 2.0)
            - 0.5 * math.log(math.pi * (df - 2.0))
        )
        total = np.sum(const - 0.5 * log_var - (df + 1.0) / 2.0 * np.log1p(z2 / (df - 2.0)))

    return float(-total) if np.isfinite(total) else 1e12


def _starting_points(sample_variance: float, innovation: Innovation) -> list[np.ndarray]:
    """A few starts, because this likelihood has a flat ridge in (alpha, beta)."""
    grid = [(0.03, 0.06, 0.88), (0.08, 0.04, 0.85), (0.05, 0.00, 0.90), (0.12, 0.10, 0.70)]
    starts = []
    for alpha, gamma, beta in grid:
        omega = max(sample_variance * (1.0 - (alpha + gamma / 2.0 + beta)), 1e-8)
        theta = [omega, alpha, gamma, beta]
        if innovation == "t":
            theta.append(8.0)
        starts.append(np.array(theta, dtype=float))
    return starts


def _bounds(sample_variance: float, innovation: Innovation) -> list[tuple[float, float]]:
    bounds = [
        (1e-10, max(10.0 * sample_variance, 1e-6)),
        (0.0, 0.5),  # alpha
        (-0.5, 0.9),  # gamma: the sign is measured, so negatives stay reachable
        (0.0, 0.9995),  # beta
    ]
    if innovation == "t":
        bounds.append((_MIN_DF, _MAX_DF))
    return bounds


def _constraints() -> list[dict[str, Any]]:
    return [
        # Variance stays positive after a negative shock.
        {"type": "ineq", "fun": lambda t: t[1] + t[2]},
        # Stationarity, so an unconditional variance exists to revert to.
        {"type": "ineq", "fun": lambda t: _MAX_PERSISTENCE - (t[1] + t[2] / 2.0 + t[3])},
    ]


def _maximise(
    news_source: np.ndarray,
    eps: np.ndarray,
    usable: np.ndarray,
    backcast: float,
    innovation: Innovation,
) -> tuple[np.ndarray, float, bool]:
    args = (news_source, eps, usable, backcast, innovation)
    bounds = _bounds(backcast, innovation)
    constraints = _constraints()

    best: np.ndarray | None = None
    best_value = np.inf
    converged = False

    for start in _starting_points(backcast, innovation):
        result = optimize.minimize(
            _negative_log_likelihood,
            start,
            args=args,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"maxiter": 500, "ftol": 1e-10},
        )
        if result.fun < best_value:
            best, best_value, converged = result.x, float(result.fun), bool(result.success)

    if best is None:  # pragma: no cover - minimize always returns something
        raise RuntimeError("the optimiser returned no result")
    return best, -best_value, converged


def _standard_errors(
    theta: np.ndarray,
    news_source: np.ndarray,
    eps: np.ndarray,
    usable: np.ndarray,
    backcast: float,
    innovation: Innovation,
) -> dict[str, float]:
    """Numerical-Hessian errors, on the scaled parameters except ``omega``.

    Reported so ``gamma`` can be read against its own uncertainty rather than by its
    sign. Where the Hessian is not positive definite the fit is at a bound and the
    errors are omitted rather than fabricated.
    """
    names = ["omega", "alpha", "gamma", "beta"] + (["df"] if innovation == "t" else [])

    def objective(candidate: np.ndarray) -> float:
        return _negative_log_likelihood(candidate, news_source, eps, usable, backcast, innovation)

    step = np.maximum(np.abs(theta) * 1e-4, 1e-7)
    size = len(theta)
    hessian = np.empty((size, size))
    for i in range(size):
        for j in range(size):
            shift_i, shift_j = np.zeros(size), np.zeros(size)
            shift_i[i], shift_j[j] = step[i], step[j]
            hessian[i, j] = (
                objective(theta + shift_i + shift_j)
                - objective(theta + shift_i - shift_j)
                - objective(theta - shift_i + shift_j)
                + objective(theta - shift_i - shift_j)
            ) / (4.0 * step[i] * step[j])

    try:
        covariance = np.linalg.inv((hessian + hessian.T) / 2.0)
    except np.linalg.LinAlgError:
        return {}

    variances = np.diag(covariance)
    if np.any(variances <= 0.0) or not np.all(np.isfinite(variances)):
        return {}

    errors = np.sqrt(variances)
    scaled = {name: float(err) for name, err in zip(names, errors, strict=True)}
    scaled["omega"] = scaled["omega"] / _SCALE**2
    return scaled
