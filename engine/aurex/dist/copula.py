"""A bivariate t-copula, for the dependence a correlation cannot express.

Two series can share a correlation of 0.3 and behave completely differently in the
week that matters. A Gaussian dependence structure has zero tail dependence by
construction: however strong the correlation, the probability that both series hit
their extremes together goes to zero as you go further out. A t-copula does not, and
which of the two describes a given pair is an estimated question — the degrees of
freedom answer it, and a large fitted value *is* the Gaussian case, arrived at
honestly rather than assumed.

This matters wherever a price in one currency is a price in another times an exchange
rate: the buyer's-currency distribution is not the product of two independent
marginals, and it is not the product of two jointly-normal ones either. Their
variances do not add, and the pair can offset as easily as compound.

**Margins stay empirical.** The copula supplies dependence and nothing else. Each
series' shocks come from its own standardised residuals through
:func:`joint_shocks`, so the marginal tail is still the observed one — the copula
never gets to reshape it.

**Two dependence modes, because one of them is the check on the other.**
``t_copula`` fits the parametric structure and can therefore produce joint extremes
the sample never contained. ``synchronised`` resamples the same dates from both
series, which cannot invent tail co-movement but also cannot extrapolate it. Running
both and comparing is how you find out whether the copula is doing real work or
manufacturing dependence, and §6's discipline is to publish the disagreement.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import pandas as pd
from scipy import optimize, stats

from aurex.dist.fhs import ResidualPool, block_indices

#: How the two series' shocks are linked. See the module docstring.
DependenceMode = Literal["t_copula", "synchronised"]

_MIN_DF = 2.05
_MAX_DF = 60.0

#: A bounded optimiser approaches its ceiling without reaching it, so "at the bound"
#: has to mean "near enough that the distinction is numerical". Anything in this band
#: is a normal dependence structure by any reading.
_DF_BOUND_TOLERANCE = 1.0


@dataclass(frozen=True, slots=True)
class TCopula:
    """A fitted bivariate t-copula."""

    #: Dependence parameter, from Kendall's tau rather than a Pearson correlation:
    #: tau is invariant to the marginal transforms and robust to their tails.
    rho: float
    df: float
    kendall_tau: float
    n_observations: int
    log_likelihood: float

    @property
    def tail_dependence(self) -> float:
        """Probability of one series being extreme given the other is, in the limit.

        Zero for any Gaussian dependence, whatever the correlation. Positive here for
        any finite ``df``, which is the entire reason for preferring this family.
        """
        if self.rho <= -1.0:
            return 0.0
        argument = -math.sqrt((self.df + 1.0) * (1.0 - self.rho) / (1.0 + self.rho))
        return float(2.0 * stats.t.cdf(argument, df=self.df + 1.0))

    @property
    def df_at_bound(self) -> bool:
        """True when the fit ran to the ceiling: the pair showed no tail dependence."""
        return self.df >= _MAX_DF - _DF_BOUND_TOLERANCE

    def sample(self, n: int, rng: np.random.Generator) -> np.ndarray:
        """``(n, 2)`` uniforms carrying this copula's dependence."""
        correlation = np.array([[1.0, self.rho], [self.rho, 1.0]])
        normal = rng.multivariate_normal(np.zeros(2), correlation, size=n)
        # A t vector is a normal one scaled by a chi-square: the shared scale factor
        # is what puts mass in the joint tails.
        scale = np.sqrt(self.df / rng.chisquare(self.df, size=n))
        variates = normal * scale[:, np.newaxis]
        return np.asarray(stats.t.cdf(variates, df=self.df), dtype=float)

    def describe(self) -> dict[str, Any]:
        return {
            "family": "student_t",
            "rho": self.rho,
            "df": self.df,
            "df_at_bound": self.df_at_bound,
            "kendall_tau": self.kendall_tau,
            "tail_dependence": self.tail_dependence,
            "observations": self.n_observations,
            "log_likelihood": self.log_likelihood,
            "note": (
                "Dependence only: marginal shocks are resampled from each series' own "
                "standardised residuals. A df at the ceiling means the pair showed no "
                "evidence of joint tail behaviour beyond the Gaussian case."
            ),
        }


def pseudo_observations(values: np.ndarray) -> np.ndarray:
    """Ranks mapped into (0, 1), the copula's view of a sample.

    Dividing by ``n + 1`` rather than ``n`` keeps the largest observation off the
    boundary, where the inverse t is infinite.
    """
    ranks = stats.rankdata(values, method="average")
    scaled: np.ndarray = ranks / (len(values) + 1.0)
    return scaled


def fit_t_copula(first: pd.Series, second: pd.Series) -> TCopula:
    """Fit to the overlapping observations of two series.

    Alignment is an inner join on the index: a copula fitted on dates where only one
    series traded is measuring a holiday calendar, not a dependence.
    """
    joined = pd.concat({"a": first, "b": second}, axis=1, join="inner").dropna()
    if len(joined) < 30:
        raise ValueError(
            f"a copula fitted on {len(joined)} overlapping observations says nothing; "
            "need at least 30"
        )

    left = joined["a"].to_numpy(dtype=float)
    right = joined["b"].to_numpy(dtype=float)

    tau = float(stats.kendalltau(left, right).statistic)
    rho = float(np.sin(math.pi * tau / 2.0))
    rho = float(np.clip(rho, -0.999, 0.999))

    u = np.column_stack([pseudo_observations(left), pseudo_observations(right)])
    result = optimize.minimize_scalar(
        lambda df: -_copula_log_likelihood(u, rho, df),
        bounds=(_MIN_DF, _MAX_DF),
        method="bounded",
        options={"xatol": 1e-3},
    )
    df = float(result.x)

    return TCopula(
        rho=rho,
        df=df,
        kendall_tau=tau,
        n_observations=len(joined),
        log_likelihood=float(_copula_log_likelihood(u, rho, df)),
    )


def _copula_log_likelihood(u: np.ndarray, rho: float, df: float) -> float:
    """Log density of the bivariate t-copula at pseudo-observations ``u``."""
    x = stats.t.ppf(u, df=df)
    if not np.all(np.isfinite(x)):
        return -np.inf

    x1, x2 = x[:, 0], x[:, 1]
    determinant = 1.0 - rho**2
    quadratic = (x1**2 - 2.0 * rho * x1 * x2 + x2**2) / determinant

    joint = (
        math.lgamma((df + 2.0) / 2.0)
        - math.lgamma(df / 2.0)
        - math.log(math.pi * df)
        - 0.5 * math.log(determinant)
        - (df + 2.0) / 2.0 * np.log1p(quadratic / df)
    )
    marginal = (
        math.lgamma((df + 1.0) / 2.0)
        - math.lgamma(df / 2.0)
        - 0.5 * math.log(math.pi * df)
        - (df + 1.0) / 2.0 * np.log1p(x**2 / df)
    ).sum(axis=1)

    return float(np.sum(joint - marginal))


def joint_shocks(
    first: ResidualPool,
    second: ResidualPool,
    *,
    copula: TCopula | None,
    mode: DependenceMode,
    n_paths: int,
    horizon: int,
    block_length: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Two ``(n_paths, horizon)`` shock arrays that move together.

    ``synchronised`` block-bootstraps a single set of dates and reads both series at
    them, so serial *and* cross dependence are whatever the sample contained.
    ``t_copula`` draws linked uniforms and inverts each through its own empirical
    residual quantiles, which buys extrapolatable tail dependence at the cost of the
    block structure — the variance recursion still carries clustering, but a run of
    jointly bad days is no longer drawn as a run.

    Both pools must agree on their drift policy. A rupee price composed from a driftless
    metal and a drifting exchange rate is a directional forecast about the currency
    wearing a joint simulation's clothes, and the mismatch is easier to create than to
    notice — the two series are fitted in different places.
    """
    if first.demeaned != second.demeaned:
        raise ValueError(
            f"both legs of a joint simulation need the same drift policy, got "
            f"demeaned={first.demeaned} and demeaned={second.demeaned}"
        )

    if mode == "synchronised":
        aligned = pd.concat(
            {"a": first.residuals, "b": second.residuals}, axis=1, join="inner"
        ).dropna()
        if aligned.empty:
            raise ValueError("the two residual series do not overlap")
        picks = block_indices(
            len(aligned),
            n_paths=n_paths,
            horizon=horizon,
            block_length=block_length,
            rng=rng,
        )
        return aligned["a"].to_numpy()[picks], aligned["b"].to_numpy()[picks]

    if copula is None:
        raise ValueError("t_copula mode needs a fitted copula")

    uniforms = copula.sample(n_paths * horizon, rng).reshape(n_paths, horizon, 2)
    return (
        _empirical_quantile(first.values, uniforms[:, :, 0]),
        _empirical_quantile(second.values, uniforms[:, :, 1]),
    )


def _empirical_quantile(sample: np.ndarray, uniforms: np.ndarray) -> np.ndarray:
    """Invert a uniform through the empirical distribution of ``sample``.

    Nearest-rank rather than interpolated: interpolation between the two most extreme
    observations invents a value the sample never produced, which is exactly the part
    of the distribution filtered historical simulation exists to leave alone.
    """
    ordered = np.sort(sample[np.isfinite(sample)])
    positions = np.clip((uniforms * len(ordered)).astype(int), 0, len(ordered) - 1)
    drawn: np.ndarray = ordered[positions]
    return drawn
