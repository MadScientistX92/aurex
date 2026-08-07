"""Is a skill score distinguishable from zero, and does it have the shape theory predicts?

§0 says a model must beat the random walk on out-of-sample CRPS, and that models which
lose ship labelled as losing. Neither half of that survives without a test: "+4.6%" and
"worth approximately nothing" are both claims about a population, made from a sample of
a few hundred overlapping windows, and a mean loss differential of either sign is worth
nothing on its own.

**Diebold-Mariano on the CRPS differential.** The statistic is the mean of
``d_t = crps_model - crps_null`` over its own standard error. Negative ``d`` means the
model lost less than the null, so a *negative* DM statistic is the model winning — the
sign convention is the opposite of the skill score's, which is why both are reported
side by side and the reading is spelled out in the artifact rather than left to be
inferred.

**The variance is HAC, because the windows overlap.** Forecasting every ``step``
sessions over a longer horizon makes consecutive differentials share most of their path,
and the loss differential inherits that as a moving-average dependence. A plain standard
error would divide by an observation count the series does not have. The Newey-West
estimator with Bartlett weights handles it, and the Bartlett kernel is used rather than
the rectangular weights of Diebold and Mariano's own paper because the rectangular form
can return a negative variance — a number no p-value can be built from, arriving exactly
when the data are least cooperative.

**The truncation lag is the overlap, measured in records.** The textbook rule for an
``h``-step-ahead forecast is ``h - 1``, and that rule assumes a forecast made every
session, where an ``h``-session window overlaps the previous ``h - 1`` of them. This
harness forecasts every ``step`` sessions, so two records overlap when they are fewer
than ``h / step`` records apart: the dependence is MA(``stride - 1``) in record time,
where ``stride`` is :attr:`~aurex.score.sampling.Sampling.stride`. At ``step = 1`` that
is exactly ``h - 1`` and the textbook rule falls out. At ``step = 5`` and ``h = 63`` it
is 12 rather than 62, and using 62 would spend fifty lags of a 568-record series
estimating autocovariances that are zero by construction.

**Harvey, Leybourne and Newbold's small-sample correction, always.** The DM statistic is
asymptotically standard normal and over-rejects badly at the sample sizes here,
especially at long horizons where the effective count is small. The correction scales the
statistic by ``sqrt((n + 1 - 2k + k(k-1)/n) / n)`` and refers it to a ``t`` distribution
on ``n - 1`` degrees of freedom. It is not optional and there is no uncorrected variant
exposed: an uncorrected p-value at 44 independent windows is the kind of number that
reads as evidence and is not.

**Overlap is declared, not assumed away.** Every other p-value in this package goes
through :func:`~aurex.score.sampling.require_independent`, which *refuses* an
overlapping series. DM is the one test that may see one, because it does not assume
independence — it models the dependence explicitly. So it takes the same
:class:`~aurex.score.sampling.Sampling` object and *derives* its lag from it, which
means it still cannot be run without declaring the sampling scheme; it just responds to
the declaration by widening its standard error instead of raising. It is then run a
second time on the thinned non-overlapping subsample, where the correction reduces
exactly to a paired ``t``-test, and both are published. Agreement between the two is
evidence the HAC lag was long enough; disagreement is a finding about the dependence
rather than a result to pick between.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy import stats

from aurex.score.sampling import Sampling

#: Bootstrap replicates for the shape test. Enough that the 2.5th percentile is not
#: itself noise, and cheap because each replicate is a handful of means.
DEFAULT_DRAWS = 4_000


@dataclass(frozen=True, slots=True)
class DieboldMariano:
    """One test of one skill score, or the reason there isn't one.

    ``statistic`` is the Harvey-Leybourne-Newbold-corrected DM statistic and
    ``p_value`` is two-sided from ``t(n - 1)``. There is no uncorrected pair of fields
    because there is no circumstance in this repository where the uncorrected number is
    the one to read.
    """

    #: Which null the model was compared against.
    null: str
    statistic: float | None
    p_value: float | None
    n: int
    #: Mean of ``crps_model - crps_null``. Negative means the model lost less.
    mean_differential: float
    #: Records that share a path, so 1 means disjoint windows. The MA order is one less.
    overlap: int
    hac_lag: int
    undefined_reason: str | None = None

    @property
    def favours_model(self) -> bool:
        """A negative differential is the model winning. Says nothing about significance."""
        return self.mean_differential < 0.0

    def describe(self) -> dict[str, Any]:
        return {
            "test": "diebold_mariano",
            "null_compared": self.null,
            "statistic": None if self.statistic is None else round(self.statistic, 4),
            "p_value": None if self.p_value is None else round(self.p_value, 4),
            "observations": self.n,
            "mean_loss_differential": round(self.mean_differential, 6),
            "variance_estimator": "newey_west_bartlett",
            "hac_truncation_lag": self.hac_lag,
            "overlapping_records": self.overlap,
            "small_sample_correction": "harvey_leybourne_newbold",
            "reference_distribution": f"t({self.n - 1})" if self.n > 1 else None,
            "undefined_reason": self.undefined_reason,
            "reading": (
                "The loss differential is crps_model - crps_null, so a negative "
                "statistic is the model losing less and a positive one is the model "
                "losing more. A large p-value means the skill score above is not "
                "distinguishable from zero on this sample, which is a statement about "
                "the evidence and not a demonstration that the two are equal."
            ),
        }


@dataclass(frozen=True, slots=True)
class HorizonLosses:
    """Per-date CRPS at one horizon, aligned across horizons by the caller.

    ``model`` and ``baseline`` are the same length and the same dates, because the
    shape test resamples *dates* and every horizon has to be read at the drawn ones.
    """

    horizon: int
    model: np.ndarray
    baseline: np.ndarray

    def __post_init__(self) -> None:
        if self.model.shape != self.baseline.shape:
            raise ValueError(
                f"model and baseline losses must align, got {self.model.shape} "
                f"and {self.baseline.shape} at horizon {self.horizon}"
            )

    @property
    def skill(self) -> float:
        baseline = float(np.mean(self.baseline))
        if baseline <= 0.0:
            raise ValueError(f"baseline CRPS must be positive, got {baseline}")
        return 1.0 - float(np.mean(self.model)) / baseline


@dataclass(frozen=True, slots=True)
class SkillDecay:
    """Skill regressed on the log of the horizon, tested by a block bootstrap.

    Theory has a prediction here and it is worth stating before the number: conditional
    variance mean-reverts to unconditional, so whatever a volatility model knows about
    the next week it knows less about the next quarter, and skill should decay with the
    horizon. A test of that shape is a different question from a test of any single
    horizon's skill, and it is the question that separates "helps at a week, washes out
    beyond" from "worth nothing anywhere".

    The p-value comes from a moving-block bootstrap over as-of *dates* rather than from
    the regression's own standard error. The five horizons are computed from the same
    forecasts on the same days, so they are not five independent observations of
    anything, and an OLS p-value across them would assume exactly the independence the
    design does not have. Resampling the shared dates in blocks long enough to carry the
    overlap keeps both dependencies intact.
    """

    slope: float
    intercept: float
    r_squared: float
    p_value: float
    #: Percentile bootstrap interval for the slope.
    interval: tuple[float, float]
    #: (horizon, skill) actually observed, in horizon order.
    points: tuple[tuple[int, float], ...]
    n_dates: int
    block: int
    draws: int

    @property
    def decays(self) -> bool:
        return self.slope < 0.0

    def describe(self) -> dict[str, Any]:
        return {
            "test": "skill_decay_with_horizon",
            "model": "skill = intercept + slope * log(horizon)",
            "slope": round(self.slope, 6),
            "intercept": round(self.intercept, 6),
            "r_squared": round(self.r_squared, 4),
            "p_value": round(self.p_value, 4),
            "slope_interval_95": [round(self.interval[0], 6), round(self.interval[1], 6)],
            "skill_by_horizon": [
                {"horizon_sessions": horizon, "skill": round(skill, 5)}
                for horizon, skill in self.points
            ],
            "method": "moving_block_bootstrap_over_as_of_dates",
            "dates": self.n_dates,
            "block_records": self.block,
            "draws": self.draws,
            "reading": (
                "A negative slope is skill falling as the horizon lengthens, which is "
                "what mean-reverting conditional variance predicts. The test is of the "
                "sign of the trend, not of the logarithm: log horizon is the regressor "
                "because the horizons are spaced geometrically, and a different "
                "monotone transform would move the slope without moving the finding."
            ),
        }


def _autocovariance(centred: np.ndarray, lag: int) -> float:
    """``gamma_k`` with the ``1/n`` divisor, matching the Newey-West convention."""
    if lag == 0:
        return float(np.dot(centred, centred) / centred.size)
    return float(np.dot(centred[lag:], centred[:-lag]) / centred.size)


def hln_factor(n: int, overlap: int) -> float:
    """``(n + 1 - 2k + k(k-1)/n) / n``, the square of the HLN scaling.

    Non-negative for ``overlap <= n`` and exactly zero at ``overlap == n``, which is the
    point where the sample contains one effectively independent window and there is
    nothing left to test.
    """
    return (n + 1 - 2 * overlap + overlap * (overlap - 1) / n) / n


def diebold_mariano(
    model: np.ndarray,
    baseline: np.ndarray,
    *,
    sampling: Sampling,
    null: str,
) -> DieboldMariano:
    """Test whether ``model`` and ``baseline`` have the same expected CRPS.

    ``sampling`` supplies the overlap, which sets both the HAC truncation lag and the
    small-sample correction. Passing the thinned series' own
    :meth:`~aurex.score.sampling.Sampling.independent` sampling collapses the lag to
    zero and the statistic to a paired ``t``-test, which is the cross-check rather than
    a different test.
    """
    losses = np.asarray(model, dtype=float)
    reference = np.asarray(baseline, dtype=float)
    if losses.shape != reference.shape:
        raise ValueError(f"loss series must align, got {losses.shape} and {reference.shape}")

    differential = losses - reference
    n = int(differential.size)
    overlap = sampling.stride
    lag = overlap - 1
    mean_differential = float(np.mean(differential)) if n else 0.0

    undefined = _undefined(n, overlap, lag)
    if undefined is not None:
        return DieboldMariano(
            null=null,
            statistic=None,
            p_value=None,
            n=n,
            mean_differential=mean_differential,
            overlap=overlap,
            hac_lag=lag,
            undefined_reason=undefined,
        )

    centred = differential - mean_differential
    variance = _autocovariance(centred, 0)
    for k in range(1, lag + 1):
        # Bartlett weights, which is what makes this Newey-West rather than the
        # rectangular form. They also guarantee the sum is non-negative, so the branch
        # below fires only on a degenerate series rather than on an unlucky one.
        variance += 2.0 * (1.0 - k / (lag + 1.0)) * _autocovariance(centred, k)

    if variance <= 0.0:
        return DieboldMariano(
            null=null,
            statistic=None,
            p_value=None,
            n=n,
            mean_differential=mean_differential,
            overlap=overlap,
            hac_lag=lag,
            undefined_reason=(
                "the loss differential has no variance, so the two forecasters scored "
                "identically on every window and there is nothing to test"
            ),
        )

    statistic = mean_differential / float(np.sqrt(variance / n))
    corrected = statistic * float(np.sqrt(hln_factor(n, overlap)))

    return DieboldMariano(
        null=null,
        statistic=corrected,
        p_value=float(2.0 * stats.t.sf(abs(corrected), df=n - 1)),
        n=n,
        mean_differential=mean_differential,
        overlap=overlap,
        hac_lag=lag,
    )


def _undefined(n: int, overlap: int, lag: int) -> str | None:
    if n < 2:
        return f"a test of a mean needs at least 2 observations, got {n}"
    if lag >= n:
        return (
            f"the windows overlap {overlap} records deep in a series of {n}, so there "
            f"is no lag at which the autocovariance is estimable"
        )
    if hln_factor(n, overlap) <= 0.0:
        return (
            f"{n} records overlapping {overlap} deep carry no independent window; the "
            f"small-sample correction is zero or negative, which is the arithmetic "
            f"saying the sample cannot support a test"
        )
    return None


def _log_linear_slope(horizons: np.ndarray, skills: np.ndarray) -> tuple[float, float]:
    """OLS of ``skills`` on ``log(horizons)``, returning ``(intercept, slope)``."""
    roots = np.log(horizons)
    centred = roots - roots.mean()
    slope = float(np.dot(centred, skills) / np.dot(centred, centred))
    return float(skills.mean() - slope * roots.mean()), slope


def skill_decay(
    points: tuple[HorizonLosses, ...],
    *,
    block: int,
    seed: int,
    draws: int = DEFAULT_DRAWS,
) -> SkillDecay | None:
    """Fit and bootstrap the slope of skill against log horizon.

    ``None`` below three horizons: two points determine a line exactly, so a
    two-horizon "trend" is arithmetic rather than evidence.
    """
    if len(points) < 3:
        return None

    ordered = tuple(sorted(points, key=lambda entry: entry.horizon))
    n_dates = int(ordered[0].model.size)
    if any(entry.model.size != n_dates for entry in ordered):
        raise ValueError("every horizon must be read at the same as-of dates to be resampled")
    if n_dates < 2 or block < 1:
        return None

    horizons = np.array([entry.horizon for entry in ordered], dtype=float)
    observed = np.array([entry.skill for entry in ordered], dtype=float)
    intercept, slope = _log_linear_slope(horizons, observed)

    fitted = intercept + slope * np.log(horizons)
    residual = float(np.sum((observed - fitted) ** 2))
    total = float(np.sum((observed - observed.mean()) ** 2))

    effective_block = min(block, n_dates)
    n_blocks = int(np.ceil(n_dates / effective_block))
    rng = np.random.default_rng(seed)
    slopes = np.empty(draws, dtype=float)

    model_losses = np.vstack([entry.model for entry in ordered])
    baseline_losses = np.vstack([entry.baseline for entry in ordered])

    for draw in range(draws):
        starts = rng.integers(0, n_dates, size=n_blocks)
        picks = (starts[:, None] + np.arange(effective_block)[None, :]) % n_dates
        index = picks.reshape(-1)[:n_dates]
        resampled_baseline = baseline_losses[:, index].mean(axis=1)
        resampled_model = model_losses[:, index].mean(axis=1)
        # A resampled baseline of zero would need every drawn window to have been
        # forecast exactly; treat it as the degenerate draw it is rather than dividing.
        with np.errstate(divide="ignore", invalid="ignore"):
            resampled_skill = 1.0 - resampled_model / resampled_baseline
        if not np.all(np.isfinite(resampled_skill)):
            slopes[draw] = np.nan
            continue
        slopes[draw] = _log_linear_slope(horizons, resampled_skill)[1]

    usable = slopes[np.isfinite(slopes)]
    if usable.size < 2:
        return None

    below = float(np.count_nonzero(usable <= 0.0)) / usable.size
    above = float(np.count_nonzero(usable >= 0.0)) / usable.size
    p_value = min(1.0, 2.0 * min(below, above))
    # A bootstrap cannot resolve a p-value below one over its own draw count, and
    # reporting 0.0 would claim a precision the resampling never had.
    p_value = max(p_value, 1.0 / usable.size)

    return SkillDecay(
        slope=slope,
        intercept=intercept,
        r_squared=1.0 - residual / total if total > 0.0 else 0.0,
        p_value=p_value,
        interval=(
            float(np.quantile(usable, 0.025)),
            float(np.quantile(usable, 0.975)),
        ),
        points=tuple((int(entry.horizon), entry.skill) for entry in ordered),
        n_dates=n_dates,
        block=effective_block,
        draws=int(usable.size),
    )
