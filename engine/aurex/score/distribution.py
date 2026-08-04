"""Scoring a whole distribution: the probability integral transform, and CRPS.

**PIT is randomised, because the forecast is an ensemble.** Under a perfect forecast
the observation is exchangeable with the simulated paths, so its rank among the
``M + 1`` values is uniform on ``{1, ..., M + 1}``. Taking ``F(y)`` from the empirical
CDF instead would land on the grid ``{0, 1/M, ..., 1}`` and pile mass on the endpoints
whenever the observation falls outside the simulated range — which a finite ensemble
guarantees it sometimes will. Spreading the rank uniformly across its own interval
gives a statistic that is exactly uniform for finite ``M`` rather than asymptotically.

**CRPS is the fair estimator by default.** The naive ensemble CRPS divides the pairwise
spread term by ``M²`` when only ``M(M-1)`` of those pairs are non-degenerate, so it
subtracts too little and scores an ensemble *worse* the smaller it is — by roughly
``1/M``. A larger ensemble therefore wins without forecasting better. The bias mostly
cancels in a skill score between two ensembles of the same size, which is exactly the
reason not to rely on it cancelling: the correction costs one denominator and removes
the incentive to win a benchmark by simulating more paths.

Both are computed in **price space**. Nothing here ever sees a return, let alone a
transform-space one, for the reason :mod:`aurex.assets.transforms` sets out at length.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy import stats

from aurex.score.sampling import Sampling, require_independent

#: PIT histogram resolution. Ten bins over twenty-odd years of weekly forecasts leaves
#: enough per bin to read a shape; more bins read as noise.
DEFAULT_PIT_BINS = 10


@dataclass(frozen=True, slots=True)
class UniformityTest:
    """Kolmogorov-Smirnov against the uniform, with the count that earned it."""

    statistic: float
    p_value: float
    n: int

    def describe(self) -> dict[str, Any]:
        return {
            "test": "kolmogorov_smirnov_vs_uniform",
            "statistic": round(self.statistic, 5),
            "p_value": round(self.p_value, 5),
            "independent_observations": self.n,
            "reading": (
                "A small p-value says the predicted distributions are the wrong shape. "
                "It does not say in which direction; read the histogram for that."
            ),
        }


@dataclass(frozen=True, slots=True)
class GoodnessOfFit:
    """Pearson chi-square of the PIT histogram against a flat one.

    KS is built on the largest gap in the *cumulative* distribution, which makes it
    sensitive to a shift and weak against mass concentrated in one bin: a single heavy
    bin barely moves the CDF anywhere. The chi-square reads each bin against its own
    expectation and sees exactly that pattern, so the two are run together and both are
    published. Neither is a substitute for looking at the histogram.
    """

    statistic: float
    p_value: float
    degrees_of_freedom: int
    n: int
    bins: tuple[int, ...]
    expected_per_bin: float

    @property
    def underpowered(self) -> bool:
        """Below five expected per bin the chi-square approximation gets unreliable."""
        return self.expected_per_bin < 5.0

    def describe(self) -> dict[str, Any]:
        return {
            "test": "pearson_chi_square_vs_uniform",
            "statistic": round(self.statistic, 5),
            "p_value": round(self.p_value, 5),
            "degrees_of_freedom": self.degrees_of_freedom,
            "independent_observations": self.n,
            "expected_per_bin": round(self.expected_per_bin, 2),
            "bins": list(self.bins),
            "validity_note": (
                f"Only {self.expected_per_bin:.1f} observations expected per bin; the "
                "chi-square approximation is unreliable below five and this p-value "
                "should not be read as a test."
                if self.underpowered
                else None
            ),
            "reading": (
                "Complements KS rather than replacing it: chi-square sees mass piled "
                "in one bin, which barely moves a cumulative statistic."
            ),
        }


def pit_value(forecast: np.ndarray, observed: float, *, rng: np.random.Generator) -> float:
    """Randomised PIT of one observation against one simulated ensemble.

    ``(below + U * (ties + 1)) / (M + 1)``, which is exactly uniform when the
    observation is exchangeable with the ensemble. See the module docstring for why
    the unrandomised form is not usable here.
    """
    sample = np.asarray(forecast, dtype=float)
    if sample.size == 0:
        raise ValueError("an empty ensemble has no PIT")
    below = float(np.count_nonzero(sample < observed))
    ties = float(np.count_nonzero(sample == observed))
    return float((below + rng.random() * (ties + 1.0)) / (sample.size + 1.0))


def pit_histogram(values: np.ndarray, *, bins: int = DEFAULT_PIT_BINS) -> list[int]:
    """Counts per equal-width bin over ``[0, 1]``.

    Descriptive, so it is computed on every observation including overlapping ones:
    a histogram of dependent draws is still an honest picture of where the mass sits,
    it just cannot support a p-value.
    """
    counts, _ = np.histogram(np.asarray(values, dtype=float), bins=bins, range=(0.0, 1.0))
    return [int(count) for count in counts]


def uniformity(values: np.ndarray, *, sampling: Sampling) -> UniformityTest:
    """KS test of the PIT values against ``U(0, 1)``. Independent observations only."""
    require_independent(sampling, "the PIT uniformity test")
    sample = np.asarray(values, dtype=float)
    if sample.size < 2:
        raise ValueError(f"need at least 2 observations for a KS test, got {sample.size}")
    result = stats.kstest(sample, "uniform")
    return UniformityTest(
        statistic=float(result.statistic), p_value=float(result.pvalue), n=int(sample.size)
    )


def chi_square_uniformity(
    values: np.ndarray, *, sampling: Sampling, bins: int = DEFAULT_PIT_BINS
) -> GoodnessOfFit:
    """Pearson chi-square of the PIT histogram against flat. Independent observations only."""
    require_independent(sampling, "the PIT chi-square goodness-of-fit test")
    sample = np.asarray(values, dtype=float)
    if sample.size < 2:
        raise ValueError(f"need at least 2 observations for a chi-square, got {sample.size}")

    counts = np.asarray(pit_histogram(sample, bins=bins), dtype=float)
    expected = sample.size / bins
    statistic = float(np.sum((counts - expected) ** 2 / expected))

    return GoodnessOfFit(
        statistic=statistic,
        p_value=float(stats.chi2.sf(statistic, df=bins - 1)),
        degrees_of_freedom=bins - 1,
        n=int(sample.size),
        bins=tuple(int(count) for count in counts),
        expected_per_bin=expected,
    )


def crps(forecast: np.ndarray, observed: float, *, fair: bool = True) -> float:
    """Continuous ranked probability score of an ensemble against one observation.

    Computed from the sorted ensemble in ``O(M log M)``; the pairwise-difference form
    is the same number and is quadratic. Lower is better, and the score is in the
    units of the observation, so a CRPS is a price.
    """
    sample = np.sort(np.asarray(forecast, dtype=float))
    n = sample.size
    if n == 0:
        raise ValueError("an empty ensemble has no CRPS")

    absolute = float(np.mean(np.abs(sample - observed)))
    if n == 1:
        # One path is a point forecast, and CRPS collapses to absolute error. There is
        # no spread term to correct, fair or otherwise.
        return absolute

    ranks = np.arange(1.0, n + 1.0)
    spread = float(np.sum((2.0 * ranks - n - 1.0) * sample))
    denominator = float(n * (n - 1)) if fair else float(n * n)
    return absolute - spread / denominator


def crps_skill(model: float, baseline: float) -> float:
    """``1 - model/baseline``. Positive means the model beat the null, zero means it tied.

    Undefined against a baseline of zero, which only happens if the null forecast every
    observation exactly; that is a broken input rather than a perfect null.
    """
    if baseline <= 0.0:
        raise ValueError(f"baseline CRPS must be positive to form a skill score, got {baseline}")
    return 1.0 - model / baseline
