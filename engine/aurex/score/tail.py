"""Did a rare event happen more often than the forecast said it would?

The reliability machinery reports ``mean_forecast`` against ``base_rate`` and calls the
gap a bias. That gap is a difference of two rates on a handful of events, and a
difference of two rates is not a result — the same standard :mod:`aurex.score.significance`
applies to a skill score applies here. Five hurdle clears in 579 windows against a
forecast of 0.79% is either a well-shaped tail or a coincidence, and only a test
separates them.

**The null is a sequence of probabilities, not one rate.** Each window carries its own
forecast probability, so the count of events is a sum of independent Bernoulli draws
with *different* parameters. That is a Poisson-binomial, and its distribution is exact
by convolution at any sample size this repository will ever see. The Poisson
approximation is reported beside it because it is the number a reader is likely to
compute by hand, and where the two disagree the exact one is the one to read.

**The test is one-sided by default, in the direction the claim makes.** "The empirical
tail is fatter than a Gaussian one" predicts *more* events than the thin-tailed
reference forecasts, so the p-value that bears on it is the upper tail. The lower tail
is reported too, because a model that over-forecasts a rare event is also wrong and
the asymmetric presentation is how that gets missed.

**Four references, because "is the tail the right shape" and "did the published number
hold up" are different questions.** Each is the same arithmetic on a different sequence
of probabilities:

* ``model`` — what the engine actually forecast, window by window.
* ``gaussian_matched_sigma`` — a driftless Gaussian given the engine's *own* forecast
  standard deviation for that window. Second moment held fixed, so any difference in
  the event probability is shape and nothing else. This is the reference that answers
  the shape question cleanly.
* ``gaussian_expanding_sigma`` — a driftless Gaussian whose volatility is the standard
  deviation of log returns over the history available at the as-of date. No lookahead,
  which makes it the benchmark a real-time Gaussian forecaster would have had.
* ``gaussian_sample_sigma`` — a driftless Gaussian at the realised standard deviation of
  the *scored window*. This is the benchmark the README published, and it uses a number
  nobody had at forecast time. It is kept because it is the claim under test, and it is
  labelled as carrying lookahead because it does.

**Only terminal events get a Gaussian reference.** A closed form for "closed through a
level at some point on the way" is a barrier-crossing probability, and the reflection
principle prices continuous monitoring. This engine monitors at session close on both
sides — see :class:`~aurex.score.events.RealisedPath` — so a continuous-monitoring
formula would hand the reference a probability for an event neither the forecast nor
the outcome was measured on. Path events get the model test alone.

**Both samples are published.** The overlapping series carries the observations; the
thinned one carries the independence every other p-value in this package requires. A
rejection on the full sample that does not survive thinning is not a result to pick
between — it is the finding, and it says the effect is the right size and the sample
is not long enough to prove it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
from scipy import stats

from aurex.score.sampling import Sampling

#: Which series a test was computed on. The overlapping one has the observations; the
#: thinned one has the independence.
TailSample = Literal["all_windows", "non_overlapping"]

#: Reference label -> whether its volatility input was available at forecast time.
#: Published beside every result, because a benchmark using the scored window's own
#: realised volatility is a stronger benchmark than the forecaster ever faced.
LOOKAHEAD_FREE: dict[str, bool] = {
    "model": True,
    "gaussian_matched_sigma": True,
    "gaussian_expanding_sigma": True,
    "gaussian_sample_sigma": False,
}


def poisson_binomial_pmf(probabilities: np.ndarray) -> np.ndarray:
    """Exact distribution of the number of successes, by convolution.

    ``O(n^2)`` and exact, which at the few hundred windows this package produces costs
    nothing worth optimising. The alternative — assuming a common rate and reaching for
    :func:`scipy.stats.binom` — would test a null the forecaster never stated, because
    the forecast probability moves with the conditional variance on every window.
    """
    values = np.asarray(probabilities, dtype=float)
    if values.ndim != 1:
        raise ValueError(f"probabilities must be one-dimensional, got shape {values.shape}")
    if np.any((values < 0.0) | (values > 1.0)):
        raise ValueError("probabilities must lie in [0, 1]")

    pmf = np.zeros(values.size + 1, dtype=float)
    pmf[0] = 1.0
    for index, probability in enumerate(values):
        window = pmf[: index + 2]
        # Right-to-left in one vectorised step: the shifted term is read before the
        # unshifted one is overwritten, because numpy evaluates the whole right side
        # before assigning any of it.
        window[1:] = window[1:] * (1.0 - probability) + window[:-1] * probability
        pmf[0] *= 1.0 - probability
    return pmf


@dataclass(frozen=True, slots=True)
class TailTest:
    """One count, tested against one sequence of forecast probabilities."""

    reference: str
    sample: TailSample
    n: int
    observed: int
    #: Sum of the forecast probabilities: how many events this reference expected.
    expected: float
    #: P(X >= observed) under the reference. The direction a fat-tail claim predicts.
    p_value_upper: float | None
    #: P(X <= observed), so an over-forecast is as visible as an under-forecast.
    p_value_lower: float | None
    #: P(X >= observed) under a Poisson with the same mean, for comparison only.
    poisson_upper: float | None
    #: True where the reference's volatility input was not available at forecast time.
    uses_lookahead: bool = False
    undefined_reason: str | None = None

    @property
    def mean_probability(self) -> float:
        return self.expected / self.n if self.n else 0.0

    @property
    def observed_rate(self) -> float:
        return self.observed / self.n if self.n else 0.0

    @property
    def rejects(self) -> bool:
        """Upper tail below 5%: more events than this reference can comfortably explain."""
        return self.p_value_upper is not None and self.p_value_upper < 0.05

    def describe(self) -> dict[str, Any]:
        def rounded(value: float | None, places: int) -> float | None:
            return None if value is None else round(value, places)

        return {
            "reference": self.reference,
            "sample": self.sample,
            "windows": self.n,
            "observed_events": self.observed,
            "expected_events": round(self.expected, 4),
            "mean_forecast_probability": round(self.mean_probability, 6),
            "observed_rate": round(self.observed_rate, 6),
            "p_value_upper": rounded(self.p_value_upper, 5),
            "p_value_lower": rounded(self.p_value_lower, 5),
            "poisson_approximation_upper": rounded(self.poisson_upper, 5),
            "rejects_at_5pct": self.rejects,
            "uses_lookahead": self.uses_lookahead,
            "method": "poisson_binomial_exact",
            "undefined_reason": self.undefined_reason,
        }


def tail_test(
    probabilities: np.ndarray,
    outcomes: np.ndarray,
    *,
    reference: str,
    sample: TailSample,
) -> TailTest:
    """Test an observed event count against the probabilities that forecast it."""
    forecast = np.asarray(probabilities, dtype=float)
    observed_flags = np.asarray(outcomes, dtype=float)
    if forecast.shape != observed_flags.shape:
        raise ValueError(
            f"forecasts and outcomes must align, got {forecast.shape} and {observed_flags.shape}"
        )

    n = int(forecast.size)
    observed = int(np.count_nonzero(observed_flags > 0.0))
    expected = float(np.sum(forecast))
    uses_lookahead = not LOOKAHEAD_FREE.get(reference, True)

    if n == 0:
        return TailTest(
            reference=reference,
            sample=sample,
            n=0,
            observed=0,
            expected=0.0,
            p_value_upper=None,
            p_value_lower=None,
            poisson_upper=None,
            uses_lookahead=uses_lookahead,
            undefined_reason="no windows at this horizon, so there is no count to test",
        )

    pmf = poisson_binomial_pmf(forecast)
    upper = float(np.sum(pmf[observed:]))
    lower = float(np.sum(pmf[: observed + 1]))

    return TailTest(
        reference=reference,
        sample=sample,
        n=n,
        observed=observed,
        expected=expected,
        # Clipped because a convolution of several hundred terms can land a hair
        # outside [0, 1], and a p-value of 1.0000000002 is a rounding artefact that
        # reads as a bug.
        p_value_upper=float(np.clip(upper, 0.0, 1.0)),
        p_value_lower=float(np.clip(lower, 0.0, 1.0)),
        poisson_upper=float(stats.poisson.sf(observed - 1, expected)) if expected > 0.0 else None,
        uses_lookahead=uses_lookahead,
    )


@dataclass(frozen=True, slots=True)
class TailCalibration:
    """Every reference, on both samples, for one event at one horizon."""

    event_id: str
    tests: tuple[TailTest, ...]

    def for_reference(self, reference: str, sample: TailSample) -> TailTest | None:
        for test in self.tests:
            if test.reference == reference and test.sample == sample:
                return test
        return None

    def describe(self) -> dict[str, Any]:
        return {
            "tail_test": {
                "tests": [test.describe() for test in self.tests],
                "reading": (
                    "Each row asks whether the number of events observed is consistent "
                    "with the probabilities one reference forecast for those same "
                    "windows. p_value_upper is one-sided in the direction a fat tail "
                    "predicts: small means more events happened than the reference can "
                    "comfortably explain. The count under a sequence of differing "
                    "probabilities is Poisson-binomial and computed exactly; the "
                    "Poisson figure beside it is the hand approximation, not a second "
                    "result. gaussian_matched_sigma holds the engine's own forecast "
                    "variance fixed, so a difference between it and the model is tail "
                    "shape rather than volatility level. Both samples are published "
                    "and a rejection that does not survive thinning has not been shown "
                    "on independent windows."
                ),
            }
        }


def driftless_gaussian_probability(
    multiples: np.ndarray,
    sigma_per_session: np.ndarray,
    *,
    horizon: int,
) -> np.ndarray:
    """P(terminal price >= anchor x multiple) under a driftless log-normal.

    The scale is log because that is the space the engine simulates in, and comparing a
    log-space model against a simple-return Gaussian would put a units difference into
    a result about tail shape. Zero or non-finite volatility yields ``nan``, which the
    caller drops rather than silently scoring as a probability of zero or a half.
    """
    scale = np.asarray(sigma_per_session, dtype=float) * float(np.sqrt(horizon))
    logged = np.log(np.asarray(multiples, dtype=float))
    with np.errstate(divide="ignore", invalid="ignore"):
        standardised = np.divide(logged, scale, out=np.full_like(logged, np.nan), where=scale > 0.0)
    result: np.ndarray = stats.norm.sf(standardised)
    return result


def tail_calibration(
    *,
    event_id: str,
    probabilities: np.ndarray,
    outcomes: np.ndarray,
    sampling: Sampling,
    gaussian_references: dict[str, np.ndarray] | None = None,
) -> TailCalibration:
    """Run every reference on the full series and on the thinned subsample.

    ``gaussian_references`` is empty for a path-monitored event, where no closed form
    matches the engine's own session-close convention. The model's own test is run
    regardless, because "did the engine forecast this event at the right rate" is
    answerable whatever the monitoring.
    """
    references: dict[str, np.ndarray] = {"model": np.asarray(probabilities, dtype=float)}
    references.update(gaussian_references or {})

    tests: list[TailTest] = []
    for reference, forecast in references.items():
        values = np.asarray(forecast, dtype=float)
        if values.size and not np.all(np.isfinite(values)):
            # A reference that could not be computed on every window would otherwise be
            # tested on a different sample from the model's, and the two counts would
            # not be comparable.
            continue
        tests.append(tail_test(values, outcomes, reference=reference, sample="all_windows"))
        tests.append(
            tail_test(
                sampling.thin(values),
                sampling.thin(np.asarray(outcomes)),
                reference=reference,
                sample="non_overlapping",
            )
        )

    return TailCalibration(event_id=event_id, tests=tuple(tests))
