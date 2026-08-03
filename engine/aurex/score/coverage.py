"""VaR breach tests: Kupiec on the rate, Christoffersen on the clustering.

A breach is an observation that fell below the forecast's own lower quantile. Two
things can be wrong with them and they are different failures:

* **Rate.** 5% of observations should breach the 5% quantile. Kupiec's proportion-of-
  failures test is a likelihood ratio against that binomial null.
* **Clustering.** They should also be *spread out*. A model that breaches five times in
  a fortnight and never again has the right rate and the wrong dynamics — it is not
  reacting to volatility, which is the one thing this engine claims to do.
  Christoffersen's independence test reads the first-order Markov transitions.

Both are chi-square likelihood ratios, and both assume independent observations, which
is why every function here takes a :class:`~aurex.score.sampling.Sampling` and refuses
an overlapping one.

**A test with no power says so.** At a 63-session horizon and a 1% quantile, twenty
years of non-overlapping windows expect well under one breach. The likelihood ratio is
then either undefined or a formality, and reporting ``p = 0.87`` from it would read as
evidence of calibration where there is only an absence of data. So every result here
carries its own observation count and, where the expected breach count is too low to
resolve anything, a note saying so.

**At zero breaches the chi-square approximation is not usable.** Kupiec's statistic is
asymptotically chi-square, and ``x = 0`` is a boundary of the parameter space where that
asymptotics does not hold — the likelihood ratio is computed against a maximum on the
edge of the simplex, and its null distribution is not chi-square on one degree of
freedom. The exact binomial is available and cheap, so both are always computed, and at
``x = 0`` or ``x = n`` the exact one is the reported p-value. The difference is not
cosmetic: 0 breaches in 44 windows against a 5% quantile gives a chi-square p of 0.034
and an exact p of 0.17. One of those reads as a rejection and the other does not, and
the exact one is right.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from scipy import special, stats

from aurex.score.sampling import Sampling, require_independent

#: Below this many expected breaches the test cannot distinguish a calibrated model
#: from a badly wrong one, so the result is labelled rather than left to be read as
#: evidence. Ten is the usual rule of thumb for a binomial normal approximation.
MIN_EXPECTED_BREACHES = 10.0


@dataclass(frozen=True, slots=True)
class TestResult:
    """One test, or the reason there isn't one. ``method`` says which p-value this is."""

    name: str
    statistic: float | None
    p_value: float | None
    n: int
    detail: dict[str, Any] = field(default_factory=dict)
    undefined_reason: str | None = None
    power_note: str | None = None
    #: Which null distribution the reported ``p_value`` came from. The statistic is
    #: always the likelihood ratio; at a boundary its chi-square p-value is not usable
    #: and the exact one is reported instead.
    method: str = "likelihood_ratio_chi2"

    def describe(self) -> dict[str, Any]:
        return {
            "test": self.name,
            "statistic": None if self.statistic is None else round(self.statistic, 5),
            "p_value": None if self.p_value is None else round(self.p_value, 5),
            "method": self.method,
            "independent_observations": self.n,
            "undefined_reason": self.undefined_reason,
            "power_note": self.power_note,
            **self.detail,
        }


def _log_likelihood(successes: float, trials: float, rate: float) -> float:
    """Binomial log-likelihood without the constant, safe at ``rate`` of 0 or 1."""
    failures = trials - successes
    return float(special.xlogy(successes, rate) + special.xlogy(failures, 1.0 - rate))


def _power_note(n: int, expected_rate: float) -> str | None:
    expected = n * expected_rate
    if expected >= MIN_EXPECTED_BREACHES:
        return None
    return (
        f"{n} independent observations at a {expected_rate:.0%} quantile expect "
        f"{expected:.1f} breaches. The test cannot resolve a miscalibration of "
        f"ordinary size at this count; read it as an absence of evidence, not as "
        f"evidence of calibration."
    )


def kupiec(breaches: np.ndarray, *, expected_rate: float, sampling: Sampling) -> TestResult:
    """Proportion-of-failures test: does the breach rate match the quantile?"""
    require_independent(sampling, "the Kupiec proportion-of-failures test")
    flags = np.asarray(breaches, dtype=bool)
    n = int(flags.size)
    observed = int(np.count_nonzero(flags))

    if n < 1:
        return TestResult(
            name="kupiec_pof",
            statistic=None,
            p_value=None,
            n=0,
            undefined_reason="no scored observations",
        )

    rate = observed / n
    statistic = -2.0 * (
        _log_likelihood(observed, n, expected_rate) - _log_likelihood(observed, n, rate)
    )
    # Floating point can push an exact tie a hair below zero, and a negative chi-square
    # statistic is a numerical artefact rather than a finding.
    statistic = max(statistic, 0.0)

    asymptotic = float(stats.chi2.sf(statistic, df=1))
    exact = float(stats.binomtest(observed, n, expected_rate).pvalue)
    # The boundary: the unrestricted maximum sits on the edge of the parameter space,
    # so the likelihood ratio is not chi-square on one degree of freedom there.
    at_boundary = observed == 0 or observed == n

    return TestResult(
        name="kupiec_pof",
        statistic=statistic,
        p_value=exact if at_boundary else asymptotic,
        method="exact_binomial" if at_boundary else "likelihood_ratio_chi2",
        n=n,
        detail={
            "expected_rate": expected_rate,
            "observed_rate": round(rate, 5),
            "breaches": observed,
            "expected_breaches": round(n * expected_rate, 2),
            "chi2_p_value": round(asymptotic, 5),
            "exact_binomial_p_value": round(exact, 5),
            # The directional reading, for a reader asking "too few, or too many?"
            "exact_lower_tail": round(float(stats.binom.cdf(observed, n, expected_rate)), 5),
            "boundary_note": (
                "Zero breaches puts the unrestricted estimate on the boundary of the "
                "parameter space, where the chi-square approximation does not hold. "
                "The exact binomial p-value is reported instead."
                if at_boundary
                else None
            ),
        },
        power_note=_power_note(n, expected_rate),
    )


def christoffersen_independence(
    breaches: np.ndarray, *, expected_rate: float, sampling: Sampling
) -> TestResult:
    """Are breaches independent of whether the previous window breached?"""
    require_independent(sampling, "the Christoffersen independence test")
    flags = np.asarray(breaches, dtype=bool)
    n = int(flags.size)

    if n < 2:
        return TestResult(
            name="christoffersen_independence",
            statistic=None,
            p_value=None,
            n=n,
            undefined_reason="fewer than two observations, so there are no transitions",
        )

    previous, current = flags[:-1], flags[1:]
    n00 = int(np.count_nonzero(~previous & ~current))
    n01 = int(np.count_nonzero(~previous & current))
    n10 = int(np.count_nonzero(previous & ~current))
    n11 = int(np.count_nonzero(previous & current))
    transitions = {"n00": n00, "n01": n01, "n10": n10, "n11": n11}

    if n10 + n11 == 0:
        return TestResult(
            name="christoffersen_independence",
            statistic=None,
            p_value=None,
            n=n,
            detail={"transitions": transitions},
            undefined_reason=(
                "no observation follows a breach, so the transition probability out of "
                "the breach state is not identified"
            ),
            power_note=_power_note(n, expected_rate),
        )

    total = n00 + n01 + n10 + n11
    pooled = (n01 + n11) / total
    from_calm = n01 / (n00 + n01) if n00 + n01 else 0.0
    from_breach = n11 / (n10 + n11)

    restricted = _log_likelihood(n01 + n11, total, pooled)
    unrestricted = _log_likelihood(n01, n00 + n01, from_calm) + _log_likelihood(
        n11, n10 + n11, from_breach
    )
    statistic = max(-2.0 * (restricted - unrestricted), 0.0)

    return TestResult(
        name="christoffersen_independence",
        statistic=statistic,
        p_value=float(stats.chi2.sf(statistic, df=1)),
        n=n,
        detail={
            "transitions": transitions,
            "breach_rate_after_calm": round(from_calm, 5),
            "breach_rate_after_breach": round(from_breach, 5),
        },
        power_note=_power_note(n, expected_rate),
    )


def conditional_coverage(
    breaches: np.ndarray, *, expected_rate: float, sampling: Sampling
) -> TestResult:
    """Kupiec and Christoffersen jointly — right rate *and* independent, on two df."""
    rate_test = kupiec(breaches, expected_rate=expected_rate, sampling=sampling)
    independence = christoffersen_independence(
        breaches, expected_rate=expected_rate, sampling=sampling
    )

    if rate_test.statistic is None or independence.statistic is None:
        return TestResult(
            name="christoffersen_conditional_coverage",
            statistic=None,
            p_value=None,
            n=rate_test.n,
            undefined_reason=(
                independence.undefined_reason or rate_test.undefined_reason or "undefined"
            ),
            power_note=rate_test.power_note,
        )

    statistic = rate_test.statistic + independence.statistic
    # No exact analogue exists for the joint statistic, so where the rate component sat
    # on a boundary the joint p-value inherits the same objection and says so.
    boundary = rate_test.method == "exact_binomial"

    return TestResult(
        name="christoffersen_conditional_coverage",
        statistic=statistic,
        p_value=float(stats.chi2.sf(statistic, df=2)),
        n=rate_test.n,
        detail={
            "components": {
                "kupiec": rate_test.statistic,
                "independence": independence.statistic,
            },
            "boundary_note": (
                "The rate component is a boundary case, so this joint chi-square "
                "p-value is not reliable either. There is no exact analogue for the "
                "joint statistic; read the two components separately."
                if boundary
                else None
            ),
        },
        power_note=rate_test.power_note,
    )
