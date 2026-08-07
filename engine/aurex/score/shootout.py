"""Six models against one null, without finding a winner that is not there.

Running six Diebold-Mariano tests at 5% and reporting whichever rejects is not a
shootout, it is a search. Under a true null of "nothing beats the random walk", the
chance that at least one of six independent tests rejects is about 26%, and the models
here are far from independent — they are fitted to the same series and their loss
differentials move together, which changes the number without changing the problem. A
leaderboard scored that way ranks noise.

**Hansen's Superior Predictive Ability test** answers the question the shootout actually
asks: *is there any model in this set that beats the benchmark?* One test, one p-value,
with the multiplicity handled inside the null distribution rather than by a correction
bolted on afterwards. The statistic is the largest studentised outperformance across
models, and its distribution comes from a moving-block bootstrap that preserves the
overlap between windows.

Hansen's recentring is the part worth understanding, because it is what separates SPA
from White's earlier Reality Check. A model that is far worse than the benchmark carries
no information about whether the *best* model beats it, but it still contributes to the
maximum's null distribution, and including it at its observed mean makes the test
conservative — add enough hopeless models and a real winner stops being detectable. SPA
recentres only those models close enough to the benchmark to matter, using the threshold
``-sqrt(2 * w_k^2 * log log n / n)``. Both the recentred (``consistent``) and the
uncentred (``conservative``) p-values are reported, because the gap between them is a
statement about how much of the result depends on that choice.

**The Model Confidence Set** answers the other question: *which models cannot be ruled
out?* It is not a ranking. It is a set, at a stated confidence level, with the property
that it contains the best model with that probability — and on a sample this size it is
usually large, which is the honest output rather than a disappointing one. A shootout
that reports "the MCS at 90% contains five of six models" has said something true and
useful; one that reports "NHITS wins" from the same data has not.

**The minimum detectable effect is published whether or not anything rejects.** Given
the loss-differential variance this sample actually has and the number of independent
windows it actually carries, there is a smallest CRPS skill the test could have found at
80% power. Without it, "no model beat the random walk" and "this sample could not have
detected a model that did" are the same sentence, and step 3a already established which
one this repository tends to be in. The MDE is computed from the same HAC variance and
the same small-sample correction the Diebold-Mariano test uses, so it describes the test
that was actually run rather than an idealised one.

**Sign convention, which is the opposite of Diebold-Mariano's here.** Throughout this
module a loss differential is ``benchmark - model``, so *positive* means the model lost
less and is therefore better. :mod:`aurex.score.significance` uses ``model - benchmark``
and reads a negative statistic as the model winning. The two are deliberately not
unified: DM's convention follows the literature it implements and so does this one, and
the artifact spells out the reading beside every number rather than relying on anyone
remembering which module produced it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy import stats

from aurex.score.sampling import Sampling
from aurex.score.significance import hln_factor

#: Bootstrap replicates. Enough that a 5% p-value is not resolved by fewer than a
#: hundred draws, and cheap because each replicate is a matrix of means.
DEFAULT_DRAWS = 5_000

#: Power the minimum detectable effect is computed at. Conventional, and fixed here
#: rather than exposed, because an MDE quoted at a power chosen after seeing the result
#: is a number tuned to look reachable.
DEFAULT_POWER = 0.80

#: Size the MDE and the confidence set are computed at.
DEFAULT_ALPHA = 0.05


class NonFiniteLossError(ValueError):
    """A forecaster's loss series contains a NaN or an infinity."""


def _require_finite(losses: dict[str, np.ndarray], benchmark: np.ndarray | None = None) -> None:
    """Refuse a loss series that is not finite, loudly and by name.

    Not defensive tidying. NaN propagates through every comparison as ``False``, so a
    single broken forecaster makes ``mean(bootstrap > statistic)`` evaluate to zero and
    the SPA test reports ``p = 0.000`` — the most significant result the test can
    produce, from a model that did not forecast anything. It is the exact shape of
    failure this repository refuses everywhere else: silent, and wrong in the direction
    that looks like a finding. A broken model must stop the run, not win it.
    """
    broken = sorted(name for name, values in losses.items() if not np.all(np.isfinite(values)))
    if benchmark is not None and not np.all(np.isfinite(benchmark)):
        broken.append("benchmark")
    if broken:
        raise NonFiniteLossError(
            f"non-finite CRPS from {broken}. A model that produced NaN did not forecast, "
            f"and scoring it would report a perfect p-value for an empty result. Fix the "
            f"forecaster or drop it from the set."
        )


def _hac_variance(centred: np.ndarray, lag: int) -> np.ndarray:
    """Newey-West with Bartlett weights, column-wise over a ``(n, k)`` matrix.

    The same estimator :func:`aurex.score.significance.diebold_mariano` uses, applied to
    every model at once so the SPA statistic studentises each column by the variance
    that column's own overlap implies.
    """
    n = centred.shape[0]
    variance = np.einsum("ij,ij->j", centred, centred) / n
    for step in range(1, lag + 1):
        weight = 1.0 - step / (lag + 1.0)
        covariance = np.einsum("ij,ij->j", centred[step:], centred[:-step]) / n
        variance = variance + 2.0 * weight * covariance
    return np.asarray(variance, dtype=float)


def _moving_block_indices(n: int, block: int, draws: int, rng: np.random.Generator) -> np.ndarray:
    """``(draws, n)`` of resampled positions, in blocks that carry the overlap."""
    block = max(1, min(block, n))
    count = int(np.ceil(n / block))
    starts = rng.integers(0, n, size=(draws, count))
    offsets = np.arange(block)
    picks = (starts[:, :, None] + offsets[None, None, :]) % n
    return picks.reshape(draws, -1)[:, :n]


@dataclass(frozen=True, slots=True)
class SuperiorPredictiveAbility:
    """One test of "does anything here beat the benchmark", over all models at once."""

    benchmark: str
    models: tuple[str, ...]
    statistic: float
    #: Hansen's recentred p-value. The one to read.
    p_value_consistent: float
    #: Uncentred, i.e. White's Reality Check. Conservative by construction.
    p_value_conservative: float
    #: Mean loss differential per model, benchmark minus model. Positive is better.
    mean_differentials: tuple[float, ...]
    #: Which model produced the maximum studentised statistic.
    best: str | None
    n: int
    block: int
    draws: int

    @property
    def rejects(self) -> bool:
        return self.p_value_consistent < DEFAULT_ALPHA

    def describe(self) -> dict[str, Any]:
        return {
            "test": "hansen_spa",
            "null": (f"no model in this set has a smaller expected CRPS than {self.benchmark}"),
            "statistic": round(self.statistic, 4),
            "p_value": round(self.p_value_consistent, 4),
            "p_value_conservative": round(self.p_value_conservative, 4),
            "rejects_at_5pct": self.rejects,
            "best_model": self.best,
            "models": list(self.models),
            "mean_loss_differential": {
                name: round(value, 6)
                for name, value in zip(self.models, self.mean_differentials, strict=True)
            },
            "observations": self.n,
            "bootstrap": {
                "method": "moving_block",
                "block_records": self.block,
                "draws": self.draws,
            },
            "reading": (
                "One test over the whole set, so the multiplicity is inside the null "
                "distribution rather than corrected for afterwards. A loss differential "
                "here is benchmark minus model, so positive means the model lost less — "
                "the opposite of the Diebold-Mariano sign convention reported per model. "
                "best_model names the largest studentised differential and is a "
                "description of this sample, not a winner: where the p-value does not "
                "reject, no model in the set has been shown to beat the benchmark and "
                "the name should be read as noise."
            ),
        }


def superior_predictive_ability(
    losses: dict[str, np.ndarray],
    benchmark_losses: np.ndarray,
    *,
    benchmark: str,
    sampling: Sampling,
    draws: int = DEFAULT_DRAWS,
    seed: int = 20260807,
) -> SuperiorPredictiveAbility | None:
    """Hansen's SPA over every model in ``losses`` against one benchmark.

    ``None`` where the sample cannot support the test at all — no models, or fewer
    observations than the bootstrap block needs.
    """
    names = tuple(sorted(losses))
    if not names:
        return None

    reference = np.asarray(benchmark_losses, dtype=float)
    _require_finite(losses, reference)
    n = int(reference.size)
    if n < 3:
        return None

    # benchmark - model, so a positive column mean is a model that lost less.
    differentials = np.column_stack(
        [reference - np.asarray(losses[name], dtype=float) for name in names]
    )
    means = differentials.mean(axis=0)

    block = max(1, sampling.stride)
    lag = block - 1
    variance = _hac_variance(differentials - means, lag)
    # A model scoring identically to the benchmark on every window has no variance to
    # studentise by. Floor it rather than divide: the alternative is an infinite
    # statistic from a model that demonstrated nothing.
    scale = np.sqrt(np.maximum(variance, 1e-300) / n)

    studentised = means / scale
    statistic = float(max(0.0, float(np.max(studentised))))
    best = names[int(np.argmax(studentised))] if float(np.max(studentised)) > 0.0 else None

    # Hansen's recentring: a model far enough below the benchmark contributes nothing to
    # whether the best one beats it, and leaving it in at its observed mean makes the
    # test conservative in proportion to how many bad models were entered.
    threshold = -np.sqrt(2.0 * variance * np.log(np.log(n)) / n) if n > 2 else np.zeros_like(means)
    recentred = np.where(means >= threshold, means, 0.0)

    rng = np.random.default_rng(seed)
    index = _moving_block_indices(n, block, draws, rng)
    resampled = differentials[index].mean(axis=1)

    consistent = np.max((resampled - recentred) / scale, axis=1)
    conservative = np.max((resampled - means) / scale, axis=1)

    return SuperiorPredictiveAbility(
        benchmark=benchmark,
        models=names,
        statistic=statistic,
        p_value_consistent=float(np.mean(np.maximum(consistent, 0.0) > statistic)),
        p_value_conservative=float(np.mean(np.maximum(conservative, 0.0) > statistic)),
        mean_differentials=tuple(float(value) for value in means),
        best=best,
        n=n,
        block=block,
        draws=draws,
    )


@dataclass(frozen=True, slots=True)
class ModelConfidenceSet:
    """The models that cannot be excluded at a stated confidence level."""

    included: tuple[str, ...]
    eliminated: tuple[tuple[str, float], ...]
    alpha: float
    n: int
    block: int
    draws: int

    def describe(self) -> dict[str, Any]:
        return {
            "test": "model_confidence_set",
            "confidence": round(1.0 - self.alpha, 3),
            "included": list(self.included),
            "eliminated": [
                {"model": name, "p_value": round(value, 4)} for name, value in self.eliminated
            ],
            "observations": self.n,
            "bootstrap": {
                "method": "moving_block",
                "block_records": self.block,
                "draws": self.draws,
                "statistic": "t_max",
            },
            "reading": (
                "A set, not a ranking. It contains the best model with the stated "
                "probability, and on a sample this size it is usually large — which is "
                "the measurement rather than a failure of it. Models are listed in the "
                "order they were eliminated, each with the p-value of the test that "
                "removed it. A set containing every model means the sample cannot tell "
                "them apart."
            ),
        }


def model_confidence_set(
    losses: dict[str, np.ndarray],
    *,
    sampling: Sampling,
    alpha: float = 0.10,
    draws: int = DEFAULT_DRAWS,
    seed: int = 20260807,
) -> ModelConfidenceSet | None:
    """Hansen, Lunde and Nason's MCS by the ``t_max`` statistic.

    Eliminates the worst model while the equal-predictive-ability hypothesis is rejected
    on the surviving set, and returns what is left. The bootstrap resamples in moving
    blocks, so the overlap between windows is carried into the null distribution the same
    way :func:`superior_predictive_ability` carries it.
    """
    names = sorted(losses)
    if len(names) < 2:
        return None

    _require_finite(losses)
    matrix = np.column_stack([np.asarray(losses[name], dtype=float) for name in names])
    n = int(matrix.shape[0])
    if n < 3:
        return None

    block = max(1, sampling.stride)
    rng = np.random.default_rng(seed)
    index = _moving_block_indices(n, block, draws, rng)

    alive = list(range(len(names)))
    eliminated: list[tuple[str, float]] = []

    while len(alive) > 1:
        subset = matrix[:, alive]
        means = subset.mean(axis=0)
        # Relative to the set's own average loss, which is what the t_max statistic
        # compares: a model far above the set mean is the candidate for elimination.
        relative = means - means.mean()

        resampled = subset[index].mean(axis=1)
        centred = resampled - means
        relative_star = centred - centred.mean(axis=1, keepdims=True)
        # Bootstrap variance of each model's relative loss, which is what studentises
        # both the observed statistic and its null distribution.
        variance = np.maximum(relative_star.var(axis=0), 1e-300)
        scale = np.sqrt(variance)

        statistic = float(np.max(relative / scale))
        null = np.max(relative_star / scale, axis=1)
        p_value = float(np.mean(null > statistic))

        if p_value >= alpha:
            break

        worst = int(np.argmax(relative / scale))
        eliminated.append((names[alive[worst]], p_value))
        alive.pop(worst)

    return ModelConfidenceSet(
        included=tuple(names[i] for i in alive),
        eliminated=tuple(eliminated),
        alpha=alpha,
        n=n,
        block=block,
        draws=draws,
    )


@dataclass(frozen=True, slots=True)
class MinimumDetectableEffect:
    """The smallest CRPS skill this sample could have found, at a stated power."""

    horizon: int
    n: int
    n_independent: int
    power: float
    alpha: float
    #: In CRPS units, on the mean loss differential.
    effect_in_loss: float | None
    #: The same effect expressed as a skill score against the benchmark's own CRPS.
    effect_in_skill: float | None
    benchmark_crps: float
    undefined_reason: str | None = None

    def describe(self) -> dict[str, Any]:
        return {
            "power": self.power,
            "alpha": self.alpha,
            "observations": self.n,
            "independent_observations": self.n_independent,
            "detectable_loss_differential": (
                None if self.effect_in_loss is None else round(self.effect_in_loss, 6)
            ),
            "detectable_skill": (
                None if self.effect_in_skill is None else round(self.effect_in_skill, 5)
            ),
            "detectable_skill_pct": (
                None if self.effect_in_skill is None else round(self.effect_in_skill * 100.0, 3)
            ),
            "benchmark_crps": round(self.benchmark_crps, 4),
            "undefined_reason": self.undefined_reason,
            "reading": (
                "The smallest true CRPS skill a Diebold-Mariano test on this sample "
                "would reject at, four times in five. It uses the loss-differential "
                "variance actually observed, the HAC lag the overlap implies and the "
                "same small-sample correction the test carries, so it describes the "
                "test that ran rather than an idealised one. A negative result above "
                "this number is evidence of absence; one below it is an absence of "
                "evidence, and the two are not the same finding."
            ),
        }


def minimum_detectable_effect(
    model_losses: np.ndarray,
    benchmark_losses: np.ndarray,
    *,
    horizon: int,
    sampling: Sampling,
    power: float = DEFAULT_POWER,
    alpha: float = DEFAULT_ALPHA,
) -> MinimumDetectableEffect:
    """How large a skill score this sample could have detected, and it did not.

    Computed against a nominated model's own loss series because the differential's
    variance depends on which two forecasters are being differenced — a model that
    tracks the benchmark closely has a small differential variance and a correspondingly
    small detectable effect, and quoting one MDE for a whole shootout would hide that.
    """
    model = np.asarray(model_losses, dtype=float)
    reference = np.asarray(benchmark_losses, dtype=float)
    _require_finite({"model": model}, reference)
    n = int(model.size)
    benchmark_crps = float(np.mean(reference)) if n else 0.0
    independent = sampling.independent_count(n)

    def undefined(reason: str) -> MinimumDetectableEffect:
        return MinimumDetectableEffect(
            horizon=horizon,
            n=n,
            n_independent=independent,
            power=power,
            alpha=alpha,
            effect_in_loss=None,
            effect_in_skill=None,
            benchmark_crps=benchmark_crps,
            undefined_reason=reason,
        )

    if n < 3:
        return undefined(f"a power calculation needs at least 3 observations, got {n}")

    overlap = sampling.stride
    factor = hln_factor(n, overlap)
    if factor <= 0.0:
        return undefined(
            f"{n} records overlapping {overlap} deep carry no independent window, so "
            f"there is no effect size this sample could detect"
        )

    differential = model - reference
    centred = differential - differential.mean()
    lag = overlap - 1
    variance = float(_hac_variance(centred.reshape(-1, 1), lag)[0])
    if variance <= 0.0:
        return undefined(
            "the loss differential has no variance, so the two forecasters scored "
            "identically on every window"
        )

    # The DM statistic is (dbar / sqrt(var/n)) * sqrt(hln) against t(n-1), so the effect
    # that reaches significance is inflated by the same correction that shrinks the
    # statistic. Using the normal quantiles here would understate the MDE at exactly the
    # long horizons where the independent count is smallest.
    critical = float(stats.t.isf(alpha / 2.0, df=n - 1))
    detect = float(stats.t.isf(1.0 - power, df=n - 1))
    standard_error = float(np.sqrt(variance / n)) / float(np.sqrt(factor))
    effect = (critical + detect) * standard_error

    return MinimumDetectableEffect(
        horizon=horizon,
        n=n,
        n_independent=independent,
        power=power,
        alpha=alpha,
        effect_in_loss=effect,
        effect_in_skill=effect / benchmark_crps if benchmark_crps > 0.0 else None,
        benchmark_crps=benchmark_crps,
    )
