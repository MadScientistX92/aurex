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

**The same discipline reaches direction, where the metric is resolution rather than
CRPS.** :func:`resolution_screen` asks "can any model in this set tell one window from
another", which is one test over six models for the same reason SPA is. Its statistic is
the largest studentised resolution and its null comes from resampling the *outcomes*
against fixed forecasts, so what is destroyed is the alignment and nothing else.

Two details there are load-bearing. Resolution is a sum of squares, so it is positive
under the null and "close to zero" is not a reading — the reference distribution says
what zero is worth on this sample and the artifact publishes it beside every figure. And
on overlapping windows the resampling is a circular *shift* rather than a permutation: a
shift preserves the outcome series' autocorrelation exactly, where a permutation would
destroy it and turn ordinary persistence into a rejection. That is the same choice
Diebold-Mariano makes when it handles dependence in its variance estimator instead of
thinning it away, and as there, both the full sample and the thinned subsample are run.

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

from aurex.score.reliability import DEFAULT_BINS, bin_assignment
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


def _require_finite(
    losses: dict[str, np.ndarray],
    reference: np.ndarray | None = None,
    *,
    quantity: str = "CRPS",
    reference_name: str = "benchmark",
) -> None:
    """Refuse a series that is not finite, loudly and by name.

    Not defensive tidying. NaN propagates through every comparison as ``False``, so a
    single broken forecaster makes ``mean(bootstrap > statistic)`` evaluate to zero and
    the SPA test reports ``p = 0.000`` — the most significant result the test can
    produce, from a model that did not forecast anything. It is the exact shape of
    failure this repository refuses everywhere else: silent, and wrong in the direction
    that looks like a finding. A broken model must stop the run, not win it.

    ``quantity`` and ``reference_name`` exist so the message names what actually went
    wrong. The resolution screen feeds this probabilities and outcomes, and a message
    reading "non-finite CRPS from ['benchmark']" would send whoever hit it to the wrong
    module.
    """
    broken = sorted(name for name, values in losses.items() if not np.all(np.isfinite(values)))
    if reference is not None and not np.all(np.isfinite(reference)):
        broken.append(reference_name)
    if broken:
        raise NonFiniteLossError(
            f"non-finite {quantity} from {broken}. A model that produced NaN did not "
            f"forecast, and scoring it would report a perfect p-value for an empty "
            f"result. Fix the forecaster or drop it from the set."
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


#: Cyclic shifts nearer than this to perfect alignment are dropped from the reference
#: set, in units of the overlap stride. At one stride the shifted outcome window still
#: shares sessions with the forecast that is being scored against it, so those shifts are
#: not draws from "this model knows nothing" — they are draws from a weakened version of
#: the alternative, and including them would make the test conservative in a way that
#: hides exactly the small effect it exists to find.
SHIFT_GUARD_STRIDES = 1

#: Reference draws required before a permutation p-value is reported at all. Twenty gives
#: a smallest attainable p of 1/21, which can reject at 5% and only just; below that the
#: test cannot produce the answer it is being asked for and says so instead.
MIN_REFERENCE_DRAWS = 20


def _resolutions(assignment: np.ndarray, outcomes: np.ndarray, *, bins: int) -> np.ndarray:
    """Murphy resolution of one model's forecasts against many outcome vectors at once.

    ``outcomes`` is ``(draws, n)``. The bin assignment is fixed across draws because the
    forecasts never move — only what they are lined up against does, which is the whole
    construction of the null.
    """
    n = int(assignment.size)
    indicator = np.zeros((n, bins), dtype=float)
    indicator[np.arange(n), assignment] = 1.0

    counts = indicator.sum(axis=0)
    occupied = counts > 0
    # Empty bins contribute nothing to resolution; the divide is guarded rather than
    # masked so the shapes stay rectangular for the vectorised path.
    safe = np.where(occupied, counts, 1.0)

    observed_rate = np.asarray(outcomes, dtype=float).mean(axis=1, keepdims=True)
    bin_rates = (np.asarray(outcomes, dtype=float) @ indicator) / safe
    contribution = np.where(occupied, (bin_rates - observed_rate) ** 2 * counts, 0.0)
    return np.asarray(contribution.sum(axis=1) / n, dtype=float)


def _equal_count_assignment(probabilities: np.ndarray, *, bins: int) -> np.ndarray:
    """Bin by rank instead of by value: each bin holds a tenth of this model's forecasts.

    The equal-width decomposition has a blind spot that matters more here than anywhere
    else the reliability machinery is used. A direction forecast lives near one half —
    an uncentred model's P(up) might range over 0.47 to 0.58 across every window it ever
    saw — and that entire range falls inside one or two of ten equal-width bins. Its
    resolution is then zero or nearly zero *by construction*, whatever the model knew,
    because resolution measures how far bin rates move from the base rate and there is
    only one bin for them to move in.

    Publishing that zero as a finding would be assuming the answer. So the same test is
    run a second time on bins of equal count, which adapt to whatever range the model
    actually used and therefore ask whether its *ordering* of windows carries
    information. A model that scores zero at equal width and non-zero here has
    discrimination the reliability diagram's binning cannot see, and that difference is
    published rather than resolved silently in either direction.
    """
    values = np.asarray(probabilities, dtype=float)
    # Rank first so ties spread deterministically rather than piling into one bin: a
    # forecaster that repeats the same probability often would otherwise leave most bins
    # empty and score an artificially low resolution here too.
    order = np.argsort(np.argsort(values, kind="stable"), kind="stable")
    assigned = (order * bins) // max(values.size, 1)
    return np.clip(assigned, 0, bins - 1).astype(int)


def _cyclic_shift_index(n: int, *, guard: int) -> np.ndarray:
    """``(k, n)`` of circularly shifted positions, alignment and its neighbourhood removed.

    A circular shift is the right null for an overlapping sample because it preserves the
    entire autocorrelation function of the outcome series exactly and destroys only its
    alignment with the forecasts. An iid permutation would break the serial dependence
    too, which makes the null's bin means less variable than they really are and turns
    ordinary persistence into a rejection.
    """
    shifts = np.arange(guard, max(n - guard, guard) + 1, dtype=int)
    shifts = shifts[(shifts > 0) & (shifts < n)]
    if shifts.size == 0:
        return np.zeros((0, n), dtype=int)
    return (np.arange(n)[None, :] + shifts[:, None]) % n


@dataclass(frozen=True, slots=True)
class ResolutionScreen:
    """Can any model here tell one window from another? One test over the whole set.

    The statistic is the largest *studentised* resolution across models, which is the
    same shape as :class:`SuperiorPredictiveAbility`'s and is there for the same reason.
    Raw resolutions are not comparable across models: a forecaster whose probabilities
    spread over eight bins can post a larger resolution than one confined to two purely
    from having more bins to be noisy in. Each model is therefore referred to its own
    null, and the maximum is taken after that.
    """

    event: str
    scheme: str
    #: Which partition of the forecast axis the decomposition used. ``equal_width``
    #: matches the published reliability diagram; ``equal_count`` is the robustness run.
    binning: str
    models: tuple[str, ...]
    #: Observed resolution per model, in the order of ``models``. What the README quotes.
    resolutions: tuple[float, ...]
    #: Mean resolution the reference set produces for that model with no skill at all.
    #: The number that says what "resolution at or near zero" is worth on this sample:
    #: binned resolution is a sum of squares and is positive under the null.
    null_means: tuple[float, ...]
    #: Marginal p per model. Description, not decision — see ``p_value``.
    model_p_values: tuple[float, ...]
    #: How many bins each model's forecasts actually reached. A model confined to one bin
    #: has a resolution of zero whatever it knew, so this is what separates "measured no
    #: discrimination" from "this partition could not have measured any".
    occupied_bins: tuple[int, ...]
    statistic: float
    best: str | None
    #: The one to read: the largest studentised resolution against its own null.
    p_value: float
    n: int
    draws: int

    @property
    def rejects(self) -> bool:
        return self.p_value < DEFAULT_ALPHA

    def describe(self) -> dict[str, Any]:
        return {
            "test": "max_studentised_resolution",
            "resampling": self.scheme,
            "binning": self.binning,
            "null": (
                f"no model in this set has any resolution on {self.event}: every "
                f"forecast is unrelated to which window it was made in"
            ),
            "statistic": round(self.statistic, 4),
            "p_value": round(self.p_value, 4),
            "rejects_at_5pct": self.rejects,
            "best_model": self.best,
            "models": {
                name: {
                    "resolution": round(value, 6),
                    "resolution_under_null": round(null, 6),
                    "p_value": round(p, 4),
                    "occupied_bins": occupied,
                    "resolution_measurable": occupied > 1,
                }
                for name, value, null, p, occupied in zip(
                    self.models,
                    self.resolutions,
                    self.null_means,
                    self.model_p_values,
                    self.occupied_bins,
                    strict=True,
                )
            },
            "observations": self.n,
            "reference_draws": self.draws,
            "reading": (
                "One test over the whole set, so the multiplicity is inside the null "
                "distribution rather than corrected for afterwards — six resolution "
                "figures eyeballed against zero is the leaderboard problem again. "
                "Resolution is level-invariant: it asks whether a model can tell one "
                "window from another, which is what calling direction means, where the "
                "Brier score confounds that with getting the base rate right. It is a "
                "sum of squares and so is positive even with no skill, which is what "
                "resolution_under_null is for: a resolution is only a finding if it "
                "exceeds what this model's own binning produces by chance. The per-model "
                "p-values are description; the single p_value above is the decision."
            ),
        }


def resolution_screen(
    probabilities: dict[str, np.ndarray],
    outcomes: np.ndarray,
    *,
    event: str,
    sampling: Sampling,
    bins: int = DEFAULT_BINS,
    binning: str = "equal_width",
    draws: int = DEFAULT_DRAWS,
    seed: int = 20260807,
) -> ResolutionScreen | None:
    """Test every model's resolution at once, against a null built by resampling outcomes.

    ``binning`` chooses the partition the decomposition is computed over: ``equal_width``
    matches the reliability diagram this repository publishes everywhere else, and
    ``equal_count`` is the robustness run that :func:`_equal_count_assignment` explains —
    on a direction forecast, which lives near one half, the two can differ for reasons
    that have nothing to do with the model.

    On an overlapping sample the reference set is every admissible circular shift of the
    outcome series, enumerated rather than sampled — there are only ``n`` of them and
    enumerating makes the test exact. On a non-overlapping one it is ``draws`` random
    permutations, which is valid precisely because thinning has already removed the
    dependence a shift exists to preserve.

    ``None`` where the sample cannot support the test: no models, or too few reference
    draws to produce a p-value that could reject.
    """
    names = tuple(sorted(probabilities))
    if not names:
        return None

    observed = np.asarray(outcomes, dtype=float)
    n = int(observed.size)
    if n < 3:
        return None
    _require_finite(
        probabilities,
        observed,
        quantity="forecast probability",
        reference_name="realised outcomes",
    )

    if sampling.overlapping:
        scheme = "cyclic_shift"
        index = _cyclic_shift_index(n, guard=SHIFT_GUARD_STRIDES * sampling.stride)
    else:
        scheme = "permutation"
        rng = np.random.default_rng(seed)
        index = np.argsort(rng.random((draws, n)), axis=1)

    if index.shape[0] < MIN_REFERENCE_DRAWS:
        return None

    reference = observed[index]
    if binning == "equal_width":
        assignments = {name: bin_assignment(probabilities[name], bins=bins) for name in names}
    elif binning == "equal_count":
        assignments = {
            name: _equal_count_assignment(probabilities[name], bins=bins) for name in names
        }
    else:
        raise ValueError(f"binning must be equal_width or equal_count, got {binning!r}")

    actual = np.array(
        [_resolutions(assignments[name], observed[None, :], bins=bins)[0] for name in names]
    )
    null = np.column_stack(
        [_resolutions(assignments[name], reference, bins=bins) for name in names]
    )

    # Studentise each model by its own reference distribution before taking the maximum,
    # so the max is over comparable quantities. A model whose forecasts all land in one
    # bin has no resolution under any alignment; its column is identically zero and
    # standardising it would be a division by nothing, so it contributes a flat zero and
    # cannot win the maximum by accident.
    centre = null.mean(axis=0)
    spread = null.std(axis=0)
    live = spread > 0.0
    scale = np.where(live, spread, 1.0)
    studentised_actual = np.where(live, (actual - centre) / scale, 0.0)
    studentised_null = np.where(live, (null - centre) / scale, 0.0)

    statistic = float(np.max(studentised_actual))
    k = int(index.shape[0])
    # The observed configuration is itself a member of the reference set, which is what
    # the +1 on both sides is: it keeps the p-value valid rather than letting it reach
    # zero, and a permutation test that can report p = 0 is reporting its draw count.
    maxima = studentised_null.max(axis=1)
    p_value = float((1.0 + np.count_nonzero(maxima >= statistic)) / (1.0 + k))
    marginal = tuple(
        float((1.0 + np.count_nonzero(null[:, i] >= actual[i])) / (1.0 + k))
        for i in range(len(names))
    )

    return ResolutionScreen(
        event=event,
        scheme=scheme,
        binning=binning,
        models=names,
        resolutions=tuple(float(value) for value in actual),
        null_means=tuple(float(value) for value in centre),
        model_p_values=marginal,
        occupied_bins=tuple(int(np.unique(assignments[name]).size) for name in names),
        statistic=statistic,
        best=names[int(np.argmax(studentised_actual))] if statistic > 0.0 else None,
        p_value=p_value,
        n=n,
        draws=k,
    )
