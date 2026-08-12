"""Loadings, the intervals around them, and how little they explain out of sample.

Four measurements, and the reason there are four rather than one is that each answers a
question the others cannot:

**The elastic-net fit** is the decomposition §5 asks for. Its penalty is chosen by
blocked cross-validation on contiguous folds — a shuffled fold in a serially dependent
series trains on the neighbours of what it is scoring, which makes the selected penalty
too small and the fit look better than it is.

**An OLS fit beside it, with HAC standard errors.** Not redundancy. A percentile
bootstrap around a penalised estimator is known not to be valid at coefficients that are
exactly zero — the estimator has an atom there and the interval inherits it — so a
bootstrap interval that excludes zero is weaker evidence than it looks. OLS on the same
design has honest inference and no selection, so the two are published side by side:
the penalised fit for the decomposition, the unpenalised one for the interval. Where
they disagree about a driver, that disagreement is the finding.

**A moving-block bootstrap** over the penalised fit, which is what a reader of the
loadings actually wants: how often does this driver survive selection at all, and how
often with the sign the point estimate has. Reported as selection and sign frequencies
rather than only as an interval, because those two numbers degrade gracefully where the
interval does not.

**Rolling windows**, which measure whether any of the above is stable enough to mean
anything. A loading that changes sign between adjacent three-year windows is not a
loading; it is a summary of one regime being read as a property of the asset.

And separately from all four, out-of-sample R-squared, in the two forms the factor set
admits. Contemporaneous — regressors from the same week as the return, coefficients
fitted on weeks that ended before it — is what attribution means. Predictive — the whole
design shifted one week, so the regression is asked to forecast — is the quantity §5's
ban is about, and it is measured precisely so the ban has a number behind it rather than
an assertion.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from aurex.factors import elasticnet
from aurex.factors.design import WeeklyDesign
from aurex.score.sampling import Sampling
from aurex.score.significance import diebold_mariano

#: Rolling window, in weeks. Three years, as the methodology section has said since
#: before this was built.
DEFAULT_WINDOW = 156

#: Contiguous cross-validation folds used to choose the penalty.
DEFAULT_FOLDS = 5

#: Moving-block bootstrap replicates. Enough that a 2.5th percentile is not itself noise.
DEFAULT_DRAWS = 2_000

#: Block length in weeks for the bootstrap. Two months, long enough to carry the serial
#: dependence weekly macro series have and short enough that a 600-week sample still
#: contains many distinct blocks.
DEFAULT_BLOCK = 8

#: Mixing between the two penalties. Fixed rather than tuned: tuning it against the same
#: sample the loadings are read from would make sparsity a fitted quantity.
L1_RATIO = 0.5


@dataclass(frozen=True, slots=True)
class Loading:
    """One driver's coefficient, on both scales, with what is known about its spread."""

    factor_id: str
    #: Change in the weekly return per one standard deviation of the driver.
    standardised: float
    #: Change in the weekly return per one natural unit of the driver.
    raw: float
    #: Percentile interval from the moving-block bootstrap of the penalised fit.
    interval: tuple[float, float] | None
    #: Share of bootstrap replicates in which this driver survived selection.
    selection_rate: float | None
    #: Share of *selected* replicates agreeing in sign with the point estimate.
    sign_agreement: float | None
    #: The unpenalised coefficient on the same design, standardised.
    ols_standardised: float
    ols_interval: tuple[float, float] | None
    ols_p_value: float | None

    @property
    def selected(self) -> bool:
        return self.standardised != 0.0

    @property
    def ols_excludes_zero(self) -> bool:
        if self.ols_interval is None:
            return False
        low, high = self.ols_interval
        return low > 0.0 or high < 0.0

    def describe(self) -> dict[str, Any]:
        return {
            "factor": self.factor_id,
            "standardised": round(self.standardised, 6),
            "per_unit": round(self.raw, 8),
            "selected": self.selected,
            "bootstrap_interval_95": (
                None
                if self.interval is None
                else [round(self.interval[0], 6), round(self.interval[1], 6)]
            ),
            "selection_rate": None
            if self.selection_rate is None
            else round(self.selection_rate, 4),
            "sign_agreement": None
            if self.sign_agreement is None
            else round(self.sign_agreement, 4),
            "ols_standardised": round(self.ols_standardised, 6),
            "ols_interval_95": (
                None
                if self.ols_interval is None
                else [round(self.ols_interval[0], 6), round(self.ols_interval[1], 6)]
            ),
            "ols_p_value": None if self.ols_p_value is None else round(self.ols_p_value, 4),
            "ols_excludes_zero": self.ols_excludes_zero,
            "reading": (
                "The standardised coefficient is the change in the weekly return per "
                "one standard deviation of this driver, which is the scale on which two "
                "drivers are comparable. The bootstrap interval is around a penalised "
                "estimator and is not valid at exactly zero, so selection_rate is the "
                "more honest summary of whether this driver survives at all; the OLS "
                "interval beside it has no selection and no such caveat."
            ),
        }


@dataclass(frozen=True, slots=True)
class Stability:
    """What the rolling window says about whether a loading is a property or a regime."""

    factor_id: str
    windows: int
    selected_windows: int
    sign_changes: int
    minimum: float
    maximum: float
    interquartile_range: float
    first_half_mean: float
    second_half_mean: float
    #: Rolling spread measured against the full-sample interval width. Above one means
    #: the loading moves further across windows than its own interval allows for.
    spread_ratio: float | None

    def describe(self) -> dict[str, Any]:
        return {
            "factor": self.factor_id,
            "windows": self.windows,
            "selected_in": self.selected_windows,
            "sign_changes": self.sign_changes,
            "min": round(self.minimum, 6),
            "max": round(self.maximum, 6),
            "iqr": round(self.interquartile_range, 6),
            "first_half_mean": round(self.first_half_mean, 6),
            "second_half_mean": round(self.second_half_mean, 6),
            "spread_ratio": None if self.spread_ratio is None else round(self.spread_ratio, 3),
        }


@dataclass(frozen=True, slots=True)
class OutOfSample:
    """Walk-forward R-squared against the benchmark a forecaster has to beat."""

    kind: str
    r_squared: float
    observations: int
    window: int
    #: Diebold-Mariano on the squared-error differential against the training mean.
    statistic: float | None
    p_value: float | None
    mean_loss_differential: float

    def describe(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "r_squared_oos": round(self.r_squared, 5),
            "observations": self.observations,
            "window_weeks": self.window,
            "benchmark": "training_window_mean",
            "dm_statistic": None if self.statistic is None else round(self.statistic, 4),
            "dm_p_value": None if self.p_value is None else round(self.p_value, 4),
            "mean_loss_differential": round(self.mean_loss_differential, 10),
            "reading": (
                "Out-of-sample R-squared is measured against the mean of the training "
                "window, not against zero, because a forecaster that says 'the average' "
                "is the thing a factor model has to beat. Negative is common and is not "
                "a bug: it means the fitted loadings did worse than that average out of "
                "sample. The differential is squared error of the model minus squared "
                "error of that benchmark, so a negative differential is the model doing "
                "better and a large p-value means the difference of either sign is not "
                "distinguishable from zero on this sample."
            ),
        }


@dataclass(frozen=True, slots=True)
class Withheld:
    """The fit with one driver removed, and how far the rest moved without it.

    This is the omitted-variable claim made measurable. A driver is in the set because
    somebody argued that leaving it out would bias what remains — and that argument is a
    hypothesis about this sample, not a fact about the world. Refitting without it says
    how large the bias actually was, and "we checked and it was negligible" is a
    different statement from "we did not check", which is the only reason to publish a
    result this boring.
    """

    factor_id: str
    #: Largest absolute change in any surviving standardised loading.
    largest_shift: float
    #: Drivers whose loading changed sign when this one was removed.
    sign_flips: tuple[str, ...]
    #: Contemporaneous out-of-sample R-squared of the reduced set.
    r_squared_without: float | None
    #: The same for the full set, repeated so the pair reads without a lookup.
    r_squared_with: float | None

    def describe(self) -> dict[str, Any]:
        return {
            "withheld": self.factor_id,
            "largest_loading_shift": round(self.largest_shift, 8),
            "sign_flips": list(self.sign_flips),
            "r_squared_oos_without": (
                None if self.r_squared_without is None else round(self.r_squared_without, 5)
            ),
            "r_squared_oos_with": (
                None if self.r_squared_with is None else round(self.r_squared_with, 5)
            ),
        }


@dataclass(frozen=True, slots=True)
class Attribution:
    """Everything the loadings layer publishes for one asset."""

    design: WeeklyDesign
    penalty: float
    loadings: tuple[Loading, ...]
    stability: tuple[Stability, ...]
    in_sample_r_squared: float
    contemporaneous: OutOfSample | None
    predictive: OutOfSample | None
    withheld: tuple[Withheld, ...]
    draws: int
    block: int
    window: int

    def describe(self) -> dict[str, Any]:
        return {
            "design": self.design.describe(),
            "estimator": {
                "model": "elastic_net",
                "l1_ratio": L1_RATIO,
                "penalty": self.penalty,
                "penalty_selection": "blocked_cross_validation",
                "folds": DEFAULT_FOLDS,
                "note": (
                    "Folds are contiguous blocks of weeks rather than a shuffle. A "
                    "shuffled fold in a serially dependent series trains on the "
                    "neighbours of what it scores, which selects too small a penalty "
                    "and makes the fit look better than it is."
                ),
            },
            "in_sample_r_squared": round(self.in_sample_r_squared, 5),
            "loadings": [loading.describe() for loading in self.loadings],
            "stability": {
                "window_weeks": self.window,
                "factors": [entry.describe() for entry in self.stability],
                "reading": (
                    "A loading that changes sign between adjacent windows is not a "
                    "loading; it is one regime being read as a property of the asset. "
                    "spread_ratio compares how far a loading moves across windows with "
                    "the width of its own full-sample interval."
                ),
            },
            "out_of_sample": {
                "contemporaneous": (
                    None if self.contemporaneous is None else self.contemporaneous.describe()
                ),
                "predictive": None if self.predictive is None else self.predictive.describe(),
                "note": (
                    "Contemporaneous uses regressors from the same week as the return, "
                    "with coefficients fitted only on weeks that ended before it: that "
                    "is what attribution means. Predictive shifts the whole design one "
                    "week so the regression is asked to forecast, which §5 forbids "
                    "using and which is measured here so the ban has a number behind it."
                ),
            },
            "omitted_variable_check": {
                "method": "leave_one_required_driver_out",
                "reading": (
                    "Each required driver removed in turn, the rest refitted, and the "
                    "largest resulting move in any surviving loading reported. A driver "
                    "is required because somebody argued that leaving it out would bias "
                    "what remains; this is that argument measured on this sample rather "
                    "than asserted. A negligible shift does not mean the driver was "
                    "unnecessary — it means the bias it defends against did not "
                    "materialise here, which is a thing that had to be checked to be "
                    "known."
                ),
                "drivers": [entry.describe() for entry in self.withheld],
            },
            "bootstrap": {
                "method": "moving_block",
                "draws": self.draws,
                "block_weeks": self.block,
                "caveat": (
                    "A percentile bootstrap around a penalised estimator is not valid "
                    "at coefficients that are exactly zero. The interval is published "
                    "with that stated rather than omitted, and the OLS interval beside "
                    "each loading is the one without the caveat."
                ),
            },
        }


def _r_squared(actual: np.ndarray, fitted: np.ndarray) -> float:
    residual = float(np.sum((actual - fitted) ** 2))
    total = float(np.sum((actual - actual.mean()) ** 2))
    return 1.0 - residual / total if total > 0.0 else 0.0


def _folds(n: int, folds: int) -> list[np.ndarray]:
    """Contiguous, equal-ish blocks of consecutive observations."""
    return [block for block in np.array_split(np.arange(n), folds) if block.size]


def choose_penalty(
    design: np.ndarray,
    target: np.ndarray,
    *,
    l1_ratio: float = L1_RATIO,
    folds: int = DEFAULT_FOLDS,
    steps: int = 60,
) -> float:
    """The penalty minimising held-out squared error over contiguous folds."""
    penalties = elasticnet.penalty_path(design, target, l1_ratio=l1_ratio, steps=steps)
    blocks = _folds(len(target), folds)
    if len(blocks) < 2:
        return float(penalties[-1])

    errors = np.zeros(len(penalties))
    for block in blocks:
        mask = np.ones(len(target), dtype=bool)
        mask[block] = False
        if mask.sum() < 2 or np.allclose(target[mask], target[mask][0]):
            continue
        fits = elasticnet.fit_path(design[mask], target[mask], penalties, l1_ratio=l1_ratio)
        for i, fitted in enumerate(fits):
            residual = target[block] - fitted.predict(design[block])
            errors[i] += float(np.sum(residual**2))

    return float(penalties[int(np.argmin(errors))])


def _newey_west_lag(n: int) -> int:
    """The standard automatic truncation, ``floor(4 (n/100)^(2/9))``."""
    return int(np.floor(4.0 * (n / 100.0) ** (2.0 / 9.0))) if n > 0 else 0


def ols_with_hac(
    design: np.ndarray, target: np.ndarray, *, lag: int | None = None
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """OLS coefficients, HAC standard errors, and the residuals.

    Standardised inside, so the coefficients returned are directly comparable with the
    penalised fit's and with each other. The intercept is estimated and discarded — it
    is the sample mean of a return series, which no reader of a loading wants.
    """
    standardised, centred, _ = elasticnet.standardise(design, target)
    n, p = standardised.shape
    with_const = np.column_stack([np.ones(n), standardised])

    xtx = with_const.T @ with_const
    coefficients = np.linalg.lstsq(with_const, centred, rcond=None)[0]
    residual = centred - with_const @ coefficients

    truncation = _newey_west_lag(n) if lag is None else lag
    scores = with_const * residual[:, None]
    meat = scores.T @ scores
    for order in range(1, truncation + 1):
        if order >= n:
            break
        gamma = scores[order:].T @ scores[:-order]
        weight = 1.0 - order / (truncation + 1.0)
        meat = meat + weight * (gamma + gamma.T)

    inverse = np.linalg.pinv(xtx)
    covariance = inverse @ meat @ inverse
    errors = np.sqrt(np.clip(np.diag(covariance), 0.0, None))
    return coefficients[1:], errors[1 : p + 1], residual


def _moving_block_indices(rng: np.random.Generator, n: int, block: int) -> np.ndarray:
    """One resampled index vector of length ``n`` from wrapped contiguous blocks."""
    span = min(block, n)
    count = int(np.ceil(n / span))
    starts = rng.integers(0, n, size=count)
    picks = (starts[:, None] + np.arange(span)[None, :]) % n
    return picks.reshape(-1)[:n]


def bootstrap(
    design: np.ndarray,
    target: np.ndarray,
    *,
    penalty: float,
    l1_ratio: float = L1_RATIO,
    draws: int = DEFAULT_DRAWS,
    block: int = DEFAULT_BLOCK,
    seed: int = 4,
) -> tuple[np.ndarray, np.ndarray]:
    """Bootstrap the penalised coefficients. Returns ``(replicates, selection_rate)``.

    The penalty is held at the full-sample choice rather than re-selected per replicate.
    Re-selecting would fold the cross-validation's own sampling variation into the
    interval, and the interval would then be answering a question about the selection
    rule rather than about the loading.
    """
    rng = np.random.default_rng(seed)
    n = len(target)
    replicates = np.empty((draws, design.shape[1]))

    for draw in range(draws):
        index = _moving_block_indices(rng, n, block)
        resampled_target = target[index]
        if np.allclose(resampled_target, resampled_target[0]):
            replicates[draw] = np.nan
            continue
        fitted = elasticnet.fit(design[index], resampled_target, lam=penalty, l1_ratio=l1_ratio)
        replicates[draw] = fitted.coefficients

    usable = replicates[np.isfinite(replicates).all(axis=1)]
    if usable.size == 0:
        return replicates, np.zeros(design.shape[1])
    return usable, (usable != 0.0).mean(axis=0)


def rolling(
    design: np.ndarray,
    target: np.ndarray,
    *,
    window: int = DEFAULT_WINDOW,
    l1_ratio: float = L1_RATIO,
    folds: int = DEFAULT_FOLDS,
    step: int = 4,
) -> np.ndarray:
    """Standardised coefficients from each rolling window, one row per window.

    ``step`` is four weeks rather than one. Adjacent weekly windows share 155 of their
    156 observations, so refitting every week would produce a smooth picture of almost
    nothing and cost 150 times as much; a month between refits keeps the windows
    distinguishable without pretending they are independent.
    """
    n = len(target)
    if n <= window:
        return np.empty((0, design.shape[1]))

    rows: list[np.ndarray] = []
    for end in range(window, n + 1, step):
        piece = slice(end - window, end)
        block_design, block_target = design[piece], target[piece]
        if np.allclose(block_target, block_target[0]):
            continue
        penalty = choose_penalty(block_design, block_target, l1_ratio=l1_ratio, folds=folds)
        rows.append(
            elasticnet.fit(block_design, block_target, lam=penalty, l1_ratio=l1_ratio).coefficients
        )
    return np.array(rows) if rows else np.empty((0, design.shape[1]))


def walk_forward(
    design: np.ndarray,
    target: np.ndarray,
    *,
    kind: str,
    window: int = DEFAULT_WINDOW,
    l1_ratio: float = L1_RATIO,
    folds: int = DEFAULT_FOLDS,
    refit_every: int = 4,
) -> OutOfSample | None:
    """Fit on the trailing window, score the next observation, and never look ahead.

    Coefficients are refitted every ``refit_every`` weeks and carried between refits,
    which is what a desk running this would actually do. The alternative — refitting on
    every observation — changes the answer by less than the noise in it and costs an
    order of magnitude more, and the interval below would not be able to tell them apart.
    """
    n = len(target)
    if n <= window + 1:
        return None

    predictions: list[float] = []
    benchmarks: list[float] = []
    actuals: list[float] = []

    fitted: elasticnet.ElasticNetFit | None = None
    penalty = 0.0
    for position in range(window, n):
        piece = slice(position - window, position)
        if fitted is None or (position - window) % refit_every == 0:
            block_design, block_target = design[piece], target[piece]
            if np.allclose(block_target, block_target[0]):
                continue
            penalty = choose_penalty(block_design, block_target, l1_ratio=l1_ratio, folds=folds)
            fitted = elasticnet.fit(block_design, block_target, lam=penalty, l1_ratio=l1_ratio)
        predictions.append(float(fitted.predict(design[position : position + 1])[0]))
        benchmarks.append(float(target[piece].mean()))
        actuals.append(float(target[position]))

    if len(actuals) < 2:
        return None

    actual = np.array(actuals)
    model_error = (actual - np.array(predictions)) ** 2
    benchmark_error = (actual - np.array(benchmarks)) ** 2
    residual = float(model_error.sum())
    total = float(benchmark_error.sum())

    # Weekly, non-overlapping, one observation per week: the sampling declaration that
    # collapses the HAC correction to a paired t-test, which is the right test here and
    # is reused rather than reimplemented.
    test = diebold_mariano(
        model_error,
        benchmark_error,
        sampling=Sampling(horizon=1, step=1),
        null="training_window_mean",
    )

    return OutOfSample(
        kind=kind,
        r_squared=1.0 - residual / total if total > 0.0 else 0.0,
        observations=len(actual),
        window=window,
        statistic=test.statistic,
        p_value=test.p_value,
        mean_loss_differential=test.mean_differential,
    )


def _stability_for(
    name: str,
    column: np.ndarray,
    *,
    interval: tuple[float, float] | None,
) -> Stability:
    selected = column[column != 0.0]
    signs = np.sign(selected)
    changes = int(np.count_nonzero(np.diff(signs) != 0)) if signs.size > 1 else 0
    half = len(column) // 2
    width = None if interval is None else interval[1] - interval[0]
    spread = float(np.subtract(*np.percentile(column, [75, 25])))

    return Stability(
        factor_id=name,
        windows=len(column),
        selected_windows=int(selected.size),
        sign_changes=changes,
        minimum=float(column.min()),
        maximum=float(column.max()),
        interquartile_range=spread,
        first_half_mean=float(column[:half].mean()) if half else float("nan"),
        second_half_mean=float(column[half:].mean()) if half else float("nan"),
        spread_ratio=None if not width else float(spread / width),
    )


def leave_one_out(
    design: WeeklyDesign,
    *,
    full_loadings: np.ndarray,
    full_r_squared: float | None,
    window: int,
    folds: int,
) -> tuple[Withheld, ...]:
    """Refit without each required driver in turn, and report what moved.

    Only required drivers. An optional one is already allowed to vanish when its source
    fails, so the artifact records what happens without it whenever that happens; a
    required one never vanishes on its own, which is exactly why the counterfactual has
    to be constructed rather than waited for.
    """
    names = design.names
    required = {spec.id for spec in design.specs if spec.required}
    matrix, values = design.matrix(), design.values()

    out: list[Withheld] = []
    for position, name in enumerate(names):
        if name not in required or matrix.shape[1] < 2:
            continue
        keep = [i for i in range(len(names)) if i != position]
        reduced = matrix[:, keep]

        penalty = choose_penalty(reduced, values, folds=folds)
        refitted = elasticnet.fit(reduced, values, lam=penalty, l1_ratio=L1_RATIO).coefficients

        shifts = np.abs(refitted - full_loadings[keep])
        flips = tuple(
            names[keep[i]]
            for i in range(len(keep))
            if np.sign(refitted[i]) != np.sign(full_loadings[keep[i]])
            and refitted[i] != 0.0
            and full_loadings[keep[i]] != 0.0
        )
        scored = walk_forward(reduced, values, kind="contemporaneous", window=window, folds=folds)
        out.append(
            Withheld(
                factor_id=name,
                largest_shift=float(shifts.max()) if shifts.size else 0.0,
                sign_flips=flips,
                r_squared_without=None if scored is None else scored.r_squared,
                r_squared_with=full_r_squared,
            )
        )
    return tuple(out)


def estimate(
    design: WeeklyDesign,
    *,
    window: int = DEFAULT_WINDOW,
    draws: int = DEFAULT_DRAWS,
    block: int = DEFAULT_BLOCK,
    folds: int = DEFAULT_FOLDS,
    seed: int = 4,
) -> Attribution:
    """Fit, bootstrap, roll and score one weekly design."""
    matrix, values = design.matrix(), design.values()
    names = design.names

    penalty = choose_penalty(matrix, values, folds=folds)
    fitted = elasticnet.fit(matrix, values, lam=penalty, l1_ratio=L1_RATIO)

    replicates, selection = bootstrap(
        matrix, values, penalty=penalty, draws=draws, block=block, seed=seed
    )
    coefficients, errors, _ = ols_with_hac(matrix, values)
    # Normal quantile: the HAC covariance is an asymptotic object, so a t reference
    # would be a precision the estimator does not have.
    half_width = 1.959963984540054 * errors

    loadings: list[Loading] = []
    for position, name in enumerate(names):
        draws_here = replicates[:, position]
        finite = draws_here[np.isfinite(draws_here)]
        point = float(fitted.coefficients[position])
        chosen = finite[finite != 0.0]
        agreement = (
            float(np.mean(np.sign(chosen) == np.sign(point)))
            if chosen.size and point != 0.0
            else None
        )
        interval = (
            (float(np.quantile(finite, 0.025)), float(np.quantile(finite, 0.975)))
            if finite.size > 1
            else None
        )
        error = float(errors[position])
        loadings.append(
            Loading(
                factor_id=name,
                standardised=point,
                raw=float(fitted.raw_coefficients[position]),
                interval=interval,
                selection_rate=float(selection[position]) if finite.size else None,
                sign_agreement=agreement,
                ols_standardised=float(coefficients[position]),
                ols_interval=(
                    None
                    if error <= 0.0
                    else (
                        float(coefficients[position] - half_width[position]),
                        float(coefficients[position] + half_width[position]),
                    )
                ),
                ols_p_value=(
                    None
                    if error <= 0.0
                    else float(2.0 * _normal_sf(abs(coefficients[position]) / error))
                ),
            )
        )

    windows = rolling(matrix, values, window=window, folds=folds)
    stability = tuple(
        _stability_for(
            name,
            windows[:, position],
            interval=loadings[position].interval,
        )
        for position, name in enumerate(names)
        if windows.size
    )

    # The predictive variant shifts every regressor one week later against the same
    # target, so the fit is asked to forecast. Built here rather than in the design so
    # that the two measurements are guaranteed to run on the same sample.
    lagged = design.frame.shift(1)
    joined = pd.concat([design.target.rename("__target__"), lagged], axis=1, sort=False).dropna()

    contemporaneous = walk_forward(
        matrix, values, kind="contemporaneous", window=window, folds=folds
    )

    return Attribution(
        design=design,
        penalty=penalty,
        loadings=tuple(loadings),
        stability=stability,
        in_sample_r_squared=_r_squared(values, fitted.predict(matrix)),
        contemporaneous=contemporaneous,
        withheld=leave_one_out(
            design,
            full_loadings=fitted.coefficients,
            full_r_squared=None if contemporaneous is None else contemporaneous.r_squared,
            window=window,
            folds=folds,
        ),
        predictive=walk_forward(
            joined.drop(columns="__target__").to_numpy(dtype=float),
            joined["__target__"].to_numpy(dtype=float),
            kind="predictive",
            window=window,
            folds=folds,
        ),
        draws=draws,
        block=block,
        window=window,
    )


def _normal_sf(value: float) -> float:
    from scipy import stats

    return float(stats.norm.sf(value))
