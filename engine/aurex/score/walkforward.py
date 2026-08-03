"""The walk-forward harness: re-ask the engine what it would have said, then grade it.

Expanding window, one refit per as-of date, no lookahead. At each date the forecaster
sees prices up to and including that day and nothing after it; the realised outcome is
read from the sessions that follow. The harness slices and the forecaster cannot
unslice, so absence of lookahead is a property of the interface rather than a claim in
a docstring — see :mod:`aurex.score.forecasters`.

**A run always carries its null.** ``baseline`` is a required argument, so there is no
way to produce a CRPS from this module without also producing the random walk's CRPS
for the same date and the same horizon. §0 says the null hypothesis is the random walk;
this is where that stops being a promise.

**Overlap is recorded, not assumed away.** Forecasting every ``step`` sessions over a
longer horizon produces records that share most of their path. Means and histograms are
computed over all of them; every p-value is computed over the thinned, non-overlapping
subsample, and both counts are reported. See :mod:`aurex.score.sampling`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from aurex.dist.paths import PathEnsemble
from aurex.score.coverage import (
    TestResult,
    christoffersen_independence,
    conditional_coverage,
    kupiec,
)
from aurex.score.distribution import (
    DEFAULT_PIT_BINS,
    GoodnessOfFit,
    UniformityTest,
    chi_square_uniformity,
    crps,
    crps_skill,
    pit_histogram,
    pit_value,
    uniformity,
)
from aurex.score.drift import DriftDisplacement, HorizonDisplacement, displacement
from aurex.score.events import BinaryEvent, RealisedPath, TerminalAbove, default_events
from aurex.score.forecasters import AsOfForecaster
from aurex.score.reliability import DEFAULT_BINS, ReliabilityCurve, reliability_curve
from aurex.score.sampling import Sampling
from aurex.vol.base import InsufficientDataError

#: Seed spacing. Each as-of date gets a thousand-wide band so a forecaster can add its
#: horizon without two (date, horizon) pairs ever sharing a stream.
_SEED_STRIDE = 1_000
#: Separates the subject's streams from the null's. A shared stream would correlate the
#: two ensembles and quietly shrink the difference the skill score is measuring.
_BASELINE_OFFSET = 500_000_000

#: Prices required before the first forecast: three years, so the earliest fit is not
#: measurably worse than the latest purely for want of sample. A module constant rather
#: than only a field default, because a slotted dataclass does not expose its defaults
#: as class attributes and callers legitimately need to reason about this one.
DEFAULT_MIN_OBSERVATIONS = 750


@dataclass(frozen=True, slots=True)
class WalkForwardRequest:
    """How the backtest is run. Every field lands in the artifact."""

    horizons: tuple[int, ...] = (5, 21, 63)
    #: Sessions between successive forecasts. Weekly by default.
    step: int = 5
    min_observations: int = DEFAULT_MIN_OBSERVATIONS
    start: pd.Timestamp | None = None
    #: Lower-tail quantiles to test coverage at, i.e. 95% and 99% VaR.
    var_levels: tuple[float, ...] = (0.05, 0.01)
    reference_moves: tuple[float, ...] = (0.05, 0.10)
    seed: int = 20260803
    pit_bins: int = DEFAULT_PIT_BINS
    reliability_bins: int = DEFAULT_BINS

    def __post_init__(self) -> None:
        if not self.horizons:
            raise ValueError("a walk-forward needs at least one horizon")
        if min(self.horizons) < 1:
            raise ValueError(f"horizons must be positive, got {self.horizons}")
        if max(self.horizons) >= _SEED_STRIDE:
            raise ValueError(f"horizons must stay below {_SEED_STRIDE} to keep seeds distinct")
        if self.step < 1:
            raise ValueError(f"step must be positive, got {self.step}")
        for level in self.var_levels:
            if not 0.0 < level < 1.0:
                raise ValueError(f"VaR levels are tail probabilities in (0, 1), got {level}")

    def sampling_for(self, horizon: int) -> Sampling:
        return Sampling(horizon=horizon, step=self.step)

    def describe(self) -> dict[str, Any]:
        return {
            "horizons": list(self.horizons),
            "step_sessions": self.step,
            "min_observations": self.min_observations,
            "start": None if self.start is None else self.start.date().isoformat(),
            "var_levels": list(self.var_levels),
            "reference_moves": list(self.reference_moves),
            "seed": self.seed,
            "window": "expanding",
        }


@dataclass(frozen=True, slots=True)
class ScoreRecord:
    """One forecast, graded. The unit the whole report is built from."""

    as_of: pd.Timestamp
    horizon: int
    anchor: float
    realised: float
    pit: float
    crps: float
    crps_baseline: float
    #: Label -> CRPS of a further null run alongside the required one. See
    #: :class:`~aurex.score.forecasters.RandomWalkForecaster` for why there is more
    #: than one sensible null.
    crps_alternatives: dict[str, float] = field(default_factory=dict)
    #: Tail level -> did the realised terminal price fall below that forecast quantile.
    breaches: dict[float, bool] = field(default_factory=dict)
    #: Event id -> (forecast probability, what happened).
    events: dict[str, tuple[float, bool]] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class Skipped:
    """A date and horizon that produced no score, and why. Never a silent gap."""

    as_of: pd.Timestamp
    horizon: int | None
    reason: str


@dataclass(frozen=True, slots=True)
class LevelCoverage:
    """Breach behaviour at one tail level."""

    level: float
    breaches: int
    rate_test: TestResult
    independence_test: TestResult
    conditional_test: TestResult

    def describe(self) -> dict[str, Any]:
        return {
            "level": self.level,
            "breaches": self.breaches,
            "kupiec": self.rate_test.describe(),
            "christoffersen_independence": self.independence_test.describe(),
            "conditional_coverage": self.conditional_test.describe(),
        }


@dataclass(frozen=True, slots=True)
class HorizonCalibration:
    """Everything scored at one horizon."""

    horizon: int
    sampling: Sampling
    n: int
    n_independent: int
    first_as_of: pd.Timestamp
    last_as_of: pd.Timestamp
    pit_bins: tuple[int, ...]
    pit_mean: float
    pit_uniformity: UniformityTest | None
    pit_chi_square: GoodnessOfFit | None
    pit_uniformity_reason: str | None
    crps_model: float
    crps_baseline: float
    crps_alternatives: dict[str, float]
    coverage: tuple[LevelCoverage, ...]
    events: tuple[tuple[dict[str, Any], ReliabilityCurve], ...]
    #: What the model said "ends higher" was worth, against what happened. Kept beside
    #: the PIT because the two read the same displacement from different reference
    #: points — see :mod:`aurex.score.drift`.
    direction_forecast_mean: float
    direction_observed_rate: float

    @property
    def skill(self) -> float:
        return crps_skill(self.crps_model, self.crps_baseline)

    def describe(self) -> dict[str, Any]:
        return {
            "horizon_sessions": self.horizon,
            "observations": self.n,
            "independent_observations": self.n_independent,
            "sampling": self.sampling.describe(),
            "first_as_of": self.first_as_of.date().isoformat(),
            "last_as_of": self.last_as_of.date().isoformat(),
            "pit": {
                "bins": list(self.pit_bins),
                "mean": round(self.pit_mean, 5),
                "uniformity": (
                    None if self.pit_uniformity is None else self.pit_uniformity.describe()
                ),
                "goodness_of_fit": (
                    None if self.pit_chi_square is None else self.pit_chi_square.describe()
                ),
                "uniformity_unavailable": self.pit_uniformity_reason,
                "note": (
                    "Histogram over every forecast; both tests over the non-overlapping "
                    "subsample only. KS and chi-square are reported together because "
                    "they fail on different shapes."
                ),
            },
            "direction": {
                "forecast_mean": round(self.direction_forecast_mean, 5),
                "observed_rate": round(self.direction_observed_rate, 5),
                "gap": round(self.direction_observed_rate - self.direction_forecast_mean, 5),
                "note": (
                    "Measured against spot rather than against the forecast centre, so "
                    "this gap reads the whole sample drift where the PIT mean reads "
                    "only what the forecast centre failed to carry."
                ),
            },
            "crps": {
                "model": round(self.crps_model, 4),
                "random_walk": round(self.crps_baseline, 4),
                "skill_score": round(self.skill, 4),
                "alternative_nulls": {
                    label: {
                        "crps": round(value, 4),
                        "skill_score": round(crps_skill(self.crps_model, value), 4),
                    }
                    for label, value in sorted(self.crps_alternatives.items())
                },
                "units": "price",
                "estimator": "fair (ensemble-size bias removed)",
                "reading": (
                    "Positive skill means the model beat a driftless random walk with "
                    "empirical increments. Zero means it tied. Negative means it lost, "
                    "and a negative number here is published rather than withheld."
                ),
            },
            "coverage": [entry.describe() for entry in self.coverage],
            "events": [{**description, **curve.describe()} for description, curve in self.events],
        }


@dataclass(frozen=True, slots=True)
class CalibrationReport:
    """The artifact block: what was run, over what, and how it scored."""

    subject: dict[str, Any]
    baseline: dict[str, Any]
    request: WalkForwardRequest
    horizons: tuple[HorizonCalibration, ...]
    skipped: tuple[Skipped, ...]
    observations: int
    drift_displacement: DriftDisplacement | None

    def describe(self) -> dict[str, Any]:
        return {
            "subject": self.subject,
            "baseline": self.baseline,
            "walk_forward": self.request.describe(),
            "scored_forecasts": self.observations,
            "drift_displacement": (
                None if self.drift_displacement is None else self.drift_displacement.describe()
            ),
            "horizons": [entry.describe() for entry in self.horizons],
            "skipped": [
                {
                    "as_of": entry.as_of.date().isoformat(),
                    "horizon": entry.horizon,
                    "reason": entry.reason,
                }
                for entry in self.skipped
            ],
            "conventions": {
                "monitoring": "session_close",
                "lookahead": "none: each fit sees prices up to and including its as-of date",
                "window": "expanding",
                "independence": (
                    "Every p-value is computed on the non-overlapping subsample. Means "
                    "and histograms use every forecast."
                ),
            },
        }


@dataclass(frozen=True, slots=True)
class WalkForwardResult:
    """The graded forecasts, before anything is aggregated."""

    records: tuple[ScoreRecord, ...]
    skipped: tuple[Skipped, ...]
    request: WalkForwardRequest
    subject: dict[str, Any]
    baseline: dict[str, Any]
    events: tuple[BinaryEvent, ...]

    def for_horizon(self, horizon: int) -> tuple[ScoreRecord, ...]:
        return tuple(record for record in self.records if record.horizon == horizon)

    def calibration(self) -> CalibrationReport:
        """Aggregate the records into the scored report."""
        horizons = []
        for horizon in self.request.horizons:
            records = self.for_horizon(horizon)
            if not records:
                continue
            horizons.append(_calibrate(horizon, records, self.request, self.events))

        return CalibrationReport(
            subject=self.subject,
            baseline=self.baseline,
            request=self.request,
            horizons=tuple(horizons),
            skipped=self.skipped,
            observations=len(self.records),
            drift_displacement=displacement(
                tuple(
                    HorizonDisplacement(
                        horizon=entry.horizon, mean_pit=entry.pit_mean, observations=entry.n
                    )
                    for entry in horizons
                )
            ),
        )


def walk_forward(
    prices: pd.Series,
    *,
    subject: AsOfForecaster,
    baseline: AsOfForecaster,
    request: WalkForwardRequest | None = None,
    events: tuple[BinaryEvent, ...] | None = None,
    extra_baselines: tuple[AsOfForecaster, ...] = (),
) -> WalkForwardResult:
    """Score ``subject`` against ``baseline`` over the history in ``prices``.

    ``baseline`` stays required and singular: a run must carry §0's null. Further nulls
    go in ``extra_baselines`` and are reported beside it rather than instead of it, so
    adding one can never remove the comparison the project committed to.
    """
    ask = request or WalkForwardRequest()
    graded = events if events is not None else default_events(ask.reference_moves)

    clean = prices.dropna().astype(float)
    if not isinstance(clean.index, pd.DatetimeIndex):
        raise TypeError("prices must be indexed by date to be walked forward")
    if not clean.index.is_monotonic_increasing:
        raise ValueError("prices must be sorted by date; a walk-forward cannot reorder them")

    total = int(clean.size)
    first = ask.min_observations
    if ask.start is not None:
        first = max(first, int(clean.index.searchsorted(ask.start)))

    records: list[ScoreRecord] = []
    skipped: list[Skipped] = []
    values = clean.to_numpy(dtype=float)

    for position in range(first, total, ask.step):
        as_of = clean.index[position]
        usable = tuple(h for h in ask.horizons if position + h < total)
        if not usable:
            break

        history = clean.iloc[: position + 1]
        seed = ask.seed + position * _SEED_STRIDE

        try:
            subject_ensembles = subject.forecast(history, horizons=usable, seed=seed)
            baseline_ensembles = baseline.forecast(
                history, horizons=usable, seed=seed + _BASELINE_OFFSET
            )
            alternatives = {
                extra.label: extra.forecast(
                    history, horizons=usable, seed=seed + _BASELINE_OFFSET * (index + 2)
                )
                for index, extra in enumerate(extra_baselines)
            }
        except InsufficientDataError as exc:
            skipped.append(Skipped(as_of=as_of, horizon=None, reason=str(exc)))
            continue

        anchor = float(values[position])
        for horizon in usable:
            realised = RealisedPath(values[position + 1 : position + 1 + horizon])
            records.append(
                _grade(
                    as_of=as_of,
                    horizon=horizon,
                    anchor=anchor,
                    realised=realised,
                    forecast=subject_ensembles[horizon],
                    null=baseline_ensembles[horizon],
                    alternatives={
                        label: ensembles[horizon] for label, ensembles in alternatives.items()
                    },
                    request=ask,
                    events=graded,
                    position=position,
                )
            )

    return WalkForwardResult(
        records=tuple(records),
        skipped=tuple(skipped),
        request=ask,
        subject=subject.describe(),
        baseline=baseline.describe(),
        events=graded,
    )


def _grade(
    *,
    as_of: pd.Timestamp,
    horizon: int,
    anchor: float,
    realised: RealisedPath,
    forecast: PathEnsemble,
    null: PathEnsemble,
    alternatives: dict[str, PathEnsemble],
    request: WalkForwardRequest,
    events: tuple[BinaryEvent, ...],
    position: int,
) -> ScoreRecord:
    terminal = forecast.terminal()
    observed = realised.terminal

    # Seeded per record rather than per run, so a PIT value does not depend on how many
    # dates were scored before it and a re-run of one date reproduces its own number.
    rng = np.random.default_rng([request.seed, position, horizon])

    return ScoreRecord(
        as_of=as_of,
        horizon=horizon,
        anchor=anchor,
        realised=observed,
        pit=pit_value(terminal, observed, rng=rng),
        crps=crps(terminal, observed),
        crps_baseline=crps(null.terminal(), observed),
        crps_alternatives={
            label: crps(ensemble.terminal(), observed) for label, ensemble in alternatives.items()
        },
        # Read off the quantile directly rather than derived from the PIT: the PIT is
        # randomised at ties, and a coverage test must not inherit that randomisation.
        breaches={
            level: bool(observed < float(np.quantile(terminal, level)))
            for level in request.var_levels
        },
        events={
            event.id: (event.probability(forecast), event.occurred(anchor, realised))
            for event in events
        },
    )


def _calibrate(
    horizon: int,
    records: tuple[ScoreRecord, ...],
    request: WalkForwardRequest,
    events: tuple[BinaryEvent, ...],
) -> HorizonCalibration:
    sampling = request.sampling_for(horizon)
    independent = sampling.independent()

    pits = np.array([record.pit for record in records], dtype=float)
    thinned_pits = sampling.thin(pits)

    uniformity_result: UniformityTest | None = None
    goodness_of_fit: GoodnessOfFit | None = None
    uniformity_reason: str | None = None
    if thinned_pits.size >= 2:
        uniformity_result = uniformity(thinned_pits, sampling=independent)
        goodness_of_fit = chi_square_uniformity(
            thinned_pits, sampling=independent, bins=request.pit_bins
        )
    else:
        uniformity_reason = (
            f"only {thinned_pits.size} non-overlapping observations at this horizon; "
            "a uniformity test needs at least two"
        )

    coverage: list[LevelCoverage] = []
    for level in request.var_levels:
        flags = np.array([record.breaches[level] for record in records], dtype=bool)
        thinned = sampling.thin(flags)
        coverage.append(
            LevelCoverage(
                level=level,
                breaches=int(np.count_nonzero(thinned)),
                rate_test=kupiec(thinned, expected_rate=level, sampling=independent),
                independence_test=christoffersen_independence(
                    thinned, expected_rate=level, sampling=independent
                ),
                conditional_test=conditional_coverage(
                    thinned, expected_rate=level, sampling=independent
                ),
            )
        )

    scored_events: list[tuple[dict[str, Any], ReliabilityCurve]] = []
    for event in events:
        probabilities = np.array([record.events[event.id][0] for record in records], dtype=float)
        outcomes = np.array([record.events[event.id][1] for record in records], dtype=float)
        scored_events.append(
            (
                event.describe(),
                reliability_curve(probabilities, outcomes, bins=request.reliability_bins),
            )
        )

    direction = TerminalAbove().id
    direction_forecast = [
        record.events[direction][0] for record in records if direction in record.events
    ]
    direction_observed = [
        record.events[direction][1] for record in records if direction in record.events
    ]

    return HorizonCalibration(
        horizon=horizon,
        sampling=sampling,
        n=len(records),
        n_independent=int(thinned_pits.size),
        first_as_of=records[0].as_of,
        last_as_of=records[-1].as_of,
        pit_bins=tuple(pit_histogram(pits, bins=request.pit_bins)),
        pit_mean=float(np.mean(pits)),
        pit_uniformity=uniformity_result,
        pit_chi_square=goodness_of_fit,
        pit_uniformity_reason=uniformity_reason,
        direction_forecast_mean=float(np.mean(direction_forecast)) if direction_forecast else 0.0,
        direction_observed_rate=float(np.mean(direction_observed)) if direction_observed else 0.0,
        crps_model=float(np.mean([record.crps for record in records])),
        crps_baseline=float(np.mean([record.crps_baseline for record in records])),
        crps_alternatives={
            label: float(np.mean([record.crps_alternatives[label] for record in records]))
            for label in records[0].crps_alternatives
        },
        coverage=tuple(coverage),
        events=tuple(scored_events),
    )
