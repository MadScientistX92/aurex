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

**Every forecaster is graded on every event, not just the subject.** This used to be a
CRPS-only courtesy: extra baselines were asked for a distribution, scored on it, and
never asked what they thought "ends higher" was worth. That made per-model directional
accuracy unmeasurable, which is the one part of §0's benchmark promise that went
ungraded through step 6 — and it was a limit of the harness rather than a finding. Each
ensemble is now asked the same events on the same date and the outcome is stored once,
because what happened after an as-of date is a property of the price and not of who
forecast it.

**Overlap is recorded, not assumed away.** Forecasting every ``step`` sessions over a
longer horizon produces records that share most of their path. Means and histograms are
computed over all of them; every p-value is computed over the thinned, non-overlapping
subsample, and both counts are reported. See :mod:`aurex.score.sampling`. The one
exception is the Diebold-Mariano test, which handles the dependence in its variance
estimator instead of thinning it away, and is run on both series so the two can be
compared — see :mod:`aurex.score.significance`.

**No skill score leaves here without a test beside it.** A skill score is a difference
of two sample means, and a difference of two sample means is not a result. Every null a
run carries is reported as a :class:`SkillTest`, which cannot be constructed without the
Diebold-Mariano statistic, its p-value and its observation count, so "the model wins by
4.6%" and "the model is worth nothing" are held to the same standard of evidence.
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
from aurex.score.significance import (
    DieboldMariano,
    HorizonLosses,
    SkillDecay,
    diebold_mariano,
    skill_decay,
)
from aurex.score.tail import (
    TailCalibration,
    driftless_gaussian_probability,
    tail_calibration,
)
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
    #: Last observation the run may see, for forecasting *or* for scoring.
    #:
    #: Without it a run ends wherever wall-clock left it, and a published number cannot
    #: be reproduced by anyone who reads it later — the sample they get is not the
    #: sample it came from. Truncating the price series rather than filtering the as-of
    #: dates is deliberate: it bounds the realised outcomes too, so a forecast whose
    #: horizon would run past the stated end is dropped rather than scored against data
    #: the stated window says is not there.
    end: pd.Timestamp | None = None
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
        if self.start is not None and self.end is not None and self.end < self.start:
            raise ValueError(f"sample ends {self.end} before it starts {self.start}")
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
            "end": None if self.end is None else self.end.date().isoformat(),
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
    #: Event id -> §0's null's own probability for it.
    #:
    #: A field of its own rather than one entry in ``event_alternatives``, mirroring
    #: ``crps_baseline`` beside ``crps_alternatives``: the required null cannot be
    #: dropped by passing no extras, so a run that grades an event grades the benchmark
    #: on it too.
    events_baseline: dict[str, float] = field(default_factory=dict)
    #: Label -> event id -> that null's forecast probability for the same event.
    #:
    #: The outcome is not repeated here because it cannot differ: what happened after a
    #: given as-of date is a property of the price, not of who forecast it. Storing it
    #: per model would create six places for one fact to disagree with itself, and a
    #: model graded against its own copy of the outcome is not being compared with
    #: anything. Read the truth from ``events`` and the probability from here.
    event_alternatives: dict[str, dict[str, float]] = field(default_factory=dict)
    #: Standard deviation of the model's own simulated terminal log return.
    #:
    #: The engine's forecast *level* of volatility for this window, separated from the
    #: shape of the distribution carrying it. A Gaussian reference given this number
    #: differs from the model in kurtosis and skew alone, which is the only way to ask
    #: whether the tail is the right shape without the answer being contaminated by the
    #: two forecasters disagreeing about how volatile the week was.
    terminal_log_sd: float = 0.0
    #: Standard deviation of daily log returns over the history available at ``as_of``.
    #:
    #: What a Gaussian forecaster with no lookahead would have had to work with. Recorded
    #: per window rather than once per run because it is an expanding-window quantity and
    #: collapsing it to a scalar would import the end of the sample into its start.
    sigma_expanding: float = 0.0


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
class SkillTest:
    """A skill score against one null, with the test that says whether it means anything.

    Both Diebold-Mariano runs are required fields rather than optional extras, so a
    skill score cannot be constructed here without them: ``overlapping`` uses the HAC
    variance on every record, ``thinned`` repeats it on the non-overlapping subsample.
    They answer the same question with different tolerance for the sampling scheme, and
    publishing both is what the existing p-value discipline asks for — the thinned run
    is the one that assumes nothing about the dependence, and the overlapping run is the
    one with the observations.
    """

    null: str
    crps_model: float
    crps_null: float
    overlapping: DieboldMariano
    thinned: DieboldMariano

    @property
    def skill(self) -> float:
        return crps_skill(self.crps_model, self.crps_null)

    @property
    def significant(self) -> bool:
        """Both runs reject at 5%, and agree on the sign. Anything less is not a result."""
        first, second = self.overlapping, self.thinned
        if first.p_value is None or second.p_value is None:
            return False
        return (
            first.p_value < 0.05
            and second.p_value < 0.05
            and first.favours_model == second.favours_model
        )

    def describe(self) -> dict[str, Any]:
        return {
            "crps": round(self.crps_null, 4),
            "skill_score": round(self.skill, 4),
            "significance": {
                "overlapping_windows_hac": self.overlapping.describe(),
                "non_overlapping_subsample": self.thinned.describe(),
                "distinguishable_from_zero": self.significant,
                "reading": (
                    "The skill score is a difference of two sample means and the test "
                    "is what decides whether that difference is a finding. Both runs "
                    "must reject and agree on the sign before this reads as one."
                ),
            },
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
    #: One per null, the required baseline first. Never empty: a run always carries §0's
    #: null, so there is always at least one tested skill score here.
    skill_tests: tuple[SkillTest, ...]
    coverage: tuple[LevelCoverage, ...]
    #: Per event: its description, its reliability curve, and the count test that says
    #: whether the gap between the two is a finding or a coincidence.
    events: tuple[tuple[dict[str, Any], ReliabilityCurve, TailCalibration], ...]
    #: What the model said "ends higher" was worth, against what happened. Kept beside
    #: the PIT because the two read the same displacement from different reference
    #: points — see :mod:`aurex.score.drift`.
    direction_forecast_mean: float
    direction_observed_rate: float

    @property
    def skill(self) -> float:
        return crps_skill(self.crps_model, self.crps_baseline)

    @property
    def baseline_test(self) -> SkillTest:
        """The test of §0's own null, which every run is required to carry."""
        return self.skill_tests[0]

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
                "significance": self.baseline_test.describe()["significance"],
                "alternative_nulls": {test.null: test.describe() for test in self.skill_tests[1:]},
                "units": "price",
                "estimator": "fair (ensemble-size bias removed)",
                "reading": (
                    "Positive skill means the model beat a driftless random walk with "
                    "empirical increments. Zero means it tied. Negative means it lost, "
                    "and a negative number here is published rather than withheld. The "
                    "sign is only half of it: read the Diebold-Mariano p-value beside "
                    "it before treating any of these as a finding."
                ),
            },
            "coverage": [entry.describe() for entry in self.coverage],
            "events": [
                {**description, **curve.describe(), **tail.describe()}
                for description, curve, tail in self.events
            ],
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
    #: Whether skill falls with the horizon, which is what mean-reverting conditional
    #: variance predicts. A cross-horizon question, so it lives here rather than on any
    #: one horizon.
    skill_decay: SkillDecay | None
    #: Realised daily log-return volatility of the scored window, published because one
    #: tail reference is built from it and a reader cannot audit that reference without
    #: the number behind it.
    sample_sigma: float = 0.0

    def describe(self) -> dict[str, Any]:
        return {
            "subject": self.subject,
            "baseline": self.baseline,
            "walk_forward": self.request.describe(),
            "scored_forecasts": self.observations,
            "drift_displacement": (
                None if self.drift_displacement is None else self.drift_displacement.describe()
            ),
            "skill_decay": (None if self.skill_decay is None else self.skill_decay.describe()),
            "scored_window_sigma": {
                "daily_log_return_sd": round(self.sample_sigma, 6),
                "note": (
                    "Realised over the scored window, so it was not available on the "
                    "first as-of date. The gaussian_sample_sigma tail reference is built "
                    "from it and is labelled as using lookahead for that reason; "
                    "gaussian_expanding_sigma is the same reference without it."
                ),
            },
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
class SampleWindow:
    """The price history a run actually saw, and the bounds that were asked for.

    Published so a number carries the sample that produced it. Both halves matter and
    they are not the same: ``requested_end`` is what the caller typed and may be
    ``None``; ``resolved_end`` is the last observation the run could see, which is what
    a reader has to pass back to get these numbers again. Recording only the request
    would leave the default case — no ``--to`` at all — undocumented, and that is the
    case every published number here came from.
    """

    series_id: str
    requested_start: pd.Timestamp | None
    requested_end: pd.Timestamp | None
    resolved_start: pd.Timestamp
    resolved_end: pd.Timestamp
    observations: int
    #: Where the underlying series ends regardless of truncation, so a reader can see
    #: how much data was deliberately held back.
    available_end: pd.Timestamp

    @property
    def truncated(self) -> bool:
        return self.resolved_end < self.available_end

    def describe(self) -> dict[str, Any]:
        def stamp(value: pd.Timestamp | None) -> str | None:
            return None if value is None else str(value.date().isoformat())

        return {
            "series_id": self.series_id,
            "requested": {
                "from": stamp(self.requested_start),
                "to": stamp(self.requested_end),
            },
            "resolved": {
                "from": stamp(self.resolved_start),
                "to": stamp(self.resolved_end),
            },
            "observations": self.observations,
            "series_available_to": stamp(self.available_end),
            "truncated": self.truncated,
            "note": (
                "resolved.to is the last observation this run could see, for "
                "forecasting and for scoring alike. Pass it back as --to to reproduce "
                "these numbers: without it the run ends at whatever the data happened "
                "to reach on the day, and the sample is not the one these numbers came "
                "from."
            ),
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
    #: The price history this run actually saw, after any requested truncation.
    sample: SampleWindow | None = None
    #: Realised daily log-return volatility over the scored window itself.
    #:
    #: The input to the one tail reference that carries lookahead, and the reason that
    #: reference is labelled as carrying it: this number is the answer to a question
    #: nobody could have asked on the first as-of date. It is computed and published
    #: because it is what the README's tail comparison was built on, and testing a
    #: published claim means testing the benchmark it actually used.
    sample_sigma: float = 0.0

    def for_horizon(self, horizon: int) -> tuple[ScoreRecord, ...]:
        return tuple(record for record in self.records if record.horizon == horizon)

    def calibration(self) -> CalibrationReport:
        """Aggregate the records into the scored report."""
        horizons = []
        for horizon in self.request.horizons:
            records = self.for_horizon(horizon)
            if not records:
                continue
            horizons.append(
                _calibrate(horizon, records, self.request, self.events, self.sample_sigma)
            )

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
            skill_decay=self._skill_decay(),
            sample_sigma=self.sample_sigma,
        )

    def _skill_decay(self) -> SkillDecay | None:
        """Skill against §0's null, regressed on log horizon and block-bootstrapped.

        The horizons are read at their *shared* as-of dates. A longer horizon runs off
        the end of the sample sooner, so the raw record counts differ, and resampling
        series of different lengths independently would break the very dependence the
        bootstrap exists to preserve.
        """
        by_horizon = {
            horizon: {record.as_of: record for record in self.for_horizon(horizon)}
            for horizon in self.request.horizons
        }
        populated = {horizon: rows for horizon, rows in by_horizon.items() if rows}
        if len(populated) < 3:
            return None

        shared = sorted(set.intersection(*(set(rows) for rows in populated.values())))
        if not shared:
            return None

        points = tuple(
            HorizonLosses(
                horizon=horizon,
                model=np.array([rows[date].crps for date in shared], dtype=float),
                baseline=np.array([rows[date].crps_baseline for date in shared], dtype=float),
            )
            for horizon, rows in sorted(populated.items())
        )
        # Blocks as long as the widest overlap, so a resampled block carries the
        # dependence between neighbouring records instead of shuffling it out.
        block = max(self.request.sampling_for(horizon).stride for horizon in populated)
        return skill_decay(points, block=block, seed=self.request.seed)


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
    if clean.empty:
        raise ValueError("no observations to walk forward over")

    available_end = clean.index[-1]
    if ask.end is not None:
        # Truncate the series rather than filter the as-of dates: this has to bound the
        # realised outcomes too, or a forecast near the stated end would be scored
        # against prices the declared window says are not in the sample.
        clean = clean.loc[: ask.end]
        if clean.empty:
            raise ValueError(
                f"no observations at or before {ask.end.date()}; the series begins "
                f"{prices.dropna().index[0].date()}"
            )

    total = int(clean.size)
    first = ask.min_observations
    if ask.start is not None:
        first = max(first, int(clean.index.searchsorted(ask.start)))

    records: list[ScoreRecord] = []
    skipped: list[Skipped] = []
    values = clean.to_numpy(dtype=float)

    # Log returns once, then sliced per as-of date. The expanding standard deviation is
    # what a no-lookahead Gaussian reference gets; recomputing it inside the loop from
    # the price series would be the same arithmetic done 580 times over.
    log_returns = np.diff(np.log(values)) if values.size > 1 else np.zeros(0, dtype=float)

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
        # Returns strictly before the as-of date's own close are all a forecaster made
        # on that date could have seen.
        history_returns = log_returns[:position]
        sigma_expanding = (
            float(np.std(history_returns, ddof=1)) if history_returns.size > 1 else 0.0
        )

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
                    sigma_expanding=sigma_expanding,
                )
            )

    # Over the scored window rather than the whole history: this is the benchmark the
    # published tail comparison used, and it is the volatility of the sample the events
    # were counted in.
    scored_returns = log_returns[first:] if first < log_returns.size else np.zeros(0, dtype=float)
    sample_sigma = float(np.std(scored_returns, ddof=1)) if scored_returns.size > 1 else 0.0

    return WalkForwardResult(
        records=tuple(records),
        skipped=tuple(skipped),
        request=ask,
        subject=subject.describe(),
        baseline=baseline.describe(),
        events=graded,
        sample_sigma=sample_sigma,
        sample=SampleWindow(
            series_id=str(prices.name or "prices"),
            requested_start=ask.start,
            requested_end=ask.end,
            resolved_start=clean.index[0],
            resolved_end=clean.index[-1],
            observations=total,
            available_end=available_end,
        ),
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
    sigma_expanding: float = 0.0,
) -> ScoreRecord:
    terminal = forecast.terminal()
    observed = realised.terminal

    # Seeded per record rather than per run, so a PIT value does not depend on how many
    # dates were scored before it and a re-run of one date reproduces its own number.
    rng = np.random.default_rng([request.seed, position, horizon])

    # The model's forecast volatility for this window, in the log space it simulates in.
    # Read off the ensemble rather than off the fit, so it is the spread of what was
    # actually simulated — session limits and the bootstrap included — rather than the
    # spread the model intended before those touched it.
    positive = terminal[terminal > 0.0]
    terminal_log_sd = (
        float(np.std(np.log(positive / anchor), ddof=1))
        if anchor > 0.0 and positive.size > 1
        else 0.0
    )

    return ScoreRecord(
        as_of=as_of,
        horizon=horizon,
        anchor=anchor,
        realised=observed,
        terminal_log_sd=terminal_log_sd,
        sigma_expanding=sigma_expanding,
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
            if event.applies_at(horizon)
        },
        events_baseline={
            event.id: event.probability(null) for event in events if event.applies_at(horizon)
        },
        # Every extra null is asked the same events as the subject, on the same ensemble
        # it was already asked for CRPS. Nothing else here changes: the outcome, the
        # anchor and the horizon filter are shared, so a probability recorded under one
        # label is comparable with a probability recorded under another by construction.
        event_alternatives={
            label: {
                event.id: event.probability(ensemble)
                for event in events
                if event.applies_at(horizon)
            }
            for label, ensemble in alternatives.items()
        },
    )


def _gaussian_references(
    event: BinaryEvent,
    records: tuple[ScoreRecord, ...],
    *,
    horizon: int,
    sample_sigma: float,
) -> dict[str, np.ndarray]:
    """Closed-form driftless-Gaussian probabilities for the same event, three ways.

    Empty for a path-monitored event. The reflection principle prices a barrier watched
    continuously, and this engine watches at session close on both sides, so a
    continuous-monitoring reference would be answering a question neither the forecast
    nor the outcome was measured on — see :mod:`aurex.score.tail`.
    """
    if event.monitoring != "terminal":
        return {}

    multiples = np.array(
        [
            event.level(record.anchor) / record.anchor if record.anchor > 0.0 else np.nan
            for record in records
        ],
        dtype=float,
    )
    matched = np.array([record.terminal_log_sd for record in records], dtype=float)
    expanding = np.array([record.sigma_expanding for record in records], dtype=float)
    fixed = np.full(len(records), sample_sigma, dtype=float)

    return {
        "gaussian_matched_sigma": driftless_gaussian_probability(
            multiples, matched / np.sqrt(horizon), horizon=horizon
        ),
        "gaussian_expanding_sigma": driftless_gaussian_probability(
            multiples, expanding, horizon=horizon
        ),
        "gaussian_sample_sigma": driftless_gaussian_probability(multiples, fixed, horizon=horizon),
    }


def _calibrate(
    horizon: int,
    records: tuple[ScoreRecord, ...],
    request: WalkForwardRequest,
    events: tuple[BinaryEvent, ...],
    sample_sigma: float,
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

    model_losses = np.array([record.crps for record in records], dtype=float)
    skill_tests = [
        _skill_test(
            "random_walk",
            model_losses,
            np.array([record.crps_baseline for record in records], dtype=float),
            sampling=sampling,
        )
    ]
    skill_tests.extend(
        _skill_test(
            label,
            model_losses,
            np.array([record.crps_alternatives[label] for record in records], dtype=float),
            sampling=sampling,
        )
        for label in sorted(records[0].crps_alternatives)
    )

    scored_events: list[tuple[dict[str, Any], ReliabilityCurve, TailCalibration]] = []
    for event in events:
        if not event.applies_at(horizon):
            continue
        probabilities = np.array([record.events[event.id][0] for record in records], dtype=float)
        outcomes = np.array([record.events[event.id][1] for record in records], dtype=float)
        scored_events.append(
            (
                event.describe(),
                reliability_curve(probabilities, outcomes, bins=request.reliability_bins),
                tail_calibration(
                    event_id=event.id,
                    probabilities=probabilities,
                    outcomes=outcomes,
                    sampling=sampling,
                    gaussian_references=_gaussian_references(
                        event, records, horizon=horizon, sample_sigma=sample_sigma
                    ),
                ),
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
        skill_tests=tuple(skill_tests),
        coverage=tuple(coverage),
        events=tuple(scored_events),
    )


def _skill_test(
    label: str, model: np.ndarray, null: np.ndarray, *, sampling: Sampling
) -> SkillTest:
    """One null, tested twice: HAC on every record, then again on the thinned subsample."""
    independent = sampling.independent()
    return SkillTest(
        null=label,
        crps_model=float(np.mean(model)),
        crps_null=float(np.mean(null)),
        overlapping=diebold_mariano(model, null, sampling=sampling, null=label),
        thinned=diebold_mariano(
            sampling.thin(model), sampling.thin(null), sampling=independent, null=label
        ),
    )
