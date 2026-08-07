"""Calibration and backtesting. Populated in step 3a.

PIT histograms, CRPS skill against a driftless random walk, Kupiec and Christoffersen
VaR breach tests, Brier scores and reliability diagrams. A forecast that is never
scored is marketing.

Everything here scores the *distribution*, not a position, so nothing in this package
may reach for friction, a route or a jurisdiction — the events it grades are direction
and touching a barrier, both properties of the asset. Step 3b adds one event whose
definition is friction (does the move clear breakeven), and it arrives as an
implementation of :class:`~aurex.score.events.BinaryEvent` fed to this machinery rather
than as a metric implemented beside it.

Three conventions belong to the scorer rather than to the caller, and each is enforced
by something that fails rather than by something that is remembered:

* **No lookahead** is a property of :class:`~aurex.score.forecasters.AsOfForecaster`,
  which receives a sliced history and never learns where it was cut.
* **A run always carries its null.** :func:`~aurex.score.walkforward.walk_forward`
  takes ``baseline`` as a required argument, so no CRPS leaves this package without
  the random walk's CRPS for the same date beside it.
* **Overlapping windows are not independent observations.** Every p-value goes through
  :func:`~aurex.score.sampling.require_independent`, which raises on a series that
  overlaps rather than quietly reporting a number that assumes it does not. The
  Diebold-Mariano test in :mod:`aurex.score.significance` is the single exception, and
  it earns it by modelling the dependence rather than ignoring it: it takes the same
  declaration and turns it into a HAC truncation lag.
* **A skill score without a test is not a result.** Every null is reported as a
  :class:`~aurex.score.walkforward.SkillTest`, whose Diebold-Mariano statistic, p-value
  and observation count are required fields. This binds in both directions — a win and
  a shrug are claims about the same population and neither publishes without evidence.

A realised touch is evaluated at session close, matching the forecast's own monitoring
convention; :class:`~aurex.score.events.RealisedPath` carries closes and nothing else,
so there is no intraday extreme available to accidentally score against.
"""

from __future__ import annotations

from aurex.score.coverage import (
    TestResult,
    christoffersen_independence,
    conditional_coverage,
    kupiec,
)
from aurex.score.distribution import (
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
from aurex.score.events import (
    BinaryEvent,
    ClearsHurdle,
    Monitoring,
    RealisedPath,
    TerminalAbove,
    TouchAbove,
    TouchBelow,
    default_events,
)
from aurex.score.forecasters import AsOfForecaster, ModelForecaster, RandomWalkForecaster
from aurex.score.reliability import (
    MIN_POSITIVE_EVENTS,
    ReliabilityBin,
    ReliabilityCurve,
    brier_score,
    reliability_curve,
)
from aurex.score.sampling import OverlappingWindowsError, Sampling, require_independent
from aurex.score.significance import (
    DieboldMariano,
    HorizonLosses,
    SkillDecay,
    diebold_mariano,
    skill_decay,
)
from aurex.score.tail import (
    TailCalibration,
    TailTest,
    driftless_gaussian_probability,
    poisson_binomial_pmf,
    tail_calibration,
    tail_test,
)
from aurex.score.walkforward import (
    CalibrationReport,
    HorizonCalibration,
    LevelCoverage,
    SampleWindow,
    ScoreRecord,
    SkillTest,
    Skipped,
    WalkForwardRequest,
    WalkForwardResult,
    walk_forward,
)

__all__ = [
    "MIN_POSITIVE_EVENTS",
    "AsOfForecaster",
    "BinaryEvent",
    "CalibrationReport",
    "ClearsHurdle",
    "DieboldMariano",
    "DriftDisplacement",
    "GoodnessOfFit",
    "HorizonCalibration",
    "HorizonDisplacement",
    "HorizonLosses",
    "LevelCoverage",
    "ModelForecaster",
    "Monitoring",
    "OverlappingWindowsError",
    "RandomWalkForecaster",
    "RealisedPath",
    "ReliabilityBin",
    "ReliabilityCurve",
    "SampleWindow",
    "Sampling",
    "ScoreRecord",
    "SkillDecay",
    "SkillTest",
    "Skipped",
    "TailCalibration",
    "TailTest",
    "TerminalAbove",
    "TestResult",
    "TouchAbove",
    "TouchBelow",
    "UniformityTest",
    "WalkForwardRequest",
    "WalkForwardResult",
    "brier_score",
    "chi_square_uniformity",
    "christoffersen_independence",
    "conditional_coverage",
    "crps",
    "crps_skill",
    "default_events",
    "diebold_mariano",
    "displacement",
    "driftless_gaussian_probability",
    "kupiec",
    "pit_histogram",
    "pit_value",
    "poisson_binomial_pmf",
    "reliability_curve",
    "require_independent",
    "skill_decay",
    "tail_calibration",
    "tail_test",
    "uniformity",
    "walk_forward",
]
