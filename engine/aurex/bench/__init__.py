"""The step 6 benchmark shootout: the ``bench`` extra, and nothing that needs it by default.

Importing this package is free. Every dependency that only the shootout needs — torch,
statsforecast, neuralforecast, chronos, about 2.5GB between them — is imported inside
the method that uses it, so ``import aurex`` and the default test suite never pay for a
stack they do not run. Nothing in :mod:`aurex.score` imports this package, and the CI
job that installs it is a separate manually-triggered workflow.

The statistics are the other way round. Hansen's SPA, the Model Confidence Set, the
minimum detectable effect and the resolution screen live in :mod:`aurex.score.shootout`
and depend on nothing heavier than SciPy, because they are the part that has to be tested
on every push. A multiple-comparisons correction that is only exercised when somebody
manually runs a two-hour job is a correction nobody is checking.

There are two runs here, not one. ``aurex bench`` grades distributions with every model
centred; ``aurex direction`` grades the binary "ends higher" with every model carrying
the drift it infers, against the drift-matched walk. They cannot be merged: centred
forecasts put P(up) at about one half by construction, so the CRPS run has no directional
claim to grade, and an uncentred CRPS table would be measuring eleven years of
appreciation.
"""

from __future__ import annotations

from aurex.bench.runner import (
    BENCHMARK,
    CHALLENGERS,
    MixedDriftPolicyError,
    ShootoutRun,
    build_forecasters,
    describe_direction,
    describe_shootout,
    run_shootout,
)

__all__ = [
    "BENCHMARK",
    "CHALLENGERS",
    "MixedDriftPolicyError",
    "ShootoutRun",
    "build_forecasters",
    "describe_direction",
    "describe_shootout",
    "run_shootout",
]
