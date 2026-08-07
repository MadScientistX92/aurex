"""The one property the shootout cannot be wrong about: every model is driftless.

A model carrying drift scored against a null denied one was worth up to +4.6% of CRPS
skill on the first asset this project measured, all of it eleven years of appreciation.
That result was withdrawn. A six-model shootout is exactly where the error would come
back, because three of the challengers fit a conditional mean by construction and one
infers a trend from its context window — so the policy is one shared function and this
file is the test that it is actually applied.

These tests import ``aurex.bench.adapters`` and never call a model. The heavy stack lives
behind method-local imports, so the drift mechanism is testable on a default install even
though the forecasters themselves are not — which is the point of putting the imports
there in the first place.
"""

from __future__ import annotations

import numpy as np
import pytest

from aurex.assets.transforms import LogReturn
from aurex.bench.adapters import (
    InsufficientResidualsError,
    _bootstrap_residuals,
    _centre,
    _ensemble,
)


class TestTheSharedDriftPolicy:
    def test_centring_removes_the_mean(self) -> None:
        rng = np.random.default_rng(1)
        # A strongly trending set of simulated returns, as an ARIMA with a fitted drift
        # or a foundation model extrapolating a rally would produce.
        simulated = rng.normal(0.004, 0.01, size=(500, 21))

        centred = _centre(simulated)

        assert float(np.mean(simulated)) > 0.003
        assert float(np.mean(centred)) == pytest.approx(0.0, abs=1e-15)

    def test_centring_preserves_the_shape_across_the_horizon(self) -> None:
        """It removes a level, not the model's dynamics. A per-step mean would flatten
        the conditional mean path itself, which is part of what these models forecast."""
        simulated = np.tile(np.array([0.01, 0.02, 0.03, 0.04]), (10, 1))

        centred = _centre(simulated)

        # The step-to-step differences survive; only the common level moved.
        assert np.allclose(np.diff(centred, axis=1), np.diff(simulated, axis=1))
        assert float(np.mean(centred)) == pytest.approx(0.0, abs=1e-15)

    def test_an_ensemble_built_here_carries_no_drift(self) -> None:
        """End to end: a drifting return set becomes a driftless price ensemble."""
        rng = np.random.default_rng(2)
        simulated = rng.normal(0.004, 0.01, size=(4_000, 21))

        ensemble = _ensemble(
            simulated,
            transform=LogReturn(),
            anchor=100.0,
            session_limit=None,
            detail={"vol_model": {"model": "probe"}},
        )

        terminal = ensemble.terminal()
        mean_log_return = float(np.mean(np.log(terminal / 100.0)))
        # Zero to within the Monte Carlo error of 4,000 paths, where the uncentred
        # version would sit at 21 * 0.004 = 0.084.
        assert mean_log_return == pytest.approx(0.0, abs=1e-12)

    def test_the_ensemble_declares_its_drift_policy(self) -> None:
        """The artifact has to say it, not just do it."""
        ensemble = _ensemble(
            np.full((10, 5), 0.01),
            transform=LogReturn(),
            anchor=100.0,
            session_limit=None,
            detail={},
        )

        pool = ensemble.diagnostics["residual_pool"]
        assert pool["demeaned"] is True
        assert "same drift policy as the null" in pool["drift"]

    def test_a_driftless_ensemble_is_left_alone(self) -> None:
        rng = np.random.default_rng(3)
        simulated = rng.normal(0.0, 0.01, size=(200, 10))

        centred = _centre(simulated)

        assert np.allclose(centred, simulated - float(np.mean(simulated)))
        assert float(np.abs(np.mean(centred))) < 1e-15


class TestTheResidualBootstrap:
    def test_it_draws_from_the_models_own_errors(self) -> None:
        residuals = np.array([-0.02, -0.01, 0.0, 0.01, 0.02])
        drawn = _bootstrap_residuals(
            residuals, n_paths=100, horizon=7, rng=np.random.default_rng(4)
        )

        assert drawn.shape == (100, 7)
        assert set(np.unique(drawn)).issubset(set(residuals.tolist()))

    def test_nan_residuals_are_dropped_rather_than_drawn(self) -> None:
        residuals = np.array([np.nan, 0.01, -0.01, np.nan])
        drawn = _bootstrap_residuals(residuals, n_paths=50, horizon=3, rng=np.random.default_rng(5))

        assert np.all(np.isfinite(drawn))

    def test_too_few_usable_residuals_is_refused(self) -> None:
        with pytest.raises(InsufficientResidualsError):
            _bootstrap_residuals(
                np.array([np.nan, 0.01]), n_paths=10, horizon=3, rng=np.random.default_rng(6)
            )
