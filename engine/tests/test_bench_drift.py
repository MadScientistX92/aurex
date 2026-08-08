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
import pandas as pd
import pytest

from aurex.assets.synthetic import SYNTHETIC
from aurex.assets.transforms import LogReturn
from aurex.bench import CHALLENGERS, MixedDriftPolicyError, build_forecasters
from aurex.bench.adapters import (
    AutoArimaForecaster,
    ChronosForecaster,
    InsufficientResidualsError,
    NhitsForecaster,
    _bootstrap_residuals,
    _centre,
    _ensemble,
)
from aurex.bench.runner import _require_one_drift_policy
from aurex.score import ModelForecaster, RandomWalkForecaster
from aurex.vol import model_for


def _prices(n: int = 400) -> pd.Series:
    """A price series only long enough to assemble against; nothing here fits a model."""
    rng = np.random.default_rng(99)
    values = 100.0 * np.exp(np.cumsum(rng.normal(0.0003, 0.01, n)))
    return pd.Series(values, index=pd.bdate_range("2020-01-01", periods=n), name="price")


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


class TestTheUncentredRunIsPossibleAndCannotBeMixed:
    """Direction needs the opposite policy, and mixing the two is the withdrawn error.

    Grading direction on centred forecasts grades the drift policy: every model returns
    P(up) of about one half by construction, so the answer is fixed before any model
    forecasts anything. The fix is to let every competitor carry the drift it infers and
    move the null to the drift-matched walk — not to abandon centring, which is still
    right for the CRPS comparison. What must never happen is one run holding both.
    """

    def test_an_uncentred_ensemble_keeps_the_drift_it_was_given(self) -> None:
        rng = np.random.default_rng(30)
        simulated = rng.normal(0.004, 0.01, size=(4_000, 21))

        ensemble = _ensemble(
            simulated,
            transform=LogReturn(),
            anchor=100.0,
            session_limit=None,
            detail={},
            demean=False,
        )

        mean_log_return = float(np.mean(np.log(ensemble.terminal() / 100.0)))
        # 21 sessions at +0.004 a session, where the centred version sits at zero.
        assert mean_log_return == pytest.approx(21 * 0.004, rel=0.05)

    def test_an_uncentred_ensemble_says_so_in_its_diagnostics(self) -> None:
        ensemble = _ensemble(
            np.full((10, 5), 0.01),
            transform=LogReturn(),
            anchor=100.0,
            session_limit=None,
            detail={},
            demean=False,
        )

        pool = ensemble.diagnostics["residual_pool"]
        assert pool["demeaned"] is False
        assert "graded on what the model inferred" in pool["drift"]

    def test_the_null_becomes_the_drift_matched_walk(self) -> None:
        """An uncentred run cannot report itself as having beaten §0's driftless null."""
        _, centred_null, _ = build_forecasters(
            SYNTHETIC,
            prices=_prices(),
            ohlc=None,
            breaks=(),
            n_paths=100,
            chronos_paths=10,
            nhits_steps=10,
            include=("gjr_garch",),
        )
        _, drifted_null, _ = build_forecasters(
            SYNTHETIC,
            prices=_prices(),
            ohlc=None,
            breaks=(),
            n_paths=100,
            chronos_paths=10,
            nhits_steps=10,
            include=("gjr_garch",),
            demean=False,
        )

        assert centred_null.label == "random_walk"
        assert drifted_null.label == "random_walk_drift_matched"

    def test_every_member_of_an_uncentred_set_carries_drift(self) -> None:
        subject, baseline, extras = build_forecasters(
            SYNTHETIC,
            prices=_prices(),
            ohlc=None,
            breaks=(),
            n_paths=100,
            chronos_paths=10,
            nhits_steps=10,
            include=CHALLENGERS,
            demean=False,
        )

        assert all(entry.carries_drift for entry in (subject, baseline, *extras))

    def test_a_mixed_set_is_refused_at_assembly(self) -> None:
        """The guard asks each forecaster what it does rather than trusting the keyword
        reached it, so a challenger that grew its own default fails here rather than
        posting skill that belongs to eleven years of appreciation."""
        centred = ChronosForecaster(transform=LogReturn(), n_paths=10)
        drifting = ChronosForecaster(transform=LogReturn(), n_paths=10, demean=False)

        with pytest.raises(MixedDriftPolicyError, match="uncentred"):
            _require_one_drift_policy((drifting, centred), demean=False)

        with pytest.raises(MixedDriftPolicyError, match="centred"):
            _require_one_drift_policy((centred, drifting), demean=True)

    def test_a_consistent_set_passes(self) -> None:
        centred = ChronosForecaster(transform=LogReturn(), n_paths=10)
        _require_one_drift_policy((centred,), demean=True)

    def test_carries_drift_is_derived_and_not_stored(self) -> None:
        """Every implementation reads it off the flag that governs its simulation, so it
        cannot be set inconsistently with the behaviour it describes."""
        for forecaster in (
            ChronosForecaster(transform=LogReturn()),
            NhitsForecaster(transform=LogReturn()),
            AutoArimaForecaster(transform=LogReturn()),
            RandomWalkForecaster(transform=LogReturn()),
            ModelForecaster(model=model_for("gjr_garch"), transform=LogReturn()),
        ):
            assert forecaster.carries_drift is False

        assert ChronosForecaster(transform=LogReturn(), demean=False).carries_drift is True
        assert RandomWalkForecaster(transform=LogReturn(), demean=False).carries_drift is True
        assert (
            ModelForecaster(
                model=model_for("gjr_garch"), transform=LogReturn(), demean_residuals=False
            ).carries_drift
            is True
        )
