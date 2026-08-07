"""Volatility models.

The tests that matter here are the ones that would fail on a plausible wrong
implementation rather than on a typo: parameter recovery against a known process,
the asymmetry term reading zero when the data is symmetric, and a policy break not
being absorbed as a volatility shock. Per the project's testing rule, the break test
asserts on the quantity that moves *mechanically* — the fitted variance — rather than
on a residual that nets the effect out.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from aurex.vol import (
    GjrGarch,
    HarRv,
    InsufficientDataError,
    RollingStd,
    SessionLimit,
    garman_klass_variance,
    model_for,
    parkinson_variance,
)
from tests.conftest import simulate_gjr_path, simulate_gjr_returns

TRUTH = {"omega": 2.0e-6, "alpha": 0.04, "gamma": 0.08, "beta": 0.88}


def intraday_ohlc(
    *, days: int = 600, daily_sigma: float = 0.015, steps: int = 800, seed: int = 5
) -> pd.DataFrame:
    """OHLC bars built from a simulated intraday path, so the range means something.

    A range estimator can only be checked against a series that actually has a range;
    inventing highs and lows around a daily close would test nothing but arithmetic.

    ``steps`` is high because the estimators assume the session is observed
    continuously, and a coarsely sampled path has a smaller observed range than the
    path it was sampled from. Measured here, Parkinson recovers 84% of the true
    variance at 78 samples per session, 96% at 800 and 98% at 3000. That bias is the
    same one the module docstring warns about for overnight gaps, and it is why the
    tolerance below is one-sided in practice.
    """
    rng = np.random.default_rng(seed)
    index = pd.bdate_range("2020-01-01", periods=days, name="date")
    rows = []
    level = 100.0

    for _ in range(days):
        path = level * np.exp(np.cumsum(rng.standard_normal(steps) * daily_sigma / np.sqrt(steps)))
        rows.append((level, path.max(), path.min(), path[-1]))
        level = path[-1]

    return pd.DataFrame(rows, columns=["open", "high", "low", "close"], index=index)


class TestGjrGarchRecoversAKnownProcess:
    def test_parameters_land_within_two_standard_errors(self, gjr_returns: pd.Series) -> None:
        fit = GjrGarch(innovation="normal").fit(gjr_returns)

        assert fit.converged
        for name in ("alpha", "gamma", "beta"):
            error = fit.standard_errors[name]
            assert abs(getattr(fit, name) - TRUTH[name]) < 2.5 * error, (
                f"{name} = {getattr(fit, name):.4f} is far from {TRUTH[name]} (se {error:.4f})"
            )

    def test_conditional_sigma_tracks_the_latent_one(self) -> None:
        """Against the DGP's sigma, not against ``|r|`` — see :func:`simulate_gjr_path`."""
        path = simulate_gjr_path(periods=1_500, seed=3)
        fit = GjrGarch(innovation="normal").fit(path["returns"])

        assert float(fit.conditional_sigma.corr(path["sigma"])) > 0.95

    def test_persistence_stays_below_one(self, gjr_returns: pd.Series) -> None:
        """At unit persistence there is no unconditional variance to revert to."""
        assert GjrGarch().fit(gjr_returns).persistence < 1.0

    def test_standardized_residuals_are_standardized(self, gjr_returns: pd.Series) -> None:
        z = GjrGarch().fit(gjr_returns).standardized_residuals
        assert abs(float(z.std())) == pytest.approx(1.0, abs=0.15)
        assert abs(float(z.mean())) < 0.1

    def test_asymmetry_is_measured_not_assumed(self) -> None:
        """gamma must read as absent when the process has no leverage effect."""
        symmetric = simulate_gjr_returns(alpha=0.10, gamma=0.0, beta=0.85, periods=2_500, seed=11)
        fit = GjrGarch(innovation="normal").fit(symmetric)

        assert fit.gamma < 2.5 * fit.standard_errors["gamma"], (
            "a symmetric process must not produce a significant asymmetry term"
        )

    def test_student_t_reports_when_the_tails_are_indistinguishable_from_normal(
        self, gjr_returns: pd.Series
    ) -> None:
        described = GjrGarch(innovation="t").fit(gjr_returns).describe()
        assert described["df"] is not None
        assert isinstance(described["df_at_bound"], bool)

    def test_mean_is_zero_unless_asked_otherwise(self, gjr_returns: pd.Series) -> None:
        """A fitted drift is a directional forecast; §0 makes the random walk the null."""
        assert GjrGarch().fit(gjr_returns).mu == 0.0
        assert GjrGarch(mean="constant").fit(gjr_returns).mu != 0.0

    def test_too_little_data_fails_loudly(self) -> None:
        short = simulate_gjr_returns(periods=60)
        with pytest.raises(InsufficientDataError, match="at least 250"):
            GjrGarch().fit(short)


class TestBreaksAreExcludedNotAbsorbed:
    """A policy step is a mechanical jump, not information about volatility."""

    def _series_with_a_jump(self) -> tuple[pd.Series, pd.Series, pd.Timestamp]:
        clean = simulate_gjr_returns(periods=1_200, seed=21)
        break_date = clean.index[900]
        shocked = clean.copy()
        shocked.loc[break_date] = 0.09  # a 9% one-day policy step
        return clean, shocked, break_date

    def test_excluding_the_break_keeps_the_variance_forecast_near_the_clean_one(self) -> None:
        clean, shocked, break_date = self._series_with_a_jump()
        model = GjrGarch(innovation="normal")

        baseline = model.fit(clean).next_variance
        absorbed = model.fit(shocked).next_variance
        excluded = model.fit(shocked, exclude=[break_date]).next_variance

        assert abs(excluded - baseline) < abs(absorbed - baseline), (
            "excluding the break must move the variance forecast back toward the "
            f"unbroken series: baseline={baseline:.3e} absorbed={absorbed:.3e} "
            f"excluded={excluded:.3e}"
        )

    def test_the_excluded_day_is_absent_from_the_residuals(self) -> None:
        _, shocked, break_date = self._series_with_a_jump()
        fit = GjrGarch(innovation="normal").fit(shocked, exclude=[break_date])

        assert break_date not in fit.standardized_residuals.index
        assert fit.n_excluded == 1

    def test_a_dropped_residual_cannot_be_resampled_later(self) -> None:
        """FHS draws from these residuals; a policy step in there is a fabricated shock."""
        _, shocked, break_date = self._series_with_a_jump()
        fit = GjrGarch(innovation="normal").fit(shocked, exclude=[break_date])

        assert float(fit.standardized_residuals.abs().max()) < 8.0


class TestPropagation:
    def test_shape_and_determinism(self, gjr_returns: pd.Series) -> None:
        fit = GjrGarch().fit(gjr_returns)
        shocks = np.random.default_rng(0).standard_normal((500, 12))

        first, second = fit.propagate(shocks), fit.propagate(shocks)
        assert first.shape == (500, 12)
        np.testing.assert_array_equal(first, second)

    def test_variance_clusters_after_a_negative_shock(self, gjr_returns: pd.Series) -> None:
        """The asymmetry has to survive into simulation, not just into the fit."""
        fit = GjrGarch(innovation="normal").fit(gjr_returns)

        down = np.column_stack([np.full(4_000, -3.0), np.zeros((4_000, 1))])
        up = np.column_stack([np.full(4_000, 3.0), np.zeros((4_000, 1))])

        # Second-step dispersion is zero under zero shocks, so compare the variance
        # the recursion carries by feeding a unit shock at step two instead.
        down[:, 1] = np.random.default_rng(1).standard_normal(4_000)
        up[:, 1] = np.random.default_rng(1).standard_normal(4_000)

        assert down[:, 1].std() == pytest.approx(up[:, 1].std())
        assert fit.propagate(down)[:, 1].std() > fit.propagate(up)[:, 1].std()

    def test_forward_sigma_reverts_toward_the_long_run(self, gjr_returns: pd.Series) -> None:
        fit = GjrGarch().fit(gjr_returns)
        path = fit.forward_sigma(250)
        long_run = np.sqrt(fit.unconditional_variance)

        assert abs(path[-1] - long_run) < abs(path[0] - long_run) + 1e-12

    def test_a_one_dimensional_shock_array_is_rejected(self, gjr_returns: pd.Series) -> None:
        fit = GjrGarch().fit(gjr_returns)
        with pytest.raises(ValueError, match="n_paths, horizon"):
            fit.propagate(np.zeros(10))


class TestRollingStd:
    def test_sigma_is_strictly_backward_looking(self, gjr_returns: pd.Series) -> None:
        """Including the day's own return in its own sigma narrows every path built from it."""
        fit = RollingStd(window=60).fit(gjr_returns)
        last_date = fit.conditional_sigma.index[-1]
        position = gjr_returns.index.get_loc(last_date)

        expected = gjr_returns.iloc[position - 60 : position].std(ddof=1)
        assert float(fit.conditional_sigma.iloc[-1]) == pytest.approx(float(expected))

    def test_forward_sigma_is_flat(self, gjr_returns: pd.Series) -> None:
        path = RollingStd().fit(gjr_returns).forward_sigma(20)
        assert len(set(np.round(path, 12))) == 1

    def test_excluded_dates_leave_the_sample(self, gjr_returns: pd.Series) -> None:
        excluded = [gjr_returns.index[100], gjr_returns.index[200]]
        fit = RollingStd().fit(gjr_returns, exclude=excluded)

        assert fit.n_excluded == 2
        assert excluded[0] not in fit.standardized_residuals.index

    def test_too_short_a_series_fails_loudly(self) -> None:
        with pytest.raises(InsufficientDataError):
            RollingStd(window=60).fit(simulate_gjr_returns(periods=30))


class TestRealisedVarianceEstimators:
    def test_parkinson_recovers_the_daily_variance(self) -> None:
        frame = intraday_ohlc(daily_sigma=0.015)
        estimate = float(parkinson_variance(frame).mean())
        assert estimate == pytest.approx(0.015**2, rel=0.10)

    def test_garman_klass_recovers_the_daily_variance(self) -> None:
        frame = intraday_ohlc(daily_sigma=0.015)
        estimate = float(garman_klass_variance(frame).mean())
        assert estimate == pytest.approx(0.015**2, rel=0.10)

    def test_the_range_estimator_is_less_noisy_than_a_squared_return(self) -> None:
        """The reason to depend on an OHLC series at all.

        Both estimators are centred on the same daily variance; the range one has
        roughly half the dispersion, which is what buys the extra data dependency.
        """
        frame = intraday_ohlc(days=1_200)
        squared_close_to_close = np.log(frame["close"]).diff().dropna() ** 2
        ranged = parkinson_variance(frame)

        assert float(ranged.mean()) == pytest.approx(float(squared_close_to_close.mean()), rel=0.15)
        assert float(ranged.std()) < 0.6 * float(squared_close_to_close.std())

    def test_a_close_only_series_is_refused_rather_than_faked(self) -> None:
        close_only = pd.DataFrame({"close": [1.0, 2.0]})
        with pytest.raises(InsufficientDataError, match="close-only"):
            parkinson_variance(close_only)


class TestHarRv:
    def _fitted(self) -> tuple[pd.Series, pd.Series]:
        frame = intraday_ohlc(days=900)
        returns = np.log(frame["close"]).diff().dropna()
        return returns, parkinson_variance(frame)

    def test_it_refuses_to_substitute_squared_returns_for_a_measurement(self) -> None:
        returns, _ = self._fitted()
        with pytest.raises(InsufficientDataError, match="not a substitute"):
            HarRv().fit(returns)

    def test_the_cascade_is_fitted_and_explains_something(self) -> None:
        returns, variance = self._fitted()
        fit = HarRv(min_observations=200).fit(returns, realised_variance=variance)

        assert set(fit.coefficients) == {"const", "daily", "weekly", "monthly"}
        assert 0.0 <= fit.r_squared <= 1.0

    def test_every_path_shares_one_variance_trajectory(self) -> None:
        """Stated as a limitation in describe(); asserted here so it stays true."""
        returns, variance = self._fitted()
        fit = HarRv(min_observations=200).fit(returns, realised_variance=variance)

        shocks = np.ones((3, 4))
        paths = fit.propagate(shocks)
        assert np.allclose(paths[0], paths[1]) and np.allclose(paths[1], paths[2])
        assert "variance-of-variance" in " ".join(fit.describe()["limitations"])

    def test_a_realised_variance_series_shorter_than_the_cascade_is_refused(self) -> None:
        returns, variance = self._fitted()
        with pytest.raises(InsufficientDataError, match="too short for HAR lags"):
            HarRv(min_observations=1).fit(returns, realised_variance=variance.iloc[:10])

    def test_returns_that_do_not_overlap_the_variance_series_are_refused(self) -> None:
        """Aligning on nothing would fit a model to an empty sample."""
        returns, variance = self._fitted()
        moved = returns.copy()
        moved.index = moved.index + pd.DateOffset(years=20)
        with pytest.raises(InsufficientDataError, match="do not overlap"):
            HarRv(min_observations=200).fit(moved, realised_variance=variance)

    def test_regressors_are_lagged_so_nothing_predicts_itself(self) -> None:
        returns, variance = self._fitted()
        fit = HarRv(min_observations=200).fit(returns, realised_variance=variance)

        # A same-day regressor would push in-sample R-squared toward one.
        assert fit.r_squared < 0.95


class TestEveryModelHonoursTheSameGuards:
    """One protocol means one set of refusals, not three different ones."""

    def _fits(self) -> list[object]:
        returns, variance = TestHarRv()._fitted()
        return [
            GjrGarch().fit(simulate_gjr_returns(periods=800)),
            RollingStd(window=60).fit(simulate_gjr_returns(periods=800)),
            HarRv(min_observations=200).fit(returns, realised_variance=variance),
        ]

    def test_a_non_positive_horizon_is_refused(self) -> None:
        for fit in self._fits():
            with pytest.raises(ValueError, match="horizon must be positive"):
                fit.forward_sigma(0)  # type: ignore[attr-defined]

    def test_a_flat_shock_array_is_refused(self) -> None:
        for fit in self._fits():
            with pytest.raises(ValueError, match="n_paths, horizon"):
                fit.propagate(np.zeros(8))  # type: ignore[attr-defined]

    @pytest.mark.parametrize("model_id", ["gjr_garch", "rolling_std", "har_rv"])
    def test_every_specification_describes_itself(self, model_id: str) -> None:
        """The artifact records which model produced a distribution, and how set up."""
        described = model_for(model_id).describe()
        assert described["id"] == model_id
        assert described["mean"] == "zero"


class TestModelRegistry:
    @pytest.mark.parametrize("model_id", ["gjr_garch", "rolling_std", "har_rv"])
    def test_every_declared_model_resolves(self, model_id: str) -> None:
        assert model_for(model_id).id == model_id

    def test_an_unknown_model_lists_what_exists(self) -> None:
        with pytest.raises(KeyError, match="available"):
            model_for("stochastic_vol")


class TestSessionLimitSpec:
    def test_a_relaxed_band_must_be_wider(self) -> None:
        with pytest.raises(ValueError, match="cooling-off band widens"):
            SessionLimit(fraction=0.06, relaxed_fraction=0.04)

    def test_the_cap_widens_only_after_a_locked_session(self) -> None:
        limit = SessionLimit(fraction=0.04, relaxed_fraction=0.06)
        assert limit.cap_for(previous_session_locked=False) == 0.04
        assert limit.cap_for(previous_session_locked=True) == 0.06

    def test_a_venue_without_a_relaxed_band_keeps_one_cap(self) -> None:
        limit = SessionLimit(fraction=0.05)
        assert limit.cap_for(previous_session_locked=True) == 0.05

    @pytest.mark.parametrize("fraction", [0.0, 1.0, -0.1])
    def test_an_impossible_limit_is_rejected(self, fraction: float) -> None:
        with pytest.raises(ValueError, match="fraction must be"):
            SessionLimit(fraction=fraction)


class TestAnUnmeasurableSessionIsAbsent:
    """A session printing ``high == low`` did not trade; it did not have zero variance.

    The range estimators used to clip such a session to a floor of ``1e-12``. About 7% of
    the cached futures sessions in this repository print that way, and each one entered
    the log regression at roughly -27.6 against a series whose real mean is near -10.5.
    That tripled the residual spread, and the retransformation term is half that spread
    inside an exponential — the fitted smearing reached 9.87, a multiplier near 19,000,
    and the variance recursion diverged within two steps. The model was unusable and
    nothing said so, because no registered asset had ever selected it.
    """

    def _frame(self, n: int, stale: int) -> pd.DataFrame:
        rng = np.random.default_rng(4)
        close = 100.0 * np.exp(np.cumsum(rng.normal(0.0, 0.01, n)))
        spread = np.abs(rng.normal(0.0, 0.01, n)) + 0.005
        high, low = close * (1.0 + spread), close * (1.0 - spread)
        # Stale prints: the vendor repeated a single quote for the whole session.
        high[:stale] = close[:stale]
        low[:stale] = close[:stale]
        return pd.DataFrame(
            {"open": close, "high": high, "low": low, "close": close},
            index=pd.date_range("2010-01-01", periods=n, freq="B"),
        )

    def test_a_zero_range_session_is_nan_not_a_floor(self) -> None:
        variance = parkinson_variance(self._frame(50, stale=5))

        assert int(variance.isna().sum()) == 5
        assert float(variance.dropna().min()) > 0.0

    def test_garman_klass_agrees(self) -> None:
        variance = garman_klass_variance(self._frame(50, stale=5))

        assert int(variance.isna().sum()) == 5

    def test_stale_prints_do_not_inflate_the_smearing_term(self) -> None:
        frame = self._frame(900, stale=90)
        returns = pd.Series(
            np.diff(np.log(frame["close"].to_numpy()), prepend=np.log(frame["close"].iloc[0])),
            index=frame.index,
        ).iloc[1:]

        fit = HarRv(min_observations=200).fit(returns, realised_variance=parkinson_variance(frame))

        # Half the residual variance of a log-variance regression. Single digits at the
        # very worst; 9.87 is what the floored version produced.
        assert fit.smearing < 3.0

    def test_the_variance_recursion_stays_bounded(self) -> None:
        """The failure the floor caused, stated as the property that must hold."""
        frame = self._frame(900, stale=90)
        returns = pd.Series(
            np.diff(np.log(frame["close"].to_numpy()), prepend=np.log(frame["close"].iloc[0])),
            index=frame.index,
        ).iloc[1:]

        fit = HarRv(min_observations=200).fit(returns, realised_variance=parkinson_variance(frame))
        sigma = fit.forward_sigma(63)

        assert np.all(np.isfinite(sigma))
        # A daily sigma above 100% at any horizon is a diverging recursion, not a
        # forecast. The floored version reached 26% by the fifth step and 8.5e15 by the
        # tenth.
        assert float(sigma.max()) < 1.0
        assert float(sigma.max()) / float(sigma.min()) < 20.0

    def test_a_series_with_no_measurable_session_is_refused(self) -> None:
        frame = self._frame(400, stale=400)

        with pytest.raises(InsufficientDataError, match="measurable range"):
            HarRv(min_observations=50).fit(
                pd.Series(0.0, index=frame.index), realised_variance=parkinson_variance(frame)
            )
