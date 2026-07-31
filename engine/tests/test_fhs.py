"""Filtered historical simulation and the path ensemble it returns.

The session-limit tests are deterministic by construction: a single oversized shock
fed through :func:`paths_from_returns` has one arithmetically correct answer, and a
carry that silently vanishes or double-counts shows up immediately. Doing that with
random shocks would only prove the cap holds, not that the remainder went anywhere.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from aurex.assets.transforms import Difference, LogReturn, ShiftedLogReturn
from aurex.dist import (
    PathEnsemble,
    SimulationSpec,
    block_indices,
    bootstrap_shocks,
    paths_from_returns,
    simulate,
)
from aurex.vol import RollingStd, SessionLimit
from tests.conftest import simulate_gjr_returns

ANCHOR = 100.0


@pytest.fixture
def flat_fit() -> RollingStd:
    return RollingStd(window=60)


@pytest.fixture
def fitted(flat_fit: RollingStd):  # type: ignore[no-untyped-def]
    return flat_fit.fit(simulate_gjr_returns(periods=1_500, seed=4))


class TestBlockBootstrap:
    def test_shape_and_range(self) -> None:
        picks = block_indices(
            500, n_paths=64, horizon=13, block_length=5, rng=np.random.default_rng(0)
        )
        assert picks.shape == (64, 13)
        assert picks.min() >= 0 and picks.max() < 500

    def test_blocks_are_contiguous_and_wrap(self) -> None:
        """Contiguity is the point: single draws would destroy serial structure."""
        picks = block_indices(
            10, n_paths=200, horizon=6, block_length=3, rng=np.random.default_rng(1)
        )
        within_block = (picks[:, 1:3] - picks[:, 0:2]) % 10
        assert np.all(within_block == 1)

    def test_every_observation_is_reachable(self) -> None:
        """A non-circular bootstrap underweights the most recent residuals."""
        picks = block_indices(
            20, n_paths=4_000, horizon=8, block_length=4, rng=np.random.default_rng(2)
        )
        assert set(np.unique(picks)) == set(range(20))

    def test_resampling_preserves_the_empirical_distribution(self) -> None:
        residuals = np.random.default_rng(3).standard_t(5, size=2_000)
        drawn = bootstrap_shocks(
            residuals,
            n_paths=5_000,
            horizon=10,
            block_length=5,
            rng=np.random.default_rng(4),
        )
        assert drawn.mean() == pytest.approx(residuals.mean(), abs=0.05)
        assert drawn.std() == pytest.approx(residuals.std(), rel=0.05)
        assert np.quantile(drawn, 0.01) == pytest.approx(np.quantile(residuals, 0.01), rel=0.1)

    def test_a_degenerate_sample_is_refused(self) -> None:
        with pytest.raises(ValueError, match="at least 2 residuals"):
            block_indices(1, n_paths=2, horizon=2, block_length=1, rng=np.random.default_rng(0))


class TestSimulation:
    def test_paths_are_retained_not_collapsed(self, fitted) -> None:  # type: ignore[no-untyped-def]
        """§18's correction, asserted: the intermediate sessions must still be there."""
        spec = SimulationSpec(horizon_days=12, n_paths=500, block_length=5, seed=1)
        ensemble = simulate(fitted, transform=LogReturn(), anchor=ANCHOR, spec=spec)

        assert ensemble.prices.shape == (500, 12)
        assert not np.allclose(ensemble.prices[:, 0], ensemble.terminal())

    def test_the_same_seed_reproduces_the_ensemble(self, fitted) -> None:  # type: ignore[no-untyped-def]
        """A distribution nobody can reproduce cannot be scored later."""
        spec = SimulationSpec(horizon_days=5, n_paths=200, seed=99)
        first = simulate(fitted, transform=LogReturn(), anchor=ANCHOR, spec=spec)
        second = simulate(fitted, transform=LogReturn(), anchor=ANCHOR, spec=spec)

        np.testing.assert_array_equal(first.prices, second.prices)

    def test_a_different_seed_gives_a_different_ensemble(self, fitted) -> None:  # type: ignore[no-untyped-def]
        base = SimulationSpec(horizon_days=5, n_paths=200, seed=1)
        other = SimulationSpec(horizon_days=5, n_paths=200, seed=2)

        first = simulate(fitted, transform=LogReturn(), anchor=ANCHOR, spec=base)
        second = simulate(fitted, transform=LogReturn(), anchor=ANCHOR, spec=other)
        assert not np.allclose(first.prices, second.prices)

    def test_terminal_dispersion_scales_with_the_square_root_of_horizon(self, fitted) -> None:  # type: ignore[no-untyped-def]
        """Flat sigma and zero drift: the ensemble must reproduce root-time scaling."""
        short = simulate(
            fitted,
            transform=LogReturn(),
            anchor=ANCHOR,
            spec=SimulationSpec(horizon_days=4, n_paths=20_000, seed=5),
        )
        long = simulate(
            fitted,
            transform=LogReturn(),
            anchor=ANCHOR,
            spec=SimulationSpec(horizon_days=16, n_paths=20_000, seed=5),
        )

        short_sd = float(np.std(np.log(short.terminal() / ANCHOR)))
        long_sd = float(np.std(np.log(long.terminal() / ANCHOR)))
        assert long_sd / short_sd == pytest.approx(2.0, rel=0.1)

    def test_the_seed_and_the_model_travel_with_the_ensemble(self, fitted) -> None:  # type: ignore[no-untyped-def]
        spec = SimulationSpec(horizon_days=3, n_paths=50, seed=1234)
        ensemble = simulate(fitted, transform=LogReturn(), anchor=ANCHOR, spec=spec)

        assert ensemble.diagnostics["simulation"]["seed"] == 1234
        assert ensemble.diagnostics["vol_model"]["model"] == "rolling_std"
        assert ensemble.diagnostics["transform"]["id"] == "log"

    def test_supplied_shocks_must_match_the_spec(self, fitted) -> None:  # type: ignore[no-untyped-def]
        spec = SimulationSpec(horizon_days=5, n_paths=10, seed=1)
        with pytest.raises(ValueError, match="shocks must be"):
            simulate(
                fitted,
                transform=LogReturn(),
                anchor=ANCHOR,
                spec=spec,
                shocks=np.zeros((10, 4)),
            )

    @pytest.mark.parametrize("field", ["horizon_days", "n_paths", "block_length"])
    def test_a_meaningless_spec_is_refused(self, field: str) -> None:
        kwargs = {"horizon_days": 5, "n_paths": 10, field: 0}
        with pytest.raises(ValueError, match="must be positive"):
            SimulationSpec(**kwargs)  # type: ignore[arg-type]


class TestSessionLimits:
    """Truncate the session, carry the remainder — §18."""

    LIMIT = SessionLimit(fraction=0.04, relaxed_fraction=0.06)

    def _one_big_shock(self, sessions: int = 6) -> np.ndarray:
        returns = np.zeros((1, sessions))
        returns[0, 0] = np.log(1.20)  # a 20% move arriving in one session
        return returns

    def test_no_session_moves_further_than_the_cap(self) -> None:
        prices, _ = paths_from_returns(
            LogReturn(), ANCHOR, self._one_big_shock(), session_limit=self.LIMIT
        )
        path = np.concatenate(([ANCHOR], prices[0]))
        moves = np.abs(np.diff(path) / path[:-1])
        assert moves.max() <= 0.06 + 1e-12

    def test_the_first_session_takes_the_ordinary_cap(self) -> None:
        prices, _ = paths_from_returns(
            LogReturn(), ANCHOR, self._one_big_shock(), session_limit=self.LIMIT
        )
        assert prices[0, 0] == pytest.approx(ANCHOR * 1.04)

    def test_the_session_after_a_locked_one_takes_the_relaxed_cap(self) -> None:
        prices, _ = paths_from_returns(
            LogReturn(), ANCHOR, self._one_big_shock(), session_limit=self.LIMIT
        )
        assert prices[0, 1] == pytest.approx(ANCHOR * 1.04 * 1.06)

    def test_the_remainder_arrives_rather_than_disappearing(self) -> None:
        """A limit delays a move; it does not delete one."""
        prices, _ = paths_from_returns(
            LogReturn(), ANCHOR, self._one_big_shock(sessions=8), session_limit=self.LIMIT
        )
        assert prices[0, -1] == pytest.approx(ANCHOR * 1.20)

    def test_carry_can_be_switched_off_and_then_the_move_is_lost(self) -> None:
        limit = SessionLimit(fraction=0.04, carry_residual=False)
        prices, _ = paths_from_returns(
            LogReturn(), ANCHOR, self._one_big_shock(), session_limit=limit
        )
        assert prices[0, 0] == pytest.approx(ANCHOR * 1.04)
        assert prices[0, -1] == pytest.approx(ANCHOR * 1.04)

    def test_truncation_is_counted_and_reported(self) -> None:
        _, diagnostics = paths_from_returns(
            LogReturn(), ANCHOR, self._one_big_shock(), session_limit=self.LIMIT
        )
        block = diagnostics["session_limit"]
        assert block["sessions_truncated"] >= 1
        assert 0.0 < block["share_of_sessions_truncated"] <= 1.0

    def test_without_a_limit_nothing_is_truncated(self) -> None:
        prices, diagnostics = paths_from_returns(LogReturn(), ANCHOR, self._one_big_shock())
        assert prices[0, 0] == pytest.approx(ANCHOR * 1.20)
        assert diagnostics["session_limit"] is None

    def test_a_downward_move_is_capped_the_same_way(self) -> None:
        returns = np.zeros((1, 4))
        returns[0, 0] = np.log(0.80)
        prices, _ = paths_from_returns(LogReturn(), ANCHOR, returns, session_limit=self.LIMIT)
        assert prices[0, 0] == pytest.approx(ANCHOR * 0.96)
        assert prices[0, -1] == pytest.approx(ANCHOR * 0.80)


class TestTransformAgnosticism:
    """The simulator must not assume a particular return definition."""

    @pytest.mark.parametrize(
        "transform",
        [LogReturn(), ShiftedLogReturn(shift=50.0), Difference()],
        ids=lambda t: t.id,
    )
    def test_paths_start_from_the_anchor_and_move(self, transform) -> None:  # type: ignore[no-untyped-def]
        returns = np.full((3, 4), 0.01)
        prices, _ = paths_from_returns(transform, ANCHOR, returns)

        first_step = transform.advance(np.array([ANCHOR]), np.array([0.01]))[0]
        assert prices[0, 0] == pytest.approx(first_step)
        assert prices.shape == (3, 4)

    def test_a_limit_applies_in_price_space_whatever_the_transform(self) -> None:
        """The cap is a price move, so a shifted transform must not rescale it."""
        returns = np.zeros((1, 3))
        returns[0, 0] = 0.5
        prices, _ = paths_from_returns(
            ShiftedLogReturn(shift=50.0),
            ANCHOR,
            returns,
            session_limit=SessionLimit(fraction=0.04),
        )
        assert prices[0, 0] == pytest.approx(ANCHOR * 1.04)


class TestPathEnsemble:
    def _ensemble(self) -> PathEnsemble:
        # Column 0 is the price after the *first* session; the anchor is not a column.
        prices = np.array([[95.0, 92.0, 90.0], [105.0, 108.0, 110.0], [89.0, 95.0, 120.0]])
        return PathEnsemble(prices=prices, anchor=100.0)

    def test_quantiles_are_ordered(self) -> None:
        quantiles = self._ensemble().quantiles()
        values = [quantiles[key] for key in sorted(quantiles)]
        assert values == sorted(values)

    def test_quantiles_can_be_taken_at_an_intermediate_session(self) -> None:
        ensemble = self._ensemble()
        assert ensemble.quantiles(step=1)["q50"] == pytest.approx(95.0)

    def test_touching_and_ending_are_different_questions(self) -> None:
        ensemble = self._ensemble()
        assert ensemble.probability_below(90.0, terminal_only=True) == pytest.approx(1 / 3)
        assert ensemble.probability_below(90.0) == pytest.approx(2 / 3)

    def test_touching_upward_is_asked_the_same_way(self) -> None:
        ensemble = self._ensemble()
        assert ensemble.probability_above(110.0, terminal_only=True) == pytest.approx(2 / 3)
        assert ensemble.probability_above(110.0) == pytest.approx(2 / 3)
        assert ensemble.running_maximum().tolist() == [95.0, 110.0, 120.0]

    def test_two_ensembles_compose_into_a_second_currency(self) -> None:
        first, second = self._ensemble(), self._ensemble()
        combined = first.rescaled(second, factor=1.5)

        assert combined.anchor == pytest.approx(100.0 * 100.0 * 1.5)
        assert combined.prices[0, 0] == pytest.approx(95.0 * 95.0 * 1.5)
        assert combined.diagnostics["constant_factor"] == 1.5
        assert set(combined.diagnostics) >= {"primary", "secondary"}

    def test_a_constant_scales_every_path_and_the_anchor(self) -> None:
        scaled = self._ensemble().scaled(3.0)
        assert scaled.anchor == pytest.approx(300.0)
        assert scaled.prices[0, 0] == pytest.approx(285.0)

    def test_mismatched_ensembles_do_not_compose(self) -> None:
        other = PathEnsemble(prices=np.ones((2, 2)), anchor=1.0)
        with pytest.raises(ValueError, match="same shape"):
            self._ensemble().rescaled(other)

    def test_an_empty_ensemble_is_not_a_distribution(self) -> None:
        with pytest.raises(ValueError, match="not a distribution"):
            PathEnsemble(prices=np.empty((0, 3)), anchor=1.0)

    def test_a_flat_array_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="n_paths, horizon"):
            PathEnsemble(prices=np.ones(5), anchor=1.0)

    def test_describe_carries_no_point_forecast(self) -> None:
        described = self._ensemble().describe()
        assert set(described["terminal_quantiles"]) == {"q05", "q25", "q50", "q75", "q95"}
        assert "mean" not in described
        assert isinstance(pd.Series(described["terminal_quantiles"]).to_dict(), dict)
