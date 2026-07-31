"""First-passage statistics — §18's requirement, and the arithmetic behind it.

Two kinds of test here. The deterministic ones use a hand-built ensemble where every
answer can be worked out by eye, because a barrier statistic that is wrong by one
session is invisible in a Monte Carlo check. The statistical ones assert the property
that motivated the whole section: touching a level before the horizon is materially
likelier than ending beyond it, and a terminal distribution cannot see the difference.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from aurex.assets.transforms import LogReturn
from aurex.dist import (
    PathEnsemble,
    SimulationSpec,
    first_passage,
    headline_statistic,
    margin_call_barrier,
    simulate,
)
from aurex.vol import RollingStd

ANCHOR = 100.0

#: §18's worked example needs a market that can actually reach the barrier: ~2% a
#: session is roughly 32% annualised, the region where a leveraged position lives.
#: Over ten sessions that is a 6.3% standard deviation, so a 10% adverse move is
#: about 1.6 standard deviations — the case the section works through.
DAILY_SIGMA = 0.02


def flat_volatility_returns(*, periods: int = 1_500, seed: int = 9) -> pd.Series:
    """Returns with a known, constant volatility, so barrier probabilities are readable.

    A fitted GJR process would work too, but its terminal variance depends on where
    the simulated path happened to end, which makes the probability under test a
    property of the fixture's random draw rather than of the code.
    """
    rng = np.random.default_rng(seed)
    index = pd.bdate_range("2018-01-01", periods=periods, name="date")
    return pd.Series(DAILY_SIGMA * rng.standard_normal(periods), index=index)


def hand_built() -> PathEnsemble:
    """Four paths over three sessions, with the answers worked out in the tests.

    ============  ==========================  ==========================
    path          prices                      barrier 90, downward
    ============  ==========================  ==========================
    0             95, 92, 90                  touches at session 3
    1             105, 108, 110               never touches
    2             89, 95, 120                 touches at session 1
    3             101, 97, 88                 touches at session 3
    ============  ==========================  ==========================
    """
    prices = np.array(
        [
            [95.0, 92.0, 90.0],
            [105.0, 108.0, 110.0],
            [89.0, 95.0, 120.0],
            [101.0, 97.0, 88.0],
        ]
    )
    return PathEnsemble(prices=prices, anchor=ANCHOR)


class TestDeterministicBarrierArithmetic:
    def test_touch_probability_counts_every_path_that_reached_it(self) -> None:
        passage = first_passage(hand_built(), barrier=90.0)
        assert passage.touch_probability == pytest.approx(0.75)
        assert passage.n_touched == 3

    def test_terminal_probability_counts_only_the_horizon(self) -> None:
        """Path 2 touched and recovered; a terminal-only view never sees it."""
        passage = first_passage(hand_built(), barrier=90.0)
        assert passage.terminal_probability == pytest.approx(0.5)

    def test_the_ratio_is_the_cost_of_reporting_terminals_alone(self) -> None:
        passage = first_passage(hand_built(), barrier=90.0)
        assert passage.path_dependence_ratio == pytest.approx(1.5)

    def test_time_to_the_barrier_is_conditional_on_reaching_it(self) -> None:
        """Averaging over paths that never got there would not be a duration."""
        passage = first_passage(hand_built(), barrier=90.0)
        assert passage.mean_sessions_to_touch == pytest.approx((3 + 1 + 3) / 3)
        assert passage.sessions_to_touch["q50"] == pytest.approx(3.0)

    def test_survivor_quantiles_exclude_everything_that_touched(self) -> None:
        passage = first_passage(hand_built(), barrier=90.0)
        assert passage.survivor_terminal_quantiles["q50"] == pytest.approx(110.0)

    def test_an_upward_barrier_is_the_mirror_image(self) -> None:
        passage = first_passage(hand_built(), barrier=110.0, direction="up")
        assert passage.touch_probability == pytest.approx(0.5)
        assert passage.mean_sessions_to_touch == pytest.approx(3.0)

    def test_an_unreachable_barrier_reports_nothing_rather_than_zero(self) -> None:
        passage = first_passage(hand_built(), barrier=1.0)
        assert passage.touch_probability == 0.0
        assert passage.mean_sessions_to_touch is None
        assert passage.sessions_to_touch == {}
        assert passage.path_dependence_ratio is None

    def test_a_barrier_above_every_path_is_certain(self) -> None:
        passage = first_passage(hand_built(), barrier=200.0)
        assert passage.touch_probability == 1.0
        assert passage.survivor_terminal_quantiles == {}

    def test_an_unknown_direction_is_refused(self) -> None:
        with pytest.raises(ValueError, match=r"down.*up"):
            first_passage(hand_built(), barrier=90.0, direction="sideways")  # type: ignore[arg-type]

    def test_monitoring_convention_is_stated_in_the_output(self) -> None:
        """The touch probability is a floor, and the artifact has to say so."""
        described = first_passage(hand_built(), barrier=90.0).describe()
        assert described["monitoring"] == "session_close"
        assert "floor" in described["note"]


class TestPathDependenceIsReal:
    def _ensemble(self, horizon: int = 10) -> PathEnsemble:
        fit = RollingStd(window=60).fit(flat_volatility_returns())
        spec = SimulationSpec(horizon_days=horizon, n_paths=40_000, block_length=5, seed=17)
        return simulate(fit, transform=LogReturn(), anchor=ANCHOR, spec=spec)

    def test_touching_is_always_at_least_as_likely_as_ending_beyond(self) -> None:
        passage = first_passage(self._ensemble(), barrier=ANCHOR * 0.9)
        assert passage.touch_probability >= passage.terminal_probability

    def test_the_gap_is_large_enough_to_change_a_decision(self) -> None:
        """§18's worked example: the touch probability is the one that matters."""
        passage = first_passage(self._ensemble(), barrier=ANCHOR * 0.9)
        assert passage.path_dependence_ratio is not None
        assert passage.path_dependence_ratio > 1.2

    def test_it_stays_below_the_continuous_monitoring_factor(self) -> None:
        """Sessions are monitored at close, so the factor of two is an upper bound.

        This is a modelling choice with a known direction, not a discrepancy: a
        barrier breached and recovered inside one session is not counted here.
        """
        passage = first_passage(self._ensemble(), barrier=ANCHOR * 0.9)
        assert passage.path_dependence_ratio is not None
        assert passage.path_dependence_ratio < 2.0

    def test_survivors_are_a_different_distribution_from_everyone(self) -> None:
        ensemble = self._ensemble()
        passage = first_passage(ensemble, barrier=ANCHOR * 0.95)

        assert passage.survivor_terminal_quantiles["q05"] > ensemble.quantiles()["q05"]


class TestMarginCallBarrier:
    def test_ten_times_leverage_liquidates_on_a_ten_percent_move(self) -> None:
        assert margin_call_barrier(100.0, leverage=10.0) == pytest.approx(90.0)

    def test_a_maintenance_requirement_liquidates_earlier(self) -> None:
        assert margin_call_barrier(100.0, leverage=10.0, maintenance_fraction=0.5) == (
            pytest.approx(95.0)
        )

    def test_a_short_position_is_liquidated_upward(self) -> None:
        assert margin_call_barrier(100.0, leverage=5.0, side="short") == pytest.approx(120.0)

    def test_an_unleveraged_long_has_no_barrier(self) -> None:
        """Nothing liquidates it: the price would have to go below zero."""
        assert margin_call_barrier(100.0, leverage=1.0) == -np.inf

    @pytest.mark.parametrize("leverage", [0.0, -2.0])
    def test_impossible_leverage_is_refused(self, leverage: float) -> None:
        with pytest.raises(ValueError, match="leverage must be positive"):
            margin_call_barrier(100.0, leverage=leverage)

    def test_a_maintenance_fraction_of_one_is_refused(self) -> None:
        with pytest.raises(ValueError, match="maintenance_fraction"):
            margin_call_barrier(100.0, leverage=10.0, maintenance_fraction=1.0)

    def test_leverage_raises_the_liquidation_probability(self) -> None:
        fit = RollingStd(window=60).fit(flat_volatility_returns())
        ensemble = simulate(
            fit,
            transform=LogReturn(),
            anchor=ANCHOR,
            spec=SimulationSpec(horizon_days=10, n_paths=20_000, seed=21),
        )

        modest = first_passage(
            ensemble, barrier=margin_call_barrier(ANCHOR, leverage=2.0)
        ).touch_probability
        aggressive = first_passage(
            ensemble, barrier=margin_call_barrier(ANCHOR, leverage=10.0)
        ).touch_probability

        assert aggressive > modest


class TestHeadlineRule:
    def test_liquidation_leads_when_it_is_likelier_than_profit(self) -> None:
        """§18 extends §7: a position that is closed out never reaches the distribution."""
        assert (
            headline_statistic(profit_probability=0.45, liquidation_probability=0.5)
            == "liquidation_probability"
        )

    def test_a_losing_position_leads_with_the_loss(self) -> None:
        assert (
            headline_statistic(profit_probability=0.4, liquidation_probability=0.1)
            == "expected_loss"
        )

    def test_otherwise_the_distribution_leads(self) -> None:
        assert (
            headline_statistic(profit_probability=0.6, liquidation_probability=0.2)
            == "distribution"
        )

    def test_an_unleveraged_position_needs_no_liquidation_input(self) -> None:
        assert headline_statistic(profit_probability=0.55) == "distribution"
        assert headline_statistic(profit_probability=0.3) == "expected_loss"
