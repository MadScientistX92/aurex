"""Friction profiles: two structurally different shapes behind one interface."""

from __future__ import annotations

import pytest

from aurex.assets.friction import (
    US_COLLECTIBLES_NOTE,
    FrictionProfile,
    PhysicalFriction,
    RollFriction,
)

RETAIL = PhysicalFriction(
    id="inr_retail",
    label="India retail",
    dealer_premium=0.03,
    consumption_tax=0.03,
    buyback_discount=0.03,
)
US_COIN = PhysicalFriction(
    id="usd_coin",
    label="US coin",
    dealer_premium=0.03,
    consumption_tax=0.0,
    buyback_discount=0.02,
)
ROLLED = RollFriction(
    id="rolled",
    label="Front-month tracker",
    annual_expense_ratio=0.0075,
    annual_roll_drag=0.10,
    bid_ask_spread=0.001,
    pinned_days=30,
)


class TestPhysicalFriction:
    def test_indian_retail_hurdle_matches_the_spec_table(self) -> None:
        """§15 quotes ~9.4% for the INR path at 3/3/3."""
        assert RETAIL.quote(30).breakeven_pct == pytest.approx(9.37, abs=0.01)

    def test_us_coin_hurdle_falls_in_the_spec_band(self) -> None:
        """§15 quotes 4-6% for the USD path — the cleaner case."""
        assert 4.0 <= US_COIN.quote(30).breakeven_pct <= 6.0

    def test_hurdle_is_horizon_independent(self) -> None:
        """Physical friction is paid at the door, not over time."""
        hurdles = {RETAIL.quote(h).breakeven_multiple for h in (1, 30, 365, 3650)}
        assert len(hurdles) == 1

    def test_components_are_itemised_for_the_ui(self) -> None:
        components = RETAIL.quote(30).components
        assert components == {
            "dealer_premium": 0.03,
            "consumption_tax": 0.03,
            "buyback_discount": 0.03,
        }

    def test_frictionless_profile_has_no_hurdle(self) -> None:
        sgb = PhysicalFriction(
            id="sgb", label="SGB", dealer_premium=0, consumption_tax=0, buyback_discount=0
        )
        assert sgb.quote(30).breakeven_pct == pytest.approx(0.0)

    @pytest.mark.parametrize("buyback", [1.0, 1.5, -0.1])
    def test_impossible_buyback_is_rejected(self, buyback: float) -> None:
        with pytest.raises(ValueError, match="buyback_discount"):
            PhysicalFriction(
                id="x", label="x", dealer_premium=0.0, consumption_tax=0.0, buyback_discount=buyback
            )

    def test_negative_premium_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="dealer_premium"):
            PhysicalFriction(
                id="x", label="x", dealer_premium=-0.01, consumption_tax=0.0, buyback_discount=0.02
            )


class TestRollFriction:
    def test_hurdle_grows_with_horizon(self) -> None:
        """The structural difference from gold: this drag compounds."""
        short = ROLLED.quote(30).breakeven_multiple
        long = ROLLED.quote(365).breakeven_multiple
        assert long > short

    def test_contango_drag_scales_roughly_linearly_in_time(self) -> None:
        one_year = ROLLED.quote(365).components["roll_drag"]
        half_year = ROLLED.quote(182).components["roll_drag"]
        assert one_year == pytest.approx(2 * half_year, rel=0.02)

    def test_backwardation_is_a_tailwind_not_a_cost(self) -> None:
        """Negative roll drag must reduce the hurdle, not be clamped at zero."""
        backwardated = RollFriction(
            id="b", label="b", annual_expense_ratio=0.0, annual_roll_drag=-0.08
        )
        assert backwardated.quote(365).breakeven_pct < 0.0

    def test_quote_discloses_when_roll_drag_stops_being_observed(self) -> None:
        """Only the imminent roll is pinned by the current curve; later rolls are
        an assumption, and saying so is the difference between a cost and a guess."""
        near = " ".join(ROLLED.quote(10).notes)
        far = " ".join(ROLLED.quote(300).notes)
        assert "pinned by the current curve over this horizon" in near
        assert "it is an assumption" in far

    def test_negative_horizon_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="horizon_days"):
            ROLLED.quote(-1)


class TestInterfaceIsGeneralEnough:
    """§16: neither shape may be forced into the other's mould."""

    @pytest.mark.parametrize("profile", [RETAIL, ROLLED])
    def test_both_satisfy_the_protocol(self, profile: FrictionProfile) -> None:
        assert isinstance(profile, FrictionProfile)

    @pytest.mark.parametrize("profile", [RETAIL, ROLLED])
    def test_both_quote_at_any_horizon(self, profile: FrictionProfile) -> None:
        for horizon in (1, 30, 365):
            quote = profile.quote(horizon)
            assert quote.cost_multiplier > 0
            assert quote.proceeds_multiplier > 0

    @pytest.mark.parametrize("profile", [RETAIL, ROLLED])
    def test_describe_declares_horizon_dependence(self, profile: FrictionProfile) -> None:
        """Downstream code reads this rather than checking the concrete type."""
        assert isinstance(profile.describe()["horizon_dependent"], bool)

    def test_the_two_shapes_disagree_about_horizon(self) -> None:
        assert RETAIL.describe()["horizon_dependent"] is False
        assert ROLLED.describe()["horizon_dependent"] is True


class TestCollectiblesNote:
    def test_note_is_informational_and_links_the_source(self) -> None:
        assert "irs.gov" in US_COLLECTIBLES_NOTE
        assert "does not compute tax" in US_COLLECTIBLES_NOTE

    def test_note_states_no_rate_and_computes_nothing(self) -> None:
        """§15: flag the collectibles issue, do not compute a liability."""
        assert "%" not in US_COLLECTIBLES_NOTE
        assert not any(char.isdigit() for char in US_COLLECTIBLES_NOTE.replace("409", ""))
