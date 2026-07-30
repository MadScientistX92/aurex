"""Parity, the domestic premium, and the duty-passthrough distinction.

The central design point here: parity is a *mechanical* function of the duty
schedule, while the premium is a *measurement* of market behaviour. Wiring tests
belong on parity. Asserting a level shift in the premium at a duty change would fail
on correct code — under complete passthrough the premium does not move — and would
quietly encode a market view into the test suite.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from aurex.config import GRAMS_PER_TROY_OUNCE
from aurex.data.parity import (
    GST_REGIME_START,
    compute_parity,
    local_premium_bps,
    passthrough_diagnostic,
)

BREAK = pd.Timestamp("2026-05-13")


def flat_inputs(
    start: str = "2026-04-15",
    end: str = "2026-06-15",
    xau: float = 4000.0,
    usdinr: float = 96.5,
) -> tuple[pd.Series, pd.Series]:
    """Constant inputs, so any movement in parity comes from the schedule alone."""
    index = pd.bdate_range(start, end, name="date")
    return (
        pd.Series(xau, index=index, name="xauusd"),
        pd.Series(usdinr, index=index, name="usdinr"),
    )


class TestParityWiring:
    """Deterministic tests that the schedule actually reaches the arithmetic."""

    def test_parity_steps_at_duty_change(self) -> None:
        """The wiring test. Inputs are constant, so parity may only move if the
        duty schedule moved it: 6% -> 15% is a +8.49% step."""
        parity = compute_parity(*flat_inputs())

        before = parity.loc[parity.index < BREAK, "parity_ex_gst"]
        after = parity.loc[parity.index >= BREAK, "parity_ex_gst"]
        assert len(before) and len(after)

        # Constant inputs => constant parity within each regime.
        assert before.nunique() == 1
        assert after.nunique() == 1

        observed_step = after.iloc[0] / before.iloc[-1] - 1.0
        expected_step = 1.15 / 1.06 - 1.0
        assert observed_step == pytest.approx(expected_step, rel=1e-9)

    def test_duty_rate_column_reflects_the_schedule(self) -> None:
        parity = compute_parity(*flat_inputs())
        assert parity.loc[parity.index < BREAK, "duty_total"].unique().tolist() == [0.06]
        assert parity.loc[parity.index >= BREAK, "duty_total"].unique().tolist() == [0.15]

    def test_parity_matches_the_closed_form(self) -> None:
        xau, usdinr = flat_inputs(start="2026-06-01", end="2026-06-10")
        parity = compute_parity(xau, usdinr)
        expected = 4000.0 / GRAMS_PER_TROY_OUNCE * 96.5 * 10 * 1.15
        assert parity["parity_ex_gst"].iloc[0] == pytest.approx(expected)

    def test_gst_is_applied_only_to_the_inclusive_column(self) -> None:
        parity = compute_parity(*flat_inputs(start="2026-06-01", end="2026-06-10"))
        ratio = parity["parity_incl_gst"] / parity["parity_ex_gst"]
        assert ratio.iloc[0] == pytest.approx(1.03)


class TestPassthroughIsMeasuredNotAsserted:
    def test_premium_is_flat_under_complete_passthrough(self) -> None:
        """This is why the wiring test targets parity, not the premium.

        With full passthrough the retail price tracks parity exactly, so the premium
        is unchanged across the duty break. A test asserting a premium shift here
        would fail on perfectly correct code.
        """
        parity = compute_parity(*flat_inputs())
        observed = parity["parity_ex_gst"] * 1.004  # a steady 40bps premium throughout

        premium = local_premium_bps(observed, parity["parity_ex_gst"])
        before = premium.loc[premium.index < BREAK]
        after = premium.loc[premium.index >= BREAK]

        assert before.mean() == pytest.approx(40.0, abs=0.5)
        assert after.mean() == pytest.approx(40.0, abs=0.5)
        assert after.mean() - before.mean() == pytest.approx(0.0, abs=1e-6)

    def test_diagnostic_reports_incomplete_passthrough_without_judging_it(self) -> None:
        """Incomplete passthrough is a real observation, not a failure."""
        parity = compute_parity(*flat_inputs())
        # Retail lags the duty hike: the premium compresses after the break.
        observed = parity["parity_ex_gst"].copy()
        observed.loc[observed.index < BREAK] *= 1.004
        observed.loc[observed.index >= BREAK] *= 0.98

        premium = local_premium_bps(observed, parity["parity_ex_gst"])
        diagnostic = passthrough_diagnostic(premium, BREAK.date())

        assert diagnostic["mean_bps_before"] == pytest.approx(40.0, abs=1.0)
        assert diagnostic["mean_bps_after"] == pytest.approx(-200.0, abs=1.0)
        assert diagnostic["shift_bps"] is not None
        assert diagnostic["n_before"] > 0 and diagnostic["n_after"] > 0

    def test_diagnostic_handles_a_missing_side(self) -> None:
        parity = compute_parity(*flat_inputs(start="2026-06-01", end="2026-06-15"))
        premium = local_premium_bps(parity["parity_ex_gst"] * 1.001, parity["parity_ex_gst"])
        diagnostic = passthrough_diagnostic(premium, BREAK.date())
        assert diagnostic["n_before"] == 0
        assert diagnostic["mean_bps_before"] is None
        assert diagnostic["shift_bps"] is None


class TestDutyLevelCorroboration:
    def test_duty_level_is_corroborated_by_observed_ibja_print(self) -> None:
        """Independent, observational check on the 15% duty.

        The CBIC primary document could not be retrieved, so duty.yaml marks the
        2026-05-13 entry `secondary`. This test corroborates the *level* against a
        real IBJA print rather than against more secondary reporting.

        IBJA 999 PM on 2026-07-29 was 142,224 Rs/10g, exclusive of GST. Inputs are
        fixed so the test is deterministic; the claim is about order of magnitude,
        not a precise premium.
        """
        stamp = pd.Timestamp("2026-07-29")
        xau = pd.Series(4000.85, index=[stamp])
        usdinr = pd.Series(96.56, index=[stamp])
        ibja_999_pm = 142_224.0

        parity = compute_parity(xau, usdinr)
        premium = local_premium_bps(pd.Series(ibja_999_pm, index=[stamp]), parity["parity_ex_gst"])
        actual_bps = float(premium.iloc[0])

        # A plausible domestic premium is tens of bps, either sign.
        assert abs(actual_bps) < 150, f"premium {actual_bps:.0f}bps implies a wrong duty level"

        # The counterfactual: had the duty still been 6%, the same print would imply
        # a domestic premium of ~+800bps, which is not a plausible standing level.
        base = 4000.85 / GRAMS_PER_TROY_OUNCE * 96.56 * 10
        counterfactual_bps = (ibja_999_pm / (base * 1.06) - 1.0) * 10_000
        assert counterfactual_bps > 700
        assert abs(actual_bps) < abs(counterfactual_bps) / 10


class TestPremiumSemantics:
    def test_premium_is_nan_where_unobserved_never_zero(self) -> None:
        """Filling an unobserved premium with parity would make the residual
        identically zero and fabricate the signal it is meant to measure."""
        parity = compute_parity(*flat_inputs(start="2026-06-01", end="2026-06-15"))
        partial = parity["parity_ex_gst"].iloc[:3] * 1.002

        premium = local_premium_bps(partial, parity["parity_ex_gst"])
        assert premium.iloc[:3].notna().all()
        assert premium.iloc[3:].isna().all()
        assert not (premium.fillna(0.0) == 0.0).all()

    def test_measuring_against_gst_inclusive_parity_is_wrong(self) -> None:
        """IBJA quotes ex-GST. Comparing to the inclusive column invents ~-291bps."""
        parity = compute_parity(*flat_inputs(start="2026-06-01", end="2026-06-10"))
        observed = parity["parity_ex_gst"]  # a true zero-premium observation

        correct = local_premium_bps(observed, parity["parity_ex_gst"])
        wrong = local_premium_bps(observed, parity["parity_incl_gst"])

        assert correct.mean() == pytest.approx(0.0, abs=1e-6)
        assert wrong.mean() == pytest.approx((1 / 1.03 - 1) * 10_000, abs=1.0)

    def test_zero_parity_does_not_divide_by_zero(self) -> None:
        index = pd.bdate_range("2026-06-01", periods=3, name="date")
        parity = pd.Series([100.0, 0.0, 100.0], index=index)
        premium = local_premium_bps(pd.Series(101.0, index=index), parity)
        assert np.isnan(premium.iloc[1])
        assert premium.iloc[0] == pytest.approx(100.0)


class TestConfidenceTagging:
    def test_pre_gst_parity_is_low_confidence(self) -> None:
        """No single national rate existed before 2017-07-01."""
        parity = compute_parity(*flat_inputs(start="2017-06-20", end="2017-07-10"))
        gst_start = pd.Timestamp(GST_REGIME_START)
        pre = parity.loc[parity.index < gst_start, "confidence"]
        post = parity.loc[parity.index >= gst_start, "confidence"]
        assert (pre == "low").all()
        assert (post == "high").all()

    def test_pre_ad_valorem_dates_are_dropped(self) -> None:
        """Duty was Rs 300/10g specific before 2012-01-17; no percentage exists."""
        parity = compute_parity(*flat_inputs(start="2011-12-01", end="2012-02-15"))
        assert (parity.index >= pd.Timestamp("2012-01-17")).all()
        assert parity["duty_total"].notna().all()

    def test_empty_input_returns_empty_frame_with_columns(self) -> None:
        empty = pd.Series(dtype=float, index=pd.DatetimeIndex([], name="date"))
        parity = compute_parity(empty, empty)
        assert parity.empty
        assert "parity_ex_gst" in parity.columns
