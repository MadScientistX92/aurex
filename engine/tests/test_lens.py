"""Currency lenses: the parity arithmetic, and the wiring tests that guard it.

Parity is a *mechanical* function of the duty schedule; the domestic premium is a
*measurement* of market behaviour. Wiring tests belong on parity. Asserting a level
shift in the premium at a duty change would fail on correct code, because under
complete passthrough the premium does not move.
"""

from __future__ import annotations

import pandas as pd
import pytest

from aurex.assets import GOLD, lens_by_code
from aurex.assets.lens import LENS_COLUMNS, LensContext, NativeLens
from aurex.config import GRAMS_PER_TROY_OUNCE
from aurex.data.parity import local_premium_bps

BREAK = pd.Timestamp("2026-05-13")
USD = lens_by_code(GOLD, "USD")
INR = lens_by_code(GOLD, "INR")


def flat_inputs(
    start: str = "2026-04-15",
    end: str = "2026-06-15",
    xau: float = 4000.0,
    usdinr: float = 96.5,
) -> tuple[pd.Series, LensContext]:
    """Constant inputs, so any movement comes from the schedule alone."""
    index = pd.bdate_range(start, end, name="date")
    return pd.Series(xau, index=index), LensContext(fx=pd.Series(usdinr, index=index))


class TestParityWiring:
    def test_parity_steps_at_duty_change(self) -> None:
        """The wiring test. Inputs are constant, so parity may only move if the
        duty schedule moved it: 6% -> 15% is a +8.49% step."""
        prices, ctx = flat_inputs()
        frame = INR.apply(prices, ctx)

        before = frame.loc[frame.index < BREAK, "price_ex_consumption_tax"]
        after = frame.loc[frame.index >= BREAK, "price_ex_consumption_tax"]
        assert len(before) and len(after)
        assert before.nunique() == 1 and after.nunique() == 1

        assert after.iloc[0] / before.iloc[-1] - 1.0 == pytest.approx(1.15 / 1.06 - 1.0, rel=1e-9)

    def test_duty_column_reflects_the_schedule(self) -> None:
        frame = INR.apply(*flat_inputs())
        assert frame.loc[frame.index < BREAK, "duty"].unique().tolist() == [0.06]
        assert frame.loc[frame.index >= BREAK, "duty"].unique().tolist() == [0.15]

    def test_parity_matches_the_closed_form(self) -> None:
        frame = INR.apply(*flat_inputs(start="2026-06-01", end="2026-06-10"))
        expected = 4000.0 / GRAMS_PER_TROY_OUNCE * 96.5 * 10 * 1.15
        assert frame["price_ex_consumption_tax"].iloc[0] == pytest.approx(expected)

    def test_gst_applies_only_to_the_inclusive_column(self) -> None:
        frame = INR.apply(*flat_inputs(start="2026-06-01", end="2026-06-10"))
        ratio = frame["price"] / frame["price_ex_consumption_tax"]
        assert ratio.iloc[0] == pytest.approx(1.03)

    def test_every_lens_emits_the_same_columns(self) -> None:
        """Downstream code must never branch on which lens produced a frame."""
        prices, ctx = flat_inputs()
        for frame in (USD.apply(prices, ctx), INR.apply(prices, ctx)):
            assert list(frame.columns) == list(LENS_COLUMNS)


class TestNativeLens:
    def test_usd_view_is_the_untouched_quote(self) -> None:
        prices, ctx = flat_inputs(start="2026-06-01", end="2026-06-10")
        frame = USD.apply(prices, ctx)
        assert (frame["price"] == 4000.0).all()
        assert (frame["duty"] == 0.0).all()
        assert (frame["fx_rate"] == 1.0).all()

    def test_usd_view_needs_no_fx(self) -> None:
        """§15: single exposure, no copula."""
        prices, _ = flat_inputs()
        assert USD.requires_fx is False
        assert USD.apply(prices, LensContext(fx=None)).empty is False

    def test_usd_view_declares_no_local_premium(self) -> None:
        """§15 lists the local-premium signal as not applicable in USD."""
        assert USD.produces_local_premium is False

    def test_unit_conversion_scales_the_quote(self) -> None:
        per_gram = NativeLens(
            code="USD", unit_label="gram", units_per_base=1 / GRAMS_PER_TROY_OUNCE
        )
        prices, ctx = flat_inputs(start="2026-06-01", end="2026-06-05")
        assert per_gram.apply(prices, ctx)["price"].iloc[0] == pytest.approx(
            4000.0 / GRAMS_PER_TROY_OUNCE
        )


class TestPassthroughIsMeasuredNotAsserted:
    def test_premium_is_flat_under_complete_passthrough(self) -> None:
        """Why the wiring test targets parity rather than the premium."""
        frame = INR.apply(*flat_inputs())
        observed = frame["price_ex_consumption_tax"] * 1.004

        premium = local_premium_bps(observed, frame["price_ex_consumption_tax"])
        before = premium.loc[premium.index < BREAK].mean()
        after = premium.loc[premium.index >= BREAK].mean()

        assert before == pytest.approx(40.0, abs=0.5)
        assert after - before == pytest.approx(0.0, abs=1e-6)


class TestDutyLevelCorroboration:
    def test_duty_level_is_corroborated_by_observed_ibja_print(self) -> None:
        """Independent, observational check on the 15% duty.

        The CBIC primary document could not be retrieved, so duty.yaml marks the
        2026-05-13 entry `secondary`. This corroborates the *level* against a real
        IBJA print rather than against more secondary reporting.

        IBJA 999 PM on 2026-07-29 was 142,224 Rs/10g, exclusive of GST. Inputs are
        fixed so the test is deterministic; the claim is about order of magnitude.
        """
        stamp = pd.Timestamp("2026-07-29")
        prices = pd.Series(4000.85, index=[stamp])
        ctx = LensContext(fx=pd.Series(96.56, index=[stamp]))
        ibja_999_pm = 142_224.0

        frame = INR.apply(prices, ctx)
        premium = local_premium_bps(
            pd.Series(ibja_999_pm, index=[stamp]), frame["price_ex_consumption_tax"]
        )
        actual_bps = float(premium.iloc[0])

        assert abs(actual_bps) < 150, f"premium {actual_bps:.0f}bps implies a wrong duty level"

        base = 4000.85 / GRAMS_PER_TROY_OUNCE * 96.56 * 10
        counterfactual_bps = (ibja_999_pm / (base * 1.06) - 1.0) * 10_000
        assert counterfactual_bps > 700
        assert abs(actual_bps) < abs(counterfactual_bps) / 10

    def test_usd_lens_agrees_with_the_london_fix_ibja_prints(self) -> None:
        """Independent cross-check: IBJA's own report carries the London PM fix."""
        stamp = pd.Timestamp("2026-07-29")
        frame = USD.apply(pd.Series(4000.85, index=[stamp]), LensContext())
        assert frame["price"].iloc[0] == pytest.approx(4000.85)


class TestConfidenceTagging:
    def test_pre_gst_parity_is_low_confidence(self) -> None:
        frame = INR.apply(*flat_inputs(start="2017-06-20", end="2017-07-10"))
        cutoff = pd.Timestamp(INR.high_confidence_from)
        assert (frame.loc[frame.index < cutoff, "confidence"] == "low").all()
        assert (frame.loc[frame.index >= cutoff, "confidence"] == "high").all()

    def test_pre_ad_valorem_dates_are_dropped(self) -> None:
        """Duty was Rs 300/10g specific before 2012-01-17; no percentage exists."""
        frame = INR.apply(*flat_inputs(start="2011-12-01", end="2012-02-15"))
        assert (frame.index >= pd.Timestamp("2012-01-17")).all()
        assert frame["duty"].notna().all()


class TestDegenerateInputs:
    def test_missing_fx_is_rejected(self) -> None:
        prices, _ = flat_inputs()
        with pytest.raises(ValueError, match="requires an FX series"):
            INR.apply(prices, LensContext(fx=None))

    def test_empty_prices_yield_an_empty_frame_with_columns(self) -> None:
        empty = pd.Series(dtype=float, index=pd.DatetimeIndex([], name="date"))
        frame = INR.apply(empty, LensContext(fx=empty))
        assert frame.empty
        assert list(frame.columns) == list(LENS_COLUMNS)

    def test_non_overlapping_price_and_fx_yield_empty(self) -> None:
        prices = pd.Series(4000.0, index=pd.bdate_range("2026-01-01", periods=5))
        fx = pd.Series(96.5, index=pd.bdate_range("2026-06-01", periods=5))
        assert INR.apply(prices, LensContext(fx=fx)).empty
