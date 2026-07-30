"""Return transforms, and the hazard they exist to contain.

The fixture is real: WTI's actual settlement window around 2020-04-20, including the
-36.98 print that makes ``log`` undefined.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from aurex.assets.transforms import (
    Difference,
    LogReturn,
    ShiftedLogReturn,
    TransformDomainError,
    round_trip,
)


@pytest.fixture
def wti_negative(fixture_dir: Path) -> pd.Series:
    frame = pd.read_csv(fixture_dir / "wti_negative_2020-04.csv", parse_dates=["date"])
    return frame.set_index("date")["wti"]


@pytest.fixture
def positive_prices() -> pd.Series:
    index = pd.bdate_range("2026-01-01", periods=40, name="date")
    rng = np.random.default_rng(7)
    return pd.Series(100.0 * np.exp(np.cumsum(rng.normal(0, 0.01, len(index)))), index=index)


class TestRoundTrip:
    def test_log_round_trips(self, positive_prices: pd.Series) -> None:
        rebuilt = round_trip(LogReturn(), positive_prices)
        np.testing.assert_allclose(rebuilt, positive_prices.to_numpy()[1:], rtol=1e-12)

    def test_shifted_log_round_trips_across_a_negative_price(self, wti_negative: pd.Series) -> None:
        """§17.1's requirement, on the real print that motivates it."""
        assert (wti_negative < 0).any(), "fixture must contain the negative settlement"
        rebuilt = round_trip(ShiftedLogReturn(shift=50.0), wti_negative)
        np.testing.assert_allclose(rebuilt, wti_negative.to_numpy()[1:], rtol=1e-10)

    def test_difference_round_trips_across_a_negative_price(self, wti_negative: pd.Series) -> None:
        rebuilt = round_trip(Difference(), wti_negative)
        np.testing.assert_allclose(rebuilt, wti_negative.to_numpy()[1:], rtol=1e-12)

    @pytest.mark.parametrize("shift", [50.0, 100.0, 500.0])
    def test_shifted_log_round_trips_for_any_valid_shift(
        self, wti_negative: pd.Series, shift: float
    ) -> None:
        rebuilt = round_trip(ShiftedLogReturn(shift=shift), wti_negative)
        np.testing.assert_allclose(rebuilt, wti_negative.to_numpy()[1:], rtol=1e-10)


class TestDomainErrors:
    def test_log_rejects_negative_prices_with_a_useful_message(
        self, wti_negative: pd.Series
    ) -> None:
        with pytest.raises(TransformDomainError, match="ShiftedLogReturn"):
            LogReturn().to_returns(wti_negative)

    def test_shift_too_small_is_rejected(self, wti_negative: pd.Series) -> None:
        with pytest.raises(TransformDomainError, match="too small"):
            ShiftedLogReturn(shift=10.0).to_returns(wti_negative)

    def test_shift_boundary_is_enforced(self, wti_negative: pd.Series) -> None:
        """shift must strictly exceed the most negative price."""
        worst = float(wti_negative.min())
        with pytest.raises(TransformDomainError):
            ShiftedLogReturn(shift=-worst).to_returns(wti_negative)
        assert ShiftedLogReturn(shift=-worst + 0.01).to_returns(wti_negative).notna().all()


class TestScaleHazard:
    def test_transform_space_volatility_depends_on_an_arbitrary_constant(
        self, wti_negative: pd.Series
    ) -> None:
        """The reason nothing is ever reported out of transform space.

        Round-tripping is exactly what a badly-scaled transform still does, so a
        round-trip test cannot catch this. Same series, same sample, different shift:
        the standard deviation moves substantially. This test documents the hazard so
        nobody reintroduces a percentage computed in transform space.
        """
        sd_50 = ShiftedLogReturn(shift=50.0).to_returns(wti_negative).std()
        sd_100 = ShiftedLogReturn(shift=100.0).to_returns(wti_negative).std()

        assert sd_50 != pytest.approx(sd_100, rel=0.1), (
            "if these ever agree, re-check the claim before relaxing the reporting rule"
        )
        assert sd_50 > sd_100, "a larger shift compresses relative moves"

    def test_price_space_output_is_invariant_to_the_shift(self, wti_negative: pd.Series) -> None:
        """The invariance that actually holds, and the one worth asserting.

        Whatever the shift, mapping back to price space reproduces the same prices.
        So anything reported from price space is unaffected by the constant, which is
        precisely why reporting is defined there.
        """
        rebuilt_50 = round_trip(ShiftedLogReturn(shift=50.0), wti_negative)
        rebuilt_100 = round_trip(ShiftedLogReturn(shift=100.0), wti_negative)
        np.testing.assert_allclose(rebuilt_50, rebuilt_100, rtol=1e-10)

    def test_module_exposes_no_volatility_helper(self) -> None:
        """Structural guard: there must be nothing here to misreport."""
        import aurex.assets.transforms as mod

        exported = [n for n in dir(mod) if not n.startswith("_")]
        for banned in ("volatility", "vol", "annualise", "annualize", "quantile", "sigma"):
            assert not any(banned in name.lower() for name in exported), (
                f"transforms must not expose {banned!r}; reporting happens in price space"
            )


class TestDescribe:
    def test_shift_is_recorded_for_the_artifact(self) -> None:
        described = ShiftedLogReturn(shift=50.0).describe()
        assert described["shift"] == 50.0
        assert described["id"] == "shifted_log"
        assert "caveat" in described, "the shift's effect on scale must be disclosed"

    def test_log_describes_its_formula(self) -> None:
        assert LogReturn().describe()["formula"] == "log(P_t) - log(P_{t-1})"

    def test_difference_declares_price_units(self) -> None:
        assert Difference().describe()["units"] == "price"
