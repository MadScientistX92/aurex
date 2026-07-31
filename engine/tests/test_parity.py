"""Residual semantics: observed versus modelled.

The parity arithmetic moved to :mod:`aurex.assets.lens`; what is tested here is
asset-neutral. See ``tests/test_lens.py`` for the wiring tests.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from aurex.data.parity import local_premium_bps, passthrough_diagnostic

BREAK = pd.Timestamp("2026-05-13")


def modelled_series(value: float = 140_000.0, periods: int = 40) -> pd.Series:
    index = pd.bdate_range("2026-04-15", periods=periods, name="date")
    return pd.Series(value, index=index)


class TestPremiumSemantics:
    def test_premium_is_nan_where_unobserved_never_zero(self) -> None:
        """Filling an unobserved premium with the model would drive the residual to
        zero and fabricate the signal it exists to measure."""
        modelled = modelled_series()
        partial = modelled.iloc[:3] * 1.002

        premium = local_premium_bps(partial, modelled)
        assert premium.iloc[:3].notna().all()
        assert premium.iloc[3:].isna().all()
        assert not (premium.fillna(0.0) == 0.0).all()

    def test_premium_sign_and_magnitude(self) -> None:
        modelled = modelled_series()
        assert local_premium_bps(modelled * 1.004, modelled).mean() == pytest.approx(40.0)
        assert local_premium_bps(modelled * 0.994, modelled).mean() == pytest.approx(-60.0)

    def test_wrong_tax_basis_manufactures_a_premium(self) -> None:
        """IBJA quotes ex-GST; differencing against the inclusive price invents
        roughly -291bps at all times."""
        ex_tax = modelled_series()
        incl_tax = ex_tax * 1.03

        assert local_premium_bps(ex_tax, ex_tax).mean() == pytest.approx(0.0, abs=1e-9)
        assert local_premium_bps(ex_tax, incl_tax).mean() == pytest.approx(
            (1 / 1.03 - 1) * 10_000, abs=1.0
        )

    def test_zero_denominator_does_not_divide_by_zero(self) -> None:
        index = pd.bdate_range("2026-06-01", periods=3, name="date")
        modelled = pd.Series([100.0, 0.0, 100.0], index=index)
        premium = local_premium_bps(pd.Series(101.0, index=index), modelled)
        assert np.isnan(premium.iloc[1])
        assert premium.iloc[0] == pytest.approx(100.0)

    def test_result_is_indexed_to_the_modelled_series(self) -> None:
        modelled = modelled_series(periods=10)
        observed = pd.Series(140_000.0, index=pd.bdate_range("2020-01-01", periods=5))
        premium = local_premium_bps(observed, modelled)
        assert premium.index.equals(modelled.index)
        assert premium.isna().all()


class TestPassthroughDiagnostic:
    def test_reports_incomplete_passthrough_without_judging_it(self) -> None:
        """Incomplete passthrough is a real observation, not a failure."""
        modelled = modelled_series()
        observed = modelled.copy()
        observed.loc[observed.index < BREAK] *= 1.004
        observed.loc[observed.index >= BREAK] *= 0.98

        diagnostic = passthrough_diagnostic(local_premium_bps(observed, modelled), BREAK.date())

        assert diagnostic["mean_bps_before"] == pytest.approx(40.0, abs=1.0)
        assert diagnostic["mean_bps_after"] == pytest.approx(-200.0, abs=1.0)
        assert diagnostic["n_before"] > 0 and diagnostic["n_after"] > 0

    def test_complete_passthrough_shows_no_shift(self) -> None:
        modelled = modelled_series()
        diagnostic = passthrough_diagnostic(
            local_premium_bps(modelled * 1.004, modelled), BREAK.date()
        )
        assert diagnostic["shift_bps"] == pytest.approx(0.0, abs=1e-6)

    def test_handles_a_missing_side(self) -> None:
        modelled = pd.Series(140_000.0, index=pd.bdate_range("2026-06-01", periods=10))
        diagnostic = passthrough_diagnostic(
            local_premium_bps(modelled * 1.001, modelled), BREAK.date()
        )
        assert diagnostic["n_before"] == 0
        assert diagnostic["mean_bps_before"] is None
        assert diagnostic["shift_bps"] is None

    def test_window_bounds_the_sample(self) -> None:
        modelled = modelled_series(periods=60)
        premium = local_premium_bps(modelled * 1.001, modelled)
        assert passthrough_diagnostic(premium, BREAK.date(), window=5)["n_before"] == 5
