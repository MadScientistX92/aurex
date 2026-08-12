"""The weekly design, and what the loadings layer is allowed to claim.

Two things are under test here and they fail in opposite directions. The design can
quietly hand the regression its own target, or quietly let one short series replace the
sample. The loadings layer can quietly turn attribution into forecasting. Every test
below is aimed at one of those three, on data built so the right answer is known.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from aurex.assets.base import FactorSpec
from aurex.assets.synthetic import SYNTHETIC
from aurex.assets.transforms import LogReturn
from aurex.factors import design as design_module
from aurex.factors import loadings as loadings_module
from aurex.factors.design import DesignError, WeeklyDesign

WEEKS = 700


def _daily(name: str, values: np.ndarray, start: str = "2010-01-01") -> pd.DataFrame:
    index = pd.bdate_range(start, periods=len(values), name="date")
    return pd.DataFrame({name: values}, index=index)


class _Stub:
    """The smallest thing satisfying what the design layer reads off an asset."""

    price_series_id = "px"
    return_transform = LogReturn()

    def __init__(self, factor_set: tuple[FactorSpec, ...]) -> None:
        self.factor_set = factor_set


@pytest.fixture
def frames() -> dict[str, pd.DataFrame]:
    rng = np.random.default_rng(19)
    n = WEEKS * 5
    price = 100.0 * np.exp(np.cumsum(rng.normal(0.0, 0.01, n)))
    return {
        "px": _daily("close", price),
        "driver": _daily("value", np.cumsum(rng.normal(0.0, 0.5, n)) + 50.0),
        "wide": pd.DataFrame(
            {
                "alpha": np.arange(float(n)),
                "beta": np.cumsum(rng.normal(0.0, 1.0, n)) + 500.0,
            },
            index=pd.bdate_range("2010-01-01", periods=n, name="date"),
        ),
    }


class TestTheDesignRefusesToCheat:
    def test_a_driver_off_the_price_series_needs_a_lag(
        self, frames: dict[str, pd.DataFrame]
    ) -> None:
        """At lag zero the regressor is the target wearing a different name."""
        asset = _Stub(
            (FactorSpec(id="own", series_id="px", transform="pct_change", description="x"),)
        )
        with pytest.raises(DesignError, match="lag of at least one week"):
            design_module.build(asset, frames)  # type: ignore[arg-type]

    def test_with_the_lag_it_is_allowed_and_explains_nothing(
        self, frames: dict[str, pd.DataFrame]
    ) -> None:
        asset = _Stub(
            (FactorSpec(id="own", series_id="px", transform="pct_change", lag=1, description="x"),)
        )
        built = design_module.build(asset, frames)  # type: ignore[arg-type]
        fitted = np.corrcoef(built.matrix()[:, 0], built.values())[0, 1]

        assert abs(fitted) < 0.2, "a lagged own return should not track this week's return"

    def test_a_missing_required_driver_is_fatal(self, frames: dict[str, pd.DataFrame]) -> None:
        asset = _Stub(
            (FactorSpec(id="gone", series_id="absent", transform="diff", description="x"),)
        )
        with pytest.raises(DesignError, match="required driver"):
            design_module.build(
                asset,  # type: ignore[arg-type]
                frames,
                unavailable={"absent": "the source declined"},
            )

    def test_a_missing_optional_driver_drops_with_the_reason(
        self, frames: dict[str, pd.DataFrame]
    ) -> None:
        asset = _Stub(
            (
                FactorSpec(id="keep", series_id="driver", transform="diff", description="x"),
                FactorSpec(
                    id="gone",
                    series_id="absent",
                    transform="diff",
                    required=False,
                    description="x",
                ),
            )
        )
        built = design_module.build(
            asset,  # type: ignore[arg-type]
            frames,
            unavailable={"absent": "the source declined"},
        )

        assert built.names == ("keep",)
        assert built.dropped["gone"] == "the source declined"


class TestOneShortSeriesMustNotReplaceTheSample:
    """The failure that produced a design of two rows and every number downstream of it."""

    @pytest.fixture
    def truncating(self, frames: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
        short = frames["driver"].tail(15).copy()
        return {**frames, "short": short}

    def _asset(self, *, required: bool) -> _Stub:
        return _Stub(
            (
                FactorSpec(id="keep", series_id="driver", transform="diff", description="x"),
                FactorSpec(
                    id="stub",
                    series_id="short",
                    transform="diff",
                    required=required,
                    description="x",
                ),
            )
        )

    def test_an_optional_one_is_dropped_and_names_both_counts(
        self, truncating: dict[str, pd.DataFrame]
    ) -> None:
        built = design_module.build(self._asset(required=False), truncating)  # type: ignore[arg-type]

        assert built.names == ("keep",)
        assert len(built.frame) > 500, "the sample survived"
        assert "would cut the sample from" in built.dropped["stub"]

    def test_a_required_one_is_allowed_to_bind_the_sample(
        self, truncating: dict[str, pd.DataFrame]
    ) -> None:
        """Required means the answer without it is a different answer, not a smaller one."""
        built = design_module.build(self._asset(required=True), truncating)  # type: ignore[arg-type]

        assert set(built.names) == {"keep", "stub"}
        assert len(built.frame) < 10
        assert built.binding_factor == "stub", "the artifact must name what bound the sample"


class TestTransformsAndAggregation:
    def test_a_named_column_beats_the_positional_fallback(
        self, frames: dict[str, pd.DataFrame]
    ) -> None:
        """The bug this field exists for: the fallback picks a column nobody meant."""
        asset = _Stub(
            (
                FactorSpec(
                    id="named",
                    series_id="wide",
                    column="beta",
                    transform="diff",
                    description="x",
                ),
            )
        )
        built = design_module.build(asset, frames)  # type: ignore[arg-type]

        # `alpha` is a ramp, so its weekly difference is a constant. Reading it instead
        # of `beta` would show up here as a regressor with no variance at all.
        assert built.frame["named"].std() > 0.5

    def test_an_unknown_column_names_what_the_series_carries(
        self, frames: dict[str, pd.DataFrame]
    ) -> None:
        asset = _Stub(
            (
                FactorSpec(
                    id="named",
                    series_id="wide",
                    column="missing",
                    transform="diff",
                    description="x",
                ),
            )
        )
        with pytest.raises(DesignError, match="alpha"):
            design_module.build(asset, frames)  # type: ignore[arg-type]

    def test_mean_and_last_are_different_summaries(self) -> None:
        """One whole Saturday-to-Friday bin, so the arithmetic is checkable by hand."""
        index = pd.date_range("2026-01-10", periods=7, freq="D", name="date")
        assert index[-1] == pd.Timestamp("2026-01-16") and index[-1].dayofweek == 4
        spiky = pd.Series([0.0] * 6 + [70.0], index=index)

        by_last = design_module.to_weekly(spiky, aggregation="last")
        by_mean = design_module.to_weekly(spiky, aggregation="mean")

        assert len(by_last) == 1, "the seven days must fall in one weekly bin"
        assert by_last.iloc[0] == 70.0
        assert by_mean.iloc[0] == pytest.approx(10.0)

    def test_a_proportional_change_through_zero_is_missing_not_infinite(self) -> None:
        weekly = pd.Series(
            [1.0, 0.0, 2.0], index=pd.date_range("2026-01-02", periods=3, freq="W-FRI")
        )
        changed = design_module.apply_transform(weekly, transform="pct_change")

        assert np.isfinite(changed.dropna()).all()
        assert changed.isna().sum() == 2

    @pytest.mark.parametrize("bad", ["mean_of_squares", "ewma"])
    def test_an_unknown_aggregation_raises(self, bad: str) -> None:
        with pytest.raises(DesignError, match="unknown aggregation"):
            design_module.to_weekly(pd.Series(dtype=float), aggregation=bad)

    def test_an_unknown_transform_raises(self) -> None:
        with pytest.raises(DesignError, match="unknown transform"):
            design_module.apply_transform(pd.Series(dtype=float), transform="detrend")


def _known_design(
    *, contemporaneous_signal: float, predictive_signal: float, seed: int = 21
) -> WeeklyDesign:
    """A design where the two out-of-sample questions have different known answers."""
    rng = np.random.default_rng(seed)
    index = pd.date_range("2010-01-01", periods=WEEKS, freq="W-FRI", name="date")
    first = rng.normal(size=WEEKS)
    second = rng.normal(size=WEEKS)
    noise = rng.normal(scale=0.5, size=WEEKS)

    target = contemporaneous_signal * first + noise
    if predictive_signal:
        target = target + predictive_signal * np.concatenate([[0.0], first[:-1]])

    frame = pd.DataFrame({"first": first, "second": second}, index=index)
    specs = tuple(
        FactorSpec(id=name, series_id=name, transform="level", description="x")
        for name in frame.columns
    )
    return WeeklyDesign(
        target=pd.Series(target, index=index),
        frame=frame,
        dropped={},
        specs=specs,
        binding_factor="first",
    )


class TestTheTwoOutOfSampleQuestions:
    """Contemporaneous and predictive must not be able to stand in for one another."""

    def test_a_contemporaneous_signal_is_found_and_does_not_forecast(self) -> None:
        design = _known_design(contemporaneous_signal=1.0, predictive_signal=0.0)
        matrix, values = design.matrix(), design.values()

        found = loadings_module.walk_forward(matrix, values, kind="contemporaneous", window=156)
        forecast = loadings_module.walk_forward(
            matrix[:-1], values[1:], kind="predictive", window=156
        )

        assert found is not None and forecast is not None
        assert found.r_squared > 0.5, "a strong same-week relationship must show up"
        assert forecast.r_squared < 0.05, "and must not become a forecast"
        assert forecast.p_value is not None and forecast.p_value > 0.05

    def test_a_genuinely_predictive_signal_is_not_hidden(self) -> None:
        """The other direction: the predictive measurement must be able to find one."""
        design = _known_design(contemporaneous_signal=0.0, predictive_signal=1.5)
        matrix, values = design.matrix(), design.values()

        forecast = loadings_module.walk_forward(
            matrix[:-1], values[1:], kind="predictive", window=156
        )
        assert forecast is not None and forecast.r_squared > 0.5

    def test_pure_noise_scores_at_or_below_the_benchmark(self) -> None:
        design = _known_design(contemporaneous_signal=0.0, predictive_signal=0.0)
        scored = loadings_module.walk_forward(
            design.matrix(), design.values(), kind="contemporaneous", window=156
        )

        assert scored is not None
        assert scored.r_squared < 0.02
        assert scored.p_value is not None and scored.p_value > 0.05

    def test_too_short_a_sample_returns_nothing_rather_than_a_number(self) -> None:
        design = _known_design(contemporaneous_signal=1.0, predictive_signal=0.0)
        assert (
            loadings_module.walk_forward(
                design.matrix()[:100], design.values()[:100], kind="contemporaneous", window=156
            )
            is None
        )


class TestWhatTheLoadingsReport:
    @pytest.fixture(scope="class")
    @staticmethod
    def attribution() -> loadings_module.Attribution:
        design = _known_design(contemporaneous_signal=1.0, predictive_signal=0.0)
        return loadings_module.estimate(design, draws=200, window=156)

    def test_the_driver_that_matters_dominates_the_one_that_does_not(
        self, attribution: loadings_module.Attribution
    ) -> None:
        """Stated as an ordering rather than as a threshold on the null driver.

        Cross-validation picks a small penalty when one regressor carries a strong
        signal, and at a small penalty a pure-noise regressor is often kept with a tiny
        coefficient — which is elastic-net behaviour, not a defect. What must hold is
        that the two are not confusable in size.
        """
        by_id = {entry.factor_id: entry for entry in attribution.loadings}

        assert by_id["first"].selected
        assert by_id["first"].selection_rate == 1.0
        assert abs(by_id["second"].standardised) < abs(by_id["first"].standardised) / 10.0

    def test_the_ols_interval_is_published_beside_the_penalised_one(
        self, attribution: loadings_module.Attribution
    ) -> None:
        """The penalised interval is not valid at zero; the OLS one is the honest pair."""
        by_id = {entry.factor_id: entry for entry in attribution.loadings}

        assert by_id["first"].ols_excludes_zero
        assert by_id["first"].ols_p_value == pytest.approx(0.0, abs=1e-12)
        # Not "the null driver's interval covers zero" — on any one sample that is a
        # coin flip at the 5% level, and a test that fails one run in twenty is a test
        # nobody keeps. What must hold is that the evidence is orders apart.
        second_p = by_id["second"].ols_p_value
        assert second_p is not None and second_p > 1e6 * max(
            by_id["first"].ols_p_value or 0.0, 1e-300
        )

    def test_hac_errors_match_plain_ols_on_independent_residuals(self) -> None:
        """The sandwich must reduce to the textbook estimator when there is no dependence."""
        rng = np.random.default_rng(5)
        n = 800
        matrix = rng.normal(size=(n, 3))
        residual = rng.normal(scale=0.4, size=n)
        target = matrix @ np.array([1.0, -0.5, 0.0]) + residual

        _, hac, _ = loadings_module.ols_with_hac(matrix, target, lag=0)

        standardised, centred, _ = loadings_module.elasticnet.standardise(matrix, target)
        with_const = np.column_stack([np.ones(n), standardised])
        coefficients = np.linalg.lstsq(with_const, centred, rcond=None)[0]
        errors = centred - with_const @ coefficients
        classical = np.sqrt(
            np.diag(np.linalg.pinv(with_const.T @ with_const)) * float(errors @ errors) / n
        )

        assert hac == pytest.approx(classical[1:], rel=0.05)

    def test_the_artifact_block_is_json_safe(
        self, attribution: loadings_module.Attribution
    ) -> None:
        import json

        json.dumps(attribution.describe())

    def test_the_caveat_travels_with_the_interval(
        self, attribution: loadings_module.Attribution
    ) -> None:
        """A bootstrap interval around a penalised estimator must not be quoted bare."""
        block = attribution.describe()
        assert "not valid" in block["bootstrap"]["caveat"]
        assert "selection_rate" in block["loadings"][0]["reading"]


class TestStabilityMeasuresARegimeChange:
    def test_a_loading_that_flips_is_reported_as_flipping(self) -> None:
        """Built from a known break, so "poor stability" has something to be graded on."""
        rng = np.random.default_rng(31)
        n = 800
        index = pd.date_range("2008-01-04", periods=n, freq="W-FRI", name="date")
        driver = rng.normal(size=n)
        sign = np.where(np.arange(n) < n // 2, 1.0, -1.0)
        target = sign * driver + rng.normal(scale=0.3, size=n)

        design = WeeklyDesign(
            target=pd.Series(target, index=index),
            frame=pd.DataFrame({"switcher": driver}, index=index),
            dropped={},
            specs=(FactorSpec(id="switcher", series_id="s", transform="level", description="x"),),
        )
        attribution = loadings_module.estimate(design, draws=100, window=156)
        stability = {entry.factor_id: entry for entry in attribution.stability}["switcher"]

        assert stability.sign_changes >= 1
        assert stability.first_half_mean > 0 > stability.second_half_mean


class TestTheGuardOnTheAssetAbstraction:
    def test_the_design_layer_runs_on_an_asset_it_has_never_heard_of(self) -> None:
        """§16, for the package this step adds: no driver may be named in ``factors/``."""
        rng = np.random.default_rng(3)
        n = 1500
        index = pd.bdate_range("2015-01-01", periods=n, name="date")
        frames = {
            series_id: pd.DataFrame(
                {"close": 100.0 * np.exp(np.cumsum(rng.normal(0.0, 0.01, n)))}, index=index
            )
            for series_id in ("widget_price", "widget_fx", "widget_local")
        }
        built = design_module.build(SYNTHETIC, frames, unavailable={"absent_series": "no source"})

        assert len(built.frame) > 100
        assert built.names, "the synthetic asset must produce a usable design"
        assert loadings_module.estimate(built, draws=50, window=156) is not None


class TestTheOmittedVariableCheck:
    """The counterfactual behind every "this driver must be here" argument."""

    def test_removing_a_driver_that_matters_moves_what_is_left(self) -> None:
        """Two correlated drivers, one of which carries the signal: removing it must show."""
        rng = np.random.default_rng(47)
        n = 700
        index = pd.date_range("2010-01-01", periods=n, freq="W-FRI", name="date")
        shared = rng.normal(size=n)
        first = shared + rng.normal(scale=0.4, size=n)
        second = shared + rng.normal(scale=0.4, size=n)
        target = 2.0 * first + rng.normal(scale=0.3, size=n)

        design = WeeklyDesign(
            target=pd.Series(target, index=index),
            frame=pd.DataFrame({"first": first, "second": second}, index=index),
            dropped={},
            specs=tuple(
                FactorSpec(id=name, series_id=name, transform="level", description="x")
                for name in ("first", "second")
            ),
        )
        attribution = loadings_module.estimate(design, draws=50, window=156)
        withheld = {entry.factor_id: entry for entry in attribution.withheld}

        assert withheld["first"].largest_shift > 0.05, (
            "dropping the driver that carries the signal must move its correlated twin"
        )
        assert withheld["first"].r_squared_without is not None
        assert withheld["first"].r_squared_without < withheld["first"].r_squared_with  # type: ignore[operator]

    def test_removing_an_irrelevant_driver_moves_almost_nothing(self) -> None:
        """The negative result this check exists to be able to report honestly."""
        rng = np.random.default_rng(53)
        n = 700
        index = pd.date_range("2010-01-01", periods=n, freq="W-FRI", name="date")
        first = rng.normal(size=n)
        noise = rng.normal(size=n)
        target = 2.0 * first + rng.normal(scale=0.3, size=n)

        design = WeeklyDesign(
            target=pd.Series(target, index=index),
            frame=pd.DataFrame({"first": first, "noise": noise}, index=index),
            dropped={},
            specs=tuple(
                FactorSpec(id=name, series_id=name, transform="level", description="x")
                for name in ("first", "noise")
            ),
        )
        attribution = loadings_module.estimate(design, draws=50, window=156)
        withheld = {entry.factor_id: entry for entry in attribution.withheld}

        assert withheld["noise"].largest_shift < 0.02
        assert withheld["noise"].sign_flips == ()

    def test_only_required_drivers_are_withheld(self) -> None:
        """An optional one already vanishes on its own whenever its source fails."""
        rng = np.random.default_rng(59)
        n = 400
        index = pd.date_range("2010-01-01", periods=n, freq="W-FRI", name="date")
        frame = pd.DataFrame(
            {"needed": rng.normal(size=n), "spare": rng.normal(size=n)}, index=index
        )
        design = WeeklyDesign(
            target=pd.Series(rng.normal(size=n), index=index),
            frame=frame,
            dropped={},
            specs=(
                FactorSpec(id="needed", series_id="a", transform="level", description="x"),
                FactorSpec(
                    id="spare",
                    series_id="b",
                    transform="level",
                    required=False,
                    description="x",
                ),
            ),
        )
        attribution = loadings_module.estimate(design, draws=50, window=156)

        assert [entry.factor_id for entry in attribution.withheld] == ["needed"]
