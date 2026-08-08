"""Tests for the multiple-comparisons machinery and the power calculation.

The shootout's whole claim is that it will not find a winner that is not there, so the
tests that matter are the ones run against data with no winner in it. Three of these are
Monte Carlo over simulated loss series rather than assertions about one number, because
"this test has approximately the right size" is a frequency statement and cannot be
checked any other way.

The heavy models are not exercised here. They live behind the ``bench`` extra and their
own workflow; what runs on every push is the statistics, because a multiple-comparisons
correction nobody checks is a correction nobody has.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from aurex.assets.synthetic import SYNTHETIC
from aurex.score import (
    Sampling,
    TerminalAbove,
    WalkForwardRequest,
    diebold_mariano,
    reliability_curve,
)
from aurex.score.shootout import (
    NonFiniteLossError,
    minimum_detectable_effect,
    model_confidence_set,
    resolution_screen,
    superior_predictive_ability,
)

DISJOINT = Sampling(horizon=1, step=1)


def _rising_prices(n: int = 700) -> pd.Series:
    """A series that appreciates, which is the case a drift policy actually bites on."""
    rng = np.random.default_rng(41)
    values = 100.0 * np.exp(np.cumsum(rng.normal(0.0006, 0.01, n)))
    return pd.Series(values, index=pd.bdate_range("2018-01-01", periods=n), name="price")


def _losses(rng: np.random.Generator, n: int, k: int, edge: float = 0.0) -> tuple[dict, np.ndarray]:
    """``k`` models and a benchmark, sharing a common shock so they are correlated.

    The shared component matters: models fitted to one series have loss differentials
    that move together, and a test validated only on independent columns would be
    validated on a case this repository never produces.
    """
    common = rng.normal(size=n)
    benchmark = 1.0 + 0.3 * common + 0.2 * rng.normal(size=n)
    models = {f"m{i}": benchmark - edge + 0.2 * rng.normal(size=n) for i in range(k)}
    return models, benchmark


class TestSuperiorPredictiveAbility:
    def test_it_does_not_find_a_winner_that_is_not_there(self) -> None:
        """Six models, none better than the benchmark. Rejection should be rare."""
        rng = np.random.default_rng(20260807)
        rejections = 0
        trials = 120
        for _ in range(trials):
            models, benchmark = _losses(rng, n=200, k=6)
            spa = superior_predictive_ability(
                models, benchmark, benchmark="rw", sampling=DISJOINT, draws=400
            )
            assert spa is not None
            rejections += spa.rejects

        # Nominal size is 5%. A one-test-per-model shootout would land near 20-25% here,
        # which is the whole reason this module exists.
        assert rejections / trials < 0.15

    def test_it_finds_a_winner_that_is_there(self) -> None:
        rng = np.random.default_rng(11)
        models, benchmark = _losses(rng, n=400, k=6)
        models["m0"] = benchmark - 0.15 + 0.05 * rng.normal(size=400)

        spa = superior_predictive_ability(
            models, benchmark, benchmark="rw", sampling=DISJOINT, draws=2_000
        )

        assert spa is not None
        assert spa.rejects
        assert spa.best == "m0"

    def test_recentring_protects_power_against_hopeless_entrants(self) -> None:
        """The difference between SPA and the Reality Check, on the case that shows it."""
        rng = np.random.default_rng(7)
        n = 400
        models, benchmark = _losses(rng, n=n, k=1)
        models["winner"] = benchmark - 0.12 + 0.05 * rng.normal(size=n)
        # Models so far behind the benchmark that they say nothing about the winner.
        for i in range(8):
            models[f"hopeless{i}"] = benchmark + 5.0 + rng.normal(size=n)

        spa = superior_predictive_ability(
            models, benchmark, benchmark="rw", sampling=DISJOINT, draws=2_000
        )

        assert spa is not None
        assert spa.p_value_consistent <= spa.p_value_conservative
        assert spa.p_value_consistent < 0.05

    def test_the_sign_convention_is_benchmark_minus_model(self) -> None:
        rng = np.random.default_rng(3)
        n = 200
        benchmark = np.full(n, 2.0)
        models = {"better": np.full(n, 1.0) + 0.01 * rng.normal(size=n)}

        spa = superior_predictive_ability(
            models, benchmark, benchmark="rw", sampling=DISJOINT, draws=500
        )

        assert spa is not None
        # Positive: the model lost less. Opposite to the DM convention next door.
        assert spa.mean_differentials[0] > 0.0

    def test_an_empty_set_is_none_rather_than_a_verdict(self) -> None:
        assert (
            superior_predictive_ability(
                {}, np.ones(50), benchmark="rw", sampling=DISJOINT, draws=100
            )
            is None
        )


class TestModelConfidenceSet:
    def test_indistinguishable_models_all_survive(self) -> None:
        rng = np.random.default_rng(5)
        models, benchmark = _losses(rng, n=300, k=4)
        models["rw"] = benchmark

        mcs = model_confidence_set(models, sampling=DISJOINT, draws=1_000)

        assert mcs is not None
        assert set(mcs.included) == set(models)
        assert mcs.eliminated == ()

    def test_a_clearly_worse_model_is_eliminated(self) -> None:
        rng = np.random.default_rng(9)
        n = 400
        models, benchmark = _losses(rng, n=n, k=3)
        models["awful"] = benchmark + 3.0 + 0.05 * rng.normal(size=n)
        models["rw"] = benchmark

        mcs = model_confidence_set(models, sampling=DISJOINT, draws=1_000)

        assert mcs is not None
        assert "awful" not in mcs.included
        assert "awful" in {name for name, _ in mcs.eliminated}

    def test_it_is_a_set_and_not_a_ranking(self) -> None:
        """The output names survivors, and says nothing about the order among them."""
        rng = np.random.default_rng(2)
        models, benchmark = _losses(rng, n=200, k=3)
        models["rw"] = benchmark

        mcs = model_confidence_set(models, sampling=DISJOINT, draws=500)

        assert mcs is not None
        assert list(mcs.included) == sorted(mcs.included)

    def test_one_model_cannot_form_a_confidence_set(self) -> None:
        assert model_confidence_set({"only": np.ones(50)}, sampling=DISJOINT) is None


class TestMinimumDetectableEffect:
    def test_it_shrinks_as_the_sample_grows(self) -> None:
        rng = np.random.default_rng(4)
        small = minimum_detectable_effect(
            rng.normal(1.0, 0.2, 100),
            rng.normal(1.0, 0.2, 100),
            horizon=1,
            sampling=DISJOINT,
        )
        big = minimum_detectable_effect(
            rng.normal(1.0, 0.2, 2_000),
            rng.normal(1.0, 0.2, 2_000),
            horizon=1,
            sampling=DISJOINT,
        )

        assert small.effect_in_skill is not None and big.effect_in_skill is not None
        assert big.effect_in_skill < small.effect_in_skill

    def test_an_effect_at_the_mde_is_detected_about_four_times_in_five(self) -> None:
        """The claim the number makes, checked by making it true and counting.

        This is the test that would catch a power calculation that quoted the wrong
        quantile or forgot the small-sample correction — both of which produce a
        plausible-looking number that is simply the wrong size.
        """
        rng = np.random.default_rng(20260807)
        n, spread = 300, 0.2

        # The MDE is a statement about the differential's variance, so the power
        # simulation has to generate differentials with that same variance. Deriving the
        # effect from one spread and then testing at another measures the mismatch.
        reference = rng.normal(1.0, 0.2, n)
        mde = minimum_detectable_effect(
            reference + rng.normal(0.0, spread, n),
            reference,
            horizon=1,
            sampling=DISJOINT,
        )
        assert mde.effect_in_loss is not None
        effect = mde.effect_in_loss

        detected = 0
        trials = 400
        for _ in range(trials):
            benchmark = rng.normal(1.0, 0.2, n)
            # A model better than the benchmark by exactly the detectable effect, with
            # the differential spread the MDE was computed from.
            model = benchmark - effect + rng.normal(0.0, spread, n)
            test = diebold_mariano(model, benchmark, sampling=DISJOINT, null="rw")
            detected += test.p_value is not None and test.p_value < 0.05

        power = detected / trials
        assert 0.70 < power < 0.90, f"power at the stated MDE was {power:.2f}, expected ~0.80"

    def test_it_is_undefined_rather_than_zero_on_an_unusable_sample(self) -> None:
        result = minimum_detectable_effect(np.ones(2), np.ones(2), horizon=1, sampling=DISJOINT)

        assert result.effect_in_skill is None
        assert result.undefined_reason is not None

    def test_identical_forecasters_have_no_detectable_effect(self) -> None:
        shared = np.linspace(1.0, 2.0, 100)
        result = minimum_detectable_effect(shared, shared, horizon=1, sampling=DISJOINT)

        assert result.effect_in_loss is None
        assert "no variance" in str(result.undefined_reason)

    def test_declaring_the_overlap_widens_the_detectable_effect(self) -> None:
        """Dependence costs power, and the MDE has to charge for it.

        The series here is a moving average of iid noise, which is exactly the
        dependence overlapping windows produce: consecutive differentials share most of
        their path. The same data is then measured twice, once with the overlap declared
        and once with it hidden. Testing this on iid draws instead would show nothing,
        because there the two declarations differ only by the small-sample correction —
        the dependence has to be in the data for the estimator to find it.
        """
        rng = np.random.default_rng(6)
        n, stride = 500, Sampling(horizon=63, step=5).stride

        noise = rng.normal(0.0, 0.2, n + stride)
        overlapped = np.convolve(noise, np.ones(stride) / stride, mode="valid")[:n]
        benchmark = rng.normal(1.0, 0.2, n)
        model = benchmark + overlapped

        declared = minimum_detectable_effect(
            model, benchmark, horizon=63, sampling=Sampling(horizon=63, step=5)
        )
        hidden = minimum_detectable_effect(
            model, benchmark, horizon=1, sampling=Sampling(horizon=1, step=1)
        )

        assert declared.effect_in_skill is not None
        assert hidden.effect_in_skill is not None
        assert declared.effect_in_skill > hidden.effect_in_skill
        assert declared.n_independent < hidden.n_independent


class TestTheBenchExtraStaysOut:
    """The 2.5GB stack must not arrive with ``import aurex``.

    A guard rather than a convention, because the failure is silent and slow: someone
    adds a module-level import, CI keeps passing, and every job in the repository starts
    installing torch. The default suite would go on being green the whole time.
    """

    def test_importing_the_engine_does_not_import_the_bench_stack(self) -> None:
        import subprocess
        import sys

        probe = (
            "import sys; import aurex, aurex.score, aurex.backtest, aurex.cli; "
            "heavy = [m for m in ('torch','statsforecast','neuralforecast','chronos') "
            "if m in sys.modules]; print(','.join(heavy))"
        )
        out = subprocess.run(
            [sys.executable, "-c", probe], capture_output=True, text=True, check=True
        )

        assert out.stdout.strip() == "", f"bench dependencies imported by default: {out.stdout}"

    def test_importing_the_bench_package_is_also_free(self) -> None:
        """Even the shootout's own package defers, so `--help` costs nothing."""
        import subprocess
        import sys

        probe = (
            "import sys; import aurex.bench; "
            "heavy = [m for m in ('torch','statsforecast','neuralforecast','chronos') "
            "if m in sys.modules]; print(','.join(heavy))"
        )
        out = subprocess.run(
            [sys.executable, "-c", probe], capture_output=True, text=True, check=True
        )

        assert out.stdout.strip() == ""


class TestSpaAgainstDieboldMariano:
    def test_spa_is_more_conservative_than_the_best_single_test(self) -> None:
        """The point of the whole module, stated as an inequality.

        Six DM tests and take the smallest p-value: that is the search this replaces.
        SPA's p-value must not be smaller, or it would be finding *more* winners than
        the uncorrected procedure it exists to discipline.
        """
        rng = np.random.default_rng(20260101)
        for _ in range(30):
            models, benchmark = _losses(rng, n=250, k=6)
            spa = superior_predictive_ability(
                models, benchmark, benchmark="rw", sampling=DISJOINT, draws=800
            )
            assert spa is not None

            best_single = min(
                test.p_value
                for name in models
                if (
                    test := diebold_mariano(models[name], benchmark, sampling=DISJOINT, null="rw")
                ).p_value
                is not None
            )
            # One-sided against two-sided, so halve the DM p-value for the comparison.
            assert spa.p_value_consistent >= best_single / 2.0 - 1e-9


class TestABrokenModelStopsTheRunRatherThanWinningIt:
    """A NaN loss series used to produce ``SPA p = 0.000``.

    NaN compares ``False`` against everything, so ``mean(bootstrap > statistic)`` came
    back as zero — the most significant p-value the test can report, produced by a model
    that had not forecast anything. Found by running the shootout with a model that was
    silently returning NaN, and worth a test because the failure is invisible in the
    output: a perfect p-value looks like a result.
    """

    def test_spa_refuses_a_nan_loss_series(self) -> None:
        rng = np.random.default_rng(1)
        models, benchmark = _losses(rng, n=100, k=2)
        models["broken"] = np.full(100, np.nan)

        with pytest.raises(NonFiniteLossError, match="broken"):
            superior_predictive_ability(
                models, benchmark, benchmark="rw", sampling=DISJOINT, draws=100
            )

    def test_the_confidence_set_refuses_one_too(self) -> None:
        rng = np.random.default_rng(1)
        models, _ = _losses(rng, n=100, k=2)
        models["broken"] = np.full(100, np.inf)

        with pytest.raises(NonFiniteLossError, match="broken"):
            model_confidence_set(models, sampling=DISJOINT, draws=100)

    def test_the_power_calculation_refuses_one_too(self) -> None:
        with pytest.raises(NonFiniteLossError):
            minimum_detectable_effect(
                np.full(50, np.nan), np.ones(50), horizon=1, sampling=DISJOINT
            )

    def test_a_finite_set_is_unaffected(self) -> None:
        rng = np.random.default_rng(1)
        models, benchmark = _losses(rng, n=100, k=2)

        assert (
            superior_predictive_ability(
                models, benchmark, benchmark="rw", sampling=DISJOINT, draws=100
            )
            is not None
        )


def _persistent(rng: np.random.Generator, n: int, rho: float = 0.9) -> np.ndarray:
    """An AR(1) series, standing in for the dependence overlapping windows produce."""
    innovations = rng.normal(size=n)
    series = np.zeros(n)
    for i in range(1, n):
        series[i] = rho * series[i - 1] + innovations[i]
    return series


class TestResolutionScreen:
    """The direction test: can any model tell one window from another?

    Resolution is the metric because it is level-invariant. A Brier score confounds
    calibration with discrimination, so a model that knows the base rate and nothing else
    beats a model that has real signal and the wrong level — which is the opposite of
    what "calls direction" means.
    """

    def test_the_resolution_it_scores_is_the_one_reliability_curve_publishes(self) -> None:
        """The binning is shared, and this is what says so.

        A published resolution and the p-value grading it must come from one binning. If
        they ever diverge, both numbers still look reasonable and they are no longer about
        the same quantity — which is the failure that has no symptom.
        """
        rng = np.random.default_rng(11)
        outcomes = rng.integers(0, 2, size=300).astype(float)
        probabilities = np.clip(0.5 + 0.3 * (outcomes - 0.5) + rng.normal(0, 0.1, 300), 0.0, 1.0)

        screen = resolution_screen(
            {"m": probabilities}, outcomes, event="direction_up", sampling=DISJOINT, draws=200
        )
        curve = reliability_curve(probabilities, outcomes)

        assert screen is not None
        assert screen.resolutions[0] == pytest.approx(curve.resolution)

    def test_a_model_with_the_base_rate_and_no_discrimination_scores_nothing(self) -> None:
        """The pre-registration's expected case, and the reason Brier is not the metric:
        this forecaster is perfectly reliable and completely uninformative."""
        rng = np.random.default_rng(12)
        outcomes = (rng.random(400) < 0.6).astype(float)
        constant = np.full(400, 0.6)

        screen = resolution_screen(
            {"flat": constant}, outcomes, event="direction_up", sampling=DISJOINT, draws=500
        )

        assert screen is not None
        assert screen.resolutions[0] == pytest.approx(0.0, abs=1e-12)
        assert not screen.rejects

    def test_a_wrong_level_with_real_signal_still_scores_resolution(self) -> None:
        """The case Brier would rank below the useless forecaster above.

        Its level is badly wrong — it forecasts 0.90 on average against a base rate of
        0.50, so it is close to certain the price rises every single window — but it
        separates the windows that rose from the ones that did not, perfectly. Resolution
        sees that; the Brier score charges it for the level and ranks it below a
        forecaster that knows nothing.
        """
        rng = np.random.default_rng(13)
        outcomes = (rng.random(400) < 0.5).astype(float)
        biased = np.where(outcomes > 0, 0.95, 0.85)

        screen = resolution_screen(
            {"biased": biased}, outcomes, event="direction_up", sampling=DISJOINT, draws=500
        )
        informative = reliability_curve(biased, outcomes)
        useless = reliability_curve(np.full(400, float(outcomes.mean())), outcomes)

        assert screen is not None
        assert screen.rejects
        # Real discrimination, and a Brier score that nonetheless loses to a constant.
        assert informative.resolution > 0.2
        assert informative.brier > useless.brier
        assert useless.resolution == pytest.approx(0.0, abs=1e-12)

    def test_it_finds_discrimination_when_one_model_of_six_has_it(self) -> None:
        rng = np.random.default_rng(14)
        outcomes = (rng.random(500) < 0.5).astype(float)
        models = {f"noise{i}": rng.random(500) for i in range(5)}
        models["signal"] = np.clip(
            0.5 + 0.25 * (outcomes - 0.5) + rng.normal(0, 0.05, 500), 0.0, 1.0
        )

        screen = resolution_screen(
            models, outcomes, event="direction_up", sampling=DISJOINT, draws=1_000
        )

        assert screen is not None
        assert screen.rejects
        assert screen.best == "signal"

    def test_it_does_not_find_discrimination_that_is_not_there(self) -> None:
        """Six models, none of which knows anything. Rejection should be rare.

        The size statement, run the only way a size statement can be checked. Note the
        forecasts here are *persistent* as well as uninformative, which is the case that
        breaks a naive permutation.
        """
        rng = np.random.default_rng(15)
        rejections = 0
        trials = 100
        for _ in range(trials):
            outcomes = (_persistent(rng, 300) > 0).astype(float)
            models = {
                f"m{i}": np.clip(0.5 + 0.1 * _persistent(rng, 300), 0.0, 1.0) for i in range(6)
            }
            screen = resolution_screen(
                models,
                outcomes,
                event="direction_up",
                sampling=Sampling(horizon=21, step=5),
                draws=400,
            )
            assert screen is not None
            rejections += screen.rejects

        assert rejections / trials < 0.15

    def test_a_permutation_would_over_reject_on_the_same_dependent_data(self) -> None:
        """Why the circular shift is not a detail.

        The identical forecasts and outcomes, told they are non-overlapping so the
        permutation branch runs. Breaking the serial dependence makes the null's bin means
        less variable than they really are, and ordinary persistence starts reading as
        skill. This is the failure the shift exists to prevent, asserted rather than
        described.
        """
        rng = np.random.default_rng(16)
        overlapping = Sampling(horizon=21, step=5)
        shift_rejections = 0
        permutation_rejections = 0
        trials = 60

        for _ in range(trials):
            outcomes = (_persistent(rng, 300) > 0).astype(float)
            models = {"m": np.clip(0.5 + 0.1 * _persistent(rng, 300), 0.0, 1.0)}
            shifted = resolution_screen(
                models, outcomes, event="direction_up", sampling=overlapping, draws=400
            )
            permuted = resolution_screen(
                models, outcomes, event="direction_up", sampling=DISJOINT, draws=400
            )
            assert shifted is not None and permuted is not None
            assert shifted.scheme == "cyclic_shift"
            assert permuted.scheme == "permutation"
            shift_rejections += shifted.rejects
            permutation_rejections += permuted.rejects

        assert permutation_rejections > shift_rejections
        assert shift_rejections / trials < 0.15

    def test_the_p_value_can_never_be_zero(self) -> None:
        """A permutation test that reports p = 0 is reporting its draw count."""
        rng = np.random.default_rng(17)
        outcomes = (rng.random(300) < 0.5).astype(float)
        perfect = outcomes.copy()

        screen = resolution_screen(
            {"oracle": perfect}, outcomes, event="direction_up", sampling=DISJOINT, draws=200
        )

        assert screen is not None
        assert screen.p_value > 0.0
        assert screen.p_value == pytest.approx(1.0 / 201.0)

    def test_a_model_confined_to_one_bin_cannot_win_the_maximum(self) -> None:
        """It has no resolution under any alignment, so its column is degenerate. Left
        unguarded that is a division by nothing, and the model would take the maximum on
        an arithmetic artifact rather than on anything it forecast."""
        rng = np.random.default_rng(18)
        outcomes = (rng.random(400) < 0.5).astype(float)
        models = {
            "degenerate": np.full(400, 0.55),
            "ordinary": np.clip(0.5 + rng.normal(0, 0.15, 400), 0.0, 1.0),
        }

        screen = resolution_screen(
            models, outcomes, event="direction_up", sampling=DISJOINT, draws=500
        )

        assert screen is not None
        assert np.isfinite(screen.statistic)
        assert screen.best != "degenerate"

    def test_the_null_mean_resolution_is_published_because_zero_is_not_free(self) -> None:
        """Binned resolution is a sum of squares, so an uninformative forecaster spread
        over ten bins still posts a positive figure. Reading it against zero rather than
        against this number would make noise look like a finding."""
        rng = np.random.default_rng(19)
        outcomes = (rng.random(300) < 0.5).astype(float)

        screen = resolution_screen(
            {"noise": rng.random(300)},
            outcomes,
            event="direction_up",
            sampling=DISJOINT,
            draws=500,
        )

        assert screen is not None
        assert screen.null_means[0] > 0.0
        described = screen.describe()["models"]["noise"]
        assert described["resolution_under_null"] > 0.0

    def test_too_few_reference_draws_is_refused_rather_than_reported(self) -> None:
        """A sample so short the guard leaves almost no admissible shift cannot produce a
        p-value that could reject, and returning one anyway would be a number with no
        test behind it."""
        rng = np.random.default_rng(20)
        outcomes = (rng.random(12) < 0.5).astype(float)

        screen = resolution_screen(
            {"m": rng.random(12)},
            outcomes,
            event="direction_up",
            sampling=Sampling(horizon=63, step=5),
            draws=200,
        )

        assert screen is None

    def test_a_narrow_forecast_range_scores_zero_at_equal_width_and_is_flagged(self) -> None:
        """The blind spot that would have made the pre-registration self-fulfilling.

        This forecaster discriminates perfectly — it is higher on every window that rose —
        but its whole range sits inside one equal-width bin, which is what a direction
        forecast near one half looks like. Resolution is then zero whatever it knew.
        Publishing that as "no discrimination" would be assuming the answer, so the bin
        count travels with the score and the equal-count run is what actually asks.
        """
        rng = np.random.default_rng(22)
        outcomes = (rng.random(400) < 0.5).astype(float)
        narrow = np.where(outcomes > 0, 0.54, 0.52)

        wide = resolution_screen(
            {"narrow": narrow}, outcomes, event="direction_up", sampling=DISJOINT, draws=500
        )
        counted = resolution_screen(
            {"narrow": narrow},
            outcomes,
            event="direction_up",
            sampling=DISJOINT,
            binning="equal_count",
            draws=500,
        )

        assert wide is not None and counted is not None
        assert wide.resolutions[0] == pytest.approx(0.0, abs=1e-12)
        assert wide.occupied_bins == (1,)
        assert wide.describe()["models"]["narrow"]["resolution_measurable"] is False

        # The same forecasts, binned by rank: the discrimination is there and found.
        assert counted.resolutions[0] > 0.2
        assert counted.rejects
        assert counted.describe()["models"]["narrow"]["resolution_measurable"] is True

    def test_equal_count_bins_do_not_manufacture_discrimination(self) -> None:
        """It must gain power without gaining size, or it is not a robustness run."""
        rng = np.random.default_rng(23)
        rejections = 0
        trials = 100
        for _ in range(trials):
            outcomes = (rng.random(300) < 0.5).astype(float)
            models = {f"m{i}": 0.5 + 0.01 * rng.normal(size=300) for i in range(6)}
            screen = resolution_screen(
                models,
                outcomes,
                event="direction_up",
                sampling=DISJOINT,
                binning="equal_count",
                draws=400,
            )
            assert screen is not None
            rejections += screen.rejects

        assert rejections / trials < 0.15

    def test_an_unknown_binning_is_refused(self) -> None:
        rng = np.random.default_rng(24)
        with pytest.raises(ValueError, match="equal_width or equal_count"):
            resolution_screen(
                {"m": rng.random(100)},
                (rng.random(100) < 0.5).astype(float),
                event="direction_up",
                sampling=DISJOINT,
                binning="deciles",
                draws=100,
            )

    def test_a_non_finite_forecast_stops_the_run(self) -> None:
        """The same guard the CRPS path carries, for the same reason: NaN compares False
        everywhere and would report the most significant result the test can produce."""
        rng = np.random.default_rng(21)
        outcomes = (rng.random(200) < 0.5).astype(float)
        broken = rng.random(200)
        broken[7] = np.nan

        with pytest.raises(NonFiniteLossError):
            resolution_screen(
                {"broken": broken}, outcomes, event="direction_up", sampling=DISJOINT, draws=100
            )


class TestTheDirectionArtifact:
    """The block ``aurex direction`` publishes, built end to end on the engine's own model.

    No heavy challenger runs here — GJR-GARCH and the random walk need nothing beyond the
    default install, and what is being checked is the shape of the artifact and the one
    property the run depends on: that an uncentred forecast actually makes a directional
    claim. If it does not, every number in this artifact is about the drift policy and
    the run has measured nothing.
    """

    def _run(self, demean: bool):  # type: ignore[no-untyped-def]
        from aurex.bench import describe_direction, run_shootout

        return run_shootout(
            SYNTHETIC,
            prices=_rising_prices(),
            request=WalkForwardRequest(horizons=(5, 21), step=5, min_observations=300),
            n_paths=500,
            include=("gjr_garch",),
            demean=demean,
            events=(TerminalAbove(),),
        ), describe_direction

    def test_an_uncentred_run_makes_a_directional_claim_a_centred_one_cannot(self) -> None:
        """The premise of the whole exercise, asserted rather than assumed.

        Centred, P(up) sits at one half whatever the sample did — there is nothing for a
        resolution term to discriminate on and grading direction would grade the policy.
        Uncentred on a rising sample it moves away from one half, which is what makes the
        question answerable either way.
        """
        centred, _ = self._run(demean=True)
        drifted, _ = self._run(demean=False)
        event = TerminalAbove().id

        def mean_probability(run) -> float:  # type: ignore[no-untyped-def]
            records = run.result.for_horizon(21)
            return float(np.mean([r.events[event][0] for r in records]))

        assert mean_probability(centred) == pytest.approx(0.5, abs=0.03)
        assert mean_probability(drifted) > 0.55

    def test_every_model_gets_the_full_row_the_requirement_asks_for(self) -> None:
        run, describe_direction = self._run(demean=False)
        block = describe_direction(SYNTHETIC, run.result, event=TerminalAbove())

        for horizon in block["horizons"]:
            assert horizon["models"]
            for entry in horizon["models"]:
                assert entry["model"]
                assert set(entry["decomposition"]) >= {
                    "reliability",
                    "resolution",
                    "uncertainty",
                }
                # Brier, the base rate, the positive count, and the mean forecast beside
                # the realised rate — the level and the discrimination never summed.
                assert isinstance(entry["brier"], float)
                assert isinstance(entry["base_rate"], float)
                assert isinstance(entry["positive_events"], int)
                assert isinstance(entry["mean_forecast"], float)
                # Each is rounded to four places independently, so the identity holds to
                # within that rather than exactly.
                assert entry["forecast_bias"] == pytest.approx(
                    entry["mean_forecast"] - entry["base_rate"], abs=2e-4
                )

    def test_the_benchmark_is_graded_as_a_competitor_not_only_as_a_null(self) -> None:
        """A drift-matched walk makes a real directional claim, so it has to be scored on
        one. A null that could not lose the comparison it defines is not a competitor."""
        run, describe_direction = self._run(demean=False)
        block = describe_direction(SYNTHETIC, run.result, event=TerminalAbove())

        graded = {entry["model"] for entry in block["horizons"][0]["models"]}
        assert "random_walk_drift_matched" in graded
        assert block["benchmark"]["label"] == "random_walk_drift_matched"

    def test_both_screens_run_and_the_decision_needs_both(self) -> None:
        run, describe_direction = self._run(demean=False)
        block = describe_direction(SYNTHETIC, run.result, event=TerminalAbove())
        by_horizon = {entry["horizon_sessions"]: entry for entry in block["horizons"]}

        # 21 sessions sampled weekly overlaps, so the full-sample run must shift rather
        # than permute; 5 sessions at a step of 5 does not, and permuting is then exact.
        overlapping = by_horizon[21]["discrimination"]
        assert overlapping["full_sample"]["resampling"] == "cyclic_shift"
        assert overlapping["non_overlapping_subsample"] is not None
        assert overlapping["equal_count_bins_robustness"]["binning"] == "equal_count"
        assert isinstance(overlapping["distinguishable_from_zero"], bool)

        # The robustness binning is held to the same two-run standard as the primary one,
        # or the table's only rejection could not be adjudicated by it.
        assert overlapping["equal_count_non_overlapping_subsample"]["binning"] == "equal_count"
        assert isinstance(overlapping["both_screens_reject_equal_count"], bool)

        # The key does not assert a scheme, because 5 sessions at a step of 5 does not
        # overlap and is permuted rather than shifted. The resampling field says which.
        assert by_horizon[5]["discrimination"]["full_sample"]["resampling"] == "permutation"

    def test_the_artifact_states_why_the_run_is_uncentred(self) -> None:
        """A reader must not have to infer the drift policy from the benchmark's label."""
        run, describe_direction = self._run(demean=False)
        block = describe_direction(SYNTHETIC, run.result, event=TerminalAbove())

        assert "drift-matched" in block["conventions"]["drift"]
        assert "Resolution" in block["conventions"]["primary_metric"]
