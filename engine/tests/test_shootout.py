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
import pytest

from aurex.score import Sampling, diebold_mariano
from aurex.score.shootout import (
    NonFiniteLossError,
    minimum_detectable_effect,
    model_confidence_set,
    superior_predictive_ability,
)

DISJOINT = Sampling(horizon=1, step=1)


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
