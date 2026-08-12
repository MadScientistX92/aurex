"""The estimator is hand-written, so it is checked against closed forms, not itself.

Three properties a correct coordinate descent cannot avoid agreeing with, and one that
catches the mistake this implementation is most likely to make.
"""

from __future__ import annotations

import itertools

import numpy as np
import pytest

from aurex.factors import elasticnet


@pytest.fixture
def sample() -> tuple[np.ndarray, np.ndarray]:
    """A design with correlated columns and units that differ by three orders."""
    rng = np.random.default_rng(11)
    n = 400
    base = rng.normal(size=n)
    design = np.column_stack(
        [
            base + rng.normal(scale=0.5, size=n),
            base * 1000.0 + rng.normal(scale=500.0, size=n),
            rng.normal(size=n),
            rng.normal(scale=0.01, size=n),
        ]
    )
    truth = np.array([2.0, 0.0, -1.0, 0.0])
    target = design @ (truth / np.array([1.0, 1000.0, 1.0, 0.01])) + rng.normal(scale=0.5, size=n)
    return design, target


class TestAgainstClosedForms:
    def test_zero_penalty_is_ordinary_least_squares(
        self, sample: tuple[np.ndarray, np.ndarray]
    ) -> None:
        design, target = sample
        with_intercept = np.column_stack([np.ones(len(design)), design])
        expected, *_ = np.linalg.lstsq(with_intercept, target, rcond=None)

        fitted = elasticnet.fit(design, target, lam=0.0, l1_ratio=0.5)

        assert fitted.intercept == pytest.approx(expected[0], rel=1e-5)
        assert fitted.raw_coefficients == pytest.approx(expected[1:], rel=1e-4)

    def test_orthonormal_lasso_is_soft_thresholding(self) -> None:
        """The one design where the answer is known in closed form for every penalty."""
        rng = np.random.default_rng(3)
        n, p = 200, 5
        # Centred *before* the factorisation, so the orthonormal columns are also
        # orthogonal to the intercept. Standardising an uncentred orthonormal design
        # re-centres each column and breaks the very orthogonality the closed form
        # needs — which showed up as a 0.3% disagreement that was the test's fault.
        raw = rng.normal(size=(n, p))
        design = np.linalg.qr(raw - raw.mean(axis=0))[0]
        target = design @ np.array([3.0, -2.0, 0.4, 0.0, 1.0]) + rng.normal(scale=0.1, size=n)

        standardised, centred, _ = elasticnet.standardise(design, target)
        ols = standardised.T @ centred / n
        lam = 0.5

        fitted = elasticnet.fit(design, target, lam=lam, l1_ratio=1.0)

        expected = np.sign(ols) * np.maximum(np.abs(ols) - lam, 0.0)
        assert fitted.coefficients == pytest.approx(expected, abs=1e-6)

    def test_the_top_of_the_path_zeroes_everything(
        self, sample: tuple[np.ndarray, np.ndarray]
    ) -> None:
        design, target = sample
        standardised, centred, _ = elasticnet.standardise(design, target)
        top = elasticnet.max_penalty(standardised, centred, l1_ratio=1.0)

        assert not elasticnet.fit(design, target, lam=top * 1.001, l1_ratio=1.0).support.any()
        assert elasticnet.fit(design, target, lam=top * 0.999, l1_ratio=1.0).support.any(), (
            "the boundary is not the boundary if a hair below it also selects nothing"
        )

    def test_selection_is_monotone_down_the_path(
        self, sample: tuple[np.ndarray, np.ndarray]
    ) -> None:
        design, target = sample
        penalties = elasticnet.penalty_path(design, target, l1_ratio=1.0, steps=30)
        counts = [
            int(np.count_nonzero(f.support))
            for f in elasticnet.fit_path(design, target, penalties, l1_ratio=1.0)
        ]
        assert counts[0] == 0
        assert counts[-1] >= counts[0]
        assert all(b >= a - 1 for a, b in itertools.pairwise(counts)), counts


class TestStandardisation:
    def test_units_do_not_decide_which_driver_survives(self) -> None:
        """The mistake this implementation is most likely to make.

        Column two carries the same information as column one scaled by a thousand. If
        the penalty were applied to raw coefficients, the rescaled column would be
        shrunk a thousand times less and would win selection on its units alone. The two
        fits must agree on the standardised scale.
        """
        rng = np.random.default_rng(5)
        n = 300
        x = rng.normal(size=(n, 2))
        target = x @ np.array([1.0, -0.5]) + rng.normal(scale=0.3, size=n)

        rescaled = x * np.array([1.0, 1000.0])
        plain = elasticnet.fit(x, target, lam=0.05, l1_ratio=0.5)
        blown_up = elasticnet.fit(rescaled, target, lam=0.05, l1_ratio=0.5)

        assert blown_up.coefficients == pytest.approx(plain.coefficients, abs=1e-8)
        assert blown_up.raw_coefficients[1] == pytest.approx(
            plain.raw_coefficients[1] / 1000.0, rel=1e-6
        )

    def test_a_constant_regressor_is_left_at_zero(self) -> None:
        """It carries no information; an epsilon divisor would make it carry noise."""
        rng = np.random.default_rng(7)
        design = np.column_stack([rng.normal(size=200), np.full(200, 4.0)])
        target = design[:, 0] * 2.0 + rng.normal(scale=0.1, size=200)

        fitted = elasticnet.fit(design, target, lam=0.01, l1_ratio=0.5)

        assert fitted.coefficients[1] == 0.0
        assert np.isfinite(fitted.raw_coefficients).all()

    def test_predictions_use_the_natural_scale(self, sample: tuple[np.ndarray, np.ndarray]) -> None:
        design, target = sample
        fitted = elasticnet.fit(design, target, lam=0.01, l1_ratio=0.5)
        assert fitted.predict(design) == pytest.approx(
            fitted.intercept + design @ fitted.raw_coefficients
        )


class TestRefusals:
    def test_mismatched_shapes_raise(self) -> None:
        with pytest.raises(ValueError, match="rows"):
            elasticnet.fit(np.zeros((10, 2)), np.zeros(9), lam=0.1)

    @pytest.mark.parametrize("l1_ratio", [-0.1, 1.5])
    def test_an_l1_ratio_outside_the_interval_raises(self, l1_ratio: float) -> None:
        with pytest.raises(ValueError, match="l1_ratio"):
            elasticnet.fit(np.zeros((10, 2)), np.zeros(10), lam=0.1, l1_ratio=l1_ratio)

    def test_a_negative_penalty_raises(self) -> None:
        with pytest.raises(ValueError, match="non-negative"):
            elasticnet.fit(np.zeros((10, 2)), np.zeros(10), lam=-1.0)

    def test_ridge_only_has_no_vanishing_penalty(self) -> None:
        with pytest.raises(ValueError, match="ridge"):
            elasticnet.max_penalty(np.zeros((10, 2)), np.zeros(10), l1_ratio=0.0)
