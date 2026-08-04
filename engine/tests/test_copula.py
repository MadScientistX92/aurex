"""The bivariate t-copula, and what it is for.

The test that carries the argument is
:meth:`TestTailDependence.test_it_separates_joint_tails_from_correlation`: two samples
are built with the *same* rank correlation, one Gaussian and one t, and the fit has to
tell them apart. A Gaussian copula cannot — that is precisely why the family is
parametric here rather than assumed.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from aurex.dist import (
    ResidualPool,
    TCopula,
    fit_t_copula,
    joint_shocks,
    pseudo_observations,
    residual_pool,
)

INDEX = pd.bdate_range("2018-01-01", periods=2_000, name="date")


def gaussian_pair(correlation: float = 0.6, seed: int = 1) -> tuple[pd.Series, pd.Series]:
    rng = np.random.default_rng(seed)
    first = rng.standard_normal(len(INDEX))
    second = correlation * first + np.sqrt(1.0 - correlation**2) * rng.standard_normal(len(INDEX))
    return pd.Series(first, index=INDEX), pd.Series(second, index=INDEX)


def t_pair(correlation: float = 0.6, df: float = 3.0, seed: int = 2) -> tuple[pd.Series, pd.Series]:
    """A genuine bivariate t: one shared scale factor drives both margins."""
    rng = np.random.default_rng(seed)
    covariance = np.array([[1.0, correlation], [correlation, 1.0]])
    normal = rng.multivariate_normal(np.zeros(2), covariance, size=len(INDEX))
    scale = np.sqrt(df / rng.chisquare(df, size=len(INDEX)))
    variates = normal * scale[:, np.newaxis]
    return (
        pd.Series(variates[:, 0], index=INDEX),
        pd.Series(variates[:, 1], index=INDEX),
    )


class TestFitting:
    def test_rho_recovers_the_rank_correlation(self) -> None:
        """Through Kendall's tau, so the estimate does not care about the margins."""
        copula = fit_t_copula(*gaussian_pair(correlation=0.6))
        assert copula.rho == pytest.approx(0.6, abs=0.06)

    def test_a_negative_relationship_is_recovered_too(self) -> None:
        copula = fit_t_copula(*gaussian_pair(correlation=-0.4))
        assert copula.rho == pytest.approx(-0.4, abs=0.06)

    def test_monotone_transforms_of_the_margins_do_not_move_it(self) -> None:
        first, second = gaussian_pair(correlation=0.5)
        plain = fit_t_copula(first, second)
        transformed = fit_t_copula(np.exp(first), second * 1_000.0)

        assert transformed.rho == pytest.approx(plain.rho, abs=1e-9)

    def test_only_overlapping_dates_are_used(self) -> None:
        first, second = gaussian_pair()
        copula = fit_t_copula(first, second.iloc[:1_200])
        assert copula.n_observations == 1_200

    def test_too_little_overlap_is_refused(self) -> None:
        first, second = gaussian_pair()
        with pytest.raises(ValueError, match="says nothing"):
            fit_t_copula(first.iloc[:20], second.iloc[:20])


class TestTailDependence:
    def test_it_separates_joint_tails_from_correlation(self) -> None:
        """Same rank correlation, different joint behaviour — the fit must notice."""
        gaussian = fit_t_copula(*gaussian_pair(correlation=0.6))
        heavy = fit_t_copula(*t_pair(correlation=0.6, df=3.0))

        assert gaussian.kendall_tau == pytest.approx(heavy.kendall_tau, abs=0.06)
        assert heavy.df < gaussian.df
        assert heavy.tail_dependence > 5.0 * gaussian.tail_dependence

    def test_a_gaussian_pair_reports_effectively_no_tail_dependence(self) -> None:
        copula = fit_t_copula(*gaussian_pair(correlation=0.6))
        assert copula.tail_dependence < 0.05

    def test_a_heavy_tailed_pair_reports_a_material_one(self) -> None:
        copula = fit_t_copula(*t_pair(df=3.0))
        assert copula.tail_dependence > 0.15

    @pytest.mark.parametrize("df", [60.0, 59.9994])
    def test_the_degrees_of_freedom_ceiling_is_flagged(self, df: float) -> None:
        """A bounded optimiser stops just short of its ceiling.

        Testing only the exact bound let this flag read ``false`` at 59.9994 on a
        real run — a fit with no evidence of tail dependence quietly presenting as
        one that had found some.
        """
        copula = TCopula(rho=0.5, df=df, kendall_tau=0.33, n_observations=100, log_likelihood=0.0)
        assert copula.df_at_bound
        assert copula.describe()["df_at_bound"] is True

    def test_a_genuinely_heavy_fit_is_not_flagged(self) -> None:
        assert not fit_t_copula(*t_pair(df=3.0)).df_at_bound


class TestSampling:
    def test_margins_are_uniform(self) -> None:
        copula = fit_t_copula(*t_pair(df=4.0))
        drawn = copula.sample(50_000, np.random.default_rng(3))

        assert drawn.shape == (50_000, 2)
        assert drawn.min() > 0.0 and drawn.max() < 1.0
        for column in range(2):
            assert drawn[:, column].mean() == pytest.approx(0.5, abs=0.01)
            assert np.quantile(drawn[:, column], 0.1) == pytest.approx(0.1, abs=0.01)

    def test_the_dependence_survives_the_round_trip(self) -> None:
        copula = fit_t_copula(*gaussian_pair(correlation=0.7))
        drawn = copula.sample(50_000, np.random.default_rng(4))
        from scipy import stats

        tau = stats.kendalltau(drawn[:, 0], drawn[:, 1]).statistic
        assert tau == pytest.approx(copula.kendall_tau, abs=0.03)

    def test_sampling_is_reproducible(self) -> None:
        copula = fit_t_copula(*gaussian_pair())
        first = copula.sample(100, np.random.default_rng(5))
        second = copula.sample(100, np.random.default_rng(5))
        np.testing.assert_array_equal(first, second)


class TestJointShocks:
    def _residuals(self) -> tuple[pd.Series, pd.Series]:
        return t_pair(correlation=0.6, df=4.0, seed=6)

    def _pools(self) -> tuple[ResidualPool, ResidualPool]:
        """Undemeaned, so the drawn values stay identical to the sample's own."""
        first, second = self._residuals()
        return residual_pool(first, demean=False), residual_pool(second, demean=False)

    def test_copula_mode_keeps_each_margin_empirical(self) -> None:
        """The copula supplies dependence; it must not reshape a marginal tail."""
        first, second = self._residuals()
        copula = fit_t_copula(first, second)
        left_pool, right_pool = self._pools()
        drawn_first, _ = joint_shocks(
            left_pool,
            right_pool,
            copula=copula,
            mode="t_copula",
            n_paths=2_000,
            horizon=10,
            block_length=5,
            rng=np.random.default_rng(7),
        )
        assert set(np.unique(drawn_first)) <= set(first.to_numpy())

    def test_copula_mode_reproduces_the_dependence(self) -> None:
        first, second = self._residuals()
        copula = fit_t_copula(first, second)
        left_pool, right_pool = self._pools()
        left, right = joint_shocks(
            left_pool,
            right_pool,
            copula=copula,
            mode="t_copula",
            n_paths=4_000,
            horizon=10,
            block_length=5,
            rng=np.random.default_rng(8),
        )
        observed = np.corrcoef(left.ravel(), right.ravel())[0, 1]
        assert observed == pytest.approx(np.corrcoef(first, second)[0, 1], abs=0.1)

    def test_synchronised_mode_preserves_the_sample_dependence(self) -> None:
        first, second = self._residuals()
        left_pool, right_pool = self._pools()
        left, right = joint_shocks(
            left_pool,
            right_pool,
            copula=None,
            mode="synchronised",
            n_paths=4_000,
            horizon=10,
            block_length=5,
            rng=np.random.default_rng(9),
        )
        observed = np.corrcoef(left.ravel(), right.ravel())[0, 1]
        assert observed == pytest.approx(np.corrcoef(first, second)[0, 1], abs=0.05)

    def test_synchronised_mode_draws_the_same_dates_from_both_series(self) -> None:
        """Which is why it cannot invent tail co-movement the sample never had."""
        first = pd.Series(np.arange(100.0), index=pd.bdate_range("2020-01-01", periods=100))
        second = first * -1.0
        left, right = joint_shocks(
            residual_pool(first, demean=False),
            residual_pool(second, demean=False),
            copula=None,
            mode="synchronised",
            n_paths=50,
            horizon=6,
            block_length=3,
            rng=np.random.default_rng(10),
        )
        np.testing.assert_allclose(left, -right)

    def test_copula_mode_without_a_copula_is_refused(self) -> None:
        left_pool, right_pool = self._pools()
        with pytest.raises(ValueError, match="needs a fitted copula"):
            joint_shocks(
                left_pool,
                right_pool,
                copula=None,
                mode="t_copula",
                n_paths=10,
                horizon=2,
                block_length=2,
                rng=np.random.default_rng(11),
            )

    def test_series_that_never_overlap_are_refused(self) -> None:
        first = pd.Series([1.0, 2.0], index=pd.bdate_range("2020-01-01", periods=2))
        second = pd.Series([1.0, 2.0], index=pd.bdate_range("2021-01-01", periods=2))
        with pytest.raises(ValueError, match="do not overlap"):
            joint_shocks(
                residual_pool(first, demean=False),
                residual_pool(second, demean=False),
                copula=None,
                mode="synchronised",
                n_paths=5,
                horizon=2,
                block_length=2,
                rng=np.random.default_rng(12),
            )

    def test_two_legs_with_different_drift_policies_are_refused(self) -> None:
        """A driftless metal composed with a drifting rate is a currency call in disguise."""
        first, second = self._residuals()
        with pytest.raises(ValueError, match="same drift policy"):
            joint_shocks(
                residual_pool(first, demean=True),
                residual_pool(second, demean=False),
                copula=None,
                mode="synchronised",
                n_paths=5,
                horizon=2,
                block_length=2,
                rng=np.random.default_rng(13),
            )


class TestPseudoObservations:
    def test_they_stay_strictly_inside_the_unit_interval(self) -> None:
        """At exactly zero or one the inverse t is infinite and the fit dies."""
        u = pseudo_observations(np.array([3.0, 1.0, 2.0, 5.0]))
        assert u.min() > 0.0 and u.max() < 1.0

    def test_they_are_ranks(self) -> None:
        u = pseudo_observations(np.array([10.0, 20.0, 30.0]))
        np.testing.assert_allclose(u, [0.25, 0.5, 0.75])
