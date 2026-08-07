"""Tests for the rare-event count test.

Three layers, matching the rest of the scoring suite:

* **Closed forms.** The Poisson-binomial against :func:`scipy.stats.binom` where every
  probability is equal, and the driftless Gaussian against the arithmetic the README
  published. If either drifts, the tail comparison is measuring something else.
* **Boundaries.** A path-monitored event gets no Gaussian reference, an empty series is
  undefined rather than zero, and the lookahead label travels with the reference that
  earns it.
* **The published claim.** Five events in 579 windows, tested against the two rates the
  README named. This is the one test in the file that exists to grade a sentence in a
  document rather than a function.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy import stats

from aurex.score import (
    ClearsHurdle,
    Sampling,
    TouchAbove,
    driftless_gaussian_probability,
    poisson_binomial_pmf,
    tail_calibration,
    tail_test,
)
from aurex.score.tail import LOOKAHEAD_FREE


class TestPoissonBinomial:
    def test_pmf_is_a_distribution(self) -> None:
        rng = np.random.default_rng(11)
        probabilities = rng.uniform(0.0, 0.3, size=200)
        pmf = poisson_binomial_pmf(probabilities)

        assert pmf.size == 201
        assert pmf.sum() == pytest.approx(1.0)
        assert np.all(pmf >= 0.0)

    def test_equal_probabilities_reduce_to_the_binomial(self) -> None:
        """The one case with a closed form, so it is the one that pins the convolution."""
        n, p = 40, 0.07
        pmf = poisson_binomial_pmf(np.full(n, p))

        expected = stats.binom.pmf(np.arange(n + 1), n, p)
        assert np.allclose(pmf, expected)

    def test_mean_is_the_sum_of_the_probabilities(self) -> None:
        rng = np.random.default_rng(3)
        probabilities = rng.uniform(0.0, 1.0, size=50)
        pmf = poisson_binomial_pmf(probabilities)

        mean = float(np.dot(np.arange(pmf.size), pmf))
        assert mean == pytest.approx(float(probabilities.sum()))

    def test_degenerate_probabilities_are_exact(self) -> None:
        pmf = poisson_binomial_pmf(np.array([1.0, 1.0, 0.0]))
        assert pmf[2] == pytest.approx(1.0)

    def test_a_probability_outside_the_unit_interval_is_refused(self) -> None:
        with pytest.raises(ValueError, match="must lie in"):
            poisson_binomial_pmf(np.array([0.5, 1.4]))


class TestTailTest:
    def test_tails_partition_the_distribution(self) -> None:
        rng = np.random.default_rng(5)
        probabilities = rng.uniform(0.0, 0.2, size=120)
        outcomes = (rng.uniform(size=120) < probabilities).astype(float)

        result = tail_test(probabilities, outcomes, reference="model", sample="all_windows")
        pmf = poisson_binomial_pmf(probabilities)

        assert result.p_value_upper is not None
        assert result.p_value_lower is not None
        # P(X >= k) + P(X <= k) counts P(X = k) twice, which is the definition rather
        # than an error, and checking it is how the two stay consistent.
        assert result.p_value_upper + result.p_value_lower - pmf[result.observed] == pytest.approx(
            1.0
        )

    def test_the_poisson_approximation_tracks_the_exact_value(self) -> None:
        probabilities = np.full(579, 0.0023)
        outcomes = np.zeros(579)
        outcomes[:5] = 1.0

        result = tail_test(probabilities, outcomes, reference="model", sample="all_windows")

        assert result.p_value_upper is not None
        assert result.poisson_upper is not None
        assert result.poisson_upper == pytest.approx(result.p_value_upper, abs=5e-4)

    def test_an_empty_series_is_undefined_rather_than_certain(self) -> None:
        result = tail_test(np.zeros(0), np.zeros(0), reference="model", sample="non_overlapping")

        assert result.p_value_upper is None
        assert result.undefined_reason is not None
        assert not result.rejects

    def test_misaligned_inputs_are_refused(self) -> None:
        with pytest.raises(ValueError, match="must align"):
            tail_test(np.zeros(3), np.zeros(4), reference="model", sample="all_windows")

    def test_the_lookahead_label_travels_with_the_reference(self) -> None:
        """Not a comment: a reader has to be able to see which benchmark had help."""
        probabilities = np.full(20, 0.1)
        outcomes = np.zeros(20)

        assert not tail_test(
            probabilities, outcomes, reference="gaussian_expanding_sigma", sample="all_windows"
        ).uses_lookahead
        assert tail_test(
            probabilities, outcomes, reference="gaussian_sample_sigma", sample="all_windows"
        ).uses_lookahead

    def test_every_declared_reference_has_a_lookahead_verdict(self) -> None:
        assert set(LOOKAHEAD_FREE) == {
            "model",
            "gaussian_matched_sigma",
            "gaussian_expanding_sigma",
            "gaussian_sample_sigma",
        }


class TestDriftlessGaussian:
    def test_it_reproduces_the_published_arithmetic(self) -> None:
        """`ln(1.0937) / (0.00999 * sqrt(10))` is 2.83 sigma, and 2.83 sigma is 0.23%."""
        probability = driftless_gaussian_probability(
            np.array([1.0937113]), np.array([0.00999]), horizon=10
        )

        assert float(probability[0]) == pytest.approx(0.0023, abs=1e-4)

    def test_a_hurdle_of_zero_is_a_coin_flip(self) -> None:
        probability = driftless_gaussian_probability(np.array([1.0]), np.array([0.01]), horizon=5)

        assert float(probability[0]) == pytest.approx(0.5)

    def test_a_longer_horizon_makes_an_upside_hurdle_likelier(self) -> None:
        near = driftless_gaussian_probability(np.array([1.05]), np.array([0.01]), horizon=5)
        far = driftless_gaussian_probability(np.array([1.05]), np.array([0.01]), horizon=63)

        assert float(far[0]) > float(near[0])

    def test_zero_volatility_is_not_scored_as_a_probability(self) -> None:
        probability = driftless_gaussian_probability(np.array([1.05]), np.array([0.0]), horizon=5)

        assert np.isnan(probability[0])


class TestTailCalibration:
    def test_a_path_event_gets_no_gaussian_reference(self) -> None:
        """No closed form matches session-close monitoring, so none is offered."""
        event = TouchAbove(0.05)
        assert event.monitoring == "session_close"

        result = tail_calibration(
            event_id=event.id,
            probabilities=np.full(50, 0.1),
            outcomes=np.zeros(50),
            sampling=Sampling(horizon=5, step=5),
            gaussian_references={},
        )

        assert {test.reference for test in result.tests} == {"model"}

    def test_both_samples_are_always_published(self) -> None:
        result = tail_calibration(
            event_id=ClearsHurdle(1.09).id,
            probabilities=np.full(100, 0.05),
            outcomes=np.zeros(100),
            sampling=Sampling(horizon=10, step=5),
            gaussian_references={"gaussian_sample_sigma": np.full(100, 0.02)},
        )

        samples = {(test.reference, test.sample) for test in result.tests}
        assert samples == {
            ("model", "all_windows"),
            ("model", "non_overlapping"),
            ("gaussian_sample_sigma", "all_windows"),
            ("gaussian_sample_sigma", "non_overlapping"),
        }

    def test_the_thinned_sample_is_smaller(self) -> None:
        sampling = Sampling(horizon=10, step=5)
        result = tail_calibration(
            event_id="e",
            probabilities=np.full(100, 0.05),
            outcomes=np.zeros(100),
            sampling=sampling,
            gaussian_references=None,
        )

        full = result.for_reference("model", "all_windows")
        thinned = result.for_reference("model", "non_overlapping")
        assert full is not None and thinned is not None
        assert full.n == 100
        assert thinned.n == 50

    def test_a_reference_with_a_hole_is_dropped_rather_than_scored_short(self) -> None:
        """A reference computed on fewer windows is not comparable with the model's."""
        broken = np.full(50, 0.02)
        broken[7] = np.nan

        result = tail_calibration(
            event_id="e",
            probabilities=np.full(50, 0.05),
            outcomes=np.zeros(50),
            sampling=Sampling(horizon=5, step=5),
            gaussian_references={"gaussian_expanding_sigma": broken},
        )

        assert {test.reference for test in result.tests} == {"model"}


class TestThePublishedTailClaim:
    """The README says a Gaussian forecast 0.23%, Aurex 0.79%, and it happened at 0.86%.

    Five events in 579 windows. Whether that count is surprising depends entirely on
    which of the two forecasts you hold it against, and until this test existed the
    document asserted the comparison without grading it.
    """

    OBSERVED = 5
    WINDOWS = 579

    def _outcomes(self) -> np.ndarray:
        outcomes = np.zeros(self.WINDOWS)
        outcomes[: self.OBSERVED] = 1.0
        return outcomes

    def test_the_gaussian_rate_is_rejected(self) -> None:
        result = tail_test(
            np.full(self.WINDOWS, 0.0023),
            self._outcomes(),
            reference="gaussian_sample_sigma",
            sample="all_windows",
        )

        assert result.expected == pytest.approx(1.33, abs=0.01)
        assert result.p_value_upper is not None
        assert result.p_value_upper == pytest.approx(0.0116, abs=1e-3)
        assert result.rejects

    def test_the_aurex_rate_is_not(self) -> None:
        result = tail_test(
            np.full(self.WINDOWS, 0.0079),
            self._outcomes(),
            reference="model",
            sample="all_windows",
        )

        assert result.expected == pytest.approx(4.57, abs=0.01)
        assert result.p_value_upper is not None
        assert result.p_value_upper > 0.4
        assert not result.rejects

    def test_thinning_costs_the_rejection(self) -> None:
        """Half the windows, and the same five events cannot be assumed to survive.

        Thinning takes every second record, so a count of five over the full sample is
        not five over the subsample. What this test pins is the weaker, structural
        claim: on 290 windows the Gaussian rate expects 0.67 events, so the evidence
        against it rests on how many of the five land in the retained half — which is
        why both samples are published rather than one.
        """
        sampling = Sampling(horizon=10, step=5)
        thinned = sampling.thin(np.full(self.WINDOWS, 0.0023))

        assert thinned.size == 290
        assert float(thinned.sum()) == pytest.approx(0.667, abs=0.01)
