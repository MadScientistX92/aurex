"""Brier score and the reliability diagram: when it says 20%, does it happen 20%?

The Brier score is the mean squared error of a probability forecast, and Murphy's
decomposition splits it into three parts that answer different questions:

.. code-block::

    BS = reliability - resolution + uncertainty

* **Reliability** is calibration: how far the observed frequency in each bin sits from
  the probability that was forecast. Zero is perfect. This is the number this project
  cares about.
* **Resolution** is usefulness: how far the bins move away from the base rate. A model
  that always forecasts the climatological rate is perfectly reliable and completely
  useless, and only resolution can tell the two apart.
* **Uncertainty** is the base rate's own variance. It belongs to the event, not to the
  forecaster, and no model can change it.

**The decomposition is binned; the score is not.** Reliability and resolution are
computed from binned forecasts, so they only reconstruct the Brier score exactly when
every forecast within a bin is identical. Both numbers are reported — the exact score
and the one the decomposition implies — so the gap between them is visible rather than
absorbed. A wide gap means the bins are too coarse to describe what the model did.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

#: Ten equal-width bins over [0, 1], the conventional reliability diagram.
DEFAULT_BINS = 10


@dataclass(frozen=True, slots=True)
class ReliabilityBin:
    """One bin of the diagram. Empty bins are kept, so the axis is not misread."""

    lower: float
    upper: float
    count: int
    forecast_mean: float | None
    observed_rate: float | None

    def describe(self) -> dict[str, Any]:
        return {
            "lower": round(self.lower, 3),
            "upper": round(self.upper, 3),
            "count": self.count,
            "forecast_mean": None if self.forecast_mean is None else round(self.forecast_mean, 4),
            "observed_rate": None if self.observed_rate is None else round(self.observed_rate, 4),
        }


@dataclass(frozen=True, slots=True)
class ReliabilityCurve:
    """A Brier score, its decomposition, and the diagram behind it."""

    n: int
    base_rate: float
    brier: float
    reliability: float
    resolution: float
    uncertainty: float
    bins: tuple[ReliabilityBin, ...]

    @property
    def brier_from_decomposition(self) -> float:
        return self.reliability - self.resolution + self.uncertainty

    @property
    def binning_error(self) -> float:
        """How much the binned decomposition fails to reconstruct the exact score."""
        return self.brier_from_decomposition - self.brier

    def describe(self) -> dict[str, Any]:
        return {
            "observations": self.n,
            "base_rate": round(self.base_rate, 4),
            "brier": round(self.brier, 5),
            "decomposition": {
                "reliability": round(self.reliability, 5),
                "resolution": round(self.resolution, 5),
                "uncertainty": round(self.uncertainty, 5),
                "implied_brier": round(self.brier_from_decomposition, 5),
                "binning_error": round(self.binning_error, 6),
            },
            "bins": [b.describe() for b in self.bins],
            "reading": (
                "Reliability is the calibration term and lower is better; resolution "
                "is how far the forecasts move from the base rate and higher is "
                "better. A model can be perfectly reliable and useless. The "
                "decomposition is binned, so it reconstructs the exact Brier score "
                "only up to the binning error reported beside it."
            ),
        }


def brier_score(probabilities: np.ndarray, outcomes: np.ndarray) -> float:
    """Mean squared error of a probability forecast against a binary outcome."""
    forecast = np.asarray(probabilities, dtype=float)
    observed = np.asarray(outcomes, dtype=float)
    if forecast.shape != observed.shape:
        raise ValueError(f"shape mismatch: {forecast.shape} forecasts, {observed.shape} outcomes")
    if forecast.size == 0:
        raise ValueError("no observations to score")
    if np.any((forecast < 0.0) | (forecast > 1.0)):
        raise ValueError("probabilities must lie in [0, 1]")
    return float(np.mean((forecast - observed) ** 2))


def reliability_curve(
    probabilities: np.ndarray, outcomes: np.ndarray, *, bins: int = DEFAULT_BINS
) -> ReliabilityCurve:
    """Bin the forecasts, compare each bin against what happened, decompose the score."""
    forecast = np.asarray(probabilities, dtype=float)
    observed = np.asarray(outcomes, dtype=float)
    score = brier_score(forecast, observed)

    n = int(forecast.size)
    base_rate = float(np.mean(observed))
    edges = np.linspace(0.0, 1.0, bins + 1)
    # Right-closed at the top so a forecast of exactly 1.0 lands in the last bin
    # rather than in a bin of its own beyond the axis.
    assignment = np.clip(np.digitize(forecast, edges[1:-1], right=False), 0, bins - 1)

    entries: list[ReliabilityBin] = []
    reliability = 0.0
    resolution = 0.0

    for index in range(bins):
        members = assignment == index
        count = int(np.count_nonzero(members))
        if count == 0:
            entries.append(
                ReliabilityBin(
                    lower=float(edges[index]),
                    upper=float(edges[index + 1]),
                    count=0,
                    forecast_mean=None,
                    observed_rate=None,
                )
            )
            continue

        bin_forecast = float(np.mean(forecast[members]))
        bin_observed = float(np.mean(observed[members]))
        reliability += count * (bin_forecast - bin_observed) ** 2
        resolution += count * (bin_observed - base_rate) ** 2
        entries.append(
            ReliabilityBin(
                lower=float(edges[index]),
                upper=float(edges[index + 1]),
                count=count,
                forecast_mean=bin_forecast,
                observed_rate=bin_observed,
            )
        )

    return ReliabilityCurve(
        n=n,
        base_rate=base_rate,
        brier=score,
        reliability=reliability / n,
        resolution=resolution / n,
        uncertainty=base_rate * (1.0 - base_rate),
        bins=tuple(entries),
    )
