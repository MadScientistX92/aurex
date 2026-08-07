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

**A rare event gets a score but not a diagram.** The Brier score, the base rate and the
positive count are meaningful at any sample size. A ten-bin reliability curve is not:
below a handful of positives the observed rate in each bin is one or two events, and
drawing that produces a picture of sampling noise with an axis on it. So the curve is
*withheld* below :data:`MIN_POSITIVE_EVENTS` positives and the artifact says why, while
the score, base rate and count are published as usual.

This matters most for the event friction defines — clearing a breakeven hurdle is a
two-sigma move in one direction, so at retail friction it happens on the order of
fifteen times in eleven years — but the rule is not special-cased to it. Every rare
event has the same problem, including a touch of a distant barrier, and a threshold that
applied to one event and not another would be a threshold chosen after seeing which
results it flattered.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

#: Ten equal-width bins over [0, 1], the conventional reliability diagram.
DEFAULT_BINS = 10

#: Positives required before a reliability diagram is drawn at all. Ten, matching the
#: rule of thumb :data:`aurex.score.coverage.MIN_EXPECTED_BREACHES` already uses for
#: breach counts — taken from the neighbouring test rather than tuned to what any
#: particular event happens to produce.
MIN_POSITIVE_EVENTS = 10


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
    #: How many times the event actually happened. Published beside every score,
    #: because a Brier score of 0.02 on a rare event looks excellent next to one of
    #: 0.25 on a coin flip and is measuring something else entirely.
    positives: int = 0
    #: Mean forecast probability, published whether or not the curve is drawn.
    #:
    #: This is the number that answers "is the modelled tail the right shape" — it sits
    #: directly against ``base_rate``, and the gap between them is the model's bias on
    #: this event. It used to travel only inside the bins, so withholding the diagram
    #: withheld it too. That was wrong: a scalar is not a diagram, and the argument for
    #: withholding a ten-bin curve at five positives says nothing about a single mean
    #: over every forecast. Rare events are precisely where the comparison matters most.
    mean_forecast: float = 0.0
    #: Why the diagram is not here, when it is not. ``None`` when the curve is drawn.
    withheld_reason: str | None = None

    @property
    def brier_from_decomposition(self) -> float:
        return self.reliability - self.resolution + self.uncertainty

    @property
    def binning_error(self) -> float:
        """How much the binned decomposition fails to reconstruct the exact score."""
        return self.brier_from_decomposition - self.brier

    @property
    def withheld(self) -> bool:
        return self.withheld_reason is not None

    def describe(self) -> dict[str, Any]:
        return {
            "observations": self.n,
            "positive_events": self.positives,
            "base_rate": round(self.base_rate, 4),
            "mean_forecast": round(self.mean_forecast, 4),
            "forecast_bias": round(self.mean_forecast - self.base_rate, 4),
            "brier": round(self.brier, 5),
            "decomposition": {
                "reliability": round(self.reliability, 5),
                "resolution": round(self.resolution, 5),
                "uncertainty": round(self.uncertainty, 5),
                "implied_brier": round(self.brier_from_decomposition, 5),
                "binning_error": round(self.binning_error, 6),
            },
            "bins": [b.describe() for b in self.bins],
            "curve_withheld": self.withheld_reason,
            "reading": (
                "Reliability is the calibration term and lower is better; resolution "
                "is how far the forecasts move from the base rate and higher is "
                "better. A model can be perfectly reliable and useless. The "
                "decomposition is binned, so it reconstructs the exact Brier score "
                "only up to the binning error reported beside it. Read the Brier score "
                "against the uncertainty term rather than against another event's "
                "score: a rare event has a low uncertainty and therefore a low "
                "achievable Brier, and comparing across base rates measures the base "
                "rates. mean_forecast against base_rate is the model's bias on this "
                "event and is published whether or not the curve is drawn: a scalar is "
                "not a diagram, and on a rare event it is the most informative number "
                "here."
            ),
        }


def bin_assignment(probabilities: np.ndarray, *, bins: int = DEFAULT_BINS) -> np.ndarray:
    """Which bin each forecast falls in. The decomposition's only free choice, in one place.

    Exposed rather than inlined because the resolution term is now read twice: once here,
    to publish it, and once by :func:`~aurex.score.shootout.discrimination` to build its
    null distribution. Two implementations of "which bin" would let a published
    resolution and the p-value that grades it come from different binnings, and the gap
    would be invisible — both numbers would look reasonable and they would not be about
    the same quantity.
    """
    edges = np.linspace(0.0, 1.0, bins + 1)
    # Right-closed at the top so a forecast of exactly 1.0 lands in the last bin
    # rather than in a bin of its own beyond the axis.
    assigned: np.ndarray = np.clip(
        np.digitize(np.asarray(probabilities, dtype=float), edges[1:-1], right=False), 0, bins - 1
    )
    return assigned


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
    probabilities: np.ndarray,
    outcomes: np.ndarray,
    *,
    bins: int = DEFAULT_BINS,
    min_positives: int = MIN_POSITIVE_EVENTS,
) -> ReliabilityCurve:
    """Bin the forecasts, compare each bin against what happened, decompose the score.

    Below ``min_positives`` the diagram is withheld and the reason travels with the
    result. Everything that does not need the bins — the score, the base rate, the
    count, the decomposition — is still computed and returned.
    """
    forecast = np.asarray(probabilities, dtype=float)
    observed = np.asarray(outcomes, dtype=float)
    score = brier_score(forecast, observed)

    n = int(forecast.size)
    positives = int(np.count_nonzero(observed > 0.0))
    base_rate = float(np.mean(observed))
    edges = np.linspace(0.0, 1.0, bins + 1)
    assignment = bin_assignment(forecast, bins=bins)

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

    withheld = None
    if positives < min_positives:
        withheld = (
            f"{positives} positive events in {n} forecasts is too few to draw a "
            f"{bins}-bin reliability diagram: each bin's observed rate would rest on "
            f"one or two events. The threshold is {min_positives}. The Brier score, "
            f"base rate and count above are unaffected and are the numbers to read."
        )
        entries = []

    return ReliabilityCurve(
        n=n,
        base_rate=base_rate,
        brier=score,
        reliability=reliability / n,
        resolution=resolution / n,
        uncertainty=base_rate * (1.0 - base_rate),
        bins=tuple(entries),
        positives=positives,
        # Computed over every forecast, not over the surviving bins, so it stays
        # meaningful on exactly the runs where the diagram is withheld.
        mean_forecast=float(np.mean(forecast)),
        withheld_reason=withheld,
    )
