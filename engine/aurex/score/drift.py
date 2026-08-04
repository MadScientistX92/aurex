"""One mechanism behind several symptoms: a forecast centre that the sample walked away from.

A model whose distribution is centred at ``mu_model`` per session, scored over a sample
that actually drifted at ``mu_sample``, is displaced in standardised units by

.. code-block::

    delta(h) = (mu_sample - mu_model) / sigma * sqrt(h)

because the displacement grows linearly in the horizon while the forecast's own spread
grows as its square root. That single square-root law is what makes this testable rather
than a story: it predicts that a location error is invisible at a week and unmissable at
a quarter, and it predicts the *ratio* between horizons before any of them is measured.

Two of the things that look like separate calibration failures are this one effect seen
from different reference points:

* **Mean PIT above one half**, because the realised price lands in the upper part of a
  distribution whose centre is behind it. Reference point: the forecast's own centre.
* **A direction forecast that scores worse than the base rate**, because "ends higher"
  is measured against *spot*, not against the forecast centre, so it picks up the whole
  sample drift rather than the residual gap.

Breach counts are **not** in that list. Displacement moves the lower tail only through
the same shift, so it predicts a specific and fairly small reduction in breach rate; a
deficit larger than that is a differently-shaped tail, and calling it drift would hide a
second finding behind the first.

**None of this is an argument for fitting a drift.** A fitted mean over a rising sample
is a directional forecast wearing a mean's clothes, and §0 bans it. The point of
measuring the displacement is to name it and publish it, not to tune it away.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy import stats

#: Mean PIT under a displacement of ``delta`` is ``Phi(delta / sqrt(2))``: the realised
#: value is ``N(delta, 1)`` against a standard normal forecast, and integrating the
#: forecast CDF over it convolves the two variances.
_PIT_DELTA_SCALE = float(np.sqrt(2.0))


@dataclass(frozen=True, slots=True)
class HorizonDisplacement:
    """What one horizon contributed to the regression."""

    horizon: int
    mean_pit: float
    observations: int

    def describe(self) -> dict[str, Any]:
        return {
            "horizon_sessions": self.horizon,
            "mean_pit": round(self.mean_pit, 5),
            "observations": self.observations,
        }


@dataclass(frozen=True, slots=True)
class DriftDisplacement:
    """Mean PIT regressed on the square root of the horizon, intercept pinned at one half."""

    slope: float
    r_squared: float
    points: tuple[HorizonDisplacement, ...]

    @property
    def implied_standardised_drift(self) -> float:
        """``(mu_sample - mu_model) / sigma`` per session, inverted from the slope."""
        return self.slope * _PIT_DELTA_SCALE / float(stats.norm.pdf(0.0))

    def predicted_mean_pit(self, horizon: int) -> float:
        delta = self.implied_standardised_drift * float(np.sqrt(horizon))
        return float(stats.norm.cdf(delta / _PIT_DELTA_SCALE))

    def describe(self) -> dict[str, Any]:
        return {
            "model": "mean_pit = 0.5 + slope * sqrt(horizon)",
            "slope": round(self.slope, 6),
            "r_squared": round(self.r_squared, 4),
            "implied_standardised_drift_per_session": round(self.implied_standardised_drift, 5),
            "points": [point.describe() for point in self.points],
            "intercept_note": (
                "The intercept is pinned at 0.5 rather than fitted: a sample with no "
                "displacement must give a mean PIT of one half at every horizon, so a "
                "free intercept would absorb the effect under test."
            ),
            "reading": (
                "A high r-squared means the location error scales as the square root of "
                "the horizon, which is what a constant per-session drift against a "
                "fixed forecast centre looks like. It is a diagnosis, not a correction: "
                "fitting a mean to remove it would be a directional forecast."
            ),
        }


def displacement(points: tuple[HorizonDisplacement, ...]) -> DriftDisplacement | None:
    """Fit the square-root law across horizons, or ``None`` if there are too few.

    Two horizons can always be joined by a one-parameter curve, so the fit is only
    reported from three, where the law can actually fail.
    """
    if len(points) < 3:
        return None

    roots = np.array([np.sqrt(point.horizon) for point in points])
    means = np.array([point.mean_pit for point in points])
    centred = means - 0.5

    slope = float(np.sum(centred * roots) / np.sum(roots**2))
    residual = float(np.sum((centred - slope * roots) ** 2))
    total = float(np.sum(centred**2))

    return DriftDisplacement(
        slope=slope,
        # Against a flat 0.5 rather than against the sample mean: the null being tested
        # is "no displacement at any horizon", not "some constant displacement".
        r_squared=1.0 - residual / total if total > 0.0 else 0.0,
        points=points,
    )
