"""How often the walk-forward stepped, and what that does to a p-value.

A forecast made every ``step`` sessions for a horizon longer than ``step`` shares most
of its path with the forecast before it. That is harmless for a histogram or a mean —
each PIT value and each CRPS is still individually valid — and fatal for a test
statistic, because every p-value in this package comes from a likelihood that assumes
independent observations. An overlapping series does not carry the number of
observations it appears to carry, and a breach-independence test run on one is
measuring the sampling scheme rather than the model.

So overlap is a *declared property* of a scored series rather than something the
caller is trusted to remember: the functions that produce p-values take a
:class:`Sampling` and refuse an overlapping one at the boundary. Thinning to the
independent subsample is a method here rather than a convention at each call site,
so every test in this package thins the same way.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np


class OverlappingWindowsError(ValueError):
    """A test that assumes independent observations was handed overlapping ones."""


@dataclass(frozen=True, slots=True)
class Sampling:
    """The shape of a scored series: how long each forecast ran, how often one started."""

    horizon: int
    step: int

    def __post_init__(self) -> None:
        if self.horizon < 1:
            raise ValueError(f"horizon must be positive, got {self.horizon}")
        if self.step < 1:
            raise ValueError(f"step must be positive, got {self.step}")

    @property
    def overlapping(self) -> bool:
        return self.step < self.horizon

    @property
    def stride(self) -> int:
        """Records apart two forecasts must sit before their windows are disjoint."""
        return math.ceil(self.horizon / self.step)

    def independent(self) -> Sampling:
        """The same series thinned: same horizon, a step wide enough to stop overlapping."""
        return Sampling(horizon=self.horizon, step=self.stride * self.step)

    def thin(self, values: np.ndarray) -> np.ndarray:
        """Every ``stride``-th observation — the subsample the tests are allowed to see."""
        thinned: np.ndarray = np.asarray(values)[:: self.stride]
        return thinned

    def independent_count(self, n: int) -> int:
        return len(range(0, max(n, 0), self.stride))

    def describe(self) -> dict[str, Any]:
        return {
            "horizon_sessions": self.horizon,
            "step_sessions": self.step,
            "overlapping": self.overlapping,
            "independent_stride": self.stride,
        }


def require_independent(sampling: Sampling, statistic: str) -> None:
    """Refuse to compute ``statistic`` on overlapping windows.

    Deliberately an exception rather than a warning or a docstring: the failure mode
    is a p-value that looks like evidence, and nothing downstream can tell that it was
    computed on a series with a quarter of the observations it claims.
    """
    if sampling.overlapping:
        raise OverlappingWindowsError(
            f"{statistic} assumes independent observations, but forecasts every "
            f"{sampling.step} sessions over a {sampling.horizon}-session horizon "
            f"overlap. Thin to every {sampling.stride}th record first "
            f"(Sampling.thin), or report the statistic as a diagnostic without a "
            f"p-value."
        )
