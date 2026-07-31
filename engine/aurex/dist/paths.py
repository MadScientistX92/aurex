"""The path array, and why it is the object rather than an implementation detail.

Every distribution in Aurex used to be a *terminal* distribution: simulate to the
horizon, throw the intermediate days away, report quantiles of what is left. §18
shows what that discards. A position that cannot survive an intermediate drawdown —
anything leveraged, anything with a stop, anything margined — does not experience the
terminal distribution at all. For a driftless walk the chance of *touching* a level
before the horizon is close to twice the chance of *ending* beyond it, so a terminal
distribution understates the event the holder actually cares about by about half.

So the simulator keeps the paths, and this class is what it returns. Terminal
quantiles are a method on it, not the thing itself.

Memory is the reason to be deliberate about size rather than the reason to collapse:
20,000 paths over 63 sessions is ten megabytes, which is nothing, and the answers it
buys cannot be recovered afterwards.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

#: Reported quantiles. Deliberately no mean and no mode: a single number extracted
#: from a distribution reads as a forecast no matter how it is labelled, and §0 bans
#: that. The median is here because it is a quantile among quantiles.
DEFAULT_QUANTILES = (0.05, 0.25, 0.5, 0.75, 0.95)


@dataclass(frozen=True, slots=True)
class PathEnsemble:
    """Simulated price paths in reportable units, one path per row."""

    #: ``(n_paths, horizon)``. Column ``t`` is the price after ``t + 1`` sessions.
    prices: np.ndarray
    #: The price the paths start from, i.e. the last observation.
    anchor: float
    #: Provenance: model, seed, block length, and anything the simulator truncated.
    diagnostics: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.prices.ndim != 2:
            raise ValueError(f"prices must be (n_paths, horizon), got {self.prices.shape}")
        if self.prices.size == 0:
            raise ValueError("an empty ensemble is not a distribution")

    @property
    def n_paths(self) -> int:
        return int(self.prices.shape[0])

    @property
    def horizon(self) -> int:
        return int(self.prices.shape[1])

    def terminal(self) -> np.ndarray:
        """Prices at the horizon. One view of the ensemble, never the only one."""
        return self.prices[:, -1]

    def quantiles(
        self, probabilities: tuple[float, ...] = DEFAULT_QUANTILES, *, step: int | None = None
    ) -> dict[str, float]:
        """Price quantiles at ``step`` sessions ahead, or at the horizon by default."""
        column = self.prices[:, -1 if step is None else step - 1]
        values = np.quantile(column, probabilities)
        return {f"q{int(p * 100):02d}": float(v) for p, v in zip(probabilities, values, strict=True)}

    def running_minimum(self) -> np.ndarray:
        """Lowest price each path reached. The input to a downside barrier test."""
        return np.min(self.prices, axis=1)

    def running_maximum(self) -> np.ndarray:
        return np.max(self.prices, axis=1)

    def probability_below(self, level: float, *, terminal_only: bool = False) -> float:
        """Chance of being below ``level`` at the horizon, or of ever touching it.

        The gap between the two readings is the whole point of keeping paths; see
        :mod:`aurex.dist.passage` for the statistic that reports both together.
        """
        series = self.terminal() if terminal_only else self.running_minimum()
        return float(np.mean(series <= level))

    def probability_above(self, level: float, *, terminal_only: bool = False) -> float:
        series = self.terminal() if terminal_only else self.running_maximum()
        return float(np.mean(series >= level))

    def scaled(self, factor: float) -> PathEnsemble:
        """Every path multiplied by a constant — a unit change, or a fixed rate."""
        return PathEnsemble(
            prices=self.prices * factor,
            anchor=self.anchor * factor,
            diagnostics=dict(self.diagnostics) | {"constant_factor": factor},
        )

    def rescaled(self, other: PathEnsemble, factor: float = 1.0) -> PathEnsemble:
        """Path-by-path product of two ensembles, times a constant.

        A price in a second currency is the first price times an exchange rate times
        a tax stack, and the multiplication has to happen *inside* each path: taking
        the product of two marginal distributions would assume an independence that
        the copula was fitted precisely to avoid. The two ensembles must therefore
        have been simulated from linked shocks — see :func:`aurex.dist.copula.joint_shocks`.

        The constant is whatever is fixed over the horizon, such as a statutory rate
        in force today. Aurex does not simulate policy: a rate that changes is a
        recorded break, not a random variable.
        """
        if self.prices.shape != other.prices.shape:
            raise ValueError(
                f"ensembles must be the same shape to combine, got "
                f"{self.prices.shape} and {other.prices.shape}"
            )
        return PathEnsemble(
            prices=self.prices * other.prices * factor,
            anchor=self.anchor * other.anchor * factor,
            # Both sets are kept under their own key rather than merged: each
            # simulation had its own model and its own truncations, and flattening
            # them would attribute one's diagnostics to the other.
            diagnostics={
                "primary": self.diagnostics,
                "secondary": other.diagnostics,
                "constant_factor": factor,
            },
        )

    def describe(self) -> dict[str, Any]:
        return {
            "n_paths": self.n_paths,
            "horizon_sessions": self.horizon,
            "anchor": self.anchor,
            "terminal_quantiles": self.quantiles(),
            "diagnostics": dict(self.diagnostics),
            "note": (
                "Quantiles of a simulated distribution, not a forecast of a price. "
                "Paths are retained, so path-dependent statistics come from the same "
                "ensemble as the terminal ones."
            ),
        }
