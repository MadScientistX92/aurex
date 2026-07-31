"""Return transforms — the internal representation, and nothing else.

Different assets need different return definitions. Gold is fine on log returns.
WTI settled at -36.98 on 2020-04-20, and ``log(-36.98)`` is undefined, so oil needs
either a shift or a switch to arithmetic differences.

**The hazard this module exists to contain.** A shifted-log transform silently
rescales anything quoted as a percentage out of transform space. Measured on the WTI
series Aurex already caches, same sample, varying only the shift constant:

===========  ==========================================
shift ``c``  annualised sd of shifted-log returns
===========  ==========================================
50           55.2%
100          24.6%
===========  ==========================================

The number more than halves on a constant that is arbitrary by construction. So a
transform is an *internal representation*: models are fitted in transform space, and
every reported quantity is mapped back to price space by :meth:`from_returns` before
anyone sees it.

This module therefore deliberately provides **no** volatility, quantile or
percentage-width helper. There is nothing here to accidentally report. Reporting
takes prices; see :mod:`aurex.assets.friction` and, later, the scoring layer.

Round-tripping is necessary but not sufficient as a check — a badly-scaled transform
round-trips perfectly. The test that matters asserts reported outputs do not move
when ``c`` moves.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

import numpy as np
import pandas as pd


class TransformDomainError(ValueError):
    """The series contains values the transform cannot represent."""


@runtime_checkable
class ReturnTransform(Protocol):
    """Maps a price series to returns and back.

    Implementations must satisfy ``from_returns(P[0], to_returns(P)) == P[1:]`` to
    floating-point tolerance, including across non-positive prices where the
    transform claims to support them.
    """

    # Declared as a read-only property rather than a bare attribute so frozen
    # dataclasses satisfy the protocol; a plain `id: str` would demand a settable
    # variable. Same reasoning applies to the other protocols in this package.
    @property
    def id(self) -> str:
        """Stable identifier, recorded in the artifact."""
        ...

    def to_returns(self, prices: pd.Series) -> pd.Series:
        """Price series -> return series. One shorter than the input."""
        ...

    def from_returns(self, anchor: float, returns: pd.Series | np.ndarray) -> np.ndarray:
        """Return series -> price path, starting from ``anchor``.

        This is the only sanctioned route back to reportable units.
        """
        ...

    def advance(self, prices: np.ndarray, step_returns: np.ndarray) -> np.ndarray:
        """One step, vectorised across an ensemble of prices.

        :meth:`from_returns` walks a whole path at once, which is the wrong shape for
        a simulator that must inspect each session before taking the next — a daily
        price limit truncates the session it lands in and carries the remainder
        forward, so the price has to exist between steps.
        """
        ...

    def step_return(self, prices: np.ndarray, next_prices: np.ndarray) -> np.ndarray:
        """Inverse of :meth:`advance`: the return that moves ``prices`` to ``next_prices``.

        A truncated session needs to know how much of the move actually executed, in
        return space, to work out what is left to carry.
        """
        ...

    def describe(self) -> dict[str, Any]:
        """Provenance for the artifact. The transform is never invisible."""
        ...


@dataclass(frozen=True, slots=True)
class LogReturn:
    """Standard log returns. Requires strictly positive prices."""

    id: str = "log"

    def to_returns(self, prices: pd.Series) -> pd.Series:
        if (prices <= 0).any():
            bad = prices[prices <= 0]
            raise TransformDomainError(
                f"log returns need positive prices; found {len(bad)} non-positive "
                f"(first: {bad.index[0].date()} = {bad.iloc[0]}). Use ShiftedLogReturn."
            )
        returns: pd.Series = np.log(prices).diff().dropna()
        return returns

    def from_returns(self, anchor: float, returns: pd.Series | np.ndarray) -> np.ndarray:
        return float(anchor) * np.exp(np.cumsum(np.asarray(returns, dtype=float)))

    def advance(self, prices: np.ndarray, step_returns: np.ndarray) -> np.ndarray:
        moved: np.ndarray = np.asarray(prices, dtype=float) * np.exp(
            np.asarray(step_returns, dtype=float)
        )
        return moved

    def step_return(self, prices: np.ndarray, next_prices: np.ndarray) -> np.ndarray:
        executed: np.ndarray = np.log(
            np.asarray(next_prices, dtype=float) / np.asarray(prices, dtype=float)
        )
        return executed

    def describe(self) -> dict[str, Any]:
        return {"id": self.id, "formula": "log(P_t) - log(P_{t-1})"}


@dataclass(frozen=True, slots=True)
class ShiftedLogReturn:
    """Log returns on ``P + shift``, so non-positive prices stay representable.

    ``shift`` must exceed the most negative price in the series. It is recorded in
    the artifact via :meth:`describe` because it is a modelling choice, not a fact
    about the world — and because, per the module docstring, it changes the scale of
    anything quoted from transform space.
    """

    shift: float
    id: str = "shifted_log"

    def to_returns(self, prices: pd.Series) -> pd.Series:
        shifted = prices + self.shift
        if (shifted <= 0).any():
            worst = float(prices.min())
            raise TransformDomainError(
                f"shift={self.shift} is too small; series minimum is {worst}. "
                f"Need shift > {-worst}."
            )
        returns: pd.Series = np.log(shifted).diff().dropna()
        return returns

    def from_returns(self, anchor: float, returns: pd.Series | np.ndarray) -> np.ndarray:
        base = float(anchor) + self.shift
        return base * np.exp(np.cumsum(np.asarray(returns, dtype=float))) - self.shift

    def advance(self, prices: np.ndarray, step_returns: np.ndarray) -> np.ndarray:
        shifted = np.asarray(prices, dtype=float) + self.shift
        return shifted * np.exp(np.asarray(step_returns, dtype=float)) - self.shift

    def step_return(self, prices: np.ndarray, next_prices: np.ndarray) -> np.ndarray:
        shifted = np.asarray(prices, dtype=float) + self.shift
        return np.log((np.asarray(next_prices, dtype=float) + self.shift) / shifted)

    def describe(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "shift": self.shift,
            "formula": "log(P_t + c) - log(P_{t-1} + c)",
            "caveat": (
                "c rescales any percentage computed in transform space. All reported "
                "quantities are back-transformed to price space before display."
            ),
        }


@dataclass(frozen=True, slots=True)
class Difference:
    """Arithmetic differences. Domain-safe everywhere, but changes the model.

    Returns are in price units rather than proportions, so a constant-variance model
    on differences means something quite different from one on log returns. Offered
    because §17 lists it; not the default for anything.
    """

    id: str = "difference"

    def to_returns(self, prices: pd.Series) -> pd.Series:
        return prices.diff().dropna()

    def from_returns(self, anchor: float, returns: pd.Series | np.ndarray) -> np.ndarray:
        return float(anchor) + np.cumsum(np.asarray(returns, dtype=float))

    def advance(self, prices: np.ndarray, step_returns: np.ndarray) -> np.ndarray:
        moved: np.ndarray = np.asarray(prices, dtype=float) + np.asarray(step_returns, dtype=float)
        return moved

    def step_return(self, prices: np.ndarray, next_prices: np.ndarray) -> np.ndarray:
        moved: np.ndarray = np.asarray(next_prices, dtype=float) - np.asarray(prices, dtype=float)
        return moved

    def describe(self) -> dict[str, Any]:
        return {"id": self.id, "formula": "P_t - P_{t-1}", "units": "price"}


def round_trip(transform: ReturnTransform, prices: pd.Series) -> np.ndarray:
    """Transform and invert, returning the reconstructed prices from index 1 on.

    Used by tests and available as a runtime self-check on a new asset's transform.
    """
    return transform.from_returns(float(prices.iloc[0]), transform.to_returns(prices))
