"""Daily price limits, and why a simulator that ignores them is wrong.

Some exchanges cap how far a contract may move in one session. A simulated path that
gaps through that cap in a single session is not a path that can occur, and the
error is not symmetric in consequence: the paths a limit truncates are exactly the
ones that trigger a margin call, so ignoring limits overstates liquidation
probability, while pretending the truncated move never happened understates it.

§18 asks for the honest middle: truncate the session to the limit and carry the
unexecuted remainder into the next session, where it is subject to the limit again.
A shock large enough to lock the market for two sessions therefore takes two
sessions to arrive, which is what a limit-locked market actually does.

The relaxed band models a cooling-off rule — a wider limit applies in the session
following a locked one. Aurex does not model the trading halt itself: nothing here
claims a position could have been exited during it, which is the conservative
direction for a leveraged holder.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class SessionLimit:
    """A cap on one session's proportional price move."""

    #: Ordinary cap, as a fraction of the previous settlement, e.g. ``0.04``.
    fraction: float
    #: Wider cap applying in the session after a locked one, if the venue has one.
    relaxed_fraction: float | None = None
    #: Whether the unexecuted remainder carries into the next session.
    carry_residual: bool = True
    #: Where the rule comes from. Provenance travels with the constant, per §1.5.
    source_url: str | None = None
    notes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not 0.0 < self.fraction < 1.0:
            raise ValueError(f"fraction must be in (0, 1), got {self.fraction}")
        if self.relaxed_fraction is not None and self.relaxed_fraction < self.fraction:
            raise ValueError(
                f"relaxed_fraction {self.relaxed_fraction} is tighter than the "
                f"ordinary limit {self.fraction}; a cooling-off band widens"
            )

    def cap_for(self, *, previous_session_locked: bool) -> float:
        """The fraction in force this session."""
        if previous_session_locked and self.relaxed_fraction is not None:
            return self.relaxed_fraction
        return self.fraction

    def describe(self) -> dict[str, Any]:
        return {
            "fraction": self.fraction,
            "relaxed_fraction": self.relaxed_fraction,
            "carry_residual": self.carry_residual,
            "source_url": self.source_url,
            "notes": list(self.notes),
        }
