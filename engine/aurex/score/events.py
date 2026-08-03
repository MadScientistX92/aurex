"""The binary events a probability forecast can be graded on.

Every event here is a property of the price alone — where it ended, or how far it
travelled on the way. None of them needs a venue, a fee, a tax or a leverage cap, which
is what lets this whole package be scored against a price history and nothing else. The
one event that *is* defined by friction — whether a move cleared a breakeven hurdle —
is not a different kind of object, it is one more implementation of
:class:`BinaryEvent`, and it arrives when there is a route to define the hurdle.

**Touch events are monitored at session close, on both sides.** The simulator walks
paths session by session and never sees an intraday extreme, so a simulated touch means
"closed through the level". The realised outcome must be measured the same way. If a
realised touch were read off intraday highs and lows while the forecast could only ever
close through a level, the model would be charged for the gap — and that gap is a
declared limitation of the engine, not a calibration failure. That is why
:class:`RealisedPath` carries closes and nothing else: the convention is enforced by
there being no other data to reach for.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Protocol, runtime_checkable

import numpy as np

from aurex.dist.paths import PathEnsemble

#: What a probability is measured over. ``terminal`` reads the horizon alone;
#: ``session_close`` reads every close on the way and is therefore a floor.
Monitoring = Literal["terminal", "session_close"]


@dataclass(frozen=True, slots=True)
class RealisedPath:
    """What actually happened after a forecast was made: closes, and nothing else.

    The single field is the enforcement. A high or a low added here would silently
    redefine every touch event in this module — every realised touch probability would
    rise, and the model would start failing a calibration test for a convention it
    never claimed. Widening this class must therefore be a deliberate edit that also
    changes what the engine publishes about how it monitors.
    """

    closes: np.ndarray

    def __post_init__(self) -> None:
        values = np.asarray(self.closes, dtype=float)
        if values.ndim != 1:
            raise ValueError(f"closes must be one-dimensional, got shape {values.shape}")
        if values.size == 0:
            raise ValueError("a realised path with no sessions cannot resolve an event")
        object.__setattr__(self, "closes", values)

    @property
    def sessions(self) -> int:
        return int(self.closes.size)

    @property
    def terminal(self) -> float:
        return float(self.closes[-1])

    @property
    def lowest_close(self) -> float:
        return float(np.min(self.closes))

    @property
    def highest_close(self) -> float:
        return float(np.max(self.closes))


@runtime_checkable
class BinaryEvent(Protocol):
    """Something that either happened or did not, and that the ensemble has a view on."""

    @property
    def id(self) -> str: ...

    @property
    def monitoring(self) -> Monitoring: ...

    def probability(self, ensemble: PathEnsemble) -> float:
        """The forecast probability, read off the simulated paths."""
        ...

    def occurred(self, anchor: float, realised: RealisedPath) -> bool:
        """Whether it happened, measured the same way the probability was."""
        ...

    def describe(self) -> dict[str, Any]: ...


def _tag(move: float) -> str:
    return f"{move * 100:g}pct"


@dataclass(frozen=True, slots=True)
class TerminalAbove:
    """Did the price end above a level? ``move`` of zero is plain direction."""

    move: float = 0.0

    @property
    def id(self) -> str:
        return "direction_up" if self.move == 0.0 else f"terminal_above_{_tag(self.move)}"

    @property
    def monitoring(self) -> Monitoring:
        return "terminal"

    def level(self, anchor: float) -> float:
        return anchor * (1.0 + self.move)

    def probability(self, ensemble: PathEnsemble) -> float:
        return float(np.mean(ensemble.terminal() > self.level(ensemble.anchor)))

    def occurred(self, anchor: float, realised: RealisedPath) -> bool:
        return bool(realised.terminal > self.level(anchor))

    def describe(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "monitoring": self.monitoring,
            "definition": f"terminal price above anchor x {1.0 + self.move:g}",
        }


@dataclass(frozen=True, slots=True)
class TouchBelow:
    """Did the price ever *close* at or below a level on the way to the horizon?"""

    move: float

    def __post_init__(self) -> None:
        if not 0.0 < self.move < 1.0:
            raise ValueError(f"a downside move must be a fraction in (0, 1), got {self.move}")

    @property
    def id(self) -> str:
        return f"touch_below_{_tag(self.move)}"

    @property
    def monitoring(self) -> Monitoring:
        return "session_close"

    def level(self, anchor: float) -> float:
        return anchor * (1.0 - self.move)

    def probability(self, ensemble: PathEnsemble) -> float:
        return float(np.mean(ensemble.running_minimum() <= self.level(ensemble.anchor)))

    def occurred(self, anchor: float, realised: RealisedPath) -> bool:
        return bool(realised.lowest_close <= self.level(anchor))

    def describe(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "monitoring": self.monitoring,
            "definition": f"any session closing at or below anchor x {1.0 - self.move:g}",
            "caveat": (
                "Monitored at session close on both sides. A level breached and "
                "recovered inside one session counts for neither the forecast nor the "
                "outcome, so this probability is a floor."
            ),
        }


@dataclass(frozen=True, slots=True)
class TouchAbove:
    """The upside mirror of :class:`TouchBelow`."""

    move: float

    def __post_init__(self) -> None:
        if self.move <= 0.0:
            raise ValueError(f"an upside move must be positive, got {self.move}")

    @property
    def id(self) -> str:
        return f"touch_above_{_tag(self.move)}"

    @property
    def monitoring(self) -> Monitoring:
        return "session_close"

    def level(self, anchor: float) -> float:
        return anchor * (1.0 + self.move)

    def probability(self, ensemble: PathEnsemble) -> float:
        return float(np.mean(ensemble.running_maximum() >= self.level(ensemble.anchor)))

    def occurred(self, anchor: float, realised: RealisedPath) -> bool:
        return bool(realised.highest_close >= self.level(anchor))

    def describe(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "monitoring": self.monitoring,
            "definition": f"any session closing at or above anchor x {1.0 + self.move:g}",
            "caveat": ("Monitored at session close on both sides, so this probability is a floor."),
        }


def default_events(moves: tuple[float, ...] = (0.05, 0.10)) -> tuple[BinaryEvent, ...]:
    """Direction, plus a touch either way at each reference move.

    Direction is here because it is the claim readers arrive wanting tested, and a
    calibrated 52% is the honest answer to it.
    """
    events: list[BinaryEvent] = [TerminalAbove()]
    for move in moves:
        events.append(TouchBelow(move))
        events.append(TouchAbove(move))
    return tuple(events)
