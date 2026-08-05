"""The binary events a probability forecast can be graded on.

Every event here is a property of the price alone — where it ended, or how far it
travelled on the way. None of them needs a venue, a fee, a tax or a leverage cap, which
is what lets this whole package be scored against a price history and nothing else.

The one event that *is* defined by friction — :class:`ClearsHurdle` — turned out to be
no exception, and that was the design bet. It takes a *number*: the gross move required,
already computed by :mod:`aurex.trade` from a route and a jurisdiction that this module
never sees. So the friction lives entirely outside the scorer and the event is graded by
the same machinery as direction and touch, which is why adding it needed no new metric.

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

    def applies_at(self, horizon: int) -> bool:
        """Is this event defined at that horizon?

        Almost always yes: "ends higher" and "touches -10%" mean the same thing over a
        week and over a quarter. The exception is an event whose *level* was derived
        from a horizon — a breakeven hurdle on friction that accrues is a different
        number for a week's hold than for a quarter's, so scoring one horizon's hurdle
        at another asks whether the price cleared a cost nobody in that window paid.

        A protocol member rather than a filter in the harness, because the event is what
        knows its own scope, and the alternative is a harness that has to guess from an
        id. Every event is asked; most answer yes to everything.
        """
        ...

    def probability(self, ensemble: PathEnsemble) -> float:
        """The forecast probability, read off the simulated paths."""
        ...

    def occurred(self, anchor: float, realised: RealisedPath) -> bool:
        """Whether it happened, measured the same way the probability was."""
        ...

    def describe(self) -> dict[str, Any]: ...


class _AnyHorizon:
    """Mixin for the price events, whose definitions do not depend on the horizon."""

    def applies_at(self, horizon: int) -> bool:
        del horizon
        return True


def _tag(move: float) -> str:
    return f"{move * 100:g}pct"


@dataclass(frozen=True, slots=True)
class TerminalAbove(_AnyHorizon):
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
class TouchBelow(_AnyHorizon):
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
class TouchAbove(_AnyHorizon):
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


@dataclass(frozen=True, slots=True)
class ClearsHurdle:
    """Did the move clear a round-trip breakeven? The one event friction defines.

    It arrives as an ordinary :class:`BinaryEvent` rather than as a new metric beside
    the others, which was the point of building the reliability machinery first: the
    hurdle changes the *level* the question is asked at, not the question. So it is
    graded by the same code, on the same records, with the same conventions.

    ``multiple`` is the gross move required — 1.0937 for a 9.37% hurdle. It arrives
    already computed, and the arithmetic that produced it lives in :mod:`aurex.trade`,
    because this module may not know what a dealer premium or a consumption tax is.
    ``label`` is opaque for the same reason: a route and a jurisdiction produced it and
    neither may be named here.

    **Read the Brier score against its own uncertainty term, never across events.** At
    retail friction this event has a base rate of a few per cent, so a forecaster that
    says "probably not" every time scores well on it in absolute terms and has learned
    nothing. The reliability machinery reports the positive count beside every score and
    withholds the diagram when that count cannot support one; see
    :mod:`aurex.score.reliability`.
    """

    multiple: float
    label: str = ""
    #: The horizon this hurdle was priced for, where the friction accrues. ``None`` for
    #: friction paid at the door, which is the same number at every horizon. Set it and
    #: the event is scored at that horizon only — see :meth:`applies_at`.
    horizon: int | None = None

    def __post_init__(self) -> None:
        if self.multiple <= 1.0:
            raise ValueError(
                f"a breakeven multiple must exceed 1, got {self.multiple}. A hurdle of "
                f"zero or less is a round trip that pays the holder to take it, which "
                f"is a friction bug rather than an event worth scoring."
            )
        if self.horizon is not None and self.horizon < 1:
            raise ValueError(f"a priced horizon must be positive, got {self.horizon}")

    def applies_at(self, horizon: int) -> bool:
        """A horizon-priced hurdle is scored at its own horizon and nowhere else.

        Accruing friction makes the hurdle a function of the holding period, so the
        quarterly hurdle asks a question about a cost a weekly holder never paid, and
        vice versa. Scoring both against every window produced four meaningless rows for
        every meaningful one in the first run that had them, which is how this was
        found.
        """
        return self.horizon is None or self.horizon == horizon

    @property
    def required_move(self) -> float:
        return self.multiple - 1.0

    @property
    def id(self) -> str:
        parts = ["clears_hurdle", self.label, _tag(self.required_move)]
        if self.horizon is not None:
            parts.append(f"h{self.horizon}")
        return "_".join(part for part in parts if part)

    @property
    def monitoring(self) -> Monitoring:
        """Terminal: a round trip is closed at the horizon, not on the way to it.

        A path that touched the hurdle mid-horizon and fell back did not clear it for a
        holder who sold at the end, and this event grades that holder. The touch version
        is a different question and :class:`TouchAbove` already asks it.
        """
        return "terminal"

    def level(self, anchor: float) -> float:
        return anchor * self.multiple

    def probability(self, ensemble: PathEnsemble) -> float:
        return float(np.mean(ensemble.terminal() >= self.level(ensemble.anchor)))

    def occurred(self, anchor: float, realised: RealisedPath) -> bool:
        return bool(realised.terminal >= self.level(anchor))

    def describe(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "monitoring": self.monitoring,
            "definition": (
                f"terminal price at or above anchor x {self.multiple:g}, i.e. a gross "
                f"move of at least {self.required_move * 100:.2f}%"
            ),
            "breakeven_multiple": round(self.multiple, 6),
            "required_move_pct": round(self.required_move * 100.0, 4),
            "label": self.label or None,
            "priced_for_horizon": self.horizon,
            "caveat": (
                "A rare event. Its Brier score is bounded below by a small uncertainty "
                "term, so it is not comparable with the score of a near-even event; "
                "read it against the base rate and positive count reported beside it."
            ),
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
