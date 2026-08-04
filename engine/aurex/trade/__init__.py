"""Friction and the hurdle it sets — what the price move has to beat.

Consumes the friction profile a route and jurisdiction resolve to, and turns it into the
gross move a round trip must clear to break even. Nothing here knows which asset, which
venue or which country it is holding: a :class:`Hurdle` carries an opaque label, and the
composition that produced it lives in :mod:`aurex.routes` and the CLI.

**Horizon is in the interface, and that is load-bearing.** Physical friction is paid at
the door and does not care how long the position is held; carry friction accrues and
compounds. A hurdle that took no horizon would have forced the second shape into the
first, understating a long hold and overstating a short one. So every hurdle records
both its horizon and whether that horizon mattered, and the rendered table keeps the
horizon columns even where a row ignores them — otherwise a reader cannot tell which
rows are being held constant and which are being evaluated.

**The table is generated, never typed.** :func:`breakeven_table` renders the markdown
the README publishes, and a test regenerates it and compares. A hand-written friction
table drifts from the data behind it the first time a rate changes, and the version a
reader sees is the one that is wrong.

P(profit), expected P&L and the liquidation probability arrive with the dashboard, along
with the rule that a headline states the expected loss rather than the win probability
where P(profit) is below one half.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from aurex.assets.friction import FrictionProfile

#: Horizons the published table is rendered at: a week, a month, a quarter, a year.
#: A single-horizon table cannot show which friction shapes accrue.
TABLE_HORIZONS: tuple[int, ...] = (5, 21, 63, 252)


@dataclass(frozen=True, slots=True)
class Hurdle:
    """The gross move a round trip must clear to break even, at one horizon.

    ``label`` is opaque on purpose. This module renders and compares hurdles; deciding
    that a particular label means a particular country is somebody else's job, and
    keeping that out of here is what lets one renderer serve every route.
    """

    label: str
    horizon_days: int
    multiple: float
    components: dict[str, float]
    horizon_dependent: bool
    notes: tuple[str, ...] = ()

    @property
    def required_move(self) -> float:
        """Fraction the price must rise before the round trip returns the outlay."""
        return self.multiple - 1.0

    @property
    def required_move_pct(self) -> float:
        return self.required_move * 100.0

    def describe(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "horizon_sessions": self.horizon_days,
            "breakeven_multiple": round(self.multiple, 6),
            "required_move_pct": round(self.required_move_pct, 4),
            "horizon_dependent": self.horizon_dependent,
            "components": {k: round(v, 6) for k, v in sorted(self.components.items())},
            "notes": list(self.notes),
            "reading": (
                "The move the price must make before the position returns what went "
                "into it. It is deterministic and knowable; whether the price makes it "
                "is neither, which is the asymmetry this project exists to show."
            ),
        }


def hurdle_for(friction: FrictionProfile, *, horizon_days: int, label: str) -> Hurdle:
    """Resolve one friction profile at one horizon."""
    if horizon_days < 0:
        raise ValueError(f"horizon_days must be non-negative, got {horizon_days}")
    quote = friction.quote(horizon_days)
    described = friction.describe()
    return Hurdle(
        label=label,
        horizon_days=horizon_days,
        multiple=quote.breakeven_multiple,
        components=dict(quote.components),
        horizon_dependent=bool(described.get("horizon_dependent", False)),
        notes=quote.notes,
    )


def hurdles_over(
    friction: FrictionProfile, *, horizons: tuple[int, ...], label: str
) -> tuple[Hurdle, ...]:
    """The same profile at several horizons, ascending."""
    return tuple(
        hurdle_for(friction, horizon_days=horizon, label=label) for horizon in sorted(horizons)
    )


def breakeven_table(
    rows: tuple[tuple[str, FrictionProfile], ...],
    *,
    horizons: tuple[int, ...] = TABLE_HORIZONS,
) -> str:
    """Render the published breakeven table as markdown.

    One row per ``(label, friction)`` pair, one column per horizon. This function is the
    single definition of what that table says; the README holds a copy that a test keeps
    honest.
    """
    if not rows:
        raise ValueError("a breakeven table needs at least one row")
    ordered = tuple(sorted(horizons))

    lines = [
        "| Route | " + " | ".join(f"{h} sessions" for h in ordered) + " | Accrues |",
        "|---|" + "---|" * (len(ordered) + 1),
    ]
    for label, friction in rows:
        quotes = hurdles_over(friction, horizons=ordered, label=label)
        cells = " | ".join(f"{entry.required_move_pct:.2f}%" for entry in quotes)
        lines.append(f"| {label} | {cells} | {'yes' if quotes[0].horizon_dependent else 'no'} |")

    return "\n".join(lines)


__all__ = [
    "TABLE_HORIZONS",
    "Hurdle",
    "breakeven_table",
    "hurdle_for",
    "hurdles_over",
]
