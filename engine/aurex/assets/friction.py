"""Friction profiles — what the price move has to beat.

§16 warns against forcing gold's and oil's friction into one another's shape, because
they are structurally different:

* **Gold** pays its friction at the door. Dealer premium and consumption tax on the
  way in, buyback discount on the way out. Hold it for a day or a decade and the
  hurdle is the same.
* **Oil** pays its friction over time. Front-month futures must be rolled, and in
  contango you sell low and buy high every month; that drag compounds. This is what
  gutted USO.

The one thing that makes them substitutable without flattening the difference is
taking **horizon** in the interface. Gold's implementation ignores it; oil's cannot.
A horizon-free interface would have forced oil's roll drag into a fictional
one-off spread, which is precisely the shape violation §16 prohibits.

`breakeven_multiple` is the number the §10 hurdle line draws.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable


@dataclass(frozen=True, slots=True)
class FrictionQuote:
    """What a round trip costs, for one horizon."""

    #: Multiplier on spot paid at entry (>= 1 for a buyer).
    cost_multiplier: float
    #: Multiplier on spot received at exit (<= 1 for a seller).
    proceeds_multiplier: float
    #: Gross move required to break even: ``cost / proceeds``.
    breakeven_multiple: float
    #: Named contributions, so the UI can show where the hurdle comes from.
    components: dict[str, float] = field(default_factory=dict)
    #: Informational notes. Never a computed tax liability — see `collectibles`.
    notes: tuple[str, ...] = ()

    @property
    def breakeven_pct(self) -> float:
        """Required move as a percentage, e.g. 9.37 for a 9.37% hurdle."""
        return (self.breakeven_multiple - 1.0) * 100.0


@runtime_checkable
class FrictionProfile(Protocol):
    """One vehicle's cost structure."""

    @property
    def id(self) -> str:
        """Stable identifier for this vehicle."""
        ...

    @property
    def label(self) -> str:
        """Human-readable name for the UI."""
        ...

    def quote(self, horizon_days: int) -> FrictionQuote:
        """Cost of a round trip held for ``horizon_days``."""
        ...

    def describe(self) -> dict[str, Any]:
        """Provenance and parameters for the artifact."""
        ...


@dataclass(frozen=True, slots=True)
class PhysicalFriction:
    """Spread-and-tax friction, paid at entry and exit. Horizon-independent.

    ``breakeven = (1 + premium) * (1 + tax) / (1 - buyback)``

    Indian retail defaults (3% / 3% / 3%) give a 9.37% hurdle. A US coin buyer with
    no state sales tax and a tighter buyback (3% / 0% / 2%) faces 5.10% — the
    same arithmetic, and the reason §15's USD path is the cleaner case.
    """

    id: str
    label: str
    dealer_premium: float
    consumption_tax: float
    buyback_discount: float
    notes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not 0.0 <= self.buyback_discount < 1.0:
            raise ValueError(f"{self.id}: buyback_discount must be in [0, 1)")
        for name in ("dealer_premium", "consumption_tax"):
            if getattr(self, name) < 0.0:
                raise ValueError(f"{self.id}: {name} must be non-negative")

    def quote(self, horizon_days: int) -> FrictionQuote:
        del horizon_days  # Physical friction is paid at the door, not over time.
        cost = (1.0 + self.dealer_premium) * (1.0 + self.consumption_tax)
        proceeds = 1.0 - self.buyback_discount
        return FrictionQuote(
            cost_multiplier=cost,
            proceeds_multiplier=proceeds,
            breakeven_multiple=cost / proceeds,
            components={
                "dealer_premium": self.dealer_premium,
                "consumption_tax": self.consumption_tax,
                "buyback_discount": self.buyback_discount,
            },
            notes=self.notes,
        )

    def describe(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "label": self.label,
            "kind": "physical",
            "horizon_dependent": False,
            "dealer_premium": self.dealer_premium,
            "consumption_tax": self.consumption_tax,
            "buyback_discount": self.buyback_discount,
            "notes": list(self.notes),
        }


@dataclass(frozen=True, slots=True)
class RollFriction:
    """Carry friction that accrues with holding period.

    ``breakeven = (1 + spread) * (1 + (expense + roll_drag) * horizon/365) / (1 - spread)``

    Used for futures-tracking vehicles. ``annual_roll_drag`` is the implied cost of
    staying in the front month at the current curve shape: positive in contango,
    negative in backwardation.

    **Not deterministic beyond the next roll.** The current curve pins the imminent
    roll; every roll after that depends on a curve shape that has not happened yet.
    Treating the whole horizon as known understates uncertainty, so
    :meth:`describe` records how far the quote is actually pinned. Making the
    subsequent rolls stochastic is step 8's job, not this layer's.
    """

    id: str
    label: str
    annual_expense_ratio: float
    annual_roll_drag: float
    bid_ask_spread: float = 0.0
    #: Horizon beyond which the roll drag is an assumption, not an observation.
    pinned_days: int = 30
    notes: tuple[str, ...] = ()

    def quote(self, horizon_days: int) -> FrictionQuote:
        if horizon_days < 0:
            raise ValueError(f"{self.id}: horizon_days must be non-negative")
        years = horizon_days / 365.0
        carry = (self.annual_expense_ratio + self.annual_roll_drag) * years

        cost = (1.0 + self.bid_ask_spread) * (1.0 + carry)
        proceeds = 1.0 - self.bid_ask_spread
        return FrictionQuote(
            cost_multiplier=cost,
            proceeds_multiplier=proceeds,
            breakeven_multiple=cost / proceeds,
            components={
                "expense": self.annual_expense_ratio * years,
                "roll_drag": self.annual_roll_drag * years,
                "bid_ask_spread": self.bid_ask_spread,
            },
            notes=(
                *self.notes,
                (
                    f"Roll drag is pinned by the current curve for about "
                    f"{self.pinned_days} days; beyond that it is an assumption."
                )
                if horizon_days > self.pinned_days
                else "Roll drag is pinned by the current curve over this horizon.",
            ),
        )

    def describe(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "label": self.label,
            "kind": "roll",
            "horizon_dependent": True,
            "annual_expense_ratio": self.annual_expense_ratio,
            "annual_roll_drag": self.annual_roll_drag,
            "bid_ask_spread": self.bid_ask_spread,
            "pinned_days": self.pinned_days,
            "notes": list(self.notes),
        }


#: Informational only. Aurex states that a tax treatment exists and links the source;
#: it never computes a liability. Surfaced in the §15 calculator.
US_COLLECTIBLES_NOTE = (
    "US long-term capital gains on physical gold are taxed as a collectible, at a "
    "higher maximum rate than equities. Aurex does not compute tax. See IRS Topic 409: "
    "https://www.irs.gov/taxtopics/tc409"
)
