"""Currency lenses — one engine, two views.

An asset has one price. A buyer has a currency, a unit, and a tax regime standing
between them and that price. A lens is that translation, and it is deliberately not
a second pipeline: §15 asks for a toggle rather than a page, because two code paths
computing the same thing will drift.

**Where the boundary sits.** A lens carries taxes that are part of the *reference
price* — national, mandatory, and known: India's import duty and GST. It does not
carry vehicle-level costs, which belong to :mod:`aurex.assets.friction`. US state
sales tax sits on the friction side rather than here, because it varies by state and
frequently exempts bullion entirely; §15 asks for a user-editable field defaulting to
zero, not fifty encoded state rules.

**The USD view is free.** The INR price is the USD price with FX and a tax stack
applied on top, so the USD figure is the intermediate that already exists partway
through the INR computation. Exposing it costs nothing and, per §15, makes visible
that the Indian buyer carries two extra sources of uncertainty.

That decomposition is displayed as measured, never asserted. XAU/USD and USD/INR are
dependent — fitting a t-copula over exactly that dependence is the point of §4 — so
their variances do not add, and the FX leg can offset rather than compound. The INR
distribution is usually wider; the UI must still be able to render the case where it
is not.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import date
from typing import Any, Protocol, runtime_checkable

import numpy as np
import pandas as pd

#: Columns every lens produces, so downstream code never branches on lens identity.
LENS_COLUMNS = (
    "price",
    "price_ex_consumption_tax",
    "base_quote_per_unit",
    "fx_rate",
    "duty",
    "consumption_tax",
    "confidence",
)


@dataclass(frozen=True, slots=True)
class LensContext:
    """Series a lens may need beyond the asset's own price."""

    #: Units of lens currency per unit of the asset's quote currency.
    fx: pd.Series | None = None


@runtime_checkable
class CurrencyLens(Protocol):
    """Translates an asset's quoted price into a buyer's currency and unit."""

    @property
    def code(self) -> str:
        """Currency code this lens presents prices in."""
        ...

    @property
    def unit_label(self) -> str:
        """Unit prices are quoted per, e.g. ``10g`` or ``troy_ounce``."""
        ...

    @property
    def requires_fx(self) -> bool:
        """Whether :meth:`apply` needs an FX series in its context."""
        ...

    @property
    def fx_series_id(self) -> str | None:
        """Series id supplying the FX rate, or None for a native lens.

        Lives on the lens rather than the asset so one asset can carry several
        foreign lenses, each with its own rate.
        """
        ...

    @property
    def produces_local_premium(self) -> bool:
        """Whether an observed local reference rate exists to difference against.

        False for USD gold — §15 notes the local-premium signal has no USD analogue.
        """
        ...

    def apply(self, base_prices: pd.Series, ctx: LensContext) -> pd.DataFrame:
        """Return a frame with :data:`LENS_COLUMNS`."""
        ...

    def describe(self) -> dict[str, Any]: ...


def _empty_frame() -> pd.DataFrame:
    return pd.DataFrame(columns=list(LENS_COLUMNS))


@dataclass(frozen=True, slots=True)
class NativeLens:
    """Identity view on an asset already quoted in the buyer's currency.

    No FX, no import duty, no copula — §15's "single exposure" case, and the reason
    the USD path is the cleaner one to teach from.
    """

    code: str = "USD"
    unit_label: str = "troy_ounce"
    #: Lens units per unit of the asset's base quote (1.0 when they match).
    units_per_base: float = 1.0
    requires_fx: bool = False
    fx_series_id: str | None = None
    produces_local_premium: bool = False

    def apply(self, base_prices: pd.Series, ctx: LensContext) -> pd.DataFrame:
        del ctx
        if base_prices.empty:
            return _empty_frame()

        price = base_prices.astype(float) * self.units_per_base
        return pd.DataFrame(
            {
                "price": price,
                "price_ex_consumption_tax": price,
                "base_quote_per_unit": price,
                "fx_rate": 1.0,
                "duty": 0.0,
                "consumption_tax": 0.0,
                "confidence": "high",
            }
        )

    def describe(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "unit": self.unit_label,
            "units_per_base": self.units_per_base,
            "requires_fx": self.requires_fx,
            "produces_local_premium": self.produces_local_premium,
        }


@dataclass(frozen=True, slots=True)
class TaxedImportLens:
    """FX plus an import-duty and consumption-tax stack.

    ``price_ex_consumption_tax = base * units_per_base * fx * (1 + duty)``
    ``price                    = price_ex_consumption_tax * (1 + consumption_tax)``

    The split is load-bearing, not cosmetic. India's IBJA publishes its reference
    rate *exclusive* of GST, so a domestic premium must be measured against
    ``price_ex_consumption_tax``; measuring against the tax-inclusive figure would
    print a spurious premium of roughly -291bps at all times.

    Duty and tax arrive as resolvers rather than imports so the lens stays free of
    any particular asset's schedule. Gold passes its own; a different import regime
    would pass different ones.
    """

    code: str
    unit_label: str
    units_per_base: float
    #: Series id supplying units of lens currency per unit of quote currency.
    fx_series_id: str
    #: date -> duty as a fraction, or None where no ad valorem rate is defined.
    duty_resolver: Callable[[date], float | None]
    #: date -> consumption tax as a fraction. Zero before a national regime exists.
    tax_resolver: Callable[[date], float]
    #: Dates before this are tagged low confidence (no single national rate).
    high_confidence_from: date
    #: date -> citation for the rates applied on that date. Every externally-sourced
    #: rate must be traceable to its source and confidence level, so this travels
    #: with the price rather than being reconstructed downstream.
    provenance_resolver: Callable[[date], dict[str, Any]] | None = None
    requires_fx: bool = True
    produces_local_premium: bool = True
    notes: tuple[str, ...] = field(default_factory=tuple)

    def provenance_for(self, day: date) -> dict[str, Any]:
        """Citations for the rates in force on ``day``."""
        return {} if self.provenance_resolver is None else self.provenance_resolver(day)

    def apply(self, base_prices: pd.Series, ctx: LensContext) -> pd.DataFrame:
        if ctx.fx is None:
            raise ValueError(f"{self.code} lens requires an FX series")

        joined = pd.concat(
            {"base": base_prices, "fx": ctx.fx}, axis=1, join="inner", sort=False
        ).dropna()
        if joined.empty:
            return _empty_frame()

        duties: list[float] = []
        taxes: list[float] = []
        confidences: list[str] = []
        for stamp in joined.index:
            day = stamp.date()
            duty = self.duty_resolver(day)
            duties.append(np.nan if duty is None else duty)
            taxes.append(self.tax_resolver(day))
            confidences.append("high" if day >= self.high_confidence_from else "low")

        duty_series = pd.Series(duties, index=joined.index)
        tax_series = pd.Series(taxes, index=joined.index)

        # The buyer's-currency price before any local levy — §15's intermediate,
        # which is the native-lens view of the same instant.
        base_quote_per_unit = joined["base"].astype(float) * self.units_per_base
        ex_tax = base_quote_per_unit * joined["fx"].astype(float) * (1.0 + duty_series)

        out = pd.DataFrame(
            {
                "price": ex_tax * (1.0 + tax_series),
                "price_ex_consumption_tax": ex_tax,
                "base_quote_per_unit": base_quote_per_unit,
                "fx_rate": joined["fx"].astype(float),
                "duty": duty_series,
                "consumption_tax": tax_series,
                "confidence": pd.Series(confidences, index=joined.index),
            }
        )
        # Drop dates with no defined ad valorem rate rather than invent one.
        return out[out["duty"].notna()]

    def describe(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "unit": self.unit_label,
            "units_per_base": self.units_per_base,
            "requires_fx": self.requires_fx,
            "fx_series_id": self.fx_series_id,
            "produces_local_premium": self.produces_local_premium,
            "high_confidence_from": self.high_confidence_from.isoformat(),
            "notes": list(self.notes),
        }
