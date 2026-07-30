"""Gold. Every gold-specific fact in Aurex lives in this file.

Two decisions here are worth reading before changing anything.

**Spot, not futures, for pricing.** ``GC=F`` is the COMEX front-month *future* and
carries a cost-of-carry basis over spot — +2.40% against the London PM fix on
2026-07-29. Pricing off it pushes that basis into the domestic premium and, because
the basis moves with rates and time to expiry, makes it look like a moving
domestic-demand signal. The London PM fix is the pricing series; it is also the
benchmark IBJA prints in its own daily report, which keeps the comparison
like-for-like. Futures are still loaded as ``xau_futures`` because the realised-
volatility estimators want true OHLC that a close-only fix cannot give them.

**GST sits in the lens, US sales tax sits in friction.** GST is national, mandatory
and known, so it belongs to the reference price. US state sales tax varies by state
and commonly exempts bullion, so per §15 it is a user-editable friction field
defaulting to zero rather than fifty encoded rules.
"""

from __future__ import annotations

from datetime import date
from typing import Any, ClassVar

from aurex.assets.base import FactorSpec, VolConfig, describe_asset
from aurex.assets.friction import US_COLLECTIBLES_NOTE, FrictionProfile, PhysicalFriction
from aurex.assets.lens import CurrencyLens, NativeLens, TaxedImportLens
from aurex.assets.transforms import LogReturn, ReturnTransform
from aurex.config import GRAMS_PER_TROY_OUNCE
from aurex.data.base import Loader
from aurex.data.cache import CacheStore
from aurex.data.chain import SourceChain
from aurex.data.schedules import duty_on, gst_on
from aurex.data.sources import FredLoader, IbjaReportLoader, LbmaGoldLoader, YahooLoader

#: India's GST regime begins here; before it, state-varying VAT plus excise with no
#: single national rate, so parity is indicative only.
GST_REGIME_START = date(2017, 7, 1)

#: IBJA quotes rupees per 10 grams.
INR_QUOTE_GRAMS = 10.0


def _duty_for(day: date) -> float | None:
    entry = duty_on(day)
    return entry.total if entry else None


def _gst_for(day: date) -> float:
    entry = gst_on(day)
    return entry.metal if entry else 0.0


def _provenance_for(day: date) -> dict[str, Any]:
    """Citations for the duty and GST applied on ``day``.

    Carried alongside the price so a reader can check the rate without leaving the
    artifact, and so `source_confidence` stays visible — the 2026-05-13 duty entry
    is `secondary` because CBIC's primary document is not machine-retrievable.
    """
    duty = duty_on(day)
    gst = gst_on(day)
    return {
        "duty": None
        if duty is None
        else {
            "effective_from": duty.effective_from.isoformat(),
            "source_url": duty.source_url,
            "source_confidence": duty.source_confidence,
        },
        "consumption_tax": None
        if gst is None
        else {
            "effective_from": gst.effective_from.isoformat(),
            "source_url": gst.source_url,
            "source_confidence": gst.source_confidence,
        },
    }


class Gold:
    """Physical gold, priced from the London PM fix."""

    id = "gold"
    label = "Gold"
    quote_currency = "USD"
    base_unit = "troy_ounce"
    price_series_id = "xauusd"
    reference_rate_series = "ibja_gold"
    reference_rate_column = "gold_999_pm"

    #: Close-only spot for pricing; OHLC futures for realised-vol estimators.
    ohlc_series_id = "xau_futures"

    return_transform: ReturnTransform = LogReturn()

    vol_defaults = VolConfig(
        default_model="gjr_garch",
        annualisation_days=252,
        min_observations=500,
        break_aware=True,
    )

    scenario_axes = ("geopolitics", "cpi", "payrolls")

    currency_lenses: tuple[CurrencyLens, ...] = (
        NativeLens(
            code="USD",
            unit_label="troy_ounce",
            units_per_base=1.0,
        ),
        TaxedImportLens(
            code="INR",
            unit_label="10g",
            units_per_base=INR_QUOTE_GRAMS / GRAMS_PER_TROY_OUNCE,
            fx_series_id="usdinr",
            duty_resolver=_duty_for,
            tax_resolver=_gst_for,
            high_confidence_from=GST_REGIME_START,
            provenance_resolver=_provenance_for,
            notes=(
                "IBJA publishes its 999 rate exclusive of GST; the domestic premium "
                "is measured against price_ex_consumption_tax.",
            ),
        ),
    )

    friction_profiles: ClassVar[dict[str, FrictionProfile]] = {
        "inr_retail": PhysicalFriction(
            id="inr_retail",
            label="India retail jeweller",
            dealer_premium=0.03,
            consumption_tax=0.03,
            buyback_discount=0.03,
            notes=("GST on metal is 3%; making charges are taxed separately at 5%.",),
        ),
        "inr_etf": PhysicalFriction(
            id="inr_etf",
            label="India gold ETF",
            dealer_premium=0.005,
            consumption_tax=0.0,
            buyback_discount=0.005,
            notes=("Expense ratio accrues separately and is not in the round-trip spread.",),
        ),
        "inr_sgb": PhysicalFriction(
            id="inr_sgb",
            label="Sovereign gold bond",
            dealer_premium=0.0,
            consumption_tax=0.0,
            buyback_discount=0.0,
            notes=(
                "No GST and no making charge. Liquidity before maturity is the "
                "binding constraint rather than spread.",
            ),
        ),
        "usd_coin": PhysicalFriction(
            id="usd_coin",
            label="US retail coin",
            dealer_premium=0.03,
            consumption_tax=0.0,
            buyback_discount=0.02,
            notes=(
                "Sales tax defaults to 0% and is user-editable: treatment varies by "
                "state and many exempt bullion.",
                US_COLLECTIBLES_NOTE,
            ),
        ),
        "usd_bar": PhysicalFriction(
            id="usd_bar",
            label="US larger bar",
            dealer_premium=0.015,
            consumption_tax=0.0,
            buyback_discount=0.01,
            notes=(
                "Sales tax defaults to 0% and is user-editable.",
                US_COLLECTIBLES_NOTE,
            ),
        ),
    }

    factor_set: tuple[FactorSpec, ...] = (
        FactorSpec(
            id="d_real_yield",
            series_id="real_yield_10y",
            transform="diff",
            description="Change in 10y TIPS real yield. The dominant carry channel.",
        ),
        FactorSpec(
            id="d_dxy",
            series_id="dxy",
            transform="pct_change",
            description="Trade-weighted dollar. Gold is priced in it.",
        ),
        FactorSpec(
            id="d_oil",
            series_id="wti",
            transform="pct_change",
            description="Crude, as an inflation-expectations proxy.",
        ),
        FactorSpec(
            id="d_vix",
            series_id="vix",
            transform="diff",
            description="Equity volatility, a partial risk-appetite proxy.",
        ),
        FactorSpec(
            id="d_geopolitical_risk",
            series_id="gpr",
            transform="level",
            required=False,
            description=(
                "Geopolitical risk. Present so the safe-haven channel is ESTIMATED "
                "rather than omitted. Without it the scenario engine reaches "
                "'escalation -> gold down' purely through the real-yield channel, "
                "with a clean causal story and very likely the wrong sign, since "
                "gold historically rallies on escalation. Omitted-variable bias here "
                "is more dangerous than a hand-typed view because it passes the "
                "check §6 and §17.7 specify."
            ),
        ),
        FactorSpec(
            id="etf_flow",
            series_id="ibja_gold",
            transform="diff",
            required=False,
            description="SPDR holdings in tonnes, from the IBJA daily report.",
        ),
        FactorSpec(
            id="momentum",
            series_id="xauusd",
            transform="pct_change",
            description="Lagged own return. Attribution only, never a directional signal.",
        ),
    )

    def price_sources(self, cache: CacheStore | None = None) -> dict[str, SourceChain]:
        store = cache or CacheStore()
        spec: dict[str, tuple[Loader, ...]] = {
            # Spot. Deliberately not GC=F — see the module docstring.
            "xauusd": (LbmaGoldLoader("xauusd"),),
            # OHLC futures, for the volatility layer only. Never for pricing.
            "xau_futures": (YahooLoader("xau_futures", "GC=F"),),
            "usdinr": (
                YahooLoader("usdinr", "INR=X"),
                FredLoader("usdinr", "DEXINUS", "close"),
            ),
            "ibja_gold": (IbjaReportLoader("ibja_gold"),),
        }
        return {sid: SourceChain(sid, loaders, store) for sid, loaders in spec.items()}

    def describe(self) -> dict[str, Any]:
        return describe_asset(self) | {"ohlc_series_id": self.ohlc_series_id}


GOLD = Gold()
