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

from aurex.assets.base import (
    ChainControl,
    ChainLink,
    FactorSpec,
    TransmissionChain,
    VolConfig,
    describe_asset,
)
from aurex.assets.friction import US_COLLECTIBLES_NOTE, FrictionProfile, PhysicalFriction
from aurex.assets.lens import CurrencyLens, NativeLens, TaxedImportLens
from aurex.assets.transforms import LogReturn, ReturnTransform
from aurex.config import GRAMS_PER_TROY_OUNCE
from aurex.data.base import Loader
from aurex.data.cache import CacheStore
from aurex.data.chain import SourceChain
from aurex.data.freshness import SeriesFreshness
from aurex.data.schedules import duty_on, fuel_excise_on, gst_on, load_fuel_excise
from aurex.data.sources import FredLoader, IbjaReportLoader, LbmaGoldLoader, YahooLoader

#: India's GST regime begins here; before it, state-varying VAT plus excise with no
#: single national rate, so parity is indicative only.
GST_REGIME_START = date(2017, 7, 1)

#: IBJA quotes rupees per 10 grams.
INR_QUOTE_GRAMS = 10.0

#: How far behind the run date each of gold's series may fall before a forecast built
#: on it is a fabrication rather than a forecast. See :mod:`aurex.data.freshness`.
#:
#: ``xauusd`` and ``usdinr`` are the blocking pair: the fix is what every lens is
#: composed from, and the rupee lens converts through the rate. Four days is the
#: binding number and it is set by the worst *ordinary* case rather than by the
#: average one — a job running 02:00 UTC on the Tuesday after a Monday holiday sees
#: Friday's fix, which is four days behind and entirely healthy. A longer closure
#: exceeds it and is refused, which is the intended behaviour: there is no new price,
#: so there is nothing new to publish, and the gap is recorded rather than papered over.
_FRESHNESS: dict[str, SeriesFreshness] = {
    "xauusd": SeriesFreshness(
        max_lag_days=4,
        calendar="London business days (LBMA PM fix)",
        rationale=(
            "The PM fix prints at 15:00 London, so it is available well before a 02:00 "
            "UTC run the next day. The longest ordinary gap is a 02:00 Tuesday run "
            "after a Monday holiday reading Friday's fix — four days. Anything beyond "
            "that is a multi-day closure or a broken source, and both must refuse: "
            "this series is the anchor every published price and every simulated path "
            "is built from."
        ),
    ),
    "usdinr": SeriesFreshness(
        max_lag_days=4,
        calendar="FX trading days",
        rationale=(
            "FX trades Sunday 22:00 to Friday 22:00 UTC, so the calendar matches the "
            "fix's. Blocking because the rupee lens converts through it: a stale rate "
            "over a fresh fix publishes a rupee price that was never quoted."
        ),
    ),
    "xau_futures": SeriesFreshness(
        max_lag_days=4,
        calendar="COMEX trading days",
        rationale=(
            "Not blocking — futures feed the realised-volatility estimators, never a "
            "published price. Declared and measured anyway so a Yahoo rate limit that "
            "silently degrades the vol layer to close-only is visible in the artifact."
        ),
    ),
    "ibja_gold": SeriesFreshness(
        max_lag_days=6,
        calendar="Indian business days",
        rationale=(
            "Not blocking. IBJA closes for local holidays that cluster several days "
            "deep, and §0's sixth rule already covers the consequence: where there is "
            "no observation the premium is NaN rather than computed. Six days keeps a "
            "Diwali week from reading as a fault while still surfacing a dead feed."
        ),
    ),
}


def _duty_for(day: date) -> float | None:
    entry = duty_on(day)
    return entry.total if entry else None


def _gst_for(day: date) -> float:
    entry = gst_on(day)
    return entry.metal if entry else 0.0


def _excise_for(day: date) -> float | None:
    """Total central excise per litre across both transport fuels on ``day``.

    ``None`` inside the window the schedule declares it does not know, rather than the
    last known level carried forward. See ``fuel_excise.yaml``: at least three changes
    fall in that window with no retrievable primary document behind them, and a control
    silently held constant across the largest fuel-tax increase in the sample would be
    worse than no control at all.
    """
    entry = fuel_excise_on(day)
    return None if entry is None else entry.combined


def _excise_provenance() -> dict[str, Any]:
    """Every citation behind the excise control, and every hole in it."""
    entries, gaps = load_fuel_excise()
    return {
        "units": "rupees_per_litre_petrol_plus_diesel",
        "entries": [
            {
                "effective_from": entry.effective_from.isoformat(),
                "petrol": entry.petrol,
                "diesel": entry.diesel,
                "combined": round(entry.combined, 2),
                "source_url": entry.source_url,
                "source_confidence": entry.source_confidence,
            }
            for entry in entries
        ],
        "gaps": [
            {
                "from": gap.start.isoformat(),
                "until": gap.end.isoformat(),
                "reason": gap.reason,
                "source_url": gap.source_url,
                "source_confidence": gap.source_confidence,
            }
            for gap in gaps
        ],
    }


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

    transmission_chain = TransmissionChain(
        id="crude_to_local_price",
        label="Crude to the rupee price of gold",
        links=(
            ChainLink(
                id="crude_to_cpi",
                source_series="wti",
                target_series="local_cpi",
                source_transform="log_diff",
                target_transform="log_diff",
                description=(
                    "Crude into the import bill and through to consumer prices. This is "
                    "the link the fuel-tax buffer sits inside: excise and state VAT are "
                    "adjusted by hand, in the opposite direction, precisely when crude "
                    "moves, so an uncontrolled estimate here measures a fiscal decision "
                    "as if it were a pricing mechanism."
                ),
            ),
            ChainLink(
                id="cpi_to_policy",
                source_series="local_cpi",
                target_series="local_policy_rate",
                source_transform="log_diff",
                target_transform="diff",
                description=(
                    "Inflation into the policy response, measured on the money-market "
                    "rate rather than on the repo rate itself. That is a deliberate "
                    "weakening of the claim: the corridor's transmission is what is "
                    "observable in a free monthly series, and calling it the policy rate "
                    "would assert a passthrough this repository has not measured."
                ),
            ),
            ChainLink(
                id="policy_to_rupee",
                source_series="local_policy_rate",
                target_series="usdinr",
                source_transform="diff",
                target_transform="log_diff",
                description="The policy response into the exchange rate.",
            ),
        ),
        terminal_lens="INR",
        # Not a loaded series: the INR lens computes it from the fix, the rate and the
        # dated tax schedules, and the caller passes that output in under this id. The
        # chain never recomputes a local price — there is one implementation of the tax
        # stack and it is the one with the citations attached.
        terminal_series_id="local_price",
        direct_source_series="wti",
        # The quote-currency fix, held constant so the direct estimate's coefficient is
        # the local-economy part of the path. Without it, the direct estimate carries the
        # global gold-price channel that the weekly crude loading already measures, and
        # the two numbers would be two views of one thing presented as two things.
        direct_controls=("xauusd",),
        controls=(
            ChainControl(
                id="fuel_excise",
                description=(
                    "Total central excise per litre on petrol and diesel, summed. The "
                    "discretionary buffer between a crude move and a consumer price."
                ),
                resolver=_excise_for,
                provenance=_excise_provenance,
            ),
        ),
        note=(
            "Four arrows, of which three are estimated and one is arithmetic. The last "
            "step from the exchange rate to the rupee price is an identity the lens "
            "computes, so it is asserted at one and checked, not fitted."
        ),
    )

    currency_lenses: tuple[CurrencyLens, ...] = (
        NativeLens(
            code="USD",
            unit_label="troy_ounce",
            units_per_base=1.0,
            grams_per_unit=GRAMS_PER_TROY_OUNCE,
        ),
        TaxedImportLens(
            code="INR",
            unit_label="10g",
            units_per_base=INR_QUOTE_GRAMS / GRAMS_PER_TROY_OUNCE,
            grams_per_unit=INR_QUOTE_GRAMS,
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
            # Changed from `level` when the series was wired, and before any loading
            # was fitted. Two reasons, both stated in the README's pre-registration so
            # neither can be read as a choice made after seeing a coefficient. First,
            # every other regressor here is news — a change, not a stock — and a
            # persistent level regressed on near-unpredictable returns is the classic
            # way to manufacture a significant-looking coefficient. Second, the channel
            # being measured is the bid that arrives when risk *rises*; a risk level
            # that has been elevated for a month is already in the price.
            transform="diff",
            # Averaged over the week rather than read off Friday. The index is
            # calendar-daily and spiky — a single day's value is a draw, not a summary —
            # and a Friday reading would discard six sevenths of what the week said.
            aggregation="mean",
            # The one factor whose absence changes the SIGN of a published story rather
            # than the size of a coefficient, so attribution refuses rather than
            # degrades when it is missing. Optional factors drop out with a recorded
            # reason; this one cannot, because the reason would be recorded in a field
            # nobody reads while the loadings underneath told a confident, inverted
            # story about escalation.
            required=True,
            description=(
                "Geopolitical risk, Caldara-Iacoviello daily index, weekly change. "
                "Present so the safe-haven channel is ESTIMATED rather than omitted. "
                "Without it the scenario engine reaches 'escalation -> gold down' "
                "purely through the real-yield channel, with a clean causal story and "
                "very likely the wrong sign, since gold historically rallies on "
                "escalation. Omitted-variable bias here is more dangerous than a "
                "hand-typed view because it passes the check §6 and §17.7 specify."
            ),
        ),
        FactorSpec(
            id="etf_flow",
            series_id="ibja_gold",
            transform="diff",
            # Named explicitly. This series carries four columns and the flow proxy is
            # not the first of them; without this the factor would have loaded the AM
            # local rate and reported it under the flow factor's name.
            column="spdr_gold_tonnes",
            required=False,
            description="SPDR holdings in tonnes, from the IBJA daily report.",
        ),
        FactorSpec(
            id="momentum",
            series_id="xauusd",
            transform="pct_change",
            # One week, and this is not a detail. Every other factor is contemporaneous
            # with the return being attributed, which is what attribution means. This
            # one is built from the price series itself, so at zero lag it would hand
            # the regression its own target under a different name and report an
            # R-squared near one.
            lag=1,
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
        return {
            sid: SourceChain(sid, loaders, store, freshness=_FRESHNESS.get(sid))
            for sid, loaders in spec.items()
        }

    def describe(self) -> dict[str, Any]:
        return describe_asset(self)


GOLD = Gold()
