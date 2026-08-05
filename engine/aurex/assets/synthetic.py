"""A synthetic asset, for proving the abstraction does not leak.

§16's test: instantiate a trivial asset and run the full pipeline on it. If that
needs any change outside this file, the abstraction has leaked.

It is deliberately unlike gold on every axis the protocol exposes — different
currency, different unit, a shifted-log transform, and both friction shapes — so a
gold-shaped assumption anywhere downstream shows up as a failure here rather than as
a surprise when oil arrives. Its price series is generated in-process, so the test
needs no network and no fixtures.
"""

from __future__ import annotations

from datetime import date
from typing import Any, ClassVar

import numpy as np
import pandas as pd

from aurex.assets.base import FactorSpec, VolConfig, describe_asset
from aurex.assets.friction import FrictionProfile, PhysicalFriction, RollFriction
from aurex.assets.lens import CurrencyLens, NativeLens, TaxedImportLens
from aurex.assets.transforms import ReturnTransform, ShiftedLogReturn
from aurex.data.base import LoadedSeries, build_meta
from aurex.data.cache import CacheStore
from aurex.data.chain import SourceChain
from aurex.data.freshness import SeriesFreshness
from aurex.vol.limits import SessionLimit

#: ISO 4217 reserves XTS for testing, so it can never collide with a real currency.
TEST_CURRENCY = "XTS"
TEST_LENS_CURRENCY = "XTT"

#: The synthetic asset declares staleness tolerances like any other, because the
#: freshness guard is part of the abstraction and an asset that skipped it would let
#: the guard become gold-specific without the leak test noticing. Generated series are
#: business-day dense, so four days covers a weekend run with room to spare.
_FRESHNESS: dict[str, SeriesFreshness] = {
    sid: SeriesFreshness(
        max_lag_days=4,
        calendar="generated business days",
        rationale=(
            "Generated in-process over a business-day index, so the only gap is the "
            "weekend. Declared rather than defaulted so this asset exercises the same "
            "path gold does."
        ),
    )
    for sid in ("widget_price", "widget_fx", "widget_local")
}


class GeneratedLoader:
    """Deterministic in-process price series. No network, no fixtures."""

    def __init__(self, series_id: str, start_value: float = 100.0, seed: int = 0) -> None:
        self.series_id = series_id
        self.source_name = f"synthetic:{series_id}"
        self.start_value = start_value
        self.seed = seed

    def fetch(self, start: date, end: date) -> LoadedSeries:
        index = pd.bdate_range(start, end, name="date")
        if len(index) == 0:
            raise ValueError(f"{self.series_id}: empty window {start}..{end}")

        rng = np.random.default_rng(self.seed)
        steps = rng.normal(0.0, 0.01, size=len(index))
        values = self.start_value * np.exp(np.cumsum(steps))
        frame = pd.DataFrame({"close": values}, index=index)

        return LoadedSeries(
            frame=frame,
            meta=build_meta(
                series_id=self.series_id,
                source_name=self.source_name,
                source_url="https://example.invalid/synthetic",
                frame=frame,
            ),
        )


class SyntheticAsset:
    """A trivial asset exercising every member of the protocol."""

    id = "widget"
    label = "Synthetic widget"
    quote_currency = TEST_CURRENCY
    base_unit = "unit"
    price_series_id = "widget_price"
    # No OHLC anywhere, so the range-based model is unavailable to this asset — the
    # case a close-only pricing series puts every asset in until a second one exists.
    ohlc_series_id = None
    reference_rate_series = "widget_local"
    reference_rate_column = "close"

    # Not log returns: proves the transform is genuinely pluggable.
    return_transform: ReturnTransform = ShiftedLogReturn(shift=25.0)

    # A short window and a session limit, so the leak test exercises the whole
    # distribution path — including the branch a venue-capped instrument takes —
    # over the three months of prices this asset generates.
    vol_defaults = VolConfig(
        default_model="rolling_std",
        min_observations=10,
        model_options={"window": 20},
        session_limit=SessionLimit(fraction=0.05, relaxed_fraction=0.08),
    )
    scenario_axes = ("geopolitics",)

    currency_lenses: tuple[CurrencyLens, ...] = (
        NativeLens(code=TEST_CURRENCY, unit_label="unit", units_per_base=1.0),
        TaxedImportLens(
            code=TEST_LENS_CURRENCY,
            unit_label="crate",
            units_per_base=12.0,
            fx_series_id="widget_fx",
            duty_resolver=lambda _day: 0.10,
            tax_resolver=lambda _day: 0.05,
            high_confidence_from=date(2000, 1, 1),
        ),
    )

    # Both friction shapes, so a downstream assumption that friction is
    # horizon-independent fails here rather than in the oil module.
    friction_profiles: ClassVar[dict[str, FrictionProfile]] = {
        "spot": PhysicalFriction(
            id="spot",
            label="Synthetic physical",
            dealer_premium=0.02,
            consumption_tax=0.05,
            buyback_discount=0.02,
        ),
        "rolled": RollFriction(
            id="rolled",
            label="Synthetic rolled future",
            annual_expense_ratio=0.005,
            annual_roll_drag=0.06,
            bid_ask_spread=0.001,
        ),
    }

    # Deliberately self-contained: referencing a shared macro series would drag the
    # leak test onto the network, and the point of this asset is that it needs
    # nothing outside its own definition. `absent_driver` exercises the
    # graceful-degradation path for a factor whose series has no loader — the same
    # path oil's EIA inventories take when no API key is present.
    factor_set: tuple[FactorSpec, ...] = (
        FactorSpec(
            id="momentum",
            series_id="widget_price",
            transform="pct_change",
            description="Lagged own return.",
        ),
        FactorSpec(
            id="absent_driver",
            series_id="never_resolves",
            transform="level",
            required=False,
            description="Optional factor with no source, to exercise degradation.",
        ),
    )

    def price_sources(self, cache: CacheStore | None = None) -> dict[str, SourceChain]:
        store = cache or CacheStore()
        spec = {
            "widget_price": (GeneratedLoader("widget_price", 100.0, seed=1),),
            "widget_fx": (GeneratedLoader("widget_fx", 80.0, seed=2),),
            # Sits slightly above parity, so the premium is non-zero and signed.
            "widget_local": (GeneratedLoader("widget_local", 1_400.0, seed=3),),
        }
        return {
            sid: SourceChain(sid, loaders, store, freshness=_FRESHNESS[sid])
            for sid, loaders in spec.items()
        }

    def describe(self) -> dict[str, Any]:
        return describe_asset(self)


SYNTHETIC = SyntheticAsset()
