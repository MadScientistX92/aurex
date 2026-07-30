"""Nightly orchestrator.

Asset-driven: the pipeline resolves whatever the requested assets declare, applies
each asset's lenses, and emits one block per asset per lens. It contains no
asset-specific knowledge — adding oil means adding an asset module, not editing this
file.

Current scope is data, lenses and residuals. There are no forecasts here yet, and
when there are, there will still be no point estimates.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any

import pandas as pd

from aurex import __version__
from aurex.assets import REGISTRY
from aurex.assets.base import Asset
from aurex.assets.lens import LensContext
from aurex.config import PUBLIC_DATA_DIR
from aurex.data.base import DataUnavailableError, LoadedSeries
from aurex.data.cache import CacheStore
from aurex.data.parity import local_premium_bps, passthrough_diagnostic
from aurex.data.registry import chains_for
from aurex.data.schedules import load_policy_breaks

log = logging.getLogger(__name__)

#: Twenty years of history, per the spec's data requirements.
DEFAULT_LOOKBACK_DAYS = 365 * 20


@dataclass(frozen=True, slots=True)
class LensResult:
    """One asset seen through one lens."""

    code: str
    frame: pd.DataFrame
    premium: pd.Series = field(default_factory=lambda: pd.Series(dtype=float))


@dataclass(frozen=True, slots=True)
class AssetResult:
    asset_id: str
    lenses: dict[str, LensResult]


@dataclass(frozen=True, slots=True)
class PipelineResult:
    artifact: dict[str, Any]
    series: dict[str, LoadedSeries]
    assets: dict[str, AssetResult]


def _price_column(frame: pd.DataFrame) -> pd.Series:
    """Pick the price column, tolerating OHLC and close-only shapes."""
    for candidate in ("close", "value"):
        if candidate in frame.columns:
            return frame[candidate]
    numeric = frame.select_dtypes("number")
    if numeric.empty:
        raise ValueError(f"no numeric column in {list(frame.columns)}")
    return numeric.iloc[:, 0]


def run(
    *,
    offline: bool = False,
    start: date | None = None,
    end: date | None = None,
    cache: CacheStore | None = None,
    assets: Sequence[Asset] | None = None,
) -> PipelineResult:
    """Resolve data, apply lenses, and assemble the artifact."""
    end = end or datetime.now(UTC).date()
    start = start or (end - timedelta(days=DEFAULT_LOOKBACK_DAYS))
    store = cache or CacheStore()
    active: Sequence[Asset] = assets if assets is not None else tuple(REGISTRY.values())

    series, provenance, unavailable = _resolve(active, start, end, store, offline)
    results = {asset.id: _apply_lenses(asset, series) for asset in active}

    artifact = _build_artifact(
        active=active,
        results=results,
        provenance=provenance,
        unavailable=unavailable,
        offline=offline,
    )
    return PipelineResult(artifact=artifact, series=series, assets=results)


def _resolve(
    assets: Iterable[Asset],
    start: date,
    end: date,
    store: CacheStore,
    offline: bool,
) -> tuple[dict[str, LoadedSeries], dict[str, Any], dict[str, str]]:
    series: dict[str, LoadedSeries] = {}
    provenance: dict[str, Any] = {}
    unavailable: dict[str, str] = {}

    for series_id, chain in chains_for(assets, store).items():
        try:
            loaded = chain.load(start, end, offline=offline)
        except DataUnavailableError as exc:
            log.warning("%s unavailable: %s", series_id, exc)
            unavailable[series_id] = str(exc)
            continue
        series[series_id] = loaded
        provenance[series_id] = loaded.meta.to_dict()

    return series, provenance, unavailable


def _apply_lenses(asset: Asset, series: dict[str, LoadedSeries]) -> AssetResult:
    """Run every lens the asset declares, skipping those missing their inputs."""
    lenses: dict[str, LensResult] = {}

    price = series.get(asset.price_series_id)
    if price is None:
        return AssetResult(asset_id=asset.id, lenses=lenses)
    base_prices = _price_column(price.frame)

    observed = _observed_reference(asset, series)

    for lens in asset.currency_lenses:
        fx = None
        if lens.requires_fx:
            if lens.fx_series_id is None:
                raise ValueError(f"{asset.id}/{lens.code}: requires_fx but no fx_series_id")
            fx_series = series.get(lens.fx_series_id)
            if fx_series is None:
                log.warning("%s/%s skipped: %s unavailable", asset.id, lens.code, lens.fx_series_id)
                continue
            fx = _price_column(fx_series.frame)

        frame = lens.apply(base_prices, LensContext(fx=fx))
        premium = pd.Series(dtype=float)
        if lens.produces_local_premium and observed is not None and not frame.empty:
            premium = local_premium_bps(observed, frame["price_ex_consumption_tax"])

        lenses[lens.code] = LensResult(code=lens.code, frame=frame, premium=premium)

    return AssetResult(asset_id=asset.id, lenses=lenses)


def _observed_reference(asset: Asset, series: dict[str, LoadedSeries]) -> pd.Series | None:
    if asset.reference_rate_series is None or asset.reference_rate_column is None:
        return None
    loaded = series.get(asset.reference_rate_series)
    if loaded is None or asset.reference_rate_column not in loaded.frame.columns:
        return None
    return loaded.frame[asset.reference_rate_column]


def _lens_block(asset: Asset, result: LensResult) -> dict[str, Any]:
    lens = next(x for x in asset.currency_lenses if x.code == result.code)
    block: dict[str, Any] = {"lens": lens.describe()}

    if result.frame.empty:
        return block | {"latest": None, "local_premium": None}

    row = result.frame.iloc[-1]
    block["latest"] = {
        "as_of": result.frame.index[-1].date().isoformat(),
        "price": round(float(row["price"]), 2),
        "price_ex_consumption_tax": round(float(row["price_ex_consumption_tax"]), 2),
        "base_quote_per_unit": round(float(row["base_quote_per_unit"]), 4),
        "fx_rate": float(row["fx_rate"]),
        "duty": float(row["duty"]),
        "consumption_tax": float(row["consumption_tax"]),
        "confidence": str(row["confidence"]),
        "unit": lens.unit_label,
        "currency": lens.code,
    }

    provenance_for = getattr(lens, "provenance_for", None)
    if provenance_for is not None:
        block["latest"]["rate_provenance"] = provenance_for(result.frame.index[-1].date())

    if not lens.produces_local_premium:
        block["local_premium"] = None
        return block

    observed = result.premium.dropna()
    premium_block: dict[str, Any] = {
        "observations": int(observed.size),
        "note": (
            "Observed local reference rate over the modelled tax-exclusive price. "
            "NaN where unobserved; the model is never substituted for an observation."
        ),
    }
    if observed.size:
        premium_block |= {
            "latest_bps": round(float(observed.iloc[-1]), 1),
            "latest_as_of": observed.index[-1].date().isoformat(),
            "observed_from": observed.index[0].date().isoformat(),
        }
    block["local_premium"] = premium_block
    return block


def _factor_availability(
    asset: Asset,
    provenance: dict[str, Any],
    unavailable: dict[str, str],
) -> list[dict[str, Any]]:
    """Which drivers resolved, and why any did not.

    A factor that silently vanishes is worse than one that is absent, because the
    fitted loadings change without anything saying so. Optional factors are allowed
    to drop out — oil's EIA inventories will, whenever no API key is present — but
    never quietly.
    """
    out: list[dict[str, Any]] = []
    for factor in asset.factor_set:
        resolved = factor.series_id in provenance
        entry: dict[str, Any] = {
            "id": factor.id,
            "series_id": factor.series_id,
            "required": factor.required,
            "available": resolved,
        }
        if not resolved:
            entry["reason"] = unavailable.get(
                factor.series_id, "no source is registered for this series"
            )
        out.append(entry)
    return out


def _build_artifact(
    *,
    active: Sequence[Asset],
    results: dict[str, AssetResult],
    provenance: dict[str, Any],
    unavailable: dict[str, str],
    offline: bool,
) -> dict[str, Any]:
    breaks = load_policy_breaks()

    # Passthrough is diagnosed on whichever lens carries an observed reference rate.
    diagnostic_premium = pd.Series(dtype=float)
    for result in results.values():
        for lens_result in result.lenses.values():
            if lens_result.premium.dropna().size:
                diagnostic_premium = lens_result.premium
                break

    assets_block = {
        asset.id: {
            "asset": asset.describe(),
            "lenses": {
                code: _lens_block(asset, lens_result)
                for code, lens_result in results[asset.id].lenses.items()
            },
            "factors": _factor_availability(asset, provenance, unavailable),
        }
        for asset in active
        if asset.id in results
    }

    return {
        "schema_version": 2,
        "generated_at": datetime.now(UTC).isoformat(),
        "engine_version": __version__,
        "mode": "offline" if offline else "live",
        "assets": assets_block,
        "policy_breaks": [
            {
                "date": b.date.isoformat(),
                "kind": b.kind,
                "description": b.description,
                "expected_effect": b.expected_effect,
                "source_url": b.source_url,
                "passthrough": (
                    passthrough_diagnostic(diagnostic_premium, b.date)
                    if diagnostic_premium.dropna().size
                    else None
                ),
            }
            for b in breaks
        ],
        "sources": provenance,
        "unavailable": unavailable,
        "disclaimer": (
            "Aurex is a research and education tool. It produces probability "
            "distributions, not advice. Short-horizon price direction is not reliably "
            "forecastable, and nothing here changes that."
        ),
    }


def write_artifact(artifact: dict[str, Any], directory: Path | None = None) -> Path:
    """Write ``latest.json``. The dated forecast log arrives with step 3."""
    target_dir = directory or PUBLIC_DATA_DIR
    target_dir.mkdir(parents=True, exist_ok=True)
    path = target_dir / "latest.json"
    path.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n")
    return path
