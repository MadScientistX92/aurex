"""Nightly orchestrator.

Step 1 scope: resolve every series, compute parity and the domestic premium, and emit
the artifact with full provenance. No forecasting happens yet — the volatility and
distribution layers land in step 2. There are deliberately no point estimates in the
output and there never will be.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any

import pandas as pd

from aurex import __version__
from aurex.config import PUBLIC_DATA_DIR
from aurex.data.base import DataUnavailableError, LoadedSeries
from aurex.data.cache import CacheStore
from aurex.data.parity import compute_parity, local_premium_bps, passthrough_diagnostic
from aurex.data.registry import all_chains
from aurex.data.schedules import duty_on, gst_on, load_policy_breaks

log = logging.getLogger(__name__)

#: Twenty years of history, per the spec's data requirements.
DEFAULT_LOOKBACK_DAYS = 365 * 20


@dataclass(frozen=True, slots=True)
class PipelineResult:
    artifact: dict[str, Any]
    series: dict[str, LoadedSeries]
    parity: pd.DataFrame
    premium: pd.Series


def _close_column(frame: pd.DataFrame) -> pd.Series:
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
) -> PipelineResult:
    """Resolve data, compute parity, and assemble the artifact."""
    end = end or datetime.now(UTC).date()
    start = start or (end - timedelta(days=DEFAULT_LOOKBACK_DAYS))
    store = cache or CacheStore()

    series: dict[str, LoadedSeries] = {}
    provenance: dict[str, Any] = {}
    unavailable: dict[str, str] = {}

    for series_id, chain in all_chains(store).items():
        try:
            loaded = chain.load(start, end, offline=offline)
        except DataUnavailableError as exc:
            log.warning("%s unavailable: %s", series_id, exc)
            unavailable[series_id] = str(exc)
            continue
        series[series_id] = loaded
        provenance[series_id] = loaded.meta.to_dict()

    parity = pd.DataFrame()
    premium = pd.Series(dtype=float)
    if "xauusd" in series and "usdinr" in series:
        parity = compute_parity(
            _close_column(series["xauusd"].frame),
            _close_column(series["usdinr"].frame),
        )

    if not parity.empty and "ibja_gold" in series:
        ibja = series["ibja_gold"].frame
        if "gold_999_pm" in ibja.columns:
            premium = local_premium_bps(ibja["gold_999_pm"], parity["parity_ex_gst"])

    artifact = _build_artifact(
        parity=parity,
        premium=premium,
        provenance=provenance,
        unavailable=unavailable,
        offline=offline,
    )
    return PipelineResult(artifact=artifact, series=series, parity=parity, premium=premium)


def _build_artifact(
    *,
    parity: pd.DataFrame,
    premium: pd.Series,
    provenance: dict[str, Any],
    unavailable: dict[str, str],
    offline: bool,
) -> dict[str, Any]:
    breaks = load_policy_breaks()
    observed = premium.dropna()

    latest: dict[str, Any] = {}
    if not parity.empty:
        last_stamp = parity.index[-1]
        row = parity.iloc[-1]
        day = last_stamp.date()
        duty = duty_on(day)
        gst = gst_on(day)
        latest = {
            "as_of": day.isoformat(),
            "parity_ex_gst_inr_per_10g": round(float(row["parity_ex_gst"]), 2),
            "parity_incl_gst_inr_per_10g": round(float(row["parity_incl_gst"]), 2),
            "duty_total": float(row["duty_total"]),
            "gst_metal": float(row["gst_metal"]),
            "confidence": str(row["confidence"]),
            "duty_provenance": {
                "effective_from": duty.effective_from.isoformat() if duty else None,
                "source_url": duty.source_url if duty else None,
                "source_confidence": duty.source_confidence if duty else None,
            },
            "gst_provenance": {
                "effective_from": gst.effective_from.isoformat() if gst else None,
                "source_url": gst.source_url if gst else None,
                "source_confidence": gst.source_confidence if gst else None,
            },
        }

    premium_block: dict[str, Any] = {
        "observations": int(observed.size),
        "note": (
            "Premium is observed IBJA 999 (GST-exclusive) over duty-inclusive parity. "
            "NaN where no IBJA observation exists; parity is never substituted for an "
            "observation."
        ),
    }
    if observed.size:
        premium_block |= {
            "latest_bps": round(float(observed.iloc[-1]), 1),
            "latest_as_of": observed.index[-1].date().isoformat(),
            "observed_from": observed.index[0].date().isoformat(),
        }

    return {
        "schema_version": 1,
        "generated_at": datetime.now(UTC).isoformat(),
        "engine_version": __version__,
        "mode": "offline" if offline else "live",
        "parity": latest,
        "local_premium": premium_block,
        "policy_breaks": [
            {
                "date": b.date.isoformat(),
                "kind": b.kind,
                "description": b.description,
                "expected_effect": b.expected_effect,
                "source_url": b.source_url,
                "passthrough": (passthrough_diagnostic(premium, b.date) if observed.size else None),
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
