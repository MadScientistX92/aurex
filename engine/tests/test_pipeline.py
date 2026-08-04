"""End-to-end pipeline, offline from a seeded cache."""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from aurex.assets import GOLD
from aurex.data.base import LoadedSeries, SeriesMeta
from aurex.data.cache import CacheStore
from aurex.pipeline import _price_column, run, write_artifact, write_forecast_log
from tests.conftest import make_series

START, END = date(2026, 6, 1), date(2026, 7, 30)


def ibja_series(index: pd.DatetimeIndex, level: float = 142_000.0) -> LoadedSeries:
    frame = pd.DataFrame(
        {
            "gold_999_pm": np.full(len(index), level),
            "spdr_gold_tonnes": np.full(len(index), 1008.73),
        },
        index=index,
    )
    return LoadedSeries(
        frame=frame,
        meta=SeriesMeta(
            series_id="ibja_gold",
            source_name="IBJA:daily-bullion-report",
            source_url="https://www.ibja.co/Upload/x.pdf",
            fetched_at=pd.Timestamp("2026-07-31", tz="UTC").to_pydatetime(),
            rows=len(frame),
            start=index[0].date(),
            end=index[-1].date(),
        ),
    )


@pytest.fixture
def seeded_cache(cache: CacheStore) -> CacheStore:
    index = pd.bdate_range(START, END, name="date")
    cache.write(make_series("xauusd", start=str(START), periods=len(index), value=4000.0))
    cache.write(make_series("usdinr", start=str(START), periods=len(index), value=96.5))
    cache.write(ibja_series(index))
    return cache


def gold_block(cache: CacheStore) -> dict:
    return run(offline=True, start=START, end=END, cache=cache).artifact["assets"]["gold"]


class TestOfflineRun:
    def test_runs_offline_and_produces_an_artifact(self, seeded_cache: CacheStore) -> None:
        artifact = run(offline=True, start=START, end=END, cache=seeded_cache).artifact
        assert artifact["mode"] == "offline"
        assert artifact["schema_version"] == 3

    def test_emits_one_block_per_lens(self, seeded_cache: CacheStore) -> None:
        assert set(gold_block(seeded_cache)["lenses"]) == {"USD", "INR"}

    def test_records_provenance_for_every_resolved_series(self, seeded_cache: CacheStore) -> None:
        artifact = run(offline=True, start=START, end=END, cache=seeded_cache).artifact
        for series_id, meta in artifact["sources"].items():
            assert meta["source_url"], f"{series_id} has no source_url"
            assert meta["fetched_at"], f"{series_id} has no fetched_at"
            assert meta["source_name"], f"{series_id} has no source_name"

    def test_unresolvable_series_are_reported_not_silently_dropped(
        self, seeded_cache: CacheStore
    ) -> None:
        artifact = run(offline=True, start=START, end=END, cache=seeded_cache).artifact
        assert set(artifact["sources"]) == {"xauusd", "usdinr", "ibja_gold"}
        assert "vix" in artifact["unavailable"]
        assert "real_yield_10y" in artifact["unavailable"]

    def test_missing_factors_state_a_reason(self, seeded_cache: CacheStore) -> None:
        factors = {f["id"]: f for f in gold_block(seeded_cache)["factors"]}
        assert factors["d_real_yield"]["available"] is False
        assert factors["d_real_yield"]["reason"]

    def test_geopolitical_risk_factor_is_declared_optional(self, seeded_cache: CacheStore) -> None:
        """Present so the safe-haven channel is estimated rather than omitted; it
        has no loader yet, so it must degrade rather than fail the run."""
        factors = {f["id"]: f for f in gold_block(seeded_cache)["factors"]}
        assert factors["d_geopolitical_risk"]["required"] is False
        assert factors["d_geopolitical_risk"]["available"] is False


class TestLensBlocks:
    def test_inr_lens_carries_duty_and_rate_provenance(self, seeded_cache: CacheStore) -> None:
        latest = gold_block(seeded_cache)["lenses"]["INR"]["latest"]
        assert latest["duty"] == pytest.approx(0.15)
        assert latest["consumption_tax"] == pytest.approx(0.03)
        assert latest["confidence"] == "high"
        assert latest["currency"] == "INR"
        assert latest["unit"] == "10g"

        provenance = latest["rate_provenance"]
        assert provenance["duty"]["source_url"].startswith("http")
        assert provenance["duty"]["source_confidence"] in {"primary", "secondary"}
        assert provenance["consumption_tax"]["source_url"].startswith("http")

    def test_usd_lens_is_the_untaxed_quote(self, seeded_cache: CacheStore) -> None:
        latest = gold_block(seeded_cache)["lenses"]["USD"]["latest"]
        assert latest["price"] == pytest.approx(4000.0)
        assert latest["duty"] == 0.0
        assert latest["consumption_tax"] == 0.0
        assert latest["unit"] == "troy_ounce"

    def test_every_lens_block_carries_a_provenance_key(self, seeded_cache: CacheStore) -> None:
        """Empty for an untaxed lens, but never absent.

        The key is emitted from a protocol method, so a lens cannot omit provenance
        by failing to advertise it — an absent key and no applicable rates would
        otherwise be indistinguishable downstream.
        """
        lenses = gold_block(seeded_cache)["lenses"]
        assert all("rate_provenance" in block["latest"] for block in lenses.values())
        assert lenses["USD"]["latest"]["rate_provenance"] == {}

    def test_usd_lens_reports_no_local_premium(self, seeded_cache: CacheStore) -> None:
        """§15: the local-premium signal has no USD analogue."""
        assert gold_block(seeded_cache)["lenses"]["USD"]["local_premium"] is None

    def test_inr_premium_is_measured(self, seeded_cache: CacheStore) -> None:
        premium = gold_block(seeded_cache)["lenses"]["INR"]["local_premium"]
        assert premium["observations"] > 0
        assert "latest_bps" in premium

    def test_premium_absent_when_reference_rate_is_not_observed(self, cache: CacheStore) -> None:
        index = pd.bdate_range(START, END, name="date")
        cache.write(make_series("xauusd", start=str(START), periods=len(index), value=4000.0))
        cache.write(make_series("usdinr", start=str(START), periods=len(index), value=96.5))

        premium = gold_block(cache)["lenses"]["INR"]["local_premium"]
        assert premium["observations"] == 0
        assert "latest_bps" not in premium

    def test_inr_lens_skipped_when_fx_is_unavailable(self, cache: CacheStore) -> None:
        """A lens missing its FX must drop out, not crash or invent a rate."""
        index = pd.bdate_range(START, END, name="date")
        cache.write(make_series("xauusd", start=str(START), periods=len(index), value=4000.0))

        lenses = gold_block(cache)["lenses"]
        assert "USD" in lenses
        assert "INR" not in lenses


class TestMisconfigurationFailsLoudly:
    def test_a_lens_wanting_fx_without_naming_a_series_is_refused(self) -> None:
        """Silently skipping it would drop a currency view with no reason recorded."""
        import dataclasses

        from aurex.assets.lens import NativeLens
        from aurex.pipeline import _apply_lenses

        class Misconfigured:
            id = "misconfigured"
            price_series_id = "xauusd"
            reference_rate_series = None
            reference_rate_column = None
            currency_lenses = (
                dataclasses.replace(NativeLens(code="XXX", unit_label="unit"), requires_fx=True),
            )

        index = pd.bdate_range(START, periods=5, name="date")
        series = {"xauusd": make_series("xauusd", start=str(START), periods=len(index))}
        with pytest.raises(ValueError, match="requires_fx but no fx_series_id"):
            _apply_lenses(Misconfigured(), series)  # type: ignore[arg-type]

    def test_a_lens_with_no_overlapping_dates_reports_an_empty_block(
        self, cache: CacheStore
    ) -> None:
        """Price and rate that never traded on the same day produce no prices at all."""
        cache.write(make_series("xauusd", start="2026-06-01", periods=20, value=4_000.0))
        cache.write(make_series("usdinr", start="2020-01-01", periods=20, value=96.5))

        block = gold_block(cache)["lenses"]["INR"]
        assert block["latest"] is None
        assert block["local_premium"] is None
        assert block["distribution"] is None


class TestArtifactShape:
    def test_policy_breaks_are_emitted(self, seeded_cache: CacheStore) -> None:
        breaks = run(offline=True, start=START, end=END, cache=seeded_cache).artifact[
            "policy_breaks"
        ]
        dates = {b["date"] for b in breaks}
        assert {"2026-05-13", "2017-07-01"} <= dates
        for entry in breaks:
            assert entry["source_url"].startswith("http")

    def test_asset_description_is_included(self, seeded_cache: CacheStore) -> None:
        described = gold_block(seeded_cache)["asset"]
        assert described["quote_currency"] == "USD"
        assert described["base_unit"] == "troy_ounce"
        assert described["return_transform"]["id"] == "log"
        assert "inr_retail" in described["friction_profiles"]

    def test_disclaimer_is_present_verbatim(self, seeded_cache: CacheStore) -> None:
        artifact = run(offline=True, start=START, end=END, cache=seeded_cache).artifact
        assert artifact["disclaimer"].startswith("Aurex is a research and education tool")
        assert "not advice" in artifact["disclaimer"]

    def test_empty_cache_yields_an_artifact_rather_than_crashing(self, cache: CacheStore) -> None:
        artifact = run(offline=True, start=START, end=END, cache=cache).artifact
        assert artifact["sources"] == {}
        assert artifact["assets"]["gold"]["lenses"] == {}
        assert len(artifact["unavailable"]) > 0

    def test_restricting_assets_restricts_the_output(self, seeded_cache: CacheStore) -> None:
        artifact = run(
            offline=True, start=START, end=END, cache=seeded_cache, assets=[GOLD]
        ).artifact
        assert set(artifact["assets"]) == {"gold"}


class TestArtifactWriting:
    def test_writes_valid_json(self, seeded_cache: CacheStore, tmp_path: Path) -> None:
        artifact = run(offline=True, start=START, end=END, cache=seeded_cache).artifact
        path = write_artifact(artifact, tmp_path)
        assert path.name == "latest.json"
        assert json.loads(path.read_text())["schema_version"] == 3

    def test_creates_the_directory(self, seeded_cache: CacheStore, tmp_path: Path) -> None:
        artifact = run(offline=True, start=START, end=END, cache=seeded_cache).artifact
        assert write_artifact(artifact, tmp_path / "nested" / "dir").exists()


class TestForecastLog:
    """§0's second rule needs a forecast that still exists in the form it was published."""

    def test_the_log_is_dated_and_separate_from_latest(
        self, seeded_cache: CacheStore, tmp_path: Path
    ) -> None:
        artifact = run(offline=True, start=START, end=END, cache=seeded_cache).artifact
        path = write_forecast_log(artifact, tmp_path)

        assert path.parent.name == "forecasts"
        assert path.name == f"{artifact['generated_at'][:10]}.json"
        assert path != write_artifact(artifact, tmp_path)

    def test_the_logged_copy_is_the_published_artifact_verbatim(
        self, seeded_cache: CacheStore, tmp_path: Path
    ) -> None:
        """A log that summarises rather than copies cannot be scored against later.

        The seed, the fitted parameters and the block length all live in the
        distribution block; a log that dropped any of them would leave a forecast
        nobody can regenerate, which §2 says is the same as no forecast at all. The
        fixture window is too short to fit a model, so what this pins is the copy
        being whole — including the recorded reason when there is no distribution.
        """
        artifact = run(offline=True, start=START, end=END, cache=seeded_cache).artifact
        logged = json.loads(write_forecast_log(artifact, tmp_path).read_text())

        assert logged == json.loads(json.dumps(artifact))
        assert "distribution" in logged["assets"][GOLD.id]["lenses"]["USD"]

    def test_rerunning_the_same_day_replaces_that_day(
        self, seeded_cache: CacheStore, tmp_path: Path
    ) -> None:
        artifact = run(offline=True, start=START, end=END, cache=seeded_cache).artifact
        first = write_forecast_log(artifact, tmp_path)
        second = write_forecast_log(artifact, tmp_path)

        assert first == second
        assert len(list((tmp_path / "forecasts").glob("*.json"))) == 1


class TestPriceColumn:
    def test_prefers_named_price_columns(self) -> None:
        index = pd.bdate_range("2026-01-01", periods=2)
        frame = pd.DataFrame({"open": [1.0, 2.0], "close": [3.0, 4.0]}, index=index)
        assert _price_column(frame).tolist() == [3.0, 4.0]

    def test_falls_back_to_the_first_numeric_column(self) -> None:
        index = pd.bdate_range("2026-01-01", periods=2)
        frame = pd.DataFrame({"real_yield": [1.8, 1.9]}, index=index)
        assert _price_column(frame).tolist() == [1.8, 1.9]

    def test_raises_when_there_is_nothing_numeric(self) -> None:
        index = pd.bdate_range("2026-01-01", periods=2)
        frame = pd.DataFrame({"label": ["a", "b"]}, index=index)
        with pytest.raises(ValueError, match="no numeric column"):
            _price_column(frame)
