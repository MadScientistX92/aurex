"""End-to-end pipeline, offline from a seeded cache."""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from aurex.data.base import LoadedSeries, SeriesMeta
from aurex.data.cache import CacheStore
from aurex.pipeline import _close_column, run, write_artifact
from tests.conftest import make_series

START, END = date(2026, 6, 1), date(2026, 7, 30)


def ibja_series(index: pd.DatetimeIndex, parity_like: float = 142_000.0) -> LoadedSeries:
    frame = pd.DataFrame(
        {
            "gold_999_pm": np.full(len(index), parity_like),
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


class TestOfflineRun:
    def test_runs_offline_and_produces_an_artifact(self, seeded_cache: CacheStore) -> None:
        result = run(offline=True, start=START, end=END, cache=seeded_cache)

        assert result.artifact["mode"] == "offline"
        assert result.artifact["schema_version"] == 1
        assert not result.parity.empty

    def test_records_provenance_for_every_resolved_series(self, seeded_cache: CacheStore) -> None:
        artifact = run(offline=True, start=START, end=END, cache=seeded_cache).artifact
        for series_id, meta in artifact["sources"].items():
            assert meta["source_url"], f"{series_id} has no source_url"
            assert meta["fetched_at"], f"{series_id} has no fetched_at"
            assert meta["source_name"], f"{series_id} has no source_name"

    def test_unresolvable_series_are_reported_not_silently_dropped(
        self, seeded_cache: CacheStore
    ) -> None:
        """Only three series are seeded; the rest must show up as unavailable."""
        artifact = run(offline=True, start=START, end=END, cache=seeded_cache).artifact
        assert set(artifact["sources"]) == {"xauusd", "usdinr", "ibja_gold"}
        assert "vix" in artifact["unavailable"]
        assert "real_yield_10y" in artifact["unavailable"]

    def test_parity_block_carries_duty_provenance(self, seeded_cache: CacheStore) -> None:
        parity = run(offline=True, start=START, end=END, cache=seeded_cache).artifact["parity"]
        assert parity["duty_total"] == pytest.approx(0.15)
        assert parity["duty_provenance"]["source_url"].startswith("http")
        assert parity["duty_provenance"]["source_confidence"] in {"primary", "secondary"}
        assert parity["confidence"] == "high"

    def test_policy_breaks_are_emitted_for_downstream_layers(
        self, seeded_cache: CacheStore
    ) -> None:
        """Step 2's vol fitting and step 4's regression both consume this."""
        breaks = run(offline=True, start=START, end=END, cache=seeded_cache).artifact[
            "policy_breaks"
        ]
        dates = {b["date"] for b in breaks}
        assert "2026-05-13" in dates
        assert "2017-07-01" in dates
        for entry in breaks:
            assert entry["source_url"].startswith("http")
            assert entry["kind"] in {
                "duty",
                "tax",
                "quantitative_restriction",
                "monetary",
            }

    def test_premium_block_reports_observation_count(self, seeded_cache: CacheStore) -> None:
        premium = run(offline=True, start=START, end=END, cache=seeded_cache).artifact[
            "local_premium"
        ]
        assert premium["observations"] > 0
        assert "latest_bps" in premium

    def test_premium_absent_when_ibja_is_not_observed(self, cache: CacheStore) -> None:
        """No observation means no premium — parity is never substituted in."""
        index = pd.bdate_range(START, END, name="date")
        cache.write(make_series("xauusd", start=str(START), periods=len(index), value=4000.0))
        cache.write(make_series("usdinr", start=str(START), periods=len(index), value=96.5))

        artifact = run(offline=True, start=START, end=END, cache=cache).artifact
        assert artifact["local_premium"]["observations"] == 0
        assert "latest_bps" not in artifact["local_premium"]

    def test_disclaimer_is_present_verbatim(self, seeded_cache: CacheStore) -> None:
        artifact = run(offline=True, start=START, end=END, cache=seeded_cache).artifact
        assert artifact["disclaimer"].startswith("Aurex is a research and education tool")
        assert "not advice" in artifact["disclaimer"]

    def test_empty_cache_yields_an_artifact_rather_than_crashing(self, cache: CacheStore) -> None:
        artifact = run(offline=True, start=START, end=END, cache=cache).artifact
        assert artifact["sources"] == {}
        assert artifact["parity"] == {}
        assert len(artifact["unavailable"]) > 0


class TestArtifactWriting:
    def test_writes_valid_json(self, seeded_cache: CacheStore, tmp_path: Path) -> None:
        artifact = run(offline=True, start=START, end=END, cache=seeded_cache).artifact
        path = write_artifact(artifact, tmp_path)

        assert path.name == "latest.json"
        reloaded = json.loads(path.read_text())
        assert reloaded["schema_version"] == 1

    def test_creates_the_directory(self, seeded_cache: CacheStore, tmp_path: Path) -> None:
        artifact = run(offline=True, start=START, end=END, cache=seeded_cache).artifact
        path = write_artifact(artifact, tmp_path / "nested" / "dir")
        assert path.exists()


class TestCloseColumn:
    def test_prefers_named_price_columns(self) -> None:
        index = pd.bdate_range("2026-01-01", periods=2)
        frame = pd.DataFrame({"open": [1.0, 2.0], "close": [3.0, 4.0]}, index=index)
        assert _close_column(frame).tolist() == [3.0, 4.0]

    def test_falls_back_to_the_first_numeric_column(self) -> None:
        index = pd.bdate_range("2026-01-01", periods=2)
        frame = pd.DataFrame({"real_yield": [1.8, 1.9]}, index=index)
        assert _close_column(frame).tolist() == [1.8, 1.9]

    def test_raises_when_there_is_nothing_numeric(self) -> None:
        index = pd.bdate_range("2026-01-01", periods=2)
        frame = pd.DataFrame({"label": ["a", "b"]}, index=index)
        with pytest.raises(ValueError, match="no numeric column"):
            _close_column(frame)
