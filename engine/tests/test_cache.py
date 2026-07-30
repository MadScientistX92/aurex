"""Parquet cache round-trips, merging, and corruption tolerance."""

from __future__ import annotations

import json

import pandas as pd

from aurex.data.cache import CacheStore
from tests.conftest import make_series


class TestRoundTrip:
    def test_write_then_read_preserves_frame_and_meta(self, cache: CacheStore) -> None:
        series = make_series("xauusd", value=2400.0)
        cache.write(series)

        restored = cache.read("xauusd")
        assert restored is not None
        # check_freq=False: parquet does not round-trip the index's inferred
        # frequency, which is metadata rather than data.
        pd.testing.assert_frame_equal(restored.frame, series.frame, check_freq=False)
        assert restored.meta.source_name == series.meta.source_name
        assert restored.meta.source_url == series.meta.source_url
        assert restored.meta.fetched_at == series.meta.fetched_at
        assert restored.meta.has_ohlc == series.meta.has_ohlc

    def test_missing_series_reads_as_none(self, cache: CacheStore) -> None:
        assert cache.read("nope") is None
        assert not cache.has("nope")

    def test_has_reports_presence(self, cache: CacheStore) -> None:
        cache.write(make_series("vix"))
        assert cache.has("vix")

    def test_write_is_idempotent(self, cache: CacheStore) -> None:
        cache.write(make_series("dxy", value=100.0))
        cache.write(make_series("dxy", value=101.0))
        restored = cache.read("dxy")
        assert restored is not None
        assert restored.frame["close"].iloc[0] == 101.0


class TestMerge:
    def test_merge_extends_history_rather_than_truncating(self, cache: CacheStore) -> None:
        """A short fetch must never shorten history we already hold — LBMA reaches
        1968 while a Yahoo call may only cover a year."""
        long_history = make_series("xauusd", start="2020-01-01", periods=500, value=1800.0)
        cache.write(long_history)

        short_recent = make_series("xauusd", start="2026-01-01", periods=10, value=4000.0)
        merged = cache.merge_write(short_recent)

        assert merged.frame.index[0] == long_history.frame.index[0]
        assert merged.frame.index[-1] == short_recent.frame.index[-1]
        assert len(merged.frame) > len(short_recent.frame)

    def test_fresh_rows_win_on_overlap(self, cache: CacheStore) -> None:
        cache.write(make_series("usdinr", start="2026-01-01", periods=10, value=83.0))
        merged = cache.merge_write(
            make_series("usdinr", start="2026-01-01", periods=10, value=96.0)
        )
        assert (merged.frame["close"] == 96.0).all()

    def test_merge_into_empty_cache_just_writes(self, cache: CacheStore) -> None:
        series = make_series("wti", value=70.0)
        merged = cache.merge_write(series)
        assert len(merged.frame) == len(series.frame)
        assert cache.has("wti")

    def test_has_ohlc_degrades_when_either_side_lacks_it(self, cache: CacheStore) -> None:
        """A close-only fallback must not inherit an OHLC claim from the cache —
        the realised-vol estimators read this flag."""
        cache.write(make_series("xauusd", periods=10, has_ohlc=True))
        merged = cache.merge_write(
            make_series("xauusd", start="2026-03-01", periods=5, has_ohlc=False)
        )
        assert merged.meta.has_ohlc is False

    def test_merged_meta_row_count_matches_frame(self, cache: CacheStore) -> None:
        cache.write(make_series("vix", start="2026-01-01", periods=20))
        merged = cache.merge_write(make_series("vix", start="2026-02-01", periods=20))
        assert merged.meta.rows == len(merged.frame)


class TestCorruption:
    def test_corrupt_metadata_reads_as_miss(self, cache: CacheStore) -> None:
        """A bad cache entry must not take down the pipeline; it falls through to
        a live source instead."""
        cache.write(make_series("vix"))
        (cache.root / "vix.meta.json").write_text("{not json")
        assert cache.read("vix") is None

    def test_metadata_missing_keys_reads_as_miss(self, cache: CacheStore) -> None:
        cache.write(make_series("vix"))
        (cache.root / "vix.meta.json").write_text(json.dumps({"series_id": "vix"}))
        assert cache.read("vix") is None

    def test_corrupt_parquet_reads_as_miss(self, cache: CacheStore) -> None:
        cache.write(make_series("vix"))
        (cache.root / "vix.parquet").write_bytes(b"definitely not parquet")
        assert cache.read("vix") is None
