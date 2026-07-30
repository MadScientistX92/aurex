"""Source-priority resolution: preference order, fallback, and honest provenance."""

from __future__ import annotations

from datetime import date

import pytest

from aurex.data.base import DataUnavailableError
from aurex.data.cache import CacheStore
from aurex.data.chain import SourceChain
from tests.conftest import StubLoader, make_series

START, END = date(2026, 1, 1), date(2026, 2, 1)


class TestPreferenceOrder:
    def test_first_source_wins_and_records_no_fallbacks(self, cache: CacheStore) -> None:
        primary = StubLoader("xauusd", "yfinance:GC=F", result=make_series("xauusd", value=4000.0))
        secondary = StubLoader("xauusd", "LBMA", result=make_series("xauusd", value=1.0))

        loaded = SourceChain("xauusd", [primary, secondary], cache).load(START, END)

        assert loaded.meta.source_name == "yfinance:GC=F"
        assert loaded.meta.fallbacks == ()
        assert secondary.calls == 0

    def test_failure_falls_through_and_records_the_reason(self, cache: CacheStore) -> None:
        """Yahoo rate-limiting is the real case this exists for."""
        primary = StubLoader(
            "xauusd", "yfinance:GC=F", error=RuntimeError("HTTP 429 Too Many Requests")
        )
        secondary = StubLoader("xauusd", "LBMA", result=make_series("xauusd", value=4000.0))

        loaded = SourceChain("xauusd", [primary, secondary], cache).load(START, END)

        assert loaded.meta.source_name == "LBMA"
        assert len(loaded.meta.fallbacks) == 1
        assert "yfinance:GC=F" in loaded.meta.fallbacks[0]
        assert "429" in loaded.meta.fallbacks[0]

    def test_successful_fetch_populates_the_cache(self, cache: CacheStore) -> None:
        loader = StubLoader("vix", "yfinance:^VIX", result=make_series("vix", value=20.0))
        SourceChain("vix", [loader], cache).load(START, END)
        assert cache.has("vix")


class TestExhaustion:
    def test_all_sources_failing_serves_cache_with_a_note(self, cache: CacheStore) -> None:
        cache.write(make_series("usdinr", value=96.0))
        failing = StubLoader("usdinr", "yfinance:INR=X", error=RuntimeError("boom"))

        loaded = SourceChain("usdinr", [failing], cache).load(START, END)

        assert loaded.frame["close"].iloc[0] == 96.0
        assert any("cache" in note for note in loaded.meta.fallbacks)

    def test_all_sources_failing_with_no_cache_raises(self, cache: CacheStore) -> None:
        failing = StubLoader("wti", "FRED:DCOILWTICO", error=RuntimeError("boom"))
        with pytest.raises(DataUnavailableError, match="all sources failed"):
            SourceChain("wti", [failing], cache).load(START, END)

    def test_the_error_names_every_source_tried(self, cache: CacheStore) -> None:
        loaders = [
            StubLoader("dxy", "alpha", error=RuntimeError("no")),
            StubLoader("dxy", "beta", error=ValueError("also no")),
        ]
        with pytest.raises(DataUnavailableError) as excinfo:
            SourceChain("dxy", loaders, cache).load(START, END)
        assert "alpha" in str(excinfo.value)
        assert "beta" in str(excinfo.value)


class TestOfflineMode:
    def test_offline_reads_cache_without_touching_a_loader(self, cache: CacheStore) -> None:
        cache.write(make_series("vix", value=18.0))
        loader = StubLoader("vix", "yfinance:^VIX", result=make_series("vix", value=99.0))

        loaded = SourceChain("vix", [loader], cache).load(START, END, offline=True)

        assert loaded.frame["close"].iloc[0] == 18.0
        assert loader.calls == 0

    def test_offline_without_cache_raises(self, cache: CacheStore) -> None:
        loader = StubLoader("vix", "yfinance:^VIX", result=make_series("vix"))
        with pytest.raises(DataUnavailableError, match="offline mode"):
            SourceChain("vix", [loader], cache).load(START, END, offline=True)


def test_chain_requires_at_least_one_loader(cache: CacheStore) -> None:
    with pytest.raises(ValueError, match="at least one loader"):
        SourceChain("empty", [], cache)


class TestRegistry:
    def test_every_declared_series_builds_a_chain(self) -> None:
        from aurex.data.registry import SERIES_IDS, all_chains

        chains = all_chains(CacheStore())
        assert set(chains) == set(SERIES_IDS)
        for series_id, chain in chains.items():
            assert chain.loaders, f"{series_id} has no loaders"

    def test_parity_gold_series_is_spot_not_futures(self) -> None:
        """Regression guard for a real contamination bug.

        `GC=F` is the COMEX front-month future, which trades over spot by the cost
        of carry (+2.40% vs the London PM fix on 2026-07-29). Sourcing parity from
        it pushes that basis straight into `local_premium_bps` — and because the
        basis varies with rates and time to expiry, it would look like a moving
        domestic-demand signal. Parity must come from the spot fix.
        """
        from aurex.data.registry import chain_for

        names = [loader.source_name for loader in chain_for("xauusd").loaders]
        assert all("GC=F" not in name for name in names), "parity must not use futures"
        assert any("LBMA" in name for name in names)

    def test_futures_are_still_available_for_the_volatility_layer(self) -> None:
        """Step 2 wants true OHLC, which the close-only fix cannot provide."""
        from aurex.data.registry import chain_for

        names = [loader.source_name for loader in chain_for("xau_futures").loaders]
        assert any("GC=F" in name for name in names)

    def test_unknown_series_is_rejected(self) -> None:
        from aurex.data.registry import chain_for

        with pytest.raises(KeyError):
            chain_for("dogecoin")
