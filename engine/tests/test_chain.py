"""Source-priority resolution: preference order, fallback, and honest provenance."""

from __future__ import annotations

from datetime import date

import pandas as pd
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


class TestTheWindowBoundsWhatComesBack:
    """``load(start, end)`` returns the window, not the cache.

    The cache is a union of every fetch ever made on one machine, and serving that
    union made the resolved sample a property of the machine rather than of the
    command. One published ``aurex score`` command returned three different sample
    starts on three runs — 2006-08-04, then 2000-01-04 on the same Mac after its cache
    was extended backwards, then 2005-01-04 on a cacheless runner — and every
    expanding-window fit behind the published numbers moved with it.

    The clip applies on every path out of :meth:`SourceChain.load`, because a caller
    cannot tell which one answered and the sample must not depend on that.
    """

    def test_a_live_fetch_is_clipped_to_the_window(self, cache: CacheStore) -> None:
        cache.write(make_series("xauusd", start="2020-01-01", periods=200, value=1_800.0))
        fresh = make_series("xauusd", start="2026-01-01", periods=30, value=4_000.0)

        loaded = SourceChain("xauusd", [StubLoader("xauusd", "LBMA", result=fresh)], cache).load(
            START, END
        )

        assert loaded.frame.index.min() >= pd.Timestamp(START)
        assert loaded.frame.index.max() <= pd.Timestamp(END)
        assert (loaded.frame["close"] == 4_000.0).all(), "2020 history leaked into the window"

    def test_the_cache_still_accumulates_what_the_window_excludes(self, cache: CacheStore) -> None:
        """Clipping the return value must not clip the store.

        History the cache has gathered is what makes a *wider* request cheap, and a
        daily report carries only a few days of it. Discarding it to satisfy this
        window would trade one machine-dependent sample for a lost one.
        """
        cache.write(make_series("xauusd", start="2020-01-01", periods=200, value=1_800.0))
        fresh = make_series("xauusd", start="2026-01-01", periods=30, value=4_000.0)

        SourceChain("xauusd", [StubLoader("xauusd", "LBMA", result=fresh)], cache).load(START, END)

        kept = cache.read("xauusd")
        assert kept is not None
        assert kept.meta.start == date(2020, 1, 1)
        assert kept.meta.rows == 230

    def test_meta_describes_the_frame_that_was_returned(self, cache: CacheStore) -> None:
        """Provenance for a wider frame than the one handed over is provenance for
        something the caller never saw."""
        cache.write(make_series("xauusd", start="2020-01-01", periods=200, value=1_800.0))
        fresh = make_series("xauusd", start="2026-01-01", periods=30, value=4_000.0)

        loaded = SourceChain("xauusd", [StubLoader("xauusd", "LBMA", result=fresh)], cache).load(
            START, END
        )

        assert loaded.meta.rows == len(loaded.frame)
        assert loaded.meta.start == loaded.frame.index[0].date()
        assert loaded.meta.end == loaded.frame.index[-1].date()

    def test_a_source_answering_outside_the_window_falls_through(self, cache: CacheStore) -> None:
        """Answering with nothing in the window is a decline, not an answer."""
        stale = StubLoader(
            "xauusd", "elsewhere", result=make_series("xauusd", start="2020-01-01", periods=10)
        )
        good = StubLoader(
            "xauusd",
            "LBMA",
            result=make_series("xauusd", start="2026-01-01", periods=30, value=4_000.0),
        )

        loaded = SourceChain("xauusd", [stale, good], cache).load(START, END)

        assert loaded.meta.source_name == "LBMA"
        assert any("no observations in" in note for note in loaded.meta.fallbacks)

    def test_the_cache_fallback_is_clipped_too(self, cache: CacheStore) -> None:
        cache.write(make_series("usdinr", start="2020-01-01", periods=200, value=75.0))
        cache.merge_write(make_series("usdinr", start="2026-01-01", periods=30, value=96.0))
        failing = StubLoader("usdinr", "yfinance:INR=X", error=RuntimeError("boom"))

        loaded = SourceChain("usdinr", [failing], cache).load(START, END)

        assert (loaded.frame["close"] == 96.0).all()
        assert loaded.frame.index.min() >= pd.Timestamp(START)

    def test_a_cache_holding_nothing_in_the_window_is_not_an_answer(
        self, cache: CacheStore
    ) -> None:
        """The old behaviour served this quietly, and it is how a lens ended up
        pricing 2026 gold against a 2020 exchange rate without anything saying so."""
        cache.write(make_series("usdinr", start="2020-01-01", periods=200, value=75.0))
        failing = StubLoader("usdinr", "yfinance:INR=X", error=RuntimeError("boom"))

        with pytest.raises(DataUnavailableError, match="all sources failed"):
            SourceChain("usdinr", [failing], cache).load(START, END)


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

    def test_offline_is_clipped_to_the_window(self, cache: CacheStore) -> None:
        """The offline and live paths must grade the same sample.

        They did not: offline returned ``cache.read(series_id)`` whole, with ``start``
        and ``end`` ignored, so an offline run against a deep local cache saw 6,662
        observations where the identical command run live on a runner saw 5,408. An
        offline artifact and a live artifact with identical flags were not comparable,
        and nothing in either said so.
        """
        cache.write(make_series("vix", start="2020-01-01", periods=200, value=15.0))
        cache.merge_write(make_series("vix", start="2026-01-01", periods=30, value=20.0))
        loader = StubLoader("vix", "yfinance:^VIX", result=make_series("vix", value=99.0))

        loaded = SourceChain("vix", [loader], cache).load(START, END, offline=True)

        assert (loaded.frame["close"] == 20.0).all()
        assert loaded.meta.rows == len(loaded.frame)
        assert loader.calls == 0

    def test_offline_with_a_cache_outside_the_window_raises(self, cache: CacheStore) -> None:
        cache.write(make_series("vix", start="2020-01-01", periods=200, value=15.0))
        loader = StubLoader("vix", "yfinance:^VIX", result=make_series("vix"))

        with pytest.raises(DataUnavailableError, match="holds nothing in"):
            SourceChain("vix", [loader], cache).load(START, END, offline=True)


def test_chain_requires_at_least_one_loader(cache: CacheStore) -> None:
    with pytest.raises(ValueError, match="at least one loader"):
        SourceChain("empty", [], cache)


class TestRegistry:
    def test_asset_series_and_referenced_macro_both_resolve(self) -> None:
        from aurex.assets import GOLD
        from aurex.data.registry import chains_for

        chains = chains_for([GOLD], CacheStore())
        assert {"xauusd", "usdinr", "ibja_gold", "xau_futures"} <= set(chains)
        # Macro series the factor set references.
        assert {"dxy", "wti", "vix", "real_yield_10y"} <= set(chains)
        for series_id, chain in chains.items():
            assert chain.loaders, f"{series_id} has no loaders"

    def test_unreferenced_macro_series_are_not_loaded(self) -> None:
        """A self-contained asset must not drag in shared macro series."""
        from aurex.assets.synthetic import SYNTHETIC
        from aurex.data.registry import chains_for

        chains = chains_for([SYNTHETIC], CacheStore())
        assert set(chains) == {"widget_price", "widget_fx", "widget_local"}

    def test_unknown_series_is_rejected(self) -> None:
        from aurex.assets import GOLD
        from aurex.data.registry import chain_for

        with pytest.raises(KeyError, match="unknown series"):
            chain_for("dogecoin", [GOLD], CacheStore())

    def test_parity_gold_series_is_spot_not_futures(self) -> None:
        """Regression guard for a real contamination bug.

        `GC=F` is the COMEX front-month future, which trades over spot by the cost
        of carry (+2.40% vs the London PM fix on 2026-07-29). Sourcing parity from
        it pushes that basis straight into `local_premium_bps` — and because the
        basis varies with rates and time to expiry, it would look like a moving
        domestic-demand signal. Parity must come from the spot fix.
        """
        from aurex.assets import GOLD

        sources = GOLD.price_sources(CacheStore())
        names = [loader.source_name for loader in sources["xauusd"].loaders]
        assert all("GC=F" not in name for name in names), "parity must not use futures"
        assert any("LBMA" in name for name in names)

    def test_futures_are_still_available_for_the_volatility_layer(self) -> None:
        """Step 2 wants true OHLC, which the close-only fix cannot provide."""
        from aurex.assets import GOLD

        sources = GOLD.price_sources(CacheStore())
        names = [loader.source_name for loader in sources["xau_futures"].loaders]
        assert any("GC=F" in name for name in names)
