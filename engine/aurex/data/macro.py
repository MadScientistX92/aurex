"""Shared macro series.

These are global economic series rather than any one asset's property — the dollar,
real yields and equity volatility drive gold and crude alike. Assets reference them
by ``series_id`` in their ``factor_set``; they are loaded once per run and shared.

Crude appears here as a *factor* (an inflation-expectations proxy for gold) and will
also appear as an *asset* when the oil module lands. That is not a conflict: the same
series can be a driver of one thing and the subject of another. The asset registry
and this table are merged by ``series_id``, so it is loaded once either way.
"""

from __future__ import annotations

from collections.abc import Sequence

from aurex.data.base import Loader
from aurex.data.cache import CacheStore
from aurex.data.chain import SourceChain
from aurex.data.freshness import SeriesFreshness
from aurex.data.sources import FredLoader, YahooLoader


def _spec() -> dict[str, Sequence[Loader]]:
    return {
        "vix": (
            YahooLoader("vix", "^VIX"),
            FredLoader("vix", "VIXCLS", "close"),
        ),
        "real_yield_10y": (FredLoader("real_yield_10y", "DFII10", "real_yield"),),
        "dxy": (FredLoader("dxy", "DTWEXBGS", "dxy"),),
        "wti": (FredLoader("wti", "DCOILWTICO", "wti"),),
    }


#: How stale each macro series may be. None of these blocks publication — they are
#: factor inputs, not prices — but they are declared and measured so a series that
#: quietly stops updating is visible in the artifact rather than only in a loading.
#:
#: The numbers come from observed FRED publication lag rather than from a stated SLA,
#: which is why each carries the calendar it was read off. FRED restates: a series can
#: reach yesterday one day and the previous week the next, so these are deliberately
#: looser than the market calendar alone would suggest.
_FRESHNESS: dict[str, SeriesFreshness] = {
    "vix": SeriesFreshness(
        max_lag_days=5,
        calendar="US equity market days",
        rationale=(
            "Yahoo carries the previous session by the following morning; the FRED "
            "fallback lags a day further. Five days covers a long weekend plus the "
            "fallback's own lag without hiding a genuine multi-day outage."
        ),
    ),
    "real_yield_10y": SeriesFreshness(
        max_lag_days=5,
        calendar="US business days",
        rationale=(
            "DFII10 is published each business day with about a one-day lag. Five days "
            "absorbs a Monday holiday landing on top of that lag."
        ),
    ),
    "dxy": SeriesFreshness(
        max_lag_days=10,
        calendar="US business days, published in arrears",
        rationale=(
            "DTWEXBGS runs materially further behind than the other FRED series — six "
            "days behind on the 2026-07-30 run, which was an ordinary day. Ten days is "
            "set from that observed lag rather than from the market calendar; a tighter "
            "number would fire on healthy runs and train the reader to ignore it."
        ),
    ),
    "wti": SeriesFreshness(
        max_lag_days=7,
        calendar="US business days, published in arrears",
        rationale=(
            "DCOILWTICO was three days behind on the 2026-07-30 run. Seven days covers "
            "that lag plus a holiday week."
        ),
    ),
}

#: Series ids this module can serve.
MACRO_SERIES_IDS: tuple[str, ...] = tuple(_spec())


def macro_freshness() -> dict[str, SeriesFreshness]:
    """Declared staleness tolerance per macro series."""
    return dict(_FRESHNESS)


def macro_chains(cache: CacheStore | None = None) -> dict[str, SourceChain]:
    """Resolution chains for every shared macro series."""
    store = cache or CacheStore()
    return {
        sid: SourceChain(sid, loaders, store, freshness=_FRESHNESS.get(sid))
        for sid, loaders in _spec().items()
    }
