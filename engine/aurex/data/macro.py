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


#: Series ids this module can serve.
MACRO_SERIES_IDS: tuple[str, ...] = tuple(_spec())


def macro_chains(cache: CacheStore | None = None) -> dict[str, SourceChain]:
    """Resolution chains for every shared macro series."""
    store = cache or CacheStore()
    return {sid: SourceChain(sid, loaders, store) for sid, loaders in _spec().items()}
